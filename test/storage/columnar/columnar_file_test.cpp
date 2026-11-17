#include <gtest/gtest.h>
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "catalog/schema.hpp"
#include "common/exception.hpp"
#include "storage/columnar/catalog_codec.hpp"
#include "storage/columnar/columnar_file.hpp"
#include "storage/columnar/encoding.hpp"
#include "storage/columnar/superblock.hpp"
#include "storage/disk_manager.hpp"
#include "storage/table_heap.hpp"  // also checks the row store and columnar headers build together
#include "types/value.hpp"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace shilmandb {
namespace {

// Remembers every write and sync, in order.
class RecordingObserver : public WriteObserver {
public:
    struct Event {
        bool sync;
        page_id_t page;
    };
    std::vector<Event> events;

    void OnWrite(page_id_t page_id, const char*) override { events.push_back({false, page_id}); }
    void OnSync() override { events.push_back({true, INVALID_PAGE_ID}); }
};

// Throws from the second sync, after the superblock write has landed.
class FailSecondSync : public WriteObserver {
public:
    void OnWrite(page_id_t, const char*) override {}
    void OnSync() override {
        if (++syncs_ == 2) throw IOException("sync failed on purpose");
    }

private:
    int syncs_{0};
};

class ColumnarFileTest : public ::testing::Test {
protected:
    std::string path_;

    void SetUp() override {
        path_ = (std::filesystem::temp_directory_path() /
                 ("shilmandb_colfile_" + std::to_string(::getpid()) + ".db"))
                    .string();
        std::filesystem::remove(path_);
    }

    void TearDown() override { std::filesystem::remove(path_); }

    // One table with enough long column names to span several catalog pages.
    static CatalogMeta OneTable(uint32_t oid, int columns) {
        ColumnarTableMeta table;
        table.oid = oid;
        table.name = "orders_" + std::to_string(oid);
        for (int i = 0; i < columns; ++i) {
            ColumnMeta col;
            col.name = "a_rather_long_column_name_number_" + std::to_string(i);
            col.number = static_cast<uint32_t>(i + 1);
            col.type = ColumnType::Integer();
            col.nullable = (i % 2) == 0;
            table.columns.push_back(std::move(col));
        }
        CatalogMeta meta;
        meta.tables.push_back(std::move(table));
        return meta;
    }

    std::vector<char> FileBytes() const {
        std::ifstream in(path_, std::ios::binary);
        return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    }

    // Overwrites bytes of the file in place.
    void Poke(std::streamoff offset, const char* data, size_t size) const {
        std::fstream f(path_, std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(offset);
        f.write(data, static_cast<std::streamsize>(size));
    }

    // Flips one byte, so the page's checksum fails.
    void Flip(std::streamoff offset) const {
        std::vector<char> bytes = FileBytes();
        const char flipped = static_cast<char>(bytes.at(static_cast<size_t>(offset)) ^ 0x5A);
        Poke(offset, &flipped, 1);
    }

    // Writes a valid superblock of another format version into a slot.
    void WriteVersion(page_id_t slot, uint32_t version) const {
        Superblock sb;
        sb.format_version = version;
        sb.generation = 7;
        WriteSlot(slot, sb);
    }

    // Writes a superblock with a good checksum into a slot, fields as given.
    void WriteSlot(page_id_t slot, const Superblock& sb) const {
        std::vector<char> page(PAGE_SIZE);
        EncodeSuperblock(sb, slot, page.data());
        Poke(static_cast<std::streamoff>(slot) * PAGE_SIZE, page.data(), page.size());
    }

    // Opens the file and returns the exception text, or "" if it opened.
    std::string OpenError() const {
        DiskManager disk(path_);
        try {
            ColumnarFile file(&disk);
        } catch (const DatabaseException& e) {
            return e.what();
        }
        return "";
    }

    SlotRead ReadSlot(page_id_t slot) const {
        DiskManager disk(path_);
        std::vector<char> page(PAGE_SIZE);
        disk.ReadPage(slot, page.data());
        return DecodeSuperblock(page.data(), slot);
    }

    void MakeNewFile() const {
        DiskManager disk(path_);
        ColumnarFile file(&disk);
    }

    // Replaces the whole file with these bytes.
    void WriteRaw(const std::vector<char>& bytes) const {
        std::ofstream out(path_, std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    // Writes the catalog as one page at id, then a newer slot 1 naming it.
    void WriteCatalogAt(page_id_t id, const CatalogMeta& meta, uint32_t next_oid) const {
        const std::vector<char> blob = EncodeCatalog(meta);
        ASSERT_LE(blob.size(), PAGE_SIZE - kPageHeaderSize);
        std::vector<char> page(PAGE_SIZE, 0);
        ColumnarPageHeader header;
        header.kind = static_cast<uint8_t>(PageKind::CATALOG);
        header.value_count = static_cast<uint16_t>(blob.size());
        std::memcpy(page.data(), &header, sizeof(header));
        std::memcpy(page.data() + kPageHeaderSize, blob.data(), blob.size());
        SealPages(page.data(), 1, id);
        Poke(static_cast<std::streamoff>(id) * PAGE_SIZE, page.data(), page.size());

        Superblock sb;
        sb.generation = 7;
        sb.catalog_first_page = id;
        sb.catalog_page_count = 1;
        sb.catalog_bytes = blob.size();
        sb.committed_pages = id + 1;
        sb.next_oid = next_oid;
        WriteSlot(1, sb);
    }

    // One BIGINT column of 10 rows on one PLAIN page, plus an index on it.
    static CatalogMeta TableWithChunk(page_id_t first_page, uint32_t page_count, page_id_t index_root) {
        CatalogMeta meta = OneTable(kFirstOid, 1);
        ColumnarTableMeta& table = meta.tables[0];
        table.columns[0].type = ColumnType::BigInt();
        ChunkMeta chunk;
        chunk.first_page = first_page;
        chunk.page_count = page_count;
        chunk.value_count = 10;
        chunk.page_first_rows.assign(page_count, 0);
        for (uint32_t i = 0; i < page_count; ++i) chunk.page_first_rows[i] = i;
        RowGroupMeta group;
        group.row_count = 10;
        group.chunks.push_back(std::move(chunk));
        table.row_groups.push_back(std::move(group));
        table.row_count = 10;

        IndexMeta index;
        index.oid = kFirstOid + 1;
        index.name = "idx_key";
        index.table_oid = kFirstOid;
        index.column_number = 1;
        index.key_type = TypeId::BIGINT;
        index.root_page = index_root;
        meta.indexes.push_back(std::move(index));
        return meta;
    }
};

TEST_F(ColumnarFileTest, NewFileHasHeaderAndEmptyCatalog) {
    DiskManager disk(path_);
    ColumnarFile file(&disk);
    EXPECT_EQ(disk.GetNumPages(), 2u);
    EXPECT_EQ(disk.GetFileBytes(), 2u * PAGE_SIZE);
    EXPECT_TRUE(file.Catalog().tables.empty());
    EXPECT_TRUE(file.Catalog().indexes.empty());
    EXPECT_EQ(file.Current().generation, 1u);
    EXPECT_EQ(file.Current().catalog_page_count, 0u);
    EXPECT_EQ(file.Current().committed_pages, 2u);
    EXPECT_EQ(file.Disk(), &disk);
}

TEST_F(ColumnarFileTest, CommitThenReopen) {
    CatalogMeta meta;
    {
        DiskManager disk(path_);
        ColumnarFile file(&disk);
        meta = OneTable(file.AllocateOid(), 400);
        file.Commit(meta);
        EXPECT_GE(file.Current().catalog_page_count, 2u);
    }
    DiskManager disk(path_);
    ColumnarFile file(&disk);
    EXPECT_EQ(file.Current().generation, 2u);
    EXPECT_GE(file.Current().catalog_page_count, 2u);
    EXPECT_EQ(EncodeCatalog(file.Catalog()), EncodeCatalog(meta));
}

TEST_F(ColumnarFileTest, CommitsAlternateSlots) {
    MakeNewFile();
    EXPECT_EQ(ReadSlot(0).block.generation, 1u);
    EXPECT_EQ(ReadSlot(1).block.generation, 0u);
    EXPECT_EQ(ReadSlot(1).state, SlotState::VALID);
    {
        DiskManager disk(path_);
        ColumnarFile file(&disk);
        const page_id_t expected_slot[] = {1, 0, 1};
        for (int i = 0; i < 3; ++i) {
            file.Commit(OneTable(file.AllocateOid(), 1));
            const SlotRead read = ReadSlot(expected_slot[i]);
            ASSERT_EQ(read.state, SlotState::VALID);
            EXPECT_EQ(read.block.generation, static_cast<uint64_t>(i + 2));
        }
    }
    EXPECT_EQ(ReadSlot(0).block.generation, 3u);
    EXPECT_EQ(ReadSlot(1).block.generation, 4u);
}

TEST_F(ColumnarFileTest, LegacyFileRefused) {
    {
        DiskManager disk(path_);
        BufferPoolManager bpm(8, &disk, std::make_unique<LRUEvictionPolicy>(8));
        TableHeap heap(&bpm);
        const Schema schema({Column("id", TypeId::INTEGER), Column("name", TypeId::VARCHAR)});
        const RID rid = heap.InsertTuple(Tuple({Value(int32_t{1}), Value(std::string("one"))}, schema));
        ASSERT_NE(rid.page_id, INVALID_PAGE_ID);
        bpm.FlushAllPages();
    }
    const std::vector<char> before = FileBytes();
    ASSERT_EQ(before.size(), PAGE_SIZE);
    EXPECT_EQ(OpenError(), path_ + ": not a ShilmanDB file, or an older format; reload it");
    EXPECT_EQ(FileBytes(), before);
}

TEST_F(ColumnarFileTest, ShortTextFileRefused) {
    {
        std::ofstream out(path_, std::ios::binary);
        out << std::string(100, 'q');
    }
    const std::vector<char> before = FileBytes();
    EXPECT_EQ(OpenError(), path_ + ": not a ShilmanDB file, or an older format; reload it");
    EXPECT_EQ(FileBytes(), before);
}

TEST_F(ColumnarFileTest, VersionMessages) {
    // A newer slot refuses the open even though slot 0 is a valid format-1 slot.
    MakeNewFile();
    WriteVersion(1, 2);
    std::vector<char> before = FileBytes();
    EXPECT_EQ(OpenError(), path_ + ": written by a newer ShilmanDB (format 2; this build reads format 1)");
    EXPECT_EQ(FileBytes(), before);

    std::filesystem::remove(path_);
    MakeNewFile();
    WriteVersion(0, 0);
    before = FileBytes();
    EXPECT_EQ(OpenError(), path_ + ": written in format 0, this build reads format 1; reload it");
    EXPECT_EQ(FileBytes(), before);
}

TEST_F(ColumnarFileTest, BothSlotsOtherVersions) {
    MakeNewFile();
    WriteVersion(0, 3);
    WriteVersion(1, 5);
    const std::vector<char> before = FileBytes();
    EXPECT_EQ(OpenError(), path_ + ": written by a newer ShilmanDB (format 5; this build reads format 1)");
    EXPECT_EQ(FileBytes(), before);
}

TEST_F(ColumnarFileTest, OtherPageSizeRefused) {
    // Slot 1 checks out and is newer, but was written for 4 KB pages.
    MakeNewFile();
    Superblock sb;
    sb.page_size = 4096;
    sb.generation = 7;
    WriteSlot(1, sb);
    const std::vector<char> before = FileBytes();
    EXPECT_EQ(OpenError(), path_ + ": written with 4096-byte pages; this build uses 8192-byte pages");
    EXPECT_EQ(FileBytes(), before);
}

TEST_F(ColumnarFileTest, OtherVersionWithBadChecksumRefused) {
    // A newer format may change the checksum; the version alone must refuse.
    MakeNewFile();
    WriteVersion(1, 2);
    Flip(PAGE_SIZE + 200);
    ASSERT_EQ(ReadSlot(0).state, SlotState::VALID);
    const std::vector<char> before = FileBytes();
    EXPECT_EQ(OpenError(), path_ + ": written by a newer ShilmanDB (format 2; this build reads format 1)");
    EXPECT_EQ(FileBytes(), before);
}

TEST_F(ColumnarFileTest, TooFewCommittedPagesRefused) {
    // A newer slot that would cut the file below its own header.
    MakeNewFile();
    Superblock sb;
    sb.generation = 7;
    sb.committed_pages = 1;
    WriteSlot(1, sb);
    const std::vector<char> before = FileBytes();
    EXPECT_EQ(OpenError(), path_ + ": the file header is damaged");
    EXPECT_EQ(FileBytes(), before);
}

TEST_F(ColumnarFileTest, DamagedSlotFallsBack) {
    {
        DiskManager disk(path_);
        ColumnarFile file(&disk);
        file.Commit(OneTable(file.AllocateOid(), 3));  // generation 2, in slot 1
    }
    Flip(PAGE_SIZE + 100);
    {
        DiskManager disk(path_);
        ColumnarFile file(&disk);
        EXPECT_EQ(file.Current().generation, 1u);
        EXPECT_TRUE(file.Catalog().tables.empty());
        EXPECT_EQ(disk.GetNumPages(), 2u);  // the lost commit's pages are cut off

        // The next commit replaces the damaged slot.
        file.Commit(OneTable(file.AllocateOid(), 2));
    }
    EXPECT_EQ(ReadSlot(1).state, SlotState::VALID);
    DiskManager disk(path_);
    ColumnarFile file(&disk);
    EXPECT_EQ(file.Current().generation, 2u);
    ASSERT_EQ(file.Catalog().tables.size(), 1u);
    EXPECT_EQ(file.Catalog().tables[0].columns.size(), 2u);
}

TEST_F(ColumnarFileTest, BothSlotsDamaged) {
    // A committed file, so it isn't taken for a crash during create.
    {
        DiskManager disk(path_);
        ColumnarFile file(&disk);
        file.Commit(OneTable(file.AllocateOid(), 1));
    }
    Flip(100);
    Flip(PAGE_SIZE + 100);
    const std::vector<char> before = FileBytes();
    EXPECT_EQ(OpenError(), path_ + ": the file header is damaged");
    EXPECT_EQ(FileBytes(), before);
}

TEST_F(ColumnarFileTest, TruncatesPastLastCommit) {
    page_id_t committed = 0;
    {
        DiskManager disk(path_);
        ColumnarFile file(&disk);
        file.Commit(OneTable(file.AllocateOid(), 1));
        committed = file.Current().committed_pages;
        const page_id_t first = disk.AllocatePages(10);
        const std::vector<char> page(PAGE_SIZE, 'z');
        for (page_id_t p = first; p < first + 10; ++p) disk.WritePage(p, page.data());
    }
    EXPECT_EQ(FileBytes().size(), (committed + 10u) * PAGE_SIZE);
    DiskManager disk(path_);
    ColumnarFile file(&disk);
    EXPECT_EQ(disk.GetNumPages(), committed);
    EXPECT_EQ(disk.GetFileBytes(), static_cast<uint64_t>(committed) * PAGE_SIZE);
    EXPECT_EQ(file.Catalog().tables.size(), 1u);
}

TEST_F(ColumnarFileTest, DataPagesBeforeCommitSurvive) {
    std::vector<std::vector<char>> pages;
    page_id_t first = 0;
    page_id_t committed = 0;
    {
        DiskManager disk(path_);
        ColumnarFile file(&disk);
        first = disk.AllocatePages(3);
        for (page_id_t i = 0; i < 3; ++i) {
            pages.emplace_back(PAGE_SIZE, static_cast<char>('a' + i));
            pages.back()[i] = 'X';
            disk.WritePage(first + i, pages.back().data());
        }
        file.Commit(OneTable(file.AllocateOid(), 1));
        committed = file.Current().committed_pages;
        EXPECT_GT(committed, first + 3);  // the catalog pages come after the data pages
    }
    DiskManager disk(path_);
    ColumnarFile file(&disk);
    EXPECT_EQ(disk.GetNumPages(), committed);
    std::vector<char> page(PAGE_SIZE);
    for (page_id_t i = 0; i < 3; ++i) {
        disk.ReadPage(first + i, page.data());
        EXPECT_EQ(page, pages[i]) << "data page " << first + i;
    }
}

TEST_F(ColumnarFileTest, ShortFileRefused) {
    // The last commit covers 3 pages, but only 2 are left.
    {
        DiskManager disk(path_);
        ColumnarFile file(&disk);
        file.Commit(OneTable(file.AllocateOid(), 1));
        ASSERT_EQ(file.Current().committed_pages, 3u);
    }
    std::filesystem::resize_file(path_, 2 * PAGE_SIZE);
    const std::vector<char> before = FileBytes();
    EXPECT_EQ(OpenError(), path_ + ": the file is shorter than its last commit (2 of 3 pages)");
    EXPECT_EQ(FileBytes(), before);
}

TEST_F(ColumnarFileTest, CreateCrashStatesReopenEmpty) {
    std::vector<char> slot0(PAGE_SIZE);
    std::vector<char> slot1(PAGE_SIZE);
    Superblock first;
    first.generation = 1;
    EncodeSuperblock(first, 0, slot0.data());
    EncodeSuperblock(Superblock{}, 1, slot1.data());
    const std::vector<char> zeros(PAGE_SIZE, 0);
    const auto join = [](std::vector<char> a, const std::vector<char>& b) {
        a.insert(a.end(), b.begin(), b.end());
        return a;
    };
    std::vector<char> torn = zeros;
    std::copy(slot0.begin(), slot0.begin() + PAGE_SIZE / 2, torn.begin());

    struct State {
        const char* name;
        std::vector<char> bytes;
    };
    const State refused_states[] = {
        {"slot 0 only", slot0},
        {"two zero pages", join(zeros, zeros)},
        {"one zero page", zeros},
        {"slot 0 torn, no slot 1", torn},
    };
    for (const State& state : refused_states) {
        SCOPED_TRACE(state.name);
        WriteRaw(state.bytes);
        {
            DiskManager disk(path_);
            ColumnarFile file(&disk);
            EXPECT_EQ(file.Current().generation, 1u);
            EXPECT_TRUE(file.Catalog().tables.empty());
            EXPECT_TRUE(file.Catalog().indexes.empty());
            EXPECT_EQ(disk.GetFileBytes(), 2u * PAGE_SIZE);
            EXPECT_EQ(disk.GetNumPages(), 2u);
            file.Commit(OneTable(file.AllocateOid(), 1));
        }
        DiskManager disk(path_);
        ColumnarFile file(&disk);
        EXPECT_EQ(file.Current().generation, 2u);
        EXPECT_EQ(file.Catalog().tables.size(), 1u);
    }

    // Slot 0 lost, slot 1 kept: slot 1 checks out and opens as it is.
    const std::vector<char> slot1_only = join(zeros, slot1);
    WriteRaw(slot1_only);
    {
        DiskManager disk(path_);
        ColumnarFile file(&disk);
        EXPECT_EQ(file.Current().generation, 0u);
        EXPECT_TRUE(file.Catalog().tables.empty());
        EXPECT_EQ(FileBytes(), slot1_only);
        file.Commit(OneTable(file.AllocateOid(), 1));
    }
    {
        DiskManager disk(path_);
        ColumnarFile file(&disk);
        EXPECT_EQ(file.Current().generation, 1u);
        EXPECT_EQ(file.Catalog().tables.size(), 1u);
        ASSERT_EQ(file.Current().committed_pages, 3u);
    }

    // That commit is generation 1 but real, so cutting it refuses rather than resets.
    std::filesystem::resize_file(path_, 2 * PAGE_SIZE);
    const std::vector<char> cut = FileBytes();
    EXPECT_EQ(OpenError(), path_ + ": the file is shorter than its last commit (2 of 3 pages)");
    EXPECT_EQ(FileBytes(), cut);

    // A finished create opens without being rewritten.
    std::filesystem::remove(path_);
    MakeNewFile();
    {
        DiskManager disk(path_);
        RecordingObserver observer;
        disk.SetWriteObserver(&observer);
        ColumnarFile file(&disk);
        disk.SetWriteObserver(nullptr);
        EXPECT_TRUE(observer.events.empty());
        EXPECT_EQ(file.Current().generation, 1u);
    }

    // Two pages of junk with no magic aren't ours, so they're left alone.
    WriteRaw(std::vector<char>(2 * PAGE_SIZE, 'j'));
    const std::vector<char> junk = FileBytes();
    EXPECT_EQ(OpenError(), path_ + ": not a ShilmanDB file, or an older format; reload it");
    EXPECT_EQ(FileBytes(), junk);
}

TEST_F(ColumnarFileTest, CreateNeedsPageZeroFirst) {
    DiskManager disk(path_);
    (void)disk.AllocatePage();
    EXPECT_THROW(ColumnarFile file(&disk), DatabaseException);
    EXPECT_EQ(disk.GetFileBytes(), 0u);
}

TEST_F(ColumnarFileTest, ChunkOutsideCommitRefused) {
    // The catalog sits on page 2, so the commit covers pages 0 to 2.
    struct Case {
        page_id_t first_page;
        uint32_t page_count;
        std::string error;
    };
    const Case cases[] = {
        {2, 2, "2 pages from page 2 lie outside pages 2 to 2"},
        {1, 1, "1 pages from page 1 lie outside pages 2 to 2"},
        {0xFFFFFFFEu, 3, "3 pages from page 4294967294 lie outside pages 2 to 2"},  // ends at page 1 in 32 bits
    };
    for (const Case& c : cases) {
        SCOPED_TRACE(c.error);
        std::filesystem::remove(path_);
        MakeNewFile();
        WriteCatalogAt(2, TableWithChunk(c.first_page, c.page_count, INVALID_PAGE_ID), kFirstOid + 2);
        const std::vector<char> before = FileBytes();
        EXPECT_EQ(OpenError(), path_ + ": damaged catalog: table orders_16384 column a_rather_long_column_name_number_0 row group 0: " + c.error);
        EXPECT_EQ(FileBytes(), before);
    }

    // The same chunk on page 2 alone fits.
    std::filesystem::remove(path_);
    MakeNewFile();
    WriteCatalogAt(2, TableWithChunk(2, 1, INVALID_PAGE_ID), kFirstOid + 2);
    EXPECT_EQ(OpenError(), "");
}

TEST_F(ColumnarFileTest, IndexRootOutsideCommitRefused) {
    for (const page_id_t root : {page_id_t{3}, page_id_t{1}, page_id_t{0x7FFFFFFF}}) {
        SCOPED_TRACE(root);
        std::filesystem::remove(path_);
        MakeNewFile();
        WriteCatalogAt(2, TableWithChunk(2, 1, root), kFirstOid + 2);
        const std::vector<char> before = FileBytes();
        EXPECT_EQ(OpenError(), path_ + ": damaged catalog: index idx_key root page " + std::to_string(root) + " lies outside pages 2 to 2");
        EXPECT_EQ(FileBytes(), before);
    }
}

TEST_F(ColumnarFileTest, EmptyIndexRootOpens) {
    // An empty B+tree has no root page yet.
    MakeNewFile();
    WriteCatalogAt(2, TableWithChunk(2, 1, INVALID_PAGE_ID), kFirstOid + 2);
    DiskManager disk(path_);
    ColumnarFile file(&disk);
    ASSERT_EQ(file.Catalog().indexes.size(), 1u);
    EXPECT_EQ(file.Catalog().indexes[0].root_page, INVALID_PAGE_ID);
    EXPECT_EQ(file.Current().generation, 7u);
}

TEST_F(ColumnarFileTest, LowNextOidRefused) {
    {
        DiskManager disk(path_);
        ColumnarFile file(&disk);
        file.Commit(OneTable(file.AllocateOid(), 1));  // generation 2, in slot 1
    }
    Superblock sb = ReadSlot(1).block;
    ASSERT_EQ(sb.next_oid, kFirstOid + 1);
    sb.next_oid = kFirstOid;
    WriteSlot(1, sb);
    const std::vector<char> before = FileBytes();
    EXPECT_EQ(OpenError(), path_ + ": damaged catalog: table orders_16384 has oid 16384, not below the next oid 16384");
    EXPECT_EQ(FileBytes(), before);

    // The same for an index.
    std::filesystem::remove(path_);
    MakeNewFile();
    WriteCatalogAt(2, TableWithChunk(2, 1, INVALID_PAGE_ID), kFirstOid + 1);
    EXPECT_EQ(OpenError(), path_ + ": damaged catalog: index idx_key has oid 16385, not below the next oid 16385");
}

TEST_F(ColumnarFileTest, CommitRefusesWhatOpenWouldRefuse) {
    {
        DiskManager disk(path_);
        ColumnarFile file(&disk);
        file.Commit(OneTable(file.AllocateOid(), 1));  // generation 2
    }
    const std::vector<char> before = FileBytes();
    {
        DiskManager disk(path_);
        ColumnarFile file(&disk);
        const page_id_t pages = disk.GetNumPages();
        RecordingObserver observer;
        disk.SetWriteObserver(&observer);
        const auto commit_error = [&](CatalogMeta meta) -> std::string {
            try {
                file.Commit(std::move(meta));
            } catch (const DatabaseException& e) {
                return e.what();
            }
            return "";
        };

        CatalogMeta twins = file.Catalog();
        twins.tables.push_back(twins.tables[0]);
        twins.tables[1].oid = file.AllocateOid();
        EXPECT_EQ(commit_error(twins), "damaged catalog: two tables named orders_16384");

        // 16386 was never handed out; 16385 went to the twin above.
        EXPECT_EQ(commit_error(OneTable(kFirstOid + 2, 1)), "damaged catalog: table orders_16386 has oid 16386, not below the next oid 16386");

        // Data pages must already be in the file, before the catalog pages.
        CatalogMeta outside = TableWithChunk(pages, 1, INVALID_PAGE_ID);
        outside.indexes.clear();
        EXPECT_EQ(commit_error(outside), "damaged catalog: table orders_16384 column a_rather_long_column_name_number_0 row group 0: 1 pages from page " + std::to_string(pages) + " lie outside pages 2 to " + std::to_string(pages - 1));

        disk.SetWriteObserver(nullptr);
        EXPECT_TRUE(observer.events.empty());
        EXPECT_EQ(disk.GetNumPages(), pages);
        EXPECT_EQ(file.Current().generation, 2u);
        EXPECT_EQ(file.Catalog().tables.size(), 1u);
    }
    EXPECT_EQ(FileBytes(), before);
    DiskManager disk(path_);
    ColumnarFile file(&disk);
    EXPECT_EQ(file.Current().generation, 2u);
    EXPECT_EQ(file.Catalog().tables.size(), 1u);
}

TEST_F(ColumnarFileTest, DamagedCatalogPageRefused) {
    page_id_t catalog_page = 0;
    {
        DiskManager disk(path_);
        ColumnarFile file(&disk);
        file.Commit(OneTable(file.AllocateOid(), 1));
        catalog_page = file.Current().catalog_first_page;
    }
    Flip(static_cast<std::streamoff>(catalog_page) * PAGE_SIZE + 40);
    const std::vector<char> before = FileBytes();
    EXPECT_EQ(OpenError(),
              path_ + ": damaged catalog: page " + std::to_string(catalog_page) + " fails its checksum");
    EXPECT_EQ(FileBytes(), before);
}

TEST_F(ColumnarFileTest, CommitOrder) {
    DiskManager disk(path_);
    ColumnarFile file(&disk);
    RecordingObserver observer;
    disk.SetWriteObserver(&observer);
    file.Commit(OneTable(file.AllocateOid(), 400));
    disk.SetWriteObserver(nullptr);

    const Superblock& sb = file.Current();
    ASSERT_GE(sb.catalog_page_count, 2u);
    ASSERT_EQ(observer.events.size(), sb.catalog_page_count + 3u);
    for (uint32_t i = 0; i < sb.catalog_page_count; ++i) {
        EXPECT_FALSE(observer.events[i].sync);
        EXPECT_EQ(observer.events[i].page, sb.catalog_first_page + i);
    }
    const auto* tail = &observer.events[sb.catalog_page_count];
    EXPECT_TRUE(tail[0].sync);
    EXPECT_FALSE(tail[1].sync);
    EXPECT_EQ(tail[1].page, 1u);  // the slot that wasn't current
    EXPECT_TRUE(tail[2].sync);
}

TEST_F(ColumnarFileTest, FailedCommitKeepsOldCatalog) {
    {
        DiskManager disk(path_);
        ColumnarFile file(&disk);
        FailSecondSync observer;
        disk.SetWriteObserver(&observer);
        EXPECT_THROW(file.Commit(OneTable(file.AllocateOid(), 1)), IOException);
        disk.SetWriteObserver(nullptr);
        EXPECT_TRUE(file.Catalog().tables.empty());
        EXPECT_EQ(file.Current().generation, 1u);
        EXPECT_EQ(file.Current().catalog_page_count, 0u);
    }
    // The superblock reached the file before the sync failed.
    DiskManager disk(path_);
    ColumnarFile file(&disk);
    EXPECT_EQ(file.Current().generation, 2u);
    EXPECT_EQ(file.Catalog().tables.size(), 1u);
}

TEST_F(ColumnarFileTest, OidsPersist) {
    {
        DiskManager disk(path_);
        ColumnarFile file(&disk);
        EXPECT_EQ(file.AllocateOid(), 16384u);
        EXPECT_EQ(file.AllocateOid(), 16385u);
        file.Commit(CatalogMeta{});
    }
    DiskManager disk(path_);
    ColumnarFile file(&disk);
    EXPECT_EQ(file.AllocateOid(), 16386u);
}

TEST_F(ColumnarFileTest, ReadStatsStartAtZero) {
    {
        DiskManager disk(path_);
        ColumnarFile file(&disk);
        EXPECT_EQ(disk.GetReadStats().reads, 0u);
        file.Commit(OneTable(file.AllocateOid(), 400));
    }
    DiskManager disk(path_);
    ColumnarFile file(&disk);
    EXPECT_EQ(disk.GetReadStats().reads, 0u);
    EXPECT_EQ(disk.GetReadStats().time.count(), 0);
}

}  // namespace
}  // namespace shilmandb
