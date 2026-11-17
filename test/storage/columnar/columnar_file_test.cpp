#include <gtest/gtest.h>
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "catalog/schema.hpp"
#include "common/exception.hpp"
#include "storage/columnar/catalog_codec.hpp"
#include "storage/columnar/columnar_file.hpp"
#include "storage/columnar/superblock.hpp"
#include "storage/disk_manager.hpp"
#include "storage/table_heap.hpp"  // also checks the row store and columnar headers build together
#include "types/value.hpp"
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
    const CatalogMeta meta = OneTable(kFirstOid, 400);
    {
        DiskManager disk(path_);
        ColumnarFile file(&disk);
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
            file.Commit(OneTable(kFirstOid + i, 1));
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

TEST_F(ColumnarFileTest, OtherPageSizeSlotSkipped) {
    // Slot 1 checks out and is newer, but was written for 4 KB pages.
    MakeNewFile();
    Superblock sb;
    sb.page_size = 4096;
    sb.generation = 7;
    WriteSlot(1, sb);
    DiskManager disk(path_);
    ColumnarFile file(&disk);
    EXPECT_EQ(file.Current().generation, 1u);
    EXPECT_EQ(file.Current().page_size, PAGE_SIZE);
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
        file.Commit(OneTable(kFirstOid, 3));  // generation 2, in slot 1
    }
    Flip(PAGE_SIZE + 100);
    {
        DiskManager disk(path_);
        ColumnarFile file(&disk);
        EXPECT_EQ(file.Current().generation, 1u);
        EXPECT_TRUE(file.Catalog().tables.empty());
        EXPECT_EQ(disk.GetNumPages(), 2u);  // the lost commit's pages are cut off

        // The next commit replaces the damaged slot.
        file.Commit(OneTable(kFirstOid, 2));
    }
    EXPECT_EQ(ReadSlot(1).state, SlotState::VALID);
    DiskManager disk(path_);
    ColumnarFile file(&disk);
    EXPECT_EQ(file.Current().generation, 2u);
    ASSERT_EQ(file.Catalog().tables.size(), 1u);
    EXPECT_EQ(file.Catalog().tables[0].columns.size(), 2u);
}

TEST_F(ColumnarFileTest, BothSlotsDamaged) {
    MakeNewFile();
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
        file.Commit(OneTable(kFirstOid, 1));
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
        file.Commit(OneTable(kFirstOid, 1));
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
    // Slot 0 says 2 pages were committed, but only 1 is left.
    MakeNewFile();
    std::filesystem::resize_file(path_, PAGE_SIZE);
    const std::vector<char> before = FileBytes();
    EXPECT_EQ(OpenError(), path_ + ": the file is shorter than its last commit (1 of 2 pages)");
    EXPECT_EQ(FileBytes(), before);
}

TEST_F(ColumnarFileTest, DamagedCatalogPageRefused) {
    page_id_t catalog_page = 0;
    {
        DiskManager disk(path_);
        ColumnarFile file(&disk);
        file.Commit(OneTable(kFirstOid, 1));
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
    file.Commit(OneTable(kFirstOid, 400));
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
        EXPECT_THROW(file.Commit(OneTable(kFirstOid, 1)), IOException);
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
        file.Commit(OneTable(kFirstOid, 400));
    }
    DiskManager disk(path_);
    ColumnarFile file(&disk);
    EXPECT_EQ(disk.GetReadStats().reads, 0u);
    EXPECT_EQ(disk.GetReadStats().time.count(), 0);
}

}  // namespace
}  // namespace shilmandb
