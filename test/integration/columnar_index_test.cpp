#include <gtest/gtest.h>

#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "catalog/catalog.hpp"
#include "common/exception.hpp"
#include "engine/database.hpp"
#include "index/btree_internal_page.hpp"
#include "index/btree_leaf_page.hpp"
#include "storage/columnar/column_vector.hpp"
#include "storage/columnar/columnar_file.hpp"
#include "storage/columnar/columnar_table.hpp"
#include "storage/disk_manager.hpp"
#include "storage/slotted_page.hpp"

namespace shilmandb {
namespace {

constexpr size_t kBigRows = 150000;
constexpr int64_t kKeySpread = 60000;
constexpr int64_t kDamageRows = 5000;  // enough sorted keys for a root over about 17 leaves
constexpr int kChildSeconds = 20;

// Runs fn and returns what it threw, or "" if it didn't throw.
template <typename Fn>
std::string ErrorOf(Fn&& fn) {
    try {
        fn();
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

// A big integer key and a small integer value.
Schema KeyValue() { return Schema({Column("k", ColumnType::BigInt()), Column("v", ColumnType::Integer())}); }

// Row n's key: each of 0 to 59,999 comes two or three times, unsorted.
int64_t KeyOf(size_t n) { return static_cast<int64_t>(n * 7) % kKeySpread; }

// Where row n (1-based) lands, as the RID an index stores.
RID RidOf(size_t n) {
    return ToRid(RowPosition{static_cast<uint32_t>((n - 1) / kRowGroupSize), static_cast<uint16_t>((n - 1) % kRowGroupSize)});
}

// Rows 1 to rows of KeyOf(n) and n.
std::string ScrambledLines(size_t rows) {
    std::string text;
    text.reserve(rows * 14);
    for (size_t n = 1; n <= rows; ++n) {
        text += std::to_string(KeyOf(n)) + "|" + std::to_string(n) + "\n";
    }
    return text;
}

// Rows with keys first to last, in order, and value 0.
std::string KeyRange(int64_t first, int64_t last) {
    std::string text;
    for (int64_t k = first; k <= last; ++k) text += std::to_string(k) + "|0\n";
    return text;
}

// Columnar storage on two threads.
LoadOptions Columnar() {
    LoadOptions options;
    options.storage = StorageKind::COLUMNAR;
    options.threads = 2;
    return options;
}

// RIDs as sortable pairs, so two lookups compare regardless of order.
std::vector<std::pair<uint32_t, uint16_t>> Sorted(const std::vector<RID>& rids) {
    std::vector<std::pair<uint32_t, uint16_t>> out;
    out.reserve(rids.size());
    for (const RID& r : rids) out.emplace_back(r.page_id, r.slot_id);
    std::sort(out.begin(), out.end());
    return out;
}

// The key column's value at one row position.
int64_t KeyAt(const ColumnarTable& table, const RID& rid) {
    ColumnVector out(ColumnType::BigInt());
    table.ReadValue(FromRid(rid), 0, out);
    return out.GetInt(0);
}

// Runs fn in a forked child, so a hang or abort fails the test.
// Returns what fn threw, or how the child ended without throwing.
std::string ErrorInChild(const std::function<void()>& fn) {
    int fds[2];
    if (::pipe(fds) != 0) return "pipe failed";
    const pid_t pid = ::fork();
    if (pid < 0) return "fork failed";
    if (pid == 0) {
        ::close(fds[0]);
        std::string text = "no throw";
        try {
            fn();
        } catch (const std::exception& e) {
            text = e.what();
        }
        (void)!::write(fds[1], text.data(), text.size());
        ::_exit(0);
    }
    ::close(fds[1]);

    std::string out;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(kChildSeconds);
    char buf[512];
    while (true) {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        pollfd p{fds[0], POLLIN, 0};
        if (left <= 0 || ::poll(&p, 1, static_cast<int>(left)) == 0) {
            ::kill(pid, SIGKILL);
            ::waitpid(pid, nullptr, 0);
            ::close(fds[0]);
            return "timed out after " + std::to_string(kChildSeconds) + " s";
        }
        const ssize_t n = ::read(fds[0], buf, sizeof(buf));
        if (n <= 0) break;
        out.append(buf, static_cast<size_t>(n));
    }
    ::close(fds[0]);
    int status = 0;
    ::waitpid(pid, &status, 0);
    if (WIFSIGNALED(status)) return "killed by signal " + std::to_string(WTERMSIG(status));
    return out;
}

// Pinned frames in the pool, as a note to add to an error, or "".
std::string PinNote(const BufferPoolManager& bpm) {
    size_t pinned = 0;
    for (size_t f = 0; f < bpm.GetPoolSize(); ++f) pinned += bpm.GetPage(static_cast<frame_id_t>(f)).GetPinCount() > 0;
    return pinned == 0 ? "" : " (and " + std::to_string(pinned) + " pages left pinned)";
}

class ColumnarIndexTest : public ::testing::Test {
protected:
    std::string db_file_;
    std::vector<std::string> tbl_files_;

    void SetUp() override {
        db_file_ = TempPath(".db");
        std::filesystem::remove(db_file_);
    }

    void TearDown() override {
        std::filesystem::remove(db_file_);
        for (const auto& f : tbl_files_) std::filesystem::remove(f);
    }

    // A temp path unique to this process.
    static std::string TempPath(const std::string& suffix) {
        return (std::filesystem::temp_directory_path() /
                ("shilmandb_columnar_index_test_" + std::to_string(::getpid()) + suffix)).string();
    }

    // Writes text to a per-test .tbl file and returns its path.
    std::string WriteTbl(const std::string& what, const std::string& text) {
        const std::string path = TempPath("_" + what + ".tbl");
        std::ofstream(path, std::ios::binary) << text;
        tbl_files_.push_back(path);
        return path;
    }

    // Reads count pages straight from the db file, starting at page first.
    std::vector<char> ReadPages(page_id_t first, uint64_t count) const {
        std::vector<char> bytes(count * PAGE_SIZE);
        std::ifstream in(db_file_, std::ios::binary);
        in.seekg(static_cast<std::streamoff>(first) * PAGE_SIZE);
        in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        EXPECT_EQ(static_cast<uint64_t>(in.gcount()), bytes.size());
        return bytes;
    }

    // Overwrites one page of the db file.
    void WritePage(page_id_t pid, const std::vector<char>& bytes) const {
        std::fstream out(db_file_, std::ios::binary | std::ios::in | std::ios::out);
        out.seekp(static_cast<std::streamoff>(pid) * PAGE_SIZE);
        out.write(bytes.data(), PAGE_SIZE);
        ASSERT_TRUE(out.good());
    }

    // Commits a table of sorted keys with a two-level index, and returns the root.
    page_id_t CommitTwoLevelIndex() {
        const std::string tbl = WriteTbl("damage", KeyRange(1, kDamageRows));
        {
            Database db(db_file_);
            db.LoadTable("t", KeyValue(), tbl, Columnar());
            EXPECT_NE(db.GetCatalog()->CreateIndex("idx_k", "t", "k"), nullptr);
        }
        const page_id_t root = SavedRoots().at(0).second;
        const std::vector<char> page = ReadPages(root, 1);
        EXPECT_EQ(SlottedPage::GetHeader(page.data())->flags & 1u, 0u) << "the root should be internal";
        const page_id_t first = BTreeInternalPage::GetChild(page.data(), 0, TypeId::BIGINT);
        EXPECT_NE(SlottedPage::GetHeader(ReadPages(first, 1).data())->flags & 1u, 0u) << "its children should be leaves";
        return root;
    }

    // Opens the file in a child and runs a point lookup, range scans and SQL.
    // Each must throw "damaged index: ..." with every page unpinned.
    void ExpectDamaged(int64_t point, int64_t low, const std::string& what) {
        const auto run = [&](const std::function<void(Database&, BTreeIndex&)>& op) {
            return ErrorInChild([&] {
                Database db(db_file_);
                try {
                    op(db, *db.GetCatalog()->GetIndex("idx_k")->index);
                } catch (const std::exception& e) {
                    throw DatabaseException(e.what() + PinNote(*db.GetBufferPoolManager()));
                }
            });
        };
        const std::vector<std::pair<std::string, std::function<void(Database&, BTreeIndex&)>>> ops = {
            {"point lookup", [&](Database&, BTreeIndex& index) { (void)index.PointLookup(Value(point)); }},
            {"range scan", [&](Database&, BTreeIndex& index) {
                 for (auto it = index.Begin(Value(low)); !it.IsEnd(); ++it) (void)*it;
             }},
            {"full scan", [&](Database&, BTreeIndex& index) {
                 for (auto it = index.Begin(); !it.IsEnd(); ++it) (void)*it;
             }},
            {"SQL point", [&](Database& db, BTreeIndex&) {
                 (void)db.ExecuteSQL("SELECT v FROM t WHERE k = " + std::to_string(point));
             }},
            {"SQL range", [&](Database& db, BTreeIndex&) {
                 (void)db.ExecuteSQL("SELECT v FROM t WHERE k >= " + std::to_string(low) + " AND k <= " +
                                     std::to_string(low + 10));
             }},
        };
        for (const auto& [name, op] : ops) {
            const std::string error = run(op);
            EXPECT_EQ(error.rfind("damaged index: ", 0), 0u) << what << ", " << name << ": " << error;
            EXPECT_EQ(error.find("pinned"), std::string::npos) << what << ", " << name << ": " << error;
        }
    }

    // The index roots the file has saved, by index name, read with a fresh open.
    std::vector<std::pair<std::string, page_id_t>> SavedRoots() const {
        DiskManager disk(db_file_);
        ColumnarFile file(&disk);
        std::vector<std::pair<std::string, page_id_t>> roots;
        for (const IndexMeta& index : file.Catalog().indexes) roots.emplace_back(index.name, index.root_page);
        return roots;
    }
};

TEST_F(ColumnarIndexTest, PointAndRangeLookups) {
    const std::string tbl = WriteTbl("big", ScrambledLines(kBigRows));
    Database db(db_file_);
    db.LoadTable("t", KeyValue(), tbl, Columnar());
    IndexInfo* index = db.GetCatalog()->CreateIndex("idx_k", "t", "k");
    ASSERT_NE(index, nullptr);
    EXPECT_EQ(index->col_idx, 0u);
    EXPECT_EQ(index->oid, 16385u);
    const ColumnarTable& table = *db.GetCatalog()->GetTable("t")->columnar;
    ASSERT_EQ(table.RowGroupCount(), 3u);

    for (int64_t k : {int64_t{0}, int64_t{7}, int64_t{12345}, int64_t{59999}}) {
        std::vector<RID> want;
        for (size_t n = 1; n <= kBigRows; ++n) {
            if (KeyOf(n) == k) want.push_back(RidOf(n));
        }
        const std::vector<RID> got = index->index->PointLookup(Value(k));
        EXPECT_EQ(Sorted(got), Sorted(want)) << "key " << k;
        for (const RID& rid : got) EXPECT_EQ(KeyAt(table, rid), k);
    }
    EXPECT_TRUE(index->index->PointLookup(Value(kKeySpread)).empty());

    const int64_t low = 1000;
    const int64_t high = 1500;
    std::vector<int64_t> want_keys;
    for (size_t n = 1; n <= kBigRows; ++n) {
        if (KeyOf(n) >= low && KeyOf(n) <= high) want_keys.push_back(KeyOf(n));
    }
    std::sort(want_keys.begin(), want_keys.end());

    std::vector<int64_t> got_keys;
    for (auto it = index->index->Begin(Value(low)); !it.IsEnd(); ++it) {
        const auto [key, rid] = *it;
        if (key.bigint_ > high) break;
        got_keys.push_back(key.bigint_);
        EXPECT_EQ(KeyAt(table, rid), key.bigint_);
    }
    EXPECT_EQ(got_keys, want_keys);
}

TEST_F(ColumnarIndexTest, IndexSurvivesReopen) {
    const std::string tbl = WriteTbl("big", ScrambledLines(kBigRows));
    const std::vector<int64_t> keys = {0, 1, 777, 30000, 59999};
    std::vector<std::vector<std::pair<uint32_t, uint16_t>>> before;
    page_id_t first_root = INVALID_PAGE_ID;
    page_id_t root = INVALID_PAGE_ID;
    {
        Database db(db_file_);
        db.LoadTable("t", KeyValue(), tbl, Columnar());
        // The tree's first page is the next one handed out
        first_root = db.GetDiskManager()->GetNumPages();
        IndexInfo* index = db.GetCatalog()->CreateIndex("idx_k", "t", "k");
        ASSERT_NE(index, nullptr);
        root = index->index->GetRootPageId();
        EXPECT_EQ(index->root_page_id, root);
        for (int64_t k : keys) before.push_back(Sorted(index->index->PointLookup(Value(k))));
    }

    // The root split, so the saved root is a later internal page
    EXPECT_NE(root, first_root);
    const std::vector<char> root_page = ReadPages(root, 1);
    EXPECT_EQ(SlottedPage::GetHeader(root_page.data())->flags & 1u, 0u);
    const auto saved = SavedRoots();
    ASSERT_EQ(saved.size(), 1u);
    EXPECT_EQ(saved[0].first, "idx_k");
    EXPECT_EQ(saved[0].second, root);

    Database db(db_file_);
    IndexInfo* index = db.GetCatalog()->GetIndex("idx_k");
    ASSERT_NE(index, nullptr);
    EXPECT_EQ(index->table_name, "t");
    EXPECT_EQ(index->column_name, "k");
    EXPECT_EQ(index->col_idx, 0u);
    EXPECT_EQ(index->oid, 16385u);
    EXPECT_EQ(index->root_page_id, root);
    EXPECT_EQ(index->index->GetRootPageId(), root);
    ASSERT_EQ(db.GetCatalog()->GetTableIndexes("t").size(), 1u);
    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_EQ(Sorted(index->index->PointLookup(Value(keys[i]))), before[i]) << "key " << keys[i];
        EXPECT_FALSE(before[i].empty());
    }
}

TEST_F(ColumnarIndexTest, IndexCreatedAfterLoadCommits) {
    const std::string tbl = WriteTbl("small", KeyRange(1, 500));
    {
        Database db(db_file_);
        db.LoadTable("t", KeyValue(), tbl, Columnar());
    }
    {
        Database db(db_file_);
        ASSERT_NE(db.GetCatalog()->CreateIndex("idx_k", "t", "k"), nullptr);
    }
    Database db(db_file_);
    IndexInfo* index = db.GetCatalog()->GetIndex("idx_k");
    ASSERT_NE(index, nullptr);
    const std::vector<RID> got = index->index->PointLookup(Value(int64_t{250}));
    ASSERT_EQ(got.size(), 1u);
    EXPECT_EQ(KeyAt(*db.GetCatalog()->GetTable("t")->columnar, got[0]), 250);
}

TEST_F(ColumnarIndexTest, ReplaceKeepsOidsAndRebuildsIndexes) {
    const std::string v1 = WriteTbl("v1", KeyRange(1, 1000));
    const std::string v2 = WriteTbl("v2", KeyRange(1001, 2000));

    // Checks the table holds v2's keys only, under the old OIDs.
    const auto expect_v2 = [](Database& db) {
        TableInfo* info = db.GetCatalog()->GetTable("t");
        ASSERT_NE(info, nullptr);
        EXPECT_EQ(info->oid, 16384u);
        EXPECT_EQ(info->stats.row_count, 1000u);
        IndexInfo* index = db.GetCatalog()->GetIndex("idx_k");
        ASSERT_NE(index, nullptr);
        EXPECT_EQ(index->oid, 16385u);
        EXPECT_EQ(index->root_page_id, index->index->GetRootPageId());
        for (int64_t k : {int64_t{1}, int64_t{500}, int64_t{1000}}) {
            EXPECT_TRUE(index->index->PointLookup(Value(k)).empty()) << "v1 key " << k;
        }
        for (int64_t k : {int64_t{1001}, int64_t{1500}, int64_t{2000}}) {
            const std::vector<RID> got = index->index->PointLookup(Value(k));
            ASSERT_EQ(got.size(), 1u) << "v2 key " << k;
            EXPECT_EQ(KeyAt(*info->columnar, got[0]), k);
        }
    };

    {
        Database db(db_file_);
        db.LoadTable("t", KeyValue(), v1, Columnar());
        IndexInfo* index = db.GetCatalog()->CreateIndex("idx_k", "t", "k");
        ASSERT_NE(index, nullptr);
        ASSERT_EQ(index->index->PointLookup(Value(int64_t{500})).size(), 1u);
        const page_id_t old_root = index->root_page_id;

        db.LoadTable("t", KeyValue(), v2, Columnar());
        EXPECT_EQ(db.GetCatalog()->GetIndex("idx_k"), index);
        EXPECT_NE(index->root_page_id, old_root);
        expect_v2(db);
    }
    Database db(db_file_);
    expect_v2(db);
}

TEST_F(ColumnarIndexTest, ReplaceWithDifferentSchemaRefused) {
    const std::string v1 = WriteTbl("v1", KeyRange(1, 1000));
    Database db(db_file_);
    db.LoadTable("t", KeyValue(), v1, Columnar());
    ASSERT_NE(db.GetCatalog()->CreateIndex("idx_k", "t", "k"), nullptr);
    const page_id_t pages = db.GetDiskManager()->GetNumPages();

    const Schema renamed({Column("k", ColumnType::BigInt()), Column("w", ColumnType::Integer())});
    const Schema retyped({Column("k", ColumnType::Integer()), Column("v", ColumnType::Integer())});
    const Schema longer({Column("k", ColumnType::BigInt()), Column("v", ColumnType::Integer()), Column("x", ColumnType::Integer())});
    for (const Schema* schema : {&renamed, &retyped, &longer}) {
        // A missing file shows the schema is checked before the file is read
        EXPECT_EQ(ErrorOf([&] { db.LoadTable("t", *schema, TempPath("_missing.tbl"), Columnar()); }),
                  "table t exists with a different schema");
    }
    EXPECT_EQ(db.GetDiskManager()->GetNumPages(), pages);

    TableInfo* info = db.GetCatalog()->GetTable("t");
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->schema.GetColumn(1).name, "v");
    ColumnVector keys(ColumnType::BigInt());
    info->columnar->ReadChunk(0, 0, keys);
    ASSERT_EQ(keys.Size(), 1000u);
    EXPECT_EQ(keys.GetInt(999), 1000);
    const std::vector<RID> got = db.GetCatalog()->GetIndex("idx_k")->index->PointLookup(Value(int64_t{42}));
    ASSERT_EQ(got.size(), 1u);
    EXPECT_EQ(KeyAt(*info->columnar, got[0]), 42);
}

TEST_F(ColumnarIndexTest, ReplaceLeavesOldPagesUntouched) {
    const std::string v1 = WriteTbl("v1", KeyRange(1, 5000));
    const std::string v2 = WriteTbl("v2", KeyRange(5001, 10000));
    page_id_t committed = 0;
    std::vector<char> before;
    {
        Database db(db_file_);
        db.LoadTable("t", KeyValue(), v1, Columnar());
        IndexInfo* index = db.GetCatalog()->CreateIndex("idx_k", "t", "k");
        ASSERT_NE(index, nullptr);
        // Nothing is handed out after a commit, so the page count is what it committed
        committed = db.GetDiskManager()->GetNumPages();
        ASSERT_EQ(std::filesystem::file_size(db_file_), static_cast<uint64_t>(committed) * PAGE_SIZE);
        ASSERT_GT(committed, 2);
        before = ReadPages(2, committed - 2);

        // Pulls the old tree into the pool before the replace
        ASSERT_EQ(index->index->PointLookup(Value(int64_t{4321})).size(), 1u);
        db.LoadTable("t", KeyValue(), v2, Columnar());
    }
    EXPECT_GT(std::filesystem::file_size(db_file_), static_cast<uint64_t>(committed) * PAGE_SIZE);
    EXPECT_TRUE(ReadPages(2, committed - 2) == before);
}

TEST_F(ColumnarIndexTest, FailedReplaceKeepsOldTable) {
    const std::string v1 = WriteTbl("v1", KeyRange(1, 1000));
    const std::string bad = WriteTbl("bad", KeyRange(1001, 1500) + "abc|0\n" + KeyRange(1502, 2000));
    const std::string nulls = WriteTbl("nulls", "1001|0\n\\N|0\n1003|0\n");
    const Schema nullable({Column("k", ColumnType::BigInt(), true), Column("v", ColumnType::Integer())});
    LoadOptions null_options = Columnar();
    null_options.null_text = "\\N";

    // Checks v1 and its index still answer, with v1's schema.
    const auto expect_v1 = [](Database& db) {
        TableInfo* info = db.GetCatalog()->GetTable("t");
        ASSERT_NE(info, nullptr);
        EXPECT_EQ(info->oid, 16384u);
        EXPECT_EQ(info->stats.row_count, 1000u);
        EXPECT_FALSE(info->schema.GetColumn(0).nullable);
        IndexInfo* index = db.GetCatalog()->GetIndex("idx_k");
        ASSERT_NE(index, nullptr);
        EXPECT_EQ(index->oid, 16385u);
        EXPECT_EQ(index->root_page_id, index->index->GetRootPageId());
        const std::vector<RID> got = index->index->PointLookup(Value(int64_t{500}));
        ASSERT_EQ(got.size(), 1u);
        EXPECT_EQ(KeyAt(*info->columnar, got[0]), 500);
        EXPECT_TRUE(index->index->PointLookup(Value(int64_t{1001})).empty());
    };

    {
        Database db(db_file_);
        db.LoadTable("t", KeyValue(), v1, Columnar());
        ASSERT_NE(db.GetCatalog()->CreateIndex("idx_k", "t", "k"), nullptr);

        const std::string msg = ErrorOf([&] { db.LoadTable("t", KeyValue(), bad, Columnar()); });
        EXPECT_EQ(msg.rfind("t: line 501, column k: ", 0), 0u) << msg;
        expect_v1(db);

        EXPECT_EQ(ErrorOf([&] { db.LoadTable("t", nullable, nulls, null_options); }),
                  "Cannot create index on column 'k': it holds a NULL");
        expect_v1(db);
    }
    Database db(db_file_);
    expect_v1(db);
}

TEST_F(ColumnarIndexTest, IndexOnEmptyTable) {
    const std::string tbl = WriteTbl("empty", "");
    {
        Database db(db_file_);
        db.LoadTable("t", KeyValue(), tbl, Columnar());
        IndexInfo* index = db.GetCatalog()->CreateIndex("idx_k", "t", "k");
        ASSERT_NE(index, nullptr);
        EXPECT_EQ(index->root_page_id, INVALID_PAGE_ID);
        EXPECT_TRUE(index->index->PointLookup(Value(int64_t{1})).empty());
    }

    const auto saved = SavedRoots();
    ASSERT_EQ(saved.size(), 1u);
    EXPECT_EQ(saved[0].second, INVALID_PAGE_ID);

    Database db(db_file_);
    IndexInfo* index = db.GetCatalog()->GetIndex("idx_k");
    ASSERT_NE(index, nullptr);
    EXPECT_EQ(index->root_page_id, INVALID_PAGE_ID);
    EXPECT_EQ(index->index->GetRootPageId(), INVALID_PAGE_ID);
    EXPECT_TRUE(index->index->PointLookup(Value(int64_t{1})).empty());
    EXPECT_TRUE(index->index->Begin().IsEnd());
}

TEST_F(ColumnarIndexTest, NullKeyRefused) {
    const std::string tbl = WriteTbl("nulls", "1|0\n\\N|0\n3|0\n");
    const Schema schema({Column("k", ColumnType::BigInt(), true), Column("v", ColumnType::Integer())});
    Database db(db_file_);
    LoadOptions options = Columnar();
    options.null_text = "\\N";
    db.LoadTable("t", schema, tbl, options);
    EXPECT_EQ(ErrorOf([&] { (void)db.GetCatalog()->CreateIndex("idx_k", "t", "k"); }),
              "Cannot create index on column 'k': it holds a NULL");
    EXPECT_EQ(db.GetCatalog()->GetIndex("idx_k"), nullptr);

    // The failed build took no OID
    IndexInfo* index = db.GetCatalog()->CreateIndex("idx_v", "t", "v");
    ASSERT_NE(index, nullptr);
    EXPECT_EQ(index->oid, 16385u);
}


TEST_F(ColumnarIndexTest, ZeroedRootThrows) {
    const page_id_t root = CommitTwoLevelIndex();
    WritePage(root, std::vector<char>(PAGE_SIZE, 0));
    ExpectDamaged(777, 777, "zeroed root");
}

TEST_F(ColumnarIndexTest, ChildPointingAtItsParentThrows) {
    const page_id_t root = CommitTwoLevelIndex();
    std::vector<char> page = ReadPages(root, 1);
    BTreeInternalPage::SetChild(page.data(), 0, root, TypeId::BIGINT);
    WritePage(root, page);
    ExpectDamaged(1, 1, "child pointing at its parent");
}

TEST_F(ColumnarIndexTest, LeafChainCycleThrows) {
    const page_id_t root = CommitTwoLevelIndex();
    const std::vector<char> root_page = ReadPages(root, 1);
    const page_id_t last = BTreeInternalPage::GetChild(root_page.data(), BTreeInternalPage::GetNumKeys(root_page.data()),
                                                       TypeId::BIGINT);
    std::vector<char> leaf = ReadPages(last, 1);
    ASSERT_EQ(BTreeLeafPage::GetNextLeafPageId(leaf.data()), INVALID_PAGE_ID) << "the last leaf ends the chain";
    BTreeLeafPage::SetNextLeafPageId(leaf.data(), last);
    WritePage(last, leaf);
    ExpectDamaged(kDamageRows, kDamageRows - 10, "leaf chain cycle");
}

}  // namespace
}  // namespace shilmandb
