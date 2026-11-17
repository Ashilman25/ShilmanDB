#include <gtest/gtest.h>

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "catalog/catalog.hpp"
#include "common/exception.hpp"
#include "engine/database.hpp"
#include "storage/columnar/bulk_loader.hpp"
#include "storage/disk_manager.hpp"
#include "storage/table_heap.hpp"

namespace shilmandb {
namespace {

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

// Id, price and name: one column each of integer, money and text.
Schema ThreeColumns() {
    return Schema({Column("id", ColumnType::Integer()), Column("price", ColumnType::Numeric(15, 2)),
                   Column("name", ColumnType::Varchar(25))});
}

// Two integer columns for the bigger files.
Schema TwoInts() { return Schema({Column("a", ColumnType::Integer()), Column("b", ColumnType::Integer())}); }

// Rows of a and 2a, with line bad_line (1-based) replaced by bad_text.
std::string IntLines(size_t rows, size_t bad_line = 0, const std::string& bad_text = "") {
    std::string text;
    for (size_t n = 1; n <= rows; ++n) {
        text += n == bad_line ? bad_text : std::to_string(n) + "|" + std::to_string(n * 2);
        text += "\n";
    }
    return text;
}

// Columnar storage on two threads.
LoadOptions Columnar() {
    LoadOptions options;
    options.storage = StorageKind::COLUMNAR;
    options.threads = 2;
    return options;
}

// Checks two stats hold the same counts and ranges.
void ExpectSameStats(const TableStats& a, const TableStats& b) {
    EXPECT_EQ(a.row_count, b.row_count);
    EXPECT_EQ(a.page_count, b.page_count);
    EXPECT_EQ(a.distinct_counts, b.distinct_counts);
    ASSERT_EQ(a.min_max.size(), b.min_max.size());
    for (const auto& [name, range] : a.min_max) {
        auto it = b.min_max.find(name);
        ASSERT_NE(it, b.min_max.end()) << name;
        EXPECT_TRUE(range.min == it->second.min) << name;
        EXPECT_TRUE(range.max == it->second.max) << name;
    }
}

class ColumnarDatabaseTest : public ::testing::Test {
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
                ("shilmandb_columnar_db_test_" + std::to_string(::getpid()) + suffix)).string();
    }

    // Writes text to a per-test .tbl file and returns its path.
    std::string WriteTbl(const std::string& what, const std::string& text) {
        const std::string path = TempPath("_" + what + ".tbl");
        std::ofstream(path, std::ios::binary) << text;
        tbl_files_.push_back(path);
        return path;
    }

    // The db file's size on disk.
    uint64_t FileBytes() const { return std::filesystem::file_size(db_file_); }
};

TEST_F(ColumnarDatabaseTest, LoadCloseReopen) {
    const std::string tbl = WriteTbl("three", "1|10.50|apple\n2|3.25|pear\n3|99.99|plum\n4|0.01|fig\n");
    TableStats loaded;
    {
        Database db(db_file_);
        db.LoadTable("fruit", ThreeColumns(), tbl, Columnar());
        auto* info = db.GetCatalog()->GetTable("fruit");
        ASSERT_NE(info, nullptr);
        EXPECT_EQ(info->storage, StorageKind::COLUMNAR);
        EXPECT_EQ(info->oid, 16384u);
        EXPECT_EQ(info->table, nullptr);
        EXPECT_EQ(info->first_page_id, INVALID_PAGE_ID);
        ASSERT_NE(info->columnar, nullptr);
        EXPECT_EQ(info->stats.row_count, 4u);
        loaded = info->stats;
    }

    Database db(db_file_);
    EXPECT_EQ(db.GetDiskManager()->GetReadStats().reads, 0u);
    EXPECT_EQ(db.GetBufferPoolManager()->GetMissCount(), 0u);

    auto* info = db.GetCatalog()->GetTable("fruit");
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->storage, StorageKind::COLUMNAR);
    EXPECT_EQ(info->oid, 16384u);
    EXPECT_EQ(info->table, nullptr);
    EXPECT_EQ(info->first_page_id, INVALID_PAGE_ID);
    ASSERT_NE(info->columnar, nullptr);
    EXPECT_EQ(info->columnar->Meta().row_count, 4u);

    const Schema want = ThreeColumns();
    ASSERT_EQ(info->schema.GetColumnCount(), want.GetColumnCount());
    for (uint32_t i = 0; i < want.GetColumnCount(); ++i) {
        EXPECT_EQ(info->schema.GetColumn(i).name, want.GetColumn(i).name);
        EXPECT_EQ(info->schema.GetColumn(i).type, want.GetColumn(i).type);
        EXPECT_TRUE(info->schema.GetColumn(i).storage_type == want.GetColumn(i).storage_type);
        EXPECT_EQ(info->schema.GetColumn(i).nullable, want.GetColumn(i).nullable);
    }
    ExpectSameStats(info->stats, loaded);
}

TEST_F(ColumnarDatabaseTest, NextOidSurvivesReopen) {
    const std::string tbl = WriteTbl("oids", IntLines(10));
    {
        Database db(db_file_);
        db.LoadTable("first", TwoInts(), tbl, Columnar());
        ASSERT_NE(nullptr, db.GetCatalog()->GetTable("first"));
        EXPECT_EQ(db.GetCatalog()->GetTable("first")->oid, 16384u);
    }
    {
        Database db(db_file_);
        db.LoadTable("second", TwoInts(), tbl, Columnar());
        ASSERT_NE(nullptr, db.GetCatalog()->GetTable("second"));
        EXPECT_EQ(db.GetCatalog()->GetTable("second")->oid, 16385u);
    }
    Database db(db_file_);
    auto* first = db.GetCatalog()->GetTable("first");
    auto* second = db.GetCatalog()->GetTable("second");
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(first->oid, 16384u);
    EXPECT_EQ(second->oid, 16385u);
}

TEST_F(ColumnarDatabaseTest, UpdateTableStatsKeepsColumnarStats) {
    const std::string tbl = WriteTbl("stats", "1|10.50|apple\n2|3.25|pear\n2|99.99|plum\n");
    Database db(db_file_);
    db.LoadTable("fruit", ThreeColumns(), tbl, Columnar());
    auto* info = db.GetCatalog()->GetTable("fruit");
    ASSERT_NE(info, nullptr);
    const TableStats before = info->stats;
    ASSERT_EQ(before.row_count, 3u);
    ASSERT_FALSE(before.min_max.empty());

    db.GetCatalog()->UpdateTableStats("fruit");
    ExpectSameStats(info->stats, before);
}

TEST_F(ColumnarDatabaseTest, NewDatabaseFileHasHeader) {
    { Database db(db_file_); }
    EXPECT_EQ(FileBytes(), 2u * PAGE_SIZE);
    {
        Database db(db_file_);
        EXPECT_EQ(db.GetDiskManager()->GetNumPages(), 2u);
    }
    EXPECT_EQ(FileBytes(), 2u * PAGE_SIZE);
}

TEST_F(ColumnarDatabaseTest, LegacyFileRefusedByDatabase) {
    {
        DiskManager disk(db_file_);
        BufferPoolManager bpm(8, &disk, std::make_unique<LRUEvictionPolicy>(8));
        TableHeap heap(&bpm);
        const Schema schema({Column("id", TypeId::INTEGER), Column("name", TypeId::VARCHAR)});
        ASSERT_NE(heap.InsertTuple(Tuple({Value(int32_t{1}), Value(std::string("one"))}, schema)).page_id, INVALID_PAGE_ID);
        bpm.FlushAllPages();
    }
    const uint64_t before = FileBytes();
    EXPECT_EQ(ErrorOf([&] { Database db(db_file_); }),
              db_file_ + ": not a ShilmanDB file, or an older format; reload it");
    EXPECT_EQ(FileBytes(), before);
}

TEST_F(ColumnarDatabaseTest, RowTablesAreSessionOnly) {
    const std::string tbl = WriteTbl("rows", IntLines(2000));
    {
        Database db(db_file_);
        db.LoadTable("r", TwoInts(), tbl);
        auto* info = db.GetCatalog()->GetTable("r");
        ASSERT_NE(info, nullptr);
        EXPECT_EQ(info->storage, StorageKind::ROW);
        EXPECT_EQ(info->oid, 0u);
        EXPECT_EQ(info->stats.row_count, 2000u);
        EXPECT_GT(db.GetDiskManager()->GetNumPages(), 2u);
    }
    EXPECT_GT(FileBytes(), 2u * PAGE_SIZE);

    Database db(db_file_);
    EXPECT_EQ(db.GetCatalog()->GetTable("r"), nullptr);
    EXPECT_EQ(db.GetDiskManager()->GetNumPages(), 2u);
    EXPECT_EQ(FileBytes(), 2u * PAGE_SIZE);
}

TEST_F(ColumnarDatabaseTest, NameCollisions) {
    const std::string tbl = WriteTbl("small", IntLines(10));
    Database db(db_file_);

    db.LoadTable("r", TwoInts(), tbl);
    EXPECT_EQ(ErrorOf([&] { db.LoadTable("r", TwoInts(), tbl, Columnar()); }),
              "table r already exists as a row-store table");
    EXPECT_EQ(db.GetCatalog()->GetTable("r")->storage, StorageKind::ROW);

    db.LoadTable("c", TwoInts(), tbl, Columnar());
    EXPECT_EQ(ErrorOf([&] { db.LoadTable("c", TwoInts(), tbl); }), "Failed to create table: c");
    EXPECT_EQ(db.GetCatalog()->CreateTable("c", TwoInts()), nullptr);
    // The same columns again replace the table
    EXPECT_EQ(ErrorOf([&] { db.LoadTable("c", TwoInts(), tbl, Columnar()); }), "");
    ASSERT_NE(nullptr, db.GetCatalog()->GetTable("c"));
    EXPECT_EQ(db.GetCatalog()->GetTable("c")->storage, StorageKind::COLUMNAR);
    EXPECT_EQ(db.GetCatalog()->GetTable("c")->oid, 16384u);
}

TEST_F(ColumnarDatabaseTest, CatalogWithoutFileRefusesColumnar) {
    const std::string tbl = WriteTbl("nofile", IntLines(10));
    DiskManager disk(db_file_);
    BufferPoolManager bpm(8, &disk, std::make_unique<LRUEvictionPolicy>(8));
    Catalog catalog(&bpm);
    EXPECT_EQ(ErrorOf([&] { (void)catalog.LoadColumnarTable("c", TwoInts(), tbl, BulkLoadOptions{}); }),
              "columnar tables need a database file");
    EXPECT_EQ(catalog.GetTable("c"), nullptr);
}

TEST_F(ColumnarDatabaseTest, RowLoadRefusesNullText) {
    const std::string tbl = WriteTbl("nulltext", IntLines(10));
    Database db(db_file_);
    LoadOptions options;
    options.null_text = "";
    EXPECT_EQ(ErrorOf([&] { db.LoadTable("r", TwoInts(), tbl, options); }), "null_text needs columnar storage");
    EXPECT_EQ(db.GetCatalog()->GetTable("r"), nullptr);

    options.null_text.reset();
    options.delimiter = ',';
    db.LoadTable("r", TwoInts(), WriteTbl("commas", "1,2\n3,4\n"), options);
    ASSERT_NE(db.GetCatalog()->GetTable("r"), nullptr);
    EXPECT_EQ(db.GetCatalog()->GetTable("r")->stats.row_count, 2u);
}

TEST_F(ColumnarDatabaseTest, EmptyFile) {
    const std::string tbl = WriteTbl("empty", "");
    {
        Database db(db_file_);
        db.LoadTable("e", TwoInts(), tbl, Columnar());
        auto* info = db.GetCatalog()->GetTable("e");
        ASSERT_NE(info, nullptr);
        EXPECT_EQ(info->stats.row_count, 0u);
        EXPECT_EQ(info->columnar->RowGroupCount(), 0u);
    }
    Database db(db_file_);
    auto* info = db.GetCatalog()->GetTable("e");
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->stats.row_count, 0u);
    EXPECT_EQ(info->columnar->RowGroupCount(), 0u);
    EXPECT_EQ(info->schema.GetColumnCount(), 2u);
}

TEST_F(ColumnarDatabaseTest, FailedLoadThenGoodLoad) {
    const std::string bad = WriteTbl("bad", IntLines(100000, 70000, "abc|5"));
    const std::string good = WriteTbl("good", IntLines(100000));
    {
        Database db(db_file_);
        const std::string msg = ErrorOf([&] { db.LoadTable("broken", TwoInts(), bad, Columnar()); });
        EXPECT_EQ(msg.rfind("broken: line 70000, column a: ", 0), 0u) << msg;
        EXPECT_EQ(db.GetCatalog()->GetTable("broken"), nullptr);

        db.LoadTable("fine", TwoInts(), good, Columnar());
        auto* info = db.GetCatalog()->GetTable("fine");
        ASSERT_NE(info, nullptr);
        EXPECT_EQ(info->stats.row_count, 100000u);
        EXPECT_EQ(info->oid, 16384u);
    }
    Database db(db_file_);
    EXPECT_EQ(db.GetCatalog()->GetTable("broken"), nullptr);
    auto* info = db.GetCatalog()->GetTable("fine");
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->stats.row_count, 100000u);
    EXPECT_EQ(info->columnar->RowGroupCount(), 2u);
}

TEST_F(ColumnarDatabaseTest, HeapPageIdsShiftByTwo) {
    Database db(db_file_);
    auto* info = db.GetCatalog()->CreateTable("t", TwoInts());
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->first_page_id, 2);
}

}  // namespace
}  // namespace shilmandb
