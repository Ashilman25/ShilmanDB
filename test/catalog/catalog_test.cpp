#include <gtest/gtest.h>
#include "catalog/catalog.hpp"
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "storage/disk_manager.hpp"
#include "common/exception.hpp"
#include <filesystem>
#include <memory>

namespace shilmandb {

class CatalogTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() / "shilmandb_catalog_test.db").string();
        std::filesystem::remove(test_file_);
    }

    void TearDown() override {
        std::filesystem::remove(test_file_);
    }

    struct BPMBundle {
        std::unique_ptr<DiskManager> disk_manager;
        std::unique_ptr<BufferPoolManager> bpm;
    };

    BPMBundle MakeBPM(size_t pool_size = 1000) {
        auto dm = std::make_unique<DiskManager>(test_file_);
        auto eviction = std::make_unique<LRUEvictionPolicy>(pool_size);
        auto bpm = std::make_unique<BufferPoolManager>(
            pool_size, dm.get(), std::move(eviction));
        return {std::move(dm), std::move(bpm)};
    }

    static Schema MakeSimpleSchema() {
        return Schema({
            Column("id", TypeId::INTEGER),
            Column("value", TypeId::INTEGER),
            Column("name", TypeId::VARCHAR)
        });
    }
};

// ---------------------------------------------------------------------------
// Table tests
// ---------------------------------------------------------------------------
TEST_F(CatalogTest, CreateTableAndGetTable) {
    auto bundle = MakeBPM();
    Catalog catalog(bundle.bpm.get());
    auto schema = MakeSimpleSchema();

    auto* info = catalog.CreateTable("test_table", schema);
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->name, "test_table");
    EXPECT_EQ(info->schema.GetColumnCount(), 3u);

    auto* retrieved = catalog.GetTable("test_table");
    EXPECT_EQ(retrieved, info);
}

TEST_F(CatalogTest, CreateTableDuplicateReturnsNullptr) {
    auto bundle = MakeBPM();
    Catalog catalog(bundle.bpm.get());
    auto schema = MakeSimpleSchema();

    auto* first = catalog.CreateTable("t", schema);
    ASSERT_NE(first, nullptr);

    auto* second = catalog.CreateTable("t", schema);
    EXPECT_EQ(second, nullptr);

    // Original is still accessible
    EXPECT_EQ(catalog.GetTable("t"), first);
}

TEST_F(CatalogTest, GetTableNonexistentReturnsNullptr) {
    auto bundle = MakeBPM();
    Catalog catalog(bundle.bpm.get());

    EXPECT_EQ(catalog.GetTable("no_such_table"), nullptr);
}

TEST_F(CatalogTest, CreateTableHasValidFirstPageId) {
    auto bundle = MakeBPM();
    Catalog catalog(bundle.bpm.get());
    auto schema = MakeSimpleSchema();

    auto* info = catalog.CreateTable("t", schema);
    ASSERT_NE(info, nullptr);
    EXPECT_NE(info->first_page_id, INVALID_PAGE_ID);
    EXPECT_EQ(info->first_page_id, info->table->GetFirstPageId());
}

// ---------------------------------------------------------------------------
// Index tests
// ---------------------------------------------------------------------------
TEST_F(CatalogTest, CreateIndexOnEmptyTable) {
    auto bundle = MakeBPM();
    Catalog catalog(bundle.bpm.get());
    auto schema = MakeSimpleSchema();

    (void)catalog.CreateTable("t", schema);
    auto* idx = catalog.CreateIndex("idx_id", "t", "id");

    ASSERT_NE(idx, nullptr);
    EXPECT_EQ(idx->index_name, "idx_id");
    EXPECT_EQ(idx->table_name, "t");
    EXPECT_EQ(idx->column_name, "id");
    EXPECT_EQ(idx->col_idx, 0u);

    // Empty table — index should have no entries
    auto rids = idx->index->PointLookup(Value(42));
    EXPECT_TRUE(rids.empty());
}

TEST_F(CatalogTest, CreateIndexAutoPopulatesFromExistingData) {
    auto bundle = MakeBPM();
    Catalog catalog(bundle.bpm.get());
    auto schema = MakeSimpleSchema();

    auto* table_info = catalog.CreateTable("t", schema);
    ASSERT_NE(table_info, nullptr);

    // Insert rows before creating the index
    constexpr int kNumRows = 50;
    std::vector<RID> inserted_rids;
    for (int32_t i = 0; i < kNumRows; ++i) {
        Tuple tuple({Value(i), Value(i * 10), Value(std::string("row_") + std::to_string(i))}, schema);
        auto rid = table_info->table->InsertTuple(tuple);
        ASSERT_NE(rid.page_id, INVALID_PAGE_ID);
        inserted_rids.push_back(rid);
    }

    // Create index on "id" column — should auto-populate
    auto* idx = catalog.CreateIndex("idx_id", "t", "id");
    ASSERT_NE(idx, nullptr);

    // Verify every key is findable via PointLookup
    for (int32_t i = 0; i < kNumRows; ++i) {
        auto rids = idx->index->PointLookup(Value(i));
        ASSERT_EQ(rids.size(), 1u) << "Key " << i << " not found in index";
        EXPECT_EQ(rids[0], inserted_rids[i]);
    }

    // Non-existent key returns empty
    auto rids = idx->index->PointLookup(Value(kNumRows + 1));
    EXPECT_TRUE(rids.empty());
}

TEST_F(CatalogTest, CreateIndexOnNonexistentTableReturnsNullptr) {
    auto bundle = MakeBPM();
    Catalog catalog(bundle.bpm.get());

    auto* idx = catalog.CreateIndex("idx", "no_table", "col");
    EXPECT_EQ(idx, nullptr);
}

TEST_F(CatalogTest, CreateIndexOnNonexistentColumnThrows) {
    auto bundle = MakeBPM();
    Catalog catalog(bundle.bpm.get());
    auto schema = MakeSimpleSchema();
    (void)catalog.CreateTable("t", schema);

    EXPECT_THROW((void)catalog.CreateIndex("idx", "t", "no_column"), DatabaseException);
}

TEST_F(CatalogTest, CreateIndexOnVarcharColumnThrows) {
    auto bundle = MakeBPM();
    Catalog catalog(bundle.bpm.get());
    auto schema = MakeSimpleSchema();  // has "name" column as VARCHAR
    (void)catalog.CreateTable("t", schema);

    EXPECT_THROW((void)catalog.CreateIndex("idx", "t", "name"), DatabaseException);
}

TEST_F(CatalogTest, CreateIndexOnNonFirstColumnAutoPopulates) {
    auto bundle = MakeBPM();
    Catalog catalog(bundle.bpm.get());
    auto schema = MakeSimpleSchema();  // columns: id(0), value(1), name(2)

    auto* table_info = catalog.CreateTable("t", schema);
    ASSERT_NE(table_info, nullptr);

    // Insert rows — "value" column gets i*10
    constexpr int kNumRows = 20;
    std::vector<RID> inserted_rids;
    for (int32_t i = 0; i < kNumRows; ++i) {
        Tuple tuple({Value(i), Value(i * 10), Value(std::string("r"))}, schema);
        auto rid = table_info->table->InsertTuple(tuple);
        ASSERT_NE(rid.page_id, INVALID_PAGE_ID);
        inserted_rids.push_back(rid);
    }

    // Index on column 1 ("value"), not column 0
    auto* idx = catalog.CreateIndex("idx_value", "t", "value");
    ASSERT_NE(idx, nullptr);
    EXPECT_EQ(idx->col_idx, 1u);

    // Verify keys are the "value" column values (i*10), not "id" column
    for (int32_t i = 0; i < kNumRows; ++i) {
        auto rids = idx->index->PointLookup(Value(i * 10));
        ASSERT_EQ(rids.size(), 1u) << "Key " << (i * 10) << " not found";
        EXPECT_EQ(rids[0], inserted_rids[i]);
    }

    // "id" values should NOT be in this index
    auto rids = idx->index->PointLookup(Value(5));  // id=5 exists but value=5 doesn't
    EXPECT_TRUE(rids.empty());
}

TEST_F(CatalogTest, GetTableIndexesReturnsCorrectSubset) {
    auto bundle = MakeBPM();
    Catalog catalog(bundle.bpm.get());
    auto schema = MakeSimpleSchema();

    (void)catalog.CreateTable("t1", schema);
    (void)catalog.CreateTable("t2", schema);

    (void)catalog.CreateIndex("idx_t1_id", "t1", "id");
    (void)catalog.CreateIndex("idx_t1_value", "t1", "value");
    (void)catalog.CreateIndex("idx_t2_id", "t2", "id");

    auto t1_indexes = catalog.GetTableIndexes("t1");
    EXPECT_EQ(t1_indexes.size(), 2u);

    auto t2_indexes = catalog.GetTableIndexes("t2");
    EXPECT_EQ(t2_indexes.size(), 1u);
    EXPECT_EQ(t2_indexes[0]->index_name, "idx_t2_id");

    auto no_indexes = catalog.GetTableIndexes("no_table");
    EXPECT_TRUE(no_indexes.empty());
}

TEST_F(CatalogTest, GetIndexAndDuplicateIndexName) {
    auto bundle = MakeBPM();
    Catalog catalog(bundle.bpm.get());
    auto schema = MakeSimpleSchema();
    (void)catalog.CreateTable("t", schema);

    auto* idx = catalog.CreateIndex("idx", "t", "id");
    ASSERT_NE(idx, nullptr);

    // GetIndex returns the same pointer
    EXPECT_EQ(catalog.GetIndex("idx"), idx);

    // Duplicate index name returns nullptr
    EXPECT_EQ(catalog.CreateIndex("idx", "t", "value"), nullptr);

    // Nonexistent index returns nullptr
    EXPECT_EQ(catalog.GetIndex("no_idx"), nullptr);
}

// ---------------------------------------------------------------------------
// Stats tests
// ---------------------------------------------------------------------------
TEST_F(CatalogTest, UpdateTableStatsRowCount) {
    auto bundle = MakeBPM();
    Catalog catalog(bundle.bpm.get());
    auto schema = MakeSimpleSchema();

    auto* table_info = catalog.CreateTable("t", schema);
    ASSERT_NE(table_info, nullptr);

    constexpr int kNumRows = 25;
    for (int32_t i = 0; i < kNumRows; ++i) {
        Tuple tuple({Value(i), Value(i * 10), Value(std::string("s"))}, schema);
        (void)table_info->table->InsertTuple(tuple);
    }

    catalog.UpdateTableStats("t");
    EXPECT_EQ(table_info->stats.row_count, kNumRows);
}

TEST_F(CatalogTest, UpdateTableStatsDistinctCounts) {
    auto bundle = MakeBPM();
    Catalog catalog(bundle.bpm.get());

    // Schema: id (unique), category (3 distinct), flag (2 distinct)
    Schema schema({
        Column("id", TypeId::INTEGER),
        Column("category", TypeId::INTEGER),
        Column("flag", TypeId::INTEGER)
    });

    auto* table_info = catalog.CreateTable("t", schema);
    ASSERT_NE(table_info, nullptr);

    // 10 rows: id is unique, category cycles 0-1-2, flag alternates 0-1
    for (int32_t i = 0; i < 10; ++i) {
        Tuple tuple({Value(i), Value(i % 3), Value(i % 2)}, schema);
        (void)table_info->table->InsertTuple(tuple);
    }

    catalog.UpdateTableStats("t");

    EXPECT_EQ(table_info->stats.row_count, 10u);
    EXPECT_EQ(table_info->stats.distinct_counts.at("id"), 10u);
    EXPECT_EQ(table_info->stats.distinct_counts.at("category"), 3u);
    EXPECT_EQ(table_info->stats.distinct_counts.at("flag"), 2u);
}

TEST_F(CatalogTest, UpdateTableStatsWithMixedTypes) {
    auto bundle = MakeBPM();
    Catalog catalog(bundle.bpm.get());

    Schema schema({
        Column("id", TypeId::INTEGER),
        Column("label", TypeId::VARCHAR)
    });

    auto* table_info = catalog.CreateTable("t", schema);
    ASSERT_NE(table_info, nullptr);

    // 6 rows: id unique, label has 3 distinct values
    std::vector<std::string> labels = {"alpha", "beta", "gamma", "alpha", "beta", "alpha"};
    for (int32_t i = 0; i < 6; ++i) {
        Tuple tuple({Value(i), Value(labels[i])}, schema);
        (void)table_info->table->InsertTuple(tuple);
    }

    catalog.UpdateTableStats("t");

    EXPECT_EQ(table_info->stats.row_count, 6u);
    EXPECT_EQ(table_info->stats.distinct_counts.at("id"), 6u);
    EXPECT_EQ(table_info->stats.distinct_counts.at("label"), 3u);
}

TEST_F(CatalogTest, UpdateTableStatsEmptyTable) {
    auto bundle = MakeBPM();
    Catalog catalog(bundle.bpm.get());
    auto schema = MakeSimpleSchema();

    auto* table_info = catalog.CreateTable("t", schema);
    ASSERT_NE(table_info, nullptr);

    catalog.UpdateTableStats("t");
    EXPECT_EQ(table_info->stats.row_count, 0u);
    // Distinct counts should exist for each column, all zero
    for (uint32_t i = 0; i < schema.GetColumnCount(); ++i) {
        EXPECT_EQ(table_info->stats.distinct_counts.at(schema.GetColumn(i).name), 0u);
    }
}

TEST_F(CatalogTest, UpdateTableStatsNonexistentTableIsNoop) {
    auto bundle = MakeBPM();
    Catalog catalog(bundle.bpm.get());

    // Should not throw or crash
    catalog.UpdateTableStats("no_table");
}

// ---------------------------------------------------------------------------
// Integration tests
// ---------------------------------------------------------------------------
TEST_F(CatalogTest, MultipleTablesAndIndexesCoexist) {
    auto bundle = MakeBPM();
    Catalog catalog(bundle.bpm.get());

    Schema schema1({Column("a", TypeId::INTEGER), Column("b", TypeId::INTEGER)});
    Schema schema2({Column("x", TypeId::BIGINT), Column("y", TypeId::DECIMAL)});

    auto* t1 = catalog.CreateTable("t1", schema1);
    auto* t2 = catalog.CreateTable("t2", schema2);
    ASSERT_NE(t1, nullptr);
    ASSERT_NE(t2, nullptr);
    EXPECT_NE(t1->first_page_id, t2->first_page_id);

    (void)catalog.CreateIndex("idx_a", "t1", "a");
    (void)catalog.CreateIndex("idx_x", "t2", "x");

    EXPECT_NE(catalog.GetIndex("idx_a"), nullptr);
    EXPECT_NE(catalog.GetIndex("idx_x"), nullptr);
    EXPECT_EQ(catalog.GetTableIndexes("t1").size(), 1u);
    EXPECT_EQ(catalog.GetTableIndexes("t2").size(), 1u);
}

TEST_F(CatalogTest, StatsUpdateReflectsNewInserts) {
    auto bundle = MakeBPM();
    Catalog catalog(bundle.bpm.get());

    Schema schema({Column("id", TypeId::INTEGER)});
    auto* table_info = catalog.CreateTable("t", schema);
    ASSERT_NE(table_info, nullptr);

    // Insert 5 rows, update stats
    for (int32_t i = 0; i < 5; ++i) {
        Tuple tuple({Value(i)}, schema);
        (void)table_info->table->InsertTuple(tuple);
    }
    catalog.UpdateTableStats("t");
    EXPECT_EQ(table_info->stats.row_count, 5u);
    EXPECT_EQ(table_info->stats.distinct_counts.at("id"), 5u);

    // Insert 5 more, re-update — stats should reflect all 10
    for (int32_t i = 5; i < 10; ++i) {
        Tuple tuple({Value(i)}, schema);
        (void)table_info->table->InsertTuple(tuple);
    }
    catalog.UpdateTableStats("t");
    EXPECT_EQ(table_info->stats.row_count, 10u);
    EXPECT_EQ(table_info->stats.distinct_counts.at("id"), 10u);
}

}  // namespace shilmandb
