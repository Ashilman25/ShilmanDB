#include <gtest/gtest.h>
#include "executor/index_scan_executor.hpp"
#include "planner/plan_node.hpp"
#include "catalog/catalog.hpp"
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "storage/disk_manager.hpp"
#include "common/exception.hpp"
#include <filesystem>
#include <memory>

namespace shilmandb {

class IndexScanExecutorTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() / "shilmandb_index_scan_test.db").string();
        std::filesystem::remove(test_file_);
    }

    void TearDown() override {
        std::filesystem::remove(test_file_);
    }

    struct BPMBundle {
        std::unique_ptr<DiskManager> disk_manager;
        std::unique_ptr<BufferPoolManager> bpm;
    };

    static BPMBundle MakeBPM(const std::string& path, size_t pool_size = 1000) {
        auto dm = std::make_unique<DiskManager>(path);
        auto eviction = std::make_unique<LRUEvictionPolicy>(pool_size);
        auto bpm = std::make_unique<BufferPoolManager>(
            pool_size, dm.get(), std::move(eviction));
        return {std::move(dm), std::move(bpm)};
    }

    static Schema MakeSchema() {
        return Schema({
            Column("id", TypeId::INTEGER),
            Column("val", TypeId::INTEGER)
        });
    }

    // Helper: creates table with N rows (id=i, val=i*10) and an index on "id"
    struct TestEnv {
        BPMBundle bundle;
        std::unique_ptr<Catalog> catalog;
        ExecutorContext ctx;
    };

    TestEnv SetUpTable(int num_rows) {
        auto bundle = MakeBPM(test_file_);
        auto catalog = std::make_unique<Catalog>(bundle.bpm.get());
        auto schema = MakeSchema();
        auto* table_info = catalog->CreateTable("t", schema);

        for (int i = 0; i < num_rows; ++i) {
            std::vector<Value> vals = {Value(static_cast<int32_t>(i)), Value(static_cast<int32_t>(i * 10))};
            (void)table_info->table->InsertTuple(Tuple(vals, schema));
        }

        (void)catalog->CreateIndex("idx_id", "t", "id");

        ExecutorContext ctx{bundle.bpm.get(), catalog.get()};
        return {std::move(bundle), std::move(catalog), ctx};
    }
};

TEST_F(IndexScanExecutorTest, FullIndexScan) {
    auto env = SetUpTable(100);
    auto schema = MakeSchema();

    auto plan = std::make_unique<IndexScanPlanNode>(schema, "t", "idx_id");
    // No low_key or high_key — full scan
    IndexScanExecutor executor(plan.get(), &env.ctx);
    executor.Init();

    int count = 0;
    int prev_id = -1;
    Tuple result;
    while (executor.Next(&result)) {
        auto id_val = result.GetValue(schema, 0);
        // B+tree yields keys in ascending order
        EXPECT_GT(id_val.integer_, prev_id);
        prev_id = id_val.integer_;

        auto val = result.GetValue(schema, 1);
        EXPECT_EQ(val.integer_, id_val.integer_ * 10);

        ++count;
    }
    EXPECT_EQ(count, 100);
    executor.Close();
}

TEST_F(IndexScanExecutorTest, RangeScanLowKeyOnly) {
    auto env = SetUpTable(100);
    auto schema = MakeSchema();

    auto plan = std::make_unique<IndexScanPlanNode>(schema, "t", "idx_id");
    plan->low_key = Value(static_cast<int32_t>(50));
    // No high_key

    IndexScanExecutor executor(plan.get(), &env.ctx);
    executor.Init();

    int count = 0;
    Tuple result;
    while (executor.Next(&result)) {
        auto id_val = result.GetValue(schema, 0);
        EXPECT_GE(id_val.integer_, 50);
        ++count;
    }
    EXPECT_EQ(count, 50);  // keys 50..99
    executor.Close();
}

TEST_F(IndexScanExecutorTest, RangeScanHighKeyOnly) {
    auto env = SetUpTable(100);
    auto schema = MakeSchema();

    auto plan = std::make_unique<IndexScanPlanNode>(schema, "t", "idx_id");
    // No low_key
    plan->high_key = Value(static_cast<int32_t>(49));

    IndexScanExecutor executor(plan.get(), &env.ctx);
    executor.Init();

    int count = 0;
    Tuple result;
    while (executor.Next(&result)) {
        auto id_val = result.GetValue(schema, 0);
        EXPECT_LE(id_val.integer_, 49);
        ++count;
    }
    EXPECT_EQ(count, 50);  // keys 0..49
    executor.Close();
}

TEST_F(IndexScanExecutorTest, RangeScanBothBounds) {
    auto env = SetUpTable(100);
    auto schema = MakeSchema();

    auto plan = std::make_unique<IndexScanPlanNode>(schema, "t", "idx_id");
    plan->low_key = Value(static_cast<int32_t>(25));
    plan->high_key = Value(static_cast<int32_t>(74));

    IndexScanExecutor executor(plan.get(), &env.ctx);
    executor.Init();

    int count = 0;
    Tuple result;
    while (executor.Next(&result)) {
        auto id_val = result.GetValue(schema, 0);
        EXPECT_GE(id_val.integer_, 25);
        EXPECT_LE(id_val.integer_, 74);
        ++count;
    }
    EXPECT_EQ(count, 50);  // keys 25..74
    executor.Close();
}

TEST_F(IndexScanExecutorTest, EmptyRange) {
    auto env = SetUpTable(100);
    auto schema = MakeSchema();

    auto plan = std::make_unique<IndexScanPlanNode>(schema, "t", "idx_id");
    plan->low_key = Value(static_cast<int32_t>(200));
    plan->high_key = Value(static_cast<int32_t>(300));

    IndexScanExecutor executor(plan.get(), &env.ctx);
    executor.Init();

    Tuple result;
    EXPECT_FALSE(executor.Next(&result));
    executor.Close();
}

TEST_F(IndexScanExecutorTest, MissingIndexThrows) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());
    (void)catalog.CreateTable("t", MakeSchema());

    auto plan = std::make_unique<IndexScanPlanNode>(MakeSchema(), "t", "nonexistent");
    ExecutorContext ctx{bundle.bpm.get(), &catalog};
    IndexScanExecutor executor(plan.get(), &ctx);

    EXPECT_THROW(executor.Init(), DatabaseException);
}

TEST_F(IndexScanExecutorTest, ReinitResetsIteration) {
    auto env = SetUpTable(50);
    auto schema = MakeSchema();

    auto plan = std::make_unique<IndexScanPlanNode>(schema, "t", "idx_id");
    IndexScanExecutor executor(plan.get(), &env.ctx);

    // First scan
    executor.Init();
    int count1 = 0;
    Tuple result;
    while (executor.Next(&result)) { ++count1; }
    EXPECT_EQ(count1, 50);
    executor.Close();

    // Re-init and scan again
    executor.Init();
    int count2 = 0;
    while (executor.Next(&result)) { ++count2; }
    EXPECT_EQ(count2, 50);
    executor.Close();
}

}  // namespace shilmandb
