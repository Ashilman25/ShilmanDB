#include <gtest/gtest.h>
#include "executor/seq_scan_executor.hpp"
#include "planner/plan_node.hpp"
#include "catalog/catalog.hpp"
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "storage/disk_manager.hpp"
#include "common/exception.hpp"
#include <filesystem>
#include <memory>

namespace shilmandb {

class SeqScanExecutorTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() / "shilmandb_seq_scan_test.db").string();
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
            Column("name", TypeId::VARCHAR)
        });
    }
};

TEST_F(SeqScanExecutorTest, ScanAllTuples) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());
    auto schema = MakeSchema();
    auto* table_info = catalog.CreateTable("t", schema);
    ASSERT_NE(table_info, nullptr);

    constexpr int kNumRows = 100;
    for (int i = 0; i < kNumRows; ++i) {
        std::vector<Value> vals = {Value(static_cast<int32_t>(i)), Value("row_" + std::to_string(i))};
        Tuple tuple(vals, schema);
        (void)table_info->table->InsertTuple(tuple);
    }

    auto plan = std::make_unique<SeqScanPlanNode>(schema, "t");
    ExecutorContext ctx{bundle.bpm.get(), &catalog};
    SeqScanExecutor executor(plan.get(), &ctx);

    executor.Init();

    int count = 0;
    Tuple result;
    while (executor.Next(&result)) {
        auto id_val = result.GetValue(schema, 0);
        EXPECT_EQ(id_val.integer_, count);

        auto name_val = result.GetValue(schema, 1);
        EXPECT_EQ(name_val.varchar_, "row_" + std::to_string(count));

        ++count;
    }
    EXPECT_EQ(count, kNumRows);

    executor.Close();
}

TEST_F(SeqScanExecutorTest, EmptyTable) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());
    auto schema = MakeSchema();
    (void)catalog.CreateTable("empty", schema);

    auto plan = std::make_unique<SeqScanPlanNode>(schema, "empty");
    ExecutorContext ctx{bundle.bpm.get(), &catalog};
    SeqScanExecutor executor(plan.get(), &ctx);

    executor.Init();

    Tuple result;
    EXPECT_FALSE(executor.Next(&result));

    executor.Close();
}

TEST_F(SeqScanExecutorTest, OutputSchemaMatchesTable) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());
    auto schema = MakeSchema();
    (void)catalog.CreateTable("t", schema);

    auto plan = std::make_unique<SeqScanPlanNode>(schema, "t");
    ExecutorContext ctx{bundle.bpm.get(), &catalog};
    SeqScanExecutor executor(plan.get(), &ctx);

    const auto& output = executor.GetOutputSchema();
    ASSERT_EQ(output.GetColumnCount(), 2u);
    EXPECT_EQ(output.GetColumn(0).name, "id");
    EXPECT_EQ(output.GetColumn(0).type, TypeId::INTEGER);
    EXPECT_EQ(output.GetColumn(1).name, "name");
    EXPECT_EQ(output.GetColumn(1).type, TypeId::VARCHAR);
}

TEST_F(SeqScanExecutorTest, MissingTableThrows) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    auto plan = std::make_unique<SeqScanPlanNode>(Schema(std::vector<Column>{}), "nonexistent");
    ExecutorContext ctx{bundle.bpm.get(), &catalog};
    SeqScanExecutor executor(plan.get(), &ctx);

    EXPECT_THROW(executor.Init(), DatabaseException);
}

TEST_F(SeqScanExecutorTest, ReinitResetsIteration) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());
    auto schema = MakeSchema();
    auto* table_info = catalog.CreateTable("t", schema);
    ASSERT_NE(table_info, nullptr);

    for (int i = 0; i < 10; ++i) {
        std::vector<Value> vals = {Value(static_cast<int32_t>(i)), Value("r")};
        (void)table_info->table->InsertTuple(Tuple(vals, schema));
    }

    auto plan = std::make_unique<SeqScanPlanNode>(schema, "t");
    ExecutorContext ctx{bundle.bpm.get(), &catalog};
    SeqScanExecutor executor(plan.get(), &ctx);

    // First scan
    executor.Init();
    int count1 = 0;
    Tuple result;
    while (executor.Next(&result)) { ++count1; }
    EXPECT_EQ(count1, 10);
    executor.Close();

    // Re-init and scan again
    executor.Init();
    int count2 = 0;
    while (executor.Next(&result)) { ++count2; }
    EXPECT_EQ(count2, 10);
    executor.Close();
}

}  // namespace shilmandb
