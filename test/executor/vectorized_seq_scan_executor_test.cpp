#include <gtest/gtest.h>
#include "executor/vectorized_seq_scan_executor.hpp"
#include "planner/plan_node.hpp"
#include "catalog/catalog.hpp"
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "storage/disk_manager.hpp"
#include "common/exception.hpp"
#include "vectorized_parity_harness.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace shilmandb {

class VectorizedSeqScanExecutorTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() /
                      "shilmandb_vec_seq_scan_test.db").string();
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
            Column("name", TypeId::VARCHAR),
        });
    }

    // Populate a fresh table "t" with `n` rows: (i, "row_{i}").
    struct TestEnv {
        BPMBundle bundle;
        std::unique_ptr<Catalog> catalog;
        std::unique_ptr<SeqScanPlanNode> plan;
        ExecutorContext ctx;
        Schema schema;
    };

    TestEnv MakeEnv(int num_rows) {
        auto bundle = MakeBPM(test_file_);
        auto catalog = std::make_unique<Catalog>(bundle.bpm.get());
        auto schema = MakeSchema();
        auto* table_info = catalog->CreateTable("t", schema);
        for (int i = 0; i < num_rows; ++i) {
            std::vector<Value> vals = {
                Value(static_cast<int32_t>(i)),
                Value("row_" + std::to_string(i)),
            };
            (void)table_info->table->InsertTuple(Tuple(vals, schema));
        }
        auto plan = std::make_unique<SeqScanPlanNode>(schema, "t");
        ExecutorContext ctx{bundle.bpm.get(), catalog.get()};
        return {std::move(bundle), std::move(catalog), std::move(plan), ctx, schema};
    }
};

// --- Tuple-interface tests ---

TEST_F(VectorizedSeqScanExecutorTest, ScanAllTuples) {
    auto env = MakeEnv(100);
    VectorizedSeqScanExecutor exec(env.plan.get(), &env.ctx);
    exec.Init();

    int count = 0;
    Tuple t;
    while (exec.Next(&t)) {
        EXPECT_EQ(t.GetValue(env.schema, 0).integer_, count);
        EXPECT_EQ(t.GetValue(env.schema, 1).varchar_, "row_" + std::to_string(count));
        ++count;
    }
    EXPECT_EQ(count, 100);
    exec.Close();
}

TEST_F(VectorizedSeqScanExecutorTest, NextDrainsBuffer) {
    auto env = MakeEnv(1500);
    VectorizedSeqScanExecutor exec(env.plan.get(), &env.ctx);
    exec.Init();

    int count = 0;
    Tuple t;
    while (exec.Next(&t)) {
        EXPECT_EQ(t.GetValue(env.schema, 0).integer_, count);
        ++count;
    }
    EXPECT_EQ(count, 1500);
    exec.Close();
}

TEST_F(VectorizedSeqScanExecutorTest, CloseClearsState) {
    auto env = MakeEnv(10);
    VectorizedSeqScanExecutor exec(env.plan.get(), &env.ctx);
    exec.Init();
    Tuple t;
    for (int i = 0; i < 3; ++i) ASSERT_TRUE(exec.Next(&t));
    exec.Close();

    // Re-initialize and drain from the top.
    exec.Init();
    int count = 0;
    while (exec.Next(&t)) {
        EXPECT_EQ(t.GetValue(env.schema, 0).integer_, count);
        ++count;
    }
    EXPECT_EQ(count, 10);
    exec.Close();
}

// --- NextBatch tests ---

TEST_F(VectorizedSeqScanExecutorTest, NextBatchFillsOneChunk) {
    auto env = MakeEnv(500);
    VectorizedSeqScanExecutor exec(env.plan.get(), &env.ctx);
    exec.Init();

    DataChunk chunk(env.schema);
    ASSERT_TRUE(exec.NextBatch(&chunk));
    EXPECT_EQ(chunk.size(), 500u);
    EXPECT_FALSE(chunk.HasSelectionVector());
    EXPECT_EQ(chunk.GetColumn(0)[0].integer_, 0);
    EXPECT_EQ(chunk.GetColumn(0)[499].integer_, 499);

    EXPECT_FALSE(exec.NextBatch(&chunk));
    EXPECT_EQ(chunk.size(), 0u);
    exec.Close();
}

TEST_F(VectorizedSeqScanExecutorTest, NextBatchExactCapacityBoundary) {
    auto env = MakeEnv(static_cast<int>(DataChunk::kDefaultBatchSize));  // 1024
    VectorizedSeqScanExecutor exec(env.plan.get(), &env.ctx);
    exec.Init();

    DataChunk chunk(env.schema);
    ASSERT_TRUE(exec.NextBatch(&chunk));
    EXPECT_EQ(chunk.size(), DataChunk::kDefaultBatchSize);
    EXPECT_TRUE(chunk.IsFull());

    EXPECT_FALSE(exec.NextBatch(&chunk));
    EXPECT_EQ(chunk.size(), 0u);
    exec.Close();
}

TEST_F(VectorizedSeqScanExecutorTest, NextBatchMultipleChunks) {
    auto env = MakeEnv(1500);
    VectorizedSeqScanExecutor exec(env.plan.get(), &env.ctx);
    exec.Init();

    DataChunk chunk(env.schema);
    ASSERT_TRUE(exec.NextBatch(&chunk));
    EXPECT_EQ(chunk.size(), DataChunk::kDefaultBatchSize);
    EXPECT_EQ(chunk.GetColumn(0)[0].integer_, 0);
    EXPECT_EQ(chunk.GetColumn(0)[1023].integer_, 1023);

    ASSERT_TRUE(exec.NextBatch(&chunk));
    EXPECT_EQ(chunk.size(), 1500u - DataChunk::kDefaultBatchSize);  // 476
    EXPECT_EQ(chunk.GetColumn(0)[0].integer_, 1024);
    EXPECT_EQ(chunk.GetColumn(0)[475].integer_, 1499);

    EXPECT_FALSE(exec.NextBatch(&chunk));
    exec.Close();
}

TEST_F(VectorizedSeqScanExecutorTest, NextBatchOnEmptyTable) {
    auto env = MakeEnv(0);
    VectorizedSeqScanExecutor exec(env.plan.get(), &env.ctx);
    exec.Init();

    DataChunk chunk(env.schema);
    EXPECT_FALSE(exec.NextBatch(&chunk));
    EXPECT_EQ(chunk.size(), 0u);
    exec.Close();
}

TEST_F(VectorizedSeqScanExecutorTest, ScanOverMultipleHeapPages) {
    // Pick a row count large enough to span multiple 8 KiB heap pages.
    // SlottedPage tuple overhead + (int32 + small varchar) ≈ ~40 B per row,
    // so 5000 rows comfortably cross page boundaries.
    auto env = MakeEnv(5000);
    VectorizedSeqScanExecutor exec(env.plan.get(), &env.ctx);
    exec.Init();

    int total = 0;
    DataChunk chunk(env.schema);
    while (exec.NextBatch(&chunk)) {
        for (size_t i = 0; i < chunk.size(); ++i) {
            auto row = chunk.MaterializeTuple(i);
            EXPECT_EQ(row.GetValue(env.schema, 0).integer_, total);
            ++total;
        }
    }
    EXPECT_EQ(total, 5000);
    exec.Close();
}

TEST_F(VectorizedSeqScanExecutorTest, ParityWithTupleScan) {
    auto env = MakeEnv(50);
    auto runs = test::RunBothModes(env.plan.get(), &env.ctx);
    test::ExpectRowsEqual(runs.tuple_mode, runs.vectorized_mode, runs.schema);
    EXPECT_EQ(runs.tuple_mode.size(), 50u);
}

}  // namespace shilmandb
