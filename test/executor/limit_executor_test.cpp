#include <gtest/gtest.h>
#include "executor/limit_executor.hpp"
#include "executor/seq_scan_executor.hpp"
#include "planner/plan_node.hpp"
#include "catalog/catalog.hpp"
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "storage/disk_manager.hpp"
#include <filesystem>
#include <memory>

namespace shilmandb {

class LimitExecutorTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() / "shilmandb_limit_test.db").string();
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
        return Schema({Column("id", TypeId::INTEGER), Column("val", TypeId::INTEGER)});
    }

    struct TestEnv {
        BPMBundle bundle;
        std::unique_ptr<Catalog> catalog;
        ExecutorContext ctx;
        Schema schema;
    };

    TestEnv SetUpEnv(int num_rows) {
        auto bundle = MakeBPM(test_file_);
        auto catalog = std::make_unique<Catalog>(bundle.bpm.get());
        auto schema = MakeSchema();
        auto* table_info = catalog->CreateTable("t", schema);

        for (int i = 0; i < num_rows; ++i) {
            std::vector<Value> vals = {Value(static_cast<int32_t>(i)),
                                       Value(static_cast<int32_t>(i * 10))};
            (void)table_info->table->InsertTuple(Tuple(vals, schema));
        }

        ExecutorContext ctx{bundle.bpm.get(), catalog.get()};
        return {std::move(bundle), std::move(catalog), ctx, schema};
    }

    int RunLimit(TestEnv& env, int64_t limit) {
        auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");
        auto limit_plan = std::make_unique<LimitPlanNode>(env.schema, limit);

        auto child = std::make_unique<SeqScanExecutor>(scan_plan.get(), &env.ctx);
        LimitExecutor limiter(limit_plan.get(), &env.ctx, std::move(child));
        limiter.Init();

        int count = 0;
        Tuple result;
        while (limiter.Next(&result)) { ++count; }
        limiter.Close();
        return count;
    }
};

TEST_F(LimitExecutorTest, LimitSmall) {
    auto env = SetUpEnv(100);
    EXPECT_EQ(RunLimit(env, 10), 10);
}

TEST_F(LimitExecutorTest, LimitSmallVerifyValues) {
    auto env = SetUpEnv(10);

    auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");
    auto limit_plan = std::make_unique<LimitPlanNode>(env.schema, 3);

    auto child = std::make_unique<SeqScanExecutor>(scan_plan.get(), &env.ctx);
    LimitExecutor limiter(limit_plan.get(), &env.ctx, std::move(child));
    limiter.Init();

    std::vector<Tuple> results;
    Tuple result;
    while (limiter.Next(&result)) {
        results.push_back(std::move(result));
    }
    limiter.Close();

    // Should return the first 3 tuples: (0,0), (1,10), (2,20)
    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0].GetValue(env.schema, 0).integer_, 0);
    EXPECT_EQ(results[0].GetValue(env.schema, 1).integer_, 0);
    EXPECT_EQ(results[1].GetValue(env.schema, 0).integer_, 1);
    EXPECT_EQ(results[1].GetValue(env.schema, 1).integer_, 10);
    EXPECT_EQ(results[2].GetValue(env.schema, 0).integer_, 2);
    EXPECT_EQ(results[2].GetValue(env.schema, 1).integer_, 20);
}

TEST_F(LimitExecutorTest, LimitExceedsInput) {
    auto env = SetUpEnv(5);
    EXPECT_EQ(RunLimit(env, 100), 5);
}

TEST_F(LimitExecutorTest, LimitZero) {
    auto env = SetUpEnv(5);
    EXPECT_EQ(RunLimit(env, 0), 0);
}

TEST_F(LimitExecutorTest, LimitOne) {
    auto env = SetUpEnv(5);
    EXPECT_EQ(RunLimit(env, 1), 1);
}

TEST_F(LimitExecutorTest, EmptyInput) {
    auto env = SetUpEnv(0);
    EXPECT_EQ(RunLimit(env, 10), 0);
}

TEST_F(LimitExecutorTest, ReinitResetsEmitted) {
    auto env = SetUpEnv(10);

    auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");
    auto limit_plan = std::make_unique<LimitPlanNode>(env.schema, 3);

    auto child = std::make_unique<SeqScanExecutor>(scan_plan.get(), &env.ctx);
    LimitExecutor limiter(limit_plan.get(), &env.ctx, std::move(child));

    // First pass
    limiter.Init();
    Tuple result;
    int count1 = 0;
    while (limiter.Next(&result)) { ++count1; }
    EXPECT_EQ(count1, 3);

    // Re-init and drain again
    limiter.Init();
    int count2 = 0;
    while (limiter.Next(&result)) { ++count2; }
    EXPECT_EQ(count2, 3);
    limiter.Close();
}

TEST_F(LimitExecutorTest, NextBeforeInitAsserts) {
    auto env = SetUpEnv(0);

    auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");
    auto limit_plan = std::make_unique<LimitPlanNode>(env.schema, 10);

    auto child = std::make_unique<SeqScanExecutor>(scan_plan.get(), &env.ctx);
    LimitExecutor limiter(limit_plan.get(), &env.ctx, std::move(child));

    Tuple t;
    EXPECT_DEATH(limiter.Next(&t), "Next.*called before Init");
}

}  // namespace shilmandb
