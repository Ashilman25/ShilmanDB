#include <gtest/gtest.h>
#include <unistd.h>
#include "executor/executor_factory.hpp"
#include "executor/profiling_executor.hpp"
#include "planner/planner.hpp"
#include "parser/parser.hpp"
#include "catalog/catalog.hpp"
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "storage/disk_manager.hpp"
#include "common/exception.hpp"
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace shilmandb {

class ProfilingExecutorTest : public ::testing::Test {
protected:
    static constexpr const char* kFilterSql = "SELECT id FROM t WHERE id > 6";
    static constexpr const char* kJoinSql = "SELECT t.id FROM t JOIN u ON t.id = u.u_t";

    struct BPMBundle {
        std::unique_ptr<DiskManager> disk_manager;
        std::unique_ptr<BufferPoolManager> bpm;
    };

    std::string test_file_;
    BPMBundle bundle_;
    std::unique_ptr<Catalog> catalog_;

    static BPMBundle MakeBPM(const std::string& path, size_t pool_size = 1000) {
        auto dm = std::make_unique<DiskManager>(path);
        auto eviction = std::make_unique<LRUEvictionPolicy>(pool_size);
        auto bpm = std::make_unique<BufferPoolManager>(
            pool_size, dm.get(), std::move(eviction));
        return {std::move(dm), std::move(bpm)};
    }

    // t holds ids 1..10. u points at t through u_t; the 11 has no match, so the join gives 4 rows.
    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() /
                      ("shilmandb_profiling_executor_test_" + std::to_string(::getpid()) + ".db")).string();
        std::filesystem::remove(test_file_);

        bundle_ = MakeBPM(test_file_);
        catalog_ = std::make_unique<Catalog>(bundle_.bpm.get());

        Schema t_schema({Column("id", TypeId::INTEGER)});
        auto* t_info = catalog_->CreateTable("t", t_schema);
        for (int32_t id = 1; id <= 10; ++id) {
            (void)t_info->table->InsertTuple(Tuple({Value(id)}, t_schema));
        }

        Schema u_schema({Column("u_id", TypeId::INTEGER), Column("u_t", TypeId::INTEGER)});
        auto* u_info = catalog_->CreateTable("u", u_schema);
        const std::vector<int32_t> u_t_values{1, 2, 2, 3, 11};
        for (size_t i = 0; i < u_t_values.size(); ++i) {
            (void)u_info->table->InsertTuple(
                Tuple({Value(static_cast<int32_t>(i + 1)), Value(u_t_values[i])}, u_schema));
        }

        catalog_->UpdateTableStats("t");
        catalog_->UpdateTableStats("u");
    }

    void TearDown() override {
        catalog_.reset();
        bundle_.bpm.reset();
        bundle_.disk_manager.reset();
        std::filesystem::remove(test_file_);
    }

    auto PlanSql(const std::string& sql) -> std::unique_ptr<PlanNode> {
        Parser parser(sql);
        auto stmt = parser.Parse();
        Planner planner(catalog_.get());
        return planner.Plan(std::move(*stmt));
    }

    auto MakeContext(OperatorStatsMap* stats) -> ExecutorContext {
        return ExecutorContext{bundle_.bpm.get(), catalog_.get(), stats};
    }

    // Runs the executor to the end and returns the first column of every row.
    static auto DrainIds(Executor& exec) -> std::vector<int32_t> {
        std::vector<int32_t> ids;
        exec.Init();
        Tuple tuple;
        while (exec.Next(&tuple)) {
            ids.push_back(tuple.GetValue(exec.GetOutputSchema(), 0).integer_);
        }
        exec.Close();
        return ids;
    }

    auto Run(const PlanNode* plan, OperatorStatsMap* stats) -> std::vector<int32_t> {
        auto ctx = MakeContext(stats);
        auto root = ExecutorFactory::CreateExecutor(plan, &ctx);
        return DrainIds(*root);
    }
};

TEST_F(ProfilingExecutorTest, CountsRowsPerOperator) {
    auto plan = PlanSql(kFilterSql);

    const PlanNode* projection = plan.get();
    ASSERT_EQ(projection->type, PlanNodeType::PROJECTION);
    const PlanNode* filter = projection->children[0].get();
    ASSERT_EQ(filter->type, PlanNodeType::FILTER);
    const PlanNode* scan = filter->children[0].get();
    ASSERT_EQ(scan->type, PlanNodeType::SEQ_SCAN);

    OperatorStatsMap stats;
    EXPECT_EQ(Run(plan.get(), &stats).size(), 4u);

    ASSERT_EQ(stats.size(), 3u);
    EXPECT_EQ(stats.at(scan).rows, 10u);
    EXPECT_EQ(stats.at(filter).rows, 4u);
    EXPECT_EQ(stats.at(projection).rows, 4u);

    EXPECT_GT(stats.at(scan).time.count(), 0);
    EXPECT_GT(stats.at(filter).time.count(), 0);
    EXPECT_GT(stats.at(projection).time.count(), 0);

    // Times are inclusive, so a parent never takes less than its child.
    EXPECT_GE(stats.at(projection).time, stats.at(filter).time);
    EXPECT_GE(stats.at(filter).time, stats.at(scan).time);
}

TEST_F(ProfilingExecutorTest, JoinTimeIncludesBuild) {
    auto plan = PlanSql(kJoinSql);
    ASSERT_EQ(plan->type, PlanNodeType::PROJECTION);
    const PlanNode* join = plan->children[0].get();
    ASSERT_EQ(join->type, PlanNodeType::HASH_JOIN);
    const PlanNode* left = join->children[0].get();
    const PlanNode* right = join->children[1].get();

    OperatorStatsMap stats;
    EXPECT_EQ(Run(plan.get(), &stats).size(), 4u);

    EXPECT_EQ(stats.at(join).rows, 4u);
    // The build side is drained inside the join's Init, so it lands in the join's time too.
    EXPECT_GE(stats.at(join).time, stats.at(left).time + stats.at(right).time);
}

TEST_F(ProfilingExecutorTest, NoWrapWithoutStats) {
    auto plan = PlanSql(kFilterSql);
    auto ctx = MakeContext(nullptr);
    auto root = ExecutorFactory::CreateExecutor(plan.get(), &ctx);

    EXPECT_EQ(dynamic_cast<ProfilingExecutor*>(root.get()), nullptr);
    EXPECT_EQ(DrainIds(*root).size(), 4u);

    auto vec_root = ExecutorFactory::CreateExecutor(plan.get(), &ctx, ExecutionMode::VECTORIZED);
    EXPECT_EQ(dynamic_cast<ProfilingExecutor*>(vec_root.get()), nullptr);
    EXPECT_EQ(DrainIds(*vec_root).size(), 4u);
}

TEST_F(ProfilingExecutorTest, WrapsWithStats) {
    auto plan = PlanSql(kFilterSql);
    OperatorStatsMap stats;
    auto ctx = MakeContext(&stats);
    auto root = ExecutorFactory::CreateExecutor(plan.get(), &ctx);

    EXPECT_NE(dynamic_cast<ProfilingExecutor*>(root.get()), nullptr);
    // Every operator gets its entry as soon as it's built.
    EXPECT_EQ(stats.size(), 3u);
}

TEST_F(ProfilingExecutorTest, SameRowsWithAndWithoutProfiling) {
    auto plan = PlanSql(kFilterSql);
    OperatorStatsMap stats;

    const auto plain = Run(plan.get(), nullptr);
    const auto profiled = Run(plan.get(), &stats);

    EXPECT_EQ(plain, (std::vector<int32_t>{7, 8, 9, 10}));
    EXPECT_EQ(profiled, plain);
}

TEST_F(ProfilingExecutorTest, VectorizedWithStatsThrows) {
    auto plan = PlanSql(kFilterSql);
    OperatorStatsMap stats;
    auto ctx = MakeContext(&stats);

    EXPECT_THROW((void)ExecutorFactory::CreateExecutor(plan.get(), &ctx, ExecutionMode::VECTORIZED),
                 DatabaseException);
    EXPECT_TRUE(stats.empty());
}

}  // namespace shilmandb
