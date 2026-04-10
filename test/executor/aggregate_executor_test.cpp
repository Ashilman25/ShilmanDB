#include <gtest/gtest.h>
#include "executor/aggregate_executor.hpp"
#include "executor/seq_scan_executor.hpp"
#include "planner/plan_node.hpp"
#include "catalog/catalog.hpp"
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "storage/disk_manager.hpp"
#include <filesystem>
#include <memory>

namespace shilmandb {

// --- AST builder helpers ---

static auto MakeColRef(const std::string& col) {
    auto e = std::make_unique<ColumnRef>();
    e->column_name = col;
    return e;
}

// --- Test fixture ---

class AggregateExecutorTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() / "shilmandb_agg_test.db").string();
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

    static Schema MakeInputSchema() {
        return Schema({
            Column("id", TypeId::INTEGER),
            Column("val", TypeId::INTEGER)
        });
    }

    struct TestEnv {
        BPMBundle bundle;
        std::unique_ptr<Catalog> catalog;
        ExecutorContext ctx;
        Schema schema;
    };

    TestEnv SetUpEnv() {
        auto bundle = MakeBPM(test_file_);
        auto catalog = std::make_unique<Catalog>(bundle.bpm.get());
        auto schema = MakeInputSchema();
        (void)catalog->CreateTable("t", schema);
        ExecutorContext ctx{bundle.bpm.get(), catalog.get()};
        return {std::move(bundle), std::move(catalog), ctx, schema};
    }

    void InsertRow(TestEnv& env, int32_t id, int32_t val) {
        auto* table_info = env.catalog->GetTable("t");
        std::vector<Value> vals = {Value(id), Value(val)};
        (void)table_info->table->InsertTuple(Tuple(vals, env.schema));
    }

    // Build SeqScan -> Aggregate pipeline, collect all output tuples
    std::vector<Tuple> RunAggregate(TestEnv& env,
                                    std::vector<std::unique_ptr<Expression>> group_by_exprs,
                                    std::vector<std::unique_ptr<Expression>> agg_exprs,
                                    std::vector<Aggregate::Func> agg_funcs,
                                    Schema output_schema) {
        auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");
        auto agg_plan = std::make_unique<AggregatePlanNode>(std::move(output_schema));
        agg_plan->group_by_exprs = std::move(group_by_exprs);
        agg_plan->aggregate_exprs = std::move(agg_exprs);
        agg_plan->aggregate_funcs = std::move(agg_funcs);

        auto child = std::make_unique<SeqScanExecutor>(scan_plan.get(), &env.ctx);
        AggregateExecutor agg(agg_plan.get(), &env.ctx, std::move(child));
        agg.Init();

        std::vector<Tuple> results;
        Tuple result;
        while (agg.Next(&result)) {
            results.push_back(result);
        }
        agg.Close();
        return results;
    }
};

// --- Tests ---

TEST_F(AggregateExecutorTest, GroupBySumSingleAgg) {
    auto env = SetUpEnv();
    InsertRow(env, 1, 10);
    InsertRow(env, 1, 20);
    InsertRow(env, 2, 30);

    // GROUP BY id, SUM(val)
    std::vector<std::unique_ptr<Expression>> gb;
    gb.push_back(MakeColRef("id"));

    std::vector<std::unique_ptr<Expression>> agg_exprs;
    agg_exprs.push_back(MakeColRef("val"));

    std::vector<Aggregate::Func> agg_funcs = {Aggregate::Func::SUM};

    Schema out_schema({Column("id", TypeId::INTEGER), Column("sum_val", TypeId::DECIMAL)});

    auto results = RunAggregate(env, std::move(gb), std::move(agg_exprs),
                                std::move(agg_funcs), std::move(out_schema));
    ASSERT_EQ(results.size(), 2u);

    // std::map orders by key, so group 1 comes before group 2
    Schema result_schema({Column("id", TypeId::INTEGER), Column("sum_val", TypeId::DECIMAL)});
    EXPECT_EQ(results[0].GetValue(result_schema, 0).integer_, 1);
    EXPECT_DOUBLE_EQ(results[0].GetValue(result_schema, 1).decimal_, 30.0);

    EXPECT_EQ(results[1].GetValue(result_schema, 0).integer_, 2);
    EXPECT_DOUBLE_EQ(results[1].GetValue(result_schema, 1).decimal_, 30.0);
}

TEST_F(AggregateExecutorTest, CountStarNoGroupBy) {
    auto env = SetUpEnv();
    for (int i = 0; i < 5; ++i) {
        InsertRow(env, i, i * 10);
    }

    // COUNT(*) — no GROUP BY, nullptr aggregate expr
    std::vector<std::unique_ptr<Expression>> gb;  // empty
    std::vector<std::unique_ptr<Expression>> agg_exprs;
    agg_exprs.push_back(nullptr);  // COUNT(*)

    std::vector<Aggregate::Func> agg_funcs = {Aggregate::Func::COUNT};

    Schema out_schema({Column("count_star", TypeId::INTEGER)});

    auto results = RunAggregate(env, std::move(gb), std::move(agg_exprs),
                                std::move(agg_funcs), std::move(out_schema));
    ASSERT_EQ(results.size(), 1u);
    Schema result_schema({Column("count_star", TypeId::INTEGER)});
    EXPECT_EQ(results[0].GetValue(result_schema, 0).integer_, 5);
}

TEST_F(AggregateExecutorTest, AvgNoGroupBy) {
    auto env = SetUpEnv();
    InsertRow(env, 1, 10);
    InsertRow(env, 2, 20);
    InsertRow(env, 3, 30);

    // AVG(val) — no GROUP BY
    std::vector<std::unique_ptr<Expression>> gb;
    std::vector<std::unique_ptr<Expression>> agg_exprs;
    agg_exprs.push_back(MakeColRef("val"));

    std::vector<Aggregate::Func> agg_funcs = {Aggregate::Func::AVG};

    Schema out_schema({Column("avg_val", TypeId::DECIMAL)});

    auto results = RunAggregate(env, std::move(gb), std::move(agg_exprs),
                                std::move(agg_funcs), std::move(out_schema));
    ASSERT_EQ(results.size(), 1u);
    Schema result_schema({Column("avg_val", TypeId::DECIMAL)});
    EXPECT_DOUBLE_EQ(results[0].GetValue(result_schema, 0).decimal_, 20.0);
}

TEST_F(AggregateExecutorTest, AvgFractionalPrecision) {
    auto env = SetUpEnv();
    InsertRow(env, 1, 10);
    InsertRow(env, 2, 15);

    // AVG(val) = (10 + 15) / 2 = 12.5, NOT 12 (integer truncation)
    std::vector<std::unique_ptr<Expression>> gb;
    std::vector<std::unique_ptr<Expression>> agg_exprs;
    agg_exprs.push_back(MakeColRef("val"));

    std::vector<Aggregate::Func> agg_funcs = {Aggregate::Func::AVG};

    Schema out_schema({Column("avg_val", TypeId::DECIMAL)});

    auto results = RunAggregate(env, std::move(gb), std::move(agg_exprs),
                                std::move(agg_funcs), std::move(out_schema));
    ASSERT_EQ(results.size(), 1u);
    Schema result_schema({Column("avg_val", TypeId::DECIMAL)});
    EXPECT_DOUBLE_EQ(results[0].GetValue(result_schema, 0).decimal_, 12.5);
}

TEST_F(AggregateExecutorTest, MinMaxNoGroupBy) {
    auto env = SetUpEnv();
    InsertRow(env, 1, 10);
    InsertRow(env, 2, 30);
    InsertRow(env, 3, 20);

    // MIN(val), MAX(val) — no GROUP BY
    std::vector<std::unique_ptr<Expression>> gb;
    std::vector<std::unique_ptr<Expression>> agg_exprs;
    agg_exprs.push_back(MakeColRef("val"));
    agg_exprs.push_back(MakeColRef("val"));

    std::vector<Aggregate::Func> agg_funcs = {Aggregate::Func::MIN, Aggregate::Func::MAX};

    Schema out_schema({Column("min_val", TypeId::INTEGER), Column("max_val", TypeId::INTEGER)});

    auto results = RunAggregate(env, std::move(gb), std::move(agg_exprs),
                                std::move(agg_funcs), std::move(out_schema));
    ASSERT_EQ(results.size(), 1u);
    Schema result_schema({Column("min_val", TypeId::INTEGER), Column("max_val", TypeId::INTEGER)});
    EXPECT_EQ(results[0].GetValue(result_schema, 0).integer_, 10);
    EXPECT_EQ(results[0].GetValue(result_schema, 1).integer_, 30);
}

TEST_F(AggregateExecutorTest, EmptyTableNoGroupBy) {
    auto env = SetUpEnv();
    // No rows inserted

    // COUNT(*) on empty table — should return 1 row with count=0
    std::vector<std::unique_ptr<Expression>> gb;
    std::vector<std::unique_ptr<Expression>> agg_exprs;
    agg_exprs.push_back(nullptr);

    std::vector<Aggregate::Func> agg_funcs = {Aggregate::Func::COUNT};

    Schema out_schema({Column("count_star", TypeId::INTEGER)});

    auto results = RunAggregate(env, std::move(gb), std::move(agg_exprs),
                                std::move(agg_funcs), std::move(out_schema));
    ASSERT_EQ(results.size(), 1u);
    Schema result_schema({Column("count_star", TypeId::INTEGER)});
    EXPECT_EQ(results[0].GetValue(result_schema, 0).integer_, 0);
}

TEST_F(AggregateExecutorTest, EmptyTableWithGroupBy) {
    auto env = SetUpEnv();
    // No rows inserted

    // GROUP BY id, SUM(val) on empty table — should return 0 rows
    std::vector<std::unique_ptr<Expression>> gb;
    gb.push_back(MakeColRef("id"));

    std::vector<std::unique_ptr<Expression>> agg_exprs;
    agg_exprs.push_back(MakeColRef("val"));

    std::vector<Aggregate::Func> agg_funcs = {Aggregate::Func::SUM};

    Schema out_schema({Column("id", TypeId::INTEGER), Column("sum_val", TypeId::DECIMAL)});

    auto results = RunAggregate(env, std::move(gb), std::move(agg_exprs),
                                std::move(agg_funcs), std::move(out_schema));
    EXPECT_TRUE(results.empty());
}

TEST_F(AggregateExecutorTest, MultipleAggregatesPerGroup) {
    auto env = SetUpEnv();
    InsertRow(env, 1, 10);
    InsertRow(env, 1, 20);
    InsertRow(env, 2, 30);

    // GROUP BY id, COUNT(*), SUM(val), MIN(val), MAX(val)
    std::vector<std::unique_ptr<Expression>> gb;
    gb.push_back(MakeColRef("id"));

    std::vector<std::unique_ptr<Expression>> agg_exprs;
    agg_exprs.push_back(nullptr);           // COUNT(*)
    agg_exprs.push_back(MakeColRef("val")); // SUM(val)
    agg_exprs.push_back(MakeColRef("val")); // MIN(val)
    agg_exprs.push_back(MakeColRef("val")); // MAX(val)

    std::vector<Aggregate::Func> agg_funcs = {
        Aggregate::Func::COUNT,
        Aggregate::Func::SUM,
        Aggregate::Func::MIN,
        Aggregate::Func::MAX
    };

    Schema out_schema({
        Column("id", TypeId::INTEGER),
        Column("count_star", TypeId::INTEGER),
        Column("sum_val", TypeId::DECIMAL),
        Column("min_val", TypeId::INTEGER),
        Column("max_val", TypeId::INTEGER)
    });

    auto results = RunAggregate(env, std::move(gb), std::move(agg_exprs),
                                std::move(agg_funcs), std::move(out_schema));
    ASSERT_EQ(results.size(), 2u);

    Schema result_schema({
        Column("id", TypeId::INTEGER),
        Column("count_star", TypeId::INTEGER),
        Column("sum_val", TypeId::DECIMAL),
        Column("min_val", TypeId::INTEGER),
        Column("max_val", TypeId::INTEGER)
    });

    // Group 1: count=2, sum=30.0, min=10, max=20
    EXPECT_EQ(results[0].GetValue(result_schema, 0).integer_, 1);
    EXPECT_EQ(results[0].GetValue(result_schema, 1).integer_, 2);
    EXPECT_DOUBLE_EQ(results[0].GetValue(result_schema, 2).decimal_, 30.0);
    EXPECT_EQ(results[0].GetValue(result_schema, 3).integer_, 10);
    EXPECT_EQ(results[0].GetValue(result_schema, 4).integer_, 20);

    // Group 2: count=1, sum=30.0, min=30, max=30
    EXPECT_EQ(results[1].GetValue(result_schema, 0).integer_, 2);
    EXPECT_EQ(results[1].GetValue(result_schema, 1).integer_, 1);
    EXPECT_DOUBLE_EQ(results[1].GetValue(result_schema, 2).decimal_, 30.0);
    EXPECT_EQ(results[1].GetValue(result_schema, 3).integer_, 30);
    EXPECT_EQ(results[1].GetValue(result_schema, 4).integer_, 30);
}

TEST_F(AggregateExecutorTest, SingleRowGroup) {
    auto env = SetUpEnv();
    InsertRow(env, 1, 10);

    // GROUP BY id, SUM(val)
    std::vector<std::unique_ptr<Expression>> gb;
    gb.push_back(MakeColRef("id"));

    std::vector<std::unique_ptr<Expression>> agg_exprs;
    agg_exprs.push_back(MakeColRef("val"));

    std::vector<Aggregate::Func> agg_funcs = {Aggregate::Func::SUM};

    Schema out_schema({Column("id", TypeId::INTEGER), Column("sum_val", TypeId::DECIMAL)});

    auto results = RunAggregate(env, std::move(gb), std::move(agg_exprs),
                                std::move(agg_funcs), std::move(out_schema));
    ASSERT_EQ(results.size(), 1u);
    Schema result_schema({Column("id", TypeId::INTEGER), Column("sum_val", TypeId::DECIMAL)});
    EXPECT_EQ(results[0].GetValue(result_schema, 0).integer_, 1);
    EXPECT_DOUBLE_EQ(results[0].GetValue(result_schema, 1).decimal_, 10.0);
}

TEST_F(AggregateExecutorTest, ReinitResetsGroups) {
    auto env = SetUpEnv();
    InsertRow(env, 1, 10);
    InsertRow(env, 2, 20);

    auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");
    Schema out_schema({Column("count_star", TypeId::INTEGER)});
    auto agg_plan = std::make_unique<AggregatePlanNode>(std::move(out_schema));
    agg_plan->aggregate_exprs.push_back(nullptr);
    agg_plan->aggregate_funcs.push_back(Aggregate::Func::COUNT);

    auto child = std::make_unique<SeqScanExecutor>(scan_plan.get(), &env.ctx);
    AggregateExecutor agg(agg_plan.get(), &env.ctx, std::move(child));

    // First pass
    agg.Init();
    Tuple result;
    int count1 = 0;
    while (agg.Next(&result)) { ++count1; }
    EXPECT_EQ(count1, 1);

    // Re-init and drain again
    agg.Init();
    int count2 = 0;
    while (agg.Next(&result)) { ++count2; }
    EXPECT_EQ(count2, 1);
    agg.Close();
}

TEST_F(AggregateExecutorTest, NextBeforeInitAsserts) {
    auto env = SetUpEnv();
    auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");
    Schema out_schema({Column("count_star", TypeId::INTEGER)});
    auto agg_plan = std::make_unique<AggregatePlanNode>(std::move(out_schema));
    agg_plan->aggregate_exprs.push_back(nullptr);
    agg_plan->aggregate_funcs.push_back(Aggregate::Func::COUNT);

    auto child = std::make_unique<SeqScanExecutor>(scan_plan.get(), &env.ctx);
    AggregateExecutor agg(agg_plan.get(), &env.ctx, std::move(child));

    Tuple t;
    EXPECT_DEATH(agg.Next(&t), "Next.*called before Init");
}

}  // namespace shilmandb
