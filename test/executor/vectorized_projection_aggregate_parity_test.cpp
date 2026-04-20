#include <gtest/gtest.h>

#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "catalog/catalog.hpp"
#include "planner/plan_node.hpp"
#include "storage/disk_manager.hpp"

#include "vectorized_parity_harness.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace shilmandb {

// --- AST builder helpers (duplicated per project convention) ---

static auto MakeColRef(const std::string& col) {
    auto e = std::make_unique<ColumnRef>();
    e->column_name = col;
    return e;
}

static auto MakeLiteral(int32_t v) {
    auto e = std::make_unique<Literal>();
    e->value = Value(v);
    return e;
}

static auto MakeBinOp(BinaryOp::Op op,
                      std::unique_ptr<Expression> lhs,
                      std::unique_ptr<Expression> rhs) {
    auto e = std::make_unique<BinaryOp>();
    e->op = op;
    e->left = std::move(lhs);
    e->right = std::move(rhs);
    return e;
}

// --- Test fixture ---

class VectorizedProjectionAggregateParityTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() /
                      "shilmandb_vec_proj_agg_test.db").string();
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
            Column("val", TypeId::INTEGER),
        });
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
        auto schema = MakeInputSchema();
        auto* table_info = catalog->CreateTable("t", schema);
        for (int i = 0; i < num_rows; ++i) {
            std::vector<Value> vals = {
                Value(static_cast<int32_t>(i)),
                Value(static_cast<int32_t>(i * 10)),
            };
            (void)table_info->table->InsertTuple(Tuple(vals, schema));
        }
        ExecutorContext ctx{bundle.bpm.get(), catalog.get()};
        return {std::move(bundle), std::move(catalog), ctx, schema};
    }
};

// --- Projection parity ---

TEST_F(VectorizedProjectionAggregateParityTest, ProjectionOverSeqScan) {
    // Plan: Projection(id, val * 2) over SeqScan(t) — 50 rows.
    auto env = SetUpEnv(50);

    auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");

    Schema out_schema({
        Column("id", TypeId::INTEGER),
        Column("val_doubled", TypeId::INTEGER),
    });

    std::vector<std::unique_ptr<Expression>> exprs;
    exprs.push_back(MakeColRef("id"));
    exprs.push_back(MakeBinOp(BinaryOp::Op::MUL, MakeColRef("val"), MakeLiteral(2)));

    auto proj_plan = std::make_unique<ProjectionPlanNode>(out_schema, std::move(exprs));
    proj_plan->children.push_back(std::move(scan_plan));

    auto runs = test::RunBothModes(proj_plan.get(), &env.ctx);
    test::ExpectRowsEqual(runs.tuple_mode, runs.vectorized_mode, runs.schema);
    EXPECT_EQ(runs.tuple_mode.size(), 50u);
    // Spot check first row: id=0 → (0, 0)
    EXPECT_EQ(runs.tuple_mode[0].GetValue(out_schema, 0).integer_, 0);
    EXPECT_EQ(runs.tuple_mode[0].GetValue(out_schema, 1).integer_, 0);
    // Spot check row 7: id=7 → (7, 140)
    EXPECT_EQ(runs.tuple_mode[7].GetValue(out_schema, 0).integer_, 7);
    EXPECT_EQ(runs.tuple_mode[7].GetValue(out_schema, 1).integer_, 140);
}

TEST_F(VectorizedProjectionAggregateParityTest, ProjectionOverFilterSeqScan) {
    // Plan: Projection(id, val) over Filter(val >= 200) over SeqScan(t) — 100 rows.
    // Exercises the selection-vector → rebuilt-tuple boundary: Filter sets a sel
    // vec on the child chunk; Projection rebuilds tuples (output has no sel vec).
    auto env = SetUpEnv(100);

    auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");

    auto filter_pred = MakeBinOp(BinaryOp::Op::GTE, MakeColRef("val"), MakeLiteral(200));
    auto filter_plan = std::make_unique<FilterPlanNode>(env.schema, std::move(filter_pred));
    filter_plan->children.push_back(std::move(scan_plan));

    Schema out_schema({
        Column("id", TypeId::INTEGER),
        Column("val", TypeId::INTEGER),
    });

    std::vector<std::unique_ptr<Expression>> exprs;
    exprs.push_back(MakeColRef("id"));
    exprs.push_back(MakeColRef("val"));

    auto proj_plan = std::make_unique<ProjectionPlanNode>(out_schema, std::move(exprs));
    proj_plan->children.push_back(std::move(filter_plan));

    auto runs = test::RunBothModes(proj_plan.get(), &env.ctx);
    test::ExpectRowsEqual(runs.tuple_mode, runs.vectorized_mode, runs.schema);
    // val >= 200 means id >= 20 (val = id*10), so 80 rows survive.
    EXPECT_EQ(runs.tuple_mode.size(), 80u);
    EXPECT_EQ(runs.tuple_mode.front().GetValue(out_schema, 0).integer_, 20);
    EXPECT_EQ(runs.tuple_mode.back().GetValue(out_schema, 0).integer_, 99);
}

// --- Aggregate parity ---

TEST_F(VectorizedProjectionAggregateParityTest, AggregateGroupBy) {
    // Plan: Aggregate(GROUP BY id; COUNT(*), SUM(val), AVG(val)) over SeqScan(t).
    // 30 rows (val = id*10) with one group per distinct id.
    auto env = SetUpEnv(30);

    auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");

    std::vector<std::unique_ptr<Expression>> group_by_exprs;
    group_by_exprs.push_back(MakeColRef("id"));

    std::vector<std::unique_ptr<Expression>> agg_exprs;
    agg_exprs.push_back(nullptr);              // COUNT(*) — arg is unused
    agg_exprs.push_back(MakeColRef("val"));    // SUM(val)
    agg_exprs.push_back(MakeColRef("val"));    // AVG(val)

    std::vector<Aggregate::Func> agg_funcs{
        Aggregate::Func::COUNT,
        Aggregate::Func::SUM,
        Aggregate::Func::AVG,
    };

    Schema out_schema({
        Column("id", TypeId::INTEGER),
        Column("cnt", TypeId::INTEGER),
        Column("sum_val", TypeId::DECIMAL),
        Column("avg_val", TypeId::DECIMAL),
    });

    auto agg_plan = std::make_unique<AggregatePlanNode>(out_schema);
    agg_plan->group_by_exprs = std::move(group_by_exprs);
    agg_plan->aggregate_exprs = std::move(agg_exprs);
    agg_plan->aggregate_funcs = std::move(agg_funcs);
    agg_plan->children.push_back(std::move(scan_plan));

    auto runs = test::RunBothModes(agg_plan.get(), &env.ctx);
    test::ExpectRowsEqual(runs.tuple_mode, runs.vectorized_mode, runs.schema);
    EXPECT_EQ(runs.tuple_mode.size(), 30u);
    // std::map iteration is lexicographic on key — first emitted group key is id=0.
    EXPECT_EQ(runs.tuple_mode.front().GetValue(out_schema, 0).integer_, 0);
    EXPECT_EQ(runs.tuple_mode.front().GetValue(out_schema, 1).integer_, 1);
    EXPECT_DOUBLE_EQ(runs.tuple_mode.front().GetValue(out_schema, 2).decimal_, 0.0);
    EXPECT_DOUBLE_EQ(runs.tuple_mode.front().GetValue(out_schema, 3).decimal_, 0.0);
}

TEST_F(VectorizedProjectionAggregateParityTest, AggregateEmptyTableZeroDefault) {
    // Plan: Aggregate(COUNT(*), SUM(val)) over SeqScan(empty table).
    // No GROUP BY → both modes MUST emit one row of zero defaults.
    // This is the flagged correctness gate for vectorized aggregation.
    auto env = SetUpEnv(0);

    auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");

    std::vector<std::unique_ptr<Expression>> agg_exprs;
    agg_exprs.push_back(nullptr);
    agg_exprs.push_back(MakeColRef("val"));

    std::vector<Aggregate::Func> agg_funcs{
        Aggregate::Func::COUNT,
        Aggregate::Func::SUM,
    };

    Schema out_schema({
        Column("cnt", TypeId::INTEGER),
        Column("sum_val", TypeId::DECIMAL),
    });

    auto agg_plan = std::make_unique<AggregatePlanNode>(out_schema);
    agg_plan->aggregate_exprs = std::move(agg_exprs);
    agg_plan->aggregate_funcs = std::move(agg_funcs);
    agg_plan->children.push_back(std::move(scan_plan));

    auto runs = test::RunBothModes(agg_plan.get(), &env.ctx);
    test::ExpectRowsEqual(runs.tuple_mode, runs.vectorized_mode, runs.schema);
    ASSERT_EQ(runs.tuple_mode.size(), 1u);
    EXPECT_EQ(runs.tuple_mode.front().GetValue(out_schema, 0).integer_, 0);
    EXPECT_DOUBLE_EQ(runs.tuple_mode.front().GetValue(out_schema, 1).decimal_, 0.0);
    ASSERT_EQ(runs.vectorized_mode.size(), 1u);
    EXPECT_EQ(runs.vectorized_mode.front().GetValue(out_schema, 0).integer_, 0);
    EXPECT_DOUBLE_EQ(runs.vectorized_mode.front().GetValue(out_schema, 1).decimal_, 0.0);
}

}  // namespace shilmandb
