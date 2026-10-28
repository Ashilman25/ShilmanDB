#include <gtest/gtest.h>
#include <unistd.h>

#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "catalog/catalog.hpp"
#include "parser/ast.hpp"
#include "planner/plan_node.hpp"
#include "storage/disk_manager.hpp"

#include "vectorized_parity_harness.hpp"

#include <algorithm>
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

static auto MakeEqPred(const std::string& left_col, const std::string& right_col) {
    auto e = std::make_unique<BinaryOp>();
    e->op = BinaryOp::Op::EQ;
    e->left = MakeColRef(left_col);
    e->right = MakeColRef(right_col);
    return e;
}

// --- Fixture ---

class VectorizedHybridPipelineParityTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() /
                      ("shilmandb_vec_hybrid_parity_test_" + std::to_string(::getpid()) + ".db")).string();
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

    static Schema MakeSingleTableSchema() {
        return Schema({
            Column("id", TypeId::INTEGER),
            Column("val", TypeId::INTEGER),
        });
    }

    // Left side of the join uses distinct column names to keep the combined
    // output schema unambiguous — mirrors hash_join_executor_test.cpp.
    static Schema MakeLeftSchema() {
        return Schema({
            Column("lid", TypeId::INTEGER),
            Column("lval", TypeId::INTEGER),
        });
    }

    static Schema MakeRightSchema() {
        return Schema({
            Column("rid", TypeId::INTEGER),
            Column("rval", TypeId::INTEGER),
        });
    }

    struct SingleTableEnv {
        BPMBundle bundle;
        std::unique_ptr<Catalog> catalog;
        ExecutorContext ctx;
        Schema schema;
    };

    SingleTableEnv SetUpSingleTable(int num_rows) {
        auto bundle = MakeBPM(test_file_);
        auto catalog = std::make_unique<Catalog>(bundle.bpm.get());
        auto schema = MakeSingleTableSchema();
        auto* ti = catalog->CreateTable("t", schema);
        for (int i = 0; i < num_rows; ++i) {
            std::vector<Value> vals = {
                Value(static_cast<int32_t>(i)),
                Value(static_cast<int32_t>(i * 10)),
            };
            (void)ti->table->InsertTuple(Tuple(vals, schema));
        }
        ExecutorContext ctx{bundle.bpm.get(), catalog.get()};
        return {std::move(bundle), std::move(catalog), ctx, schema};
    }

    struct JoinEnv {
        BPMBundle bundle;
        std::unique_ptr<Catalog> catalog;
        ExecutorContext ctx;
        Schema left_schema;
        Schema right_schema;
    };

    JoinEnv SetUpJoinTables(int num_rows) {
        auto bundle = MakeBPM(test_file_);
        auto catalog = std::make_unique<Catalog>(bundle.bpm.get());
        auto left_schema = MakeLeftSchema();
        auto right_schema = MakeRightSchema();
        auto* lt = catalog->CreateTable("lhs", left_schema);
        auto* rt = catalog->CreateTable("rhs", right_schema);
        for (int i = 0; i < num_rows; ++i) {
            (void)lt->table->InsertTuple(Tuple(
                {Value(static_cast<int32_t>(i)),
                 Value(static_cast<int32_t>(i * 10))},
                left_schema));
            (void)rt->table->InsertTuple(Tuple(
                {Value(static_cast<int32_t>(i)),
                 Value(static_cast<int32_t>(i + 1000))},
                right_schema));
        }
        ExecutorContext ctx{bundle.bpm.get(), catalog.get()};
        return {std::move(bundle), std::move(catalog), ctx,
                std::move(left_schema), std::move(right_schema)};
    }
};

// 1. Sort over vectorized scan — exercises BatchToTupleAdapter under SortExecutor.
TEST_F(VectorizedHybridPipelineParityTest, SortOverVectorizedScan) {
    auto env = SetUpSingleTable(/*num_rows=*/50);

    auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");

    std::vector<OrderByItem> order_by;
    order_by.push_back({MakeColRef("val"), /*ascending=*/true});
    auto sort_plan = std::make_unique<SortPlanNode>(env.schema, std::move(order_by));
    sort_plan->children.push_back(std::move(scan_plan));

    auto runs = test::RunBothModes(sort_plan.get(), &env.ctx);
    test::ExpectRowsEqual(runs.tuple_mode, runs.vectorized_mode, runs.schema);
    EXPECT_EQ(runs.tuple_mode.size(), 50u);
}

// 2. Limit over vectorized filter — adapter inserted between Filter (vec) and Limit (tuple).
TEST_F(VectorizedHybridPipelineParityTest, LimitOverVectorizedFilter) {
    auto env = SetUpSingleTable(/*num_rows=*/100);

    auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");

    auto pred = MakeBinOp(BinaryOp::Op::GTE, MakeColRef("id"), MakeLiteral(50));
    auto filter_plan = std::make_unique<FilterPlanNode>(env.schema, std::move(pred));
    filter_plan->children.push_back(std::move(scan_plan));

    auto limit_plan = std::make_unique<LimitPlanNode>(env.schema, /*limit=*/10);
    limit_plan->children.push_back(std::move(filter_plan));

    auto runs = test::RunBothModes(limit_plan.get(), &env.ctx);
    test::ExpectRowsEqual(runs.tuple_mode, runs.vectorized_mode, runs.schema);
    EXPECT_EQ(runs.tuple_mode.size(), 10u);
}

// 3. Hash join over two vectorized scans — adapter inserted on BOTH children.
TEST_F(VectorizedHybridPipelineParityTest, HashJoinOverVectorizedScans) {
    auto env = SetUpJoinTables(/*num_rows=*/20);

    auto left_scan  = std::make_unique<SeqScanPlanNode>(env.left_schema,  "lhs");
    auto right_scan = std::make_unique<SeqScanPlanNode>(env.right_schema, "rhs");

    Schema combined({
        Column("lid",  TypeId::INTEGER),
        Column("lval", TypeId::INTEGER),
        Column("rid",  TypeId::INTEGER),
        Column("rval", TypeId::INTEGER),
    });

    auto join_plan = std::make_unique<HashJoinPlanNode>(combined, MakeEqPred("lid", "rid"));
    join_plan->children.push_back(std::move(left_scan));
    join_plan->children.push_back(std::move(right_scan));

    auto runs = test::RunBothModes(join_plan.get(), &env.ctx);

    // HashJoin emission order depends on hash-bucket layout — sort both sides
    // by lid before row-by-row compare.
    auto t = runs.tuple_mode;
    auto v = runs.vectorized_mode;
    auto by_lid = [&](const Tuple& a, const Tuple& b) {
        return a.GetValue(runs.schema, 0) < b.GetValue(runs.schema, 0);
    };
    std::sort(t.begin(), t.end(), by_lid);
    std::sort(v.begin(), v.end(), by_lid);
    test::ExpectRowsEqual(t, v, runs.schema);
    EXPECT_EQ(t.size(), 20u);
}

// 4. Sort(Aggregate(Filter(Scan))) — Blocker 4 "one conversion per pipeline segment".
// The vectorized sub-pipeline (Scan -> Filter -> Aggregate) runs entirely in batch
// mode. A single BatchToTupleAdapter is inserted between Aggregate (vec) and Sort
// (tuple). No per-operator conversion.
TEST_F(VectorizedHybridPipelineParityTest, SortAggregateFilterPipeline) {
    auto env = SetUpSingleTable(/*num_rows=*/100);

    auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");

    auto pred = MakeBinOp(BinaryOp::Op::GT, MakeColRef("id"), MakeLiteral(49));
    auto filter_plan = std::make_unique<FilterPlanNode>(env.schema, std::move(pred));
    filter_plan->children.push_back(std::move(scan_plan));

    Schema agg_schema({
        Column("id",      TypeId::INTEGER),
        Column("sum_val", TypeId::DECIMAL),
    });

    auto agg_plan = std::make_unique<AggregatePlanNode>(agg_schema);
    agg_plan->group_by_exprs.push_back(MakeColRef("id"));
    agg_plan->aggregate_exprs.push_back(MakeColRef("val"));
    agg_plan->aggregate_funcs.push_back(Aggregate::Func::SUM);
    agg_plan->children.push_back(std::move(filter_plan));

    std::vector<OrderByItem> order_by;
    order_by.push_back({MakeColRef("id"), /*ascending=*/true});
    auto sort_plan = std::make_unique<SortPlanNode>(agg_schema, std::move(order_by));
    sort_plan->children.push_back(std::move(agg_plan));

    auto runs = test::RunBothModes(sort_plan.get(), &env.ctx);
    test::ExpectRowsEqual(runs.tuple_mode, runs.vectorized_mode, runs.schema);
    EXPECT_EQ(runs.tuple_mode.size(), 50u);  // ids 50..99 survive the filter
}

}  // namespace shilmandb
