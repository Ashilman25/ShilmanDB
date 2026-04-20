#include <gtest/gtest.h>
#include "executor/vectorized_filter_executor.hpp"
#include "executor/vectorized_seq_scan_executor.hpp"
#include "planner/plan_node.hpp"
#include "catalog/catalog.hpp"
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
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

class VectorizedFilterExecutorTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() /
                      "shilmandb_vec_filter_test.db").string();
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
        std::unique_ptr<SeqScanPlanNode> scan_plan;
        std::unique_ptr<FilterPlanNode> filter_plan;
        ExecutorContext ctx;
        Schema schema;
    };

    // Creates table "t" with `num_rows` rows: (i, i*10).
    // Filter predicate = `val <op> literal`.
    TestEnv MakeEnv(int num_rows, BinaryOp::Op op, int32_t literal) {
        auto bundle = MakeBPM(test_file_);
        auto catalog = std::make_unique<Catalog>(bundle.bpm.get());
        auto schema = MakeSchema();
        auto* table_info = catalog->CreateTable("t", schema);
        for (int i = 0; i < num_rows; ++i) {
            std::vector<Value> vals = {
                Value(static_cast<int32_t>(i)),
                Value(static_cast<int32_t>(i * 10)),
            };
            (void)table_info->table->InsertTuple(Tuple(vals, schema));
        }
        auto scan_plan = std::make_unique<SeqScanPlanNode>(schema, "t");
        auto pred = MakeBinOp(op, MakeColRef("val"), MakeLiteral(literal));
        auto filter_plan = std::make_unique<FilterPlanNode>(schema, std::move(pred));
        ExecutorContext ctx{bundle.bpm.get(), catalog.get()};
        return {std::move(bundle), std::move(catalog),
                std::move(scan_plan), std::move(filter_plan),
                ctx, schema};
    }

    // Build a Filter above a VectorizedSeqScan. Both children are vectorized.
    static std::unique_ptr<VectorizedFilterExecutor> BuildPipeline(TestEnv& env) {
        auto scan = std::make_unique<VectorizedSeqScanExecutor>(
            env.scan_plan.get(), &env.ctx);
        return std::make_unique<VectorizedFilterExecutor>(
            env.filter_plan.get(), &env.ctx, std::move(scan));
    }
};

// --- Core predicate tests ---

TEST_F(VectorizedFilterExecutorTest, FilterAllTrue) {
    auto env = MakeEnv(100, BinaryOp::Op::GT, -1);  // val > -1 matches all
    auto exec = BuildPipeline(env);
    exec->Init();

    int count = 0;
    Tuple t;
    while (exec->Next(&t)) {
        EXPECT_EQ(t.GetValue(env.schema, 0).integer_, count);
        ++count;
    }
    EXPECT_EQ(count, 100);
    exec->Close();
}

TEST_F(VectorizedFilterExecutorTest, FilterAllFalse) {
    auto env = MakeEnv(100, BinaryOp::Op::LT, -1);  // val < -1 matches none
    auto exec = BuildPipeline(env);
    exec->Init();

    Tuple t;
    EXPECT_FALSE(exec->Next(&t));

    DataChunk chunk(env.schema);
    EXPECT_FALSE(exec->NextBatch(&chunk));
    exec->Close();
}

TEST_F(VectorizedFilterExecutorTest, FilterHalf) {
    // val >= 5000 → rows with id >= 500 survive (val = id*10).
    auto env = MakeEnv(1000, BinaryOp::Op::GTE, 5000);
    auto exec = BuildPipeline(env);
    exec->Init();

    int count = 0;
    int expected_id = 500;
    Tuple t;
    while (exec->Next(&t)) {
        EXPECT_EQ(t.GetValue(env.schema, 0).integer_, expected_id);
        ++expected_id;
        ++count;
    }
    EXPECT_EQ(count, 500);
    exec->Close();
}

TEST_F(VectorizedFilterExecutorTest, SelectionVectorIsPhysicalIndex) {
    // val >= 100 → id >= 10 survives in a 20-row table.
    auto env = MakeEnv(20, BinaryOp::Op::GTE, 100);
    auto exec = BuildPipeline(env);
    exec->Init();

    DataChunk chunk(env.schema);
    ASSERT_TRUE(exec->NextBatch(&chunk));
    ASSERT_TRUE(chunk.HasSelectionVector());
    const auto& sel = chunk.GetSelectionVector();
    ASSERT_EQ(sel.size(), 10u);
    for (size_t i = 0; i < sel.size(); ++i) {
        // sel[i] is a physical index whose `val` column satisfies val >= 100.
        EXPECT_EQ(chunk.GetValue(0, sel[i]).integer_, static_cast<int32_t>(10 + i));
        EXPECT_GE(chunk.GetValue(1, sel[i]).integer_, 100);
    }
    exec->Close();
}

TEST_F(VectorizedFilterExecutorTest, NextDrainsFilteredRows) {
    auto env = MakeEnv(50, BinaryOp::Op::GTE, 200);  // id >= 20 → 30 rows
    auto exec = BuildPipeline(env);
    exec->Init();

    int count = 0;
    int expected_id = 20;
    Tuple t;
    while (exec->Next(&t)) {
        EXPECT_EQ(t.GetValue(env.schema, 0).integer_, expected_id);
        ++expected_id;
        ++count;
    }
    EXPECT_EQ(count, 30);
    exec->Close();
}

TEST_F(VectorizedFilterExecutorTest, AllBatchesFilteredOutRetriesUntilEnd) {
    // 3000 rows span 3 full batches. Predicate matches none. NextBatch must
    // pull all 3 batches internally and return false once, not spuriously.
    auto env = MakeEnv(3000, BinaryOp::Op::LT, -1);
    auto exec = BuildPipeline(env);
    exec->Init();

    DataChunk chunk(env.schema);
    EXPECT_FALSE(exec->NextBatch(&chunk));
    exec->Close();
}

// --- Composability guard: nested selection vectors ---

namespace {

// Mock executor that returns a single pre-built chunk once, then false.
// Used to inject a chunk that already carries a selection vector into
// VectorizedFilterExecutor, exercising its sel-vec-aware branch directly.
class SingleChunkMockExecutor : public Executor {
public:
    SingleChunkMockExecutor(const PlanNode* plan, DataChunk chunk)
        : Executor(plan, nullptr), chunk_(std::move(chunk)) {}

    void Init() override {}
    bool Next(Tuple*) override { return false; }  // unused in this test
    bool NextBatch(DataChunk* out) override {
        if (emitted_) return false;
        emitted_ = true;
        // Copy physical rows directly from chunk_ to out (bypassing
        // MaterializeTuple so a selection vector on chunk_ does not
        // rewrite physical indices). Then graft the sel vec onto out.
        out->Reset();
        const auto n_physical = chunk_.GetColumn(0).size();
        const Schema& s = chunk_.GetSchema();
        const auto n_cols = s.GetColumnCount();
        for (size_t i = 0; i < n_physical; ++i) {
            std::vector<Value> row_vals;
            row_vals.reserve(n_cols);
            for (uint32_t c = 0; c < n_cols; ++c) {
                row_vals.push_back(chunk_.GetColumn(c)[i]);
            }
            out->AppendTuple(Tuple(row_vals, s));
        }
        if (chunk_.HasSelectionVector()) {
            out->SetSelectionVector(chunk_.GetSelectionVector());
        }
        return true;
    }
    void Close() override {}

private:
    DataChunk chunk_;
    bool emitted_{false};
};

}  // namespace

TEST_F(VectorizedFilterExecutorTest, NestedSelectionVectorPreservation) {
    // Build a chunk with 5 physical rows and sel vec = [1, 3, 4].
    // The outer filter accepts rows where val > 25; apply against physical
    // rows at sel positions: val = id*10 = 10, 30, 40 → surviving: 30, 40 at
    // logical indices 1, 2 → physical indices 3, 4.
    // Expected emitted sel vec after filter: [3, 4] (physical), NOT [1, 2].
    Schema schema({Column("id", TypeId::INTEGER), Column("val", TypeId::INTEGER)});

    DataChunk source(schema, /*capacity=*/16);
    for (int i = 0; i < 5; ++i) {
        std::vector<Value> vals = {
            Value(static_cast<int32_t>(i)),
            Value(static_cast<int32_t>(i * 10)),
        };
        source.AppendTuple(Tuple(vals, schema));
    }
    source.SetSelectionVector({1u, 3u, 4u});  // physical rows 1, 3, 4

    // Filter plan: val > 25
    auto pred = MakeBinOp(BinaryOp::Op::GT, MakeColRef("val"), MakeLiteral(25));
    FilterPlanNode filter_plan(schema, std::move(pred));

    // Separate plan for the mock source (schema match).
    SeqScanPlanNode source_plan(schema, "t_unused");
    auto mock = std::make_unique<SingleChunkMockExecutor>(&source_plan, std::move(source));

    // Drive the Filter directly with the mock child. No ExecutorContext needed
    // because VectorizedFilterExecutor does not access ctx_ in Init/NextBatch.
    VectorizedFilterExecutor exec(&filter_plan, /*ctx=*/nullptr, std::move(mock));
    exec.Init();

    DataChunk out(schema, /*capacity=*/16);
    ASSERT_TRUE(exec.NextBatch(&out));
    ASSERT_TRUE(out.HasSelectionVector());
    const auto& sel = out.GetSelectionVector();
    ASSERT_EQ(sel.size(), 2u);
    EXPECT_EQ(sel[0], 3u);
    EXPECT_EQ(sel[1], 4u);
    // Double check the surviving rows are the physically correct ones.
    EXPECT_EQ(out.GetValue(1, sel[0]).integer_, 30);
    EXPECT_EQ(out.GetValue(1, sel[1]).integer_, 40);

    EXPECT_FALSE(exec.NextBatch(&out));
    exec.Close();
}

TEST_F(VectorizedFilterExecutorTest, ParityWithTupleFilter) {
    // val > 1234 → id > 123 survives from a 500-row table.
    auto env = MakeEnv(500, BinaryOp::Op::GT, 1234);
    // Wire filter over scan in the plan tree:
    env.filter_plan->children.push_back(std::move(env.scan_plan));
    auto runs = test::RunBothModes(env.filter_plan.get(), &env.ctx);
    test::ExpectRowsEqual(runs.tuple_mode, runs.vectorized_mode, runs.schema);
    EXPECT_EQ(runs.tuple_mode.size(), 500u - 124u);  // id >= 124 survives
}

TEST_F(VectorizedFilterExecutorTest, ParityChainedFilters) {
    // FilterPlan(val < 9000) above FilterPlan(val > 1000) above SeqScan.
    // Exercises the composability branch end-to-end via both modes.
    auto env = MakeEnv(1000, BinaryOp::Op::GT, 1000);  // outer predicate; inner built below

    // Build inner filter (val < 9000) above the scan.
    auto inner_pred = MakeBinOp(BinaryOp::Op::LT, MakeColRef("val"), MakeLiteral(9000));
    auto inner_filter = std::make_unique<FilterPlanNode>(env.schema, std::move(inner_pred));
    inner_filter->children.push_back(std::move(env.scan_plan));

    // env.filter_plan currently holds val > 1000; install inner_filter as its child.
    env.filter_plan->children.push_back(std::move(inner_filter));

    auto runs = test::RunBothModes(env.filter_plan.get(), &env.ctx);
    test::ExpectRowsEqual(runs.tuple_mode, runs.vectorized_mode, runs.schema);
    // Surviving: val > 1000 AND val < 9000 → id in (100, 900) exclusive,
    // i.e., id in [101, 899] → 799 rows.
    EXPECT_EQ(runs.tuple_mode.size(), 799u);
}

}  // namespace shilmandb
