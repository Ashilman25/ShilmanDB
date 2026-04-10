#include <gtest/gtest.h>
#include "executor/filter_executor.hpp"
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

class FilterExecutorTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() / "shilmandb_filter_test.db").string();
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

    TestEnv SetUpTable(int num_rows) {
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

    // Build a SeqScan → Filter pipeline and count/collect results
    int RunFilter(TestEnv& env, std::unique_ptr<Expression> predicate) {
        auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");
        auto filter_plan = std::make_unique<FilterPlanNode>(env.schema, std::move(predicate));

        auto child = std::make_unique<SeqScanExecutor>(scan_plan.get(), &env.ctx);
        FilterExecutor filter(filter_plan.get(), &env.ctx, std::move(child));
        filter.Init();

        int count = 0;
        Tuple result;
        while (filter.Next(&result)) { ++count; }
        filter.Close();
        return count;
    }
};

// --- Tests ---

TEST_F(FilterExecutorTest, GreaterThanFilter) {
    auto env = SetUpTable(100);

    // id > 50  → rows 51..99 = 49 results
    auto pred = MakeBinOp(BinaryOp::Op::GT, MakeColRef("id"), MakeLiteral(50));
    EXPECT_EQ(RunFilter(env, std::move(pred)), 49);
}

TEST_F(FilterExecutorTest, FilterMatchesNothing) {
    auto env = SetUpTable(100);

    // id > 999 → 0 results
    auto pred = MakeBinOp(BinaryOp::Op::GT, MakeColRef("id"), MakeLiteral(999));
    EXPECT_EQ(RunFilter(env, std::move(pred)), 0);
}

TEST_F(FilterExecutorTest, CompoundAndPredicate) {
    auto env = SetUpTable(100);

    // id > 20 AND id < 40  → rows 21..39 = 19 results
    auto left = MakeBinOp(BinaryOp::Op::GT, MakeColRef("id"), MakeLiteral(20));
    auto right = MakeBinOp(BinaryOp::Op::LT, MakeColRef("id"), MakeLiteral(40));
    auto pred = MakeBinOp(BinaryOp::Op::AND, std::move(left), std::move(right));
    EXPECT_EQ(RunFilter(env, std::move(pred)), 19);
}

TEST_F(FilterExecutorTest, EqualityFilter) {
    auto env = SetUpTable(100);

    // id == 42 → 1 result
    auto pred = MakeBinOp(BinaryOp::Op::EQ, MakeColRef("id"), MakeLiteral(42));

    auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");
    auto filter_plan = std::make_unique<FilterPlanNode>(env.schema, std::move(pred));

    auto child = std::make_unique<SeqScanExecutor>(scan_plan.get(), &env.ctx);
    FilterExecutor filter(filter_plan.get(), &env.ctx, std::move(child));
    filter.Init();

    Tuple result;
    ASSERT_TRUE(filter.Next(&result));
    EXPECT_EQ(result.GetValue(env.schema, 0).integer_, 42);
    EXPECT_EQ(result.GetValue(env.schema, 1).integer_, 420);
    EXPECT_FALSE(filter.Next(&result));
    filter.Close();
}

TEST_F(FilterExecutorTest, OrPredicate) {
    auto env = SetUpTable(100);

    // id < 10 OR id > 90  → rows 0..9 + 91..99 = 19 results
    auto left = MakeBinOp(BinaryOp::Op::LT, MakeColRef("id"), MakeLiteral(10));
    auto right = MakeBinOp(BinaryOp::Op::GT, MakeColRef("id"), MakeLiteral(90));
    auto pred = MakeBinOp(BinaryOp::Op::OR, std::move(left), std::move(right));
    EXPECT_EQ(RunFilter(env, std::move(pred)), 19);
}

TEST_F(FilterExecutorTest, EmptyTableReturnsNothing) {
    auto env = SetUpTable(0);

    auto pred = MakeBinOp(BinaryOp::Op::GT, MakeColRef("id"), MakeLiteral(0));
    EXPECT_EQ(RunFilter(env, std::move(pred)), 0);
}

// --- Expression evaluator coverage tests ---

TEST_F(FilterExecutorTest, NotUnaryOp) {
    auto env = SetUpTable(100);

    // NOT (id > 50) → rows 0..50 = 51 results
    auto inner = MakeBinOp(BinaryOp::Op::GT, MakeColRef("id"), MakeLiteral(50));
    auto not_op = std::make_unique<UnaryOp>();
    not_op->op = UnaryOp::Op::NOT;
    not_op->operand = std::move(inner);

    EXPECT_EQ(RunFilter(env, std::move(not_op)), 51);
}

TEST_F(FilterExecutorTest, NegateUnaryOp) {
    auto env = SetUpTable(100);

    // -val == -420  (val = id * 10, so id=42 has val=420, -val=-420)
    auto negate = std::make_unique<UnaryOp>();
    negate->op = UnaryOp::Op::NEGATE;
    negate->operand = MakeColRef("val");

    auto lit = std::make_unique<Literal>();
    lit->value = Value(static_cast<int32_t>(-420));

    auto pred = MakeBinOp(BinaryOp::Op::EQ, std::move(negate), std::move(lit));
    EXPECT_EQ(RunFilter(env, std::move(pred)), 1);
}

TEST_F(FilterExecutorTest, ArithmeticAdd) {
    auto env = SetUpTable(100);

    // id + val == 55  (val = id*10, so id + id*10 = 11*id = 55 → id=5)
    auto add = MakeBinOp(BinaryOp::Op::ADD, MakeColRef("id"), MakeColRef("val"));
    auto pred = MakeBinOp(BinaryOp::Op::EQ, std::move(add), MakeLiteral(55));
    EXPECT_EQ(RunFilter(env, std::move(pred)), 1);
}

TEST_F(FilterExecutorTest, ArithmeticMul) {
    auto env = SetUpTable(100);

    // id * 10 == val  → all 100 rows match (val = id * 10)
    auto mul = MakeBinOp(BinaryOp::Op::MUL, MakeColRef("id"), MakeLiteral(10));
    auto pred = MakeBinOp(BinaryOp::Op::EQ, std::move(mul), MakeColRef("val"));
    EXPECT_EQ(RunFilter(env, std::move(pred)), 100);
}

TEST_F(FilterExecutorTest, ArithmeticSubAndDiv) {
    auto env = SetUpTable(100);

    // (val - id) / 9 == id  → (id*10 - id)/9 = 9*id/9 = id, all 100 match
    auto sub = MakeBinOp(BinaryOp::Op::SUB, MakeColRef("val"), MakeColRef("id"));
    auto div = MakeBinOp(BinaryOp::Op::DIV, std::move(sub), MakeLiteral(9));
    auto pred = MakeBinOp(BinaryOp::Op::EQ, std::move(div), MakeColRef("id"));
    EXPECT_EQ(RunFilter(env, std::move(pred)), 100);
}

}  // namespace shilmandb
