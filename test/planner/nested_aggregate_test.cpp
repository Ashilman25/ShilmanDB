#include <gtest/gtest.h>
#include "planner/planner.hpp"
#include "parser/parser.hpp"
#include "catalog/catalog.hpp"
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "storage/disk_manager.hpp"
#include <filesystem>
#include <memory>

namespace shilmandb {

class NestedAggregateTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path()
                       / "shilmandb_nested_agg_test.db").string();
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

    // Walks children[0] recursively until an AggregatePlanNode is found.
    // Returns nullptr if not present.
    static const AggregatePlanNode* FindAggregate(const PlanNode* node) {
        while (node) {
            if (node->type == PlanNodeType::AGGREGATE) {
                return static_cast<const AggregatePlanNode*>(node);
            }
            if (node->children.empty()) return nullptr;
            node = node->children[0].get();
        }
        return nullptr;
    }
};

// -----------------------------------------------------------------------
// TopLevelAggregateUnchanged: SELECT SUM(x) FROM t still produces a single
// SUM agg named sum_x; projection is a bare ColumnRef(sum_x).
// Guards against regression for the most common pattern.
// -----------------------------------------------------------------------
TEST_F(NestedAggregateTest, TopLevelAggregateUnchanged) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());
    (void)catalog.CreateTable("t", Schema({
        Column("x", TypeId::INTEGER),
        Column("y", TypeId::INTEGER)
    }));

    Parser parser("SELECT SUM(x) FROM t");
    auto stmt = parser.Parse();

    Planner planner(&catalog);
    auto plan = planner.Plan(std::move(*stmt));
    ASSERT_NE(plan, nullptr);

    // Root is Projection; its first child contains the AggregatePlanNode.
    ASSERT_EQ(plan->type, PlanNodeType::PROJECTION);
    const auto* agg = FindAggregate(plan.get());
    ASSERT_NE(agg, nullptr);
    ASSERT_EQ(agg->aggregate_funcs.size(), 1u);
    EXPECT_EQ(agg->aggregate_funcs[0], Aggregate::Func::SUM);

    // Agg output schema has a single column named sum_x.
    ASSERT_EQ(agg->output_schema.GetColumnCount(), 1u);
    EXPECT_EQ(agg->output_schema.GetColumn(0).name, "sum_x");

    // Projection's single expression is a ColumnRef(sum_x).
    const auto* proj = static_cast<const ProjectionPlanNode*>(plan.get());
    ASSERT_EQ(proj->expressions.size(), 1u);
    const auto* cr = dynamic_cast<const ColumnRef*>(proj->expressions[0].get());
    ASSERT_NE(cr, nullptr);
    EXPECT_EQ(cr->column_name, "sum_x");
}

// -----------------------------------------------------------------------
// NestedAggregateInArithmetic: SELECT SUM(x) + SUM(y) FROM t
// Extracts 2 aggs named sum_x, sum_y. Projection is
// BinaryOp(ADD, ColumnRef(sum_x), ColumnRef(sum_y)).
// -----------------------------------------------------------------------
TEST_F(NestedAggregateTest, NestedAggregateInArithmetic) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());
    (void)catalog.CreateTable("t", Schema({
        Column("x", TypeId::INTEGER),
        Column("y", TypeId::INTEGER)
    }));

    Parser parser("SELECT SUM(x) + SUM(y) FROM t");
    auto stmt = parser.Parse();

    Planner planner(&catalog);
    auto plan = planner.Plan(std::move(*stmt));
    ASSERT_NE(plan, nullptr);

    const auto* agg = FindAggregate(plan.get());
    ASSERT_NE(agg, nullptr);
    ASSERT_EQ(agg->aggregate_funcs.size(), 2u);
    EXPECT_EQ(agg->aggregate_funcs[0], Aggregate::Func::SUM);
    EXPECT_EQ(agg->aggregate_funcs[1], Aggregate::Func::SUM);
    ASSERT_EQ(agg->output_schema.GetColumnCount(), 2u);
    EXPECT_EQ(agg->output_schema.GetColumn(0).name, "sum_x");
    EXPECT_EQ(agg->output_schema.GetColumn(1).name, "sum_y");

    // Projection has one expression: BinaryOp(ADD, ColumnRef(sum_x), ColumnRef(sum_y)).
    const auto* proj = static_cast<const ProjectionPlanNode*>(plan.get());
    ASSERT_EQ(proj->expressions.size(), 1u);
    const auto* add = dynamic_cast<const BinaryOp*>(proj->expressions[0].get());
    ASSERT_NE(add, nullptr);
    EXPECT_EQ(add->op, BinaryOp::Op::ADD);
    const auto* lhs = dynamic_cast<const ColumnRef*>(add->left.get());
    const auto* rhs = dynamic_cast<const ColumnRef*>(add->right.get());
    ASSERT_NE(lhs, nullptr);
    ASSERT_NE(rhs, nullptr);
    EXPECT_EQ(lhs->column_name, "sum_x");
    EXPECT_EQ(rhs->column_name, "sum_y");
}

// -----------------------------------------------------------------------
// Q14ShapeDivOfSums: SELECT 100.0 * SUM(CASE WHEN x > 0 THEN y ELSE 0 END)
//                         / SUM(y + 1)
//                     FROM t
// Both aggregates have non-ColumnRef args, so naming disambiguates:
// sum_expr_0 (the CASE) and sum_expr_1 (the y + 1 BinaryOp).
// Top projection: BinaryOp(DIV,
//                          BinaryOp(MUL, Literal(100.0), ColumnRef(sum_expr_0)),
//                          ColumnRef(sum_expr_1))
// -----------------------------------------------------------------------
TEST_F(NestedAggregateTest, Q14ShapeDivOfSums) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());
    (void)catalog.CreateTable("t", Schema({
        Column("x", TypeId::INTEGER),
        Column("y", TypeId::INTEGER)
    }));

    Parser parser(
        "SELECT 100.0 * SUM(CASE WHEN x > 0 THEN y ELSE 0 END) "
        "       / SUM(y + 1) FROM t");
    auto stmt = parser.Parse();

    Planner planner(&catalog);
    auto plan = planner.Plan(std::move(*stmt));
    ASSERT_NE(plan, nullptr);

    const auto* agg = FindAggregate(plan.get());
    ASSERT_NE(agg, nullptr);
    ASSERT_EQ(agg->aggregate_funcs.size(), 2u);
    EXPECT_EQ(agg->aggregate_funcs[0], Aggregate::Func::SUM);
    EXPECT_EQ(agg->aggregate_funcs[1], Aggregate::Func::SUM);
    ASSERT_EQ(agg->output_schema.GetColumnCount(), 2u);
    EXPECT_EQ(agg->output_schema.GetColumn(0).name, "sum_expr_0");
    EXPECT_EQ(agg->output_schema.GetColumn(1).name, "sum_expr_1");

    // Projection expression: DIV(MUL(Literal, CR(sum_expr_0)), CR(sum_expr_1))
    const auto* proj = static_cast<const ProjectionPlanNode*>(plan.get());
    ASSERT_EQ(proj->expressions.size(), 1u);
    const auto* div = dynamic_cast<const BinaryOp*>(proj->expressions[0].get());
    ASSERT_NE(div, nullptr);
    EXPECT_EQ(div->op, BinaryOp::Op::DIV);

    const auto* mul = dynamic_cast<const BinaryOp*>(div->left.get());
    ASSERT_NE(mul, nullptr);
    EXPECT_EQ(mul->op, BinaryOp::Op::MUL);
    const auto* lit = dynamic_cast<const Literal*>(mul->left.get());
    ASSERT_NE(lit, nullptr);
    const auto* cr0 = dynamic_cast<const ColumnRef*>(mul->right.get());
    ASSERT_NE(cr0, nullptr);
    EXPECT_EQ(cr0->column_name, "sum_expr_0");

    const auto* cr1 = dynamic_cast<const ColumnRef*>(div->right.get());
    ASSERT_NE(cr1, nullptr);
    EXPECT_EQ(cr1->column_name, "sum_expr_1");
}

}  // namespace shilmandb
