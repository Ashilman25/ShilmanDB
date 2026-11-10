#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "planner/column_usage.hpp"

namespace shilmandb {

namespace {

using Cols = std::vector<uint32_t>;

Schema TestSchema() {
    return Schema({Column("a", TypeId::INTEGER), Column("b", TypeId::DECIMAL),
                   Column("c", TypeId::VARCHAR), Column("d", TypeId::DATE)});
}

std::unique_ptr<Expression> MakeColRef(const std::string& name, std::optional<uint32_t> bound = std::nullopt) {
    auto c = std::make_unique<ColumnRef>();
    c->column_name = name;
    c->bound_index = bound;
    return c;
}

std::unique_ptr<Expression> MakeLiteral(int32_t v) {
    auto l = std::make_unique<Literal>();
    l->value = Value(v);
    return l;
}

std::unique_ptr<Expression> MakeBinOp(BinaryOp::Op op, std::unique_ptr<Expression> l,
                                      std::unique_ptr<Expression> r) {
    auto b = std::make_unique<BinaryOp>();
    b->op = op;
    b->left = std::move(l);
    b->right = std::move(r);
    return b;
}

std::unique_ptr<PlanNode> Scan() {
    return std::make_unique<SeqScanPlanNode>(TestSchema(), "t");
}

}  // namespace

TEST(ColumnUsageTest, BoundRefUsesBoundIndex) {
    Cols out;
    CollectColumns(MakeColRef("a", 2).get(), TestSchema(), out);
    EXPECT_EQ(out, Cols({2}));
}

TEST(ColumnUsageTest, UnboundRefUsesFirstNameMatch) {
    Cols out;
    CollectColumns(MakeColRef("c").get(), TestSchema(), out);
    EXPECT_EQ(out, Cols({2}));
}

TEST(ColumnUsageTest, MissingNameIsSkipped) {
    Cols out;
    EXPECT_NO_THROW(CollectColumns(MakeColRef("zzz").get(), TestSchema(), out));
    EXPECT_TRUE(out.empty());
}

TEST(ColumnUsageTest, NestedCaseAndArithmetic) {
    auto c = std::make_unique<CaseExpression>();
    c->when_clauses.emplace_back(
        MakeBinOp(BinaryOp::Op::GT, MakeColRef("d"), MakeLiteral(5)),
        MakeBinOp(BinaryOp::Op::MUL, MakeColRef("b"), MakeColRef("a")));
    c->else_clause = MakeLiteral(0);
    Cols out;
    CollectColumns(c.get(), TestSchema(), out);
    EXPECT_EQ(out, Cols({0, 1, 3}));
}

TEST(ColumnUsageTest, SortedAndUniqueAcrossCalls) {
    Cols out;
    auto sum = MakeBinOp(BinaryOp::Op::ADD, MakeColRef("b"), MakeColRef("a"));
    CollectColumns(sum.get(), TestSchema(), out);
    CollectColumns(MakeColRef("a").get(), TestSchema(), out);
    EXPECT_EQ(out, Cols({0, 1}));
}

TEST(ColumnUsageTest, ColumnsReadByFilter) {
    auto filter = std::make_unique<FilterPlanNode>(
        TestSchema(), MakeBinOp(BinaryOp::Op::GTE, MakeColRef("b", 1), MakeLiteral(5)));
    filter->children.push_back(Scan());
    EXPECT_EQ(ColumnsReadBy(*filter), Cols({1}));
}

TEST(ColumnUsageTest, ColumnsReadByAggregateSkipsCountStar) {
    auto agg = std::make_unique<AggregatePlanNode>(TestSchema());
    agg->group_by_exprs.push_back(MakeColRef("c"));
    agg->aggregate_exprs.push_back(nullptr);
    agg->aggregate_exprs.push_back(MakeColRef("b"));
    agg->children.push_back(Scan());
    EXPECT_EQ(ColumnsReadBy(*agg), Cols({1, 2}));
}

TEST(ColumnUsageTest, ColumnsReadByProjectionAndSort) {
    std::vector<std::unique_ptr<Expression>> exprs;
    exprs.push_back(MakeColRef("d"));
    exprs.push_back(MakeColRef("a"));
    auto proj = std::make_unique<ProjectionPlanNode>(TestSchema(), std::move(exprs));
    proj->children.push_back(Scan());
    EXPECT_EQ(ColumnsReadBy(*proj), Cols({0, 3}));

    std::vector<OrderByItem> items;
    items.push_back(OrderByItem{MakeColRef("c"), true});
    auto sort = std::make_unique<SortPlanNode>(TestSchema(), std::move(items));
    sort->children.push_back(Scan());
    EXPECT_EQ(ColumnsReadBy(*sort), Cols({2}));
}

TEST(ColumnUsageTest, ColumnsReadByJoinUsesOwnSchema) {
    Schema joined({Column("x", TypeId::INTEGER), Column("y", TypeId::INTEGER)});
    auto join = std::make_unique<HashJoinPlanNode>(
        joined, MakeBinOp(BinaryOp::Op::EQ, MakeColRef("x"), MakeColRef("y")));
    EXPECT_EQ(ColumnsReadBy(*join), Cols({0, 1}));
}

TEST(ColumnUsageTest, ColumnsReadByScanAndLimitEmpty) {
    EXPECT_TRUE(ColumnsReadBy(*Scan()).empty());
    LimitPlanNode limit(TestSchema(), 10);
    EXPECT_TRUE(ColumnsReadBy(limit).empty());
}

}  // namespace shilmandb
