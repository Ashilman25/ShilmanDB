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

// A seq scan of table that lists only cols, or every column when unset.
std::unique_ptr<SeqScanPlanNode> ListedScan(Schema schema, const std::string& table, std::optional<Cols> cols) {
    auto scan = std::make_unique<SeqScanPlanNode>(std::move(schema), table);
    scan->read_columns = std::move(cols);
    return scan;
}

TEST(UnlistedReadsTest, FilterOnUnlistedColumnIsReported) {
    auto scan = ListedScan(TestSchema(), "t", Cols{0});
    auto* scan_ptr = scan.get();
    FilterPlanNode filter(TestSchema(), MakeBinOp(BinaryOp::Op::GT, MakeColRef("b", 1), MakeLiteral(5)));
    filter.children.push_back(std::move(scan));

    const auto misses = FindUnlistedReads(filter);
    ASSERT_EQ(misses.size(), 1u);
    EXPECT_EQ(misses[0].reader, &filter);
    EXPECT_EQ(misses[0].index, 1u);
    EXPECT_EQ(misses[0].scan, scan_ptr);
    EXPECT_EQ(misses[0].scan_column, 1u);

    scan_ptr->read_columns = Cols{0, 1};
    EXPECT_TRUE(FindUnlistedReads(filter).empty());

    // Unset means every column.
    scan_ptr->read_columns.reset();
    EXPECT_TRUE(FindUnlistedReads(filter).empty());

    // An empty list reads nothing, so reading b is still a miss.
    scan_ptr->read_columns = Cols{};
    EXPECT_EQ(FindUnlistedReads(filter).size(), 1u);
}

TEST(UnlistedReadsTest, UnknownNameIsNotReported) {
    FilterPlanNode filter(TestSchema(), MakeBinOp(BinaryOp::Op::GT, MakeColRef("zzz"), MakeLiteral(5)));
    filter.children.push_back(ListedScan(TestSchema(), "t", Cols{0}));
    EXPECT_TRUE(FindUnlistedReads(filter).empty());
}

TEST(UnlistedReadsTest, PassesThroughSortAndLimitButNotProjection) {
    // Limit, sort, filter, scan: the sort key d isn't in the scan's list.
    std::vector<OrderByItem> items;
    items.push_back(OrderByItem{MakeColRef("d", 3), true});
    auto sort = std::make_unique<SortPlanNode>(TestSchema(), std::move(items));
    auto* sort_ptr = sort.get();
    auto filter = std::make_unique<FilterPlanNode>(TestSchema(), MakeBinOp(BinaryOp::Op::GT, MakeColRef("a", 0), MakeLiteral(5)));
    filter->children.push_back(ListedScan(TestSchema(), "t", Cols{0}));
    sort->children.push_back(std::move(filter));
    LimitPlanNode limit(TestSchema(), 10);
    limit.children.push_back(std::move(sort));

    const auto misses = FindUnlistedReads(limit);
    ASSERT_EQ(misses.size(), 1u);
    EXPECT_EQ(misses[0].reader, sort_ptr);
    EXPECT_EQ(misses[0].scan_column, 3u);

    // A projection's output is derived: a sort over it reads no scan column.
    std::vector<std::unique_ptr<Expression>> exprs;
    exprs.push_back(MakeColRef("a", 0));
    Schema one({Column("a", TypeId::INTEGER), Column("b", TypeId::DECIMAL)});
    auto proj = std::make_unique<ProjectionPlanNode>(one, std::move(exprs));
    proj->children.push_back(ListedScan(TestSchema(), "t", Cols{0}));
    std::vector<OrderByItem> over_proj;
    over_proj.push_back(OrderByItem{MakeColRef("b", 1), true});
    SortPlanNode derived(one, std::move(over_proj));
    derived.children.push_back(std::move(proj));
    EXPECT_TRUE(FindUnlistedReads(derived).empty());
}

TEST(UnlistedReadsTest, JoinRightSideReportedAtItsOffset) {
    // Left is TestSchema (4 columns, all read); right is (x, y) listing only x.
    Schema right_schema({Column("x", TypeId::INTEGER), Column("y", TypeId::INTEGER)});
    Schema joined({Column("a", TypeId::INTEGER), Column("b", TypeId::DECIMAL), Column("c", TypeId::VARCHAR),
                   Column("d", TypeId::DATE), Column("x", TypeId::INTEGER), Column("y", TypeId::INTEGER)});
    auto right = ListedScan(right_schema, "r", Cols{0});
    auto* right_ptr = right.get();
    auto join = std::make_unique<HashJoinPlanNode>(
        joined, MakeBinOp(BinaryOp::Op::EQ, MakeColRef("a", 0), MakeColRef("y", 5)));
    auto* join_ptr = join.get();
    join->children.push_back(ListedScan(TestSchema(), "l", std::nullopt));
    join->children.push_back(std::move(right));

    // A filter above the join reads x, which the right side lists.
    FilterPlanNode filter(joined, MakeBinOp(BinaryOp::Op::GT, MakeColRef("x", 4), MakeLiteral(1)));
    filter.children.push_back(std::move(join));

    const auto misses = FindUnlistedReads(filter);
    ASSERT_EQ(misses.size(), 1u);
    EXPECT_EQ(misses[0].reader, join_ptr);
    EXPECT_EQ(misses[0].index, 5u);
    EXPECT_EQ(misses[0].scan, right_ptr);
    EXPECT_EQ(misses[0].scan_column, 1u);

    right_ptr->read_columns = Cols{0, 1};
    EXPECT_TRUE(FindUnlistedReads(filter).empty());
}

}  // namespace shilmandb
