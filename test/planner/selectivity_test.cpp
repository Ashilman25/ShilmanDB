// test/planner/selectivity_test.cpp
#include <gtest/gtest.h>
#include "catalog/table_stats.hpp"
#include "parser/ast.hpp"
#include "planner/cost_model.hpp"
#include "types/value.hpp"
#include <memory>
#include <optional>
#include <string>

namespace shilmandb {
namespace {

using Op = BinaryOp::Op;
using ExprPtr = std::unique_ptr<Expression>;

ExprPtr Col(const std::string& name, std::optional<std::string> table = std::nullopt) {
    auto e = std::make_unique<ColumnRef>();
    e->column_name = name;
    e->table_name = std::move(table);
    return e;
}

ExprPtr LitOf(Value v) {
    auto e = std::make_unique<Literal>();
    e->value = std::move(v);
    return e;
}

ExprPtr Int(int32_t v) { return LitOf(Value(v)); }
ExprPtr Dec(double v) { return LitOf(Value(v)); }
ExprPtr Str(const std::string& v) { return LitOf(Value(v)); }
ExprPtr Date(const std::string& iso) { return LitOf(Value::FromString(TypeId::DATE, iso)); }

ExprPtr Bin(Op op, ExprPtr l, ExprPtr r) {
    auto e = std::make_unique<BinaryOp>();
    e->op = op;
    e->left = std::move(l);
    e->right = std::move(r);
    return e;
}

TableStats Stats(uint64_t rows = 1000) {
    TableStats s;
    s.row_count = rows;
    return s;
}

void SetRange(TableStats& s, const std::string& col, Value lo, Value hi) {
    s.min_max[col] = MinMax{std::move(lo), std::move(hi)};
}

double Sel(const TableStats& s, std::initializer_list<const Expression*> preds, std::string_view only = {}) {
    return CostModel::EstimateSelectivity(std::vector<const Expression*>(preds), s, only);
}

TableStats ShipdateStats() {
    TableStats s = Stats(6000);
    SetRange(s, "l_shipdate", Value::FromString(TypeId::DATE, "1992-01-04"), Value::FromString(TypeId::DATE, "1998-11-29"));
    return s;
}

TableStats XStats(int32_t lo, int32_t hi) {
    TableStats s = Stats();
    SetRange(s, "x", Value(lo), Value(hi));
    return s;
}

}  // namespace

TEST(SelectivityTest, Q6ShipdateYearFromVarcharLiterals) {
    const TableStats s = ShipdateStats();
    auto a = Bin(Op::GTE, Col("l_shipdate"), Str("1994-01-01"));
    auto b = Bin(Op::LT, Col("l_shipdate"), Str("1995-01-01"));
    EXPECT_NEAR(Sel(s, {a.get(), b.get()}), 365.0 / 2522.0, 1e-9);
}

TEST(SelectivityTest, Q6ShipdateYearFromDateLiterals) {
    const TableStats s = ShipdateStats();
    auto a = Bin(Op::GTE, Col("l_shipdate"), Date("1994-01-01"));
    auto b = Bin(Op::LT, Col("l_shipdate"), Date("1995-01-01"));
    EXPECT_NEAR(Sel(s, {a.get(), b.get()}), 365.0 / 2522.0, 1e-9);
}

TEST(SelectivityTest, TwoSidedRangeIsOneInterval) {
    const TableStats s = XStats(0, 999);
    auto a = Bin(Op::GTE, Col("x"), Int(100));
    auto b = Bin(Op::LT, Col("x"), Int(200));
    EXPECT_NEAR(Sel(s, {a.get(), b.get()}), 0.1, 1e-9);
}

TEST(SelectivityTest, BetweenMatchesTwoConjuncts) {
    const TableStats s = XStats(0, 999);
    auto between = Bin(Op::AND, Bin(Op::GTE, Col("x"), Int(100)), Bin(Op::LTE, Col("x"), Int(199)));
    EXPECT_NEAR(Sel(s, {between.get()}), 0.1, 1e-9);
}

TEST(SelectivityTest, StrictBoundsOnWholeNumbers) {
    const TableStats s = XStats(0, 99);
    auto a = Bin(Op::GT, Col("x"), Int(5));
    auto b = Bin(Op::LT, Col("x"), Int(7));
    EXPECT_NEAR(Sel(s, {a.get(), b.get()}), 0.01, 1e-9);  // only x = 6
}

TEST(SelectivityTest, LiteralOnTheLeft) {
    const TableStats s = XStats(0, 999);
    auto a = Bin(Op::LTE, Int(100), Col("x"));  // x >= 100
    auto b = Bin(Op::GT, Int(200), Col("x"));   // x < 200
    EXPECT_NEAR(Sel(s, {a.get(), b.get()}), 0.1, 1e-9);
}

TEST(SelectivityTest, QualifiedColumnUsesBareName) {
    const TableStats s = XStats(0, 999);
    auto a = Bin(Op::GTE, Col("x", "o"), Int(100));
    auto b = Bin(Op::LT, Col("x", "o"), Int(200));
    EXPECT_NEAR(Sel(s, {a.get(), b.get()}), 0.1, 1e-9);
}

TEST(SelectivityTest, DecimalIsContinuous) {
    TableStats s = Stats();
    SetRange(s, "d", Value(0.0), Value(0.10));
    auto a = Bin(Op::GTE, Col("d"), Dec(0.05));
    auto b = Bin(Op::LTE, Col("d"), Dec(0.07));
    EXPECT_NEAR(Sel(s, {a.get(), b.get()}), 0.2, 1e-9);
}

TEST(SelectivityTest, IntegerLiteralOnDecimalColumn) {
    TableStats s = Stats();
    SetRange(s, "q", Value(1.0), Value(50.0));
    auto a = Bin(Op::LT, Col("q"), Int(24));
    EXPECT_NEAR(Sel(s, {a.get()}), 23.0 / 49.0, 1e-9);
}

TEST(SelectivityTest, IntegerLiteralOnBigintColumn) {
    TableStats s = Stats();
    SetRange(s, "k", Value(int64_t{1}), Value(int64_t{1000}));
    auto a = Bin(Op::LTE, Col("k"), Int(100));
    EXPECT_NEAR(Sel(s, {a.get()}), 0.1, 1e-9);
}

TEST(SelectivityTest, EmptyIntervalHitsFloor) {
    const TableStats s = XStats(0, 999);
    auto a = Bin(Op::GT, Col("x"), Int(10));
    auto b = Bin(Op::LT, Col("x"), Int(5));
    EXPECT_DOUBLE_EQ(Sel(s, {a.get(), b.get()}), CostModel::kMinSelectivity);
}

TEST(SelectivityTest, OutOfRangeHitsFloor) {
    const TableStats s = XStats(0, 999);
    auto a = Bin(Op::GT, Col("x"), Int(5000));
    EXPECT_DOUBLE_EQ(Sel(s, {a.get()}), CostModel::kMinSelectivity);
}

TEST(SelectivityTest, MinEqualsMaxWholeNumber) {
    const TableStats s = XStats(5, 5);
    auto ge = Bin(Op::GTE, Col("x"), Int(5));
    auto gt = Bin(Op::GT, Col("x"), Int(5));
    EXPECT_DOUBLE_EQ(Sel(s, {ge.get()}), 1.0);
    EXPECT_DOUBLE_EQ(Sel(s, {gt.get()}), CostModel::kMinSelectivity);
}

TEST(SelectivityTest, MinEqualsMaxDecimal) {
    TableStats s = Stats();
    SetRange(s, "d", Value(2.5), Value(2.5));
    auto inside = Bin(Op::LTE, Col("d"), Dec(3.0));
    auto outside = Bin(Op::LT, Col("d"), Dec(2.0));
    EXPECT_DOUBLE_EQ(Sel(s, {inside.get()}), 1.0);
    EXPECT_DOUBLE_EQ(Sel(s, {outside.get()}), CostModel::kMinSelectivity);
}

TEST(SelectivityTest, NoMinMaxFallsBackToOpenRange) {
    const TableStats s = Stats();
    auto a = Bin(Op::GTE, Col("x"), Int(1));
    auto b = Bin(Op::LT, Col("x"), Int(5));
    EXPECT_NEAR(Sel(s, {a.get(), b.get()}), 0.33 * 0.33, 1e-9);
}

TEST(SelectivityTest, BadDateFallsBackToOpenRange) {
    const TableStats s = ShipdateStats();
    auto a = Bin(Op::GTE, Col("l_shipdate"), Str("not-a-date"));
    EXPECT_DOUBLE_EQ(Sel(s, {a.get()}), CostModel::kOpenRangeSelectivity);
}

TEST(SelectivityTest, EqualityUsesDistinct) {
    TableStats s = Stats();
    s.distinct_counts["x"] = 100;
    auto a = Bin(Op::EQ, Col("x"), Int(5));
    EXPECT_NEAR(Sel(s, {a.get()}), 0.01, 1e-9);
}

TEST(SelectivityTest, OrMergesInsideEachBranch) {
    const TableStats s = XStats(0, 99);
    auto low = Bin(Op::AND, Bin(Op::GTE, Col("x"), Int(0)), Bin(Op::LTE, Col("x"), Int(9)));
    auto high = Bin(Op::AND, Bin(Op::GTE, Col("x"), Int(90)), Bin(Op::LTE, Col("x"), Int(99)));
    auto either = Bin(Op::OR, std::move(low), std::move(high));
    EXPECT_NEAR(Sel(s, {either.get()}), 0.19, 1e-9);
}

TEST(SelectivityTest, OnlyColumnIgnoresOthers) {
    TableStats s = XStats(0, 999);
    s.distinct_counts["y"] = 10;
    auto a = Bin(Op::GTE, Col("x"), Int(0));
    auto b = Bin(Op::LT, Col("x"), Int(100));
    auto c = Bin(Op::EQ, Col("y"), Int(3));
    EXPECT_NEAR(Sel(s, {a.get(), b.get(), c.get()}, "x"), 0.1, 1e-9);
    EXPECT_NEAR(Sel(s, {a.get(), b.get(), c.get()}), 0.01, 1e-9);
}

TEST(SelectivityTest, ColumnVsColumnInequality) {
    const TableStats s = Stats();
    auto a = Bin(Op::LT, Col("a"), Col("b"));
    EXPECT_DOUBLE_EQ(Sel(s, {a.get()}), CostModel::kOpenRangeSelectivity);
}

TEST(SelectivityTest, FractionalLiteralOnWholeNumberColumn) {
    const TableStats s = XStats(0, 99);
    auto lt = Bin(Op::LT, Col("x"), Dec(5.5));
    auto gt = Bin(Op::GT, Col("x"), Dec(5.5));
    EXPECT_NEAR(Sel(s, {lt.get()}), 0.06, 1e-9);  // x = 0..5
    EXPECT_NEAR(Sel(s, {gt.get()}), 0.94, 1e-9);  // x = 6..99
}

TEST(SelectivityTest, DateColumnVsNumberLiteral) {
    const TableStats s = ShipdateStats();
    auto a = Bin(Op::GTE, Col("l_shipdate"), Int(5));
    EXPECT_DOUBLE_EQ(Sel(s, {a.get()}), CostModel::kOpenRangeSelectivity);
}

TEST(SelectivityTest, NumberColumnVsDateLiteral) {
    const TableStats s = XStats(0, 999);
    auto a = Bin(Op::GTE, Col("x"), Date("1994-01-01"));
    EXPECT_DOUBLE_EQ(Sel(s, {a.get()}), CostModel::kOpenRangeSelectivity);
}

TEST(SelectivityTest, OtherPredicatesUseDefault) {
    TableStats s = XStats(0, 999);
    s.distinct_counts["x"] = 100;
    auto ne = Bin(Op::NEQ, Col("x"), Int(5));
    auto like = Bin(Op::LIKE, Col("name"), Str("A%"));
    auto negated = std::make_unique<UnaryOp>();
    negated->op = UnaryOp::Op::NOT;
    negated->operand = Bin(Op::EQ, Col("x"), Int(5));
    EXPECT_DOUBLE_EQ(Sel(s, {ne.get()}), CostModel::kDefaultSelectivity);
    EXPECT_DOUBLE_EQ(Sel(s, {like.get()}), CostModel::kDefaultSelectivity);
    EXPECT_DOUBLE_EQ(Sel(s, {negated.get()}), CostModel::kDefaultSelectivity);
}

TEST(SelectivityTest, OrWithOnlyColumn) {
    TableStats s = XStats(0, 999);
    SetRange(s, "y", Value(0), Value(999));
    auto either = Bin(Op::OR, Bin(Op::LT, Col("x"), Int(100)), Bin(Op::LT, Col("y"), Int(100)));
    EXPECT_DOUBLE_EQ(Sel(s, {either.get()}, "x"), 1.0);
    EXPECT_DOUBLE_EQ(Sel(s, {either.get()}, "z"), 1.0);
}

TEST(SelectivityTest, FilteredRowsScalesAndFloors) {
    const TableStats s = XStats(0, 999);
    auto lt = Bin(Op::LT, Col("x"), Int(100));
    EXPECT_NEAR(CostModel::EstimateFilteredRows({lt.get()}, s), 100.0, 1e-9);

    TableStats small = XStats(0, 999);
    small.row_count = 10;
    auto a = Bin(Op::GT, Col("x"), Int(10));
    auto b = Bin(Op::LT, Col("x"), Int(5));
    EXPECT_DOUBLE_EQ(CostModel::EstimateFilteredRows({a.get(), b.get()}, small), 1.0);

    TableStats empty = XStats(0, 999);
    empty.row_count = 0;
    EXPECT_DOUBLE_EQ(CostModel::EstimateFilteredRows({lt.get()}, empty), 0.0);

    EXPECT_DOUBLE_EQ(CostModel::EstimateFilteredRows({}, s), 1000.0);
}

}  // namespace shilmandb
