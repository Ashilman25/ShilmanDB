#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "planner/cost_model.hpp"

namespace shilmandb {
namespace {

using Op = BinaryOp::Op;
using Ranges = std::unordered_map<std::string, MinMax>;

std::unique_ptr<Expression> Col(const std::string& name) {
    auto c = std::make_unique<ColumnRef>();
    c->column_name = name;
    return c;
}

std::unique_ptr<Expression> Lit(Value v) {
    auto l = std::make_unique<Literal>();
    l->value = std::move(v);
    return l;
}

std::unique_ptr<Expression> Bin(Op op, std::unique_ptr<Expression> l, std::unique_ptr<Expression> r) {
    auto b = std::make_unique<BinaryOp>();
    b->op = op;
    b->left = std::move(l);
    b->right = std::move(r);
    return b;
}

std::unique_ptr<Expression> Negate(std::unique_ptr<Expression> e) {
    auto u = std::make_unique<UnaryOp>();
    u->op = UnaryOp::Op::NEGATE;
    u->operand = std::move(e);
    return u;
}

// col op v, with an INTEGER literal.
std::unique_ptr<Expression> Cmp(const std::string& col, Op op, int32_t v) {
    return Bin(op, Col(col), Lit(Value(v)));
}

Value Date(const std::string& text) { return Value(text).CastTo(TypeId::DATE); }

// MayMatch over one expression.
bool May(const std::unique_ptr<Expression>& e, const Ranges& ranges) {
    return CostModel::MayMatch({e.get()}, ranges);
}

// x holds 10 through 20.
Ranges XRange() { return {{"x", MinMax{Value(int32_t{10}), Value(int32_t{20})}}}; }

TEST(MayMatchTest, EqualityOutsideRange) {
    EXPECT_FALSE(May(Cmp("x", Op::EQ, 5), XRange()));
    EXPECT_FALSE(May(Cmp("x", Op::EQ, 21), XRange()));
    EXPECT_TRUE(May(Cmp("x", Op::EQ, 15), XRange()));
    EXPECT_TRUE(May(Cmp("x", Op::EQ, 10), XRange()));
    EXPECT_TRUE(May(Cmp("x", Op::EQ, 20), XRange()));
}

TEST(MayMatchTest, Inequalities) {
    EXPECT_TRUE(May(Cmp("x", Op::GT, 19), XRange()));
    EXPECT_TRUE(May(Cmp("x", Op::GTE, 20), XRange()));
    EXPECT_FALSE(May(Cmp("x", Op::GT, 20), XRange()));
    EXPECT_FALSE(May(Cmp("x", Op::LT, 10), XRange()));
    EXPECT_TRUE(May(Cmp("x", Op::LTE, 10), XRange()));
    EXPECT_FALSE(May(Cmp("x", Op::GTE, 21), XRange()));
}

TEST(MayMatchTest, LiteralOnTheLeftIsMirrored) {
    EXPECT_FALSE(May(Bin(Op::LT, Lit(Value(int32_t{20})), Col("x")), XRange()));  // x > 20
    EXPECT_TRUE(May(Bin(Op::LTE, Lit(Value(int32_t{20})), Col("x")), XRange()));  // x >= 20
    EXPECT_FALSE(May(Bin(Op::GT, Lit(Value(int32_t{10})), Col("x")), XRange()));  // x < 10
}

TEST(MayMatchTest, BetweenOutsideRange) {
    // BETWEEN 21 AND 30 parses to x >= 21 AND x <= 30.
    const auto between = Bin(Op::AND, Cmp("x", Op::GTE, 21), Cmp("x", Op::LTE, 30));
    EXPECT_FALSE(May(between, XRange()));
    const auto inside = Bin(Op::AND, Cmp("x", Op::GTE, 18), Cmp("x", Op::LTE, 30));
    EXPECT_TRUE(May(inside, XRange()));
}

TEST(MayMatchTest, BoundsOnOneColumnMerge) {
    // Each bound alone may match; together they can't.
    const auto a = Cmp("x", Op::GT, 15);
    const auto b = Cmp("x", Op::LT, 12);
    EXPECT_TRUE(May(a, XRange()));
    EXPECT_TRUE(May(b, XRange()));
    EXPECT_FALSE(CostModel::MayMatch({a.get(), b.get()}, XRange()));
}

TEST(MayMatchTest, OrMayMatchIfEitherSideMay) {
    EXPECT_TRUE(May(Bin(Op::OR, Cmp("x", Op::EQ, 5), Cmp("x", Op::EQ, 15)), XRange()));
    EXPECT_FALSE(May(Bin(Op::OR, Cmp("x", Op::EQ, 5), Cmp("x", Op::EQ, 25)), XRange()));
}

TEST(MayMatchTest, DateColumnWithStringLiteral) {
    const Ranges ranges = {{"d", MinMax{Date("1994-06-01"), Date("1994-12-31")}}};
    EXPECT_FALSE(May(Bin(Op::EQ, Col("d"), Lit(Value(std::string("1995-01-01")))), ranges));
    EXPECT_FALSE(May(Bin(Op::GTE, Col("d"), Lit(Value(std::string("1995-01-01")))), ranges));
    EXPECT_FALSE(May(Bin(Op::LT, Col("d"), Lit(Value(std::string("1994-06-01")))), ranges));
    EXPECT_TRUE(May(Bin(Op::LT, Col("d"), Lit(Value(std::string("1994-06-02")))), ranges));
    EXPECT_TRUE(May(Bin(Op::GTE, Col("d"), Lit(Value(std::string("1994-12-31")))), ranges));
    // A string that isn't a date can't be compared, so it may match.
    EXPECT_TRUE(May(Bin(Op::EQ, Col("d"), Lit(Value(std::string("not a date")))), ranges));
}

TEST(MayMatchTest, DecimalColumn) {
    const Ranges ranges = {{"p", MinMax{Value(1.5), Value(2.5)}}};
    EXPECT_FALSE(May(Bin(Op::GT, Col("p"), Lit(Value(2.6))), ranges));
    EXPECT_FALSE(May(Bin(Op::EQ, Col("p"), Lit(Value(int32_t{3}))), ranges));
    EXPECT_TRUE(May(Bin(Op::EQ, Col("p"), Lit(Value(int32_t{2}))), ranges));
    // A strict bound at the edge may match: safe, just not tight.
    EXPECT_TRUE(May(Bin(Op::GT, Col("p"), Lit(Value(2.5))), ranges));
}

TEST(MayMatchTest, WholeNumberColumnAgainstFraction) {
    EXPECT_FALSE(May(Bin(Op::EQ, Col("x"), Lit(Value(15.5))), XRange()));
    EXPECT_FALSE(May(Bin(Op::GT, Col("x"), Lit(Value(19.5))), Ranges{{"x", MinMax{Value(int32_t{10}), Value(int32_t{19})}}}));
    EXPECT_TRUE(May(Bin(Op::GT, Col("x"), Lit(Value(19.5))), XRange()));
}

TEST(MayMatchTest, NegatedLiteral) {
    EXPECT_FALSE(May(Bin(Op::EQ, Col("x"), Negate(Lit(Value(int32_t{1})))), XRange()));
    EXPECT_TRUE(May(Bin(Op::GT, Col("x"), Negate(Lit(Value(int32_t{5})))), XRange()));
    EXPECT_FALSE(May(Bin(Op::LT, Negate(Lit(Value(int32_t{25}))), Col("x")), Ranges{{"x", MinMax{Value(int32_t{-40}), Value(int32_t{-30})}}}));
}

TEST(MayMatchTest, VarcharColumnAlwaysMayMatch) {
    const Ranges ranges = {{"s", MinMax{Value(std::string("apple")), Value(std::string("melon"))}}};
    EXPECT_TRUE(May(Bin(Op::EQ, Col("s"), Lit(Value(std::string("zebra")))), ranges));
    EXPECT_TRUE(May(Bin(Op::LT, Col("s"), Lit(Value(std::string("aardvark")))), ranges));
}

TEST(MayMatchTest, UnknownColumnMayMatch) {
    EXPECT_TRUE(May(Cmp("y", Op::EQ, 5), XRange()));
    EXPECT_TRUE(May(Cmp("x", Op::EQ, 5), Ranges{}));
}

TEST(MayMatchTest, OtherShapesMayMatch) {
    const Ranges single = {{"x", MinMax{Value(int32_t{15}), Value(int32_t{15})}}};
    EXPECT_TRUE(May(Cmp("x", Op::NEQ, 15), single));
    EXPECT_TRUE(May(Bin(Op::LT, Col("x"), Col("y")), XRange()));
    EXPECT_TRUE(May(Bin(Op::EQ, Bin(Op::ADD, Col("x"), Lit(Value(int32_t{1}))), Lit(Value(int32_t{100}))), XRange()));
    EXPECT_TRUE(May(Bin(Op::LIKE, Col("x"), Lit(Value(std::string("1%")))), XRange()));

    auto negation = std::make_unique<UnaryOp>();
    negation->op = UnaryOp::Op::NOT;
    negation->operand = Cmp("x", Op::EQ, 15);
    const std::unique_ptr<Expression> wrapped = std::move(negation);
    EXPECT_TRUE(May(wrapped, single));

    EXPECT_TRUE(May(Bin(Op::EQ, Col("x"), Lit(Value(std::numeric_limits<double>::quiet_NaN()))), XRange()));
    EXPECT_TRUE(CostModel::MayMatch({}, XRange()));
}

TEST(MayMatchTest, BigintBeyondExactDoublesMayMatch) {
    const int64_t base = int64_t{1} << 53;
    const Ranges ranges = {{"b", MinMax{Value(base + 1), Value(base + 3)}}};
    EXPECT_TRUE(May(Bin(Op::GT, Col("b"), Lit(Value(base + 2))), ranges));
    EXPECT_TRUE(May(Bin(Op::EQ, Col("b"), Lit(Value(int64_t{5}))), ranges));

    // Small BIGINT values still skip.
    const Ranges small = {{"b", MinMax{Value(int64_t{100}), Value(int64_t{200})}}};
    EXPECT_FALSE(May(Bin(Op::GT, Col("b"), Lit(Value(int64_t{200}))), small));
    EXPECT_FALSE(May(Bin(Op::EQ, Col("b"), Lit(Value(int32_t{99}))), small));

    // Past 2^53 I never skip, even when skipping would be right.
    EXPECT_TRUE(May(Bin(Op::LT, Col("b"), Lit(Value(base * 4))), Ranges{{"b", MinMax{Value(base * 8), Value(base * 9)}}}));

    // At the edge: just below 2^53 still skips; 2^53 and 2^53 + 1 never do.
    EXPECT_FALSE(May(Bin(Op::GT, Col("b"), Lit(Value(base - 1))), small));
    EXPECT_TRUE(May(Bin(Op::GT, Col("b"), Lit(Value(base))), small));
    EXPECT_TRUE(May(Bin(Op::GT, Col("b"), Lit(Value(base + 1))), small));
    EXPECT_TRUE(May(Bin(Op::LT, Col("b"), Lit(Value(-base - 1))), small));
    EXPECT_TRUE(May(Bin(Op::EQ, Col("b"), Lit(Value(int64_t{150}))), Ranges{{"b", MinMax{Value(-base - 1), Value(int64_t{200})}}}));
}

}  // namespace
}  // namespace shilmandb
