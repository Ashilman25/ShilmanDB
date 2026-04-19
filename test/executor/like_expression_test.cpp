#include <gtest/gtest.h>
#include "executor/expression_evaluator.hpp"
#include "parser/ast.hpp"
#include "catalog/schema.hpp"
#include "types/tuple.hpp"
#include "types/value.hpp"
#include <memory>
#include <string>
#include <utility>

namespace shilmandb {

// --- AST builder helpers (same pattern as case_expression_test.cpp) ---

static std::unique_ptr<Expression> ColRef(const std::string& col) {
    auto e = std::make_unique<ColumnRef>();
    e->column_name = col;
    return e;
}

static std::unique_ptr<Expression> StrLit(const std::string& v) {
    auto e = std::make_unique<Literal>();
    e->value = Value(v);
    return e;
}

static std::unique_ptr<Expression> Like(std::unique_ptr<Expression> l,
                                        std::unique_ptr<Expression> r) {
    auto e = std::make_unique<BinaryOp>();
    e->op = BinaryOp::Op::LIKE;
    e->left = std::move(l);
    e->right = std::move(r);
    return e;
}

// --- Single-column VARCHAR schema "s", pluggable value ---

static Schema StrSchema() {
    return Schema({Column("s", TypeId::VARCHAR)});
}

static Tuple StrTuple(const std::string& v) {
    return Tuple(std::vector<Value>{Value(v)}, StrSchema());
}

// -----------------------------------------------------------------------
// LikePrefixMatch: 'PROMO123' LIKE 'PROMO%' -> true
// -----------------------------------------------------------------------
TEST(LikeExpressionTest, LikePrefixMatch) {
    auto expr = Like(ColRef("s"), StrLit("PROMO%"));
    auto schema = StrSchema();
    auto tup = StrTuple("PROMO123");

    auto result = EvaluateExpression(expr.get(), tup, schema);
    EXPECT_EQ(result.type_, TypeId::INTEGER);
    EXPECT_EQ(result.integer_, 1);
}

// -----------------------------------------------------------------------
// LikeSuffixMatch: 'hello world' LIKE '%world' -> true
// -----------------------------------------------------------------------
TEST(LikeExpressionTest, LikeSuffixMatch) {
    auto expr = Like(ColRef("s"), StrLit("%world"));
    auto schema = StrSchema();
    auto tup = StrTuple("hello world");

    auto result = EvaluateExpression(expr.get(), tup, schema);
    EXPECT_EQ(result.integer_, 1);
}

// -----------------------------------------------------------------------
// LikeContainsMatch: 'some green item' LIKE '%green%' -> true
// -----------------------------------------------------------------------
TEST(LikeExpressionTest, LikeContainsMatch) {
    auto expr = Like(ColRef("s"), StrLit("%green%"));
    auto schema = StrSchema();
    auto tup = StrTuple("some green item");

    auto result = EvaluateExpression(expr.get(), tup, schema);
    EXPECT_EQ(result.integer_, 1);
}

// -----------------------------------------------------------------------
// LikeUnderscoreMatch: 'abc' LIKE 'a_c' -> true
// -----------------------------------------------------------------------
TEST(LikeExpressionTest, LikeUnderscoreMatch) {
    auto expr = Like(ColRef("s"), StrLit("a_c"));
    auto schema = StrSchema();
    auto tup = StrTuple("abc");

    auto result = EvaluateExpression(expr.get(), tup, schema);
    EXPECT_EQ(result.integer_, 1);
}

// -----------------------------------------------------------------------
// LikeNoMatch: 'xyz' LIKE 'abc%' -> false
// -----------------------------------------------------------------------
TEST(LikeExpressionTest, LikeNoMatch) {
    auto expr = Like(ColRef("s"), StrLit("abc%"));
    auto schema = StrSchema();
    auto tup = StrTuple("xyz");

    auto result = EvaluateExpression(expr.get(), tup, schema);
    EXPECT_EQ(result.type_, TypeId::INTEGER);
    EXPECT_EQ(result.integer_, 0);
}

// -----------------------------------------------------------------------
// LikeEmptyPattern: empty pattern matches only empty string
// -----------------------------------------------------------------------
TEST(LikeExpressionTest, LikeEmptyPattern) {
    auto schema = StrSchema();

    {
        auto expr = Like(ColRef("s"), StrLit(""));
        auto tup = StrTuple("");
        auto result = EvaluateExpression(expr.get(), tup, schema);
        EXPECT_EQ(result.integer_, 1);
    }
    {
        auto expr = Like(ColRef("s"), StrLit(""));
        auto tup = StrTuple("x");
        auto result = EvaluateExpression(expr.get(), tup, schema);
        EXPECT_EQ(result.integer_, 0);
    }
}

}  // namespace shilmandb
