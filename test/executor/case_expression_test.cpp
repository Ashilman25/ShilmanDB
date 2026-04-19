#include <gtest/gtest.h>
#include "executor/expression_evaluator.hpp"
#include "parser/ast.hpp"
#include "catalog/schema.hpp"
#include "types/tuple.hpp"
#include "types/value.hpp"
#include <memory>
#include <utility>

namespace shilmandb {

// --- AST builder helpers ---

static std::unique_ptr<Expression> ColRef(const std::string& col) {
    auto e = std::make_unique<ColumnRef>();
    e->column_name = col;
    return e;
}

static std::unique_ptr<Expression> IntLit(int32_t v) {
    auto e = std::make_unique<Literal>();
    e->value = Value(v);
    return e;
}

static std::unique_ptr<Expression> StrLit(const std::string& v) {
    auto e = std::make_unique<Literal>();
    e->value = Value(v);
    return e;
}

static std::unique_ptr<Expression> BinOp(BinaryOp::Op op,
                                         std::unique_ptr<Expression> l,
                                         std::unique_ptr<Expression> r) {
    auto e = std::make_unique<BinaryOp>();
    e->op = op;
    e->left = std::move(l);
    e->right = std::move(r);
    return e;
}

static std::unique_ptr<CaseExpression> MakeCase(
    std::vector<std::pair<std::unique_ptr<Expression>,
                          std::unique_ptr<Expression>>> whens,
    std::unique_ptr<Expression> else_branch) {
    auto ce = std::make_unique<CaseExpression>();
    for (auto& [c, r] : whens) {
        ce->when_clauses.emplace_back(std::move(c), std::move(r));
    }
    ce->else_clause = std::move(else_branch);
    return ce;
}

// --- Fixture: one-column INTEGER schema "x", pluggable value ---

static Schema IntSchema() {
    return Schema({Column("x", TypeId::INTEGER)});
}

static Tuple IntTuple(int32_t v) {
    return Tuple(std::vector<Value>{Value(v)}, IntSchema());
}

// -----------------------------------------------------------------------
// CaseReturnsFirstMatch: earlier WHEN wins
// -----------------------------------------------------------------------
TEST(CaseExpressionTest, CaseReturnsFirstMatch) {
    std::vector<std::pair<std::unique_ptr<Expression>,
                          std::unique_ptr<Expression>>> whens;
    whens.emplace_back(BinOp(BinaryOp::Op::GT, ColRef("x"), IntLit(0)), IntLit(1));
    whens.emplace_back(BinOp(BinaryOp::Op::GT, ColRef("x"), IntLit(3)), IntLit(2));

    auto ce = MakeCase(std::move(whens), IntLit(0));
    auto schema = IntSchema();
    auto tup = IntTuple(5);  // matches both WHENs

    auto result = EvaluateExpression(ce.get(), tup, schema);
    EXPECT_EQ(result.type_, TypeId::INTEGER);
    EXPECT_EQ(result.integer_, 1);  // first match wins
}

// -----------------------------------------------------------------------
// CaseFallsThroughToElse: no WHEN matches, ELSE returned
// -----------------------------------------------------------------------
TEST(CaseExpressionTest, CaseFallsThroughToElse) {
    std::vector<std::pair<std::unique_ptr<Expression>,
                          std::unique_ptr<Expression>>> whens;
    whens.emplace_back(BinOp(BinaryOp::Op::GT, ColRef("x"), IntLit(0)), IntLit(1));

    auto ce = MakeCase(std::move(whens), IntLit(99));
    auto schema = IntSchema();
    auto tup = IntTuple(-5);

    auto result = EvaluateExpression(ce.get(), tup, schema);
    EXPECT_EQ(result.integer_, 99);
}

// -----------------------------------------------------------------------
// CaseNoElseReturnsZero: no match, no ELSE -> Value(int32_t{0})
// -----------------------------------------------------------------------
TEST(CaseExpressionTest, CaseNoElseReturnsZero) {
    std::vector<std::pair<std::unique_ptr<Expression>,
                          std::unique_ptr<Expression>>> whens;
    whens.emplace_back(BinOp(BinaryOp::Op::GT, ColRef("x"), IntLit(0)), IntLit(1));

    auto ce = MakeCase(std::move(whens), nullptr);
    auto schema = IntSchema();
    auto tup = IntTuple(-5);

    auto result = EvaluateExpression(ce.get(), tup, schema);
    EXPECT_EQ(result.type_, TypeId::INTEGER);
    EXPECT_EQ(result.integer_, 0);
}

// -----------------------------------------------------------------------
// CaseWithArithmeticInThen: THEN evaluates against the tuple
// -----------------------------------------------------------------------
TEST(CaseExpressionTest, CaseWithArithmeticInThen) {
    // CASE WHEN x > 0 THEN x + 10 ELSE 0 END
    auto then_expr = BinOp(BinaryOp::Op::ADD, ColRef("x"), IntLit(10));
    std::vector<std::pair<std::unique_ptr<Expression>,
                          std::unique_ptr<Expression>>> whens;
    whens.emplace_back(
        BinOp(BinaryOp::Op::GT, ColRef("x"), IntLit(0)), std::move(then_expr));

    auto ce = MakeCase(std::move(whens), IntLit(0));
    auto schema = IntSchema();
    auto tup = IntTuple(5);

    auto result = EvaluateExpression(ce.get(), tup, schema);
    EXPECT_EQ(result.integer_, 15);
}

// -----------------------------------------------------------------------
// CaseWithVarcharCondition: VARCHAR equality in WHEN
// -----------------------------------------------------------------------
TEST(CaseExpressionTest, CaseWithVarcharCondition) {
    Schema schema({Column("name", TypeId::VARCHAR)});
    Tuple tup(std::vector<Value>{Value(std::string("alice"))}, schema);

    std::vector<std::pair<std::unique_ptr<Expression>,
                          std::unique_ptr<Expression>>> whens;
    whens.emplace_back(
        BinOp(BinaryOp::Op::EQ, ColRef("name"), StrLit("alice")), IntLit(1));

    auto ce = MakeCase(std::move(whens), IntLit(0));

    auto result = EvaluateExpression(ce.get(), tup, schema);
    EXPECT_EQ(result.integer_, 1);
}

// -----------------------------------------------------------------------
// CaseNestedExecutes: nested CASE returns inner result
// -----------------------------------------------------------------------
TEST(CaseExpressionTest, CaseNestedExecutes) {
    // Outer: CASE WHEN x > 0 THEN <inner> ELSE 0 END
    //   Inner: CASE WHEN x > 10 THEN 100 ELSE 50 END
    std::vector<std::pair<std::unique_ptr<Expression>,
                          std::unique_ptr<Expression>>> inner_whens;
    inner_whens.emplace_back(
        BinOp(BinaryOp::Op::GT, ColRef("x"), IntLit(10)), IntLit(100));
    auto inner = MakeCase(std::move(inner_whens), IntLit(50));

    std::vector<std::pair<std::unique_ptr<Expression>,
                          std::unique_ptr<Expression>>> outer_whens;
    outer_whens.emplace_back(
        BinOp(BinaryOp::Op::GT, ColRef("x"), IntLit(0)), std::move(inner));
    auto outer = MakeCase(std::move(outer_whens), IntLit(0));

    auto schema = IntSchema();
    auto tup = IntTuple(5);  // outer matches, inner does not

    auto result = EvaluateExpression(outer.get(), tup, schema);
    EXPECT_EQ(result.integer_, 50);
}

// -----------------------------------------------------------------------
// CaseInSumAggregate: Q12 shape. Fold CASE evaluations across tuples the
// same way AggregateExecutor accumulates SUM — verifies the CASE path
// integrates with aggregate semantics.
// -----------------------------------------------------------------------
TEST(CaseExpressionTest, CaseInSumAggregate) {
    // CASE WHEN x > 0 THEN 1 ELSE 0 END
    std::vector<std::pair<std::unique_ptr<Expression>,
                          std::unique_ptr<Expression>>> whens;
    whens.emplace_back(
        BinOp(BinaryOp::Op::GT, ColRef("x"), IntLit(0)), IntLit(1));
    auto ce = MakeCase(std::move(whens), IntLit(0));
    auto schema = IntSchema();

    const std::vector<int32_t> inputs = {-2, 5, 0, 7, -1, 3};
    int32_t sum = 0;
    for (int32_t v : inputs) {
        auto tup = IntTuple(v);
        sum += EvaluateExpression(ce.get(), tup, schema).integer_;
    }
    // Three strictly-positive values (5, 7, 3) each contribute 1.
    EXPECT_EQ(sum, 3);
}

// -----------------------------------------------------------------------
// BetweenInFilter: the AND(GTE, LTE) tree the parser produces for
// BETWEEN evaluates correctly as a filter predicate.
// -----------------------------------------------------------------------
TEST(CaseExpressionTest, BetweenInFilter) {
    // x BETWEEN 1 AND 10  ->  AND(x >= 1, x <= 10)
    auto predicate =
        BinOp(BinaryOp::Op::AND,
              BinOp(BinaryOp::Op::GTE, ColRef("x"), IntLit(1)),
              BinOp(BinaryOp::Op::LTE, ColRef("x"), IntLit(10)));
    auto schema = IntSchema();

    EXPECT_FALSE(IsTruthy(EvaluateExpression(predicate.get(), IntTuple(0),  schema)));
    EXPECT_TRUE (IsTruthy(EvaluateExpression(predicate.get(), IntTuple(1),  schema)));
    EXPECT_TRUE (IsTruthy(EvaluateExpression(predicate.get(), IntTuple(5),  schema)));
    EXPECT_TRUE (IsTruthy(EvaluateExpression(predicate.get(), IntTuple(10), schema)));
    EXPECT_FALSE(IsTruthy(EvaluateExpression(predicate.get(), IntTuple(11), schema)));
}

// -----------------------------------------------------------------------
// InListFilter: the OR chain the parser produces for IN (...) evaluates
// correctly as a filter predicate.
// -----------------------------------------------------------------------
TEST(CaseExpressionTest, InListFilter) {
    // x IN (1, 5, 10)  ->  OR(x=1, OR(x=5, x=10))
    auto predicate =
        BinOp(BinaryOp::Op::OR,
              BinOp(BinaryOp::Op::EQ, ColRef("x"), IntLit(1)),
              BinOp(BinaryOp::Op::OR,
                    BinOp(BinaryOp::Op::EQ, ColRef("x"), IntLit(5)),
                    BinOp(BinaryOp::Op::EQ, ColRef("x"), IntLit(10))));
    auto schema = IntSchema();

    EXPECT_TRUE (IsTruthy(EvaluateExpression(predicate.get(), IntTuple(1),  schema)));
    EXPECT_TRUE (IsTruthy(EvaluateExpression(predicate.get(), IntTuple(5),  schema)));
    EXPECT_TRUE (IsTruthy(EvaluateExpression(predicate.get(), IntTuple(10), schema)));
    EXPECT_FALSE(IsTruthy(EvaluateExpression(predicate.get(), IntTuple(2),  schema)));
    EXPECT_FALSE(IsTruthy(EvaluateExpression(predicate.get(), IntTuple(11), schema)));
}

}  // namespace shilmandb
