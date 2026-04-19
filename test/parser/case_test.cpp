#include <gtest/gtest.h>
#include "parser/parser.hpp"
#include "parser/parse_exception.hpp"
#include "types/value.hpp"
#include <memory>

namespace shilmandb {

template <typename T>
static const T* As(const Expression* expr) {
    return dynamic_cast<const T*>(expr);
}

// Locate the first CaseExpression anywhere in a parsed SelectStatement's
// select_list, regardless of aliasing/wrapping, for the nested / composed
// tests that do not care about the surrounding expression shape.
static const CaseExpression* FirstCase(const Expression* root) {
    if (!root) return nullptr;
    if (auto* c = As<CaseExpression>(root)) return c;
    if (auto* bin = As<BinaryOp>(root)) {
        if (auto* l = FirstCase(bin->left.get())) return l;
        if (auto* r = FirstCase(bin->right.get())) return r;
    }
    if (auto* un = As<UnaryOp>(root)) {
        if (auto* o = FirstCase(un->operand.get())) return o;
    }
    if (auto* agg = As<Aggregate>(root)) {
        if (auto* a = FirstCase(agg->arg.get())) return a;
    }
    return nullptr;
}

// -----------------------------------------------------------------------
// CaseSimple: one WHEN and one ELSE
// -----------------------------------------------------------------------
TEST(CaseTest, CaseSimple) {
    Parser parser("SELECT CASE WHEN x > 0 THEN 1 ELSE 0 END FROM t");
    auto stmt = parser.Parse();
    ASSERT_EQ(stmt->select_list.size(), 1u);

    auto* ce = As<CaseExpression>(stmt->select_list[0].expr.get());
    ASSERT_NE(ce, nullptr);
    ASSERT_EQ(ce->when_clauses.size(), 1u);
    ASSERT_NE(ce->else_clause, nullptr);

    auto* cond = As<BinaryOp>(ce->when_clauses[0].first.get());
    ASSERT_NE(cond, nullptr);
    EXPECT_EQ(cond->op, BinaryOp::Op::GT);

    auto* then_lit = As<Literal>(ce->when_clauses[0].second.get());
    ASSERT_NE(then_lit, nullptr);
    EXPECT_EQ(then_lit->value.integer_, 1);

    auto* else_lit = As<Literal>(ce->else_clause.get());
    ASSERT_NE(else_lit, nullptr);
    EXPECT_EQ(else_lit->value.integer_, 0);
}

// -----------------------------------------------------------------------
// CaseMultipleWhens: three WHEN branches, no ELSE
// -----------------------------------------------------------------------
TEST(CaseTest, CaseMultipleWhens) {
    Parser parser(
        "SELECT CASE "
        "WHEN x < 0 THEN 1 "
        "WHEN x = 0 THEN 2 "
        "WHEN x > 0 THEN 3 "
        "END FROM t");
    auto stmt = parser.Parse();
    auto* ce = As<CaseExpression>(stmt->select_list[0].expr.get());
    ASSERT_NE(ce, nullptr);
    ASSERT_EQ(ce->when_clauses.size(), 3u);
    EXPECT_EQ(ce->else_clause, nullptr);

    auto get_then = [](const CaseExpression* ce, size_t i) {
        return As<Literal>(ce->when_clauses[i].second.get());
    };
    ASSERT_NE(get_then(ce, 0), nullptr);
    EXPECT_EQ(get_then(ce, 0)->value.integer_, 1);
    EXPECT_EQ(get_then(ce, 1)->value.integer_, 2);
    EXPECT_EQ(get_then(ce, 2)->value.integer_, 3);
}

// -----------------------------------------------------------------------
// CaseNoElse: else_clause is nullptr
// -----------------------------------------------------------------------
TEST(CaseTest, CaseNoElse) {
    Parser parser("SELECT CASE WHEN x > 0 THEN 1 END FROM t");
    auto stmt = parser.Parse();
    auto* ce = As<CaseExpression>(stmt->select_list[0].expr.get());
    ASSERT_NE(ce, nullptr);
    ASSERT_EQ(ce->when_clauses.size(), 1u);
    EXPECT_EQ(ce->else_clause, nullptr);
}

// -----------------------------------------------------------------------
// CaseEmptyWhensThrows: CASE ELSE 0 END with no WHEN clauses
// -----------------------------------------------------------------------
TEST(CaseTest, CaseEmptyWhensThrows) {
    Parser parser("SELECT CASE ELSE 0 END FROM t");
    EXPECT_THROW((void)parser.Parse(), ParseException);
}

// -----------------------------------------------------------------------
// CaseMissingEndThrows: unterminated CASE (no END)
// -----------------------------------------------------------------------
TEST(CaseTest, CaseMissingEndThrows) {
    Parser parser("SELECT CASE WHEN x > 0 THEN 1 FROM t");
    EXPECT_THROW((void)parser.Parse(), ParseException);
}

// -----------------------------------------------------------------------
// CaseNested: a CASE inside a THEN branch
// -----------------------------------------------------------------------
TEST(CaseTest, CaseNested) {
    Parser parser(
        "SELECT CASE WHEN x > 0 "
        "THEN CASE WHEN x > 10 THEN 100 ELSE 50 END "
        "ELSE 0 END FROM t");
    auto stmt = parser.Parse();
    auto* outer = As<CaseExpression>(stmt->select_list[0].expr.get());
    ASSERT_NE(outer, nullptr);
    ASSERT_EQ(outer->when_clauses.size(), 1u);

    auto* inner = As<CaseExpression>(outer->when_clauses[0].second.get());
    ASSERT_NE(inner, nullptr);
    ASSERT_EQ(inner->when_clauses.size(), 1u);
    ASSERT_NE(inner->else_clause, nullptr);
    auto* inner_else = As<Literal>(inner->else_clause.get());
    ASSERT_NE(inner_else, nullptr);
    EXPECT_EQ(inner_else->value.integer_, 50);
}

// -----------------------------------------------------------------------
// CaseInsideAggregate: SUM(CASE WHEN ... THEN 1 ELSE 0 END) (Q12 shape)
// -----------------------------------------------------------------------
TEST(CaseTest, CaseInsideAggregate) {
    Parser parser(
        "SELECT SUM(CASE WHEN x = 'URGENT' THEN 1 ELSE 0 END) FROM t");
    auto stmt = parser.Parse();
    auto* agg = As<Aggregate>(stmt->select_list[0].expr.get());
    ASSERT_NE(agg, nullptr);
    EXPECT_EQ(agg->func, Aggregate::Func::SUM);

    auto* ce = As<CaseExpression>(agg->arg.get());
    ASSERT_NE(ce, nullptr);
    ASSERT_EQ(ce->when_clauses.size(), 1u);
    ASSERT_NE(ce->else_clause, nullptr);
}

// -----------------------------------------------------------------------
// CaseInsideArithmetic: 100 * SUM(CASE ... END) / SUM(col) (Q14 shape)
// -----------------------------------------------------------------------
TEST(CaseTest, CaseInsideArithmetic) {
    Parser parser(
        "SELECT 100 * SUM(CASE WHEN x > 0 "
        "THEN a ELSE 0 END) / SUM(a) FROM t");
    auto stmt = parser.Parse();
    // A CaseExpression must appear somewhere in the SELECT-list expression.
    EXPECT_NE(FirstCase(stmt->select_list[0].expr.get()), nullptr);
}

// -----------------------------------------------------------------------
// CaseWithStringComparison: WHEN with VARCHAR equality
// -----------------------------------------------------------------------
TEST(CaseTest, CaseWithStringComparison) {
    Parser parser(
        "SELECT CASE WHEN priority = '1-URGENT' THEN 1 ELSE 0 END FROM t");
    auto stmt = parser.Parse();
    auto* ce = As<CaseExpression>(stmt->select_list[0].expr.get());
    ASSERT_NE(ce, nullptr);
    ASSERT_EQ(ce->when_clauses.size(), 1u);

    auto* cond = As<BinaryOp>(ce->when_clauses[0].first.get());
    ASSERT_NE(cond, nullptr);
    EXPECT_EQ(cond->op, BinaryOp::Op::EQ);
    auto* rhs = As<Literal>(cond->right.get());
    ASSERT_NE(rhs, nullptr);
    EXPECT_EQ(rhs->value.type_, TypeId::VARCHAR);
    EXPECT_EQ(rhs->value.varchar_, "1-URGENT");
}

// -----------------------------------------------------------------------
// CaseInWhereClause: CASE used as a predicate (rare but valid)
// -----------------------------------------------------------------------
TEST(CaseTest, CaseInWhereClause) {
    Parser parser(
        "SELECT * FROM t "
        "WHERE (CASE WHEN x > 0 THEN 1 ELSE 0 END) = 1");
    auto stmt = parser.Parse();
    ASSERT_NE(stmt->where_clause, nullptr);
    auto* eq = As<BinaryOp>(stmt->where_clause.get());
    ASSERT_NE(eq, nullptr);
    EXPECT_EQ(eq->op, BinaryOp::Op::EQ);
    EXPECT_NE(As<CaseExpression>(eq->left.get()), nullptr);
}

}  // namespace shilmandb
