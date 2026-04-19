#include <gtest/gtest.h>
#include "parser/parser.hpp"
#include "types/value.hpp"
#include <cstdint>

namespace shilmandb {

// Tier 1 smoke suite: compact shape-checks for BETWEEN / IN / CASE. The
// dedicated tier-2 files (between_in_test.cpp, case_test.cpp) cover edge
// cases; these tests assert only the top-level desugared shape so a
// regression in the happy path is caught fast.

template <typename T>
static const T* As(const Expression* expr) {
    return dynamic_cast<const T*>(expr);
}

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
// BETWEEN / NOT BETWEEN
// -----------------------------------------------------------------------

TEST(BetweenInCaseTest, BetweenDesugars) {
    Parser parser("SELECT * FROM t WHERE x BETWEEN 1 AND 10");
    auto stmt = parser.Parse();

    auto* root = As<BinaryOp>(stmt->where_clause.get());
    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->op, BinaryOp::Op::AND);

    auto* gte = As<BinaryOp>(root->left.get());
    ASSERT_NE(gte, nullptr);
    EXPECT_EQ(gte->op, BinaryOp::Op::GTE);

    auto* lte = As<BinaryOp>(root->right.get());
    ASSERT_NE(lte, nullptr);
    EXPECT_EQ(lte->op, BinaryOp::Op::LTE);
}

TEST(BetweenInCaseTest, NotBetween) {
    Parser parser("SELECT * FROM t WHERE x NOT BETWEEN 1 AND 10");
    auto stmt = parser.Parse();

    auto* not_node = As<UnaryOp>(stmt->where_clause.get());
    ASSERT_NE(not_node, nullptr);
    EXPECT_EQ(not_node->op, UnaryOp::Op::NOT);

    auto* inner = As<BinaryOp>(not_node->operand.get());
    ASSERT_NE(inner, nullptr);
    EXPECT_EQ(inner->op, BinaryOp::Op::AND);
}

// -----------------------------------------------------------------------
// IN / NOT IN
// -----------------------------------------------------------------------

TEST(BetweenInCaseTest, InValueList) {
    // Right-associative: OR( EQ(x,1), OR( EQ(x,2), EQ(x,3) ) )
    Parser parser("SELECT * FROM t WHERE x IN (1, 2, 3)");
    auto stmt = parser.Parse();

    auto* outer = As<BinaryOp>(stmt->where_clause.get());
    ASSERT_NE(outer, nullptr);
    EXPECT_EQ(outer->op, BinaryOp::Op::OR);

    auto* first = As<BinaryOp>(outer->left.get());
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->op, BinaryOp::Op::EQ);

    auto* tail = As<BinaryOp>(outer->right.get());
    ASSERT_NE(tail, nullptr);
    EXPECT_EQ(tail->op, BinaryOp::Op::OR);
}

TEST(BetweenInCaseTest, InSingleValue) {
    Parser parser("SELECT * FROM t WHERE x IN (5)");
    auto stmt = parser.Parse();

    auto* root = As<BinaryOp>(stmt->where_clause.get());
    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->op, BinaryOp::Op::EQ);

    auto* lit = As<Literal>(root->right.get());
    ASSERT_NE(lit, nullptr);
    EXPECT_EQ(lit->value.integer_, 5);
}

TEST(BetweenInCaseTest, NotIn) {
    // De Morgan at parse time: AND chain of NEQ, no UnaryOp wrapper.
    Parser parser("SELECT * FROM t WHERE x NOT IN (1, 2, 3)");
    auto stmt = parser.Parse();

    auto* outer = As<BinaryOp>(stmt->where_clause.get());
    ASSERT_NE(outer, nullptr);
    EXPECT_EQ(outer->op, BinaryOp::Op::AND);

    auto* first = As<BinaryOp>(outer->left.get());
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->op, BinaryOp::Op::NEQ);

    auto* tail = As<BinaryOp>(outer->right.get());
    ASSERT_NE(tail, nullptr);
    EXPECT_EQ(tail->op, BinaryOp::Op::AND);
    EXPECT_EQ(As<BinaryOp>(tail->left.get())->op, BinaryOp::Op::NEQ);
    EXPECT_EQ(As<BinaryOp>(tail->right.get())->op, BinaryOp::Op::NEQ);
}

// -----------------------------------------------------------------------
// CASE WHEN
// -----------------------------------------------------------------------

TEST(BetweenInCaseTest, CaseSimple) {
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

TEST(BetweenInCaseTest, CaseMultipleWhens) {
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

    for (size_t i = 0; i < 3; ++i) {
        auto* then_lit = As<Literal>(ce->when_clauses[i].second.get());
        ASSERT_NE(then_lit, nullptr);
        EXPECT_EQ(then_lit->value.integer_, static_cast<int32_t>(i + 1));
    }
}

TEST(BetweenInCaseTest, CaseNoElse) {
    Parser parser("SELECT CASE WHEN x > 0 THEN 1 END FROM t");
    auto stmt = parser.Parse();

    auto* ce = As<CaseExpression>(stmt->select_list[0].expr.get());
    ASSERT_NE(ce, nullptr);
    EXPECT_EQ(ce->when_clauses.size(), 1u);
    EXPECT_EQ(ce->else_clause, nullptr);
}

TEST(BetweenInCaseTest, CaseNested) {
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
    EXPECT_EQ(inner->when_clauses.size(), 1u);
    ASSERT_NE(inner->else_clause, nullptr);
    EXPECT_EQ(As<Literal>(inner->else_clause.get())->value.integer_, 50);
}

TEST(BetweenInCaseTest, CaseInAggregate) {
    // Q12 shape: SUM(CASE WHEN priority = 'URGENT' THEN 1 ELSE 0 END)
    Parser parser(
        "SELECT SUM(CASE WHEN priority = 'URGENT' THEN 1 ELSE 0 END) FROM t");
    auto stmt = parser.Parse();

    auto* agg = As<Aggregate>(stmt->select_list[0].expr.get());
    ASSERT_NE(agg, nullptr);
    EXPECT_EQ(agg->func, Aggregate::Func::SUM);
    EXPECT_NE(FirstCase(agg->arg.get()), nullptr);
}

}  // namespace shilmandb
