#include <gtest/gtest.h>
#include "parser/parser.hpp"
#include "types/value.hpp"

namespace shilmandb {

// Helper: downcast Expression* to a specific type (same as between_in_test.cpp).
template <typename T>
static const T* As(const Expression* expr) {
    return dynamic_cast<const T*>(expr);
}

// -----------------------------------------------------------------------
// LikeSimple: col LIKE 'abc%' -> BinaryOp(LIKE, ColumnRef, Literal)
// -----------------------------------------------------------------------
TEST(LikeParserTest, LikeSimple) {
    Parser parser("SELECT * FROM t WHERE col LIKE 'abc%'");
    auto stmt = parser.Parse();

    ASSERT_NE(stmt->where_clause, nullptr);
    auto* root = As<BinaryOp>(stmt->where_clause.get());
    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->op, BinaryOp::Op::LIKE);

    auto* lhs = As<ColumnRef>(root->left.get());
    ASSERT_NE(lhs, nullptr);
    EXPECT_EQ(lhs->column_name, "col");

    auto* rhs = As<Literal>(root->right.get());
    ASSERT_NE(rhs, nullptr);
    EXPECT_EQ(rhs->value.type_, TypeId::VARCHAR);
    EXPECT_EQ(rhs->value.ToString(), "abc%");
}

// -----------------------------------------------------------------------
// NotLike: col NOT LIKE 'abc%' -> UnaryOp(NOT, BinaryOp(LIKE, ...))
// -----------------------------------------------------------------------
TEST(LikeParserTest, NotLike) {
    Parser parser("SELECT * FROM t WHERE col NOT LIKE 'abc%'");
    auto stmt = parser.Parse();

    ASSERT_NE(stmt->where_clause, nullptr);
    auto* un = As<UnaryOp>(stmt->where_clause.get());
    ASSERT_NE(un, nullptr);
    EXPECT_EQ(un->op, UnaryOp::Op::NOT);

    auto* inner = As<BinaryOp>(un->operand.get());
    ASSERT_NE(inner, nullptr);
    EXPECT_EQ(inner->op, BinaryOp::Op::LIKE);

    auto* lhs = As<ColumnRef>(inner->left.get());
    ASSERT_NE(lhs, nullptr);
    EXPECT_EQ(lhs->column_name, "col");
}

// -----------------------------------------------------------------------
// LikeInWhere: full SELECT with LIKE in WHERE clause parses cleanly.
// -----------------------------------------------------------------------
TEST(LikeParserTest, LikeInWhere) {
    Parser parser(
        "SELECT p_name FROM part WHERE p_name LIKE '%green%'");
    auto stmt = parser.Parse();

    ASSERT_EQ(stmt->select_list.size(), 1u);
    ASSERT_NE(stmt->where_clause, nullptr);

    auto* like = As<BinaryOp>(stmt->where_clause.get());
    ASSERT_NE(like, nullptr);
    EXPECT_EQ(like->op, BinaryOp::Op::LIKE);
    auto* pat = As<Literal>(like->right.get());
    ASSERT_NE(pat, nullptr);
    EXPECT_EQ(pat->value.ToString(), "%green%");
}

// -----------------------------------------------------------------------
// LikeInCase: CASE WHEN col LIKE 'PROMO%' THEN 1 ELSE 0 END parses.
// -----------------------------------------------------------------------
TEST(LikeParserTest, LikeInCase) {
    Parser parser(
        "SELECT CASE WHEN p_type LIKE 'PROMO%' THEN 1 ELSE 0 END FROM part");
    auto stmt = parser.Parse();

    ASSERT_EQ(stmt->select_list.size(), 1u);
    auto* ce = As<CaseExpression>(stmt->select_list[0].expr.get());
    ASSERT_NE(ce, nullptr);
    ASSERT_EQ(ce->when_clauses.size(), 1u);

    auto* cond = As<BinaryOp>(ce->when_clauses[0].first.get());
    ASSERT_NE(cond, nullptr);
    EXPECT_EQ(cond->op, BinaryOp::Op::LIKE);
}

// -----------------------------------------------------------------------
// LikeInsideAnd: WHERE x = 1 AND y LIKE '%z' — LIKE and EQ bind tighter
// than AND, so the result is AND(EQ(...), LIKE(...)).
// -----------------------------------------------------------------------
TEST(LikeParserTest, LikeInsideAnd) {
    Parser parser("SELECT * FROM t WHERE x = 1 AND y LIKE '%z'");
    auto stmt = parser.Parse();

    ASSERT_NE(stmt->where_clause, nullptr);
    auto* root = As<BinaryOp>(stmt->where_clause.get());
    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->op, BinaryOp::Op::AND);

    auto* left = As<BinaryOp>(root->left.get());
    ASSERT_NE(left, nullptr);
    EXPECT_EQ(left->op, BinaryOp::Op::EQ);

    auto* right = As<BinaryOp>(root->right.get());
    ASSERT_NE(right, nullptr);
    EXPECT_EQ(right->op, BinaryOp::Op::LIKE);
}

// -----------------------------------------------------------------------
// LikeWithUnderscorePattern: 'a_c' preserved verbatim in the Literal
// (lexer does not interpret _ or %).
// -----------------------------------------------------------------------
TEST(LikeParserTest, LikeWithUnderscorePattern) {
    Parser parser("SELECT * FROM t WHERE s LIKE 'a_c'");
    auto stmt = parser.Parse();

    auto* like = As<BinaryOp>(stmt->where_clause.get());
    ASSERT_NE(like, nullptr);
    EXPECT_EQ(like->op, BinaryOp::Op::LIKE);
    auto* pat = As<Literal>(like->right.get());
    ASSERT_NE(pat, nullptr);
    EXPECT_EQ(pat->value.ToString(), "a_c");
}

}  // namespace shilmandb
