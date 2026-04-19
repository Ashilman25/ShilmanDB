#include <gtest/gtest.h>
#include "parser/parser.hpp"
#include "parser/parse_exception.hpp"
#include "types/value.hpp"
#include <algorithm>
#include <functional>
#include <vector>
#include <cstdint>

namespace shilmandb {

// Helper: downcast Expression* to a specific type
template <typename T>
const T* As(const Expression* expr) {
    return dynamic_cast<const T*>(expr);
}

// -----------------------------------------------------------------------
// Test 1: BetweenDesugarsToAndOfGteAndLte
// -----------------------------------------------------------------------
TEST(BetweenInTest, BetweenDesugarsToAndOfGteAndLte) {
    Parser parser("SELECT * FROM t WHERE x BETWEEN 1 AND 10");
    auto stmt = parser.Parse();

    ASSERT_NE(stmt->where_clause, nullptr);
    auto* root = As<BinaryOp>(stmt->where_clause.get());
    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->op, BinaryOp::Op::AND);

    auto* gte = As<BinaryOp>(root->left.get());
    ASSERT_NE(gte, nullptr);
    EXPECT_EQ(gte->op, BinaryOp::Op::GTE);
    auto* gte_col = As<ColumnRef>(gte->left.get());
    ASSERT_NE(gte_col, nullptr);
    EXPECT_EQ(gte_col->column_name, "x");
    auto* gte_val = As<Literal>(gte->right.get());
    ASSERT_NE(gte_val, nullptr);
    EXPECT_EQ(gte_val->value.type_, TypeId::INTEGER);
    EXPECT_EQ(gte_val->value.integer_, 1);

    auto* lte = As<BinaryOp>(root->right.get());
    ASSERT_NE(lte, nullptr);
    EXPECT_EQ(lte->op, BinaryOp::Op::LTE);
    auto* lte_col = As<ColumnRef>(lte->left.get());
    ASSERT_NE(lte_col, nullptr);
    EXPECT_EQ(lte_col->column_name, "x");
    auto* lte_val = As<Literal>(lte->right.get());
    ASSERT_NE(lte_val, nullptr);
    EXPECT_EQ(lte_val->value.type_, TypeId::INTEGER);
    EXPECT_EQ(lte_val->value.integer_, 10);

    // Distinct ColumnRef objects (clone, not alias)
    EXPECT_NE(gte_col, lte_col);
}

// -----------------------------------------------------------------------
// Test 2: BetweenWithDateLiteral
// -----------------------------------------------------------------------
TEST(BetweenInTest, BetweenWithDateLiteral) {
    Parser parser(
        "SELECT * FROM lineitem "
        "WHERE l_shipdate BETWEEN DATE '1994-01-01' AND DATE '1995-01-01'");
    auto stmt = parser.Parse();

    ASSERT_NE(stmt->where_clause, nullptr);
    auto* root = As<BinaryOp>(stmt->where_clause.get());
    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->op, BinaryOp::Op::AND);

    auto* gte = As<BinaryOp>(root->left.get());
    ASSERT_NE(gte, nullptr);
    auto* gte_val = As<Literal>(gte->right.get());
    ASSERT_NE(gte_val, nullptr);
    EXPECT_EQ(gte_val->value.type_, TypeId::DATE);

    auto* lte = As<BinaryOp>(root->right.get());
    ASSERT_NE(lte, nullptr);
    auto* lte_val = As<Literal>(lte->right.get());
    ASSERT_NE(lte_val, nullptr);
    EXPECT_EQ(lte_val->value.type_, TypeId::DATE);
}

// -----------------------------------------------------------------------
// Test 3: BetweenInAndChainDoesNotSwallowOuterAnd
// -----------------------------------------------------------------------
TEST(BetweenInTest, BetweenInAndChainDoesNotSwallowOuterAnd) {
    // The `AND` inside BETWEEN must not swallow the outer logical AND.
    // Expected tree: AND( BETWEEN-desugared-tree , GT(y, 5) )
    Parser parser("SELECT * FROM t WHERE x BETWEEN 1 AND 10 AND y > 5");
    auto stmt = parser.Parse();

    ASSERT_NE(stmt->where_clause, nullptr);
    auto* outer = As<BinaryOp>(stmt->where_clause.get());
    ASSERT_NE(outer, nullptr);
    EXPECT_EQ(outer->op, BinaryOp::Op::AND);

    // Left side: the BETWEEN-desugared AND tree
    auto* between_tree = As<BinaryOp>(outer->left.get());
    ASSERT_NE(between_tree, nullptr);
    EXPECT_EQ(between_tree->op, BinaryOp::Op::AND);
    auto* gte = As<BinaryOp>(between_tree->left.get());
    ASSERT_NE(gte, nullptr);
    EXPECT_EQ(gte->op, BinaryOp::Op::GTE);
    auto* lte = As<BinaryOp>(between_tree->right.get());
    ASSERT_NE(lte, nullptr);
    EXPECT_EQ(lte->op, BinaryOp::Op::LTE);

    // Right side: y > 5
    auto* gt = As<BinaryOp>(outer->right.get());
    ASSERT_NE(gt, nullptr);
    EXPECT_EQ(gt->op, BinaryOp::Op::GT);
    auto* gt_col = As<ColumnRef>(gt->left.get());
    ASSERT_NE(gt_col, nullptr);
    EXPECT_EQ(gt_col->column_name, "y");
}

// -----------------------------------------------------------------------
// Test 4: BetweenInsideFullSelect
// -----------------------------------------------------------------------
TEST(BetweenInTest, BetweenInsideFullSelect) {
    Parser parser("SELECT * FROM lineitem WHERE l_quantity BETWEEN 1 AND 24");
    auto stmt = parser.Parse();

    ASSERT_EQ(stmt->from_clause.size(), 1u);
    EXPECT_EQ(stmt->from_clause[0].table_name, "lineitem");

    ASSERT_NE(stmt->where_clause, nullptr);
    auto* root = As<BinaryOp>(stmt->where_clause.get());
    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->op, BinaryOp::Op::AND);
}

// -----------------------------------------------------------------------
// Test 5: InListDesugarsToOrChain
// -----------------------------------------------------------------------
TEST(BetweenInTest, InListDesugarsToOrChain) {
    // Right-associative shape: OR( EQ(x, 1), OR( EQ(x, 2), EQ(x, 3) ) )
    Parser parser("SELECT * FROM t WHERE x IN (1, 2, 3)");
    auto stmt = parser.Parse();

    ASSERT_NE(stmt->where_clause, nullptr);
    auto* outer = As<BinaryOp>(stmt->where_clause.get());
    ASSERT_NE(outer, nullptr);
    EXPECT_EQ(outer->op, BinaryOp::Op::OR);

    auto* first = As<BinaryOp>(outer->left.get());
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->op, BinaryOp::Op::EQ);
    auto* first_val = As<Literal>(first->right.get());
    ASSERT_NE(first_val, nullptr);
    EXPECT_EQ(first_val->value.integer_, 1);

    auto* inner_or = As<BinaryOp>(outer->right.get());
    ASSERT_NE(inner_or, nullptr);
    EXPECT_EQ(inner_or->op, BinaryOp::Op::OR);

    auto* second = As<BinaryOp>(inner_or->left.get());
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(second->op, BinaryOp::Op::EQ);
    auto* second_val = As<Literal>(second->right.get());
    ASSERT_NE(second_val, nullptr);
    EXPECT_EQ(second_val->value.integer_, 2);

    auto* third = As<BinaryOp>(inner_or->right.get());
    ASSERT_NE(third, nullptr);
    EXPECT_EQ(third->op, BinaryOp::Op::EQ);
    auto* third_val = As<Literal>(third->right.get());
    ASSERT_NE(third_val, nullptr);
    EXPECT_EQ(third_val->value.integer_, 3);
}

// -----------------------------------------------------------------------
// Test 6: InSingleValueDesugarsToEq
// -----------------------------------------------------------------------
TEST(BetweenInTest, InSingleValueDesugarsToEq) {
    // No surrounding OR for a single-value list.
    Parser parser("SELECT * FROM t WHERE x IN (5)");
    auto stmt = parser.Parse();

    ASSERT_NE(stmt->where_clause, nullptr);
    auto* root = As<BinaryOp>(stmt->where_clause.get());
    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->op, BinaryOp::Op::EQ);

    auto* col = As<ColumnRef>(root->left.get());
    ASSERT_NE(col, nullptr);
    EXPECT_EQ(col->column_name, "x");

    auto* lit = As<Literal>(root->right.get());
    ASSERT_NE(lit, nullptr);
    EXPECT_EQ(lit->value.integer_, 5);
}

// -----------------------------------------------------------------------
// Test 7: InListWithStringLiterals (Q19 shape)
// -----------------------------------------------------------------------
TEST(BetweenInTest, InListWithStringLiterals) {
    Parser parser(
        "SELECT * FROM lineitem "
        "WHERE l_shipmode IN ('AIR', 'AIR REG', 'RAIL')");
    auto stmt = parser.Parse();

    ASSERT_NE(stmt->where_clause, nullptr);
    auto* outer = As<BinaryOp>(stmt->where_clause.get());
    ASSERT_NE(outer, nullptr);
    EXPECT_EQ(outer->op, BinaryOp::Op::OR);

    auto* first = As<BinaryOp>(outer->left.get());
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->op, BinaryOp::Op::EQ);
    auto* first_val = As<Literal>(first->right.get());
    ASSERT_NE(first_val, nullptr);
    EXPECT_EQ(first_val->value.type_, TypeId::VARCHAR);
    EXPECT_EQ(first_val->value.varchar_, "AIR");
}

// -----------------------------------------------------------------------
// Test 8: InListPreservesAllValues
// -----------------------------------------------------------------------
TEST(BetweenInTest, InListPreservesAllValues) {
    // The chain is built right-to-left, so order varies — assert as set.
    Parser parser("SELECT * FROM t WHERE x IN (10, 20, 30, 40)");
    auto stmt = parser.Parse();

    ASSERT_NE(stmt->where_clause, nullptr);

    // Walk the OR chain, collect all RHS literal integers.
    std::vector<int32_t> seen;
    std::function<void(const Expression*)> walk =
        [&](const Expression* e) {
            auto* bin = As<BinaryOp>(e);
            ASSERT_NE(bin, nullptr);
            if (bin->op == BinaryOp::Op::OR) {
                walk(bin->left.get());
                walk(bin->right.get());
            } else if (bin->op == BinaryOp::Op::EQ) {
                auto* lit = As<Literal>(bin->right.get());
                ASSERT_NE(lit, nullptr);
                seen.push_back(lit->value.integer_);
            } else {
                FAIL() << "unexpected op in IN-desugared chain";
            }
        };
    walk(stmt->where_clause.get());

    std::sort(seen.begin(), seen.end());
    EXPECT_EQ(seen, (std::vector<int32_t>{10, 20, 30, 40}));
}

// -----------------------------------------------------------------------
// Test 9: EmptyInListThrows
// -----------------------------------------------------------------------
TEST(BetweenInTest, EmptyInListThrows) {
    Parser parser("SELECT * FROM t WHERE x IN ()");
    EXPECT_THROW((void)parser.Parse(), ParseException);
}

// -----------------------------------------------------------------------
// Test 10: InListMissingRParenThrows
// -----------------------------------------------------------------------
TEST(BetweenInTest, InListMissingRParenThrows) {
    Parser parser("SELECT * FROM t WHERE x IN (1, 2");
    EXPECT_THROW((void)parser.Parse(), ParseException);
}

// -----------------------------------------------------------------------
// Test 11: NotBetweenWrapsInUnaryNot
// -----------------------------------------------------------------------
TEST(BetweenInTest, NotBetweenWrapsInUnaryNot) {
    Parser parser("SELECT * FROM t WHERE x NOT BETWEEN 1 AND 10");
    auto stmt = parser.Parse();

    ASSERT_NE(stmt->where_clause, nullptr);
    auto* not_node = As<UnaryOp>(stmt->where_clause.get());
    ASSERT_NE(not_node, nullptr);
    EXPECT_EQ(not_node->op, UnaryOp::Op::NOT);

    // Operand is the same desugared AND tree as a bare BETWEEN
    auto* and_node = As<BinaryOp>(not_node->operand.get());
    ASSERT_NE(and_node, nullptr);
    EXPECT_EQ(and_node->op, BinaryOp::Op::AND);

    auto* gte = As<BinaryOp>(and_node->left.get());
    ASSERT_NE(gte, nullptr);
    EXPECT_EQ(gte->op, BinaryOp::Op::GTE);

    auto* lte = As<BinaryOp>(and_node->right.get());
    ASSERT_NE(lte, nullptr);
    EXPECT_EQ(lte->op, BinaryOp::Op::LTE);
}

// -----------------------------------------------------------------------
// Test 12: NotInDesugarsToAndChainOfNeq
// -----------------------------------------------------------------------
TEST(BetweenInTest, NotInDesugarsToAndChainOfNeq) {
    // De Morgan applied at parse time: AND chain of NEQ, no UnaryOp wrapper.
    Parser parser("SELECT * FROM t WHERE x NOT IN (1, 2, 3)");
    auto stmt = parser.Parse();

    ASSERT_NE(stmt->where_clause, nullptr);
    auto* outer = As<BinaryOp>(stmt->where_clause.get());
    ASSERT_NE(outer, nullptr);
    EXPECT_EQ(outer->op, BinaryOp::Op::AND);

    auto* first = As<BinaryOp>(outer->left.get());
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->op, BinaryOp::Op::NEQ);

    auto* inner = As<BinaryOp>(outer->right.get());
    ASSERT_NE(inner, nullptr);
    EXPECT_EQ(inner->op, BinaryOp::Op::AND);

    auto* second = As<BinaryOp>(inner->left.get());
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(second->op, BinaryOp::Op::NEQ);

    auto* third = As<BinaryOp>(inner->right.get());
    ASSERT_NE(third, nullptr);
    EXPECT_EQ(third->op, BinaryOp::Op::NEQ);
}

// -----------------------------------------------------------------------
// Test 13: Q19FragmentParses (BETWEEN + IN composed inside one query)
// -----------------------------------------------------------------------
TEST(BetweenInTest, Q19FragmentParses) {
    // Smallest realistic end-to-end stress test:
    //   - implicit join (comma-separated FROM)
    //   - BETWEEN with integer range
    //   - IN list with VARCHAR literals
    // Expected WHERE shape:
    //   AND( BETWEEN-desugared(p_size, 1, 5),
    //        IN-desugared(l_shipmode, 'AIR', 'AIR REG') )
    Parser parser(
        "SELECT * FROM part p, lineitem l "
        "WHERE p_size BETWEEN 1 AND 5 "
        "AND l_shipmode IN ('AIR', 'AIR REG')");
    auto stmt = parser.Parse();

    ASSERT_EQ(stmt->from_clause.size(), 2u);
    EXPECT_EQ(stmt->from_clause[0].table_name, "part");
    EXPECT_EQ(stmt->from_clause[1].table_name, "lineitem");

    ASSERT_NE(stmt->where_clause, nullptr);
    auto* outer = As<BinaryOp>(stmt->where_clause.get());
    ASSERT_NE(outer, nullptr);
    EXPECT_EQ(outer->op, BinaryOp::Op::AND);

    // Left: BETWEEN desugared into AND(GTE, LTE)
    auto* between_tree = As<BinaryOp>(outer->left.get());
    ASSERT_NE(between_tree, nullptr);
    EXPECT_EQ(between_tree->op, BinaryOp::Op::AND);
    auto* gte = As<BinaryOp>(between_tree->left.get());
    ASSERT_NE(gte, nullptr);
    EXPECT_EQ(gte->op, BinaryOp::Op::GTE);
    auto* lte = As<BinaryOp>(between_tree->right.get());
    ASSERT_NE(lte, nullptr);
    EXPECT_EQ(lte->op, BinaryOp::Op::LTE);

    // Right: IN desugared into OR chain of EQ
    auto* in_tree = As<BinaryOp>(outer->right.get());
    ASSERT_NE(in_tree, nullptr);
    EXPECT_EQ(in_tree->op, BinaryOp::Op::OR);
    auto* first_eq = As<BinaryOp>(in_tree->left.get());
    ASSERT_NE(first_eq, nullptr);
    EXPECT_EQ(first_eq->op, BinaryOp::Op::EQ);
    auto* second_eq = As<BinaryOp>(in_tree->right.get());
    ASSERT_NE(second_eq, nullptr);
    EXPECT_EQ(second_eq->op, BinaryOp::Op::EQ);
}

}  // namespace shilmandb
