#include <gtest/gtest.h>
#include <cmath>
#include "planner/join_order_optimizer.hpp"

namespace shilmandb {

// ---------------------------------------------------------------------------
// Test helpers: build AST nodes for join conditions
// ---------------------------------------------------------------------------

auto MakeTableRef(const std::string& name,
                  std::optional<std::string> alias = std::nullopt) -> TableRef {
    TableRef ref;
    ref.table_name = name;
    ref.alias = std::move(alias);
    return ref;
}

auto MakeStats(uint64_t row_count,
               std::unordered_map<std::string, uint64_t> distinct = {}) -> TableStats {
    TableStats s;
    s.row_count = row_count;
    s.distinct_counts = std::move(distinct);
    return s;
}

auto MakeEquiJoinExpr(const std::string& table_a, const std::string& col_a,
                      const std::string& table_b, const std::string& col_b)
    -> std::unique_ptr<Expression> {
    auto eq = std::make_unique<BinaryOp>();
    eq->op = BinaryOp::Op::EQ;

    auto left = std::make_unique<ColumnRef>();
    left->table_name = table_a;
    left->column_name = col_a;

    auto right = std::make_unique<ColumnRef>();
    right->table_name = table_b;
    right->column_name = col_b;

    eq->left = std::move(left);
    eq->right = std::move(right);
    return eq;
}

auto MakeAnd(std::unique_ptr<Expression> left,
             std::unique_ptr<Expression> right) -> std::unique_ptr<Expression> {
    auto and_op = std::make_unique<BinaryOp>();
    and_op->op = BinaryOp::Op::AND;
    and_op->left = std::move(left);
    and_op->right = std::move(right);
    return and_op;
}

auto MakeJoinClause(const std::string& right_table,
                    std::unique_ptr<Expression> condition,
                    JoinType type = JoinType::INNER) -> JoinClause {
    JoinClause clause;
    clause.join_type = type;
    clause.right_table.table_name = right_table;
    clause.on_condition = std::move(condition);
    return clause;
}

// ---------------------------------------------------------------------------
// Smoke test: build compiles and links
// ---------------------------------------------------------------------------
TEST(JoinOrderOptimizerTest, SmokeTest) {
    std::vector<TableRef> tables = {MakeTableRef("t")};
    std::vector<JoinClause> joins;
    std::vector<TableStats> stats = {MakeStats(100)};

    auto order = JoinOrderOptimizer::FindBestOrder(tables, joins, stats);
    ASSERT_EQ(order.size(), 1u);
    EXPECT_EQ(order[0], 0);
}

// ---------------------------------------------------------------------------
// EstimateCost: trivial cases
// ---------------------------------------------------------------------------
TEST(JoinOrderOptimizerTest, EstimateCostEmptyOrder) {
    std::vector<TableRef> tables;
    std::vector<JoinClause> joins;
    std::vector<TableStats> stats;

    double cost = JoinOrderOptimizer::EstimateCost({}, tables, joins, stats);
    EXPECT_DOUBLE_EQ(cost, 0.0);
}

TEST(JoinOrderOptimizerTest, EstimateCostSingleTable) {
    std::vector<TableRef> tables = {MakeTableRef("orders")};
    std::vector<JoinClause> joins;
    std::vector<TableStats> stats = {MakeStats(1000)};

    double cost = JoinOrderOptimizer::EstimateCost({0}, tables, joins, stats);
    EXPECT_DOUBLE_EQ(cost, 1000.0);
}

// ---------------------------------------------------------------------------
// EstimateCost: two-table equi-join
// ---------------------------------------------------------------------------
TEST(JoinOrderOptimizerTest, EstimateCostTwoTableEquiJoin) {
    // A: 1000 rows, id has 1000 distinct values
    // B: 500 rows, id has 200 distinct values
    // Join: A.id = B.id
    //
    // Order [0, 1]:
    //   intermediate = 1000, cost = 1000
    //   selectivity = 1 / max(1000, 200) = 1/1000
    //   intermediate = 1000 * 500 / 1000 = 500, cost = 1000 + 500 = 1500
    //
    // Order [1, 0]:
    //   intermediate = 500, cost = 500
    //   selectivity = 1 / max(1000, 200) = 1/1000
    //   intermediate = 500 * 1000 / 1000 = 500, cost = 500 + 500 = 1000

    std::vector<TableRef> tables = {MakeTableRef("A"), MakeTableRef("B")};
    std::vector<TableStats> stats = {
        MakeStats(1000, {{"id", 1000}}),
        MakeStats(500,  {{"id", 200}})
    };

    std::vector<JoinClause> joins;
    joins.push_back(MakeJoinClause("B", MakeEquiJoinExpr("A", "id", "B", "id")));

    double cost_01 = JoinOrderOptimizer::EstimateCost({0, 1}, tables, joins, stats);
    double cost_10 = JoinOrderOptimizer::EstimateCost({1, 0}, tables, joins, stats);

    EXPECT_DOUBLE_EQ(cost_01, 1500.0);
    EXPECT_DOUBLE_EQ(cost_10, 1000.0);
    EXPECT_LT(cost_10, cost_01);
}

TEST(JoinOrderOptimizerTest, EstimateCostSymmetricSelectivity) {
    // Both tables same size, same distinct counts -> both orderings equal cost
    std::vector<TableRef> tables = {MakeTableRef("A"), MakeTableRef("B")};
    std::vector<TableStats> stats = {
        MakeStats(500, {{"id", 500}}),
        MakeStats(500, {{"id", 500}})
    };

    std::vector<JoinClause> joins;
    joins.push_back(MakeJoinClause("B", MakeEquiJoinExpr("A", "id", "B", "id")));

    double cost_01 = JoinOrderOptimizer::EstimateCost({0, 1}, tables, joins, stats);
    double cost_10 = JoinOrderOptimizer::EstimateCost({1, 0}, tables, joins, stats);

    EXPECT_DOUBLE_EQ(cost_01, cost_10);
}

// ---------------------------------------------------------------------------
// FindBestOrder: empty and single table
// ---------------------------------------------------------------------------
TEST(JoinOrderOptimizerTest, FindBestOrderEmpty) {
    std::vector<TableRef> tables;
    std::vector<JoinClause> joins;
    std::vector<TableStats> stats;

    auto order = JoinOrderOptimizer::FindBestOrder(tables, joins, stats);
    EXPECT_TRUE(order.empty());
}

// ---------------------------------------------------------------------------
// FindBestOrder: three tables — optimizer must pick optimal ordering
// ---------------------------------------------------------------------------
TEST(JoinOrderOptimizerTest, FindBestOrderThreeTables) {
    // Table 0 (A): 1000 rows, a_id: 1000 distinct
    // Table 1 (B): 100 rows,  b_id: 100, a_id: 50 distinct
    // Table 2 (C): 10 rows,   c_id: 10,  b_id: 10 distinct
    //
    // Joins: A.a_id = B.a_id, B.b_id = C.b_id
    //
    // Optimal: [2, 1, 0] with cost 30 (start small, chain joins)
    // Worst:   [0, 1, 2] with cost 1110

    std::vector<TableRef> tables = {
        MakeTableRef("A"), MakeTableRef("B"), MakeTableRef("C")
    };
    std::vector<TableStats> stats = {
        MakeStats(1000, {{"a_id", 1000}}),
        MakeStats(100,  {{"b_id", 100}, {"a_id", 50}}),
        MakeStats(10,   {{"c_id", 10},  {"b_id", 10}})
    };

    std::vector<JoinClause> joins;
    joins.push_back(MakeJoinClause("B", MakeEquiJoinExpr("A", "a_id", "B", "a_id")));
    joins.push_back(MakeJoinClause("C", MakeEquiJoinExpr("B", "b_id", "C", "b_id")));

    auto order = JoinOrderOptimizer::FindBestOrder(tables, joins, stats);

    ASSERT_EQ(order.size(), 3u);
    EXPECT_EQ(order[0], 2);  // C first (smallest)
    EXPECT_EQ(order[1], 1);  // B second (joins to C)
    EXPECT_EQ(order[2], 0);  // A last (joins to B)

    // Verify the cost is indeed 30
    double best_cost = JoinOrderOptimizer::EstimateCost(order, tables, joins, stats);
    EXPECT_DOUBLE_EQ(best_cost, 30.0);
}

// ---------------------------------------------------------------------------
// FindBestOrder: N > 6 falls back to identity order
// ---------------------------------------------------------------------------
TEST(JoinOrderOptimizerTest, FindBestOrderFallbackOver6) {
    std::vector<TableRef> tables;
    std::vector<TableStats> stats;
    for (int i = 0; i < 7; ++i) {
        tables.push_back(MakeTableRef("t" + std::to_string(i)));
        stats.push_back(MakeStats(100));
    }
    std::vector<JoinClause> joins;

    auto order = JoinOrderOptimizer::FindBestOrder(tables, joins, stats);

    ASSERT_EQ(order.size(), 7u);
    for (int i = 0; i < 7; ++i) {
        EXPECT_EQ(order[i], i);
    }
}

// ---------------------------------------------------------------------------
// EstimateCost: cross product (no join predicate)
// ---------------------------------------------------------------------------
TEST(JoinOrderOptimizerTest, EstimateCostCrossProduct) {
    // No join -> cross product: intermediate = A.rows * B.rows
    std::vector<TableRef> tables = {MakeTableRef("A"), MakeTableRef("B")};
    std::vector<TableStats> stats = {MakeStats(100), MakeStats(50)};
    std::vector<JoinClause> joins;

    // Order [0, 1]: cost = 100 + 100*50 = 5100
    double cost_01 = JoinOrderOptimizer::EstimateCost({0, 1}, tables, joins, stats);
    EXPECT_DOUBLE_EQ(cost_01, 5100.0);

    // Order [1, 0]: cost = 50 + 50*100 = 5050
    double cost_10 = JoinOrderOptimizer::EstimateCost({1, 0}, tables, joins, stats);
    EXPECT_DOUBLE_EQ(cost_10, 5050.0);
}

// ---------------------------------------------------------------------------
// EstimateCost: multi-column join (AND chain with two EQ pairs)
// ---------------------------------------------------------------------------
TEST(JoinOrderOptimizerTest, EstimateCostMultiColumnJoin) {
    // A: 1000 rows, x: 100 distinct, y: 50 distinct
    // B: 500 rows,  x: 200 distinct, y: 25 distinct
    // Join: A.x = B.x AND A.y = B.y
    //
    // Selectivity = 1/max(100,200) * 1/max(50,25) = 1/200 * 1/50 = 1/10000
    // Order [0,1]: cost = 1000 + 1000*500/10000 = 1000 + 50 = 1050

    std::vector<TableRef> tables = {MakeTableRef("A"), MakeTableRef("B")};
    std::vector<TableStats> stats = {
        MakeStats(1000, {{"x", 100}, {"y", 50}}),
        MakeStats(500,  {{"x", 200}, {"y", 25}})
    };

    auto condition = MakeAnd(
        MakeEquiJoinExpr("A", "x", "B", "x"),
        MakeEquiJoinExpr("A", "y", "B", "y")
    );
    std::vector<JoinClause> joins;
    joins.push_back(MakeJoinClause("B", std::move(condition)));

    double cost = JoinOrderOptimizer::EstimateCost({0, 1}, tables, joins, stats);
    EXPECT_DOUBLE_EQ(cost, 1050.0);
}

// ---------------------------------------------------------------------------
// EstimateCost: missing distinct counts -> fallback to row_count
// ---------------------------------------------------------------------------
TEST(JoinOrderOptimizerTest, EstimateCostMissingDistinctCounts) {
    // A: 1000 rows, NO distinct_counts at all
    // B: 500 rows,  id: 200 distinct
    // Join: A.id = B.id
    //
    // V_a for "id" in A: not found -> fallback to row_count = 1000
    // Selectivity = 1 / max(1000, 200) = 1/1000
    // Order [0,1]: cost = 1000 + 1000*500/1000 = 1500

    std::vector<TableRef> tables = {MakeTableRef("A"), MakeTableRef("B")};
    std::vector<TableStats> stats = {
        MakeStats(1000),  // no distinct_counts
        MakeStats(500, {{"id", 200}})
    };

    std::vector<JoinClause> joins;
    joins.push_back(MakeJoinClause("B", MakeEquiJoinExpr("A", "id", "B", "id")));

    double cost = JoinOrderOptimizer::EstimateCost({0, 1}, tables, joins, stats);
    EXPECT_DOUBLE_EQ(cost, 1500.0);
}

// ---------------------------------------------------------------------------
// EstimateCost: alias-based table resolution
// ---------------------------------------------------------------------------
TEST(JoinOrderOptimizerTest, EstimateCostWithAliases) {
    // Tables use aliases: "orders" aliased as "o", "lineitem" aliased as "l"
    // Join condition references aliases: o.order_id = l.order_id
    std::vector<TableRef> tables = {
        MakeTableRef("orders", "o"),
        MakeTableRef("lineitem", "l")
    };
    std::vector<TableStats> stats = {
        MakeStats(1000, {{"order_id", 1000}}),
        MakeStats(5000, {{"order_id", 1000}})
    };

    // Join uses aliases in column refs
    std::vector<JoinClause> joins;
    joins.push_back(MakeJoinClause("lineitem", MakeEquiJoinExpr("o", "order_id", "l", "order_id")));

    // Selectivity = 1/max(1000,1000) = 1/1000
    // Order [0,1]: cost = 1000 + 1000*5000/1000 = 1000 + 5000 = 6000
    double cost = JoinOrderOptimizer::EstimateCost({0, 1}, tables, joins, stats);
    EXPECT_DOUBLE_EQ(cost, 6000.0);
}

// ---------------------------------------------------------------------------
// EstimateCost: zero distinct count -> selectivity = 1.0 (cross product)
// ---------------------------------------------------------------------------
TEST(JoinOrderOptimizerTest, EstimateCostZeroDistinctCount) {
    // Both tables have 0 distinct values for the join column
    // max(V_a, V_b) == 0 -> selectivity contribution skipped (1.0)
    // Effectively a cross product despite having a join predicate
    std::vector<TableRef> tables = {MakeTableRef("A"), MakeTableRef("B")};
    std::vector<TableStats> stats = {
        MakeStats(100, {{"id", 0}}),
        MakeStats(50,  {{"id", 0}})
    };

    std::vector<JoinClause> joins;
    joins.push_back(MakeJoinClause("B", MakeEquiJoinExpr("A", "id", "B", "id")));

    // has_join is true but selectivity stays 1.0 (guard: max_v == 0)
    // cost = 100 + 100*50*1.0 = 100 + 5000 = 5100
    double cost = JoinOrderOptimizer::EstimateCost({0, 1}, tables, joins, stats);
    EXPECT_DOUBLE_EQ(cost, 5100.0);
}

// ---------------------------------------------------------------------------
// BuildFeatureVector: always returns 48 floats
// ---------------------------------------------------------------------------
TEST(JoinOrderOptimizerTest, FeatureVectorSize) {
    // 0 tables
    {
        std::vector<TableRef> tables;
        std::vector<JoinClause> joins;
        std::vector<TableStats> stats;
        auto fv = JoinOrderOptimizer::BuildFeatureVector(tables, joins, stats);
        EXPECT_EQ(fv.size(), 48u);
    }
    // 1 table
    {
        std::vector<TableRef> tables = {MakeTableRef("t")};
        std::vector<JoinClause> joins;
        std::vector<TableStats> stats = {MakeStats(100, {{"id", 50}})};
        auto fv = JoinOrderOptimizer::BuildFeatureVector(tables, joins, stats);
        EXPECT_EQ(fv.size(), 48u);
    }
    // 6 tables
    {
        std::vector<TableRef> tables;
        std::vector<TableStats> stats;
        for (int i = 0; i < 6; ++i) {
            tables.push_back(MakeTableRef("t" + std::to_string(i)));
            stats.push_back(MakeStats(100));
        }
        std::vector<JoinClause> joins;
        auto fv = JoinOrderOptimizer::BuildFeatureVector(tables, joins, stats);
        EXPECT_EQ(fv.size(), 48u);
    }
}

// ---------------------------------------------------------------------------
// BuildFeatureVector: table features (log-scaled) + zero-padding
// ---------------------------------------------------------------------------
TEST(JoinOrderOptimizerTest, FeatureVectorTableFeatures) {
    // 2 tables, verify per-table features and zero-padding for slots 2-5
    std::vector<TableRef> tables = {MakeTableRef("A"), MakeTableRef("B")};
    std::vector<TableStats> stats = {
        MakeStats(1000, {{"id", 500}}),
        MakeStats(100,  {{"id", 50}})
    };
    std::vector<JoinClause> joins;

    auto fv = JoinOrderOptimizer::BuildFeatureVector(tables, joins, stats);

    // Table 0: log(1001), log(501), 1.0
    EXPECT_FLOAT_EQ(fv[0], static_cast<float>(std::log(1001.0)));
    EXPECT_FLOAT_EQ(fv[1], static_cast<float>(std::log(501.0)));
    EXPECT_FLOAT_EQ(fv[2], 1.0f);

    // Table 1: log(101), log(51), 1.0
    EXPECT_FLOAT_EQ(fv[3], static_cast<float>(std::log(101.0)));
    EXPECT_FLOAT_EQ(fv[4], static_cast<float>(std::log(51.0)));
    EXPECT_FLOAT_EQ(fv[5], 1.0f);

    // Tables 2-5: zero-padded
    for (int i = 6; i < 18; ++i) {
        EXPECT_FLOAT_EQ(fv[i], 0.0f) << "slot index " << i;
    }
}

// ---------------------------------------------------------------------------
// BuildFeatureVector: pair features (is_joined + cardinality_ratio)
// ---------------------------------------------------------------------------
TEST(JoinOrderOptimizerTest, FeatureVectorPairFeatures) {
    // 2 tables with a join: A.id = B.id
    // A: id has 500 distinct, B: id has 50 distinct
    // Pair (0,1) is pair index 0 -> offset 18
    //   is_joined = 1.0
    //   ratio = min(500,50)/max(500,50) = 50/500 = 0.1

    std::vector<TableRef> tables = {MakeTableRef("A"), MakeTableRef("B")};
    std::vector<TableStats> stats = {
        MakeStats(1000, {{"id", 500}}),
        MakeStats(100,  {{"id", 50}})
    };

    std::vector<JoinClause> joins;
    joins.push_back(MakeJoinClause("B", MakeEquiJoinExpr("A", "id", "B", "id")));

    auto fv = JoinOrderOptimizer::BuildFeatureVector(tables, joins, stats);

    // Pair (0,1) at offset 18
    EXPECT_FLOAT_EQ(fv[18], 1.0f);   // is_joined
    EXPECT_FLOAT_EQ(fv[19], 0.1f);   // cardinality_ratio = 50/500

    // All other pairs: not joined (0.0, 0.0)
    for (int p = 1; p < 15; ++p) {
        int offset = 18 + p * 2;
        EXPECT_FLOAT_EQ(fv[offset], 0.0f) << "pair " << p << " is_joined";
        EXPECT_FLOAT_EQ(fv[offset + 1], 0.0f) << "pair " << p << " ratio";
    }
}

}  // namespace shilmandb
