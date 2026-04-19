// test/planner/index_selection_test.cpp
#include <gtest/gtest.h>
#include "planner/cost_model.hpp"
#include "planner/index_selector.hpp"
#include "parser/ast.hpp"
#include "catalog/schema.hpp"
#include "catalog/table_stats.hpp"
#include "types/types.hpp"
#include "types/value.hpp"
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "storage/disk_manager.hpp"
#include "catalog/catalog.hpp"
#include <filesystem>
#include <cmath>
#include <memory>
#include <string>

namespace shilmandb {

// --- AST builder helpers (reused across all tests) ---

static std::unique_ptr<Expression> MakeColRef(const std::string& col) {
    auto e = std::make_unique<ColumnRef>();
    e->column_name = col;
    return e;
}

static std::unique_ptr<Expression> MakeLit(int32_t v) {
    auto e = std::make_unique<Literal>();
    e->value = Value(v);
    return e;
}

static std::unique_ptr<Expression> MakeBinOp(BinaryOp::Op op,
                                             std::unique_ptr<Expression> lhs,
                                             std::unique_ptr<Expression> rhs) {
    auto e = std::make_unique<BinaryOp>();
    e->op = op;
    e->left = std::move(lhs);
    e->right = std::move(rhs);
    return e;
}

// --- CostModel tests ---

TEST(CostModelTest, PageCountGrowsWithRows) {
    Schema schema({Column("id", TypeId::INTEGER), Column("val", TypeId::INTEGER)});  // 8 bytes fixed
    const double p1   = CostModel::EstimateTablePages(1,       schema);
    const double p2   = CostModel::EstimateTablePages(1000,    schema);
    const double p3   = CostModel::EstimateTablePages(1000000, schema);
    EXPECT_GE(p1, 1.0);
    EXPECT_LE(p1, p2);
    EXPECT_LE(p2, p3);
}

TEST(CostModelTest, TreeHeightMonotonic) {
    EXPECT_EQ(CostModel::EstimateTreeHeight(1,       TypeId::INTEGER), 1);
    const int h100  = CostModel::EstimateTreeHeight(100,     TypeId::INTEGER);
    const int h10k  = CostModel::EstimateTreeHeight(10000,   TypeId::INTEGER);
    const int h1M   = CostModel::EstimateTreeHeight(1000000, TypeId::INTEGER);
    EXPECT_GE(h100, 1);
    EXPECT_LE(h100, h10k);
    EXPECT_LE(h10k, h1M);
}

TEST(CostModelTest, SelectivityEqualityUsesDistinctCount) {
    TableStats stats;
    stats.row_count = 1000;
    stats.distinct_counts["col"] = 100;

    auto pred = MakeBinOp(BinaryOp::Op::EQ, MakeColRef("col"), MakeLit(5));
    const double sel = CostModel::EstimateSelectivity(pred.get(), "col", stats);
    EXPECT_NEAR(sel, 0.01, 1e-9);
}

TEST(CostModelTest, SelectivityRangeConstant) {
    TableStats stats;
    stats.row_count = 1000;
    stats.distinct_counts["col"] = 1000000;  // should be ignored for range predicates

    auto pred = MakeBinOp(BinaryOp::Op::GT, MakeColRef("col"), MakeLit(5));
    const double sel = CostModel::EstimateSelectivity(pred.get(), "col", stats);
    EXPECT_NEAR(sel, CostModel::kOpenRangeSelectivity, 1e-9);
}

TEST(CostModelTest, SelectivityConjunctionClampsToFloor) {
    TableStats stats;
    stats.row_count = 1000;
    stats.distinct_counts["col"] = 1000000;  // 1e-6 per equality, product = 1e-12

    auto a    = MakeBinOp(BinaryOp::Op::EQ, MakeColRef("col"), MakeLit(1));
    auto b    = MakeBinOp(BinaryOp::Op::EQ, MakeColRef("col"), MakeLit(2));
    auto pred = MakeBinOp(BinaryOp::Op::AND, std::move(a), std::move(b));
    const double sel = CostModel::EstimateSelectivity(pred.get(), "col", stats);
    EXPECT_NEAR(sel, CostModel::kMinSelectivity, 1e-9);
}

// --- IndexSelector: MatchSingleConjunct (Group B) tests ---
// These tests validate strict-bound and combined-range predicate recognition
// without relying on SelectScanStrategy's cost decisions, because pure open-range
// predicates never win cost comparison under the current CostModel constants
// (see design spec §5.3).

TEST(IndexSelectorMatchTest, RangePredicateLowBound) {
    auto expr = MakeBinOp(BinaryOp::Op::GT, MakeColRef("id"), MakeLit(100));
    const ConjunctMatch m = IndexSelector::MatchSingleConjunct(expr.get(), "id");
    EXPECT_TRUE(m.referenced_column);
    EXPECT_TRUE(m.contributes_low);
    EXPECT_FALSE(m.contributes_high);
    EXPECT_EQ(m.low_value, Value(static_cast<int32_t>(100)));
    EXPECT_FALSE(m.fully_consumed);  // strict > must stay in residual
}

TEST(IndexSelectorMatchTest, RangePredicateHighBound) {
    auto expr = MakeBinOp(BinaryOp::Op::LT, MakeColRef("id"), MakeLit(500));
    const ConjunctMatch m = IndexSelector::MatchSingleConjunct(expr.get(), "id");
    EXPECT_TRUE(m.referenced_column);
    EXPECT_FALSE(m.contributes_low);
    EXPECT_TRUE(m.contributes_high);
    EXPECT_EQ(m.high_value, Value(static_cast<int32_t>(500)));
    EXPECT_FALSE(m.fully_consumed);  // strict < must stay in residual
}

TEST(IndexSelectorMatchTest, CombinedRangeBothBounds) {
    // Two conjuncts (matches planner post-SplitConjuncts input).
    auto gte = MakeBinOp(BinaryOp::Op::GTE, MakeColRef("id"), MakeLit(100));
    auto lt  = MakeBinOp(BinaryOp::Op::LT,  MakeColRef("id"), MakeLit(500));

    const ConjunctMatch lo = IndexSelector::MatchSingleConjunct(gte.get(), "id");
    EXPECT_TRUE(lo.referenced_column);
    EXPECT_TRUE(lo.contributes_low);
    EXPECT_FALSE(lo.contributes_high);
    EXPECT_EQ(lo.low_value, Value(static_cast<int32_t>(100)));
    EXPECT_TRUE(lo.fully_consumed);  // >= is inclusive: fully covered by [low, high]

    const ConjunctMatch hi = IndexSelector::MatchSingleConjunct(lt.get(), "id");
    EXPECT_TRUE(hi.referenced_column);
    EXPECT_FALSE(hi.contributes_low);
    EXPECT_TRUE(hi.contributes_high);
    EXPECT_EQ(hi.high_value, Value(static_cast<int32_t>(500)));
    EXPECT_FALSE(hi.fully_consumed);  // strict < stays in residual
}

// --- IndexSelector: SelectScanStrategy (Group A) tests ---
//
// These tests invoke SelectScanStrategy end-to-end. Each test manually sets
// TableStats on the created TableInfo (no row insertion) because the cost model
// only consults stats, never scans tuples. Indexes are created on empty tables;
// Catalog::CreateIndex handles the empty-heap case by iterating zero rows.

class IndexSelectorStrategyTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() /
                      "shilmandb_index_selection_test.db").string();
        std::filesystem::remove(test_file_);
    }

    void TearDown() override {
        std::filesystem::remove(test_file_);
    }

    struct BPMBundle {
        std::unique_ptr<DiskManager> disk_manager;
        std::unique_ptr<BufferPoolManager> bpm;
    };

    static BPMBundle MakeBPM(const std::string& path, size_t pool_size = 1000) {
        auto dm = std::make_unique<DiskManager>(path);
        auto eviction = std::make_unique<LRUEvictionPolicy>(pool_size);
        auto bpm = std::make_unique<BufferPoolManager>(
            pool_size, dm.get(), std::move(eviction));
        return {std::move(dm), std::move(bpm)};
    }
};

TEST_F(IndexSelectorStrategyTest, NoIndexAvailable) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    auto* ti = catalog.CreateTable("t", Schema({Column("id", TypeId::INTEGER)}));
    ASSERT_NE(ti, nullptr);
    ti->stats.row_count = 1000;
    ti->stats.distinct_counts["id"] = 100;
    // No index created.

    std::vector<std::unique_ptr<Expression>> preds;
    preds.push_back(MakeBinOp(BinaryOp::Op::EQ, MakeColRef("id"), MakeLit(5)));

    std::vector<std::unique_ptr<Expression>> residual;
    auto scan = IndexSelector::SelectScanStrategy("t", ti->schema, preds, residual, &catalog);

    ASSERT_NE(scan, nullptr);
    EXPECT_EQ(scan->type, PlanNodeType::SEQ_SCAN);
    EXPECT_TRUE(preds.empty());          // caller's vector is drained
    EXPECT_EQ(residual.size(), 1u);      // predicate moved to residual
}

TEST_F(IndexSelectorStrategyTest, PredicateOnNonIndexedColumn) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    auto* ti = catalog.CreateTable("t", Schema({
        Column("id",    TypeId::INTEGER),
        Column("other", TypeId::INTEGER)
    }));
    ASSERT_NE(ti, nullptr);
    ti->stats.row_count = 1000;
    ti->stats.distinct_counts["id"]    = 100;
    ti->stats.distinct_counts["other"] = 100;
    (void)catalog.CreateIndex("idx_t_id", "t", "id");

    std::vector<std::unique_ptr<Expression>> preds;
    preds.push_back(MakeBinOp(BinaryOp::Op::EQ, MakeColRef("other"), MakeLit(5)));

    std::vector<std::unique_ptr<Expression>> residual;
    auto scan = IndexSelector::SelectScanStrategy("t", ti->schema, preds, residual, &catalog);

    ASSERT_NE(scan, nullptr);
    EXPECT_EQ(scan->type, PlanNodeType::SEQ_SCAN);
    EXPECT_TRUE(preds.empty());
    EXPECT_EQ(residual.size(), 1u);  // unindexed-column predicate goes to residual
}

// High selectivity (many rows match per key value) should pick SeqScan.
// rows=1000, distinct=5, sel=0.2, matching=200
// seq_cost = 1 page + 1000 * 0.01 = 11
// idx_cost = 2 * 2.0 + 200 * 4.0 + 200 * 0.01 = 806
TEST_F(IndexSelectorStrategyTest, HighSelectivityUsesSeqScan) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    auto* ti = catalog.CreateTable("t", Schema({Column("id", TypeId::INTEGER)}));
    ASSERT_NE(ti, nullptr);
    ti->stats.row_count = 1000;
    ti->stats.distinct_counts["id"] = 5;
    (void)catalog.CreateIndex("idx_t_id", "t", "id");

    std::vector<std::unique_ptr<Expression>> preds;
    preds.push_back(MakeBinOp(BinaryOp::Op::EQ, MakeColRef("id"), MakeLit(3)));

    std::vector<std::unique_ptr<Expression>> residual;
    auto scan = IndexSelector::SelectScanStrategy("t", ti->schema, preds, residual, &catalog);

    ASSERT_NE(scan, nullptr);
    EXPECT_EQ(scan->type, PlanNodeType::SEQ_SCAN);
    EXPECT_TRUE(preds.empty());
    EXPECT_EQ(residual.size(), 1u);  // predicate not consumed; goes to Filter
}

// Low selectivity (unique per key) should pick IndexScan with point lookup bounds.
// rows=1M, distinct=10k, sel=1e-4, matching=100
// seq_cost = 489 pages + 1M * 0.01 = 10489
// idx_cost = 3 * 2.0 + 100 * 4.0 + 100 * 0.01 = 407
TEST_F(IndexSelectorStrategyTest, LowSelectivityUsesIndexScan) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    auto* ti = catalog.CreateTable("t", Schema({Column("id", TypeId::INTEGER)}));
    ASSERT_NE(ti, nullptr);
    ti->stats.row_count = 1000000;
    ti->stats.distinct_counts["id"] = 10000;
    (void)catalog.CreateIndex("idx_t_id", "t", "id");

    std::vector<std::unique_ptr<Expression>> preds;
    preds.push_back(MakeBinOp(BinaryOp::Op::EQ, MakeColRef("id"), MakeLit(3)));

    std::vector<std::unique_ptr<Expression>> residual;
    auto scan = IndexSelector::SelectScanStrategy("t", ti->schema, preds, residual, &catalog);

    ASSERT_NE(scan, nullptr);
    ASSERT_EQ(scan->type, PlanNodeType::INDEX_SCAN);
    auto* idx_scan = dynamic_cast<IndexScanPlanNode*>(scan.get());
    ASSERT_NE(idx_scan, nullptr);
    EXPECT_EQ(idx_scan->index_name, "idx_t_id");
    ASSERT_TRUE(idx_scan->low_key.has_value());
    ASSERT_TRUE(idx_scan->high_key.has_value());
    EXPECT_EQ(idx_scan->low_key.value(),  Value(static_cast<int32_t>(3)));
    EXPECT_EQ(idx_scan->high_key.value(), Value(static_cast<int32_t>(3)));
    EXPECT_TRUE(preds.empty());
    EXPECT_TRUE(residual.empty());  // equality predicate fully consumed
}

TEST_F(IndexSelectorStrategyTest, EqualityPredicatePointLookup) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    auto* ti = catalog.CreateTable("t", Schema({Column("id", TypeId::INTEGER)}));
    ASSERT_NE(ti, nullptr);
    ti->stats.row_count = 1000000;
    ti->stats.distinct_counts["id"] = 10000;
    (void)catalog.CreateIndex("idx_t_id", "t", "id");

    std::vector<std::unique_ptr<Expression>> preds;
    preds.push_back(MakeBinOp(BinaryOp::Op::EQ, MakeColRef("id"), MakeLit(42)));

    std::vector<std::unique_ptr<Expression>> residual;
    auto scan = IndexSelector::SelectScanStrategy("t", ti->schema, preds, residual, &catalog);

    ASSERT_NE(scan, nullptr);
    ASSERT_EQ(scan->type, PlanNodeType::INDEX_SCAN);
    auto* idx_scan = dynamic_cast<IndexScanPlanNode*>(scan.get());
    ASSERT_NE(idx_scan, nullptr);
    ASSERT_TRUE(idx_scan->low_key.has_value());
    ASSERT_TRUE(idx_scan->high_key.has_value());
    EXPECT_EQ(idx_scan->low_key.value(),  Value(static_cast<int32_t>(42)));
    EXPECT_EQ(idx_scan->high_key.value(), Value(static_cast<int32_t>(42)));
    EXPECT_TRUE(residual.empty());
}

// Non-index predicate stays in residual so caller wraps it in a Filter above IndexScan.
TEST_F(IndexSelectorStrategyTest, ResidualFilterPreserved) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    auto* ti = catalog.CreateTable("t", Schema({
        Column("id",    TypeId::INTEGER),
        Column("other", TypeId::INTEGER)
    }));
    ASSERT_NE(ti, nullptr);
    ti->stats.row_count = 1000000;
    ti->stats.distinct_counts["id"]    = 10000;
    ti->stats.distinct_counts["other"] = 10;
    (void)catalog.CreateIndex("idx_t_id", "t", "id");

    std::vector<std::unique_ptr<Expression>> preds;
    preds.push_back(MakeBinOp(BinaryOp::Op::EQ, MakeColRef("id"),    MakeLit(5)));
    preds.push_back(MakeBinOp(BinaryOp::Op::GT, MakeColRef("other"), MakeLit(10)));

    std::vector<std::unique_ptr<Expression>> residual;
    auto scan = IndexSelector::SelectScanStrategy("t", ti->schema, preds, residual, &catalog);

    ASSERT_NE(scan, nullptr);
    ASSERT_EQ(scan->type, PlanNodeType::INDEX_SCAN);
    auto* idx_scan = dynamic_cast<IndexScanPlanNode*>(scan.get());
    ASSERT_NE(idx_scan, nullptr);
    EXPECT_EQ(idx_scan->index_name, "idx_t_id");
    ASSERT_TRUE(idx_scan->low_key.has_value());
    ASSERT_TRUE(idx_scan->high_key.has_value());
    EXPECT_EQ(idx_scan->low_key.value(),  Value(static_cast<int32_t>(5)));
    EXPECT_EQ(idx_scan->high_key.value(), Value(static_cast<int32_t>(5)));
    EXPECT_EQ(residual.size(), 1u);  // `other > 10` preserved for Filter
}

// Two indexes (a, b), predicates on both; the index with lower cost wins.
// idx_a: sel=1e-4 -> matching=100 -> idx_cost=407
// idx_b: sel=1e-2 -> matching=10000 -> idx_cost=40106
// idx_a wins; idx_b's predicate stays in residual.
TEST_F(IndexSelectorStrategyTest, MultipleIndexesBestChosen) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    auto* ti = catalog.CreateTable("t", Schema({
        Column("a", TypeId::INTEGER),
        Column("b", TypeId::INTEGER)
    }));
    ASSERT_NE(ti, nullptr);
    ti->stats.row_count = 1000000;
    ti->stats.distinct_counts["a"] = 10000;
    ti->stats.distinct_counts["b"] = 100;
    (void)catalog.CreateIndex("idx_t_a", "t", "a");
    (void)catalog.CreateIndex("idx_t_b", "t", "b");

    std::vector<std::unique_ptr<Expression>> preds;
    preds.push_back(MakeBinOp(BinaryOp::Op::EQ, MakeColRef("a"), MakeLit(5)));
    preds.push_back(MakeBinOp(BinaryOp::Op::EQ, MakeColRef("b"), MakeLit(7)));

    std::vector<std::unique_ptr<Expression>> residual;
    auto scan = IndexSelector::SelectScanStrategy("t", ti->schema, preds, residual, &catalog);

    ASSERT_NE(scan, nullptr);
    ASSERT_EQ(scan->type, PlanNodeType::INDEX_SCAN);
    auto* idx_scan = dynamic_cast<IndexScanPlanNode*>(scan.get());
    ASSERT_NE(idx_scan, nullptr);
    EXPECT_EQ(idx_scan->index_name, "idx_t_a");
    ASSERT_TRUE(idx_scan->low_key.has_value());
    EXPECT_EQ(idx_scan->low_key.value(), Value(static_cast<int32_t>(5)));
    EXPECT_EQ(residual.size(), 1u);  // b=7 must still be applied via Filter
}

// Sweep distinct counts; the cost model's actual crossover (~0.26%) should
// produce SeqScan for distinct=100 and IndexScan for distinct=1000/10000.
TEST_F(IndexSelectorStrategyTest, CostCrossoverPoint) {
    struct Case {
        uint64_t      distinct;
        PlanNodeType  expected;
    };
    const Case cases[] = {
        {100,   PlanNodeType::SEQ_SCAN},
        {1000,  PlanNodeType::INDEX_SCAN},
        {10000, PlanNodeType::INDEX_SCAN},
    };

    for (const auto& c : cases) {
        // Each iteration gets a fresh DB file + BPM to avoid catalog reuse issues.
        std::filesystem::remove(test_file_);
        auto bundle = MakeBPM(test_file_);
        Catalog catalog(bundle.bpm.get());

        auto* ti = catalog.CreateTable("t", Schema({Column("id", TypeId::INTEGER)}));
        ASSERT_NE(ti, nullptr);
        ti->stats.row_count = 1000000;
        ti->stats.distinct_counts["id"] = c.distinct;
        (void)catalog.CreateIndex("idx_t_id", "t", "id");

        std::vector<std::unique_ptr<Expression>> preds;
        preds.push_back(MakeBinOp(BinaryOp::Op::EQ, MakeColRef("id"), MakeLit(1)));

        std::vector<std::unique_ptr<Expression>> residual;
        auto scan = IndexSelector::SelectScanStrategy("t", ti->schema, preds, residual, &catalog);

        ASSERT_NE(scan, nullptr);
        EXPECT_EQ(scan->type, c.expected)
            << "distinct=" << c.distinct
            << " should have produced "
            << (c.expected == PlanNodeType::SEQ_SCAN ? "SEQ_SCAN" : "INDEX_SCAN");
    }
}

// Literal on the left (`42 = id`) must normalize to the same IndexScan as `id = 42`.
TEST_F(IndexSelectorStrategyTest, LiteralOnLeftIsNormalized) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    auto* ti = catalog.CreateTable("t", Schema({Column("id", TypeId::INTEGER)}));
    ASSERT_NE(ti, nullptr);
    ti->stats.row_count = 1000000;
    ti->stats.distinct_counts["id"] = 10000;
    (void)catalog.CreateIndex("idx_t_id", "t", "id");

    std::vector<std::unique_ptr<Expression>> preds;
    preds.push_back(MakeBinOp(BinaryOp::Op::EQ, MakeLit(42), MakeColRef("id")));

    std::vector<std::unique_ptr<Expression>> residual;
    auto scan = IndexSelector::SelectScanStrategy("t", ti->schema, preds, residual, &catalog);

    ASSERT_NE(scan, nullptr);
    ASSERT_EQ(scan->type, PlanNodeType::INDEX_SCAN);
    auto* idx_scan = dynamic_cast<IndexScanPlanNode*>(scan.get());
    ASSERT_NE(idx_scan, nullptr);
    ASSERT_TRUE(idx_scan->low_key.has_value());
    ASSERT_TRUE(idx_scan->high_key.has_value());
    EXPECT_EQ(idx_scan->low_key.value(),  Value(static_cast<int32_t>(42)));
    EXPECT_EQ(idx_scan->high_key.value(), Value(static_cast<int32_t>(42)));
    EXPECT_TRUE(residual.empty());
}

}  // namespace shilmandb
