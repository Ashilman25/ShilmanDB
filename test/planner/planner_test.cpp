#include <gtest/gtest.h>
#include "planner/planner.hpp"
#include "parser/parser.hpp"
#include "catalog/catalog.hpp"
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "storage/disk_manager.hpp"
#include "common/exception.hpp"
#include <filesystem>
#include <memory>

namespace shilmandb {

class PlannerTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() / "shilmandb_planner_test.db").string();
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

TEST_F(PlannerTest, SingleTableScan) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    (void)catalog.CreateTable("t", Schema({
        Column("id", TypeId::INTEGER),
        Column("name", TypeId::VARCHAR)
    }));

    Parser parser("SELECT * FROM t");
    auto stmt = parser.Parse();

    Planner planner(&catalog);
    auto plan = planner.Plan(std::move(*stmt));

    // Root: Projection
    ASSERT_NE(plan, nullptr);
    ASSERT_EQ(plan->type, PlanNodeType::PROJECTION);
    EXPECT_EQ(plan->output_schema.GetColumnCount(), 2u);
    EXPECT_EQ(plan->output_schema.GetColumn(0).name, "id");
    EXPECT_EQ(plan->output_schema.GetColumn(1).name, "name");

    // Child: SeqScan
    ASSERT_EQ(plan->children.size(), 1u);
    ASSERT_EQ(plan->children[0]->type, PlanNodeType::SEQ_SCAN);
    auto* scan = dynamic_cast<SeqScanPlanNode*>(plan->children[0].get());
    ASSERT_NE(scan, nullptr);
    EXPECT_EQ(scan->table_name, "t");
}

TEST_F(PlannerTest, UnknownTableThrows) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    Parser parser("SELECT * FROM nonexistent");
    auto stmt = parser.Parse();

    Planner planner(&catalog);
    EXPECT_THROW((void)planner.Plan(std::move(*stmt)), DatabaseException);
}

TEST_F(PlannerTest, FilterPushdown) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    (void)catalog.CreateTable("t", Schema({
        Column("x", TypeId::INTEGER),
        Column("y", TypeId::INTEGER)
    }));

    Parser parser("SELECT * FROM t WHERE x > 5");
    auto stmt = parser.Parse();

    Planner planner(&catalog);
    auto plan = planner.Plan(std::move(*stmt));

    // Projection -> Filter -> SeqScan
    ASSERT_EQ(plan->type, PlanNodeType::PROJECTION);
    ASSERT_EQ(plan->children.size(), 1u);
    ASSERT_EQ(plan->children[0]->type, PlanNodeType::FILTER);

    auto* filter = dynamic_cast<FilterPlanNode*>(plan->children[0].get());
    ASSERT_NE(filter, nullptr);
    ASSERT_NE(filter->predicate, nullptr);

    ASSERT_EQ(filter->children.size(), 1u);
    ASSERT_EQ(filter->children[0]->type, PlanNodeType::SEQ_SCAN);
}

TEST_F(PlannerTest, MultiPredicateAND) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    (void)catalog.CreateTable("t", Schema({
        Column("x", TypeId::INTEGER),
        Column("y", TypeId::INTEGER)
    }));

    Parser parser("SELECT * FROM t WHERE x > 5 AND y < 10");
    auto stmt = parser.Parse();

    Planner planner(&catalog);
    auto plan = planner.Plan(std::move(*stmt));

    // Projection -> Filter -> SeqScan
    // Both predicates should be pushed down to the same table
    ASSERT_EQ(plan->type, PlanNodeType::PROJECTION);
    ASSERT_EQ(plan->children.size(), 1u);
    ASSERT_EQ(plan->children[0]->type, PlanNodeType::FILTER);

    auto* filter = dynamic_cast<FilterPlanNode*>(plan->children[0].get());
    ASSERT_NE(filter, nullptr);
    // Combined predicate should be an AND node
    auto* and_pred = dynamic_cast<BinaryOp*>(filter->predicate.get());
    ASSERT_NE(and_pred, nullptr);
    EXPECT_EQ(and_pred->op, BinaryOp::Op::AND);

    ASSERT_EQ(filter->children.size(), 1u);
    EXPECT_EQ(filter->children[0]->type, PlanNodeType::SEQ_SCAN);
}

TEST_F(PlannerTest, TwoTableHashJoin) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    (void)catalog.CreateTable("a", Schema({
        Column("id", TypeId::INTEGER),
        Column("val", TypeId::INTEGER)
    }));
    (void)catalog.CreateTable("b", Schema({
        Column("id", TypeId::INTEGER),
        Column("aid", TypeId::INTEGER)
    }));

    Parser parser("SELECT * FROM a JOIN b ON a.id = b.aid");
    auto stmt = parser.Parse();

    Planner planner(&catalog);
    auto plan = planner.Plan(std::move(*stmt));

    // Projection -> HashJoin(SeqScan, SeqScan)
    ASSERT_EQ(plan->type, PlanNodeType::PROJECTION);
    EXPECT_EQ(plan->output_schema.GetColumnCount(), 4u);

    ASSERT_EQ(plan->children.size(), 1u);
    ASSERT_EQ(plan->children[0]->type, PlanNodeType::HASH_JOIN);

    auto* join = dynamic_cast<HashJoinPlanNode*>(plan->children[0].get());
    ASSERT_NE(join, nullptr);
    ASSERT_NE(join->join_predicate, nullptr);

    ASSERT_EQ(join->children.size(), 2u);
    EXPECT_EQ(join->children[0]->type, PlanNodeType::SEQ_SCAN);
    EXPECT_EQ(join->children[1]->type, PlanNodeType::SEQ_SCAN);
}

TEST_F(PlannerTest, ThreeTableJoinOrder) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    // Create 3 tables with deliberately skewed sizes
    auto* ta = catalog.CreateTable("a", Schema({
        Column("id", TypeId::INTEGER),
        Column("val", TypeId::INTEGER)
    }));
    auto* tb = catalog.CreateTable("b", Schema({
        Column("id", TypeId::INTEGER),
        Column("aid", TypeId::INTEGER)
    }));
    auto* tc = catalog.CreateTable("c", Schema({
        Column("id", TypeId::INTEGER),
        Column("bid", TypeId::INTEGER)
    }));

    // Manually set stats (UpdateTableStats requires actual rows)
    ta->stats.row_count = 1000;
    ta->stats.distinct_counts["id"] = 1000;
    ta->stats.distinct_counts["val"] = 500;

    tb->stats.row_count = 10;
    tb->stats.distinct_counts["id"] = 10;
    tb->stats.distinct_counts["aid"] = 10;

    tc->stats.row_count = 100;
    tc->stats.distinct_counts["id"] = 100;
    tc->stats.distinct_counts["bid"] = 10;

    Parser parser("SELECT * FROM a JOIN b ON a.id = b.aid JOIN c ON b.id = c.bid");
    auto stmt = parser.Parse();

    Planner planner(&catalog);
    auto plan = planner.Plan(std::move(*stmt));

    // Should be Projection -> HashJoin -> HashJoin at root
    ASSERT_EQ(plan->type, PlanNodeType::PROJECTION);
    ASSERT_EQ(plan->children.size(), 1u);
    ASSERT_EQ(plan->children[0]->type, PlanNodeType::HASH_JOIN);

    // The optimizer should start with the smallest table (b, 10 rows)
    // Verify we get a two-level join tree
    auto* outer_join = dynamic_cast<HashJoinPlanNode*>(plan->children[0].get());
    ASSERT_NE(outer_join, nullptr);
    ASSERT_EQ(outer_join->children.size(), 2u);

    // One child should be a HashJoin (the inner join of two tables)
    // The other should be a SeqScan
    bool has_inner_join = outer_join->children[0]->type == PlanNodeType::HASH_JOIN ||
                          outer_join->children[1]->type == PlanNodeType::HASH_JOIN;
    EXPECT_TRUE(has_inner_join);
}

TEST_F(PlannerTest, AggregateGroupBy) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    (void)catalog.CreateTable("t", Schema({
        Column("x", TypeId::INTEGER),
        Column("y", TypeId::INTEGER)
    }));

    Parser parser("SELECT x, COUNT(*) FROM t GROUP BY x");
    auto stmt = parser.Parse();

    Planner planner(&catalog);
    auto plan = planner.Plan(std::move(*stmt));

    // Projection -> Aggregate -> SeqScan
    ASSERT_EQ(plan->type, PlanNodeType::PROJECTION);
    ASSERT_EQ(plan->children.size(), 1u);
    ASSERT_EQ(plan->children[0]->type, PlanNodeType::AGGREGATE);

    auto* agg = dynamic_cast<AggregatePlanNode*>(plan->children[0].get());
    ASSERT_NE(agg, nullptr);
    EXPECT_EQ(agg->group_by_exprs.size(), 1u);
    EXPECT_EQ(agg->aggregate_funcs.size(), 1u);
    EXPECT_EQ(agg->aggregate_funcs[0], Aggregate::Func::COUNT);

    // Output schema: group-by col (x, INTEGER) + aggregate (count_star, INTEGER)
    EXPECT_EQ(agg->output_schema.GetColumnCount(), 2u);
    EXPECT_EQ(agg->output_schema.GetColumn(0).name, "x");
    EXPECT_EQ(agg->output_schema.GetColumn(0).type, TypeId::INTEGER);
    EXPECT_EQ(agg->output_schema.GetColumn(1).name, "count_star");
    EXPECT_EQ(agg->output_schema.GetColumn(1).type, TypeId::INTEGER);

    ASSERT_EQ(agg->children.size(), 1u);
    EXPECT_EQ(agg->children[0]->type, PlanNodeType::SEQ_SCAN);
}

TEST_F(PlannerTest, GroupByWithoutAggregates) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    (void)catalog.CreateTable("t", Schema({
        Column("x", TypeId::INTEGER),
        Column("y", TypeId::INTEGER)
    }));

    Parser parser("SELECT x FROM t GROUP BY x");
    auto stmt = parser.Parse();

    Planner planner(&catalog);
    auto plan = planner.Plan(std::move(*stmt));

    // Projection -> Aggregate -> SeqScan
    ASSERT_EQ(plan->type, PlanNodeType::PROJECTION);
    ASSERT_EQ(plan->children.size(), 1u);
    ASSERT_EQ(plan->children[0]->type, PlanNodeType::AGGREGATE);

    auto* agg = dynamic_cast<AggregatePlanNode*>(plan->children[0].get());
    ASSERT_NE(agg, nullptr);
    EXPECT_EQ(agg->group_by_exprs.size(), 1u);
    EXPECT_EQ(agg->aggregate_funcs.size(), 0u);  // no aggregates
}

TEST_F(PlannerTest, HavingFilter) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    (void)catalog.CreateTable("t", Schema({
        Column("x", TypeId::INTEGER),
        Column("y", TypeId::INTEGER)
    }));

    Parser parser("SELECT x, COUNT(*) FROM t GROUP BY x HAVING COUNT(*) > 1");
    auto stmt = parser.Parse();

    Planner planner(&catalog);
    auto plan = planner.Plan(std::move(*stmt));

    // Projection -> Filter (HAVING) -> Aggregate -> SeqScan
    ASSERT_EQ(plan->type, PlanNodeType::PROJECTION);
    ASSERT_EQ(plan->children.size(), 1u);
    ASSERT_EQ(plan->children[0]->type, PlanNodeType::FILTER);

    auto* having = dynamic_cast<FilterPlanNode*>(plan->children[0].get());
    ASSERT_NE(having, nullptr);
    ASSERT_NE(having->predicate, nullptr);

    ASSERT_EQ(having->children.size(), 1u);
    ASSERT_EQ(having->children[0]->type, PlanNodeType::AGGREGATE);

    ASSERT_EQ(having->children[0]->children.size(), 1u);
    EXPECT_EQ(having->children[0]->children[0]->type, PlanNodeType::SEQ_SCAN);
}

TEST_F(PlannerTest, NestedLoopJoinNonEqui) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    (void)catalog.CreateTable("a", Schema({
        Column("id", TypeId::INTEGER),
        Column("val", TypeId::INTEGER)
    }));
    (void)catalog.CreateTable("b", Schema({
        Column("id", TypeId::INTEGER),
        Column("val", TypeId::INTEGER)
    }));

    Parser parser("SELECT * FROM a JOIN b ON a.val > b.val");
    auto stmt = parser.Parse();

    Planner planner(&catalog);
    auto plan = planner.Plan(std::move(*stmt));

    // Projection -> NestedLoopJoin(SeqScan, SeqScan)
    ASSERT_EQ(plan->type, PlanNodeType::PROJECTION);
    ASSERT_EQ(plan->children.size(), 1u);
    ASSERT_EQ(plan->children[0]->type, PlanNodeType::NESTED_LOOP_JOIN);

    auto* nlj = dynamic_cast<NestedLoopJoinPlanNode*>(plan->children[0].get());
    ASSERT_NE(nlj, nullptr);
    ASSERT_NE(nlj->predicate, nullptr);

    ASSERT_EQ(nlj->children.size(), 2u);
    EXPECT_EQ(nlj->children[0]->type, PlanNodeType::SEQ_SCAN);
    EXPECT_EQ(nlj->children[1]->type, PlanNodeType::SEQ_SCAN);
}

TEST_F(PlannerTest, OrderByAndLimit) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    (void)catalog.CreateTable("t", Schema({
        Column("x", TypeId::INTEGER),
        Column("y", TypeId::INTEGER)
    }));

    Parser parser("SELECT * FROM t ORDER BY x LIMIT 10");
    auto stmt = parser.Parse();

    Planner planner(&catalog);
    auto plan = planner.Plan(std::move(*stmt));

    // Limit -> Sort -> Projection -> SeqScan
    ASSERT_EQ(plan->type, PlanNodeType::LIMIT);
    auto* limit = dynamic_cast<LimitPlanNode*>(plan.get());
    ASSERT_NE(limit, nullptr);
    EXPECT_EQ(limit->limit, 10);

    ASSERT_EQ(plan->children.size(), 1u);
    ASSERT_EQ(plan->children[0]->type, PlanNodeType::SORT);

    auto* sort = dynamic_cast<SortPlanNode*>(plan->children[0].get());
    ASSERT_NE(sort, nullptr);
    ASSERT_EQ(sort->order_by.size(), 1u);
    EXPECT_TRUE(sort->order_by[0].ascending);

    ASSERT_EQ(sort->children.size(), 1u);
    EXPECT_EQ(sort->children[0]->type, PlanNodeType::PROJECTION);

    ASSERT_EQ(sort->children[0]->children.size(), 1u);
    EXPECT_EQ(sort->children[0]->children[0]->type, PlanNodeType::SEQ_SCAN);
}

TEST_F(PlannerTest, LimitOnly) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    (void)catalog.CreateTable("t", Schema({
        Column("x", TypeId::INTEGER)
    }));

    Parser parser("SELECT * FROM t LIMIT 5");
    auto stmt = parser.Parse();

    Planner planner(&catalog);
    auto plan = planner.Plan(std::move(*stmt));

    // Limit -> Projection -> SeqScan (no Sort)
    ASSERT_EQ(plan->type, PlanNodeType::LIMIT);
    auto* limit = dynamic_cast<LimitPlanNode*>(plan.get());
    ASSERT_NE(limit, nullptr);
    EXPECT_EQ(limit->limit, 5);

    ASSERT_EQ(plan->children.size(), 1u);
    EXPECT_EQ(plan->children[0]->type, PlanNodeType::PROJECTION);
}

TEST_F(PlannerTest, NoWhereNoJoins) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    (void)catalog.CreateTable("t", Schema({
        Column("a", TypeId::INTEGER),
        Column("b", TypeId::VARCHAR),
        Column("c", TypeId::DECIMAL)
    }));

    Parser parser("SELECT a, b FROM t");
    auto stmt = parser.Parse();

    Planner planner(&catalog);
    auto plan = planner.Plan(std::move(*stmt));

    // Projection -> SeqScan, projected to 2 columns
    ASSERT_EQ(plan->type, PlanNodeType::PROJECTION);
    EXPECT_EQ(plan->output_schema.GetColumnCount(), 2u);
    EXPECT_EQ(plan->output_schema.GetColumn(0).name, "a");
    EXPECT_EQ(plan->output_schema.GetColumn(1).name, "b");
    EXPECT_EQ(plan->output_schema.GetColumn(0).type, TypeId::INTEGER);
    EXPECT_EQ(plan->output_schema.GetColumn(1).type, TypeId::VARCHAR);

    ASSERT_EQ(plan->children.size(), 1u);
    EXPECT_EQ(plan->children[0]->type, PlanNodeType::SEQ_SCAN);
}

TEST_F(PlannerTest, BareAggregateNoGroupBy) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    (void)catalog.CreateTable("t", Schema({
        Column("x", TypeId::INTEGER),
        Column("y", TypeId::INTEGER)
    }));

    Parser parser("SELECT COUNT(*) FROM t");
    auto stmt = parser.Parse();

    Planner planner(&catalog);
    auto plan = planner.Plan(std::move(*stmt));

    // Projection -> Aggregate -> SeqScan
    ASSERT_EQ(plan->type, PlanNodeType::PROJECTION);
    ASSERT_EQ(plan->children.size(), 1u);
    ASSERT_EQ(plan->children[0]->type, PlanNodeType::AGGREGATE);

    auto* agg = dynamic_cast<AggregatePlanNode*>(plan->children[0].get());
    ASSERT_NE(agg, nullptr);
    EXPECT_EQ(agg->group_by_exprs.size(), 0u);  // no GROUP BY
    EXPECT_EQ(agg->aggregate_funcs.size(), 1u);
    EXPECT_EQ(agg->aggregate_funcs[0], Aggregate::Func::COUNT);

    ASSERT_EQ(agg->children.size(), 1u);
    EXPECT_EQ(agg->children[0]->type, PlanNodeType::SEQ_SCAN);
}

TEST_F(PlannerTest, CompoundEquiJoinUsesHashJoin) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    (void)catalog.CreateTable("a", Schema({
        Column("id", TypeId::INTEGER),
        Column("code", TypeId::INTEGER)
    }));
    (void)catalog.CreateTable("b", Schema({
        Column("id", TypeId::INTEGER),
        Column("code", TypeId::INTEGER)
    }));

    Parser parser("SELECT * FROM a JOIN b ON a.id = b.id AND a.code = b.code");
    auto stmt = parser.Parse();

    Planner planner(&catalog);
    auto plan = planner.Plan(std::move(*stmt));

    // Projection -> HashJoin (not NestedLoopJoin)
    ASSERT_EQ(plan->type, PlanNodeType::PROJECTION);
    ASSERT_EQ(plan->children.size(), 1u);
    ASSERT_EQ(plan->children[0]->type, PlanNodeType::HASH_JOIN);

    auto* join = dynamic_cast<HashJoinPlanNode*>(plan->children[0].get());
    ASSERT_NE(join, nullptr);
    ASSERT_NE(join->join_predicate, nullptr);

    // The predicate root should be AND
    auto* and_pred = dynamic_cast<BinaryOp*>(join->join_predicate.get());
    ASSERT_NE(and_pred, nullptr);
    EXPECT_EQ(and_pred->op, BinaryOp::Op::AND);
}

TEST_F(PlannerTest, EmptyFromClauseThrows) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    // Construct a SelectStatement with empty from_clause directly
    SelectStatement stmt;
    auto item = SelectItem();
    item.expr = std::make_unique<StarExpr>();
    stmt.select_list.push_back(std::move(item));

    Planner planner(&catalog);
    EXPECT_THROW((void)planner.Plan(std::move(stmt)), DatabaseException);
}

TEST_F(PlannerTest, IndexScanWhenSelective) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    auto* ti = catalog.CreateTable("t", Schema({Column("id", TypeId::INTEGER)}));
    ASSERT_NE(ti, nullptr);
    ti->stats.row_count = 1'000'000;
    ti->stats.distinct_counts["id"] = 10'000;
    (void)catalog.CreateIndex("idx_t_id", "t", "id");

    Parser parser("SELECT * FROM t WHERE id = 42");
    auto stmt = parser.Parse();

    Planner planner(&catalog);
    auto plan = planner.Plan(std::move(*stmt));

    ASSERT_EQ(plan->type, PlanNodeType::PROJECTION);
    ASSERT_EQ(plan->children.size(), 1u);
    ASSERT_EQ(plan->children[0]->type, PlanNodeType::INDEX_SCAN);

    auto* scan = dynamic_cast<IndexScanPlanNode*>(plan->children[0].get());
    ASSERT_NE(scan, nullptr);
    EXPECT_EQ(scan->table_name, "t");
    EXPECT_EQ(scan->index_name, "idx_t_id");
    ASSERT_TRUE(scan->low_key.has_value());
    ASSERT_TRUE(scan->high_key.has_value());
    EXPECT_EQ(scan->low_key.value(),  Value(static_cast<int32_t>(42)));
    EXPECT_EQ(scan->high_key.value(), Value(static_cast<int32_t>(42)));
}

TEST_F(PlannerTest, IndexScanWithResidualFilter) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    auto* ti = catalog.CreateTable("t", Schema({
        Column("id",    TypeId::INTEGER),
        Column("other", TypeId::INTEGER)
    }));
    ASSERT_NE(ti, nullptr);
    ti->stats.row_count = 1'000'000;
    ti->stats.distinct_counts["id"]    = 10'000;
    ti->stats.distinct_counts["other"] = 100;
    (void)catalog.CreateIndex("idx_t_id", "t", "id");

    Parser parser("SELECT * FROM t WHERE id = 5 AND other > 7");
    auto stmt = parser.Parse();

    Planner planner(&catalog);
    auto plan = planner.Plan(std::move(*stmt));

    ASSERT_EQ(plan->type, PlanNodeType::PROJECTION);
    ASSERT_EQ(plan->children.size(), 1u);
    ASSERT_EQ(plan->children[0]->type, PlanNodeType::FILTER);

    auto* filter = dynamic_cast<FilterPlanNode*>(plan->children[0].get());
    ASSERT_NE(filter, nullptr);
    ASSERT_EQ(filter->children.size(), 1u);
    ASSERT_EQ(filter->children[0]->type, PlanNodeType::INDEX_SCAN);

    auto* scan = dynamic_cast<IndexScanPlanNode*>(filter->children[0].get());
    ASSERT_NE(scan, nullptr);
    EXPECT_EQ(scan->index_name, "idx_t_id");
    ASSERT_TRUE(scan->low_key.has_value());
    ASSERT_TRUE(scan->high_key.has_value());
    EXPECT_EQ(scan->low_key.value(),  Value(static_cast<int32_t>(5)));
    EXPECT_EQ(scan->high_key.value(), Value(static_cast<int32_t>(5)));
}

}  // namespace shilmandb
