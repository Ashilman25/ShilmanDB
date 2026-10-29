#include <gtest/gtest.h>
#include <string>
#include <unistd.h>
#include "planner/planner.hpp"
#include "parser/parser.hpp"
#include "catalog/catalog.hpp"
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "storage/disk_manager.hpp"
#include "common/exception.hpp"
#include <algorithm>
#include <filesystem>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

namespace shilmandb {

class ForcedJoinOrderTest : public ::testing::Test {
protected:
    std::string test_file_;

    static constexpr const char* kJoinSql =
        "SELECT a.a_id FROM a JOIN b ON a.a_id = b.b_a JOIN c ON b.b_id = c.c_b";
    static constexpr const char* kWhereSql =
        "SELECT a.a_id FROM a, b, c WHERE a.a_id = b.b_a AND b.b_id = c.c_b";

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() /
                      ("shilmandb_forced_join_order_test_" + std::to_string(::getpid()) + ".db")).string();
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

    // Three empty tables chained a <- b <- c.
    static void CreateTables(Catalog& catalog) {
        (void)catalog.CreateTable("a", Schema({Column("a_id", TypeId::INTEGER), Column("a_x", TypeId::INTEGER)}));
        (void)catalog.CreateTable("b", Schema({Column("b_id", TypeId::INTEGER), Column("b_a", TypeId::INTEGER)}));
        (void)catalog.CreateTable("c", Schema({Column("c_id", TypeId::INTEGER), Column("c_b", TypeId::INTEGER)}));
    }

    static auto PlanSql(Planner& planner, const std::string& sql) -> std::unique_ptr<PlanNode> {
        Parser parser(sql);
        auto stmt = parser.Parse();
        return planner.Plan(std::move(*stmt));
    }

    // Scan table names, left to right.
    static auto LeafTables(const PlanNode* node) -> std::vector<std::string> {
        std::vector<std::string> names;
        std::function<void(const PlanNode*)> walk = [&](const PlanNode* n) {
            if (const auto* seq = dynamic_cast<const SeqScanPlanNode*>(n)) names.push_back(seq->table_name);
            if (const auto* idx = dynamic_cast<const IndexScanPlanNode*>(n)) names.push_back(idx->table_name);
            for (const auto& child : n->children) walk(child.get());
        };
        walk(node);
        return names;
    }

    static auto CountNodes(const PlanNode* node, PlanNodeType type) -> int {
        int count = node->type == type ? 1 : 0;
        for (const auto& child : node->children) count += CountNodes(child.get(), type);
        return count;
    }
};

using Edges = std::vector<TableEdge>;

TEST_F(ForcedJoinOrderTest, ForcedOrderIsUsed) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());
    CreateTables(catalog);

    Planner planner(&catalog);
    planner.SetForcedJoinOrder({2, 1, 0});
    auto plan = PlanSql(planner, kJoinSql);

    EXPECT_EQ(LeafTables(plan.get()), (std::vector<std::string>{"c", "b", "a"}));

    const auto& info = planner.GetJoinOrderInfo();
    EXPECT_EQ(info.source, JoinOrderSource::FORCED);
    EXPECT_EQ(info.order, (std::vector<int>{2, 1, 0}));
    EXPECT_EQ(info.tables, (std::vector<std::string>{"a", "b", "c"}));
    EXPECT_EQ(info.edges, (Edges{{0, 1}, {1, 2}}));
    EXPECT_STREQ(ToString(info.source), "forced");
}

TEST_F(ForcedJoinOrderTest, ForcedOrderKeepsHashJoins) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());
    CreateTables(catalog);

    Planner planner(&catalog);
    planner.SetForcedJoinOrder({1, 2, 0});
    auto plan = PlanSql(planner, kJoinSql);

    EXPECT_EQ(CountNodes(plan.get(), PlanNodeType::HASH_JOIN), 2);
    EXPECT_EQ(CountNodes(plan.get(), PlanNodeType::NESTED_LOOP_JOIN), 0);
    EXPECT_EQ(LeafTables(plan.get()), (std::vector<std::string>{"b", "c", "a"}));
}

TEST_F(ForcedJoinOrderTest, ForcedWrongLengthThrows) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());
    CreateTables(catalog);

    Planner planner(&catalog);
    planner.SetForcedJoinOrder({0, 1});
    EXPECT_THROW((void)PlanSql(planner, kJoinSql), DatabaseException);
}

TEST_F(ForcedJoinOrderTest, ForcedNotPermutationThrows) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());
    CreateTables(catalog);

    // {1,0,1} is connected on the chain, so only the permutation check stops it.
    for (const auto& order : {std::vector<int>{0, 0, 1}, std::vector<int>{0, 1, 3}, std::vector<int>{1, 0, 1}}) {
        Planner planner(&catalog);
        planner.SetForcedJoinOrder(order);
        try {
            (void)PlanSql(planner, kJoinSql);
            ADD_FAILURE() << "order " << ::testing::PrintToString(order) << " did not throw";
        } catch (const DatabaseException& e) {
            EXPECT_NE(std::string(e.what()).find("exactly once"), std::string::npos) << e.what();
        }
    }
}

TEST_F(ForcedJoinOrderTest, ForcedCrossProductThrows) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());
    CreateTables(catalog);

    Planner planner(&catalog);
    planner.SetForcedJoinOrder({0, 2, 1});
    EXPECT_THROW((void)PlanSql(planner, kJoinSql), DatabaseException);
}

TEST_F(ForcedJoinOrderTest, WhereJoinsCountForForcing) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());
    CreateTables(catalog);

    Planner planner(&catalog);
    planner.SetForcedJoinOrder({2, 1, 0});
    auto plan = PlanSql(planner, kWhereSql);
    EXPECT_EQ(LeafTables(plan.get()), (std::vector<std::string>{"c", "b", "a"}));
    EXPECT_EQ(planner.GetJoinOrderInfo().edges, (Edges{{0, 1}, {1, 2}}));

    Planner cross(&catalog);
    cross.SetForcedJoinOrder({0, 2, 1});
    EXPECT_THROW((void)PlanSql(cross, kWhereSql), DatabaseException);
}

TEST_F(ForcedJoinOrderTest, ThreeTableWhereConjunctIsNotAnEdge) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());
    CreateTables(catalog);

    Planner planner(&catalog);
    auto plan = PlanSql(planner, "SELECT a.a_id FROM a, b, c WHERE a.a_id = b.b_a AND a.a_x + b.b_id = c.c_b");
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(planner.GetJoinOrderInfo().edges, (Edges{{0, 1}}));
}

TEST_F(ForcedJoinOrderTest, ClassicalDecisionRecorded) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());
    CreateTables(catalog);

    Planner planner(&catalog);
    auto plan = PlanSql(planner, kJoinSql);

    const auto& info = planner.GetJoinOrderInfo();
    EXPECT_EQ(info.source, JoinOrderSource::CLASSICAL);
    EXPECT_STREQ(ToString(info.source), "classical");

    auto sorted = info.order;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(sorted, (std::vector<int>{0, 1, 2}));

    ASSERT_EQ(info.est_step_rows.size(), 3u);
    EXPECT_EQ(info.est_cost, std::accumulate(info.est_step_rows.begin(), info.est_step_rows.end(), 0.0));
    EXPECT_TRUE(info.model_order.empty());

    ASSERT_EQ(planner.GetStepNodes().size(), 3u);
    EXPECT_EQ(planner.GetScanNodes().size(), 3u);
    EXPECT_EQ(planner.GetStepNodes().back()->type, PlanNodeType::HASH_JOIN);

    // A second statement starts a fresh record instead of adding to the old one.
    auto again = PlanSql(planner, kJoinSql);
    EXPECT_EQ(planner.GetStepNodes().size(), 3u);
    EXPECT_EQ(planner.GetScanNodes().size(), 3u);
}

TEST_F(ForcedJoinOrderTest, SingleTableDecisionRecorded) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());
    CreateTables(catalog);

    Planner planner(&catalog);
    auto plan = PlanSql(planner, "SELECT * FROM a");

    const auto& info = planner.GetJoinOrderInfo();
    EXPECT_EQ(info.source, JoinOrderSource::SINGLE_TABLE);
    EXPECT_STREQ(ToString(info.source), "single_table");
    EXPECT_EQ(info.order, (std::vector<int>{0}));
    EXPECT_TRUE(info.edges.empty());
    EXPECT_EQ(info.est_step_rows, (std::vector<double>{0.0}));
    EXPECT_EQ(planner.GetStepNodes().size(), 1u);
    EXPECT_EQ(planner.GetScanNodes().size(), 1u);

    // Forcing works on one table too: {0} is the only valid order.
    Planner forced(&catalog);
    forced.SetForcedJoinOrder({0});
    auto forced_plan = PlanSql(forced, "SELECT * FROM a");
    EXPECT_EQ(forced.GetJoinOrderInfo().source, JoinOrderSource::FORCED);

    for (const auto& order : {std::vector<int>{1}, std::vector<int>{0, 1}}) {
        Planner bad(&catalog);
        bad.SetForcedJoinOrder(order);
        EXPECT_THROW((void)PlanSql(bad, "SELECT * FROM a"), DatabaseException);
    }
}

}  // namespace shilmandb
