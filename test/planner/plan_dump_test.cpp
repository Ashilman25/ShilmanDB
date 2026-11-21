// Golden plans for the 8 TPC-H queries at SF=0.01, row and columnar storage,
// each with and without the 8 indexes, for tuple and vectorized execution.
// Any change to such a plan fails PlansMatchGolden.
// After a plan change I mean to make, rewrite the golden with
//   SHILMANDB_WRITE_GOLDEN=1 ./build-debug/test/plan_dump_test --gtest_filter=PlanDumpTest.PlansMatchGolden
// and review its diff before keeping it. That run fails on purpose;
// rerun without the variable to check the new golden.

#include <gtest/gtest.h>
#include <unistd.h>

#include "engine/database.hpp"
#include "parser/parser.hpp"
#include "planner/plan_node.hpp"
#include "planner/planner.hpp"
#include "planner/plan_dump.hpp"
#include "tpch/tpch_workload.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifndef PLAN_DUMP_GOLDEN_PATH
#error "PLAN_DUMP_GOLDEN_PATH must name test/planner/golden/tpch_plans.txt"
#endif

namespace shilmandb {
namespace {

constexpr size_t kPoolSize = 4096;

// CTest runs from build-debug/test/, so the data is usually two levels up.
std::string FindDataDir() {
    const std::vector<std::string> candidates = {
        "bench/tpch_data/sf0.01/",
        "../bench/tpch_data/sf0.01/",
        "../../bench/tpch_data/sf0.01/",
    };
    for (const auto& dir : candidates) {
        if (std::filesystem::exists(dir + "lineitem.tbl")) return dir;
    }
    return {};
}

// True when SHILMANDB_WRITE_GOLDEN=1, which rewrites the golden instead of checking it.
bool WritingGolden() {
    const char* flag = std::getenv("SHILMANDB_WRITE_GOLDEN");
    return flag != nullptr && std::string(flag) == "1";
}

// The whole file as bytes; empty when it can't be read.
std::string ReadFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

// A plan, its join decision, and its step nodes (pointers into the plan).
struct PlannedQuery {
    std::unique_ptr<PlanNode> plan;
    JoinOrderInfo join;
    std::vector<const PlanNode*> steps;
};

// Plans sql with the database's own planner for these options, as ExecuteSQL does, running nothing.
PlannedQuery PlanThroughDatabase(const Database& db, const std::string& sql, const QueryOptions& options) {
    Parser parser(sql);
    auto stmt = parser.Parse();
    Planner planner = db.MakePlanner(options);
    PlannedQuery out;
    out.plan = planner.Plan(std::move(*stmt));
    out.join = planner.GetJoinOrderInfo();
    out.steps = planner.GetStepNodes();
    return out;
}

// Every field of two join decisions, doubles compared exactly.
::testing::AssertionResult SameJoinInfo(const JoinOrderInfo& a, const JoinOrderInfo& b) {
    if (a.tables != b.tables) return ::testing::AssertionFailure() << "tables differ";
    if (a.est_scan_rows != b.est_scan_rows) return ::testing::AssertionFailure() << "est_scan_rows differ";
    if (a.edges != b.edges) return ::testing::AssertionFailure() << "edges differ";
    if (a.order != b.order) return ::testing::AssertionFailure() << "order differs";
    if (a.source != b.source) return ::testing::AssertionFailure() << "source differs";
    if (a.model_order != b.model_order) return ::testing::AssertionFailure() << "model_order differs";
    if (a.est_step_rows != b.est_step_rows) return ::testing::AssertionFailure() << "est_step_rows differ";
    if (a.est_cost != b.est_cost) return ::testing::AssertionFailure() << "est_cost differs";
    return ::testing::AssertionSuccess();
}

// Nodes of this type anywhere in the plan.
size_t CountNodes(const PlanNode& node, PlanNodeType type) {
    size_t count = node.type == type ? 1 : 0;
    for (const auto& child : node.children) count += CountNodes(*child, type);
    return count;
}

// Hash and nested-loop joins anywhere in the plan.
size_t CountJoinNodes(const PlanNode& node) {
    return CountNodes(node, PlanNodeType::HASH_JOIN) + CountNodes(node, PlanNodeType::NESTED_LOOP_JOIN);
}

// Each join holds the previous step's estimate left and the new table's right.
void ExpectJoinEstimates(const PlannedQuery& planned, const std::string& what) {
    const JoinOrderInfo& join = planned.join;
    ASSERT_EQ(planned.steps.size(), join.order.size()) << what;
    ASSERT_EQ(join.est_step_rows.size(), join.order.size()) << what;
    EXPECT_EQ(join.est_step_rows[0], join.est_scan_rows[static_cast<size_t>(join.order[0])]) << what;
    EXPECT_EQ(CountJoinNodes(*planned.plan), join.order.size() - 1) << what;

    for (size_t k = 1; k < planned.steps.size(); ++k) {
        const std::string step = what + " step " + std::to_string(k);
        const PlanNode* node = planned.steps[k];
        // A post-join filter wraps the hash join; the estimates sit on the join itself.
        if (node->type == PlanNodeType::FILTER) {
            node = node->children[0].get();
            ASSERT_EQ(node->type, PlanNodeType::HASH_JOIN) << step;
        }

        double left = 0.0;
        double right = 0.0;
        if (node->type == PlanNodeType::HASH_JOIN) {
            const auto* hash = static_cast<const HashJoinPlanNode*>(node);
            left = hash->est_left_rows;
            right = hash->est_right_rows;
        } else {
            ASSERT_EQ(node->type, PlanNodeType::NESTED_LOOP_JOIN) << step;
            const auto* loop = static_cast<const NestedLoopJoinPlanNode*>(node);
            left = loop->est_left_rows;
            right = loop->est_right_rows;
        }
        ASSERT_EQ(node->children.size(), 2u) << step;
        EXPECT_EQ(node->children[0].get(), planned.steps[k - 1]) << step << ": left child isn't the previous step";
        EXPECT_EQ(left, join.est_step_rows[k - 1]) << step;
        EXPECT_EQ(right, join.est_scan_rows[static_cast<size_t>(join.order[k])]) << step;
        EXPECT_GE(left, 0.0) << step;
        EXPECT_GE(right, 0.0) << step;
    }
}

// How far apart an estimate and a measured count are, as a ratio >= 1.
double Ratio(double estimated, double measured) {
    return std::max(estimated / measured, measured / estimated);
}

// Splits golden text into (header, body) blocks; a header line starts with "=== ".
std::vector<std::pair<std::string, std::string>> SplitBlocks(const std::string& text) {
    std::vector<std::pair<std::string, std::string>> blocks;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("=== ", 0) == 0) {
            blocks.emplace_back(line, "");
        } else if (!blocks.empty()) {
            blocks.back().second += line + "\n";
        }
    }
    return blocks;
}

// The first line where two texts differ, for failure messages.
std::string FirstDifference(const std::string& want, const std::string& got) {
    std::istringstream a(want);
    std::istringstream b(got);
    std::string la;
    std::string lb;
    for (int n = 1;; ++n) {
        const bool more_a = static_cast<bool>(std::getline(a, la));
        const bool more_b = static_cast<bool>(std::getline(b, lb));
        if (!more_a && !more_b) return "no difference";
        if (!more_a || !more_b || la != lb) {
            return "line " + std::to_string(n) + "\n  golden: " + (more_a ? la : "<end>") +
                   "\n  now:    " + (more_b ? lb : "<end>");
        }
    }
}

class PlanDumpTest : public ::testing::Test {
protected:
    std::string data_dir_;
    std::vector<std::string> files_;

    void SetUp() override {
        data_dir_ = FindDataDir();
        if (data_dir_.empty()) {
            GTEST_SKIP() << "TPC-H SF=0.01 data not found. Run bench/tpch/generate_tpch_data.sh first.";
        }
    }

    void TearDown() override {
        for (const auto& file : files_) std::filesystem::remove(file);
    }

    // A fresh pid-named database with the 7 tables loaded, no indexes yet.
    std::unique_ptr<Database> Load(StorageKind storage) {
        const std::string what = storage == StorageKind::COLUMNAR ? "col" : "row";
        const std::string file = (std::filesystem::temp_directory_path() /
                                  ("shilmandb_plan_dump_test_" + what + "_" + std::to_string(::getpid()) + ".db"))
                                     .string();
        std::filesystem::remove(file);
        files_.push_back(file);

        auto db = std::make_unique<Database>(file, kPoolSize);
        LoadOptions options;
        options.storage = storage;
        tpch::LoadTables(*db, data_dir_, options);
        return db;
    }

    // Builds the 8 TPC-H indexes, as load_tpch --create-indexes does.
    static void CreateIndexes(Database& db) {
        for (const auto& idx : tpch::IndexDescs()) {
            ASSERT_NE(db.GetCatalog()->CreateIndex(idx.index_name, idx.table_name, idx.column_name), nullptr)
                << idx.index_name;
        }
    }
};

}  // namespace

// Today's tuple and vectorized plans, dumped, must equal the golden file block for block.
TEST_F(PlanDumpTest, PlansMatchGolden) {
    std::string text;
    size_t dumps = 0;

    for (const StorageKind storage : {StorageKind::ROW, StorageKind::COLUMNAR}) {
        auto db = Load(storage);
        // Indexes come second: building them changes no table's stats.
        for (const bool indexes : {false, true}) {
            if (indexes) CreateIndexes(*db);
            for (const auto& q : tpch::Queries()) {
                const std::string header = std::string("=== ") + (storage == StorageKind::COLUMNAR ? "columnar" : "row") +
                                           (indexes ? " indexes " : " no-indexes ") + q.tag + " ===";

                std::string block;
                JoinOrderInfo tuple_join;
                for (const ExecutionMode mode : {ExecutionMode::TUPLE, ExecutionMode::VECTORIZED}) {
                    const PlannedQuery planned = PlanThroughDatabase(*db, q.sql, QueryOptions{mode});
                    const std::string dump = test::DumpPlan(*planned.plan);
                    ++dumps;

                    if (mode == ExecutionMode::TUPLE) {
                        block = dump;
                        tuple_join = planned.join;
                    } else {
                        EXPECT_EQ(block, dump) << header << ": vectorized plan differs from tuple";
                        // The dump leaves out estimates, so compare the join decisions too.
                        EXPECT_TRUE(SameJoinInfo(tuple_join, planned.join)) << header;
                    }
                }
                text += header + "\n" + block + "\n";
            }
        }
    }
    EXPECT_EQ(dumps, 64u);

    const std::string path = PLAN_DUMP_GOLDEN_PATH;
    if (WritingGolden()) {
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << text;
        ASSERT_TRUE(out.good()) << "could not write " << path;
        std::cout << "wrote " << path << "\n";
        // A rewrite never passes, so a variable left set can't hide a changed plan.
        ADD_FAILURE() << "golden rewritten; rerun without SHILMANDB_WRITE_GOLDEN";
    }

    ASSERT_TRUE(std::filesystem::exists(path)) << path << " is missing; write it with SHILMANDB_WRITE_GOLDEN=1";
    const std::string golden = ReadFile(path);
    const auto want = SplitBlocks(golden);
    const auto got = SplitBlocks(text);
    EXPECT_EQ(got.size(), 32u);
    ASSERT_EQ(want.size(), got.size()) << "golden has " << want.size() << " blocks, now " << got.size();
    for (size_t i = 0; i < want.size(); ++i) {
        ASSERT_EQ(want[i].first, got[i].first) << "block " << i;
        EXPECT_EQ(want[i].second, got[i].second) << got[i].first << " " << FirstDifference(want[i].second, got[i].second);
    }
    EXPECT_TRUE(golden == text) << "golden file differs from today's dumps outside the blocks";
}

// A join node with no planner estimate says so with -1.
TEST_F(PlanDumpTest, JoinNodesDefaultToUnknownEstimates) {
    const HashJoinPlanNode hash(Schema{}, nullptr);
    const NestedLoopJoinPlanNode loop(Schema{}, nullptr);
    EXPECT_EQ(hash.est_left_rows, -1.0);
    EXPECT_EQ(hash.est_right_rows, -1.0);
    EXPECT_EQ(loop.est_left_rows, -1.0);
    EXPECT_EQ(loop.est_right_rows, -1.0);
}

// Q5's join nodes carry the join decision's estimates in every mode; so do other joins.
TEST_F(PlanDumpTest, JoinNodesCarryEstimates) {
    auto db = Load(StorageKind::COLUMNAR);
    const auto q5 = tpch::FindQuery("Q5");
    ASSERT_TRUE(q5.has_value());

    for (const ExecutionMode mode : {ExecutionMode::TUPLE, ExecutionMode::VECTORIZED, ExecutionMode::PARALLEL}) {
        const std::string what = "Q5 mode " + std::to_string(static_cast<int>(mode));
        const PlannedQuery planned = PlanThroughDatabase(*db, q5->sql, QueryOptions{mode});
        ASSERT_EQ(planned.join.order.size(), 5u) << what;
        ExpectJoinEstimates(planned, what);
    }

    // Every multi-table query, plus a nested-loop join, with the switch off and on.
    std::vector<std::pair<std::string, std::string>> queries;
    for (const auto& q : tpch::Queries()) queries.emplace_back(q.tag, q.sql);
    queries.emplace_back("nested loop", "SELECT n_name, r_name FROM nation, region WHERE n_regionkey < r_regionkey");
    size_t nested_loops = 0;
    for (const auto& [tag, sql] : queries) {
        for (const ExecutionMode mode : {ExecutionMode::TUPLE, ExecutionMode::PARALLEL}) {
            const PlannedQuery planned = PlanThroughDatabase(*db, sql, QueryOptions{mode});
            if (planned.join.order.size() < 2) continue;
            ExpectJoinEstimates(planned, tag + " mode " + std::to_string(static_cast<int>(mode)));
            nested_loops += CountNodes(*planned.plan, PlanNodeType::NESTED_LOOP_JOIN);
        }
    }
    EXPECT_EQ(nested_loops, 2u) << "the nested-loop query should plan a nested-loop join in both modes";
}

// Q10's WHERE joins: off, today's decision exactly; on, every step estimate within 10x of measured.
TEST_F(PlanDumpTest, WhereJoinsCountInParallelPlanning) {
    // Today's Q10 decision at SF=0.01, recorded before parallel planning existed (row and columnar agree).
    const std::vector<int> today_order = {3, 0, 1, 2};
    const std::vector<TableEdge> today_edges = {{0, 1}, {0, 3}, {1, 2}};
    const std::vector<double> today_scan_rows = {1500, 574, 20058, 25};
    const std::vector<double> today_step_rows = {25, 37500, 21525000, 431748450000};
    const double today_cost = 431770012525;

    const auto q10 = tpch::FindQuery("Q10");
    ASSERT_TRUE(q10.has_value());

    for (const StorageKind storage : {StorageKind::ROW, StorageKind::COLUMNAR}) {
        const std::string name = storage == StorageKind::COLUMNAR ? "columnar" : "row";
        auto db = Load(storage);

        // Off: exactly today's order and estimates.
        for (const ExecutionMode mode : {ExecutionMode::TUPLE, ExecutionMode::VECTORIZED}) {
            const std::string what = name + " mode " + std::to_string(static_cast<int>(mode));
            const JoinOrderInfo off = db->ExplainJoinOrder(q10->sql, QueryOptions{mode});
            EXPECT_EQ(off.source, JoinOrderSource::CLASSICAL) << what;
            EXPECT_EQ(off.order, today_order) << what;
            EXPECT_EQ(off.edges, today_edges) << what;
            EXPECT_EQ(off.est_scan_rows, today_scan_rows) << what;
            EXPECT_EQ(off.est_step_rows, today_step_rows) << what;
            EXPECT_EQ(off.est_cost, today_cost) << what;
        }

        // On: WHERE joins price the orders; tables, edges and scan estimates are unchanged.
        const JoinOrderInfo on = db->ExplainJoinOrder(q10->sql, QueryOptions{ExecutionMode::PARALLEL});
        EXPECT_EQ(on.source, JoinOrderSource::CLASSICAL) << name;
        EXPECT_EQ(on.edges, today_edges) << name;
        EXPECT_EQ(on.est_scan_rows, today_scan_rows) << name;
        ASSERT_EQ(on.est_step_rows.size(), 4u) << name;

        // The order may differ from today's, so measure that order's steps in tuple mode.
        QueryOptions forced;
        forced.analyze = true;
        forced.forced_join_order = on.order;
        const QueryResult run = db->ExecuteSQL(q10->sql, forced);
        ASSERT_EQ(run.profile.step_rows.size(), 4u) << name;

        std::string order_text;
        for (int slot : on.order) order_text += " " + std::to_string(slot);
        bool today_far_off = false;
        for (size_t k = 0; k < 4; ++k) {
            const double measured = static_cast<double>(run.profile.step_rows[k]);
            const double estimated = on.est_step_rows[k];
            const double estimated_off = run.profile.join.est_step_rows[k];
            const std::string what = name + " order" + order_text + " step " + std::to_string(k);
            ASSERT_GT(measured, 0.0) << what;
            EXPECT_LE(Ratio(estimated, measured), 10.0) << what << ": estimated " << estimated << " (switch off "
                                                        << estimated_off << "), measured " << measured;
            if (Ratio(estimated_off, measured) > 10.0) today_far_off = true;
        }
        // Without the WHERE joins the same order prices as cross products, past the bound.
        EXPECT_TRUE(today_far_off) << name;

        // Parallel planning only picks the order: tuple mode forced to it plans the same tree.
        // Q19's OR is factored in parallel planning; OrFactoringTest.Q19NewModePlan checks it.
        for (const auto& q : tpch::Queries()) {
            if (q.tag == "Q19") continue;
            const PlannedQuery parallel = PlanThroughDatabase(*db, q.sql, QueryOptions{ExecutionMode::PARALLEL});
            QueryOptions same_order;
            same_order.forced_join_order = parallel.join.order;
            const PlannedQuery tuple = PlanThroughDatabase(*db, q.sql, same_order);
            EXPECT_EQ(test::DumpPlan(*tuple.plan), test::DumpPlan(*parallel.plan)) << name << " " << q.tag;
        }
    }
}

}  // namespace shilmandb
