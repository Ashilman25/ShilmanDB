#include <gtest/gtest.h>
#include <unistd.h>

#include "engine/database.hpp"
#include "parser/parser.hpp"
#include "planner/plan_node.hpp"
#include "planner/planner.hpp"
#include "tpch_workload.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace shilmandb {

namespace {

// Walks down through any filters to the scan underneath.
const PlanNode* LeafScan(const PlanNode* node) {
    while (!node->children.empty()) node = node->children[0].get();
    return node;
}

// How far apart two row counts are, as a ratio >= 1. Both are floored at one row.
double QError(double estimated, double actual) {
    estimated = std::max(estimated, 1.0);
    actual = std::max(actual, 1.0);
    return std::max(estimated / actual, actual / estimated);
}

}  // namespace

// Plans and estimates on real TPC-H data (SF=0.01) with every TPC-H index built.
class TpchEstimatesTest : public ::testing::Test {
protected:
    std::string test_file_;
    std::unique_ptr<Database> db_;

    // CTest runs from build/test/, so the data is usually two levels up.
    static std::string FindDataDir() {
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

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() /
                      ("shilmandb_tpch_estimates_test_" + std::to_string(::getpid()) + ".db")).string();
        std::filesystem::remove(test_file_);

        const auto data_dir = FindDataDir();
        if (data_dir.empty()) {
            GTEST_SKIP() << "TPC-H SF=0.01 data not found. Run bench/generate_tpch_data.sh first.";
        }

        // 32 MB, enough to hold all of SF=0.01.
        constexpr size_t kPoolSize = 4096;
        db_ = std::make_unique<Database>(test_file_, kPoolSize);

        for (const auto& td : tpch::TableDescs()) {
            db_->LoadTable(td.name, td.schema, data_dir + td.tbl_file);
        }
        for (const auto& idx : tpch::IndexDescs()) {
            (void)db_->GetCatalog()->CreateIndex(idx.index_name, idx.table_name, idx.column_name);
        }
    }

    void TearDown() override {
        db_.reset();
        std::filesystem::remove(test_file_);
    }

    // Plans `sql` and returns the leaf scan of every table slot. The plan must outlive them.
    std::vector<const PlanNode*> LeafScans(const std::string& sql, std::unique_ptr<PlanNode>* plan_out,
                                           Planner* planner) {
        Parser parser(sql);
        auto stmt = parser.Parse();
        *plan_out = planner->Plan(std::move(*stmt));
        std::vector<const PlanNode*> leaves;
        for (const auto* scan : planner->GetScanNodes()) leaves.push_back(LeafScan(scan));
        return leaves;
    }

    // The single leaf scan of a one-table query.
    const PlanNode* OnlyScan(const std::string& sql, std::unique_ptr<PlanNode>* plan_out, Planner* planner) {
        const auto leaves = LeafScans(sql, plan_out, planner);
        EXPECT_EQ(leaves.size(), 1u) << sql;
        return leaves.empty() ? nullptr : leaves[0];
    }

    int32_t CountStar(const std::string& sql) {
        const auto result = db_->ExecuteSQL(sql);
        EXPECT_EQ(result.tuples.size(), 1u) << sql;
        return result.tuples.empty() ? -1 : result.tuples[0].GetValue(result.schema, 0).integer_;
    }
};

// With every index built, none of the 8 queries is cheaper through an index.
TEST_F(TpchEstimatesTest, TpchQueriesScanEveryTable) {
    for (const auto& query : tpch::Queries()) {
        std::unique_ptr<PlanNode> plan;
        Planner planner(db_->GetCatalog());
        const auto leaves = LeafScans(query.sql, &plan, &planner);
        EXPECT_FALSE(leaves.empty()) << query.tag;
        for (const auto* leaf : leaves) {
            EXPECT_EQ(leaf->type, PlanNodeType::SEQ_SCAN) << query.tag << ": a table is read through an index";
        }
    }
}

// Q6's one-year range keeps about 15% of lineitem, far too many rows for an index.
TEST_F(TpchEstimatesTest, Q6YearIsAboutFifteenPercentAndScans) {
    const std::string sql =
        "SELECT COUNT(*) FROM lineitem WHERE l_shipdate >= '1994-01-01' AND l_shipdate < '1995-01-01'";

    const auto info = db_->ExplainJoinOrder(sql);
    ASSERT_EQ(info.est_scan_rows.size(), 1u);
    const auto* table = db_->GetCatalog()->GetTable("lineitem");
    ASSERT_NE(table, nullptr);
    const double fraction = info.est_scan_rows[0] / static_cast<double>(table->stats.row_count);
    EXPECT_GE(fraction, 0.13);
    EXPECT_LE(fraction, 0.17);

    std::unique_ptr<PlanNode> plan;
    Planner planner(db_->GetCatalog());
    const auto* leaf = OnlyScan(sql, &plan, &planner);
    ASSERT_NE(leaf, nullptr);
    EXPECT_EQ(leaf->type, PlanNodeType::SEQ_SCAN);
}

// A one-day range keeps a handful of rows, so the index wins. The three spellings
// (half-open, BETWEEN, strict on both sides) must all pick it and count the same rows.
TEST_F(TpchEstimatesTest, OneDayShipdateRangePicksIndex) {
    // The true count: scan a month and count the rows on that day.
    const std::string month_sql =
        "SELECT l_shipdate FROM lineitem WHERE l_shipdate >= '1995-06-01' AND l_shipdate < '1995-07-01'";
    {
        // The reference count must come from a plain scan, not the index it is checking.
        std::unique_ptr<PlanNode> month_plan;
        Planner month_planner(db_->GetCatalog());
        const auto* month_leaf = OnlyScan(month_sql, &month_plan, &month_planner);
        ASSERT_NE(month_leaf, nullptr);
        ASSERT_EQ(month_leaf->type, PlanNodeType::SEQ_SCAN);
    }
    const auto month = db_->ExecuteSQL(month_sql);
    const Value day = Value::FromString(TypeId::DATE, "1995-06-17");
    int32_t expected = 0;
    for (const auto& tuple : month.tuples) {
        if (tuple.GetValue(month.schema, 0) == day) ++expected;
    }
    ASSERT_GT(expected, 0);

    const std::vector<std::string> predicates = {
        "l_shipdate >= '1995-06-17' AND l_shipdate < '1995-06-18'",
        "l_shipdate BETWEEN '1995-06-17' AND '1995-06-17'",
        "l_shipdate > '1995-06-16' AND l_shipdate < '1995-06-18'",
    };
    for (const auto& predicate : predicates) {
        const std::string sql = "SELECT COUNT(*) FROM lineitem WHERE " + predicate;

        std::unique_ptr<PlanNode> plan;
        Planner planner(db_->GetCatalog());
        const auto* leaf = OnlyScan(sql, &plan, &planner);
        ASSERT_NE(leaf, nullptr) << predicate;
        ASSERT_EQ(leaf->type, PlanNodeType::INDEX_SCAN) << predicate;
        const auto* index_scan = static_cast<const IndexScanPlanNode*>(leaf);
        EXPECT_EQ(index_scan->index_name, "idx_lineitem_shipdate") << predicate;
        ASSERT_TRUE(index_scan->low_key.has_value()) << predicate;
        ASSERT_TRUE(index_scan->high_key.has_value()) << predicate;
        EXPECT_EQ(index_scan->low_key->type_, TypeId::DATE) << predicate;
        EXPECT_EQ(index_scan->high_key->type_, TypeId::DATE) << predicate;

        EXPECT_EQ(CountStar(sql), expected) << predicate;
    }
}

// An equality on a nearly unique key reads one row through its index. The keys are
// BIGINT and the literal is an INTEGER, so the index bounds have to be widened.
TEST_F(TpchEstimatesTest, NearUniqueKeyEqualityPicksIndex) {
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"SELECT c_name FROM customer WHERE c_custkey = 42", "idx_customer_custkey"},
        {"SELECT p_name FROM part WHERE p_partkey = 42", "idx_part_partkey"},
    };
    for (const auto& [sql, index_name] : cases) {
        std::unique_ptr<PlanNode> plan;
        Planner planner(db_->GetCatalog());
        const auto* leaf = OnlyScan(sql, &plan, &planner);
        ASSERT_NE(leaf, nullptr) << sql;
        ASSERT_EQ(leaf->type, PlanNodeType::INDEX_SCAN) << sql;
        EXPECT_EQ(static_cast<const IndexScanPlanNode*>(leaf)->index_name, index_name) << sql;

        EXPECT_EQ(db_->ExecuteSQL(sql).tuples.size(), 1u) << sql;
    }
}

// Each table's estimated row count after its own filters is within 2x of what the scan returned.
TEST_F(TpchEstimatesTest, ScanEstimatesWithinTwoXOfActual) {
    constexpr double kMaxQError = 2.0;

    for (const auto& query : tpch::Queries()) {
        QueryOptions options;
        options.analyze = true;
        const auto result = db_->ExecuteSQL(query.sql, options);
        const auto& info = result.profile.join;

        ASSERT_EQ(result.profile.scan_rows.size(), info.order.size()) << query.tag;
        double worst = 1.0;
        for (size_t k = 0; k < info.order.size(); ++k) {
            const auto slot = static_cast<size_t>(info.order[k]);
            ASSERT_LT(slot, info.est_scan_rows.size()) << query.tag;
            const double estimated = info.est_scan_rows[slot];
            const double actual = static_cast<double>(result.profile.scan_rows[k]);
            const double q = QError(estimated, actual);
            worst = std::max(worst, q);
            EXPECT_LE(q, kMaxQError) << query.tag << " " << info.tables[slot] << ": estimated " << estimated
                                     << " rows, actual " << actual;
        }
        std::cout << "[ q-error ] " << query.tag << " worst scan q-error " << worst << "\n";
    }
}

}  // namespace shilmandb
