#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "catalog/catalog.hpp"
#include "parser/parser.hpp"
#include "planner/column_usage.hpp"
#include "planner/planner.hpp"
#include "storage/disk_manager.hpp"
#include "tpch/tpch_workload.hpp"

namespace shilmandb {
namespace {

using Cols = std::vector<uint32_t>;

class ReadColumnsTest : public ::testing::Test {
protected:
    std::string db_file_;
    std::unique_ptr<DiskManager> disk_;
    std::unique_ptr<BufferPoolManager> bpm_;
    std::unique_ptr<Catalog> catalog_;

    void SetUp() override {
        db_file_ = (std::filesystem::temp_directory_path() /
                    ("shilmandb_read_columns_test_" + std::to_string(::getpid()) + ".db")).string();
        std::filesystem::remove(db_file_);
        disk_ = std::make_unique<DiskManager>(db_file_);
        bpm_ = std::make_unique<BufferPoolManager>(256, disk_.get(), std::make_unique<LRUEvictionPolicy>(256));
        catalog_ = std::make_unique<Catalog>(bpm_.get());

        // Empty tables with made-up stats, so the planner has something to price.
        for (const auto& desc : tpch::TableDescs()) {
            auto* info = catalog_->CreateTable(desc.name, desc.schema);
            ASSERT_NE(info, nullptr);
            info->stats.row_count = 10000;
        }
    }

    void TearDown() override {
        catalog_.reset();
        bpm_.reset();
        disk_.reset();
        std::filesystem::remove(db_file_);
    }

    std::unique_ptr<PlanNode> Plan(const std::string& sql) {
        Parser parser(sql);
        auto stmt = parser.Parse();
        Planner planner(catalog_.get());
        return planner.Plan(std::move(*stmt));
    }

    // Indexes of the named columns in a table's schema, sorted.
    Cols IndexesOf(const std::string& table, const std::vector<std::string>& names) {
        const Schema& schema = catalog_->GetTable(table)->schema;
        Cols out;
        for (const auto& n : names) out.push_back(schema.GetColumnIndex(n));
        std::sort(out.begin(), out.end());
        return out;
    }
};

// Every scan node in the plan, in tree order.
void CollectScans(const PlanNode* node, std::vector<const PlanNode*>& out) {
    if (node->type == PlanNodeType::SEQ_SCAN || node->type == PlanNodeType::INDEX_SCAN) out.push_back(node);
    for (const auto& c : node->children) CollectScans(c.get(), out);
}

std::string TableOf(const PlanNode* scan) {
    if (scan->type == PlanNodeType::SEQ_SCAN) return static_cast<const SeqScanPlanNode*>(scan)->table_name;
    return static_cast<const IndexScanPlanNode*>(scan)->table_name;
}

const std::optional<Cols>& ReadOf(const PlanNode* scan) {
    if (scan->type == PlanNodeType::SEQ_SCAN) return static_cast<const SeqScanPlanNode*>(scan)->read_columns;
    return static_cast<const IndexScanPlanNode*>(scan)->read_columns;
}

// The one scan of table in the plan.
const PlanNode* ScanOf(const PlanNode* root, const std::string& table) {
    std::vector<const PlanNode*> scans;
    CollectScans(root, scans);
    const PlanNode* found = nullptr;
    for (const auto* s : scans) {
        if (TableOf(s) == table) {
            EXPECT_EQ(found, nullptr) << "two scans of " << table;
            found = s;
        }
    }
    EXPECT_NE(found, nullptr) << "no scan of " << table;
    return found;
}

TEST_F(ReadColumnsTest, Q6ReadsItsFourColumns) {
    std::string sql;
    for (const auto& q : tpch::Queries()) {
        if (q.tag == "Q6") sql = q.sql;
    }
    ASSERT_FALSE(sql.empty());

    const auto plan = Plan(sql);
    const auto* scan = ScanOf(plan.get(), "lineitem");
    ASSERT_NE(scan, nullptr);
    ASSERT_TRUE(ReadOf(scan).has_value());
    EXPECT_EQ(*ReadOf(scan), IndexesOf("lineitem", {"l_shipdate", "l_discount", "l_quantity", "l_extendedprice"}));
}

TEST_F(ReadColumnsTest, SelectStarReadsEveryColumn) {
    const auto plan = Plan("SELECT * FROM region");
    const auto* scan = ScanOf(plan.get(), "region");
    ASSERT_NE(scan, nullptr);
    ASSERT_TRUE(ReadOf(scan).has_value());
    EXPECT_EQ(*ReadOf(scan), (Cols{0, 1, 2}));
}

TEST_F(ReadColumnsTest, CountStarReadsNoColumn) {
    const auto plan = Plan("SELECT COUNT(*) FROM nation");
    const auto* scan = ScanOf(plan.get(), "nation");
    ASSERT_NE(scan, nullptr);
    ASSERT_TRUE(ReadOf(scan).has_value());
    EXPECT_TRUE(ReadOf(scan)->empty());
}

TEST_F(ReadColumnsTest, JoinScansListOnlyTheirOwnColumns) {
    const auto plan = Plan(
        "SELECT n_name, COUNT(*) FROM nation n JOIN region r ON n.n_regionkey = r.r_regionkey "
        "WHERE r.r_name = 'ASIA' GROUP BY n_name ORDER BY n_name");
    const auto* nation = ScanOf(plan.get(), "nation");
    const auto* region = ScanOf(plan.get(), "region");
    ASSERT_NE(nation, nullptr);
    ASSERT_NE(region, nullptr);
    ASSERT_TRUE(ReadOf(nation).has_value());
    ASSERT_TRUE(ReadOf(region).has_value());
    EXPECT_EQ(*ReadOf(nation), IndexesOf("nation", {"n_name", "n_regionkey"}));
    EXPECT_EQ(*ReadOf(region), IndexesOf("region", {"r_regionkey", "r_name"}));
}

TEST_F(ReadColumnsTest, EveryTpchQueryListsColumnsPerScan) {
    // Each scan's list is sorted and unique.
    // It holds exactly the columns whose names appear in the query text.
    for (const auto& q : tpch::Queries()) {
        SCOPED_TRACE(q.tag);
        const auto plan = Plan(q.sql);
        std::vector<const PlanNode*> scans;
        CollectScans(plan.get(), scans);
        ASSERT_FALSE(scans.empty());
        for (const auto* scan : scans) {
            ASSERT_TRUE(ReadOf(scan).has_value());
            const Cols& cols = *ReadOf(scan);
            EXPECT_TRUE(std::is_sorted(cols.begin(), cols.end()));
            EXPECT_EQ(std::adjacent_find(cols.begin(), cols.end()), cols.end());
            const Schema& schema = scan->output_schema;
            for (uint32_t c = 0; c < schema.GetColumnCount(); ++c) {
                const bool named = q.sql.find(schema.GetColumn(c).name) != std::string::npos;
                const bool listed = std::find(cols.begin(), cols.end(), c) != cols.end();
                EXPECT_EQ(named, listed) << TableOf(scan) << "." << schema.GetColumn(c).name;
            }
        }
    }
}

TEST_F(ReadColumnsTest, CaseAndArithmeticAreWalked) {
    const auto plan = Plan(
        "SELECT SUM(CASE WHEN l_shipmode = 'MAIL' THEN l_tax ELSE -l_discount END), MAX(l_quantity) "
        "FROM lineitem WHERE NOT (l_linestatus = 'F')");
    const auto* scan = ScanOf(plan.get(), "lineitem");
    ASSERT_NE(scan, nullptr);
    ASSERT_TRUE(ReadOf(scan).has_value());
    EXPECT_EQ(*ReadOf(scan), IndexesOf("lineitem", {"l_shipmode", "l_tax", "l_discount", "l_linestatus", "l_quantity"}));
}

TEST_F(ReadColumnsTest, IndexScanGetsItsList) {
    auto* orders = catalog_->GetTable("orders");
    orders->stats.row_count = 1000000;
    orders->stats.distinct_counts["o_custkey"] = 100000;
    ASSERT_NE(catalog_->CreateIndex("idx_orders_custkey", "orders", "o_custkey"), nullptr);

    const auto plan = Plan("SELECT o_totalprice, o_orderdate FROM orders WHERE o_custkey = 42");
    const auto* scan = ScanOf(plan.get(), "orders");
    ASSERT_NE(scan, nullptr);
    ASSERT_EQ(scan->type, PlanNodeType::INDEX_SCAN);
    ASSERT_TRUE(ReadOf(scan).has_value());
    // The index answers o_custkey = 42 by itself, so nothing above reads o_custkey.
    EXPECT_EQ(*ReadOf(scan), IndexesOf("orders", {"o_totalprice", "o_orderdate"}));

    const auto filtered = Plan("SELECT o_totalprice FROM orders WHERE o_custkey = 42 AND o_orderstatus = 'F'");
    const auto* scan2 = ScanOf(filtered.get(), "orders");
    ASSERT_NE(scan2, nullptr);
    ASSERT_EQ(scan2->type, PlanNodeType::INDEX_SCAN);
    ASSERT_TRUE(ReadOf(scan2).has_value());
    EXPECT_EQ(*ReadOf(scan2), IndexesOf("orders", {"o_totalprice", "o_orderstatus"}));
}

TEST_F(ReadColumnsTest, SeqScanCarriesItsPredicatesAsSkipHints) {
    const auto plan = Plan("SELECT COUNT(*) FROM lineitem WHERE l_quantity < 24 AND l_shipdate >= '1994-01-01'");
    const auto* scan = ScanOf(plan.get(), "lineitem");
    ASSERT_NE(scan, nullptr);
    ASSERT_EQ(scan->type, PlanNodeType::SEQ_SCAN);
    EXPECT_EQ(static_cast<const SeqScanPlanNode*>(scan)->skip_conjuncts.size(), 2u);

    const auto no_where = Plan("SELECT COUNT(*) FROM lineitem");
    const auto* bare = ScanOf(no_where.get(), "lineitem");
    ASSERT_NE(bare, nullptr);
    EXPECT_TRUE(static_cast<const SeqScanPlanNode*>(bare)->skip_conjuncts.empty());
}

TEST_F(ReadColumnsTest, EveryTpchPlanReadsOnlyListedColumns) {
    for (const bool with_indexes : {false, true}) {
        SCOPED_TRACE(with_indexes ? "with indexes" : "without indexes");
        if (with_indexes) {
            // SF=1-sized stats on the two date columns, so date ranges pick their index.
            auto* lineitem = catalog_->GetTable("lineitem");
            lineitem->stats.row_count = 6000000;
            lineitem->stats.page_count = 120000;
            lineitem->stats.min_max["l_shipdate"] = {Value::FromString(TypeId::DATE, "1992-01-02"),
                                                     Value::FromString(TypeId::DATE, "1998-12-01")};
            auto* orders = catalog_->GetTable("orders");
            orders->stats.row_count = 1500000;
            orders->stats.page_count = 25000;
            orders->stats.min_max["o_orderdate"] = {Value::FromString(TypeId::DATE, "1992-01-01"),
                                                    Value::FromString(TypeId::DATE, "1998-08-02")};
            for (const auto& idx : tpch::IndexDescs()) {
                ASSERT_NE(catalog_->CreateIndex(idx.index_name, idx.table_name, idx.column_name), nullptr);
            }
        }
        size_t index_scans = 0;
        for (const auto& q : tpch::Queries()) {
            SCOPED_TRACE(q.tag);
            const auto plan = Plan(q.sql);
            EXPECT_TRUE(FindUnlistedReads(*plan).empty());
            std::vector<const PlanNode*> scans;
            CollectScans(plan.get(), scans);
            for (const auto* s : scans) index_scans += s->type == PlanNodeType::INDEX_SCAN ? 1 : 0;
        }
        if (with_indexes) EXPECT_GT(index_scans, 0u);
    }
}

}  // namespace
}  // namespace shilmandb
