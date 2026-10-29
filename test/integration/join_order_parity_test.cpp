#include <gtest/gtest.h>
#include <unistd.h>

#include "engine/database.hpp"
#include "planner/join_graph.hpp"
#include "tpch_workload.hpp"

#include <algorithm>
#include <filesystem>
#include <iterator>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace shilmandb {

namespace {

// "[2,1,0]", for failure messages.
std::string OrderString(const std::vector<int>& order) {
    std::string out = "[";
    for (size_t i = 0; i < order.size(); ++i) {
        if (i > 0) out += ',';
        out += std::to_string(order[i]);
    }
    return out + "]";
}

}  // namespace

// Runs each TPC-H join query in every connected join order and checks the answer
// never changes.
class JoinOrderParityTest : public ::testing::Test {
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
                      ("shilmandb_join_order_parity_test_" + std::to_string(::getpid()) + ".db")).string();
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
    }

    void TearDown() override {
        db_.reset();
        std::filesystem::remove(test_file_);
    }

    // Forces every connected join order of the query and compares each answer to the default plan's.
    void CheckAllOrders(const std::string& tag, size_t expected_count) {
        const auto query = tpch::FindQuery(tag);
        ASSERT_TRUE(query.has_value()) << tag;

        const auto info = db_->ExplainJoinOrder(query->sql);
        const auto orders = ConnectedOrders(static_cast<int>(info.tables.size()), info.edges);
        ASSERT_EQ(orders.size(), expected_count) << tag << ": wrong number of connected orders";
        EXPECT_EQ(std::set<std::vector<int>>(orders.begin(), orders.end()).size(), orders.size())
            << tag << ": an order is listed twice";

        const auto ref = db_->ExecuteSQL(query->sql);
        ASSERT_GT(ref.tuples.size(), 0u) << tag << ": default plan returned no rows";

        for (const auto& order : orders) {
            QueryOptions options;
            options.forced_join_order = order;
            const auto result = db_->ExecuteSQL(query->sql, options);

            EXPECT_TRUE(tpch::SameResult(ref, result))
                << tag << " order " << OrderString(order) << ": result differs from the default plan (forced "
                << result.tuples.size() << " rows, default " << ref.tuples.size() << ")";
            EXPECT_EQ(result.profile.join.order, order) << tag << " order " << OrderString(order);
            EXPECT_EQ(result.profile.join.source, JoinOrderSource::FORCED) << tag << " order " << OrderString(order);
        }
    }
};

TEST_F(JoinOrderParityTest, Q3AllOrders) { CheckAllOrders("Q3", 4); }
TEST_F(JoinOrderParityTest, Q5AllOrders) { CheckAllOrders("Q5", 16); }
TEST_F(JoinOrderParityTest, Q10AllOrders) { CheckAllOrders("Q10", 8); }
TEST_F(JoinOrderParityTest, Q12AllOrders) { CheckAllOrders("Q12", 2); }
TEST_F(JoinOrderParityTest, Q14AllOrders) { CheckAllOrders("Q14", 2); }
TEST_F(JoinOrderParityTest, Q19AllOrders) { CheckAllOrders("Q19", 2); }

// The classical planner's picks, as slots into info.tables. Pinned so a planner change
// can't move them without anyone noticing: the answer would stay right, the time wouldn't.
TEST_F(JoinOrderParityTest, DefaultOrdersUnchanged) {
    struct Expected {
        std::string tag;
        std::vector<int> order;
    };
    const std::vector<Expected> expected = {
        {"Q3", {0, 1, 2}},        // customer, orders, lineitem
        {"Q5", {4, 3, 2, 1, 0}},  // nation, supplier, lineitem, orders, customer
        {"Q10", {3, 0, 1, 2}},    // nation, customer, orders, lineitem
        {"Q12", {0, 1}},          // orders, lineitem
        {"Q14", {1, 0}},          // part, lineitem
        {"Q19", {1, 0}},          // part, lineitem
    };

    for (const auto& [tag, order] : expected) {
        const auto query = tpch::FindQuery(tag);
        ASSERT_TRUE(query.has_value()) << tag;

        const auto info = db_->ExplainJoinOrder(query->sql);
        EXPECT_EQ(info.source, JoinOrderSource::CLASSICAL) << tag;
        EXPECT_EQ(info.order, order) << tag << ": got " << OrderString(info.order);
    }
}

TEST_F(JoinOrderParityTest, Q5AnalyzeSanity) {
    const auto query = tpch::FindQuery("Q5");
    ASSERT_TRUE(query.has_value());

    QueryOptions options;
    options.analyze = true;
    const auto result = db_->ExecuteSQL(query->sql, options);
    const auto& profile = result.profile;
    const auto& info = profile.join;

    EXPECT_EQ(info.source, JoinOrderSource::CLASSICAL);
    ASSERT_EQ(info.order.size(), 5u);
    ASSERT_EQ(profile.step_rows.size(), 5u);
    ASSERT_EQ(profile.scan_rows.size(), 5u);

    // A scan plus its filter can't return more rows than the table has.
    for (size_t k = 0; k < info.order.size(); ++k) {
        const auto& table = info.tables[info.order[k]];
        const auto* table_info = db_->GetCatalog()->GetTable(table);
        ASSERT_NE(table_info, nullptr) << table;
        EXPECT_LE(profile.scan_rows[k], table_info->stats.row_count) << table;
    }

    // The aggregate's input is the last join step's output.
    const auto& ops = profile.operators;
    const auto agg = std::find_if(ops.begin(), ops.end(),
                                  [](const OperatorProfile& op) { return op.name == "Aggregate"; });
    ASSERT_NE(agg, ops.end());
    const auto child = std::next(agg);
    ASSERT_NE(child, ops.end());
    EXPECT_EQ(child->depth, agg->depth + 1);
    EXPECT_EQ(child->rows, profile.step_rows.back());
    EXPECT_GT(profile.step_rows.back(), 0u);
}

}  // namespace shilmandb
