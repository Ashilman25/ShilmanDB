#include <gtest/gtest.h>
#include <string>
#include <unistd.h>

#include "engine/database.hpp"
#include "common/exception.hpp"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace shilmandb {

namespace {

// a(a_id, a_val) -> b(b_id, b_a) -> c(c_id, c_b). The filter keeps a rows 2, 3, 4.
const std::string kSql =
    "SELECT a.a_val, b.b_id, c.c_id FROM a JOIN b ON a.a_id = b.b_a "
    "JOIN c ON b.b_id = c.c_b WHERE a.a_val >= 20";

// Each row as one string, sorted, so results can be compared regardless of join order.
std::vector<std::string> SortedRows(const QueryResult& result) {
    std::vector<std::string> rows;
    rows.reserve(result.tuples.size());
    for (const auto& tuple : result.tuples) {
        std::string row;
        for (uint32_t c = 0; c < result.schema.GetColumnCount(); ++c) {
            row += tuple.GetValue(result.schema, c).ToString();
            row += '|';
        }
        rows.push_back(std::move(row));
    }
    std::sort(rows.begin(), rows.end());
    return rows;
}

QueryOptions Forced(std::vector<int> order, bool analyze = false) {
    QueryOptions options;
    options.forced_join_order = std::move(order);
    options.analyze = analyze;
    return options;
}

}  // namespace

class QueryProfileTest : public ::testing::Test {
protected:
    std::string test_file_;
    std::unique_ptr<Database> db_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() /
                      ("shilmandb_query_profile_test_" + std::to_string(::getpid()) + ".db")).string();
        std::filesystem::remove(test_file_);
        db_ = std::make_unique<Database>(test_file_);

        LoadSimpleTable("a", Schema({Column("a_id", TypeId::INTEGER), Column("a_val", TypeId::INTEGER)}),
                        {{1, 10}, {2, 20}, {3, 30}, {4, 40}});
        LoadSimpleTable("b", Schema({Column("b_id", TypeId::INTEGER), Column("b_a", TypeId::INTEGER)}),
                        {{1, 1}, {2, 1}, {3, 2}, {4, 3}, {5, 3}, {6, 4}});
        LoadSimpleTable("c", Schema({Column("c_id", TypeId::INTEGER), Column("c_b", TypeId::INTEGER)}),
                        {{1, 1}, {2, 2}, {3, 2}, {4, 3}, {5, 4}, {6, 5}, {7, 6}, {8, 6}});
    }

    void TearDown() override {
        db_.reset();
        std::filesystem::remove(test_file_);
    }

    void LoadSimpleTable(const std::string& name, const Schema& schema,
                         const std::vector<std::vector<int32_t>>& rows) {
        auto* table_info = db_->GetCatalog()->CreateTable(name, schema);
        for (const auto& row : rows) {
            std::vector<Value> values;
            for (int32_t v : row) values.emplace_back(v);
            (void)table_info->table->InsertTuple(Tuple(values, schema));
        }
        db_->GetCatalog()->UpdateTableStats(name);
    }
};

TEST_F(QueryProfileTest, DefaultFillsJoinInfoOnly) {
    auto result = db_->ExecuteSQL(kSql);

    ASSERT_EQ(result.tuples.size(), 5u);
    EXPECT_EQ(result.profile.join.tables, (std::vector<std::string>{"a", "b", "c"}));
    EXPECT_EQ(result.profile.join.source, JoinOrderSource::CLASSICAL);
    EXPECT_EQ(result.profile.join.order.size(), 3u);
    EXPECT_TRUE(result.profile.step_rows.empty());
    EXPECT_TRUE(result.profile.scan_rows.empty());
    EXPECT_TRUE(result.profile.operators.empty());
}

TEST_F(QueryProfileTest, EveryConnectedOrderSameRows) {
    const auto expected = SortedRows(db_->ExecuteSQL(kSql));
    ASSERT_EQ(expected.size(), 5u);

    const std::vector<std::vector<int>> orders = {{0, 1, 2}, {1, 0, 2}, {1, 2, 0}, {2, 1, 0}};
    for (const auto& order : orders) {
        auto result = db_->ExecuteSQL(kSql, Forced(order));
        EXPECT_EQ(SortedRows(result), expected);
        EXPECT_EQ(result.profile.join.order, order);
        EXPECT_EQ(result.profile.join.source, JoinOrderSource::FORCED);
    }
}

TEST_F(QueryProfileTest, AnalyzeRowsForward) {
    auto result = db_->ExecuteSQL(kSql, Forced({0, 1, 2}, true));

    EXPECT_EQ(result.profile.scan_rows, (std::vector<uint64_t>{3, 6, 8}));
    EXPECT_EQ(result.profile.step_rows, (std::vector<uint64_t>{3, 4, 5}));
}

TEST_F(QueryProfileTest, AnalyzeRowsReverse) {
    auto result = db_->ExecuteSQL(kSql, Forced({2, 1, 0}, true));

    EXPECT_EQ(result.profile.scan_rows, (std::vector<uint64_t>{8, 6, 3}));
    EXPECT_EQ(result.profile.step_rows, (std::vector<uint64_t>{8, 8, 5}));
}

TEST_F(QueryProfileTest, AnalyzeOperatorTree) {
    auto result = db_->ExecuteSQL(kSql, Forced({0, 1, 2}, true));
    const auto& ops = result.profile.operators;

    ASSERT_FALSE(ops.empty());
    EXPECT_EQ(ops[0].name, "Projection");
    EXPECT_EQ(ops[0].depth, 0);
    EXPECT_EQ(ops[0].rows, 5u);

    const bool has_join_c = std::any_of(ops.begin(), ops.end(),
                                        [](const OperatorProfile& op) { return op.name == "HashJoin +c"; });
    EXPECT_TRUE(has_join_c);

    for (const auto& op : ops) {
        EXPECT_GE(op.self_ms, 0.0) << op.name;
        EXPECT_LE(op.self_ms, op.total_ms) << op.name;
    }
}

TEST_F(QueryProfileTest, AnalyzeVectorizedThrows) {
    EXPECT_THROW((void)db_->ExecuteSQL(kSql, QueryOptions{ExecutionMode::VECTORIZED, std::nullopt, true}),
                 DatabaseException);
}

TEST_F(QueryProfileTest, ExplainDoesNotExecute) {
    auto* bpm = db_->GetBufferPoolManager();
    bpm->ResetStats();

    const auto info = db_->ExplainJoinOrder(kSql);
    EXPECT_EQ(info.order.size(), 3u);
    EXPECT_EQ(info.edges, (std::vector<TableEdge>{{0, 1}, {1, 2}}));
    EXPECT_EQ(bpm->GetHitCount(), 0u);
    EXPECT_EQ(bpm->GetMissCount(), 0u);

    const auto forced = db_->ExplainJoinOrder(kSql, Forced({2, 1, 0}));
    EXPECT_EQ(forced.source, JoinOrderSource::FORCED);
}

TEST_F(QueryProfileTest, ModeOverloadUnchanged) {
    auto result = db_->ExecuteSQL(kSql, ExecutionMode::VECTORIZED);
    EXPECT_EQ(result.tuples.size(), 5u);
}

}  // namespace shilmandb
