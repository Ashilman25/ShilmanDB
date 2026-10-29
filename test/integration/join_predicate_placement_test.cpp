#include <gtest/gtest.h>
#include <string>
#include <unistd.h>

#include "engine/database.hpp"
#include "planner/join_graph.hpp"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <vector>

namespace shilmandb {

namespace {

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

// Join graphs where a new table can be tied in by more than one predicate, or where one
// ON condition spans three tables. Every connected order must give the same answer.
class JoinPredicatePlacementTest : public ::testing::Test {
protected:
    std::string test_file_;
    std::unique_ptr<Database> db_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() /
                      ("shilmandb_join_predicate_placement_test_" + std::to_string(::getpid()) + ".db")).string();
        std::filesystem::remove(test_file_);
        db_ = std::make_unique<Database>(test_file_);
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

    // The default plan must return `expected`, and so must every connected order.
    void ExpectAllOrdersMatch(const std::string& sql, const std::vector<std::string>& expected,
                              size_t expected_orders) {
        const auto ref = SortedRows(db_->ExecuteSQL(sql));
        ASSERT_EQ(ref, expected) << "default plan";

        const auto info = db_->ExplainJoinOrder(sql);
        const auto orders = ConnectedOrders(static_cast<int>(info.tables.size()), info.edges);
        ASSERT_EQ(orders.size(), expected_orders);

        for (const auto& order : orders) {
            QueryOptions options;
            options.forced_join_order = order;
            try {
                EXPECT_EQ(SortedRows(db_->ExecuteSQL(sql, options)), ref) << "order " << OrderString(order);
            } catch (const std::exception& e) {
                ADD_FAILURE() << "order " << OrderString(order) << " threw: " << e.what();
            }
        }
    }
};

// A triangle: a-c and c-b through ON, a-b through WHERE. With b placed before c, c is
// tied in by both ON conditions, and both must be applied.
TEST_F(JoinPredicatePlacementTest, TriangleAllOrdersSameRows) {
    LoadSimpleTable("a", Schema({Column("a_id", TypeId::INTEGER), Column("a_k", TypeId::INTEGER)}),
                    {{1, 1}, {2, 1}});
    LoadSimpleTable("c", Schema({Column("c_id", TypeId::INTEGER), Column("c_a", TypeId::INTEGER)}),
                    {{10, 1}, {11, 2}});
    LoadSimpleTable("b", Schema({Column("b_id", TypeId::INTEGER), Column("b_c", TypeId::INTEGER),
                                 Column("b_k", TypeId::INTEGER)}),
                    {{100, 10, 1}, {101, 11, 1}});

    ExpectAllOrdersMatch(
        "SELECT a.a_id, b.b_id, c.c_id FROM a JOIN c ON c.c_a = a.a_id "
        "JOIN b ON b.b_c = c.c_id WHERE a.a_k = b.b_k",
        {"1|100|10|", "2|101|11|"}, 6);
}

// A 4-cycle: an ON chain c-o-l-s closed by a WHERE predicate between c and s, the shape
// TPC-H Q5 takes with its customer and supplier nation keys.
TEST_F(JoinPredicatePlacementTest, CycleAllOrdersSameRows) {
    LoadSimpleTable("c", Schema({Column("c_id", TypeId::INTEGER), Column("c_k", TypeId::INTEGER)}),
                    {{1, 7}, {2, 8}});
    LoadSimpleTable("o", Schema({Column("o_id", TypeId::INTEGER), Column("o_c", TypeId::INTEGER)}),
                    {{10, 1}, {11, 2}});
    LoadSimpleTable("l", Schema({Column("l_id", TypeId::INTEGER), Column("l_o", TypeId::INTEGER),
                                 Column("l_s", TypeId::INTEGER)}),
                    {{100, 10, 5}, {101, 11, 6}, {102, 10, 6}});
    LoadSimpleTable("s", Schema({Column("s_id", TypeId::INTEGER), Column("s_k", TypeId::INTEGER)}),
                    {{5, 7}, {6, 8}});

    ExpectAllOrdersMatch(
        "SELECT c.c_id, o.o_id, l.l_id, s.s_id FROM c JOIN o ON c.c_id = o.o_c "
        "JOIN l ON l.l_o = o.o_id JOIN s ON l.l_s = s.s_id WHERE c.c_k = s.s_k",
        {"1|10|100|5|", "2|11|101|6|"}, 16);
}

// One ON spans x, y and z. It can only be applied once all three are placed.
TEST_F(JoinPredicatePlacementTest, ThreeTableOnAllOrdersSameRows) {
    LoadSimpleTable("x", Schema({Column("x_id", TypeId::INTEGER), Column("x_v", TypeId::INTEGER)}),
                    {{1, 10}, {2, 20}});
    LoadSimpleTable("y", Schema({Column("y_id", TypeId::INTEGER), Column("y_x", TypeId::INTEGER)}),
                    {{1, 1}, {2, 1}, {3, 2}});
    LoadSimpleTable("z", Schema({Column("z_id", TypeId::INTEGER), Column("z_x", TypeId::INTEGER),
                                 Column("z_y", TypeId::INTEGER)}),
                    {{7, 1, 1}, {8, 2, 3}, {9, 1, 2}});

    ExpectAllOrdersMatch(
        "SELECT x.x_id, y.y_id, z.z_id FROM x JOIN y ON y.y_x = x.x_id "
        "JOIN z ON z.z_x = x.x_id AND z.z_y = y.y_id",
        {"1|1|7|", "1|2|9|", "2|3|8|"}, 6);
}

}  // namespace shilmandb
