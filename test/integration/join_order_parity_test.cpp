#include <gtest/gtest.h>
#include <unistd.h>

#include "engine/database.hpp"
#include "parser/parser.hpp"
#include "planner/join_graph.hpp"
#include "planner/plan_node.hpp"
#include "planner/planner.hpp"
#include "tpch_workload.hpp"

#include <algorithm>
#include <filesystem>
#include <iterator>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
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

// How many column references and DATE-column-vs-literal comparisons CheckBound looked at.
int g_refs_checked = 0;
int g_date_comparisons_checked = 0;

// Checks every column reference in an expression is bound to its place in `schema`, and
// every DATE column compared with a literal has a DATE literal. Counts the problems it finds.
void CheckBound(const Expression* expr, const Schema& schema, const std::string& where, int* problems) {
    if (expr == nullptr) return;
    switch (expr->type) {
        case ExprType::COLUMN_REF: {
            const auto* col = static_cast<const ColumnRef*>(expr);
            ++g_refs_checked;
            if (!col->bound_index.has_value()) {
                ADD_FAILURE() << where << ": column " << col->column_name << " is not bound";
                ++*problems;
            } else if (schema.GetColumnIndex(col->column_name) != *col->bound_index) {
                ADD_FAILURE() << where << ": column " << col->column_name << " is bound to " << *col->bound_index
                              << ", the schema has it at " << schema.GetColumnIndex(col->column_name);
                ++*problems;
            }
            break;
        }
        case ExprType::BINARY_OP: {
            const auto* bin = static_cast<const BinaryOp*>(expr);
            const bool is_comparison = bin->op == BinaryOp::Op::EQ || bin->op == BinaryOp::Op::NEQ ||
                                       bin->op == BinaryOp::Op::LT || bin->op == BinaryOp::Op::GT ||
                                       bin->op == BinaryOp::Op::LTE || bin->op == BinaryOp::Op::GTE;
            if (is_comparison && bin->left && bin->right) {
                for (const auto& [col_side, lit_side] : {std::pair{bin->left.get(), bin->right.get()},
                                                         std::pair{bin->right.get(), bin->left.get()}}) {
                    if (col_side->type != ExprType::COLUMN_REF || lit_side->type != ExprType::LITERAL) continue;
                    const auto* col = static_cast<const ColumnRef*>(col_side);
                    const auto* lit = static_cast<const Literal*>(lit_side);
                    if (!col->bound_index.has_value()) continue;
                    if (schema.GetColumn(*col->bound_index).type == TypeId::DATE) ++g_date_comparisons_checked;
                    if (schema.GetColumn(*col->bound_index).type == TypeId::DATE && lit->value.type_ != TypeId::DATE) {
                        ADD_FAILURE() << where << ": DATE column " << col->column_name << " is compared with a non-DATE literal";
                        ++*problems;
                    }
                }
            }
            CheckBound(bin->left.get(), schema, where, problems);
            CheckBound(bin->right.get(), schema, where, problems);
            break;
        }
        case ExprType::UNARY_OP:
            CheckBound(static_cast<const UnaryOp*>(expr)->operand.get(), schema, where, problems);
            break;
        case ExprType::AGGREGATE:
            CheckBound(static_cast<const Aggregate*>(expr)->arg.get(), schema, where, problems);
            break;
        case ExprType::CASE: {
            const auto* c = static_cast<const CaseExpression*>(expr);
            for (const auto& [cond, result] : c->when_clauses) {
                CheckBound(cond.get(), schema, where, problems);
                CheckBound(result.get(), schema, where, problems);
            }
            CheckBound(c->else_clause.get(), schema, where, problems);
            break;
        }
        case ExprType::LITERAL:
        case ExprType::STAR:
            break;
    }
}

// Walks a plan and checks each node's expressions against the schema its executor reads rows with:
// joins read the combined row (their own output schema), everything else reads its child's output.
void CheckPlanBound(const PlanNode* node, const std::string& where, int* problems) {
    switch (node->type) {
        case PlanNodeType::FILTER:
            CheckBound(static_cast<const FilterPlanNode*>(node)->predicate.get(),
                       node->children[0]->output_schema, where + " Filter", problems);
            break;
        case PlanNodeType::HASH_JOIN:
            CheckBound(static_cast<const HashJoinPlanNode*>(node)->join_predicate.get(),
                       node->output_schema, where + " HashJoin", problems);
            break;
        case PlanNodeType::NESTED_LOOP_JOIN:
            CheckBound(static_cast<const NestedLoopJoinPlanNode*>(node)->predicate.get(),
                       node->output_schema, where + " NestedLoopJoin", problems);
            break;
        case PlanNodeType::AGGREGATE: {
            const auto* agg = static_cast<const AggregatePlanNode*>(node);
            for (const auto& e : agg->group_by_exprs)
                CheckBound(e.get(), node->children[0]->output_schema, where + " Aggregate", problems);
            for (const auto& e : agg->aggregate_exprs)
                CheckBound(e.get(), node->children[0]->output_schema, where + " Aggregate", problems);
            break;
        }
        case PlanNodeType::PROJECTION:
            for (const auto& e : static_cast<const ProjectionPlanNode*>(node)->expressions)
                CheckBound(e.get(), node->children[0]->output_schema, where + " Projection", problems);
            break;
        case PlanNodeType::SORT:
            for (const auto& item : static_cast<const SortPlanNode*>(node)->order_by)
                CheckBound(item.expr.get(), node->children[0]->output_schema, where + " Sort", problems);
            break;
        case PlanNodeType::SEQ_SCAN:
        case PlanNodeType::INDEX_SCAN:
        case PlanNodeType::LIMIT:
            break;
    }
    for (const auto& child : node->children) CheckPlanBound(child.get(), where, problems);
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
// The orders follow each table's filtered row count.
TEST_F(JoinOrderParityTest, DefaultOrdersUnchanged) {
    struct Expected {
        std::string tag;
        std::vector<int> order;
    };
    const std::vector<Expected> expected = {
        {"Q3", {0, 1, 2}},        // customer, orders, lineitem
        {"Q5", {0, 1, 2, 3, 4}},  // customer, orders, lineitem, supplier, nation
        {"Q10", {3, 0, 1, 2}},    // nation, customer, orders, lineitem
        {"Q12", {1, 0}},          // lineitem, orders
        {"Q14", {0, 1}},          // lineitem, part
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

// Every plan the planner builds for TPC-H, default and every forced connected order, has
// each column reference bound to the right position and DATE comparisons holding DATE literals.
TEST_F(JoinOrderParityTest, PlansAreFullyBound) {
    int plans_checked = 0;
    int problems = 0;
    g_refs_checked = 0;
    g_date_comparisons_checked = 0;

    const auto check_plan = [&](const std::string& sql, const std::string& where,
                                const std::optional<std::vector<int>>& order) {
        Parser parser(sql);
        auto stmt = parser.Parse();
        Planner planner(db_->GetCatalog());
        if (order) planner.SetForcedJoinOrder(*order);
        const auto plan = planner.Plan(std::move(*stmt));
        CheckPlanBound(plan.get(), where, &problems);
        ++plans_checked;
    };

    for (const auto& query : tpch::Queries()) {
        check_plan(query.sql, query.tag + " default", std::nullopt);
    }

    for (const std::string tag : {"Q3", "Q5", "Q10", "Q12", "Q14", "Q19"}) {
        const auto query = tpch::FindQuery(tag);
        ASSERT_TRUE(query.has_value()) << tag;
        const auto info = db_->ExplainJoinOrder(query->sql);
        for (const auto& order : ConnectedOrders(static_cast<int>(info.tables.size()), info.edges)) {
            check_plan(query->sql, tag + " order " + OrderString(order), order);
        }
    }

    EXPECT_EQ(problems, 0);
    EXPECT_EQ(plans_checked, 42);
    EXPECT_GT(g_refs_checked, 0);
    EXPECT_GT(g_date_comparisons_checked, 0);
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
