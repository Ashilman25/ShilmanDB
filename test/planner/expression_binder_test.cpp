#include <gtest/gtest.h>
#include <algorithm>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unistd.h>
#include <vector>

#include "common/exception.hpp"
#include "engine/database.hpp"
#include "executor/expression_evaluator.hpp"
#include "parser/parser.hpp"
#include "planner/expression_binder.hpp"
#include "planner/planner.hpp"

namespace shilmandb {

class ExpressionBinderTest : public ::testing::Test {
protected:
    std::string test_file_;
    std::unique_ptr<Database> db_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() /
                      ("shilmandb_expression_binder_test_" + std::to_string(::getpid()) + ".db")).string();
        std::filesystem::remove(test_file_);
        db_ = std::make_unique<Database>(test_file_);
    }

    void TearDown() override {
        db_.reset();
        std::filesystem::remove(test_file_);
    }

    void MakeTable(const std::string& name, const Schema& schema,
                   const std::vector<std::vector<Value>>& rows) {
        auto* info = db_->GetCatalog()->CreateTable(name, schema);
        for (const auto& row : rows) {
            (void)info->table->InsertTuple(Tuple(row, schema));
        }
        db_->GetCatalog()->UpdateTableStats(name);
    }

    std::unique_ptr<PlanNode> PlanSql(const std::string& sql,
                                      std::optional<std::vector<int>> order = std::nullopt) {
        Parser parser(sql);
        auto stmt = parser.Parse();
        Planner planner(db_->GetCatalog());
        if (order) planner.SetForcedJoinOrder(*order);
        return planner.Plan(std::move(*stmt));
    }

    static const PlanNode* FindNode(const PlanNode* node, PlanNodeType type) {
        if (node->type == type) return node;
        for (const auto& child : node->children) {
            if (const auto* found = FindNode(child.get(), type)) return found;
        }
        return nullptr;
    }

    static std::unique_ptr<Expression> MakeColRef(const std::string& name) {
        auto c = std::make_unique<ColumnRef>();
        c->column_name = name;
        return c;
    }

    static std::unique_ptr<Expression> MakeLiteral(Value v) {
        auto l = std::make_unique<Literal>();
        l->value = std::move(v);
        return l;
    }

    static std::unique_ptr<Expression> MakeBinOp(BinaryOp::Op op, std::unique_ptr<Expression> lhs,
                                                 std::unique_ptr<Expression> rhs) {
        auto b = std::make_unique<BinaryOp>();
        b->op = op;
        b->left = std::move(lhs);
        b->right = std::move(rhs);
        return b;
    }

    static const ColumnRef* AsRef(const Expression* e) {
        return static_cast<const ColumnRef*>(e);
    }
    static const BinaryOp* AsBin(const Expression* e) {
        return static_cast<const BinaryOp*>(e);
    }
};

TEST_F(ExpressionBinderTest, FilterBindsAgainstScan) {
    MakeTable("t", Schema({Column("a", TypeId::INTEGER), Column("b", TypeId::INTEGER),
                           Column("c", TypeId::INTEGER)}), {});
    auto plan = PlanSql("SELECT a FROM t WHERE b = 5");
    const auto* filter = static_cast<const FilterPlanNode*>(FindNode(plan.get(), PlanNodeType::FILTER));
    ASSERT_NE(filter, nullptr);
    EXPECT_EQ(AsRef(AsBin(filter->predicate.get())->left.get())->bound_index, 1u);
    const auto* proj = static_cast<const ProjectionPlanNode*>(FindNode(plan.get(), PlanNodeType::PROJECTION));
    ASSERT_NE(proj, nullptr);
    EXPECT_EQ(AsRef(proj->expressions[0].get())->bound_index, 0u);
}

TEST_F(ExpressionBinderTest, HashJoinPredicateBindsAgainstJoinOutput) {
    MakeTable("t1", Schema({Column("x", TypeId::INTEGER), Column("a", TypeId::INTEGER)}), {});
    MakeTable("t2", Schema({Column("y", TypeId::INTEGER), Column("b", TypeId::INTEGER)}), {});
    auto plan = PlanSql("SELECT a FROM t1 JOIN t2 ON t1.x = t2.y", std::vector<int>{0, 1});
    const auto* join = static_cast<const HashJoinPlanNode*>(FindNode(plan.get(), PlanNodeType::HASH_JOIN));
    ASSERT_NE(join, nullptr);
    const auto* eq = AsBin(join->join_predicate.get());
    EXPECT_EQ(AsRef(eq->left.get())->bound_index, 0u);
    EXPECT_EQ(AsRef(eq->right.get())->bound_index, 2u);
}

TEST_F(ExpressionBinderTest, NestedLoopJoinPredicateBinds) {
    MakeTable("t1", Schema({Column("x", TypeId::INTEGER), Column("a", TypeId::INTEGER)}), {});
    MakeTable("t2", Schema({Column("y", TypeId::INTEGER), Column("b", TypeId::INTEGER)}), {});
    auto plan = PlanSql("SELECT a FROM t1 JOIN t2 ON t1.x < t2.y", std::vector<int>{0, 1});
    const auto* join = static_cast<const NestedLoopJoinPlanNode*>(
        FindNode(plan.get(), PlanNodeType::NESTED_LOOP_JOIN));
    ASSERT_NE(join, nullptr);
    const auto* lt = AsBin(join->predicate.get());
    EXPECT_EQ(AsRef(lt->left.get())->bound_index, 0u);
    EXPECT_EQ(AsRef(lt->right.get())->bound_index, 2u);
}

TEST_F(ExpressionBinderTest, DuplicateNamesBindToFirstMatch) {
    MakeTable("nation", Schema({Column("n_nationkey", TypeId::INTEGER),
                                Column("n_name", TypeId::VARCHAR)}), {});
    auto plan = PlanSql("SELECT n1.n_name FROM nation n1, nation n2 "
                        "WHERE n1.n_nationkey = n2.n_nationkey");
    const PlanNode* join = FindNode(plan.get(), PlanNodeType::HASH_JOIN);
    ASSERT_NE(join, nullptr);
    const auto* eq = AsBin(static_cast<const HashJoinPlanNode*>(join)->join_predicate.get());
    for (const Expression* side : {eq->left.get(), eq->right.get()}) {
        const auto* ref = AsRef(side);
        ASSERT_TRUE(ref->bound_index.has_value());
        EXPECT_EQ(*ref->bound_index, join->output_schema.GetColumnIndex(ref->column_name));
        EXPECT_EQ(*ref->bound_index, 0u);
    }
}

TEST_F(ExpressionBinderTest, AggregateProjectionSortBind) {
    MakeTable("t", Schema({Column("v", TypeId::DECIMAL), Column("g", TypeId::INTEGER)}), {});
    auto plan = PlanSql("SELECT g AS grp, SUM(v) AS total FROM t GROUP BY g ORDER BY total");
    const auto* agg = static_cast<const AggregatePlanNode*>(FindNode(plan.get(), PlanNodeType::AGGREGATE));
    ASSERT_NE(agg, nullptr);
    EXPECT_EQ(AsRef(agg->group_by_exprs[0].get())->bound_index, 1u);
    EXPECT_EQ(AsRef(agg->aggregate_exprs[0].get())->bound_index, 0u);
    const auto* proj = static_cast<const ProjectionPlanNode*>(FindNode(plan.get(), PlanNodeType::PROJECTION));
    ASSERT_NE(proj, nullptr);
    EXPECT_EQ(AsRef(proj->expressions[0].get())->bound_index, 0u);
    EXPECT_EQ(AsRef(proj->expressions[1].get())->bound_index, 1u);
    const auto* sort = static_cast<const SortPlanNode*>(FindNode(plan.get(), PlanNodeType::SORT));
    ASSERT_NE(sort, nullptr);
    EXPECT_EQ(AsRef(sort->order_by[0].expr.get())->bound_index, 1u);
}

TEST_F(ExpressionBinderTest, CountStarPlans) {
    MakeTable("t", Schema({Column("a", TypeId::INTEGER)}), {});
    std::unique_ptr<PlanNode> plan;
    EXPECT_NO_THROW(plan = PlanSql("SELECT COUNT(*) FROM t"));
    const auto* agg = static_cast<const AggregatePlanNode*>(FindNode(plan.get(), PlanNodeType::AGGREGATE));
    ASSERT_NE(agg, nullptr);
    EXPECT_EQ(agg->aggregate_exprs[0], nullptr);
}

TEST_F(ExpressionBinderTest, HavingWithAggregatePlans) {
    MakeTable("t", Schema({Column("x", TypeId::INTEGER), Column("y", TypeId::INTEGER)}), {});
    EXPECT_NO_THROW(PlanSql("SELECT x, COUNT(*) FROM t GROUP BY x HAVING COUNT(*) > 1"));
}

TEST_F(ExpressionBinderTest, BindPlanSkipsNullPredicate) {
    NestedLoopJoinPlanNode node(Schema({Column("a", TypeId::INTEGER)}), nullptr);
    EXPECT_NO_THROW(BindPlan(&node));
}

TEST_F(ExpressionBinderTest, UnknownNameStaysUnbound) {
    auto ref = MakeColRef("zzz");
    Schema schema({Column("a", TypeId::INTEGER)});
    EXPECT_NO_THROW(BindExpression(ref.get(), schema));
    EXPECT_FALSE(AsRef(ref.get())->bound_index.has_value());
}

TEST_F(ExpressionBinderTest, RebindWithoutNameClearsIndex) {
    auto ref = MakeColRef("b");
    Schema both({Column("a", TypeId::INTEGER), Column("b", TypeId::INTEGER)});
    BindExpression(ref.get(), both);
    EXPECT_EQ(AsRef(ref.get())->bound_index, 1u);

    Schema only_a({Column("a", TypeId::INTEGER)});
    EXPECT_NO_THROW(BindExpression(ref.get(), only_a));
    EXPECT_FALSE(AsRef(ref.get())->bound_index.has_value());
}

TEST_F(ExpressionBinderTest, UnknownColumnStillFailsOnlyAtRunTime) {
    Schema schema({Column("a", TypeId::INTEGER)});
    MakeTable("t", schema, {});
    EXPECT_EQ(db_->ExecuteSQL("SELECT a FROM t ORDER BY zzz").tuples.size(), 0u);

    auto* info = db_->GetCatalog()->GetTable("t");
    (void)info->table->InsertTuple(Tuple({Value(1)}, schema));
    (void)info->table->InsertTuple(Tuple({Value(2)}, schema));
    EXPECT_THROW((void)db_->ExecuteSQL("SELECT a FROM t ORDER BY zzz"), DatabaseException);
}

TEST_F(ExpressionBinderTest, CloneIsUnbound) {
    ColumnRef ref;
    ref.column_name = "k";
    ref.bound_index = 3;
    auto clone = ref.Clone();
    const auto* c = AsRef(clone.get());
    EXPECT_FALSE(c->bound_index.has_value());
    EXPECT_EQ(c->column_name, "k");
}

TEST_F(ExpressionBinderTest, EvaluateUsesBoundIndex) {
    Schema schema({Column("a", TypeId::INTEGER), Column("b", TypeId::INTEGER)});
    Tuple tuple({Value(1), Value(2)}, schema);
    ColumnRef bound;
    bound.column_name = "b";
    bound.bound_index = 1;
    EXPECT_EQ(EvaluateExpression(&bound, tuple, schema).integer_, 2);
    ColumnRef unbound;
    unbound.column_name = "b";
    EXPECT_EQ(EvaluateExpression(&unbound, tuple, schema).integer_, 2);
}

// A one-column schema (named after the column, or `schema_col` if given), and a comparison against a literal.
static const Literal* BindAndGetLiteral(BinaryOp::Op op, const std::string& col, TypeId col_type,
                                        Value lit, bool literal_on_left, std::unique_ptr<Expression>* keep,
                                        const std::string& schema_col = "") {
    Schema schema({Column(schema_col.empty() ? col : schema_col, col_type)});
    auto ref = std::make_unique<ColumnRef>();
    ref->column_name = col;
    auto l = std::make_unique<Literal>();
    l->value = std::move(lit);
    const Literal* raw = l.get();
    auto b = std::make_unique<BinaryOp>();
    b->op = op;
    if (literal_on_left) {
        b->left = std::move(l);
        b->right = std::move(ref);
    } else {
        b->left = std::move(ref);
        b->right = std::move(l);
    }
    BindExpression(b.get(), schema);
    *keep = std::move(b);
    return raw;
}

TEST_F(ExpressionBinderTest, DateLiteralBecomesDate) {
    std::unique_ptr<Expression> keep;
    const auto* lit = BindAndGetLiteral(BinaryOp::Op::GTE, "d", TypeId::DATE, Value(std::string("1994-01-01")), false, &keep);
    EXPECT_EQ(lit->value.type_, TypeId::DATE);
    EXPECT_EQ(lit->value.date_, Value::FromString(TypeId::DATE, "1994-01-01").date_);
}

TEST_F(ExpressionBinderTest, LiteralOnLeftConverts) {
    std::unique_ptr<Expression> keep;
    const auto* lit = BindAndGetLiteral(BinaryOp::Op::GT, "d", TypeId::DATE, Value(std::string("1995-03-15")), true, &keep);
    EXPECT_EQ(lit->value.type_, TypeId::DATE);
    EXPECT_EQ(lit->value.date_, Value::FromString(TypeId::DATE, "1995-03-15").date_);
}

TEST_F(ExpressionBinderTest, IntegerLiteralWidensToDecimalColumn) {
    std::unique_ptr<Expression> keep;
    const auto* lit = BindAndGetLiteral(BinaryOp::Op::LT, "p", TypeId::DECIMAL, Value(24), false, &keep);
    EXPECT_EQ(lit->value.type_, TypeId::DECIMAL);
    EXPECT_DOUBLE_EQ(lit->value.decimal_, 24.0);
}

TEST_F(ExpressionBinderTest, IntegerLiteralWidensToBigintColumn) {
    std::unique_ptr<Expression> keep;
    const auto* lit = BindAndGetLiteral(BinaryOp::Op::EQ, "k", TypeId::BIGINT, Value(7), false, &keep);
    EXPECT_EQ(lit->value.type_, TypeId::BIGINT);
    EXPECT_EQ(lit->value.bigint_, 7);
}

TEST_F(ExpressionBinderTest, DecimalLiteralNextToIntegerColumnUnchanged) {
    std::unique_ptr<Expression> keep;
    const auto* lit = BindAndGetLiteral(BinaryOp::Op::LT, "i", TypeId::INTEGER, Value(0.05), false, &keep);
    EXPECT_EQ(lit->value.type_, TypeId::DECIMAL);
    EXPECT_DOUBLE_EQ(lit->value.decimal_, 0.05);
}

TEST_F(ExpressionBinderTest, SameTypeLiteralUnchanged) {
    std::unique_ptr<Expression> keep;
    const auto* lit = BindAndGetLiteral(BinaryOp::Op::EQ, "s", TypeId::VARCHAR, Value(std::string("BUILDING")), false, &keep);
    EXPECT_EQ(lit->value.type_, TypeId::VARCHAR);
    EXPECT_EQ(lit->value.varchar_, "BUILDING");
}

TEST_F(ExpressionBinderTest, IncompatibleLiteralUnchanged) {
    std::unique_ptr<Expression> keep;
    const Literal* lit = nullptr;
    EXPECT_NO_THROW(lit = BindAndGetLiteral(BinaryOp::Op::EQ, "s", TypeId::VARCHAR, Value(5), false, &keep));
    EXPECT_EQ(lit->value.type_, TypeId::INTEGER);
    EXPECT_EQ(lit->value.integer_, 5);
}

TEST_F(ExpressionBinderTest, BadDateStringUnchanged) {
    std::unique_ptr<Expression> keep;
    const Literal* lit = nullptr;
    EXPECT_NO_THROW(lit = BindAndGetLiteral(BinaryOp::Op::EQ, "d", TypeId::DATE, Value(std::string("not-a-date")), false, &keep));
    EXPECT_EQ(lit->value.type_, TypeId::VARCHAR);
    EXPECT_EQ(lit->value.varchar_, "not-a-date");
}

TEST_F(ExpressionBinderTest, ArithmeticLiteralUnchanged) {
    std::unique_ptr<Expression> keep;
    const auto* lit = BindAndGetLiteral(BinaryOp::Op::SUB, "p", TypeId::DECIMAL, Value(1), false, &keep);
    EXPECT_EQ(lit->value.type_, TypeId::INTEGER);
    EXPECT_EQ(lit->value.integer_, 1);
}

TEST_F(ExpressionBinderTest, UnboundColumnLeavesLiteral) {
    std::unique_ptr<Expression> keep;
    const auto* lit = BindAndGetLiteral(BinaryOp::Op::EQ, "zzz", TypeId::DATE, Value(std::string("1994-01-01")), false, &keep, "d");
    EXPECT_EQ(lit->value.type_, TypeId::VARCHAR);
}

TEST_F(ExpressionBinderTest, LikeLiteralUnchanged) {
    Schema schema({Column("d", TypeId::DATE)});
    auto expr = MakeBinOp(BinaryOp::Op::LIKE, MakeColRef("d"), MakeLiteral(Value(std::string("1994-01-0_"))));
    BindExpression(expr.get(), schema);
    const auto* lit = static_cast<const Literal*>(AsBin(expr.get())->right.get());
    EXPECT_EQ(lit->value.type_, TypeId::VARCHAR);
    EXPECT_EQ(lit->value.varchar_, "1994-01-0_");
}

TEST_F(ExpressionBinderTest, NotComparisonConverts) {
    Schema schema({Column("d", TypeId::DATE)});
    auto cmp = MakeBinOp(BinaryOp::Op::EQ, MakeColRef("d"), MakeLiteral(Value(std::string("1994-01-01"))));
    auto not_expr = std::make_unique<UnaryOp>();
    not_expr->op = UnaryOp::Op::NOT;
    not_expr->operand = std::move(cmp);
    BindExpression(not_expr.get(), schema);
    const auto* inner = AsBin(not_expr->operand.get());
    const auto* lit = static_cast<const Literal*>(inner->right.get());
    EXPECT_EQ(lit->value.type_, TypeId::DATE);
    EXPECT_EQ(lit->value.date_, Value::FromString(TypeId::DATE, "1994-01-01").date_);
}

TEST_F(ExpressionBinderTest, BetweenDatesConvertBothEnds) {
    MakeTable("t", Schema({Column("a", TypeId::INTEGER), Column("d", TypeId::DATE)}), {});
    auto plan = PlanSql("SELECT a FROM t WHERE d BETWEEN '1994-01-01' AND '1994-12-31'");
    const auto* filter = static_cast<const FilterPlanNode*>(FindNode(plan.get(), PlanNodeType::FILTER));
    ASSERT_NE(filter, nullptr);
    int literals = 0;
    auto check = [&](auto&& self, const Expression* e) -> void {
        if (e == nullptr) return;
        if (e->type == ExprType::LITERAL) {
            ++literals;
            EXPECT_EQ(static_cast<const Literal*>(e)->value.type_, TypeId::DATE);
        } else if (e->type == ExprType::BINARY_OP) {
            self(self, AsBin(e)->left.get());
            self(self, AsBin(e)->right.get());
        }
    };
    check(check, filter->predicate.get());
    EXPECT_EQ(literals, 2);
}

TEST_F(ExpressionBinderTest, BadDateFailsOnlyAtRunTime) {
    Schema schema({Column("a", TypeId::INTEGER), Column("d", TypeId::DATE)});
    MakeTable("t", schema, {});
    QueryResult empty;
    EXPECT_NO_THROW(empty = db_->ExecuteSQL("SELECT a FROM t WHERE d = 'not-a-date'"));
    EXPECT_EQ(empty.tuples.size(), 0u);
    (void)db_->GetCatalog()->GetTable("t")->table->InsertTuple(Tuple({Value(1), Value::MakeDate(0)}, schema));
    EXPECT_ANY_THROW((void)db_->ExecuteSQL("SELECT a FROM t WHERE d = 'not-a-date'"));
}

TEST_F(ExpressionBinderTest, ConvertedDatesGiveSameAnswer) {
    Schema schema({Column("a", TypeId::INTEGER), Column("d", TypeId::DATE)});
    MakeTable("t", schema, {
        {Value(1), Value::FromString(TypeId::DATE, "1993-12-31")},
        {Value(2), Value::FromString(TypeId::DATE, "1994-01-01")},
        {Value(3), Value::FromString(TypeId::DATE, "1994-06-15")},
        {Value(4), Value::FromString(TypeId::DATE, "1995-01-01")},
    });
    for (auto mode : {ExecutionMode::TUPLE, ExecutionMode::VECTORIZED}) {
        auto result = db_->ExecuteSQL("SELECT a FROM t WHERE d >= '1994-01-01' AND d < '1995-01-01'", mode);
        std::vector<int32_t> got;
        for (const auto& t : result.tuples) got.push_back(t.GetValue(result.schema, 0).integer_);
        std::sort(got.begin(), got.end());
        EXPECT_EQ(got, (std::vector<int32_t>{2, 3}));
    }
}

}  // namespace shilmandb
