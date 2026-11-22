// OR factoring: SameExpression, FactorOrConjuncts, and Q19 planned for the parallel executor.

#include <gtest/gtest.h>
#include <unistd.h>

#include "engine/database.hpp"
#include "parser/parser.hpp"
#include "planner/or_factoring.hpp"
#include "planner/plan_dump.hpp"
#include "planner/plan_node.hpp"
#include "planner/planner.hpp"
#include "tpch/tpch_workload.hpp"

#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace shilmandb {
namespace {

using Conjuncts = std::vector<std::unique_ptr<Expression>>;

std::unique_ptr<Expression> MakeLiteral(Value value, std::optional<std::string> text = std::nullopt) {
    auto lit = std::make_unique<Literal>();
    lit->value = std::move(value);
    lit->text = std::move(text);
    return lit;
}

std::unique_ptr<Expression> MakeColRef(const std::string& name, std::optional<std::string> table = std::nullopt) {
    auto col = std::make_unique<ColumnRef>();
    col->column_name = name;
    col->table_name = std::move(table);
    return col;
}

// The WHERE clause of "SELECT * FROM t WHERE <where>", parsed.
std::unique_ptr<Expression> ParseWhere(const std::string& where) {
    Parser parser("SELECT * FROM t WHERE " + where);
    auto stmt = parser.Parse();
    return std::move(stmt->where_clause);
}

// Top-level AND parts, left first, as the planner splits a WHERE clause.
void SplitInto(std::unique_ptr<Expression> expr, Conjuncts& out) {
    auto* bin = dynamic_cast<BinaryOp*>(expr.get());
    if (bin && bin->op == BinaryOp::Op::AND) {
        SplitInto(std::move(bin->left), out);
        SplitInto(std::move(bin->right), out);
    } else {
        out.push_back(std::move(expr));
    }
}

Conjuncts SplitWhere(const std::string& where) {
    Conjuncts out;
    SplitInto(ParseWhere(where), out);
    return out;
}

// Tables by a column's first letter: a and l are 0, b and p 1, c 2.
// A column named x is in two tables, so it gets -1, like the planner.
void CollectTables(const Expression* expr, std::unordered_set<int>& out) {
    if (!expr) return;
    switch (expr->type) {
        case ExprType::COLUMN_REF: {
            const std::string& name = static_cast<const ColumnRef*>(expr)->column_name;
            if (name == "x") out.insert(-1);
            const char first = name[0];
            if (first == 'a' || first == 'l') out.insert(0);
            if (first == 'b' || first == 'p') out.insert(1);
            if (first == 'c') out.insert(2);
            break;
        }
        case ExprType::BINARY_OP: {
            const auto* bin = static_cast<const BinaryOp*>(expr);
            CollectTables(bin->left.get(), out);
            CollectTables(bin->right.get(), out);
            break;
        }
        case ExprType::UNARY_OP:
            CollectTables(static_cast<const UnaryOp*>(expr)->operand.get(), out);
            break;
        case ExprType::CASE: {
            // Unlike the planner, this sees inside CASE, so the CASE rule itself is tested.
            const auto* ce = static_cast<const CaseExpression*>(expr);
            for (const auto& [cond, result] : ce->when_clauses) {
                CollectTables(cond.get(), out);
                CollectTables(result.get(), out);
            }
            CollectTables(ce->else_clause.get(), out);
            break;
        }
        default:
            break;
    }
}

std::unordered_set<int> TablesOf(const Expression& expr) {
    std::unordered_set<int> out;
    CollectTables(&expr, out);
    return out;
}

// An expression's text without bound indexes, so parsed and planned trees compare.
std::string Text(const Expression& expr) {
    return std::regex_replace(test::ExprText(&expr), std::regex("@[0-9?]+"), "");
}

// Each conjunct's text must equal the parsed expected text, in order.
void ExpectConjuncts(const Conjuncts& got, const std::vector<std::string>& want) {
    ASSERT_EQ(got.size(), want.size());
    for (size_t i = 0; i < want.size(); ++i) {
        EXPECT_EQ(Text(*got[i]), Text(*ParseWhere(want[i]))) << "conjunct " << i;
    }
}

// Q19's WHERE clause, from the query text.
std::string Q19Where() {
    const auto q19 = tpch::FindQuery("Q19");
    const std::string sql = q19 ? q19->sql : "";
    return sql.substr(sql.find("WHERE ") + 6);
}

// CTest runs from build-debug/test/, so the data is usually two levels up.
std::string FindDataDir() {
    for (const std::string dir : {"bench/tpch_data/sf0.01/", "../bench/tpch_data/sf0.01/", "../../bench/tpch_data/sf0.01/"}) {
        if (std::filesystem::exists(dir + "lineitem.tbl")) return dir;
    }
    return {};
}

// sql planned in this mode by the database's planner, with its join decision.
std::pair<std::unique_ptr<PlanNode>, JoinOrderInfo> PlanQuery(const Database& db, const std::string& sql, ExecutionMode mode) {
    Parser parser(sql);
    auto stmt = parser.Parse();
    Planner planner = db.MakePlanner(QueryOptions{mode});
    auto plan = planner.Plan(std::move(*stmt));
    return {std::move(plan), planner.GetJoinOrderInfo()};
}

// The node right above table's scan, or nullptr when the scan is the top.
const PlanNode* ParentOfScan(const PlanNode& node, const std::string& table) {
    for (const auto& child : node.children) {
        if (child->type == PlanNodeType::SEQ_SCAN && static_cast<const SeqScanPlanNode&>(*child).table_name == table) {
            return &node;
        }
        if (const PlanNode* found = ParentOfScan(*child, table)) return found;
    }
    return nullptr;
}

// The node right above the first hash join.
const PlanNode* ParentOfJoin(const PlanNode& node) {
    for (const auto& child : node.children) {
        if (child->type == PlanNodeType::HASH_JOIN) return &node;
        if (const PlanNode* found = ParentOfJoin(*child)) return found;
    }
    return nullptr;
}

// Every BETWEEN starts at 1, so that bound is shared, not in part's OR.
const std::string kQ19PartOr =
    "(p_brand = 'Brand#12' AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG') AND p_size <= 5)"
    " OR (p_brand = 'Brand#23' AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK') AND p_size <= 10)"
    " OR (p_brand = 'Brand#34' AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG') AND p_size <= 15)";
const std::string kQ19LineitemOr =
    "(l_quantity >= 1 AND l_quantity <= 11) OR (l_quantity >= 10 AND l_quantity <= 20) OR (l_quantity >= 20 AND l_quantity <= 30)";
// The same once planned: the binder turns the quantity literals into decimals.
const std::string kQ19LineitemOrPlanned =
    "(l_quantity >= 1.0 AND l_quantity <= 11.0) OR (l_quantity >= 10.0 AND l_quantity <= 20.0)"
    " OR (l_quantity >= 20.0 AND l_quantity <= 30.0)";

class OrFactoringTest : public ::testing::Test {
protected:
    std::string data_dir_ = FindDataDir();
    std::vector<std::string> files_;

    void TearDown() override {
        for (const auto& file : files_) std::filesystem::remove(file);
    }

    // A temp path unique to this process.
    std::string TempPath(const std::string& what) {
        const std::string path = (std::filesystem::temp_directory_path() /
                                  ("shilmandb_or_factoring_test_" + what + "_" + std::to_string(::getpid())))
                                     .string();
        std::filesystem::remove(path);
        files_.push_back(path);
        return path;
    }

    // A fresh database with the 7 tables loaded into this storage.
    std::unique_ptr<Database> Load(StorageKind storage) {
        auto db = std::make_unique<Database>(TempPath(storage == StorageKind::COLUMNAR ? "col.db" : "row.db"), 4096);
        LoadOptions options;
        options.storage = storage;
        tpch::LoadTables(*db, data_dir_, options);
        return db;
    }

    std::unique_ptr<Database> LoadColumnar() { return Load(StorageKind::COLUMNAR); }

    // t1(a_k, x, a_y) and t2(b_k, x, b_y), columnar, from these rows: both have an x.
    std::unique_ptr<Database> LoadSharedX(const std::string& what, const std::string& t1_rows, const std::string& t2_rows) {
        auto db = std::make_unique<Database>(TempPath(what + ".db"), 64);
        LoadOptions columnar;
        columnar.storage = StorageKind::COLUMNAR;
        const auto load = [&](const std::string& name, const Schema& schema, const std::string& text) {
            const std::string path = TempPath(what + "_" + name + ".tbl");
            std::ofstream(path, std::ios::binary) << text;
            db->LoadTable(name, schema, path, columnar);
        };
        load("t1", Schema({Column("a_k", ColumnType::Integer()), Column("x", ColumnType::Integer()), Column("a_y", ColumnType::Integer())}), t1_rows);
        load("t2", Schema({Column("b_k", ColumnType::Integer()), Column("x", ColumnType::Integer()), Column("b_y", ColumnType::Integer())}), t2_rows);
        return db;
    }

    // PARALLEL on one thread must answer as VECTORIZED does, both joining t1 then t2.
    static void ExpectSameAsVectorized(Database& db, const std::string& sql, size_t want_rows) {
        QueryOptions vectorized{ExecutionMode::VECTORIZED};
        vectorized.forced_join_order = std::vector<int>{0, 1};
        QueryOptions parallel = vectorized;
        parallel.mode = ExecutionMode::PARALLEL;
        parallel.threads = 1;
        const QueryResult want = db.ExecuteSQL(sql, vectorized);
        ASSERT_EQ(want.tuples.size(), want_rows);
        const QueryResult got = db.ExecuteSQL(sql, parallel);
        EXPECT_TRUE(tpch::SameResult(got, want)) << got.tuples.size() << " rows";
    }

    // The filter right above table's scan in sql's parallel plan, as text, or "".
    static std::string ScanFilter(Database& db, const std::string& sql, const std::string& table) {
        QueryOptions parallel{ExecutionMode::PARALLEL};
        parallel.forced_join_order = std::vector<int>{0, 1};
        Parser parser(sql);
        auto stmt = parser.Parse();
        Planner planner = db.MakePlanner(parallel);
        const auto plan = planner.Plan(std::move(*stmt));
        const PlanNode* parent = ParentOfScan(*plan, table);
        if (parent == nullptr || parent->type != PlanNodeType::FILTER) return "";
        return Text(*static_cast<const FilterPlanNode*>(parent)->predicate);
    }
};

}  // namespace

// A literal of every type equals itself and differs from another value of its type.
TEST_F(OrFactoringTest, SameExpressionComparesLiteralsOfEveryType) {
    const std::vector<std::pair<Value, Value>> cases = {
        {Value(), Value()},
        {Value(int32_t{7}), Value(int32_t{8})},
        {Value(int64_t{7}), Value(int64_t{8})},
        {Value(0.5), Value(0.25)},
        {Value(std::string("AIR")), Value(std::string("AIR REG"))},
        {Value::MakeDate(9000), Value::MakeDate(9001)},
    };
    for (const auto& [value, other] : cases) {
        const std::string what = test::ValueText(value);
        EXPECT_TRUE(SameExpression(*MakeLiteral(value), *MakeLiteral(value))) << what;
        if (value.type_ != TypeId::INVALID) {
            EXPECT_FALSE(SameExpression(*MakeLiteral(value), *MakeLiteral(other))) << what;
        }
    }
}

// Literals of different types are different, and comparing them never throws.
TEST_F(OrFactoringTest, SameExpressionMixedTypesAreDifferent) {
    const std::vector<Value> values = {
        Value(), Value(int32_t{1}), Value(int64_t{1}), Value(1.0), Value(std::string("1")), Value::MakeDate(1),
    };
    for (size_t i = 0; i < values.size(); ++i) {
        for (size_t j = 0; j < values.size(); ++j) {
            if (i == j) continue;
            bool same = true;
            EXPECT_NO_THROW(same = SameExpression(*MakeLiteral(values[i]), *MakeLiteral(values[j]))) << i << " " << j;
            EXPECT_FALSE(same) << i << " " << j;
        }
    }
}

// Two decimal literals must also be written the same way.
TEST_F(OrFactoringTest, SameExpressionChecksLiteralText) {
    EXPECT_TRUE(SameExpression(*MakeLiteral(Value(0.1), "0.1"), *MakeLiteral(Value(0.1), "0.1")));
    EXPECT_FALSE(SameExpression(*MakeLiteral(Value(0.1), "0.1"), *MakeLiteral(Value(0.1), "0.10")));
    EXPECT_FALSE(SameExpression(*MakeLiteral(Value(0.1), "0.1"), *MakeLiteral(Value(0.1))));
    EXPECT_FALSE(SameExpression(*MakeLiteral(Value(0.1)), *MakeLiteral(Value(0.1), "0.1")));
    // 17 digits that round to the same double as 0.1 are still different.
    EXPECT_FALSE(SameExpression(*MakeLiteral(Value(0.1), "0.1"), *MakeLiteral(Value(0.10000000000000001), "0.10000000000000001")));
}

// Columns, operators, aggregates, star and CASE all compare by shape and names.
TEST_F(OrFactoringTest, SameExpressionComparesEveryNode) {
    EXPECT_TRUE(SameExpression(*MakeColRef("a_x"), *MakeColRef("a_x")));
    EXPECT_TRUE(SameExpression(*MakeColRef("a_x", "t"), *MakeColRef("a_x", "t")));
    EXPECT_FALSE(SameExpression(*MakeColRef("a_x"), *MakeColRef("a_y")));
    EXPECT_FALSE(SameExpression(*MakeColRef("a_x", "t"), *MakeColRef("a_x")));
    EXPECT_FALSE(SameExpression(*MakeColRef("a_x", "t"), *MakeColRef("a_x", "u")));
    EXPECT_FALSE(SameExpression(*MakeColRef("a_x"), *MakeLiteral(Value(int32_t{1}))));

    const auto same = [](const std::string& a, const std::string& b) { return SameExpression(*ParseWhere(a), *ParseWhere(b)); };
    EXPECT_TRUE(same("a_x = 1", "a_x = 1"));
    EXPECT_FALSE(same("a_x = 1", "a_x < 1"));
    EXPECT_FALSE(same("a_x = 1", "1 = a_x"));
    EXPECT_TRUE(same("a_x + 2 * a_y = 3", "a_x + 2 * a_y = 3"));
    EXPECT_FALSE(same("a_x + 2 * a_y = 3", "a_x - 2 * a_y = 3"));
    EXPECT_TRUE(same("NOT a_x = 1", "NOT a_x = 1"));
    EXPECT_FALSE(same("NOT a_x = 1", "a_x = 1"));
    EXPECT_TRUE(same("-a_x = 1", "-a_x = 1"));
    EXPECT_FALSE(same("-a_x = 1", "NOT a_x = 1"));
    EXPECT_TRUE(same("a_s LIKE 'x%'", "a_s LIKE 'x%'"));
    EXPECT_FALSE(same("a_s LIKE 'x%'", "a_s LIKE 'y%'"));
    EXPECT_TRUE(same("CASE WHEN a_x = 1 THEN 1 ELSE 0 END = 1", "CASE WHEN a_x = 1 THEN 1 ELSE 0 END = 1"));
    EXPECT_FALSE(same("CASE WHEN a_x = 1 THEN 1 ELSE 0 END = 1", "CASE WHEN a_x = 1 THEN 1 END = 1"));
    EXPECT_FALSE(same("CASE WHEN a_x = 1 THEN 1 ELSE 0 END = 1", "CASE WHEN a_x = 2 THEN 1 ELSE 0 END = 1"));
    EXPECT_FALSE(same("CASE WHEN a_x = 1 THEN 1 END = 1", "CASE WHEN a_x = 1 THEN 1 WHEN a_x = 2 THEN 1 END = 1"));

    Parser parser("SELECT COUNT(*), COUNT(a_x), SUM(a_x), *, SUM(a_x) FROM t");
    const auto stmt = parser.Parse();
    const auto item = [&](size_t i) -> const Expression& { return *stmt->select_list[i].expr; };
    EXPECT_TRUE(SameExpression(item(0), item(0)));
    EXPECT_FALSE(SameExpression(item(0), item(1)));
    EXPECT_FALSE(SameExpression(item(1), item(2)));
    EXPECT_TRUE(SameExpression(item(2), item(4)));
    EXPECT_TRUE(SameExpression(item(3), *std::make_unique<StarExpr>()));
}

// A conjunct in every branch comes out once, in the first branch's order.
TEST_F(OrFactoringTest, FactorsSharedConjuncts) {
    const std::string or_text = "(a_x = 1 AND b_y = 2 AND a_z = 3) OR (b_y = 2 AND a_x = 1 AND a_z = 4)";
    const auto out = FactorOrConjuncts(SplitWhere(or_text), TablesOf);
    ExpectConjuncts(out, {"a_x = 1", "b_y = 2", "a_z = 3 OR a_z = 4", or_text});
}

// Q19 gets its three shared filters and one implied OR each for part and lineitem.
TEST_F(OrFactoringTest, AddsPerTableImpliedOrs) {
    const std::string where = Q19Where();
    ASSERT_FALSE(where.empty());
    const auto original = SplitWhere(where);
    ASSERT_EQ(original.size(), 2u);

    const auto out = FactorOrConjuncts(SplitWhere(where), TablesOf);
    const std::vector<std::string> added = {
        "p_size >= 1",
        "l_shipmode IN ('AIR', 'AIR REG')",
        "l_shipinstruct = 'DELIVER IN PERSON'",
        kQ19PartOr,
        kQ19LineitemOr,
    };
    // The join conjunct first, then the additions, then the original OR unchanged.
    ASSERT_EQ(out.size(), 7u);
    EXPECT_EQ(Text(*out[0]), Text(*original[0]));
    for (size_t i = 0; i < added.size(); ++i) EXPECT_EQ(Text(*out[i + 1]), Text(*ParseWhere(added[i]))) << added[i];
    EXPECT_EQ(Text(*out[6]), Text(*original[1]));
}

// Conjuncts that aren't an OR, and ORs that name one table, pass through untouched.
TEST_F(OrFactoringTest, LeavesNonOrAlone) {
    for (const std::string where : {
             "a_x = 1 AND a_y = b_y AND NOT a_z = 2",
             "a_x IN (1, 2, 3) AND b_y = 4",
             "(a_x = 1 AND a_y = 2) OR (a_x = 3 AND a_y = 4)",
             "a_x = 1 OR b_y = 2",
             "a_x = b_y OR a_y = b_z",
         }) {
        const auto before = SplitWhere(where);
        const auto out = FactorOrConjuncts(SplitWhere(where), TablesOf);
        ASSERT_EQ(out.size(), before.size()) << where;
        for (size_t i = 0; i < out.size(); ++i) EXPECT_EQ(Text(*out[i]), Text(*before[i])) << where;
    }
    EXPECT_TRUE(FactorOrConjuncts({}, TablesOf).empty());
}

// A branch with nothing on a table means no implied OR for that table.
TEST_F(OrFactoringTest, NoImpliedOrWhenABranchLacksTheTable) {
    const std::string or_text = "(a_x = 1 AND b_y = 2) OR (a_x = 3 AND a_y = b_z)";
    const auto out = FactorOrConjuncts(SplitWhere(or_text), TablesOf);
    ExpectConjuncts(out, {"a_x = 1 OR a_x = 3", or_text});
}

// A conjunct in only some branches is never promoted.
TEST_F(OrFactoringTest, NeverPromotesFromSomeBranches) {
    const std::string or_text = "(a_x = 1 AND b_y = 2) OR (a_x = 1 AND b_y = 3) OR (b_y = 4 AND c_w = a_x)";
    const auto out = FactorOrConjuncts(SplitWhere(or_text), TablesOf);
    ExpectConjuncts(out, {"b_y = 2 OR b_y = 3 OR b_y = 4", or_text});
}

// CASE conjuncts and conjuncts naming no table are never promoted or used.
TEST_F(OrFactoringTest, SkipsCaseAndTablelessConjuncts) {
    const std::string or_text =
        "(CASE WHEN a_x = 1 THEN 1 ELSE 0 END = 1 AND 1 = 1 AND b_y = 2 AND a_q = b_q)"
        " OR (CASE WHEN a_x = 1 THEN 1 ELSE 0 END = 1 AND 1 = 1 AND b_y = 3 AND a_q = b_q)";
    const auto out = FactorOrConjuncts(SplitWhere(or_text), TablesOf);
    ExpectConjuncts(out, {"a_q = b_q", "b_y = 2 OR b_y = 3", or_text});
}

// Each OR conjunct is factored on its own, its additions just before it.
TEST_F(OrFactoringTest, FactorsEachOrInPlace) {
    const std::string first = "(a_x = 1 AND b_y = 2) OR (a_x = 3 AND b_y = 2)";
    const std::string second = "(c_w = 1 AND a_q = b_q) OR (c_w = 2 AND a_q = b_q)";
    const auto out = FactorOrConjuncts(SplitWhere("(" + first + ") AND a_z = 5 AND (" + second + ")"), TablesOf);
    ExpectConjuncts(out, {"b_y = 2", "a_x = 1 OR a_x = 3", first, "a_z = 5", "a_q = b_q", "c_w = 1 OR c_w = 2", second});
}

// A part holding a column with no known table is never promoted or used.
TEST_F(OrFactoringTest, UnplacedColumnsAreNeverFactored) {
    const std::string or_text = "(a_x = 1 AND b_y = x AND b_z = 2) OR (a_x = 2 AND b_y = x AND b_z = 3)";
    const auto out = FactorOrConjuncts(SplitWhere(or_text), TablesOf);
    ExpectConjuncts(out, {"a_x = 1 OR a_x = 2", "b_z = 2 OR b_z = 3", or_text});

    // x = 1 is in every branch but names no known table, so it stays.
    const std::string only_x = "(x = 1 AND a_x = 1 AND b_y = 1) OR (x = 1 AND a_x = 2 AND b_y = 2)";
    ExpectConjuncts(FactorOrConjuncts(SplitWhere(only_x), TablesOf), {"a_x = 1 OR a_x = 2", "b_y = 1 OR b_y = 2", only_x});
}

// Nothing equal to a conjunct already there, or added before, is added again.
TEST_F(OrFactoringTest, NeverAddsADuplicate) {
    const std::string or_text = "(a_x = 1 AND b_y = 2) OR (a_x = 1 AND b_y = 3)";
    ExpectConjuncts(FactorOrConjuncts(SplitWhere("a_x = 1 AND (" + or_text + ")"), TablesOf),
                    {"a_x = 1", "b_y = 2 OR b_y = 3", or_text});
    ExpectConjuncts(FactorOrConjuncts(SplitWhere("(" + or_text + ") AND a_x = 1"), TablesOf),
                    {"b_y = 2 OR b_y = 3", or_text, "a_x = 1"});

    // The same OR twice: its additions come once, both originals stay.
    ExpectConjuncts(FactorOrConjuncts(SplitWhere("(" + or_text + ") AND (" + or_text + ")"), TablesOf),
                    {"a_x = 1", "b_y = 2 OR b_y = 3", or_text, or_text});

    // Two ORs sharing c_w = 5: it comes out once, before the first.
    const std::string first = "(c_w = 5 AND a_x = 1 AND b_y = 1) OR (c_w = 5 AND a_x = 2 AND b_y = 2)";
    const std::string second = "(c_w = 5 AND a_x = 3 AND b_y = 3) OR (c_w = 5 AND a_x = 4 AND b_y = 4)";
    ExpectConjuncts(FactorOrConjuncts(SplitWhere("(" + first + ") AND (" + second + ")"), TablesOf),
                    {"c_w = 5", "a_x = 1 OR a_x = 2", "b_y = 1 OR b_y = 2", first, "a_x = 3 OR a_x = 4",
                     "b_y = 3 OR b_y = 4", second});
}

// x is in both tables, so b_y = x stays above the join.
TEST_F(OrFactoringTest, AmbiguousColumnsAreNeverFactored) {
    auto db = std::make_unique<Database>(TempPath("ambiguous.db"), 64);
    LoadOptions columnar;
    columnar.storage = StorageKind::COLUMNAR;
    const auto load = [&](const std::string& name, const Schema& schema, const std::string& text) {
        const std::string path = TempPath(name + ".tbl");
        std::ofstream(path, std::ios::binary) << text;
        db->LoadTable(name, schema, path, columnar);
    };
    load("t1", Schema({Column("a_k", ColumnType::Integer()), Column("x", ColumnType::Integer()), Column("a_y", ColumnType::Integer())}),
         "1|10|1\n2|30|2\n");
    load("t2", Schema({Column("b_k", ColumnType::Integer()), Column("x", ColumnType::Integer()), Column("b_y", ColumnType::Integer())}),
         "1|20|10\n2|40|30\n");

    const std::string sql = "SELECT a_y, b_y FROM t1, t2 WHERE a_k = b_k AND ((b_y = x AND a_y = 1) OR (b_y = x AND a_y = 2))";
    QueryOptions vectorized{ExecutionMode::VECTORIZED};
    vectorized.forced_join_order = std::vector<int>{0, 1};
    QueryOptions parallel = vectorized;
    parallel.mode = ExecutionMode::PARALLEL;
    parallel.threads = 1;

    // Above the join x reads t1's column, so both rows match.
    const QueryResult want = db->ExecuteSQL(sql, vectorized);
    ASSERT_EQ(want.tuples.size(), 2u);
    const QueryResult got = db->ExecuteSQL(sql, parallel);
    EXPECT_TRUE(tpch::SameResult(got, want)) << got.tuples.size() << " rows";

    // t2's scan gets no filter; t1 still gets its implied OR.
    Parser parser(sql);
    auto stmt = parser.Parse();
    Planner planner = db->MakePlanner(parallel);
    const auto plan = planner.Plan(std::move(*stmt));
    const PlanNode* t2 = ParentOfScan(*plan, "t2");
    ASSERT_NE(t2, nullptr);
    EXPECT_NE(t2->type, PlanNodeType::FILTER);
    const PlanNode* t1 = ParentOfScan(*plan, "t1");
    ASSERT_NE(t1, nullptr);
    ASSERT_EQ(t1->type, PlanNodeType::FILTER);
    EXPECT_EQ(Text(*static_cast<const FilterPlanNode*>(t1)->predicate), Text(*ParseWhere("a_y = 1 OR a_y = 2")));
}

// Binding ignores qualifiers, so above the join t2.x reads t1's x:
// a filter on t2's own x would drop the row the OR keeps.
TEST_F(OrFactoringTest, QualifiedSharedColumnsAreNeverFactored) {
    auto db = LoadSharedX("qualified", "1|20|1\n2|30|2\n", "1|99|10\n2|40|30\n");
    const std::string sql = "SELECT a_k, b_k FROM t1, t2 WHERE a_k = b_k AND ((t2.x = 20 AND a_y = 1) OR (t2.x = 40 AND a_y = 2))";
    ExpectSameAsVectorized(*db, sql, 1);
    EXPECT_EQ(ScanFilter(*db, sql, "t2"), "");
    EXPECT_EQ(ScanFilter(*db, sql, "t1"), Text(*ParseWhere("a_y = 1 OR a_y = 2")));
}

// t1.b_y names a table without b_y; the OR above the join reads t2's.
TEST_F(OrFactoringTest, QualifierNamingAnotherTableIsNeverFactored) {
    auto db = LoadSharedX("other_table", "1|20|1\n2|30|2\n", "1|99|10\n2|40|30\n");
    const std::string sql =
        "SELECT a_k, b_k FROM t1, t2 WHERE a_k = b_k AND ((t1.b_y = 10 AND a_y = 1 AND b_k = 1) OR (t1.b_y = 30 AND a_y = 2 AND b_k = 2))";
    ExpectSameAsVectorized(*db, sql, 2);
    EXPECT_EQ(ScanFilter(*db, sql, "t1"), Text(*ParseWhere("a_y = 1 OR a_y = 2")));
    EXPECT_EQ(ScanFilter(*db, sql, "t2"), Text(*ParseWhere("b_k = 1 OR b_k = 2")));
}

// Planned for the parallel executor, Q19 filters part and lineitem below the join.
TEST_F(OrFactoringTest, Q19NewModePlan) {
    if (data_dir_.empty()) GTEST_SKIP() << "TPC-H SF=0.01 data not found. Run bench/tpch/generate_tpch_data.sh first.";
    const auto q19 = tpch::FindQuery("Q19");
    ASSERT_TRUE(q19.has_value());

    for (const StorageKind storage : {StorageKind::COLUMNAR, StorageKind::ROW}) {
        auto db = Load(storage);
        for (const bool indexes : {false, true}) {
            const std::string what = std::string(storage == StorageKind::COLUMNAR ? "columnar" : "row") +
                                     (indexes ? " indexes" : " no indexes");
            if (indexes) {
                for (const auto& idx : tpch::IndexDescs()) {
                    ASSERT_NE(db->GetCatalog()->CreateIndex(idx.index_name, idx.table_name, idx.column_name), nullptr);
                }
            }
            const auto [tuple_plan, tuple_join] = PlanQuery(*db, q19->sql, ExecutionMode::TUPLE);
            const auto [plan, join] = PlanQuery(*db, q19->sql, ExecutionMode::PARALLEL);

            // The join order recorded when this was written: part (slot 1), then lineitem.
            EXPECT_EQ(join.order, (std::vector<int>{1, 0})) << what;
            EXPECT_EQ(join.order, tuple_join.order) << what;

            // part: the shared size bound and its implied OR, below the join.
            const PlanNode* part = ParentOfScan(*plan, "part");
            ASSERT_NE(part, nullptr) << what;
            ASSERT_EQ(part->type, PlanNodeType::FILTER) << what;
            EXPECT_EQ(Text(*static_cast<const FilterPlanNode*>(part)->predicate),
                      Text(*ParseWhere("p_size >= 1 AND (" + kQ19PartOr + ")")))
                << what;

            // lineitem: the two shared filters, then its implied OR.
            const PlanNode* lineitem = ParentOfScan(*plan, "lineitem");
            ASSERT_NE(lineitem, nullptr) << what;
            ASSERT_EQ(lineitem->type, PlanNodeType::FILTER) << what;
            EXPECT_EQ(Text(*static_cast<const FilterPlanNode*>(lineitem)->predicate),
                      Text(*ParseWhere("l_shipmode IN ('AIR', 'AIR REG') AND (l_shipinstruct = 'DELIVER IN PERSON' AND (" +
                                       kQ19LineitemOrPlanned + "))")))
                << what;

            // Both scans get the same filters as skip hints, unbound copies taken before binding.
            const auto skip_text = [](const PlanNode& scan) {
                std::string out;
                for (const auto& conj : static_cast<const SeqScanPlanNode&>(scan).skip_conjuncts) out += Text(*conj) + "; ";
                return out;
            };
            const auto parsed_text = [](const std::vector<std::string>& conjuncts) {
                std::string out;
                for (const auto& conj : conjuncts) out += Text(*ParseWhere(conj)) + "; ";
                return out;
            };
            EXPECT_EQ(skip_text(*part->children[0]), parsed_text({"p_size >= 1", kQ19PartOr})) << what;
            EXPECT_EQ(skip_text(*lineitem->children[0]),
                      parsed_text({"l_shipmode IN ('AIR', 'AIR REG')", "l_shipinstruct = 'DELIVER IN PERSON'", kQ19LineitemOr}))
                << what;

            // The original OR still runs above the join, as in tuple mode.
            const PlanNode* above = ParentOfJoin(*plan);
            const PlanNode* tuple_above = ParentOfJoin(*tuple_plan);
            ASSERT_NE(above, nullptr) << what;
            ASSERT_NE(tuple_above, nullptr) << what;
            ASSERT_EQ(above->type, PlanNodeType::FILTER) << what;
            ASSERT_EQ(tuple_above->type, PlanNodeType::FILTER) << what;
            const auto* filter = static_cast<const FilterPlanNode*>(above);
            EXPECT_EQ(test::ExprText(filter->predicate.get()),
                      test::ExprText(static_cast<const FilterPlanNode*>(tuple_above)->predicate.get()))
                << what;

            // Tuple mode filters neither table; the estimates see the new filters.
            EXPECT_EQ(ParentOfScan(*tuple_plan, "part")->type, PlanNodeType::HASH_JOIN) << what;
            EXPECT_EQ(ParentOfScan(*tuple_plan, "lineitem")->type, PlanNodeType::HASH_JOIN) << what;
            ASSERT_EQ(join.est_scan_rows.size(), 2u) << what;
            EXPECT_LT(join.est_scan_rows[0], tuple_join.est_scan_rows[0]) << what;
            EXPECT_LT(join.est_scan_rows[1], tuple_join.est_scan_rows[1]) << what;
        }
    }
}

// Q19 in the parallel executor still matches vectorized execution.
TEST_F(OrFactoringTest, Q19AnswerUnchanged) {
    if (data_dir_.empty()) GTEST_SKIP() << "TPC-H SF=0.01 data not found. Run bench/tpch/generate_tpch_data.sh first.";
    auto db = LoadColumnar();
    const auto q19 = tpch::FindQuery("Q19");
    ASSERT_TRUE(q19.has_value());

    const QueryResult vectorized = db->ExecuteSQL(q19->sql, QueryOptions{ExecutionMode::VECTORIZED});
    for (const unsigned threads : {1u, 4u}) {
        QueryOptions options{ExecutionMode::PARALLEL};
        options.threads = threads;
        const QueryResult parallel = db->ExecuteSQL(q19->sql, options);
        ASSERT_EQ(parallel.tuples.size(), 1u) << threads;
        EXPECT_TRUE(tpch::SameResult(parallel, vectorized)) << threads << " threads";
    }
    ASSERT_EQ(vectorized.tuples.size(), 1u);
    EXPECT_GT(vectorized.tuples[0].GetValue(vectorized.schema, 0).decimal_, 0.0);
}

// Tuple and vectorized Q19 plans stay as they were: no implied filters, the same tree.
TEST_F(OrFactoringTest, OtherModesUnchanged) {
    if (data_dir_.empty()) GTEST_SKIP() << "TPC-H SF=0.01 data not found. Run bench/tpch/generate_tpch_data.sh first.";
    auto db = LoadColumnar();
    const auto q19 = tpch::FindQuery("Q19");
    ASSERT_TRUE(q19.has_value());

    const auto [tuple_plan, tuple_join] = PlanQuery(*db, q19->sql, ExecutionMode::TUPLE);
    const auto [vector_plan, vector_join] = PlanQuery(*db, q19->sql, ExecutionMode::VECTORIZED);
    EXPECT_EQ(test::DumpPlan(*tuple_plan), test::DumpPlan(*vector_plan));
    EXPECT_EQ(tuple_join.est_scan_rows, vector_join.est_scan_rows);
    for (const PlanNode* plan : {tuple_plan.get(), vector_plan.get()}) {
        EXPECT_EQ(ParentOfScan(*plan, "part")->type, PlanNodeType::HASH_JOIN);
        EXPECT_EQ(ParentOfScan(*plan, "lineitem")->type, PlanNodeType::HASH_JOIN);
    }
}

}  // namespace shilmandb
