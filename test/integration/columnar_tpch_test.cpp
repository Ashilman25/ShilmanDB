#include <gtest/gtest.h>
#include <unistd.h>

#include "catalog/catalog.hpp"
#include "engine/database.hpp"
#include "executor/executor_context.hpp"
#include "executor/executor_factory.hpp"
#include "planner/plan_node.hpp"
#include "tpch/tpch_workload.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace shilmandb {
namespace {

constexpr size_t kPoolSize = 4096;
constexpr const char* kNoData = "TPC-H SF=0.01 data not found. Run bench/tpch/generate_tpch_data.sh first.";

// The 8 TPC-H money columns, stored as NUMERIC(15,2).
const std::vector<std::string>& MoneyColumns() {
    static const std::vector<std::string> names = {
        "l_quantity", "l_extendedprice", "l_discount", "l_tax",
        "o_totalprice", "c_acctbal", "s_acctbal", "p_retailprice",
    };
    return names;
}

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

const char* ModeName(ExecutionMode mode) {
    return mode == ExecutionMode::TUPLE ? "tuple" : "vectorized";
}

// Hex of the bytes behind p.
std::string HexBytes(const void* p, size_t n) {
    const auto* bytes = static_cast<const unsigned char*>(p);
    std::string out = "0x";
    char buf[3];
    for (size_t i = 0; i < n; ++i) {
        std::snprintf(buf, sizeof(buf), "%02x", bytes[i]);
        out += buf;
    }
    return out;
}

// A value's type and active bytes, for failure messages.
std::string Describe(const Value& v) {
    std::ostringstream out;
    out << "type " << static_cast<int>(v.type_) << " ";
    switch (v.type_) {
        case TypeId::INTEGER: out << HexBytes(&v.integer_, sizeof(v.integer_)); break;
        case TypeId::BIGINT: out << HexBytes(&v.bigint_, sizeof(v.bigint_)); break;
        case TypeId::DECIMAL: out << HexBytes(&v.decimal_, sizeof(v.decimal_)); break;
        case TypeId::DATE: out << HexBytes(&v.date_, sizeof(v.date_)); break;
        case TypeId::VARCHAR: out << HexBytes(v.varchar_.data(), v.varchar_.size()); break;
        case TypeId::INVALID: out << "(null)"; break;
    }
    out << " (" << v.ToString() << ")";
    return out.str();
}

// Same type, then the same bytes in the member that type uses.
bool SameBits(const Value& a, const Value& b) {
    if (a.type_ != b.type_) return false;
    switch (a.type_) {
        case TypeId::INTEGER: return std::memcmp(&a.integer_, &b.integer_, sizeof(a.integer_)) == 0;
        case TypeId::BIGINT: return std::memcmp(&a.bigint_, &b.bigint_, sizeof(a.bigint_)) == 0;
        case TypeId::DECIMAL: return std::memcmp(&a.decimal_, &b.decimal_, sizeof(a.decimal_)) == 0;
        case TypeId::DATE: return std::memcmp(&a.date_, &b.date_, sizeof(a.date_)) == 0;
        case TypeId::VARCHAR: return a.varchar_ == b.varchar_;
        case TypeId::INVALID: return true;
    }
    return false;
}

// Equal shape and every value bit for bit; names the first difference.
::testing::AssertionResult BitIdentical(const QueryResult& want, const QueryResult& got, const std::string& what) {
    const uint32_t cols = want.schema.GetColumnCount();
    if (cols != got.schema.GetColumnCount()) {
        return ::testing::AssertionFailure() << what << ": " << cols << " columns vs "
                                             << got.schema.GetColumnCount();
    }
    for (uint32_t c = 0; c < cols; ++c) {
        const Column& x = want.schema.GetColumn(c);
        const Column& y = got.schema.GetColumn(c);
        if (x.name != y.name || x.type != y.type) {
            return ::testing::AssertionFailure() << what << ": column " << c << " is " << x.name << "/"
                                                 << static_cast<int>(x.type) << " vs " << y.name << "/"
                                                 << static_cast<int>(y.type);
        }
    }
    if (want.tuples.size() != got.tuples.size()) {
        return ::testing::AssertionFailure() << what << ": " << want.tuples.size() << " rows vs "
                                             << got.tuples.size();
    }
    for (size_t r = 0; r < want.tuples.size(); ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            const Value x = want.tuples[r].GetValue(want.schema, c);
            const Value y = got.tuples[r].GetValue(got.schema, c);
            if (!SameBits(x, y)) {
                return ::testing::AssertionFailure()
                       << what << ": row " << r << " column " << c << " (" << want.schema.GetColumn(c).name
                       << ") want " << Describe(x) << ", got " << Describe(y);
            }
        }
    }
    return ::testing::AssertionSuccess();
}

// The join decision and operator tree, without estimates.
struct PlanShape {
    std::vector<std::string> tables;
    std::vector<int> order;
    JoinOrderSource source{JoinOrderSource::SINGLE_TABLE};
    std::vector<TableEdge> edges;
    std::vector<std::pair<std::string, int>> operators;  // name and depth, root first

    bool operator==(const PlanShape& o) const {
        return tables == o.tables && order == o.order && source == o.source && edges == o.edges &&
               operators == o.operators;
    }
    bool operator!=(const PlanShape& o) const { return !(*this == o); }

    std::string ToString() const {
        std::ostringstream out;
        out << "join " << shilmandb::ToString(source) << " order [";
        for (size_t i = 0; i < order.size(); ++i) {
            out << (i ? " " : "") << tables[static_cast<size_t>(order[i])];
        }
        out << "] edges [";
        for (size_t i = 0; i < edges.size(); ++i) {
            out << (i ? " " : "") << edges[i].first << "-" << edges[i].second;
        }
        out << "] ops [";
        for (size_t i = 0; i < operators.size(); ++i) {
            out << (i ? ", " : "") << operators[i].second << ":" << operators[i].first;
        }
        out << "]";
        return out.str();
    }
};

// A query's answer in one mode and the plan it ran with.
struct QueryRun {
    QueryResult result;
    PlanShape plan;
};

// Operators come from a tuple-mode analyze run; analyze has no vectorized mode.
std::vector<std::pair<std::string, int>> OperatorsOf(Database& db, const std::string& sql) {
    QueryOptions options;
    options.analyze = true;
    const auto analyzed = db.ExecuteSQL(sql, options);
    std::vector<std::pair<std::string, int>> ops;
    for (const auto& op : analyzed.profile.operators) ops.emplace_back(op.name, op.depth);
    return ops;
}

// Runs sql in mode and pairs its answer with its plan.
QueryRun RunQuery(Database& db, const std::string& sql, ExecutionMode mode,
                  const std::vector<std::pair<std::string, int>>& operators) {
    QueryRun run;
    run.result = db.ExecuteSQL(sql, mode);
    const JoinOrderInfo& join = run.result.profile.join;
    run.plan.tables = join.tables;
    run.plan.order = join.order;
    run.plan.source = join.source;
    run.plan.edges = join.edges;
    run.plan.operators = operators;
    return run;
}

// Every row of a table read through a full-range scan of one index, all columns.
QueryResult ScanWholeIndex(Database& db, const std::string& table, const std::string& index) {
    const Schema& schema = db.GetCatalog()->GetTable(table)->schema;
    const IndexScanPlanNode plan(schema, table, index);
    ExecutorContext ctx{db.GetBufferPoolManager(), db.GetCatalog()};
    auto exec = ExecutorFactory::CreateExecutor(&plan, &ctx, ExecutionMode::TUPLE);
    QueryResult result;
    result.schema = schema;
    exec->Init();
    Tuple tuple;
    while (exec->Next(&tuple)) result.tuples.push_back(tuple);
    exec->Close();
    return result;
}

class ColumnarTpchTest : public ::testing::Test {
protected:
    static std::string data_dir_;
    // Indexed by [columnar][indexes]; each built the first time a test asks.
    static std::array<std::array<std::unique_ptr<Database>, 2>, 2> dbs_;

    // A pid-named file per storage and index setting.
    static std::string FileFor(bool columnar, bool indexes) {
        const std::string what = std::string(columnar ? "col" : "row") + (indexes ? "_idx" : "_noidx");
        return (std::filesystem::temp_directory_path() /
                ("shilmandb_columnar_tpch_test_" + what + "_" + std::to_string(::getpid()) + ".db"))
            .string();
    }

    // The database for this storage and index setting, loaded on first use.
    static Database& Get(bool columnar, bool indexes) {
        auto& slot = dbs_[columnar][indexes];
        if (slot) return *slot;

        const std::string file = FileFor(columnar, indexes);
        std::filesystem::remove(file);
        slot = std::make_unique<Database>(file, kPoolSize);

        LoadOptions options;
        options.storage = columnar ? StorageKind::COLUMNAR : StorageKind::ROW;
        tpch::LoadTables(*slot, data_dir_, options);

        auto* catalog = slot->GetCatalog();
        if (indexes) {
            for (const auto& idx : tpch::IndexDescs()) {
                (void)catalog->CreateIndex(idx.index_name, idx.table_name, idx.column_name);
            }
        }
        for (const auto& td : tpch::TableDescs()) catalog->UpdateTableStats(td.name);
        return *slot;
    }

    // Closes and reopens the indexed columnar file.
    static Database& ReopenIndexedColumnar() {
        auto& slot = dbs_[1][1];
        slot.reset();
        slot = std::make_unique<Database>(FileFor(true, true), kPoolSize);
        return *slot;
    }

    static void TearDownTestSuite() {
        for (int columnar = 0; columnar < 2; ++columnar) {
            for (int indexes = 0; indexes < 2; ++indexes) {
                dbs_[columnar][indexes].reset();
                std::filesystem::remove(FileFor(columnar != 0, indexes != 0));
            }
        }
    }

    void SetUp() override { data_dir_ = FindDataDir(); }

    // Same plan: same bits. A different plan: same answer within 1e-6, and say so.
    static void CheckQueries(bool indexes, ExecutionMode mode) {
        Database& row = Get(false, indexes);
        Database& col = Get(true, indexes);
        for (const auto& q : tpch::Queries()) {
            const auto row_ops = OperatorsOf(row, q.sql);
            const auto col_ops = OperatorsOf(col, q.sql);
            const std::string what = q.tag + " " + ModeName(mode) + " " + (indexes ? "indexes" : "no-indexes");
            const QueryRun r = RunQuery(row, q.sql, mode, row_ops);
            const QueryRun c = RunQuery(col, q.sql, mode, col_ops);
            ASSERT_FALSE(r.result.tuples.empty()) << what;

            if (r.plan == c.plan) {
                EXPECT_TRUE(BitIdentical(r.result, c.result, what));
            } else {
                std::cout << "PLAN DIFFERS: " << what << "\n"
                          << "  row:      " << r.plan.ToString() << "\n"
                          << "  columnar: " << c.plan.ToString() << "\n";
                EXPECT_TRUE(tpch::SameResult(r.result, c.result)) << what;
            }
        }
    }

    // Closing and reopening the columnar file changes no answer and no plan.
    static void CheckReopen(ExecutionMode mode) {
        const auto queries = tpch::Queries();

        std::vector<QueryRun> before;
        {
            Database& col = Get(true, true);
            for (const auto& q : queries) before.push_back(RunQuery(col, q.sql, mode, OperatorsOf(col, q.sql)));
        }

        Database& reopened = ReopenIndexedColumnar();
        for (size_t i = 0; i < queries.size(); ++i) {
            const auto& q = queries[i];
            const std::string what = q.tag + " " + ModeName(mode) + " after reopen";
            const QueryRun after = RunQuery(reopened, q.sql, mode, OperatorsOf(reopened, q.sql));
            ASSERT_FALSE(after.result.tuples.empty()) << what;
            EXPECT_TRUE(BitIdentical(before[i].result, after.result, what));
            EXPECT_TRUE(before[i].plan == after.plan)
                << what << "\n  before: " << before[i].plan.ToString() << "\n  after:  " << after.plan.ToString();
        }
    }
};

std::string ColumnarTpchTest::data_dir_;
std::array<std::array<std::unique_ptr<Database>, 2>, 2> ColumnarTpchTest::dbs_;

}  // namespace

// The row store's layout is what it was before the money columns became NUMERIC.
TEST_F(ColumnarTpchTest, RowStoreSchemasUnchanged) {
    size_t money_seen = 0;
    for (const auto& td : tpch::TableDescs()) {
        const Schema& shared = td.schema;
        std::vector<Column> old_columns;
        for (const Column& c : shared.GetColumns()) old_columns.emplace_back(c.name, c.type, c.nullable);
        const Schema old(old_columns);

        ASSERT_EQ(shared.GetColumnCount(), old.GetColumnCount()) << td.name;
        EXPECT_EQ(shared.GetFixedLength(), old.GetFixedLength()) << td.name;
        EXPECT_EQ(shared.GetVarcharOffsets(), old.GetVarcharOffsets()) << td.name;
        for (uint32_t i = 0; i < shared.GetColumnCount(); ++i) {
            const Column& c = shared.GetColumn(i);
            EXPECT_EQ(c.type, old.GetColumn(i).type) << c.name;
            EXPECT_EQ(c.fixed_length, old.GetColumn(i).fixed_length) << c.name;
            EXPECT_EQ(shared.GetOffset(i), old.GetOffset(i)) << c.name;

            const auto& money = MoneyColumns();
            const bool is_money = std::find(money.begin(), money.end(), c.name) != money.end();
            if (is_money) {
                ++money_seen;
                EXPECT_EQ(c.type, TypeId::DECIMAL) << c.name;
                EXPECT_TRUE(c.storage_type == ColumnType::Numeric(15, 2))
                    << c.name << " stores " << ColumnTypeName(c.storage_type);
            } else {
                EXPECT_TRUE(c.storage_type == DefaultColumnType(c.type))
                    << c.name << " stores " << ColumnTypeName(c.storage_type);
            }
        }
    }
    EXPECT_EQ(money_seen, MoneyColumns().size());
}

// A two-column result (DECIMAL, VARCHAR) built by hand.
static QueryResult MakeResult(const std::vector<std::pair<double, std::string>>& rows) {
    QueryResult result;
    result.schema = Schema({{"d", TypeId::DECIMAL}, {"s", TypeId::VARCHAR}});
    for (const auto& [d, str] : rows) result.tuples.emplace_back(std::vector<Value>{Value(d), Value(str)}, result.schema);
    return result;
}

// A one-column result holding n as an INTEGER or a BIGINT.
static QueryResult MakeNumber(TypeId type, int32_t n) {
    QueryResult result;
    result.schema = Schema({{"n", type}});
    const Value v = type == TypeId::INTEGER ? Value(n) : Value(static_cast<int64_t>(n));
    result.tuples.emplace_back(std::vector<Value>{v}, result.schema);
    return result;
}

// The helpers the TPC-H tests trust really see small differences.
TEST_F(ColumnarTpchTest, ComparisonsCatchDifferences) {
    const double price = 1234.56;
    const QueryResult base = MakeResult({{price, "BUILDING"}, {0.0, "MAIL"}});

    EXPECT_TRUE(BitIdentical(base, MakeResult({{price, "BUILDING"}, {0.0, "MAIL"}}), "identical"));
    EXPECT_FALSE(BitIdentical(base, MakeResult({{std::nextafter(price, 1e9), "BUILDING"}, {0.0, "MAIL"}}), "one ulp"));
    EXPECT_FALSE(BitIdentical(base, MakeResult({{price, "BUILDING"}, {-0.0, "MAIL"}}), "negative zero"));
    EXPECT_FALSE(BitIdentical(base, MakeResult({{price, "BUILDINH"}, {0.0, "MAIL"}}), "one byte"));
    EXPECT_FALSE(BitIdentical(base, MakeResult({{0.0, "MAIL"}, {price, "BUILDING"}}), "rows swapped"));
    EXPECT_FALSE(BitIdentical(base, MakeResult({{price, "BUILDING"}}), "row count"));

    EXPECT_TRUE(BitIdentical(MakeNumber(TypeId::BIGINT, 7), MakeNumber(TypeId::BIGINT, 7), "same bigint"));
    EXPECT_FALSE(BitIdentical(MakeNumber(TypeId::INTEGER, 7), MakeNumber(TypeId::BIGINT, 7), "integer vs bigint"));
    EXPECT_FALSE(SameBits(Value(int32_t{7}), Value(int64_t{7})));

    PlanShape plan;
    plan.tables = {"orders", "lineitem"};
    plan.order = {0, 1};
    plan.source = JoinOrderSource::CLASSICAL;
    plan.edges = {{0, 1}};
    plan.operators = {{"HashJoin +lineitem", 0}, {"SeqScan orders", 1}, {"SeqScan lineitem", 1}};

    PlanShape same = plan;
    EXPECT_TRUE(plan == same);

    PlanShape other_order = plan;
    other_order.order = {1, 0};
    EXPECT_FALSE(plan == other_order);

    PlanShape other_name = plan;
    other_name.operators[1].first = "IndexScan orders";
    EXPECT_FALSE(plan == other_name);

    PlanShape other_depth = plan;
    other_depth.operators[2].second = 2;
    EXPECT_FALSE(plan == other_depth);
}

// Every scanned value matches the row store, in both modes.
TEST_F(ColumnarTpchTest, EveryValueBitIdentical) {
    if (data_dir_.empty()) GTEST_SKIP() << kNoData;
    Database& row = Get(false, false);
    Database& col = Get(true, false);
    for (const auto& td : tpch::TableDescs()) {
        const std::string sql = "SELECT * FROM " + td.name;
        const auto want = row.ExecuteSQL(sql, ExecutionMode::TUPLE);
        ASSERT_EQ(want.tuples.size(), row.GetCatalog()->GetTable(td.name)->stats.row_count) << td.name;
        ASSERT_GT(want.tuples.size(), 0u) << td.name;

        EXPECT_TRUE(BitIdentical(want, row.ExecuteSQL(sql, ExecutionMode::VECTORIZED), td.name + " row vectorized"));
        EXPECT_TRUE(BitIdentical(want, col.ExecuteSQL(sql, ExecutionMode::TUPLE), td.name + " columnar tuple"));
        EXPECT_TRUE(BitIdentical(want, col.ExecuteSQL(sql, ExecutionMode::VECTORIZED), td.name + " columnar vectorized"));
    }
}

// Every value fetched through each index matches the row store, in index order.
TEST_F(ColumnarTpchTest, EveryIndexFetchBitIdentical) {
    if (data_dir_.empty()) GTEST_SKIP() << kNoData;
    Database& row = Get(false, true);
    Database& col = Get(true, true);
    for (const auto& idx : tpch::IndexDescs()) {
        const QueryResult want = ScanWholeIndex(row, idx.table_name, idx.index_name);
        ASSERT_EQ(want.tuples.size(), row.GetCatalog()->GetTable(idx.table_name)->stats.row_count) << idx.index_name;
        ASSERT_GT(want.tuples.size(), 0u) << idx.index_name;
        EXPECT_TRUE(BitIdentical(want, ScanWholeIndex(col, idx.table_name, idx.index_name), idx.index_name));
    }
}

TEST_F(ColumnarTpchTest, QueriesMatchRowStoreIndexedTuple) {
    if (data_dir_.empty()) GTEST_SKIP() << kNoData;
    CheckQueries(true, ExecutionMode::TUPLE);
}

TEST_F(ColumnarTpchTest, QueriesMatchRowStoreIndexedVectorized) {
    if (data_dir_.empty()) GTEST_SKIP() << kNoData;
    CheckQueries(true, ExecutionMode::VECTORIZED);
}

TEST_F(ColumnarTpchTest, QueriesMatchRowStoreNoIndexTuple) {
    if (data_dir_.empty()) GTEST_SKIP() << kNoData;
    CheckQueries(false, ExecutionMode::TUPLE);
}

TEST_F(ColumnarTpchTest, QueriesMatchRowStoreNoIndexVectorized) {
    if (data_dir_.empty()) GTEST_SKIP() << kNoData;
    CheckQueries(false, ExecutionMode::VECTORIZED);
}

TEST_F(ColumnarTpchTest, ReopenGivesSameAnswersAndPlansTuple) {
    if (data_dir_.empty()) GTEST_SKIP() << kNoData;
    CheckReopen(ExecutionMode::TUPLE);
}

TEST_F(ColumnarTpchTest, ReopenGivesSameAnswersAndPlansVectorized) {
    if (data_dir_.empty()) GTEST_SKIP() << kNoData;
    CheckReopen(ExecutionMode::VECTORIZED);
}

}  // namespace shilmandb
