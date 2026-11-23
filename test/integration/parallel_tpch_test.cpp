#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "catalog/catalog.hpp"
#include "engine/database.hpp"
#include "executor/parallel/decimal.hpp"
#include "parser/parser.hpp"
#include "planner/plan_dump.hpp"
#include "planner/planner.hpp"
#include "tpch/tpch_workload.hpp"

namespace shilmandb {
namespace {

// The SF=0.01 data directory, or "" when it hasn't been generated.
std::string FindDataDir() {
    for (const std::string dir : {"bench/tpch_data/sf0.01/", "../bench/tpch_data/sf0.01/", "../../bench/tpch_data/sf0.01/"}) {
        if (std::filesystem::exists(dir + "lineitem.tbl")) return dir;
    }
    return {};
}

// True when the plan for sql in this mode reads a table through an index.
bool PlansIndexScan(const Database& db, const std::string& sql, ExecutionMode mode) {
    Parser parser(sql);
    auto stmt = parser.Parse();
    QueryOptions options;
    options.mode = mode;
    Planner planner = db.MakePlanner(options);
    const auto plan = planner.Plan(std::move(*stmt));
    std::vector<const PlanNode*> stack{plan.get()};
    while (!stack.empty()) {
        const PlanNode* node = stack.back();
        stack.pop_back();
        if (node->type == PlanNodeType::INDEX_SCAN) return true;
        for (const auto& child : node->children) stack.push_back(child.get());
    }
    return false;
}

// A money field like "901.00" as whole cents.
__int128 Cents(const std::string& text) {
    const size_t dot = text.find('.');
    const std::string whole = text.substr(0, dot);
    std::string frac = dot == std::string::npos ? "" : text.substr(dot + 1);
    frac.resize(2, '0');
    const bool negative = !whole.empty() && whole[0] == '-';
    const __int128 units = static_cast<__int128>(std::stoll(negative ? whole.substr(1) : whole)) * 100 + std::stoll(frac);
    return negative ? -units : units;
}

// With or without the 8 indexes.
enum class Indexes { kNone, kAll };

// Names the setting in test output, so ctest shows /NoIndexes and /WithIndexes.
void PrintTo(Indexes setting, std::ostream* os) {
    *os << (setting == Indexes::kAll ? "WithIndexes" : "NoIndexes");
}

// One fixture per index setting: TPC-H SF=0.01 loaded columnar, once for the suite.
class ParallelTpchTest : public ::testing::TestWithParam<Indexes> {
protected:
    static std::unique_ptr<Database> plain_db_;
    static std::unique_ptr<Database> indexed_db_;
    static std::string plain_file_;
    static std::string indexed_file_;
    static std::string data_dir_;
    Database* db_{nullptr};

    // Temp file names carry the process id, so parallel ctest runs never collide.
    static void SetUpTestSuite() {
        const auto temp = std::filesystem::temp_directory_path();
        const std::string pid = std::to_string(::getpid());
        plain_file_ = (temp / ("shilmandb_parallel_tpch_test_plain_" + pid + ".db")).string();
        indexed_file_ = (temp / ("shilmandb_parallel_tpch_test_indexed_" + pid + ".db")).string();
        data_dir_ = FindDataDir();
    }

    // Closes both databases before their files go.
    static void TearDownTestSuite() {
        plain_db_.reset();
        indexed_db_.reset();
        std::filesystem::remove(plain_file_);
        std::filesystem::remove(indexed_file_);
    }

    // Loads every table columnar, then the 8 indexes when asked, then stats.
    static std::unique_ptr<Database> Load(const std::string& file, bool indexes) {
        std::filesystem::remove(file);
        auto db = std::make_unique<Database>(file, 4096);
        LoadOptions options;
        options.storage = StorageKind::COLUMNAR;
        for (const auto& td : tpch::TableDescs()) db->LoadTable(td.name, td.schema, data_dir_ + td.tbl_file, options);
        auto* catalog = db->GetCatalog();
        if (indexes) {
            for (const auto& idx : tpch::IndexDescs()) (void)catalog->CreateIndex(idx.index_name, idx.table_name, idx.column_name);
        }
        for (const auto& td : tpch::TableDescs()) catalog->UpdateTableStats(td.name);
        return db;
    }

    // Loads only this test's setting, the first time a test asks for it.
    void SetUp() override {
        if (data_dir_.empty()) GTEST_SKIP() << "TPC-H SF=0.01 data not found. Run bench/tpch/generate_tpch_data.sh first.";
        const bool indexed = GetParam() == Indexes::kAll;
        auto& slot = indexed ? indexed_db_ : plain_db_;
        if (!slot) slot = Load(indexed ? indexed_file_ : plain_file_, indexed);
        db_ = slot.get();
    }

    // Runs sql in one mode; the parallel executor on one thread.
    QueryResult Run(const std::string& sql, ExecutionMode mode) {
        QueryOptions options;
        options.mode = mode;
        options.threads = 1;
        return db_->ExecuteSQL(sql, options);
    }

    // sql's COUNT columns in today's plan, which parallel plans type BIGINT.
    std::vector<bool> Counts(const std::string& sql) const {
        return test::CountColumns(*db_->MakePlanner(QueryOptions{}).Plan(std::move(*Parser(sql).Parse())));
    }
};

std::unique_ptr<Database> ParallelTpchTest::plain_db_;
std::unique_ptr<Database> ParallelTpchTest::indexed_db_;
std::string ParallelTpchTest::plain_file_;
std::string ParallelTpchTest::indexed_file_;
std::string ParallelTpchTest::data_dir_;

INSTANTIATE_TEST_SUITE_P(Indexes, ParallelTpchTest, ::testing::Values(Indexes::kNone, Indexes::kAll),
                         [](const ::testing::TestParamInfo<Indexes>& info) {
                             return std::string(info.param == Indexes::kAll ? "WithIndexes" : "NoIndexes");
                         });

TEST_P(ParallelTpchTest, AllQueriesMatchVectorized) {
    int index_scans = 0;
    int count_columns = 0;
    for (const auto& q : tpch::Queries()) {
        SCOPED_TRACE(q.tag);
        const QueryResult want = Run(q.sql, ExecutionMode::VECTORIZED);
        const QueryResult got = Run(q.sql, ExecutionMode::PARALLEL);

        // Rows compare in order, so the order must be fixed.
        const bool ordered = q.sql.find("ORDER BY") != std::string::npos;
        ASSERT_TRUE(ordered || want.tuples.size() == 1u) << "an unordered query with several rows";
        ASSERT_FALSE(want.tuples.empty());

        // Today's types, except COUNT columns: BIGINT here, INTEGER today.
        ASSERT_EQ(got.schema.GetColumnCount(), want.schema.GetColumnCount());
        const std::vector<bool> counts = Counts(q.sql);
        ASSERT_EQ(counts.size(), want.schema.GetColumnCount());
        for (uint32_t c = 0; c < want.schema.GetColumnCount(); ++c) {
            const TypeId today = want.schema.GetColumn(c).type;
            if (counts[c]) {
                ++count_columns;
                EXPECT_EQ(today, TypeId::INTEGER) << "column " << c;
            }
            EXPECT_EQ(got.schema.GetColumn(c).type, counts[c] ? TypeId::BIGINT : today) << "column " << c;
        }
        EXPECT_EQ(got.tuples.size(), want.tuples.size());
        EXPECT_TRUE(tpch::SameResult(got, want));
        if (PlansIndexScan(*db_, q.sql, ExecutionMode::PARALLEL)) ++index_scans;
    }
    // Only Q1 counts: its count_star.
    EXPECT_EQ(count_columns, 1);
    // With the indexes, some queries (Q10, Q14) read through one.
    if (GetParam() == Indexes::kAll) {
        EXPECT_GT(index_scans, 0);
    } else {
        EXPECT_EQ(index_scans, 0);
    }
}

TEST_P(ParallelTpchTest, ExactWhereTodayIsExact) {
    for (const auto& q : tpch::Queries()) {
        SCOPED_TRACE(q.tag);
        const QueryResult want = Run(q.sql, ExecutionMode::VECTORIZED);
        const QueryResult got = Run(q.sql, ExecutionMode::PARALLEL);
        ASSERT_EQ(got.tuples.size(), want.tuples.size());
        ASSERT_EQ(got.schema.GetColumnCount(), want.schema.GetColumnCount());
        const std::vector<bool> counts = Counts(q.sql);
        ASSERT_EQ(counts.size(), want.schema.GetColumnCount());
        for (size_t r = 0; r < want.tuples.size(); ++r) {
            for (uint32_t c = 0; c < want.schema.GetColumnCount(); ++c) {
                const Value x = got.tuples[r].GetValue(got.schema, c);
                const Value y = want.tuples[r].GetValue(want.schema, c);
                if (y.type_ == TypeId::DECIMAL) continue;
                // A COUNT is the same number, as 64 bits here.
                if (counts[c]) {
                    ASSERT_EQ(x.type_, TypeId::BIGINT) << "row " << r << " col " << c;
                    ASSERT_EQ(y.type_, TypeId::INTEGER) << "row " << r << " col " << c;
                    EXPECT_EQ(x.bigint_, y.integer_) << "row " << r << " col " << c;
                    continue;
                }
                ASSERT_EQ(x.type_, y.type_) << "row " << r << " col " << c;
                switch (y.type_) {
                    case TypeId::INTEGER:
                        EXPECT_EQ(x.integer_, y.integer_) << "row " << r << " col " << c;
                        break;
                    case TypeId::BIGINT:
                        EXPECT_EQ(x.bigint_, y.bigint_) << "row " << r << " col " << c;
                        break;
                    case TypeId::DATE:
                        EXPECT_EQ(x.date_, y.date_) << "row " << r << " col " << c;
                        break;
                    default:
                        EXPECT_EQ(x.varchar_, y.varchar_) << "row " << r << " col " << c;
                        break;
                }
            }
        }
    }
}

TEST_P(ParallelTpchTest, Q1SumsAreExact) {
    // The exact sums of each (returnflag, linestatus) group, straight from the file.
    struct Sums {
        __int128 quantity{0};
        __int128 price{0};
        __int128 discount{0};
    };
    std::map<std::pair<std::string, std::string>, Sums> sums;
    std::ifstream in(data_dir_ + "lineitem.tbl");
    ASSERT_TRUE(in.good());
    std::string line;
    while (std::getline(in, line)) {
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string field;
        while (std::getline(ss, field, '|')) fields.push_back(field);
        ASSERT_GE(fields.size(), 11u);
        if (fields[10] > "1998-09-02") continue;
        Sums& s = sums[{fields[8], fields[9]}];
        s.quantity += Cents(fields[4]);
        s.price += Cents(fields[5]);
        s.discount += Cents(fields[6]);
    }
    ASSERT_EQ(sums.size(), 4u);

    const QueryResult got = Run(tpch::FindQuery("Q1")->sql, ExecutionMode::PARALLEL);
    ASSERT_EQ(got.tuples.size(), 4u);
    size_t r = 0;
    for (const auto& [key, s] : sums) {
        const Tuple& t = got.tuples[r++];
        ASSERT_EQ(t.GetValue(got.schema, 0).varchar_, key.first);
        ASSERT_EQ(t.GetValue(got.schema, 1).varchar_, key.second);
        // Bit for bit: the nearest double to each exact sum.
        EXPECT_EQ(t.GetValue(got.schema, 3).decimal_, parallel::DecimalToDouble(s.price, 2)) << key.first << key.second;
        EXPECT_EQ(t.GetValue(got.schema, 2).decimal_, parallel::DecimalToDouble(s.quantity, 2)) << key.first << key.second;
        EXPECT_EQ(t.GetValue(got.schema, 4).decimal_, parallel::DecimalToDouble(s.discount, 2)) << key.first << key.second;
    }
}

}  // namespace
}  // namespace shilmandb
