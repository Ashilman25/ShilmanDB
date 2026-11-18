#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "buffer/buffer_pool_manager.hpp"
#include "catalog/catalog.hpp"
#include "common/exception.hpp"
#include "engine/database.hpp"

namespace shilmandb {
namespace {

constexpr uint32_t kRows = 200000;  // four row groups: three full, one of 3,392 rows
constexpr size_t kGroups = 4;

// One column of every storage type, a to k.
Schema AllTypes() {
    return Schema({Column("a", ColumnType::Integer()), Column("b", ColumnType::Numeric(15, 2)),
                   Column("c", ColumnType::SmallInt()), Column("d", ColumnType::BigInt()),
                   Column("e", ColumnType::Boolean()), Column("f", ColumnType::Double()),
                   Column("g", ColumnType::Date()), Column("h", ColumnType::Timestamp()),
                   Column("i", ColumnType::Varchar(12)), Column("j", ColumnType::Text()),
                   Column("k", ColumnType::Numeric(25, 3))});
}

// Row r of the table: a is r, so each row group holds one stretch.
std::string AllTypesLine(uint32_t r) {
    static const char* const kWords[] = {"alpha", "beta", "", "gamma delta", "épée"};
    const uint32_t money = (r * 7919u) % 1000000u;
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "%u|%s%u.%02u|%u|%lld|%s|%.2f|%04u-%02u-%02u|%04u-%02u-%02u %02u:%02u:%02u%s|%s|row-%u%s|%s%u000000000000.%03u\n",
                  r, r % 3 == 0 ? "-" : "", money / 100, money % 100, r % 1000,
                  static_cast<long long>(r) * 1000003LL - 50000000000LL, r % 2 ? "t" : "f", r / 4.0 - 1000.5,
                  1992 + r % 7, 1 + r % 12, 1 + r % 28,
                  2000 + r % 20, 1 + r % 12, 1 + r % 28, r % 24, r % 60, (r / 60) % 60,
                  r % 5 == 0 ? ".123456" : "", kWords[r % 5], r, r % 7 == 0 ? " ünï" : "",
                  r % 4 == 1 ? "-" : "", r, r % 1000);
    return buf;
}

// Runs fn and returns what it threw, or "" if it didn't throw.
template <typename Fn>
std::string ErrorOf(Fn&& fn) {
    try {
        fn();
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

// Columnar storage on two threads.
LoadOptions Columnar() {
    LoadOptions options;
    options.storage = StorageKind::COLUMNAR;
    options.threads = 2;
    return options;
}

// A result row: its non-decimal cells as text and its decimals as numbers.
struct Row {
    std::vector<std::string> text;
    std::vector<double> decimals;

    bool operator<(const Row& o) const { return text != o.text ? text < o.text : decimals < o.decimals; }
};

// Row i of a result, split into text cells and decimals.
Row RowOf(const QueryResult& r, size_t i) {
    Row row;
    for (uint32_t c = 0; c < r.schema.GetColumnCount(); ++c) {
        const Value v = r.tuples[i].GetValue(r.schema, c);
        if (v.type_ == TypeId::DECIMAL) {
            row.decimals.push_back(v.decimal_);
        } else {
            row.text.push_back(v.ToString());
        }
    }
    return row;
}

// Exact text cells, decimals within SameResult's 1e-6 relative.
bool SameRow(const Row& x, const Row& y) {
    if (x.text != y.text || x.decimals.size() != y.decimals.size()) return false;
    for (size_t d = 0; d < x.decimals.size(); ++d) {
        const double a = x.decimals[d];
        const double b = y.decimals[d];
        if (std::abs(a - b) > 1e-6 * std::max({1.0, std::abs(a), std::abs(b)})) return false;
    }
    return true;
}

// A row as one line of text, for failure messages.
std::string Describe(const Row& row) {
    std::string out;
    for (const auto& t : row.text) out += t + " | ";
    for (double d : row.decimals) out += std::to_string(d) + " | ";
    return out;
}

// Every row of a result, sorted so results compare as multisets.
std::vector<Row> SortedRows(const QueryResult& r) {
    std::vector<Row> rows;
    rows.reserve(r.tuples.size());
    for (size_t i = 0; i < r.tuples.size(); ++i) rows.push_back(RowOf(r, i));
    std::sort(rows.begin(), rows.end());
    return rows;
}

// How got and want first differ as multisets of rows, or "" when they match.
std::string Difference(const QueryResult& got, const QueryResult& want) {
    if (got.schema.GetColumnCount() != want.schema.GetColumnCount()) return "column counts differ";
    for (uint32_t c = 0; c < got.schema.GetColumnCount(); ++c) {
        if (got.schema.GetColumn(c).type != want.schema.GetColumn(c).type) return "column " + std::to_string(c) + " has another type";
    }
    if (got.tuples.size() != want.tuples.size()) {
        return std::to_string(got.tuples.size()) + " rows, expected " + std::to_string(want.tuples.size());
    }
    const std::vector<Row> x = SortedRows(got);
    const std::vector<Row> y = SortedRows(want);
    for (size_t i = 0; i < x.size(); ++i) {
        if (!SameRow(x[i], y[i])) return "row " + std::to_string(i) + ": got " + Describe(x[i]) + "expected " + Describe(y[i]);
    }
    return "";
}

// a, then the bit patterns of decimals b and f, sorted by a.
std::vector<std::array<uint64_t, 3>> PassThroughBits(const QueryResult& r) {
    std::vector<std::array<uint64_t, 3>> rows;
    rows.reserve(r.tuples.size());
    for (const Tuple& t : r.tuples) {
        std::array<uint64_t, 3> row{};
        row[0] = static_cast<uint64_t>(t.GetValue(r.schema, 0).integer_);
        const double b = t.GetValue(r.schema, 1).decimal_;
        const double f = t.GetValue(r.schema, 2).decimal_;
        std::memcpy(&row[1], &b, sizeof(b));
        std::memcpy(&row[2], &f, sizeof(f));
        rows.push_back(row);
    }
    std::sort(rows.begin(), rows.end());
    return rows;
}

class ParallelBasicTest : public ::testing::Test {
protected:
    std::string db_file_;
    std::vector<std::string> tbl_files_;
    std::unique_ptr<Database> db_;

    void SetUp() override {
        db_file_ = TempPath(".db");
        std::filesystem::remove(db_file_);

        std::string text;
        text.reserve(static_cast<size_t>(kRows) * 140);
        for (uint32_t r = 0; r < kRows; ++r) text += AllTypesLine(r);

        db_ = std::make_unique<Database>(db_file_);
        db_->LoadTable("t", AllTypes(), WriteTbl("t", text), Columnar());
        ASSERT_EQ(db_->GetCatalog()->GetTable("t")->columnar->RowGroupCount(), kGroups);
    }

    void TearDown() override {
        db_.reset();
        std::filesystem::remove(db_file_);
        for (const auto& f : tbl_files_) std::filesystem::remove(f);
    }

    // A temp path unique to this process.
    static std::string TempPath(const std::string& suffix) {
        return (std::filesystem::temp_directory_path() /
                ("shilmandb_parallel_basic_test_" + std::to_string(::getpid()) + suffix)).string();
    }

    // Writes text to a per-test .tbl file and returns its path.
    std::string WriteTbl(const std::string& what, const std::string& text) {
        const std::string path = TempPath("_" + what + ".tbl");
        std::ofstream(path, std::ios::binary) << text;
        tbl_files_.push_back(path);
        return path;
    }

    // Runs sql in the given mode; threads matters only for PARALLEL.
    QueryResult Run(const std::string& sql, ExecutionMode mode, unsigned threads = 0) {
        QueryOptions options;
        options.mode = mode;
        options.threads = threads;
        return db_->ExecuteSQL(sql, options);
    }

    // Pages of the listed columns of t in row groups [from, to).
    uint64_t PagesOf(const std::vector<uint32_t>& columns, size_t from, size_t to) {
        const auto& meta = db_->GetCatalog()->GetTable("t")->columnar->Meta();
        uint64_t pages = 0;
        for (size_t g = from; g < to; ++g) {
            for (uint32_t c : columns) pages += meta.row_groups[g].chunks[c].page_count;
        }
        return pages;
    }
};

TEST_F(ParallelBasicTest, ScanProjectFilterLimit) {
    const std::string sql = "SELECT a, b * 2 FROM t WHERE c < 100";
    const QueryResult want = Run(sql, ExecutionMode::VECTORIZED);
    ASSERT_EQ(want.tuples.size(), 20000u);
    const QueryResult got = Run(sql, ExecutionMode::PARALLEL);
    EXPECT_EQ(Difference(got, want), "");
    EXPECT_EQ(Difference(Run(sql, ExecutionMode::PARALLEL, 3), want), "");

    // Every storage type passes through to the result as today's scan gives it.
    const std::string all = "SELECT * FROM t WHERE c < 100";
    EXPECT_EQ(Difference(Run(all, ExecutionMode::PARALLEL), Run(all, ExecutionMode::VECTORIZED)), "");

    // A scan that hands up no column still counts its rows.
    const QueryResult constant = Run("SELECT 7 FROM t", ExecutionMode::PARALLEL);
    ASSERT_EQ(constant.tuples.size(), kRows);
    EXPECT_EQ(constant.tuples[0].GetValue(constant.schema, 0).integer_, 7);
    EXPECT_EQ(constant.tuples[kRows - 1].GetValue(constant.schema, 0).integer_, 7);

    // Computed values of each kind.
    const std::string computed = "SELECT a, k + b, f * 2, d - 1, -c, i, j FROM t WHERE c < 100 AND i <> 'beta'";
    EXPECT_EQ(Difference(Run(computed, ExecutionMode::PARALLEL), Run(computed, ExecutionMode::VECTORIZED)), "");

    // A LIMIT gives 7 rows of the full answer, reading only the first group.
    std::map<std::string, Row> full;
    for (size_t i = 0; i < got.tuples.size(); ++i) {
        const Row row = RowOf(got, i);
        full.emplace(row.text[0], row);
    }
    auto* bpm = db_->GetBufferPoolManager();
    bpm->DropAllPages();
    bpm->ResetStats();
    const QueryResult limited = Run(sql + " LIMIT 7", ExecutionMode::PARALLEL);
    ASSERT_EQ(limited.tuples.size(), 7u);
    EXPECT_LE(bpm->GetMissCount(), PagesOf({0, 1, 2}, 0, 1));
    EXPECT_GT(bpm->GetMissCount(), 0u);
    std::vector<std::string> keys;
    for (size_t i = 0; i < limited.tuples.size(); ++i) {
        const Row row = RowOf(limited, i);
        auto it = full.find(row.text[0]);
        ASSERT_NE(it, full.end()) << Describe(row);
        EXPECT_TRUE(SameRow(row, it->second)) << Describe(row);
        keys.push_back(row.text[0]);
    }
    std::sort(keys.begin(), keys.end());
    EXPECT_EQ(std::unique(keys.begin(), keys.end()), keys.end()) << "a row came back twice";

    EXPECT_TRUE(Run(sql + " LIMIT 0", ExecutionMode::PARALLEL).tuples.empty());
}

TEST_F(ParallelBasicTest, SkipsRowGroupsByZoneMap) {
    auto* bpm = db_->GetBufferPoolManager();
    bpm->DropAllPages();
    bpm->ResetStats();
    const QueryResult third = Run("SELECT a FROM t WHERE a >= 131072 AND a < 196608", ExecutionMode::PARALLEL);
    EXPECT_EQ(third.tuples.size(), 65536u);
    EXPECT_EQ(bpm->GetMissCount(), PagesOf({0}, 2, 3));

    // No group may match, so no page is read.
    bpm->DropAllPages();
    bpm->ResetStats();
    EXPECT_TRUE(Run("SELECT a FROM t WHERE a > 200000", ExecutionMode::PARALLEL).tuples.empty());
    EXPECT_EQ(bpm->GetMissCount(), 0u);
}

TEST_F(ParallelBasicTest, PassThroughValuesAreBitIdentical) {
    // NUMERIC(15,2) and DOUBLE leave exactly as today's scan gives them, bit for bit.
    const std::string sql = "SELECT a, b, f FROM t";
    const QueryResult got = Run(sql, ExecutionMode::PARALLEL);
    const QueryResult want = Run(sql, ExecutionMode::VECTORIZED);
    ASSERT_EQ(got.schema.GetColumn(1).type, TypeId::DECIMAL);
    ASSERT_EQ(got.schema.GetColumn(2).type, TypeId::DECIMAL);
    const auto x = PassThroughBits(got);
    const auto y = PassThroughBits(want);
    ASSERT_EQ(x.size(), static_cast<size_t>(kRows));
    ASSERT_EQ(y.size(), static_cast<size_t>(kRows));
    for (size_t i = 0; i < x.size(); ++i) {
        ASSERT_EQ(x[i], y[i]) << "row with a = " << x[i][0];
    }
}

TEST_F(ParallelBasicTest, FilterKeepsNothingInSomeBatches) {
    // a + 0 is no plain column, so no zone map can skip a group.
    auto* bpm = db_->GetBufferPoolManager();
    bpm->DropAllPages();
    bpm->ResetStats();
    const std::string none = "SELECT a, c, i FROM t WHERE a + 0 > 300000";
    const QueryResult empty = Run(none, ExecutionMode::PARALLEL);
    EXPECT_TRUE(empty.tuples.empty());
    EXPECT_GE(bpm->GetMissCount(), PagesOf({0}, 0, kGroups));
    EXPECT_EQ(Difference(empty, Run(none, ExecutionMode::VECTORIZED)), "");

    // Rows that span two groups; every other batch keeps none.
    const std::string some = "SELECT a, c, i FROM t WHERE a + 0 >= 131000 AND a + 0 < 133000";
    const QueryResult got = Run(some, ExecutionMode::PARALLEL);
    EXPECT_EQ(got.tuples.size(), 2000u);
    EXPECT_EQ(Difference(got, Run(some, ExecutionMode::VECTORIZED)), "");
}

TEST_F(ParallelBasicTest, RowStoreTableIsRefused) {
    db_->LoadTable("r", Schema({Column("ra", TypeId::INTEGER)}), WriteTbl("r", "1\n2\n"));
    const std::string want = "the parallel executor reads only columnar tables: r";
    EXPECT_EQ(ErrorOf([&] { (void)Run("SELECT ra FROM r", ExecutionMode::PARALLEL); }), want);
    // Checked before anything else, even under a node this executor can't run yet.
    EXPECT_EQ(ErrorOf([&] { (void)Run("SELECT a FROM t, r WHERE a = ra", ExecutionMode::PARALLEL); }), want);
    EXPECT_EQ(Run("SELECT ra FROM r", ExecutionMode::TUPLE).tuples.size(), 2u);
}

TEST_F(ParallelBasicTest, UnsupportedNodeIsRefused) {
    db_->LoadTable("u", Schema({Column("ua", ColumnType::Integer())}), WriteTbl("u", "1\n2\n"), Columnar());
    EXPECT_EQ(ErrorOf([&] { (void)Run("SELECT a FROM t, u WHERE a = ua", ExecutionMode::PARALLEL); }),
              "the parallel executor can't run HASH_JOIN yet");
}

TEST_F(ParallelBasicTest, NullIsRefused) {
    LoadOptions options = Columnar();
    options.null_text = "";
    const Schema schema({Column("x", ColumnType::Integer(), true), Column("y", ColumnType::Integer())});
    db_->LoadTable("n", schema, WriteTbl("n", "1|5\n|6\n3|7\n"), options);

    const std::string today = "column 'x' holds NULLs, which queries can't read yet";
    EXPECT_EQ(ErrorOf([&] { (void)Run("SELECT x FROM n", ExecutionMode::VECTORIZED); }), today);
    EXPECT_EQ(ErrorOf([&] { (void)Run("SELECT x FROM n", ExecutionMode::PARALLEL); }), today);
    // Only the columns a query reads are checked.
    EXPECT_EQ(Run("SELECT y FROM n", ExecutionMode::PARALLEL).tuples.size(), 3u);
    EXPECT_NO_THROW(db_->GetBufferPoolManager()->DropAllPages());
}

TEST_F(ParallelBasicTest, ErrorLeavesNothingPinned) {
    // a reaches 150000 in the third row group, after two groups ran.
    const std::string sql = "SELECT 10 / (a - 150000) FROM t";
    EXPECT_EQ(ErrorOf([&] { (void)Run(sql, ExecutionMode::VECTORIZED); }), "Division by zero");
    EXPECT_EQ(ErrorOf([&] { (void)Run(sql, ExecutionMode::PARALLEL); }), "Division by zero");

    auto* bpm = db_->GetBufferPoolManager();
    EXPECT_NO_THROW(bpm->DropAllPages());
    const std::string next = "SELECT a FROM t WHERE a < 10";
    EXPECT_EQ(Run(next, ExecutionMode::PARALLEL).tuples.size(), 10u);
    EXPECT_EQ(Run(next, ExecutionMode::TUPLE).tuples.size(), 10u);
}

TEST_F(ParallelBasicTest, AnalyzeLeavesTheProfileEmpty) {
    QueryOptions options;
    options.mode = ExecutionMode::PARALLEL;
    options.analyze = true;
    const QueryResult r = db_->ExecuteSQL("SELECT a FROM t WHERE a < 10", options);
    EXPECT_EQ(r.tuples.size(), 10u);
    EXPECT_TRUE(r.profile.operators.empty());
    EXPECT_TRUE(r.profile.step_rows.empty());
    EXPECT_TRUE(r.profile.scan_rows.empty());
    EXPECT_EQ(r.profile.join.tables, std::vector<std::string>{"t"});
    EXPECT_EQ(r.profile.join.source, JoinOrderSource::SINGLE_TABLE);
}

}  // namespace
}  // namespace shilmandb
