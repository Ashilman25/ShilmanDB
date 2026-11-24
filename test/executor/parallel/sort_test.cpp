#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <system_error>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "catalog/catalog.hpp"
#include "engine/database.hpp"
#include "executor/parallel/batch.hpp"
#include "executor/parallel/column_buffer.hpp"
#include "executor/parallel/expression.hpp"
#include "executor/parallel/memory_budget.hpp"
#include "executor/parallel/pipeline.hpp"
#include "executor/parallel/sort.hpp"
#include "executor/parallel/temp_file.hpp"
#include "executor/parallel/worker_pool.hpp"
#include "parser/ast.hpp"
#include "storage/columnar/columnar_table.hpp"

namespace shilmandb::parallel {
namespace {

constexpr uint32_t kUniqueRows = 70000;  // two row groups, so two morsels
constexpr uint32_t kSmallRows = 6000;    // one morsel, three batches
constexpr uint32_t kTieRows = 20000;
constexpr uint32_t kStrRows = 70000;

const char* const kWords[] = {"alpha", "beta", "", "gamma delta", "épée", "Zeta", "zeta"};
const char* const kModes[] = {"AIR", "FOB", "MAIL", "RAIL", "REG AIR", "SHIP", "TRUCK"};

const VType kInt{};
const VType kStr{Kind::STRING, 0, false, TypeId::VARCHAR};

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

// "" when types match and rows match in order, else the first difference.
std::string OrderedDifference(const QueryResult& got, const QueryResult& want) {
    if (got.schema.GetColumnCount() != want.schema.GetColumnCount()) return "column counts differ";
    for (uint32_t c = 0; c < got.schema.GetColumnCount(); ++c) {
        if (got.schema.GetColumn(c).type != want.schema.GetColumn(c).type) return "column " + std::to_string(c) + " has another type";
    }
    if (got.tuples.size() != want.tuples.size()) {
        return std::to_string(got.tuples.size()) + " rows, expected " + std::to_string(want.tuples.size());
    }
    for (size_t i = 0; i < got.tuples.size(); ++i) {
        const Row x = RowOf(got, i);
        const Row y = RowOf(want, i);
        if (!SameRow(x, y)) return "row " + std::to_string(i) + ": got " + Describe(x) + "expected " + Describe(y);
    }
    return "";
}

// "" when types and rows match in order, byte for byte; else the first difference.
// For spilled against unspilled: the sort does no arithmetic, so nothing may differ.
std::string ExactDifference(const QueryResult& got, const QueryResult& want) {
    if (got.schema.GetColumnCount() != want.schema.GetColumnCount()) return "column counts differ";
    for (uint32_t c = 0; c < got.schema.GetColumnCount(); ++c) {
        if (got.schema.GetColumn(c).type != want.schema.GetColumn(c).type) return "column " + std::to_string(c) + " has another type";
    }
    if (got.tuples.size() != want.tuples.size()) {
        return std::to_string(got.tuples.size()) + " rows, expected " + std::to_string(want.tuples.size());
    }
    for (size_t i = 0; i < got.tuples.size(); ++i) {
        const Tuple& x = got.tuples[i];
        const Tuple& y = want.tuples[i];
        if (x.GetLength() != y.GetLength() || std::memcmp(x.GetData(), y.GetData(), x.GetLength()) != 0) {
            return "row " + std::to_string(i) + ": got " + Describe(RowOf(got, i)) + "expected " + Describe(RowOf(want, i));
        }
    }
    return "";
}

// The first rows of a result, as their own result.
QueryResult Prefix(const QueryResult& r, size_t n) {
    QueryResult out{r.schema, {}, {}};
    for (size_t i = 0; i < std::min(n, r.tuples.size()); ++i) out.tuples.push_back(r.tuples[i]);
    return out;
}

// 0..n-1 shuffled by seed.
std::vector<uint32_t> Permutation(uint32_t n, uint32_t seed) {
    std::vector<uint32_t> p(n);
    std::iota(p.begin(), p.end(), 0u);
    std::mt19937 rng(seed);
    std::shuffle(p.begin(), p.end(), rng);
    return p;
}

// Signed cents as text with two decimals.
std::string Money(int64_t cents) {
    const int64_t abs = cents < 0 ? -cents : cents;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s%lld.%02lld", cents < 0 ? "-" : "", static_cast<long long>(abs / 100),
                  static_cast<long long>(abs % 100));
    return buf;
}

// Columns k, g, m, f, d, s; g and d repeat, the rest are unique.
Schema UniqueSchema() {
    return Schema({Column("k", ColumnType::Integer()), Column("g", ColumnType::Integer()),
                   Column("m", ColumnType::Numeric(15, 2)), Column("f", ColumnType::Double()),
                   Column("d", ColumnType::Date()), Column("s", ColumnType::Varchar(24))});
}

// A unique-key table of n rows, every unique column a different shuffle.
std::string UniqueText(uint32_t n) {
    const auto pk = Permutation(n, 1);
    const auto pm = Permutation(n, 2);
    const auto pf = Permutation(n, 3);
    const auto ps = Permutation(n, 4);
    std::string text;
    text.reserve(static_cast<size_t>(n) * 64);
    char buf[256];
    for (uint32_t r = 0; r < n; ++r) {
        const uint32_t k = pk[r];
        const std::string s = std::string(kWords[ps[r] % 7]) + std::to_string(ps[r]);
        std::snprintf(buf, sizeof(buf), "%u|%u|%s|%.2f|%04u-%02u-%02u|%s\n", k, k % 13,
                      Money(static_cast<int64_t>(pm[r]) * 3 - 100000).c_str(), pf[r] / 4.0 - 1000.25, 1990 + k % 30,
                      1 + (k / 30) % 12, 1 + (k / 360) % 28, s.c_str());
        text += buf;
    }
    return text;
}

// One row of the ties table: few distinct values per column.
struct TieRow {
    int32_t k;
    int32_t x;
    std::string s;
    int64_t cents;
};

// Rows with many ties on every column, and whole duplicates.
std::vector<TieRow> TieRows() {
    std::mt19937 rng(7);
    std::vector<TieRow> rows;
    rows.reserve(kTieRows);
    for (uint32_t r = 0; r < kTieRows; ++r) {
        rows.push_back(TieRow{static_cast<int32_t>(rng() % 5), static_cast<int32_t>(rng() % 10), kWords[rng() % 7],
                              static_cast<int64_t>(rng() % 3) - 1});
    }
    return rows;
}

// The ties table's rows in this order, as load text.
std::string TieText(const std::vector<TieRow>& rows) {
    std::string text;
    for (const TieRow& t : rows) text += std::to_string(t.k) + "|" + std::to_string(t.x) + "|" + t.s + "|" + Money(t.cents) + "\n";
    return text;
}

// k, x, s, m.
Schema TieSchema() {
    return Schema({Column("k", ColumnType::Integer()), Column("x", ColumnType::Integer()),
                   Column("s", ColumnType::Varchar(16)), Column("m", ColumnType::Numeric(15, 2))});
}

// Ties-table result rows as (k, x, s, cents), found by column name.
std::vector<TieRow> TieRowsOf(const QueryResult& r) {
    std::vector<TieRow> out;
    for (const Tuple& t : r.tuples) {
        TieRow row{};
        for (uint32_t c = 0; c < r.schema.GetColumnCount(); ++c) {
            const Value v = t.GetValue(r.schema, c);
            const std::string& name = r.schema.GetColumn(c).name;
            if (name == "k") row.k = v.integer_;
            if (name == "x") row.x = v.integer_;
            if (name == "s") row.s = v.varchar_;
            if (name == "m") row.cents = std::llround(v.decimal_ * 100);
        }
        out.push_back(row);
    }
    return out;
}

// Long notes, multibyte notes, empty notes and plain ones; long ones pass 8,000 bytes.
std::string Note(uint32_t r) {
    if (r % 10007 == 3 || r == kStrRows - 2) return std::string(8000 + r % 100, static_cast<char>('a' + r % 26)) + "-" + std::to_string(r);
    if (r % 11 == 0) return "";
    if (r % 7 == 0) return "ñandú-" + std::to_string(r) + " ünï";
    return "note-" + std::to_string(r);
}

// id, a dictionary-coded mode and a plain note.
Schema StrSchema() {
    return Schema({Column("id", ColumnType::Integer()), Column("mode", ColumnType::Varchar(10)), Column("note", ColumnType::Text())});
}

// A batch of an INTEGER slot and a plain STRING slot pointing at strs.
Batch KeyedBatch(const std::vector<int64_t>& keys, const std::vector<std::string>& strs) {
    Batch b;
    ShapeBatch(b, {kInt, kStr});
    b.Reset(static_cast<uint32_t>(keys.size()));
    for (size_t i = 0; i < keys.size(); ++i) {
        b.slots[0].I64()[i] = keys[i];
        b.slots[1].Str()[i] = StringRef{strs[i].data(), static_cast<uint32_t>(strs[i].size())};
    }
    return b;
}

// Sorts by slot 0 and keeps both slots.
SortSpec KeyedSpec(bool ascending, std::optional<int64_t> limit) {
    InputLayout in;
    in.slot_of_column = {0, 1};
    in.slot_types = {kInt, kStr};
    in.slot_precision = {0, 0};
    ColumnRef key;
    key.column_name = "k";
    key.bound_index = 0;
    SortSpec spec;
    spec.keys.push_back(CompiledExpr::Compile(key, in));
    spec.ascending.push_back(ascending);
    spec.columns = {0, 1};
    spec.types = {kInt, kStr};
    spec.limit = limit;
    return spec;
}

using KeyedRow = std::pair<int64_t, std::string>;

// Feeds each worker's batches to its own local, wipes the input strings, reads the result.
std::vector<KeyedRow> RunSort(SortSpec spec, std::vector<std::vector<Batch*>> workers, std::vector<std::string>& strs) {
    auto made = MakeSort(std::move(spec));
    Sink& sink = *made.first;
    Source& source = *made.second;
    QueryContext ctx{nullptr, nullptr};
    for (auto& batches : workers) {
        auto local = sink.MakeLocal();
        for (Batch* b : batches) sink.Consume(*b, *local);
        sink.Combine(std::move(local));
    }
    // The input's strings die with their morsel; the sort must have copied them.
    for (std::string& s : strs) std::fill(s.begin(), s.end(), '#');
    sink.Finalize(ctx);

    source.Prepare(ctx);
    EXPECT_EQ(source.MorselCount(), 1u);
    auto local = source.MakeLocal();
    source.StartMorsel(0, *local);
    std::vector<KeyedRow> rows;
    Batch out;
    while (source.NextBatch(*local, out)) {
        EXPECT_LE(out.Active(), kBatchSize);
        for (uint32_t k = 0; k < out.Active(); ++k) {
            const uint32_t row = out.sel[k];
            rows.emplace_back(out.slots[0].I64()[row], std::string(View(out.slots[1].StrAt(row))));
        }
    }
    return rows;
}

class ParallelSortTest : public ::testing::Test {
protected:
    std::string db_file_;
    std::vector<std::string> tbl_files_;
    std::unique_ptr<Database> db_;

    void SetUp() override {
        db_file_ = TempPath(".db");
        std::filesystem::remove(db_file_);
        db_ = std::make_unique<Database>(db_file_);
    }

    void TearDown() override {
        db_.reset();
        std::filesystem::remove(db_file_);
        for (const auto& f : tbl_files_) std::filesystem::remove(f);
    }

    // A temp path unique to this process.
    static std::string TempPath(const std::string& suffix) {
        return (std::filesystem::temp_directory_path() /
                ("shilmandb_parallel_sort_test_" + std::to_string(::getpid()) + suffix)).string();
    }

    // Loads text into a columnar table on two threads.
    void Load(const std::string& name, const Schema& schema, const std::string& text) {
        const std::string path = TempPath("_" + name + ".tbl");
        std::ofstream(path, std::ios::binary) << text;
        tbl_files_.push_back(path);
        LoadOptions options;
        options.storage = StorageKind::COLUMNAR;
        options.threads = 2;
        db_->LoadTable(name, schema, path, options);
    }

    // Runs sql in the given mode; threads matters only for PARALLEL.
    QueryResult Run(const std::string& sql, ExecutionMode mode, unsigned threads = 1) {
        QueryOptions options;
        options.mode = mode;
        options.threads = threads;
        return db_->ExecuteSQL(sql, options);
    }
};

TEST_F(ParallelSortTest, SortMatchesSortExecutorOnDistinctKeys) {
    // Every key or key list is unique, so today's stable order is the only order.
    const auto check = [&](const std::string& table, uint32_t rows, const std::vector<std::string>& orders) {
        for (const std::string& order : orders) {
            const std::string sql = "SELECT k, g, m, f, d, s FROM " + table + " ORDER BY " + order;
            const QueryResult want = Run(sql, ExecutionMode::VECTORIZED);
            ASSERT_EQ(want.tuples.size(), rows) << sql;
            EXPECT_EQ(OrderedDifference(Run(sql, ExecutionMode::PARALLEL), want), "") << sql;
        }
    };

    // Two morsels: an integer, a string and a two-key order.
    Load("u", UniqueSchema(), UniqueText(kUniqueRows));
    ASSERT_EQ(db_->GetCatalog()->GetTable("u")->columnar->RowGroupCount(), 2u);
    check("u", kUniqueRows, {"k", "s DESC", "g DESC, s"});

    // Every key type both ways, and more key lists, on a smaller table.
    Load("v", UniqueSchema(), UniqueText(kSmallRows));
    check("v", kSmallRows, {"k DESC", "m", "m DESC", "f", "f DESC", "s", "g, k DESC", "d, k", "d DESC, m", "-k"});

    // A key the result doesn't show, and fewer columns kept than the table has.
    const std::string fewer = "SELECT s, k FROM v ORDER BY k + 1";
    EXPECT_EQ(OrderedDifference(Run(fewer, ExecutionMode::PARALLEL), Run(fewer, ExecutionMode::VECTORIZED)), "");
    const std::string filtered = "SELECT m, k FROM u WHERE g = 4 ORDER BY m DESC";
    EXPECT_EQ(OrderedDifference(Run(filtered, ExecutionMode::PARALLEL), Run(filtered, ExecutionMode::VECTORIZED)), "");
}

TEST_F(ParallelSortTest, TiesBrokenByRemainingColumns) {
    const std::vector<TieRow> rows = TieRows();
    std::vector<TieRow> shuffled = rows;
    std::shuffle(shuffled.begin(), shuffled.end(), std::mt19937(11));
    Load("t1", TieSchema(), TieText(rows));
    Load("t2", TieSchema(), TieText(shuffled));

    // Ascending on the key, then on k, x, s, m in that order.
    auto by_k = [](const TieRow& a, const TieRow& b) {
        return std::tie(a.k, a.x, a.s, a.cents) < std::tie(b.k, b.x, b.s, b.cents);
    };
    // Descending on k; ties still ascend on k, x, s, m.
    auto by_k_desc = [](const TieRow& a, const TieRow& b) {
        if (a.k != b.k) return a.k > b.k;
        return std::tie(a.x, a.s, a.cents) < std::tie(b.x, b.s, b.cents);
    };
    // Descending on x; ties ascend on the kept columns s, m, k, x.
    auto by_x_desc_kept_s_first = [](const TieRow& a, const TieRow& b) {
        if (a.x != b.x) return a.x > b.x;
        return std::tie(a.s, a.cents, a.k) < std::tie(b.s, b.cents, b.k);
    };

    struct Case {
        std::string sql;
        std::function<bool(const TieRow&, const TieRow&)> less;
    };
    const std::vector<Case> cases = {
        {"SELECT k, x, s, m FROM %s ORDER BY k", by_k},
        {"SELECT k, x, s, m FROM %s ORDER BY k DESC", by_k_desc},
        {"SELECT s, m, k, x FROM %s ORDER BY x DESC", by_x_desc_kept_s_first},
    };
    for (const Case& c : cases) {
        std::vector<TieRow> want = rows;
        std::sort(want.begin(), want.end(), c.less);
        const auto sql = [&](const char* table) {
            std::string s = c.sql;
            s.replace(s.find("%s"), 2, table);
            return s;
        };

        const QueryResult got = Run(sql("t1"), ExecutionMode::PARALLEL);
        const std::vector<TieRow> seen = TieRowsOf(got);
        ASSERT_EQ(seen.size(), want.size()) << c.sql;
        for (size_t i = 0; i < want.size(); ++i) {
            ASSERT_TRUE(!c.less(seen[i], want[i]) && !c.less(want[i], seen[i]))
                << c.sql << ": row " << i << " is (" << seen[i].k << ", " << seen[i].x << ", " << seen[i].s << ", " << seen[i].cents
                << ")";
        }

        // The same rows loaded in another order, and other thread counts, give the same rows.
        EXPECT_EQ(OrderedDifference(Run(sql("t2"), ExecutionMode::PARALLEL), got), "") << c.sql;
        EXPECT_EQ(OrderedDifference(Run(sql("t1"), ExecutionMode::PARALLEL, 18), got), "") << c.sql;
        EXPECT_EQ(OrderedDifference(Run(sql("t2"), ExecutionMode::PARALLEL, 3), got), "") << c.sql;
    }
}

TEST_F(ParallelSortTest, TopNPastTheEnd) {
    Load("three", Schema({Column("k", ColumnType::Integer()), Column("s", ColumnType::Varchar(4))}), "2|b\n1|a\n3|c\n");

    const auto keys = [&](const std::string& sql) {
        const QueryResult r = Run(sql, ExecutionMode::PARALLEL);
        std::vector<int32_t> out;
        for (const Tuple& t : r.tuples) out.push_back(t.GetValue(r.schema, 0).integer_);
        return out;
    };
    EXPECT_EQ(keys("SELECT k, s FROM three ORDER BY k LIMIT 10"), (std::vector<int32_t>{1, 2, 3}));
    EXPECT_EQ(keys("SELECT k, s FROM three ORDER BY k DESC LIMIT 10"), (std::vector<int32_t>{3, 2, 1}));
    EXPECT_EQ(keys("SELECT k, s FROM three ORDER BY k LIMIT 3"), (std::vector<int32_t>{1, 2, 3}));
    EXPECT_EQ(keys("SELECT k, s FROM three ORDER BY k LIMIT 2"), (std::vector<int32_t>{1, 2}));
    EXPECT_EQ(keys("SELECT k, s FROM three ORDER BY s DESC LIMIT 1"), (std::vector<int32_t>{3}));
    EXPECT_TRUE(keys("SELECT k, s FROM three ORDER BY k LIMIT 0").empty());

    // No rows reach the sort at all.
    EXPECT_TRUE(keys("SELECT k FROM three WHERE k > 5 ORDER BY k LIMIT 10").empty());
    EXPECT_TRUE(keys("SELECT k FROM three WHERE k > 5 ORDER BY k").empty());

    const std::string sql = "SELECT s, k FROM three ORDER BY k DESC LIMIT 10";
    EXPECT_EQ(OrderedDifference(Run(sql, ExecutionMode::PARALLEL), Run(sql, ExecutionMode::VECTORIZED)), "");
}

TEST_F(ParallelSortTest, TopNEqualsSortThenLimit) {
    // With ties, a top n is the full sort's first n rows, tie rule included.
    Load("t1", TieSchema(), TieText(TieRows()));
    const std::string ties = "SELECT k, x, s, m FROM t1 ORDER BY k DESC, s";
    const QueryResult full = Run(ties, ExecutionMode::PARALLEL);
    ASSERT_EQ(full.tuples.size(), kTieRows);
    for (const size_t n : {size_t{0}, size_t{1}, size_t{7}, size_t{2047}, size_t{2048}, size_t{2049}, size_t{4097},
                           size_t{kTieRows - 1}, size_t{kTieRows}, size_t{kTieRows + 5}}) {
        const std::string sql = ties + " LIMIT " + std::to_string(n);
        EXPECT_EQ(OrderedDifference(Run(sql, ExecutionMode::PARALLEL), Prefix(full, n)), "") << sql;
    }

    // Distinct keys across two morsels: also today's sort then limit.
    Load("u", UniqueSchema(), UniqueText(kUniqueRows));
    const std::string unique = "SELECT k, s, m FROM u ORDER BY s DESC";
    const QueryResult all = Run(unique, ExecutionMode::PARALLEL);
    EXPECT_EQ(OrderedDifference(all, Run(unique, ExecutionMode::VECTORIZED)), "");
    for (const size_t n : {size_t{1}, size_t{100}, size_t{2049}, size_t{65537}, size_t{kUniqueRows + 1}}) {
        const std::string sql = unique + " LIMIT " + std::to_string(n);
        EXPECT_EQ(OrderedDifference(Run(sql, ExecutionMode::PARALLEL), Prefix(all, n)), "") << sql;
    }
    const std::string limited = unique + " LIMIT 333";
    EXPECT_EQ(OrderedDifference(Run(limited, ExecutionMode::PARALLEL), Run(limited, ExecutionMode::VECTORIZED)), "");
}

TEST_F(ParallelSortTest, SortStringsOutliveMorsel) {
    std::string text;
    // The trailing delimiter keeps an empty last field.
    for (uint32_t r = 0; r < kStrRows; ++r) text += std::to_string(r) + "|" + kModes[r % 7] + "|" + Note(r) + "|\n";
    Load("str", StrSchema(), text);
    const ColumnarTable& table = *db_->GetCatalog()->GetTable("str")->columnar;
    ASSERT_EQ(table.RowGroupCount(), 2u);
    for (size_t g = 0; g < table.RowGroupCount(); ++g) {
        ASSERT_EQ(table.ChunkAt(g, 1).encoding, Encoding::DICTIONARY) << "group " << g;
        ASSERT_EQ(table.ChunkAt(g, 2).encoding, Encoding::PLAIN) << "group " << g;
    }

    // Strings as keys and kept columns, plain and coded, after their morsel ends.
    const std::vector<std::string> queries = {
        "SELECT id, mode, note FROM str ORDER BY note, id",
        "SELECT id, mode, note FROM str ORDER BY mode DESC, id DESC",
        "SELECT note, id FROM str ORDER BY id DESC",
        "SELECT id, note FROM str ORDER BY id LIMIT 5000",
        "SELECT id, note FROM str ORDER BY id DESC LIMIT 5000",
        "SELECT id, note, mode FROM str ORDER BY note DESC, id LIMIT 100",
    };
    for (const std::string& sql : queries) {
        const QueryResult got = Run(sql, ExecutionMode::PARALLEL);
        EXPECT_EQ(OrderedDifference(got, Run(sql, ExecutionMode::VECTORIZED)), "") << sql;
    }

    // The exact bytes of every long note come back.
    const QueryResult got = Run("SELECT id, note FROM str ORDER BY note DESC, id", ExecutionMode::PARALLEL);
    ASSERT_EQ(got.tuples.size(), kStrRows);
    size_t long_notes = 0;
    for (const Tuple& t : got.tuples) {
        const uint32_t id = static_cast<uint32_t>(t.GetValue(got.schema, 0).integer_);
        const std::string note = t.GetValue(got.schema, 1).varchar_;
        ASSERT_EQ(note, Note(id)) << "id " << id;
        if (note.size() > 8000) ++long_notes;
    }
    EXPECT_EQ(long_notes, 8u);
}

TEST_F(ParallelSortTest, DoubleTiesHaveATotalOrder) {
    // Rows tied on the key and on a kept DOUBLE still get one order:
    // -0.0 before +0.0, NaN last, whatever the input order or worker split.
    const VType kDouble{Kind::DOUBLE, 0, false, TypeId::DECIMAL};
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    using DoubleRow = std::pair<int64_t, double>;
    const std::vector<DoubleRow> rows = {{1, 0.0}, {1, -0.0}, {1, nan}, {1, 2.0}, {1, -inf}, {0, nan}, {1, -0.0}, {1, 0.0}};

    // The output as (key, bits of the double), so -0.0 and +0.0 differ.
    const auto run = [&](std::vector<DoubleRow> input, size_t workers, std::optional<int64_t> limit) {
        InputLayout in;
        in.slot_of_column = {0, 1};
        in.slot_types = {kInt, kDouble};
        in.slot_precision = {0, 0};
        ColumnRef key;
        key.column_name = "k";
        key.bound_index = 0;
        SortSpec spec;
        spec.keys.push_back(CompiledExpr::Compile(key, in));
        spec.ascending.push_back(true);
        spec.columns = {0, 1};
        spec.types = {kInt, kDouble};
        spec.limit = limit;
        auto made = MakeSort(std::move(spec));
        QueryContext ctx{nullptr, nullptr};
        for (size_t w = 0; w < workers; ++w) {
            Batch b;
            ShapeBatch(b, {kInt, kDouble});
            uint32_t n = 0;
            for (size_t i = w; i < input.size(); i += workers) {
                b.slots[0].I64()[n] = input[i].first;
                b.slots[1].F64()[n] = input[i].second;
                ++n;
            }
            b.Reset(n);
            auto local = made.first->MakeLocal();
            made.first->Consume(b, *local);
            made.first->Combine(std::move(local));
        }
        made.first->Finalize(ctx);
        made.second->Prepare(ctx);
        auto local = made.second->MakeLocal();
        made.second->StartMorsel(0, *local);
        std::vector<std::pair<int64_t, uint64_t>> out;
        Batch batch;
        while (made.second->NextBatch(*local, batch)) {
            for (uint32_t k = 0; k < batch.Active(); ++k) {
                uint64_t bits = 0;
                std::memcpy(&bits, &batch.slots[1].F64()[batch.sel[k]], sizeof(bits));
                out.emplace_back(batch.slots[0].I64()[batch.sel[k]], bits);
            }
        }
        return out;
    };
    const auto bits = [](double d) {
        uint64_t b = 0;
        std::memcpy(&b, &d, sizeof(b));
        return b;
    };

    const std::vector<std::pair<int64_t, uint64_t>> want = {{0, bits(nan)},  {1, bits(-inf)}, {1, bits(-0.0)}, {1, bits(-0.0)},
                                                            {1, bits(0.0)},  {1, bits(0.0)},  {1, bits(2.0)},  {1, bits(nan)}};
    std::vector<DoubleRow> reversed(rows.rbegin(), rows.rend());
    for (const size_t workers : {size_t{1}, size_t{2}, size_t{3}}) {
        EXPECT_EQ(run(rows, workers, std::nullopt), want) << workers << " workers";
        EXPECT_EQ(run(reversed, workers, std::nullopt), want) << workers << " workers, reversed";
        const std::vector<std::pair<int64_t, uint64_t>> top(want.begin(), want.begin() + 4);
        EXPECT_EQ(run(reversed, workers, 4), top) << workers << " workers, top 4";
    }
}

TEST_F(ParallelSortTest, NanKeysSortAfterNumbersInOneOrder) {
    // A NaN key comes after every number in ASC (before in DESC),
    // NaNs by their bits.
    // That's one order, so sort and top n agree in any worker split.
    const VType kDouble{Kind::DOUBLE, 0, false, TypeId::DECIMAL};
    const auto from_bits = [](uint64_t b) {
        double d = 0.0;
        std::memcpy(&d, &b, sizeof(d));
        return d;
    };
    const auto bits = [](double d) {
        uint64_t b = 0;
        std::memcpy(&b, &d, sizeof(b));
        return b;
    };
    const double inf = std::numeric_limits<double>::infinity();
    const std::vector<double> pool = {from_bits(0x7FF8000000000000ULL), from_bits(0x7FF8000000000003ULL), from_bits(0xFFF8000000000000ULL),
                                      -inf, inf, -0.0, 0.0, 1.5, -2.25, 1e300, 3.0, 7.0};
    using KeyRow = std::pair<double, int64_t>;  // (key, unique id)
    std::vector<KeyRow> rows;
    std::mt19937 rng(20261118);
    for (int64_t id = 0; id < 3000; ++id) rows.emplace_back(pool[rng() % pool.size()], id);

    // Written without the code under test: numbers by value (zeros tie), then NaNs by bits.
    const auto key_order = [&](double a, double b) {
        if (std::isnan(a) != std::isnan(b)) return std::isnan(a) ? 1 : -1;
        if (std::isnan(a)) return bits(a) < bits(b) ? -1 : (bits(b) < bits(a) ? 1 : 0);
        return a < b ? -1 : (b < a ? 1 : 0);
    };
    // Ties go to the kept columns ascending: the key if kept (-0.0 first), then id.
    // A key that isn't kept shows as 0.
    const auto reference = [&](bool ascending, std::optional<int64_t> limit, bool keep_key) {
        std::vector<KeyRow> want = rows;
        std::sort(want.begin(), want.end(), [&](const KeyRow& a, const KeyRow& b) {
            const int c = key_order(a.first, b.first);
            if (c != 0) return ascending ? c < 0 : c > 0;
            if (keep_key && std::signbit(a.first) != std::signbit(b.first) && a.first == 0.0) return std::signbit(a.first);
            return a.second < b.second;
        });
        if (limit) want.resize(std::min<size_t>(want.size(), static_cast<size_t>(*limit)));
        std::vector<std::pair<uint64_t, int64_t>> out;
        for (const KeyRow& r : want) out.emplace_back(keep_key ? bits(r.first) : 0, r.second);
        return out;
    };

    // Rows shuffled by seed, dealt to workers in batches of up to 700.
    const auto run = [&](bool ascending, std::optional<int64_t> limit, bool keep_key, size_t workers, uint32_t seed) {
        std::vector<KeyRow> input = rows;
        std::mt19937 shuffle(seed);
        for (size_t i = input.size() - 1; i > 0; --i) std::swap(input[i], input[shuffle() % (i + 1)]);
        InputLayout in;
        in.slot_of_column = {0, 1};
        in.slot_types = {kDouble, kInt};
        in.slot_precision = {0, 0};
        ColumnRef key;
        key.column_name = "k";
        key.bound_index = 0;
        SortSpec spec;
        spec.keys.push_back(CompiledExpr::Compile(key, in));
        spec.ascending.push_back(ascending);
        spec.columns = keep_key ? std::vector<uint32_t>{0, 1} : std::vector<uint32_t>{1};
        spec.types = keep_key ? std::vector<VType>{kDouble, kInt} : std::vector<VType>{kInt};
        spec.limit = limit;
        auto made = MakeSort(std::move(spec));
        QueryContext ctx{nullptr, nullptr};
        std::vector<std::unique_ptr<LocalState>> locals;
        for (size_t w = 0; w < workers; ++w) locals.push_back(made.first->MakeLocal());
        size_t turn = 0;
        for (size_t at = 0; at < input.size(); at += 700, ++turn) {
            const auto n = static_cast<uint32_t>(std::min<size_t>(700, input.size() - at));
            Batch b;
            ShapeBatch(b, {kDouble, kInt});
            b.Reset(n);
            for (uint32_t i = 0; i < n; ++i) {
                b.slots[0].F64()[i] = input[at + i].first;
                b.slots[1].I64()[i] = input[at + i].second;
            }
            made.first->Consume(b, *locals[turn % workers]);
        }
        for (auto& local : locals) made.first->Combine(std::move(local));
        made.first->Finalize(ctx);
        made.second->Prepare(ctx);
        auto local = made.second->MakeLocal();
        made.second->StartMorsel(0, *local);
        std::vector<std::pair<uint64_t, int64_t>> out;
        Batch batch;
        while (made.second->NextBatch(*local, batch)) {
            for (uint32_t k = 0; k < batch.Active(); ++k) {
                const uint32_t r = batch.sel[k];
                out.emplace_back(keep_key ? bits(batch.slots[0].F64()[r]) : 0, batch.slots[keep_key ? 1 : 0].I64()[r]);
            }
        }
        return out;
    };

    for (const bool keep_key : {true, false}) {
        for (const bool ascending : {true, false}) {
            for (const std::optional<int64_t> limit : {std::optional<int64_t>{}, std::optional<int64_t>{1}, std::optional<int64_t>{10},
                                                       std::optional<int64_t>{900}, std::optional<int64_t>{2999}}) {
                const auto want = reference(ascending, limit, keep_key);
                for (const size_t workers : {size_t{1}, size_t{2}, size_t{5}}) {
                    for (const uint32_t seed : {1u, 2u, 3u}) {
                        EXPECT_EQ(run(ascending, limit, keep_key, workers, seed), want)
                            << (keep_key ? "key kept, " : "key not kept, ") << (ascending ? "asc" : "desc") << " limit "
                            << (limit ? std::to_string(*limit) : "none") << ", " << workers << " workers, seed " << seed;
                    }
                }
            }
        }
    }
    // The whole ascending order ends in the NaNs.
    const auto all = reference(true, std::nullopt, true);
    EXPECT_TRUE(std::isnan(from_bits(all.back().first)));
    EXPECT_FALSE(std::isnan(from_bits(all.front().first)));
}

TEST_F(ParallelSortTest, WorkersCombineIntoOneOrder) {
    // 5,000 rows, 50 distinct keys, split across workers and batches in different ways.
    constexpr size_t kRows = 5000;
    std::vector<int64_t> keys(kRows);
    std::vector<std::string> strs(kRows);
    for (size_t i = 0; i < kRows; ++i) {
        keys[i] = static_cast<int64_t>((i * 7919) % 50);
        strs[i] = (i % 3 == 0 ? "ü" : "w") + std::to_string((i * 104729) % kRows);
    }

    // Batches cut at these sizes, over a copy of the strings that the run wipes.
    struct Split {
        std::vector<std::string> strs;
        std::vector<Batch> batches;
    };
    const auto cut = [&](const std::vector<size_t>& sizes, bool reverse) {
        Split s;
        s.strs = strs;
        std::vector<size_t> order(kRows);
        std::iota(order.begin(), order.end(), size_t{0});
        if (reverse) std::reverse(order.begin(), order.end());
        size_t at = 0;
        for (size_t b = 0; at < kRows; ++b) {
            const size_t n = std::min(sizes[b % sizes.size()], kRows - at);
            std::vector<int64_t> k;
            std::vector<std::string> v;
            for (size_t i = at; i < at + n; ++i) {
                k.push_back(keys[order[i]]);
                v.push_back(s.strs[order[i]]);
            }
            // Point the batch at the split's own strings, so wiping them shows a missed copy.
            Batch batch = KeyedBatch(k, v);
            for (size_t i = 0; i < n; ++i) {
                const std::string& own = s.strs[order[at + i]];
                batch.slots[1].Str()[i] = StringRef{own.data(), static_cast<uint32_t>(own.size())};
            }
            s.batches.push_back(std::move(batch));
            at += n;
        }
        return s;
    };

    for (const bool ascending : {true, false}) {
        for (const std::optional<int64_t> limit :
             {std::optional<int64_t>{}, std::optional<int64_t>{0}, std::optional<int64_t>{1}, std::optional<int64_t>{10},
              std::optional<int64_t>{2049}, std::optional<int64_t>{4999}, std::optional<int64_t>{6000}}) {
            // Expected: the key's direction, then every kept column ascending.
            std::vector<KeyedRow> want;
            for (size_t i = 0; i < kRows; ++i) want.emplace_back(keys[i], strs[i]);
            std::sort(want.begin(), want.end(), [&](const KeyedRow& a, const KeyedRow& b) {
                if (a.first != b.first) return ascending ? a.first < b.first : a.first > b.first;
                return a.second < b.second;
            });
            if (limit) want.resize(std::min<size_t>(want.size(), static_cast<size_t>(*limit)));
            const std::string label = std::string(ascending ? "asc" : "desc") + " limit " + (limit ? std::to_string(*limit) : "none");

            Split one = cut({kBatchSize}, false);
            std::vector<Batch*> all;
            for (Batch& b : one.batches) all.push_back(&b);
            EXPECT_EQ(RunSort(KeyedSpec(ascending, limit), {all}, one.strs), want) << label << ", one worker";

            Split three = cut({1000, 7, 2048, 333}, true);
            std::vector<std::vector<Batch*>> workers(3);
            for (size_t b = 0; b < three.batches.size(); ++b) workers[b % 3].push_back(&three.batches[b]);
            EXPECT_EQ(RunSort(KeyedSpec(ascending, limit), workers, three.strs), want) << label << ", three workers";

            // A worker that saw nothing changes nothing.
            Split idle = cut({kBatchSize}, true);
            std::vector<Batch*> busy;
            for (Batch& b : idle.batches) busy.push_back(&b);
            EXPECT_EQ(RunSort(KeyedSpec(ascending, limit), {{}, busy, {}}, idle.strs), want) << label << ", idle workers";
        }
    }

    // A batch whose selection skips rows keeps only the active ones.
    std::vector<std::string> few = {"c", "skipped", "a", "skipped", "b"};
    Batch narrowed = KeyedBatch({3, -100, 1, -100, 2}, few);
    narrowed.sel.Data()[0] = 0;
    narrowed.sel.Data()[1] = 2;
    narrowed.sel.Data()[2] = 4;
    narrowed.sel.Resize(3);
    EXPECT_EQ(RunSort(KeyedSpec(true, std::nullopt), {{&narrowed}}, few),
              (std::vector<KeyedRow>{{1, "a"}, {2, "b"}, {3, "c"}}));
}

// ---------------------------------------------------------------------------
// Spilling: sorted runs on disk, merged back, byte for byte the in-memory sort.

const VType kF64{Kind::DOUBLE, 0, false, TypeId::DECIMAL};
const VType kWide{Kind::DECIMAL, 2, true, TypeId::DECIMAL};

constexpr unsigned kSpillThreads = 4;
constexpr size_t kSpillMorsels = 8;  // enough for every worker

// One input row: a value for every array the sort keeps.
struct SpillRow {
    int64_t k{0};
    std::string s;
    double f{0.0};
    __int128 w{0};
    int64_t id{0};
};

using RowMaker = std::function<SpillRow(uint32_t)>;

// The input slots k, s, f, w and id.
const std::vector<VType>& SpillTypes() {
    static const std::vector<VType> types = {kInt, kStr, kF64, kWide, kInt};
    return types;
}

// About 300 bytes a row: 1,000 keys, unique strings, wide decimals past 64 bits.
SpillRow WideRow(uint32_t i) {
    SpillRow r;
    r.k = static_cast<int64_t>((uint64_t{i} * 7919) % 1000);
    r.s = std::string(kWords[i % 7]) + std::to_string((uint64_t{i} * 104729) % 1000003) +
          std::string(240 + i % 64, static_cast<char>('a' + i % 26));
    r.f = static_cast<double>((uint64_t{i} * 31) % 977) / 8.0 - 50.0;
    r.w = static_cast<__int128>(i % 89) * 1000000000000000000LL * 1000 - 7;
    r.id = i;
    return r;
}

// Short rows, so many runs stay small on disk.
SpillRow SmallRow(uint32_t i) {
    SpillRow r;
    r.k = static_cast<int64_t>((uint64_t{i} * 7919) % 5000);
    r.s = kModes[i % 7];
    r.f = (i % 13) * 0.5;
    r.w = i % 1000;
    r.id = i;
    return r;
}

// Every key the same, few values per column, whole rows repeated.
SpillRow TieRowOf(uint32_t i) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const double values[] = {0.0, -0.0, 1.5, nan, -inf, 2.25};
    SpillRow r;
    r.k = 7;
    r.s = kWords[(i * 3) % 7];
    r.f = values[(i / 7) % 6];
    r.w = static_cast<__int128>((i / 3) % 4) - 1;
    r.id = (i * 13) % 997;
    return r;
}

// Empty, multibyte, 8,000-byte and plain strings.
SpillRow NoteRow(uint32_t i) {
    SpillRow r;
    r.k = i % 5;
    if (i % 1009 == 3) {
        r.s = std::string(8000 + i % 100, static_cast<char>('a' + i % 26)) + "-" + std::to_string(i);
    } else if (i % 11 == 0) {
        r.s = "";
    } else if (i % 7 == 0) {
        r.s = "ñandú-" + std::to_string(i % 500) + " ünï";
    } else {
        r.s = "note-" + std::to_string(i % 3000);
    }
    r.f = i % 3;
    r.w = i;
    r.id = i;
    return r;
}

// Every 40th string is past 300,000 bytes, a row bigger than a run block.
SpillRow BigRow(uint32_t i) {
    SpillRow r = NoteRow(i);
    if (i % 40 == 1) r.s = std::string(300000 + i, static_cast<char>('A' + i % 26)) + std::to_string(i);
    return r;
}

// Appends one cell's exact bytes: numbers raw, strings with their length first.
void AppendBytes(const Vector& v, uint32_t row, std::string& out) {
    const auto raw = [&out](const void* p, size_t n) { out.append(static_cast<const char*>(p), n); };
    switch (ArrayOf(v.Type())) {
        case ValueArray::I64: raw(&v.I64()[row], sizeof(int64_t)); break;
        case ValueArray::I128: raw(&v.I128()[row], sizeof(__int128)); break;
        case ValueArray::F64: raw(&v.F64()[row], sizeof(double)); break;
        case ValueArray::STR: {
            const StringRef s = v.StrAt(row);
            raw(&s.size, sizeof(s.size));
            raw(s.data, s.size);
            break;
        }
    }
}

// A row's kept slots in AppendBytes' form, for references built without the sort.
std::string EncodeRow(const SpillRow& r, const std::vector<uint32_t>& kept) {
    std::string out;
    const auto raw = [&out](const void* p, size_t n) { out.append(static_cast<const char*>(p), n); };
    for (const uint32_t slot : kept) {
        switch (slot) {
            case 0: raw(&r.k, sizeof(r.k)); break;
            case 1: {
                const auto size = static_cast<uint32_t>(r.s.size());
                raw(&size, sizeof(size));
                raw(r.s.data(), r.s.size());
                break;
            }
            case 2: raw(&r.f, sizeof(r.f)); break;
            case 3: raw(&r.w, sizeof(r.w)); break;
            default: raw(&r.id, sizeof(r.id)); break;
        }
    }
    return out;
}

// Hands out rows 0..n-1 in morsels, strings plain or dictionary-coded.
// A morsel's strings are wiped when its worker starts another or finishes,
// so a sort that kept a pointer instead of a copy reads '#'.
class SpillSource : public Source {
public:
    struct Local : LocalState {
        std::vector<SpillRow> rows;         // this morsel's rows
        std::vector<std::string> distinct;  // dictionary strings
        std::vector<StringRef> entries;     // dictionary entries over distinct
        std::vector<uint32_t> codes;        // one per row
        size_t next{0};
        size_t batch{0};

        ~Local() override { Wipe(); }

        void Wipe() {
            for (SpillRow& r : rows) std::fill(r.s.begin(), r.s.end(), '#');
            for (std::string& s : distinct) std::fill(s.begin(), s.end(), '#');
        }
    };

    SpillSource(uint32_t rows, size_t morsels, RowMaker make, bool dictionary)
        : rows_(rows), morsels_(morsels), make_(std::move(make)), dictionary_(dictionary) {}

    void Prepare(QueryContext&) override {}
    size_t MorselCount() const override { return morsels_; }
    std::unique_ptr<LocalState> MakeLocal() const override { return std::make_unique<Local>(); }

    void StartMorsel(size_t morsel, LocalState& local) const override {
        auto& l = static_cast<Local&>(local);
        l.Wipe();
        l.rows.clear();
        l.distinct.clear();
        l.entries.clear();
        l.codes.clear();
        const auto begin = static_cast<uint32_t>(uint64_t{rows_} * morsel / morsels_);
        const auto end = static_cast<uint32_t>(uint64_t{rows_} * (morsel + 1) / morsels_);
        for (uint32_t i = begin; i < end; ++i) l.rows.push_back(make_(i));
        if (dictionary_) {
            std::unordered_map<std::string, uint32_t> code_of;
            for (const SpillRow& r : l.rows) {
                const auto found = code_of.find(r.s);
                if (found != code_of.end()) {
                    l.codes.push_back(found->second);
                    continue;
                }
                const auto code = static_cast<uint32_t>(l.distinct.size());
                code_of.emplace(r.s, code);
                l.distinct.push_back(r.s);
                l.codes.push_back(code);
            }
            for (const std::string& s : l.distinct) l.entries.push_back(StringRef{s.data(), static_cast<uint32_t>(s.size())});
        }
        l.next = 0;
        l.batch = 0;
    }

    // Batches of 2,048, 1,000, 7 and 333 rows in turn.
    bool NextBatch(LocalState& local, Batch& out) const override {
        auto& l = static_cast<Local&>(local);
        if (l.next >= l.rows.size()) return false;
        static constexpr uint32_t kSizes[] = {kBatchSize, 1000, 7, 333};
        const auto n = static_cast<uint32_t>(std::min<size_t>(kSizes[l.batch++ % 4], l.rows.size() - l.next));
        ShapeBatch(out, SpillTypes());
        out.Reset(n);
        for (uint32_t i = 0; i < n; ++i) {
            const SpillRow& r = l.rows[l.next + i];
            out.slots[0].I64()[i] = r.k;
            out.slots[2].F64()[i] = r.f;
            out.slots[3].I128()[i] = r.w;
            out.slots[4].I64()[i] = r.id;
            if (!dictionary_) out.slots[1].Str()[i] = StringRef{r.s.data(), static_cast<uint32_t>(r.s.size())};
        }
        if (dictionary_) {
            out.slots[1].SetDictionary(&l.entries);
            for (uint32_t i = 0; i < n; ++i) out.slots[1].Codes()[i] = l.codes[l.next + i];
        }
        l.next += n;
        return true;
    }

private:
    uint32_t rows_;
    size_t morsels_;
    RowMaker make_;
    bool dictionary_;
};

// Keeps every row it's given as exact bytes, in the order they came.
class ByteSink : public Sink {
public:
    struct Local : LocalState {
        std::vector<std::string> rows;
    };

    std::unique_ptr<LocalState> MakeLocal() const override { return std::make_unique<Local>(); }

    void Consume(Batch& in, LocalState& local) const override {
        auto& l = static_cast<Local&>(local);
        for (uint32_t k = 0; k < in.Active(); ++k) {
            std::string row;
            for (const Vector& v : in.slots) AppendBytes(v, in.sel[k], row);
            l.rows.push_back(std::move(row));
        }
    }

    void Combine(std::unique_ptr<LocalState> local) override {
        auto& l = static_cast<Local&>(*local);
        rows.insert(rows.end(), std::make_move_iterator(l.rows.begin()), std::make_move_iterator(l.rows.end()));
    }

    void Finalize(QueryContext&) override {}

    std::vector<std::string> rows;
};

// A sort over the spill input: (slot, ascending) keys, kept slots and a limit.
struct SortCase {
    std::vector<std::pair<uint32_t, bool>> keys;
    std::vector<uint32_t> kept{0, 1, 2, 3, 4};
    std::optional<int64_t> limit;
    std::string label;
};

// The case compiled over the input slots.
SortSpec SpecFor(const SortCase& c) {
    InputLayout in;
    in.slot_of_column = {0, 1, 2, 3, 4};
    in.slot_types = SpillTypes();
    in.slot_precision = {0, 0, 0, 30, 0};
    SortSpec spec;
    for (const auto& [slot, ascending] : c.keys) {
        ColumnRef key;
        key.column_name = "c" + std::to_string(slot);
        key.bound_index = slot;
        spec.keys.push_back(CompiledExpr::Compile(key, in));
        spec.ascending.push_back(ascending);
    }
    for (const uint32_t slot : c.kept) {
        spec.columns.push_back(slot);
        spec.types.push_back(SpillTypes()[slot]);
    }
    spec.limit = c.limit;
    return spec;
}

// How one run of the sort is set up.
struct SpillConfig {
    unsigned threads{1};
    bool spill_everything{false};
    std::optional<size_t> limit;  // the budget's limit; none means no budget
    size_t held{0};               // bytes the test holds on every worker's floor first
    int fail_writes{0};           // errno every temp write fails with; 0 for none
};

// No budget at all.
SpillConfig Unspilled(unsigned threads) {
    SpillConfig c;
    c.threads = threads;
    return c;
}

// A budget of just the floors, with `held` of each floor taken by the test.
SpillConfig FloorsOnly(unsigned threads, size_t held) {
    SpillConfig c;
    c.threads = threads;
    c.limit = threads * kWorkerMinimumBytes;
    c.held = held;
    return c;
}

// Every breaker spills everything it holds; `held` as above.
SpillConfig Everything(unsigned threads, size_t held = 0) {
    SpillConfig c = FloorsOnly(threads, held);
    c.spill_everything = true;
    return c;
}

// The rows a run gave and what it spilled.
struct SpillResult {
    std::vector<std::string> rows;
    uint64_t files{0};
    uint64_t written{0};
    uint64_t read{0};
    std::string error;  // what the run threw, if it did
};

// Descriptors this process has open, counted from /dev/fd.
size_t OpenDescriptors() {
    std::error_code ec;
    size_t n = 0;
    for (std::filesystem::directory_iterator it("/dev/fd", ec), end; !ec && it != end; it.increment(ec)) ++n;
    return n;
}

// Makes every temp write fail with this errno while alive.
class FailWritesScope {
public:
    explicit FailWritesScope(int error_number) { TempFile::FailWritesForTest(error_number); }
    ~FailWritesScope() { TempFile::FailWritesForTest(0); }
    FailWritesScope(const FailWritesScope&) = delete;
    FailWritesScope& operator=(const FailWritesScope&) = delete;
};

// "" when both hold the same rows in the same order, else where they part.
std::string FirstDifference(const std::vector<std::string>& got, const std::vector<std::string>& want) {
    if (got.size() != want.size()) return std::to_string(got.size()) + " rows, expected " + std::to_string(want.size());
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != want[i]) return "row " + std::to_string(i) + " differs";
    }
    return "";
}

// The first n rows.
std::vector<std::string> FirstRows(const std::vector<std::string>& rows, size_t n) {
    return {rows.begin(), rows.begin() + static_cast<std::ptrdiff_t>(std::min(n, rows.size()))};
}

// Caps the merge's fan-in while alive.
class FanInScope {
public:
    explicit FanInScope(size_t most) { SetSortFanInForTest(most); }
    ~FanInScope() { SetSortFanInForTest(0); }
    FanInScope(const FanInScope&) = delete;
    FanInScope& operator=(const FanInScope&) = delete;
};

class ParallelSortSpillTest : public ParallelSortTest {
protected:
    std::string temp_dir_;
    std::unique_ptr<WorkerPool> pool_;

    void SetUp() override {
        ParallelSortTest::SetUp();
        temp_dir_ = TempPath("_spill_dir");
        std::filesystem::remove_all(temp_dir_);
        db_->SetTempDirectory(temp_dir_);
        pool_ = std::make_unique<WorkerPool>(kSpillThreads);
    }

    void TearDown() override {
        pool_.reset();
        ParallelSortTest::TearDown();
        std::filesystem::remove_all(temp_dir_);
    }

    // The temp directory has no entries, or doesn't exist yet.
    bool TempDirEmpty() const {
        std::error_code ec;
        return !std::filesystem::exists(temp_dir_, ec) || std::filesystem::is_empty(temp_dir_, ec);
    }

    // Sorts the source's rows through two pipelines, as a query would; after_sink runs between.
    // Once the sort is gone, no grant, file or descriptor is left.
    SpillResult RunSpill(const SortCase& c, uint32_t rows, size_t morsels, const RowMaker& make, bool dictionary,
                         const SpillConfig& cfg, const std::function<void(MemoryBudget&)>& after_sink = {}) {
        const std::string label = c.label + ", " + std::to_string(cfg.threads) + " threads" +
                                  (cfg.spill_everything ? ", everything" : "") +
                                  (cfg.limit ? ", limit " + std::to_string(*cfg.limit) : "") +
                                  ", held " + std::to_string(cfg.held);
        const size_t descriptors = OpenDescriptors();
        EXPECT_GT(descriptors, 0u) << "can't list /dev/fd";
        // The budget comes first, so everything counted in it goes before it.
        std::optional<MemoryBudget> budget;
        if (cfg.limit) budget.emplace(*cfg.limit, cfg.threads);
        MemoryBudget* b = budget ? &*budget : nullptr;
        std::vector<MemoryReservation> held;
        for (unsigned w = 0; w < cfg.threads && cfg.held > 0; ++w) {
            held.emplace_back(b, w);
            EXPECT_TRUE(held.back().Grow(cfg.held)) << label;
        }
        SpillCounters counters;
        auto made = MakeSort(SpecFor(c));
        SpillSource input(rows, morsels, make, dictionary);
        ByteSink out;
        QueryContext ctx{nullptr, nullptr, cfg.threads, cfg.threads > 1 ? pool_.get() : nullptr};
        ctx.budget = b;
        ctx.spill = &counters;
        ctx.temp_dir = &temp_dir_;
        ctx.spill_everything = cfg.spill_everything;

        Pipeline sort_in{&input, {}, made.first.get(), {}};
        Pipeline sort_out{made.second.get(), {}, &out, {0}};
        std::string error;
        try {
            std::optional<FailWritesScope> fail;
            if (cfg.fail_writes != 0) fail.emplace(cfg.fail_writes);
            RunPipeline(sort_in, ctx);
            if (after_sink && b != nullptr) after_sink(*b);
            RunPipeline(sort_out, ctx);
        } catch (const std::exception& e) {
            error = e.what();
        }

        made.first.reset();
        made.second.reset();
        held.clear();
        EXPECT_EQ(error.empty(), cfg.fail_writes == 0) << label << ": " << error;
        if (!cfg.limit && !cfg.spill_everything) EXPECT_EQ(counters.files.load(), 0u) << label;
        if (b != nullptr) EXPECT_EQ(b->HeldBytes(), 0u) << label;
        EXPECT_TRUE(TempDirEmpty()) << label;
        EXPECT_EQ(OpenDescriptors(), descriptors) << label;
        return SpillResult{std::move(out.rows), counters.files.load(), counters.bytes_written.load(), counters.bytes_read.load(), error};
    }

    // A query with these threads, spilling everything or not.
    QueryResult RunWith(const std::string& sql, unsigned threads, bool spill_everything) {
        QueryOptions options;
        options.mode = ExecutionMode::PARALLEL;
        options.threads = threads;
        options.spill_everything = spill_everything;
        return db_->ExecuteSQL(sql, options);
    }
};

TEST_F(ParallelSortSpillTest, ManyRunsMergeToTheSameOrder) {
    // 100,000 rows of about 300 bytes. The test holds all but one grant of each
    // worker's floor, so each worker sorts in 4 MiB, the least a grant allows.
    constexpr uint32_t kRows = 100000;
    const std::vector<SortCase> cases = {
        {{{0, true}, {1, false}}, {0, 1, 2, 3, 4}, std::nullopt, "k, s DESC"},
        {{{3, false}}, {1, 2, 4}, std::nullopt, "w DESC, key not kept"},
    };
    for (const SortCase& c : cases) {
        const SpillResult want = RunSpill(c, kRows, kSpillMorsels, WideRow, false, Unspilled(1));
        ASSERT_EQ(want.rows.size(), kRows) << c.label;
        EXPECT_EQ(want.files, 0u) << c.label;
        for (const unsigned threads : {1u, kSpillThreads}) {
            EXPECT_EQ(FirstDifference(RunSpill(c, kRows, kSpillMorsels, WideRow, false, Unspilled(threads)).rows, want.rows), "")
                << c.label << ", " << threads << " threads, unspilled";
            const SpillResult got = RunSpill(c, kRows, kSpillMorsels, WideRow, false, FloorsOnly(threads, kWorkerMinimumBytes - kGrantBytes));
            EXPECT_EQ(FirstDifference(got.rows, want.rows), "") << c.label << ", " << threads << " threads";
            EXPECT_GT(got.files, 0u) << c.label << ", " << threads << " threads";
            EXPECT_GT(got.written, 0u) << c.label << ", " << threads << " threads";
            EXPECT_GT(got.read, 0u) << c.label << ", " << threads << " threads";
        }
    }
}

TEST_F(ParallelSortSpillTest, MultiPassMerge) {
    // A run per batch (over 100), more than one merge reads at once.
    constexpr uint32_t kRows = 100000;
    const SortCase c{{{0, false}}, {0, 1, 2, 3, 4}, std::nullopt, "k DESC"};
    const SpillResult want = RunSpill(c, kRows, kSpillMorsels, SmallRow, false, Unspilled(1));
    ASSERT_EQ(want.rows.size(), kRows);
    for (const unsigned threads : {1u, kSpillThreads}) {
        // At most two runs per merge: many passes, each in a new file.
        {
            const FanInScope two(2);
            const SpillResult got = RunSpill(c, kRows, kSpillMorsels, SmallRow, false, Everything(threads));
            EXPECT_EQ(FirstDifference(got.rows, want.rows), "") << threads << " threads, fan-in 2";
            EXPECT_GT(got.files, threads) << threads << " threads, fan-in 2";
            EXPECT_GT(got.written, 0u) << threads << " threads, fan-in 2";
            EXPECT_EQ(got.read, got.written) << threads << " threads, fan-in 2";
        }
        // No cap: one free grant holds fewer blocks than there are runs.
        const SpillResult got = RunSpill(c, kRows, kSpillMorsels, SmallRow, false, Everything(threads, kWorkerMinimumBytes - kGrantBytes));
        EXPECT_EQ(FirstDifference(got.rows, want.rows), "") << threads << " threads, one grant";
        EXPECT_GT(got.files, threads) << threads << " threads, one grant";
        EXPECT_GT(got.written, 0u) << threads << " threads, one grant";
    }
}

TEST_F(ParallelSortSpillTest, AllKeysEqualTieBreaksOnColumns) {
    // Every key is 7, so kept columns decide: k, s, f (total order), w, id.
    constexpr uint32_t kRows = 30000;
    std::vector<SpillRow> rows;
    for (uint32_t i = 0; i < kRows; ++i) rows.push_back(TieRowOf(i));
    std::sort(rows.begin(), rows.end(), [](const SpillRow& a, const SpillRow& b) {
        if (a.s != b.s) return a.s < b.s;
        if (const int c = TotalOrder(a.f, b.f); c != 0) return c < 0;
        if (a.w != b.w) return a.w < b.w;
        return a.id < b.id;
    });
    std::vector<std::string> want;
    for (const SpillRow& r : rows) want.push_back(EncodeRow(r, {0, 1, 2, 3, 4}));

    for (const bool ascending : {true, false}) {
        const SortCase c{{{0, ascending}}, {0, 1, 2, 3, 4}, std::nullopt, ascending ? "k" : "k DESC"};
        for (const unsigned threads : {1u, kSpillThreads}) {
            EXPECT_EQ(FirstDifference(RunSpill(c, kRows, kSpillMorsels, TieRowOf, false, Unspilled(threads)).rows, want), "")
                << c.label << ", " << threads << " threads, unspilled";
            for (const SpillConfig& cfg : {Everything(threads), FloorsOnly(threads, kWorkerMinimumBytes)}) {
                const SpillResult got = RunSpill(c, kRows, kSpillMorsels, TieRowOf, false, cfg);
                EXPECT_EQ(FirstDifference(got.rows, want), "") << c.label << ", " << threads << " threads";
                EXPECT_GT(got.files, 0u) << c.label << ", " << threads << " threads";
                EXPECT_GT(got.written, 0u) << c.label << ", " << threads << " threads";
            }
        }
    }
}

TEST_F(ParallelSortSpillTest, TopNSpillsOnlyWhenNRowsDontFit) {
    // The test holds every floor; the sort gets one shared grant per worker.
    constexpr uint32_t kRows = 100000;
    const SortCase all{{{1, true}}, {1, 4}, std::nullopt, "s"};
    const SpillResult full = RunSpill(all, kRows, kSpillMorsels, WideRow, false, Unspilled(1));
    ASSERT_EQ(full.rows.size(), kRows);
    for (const unsigned threads : {1u, kSpillThreads}) {
        SpillConfig tight;
        tight.threads = threads;
        tight.limit = threads * (kWorkerMinimumBytes + kGrantBytes);
        tight.held = kWorkerMinimumBytes;

        // Ten rows fit: no file.
        SortCase ten = all;
        ten.limit = 10;
        ten.label = "s LIMIT 10";
        const SpillResult few = RunSpill(ten, kRows, kSpillMorsels, WideRow, false, tight);
        EXPECT_EQ(FirstDifference(few.rows, FirstRows(full.rows, 10)), "") << threads << " threads";
        EXPECT_EQ(few.files, 0u) << threads << " threads";

        // 50,000 rows don't: runs, and the merge stops after 50,000.
        SortCase many = all;
        many.limit = 50000;
        many.label = "s LIMIT 50000";
        const SpillResult got = RunSpill(many, kRows, kSpillMorsels, WideRow, false, tight);
        EXPECT_EQ(FirstDifference(got.rows, FirstRows(full.rows, 50000)), "") << threads << " threads";
        EXPECT_GT(got.files, 0u) << threads << " threads";
        EXPECT_GT(got.written, 0u) << threads << " threads";
    }
}

TEST_F(ParallelSortSpillTest, EmptyInputAndLimitZero) {
    constexpr uint32_t kRows = 10000;
    const SortCase c{{{0, true}, {4, false}}, {0, 1, 2, 3, 4}, std::nullopt, "k, id DESC"};
    SortCase zero = c;
    zero.limit = 0;
    zero.label = "LIMIT 0";
    const SpillResult want = RunSpill(c, kRows, kSpillMorsels, WideRow, false, Unspilled(1));
    ASSERT_EQ(want.rows.size(), kRows);

    for (const unsigned threads : {1u, kSpillThreads}) {
        for (const bool everything : {false, true}) {
            SpillConfig free_floors = FloorsOnly(threads, 0);
            SpillConfig held_floors = FloorsOnly(threads, kWorkerMinimumBytes);
            free_floors.spill_everything = everything;
            held_floors.spill_everything = everything;
            const std::string label = std::to_string(threads) + " threads" + (everything ? ", everything" : "");

            // No rows at all, and LIMIT 0: nothing kept, nothing written.
            for (const SpillConfig& cfg : {free_floors, held_floors}) {
                const SpillResult none = RunSpill(c, 0, kSpillMorsels, WideRow, false, cfg);
                EXPECT_TRUE(none.rows.empty()) << label;
                EXPECT_EQ(none.files, 0u) << label;
                const SpillResult limited = RunSpill(zero, kRows, kSpillMorsels, WideRow, false, cfg);
                EXPECT_TRUE(limited.rows.empty()) << label;
                EXPECT_EQ(limited.files, 0u) << label;
            }

            // Floors held from the first row, on one morsel and eight: each batch a run.
            for (const size_t morsels : {size_t{1}, kSpillMorsels}) {
                const SpillResult got = RunSpill(c, kRows, morsels, WideRow, false, held_floors);
                EXPECT_EQ(FirstDifference(got.rows, want.rows), "") << label << ", " << morsels << " morsels";
                EXPECT_GT(got.files, 0u) << label << ", " << morsels << " morsels";
                EXPECT_GT(got.written, 0u) << label << ", " << morsels << " morsels";
            }
        }
    }
}

TEST_F(ParallelSortSpillTest, StringsSurviveTheirMorsel) {
    constexpr uint32_t kRows = 20000;
    std::vector<SpillRow> rows;
    for (uint32_t i = 0; i < kRows; ++i) rows.push_back(NoteRow(i));
    const std::vector<SortCase> cases = {
        {{{1, true}, {4, true}}, {0, 1, 2, 3, 4}, std::nullopt, "s, id"},
        {{{4, false}}, {1, 4}, std::nullopt, "id DESC keeping s"},
    };
    // Written without the sort: by string bytes then id, or by id descending.
    std::vector<std::vector<std::string>> wants;
    for (size_t i = 0; i < cases.size(); ++i) {
        std::vector<SpillRow> sorted = rows;
        std::sort(sorted.begin(), sorted.end(), [&](const SpillRow& a, const SpillRow& b) {
            return i == 0 ? std::tie(a.s, a.id) < std::tie(b.s, b.id) : a.id > b.id;
        });
        std::vector<std::string> want;
        for (const SpillRow& r : sorted) want.push_back(EncodeRow(r, cases[i].kept));
        wants.push_back(std::move(want));
    }

    for (const bool dictionary : {false, true}) {
        for (size_t i = 0; i < cases.size(); ++i) {
            for (const unsigned threads : {1u, kSpillThreads}) {
                const std::string label = cases[i].label + (dictionary ? ", dictionary, " : ", plain, ") + std::to_string(threads) + " threads";
                EXPECT_EQ(FirstDifference(RunSpill(cases[i], kRows, kSpillMorsels, NoteRow, dictionary, Unspilled(threads)).rows, wants[i]), "")
                    << label << ", unspilled";
                for (const SpillConfig& cfg : {Everything(threads), FloorsOnly(threads, kWorkerMinimumBytes)}) {
                    const SpillResult got = RunSpill(cases[i], kRows, kSpillMorsels, NoteRow, dictionary, cfg);
                    EXPECT_EQ(FirstDifference(got.rows, wants[i]), "") << label;
                    EXPECT_GT(got.files, 0u) << label;
                    EXPECT_GT(got.written, 0u) << label;
                }
            }
        }
    }

    // Rows bigger than a run block go in blocks of their own, read back exact.
    constexpr uint32_t kBigRows = 2000;
    const SortCase big{{{1, false}, {4, true}}, {1, 4}, std::nullopt, "big s DESC, id"};
    std::vector<SpillRow> big_rows;
    for (uint32_t i = 0; i < kBigRows; ++i) big_rows.push_back(BigRow(i));
    std::sort(big_rows.begin(), big_rows.end(), [](const SpillRow& a, const SpillRow& b) { return a.s != b.s ? a.s > b.s : a.id < b.id; });
    std::vector<std::string> want_big;
    for (const SpillRow& r : big_rows) want_big.push_back(EncodeRow(r, big.kept));
    for (const unsigned threads : {1u, kSpillThreads}) {
        EXPECT_EQ(FirstDifference(RunSpill(big, kBigRows, kSpillMorsels, BigRow, false, Unspilled(threads)).rows, want_big), "")
            << threads << " threads, unspilled";
        for (const SpillConfig& cfg : {Everything(threads), FloorsOnly(threads, kWorkerMinimumBytes)}) {
            const SpillResult got = RunSpill(big, kBigRows, kSpillMorsels, BigRow, false, cfg);
            EXPECT_EQ(FirstDifference(got.rows, want_big), "") << threads << " threads, big rows";
            EXPECT_GT(got.files, 0u) << threads << " threads, big rows";
            EXPECT_GT(got.written, 0u) << threads << " threads, big rows";
        }
    }

    // A columnar table's dictionary-coded and plain strings, through the whole query.
    std::string text;
    for (uint32_t r = 0; r < kStrRows; ++r) text += std::to_string(r) + "|" + kModes[r % 7] + "|" + Note(r) + "|\n";
    Load("str", StrSchema(), text);
    for (const std::string sql : {"SELECT id, mode, note FROM str ORDER BY note, id", "SELECT id, mode, note FROM str ORDER BY mode DESC, id DESC LIMIT 5000"}) {
        const QueryResult want = RunWith(sql, 1, false);
        EXPECT_EQ(db_->LastParallelStats().spill_files, 0u) << sql << ", unspilled";
        for (const unsigned threads : {1u, 2u}) {
            EXPECT_EQ(ExactDifference(RunWith(sql, threads, true), want), "") << sql << ", " << threads << " threads";
            EXPECT_GT(db_->LastParallelStats().spill_files, 0u) << sql << ", " << threads << " threads";
            EXPECT_GT(db_->LastParallelStats().spill_bytes_written, 0u) << sql << ", " << threads << " threads";
            EXPECT_GT(db_->LastParallelStats().spill_bytes_read, 0u) << sql << ", " << threads << " threads";
        }
    }
    EXPECT_TRUE(TempDirEmpty());
}

TEST_F(ParallelSortSpillTest, FailedWriteLeavesNothingBehind) {
    constexpr uint32_t kRows = 20000;
    const SortCase c{{{0, true}, {1, false}}, {0, 1, 2, 3, 4}, std::nullopt, "k, s DESC"};
    const SpillResult want = RunSpill(c, kRows, kSpillMorsels, WideRow, false, Unspilled(1));
    const std::string prefix = "out of temp space in " + temp_dir_ + " while spilling: ";
    for (const unsigned threads : {1u, kSpillThreads}) {
        for (const SpillConfig& base : {Everything(threads), FloorsOnly(threads, kWorkerMinimumBytes)}) {
            // Every write fails with ENOSPC; the first fails partway through a run.
            SpillConfig failing = base;
            failing.fail_writes = ENOSPC;
            const SpillResult failed = RunSpill(c, kRows, kSpillMorsels, WideRow, false, failing);
            EXPECT_EQ(failed.error, prefix + std::strerror(ENOSPC)) << threads << " threads";
            EXPECT_TRUE(failed.rows.empty()) << threads << " threads";

            // The next sort, on fresh objects with writes working, answers.
            const SpillResult next = RunSpill(c, kRows, kSpillMorsels, WideRow, false, base);
            EXPECT_EQ(FirstDifference(next.rows, want.rows), "") << threads << " threads";
            EXPECT_GT(next.files, 0u) << threads << " threads";
            EXPECT_GT(next.written, 0u) << threads << " threads";
        }
    }
}

TEST_F(ParallelSortSpillTest, RowsStayInMemoryOffTheFloor) {
    // Each worker's rows fit one floor grant; the shared pool has a grant per worker.
    constexpr uint32_t kRows = 4000;
    const SortCase c{{{0, true}}, {0, 1, 2, 3, 4}, std::nullopt, "k"};
    const SpillResult want = RunSpill(c, kRows, kSpillMorsels, WideRow, false, Unspilled(1));
    for (const unsigned threads : {1u, kSpillThreads}) {
        SpillConfig roomy;
        roomy.threads = threads;
        roomy.limit = threads * (kWorkerMinimumBytes + kGrantBytes);
        // Between the pipelines the rows are held, moved off the floors into the shared pool.
        const auto moved = [threads](MemoryBudget& budget) {
            EXPECT_GT(budget.HeldBytes(), 0u) << threads << " threads";
            EXPECT_GT(budget.PeakBytes(), threads * kWorkerMinimumBytes) << threads << " threads";
        };
        const SpillResult got = RunSpill(c, kRows, kSpillMorsels, WideRow, false, roomy, moved);
        EXPECT_EQ(FirstDifference(got.rows, want.rows), "") << threads << " threads";
        EXPECT_EQ(got.files, 0u) << threads << " threads";
    }
}

TEST_F(ParallelSortSpillTest, SpillEverythingMatches) {
    // Every input above, spilling everything, against the unspilled sort.
    struct Input {
        RowMaker make;
        uint32_t rows;
        bool dictionary;
        std::vector<SortCase> cases;
    };
    std::vector<SortCase> wide = {
        {{{0, true}, {1, false}}, {0, 1, 2, 3, 4}, std::nullopt, "k, s DESC"},
        {{{0, false}}, {1, 2, 3, 4}, std::nullopt, "k DESC, key not kept"},
    };
    for (const int64_t n : {0, 1, 10, 2047, 2048, 2049, 15000, 20005}) {
        wide.push_back(SortCase{{{1, true}}, {1, 4}, n, "s LIMIT " + std::to_string(n)});
    }
    const std::vector<Input> inputs = {
        {WideRow, 20000, false, wide},
        {TieRowOf, 20000, false, {{{{0, true}}, {0, 1, 2, 3, 4}, std::nullopt, "ties"}, {{{0, false}}, {0, 1, 2, 3, 4}, 100, "ties LIMIT 100"}}},
        {NoteRow, 20000, false, {{{{1, true}, {4, true}}, {0, 1, 2, 3, 4}, std::nullopt, "notes"}}},
        {NoteRow, 20000, true, {{{{1, false}, {4, true}}, {1, 4}, 3000, "coded notes LIMIT 3000"}}},
        {WideRow, 0, false, {{{{0, true}}, {0, 1, 2, 3, 4}, std::nullopt, "no rows"}}},
    };
    for (const Input& input : inputs) {
        for (const SortCase& c : input.cases) {
            const SpillResult want = RunSpill(c, input.rows, kSpillMorsels, input.make, input.dictionary, Unspilled(1));
            const bool writes = input.rows > 0 && c.limit != std::optional<int64_t>{0};
            for (const unsigned threads : {1u, kSpillThreads}) {
                const SpillResult got = RunSpill(c, input.rows, kSpillMorsels, input.make, input.dictionary, Everything(threads));
                EXPECT_EQ(FirstDifference(got.rows, want.rows), "") << c.label << ", " << threads << " threads";
                if (writes) {
                    EXPECT_GT(got.files, 0u) << c.label << ", " << threads << " threads";
                    EXPECT_GT(got.written, 0u) << c.label << ", " << threads << " threads";
                } else {
                    EXPECT_EQ(got.files, 0u) << c.label << ", " << threads << " threads";
                }
            }
        }
    }

    // Through whole queries, top n included, on two row groups.
    Load("u", UniqueSchema(), UniqueText(kUniqueRows));
    Load("t1", TieSchema(), TieText(TieRows()));
    for (const std::string sql : {"SELECT k, g, m, f, d, s FROM u ORDER BY g DESC, s", "SELECT k, s, m FROM u ORDER BY s DESC LIMIT 333",
                                  "SELECT s, m, k, x FROM t1 ORDER BY x DESC", "SELECT k, x, s, m FROM t1 ORDER BY k DESC, s LIMIT 2049"}) {
        const QueryResult want = RunWith(sql, 1, false);
        EXPECT_EQ(db_->LastParallelStats().spill_files, 0u) << sql << ", unspilled";
        for (const unsigned threads : {1u, kSpillThreads}) {
            EXPECT_EQ(ExactDifference(RunWith(sql, threads, true), want), "") << sql << ", " << threads << " threads";
            EXPECT_GT(db_->LastParallelStats().spill_files, 0u) << sql << ", " << threads << " threads";
            EXPECT_GT(db_->LastParallelStats().spill_bytes_written, 0u) << sql << ", " << threads << " threads";
        }
    }
    EXPECT_TRUE(TempDirEmpty());
}

}  // namespace
}  // namespace shilmandb::parallel
