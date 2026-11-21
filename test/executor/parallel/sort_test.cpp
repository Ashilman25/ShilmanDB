#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
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
#include <tuple>
#include <utility>
#include <vector>

#include "catalog/catalog.hpp"
#include "engine/database.hpp"
#include "executor/parallel/batch.hpp"
#include "executor/parallel/expression.hpp"
#include "executor/parallel/pipeline.hpp"
#include "executor/parallel/sort.hpp"
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

}  // namespace
}  // namespace shilmandb::parallel
