#include <gtest/gtest.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "catalog/catalog.hpp"
#include "engine/database.hpp"
#include "executor/parallel/batch.hpp"
#include "executor/parallel/column_buffer.hpp"
#include "executor/parallel/compiler.hpp"
#include "executor/parallel/decimal.hpp"
#include "executor/parallel/expression.hpp"
#include "executor/parallel/hash_aggregate.hpp"
#include "executor/parallel/key_hash.hpp"
#include "executor/parallel/memory_budget.hpp"
#include "executor/parallel/parallel_executor.hpp"
#include "executor/parallel/pipeline.hpp"
#include "executor/parallel/spill_partitions.hpp"
#include "executor/parallel/temp_file.hpp"
#include "executor/parallel/worker_pool.hpp"
#include "parser/ast.hpp"
#include "parser/parser.hpp"
#include "planner/plan_dump.hpp"
#include "planner/planner.hpp"
#include "storage/columnar/columnar_table.hpp"
#include "tpch/tpch_workload.hpp"

namespace shilmandb::parallel {
namespace {

using Func = Aggregate::Func;

constexpr uint32_t kManyRows = 1000000;
constexpr uint32_t kManyKeys = 300000;
constexpr uint32_t kStrRows = 70000;  // two row groups, so two morsels

const char* const kWords[] = {"alpha", "beta", "", "gamma delta", "épée", "Zeta", "zeta"};
const char* const kModes[] = {"AIR", "FOB", "MAIL", "RAIL", "REG AIR", "SHIP", "TRUCK"};

const VType kInt{};
const VType kBig{Kind::INTEGER, 0, false, TypeId::BIGINT};
const VType kStr{Kind::STRING, 0, false, TypeId::VARCHAR};
const VType kMoney{Kind::DECIMAL, 2, false, TypeId::DECIMAL};
const VType kWideMoney{Kind::DECIMAL, 2, true, TypeId::DECIMAL};
const VType kDouble{Kind::DOUBLE, 0, false, TypeId::DECIMAL};
const VType kDate{Kind::DATE, 0, false, TypeId::DATE};

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

// a and b within SameResult's 1e-6 relative.
bool Near(double a, double b) {
    return std::abs(a - b) <= 1e-6 * std::max({1.0, std::abs(a), std::abs(b)});
}

// Exact text cells, decimals within 1e-6 relative.
bool SameRow(const Row& x, const Row& y) {
    if (x.text != y.text || x.decimals.size() != y.decimals.size()) return false;
    for (size_t d = 0; d < x.decimals.size(); ++d) {
        if (!Near(x.decimals[d], y.decimals[d])) return false;
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

// "" when the two results have the same column types, else the first difference.
// counts marks want's COUNT columns, which got must have as BIGINT where want has INTEGER.
std::string TypeDifference(const QueryResult& got, const QueryResult& want, const std::vector<bool>& counts = {}) {
    if (got.schema.GetColumnCount() != want.schema.GetColumnCount()) return "column counts differ";
    for (uint32_t c = 0; c < got.schema.GetColumnCount(); ++c) {
        const TypeId today = want.schema.GetColumn(c).type;
        const bool count = c < counts.size() && counts[c];
        if (count && today != TypeId::INTEGER) return "column " + std::to_string(c) + " is a COUNT but not INTEGER today";
        if (got.schema.GetColumn(c).type != (count ? TypeId::BIGINT : today)) return "column " + std::to_string(c) + " has another type";
    }
    if (got.tuples.size() != want.tuples.size()) {
        return std::to_string(got.tuples.size()) + " rows, expected " + std::to_string(want.tuples.size());
    }
    return "";
}

// How got and want first differ as multisets of rows, or "" when they match.
std::string Difference(const QueryResult& got, const QueryResult& want, const std::vector<bool>& counts = {}) {
    const std::string types = TypeDifference(got, want, counts);
    if (!types.empty()) return types;
    std::vector<Row> x;
    std::vector<Row> y;
    for (size_t i = 0; i < got.tuples.size(); ++i) x.push_back(RowOf(got, i));
    for (size_t i = 0; i < want.tuples.size(); ++i) y.push_back(RowOf(want, i));
    std::sort(x.begin(), x.end());
    std::sort(y.begin(), y.end());
    for (size_t i = 0; i < x.size(); ++i) {
        if (!SameRow(x[i], y[i])) return "row " + std::to_string(i) + ": got " + Describe(x[i]) + "expected " + Describe(y[i]);
    }
    return "";
}

// The same, but rows must match in order.
std::string OrderedDifference(const QueryResult& got, const QueryResult& want, const std::vector<bool>& counts = {}) {
    const std::string types = TypeDifference(got, want, counts);
    if (!types.empty()) return types;
    for (size_t i = 0; i < got.tuples.size(); ++i) {
        const Row x = RowOf(got, i);
        const Row y = RowOf(want, i);
        if (!SameRow(x, y)) return "row " + std::to_string(i) + ": got " + Describe(x) + "expected " + Describe(y);
    }
    return "";
}

// Signed cents as text with two decimals.
std::string Money(int64_t cents) {
    const int64_t abs = cents < 0 ? -cents : cents;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s%lld.%02lld", cents < 0 ? "-" : "", static_cast<long long>(abs / 100),
                  static_cast<long long>(abs % 100));
    return buf;
}

// Text like "-123.4" or "17" as an exact integer at `scale` decimals.
__int128 Scaled(const std::string& text, int scale) {
    size_t i = 0;
    const bool negative = !text.empty() && text[0] == '-';
    if (negative) i = 1;
    __int128 v = 0;
    int decimals = -1;
    for (; i < text.size(); ++i) {
        if (text[i] == '.') {
            decimals = 0;
            continue;
        }
        v = v * 10 + (text[i] - '0');
        if (decimals >= 0) ++decimals;
    }
    for (decimals = std::max(decimals, 0); decimals < scale; ++decimals) v *= 10;
    return negative ? -v : v;
}

// A line's fields split at '|'.
std::vector<std::string> Fields(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string f;
    while (std::getline(ss, f, '|')) out.push_back(f);
    return out;
}

// An exact integer as text, for failure messages.
std::string Text(__int128 v) {
    return DecimalToString(v, 0);
}

// A column reference bound to plan column c.
std::unique_ptr<CompiledExpr> Col(uint32_t c, const InputLayout& in) {
    ColumnRef ref;
    ref.column_name = "c" + std::to_string(c);
    ref.bound_index = c;
    return CompiledExpr::Compile(ref, in);
}

//  Hand-fed sinks

// Hand-fed batches: slot 0 k INTEGER, 1 s VARCHAR, 2 m NUMERIC(15,2), 3 f DOUBLE.
InputLayout HandLayout() {
    InputLayout in;
    in.slot_of_column = {0, 1, 2, 3};
    in.slot_types = {kInt, kStr, kMoney, kDouble};
    in.slot_precision = {0, 0, 15, 0};
    return in;
}

// One hand-fed row.
struct HandRow {
    int64_t k;
    std::string s;
    int64_t cents;
    double f;
};

// The given keys, then COUNT(*), SUM(m), MIN(s), MAX(s), AVG(m), MIN(m), MAX(f).
AggregateSpec HandSpec(const std::vector<uint32_t>& keys) {
    const InputLayout in = HandLayout();
    AggregateSpec spec;
    for (uint32_t k : keys) {
        spec.keys.push_back(Col(k, in));
        spec.out_types.push_back(in.slot_types[k]);
    }
    const std::vector<std::pair<Func, int>> aggs = {{Func::COUNT, -1}, {Func::SUM, 2}, {Func::MIN, 1}, {Func::MAX, 1},
                                                    {Func::AVG, 2},    {Func::MIN, 2}, {Func::MAX, 3}};
    for (const auto& a : aggs) {
        spec.funcs.push_back(a.first);
        spec.args.push_back(a.second < 0 ? nullptr : Col(static_cast<uint32_t>(a.second), in));
        spec.out_types.push_back(AggregateOutputType(a.first, spec.args.back().get()));
    }
    spec.global = keys.empty();
    return spec;
}

// One group as the hand spec emits it; missing keys read 0 and "".
struct HandGroup {
    int64_t k{0};
    std::string s;
    int64_t count{0};
    __int128 sum{0};
    std::string min_s;
    std::string max_s;
    double avg{0.0};
    int64_t min_m{0};
    double max_f{0.0};
};

// Feeds each worker's rows to its own local in batches, then reads the groups.
// After each Consume the batch's string bytes are overwritten, as the scan's next morsel would.
std::vector<HandGroup> RunHand(const std::vector<uint32_t>& keys, const std::vector<std::vector<HandRow>>& workers, uint32_t batch,
                               bool dictionary, unsigned threads = 1, size_t* morsels = nullptr) {
    auto made = MakeHashAggregate(HandSpec(keys));
    Sink& sink = *made.first;
    Source& source = *made.second;
    QueryContext ctx{nullptr, nullptr};
    ctx.threads = threads;

    for (const auto& rows : workers) {
        auto local = sink.MakeLocal();
        Batch b;
        ShapeBatch(b, HandLayout().slot_types);
        for (size_t at = 0; at < rows.size(); at += batch) {
            const auto n = static_cast<uint32_t>(std::min<size_t>(batch, rows.size() - at));
            std::vector<std::string> bytes;  // the batch's strings, wiped after Consume
            std::vector<StringRef> entries;
            b.Reset(n);
            for (uint32_t i = 0; i < n; ++i) bytes.push_back(rows[at + i].s);
            for (uint32_t i = 0; i < n; ++i) {
                const HandRow& r = rows[at + i];
                b.slots[0].I64()[i] = r.k;
                b.slots[2].I64()[i] = r.cents;
                b.slots[3].F64()[i] = r.f;
                const StringRef ref{bytes[i].data(), static_cast<uint32_t>(bytes[i].size())};
                if (dictionary) {
                    b.slots[1].Codes()[i] = static_cast<uint32_t>(entries.size());
                    entries.push_back(ref);
                } else {
                    b.slots[1].Str()[i] = ref;
                }
            }
            if (dictionary) b.slots[1].SetDictionary(&entries);
            // Every third row is inactive, so the selection matters.
            if (n > 2) {
                uint32_t kept = 0;
                for (uint32_t i = 0; i < n; ++i) {
                    if (i % 3 != 1) b.sel.Data()[kept++] = static_cast<uint16_t>(i);
                }
                b.sel.Resize(kept);
            }
            sink.Consume(b, *local);
            for (std::string& s : bytes) std::fill(s.begin(), s.end(), '#');
            b.Reset(0);
        }
        sink.Combine(std::move(local));
    }
    sink.Finalize(ctx);

    source.Prepare(ctx);
    if (morsels != nullptr) *morsels = source.MorselCount();
    std::vector<HandGroup> out;
    auto local = source.MakeLocal();
    Batch o;
    const size_t a = keys.size();  // the first aggregate's slot
    for (size_t m = 0; m < source.MorselCount(); ++m) {
        source.StartMorsel(m, *local);
        while (source.NextBatch(*local, o)) {
            EXPECT_EQ(o.slots.size(), a + 7);
            for (uint32_t i = 0; i < o.Active(); ++i) {
                const uint32_t r = o.sel[i];
                HandGroup g;
                for (size_t k = 0; k < keys.size(); ++k) {
                    if (keys[k] == 0) g.k = o.slots[k].I64()[r];
                    if (keys[k] == 1) g.s = std::string(View(o.slots[k].StrAt(r)));
                }
                g.count = o.slots[a].I64()[r];
                g.sum = o.slots[a + 1].I128()[r];
                g.min_s = std::string(View(o.slots[a + 2].StrAt(r)));
                g.max_s = std::string(View(o.slots[a + 3].StrAt(r)));
                g.avg = o.slots[a + 4].F64()[r];
                g.min_m = o.slots[a + 5].I64()[r];
                g.max_f = o.slots[a + 6].F64()[r];
                out.push_back(g);
            }
        }
    }
    return out;
}

using HandKey = std::pair<int64_t, std::string>;

// The rows RunHand keeps: all but every third row of each batch over 2 rows.
std::vector<HandRow> KeptRows(const std::vector<HandRow>& rows, uint32_t batch) {
    std::vector<HandRow> kept;
    for (size_t at = 0; at < rows.size(); at += batch) {
        const size_t n = std::min<size_t>(batch, rows.size() - at);
        for (size_t i = 0; i < n; ++i) {
            if (n <= 2 || i % 3 != 1) kept.push_back(rows[at + i]);
        }
    }
    return kept;
}

// What the hand spec over these key columns should give, from the kept rows.
// A column left out of the key reads 0 or "", as RunHand reports it.
std::map<HandKey, HandGroup> HandReference(const std::vector<std::vector<HandRow>>& workers, uint32_t batch,
                                           const std::vector<uint32_t>& keys = {0, 1}) {
    const bool by_k = std::find(keys.begin(), keys.end(), 0u) != keys.end();
    const bool by_s = std::find(keys.begin(), keys.end(), 1u) != keys.end();
    std::map<HandKey, HandGroup> ref;
    for (const auto& rows : workers) {
        for (const HandRow& r : KeptRows(rows, batch)) {
            const HandKey key{by_k ? r.k : 0, by_s ? r.s : std::string()};
            auto it = ref.find(key);
            if (it == ref.end()) {
                HandGroup g;
                g.k = key.first;
                g.s = key.second;
                g.min_s = r.s;
                g.max_s = r.s;
                g.min_m = r.cents;
                g.max_f = r.f;
                it = ref.emplace(key, g).first;
            }
            HandGroup& g = it->second;
            ++g.count;
            g.sum += r.cents;
            g.min_s = std::min(g.min_s, r.s);
            g.max_s = std::max(g.max_s, r.s);
            g.min_m = std::min(g.min_m, r.cents);
            g.max_f = std::max(g.max_f, r.f);
        }
    }
    for (auto& entry : ref) entry.second.avg = DecimalToDouble(entry.second.sum, 2) / static_cast<double>(entry.second.count);
    return ref;
}

// "" when out holds exactly the reference's groups, each once, else the first difference.
std::string HandDifference(const std::vector<HandGroup>& out, const std::map<HandKey, HandGroup>& ref) {
    std::set<HandKey> seen;
    for (const HandGroup& g : out) {
        const HandKey key{g.k, g.s};
        if (!seen.insert(key).second) return "group (" + std::to_string(g.k) + ", " + g.s + ") came out twice";
        const auto it = ref.find(key);
        if (it == ref.end()) return "unexpected group (" + std::to_string(g.k) + ", " + g.s + ")";
        const HandGroup& w = it->second;
        const std::string at = " in group (" + std::to_string(g.k) + ", " + g.s + ")";
        if (g.count != w.count) return "count " + std::to_string(g.count) + " != " + std::to_string(w.count) + at;
        if (g.sum != w.sum) return "sum " + Text(g.sum) + " != " + Text(w.sum) + at;
        if (g.min_s != w.min_s || g.max_s != w.max_s) return "min/max string '" + g.min_s + "', '" + g.max_s + "'" + at;
        if (g.avg != w.avg) return "avg " + std::to_string(g.avg) + " != " + std::to_string(w.avg) + at;
        if (g.min_m != w.min_m) return "min(m) " + std::to_string(g.min_m) + at;
        if (g.max_f != w.max_f) return "max(f) " + std::to_string(g.max_f) + at;
    }
    if (seen.size() != ref.size()) return std::to_string(seen.size()) + " groups, expected " + std::to_string(ref.size());
    return "";
}

// Rows with keys drawn from `keys` distinct (k, s) pairs, strings of varied length.
std::vector<HandRow> HandRows(size_t n, uint32_t keys, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<HandRow> rows;
    rows.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t key = rng() % keys;
        HandRow r;
        r.k = static_cast<int64_t>(key % 1009) - 500;
        r.s = std::string(kWords[key % 7]) + std::to_string(key / 1009);
        if (key % 97 == 5) r.s += std::string(300, 'x');  // a few long strings
        r.cents = static_cast<int64_t>(rng() % 2000001) - 1000000;
        r.f = static_cast<double>(static_cast<int32_t>(rng() % 20001) - 10000) / 4.0;
        rows.push_back(std::move(r));
    }
    return rows;
}

// How many distinct (k, s) keys each worker's kept rows hold, summed.
size_t SummedGroups(const std::vector<std::vector<HandRow>>& workers, uint32_t batch) {
    size_t total = 0;
    for (const auto& rows : workers) {
        std::set<HandKey> keys;
        for (const HandRow& r : KeptRows(rows, batch)) keys.insert({r.k, r.s});
        total += keys.size();
    }
    return total;
}

class ParallelHashAggregateTest : public ::testing::Test {
protected:
    std::string db_file_;
    std::vector<std::string> extra_files_;
    std::unique_ptr<Database> db_;

    void SetUp() override {
        db_file_ = TempPath(".db");
        std::filesystem::remove(db_file_);
        db_ = std::make_unique<Database>(db_file_);
    }

    void TearDown() override {
        db_.reset();
        std::filesystem::remove(db_file_);
        for (const auto& f : extra_files_) std::filesystem::remove(f);
    }

    // A temp path unique to this process.
    static std::string TempPath(const std::string& suffix) {
        return (std::filesystem::temp_directory_path() /
                ("shilmandb_parallel_hash_aggregate_test_" + std::to_string(::getpid()) + suffix)).string();
    }

    // Loads text into a columnar table on two threads.
    void Load(const std::string& name, const Schema& schema, const std::string& text) {
        const std::string path = TempPath("_" + name + ".tbl");
        std::ofstream(path, std::ios::binary) << text;
        extra_files_.push_back(path);
        LoadOptions options;
        options.storage = StorageKind::COLUMNAR;
        options.threads = 2;
        db_->LoadTable(name, schema, path, options);
    }

    // Runs sql in the given mode; threads matters only for PARALLEL.
    QueryResult Run(const std::string& sql, ExecutionMode mode, unsigned threads = 1) {
        return RunOn(*db_, sql, mode, threads);
    }

    // Runs sql on db in one mode.
    static QueryResult RunOn(Database& db, const std::string& sql, ExecutionMode mode, unsigned threads = 1) {
        QueryOptions options;
        options.mode = mode;
        options.threads = threads;
        return db.ExecuteSQL(sql, options);
    }

    // sql's COUNT columns in today's plan on db, which parallel plans type BIGINT.
    static std::vector<bool> CountsOn(const Database& db, const std::string& sql) {
        return test::CountColumns(*db.MakePlanner(QueryOptions{}).Plan(std::move(*Parser(sql).Parse())));
    }

    // The same on this test's database.
    std::vector<bool> Counts(const std::string& sql) const { return CountsOn(*db_, sql); }

    // The SF=0.01 TPC-H data, or "" when it hasn't been generated.
    static std::string FindTpchDir() {
        for (const std::string dir : {"bench/tpch_data/sf0.01/", "../bench/tpch_data/sf0.01/", "../../bench/tpch_data/sf0.01/"}) {
            if (std::filesystem::exists(dir + "lineitem.tbl")) return dir;
        }
        return "";
    }

    // SF=0.01 lineitem alone, stored by column, in its own file.
    std::unique_ptr<Database> LoadLineitem(const std::string& dir) {
        const std::string file = TempPath("_tpch.db");
        std::filesystem::remove(file);
        extra_files_.push_back(file);
        auto tpch = std::make_unique<Database>(file, 4096);
        LoadOptions options;
        options.storage = StorageKind::COLUMNAR;
        for (const auto& td : tpch::TableDescs()) {
            if (td.name == "lineitem") tpch->LoadTable(td.name, td.schema, dir + td.tbl_file, options);
        }
        return tpch;
    }

    // Every line of lineitem.tbl, split into fields.
    static std::vector<std::vector<std::string>> LineitemRows(const std::string& dir) {
        std::vector<std::vector<std::string>> rows;
        std::ifstream in(dir + "lineitem.tbl");
        std::string line;
        while (std::getline(in, line)) rows.push_back(Fields(line));
        return rows;
    }

    // The empty-input table: 100 rows, x from 0 to 99.
    void LoadSmall() {
        std::string text;
        for (int x = 0; x < 100; ++x) {
            text += std::to_string(x) + "|" + std::to_string(int64_t{x} * 10000000007LL) + "|" + Money(x * 131 - 5000) + "|" +
                    std::to_string(x / 4.0) + "|1995-01-" + (x % 28 < 9 ? "0" : "") + std::to_string(x % 28 + 1) + "|" +
                    kWords[x % 7] + "|\n";
        }
        Load("e", Schema({Column("x", ColumnType::Integer()), Column("b", ColumnType::BigInt()), Column("m", ColumnType::Numeric(15, 2)),
                          Column("f", ColumnType::Double()), Column("d", ColumnType::Date()), Column("s", ColumnType::Varchar(16))}),
             text);
    }
};

//  Output types

TEST_F(ParallelHashAggregateTest, OutputTypes) {
    InputLayout in;
    in.slot_of_column = {0, 1, 2, 3, 4, 5, 6};
    in.slot_types = {kInt, kBig, kMoney, kDouble, kDate, kStr, VType{Kind::DECIMAL, 3, true, TypeId::DECIMAL}};
    in.slot_precision = {0, 0, 15, 0, 0, 0, 25};
    const auto type = [&](Func f, int c) {
        const auto arg = c < 0 ? nullptr : Col(static_cast<uint32_t>(c), in);
        return AggregateOutputType(f, arg.get());
    };
    EXPECT_EQ(type(Func::COUNT, -1), kBig);
    EXPECT_EQ(type(Func::COUNT, 5), kBig);
    EXPECT_EQ(type(Func::SUM, 0), (VType{Kind::DECIMAL, 0, true, TypeId::DECIMAL}));
    EXPECT_EQ(type(Func::SUM, 1), (VType{Kind::DECIMAL, 0, true, TypeId::DECIMAL}));
    EXPECT_EQ(type(Func::SUM, 2), kWideMoney);
    EXPECT_EQ(type(Func::SUM, 6), (VType{Kind::DECIMAL, 3, true, TypeId::DECIMAL}));
    EXPECT_EQ(type(Func::SUM, 3), kDouble);
    for (int c = 0; c < 7; ++c) EXPECT_EQ(type(Func::AVG, c), kDouble) << c;
    // Non-numeric SUM / AVG still type as DECIMAL; they throw only when a row arrives.
    EXPECT_EQ(type(Func::SUM, 4), kDouble);
    EXPECT_EQ(type(Func::SUM, 5), kDouble);
    for (int c = 0; c < 7; ++c) {
        EXPECT_EQ(type(Func::MIN, c), in.slot_types[c]) << c;
        EXPECT_EQ(type(Func::MAX, c), in.slot_types[c]) << c;
    }
    EXPECT_NE(ErrorOf([&] { (void)type(Func::SUM, -1); }), "");

    EXPECT_EQ(MergePartitionCount(0, 1), 1u);
    EXPECT_EQ(MergePartitionCount(kMergeIntoOneMaxGroups, 18), 1u);
    EXPECT_GT(MergePartitionCount(kMergeIntoOneMaxGroups + 1, 1), 1u);
}

//  TPC-H shapes

TEST_F(ParallelHashAggregateTest, Q6ExactSum) {
    const std::string dir = FindTpchDir();
    if (dir.empty()) GTEST_SKIP() << "TPC-H SF=0.01 data not found. Run bench/tpch/generate_tpch_data.sh first.";
    const auto tpch = LoadLineitem(dir);

    // The scaled products, summed exactly, straight from the file.
    __int128 sum = 0;
    size_t matched = 0;
    for (const auto& f : LineitemRows(dir)) {
        const __int128 quantity = Scaled(f[4], 2);
        const __int128 price = Scaled(f[5], 2);
        const __int128 discount = Scaled(f[6], 2);
        const std::string& shipdate = f[10];
        if (shipdate >= "1994-01-01" && shipdate < "1995-01-01" && discount >= 5 && discount <= 7 && quantity < 2400) {
            sum += price * discount;
            ++matched;
        }
    }
    ASSERT_GT(matched, 0u);
    const double want = DecimalToDouble(sum, 4);

    const std::string q6 = tpch::FindQuery("Q6")->sql;
    const QueryResult today = RunOn(*tpch, q6, ExecutionMode::VECTORIZED);
    ASSERT_EQ(today.tuples.size(), 1u);
    for (const unsigned threads : {1u, 3u, 18u}) {
        const QueryResult got = RunOn(*tpch, q6, ExecutionMode::PARALLEL, threads);
        ASSERT_EQ(got.tuples.size(), 1u);
        ASSERT_EQ(got.schema.GetColumn(0).type, TypeId::DECIMAL);
        const double value = got.tuples[0].GetValue(got.schema, 0).decimal_;
        EXPECT_EQ(value, want) << "exact sum " << Text(sum) << " at scale 4";
        EXPECT_TRUE(Near(value, today.tuples[0].GetValue(today.schema, 0).decimal_));
    }
}

TEST_F(ParallelHashAggregateTest, Q1ShapeExactSums) {
    const std::string dir = FindTpchDir();
    if (dir.empty()) GTEST_SKIP() << "TPC-H SF=0.01 data not found. Run bench/tpch/generate_tpch_data.sh first.";
    const auto tpch = LoadLineitem(dir);

    // Exact sums per (returnflag, linestatus) from the file.
    struct Sums {
        __int128 quantity{0};
        __int128 price{0};
        __int128 discount{0};
        int64_t count{0};
    };
    std::map<std::pair<std::string, std::string>, Sums> ref;
    for (const auto& f : LineitemRows(dir)) {
        if (f[10] > "1998-09-02") continue;
        Sums& s = ref[{f[8], f[9]}];
        s.quantity += Scaled(f[4], 2);
        s.price += Scaled(f[5], 2);
        s.discount += Scaled(f[6], 2);
        ++s.count;
    }
    ASSERT_EQ(ref.size(), 4u);

    const std::string q1 = tpch::FindQuery("Q1")->sql;
    const QueryResult got = RunOn(*tpch, q1, ExecutionMode::PARALLEL);
    ASSERT_EQ(got.tuples.size(), 4u);
    size_t i = 0;
    for (const auto& entry : ref) {
        const Tuple& t = got.tuples[i++];
        EXPECT_EQ(t.GetValue(got.schema, 0).varchar_, entry.first.first);
        EXPECT_EQ(t.GetValue(got.schema, 1).varchar_, entry.first.second);
        EXPECT_EQ(t.GetValue(got.schema, 2).decimal_, DecimalToDouble(entry.second.quantity, 2));
        EXPECT_EQ(t.GetValue(got.schema, 3).decimal_, DecimalToDouble(entry.second.price, 2)) << Text(entry.second.price);
        EXPECT_EQ(t.GetValue(got.schema, 4).decimal_, DecimalToDouble(entry.second.discount, 2));
        EXPECT_EQ(t.GetValue(got.schema, 5).type_, TypeId::BIGINT);
        EXPECT_EQ(t.GetValue(got.schema, 5).bigint_, entry.second.count);
    }
    // count_star is BIGINT here and INTEGER today, with the same digits.
    const QueryResult today = RunOn(*tpch, q1, ExecutionMode::VECTORIZED);
    ASSERT_EQ(got.schema.GetColumn(5).type, TypeId::BIGINT);
    ASSERT_EQ(today.schema.GetColumn(5).type, TypeId::INTEGER);
    EXPECT_TRUE(tpch::SameResult(got, today));
    EXPECT_EQ(OrderedDifference(got, today, CountsOn(*tpch, q1)), "");
    EXPECT_EQ(OrderedDifference(RunOn(*tpch, q1, ExecutionMode::PARALLEL, 18), got), "");
}

//  Many groups

TEST_F(ParallelHashAggregateTest, ManyGroups) {
    // Every key in 0..299,999 appears; row r has key (r * 104729) mod 300,000.
    std::string text;
    text.reserve(static_cast<size_t>(kManyRows) * 28);
    std::map<int64_t, std::pair<int64_t, __int128>> ref;
    std::vector<int64_t> cents(kManyRows);
    for (uint32_t r = 0; r < kManyRows; ++r) {
        const int64_t k = (static_cast<int64_t>(r) * 104729) % kManyKeys;
        cents[r] = (static_cast<int64_t>(r) * 7919) % 2000001 - 1000000;
        text += std::to_string(k);
        text += '|';
        text += Money(cents[r]);
        text += "|key";
        text += std::to_string(k);
        text += '\n';
    }
    for (uint32_t r = 0; r < kManyRows; ++r) {
        auto& entry = ref[(static_cast<int64_t>(r) * 104729) % kManyKeys];
        ++entry.first;
        entry.second += cents[r];
    }
    ASSERT_EQ(ref.size(), kManyKeys);
    Load("big", Schema({Column("k", ColumnType::Integer()), Column("v", ColumnType::Numeric(15, 2)), Column("s", ColumnType::Varchar(16))}),
         text);

    // By an integer key, then by a string key: every group once, exact.
    for (const std::string& key : {std::string("k"), std::string("s")}) {
        const QueryResult got = Run("SELECT " + key + ", COUNT(*), SUM(v) FROM big GROUP BY " + key, ExecutionMode::PARALLEL);
        ASSERT_EQ(got.tuples.size(), kManyKeys) << key;
        std::set<int64_t> seen;
        for (const Tuple& t : got.tuples) {
            const Value kv = t.GetValue(got.schema, 0);
            const int64_t k = key == "k" ? kv.integer_ : std::stoll(kv.varchar_.substr(3));
            ASSERT_TRUE(seen.insert(k).second) << "key " << k << " came out twice";
            const auto& want = ref.at(k);
            ASSERT_EQ(t.GetValue(got.schema, 1).bigint_, want.first) << "key " << k;
            ASSERT_EQ(t.GetValue(got.schema, 2).decimal_, DecimalToDouble(want.second, 2)) << "key " << k;
        }
    }
}

//  Empty input

TEST_F(ParallelHashAggregateTest, EmptyInputGlobalRow) {
    LoadSmall();
    const std::string none = " FROM e WHERE x < -1000000";

    // The row of zero-defaults today's executors emit, through any thread count.
    const std::string first = "SELECT COUNT(*), SUM(x), MIN(x)" + none;
    const QueryResult want = Run(first, ExecutionMode::VECTORIZED);
    ASSERT_EQ(want.tuples.size(), 1u);
    for (const unsigned threads : {1u, 3u, 18u}) {
        const QueryResult got = Run(first, ExecutionMode::PARALLEL, threads);
        ASSERT_EQ(got.tuples.size(), 1u);
        EXPECT_EQ(got.schema.GetColumn(0).type, TypeId::BIGINT);
        EXPECT_EQ(got.schema.GetColumn(1).type, TypeId::DECIMAL);
        EXPECT_EQ(got.schema.GetColumn(2).type, TypeId::INTEGER);
        EXPECT_EQ(got.tuples[0].GetValue(got.schema, 0).type_, TypeId::BIGINT);
        EXPECT_EQ(got.tuples[0].GetValue(got.schema, 0).bigint_, 0);
        const double sum = got.tuples[0].GetValue(got.schema, 1).decimal_;
        EXPECT_EQ(sum, 0.0);
        EXPECT_FALSE(std::signbit(sum));
        EXPECT_EQ(got.tuples[0].GetValue(got.schema, 2).integer_, 0);
        EXPECT_EQ(Difference(got, want, Counts(first)), "");
    }

    // Types whose empty MIN / MAX today reads back well defined, SUM and AVG of
    // every type (non-numeric ones don't throw without rows), and expressions above.
    for (const std::string& select : {"SELECT MIN(d), MAX(d), MIN(s), MAX(s), MIN(x)", "SELECT AVG(m), SUM(f), AVG(x), SUM(b), AVG(f)",
                                      "SELECT SUM(d), AVG(s), SUM(s), AVG(d)", "SELECT COUNT(*) + 1, SUM(m) * 2, MAX(x) - 3"}) {
        const std::string sql = select + none;
        const QueryResult got = Run(sql, ExecutionMode::PARALLEL);
        ASSERT_EQ(got.tuples.size(), 1u) << sql;
        EXPECT_EQ(Difference(got, Run(sql, ExecutionMode::VECTORIZED), Counts(sql)), "") << sql;
    }

    // BIGINT and DECIMAL get their type's zero (today's are indeterminate).
    const QueryResult zeros = Run("SELECT MIN(m), MAX(b), MIN(f), MAX(m), MIN(b)" + none, ExecutionMode::PARALLEL);
    ASSERT_EQ(zeros.tuples.size(), 1u);
    const std::vector<TypeId> types = {TypeId::DECIMAL, TypeId::BIGINT, TypeId::DECIMAL, TypeId::DECIMAL, TypeId::BIGINT};
    for (uint32_t c = 0; c < types.size(); ++c) {
        ASSERT_EQ(zeros.schema.GetColumn(c).type, types[c]) << c;
        const Value v = zeros.tuples[0].GetValue(zeros.schema, c);
        EXPECT_TRUE(v == Value::DefaultFor(types[c])) << c << ": " << v.ToString();
        if (types[c] == TypeId::BIGINT) EXPECT_EQ(v.bigint_, 0);
    }

    // With rows, a non-numeric SUM or AVG throws today's error.
    for (const std::string& sql : {"SELECT SUM(d) FROM e", "SELECT AVG(s) FROM e", "SELECT x, SUM(s) FROM e GROUP BY x"}) {
        const std::string today = ErrorOf([&] { (void)Run(sql, ExecutionMode::VECTORIZED); });
        EXPECT_EQ(today, "Cannot aggregate non-numeric type") << sql;
        EXPECT_EQ(ErrorOf([&] { (void)Run(sql, ExecutionMode::PARALLEL); }), today) << sql;
    }

    // Hand-fed: no locals at all, and workers that saw nothing, still give one row.
    for (size_t workers : {size_t{0}, size_t{1}, size_t{3}}) {
        const std::vector<std::vector<HandRow>> empty(workers);
        size_t morsels = 0;
        const std::vector<HandGroup> out = RunHand({}, empty, 100, false, 1, &morsels);
        ASSERT_EQ(out.size(), 1u) << workers;
        EXPECT_EQ(morsels, 1u);
        EXPECT_EQ(out[0].count, 0);
        EXPECT_EQ(out[0].sum, 0);
        EXPECT_EQ(out[0].min_s, "");
        EXPECT_EQ(out[0].max_s, "");
        EXPECT_EQ(out[0].avg, 0.0);
        EXPECT_EQ(out[0].min_m, 0);
        EXPECT_EQ(out[0].max_f, 0.0);
    }
}

TEST_F(ParallelHashAggregateTest, EmptyInputWithGroupByGivesNoRows) {
    LoadSmall();
    const std::vector<std::string> queries = {
        "SELECT s, COUNT(*), SUM(m), MIN(d) FROM e WHERE x < -1000000 GROUP BY s",
        "SELECT s, COUNT(*) FROM e WHERE x < -1000000 GROUP BY s ORDER BY s",
        "SELECT x, COUNT(*) FROM e WHERE x < -1000000 GROUP BY x HAVING x > 5",
        "SELECT x, SUM(d), AVG(s) FROM e WHERE x < -1000000 GROUP BY x",
        "SELECT x, d, MAX(m) FROM e WHERE x < -1000000 GROUP BY x, d LIMIT 5",
    };
    for (const std::string& sql : queries) {
        const QueryResult want = Run(sql, ExecutionMode::VECTORIZED);
        EXPECT_TRUE(want.tuples.empty()) << sql;
        const QueryResult got = Run(sql, ExecutionMode::PARALLEL);
        EXPECT_TRUE(got.tuples.empty()) << sql;
        EXPECT_EQ(Difference(got, want, Counts(sql)), "") << sql;
    }

    // Hand-fed: no locals, and locals that saw nothing, emit no groups and no morsels.
    for (size_t workers : {size_t{0}, size_t{1}, size_t{3}}) {
        size_t morsels = 7;
        EXPECT_TRUE(RunHand({0, 1}, std::vector<std::vector<HandRow>>(workers), 100, false, 1, &morsels).empty());
        EXPECT_EQ(morsels, 0u);
    }
}

//  Exact sums

TEST_F(ParallelHashAggregateTest, SumPastInt64) {
    // 100 values near INT64_MAX / 10; group 1's sum is group 2's plus 1.
    constexpr int64_t kBase = std::numeric_limits<int64_t>::max() / 10;
    std::string text;
    __int128 total = 0;
    __int128 sums[3] = {0, 0, 0};
    for (int i = 0; i < 100; ++i) {
        const int g = i < 50 ? 1 : 2;
        const int64_t v = kBase - (i % 50) - (i == 99 ? 1 : 0);
        text += std::to_string(g) + "|" + std::to_string(v) + "\n";
        total += v;
        sums[g] += v;
    }
    ASSERT_GT(total, static_cast<__int128>(std::numeric_limits<int64_t>::max()));
    ASSERT_EQ(sums[1], sums[2] + 1);
    // As doubles the two sums tie, so only an exact sum orders them.
    ASSERT_EQ(DecimalToDouble(sums[1], 0), DecimalToDouble(sums[2], 0));
    Load("w", Schema({Column("g", ColumnType::Integer()), Column("v", ColumnType::Numeric(18, 0))}), text);

    const QueryResult got = Run("SELECT SUM(v) FROM w", ExecutionMode::PARALLEL);
    ASSERT_EQ(got.tuples.size(), 1u);
    EXPECT_EQ(got.tuples[0].GetValue(got.schema, 0).decimal_, DecimalToDouble(total, 0)) << Text(total);
    const QueryResult today = Run("SELECT SUM(v) FROM w", ExecutionMode::VECTORIZED);
    EXPECT_TRUE(Near(got.tuples[0].GetValue(got.schema, 0).decimal_, today.tuples[0].GetValue(today.schema, 0).decimal_));

    // The sort above sees exact sums: group 2's smaller sum comes first.
    // As doubles the two would tie, and the tie rule would put group 1 first.
    const QueryResult ordered = Run("SELECT g, SUM(v) FROM w GROUP BY g ORDER BY SUM(v)", ExecutionMode::PARALLEL);
    ASSERT_EQ(ordered.tuples.size(), 2u);
    EXPECT_EQ(ordered.tuples[0].GetValue(ordered.schema, 0).integer_, 2);
    EXPECT_EQ(ordered.tuples[1].GetValue(ordered.schema, 0).integer_, 1);
    const QueryResult desc = Run("SELECT g, SUM(v) FROM w GROUP BY g ORDER BY SUM(v) DESC", ExecutionMode::PARALLEL);
    ASSERT_EQ(desc.tuples.size(), 2u);
    EXPECT_EQ(desc.tuples[0].GetValue(desc.schema, 0).integer_, 1);

    // Past 38 digits the sum refuses, globally and per group.
    const std::string big = "60000000000000000000000000000000000000";
    Load("z", Schema({Column("zg", ColumnType::Integer()), Column("zv", ColumnType::Numeric(38, 0))}), "1|" + big + "\n1|" + big + "\n");
    EXPECT_EQ(ErrorOf([&] { (void)Run("SELECT SUM(zv) FROM z", ExecutionMode::PARALLEL); }), "NUMERIC result is too wide");
    EXPECT_EQ(ErrorOf([&] { (void)Run("SELECT zg, SUM(zv) FROM z GROUP BY zg", ExecutionMode::PARALLEL); }), "NUMERIC result is too wide");
    EXPECT_EQ(Run("SELECT MAX(zv), COUNT(*) FROM z", ExecutionMode::PARALLEL).tuples.size(), 1u);
}

//  AVG, MIN and MAX

TEST_F(ParallelHashAggregateTest, AvgAndMinMaxMatchTupleMode) {
    std::mt19937 rng(5);
    std::string text;
    for (uint32_t r = 0; r < 10000; ++r) {
        const int64_t i = static_cast<int64_t>(rng() % 200001) - 100000;
        const int64_t b = (static_cast<int64_t>(rng() % 2000001) - 1000000) * 1000003;
        const int64_t m = static_cast<int64_t>(rng() % 2000000001) - 1000000000;
        const double f = static_cast<double>(static_cast<int64_t>(rng() % 200001) - 100000) / 8.0;
        const std::string s = std::string(kWords[rng() % 7]) + std::to_string(rng() % 50);
        const uint64_t whole = static_cast<uint64_t>(rng() % 1000000000) * 1000000000ULL + rng() % 1000000000;
        char w[64];
        std::snprintf(w, sizeof(w), "%s%llu.%03u", rng() % 2 ? "-" : "", static_cast<unsigned long long>(whole), static_cast<unsigned>(rng() % 1000));
        char line[256];
        std::snprintf(line, sizeof(line), "%u|%lld|%lld|%s|%.3f|%04u-%02u-%02u|%s|%s|\n", r % 37, static_cast<long long>(i),
                      static_cast<long long>(b), Money(m).c_str(), f, 1990 + r % 30, 1 + (r / 30) % 12, 1 + (r / 360) % 28, s.c_str(), w);
        text += line;
    }
    Load("a", Schema({Column("g", ColumnType::Integer()), Column("i", ColumnType::Integer()), Column("b", ColumnType::BigInt()),
                      Column("m", ColumnType::Numeric(15, 2)), Column("f", ColumnType::Double()), Column("d", ColumnType::Date()),
                      Column("s", ColumnType::Varchar(16)), Column("w", ColumnType::Numeric(25, 3))}),
         text);

    const std::vector<std::string> queries = {
        "SELECT g, AVG(i), AVG(b), AVG(m), AVG(f), AVG(w) FROM a GROUP BY g",
        "SELECT g, MIN(i), MAX(i), MIN(b), MAX(b), MIN(m), MAX(m), MIN(f), MAX(f), MIN(d), MAX(d), MIN(s), MAX(s), MIN(w), MAX(w) "
        "FROM a GROUP BY g",
        "SELECT AVG(i), MIN(s), MAX(d), MIN(m), MAX(f), COUNT(*), SUM(i), SUM(b), SUM(m), SUM(f), SUM(w), AVG(w) FROM a",
        "SELECT s, d, COUNT(*), SUM(m), MIN(i) FROM a GROUP BY s, d",
        "SELECT m, COUNT(*) FROM a GROUP BY m",
        "SELECT f, COUNT(*), SUM(i) FROM a GROUP BY f",
        "SELECT b, w, COUNT(*) FROM a GROUP BY b, w",
        "SELECT g, COUNT(*), SUM(m) FROM a GROUP BY g HAVING g > 5 AND g < 30",
        "SELECT g, SUM(m) * 2, COUNT(*) + 1, MAX(i) - MIN(i), AVG(i) * 2 FROM a GROUP BY g",
        "SELECT g, AVG(m) FROM a WHERE i > 0 GROUP BY g",
        "SELECT 100.00 * SUM(CASE WHEN i > 0 THEN m ELSE 0 END) / SUM(m) FROM a",
        "SELECT g, SUM(m * (1 - m)), SUM(i * b) FROM a GROUP BY g",
        "SELECT g, SUM(f * m), AVG(f + m) FROM a GROUP BY g",
        "SELECT SUM(f * m), AVG(f + m), MIN(f * m), MAX(f + m) FROM a",
        "SELECT g, s, MAX(f) FROM a WHERE m < 0 GROUP BY g, s HAVING s > 'beta'",
        "SELECT d, MIN(s), COUNT(*) FROM a GROUP BY d HAVING d >= '2000-01-01'",
    };
    for (const std::string& sql : queries) {
        const QueryResult want = Run(sql, ExecutionMode::TUPLE);
        ASSERT_FALSE(want.tuples.empty()) << sql;
        EXPECT_EQ(Difference(Run(sql, ExecutionMode::PARALLEL), want, Counts(sql)), "") << sql;
        EXPECT_EQ(Difference(Run(sql, ExecutionMode::PARALLEL, 18), want, Counts(sql)), "") << sql;
    }

    // A sort above the groups, with and without a top n.
    for (const std::string& sql : {"SELECT g, SUM(m), COUNT(*) FROM a GROUP BY g ORDER BY SUM(m) DESC LIMIT 5",
                                   "SELECT s, MIN(d) FROM a GROUP BY s ORDER BY s"}) {
        EXPECT_EQ(OrderedDifference(Run(sql, ExecutionMode::PARALLEL), Run(sql, ExecutionMode::TUPLE), Counts(sql)), "") << sql;
    }

    // A LIMIT with no order: the right count, and every row one today's result holds.
    const std::string limited = "SELECT g, COUNT(*), MAX(s) FROM a GROUP BY g LIMIT 10";
    const QueryResult got = Run(limited, ExecutionMode::PARALLEL, 1);
    ASSERT_EQ(got.tuples.size(), 10u);
    const QueryResult all = Run("SELECT g, COUNT(*), MAX(s) FROM a GROUP BY g", ExecutionMode::TUPLE);
    std::set<std::vector<std::string>> today;
    for (size_t i = 0; i < all.tuples.size(); ++i) today.insert(RowOf(all, i).text);
    for (size_t i = 0; i < got.tuples.size(); ++i) EXPECT_EQ(today.count(RowOf(got, i).text), 1u) << Describe(RowOf(got, i));
}

//  Strings

TEST_F(ParallelHashAggregateTest, StringKeysOutliveMorsel) {
    // Long notes, multibyte notes, empty notes and plain ones; long ones pass 8,000 bytes.
    const auto note = [](uint32_t r) -> std::string {
        if (r % 10007 == 3 || r == kStrRows - 2) return std::string(8000 + r % 100, static_cast<char>('a' + r % 26)) + "-" + std::to_string(r);
        if (r % 11 == 0) return "";
        if (r % 7 == 0) return "ñandú-" + std::to_string(r) + " ünï";
        return "note-" + std::to_string(r % 20000);
    };
    std::string text;
    for (uint32_t r = 0; r < kStrRows; ++r) text += std::to_string(r) + "|" + kModes[r % 7] + "|" + note(r) + "|\n";
    Load("str", Schema({Column("id", ColumnType::Integer()), Column("mode", ColumnType::Varchar(10)), Column("note", ColumnType::Text())}), text);
    const ColumnarTable& table = *db_->GetCatalog()->GetTable("str")->columnar;
    ASSERT_EQ(table.RowGroupCount(), 2u);
    for (size_t g = 0; g < table.RowGroupCount(); ++g) ASSERT_EQ(table.ChunkAt(g, 1).encoding, Encoding::DICTIONARY) << "group " << g;

    // Keys and MIN / MAX strings, plain and coded, read after both morsels are gone.
    const std::vector<std::string> queries = {
        "SELECT note, COUNT(*), MIN(id), MAX(mode) FROM str GROUP BY note",
        "SELECT mode, MIN(note), MAX(note), COUNT(*) FROM str GROUP BY mode",
        "SELECT mode, note, COUNT(*) FROM str GROUP BY mode, note",
        "SELECT MIN(note), MAX(note), MIN(mode), MAX(mode) FROM str",
    };
    for (const std::string& sql : queries) {
        EXPECT_EQ(Difference(Run(sql, ExecutionMode::PARALLEL), Run(sql, ExecutionMode::VECTORIZED), Counts(sql)), "") << sql;
    }

    // Every key's bytes come back exactly, long ones included.
    const QueryResult got = Run("SELECT note, MIN(id), COUNT(*) FROM str GROUP BY note", ExecutionMode::PARALLEL);
    size_t long_notes = 0;
    for (const Tuple& t : got.tuples) {
        const std::string key = t.GetValue(got.schema, 0).varchar_;
        ASSERT_EQ(key, note(static_cast<uint32_t>(t.GetValue(got.schema, 1).integer_)));
        if (key.size() > 8000) ++long_notes;
    }
    EXPECT_EQ(long_notes, 8u);

    // Hand-fed: every batch's bytes are wiped right after Consume, plain and coded.
    const std::vector<std::vector<HandRow>> workers = {HandRows(5000, 700, 1), HandRows(3000, 700, 2)};
    const auto ref = HandReference(workers, 1000);
    const auto by_s_ref = HandReference({workers[0]}, 1000, {1});
    for (const bool dictionary : {false, true}) {
        EXPECT_EQ(HandDifference(RunHand({0, 1}, workers, 1000, dictionary), ref), "") << dictionary;
        EXPECT_EQ(HandDifference(RunHand({1}, {workers[0]}, 1000, dictionary), by_s_ref), "") << dictionary;
    }
}

//  Merging worker tables

TEST_F(ParallelHashAggregateTest, MergeIntoOneBelowThreshold) {
    const std::vector<std::vector<HandRow>> workers = {HandRows(50000, 40000, 11), HandRows(50000, 40000, 12), HandRows(30000, 40000, 13)};
    const size_t summed = SummedGroups(workers, kBatchSize);
    ASSERT_LE(summed, kMergeIntoOneMaxGroups);
    ASSERT_EQ(MergePartitionCount(summed, 1), 1u);
    const auto ref = HandReference(workers, kBatchSize);

    size_t morsels = 0;
    const std::vector<HandGroup> out = RunHand({0, 1}, workers, kBatchSize, false, 1, &morsels);
    EXPECT_EQ(HandDifference(out, ref), "");
    EXPECT_GT(morsels, 1u);

    // The same rows split another way, in another order, give the same groups.
    const std::vector<std::vector<HandRow>> reversed = {workers[2], workers[1], workers[0]};
    EXPECT_EQ(HandDifference(RunHand({0, 1}, reversed, kBatchSize, true, 4), ref), "");
}

TEST_F(ParallelHashAggregateTest, MergeByPartitionAboveThreshold) {
    const std::vector<std::vector<HandRow>> workers = {HandRows(100000, 120000, 21), HandRows(100000, 120000, 22),
                                                       HandRows(60000, 120000, 23), {}};
    const size_t summed = SummedGroups(workers, kBatchSize);
    ASSERT_GT(summed, kMergeIntoOneMaxGroups);
    ASSERT_GT(MergePartitionCount(summed, 1), 1u);
    const auto ref = HandReference(workers, kBatchSize);

    size_t morsels = 0;
    EXPECT_EQ(HandDifference(RunHand({0, 1}, workers, kBatchSize, false, 1, &morsels), ref), "");
    EXPECT_GT(morsels, 1u);
    const std::vector<std::vector<HandRow>> reversed = {workers[3], workers[2], workers[1], workers[0]};
    EXPECT_EQ(HandDifference(RunHand({0, 1}, reversed, kBatchSize, true, 16), ref), "");
}

TEST_F(ParallelHashAggregateTest, GlobalMergesWorkerStates) {
    const std::vector<std::vector<HandRow>> workers = {HandRows(5000, 50, 31), {}, HandRows(7, 50, 32), HandRows(4000, 50, 33)};
    const auto ref = HandReference(workers, 1000, {});
    ASSERT_EQ(ref.size(), 1u);
    const std::vector<HandGroup> out = RunHand({}, workers, 1000, false);
    EXPECT_EQ(HandDifference(out, ref), "");
}

TEST_F(ParallelHashAggregateTest, DoubleKeysGroupAsValueDoes) {
    // -0.0 and +0.0 are one group, as under Value's <; NaNs are one group too.
    const double nan = std::numeric_limits<double>::quiet_NaN();
    InputLayout in = HandLayout();
    const auto spec = [&] {
        AggregateSpec s;
        s.keys.push_back(Col(3, in));
        s.out_types.push_back(kDouble);
        s.funcs.push_back(Func::COUNT);
        s.args.push_back(nullptr);
        s.out_types.push_back(kBig);
        return s;
    };
    const std::vector<std::vector<double>> workers = {{0.0, -0.0, 1.5, nan, 2.0}, {-0.0, nan, 1.5}, {0.0}};
    auto made = MakeHashAggregate(spec());
    QueryContext ctx{nullptr, nullptr};
    for (const auto& keys : workers) {
        auto local = made.first->MakeLocal();
        Batch b;
        ShapeBatch(b, in.slot_types);
        b.Reset(static_cast<uint32_t>(keys.size()));
        for (size_t i = 0; i < keys.size(); ++i) b.slots[3].F64()[i] = keys[i];
        made.first->Consume(b, *local);
        made.first->Combine(std::move(local));
    }
    made.first->Finalize(ctx);
    made.second->Prepare(ctx);
    auto local = made.second->MakeLocal();
    std::map<std::string, int64_t> counts;
    Batch o;
    for (size_t m = 0; m < made.second->MorselCount(); ++m) {
        made.second->StartMorsel(m, *local);
        while (made.second->NextBatch(*local, o)) {
            for (uint32_t i = 0; i < o.Active(); ++i) {
                const double k = o.slots[0].F64()[o.sel[i]];
                const std::string name = std::isnan(k) ? "nan" : std::to_string(k == 0.0 ? 0.0 : k);
                EXPECT_EQ(counts.count(name), 0u) << name << " came out twice";
                counts[name] = o.slots[1].I64()[o.sel[i]];
            }
        }
    }
    const std::map<std::string, int64_t> want = {{std::to_string(0.0), 4}, {std::to_string(1.5), 2}, {std::to_string(2.0), 1}, {"nan", 2}};
    EXPECT_EQ(counts, want);
}

TEST_F(ParallelHashAggregateTest, CountPastInt32MaxOverWorkers) {
    // Just past 2^31 rows over two workers, neither past it alone: COUNT keeps
    // all 64 bits through the merge and the emit.
    const InputLayout in = HandLayout();
    AggregateSpec spec;
    spec.funcs = {Func::COUNT, Func::COUNT};
    spec.args.push_back(nullptr);
    spec.args.push_back(Col(0, in));  // COUNT(k) reads no argument either
    spec.out_types = {kBig, kBig};
    spec.global = true;
    auto made = MakeHashAggregate(std::move(spec));
    QueryContext ctx{nullptr, nullptr};

    Batch b;
    ShapeBatch(b, in.slot_types);
    b.Reset(kBatchSize);
    const int64_t batches = ((int64_t{1} << 31) / kBatchSize) + 1;
    for (const int64_t share : {batches / 2, batches - batches / 2}) {
        auto local = made.first->MakeLocal();
        for (int64_t i = 0; i < share; ++i) made.first->Consume(b, *local);
        made.first->Combine(std::move(local));
    }
    made.first->Finalize(ctx);
    made.second->Prepare(ctx);
    ASSERT_EQ(made.second->MorselCount(), 1u);
    auto local = made.second->MakeLocal();
    made.second->StartMorsel(0, *local);
    Batch o;
    ASSERT_TRUE(made.second->NextBatch(*local, o));
    ASSERT_EQ(o.Active(), 1u);
    const int64_t rows = batches * kBatchSize;
    ASSERT_GT(rows, int64_t{std::numeric_limits<int32_t>::max()});
    EXPECT_EQ(o.slots[0].I64()[0], rows);
    EXPECT_EQ(o.slots[1].I64()[0], rows);
    EXPECT_FALSE(made.second->NextBatch(*local, o));
}

TEST_F(ParallelHashAggregateTest, CountPastInt32Max) {
    // A global COUNT(*) over 3 rows, offset by INT32_MAX, and grouped counts the same way.
    Load("c", Schema({Column("g", ColumnType::Integer()), Column("x", ColumnType::Integer())}), "1|10|\n1|11|\n2|12|\n");
    const auto run = [&](const std::string& sql) {
        QueryOptions options;
        options.mode = ExecutionMode::PARALLEL;
        const auto plan = db_->MakePlanner(options).Plan(std::move(*Parser(sql).Parse()));
        QueryContext ctx{db_->GetBufferPoolManager(), db_->GetCatalog()};
        PhysicalPlan physical = CompilePlan(*plan, ctx);
        size_t aggregates = 0;
        for (Pipeline& p : physical.pipelines) {
            if (p.sink_node == nullptr || p.sink_node->type != PlanNodeType::AGGREGATE) continue;
            OffsetCountsForTest(*p.sink, std::numeric_limits<int32_t>::max());
            ++aggregates;
        }
        EXPECT_EQ(aggregates, 1u) << sql;
        return QueryResult{plan->output_schema, RunPhysicalPlan(physical, ctx), {}};
    };

    const QueryResult result = run("SELECT COUNT(*), COUNT(x) + 1 FROM c");
    const Schema& schema = result.schema;
    ASSERT_EQ(result.tuples.size(), 1u);
    EXPECT_EQ(schema.GetColumn(0).type, TypeId::BIGINT);
    EXPECT_EQ(result.tuples[0].GetValue(schema, 0).type_, TypeId::BIGINT);
    EXPECT_EQ(result.tuples[0].GetValue(schema, 0).bigint_, int64_t{2147483650});
    EXPECT_EQ(schema.GetColumn(1).type, TypeId::BIGINT);
    EXPECT_EQ(result.tuples[0].GetValue(schema, 1).bigint_, int64_t{2147483651});

    const QueryResult grouped = run("SELECT g, COUNT(*) FROM c GROUP BY g ORDER BY g");
    ASSERT_EQ(grouped.tuples.size(), 2u);
    EXPECT_EQ(grouped.tuples[0].GetValue(grouped.schema, 1).bigint_, int64_t{2147483649});
    EXPECT_EQ(grouped.tuples[1].GetValue(grouped.schema, 1).bigint_, int64_t{2147483648});

    // Without the offset the same query counts its 3 rows.
    const QueryResult plain = Run("SELECT COUNT(*) FROM c", ExecutionMode::PARALLEL);
    ASSERT_EQ(plain.tuples.size(), 1u);
    EXPECT_EQ(plain.tuples[0].GetValue(plain.schema, 0).bigint_, 3);
}

//  Intended differences

TEST_F(ParallelHashAggregateTest, IntendedDifferencesFromToday) {
    LoadSmall();
    const std::string aggregate_error = "Cannot evaluate AGGREGATE or STAR expression in this context";

    // Today's planner leaves an aggregate in HAVING unrewritten: today throws once a group
    // reaches the filter, and gives no rows without one. Here it's refused when compiled.
    const std::string empty = "SELECT x, COUNT(*) FROM e WHERE x < -1000000 GROUP BY x HAVING COUNT(*) > 0";
    EXPECT_TRUE(Run(empty, ExecutionMode::VECTORIZED).tuples.empty());
    EXPECT_EQ(ErrorOf([&] { (void)Run(empty, ExecutionMode::PARALLEL); }), aggregate_error);
    const std::string with_rows = "SELECT x, COUNT(*) FROM e GROUP BY x HAVING COUNT(*) > 0";
    EXPECT_EQ(ErrorOf([&] { (void)Run(with_rows, ExecutionMode::VECTORIZED); }), aggregate_error);
    EXPECT_EQ(ErrorOf([&] { (void)Run(with_rows, ExecutionMode::PARALLEL); }), aggregate_error);

    // The planner gives a GROUP BY expression no output column, so it's refused here.
    EXPECT_EQ(ErrorOf([&] { (void)Run("SELECT x + 1, COUNT(*) FROM e GROUP BY x + 1", ExecutionMode::PARALLEL); }),
              "the parallel executor groups only by plain columns");
}

//  Hash collisions and double ties

TEST_F(ParallelHashAggregateTest, DistinctKeysWithOneHashStayApart) {
    // A row's hash folds its keys in turn with FoldKey; HashMix is a bijection, so
    // the last folded key can always be solved for to hit a given hash.
    using detail::FoldKey;
    using detail::kKeySeed;
    using Key = std::array<int64_t, 3>;
    const auto hash = [](const Key& k) {
        uint64_t h = 0;
        for (int64_t v : k) h = FoldKey(h, static_cast<uint64_t>(v));
        return h;
    };
    // The key whose fold onto `before` lands where `want_before` plus `want_key` does.
    const auto solve = [](uint64_t before, uint64_t want_before, int64_t want_key) {
        return static_cast<int64_t>((want_before ^ (static_cast<uint64_t>(want_key) + kKeySeed) ^ before) - kKeySeed);
    };
    const Key a{1, 2, 5};
    // b shares a's first key, c shares its last; all three hash alike.
    const Key b{1, 7, solve(FoldKey(FoldKey(0, 1), 7), FoldKey(FoldKey(0, 1), 2), 5)};
    const Key c{3, solve(FoldKey(0, 3), FoldKey(0, 1), 2), 5};
    ASSERT_EQ(hash(a), hash(b));
    ASSERT_EQ(hash(a), hash(c));
    ASSERT_TRUE(a != b && a != c && b != c);

    InputLayout in;
    in.slot_of_column = {0, 1, 2};
    in.slot_types = {kBig, kBig, kBig};
    in.slot_precision = {0, 0, 0};
    // Rows that collide inside one worker, and again when the workers merge.
    const std::vector<std::vector<Key>> workers = {{a, b, c, a}, {c, b, a, b}};
    for (const bool reversed : {false, true}) {
        AggregateSpec spec;
        for (uint32_t k = 0; k < 3; ++k) spec.keys.push_back(Col(k, in));
        spec.funcs = {Func::COUNT, Func::MAX};
        spec.args.push_back(nullptr);
        spec.args.push_back(Col(1, in));
        spec.out_types = {kBig, kBig, kBig, kBig, kBig};
        auto made = MakeHashAggregate(std::move(spec));
        QueryContext ctx{nullptr, nullptr};
        for (size_t w = 0; w < workers.size(); ++w) {
            const std::vector<Key>& rows = workers[reversed ? workers.size() - 1 - w : w];
            auto local = made.first->MakeLocal();
            Batch batch;
            ShapeBatch(batch, in.slot_types);
            batch.Reset(static_cast<uint32_t>(rows.size()));
            for (size_t i = 0; i < rows.size(); ++i) {
                for (size_t k = 0; k < 3; ++k) batch.slots[k].I64()[i] = rows[i][k];
            }
            made.first->Consume(batch, *local);
            made.first->Combine(std::move(local));
        }
        made.first->Finalize(ctx);
        made.second->Prepare(ctx);
        std::map<Key, std::pair<int64_t, int64_t>> groups;  // key -> (count, max of the middle key)
        auto local = made.second->MakeLocal();
        Batch o;
        for (size_t m = 0; m < made.second->MorselCount(); ++m) {
            made.second->StartMorsel(m, *local);
            while (made.second->NextBatch(*local, o)) {
                for (uint32_t i = 0; i < o.Active(); ++i) {
                    const uint32_t r = o.sel[i];
                    const Key key{o.slots[0].I64()[r], o.slots[1].I64()[r], o.slots[2].I64()[r]};
                    EXPECT_EQ(groups.count(key), 0u) << "a group came out twice";
                    groups[key] = {o.slots[3].I64()[r], o.slots[4].I64()[r]};
                }
            }
        }
        const std::map<Key, std::pair<int64_t, int64_t>> want = {{a, {3, a[1]}}, {b, {3, b[1]}}, {c, {2, c[1]}}};
        EXPECT_EQ(groups, want) << (reversed ? "reversed" : "in order");
    }
}

TEST_F(ParallelHashAggregateTest, DoubleMinMaxSameInAnyMergeOrder) {
    // In a total order -0.0 precedes +0.0 and NaN is last, so MIN keeps -0.0,
    // MAX keeps +0.0, and MAX finds a NaN that MIN skips, whatever the order.
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const InputLayout in = HandLayout();
    using Rows = std::vector<std::pair<int64_t, double>>;
    const auto bits = [](double d) {
        uint64_t b = 0;
        std::memcpy(&b, &d, sizeof(b));
        return b;
    };
    // MIN(f) and MAX(f) as bits per key (0 for a global row).
    const auto run = [&](bool global, const std::vector<Rows>& workers) {
        AggregateSpec spec;
        if (!global) {
            spec.keys.push_back(Col(0, in));
            spec.out_types.push_back(kInt);
        }
        for (const Func f : {Func::MIN, Func::MAX}) {
            spec.funcs.push_back(f);
            spec.args.push_back(Col(3, in));
            spec.out_types.push_back(kDouble);
        }
        spec.global = global;
        auto made = MakeHashAggregate(std::move(spec));
        QueryContext ctx{nullptr, nullptr};
        for (const Rows& rows : workers) {
            auto local = made.first->MakeLocal();
            Batch batch;
            ShapeBatch(batch, in.slot_types);
            batch.Reset(static_cast<uint32_t>(rows.size()));
            for (size_t i = 0; i < rows.size(); ++i) {
                batch.slots[0].I64()[i] = rows[i].first;
                batch.slots[3].F64()[i] = rows[i].second;
            }
            made.first->Consume(batch, *local);
            made.first->Combine(std::move(local));
        }
        made.first->Finalize(ctx);
        made.second->Prepare(ctx);
        std::map<int64_t, std::pair<uint64_t, uint64_t>> out;
        auto local = made.second->MakeLocal();
        Batch o;
        const size_t first = global ? 0 : 1;
        for (size_t m = 0; m < made.second->MorselCount(); ++m) {
            made.second->StartMorsel(m, *local);
            while (made.second->NextBatch(*local, o)) {
                for (uint32_t i = 0; i < o.Active(); ++i) {
                    const uint32_t r = o.sel[i];
                    out[global ? 0 : o.slots[0].I64()[r]] = {bits(o.slots[first].F64()[r]), bits(o.slots[first + 1].F64()[r])};
                }
            }
        }
        return out;
    };

    struct Case {
        std::vector<Rows> workers;
        double min;
        double max;
    };
    const std::vector<Case> cases = {
        {{{{1, 0.0}}, {{1, -0.0}}}, -0.0, 0.0},
        {{{{1, -0.0}, {1, 0.0}}, {{1, 0.0}}}, -0.0, 0.0},
        {{{{1, nan}, {1, 1.0}}, {{1, 2.0}}}, 1.0, nan},
        {{{{1, 1.0}, {1, nan}}, {{1, -3.5}}}, -3.5, nan},
        {{{{1, nan}}, {{1, nan}}}, nan, nan},
    };
    for (size_t c = 0; c < cases.size(); ++c) {
        for (const bool global : {true, false}) {
            const std::map<int64_t, std::pair<uint64_t, uint64_t>> want = {{global ? 0 : 1, {bits(cases[c].min), bits(cases[c].max)}}};
            // Workers in both orders, each worker's rows in both orders.
            for (int variant = 0; variant < 4; ++variant) {
                std::vector<Rows> workers = cases[c].workers;
                if (variant & 1) std::reverse(workers.begin(), workers.end());
                if (variant & 2) {
                    for (Rows& rows : workers) std::reverse(rows.begin(), rows.end());
                }
                EXPECT_EQ(run(global, workers), want) << "case " << c << (global ? " global" : " grouped") << " variant " << variant;
            }
        }
    }
}

TEST_F(ParallelHashAggregateTest, DoubleKeysShowTheSameBitsInAnyMergeOrder) {
    // Equal DOUBLE keys show their smallest member in a total order: -0.0 over
    // +0.0, and the NaN with the smallest bits, however rows and tables merge.
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
    const uint64_t kNanLow = 0x7FF8000000000000ULL;
    const std::vector<double> special = {0.0, -0.0, from_bits(0x7FF8000000000005ULL), from_bits(kNanLow), from_bits(0xFFF8000000000000ULL), 0.0,
                                         from_bits(0x7FF8000000000005ULL), 1.5};
    const InputLayout in = HandLayout();

    // Per key shown, as bits: its count. Filler keys are left out.
    const auto run = [&](const std::vector<std::vector<double>>& workers) {
        AggregateSpec spec;
        spec.keys.push_back(Col(3, in));
        spec.out_types.push_back(kDouble);
        spec.funcs.push_back(Func::COUNT);
        spec.args.push_back(nullptr);
        spec.out_types.push_back(kBig);
        auto made = MakeHashAggregate(std::move(spec));
        QueryContext ctx{nullptr, nullptr};
        for (const auto& keys : workers) {
            auto local = made.first->MakeLocal();
            Batch b;
            ShapeBatch(b, in.slot_types);
            for (size_t at = 0; at < keys.size(); at += kBatchSize) {
                const auto n = static_cast<uint32_t>(std::min<size_t>(kBatchSize, keys.size() - at));
                b.Reset(n);
                for (uint32_t i = 0; i < n; ++i) b.slots[3].F64()[i] = keys[at + i];
                made.first->Consume(b, *local);
            }
            made.first->Combine(std::move(local));
        }
        made.first->Finalize(ctx);
        made.second->Prepare(ctx);
        std::map<uint64_t, int64_t> shown;
        auto local = made.second->MakeLocal();
        Batch o;
        for (size_t m = 0; m < made.second->MorselCount(); ++m) {
            made.second->StartMorsel(m, *local);
            while (made.second->NextBatch(*local, o)) {
                for (uint32_t i = 0; i < o.Active(); ++i) {
                    const double k = o.slots[0].F64()[o.sel[i]];
                    if (k == 0.0 || std::isnan(k) || k == 1.5) shown[bits(k)] = o.slots[1].I64()[o.sel[i]];
                }
            }
        }
        return shown;
    };
    const std::map<uint64_t, int64_t> want = {{bits(-0.0), 3}, {kNanLow, 4}, {bits(1.5), 1}};

    // The special keys shuffled and cut into three workers; the biggest table varies.
    std::mt19937 rng(20261118);
    for (int variant = 0; variant < 24; ++variant) {
        std::vector<double> keys = special;
        for (size_t i = keys.size() - 1; i > 0; --i) std::swap(keys[i], keys[rng() % (i + 1)]);
        const size_t cut1 = rng() % (keys.size() + 1);
        const size_t cut2 = cut1 + rng() % (keys.size() - cut1 + 1);
        std::vector<std::vector<double>> workers = {std::vector<double>(keys.begin(), keys.begin() + cut1),
                                                    std::vector<double>(keys.begin() + cut1, keys.begin() + cut2),
                                                    std::vector<double>(keys.begin() + cut2, keys.end())};
        // Odd variants add 40,000 to 60,000 filler keys per worker, so tables merge by partition.
        double next = 2.0;
        size_t fillers = 0;
        for (size_t w = 0; w < workers.size(); ++w) {
            const size_t filler = variant % 2 == 0 ? (w == static_cast<size_t>(variant / 2) % 3 ? 5 : 0) : 40000 + 10000 * ((w + variant) % 3);
            for (size_t f = 0; f < filler; ++f) workers[w].push_back(next++);
            fillers += filler;
        }
        ASSERT_EQ(MergePartitionCount(fillers + special.size(), 1) > 1, variant % 2 == 1);
        EXPECT_EQ(run(workers), want) << "variant " << variant;
    }
}

TEST_F(ParallelHashAggregateTest, IntendedDifferenceDoubleKeyShowsNegativeZero) {
    // One group holds 0.0 and -0.0. Today's executor shows the first one seen;
    // this one shows -0.0, as it would at any thread count.
    Load("z", Schema({Column("zf", ColumnType::Double()), Column("zi", ColumnType::Integer())}), "0.0|1|\n-0.0|2|\n0.0|3|\n2.5|4|\n");
    const QueryResult got = Run("SELECT zf, COUNT(*) FROM z GROUP BY zf", ExecutionMode::PARALLEL, 1);
    ASSERT_EQ(got.tuples.size(), 2u);
    bool saw_zero = false;
    for (const Tuple& row : got.tuples) {
        const double k = row.GetValue(got.schema, 0).decimal_;
        if (k != 0.0) continue;
        saw_zero = true;
        EXPECT_TRUE(std::signbit(k)) << "the zero group shows +0.0";
        ASSERT_EQ(row.GetValue(got.schema, 1).type_, TypeId::BIGINT);
        EXPECT_EQ(row.GetValue(got.schema, 1).bigint_, 3);
    }
    EXPECT_TRUE(saw_zero);
    const QueryResult today = Run("SELECT zf, COUNT(*) FROM z GROUP BY zf", ExecutionMode::VECTORIZED);
    for (const Tuple& row : today.tuples) {
        const double k = row.GetValue(today.schema, 0).decimal_;
        if (k == 0.0) EXPECT_FALSE(std::signbit(k)) << "today shows the first zero seen";
    }
}

//  Spilling

constexpr unsigned kSpillThreads = 4;
constexpr size_t kSpillMorsels = 8;  // enough for every worker

const VType kWide3{Kind::DECIMAL, 3, true, TypeId::DECIMAL};

// One spill input row: slots k, s, m (cents), f, w (thousandths), d.
struct AggRow {
    int64_t k{0};
    std::string s;
    int64_t cents{0};
    double f{0.0};
    __int128 w{0};
    int64_t d{0};
};

using AggRowMaker = std::function<AggRow(uint32_t)>;

// Slots: k INTEGER, s VARCHAR, m NUMERIC(15,2), f DOUBLE, w NUMERIC(30,3), d DATE.
InputLayout AggLayout() {
    InputLayout in;
    in.slot_of_column = {0, 1, 2, 3, 4, 5};
    in.slot_types = {kInt, kStr, kMoney, kDouble, kWide3, kDate};
    in.slot_precision = {0, 0, 15, 0, 30, 0};
    return in;
}

// An aggregate over the spill input: key slots, then (func, slot); slot -1 is COUNT(*).
struct AggCase {
    std::vector<uint32_t> keys;
    std::vector<std::pair<Func, int>> aggs;
    std::string label;
};

// The case compiled over the input slots; no keys is a global aggregate.
AggregateSpec SpecOf(const AggCase& c) {
    const InputLayout in = AggLayout();
    AggregateSpec spec;
    for (uint32_t k : c.keys) {
        spec.keys.push_back(Col(k, in));
        spec.out_types.push_back(in.slot_types[k]);
    }
    for (const auto& a : c.aggs) {
        spec.funcs.push_back(a.first);
        spec.args.push_back(a.second < 0 ? nullptr : Col(static_cast<uint32_t>(a.second), in));
        spec.out_types.push_back(AggregateOutputType(a.first, spec.args.back().get()));
    }
    spec.global = c.keys.empty();
    return spec;
}

// Output slots that may move in their last bits: DOUBLE sums and averages.
std::vector<bool> ApproxSlots(const AggCase& c) {
    std::vector<bool> approx(c.keys.size(), false);
    for (const auto& a : c.aggs) approx.push_back((a.first == Func::SUM || a.first == Func::AVG) && a.second == 3);
    return approx;
}

// Hands out rows 0..n-1 in morsels, s plain or dictionary-coded.
// Batches of 2,048, 1,000, 7 and 333 rows in turn.
// A morsel's strings are wiped when its worker starts another or finishes,
// so an aggregate that kept a pointer instead of a copy reads '#'.
class AggInput : public Source {
public:
    struct Local : LocalState {
        std::vector<AggRow> rows;           // this morsel's rows
        std::vector<std::string> distinct;  // dictionary strings
        std::vector<StringRef> entries;     // dictionary entries over distinct
        std::vector<uint32_t> codes;        // one per row
        size_t next{0};
        size_t batch{0};

        ~Local() override { Wipe(); }

        void Wipe() {
            for (AggRow& r : rows) std::fill(r.s.begin(), r.s.end(), '#');
            for (std::string& s : distinct) std::fill(s.begin(), s.end(), '#');
        }
    };

    AggInput(uint32_t rows, size_t morsels, AggRowMaker make, bool dictionary)
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
            for (const AggRow& r : l.rows) {
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

    bool NextBatch(LocalState& local, Batch& out) const override {
        auto& l = static_cast<Local&>(local);
        if (l.next >= l.rows.size()) return false;
        static constexpr uint32_t kSizes[] = {kBatchSize, 1000, 7, 333};
        const auto n = static_cast<uint32_t>(std::min<size_t>(kSizes[l.batch++ % 4], l.rows.size() - l.next));
        ShapeBatch(out, AggLayout().slot_types);
        out.Reset(n);
        for (uint32_t i = 0; i < n; ++i) {
            const AggRow& r = l.rows[l.next + i];
            out.slots[0].I64()[i] = r.k;
            out.slots[2].I64()[i] = r.cents;
            out.slots[3].F64()[i] = r.f;
            out.slots[4].I128()[i] = r.w;
            out.slots[5].I64()[i] = r.d;
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
    AggRowMaker make_;
    bool dictionary_;
};

// The input: how many rows, made how, and whether s is dictionary-coded.
struct AggData {
    uint32_t rows;
    AggRowMaker make;
    bool dictionary{false};
};

// One output row: its exact cells as bytes, its DOUBLE sums and averages as numbers.
struct OutRow {
    std::string exact;
    std::vector<double> approx;
};

// Appends raw bytes.
void AppendRaw(std::string& out, const void* data, size_t n) {
    out.append(static_cast<const char*>(data), n);
}

// Appends a cell's exact bytes: numbers raw, strings length-prefixed.
void AppendCell(const Vector& v, uint32_t row, std::string& out) {
    switch (ArrayOf(v.Type())) {
        case ValueArray::I64: AppendRaw(out, &v.I64()[row], sizeof(int64_t)); break;
        case ValueArray::I128: AppendRaw(out, &v.I128()[row], sizeof(__int128)); break;
        case ValueArray::F64: AppendRaw(out, &v.F64()[row], sizeof(double)); break;
        case ValueArray::STR: {
            const StringRef s = v.StrAt(row);
            AppendRaw(out, &s.size, sizeof(s.size));
            AppendRaw(out, s.data, s.size);
            break;
        }
    }
}

// Keeps every row it's given: exact cells as bytes, approximate ones as numbers.
class GroupSink : public Sink {
public:
    struct Local : LocalState {
        std::vector<OutRow> rows;
    };

    explicit GroupSink(std::vector<bool> approx) : approx_(std::move(approx)) {}

    std::unique_ptr<LocalState> MakeLocal() const override { return std::make_unique<Local>(); }

    void Consume(Batch& in, LocalState& local) const override {
        auto& l = static_cast<Local&>(local);
        EXPECT_EQ(in.slots.size(), approx_.size());
        for (uint32_t i = 0; i < in.Active(); ++i) {
            OutRow row;
            for (size_t c = 0; c < in.slots.size(); ++c) {
                if (approx_[c]) {
                    row.approx.push_back(in.slots[c].F64()[in.sel[i]]);
                } else {
                    AppendCell(in.slots[c], in.sel[i], row.exact);
                }
            }
            l.rows.push_back(std::move(row));
        }
    }

    void Combine(std::unique_ptr<LocalState> local) override {
        auto& l = static_cast<Local&>(*local);
        rows.insert(rows.end(), std::make_move_iterator(l.rows.begin()), std::make_move_iterator(l.rows.end()));
    }

    void Finalize(QueryContext&) override {}

    std::vector<OutRow> rows;

private:
    std::vector<bool> approx_;
};

// Equal, both NaN, or within 1e-9 relative.
bool CloseDouble(double a, double b) {
    if (a == b || (std::isnan(a) && std::isnan(b))) return true;
    return std::abs(a - b) <= 1e-9 * std::max({1.0, std::abs(a), std::abs(b)});
}

// "" when both hold the same rows in any order, exact cells byte for byte.
// Exact cells hold the group keys, so each row's place after sorting is its group's.
std::string GroupDifference(std::vector<OutRow> got, std::vector<OutRow> want) {
    if (got.size() != want.size()) return std::to_string(got.size()) + " rows, expected " + std::to_string(want.size());
    const auto by_exact = [](const OutRow& a, const OutRow& b) { return a.exact < b.exact; };
    std::sort(got.begin(), got.end(), by_exact);
    std::sort(want.begin(), want.end(), by_exact);
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i].exact != want[i].exact) return "row " + std::to_string(i) + ": exact cells differ";
        if (got[i].approx.size() != want[i].approx.size()) return "row " + std::to_string(i) + ": approximate cell counts differ";
        for (size_t j = 0; j < got[i].approx.size(); ++j) {
            if (!CloseDouble(got[i].approx[j], want[i].approx[j])) {
                return "row " + std::to_string(i) + ": " + std::to_string(got[i].approx[j]) + " != " + std::to_string(want[i].approx[j]);
            }
        }
    }
    return "";
}

// "" when both results have the same types and tuples byte for byte, any order.
std::string ExactMultisetDifference(const QueryResult& got, const QueryResult& want) {
    if (got.schema.GetColumnCount() != want.schema.GetColumnCount()) return "column counts differ";
    for (uint32_t c = 0; c < got.schema.GetColumnCount(); ++c) {
        if (got.schema.GetColumn(c).type != want.schema.GetColumn(c).type) return "column " + std::to_string(c) + " has another type";
    }
    const auto bytes = [](const QueryResult& r) {
        std::vector<std::string> out;
        for (const Tuple& t : r.tuples) out.emplace_back(t.GetData(), t.GetLength());
        std::sort(out.begin(), out.end());
        return out;
    };
    const std::vector<std::string> x = bytes(got);
    const std::vector<std::string> y = bytes(want);
    if (x.size() != y.size()) return std::to_string(x.size()) + " rows, expected " + std::to_string(y.size());
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] != y[i]) return "sorted row " + std::to_string(i) + " differs";
    }
    return "";
}

// "" when types and rows match in order, byte for byte.
std::string ExactOrderedDifference(const QueryResult& got, const QueryResult& want) {
    if (got.schema.GetColumnCount() != want.schema.GetColumnCount()) return "column counts differ";
    for (uint32_t c = 0; c < got.schema.GetColumnCount(); ++c) {
        if (got.schema.GetColumn(c).type != want.schema.GetColumn(c).type) return "column " + std::to_string(c) + " has another type";
    }
    if (got.tuples.size() != want.tuples.size()) return std::to_string(got.tuples.size()) + " rows, expected " + std::to_string(want.tuples.size());
    for (size_t i = 0; i < got.tuples.size(); ++i) {
        const Tuple& x = got.tuples[i];
        const Tuple& y = want.tuples[i];
        if (x.GetLength() != y.GetLength() || std::memcmp(x.GetData(), y.GetData(), x.GetLength()) != 0) {
            return "row " + std::to_string(i) + ": got " + Describe(RowOf(got, i)) + "expected " + Describe(RowOf(want, i));
        }
    }
    return "";
}

// How one run of the aggregate is set up.
struct AggSpillConfig {
    unsigned threads{1};
    bool spill_everything{false};
    std::optional<size_t> limit;  // the budget's limit; none means no budget
    size_t held{0};               // bytes the test holds on every worker's floor first
    int fail_writes{0};           // errno every temp write fails with; 0 for none
    bool expect_error{false};     // the run should throw even without failing writes
    int64_t count_offset{0};      // added to every COUNT, through OffsetCountsForTest
    std::function<void(MemoryBudget*, const std::string&)> on_open;  // told of each new temp file
    std::function<void()> on_done;                                   // after the run, before the checks
};

// No budget at all.
AggSpillConfig Unspilled(unsigned threads) {
    AggSpillConfig c;
    c.threads = threads;
    return c;
}

// A budget of just the floors, with `held` of each floor taken by the test.
AggSpillConfig FloorsOnly(unsigned threads, size_t held) {
    AggSpillConfig c;
    c.threads = threads;
    c.limit = threads * kWorkerMinimumBytes;
    c.held = held;
    return c;
}

// Every breaker spills everything it holds; `held` as above.
AggSpillConfig Everything(unsigned threads, size_t held = 0) {
    AggSpillConfig c = FloorsOnly(threads, held);
    c.spill_everything = true;
    return c;
}

// The groups a run gave and what it spilled.
struct AggSpillResult {
    std::vector<OutRow> rows;
    uint64_t files{0};
    uint64_t written{0};
    uint64_t read{0};
    uint64_t partitions{0};
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

// Passes each new temp file's path to hook while alive.
class OpenHookScope {
public:
    explicit OpenHookScope(std::function<void(const std::string&)> hook) { TempFile::SetOpenHookForTest(std::move(hook)); }
    ~OpenHookScope() { TempFile::SetOpenHookForTest(nullptr); }
    OpenHookScope(const OpenHookScope&) = delete;
    OpenHookScope& operator=(const OpenHookScope&) = delete;
};

// Forwards to a sink, running a hook just before its Finalize.
class BeforeFinalize : public Sink {
public:
    BeforeFinalize(Sink& inner, std::function<void()> hook) : inner_(inner), hook_(std::move(hook)) {}

    std::unique_ptr<LocalState> MakeLocal() const override { return inner_.MakeLocal(); }
    std::unique_ptr<LocalState> MakeWorkerLocal(const QueryContext& ctx, unsigned worker) const override {
        return inner_.MakeWorkerLocal(ctx, worker);
    }
    void Consume(Batch& in, LocalState& local) const override { inner_.Consume(in, local); }
    void Combine(std::unique_ptr<LocalState> local) override { inner_.Combine(std::move(local)); }
    void Finalize(QueryContext& ctx) override {
        if (hook_) hook_();
        inner_.Finalize(ctx);
    }

private:
    Sink& inner_;
    std::function<void()> hook_;
};

// Caps the groups a merge of spilled groups holds before it splits, while alive.
class MergeCapScope {
public:
    explicit MergeCapScope(size_t most) { SetAggregateMergeGroupsForTest(most); }
    ~MergeCapScope() { SetAggregateMergeGroupsForTest(0); }
    MergeCapScope(const MergeCapScope&) = delete;
    MergeCapScope& operator=(const MergeCapScope&) = delete;
};

// Opens a second descriptor on every new temp file while alive; closes them when done.
class SecondDescriptors {
public:
    SecondDescriptors() {
        TempFile::SetOpenHookForTest([this](const std::string& path) {
            const int fd = ::open(path.c_str(), O_RDWR);
            const std::lock_guard<std::mutex> lock(mutex_);
            fds_.push_back(fd);
        });
    }
    ~SecondDescriptors() { Close(); }
    SecondDescriptors(const SecondDescriptors&) = delete;
    SecondDescriptors& operator=(const SecondDescriptors&) = delete;

    // Flips every bit of the byte at offset in each file; how many it flipped.
    size_t FlipByte(uint64_t offset) {
        const std::lock_guard<std::mutex> lock(mutex_);
        size_t flipped = 0;
        for (const int fd : fds_) {
            unsigned char b = 0;
            if (fd < 0 || ::pread(fd, &b, 1, static_cast<off_t>(offset)) != 1) continue;
            b = static_cast<unsigned char>(~b);
            if (::pwrite(fd, &b, 1, static_cast<off_t>(offset)) == 1) ++flipped;
        }
        return flipped;
    }

    // Stops watching and closes every descriptor.
    void Close() {
        TempFile::SetOpenHookForTest(nullptr);
        const std::lock_guard<std::mutex> lock(mutex_);
        for (const int fd : fds_) {
            if (fd >= 0) ::close(fd);
        }
        fds_.clear();
    }

private:
    std::mutex mutex_;
    std::vector<int> fds_;
};

// Flips CRC bits in every block of a finished temp file; returns the blocks damaged.
size_t DamageEveryBlock(int fd) {
    const off_t size = ::lseek(fd, 0, SEEK_END);
    size_t damaged = 0;
    for (off_t at = 0; at + static_cast<off_t>(sizeof(TempBlockHeader)) <= size;) {
        TempBlockHeader h{};
        if (::pread(fd, &h, sizeof(h), at) != static_cast<ssize_t>(sizeof(h))) break;
        h.crc ^= 0xFFu;
        if (::pwrite(fd, &h, sizeof(h), at) == static_cast<ssize_t>(sizeof(h))) ++damaged;
        at += static_cast<off_t>(sizeof(TempBlockHeader) + h.payload_bytes);
    }
    return damaged;
}

// Keeps a second descriptor on every new temp file, and damages files already finished:
// the sink's at the first open after the sink,
// or a thread's earlier splits at that thread's next open.
class MergeReadDamage {
public:
    enum class Target { SINK_FILES, SPLIT_FILES };

    explicit MergeReadDamage(Target target) : target_(target) {}
    ~MergeReadDamage() { Close(); }
    MergeReadDamage(const MergeReadDamage&) = delete;
    MergeReadDamage& operator=(const MergeReadDamage&) = delete;

    // From the open hook, on any thread. A thread writes one split at a time,
    // so its earlier splits are finished.
    void Opened(const std::string& path) {
        const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
        const std::thread::id self = std::this_thread::get_id();
        const std::lock_guard<std::mutex> lock(mutex_);
        if (sink_done_) {
            for (Kept& k : kept_) {
                const bool hit = target_ == Target::SINK_FILES ? k.in_sink : !k.in_sink && k.thread == self;
                if (!hit || k.fd < 0) continue;
                damaged_ += DamageEveryBlock(k.fd);
                ::close(k.fd);
                k.fd = -1;
            }
        }
        kept_.push_back(Kept{fd, self, !sink_done_});
    }

    // Files opened from now on are the source's splits.
    void SinkDone() {
        const std::lock_guard<std::mutex> lock(mutex_);
        sink_done_ = true;
    }

    // Blocks damaged so far.
    [[nodiscard]] size_t Damaged() {
        const std::lock_guard<std::mutex> lock(mutex_);
        return damaged_;
    }

    // Closes every descriptor still kept.
    void Close() {
        const std::lock_guard<std::mutex> lock(mutex_);
        for (Kept& k : kept_) {
            if (k.fd >= 0) ::close(k.fd);
            k.fd = -1;
        }
    }

private:
    struct Kept {
        int fd;
        std::thread::id thread;
        bool in_sink;
    };

    Target target_;
    std::mutex mutex_;
    std::vector<Kept> kept_;
    bool sink_done_{false};
    size_t damaged_{0};
};

// One row with key k: cents from k and a salt, nothing else set.
AggRow KeyRow(int64_t k, int64_t salt) {
    AggRow r;
    r.k = k;
    r.cents = (k * 7919 + salt * 104729) % 2000001 - 1000000;
    return r;
}

// GROUP BY k with COUNT(*) and SUM(m), exactly, as the aggregate's bytes.
std::vector<OutRow> CountSumReference(const std::vector<std::vector<AggRow>>& parts) {
    std::map<int64_t, std::pair<int64_t, __int128>> ref;
    for (const auto& rows : parts) {
        for (const AggRow& r : rows) {
            auto& e = ref[r.k];
            ++e.first;
            e.second += r.cents;
        }
    }
    std::vector<OutRow> want;
    for (const auto& [k, e] : ref) {
        OutRow row;
        AppendRaw(row.exact, &k, sizeof(k));
        AppendRaw(row.exact, &e.first, sizeof(int64_t));
        AppendRaw(row.exact, &e.second, sizeof(__int128));
        want.push_back(std::move(row));
    }
    return want;
}

// What a hand-fed run gave.
struct DrivenAgg {
    std::vector<OutRow> rows;
    uint64_t files{0};
    size_t peak{0};     // the budget's peak, read right after Finalize
    size_t morsels{0};  // the source's morsels
};

// The spill partition of a lone INTEGER key at one level.
size_t KeyPartition(int64_t k, int level) {
    return SpillPartitionOf(detail::FoldKey(0, static_cast<uint64_t>(k)), level);
}

class ParallelHashAggregateSpillTest : public ParallelHashAggregateTest {
protected:
    std::string temp_dir_;
    std::unique_ptr<WorkerPool> pool_;

    void SetUp() override {
        ParallelHashAggregateTest::SetUp();
        temp_dir_ = TempPath("_spill_dir");
        std::filesystem::remove_all(temp_dir_);
        db_->SetTempDirectory(temp_dir_);
        pool_ = std::make_unique<WorkerPool>(kSpillThreads);
    }

    void TearDown() override {
        pool_.reset();
        ParallelHashAggregateTest::TearDown();
        std::filesystem::remove_all(temp_dir_);
    }

    // The temp directory has no entries, or doesn't exist yet.
    bool TempDirEmpty() const {
        std::error_code ec;
        return !std::filesystem::exists(temp_dir_, ec) || std::filesystem::is_empty(temp_dir_, ec);
    }

    // Groups the input through two pipelines, as a query would; after_sink runs between them,
    // before_finalize just before the sink's Finalize. Afterwards no grant, file or descriptor is left.
    AggSpillResult RunAgg(const AggCase& c, const AggData& data, const AggSpillConfig& cfg,
                          const std::function<void(MemoryBudget*)>& after_sink = {}, const std::function<void()>& before_finalize = {}) {
        const std::string label = c.label + ", " + std::to_string(cfg.threads) + " threads" + (cfg.spill_everything ? ", everything" : "") +
                                  (cfg.limit ? ", limit " + std::to_string(*cfg.limit) : "") + ", held " + std::to_string(cfg.held) +
                                  (data.dictionary ? ", coded" : "");
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
        auto made = MakeHashAggregate(SpecOf(c));
        if (cfg.count_offset != 0) OffsetCountsForTest(*made.first, cfg.count_offset);
        AggInput input(data.rows, kSpillMorsels, data.make, data.dictionary);
        GroupSink out(ApproxSlots(c));
        QueryContext ctx{nullptr, nullptr, cfg.threads, cfg.threads > 1 ? pool_.get() : nullptr};
        ctx.budget = b;
        ctx.spill = &counters;
        ctx.temp_dir = &temp_dir_;
        ctx.spill_everything = cfg.spill_everything;

        BeforeFinalize sink(*made.first, before_finalize);
        Pipeline group_in{&input, {}, &sink, {}};
        Pipeline group_out{made.second.get(), {}, &out, {0}};
        std::string error;
        try {
            std::optional<FailWritesScope> fail;
            if (cfg.fail_writes != 0) fail.emplace(cfg.fail_writes);
            std::optional<OpenHookScope> opens;
            if (cfg.on_open) opens.emplace([&cfg, b](const std::string& path) { cfg.on_open(b, path); });
            RunPipeline(group_in, ctx);
            if (after_sink) after_sink(b);
            RunPipeline(group_out, ctx);
        } catch (const std::exception& e) {
            error = e.what();
        }
        if (cfg.on_done) cfg.on_done();

        made.first.reset();
        made.second.reset();
        held.clear();
        EXPECT_EQ(error.empty(), cfg.fail_writes == 0 && !cfg.expect_error) << label << ": " << error;
        if (!cfg.limit && !cfg.spill_everything) EXPECT_EQ(counters.files.load(), 0u) << label;
        if (b != nullptr) EXPECT_EQ(b->HeldBytes(), 0u) << label;
        EXPECT_TRUE(TempDirEmpty()) << label;
        EXPECT_EQ(OpenDescriptors(), descriptors) << label;
        return AggSpillResult{std::move(out.rows), counters.files.load(), counters.bytes_written.load(), counters.bytes_read.load(),
                              counters.partitions.load(), error};
    }

    // A query with these threads, spilling everything or not.
    QueryResult RunWith(Database& db, const std::string& sql, unsigned threads, bool spill_everything) {
        QueryOptions options;
        options.mode = ExecutionMode::PARALLEL;
        options.threads = threads;
        options.spill_everything = spill_everything;
        return db.ExecuteSQL(sql, options);
    }

    // Feeds parts[w] to worker w's counted local by hand, so each worker's table is known,
    // then finalizes and reads the groups through a pipeline. Afterwards nothing is left.
    // The hooks run just before and just after Finalize.
    DrivenAgg DriveAgg(const AggCase& c, const std::vector<std::vector<AggRow>>& parts, size_t limit, bool pooled,
                       const std::function<void(MemoryBudget&)>& before_finalize = {}, const std::function<void(MemoryBudget&)>& after_finalize = {}) {
        const auto threads = static_cast<unsigned>(parts.size());
        const size_t descriptors = OpenDescriptors();
        DrivenAgg got;
        {
            MemoryBudget budget(limit, threads);
            SpillCounters counters;
            auto made = MakeHashAggregate(SpecOf(c));
            QueryContext ctx{nullptr, nullptr, threads, pooled ? pool_.get() : nullptr};
            ctx.budget = &budget;
            ctx.spill = &counters;
            ctx.temp_dir = &temp_dir_;
            Sink& sink = *made.first;
            for (unsigned w = 0; w < threads; ++w) {
                auto local = sink.MakeWorkerLocal(ctx, w);
                Batch b;
                for (size_t at = 0; at < parts[w].size(); at += kBatchSize) {
                    const auto n = static_cast<uint32_t>(std::min<size_t>(kBatchSize, parts[w].size() - at));
                    ShapeBatch(b, AggLayout().slot_types);
                    b.Reset(n);
                    for (uint32_t i = 0; i < n; ++i) {
                        const AggRow& r = parts[w][at + i];
                        b.slots[0].I64()[i] = r.k;
                        b.slots[1].Str()[i] = StringRef{r.s.data(), static_cast<uint32_t>(r.s.size())};
                        b.slots[2].I64()[i] = r.cents;
                        b.slots[3].F64()[i] = r.f;
                        b.slots[4].I128()[i] = r.w;
                        b.slots[5].I64()[i] = r.d;
                    }
                    sink.Consume(b, *local);
                }
                sink.Combine(std::move(local));
            }
            if (before_finalize) before_finalize(budget);
            sink.Finalize(ctx);
            got.peak = budget.PeakBytes();
            if (after_finalize) after_finalize(budget);

            GroupSink out(ApproxSlots(c));
            Pipeline read{made.second.get(), {}, &out, {0}};
            RunPipeline(read, ctx);
            got.morsels = made.second->MorselCount();
            got.rows = std::move(out.rows);
            got.files = counters.files.load();
            made.first.reset();
            made.second.reset();
            EXPECT_EQ(budget.HeldBytes(), 0u);
        }
        EXPECT_TRUE(TempDirEmpty());
        EXPECT_EQ(OpenDescriptors(), descriptors);
        return got;
    }
};

TEST_F(ParallelHashAggregateSpillTest, ManyGroupsSpillAndMergeExactly) {
    // 450,000 rows over 300,000 keys: row i has key (i * 104729) mod 300,000.
    constexpr uint32_t kRows = 450000;
    const AggRowMaker make = [](uint32_t i) {
        AggRow r;
        r.k = (static_cast<int64_t>(i) * 104729) % kManyKeys;
        r.cents = (static_cast<int64_t>(i) * 7919) % 2000001 - 1000000;
        r.w = static_cast<__int128>(r.cents) * 1000000007 * 1000000007 + i;  // well past 64 bits
        return r;
    };
    // The exact reference, as the aggregate's bytes: key, COUNT(*), SUM(m), SUM(w).
    std::map<int64_t, std::tuple<int64_t, __int128, __int128>> ref;
    for (uint32_t i = 0; i < kRows; ++i) {
        const AggRow r = make(i);
        auto& e = ref[r.k];
        ++std::get<0>(e);
        std::get<1>(e) += r.cents;
        std::get<2>(e) += r.w;
    }
    ASSERT_EQ(ref.size(), kManyKeys);
    std::vector<OutRow> want;
    for (const auto& [k, e] : ref) {
        OutRow row;
        AppendRaw(row.exact, &k, sizeof(k));
        AppendRaw(row.exact, &std::get<0>(e), sizeof(int64_t));
        AppendRaw(row.exact, &std::get<1>(e), sizeof(__int128));
        AppendRaw(row.exact, &std::get<2>(e), sizeof(__int128));
        want.push_back(std::move(row));
    }

    const AggCase c{{0}, {{Func::COUNT, -1}, {Func::SUM, 2}, {Func::SUM, 4}}, "many groups"};
    const AggData data{kRows, make};
    for (const unsigned threads : {1u, kSpillThreads}) {
        const AggSpillResult plain = RunAgg(c, data, Unspilled(threads));
        EXPECT_EQ(GroupDifference(plain.rows, want), "") << threads << " threads";
        // Every Consume spills; then only past the 8 MiB floors (300,000 groups pass one).
        for (const AggSpillConfig& cfg : {Everything(threads), FloorsOnly(threads, 0)}) {
            const AggSpillResult got = RunAgg(c, data, cfg);
            EXPECT_EQ(GroupDifference(got.rows, want), "") << threads << " threads";
            EXPECT_GT(got.files, 0u) << threads << " threads";
            EXPECT_GT(got.written, 0u) << threads << " threads";
            EXPECT_EQ(got.read, got.written) << threads << " threads: every block read once";
            EXPECT_EQ(got.partitions, kSpillPartitions) << threads << " threads";
        }
    }
}

TEST_F(ParallelHashAggregateSpillTest, SpilledAndInMemoryGroupsMerge) {
    // Two grants of shared room. At one thread the table spills past 16 MiB,
    // and the last ~47,000 groups stay in memory; at four, one worker's table stays.
    // Keys 0 to 9,999 come before and after the spill, so their states meet.
    constexpr uint32_t kRows = 310000;
    const AggRowMaker make = [](uint32_t i) {
        AggRow r;
        r.k = i % kManyKeys;
        r.s = "s" + std::to_string(i % 1000);
        r.cents = (static_cast<int64_t>(i) * 7919) % 2000001 - 1000000;
        r.w = static_cast<__int128>(r.cents) * 1000000007 * 1000000007 + i;
        return r;
    };
    const AggCase c{{0}, {{Func::COUNT, -1}, {Func::SUM, 2}, {Func::SUM, 4}, {Func::MIN, 1}}, "spilled and in memory"};
    const AggData data{kRows, make};
    const AggSpillResult want = RunAgg(c, data, Unspilled(1));
    ASSERT_EQ(want.rows.size(), kManyKeys);
    for (const unsigned threads : {1u, kSpillThreads}) {
        AggSpillConfig cfg = FloorsOnly(threads, 0);
        cfg.limit = threads * kWorkerMinimumBytes + 2 * kGrantBytes;
        const AggSpillResult got = RunAgg(c, data, cfg);
        EXPECT_EQ(GroupDifference(got.rows, want.rows), "") << threads << " threads";
        EXPECT_GT(got.files, 0u) << threads << " threads";
        EXPECT_GT(got.written, 0u) << threads << " threads";
        EXPECT_EQ(got.read, got.written) << threads << " threads";
    }
}

TEST_F(ParallelHashAggregateSpillTest, OneGroupHoldsEveryRow) {
    // Every row in one group, grouped and global: many partial states of it on disk.
    constexpr uint32_t kRows = 20000;
    const AggRowMaker make = [](uint32_t i) {
        AggRow r;
        r.k = 7;
        r.s = std::string(kWords[i % 7]) + std::to_string(i % 13);
        r.cents = (static_cast<int64_t>(i) * 7919) % 2000001 - 1000000;
        r.f = static_cast<double>(static_cast<int64_t>(i % 2001) - 1000) / 4.0;
        r.w = static_cast<__int128>(r.cents) * 1000000007;
        r.d = 9000 + i % 400;
        return r;
    };
    const std::vector<std::pair<Func, int>> aggs = {{Func::COUNT, -1}, {Func::SUM, 2}, {Func::MIN, 1}, {Func::MAX, 1}, {Func::AVG, 2},
                                                    {Func::MIN, 3},    {Func::MAX, 3}, {Func::SUM, 4}, {Func::MAX, 5}};
    for (const AggCase& c : {AggCase{{0}, aggs, "one group"}, AggCase{{}, aggs, "global"}}) {
        const AggData data{kRows, make};
        const AggSpillResult want = RunAgg(c, data, Unspilled(1));
        ASSERT_EQ(want.rows.size(), 1u) << c.label;
        for (const unsigned threads : {1u, kSpillThreads}) {
            // Every Consume spills; every Grow is refused (each batch spills, the merge can't count);
            // and both.
            for (const AggSpillConfig& cfg : {Everything(threads), FloorsOnly(threads, kWorkerMinimumBytes), Everything(threads, kWorkerMinimumBytes)}) {
                const AggSpillResult got = RunAgg(c, data, cfg);
                EXPECT_EQ(GroupDifference(got.rows, want.rows), "") << c.label << ", " << threads << " threads";
                EXPECT_GT(got.files, 0u) << c.label << ", " << threads << " threads";
                EXPECT_GT(got.written, 0u) << c.label << ", " << threads << " threads";
                EXPECT_EQ(got.partitions, 1u) << c.label << ", " << threads << " threads";
            }
        }
    }
}

TEST_F(ParallelHashAggregateSpillTest, PartitionTooBigRepartitions) {
    // 64,000 groups, about 1,000 per partition; past 500 a merge splits, so each splits once.
    {
        constexpr uint32_t kRows = 128000;
        const AggRowMaker make = [](uint32_t i) {
            AggRow r;
            r.k = i % 64000;
            r.s = "g" + std::to_string(i % 977);
            r.cents = (static_cast<int64_t>(i) * 7919) % 2000001 - 1000000;
            r.f = static_cast<double>(i % 1001) / 8.0;
            return r;
        };
        const AggCase c{{0}, {{Func::COUNT, -1}, {Func::SUM, 2}, {Func::MIN, 1}, {Func::MAX, 3}}, "level 1"};
        const AggData data{kRows, make};
        const AggSpillResult want = RunAgg(c, data, Unspilled(1));
        ASSERT_EQ(want.rows.size(), 64000u);
        for (const unsigned threads : {1u, kSpillThreads}) {
            const AggSpillResult whole = RunAgg(c, data, Everything(threads));
            EXPECT_EQ(GroupDifference(whole.rows, want.rows), "") << threads << " threads";
            ASSERT_GE(whole.files, 1u);
            ASSERT_LE(whole.files, threads);  // one file per spilling worker, no splits
            EXPECT_GT(whole.written, 0u) << threads << " threads";
            const MergeCapScope cap(500);
            const AggSpillResult split = RunAgg(c, data, Everything(threads));
            EXPECT_EQ(GroupDifference(split.rows, want.rows), "") << threads << " threads";
            EXPECT_GE(split.files, kSpillPartitions + 1) << threads << " threads";
            EXPECT_GT(split.written, 0u) << threads << " threads";
            EXPECT_LE(split.files, kSpillPartitions + threads) << threads << " threads";
            // A refused merge has read some blocks, which the split reads again.
            EXPECT_GE(split.read, split.written) << threads << " threads";
            if (threads == 1) EXPECT_EQ(split.files, kSpillPartitions + 1);
        }
    }

    // Three keys (k, d) with one hash: no split can part them,
    // so after one split their piece holds every row and merges anyway.
    {
        using detail::FoldKey;
        using detail::kKeySeed;
        const auto hash = [](int64_t k, int64_t d) { return FoldKey(FoldKey(0, static_cast<uint64_t>(k)), static_cast<uint64_t>(d)); };
        // The d that makes (k, d) hash as (1, 5) does.
        const auto solve = [](int64_t k) {
            return static_cast<int64_t>((FoldKey(0, 1) ^ (uint64_t{5} + kKeySeed) ^ FoldKey(0, static_cast<uint64_t>(k))) - kKeySeed);
        };
        const std::array<std::pair<int64_t, int64_t>, 3> keys = {{{1, 5}, {2, solve(2)}, {3, solve(3)}}};
        ASSERT_EQ(hash(keys[0].first, keys[0].second), hash(keys[1].first, keys[1].second));
        ASSERT_EQ(hash(keys[0].first, keys[0].second), hash(keys[2].first, keys[2].second));
        const AggRowMaker make = [keys](uint32_t i) {
            AggRow r;
            r.k = keys[i % 3].first;
            r.d = keys[i % 3].second;
            r.s = kWords[i % 7];
            r.cents = static_cast<int64_t>(i) * 3 - 4000;
            return r;
        };
        const AggCase c{{0, 5}, {{Func::COUNT, -1}, {Func::SUM, 2}, {Func::MIN, 1}}, "one hash"};
        const AggData data{3000, make};
        const AggSpillResult want = RunAgg(c, data, Unspilled(1));
        ASSERT_EQ(want.rows.size(), 3u);
        const MergeCapScope cap(2);
        for (const unsigned threads : {1u, kSpillThreads}) {
            const AggSpillResult got = RunAgg(c, data, Everything(threads));
            EXPECT_EQ(GroupDifference(got.rows, want.rows), "") << threads << " threads";
            EXPECT_GE(got.files, 2u) << threads << " threads";
            EXPECT_GT(got.written, 0u) << threads << " threads";
            if (threads == 1) EXPECT_EQ(got.files, 2u) << "the sink's file and one split";
        }
    }

    // Every key in one spill partition, each with a 300-byte MIN(s): about 6.6 MB
    // against one free grant per worker, so the budget refuses partway and it splits.
    {
        constexpr uint32_t kRows = 20000;
        std::vector<int64_t> keys;
        for (int64_t k = 1; keys.size() < kRows; ++k) {
            if (KeyPartition(k, 0) == KeyPartition(1, 0)) keys.push_back(k);
        }
        const AggRowMaker make = [keys](uint32_t i) {
            AggRow r;
            r.k = keys[i];
            r.s = std::to_string(i) + std::string(300, 'x');
            r.cents = i;
            return r;
        };
        const AggCase c{{0}, {{Func::COUNT, -1}, {Func::MIN, 1}, {Func::SUM, 2}}, "one big partition"};
        const AggData data{kRows, make};
        const AggSpillResult want = RunAgg(c, data, Unspilled(1));
        for (const unsigned threads : {1u, kSpillThreads}) {
            const AggSpillResult got = RunAgg(c, data, FloorsOnly(threads, kWorkerMinimumBytes - kGrantBytes));
            EXPECT_EQ(GroupDifference(got.rows, want.rows), "") << threads << " threads";
            EXPECT_EQ(got.partitions, 1u) << threads << " threads";
            EXPECT_GE(got.files, 2u) << threads << " threads";
            EXPECT_GT(got.written, 0u) << threads << " threads";
            if (threads == 1) EXPECT_EQ(got.files, 2u) << "the sink's file and one split";
        }
    }

    // Groups of 130 sums (2,096 bytes) in one partition; 8 morsels of 1,125 rows:
    // the sink spills 6,750 groups and keeps 2,250 in memory, moved to the shared pool.
    // The merge has one grant free, and its first check (2,048 groups) needs two.
    // Refused with a grant free, it splits.
    // Several workers would spill every table here, so this case runs at one thread.
    {
        constexpr uint32_t kRows = 9000;
        std::vector<int64_t> keys;
        for (int64_t k = 1; keys.size() < kRows; ++k) {
            if (KeyPartition(k, 0) == KeyPartition(1, 0)) keys.push_back(k);
        }
        const AggRowMaker make = [keys](uint32_t i) {
            AggRow r;
            r.k = keys[i];
            r.w = static_cast<__int128>(i) * 1000003;
            return r;
        };
        const AggCase c{{0}, std::vector<std::pair<Func, int>>(130, {Func::SUM, 4}), "wide groups"};
        const AggData data{kRows, make};
        const AggSpillResult want = RunAgg(c, data, Unspilled(1));
        AggSpillConfig cfg = FloorsOnly(1, kWorkerMinimumBytes - kGrantBytes);
        cfg.limit = kWorkerMinimumBytes + 2 * kGrantBytes;
        // Between the pipelines everything but one grant is held: the test's and the kept groups'.
        const auto one_grant_free = [&cfg](MemoryBudget* budget) { EXPECT_EQ(budget->HeldBytes(), *cfg.limit - kGrantBytes); };
        const AggSpillResult got = RunAgg(c, data, cfg, one_grant_free);
        EXPECT_EQ(GroupDifference(got.rows, want.rows), "");
        EXPECT_GT(got.written, 0u);
        EXPECT_EQ(got.files, 2u) << "the sink's file and one split; a forced merge leaves one";
    }

    // a and b share their spill partition at levels 0 to 4, not 5.
    // x1 to x4 leave them one level at a time: every split has two pieces.
    // Their piece merges at level 4 after four splits: five without the depth cap.
    {
        constexpr int64_t a = 38321;
        constexpr int64_t b = 165467;
        const std::array<int64_t, 4> x = {39, 10034, 874565, 3105613};
        for (int level = 0; level <= 4; ++level) ASSERT_EQ(KeyPartition(a, level), KeyPartition(b, level)) << level;
        for (int j = 1; j <= 4; ++j) {
            for (int level = 0; level < j; ++level) ASSERT_EQ(KeyPartition(x[j - 1], level), KeyPartition(a, level)) << j;
            ASSERT_NE(KeyPartition(x[j - 1], j), KeyPartition(a, j)) << j;
        }
        const std::array<int64_t, 6> keys = {a, b, x[0], x[1], x[2], x[3]};
        const AggRowMaker make = [keys](uint32_t i) {
            AggRow r;
            r.k = keys[i % 6];
            r.cents = static_cast<int64_t>(i) * 7 - 9000;
            return r;
        };
        const AggCase c{{0}, {{Func::COUNT, -1}, {Func::SUM, 2}}, "depth cap"};
        const AggData data{6000, make};
        const AggSpillResult want = RunAgg(c, data, Unspilled(1));
        ASSERT_EQ(want.rows.size(), 6u);
        const MergeCapScope cap(1);
        for (const unsigned threads : {1u, kSpillThreads}) {
            const AggSpillResult got = RunAgg(c, data, Everything(threads));
            EXPECT_EQ(GroupDifference(got.rows, want.rows), "") << threads << " threads";
            EXPECT_GE(got.files, 5u) << threads << " threads";
            EXPECT_GT(got.written, 0u) << threads << " threads";
            if (threads == 1) EXPECT_EQ(got.files, 5u) << "the sink's file and four splits";
        }
    }
}

TEST_F(ParallelHashAggregateSpillTest, EmptyInputKeepsTheZeroRow) {
    // No rows: global gives today's zero row, grouped gives none, and nothing spills.
    const std::vector<std::pair<Func, int>> aggs = {{Func::COUNT, -1}, {Func::SUM, 2}, {Func::MIN, 1}, {Func::MAX, 3},
                                                    {Func::AVG, 4},    {Func::MIN, 5}, {Func::SUM, 3}};
    const AggData none{0, [](uint32_t) { return AggRow{}; }};
    for (const AggCase& c : {AggCase{{}, aggs, "global"}, AggCase{{0, 1}, aggs, "grouped"}}) {
        const AggSpillResult want = RunAgg(c, none, Unspilled(1));
        ASSERT_EQ(want.rows.size(), c.keys.empty() ? 1u : 0u) << c.label;
        for (const unsigned threads : {1u, kSpillThreads}) {
            for (const AggSpillConfig& cfg : {Everything(threads), FloorsOnly(threads, kWorkerMinimumBytes), Everything(threads, kWorkerMinimumBytes)}) {
                const AggSpillResult got = RunAgg(c, none, cfg);
                EXPECT_EQ(GroupDifference(got.rows, want.rows), "") << c.label << ", " << threads << " threads";
                EXPECT_EQ(got.files, 0u) << c.label << ", " << threads << " threads";
            }
        }
    }
}

TEST_F(ParallelHashAggregateSpillTest, StringKeysSurvive) {
    // Empty, multibyte and 8,000-byte strings; the input wipes each morsel's bytes.
    const auto note = [](uint32_t i) -> std::string {
        if (i % 997 == 3) return std::string(8000 + i % 50, static_cast<char>('a' + i % 26)) + "-" + std::to_string(i);
        if (i % 11 == 0) return "";
        if (i % 7 == 0) return "ñandú-" + std::to_string(i % 300) + " ünï";
        return "note-" + std::to_string(i % 2000);
    };
    // Hand-fed rows add one 70,000-byte string, past a spill buffer (too long for tuples).
    const AggRowMaker make = [note](uint32_t i) {
        AggRow r;
        r.k = i % 50;
        r.s = i == 777 ? std::string(70000, 'q') : note(i);
        r.cents = static_cast<int64_t>(i % 1000) - 500;
        return r;
    };
    const std::vector<AggCase> cases = {
        {{1}, {{Func::COUNT, -1}, {Func::MIN, 0}, {Func::MAX, 0}}, "by note"},
        {{0}, {{Func::MIN, 1}, {Func::MAX, 1}, {Func::COUNT, -1}}, "min / max note"},
        {{0, 1}, {{Func::COUNT, -1}, {Func::SUM, 2}}, "by k and note"},
        {{}, {{Func::MIN, 1}, {Func::MAX, 1}, {Func::COUNT, -1}}, "global"},
    };
    constexpr uint32_t kRows = 30000;
    for (const bool dictionary : {false, true}) {
        const AggData data{kRows, make, dictionary};
        for (const AggCase& c : cases) {
            const AggSpillResult want = RunAgg(c, data, Unspilled(1));
            for (const unsigned threads : {1u, kSpillThreads}) {
                // Every Consume spills; then one grant per worker, so Finalize spills what's left.
                for (const AggSpillConfig& cfg : {Everything(threads), FloorsOnly(threads, kWorkerMinimumBytes - kGrantBytes)}) {
                    const AggSpillResult got = RunAgg(c, data, cfg);
                    EXPECT_EQ(GroupDifference(got.rows, want.rows), "") << c.label << ", " << threads << " threads, coded " << dictionary;
                    EXPECT_GT(got.files, 0u) << c.label << ", " << threads << " threads, coded " << dictionary;
                    EXPECT_GT(got.written, 0u) << c.label << ", " << threads << " threads, coded " << dictionary;
                }
            }
        }
    }

    // A columnar table with dictionary chunks, through ExecuteSQL, spilled against unspilled.
    std::string text;
    for (uint32_t r = 0; r < kStrRows; ++r) text += std::to_string(r) + "|" + kModes[r % 7] + "|" + note(r) + "|\n";
    Load("str", Schema({Column("id", ColumnType::Integer()), Column("mode", ColumnType::Varchar(10)), Column("note", ColumnType::Text())}), text);
    const ColumnarTable& table = *db_->GetCatalog()->GetTable("str")->columnar;
    ASSERT_EQ(table.RowGroupCount(), 2u);
    ASSERT_EQ(table.ChunkAt(0, 1).encoding, Encoding::DICTIONARY);
    for (const std::string& sql : {"SELECT note, COUNT(*), MIN(id), MAX(mode) FROM str GROUP BY note",
                                   "SELECT mode, MIN(note), MAX(note), COUNT(*) FROM str GROUP BY mode",
                                   "SELECT MIN(note), MAX(note), MIN(mode), MAX(mode) FROM str"}) {
        const QueryResult want = RunWith(*db_, sql, 1, false);
        for (const unsigned threads : {1u, 2u}) {
            EXPECT_EQ(ExactMultisetDifference(RunWith(*db_, sql, threads, true), want), "") << sql << ", " << threads << " threads";
            EXPECT_GT(db_->LastParallelStats().spill_files, 0u) << sql;
            EXPECT_GT(db_->LastParallelStats().spill_bytes_written, 0u) << sql;
            EXPECT_GT(db_->LastParallelStats().spill_partitions, 0u) << sql;
            EXPECT_TRUE(TempDirEmpty()) << sql;
        }
    }
}

TEST_F(ParallelHashAggregateSpillTest, MinMaxCountAvgAllMerge) {
    // Every state kind over every type, with -0.0, NaN and infinities among the doubles.
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const AggRowMaker make = [nan, inf](uint32_t i) {
        AggRow r;
        r.k = static_cast<int64_t>(i % 300) - 150;
        r.s = std::string(kWords[i % 7]) + std::to_string(i % 41);
        r.cents = (static_cast<int64_t>(i) * 7919) % 2000001 - 1000000;
        if (i % 97 == 0) {
            r.f = nan;
        } else if (i % 89 == 0) {
            r.f = -0.0;
        } else if (i % 83 == 0) {
            r.f = 0.0;
        } else if (i % 1009 == 0) {
            r.f = i % 2 ? inf : -inf;
        } else {
            r.f = static_cast<double>(static_cast<int64_t>(i % 20001) - 10000) / 4.0;
        }
        r.w = static_cast<__int128>((static_cast<int64_t>(i) * 104729) % 1000003 - 500000) * 1000000000000LL;
        r.d = 8000 + (i * 37) % 3000;
        return r;
    };
    std::vector<std::pair<Func, int>> aggs = {{Func::COUNT, -1}, {Func::SUM, 0}, {Func::SUM, 2}, {Func::SUM, 4}, {Func::SUM, 3},
                                              {Func::AVG, 0},    {Func::AVG, 2}, {Func::AVG, 4}, {Func::AVG, 3}};
    for (const int slot : {0, 1, 2, 3, 4, 5}) {
        aggs.emplace_back(Func::MIN, slot);
        aggs.emplace_back(Func::MAX, slot);
    }
    const std::vector<AggCase> cases = {
        {{0}, aggs, "grouped"},
        {{}, aggs, "global"},
        {{3}, {{Func::COUNT, -1}, {Func::SUM, 2}, {Func::MIN, 1}, {Func::MAX, 5}}, "by double"},
    };
    const AggData data{12000, make};
    for (const AggCase& c : cases) {
        const AggSpillResult want = RunAgg(c, data, Unspilled(1));
        ASSERT_FALSE(want.rows.empty()) << c.label;
        for (const unsigned threads : {1u, kSpillThreads}) {
            EXPECT_EQ(GroupDifference(RunAgg(c, data, Unspilled(threads)).rows, want.rows), "") << c.label << ", " << threads << " threads";
            for (const AggSpillConfig& cfg : {Everything(threads), FloorsOnly(threads, kWorkerMinimumBytes)}) {
                const AggSpillResult got = RunAgg(c, data, cfg);
                EXPECT_EQ(GroupDifference(got.rows, want.rows), "") << c.label << ", " << threads << " threads";
                EXPECT_GT(got.files, 0u) << c.label << ", " << threads << " threads";
                EXPECT_GT(got.written, 0u) << c.label << ", " << threads << " threads";
            }
        }
    }
}

TEST_F(ParallelHashAggregateSpillTest, CountOffsetAddsOncePerGroupWhenSpilled) {
    // Each count comes from many partial states on disk; the offset comes once per group.
    constexpr uint32_t kRows = 20000;
    constexpr int64_t kOffset = std::numeric_limits<int32_t>::max();
    const AggRowMaker make = [](uint32_t i) {
        AggRow r;
        r.k = i % 4;
        r.cents = i;
        return r;
    };
    const AggData data{kRows, make};
    const AggCase grouped{{0}, {{Func::COUNT, -1}, {Func::COUNT, 2}}, "grouped"};
    const AggCase global{{}, {{Func::COUNT, -1}}, "global"};
    std::vector<OutRow> want_grouped;
    for (int64_t k = 0; k < 4; ++k) {
        OutRow row;
        const int64_t count = kRows / 4 + kOffset;
        AppendRaw(row.exact, &k, sizeof(k));
        AppendRaw(row.exact, &count, sizeof(count));
        AppendRaw(row.exact, &count, sizeof(count));
        want_grouped.push_back(std::move(row));
    }
    std::vector<OutRow> want_global(1);
    const int64_t total = kRows + kOffset;
    ASSERT_GT(total, int64_t{std::numeric_limits<int32_t>::max()});
    AppendRaw(want_global[0].exact, &total, sizeof(total));

    for (const unsigned threads : {1u, kSpillThreads}) {
        for (AggSpillConfig cfg : {Unspilled(threads), Everything(threads)}) {
            cfg.count_offset = kOffset;
            const AggSpillResult g = RunAgg(grouped, data, cfg);
            EXPECT_EQ(GroupDifference(g.rows, want_grouped), "") << threads << " threads, everything " << cfg.spill_everything;
            const AggSpillResult t = RunAgg(global, data, cfg);
            EXPECT_EQ(GroupDifference(t.rows, want_global), "") << threads << " threads, everything " << cfg.spill_everything;
            if (cfg.spill_everything) {
                EXPECT_GT(g.files, 0u);
                EXPECT_GT(g.written, 0u);
                EXPECT_GT(t.files, 0u);
                EXPECT_GT(t.written, 0u);
            }
        }
    }
}

TEST_F(ParallelHashAggregateSpillTest, FailedWriteLeavesNothingBehind) {
    // Every key distinct, so one spill is bigger than the 64 KiB write buffer.
    constexpr uint32_t kRows = 6000;
    const AggRowMaker make = [](uint32_t i) {
        AggRow r;
        r.k = i;
        r.s = "s" + std::to_string(i % 100);
        r.cents = static_cast<int64_t>(i) - 3000;
        return r;
    };
    const AggCase c{{0}, {{Func::COUNT, -1}, {Func::SUM, 2}, {Func::MIN, 1}}, "distinct keys"};
    const AggData data{kRows, make};
    const AggSpillResult want = RunAgg(c, data, Unspilled(1));
    const std::string prefix = "out of temp space in " + temp_dir_ + " while spilling: ";
    for (const unsigned threads : {1u, kSpillThreads}) {
        for (const AggSpillConfig& base : {Everything(threads), FloorsOnly(threads, kWorkerMinimumBytes)}) {
            // Every write fails with ENOSPC; the first fails partway through a spill.
            AggSpillConfig failing = base;
            failing.fail_writes = ENOSPC;
            const AggSpillResult failed = RunAgg(c, data, failing);
            EXPECT_EQ(failed.error, prefix + std::strerror(ENOSPC)) << threads << " threads";
            EXPECT_TRUE(failed.rows.empty()) << threads << " threads";

            // The next aggregate, on fresh objects with writes working, answers.
            const AggSpillResult next = RunAgg(c, data, base);
            EXPECT_EQ(GroupDifference(next.rows, want.rows), "") << threads << " threads";
            EXPECT_GT(next.files, 0u) << threads << " threads";
            EXPECT_GT(next.written, 0u) << threads << " threads";
        }
    }
}

TEST_F(ParallelHashAggregateSpillTest, FailedWriteInFinalizeOrSplitLeavesNothingBehind) {
    const std::string error = "out of temp space in " + temp_dir_ + " while spilling: " + std::strerror(ENOSPC);

    // Each table fits one free grant and there's no shared room, so tables are written
    // only by Finalize's LeaveFloor loop; writes fail from Finalize on.
    {
        const AggRowMaker make = [](uint32_t i) {
            AggRow r;
            r.k = i % 500;
            r.s = kWords[i % 7];
            r.cents = i;
            return r;
        };
        const AggCase c{{0}, {{Func::COUNT, -1}, {Func::SUM, 2}, {Func::MAX, 1}}, "finalize"};
        const AggData data{4000, make};
        const AggSpillResult want = RunAgg(c, data, Unspilled(1));
        for (const unsigned threads : {1u, kSpillThreads}) {
            AggSpillConfig cfg = FloorsOnly(threads, kWorkerMinimumBytes - kGrantBytes);
            const AggSpillResult spilled = RunAgg(c, data, cfg);
            EXPECT_EQ(GroupDifference(spilled.rows, want.rows), "") << threads << " threads";
            EXPECT_GT(spilled.files, 0u) << threads << " threads";
            EXPECT_GT(spilled.written, 0u) << threads << " threads";

            std::optional<FailWritesScope> fail;
            cfg.expect_error = true;
            const AggSpillResult failed = RunAgg(c, data, cfg, {}, [&fail] { fail.emplace(ENOSPC); });
            fail.reset();
            EXPECT_EQ(failed.error, error) << threads << " threads";
            EXPECT_TRUE(failed.rows.empty()) << threads << " threads";
        }
    }

    // Every partition holds about 312 groups and a merge splits past 100;
    // writes fail from between the pipelines on, so the first split's write fails.
    {
        const AggRowMaker make = [](uint32_t i) {
            AggRow r;
            r.k = i % 20000;
            r.cents = i;
            return r;
        };
        const AggCase c{{0}, {{Func::COUNT, -1}, {Func::SUM, 2}}, "split"};
        const AggData data{40000, make};
        const AggSpillResult want = RunAgg(c, data, Unspilled(1));
        const MergeCapScope cap(100);
        for (const unsigned threads : {1u, kSpillThreads}) {
            const AggSpillResult split = RunAgg(c, data, Everything(threads));
            EXPECT_EQ(GroupDifference(split.rows, want.rows), "") << threads << " threads";
            EXPECT_GT(split.files, threads) << threads << " threads";
            EXPECT_GT(split.written, 0u) << threads << " threads";

            std::optional<FailWritesScope> fail;
            AggSpillConfig cfg = Everything(threads);
            cfg.expect_error = true;
            const AggSpillResult failed = RunAgg(c, data, cfg, [&fail](MemoryBudget*) { fail.emplace(ENOSPC); });
            fail.reset();
            EXPECT_EQ(failed.error, error) << threads << " threads";
        }
    }
}

TEST_F(ParallelHashAggregateSpillTest, DamagedSpillIsRefused) {
    // A byte of every spill file's first block is flipped between the pipelines.
    const AggRowMaker make = [](uint32_t i) {
        AggRow r;
        r.k = i % 1000;
        r.cents = i;
        return r;
    };
    const AggCase c{{0}, {{Func::COUNT, -1}, {Func::SUM, 2}}, "damaged"};
    const AggData data{6000, make};
    for (const unsigned threads : {1u, kSpillThreads}) {
        SecondDescriptors second;
        AggSpillConfig cfg = Everything(threads);
        cfg.expect_error = true;
        size_t flipped = 0;
        const AggSpillResult got = RunAgg(c, data, cfg, [&](MemoryBudget*) {
            flipped = second.FlipByte(sizeof(TempBlockHeader) + 4);  // inside the first block's payload
            second.Close();
        });
        EXPECT_GT(flipped, 0u) << threads << " threads";
        EXPECT_EQ(got.error, "damaged temp file: checksum mismatch") << threads << " threads";
    }
}

TEST_F(ParallelHashAggregateSpillTest, GroupsStayInMemoryOffTheFloor) {
    // Each worker's groups fit one floor grant; the shared pool has a grant per worker.
    const AggRowMaker make = [](uint32_t i) {
        AggRow r;
        r.k = i % 500;
        r.s = kWords[i % 7];
        r.cents = i;
        return r;
    };
    const AggCase c{{0}, {{Func::COUNT, -1}, {Func::SUM, 2}, {Func::MAX, 1}}, "in memory"};
    const AggData data{4000, make};
    const AggSpillResult want = RunAgg(c, data, Unspilled(1));
    for (const unsigned threads : {1u, kSpillThreads}) {
        AggSpillConfig roomy;
        roomy.threads = threads;
        roomy.limit = threads * (kWorkerMinimumBytes + kGrantBytes);
        // Between the pipelines the groups are held, moved off the floors into the shared pool.
        bool checked = false;
        const AggSpillResult got = RunAgg(c, data, roomy, [&](MemoryBudget* budget) {
            ASSERT_NE(budget, nullptr);
            checked = true;
            EXPECT_GT(budget->HeldBytes(), 0u) << threads << " threads";
            EXPECT_GT(budget->PeakBytes(), threads * kWorkerMinimumBytes) << threads << " threads";
        });
        EXPECT_TRUE(checked);
        EXPECT_EQ(GroupDifference(got.rows, want.rows), "") << threads << " threads";
        EXPECT_EQ(got.files, 0u) << threads << " threads";
    }
}

TEST_F(ParallelHashAggregateSpillTest, TpchQueriesMatchSpilled) {
    const std::string dir = FindTpchDir();
    if (dir.empty()) GTEST_SKIP() << "TPC-H SF=0.01 data not found. Run bench/tpch/generate_tpch_data.sh first.";
    const std::string file = TempPath("_tpch_spill.db");
    std::filesystem::remove(file);
    extra_files_.push_back(file);
    Database tpch(file, 4096);
    tpch.SetTempDirectory(temp_dir_);
    LoadOptions options;
    options.storage = StorageKind::COLUMNAR;
    for (const auto& td : tpch::TableDescs()) {
        if (td.name == "lineitem" || td.name == "part") tpch.LoadTable(td.name, td.schema, dir + td.tbl_file, options);
    }
    // Q1 groups; Q6, Q14 and Q19 are global, their one group merged from disk.
    for (const std::string q : {"Q1", "Q6", "Q14", "Q19"}) {
        const std::string sql = tpch::FindQuery(q)->sql;
        const QueryResult want = RunWith(tpch, sql, 1, false);
        ASSERT_FALSE(want.tuples.empty()) << q;
        for (const unsigned threads : {1u, kSpillThreads}) {
            EXPECT_EQ(ExactOrderedDifference(RunWith(tpch, sql, threads, true), want), "") << q << ", " << threads << " threads";
            EXPECT_GT(tpch.LastParallelStats().spill_files, 0u) << q;
            EXPECT_GT(tpch.LastParallelStats().spill_bytes_written, 0u) << q;
            EXPECT_GT(tpch.LastParallelStats().spill_partitions, 0u) << q;
            EXPECT_TRUE(TempDirEmpty()) << q;
        }
    }
}

TEST_F(ParallelHashAggregateSpillTest, TableStaysCountedWhileWritten) {
    // One worker, two grants of shared room: its table fills all four grants, then spills.
    // That spill's file opens while the table is being written, and it's still counted.
    constexpr uint32_t kRows = 350000;
    const AggRowMaker make = [](uint32_t i) { return KeyRow(i, 0); };
    const AggCase c{{0}, {{Func::COUNT, -1}, {Func::SUM, 2}}, "distinct keys"};
    const AggData data{kRows, make};
    const AggSpillResult want = RunAgg(c, data, Unspilled(1));
    bool opened = false;
    size_t held_at_open = 0;
    AggSpillConfig cfg = FloorsOnly(1, 0);
    cfg.limit = kWorkerMinimumBytes + 2 * kGrantBytes;
    cfg.on_open = [&](MemoryBudget* budget, const std::string&) {
        if (opened) return;
        opened = true;
        held_at_open = budget->HeldBytes();
    };
    const AggSpillResult got = RunAgg(c, data, cfg);
    EXPECT_EQ(GroupDifference(got.rows, want.rows), "");
    EXPECT_GT(got.files, 0u);
    ASSERT_TRUE(opened);
    // The writer's buffers alone never need more than two grants.
    EXPECT_GT(held_at_open, 2 * kGrantBytes);
}

TEST_F(ParallelHashAggregateSpillTest, InMemoryMergeIsCountedOrSpills) {
    // Four workers with their own keys, each table on one floor grant. At Finalize
    // the tables pool, and the merge is counted in the pool before it's built.
    // 80,000 groups: tables and merge share two grants, so they merge in memory either way.
    // 160,000: the merge grows the pool past four grants, so with room for only four
    // the source merges them, one spill partition per morsel.
    const size_t floors = kSpillThreads * kWorkerMinimumBytes;
    const AggCase c{{0}, {{Func::COUNT, -1}, {Func::SUM, 2}}, "own keys"};
    // 80,000 groups merge into one table; 160,000 merge by partition.
    for (const int64_t per : {20000, 40000}) {
        std::vector<std::vector<AggRow>> parts(kSpillThreads);
        std::set<size_t> partitions;
        for (int64_t w = 0; w < static_cast<int64_t>(kSpillThreads); ++w) {
            for (int64_t j = 0; j < per; ++j) {
                parts[w].push_back(KeyRow(w * per + j, w));
                partitions.insert(KeyPartition(w * per + j, 0));
            }
        }
        const std::vector<OutRow> want = CountSumReference(parts);
        const auto four_grants = [](MemoryBudget& budget) { EXPECT_EQ(budget.HeldBytes(), 4 * kGrantBytes) << "one grant per table"; };

        const bool into_one = per == 20000;

        const DrivenAgg roomy = DriveAgg(c, parts, floors + 8 * kGrantBytes, true, four_grants);
        EXPECT_EQ(GroupDifference(roomy.rows, want), "") << per;
        EXPECT_EQ(roomy.files, 0u) << per;
        if (into_one) {
            EXPECT_EQ(roomy.peak, floors + 2 * kGrantBytes) << per << ": the tables and the merge share two grants";
        } else {
            EXPECT_GT(roomy.peak, floors + 4 * kGrantBytes) << per << ": the merge grew the pool";
        }

        const DrivenAgg tight = DriveAgg(c, parts, floors + 4 * kGrantBytes, true, four_grants);
        EXPECT_EQ(GroupDifference(tight.rows, want), "") << per;
        EXPECT_EQ(tight.files, 0u) << per << ": each spill partition fits a worker's floor";
        if (into_one) {
            EXPECT_EQ(tight.morsels, (kSpillThreads * per + kBatchSize - 1) / kBatchSize) << per << ": merged in memory";
        } else {
            EXPECT_EQ(tight.morsels, partitions.size()) << per << ": merged by spill partition";
        }
    }
}

TEST_F(ParallelHashAggregateSpillTest, FinalizeSpillsTheLargestStuckPartFirst) {
    // Workers 0 to 2 hold 20,000 groups on one floor grant each. Worker 3 holds
    // 340,000 on two floor grants and every shared grant: at Finalize no part can move.
    // Spilling worker 3 first frees the shared room the others need: only it is written.
    // Spilling each part that fails to move would write all four.
    std::vector<std::vector<AggRow>> parts(kSpillThreads);
    for (int64_t w = 0; w < 3; ++w) {
        for (int64_t j = 0; j < 20000; ++j) parts[w].push_back(KeyRow(100000 * w + j, w));
    }
    for (int64_t k = 0; k < 340000; ++k) parts[3].push_back(KeyRow(k, 3));
    const std::vector<OutRow> want = CountSumReference(parts);
    const AggCase c{{0}, {{Func::COUNT, -1}, {Func::SUM, 2}}, "one big part"};
    const size_t limit = kSpillThreads * kWorkerMinimumBytes + 3 * kGrantBytes;
    for (const bool pooled : {false, true}) {
        const DrivenAgg got = DriveAgg(c, parts, limit, pooled, [](MemoryBudget& budget) {
            EXPECT_EQ(budget.HeldBytes(), 8 * kGrantBytes) << "a grant each for workers 0 to 2, five for worker 3";
        });
        EXPECT_EQ(GroupDifference(got.rows, want), "") << "pooled " << pooled;
        EXPECT_EQ(got.files, 1u) << "pooled " << pooled << ": only worker 3's table is written";
    }
}

TEST_F(ParallelHashAggregateSpillTest, FinalizePoolsSmallPartsInsteadOfSpilling) {
    // Each worker groups 2,000 rows into the same 1,000 keys: a table under one grant,
    // on its floor. With one grant of shared room only one table could move alone.
    // Pooled, every table and the merge fit it, so they merge in memory: one batch.
    const AggCase c{{0}, {{Func::COUNT, -1}, {Func::SUM, 2}}, "small parts"};
    for (const unsigned workers : {4u, 18u}) {
        std::vector<std::vector<AggRow>> parts(workers);
        for (unsigned w = 0; w < workers; ++w) {
            for (int64_t j = 0; j < 2000; ++j) parts[w].push_back(KeyRow(j % 1000, w * 2000 + j));
        }
        const std::vector<OutRow> want = CountSumReference(parts);
        const size_t floors = workers * kWorkerMinimumBytes;
        for (const bool pooled : {false, true}) {
            const std::string where = std::to_string(workers) + " workers" + (pooled ? ", on the pool" : ", serial");
            size_t held = 0;
            const DrivenAgg got = DriveAgg(c, parts, floors + kGrantBytes, pooled, {}, [&held](MemoryBudget& b) { held = b.HeldBytes(); });
            EXPECT_EQ(GroupDifference(got.rows, want), "") << where;
            EXPECT_EQ(got.files, 0u) << where;
            EXPECT_EQ(got.morsels, 1u) << where << ": merged in memory, with the merge counted in the pool";
            EXPECT_EQ(held, kGrantBytes) << where << ": the merged groups hold one grant, off every floor";
        }

        // Just the floors: no shared room to pool into, so the tables spill at Finalize.
        const DrivenAgg tight = DriveAgg(c, parts, floors, true);
        EXPECT_EQ(GroupDifference(tight.rows, want), "") << workers << " workers, floors only";
        EXPECT_GT(tight.files, 0u) << workers << " workers, floors only";
    }
}

TEST_F(ParallelHashAggregateSpillTest, PooledTablesTheMergeDoesntFitStayUnmerged) {
    // 18 workers group 3,000 keys of their own each. The tables pool into one grant,
    // but the merge needs a second. With one grant of room the source merges them
    // by spill partition, nothing written; with two, they merge in memory.
    constexpr unsigned kWorkers = 18;
    constexpr int64_t kPer = 3000;
    std::vector<std::vector<AggRow>> parts(kWorkers);
    std::set<size_t> partitions;
    for (int64_t w = 0; w < static_cast<int64_t>(kWorkers); ++w) {
        for (int64_t j = 0; j < kPer; ++j) {
            parts[w].push_back(KeyRow(w * kPer + j, w));
            partitions.insert(KeyPartition(w * kPer + j, 0));
        }
    }
    const std::vector<OutRow> want = CountSumReference(parts);
    const AggCase c{{0}, {{Func::COUNT, -1}, {Func::SUM, 2}}, "own keys"};
    const size_t floors = kWorkers * kWorkerMinimumBytes;

    const DrivenAgg unmerged = DriveAgg(c, parts, floors + kGrantBytes, true);
    EXPECT_EQ(GroupDifference(unmerged.rows, want), "");
    EXPECT_EQ(unmerged.files, 0u) << "pooled, nothing is written";
    EXPECT_EQ(unmerged.morsels, partitions.size()) << "merged by spill partition";

    const DrivenAgg merged = DriveAgg(c, parts, floors + 2 * kGrantBytes, true);
    EXPECT_EQ(GroupDifference(merged.rows, want), "");
    EXPECT_EQ(merged.files, 0u);
    EXPECT_EQ(merged.morsels, (kWorkers * kPer + kBatchSize - 1) / kBatchSize) << "merged in memory";
}

TEST_F(ParallelHashAggregateSpillTest, PoolIsTriedAgainAfterEachSpill) {
    // Workers 0 to 4 hold 20,000 groups on one floor grant each. Worker 5 holds
    // 340,000 on two floor grants and every shared grant, so nothing pools at first.
    // Spilling worker 5 frees three shared grants: three tables move, two stay stuck,
    // and then all five pool.
    constexpr unsigned kWorkers = 6;
    std::vector<std::vector<AggRow>> parts(kWorkers);
    for (int64_t w = 0; w < 5; ++w) {
        for (int64_t j = 0; j < 20000; ++j) parts[w].push_back(KeyRow(100000 * w + j, w));
    }
    for (int64_t k = 0; k < 340000; ++k) parts[5].push_back(KeyRow(k, 5));
    const std::vector<OutRow> want = CountSumReference(parts);
    const AggCase c{{0}, {{Func::COUNT, -1}, {Func::SUM, 2}}, "one big part"};
    const size_t limit = kWorkers * kWorkerMinimumBytes + 3 * kGrantBytes;
    for (const bool pooled : {false, true}) {
        const DrivenAgg got = DriveAgg(c, parts, limit, pooled, [](MemoryBudget& budget) {
            EXPECT_EQ(budget.HeldBytes(), 10 * kGrantBytes) << "a grant each for workers 0 to 4, five for worker 5";
        });
        EXPECT_EQ(GroupDifference(got.rows, want), "") << "pooled " << pooled;
        EXPECT_EQ(got.files, 1u) << "pooled " << pooled << ": only worker 5's table is written";
    }
}

TEST_F(ParallelHashAggregateSpillTest, SplitFilesCloseOnceMerged) {
    // 20,000 keys, about 312 per spill partition, and a merge cap of one group:
    // pieces split down the levels. A split's file closes once its pieces have merged,
    // so a worker holds at most one split file per level.
    const AggRowMaker make = [](uint32_t i) { return KeyRow(i % 20000, i); };
    const AggCase c{{0}, {{Func::COUNT, -1}, {Func::SUM, 2}}, "deep splits"};
    const AggData data{40000, make};
    const AggSpillResult want = RunAgg(c, data, Unspilled(1));
    const MergeCapScope cap(1);
    for (const unsigned threads : {1u, kSpillThreads}) {
        std::mutex mutex;
        size_t most = 0;
        const size_t before = OpenDescriptors();
        AggSpillConfig cfg = Everything(threads);
        cfg.on_open = [&](MemoryBudget*, const std::string&) {
            const std::lock_guard<std::mutex> lock(mutex);
            most = std::max(most, OpenDescriptors());
        };
        const AggSpillResult got = RunAgg(c, data, cfg);
        EXPECT_EQ(GroupDifference(got.rows, want.rows), "") << threads << " threads";
        EXPECT_GT(got.files, threads + kSpillPartitions) << threads << " threads: every partition splits";
        // Each worker's sink file, then per worker a split file per level.
        EXPECT_LE(most, before + threads * (1 + kMaxSpillLevel)) << threads << " threads";
    }
}

TEST_F(ParallelHashAggregateSpillTest, DamagedMergeReadLeavesNothing) {
    // 20,000 keys and a merge cap of one group: every partition splits down the levels.
    // Files are damaged at a later open: the sink's when the first split opens,
    // or a thread's earlier splits when it opens another.
    // With short rows the next read is a merge's. One 70,000-byte MIN(s) is written
    // alone, opening a split's file mid-split, so a split's own next read fails.
    // Its key is in spill partition 0 (the first morsel), and in the last
    // level-1 piece that splits, so that piece splits first.
    constexpr int64_t kKeys = 20000;
    std::map<size_t, std::vector<int64_t>> level1;  // partition 0's keys by level-1 partition
    for (int64_t k = 0; k < kKeys; ++k) {
        if (KeyPartition(k, 0) == 0) level1[KeyPartition(k, 1)].push_back(k);
    }
    auto last = level1.rbegin();
    while (last != level1.rend() && last->second.size() < 2) ++last;
    ASSERT_NE(last, level1.rend());
    const int64_t big_key = last->second.back();
    for (auto it = level1.rbegin(); it != last; ++it) ASSERT_EQ(it->second.size(), 1u) << "a later piece would split first";

    const auto row = [](int64_t i) {
        AggRow r = KeyRow(i % kKeys, i);
        r.s = "s" + std::to_string(i % 7);
        return r;
    };
    const std::vector<AggData> inputs = {
        {2 * kKeys, [row](uint32_t i) { return row(i); }},
        {2 * kKeys, [row, big_key](uint32_t i) {
             AggRow r = row(i);
             if (i == big_key) r.s = std::string(70000, 'q');
             return r;
         }},
    };
    const AggCase c{{0}, {{Func::COUNT, -1}, {Func::SUM, 2}, {Func::MIN, 1}}, "damaged merge"};
    const MergeCapScope cap(1);
    for (size_t input = 0; input < inputs.size(); ++input) {
        const AggData& data = inputs[input];
        const AggSpillResult want = RunAgg(c, data, Unspilled(1));
        for (const unsigned threads : {1u, kSpillThreads}) {
            for (const auto target : {MergeReadDamage::Target::SINK_FILES, MergeReadDamage::Target::SPLIT_FILES}) {
                const std::string label = std::string(input == 0 ? "short rows, " : "one big value, ") + std::to_string(threads) +
                                          " threads, " + (target == MergeReadDamage::Target::SINK_FILES ? "sink files" : "split files");
                MergeReadDamage damage(target);
                AggSpillConfig cfg = Everything(threads);
                cfg.expect_error = true;
                cfg.on_open = [&damage](MemoryBudget*, const std::string& path) { damage.Opened(path); };
                cfg.on_done = [&damage] { damage.Close(); };
                const AggSpillResult failed = RunAgg(c, data, cfg, [&damage](MemoryBudget*) { damage.SinkDone(); });
                EXPECT_GT(damage.Damaged(), 0u) << label;
                EXPECT_EQ(failed.error, "damaged temp file: checksum mismatch") << label;

                // The next aggregate, on fresh objects, answers.
                const AggSpillResult next = RunAgg(c, data, Everything(threads));
                EXPECT_EQ(GroupDifference(next.rows, want.rows), "") << label;
                EXPECT_GT(next.files, 0u) << label;
            }
        }
    }
}

}  // namespace
}  // namespace shilmandb::parallel
