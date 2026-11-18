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
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "catalog/catalog.hpp"
#include "engine/database.hpp"
#include "executor/parallel/batch.hpp"
#include "executor/parallel/decimal.hpp"
#include "executor/parallel/expression.hpp"
#include "executor/parallel/hash_aggregate.hpp"
#include "executor/parallel/pipeline.hpp"
#include "parser/ast.hpp"
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
std::string TypeDifference(const QueryResult& got, const QueryResult& want) {
    if (got.schema.GetColumnCount() != want.schema.GetColumnCount()) return "column counts differ";
    for (uint32_t c = 0; c < got.schema.GetColumnCount(); ++c) {
        if (got.schema.GetColumn(c).type != want.schema.GetColumn(c).type) return "column " + std::to_string(c) + " has another type";
    }
    if (got.tuples.size() != want.tuples.size()) {
        return std::to_string(got.tuples.size()) + " rows, expected " + std::to_string(want.tuples.size());
    }
    return "";
}

// How got and want first differ as multisets of rows, or "" when they match.
std::string Difference(const QueryResult& got, const QueryResult& want) {
    const std::string types = TypeDifference(got, want);
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
std::string OrderedDifference(const QueryResult& got, const QueryResult& want) {
    const std::string types = TypeDifference(got, want);
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
    QueryResult Run(const std::string& sql, ExecutionMode mode, unsigned threads = 0) {
        return RunOn(*db_, sql, mode, threads);
    }

    // Runs sql on db in one mode.
    static QueryResult RunOn(Database& db, const std::string& sql, ExecutionMode mode, unsigned threads = 0) {
        QueryOptions options;
        options.mode = mode;
        options.threads = threads;
        return db.ExecuteSQL(sql, options);
    }

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
    EXPECT_EQ(type(Func::COUNT, -1), kInt);
    EXPECT_EQ(type(Func::COUNT, 5), kInt);
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
    for (const unsigned threads : {0u, 1u, 3u}) {
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
        EXPECT_EQ(t.GetValue(got.schema, 5).integer_, entry.second.count);
    }
    EXPECT_EQ(OrderedDifference(got, RunOn(*tpch, q1, ExecutionMode::VECTORIZED)), "");
    EXPECT_EQ(OrderedDifference(RunOn(*tpch, q1, ExecutionMode::PARALLEL, 1), got), "");
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
            ASSERT_EQ(t.GetValue(got.schema, 1).integer_, want.first) << "key " << k;
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
    for (const unsigned threads : {0u, 1u, 3u}) {
        const QueryResult got = Run(first, ExecutionMode::PARALLEL, threads);
        ASSERT_EQ(got.tuples.size(), 1u);
        EXPECT_EQ(got.schema.GetColumn(0).type, TypeId::INTEGER);
        EXPECT_EQ(got.schema.GetColumn(1).type, TypeId::DECIMAL);
        EXPECT_EQ(got.schema.GetColumn(2).type, TypeId::INTEGER);
        EXPECT_EQ(got.tuples[0].GetValue(got.schema, 0).integer_, 0);
        const double sum = got.tuples[0].GetValue(got.schema, 1).decimal_;
        EXPECT_EQ(sum, 0.0);
        EXPECT_FALSE(std::signbit(sum));
        EXPECT_EQ(got.tuples[0].GetValue(got.schema, 2).integer_, 0);
        EXPECT_EQ(Difference(got, want), "");
    }

    // Types whose empty MIN / MAX today reads back well defined, SUM and AVG of
    // every type (non-numeric ones don't throw without rows), and expressions above.
    for (const std::string& select : {"SELECT MIN(d), MAX(d), MIN(s), MAX(s), MIN(x)", "SELECT AVG(m), SUM(f), AVG(x), SUM(b), AVG(f)",
                                      "SELECT SUM(d), AVG(s), SUM(s), AVG(d)", "SELECT COUNT(*) + 1, SUM(m) * 2, MAX(x) - 3"}) {
        const std::string sql = select + none;
        const QueryResult got = Run(sql, ExecutionMode::PARALLEL);
        ASSERT_EQ(got.tuples.size(), 1u) << sql;
        EXPECT_EQ(Difference(got, Run(sql, ExecutionMode::VECTORIZED)), "") << sql;
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
        EXPECT_EQ(Difference(got, want), "") << sql;
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
        EXPECT_EQ(Difference(Run(sql, ExecutionMode::PARALLEL), want), "") << sql;
        EXPECT_EQ(Difference(Run(sql, ExecutionMode::PARALLEL, 1), want), "") << sql;
    }

    // A sort above the groups, with and without a top n.
    for (const std::string& sql : {"SELECT g, SUM(m), COUNT(*) FROM a GROUP BY g ORDER BY SUM(m) DESC LIMIT 5",
                                   "SELECT s, MIN(d) FROM a GROUP BY s ORDER BY s"}) {
        EXPECT_EQ(OrderedDifference(Run(sql, ExecutionMode::PARALLEL), Run(sql, ExecutionMode::TUPLE)), "") << sql;
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
    for (const std::string& sql : queries) EXPECT_EQ(Difference(Run(sql, ExecutionMode::PARALLEL), Run(sql, ExecutionMode::VECTORIZED)), "") << sql;

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
        s.out_types.push_back(kInt);
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

TEST_F(ParallelHashAggregateTest, CountNarrowsAsToday) {
    // Just past 2^31 rows over two workers, neither past it alone: COUNT wraps as
    // static_cast<int32_t> does today, and an expression above sees the same.
    const InputLayout in = HandLayout();
    AggregateSpec spec;
    spec.funcs = {Func::COUNT, Func::COUNT};
    spec.args.push_back(nullptr);
    spec.args.push_back(Col(0, in));  // COUNT(k) reads no argument either
    spec.out_types = {kInt, kInt};
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
    EXPECT_EQ(o.slots[0].I64()[0], static_cast<int32_t>(rows));
    EXPECT_EQ(o.slots[1].I64()[0], static_cast<int32_t>(rows));
    EXPECT_LT(o.slots[0].I64()[0], 0);
    EXPECT_FALSE(made.second->NextBatch(*local, o));
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
        spec.out_types = {kBig, kBig, kBig, kInt, kBig};
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
        spec.out_types.push_back(kInt);
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
        EXPECT_EQ(row.GetValue(got.schema, 1).integer_, 3);
    }
    EXPECT_TRUE(saw_zero);
    const QueryResult today = Run("SELECT zf, COUNT(*) FROM z GROUP BY zf", ExecutionMode::VECTORIZED);
    for (const Tuple& row : today.tuples) {
        const double k = row.GetValue(today.schema, 0).decimal_;
        if (k == 0.0) EXPECT_FALSE(std::signbit(k)) << "today shows the first zero seen";
    }
}

}  // namespace
}  // namespace shilmandb::parallel
