#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "catalog/schema.hpp"
#include "types/column_type.hpp"

// Random tables and random queries for testing the parallel executor.
// The generator stays clear of the known, intended differences between it and
// vectorized mode. Every draw comes from mt19937_64's raw output (fixed by the
// standard) and happens in its own statement, so every platform and compiler
// generates the same queries.
namespace shilmandb::test {

inline constexpr uint64_t kMaxJoinRows = 20000;  // a join's rows before any filter
inline constexpr uint64_t kMaxLoopPairs = 50000;
inline constexpr size_t kGroupRows = 65536;      // rows per row group
inline constexpr int kSeedShards = 15;           // how seeds are dealt out; sets which get a big table

// ---- Small helpers ----

using Rng = std::mt19937_64;

// An integer in [lo, hi]; the tiny modulo bias doesn't matter here.
inline int64_t Between(Rng& rng, int64_t lo, int64_t hi) {
    const uint64_t span = static_cast<uint64_t>(hi - lo) + 1;
    return lo + static_cast<int64_t>(rng() % span);
}

// An index below n (n > 0).
inline size_t Pick(Rng& rng, size_t n) { return static_cast<size_t>(rng() % n); }

// True with probability p, from the top 53 bits of one draw.
inline bool Chance(Rng& rng, double p) { return static_cast<double>(rng() >> 11) * 0x1.0p-53 < p; }

// Fisher-Yates, so the order doesn't depend on the standard library.
template <typename V>
inline void Shuffle(std::vector<V>& v, Rng& rng) {
    for (size_t i = v.size(); i > 1; --i) std::swap(v[i - 1], v[Pick(rng, i)]);
}

// 10^n for small n.
inline int64_t Pow10(int n) {
    int64_t p = 1;
    for (int k = 0; k < n; ++k) p *= 10;
    return p;
}

// units / 10^scale as text, never through a double; scale 0 gets ".0" when asked.
inline std::string Fixed(int64_t units, int scale, bool point_for_whole) {
    const bool negative = units < 0;
    const uint64_t a = negative ? static_cast<uint64_t>(-(units + 1)) + 1 : static_cast<uint64_t>(units);
    const uint64_t p = static_cast<uint64_t>(Pow10(scale));
    std::string out = (negative ? "-" : "") + std::to_string(a / p);
    if (scale > 0) {
        std::string frac = std::to_string(a % p);
        out += "." + std::string(static_cast<size_t>(scale) - frac.size(), '0') + frac;
    } else if (point_for_whole) {
        out += ".0";
    }
    return out;
}

// Days since 1970-01-01 as YYYY-MM-DD (Hinnant's civil_from_days).
inline std::string DateText(int64_t z) {
    z += 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const int64_t doe = z - era * 146097;
    const int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const int64_t mp = (5 * doy + 2) / 153;
    const int64_t d = doy - (153 * mp + 2) / 5 + 1;
    const int64_t m = mp < 10 ? mp + 3 : mp - 9;
    const int64_t y = yoe + era * 400 + (m <= 2 ? 1 : 0);
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%04lld-%02lld-%02lld", static_cast<long long>(y), static_cast<long long>(m), static_cast<long long>(d));
    return buf;
}

inline constexpr int64_t kDays1994 = 8766;  // 1994-01-01

// ---- The tables ----

// How the executor sees a value: today's TypeIds, with DECIMAL split by storage.
enum class T { INT, BIG, NUM, DBL, DATE, STR };

// INTEGER, BIGINT or DECIMAL.
inline bool IsNumeric(T t) { return t == T::INT || t == T::BIG || t == T::NUM || t == T::DBL; }

// The TypeId family two columns must share to be compared or joined.
inline int Family(T t) {
    switch (t) {
        case T::INT:
            return 0;
        case T::BIG:
            return 1;
        case T::NUM:
        case T::DBL:
            return 2;
        case T::DATE:
            return 3;
        case T::STR:
            return 4;
    }
    return -1;
}

struct ColumnInfo {
    std::string name;
    T type{T::INT};
    int scale{0};    // decimals every value has at most (NUM, DBL)
    double mag{0};   // no |value| is larger
    ColumnType storage;
};

struct TableData {
    std::string name;
    size_t rows{0};
    std::vector<ColumnInfo> cols;
    std::vector<std::vector<std::string>> text;  // per column: each field as written
    std::vector<std::vector<int64_t>> num;       // per column: integers, NUMERIC and DOUBLE in millionths, days
};

// The column order every table shares; names get the table's letter in front.
enum ColumnId { kK, kG, kS, kB, kN, kW, kD, kDt, kTs, kV, kTx, kColumnCount };

inline constexpr const char* kWords[] = {"", "alpha", "beta", "Zeta", "\xC3\xA9p\xC3\xA9\x65", "\xC3\xBCn\xC3\xAF", "a%b", "x_y", "Alpha", "b"};
inline constexpr const char* kPieces[] = {"a", "b", "c", "Z", "%", "_", "\xC3\xA9", " "};

// A random short string; at most 12 bytes, so it fits VARCHAR(12) however it's counted.
inline std::string RandomWord(Rng& rng) {
    std::string s;
    const int64_t n = Between(rng, 0, 6);
    for (int64_t k = 0; k < n; ++k) s += kPieces[Pick(rng, 8)];
    return s;
}

// Choices one seed's tables share, so their keys and values meet in joins.
struct SharedShape {
    int64_t big_base{0};     // added to every BIGINT, sometimes past 32 bits
    int64_t date_base{0};    // first day of every table's dates
    int64_t g_domain{1};     // BIGINT values per table when they're few
    bool words{true};        // strings come from the word list
};

// Draws one seed's shared choices.
inline SharedShape MakeShape(Rng& rng) {
    SharedShape shape;
    shape.big_base = Chance(rng, 0.5) ? 5000000000LL : 0;
    shape.date_base = kDays1994 + Between(rng, 0, 1000);
    shape.g_domain = Between(rng, 1, 40);
    shape.words = Chance(rng, 0.7);
    return shape;
}

// One table of random rows in this many row groups, a column per storage type.
inline TableData MakeTable(Rng& rng, char letter, const SharedShape& shape, size_t groups) {
    TableData t;
    t.name = std::string("t") + letter;
    const std::string p = std::string(1, letter) + "_";

    static const int64_t kSmall[] = {6, 150, 2500};
    t.rows = (groups - 1) * kGroupRows + static_cast<size_t>(Between(rng, 1, kSmall[Pick(rng, 3)]));
    const int64_t rows = static_cast<int64_t>(t.rows);

    static const int kNarrow[][2] = {{9, 2}, {12, 4}, {15, 2}, {18, 3}, {5, 0}};
    static const int kWide[][2] = {{25, 3}, {20, 2}, {38, 6}};
    const auto& narrow = kNarrow[Pick(rng, 5)];
    const auto& wide = kWide[Pick(rng, 3)];
    const bool eighths = Chance(rng, 0.5);  // DOUBLE values in eighths, else hundredths

    t.cols = {
        {p + "k", T::INT, 0, 0, ColumnType::Integer()},
        {p + "g", T::BIG, 0, 0, ColumnType::BigInt()},
        {p + "s", T::INT, 0, 30000, ColumnType::SmallInt()},
        {p + "b", T::INT, 0, 1, ColumnType::Boolean()},
        {p + "n", T::NUM, narrow[1], 0, ColumnType::Numeric(narrow[0], narrow[1])},
        {p + "w", T::NUM, wide[1], 1e6, ColumnType::Numeric(wide[0], wide[1])},
        {p + "d", T::DBL, eighths ? 3 : 2, 1e4, ColumnType::Double()},
        {p + "dt", T::DATE, 0, 0, ColumnType::Date()},
        {p + "ts", T::STR, 0, 0, ColumnType::Timestamp()},
        {p + "v", T::STR, 0, 0, ColumnType::Varchar(12)},
        {p + "t", T::STR, 0, 0, ColumnType::Text()},
    };
    t.text.assign(kColumnCount, {});
    t.num.assign(kColumnCount, {});
    for (int c = 0; c < kColumnCount; ++c) {
        t.text[c].reserve(t.rows);
        t.num[c].reserve(t.rows);
    }

    // Per-column shapes: a constant, a few values, or many.
    const int64_t dup = std::vector<int64_t>{1, 2, 3, 8}[Pick(rng, 4)];
    const int64_t key_domain = std::max<int64_t>(1, rows / dup);
    const int64_t big_base = shape.big_base;
    const int64_t g_domain = Chance(rng, 0.6) ? shape.g_domain : key_domain;
    static const int64_t kSmallRanges[] = {0, 3, 100, 30000};
    const int64_t s_range = kSmallRanges[Pick(rng, 4)];
    const double true_share = std::vector<double>{0.0, 0.3, 1.0}[Pick(rng, 3)];
    const int64_t n_max = std::min<int64_t>(Pow10(narrow[0]) - 1, Pow10(4 + narrow[1]));
    const int64_t w_max = Pow10(6 + wide[1]);
    const int64_t d_max = Chance(rng, 0.5) ? 80 : 80000;  // in eighths or hundredths
    const int64_t date_base = shape.date_base + Between(rng, 0, 3);
    const int64_t date_span = std::vector<int64_t>{1, 31, 2500}[Pick(rng, 3)];
    const int64_t ts_span = std::vector<int64_t>{1, 3600, 400000000}[Pick(rng, 3)];
    const bool v_words = shape.words || Chance(rng, 0.2);
    const int t_style = shape.words ? (Chance(rng, 0.8) ? 0 : 2) : static_cast<int>(Pick(rng, 3));  // words, random, long
    const bool n_few = Chance(rng, 0.4);  // whole numbers 0 to 20, which every table can hold
    std::vector<bool> constant(kColumnCount);
    for (int c = 0; c < kColumnCount; ++c) constant[c] = Chance(rng, 0.08);

    t.cols[kK].mag = static_cast<double>(key_domain);
    t.cols[kG].mag = static_cast<double>(big_base + g_domain);
    t.cols[kN].mag = static_cast<double>(n_max) / static_cast<double>(Pow10(narrow[1]));

    // Draws row values; a constant column repeats its first row.
    for (int64_t r = 0; r < rows; ++r) {
        const auto put = [&](int c, std::string text, int64_t num) {
            if (constant[c] && r > 0) {
                t.text[c].push_back(t.text[c][0]);
                t.num[c].push_back(t.num[c][0]);
                return;
            }
            t.text[c].push_back(std::move(text));
            t.num[c].push_back(num);
        };
        const int64_t k = Between(rng, 0, key_domain - 1);
        put(kK, std::to_string(k), k);
        const int64_t g = big_base + Between(rng, 0, g_domain - 1);
        put(kG, std::to_string(g), g);
        const int64_t s = Between(rng, -s_range, s_range);
        put(kS, std::to_string(s), s);
        const bool b = Chance(rng, true_share);
        put(kB, b ? "t" : "f", b ? 1 : 0);
        const int64_t n = n_few ? Between(rng, 0, 20) * Pow10(narrow[1]) : Between(rng, -n_max / 8, n_max);
        put(kN, Fixed(n, narrow[1], false), n * Pow10(6 - narrow[1]));
        const int64_t w = n_few ? Between(rng, 0, 20) * Pow10(wide[1]) : Between(rng, -w_max / 8, w_max);
        put(kW, Fixed(w, wide[1], false), w * Pow10(6 - wide[1]));
        const int64_t d = n_few ? Between(rng, 0, 20) * (eighths ? 8 : 100) : Between(rng, -d_max / 9, d_max);
        if (eighths) {
            put(kD, Fixed(d * 125, 3, false), d * 125000);
        } else {
            put(kD, Fixed(d, 2, false), d * 10000);
        }
        const int64_t day = date_base + Between(rng, 0, date_span - 1);
        put(kDt, DateText(day), day);
        const int64_t secs = Between(rng, 0, ts_span - 1);
        char ts[48];
        std::snprintf(ts, sizeof(ts), "%s %02lld:%02lld:%02lld", DateText(date_base + secs / 86400).c_str(),
                      static_cast<long long>(secs / 3600 % 24), static_cast<long long>(secs / 60 % 60), static_cast<long long>(secs % 60));
        std::string ts_text = ts;
        if (Chance(rng, 0.2)) ts_text += "." + std::to_string(100000 + Between(rng, 0, 899999));
        put(kTs, ts_text, 0);
        put(kV, v_words ? kWords[Pick(rng, 10)] : RandomWord(rng), 0);
        std::string text = t_style == 0 ? kWords[Pick(rng, 10)] : RandomWord(rng);
        if (t_style == 2 && Chance(rng, 0.3)) text = std::string(static_cast<size_t>(Between(rng, 100, 400)), 'q') + text;
        put(kTx, text, 0);
    }
    return t;
}

// The table as a .tbl file; a trailing '|' keeps an empty last field.
inline void WriteTable(const TableData& t, const std::string& path) {
    std::string out;
    out.reserve(t.rows * 120);
    for (size_t r = 0; r < t.rows; ++r) {
        for (int c = 0; c < kColumnCount; ++c) {
            out += t.text[c][r];
            out += '|';
        }
        out += '\n';
    }
    std::ofstream(path, std::ios::binary) << out;
}

// The table's columns with their storage types.
inline Schema SchemaOf(const TableData& t) {
    std::vector<Column> cols;
    for (const auto& c : t.cols) cols.emplace_back(c.name, c.storage);
    return Schema(cols);
}

// ---- The query generator ----

// A generated expression and what's known about its values.
struct Expr {
    std::string sql;
    T type{T::INT};
    int scale{0};       // the true value has at most this many decimals
    double mag{0};      // |value| is at most this
    bool noisy{false};  // decimal arithmetic: today's doubles may differ in the last bits
    int muls{0};        // decimal multiplications inside
    bool empty_unsafe{false};  // a MIN / MAX of BIGINT or DECIMAL: undefined today on empty input
};

// A query, its output column names, and how it orders and limits.
struct GeneratedQuery {
    std::string body;                            // SELECT ... up to ORDER BY
    std::vector<std::string> outputs;            // output column names, in order
    std::vector<std::pair<size_t, bool>> keys;   // ORDER BY: output column, ascending
    int64_t limit{-1};
    std::string shape;                           // for the coverage counts
    bool star{false};                            // SELECT * over a join: columns matched by name
    std::vector<size_t> empty_unsafe;            // outputs not compared when the input is empty
    int count_column{-1};                        // the COUNT(*) output that says so
    bool uses_double{false};                     // some expression reads a DOUBLE value

    // The full text, with or without the LIMIT.
    std::string Sql(bool with_limit = true) const {
        std::string sql = body;
        if (!keys.empty()) {
            sql += " ORDER BY ";
            for (size_t i = 0; i < keys.size(); ++i) {
                sql += (i ? ", " : "") + outputs[keys[i].first] + (keys[i].second ? " ASC" : " DESC");
            }
        }
        if (with_limit && limit >= 0) sql += " LIMIT " + std::to_string(limit);
        return sql;
    }
};

inline constexpr const char* kCmp[] = {"=", "<>", "<", ">", "<=", ">="};

// Builds random queries over the three tables. Every query is well typed and
// avoids what the two executors are known to answer differently.
class QueryGen {
public:
    QueryGen(Rng& rng, const std::vector<TableData>& tables) : rng_(rng), tables_(tables) {}

    // One query: tables, then a shape, then maybe an order and a limit.
    GeneratedQuery Next() {
        GeneratedQuery q;
        double_seen_ = false;
        ordered_ = Chance(rng_, 0.5);
        const bool limited = Chance(rng_, 0.3);
        division_ok_ = ordered_ || !limited;  // an unordered LIMIT may stop before a bad row
        const int roll = static_cast<int>(Pick(rng_, 20));
        std::vector<std::string> where;
        const std::string from = From(where, q.shape);
        if (roll < 9) {
            Select(q, from, where);
        } else if (roll < 17) {
            Grouped(q, from, where);
        } else {
            Global(q, from, where);
        }
        if (ordered_) {
            std::vector<size_t> order(q.outputs.size());
            for (size_t i = 0; i < order.size(); ++i) order[i] = i;
            Shuffle(order, rng_);
            for (size_t i : order) q.keys.emplace_back(i, Chance(rng_, 0.6));
        }
        if (limited) q.limit = std::vector<int64_t>{0, 1, 3, 10, 100, 5000}[Pick(rng_, 6)];
        q.uses_double = double_seen_;
        return q;
    }

private:
    // ---- FROM and joins ----

    // Picks one to three tables and their join conditions; joins stay under the row budgets.
    std::string From(std::vector<std::string>& where, std::string& shape) {
        scope_.clear();
        const int roll = static_cast<int>(Pick(rng_, 20));
        if (roll >= 7) {
            for (int attempt = 0; attempt < 10; ++attempt) {
                std::string from;
                const bool made = roll < 13 ? TwoTables(from, where, shape) : roll < 17 ? ThreeTables(from, where, shape) : LoopJoin(from, where, shape);
                if (made) {
                    return from;
                }
                where.clear();
                scope_.clear();
            }
        }
        const int t = static_cast<int>(Pick(rng_, tables_.size()));
        scope_ = {t};
        shape = "single";
        return tables_[t].name;
    }

    // Rows whose join column has each value, by the value's number or text.
    std::unordered_map<std::string, uint64_t> Counts(int t, int c) const {
        std::unordered_map<std::string, uint64_t> counts;
        for (size_t r = 0; r < tables_[t].rows; ++r) ++counts[KeyOf(t, c, r)];
        return counts;
    }

    // Equal keys give equal strings across tables of one family.
    std::string KeyOf(int t, int c, size_t r) const {
        return tables_[t].cols[c].type == T::STR ? tables_[t].text[c][r] : std::to_string(tables_[t].num[c][r]);
    }

    // Two join columns of one family: INTEGER, BIGINT, DECIMAL, DATE or strings.
    std::pair<int, int> KeyPair() {
        static const int kPairs[][2] = {{kK, kK}, {kK, kK}, {kG, kG}, {kS, kK}, {kN, kN}, {kN, kD}, {kD, kD},
                                        {kW, kN}, {kDt, kDt}, {kV, kV}, {kV, kTx}, {kTx, kTx}, {kB, kK}};
        const auto& p = kPairs[Pick(rng_, 13)];
        return Chance(rng_, 0.5) ? std::make_pair(p[0], p[1]) : std::make_pair(p[1], p[0]);
    }

    // "x = y" between two tables' columns.
    std::string Eq(int t1, int c1, int t2, int c2) const {
        return tables_[t1].cols[c1].name + " = " + tables_[t2].cols[c2].name;
    }

    // Sometimes a second, non-equal condition between two tables rides along. In an ON
    // it makes the join a nested loop, so only small pairs keep it there.
    std::string ExtraCondition(int t1, int t2, std::vector<std::string>& where, bool in_on) {
        if (!Chance(rng_, 0.2)) return "";
        const std::string extra = ColumnPair(t1, t2);
        if (in_on && static_cast<uint64_t>(tables_[t1].rows) * tables_[t2].rows <= kMaxLoopPairs) return " AND " + extra;
        where.push_back(extra);
        return "";
    }

    // Two tables on equal keys, written as JOIN ... ON or as a WHERE condition.
    bool TwoTables(std::string& from, std::vector<std::string>& where, std::string& shape) {
        const int a = static_cast<int>(Pick(rng_, 3));
        const int b = (a + 1 + static_cast<int>(Pick(rng_, 2))) % 3;
        const auto [ca, cb] = KeyPair();
        const auto right = Counts(b, cb);
        uint64_t rows = 0;
        for (size_t r = 0; r < tables_[a].rows; ++r) {
            const auto it = right.find(KeyOf(a, ca, r));
            if (it != right.end()) rows += it->second;
        }
        if (rows == 0 || rows > kMaxJoinRows) return false;
        scope_ = {a, b};
        if (Chance(rng_, 0.5)) {
            from = tables_[a].name + " JOIN " + tables_[b].name + " ON " + Eq(a, ca, b, cb) + ExtraCondition(a, b, where, true);
        } else {
            from = tables_[a].name + ", " + tables_[b].name;
            where.push_back(Eq(a, ca, b, cb));
        }
        shape = "join2";
        return true;
    }

    // A chain: the middle table joins each of the other two.
    bool ThreeTables(std::string& from, std::vector<std::string>& where, std::string& shape) {
        std::vector<int> order = {0, 1, 2};
        Shuffle(order, rng_);
        const int l = order[0];
        const int m = order[1];
        const int r = order[2];
        const auto [cl, cml] = KeyPair();
        const auto [cmr, cr] = KeyPair();
        const auto left = Counts(l, cl);
        const auto right = Counts(r, cr);
        // Every join order starts with l and m or with m and r,
        // so each pair stays in budget too, not only all three.
        uint64_t rows = 0;
        uint64_t with_left = 0;
        uint64_t with_right = 0;
        for (size_t i = 0; i < tables_[m].rows; ++i) {
            const auto x = left.find(KeyOf(m, cml, i));
            const auto y = right.find(KeyOf(m, cmr, i));
            if (x != left.end()) with_left += x->second;
            if (y != right.end()) with_right += y->second;
            if (x != left.end() && y != right.end()) rows += x->second * y->second;
        }
        if (rows == 0 || std::max({rows, with_left, with_right}) > kMaxJoinRows) return false;
        scope_ = {l, m, r};
        if (Chance(rng_, 0.5)) {
            from = tables_[l].name + " JOIN " + tables_[m].name + " ON " + Eq(l, cl, m, cml);
            from += " JOIN " + tables_[r].name + " ON " + Eq(m, cmr, r, cr);
            (void)ExtraCondition(m, r, where, false);
        } else {
            from = tables_[l].name + ", " + tables_[m].name + ", " + tables_[r].name;
            where.push_back(Eq(l, cl, m, cml));
            where.push_back(Eq(m, cmr, r, cr));
        }
        shape = "join3";
        return true;
    }

    // No equal keys: a nested-loop join on a comparison, or a plain cross product.
    bool LoopJoin(std::string& from, std::vector<std::string>& where, std::string& shape) {
        const int a = static_cast<int>(Pick(rng_, 3));
        const int b = (a + 1 + static_cast<int>(Pick(rng_, 2))) % 3;
        if (static_cast<uint64_t>(tables_[a].rows) * tables_[b].rows > kMaxLoopPairs) return false;
        scope_ = {a, b};
        if (Chance(rng_, 0.7)) {
            from = tables_[a].name + " JOIN " + tables_[b].name + " ON " + ColumnPair(a, b);
        } else {
            from = tables_[a].name + ", " + tables_[b].name;
        }
        (void)where;
        shape = "loop";
        return true;
    }

    // A comparison other than = between plain columns of two tables, one family.
    std::string ColumnPair(int t1, int t2) {
        static const int kPairs[][2] = {{kK, kS}, {kN, kW}, {kN, kD}, {kDt, kDt}, {kV, kTx}, {kG, kG}, {kS, kN}};
        const auto& p = kPairs[Pick(rng_, 7)];
        static const char* const kOps[] = {"<", ">", "<=", ">=", "<>"};
        return tables_[t1].cols[p[0]].name + " " + kOps[Pick(rng_, 5)] + " " + tables_[t2].cols[p[1]].name;
    }

    // ---- Columns and literals ----

    // A random (table, column) in scope whose type passes want.
    std::pair<int, int> AnyColumn(bool (*want)(T)) {
        for (;;) {
            const int t = scope_[Pick(rng_, scope_.size())];
            const int c = static_cast<int>(Pick(rng_, kColumnCount));
            if (want(tables_[t].cols[c].type)) return {t, c};
        }
    }

    // Filters for AnyColumn.
    static bool NumericType(T t) { return IsNumeric(t); }
    static bool StringType(T t) { return t == T::STR; }
    static bool DateType(T t) { return t == T::DATE; }
    static bool AnyType(T) { return true; }

    // A plain column, with what its values can be.
    Expr ColumnExpr(int t, int c) const {
        const ColumnInfo& info = tables_[t].cols[c];
        Expr e;
        e.sql = info.name;
        e.type = info.type;
        e.scale = info.scale;
        e.mag = info.mag;
        if (info.type == T::DBL) double_seen_ = true;
        return e;
    }

    // A literal near one of the column's values, as the parser reads it.
    std::string LiteralFor(int t, int c) { return LiteralAt(t, c, Pick(rng_, tables_[t].rows)); }

    // A literal near row r's value.
    std::string LiteralAt(int t, int c, size_t r) {
        const TableData& table = tables_[t];
        const ColumnInfo& info = table.cols[c];
        switch (info.type) {
            case T::INT:
            case T::BIG: {
                const int64_t v = table.num[c][r] + (Chance(rng_, 0.3) ? Between(rng_, -2, 2) : 0);
                const bool small = v > std::numeric_limits<int32_t>::min() && v <= std::numeric_limits<int32_t>::max();
                return small ? std::to_string(v) : std::to_string(v) + ".0";
            }
            case T::NUM:
            case T::DBL: {
                // Millionths, cut to the column's scale or one digit more.
                const int scale = std::min(6, info.scale + static_cast<int>(Pick(rng_, 2)));
                int64_t units = table.num[c][r] / Pow10(6 - scale);
                if (Chance(rng_, 0.3)) units += Between(rng_, -3, 3);
                return Fixed(units, scale, true);
            }
            case T::DATE:
                return "'" + DateText(table.num[c][r] + (Chance(rng_, 0.3) ? Between(rng_, -3, 3) : 0)) + "'";
            case T::STR:
                return "'" + (Chance(rng_, 0.7) ? table.text[c][r] : std::string(kWords[Pick(rng_, 10)])) + "'";
        }
        return "0";
    }

    // "low AND high" from two rows' values, mostly in order.
    std::string Range(int t, int c) {
        size_t a = Pick(rng_, tables_[t].rows);
        size_t b = Pick(rng_, tables_[t].rows);
        const bool strings = tables_[t].cols[c].type == T::STR;
        const bool reversed = strings ? tables_[t].text[c][a] > tables_[t].text[c][b] : tables_[t].num[c][a] > tables_[t].num[c][b];
        if (reversed && Chance(rng_, 0.9)) std::swap(a, b);
        const std::string low = LiteralAt(t, c, a);
        const std::string high = LiteralAt(t, c, b);
        return low + " AND " + high;
    }

    // ---- Numeric expressions ----

    // A numeric column, a small integer, or a short decimal literal.
    Expr NumLeaf() {
        if (Chance(rng_, 0.75)) {
            const auto [t, c] = AnyColumn(NumericType);
            return ColumnExpr(t, c);
        }
        Expr e;
        if (Chance(rng_, 0.5)) {
            const int64_t v = Between(rng_, 0, 20);
            e.sql = std::to_string(v);
            e.mag = static_cast<double>(v);
        } else {
            const int scale = static_cast<int>(Between(rng_, 1, 3));
            const int64_t units = Between(rng_, 0, 10 * Pow10(scale));
            e.sql = Fixed(units, scale, true);
            e.type = T::NUM;
            e.scale = scale;
            e.mag = static_cast<double>(units) / static_cast<double>(Pow10(scale));
        }
        return e;
    }

    // a op b with today's type rules; the smaller side alone when it could overflow.
    Expr Arith(char op, Expr a, Expr b) {
        Expr e;
        e.type = (a.type == T::DBL || b.type == T::DBL) ? T::DBL
                 : (a.type == T::NUM || b.type == T::NUM) ? T::NUM
                 : (a.type == T::BIG || b.type == T::BIG) ? T::BIG
                                                          : T::INT;
        e.scale = op == '*' ? a.scale + b.scale : std::max(a.scale, b.scale);
        e.mag = op == '*' ? a.mag * b.mag : a.mag + b.mag;
        e.muls = a.muls + b.muls + (op == '*' && (e.type == T::NUM || e.type == T::DBL) ? 1 : 0);
        e.noisy = a.noisy || b.noisy || e.type == T::NUM || e.type == T::DBL;
        const bool fits = e.type == T::INT ? e.mag <= 2e9 : e.type == T::BIG ? e.mag <= 9e17 : (e.mag <= 1e11 && e.muls <= 2 && e.scale <= 12);
        if (!fits) return a.mag <= b.mag ? a : b;
        e.sql = "(" + a.sql + " " + op + " " + b.sql + ")";
        return e;
    }

    // A numeric expression: arithmetic, negation, CASE (outside conditions) or a leaf.
    Expr Num(int depth) {
        if (depth <= 0 || Chance(rng_, 0.3)) return NumLeaf();
        const size_t roll = Pick(rng_, 10);
        if (roll < 6) {
            static const char kOps[] = {'+', '-', '*'};
            const char op = kOps[Pick(rng_, 3)];
            Expr left = Num(depth - 1);
            Expr right = Num(depth - 1);
            return Arith(op, std::move(left), std::move(right));
        }
        if (roll == 6) {
            Expr e = Num(depth - 1);
            e.sql = "(-(" + e.sql + "))";
            return e;
        }
        if (roll < 9 && allow_case_) {
            Expr a = Num(depth - 1);
            Expr b = NumOfType(a.type, depth - 1);
            const std::string cond = Bool(1);
            return Case(cond, a, b);
        }
        return NumLeaf();
    }

    // A numeric expression a CASE may pair with this type: the same type, or NUMERIC
    // with DOUBLE (both DECIMAL). INTEGER never pairs with BIGINT or DECIMAL: today
    // keeps each row's own branch type, which gives garbage under one column type.
    Expr NumOfType(T type, int depth) {
        const auto decimal = [](T t) { return t == T::NUM || t == T::DBL; };
        for (int attempt = 0; attempt < 8; ++attempt) {
            Expr e = Num(depth);
            if (e.type == type || (decimal(e.type) && decimal(type))) return e;
        }
        for (;;) {
            const auto [t, c] = AnyColumn(NumericType);
            if (tables_[t].cols[c].type == type) return ColumnExpr(t, c);
        }
    }

    // CASE WHEN cond THEN a ELSE b END; both results share one type.
    Expr Case(const std::string& cond, const Expr& a, const Expr& b) {
        Expr e = a;
        if (b.type == T::DBL) e.type = T::DBL;
        e.sql = "CASE WHEN " + cond + " THEN " + a.sql + " ELSE " + b.sql + " END";
        e.scale = std::max(a.scale, b.scale);
        e.mag = std::max(a.mag, b.mag);
        e.noisy = a.noisy || b.noisy;
        e.muls = std::max(a.muls, b.muls);
        return e;
    }

    // ---- Conditions: never throw, and never tie where exact and double answers differ ----

    // A condition: leaves joined by AND, OR and NOT.
    std::string Bool(int depth) {
        if (depth <= 0 || Chance(rng_, 0.45)) return BoolLeaf();
        const size_t roll = Pick(rng_, 6);
        if (roll == 5) return "(NOT (" + Bool(depth - 1) + "))";  // NOT binds tighter than LIKE and comparisons
        const std::string left = Bool(depth - 1);
        const std::string right = Bool(depth - 1);
        return "(" + left + (roll < 3 ? " AND " : " OR ") + right + ")";
    }

    // "column op literal" with a random comparison.
    std::string CompareToLiteral(int t, int c) {
        const char* op = kCmp[Pick(rng_, 6)];
        const std::string literal = LiteralFor(t, c);
        return tables_[t].cols[c].name + " " + op + " " + literal;
    }

    // One comparison, BETWEEN, IN or LIKE; every side well typed.
    std::string BoolLeaf() {
        const size_t roll = Pick(rng_, 13);
        if (roll < 3) {
            const auto [t, c] = AnyColumn(AnyType);
            return CompareToLiteral(t, c);
        }
        if (roll == 3) {
            const auto [t1, c1] = AnyColumn(AnyType);
            const T first = tables_[t1].cols[c1].type;
            for (int attempt = 0; attempt < 10; ++attempt) {
                const auto [t2, c2] = AnyColumn(AnyType);
                const T second = tables_[t2].cols[c2].type;
                const bool comparable = Family(first) == Family(second) || (IsNumeric(first) && IsNumeric(second));
                if (comparable) return tables_[t1].cols[c1].name + " " + kCmp[Pick(rng_, 6)] + " " + tables_[t2].cols[c2].name;
            }
            return BoolLeaf();
        }
        if (roll < 6) return ExpressionTest();
        if (roll == 6) {
            const auto [t, c] = AnyColumn(AnyType);
            const bool negated = Chance(rng_, 0.2);
            const std::string range = Range(t, c);
            return tables_[t].cols[c].name + (negated ? " NOT" : "") + " BETWEEN " + range;
        }
        if (roll == 7) {
            const auto [t, c] = AnyColumn(AnyType);
            std::string in = tables_[t].cols[c].name + (Chance(rng_, 0.2) ? " NOT IN (" : " IN (");
            const int64_t n = Between(rng_, 1, 4);
            for (int64_t i = 0; i < n; ++i) in += (i ? ", " : "") + LiteralFor(t, c);
            return in + ")";
        }
        if (roll < 10) {
            const auto [t, c] = AnyColumn(StringType);
            const bool negated = Chance(rng_, 0.2);
            const std::string pattern = Pattern();
            return tables_[t].cols[c].name + (negated ? " NOT LIKE '" : " LIKE '") + pattern + "'";
        }
        if (roll == 10) {
            const int t = scope_[Pick(rng_, scope_.size())];
            return tables_[t].cols[kB].name + (Chance(rng_, 0.5) ? " = 1" : " = 0");
        }
        if (roll == 11) {
            const auto [t, c] = AnyColumn(DateType);
            return CompareToLiteral(t, c);
        }
        const auto [t, c] = AnyColumn(StringType);
        return CompareToLiteral(t, c);
    }

    // Arithmetic against a literal. Decimal arithmetic meets a literal one digit finer
    // than it can be, ending in 5, so exact and double answers agree.
    std::string ExpressionTest() {
        for (int attempt = 0; attempt < 5; ++attempt) {
            const Expr e = Num(2);
            if (!e.noisy) {
                const int64_t v = Between(rng_, -20, static_cast<int64_t>(std::min(e.mag, 1e6)) + 1);
                const char* op = kCmp[Pick(rng_, 6)];
                return e.sql + " " + op + " " + std::to_string(v);
            }
            const double grain = static_cast<double>(Pow10(e.scale));
            if (e.scale > 6 || e.mag * grain >= 1e13) continue;
            const int64_t whole = Between(rng_, -10, static_cast<int64_t>(std::min(e.mag, 1e6)) + 1);
            const int64_t fraction = Between(rng_, 0, Pow10(e.scale) - 1);
            const int64_t units = (whole * Pow10(e.scale) + fraction) * 10 + (whole < 0 ? -5 : 5);
            const char* op = kCmp[Pick(rng_, 6)];
            return e.sql + " " + op + " " + Fixed(units, e.scale + 1, true);
        }
        const auto [t, c] = AnyColumn(NumericType);
        return CompareToLiteral(t, c);
    }

    // A LIKE pattern from the words and pieces, with % and _.
    std::string Pattern() {
        switch (Pick(rng_, 6)) {
            case 0:
                return std::string(kWords[Pick(rng_, 10)]);
            case 1:
                return std::string(kPieces[Pick(rng_, 8)]) + "%";
            case 2:
                return "%" + std::string(kPieces[Pick(rng_, 8)]) + "%";
            case 3:
                return "_" + std::string(kPieces[Pick(rng_, 8)]) + "%";
            case 4:
                return "%";
            default:
                return RandomWord(rng_) + "%";
        }
    }

    // ---- Values for the SELECT list ----

    // A string column, sometimes a CASE between two of them.
    Expr StrValue() {
        const auto [t, c] = AnyColumn(StringType);
        Expr e = ColumnExpr(t, c);
        if (Chance(rng_, 0.2)) {
            const auto [t2, c2] = AnyColumn(StringType);
            e.sql = "CASE WHEN " + Bool(1) + " THEN " + e.sql + " ELSE " + tables_[t2].cols[c2].name + " END";
        }
        return e;
    }

    // A date column, sometimes a CASE between two of them.
    Expr DateValue() {
        const auto [t, c] = AnyColumn(DateType);
        Expr e = ColumnExpr(t, c);
        if (Chance(rng_, 0.2)) {
            const auto [t2, c2] = AnyColumn(DateType);
            e.sql = "CASE WHEN " + Bool(1) + " THEN " + e.sql + " ELSE " + tables_[t2].cols[c2].name + " END";
        }
        return e;
    }

    // A column, arithmetic, a 0 or 1 comparison, a string, a date or a division.
    Expr SelectValue() {
        const size_t roll = Pick(rng_, 20);
        if (roll < 5) {
            const auto [t, c] = AnyColumn(AnyType);
            return ColumnExpr(t, c);
        }
        if (roll < 11) return Num(3);
        if (roll < 13) {
            Expr e;
            e.sql = "(" + Bool(1) + ")";
            e.mag = 1;
            return e;
        }
        if (roll < 15) return StrValue();
        if (roll < 17) return DateValue();
        return division_ok_ ? Division() : Num(2);
    }

    // x / column. A plain divisor is zero in both executors at once:
    // an integer zero throws, a decimal one gives infinity, in both.
    Expr Division() {
        Expr num;
        if (Chance(rng_, 0.5)) {
            static const int kIntegers[] = {kK, kS, kB, kG};
            const int t = scope_[Pick(rng_, scope_.size())];
            num = ColumnExpr(t, kIntegers[Pick(rng_, 4)]);
        } else {
            num = Num(2);
        }
        auto [t, c] = AnyColumn(NumericType);
        if (Chance(rng_, 0.6)) c = Chance(rng_, 0.5) ? kS : kB;  // small integers, often zero
        Expr e;
        const T dt = tables_[t].cols[c].type;
        const bool integer = (num.type == T::INT || num.type == T::BIG) && (dt == T::INT || dt == T::BIG);
        e.type = integer ? ((num.type == T::BIG || dt == T::BIG) ? T::BIG : T::INT) : T::DBL;
        if (e.type == T::DBL) double_seen_ = true;
        e.sql = "(" + num.sql + " / " + tables_[t].cols[c].name + ")";
        e.noisy = true;
        e.scale = 99;
        e.mag = std::numeric_limits<double>::infinity();
        return e;
    }

    // ---- Query shapes ----

    // No CASE in a condition: the planner misses a CASE's columns when placing one.
    std::string Where(std::vector<std::string> where) {
        allow_case_ = false;
        if (Chance(rng_, 0.7)) where.push_back(Bool(static_cast<int>(Pick(rng_, 3))));
        if (scope_.size() > 1 && Chance(rng_, 0.25)) {
            const int a = scope_[0];
            const int b = scope_[1];
            where.push_back(ColumnPair(a, b));
        }
        allow_case_ = true;
        std::string sql;
        for (size_t i = 0; i < where.size(); ++i) sql += (i ? " AND " : " WHERE ") + where[i];
        return sql;
    }

    // Plain rows: a few values, or SELECT *. Over a join, * lists columns
    // in the join's order, which parallel planning may choose differently,
    // so those answers are compared with columns matched by name.
    void Select(GeneratedQuery& q, const std::string& from, const std::vector<std::string>& where) {
        q.shape += "/select";
        if (Chance(rng_, 0.1)) {
            q.body = "SELECT * FROM " + from + Where(where);
            for (int t : scope_) {
                for (const auto& c : tables_[t].cols) q.outputs.push_back(c.name);
            }
            q.star = scope_.size() > 1;  // one table keeps its column order
            return;
        }
        const int64_t n = Between(rng_, 1, 5);
        std::string list;
        for (int64_t i = 0; i < n; ++i) {
            const std::string name = "o" + std::to_string(i);
            list += (i ? ", " : "") + SelectValue().sql + " AS " + name;
            q.outputs.push_back(name);
        }
        q.body = "SELECT " + list + " FROM " + from + Where(where);
    }

    // One aggregate and its result type. A MIN / MAX of BIGINT or DECIMAL is
    // marked, since today's empty-input row leaves it undefined.
    Expr AggregateCall() {
        const size_t roll = Pick(rng_, 12);
        Expr out;
        if (roll < 2) {
            out.sql = "COUNT(*)";
            return out;
        }
        if (roll == 2) {
            const auto [t, c] = AnyColumn(AnyType);
            out.sql = "COUNT(" + tables_[t].cols[c].name + ")";
            return out;
        }
        out.type = T::NUM;
        if (roll < 6) {
            out.sql = "SUM(" + Num(2).sql + ")";
            return out;
        }
        if (roll < 8) {
            out.sql = "AVG(" + Num(2).sql + ")";
            return out;
        }
        const char* func = Chance(rng_, 0.5) ? "MIN(" : "MAX(";
        for (;;) {
            Expr e;
            switch (Pick(rng_, 4)) {
                case 0:
                    e = Num(2);
                    break;
                case 1:
                    e = StrValue();
                    break;
                case 2:
                    e = DateValue();
                    break;
                default: {
                    const auto [t, c] = AnyColumn(AnyType);
                    e = ColumnExpr(t, c);
                    break;
                }
            }
            e.sql = func + e.sql + ")";
            e.empty_unsafe = e.type == T::BIG || e.type == T::NUM || e.type == T::DBL;
            return e;
        }
    }

    // GROUP BY one to three plain columns, with aggregates and maybe HAVING.
    void Grouped(GeneratedQuery& q, const std::string& from, const std::vector<std::string>& where) {
        q.shape += "/grouped";
        const int64_t key_count = Between(rng_, 1, 3);
        std::vector<std::pair<int, int>> keys;
        for (int64_t i = 0; i < key_count; ++i) {
            const auto key = AnyColumn(AnyType);
            if (std::find(keys.begin(), keys.end(), key) == keys.end()) keys.push_back(key);
        }
        std::string list;
        size_t out = 0;
        const auto add = [&](const std::string& sql) {
            const std::string name = "o" + std::to_string(out++);
            list += (list.empty() ? "" : ", ") + sql + " AS " + name;
            q.outputs.push_back(name);
        };
        for (const auto& [t, c] : keys) {
            if (Chance(rng_, 0.8)) add(tables_[t].cols[c].name);
        }
        const int64_t aggs = Between(rng_, 1, 4);
        for (int64_t i = 0; i < aggs; ++i) add(AggregateValue().sql);
        std::string group_by;
        for (size_t i = 0; i < keys.size(); ++i) group_by += (i ? ", " : "") + tables_[keys[i].first].cols[keys[i].second].name;
        q.body = "SELECT " + list + " FROM " + from + Where(where) + " GROUP BY " + group_by;
        if (Chance(rng_, 0.3)) q.body += " HAVING " + KeyCondition(keys);
    }

    // An aggregate, sometimes inside arithmetic or a division.
    Expr AggregateValue() {
        const size_t roll = Pick(rng_, 10);
        if (roll == 1 && division_ok_) {
            Expr e;
            const size_t kind = Pick(rng_, 3);
            if (kind == 0) {
                e.sql = "(SUM(" + Num(1).sql + ") / COUNT(*))";
            } else {
                // A boolean's or small integer's MIN is often 0: an integer division by zero.
                const int t = scope_[Pick(rng_, scope_.size())];
                const int c = Chance(rng_, 0.5) ? kB : kS;
                e.sql = "(COUNT(*) / MIN(" + tables_[t].cols[c].name + "))";
            }
            return e;
        }
        Expr agg = AggregateCall();
        if (roll == 0 && IsNumeric(agg.type)) agg.sql = "(" + agg.sql + " + " + std::to_string(Between(rng_, 0, 9)) + ")";
        return agg;
    }

    // HAVING reads only the group keys.
    std::string KeyCondition(const std::vector<std::pair<int, int>>& keys) {
        const auto [t, c] = keys[Pick(rng_, keys.size())];
        const std::string name = tables_[t].cols[c].name;
        switch (Pick(rng_, 3)) {
            case 0:
                return CompareToLiteral(t, c);
            case 1:
                return name + " BETWEEN " + Range(t, c);
            default: {
                const std::string test = CompareToLiteral(t, c);
                const std::string literal = LiteralFor(t, c);
                return "(" + test + " OR " + name + " = " + literal + ")";
            }
        }
    }

    // Aggregates over the whole input: always one row.
    void Global(GeneratedQuery& q, const std::string& from, const std::vector<std::string>& where) {
        q.shape += "/global";
        const int64_t aggs = Between(rng_, 1, 4);
        std::string list;
        for (int64_t i = 0; i < aggs; ++i) {
            const std::string name = "o" + std::to_string(i);
            const Expr agg = AggregateValue();
            if (agg.empty_unsafe) q.empty_unsafe.push_back(static_cast<size_t>(i));
            list += (i ? ", " : "") + agg.sql + " AS " + name;
            q.outputs.push_back(name);
        }
        // A COUNT(*) beside them says whether the input was empty.
        if (!q.empty_unsafe.empty()) {
            q.count_column = static_cast<int>(aggs);
            q.outputs.push_back("o" + std::to_string(aggs));
            list += ", COUNT(*) AS " + q.outputs.back();
        }
        q.body = "SELECT " + list + " FROM " + from + Where(where);
    }

    Rng& rng_;
    const std::vector<TableData>& tables_;
    std::vector<int> scope_;
    bool ordered_{false};
    bool division_ok_{true};
    bool allow_case_{true};
    mutable bool double_seen_{false};  // this query made a DOUBLE expression; never drawn from rng_
};

// One seed's three tables, drawn from rng right after it is seeded.
// Every tenth seed of a shard has one table of two or three row groups,
// so shards take about the same time; the rest keep to one group each.
inline std::vector<TableData> SeedTables(Rng& rng, int seed) {
    const SharedShape shape = MakeShape(rng);
    const int big = (seed - 1) / kSeedShards % 10 == 0 ? static_cast<int>(Pick(rng, 3)) : -1;
    const size_t big_groups = Chance(rng, 1.0 / 3) ? 3 : 2;
    std::vector<TableData> tables;
    for (int i = 0; i < 3; ++i) tables.push_back(MakeTable(rng, static_cast<char>('a' + i), shape, i == big ? big_groups : 1));
    return tables;
}

// The generator's seeding for one seed.
inline Rng SeedRng(int seed) {
    return Rng(static_cast<uint64_t>(seed) * 0x9E3779B97F4A7C15ULL);
}

}  // namespace shilmandb::test
