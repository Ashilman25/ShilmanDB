#pragma once

#include "common/exception.hpp"
#include "engine/database.hpp"
#include "parser/parser.hpp"
#include "planner/plan_node.hpp"
#include "planner/planner.hpp"
#include "tpch/tpch_workload.hpp"
#include "types/value.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

// Workloads made of many queries: random constants for the TPC-H queries and a stream of them.
namespace shilmandb::tpch {

// Random numbers

// The same seed gives the same numbers with any standard library, because it only uses the
// engine's output (the std distributions and std::shuffle are free to differ between libraries).
class Rng {
public:
    explicit Rng(uint64_t seed) : engine_(seed) {}

    // A whole number from lo to hi, both included.
    [[nodiscard]] int64_t Uniform(int64_t lo, int64_t hi) {
        assert(lo <= hi);
        const uint64_t span = static_cast<uint64_t>(hi) - static_cast<uint64_t>(lo) + 1;
        if (span == 0) return static_cast<int64_t>(engine_());  // the whole 64-bit range

        // Throw away the few low draws that would make some results more likely than others.
        const uint64_t too_low = (0 - span) % span;
        uint64_t draw = engine_();
        while (draw < too_low) draw = engine_();
        return static_cast<int64_t>(static_cast<uint64_t>(lo) + draw % span);
    }

    // A number from 0 up to (not including) 1.
    [[nodiscard]] double UniformDouble() {
        constexpr double kScale = 1.0 / (uint64_t{1} << 53);
        return static_cast<double>(engine_() >> 11) * kScale;
    }

    template <typename T>
    void Shuffle(std::vector<T>& v) {
        for (size_t i = v.size(); i > 1; --i) {
            std::swap(v[i - 1], v[static_cast<size_t>(Uniform(0, static_cast<int64_t>(i) - 1))]);
        }
    }

private:
    std::mt19937_64 engine_;
};

// Lookups hit a few dates over and over: rank 0 is the most likely, and rank k is drawn with
// weight 1 / (k + 1)^s.
class Zipf {
public:
    Zipf(size_t n, double s) {
        assert(n > 0);
        cumulative_.reserve(n);
        double total = 0;
        for (size_t k = 0; k < n; ++k) {
            total += 1.0 / std::pow(static_cast<double>(k + 1), s);
            cumulative_.push_back(total);
        }
    }

    // A rank from 0 up to (not including) n.
    [[nodiscard]] size_t Draw(Rng& rng) const {
        const double target = rng.UniformDouble() * cumulative_.back();
        const auto it = std::upper_bound(cumulative_.begin(), cumulative_.end(), target);
        return std::min(static_cast<size_t>(it - cumulative_.begin()), cumulative_.size() - 1);
    }

private:
    std::vector<double> cumulative_;  // running total of the weights
};

// Query templates

using Params = std::vector<std::pair<std::string, std::string>>;  // placeholder name -> text

struct QueryTemplate {
    std::string tag;
    std::string sql;  // the query with {name} where a constant goes
};

// Same queries, in the same order, as Queries(), with each constant as a placeholder.
[[nodiscard]] inline auto QueryTemplates() -> std::vector<QueryTemplate> {
    return {
        {"Q1",
         "SELECT l_returnflag, l_linestatus, SUM(l_quantity), SUM(l_extendedprice), "
         "SUM(l_discount), COUNT(*) "
         "FROM lineitem "
         "WHERE l_shipdate <= '{date}' "
         "GROUP BY l_returnflag, l_linestatus "
         "ORDER BY l_returnflag, l_linestatus"},

        {"Q6",
         "SELECT SUM(l_extendedprice * l_discount) "
         "FROM lineitem "
         "WHERE l_shipdate >= '{date}' AND l_shipdate < '{date_end}' "
         "AND l_discount >= {disc_lo} AND l_discount <= {disc_hi} AND l_quantity < {qty}"},

        {"Q3",
         "SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)), o_orderdate, o_shippriority "
         "FROM customer c "
         "JOIN orders o ON c.c_custkey = o.o_custkey "
         "JOIN lineitem l ON l.l_orderkey = o.o_orderkey "
         "WHERE c.c_mktsegment = '{segment}' "
         "AND o.o_orderdate < '{date}' "
         "AND l.l_shipdate > '{date}' "
         "GROUP BY l_orderkey, o_orderdate, o_shippriority "
         "ORDER BY SUM(l_extendedprice * (1 - l_discount)) DESC "
         "LIMIT 10"},

        {"Q5",
         "SELECT n_name, SUM(l_extendedprice * (1 - l_discount)) "
         "FROM customer c "
         "JOIN orders o ON c.c_custkey = o.o_custkey "
         "JOIN lineitem l ON l.l_orderkey = o.o_orderkey "
         "JOIN supplier s ON l.l_suppkey = s.s_suppkey "
         "JOIN nation n ON s.s_nationkey = n.n_nationkey "
         "WHERE o.o_orderdate >= '{date}' AND o.o_orderdate < '{date_end}' "
         "GROUP BY n_name "
         "ORDER BY SUM(l_extendedprice * (1 - l_discount)) DESC"},

        {"Q10",
         "SELECT c_custkey, c_name, SUM(l_extendedprice * (1 - l_discount)), "
         "c_acctbal, n_name, c_address, c_phone, c_comment "
         "FROM customer, orders, lineitem, nation "
         "WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey "
         "AND o_orderdate >= '{date}' AND o_orderdate < '{date_end}' "
         "AND l_returnflag = 'R' AND c_nationkey = n_nationkey "
         "GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment "
         "ORDER BY SUM(l_extendedprice * (1 - l_discount)) DESC "
         "LIMIT 20"},

        {"Q12",
         "SELECT l_shipmode, "
         "SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END), "
         "SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) "
         "FROM orders, lineitem "
         "WHERE o_orderkey = l_orderkey "
         "AND l_shipmode IN ('{mode1}', '{mode2}') "
         "AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate "
         "AND l_receiptdate >= '{date}' AND l_receiptdate < '{date_end}' "
         "GROUP BY l_shipmode "
         "ORDER BY l_shipmode"},

        {"Q14",
         "SELECT 100.00 * SUM(CASE WHEN p_type LIKE 'PROMO%' "
         "THEN l_extendedprice * (1 - l_discount) ELSE 0 END) "
         "/ SUM(l_extendedprice * (1 - l_discount)) "
         "FROM lineitem, part "
         "WHERE l_partkey = p_partkey "
         "AND l_shipdate >= '{date}' AND l_shipdate < '{date_end}'"},

        {"Q19",
         "SELECT SUM(l_extendedprice * (1 - l_discount)) "
         "FROM lineitem, part "
         "WHERE p_partkey = l_partkey "
         "AND ((p_brand = '{brand1}' "
         "AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG') "
         "AND l_quantity >= {q1} AND l_quantity <= {q1_hi} "
         "AND p_size BETWEEN 1 AND 5 "
         "AND l_shipmode IN ('AIR', 'AIR REG') "
         "AND l_shipinstruct = 'DELIVER IN PERSON') "
         "OR (p_brand = '{brand2}' "
         "AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK') "
         "AND l_quantity >= {q2} AND l_quantity <= {q2_hi} "
         "AND p_size BETWEEN 1 AND 10 "
         "AND l_shipmode IN ('AIR', 'AIR REG') "
         "AND l_shipinstruct = 'DELIVER IN PERSON') "
         "OR (p_brand = '{brand3}' "
         "AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG') "
         "AND l_quantity >= {q3} AND l_quantity <= {q3_hi} "
         "AND p_size BETWEEN 1 AND 15 "
         "AND l_shipmode IN ('AIR', 'AIR REG') "
         "AND l_shipinstruct = 'DELIVER IN PERSON'))"},
    };
}

namespace detail {

inline constexpr std::array<const char*, 5> kSegments{"AUTOMOBILE", "BUILDING", "FURNITURE", "HOUSEHOLD", "MACHINERY"};
inline constexpr std::array<const char*, 7> kShipModes{"REG AIR", "AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB"};

// Months are counted from year 0, so a month plus a number of months never needs a calendar.
[[nodiscard]] constexpr int MonthNumber(int year, int month) { return year * 12 + month - 1; }

[[nodiscard]] inline std::string FirstOfMonth(int month_number) {
    char buf[16];
    std::snprintf(buf, sizeof buf, "%04d-%02d-01", month_number / 12, month_number % 12 + 1);
    return buf;
}

// A date moved by a number of days, written the way the engine writes dates.
[[nodiscard]] inline std::string AddDays(const std::string& date, int days) {
    return Value::MakeDate(Value::FromString(TypeId::DATE, date).date_ + days).ToString();
}

// 0.01 to 0.10 from a count of hundredths.
[[nodiscard]] inline std::string Hundredths(int n) {
    return std::string("0.") + (n < 10 ? "0" : "") + std::to_string(n);
}

// [first of the month, first of the month + months)
[[nodiscard]] inline Params MonthWindow(int month_number, int months) {
    return {{"date", FirstOfMonth(month_number)}, {"date_end", FirstOfMonth(month_number + months)}};
}

[[nodiscard]] inline Params YearWindow(int year) { return MonthWindow(MonthNumber(year, 1), 12); }

[[nodiscard]] inline Params Q1Params(int days_back) {
    return {{"date", AddDays("1998-12-01", -days_back)}};
}

[[nodiscard]] inline Params Q3Params(const std::string& segment, int days_into_march) {
    return {{"segment", segment}, {"date", AddDays("1995-03-01", days_into_march)}};
}

// The discount range is the given one-hundredths value, plus and minus 0.01.
[[nodiscard]] inline Params Q6Params(int year, int discount, int quantity) {
    Params p = YearWindow(year);
    p.emplace_back("disc_lo", Hundredths(discount - 1));
    p.emplace_back("disc_hi", Hundredths(discount + 1));
    p.emplace_back("qty", std::to_string(quantity));
    return p;
}

[[nodiscard]] inline Params Q12Params(const std::string& mode1, const std::string& mode2, int year) {
    Params p = YearWindow(year);
    p.emplace_back("mode1", mode1);
    p.emplace_back("mode2", mode2);
    return p;
}

// Each quantity range runs from the given number up by 10.
[[nodiscard]] inline Params Q19Params(const std::string& brand1, const std::string& brand2, const std::string& brand3, int q1, int q2, int q3) {
    return {{"brand1", brand1}, {"brand2", brand2}, {"brand3", brand3},
            {"q1", std::to_string(q1)}, {"q1_hi", std::to_string(q1 + 10)},
            {"q2", std::to_string(q2)}, {"q2_hi", std::to_string(q2 + 10)},
            {"q3", std::to_string(q3)}, {"q3_hi", std::to_string(q3 + 10)}};
}

// A position from 0 up to (not including) count.
[[nodiscard]] inline size_t RandomIndex(Rng& rng, size_t count) {
    return static_cast<size_t>(rng.Uniform(0, static_cast<int64_t>(count) - 1));
}

[[nodiscard]] inline std::string RandomBrand(Rng& rng) {
    const int64_t m = rng.Uniform(1, 5);
    const int64_t n = rng.Uniform(1, 5);
    return "Brand#" + std::to_string(m) + std::to_string(n);
}

}  // namespace detail

// The constants the fixed queries use today.
[[nodiscard]] inline Params DefaultParams(const std::string& tag) {
    using namespace detail;
    if (tag == "Q1") return Q1Params(90);
    if (tag == "Q3") return Q3Params("BUILDING", 14);
    if (tag == "Q5") return YearWindow(1994);
    if (tag == "Q6") return Q6Params(1994, 6, 24);
    if (tag == "Q10") return MonthWindow(MonthNumber(1993, 10), 3);
    if (tag == "Q12") return Q12Params("MAIL", "SHIP", 1994);
    if (tag == "Q14") return MonthWindow(MonthNumber(1995, 9), 1);
    if (tag == "Q19") return Q19Params("Brand#12", "Brand#23", "Brand#34", 1, 10, 20);
    throw DatabaseException("No query template for " + tag);
}

// Random constants for one query. Every draw is its own statement, so the order they are
// made in (and so the result) doesn't depend on the compiler.
[[nodiscard]] inline Params RandomParams(const std::string& tag, Rng& rng) {
    using namespace detail;
    if (tag == "Q1") {
        const int days_back = static_cast<int>(rng.Uniform(60, 120));
        return Q1Params(days_back);
    }
    if (tag == "Q3") {
        const auto segment = kSegments[RandomIndex(rng, kSegments.size())];
        const int days_into_march = static_cast<int>(rng.Uniform(0, 30));
        return Q3Params(segment, days_into_march);
    }
    if (tag == "Q5") {
        const int year = static_cast<int>(rng.Uniform(1993, 1997));
        return YearWindow(year);
    }
    if (tag == "Q6") {
        const int year = static_cast<int>(rng.Uniform(1993, 1997));
        const int discount = static_cast<int>(rng.Uniform(2, 9));
        const int quantity = static_cast<int>(rng.Uniform(24, 25));
        return Q6Params(year, discount, quantity);
    }
    if (tag == "Q10") {
        const int first = MonthNumber(1993, 2);
        const int month = first + static_cast<int>(rng.Uniform(0, MonthNumber(1995, 1) - first));
        return MonthWindow(month, 3);
    }
    if (tag == "Q12") {
        // Two different modes: the second is drawn from the six the first left over.
        const size_t first = RandomIndex(rng, kShipModes.size());
        size_t second = RandomIndex(rng, kShipModes.size() - 1);
        if (second >= first) ++second;
        const int year = static_cast<int>(rng.Uniform(1993, 1997));
        return Q12Params(kShipModes[first], kShipModes[second], year);
    }
    if (tag == "Q14") {
        const int first = MonthNumber(1993, 1);
        const int month = first + static_cast<int>(rng.Uniform(0, MonthNumber(1997, 12) - first));
        return MonthWindow(month, 1);
    }
    if (tag == "Q19") {
        const std::string brand1 = RandomBrand(rng);
        const std::string brand2 = RandomBrand(rng);
        const std::string brand3 = RandomBrand(rng);
        const int q1 = static_cast<int>(rng.Uniform(1, 10));
        const int q2 = static_cast<int>(rng.Uniform(10, 20));
        const int q3 = static_cast<int>(rng.Uniform(20, 30));
        return Q19Params(brand1, brand2, brand3, q1, q2, q3);
    }
    throw DatabaseException("No query template for " + tag);
}

// Puts each param's text where its {name} is. Throws if the SQL has a {name} with no param,
// or a param's {name} is nowhere in the SQL.
[[nodiscard]] inline std::string Render(const std::string& sql, const Params& params) {
    std::string out;
    out.reserve(sql.size());
    std::vector<bool> used(params.size(), false);

    for (size_t pos = 0; pos < sql.size();) {
        if (sql[pos] != '{') {
            out += sql[pos++];
            continue;
        }

        const size_t close = sql.find('}', pos);
        if (close == std::string::npos) throw DatabaseException("Unclosed '{' in query template");

        const std::string name = sql.substr(pos + 1, close - pos - 1);
        const auto it = std::find_if(params.begin(), params.end(), [&](const auto& param) { return param.first == name; });
        if (it == params.end()) throw DatabaseException("No value for {" + name + "} in query template");

        used[static_cast<size_t>(it - params.begin())] = true;
        out += it->second;
        pos = close + 1;
    }

    for (size_t i = 0; i < params.size(); ++i) {
        if (!used[i]) throw DatabaseException("{" + params[i].first + "} is not in the query template");
    }
    return out;
}

// Workloads

enum class Workload { SINGLE, STREAM, LOOKUP, MIXED };

[[nodiscard]] inline std::optional<Workload> ParseWorkload(const std::string& name) {
    if (name == "single") return Workload::SINGLE;
    if (name == "stream") return Workload::STREAM;
    if (name == "lookup") return Workload::LOOKUP;
    if (name == "mixed") return Workload::MIXED;
    return std::nullopt;
}

[[nodiscard]] inline const char* WorkloadName(Workload w) {
    switch (w) {
        case Workload::SINGLE: return "single";
        case Workload::STREAM: return "stream";
        case Workload::LOOKUP: return "lookup";
        case Workload::MIXED: return "mixed";
    }
    return "single";
}

// Lookups after each TPC-H query in the mixed workload.
inline constexpr size_t kMixedLookupsPerQuery = 20;

// How strongly lookups favor a few dates. At 1.0 the busiest 1% of dates get about 45% of them.
inline constexpr double kLookupZipfS = 1.0;

// One query a workload runs.
struct StreamItem {
    enum class Kind { TPCH, LOOKUP };

    Kind kind;
    std::string tag;         // "Q6", "LS", "LO"
    std::string sql;
    std::string index_name;  // lookups only: the index the plan must use
};

// Rounds of the 8 TPC-H queries, each round in its own random order and each query with
// random constants, until there are `queries` of them. The last round may be cut short.
[[nodiscard]] inline std::vector<StreamItem> MakeStream(uint64_t seed, size_t queries) {
    const auto templates = QueryTemplates();
    Rng rng(seed);

    std::vector<size_t> order(templates.size());

    std::vector<StreamItem> items;
    items.reserve(queries);
    while (items.size() < queries) {
        // Each round shuffles the queries from their usual order, not from the last round's.
        std::iota(order.begin(), order.end(), size_t{0});
        rng.Shuffle(order);
        for (const size_t i : order) {
            if (items.size() == queries) break;
            const auto& t = templates[i];
            items.push_back({StreamItem::Kind::TPCH, t.tag, Render(t.sql, RandomParams(t.tag, rng)), {}});
        }
    }
    return items;
}

namespace detail {

// A table the lookups search by date.
struct LookupTable {
    const char* tag;
    const char* index_name;
    const char* sql;            // with {d1} and {d2} where the dates go
    std::vector<int32_t> days;  // every day of the table's date range, shuffled
    Zipf zipf;                  // picks a position in `days`
};

// The days from `first` to `last`, both included, in a random order so that the dates the
// Zipf favors are spread over the whole range and not bunched together.
[[nodiscard]] inline LookupTable MakeLookupTable(Rng& rng, const char* tag, const char* index_name, const char* sql, const char* first, const char* last) {
    const int32_t first_day = Value::FromString(TypeId::DATE, first).date_;
    const int32_t last_day = Value::FromString(TypeId::DATE, last).date_;

    std::vector<int32_t> days(static_cast<size_t>(last_day - first_day) + 1);
    std::iota(days.begin(), days.end(), first_day);
    rng.Shuffle(days);

    Zipf zipf(days.size(), kLookupZipfS);
    return {tag, index_name, sql, std::move(days), std::move(zipf)};
}

[[nodiscard]] inline std::string DayToString(int32_t day) { return Value::MakeDate(day).ToString(); }

}  // namespace detail

// Short date-range lookups on lineitem and orders, half on each. Each starts at a Zipf-picked
// date and covers 1 to 3 days. They have their own random generator, so the same seed gives
// the same lookups whether or not TPC-H queries are drawn alongside.
[[nodiscard]] inline std::vector<StreamItem> MakeLookups(uint64_t seed, size_t lookups) {
    Rng rng(seed ^ 0x9E3779B97F4A7C15ULL);

    const auto lineitem = detail::MakeLookupTable(
        rng, "LS", "idx_lineitem_shipdate",
        "SELECT COUNT(*), SUM(l_extendedprice) FROM lineitem "
        "WHERE l_shipdate >= '{d1}' AND l_shipdate < '{d2}'",
        "1992-01-02", "1998-12-01");
    const auto orders = detail::MakeLookupTable(
        rng, "LO", "idx_orders_orderdate",
        "SELECT COUNT(*), SUM(o_totalprice) FROM orders "
        "WHERE o_orderdate >= '{d1}' AND o_orderdate < '{d2}'",
        "1992-01-01", "1998-08-02");

    std::vector<StreamItem> items;
    items.reserve(lookups);
    for (size_t i = 0; i < lookups; ++i) {
        const bool on_lineitem = rng.Uniform(0, 1) == 0;
        const auto& table = on_lineitem ? lineitem : orders;
        const int32_t start = table.days[table.zipf.Draw(rng)];
        const int32_t width = static_cast<int32_t>(rng.Uniform(1, 3));

        const Params dates{{"d1", detail::DayToString(start)}, {"d2", detail::DayToString(start + width)}};
        items.push_back({StreamItem::Kind::LOOKUP, table.tag, Render(table.sql, dates), table.index_name});
    }
    return items;
}

// The same queries as MakeStream, each followed by its lookups: the first 20 for the first
// query, the next 20 for the second, and so on.
[[nodiscard]] inline std::vector<StreamItem> MakeMixed(uint64_t seed, size_t queries) {
    auto stream = MakeStream(seed, queries);
    auto lookups = MakeLookups(seed, queries * kMixedLookupsPerQuery);

    std::vector<StreamItem> items;
    items.reserve(stream.size() + lookups.size());
    for (size_t i = 0; i < stream.size(); ++i) {
        items.push_back(std::move(stream[i]));
        for (size_t j = 0; j < kMixedLookupsPerQuery; ++j) {
            items.push_back(std::move(lookups[i * kMixedLookupsPerQuery + j]));
        }
    }
    return items;
}

// What a workload runs, `length` items long (TPC-H queries for the mixed one, which adds its
// lookups on top). The single workload runs the fixed queries and has no stream.
[[nodiscard]] inline std::vector<StreamItem> MakeWorkload(Workload w, uint64_t seed, size_t length) {
    switch (w) {
        case Workload::STREAM: return MakeStream(seed, length);
        case Workload::LOOKUP: return MakeLookups(seed, length);
        case Workload::MIXED: return MakeMixed(seed, length);
        case Workload::SINGLE: break;
    }
    throw DatabaseException("The single workload has no stream of queries");
}

namespace detail {

[[nodiscard]] inline bool HasIndexScan(const PlanNode& node, const std::string& index_name) {
    if (node.type == PlanNodeType::INDEX_SCAN &&
        static_cast<const IndexScanPlanNode&>(node).index_name == index_name) {
        return true;
    }
    return std::any_of(node.children.begin(), node.children.end(), [&](const auto& child) { return HasIndexScan(*child, index_name); });
}

}  // namespace detail

// Whether the query's plan reads through this index. It only parses and plans, which uses the
// catalog's statistics and never the buffer pool.
[[nodiscard]] inline bool PlanUsesIndex(const Database& db, const std::string& sql, const std::string& index_name) {
    Parser parser(sql);
    auto stmt = parser.Parse();

    Planner planner(db.GetCatalog());
    planner.SetScanCosts(db.GetScanCosts());
    const auto plan = planner.Plan(std::move(*stmt));
    return detail::HasIndexScan(*plan, index_name);
}

// How many items a workload runs when no length is given. `sf` is any spelling of 0.01, 0.1
// or 1.0; anything else, and the single workload, has no default.
[[nodiscard]] inline std::optional<size_t> DefaultStreamLength(Workload w, const std::string& sf) {
    struct Lengths {
        double sf;
        size_t stream;
        size_t lookup;
        size_t mixed;
    };
    static constexpr std::array<Lengths, 3> kLengths{{
        {0.01, 64, 2000, 32},
        {0.1, 32, 1000, 16},
        {1.0, 16, 500, 8},
    }};

    double value = 0;
    try {
        value = std::stod(sf);
    } catch (const std::exception&) {
        return std::nullopt;
    }

    const auto it = std::find_if(kLengths.begin(), kLengths.end(), [&](const Lengths& l) { return l.sf == value; });
    if (it == kLengths.end()) return std::nullopt;

    switch (w) {
        case Workload::STREAM: return it->stream;
        case Workload::LOOKUP: return it->lookup;
        case Workload::MIXED: return it->mixed;
        case Workload::SINGLE: break;
    }
    return std::nullopt;
}

}  // namespace shilmandb::tpch
