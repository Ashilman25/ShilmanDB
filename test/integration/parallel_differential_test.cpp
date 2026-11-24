#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "catalog/catalog.hpp"
#include "engine/database.hpp"
#include "executor/parallel/memory_budget.hpp"
#include "integration/parallel_random_queries.hpp"
#include "parser/parser.hpp"
#include "planner/plan_dump.hpp"
#include "planner/planner.hpp"

// Random tables and random queries: the parallel executor against vectorized mode.
// The generator (parallel_random_queries.hpp) stays clear of the known, intended
// differences between the two, so every mismatch it finds is a bug.
namespace shilmandb {
namespace {

using namespace test;

constexpr int kSeeds = 300;
constexpr int kShards = kSeedShards;     // each a gtest of its own, so ctest -j spreads them
constexpr int kQueriesPerSeed = 8;
constexpr double kRelTol = 1e-6;         // SameResult's tolerance for DECIMAL cells
constexpr int kMaxFailures = 20;         // per shard, then stop printing
constexpr size_t kTinyLimit = parallel::kWorkerMinimumBytes;  // the floor of one thread

// ---- Comparing answers ----

// One result cell: DECIMAL as a number, everything else exactly as text.
struct Cell {
    Value value;
    bool decimal{false};
    std::string text;
};
using Row = std::vector<Cell>;

// Where each of the query's output names sits in a schema. Columns keep their
// order, except that SELECT * over a join may list them in another.
std::string ColumnsByName(const GeneratedQuery& q, const Schema& schema, std::vector<uint32_t>& columns) {
    if (schema.GetColumnCount() != q.outputs.size()) return "column counts differ";
    for (size_t k = 0; k < q.outputs.size(); ++k) {
        uint32_t c = 0;
        while (c < schema.GetColumnCount() && schema.GetColumn(c).name != q.outputs[k]) ++c;
        if (c == schema.GetColumnCount()) return "no column named " + q.outputs[k];
        if (!q.star && c != k) return "column " + q.outputs[k] + " moved";
        columns.push_back(c);
    }
    return "";
}

// Every row of a result as cells, its columns taken in this order.
std::vector<Row> RowsOf(const QueryResult& r, const std::vector<uint32_t>& columns) {
    std::vector<Row> rows;
    rows.reserve(r.tuples.size());
    for (const Tuple& t : r.tuples) {
        Row row;
        for (uint32_t c : columns) {
            Cell cell;
            cell.value = t.GetValue(r.schema, c);
            cell.decimal = cell.value.type_ == TypeId::DECIMAL;
            if (!cell.decimal) cell.text = cell.value.ToString();
            row.push_back(std::move(cell));
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

// SameResult's rule; equal infinities count as the same.
bool SameCell(const Cell& a, const Cell& b) {
    if (a.decimal != b.decimal) return false;
    if (!a.decimal) return a.text == b.text;
    const double x = a.value.decimal_;
    const double y = b.value.decimal_;
    return x == y || std::abs(x - y) <= kRelTol * std::max({1.0, std::abs(x), std::abs(y)});
}

// Every cell the same by SameCell.
bool SameRow(const Row& a, const Row& b) {
    if (a.size() != b.size()) return false;
    for (size_t c = 0; c < a.size(); ++c) {
        if (!SameCell(a[c], b[c])) return false;
    }
    return true;
}

// A strict order over rows for lining two results up: text, then numbers.
bool RowLess(const Row& a, const Row& b) {
    for (size_t c = 0; c < a.size(); ++c) {
        if (a[c].decimal) {
            if (a[c].value.decimal_ != b[c].value.decimal_) return a[c].value.decimal_ < b[c].value.decimal_;
        } else if (a[c].text != b[c].text) {
            return a[c].text < b[c].text;
        }
    }
    return false;
}

// A row as text, for failure messages.
std::string Describe(const Row& row) {
    std::string out;
    for (const Cell& c : row) {
        if (c.decimal) {
            char buf[40];
            std::snprintf(buf, sizeof(buf), "%.17g", c.value.decimal_);
            out += buf;
        } else {
            out += "'" + c.text + "'";
        }
        out += " | ";
    }
    return out;
}

// Each row of got matches a different row of want; with all, counts agree too.
// Both are sorted and walked together; rows near-ties put out of step are paired after.
std::string Unmatched(std::vector<Row> got, std::vector<Row> want, bool all) {
    if (all && got.size() != want.size()) return std::to_string(got.size()) + " rows, expected " + std::to_string(want.size());
    if (got.size() > want.size()) return std::to_string(got.size()) + " rows, more than the " + std::to_string(want.size()) + " expected";
    std::sort(got.begin(), got.end(), RowLess);
    std::sort(want.begin(), want.end(), RowLess);
    std::vector<size_t> spare_got;
    std::vector<size_t> spare_want;
    size_t i = 0;
    size_t j = 0;
    while (i < got.size() && j < want.size()) {
        if (SameRow(got[i], want[j])) {
            ++i;
            ++j;
        } else if (RowLess(got[i], want[j])) {
            spare_got.push_back(i++);
        } else {
            spare_want.push_back(j++);
        }
    }
    for (; i < got.size(); ++i) spare_got.push_back(i);
    for (; j < want.size(); ++j) spare_want.push_back(j);
    if (static_cast<double>(spare_got.size()) * static_cast<double>(spare_want.size()) > 5e7) {
        return std::to_string(spare_got.size()) + " rows out of step, first " + Describe(got[spare_got[0]]);
    }
    std::vector<bool> used(spare_want.size(), false);
    for (size_t g : spare_got) {
        bool found = false;
        for (size_t k = 0; k < spare_want.size() && !found; ++k) {
            if (!used[k] && SameRow(got[g], want[spare_want[k]])) found = used[k] = true;
        }
        if (!found) return "unexpected row " + Describe(got[g]);
    }
    if (all) {
        for (size_t k = 0; k < spare_want.size(); ++k) {
            if (!used[k]) return "missing row " + Describe(want[spare_want[k]]);
        }
    }
    return "";
}

// -1, 0 or 1 by the ORDER BY keys, with Value's own comparison.
int CompareKeys(const Row& a, const Row& b, const std::vector<std::pair<size_t, bool>>& keys) {
    for (const auto& [c, ascending] : keys) {
        if (a[c].value < b[c].value) return ascending ? -1 : 1;
        if (b[c].value < a[c].value) return ascending ? 1 : -1;
    }
    return 0;
}

// rows[from, to) as a vector of its own.
std::vector<Row> Slice(const std::vector<Row>& rows, size_t from, size_t to) {
    return std::vector<Row>(rows.begin() + static_cast<std::ptrdiff_t>(from), rows.begin() + static_cast<std::ptrdiff_t>(to));
}

// A LIMIT cuts the near-tie block want[w, end) after take rows. Kept rows
// whose key isn't tied with the last kept key must all be there;
// the rest may be any rows of the block tied with that key.
std::string CutBlock(const std::vector<Row>& got, size_t g, size_t take, const std::vector<Row>& want, size_t w, size_t end, size_t col) {
    const Cell& cut = want[w + take - 1][col];
    std::vector<Row> fixed;
    std::vector<Row> tied;
    for (size_t k = w; k < end; ++k) {
        if (SameCell(want[k][col], cut)) {
            tied.push_back(want[k]);
        } else if (k < w + take) {
            fixed.push_back(want[k]);
        }
    }
    const std::vector<Row> kept = Slice(got, g, g + take);
    std::string why = Unmatched(fixed, kept, false);
    if (!why.empty()) return "a row before the cut is missing (" + why + ")";
    std::vector<Row> allowed = fixed;
    allowed.insert(allowed.end(), tied.begin(), tied.end());
    return Unmatched(kept, allowed, false);
}

// got[g0, g1) must be want[w0, w1)'s first rows, in an order the keys allow.
// Rows tied on key `level` (DECIMALs within the tolerance) form a block.
// A block whose key is exactly equal in both is ordered by the next key;
// any other block may come in any order.
std::string OrderedBlocks(const std::vector<Row>& got, size_t g0, size_t g1, const std::vector<Row>& want, size_t w0, size_t w1,
                          const std::vector<std::pair<size_t, bool>>& keys, size_t level) {
    if (level == keys.size()) return Unmatched(Slice(got, g0, g1), Slice(want, w0, w1), g1 - g0 == w1 - w0);
    const size_t col = keys[level].first;
    size_t w = w0;
    size_t g = g0;
    while (w < w1 && g < g1) {
        size_t end = w + 1;
        while (end < w1 && SameCell(want[end - 1][col], want[end][col])) ++end;
        const size_t take = std::min(end - w, g1 - g);
        // Text keys in a block are equal; decimal ones only maybe.
        bool exact = true;
        if (want[w][col].decimal) {
            for (size_t k = w + 1; k < end && exact; ++k) exact = want[k][col].value.decimal_ == want[w][col].value.decimal_;
            for (size_t k = g; k < g + take && exact; ++k) exact = got[k][col].value.decimal_ == got[g][col].value.decimal_;
        }
        std::string why;
        if (exact) {
            why = OrderedBlocks(got, g, g + take, want, w, end, keys, level + 1);
        } else if (take == end - w) {
            why = Unmatched(Slice(got, g, g + take), Slice(want, w, end), true);
        } else {
            why = CutBlock(got, g, take, want, w, end, col);
        }
        if (!why.empty()) return "rows " + std::to_string(g) + ".." + std::to_string(g + take) + ": " + why;
        w = end;
        g += take;
    }
    return "";
}

// Every tuple's bytes, one string per row.
std::vector<std::string> RowBytes(const QueryResult& r) {
    std::vector<std::string> rows;
    rows.reserve(r.tuples.size());
    for (const Tuple& t : r.tuples) rows.emplace_back(t.GetData(), t.GetLength());
    return rows;
}

// How got differs from want bit for bit, or "": in order when ordered,
// else as a multiset; an unordered LIMIT's rows must all be in want_all.
std::string ExactDifference(const GeneratedQuery& q, const QueryResult& got, const QueryResult& want, const QueryResult& want_all) {
    if (got.schema.GetColumnCount() != want.schema.GetColumnCount()) return "column counts differ";
    for (uint32_t c = 0; c < want.schema.GetColumnCount(); ++c) {
        const Column& a = got.schema.GetColumn(c);
        const Column& b = want.schema.GetColumn(c);
        if (a.name != b.name || a.type != b.type) return "column " + std::to_string(c) + " differs";
    }
    if (got.tuples.size() != want.tuples.size()) return std::to_string(got.tuples.size()) + " rows, expected " + std::to_string(want.tuples.size());
    const std::vector<std::string> g = RowBytes(got);
    if (!q.keys.empty()) {
        const std::vector<std::string> w = RowBytes(want);
        for (size_t i = 0; i < g.size(); ++i) {
            if (g[i] != w[i]) return "row " + std::to_string(i) + " differs in its bits or order";
        }
        return "";
    }
    const std::vector<std::string> w = RowBytes(q.limit >= 0 ? want_all : want);
    std::multiset<std::string> pool(w.begin(), w.end());
    for (const std::string& row : g) {
        auto it = pool.find(row);
        if (it == pool.end()) return "a row with no bit-equal match";
        pool.erase(it);
    }
    return "";
}

// Over empty input today's MIN / MAX of BIGINT or DECIMAL is undefined, so those
// cells are set aside in a row whose COUNT(*) is 0.
void SkipEmptyInputCells(const GeneratedQuery& q, std::vector<Row>& rows) {
    if (q.count_column < 0) return;
    for (Row& row : rows) {
        const Value& count = row[static_cast<size_t>(q.count_column)].value;
        if ((count.type_ == TypeId::BIGINT ? count.bigint_ : count.integer_) != 0) continue;
        for (size_t c : q.empty_unsafe) {
            row[c] = Cell();
            row[c].text = "(empty input)";
        }
    }
}

// sql's COUNT columns in today's plan on db, which parallel plans type BIGINT.
std::vector<bool> CountsOf(const Database& db, const std::string& sql) {
    return test::CountColumns(*db.MakePlanner(QueryOptions{}).Plan(std::move(*Parser(sql).Parse())));
}

// How got differs from want_all, or "" when it may stand in for it.
// With a LIMIT, want_all is vectorized mode's answer without the LIMIT.
// counts marks want_all's COUNT columns: BIGINT in got where want_all has INTEGER.
std::string Difference(const GeneratedQuery& q, const QueryResult& got, const QueryResult& want_all, const std::vector<bool>& counts) {
    std::vector<uint32_t> got_columns;
    std::vector<uint32_t> want_columns;
    std::string why = ColumnsByName(q, got.schema, got_columns);
    if (why.empty()) why = ColumnsByName(q, want_all.schema, want_columns);
    if (!why.empty()) return why;
    for (size_t k = 0; k < got_columns.size(); ++k) {
        const TypeId today = want_all.schema.GetColumn(want_columns[k]).type;
        const bool count = want_columns[k] < counts.size() && counts[want_columns[k]];
        if (count && today != TypeId::INTEGER) return "column " + q.outputs[k] + " is a COUNT but not INTEGER today";
        if (got.schema.GetColumn(got_columns[k]).type != (count ? TypeId::BIGINT : today)) {
            return "column " + q.outputs[k] + " has another type";
        }
    }
    std::vector<Row> g = RowsOf(got, got_columns);
    std::vector<Row> w = RowsOf(want_all, want_columns);
    SkipEmptyInputCells(q, g);
    SkipEmptyInputCells(q, w);
    const size_t expected = q.limit >= 0 ? std::min(w.size(), static_cast<size_t>(q.limit)) : w.size();
    if (g.size() != expected) return std::to_string(g.size()) + " rows, expected " + std::to_string(expected);
    if (q.keys.empty()) return Unmatched(g, w, q.limit < 0);
    for (size_t i = 1; i < g.size(); ++i) {
        if (CompareKeys(g[i - 1], g[i], q.keys) > 0) return "out of order at row " + std::to_string(i) + ": " + Describe(g[i]);
    }
    return OrderedBlocks(g, 0, g.size(), w, 0, w.size(), q.keys, 0);
}

// What a query gave: its rows, or the text it threw.
struct Outcome {
    QueryResult result;
    std::string error;
    bool threw{false};
};

// Runs sql in one mode, catching what it throws; the parallel executor on `threads` threads,
// with this memory limit (unset = the database's default) and spilling everything or not.
Outcome RunQuery(Database& db, const std::string& sql, ExecutionMode mode, unsigned threads = 1,
                 std::optional<size_t> memory_limit = std::nullopt, bool spill_everything = false) {
    Outcome out;
    QueryOptions options;
    options.mode = mode;
    options.threads = threads;
    options.memory_limit = memory_limit;
    options.spill_everything = spill_everything;
    try {
        out.result = db.ExecuteSQL(sql, options);
    } catch (const std::exception& e) {
        out.threw = true;
        out.error = e.what();
    }
    return out;
}

// ---- The test ----

// One shard of the seeds per test, each seed on a fresh file.
class ParallelDifferentialTest : public ::testing::TestWithParam<int> {
protected:
    std::string base_;

    void SetUp() override {
        base_ = (std::filesystem::temp_directory_path() /
                 ("shilmandb_parallel_diff_test_" + std::to_string(::getpid()) + "_" + std::to_string(GetParam())))
                    .string();
    }

    void TearDown() override {
        std::filesystem::remove(base_ + ".db");
        for (const char* t : {"ta", "tb", "tc"}) std::filesystem::remove(base_ + "_" + t + ".tbl");
    }
};

INSTANTIATE_TEST_SUITE_P(Seeds, ParallelDifferentialTest, ::testing::Range(0, kShards));

// Shard k runs every kShards-th seed, starting at k + 1.
TEST_P(ParallelDifferentialTest, RandomQueriesMatchVectorized) {
    std::map<std::string, int> shapes;
    int failures = 0;
    int errors = 0;
    int ordered = 0;
    int unordered_limits = 0;
    int star_joins = 0;
    int empty_unsafe = 0;
    int tiny_runs[2] = {0, 0};     // by spill_everything
    int tiny_spilled[2] = {0, 0};
    int tiny_exact = 0;            // tiny-budget answers compared bit for bit
    for (int seed = 1 + GetParam(); seed <= kSeeds; seed += kShards) {
        Rng rng = SeedRng(seed);
        const std::vector<TableData> tables = SeedTables(rng, seed);

        std::filesystem::remove(base_ + ".db");
        Database db(base_ + ".db", 4096);
        LoadOptions options;
        options.storage = StorageKind::COLUMNAR;
        options.threads = 4;
        for (const TableData& t : tables) {
            const std::string path = base_ + "_" + t.name + ".tbl";
            WriteTable(t, path);
            db.LoadTable(t.name, SchemaOf(t), path, options);
        }

        QueryGen gen(rng, tables);
        for (int n = 0; n < kQueriesPerSeed; ++n) {
            const GeneratedQuery q = gen.Next();
            ++shapes[q.shape];
            if (q.star) ++star_joins;
            if (q.count_column >= 0) ++empty_unsafe;
            const std::string sql = q.Sql();
            const Outcome want = RunQuery(db, sql, ExecutionMode::VECTORIZED);
            const Outcome got = RunQuery(db, sql, ExecutionMode::PARALLEL);
            // The smallest budget one thread allows; every other seed spills everything too.
            const bool spill_all = seed % 2 == 0;
            const Outcome tiny = RunQuery(db, sql, ExecutionMode::PARALLEL, 1, kTinyLimit, spill_all);
            const ParallelQueryStats tiny_stats = db.LastParallelStats();
            ++tiny_runs[spill_all];
            if (!tiny.threw && tiny_stats.spill_files > 0) ++tiny_spilled[spill_all];

            if (want.threw || got.threw) ++errors;
            if (!want.threw && !got.threw) {
                if (!q.keys.empty()) ++ordered;
                if (q.keys.empty() && q.limit >= 0) ++unordered_limits;
            }
            Outcome all;
            if (!want.threw && q.limit >= 0) all = RunQuery(db, q.Sql(false), ExecutionMode::VECTORIZED);
            // How a parallel outcome differs from vectorized mode's, or "".
            const auto differs = [&](const Outcome& out, const std::string& who) -> std::string {
                if (want.threw || out.threw) {
                    if (!want.threw) return who + " threw: " + out.error;
                    if (!out.threw) return who + " didn't throw; vectorized threw: " + want.error;
                    if (out.error != want.error) return who + " threw '" + out.error + "', vectorized threw '" + want.error + "'";
                    return "";
                }
                if (q.limit >= 0) {
                    return all.threw ? "the unlimited query threw: " + all.error : Difference(q, out.result, all.result, CountsOf(db, q.Sql(false)));
                }
                return Difference(q, out.result, want.result, CountsOf(db, sql));
            };
            std::string why = differs(got, "the parallel executor");
            if (why.empty()) why = differs(tiny, "the parallel executor at the tiny budget");
            if (why.empty() && !tiny.threw && tiny_stats.memory_limit != kTinyLimit) {
                why = "the tiny-budget run had a memory limit of " + std::to_string(tiny_stats.memory_limit);
            }
            // Exact types: the tiny budget against the default, both on one thread, bit for bit.
            if (why.empty() && !got.threw && !tiny.threw && !q.uses_double) {
                ++tiny_exact;
                Outcome got_all;
                if (q.keys.empty() && q.limit >= 0) got_all = RunQuery(db, q.Sql(false), ExecutionMode::PARALLEL);
                why = got_all.threw ? "the unlimited parallel query threw: " + got_all.error : ExactDifference(q, tiny.result, got.result, got_all.result);
                if (!why.empty()) why = "the tiny budget against the default one: " + why;
            }
            if (!why.empty()) {
                ++failures;
                if (failures <= kMaxFailures) ADD_FAILURE() << "seed " << seed << ", query " << n << ": " << sql << "\n  " << why;
            }
        }
    }
    EXPECT_EQ(failures, 0);

    // The generator reaches every shape it is meant to.
    for (const char* shape : {"single/select", "single/grouped", "join2/select", "join2/grouped", "join3/select", "loop/select"}) {
        EXPECT_GT(shapes[shape], 0) << shape;
    }
    EXPECT_GT(ordered, 0);
    EXPECT_GT(unordered_limits, 0);
    EXPECT_GT(errors, 0);
    EXPECT_GT(tiny_spilled[1], 0) << "some runs spill everything";
    EXPECT_GT(tiny_exact, 0) << "some tiny-budget answers are compared bit for bit";
    // Counts for the record, read from --gtest_output=xml.
    RecordProperty("errors", errors);
    RecordProperty("ordered", ordered);
    RecordProperty("unordered_limits", unordered_limits);
    RecordProperty("star_joins", star_joins);
    RecordProperty("global_min_max_with_count", empty_unsafe);
    RecordProperty("tiny_runs", tiny_runs[0]);
    RecordProperty("tiny_spilled", tiny_spilled[0]);
    RecordProperty("tiny_everything_runs", tiny_runs[1]);
    RecordProperty("tiny_everything_spilled", tiny_spilled[1]);
    RecordProperty("tiny_exact", tiny_exact);
    for (const auto& [shape, count] : shapes) RecordProperty(shape, count);
}

}  // namespace
}  // namespace shilmandb
