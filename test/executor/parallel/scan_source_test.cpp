#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "buffer/buffer_pool_manager.hpp"
#include "catalog/catalog.hpp"
#include "engine/database.hpp"
#include "executor/parallel/batch.hpp"
#include "executor/parallel/pipeline.hpp"
#include "executor/parallel/scan_source.hpp"
#include "parser/parser.hpp"
#include "planner/plan_node.hpp"
#include "planner/planner.hpp"
#include "storage/columnar/columnar_table.hpp"
#include "storage/columnar/string_arena.hpp"
#include "tpch/tpch_workload.hpp"

namespace shilmandb::parallel {
namespace {

constexpr uint32_t kRows = 150000;  // three row groups: two full, one of 18,928 rows

// The test table's columns, by number.
constexpr uint32_t kA = 0, kB = 1, kC = 2, kD = 3, kF = 4, kMode = 5, kNote = 6, kTs = 7, kColumns = 8;
const char* const kNames[kColumns] = {"a", "b", "c", "d", "f", "shipmode", "note", "ts"};
const char* const kModes[] = {"AIR", "FOB", "MAIL", "RAIL", "REG AIR", "SHIP", "TRUCK"};

// a is the row number, shipmode repeats seven words, note is unique per row.
Schema TestSchema() {
    return Schema({Column("a", ColumnType::Integer()), Column("b", ColumnType::Numeric(15, 2)),
                   Column("c", ColumnType::Integer()), Column("d", ColumnType::Date()),
                   Column("f", ColumnType::Double()), Column("shipmode", ColumnType::Varchar(10)),
                   Column("note", ColumnType::Text()), Column("ts", ColumnType::Timestamp())});
}

// Row r's b in cents, spread over 0..9,999.99.
uint32_t Cents(uint32_t r) {
    return (r * 7919u) % 1000000u;
}

// Row r's note, some with multibyte text.
std::string Note(uint32_t r) {
    return "note-" + std::to_string(r) + (r % 9 == 0 ? " ünï" : "");
}

// Row r of the table as one line of the load file.
std::string Line(uint32_t r) {
    const uint32_t cents = Cents(r);
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%u|%u.%02u|%u|%04u-%02u-%02u|%.2f|%s|%s|%04u-%02u-%02u %02u:%02u:%02u%s\n",
                  r, cents / 100, cents % 100, r % 1000, 1992 + r % 7, 1 + r % 12, 1 + r % 28, r / 4.0 - 1000.5,
                  kModes[r % 7], Note(r).c_str(), 2000 + r % 20, 1 + r % 12, 1 + r % 28, r % 24, r % 60,
                  (r / 60) % 60, r % 5 == 0 ? ".123456" : "");
    return buf;
}

// Row r of a batch as text: integers raw, doubles as bits, strings as stored.
std::string RowText(const Batch& b, uint32_t r) {
    std::string out;
    for (const Vector& v : b.slots) {
        const VType& t = v.Type();
        if (t.kind == Kind::STRING) {
            out += std::string(View(v.StrAt(r)));
        } else if (t.kind == Kind::DOUBLE) {
            uint64_t bits = 0;
            std::memcpy(&bits, &v.F64()[r], sizeof(bits));
            out += std::to_string(bits);
        } else if (t.kind == Kind::DECIMAL && t.wide) {
            const __int128 x = v.I128()[r];
            out += std::to_string(static_cast<int64_t>(x >> 64)) + ":" + std::to_string(static_cast<uint64_t>(x));
        } else {
            out += std::to_string(v.I64()[r]);
        }
        out += '|';
    }
    return out;
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

// One random conjunct over the test table, as SQL text.
std::string RandomConjunct(std::mt19937& rng) {
    auto pick = [&](uint32_t n) { return static_cast<uint32_t>(rng() % n); };
    char buf[128];
    switch (pick(16)) {
        case 0: return "c < " + std::to_string(pick(1001));
        case 1: return "c >= " + std::to_string(pick(1001));
        case 2: return "a >= " + std::to_string(pick(kRows));
        case 3: return "a < " + std::to_string(pick(kRows));
        case 4:
            std::snprintf(buf, sizeof(buf), "b < %u.%02u", pick(10000), pick(100));
            return buf;
        case 5:
            std::snprintf(buf, sizeof(buf), "b >= %u.%02u", pick(10000), pick(100));
            return buf;
        case 6:
            std::snprintf(buf, sizeof(buf), "d < '%04u-%02u-%02u'", 1992 + pick(7), 1 + pick(12), 1 + pick(28));
            return buf;
        case 7: return "f > " + std::to_string(static_cast<int>(pick(40000)) - 1000) + ".5";
        case 8: return "shipmode IN ('MAIL', 'SHIP')";
        case 9: return std::string("shipmode <> '") + kModes[pick(7)] + "'";
        case 10: return pick(2) == 0 ? "shipmode LIKE '%AI%'" : "shipmode < 'R'";
        case 11: return "note LIKE 'note-" + std::to_string(1 + pick(9)) + "%'";
        case 12:
            std::snprintf(buf, sizeof(buf), "ts < '%04u-%02u-01 00:00:00'", 2000 + pick(20), 1 + pick(12));
            return buf;
        case 13: return "(c < " + std::to_string(pick(200)) + " OR c >= " + std::to_string(800 + pick(200)) + ")";
        case 14: return pick(2) == 0 ? "a + c > " + std::to_string(pick(kRows)) : "c < b";
        default: return "(c < " + std::to_string(pick(300)) + " OR shipmode = 'FOB')";
    }
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

// A planned query whose filter sits right above its table scan.
struct PlannedFilter {
    Database* db{nullptr};
    std::unique_ptr<PlanNode> plan;
    const SeqScanPlanNode* scan{nullptr};
    const Expression* predicate{nullptr};
};

// What one scan source gave when run alone.
struct ScanRun {
    std::vector<std::string> rows;  // RowText of each row, in order
    ScanCounters counters;
    size_t batches{0};
    std::vector<size_t> coded;      // per slot, batches that held it in dictionary form
};

class ParallelScanSourceTest : public ::testing::Test {
protected:
    std::string db_file_;
    std::vector<std::string> extra_files_;
    std::unique_ptr<Database> db_;

    void SetUp() override {
        db_file_ = TempPath(".db");
        std::filesystem::remove(db_file_);

        std::string text;
        text.reserve(static_cast<size_t>(kRows) * 100);
        for (uint32_t r = 0; r < kRows; ++r) text += Line(r);
        const std::string tbl = TempPath("_t.tbl");
        std::ofstream(tbl, std::ios::binary) << text;
        extra_files_.push_back(tbl);

        LoadOptions options;
        options.storage = StorageKind::COLUMNAR;
        options.threads = 2;
        db_ = std::make_unique<Database>(db_file_);
        db_->LoadTable("t", TestSchema(), tbl, options);
        ASSERT_EQ(Table().RowGroupCount(), 3u);
    }

    void TearDown() override {
        db_.reset();
        std::filesystem::remove(db_file_);
        for (const auto& f : extra_files_) std::filesystem::remove(f);
    }

    // A temp path unique to this process.
    static std::string TempPath(const std::string& suffix) {
        return (std::filesystem::temp_directory_path() /
                ("shilmandb_parallel_scan_source_test_" + std::to_string(::getpid()) + suffix)).string();
    }

    // The SF=0.01 TPC-H data, or "" when it hasn't been generated.
    static std::string FindTpchDir() {
        for (const std::string dir : {"bench/tpch_data/sf0.01/", "../bench/tpch_data/sf0.01/", "../../bench/tpch_data/sf0.01/"}) {
            if (std::filesystem::exists(dir + "lineitem.tbl")) return dir;
        }
        return "";
    }

    // The loaded test table.
    const ColumnarTable& Table() const {
        return *db_->GetCatalog()->GetTable("t")->columnar;
    }

    // Pages of the listed columns of t in row groups [from, to).
    uint64_t PagesOf(const std::vector<uint32_t>& columns, size_t from, size_t to) const {
        uint64_t pages = 0;
        for (size_t g = from; g < to; ++g) {
            for (uint32_t c : columns) pages += Table().Meta().row_groups[g].chunks[c].page_count;
        }
        return pages;
    }

    // Runs sql through the whole engine in one mode.
    QueryResult Run(const std::string& sql, ExecutionMode mode) {
        return RunOn(*db_, sql, mode);
    }

    // Runs sql on db in one mode; the parallel executor on one thread.
    static QueryResult RunOn(Database& db, const std::string& sql, ExecutionMode mode) {
        QueryOptions options;
        options.mode = mode;
        options.threads = 1;
        return db.ExecuteSQL(sql, options);
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

    // Plans a query on the test table with this WHERE.
    PlannedFilter PlanFilter(const std::string& where) {
        return PlanFilter(where, *db_, "t");
    }

    // Plans a query on db's table and finds the filter above the scan.
    static PlannedFilter PlanFilter(const std::string& where, Database& db, const std::string& table) {
        auto stmt = Parser("SELECT * FROM " + table + " WHERE " + where).Parse();
        Planner planner(db.GetCatalog());
        PlannedFilter out;
        out.db = &db;
        out.plan = planner.Plan(std::move(*stmt));
        for (const PlanNode* n = out.plan.get(); n != nullptr && !n->children.empty(); n = n->children[0].get()) {
            if (n->type == PlanNodeType::FILTER && n->children[0]->type == PlanNodeType::SEQ_SCAN) {
                out.predicate = static_cast<const FilterPlanNode*>(n)->predicate.get();
                out.scan = static_cast<const SeqScanPlanNode*>(n->children[0].get());
            }
        }
        EXPECT_NE(out.predicate, nullptr) << where;
        return out;
    }

    // A fused scan of the planned table: the plan's filter, these columns handed up.
    static ScanSpec Spec(const PlannedFilter& planned, std::vector<uint32_t> columns, double dense_fraction = kDenseFraction) {
        ScanSpec spec;
        spec.table = planned.scan->table_name;
        spec.columns = std::move(columns);
        spec.dense_fraction = dense_fraction;
        for (const auto& conj : planned.scan->skip_conjuncts) spec.skip_conjuncts.push_back(conj.get());
        AddConjuncts(spec, *planned.db->GetCatalog()->GetTable(spec.table)->columnar, *planned.predicate);
        return spec;
    }

    // Runs one scan source on the test table's database.
    ScanRun Drain(ScanSpec spec) {
        return Drain(std::move(spec), *db_);
    }

    // Runs one scan source alone, morsels in order, and keeps what it gave.
    static ScanRun Drain(ScanSpec spec, Database& db) {
        ScanSource source(std::move(spec));
        QueryContext ctx{db.GetBufferPoolManager(), db.GetCatalog()};
        source.Prepare(ctx);
        auto local = source.MakeLocal();
        Batch batch;
        ScanRun run;
        for (size_t m = 0; m < source.MorselCount(); ++m) {
            source.StartMorsel(m, *local);
            while (source.NextBatch(*local, batch)) {
                // Batches leave compacted: every row active.
                EXPECT_TRUE(batch.sel.IsAll());
                EXPECT_EQ(batch.Active(), batch.rows);
                ++run.batches;
                run.coded.resize(batch.slots.size());
                for (size_t k = 0; k < batch.slots.size(); ++k) {
                    if (batch.slots[k].Type().kind == Kind::STRING && batch.slots[k].IsDictionary()) ++run.coded[k];
                }
                for (uint32_t r = 0; r < batch.rows; ++r) run.rows.push_back(RowText(batch, r));
            }
        }
        run.counters = source.Counters();
        return run;
    }
};

TEST_F(ParallelScanSourceTest, LateMaterializationDecodesOnlySurvivors) {
    // c < 500 keeps half, so b is read whole; both keep about 5%,
    // so note is read only for the rows they kept.
    ScanRun run = Drain(Spec(PlanFilter("c < 500 AND b < 1000.00"), {kNote}));
    std::vector<std::string> want;
    for (uint32_t r = 0; r < kRows; ++r) {
        if (r % 1000 < 500 && Cents(r) < 100000) want.push_back(Note(r) + "|");
    }
    ASSERT_LT(want.size() * 8, static_cast<size_t>(kRows) / 2);
    EXPECT_EQ(run.rows, want);
    ASSERT_EQ(run.counters.rows_decoded.size(), kColumns);
    EXPECT_EQ(run.counters.rows_passed, want.size());
    EXPECT_EQ(run.counters.rows_decoded[kC], kRows);
    EXPECT_EQ(run.counters.rows_decoded[kB], kRows);
    EXPECT_EQ(run.counters.rows_decoded[kNote], want.size());
    for (uint32_t c : {kA, kD, kF, kMode, kTs}) EXPECT_EQ(run.counters.rows_decoded[c], 0u) << kNames[c];

    // c < 50 keeps 5%, so b too is read only for those rows.
    run = Drain(Spec(PlanFilter("c < 50 AND b < 1000.00"), {kNote}));
    want.clear();
    uint64_t kept_by_c = 0;
    for (uint32_t r = 0; r < kRows; ++r) {
        if (r % 1000 >= 50) continue;
        ++kept_by_c;
        if (Cents(r) < 100000) want.push_back(Note(r) + "|");
    }
    EXPECT_EQ(run.rows, want);
    EXPECT_EQ(run.counters.rows_decoded[kC], kRows);
    EXPECT_EQ(run.counters.rows_decoded[kB], kept_by_c);
    EXPECT_EQ(run.counters.rows_decoded[kNote], want.size());

    // Through the engine the filter is fused. The 1,000 survivors open the first group
    // and no zone map sees a + 0, so few note pages are read.
    auto* bpm = db_->GetBufferPoolManager();
    bpm->DropAllPages();
    bpm->ResetStats();
    const std::string sql = "SELECT note FROM t WHERE a + 0 < 1000";
    const QueryResult got = Run(sql, ExecutionMode::PARALLEL);
    EXPECT_EQ(got.tuples.size(), 1000u);
    EXPECT_GE(bpm->GetMissCount(), PagesOf({kA}, 0, 3));
    EXPECT_LE(bpm->GetMissCount(), PagesOf({kA}, 0, 3) + 4);
    ASSERT_GT(PagesOf({kNote}, 0, 1), 40u);
    EXPECT_EQ(Difference(got, Run(sql, ExecutionMode::VECTORIZED)), "");
}

TEST_F(ParallelScanSourceTest, DenseAndSparsePathsAgree) {
    std::mt19937 rng(20261117);
    size_t total = 0;
    bool saw_whole_after_narrowing = false;
    bool saw_by_row = false;
    for (int i = 0; i < 10; ++i) {
        std::string where = RandomConjunct(rng);
        const uint32_t more = rng() % 4;
        for (uint32_t k = 0; k < more; ++k) where += " AND " + RandomConjunct(rng);
        std::vector<uint32_t> columns;
        while (columns.empty()) {
            for (uint32_t c = 0; c < kColumns; ++c) {
                if (rng() % 3 == 0) columns.push_back(c);
            }
        }
        SCOPED_TRACE(where);

        // Whole chunks always, selected rows always, and the usual switch.
        const PlannedFilter planned = PlanFilter(where);
        const ScanRun dense = Drain(Spec(planned, columns, 0.0));
        const ScanRun sparse = Drain(Spec(planned, columns, 1.0));
        const ScanRun usual = Drain(Spec(planned, columns));
        EXPECT_EQ(dense.rows, sparse.rows);
        EXPECT_EQ(dense.rows, usual.rows);
        // Where narrowing came first, the usual switch matches the dense count when it
        // read whole and the sparse count when it read by row.
        for (uint32_t c = 0; c < kColumns; ++c) {
            const uint64_t d = dense.counters.rows_decoded[c];
            const uint64_t s = sparse.counters.rows_decoded[c];
            const uint64_t u = usual.counters.rows_decoded[c];
            EXPECT_LE(s, d) << kNames[c];
            saw_whole_after_narrowing = saw_whole_after_narrowing || (s < d && u == d);
            saw_by_row = saw_by_row || (s < d && u == s);
        }

        // The whole engine gives what today's vectorized mode gives.
        std::string sql = "SELECT ";
        for (size_t k = 0; k < columns.size(); ++k) sql += (k > 0 ? ", " : "") + std::string(kNames[columns[k]]);
        sql += " FROM t WHERE " + where;
        const QueryResult want = Run(sql, ExecutionMode::VECTORIZED);
        EXPECT_EQ(dense.rows.size(), want.tuples.size());
        EXPECT_EQ(Difference(Run(sql, ExecutionMode::PARALLEL), want), "") << sql;
        total += want.tuples.size();
    }
    EXPECT_GT(total, 0u);
    EXPECT_TRUE(saw_whole_after_narrowing);
    EXPECT_TRUE(saw_by_row);
}

TEST_F(ParallelScanSourceTest, MixedWholeAndByRowReads) {
    // Each step's rule per row group: whole above 1/8 of its rows, else by row.
    const ColumnarTable& table = Table();
    struct Case {
        std::string where;
        std::vector<uint32_t> columns;
    };
    // c < 500 keeps half, so the coded shipmode is read whole and picked out;
    // the rest leave under 1/8, so a and note are read by row.
    // c < 50 keeps 5%, so the coded shipmode is read by row.
    const Case cases[] = {{"c < 500 AND shipmode <> 'AIR' AND b < 2000.00", {kA, kMode, kNote}},
                          {"c < 50 AND shipmode <> 'AIR'", {kA, kMode}}};
    for (const Case& k : cases) {
        SCOPED_TRACE(k.where);
        const bool wide = k.columns.size() == 3;
        std::vector<uint64_t> want_decoded(kColumns, 0);
        std::vector<std::string> want_rows;
        uint32_t first = 0;
        for (size_t g = 0; g < table.RowGroupCount(); ++g) {
            const uint32_t rows = table.RowGroupRows(g);
            auto read = [&](uint64_t n) { return static_cast<double>(n) > kDenseFraction * rows ? rows : n; };
            uint64_t after_c = 0;
            uint64_t after_mode = 0;
            uint64_t passed = 0;
            for (uint32_t r = first; r < first + rows; ++r) {
                if (r % 1000 >= (wide ? 500u : 50u)) continue;
                ++after_c;
                if (r % 7 == 0) continue;  // AIR
                ++after_mode;
                if (wide && Cents(r) >= 200000) continue;
                ++passed;
                want_rows.push_back(std::to_string(r) + "|" + kModes[r % 7] + "|" + (wide ? Note(r) + "|" : ""));
            }
            want_decoded[kC] += rows;
            want_decoded[kMode] += read(after_c);
            if (wide) want_decoded[kB] += read(after_mode);
            want_decoded[kA] += read(passed);
            if (wide) want_decoded[kNote] += read(passed);
            first += rows;
        }

        // The case really mixes both kinds of read.
        if (wide) {
            ASSERT_EQ(want_decoded[kMode], kRows);
        } else {
            ASSERT_LT(want_decoded[kMode], kRows / 8);
        }
        ASSERT_EQ(want_decoded[kA], want_rows.size());

        const ScanRun run = Drain(Spec(PlanFilter(k.where), k.columns));
        EXPECT_EQ(run.counters.rows_decoded, want_decoded);
        ASSERT_GT(run.batches, 0u);
        EXPECT_EQ(run.coded[1], run.batches);
        EXPECT_EQ(run.rows, want_rows);

        std::string sql = "SELECT a, shipmode";
        if (wide) sql += ", note";
        sql += " FROM t WHERE " + k.where;
        EXPECT_EQ(Difference(Run(sql, ExecutionMode::PARALLEL), Run(sql, ExecutionMode::VECTORIZED)), "") << sql;
    }
}

TEST_F(ParallelScanSourceTest, IntendedDifferenceSingleColumnConjunctsRunFirst) {
    // Single-column conjuncts run before multi-column ones. Since c is a % 1000,
    // a < c never holds and today never divides. Here 10 / c runs first: c = 0.
    const std::string guarded = "SELECT a FROM t WHERE a < c AND 10 / c > 0";
    EXPECT_TRUE(Run(guarded, ExecutionMode::VECTORIZED).tuples.empty());
    EXPECT_EQ(ErrorOf([&] { (void)Run(guarded, ExecutionMode::PARALLEL); }), "Division by zero");

    // A morsel's conjuncts cover its whole row group before a row is handed up.
    // Today stops after five rows; here the filter reaches a = 3000 first.
    const std::string limited = "SELECT a FROM t WHERE 100 / (a - 3000) <> 7 LIMIT 5";
    const QueryResult today = Run(limited, ExecutionMode::VECTORIZED);
    std::vector<int32_t> firsts;
    for (const Tuple& t : today.tuples) firsts.push_back(t.GetValue(today.schema, 0).integer_);
    std::sort(firsts.begin(), firsts.end());
    EXPECT_EQ(firsts, (std::vector<int32_t>{0, 1, 2, 3, 4}));
    EXPECT_EQ(ErrorOf([&] { (void)Run(limited, ExecutionMode::PARALLEL); }), "Division by zero");

    // Only errors differ: with no zero divisor, both give the same answer.
    for (const std::string sql : {"SELECT a FROM t WHERE a < c AND 10 / (c + 1) > 0",
                                  "SELECT a FROM t WHERE 100 / (a - 300000) <> 7 LIMIT 5"}) {
        EXPECT_EQ(Difference(Run(sql, ExecutionMode::PARALLEL), Run(sql, ExecutionMode::VECTORIZED)), "") << sql;
    }
    EXPECT_NO_THROW(db_->GetBufferPoolManager()->DropAllPages());
}

TEST_F(ParallelScanSourceTest, DictionaryColumnsStayCoded) {
    const ColumnarTable& table = Table();
    for (size_t g = 0; g < table.RowGroupCount(); ++g) {
        ASSERT_EQ(table.ChunkAt(g, kMode).encoding, Encoding::DICTIONARY) << "group " << g;
        ASSERT_NE(table.ChunkAt(g, kNote).encoding, Encoding::DICTIONARY) << "group " << g;
    }

    // The plain decode of every row group, for the rows the filter keeps.
    std::vector<std::string> want;
    StringArena arena;
    std::vector<StringRef> plain;
    uint32_t first = 0;
    for (size_t g = 0; g < table.RowGroupCount(); ++g) {
        const uint32_t n = table.RowGroupRows(g);
        plain.resize(n);
        table.ReadRows(g, kMode, nullptr, n, plain.data(), arena);
        for (uint32_t i = 0; i < n; ++i) {
            const std::string s(View(plain[i]));
            if (s == "MAIL" || s == "SHIP") want.push_back(std::to_string(first + i) + "|" + s + "|");
        }
        first += n;
    }

    // Read whole and read by row, shipmode is handed up coded and matches.
    const PlannedFilter planned = PlanFilter("shipmode IN ('MAIL', 'SHIP')");
    for (const double fraction : {kDenseFraction, 1.0}) {
        const ScanRun run = Drain(Spec(planned, {kA, kMode}, fraction));
        ASSERT_GT(run.batches, 0u);
        EXPECT_EQ(run.coded[1], run.batches) << "fraction " << fraction;
        EXPECT_EQ(run.rows, want) << "fraction " << fraction;
    }

    // Through the whole engine: coded strings in the result, a filter, a projection.
    for (const std::string sql : {"SELECT a, shipmode FROM t WHERE shipmode IN ('MAIL', 'SHIP')",
                                  "SELECT shipmode, a FROM t WHERE c < 50 AND shipmode <> 'AIR'",
                                  "SELECT a, shipmode FROM t WHERE shipmode LIKE '%AI%' AND c + 0 < 300",
                                  "SELECT a FROM t WHERE c < 20 AND (shipmode = 'FOB' OR a < 1000)"}) {
        EXPECT_EQ(Difference(Run(sql, ExecutionMode::PARALLEL), Run(sql, ExecutionMode::VECTORIZED)), "") << sql;
    }
}

TEST_F(ParallelScanSourceTest, Q6FilterRows) {
    const std::string dir = FindTpchDir();
    if (dir.empty()) GTEST_SKIP() << "TPC-H SF=0.01 data not found. Run bench/tpch/generate_tpch_data.sh first.";

    const auto tpch = LoadLineitem(dir);

    // Q6's filter alone; its sum comes later.
    const std::string q6 = tpch::FindQuery("Q6")->sql;
    const std::string sql = "SELECT l_extendedprice, l_discount FROM lineitem " + q6.substr(q6.find("WHERE"));
    const QueryResult want = RunOn(*tpch, sql, ExecutionMode::VECTORIZED);
    ASSERT_GT(want.tuples.size(), 0u);
    EXPECT_EQ(Difference(RunOn(*tpch, sql, ExecutionMode::PARALLEL), want), "");
}

TEST_F(ParallelScanSourceTest, TpchShipmodeStaysCoded) {
    const std::string dir = FindTpchDir();
    if (dir.empty()) GTEST_SKIP() << "TPC-H SF=0.01 data not found. Run bench/tpch/generate_tpch_data.sh first.";
    const auto tpch = LoadLineitem(dir);

    const ColumnarTable& lineitem = *tpch->GetCatalog()->GetTable("lineitem")->columnar;
    auto column = [&](const std::string& name) {
        const auto& columns = lineitem.Meta().columns;
        for (uint32_t c = 0; c < columns.size(); ++c) {
            if (columns[c].name == name) return c;
        }
        ADD_FAILURE() << "no column " << name;
        return 0u;
    };
    const uint32_t mode = column("l_shipmode");
    for (size_t g = 0; g < lineitem.RowGroupCount(); ++g) {
        ASSERT_EQ(lineitem.ChunkAt(g, mode).encoding, Encoding::DICTIONARY) << "group " << g;
    }

    // Read alone, l_shipmode is handed up coded in every batch.
    const std::string where = "l_shipmode IN ('MAIL', 'SHIP')";
    const ScanRun run = Drain(Spec(PlanFilter(where, *tpch, "lineitem"), {column("l_orderkey"), column("l_linenumber"), mode}), *tpch);
    ASSERT_GT(run.batches, 0u);
    EXPECT_EQ(run.coded[2], run.batches);

    // Its rows are VECTORIZED's, as a multiset, read alone and through the engine.
    const std::string sql = "SELECT l_orderkey, l_linenumber, l_shipmode FROM lineitem WHERE " + where;
    const QueryResult want = RunOn(*tpch, sql, ExecutionMode::VECTORIZED);
    std::vector<std::string> want_rows;
    for (const Tuple& t : want.tuples) {
        std::string row;
        for (uint32_t c = 0; c < want.schema.GetColumnCount(); ++c) row += t.GetValue(want.schema, c).ToString() + "|";
        want_rows.push_back(row);
    }
    std::vector<std::string> got_rows = run.rows;
    std::sort(got_rows.begin(), got_rows.end());
    std::sort(want_rows.begin(), want_rows.end());
    ASSERT_GT(want_rows.size(), 0u);
    EXPECT_EQ(got_rows, want_rows);
    EXPECT_EQ(Difference(RunOn(*tpch, sql, ExecutionMode::PARALLEL), want), "");
}

TEST_F(ParallelScanSourceTest, CountersCountRowsAndGroups) {
    // a >= 70000 skips the first row group by its zone map.
    const PlannedFilter planned = PlanFilter("a >= 70000 AND c < 100");
    uint64_t passed = 0;
    for (uint32_t r = 70000; r < kRows; ++r) passed += r % 1000 < 100 ? 1 : 0;
    const uint64_t considered = kRows - Table().RowGroupRows(0);

    ScanSource source(Spec(planned, {kNote}));
    QueryContext ctx{db_->GetBufferPoolManager(), db_->GetCatalog()};
    for (int run = 0; run < 2; ++run) {
        // A second run starts its counts again.
        source.Prepare(ctx);
        auto local = source.MakeLocal();
        Batch batch;
        uint64_t rows = 0;
        for (size_t m = 0; m < source.MorselCount(); ++m) {
            source.StartMorsel(m, *local);
            while (source.NextBatch(*local, batch)) rows += batch.Active();
        }
        const ScanCounters k = source.Counters();
        EXPECT_EQ(rows, passed) << "run " << run;
        EXPECT_EQ(k.groups_skipped, 1u) << "run " << run;
        EXPECT_EQ(k.rows_considered, considered) << "run " << run;
        EXPECT_EQ(k.rows_passed, passed) << "run " << run;
        ASSERT_EQ(k.rows_decoded.size(), kColumns);
        EXPECT_EQ(k.rows_decoded[kA], considered) << "run " << run;
        EXPECT_EQ(k.rows_decoded[kC], considered) << "run " << run;
        EXPECT_EQ(k.rows_decoded[kNote], passed) << "run " << run;
        for (uint32_t c : {kB, kD, kF, kMode, kTs}) EXPECT_EQ(k.rows_decoded[c], 0u) << kNames[c];
    }

    // No filter: every group read and every row passed on.
    ScanSpec all;
    all.table = "t";
    all.columns = {kA, kTs};
    const ScanRun run = Drain(std::move(all));
    EXPECT_EQ(run.rows.size(), kRows);
    EXPECT_EQ(run.counters.groups_skipped, 0u);
    EXPECT_EQ(run.counters.rows_considered, kRows);
    EXPECT_EQ(run.counters.rows_passed, kRows);
    EXPECT_EQ(run.counters.rows_decoded[kA], kRows);
    EXPECT_EQ(run.counters.rows_decoded[kTs], kRows);
    EXPECT_EQ(run.counters.rows_decoded[kB], 0u);
}

TEST_F(ParallelScanSourceTest, NullInFilterColumnIsRefused) {
    const std::string tbl = TempPath("_n.tbl");
    std::ofstream(tbl, std::ios::binary) << "1|5\n|6\n3|7\n";
    extra_files_.push_back(tbl);
    LoadOptions options;
    options.storage = StorageKind::COLUMNAR;
    options.null_text = "";
    db_->LoadTable("n", Schema({Column("x", ColumnType::Integer(), true), Column("y", ColumnType::Integer())}), tbl, options);

    // x is only filtered on, never handed up, and still refused with today's text.
    const std::string sql = "SELECT y FROM n WHERE x > 0";
    const std::string today = "column 'x' holds NULLs, which queries can't read yet";
    EXPECT_EQ(ErrorOf([&] { (void)Run(sql, ExecutionMode::VECTORIZED); }), today);
    EXPECT_EQ(ErrorOf([&] { (void)Run(sql, ExecutionMode::PARALLEL); }), today);
    EXPECT_EQ(Run("SELECT y FROM n WHERE y > 5", ExecutionMode::PARALLEL).tuples.size(), 2u);
    EXPECT_NO_THROW(db_->GetBufferPoolManager()->DropAllPages());
}

TEST_F(ParallelScanSourceTest, CountOfANullColumnIsRefused) {
    const std::string tbl = TempPath("_cn.tbl");
    std::ofstream(tbl, std::ios::binary) << "1|5\n|6\n3|7\n";
    extra_files_.push_back(tbl);
    LoadOptions options;
    options.storage = StorageKind::COLUMNAR;
    options.null_text = "";
    db_->LoadTable("cn", Schema({Column("x", ColumnType::Integer(), true), Column("y", ColumnType::Integer())}), tbl, options);

    // COUNT never reads its argument, but the plan lists x, so its NULL is refused.
    const std::string today = "column 'x' holds NULLs, which queries can't read yet";
    for (const std::string sql : {"SELECT COUNT(x) FROM cn", "SELECT COUNT(x + 1) FROM cn", "SELECT y, COUNT(x) FROM cn GROUP BY y"}) {
        EXPECT_EQ(ErrorOf([&] { (void)Run(sql, ExecutionMode::VECTORIZED); }), today) << sql;
        EXPECT_EQ(ErrorOf([&] { (void)Run(sql, ExecutionMode::PARALLEL); }), today) << sql;
    }
    const QueryResult all = Run("SELECT COUNT(*) FROM cn", ExecutionMode::PARALLEL);
    ASSERT_EQ(all.tuples.size(), 1u);
    EXPECT_EQ(all.tuples[0].GetValue(all.schema, 0).bigint_, 3);
    EXPECT_NO_THROW(db_->GetBufferPoolManager()->DropAllPages());
}

TEST_F(ParallelScanSourceTest, IndexScanRefusesNullsInListedColumns) {
    // y is the row number; x is too, except a NULL at y = 995.
    std::string text;
    for (int r = 0; r < 1000; ++r) text += (r == 995 ? std::string() : std::to_string(r)) + "|" + std::to_string(r) + "\n";
    const std::string tbl = TempPath("_in.tbl");
    std::ofstream(tbl, std::ios::binary) << text;
    extra_files_.push_back(tbl);
    LoadOptions options;
    options.storage = StorageKind::COLUMNAR;
    options.null_text = "";
    db_->LoadTable("ix", Schema({Column("x", ColumnType::Integer(), true), Column("y", ColumnType::Integer())}), tbl, options);
    (void)db_->GetCatalog()->CreateIndex("ix_y", "ix", "y");

    const std::string sql = "SELECT COUNT(x) FROM ix WHERE y >= 990";
    for (ExecutionMode mode : {ExecutionMode::VECTORIZED, ExecutionMode::PARALLEL}) {
        QueryOptions plan_options;
        plan_options.mode = mode;
        const auto plan = db_->MakePlanner(plan_options).Plan(std::move(*Parser(sql).Parse()));
        bool index_scan = false;
        for (const PlanNode* n = plan.get(); n != nullptr; n = n->children.empty() ? nullptr : n->children[0].get()) {
            index_scan = index_scan || n->type == PlanNodeType::INDEX_SCAN;
        }
        ASSERT_TRUE(index_scan) << "no index scan planned";
    }
    const std::string today = "column 'x' holds NULLs, which queries can't read yet";
    EXPECT_EQ(ErrorOf([&] { (void)Run(sql, ExecutionMode::VECTORIZED); }), today);
    EXPECT_EQ(ErrorOf([&] { (void)Run(sql, ExecutionMode::PARALLEL); }), today);
    const QueryResult all = Run("SELECT COUNT(*) FROM ix WHERE y >= 990", ExecutionMode::PARALLEL);
    ASSERT_EQ(all.tuples.size(), 1u);
    EXPECT_EQ(all.tuples[0].GetValue(all.schema, 0).bigint_, 10);
    EXPECT_NO_THROW(db_->GetBufferPoolManager()->DropAllPages());
}

}  // namespace
}  // namespace shilmandb::parallel
