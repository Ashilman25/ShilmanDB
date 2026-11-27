#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "buffer/buffer_pool_manager.hpp"
#include "catalog/catalog.hpp"
#include "engine/database.hpp"
#include "executor/parallel/batch.hpp"
#include "executor/parallel/pipeline.hpp"
#include "executor/parallel/scan_source.hpp"
#include "executor/parallel/worker_pool.hpp"
#include "index/btree_index.hpp"
#include "parser/parser.hpp"
#include "planner/plan_node.hpp"
#include "planner/planner.hpp"
#include "storage/columnar/columnar_table.hpp"
#include "tpch/tpch_workload.hpp"

// Index scans on columnar tables in the parallel executor: positions from the
// B+tree, sorted and grouped by row group, each page read at most once.
namespace shilmandb::parallel {
namespace {

constexpr uint32_t kGroupRows = 65536;
constexpr uint32_t kGroups = 20;
constexpr uint32_t kRows = (kGroups - 1) * kGroupRows + 1000;  // 20 row groups
constexpr uint32_t kNullRow = 15 * kGroupRows + 77;            // nul's one NULL
constexpr unsigned kThreads[] = {1, 8};

// The test table's columns, by number.
constexpr uint32_t kK = 0, kP = 1, kAmt = 2, kD = 3, kNote = 4, kTag = 5, kNul = 6;
const char* const kTags[] = {"amber", "blue", "coral", "dune", "ember", "fern", "gold"};

// k is the row number; p scatters it,
// so p's index order jumps between row groups.
uint32_t P(uint32_t r) {
    return static_cast<uint32_t>((static_cast<uint64_t>(r) * 7919u) % kRows);
}

// Row r's amt in cents, 0 .. 9,999.99.
uint32_t Cents(uint32_t r) {
    return static_cast<uint32_t>((static_cast<uint64_t>(r) * 104729u) % 1000000u);
}

// Row r of the load file: k, p, amt, d, note, tag, nul.
void WriteTable(const std::string& path) {
    std::string out;
    out.reserve(static_cast<size_t>(kRows) * 48);
    char buf[128];
    for (uint32_t r = 0; r < kRows; ++r) {
        const uint32_t cents = Cents(r);
        const std::string nul = r == kNullRow ? "\\N" : std::to_string(r % 100);
        std::snprintf(buf, sizeof(buf), "%u|%u|%u.%02u|%04u-%02u-%02u|n%u|%s|%s\n", r, P(r), cents / 100, cents % 100, 1992 + r % 7,
                      1 + r % 12, 1 + r % 28, r, kTags[r % 7], nul.c_str());
        out += buf;
    }
    std::ofstream(path, std::ios::binary) << out;
}

// One result row as text; decimals keep every bit.
std::string RowText(const QueryResult& r, size_t i) {
    std::string out;
    for (uint32_t c = 0; c < r.schema.GetColumnCount(); ++c) {
        const Value v = r.tuples[i].GetValue(r.schema, c);
        if (v.type_ == TypeId::DECIMAL) {
            char buf[40];
            std::snprintf(buf, sizeof(buf), "%.17g", v.decimal_);
            out += buf;
        } else {
            out += v.ToString();
        }
        out += '|';
    }
    return out;
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
    std::vector<std::string> x, y;
    for (size_t i = 0; i < got.tuples.size(); ++i) x.push_back(RowText(got, i));
    for (size_t i = 0; i < want.tuples.size(); ++i) y.push_back(RowText(want, i));
    std::sort(x.begin(), x.end());
    std::sort(y.begin(), y.end());
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] != y[i]) return "row " + std::to_string(i) + ": got " + x[i] + " expected " + y[i];
    }
    return "";
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

// The first node of this type in the plan, or null.
const PlanNode* FindNode(const PlanNode& root, PlanNodeType type) {
    if (root.type == type) return &root;
    for (const auto& child : root.children) {
        if (const PlanNode* found = FindNode(*child, type)) return found;
    }
    return nullptr;
}

// The filter right above the plan's index scan, or null.
const FilterPlanNode* FilterAboveIndexScan(const PlanNode& root) {
    if (root.type == PlanNodeType::FILTER && root.children[0]->type == PlanNodeType::INDEX_SCAN) return static_cast<const FilterPlanNode*>(&root);
    for (const auto& child : root.children) {
        if (const FilterPlanNode* found = FilterAboveIndexScan(*child)) return found;
    }
    return nullptr;
}

// A negative entry cost makes any usable index win,
// so these tests always plan their index scan.
ColumnarScanCosts ForceIndex() { return ColumnarScanCosts{0.0, 0.0, -1.0, 0.0}; }

// Data shared by the tests of one process, loaded on first use.
class ParallelIndexScanSourceTest : public ::testing::Test {
protected:
    static std::unique_ptr<Database> db_;
    static std::unique_ptr<Database> tpch_;
    static std::string file_;

    static void SetUpTestSuite() {
        file_ = (std::filesystem::temp_directory_path() / ("shilmandb_parallel_index_scan_source_test_" + std::to_string(::getpid()) + ".db")).string();
    }

    static void TearDownTestSuite() {
        db_.reset();
        tpch_.reset();
        std::filesystem::remove(file_);
        for (const char* other : {"_tpch", "_dup", "_dup.tbl", "_damaged", "_damaged.tbl"}) std::filesystem::remove(file_ + other);
    }

    // ix, indexed on k and p, 20 row groups.
    // The pool holds every page one query reads.
    static Database& Db() {
        if (db_) return *db_;
        std::filesystem::remove(file_);
        db_ = std::make_unique<Database>(file_, 8192);
        const std::string tbl = file_ + "_ix.tbl";
        WriteTable(tbl);
        LoadOptions options;
        options.storage = StorageKind::COLUMNAR;
        options.null_text = "\\N";
        const Schema schema({Column("k", ColumnType::Integer()), Column("p", ColumnType::Integer()), Column("amt", ColumnType::Numeric(12, 2)),
                             Column("d", ColumnType::Date()), Column("note", ColumnType::Text()), Column("tag", ColumnType::Varchar(8)),
                             Column("nul", ColumnType::Integer(), true)});
        db_->LoadTable("ix", schema, tbl, options);
        std::filesystem::remove(tbl);
        (void)db_->GetCatalog()->CreateIndex("ix_k", "ix", "k");
        (void)db_->GetCatalog()->CreateIndex("ix_p", "ix", "p");
        db_->GetCatalog()->UpdateTableStats("ix");
        db_->SetColumnarScanCosts(ForceIndex());
        return *db_;
    }

    // TPC-H SF=0.01, columnar, with the 8 indexes; skips without the data.
    // The skip ends only this helper, so callers check IsSkipped() after it.
    void NeedTpch() {
        if (tpch_) return;
        std::string dir;
        for (const std::string d : {"bench/tpch_data/sf0.01/", "../bench/tpch_data/sf0.01/", "../../bench/tpch_data/sf0.01/"}) {
            if (dir.empty() && std::filesystem::exists(d + "lineitem.tbl")) dir = d;
        }
        if (dir.empty()) GTEST_SKIP() << "TPC-H SF=0.01 data not found. Run bench/tpch/generate_tpch_data.sh first.";
        const std::string file = file_ + "_tpch";
        std::filesystem::remove(file);
        tpch_ = std::make_unique<Database>(file, 4096);
        LoadOptions options;
        options.storage = StorageKind::COLUMNAR;
        for (const auto& td : tpch::TableDescs()) tpch_->LoadTable(td.name, td.schema, dir + td.tbl_file, options);
        auto* catalog = tpch_->GetCatalog();
        for (const auto& idx : tpch::IndexDescs()) (void)catalog->CreateIndex(idx.index_name, idx.table_name, idx.column_name);
        for (const auto& td : tpch::TableDescs()) catalog->UpdateTableStats(td.name);
        tpch_->SetColumnarScanCosts(ForceIndex());
    }

    // Runs sql on db in one mode; the parallel executor on this many threads.
    static QueryResult Run(Database& db, const std::string& sql, ExecutionMode mode, unsigned threads = 1) {
        QueryOptions options;
        options.mode = mode;
        options.threads = threads;
        return db.ExecuteSQL(sql, options);
    }

    // The plan the parallel executor gets for sql.
    static std::unique_ptr<PlanNode> ParallelPlan(Database& db, const std::string& sql) {
        QueryOptions options;
        options.mode = ExecutionMode::PARALLEL;
        return db.MakePlanner(options).Plan(std::move(*Parser(sql).Parse()));
    }

    // The loaded test table.
    static const ColumnarTable& Table() {
        return *Db().GetCatalog()->GetTable("ix")->columnar;
    }

    // The B+tree walk an index scan makes over [low, high]; returns the rows it found.
    static std::vector<uint32_t> WalkIndex(const std::string& index, const std::optional<Value>& low, const std::optional<Value>& high) {
        BTreeIndex& tree = *Db().GetCatalog()->GetIndex(index)->index;
        std::vector<uint32_t> rows;
        for (auto it = low ? tree.Begin(*low) : tree.Begin(); !it.IsEnd(); ++it) {
            const auto entry = *it;
            if (high && entry.first > *high) break;
            const RowPosition pos = FromRid(entry.second);
            rows.push_back(pos.row_group * kGroupRows + pos.offset);
        }
        return rows;
    }

    // Fetches and misses of an index scan's walk on this many threads, from an empty pool:
    // the iterator's walk without cuts, else each piece SplitKeys cuts, leaf by leaf.
    static std::pair<uint64_t, uint64_t> WalkCost(const std::string& index, const std::optional<Value>& low, const std::optional<Value>& high,
                                                  unsigned threads) {
        BufferPoolManager* bpm = Db().GetBufferPoolManager();
        BTreeIndex& tree = *Db().GetCatalog()->GetIndex(index)->index;
        bpm->DropAllPages();
        bpm->ResetStats();
        std::vector<Value> cuts;
        if (threads > 1) cuts = tree.SplitKeys(low.value_or(Value()), high.value_or(Value()), threads);
        if (cuts.empty()) (void)WalkIndex(index, low, high);
        for (size_t i = 0; !cuts.empty() && i <= cuts.size(); ++i) {
            const Value* until = i < cuts.size() ? &cuts[i] : nullptr;
            tree.ForEachEntry(i == 0 ? low.value_or(Value()) : cuts[i - 1], [&](const Value& key, const RID&) {
                return until != nullptr ? key < *until : !high || key <= *high;
            });
        }
        return {bpm->GetHitCount() + bpm->GetMissCount(), bpm->GetMissCount()};
    }

    // Every morsel's column 0 after Prepare in this context.
    // Column 0 holds the row number, so these are the positions.
    static std::vector<std::vector<uint32_t>> MorselRows(const std::string& table, const std::string& index, const std::optional<Value>& low,
                                                         const std::optional<Value>& high, QueryContext& ctx) {
        ScanSpec spec;
        spec.table = table;
        spec.columns = {0};
        auto source = MakeIndexScanSource(std::move(spec), index, low, high);
        source->Prepare(ctx);
        std::vector<std::vector<uint32_t>> morsels(source->MorselCount());
        auto local = source->MakeLocal();
        Batch batch;
        for (size_t m = 0; m < morsels.size(); ++m) {
            source->StartMorsel(m, *local);
            while (source->NextBatch(*local, batch)) {
                for (uint32_t r = 0; r < batch.rows; ++r) morsels[m].push_back(static_cast<uint32_t>(batch.slots[0].I64()[r]));
            }
        }
        return morsels;
    }

    // Expects every frame unpinned and the pool's concurrent mode off.
    static void ExpectPoolIdle(const BufferPoolManager& bpm) {
        for (size_t f = 0; f < bpm.GetPoolSize(); ++f) {
            ASSERT_EQ(bpm.GetPage(static_cast<frame_id_t>(f)).GetPinCount(), 0) << "frame " << f << " stayed pinned";
        }
        EXPECT_FALSE(bpm.IsConcurrent());
    }

    // Pages of the listed columns of ix in the row groups holding these rows.
    static uint64_t ChunkPagesOf(const std::vector<uint32_t>& columns, const std::vector<uint32_t>& rows) {
        std::set<uint32_t> groups;
        for (uint32_t r : rows) groups.insert(r / kGroupRows);
        uint64_t pages = 0;
        for (uint32_t g : groups) {
            for (uint32_t c : columns) pages += Table().Meta().row_groups[g].chunks[c].page_count;
        }
        return pages;
    }
};

std::unique_ptr<Database> ParallelIndexScanSourceTest::db_;
std::unique_ptr<Database> ParallelIndexScanSourceTest::tpch_;
std::string ParallelIndexScanSourceTest::file_;

TEST_F(ParallelIndexScanSourceTest, IndexScanEqualsSeqScan) {
    // seq is the same range written so no index can serve it: the reference.
    struct Range {
        std::string where;
        std::string seq;
        size_t rows;
    };
    // k follows row positions; p jumps between groups. Group 1 is k = 65536 .. 131071.
    std::vector<Range> ranges = {
        {"k = 70000", "k + 0 = 70000", 1},
        {"p = 12345", "p + 0 = 12345", 1},
        {"k >= 70000 AND k <= 70500", "k + 0 >= 70000 AND k + 0 <= 70500", 501},
        {"k >= 60000 AND k <= 140000", "k + 0 >= 60000 AND k + 0 <= 140000", 80001},  // all of group 1 matches
        {"p >= 1000 AND p <= 41000", "p + 0 >= 1000 AND p + 0 <= 41000", 40001},
        {"k <= 1500", "k + 0 <= 1500", 1501},
        {"p <= 3000", "p + 0 <= 3000", 3001},
        {"k >= 1245000", "k + 0 >= 1245000", kRows - 1245000},
        {"p >= 1243000", "p + 0 >= 1243000", kRows - 1243000},
        {"k >= 10 AND k <= 5", "k + 0 >= 10 AND k + 0 <= 5", 0},
        {"p >= 2000000", "p + 0 >= 2000000", 0},
    };
    const std::vector<std::string> filters = {"", " AND amt < 2500.00", " AND tag <> 'fern' AND k + p > 600000", " AND note LIKE '%7'"};
    Database& db = Db();
    size_t total = 0;
    for (const Range& range : ranges) {
        for (const std::string& filter : filters) {
            const std::string where = range.where + filter;
            const std::string sql = "SELECT k, p, amt, d, note, tag FROM ix WHERE " + where;
            SCOPED_TRACE(sql);
            const auto plan = ParallelPlan(db, sql);
            ASSERT_NE(FindNode(*plan, PlanNodeType::INDEX_SCAN), nullptr) << "no index scan planned";
            ASSERT_EQ(FindNode(*plan, PlanNodeType::SEQ_SCAN), nullptr);
            if (!filter.empty()) ASSERT_NE(FilterAboveIndexScan(*plan), nullptr) << "no filter above the index scan";
            const std::string reference = "SELECT k, p, amt, d, note, tag FROM ix WHERE " + range.seq + filter;
            const auto seq_plan = ParallelPlan(db, reference);
            ASSERT_NE(FindNode(*seq_plan, PlanNodeType::SEQ_SCAN), nullptr) << reference;
            ASSERT_EQ(FindNode(*seq_plan, PlanNodeType::INDEX_SCAN), nullptr) << reference;
            const QueryResult want = Run(db, reference, ExecutionMode::PARALLEL, 8);
            if (filter.empty()) EXPECT_EQ(want.tuples.size(), range.rows);
            for (unsigned threads : kThreads) EXPECT_EQ(Difference(Run(db, sql, ExecutionMode::PARALLEL, threads), want), "") << threads << " threads";
            total += want.tuples.size();
        }
    }
    EXPECT_GT(total, 0u);

    // An aggregate over an index scan, against today's index scan, which sums doubles.
    const std::string count = "SELECT COUNT(*), SUM(amt) FROM ix WHERE p >= 1000 AND p <= 41000 AND tag = 'gold'";
    EXPECT_TRUE(tpch::SameResult(Run(db, count, ExecutionMode::PARALLEL, 8), Run(db, count, ExecutionMode::VECTORIZED)));
}

TEST_F(ParallelIndexScanSourceTest, MorselsStartFromTheirMatchesInPositionOrder) {
    // The filter's conjunct runs inside the index scan, on the matches only.
    Database& db = Db();
    struct Case {
        std::string where;
        bool filtered;
    };
    for (const Case& c : {Case{"p >= 1000 AND p <= 41000", false}, Case{"k >= 70000 AND k <= 70500 AND amt < 5000.00", true},
                          Case{"p <= 3000 AND amt < 5000.00", true}}) {
        SCOPED_TRACE(c.where);
        const auto plan = ParallelPlan(db, "SELECT k FROM ix WHERE " + c.where);
        const auto* scan = static_cast<const IndexScanPlanNode*>(FindNode(*plan, PlanNodeType::INDEX_SCAN));
        ASSERT_NE(scan, nullptr);
        const FilterPlanNode* filter = FilterAboveIndexScan(*plan);
        ASSERT_EQ(filter != nullptr, c.filtered);

        ScanSpec spec;
        spec.table = "ix";
        spec.columns = {kK};
        if (filter != nullptr) AddConjuncts(spec, Table(), *filter->predicate);
        auto source = MakeIndexScanSource(std::move(spec), scan->index_name, scan->low_key, scan->high_key);
        QueryContext ctx{db.GetBufferPoolManager(), db.GetCatalog()};
        source->Prepare(ctx);

        // Matches, by row number, ascending.
        std::vector<uint32_t> matches = WalkIndex(scan->index_name, scan->low_key, scan->high_key);
        std::sort(matches.begin(), matches.end());
        std::set<uint32_t> groups;
        for (uint32_t r : matches) groups.insert(r / kGroupRows);
        ASSERT_EQ(source->MorselCount(), groups.size());

        auto local = source->MakeLocal();
        Batch batch;
        std::vector<uint32_t> got;
        for (size_t m = 0; m < source->MorselCount(); ++m) {
            source->StartMorsel(m, *local);
            while (source->NextBatch(*local, batch)) {
                for (uint32_t r = 0; r < batch.rows; ++r) got.push_back(static_cast<uint32_t>(batch.slots[0].I64()[r]));
            }
        }
        std::vector<uint32_t> want;
        for (uint32_t r : matches) {
            if (!c.filtered || Cents(r) < 500000) want.push_back(r);
        }
        ASSERT_GT(want.size(), 0u);
        EXPECT_EQ(got, want);  // row-position order, not key order

        const ScanCounters k = static_cast<const ScanSource&>(*source).Counters();
        EXPECT_EQ(k.groups_skipped, kGroups - groups.size());
        EXPECT_EQ(k.rows_considered, matches.size());
        EXPECT_EQ(k.rows_passed, want.size());
    }
}

TEST_F(ParallelIndexScanSourceTest, EachPageAtMostOncePerColumn) {
    // Fetches count every page pinned, hit or miss, so a page read twice shows.
    // The walk is the query's own, in pieces on several threads.
    // The source alone, after Prepare's walk: its morsels pin only chunk pages.
    Database& db = Db();
    BufferPoolManager* bpm = db.GetBufferPoolManager();
    struct Case {
        std::string where;
        std::string index;
        std::optional<Value> low;
        std::optional<Value> high;
    };
    const std::vector<Case> cases = {
        {"p >= 1000 AND p <= 41000", "ix_p", Value(int32_t(1000)), Value(int32_t(41000))},                          // sparse, every group
        {"k >= 60000 AND k <= 140000", "ix_k", Value(int32_t(60000)), Value(int32_t(140000))},                      // one whole group
        {"p <= 20000 AND amt < 5000.00", "ix_p", std::nullopt, Value(int32_t(20000))},                              // a fused filter
    };
    const std::vector<uint32_t> read = {kK, kP, kAmt, kD, kNote};
    for (const Case& c : cases) {
        for (unsigned threads : kThreads) {
            const std::string sql = "SELECT k, p, amt, d, note FROM ix WHERE " + c.where;
            SCOPED_TRACE(sql + " on " + std::to_string(threads) + " threads");
            ASSERT_NE(FindNode(*ParallelPlan(db, sql), PlanNodeType::INDEX_SCAN), nullptr);

            const auto [walk_fetches, walk_misses] = WalkCost(c.index, c.low, c.high, threads);
            const uint64_t chunk_pages = ChunkPagesOf(read, WalkIndex(c.index, c.low, c.high));
            ASSERT_GT(chunk_pages, 0u);

            bpm->DropAllPages();
            bpm->ResetStats();
            const QueryResult got = Run(db, sql, ExecutionMode::PARALLEL, threads);
            EXPECT_GT(got.tuples.size(), 0u);
            EXPECT_LE(bpm->GetHitCount() + bpm->GetMissCount(), walk_fetches + chunk_pages);
            EXPECT_LE(bpm->GetMissCount(), walk_misses + chunk_pages);
        }

        // Exactly the read columns' chunk pages, each once: every group's pages are
        // read whole or by row, and the dense case reads some chunks whole.
        const auto plan = ParallelPlan(db, "SELECT k, p, amt, d, note FROM ix WHERE " + c.where);
        const auto* scan = static_cast<const IndexScanPlanNode*>(FindNode(*plan, PlanNodeType::INDEX_SCAN));
        ScanSpec spec;
        spec.table = "ix";
        spec.columns = read;
        if (const FilterPlanNode* filter = FilterAboveIndexScan(*plan)) AddConjuncts(spec, Table(), *filter->predicate);
        auto source = MakeIndexScanSource(std::move(spec), scan->index_name, scan->low_key, scan->high_key);
        QueryContext ctx{bpm, db.GetCatalog()};
        source->Prepare(ctx);
        const uint64_t bound = ChunkPagesOf(read, WalkIndex(c.index, c.low, c.high));
        bpm->DropAllPages();
        bpm->ResetStats();
        auto local = source->MakeLocal();
        Batch batch;
        size_t rows = 0;
        for (size_t m = 0; m < source->MorselCount(); ++m) {
            source->StartMorsel(m, *local);
            while (source->NextBatch(*local, batch)) rows += batch.rows;
        }
        SCOPED_TRACE(c.where + " through the source");
        EXPECT_GT(rows, 0u);
        EXPECT_LE(bpm->GetHitCount() + bpm->GetMissCount(), bound);
    }
}

TEST_F(ParallelIndexScanSourceTest, NullsRefusedOnlyInGroupsWithMatches) {
    // nul's one NULL sits in group 15; the clean range reads only group 1.
    Database& db = Db();
    const std::string clean = "SELECT k, nul FROM ix WHERE k >= 70000 AND k <= 70500";
    const QueryResult want = Run(db, clean, ExecutionMode::VECTORIZED);
    EXPECT_EQ(want.tuples.size(), 501u);
    const std::string today = "column 'nul' holds NULLs, which queries can't read yet";
    const std::string hit = "SELECT k, nul FROM ix WHERE k >= 983100 AND k <= 983200";
    EXPECT_EQ(ErrorOf([&] { (void)Run(db, hit, ExecutionMode::VECTORIZED); }), today);
    for (unsigned threads : kThreads) {
        SCOPED_TRACE(threads);
        EXPECT_EQ(Difference(Run(db, clean, ExecutionMode::PARALLEL, threads), want), "");
        EXPECT_EQ(ErrorOf([&] { (void)Run(db, hit, ExecutionMode::PARALLEL, threads); }), today);
        // Matches across many groups, one of them the NULL's, on several workers.
        EXPECT_EQ(ErrorOf([&] { (void)Run(db, "SELECT nul FROM ix WHERE p >= 1000 AND p <= 41000", ExecutionMode::PARALLEL, threads); }), today);

        BufferPoolManager* bpm = db.GetBufferPoolManager();
        for (size_t f = 0; f < bpm->GetPoolSize(); ++f) {
            ASSERT_EQ(bpm->GetPage(static_cast<frame_id_t>(f)).GetPinCount(), 0) << "frame " << f << " stayed pinned";
        }
        EXPECT_FALSE(bpm->IsConcurrent());
        EXPECT_NO_THROW(bpm->DropAllPages());
        EXPECT_EQ(Run(db, "SELECT k FROM ix WHERE k <= 9", ExecutionMode::PARALLEL, threads).tuples.size(), 10u);
    }
}

TEST_F(ParallelIndexScanSourceTest, Q14AndQ10WithIndexes) {
    NeedTpch();
    if (IsSkipped()) return;
    Database* tpch = tpch_.get();
    for (const std::string tag : {"Q14", "Q10"}) {
        SCOPED_TRACE(tag);
        const std::string sql = tpch::FindQuery(tag)->sql;
        ASSERT_NE(FindNode(*ParallelPlan(*tpch, sql), PlanNodeType::INDEX_SCAN), nullptr) << "no index scan planned";
        const QueryResult want = Run(*tpch, sql, ExecutionMode::VECTORIZED);
        ASSERT_GT(want.tuples.size(), 0u);
        for (unsigned threads : kThreads) EXPECT_TRUE(tpch::SameResult(Run(*tpch, sql, ExecutionMode::PARALLEL, threads), want)) << threads << " threads";
    }
}

TEST_F(ParallelIndexScanSourceTest, SplitWalkFindsTheSamePositions) {
    // dup: 70,000 rows in 2 row groups; each g has 1,750 copies spread over several leaves.
    const std::string file = file_ + "_dup";
    const std::string tbl = file + ".tbl";
    std::filesystem::remove(file);
    {
        std::string rows;
        for (int r = 0; r < 70000; ++r) rows += std::to_string(r) + "|" + std::to_string(r % 40) + "\n";
        std::ofstream(tbl, std::ios::binary) << rows;
    }
    auto dup = std::make_unique<Database>(file, 2048);
    LoadOptions options;
    options.storage = StorageKind::COLUMNAR;
    dup->LoadTable("dup", Schema({Column("k", ColumnType::Integer()), Column("g", ColumnType::Integer())}), tbl, options);
    std::filesystem::remove(tbl);
    (void)dup->GetCatalog()->CreateIndex("dup_g", "dup", "g");

    struct Case {
        std::string name;
        Database* db;
        std::string table;
        std::string index;
        std::optional<Value> low;
        std::optional<Value> high;
        size_t rows;
        bool split;  // whether 4 threads cut the range into pieces
    };
    const auto v = [](int32_t x) { return std::optional<Value>(Value(x)); };
    Database* ix = &Db();
    const std::vector<Case> cases = {
        {"inside one leaf", ix, "ix", "ix_k", v(70000), v(70009), 10, false},
        {"across many leaves", ix, "ix", "ix_k", v(1000), v(900000), 899001, true},
        {"scattered positions", ix, "ix", "ix_p", v(1000), v(41000), 40001, true},
        {"open low", ix, "ix", "ix_p", std::nullopt, v(300000), 300001, true},
        {"open high", ix, "ix", "ix_k", v(1100000), std::nullopt, kRows - 1100000, true},
        {"empty, low past high", ix, "ix", "ix_k", v(10), v(5), 0, false},
        {"empty, past the end", ix, "ix", "ix_p", v(2000000), std::nullopt, 0, false},
        {"the whole tree", ix, "ix", "ix_k", std::nullopt, std::nullopt, kRows, true},
        {"copies across leaf splits", dup.get(), "dup", "dup_g", v(5), v(30), 26 * 1750, true},
        {"one key's copies", dup.get(), "dup", "dup_g", v(7), v(7), 1750, false},
        {"all of g", dup.get(), "dup", "dup_g", std::nullopt, std::nullopt, 70000, true},
    };

    WorkerPool pool(4);
    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        Catalog* catalog = c.db->GetCatalog();
        BufferPoolManager* bpm = c.db->GetBufferPoolManager();
        BTreeIndex& tree = *catalog->GetIndex(c.index)->index;
        EXPECT_EQ(!tree.SplitKeys(c.low.value_or(Value()), c.high.value_or(Value()), 4).empty(), c.split);

        QueryContext serial{bpm, catalog};
        const auto want = MorselRows(c.table, c.index, c.low, c.high, serial);
        QueryContext split{bpm, catalog};
        split.threads = 4;
        split.pool = &pool;
        const auto got = MorselRows(c.table, c.index, c.low, c.high, split);
        ExpectPoolIdle(*bpm);

        ASSERT_EQ(got.size(), want.size()) << "morsels";
        size_t rows = 0;
        for (size_t m = 0; m < want.size(); ++m) {
            EXPECT_EQ(got[m], want[m]) << "morsel " << m;
            EXPECT_TRUE(std::is_sorted(got[m].begin(), got[m].end())) << "morsel " << m;
            rows += got[m].size();
        }
        EXPECT_EQ(rows, c.rows);
    }
    dup.reset();
    std::filesystem::remove(file);
}

TEST_F(ParallelIndexScanSourceTest, SplitWalkRefusesADamagedIndex) {
    // A small table of its own, one row group; one extra index entry points at row group 5.
    const std::string file = file_ + "_damaged";
    const std::string tbl = file + ".tbl";
    std::filesystem::remove(file);
    {
        std::string rows;
        for (int r = 0; r < 20000; ++r) rows += std::to_string(r) + "|" + std::to_string(r % 97) + "\n";
        std::ofstream(tbl, std::ios::binary) << rows;
    }
    {
        Database db(file, 1024);
        LoadOptions options;
        options.storage = StorageKind::COLUMNAR;
        db.LoadTable("dmg", Schema({Column("k", ColumnType::Integer()), Column("v", ColumnType::Integer())}), tbl, options);
        Catalog* catalog = db.GetCatalog();
        (void)catalog->CreateIndex("dmg_k", "dmg", "k");
        catalog->UpdateTableStats("dmg");
        db.SetColumnarScanCosts(ForceIndex());
        BTreeIndex& tree = *catalog->GetIndex("dmg_k")->index;
        tree.Insert(Value(int32_t(15000)), ToRid(RowPosition{5, 0}));
        const Value low(int32_t(0));
        const Value high(int32_t(19999));
        ASSERT_FALSE(tree.SplitKeys(low, high, 4).empty()) << "the walk isn't cut into pieces";

        const std::string text = "damaged index: dmg_k points past table dmg";
        BufferPoolManager* bpm = db.GetBufferPoolManager();
        WorkerPool pool(4);
        for (const unsigned threads : {1u, 4u}) {
            SCOPED_TRACE(std::to_string(threads) + " threads");
            ScanSpec spec;
            spec.table = "dmg";
            spec.columns = {0};
            auto source = MakeIndexScanSource(std::move(spec), "dmg_k", low, high);
            QueryContext ctx{bpm, catalog};
            ctx.threads = threads;
            ctx.pool = threads > 1 ? &pool : nullptr;
            EXPECT_EQ(ErrorOf([&] { source->Prepare(ctx); }), text);
            ExpectPoolIdle(*bpm);

            // The whole query, through the database's own pool.
            EXPECT_EQ(ErrorOf([&] { (void)Run(db, "SELECT k, v FROM dmg WHERE k >= 0 AND k <= 19999", ExecutionMode::PARALLEL, threads); }), text);
            ExpectPoolIdle(*bpm);
            EXPECT_NO_THROW(bpm->DropAllPages());
            EXPECT_EQ(Run(db, "SELECT k FROM dmg WHERE k >= 100 AND k <= 14000", ExecutionMode::PARALLEL, threads).tuples.size(), 13901u);
        }
    }
    std::filesystem::remove(file);
    std::filesystem::remove(tbl);
}

}  // namespace
}  // namespace shilmandb::parallel
