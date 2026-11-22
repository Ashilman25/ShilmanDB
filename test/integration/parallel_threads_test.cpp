#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "buffer/buffer_pool_manager.hpp"
#include "catalog/catalog.hpp"
#include "engine/database.hpp"
#include "executor/parallel/parallel_executor.hpp"
#include "integration/parallel_random_queries.hpp"
#include "tpch/tpch_workload.hpp"

// The parallel executor at several thread counts: answers that must not change
// with the count, tables smaller than the count, empty morsels, errors mid-query.
namespace shilmandb {
namespace {

constexpr size_t kWideGroups = 40;
constexpr size_t kWideRows = (kWideGroups - 1) * 65536 + 1000;  // 40 row groups
constexpr size_t kZeroGroup = 17;                              // w_div is 0 only here
constexpr size_t kNullRow = 23 * 65536 + 500;                  // w_null's one NULL
constexpr size_t kSideRows = 70000;                            // past one hash partition
constexpr double kDoubleTol = 1e-9;                            // DECIMAL cells, random queries
constexpr unsigned kThreadCounts[] = {1, 2, 4, 8, 18};
constexpr int kRandomShards = 5;
constexpr int kRandomSeeds = 100;
constexpr int kRandomQueriesPerSeed = 8;
constexpr size_t kDeepRows = 3 * 65536;  // three row groups, so three workers
// Overflows a 512 KiB stack in Debug; fits 8 MiB with room, even under ASan.
constexpr int kDeepTerms = 2000;

const char* const kTags[] = {"amber", "blue", "coral", "dune", "ember", "fern", "gold"};

// The SF=0.01 data directory, or "" when it hasn't been generated.
std::string FindDataDir() {
    for (const std::string dir : {"bench/tpch_data/sf0.01/", "../bench/tpch_data/sf0.01/", "../../bench/tpch_data/sf0.01/"}) {
        if (std::filesystem::exists(dir + "lineitem.tbl")) return dir;
    }
    return {};
}

// Row r of wide40: w_id, w_grp, w_amt, w_div, w_null, w_tag, w_dbl.
// w_dbl holds quarters, so its sums are exact in any order.
void WriteWide(const std::string& path) {
    std::string out;
    out.reserve(kWideRows * 40);
    char buf[128];
    for (size_t r = 0; r < kWideRows; ++r) {
        const size_t group = r / 65536;
        const unsigned cents = static_cast<unsigned>((r * 7919) % 100000);
        const unsigned div = group == kZeroGroup ? 0 : static_cast<unsigned>(1 + r % 5);
        const std::string null_field = r == kNullRow ? "\\N" : std::to_string(r % 50);
        std::snprintf(buf, sizeof(buf), "%zu|%zu|%u.%02u|%u|%s|%s|%.2f|\n", r, group, cents / 100, cents % 100, div, null_field.c_str(),
                      kTags[r % 7], static_cast<double>(r % 1000) * 0.25);
        out += buf;
    }
    std::ofstream(path, std::ios::binary) << out;
}

// Row r of side70: s_id = 31 r (so every row meets wide40), s_val, s_tag.
void WriteSide(const std::string& path) {
    std::string out;
    char buf[96];
    for (size_t r = 0; r < kSideRows; ++r) {
        const unsigned cents = static_cast<unsigned>((r * 104729) % 50000);
        std::snprintf(buf, sizeof(buf), "%zu|%u.%02u|%s|\n", r * 31, cents / 100, cents % 100, kTags[(r * 3) % 7]);
        out += buf;
    }
    std::ofstream(path, std::ios::binary) << out;
}

// Every tuple's bytes equal: the same cells, bit for bit.
::testing::AssertionResult SameBits(const QueryResult& got, const QueryResult& want) {
    if (got.schema.GetColumnCount() != want.schema.GetColumnCount()) return ::testing::AssertionFailure() << "column counts differ";
    for (uint32_t c = 0; c < want.schema.GetColumnCount(); ++c) {
        if (got.schema.GetColumn(c).type != want.schema.GetColumn(c).type) return ::testing::AssertionFailure() << "column " << c << " has another type";
    }
    if (got.tuples.size() != want.tuples.size()) {
        return ::testing::AssertionFailure() << got.tuples.size() << " rows, expected " << want.tuples.size();
    }
    for (size_t r = 0; r < want.tuples.size(); ++r) {
        const Tuple& a = got.tuples[r];
        const Tuple& b = want.tuples[r];
        if (a.GetLength() != b.GetLength() || std::memcmp(a.GetData(), b.GetData(), b.GetLength()) != 0) {
            auto failure = ::testing::AssertionFailure() << "row " << r << " differs:";
            for (uint32_t c = 0; c < want.schema.GetColumnCount(); ++c) {
                failure << " [" << a.GetValue(got.schema, c).ToString() << " vs " << b.GetValue(want.schema, c).ToString() << "]";
            }
            return failure;
        }
    }
    return ::testing::AssertionSuccess();
}

// Data shared by the tests of one process, loaded on first use.
class ParallelThreadsTest : public ::testing::Test {
protected:
    static std::unique_ptr<Database> db_;
    static std::unique_ptr<Database> plain_db_;  // TPC-H without indexes
    static std::string file_;
    static std::string plain_file_;
    static std::string data_dir_;
    static bool tpch_loaded_;
    static bool wide_loaded_;

    static void SetUpTestSuite() {
        const std::string pid = std::to_string(::getpid());
        file_ = (std::filesystem::temp_directory_path() / ("shilmandb_parallel_threads_test_" + pid + ".db")).string();
        plain_file_ = (std::filesystem::temp_directory_path() / ("shilmandb_parallel_threads_plain_" + pid + ".db")).string();
        data_dir_ = FindDataDir();
    }

    static void TearDownTestSuite() {
        db_.reset();
        plain_db_.reset();
        std::filesystem::remove(file_);
        std::filesystem::remove(plain_file_);
    }

    // A small pool, so many workers evict each other's pages.
    static Database& Db() {
        if (!db_) {
            std::filesystem::remove(file_);
            db_ = std::make_unique<Database>(file_, 512);
        }
        return *db_;
    }

    // TPC-H SF=0.01 into db, columnar, then the 8 indexes when asked, then stats.
    static void LoadTpch(Database& db, bool indexes) {
        LoadOptions options;
        options.storage = StorageKind::COLUMNAR;
        for (const auto& td : tpch::TableDescs()) db.LoadTable(td.name, td.schema, data_dir_ + td.tbl_file, options);
        auto* catalog = db.GetCatalog();
        if (indexes) {
            for (const auto& idx : tpch::IndexDescs()) (void)catalog->CreateIndex(idx.index_name, idx.table_name, idx.column_name);
        }
        for (const auto& td : tpch::TableDescs()) catalog->UpdateTableStats(td.name);
    }

    // TPC-H SF=0.01, columnar, with the 8 indexes; skips without the data.
    // The skip ends only this helper, so callers check IsSkipped() after it.
    void NeedTpch() {
        if (data_dir_.empty()) GTEST_SKIP() << "TPC-H SF=0.01 data not found. Run bench/tpch/generate_tpch_data.sh first.";
        if (tpch_loaded_) return;
        LoadTpch(Db(), true);
        tpch_loaded_ = true;
    }

    // TPC-H SF=0.01 without indexes, in its own database with the same small pool.
    // Skips without the data, so callers check IsSkipped() after it.
    void NeedPlainTpch() {
        if (data_dir_.empty()) GTEST_SKIP() << "TPC-H SF=0.01 data not found. Run bench/tpch/generate_tpch_data.sh first.";
        if (plain_db_) return;
        std::filesystem::remove(plain_file_);
        plain_db_ = std::make_unique<Database>(plain_file_, 512);
        LoadTpch(*plain_db_, false);
    }

    // wide40 (40 row groups) and side70 (two row groups).
    void NeedWide() {
        if (wide_loaded_) return;
        Database& db = Db();
        LoadOptions options;
        options.storage = StorageKind::COLUMNAR;
        options.null_text = "\\N";
        const std::string wide_path = file_ + "_wide40.tbl";
        const std::string side_path = file_ + "_side70.tbl";
        WriteWide(wide_path);
        WriteSide(side_path);
        db.LoadTable("wide40",
                     Schema({Column("w_id", ColumnType::BigInt()), Column("w_grp", ColumnType::Integer()), Column("w_amt", ColumnType::Numeric(12, 2)),
                             Column("w_div", ColumnType::Integer()), Column("w_null", ColumnType::Integer(), true), Column("w_tag", ColumnType::Varchar(8)),
                             Column("w_dbl", ColumnType::Double())}),
                     wide_path, options);
        db.LoadTable("side70", Schema({Column("s_id", ColumnType::BigInt()), Column("s_val", ColumnType::Numeric(10, 2)), Column("s_tag", ColumnType::Varchar(8))}),
                     side_path, options);
        std::filesystem::remove(wide_path);
        std::filesystem::remove(side_path);
        db.GetCatalog()->UpdateTableStats("wide40");
        db.GetCatalog()->UpdateTableStats("side70");
        wide_loaded_ = true;
    }

    // Runs sql in db's parallel executor on this many threads.
    static QueryResult Run(Database& db, const std::string& sql, unsigned threads) {
        QueryOptions options;
        options.mode = ExecutionMode::PARALLEL;
        options.threads = threads;
        return db.ExecuteSQL(sql, options);
    }

    // Runs sql in the shared database's parallel executor on this many threads.
    static QueryResult Run(const std::string& sql, unsigned threads) { return Run(Db(), sql, threads); }

    // Every query bit-identical at every thread count to its answer on one thread.
    static void ExpectSameAtEveryCount(Database& db, const std::string& tag, const std::string& sql) {
        SCOPED_TRACE(tag);
        const QueryResult one = Run(db, sql, 1);
        for (unsigned threads : kThreadCounts) {
            if (threads == 1) continue;
            const QueryResult many = Run(db, sql, threads);
            EXPECT_TRUE(SameBits(many, one)) << threads << " threads: " << sql;
            EXPECT_EQ(db.LastParallelStats().threads, threads);
        }
    }

    // After an error: no pin left, the mode off, the pool drops,
    // and both modes still answer, the parallel one on several workers.
    static void ExpectCleanAfterError() {
        BufferPoolManager* bpm = Db().GetBufferPoolManager();
        for (size_t f = 0; f < bpm->GetPoolSize(); ++f) {
            ASSERT_EQ(bpm->GetPage(static_cast<frame_id_t>(f)).GetPinCount(), 0) << "frame " << f << " stayed pinned";
        }
        EXPECT_FALSE(bpm->IsConcurrent());
        EXPECT_NO_THROW(bpm->DropAllPages());
        const QueryResult parallel = Run("SELECT COUNT(*), SUM(w_amt) FROM wide40 WHERE w_id < 2000000", 18);
        ASSERT_EQ(parallel.tuples.size(), 1u);
        EXPECT_EQ(parallel.tuples[0].GetValue(parallel.schema, 0).integer_, 2000000);
        EXPECT_GT(Db().LastParallelStats().morsels, 30u) << "the scan spans many row groups";
        const QueryResult tuple = Db().ExecuteSQL("SELECT COUNT(*) FROM wide40 WHERE w_id < 1000", ExecutionMode::TUPLE);
        ASSERT_EQ(tuple.tuples.size(), 1u);
        EXPECT_EQ(tuple.tuples[0].GetValue(tuple.schema, 0).integer_, 1000);
    }

    // One shard of the random queries: every kRandomShards-th seed from shard + 1.
    static void RandomShard(int shard);
};

std::unique_ptr<Database> ParallelThreadsTest::db_;
std::unique_ptr<Database> ParallelThreadsTest::plain_db_;
std::string ParallelThreadsTest::file_;
std::string ParallelThreadsTest::plain_file_;
std::string ParallelThreadsTest::data_dir_;
bool ParallelThreadsTest::tpch_loaded_ = false;
bool ParallelThreadsTest::wide_loaded_ = false;

// ---- Comparing random answers ----

// One cell: DECIMAL as a number, everything else exactly as text.
struct Cell {
    bool decimal{false};
    double number{0};
    std::string text;
};
using Row = std::vector<Cell>;

// Every row of a result as cells.
std::vector<Row> RowsOf(const QueryResult& r) {
    std::vector<Row> rows;
    rows.reserve(r.tuples.size());
    for (const Tuple& t : r.tuples) {
        Row row;
        for (uint32_t c = 0; c < r.schema.GetColumnCount(); ++c) {
            const Value v = t.GetValue(r.schema, c);
            Cell cell;
            cell.decimal = v.type_ == TypeId::DECIMAL;
            if (cell.decimal) {
                cell.number = v.decimal_;
            } else {
                cell.text = v.ToString();
            }
            row.push_back(std::move(cell));
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

// Text exactly; numbers within kDoubleTol relative, or equal (infinities too).
bool SameRow(const Row& a, const Row& b) {
    if (a.size() != b.size()) return false;
    for (size_t c = 0; c < a.size(); ++c) {
        if (a[c].decimal != b[c].decimal) return false;
        if (!a[c].decimal) {
            if (a[c].text != b[c].text) return false;
            continue;
        }
        const double x = a[c].number;
        const double y = b[c].number;
        if (x == y || (std::isnan(x) && std::isnan(y))) continue;
        if (!(std::abs(x - y) <= kDoubleTol * std::max({1.0, std::abs(x), std::abs(y)}))) return false;
    }
    return true;
}

// The row's text cells joined, so near-equal rows land in one bucket.
std::string TextKey(const Row& row) {
    std::string key;
    for (const Cell& c : row) {
        key += c.decimal ? std::string("#") : c.text;
        key += '\x1f';
    }
    return key;
}

// The row's cells exactly, each decimal by its bits.
std::string ExactKey(const Row& row) {
    std::string key;
    for (const Cell& c : row) {
        if (c.decimal) {
            char bits[sizeof(double)];
            std::memcpy(bits, &c.number, sizeof(double));
            key.append(bits, sizeof(double));
        } else {
            key += c.text;
        }
        key += '\x1f';
    }
    return key;
}

// Each row of got matches a different row of pool;
// with all, nothing is left over. Equal rows pair first,
// so a near-equal row can't take the row another one equals.
std::string Unmatched(const std::vector<Row>& got, const std::vector<Row>& pool, bool all) {
    std::multimap<std::string, size_t> equal;
    for (size_t i = 0; i < pool.size(); ++i) equal.emplace(ExactKey(pool[i]), i);
    std::vector<const Row*> rest;
    for (const Row& row : got) {
        auto it = equal.find(ExactKey(row));
        if (it == equal.end()) {
            rest.push_back(&row);
        } else {
            equal.erase(it);
        }
    }
    std::multimap<std::string, size_t> spare;
    for (const auto& entry : equal) spare.emplace(TextKey(pool[entry.second]), entry.second);
    for (const Row* row : rest) {
        auto range = spare.equal_range(TextKey(*row));
        auto it = range.first;
        while (it != range.second && !SameRow(*row, pool[it->second])) ++it;
        if (it == range.second) return "a row with no match";
        spare.erase(it);
    }
    if (all && !spare.empty()) return std::to_string(spare.size()) + " rows missing";
    return "";
}

// True when no row comes after the next one by the ORDER BY keys.
bool InKeyOrder(const QueryResult& r, const std::vector<std::pair<size_t, bool>>& keys) {
    for (size_t i = 1; i < r.tuples.size(); ++i) {
        for (const auto& [c, ascending] : keys) {
            const Value a = r.tuples[i - 1].GetValue(r.schema, static_cast<uint32_t>(c));
            const Value b = r.tuples[i].GetValue(r.schema, static_cast<uint32_t>(c));
            if (a < b) break;
            if (b < a) {
                if (ascending) return false;
                break;
            }
        }
    }
    return true;
}

// Every tuple's bytes, one string per row.
std::vector<std::string> RowBytes(const QueryResult& r) {
    std::vector<std::string> rows;
    rows.reserve(r.tuples.size());
    for (const Tuple& t : r.tuples) rows.emplace_back(t.GetData(), t.GetLength());
    return rows;
}

// Bit-equal rows: in the same order when ordered, else as a multiset;
// an unordered LIMIT's rows must each come from the unlimited answer.
std::string ExactDifference(const test::GeneratedQuery& q, const QueryResult& got, const QueryResult& one, const QueryResult* one_unlimited) {
    std::vector<std::string> g = RowBytes(got);
    std::vector<std::string> w = RowBytes(one);
    if (!q.keys.empty()) {
        for (size_t i = 0; i < g.size(); ++i) {
            if (g[i] != w[i]) return "row " + std::to_string(i) + " differs in its bits or order";
        }
        return "";
    }
    if (q.limit >= 0) w = RowBytes(*one_unlimited);
    std::multiset<std::string> pool(w.begin(), w.end());
    for (const std::string& row : g) {
        auto it = pool.find(row);
        if (it == pool.end()) return "a row with no bit-equal match";
        pool.erase(it);
    }
    if (q.limit < 0 && !pool.empty()) return std::to_string(pool.size()) + " rows missing";
    return "";
}

// How the 4-thread answer differs from the 1-thread one, or "".
// Only a query reading DOUBLE values may differ in last bits or near-tie order.
std::string Difference(const test::GeneratedQuery& q, const QueryResult& got, const QueryResult& one, const QueryResult* one_unlimited) {
    if (got.schema.GetColumnCount() != one.schema.GetColumnCount()) return "column counts differ";
    for (uint32_t c = 0; c < one.schema.GetColumnCount(); ++c) {
        if (got.schema.GetColumn(c).name != one.schema.GetColumn(c).name || got.schema.GetColumn(c).type != one.schema.GetColumn(c).type) {
            return "column " + std::to_string(c) + " differs";
        }
    }
    if (got.tuples.size() != one.tuples.size()) return std::to_string(got.tuples.size()) + " rows, expected " + std::to_string(one.tuples.size());
    if (!q.uses_double) return ExactDifference(q, got, one, one_unlimited);
    const std::vector<Row> g = RowsOf(got);
    const std::vector<Row> w = RowsOf(one);
    if (q.keys.empty()) {
        if (q.limit < 0) return Unmatched(g, w, true);
        return Unmatched(g, RowsOf(*one_unlimited), false);
    }
    bool same = true;
    for (size_t i = 0; i < g.size() && same; ++i) same = SameRow(g[i], w[i]);
    if (same) return "";
    // Noise in a DOUBLE key may swap near-ties:
    // then the rows must still be in order and valid.
    if (!InKeyOrder(got, q.keys)) return "rows out of order";
    if (q.limit < 0) return Unmatched(g, w, true);
    return Unmatched(g, RowsOf(*one_unlimited), false);
}

// What a query gave: its rows, or the text it threw.
struct Outcome {
    QueryResult result;
    std::string error;
    bool threw{false};
};

// Runs sql on a database in the parallel executor, catching what it throws.
Outcome RunCaught(Database& db, const std::string& sql, unsigned threads) {
    Outcome out;
    QueryOptions options;
    options.mode = ExecutionMode::PARALLEL;
    options.threads = threads;
    try {
        out.result = db.ExecuteSQL(sql, options);
    } catch (const std::exception& e) {
        out.threw = true;
        out.error = e.what();
    }
    return out;
}

void ParallelThreadsTest::RandomShard(int shard) {
    const std::string base = (std::filesystem::temp_directory_path() / ("shilmandb_parallel_threads_random_" + std::to_string(::getpid()))).string();
    int failures = 0;
    int compared = 0;
    int exact = 0;
    for (int seed = 1 + shard; seed <= kRandomSeeds; seed += kRandomShards) {
        test::Rng rng = test::SeedRng(seed);
        // Each seed gets a table of 2 or 3 row groups, so 4 threads use several workers.
        const std::vector<test::TableData> tables = test::SeedTables(rng, seed, true);
        std::filesystem::remove(base + ".db");
        {
            Database db(base + ".db", 4096);
            LoadOptions options;
            options.storage = StorageKind::COLUMNAR;
            options.threads = 4;
            for (const test::TableData& t : tables) {
                const std::string path = base + "_" + t.name + ".tbl";
                test::WriteTable(t, path);
                db.LoadTable(t.name, test::SchemaOf(t), path, options);
                std::filesystem::remove(path);
            }
            test::QueryGen gen(rng, tables);
            for (int n = 0; n < kRandomQueriesPerSeed; ++n) {
                const test::GeneratedQuery q = gen.Next();
                const std::string sql = q.Sql();
                const Outcome one = RunCaught(db, sql, 1);
                const Outcome four = RunCaught(db, sql, 4);
                std::string why;
                if (one.threw || four.threw) {
                    if (!one.threw) why = "4 threads threw: " + four.error;
                    else if (!four.threw) why = "4 threads didn't throw; 1 thread threw: " + one.error;
                    else if (four.error != one.error) why = "threw '" + four.error + "', 1 thread threw '" + one.error + "'";
                } else {
                    ++compared;
                    if (!q.uses_double) ++exact;
                    Outcome unlimited;
                    if (q.limit >= 0) unlimited = RunCaught(db, q.Sql(false), 1);
                    if (unlimited.threw) {
                        why = "the unlimited query threw: " + unlimited.error;
                    } else {
                        why = Difference(q, four.result, one.result, &unlimited.result);
                    }
                }
                if (!why.empty()) {
                    ++failures;
                    if (failures <= 20) ADD_FAILURE() << "seed " << seed << ", query " << n << ": " << sql << "\n  " << why;
                }
            }
        }
        std::filesystem::remove(base + ".db");
    }
    EXPECT_EQ(failures, 0);
    EXPECT_GT(compared, 0);
    EXPECT_GT(exact, 0) << "some queries are compared bit for bit";
    RecordProperty("compared", compared);
    RecordProperty("exact", exact);
}

}  // namespace

TEST_F(ParallelThreadsTest, SameAnswersAtEveryThreadCount) {
    NeedTpch();
    if (IsSkipped()) return;
    for (const auto& q : tpch::Queries()) ExpectSameAtEveryCount(Db(), q.tag, q.sql);
}

// Without indexes Q10 and Q14 plan seq scans, unlike on the indexed database.
TEST_F(ParallelThreadsTest, SameAnswersAtEveryThreadCountWithoutIndexes) {
    NeedPlainTpch();
    if (IsSkipped()) return;
    for (const auto& td : tpch::TableDescs()) ASSERT_TRUE(plain_db_->GetCatalog()->GetTableIndexes(td.name).empty()) << td.name;
    for (const auto& q : tpch::Queries()) ExpectSameAtEveryCount(*plain_db_, q.tag, q.sql);
}

TEST_F(ParallelThreadsTest, SameAnswersOnFortyRowGroups) {
    NeedWide();
    const std::vector<std::pair<std::string, std::string>> queries = {
        {"grouped", "SELECT w_grp, COUNT(*), SUM(w_amt), MIN(w_amt), MAX(w_tag), AVG(w_amt), SUM(w_dbl) FROM wide40 GROUP BY w_grp ORDER BY w_grp"},
        {"filtered", "SELECT w_tag, COUNT(*), SUM(w_amt), SUM(w_div), MIN(w_id) FROM wide40 WHERE w_amt > 500.00 GROUP BY w_tag ORDER BY w_tag"},
        {"join", "SELECT COUNT(*), SUM(s_val), SUM(w_amt), MIN(w_id), MAX(s_tag) FROM wide40 JOIN side70 ON w_id = s_id"},
        {"join grouped", "SELECT w_tag, s_tag, COUNT(*), SUM(s_val) FROM wide40 JOIN side70 ON w_id = s_id GROUP BY w_tag, s_tag ORDER BY w_tag, s_tag"},
        {"many groups", "SELECT w_id, COUNT(*), SUM(w_amt) FROM wide40 WHERE w_id < 200000 GROUP BY w_id ORDER BY w_id DESC LIMIT 25"},
        {"top n", "SELECT w_id, w_amt, w_tag FROM wide40 WHERE w_amt > 999.00 ORDER BY w_amt DESC, w_id LIMIT 30"},
    };
    for (const auto& [tag, sql] : queries) ExpectSameAtEveryCount(Db(), tag, sql);
}

TEST_F(ParallelThreadsTest, TinyTablesAtEighteenThreads) {
    NeedTpch();
    if (IsSkipped()) return;
    for (const std::string sql : {"SELECT * FROM region", "SELECT n_name, r_name FROM nation JOIN region ON n_regionkey = r_regionkey ORDER BY n_name",
                                  "SELECT r_name, COUNT(*) FROM nation JOIN region ON n_regionkey = r_regionkey GROUP BY r_name ORDER BY r_name"}) {
        SCOPED_TRACE(sql);
        const QueryResult one = Run(sql, 1);
        const QueryResult many = Run(sql, 18);
        EXPECT_TRUE(SameBits(many, one));
        EXPECT_FALSE(one.tuples.empty());
    }
}

TEST_F(ParallelThreadsTest, EmptyMorselsAndEmptyWorkers) {
    NeedWide();
    // w_grp - 7 = 0 can't skip a row group:
    // all 40 become morsels, and 39 keep nothing.
    for (const std::string sql : {"SELECT w_id, w_amt, w_tag FROM wide40 WHERE w_grp - 7 = 0 ORDER BY w_id",
                                  "SELECT COUNT(*), SUM(w_amt), MIN(w_id), MAX(w_tag) FROM wide40 WHERE w_grp - 7 = 0",
                                  "SELECT w_tag, COUNT(*), SUM(w_amt) FROM wide40 WHERE w_grp - 7 = 0 GROUP BY w_tag ORDER BY w_tag"}) {
        SCOPED_TRACE(sql);
        const QueryResult one = Run(sql, 1);
        const uint64_t one_morsels = Db().LastParallelStats().morsels;
        const QueryResult many = Run(sql, 18);
        // The scan's 40, plus the later pipelines' morsels.
        EXPECT_GE(Db().LastParallelStats().morsels, kWideGroups) << "every row group ran as a morsel";
        EXPECT_EQ(Db().LastParallelStats().morsels, one_morsels);
        EXPECT_TRUE(SameBits(many, one));
        EXPECT_FALSE(one.tuples.empty());
    }

    // A filter that keeps nothing: the aggregate's one zero-default row, and no groups.
    const QueryResult none = Run("SELECT COUNT(*), SUM(w_amt) FROM wide40 WHERE w_grp - 99 = 0", 18);
    ASSERT_EQ(none.tuples.size(), 1u);
    EXPECT_EQ(none.tuples[0].GetValue(none.schema, 0).integer_, 0);
    EXPECT_EQ(none.tuples[0].GetValue(none.schema, 1).decimal_, 0.0);
    EXPECT_TRUE(SameBits(none, Run("SELECT COUNT(*), SUM(w_amt) FROM wide40 WHERE w_grp - 99 = 0", 1)));
    EXPECT_TRUE(Run("SELECT w_tag, COUNT(*) FROM wide40 WHERE w_grp - 99 = 0 GROUP BY w_tag", 18).tuples.empty());
}

TEST_F(ParallelThreadsTest, ErrorMidQuery) {
    NeedWide();
    // Only row group 17's rows divide by zero.
    const std::string divide = "SELECT SUM(100 / w_div) FROM wide40";
    std::string one_error;
    try {
        (void)Run(divide, 1);
    } catch (const std::exception& e) {
        one_error = e.what();
    }
    EXPECT_EQ(one_error, "Division by zero");
    try {
        (void)Run(divide, 18);
        ADD_FAILURE() << "the division should throw";
    } catch (const DatabaseException& e) {
        EXPECT_EQ(std::string(e.what()), one_error);
    }
    ExpectCleanAfterError();

    // Only row group 23's w_null chunk holds a NULL.
    try {
        (void)Run("SELECT SUM(w_null) FROM wide40", 18);
        ADD_FAILURE() << "the NULL should throw";
    } catch (const DatabaseException& e) {
        EXPECT_EQ(std::string(e.what()), "column 'w_null' holds NULLs, which queries can't read yet");
    }
    ExpectCleanAfterError();
}

TEST_F(ParallelThreadsTest, LimitStopsEarly) {
    NeedTpch();
    if (IsSkipped()) return;
    NeedWide();
    const QueryResult wide = Run("SELECT w_id, w_grp FROM wide40 LIMIT 5", 18);
    EXPECT_EQ(wide.tuples.size(), 5u);
    EXPECT_LT(Db().LastParallelStats().morsels, kWideGroups) << "workers stop taking morsels at the limit";

    const QueryResult lineitem = Run("SELECT * FROM lineitem LIMIT 5", 18);
    EXPECT_EQ(lineitem.tuples.size(), 5u);
    const size_t groups = Db().GetCatalog()->GetTable("lineitem")->columnar->RowGroupCount();
    EXPECT_LE(Db().LastParallelStats().morsels, groups);
    EXPECT_EQ(Db().LastParallelStats().threads, 18u);
}

TEST_F(ParallelThreadsTest, DefaultThreadCountAnswers) {
    NeedWide();
    const std::string sql = "SELECT w_tag, COUNT(*), SUM(w_amt), MAX(w_id) FROM wide40 GROUP BY w_tag ORDER BY w_tag";
    const QueryResult one = Run(sql, 1);
    const QueryResult deflt = Run(sql, 0);
    EXPECT_EQ(Db().LastParallelStats().threads, parallel::ResolveThreads(0));
    EXPECT_TRUE(SameBits(deflt, one));
    EXPECT_EQ(one.tuples.size(), 7u);
}

// A 2,000-level sum recurses that deep on each worker's stack, the pool's threads included.
TEST_F(ParallelThreadsTest, DeepExpressionOnPoolThreads) {
    const std::string path = file_ + "_deep3.tbl";
    int32_t want = 0;
    {
        std::ofstream out(path, std::ios::binary);
        for (size_t r = 0; r < kDeepRows; ++r) {
            out << r << '|' << r % 50 << "|\n";
            if (r % 50 == 1) ++want;
        }
    }
    LoadOptions options;
    options.storage = StorageKind::COLUMNAR;
    Db().LoadTable("deep3", Schema({Column("d_a", ColumnType::BigInt()), Column("d_b", ColumnType::Integer())}), path, options);
    std::filesystem::remove(path);

    // d_b = 1 runs first, so the sum sees few rows but still recurses all the way.
    std::string sum = "d_a";
    for (int i = 1; i < kDeepTerms; ++i) sum += " + d_b";
    const std::string sql = "SELECT COUNT(*) FROM deep3 WHERE d_b = 1 AND " + sum + " > 0";
    const QueryResult one = Run(sql, 1);
    const QueryResult many = Run(sql, 18);
    EXPECT_EQ(Db().LastParallelStats().threads, 18u);
    EXPECT_GE(Db().LastParallelStats().morsels, 3u) << "each row group is a morsel";
    EXPECT_TRUE(SameBits(many, one));
    ASSERT_EQ(one.tuples.size(), 1u);
    EXPECT_EQ(one.tuples[0].GetValue(one.schema, 0).integer_, want);
}

TEST_F(ParallelThreadsTest, RandomQueriesAtFourThreads0) { RandomShard(0); }
TEST_F(ParallelThreadsTest, RandomQueriesAtFourThreads1) { RandomShard(1); }
TEST_F(ParallelThreadsTest, RandomQueriesAtFourThreads2) { RandomShard(2); }
TEST_F(ParallelThreadsTest, RandomQueriesAtFourThreads3) { RandomShard(3); }
TEST_F(ParallelThreadsTest, RandomQueriesAtFourThreads4) { RandomShard(4); }

}  // namespace shilmandb
