#include <gtest/gtest.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "buffer/buffer_pool_manager.hpp"
#include "catalog/catalog.hpp"
#include "engine/database.hpp"
#include "executor/parallel/memory_budget.hpp"
#include "executor/parallel/temp_file.hpp"
#include "tpch/tpch_workload.hpp"

// Spilling end to end: the TPC-H queries with every breaker spilling, at the
// smallest budget the threads allow, and with temp writes failing partway.
namespace shilmandb {
namespace {

using parallel::kWorkerMinimumBytes;
using parallel::TempBlockHeader;
using parallel::TempFile;

constexpr unsigned kThreadCounts[] = {1, 2, 4, 18};
constexpr unsigned kFailureThreads = 18;
constexpr size_t kPoolPages = 4096;

// One scale factor's data directory ("0.01" or "0.1"), or "" without the data.
std::string FindDataDir(const std::string& sf) {
    for (const std::string up : {"", "../", "../../"}) {
        const std::string dir = up + "bench/tpch_data/sf" + sf + "/";
        if (std::filesystem::exists(dir + "lineitem.tbl")) return dir;
    }
    return {};
}

// A TPC-H shard: a scale factor and a thread count.
struct Shard {
    std::string sf;
    unsigned threads;
};

// A shard's name, like SF0_1_4Threads.
std::string NameOf(const Shard& shard) {
    std::string sf = shard.sf;
    for (char& ch : sf) {
        if (ch == '.') ch = '_';
    }
    return "SF" + sf + "_" + std::to_string(shard.threads) + "Threads";
}

// Shows a shard by its name, so ctest's test names match gtest's.
void PrintTo(const Shard& shard, std::ostream* os) { *os << NameOf(shard); }

// Every shard. SF=0.01's skip without data; SF=0.1's are listed only when its data is
// there at listing time, since CI generates only SF=0.01 and fails on any skip.
std::vector<Shard> Shards() {
    std::vector<Shard> shards;
    for (const std::string sf : {"0.01", "0.1"}) {
        if (sf != "0.01" && FindDataDir(sf).empty()) continue;
        for (const unsigned threads : kThreadCounts) shards.push_back(Shard{sf, threads});
    }
    return shards;
}

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

// Calls hook with each new temp file's path while alive; then writes work again.
class OpenHookScope {
public:
    explicit OpenHookScope(std::function<void(const std::string&)> hook) { TempFile::SetOpenHookForTest(std::move(hook)); }
    ~OpenHookScope() {
        TempFile::SetOpenHookForTest(nullptr);
        TempFile::FailWritesForTest(0);
    }
    OpenHookScope(const OpenHookScope&) = delete;
    OpenHookScope& operator=(const OpenHookScope&) = delete;
};

// A row's cells as text, for failure messages.
std::string Describe(const QueryResult& r, size_t row) {
    std::string out;
    for (uint32_t c = 0; c < r.schema.GetColumnCount(); ++c) out += r.tuples[row].GetValue(r.schema, c).ToString() + " | ";
    return out;
}

// "" when types and rows match in order, byte for byte; else the first difference.
// Every TPC-H query is ordered or one row, so the order is fixed.
std::string ExactDifference(const QueryResult& got, const QueryResult& want) {
    if (got.schema.GetColumnCount() != want.schema.GetColumnCount()) return "column counts differ";
    for (uint32_t c = 0; c < got.schema.GetColumnCount(); ++c) {
        if (got.schema.GetColumn(c).type != want.schema.GetColumn(c).type) return "column " + std::to_string(c) + " has another type";
    }
    if (got.tuples.size() != want.tuples.size()) return std::to_string(got.tuples.size()) + " rows, expected " + std::to_string(want.tuples.size());
    for (size_t i = 0; i < got.tuples.size(); ++i) {
        const Tuple& x = got.tuples[i];
        const Tuple& y = want.tuples[i];
        if (x.GetLength() != y.GetLength() || std::memcmp(x.GetData(), y.GetData(), x.GetLength()) != 0) {
            return "row " + std::to_string(i) + ": got " + Describe(got, i) + "expected " + Describe(want, i);
        }
    }
    return "";
}

// A parallel query's threads, memory limit (unset = the default) and spill switch.
struct RunSettings {
    unsigned threads{1};
    std::optional<size_t> memory_limit;
    bool spill_everything{false};
};

// Runs sql in the parallel executor with these settings.
QueryResult RunParallel(Database& db, const std::string& sql, const RunSettings& settings) {
    QueryOptions options;
    options.mode = ExecutionMode::PARALLEL;
    options.threads = settings.threads;
    options.memory_limit = settings.memory_limit;
    options.spill_everything = settings.spill_everything;
    return db.ExecuteSQL(sql, options);
}

// The temp directory has no entries, or doesn't exist; false if either can't be told.
bool TempDirEmpty(const Database& db) {
    std::error_code ec;
    const bool exists = std::filesystem::exists(db.GetTempDirectory(), ec);
    if (ec) return false;
    if (!exists) return true;
    const bool empty = std::filesystem::is_empty(db.GetTempDirectory(), ec);
    return !ec && empty;
}

// TPC-H loaded columnar, one database per scale factor and index setting,
// shared by the tests of one process and loaded on first use.
class ParallelSpillTest : public ::testing::TestWithParam<Shard> {
protected:
    static std::map<std::string, std::unique_ptr<Database>> dbs_;
    static std::string base_;
    Database* db_{nullptr};

    static void SetUpTestSuite() {
        base_ = (std::filesystem::temp_directory_path() / ("shilmandb_parallel_spill_test_" + std::to_string(::getpid()))).string();
    }

    // Closes every database before its file and temp directory go.
    static void TearDownTestSuite() {
        for (auto& [key, db] : dbs_) {
            const std::string file = base_ + "_" + key + ".db";
            const std::string temp_dir = db->GetTempDirectory();
            db.reset();
            std::filesystem::remove(file);
            std::filesystem::remove_all(temp_dir);
        }
        dbs_.clear();
    }

    // The database for sf with or without the 8 indexes, or nullptr without the data.
    static Database* Db(const std::string& sf, bool indexes) {
        const std::string data_dir = FindDataDir(sf);
        if (data_dir.empty()) return nullptr;
        const std::string key = "sf" + sf + (indexes ? "_indexed" : "_plain");
        auto& slot = dbs_[key];
        if (slot) return slot.get();
        const std::string file = base_ + "_" + key + ".db";
        std::filesystem::remove(file);
        slot = std::make_unique<Database>(file, kPoolPages);
        slot->SetTempDirectory(file + "_spill");
        std::filesystem::remove_all(slot->GetTempDirectory());
        LoadOptions options;
        options.storage = StorageKind::COLUMNAR;
        for (const auto& td : tpch::TableDescs()) slot->LoadTable(td.name, td.schema, data_dir + td.tbl_file, options);
        auto* catalog = slot->GetCatalog();
        if (indexes) {
            for (const auto& idx : tpch::IndexDescs()) (void)catalog->CreateIndex(idx.index_name, idx.table_name, idx.column_name);
        }
        for (const auto& td : tpch::TableDescs()) catalog->UpdateTableStats(td.name);
        return slot.get();
    }

    // This shard's database with or without the indexes. Skips without the data:
    // callers check IsSkipped().
    void NeedShardDb(bool indexes) {
        db_ = Db(GetParam().sf, indexes);
        if (db_ == nullptr) GTEST_SKIP() << "TPC-H SF=" << GetParam().sf << " data not found. Run bench/tpch/generate_tpch_data.sh first.";
    }
};

std::map<std::string, std::unique_ptr<Database>> ParallelSpillTest::dbs_;
std::string ParallelSpillTest::base_;

INSTANTIATE_TEST_SUITE_P(Shards, ParallelSpillTest, ::testing::ValuesIn(Shards()),
                         [](const ::testing::TestParamInfo<Shard>& info) { return NameOf(info.param); });

TEST_P(ParallelSpillTest, TpchSpillEverythingIsByteIdentical) {
    const unsigned threads = GetParam().threads;
    for (const bool indexes : {false, true}) {
        NeedShardDb(indexes);
        if (IsSkipped()) return;
        for (const auto& q : tpch::Queries()) {
            SCOPED_TRACE(q.tag + (indexes ? " with indexes" : " without indexes"));
            RunSettings settings;
            settings.threads = threads;
            const QueryResult want = RunParallel(*db_, q.sql, settings);
            ASSERT_EQ(db_->LastParallelStats().spill_files, 0u) << "the default limit spills nothing";
            ASSERT_FALSE(want.tuples.empty());

            settings.spill_everything = true;
            const QueryResult got = RunParallel(*db_, q.sql, settings);
            const ParallelQueryStats stats = db_->LastParallelStats();
            EXPECT_EQ(stats.threads, threads);
            EXPECT_EQ(ExactDifference(got, want), "");
            // Every query has a breaker, and each breaker reads back what it wrote.
            EXPECT_GT(stats.spill_files, 0u);
            EXPECT_GT(stats.spill_bytes_written, 0u);
            EXPECT_GT(stats.spill_bytes_read, 0u);
        }
        EXPECT_TRUE(TempDirEmpty(*db_));
    }
}

TEST_P(ParallelSpillTest, TinyBudgetMatches) {
    NeedShardDb(false);
    if (IsSkipped()) return;
    const unsigned threads = GetParam().threads;
    const size_t floors = threads * kWorkerMinimumBytes;
    for (const auto& q : tpch::Queries()) {
        SCOPED_TRACE(q.tag);
        RunSettings settings;
        settings.threads = threads;
        const QueryResult want = RunParallel(*db_, q.sql, settings);

        settings.memory_limit = floors;
        const QueryResult got = RunParallel(*db_, q.sql, settings);
        const ParallelQueryStats stats = db_->LastParallelStats();
        EXPECT_EQ(ExactDifference(got, want), "");
        EXPECT_EQ(stats.memory_limit, floors);
        EXPECT_LE(stats.peak_tracked_bytes, floors);
        // With only the floors there's no shared pool, so state a breaker keeps for
        // the next pipeline can't leave its floor and is spilled: every query spills.
        const char* const floors_rule = "relies on the floors-only rule: with no shared room, kept state spills at Finalize";
        EXPECT_GT(stats.spill_files, 0u) << floors_rule;
        EXPECT_GT(stats.spill_bytes_written, 0u) << floors_rule;
        RecordProperty(q.tag + "_spill_bytes_written", std::to_string(stats.spill_bytes_written));
    }
    EXPECT_TRUE(TempDirEmpty(*db_));
}

// After a failed spill: the pool's mode off, no page pinned, the pool drops,
// the temp directory empty and no descriptor left open.
void ExpectNothingLeft(Database& db, size_t descriptors, const std::string& label) {
    BufferPoolManager* bpm = db.GetBufferPoolManager();
    EXPECT_FALSE(bpm->IsConcurrent()) << label;
    for (size_t f = 0; f < bpm->GetPoolSize(); ++f) {
        ASSERT_EQ(bpm->GetPage(static_cast<frame_id_t>(f)).GetPinCount(), 0) << label << ": frame " << f << " stayed pinned";
    }
    EXPECT_NO_THROW(bpm->DropAllPages()) << label;
    EXPECT_TRUE(TempDirEmpty(db)) << label;
    EXPECT_EQ(OpenDescriptors(), descriptors) << label;
}

// Runs sql, returning what it threw, or "" when it answered (into *result if given).
std::string RunCaught(Database& db, const std::string& sql, const RunSettings& settings, QueryResult* result = nullptr) {
    try {
        QueryResult r = RunParallel(db, sql, settings);
        if (result != nullptr) *result = std::move(r);
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

// Flips the checksum's first byte in every block of the temp file open on fd.
void DamageEveryBlock(int fd) {
    const off_t size = ::lseek(fd, 0, SEEK_END);
    off_t at = 0;
    while (at + static_cast<off_t>(sizeof(TempBlockHeader)) <= size) {
        TempBlockHeader header{};
        ASSERT_EQ(::pread(fd, &header, sizeof(header), at), static_cast<ssize_t>(sizeof(header)));
        header.crc ^= 0xFFu;
        ASSERT_EQ(::pwrite(fd, &header, sizeof(header), at), static_cast<ssize_t>(sizeof(header)));
        at += static_cast<off_t>(sizeof(TempBlockHeader) + header.payload_bytes);
    }
}

TEST_F(ParallelSpillTest, FailureMidSpillLeavesNothing) {
    const std::string sf = FindDataDir("0.1").empty() ? "0.01" : "0.1";
    Database* db = Db(sf, false);
    if (db == nullptr) GTEST_SKIP() << "TPC-H SF=0.01 data not found. Run bench/tpch/generate_tpch_data.sh first.";
    const std::string no_space = "out of temp space in " + db->GetTempDirectory() + " while spilling: " + std::strerror(ENOSPC);
    const std::string tuple_sql = "SELECT n_name, r_name FROM nation JOIN region ON n_regionkey = r_regionkey ORDER BY n_name";
    const QueryResult tuple_want = db->ExecuteSQL(tuple_sql, ExecutionMode::TUPLE);
    ASSERT_EQ(tuple_want.tuples.size(), 25u);

    // Q3's joins spill, so their spill phase runs; Q1 spills only in aggregate and sort.
    for (const std::string tag : {"Q3", "Q1"}) {
        const std::string sql = tpch::FindQuery(tag)->sql;
        RunSettings settings;
        settings.threads = kFailureThreads;
        const QueryResult want = RunParallel(*db, sql, settings);
        settings.spill_everything = true;

        // Every write fails, from the first; then writes fail from the n-th file opened on.
        int threw = 0;
        for (const int fail_from : {0, 1, 4, 16}) {
            const std::string label = "SF=" + sf + " " + tag + (fail_from == 0 ? ", every write failing" : ", writes failing from file " + std::to_string(fail_from));
            const size_t descriptors = OpenDescriptors();
            ASSERT_GT(descriptors, 0u) << "can't list /dev/fd";
            std::atomic<int> opened{0};
            std::string error;
            QueryResult got;
            {
                OpenHookScope hook([&opened, fail_from](const std::string&) {
                    if (opened.fetch_add(1) + 1 == fail_from) TempFile::FailWritesForTest(ENOSPC);
                });
                std::optional<FailWritesScope> fail;
                if (fail_from == 0) fail.emplace(ENOSPC);
                error = RunCaught(*db, sql, settings, &got);
            }
            // Once writes are armed a write follows the file's opening, so it must fail.
            const bool armed = opened.load() >= std::max(fail_from, 1);
            if (armed) EXPECT_FALSE(error.empty()) << label << ": " << opened.load() << " files opened, yet no write failed";
            if (!error.empty()) {
                ++threw;
                EXPECT_EQ(error, no_space) << label;
            } else {
                EXPECT_EQ(ExactDifference(got, want), "") << label << ": no write failed, so it must answer";
            }
            ExpectNothingLeft(*db, descriptors, label);

            // The same database answers again: spilling everything, then in tuple mode.
            const QueryResult again = RunParallel(*db, sql, settings);
            EXPECT_EQ(ExactDifference(again, want), "") << label;
            EXPECT_GT(db->LastParallelStats().spill_files, 0u) << label;
            EXPECT_EQ(ExactDifference(db->ExecuteSQL(tuple_sql, ExecutionMode::TUPLE), tuple_want), "") << label;
        }
        EXPECT_GE(threw, 2) << tag << ": failing from the start, and from the first file, always throws";
        RecordProperty(tag + "_runs_that_threw", threw);
    }

    // A block damaged once written: Q3 on one thread, so the first file opened
    // (the first join's build) is finished when the second one opens.
    const std::string q3 = tpch::FindQuery("Q3")->sql;
    RunSettings one;
    one.threads = 1;
    const QueryResult want = RunParallel(*db, q3, one);
    one.spill_everything = true;
    const size_t descriptors = OpenDescriptors();
    int first_fd = -1;
    int opened = 0;
    std::string error;
    {
        OpenHookScope hook([&first_fd, &opened](const std::string& path) {
            if (++opened == 1) first_fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
            if (opened == 2 && first_fd >= 0) DamageEveryBlock(first_fd);
        });
        error = RunCaught(*db, q3, one);
    }
    ASSERT_GE(first_fd, 0) << "couldn't open the first temp file";
    ::close(first_fd);
    EXPECT_GE(opened, 2);
    EXPECT_EQ(error, "damaged temp file: checksum mismatch");
    ExpectNothingLeft(*db, descriptors, "damaged block");
    EXPECT_EQ(ExactDifference(RunParallel(*db, q3, one), want), "") << "after the damaged block";
}

}  // namespace
}  // namespace shilmandb
