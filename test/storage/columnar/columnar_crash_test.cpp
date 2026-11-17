#include <gtest/gtest.h>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "catalog/catalog.hpp"
#include "engine/database.hpp"
#include "executor/executor_context.hpp"
#include "executor/executor_factory.hpp"
#include "planner/plan_node.hpp"
#include "storage/columnar/superblock.hpp"
#include "storage/slotted_page.hpp"
#include "tpch/tpch_workload.hpp"

extern char** environ;

namespace shilmandb {
namespace {

constexpr size_t kPoolSize = 64;
constexpr size_t kIndexBuildPool = 8;   // small, so the tree outgrows it and pages leave before the flush
constexpr int64_t kIndexRows = 5000;    // a supplier-shaped table big enough for a two-level tree
constexpr size_t kMaxCrashPoints = 400;
constexpr int64_t kKey = 3;  // the s_nationkey and n_nationkey value the index queries look up
constexpr size_t kTornSplitFull = 4096;  // covers every slot field, so it equals the whole write
constexpr size_t kTornSplitHeader = 64;  // header, magic, version, page size and generation only
constexpr int kKillRuns = 40;
constexpr int kKillRounds = 1000;
constexpr uint64_t kHelperSetupGeneration = 5;  // create, nation, region, supplier, the index
constexpr const char* kNoData = "TPC-H SF=0.01 data not found. Run bench/tpch/generate_tpch_data.sh first.";

// The SF=0.01 data folder as an absolute path, or "" if missing.
std::string FindDataDir() {
    const std::vector<std::string> candidates = {
        "bench/tpch_data/sf0.01/",
        "../bench/tpch_data/sf0.01/",
        "../../bench/tpch_data/sf0.01/",
    };
    for (const auto& dir : candidates) {
        if (std::filesystem::exists(dir + "supplier.tbl")) return std::filesystem::absolute(dir).string();
    }
    return {};
}

// A temp file named with the tag and this process id.
std::string TempPath(const std::string& tag) {
    return (std::filesystem::temp_directory_path() /
            ("shilmandb_columnar_crash_" + tag + "_" + std::to_string(::getpid()) + ".db"))
        .string();
}

// Removes the file when it goes out of scope, even after a failed assert.
struct RemoveOnExit {
    std::string path;
    ~RemoveOnExit() { std::filesystem::remove(path); }
};

// The whole file.
std::vector<char> ReadBytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// Replaces the file with these bytes.
void WriteBytes(const std::string& path, const std::vector<char>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// The first max_lines lines of a .tbl file (all when 0).
std::vector<std::string> TblLines(const std::string& path, size_t max_lines = 0) {
    std::ifstream in(path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line) && (max_lines == 0 || lines.size() < max_lines)) {
        if (!line.empty()) lines.push_back(line);
    }
    return lines;
}

// How many lines hold key in the given field (0-based).
int64_t CountKey(const std::vector<std::string>& lines, size_t field, int64_t key) {
    int64_t count = 0;
    for (const std::string& line : lines) {
        size_t start = 0;
        for (size_t f = 0; f < field; ++f) start = line.find('|', start) + 1;
        if (std::stoll(line.substr(start, line.find('|', start) - start)) == key) ++count;
    }
    return count;
}

// Columnar storage on four loader threads, as a real load would use.
LoadOptions Columnar() {
    LoadOptions options;
    options.storage = StorageKind::COLUMNAR;
    options.threads = 4;
    return options;
}

// The single number a COUNT(*) query returns.
int64_t CountOf(Database& db, const std::string& sql) {
    const QueryResult result = db.ExecuteSQL(sql);
    return result.tuples.at(0).GetValue(result.schema, 0).integer_;
}

// Rows a hand-built index scan returns for one key; checks each row's key.
int64_t IndexCount(Database& db, const std::string& table, const std::string& index, const std::string& column,
                   const std::string& where) {
    const TableInfo* info = db.GetCatalog()->GetTable(table);
    const uint32_t col = info->schema.GetColumnIndex(column);
    auto plan = std::make_unique<IndexScanPlanNode>(info->schema, table, index);
    plan->low_key = Value(kKey);
    plan->high_key = Value(kKey);

    ExecutorContext ctx{db.GetBufferPoolManager(), db.GetCatalog()};
    auto exec = ExecutorFactory::CreateExecutor(plan.get(), &ctx, ExecutionMode::TUPLE);
    exec->Init();
    Tuple tuple;
    int64_t rows = 0;
    while (exec->Next(&tuple)) {
        ++rows;
        EXPECT_EQ(tuple.GetValue(info->schema, col).bigint_, kKey) << where;
    }
    exec->Close();
    return rows;
}

// Same type, then the same bytes in the member that type uses.
bool SameBits(const Value& a, const Value& b) {
    if (a.type_ != b.type_) return false;
    switch (a.type_) {
        case TypeId::INTEGER: return std::memcmp(&a.integer_, &b.integer_, sizeof(a.integer_)) == 0;
        case TypeId::BIGINT: return std::memcmp(&a.bigint_, &b.bigint_, sizeof(a.bigint_)) == 0;
        case TypeId::DECIMAL: return std::memcmp(&a.decimal_, &b.decimal_, sizeof(a.decimal_)) == 0;
        case TypeId::DATE: return std::memcmp(&a.date_, &b.date_, sizeof(a.date_)) == 0;
        case TypeId::VARCHAR: return a.varchar_ == b.varchar_;
        case TypeId::INVALID: return true;
    }
    return false;
}

// SELECT * from a row-store load of file, loaded once per table and file.
const QueryResult& RowStoreRows(const std::string& table, const Schema& schema, const std::string& file) {
    static std::map<std::string, QueryResult> loaded;
    const std::string key = table + "|" + file;
    if (auto it = loaded.find(key); it != loaded.end()) return it->second;

    const std::string path = TempPath("row_" + std::to_string(loaded.size()));
    std::filesystem::remove(path);
    const RemoveOnExit remove_path{path};
    QueryResult rows;
    {
        Database db(path, kPoolSize);
        db.LoadTable(table, schema, file);
        rows = db.ExecuteSQL("SELECT * FROM " + table);
    }
    return loaded.emplace(key, std::move(rows)).first->second;
}

// Checks two row lists match bit for bit, row by row, in order.
void ExpectSameTuples(const Schema& schema, const std::vector<Tuple>& want, const std::vector<Tuple>& got,
                      const std::string& what) {
    ASSERT_EQ(got.size(), want.size()) << what;
    for (size_t r = 0; r < want.size(); ++r) {
        for (uint32_t c = 0; c < schema.GetColumnCount(); ++c) {
            const Value x = want[r].GetValue(schema, c);
            const Value y = got[r].GetValue(schema, c);
            ASSERT_TRUE(SameBits(x, y)) << what << " row " << r << " column " << schema.GetColumn(c).name << " is "
                                        << y.ToString() << ", want " << x.ToString();
        }
    }
}

// Checks every row and column, in row order, against a row-store load of file.
void ExpectSameRows(Database& db, const std::string& table, const Schema& schema, const std::string& file,
                    const std::string& at) {
    ASSERT_FALSE(file.empty()) << at << ": no input file named for " << table;
    const QueryResult& want = RowStoreRows(table, schema, file);
    const QueryResult got = db.ExecuteSQL("SELECT * FROM " + table);
    ASSERT_EQ(got.schema.GetColumnCount(), want.schema.GetColumnCount()) << at << ": " << table;
    ExpectSameTuples(want.schema, want.tuples, got.tuples, at + ": " + table);
}

// Reads every entry of the index, all columns, and checks it against a row-store load.
// Equal keys sit in file order, so a stable sort by key matches the tree.
void ExpectSameIndexRows(Database& db, const std::string& table, const Schema& schema, const std::string& file,
                         const std::string& index, const std::string& column, const std::string& at) {
    ASSERT_FALSE(file.empty()) << at << ": no input file named for " << table;
    const QueryResult& loaded = RowStoreRows(table, schema, file);
    const uint32_t key = loaded.schema.GetColumnIndex(column);
    std::vector<Tuple> want = loaded.tuples;
    std::stable_sort(want.begin(), want.end(), [&](const Tuple& a, const Tuple& b) {
        return a.GetValue(loaded.schema, key) < b.GetValue(loaded.schema, key);
    });

    const TableInfo* info = db.GetCatalog()->GetTable(table);
    ASSERT_NE(info, nullptr) << at;
    const IndexScanPlanNode plan(info->schema, table, index);
    ExecutorContext ctx{db.GetBufferPoolManager(), db.GetCatalog()};
    auto exec = ExecutorFactory::CreateExecutor(&plan, &ctx, ExecutionMode::TUPLE);
    std::vector<Tuple> got;
    exec->Init();
    Tuple tuple;
    while (exec->Next(&tuple)) got.push_back(tuple);
    exec->Close();
    ExpectSameTuples(loaded.schema, want, got, at + ": " + index);
}

// What one committed version of the file holds.
struct Version {
    std::string name;
    std::string nation_file;     // the nation rows' input
    std::string supplier_file;   // the supplier rows' input, when there is a supplier table
    int64_t supplier_rows{0};    // 0 means no supplier table
    int64_t supplier_key{0};     // supplier rows with s_nationkey = kKey
    bool supplier_index{false};
    bool nation_index{false};
};

// Checks the open database shows exactly this version.
void ExpectVersion(Database& db, const Version& v, const std::string& where) {
    const std::string at = where + " (expecting the " + v.name + " version)";
    Catalog* catalog = db.GetCatalog();

    EXPECT_EQ(CountOf(db, "SELECT COUNT(*) FROM nation"), 25) << at;
    ExpectSameRows(db, "nation", tpch::NationSchema(), v.nation_file, at);
    if (v.nation_index) {
        ASSERT_NE(catalog->GetIndex("idx_nation_nationkey"), nullptr) << at;
        EXPECT_EQ(IndexCount(db, "nation", "idx_nation_nationkey", "n_nationkey", at), 1) << at;
        ExpectSameIndexRows(db, "nation", tpch::NationSchema(), v.nation_file, "idx_nation_nationkey", "n_nationkey", at);
        EXPECT_EQ(CountOf(db, "SELECT COUNT(*) FROM nation WHERE n_nationkey = " + std::to_string(kKey)), 1) << at;
    }

    if (v.supplier_rows == 0) {
        EXPECT_EQ(catalog->GetTable("supplier"), nullptr) << at;
        EXPECT_THROW((void)db.ExecuteSQL("SELECT COUNT(*) FROM supplier"), std::exception) << at;
        return;
    }
    ASSERT_NE(catalog->GetTable("supplier"), nullptr) << at;
    EXPECT_EQ(CountOf(db, "SELECT COUNT(*) FROM supplier"), v.supplier_rows) << at;
    EXPECT_EQ(CountOf(db, "SELECT COUNT(*) FROM supplier WHERE s_nationkey = " + std::to_string(kKey)), v.supplier_key) << at;
    if (v.supplier_index) {
        ASSERT_NE(catalog->GetIndex("idx_supplier_nationkey"), nullptr) << at;
        EXPECT_EQ(IndexCount(db, "supplier", "idx_supplier_nationkey", "s_nationkey", at), v.supplier_key) << at;
        ExpectSameIndexRows(db, "supplier", tpch::SupplierSchema(), v.supplier_file, "idx_supplier_nationkey",
                            "s_nationkey", at);
    } else {
        EXPECT_EQ(catalog->GetIndex("idx_supplier_nationkey"), nullptr) << at;
    }
    ExpectSameRows(db, "supplier", tpch::SupplierSchema(), v.supplier_file, at);
}

// One page write or sync, as the disk manager reported it.
struct Event {
    bool sync{false};
    page_id_t page{INVALID_PAGE_ID};
    std::vector<char> bytes;
};

// Logs every write with a copy of its bytes, and every sync; safe across threads.
class Recorder : public WriteObserver {
public:
    void OnWrite(page_id_t page_id, const char* data) override {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.push_back(Event{false, page_id, std::vector<char>(data, data + PAGE_SIZE)});
    }
    void OnSync() override {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.push_back(Event{true, INVALID_PAGE_ID, {}});
    }
    std::vector<Event> Take() {
        std::lock_guard<std::mutex> lock(mutex_);
        return std::move(events_);
    }

private:
    std::mutex mutex_;
    std::vector<Event> events_;
};

// A file a crash could leave: the snapshot plus some writes, the last maybe torn.
struct State {
    std::string kind;
    std::vector<size_t> writes;  // event indexes, applied in order
    size_t torn_bytes{0};        // new bytes of the last write; 0 means all of it
};

// Every crash point when they fit, else evenly spaced ones plus the must-haves.
std::vector<size_t> CrashPoints(size_t events, const std::set<size_t>& must) {
    std::set<size_t> points;
    for (size_t k : must) {
        if (k <= events) points.insert(k);
    }
    if (events + 1 <= kMaxCrashPoints) {
        for (size_t k = 0; k <= events; ++k) points.insert(k);
    } else {
        const size_t spaced = std::max<size_t>(kMaxCrashPoints - std::min(points.size(), kMaxCrashPoints), 2);
        for (size_t i = 0; i < spaced; ++i) points.insert(i * events / (spaced - 1));
    }
    return {points.begin(), points.end()};
}

// The state's file bytes: the snapshot with its writes laid over it in order.
std::vector<char> Build(const std::vector<char>& snapshot, const std::vector<Event>& events, const State& state) {
    std::vector<char> bytes = snapshot;
    for (size_t i = 0; i < state.writes.size(); ++i) {
        const Event& e = events[state.writes[i]];
        const size_t offset = static_cast<size_t>(e.page) * PAGE_SIZE;
        if (bytes.size() < offset + PAGE_SIZE) bytes.resize(offset + PAGE_SIZE, 0);
        const bool last = i + 1 == state.writes.size();
        const size_t n = last && state.torn_bytes != 0 ? state.torn_bytes : PAGE_SIZE;
        std::memcpy(bytes.data() + offset, e.bytes.data(), n);
    }
    return bytes;
}

// Runs setup, snapshots the file, records op, then opens every crash state.
void RunScenario(const std::string& tag, const std::function<void(Database&)>& setup,
                 const std::function<void(Database&)>& op, const Version& old_v, const Version& new_v,
                 size_t op_pool = kPoolSize) {
    const std::string work = TempPath(tag + "_work");
    std::filesystem::remove(work);
    const RemoveOnExit remove_work{work};
    {
        Database db(work, kPoolSize);
        setup(db);
        ExpectVersion(db, old_v, tag + " setup");
    }
    const std::vector<char> snapshot = ReadBytes(work);

    Recorder recorder;
    {
        Database db(work, op_pool);
        db.GetDiskManager()->SetWriteObserver(&recorder);
        op(db);
        db.GetDiskManager()->SetWriteObserver(nullptr);
        ExpectVersion(db, new_v, tag + " after the operation");
    }
    std::filesystem::remove(work);
    const std::vector<Event> events = recorder.Take();
    if (::testing::Test::HasFailure()) return;

    // One superblock write, a sync before it and a sync after it
    std::vector<size_t> syncs;
    std::vector<size_t> header_writes;
    size_t writes = 0;
    for (size_t i = 0; i < events.size(); ++i) {
        if (events[i].sync) {
            syncs.push_back(i);
        } else {
            ++writes;
            if (events[i].page < kHeaderPages) header_writes.push_back(i);
        }
    }
    ASSERT_EQ(header_writes.size(), 1u) << tag;
    const size_t sb = header_writes[0];
    const page_id_t slot = events[sb].page;
    ASSERT_EQ(DecodeSuperblock(events[sb].bytes.data(), slot).state, SlotState::VALID) << tag;
    const auto after = std::find_if(syncs.begin(), syncs.end(), [&](size_t s) { return s > sb; });
    ASSERT_NE(after, syncs.end()) << tag << ": no sync after the superblock write";
    ASSERT_NE(after, syncs.begin()) << tag << ": no sync before the superblock write";
    const size_t sb_sync = *after;

    std::set<size_t> must = {sb, sb + 1};
    for (size_t s : syncs) must.insert({s, s + 1});
    const std::vector<size_t> points = CrashPoints(events.size(), must);

    size_t built = 0;
    size_t shown_old = 0;
    size_t shown_new = 0;
    for (size_t k : points) {
        // The last sync before k; writes before it are on the disk for sure
        size_t synced = 0;
        for (size_t s : syncs) {
            if (s < k) synced = s;
        }
        std::vector<size_t> durable;
        std::vector<size_t> pending;
        for (size_t i = 0; i < k; ++i) {
            if (events[i].sync) continue;
            (i < synced ? durable : pending).push_back(i);
        }

        std::vector<State> states;
        states.push_back(State{"a", durable, 0});
        std::vector<size_t> all = durable;
        all.insert(all.end(), pending.begin(), pending.end());
        states.push_back(State{"b", all, 0});
        for (unsigned seed = 1; seed <= 3; ++seed) {
            std::mt19937 rng(seed * 7919u + static_cast<unsigned>(k));
            std::vector<size_t> some = durable;
            for (size_t i : pending) {
                if (rng() % 2 == 0) some.push_back(i);
            }
            states.push_back(State{"c" + std::to_string(seed), some, 0});
        }
        // The superblock write is the last one issued and isn't synced yet
        if (!all.empty() && all.back() == sb && k <= sb_sync) {
            states.push_back(State{"d1", all, kTornSplitFull});
            states.push_back(State{"d2", all, kTornSplitHeader});
        }

        for (const State& state : states) {
            const bool has_sb = std::find(state.writes.begin(), state.writes.end(), sb) != state.writes.end();
            const bool torn_header = state.kind == "d2";
            const Version& want = has_sb && !torn_header ? new_v : old_v;
            const std::string where = tag + " crash point " + std::to_string(k) + " of " +
                                      std::to_string(events.size()) + " state " + state.kind;
            if (k > sb_sync) EXPECT_EQ(&want, &new_v) << where << ": a state past the commit must hold it";
            if (k <= sb) EXPECT_EQ(&want, &old_v) << where << ": a state before the commit must not hold it";

            const std::vector<char> bytes = Build(snapshot, events, state);
            if (state.kind == "d1" || state.kind == "d2") {
                const SlotState torn = DecodeSuperblock(bytes.data() + static_cast<size_t>(slot) * PAGE_SIZE, slot).state;
                if (torn_header) {
                    EXPECT_NE(torn, SlotState::VALID) << where << ": the tear must break the checksum";
                } else {
                    EXPECT_EQ(torn, SlotState::VALID) << where;
                }
            }

            const std::string path = TempPath(tag + "_state_" + std::to_string(built));
            const RemoveOnExit remove_state{path};
            WriteBytes(path, bytes);
            try {
                Database db(path, kPoolSize);
                ExpectVersion(db, want, where);
            } catch (const std::exception& e) {
                ADD_FAILURE() << where << ": " << e.what();
            }
            ++built;
            ++(&want == &new_v ? shown_new : shown_old);
            if (::testing::Test::HasFailure()) return;
        }
    }

    std::cout << tag << ": " << events.size() << " events (" << writes << " writes, " << syncs.size()
              << " syncs), superblock write at " << sb << ", its sync at " << sb_sync << "; " << points.size()
              << " crash points, " << built << " states opened: " << shown_old << " old, " << shown_new
              << " new\n";
}

TEST(ColumnarCrashTestHelpers, CrashPointsAllWhenFewCappedWhenMany) {
    EXPECT_EQ(CrashPoints(12, {9, 10, 11, 13}).size(), 13u);

    const std::set<size_t> must = {4321, 4322, 9999};
    const std::vector<size_t> points = CrashPoints(10000, must);
    EXPECT_LE(points.size(), kMaxCrashPoints);
    EXPECT_GE(points.size(), kMaxCrashPoints - must.size());
    EXPECT_EQ(points.front(), 0u);
    EXPECT_EQ(points.back(), 10000u);
    for (size_t k : must) EXPECT_TRUE(std::binary_search(points.begin(), points.end(), k)) << k;
}

class ColumnarCrashTest : public ::testing::Test {
protected:
    void SetUp() override {
        data_dir_ = FindDataDir();
        if (data_dir_.empty()) GTEST_SKIP() << kNoData;
        supplier_ = TblLines(data_dir_ + "supplier.tbl");
    }

    std::string data_dir_;
    std::vector<std::string> supplier_;
};

TEST_F(ColumnarCrashTest, SimulatedCrashesDuringLoad) {
    Version old_v;
    old_v.name = "old";
    old_v.nation_file = data_dir_ + "nation.tbl";
    old_v.nation_index = true;
    Version new_v = old_v;
    new_v.name = "new";
    new_v.supplier_file = data_dir_ + "supplier.tbl";
    new_v.supplier_rows = static_cast<int64_t>(supplier_.size());
    new_v.supplier_key = CountKey(supplier_, 3, kKey);
    ASSERT_EQ(new_v.supplier_rows, 100);
    ASSERT_GT(new_v.supplier_key, 0);

    RunScenario(
        "load",
        [&](Database& db) {
            db.LoadTable("nation", tpch::NationSchema(), data_dir_ + "nation.tbl", Columnar());
            ASSERT_NE(db.GetCatalog()->CreateIndex("idx_nation_nationkey", "nation", "n_nationkey"), nullptr);
        },
        [&](Database& db) { db.LoadTable("supplier", tpch::SupplierSchema(), data_dir_ + "supplier.tbl", Columnar()); },
        old_v, new_v);
}

TEST_F(ColumnarCrashTest, SimulatedCrashesDuringReplace) {
    const std::vector<std::string> half = TblLines(data_dir_ + "supplier.tbl", 50);
    const std::string half_path = TempPath("supplier_half") + ".tbl";
    const RemoveOnExit remove_half{half_path};
    {
        std::ofstream out(half_path, std::ios::trunc);
        for (const std::string& line : half) out << line << '\n';
    }

    Version old_v;
    old_v.name = "old";
    old_v.nation_file = data_dir_ + "nation.tbl";
    old_v.supplier_file = data_dir_ + "supplier.tbl";
    old_v.supplier_rows = static_cast<int64_t>(supplier_.size());
    old_v.supplier_key = CountKey(supplier_, 3, kKey);
    old_v.supplier_index = true;
    Version new_v = old_v;
    new_v.name = "new";
    new_v.supplier_file = half_path;
    new_v.supplier_rows = static_cast<int64_t>(half.size());
    new_v.supplier_key = CountKey(half, 3, kKey);
    ASSERT_EQ(new_v.supplier_rows, 50);
    ASSERT_NE(old_v.supplier_rows, new_v.supplier_rows);
    ASSERT_NE(old_v.supplier_key, new_v.supplier_key);

    RunScenario(
        "replace",
        [&](Database& db) {
            db.LoadTable("nation", tpch::NationSchema(), data_dir_ + "nation.tbl", Columnar());
            db.LoadTable("supplier", tpch::SupplierSchema(), data_dir_ + "supplier.tbl", Columnar());
            ASSERT_NE(db.GetCatalog()->CreateIndex("idx_supplier_nationkey", "supplier", "s_nationkey"), nullptr);
        },
        [&](Database& db) { db.LoadTable("supplier", tpch::SupplierSchema(), half_path, Columnar()); }, old_v,
        new_v);
}

TEST_F(ColumnarCrashTest, SimulatedCrashesDuringIndexBuild) {
    // Supplier's lines over and over, sorted by s_nationkey so each leaf fills once
    const auto nation_of = [](const std::string& line) {
        const size_t start = line.find('|', line.find('|', line.find('|') + 1) + 1) + 1;
        return std::stoll(line.substr(start, line.find('|', start) - start));
    };
    std::vector<std::string> lines;
    for (size_t n = 0; n < static_cast<size_t>(kIndexRows); ++n) lines.push_back(supplier_[n % supplier_.size()]);
    std::stable_sort(lines.begin(), lines.end(),
                     [&](const std::string& a, const std::string& b) { return nation_of(a) < nation_of(b); });
    for (size_t n = 0; n < lines.size(); ++n) lines[n] = std::to_string(n + 1) + lines[n].substr(lines[n].find('|'));
    const std::string big_path = TempPath("supplier_big") + ".tbl";
    const RemoveOnExit remove_big{big_path};
    {
        std::ofstream out(big_path, std::ios::trunc);
        for (const std::string& line : lines) out << line << '\n';
    }

    Version old_v;
    old_v.name = "old";
    old_v.nation_file = data_dir_ + "nation.tbl";
    old_v.supplier_file = big_path;
    old_v.supplier_rows = kIndexRows;
    old_v.supplier_key = CountKey(lines, 3, kKey);
    Version new_v = old_v;
    new_v.name = "new";
    new_v.supplier_index = true;
    ASSERT_GT(old_v.supplier_key, 0);

    RunScenario(
        "index",
        [&](Database& db) {
            db.LoadTable("nation", tpch::NationSchema(), data_dir_ + "nation.tbl", Columnar());
            db.LoadTable("supplier", tpch::SupplierSchema(), big_path, Columnar());
        },
        [&](Database& db) {
            const page_id_t before = db.GetDiskManager()->GetNumPages();
            IndexInfo* index = db.GetCatalog()->CreateIndex("idx_supplier_nationkey", "supplier", "s_nationkey");
            ASSERT_NE(index, nullptr);

            // More new pages than frames, so tree pages were written before the flush
            const page_id_t added = db.GetDiskManager()->GetNumPages() - before;
            EXPECT_GT(added, kIndexBuildPool + 2) << "the tree must outgrow the pool";
            BufferPoolManager* bpm = db.GetBufferPoolManager();
            const page_id_t root = index->index->GetRootPageId();
            const bool internal = (SlottedPage::GetHeader(bpm->FetchPage(root)->GetData())->flags & 1u) == 0;
            (void)bpm->UnpinPage(root, false);
            EXPECT_TRUE(internal) << "the tree should have two levels or more";
            std::cout << "index: " << added << " new pages for the tree and catalog, pool of " << kIndexBuildPool
                      << " frames\n";
        },
        old_v, new_v, kIndexBuildPool);
}

// The generation of the slot an open would pick, or 0 if none is valid.
uint64_t WinningGeneration(const std::string& path) {
    const std::vector<char> bytes = ReadBytes(path);
    uint64_t best = 0;
    for (page_id_t slot = 0; slot < kHeaderPages; ++slot) {
        if (bytes.size() < static_cast<size_t>(slot + 1) * PAGE_SIZE) break;
        const SlotRead read = DecodeSuperblock(bytes.data() + static_cast<size_t>(slot) * PAGE_SIZE, slot);
        if (read.state == SlotState::VALID) best = std::max(best, read.block.generation);
    }
    return best;
}

// Reads from fd until text appears, EOF, or the timeout; returns whether text appeared.
bool ReadUntil(int fd, std::string& out, const std::string& text, int timeout_ms) {
    char buf[4096];
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (out.find(text) == std::string::npos) {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        if (left <= 0) return false;
        pollfd p{fd, POLLIN, 0};
        const int polled = ::poll(&p, 1, static_cast<int>(left));
        if (polled == 0 || (polled < 0 && errno == EINTR)) continue;
        if (polled < 0) return false;
        const ssize_t n = ::read(fd, buf, sizeof(buf));
        if (n <= 0) return false;
        out.append(buf, static_cast<size_t>(n));
    }
    return true;
}

// Reads whatever is left in fd until EOF.
void ReadToEnd(int fd, std::string& out) {
    char buf[4096];
    ssize_t n = 0;
    while ((n = ::read(fd, buf, sizeof(buf))) > 0) out.append(buf, static_cast<size_t>(n));
}

// The highest N in the "round N committed" lines, 0 if none.
int LastRound(const std::string& out) {
    int last = 0;
    const std::string word = "round ";
    for (size_t at = out.find(word); at != std::string::npos; at = out.find(word, at + 1)) {
        const size_t end = out.find(" committed", at);
        if (end == std::string::npos) break;
        last = std::max(last, std::stoi(out.substr(at + word.size(), end - at - word.size())));
    }
    return last;
}

TEST_F(ColumnarCrashTest, KillDuringReplaces) {
    // The helper loads the full file in even rounds, 50 lines in odd ones
    const std::string half_path = TempPath("kill_half") + ".tbl";
    const RemoveOnExit remove_half{half_path};
    {
        std::ofstream out(half_path, std::ios::trunc);
        for (const std::string& line : TblLines(data_dir_ + "supplier.tbl", 50)) out << line << '\n';
    }

    // The row store's answers for both files
    const std::string key_sql = "SELECT COUNT(*) FROM supplier WHERE s_nationkey = " + std::to_string(kKey);
    int64_t row_rows[2] = {0, 0};    // [0] the full file, [1] the 50-line copy
    int64_t row_answer[2] = {0, 0};
    const std::string sources[2] = {data_dir_ + "supplier.tbl", half_path};
    for (int v = 0; v < 2; ++v) {
        const std::string row_file = TempPath("kill_row_" + std::to_string(v));
        std::filesystem::remove(row_file);
        const RemoveOnExit remove_row{row_file};
        Database row(row_file, kPoolSize);
        row.LoadTable("supplier", tpch::SupplierSchema(), sources[v]);
        row_rows[v] = CountOf(row, "SELECT COUNT(*) FROM supplier");
        row_answer[v] = CountOf(row, key_sql);
    }
    ASSERT_EQ(row_rows[0], 100);
    ASSERT_EQ(row_rows[1], 50);
    ASSERT_EQ(row_answer[0], CountKey(supplier_, 3, kKey));
    ASSERT_NE(row_answer[0], row_answer[1]);

    std::mt19937 rng(20261117u);
    std::uniform_int_distribution<int> sleep_ms(5, 400);
    const std::vector<std::string> ranges = {"0", "1-4", "5-9", "10-19", "20+"};
    std::vector<int> landed(ranges.size(), 0);  // kills per range of the last committed round
    int shown_last = 0;
    int shown_next = 0;
    int shown_version[2] = {0, 0};

    for (int run = 0; run < kKillRuns; ++run) {
        const std::string where = "kill run " + std::to_string(run);
        const std::string file = TempPath("kill_" + std::to_string(run));
        std::filesystem::remove(file);
        const RemoveOnExit remove_file{file};
        const RemoveOnExit remove_helper_half{file + ".half.tbl"};  // the helper's copy, if a kill left it

        int fds[2];
        ASSERT_EQ(::pipe(fds), 0) << where;
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_adddup2(&actions, fds[1], STDOUT_FILENO);
        posix_spawn_file_actions_addclose(&actions, fds[0]);
        posix_spawn_file_actions_addclose(&actions, fds[1]);
        const std::string rounds = std::to_string(kKillRounds);
        std::vector<std::string> args = {CRASH_HELPER_PATH, file, data_dir_, rounds};
        std::vector<char*> argv;
        for (std::string& a : args) argv.push_back(a.data());
        argv.push_back(nullptr);
        pid_t pid = 0;
        const int spawned = posix_spawn(&pid, CRASH_HELPER_PATH, &actions, nullptr, argv.data(), environ);
        posix_spawn_file_actions_destroy(&actions);
        ::close(fds[1]);
        ASSERT_EQ(spawned, 0) << where << ": can't start " << CRASH_HELPER_PATH;

        std::string out;
        const bool ready = ReadUntil(fds[0], out, "ready\n", 60000);
        if (ready) std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms(rng)));
        ::kill(pid, SIGKILL);
        int status = 0;
        ::waitpid(pid, &status, 0);
        ReadToEnd(fds[0], out);
        ::close(fds[0]);
        ASSERT_TRUE(ready) << where << ": the helper never got ready; it printed: " << out;
        ASSERT_TRUE(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL)
            << where << ": the helper ended before the kill; it printed: " << out;

        // The open picks the last commit, at most one past the last printed round
        const int last = LastRound(out);
        const uint64_t generation = WinningGeneration(file);
        const uint64_t at_last = kHelperSetupGeneration + static_cast<uint64_t>(last);
        EXPECT_TRUE(generation == at_last || generation == at_last + 1)
            << where << ": generation " << generation << " after round " << last;
        ++(generation == at_last ? shown_last : shown_next);
        ++landed[last == 0 ? 0 : last < 5 ? 1 : last < 10 ? 2 : last < 20 ? 3 : 4];

        // Round 0 is the setup's full load; then odd rounds hold the copy
        ASSERT_GE(generation, kHelperSetupGeneration) << where;
        const uint64_t round = generation - kHelperSetupGeneration;
        const int v = static_cast<int>(round % 2);
        ++shown_version[v];
        const std::string at = where + " (round " + std::to_string(round) + ", " + (v == 0 ? "full" : "50-line") + " version)";

        try {
            Database db(file, kPoolSize);
            ASSERT_NE(db.GetCatalog()->GetIndex("idx_supplier_nationkey"), nullptr) << at;
            EXPECT_EQ(CountOf(db, "SELECT COUNT(*) FROM supplier"), row_rows[v]) << at;
            EXPECT_EQ(CountOf(db, key_sql), row_answer[v]) << at;
            EXPECT_EQ(IndexCount(db, "supplier", "idx_supplier_nationkey", "s_nationkey", at), row_answer[v]) << at;
            ExpectSameRows(db, "supplier", tpch::SupplierSchema(), sources[v], at);
            ExpectSameIndexRows(db, "supplier", tpch::SupplierSchema(), sources[v], "idx_supplier_nationkey",
                                "s_nationkey", at);
            ExpectSameRows(db, "nation", tpch::NationSchema(), data_dir_ + "nation.tbl", at);
        } catch (const std::exception& e) {
            ADD_FAILURE() << where << ": " << e.what();
        }
        if (HasFailure()) return;
    }

    std::cout << "kills by last committed round:";
    for (size_t i = 0; i < ranges.size(); ++i) std::cout << " " << ranges[i] << ": " << landed[i] << ";";
    std::cout << " files at the last printed round: " << shown_last << ", one round past it: " << shown_next
              << "; full version: " << shown_version[0] << ", 50-line version: " << shown_version[1] << "\n";
}

}  // namespace
}  // namespace shilmandb
