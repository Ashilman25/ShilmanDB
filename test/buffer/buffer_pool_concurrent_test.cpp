#include <gtest/gtest.h>
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/eviction_policy_factory.hpp"
#include "common/exception.hpp"
#include "storage/columnar/encoding.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace shilmandb {
namespace {

// The policies the engine can run without a model
constexpr EvictionPolicyKind kKinds[] = {
    EvictionPolicyKind::LRU, EvictionPolicyKind::CLOCK, EvictionPolicyKind::LRU2,
    EvictionPolicyKind::ARC, EvictionPolicyKind::LIRS,
};

// One call the pool made on its policy
struct PolicyCall {
    enum class Kind : uint8_t { RECORD, RECORD_PAGE, RECORD_TYPED, SET_EVICTABLE, EVICT, EVICT_PAGE, REMOVE, RESET };
    Kind kind{Kind::RECORD};
    frame_id_t frame{0};
    page_id_t page{INVALID_PAGE_ID};
    int64_t detail{0};  // the access type, the evictable flag, or the victim (-1 for none)

    bool operator==(const PolicyCall& other) const {
        return kind == other.kind && frame == other.frame && page == other.page && detail == other.detail;
    }
};

std::ostream& operator<<(std::ostream& out, const PolicyCall& call) {
    return out << "{kind " << static_cast<int>(call.kind) << ", frame " << call.frame << ", page " << call.page
               << ", detail " << call.detail << "}";
}

// Forwards every call to a real policy and writes each one down
class RecordingPolicy : public EvictionPolicy {
public:
    using Kind = PolicyCall::Kind;

    explicit RecordingPolicy(std::unique_ptr<EvictionPolicy> inner) : inner_(std::move(inner)) {}

    void RecordAccess(frame_id_t frame_id) override {
        calls_.push_back({Kind::RECORD, frame_id, INVALID_PAGE_ID, 0});
        inner_->RecordAccess(frame_id);
    }
    void RecordAccess(frame_id_t frame_id, page_id_t page_id) override {
        calls_.push_back({Kind::RECORD_PAGE, frame_id, page_id, 0});
        inner_->RecordAccess(frame_id, page_id);
    }
    void RecordAccess(frame_id_t frame_id, page_id_t page_id, AccessType type) override {
        calls_.push_back({Kind::RECORD_TYPED, frame_id, page_id, static_cast<int64_t>(type)});
        inner_->RecordAccess(frame_id, page_id, type);
    }
    void SetEvictable(frame_id_t frame_id, bool evictable) override {
        calls_.push_back({Kind::SET_EVICTABLE, frame_id, INVALID_PAGE_ID, evictable ? 1 : 0});
        inner_->SetEvictable(frame_id, evictable);
    }
    std::optional<frame_id_t> Evict() override {
        const auto victim = inner_->Evict();
        calls_.push_back({Kind::EVICT, 0, INVALID_PAGE_ID, victim ? static_cast<int64_t>(*victim) : -1});
        return victim;
    }
    std::optional<frame_id_t> Evict(page_id_t incoming_page) override {
        const auto victim = inner_->Evict(incoming_page);
        calls_.push_back({Kind::EVICT_PAGE, 0, incoming_page, victim ? static_cast<int64_t>(*victim) : -1});
        return victim;
    }
    void Remove(frame_id_t frame_id) override {
        calls_.push_back({Kind::REMOVE, frame_id, INVALID_PAGE_ID, 0});
        inner_->Remove(frame_id);
    }
    size_t Size() const override { return inner_->Size(); }
    [[nodiscard]] bool UsesAccessTypes() const override { return inner_->UsesAccessTypes(); }
    void Reset() override {
        calls_.push_back({Kind::RESET, 0, INVALID_PAGE_ID, 0});
        inner_->Reset();
    }

    [[nodiscard]] const std::vector<PolicyCall>& Calls() const { return calls_; }

private:
    std::unique_ptr<EvictionPolicy> inner_;
    std::vector<PolicyCall> calls_;
};

// Holds threads until all of them have arrived, then lets them go together
class StartGate {
public:
    explicit StartGate(int count) : waiting_(count) {}

    void ArriveAndWait() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (--waiting_ == 0) {
            cv_.notify_all();
            return;
        }
        cv_.wait(lock, [this] { return waiting_ == 0; });
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    int waiting_;
};

// Aborts with a message if not done in time, so a deadlock fails, not hangs
class Watchdog {
public:
    Watchdog(std::chrono::seconds limit, std::string what)
        : thread_([this, limit, what = std::move(what)] {
              std::unique_lock<std::mutex> lock(mutex_);
              if (!cv_.wait_for(lock, limit, [this] { return done_; })) {
                  std::fprintf(stderr, "watchdog: %s did not finish in %lld s\n", what.c_str(),
                               static_cast<long long>(limit.count()));
                  std::abort();
              }
          }) {}

    ~Watchdog() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            done_ = true;
        }
        cv_.notify_one();
        thread_.join();
    }

    Watchdog(const Watchdog&) = delete;
    Watchdog& operator=(const Watchdog&) = delete;

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool done_{false};
    std::thread thread_;  // last, so the members above exist before it starts
};

// Writes count plain pages, each starting with its own page id
void WritePlainPages(const std::string& path, page_id_t count) {
    DiskManager dm(path);
    std::vector<char> page(PAGE_SIZE, 0);
    for (page_id_t pid = 0; pid < count; ++pid) {
        std::memcpy(page.data(), &pid, sizeof(pid));
        dm.WritePage(pid, page.data());
    }
}

constexpr size_t kPayloadStart = kPageHeaderSize;

// The payload byte at offset i of page pid
char PayloadByte(page_id_t pid, size_t i) {
    return static_cast<char>((pid * 131u + i * 7u) & 0xFFu);
}

// Writes count sealed columnar data pages; each payload names its page
void WriteColumnarPages(const std::string& path, page_id_t count) {
    DiskManager dm(path);
    std::vector<char> page(PAGE_SIZE, 0);
    for (page_id_t pid = 0; pid < count; ++pid) {
        ColumnarPageHeader header;
        header.kind = static_cast<uint8_t>(PageKind::DATA);
        header.encoding = static_cast<uint8_t>(Encoding::PLAIN);
        header.first_row = pid;
        std::memcpy(page.data(), &header, sizeof(header));
        std::memcpy(page.data() + kPayloadStart, &pid, sizeof(pid));
        for (size_t i = kPayloadStart + sizeof(pid); i < PAGE_SIZE; ++i) {
            page[i] = PayloadByte(pid, i);
        }
        SealPages(page.data(), 1, pid);
        dm.WritePage(pid, page.data());
    }
}

// True when the page's checksum holds for pid and its payload names pid
bool ColumnarPageIs(const char* page, page_id_t pid) {
    if (!VerifyPage(page, pid)) return false;
    page_id_t stored;
    std::memcpy(&stored, page + kPayloadStart, sizeof(stored));
    if (stored != pid) return false;
    for (size_t i = kPayloadStart + sizeof(pid); i < PAGE_SIZE; ++i) {
        if (page[i] != PayloadByte(pid, i)) return false;
    }
    return true;
}

std::string ReadFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// Everything a twin run produced
struct TwinRun {
    std::vector<PolicyCall> calls;
    std::vector<int64_t> results;  // what each operation returned
    uint64_t hits{0};
    uint64_t misses{0};
    uint64_t visits{0};
    std::string trace;
};

enum class TwinMode { OFF, ON, ON_THEN_OFF };

constexpr size_t kTwinFrames = 64;
constexpr page_id_t kTwinStartPages = 360;
constexpr page_id_t kTwinMaxPages = 400;
constexpr int kTwinOps = 200000;
constexpr size_t kTwinMaxPinned = 8;
constexpr uint32_t kTwinSeed = 20261118;

class BufferPoolConcurrentTest : public ::testing::Test {
protected:
    std::string db_file_;
    std::string twin_file_;
    std::string trace_file_;
    std::string twin_trace_file_;

    void SetUp() override {
        const auto dir = std::filesystem::temp_directory_path();
        const std::string pid = std::to_string(::getpid());
        db_file_ = (dir / ("shilmandb_bpm_concurrent_" + pid + ".db")).string();
        twin_file_ = (dir / ("shilmandb_bpm_concurrent_twin_" + pid + ".db")).string();
        trace_file_ = (dir / ("shilmandb_bpm_concurrent_" + pid + ".csv")).string();
        twin_trace_file_ = (dir / ("shilmandb_bpm_concurrent_twin_" + pid + ".csv")).string();
        RemoveFiles();
    }

    void TearDown() override { RemoveFiles(); }

    void RemoveFiles() {
        for (const auto& path : {db_file_, twin_file_, trace_file_, twin_trace_file_}) {
            std::filesystem::remove(path);
        }
    }

    // Runs one fixed random sequence of fetches, unpins, new pages and flushes
    static void RunTwin(const std::string& db_path, const std::string& trace_path, EvictionPolicyKind kind,
                        TwinMode mode, TwinRun* out) {
        std::filesystem::remove(db_path);
        WritePlainPages(db_path, kTwinStartPages);
        DiskManager dm(db_path);
        auto recorder = std::make_unique<RecordingPolicy>(MakeEvictionPolicy(kind, kTwinFrames));
        const RecordingPolicy* recording = recorder.get();
        BufferPoolManager bpm(kTwinFrames, &dm, std::move(recorder));

        if (mode == TwinMode::ON_THEN_OFF) {
            bpm.SetConcurrent(true);
            bpm.SetConcurrent(false);
        }
        if (mode == TwinMode::ON) {
            bpm.SetConcurrent(true);
            ASSERT_TRUE(bpm.IsConcurrent());
        }
        bpm.EnableTracing(trace_path);

        std::mt19937 rng(kTwinSeed);
        std::vector<page_id_t> pinned;
        const Page* first_frame = &bpm.GetPage(0);
        auto& results = out->results;

        // Hot pages half the time, so there are hits and repeats as well as misses
        auto pick_page = [&] {
            const page_id_t pages = dm.GetNumPages();
            return rng() % 2 == 0 ? static_cast<page_id_t>(rng() % 80) : static_cast<page_id_t>(rng() % pages);
        };

        for (int op = 0; op < kTwinOps; ++op) {
            const uint32_t roll = rng() % 100;
            if (roll < 2 && dm.GetNumPages() < kTwinMaxPages) {
                page_id_t pid;
                Page* page = bpm.NewPage(&pid);
                ASSERT_NE(page, nullptr) << "op " << op;
                std::memcpy(page->GetData(), &pid, sizeof(pid));
                results.push_back(pid);
                results.push_back(page - first_frame);
                results.push_back(bpm.UnpinPage(pid, true));
            } else if (roll < 4) {
                results.push_back(bpm.FlushPage(pick_page()));
            } else if (roll < 54 && pinned.size() < kTwinMaxPinned) {
                const page_id_t pid = rng() % 4 == 0 && !pinned.empty() ? pinned.back() : pick_page();
                Page* page = bpm.FetchPage(pid);
                ASSERT_NE(page, nullptr) << "op " << op;
                page_id_t stored;
                std::memcpy(&stored, page->GetData(), sizeof(stored));
                ASSERT_EQ(stored, pid) << "op " << op;
                results.push_back(page - first_frame);
                pinned.push_back(pid);
            } else if (!pinned.empty()) {
                const size_t index = rng() % pinned.size();
                const page_id_t pid = pinned[index];
                pinned.erase(pinned.begin() + static_cast<std::ptrdiff_t>(index));
                const bool dirty = rng() % 10 < 3;
                results.push_back(bpm.UnpinPage(pid, dirty));
            }
        }
        for (const page_id_t pid : pinned) {
            results.push_back(bpm.UnpinPage(pid, false));
        }

        out->hits = bpm.GetHitCount();
        out->misses = bpm.GetMissCount();
        out->visits = bpm.GetVisitCount();
        bpm.DisableTracing();
        bpm.SetConcurrent(false);
        out->calls = recording->Calls();
        out->trace = ReadFile(trace_path);
    }

    // Twin runs must match call for call, result for result, byte for byte
    static void ExpectSameRun(const TwinRun& plain, const TwinRun& other) {
        ASSERT_EQ(plain.calls.size(), other.calls.size());
        for (size_t i = 0; i < plain.calls.size(); ++i) {
            ASSERT_EQ(plain.calls[i], other.calls[i]) << "policy call " << i;
        }
        ASSERT_EQ(plain.results.size(), other.results.size());
        for (size_t i = 0; i < plain.results.size(); ++i) {
            ASSERT_EQ(plain.results[i], other.results[i]) << "result " << i;
        }
        EXPECT_EQ(plain.hits, other.hits);
        EXPECT_EQ(plain.misses, other.misses);
        EXPECT_EQ(plain.visits, other.visits);
        EXPECT_GT(plain.hits, 0u);
        EXPECT_GT(plain.misses, 0u);
        EXPECT_FALSE(plain.trace.empty());
        EXPECT_TRUE(plain.trace == other.trace) << "trace files differ";
    }
};

// One thread with the mode on makes the same policy calls, stats and trace
TEST_F(BufferPoolConcurrentTest, TwinSequencesDecideIdentically) {
    for (const auto kind : kKinds) {
        SCOPED_TRACE(std::string(EvictionPolicyKindName(kind)));
        TwinRun plain, concurrent;
        RunTwin(db_file_, trace_file_, kind, TwinMode::OFF, &plain);
        RunTwin(twin_file_, twin_trace_file_, kind, TwinMode::ON, &concurrent);
        ExpectSameRun(plain, concurrent);
    }
}

// Switching on and back off before any work changes nothing
TEST_F(BufferPoolConcurrentTest, SwitchedOnThenOffIsToday) {
    for (const auto kind : kKinds) {
        SCOPED_TRACE(std::string(EvictionPolicyKindName(kind)));
        TwinRun plain, switched;
        RunTwin(db_file_, trace_file_, kind, TwinMode::OFF, &plain);
        RunTwin(twin_file_, twin_trace_file_, kind, TwinMode::ON_THEN_OFF, &switched);
        ExpectSameRun(plain, switched);
    }
}

// Many threads fetch random pages through a small pool; every page they see is whole
TEST_F(BufferPoolConcurrentTest, StressManyThreads) {
    constexpr page_id_t kPages = 2000;
    constexpr size_t kFrames = 128;
    constexpr int kThreads = 16;
    constexpr auto kRunTime = std::chrono::seconds(3);
    WriteColumnarPages(db_file_, kPages);

    for (const auto kind : kKinds) {
        SCOPED_TRACE(std::string(EvictionPolicyKindName(kind)));
        DiskManager dm(db_file_);
        auto policy = MakeEvictionPolicy(kind, kFrames);
        const EvictionPolicy* raw_policy = policy.get();
        BufferPoolManager bpm(kFrames, &dm, std::move(policy));

        std::atomic<bool> stop{false};
        std::atomic<uint64_t> fetches{0}, bad_pages{0}, null_fetches{0}, failed_unpins{0};
        {
            Watchdog watchdog(std::chrono::seconds(30), "StressManyThreads");
            bpm.SetConcurrent(true);
            StartGate gate(kThreads);
            std::vector<std::thread> workers;
            for (int t = 0; t < kThreads; ++t) {
                workers.emplace_back([&, t] {
                    std::mt19937 rng(static_cast<uint32_t>(1000 + t));
                    gate.ArriveAndWait();
                    while (!stop.load(std::memory_order_relaxed)) {
                        const auto pid = static_cast<page_id_t>(rng() % kPages);
                        Page* page = bpm.FetchPage(pid);
                        if (page == nullptr) {
                            null_fetches.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                        fetches.fetch_add(1, std::memory_order_relaxed);
                        if (page->GetPageId() != pid || !ColumnarPageIs(page->GetData(), pid)) {
                            bad_pages.fetch_add(1, std::memory_order_relaxed);
                        } else {
                            page->MarkVerified();
                        }
                        const bool dirty = rng() % 16 == 0;
                        if (!bpm.UnpinPage(pid, dirty)) {
                            failed_unpins.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                });
            }
            std::this_thread::sleep_for(kRunTime);
            stop.store(true);
            for (auto& worker : workers) worker.join();
            bpm.SetConcurrent(false);
        }

        EXPECT_GT(fetches.load(), 0u);
        EXPECT_EQ(bad_pages.load(), 0u);
        EXPECT_EQ(null_fetches.load(), 0u);
        EXPECT_EQ(failed_unpins.load(), 0u);
        EXPECT_EQ(bpm.GetHitCount() + bpm.GetMissCount(), fetches.load());
        EXPECT_EQ(dm.GetReadStats().reads, bpm.GetMissCount());

        // Nothing pinned, and the frames holding a page are exactly the policy's evictable ones
        size_t resident = 0;
        for (size_t f = 0; f < kFrames; ++f) {
            const Page& page = bpm.GetPage(static_cast<frame_id_t>(f));
            EXPECT_EQ(page.GetPinCount(), 0) << "frame " << f;
            if (page.GetPageId() != INVALID_PAGE_ID) ++resident;
        }
        EXPECT_EQ(resident, raw_policy->Size());
        EXPECT_GE(bpm.GetLatchWaitTime().count(), 0);
        bpm.ResetStats();
        EXPECT_EQ(bpm.GetLatchWaitTime().count(), 0);
        EXPECT_NO_THROW(bpm.DropAllPages());
    }
}

// Threads that ask for a page while it loads wait for that one read
TEST_F(BufferPoolConcurrentTest, LoadingPageIsReadOnce) {
    constexpr int kThreads = 8;
    constexpr page_id_t kPage = 2;
    WritePlainPages(db_file_, 4);
    DiskManager dm(db_file_);
    BufferPoolManager bpm(16, &dm, MakeEvictionPolicy(EvictionPolicyKind::LRU, 16));
    dm.SetReadPenalty(std::chrono::milliseconds(40));
    const uint64_t reads_before = dm.GetReadStats().reads;

    std::vector<Page*> seen(kThreads, nullptr);
    std::atomic<int> wrong_data{0};
    {
        Watchdog watchdog(std::chrono::seconds(30), "LoadingPageIsReadOnce");
        bpm.SetConcurrent(true);
        StartGate fetch_gate(kThreads), unpin_gate(kThreads);
        std::vector<std::thread> workers;
        for (int t = 0; t < kThreads; ++t) {
            workers.emplace_back([&, t] {
                fetch_gate.ArriveAndWait();
                Page* page = bpm.FetchPage(kPage);
                seen[t] = page;
                page_id_t stored = INVALID_PAGE_ID;
                if (page != nullptr) std::memcpy(&stored, page->GetData(), sizeof(stored));
                if (stored != kPage) wrong_data.fetch_add(1);
                unpin_gate.ArriveAndWait();  // all hold the page at once
                if (page != nullptr) (void)bpm.UnpinPage(kPage, false);
            });
        }
        for (auto& worker : workers) worker.join();
        bpm.SetConcurrent(false);
    }
    dm.SetReadPenalty(std::chrono::microseconds(0));

    EXPECT_EQ(dm.GetReadStats().reads - reads_before, 1u);
    EXPECT_EQ(bpm.GetMissCount(), 1u);
    EXPECT_EQ(bpm.GetHitCount(), static_cast<uint64_t>(kThreads - 1));
    EXPECT_EQ(wrong_data.load(), 0);
    for (Page* page : seen) {
        ASSERT_NE(page, nullptr);
        EXPECT_EQ(page, seen[0]);
        EXPECT_EQ(page->GetPinCount(), 0);
    }
}

// A failed read reaches every thread that asked, and gives its frame back
TEST_F(BufferPoolConcurrentTest, FailedReadWakesWaiters) {
    constexpr int kThreads = 4;
    constexpr int kRounds = 200;
    constexpr size_t kFrames = 8;
    WritePlainPages(db_file_, kFrames);

    for (const auto kind : kKinds) {
        SCOPED_TRACE(std::string(EvictionPolicyKindName(kind)));
        DiskManager dm(db_file_);
        BufferPoolManager bpm(kFrames, &dm, MakeEvictionPolicy(kind, kFrames));

        std::atomic<int> thrown{0}, io_thrown{0};
        {
            Watchdog watchdog(std::chrono::seconds(30), "FailedReadWakesWaiters");
            bpm.SetConcurrent(true);
            for (int round = 0; round < kRounds; ++round) {
                StartGate gate(kThreads);
                std::vector<std::thread> workers;
                for (int t = 0; t < kThreads; ++t) {
                    workers.emplace_back([&] {
                        gate.ArriveAndWait();
                        try {
                            (void)bpm.FetchPage(INVALID_PAGE_ID);
                        } catch (const IOException&) {
                            io_thrown.fetch_add(1);
                            thrown.fetch_add(1);
                        } catch (const DatabaseException&) {
                            thrown.fetch_add(1);
                        }
                    });
                }
                for (auto& worker : workers) worker.join();
            }
            bpm.SetConcurrent(false);
        }

        EXPECT_EQ(thrown.load(), kThreads * kRounds);
        // ARC, LRU-2 and LIRS refuse the invalid page id in RecordAccess, before the read
        if (kind == EvictionPolicyKind::LRU || kind == EvictionPolicyKind::CLOCK) {
            EXPECT_EQ(io_thrown.load(), kThreads * kRounds);
        }
        EXPECT_EQ(bpm.GetMissCount(), 0u);
        EXPECT_EQ(bpm.GetHitCount(), 0u);
        EXPECT_EQ(bpm.GetVisitCount(), 0u);
        for (size_t f = 0; f < kFrames; ++f) {
            const Page& page = bpm.GetPage(static_cast<frame_id_t>(f));
            EXPECT_EQ(page.GetPinCount(), 0) << "frame " << f;
            EXPECT_EQ(page.GetPageId(), INVALID_PAGE_ID) << "frame " << f;
        }

        // Every frame is free: a full pool of real pages fits at once
        for (page_id_t pid = 0; pid < kFrames; ++pid) {
            ASSERT_NE(bpm.FetchPage(pid), nullptr) << "page " << pid;
        }
        for (page_id_t pid = 0; pid < kFrames; ++pid) {
            EXPECT_TRUE(bpm.UnpinPage(pid, false));
        }
    }
}

// DropAllPages with the mode on writes out dirty pages and empties the pool
TEST_F(BufferPoolConcurrentTest, DropAllPagesWhileConcurrent) {
    constexpr size_t kFrames = 8;
    constexpr page_id_t kPages = 12;
    for (const auto kind : kKinds) {
        SCOPED_TRACE(std::string(EvictionPolicyKindName(kind)));
        WritePlainPages(db_file_, kPages);
        DiskManager dm(db_file_);
        auto policy = MakeEvictionPolicy(kind, kFrames);
        const EvictionPolicy* raw_policy = policy.get();
        BufferPoolManager bpm(kFrames, &dm, std::move(policy));
        bpm.SetConcurrent(true);

        // Fill the pool, leaving one page pinned and every page dirty
        for (page_id_t pid = 0; pid < kFrames; ++pid) {
            Page* page = bpm.FetchPage(pid);
            ASSERT_NE(page, nullptr);
            page->GetData()[100] = 'd';
            if (pid != 0) ASSERT_TRUE(bpm.UnpinPage(pid, true));
        }
        EXPECT_THROW(bpm.DropAllPages(), DatabaseException);
        EXPECT_EQ(bpm.GetPage(0).GetPageId(), 0u);  // a refused drop leaves the pool alone
        ASSERT_TRUE(bpm.UnpinPage(0, true));

        ASSERT_NO_THROW(bpm.DropAllPages());
        for (size_t f = 0; f < kFrames; ++f) {
            EXPECT_EQ(bpm.GetPage(static_cast<frame_id_t>(f)).GetPageId(), INVALID_PAGE_ID) << "frame " << f;
        }
        EXPECT_EQ(raw_policy->Size(), 0u);

        // Fetches work after, in order from frame 0, and see the written-out bytes
        const uint64_t misses_before = bpm.GetMissCount();
        for (page_id_t pid = 0; pid < kFrames; ++pid) {
            Page* page = bpm.FetchPage(pid);
            ASSERT_NE(page, nullptr);
            EXPECT_EQ(page, &bpm.GetPage(static_cast<frame_id_t>(pid)));
            EXPECT_EQ(page->GetData()[100], 'd') << "page " << pid;
        }
        EXPECT_EQ(bpm.GetMissCount() - misses_before, kFrames);
        for (page_id_t pid = 0; pid < kFrames; ++pid) {
            EXPECT_TRUE(bpm.UnpinPage(pid, false));
        }
        bpm.SetConcurrent(false);
    }
}

}  // namespace
}  // namespace shilmandb
