#pragma once
#include "storage/page.hpp"
#include "storage/disk_manager.hpp"
#include "buffer/eviction_policy.hpp"
#include "buffer/timed_eviction_policy.hpp"
#include "common/config.hpp"
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <list>
#include <fstream>
#include <cstdint>
#include <vector>

namespace shilmandb {

class BufferPoolManager {
public:
    // A page used again after more than this many page changes counts as a new visit
    static constexpr uint64_t kVisitWindow = 64;

    BufferPoolManager(size_t pool_size, DiskManager* disk_manager, std::unique_ptr<EvictionPolicy> eviction_policy);
    ~BufferPoolManager();

    BufferPoolManager(const BufferPoolManager&) = delete;
    BufferPoolManager& operator=(const BufferPoolManager&) = delete;

    [[nodiscard]] Page* FetchPage(page_id_t page_id);
    [[nodiscard]] bool UnpinPage(page_id_t page_id, bool is_dirty);
    [[nodiscard]] Page* NewPage(page_id_t* page_id);
    [[nodiscard]] bool DeletePage(page_id_t page_id);
    [[nodiscard]] bool FlushPage(page_id_t page_id);
    void FlushAllPages();

    // Pins pages ids[0..n) in order; returns how many, a prefix when frames run out.
    // A failed read leaves none pinned and rethrows; on, the batch takes two latch holds.
    [[nodiscard]] size_t FetchPages(const page_id_t* ids, size_t n, Page** out);
    // Unpins pages ids[0..n) clean, each as UnpinPage(id, false); on, in one latch hold.
    void UnpinPages(const page_id_t* ids, size_t n);

    // Writes out dirty pages and empties the pool, and the policy starts over: like a restart
    // with an empty cache. Throws DatabaseException if a page is pinned.
    void DropAllPages();

    // Stats for benchmarking
    uint64_t GetHitCount() const;
    uint64_t GetMissCount() const;
    // Runs of fetches to one page, so a scan's repeated pins of a page count once
    [[nodiscard]] uint64_t GetVisitCount() const;
    void ResetStats();

    // Times every eviction policy call while on. Off by default, and off leaves the
    // policy called directly. The counts stay readable after turning it off.
    void SetPolicyTiming(bool on);
    [[nodiscard]] PolicyStats GetPolicyStats() const;  // all zero if timing was never on

    // Access tracing for ML training data generation. Each line is "timestamp,page_id",
    // where timestamp is the fetch number since tracing was turned on. With collapse_repeats,
    // a fetch of the same page as the fetch just before it is left out (the numbering still
    // counts it).
    void EnableTracing(const std::string& trace_path, bool collapse_repeats = false);
    void DisableTracing();
    [[nodiscard]] uint64_t GetTraceCount() const {  // fetches since EnableTracing
        auto lock = LatchIfConcurrent();
        return access_counter_;
    }

    // Pages the file has handed out; every valid page id is below this
    [[nodiscard]] page_id_t GetNumPages() const { return disk_manager_->GetNumPages(); }

    size_t GetPoolSize() const;
    const Page& GetPage(frame_id_t frame_id) const;

    // On, every call takes one latch; workers says how many threads share the pool.
    // Switch it only from the calling thread, with no worker inside the pool.
    void SetConcurrent(bool on, unsigned workers = 1);
    [[nodiscard]] bool IsConcurrent() const { return concurrent_; }
    // Threads sharing the pool: the count given with the mode on, else 1
    [[nodiscard]] unsigned ConcurrentWorkers() const { return workers_; }
    // Time spent waiting for the latch; ResetStats() zeroes it
    [[nodiscard]] std::chrono::nanoseconds GetLatchWaitTime() const;

private:
    size_t pool_size_;
    Page* pages_;
    DiskManager* disk_manager_;
    std::unique_ptr<EvictionPolicy> eviction_policy_;
    EvictionPolicy* policy_;  // what every call goes through: the real policy, or the timer around it
    std::unique_ptr<TimedEvictionPolicy> timed_policy_;
    std::unordered_map<page_id_t, frame_id_t> page_table_;
    std::list<frame_id_t> free_list_;
    uint64_t hit_count_{0};
    uint64_t miss_count_{0};

    // Page visit tracking
    uint64_t visit_count_{0};
    uint64_t page_changes_{0};   // fetches whose page differs from the one before
    uint64_t epoch_start_{0};    // page_changes_ at the last ResetStats()
    page_id_t last_page_id_{INVALID_PAGE_ID};
    std::vector<uint64_t> last_seen_;  // per frame: page_changes_ at its last fetch, 0 if none

    // What a policy that counts visits is told. Kept apart from the visit stats above:
    // ResetStats() never touches it.
    bool uses_access_types_;
    uint64_t policy_run_{0};                       // runs of fetches to one page so far
    page_id_t policy_last_page_{INVALID_PAGE_ID};  // the page of the last fetch or NewPage
    std::vector<uint64_t> frame_run_;              // per frame: policy_run_ at its last fetch or NewPage

    // Access tracing
    bool trace_enabled_{false};
    std::ofstream trace_file_;
    uint64_t access_counter_{0};
    bool trace_collapse_{false};
    page_id_t last_traced_page_{INVALID_PAGE_ID};

    // Concurrent mode. Off, only the owner check runs, and ResetStats zeroes latch_wait_.
    bool concurrent_{false};
    unsigned workers_{1};
    std::thread::id owner_;                     // the thread that built the pool or last switched the mode
    mutable std::mutex latch_;
    mutable std::chrono::nanoseconds latch_wait_{0};
    std::condition_variable load_done_;         // woken when a load finishes, well or badly
    std::vector<bool> loading_;                 // per frame: its page is being read outside the latch

    // Off, every change must come from the owner thread (checked in Debug only)
    void AssertOwnerThread() const {
        assert((concurrent_ || std::this_thread::get_id() == owner_) && "buffer pool used from a second thread while not concurrent");
    }
    // Holds the latch only while the mode is on
    [[nodiscard]] std::unique_lock<std::mutex> LatchIfConcurrent() const {
        return concurrent_ ? Latch() : std::unique_lock<std::mutex>{};
    }
    [[nodiscard]] std::unique_lock<std::mutex> Latch() const;
    [[nodiscard]] std::unordered_map<page_id_t, frame_id_t>::iterator WaitUntilLoaded(std::unique_lock<std::mutex>& lock, page_id_t page_id);
    [[nodiscard]] Page* FetchPageConcurrent(page_id_t page_id);
    void AbandonLoad(page_id_t page_id, frame_id_t frame_id);
    [[nodiscard]] size_t FetchPagesConcurrent(const page_id_t* ids, size_t n, Page** out);
    void UnpinBatch(const page_id_t* ids, size_t n);
    void UnpinClean(page_id_t page_id);
    [[nodiscard]] bool FlushPageConcurrent(page_id_t page_id);
    void DropAllPagesConcurrent();
    void CloseTrace();
    void OpenTrace(const std::string& trace_path, bool collapse_repeats);

    bool FindVictimFrame(page_id_t incoming_page, frame_id_t* frame_id);
    void RecordPolicyAccess(frame_id_t frame_id, page_id_t page_id, bool is_miss);
    [[nodiscard]] AccessType NextAccessType(page_id_t page_id, frame_id_t frame_id, bool is_miss);
    void RecordVisit(page_id_t page_id, frame_id_t frame_id, bool is_miss);
    void TraceFetch(page_id_t page_id);
};

}  // namespace shilmandb
