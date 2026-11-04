#pragma once
#include "storage/page.hpp"
#include "storage/disk_manager.hpp"
#include "buffer/eviction_policy.hpp"
#include "buffer/timed_eviction_policy.hpp"
#include "common/config.hpp"
#include <memory>
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
    [[nodiscard]] uint64_t GetTraceCount() const { return access_counter_; }  // fetches since EnableTracing

    size_t GetPoolSize() const;
    const Page& GetPage(frame_id_t frame_id) const;

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

    bool FindVictimFrame(page_id_t incoming_page, frame_id_t* frame_id);
    void RecordPolicyAccess(frame_id_t frame_id, page_id_t page_id, bool is_miss);
    [[nodiscard]] AccessType NextAccessType(page_id_t page_id, frame_id_t frame_id, bool is_miss);
    void RecordVisit(page_id_t page_id, frame_id_t frame_id, bool is_miss);
    void TraceFetch(page_id_t page_id);
};

}  // namespace shilmandb
