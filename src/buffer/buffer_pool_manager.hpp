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

    // Access tracing for ML training data generation
    void EnableTracing(const std::string& trace_path);
    void DisableTracing();

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

    // Access tracing
    bool trace_enabled_{false};
    std::ofstream trace_file_;
    uint64_t access_counter_{0};

    bool FindVictimFrame(frame_id_t* frame_id);
    void RecordVisit(page_id_t page_id, frame_id_t frame_id, bool is_miss);
};

}  // namespace shilmandb
