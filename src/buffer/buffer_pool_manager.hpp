#pragma once
#include "storage/page.hpp"
#include "storage/disk_manager.hpp"
#include "buffer/eviction_policy.hpp"
#include "common/config.hpp"
#include <memory>
#include <unordered_map>
#include <list>
#include <fstream>
#include <cstdint>

namespace shilmandb {

class BufferPoolManager {
public:
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
    void ResetStats();

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
    std::unordered_map<page_id_t, frame_id_t> page_table_;
    std::list<frame_id_t> free_list_;
    uint64_t hit_count_{0};
    uint64_t miss_count_{0};

    // Access tracing
    bool trace_enabled_{false};
    std::ofstream trace_file_;
    uint64_t access_counter_{0};

    bool FindVictimFrame(frame_id_t* frame_id);
};

}  // namespace shilmandb
