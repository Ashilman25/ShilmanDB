#pragma once
#include "buffer/eviction_policy.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>

namespace shilmandb {

// Call counts and time spent inside an eviction policy.
// The times are raw: each one includes the cost of reading the clock around the call.
struct PolicyStats {
    uint64_t record_calls{0};
    uint64_t set_evictable_calls{0};
    uint64_t evict_calls{0};
    uint64_t remove_calls{0};
    std::chrono::nanoseconds record_time{0};
    std::chrono::nanoseconds set_evictable_time{0};
    std::chrono::nanoseconds evict_time{0};
    std::chrono::nanoseconds remove_time{0};
    std::chrono::nanoseconds timer_cost{0};  // median cost of reading the clock around one call

    [[nodiscard]] uint64_t TotalCalls() const {
        return record_calls + set_evictable_calls + evict_calls + remove_calls;
    }

    [[nodiscard]] std::chrono::nanoseconds RawTime() const {
        return record_time + set_evictable_time + evict_time + remove_time;
    }

    // Raw time minus the timer cost of every call, never below zero
    [[nodiscard]] std::chrono::nanoseconds NetTime() const {
        const auto overhead = timer_cost * static_cast<std::chrono::nanoseconds::rep>(TotalCalls());
        return std::max(std::chrono::nanoseconds{0}, RawTime() - overhead);
    }
};

// Wraps another policy and times its calls. It forwards everything unchanged,
// so the wrapped policy decides exactly what it would have decided alone.
class TimedEvictionPolicy : public EvictionPolicy {
public:
    // The wrapped policy is not owned and must outlive this object
    explicit TimedEvictionPolicy(EvictionPolicy* inner);

    void RecordAccess(frame_id_t frame_id) override;
    void RecordAccess(frame_id_t frame_id, page_id_t page_id) override;
    void RecordAccess(frame_id_t frame_id, page_id_t page_id, AccessType type) override;
    void SetEvictable(frame_id_t frame_id, bool evictable) override;

    [[nodiscard]] std::optional<frame_id_t> Evict() override;
    [[nodiscard]] std::optional<frame_id_t> Evict(page_id_t incoming_page) override;
    void Remove(frame_id_t frame_id) override;

    // Not timed: plain reads
    [[nodiscard]] size_t Size() const override;
    [[nodiscard]] bool UsesAccessTypes() const override;

    // Not timed: it only runs on a cold start, not per fetch
    void Reset() override;

    [[nodiscard]] const PolicyStats& GetStats() const;

    // Zeroes the counts and times, keeps the measured timer cost
    void ResetStats();

private:
    EvictionPolicy* inner_;
    PolicyStats stats_;
};

}  // namespace shilmandb
