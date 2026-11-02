#include "buffer/timed_eviction_policy.hpp"
#include <algorithm>
#include <vector>

namespace shilmandb {

namespace {

using Clock = std::chrono::steady_clock;

// Counts one call and adds its duration to a total when it goes out of scope
class CallTimer {
public:
    CallTimer(uint64_t& calls, std::chrono::nanoseconds& total) : total_(total), start_(Clock::now()) {
        ++calls;
    }

    ~CallTimer() { total_ += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_); }

    CallTimer(const CallTimer&) = delete;
    CallTimer& operator=(const CallTimer&) = delete;

private:
    std::chrono::nanoseconds& total_;
    Clock::time_point start_;
};

// Cost of one clock read. The clock ticks about every 42 ns here, which is coarser than a
// single read, so one back-to-back pair usually shows 0. Each sample times a short run of
// reads and divides by their count; the median of the samples ignores the odd interruption.
std::chrono::nanoseconds MeasureTimerCost() {
    constexpr size_t kSamples = 1001;
    constexpr int kReadsPerSample = 32;
    std::vector<std::chrono::nanoseconds> samples(kSamples);
    
    for (auto& sample : samples) {
        const auto first = Clock::now();
        for (int i = 0; i < kReadsPerSample; ++i) {
            (void)Clock::now();
        }
        const auto last = Clock::now();
        // The run has one more gap between reads than it has reads in the middle
        sample = std::chrono::duration_cast<std::chrono::nanoseconds>(last - first) / (kReadsPerSample + 1);
    }
    std::nth_element(samples.begin(), samples.begin() + kSamples / 2, samples.end());
    return samples[kSamples / 2];
}

}  // namespace

TimedEvictionPolicy::TimedEvictionPolicy(EvictionPolicy* inner) : inner_(inner) {
    stats_.timer_cost = MeasureTimerCost();
}

void TimedEvictionPolicy::RecordAccess(frame_id_t frame_id) {
    CallTimer timer(stats_.record_calls, stats_.record_time);
    inner_->RecordAccess(frame_id);
}

void TimedEvictionPolicy::RecordAccess(frame_id_t frame_id, page_id_t page_id) {
    CallTimer timer(stats_.record_calls, stats_.record_time);
    inner_->RecordAccess(frame_id, page_id);
}

void TimedEvictionPolicy::SetEvictable(frame_id_t frame_id, bool evictable) {
    CallTimer timer(stats_.set_evictable_calls, stats_.set_evictable_time);
    inner_->SetEvictable(frame_id, evictable);
}

std::optional<frame_id_t> TimedEvictionPolicy::Evict() {
    CallTimer timer(stats_.evict_calls, stats_.evict_time);
    return inner_->Evict();
}

void TimedEvictionPolicy::Remove(frame_id_t frame_id) {
    CallTimer timer(stats_.remove_calls, stats_.remove_time);
    inner_->Remove(frame_id);
}

size_t TimedEvictionPolicy::Size() const { return inner_->Size(); }

const PolicyStats& TimedEvictionPolicy::GetStats() const { return stats_; }

void TimedEvictionPolicy::ResetStats() {
    const auto timer_cost = stats_.timer_cost;
    stats_ = PolicyStats{};
    stats_.timer_cost = timer_cost;
}

}  // namespace shilmandb
