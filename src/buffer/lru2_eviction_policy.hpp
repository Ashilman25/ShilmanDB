#pragma once
#include "buffer/eviction_policy.hpp"
#include <cstdint>
#include <limits>
#include <map>
#include <tuple>
#include <vector>

namespace shilmandb {

// LRU-2
class Lru2EvictionPolicy : public EvictionPolicy {
public:
    explicit Lru2EvictionPolicy(size_t num_frames);

    using EvictionPolicy::Evict;

    // Throws: the buffer pool always passes the page
    void RecordAccess(frame_id_t frame_id) override;
    // A new use
    void RecordAccess(frame_id_t frame_id, page_id_t page_id) override;
    void RecordAccess(frame_id_t frame_id, page_id_t page_id, AccessType type) override;
    void SetEvictable(frame_id_t frame_id, bool evictable) override;

    [[nodiscard]] std::optional<frame_id_t> Evict() override;
    void Remove(frame_id_t frame_id) override;

    [[nodiscard]] size_t Size() const override;
    [[nodiscard]] bool UsesAccessTypes() const override { return true; }
    void Reset() override;

private:
    static constexpr int64_t kNone = -1;
    static constexpr frame_id_t kNoFrame = std::numeric_limits<frame_id_t>::max();

    using Key = std::tuple<int64_t, int64_t, page_id_t>;  // (second-to-last use, last use, page)
    using Order = std::map<Key, frame_id_t>;               // tracked pages, next victim first

    struct FrameState {
        page_id_t page{INVALID_PAGE_ID};
        Order::iterator pos;  // its node in order_, valid only while tracked
        bool tracked{false};
        bool evictable{false};
    };

    // Throws if the frame id is past the last frame
    void CheckRange(frame_id_t frame_id) const;

    // Updates the page's history for this access
    void Use(page_id_t page_id, AccessType type);

    [[nodiscard]] Key KeyOf(page_id_t page_id) const { return {second_[page_id], last_[page_id], page_id}; }

    std::vector<FrameState> frames_;     // one entry per frame, indexed by frame id
    std::vector<int64_t> last_, second_;  // per page, kNone if never used; grown on demand
    std::vector<frame_id_t> frame_of_;    // per page, the frame holding it or kNoFrame; grown with them
    Order order_;
    int64_t clock_{0};  // goes up on every access that isn't a repeat
    size_t evictable_count_{0};
};

}  // namespace shilmandb
