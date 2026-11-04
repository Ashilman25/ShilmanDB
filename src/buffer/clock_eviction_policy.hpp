#pragma once
#include "buffer/eviction_policy.hpp"
#include <vector>

namespace shilmandb {

// CLOCK
class ClockEvictionPolicy : public EvictionPolicy {
public:
    explicit ClockEvictionPolicy(size_t num_frames);

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
    struct FrameState {
        bool referenced{false};
        bool tracked{false};
        bool evictable{false};
    };

    // Throws if the frame id is past the last frame
    void CheckRange(frame_id_t frame_id) const;

    std::vector<FrameState> frames_;  // one entry per frame, in the hand's order
    size_t hand_{0};
    size_t evictable_count_{0};
};

}  // namespace shilmandb
