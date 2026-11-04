#pragma once
#include "buffer/eviction_policy.hpp"
#include <list>
#include <vector>

namespace shilmandb {

class LRUEvictionPolicy : public EvictionPolicy {
public:
    explicit LRUEvictionPolicy(size_t num_frames);

    void RecordAccess(frame_id_t frame_id) override;
    void SetEvictable(frame_id_t frame_id, bool evictable) override;

    [[nodiscard]] std::optional<frame_id_t> Evict() override;
    void Remove(frame_id_t frame_id) override;

    [[nodiscard]] size_t Size() const override;

    void Reset() override;

private:
    struct FrameState {
        std::list<frame_id_t>::iterator pos;  // its node in lru_list_, valid only while tracked
        bool tracked{false};
        bool evictable{false};
    };

    // Throws if the frame id is past the last frame
    void CheckRange(frame_id_t frame_id) const;

    std::list<frame_id_t> lru_list_;  // front = LRU (evict first), back = MRU
    std::vector<FrameState> frames_;  // one entry per frame, indexed by frame id
    size_t evictable_count_{0};
};

}  // namespace shilmandb
