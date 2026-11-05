#include "buffer/lru_eviction_policy.hpp"
#include "common/exception.hpp"
#include <algorithm>

namespace shilmandb {

LRUEvictionPolicy::LRUEvictionPolicy(size_t num_frames) : frames_(num_frames) {}

void LRUEvictionPolicy::CheckRange(frame_id_t frame_id) const {
    if (frame_id >= frames_.size()) {
        throw DatabaseException("frame id out of range");
    }
}

void LRUEvictionPolicy::RecordAccess(frame_id_t frame_id) {
    CheckRange(frame_id);
    FrameState& frame = frames_[frame_id];

    // already tracked, move its node to MRU (back) without allocating
    if (frame.tracked) {
        lru_list_.splice(lru_list_.end(), lru_list_, frame.pos);
        return;
    }

    // new frame, add to MRU
    lru_list_.push_back(frame_id);
    frame.pos = std::prev(lru_list_.end());
    frame.tracked = true;
    frame.evictable = true;
    ++evictable_count_;
}

void LRUEvictionPolicy::SetEvictable(frame_id_t frame_id, bool evictable) {
    CheckRange(frame_id);
    FrameState& frame = frames_[frame_id];

    // ignore frames not tracked
    if (!frame.tracked || frame.evictable == evictable) { return; }

    frame.evictable = evictable;
    if (evictable) {
        ++evictable_count_;
    } else {
        --evictable_count_;
    }
}

std::optional<frame_id_t> LRUEvictionPolicy::Evict() {
    for (auto it = lru_list_.begin(); it != lru_list_.end(); ++it) {
        FrameState& frame = frames_[*it];
        if (frame.evictable) {
            frame_id_t victim = *it;
            lru_list_.erase(it);
            frame.tracked = false;
            frame.evictable = false;
            --evictable_count_;
            return victim;
        }
    }
    return std::nullopt;
}

void LRUEvictionPolicy::Remove(frame_id_t frame_id) {
    CheckRange(frame_id);
    FrameState& frame = frames_[frame_id];

    if (!frame.tracked) { return; }

    if (frame.evictable) {
        --evictable_count_;
    }
    lru_list_.erase(frame.pos);
    frame.tracked = false;
    frame.evictable = false;
}

size_t LRUEvictionPolicy::Size() const {
    return evictable_count_;
}

void LRUEvictionPolicy::Reset() {
    lru_list_.clear();
    std::fill(frames_.begin(), frames_.end(), FrameState{});
    evictable_count_ = 0;
}

}  // namespace shilmandb
