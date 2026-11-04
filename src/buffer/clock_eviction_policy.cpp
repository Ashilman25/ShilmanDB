#include "buffer/clock_eviction_policy.hpp"
#include "common/exception.hpp"
#include <algorithm>

namespace shilmandb {

ClockEvictionPolicy::ClockEvictionPolicy(size_t num_frames) : frames_(num_frames) {}

void ClockEvictionPolicy::CheckRange(frame_id_t frame_id) const {
    if (frame_id >= frames_.size()) {
        throw DatabaseException("frame id out of range");
    }
}

void ClockEvictionPolicy::RecordAccess(frame_id_t /*frame_id*/) {
    throw DatabaseException("CLOCK eviction needs the page id");
}

void ClockEvictionPolicy::RecordAccess(frame_id_t frame_id, page_id_t page_id) {
    RecordAccess(frame_id, page_id, AccessType::NEW_USE);
}

// A load and a hit both set the bit, so the page itself doesn't matter here
void ClockEvictionPolicy::RecordAccess(frame_id_t frame_id, page_id_t /*page_id*/, AccessType type) {
    CheckRange(frame_id);
    if (type == AccessType::REPEAT) { return; }

    FrameState& frame = frames_[frame_id];
    if (!frame.tracked) {
        frame.tracked = true;
        frame.evictable = true;
        ++evictable_count_;
    }
    frame.referenced = true;
}

void ClockEvictionPolicy::SetEvictable(frame_id_t frame_id, bool evictable) {
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

std::optional<frame_id_t> ClockEvictionPolicy::Evict() {
    if (evictable_count_ == 0) { return std::nullopt; }

    // The first sweep clears every evictable frame's bit, so two sweeps always find one
    const size_t n = frames_.size();
    for (size_t step = 0; step < 2 * n; ++step) {
        const size_t at = hand_;
        hand_ = at + 1 == n ? 0 : at + 1;

        FrameState& frame = frames_[at];
        if (!frame.tracked || !frame.evictable) { continue; }  // empty or pinned: bit left as it is
        if (frame.referenced) {
            frame.referenced = false;
            continue;
        }
        frame = FrameState{};
        --evictable_count_;
        return static_cast<frame_id_t>(at);
    }
    return std::nullopt;
}

void ClockEvictionPolicy::Remove(frame_id_t frame_id) {
    CheckRange(frame_id);
    FrameState& frame = frames_[frame_id];

    if (!frame.tracked) { return; }

    if (frame.evictable) {
        --evictable_count_;
    }
    frame = FrameState{};
}

size_t ClockEvictionPolicy::Size() const {
    return evictable_count_;
}

void ClockEvictionPolicy::Reset() {
    std::fill(frames_.begin(), frames_.end(), FrameState{});
    hand_ = 0;
    evictable_count_ = 0;
}

}  // namespace shilmandb
