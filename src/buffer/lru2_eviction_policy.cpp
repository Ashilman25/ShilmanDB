#include "buffer/lru2_eviction_policy.hpp"
#include "common/exception.hpp"
#include <algorithm>
#include <utility>

namespace shilmandb {

Lru2EvictionPolicy::Lru2EvictionPolicy(size_t num_frames) : frames_(num_frames) {}

void Lru2EvictionPolicy::CheckRange(frame_id_t frame_id) const {
    if (frame_id >= frames_.size()) {
        throw DatabaseException("frame id out of range");
    }
}

void Lru2EvictionPolicy::RecordAccess(frame_id_t /*frame_id*/) {
    throw DatabaseException("LRU-2 eviction needs the page id");
}

void Lru2EvictionPolicy::RecordAccess(frame_id_t frame_id, page_id_t page_id) {
    RecordAccess(frame_id, page_id, AccessType::NEW_USE);
}

void Lru2EvictionPolicy::RecordAccess(frame_id_t frame_id, page_id_t page_id, AccessType type) {
    CheckRange(frame_id);
    if (type == AccessType::REPEAT) { return; }
    if (page_id == INVALID_PAGE_ID) {
        throw DatabaseException("LRU-2 eviction needs a valid page id");
    }

    FrameState& frame = frames_[frame_id];
    const bool hit = frame.tracked && frame.page == page_id;

    // a page lives in one frame only
    if (!hit && page_id < frame_of_.size() && frame_of_[page_id] != kNoFrame) {
        throw DatabaseException("LRU-2 eviction: page is already in another frame");
    }

    if (page_id >= last_.size()) {
        last_.resize(static_cast<size_t>(page_id) + 1, kNone);
        second_.resize(static_cast<size_t>(page_id) + 1, kNone);
        frame_of_.resize(static_cast<size_t>(page_id) + 1, kNoFrame);
    }
    ++clock_;

    // a hit: move the page's node to its new place without allocating
    if (hit) {
        auto node = order_.extract(frame.pos);
        Use(page_id, type);
        node.key() = KeyOf(page_id);
        frame.pos = order_.insert(std::move(node)).position;
        return;
    }

    // a load is always a new use; a frame that held another page lets it go (its history stays)
    if (frame.tracked) {
        order_.erase(frame.pos);
        frame_of_[frame.page] = kNoFrame;
    } else {
        frame.tracked = true;
        frame.evictable = true;
        ++evictable_count_;
    }
    Use(page_id, AccessType::NEW_USE);
    frame.page = page_id;
    frame_of_[page_id] = frame_id;
    frame.pos = order_.emplace(KeyOf(page_id), frame_id).first;
}

void Lru2EvictionPolicy::Use(page_id_t page_id, AccessType type) {
    if (type != AccessType::TOUCH) {
        second_[page_id] = last_[page_id];  // a touch is not a second use
    }
    last_[page_id] = clock_;
}

void Lru2EvictionPolicy::SetEvictable(frame_id_t frame_id, bool evictable) {
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

std::optional<frame_id_t> Lru2EvictionPolicy::Evict() {
    if (evictable_count_ == 0) { return std::nullopt; }

    for (auto it = order_.begin(); it != order_.end(); ++it) {
        const frame_id_t victim = it->second;
        FrameState& frame = frames_[victim];
        if (!frame.evictable) { continue; }  // pinned

        order_.erase(it);
        frame_of_[frame.page] = kNoFrame;
        frame = FrameState{};
        --evictable_count_;
        return victim;
    }
    return std::nullopt;
}

void Lru2EvictionPolicy::Remove(frame_id_t frame_id) {
    CheckRange(frame_id);
    FrameState& frame = frames_[frame_id];

    if (!frame.tracked) { return; }

    if (frame.evictable) {
        --evictable_count_;
    }
    order_.erase(frame.pos);

    // a deleted page leaves no history behind
    last_[frame.page] = kNone;
    second_[frame.page] = kNone;
    frame_of_[frame.page] = kNoFrame;
    frame = FrameState{};
}

size_t Lru2EvictionPolicy::Size() const {
    return evictable_count_;
}

void Lru2EvictionPolicy::Reset() {
    order_.clear();
    std::fill(frames_.begin(), frames_.end(), FrameState{});
    last_.clear();
    second_.clear();
    frame_of_.clear();
    clock_ = 0;
    evictable_count_ = 0;
}

}  // namespace shilmandb
