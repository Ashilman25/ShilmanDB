#include "buffer/lirs_eviction_policy.hpp"
#include "common/exception.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>

namespace shilmandb {

namespace {

// All but 1% of the pool (rounded, at least one frame) is for LIR pages
size_t LirLimit(size_t num_frames) {
    if (num_frames < 2) {
        throw DatabaseException("LIRS eviction needs a pool of at least 2 frames");
    }
    const long hir = std::max(1L, std::lround(0.01 * static_cast<double>(num_frames)));
    return num_frames - static_cast<size_t>(hir);
}

}  // namespace

LirsEvictionPolicy::LirsEvictionPolicy(size_t num_frames) : lir_limit_(LirLimit(num_frames)), frames_(num_frames) {}

void LirsEvictionPolicy::CheckRange(frame_id_t frame_id) const {
    if (frame_id >= frames_.size()) {
        throw DatabaseException("frame id out of range");
    }
}

void LirsEvictionPolicy::PushBack(page_id_t page_id, ListId list) {
    Ends& ends = ends_[list];
    Links& links = pages_[page_id].links[list];
    links.prev = ends.tail;
    links.next = kNone;

    if (ends.tail == kNone) {
        ends.head = page_id;
    } else {
        pages_[ends.tail].links[list].next = page_id;
    }
    ends.tail = page_id;
    ++ends.size;
}

void LirsEvictionPolicy::Unlink(page_id_t page_id, ListId list) {
    Ends& ends = ends_[list];
    Links& links = pages_[page_id].links[list];

    if (links.prev == kNone) {
        ends.head = links.next;
    } else {
        pages_[links.prev].links[list].next = links.next;
    }

    if (links.next == kNone) {
        ends.tail = links.prev;
    } else {
        pages_[links.next].links[list].prev = links.prev;
    }
    links = Links{};
    --ends.size;
}

void LirsEvictionPolicy::MoveToTop(page_id_t page_id) {
    PageState& page = pages_[page_id];
    if (page.in_s) {
        Unlink(page_id, kS);
    }
    PushBack(page_id, kS);
    page.in_s = true;
}

void LirsEvictionPolicy::Prune() {
    while (ends_[kS].head != kNone && !pages_[ends_[kS].head].lir) {
        const page_id_t bottom = ends_[kS].head;
        Unlink(bottom, kS);
        pages_[bottom].in_s = false;
    }
}

void LirsEvictionPolicy::DemoteBottom() {
    const page_id_t bottom = ends_[kS].head;
    PageState& page = pages_[bottom];
    assert(page.lir);

    Unlink(bottom, kS);
    page.in_s = false;
    page.lir = false;
    --lir_count_;
    PushBack(bottom, kQ);
    Prune();
}

void LirsEvictionPolicy::Touch(page_id_t page_id) {
    if (pages_[page_id].lir) {
        MoveToTop(page_id);
        Prune();
    } else {
        Unlink(page_id, kQ);  // S stays as it is
        PushBack(page_id, kQ);
    }
}

void LirsEvictionPolicy::Hit(page_id_t page_id) {
    PageState& page = pages_[page_id];
    if (page.lir) {
        MoveToTop(page_id);
        Prune();
    } else if (page.in_s) {
        // a HIR page back while still in S: it becomes LIR, and S's bottom page HIR
        MoveToTop(page_id);
        Unlink(page_id, kQ);
        page.lir = true;
        ++lir_count_;
        DemoteBottom();
    } else {
        MoveToTop(page_id);
        Unlink(page_id, kQ);
        PushBack(page_id, kQ);
    }
}

void LirsEvictionPolicy::Load(page_id_t page_id) {
    PageState& page = pages_[page_id];
    const bool in_s = page.in_s;
    MoveToTop(page_id);

    if (lir_count_ < lir_limit_) {
        page.lir = true;
        ++lir_count_;
    } else if (in_s) {
        // evicted, but back while still in S
        page.lir = true;
        ++lir_count_;
        DemoteBottom();
    } else {
        PushBack(page_id, kQ);
    }
}

void LirsEvictionPolicy::Unload(page_id_t page_id) {
    PageState& page = pages_[page_id];
    page.frame = kNoFrame;
    
    if (!page.lir) {
        Unlink(page_id, kQ);
        return;
    }
    page.lir = false;
    --lir_count_;
    Unlink(page_id, kS);
    page.in_s = false;
    Prune();
}

frame_id_t LirsEvictionPolicy::EvictPage(page_id_t page_id) {
    const frame_id_t frame_id = pages_[page_id].frame;
    Unload(page_id);
    frames_[frame_id] = FrameState{};
    --evictable_count_;
    return frame_id;
}

void LirsEvictionPolicy::RecordAccess(frame_id_t /*frame_id*/) {
    throw DatabaseException("LIRS eviction needs the page id");
}

void LirsEvictionPolicy::RecordAccess(frame_id_t frame_id, page_id_t page_id) {
    RecordAccess(frame_id, page_id, AccessType::NEW_USE);
}

void LirsEvictionPolicy::RecordAccess(frame_id_t frame_id, page_id_t page_id, AccessType type) {
    CheckRange(frame_id);
    if (type == AccessType::REPEAT) { return; }
    if (page_id == INVALID_PAGE_ID) {
        throw DatabaseException("LIRS eviction needs a valid page id");
    }

    FrameState& frame = frames_[frame_id];
    if (frame.page == page_id) {
        if (type == AccessType::TOUCH) {
            Touch(page_id);
        } else {
            Hit(page_id);
        }
        return;
    }

    // a load, always a new use
    if (page_id < pages_.size() && pages_[page_id].frame != kNoFrame) {
        throw DatabaseException("LIRS eviction: page is already in another frame");
    }
    if (page_id >= pages_.size()) {
        pages_.resize(static_cast<size_t>(page_id) + 1);
    }

    // a frame that still held another page lets it go, as if evicted, and keeps its pin
    if (frame.page != INVALID_PAGE_ID) {
        Unload(frame.page);
    } else {
        frame.evictable = true;
        ++evictable_count_;
    }
    frame.page = page_id;
    pages_[page_id].frame = frame_id;
    Load(page_id);
}

void LirsEvictionPolicy::SetEvictable(frame_id_t frame_id, bool evictable) {
    CheckRange(frame_id);
    FrameState& frame = frames_[frame_id];

    // ignore frames not tracked
    if (frame.page == INVALID_PAGE_ID || frame.evictable == evictable) { return; }

    frame.evictable = evictable;
    if (evictable) {
        ++evictable_count_;
    } else {
        --evictable_count_;
    }
}

std::optional<frame_id_t> LirsEvictionPolicy::Evict() {
    if (evictable_count_ == 0) { return std::nullopt; }

    // Q's oldest evictable page; it keeps its place in S
    for (page_id_t page_id = ends_[kQ].head; page_id != kNone; page_id = pages_[page_id].links[kQ].next) {
        if (frames_[pages_[page_id].frame].evictable) { return EvictPage(page_id); }
    }

    // every resident HIR page is pinned: the evictable LIR page nearest the bottom of S goes
    for (page_id_t page_id = ends_[kS].head; page_id != kNone; page_id = pages_[page_id].links[kS].next) {
        const PageState& page = pages_[page_id];
        if (page.lir && frames_[page.frame].evictable) { return EvictPage(page_id); }
    }

    // every tracked page is in Q or is an LIR page in S, so getting here means the lists are broken
    throw DatabaseException("LIRS eviction: an evictable frame is in neither Q nor S");
}

void LirsEvictionPolicy::Remove(frame_id_t frame_id) {
    CheckRange(frame_id);
    FrameState& frame = frames_[frame_id];

    const page_id_t page_id = frame.page;
    if (page_id == INVALID_PAGE_ID) { return; }

    if (frame.evictable) {
        --evictable_count_;
    }
    frame = FrameState{};

    // a deleted page leaves nothing behind, not even its place in S
    Unload(page_id);
    PageState& page = pages_[page_id];
    if (page.in_s) {  // a HIR page, never S's bottom, so S stays pruned
        Unlink(page_id, kS);
        page.in_s = false;
    }
}

size_t LirsEvictionPolicy::Size() const {
    return evictable_count_;
}

void LirsEvictionPolicy::Reset() {
    std::fill(frames_.begin(), frames_.end(), FrameState{});
    pages_.clear();
    ends_.fill(Ends{});
    lir_count_ = 0;
    evictable_count_ = 0;
}

LirsSizes LirsEvictionPolicy::Sizes() const {
    return {lir_count_, ends_[kQ].size, ends_[kS].size};
}

}  // namespace shilmandb
