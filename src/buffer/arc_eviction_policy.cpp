#include "buffer/arc_eviction_policy.hpp"
#include "common/exception.hpp"
#include <algorithm>

namespace shilmandb {

ArcEvictionPolicy::ArcEvictionPolicy(size_t num_frames) : c_(num_frames), frames_(num_frames) {}

void ArcEvictionPolicy::CheckRange(frame_id_t frame_id) const {
    if (frame_id >= frames_.size()) {
        throw DatabaseException("frame id out of range");
    }
}

ArcEvictionPolicy::List ArcEvictionPolicy::ListOf(page_id_t page_id) const {
    return page_id < pages_.size() ? pages_[page_id].list : kOut;
}

void ArcEvictionPolicy::PushBack(page_id_t page_id, List list) {
    Ends& ends = ends_[list];
    PageState& page = pages_[page_id];
    page.prev = ends.tail;
    page.next = kNone;
    page.list = list;

    if (ends.tail == kNone) {
        ends.head = page_id;
    } else {
        pages_[ends.tail].next = page_id;
    }
    ends.tail = page_id;
    ++ends.size;
}

// Takes the page out of whatever list it is in
void ArcEvictionPolicy::Unlink(page_id_t page_id) {
    PageState& page = pages_[page_id];
    Ends& ends = ends_[page.list];

    if (page.prev == kNone) {
        ends.head = page.next;
    } else {
        pages_[page.prev].next = page.next;
    }

    if (page.next == kNone) {
        ends.tail = page.prev;
    } else {
        pages_[page.next].prev = page.prev;
    }

    --ends.size;
    page.prev = kNone;
    page.next = kNone;
    page.list = kOut;
}

// Puts the page at the newest end of `list`, leaving the list it was in
void ArcEvictionPolicy::MoveTo(page_id_t page_id, List list) {
    if (pages_[page_id].list != kOut) {
        Unlink(page_id);
    }
    PushBack(page_id, list);
}

void ArcEvictionPolicy::GrowP() {
    const auto b1 = static_cast<double>(Count(kB1)), b2 = static_cast<double>(Count(kB2));
    p_ = std::min(static_cast<double>(c_), p_ + (b1 >= b2 ? 1.0 : b2 / b1));
}

void ArcEvictionPolicy::ShrinkP() {
    const auto b1 = static_cast<double>(Count(kB1)), b2 = static_cast<double>(Count(kB2));
    p_ = std::max(0.0, p_ - (b2 >= b1 ? 1.0 : b1 / b2));
}

page_id_t ArcEvictionPolicy::OldestEvictable(List list) const {
    for (page_id_t page_id = ends_[list].head; page_id != kNone; page_id = pages_[page_id].next) {
        if (frames_[pages_[page_id].frame].evictable) { return page_id; }
    }
    return kNone;
}

page_id_t ArcEvictionPolicy::PickVictim(List list) const {
    page_id_t page_id = OldestEvictable(list);
    if (page_id == kNone) {
        page_id = OldestEvictable(list == kT1 ? kT2 : kT1);
    }
    // callers made sure some frame is evictable, so this means the lists are broken
    if (page_id == kNone) {
        throw DatabaseException("ARC eviction: no evictable page in T1 or T2");
    }
    return page_id;
}

void ArcEvictionPolicy::DropOldestGhost(List list) {
    const page_id_t page_id = ends_[list].head;
    // the counts say there is one, so an empty list means they are broken
    if (page_id == kNone) {
        throw DatabaseException("ARC eviction: no ghost to drop");
    }
    Unlink(page_id);
}

frame_id_t ArcEvictionPolicy::Untrack(page_id_t page_id) {
    const frame_id_t frame_id = pages_[page_id].frame;
    pages_[page_id].frame = kNoFrame;
    if (frames_[frame_id].evictable) {
        --evictable_count_;
    }
    frames_[frame_id] = FrameState{};
    return frame_id;
}

frame_id_t ArcEvictionPolicy::Replace(bool incoming_in_b2) {
    const auto t1 = static_cast<double>(Count(kT1));
    const List from = Count(kT1) >= 1 && ((incoming_in_b2 && t1 == p_) || t1 > p_) ? kT1 : kT2;

    const page_id_t victim = PickVictim(from);
    const frame_id_t frame_id = Untrack(victim);
    MoveTo(victim, pages_[victim].list == kT1 ? kB1 : kB2);
    return frame_id;
}

void ArcEvictionPolicy::MakeRoom(List incoming_list) {
    switch (incoming_list) {
        case kB1:
            GrowP();
            break;
        case kB2:
            ShrinkP();
            break;
        case kOut: {
            // a frame is free here, so T1 is short of c: L1 full means B1 has a ghost to drop,
            // and a full directory means B2 has one
            const size_t l1 = Count(kT1) + Count(kB1);
            const size_t total = l1 + Count(kT2) + Count(kB2);
            if (l1 == c_) {
                DropOldestGhost(kB1);
            } else if (total == 2 * c_) {
                DropOldestGhost(kB2);
            }
            break;
        }
        default:
            break;
    }
}

void ArcEvictionPolicy::RecordAccess(frame_id_t /*frame_id*/) {
    throw DatabaseException("ARC eviction needs the page id");
}

void ArcEvictionPolicy::RecordAccess(frame_id_t frame_id, page_id_t page_id) {
    RecordAccess(frame_id, page_id, AccessType::NEW_USE);
}

void ArcEvictionPolicy::RecordAccess(frame_id_t frame_id, page_id_t page_id, AccessType type) {
    CheckRange(frame_id);
    if (type == AccessType::REPEAT) { return; }
    if (page_id == INVALID_PAGE_ID) {
        throw DatabaseException("ARC eviction needs a valid page id");
    }

    FrameState& frame = frames_[frame_id];

    // a hit: a touch keeps the page in its own list, a new use makes it a T2 page
    if (frame.page == page_id) {
        MoveTo(page_id, type == AccessType::TOUCH ? pages_[page_id].list : kT2);
        return;
    }

    // a load
    if (page_id < pages_.size() && pages_[page_id].frame != kNoFrame) {
        throw DatabaseException("ARC eviction: page is already in another frame");
    }
    if (page_id >= pages_.size()) {
        pages_.resize(static_cast<size_t>(page_id) + 1);
    }

    // only the load the last Evict was for skips the paper's work, since Evict already did it
    const bool prepared = frame_id == prepared_frame_ && (page_id == prepared_page_ || (prepared_page_ == INVALID_PAGE_ID && pages_[page_id].list == kOut));
    prepared_frame_ = kNoFrame;
    prepared_page_ = INVALID_PAGE_ID;

    // a frame that still held another page lets it go, as if evicted
    if (frame.page != INVALID_PAGE_ID) {
        const page_id_t old = frame.page;
        Untrack(old);
        MoveTo(old, pages_[old].list == kT1 ? kB1 : kB2);
    }

    const List from = pages_[page_id].list;
    if (!prepared) {
        MakeRoom(from);
    }
    MoveTo(page_id, from == kOut ? kT1 : kT2);  // a ghost comes back used twice
    pages_[page_id].frame = frame_id;
    frame.page = page_id;
    frame.evictable = true;
    ++evictable_count_;
}

void ArcEvictionPolicy::SetEvictable(frame_id_t frame_id, bool evictable) {
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

std::optional<frame_id_t> ArcEvictionPolicy::Evict() {
    return Evict(INVALID_PAGE_ID);
}

std::optional<frame_id_t> ArcEvictionPolicy::Evict(page_id_t incoming_page) {
    if (evictable_count_ == 0) { return std::nullopt; }

    prepared_frame_ = kNoFrame;
    prepared_page_ = INVALID_PAGE_ID;

    frame_id_t frame_id = kNoFrame;
    switch (ListOf(incoming_page)) {
        case kT1:
        case kT2:
            // already cached, so no room to make: just a victim, and nothing prepared
            return Replace(false);
        case kB1:
            GrowP();
            frame_id = Replace(false);
            break;
        case kB2:
            ShrinkP();
            frame_id = Replace(true);
            break;
        case kOut: {
            const size_t l1 = Count(kT1) + Count(kB1);
            const size_t total = l1 + Count(kT2) + Count(kB2);
            if (l1 == c_) {
                if (Count(kT1) < c_) {
                    DropOldestGhost(kB1);
                    frame_id = Replace(false);
                } else {
                    // T1 fills the cache: its oldest page leaves without a ghost
                    const page_id_t victim = PickVictim(kT1);
                    frame_id = Untrack(victim);
                    Unlink(victim);
                }
            } else {
                if (total == 2 * c_) {
                    DropOldestGhost(kB2);
                }
                frame_id = Replace(false);
            }
            break;
        }
    }

    prepared_frame_ = frame_id;
    prepared_page_ = incoming_page;
    return frame_id;
}

void ArcEvictionPolicy::Remove(frame_id_t frame_id) {
    CheckRange(frame_id);
    if (frame_id == prepared_frame_) {
        prepared_frame_ = kNoFrame;
        prepared_page_ = INVALID_PAGE_ID;
    }

    const page_id_t page_id = frames_[frame_id].page;
    if (page_id == INVALID_PAGE_ID) { return; }

    // a deleted page leaves no ghost behind
    Untrack(page_id);
    Unlink(page_id);
}

size_t ArcEvictionPolicy::Size() const {
    return evictable_count_;
}

void ArcEvictionPolicy::Reset() {
    std::fill(frames_.begin(), frames_.end(), FrameState{});
    pages_.clear();
    ends_.fill(Ends{});
    p_ = 0;
    evictable_count_ = 0;
    prepared_frame_ = kNoFrame;
    prepared_page_ = INVALID_PAGE_ID;
}

ArcSizes ArcEvictionPolicy::Sizes() const {
    return {Count(kT1), Count(kT2), Count(kB1), Count(kB2), p_};
}

}  // namespace shilmandb
