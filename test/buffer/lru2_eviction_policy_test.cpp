#include <gtest/gtest.h>
#include <algorithm>
#include <list>
#include <map>
#include <optional>
#include <random>
#include <vector>
#include "buffer/lru2_eviction_policy.hpp"
#include "common/exception.hpp"
#include "eviction_policy_harness.hpp"

namespace shilmandb {

namespace {

using test::CountMisses;
using test::PolicyHarness;
using test::VisitTypes;

// LRU-2 written the slow, literal way: tracked frames in a std::list in load order, use
// history in a std::map, and a linear search for the victim.
class ReferenceLru2 {
public:
    explicit ReferenceLru2(size_t /*num_frames*/) {}

    void RecordAccess(frame_id_t frame_id, page_id_t page_id, AccessType type) {
        if (type == AccessType::REPEAT) return;
        ++clock_;
        History& history = history_[page_id];
        auto it = Find(frame_id);
        if (it != frames_.end() && it->page == page_id) {
            if (type != AccessType::TOUCH) history.second = history.last;  // a touch is not a second use
            history.last = clock_;
            return;
        }
        if (it == frames_.end()) {
            frames_.push_back(Entry{frame_id, page_id, true});
        } else {
            it->page = page_id;
        }
        history.second = history.last;  // a load is always a new use
        history.last = clock_;
    }

    void SetEvictable(frame_id_t frame_id, bool evictable) {
        if (auto it = Find(frame_id); it != frames_.end()) it->evictable = evictable;
    }

    std::optional<frame_id_t> Evict(page_id_t /*incoming_page*/) {
        auto victim = frames_.end();
        for (auto it = frames_.begin(); it != frames_.end(); ++it) {
            if (!it->evictable) continue;
            if (victim == frames_.end() || EvictsBefore(it->page, victim->page)) victim = it;
        }
        if (victim == frames_.end()) return std::nullopt;
        const frame_id_t frame = victim->frame;
        frames_.erase(victim);
        return frame;
    }

    void Remove(frame_id_t frame_id) {
        auto it = Find(frame_id);
        if (it == frames_.end()) return;
        history_.erase(it->page);
        frames_.erase(it);
    }

    size_t Size() const {
        return static_cast<size_t>(std::count_if(frames_.begin(), frames_.end(), [](const Entry& e) { return e.evictable; }));
    }

private:
    static constexpr int64_t kNever = -1;

    struct History {
        int64_t last{kNever};
        int64_t second{kNever};
    };
    struct Entry {
        frame_id_t frame;
        page_id_t page;
        bool evictable;
    };

    // A page used once goes before one used twice (older last use first); between two used
    // twice, the older second-to-last use goes first.
    bool EvictsBefore(page_id_t a, page_id_t b) const {
        const History& ha = history_.at(a);
        const History& hb = history_.at(b);
        const bool a_twice = ha.second != kNever;
        const bool b_twice = hb.second != kNever;
        if (a_twice != b_twice) return !a_twice;
        if (!a_twice) return ha.last < hb.last;
        return ha.second < hb.second;
    }

    std::list<Entry>::iterator Find(frame_id_t frame_id) {
        for (auto it = frames_.begin(); it != frames_.end(); ++it) {
            if (it->frame == frame_id) return it;
        }
        return frames_.end();
    }

    std::list<Entry> frames_;
    std::map<page_id_t, History> history_;
    int64_t clock_{0};
};

}  // namespace

TEST(Lru2EvictionPolicyTest, EvictsInItsOrder) {
    Lru2EvictionPolicy lru2(3);
    lru2.RecordAccess(0, 10, AccessType::NEW_USE);  // 10 used at 1
    lru2.RecordAccess(1, 11, AccessType::NEW_USE);  // 11 at 2
    lru2.RecordAccess(2, 12, AccessType::NEW_USE);  // 12 at 3
    lru2.RecordAccess(1, 11, AccessType::NEW_USE);  // 11 at 2 and 4
    lru2.RecordAccess(0, 10, AccessType::NEW_USE);  // 10 at 1 and 5

    // used once goes first; then the older second-to-last use, so 10 before 11 (plain LRU would take 11)
    EXPECT_EQ(lru2.Evict(), std::optional<frame_id_t>(2));
    EXPECT_EQ(lru2.Evict(), std::optional<frame_id_t>(0));
    EXPECT_EQ(lru2.Evict(), std::optional<frame_id_t>(1));
    EXPECT_FALSE(lru2.Evict().has_value());
}

TEST(Lru2EvictionPolicyTest, RepeatChangesNothing) {
    Lru2EvictionPolicy lru2(3);
    lru2.RecordAccess(0, 10, AccessType::NEW_USE);  // 10 at 1
    lru2.RecordAccess(1, 11, AccessType::NEW_USE);  // 11 at 2
    lru2.RecordAccess(0, 10, AccessType::REPEAT);   // not a use: as a use, 10 would be kept over 11

    EXPECT_EQ(lru2.Evict(), std::optional<frame_id_t>(0));

    // a repeat on a frame it doesn't track starts nothing
    Lru2EvictionPolicy other(3);
    other.RecordAccess(0, 10, AccessType::REPEAT);
    EXPECT_EQ(other.Size(), 0u);
    EXPECT_FALSE(other.Evict().has_value());
}

TEST(Lru2EvictionPolicyTest, TouchSetsLastOnly) {
    for (const bool touch : {true, false}) {
        Lru2EvictionPolicy lru2(2);
        lru2.RecordAccess(0, 10, AccessType::NEW_USE);  // 10 at 1
        lru2.RecordAccess(1, 11, AccessType::NEW_USE);  // 11 at 2
        lru2.RecordAccess(0, 10, AccessType::NEW_USE);  // 10 at 1 and 3: used twice

        // a touch moves 11's last use only, so it still counts as used once and goes first;
        // as a new use 11 has 2 and 4, and 10's older second-to-last use (1) goes first
        lru2.RecordAccess(1, 11, touch ? AccessType::TOUCH : AccessType::NEW_USE);
        EXPECT_EQ(lru2.Evict(), std::optional<frame_id_t>(touch ? 1 : 0)) << (touch ? "touch" : "new use");
    }

    // a load is always a new use, even sent as a touch: the page wasn't resident
    Lru2EvictionPolicy lru2(2);
    lru2.RecordAccess(0, 10, AccessType::NEW_USE);  // 10 at 1
    ASSERT_EQ(lru2.Evict(), std::optional<frame_id_t>(0));
    lru2.RecordAccess(0, 10, AccessType::TOUCH);    // back in: 10 at 1 and 2
    lru2.RecordAccess(1, 11, AccessType::NEW_USE);  // 11 at 3
    EXPECT_EQ(lru2.Evict(), std::optional<frame_id_t>(1));
}

TEST(Lru2EvictionPolicyTest, NonEvictableSkipped) {
    Lru2EvictionPolicy lru2(3);
    lru2.RecordAccess(0, 10, AccessType::NEW_USE);
    lru2.RecordAccess(1, 11, AccessType::NEW_USE);

    lru2.SetEvictable(0, false);
    EXPECT_EQ(lru2.Evict(), std::optional<frame_id_t>(1));

    // still first in order once it is evictable again
    lru2.RecordAccess(1, 12, AccessType::NEW_USE);
    lru2.SetEvictable(0, true);
    EXPECT_EQ(lru2.Evict(), std::optional<frame_id_t>(0));
}

TEST(Lru2EvictionPolicyTest, NoEvictableReturnsNulloptAndKeepsState) {
    Lru2EvictionPolicy lru2(3);
    Lru2EvictionPolicy fresh(3);
    for (Lru2EvictionPolicy* p : {&lru2, &fresh}) {
        p->RecordAccess(0, 10, AccessType::NEW_USE);
        p->RecordAccess(1, 11, AccessType::NEW_USE);
        p->RecordAccess(2, 12, AccessType::NEW_USE);
        p->RecordAccess(1, 11, AccessType::NEW_USE);
    }

    for (frame_id_t f = 0; f < 3; ++f) lru2.SetEvictable(f, false);
    EXPECT_FALSE(lru2.Evict().has_value());
    EXPECT_EQ(lru2.Size(), 0u);
    for (frame_id_t f = 0; f < 3; ++f) lru2.SetEvictable(f, true);

    // it goes on exactly like a policy that never tried
    page_id_t page_in[3] = {10, 11, 12};
    for (page_id_t page = 20; page < 30; ++page) {
        const auto victim = lru2.Evict();
        ASSERT_EQ(victim, fresh.Evict()) << "page " << page;
        ASSERT_TRUE(victim.has_value());
        page_in[*victim] = page;
        lru2.RecordAccess(*victim, page, AccessType::NEW_USE);
        fresh.RecordAccess(*victim, page, AccessType::NEW_USE);
        const frame_id_t other = (*victim + 1) % 3;
        const AccessType type = page % 2 == 0 ? AccessType::NEW_USE : AccessType::TOUCH;
        lru2.RecordAccess(other, page_in[other], type);
        fresh.RecordAccess(other, page_in[other], type);
    }
}

TEST(Lru2EvictionPolicyTest, RemoveForgetsFrame) {
    Lru2EvictionPolicy lru2(3);
    lru2.RecordAccess(1, 11, AccessType::NEW_USE);  // 11 at 1
    lru2.RecordAccess(0, 10, AccessType::NEW_USE);  // 10 at 2
    lru2.RecordAccess(1, 11, AccessType::NEW_USE);  // 11 at 1 and 3
    lru2.RecordAccess(0, 10, AccessType::NEW_USE);  // 10 at 2 and 4

    lru2.Remove(0);
    EXPECT_EQ(lru2.Size(), 1u);

    // the removed page left no history: back in, it is used once and goes first
    // (with its old uses kept, 11's older second-to-last use would go first)
    lru2.RecordAccess(0, 10, AccessType::NEW_USE);
    EXPECT_EQ(lru2.Evict(), std::optional<frame_id_t>(0));
    EXPECT_EQ(lru2.Evict(), std::optional<frame_id_t>(1));
    EXPECT_FALSE(lru2.Evict().has_value());

    lru2.Remove(2);  // never tracked: nothing happens
    EXPECT_EQ(lru2.Size(), 0u);
}

TEST(Lru2EvictionPolicyTest, SizeCountsEvictableOnly) {
    Lru2EvictionPolicy lru2(3);
    lru2.RecordAccess(0, 10, AccessType::NEW_USE);
    lru2.RecordAccess(1, 11, AccessType::NEW_USE);
    EXPECT_EQ(lru2.Size(), 2u);  // both evictable once loaded

    lru2.SetEvictable(0, false);
    EXPECT_EQ(lru2.Size(), 1u);
    lru2.SetEvictable(0, false);  // twice changes nothing
    EXPECT_EQ(lru2.Size(), 1u);

    lru2.SetEvictable(0, true);
    EXPECT_EQ(lru2.Size(), 2u);
    lru2.RecordAccess(0, 10, AccessType::NEW_USE);  // a hit doesn't count twice
    EXPECT_EQ(lru2.Size(), 2u);
}

TEST(Lru2EvictionPolicyTest, OutOfRangeFrameThrows) {
    Lru2EvictionPolicy lru2(3);

    EXPECT_THROW(lru2.RecordAccess(3, 1, AccessType::NEW_USE), DatabaseException);
    EXPECT_THROW(lru2.RecordAccess(3, 1, AccessType::REPEAT), DatabaseException);
    EXPECT_THROW(lru2.RecordAccess(3, 1), DatabaseException);
    EXPECT_THROW(lru2.SetEvictable(3, true), DatabaseException);
    EXPECT_THROW(lru2.Remove(3), DatabaseException);

    EXPECT_EQ(lru2.Size(), 0u);
    EXPECT_FALSE(lru2.Evict().has_value());
}

TEST(Lru2EvictionPolicyTest, NeedsThePageId) {
    Lru2EvictionPolicy lru2(3);
    EXPECT_THROW(lru2.RecordAccess(0), DatabaseException);
    EXPECT_THROW(lru2.RecordAccess(0, INVALID_PAGE_ID, AccessType::NEW_USE), DatabaseException);
    EXPECT_EQ(lru2.Size(), 0u);

    // the two-argument call is a new use
    lru2.RecordAccess(0, 10);
    EXPECT_EQ(lru2.Size(), 1u);
    EXPECT_TRUE(lru2.UsesAccessTypes());

    // a page lives in one frame only
    EXPECT_THROW(lru2.RecordAccess(1, 10, AccessType::NEW_USE), DatabaseException);
    EXPECT_EQ(lru2.Size(), 1u);
}

TEST(Lru2EvictionPolicyTest, PageInAnotherFrameThrowsAndChangesNothing) {
    // the same calls on both, except that `a` is also asked to load pages other frames hold
    Lru2EvictionPolicy a(3), b(3);
    for (Lru2EvictionPolicy* p : {&a, &b}) {
        p->RecordAccess(0, 10, AccessType::NEW_USE);  // 10 at 1
        p->RecordAccess(1, 11, AccessType::NEW_USE);  // 11 at 2
    }
    EXPECT_THROW(a.RecordAccess(2, 10, AccessType::NEW_USE), DatabaseException);
    EXPECT_THROW(a.RecordAccess(2, 11, AccessType::TOUCH), DatabaseException);
    EXPECT_THROW(a.RecordAccess(1, 10, AccessType::NEW_USE), DatabaseException);
    EXPECT_EQ(a.Size(), 2u);

    // had a throw counted as a use of 10, 10 would look used twice and 11 would go first
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(a.Evict(), b.Evict()) << "eviction " << i;
    }
}

TEST(Lru2EvictionPolicyTest, PageCanMoveOnceItLeavesItsFrame) {
    Lru2EvictionPolicy lru2(3);

    lru2.RecordAccess(0, 10, AccessType::NEW_USE);
    ASSERT_EQ(lru2.Evict(), std::optional<frame_id_t>(0));
    lru2.RecordAccess(1, 10, AccessType::NEW_USE);  // evicted from frame 0
    lru2.Remove(1);
    lru2.RecordAccess(2, 10, AccessType::NEW_USE);  // removed from frame 1
    lru2.RecordAccess(2, 11, AccessType::NEW_USE);  // frame 2 lets 10 go
    lru2.RecordAccess(0, 10, AccessType::NEW_USE);
    EXPECT_EQ(lru2.Size(), 2u);
    EXPECT_THROW(lru2.RecordAccess(1, 11, AccessType::NEW_USE), DatabaseException);

    // a reset forgets which frame held each page
    lru2.Reset();
    lru2.RecordAccess(1, 10, AccessType::NEW_USE);
    lru2.RecordAccess(2, 11, AccessType::NEW_USE);
    EXPECT_EQ(lru2.Size(), 2u);
}

TEST(Lru2EvictionPolicyTest, EvictedFrameCanBeTrackedAgain) {
    Lru2EvictionPolicy lru2(3);

    lru2.RecordAccess(0, 10, AccessType::NEW_USE);  // 10 at 1
    EXPECT_EQ(lru2.Evict(), std::optional<frame_id_t>(0));
    EXPECT_EQ(lru2.Size(), 0u);

    // an evicted page keeps its history: back in, 10 has two uses (1 and 2), 11 has one
    lru2.RecordAccess(0, 10, AccessType::NEW_USE);
    lru2.RecordAccess(1, 11, AccessType::NEW_USE);
    EXPECT_EQ(lru2.Size(), 2u);
    EXPECT_EQ(lru2.Evict(), std::optional<frame_id_t>(1));
    EXPECT_EQ(lru2.Evict(), std::optional<frame_id_t>(0));
    EXPECT_FALSE(lru2.Evict().has_value());
}

TEST(Lru2EvictionPolicyTest, SetEvictableOnUntrackedFrameIgnored) {
    Lru2EvictionPolicy lru2(3);

    lru2.SetEvictable(1, true);
    EXPECT_EQ(lru2.Size(), 0u);
    EXPECT_FALSE(lru2.Evict().has_value());

    lru2.RecordAccess(1, 10, AccessType::NEW_USE);
    EXPECT_EQ(lru2.Size(), 1u);
}

TEST(Lru2EvictionPolicyTest, ResetEqualsNewPolicy) {
    constexpr size_t kPool = 8;
    Lru2EvictionPolicy lru2(kPool);
    std::mt19937 rng(7);
    {
        PolicyHarness harness(lru2, kPool);
        VisitTypes visits(64);
        for (int i = 0; i < 500; ++i) {
            const page_id_t page = rng() % 40;
            harness.Access(page, visits.Next(page, harness.Resident(page)));
        }
        harness.Hold(rng() % 40);  // leaves a pinned frame behind
    }

    lru2.Reset();
    EXPECT_EQ(lru2.Size(), 0u);
    EXPECT_FALSE(lru2.Evict().has_value());

    // from here it acts exactly like a new policy: no page keeps any history
    Lru2EvictionPolicy fresh(kPool);
    PolicyHarness a(lru2, kPool), b(fresh, kPool);
    VisitTypes va(64), vb(64);
    for (int i = 0; i < 20000; ++i) {
        const page_id_t page = rng() % 40;
        ASSERT_EQ(a.Access(page, va.Next(page, a.Resident(page))), b.Access(page, vb.Next(page, b.Resident(page)))) << "op " << i;
        ASSERT_EQ(a.LastVictim(), b.LastVictim()) << "op " << i;
        ASSERT_EQ(lru2.Size(), fresh.Size()) << "op " << i;
    }
}

TEST(Lru2EvictionPolicyTest, LargePageIdGrows) {
    Lru2EvictionPolicy lru2(2);
    PolicyHarness harness(lru2, 2);

    EXPECT_TRUE(harness.Access(1'000'000, AccessType::NEW_USE));   // used at 1
    EXPECT_TRUE(harness.Access(5, AccessType::NEW_USE));           // 5 at 2
    EXPECT_FALSE(harness.Access(1'000'000, AccessType::NEW_USE));  // at 1 and 3

    // the big page has two uses, so the pages used once go before it
    EXPECT_TRUE(harness.Access(6, AccessType::NEW_USE));
    EXPECT_EQ(harness.LastVictim(), std::optional<page_id_t>(5));
    EXPECT_TRUE(harness.Access(7, AccessType::NEW_USE));
    EXPECT_EQ(harness.LastVictim(), std::optional<page_id_t>(6));
    EXPECT_TRUE(harness.Resident(1'000'000));

    // a brand new page as the incoming one is ignored
    Lru2EvictionPolicy other(1);
    other.RecordAccess(0, 3, AccessType::NEW_USE);
    EXPECT_EQ(other.Evict(INVALID_PAGE_ID), std::optional<frame_id_t>(0));
}

TEST(Lru2EvictionPolicyTest, PoolOfOne) {
    Lru2EvictionPolicy lru2(1);
    lru2.RecordAccess(0, 10, AccessType::NEW_USE);
    EXPECT_EQ(lru2.Evict(), std::optional<frame_id_t>(0));
    EXPECT_FALSE(lru2.Evict().has_value());

    for (const std::optional<uint64_t> window : {std::optional<uint64_t>(), std::optional<uint64_t>(64)}) {
        Lru2EvictionPolicy one(1);
        EXPECT_EQ(CountMisses(one, 1, {1, 1, 2, 2, 1, 3}, window), 4u);
    }

    Lru2EvictionPolicy one(1);
    PolicyHarness harness(one, 1);
    harness.Access(1, AccessType::NEW_USE);
    harness.Access(1, AccessType::NEW_USE);
    harness.Access(2, AccessType::NEW_USE);
    EXPECT_EQ(harness.LastVictim(), std::optional<page_id_t>(1));
}

TEST(Lru2EvictionPolicyTest, TeachingsExample) {
    // A B C D A B E A B C D E
    Lru2EvictionPolicy lru2(3);
    EXPECT_EQ(CountMisses(lru2, 3, {0, 1, 2, 3, 0, 1, 4, 0, 1, 2, 3, 4}), 10u);
}

TEST(Lru2EvictionPolicyTest, ScanAfterHotPair) {
    // The pair 1 2 twice, a scan of pages 10..19, then the pair again.
    std::vector<page_id_t> pages = {1, 2, 1, 2};
    for (page_id_t p = 10; p < 20; ++p) pages.push_back(p);
    pages.push_back(1);
    pages.push_back(2);
    Lru2EvictionPolicy lru2(4);
    EXPECT_EQ(CountMisses(lru2, 4, pages), 12u);
}

TEST(Lru2EvictionPolicyTest, VisitModeSkipsInterleavedReturns) {
    // Pairs read in turn (10 20 10 20 ...) look used twice unless a quick return is a touch.
    std::vector<page_id_t> pages = {1, 2, 5, 6, 5, 6, 1, 2};
    for (page_id_t k = 0; k < 4; ++k) {
        for (const page_id_t p : {10 + k, 20 + k, 10 + k, 20 + k}) pages.push_back(p);
    }
    pages.push_back(1);
    pages.push_back(2);

    Lru2EvictionPolicy visit(4);
    EXPECT_EQ(CountMisses(visit, 4, pages, 3), 12u);
    Lru2EvictionPolicy plain(4);
    EXPECT_EQ(CountMisses(plain, 4, pages), 22u);
}

TEST(Lru2EvictionPolicyTest, MatchesReferenceOnRandomOps) {
    const auto result = test::MatchReferenceOnRandomOps<Lru2EvictionPolicy, ReferenceLru2>({1, 2, 3, 5, 9, 17, 30, 40});
    EXPECT_TRUE(result.error.empty()) << result.error;
    EXPECT_GE(result.ops, 200000u);
}

}  // namespace shilmandb
