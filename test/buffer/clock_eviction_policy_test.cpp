#include <gtest/gtest.h>
#include <algorithm>
#include <list>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>
#include "buffer/clock_eviction_policy.hpp"
#include "common/exception.hpp"
#include "eviction_policy_harness.hpp"

namespace shilmandb {

namespace {

using test::CountMisses;
using test::PolicyHarness;
using test::VisitTypes;

// CLOCK written the slow, literal way: every frame in a std::list ring whose front is the
// frame under the hand. Moving the hand moves the front to the back.
class ReferenceClock {
public:
    explicit ReferenceClock(size_t num_frames) {
        for (size_t f = 0; f < num_frames; ++f) ring_.push_back(Slot{static_cast<frame_id_t>(f)});
    }

    void RecordAccess(frame_id_t frame_id, page_id_t, AccessType type) {
        if (type == AccessType::REPEAT) return;
        Slot& slot = Find(frame_id);
        if (!slot.tracked) {
            slot.tracked = true;
            slot.evictable = true;
        }
        slot.referenced = true;
    }

    void SetEvictable(frame_id_t frame_id, bool evictable) {
        Slot& slot = Find(frame_id);
        if (slot.tracked) slot.evictable = evictable;
    }

    std::optional<frame_id_t> Evict(page_id_t /*incoming_page*/) {
        if (Size() == 0) return std::nullopt;
        while (true) {
            ring_.splice(ring_.end(), ring_, ring_.begin());  // the hand moves past the front frame
            Slot& slot = ring_.back();
            if (!slot.tracked || !slot.evictable) continue;
            if (slot.referenced) {
                slot.referenced = false;
                continue;
            }
            slot.tracked = false;
            slot.evictable = false;
            return slot.frame;
        }
    }

    void Remove(frame_id_t frame_id) {
        Slot& slot = Find(frame_id);
        slot = Slot{frame_id};
    }

    size_t Size() const {
        return static_cast<size_t>(std::count_if(ring_.begin(), ring_.end(), [](const Slot& s) { return s.tracked && s.evictable; }));
    }

private:
    struct Slot {
        frame_id_t frame;
        bool referenced{false};
        bool tracked{false};
        bool evictable{false};
    };

    Slot& Find(frame_id_t frame_id) {
        for (Slot& slot : ring_) {
            if (slot.frame == frame_id) return slot;
        }
        throw std::logic_error("no such frame");
    }

    std::list<Slot> ring_;
};

}  // namespace

TEST(ClockEvictionPolicyTest, EvictsInItsOrder) {
    ClockEvictionPolicy clock(3);
    clock.RecordAccess(0, 10, AccessType::NEW_USE);
    clock.RecordAccess(1, 11, AccessType::NEW_USE);
    clock.RecordAccess(2, 12, AccessType::NEW_USE);

    // every bit is set: the hand clears all three, comes back to frame 0 and stops at 1
    EXPECT_EQ(clock.Evict(), std::optional<frame_id_t>(0));

    // the new page in frame 0 has its bit set; frames 1 and 2 were cleared on the way round
    clock.RecordAccess(0, 13, AccessType::NEW_USE);
    EXPECT_EQ(clock.Evict(), std::optional<frame_id_t>(1));
    EXPECT_EQ(clock.Evict(), std::optional<frame_id_t>(2));

    // only frame 0 is left: its bit is cleared, the hand goes round once more and takes it
    EXPECT_EQ(clock.Evict(), std::optional<frame_id_t>(0));
    EXPECT_FALSE(clock.Evict().has_value());
}

TEST(ClockEvictionPolicyTest, RepeatChangesNothing) {
    for (const bool touch : {false, true}) {
        ClockEvictionPolicy clock(3);
        clock.RecordAccess(0, 10, AccessType::NEW_USE);
        clock.RecordAccess(1, 11, AccessType::NEW_USE);
        clock.RecordAccess(2, 12, AccessType::NEW_USE);
        ASSERT_EQ(clock.Evict(), std::optional<frame_id_t>(0));  // clears every bit, hand at 1

        // a repeat leaves frame 1's bit clear, so it goes next; a touch sets it, so frame 2 goes
        clock.RecordAccess(1, 11, touch ? AccessType::TOUCH : AccessType::REPEAT);
        EXPECT_EQ(clock.Evict(), std::optional<frame_id_t>(touch ? 2 : 1)) << (touch ? "touch" : "repeat");
    }

    // a repeat on a frame it doesn't track starts nothing
    ClockEvictionPolicy clock(3);
    clock.RecordAccess(0, 10, AccessType::REPEAT);
    EXPECT_EQ(clock.Size(), 0u);
    EXPECT_FALSE(clock.Evict().has_value());
}

TEST(ClockEvictionPolicyTest, NonEvictableSkipped) {
    ClockEvictionPolicy clock(3);
    clock.RecordAccess(0, 10, AccessType::NEW_USE);
    clock.RecordAccess(1, 11, AccessType::NEW_USE);

    clock.SetEvictable(0, false);
    EXPECT_EQ(clock.Evict(), std::optional<frame_id_t>(1));

    clock.SetEvictable(0, true);
    EXPECT_EQ(clock.Evict(), std::optional<frame_id_t>(0));
}

TEST(ClockEvictionPolicyTest, PinnedFrameKeepsItsBit) {
    ClockEvictionPolicy clock(3);
    clock.RecordAccess(0, 10, AccessType::NEW_USE);
    clock.RecordAccess(1, 11, AccessType::NEW_USE);
    clock.RecordAccess(2, 12, AccessType::NEW_USE);
    ASSERT_EQ(clock.Evict(), std::optional<frame_id_t>(0));  // every bit cleared, hand at 1
    clock.RecordAccess(0, 13, AccessType::NEW_USE);          // frame 0's bit set
    clock.RecordAccess(1, 11, AccessType::TOUCH);            // frame 1's bit set

    clock.SetEvictable(0, false);
    ASSERT_EQ(clock.Evict(), std::optional<frame_id_t>(2));  // clears frame 1, takes 2, hand at 0
    clock.RecordAccess(2, 14, AccessType::NEW_USE);

    // the hand passes pinned frame 0 and leaves its bit set
    ASSERT_EQ(clock.Evict(), std::optional<frame_id_t>(1));
    clock.RecordAccess(1, 15, AccessType::NEW_USE);

    // unpinned, frame 0 still has its bit, so it survives this sweep and frame 2 goes
    clock.SetEvictable(0, true);
    EXPECT_EQ(clock.Evict(), std::optional<frame_id_t>(2));
}

TEST(ClockEvictionPolicyTest, NoEvictableReturnsNulloptAndKeepsState) {
    ClockEvictionPolicy clock(3);
    ClockEvictionPolicy fresh(3);
    for (ClockEvictionPolicy* p : {&clock, &fresh}) {
        p->RecordAccess(0, 10, AccessType::NEW_USE);
        p->RecordAccess(1, 11, AccessType::NEW_USE);
        p->RecordAccess(2, 12, AccessType::NEW_USE);
    }

    for (frame_id_t f = 0; f < 3; ++f) clock.SetEvictable(f, false);
    EXPECT_FALSE(clock.Evict().has_value());
    EXPECT_EQ(clock.Size(), 0u);
    for (frame_id_t f = 0; f < 3; ++f) clock.SetEvictable(f, true);

    // no bit and no hand moved: it goes on exactly like a policy that never tried
    page_id_t page_in[3] = {10, 11, 12};
    for (page_id_t page = 20; page < 30; ++page) {
        const auto victim = clock.Evict();
        ASSERT_EQ(victim, fresh.Evict()) << "page " << page;
        ASSERT_TRUE(victim.has_value());
        page_in[*victim] = page;
        clock.RecordAccess(*victim, page, AccessType::NEW_USE);
        fresh.RecordAccess(*victim, page, AccessType::NEW_USE);
        if (page % 3 == 0) {
            const frame_id_t other = (*victim + 1) % 3;
            clock.RecordAccess(other, page_in[other], AccessType::TOUCH);
            fresh.RecordAccess(other, page_in[other], AccessType::TOUCH);
        }
    }
}

TEST(ClockEvictionPolicyTest, RemoveForgetsFrame) {
    ClockEvictionPolicy clock(3);
    clock.RecordAccess(0, 10, AccessType::NEW_USE);
    clock.RecordAccess(1, 11, AccessType::NEW_USE);

    clock.Remove(0);
    EXPECT_EQ(clock.Size(), 1u);
    EXPECT_EQ(clock.Evict(), std::optional<frame_id_t>(1));
    EXPECT_FALSE(clock.Evict().has_value());

    clock.Remove(2);  // never tracked: nothing happens
    EXPECT_EQ(clock.Size(), 0u);
}

TEST(ClockEvictionPolicyTest, SizeCountsEvictableOnly) {
    ClockEvictionPolicy clock(3);
    clock.RecordAccess(0, 10, AccessType::NEW_USE);
    clock.RecordAccess(1, 11, AccessType::NEW_USE);
    EXPECT_EQ(clock.Size(), 2u);  // both evictable once loaded

    clock.SetEvictable(0, false);
    EXPECT_EQ(clock.Size(), 1u);
    clock.SetEvictable(0, false);  // twice changes nothing
    EXPECT_EQ(clock.Size(), 1u);

    clock.SetEvictable(0, true);
    EXPECT_EQ(clock.Size(), 2u);
    clock.RecordAccess(0, 10, AccessType::TOUCH);  // a hit doesn't count twice
    EXPECT_EQ(clock.Size(), 2u);
}

TEST(ClockEvictionPolicyTest, OutOfRangeFrameThrows) {
    ClockEvictionPolicy clock(3);

    EXPECT_THROW(clock.RecordAccess(3, 1, AccessType::NEW_USE), DatabaseException);
    EXPECT_THROW(clock.RecordAccess(3, 1, AccessType::REPEAT), DatabaseException);
    EXPECT_THROW(clock.RecordAccess(3, 1), DatabaseException);
    EXPECT_THROW(clock.SetEvictable(3, true), DatabaseException);
    EXPECT_THROW(clock.Remove(3), DatabaseException);

    EXPECT_EQ(clock.Size(), 0u);
    EXPECT_FALSE(clock.Evict().has_value());
}

TEST(ClockEvictionPolicyTest, NeedsThePageId) {
    ClockEvictionPolicy clock(3);
    EXPECT_THROW(clock.RecordAccess(0), DatabaseException);
    EXPECT_EQ(clock.Size(), 0u);

    // the two-argument call is a new use
    clock.RecordAccess(0, 10);
    EXPECT_EQ(clock.Size(), 1u);
    EXPECT_TRUE(clock.UsesAccessTypes());
}

TEST(ClockEvictionPolicyTest, EvictedFrameCanBeTrackedAgain) {
    ClockEvictionPolicy clock(3);

    clock.RecordAccess(0, 10, AccessType::NEW_USE);
    EXPECT_EQ(clock.Evict(), std::optional<frame_id_t>(0));
    EXPECT_EQ(clock.Size(), 0u);

    clock.RecordAccess(0, 11, AccessType::NEW_USE);
    EXPECT_EQ(clock.Size(), 1u);
    EXPECT_EQ(clock.Evict(), std::optional<frame_id_t>(0));
    EXPECT_FALSE(clock.Evict().has_value());
}

TEST(ClockEvictionPolicyTest, SetEvictableOnUntrackedFrameIgnored) {
    ClockEvictionPolicy clock(3);

    clock.SetEvictable(1, true);
    EXPECT_EQ(clock.Size(), 0u);
    EXPECT_FALSE(clock.Evict().has_value());

    clock.RecordAccess(1, 10, AccessType::NEW_USE);
    EXPECT_EQ(clock.Size(), 1u);
}

TEST(ClockEvictionPolicyTest, ResetEqualsNewPolicy) {
    constexpr size_t kPool = 8;
    ClockEvictionPolicy clock(kPool);
    std::mt19937 rng(7);
    {
        PolicyHarness harness(clock, kPool);
        VisitTypes visits(64);
        for (int i = 0; i < 500; ++i) {
            const page_id_t page = rng() % 40;
            harness.Access(page, visits.Next(page, harness.Resident(page)));
        }
        harness.Hold(rng() % 40);  // leaves a pinned frame behind
    }

    clock.Reset();
    EXPECT_EQ(clock.Size(), 0u);
    EXPECT_FALSE(clock.Evict().has_value());

    // from here it acts exactly like a new policy (the hand is back at frame 0)
    ClockEvictionPolicy fresh(kPool);
    PolicyHarness a(clock, kPool), b(fresh, kPool);
    VisitTypes va(64), vb(64);
    for (int i = 0; i < 20000; ++i) {
        const page_id_t page = rng() % 40;
        ASSERT_EQ(a.Access(page, va.Next(page, a.Resident(page))), b.Access(page, vb.Next(page, b.Resident(page)))) << "op " << i;
        ASSERT_EQ(a.LastVictim(), b.LastVictim()) << "op " << i;
        ASSERT_EQ(clock.Size(), fresh.Size()) << "op " << i;
    }
}

TEST(ClockEvictionPolicyTest, LargePageIdGrows) {
    ClockEvictionPolicy clock(2);
    PolicyHarness harness(clock, 2);

    EXPECT_TRUE(harness.Access(1'000'000, AccessType::NEW_USE));
    EXPECT_TRUE(harness.Access(5, AccessType::NEW_USE));
    EXPECT_FALSE(harness.Access(1'000'000, AccessType::NEW_USE));

    // both bits set: one sweep clears them and the page in frame 0 goes
    EXPECT_TRUE(harness.Access(6, AccessType::NEW_USE));
    EXPECT_EQ(harness.LastVictim(), std::optional<page_id_t>(1'000'000));
    EXPECT_TRUE(harness.Access(1'000'000, AccessType::NEW_USE));
    EXPECT_EQ(harness.LastVictim(), std::optional<page_id_t>(5));
}

TEST(ClockEvictionPolicyTest, PoolOfOne) {
    ClockEvictionPolicy clock(1);
    clock.RecordAccess(0, 10, AccessType::NEW_USE);
    EXPECT_EQ(clock.Evict(), std::optional<frame_id_t>(0));  // its own bit is cleared first
    EXPECT_FALSE(clock.Evict().has_value());

    for (const std::optional<uint64_t> window : {std::optional<uint64_t>(), std::optional<uint64_t>(64)}) {
        ClockEvictionPolicy one(1);
        EXPECT_EQ(CountMisses(one, 1, {1, 1, 2, 2, 1, 3}, window), 4u);
    }

    ClockEvictionPolicy one(1);
    PolicyHarness harness(one, 1);
    harness.Access(1, AccessType::NEW_USE);
    harness.Access(2, AccessType::NEW_USE);
    EXPECT_EQ(harness.LastVictim(), std::optional<page_id_t>(1));
}

TEST(ClockEvictionPolicyTest, TeachingsExample) {
    // A B C D A B E A B C D E
    ClockEvictionPolicy clock(3);
    EXPECT_EQ(CountMisses(clock, 3, {0, 1, 2, 3, 0, 1, 4, 0, 1, 2, 3, 4}), 9u);
}

TEST(ClockEvictionPolicyTest, ScanAfterHotPair) {
    // The pair 1 2 twice, a scan of pages 10..19, then the pair again.
    std::vector<page_id_t> pages = {1, 2, 1, 2};
    for (page_id_t p = 10; p < 20; ++p) pages.push_back(p);
    pages.push_back(1);
    pages.push_back(2);
    ClockEvictionPolicy clock(4);
    EXPECT_EQ(CountMisses(clock, 4, pages), 14u);
}

TEST(ClockEvictionPolicyTest, MatchesReferenceOnRandomOps) {
    const auto result = test::MatchReferenceOnRandomOps<ClockEvictionPolicy, ReferenceClock>({1, 2, 3, 5, 9, 17, 30, 40});
    EXPECT_TRUE(result.error.empty()) << result.error;
    EXPECT_GE(result.ops, 200000u);
}

}  // namespace shilmandb
