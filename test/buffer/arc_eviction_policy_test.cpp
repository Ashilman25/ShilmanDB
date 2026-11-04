#include <gtest/gtest.h>
#include <algorithm>
#include <deque>
#include <iterator>
#include <list>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "buffer/arc_eviction_policy.hpp"
#include "common/exception.hpp"
#include "eviction_policy_harness.hpp"

namespace shilmandb {

namespace {

using test::CountMisses;
using test::PolicyHarness;
using test::Twin;
using test::VisitTypes;

std::string Describe(const ArcSizes& s) {
    std::ostringstream out;
    out << "T1 " << s.t1 << " T2 " << s.t2 << " B1 " << s.b1 << " B2 " << s.b2 << " p " << s.p;
    return out.str();
}

bool SameSizes(const ArcSizes& a, const ArcSizes& b) {
    return a.t1 == b.t1 && a.t2 == b.t2 && a.b1 == b.b1 && a.b2 == b.b2 && a.p == b.p;
}

// Empty when the paper's invariants hold for a cache of c pages, else what broke
std::string InvariantProblem(const ArcSizes& s, size_t c) {
    const bool ok = s.t1 + s.t2 <= c && s.t1 + s.b1 <= c && s.t1 + s.t2 + s.b1 + s.b2 <= 2 * c &&
                    s.p >= 0.0 && s.p <= static_cast<double>(c);
    return ok ? "" : "invariants broken at c " + std::to_string(c) + ": " + Describe(s);
}

// ARC written the slow, literal way (Megiddo & Modha, FAST 2003, Figure 4, case by case):
// the four lists are std::lists with the oldest page in front, the tracked frames another
// std::list, and every lookup a linear search. Pins, the prepared eviction and loads nobody
// prepared follow the same rules as the real policy.
class ReferenceArc {
public:
    explicit ReferenceArc(size_t num_frames) : c_(num_frames) {}

    void RecordAccess(frame_id_t frame_id, page_id_t page_id, AccessType type) {
        if (type == AccessType::REPEAT) return;
        auto frame = FindFrame(frame_id);
        if (frame != frames_.end() && frame->page == page_id) {  // case I
            if (type == AccessType::TOUCH) {
                auto& list = Has(t1_, page_id) ? t1_ : t2_;
                list.remove(page_id);
                list.push_back(page_id);
            } else {
                t1_.remove(page_id);
                t2_.remove(page_id);
                t2_.push_back(page_id);
            }
            return;
        }

        // a load: only the one Evict prepared skips the paper's work
        const bool prepared = prepared_ && prepared_frame_ == frame_id &&
                              (prepared_page_ == page_id || (prepared_page_ == INVALID_PAGE_ID && !Known(page_id)));
        prepared_ = false;
        if (frame != frames_.end()) {  // the frame's old page leaves as if evicted
            const page_id_t old = frame->page;
            const bool in_t1 = Has(t1_, old);
            (in_t1 ? t1_ : t2_).remove(old);
            (in_t1 ? b1_ : b2_).push_back(old);
            frames_.erase(frame);
        }

        const bool in_b1 = Has(b1_, page_id), in_b2 = Has(b2_, page_id);
        if (!prepared) {
            if (in_b1) {
                GrowP();
            } else if (in_b2) {
                ShrinkP();
            } else {
                const size_t l1 = t1_.size() + b1_.size();
                if (l1 == c_) {
                    b1_.pop_front();
                } else if (l1 + t2_.size() + b2_.size() == 2 * c_) {
                    b2_.pop_front();
                }
            }
        }
        if (in_b1 || in_b2) {
            b1_.remove(page_id);
            b2_.remove(page_id);
            t2_.push_back(page_id);
        } else {
            t1_.push_back(page_id);
        }
        frames_.push_back(Entry{frame_id, page_id, true});
    }

    void SetEvictable(frame_id_t frame_id, bool evictable) {
        if (auto it = FindFrame(frame_id); it != frames_.end()) it->evictable = evictable;
    }

    std::optional<frame_id_t> Evict(page_id_t x) {
        if (Size() == 0) return std::nullopt;
        prepared_ = false;
        if (Has(t1_, x) || Has(t2_, x)) return Replace(x);  // already cached: a victim, nothing prepared

        frame_id_t frame;
        if (Has(b1_, x)) {  // case II
            GrowP();
            frame = Replace(x);
        } else if (Has(b2_, x)) {  // case III
            ShrinkP();
            frame = Replace(x);
        } else {  // case IV
            const size_t l1 = t1_.size() + b1_.size();
            const size_t total = l1 + t2_.size() + b2_.size();
            if (l1 == c_) {
                if (t1_.size() < c_) {
                    b1_.pop_front();
                    frame = Replace(x);
                } else {
                    frame = Take(t1_, false);  // no ghost
                }
            } else {
                if (total == 2 * c_) b2_.pop_front();
                frame = Replace(x);
            }
        }
        prepared_ = true;
        prepared_frame_ = frame;
        prepared_page_ = x;
        return frame;
    }

    void Remove(frame_id_t frame_id) {
        if (prepared_ && prepared_frame_ == frame_id) prepared_ = false;
        auto it = FindFrame(frame_id);
        if (it == frames_.end()) return;
        t1_.remove(it->page);
        t2_.remove(it->page);
        frames_.erase(it);
    }

    size_t Size() const {
        return static_cast<size_t>(std::count_if(frames_.begin(), frames_.end(), [](const Entry& e) { return e.evictable; }));
    }

    [[nodiscard]] ArcSizes Sizes() const { return {t1_.size(), t2_.size(), b1_.size(), b2_.size(), p_}; }
    [[nodiscard]] size_t Capacity() const { return c_; }

private:
    struct Entry {
        frame_id_t frame;
        page_id_t page;
        bool evictable;
    };
    using Pages = std::list<page_id_t>;

    static bool Has(const Pages& list, page_id_t page) { return std::find(list.begin(), list.end(), page) != list.end(); }
    bool Known(page_id_t page) const { return Has(t1_, page) || Has(t2_, page) || Has(b1_, page) || Has(b2_, page); }

    std::list<Entry>::iterator FindFrame(frame_id_t frame_id) {
        return std::find_if(frames_.begin(), frames_.end(), [&](const Entry& e) { return e.frame == frame_id; });
    }
    std::list<Entry>::const_iterator FindPage(page_id_t page) const {
        return std::find_if(frames_.begin(), frames_.end(), [&](const Entry& e) { return e.page == page; });
    }

    void GrowP() {
        const auto b1 = static_cast<double>(b1_.size()), b2 = static_cast<double>(b2_.size());
        p_ = std::min(static_cast<double>(c_), p_ + (b1 >= b2 ? 1.0 : b2 / b1));
    }
    void ShrinkP() {
        const auto b1 = static_cast<double>(b1_.size()), b2 = static_cast<double>(b2_.size());
        p_ = std::max(0.0, p_ - (b2 >= b1 ? 1.0 : b1 / b2));
    }

    // REPLACE: the paper's pick between T1 and T2, then that list's oldest evictable page
    frame_id_t Replace(page_id_t x) {
        const auto size_t1 = static_cast<double>(t1_.size());
        const bool from_t1 = !t1_.empty() && ((Has(b2_, x) && size_t1 == p_) || size_t1 > p_);
        return Take(from_t1 ? t1_ : t2_, true);
    }

    // Takes the oldest evictable page of `list`, or of the other cache list if it has none,
    // optionally into its ghost list, and frees its frame
    frame_id_t Take(Pages& list, bool ghost) {
        Pages* from = &list;
        auto it = OldestEvictable(*from);
        if (it == from->end()) {
            from = from == &t1_ ? &t2_ : &t1_;
            it = OldestEvictable(*from);
        }
        const page_id_t page = *it;
        from->erase(it);
        if (ghost) (from == &t1_ ? b1_ : b2_).push_back(page);
        const auto entry = FindPage(page);
        const frame_id_t frame = entry->frame;
        frames_.erase(entry);
        return frame;
    }

    Pages::iterator OldestEvictable(Pages& list) {
        return std::find_if(list.begin(), list.end(), [&](page_id_t page) { return FindPage(page)->evictable; });
    }

    size_t c_;
    double p_{0};
    Pages t1_, t2_, b1_, b2_;
    std::list<Entry> frames_;
    bool prepared_{false};
    frame_id_t prepared_frame_{0};
    page_id_t prepared_page_{INVALID_PAGE_ID};
};

// After every call: the real policy keeps the invariants and has the reference's sizes and p
std::string CheckAgainstReference(const ArcEvictionPolicy& arc, const ReferenceArc& ref) {
    const ArcSizes s = arc.Sizes();
    if (std::string problem = InvariantProblem(s, ref.Capacity()); !problem.empty()) return problem;
    if (!SameSizes(s, ref.Sizes())) return Describe(s) + ", reference " + Describe(ref.Sizes());
    return "";
}

// Plain feeding: every access a new use
void Feed(PolicyHarness& harness, const std::vector<page_id_t>& pages) {
    for (const page_id_t page : pages) harness.Access(page, AccessType::NEW_USE);
}

}  // namespace

TEST(ArcEvictionPolicyTest, EvictsInItsOrder) {
    ArcEvictionPolicy arc(3);
    arc.RecordAccess(0, 10, AccessType::NEW_USE);  // T1 [10]
    arc.RecordAccess(1, 11, AccessType::NEW_USE);  // T1 [10 11]
    arc.RecordAccess(0, 10, AccessType::NEW_USE);  // used twice: T1 [11], T2 [10]
    arc.RecordAccess(2, 12, AccessType::NEW_USE);  // T1 [11 12]

    // T1 empties first while it is above its target p = 0 (LRU would take 11, 10, 12);
    // each victim leaves a ghost
    EXPECT_EQ(arc.Evict(), std::optional<frame_id_t>(1));
    EXPECT_EQ(arc.Evict(), std::optional<frame_id_t>(2));
    EXPECT_EQ(arc.Evict(), std::optional<frame_id_t>(0));
    EXPECT_FALSE(arc.Evict().has_value());
    EXPECT_EQ(Describe(arc.Sizes()), "T1 0 T2 0 B1 2 B2 1 p 0");

    // when T1 alone fills the cache its oldest page leaves without a ghost
    ArcEvictionPolicy full(2);
    full.RecordAccess(0, 10, AccessType::NEW_USE);
    full.RecordAccess(1, 11, AccessType::NEW_USE);
    EXPECT_EQ(full.Evict(), std::optional<frame_id_t>(0));
    EXPECT_EQ(Describe(full.Sizes()), "T1 1 T2 0 B1 0 B2 0 p 0");
}

TEST(ArcEvictionPolicyTest, RepeatChangesNothing) {
    ArcEvictionPolicy arc(3);
    arc.RecordAccess(0, 10, AccessType::NEW_USE);
    arc.RecordAccess(1, 11, AccessType::NEW_USE);  // T1 [10 11]
    const std::string before = Describe(arc.Sizes());
    arc.RecordAccess(0, 10, AccessType::REPEAT);   // as a new use, 10 would go to T2 and 11 first
    EXPECT_EQ(Describe(arc.Sizes()), before);

    EXPECT_EQ(arc.Evict(), std::optional<frame_id_t>(0));

    // a repeat on a frame it doesn't track starts nothing
    ArcEvictionPolicy other(3);
    other.RecordAccess(0, 10, AccessType::REPEAT);
    EXPECT_EQ(other.Size(), 0u);
    EXPECT_FALSE(other.Evict().has_value());
}

TEST(ArcEvictionPolicyTest, TouchStaysInItsList) {
    // a page in T1 touched stays in T1, at its newest end
    ArcEvictionPolicy arc(3);
    arc.RecordAccess(0, 10, AccessType::NEW_USE);
    arc.RecordAccess(1, 11, AccessType::NEW_USE);  // T1 [10 11]
    arc.RecordAccess(0, 10, AccessType::TOUCH);    // T1 [11 10]
    EXPECT_EQ(Describe(arc.Sizes()), "T1 2 T2 0 B1 0 B2 0 p 0");
    EXPECT_EQ(arc.Evict(), std::optional<frame_id_t>(1));  // untouched, 10 would go

    // the same in T2
    ArcEvictionPolicy t2(2);
    t2.RecordAccess(0, 10, AccessType::NEW_USE);
    t2.RecordAccess(0, 10, AccessType::NEW_USE);  // T2 [10]
    t2.RecordAccess(1, 11, AccessType::NEW_USE);
    t2.RecordAccess(1, 11, AccessType::NEW_USE);  // T2 [10 11]
    t2.RecordAccess(0, 10, AccessType::TOUCH);    // T2 [11 10]
    EXPECT_EQ(Describe(t2.Sizes()), "T1 0 T2 2 B1 0 B2 0 p 0");
    EXPECT_EQ(t2.Evict(), std::optional<frame_id_t>(1));

    // a load sent as a touch is still a load into T1: the page wasn't resident
    ArcEvictionPolicy load(2);
    load.RecordAccess(0, 10, AccessType::TOUCH);
    EXPECT_EQ(Describe(load.Sizes()), "T1 1 T2 0 B1 0 B2 0 p 0");
    EXPECT_EQ(load.Size(), 1u);
}

TEST(ArcEvictionPolicyTest, NonEvictableSkipped) {
    ArcEvictionPolicy arc(3);
    arc.RecordAccess(0, 10, AccessType::NEW_USE);
    arc.RecordAccess(1, 11, AccessType::NEW_USE);  // T1 [10 11]

    arc.SetEvictable(0, false);
    EXPECT_EQ(arc.Evict(), std::optional<frame_id_t>(1));  // T1's oldest evictable page

    // still T1's oldest once it is evictable again
    arc.RecordAccess(1, 12, AccessType::NEW_USE);
    arc.SetEvictable(0, true);
    EXPECT_EQ(arc.Evict(), std::optional<frame_id_t>(0));
}

TEST(ArcEvictionPolicyTest, NoEvictableReturnsNulloptAndKeepsState) {
    ArcEvictionPolicy arc(3);
    ArcEvictionPolicy fresh(3);
    for (ArcEvictionPolicy* p : {&arc, &fresh}) {
        p->RecordAccess(0, 10, AccessType::NEW_USE);
        p->RecordAccess(1, 11, AccessType::NEW_USE);
        p->RecordAccess(2, 12, AccessType::NEW_USE);
        p->RecordAccess(1, 11, AccessType::NEW_USE);
        ASSERT_EQ(p->Evict(), std::optional<frame_id_t>(0));  // leaves a ghost, and a preparation
        p->RecordAccess(0, 13, AccessType::NEW_USE);
    }

    for (frame_id_t f = 0; f < 3; ++f) arc.SetEvictable(f, false);
    const std::string before = Describe(arc.Sizes());
    EXPECT_FALSE(arc.Evict(10).has_value());  // a ghost would move p
    EXPECT_FALSE(arc.Evict().has_value());
    EXPECT_EQ(arc.Size(), 0u);
    EXPECT_EQ(Describe(arc.Sizes()), before);
    for (frame_id_t f = 0; f < 3; ++f) arc.SetEvictable(f, true);

    // it goes on exactly like a policy that never tried
    page_id_t page_in[3] = {13, 11, 12};
    for (page_id_t page = 20; page < 40; ++page) {
        // every third page is one from 4 loads back, often a ghost by now
        page_id_t incoming = page % 3 == 0 ? page - 4 : page;
        if (std::find(std::begin(page_in), std::end(page_in), incoming) != std::end(page_in)) incoming = page;
        const auto victim = arc.Evict(incoming);
        ASSERT_EQ(victim, fresh.Evict(incoming)) << "page " << page;
        ASSERT_TRUE(victim.has_value());
        page_in[*victim] = incoming;
        arc.RecordAccess(*victim, incoming, AccessType::NEW_USE);
        fresh.RecordAccess(*victim, incoming, AccessType::NEW_USE);
        const frame_id_t other = (*victim + 1) % 3;
        const AccessType type = page % 2 == 0 ? AccessType::NEW_USE : AccessType::TOUCH;
        arc.RecordAccess(other, page_in[other], type);
        fresh.RecordAccess(other, page_in[other], type);
        ASSERT_TRUE(SameSizes(arc.Sizes(), fresh.Sizes())) << Describe(arc.Sizes()) << " vs " << Describe(fresh.Sizes());
    }
}

TEST(ArcEvictionPolicyTest, RemoveForgetsFrame) {
    ArcEvictionPolicy arc(3);
    arc.RecordAccess(0, 10, AccessType::NEW_USE);
    arc.RecordAccess(0, 10, AccessType::NEW_USE);  // T2 [10]
    arc.RecordAccess(1, 11, AccessType::NEW_USE);  // T1 [11]

    arc.Remove(0);  // 10 leaves without a ghost
    EXPECT_EQ(arc.Size(), 1u);
    EXPECT_EQ(Describe(arc.Sizes()), "T1 1 T2 0 B1 0 B2 0 p 0");

    // back in, 10 is a page never seen: T1, where a ghost would have gone to T2
    arc.RecordAccess(0, 10, AccessType::NEW_USE);
    EXPECT_EQ(Describe(arc.Sizes()), "T1 2 T2 0 B1 0 B2 0 p 0");

    arc.Remove(2);  // never tracked: nothing happens
    EXPECT_EQ(arc.Size(), 2u);

    // Remove also drops a preparation for its frame: the load there then moves p itself
    for (const bool remove : {false, true}) {
        ArcEvictionPolicy two(2);
        PolicyHarness harness(two, 2);
        Feed(harness, {1, 1, 2, 3});  // T1 [3], T2 [1], B1 [2]
        const auto frame = two.Evict(2);  // B1 hit: p 0 -> 1, and 1 goes to B2
        ASSERT_EQ(frame, std::optional<frame_id_t>(0));
        if (remove) two.Remove(*frame);
        two.RecordAccess(*frame, 2, AccessType::NEW_USE);
        EXPECT_EQ(Describe(two.Sizes()), remove ? "T1 1 T2 1 B1 0 B2 1 p 2" : "T1 1 T2 1 B1 0 B2 1 p 1");
    }
}

TEST(ArcEvictionPolicyTest, SizeCountsEvictableOnly) {
    ArcEvictionPolicy arc(3);
    arc.RecordAccess(0, 10, AccessType::NEW_USE);
    arc.RecordAccess(1, 11, AccessType::NEW_USE);
    EXPECT_EQ(arc.Size(), 2u);  // both evictable once loaded

    arc.SetEvictable(0, false);
    EXPECT_EQ(arc.Size(), 1u);
    arc.SetEvictable(0, false);  // twice changes nothing
    EXPECT_EQ(arc.Size(), 1u);

    arc.SetEvictable(0, true);
    EXPECT_EQ(arc.Size(), 2u);
    arc.RecordAccess(0, 10, AccessType::NEW_USE);  // a hit doesn't count twice
    EXPECT_EQ(arc.Size(), 2u);
}

TEST(ArcEvictionPolicyTest, OutOfRangeFrameThrows) {
    ArcEvictionPolicy arc(3);

    EXPECT_THROW(arc.RecordAccess(3, 1, AccessType::NEW_USE), DatabaseException);
    EXPECT_THROW(arc.RecordAccess(3, 1, AccessType::REPEAT), DatabaseException);
    EXPECT_THROW(arc.RecordAccess(3, 1), DatabaseException);
    EXPECT_THROW(arc.SetEvictable(3, true), DatabaseException);
    EXPECT_THROW(arc.Remove(3), DatabaseException);

    EXPECT_EQ(arc.Size(), 0u);
    EXPECT_FALSE(arc.Evict().has_value());
    EXPECT_FALSE(arc.Evict(5).has_value());
}

TEST(ArcEvictionPolicyTest, NeedsThePageId) {
    ArcEvictionPolicy arc(3);
    EXPECT_THROW(arc.RecordAccess(0), DatabaseException);
    EXPECT_THROW(arc.RecordAccess(0, INVALID_PAGE_ID, AccessType::NEW_USE), DatabaseException);
    EXPECT_EQ(arc.Size(), 0u);

    // the two-argument call is a new use
    arc.RecordAccess(0, 10);
    EXPECT_EQ(arc.Size(), 1u);
    EXPECT_TRUE(arc.UsesAccessTypes());

    // a page lives in one frame only
    EXPECT_THROW(arc.RecordAccess(1, 10, AccessType::NEW_USE), DatabaseException);
    EXPECT_EQ(arc.Size(), 1u);
    EXPECT_EQ(Describe(arc.Sizes()), "T1 1 T2 0 B1 0 B2 0 p 0");
}

TEST(ArcEvictionPolicyTest, EvictedFrameCanBeTrackedAgain) {
    ArcEvictionPolicy arc(3);

    arc.RecordAccess(0, 10, AccessType::NEW_USE);
    EXPECT_EQ(arc.Evict(), std::optional<frame_id_t>(0));
    EXPECT_EQ(arc.Size(), 0u);
    EXPECT_EQ(Describe(arc.Sizes()), "T1 0 T2 0 B1 1 B2 0 p 0");

    // an evicted page leaves a ghost: back in, it goes straight to T2 and p grows
    arc.RecordAccess(0, 10, AccessType::NEW_USE);
    arc.RecordAccess(1, 11, AccessType::NEW_USE);
    EXPECT_EQ(arc.Size(), 2u);
    EXPECT_EQ(Describe(arc.Sizes()), "T1 1 T2 1 B1 0 B2 0 p 1");
    EXPECT_EQ(arc.Evict(), std::optional<frame_id_t>(0));  // T1 is not above p = 1, so T2 goes
    EXPECT_EQ(arc.Evict(), std::optional<frame_id_t>(1));
    EXPECT_FALSE(arc.Evict().has_value());
}

TEST(ArcEvictionPolicyTest, SetEvictableOnUntrackedFrameIgnored) {
    ArcEvictionPolicy arc(3);

    arc.SetEvictable(1, true);
    EXPECT_EQ(arc.Size(), 0u);
    EXPECT_FALSE(arc.Evict().has_value());

    arc.RecordAccess(1, 10, AccessType::NEW_USE);
    EXPECT_EQ(arc.Size(), 1u);
}

TEST(ArcEvictionPolicyTest, ResetEqualsNewPolicy) {
    constexpr size_t kPool = 8;
    ArcEvictionPolicy arc(kPool);
    std::mt19937 rng(7);
    {
        PolicyHarness harness(arc, kPool);
        VisitTypes visits(64);
        for (int i = 0; i < 500; ++i) {
            const page_id_t page = rng() % 40;
            harness.Access(page, visits.Next(page, harness.Resident(page)));
        }
        harness.Hold(rng() % 40);  // leaves a pinned frame behind
        (void)arc.Evict(7);        // and a preparation
    }

    arc.Reset();
    EXPECT_EQ(arc.Size(), 0u);
    EXPECT_FALSE(arc.Evict().has_value());
    EXPECT_EQ(Describe(arc.Sizes()), "T1 0 T2 0 B1 0 B2 0 p 0");

    // from here it acts exactly like a new policy: no ghosts, no p, no preparation
    ArcEvictionPolicy fresh(kPool);
    PolicyHarness a(arc, kPool), b(fresh, kPool);
    VisitTypes va(64), vb(64);
    for (int i = 0; i < 20000; ++i) {
        const page_id_t page = rng() % 40;
        ASSERT_EQ(a.Access(page, va.Next(page, a.Resident(page))), b.Access(page, vb.Next(page, b.Resident(page)))) << "op " << i;
        ASSERT_EQ(a.LastVictim(), b.LastVictim()) << "op " << i;
        ASSERT_EQ(arc.Size(), fresh.Size()) << "op " << i;
        ASSERT_TRUE(SameSizes(arc.Sizes(), fresh.Sizes())) << "op " << i;
    }
}

TEST(ArcEvictionPolicyTest, LargePageIdGrows) {
    ArcEvictionPolicy arc(2);
    PolicyHarness harness(arc, 2);

    EXPECT_TRUE(harness.Access(1'000'000, AccessType::NEW_USE));   // T1
    EXPECT_TRUE(harness.Access(5, AccessType::NEW_USE));           // T1
    EXPECT_FALSE(harness.Access(1'000'000, AccessType::NEW_USE));  // T2

    // the big page is in T2, so the pages used once go before it
    EXPECT_TRUE(harness.Access(6, AccessType::NEW_USE));
    EXPECT_EQ(harness.LastVictim(), std::optional<page_id_t>(5));
    EXPECT_TRUE(harness.Access(7, AccessType::NEW_USE));
    EXPECT_EQ(harness.LastVictim(), std::optional<page_id_t>(6));
    EXPECT_TRUE(harness.Resident(1'000'000));

    // a brand new page as the incoming one
    ArcEvictionPolicy other(1);
    other.RecordAccess(0, 3, AccessType::NEW_USE);
    EXPECT_EQ(other.Evict(INVALID_PAGE_ID), std::optional<frame_id_t>(0));
}

TEST(ArcEvictionPolicyTest, PoolOfOne) {
    ArcEvictionPolicy arc(1);
    arc.RecordAccess(0, 10, AccessType::NEW_USE);
    EXPECT_EQ(arc.Evict(), std::optional<frame_id_t>(0));
    EXPECT_FALSE(arc.Evict().has_value());

    for (const std::optional<uint64_t> window : {std::optional<uint64_t>(), std::optional<uint64_t>(64)}) {
        ArcEvictionPolicy one(1);
        EXPECT_EQ(CountMisses(one, 1, {1, 1, 2, 2, 1, 3}, window), 4u);
    }

    ArcEvictionPolicy one(1);
    PolicyHarness harness(one, 1);
    harness.Access(1, AccessType::NEW_USE);
    harness.Access(1, AccessType::NEW_USE);
    harness.Access(2, AccessType::NEW_USE);
    EXPECT_EQ(harness.LastVictim(), std::optional<page_id_t>(1));
}

TEST(ArcEvictionPolicyTest, TeachingsExample) {
    // A B C D A B E A B C D E
    ArcEvictionPolicy arc(3);
    EXPECT_EQ(CountMisses(arc, 3, {0, 1, 2, 3, 0, 1, 4, 0, 1, 2, 3, 4}), 10u);
}

TEST(ArcEvictionPolicyTest, ScanAfterHotPair) {
    // The pair 1 2 twice, a scan of pages 10..19, then the pair again.
    std::vector<page_id_t> pages = {1, 2, 1, 2};
    for (page_id_t p = 10; p < 20; ++p) pages.push_back(p);
    pages.push_back(1);
    pages.push_back(2);
    ArcEvictionPolicy arc(4);
    EXPECT_EQ(CountMisses(arc, 4, pages), 12u);
}

TEST(ArcEvictionPolicyTest, VisitModeSkipsInterleavedReturns) {
    // Pairs read in turn (10 20 10 20 ...) look used twice unless a quick return is a touch.
    std::vector<page_id_t> pages = {1, 2, 5, 6, 5, 6, 1, 2};
    for (page_id_t k = 0; k < 4; ++k) {
        for (const page_id_t p : {10 + k, 20 + k, 10 + k, 20 + k}) pages.push_back(p);
    }
    pages.push_back(1);
    pages.push_back(2);

    ArcEvictionPolicy visit(4);
    EXPECT_EQ(CountMisses(visit, 4, pages, 3), 12u);
    ArcEvictionPolicy plain(4);
    EXPECT_EQ(CountMisses(plain, 4, pages), 15u);
}

TEST(ArcEvictionPolicyTest, GhostHitMovesP) {
    ArcEvictionPolicy arc(3);
    PolicyHarness harness(arc, 3);
    Feed(harness, {1, 1, 2, 2, 3, 3, 4, 5, 5, 7});  // 4 evicted into B1, then 1 and 2 into B2
    ASSERT_EQ(Describe(arc.Sizes()), "T1 1 T2 2 B1 1 B2 2 p 0");
    ASSERT_FALSE(harness.Resident(4));

    // a B1 hit: p grows by |B2| / |B1| = 2, T1 (1 page) is not above p, so T2's oldest (3)
    // goes to B2, and 4 comes back into T2
    EXPECT_TRUE(harness.Access(4, AccessType::NEW_USE));
    EXPECT_EQ(harness.LastVictim(), std::optional<page_id_t>(3));
    EXPECT_EQ(Describe(arc.Sizes()), "T1 1 T2 2 B1 0 B2 3 p 2");
    EXPECT_TRUE(harness.Resident(4));

    // a B2 hit: p shrinks by 1 (|B2| >= |B1|); now T1 equals p and the page is from B2, so
    // T1's page (7) goes to B1
    EXPECT_TRUE(harness.Access(1, AccessType::NEW_USE));
    EXPECT_EQ(harness.LastVictim(), std::optional<page_id_t>(7));
    EXPECT_EQ(Describe(arc.Sizes()), "T1 0 T2 3 B1 1 B2 2 p 1");
}

TEST(ArcEvictionPolicyTest, IncomingInvalidPageIsNew) {
    // Evict(INVALID_PAGE_ID) and Evict(a page never seen) decide the same
    ArcEvictionPolicy invalid(3), unseen(3);
    PolicyHarness a(invalid, 3), b(unseen, 3);
    for (PolicyHarness* h : {&a, &b}) Feed(*h, {1, 1, 2, 2, 3, 3, 4, 5, 5, 7});

    const auto frame = invalid.Evict(INVALID_PAGE_ID);
    ASSERT_EQ(frame, unseen.Evict(50));
    ASSERT_TRUE(frame.has_value());
    EXPECT_TRUE(SameSizes(invalid.Sizes(), unseen.Sizes()));
    EXPECT_EQ(Describe(invalid.Sizes()), "T1 0 T2 2 B1 2 B2 1 p 0");  // made room: 7 to B1, 1 dropped from B2

    // and the load of a page never seen just goes into T1
    invalid.RecordAccess(*frame, 50, AccessType::NEW_USE);
    unseen.RecordAccess(*frame, 50, AccessType::NEW_USE);
    EXPECT_TRUE(SameSizes(invalid.Sizes(), unseen.Sizes()));
    EXPECT_EQ(Describe(invalid.Sizes()), "T1 1 T2 2 B1 2 B2 1 p 0");

    // nothing was grown for INVALID_PAGE_ID, and a big page still works
    ArcEvictionPolicy big(1);
    big.RecordAccess(0, 3, AccessType::NEW_USE);
    const auto victim = big.Evict(INVALID_PAGE_ID);
    ASSERT_EQ(victim, std::optional<frame_id_t>(0));
    big.RecordAccess(0, 1'000'000, AccessType::NEW_USE);
    EXPECT_EQ(big.Size(), 1u);
    EXPECT_EQ(Describe(big.Sizes()), "T1 1 T2 0 B1 0 B2 0 p 0");
}

TEST(ArcEvictionPolicyTest, IncomingResidentPageOnlyPicksAVictim) {
    // the page is already cached, so REPLACE picks a victim and nothing is prepared
    ArcEvictionPolicy arc(2);
    arc.RecordAccess(0, 10, AccessType::NEW_USE);
    arc.RecordAccess(1, 11, AccessType::NEW_USE);  // T1 [10 11]
    EXPECT_EQ(arc.Evict(11), std::optional<frame_id_t>(0));
    EXPECT_EQ(Describe(arc.Sizes()), "T1 1 T2 0 B1 1 B2 0 p 0");  // 10 to B1

    // so the next load there makes its own room: |T1| + |B1| = c drops B1's ghost first
    arc.RecordAccess(0, 12, AccessType::NEW_USE);
    EXPECT_EQ(Describe(arc.Sizes()), "T1 2 T2 0 B1 0 B2 0 p 0");
}

TEST(ArcEvictionPolicyTest, UnpreparedLoadKeepsInvariants) {
    // Loads into frames freed by Remove: no Evict comes first, so the load does the paper's
    // work without REPLACE
    ArcEvictionPolicy arc(3);
    PolicyHarness harness(arc, 3);
    const auto expect = [&](page_id_t page, const char* sizes) {
        harness.Access(page, AccessType::NEW_USE);
        EXPECT_EQ(InvariantProblem(arc.Sizes(), 3), "") << "page " << page;
        EXPECT_EQ(Describe(arc.Sizes()), sizes) << "page " << page;
    };

    Feed(harness, {1, 1, 2, 2, 3, 3, 4, 5, 5, 7});  // T1 [7], T2 [3 5], B1 [4], B2 [1 2]
    harness.Drop(7);
    expect(4, "T1 0 T2 3 B1 0 B2 2 p 2");  // a B1 ghost: p grows by 2/1, into T2
    harness.Drop(3);
    expect(1, "T1 0 T2 3 B1 0 B2 1 p 1");  // a B2 ghost: p shrinks by 1

    // with |T1| + |B1| = c a new page first drops B1's oldest ghost
    ArcEvictionPolicy full(3);
    PolicyHarness h(full, 3);
    Feed(h, {1, 1, 2, 3, 4});  // T1 [3 4], T2 [1], B1 [2]
    h.Drop(1);
    h.Access(5, AccessType::NEW_USE);
    EXPECT_EQ(InvariantProblem(full.Sizes(), 3), "");
    EXPECT_EQ(Describe(full.Sizes()), "T1 3 T2 0 B1 0 B2 0 p 0");

    // a page never seen loaded into a free frame of a new policy just goes into T1
    ArcEvictionPolicy empty(2);
    empty.RecordAccess(1, 8, AccessType::NEW_USE);
    EXPECT_EQ(Describe(empty.Sizes()), "T1 1 T2 0 B1 0 B2 0 p 0");
}

TEST(ArcEvictionPolicyTest, FailedLoadAfterEvictKeepsInvariants) {
    // Evict(4) prepares for 4, but the read fails and page 9 is loaded into the frame instead
    {
        ArcEvictionPolicy arc(3);
        PolicyHarness harness(arc, 3);
        Feed(harness, {1, 1, 2, 2, 3, 3, 4, 5, 5, 7});  // T1 [7], T2 [3 5], B1 [4], B2 [1 2]
        const auto frame = arc.Evict(4);  // B1 hit: p grows to 2, 3 goes to B2
        ASSERT_TRUE(frame.has_value());
        EXPECT_EQ(Describe(arc.Sizes()), "T1 1 T2 1 B1 1 B2 3 p 2");

        // the directory is full, so 9 first drops B2's oldest ghost
        arc.RecordAccess(*frame, 9, AccessType::NEW_USE);
        EXPECT_EQ(InvariantProblem(arc.Sizes(), 3), "");
        EXPECT_EQ(Describe(arc.Sizes()), "T1 2 T2 1 B1 1 B2 2 p 2");
    }
    // prepared for a new page, then a ghost loads into the frame: it moves p itself
    {
        ArcEvictionPolicy arc(3);
        PolicyHarness harness(arc, 3);
        Feed(harness, {1, 1, 2, 2, 3, 3, 4, 5, 5, 7});
        const auto frame = arc.Evict(INVALID_PAGE_ID);  // 7 to B1, 1 dropped from B2
        ASSERT_TRUE(frame.has_value());
        arc.RecordAccess(*frame, 4, AccessType::NEW_USE);  // B1 [4 7], B2 [2]: p grows by 1
        EXPECT_EQ(InvariantProblem(arc.Sizes(), 3), "");
        EXPECT_EQ(Describe(arc.Sizes()), "T1 0 T2 3 B1 1 B2 1 p 1");
    }

    // random runs where a third of the reads fail and the next miss takes the freed frame,
    // checked against the reference after every call
    for (const unsigned seed : {1u, 2u, 3u}) {
        std::mt19937 rng(seed);
        for (const size_t pool : {1u, 2u, 3u, 5u, 9u}) {
            ArcEvictionPolicy arc(pool);
            Twin<ReferenceArc> twin(arc, pool);
            twin.SetAfterCall([&] { return CheckAgainstReference(arc, twin.GetReference()); });
            std::vector<page_id_t> page_in(pool, INVALID_PAGE_ID);
            std::deque<frame_id_t> free;
            for (size_t f = 0; f < pool; ++f) free.push_back(static_cast<frame_id_t>(f));
            const auto pages = static_cast<uint32_t>(3 * pool + 4);

            for (int i = 0; i < 3000 && twin.Error().empty(); ++i) {
                const page_id_t page = rng() % pages;
                if (const auto at = std::find(page_in.begin(), page_in.end(), page); at != page_in.end()) {
                    const AccessType type = rng() % 2 == 0 ? AccessType::NEW_USE : AccessType::TOUCH;
                    twin.RecordAccess(static_cast<frame_id_t>(at - page_in.begin()), page, type);
                    continue;
                }
                frame_id_t frame;
                if (!free.empty()) {
                    frame = free.front();
                    free.pop_front();
                } else {
                    const auto victim = twin.Evict(page);
                    ASSERT_TRUE(victim.has_value());
                    frame = *victim;
                    page_in[frame] = INVALID_PAGE_ID;
                    if (rng() % 3 == 0) {  // the read failed: the frame goes back on the free list
                        free.push_back(frame);
                        continue;
                    }
                }
                twin.RecordAccess(frame, page, AccessType::NEW_USE);
                page_in[frame] = page;
            }
            EXPECT_TRUE(twin.Error().empty()) << twin.Error() << " (seed " << seed << ", pool " << pool << ")";
        }
    }
}

TEST(ArcEvictionPolicyTest, PinnedLruTakesOtherList) {
    // REPLACE picks T1, but every T1 page is pinned: T2's oldest evictable page goes, to B2
    ArcEvictionPolicy arc(3);
    PolicyHarness harness(arc, 3);
    Feed(harness, {1, 1, 2, 3});  // T1 [2 3], T2 [1]
    harness.Hold(2);
    harness.Hold(3);
    EXPECT_TRUE(harness.Access(4, AccessType::NEW_USE));
    EXPECT_EQ(harness.LastVictim(), std::optional<page_id_t>(1));
    EXPECT_EQ(Describe(arc.Sizes()), "T1 3 T2 0 B1 0 B2 1 p 0");

    // and the other way: REPLACE picks T2, every T2 page is pinned, so T1's page goes to B1
    ArcEvictionPolicy other(3);
    PolicyHarness h(other, 3);
    Feed(h, {1, 1, 2, 2, 3, 3, 4, 5, 5, 7, 4});  // T1 [7], T2 [5 4], B2 [1 2 3], p 2
    h.Hold(5);
    h.Hold(4);
    EXPECT_TRUE(h.Access(8, AccessType::NEW_USE));  // the full directory drops 1 from B2 first
    EXPECT_EQ(h.LastVictim(), std::optional<page_id_t>(7));
    EXPECT_EQ(Describe(other.Sizes()), "T1 1 T2 2 B1 1 B2 2 p 2");
}

TEST(ArcEvictionPolicyTest, MatchesReferenceOnRandomOps) {
    const auto result = test::MatchReferenceOnRandomOps<ArcEvictionPolicy, ReferenceArc>(
        {1, 2, 3, 5, 9, 17, 30, 40}, CheckAgainstReference);
    EXPECT_TRUE(result.error.empty()) << result.error;
    EXPECT_GE(result.ops, 200000u);
}

}  // namespace shilmandb
