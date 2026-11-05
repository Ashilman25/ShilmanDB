#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <list>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "buffer/lirs_eviction_policy.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "common/exception.hpp"
#include "eviction_policy_harness.hpp"

namespace shilmandb {

namespace {

using test::CountMisses;
using test::PolicyHarness;
using test::VisitTypes;

std::string Describe(const LirsSizes& s) {
    std::ostringstream out;
    out << "LIR " << s.lir << " HIR " << s.hir << " S " << s.stack;
    return out.str();
}

bool SameSizes(const LirsSizes& a, const LirsSizes& b) {
    return a.lir == b.lir && a.hir == b.hir && a.stack == b.stack;
}

// All but 1% of the pool (at least one frame) is for LIR pages
size_t LirLimit(size_t pool) {
    return pool - static_cast<size_t>(std::max(1L, std::lround(0.01 * static_cast<double>(pool))));
}

// LIRS written the slow, literal way (Jiang & Zhang, SIGMETRICS 2002): the stack S and the
// queue Q are std::lists (S's front is its bottom, Q's front the next victim), the LIR pages
// another std::list, the tracked frames another, and every lookup a linear search. Pins
// follow the same rules as the real policy.
class ReferenceLirs {
public:
    explicit ReferenceLirs(size_t num_frames) : pool_(num_frames), lir_limit_(LirLimit(num_frames)) {}

    void RecordAccess(frame_id_t frame_id, page_id_t page_id, AccessType type) {
        if (type == AccessType::REPEAT) return;
        auto frame = FindFrame(frame_id);
        if (frame != frames_.end() && frame->page == page_id) {
            if (IsLir(page_id)) {  // a touch or a new use: to the top of S
                ToTop(page_id);
                Prune();
            } else if (type == AccessType::TOUCH) {  // to the back of Q, S as it is
                q_.remove(page_id);
                q_.push_back(page_id);
            } else if (Has(s_, page_id)) {  // a HIR page back while still in S becomes LIR
                ToTop(page_id);
                q_.remove(page_id);
                lir_.push_back(page_id);
                DemoteBottom();
            } else {
                ToTop(page_id);
                q_.remove(page_id);
                q_.push_back(page_id);
            }
            return;
        }

        // a load; a frame that still held another page lets it go, as if evicted
        if (frame != frames_.end()) {
            Unload(frame->page);
            frame->page = page_id;
        } else {
            frames_.push_back(Entry{frame_id, page_id, true});
        }
        if (lir_.size() < lir_limit_) {
            ToTop(page_id);
            lir_.push_back(page_id);
        } else if (Has(s_, page_id)) {
            ToTop(page_id);
            lir_.push_back(page_id);
            DemoteBottom();
        } else {
            ToTop(page_id);
            q_.push_back(page_id);
        }
    }

    void SetEvictable(frame_id_t frame_id, bool evictable) {
        if (auto it = FindFrame(frame_id); it != frames_.end()) it->evictable = evictable;
    }

    std::optional<frame_id_t> Evict(page_id_t /*incoming_page*/) {
        // Q's oldest evictable page; it keeps its place in S
        for (const page_id_t page : q_) {
            if (Evictable(page)) return Free(page);
        }
        // every resident HIR page is pinned: the evictable LIR page nearest the bottom of S
        for (const page_id_t page : s_) {
            if (IsLir(page) && Evictable(page)) return Free(page);
        }
        return std::nullopt;
    }

    void Remove(frame_id_t frame_id) {
        auto it = FindFrame(frame_id);
        if (it == frames_.end()) return;
        const page_id_t page = it->page;
        lir_.remove(page);
        q_.remove(page);
        s_.remove(page);
        frames_.erase(it);
        Prune();
    }

    size_t Size() const {
        return static_cast<size_t>(std::count_if(frames_.begin(), frames_.end(), [](const Entry& e) { return e.evictable; }));
    }

    [[nodiscard]] LirsSizes Sizes() const { return {lir_.size(), q_.size(), s_.size()}; }
    [[nodiscard]] size_t Pool() const { return pool_; }
    [[nodiscard]] size_t Tracked() const { return frames_.size(); }

private:
    struct Entry {
        frame_id_t frame;
        page_id_t page;
        bool evictable;
    };
    using Pages = std::list<page_id_t>;

    static bool Has(const Pages& list, page_id_t page) { return std::find(list.begin(), list.end(), page) != list.end(); }
    bool IsLir(page_id_t page) const { return Has(lir_, page); }

    std::list<Entry>::iterator FindFrame(frame_id_t frame_id) {
        return std::find_if(frames_.begin(), frames_.end(), [&](const Entry& e) { return e.frame == frame_id; });
    }
    std::list<Entry>::iterator FindPage(page_id_t page) {
        return std::find_if(frames_.begin(), frames_.end(), [&](const Entry& e) { return e.page == page; });
    }
    bool Evictable(page_id_t page) { return FindPage(page)->evictable; }

    void ToTop(page_id_t page) {
        s_.remove(page);
        s_.push_back(page);
    }
    void Prune() {
        while (!s_.empty() && !IsLir(s_.front())) s_.pop_front();
    }
    void DemoteBottom() {
        const page_id_t bottom = s_.front();
        s_.pop_front();
        lir_.remove(bottom);
        q_.push_back(bottom);
        Prune();
    }

    // The page stops being resident: a HIR page leaves Q (its place in S stays), an LIR page
    // leaves S and the LIR set
    void Unload(page_id_t page) {
        if (IsLir(page)) {
            lir_.remove(page);
            s_.remove(page);
            Prune();
        } else {
            q_.remove(page);
        }
    }

    frame_id_t Free(page_id_t page) {
        Unload(page);
        const auto entry = FindPage(page);
        const frame_id_t frame = entry->frame;
        frames_.erase(entry);
        return frame;
    }

    size_t pool_;
    size_t lir_limit_;
    Pages s_, q_, lir_;
    std::list<Entry> frames_;
};

// After every call: the real policy has the reference's sizes, and they make sense
std::string CheckAgainstReference(const LirsEvictionPolicy& lirs, const ReferenceLirs& ref) {
    const LirsSizes s = lirs.Sizes();
    if (!SameSizes(s, ref.Sizes())) return Describe(s) + ", reference " + Describe(ref.Sizes());
    const bool ok = s.lir <= LirLimit(ref.Pool()) && s.lir + s.hir == ref.Tracked() && s.stack >= s.lir;
    return ok ? "" : "bad sizes at pool " + std::to_string(ref.Pool()) + ": " + Describe(s);
}

// Plain feeding: every access a new use
void Feed(PolicyHarness& harness, const std::vector<page_id_t>& pages) {
    for (const page_id_t page : pages) harness.Access(page, AccessType::NEW_USE);
}

// Loads page f into frame f for every frame
void Fill(LirsEvictionPolicy& lirs, size_t pool) {
    for (size_t f = 0; f < pool; ++f) {
        lirs.RecordAccess(static_cast<frame_id_t>(f), static_cast<page_id_t>(f), AccessType::NEW_USE);
    }
}

}  // namespace

TEST(LirsEvictionPolicyTest, EvictsInItsOrder) {
    LirsEvictionPolicy lirs(3);
    lirs.RecordAccess(0, 10, AccessType::NEW_USE);  // LIR
    lirs.RecordAccess(1, 11, AccessType::NEW_USE);  // LIR: the LIR set (2 of 3 frames) is full
    lirs.RecordAccess(2, 12, AccessType::NEW_USE);  // HIR, in Q
    lirs.RecordAccess(0, 10, AccessType::NEW_USE);  // to the top of S: S [11 12 10]
    EXPECT_EQ(Describe(lirs.Sizes()), "LIR 2 HIR 1 S 3");

    // Q's page goes first though 11 is older (LRU would take 11, 12, 10); then, with Q empty,
    // the LIR pages go from the bottom of S
    EXPECT_EQ(lirs.Evict(), std::optional<frame_id_t>(2));
    EXPECT_EQ(lirs.Evict(), std::optional<frame_id_t>(1));
    EXPECT_EQ(lirs.Evict(), std::optional<frame_id_t>(0));
    EXPECT_FALSE(lirs.Evict().has_value());
    EXPECT_EQ(Describe(lirs.Sizes()), "LIR 0 HIR 0 S 0");
}

TEST(LirsEvictionPolicyTest, HirShareIsOnePercent) {
    // Fill the pool with pages never seen: the first ones are LIR until the LIR set is full,
    // the rest go to Q in order. 1% of the pool, rounded half up, at least one frame.
    const std::vector<std::pair<size_t, std::vector<frame_id_t>>> cases = {
        {2, {1, 0}},                  // 1 HIR frame
        {149, {148, 0}},              // 1.49 rounds to 1
        {150, {148, 149, 0}},         // 1.5 rounds to 2
        {250, {247, 248, 249, 0}},    // 2.5 rounds to 3
        {349, {346, 347, 348, 0}},    // 3.49 rounds to 3
    };
    for (const auto& [pool, victims] : cases) {
        LirsEvictionPolicy lirs(pool);
        Fill(lirs, pool);
        EXPECT_EQ(lirs.Sizes().lir, pool - (victims.size() - 1)) << "pool " << pool;

        // Q oldest first, then the LIR page at the bottom of S
        for (const frame_id_t victim : victims) {
            EXPECT_EQ(lirs.Evict(), std::optional<frame_id_t>(victim)) << "pool " << pool;
        }
    }
}

TEST(LirsEvictionPolicyTest, RepeatChangesNothing) {
    LirsEvictionPolicy lirs(3);
    lirs.RecordAccess(0, 10, AccessType::NEW_USE);
    lirs.RecordAccess(1, 11, AccessType::NEW_USE);
    lirs.RecordAccess(2, 12, AccessType::NEW_USE);  // HIR
    const std::string before = Describe(lirs.Sizes());

    // as a new use, 12 would become LIR and 10 HIR, and 10 would go first
    lirs.RecordAccess(2, 12, AccessType::REPEAT);
    EXPECT_EQ(Describe(lirs.Sizes()), before);
    EXPECT_EQ(lirs.Evict(), std::optional<frame_id_t>(2));

    // a repeat on a frame it doesn't track starts nothing
    LirsEvictionPolicy other(3);
    other.RecordAccess(0, 10, AccessType::REPEAT);
    EXPECT_EQ(other.Size(), 0u);
    EXPECT_FALSE(other.Evict().has_value());
}

TEST(LirsEvictionPolicyTest, TouchOfLirPageMovesItToTop) {
    LirsEvictionPolicy lirs(3);
    lirs.RecordAccess(0, 10, AccessType::NEW_USE);
    lirs.RecordAccess(1, 11, AccessType::NEW_USE);
    lirs.RecordAccess(2, 12, AccessType::NEW_USE);  // S [10 11 12], Q [12]
    lirs.RecordAccess(0, 10, AccessType::TOUCH);    // S [11 12 10]
    EXPECT_EQ(Describe(lirs.Sizes()), "LIR 2 HIR 1 S 3");
    EXPECT_EQ(lirs.Evict(), std::optional<frame_id_t>(2));  // Q
    EXPECT_EQ(lirs.Evict(), std::optional<frame_id_t>(1));  // S's bottom LIR page (untouched, 10)

    // a touch prunes S like a new use: the HIR page left at the bottom drops out of S
    LirsEvictionPolicy pruned(3);
    pruned.RecordAccess(0, 10, AccessType::NEW_USE);
    pruned.RecordAccess(1, 11, AccessType::NEW_USE);
    pruned.RecordAccess(2, 12, AccessType::NEW_USE);
    pruned.RecordAccess(0, 10, AccessType::TOUCH);  // S [11 12 10]
    pruned.RecordAccess(1, 11, AccessType::TOUCH);  // S [12 10 11], then 12 leaves: S [10 11]
    EXPECT_EQ(Describe(pruned.Sizes()), "LIR 2 HIR 1 S 2");

    // out of S, a new use of 12 leaves it HIR (it goes back on top of S and to the back of Q)
    pruned.RecordAccess(2, 12, AccessType::NEW_USE);
    EXPECT_EQ(Describe(pruned.Sizes()), "LIR 2 HIR 1 S 3");
    EXPECT_EQ(pruned.Evict(), std::optional<frame_id_t>(2));

    // a load sent as a touch is still a load: the page wasn't resident
    LirsEvictionPolicy load(2);
    load.RecordAccess(0, 10, AccessType::TOUCH);
    EXPECT_EQ(Describe(load.Sizes()), "LIR 1 HIR 0 S 1");
    EXPECT_EQ(load.Size(), 1u);
}

TEST(LirsEvictionPolicyTest, TouchOfHirPageLeavesStack) {
    // 148 LIR frames and 2 HIR: pages 148 and 149 are in Q, in that order, and in S
    LirsEvictionPolicy lirs(150);
    Fill(lirs, 150);
    ASSERT_EQ(Describe(lirs.Sizes()), "LIR 148 HIR 2 S 150");

    // a touch moves 148 to the back of Q but doesn't make it LIR, and S stays as it is
    lirs.RecordAccess(148, 148, AccessType::TOUCH);
    EXPECT_EQ(Describe(lirs.Sizes()), "LIR 148 HIR 2 S 150");
    EXPECT_EQ(lirs.Evict(), std::optional<frame_id_t>(149));
    EXPECT_EQ(lirs.Evict(), std::optional<frame_id_t>(148));  // made LIR, it would stay, and page 0 would go

    // a HIR page out of S stays out after a touch, so its next new use doesn't make it LIR
    for (const bool touch : {true, false}) {
        LirsEvictionPolicy two(2);
        two.RecordAccess(0, 10, AccessType::NEW_USE);  // LIR
        two.RecordAccess(1, 11, AccessType::NEW_USE);  // HIR: S [10 11], Q [11]
        two.RecordAccess(0, 10, AccessType::NEW_USE);  // S [11 10], pruned to [10]
        two.RecordAccess(1, 11, touch ? AccessType::TOUCH : AccessType::NEW_USE);
        EXPECT_EQ(two.Sizes().stack, touch ? 1u : 2u);

        // 11 is in S only after a new use, and then this second one makes it LIR and 10 HIR
        two.RecordAccess(1, 11, AccessType::NEW_USE);
        EXPECT_EQ(two.Evict(), std::optional<frame_id_t>(touch ? 1 : 0)) << (touch ? "touch" : "new use");
    }
}

TEST(LirsEvictionPolicyTest, NonEvictableSkipped) {
    LirsEvictionPolicy lirs(150);
    Fill(lirs, 150);  // Q [148 149]

    lirs.SetEvictable(148, false);
    EXPECT_EQ(lirs.Evict(), std::optional<frame_id_t>(149));  // Q's oldest evictable page

    // still Q's oldest once it is evictable again
    lirs.RecordAccess(149, 200, AccessType::NEW_USE);  // never seen, so HIR: Q [148 200]
    lirs.SetEvictable(148, true);
    EXPECT_EQ(lirs.Evict(), std::optional<frame_id_t>(148));
}

TEST(LirsEvictionPolicyTest, AllHirPinnedEvictsLir) {
    {
        // 3 LIR frames and 1 HIR; S [1 4 2 3] (bottom first), Q [4]
        LirsEvictionPolicy lirs(4);
        PolicyHarness harness(lirs, 4);
        Feed(harness, {1, 2, 3, 4, 2, 3});
        ASSERT_EQ(Describe(lirs.Sizes()), "LIR 3 HIR 1 S 4");

        // the only HIR page and S's bottom page are pinned: the next LIR page up S goes
        harness.Hold(4);
        harness.Hold(1);
        EXPECT_TRUE(harness.Access(5, AccessType::NEW_USE));
        EXPECT_EQ(harness.LastVictim(), std::optional<page_id_t>(2));
        EXPECT_EQ(Describe(lirs.Sizes()), "LIR 3 HIR 1 S 4");  // 5 fills the LIR set back up: S [1 4 3 5]
        EXPECT_EQ(lirs.Size(), 2u);

        // and it keeps working
        harness.Release(4);
        harness.Release(1);
        EXPECT_EQ(lirs.Size(), 4u);
        EXPECT_TRUE(harness.Access(6, AccessType::NEW_USE));  // Q's page goes again, and 6 is HIR
        EXPECT_EQ(harness.LastVictim(), std::optional<page_id_t>(4));
        EXPECT_EQ(Describe(lirs.Sizes()), "LIR 3 HIR 1 S 5");
        EXPECT_TRUE(harness.Access(4, AccessType::NEW_USE));  // back while in S: LIR, and 1 HIR
        EXPECT_EQ(harness.LastVictim(), std::optional<page_id_t>(6));
        EXPECT_EQ(Describe(lirs.Sizes()), "LIR 3 HIR 1 S 4");
        EXPECT_TRUE(harness.Access(7, AccessType::NEW_USE));
        EXPECT_EQ(harness.LastVictim(), std::optional<page_id_t>(1));
        EXPECT_EQ(lirs.Size(), 4u);
    }
    {
        // the victim is S's bottom page: the HIR page above it then drops out of S
        LirsEvictionPolicy lirs(4);
        PolicyHarness harness(lirs, 4);
        Feed(harness, {1, 2, 3, 4, 2, 3});  // S [1 4 2 3], Q [4]
        harness.Hold(4);
        EXPECT_TRUE(harness.Access(5, AccessType::NEW_USE));
        EXPECT_EQ(harness.LastVictim(), std::optional<page_id_t>(1));
        EXPECT_EQ(Describe(lirs.Sizes()), "LIR 3 HIR 1 S 3");  // S [2 3 5]

        // out of S, 4 stays HIR on its next use, and goes next (kept in S, it would become
        // LIR and 2 would go)
        harness.Release(4);
        EXPECT_FALSE(harness.Access(4, AccessType::NEW_USE));
        EXPECT_EQ(Describe(lirs.Sizes()), "LIR 3 HIR 1 S 4");
        EXPECT_TRUE(harness.Access(6, AccessType::NEW_USE));
        EXPECT_EQ(harness.LastVictim(), std::optional<page_id_t>(4));
    }
}

TEST(LirsEvictionPolicyTest, NoEvictableReturnsNulloptAndKeepsState) {
    LirsEvictionPolicy lirs(3);
    LirsEvictionPolicy fresh(3);
    for (LirsEvictionPolicy* p : {&lirs, &fresh}) {
        p->RecordAccess(0, 10, AccessType::NEW_USE);
        p->RecordAccess(1, 11, AccessType::NEW_USE);
        p->RecordAccess(2, 12, AccessType::NEW_USE);
        p->RecordAccess(1, 11, AccessType::NEW_USE);
    }

    for (frame_id_t f = 0; f < 3; ++f) lirs.SetEvictable(f, false);
    const std::string before = Describe(lirs.Sizes());
    EXPECT_FALSE(lirs.Evict().has_value());
    EXPECT_FALSE(lirs.Evict(13).has_value());
    EXPECT_EQ(lirs.Size(), 0u);
    EXPECT_EQ(Describe(lirs.Sizes()), before);
    for (frame_id_t f = 0; f < 3; ++f) lirs.SetEvictable(f, true);

    // it goes on exactly like a policy that never tried
    page_id_t page_in[3] = {10, 11, 12};
    for (page_id_t page = 20; page < 40; ++page) {
        // every third page is one from 4 loads back, often still in S by now
        page_id_t incoming = page % 3 == 0 ? page - 4 : page;
        if (std::find(std::begin(page_in), std::end(page_in), incoming) != std::end(page_in)) incoming = page;
        const auto victim = lirs.Evict(incoming);
        ASSERT_EQ(victim, fresh.Evict(incoming)) << "page " << page;
        ASSERT_TRUE(victim.has_value());
        page_in[*victim] = incoming;
        lirs.RecordAccess(*victim, incoming, AccessType::NEW_USE);
        fresh.RecordAccess(*victim, incoming, AccessType::NEW_USE);
        const frame_id_t other = (*victim + 1) % 3;
        const AccessType type = page % 2 == 0 ? AccessType::NEW_USE : AccessType::TOUCH;
        lirs.RecordAccess(other, page_in[other], type);
        fresh.RecordAccess(other, page_in[other], type);
        ASSERT_TRUE(SameSizes(lirs.Sizes(), fresh.Sizes())) << Describe(lirs.Sizes()) << " vs " << Describe(fresh.Sizes());
    }
}

TEST(LirsEvictionPolicyTest, RemoveForgetsFrame) {
    LirsEvictionPolicy lirs(3);
    lirs.RecordAccess(0, 10, AccessType::NEW_USE);
    lirs.RecordAccess(1, 11, AccessType::NEW_USE);
    lirs.RecordAccess(2, 12, AccessType::NEW_USE);  // HIR, in S and Q

    lirs.Remove(2);  // 12 leaves S too
    EXPECT_EQ(lirs.Size(), 2u);
    EXPECT_EQ(Describe(lirs.Sizes()), "LIR 2 HIR 0 S 2");

    // back in, 12 has no place in S, so it stays HIR (with one, it would become LIR and 10 would go)
    lirs.RecordAccess(2, 12, AccessType::NEW_USE);
    EXPECT_EQ(Describe(lirs.Sizes()), "LIR 2 HIR 1 S 3");
    EXPECT_EQ(lirs.Evict(), std::optional<frame_id_t>(2));  // 12 keeps its place in S

    // an LIR page removed leaves the LIR set, so the next load is LIR
    lirs.Remove(0);
    EXPECT_EQ(Describe(lirs.Sizes()), "LIR 1 HIR 0 S 2");
    lirs.RecordAccess(0, 13, AccessType::NEW_USE);
    EXPECT_EQ(Describe(lirs.Sizes()), "LIR 2 HIR 0 S 3");

    lirs.Remove(2);  // not tracked since its eviction: nothing happens
    EXPECT_EQ(lirs.Size(), 2u);
    EXPECT_EQ(Describe(lirs.Sizes()), "LIR 2 HIR 0 S 3");
}

TEST(LirsEvictionPolicyTest, SizeCountsEvictableOnly) {
    LirsEvictionPolicy lirs(3);
    lirs.RecordAccess(0, 10, AccessType::NEW_USE);
    lirs.RecordAccess(1, 11, AccessType::NEW_USE);
    EXPECT_EQ(lirs.Size(), 2u);  // both evictable once loaded

    lirs.SetEvictable(0, false);
    EXPECT_EQ(lirs.Size(), 1u);
    lirs.SetEvictable(0, false);  // twice changes nothing
    EXPECT_EQ(lirs.Size(), 1u);

    lirs.SetEvictable(0, true);
    EXPECT_EQ(lirs.Size(), 2u);
    lirs.RecordAccess(0, 10, AccessType::NEW_USE);  // a hit doesn't count twice
    EXPECT_EQ(lirs.Size(), 2u);
}

TEST(LirsEvictionPolicyTest, OutOfRangeFrameThrows) {
    LirsEvictionPolicy lirs(3);

    EXPECT_THROW(lirs.RecordAccess(3, 1, AccessType::NEW_USE), DatabaseException);
    EXPECT_THROW(lirs.RecordAccess(3, 1, AccessType::REPEAT), DatabaseException);
    EXPECT_THROW(lirs.RecordAccess(3, 1), DatabaseException);
    EXPECT_THROW(lirs.SetEvictable(3, true), DatabaseException);
    EXPECT_THROW(lirs.Remove(3), DatabaseException);

    EXPECT_EQ(lirs.Size(), 0u);
    EXPECT_FALSE(lirs.Evict().has_value());
    EXPECT_FALSE(lirs.Evict(5).has_value());
}

TEST(LirsEvictionPolicyTest, NeedsThePageId) {
    LirsEvictionPolicy lirs(3);
    EXPECT_THROW(lirs.RecordAccess(0), DatabaseException);
    EXPECT_THROW(lirs.RecordAccess(0, INVALID_PAGE_ID, AccessType::NEW_USE), DatabaseException);
    EXPECT_EQ(lirs.Size(), 0u);

    // the two-argument call is a new use
    lirs.RecordAccess(0, 10);
    EXPECT_EQ(lirs.Size(), 1u);
    EXPECT_TRUE(lirs.UsesAccessTypes());

    // a page lives in one frame only
    EXPECT_THROW(lirs.RecordAccess(1, 10, AccessType::NEW_USE), DatabaseException);
    EXPECT_EQ(lirs.Size(), 1u);
    EXPECT_EQ(Describe(lirs.Sizes()), "LIR 1 HIR 0 S 1");
}

TEST(LirsEvictionPolicyTest, EvictedFrameCanBeTrackedAgain) {
    LirsEvictionPolicy lirs(3);
    lirs.RecordAccess(0, 10, AccessType::NEW_USE);
    lirs.RecordAccess(1, 11, AccessType::NEW_USE);
    lirs.RecordAccess(2, 12, AccessType::NEW_USE);

    EXPECT_EQ(lirs.Evict(), std::optional<frame_id_t>(2));
    EXPECT_EQ(lirs.Size(), 2u);
    EXPECT_EQ(Describe(lirs.Sizes()), "LIR 2 HIR 0 S 3");  // 12 keeps its place in S

    // back while still in S: 12 becomes LIR, and 10, at S's bottom, becomes HIR
    lirs.RecordAccess(2, 12, AccessType::NEW_USE);
    EXPECT_EQ(lirs.Size(), 3u);
    EXPECT_EQ(Describe(lirs.Sizes()), "LIR 2 HIR 1 S 2");
    EXPECT_EQ(lirs.Evict(), std::optional<frame_id_t>(0));
    EXPECT_EQ(lirs.Evict(), std::optional<frame_id_t>(1));
    EXPECT_EQ(lirs.Evict(), std::optional<frame_id_t>(2));
    EXPECT_FALSE(lirs.Evict().has_value());
}

TEST(LirsEvictionPolicyTest, SetEvictableOnUntrackedFrameIgnored) {
    LirsEvictionPolicy lirs(3);

    lirs.SetEvictable(1, true);
    EXPECT_EQ(lirs.Size(), 0u);
    EXPECT_FALSE(lirs.Evict().has_value());

    lirs.RecordAccess(1, 10, AccessType::NEW_USE);
    EXPECT_EQ(lirs.Size(), 1u);
}

TEST(LirsEvictionPolicyTest, LoadIntoBusyFrameLetsOldPageGo) {
    // a load into a frame that still holds another page, with no Evict first
    LirsEvictionPolicy lirs(3);
    lirs.RecordAccess(0, 10, AccessType::NEW_USE);
    lirs.RecordAccess(1, 11, AccessType::NEW_USE);
    lirs.RecordAccess(2, 12, AccessType::NEW_USE);  // S [10 11 12], Q [12]

    // a HIR page leaves Q and keeps its place in S, as if evicted; 13 is HIR
    lirs.RecordAccess(2, 13, AccessType::NEW_USE);
    EXPECT_EQ(Describe(lirs.Sizes()), "LIR 2 HIR 1 S 4");
    EXPECT_EQ(lirs.Size(), 3u);

    // an LIR page leaves S and the LIR set, so 14 is LIR; the frame keeps its pin
    lirs.SetEvictable(0, false);
    lirs.RecordAccess(0, 14, AccessType::NEW_USE);
    EXPECT_EQ(Describe(lirs.Sizes()), "LIR 2 HIR 1 S 4");  // S [11 12 13 14]
    EXPECT_EQ(lirs.Size(), 2u);
    EXPECT_EQ(lirs.Evict(), std::optional<frame_id_t>(2));  // 13, from Q
}

TEST(LirsEvictionPolicyTest, ResetEqualsNewPolicy) {
    constexpr size_t kPool = 8;
    LirsEvictionPolicy lirs(kPool);
    std::mt19937 rng(7);
    {
        PolicyHarness harness(lirs, kPool);
        VisitTypes visits(64);
        for (int i = 0; i < 500; ++i) {
            const page_id_t page = rng() % 40;
            harness.Access(page, visits.Next(page, harness.Resident(page)));
        }
        harness.Hold(rng() % 40);  // leaves a pinned frame behind
    }

    lirs.Reset();
    EXPECT_EQ(lirs.Size(), 0u);
    EXPECT_FALSE(lirs.Evict().has_value());
    EXPECT_EQ(Describe(lirs.Sizes()), "LIR 0 HIR 0 S 0");

    // from here it acts exactly like a new policy: nothing left in S, Q or the LIR set
    LirsEvictionPolicy fresh(kPool);
    PolicyHarness a(lirs, kPool), b(fresh, kPool);
    VisitTypes va(64), vb(64);
    for (int i = 0; i < 20000; ++i) {
        const page_id_t page = rng() % 40;
        ASSERT_EQ(a.Access(page, va.Next(page, a.Resident(page))), b.Access(page, vb.Next(page, b.Resident(page)))) << "op " << i;
        ASSERT_EQ(a.LastVictim(), b.LastVictim()) << "op " << i;
        ASSERT_EQ(lirs.Size(), fresh.Size()) << "op " << i;
        ASSERT_TRUE(SameSizes(lirs.Sizes(), fresh.Sizes())) << "op " << i;
    }
}

TEST(LirsEvictionPolicyTest, LargePageIdGrows) {
    LirsEvictionPolicy lirs(2);
    PolicyHarness harness(lirs, 2);

    EXPECT_TRUE(harness.Access(1'000'000, AccessType::NEW_USE));   // LIR
    EXPECT_TRUE(harness.Access(5, AccessType::NEW_USE));           // HIR
    EXPECT_FALSE(harness.Access(1'000'000, AccessType::NEW_USE));

    // the big page is LIR, so the HIR pages go
    EXPECT_TRUE(harness.Access(6, AccessType::NEW_USE));
    EXPECT_EQ(harness.LastVictim(), std::optional<page_id_t>(5));
    EXPECT_TRUE(harness.Access(7, AccessType::NEW_USE));
    EXPECT_EQ(harness.LastVictim(), std::optional<page_id_t>(6));
    EXPECT_TRUE(harness.Resident(1'000'000));

    // a brand new page as the incoming one is ignored
    LirsEvictionPolicy other(2);
    other.RecordAccess(0, 3, AccessType::NEW_USE);
    EXPECT_EQ(other.Evict(INVALID_PAGE_ID), std::optional<frame_id_t>(0));
}

TEST(LirsEvictionPolicyTest, PoolOfOneThrows) {
    EXPECT_THROW({ LirsEvictionPolicy lirs(1); }, DatabaseException);
    EXPECT_THROW({ LirsEvictionPolicy lirs(0); }, DatabaseException);
    EXPECT_NO_THROW({ LirsEvictionPolicy lirs(2); });
}

TEST(LirsEvictionPolicyTest, PoolOfTwo) {
    // one LIR frame and one HIR
    LirsEvictionPolicy lirs(2);
    lirs.RecordAccess(0, 10, AccessType::NEW_USE);
    EXPECT_EQ(lirs.Evict(), std::optional<frame_id_t>(0));  // Q is empty: the LIR page goes
    EXPECT_FALSE(lirs.Evict().has_value());

    for (const std::optional<uint64_t> window : {std::optional<uint64_t>(), std::optional<uint64_t>(64)}) {
        LirsEvictionPolicy two(2);
        EXPECT_EQ(CountMisses(two, 2, {1, 1, 2, 2, 1, 3}, window), 3u);
    }

    // the HIR page goes, though 1 is older
    LirsEvictionPolicy two(2);
    PolicyHarness harness(two, 2);
    Feed(harness, {1, 1, 2, 3});
    EXPECT_EQ(harness.LastVictim(), std::optional<page_id_t>(2));
}

TEST(LirsEvictionPolicyTest, TeachingsExample) {
    // A B C D A B E A B C D E
    LirsEvictionPolicy lirs(3);
    EXPECT_EQ(CountMisses(lirs, 3, {0, 1, 2, 3, 0, 1, 4, 0, 1, 2, 3, 4}), 8u);
}

TEST(LirsEvictionPolicyTest, ScanAfterHotPair) {
    // The pair 1 2 twice, a scan of pages 10..19, then the pair again.
    std::vector<page_id_t> pages = {1, 2, 1, 2};
    for (page_id_t p = 10; p < 20; ++p) pages.push_back(p);
    pages.push_back(1);
    pages.push_back(2);
    LirsEvictionPolicy lirs(4);
    EXPECT_EQ(CountMisses(lirs, 4, pages), 12u);
}

TEST(LirsEvictionPolicyTest, KeepsALoop) {
    // Pages 0..10 five times over: a loop one page bigger than the pool.
    std::vector<page_id_t> pages;
    for (int loop = 0; loop < 5; ++loop) {
        for (page_id_t p = 0; p <= 10; ++p) pages.push_back(p);
    }
    LirsEvictionPolicy lirs(10);
    EXPECT_EQ(CountMisses(lirs, 10, pages), 19u);
    LRUEvictionPolicy lru(10);
    EXPECT_EQ(CountMisses(lru, 10, pages), 55u);
}

TEST(LirsEvictionPolicyTest, VisitModeSkipsQuickReturns) {
    // A scan page read twice close together (5 1 5). Fed plain, its second read makes it LIR
    // and turns the hot page 2 into the HIR page that 6 then evicts. Counted as one visit,
    // 5 stays HIR and 6 evicts it instead.
    const std::vector<page_id_t> pages = {1, 2, 5, 1, 5, 6, 1, 2};
    LirsEvictionPolicy visit(3);
    EXPECT_EQ(CountMisses(visit, 3, pages, 3), 4u);
    LirsEvictionPolicy plain(3);
    EXPECT_EQ(CountMisses(plain, 3, pages), 5u);
}

TEST(LirsEvictionPolicyTest, MatchesReferenceOnRandomOps) {
    // Under 150 frames Q holds a single page, so its order only shows at 150 and 250 (2 and 3 HIR frames)
    const auto result = test::MatchReferenceOnRandomOps<LirsEvictionPolicy, ReferenceLirs>(
        {2, 3, 4, 5, 9, 17, 30, 40, 150, 250}, CheckAgainstReference);
    EXPECT_TRUE(result.error.empty()) << result.error;
    EXPECT_GE(result.ops, 200000u);
}

}  // namespace shilmandb
