#include <gtest/gtest.h>
#include <list>
#include <optional>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include "buffer/lru_eviction_policy.hpp"
#include "common/exception.hpp"

namespace shilmandb {

namespace {

// The LRU policy as it was written with a hash map and a hash set.
// Used to check the faster version picks the same victims.
class ReferenceLRU {
public:
    void RecordAccess(frame_id_t frame_id) {
        if (auto it = map_.find(frame_id); it != map_.end()) {
            list_.erase(it->second);
            list_.push_back(frame_id);
            it->second = std::prev(list_.end());
            return;
        }
        list_.push_back(frame_id);
        map_[frame_id] = std::prev(list_.end());
        ++evictable_count_;
    }

    void SetEvictable(frame_id_t frame_id, bool evictable) {
        if (map_.find(frame_id) == map_.end()) { return; }

        bool was_non_evictable = non_evictable_.count(frame_id) > 0;
        if (evictable && was_non_evictable) {
            non_evictable_.erase(frame_id);
            ++evictable_count_;
        } else if (!evictable && !was_non_evictable) {
            non_evictable_.insert(frame_id);
            --evictable_count_;
        }
    }

    std::optional<frame_id_t> Evict() {
        for (auto it = list_.begin(); it != list_.end(); ++it) {
            if (non_evictable_.count(*it) == 0) {
                frame_id_t victim = *it;
                map_.erase(victim);
                list_.erase(it);
                --evictable_count_;
                return victim;
            }
        }
        return std::nullopt;
    }

    void Remove(frame_id_t frame_id) {
        if (auto it = map_.find(frame_id); it != map_.end()) {
            if (non_evictable_.count(frame_id) == 0) { --evictable_count_; }
            list_.erase(it->second);
            map_.erase(it);
        }
        non_evictable_.erase(frame_id);
    }

    size_t Size() const { return evictable_count_; }

private:
    std::list<frame_id_t> list_;
    std::unordered_map<frame_id_t, std::list<frame_id_t>::iterator> map_;
    std::unordered_set<frame_id_t> non_evictable_;
    size_t evictable_count_{0};
};

}  // namespace

// ---------------------------------------------------------------------------
// Test 1: EvictLeastRecentlyUsed
// ---------------------------------------------------------------------------
TEST(LRUEvictionPolicyTest, EvictLeastRecentlyUsed) {
    LRUEvictionPolicy lru(3);

    lru.RecordAccess(0);
    lru.RecordAccess(1);
    lru.RecordAccess(2);

    lru.SetEvictable(0, true);
    lru.SetEvictable(1, true);
    lru.SetEvictable(2, true);

    // Evict in LRU order: 0 → 1 → 2
    auto victim = lru.Evict();
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(*victim, 0u);

    victim = lru.Evict();
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(*victim, 1u);

    victim = lru.Evict();
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(*victim, 2u);
}

// ---------------------------------------------------------------------------
// Test 2: AccessUpdatesOrder
// ---------------------------------------------------------------------------
TEST(LRUEvictionPolicyTest, AccessUpdatesOrder) {
    LRUEvictionPolicy lru(3);

    lru.RecordAccess(0);
    lru.RecordAccess(1);
    lru.RecordAccess(0);  // 0 moves to MRU — order is now 1, 0

    lru.SetEvictable(0, true);
    lru.SetEvictable(1, true);

    auto victim = lru.Evict();
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(*victim, 1u);  // 1 is now the oldest
}

// ---------------------------------------------------------------------------
// Test 3: NonEvictableSkipped
// ---------------------------------------------------------------------------
TEST(LRUEvictionPolicyTest, NonEvictableSkipped) {
    LRUEvictionPolicy lru(3);

    lru.RecordAccess(0);
    lru.RecordAccess(1);

    lru.SetEvictable(0, false);
    lru.SetEvictable(1, true);

    auto victim = lru.Evict();
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(*victim, 1u);  // 0 is pinned, skip to 1
}

// ---------------------------------------------------------------------------
// Test 4: NoEvictableReturnsNullopt
// ---------------------------------------------------------------------------
TEST(LRUEvictionPolicyTest, NoEvictableReturnsNullopt) {
    LRUEvictionPolicy lru(3);

    lru.RecordAccess(0);
    lru.SetEvictable(0, false);

    auto victim = lru.Evict();
    EXPECT_FALSE(victim.has_value());
}

// ---------------------------------------------------------------------------
// Test 5: RemoveDeletesFrame
// ---------------------------------------------------------------------------
TEST(LRUEvictionPolicyTest, RemoveDeletesFrame) {
    LRUEvictionPolicy lru(3);

    lru.RecordAccess(0);
    lru.Remove(0);

    EXPECT_EQ(lru.Size(), 0u);

    // Evict should return nothing
    auto victim = lru.Evict();
    EXPECT_FALSE(victim.has_value());
}

// ---------------------------------------------------------------------------
// Test 6: SizeReflectsEvictableOnly
// ---------------------------------------------------------------------------
TEST(LRUEvictionPolicyTest, SizeReflectsEvictableOnly) {
    LRUEvictionPolicy lru(3);

    lru.RecordAccess(0);
    lru.RecordAccess(1);
    EXPECT_EQ(lru.Size(), 2u);  // both evictable by default

    lru.SetEvictable(0, false);
    EXPECT_EQ(lru.Size(), 1u);  // only frame 1 is evictable

    lru.SetEvictable(0, true);
    EXPECT_EQ(lru.Size(), 2u);  // frame 0 restored to evictable
}

// ---------------------------------------------------------------------------
// Test 7: MatchesReferenceOnRandomOps
// ---------------------------------------------------------------------------
TEST(LRUEvictionPolicyTest, MatchesReferenceOnRandomOps) {
    constexpr size_t kFrames = 8;
    constexpr int kOps = 200000;

    LRUEvictionPolicy lru(kFrames);
    ReferenceLRU ref;
    std::mt19937 rng(42);

    for (int i = 0; i < kOps; ++i) {
        frame_id_t frame = rng() % kFrames;
        switch (rng() % 10) {
            case 0: case 1: case 2: case 3:
                lru.RecordAccess(frame);
                ref.RecordAccess(frame);
                break;
            case 4: case 5:
                lru.SetEvictable(frame, true);
                ref.SetEvictable(frame, true);
                break;
            case 6: case 7:
                lru.SetEvictable(frame, false);
                ref.SetEvictable(frame, false);
                break;
            case 8:
                ASSERT_EQ(lru.Evict(), ref.Evict()) << "op " << i;
                break;
            default:
                lru.Remove(frame);
                ref.Remove(frame);
                break;
        }
        ASSERT_EQ(lru.Size(), ref.Size()) << "op " << i;
    }
}

// ---------------------------------------------------------------------------
// Test 8: PinnedFrameKeepsItsPlace
// ---------------------------------------------------------------------------
TEST(LRUEvictionPolicyTest, PinnedFrameKeepsItsPlace) {
    LRUEvictionPolicy lru(3);

    lru.RecordAccess(0);
    lru.RecordAccess(1);
    lru.RecordAccess(2);

    lru.SetEvictable(0, false);
    EXPECT_EQ(*lru.Evict(), 1u);

    // Frame 0 was accessed first, so it is still the oldest once it is evictable again
    lru.SetEvictable(0, true);
    EXPECT_EQ(*lru.Evict(), 0u);
}

// ---------------------------------------------------------------------------
// Test 9: EvictedFrameCanBeTrackedAgain
// ---------------------------------------------------------------------------
TEST(LRUEvictionPolicyTest, EvictedFrameCanBeTrackedAgain) {
    LRUEvictionPolicy lru(3);

    lru.RecordAccess(0);
    EXPECT_EQ(*lru.Evict(), 0u);
    EXPECT_EQ(lru.Size(), 0u);

    lru.RecordAccess(0);
    EXPECT_EQ(lru.Size(), 1u);
    EXPECT_EQ(*lru.Evict(), 0u);
    EXPECT_FALSE(lru.Evict().has_value());
}

// ---------------------------------------------------------------------------
// Test 10: SetEvictableOnUntrackedFrameIgnored
// ---------------------------------------------------------------------------
TEST(LRUEvictionPolicyTest, SetEvictableOnUntrackedFrameIgnored) {
    LRUEvictionPolicy lru(3);

    lru.SetEvictable(1, true);
    EXPECT_EQ(lru.Size(), 0u);
    EXPECT_FALSE(lru.Evict().has_value());

    lru.RecordAccess(1);
    EXPECT_EQ(lru.Size(), 1u);
}

// ---------------------------------------------------------------------------
// Test 11: OutOfRangeFrameThrows
// ---------------------------------------------------------------------------
TEST(LRUEvictionPolicyTest, OutOfRangeFrameThrows) {
    LRUEvictionPolicy lru(3);

    EXPECT_THROW(lru.RecordAccess(3), DatabaseException);
    EXPECT_THROW(lru.SetEvictable(3, true), DatabaseException);
    EXPECT_THROW(lru.Remove(3), DatabaseException);

    EXPECT_EQ(lru.Size(), 0u);
    EXPECT_FALSE(lru.Evict().has_value());
}

}  // namespace shilmandb
