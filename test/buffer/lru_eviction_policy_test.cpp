#include <gtest/gtest.h>
#include "buffer/lru_eviction_policy.hpp"

namespace shilmandb {

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

}  // namespace shilmandb
