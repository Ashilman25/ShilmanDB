#include <gtest/gtest.h>
#include <chrono>
#include <optional>
#include <random>
#include <vector>
#include "buffer/lru_eviction_policy.hpp"
#include "buffer/timed_eviction_policy.hpp"

namespace shilmandb {

namespace {

// Remembers every call it gets, so a test can see what was forwarded.
class FakePolicy : public EvictionPolicy {
public:
    void RecordAccess(frame_id_t frame_id) override { one_arg_record.push_back(frame_id); }
    void RecordAccess(frame_id_t frame_id, page_id_t page_id) override {
        two_arg_record.emplace_back(frame_id, page_id);
    }
    void RecordAccess(frame_id_t frame_id, page_id_t page_id, AccessType type) override {
        typed_record.push_back({frame_id, page_id, type});
    }
    void SetEvictable(frame_id_t frame_id, bool evictable) override {
        set_evictable.emplace_back(frame_id, evictable);
    }
    std::optional<frame_id_t> Evict() override {
        ++evict_calls;
        return 5;
    }
    std::optional<frame_id_t> Evict(page_id_t incoming_page) override {
        evict_pages.push_back(incoming_page);
        return 6;
    }
    void Remove(frame_id_t frame_id) override { removed.push_back(frame_id); }
    size_t Size() const override { return 3; }
    bool UsesAccessTypes() const override { return true; }
    void Reset() override { ++reset_calls; }

    struct TypedRecord {
        frame_id_t frame;
        page_id_t page;
        AccessType type;
    };

    std::vector<frame_id_t> one_arg_record;
    std::vector<std::pair<frame_id_t, page_id_t>> two_arg_record;
    std::vector<TypedRecord> typed_record;
    std::vector<std::pair<frame_id_t, bool>> set_evictable;
    std::vector<frame_id_t> removed;
    std::vector<page_id_t> evict_pages;
    int evict_calls{0};
    int reset_calls{0};
};

}  // namespace

TEST(TimedEvictionPolicyTest, ForwardsEveryCall) {
    FakePolicy fake;
    TimedEvictionPolicy timed(&fake);

    timed.RecordAccess(1);
    timed.RecordAccess(2, 42);
    timed.SetEvictable(1, true);
    auto victim = timed.Evict();
    timed.Remove(2);
    size_t size = timed.Size();

    EXPECT_EQ(fake.one_arg_record, std::vector<frame_id_t>{1});
    ASSERT_EQ(fake.two_arg_record.size(), 1u);
    EXPECT_EQ(fake.two_arg_record[0].first, 2u);
    EXPECT_EQ(fake.two_arg_record[0].second, 42u);  // the page id came through
    ASSERT_EQ(fake.set_evictable.size(), 1u);
    EXPECT_EQ(fake.set_evictable[0].first, 1u);
    EXPECT_TRUE(fake.set_evictable[0].second);
    EXPECT_EQ(fake.evict_calls, 1);
    EXPECT_EQ(fake.removed, std::vector<frame_id_t>{2});
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(*victim, 5u);
    EXPECT_EQ(size, 3u);

    const PolicyStats& stats = timed.GetStats();
    EXPECT_EQ(stats.record_calls, 2u);
    EXPECT_EQ(stats.set_evictable_calls, 1u);
    EXPECT_EQ(stats.evict_calls, 1u);
    EXPECT_EQ(stats.remove_calls, 1u);
    EXPECT_EQ(stats.TotalCalls(), 5u);
}

TEST(TimedEvictionPolicyTest, ForwardsNewCalls) {
    FakePolicy fake;
    TimedEvictionPolicy timed(&fake);

    timed.RecordAccess(3, 77, AccessType::TOUCH);
    auto victim = timed.Evict(7);
    const bool uses_types = timed.UsesAccessTypes();
    timed.Reset();

    ASSERT_EQ(fake.typed_record.size(), 1u);
    EXPECT_EQ(fake.typed_record[0].frame, 3u);
    EXPECT_EQ(fake.typed_record[0].page, 77u);
    EXPECT_EQ(fake.typed_record[0].type, AccessType::TOUCH);
    EXPECT_TRUE(fake.two_arg_record.empty());  // not turned into the old call

    EXPECT_EQ(fake.evict_pages, std::vector<page_id_t>{7});
    EXPECT_EQ(fake.evict_calls, 0);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(*victim, 6u);

    EXPECT_TRUE(uses_types);  // the default would say false
    EXPECT_EQ(fake.reset_calls, 1);

    // the typed call counts as a record call and Evict(page) as an evict call;
    // Reset and UsesAccessTypes are not counted
    const PolicyStats& stats = timed.GetStats();
    EXPECT_EQ(stats.record_calls, 1u);
    EXPECT_EQ(stats.evict_calls, 1u);
    EXPECT_EQ(stats.TotalCalls(), 2u);
}

TEST(TimedEvictionPolicyTest, MatchesWrappedLRU) {
    constexpr size_t kFrames = 8;
    constexpr int kOps = 20000;

    LRUEvictionPolicy inner(kFrames);
    TimedEvictionPolicy timed(&inner);
    LRUEvictionPolicy bare(kFrames);
    std::mt19937 rng(7);

    for (int i = 0; i < kOps; ++i) {
        frame_id_t frame = rng() % kFrames;
        switch (rng() % 10) {
            case 0: case 1: case 2: case 3:
                timed.RecordAccess(frame);
                bare.RecordAccess(frame);
                break;
            case 4: case 5:
                timed.SetEvictable(frame, true);
                bare.SetEvictable(frame, true);
                break;
            case 6: case 7:
                timed.SetEvictable(frame, false);
                bare.SetEvictable(frame, false);
                break;
            case 8:
                ASSERT_EQ(timed.Evict(), bare.Evict()) << "op " << i;
                break;
            default:
                timed.Remove(frame);
                bare.Remove(frame);
                break;
        }
        ASSERT_EQ(timed.Size(), bare.Size()) << "op " << i;
    }
}

TEST(TimedEvictionPolicyTest, NetTimeNeverNegative) {
    LRUEvictionPolicy inner(4);
    TimedEvictionPolicy timed(&inner);
    for (int i = 0; i < 10000; ++i) {
        timed.SetEvictable(static_cast<frame_id_t>(i % 4), i % 2 == 0);
    }

    const PolicyStats& stats = timed.GetStats();
    EXPECT_GE(stats.NetTime(), std::chrono::nanoseconds{0});
    EXPECT_LE(stats.NetTime(), stats.RawTime());
    EXPECT_GT(stats.timer_cost, std::chrono::nanoseconds{0});
}

}  // namespace shilmandb
