#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "buffer/arc_eviction_policy.hpp"
#include "buffer/clock_eviction_policy.hpp"
#include "buffer/eviction_policy_factory.hpp"
#include "buffer/lirs_eviction_policy.hpp"
#include "buffer/lru2_eviction_policy.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "common/exception.hpp"

namespace shilmandb {
namespace {

const std::vector<std::pair<EvictionPolicyKind, std::string_view>>& AllKinds() {
    static const std::vector<std::pair<EvictionPolicyKind, std::string_view>> kinds = {
        {EvictionPolicyKind::LRU, "lru"},
        {EvictionPolicyKind::CLOCK, "clock"},
        {EvictionPolicyKind::LRU2, "lru2"},
        {EvictionPolicyKind::ARC, "arc"},
        {EvictionPolicyKind::LIRS, "lirs"},
        {EvictionPolicyKind::LEARNED_V1, "learned-v1"},
        {EvictionPolicyKind::LEARNED_V2, "learned-v2"},
    };
    return kinds;
}

}  // namespace

TEST(EvictionPolicyFactoryTest, NamesRoundTrip) {
    for (const auto& [kind, name] : AllKinds()) {
        EXPECT_EQ(EvictionPolicyKindName(kind), name);
        const auto parsed = ParseEvictionPolicyKind(name);
        ASSERT_TRUE(parsed.has_value()) << name;
        EXPECT_EQ(*parsed, kind) << name;
    }
}

TEST(EvictionPolicyFactoryTest, UnknownNameIsNullopt) {
    EXPECT_FALSE(ParseEvictionPolicyKind("LRU").has_value());
    EXPECT_FALSE(ParseEvictionPolicyKind("arc_visit").has_value());
    EXPECT_FALSE(ParseEvictionPolicyKind("").has_value());
}

TEST(EvictionPolicyFactoryTest, MakesEachClassicalKind) {
    constexpr size_t kPool = 16;

    const auto lru = MakeEvictionPolicy(EvictionPolicyKind::LRU, kPool);
    EXPECT_NE(dynamic_cast<LRUEvictionPolicy*>(lru.get()), nullptr);
    EXPECT_FALSE(lru->UsesAccessTypes());

    const auto clock = MakeEvictionPolicy(EvictionPolicyKind::CLOCK, kPool);
    EXPECT_NE(dynamic_cast<ClockEvictionPolicy*>(clock.get()), nullptr);
    EXPECT_TRUE(clock->UsesAccessTypes());

    const auto lru2 = MakeEvictionPolicy(EvictionPolicyKind::LRU2, kPool);
    EXPECT_NE(dynamic_cast<Lru2EvictionPolicy*>(lru2.get()), nullptr);
    EXPECT_TRUE(lru2->UsesAccessTypes());

    const auto arc = MakeEvictionPolicy(EvictionPolicyKind::ARC, kPool);
    EXPECT_NE(dynamic_cast<ArcEvictionPolicy*>(arc.get()), nullptr);
    EXPECT_TRUE(arc->UsesAccessTypes());

    const auto lirs = MakeEvictionPolicy(EvictionPolicyKind::LIRS, kPool);
    EXPECT_NE(dynamic_cast<LirsEvictionPolicy*>(lirs.get()), nullptr);
    EXPECT_TRUE(lirs->UsesAccessTypes());

    // the pool size reaches the policy: frame 15 is in range, 16 is not
    EXPECT_NO_THROW(lru->RecordAccess(15, 0));
    EXPECT_THROW(lru->RecordAccess(16, 0), DatabaseException);
}

TEST(EvictionPolicyFactoryTest, LirsPoolOfOneThrows) {
    EXPECT_THROW((void)MakeEvictionPolicy(EvictionPolicyKind::LIRS, 1), DatabaseException);
}

TEST(EvictionPolicyFactoryTest, LearnedNeedsModelPath) {
    EXPECT_THROW((void)MakeEvictionPolicy(EvictionPolicyKind::LEARNED_V1, 16), DatabaseException);
    EXPECT_THROW((void)MakeEvictionPolicy(EvictionPolicyKind::LEARNED_V2, 16, ""), DatabaseException);
}

#ifndef SHILMANDB_HAS_LIBTORCH
TEST(EvictionPolicyFactoryTest, LearnedNeedsMlBuild) {
    EXPECT_THROW((void)MakeEvictionPolicy(EvictionPolicyKind::LEARNED_V1, 16, "model.pt"), DatabaseException);
    EXPECT_THROW((void)MakeEvictionPolicy(EvictionPolicyKind::LEARNED_V2, 16, "model.pt"), DatabaseException);
}
#else
TEST(EvictionPolicyFactoryTest, LearnedBadModelPathThrows) {
    EXPECT_THROW((void)MakeEvictionPolicy(EvictionPolicyKind::LEARNED_V1, 16, "no_such_model.pt"), DatabaseException);
    EXPECT_THROW((void)MakeEvictionPolicy(EvictionPolicyKind::LEARNED_V2, 16, "no_such_model.pt"), DatabaseException);
}
#endif

}  // namespace shilmandb
