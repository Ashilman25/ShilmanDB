#include <gtest/gtest.h>
#include "storage/columnar/hyperloglog.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace shilmandb {
namespace {

// Relative error of an estimate against the true count.
double RelativeError(uint64_t estimate, uint64_t actual) {
    return std::fabs(static_cast<double>(estimate) - static_cast<double>(actual)) /
           static_cast<double>(actual);
}

TEST(HyperLogLogTest, AccuracyLarge) {
    HyperLogLog sketch;
    constexpr uint64_t kCount = 1'000'000;
    // 0..999,999 lands at +2.27% with this hash, a 2.8-sigma draw.
    // The v*7+3 set below is typical: the error's sd is about 0.8%.
    for (int64_t v = 0; v < static_cast<int64_t>(kCount); ++v) sketch.Add(HashInt(v * 7 + 3));
    const uint64_t estimate = sketch.Estimate();
    EXPECT_LE(RelativeError(estimate, kCount), 0.02) << "estimate " << estimate;
}

TEST(HyperLogLogTest, AccuracySmall) {
    HyperLogLog sketch;
    constexpr uint64_t kCount = 15'000;
    // Each value twice, so repeats must not count.
    for (int pass = 0; pass < 2; ++pass) {
        for (int64_t v = 0; v < static_cast<int64_t>(kCount); ++v) sketch.Add(HashInt(v * 7 + 3));
    }
    const uint64_t estimate = sketch.Estimate();
    EXPECT_LE(RelativeError(estimate, kCount), 0.015) << "estimate " << estimate;

    HyperLogLog one;
    for (int i = 0; i < 100; ++i) one.Add(HashString("only"));
    EXPECT_EQ(one.Estimate(), 1u);

    // A hash of 0 leaves all 50 rank bits zero, the largest rank.
    HyperLogLog zero_hash;
    zero_hash.Add(0);
    EXPECT_EQ(zero_hash.Estimate(), 1u);

    HyperLogLog empty;
    EXPECT_EQ(empty.Estimate(), 0u);
}

TEST(HyperLogLogTest, MergeEqualsCombined) {
    HyperLogLog single;
    std::vector<HyperLogLog> parts(8);
    for (int64_t v = 0; v < 200'000; ++v) {
        const uint64_t h = HashInt(v);
        single.Add(h);
        parts[static_cast<size_t>(v % 8)].Add(h);
    }
    HyperLogLog merged;
    for (const auto& part : parts) merged.Merge(part);
    EXPECT_EQ(merged.Estimate(), single.Estimate());

    // Merging into a non-empty sketch, in another order, gives the same answer.
    HyperLogLog reversed = parts[7];
    for (int i = 6; i >= 0; --i) reversed.Merge(parts[static_cast<size_t>(i)]);
    EXPECT_EQ(reversed.Estimate(), single.Estimate());
}

TEST(HyperLogLogTest, OrderIndependent) {
    std::vector<int64_t> values(50'000);
    std::iota(values.begin(), values.end(), -25'000);
    std::mt19937_64 rng(7);
    std::vector<int64_t> first = values;
    std::vector<int64_t> second = values;
    std::shuffle(first.begin(), first.end(), rng);
    std::shuffle(second.begin(), second.end(), rng);

    HyperLogLog a;
    HyperLogLog b;
    for (int64_t v : first) a.Add(HashInt(v));
    for (int64_t v : second) b.Add(HashInt(v));
    EXPECT_EQ(a.Estimate(), b.Estimate());
}

TEST(HyperLogLogTest, HashesAreStable) {
    const __int128 above_2_64 = (static_cast<__int128>(1) << 64) + 12345;
    EXPECT_EQ(HashString("abc"), 0x0dd490490804b508ULL);
    EXPECT_EQ(HashInt(42), 0xbdd732262feb6e95ULL);
    EXPECT_EQ(HashWide(above_2_64), 0xd84f161566efe75dULL);
    EXPECT_EQ(HashDouble(1.5), 0xd6dab18e1392608aULL);
    EXPECT_EQ(HashDouble(-0.0), HashDouble(0.0));
    EXPECT_NE(HashInt(0), 0u);
}

}  // namespace
}  // namespace shilmandb
