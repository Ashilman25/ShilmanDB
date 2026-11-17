#include <gtest/gtest.h>
#include "storage/columnar/crc32c.hpp"
#include <cstdint>
#include <random>
#include <vector>

namespace shilmandb {

TEST(Crc32cTest, Crc32cKnownVector) {
    EXPECT_EQ(Crc32c("123456789", 9), 0xE3069283u);
    EXPECT_EQ(Crc32cPortable("123456789", 9), 0xE3069283u);
    EXPECT_EQ(Crc32c("", 0), 0u);
}

// The table version must agree with whatever path this machine runs.
TEST(Crc32cTest, PortableMatchesCrc32c) {
    std::mt19937_64 rng(7);
    std::vector<unsigned char> bytes(4200);
    for (auto& b : bytes) b = static_cast<unsigned char>(rng());

    for (size_t len = 0; len <= 300; ++len) {
        for (size_t offset = 0; offset < 8; ++offset) {
            const auto seed = static_cast<uint32_t>(rng());
            ASSERT_EQ(Crc32c(bytes.data() + offset, len, seed),
                      Crc32cPortable(bytes.data() + offset, len, seed))
                << "len " << len << " offset " << offset;
        }
    }
    EXPECT_EQ(Crc32c(bytes.data() + 3, 4188, 99), Crc32cPortable(bytes.data() + 3, 4188, 99));
}

TEST(Crc32cTest, SeedChainsAcrossPieces) {
    std::mt19937_64 rng(11);
    std::vector<unsigned char> bytes(1000);
    for (auto& b : bytes) b = static_cast<unsigned char>(rng());

    const uint32_t whole = Crc32c(bytes.data(), bytes.size());
    for (size_t cut : {0u, 1u, 7u, 8u, 9u, 500u, 999u, 1000u}) {
        const uint32_t first = Crc32c(bytes.data(), cut);
        EXPECT_EQ(Crc32c(bytes.data() + cut, bytes.size() - cut, first), whole) << "cut " << cut;
        const uint32_t first_portable = Crc32cPortable(bytes.data(), cut);
        EXPECT_EQ(Crc32cPortable(bytes.data() + cut, bytes.size() - cut, first_portable), whole);
    }
}

}  // namespace shilmandb
