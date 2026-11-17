#include "storage/columnar/crc32c.hpp"
#include <array>
#include <cstring>

#if defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#elif defined(__SSE4_2__)
#include <nmmintrin.h>
#endif

namespace shilmandb {
namespace {

constexpr uint32_t kPolynomial = 0x82F63B78u;  // Castagnoli, reflected

// One entry per byte value, built at compile time.
constexpr std::array<uint32_t, 256> MakeTable() {
    std::array<uint32_t, 256> table{};
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t crc = i;
        for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ ((crc & 1) ? kPolynomial : 0);
        table[i] = crc;
    }
    return table;
}

constexpr std::array<uint32_t, 256> kTable = MakeTable();

}  // namespace

// A byte at a time through the table.
uint32_t Crc32cPortable(const void* data, size_t len, uint32_t seed) {
    const auto* p = static_cast<const unsigned char*>(data);
    uint32_t crc = ~seed;
    for (size_t i = 0; i < len; ++i) crc = kTable[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

// Eight bytes per instruction where the CPU has one, then the tail by bytes.
uint32_t Crc32c(const void* data, size_t len, uint32_t seed) {
#if defined(__ARM_FEATURE_CRC32)
    const auto* p = static_cast<const unsigned char*>(data);
    uint32_t crc = ~seed;
    for (; len >= 8; p += 8, len -= 8) {
        uint64_t word;
        std::memcpy(&word, p, sizeof(word));
        crc = __crc32cd(crc, word);
    }
    for (; len > 0; ++p, --len) crc = __crc32cb(crc, *p);
    return ~crc;
    
#elif defined(__SSE4_2__)
    const auto* p = static_cast<const unsigned char*>(data);
    uint64_t crc = ~seed;
    for (; len >= 8; p += 8, len -= 8) {
        uint64_t word;
        std::memcpy(&word, p, sizeof(word));
        crc = _mm_crc32_u64(crc, word);
    }
    auto crc32 = static_cast<uint32_t>(crc);
    for (; len > 0; ++p, --len) crc32 = _mm_crc32_u8(crc32, *p);
    return ~crc32;
#else
    return Crc32cPortable(data, len, seed);
#endif
}

}  // namespace shilmandb
