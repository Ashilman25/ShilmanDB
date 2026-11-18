#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>

namespace shilmandb::parallel {

namespace detail {

// Added before mixing each key, so a zero key still moves the hash.
inline constexpr uint64_t kKeySeed = 0x9e3779b97f4a7c15ULL;

// Spreads 64 bits over the whole word (murmur3's finalizer); a bijection.
[[nodiscard]] constexpr uint64_t HashMix(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

// Folds one key's bits into a row's hash, which starts at 0; key order matters.
[[nodiscard]] constexpr uint64_t FoldKey(uint64_t h, uint64_t bits) {
    return HashMix(h ^ (bits + kKeySeed));
}

// A 128-bit value's two halves in one word.
[[nodiscard]] constexpr uint64_t WideBits(__int128 v) {
    const auto u = static_cast<unsigned __int128>(v);
    return static_cast<uint64_t>(u) ^ HashMix(static_cast<uint64_t>(u >> 64));
}

// A string's bytes, eight at a time.
[[nodiscard]] inline uint64_t HashBytes(std::string_view s) {
    uint64_t h = kKeySeed ^ s.size();
    size_t i = 0;
    for (; i + 8 <= s.size(); i += 8) {
        uint64_t word = 0;
        std::memcpy(&word, s.data() + i, 8);
        h = HashMix(h ^ word);
    }
    uint64_t tail = 0;
    if (i < s.size()) std::memcpy(&tail, s.data() + i, s.size() - i);
    return HashMix(h ^ tail);
}

// A double by value: -0.0 as +0.0 and every NaN alike.
[[nodiscard]] inline uint64_t DoubleBits(double d) {
    if (d == 0.0) d = 0.0;
    if (std::isnan(d)) d = std::numeric_limits<double>::quiet_NaN();
    uint64_t bits = 0;
    std::memcpy(&bits, &d, sizeof(bits));
    return bits;
}

}  // namespace detail

}  // namespace shilmandb::parallel
