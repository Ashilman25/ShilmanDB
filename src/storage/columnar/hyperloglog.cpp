#include "storage/columnar/hyperloglog.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace shilmandb {
namespace {

// The splitmix64 finalizer: spreads every input bit over the whole word.
constexpr uint64_t Mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

constexpr uint64_t SplitMix64(uint64_t x) { return Mix64(x + 0x9e3779b97f4a7c15ULL); }

}  // namespace

void HyperLogLog::Add(uint64_t hash) {
    // The top bits pick the register; the rest give the rank (leading zeros + 1).
    const size_t index = static_cast<size_t>(hash >> (64 - kPrecision));
    const uint64_t rest = hash << kPrecision;
    const auto rank = static_cast<uint8_t>(rest == 0 ? 64 - kPrecision + 1 : __builtin_clzll(rest) + 1);
    registers_[index] = std::max(registers_[index], rank);
}

void HyperLogLog::Merge(const HyperLogLog& other) {
    std::transform(registers_.begin(), registers_.end(), other.registers_.begin(), registers_.begin(), [](uint8_t a, uint8_t b) { return std::max(a, b); });
}

uint64_t HyperLogLog::Estimate() const {
    constexpr double m = static_cast<double>(kRegisters);
    constexpr double alpha = 0.7213 / (1.0 + 1.079 / m);

    double sum = 0.0;
    size_t zeros = 0;
    for (uint8_t reg : registers_) {
        sum += std::ldexp(1.0, -reg);
        zeros += reg == 0;
    }
    double estimate = alpha * m * m / sum;
    if (estimate <= 2.5 * m && zeros > 0) {
        estimate = m * std::log(m / static_cast<double>(zeros));
    }
    return static_cast<uint64_t>(std::llround(estimate));
}

uint64_t HashInt(int64_t v) { return SplitMix64(static_cast<uint64_t>(v)); }

// I hash the low half, xor in the high half, then hash again.
uint64_t HashWide(__int128 v) {
    const auto bits = static_cast<unsigned __int128>(v);
    const auto low = static_cast<uint64_t>(bits);
    const auto high = static_cast<uint64_t>(bits >> 64);
    return SplitMix64(SplitMix64(low) ^ high);
}

uint64_t HashDouble(double v) {
    if (v == 0.0) v = 0.0;  // folds -0.0 into +0.0
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return SplitMix64(bits);
}

uint64_t HashString(std::string_view s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (char c : s) {
        h ^= static_cast<uint8_t>(c);
        h *= 0x100000001b3ULL;
    }
    return Mix64(h);
}

}  // namespace shilmandb
