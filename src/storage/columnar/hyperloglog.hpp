#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace shilmandb {

// A distinct-count sketch. Sketches built on separate parts merge into one.
class HyperLogLog {
public:
    static constexpr int kPrecision = 14;  // 16,384 registers
    static constexpr size_t kRegisters = size_t{1} << kPrecision;

    // Records one value's 64-bit hash.
    void Add(uint64_t hash);

    // Folds another sketch in, register by register (the larger wins).
    void Merge(const HyperLogLog& other);

    // Estimated distinct count; linear counting while that is the better estimate.
    [[nodiscard]] uint64_t Estimate() const;

private:
    std::array<uint8_t, kRegisters> registers_{};
};

// One splitmix64 step over the value's bits.
[[nodiscard]] uint64_t HashInt(int64_t v);

// Hashes a 128-bit value through both 64-bit halves.
// Differs from HashInt on 64-bit values, so one column never mixes the two.
[[nodiscard]] uint64_t HashWide(__int128 v);

// Hashes a double's bits; +0.0 and -0.0 hash the same.
[[nodiscard]] uint64_t HashDouble(double v);

// FNV-1a 64 over the bytes, then the splitmix64 finalizer.
[[nodiscard]] uint64_t HashString(std::string_view s);

}  // namespace shilmandb
