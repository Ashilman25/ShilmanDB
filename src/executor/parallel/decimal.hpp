#pragma once
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include "types/types.hpp"

namespace shilmandb::parallel {

// The most digits a NUMERIC holds, so the widest exact decimal here.
inline constexpr uint8_t kMaxDecimalDigits = 38;

// Shared with the planner; see types/types.hpp.
using shilmandb::kExactDoubleDigits;

// An exact decimal: value / 10^scale.
struct DecimalLiteral {
    __int128 value{0};
    uint8_t scale{0};
};

namespace detail {

// 10^0 to 10^38 as exact integers.
struct Pow10Table {
    __int128 v[kMaxDecimalDigits + 1];
};

// Builds the table at compile time.
constexpr Pow10Table MakePow10Table() {
    Pow10Table t{};
    t.v[0] = 1;
    for (int i = 1; i <= kMaxDecimalDigits; ++i) t.v[i] = t.v[i - 1] * 10;
    return t;
}

inline constexpr Pow10Table kPow10Table = MakePow10Table();

// 10^38 - 1, the largest magnitude a decimal may reach.
inline constexpr __int128 kMaxDecimal = kPow10Table.v[kMaxDecimalDigits] - 1;

// True while |v| has at most 38 digits.
constexpr bool FitsDecimal(__int128 v) {
    return v <= kMaxDecimal && v >= -kMaxDecimal;
}

// Throws DatabaseException("NUMERIC result is too wide"); kept out of line.
[[noreturn]] void ThrowTooWide();

// True when a * b passes __int128; out gets the product mod 2^128, like the builtin.
// The 128-bit builtin needs compiler-rt, which libgcc builds can't link.
constexpr bool MulOverflow128(__int128 a, __int128 b, __int128* out) {
    using U128 = unsigned __int128;
    *out = static_cast<__int128>(static_cast<U128>(a) * static_cast<U128>(b));
    constexpr __int128 lo = std::numeric_limits<int64_t>::min();
    constexpr __int128 hi = std::numeric_limits<int64_t>::max();
    if (a >= lo && a <= hi && b >= lo && b <= hi) return false;  // |a * b| <= 2^126
    const U128 ua = a < 0 ? U128{0} - static_cast<U128>(a) : static_cast<U128>(a);
    const U128 ub = b < 0 ? U128{0} - static_cast<U128>(b) : static_cast<U128>(b);
    // A negative product may reach 2^127, a positive one 2^127 - 1.
    const U128 limit = (U128{1} << 127) - ((a < 0) == (b < 0) ? 1 : 0);
    return ub != 0 && ua > limit / ub;
}

}  // namespace detail

// The exact value of the shortest text that round-trips v (tries %.1g to %.<max_digits>g).
// nullopt for NaN, infinity, no such text, a scale over 38 or more than 38 digits.
[[nodiscard]] std::optional<DecimalLiteral> DecimalFromDouble(double v, int max_digits = 17);

// The exact value of a text like "-123.45", "0.055" or "1e-05", at its smallest scale.
// nullopt for other text, more than 38 significant digits or a scale over 38.
[[nodiscard]] std::optional<DecimalLiteral> DecimalFromText(std::string_view text);

// 10^n as an exact integer, for n <= 38.
[[nodiscard]] inline __int128 Pow10(uint8_t n) {
    assert(n <= kMaxDecimalDigits && "Pow10 past 38 digits");
    return detail::kPow10Table.v[n];
}

// a + b; throws when the sum passes 38 digits.
[[nodiscard]] inline __int128 CheckedAdd(__int128 a, __int128 b) {
    __int128 sum;
    if (__builtin_add_overflow(a, b, &sum) || !detail::FitsDecimal(sum)) detail::ThrowTooWide();
    return sum;
}

// a * b; throws when the product passes 38 digits.
[[nodiscard]] inline __int128 CheckedMul(__int128 a, __int128 b) {
    __int128 product;
    if (detail::MulOverflow128(a, b, &product) || !detail::FitsDecimal(product)) detail::ThrowTooWide();
    return product;
}

// v moved from scale `from` up to scale `to`; throws past 38 digits.
[[nodiscard]] inline __int128 Rescale(__int128 v, uint8_t from, uint8_t to) {
    assert(from <= to && to <= kMaxDecimalDigits && "Rescale only raises the scale");
    if (from == to) return v;
    return CheckedMul(v, Pow10(static_cast<uint8_t>(to - from)));
}

// a * b into out when it fits in 64 bits; out means nothing otherwise.
[[nodiscard]] inline bool MulFits64(int64_t a, int64_t b, int64_t* out) {
    return !__builtin_mul_overflow(a, b, out);
}

// The double nearest v / 10^scale. With |v| <= 2^53 and scale <= 22,
// one exact division, bit-identical to the columnar scan; else strtod of the text.
[[nodiscard]] double DecimalToDouble(__int128 v, uint8_t scale);

// Exact text like "-123.45": all `scale` decimals kept, no exponent.
[[nodiscard]] std::string DecimalToString(__int128 v, uint8_t scale);

}  // namespace shilmandb::parallel
