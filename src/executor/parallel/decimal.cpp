#include "executor/parallel/decimal.hpp"
#include "common/exception.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace shilmandb::parallel {

namespace {

// Every power of ten up to 1e22 is an exact double.
constexpr int kMaxExactPow10 = 22;
constexpr double kExactPow10[kMaxExactPow10 + 1] = {
    1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,
    1e8,  1e9,  1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
    1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22
};

// Every integer up to 2^53 converts to a double exactly.
constexpr __int128 kMaxExactInteger = __int128{1} << 53;

// Room for %.17g of any double: sign, 17 digits, point, exponent.
constexpr size_t kTextBytes = 32;

// The most digits a double's round-trip text needs.
constexpr int kMaxRoundTripDigits = 17;

// A larger exponent never gives a decimal of 38 digits; the cap stops overflow.
constexpr int kMaxExponent = 1000;

// The exact value of a text like "-0.055", "120.50" or "1e-05", at its smallest scale.
// Trailing zeros are held back, so they never overflow the 38 digits kept.
std::optional<DecimalLiteral> ParseDecimalText(std::string_view text) {
    size_t i = 0;
    const bool negative = i < text.size() && text[i] == '-';
    if (negative) ++i;

    __int128 digits = 0;  // the significant digits so far, without trailing zeros
    int digit_count = 0;
    int zeros = 0;        // zeros after the last non-zero digit
    int exponent = 0;     // the value is digits * 10^(exponent + zeros)
    bool any_digit = false;
    bool after_point = false;
    for (; i < text.size() && text[i] != 'e' && text[i] != 'E'; ++i) {
        const char c = text[i];
        if (c == '.') {
            if (after_point) return std::nullopt;
            after_point = true;
            continue;
        }
        if (c < '0' || c > '9') return std::nullopt;
        any_digit = true;
        if (after_point) --exponent;
        if (c == '0') {
            if (digits != 0) ++zeros;
            continue;
        }
        if (digit_count + zeros + 1 > kMaxDecimalDigits) return std::nullopt;
        digits = digits * Pow10(static_cast<uint8_t>(zeros + 1)) + (c - '0');
        digit_count += zeros + 1;
        zeros = 0;
    }
    if (!any_digit) return std::nullopt;

    // An exponent: e, maybe a sign, then digits.
    if (i < text.size()) {
        ++i;
        bool exponent_negative = false;
        if (i < text.size() && (text[i] == '+' || text[i] == '-')) exponent_negative = text[i++] == '-';
        if (i == text.size()) return std::nullopt;
        int e = 0;
        for (; i < text.size(); ++i) {
            if (text[i] < '0' || text[i] > '9' || e > kMaxExponent) return std::nullopt;
            e = e * 10 + (text[i] - '0');
        }
        exponent += exponent_negative ? -e : e;
    }
    if (digits == 0) return DecimalLiteral{};

    exponent += zeros;
    if (-exponent > kMaxDecimalDigits) return std::nullopt;
    if (exponent > 0 && digit_count + exponent > kMaxDecimalDigits) return std::nullopt;

    DecimalLiteral lit;
    lit.value = exponent > 0 ? digits * Pow10(static_cast<uint8_t>(exponent)) : digits;
    if (negative) lit.value = -lit.value;
    lit.scale = static_cast<uint8_t>(exponent < 0 ? -exponent : 0);
    return lit;
}

}  // namespace

namespace detail {

// One shared throw, so the inline checks stay small.
void ThrowTooWide() {
    throw DatabaseException("NUMERIC result is too wide");
}

}  // namespace detail

// Tries 1 to max_digits significant digits; the first text strtod reads back as v wins.
std::optional<DecimalLiteral> DecimalFromDouble(double v, int max_digits) {
    if (!std::isfinite(v)) return std::nullopt;
    if (v == 0.0) return DecimalLiteral{};  // -0.0 as well

    char text[kTextBytes];
    for (int precision = 1; precision <= std::min(max_digits, kMaxRoundTripDigits); ++precision) {
        std::snprintf(text, sizeof(text), "%.*g", precision, v);
        if (std::strtod(text, nullptr) == v) return ParseDecimalText(text);
    }
    return std::nullopt;
}

// Written text, so it holds digits a double would round away.
std::optional<DecimalLiteral> DecimalFromText(std::string_view text) {
    return ParseDecimalText(text);
}

// One exact division while |v| <= 2^53 and scale <= 22, else strtod of the text.
// Both give the nearest double.
double DecimalToDouble(__int128 v, uint8_t scale) {
    assert(scale <= kMaxDecimalDigits && "scale past 38 digits");
    if (scale <= kMaxExactPow10 && v >= -kMaxExactInteger && v <= kMaxExactInteger) {
        return static_cast<double>(v) / kExactPow10[scale];
    }
    return std::strtod(DecimalToString(v, scale).c_str(), nullptr);
}

// Writes digits from the right, putting the point after `scale` of them.
std::string DecimalToString(__int128 v, uint8_t scale) {
    char buf[260];  // 256 digits, a point, a sign: any uint8_t scale fits
    char* const end = buf + sizeof(buf);
    char* p = end;

    unsigned __int128 m = v < 0 ? -static_cast<unsigned __int128>(v) : static_cast<unsigned __int128>(v);
    int written = 0;
    do {
        *--p = static_cast<char>('0' + static_cast<int>(m % 10));
        m /= 10;
        if (++written == scale) *--p = '.';
    } while (m != 0 || written <= scale);

    if (v < 0) *--p = '-';
    return std::string(p, end);
}

}  // namespace shilmandb::parallel
