#include "executor/parallel/decimal.hpp"
#include "common/exception.hpp"
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

// The exact value of a %g text like "-0.055" or "1e-05", at its smallest scale.
std::optional<DecimalLiteral> ParseShortText(const char* text) {
    const char* p = text;
    const bool negative = *p == '-';
    if (negative) ++p;

    int64_t digits = 0;  // at most 17 significant digits, so it fits
    int digit_count = 0;
    int exponent = 0;    // the value is digits * 10^exponent
    bool after_point = false;
    for (; *p != '\0' && *p != 'e' && *p != 'E'; ++p) {
        if (*p >= '0' && *p <= '9') {
            if (digits != 0 || *p != '0') {
                digits = digits * 10 + (*p - '0');
                ++digit_count;
            }
            if (after_point) --exponent;
        } else {
            after_point = true;  // '.', since the engine never sets a locale
        }
    }
    if (*p != '\0') exponent += std::atoi(p + 1);
    assert(digits != 0 && "the caller handles zero");

    while (digits % 10 == 0) {
        digits /= 10;
        --digit_count;
        ++exponent;
    }
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

// Tries 1 to 17 significant digits; the first text strtod reads back as v wins.
std::optional<DecimalLiteral> DecimalFromDouble(double v) {
    if (!std::isfinite(v)) return std::nullopt;
    if (v == 0.0) return DecimalLiteral{};  // -0.0 as well

    char text[kTextBytes];
    for (int precision = 1; precision <= kMaxRoundTripDigits; ++precision) {
        std::snprintf(text, sizeof(text), "%.*g", precision, v);
        if (std::strtod(text, nullptr) == v) break;
    }
    return ParseShortText(text);
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
