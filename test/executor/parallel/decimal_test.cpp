#include <gtest/gtest.h>
#include "common/exception.hpp"
#include "executor/parallel/decimal.hpp"
#include "storage/columnar/column_vector.hpp"
#include "storage/columnar/field_parser.hpp"
#include "types/column_type.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace shilmandb::parallel {

namespace {

constexpr int64_t k2Pow53 = int64_t{1} << 53;
constexpr int64_t k10Pow15 = 1'000'000'000'000'000;

// 10^n built by hand, so the tests don't lean on Pow10.
__int128 TenTo(int n) {
    __int128 v = 1;
    for (int i = 0; i < n; ++i) v *= 10;
    return v;
}

// 10^38 - 1, the largest magnitude a NUMERIC holds.
__int128 MaxNumeric() {
    return TenTo(38) - 1;
}

// Plain integer text for failure messages.
std::string Str(__int128 v) {
    if (v == 0) return "0";
    const bool negative = v < 0;
    unsigned __int128 m = negative ? -static_cast<unsigned __int128>(v) : static_cast<unsigned __int128>(v);
    std::string digits;
    while (m != 0) {
        digits.insert(digits.begin(), static_cast<char>('0' + static_cast<int>(m % 10)));
        m /= 10;
    }
    return negative ? "-" + digits : digits;
}

// The exact decimal text of v / 10^scale, written without the code under test.
std::string ExactText(__int128 v, int scale) {
    std::string digits = Str(v < 0 ? -v : v);
    if (scale > 0) {
        if (digits.size() <= static_cast<size_t>(scale)) {
            digits.insert(0, static_cast<size_t>(scale) + 1 - digits.size(), '0');
        }
        digits.insert(digits.size() - static_cast<size_t>(scale), ".");
    }
    return v < 0 ? "-" + digits : digits;
}

// A double's bit pattern, so -0.0 and 0.0 differ.
uint64_t Bits(double d) {
    uint64_t b;
    std::memcpy(&b, &d, sizeof(b));
    return b;
}

// Checks a literal's value and scale.
void ExpectLiteral(double v, __int128 value, uint8_t scale) {
    const std::optional<DecimalLiteral> got = DecimalFromDouble(v);
    ASSERT_TRUE(got.has_value()) << "no literal for " << v;
    EXPECT_TRUE(got->value == value) << "value " << Str(got->value) << ", expected " << Str(value);
    EXPECT_EQ(static_cast<int>(got->scale), static_cast<int>(scale));
}

// Runs fn and checks it throws the too-wide error.
template <typename Fn>
void ExpectTooWide(Fn fn) {
    try {
        (void)fn();
        ADD_FAILURE() << "expected a too-wide error";
    } catch (const DatabaseException& e) {
        EXPECT_STREQ(e.what(), "NUMERIC result is too wide");
    }
}

// A random value of 1 to 16 digits with |n| <= 2^53, sign included.
int64_t RandomNarrow(std::mt19937_64& rng) {
    std::uniform_int_distribution<int> digit_count(1, 16);
    const int d = digit_count(rng);
    int64_t limit = 1;
    for (int i = 0; i < d; ++i) limit *= 10;
    limit = std::min(limit - 1, k2Pow53);
    std::uniform_int_distribution<int64_t> magnitude(0, limit);
    const int64_t n = magnitude(rng);
    return (rng() & 1) ? -n : n;
}

}  // namespace

TEST(ParallelDecimalTest, LiteralFromShortestText) {
    ExpectLiteral(0.05, 5, 2);
    ExpectLiteral(100.00, 100, 0);
    ExpectLiteral(-0.055, -55, 3);
    ExpectLiteral(1e-5, 1, 5);
    EXPECT_FALSE(DecimalFromDouble(std::nan("")).has_value());

    ExpectLiteral(0.0, 0, 0);
    ExpectLiteral(-0.0, 0, 0);
    ExpectLiteral(0.1, 1, 1);
    ExpectLiteral(0.07, 7, 2);
    ExpectLiteral(24.0, 24, 0);
    ExpectLiteral(1e22, TenTo(22), 0);
    ExpectLiteral(1e-38, 1, 38);
    ExpectLiteral(9.9e37, 99 * TenTo(36), 0);

    // Shortest round trip, not a fixed precision: 15 digits would give 0.3.
    ExpectLiteral(0.1 + 0.2, 30000000000000004, 17);

    EXPECT_FALSE(DecimalFromDouble(std::numeric_limits<double>::infinity()).has_value());
    EXPECT_FALSE(DecimalFromDouble(-std::numeric_limits<double>::infinity()).has_value());
    EXPECT_FALSE(DecimalFromDouble(1e-39).has_value());   // scale 39
    EXPECT_FALSE(DecimalFromDouble(1.5e-38).has_value()); // scale 39
    EXPECT_FALSE(DecimalFromDouble(1e38).has_value());    // 39 digits
    EXPECT_FALSE(DecimalFromDouble(1e300).has_value());
}

TEST(ParallelDecimalTest, LiteralRoundTripsTextOfFifteenDigits) {
    // Any decimal of at most 15 digits comes back exactly, at its smallest scale.
    std::mt19937_64 rng(20261117);
    std::uniform_int_distribution<int64_t> mantissa(-(k10Pow15 - 1), k10Pow15 - 1);
    std::uniform_int_distribution<int> scale_pick(0, 20);
    for (int i = 0; i < 20000; ++i) {
        int64_t n = mantissa(rng);
        int scale = scale_pick(rng);
        const double d = std::strtod(ExactText(n, scale).c_str(), nullptr);
        while (n != 0 && n % 10 == 0 && scale > 0) {
            n /= 10;
            --scale;
        }
        if (n == 0) scale = 0;
        ExpectLiteral(d, n, static_cast<uint8_t>(scale));
        if (::testing::Test::HasFailure()) break;
    }
}

TEST(ParallelDecimalTest, Pow10Table) {
    EXPECT_TRUE(Pow10(0) == 1);
    EXPECT_TRUE(Pow10(2) == 100);
    EXPECT_TRUE(Pow10(18) == int64_t{1'000'000'000'000'000'000});
    for (int n = 0; n <= kMaxDecimalDigits; ++n) {
        EXPECT_TRUE(Pow10(static_cast<uint8_t>(n)) == TenTo(n)) << "10^" << n;
    }
}

TEST(ParallelDecimalTest, RescaleAndOverflow) {
    EXPECT_TRUE(Rescale(123, 2, 4) == 12300);
    EXPECT_TRUE(Rescale(123, 2, 2) == 123);
    EXPECT_TRUE(Rescale(9, 0, 37) == 9 * TenTo(37));
    EXPECT_TRUE(Rescale(TenTo(35), 0, 2) == TenTo(37));
    ExpectTooWide([] { return Rescale(TenTo(37), 0, 2); });
    ExpectTooWide([] { return Rescale(TenTo(36), 0, 2); });  // exactly 10^38
    ExpectTooWide([] { return Rescale(1, 0, 38); });
    ExpectTooWide([] { return Rescale(-TenTo(37), 1, 3); });
}

TEST(ParallelDecimalTest, CheckedAddPast38Digits) {
    ExpectTooWide([] { return CheckedAdd(MaxNumeric(), 1); });
    ExpectTooWide([] { return CheckedAdd(-MaxNumeric(), -1); });
    ExpectTooWide([] { return CheckedAdd(MaxNumeric(), MaxNumeric()); });
    ExpectTooWide([] { return CheckedAdd(std::numeric_limits<__int128>::max(), 1); });
    ExpectTooWide([] { return CheckedAdd(std::numeric_limits<__int128>::min(), -1); });
    ExpectTooWide([] { return CheckedAdd(std::numeric_limits<__int128>::min(), 0); });
    EXPECT_TRUE(CheckedAdd(MaxNumeric(), 0) == MaxNumeric());
    EXPECT_TRUE(CheckedAdd(MaxNumeric(), -MaxNumeric()) == 0);
    EXPECT_TRUE(CheckedAdd(-5, 3) == -2);
}

TEST(ParallelDecimalTest, CheckedMulPast38Digits) {
    EXPECT_TRUE(CheckedMul(TenTo(19), TenTo(18)) == TenTo(37));
    EXPECT_TRUE(CheckedMul(-TenTo(19), TenTo(18)) == -TenTo(37));
    EXPECT_TRUE(CheckedMul(-7, -6) == 42);
    ExpectTooWide([] { return CheckedMul(TenTo(19), TenTo(19)); });   // 10^38
    ExpectTooWide([] { return CheckedMul(TenTo(20), TenTo(20)); });   // past 128 bits
    ExpectTooWide([] { return CheckedMul(-TenTo(20), TenTo(20)); });
    ExpectTooWide([] { return CheckedMul(std::numeric_limits<__int128>::min(), -1); });
    ExpectTooWide([] { return CheckedMul(std::numeric_limits<__int128>::min(), 1); });
}

TEST(ParallelDecimalTest, MulFits64) {
    int64_t out = 0;
    EXPECT_TRUE(MulFits64(10494950, 100, &out));
    EXPECT_EQ(out, 1049495000);
    EXPECT_FALSE(MulFits64(std::numeric_limits<int64_t>::max(), 2, &out));
    EXPECT_FALSE(MulFits64(std::numeric_limits<int64_t>::min(), -1, &out));
    EXPECT_TRUE(MulFits64(-3, 4, &out));
    EXPECT_EQ(out, -12);
}

TEST(ParallelDecimalTest, ToDoubleMatchesScanForNarrow) {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int64_t> money(-(k10Pow15 - 1), k10Pow15 - 1);
    for (int i = 0; i < 100000; ++i) {
        const int64_t n = money(rng);
        const double expected = static_cast<double>(n) / 100.0;
        ASSERT_EQ(Bits(DecimalToDouble(n, 2)), Bits(expected)) << "n = " << n;
    }

    // The real scan conversion up to scale 22, narrow and wide columns.
    std::vector<int64_t> values = {0, 1, -1, 7, -7, k2Pow53, -k2Pow53, k2Pow53 - 1, -(k2Pow53 - 1)};
    for (int i = 0; i < 2000; ++i) values.push_back(RandomNarrow(rng));

    for (int scale = 0; scale <= 22; ++scale) {
        std::vector<ColumnType> types = {ColumnType::Numeric(38, scale)};
        if (scale <= 18) types.push_back(ColumnType::Numeric(18, scale));
        for (const ColumnType& type : types) {
            ColumnVector col(type);
            for (const int64_t n : values) {
                if (col.Kind() == PhysicalKind::INT64) {
                    col.AppendInt(n);
                } else {
                    col.AppendWide(n);
                }
            }
            for (size_t i = 0; i < values.size(); ++i) {
                const double scan = ToExecutorValue(col, i).decimal_;
                const double ours = DecimalToDouble(values[i], static_cast<uint8_t>(scale));
                ASSERT_EQ(Bits(ours), Bits(scan))
                    << "n = " << values[i] << ", " << ColumnTypeName(type);
            }
        }
    }

    // Past scale 22 the nearest double, from the exact text.
    for (int scale = 23; scale <= kMaxDecimalDigits; ++scale) {
        const auto s = static_cast<uint8_t>(scale);
        for (const int64_t n : values) {
            ASSERT_EQ(DecimalToString(n, s), ExactText(n, scale));
            const double nearest = std::strtod(DecimalToString(n, s).c_str(), nullptr);
            ASSERT_EQ(Bits(DecimalToDouble(n, s)), Bits(nearest)) << "n = " << n << ", scale " << scale;
        }
    }

    // Two divisions would give 1.0000000000000001e-23, one ulp off.
    EXPECT_EQ(Bits(DecimalToDouble(1, 23)), Bits(1e-23));
}

TEST(ParallelDecimalTest, ToDoubleNearestForWide) {
    std::mt19937_64 rng(7);
    std::uniform_int_distribution<int> digit(0, 9);
    std::uniform_int_distribution<int> lead(1, 9);
    std::uniform_int_distribution<int> scale_pick(0, kMaxDecimalDigits);
    for (int i = 0; i < 1000; ++i) {
        __int128 v = lead(rng);
        for (int d = 1; d < 30; ++d) v = v * 10 + digit(rng);
        if (rng() & 1) v = -v;
        const auto scale = static_cast<uint8_t>(scale_pick(rng));

        const std::string text = ExactText(v, scale);
        ASSERT_EQ(DecimalToString(v, scale), text);
        const double nearest = std::strtod(text.c_str(), nullptr);
        ASSERT_EQ(Bits(DecimalToDouble(v, scale)), Bits(nearest)) << text;
    }

    // Just past 2^53 the nearest double differs from double(v) / 10.
    // 2^53 + 3 rounds to 2^53 + 4 as a double, but v / 10 ends in .5 exactly.
    EXPECT_EQ(DecimalToDouble(k2Pow53 + 3, 1), 900719925474099.5);
    EXPECT_NE(static_cast<double>(k2Pow53 + 3) / 10.0, 900719925474099.5);
    EXPECT_EQ(DecimalToDouble(MaxNumeric(), 0), 1e38);
    EXPECT_EQ(DecimalToDouble(-MaxNumeric(), 38), -1.0);
}

TEST(ParallelDecimalTest, NegativeAndMixedScales) {
    EXPECT_EQ(DecimalToString(-12345, 2), "-123.45");
    EXPECT_TRUE(Rescale(-5, 2, 4) == -500);

    EXPECT_EQ(DecimalToString(12345, 2), "123.45");
    EXPECT_EQ(DecimalToString(5, 3), "0.005");
    EXPECT_EQ(DecimalToString(-5, 3), "-0.005");
    EXPECT_EQ(DecimalToString(0, 2), "0.00");
    EXPECT_EQ(DecimalToString(42, 0), "42");
    EXPECT_EQ(DecimalToString(-MaxNumeric(), 38), "-0." + std::string(38, '9'));
    EXPECT_EQ(DecimalToString(MaxNumeric(), 0), std::string(38, '9'));

    // NUMERIC(15,2) plus NUMERIC(12,4): 123.45 + 0.0067 = 123.4567.
    EXPECT_EQ(DecimalToString(CheckedAdd(Rescale(12345, 2, 4), 67), 4), "123.4567");
    EXPECT_TRUE(Rescale(-12345, 2, 4) == -1234500);

    // A literal with more decimals than the column: 0.06 >= 0.055, 0.05 < 0.055.
    const std::optional<DecimalLiteral> lit = DecimalFromDouble(0.055);
    ASSERT_TRUE(lit.has_value());
    EXPECT_TRUE(Rescale(6, 2, lit->scale) >= lit->value);
    EXPECT_TRUE(Rescale(5, 2, lit->scale) < lit->value);

    // A sum past INT64_MAX stays exact in 128 bits.
    const int64_t big = std::numeric_limits<int64_t>::max();
    const __int128 sum = CheckedAdd(big, big);
    EXPECT_EQ(DecimalToString(sum, 2), "184467440737095516.14");
    int64_t out = 0;
    EXPECT_FALSE(MulFits64(big, 10, &out));
    EXPECT_TRUE(CheckedMul(big, 10) == static_cast<__int128>(big) * 10);
}

}  // namespace shilmandb::parallel
