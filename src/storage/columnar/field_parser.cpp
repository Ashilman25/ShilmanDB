#include "storage/columnar/field_parser.hpp"
#include "common/exception.hpp"
#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace shilmandb {

namespace {

using Reason = std::optional<std::string>;

constexpr int64_t kMicrosPerSecond = 1'000'000;
constexpr int64_t kMicrosPerDay = 86'400 * kMicrosPerSecond;
constexpr int64_t kDaysFrom1970To2000 = 10'957;
constexpr size_t kMaxShownBytes = 64;
constexpr const char* kNumericOverflow = "numeric field overflow";

// Every power of ten up to 1e22 is an exact double.
constexpr int kMaxExactPow10 = 22;
constexpr double kPow10[kMaxExactPow10 + 1] = {
    1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,
    1e8,  1e9,  1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
    1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22
};

// 10^0 to 10^38 as exact integers, the NUMERIC magnitude limits.
struct WidePow10Table {
    unsigned __int128 v[39];
};

// Builds the table at compile time.
constexpr WidePow10Table MakeWidePow10() {
    WidePow10Table t{};
    t.v[0] = 1;
    for (int i = 1; i <= 38; ++i) t.v[i] = t.v[i - 1] * 10;
    return t;
}

constexpr WidePow10Table kWidePow10 = MakeWidePow10();

// ASCII digits only; isdigit would depend on the locale.
constexpr bool IsDigit(char c) {
    return c >= '0' && c <= '9';
}

// The field as an error shows it, cut short when long.
std::string Shown(std::string_view text) {
    if (text.size() <= kMaxShownBytes) return std::string(text);
    return std::string(text.substr(0, kMaxShownBytes)) + "...";
}

// "invalid <TYPE>: <field>".
std::string Invalid(const ColumnType& type, std::string_view text) {
    return "invalid " + ColumnTypeName(type) + ": " + Shown(text);
}

// "out of range for <TYPE>: <field>".
std::string OutOfRange(const ColumnType& type, std::string_view text) {
    return "out of range for " + ColumnTypeName(type) + ": " + Shown(text);
}

enum class IntStatus { OK, BAD, OUT_OF_RANGE };

// Digits with an optional sign. from_chars refuses a leading '+', so I skip it.
IntStatus ParseInt64(std::string_view text, int64_t& out) {
    if (!text.empty() && text.front() == '+') {
        text.remove_prefix(1);
        if (text.empty() || !IsDigit(text.front())) return IntStatus::BAD;
    }

    const char* end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(text.data(), end, out);
    if (ptr != end) return IntStatus::BAD;
    if (ec == std::errc::result_out_of_range) return IntStatus::OUT_OF_RANGE;
    return ec == std::errc() ? IntStatus::OK : IntStatus::BAD;
}

// An integer that must land in [lo, hi].
Reason ParseInteger(std::string_view text, ColumnVector& col, int64_t lo, int64_t hi) {
    int64_t v = 0;
    switch (ParseInt64(text, v)) {
        case IntStatus::BAD:          return Invalid(col.Type(), text);
        case IntStatus::OUT_OF_RANGE: return OutOfRange(col.Type(), text);
        case IntStatus::OK:           break;
    }
    if (v < lo || v > hi) return OutOfRange(col.Type(), text);
    col.AppendInt(v);
    return std::nullopt;
}

// Postgres's spellings in any case, stored as 1 or 0.
Reason ParseBoolean(std::string_view text, ColumnVector& col) {
    constexpr std::string_view kTrueWords[] = {"t", "true", "1", "yes", "on"};
    constexpr std::string_view kFalseWords[] = {"f", "false", "0", "no", "off"};
    constexpr size_t kLongest = 5;

    if (text.empty() || text.size() > kLongest) return Invalid(col.Type(), text);
    char lower[kLongest];
    std::transform(text.begin(), text.end(), lower, [](char c) {
        return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
    });
    const std::string_view word(lower, text.size());
    auto matches = [word](std::string_view w) { return w == word; };

    if (std::any_of(std::begin(kTrueWords), std::end(kTrueWords), matches)) {
        col.AppendInt(1);
    } else if (std::any_of(std::begin(kFalseWords), std::end(kFalseWords), matches)) {
        col.AppendInt(0);
    } else {
        return Invalid(col.Type(), text);
    }
    return std::nullopt;
}

struct NumericParts {
    bool negative{false};
    std::string_view int_digits;   // leading zeros dropped
    std::string_view frac_digits;
};

// Splits "[sign]digits[.digits]"; false unless the whole field fits and has a digit.
bool SplitNumeric(std::string_view text, NumericParts& parts) {
    size_t i = 0;
    if (i < text.size() && (text[i] == '+' || text[i] == '-')) {
        parts.negative = text[i] == '-';
        ++i;
    }
    const size_t int_start = i;
    while (i < text.size() && IsDigit(text[i])) ++i;
    std::string_view int_digits = text.substr(int_start, i - int_start);

    std::string_view frac_digits;
    if (i < text.size() && text[i] == '.') {
        const size_t frac_start = ++i;
        while (i < text.size() && IsDigit(text[i])) ++i;
        frac_digits = text.substr(frac_start, i - frac_start);
    }
    if (i != text.size() || (int_digits.empty() && frac_digits.empty())) return false;

    while (!int_digits.empty() && int_digits.front() == '0') int_digits.remove_prefix(1);
    parts.int_digits = int_digits;
    parts.frac_digits = frac_digits;
    return true;
}

// The value times 10^scale, rounded half away from zero on the first dropped digit.
// The caller caps the digits gathered at the precision, so U never overflows.
template <typename U>
U ScaledMagnitude(const NumericParts& parts, size_t scale) {
    U m = 0;
    for (const char c : parts.int_digits) m = m * 10 + static_cast<U>(c - '0');

    const std::string_view frac = parts.frac_digits;
    for (size_t k = 0; k < scale; ++k) {
        m = m * 10 + static_cast<U>(k < frac.size() ? frac[k] - '0' : 0);
    }
    if (frac.size() > scale && frac[scale] >= '5') ++m;
    return m;
}

// Exact decimal parse into value x 10^scale, never through a double.
Reason ParseNumeric(std::string_view text, ColumnVector& col) {
    const ColumnType& type = col.Type();
    NumericParts parts;
    if (!SplitNumeric(text, parts)) return Invalid(type, text);

    // Rounding only grows a value, so too many integer digits already overflow.
    // This also keeps the digits gathered at precision or fewer (at most 38).
    const size_t scale = type.scale;
    if (parts.int_digits.size() > static_cast<size_t>(type.precision) - scale) return kNumericOverflow;

    // After rounding, the magnitude must stay below 10^precision.
    const unsigned __int128 limit = kWidePow10.v[type.precision];
    if (col.Kind() == PhysicalKind::INT64) {
        const auto m = ScaledMagnitude<uint64_t>(parts, scale);  // at most 10^18
        if (m >= static_cast<uint64_t>(limit)) return kNumericOverflow;
        const auto v = static_cast<int64_t>(m);
        col.AppendInt(parts.negative ? -v : v);

    } else {
        const auto m = ScaledMagnitude<unsigned __int128>(parts, scale);  // at most 10^38
        if (m >= limit) return kNumericOverflow;
        const auto v = static_cast<__int128>(m);
        col.AppendWide(parts.negative ? -v : v);
    }
    return std::nullopt;
}

// strtod exactly as std::stod calls it, so the double has the same bits.
Reason ParseDouble(std::string_view text, ColumnVector& col) {
    if (text.empty()) return Invalid(col.Type(), text);

    // strtod needs a terminated string; a short field is copied to the stack.
    constexpr size_t kStackBytes = 128;
    char stack_buf[kStackBytes];
    std::string heap_buf;
    const char* start = stack_buf;
    if (text.size() < kStackBytes) {
        std::memcpy(stack_buf, text.data(), text.size());
        stack_buf[text.size()] = '\0';

    } else {
        heap_buf.assign(text);
        start = heap_buf.c_str();
    }

    char* end = nullptr;
    const int saved_errno = errno;
    errno = 0;
    const double v = std::strtod(start, &end);
    const bool out_of_range = errno == ERANGE;
    errno = saved_errno;

    if (end == start || end != start + text.size()) return Invalid(col.Type(), text);
    if (out_of_range) return OutOfRange(col.Type(), text);
    col.AppendDouble(v);
    return std::nullopt;
}

// Gregorian leap years.
constexpr bool IsLeapYear(int64_t y) {
    return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
}

// Days in month m (1 to 12) of year y.
constexpr int DaysInMonth(int64_t y, int m) {
    constexpr int kDays[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    return (m == 2 && IsLeapYear(y)) ? 29 : kDays[m];
}

// Days since 1970-01-01, proleptic Gregorian, in O(1) (Howard Hinnant's days_from_civil).
constexpr int64_t DaysFromCivil(int64_t y, int64_t m, int64_t d) {
    y -= m <= 2 ? 1 : 0;
    const int64_t era = (y >= 0 ? y : y - 399) / 400;
    const int64_t yoe = y - era * 400;                                    // [0, 399]
    const int64_t doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;  // [0, 365]
    const int64_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;            // [0, 146096]
    return era * 146097 + doe - 719468;
}

struct CivilDate {
    int64_t year;
    int month;
    int day;
};

// The inverse of DaysFromCivil (Hinnant's civil_from_days).
constexpr CivilDate CivilFromDays(int64_t z) {
    z += 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const int64_t doe = z - era * 146097;                                        // [0, 146096]
    
    const int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;  // [0, 399]
    const int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);                 // [0, 365]
    const int64_t mp = (5 * doy + 2) / 153;                                      // [0, 11]

    const int day = static_cast<int>(doy - (153 * mp + 2) / 5 + 1);
    const int month = static_cast<int>(mp < 10 ? mp + 3 : mp - 9);
    return {
        yoe + era * 400 + (month <= 2 ? 1 : 0), 
        month, 
        day
    };
}

static_assert(DaysFromCivil(1970, 1, 1) == 0);
static_assert(DaysFromCivil(1969, 12, 31) == -1);
static_assert(DaysFromCivil(2000, 1, 1) == kDaysFrom1970To2000);
static_assert(CivilFromDays(-1).year == 1969 && CivilFromDays(-1).month == 12 && CivilFromDays(-1).day == 31);

// The n digits at text[pos], as a number; false if any isn't a digit.
bool ReadDigits(std::string_view text, size_t pos, size_t n, int& out) {
    int v = 0;
    for (size_t k = pos; k < pos + n; ++k) {
        if (!IsDigit(text[k])) return false;
        v = v * 10 + (text[k] - '0');
    }
    out = v;
    return true;
}

// Reads "YYYY-MM-DD" from the first 10 bytes; false unless it's a real date.
bool ReadDate(std::string_view text, int64_t& days) {
    int y = 0;
    int m = 0;
    int d = 0;
    if (text.size() < 10 || text[4] != '-' || text[7] != '-' || !ReadDigits(text, 0, 4, y) ||
        !ReadDigits(text, 5, 2, m) || !ReadDigits(text, 8, 2, d)) {
        return false;
    }
    if (y < 1 || m < 1 || m > 12 || d < 1 || d > DaysInMonth(y, m)) return false;
    days = DaysFromCivil(y, m, d);
    return true;
}

// Exactly "YYYY-MM-DD", stored as days since 1970-01-01.
Reason ParseDate(std::string_view text, ColumnVector& col) {
    int64_t days = 0;
    if (text.size() != 10 || !ReadDate(text, days)) return Invalid(col.Type(), text);
    col.AppendInt(days);
    return std::nullopt;
}

// "YYYY-MM-DD HH:MM:SS" plus an optional '.' and 1 to 6 digits; no time zone.
Reason ParseTimestamp(std::string_view text, ColumnVector& col) {
    constexpr size_t kBaseLength = 19;
    constexpr size_t kMaxFractionDigits = 6;
    const size_t n = text.size();
    if (n < kBaseLength || n == kBaseLength + 1 || n > kBaseLength + 1 + kMaxFractionDigits) {
        return Invalid(col.Type(), text);
    }

    int64_t days = 0;
    int hour = 0;
    int minute = 0;
    int second = 0;
    if (!ReadDate(text, days) || text[10] != ' ' || text[13] != ':' || text[16] != ':' ||
        !ReadDigits(text, 11, 2, hour) || !ReadDigits(text, 14, 2, minute) ||
        !ReadDigits(text, 17, 2, second) || hour > 23 || minute > 59 || second > 59) {
        return Invalid(col.Type(), text);
    }

    int fraction = 0;
    if (n > kBaseLength) {
        const size_t digits = n - kBaseLength - 1;
        if (text[kBaseLength] != '.' || !ReadDigits(text, kBaseLength + 1, digits, fraction)) {
            return Invalid(col.Type(), text);
        }
        for (size_t k = digits; k < kMaxFractionDigits; ++k) fraction *= 10;
    }

    const int64_t seconds = (static_cast<int64_t>(hour) * 60 + minute) * 60 + second;
    col.AppendInt((days - kDaysFrom1970To2000) * kMicrosPerDay + seconds * kMicrosPerSecond + fraction);
    return std::nullopt;
}

// Code points in text, or nullopt unless it is strictly valid UTF-8.
// Refuses truncated and overlong forms, surrogates and anything above U+10FFFF.
std::optional<size_t> Utf8Length(std::string_view text) {
    constexpr uint64_t kHighBits = 0x8080808080808080ULL;
    const auto* p = reinterpret_cast<const unsigned char*>(text.data());
    const auto* const end = p + text.size();
    size_t count = 0;

    while (p < end) {
        // Eight ASCII bytes at a time, the common case.
        if (end - p >= 8) {
            uint64_t word = 0;
            std::memcpy(&word, p, sizeof(word));
            if ((word & kHighBits) == 0) {
                p += 8;
                count += 8;
                continue;
            }
        }

        const unsigned char lead = *p;
        if (lead < 0x80) {
            ++p;
            ++count;
            continue;
        }

        // The sequence length, and the allowed range of the second byte.
        size_t len = 0;
        unsigned char lo = 0x80;
        unsigned char hi = 0xBF;
        if (lead >= 0xC2 && lead <= 0xDF) {
            len = 2;
        } else if (lead == 0xE0) {
            len = 3;
            lo = 0xA0;  // no overlong forms
        } else if (lead == 0xED) {
            len = 3;
            hi = 0x9F;  // no surrogates
        } else if (lead >= 0xE1 && lead <= 0xEF) {
            len = 3;
        } else if (lead == 0xF0) {
            len = 4;
            lo = 0x90;  // no overlong forms
        } else if (lead == 0xF4) {
            len = 4;
            hi = 0x8F;  // nothing above U+10FFFF
        } else if (lead >= 0xF1 && lead <= 0xF3) {
            len = 4;
        } else {
            return std::nullopt;
        }

        if (static_cast<size_t>(end - p) < len || p[1] < lo || p[1] > hi) return std::nullopt;
        for (size_t k = 2; k < len; ++k) {
            if (p[k] < 0x80 || p[k] > 0xBF) return std::nullopt;
        }
        p += len;
        ++count;
    }
    return count;
}

// Valid UTF-8; VARCHAR(n) also holds at most n characters.
Reason ParseString(std::string_view text, ColumnVector& col) {
    const auto chars = Utf8Length(text);
    if (!chars) return std::string("invalid UTF-8");

    const ColumnType& type = col.Type();
    if (type.id == ColumnTypeId::VARCHAR && *chars > type.max_length) {
        return "value too long for VARCHAR(" + std::to_string(type.max_length) + ")";
    }
    col.AppendString(text);
    return std::nullopt;
}

// n / 10^scale: one correctly rounded division up to scale 22, else two.
double ScaledToDouble(double n, size_t scale) {
    const auto exact = static_cast<size_t>(kMaxExactPow10);
    if (scale <= exact) return n / kPow10[scale];
    return (n / kPow10[exact]) / kPow10[scale - exact];
}

}  // namespace

// Picks the parse rule by the column's storage type.
std::optional<std::string> ParseField(std::string_view text, ColumnVector& col) {
    switch (col.Type().id) {
        case ColumnTypeId::BOOLEAN:
            return ParseBoolean(text, col);
        case ColumnTypeId::SMALLINT:
            return ParseInteger(text, col, std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max());
        case ColumnTypeId::INTEGER:
            return ParseInteger(text, col, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max());
        case ColumnTypeId::BIGINT:
            return ParseInteger(text, col, std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max());
        case ColumnTypeId::NUMERIC:
            return ParseNumeric(text, col);
        case ColumnTypeId::DOUBLE:
            return ParseDouble(text, col);
        case ColumnTypeId::DATE:
            return ParseDate(text, col);
        case ColumnTypeId::TIMESTAMP:
            return ParseTimestamp(text, col);
        case ColumnTypeId::VARCHAR:
        case ColumnTypeId::TEXT:
            return ParseString(text, col);
    }
    return "unknown column type " + ColumnTypeName(col.Type());
}

// Builds the Value today's executor expects for each storage type.
Value ToExecutorValue(const ColumnVector& col, size_t i) {
    if (col.IsNull(i)) {
        throw DatabaseException("Cell " + std::to_string(i) + " is NULL, which the executor can't read yet");
    }
    const ColumnType& type = col.Type();
    switch (type.id) {
        case ColumnTypeId::BOOLEAN:
        case ColumnTypeId::SMALLINT:
        case ColumnTypeId::INTEGER:
            return Value(static_cast<int32_t>(col.GetInt(i)));
        case ColumnTypeId::BIGINT:
            return Value(col.GetInt(i));
        case ColumnTypeId::NUMERIC: {
            const double n = col.Kind() == PhysicalKind::INT64 ? static_cast<double>(col.GetInt(i)) : static_cast<double>(col.GetWide(i));
            return Value(ScaledToDouble(n, type.scale));
        }
        case ColumnTypeId::DOUBLE:
            return Value(col.GetDouble(i));
        case ColumnTypeId::DATE:
            return Value::MakeDate(static_cast<int32_t>(col.GetInt(i)));
        case ColumnTypeId::TIMESTAMP:
            return Value(FormatTimestamp(col.GetInt(i)));
        case ColumnTypeId::VARCHAR:
        case ColumnTypeId::TEXT:
            return Value(std::string(col.GetString(i)));
    }
    throw DatabaseException("Unknown column type " + ColumnTypeName(type));
}

// Floor division, so a negative time still shows a time of day from 00:00.
std::string FormatTimestamp(int64_t micros_since_2000) {
    int64_t days = micros_since_2000 / kMicrosPerDay;
    int64_t rest = micros_since_2000 % kMicrosPerDay;
    if (rest < 0) {
        rest += kMicrosPerDay;
        --days;
    }
    const CivilDate date = CivilFromDays(days + kDaysFrom1970To2000);
    const int64_t seconds = rest / kMicrosPerSecond;
    const int64_t micros = rest % kMicrosPerSecond;

    char buf[64];
    int len = std::snprintf(
        buf, sizeof(buf), "%04lld-%02d-%02d %02lld:%02lld:%02lld",
        static_cast<long long>(date.year), date.month, date.day,
        static_cast<long long>(seconds / 3600), static_cast<long long>(seconds / 60 % 60),
        static_cast<long long>(seconds % 60)
    );
    if (micros != 0) {
        len += std::snprintf(buf + len, sizeof(buf) - static_cast<size_t>(len), ".%06lld", static_cast<long long>(micros));
    }
    return std::string(buf, static_cast<size_t>(len));
}

}  // namespace shilmandb
