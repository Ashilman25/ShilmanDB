#include <gtest/gtest.h>
#include "storage/columnar/field_parser.hpp"
#include "common/exception.hpp"
#include "types/value.hpp"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace shilmandb {

namespace {

// Parses text into a fresh column; true when it parsed.
bool Parses(const ColumnType& type, std::string_view text) {
    ColumnVector col(type);
    const auto reason = ParseField(text, col);
    EXPECT_EQ(col.Size(), reason ? 0u : 1u) << "a failed parse must append nothing";
    return !reason.has_value();
}

// The int64 a field parses to; fails the test if it doesn't parse.
int64_t ParsedInt(const ColumnType& type, std::string_view text) {
    ColumnVector col(type);
    const auto reason = ParseField(text, col);
    EXPECT_FALSE(reason.has_value()) << text << ": " << reason.value_or("");
    return col.Size() == 1 ? col.GetInt(0) : 0;
}

// The reason a field fails, or "" if it parses.
std::string Reason(const ColumnType& type, std::string_view text) {
    ColumnVector col(type);
    return ParseField(text, col).value_or("");
}

// An __int128 in decimal, so failures print readably.
std::string Int128ToString(__int128 v) {
    if (v == 0) return "0";
    const bool negative = v < 0;
    unsigned __int128 m = negative ? -static_cast<unsigned __int128>(v) : static_cast<unsigned __int128>(v);
    std::string out;
    while (m > 0) {
        out.insert(out.begin(), static_cast<char>('0' + static_cast<int>(m % 10)));
        m /= 10;
    }
    return negative ? "-" + out : out;
}

// True when two doubles have the same 8 bytes.
bool SameBits(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

bool IsLeap(int y) {
    return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
}

int MonthDays(int y, int m) {
    constexpr int kDays[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    return (m == 2 && IsLeap(y)) ? 29 : kDays[m];
}

// The TPC-H SF=0.01 folder, searched like the other TPC-H tests do.
std::string FindDataDir() {
    for (const std::string dir : {"bench/tpch_data/sf0.01/", "../bench/tpch_data/sf0.01/",
                                  "../../bench/tpch_data/sf0.01/"}) {
        if (std::filesystem::exists(dir + "lineitem.tbl")) return dir;
    }
    return "";
}

constexpr int64_t kDaysFrom1970To2000 = 10957;
constexpr int64_t kMicrosPerDay = 86'400'000'000LL;

}  // namespace

TEST(FieldParserTest, IntegersAndRanges) {
    const auto small = ColumnType::SmallInt();
    EXPECT_EQ(ParsedInt(small, "32767"), 32767);
    EXPECT_EQ(ParsedInt(small, "-32768"), -32768);
    EXPECT_FALSE(Parses(small, "32768"));
    EXPECT_FALSE(Parses(small, "-32769"));
    EXPECT_EQ(Reason(small, "32768"), "out of range for SMALLINT: 32768");

    const auto integer = ColumnType::Integer();
    EXPECT_EQ(ParsedInt(integer, "-5"), -5);
    EXPECT_EQ(ParsedInt(integer, "+5"), 5);
    EXPECT_EQ(ParsedInt(integer, "007"), 7);
    EXPECT_EQ(ParsedInt(integer, "2147483647"), 2147483647);
    EXPECT_FALSE(Parses(integer, "2147483648"));
    for (const char* bad : {"5x", "", " 5", "5 ", "+", "-", "+-5", "-+5", "1.0", "abc"}) {
        EXPECT_FALSE(Parses(integer, bad)) << '"' << bad << '"';
    }
    EXPECT_EQ(Reason(integer, "abc"), "invalid INTEGER: abc");

    const auto big = ColumnType::BigInt();
    EXPECT_EQ(ParsedInt(big, "9223372036854775807"), INT64_MAX);
    EXPECT_EQ(ParsedInt(big, "-9223372036854775808"), INT64_MIN);
    EXPECT_FALSE(Parses(big, "9223372036854775808"));
    EXPECT_EQ(Reason(big, "9223372036854775808"), "out of range for BIGINT: 9223372036854775808");
}

TEST(FieldParserTest, BooleanForms) {
    const auto b = ColumnType::Boolean();
    EXPECT_EQ(ParsedInt(b, "TRUE"), 1);
    EXPECT_EQ(ParsedInt(b, "off"), 0);
    EXPECT_EQ(ParsedInt(b, "0"), 0);
    EXPECT_EQ(ParsedInt(b, "1"), 1);
    EXPECT_EQ(ParsedInt(b, "t"), 1);
    EXPECT_EQ(ParsedInt(b, "F"), 0);
    EXPECT_EQ(ParsedInt(b, "Yes"), 1);
    EXPECT_EQ(ParsedInt(b, "nO"), 0);
    EXPECT_EQ(ParsedInt(b, "On"), 1);
    EXPECT_EQ(ParsedInt(b, "faLSE"), 0);
    for (const char* bad : {"2", "", "tru", "truee", " t", "y", "of"}) {
        EXPECT_FALSE(Parses(b, bad)) << '"' << bad << '"';
    }
    EXPECT_EQ(Reason(b, "2"), "invalid BOOLEAN: 2");
}

TEST(FieldParserTest, NumericRounding) {
    const auto money = ColumnType::Numeric(15, 2);
    EXPECT_EQ(ParsedInt(money, "1.005"), 101);
    EXPECT_EQ(ParsedInt(money, "-1.005"), -101);
    EXPECT_EQ(ParsedInt(money, "1.004"), 100);
    EXPECT_EQ(ParsedInt(money, "1.0049999"), 100);
    EXPECT_EQ(ParsedInt(ColumnType::Numeric(15, 0), "12.5"), 13);
    EXPECT_EQ(ParsedInt(ColumnType::Numeric(15, 0), "-12.5"), -13);
    EXPECT_EQ(ParsedInt(ColumnType::Numeric(15, 0), "12.4"), 12);

    EXPECT_EQ(ParsedInt(money, "9999999999999.99"), 999999999999999);
    EXPECT_EQ(ParsedInt(money, "-9999999999999.99"), -999999999999999);
    EXPECT_EQ(Reason(money, "10000000000000.00"), "numeric field overflow");
    // The limit applies after rounding, as in Postgres.
    EXPECT_EQ(Reason(money, "9999999999999.995"), "numeric field overflow");
    EXPECT_EQ(Reason(ColumnType::Numeric(2, 2), "0.995"), "numeric field overflow");
    EXPECT_EQ(ParsedInt(ColumnType::Numeric(2, 2), "0.994"), 99);

    EXPECT_EQ(ParsedInt(money, "1"), 100);
    EXPECT_EQ(ParsedInt(money, ".5"), 50);
    EXPECT_EQ(ParsedInt(money, "5."), 500);
    EXPECT_EQ(ParsedInt(money, "+2.5"), 250);
    EXPECT_EQ(ParsedInt(money, "000123.45"), 12345);
    EXPECT_EQ(ParsedInt(money, "-0"), 0);
    EXPECT_EQ(ParsedInt(money, "-0.001"), 0);
    EXPECT_EQ(ParsedInt(ColumnType::Numeric(1, 0), "9"), 9);
    EXPECT_EQ(Reason(ColumnType::Numeric(1, 0), "10"), "numeric field overflow");

    for (const char* bad : {".", "-", "+", "", "1e5", " 1", "1 ", "1.2.3", "--1", "1,5", "abc"}) {
        EXPECT_FALSE(Parses(money, bad)) << '"' << bad << '"';
        EXPECT_NE(Reason(money, bad), "numeric field overflow") << '"' << bad << '"';
    }

    // 38 digits need 128 bits.
    const auto wide = ColumnType::Numeric(38, 0);
    const std::string digits38 = "12345678901234567890123456789012345678";
    ColumnVector col(wide);
    ASSERT_FALSE(ParseField(digits38, col).has_value());
    ASSERT_FALSE(ParseField("-" + std::string(38, '9'), col).has_value());
    ASSERT_EQ(col.Size(), 2u);
    EXPECT_EQ(Int128ToString(col.GetWide(0)), digits38);
    EXPECT_EQ(Int128ToString(col.GetWide(1)), "-" + std::string(38, '9'));
    EXPECT_EQ(Reason(wide, std::string(39, '9')), "numeric field overflow");
    EXPECT_EQ(Reason(wide, std::string(38, '9') + ".5"), "numeric field overflow");

    ColumnVector scaled(ColumnType::Numeric(38, 38));
    ASSERT_FALSE(ParseField("0.5", scaled).has_value());
    EXPECT_EQ(Int128ToString(scaled.GetWide(0)), "5" + std::string(37, '0'));
}

TEST(FieldParserTest, NumericMatchesStodBitForBit) {
    const std::string dir = FindDataDir();
    if (dir.empty()) {
        GTEST_SKIP() << "TPC-H SF=0.01 data not found. Run bench/tpch/generate_tpch_data.sh first.";
    }
    std::ifstream in(dir + "lineitem.tbl");
    ASSERT_TRUE(in.is_open());

    // l_quantity, l_extendedprice, l_discount, l_tax.
    constexpr size_t kFirstMoney = 4;
    constexpr size_t kLastMoney = 7;
    constexpr int kLines = 20000;

    ColumnVector col(ColumnType::Numeric(15, 2));
    std::vector<std::string> texts;
    std::string line;
    for (int n = 0; n < kLines && std::getline(in, line); ++n) {
        size_t start = 0;
        for (size_t field = 0; field <= kLastMoney; ++field) {
            const size_t bar = line.find('|', start);
            ASSERT_NE(bar, std::string::npos) << line;
            if (field >= kFirstMoney) {
                texts.push_back(line.substr(start, bar - start));
                const auto reason = ParseField(texts.back(), col);
                ASSERT_FALSE(reason.has_value()) << texts.back() << ": " << *reason;
            }
            start = bar + 1;
        }
    }

    ASSERT_EQ(col.Size(), texts.size());
    size_t compared = 0;
    for (size_t i = 0; i < texts.size(); ++i) {
        const double stod = std::stod(texts[i]);
        const double ours = ToExecutorValue(col, i).decimal_;
        EXPECT_EQ(std::memcmp(&ours, &stod, sizeof(double)), 0) << texts[i];
        ++compared;
    }
    EXPECT_EQ(compared, static_cast<size_t>(kLines) * 4);
    std::cout << "NumericMatchesStodBitForBit compared " << compared << " money values\n";
}

TEST(FieldParserTest, DoubleMatchesStod) {
    const auto d = ColumnType::Double();
    for (const char* text : {"0.1", "1e-300", "24710.35", "-3.25", " 2.5", "1E10"}) {
        ColumnVector col(d);
        ASSERT_FALSE(ParseField(text, col).has_value()) << text;
        EXPECT_TRUE(SameBits(col.GetDouble(0), std::stod(text))) << text;
    }
    for (const char* bad : {"1.0abc", "", " ", "2.5 ", "abc", "1e999"}) {
        EXPECT_FALSE(Parses(d, bad)) << '"' << bad << '"';
    }
    // A field longer than the stack buffer still parses.
    const std::string long_text = "1." + std::string(300, '5');
    ColumnVector col(d);
    ASSERT_FALSE(ParseField(long_text, col).has_value());
    EXPECT_TRUE(SameBits(col.GetDouble(0), std::stod(long_text)));
}

TEST(FieldParserTest, DateRules) {
    const auto date = ColumnType::Date();
    EXPECT_EQ(ParsedInt(date, "1995-03-15"), Value::FromString(TypeId::DATE, "1995-03-15").date_);
    EXPECT_EQ(ParsedInt(date, "1969-12-31"), -1);
    EXPECT_EQ(ParsedInt(date, "1970-01-01"), 0);
    EXPECT_EQ(ParsedInt(date, "2000-01-01"), kDaysFrom1970To2000);
    EXPECT_TRUE(Parses(date, "2000-02-29"));
    EXPECT_TRUE(Parses(date, "2024-02-29"));
    EXPECT_TRUE(Parses(date, "0001-01-01"));
    EXPECT_TRUE(Parses(date, "9999-12-31"));
    for (const char* bad : {"1995-02-30", "95-03-15", "1900-02-29", "1995-13-01", "1995-00-10",
                            "1995-01-00", "1995-04-31", "1995-3-15", "1995/03/15", "1995-03-15 ",
                            " 1995-03-15", "1995-03-1x", "", "0000-01-01"}) {
        EXPECT_FALSE(Parses(date, bad)) << '"' << bad << '"';
    }
    EXPECT_EQ(Reason(date, "1995-02-30"), "invalid DATE: 1995-02-30");
}

TEST(FieldParserTest, DateMatchesValueEveryDay1900To2100) {
    ColumnVector col(ColumnType::Date());
    size_t days_checked = 0;
    int64_t previous = 0;
    char text[16];
    for (int y = 1900; y <= 2100; ++y) {
        for (int m = 1; m <= 12; ++m) {
            for (int d = 1; d <= MonthDays(y, m); ++d) {
                std::snprintf(text, sizeof(text), "%04d-%02d-%02d", y, m, d);
                col.Clear();
                ASSERT_FALSE(ParseField(text, col).has_value()) << text;
                const int64_t days = col.GetInt(0);
                ASSERT_EQ(days, Value::FromString(TypeId::DATE, text).date_) << text;
                if (days_checked > 0) ASSERT_EQ(days, previous + 1) << text;
                // The inverse, through the timestamp text.
                ASSERT_EQ(FormatTimestamp((days - kDaysFrom1970To2000) * kMicrosPerDay),
                          std::string(text) + " 00:00:00");
                previous = days;
                ++days_checked;
            }
        }
    }
    EXPECT_EQ(days_checked, 73414u);
}

TEST(FieldParserTest, TimestampRoundTrip) {
    const auto ts = ColumnType::Timestamp();
    EXPECT_EQ(ParsedInt(ts, "2000-01-01 00:00:00"), 0);
    EXPECT_EQ(ParsedInt(ts, "2000-01-01 00:00:01.5"), 1'500'000);
    EXPECT_EQ(ParsedInt(ts, "1999-12-31 23:59:59.999999"), -1);

    ColumnVector col(ts);
    ASSERT_FALSE(ParseField("2026-11-16 12:34:56.789", col).has_value());
    ASSERT_FALSE(ParseField("2000-01-01 00:00:00", col).has_value());
    EXPECT_EQ(ToExecutorValue(col, 0).varchar_, "2026-11-16 12:34:56.789000");
    EXPECT_EQ(ToExecutorValue(col, 1).varchar_, "2000-01-01 00:00:00");

    EXPECT_EQ(FormatTimestamp(-1), "1999-12-31 23:59:59.999999");
    EXPECT_EQ(FormatTimestamp(0), "2000-01-01 00:00:00");
    EXPECT_EQ(FormatTimestamp(-kMicrosPerDay), "1999-12-31 00:00:00");
    EXPECT_FALSE(FormatTimestamp(INT64_MIN).empty());
    EXPECT_FALSE(FormatTimestamp(INT64_MAX).empty());

    for (const char* bad : {"2026-11-16", "2026-11-16T12:34:56", "2026-11-16 24:00:00",
                            "2026-11-16 12:60:00", "2026-11-16 12:34:60", "2026-11-16 12:34:56.",
                            "2026-11-16 12:34:56.1234567", "2026-11-16 12:34:56+00",
                            "2026-11-16 12:34:56 ", "2026-02-30 00:00:00", "2026-11-16 1:34:56",
                            "2026-11-16 12:34:56.12a"}) {
        EXPECT_FALSE(Parses(ts, bad)) << '"' << bad << '"';
    }
}

TEST(FieldParserTest, VarcharCountsCharacters) {
    const auto v5 = ColumnType::Varchar(5);
    const std::string hello = "h\xC3\xA9llo";  // 6 bytes, 5 characters
    ASSERT_EQ(hello.size(), 6u);
    ColumnVector col(v5);
    ASSERT_FALSE(ParseField(hello, col).has_value());
    EXPECT_EQ(col.GetString(0), hello);

    EXPECT_EQ(Reason(v5, hello + "!"), "value too long for VARCHAR(5)");
    EXPECT_EQ(Reason(v5, "abcdef"), "value too long for VARCHAR(5)");
    EXPECT_TRUE(Parses(v5, "abcde"));
    EXPECT_TRUE(Parses(v5, ""));
    EXPECT_EQ(Reason(v5, "\xC3"), "invalid UTF-8");

    // Two-, three- and four-byte characters each count once.
    const std::string euro = "\xE2\x82\xAC";
    const std::string smile = "\xF0\x9F\x98\x80";
    EXPECT_TRUE(Parses(v5, smile + smile + smile + smile + smile));
    EXPECT_FALSE(Parses(v5, smile + smile + smile + smile + smile + smile));
    EXPECT_TRUE(Parses(v5, "a\xC3\xA9" + euro + smile + "z"));
    EXPECT_TRUE(Parses(ColumnType::Varchar(1), "\xF4\x8F\xBF\xBF"));  // U+10FFFF

    // Long ASCII runs go through the eight-bytes-at-a-time path.
    EXPECT_TRUE(Parses(ColumnType::Varchar(20), std::string(20, 'x')));
    EXPECT_FALSE(Parses(ColumnType::Varchar(20), std::string(21, 'x')));
    EXPECT_TRUE(Parses(ColumnType::Varchar(17), std::string(16, 'x') + smile));

    const std::vector<std::string> invalid = {
        "\xC3",              // truncated
        "\xE2\x82",          // truncated
        "\xF0\x9F\x98",      // truncated
        "\x80",              // lone continuation byte
        "\xC3\x28",          // bad continuation byte
        "\xC0\x80",          // overlong
        "\xC1\xBF",          // overlong
        "\xE0\x80\x80",      // overlong
        "\xF0\x80\x80\x80",  // overlong
        "\xED\xA0\x80",      // surrogate U+D800
        "\xED\xBF\xBF",      // surrogate U+DFFF
        "\xF4\x90\x80\x80",  // above U+10FFFF
        "\xF5\x80\x80\x80",  // above U+10FFFF
        "\xFF",
        "abcdefghij\xC3",    // bad byte after an ASCII run
    };
    for (const auto& bad : invalid) {
        EXPECT_EQ(Reason(ColumnType::Varchar(100), bad), "invalid UTF-8");
        EXPECT_EQ(Reason(ColumnType::Text(), bad), "invalid UTF-8");
    }

    const std::string long_text(10000, 'q');
    ColumnVector text(ColumnType::Text());
    ASSERT_FALSE(ParseField(long_text + smile, text).has_value());
    EXPECT_EQ(text.GetString(0), long_text + smile);
}

TEST(FieldParserTest, ExecutorValueTypes) {
    struct Case {
        ColumnType type;
        const char* text;
    };
    const std::vector<Case> cases = {
        {ColumnType::Boolean(), "t"},
        {ColumnType::SmallInt(), "-7"},
        {ColumnType::Integer(), "7"},
        {ColumnType::BigInt(), "9223372036854775807"},
        {ColumnType::Numeric(15, 2), "1.25"},
        {ColumnType::Numeric(38, 10), "1.25"},
        {ColumnType::Double(), "1.5"},
        {ColumnType::Date(), "1995-03-15"},
        {ColumnType::Timestamp(), "2000-01-01 00:00:00"},
        {ColumnType::Varchar(5), "abc"},
        {ColumnType::Text(), "abc"},
    };
    for (const auto& c : cases) {
        ColumnVector col(c.type);
        ASSERT_FALSE(ParseField(c.text, col).has_value()) << ColumnTypeName(c.type);
        col.AppendNull();
        EXPECT_EQ(ToExecutorValue(col, 0).type_, ExecutorType(c.type)) << ColumnTypeName(c.type);
        EXPECT_THROW((void)ToExecutorValue(col, 1), DatabaseException) << ColumnTypeName(c.type);
    }

    auto value_of = [](const ColumnType& type, const char* text) {
        ColumnVector col(type);
        EXPECT_FALSE(ParseField(text, col).has_value()) << text;
        return ToExecutorValue(col, 0);
    };
    EXPECT_EQ(value_of(ColumnType::Boolean(), "yes").integer_, 1);
    EXPECT_EQ(value_of(ColumnType::Boolean(), "no").integer_, 0);
    EXPECT_EQ(value_of(ColumnType::SmallInt(), "-7").integer_, -7);
    EXPECT_EQ(value_of(ColumnType::BigInt(), "9223372036854775807").bigint_, INT64_MAX);
    EXPECT_EQ(value_of(ColumnType::Numeric(15, 2), "-24710.35").decimal_, std::stod("-24710.35"));
    EXPECT_EQ(value_of(ColumnType::Numeric(38, 10), "1.25").decimal_, 1.25);
    EXPECT_DOUBLE_EQ(value_of(ColumnType::Numeric(38, 30), "1.5").decimal_, 1.5);
    EXPECT_EQ(value_of(ColumnType::Date(), "1995-03-15").date_,
              Value::FromString(TypeId::DATE, "1995-03-15").date_);
    EXPECT_EQ(value_of(ColumnType::Text(), "h\xC3\xA9").varchar_, "h\xC3\xA9");
}

}  // namespace shilmandb
