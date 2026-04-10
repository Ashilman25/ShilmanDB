#include <gtest/gtest.h>
#include "types/value.hpp"
#include "common/exception.hpp"
#include <cmath>
#include <limits>
#include <unordered_set>

namespace shilmandb {

// --- Construction & IsNull ---

TEST(ValueTest, DefaultIsInvalid) {
    Value v;
    EXPECT_TRUE(v.IsNull());
    EXPECT_EQ(v.type_, TypeId::INVALID);
}

TEST(ValueTest, ConstructInteger) {
    Value v(static_cast<int32_t>(42));
    EXPECT_EQ(v.type_, TypeId::INTEGER);
    EXPECT_EQ(v.integer_, 42);
    EXPECT_FALSE(v.IsNull());
}

TEST(ValueTest, ConstructBigint) {
    Value v(static_cast<int64_t>(1'000'000'000'000LL));
    EXPECT_EQ(v.type_, TypeId::BIGINT);
    EXPECT_EQ(v.bigint_, 1'000'000'000'000LL);
}

TEST(ValueTest, ConstructDecimal) {
    Value v(3.14);
    EXPECT_EQ(v.type_, TypeId::DECIMAL);
    EXPECT_DOUBLE_EQ(v.decimal_, 3.14);
}

TEST(ValueTest, ConstructVarchar) {
    Value v(std::string("hello"));
    EXPECT_EQ(v.type_, TypeId::VARCHAR);
    EXPECT_EQ(v.varchar_, "hello");
}

TEST(ValueTest, MakeDate) {
    Value v = Value::MakeDate(0);
    EXPECT_EQ(v.type_, TypeId::DATE);
    EXPECT_EQ(v.date_, 0);
}

// --- Comparison operators ---

TEST(ValueTest, IntegerEquality) {
    Value a(static_cast<int32_t>(10));
    Value b(static_cast<int32_t>(10));
    Value c(static_cast<int32_t>(20));
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    EXPECT_TRUE(a != c);
    EXPECT_FALSE(a != b);
}

TEST(ValueTest, IntegerOrdering) {
    Value a(static_cast<int32_t>(5));
    Value b(static_cast<int32_t>(10));
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(b > a);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(b >= a);
    EXPECT_TRUE(a <= a);
    EXPECT_TRUE(a >= a);
}

TEST(ValueTest, BigintComparison) {
    Value a(static_cast<int64_t>(100));
    Value b(static_cast<int64_t>(200));
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a == a);
}

TEST(ValueTest, DecimalComparison) {
    Value a(1.5);
    Value b(2.5);
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a == a);
}

TEST(ValueTest, VarcharComparison) {
    Value a(std::string("apple"));
    Value b(std::string("banana"));
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a == a);
    EXPECT_TRUE(b > a);
}

TEST(ValueTest, DateComparison) {
    Value a = Value::MakeDate(100);
    Value b = Value::MakeDate(200);
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a == a);
}

TEST(ValueTest, CrossTypeComparisonCoerces) {
    Value i(static_cast<int32_t>(1));
    Value d(1.0);
    // INTEGER promotes to DECIMAL for comparison
    EXPECT_TRUE(i == d);
    EXPECT_FALSE(i < d);

    // Incompatible types still throw
    Value s(std::string("hello"));
    EXPECT_THROW((void)(i == s), DatabaseException);
}

// --- Arithmetic ---

TEST(ValueTest, IntegerArithmetic) {
    Value a(static_cast<int32_t>(10));
    Value b(static_cast<int32_t>(3));
    EXPECT_EQ(a.Add(b).integer_, 13);
    EXPECT_EQ(a.Subtract(b).integer_, 7);
    EXPECT_EQ(a.Multiply(b).integer_, 30);
    EXPECT_EQ(a.Divide(b).integer_, 3);
}

TEST(ValueTest, BigintArithmetic) {
    Value a(static_cast<int64_t>(100));
    Value b(static_cast<int64_t>(25));
    EXPECT_EQ(a.Add(b).bigint_, 125);
    EXPECT_EQ(a.Subtract(b).bigint_, 75);
    EXPECT_EQ(a.Multiply(b).bigint_, 2500);
    EXPECT_EQ(a.Divide(b).bigint_, 4);
}

TEST(ValueTest, DecimalArithmetic) {
    Value a(10.0);
    Value b(3.0);
    EXPECT_DOUBLE_EQ(a.Add(b).decimal_, 13.0);
    EXPECT_DOUBLE_EQ(a.Subtract(b).decimal_, 7.0);
    EXPECT_DOUBLE_EQ(a.Multiply(b).decimal_, 30.0);
    EXPECT_NEAR(a.Divide(b).decimal_, 3.333333, 0.001);
}

TEST(ValueTest, IntegerDivisionByZeroThrows) {
    Value a(static_cast<int32_t>(10));
    Value zero(static_cast<int32_t>(0));
    EXPECT_THROW(a.Divide(zero), DatabaseException);
}

TEST(ValueTest, BigintDivisionByZeroThrows) {
    Value a(static_cast<int64_t>(10));
    Value zero(static_cast<int64_t>(0));
    EXPECT_THROW(a.Divide(zero), DatabaseException);
}

TEST(ValueTest, DecimalDivisionByZeroReturnsInfinity) {
    Value a(10.0);
    Value zero(0.0);
    Value result = a.Divide(zero);
    EXPECT_TRUE(std::isinf(result.decimal_));
}

TEST(ValueTest, ArithmeticOnVarcharThrows) {
    Value a(std::string("hello"));
    Value b(std::string("world"));
    EXPECT_THROW(a.Add(b), DatabaseException);
    EXPECT_THROW(a.Subtract(b), DatabaseException);
    EXPECT_THROW(a.Multiply(b), DatabaseException);
    EXPECT_THROW(a.Divide(b), DatabaseException);
}

TEST(ValueTest, ArithmeticOnDateThrows) {
    Value a = Value::MakeDate(100);
    Value b = Value::MakeDate(200);
    EXPECT_THROW(a.Add(b), DatabaseException);
}

TEST(ValueTest, CrossTypeArithmeticCoerces) {
    Value i(static_cast<int32_t>(3));
    Value d(1.5);
    // INTEGER promotes to DECIMAL for arithmetic
    Value result = i.Add(d);
    EXPECT_EQ(result.type_, TypeId::DECIMAL);
    EXPECT_DOUBLE_EQ(result.decimal_, 4.5);

    // Incompatible types still throw
    Value s(std::string("hello"));
    EXPECT_THROW(i.Add(s), DatabaseException);
}

// --- ToString / FromString round-trips ---

TEST(ValueTest, IntegerToStringRoundTrip) {
    Value v(static_cast<int32_t>(42));
    EXPECT_EQ(v.ToString(), "42");
    Value parsed = Value::FromString(TypeId::INTEGER, "42");
    EXPECT_EQ(parsed, v);
}

TEST(ValueTest, BigintToStringRoundTrip) {
    Value v(static_cast<int64_t>(1'000'000'000'000LL));
    EXPECT_EQ(v.ToString(), "1000000000000");
    Value parsed = Value::FromString(TypeId::BIGINT, "1000000000000");
    EXPECT_EQ(parsed, v);
}

TEST(ValueTest, DecimalToStringRoundTrip) {
    Value v(3.14);
    EXPECT_EQ(v.ToString(), "3.14");
    Value parsed = Value::FromString(TypeId::DECIMAL, "3.14");
    EXPECT_DOUBLE_EQ(parsed.decimal_, 3.14);
}

TEST(ValueTest, VarcharToStringRoundTrip) {
    Value v(std::string("hello world"));
    EXPECT_EQ(v.ToString(), "hello world");
    Value parsed = Value::FromString(TypeId::VARCHAR, "hello world");
    EXPECT_EQ(parsed, v);
}

TEST(ValueTest, DateToStringEpoch) {
    Value v = Value::MakeDate(0);
    EXPECT_EQ(v.ToString(), "1970-01-01");
}

TEST(ValueTest, DateToStringKnownDate) {
    // 2024-01-15 = 19737 days since 1970-01-01
    // (verified via Python: (date(2024,1,15) - date(1970,1,1)).days == 19737)
    Value v = Value::MakeDate(19737);
    EXPECT_EQ(v.ToString(), "2024-01-15");
}

TEST(ValueTest, DateFromStringRoundTrip) {
    Value v = Value::FromString(TypeId::DATE, "2024-01-15");
    EXPECT_EQ(v.type_, TypeId::DATE);
    EXPECT_EQ(v.ToString(), "2024-01-15");
}

TEST(ValueTest, NegativeDate) {
    Value v = Value::MakeDate(-1);
    EXPECT_EQ(v.ToString(), "1969-12-31");
}

// --- Hash ---

TEST(ValueTest, EqualValuesEqualHash) {
    Value a(static_cast<int32_t>(42));
    Value b(static_cast<int32_t>(42));
    EXPECT_EQ(a.Hash(), b.Hash());
}

TEST(ValueTest, HashWorksInUnorderedSet) {
    std::unordered_set<Value> s;
    s.insert(Value(static_cast<int32_t>(1)));
    s.insert(Value(static_cast<int32_t>(1)));
    s.insert(Value(static_cast<int32_t>(2)));
    EXPECT_EQ(s.size(), 2u);
}

// --- GetFixedLength ---

TEST(ValueTest, GetFixedLength) {
    EXPECT_EQ(Value(static_cast<int32_t>(0)).GetFixedLength(), 4u);
    EXPECT_EQ(Value(static_cast<int64_t>(0)).GetFixedLength(), 8u);
    EXPECT_EQ(Value(0.0).GetFixedLength(), 8u);
    EXPECT_EQ(Value::MakeDate(0).GetFixedLength(), 4u);
    EXPECT_EQ(Value(std::string("")).GetFixedLength(), 0u);
}

}  // namespace shilmandb
