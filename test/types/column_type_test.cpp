#include <gtest/gtest.h>
#include "types/column_type.hpp"
#include "types/value.hpp"
#include "catalog/schema.hpp"
#include "common/exception.hpp"
#include <ostream>

namespace shilmandb {

// Lets gtest print a type by name when a check fails.
void PrintTo(const ColumnType& type, std::ostream* os) {
    *os << ColumnTypeName(type);
}

TEST(ColumnTypeTest, DefaultsFromTypeId) {
    EXPECT_EQ(DefaultColumnType(TypeId::INTEGER), ColumnType::Integer());
    EXPECT_EQ(DefaultColumnType(TypeId::BIGINT), ColumnType::BigInt());
    EXPECT_EQ(DefaultColumnType(TypeId::DECIMAL), ColumnType::Double());
    EXPECT_EQ(DefaultColumnType(TypeId::VARCHAR), ColumnType::Text());
    EXPECT_EQ(DefaultColumnType(TypeId::DATE), ColumnType::Date());
    EXPECT_THROW((void)DefaultColumnType(TypeId::INVALID), DatabaseException);
}

TEST(ColumnTypeTest, ExecutorViewOfEveryType) {
    EXPECT_EQ(ExecutorType(ColumnType::Boolean()), TypeId::INTEGER);
    EXPECT_EQ(ExecutorType(ColumnType::SmallInt()), TypeId::INTEGER);
    EXPECT_EQ(ExecutorType(ColumnType::Integer()), TypeId::INTEGER);
    EXPECT_EQ(ExecutorType(ColumnType::BigInt()), TypeId::BIGINT);
    EXPECT_EQ(ExecutorType(ColumnType::Numeric(15, 2)), TypeId::DECIMAL);
    EXPECT_EQ(ExecutorType(ColumnType::Numeric(38, 10)), TypeId::DECIMAL);
    EXPECT_EQ(ExecutorType(ColumnType::Double()), TypeId::DECIMAL);
    EXPECT_EQ(ExecutorType(ColumnType::Date()), TypeId::DATE);
    EXPECT_EQ(ExecutorType(ColumnType::Timestamp()), TypeId::VARCHAR);
    EXPECT_EQ(ExecutorType(ColumnType::Varchar(25)), TypeId::VARCHAR);
    EXPECT_EQ(ExecutorType(ColumnType::Text()), TypeId::VARCHAR);
}

TEST(ColumnTypeTest, NumericLimits) {
    EXPECT_THROW((void)ColumnType::Numeric(0, 0), DatabaseException);
    EXPECT_THROW((void)ColumnType::Numeric(39, 0), DatabaseException);
    EXPECT_THROW((void)ColumnType::Numeric(5, 6), DatabaseException);
    EXPECT_THROW((void)ColumnType::Numeric(5, -1), DatabaseException);
    EXPECT_NO_THROW((void)ColumnType::Numeric(38, 38));
    EXPECT_NO_THROW((void)ColumnType::Numeric(1, 0));

    const auto n = ColumnType::Numeric(15, 2);
    EXPECT_EQ(n.id, ColumnTypeId::NUMERIC);
    EXPECT_EQ(n.precision, 15);
    EXPECT_EQ(n.scale, 2);
    EXPECT_EQ(n.max_length, 0u);
}

TEST(ColumnTypeTest, VarcharLimits) {
    EXPECT_THROW((void)ColumnType::Varchar(0), DatabaseException);

    const auto v = ColumnType::Varchar(25);
    EXPECT_EQ(v.id, ColumnTypeId::VARCHAR);
    EXPECT_EQ(v.max_length, 25u);
    EXPECT_EQ(v.precision, 0);
    EXPECT_EQ(v.scale, 0);
}

TEST(ColumnTypeTest, Equality) {
    EXPECT_EQ(ColumnType::Numeric(15, 2), ColumnType::Numeric(15, 2));
    EXPECT_NE(ColumnType::Numeric(15, 2), ColumnType::Numeric(15, 3));
    EXPECT_NE(ColumnType::Numeric(15, 2), ColumnType::Numeric(16, 2));
    EXPECT_EQ(ColumnType::Varchar(25), ColumnType::Varchar(25));
    EXPECT_NE(ColumnType::Varchar(25), ColumnType::Varchar(26));
    EXPECT_NE(ColumnType::Varchar(25), ColumnType::Text());
    EXPECT_NE(ColumnType::Integer(), ColumnType::BigInt());
    EXPECT_EQ(ColumnType{}, ColumnType::Integer());
}

TEST(ColumnTypeTest, PhysicalKinds) {
    EXPECT_EQ(PhysicalKindOf(ColumnType::Numeric(18, 2)), PhysicalKind::INT64);
    EXPECT_EQ(PhysicalKindOf(ColumnType::Numeric(19, 2)), PhysicalKind::INT128);
    EXPECT_EQ(PhysicalKindOf(ColumnType::Boolean()), PhysicalKind::INT64);
    EXPECT_EQ(PhysicalKindOf(ColumnType::SmallInt()), PhysicalKind::INT64);
    EXPECT_EQ(PhysicalKindOf(ColumnType::Integer()), PhysicalKind::INT64);
    EXPECT_EQ(PhysicalKindOf(ColumnType::BigInt()), PhysicalKind::INT64);
    EXPECT_EQ(PhysicalKindOf(ColumnType::Date()), PhysicalKind::INT64);
    EXPECT_EQ(PhysicalKindOf(ColumnType::Timestamp()), PhysicalKind::INT64);
    EXPECT_EQ(PhysicalKindOf(ColumnType::Double()), PhysicalKind::DOUBLE);
    EXPECT_EQ(PhysicalKindOf(ColumnType::Varchar(10)), PhysicalKind::STRING);
    EXPECT_EQ(PhysicalKindOf(ColumnType::Text()), PhysicalKind::STRING);
}

TEST(ColumnTypeTest, PlainWidths) {
    EXPECT_EQ(PlainWidth(ColumnType::Boolean()), 1u);
    EXPECT_EQ(PlainWidth(ColumnType::SmallInt()), 2u);
    EXPECT_EQ(PlainWidth(ColumnType::Integer()), 4u);
    EXPECT_EQ(PlainWidth(ColumnType::Date()), 4u);
    EXPECT_EQ(PlainWidth(ColumnType::BigInt()), 8u);
    EXPECT_EQ(PlainWidth(ColumnType::Timestamp()), 8u);
    EXPECT_EQ(PlainWidth(ColumnType::Numeric(18, 2)), 8u);
    EXPECT_EQ(PlainWidth(ColumnType::Double()), 8u);
    EXPECT_EQ(PlainWidth(ColumnType::Numeric(19, 2)), 16u);
    EXPECT_EQ(PlainWidth(ColumnType::Varchar(25)), 0u);
    EXPECT_EQ(PlainWidth(ColumnType::Text()), 0u);
}

TEST(ColumnTypeTest, Names) {
    EXPECT_EQ(ColumnTypeName(ColumnType::Boolean()), "BOOLEAN");
    EXPECT_EQ(ColumnTypeName(ColumnType::SmallInt()), "SMALLINT");
    EXPECT_EQ(ColumnTypeName(ColumnType::Integer()), "INTEGER");
    EXPECT_EQ(ColumnTypeName(ColumnType::BigInt()), "BIGINT");
    EXPECT_EQ(ColumnTypeName(ColumnType::Numeric(15, 2)), "NUMERIC(15,2)");
    EXPECT_EQ(ColumnTypeName(ColumnType::Double()), "DOUBLE");
    EXPECT_EQ(ColumnTypeName(ColumnType::Date()), "DATE");
    EXPECT_EQ(ColumnTypeName(ColumnType::Timestamp()), "TIMESTAMP");
    EXPECT_EQ(ColumnTypeName(ColumnType::Varchar(25)), "VARCHAR(25)");
    EXPECT_EQ(ColumnTypeName(ColumnType::Text()), "TEXT");
}

TEST(ColumnTypeTest, ColumnFromColumnType) {
    const Column money("x", ColumnType::Numeric(15, 2));
    EXPECT_EQ(money.name, "x");
    EXPECT_EQ(money.type, TypeId::DECIMAL);
    EXPECT_EQ(money.fixed_length, 8u);
    EXPECT_FALSE(money.nullable);
    EXPECT_EQ(money.storage_type, ColumnType::Numeric(15, 2));

    const Column flag("f", ColumnType::Boolean(), true);
    EXPECT_EQ(flag.type, TypeId::INTEGER);
    EXPECT_EQ(flag.fixed_length, 4u);
    EXPECT_TRUE(flag.nullable);

    const Column stamp("t", ColumnType::Timestamp());
    EXPECT_EQ(stamp.type, TypeId::VARCHAR);
    EXPECT_EQ(stamp.fixed_length, 4u);
}

TEST(ColumnTypeTest, ColumnFromTypeIdKeepsDefault) {
    const Column name("x", TypeId::VARCHAR);
    EXPECT_EQ(name.type, TypeId::VARCHAR);
    EXPECT_EQ(name.fixed_length, 4u);
    EXPECT_EQ(name.storage_type, ColumnType::Text());

    EXPECT_EQ(Column("d", TypeId::DECIMAL).storage_type, ColumnType::Double());
    EXPECT_EQ(Column("i", TypeId::INTEGER, true).storage_type, ColumnType::Integer());
    EXPECT_TRUE(Column("i", TypeId::INTEGER, true).nullable);
}

TEST(ColumnTypeTest, ColumnFromInvalidTypeIdStillBuilds) {
    EXPECT_NO_THROW(Column("x", TypeId::INVALID));
    const Column col("x", TypeId::INVALID);
    EXPECT_EQ(col.type, TypeId::INVALID);
    EXPECT_EQ(col.fixed_length, 0u);
}

TEST(ColumnTypeTest, SchemaOffsetsUnchangedByStorageType) {
    const Schema schema({Column("a", ColumnType::SmallInt()),
                         Column("b", ColumnType::Numeric(15, 2)),
                         Column("c", ColumnType::Varchar(10)),
                         Column("d", TypeId::DATE)});
    EXPECT_EQ(schema.GetOffset(0), 0u);
    EXPECT_EQ(schema.GetOffset(1), 4u);
    EXPECT_EQ(schema.GetOffset(2), 12u);
    EXPECT_EQ(schema.GetOffset(3), 16u);
    EXPECT_EQ(schema.GetFixedLength(), 20u);
    ASSERT_EQ(schema.GetVarcharOffsets().size(), 1u);
    EXPECT_EQ(schema.GetVarcharOffsets()[0], 12u);
}

TEST(ColumnTypeTest, ValueDefaultFor) {
    const Value dec = Value::DefaultFor(TypeId::DECIMAL);
    EXPECT_EQ(dec.type_, TypeId::DECIMAL);
    EXPECT_EQ(dec.decimal_, 0.0);

    const Value big = Value::DefaultFor(TypeId::BIGINT);
    EXPECT_EQ(big.type_, TypeId::BIGINT);
    EXPECT_EQ(big.bigint_, 0);

    const Value integer = Value::DefaultFor(TypeId::INTEGER);
    EXPECT_EQ(integer.type_, TypeId::INTEGER);
    EXPECT_EQ(integer.integer_, 0);

    const Value text = Value::DefaultFor(TypeId::VARCHAR);
    EXPECT_EQ(text.type_, TypeId::VARCHAR);
    EXPECT_EQ(text.varchar_, "");

    const Value date = Value::DefaultFor(TypeId::DATE);
    EXPECT_EQ(date.type_, TypeId::DATE);
    EXPECT_EQ(date.date_, 0);

    EXPECT_TRUE(Value::DefaultFor(TypeId::INVALID).IsNull());
}

}  // namespace shilmandb
