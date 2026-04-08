#include <gtest/gtest.h>
#include "types/tuple.hpp"
#include "types/value.hpp"
#include "catalog/schema.hpp"

namespace shilmandb {

// --- Fixed-type round-trips ---

TEST(TupleTest, FixedTypesRoundTrip) {
    Schema schema({
        Column("a", TypeId::INTEGER),
        Column("b", TypeId::BIGINT),
        Column("c", TypeId::DECIMAL),
        Column("d", TypeId::DATE)
    });

    std::vector<Value> values = {
        Value(static_cast<int32_t>(42)),
        Value(static_cast<int64_t>(1'000'000'000'000LL)),
        Value(3.14),
        Value::MakeDate(19737)
    };

    Tuple tuple(values, schema);

    EXPECT_EQ(tuple.GetValue(schema, 0), values[0]);
    EXPECT_EQ(tuple.GetValue(schema, 1), values[1]);
    EXPECT_DOUBLE_EQ(tuple.GetValue(schema, 2).decimal_, 3.14);
    EXPECT_EQ(tuple.GetValue(schema, 3), values[3]);
}

TEST(TupleTest, GetLengthFixedOnly) {
    Schema schema({
        Column("a", TypeId::INTEGER),
        Column("b", TypeId::BIGINT)
    });

    Tuple tuple(
        {Value(static_cast<int32_t>(1)), Value(static_cast<int64_t>(2))},
        schema
    );

    // 4 (INTEGER) + 8 (BIGINT) = 12
    EXPECT_EQ(tuple.GetLength(), 12u);
    EXPECT_FALSE(tuple.IsEmpty());
}

TEST(TupleTest, DefaultConstructorIsEmpty) {
    Tuple tuple;
    EXPECT_TRUE(tuple.IsEmpty());
    EXPECT_EQ(tuple.GetLength(), 0u);
}

// --- VARCHAR round-trips ---

TEST(TupleTest, VarcharRoundTrip) {
    Schema schema({Column("s", TypeId::VARCHAR)});

    Tuple tuple({Value(std::string("hello"))}, schema);

    Value result = tuple.GetValue(schema, 0);
    EXPECT_EQ(result.type_, TypeId::VARCHAR);
    EXPECT_EQ(result.varchar_, "hello");
}

TEST(TupleTest, MixedSchemaRoundTrip) {
    Schema schema({
        Column("id", TypeId::INTEGER),
        Column("name", TypeId::VARCHAR),
        Column("score", TypeId::BIGINT),
        Column("label", TypeId::VARCHAR)
    });

    std::vector<Value> values = {
        Value(static_cast<int32_t>(99)),
        Value(std::string("alice")),
        Value(static_cast<int64_t>(500)),
        Value(std::string("engineer"))
    };

    Tuple tuple(values, schema);

    EXPECT_EQ(tuple.GetValue(schema, 0), values[0]);
    EXPECT_EQ(tuple.GetValue(schema, 1).varchar_, "alice");
    EXPECT_EQ(tuple.GetValue(schema, 2), values[2]);
    EXPECT_EQ(tuple.GetValue(schema, 3).varchar_, "engineer");
}

TEST(TupleTest, EmptyVarchar) {
    Schema schema({
        Column("a", TypeId::INTEGER),
        Column("s", TypeId::VARCHAR)
    });

    Tuple tuple(
        {Value(static_cast<int32_t>(7)), Value(std::string(""))},
        schema
    );

    EXPECT_EQ(tuple.GetValue(schema, 0).integer_, 7);
    EXPECT_EQ(tuple.GetValue(schema, 1).varchar_, "");
}

TEST(TupleTest, GetLengthWithVarchar) {
    Schema schema({
        Column("id", TypeId::INTEGER),
        Column("name", TypeId::VARCHAR)
    });

    Tuple tuple(
        {Value(static_cast<int32_t>(1)), Value(std::string("hello"))},
        schema
    );

    // fixed prefix: 4 (int) + 4 (offset ptr) = 8
    // variable region: 2 (uint16_t len) + 5 (chars) = 7
    EXPECT_EQ(tuple.GetLength(), 15u);
}

// --- Edge cases ---

TEST(TupleTest, EmptyTuple) {
    Schema schema(std::vector<Column>{});
    Tuple tuple({}, schema);
    EXPECT_TRUE(tuple.IsEmpty());
    EXPECT_EQ(tuple.GetLength(), 0u);
}

// --- SerializeTo / DeserializeFrom ---

TEST(TupleTest, SerializeDeserializeRoundTrip) {
    Schema schema({
        Column("id", TypeId::INTEGER),
        Column("name", TypeId::VARCHAR),
        Column("score", TypeId::DECIMAL)
    });

    std::vector<Value> values = {
        Value(static_cast<int32_t>(42)),
        Value(std::string("test_string")),
        Value(99.5)
    };

    Tuple original(values, schema);

    // Serialize to external buffer
    std::vector<char> buffer(original.GetLength());
    original.SerializeTo(buffer.data());

    // Deserialize into new tuple
    Tuple restored;
    restored.DeserializeFrom(buffer.data(), original.GetLength(), schema);

    EXPECT_EQ(restored.GetLength(), original.GetLength());
    EXPECT_EQ(restored.GetValue(schema, 0), values[0]);
    EXPECT_EQ(restored.GetValue(schema, 1).varchar_, "test_string");
    EXPECT_DOUBLE_EQ(restored.GetValue(schema, 2).decimal_, 99.5);
}

}  // namespace shilmandb
