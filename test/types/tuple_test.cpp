#include <gtest/gtest.h>
#include <cstring>
#include "common/exception.hpp"
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

// --- Concat ---

// The joined row built the long way: the constructor over all the values.
static Tuple JoinedByConstructor(const std::vector<Value>& l, const std::vector<Value>& r, const Schema& joined) {
    std::vector<Value> all = l;
    all.insert(all.end(), r.begin(), r.end());
    return Tuple(all, joined);
}

// Same length, same bytes, and every column reads back the same.
static void ExpectSameRow(const Tuple& got, const Tuple& want, const Schema& schema) {
    ASSERT_EQ(got.GetLength(), want.GetLength());
    if (want.GetLength() > 0) EXPECT_EQ(std::memcmp(got.GetData(), want.GetData(), want.GetLength()), 0);
    for (uint32_t i = 0; i < schema.GetColumnCount(); ++i) EXPECT_EQ(got.GetValue(schema, i), want.GetValue(schema, i));
}

// The left schema's columns, then the right schema's.
static Schema JoinSchemas(const Schema& l, const Schema& r) {
    std::vector<Column> cols = l.GetColumns();
    cols.insert(cols.end(), r.GetColumns().begin(), r.GetColumns().end());
    return Schema(cols);
}

// Concat of the two rows must match the constructor over all the values.
static void ExpectConcatMatches(const Schema& ls, const Schema& rs,
                                const std::vector<Value>& lv, const std::vector<Value>& rv) {
    Schema joined = JoinSchemas(ls, rs);
    ExpectSameRow(Tuple::Concat(Tuple(lv, ls), ls, Tuple(rv, rs), rs), JoinedByConstructor(lv, rv, joined), joined);
}

TEST(TupleTest, ConcatWorkedExample) {
    Schema ls({Column("a", TypeId::INTEGER), Column("s", TypeId::VARCHAR)});
    Schema rs({Column("t", TypeId::VARCHAR), Column("d", TypeId::DATE)});
    Schema joined({Column("a", TypeId::INTEGER), Column("s", TypeId::VARCHAR),
                   Column("t", TypeId::VARCHAR), Column("d", TypeId::DATE)});
    std::vector<Value> lv = {Value(int32_t{1}), Value(std::string("ab"))};
    std::vector<Value> rv = {Value(std::string("xyz")), Value::MakeDate(19737)};
    Tuple got = Tuple::Concat(Tuple(lv, ls), ls, Tuple(rv, rs), rs);
    EXPECT_EQ(got.GetLength(), 25u);
    uint32_t left_str = 0, right_str = 0;
    std::memcpy(&left_str, got.GetData() + 4, 4);
    std::memcpy(&right_str, got.GetData() + 8, 4);
    EXPECT_EQ(left_str, 16u);
    EXPECT_EQ(right_str, 20u);
    ExpectSameRow(got, JoinedByConstructor(lv, rv, joined), joined);
}

TEST(TupleTest, ConcatMixedTypesBothSides) {
    Schema ls({Column("a", TypeId::INTEGER), Column("b", TypeId::VARCHAR),
               Column("c", TypeId::DATE), Column("d", TypeId::VARCHAR)});
    Schema rs({Column("e", TypeId::BIGINT), Column("f", TypeId::VARCHAR), Column("g", TypeId::DECIMAL)});
    std::vector<Value> lv = {Value(int32_t{7}), Value(std::string("ab")), Value::MakeDate(19737), Value(std::string(""))};
    std::vector<Value> rv = {Value(int64_t{1'000'000'000'000}), Value(std::string("xyz")), Value(3.25)};
    ExpectConcatMatches(ls, rs, lv, rv);
}

TEST(TupleTest, ConcatVarcharOnlyOnLeft) {
    Schema ls({Column("a", TypeId::INTEGER), Column("s", TypeId::VARCHAR)});
    Schema rs({Column("b", TypeId::BIGINT), Column("c", TypeId::DECIMAL)});
    std::vector<Value> lv = {Value(int32_t{1}), Value(std::string("left"))};
    std::vector<Value> rv = {Value(int64_t{2}), Value(0.5)};
    ExpectConcatMatches(ls, rs, lv, rv);
}

TEST(TupleTest, ConcatVarcharOnlyOnRight) {
    Schema ls({Column("a", TypeId::INTEGER), Column("b", TypeId::DATE)});
    Schema rs({Column("s", TypeId::VARCHAR), Column("c", TypeId::INTEGER)});
    std::vector<Value> lv = {Value(int32_t{1}), Value::MakeDate(100)};
    std::vector<Value> rv = {Value(std::string("right")), Value(int32_t{3})};
    ExpectConcatMatches(ls, rs, lv, rv);
}

TEST(TupleTest, ConcatNoVarchar) {
    Schema ls({Column("a", TypeId::INTEGER), Column("b", TypeId::BIGINT)});
    Schema rs({Column("c", TypeId::DECIMAL), Column("d", TypeId::DATE)});
    std::vector<Value> lv = {Value(int32_t{1}), Value(int64_t{2})};
    std::vector<Value> rv = {Value(1.5), Value::MakeDate(9)};
    ExpectConcatMatches(ls, rs, lv, rv);
}

TEST(TupleTest, ConcatVarcharOnlyBothSides) {
    Schema ls({Column("a", TypeId::VARCHAR), Column("b", TypeId::VARCHAR)});
    Schema rs({Column("c", TypeId::VARCHAR)});
    std::vector<Value> lv = {Value(std::string("a")), Value(std::string(""))};
    std::vector<Value> rv = {Value(std::string("bcd"))};
    ExpectConcatMatches(ls, rs, lv, rv);
}

TEST(TupleTest, ConcatLongStrings) {
    Schema ls({Column("a", TypeId::VARCHAR), Column("b", TypeId::VARCHAR), Column("c", TypeId::VARCHAR)});
    Schema rs({Column("d", TypeId::VARCHAR), Column("e", TypeId::INTEGER), Column("f", TypeId::VARCHAR)});
    std::vector<Value> lv = {Value(std::string(40'000, 'x')), Value(std::string(40'000, 'w')), Value(std::string("end"))};
    std::vector<Value> rv = {Value(std::string(40'000, 'y')), Value(int32_t{5}), Value(std::string(40'000, 'z'))};
    Schema joined = JoinSchemas(ls, rs);
    Tuple got = Tuple::Concat(Tuple(lv, ls), ls, Tuple(rv, rs), rs);

    // The left third string and both right strings sit past byte 65,535.
    auto slot = [&](uint32_t col) {
        uint32_t offset = 0;
        std::memcpy(&offset, got.GetData() + joined.GetOffset(col), 4);
        return offset;
    };
    EXPECT_GT(slot(2), 65535u);
    EXPECT_GT(slot(3), 65535u);
    EXPECT_GT(slot(5), 65535u);
    ExpectSameRow(got, JoinedByConstructor(lv, rv, joined), joined);
}

TEST(TupleTest, ConcatChainsThreeRows) {
    Schema a({Column("a1", TypeId::INTEGER), Column("a2", TypeId::VARCHAR)});
    Schema b({Column("b1", TypeId::VARCHAR), Column("b2", TypeId::DATE)});
    Schema c({Column("c1", TypeId::DECIMAL), Column("c2", TypeId::VARCHAR)});
    Schema ab = JoinSchemas(a, b);
    Schema abc = JoinSchemas(ab, c);

    std::vector<Value> av = {Value(int32_t{1}), Value(std::string("a1"))};
    std::vector<Value> bv = {Value(std::string("bb")), Value::MakeDate(7)};
    std::vector<Value> cv = {Value(2.5), Value(std::string("ccc"))};

    Tuple got = Tuple::Concat(Tuple::Concat(Tuple(av, a), a, Tuple(bv, b), b), ab, Tuple(cv, c), c);

    std::vector<Value> all = av;
    all.insert(all.end(), bv.begin(), bv.end());
    all.insert(all.end(), cv.begin(), cv.end());
    ExpectSameRow(got, Tuple(all, abc), abc);
}

TEST(TupleTest, ConcatZeroColumnSide) {
    Schema empty(std::vector<Column>{});
    Tuple te(std::vector<Value>{}, empty);
    Schema xs({Column("a", TypeId::INTEGER), Column("s", TypeId::VARCHAR)});
    Tuple tx({Value(int32_t{4}), Value(std::string("q"))}, xs);

    ExpectSameRow(Tuple::Concat(te, empty, tx, xs), tx, xs);
    ExpectSameRow(Tuple::Concat(tx, xs, te, empty), tx, xs);
}

TEST(TupleTest, ConcatRowShorterThanFixedPartThrows) {
    Schema xs({Column("a", TypeId::INTEGER), Column("s", TypeId::VARCHAR)});
    Tuple tx({Value(int32_t{4}), Value(std::string("q"))}, xs);

    EXPECT_THROW(Tuple::Concat(Tuple{}, xs, tx, xs), DatabaseException);
    EXPECT_THROW(Tuple::Concat(tx, xs, Tuple{}, xs), DatabaseException);
}

TEST(TupleTest, ConcatRowsFromStorage) {
    Schema ls({Column("a", TypeId::INTEGER), Column("b", TypeId::VARCHAR),
               Column("c", TypeId::DATE), Column("d", TypeId::VARCHAR)});
    Schema rs({Column("e", TypeId::BIGINT), Column("f", TypeId::VARCHAR), Column("g", TypeId::DECIMAL)});
    Schema joined = JoinSchemas(ls, rs);
    std::vector<Value> lv = {Value(int32_t{7}), Value(std::string("ab")), Value::MakeDate(19737), Value(std::string(""))};
    std::vector<Value> rv = {Value(int64_t{1'000'000'000'000}), Value(std::string("xyz")), Value(3.25)};

    // Send each row through a byte buffer, the way a page stores it.
    auto read_back = [](const Tuple& t, const Schema& schema) {
        std::vector<char> buffer(t.GetLength());
        t.SerializeTo(buffer.data());
        Tuple restored;
        restored.DeserializeFrom(buffer.data(), t.GetLength(), schema);
        return restored;
    };
    Tuple left = read_back(Tuple(lv, ls), ls);
    Tuple right = read_back(Tuple(rv, rs), rs);

    ExpectSameRow(Tuple::Concat(left, ls, right, rs), JoinedByConstructor(lv, rv, joined), joined);
}

TEST(TupleTest, SchemaVarcharOffsets) {
    Schema mixed({Column("a", TypeId::INTEGER), Column("b", TypeId::VARCHAR),
                  Column("c", TypeId::BIGINT), Column("d", TypeId::VARCHAR)});
    EXPECT_EQ(mixed.GetVarcharOffsets(), (std::vector<uint32_t>{4, 16}));

    Schema none({Column("a", TypeId::INTEGER), Column("b", TypeId::DATE)});
    EXPECT_TRUE(none.GetVarcharOffsets().empty());
}

}  // namespace shilmandb
