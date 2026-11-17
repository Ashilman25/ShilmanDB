#include <gtest/gtest.h>
#include "catalog/schema.hpp"
#include "executor/data_chunk.hpp"
#include "types/tuple.hpp"
#include "types/value.hpp"

#include <string>
#include <vector>

namespace shilmandb {

namespace {

// An integer, a string and a decimal.
Schema MakeSchema() {
    return Schema({
        Column("id",   TypeId::INTEGER),
        Column("name", TypeId::VARCHAR),
        Column("bal",  TypeId::DECIMAL),
    });
}

// Fills columns 0 and 2 of four rows.
// id is 10 times the row, bal is the row plus 0.5.
void FillIdAndBal(DataChunk& chunk) {
    chunk.StartColumnFill(4, {0, 2});
    auto& ids = chunk.GetMutableColumn(0);
    auto& bals = chunk.GetMutableColumn(2);
    for (size_t r = 0; r < 4; ++r) {
        ids[r] = Value(static_cast<int32_t>(r * 10));
        bals[r] = Value(static_cast<double>(r) + 0.5);
    }
}

// One row of the schema as a tuple.
Tuple SevenRow(const Schema& schema) {
    std::vector<Value> values;
    values.emplace_back(int32_t{7});
    values.emplace_back(std::string("seven"));
    values.emplace_back(7.25);
    return Tuple(std::move(values), schema);
}

}  // namespace

TEST(DataChunkColumnFillTest, ColumnFillMaterializesDefaults) {
    const Schema schema = MakeSchema();
    DataChunk chunk(schema);
    FillIdAndBal(chunk);

    EXPECT_EQ(chunk.size(), 4u);
    EXPECT_FALSE(chunk.HasRowStore());
    EXPECT_TRUE(chunk.IsDecoded(0));
    EXPECT_FALSE(chunk.IsDecoded(1));
    EXPECT_TRUE(chunk.IsDecoded(2));

    for (size_t r = 0; r < 4; ++r) {
        const Tuple t = chunk.MaterializeTuple(r);
        EXPECT_EQ(t.GetValue(schema, 0).integer_, static_cast<int32_t>(r * 10));
        EXPECT_EQ(t.GetValue(schema, 1).type_, TypeId::VARCHAR);
        EXPECT_EQ(t.GetValue(schema, 1).varchar_, "");
        EXPECT_DOUBLE_EQ(t.GetValue(schema, 2).decimal_, static_cast<double>(r) + 0.5);
    }
}

TEST(DataChunkColumnFillTest, ColumnFillWithSelection) {
    const Schema schema = MakeSchema();
    DataChunk chunk(schema);
    FillIdAndBal(chunk);
    chunk.SetSelectionVector({1, 3});

    ASSERT_EQ(chunk.size(), 2u);
    const Tuple first = chunk.MaterializeTuple(0);
    const Tuple second = chunk.MaterializeTuple(1);
    EXPECT_EQ(first.GetValue(schema, 0).integer_, 10);
    EXPECT_DOUBLE_EQ(first.GetValue(schema, 2).decimal_, 1.5);
    EXPECT_EQ(second.GetValue(schema, 0).integer_, 30);
    EXPECT_DOUBLE_EQ(second.GetValue(schema, 2).decimal_, 3.5);
    EXPECT_EQ(second.GetValue(schema, 1).varchar_, "");
}

TEST(DataChunkColumnFillTest, ResetClearsColumnFill) {
    const Schema schema = MakeSchema();
    DataChunk chunk(schema);
    FillIdAndBal(chunk);
    chunk.Reset();

    EXPECT_EQ(chunk.size(), 0u);
    EXPECT_FALSE(chunk.IsDecoded(0));
    EXPECT_FALSE(chunk.IsDecoded(2));

    // After a reset the chunk takes row bytes again.
    std::vector<Value> values;
    values.emplace_back(int32_t{7});
    values.emplace_back(std::string("seven"));
    values.emplace_back(7.25);
    const Tuple row(std::move(values), schema);
    chunk.AppendRow(row.GetData(), row.GetLength());
    ASSERT_EQ(chunk.size(), 1u);
    EXPECT_TRUE(chunk.HasRowStore());
    const Tuple back = chunk.MaterializeTuple(0);
    EXPECT_EQ(back.GetValue(schema, 1).varchar_, "seven");
}

TEST(DataChunkColumnFillTest, FlattenKeepsSelectedRowsOfFilledColumns) {
    const Schema schema = MakeSchema();
    DataChunk chunk(schema);
    FillIdAndBal(chunk);
    chunk.SetSelectionVector({1, 3});
    chunk.Flatten();

    ASSERT_EQ(chunk.size(), 2u);
    EXPECT_FALSE(chunk.HasSelectionVector());
    EXPECT_FALSE(chunk.IsDecoded(1));
    EXPECT_EQ(chunk.GetValue(0, 0).integer_, 10);
    EXPECT_EQ(chunk.GetValue(0, 1).integer_, 30);
    EXPECT_DOUBLE_EQ(chunk.GetValue(2, 0).decimal_, 1.5);
    EXPECT_DOUBLE_EQ(chunk.GetValue(2, 1).decimal_, 3.5);
    EXPECT_EQ(chunk.MaterializeTuple(1).GetValue(schema, 1).varchar_, "");
}

#ifndef NDEBUG
TEST(DataChunkColumnFillDeathTest, NoRowsTuplesOrValuesAfterColumnFill) {
    const Schema schema = MakeSchema();
    const Tuple row = SevenRow(schema);

    DataChunk rows(schema);
    FillIdAndBal(rows);
    EXPECT_DEATH(rows.AppendRow(row.GetData(), row.GetLength()), "AppendRow on a column-filled chunk");

    DataChunk tuples(schema);
    FillIdAndBal(tuples);
    EXPECT_DEATH(tuples.AppendTuple(row), "AppendTuple on a column-filled chunk");

    DataChunk values(schema);
    FillIdAndBal(values);
    std::vector<Value> one;
    one.emplace_back(int32_t{1});
    one.emplace_back(std::string("one"));
    one.emplace_back(1.5);
    EXPECT_DEATH(values.AppendValues(std::move(one)), "AppendValues on a column-filled chunk");
}

TEST(DataChunkColumnFillDeathTest, NoColumnFillOnAFilledChunk) {
    const Schema schema = MakeSchema();

    DataChunk by_rows(schema);
    const Tuple row = SevenRow(schema);
    by_rows.AppendRow(row.GetData(), row.GetLength());
    EXPECT_DEATH(by_rows.StartColumnFill(1, {0}), "StartColumnFill needs an empty chunk");

    DataChunk by_values(schema);
    std::vector<Value> one;
    one.emplace_back(int32_t{1});
    one.emplace_back(std::string("one"));
    one.emplace_back(1.5);
    by_values.AppendValues(std::move(one));
    EXPECT_DEATH(by_values.StartColumnFill(1, {0}), "StartColumnFill needs an empty chunk");

    // A zero-row fill leaves the chunk empty; only the flag catches a second fill.
    DataChunk twice(schema);
    twice.StartColumnFill(0, {0});
    EXPECT_DEATH(twice.StartColumnFill(0, {0}), "StartColumnFill on a chunk already filled");
}

TEST(DataChunkColumnFillDeathTest, UnfilledColumnCantBeRead) {
    const Schema schema = MakeSchema();
    DataChunk chunk(schema);
    FillIdAndBal(chunk);
    EXPECT_DEATH(chunk.EnsureDecoded({1}), "EnsureDecoded on a column the column fill didn't fill");
    EXPECT_DEATH((void)chunk.GetValue(1, 0), "column is not decoded");
}
#endif

}  // namespace shilmandb
