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

// Fills columns 0 and 2 of four rows: id = 10 * row, bal = row + 0.5.
void FillIdAndBal(DataChunk& chunk) {
    chunk.StartColumnFill(4, {0, 2});
    auto& ids = chunk.GetMutableColumn(0);
    auto& bals = chunk.GetMutableColumn(2);
    for (size_t r = 0; r < 4; ++r) {
        ids[r] = Value(static_cast<int32_t>(r * 10));
        bals[r] = Value(static_cast<double>(r) + 0.5);
    }
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

}  // namespace shilmandb
