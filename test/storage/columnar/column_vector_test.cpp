#include <gtest/gtest.h>
#include "storage/columnar/column_vector.hpp"

namespace shilmandb {

TEST(ColumnVectorTest, ColumnVectorNullsAndStrings) {
    ColumnVector col(ColumnType::Text());
    EXPECT_EQ(col.Kind(), PhysicalKind::STRING);

    col.AppendString("a");
    col.AppendNull();
    col.AppendString("ccc");

    ASSERT_EQ(col.Size(), 3u);
    EXPECT_FALSE(col.IsNull(0));
    EXPECT_TRUE(col.IsNull(1));
    EXPECT_FALSE(col.IsNull(2));
    EXPECT_EQ(col.NullCount(), 1u);
    EXPECT_EQ(col.GetString(0), "a");
    EXPECT_EQ(col.GetString(1), "");
    EXPECT_EQ(col.GetString(2), "ccc");
}

TEST(ColumnVectorTest, EveryKindHoldsValuesAndZeroForNull) {
    ColumnVector ints(ColumnType::BigInt());
    ints.AppendInt(-7);
    ints.AppendNull();
    EXPECT_EQ(ints.Kind(), PhysicalKind::INT64);
    EXPECT_EQ(ints.GetInt(0), -7);
    EXPECT_EQ(ints.GetInt(1), 0);

    ColumnVector wides(ColumnType::Numeric(38, 0));
    const __int128 big = static_cast<__int128>(1) << 100;
    wides.AppendWide(big);
    wides.AppendNull();
    EXPECT_EQ(wides.Kind(), PhysicalKind::INT128);
    EXPECT_TRUE(wides.GetWide(0) == big);
    EXPECT_TRUE(wides.GetWide(1) == 0);

    ColumnVector doubles(ColumnType::Double());
    doubles.AppendDouble(2.5);
    doubles.AppendNull();
    EXPECT_EQ(doubles.Kind(), PhysicalKind::DOUBLE);
    EXPECT_EQ(doubles.GetDouble(0), 2.5);
    EXPECT_EQ(doubles.GetDouble(1), 0.0);
    EXPECT_EQ(doubles.NullCount(), 1u);
}

TEST(ColumnVectorTest, ClearAndReserve) {
    ColumnVector col(ColumnType::Varchar(10));
    col.Reserve(100);
    EXPECT_EQ(col.Size(), 0u);

    col.AppendString("first");
    col.AppendNull();
    col.Clear();
    EXPECT_EQ(col.Size(), 0u);
    EXPECT_EQ(col.NullCount(), 0u);
    EXPECT_EQ(col.Type(), ColumnType::Varchar(10));

    col.AppendString("second");
    col.AppendString("");
    ASSERT_EQ(col.Size(), 2u);
    EXPECT_EQ(col.GetString(0), "second");
    EXPECT_EQ(col.GetString(1), "");
    EXPECT_FALSE(col.IsNull(1));
}

}  // namespace shilmandb
