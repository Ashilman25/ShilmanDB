#include <gtest/gtest.h>
#include "catalog/schema.hpp"
#include "executor/data_chunk.hpp"
#include "executor/executor.hpp"
#include "types/tuple.hpp"
#include "types/value.hpp"

#include <cstring>
#include <string>
#include <vector>

namespace shilmandb {

namespace {

Schema MakeTestSchema() {
    return Schema({
        Column("id",   TypeId::INTEGER),
        Column("name", TypeId::VARCHAR),
        Column("bal",  TypeId::DECIMAL),
    });
}

Tuple MakeRow(int32_t id, const std::string& name, double bal, const Schema& s) {
    std::vector<Value> vals;
    vals.emplace_back(id);
    vals.emplace_back(name);
    vals.emplace_back(bal);
    return Tuple(std::move(vals), s);
}

// Minimal Executor mock that emits `total` synthetic rows via Next()
// and exercises the default NextBatch on the base class.
class CountingMockExecutor : public Executor {
public:
    CountingMockExecutor(const Schema& out_schema, size_t total)
        : Executor(nullptr, nullptr), schema_(out_schema), total_(total) {}

    void Init() override {}
    bool Next(Tuple* tuple) override {
        if (produced_ >= total_) return false;
        *tuple = MakeRow(static_cast<int32_t>(produced_), "row", static_cast<double>(produced_), schema_);
        ++produced_;
        return true;
    }
    void Close() override {}

private:
    Schema schema_;
    size_t total_;
    size_t produced_{0};
};

}  // namespace

// --- Construction & capacity ---

TEST(DataChunkTest, ConstructorReservesCapacity) {
    DataChunk chunk(MakeTestSchema());
    EXPECT_EQ(chunk.capacity(), DataChunk::kDefaultBatchSize);
    EXPECT_EQ(chunk.size(), 0u);
    EXPECT_EQ(chunk.ColumnCount(), 3u);
    EXPECT_FALSE(chunk.IsFull());
    EXPECT_FALSE(chunk.HasSelectionVector());
}

// --- AppendTuple + MaterializeTuple round-trip ---

TEST(DataChunkTest, AppendTupleRoundTrip) {
    auto s = MakeTestSchema();
    DataChunk chunk(s);
    chunk.AppendTuple(MakeRow(1, "alice", 100.5, s));
    chunk.AppendTuple(MakeRow(2, "bob",   200.25, s));

    ASSERT_EQ(chunk.size(), 2u);
    const auto& schema = chunk.GetSchema();

    auto r0 = chunk.MaterializeTuple(0);
    EXPECT_EQ(r0.GetValue(schema, 0).integer_, 1);
    EXPECT_EQ(r0.GetValue(schema, 1).varchar_, "alice");
    EXPECT_DOUBLE_EQ(r0.GetValue(schema, 2).decimal_, 100.5);

    auto r1 = chunk.MaterializeTuple(1);
    EXPECT_EQ(r1.GetValue(schema, 0).integer_, 2);
    EXPECT_EQ(r1.GetValue(schema, 1).varchar_, "bob");
    EXPECT_DOUBLE_EQ(r1.GetValue(schema, 2).decimal_, 200.25);
}

TEST(DataChunkTest, AppendTupleIncrementsSize) {
    auto s = MakeTestSchema();
    DataChunk chunk(s, /*capacity=*/4);
    EXPECT_FALSE(chunk.IsFull());
    chunk.AppendTuple(MakeRow(1, "a", 1.0, s));
    EXPECT_EQ(chunk.size(), 1u);
    EXPECT_FALSE(chunk.IsFull());
    chunk.AppendTuple(MakeRow(2, "b", 2.0, s));
    chunk.AppendTuple(MakeRow(3, "c", 3.0, s));
    chunk.AppendTuple(MakeRow(4, "d", 4.0, s));
    EXPECT_EQ(chunk.size(), 4u);
    EXPECT_TRUE(chunk.IsFull());
}

// --- Columnar access ---

TEST(DataChunkTest, GetColumnReadsRawValues) {
    auto s = MakeTestSchema();
    DataChunk chunk(s);
    chunk.AppendTuple(MakeRow(10, "x", 1.5, s));
    chunk.AppendTuple(MakeRow(20, "y", 2.5, s));
    chunk.AppendTuple(MakeRow(30, "z", 3.5, s));

    const auto& col0 = chunk.GetColumn(0);
    const auto& col1 = chunk.GetColumn(1);
    const auto& col2 = chunk.GetColumn(2);
    ASSERT_EQ(col0.size(), 3u);
    EXPECT_EQ(col0[0].integer_, 10);
    EXPECT_EQ(col0[1].integer_, 20);
    EXPECT_EQ(col0[2].integer_, 30);
    EXPECT_EQ(col1[0].varchar_, "x");
    EXPECT_EQ(col1[1].varchar_, "y");
    EXPECT_EQ(col1[2].varchar_, "z");
    EXPECT_DOUBLE_EQ(col2[0].decimal_, 1.5);
    EXPECT_DOUBLE_EQ(col2[1].decimal_, 2.5);
    EXPECT_DOUBLE_EQ(col2[2].decimal_, 3.5);
}

// --- Selection vector semantics ---

TEST(DataChunkTest, SetSelectionVectorReshapesSize) {
    auto s = MakeTestSchema();
    DataChunk chunk(s);
    for (int32_t i = 0; i < 10; ++i) {
        chunk.AppendTuple(MakeRow(i, "r" + std::to_string(i), i * 1.0, s));
    }
    ASSERT_EQ(chunk.size(), 10u);

    chunk.SetSelectionVector({3u, 5u, 7u});
    EXPECT_TRUE(chunk.HasSelectionVector());
    EXPECT_EQ(chunk.size(), 3u);

    const auto& schema = chunk.GetSchema();
    EXPECT_EQ(chunk.MaterializeTuple(0).GetValue(schema, 0).integer_, 3);
    EXPECT_EQ(chunk.MaterializeTuple(1).GetValue(schema, 0).integer_, 5);
    EXPECT_EQ(chunk.MaterializeTuple(2).GetValue(schema, 0).integer_, 7);
}

TEST(DataChunkTest, GetColumnIgnoresSelectionVector) {
    auto s = MakeTestSchema();
    DataChunk chunk(s);
    for (int32_t i = 0; i < 5; ++i) chunk.AppendTuple(MakeRow(i, "", 0.0, s));
    chunk.SetSelectionVector({1u, 3u});

    const auto& col0 = chunk.GetColumn(0);
    EXPECT_EQ(col0.size(), 5u);  // physical count unchanged
    EXPECT_EQ(col0[0].integer_, 0);
    EXPECT_EQ(col0[4].integer_, 4);
    EXPECT_EQ(chunk.GetValue(0, 3).integer_, 3);  // physical indexing
}

TEST(DataChunkTest, FlattenCompactsInPlace) {
    auto s = MakeTestSchema();
    DataChunk chunk(s);
    for (int32_t i = 0; i < 4; ++i) {
        chunk.AppendTuple(MakeRow(i, "n" + std::to_string(i), i * 1.0, s));
    }
    chunk.SetSelectionVector({1u, 3u});
    chunk.Flatten();

    EXPECT_FALSE(chunk.HasSelectionVector());
    EXPECT_EQ(chunk.size(), 2u);
    const auto& col0 = chunk.GetColumn(0);
    ASSERT_EQ(col0.size(), 2u);
    EXPECT_EQ(col0[0].integer_, 1);
    EXPECT_EQ(col0[1].integer_, 3);
    const auto& col1 = chunk.GetColumn(1);
    EXPECT_EQ(col1[0].varchar_, "n1");
    EXPECT_EQ(col1[1].varchar_, "n3");
}

TEST(DataChunkTest, FlattenNoOpWithoutSelection) {
    auto s = MakeTestSchema();
    DataChunk chunk(s);
    chunk.AppendTuple(MakeRow(1, "a", 1.0, s));
    chunk.AppendTuple(MakeRow(2, "b", 2.0, s));
    chunk.Flatten();  // no prior SetSelectionVector — must be a no-op
    EXPECT_EQ(chunk.size(), 2u);
    EXPECT_FALSE(chunk.HasSelectionVector());
    EXPECT_EQ(chunk.GetColumn(0)[0].integer_, 1);
    EXPECT_EQ(chunk.GetColumn(0)[1].integer_, 2);
}

// --- Reset + capacity preservation ---

TEST(DataChunkTest, ResetClearsSizeAndSelection) {
    auto s = MakeTestSchema();
    DataChunk chunk(s);
    chunk.AppendTuple(MakeRow(1, "a", 1.0, s));
    chunk.AppendTuple(MakeRow(2, "b", 2.0, s));
    chunk.SetSelectionVector({0u});
    chunk.Reset();
    EXPECT_EQ(chunk.size(), 0u);
    EXPECT_FALSE(chunk.HasSelectionVector());
    EXPECT_FALSE(chunk.IsFull());
    EXPECT_EQ(chunk.capacity(), DataChunk::kDefaultBatchSize);
}

TEST(DataChunkTest, ResetPreservesCapacityAfterFillCycle) {
    auto s = MakeTestSchema();
    DataChunk chunk(s, /*capacity=*/1024);

    for (int32_t i = 0; i < 1024; ++i) chunk.AppendTuple(MakeRow(i, "", 0.0, s));
    const auto cap_after_first_fill = chunk.GetColumn(0).capacity();
    EXPECT_GE(cap_after_first_fill, 1024u);

    chunk.Reset();
    EXPECT_EQ(chunk.GetColumn(0).capacity(), cap_after_first_fill);

    // Second fill-cycle must not cause a reallocation of the underlying column.
    for (int32_t i = 0; i < 1024; ++i) chunk.AppendTuple(MakeRow(i, "", 0.0, s));
    EXPECT_EQ(chunk.GetColumn(0).capacity(), cap_after_first_fill);
}

// --- Default NextBatch (Executor base class) ---

TEST(DataChunkTest, DefaultNextBatchDrainsChild) {
    auto s = MakeTestSchema();
    CountingMockExecutor mock(s, /*total=*/1500);
    mock.Init();
    DataChunk chunk(s);  // default capacity 1024

    ASSERT_TRUE(mock.NextBatch(&chunk));
    EXPECT_EQ(chunk.size(), 1024u);

    ASSERT_TRUE(mock.NextBatch(&chunk));
    EXPECT_EQ(chunk.size(), 1500u - 1024u);  // 476

    EXPECT_FALSE(mock.NextBatch(&chunk));
    EXPECT_EQ(chunk.size(), 0u);
    mock.Close();
}

TEST(DataChunkTest, DefaultNextBatchEmptyChildReturnsFalse) {
    auto s = MakeTestSchema();
    CountingMockExecutor mock(s, /*total=*/0);
    mock.Init();
    DataChunk chunk(s);
    EXPECT_FALSE(mock.NextBatch(&chunk));
    EXPECT_EQ(chunk.size(), 0u);
    mock.Close();
}

// --- Row store and lazy decode ---

namespace {
void AppendRaw(DataChunk& chunk, const Tuple& t) { chunk.AppendRow(t.GetData(), t.GetLength()); }
}  // namespace

TEST(DataChunkTest, AppendRowDecodesNothing) {
    auto s = MakeTestSchema();
    DataChunk chunk(s);
    for (int i = 0; i < 3; ++i) AppendRaw(chunk, MakeRow(i, "n", i * 1.5, s));
    EXPECT_EQ(chunk.size(), 3u);
    EXPECT_TRUE(chunk.HasRowStore());
    for (size_t c = 0; c < 3; ++c) EXPECT_FALSE(chunk.IsDecoded(c));
}

TEST(DataChunkTest, EnsureDecodedOnEmptyChunkThenAppendRow) {
    auto s = MakeTestSchema();
    DataChunk chunk(s);
    chunk.EnsureDecoded({0});
    AppendRaw(chunk, MakeRow(7, "n", 1.5, s));
    EXPECT_FALSE(chunk.IsDecoded(0));
    chunk.EnsureDecoded({0});
    EXPECT_EQ(chunk.GetValue(0, 0).integer_, 7);
}

TEST(DataChunkTest, EnsureDecodedReadsColumns) {
    auto s = MakeTestSchema();
    DataChunk chunk(s);
    for (int i = 0; i < 3; ++i) AppendRaw(chunk, MakeRow(i, "nm" + std::to_string(i), i * 1.5, s));
    chunk.EnsureDecoded({0, 2});
    for (size_t r = 0; r < 3; ++r) {
        EXPECT_EQ(chunk.GetValue(0, r).integer_, static_cast<int32_t>(r));
        EXPECT_DOUBLE_EQ(chunk.GetValue(2, r).decimal_, r * 1.5);
    }
    EXPECT_FALSE(chunk.IsDecoded(1));
}

TEST(DataChunkTest, EnsureDecodedOnlyActiveRows) {
    auto s = MakeTestSchema();
    DataChunk chunk(s);
    for (int i = 0; i < 5; ++i) AppendRaw(chunk, MakeRow(i, "n", 0.0, s));
    chunk.SetSelectionVector({1, 3});
    chunk.EnsureDecoded({0});
    EXPECT_EQ(chunk.GetValue(0, 1).integer_, 1);
    EXPECT_EQ(chunk.GetValue(0, 3).integer_, 3);
}

TEST(DataChunkTest, EnsureDecodedTwiceIsNoOp) {
    auto s = MakeTestSchema();
    DataChunk chunk(s);
    AppendRaw(chunk, MakeRow(5, "n", 0.0, s));
    chunk.EnsureDecoded({0});
    chunk.GetMutableColumn(0)[0] = Value(int32_t(99));
    chunk.EnsureDecoded({0});
    EXPECT_EQ(chunk.GetValue(0, 0).integer_, 99);
}

TEST(DataChunkTest, EnsureDecodedVarcharEdgeCases) {
    auto s = MakeTestSchema();
    DataChunk chunk(s);
    const std::string big(300, 'x');
    AppendRaw(chunk, MakeRow(1, "", 0.0, s));
    AppendRaw(chunk, MakeRow(2, big, 0.0, s));
    chunk.EnsureDecoded({1});
    EXPECT_EQ(chunk.GetValue(1, 0).varchar_, "");
    EXPECT_EQ(chunk.GetValue(1, 1).varchar_, big);
}

TEST(DataChunkTest, MaterializeTupleFromRowStoreIsByteIdentical) {
    auto s = MakeTestSchema();
    DataChunk chunk(s);
    std::vector<Tuple> rows;
    for (int i = 0; i < 3; ++i) rows.push_back(MakeRow(i, std::string(i * 7, 'q'), i * 2.0, s));
    for (const auto& r : rows) AppendRaw(chunk, r);
    chunk.SetSelectionVector({0, 2});
    const size_t src[] = {0, 2};
    for (size_t i = 0; i < 2; ++i) {
        Tuple t = chunk.MaterializeTuple(i);
        ASSERT_EQ(t.GetLength(), rows[src[i]].GetLength());
        EXPECT_EQ(std::memcmp(t.GetData(), rows[src[i]].GetData(), t.GetLength()), 0);
    }
}

TEST(DataChunkTest, AppendValuesDecodesAll) {
    auto s = MakeTestSchema();
    DataChunk chunk(s);
    std::vector<Value> vals;
    vals.emplace_back(int32_t(7));
    vals.emplace_back(std::string("z"));
    vals.emplace_back(1.5);
    Tuple expected(vals, s);
    chunk.AppendValues(std::move(vals));
    for (size_t c = 0; c < 3; ++c) EXPECT_TRUE(chunk.IsDecoded(c));
    EXPECT_FALSE(chunk.HasRowStore());
    Tuple t = chunk.MaterializeTuple(0);
    ASSERT_EQ(t.GetLength(), expected.GetLength());
    EXPECT_EQ(std::memcmp(t.GetData(), expected.GetData(), t.GetLength()), 0);
}

TEST(DataChunkTest, EnsureDecodedOnValuesChunkIsNoOp) {
    auto s = MakeTestSchema();
    DataChunk chunk(s);
    std::vector<Value> vals;
    vals.emplace_back(int32_t(7));
    vals.emplace_back(std::string("z"));
    vals.emplace_back(1.5);
    chunk.AppendValues(std::move(vals));
    chunk.EnsureDecoded({0, 1, 2});
    EXPECT_EQ(chunk.GetValue(0, 0).integer_, 7);
    EXPECT_EQ(chunk.GetValue(1, 0).varchar_, "z");
    EXPECT_DOUBLE_EQ(chunk.GetValue(2, 0).decimal_, 1.5);
}

TEST(DataChunkTest, AppendValuesMismatchedTypeRoundTrips) {
    Schema s({Column("id", TypeId::INTEGER), Column("dt", TypeId::DATE)});
    DataChunk chunk(s);
    std::vector<Value> vals;
    vals.emplace_back(int32_t(1));
    vals.emplace_back(int32_t(0));
    Tuple expected(vals, s);
    chunk.AppendValues(std::move(vals));
    const Value& got = chunk.GetValue(1, 0);
    EXPECT_EQ(got.type_, TypeId::DATE);
    EXPECT_TRUE(got == expected.GetValue(s, 1));
}

TEST(DataChunkTest, FlattenWithRowStore) {
    auto s = MakeTestSchema();
    DataChunk chunk(s);
    std::vector<Tuple> rows;
    for (int i = 0; i < 5; ++i) rows.push_back(MakeRow(i, std::string(i * 3, 'a'), 0.0, s));
    for (const auto& r : rows) AppendRaw(chunk, r);
    chunk.EnsureDecoded({0});
    chunk.SetSelectionVector({1, 4});
    chunk.Flatten();
    EXPECT_EQ(chunk.size(), 2u);
    EXPECT_EQ(chunk.GetValue(0, 0).integer_, 1);
    EXPECT_EQ(chunk.GetValue(0, 1).integer_, 4);
    Tuple t = chunk.MaterializeTuple(1);
    ASSERT_EQ(t.GetLength(), rows[4].GetLength());
    EXPECT_EQ(std::memcmp(t.GetData(), rows[4].GetData(), t.GetLength()), 0);
}

TEST(DataChunkTest, ResetClearsRowStoreAndFlags) {
    auto s = MakeTestSchema();
    DataChunk chunk(s);
    AppendRaw(chunk, MakeRow(11, "a", 0.0, s));
    chunk.EnsureDecoded({0});
    chunk.Reset();
    EXPECT_FALSE(chunk.HasRowStore());
    EXPECT_FALSE(chunk.IsDecoded(0));
    AppendRaw(chunk, MakeRow(22, "b", 0.0, s));
    chunk.EnsureDecoded({0});
    EXPECT_EQ(chunk.GetValue(0, 0).integer_, 22);
}

TEST(DataChunkTest, DefaultNextBatchStoresBytes) {
    auto s = MakeTestSchema();
    CountingMockExecutor exec(s, 5);
    DataChunk chunk(s);
    ASSERT_TRUE(exec.NextBatch(&chunk));
    ASSERT_EQ(chunk.size(), 5u);
    EXPECT_TRUE(chunk.HasRowStore());
    EXPECT_FALSE(chunk.IsDecoded(0));

    chunk.EnsureDecoded({0, 1, 2});
    for (int32_t i = 0; i < 5; ++i) {
        auto expected = MakeRow(i, "row", static_cast<double>(i), s);
        EXPECT_EQ(chunk.GetValue(0, i).integer_, i);
        EXPECT_EQ(chunk.GetValue(1, i).ToString(), "row");
        EXPECT_DOUBLE_EQ(chunk.GetValue(2, i).decimal_, static_cast<double>(i));
        Tuple t = chunk.MaterializeTuple(i);
        ASSERT_EQ(t.GetLength(), expected.GetLength());
        EXPECT_EQ(std::memcmp(t.GetData(), expected.GetData(), t.GetLength()), 0);
    }
}

TEST(DataChunkTest, DecodeCellMatchesEnsureDecoded) {
    auto s = MakeTestSchema();
    DataChunk chunk(s);
    for (const Tuple& t : {MakeRow(1, "abc", 1.5, s), MakeRow(2, "", -2.25, s),
                           MakeRow(3, std::string(300, 'z'), 0.0, s)}) {
        chunk.AppendRow(t.GetData(), t.GetLength());
    }
    std::vector<Value> direct;
    for (size_t c = 0; c < 3; ++c)
        for (size_t r = 0; r < 3; ++r) direct.push_back(chunk.DecodeCell(c, r));
    EXPECT_FALSE(chunk.IsDecoded(0));

    chunk.EnsureDecoded({0, 1, 2});
    size_t i = 0;
    for (size_t c = 0; c < 3; ++c) {
        for (size_t r = 0; r < 3; ++r, ++i) {
            const Value& want = chunk.GetValue(c, r);
            EXPECT_EQ(direct[i].type_, want.type_);
            EXPECT_TRUE(direct[i] == want);
            EXPECT_EQ(direct[i].ToString(), want.ToString());
        }
    }
}

}  // namespace shilmandb
