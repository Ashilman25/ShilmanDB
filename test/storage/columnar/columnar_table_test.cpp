#include <gtest/gtest.h>
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "common/exception.hpp"
#include "storage/columnar/columnar_table.hpp"
#include "storage/disk_manager.hpp"
#include "storage/page.hpp"
#include "types/value.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

namespace shilmandb {
namespace {

constexpr uint32_t kGroupRows[] = {kRowGroupSize, kRowGroupSize, 10};
constexpr size_t kGroups = 3;
constexpr uint32_t kBigInt = 0, kNumeric = 1, kDate = 2, kText = 3;
const char* const kWords[] = {"alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"};

// Compares two vectors row by row; doubles by their bits.
::testing::AssertionResult SameValues(const ColumnVector& a, const ColumnVector& b) {
    if (a.Size() != b.Size()) {
        return ::testing::AssertionFailure() << "sizes " << a.Size() << " vs " << b.Size();
    }
    for (size_t i = 0; i < a.Size(); ++i) {
        if (a.IsNull(i) != b.IsNull(i)) return ::testing::AssertionFailure() << "NULL flag at row " << i;
        if (a.IsNull(i)) continue;
        bool same = true;
        switch (a.Kind()) {
            case PhysicalKind::INT64:  same = a.GetInt(i) == b.GetInt(i); break;
            case PhysicalKind::INT128: same = a.GetWide(i) == b.GetWide(i); break;
            case PhysicalKind::DOUBLE: {
                const double x = a.GetDouble(i), y = b.GetDouble(i);
                same = std::memcmp(&x, &y, sizeof(double)) == 0;
                break;
            }
            case PhysicalKind::STRING: same = a.GetString(i) == b.GetString(i); break;
        }
        if (!same) return ::testing::AssertionFailure() << "value at row " << i;
    }
    return ::testing::AssertionSuccess();
}

// Appends row i of src to out.
void AppendRow(const ColumnVector& src, size_t i, ColumnVector& out) {
    if (src.IsNull(i)) {
        out.AppendNull();
        return;
    }
    switch (src.Kind()) {
        case PhysicalKind::INT64:  out.AppendInt(src.GetInt(i)); break;
        case PhysicalKind::INT128: out.AppendWide(src.GetWide(i)); break;
        case PhysicalKind::DOUBLE: out.AppendDouble(src.GetDouble(i)); break;
        case PhysicalKind::STRING: out.AppendString(src.GetString(i)); break;
    }
}

class ColumnarTableTest : public ::testing::Test {
protected:
    std::string path_;
    std::unique_ptr<DiskManager> disk_;
    std::vector<std::vector<ColumnVector>> source_;  // per group, per column
    ColumnarTableMeta meta_;

    void SetUp() override {
        path_ = (std::filesystem::temp_directory_path() /
                 ("shilmandb_coltable_" + std::to_string(::getpid()) + ".db"))
                    .string();
        std::filesystem::remove(path_);
        disk_ = std::make_unique<DiskManager>(path_);

        meta_.oid = 1;
        meta_.name = "orders";
        meta_.columns = {{"o_id", 1, ColumnType::BigInt(), false},
                         {"o_price", 2, ColumnType::Numeric(15, 2), false},
                         {"o_date", 3, ColumnType::Date(), false},
                         {"o_word", 4, ColumnType::Text(), false}};

        std::mt19937_64 rng(42);
        std::uniform_int_distribution<int64_t> any_int(INT64_MIN / 2, INT64_MAX / 2);
        std::uniform_int_distribution<int64_t> cents(-500000, 10000000);
        std::uniform_int_distribution<int64_t> days(8000, 10500);
        std::uniform_int_distribution<int> word(0, 6);

        for (size_t g = 0; g < kGroups; ++g) {
            std::vector<ColumnVector> cols;
            for (const ColumnMeta& c : meta_.columns) cols.emplace_back(c.type);
            for (uint32_t r = 0; r < kGroupRows[g]; ++r) {
                cols[kBigInt].AppendInt(any_int(rng));
                cols[kNumeric].AppendInt(cents(rng));
                cols[kDate].AppendInt(days(rng));
                cols[kText].AppendString(kWords[word(rng)]);
            }
            RowGroupMeta group;
            group.row_count = kGroupRows[g];
            for (const ColumnVector& col : cols) group.chunks.push_back(WriteChunk(disk_.get(), col));
            meta_.row_count += kGroupRows[g];
            meta_.row_groups.push_back(std::move(group));
            source_.push_back(std::move(cols));
        }
    }

    void TearDown() override {
        disk_.reset();
        std::filesystem::remove(path_);
    }

    std::unique_ptr<BufferPoolManager> MakePool(size_t frames) {
        return std::make_unique<BufferPoolManager>(frames, disk_.get(), std::make_unique<LRUEvictionPolicy>(frames));
    }

    ColumnVector Read(const ColumnarTable& table, size_t group, uint32_t column) {
        ColumnVector out(meta_.columns[column].type);
        table.ReadChunk(group, column, out);
        return out;
    }

    // The page id of a chunk's first data page, after its dictionary pages.
    page_id_t FirstDataPage(size_t group, uint32_t column) const {
        const ChunkMeta& chunk = meta_.row_groups[group].chunks[column];
        EXPECT_GT(chunk.page_count, chunk.dict_page_count);
        return chunk.first_page + chunk.dict_page_count;
    }

    // Flips one bit of a chunk's first data page on disk.
    void CorruptFirstDataPage(size_t group, uint32_t column) {
        const page_id_t pid = FirstDataPage(group, column);
        std::vector<char> page(PAGE_SIZE);
        disk_->ReadPage(pid, page.data());
        page[PAGE_SIZE / 2] ^= 0x10;
        disk_->WritePage(pid, page.data());
    }
};

TEST_F(ColumnarTableTest, ReadsEveryValue) {
    auto bpm = MakePool(64);
    const ColumnarTable table(bpm.get(), meta_);
    ASSERT_EQ(table.RowGroupCount(), kGroups);
    for (size_t g = 0; g < kGroups; ++g) {
        EXPECT_EQ(table.RowGroupRows(g), kGroupRows[g]);
        for (uint32_t c = 0; c < meta_.columns.size(); ++c) {
            EXPECT_TRUE(SameValues(Read(table, g, c), source_[g][c])) << "group " << g << " column " << c;
        }
    }
}

TEST_F(ColumnarTableTest, ReadValueAtRandomPositions) {
    auto bpm = MakePool(64);
    const ColumnarTable table(bpm.get(), meta_);
    std::mt19937 rng(7);
    for (int n = 0; n < 1000; ++n) {
        const auto g = static_cast<uint32_t>(rng() % kGroups);
        const auto offset = static_cast<uint16_t>(rng() % kGroupRows[g]);
        const auto c = static_cast<uint32_t>(rng() % meta_.columns.size());
        const RowPosition pos{g, offset};
        EXPECT_EQ(FromRid(ToRid(pos)).row_group, g);
        EXPECT_EQ(FromRid(ToRid(pos)).offset, offset);

        ColumnVector got(meta_.columns[c].type);
        table.ReadValue(pos, c, got);
        ColumnVector want(meta_.columns[c].type);
        AppendRow(source_[g][c], offset, want);
        ASSERT_TRUE(SameValues(got, want)) << "group " << g << " offset " << offset << " column " << c;
    }
}

TEST_F(ColumnarTableTest, OutOfRangeThrows) {
    auto bpm = MakePool(64);
    const ColumnarTable table(bpm.get(), meta_);
    ColumnVector out(ColumnType::BigInt());
    EXPECT_THROW(table.ReadChunk(3, 0, out), DatabaseException);
    EXPECT_THROW(table.ReadChunk(0, 4, out), DatabaseException);
    EXPECT_THROW(table.ReadValue(RowPosition{2, 10}, 0, out), DatabaseException);
    EXPECT_THROW((void)table.GroupRanges(3), DatabaseException);
}

TEST_F(ColumnarTableTest, GroupRangesAreExecutorValues) {
    auto bpm = MakePool(64);
    const ColumnarTable table(bpm.get(), meta_);
    for (size_t g = 0; g < kGroups; ++g) {
        const auto ranges = table.GroupRanges(g);
        ASSERT_EQ(ranges.size(), 4u);

        const ColumnVector& prices = source_[g][kNumeric];
        int64_t lo = prices.GetInt(0), hi = prices.GetInt(0);
        for (size_t i = 1; i < prices.Size(); ++i) {
            lo = std::min(lo, prices.GetInt(i));
            hi = std::max(hi, prices.GetInt(i));
        }
        const MinMax& price = ranges.at("o_price");
        EXPECT_EQ(price.min.type_, TypeId::DECIMAL);
        EXPECT_DOUBLE_EQ(price.min.decimal_, static_cast<double>(lo) / 100);
        EXPECT_DOUBLE_EQ(price.max.decimal_, static_cast<double>(hi) / 100);

        EXPECT_EQ(ranges.at("o_id").min.type_, TypeId::BIGINT);
        EXPECT_EQ(ranges.at("o_date").min.type_, TypeId::DATE);

        const ColumnVector& words = source_[g][kText];
        std::string first(words.GetString(0)), last(words.GetString(0));
        for (size_t i = 1; i < words.Size(); ++i) {
            first = std::min(first, std::string(words.GetString(i)));
            last = std::max(last, std::string(words.GetString(i)));
        }
        EXPECT_EQ(ranges.at("o_word").min, Value(first));
        EXPECT_EQ(ranges.at("o_word").max, Value(last));
    }
}

TEST_F(ColumnarTableTest, BadChecksumThrows) {
    CorruptFirstDataPage(0, kBigInt);
    auto bpm = MakePool(64);
    bpm->DropAllPages();
    const ColumnarTable table(bpm.get(), meta_);
    ColumnVector out(ColumnType::BigInt());
    try {
        table.ReadChunk(0, kBigInt, out);
        FAIL() << "expected a checksum error";
    } catch (const DatabaseException& e) {
        const std::string want = "orders: page " + std::to_string(FirstDataPage(0, kBigInt)) + " failed its checksum";
        EXPECT_EQ(std::string(e.what()), want);
    }
}

TEST_F(ColumnarTableTest, VerifiedOncePerRead) {
    auto bpm = MakePool(64);
    const ColumnarTable table(bpm.get(), meta_);
    // The 10-row group's chunks are one page each, so the pool keeps them all.
    const ChunkMeta& chunk = meta_.row_groups[2].chunks[kBigInt];
    ASSERT_EQ(chunk.page_count, 1u);
    ASSERT_TRUE(chunk.encoding == Encoding::BITPACK || chunk.encoding == Encoding::PLAIN);
    EXPECT_TRUE(SameValues(Read(table, 2, kBigInt), source_[2][kBigInt]));

    // Change the cached frame inside its values. A second check would now fail.
    const page_id_t pid = FirstDataPage(2, kBigInt);
    Page* page = bpm->FetchPage(pid);
    ASSERT_NE(page, nullptr);
    page->GetData()[kPageHeaderSize + 8] ^= 0x01;
    (void)bpm->UnpinPage(pid, false);
    try {
        const ColumnVector served = Read(table, 2, kBigInt);
        EXPECT_EQ(served.Size(), 10u);
        EXPECT_FALSE(SameValues(served, source_[2][kBigInt])) << "the changed frame was not the one served";
    } catch (const DatabaseException& e) {
        FAIL() << "a verified frame was checked again: " << e.what();
    }

    CorruptFirstDataPage(2, kBigInt);
    bpm->DropAllPages();
    ColumnVector out(ColumnType::BigInt());
    EXPECT_THROW(table.ReadChunk(2, kBigInt, out), DatabaseException);
}

TEST_F(ColumnarTableTest, OnePinAtATime) {
    auto bpm = MakePool(1);
    const ColumnarTable table(bpm.get(), meta_);
    for (size_t g = 0; g < kGroups; ++g) {
        for (uint32_t c = 0; c < meta_.columns.size(); ++c) {
            EXPECT_TRUE(SameValues(Read(table, g, c), source_[g][c])) << "group " << g << " column " << c;
        }
    }
    ColumnVector word(ColumnType::Text());
    table.ReadValue(RowPosition{1, 1234}, kText, word);
    EXPECT_EQ(word.GetString(0), source_[1][kText].GetString(1234));
}

TEST_F(ColumnarTableTest, BadChecksumUnpinsFirst) {
    CorruptFirstDataPage(2, kBigInt);
    auto bpm = MakePool(1);
    const ColumnarTable table(bpm.get(), meta_);
    ColumnVector bad(ColumnType::BigInt());
    EXPECT_THROW(table.ReadChunk(2, kBigInt, bad), DatabaseException);
    // A frame still pinned by the failed read would leave no frame for this one.
    EXPECT_TRUE(SameValues(Read(table, 2, kDate), source_[2][kDate]));
}

TEST_F(ColumnarTableTest, ConstantChunkNeedsNoPages) {
    ColumnVector sevens(ColumnType::BigInt());
    for (int i = 0; i < 1000; ++i) sevens.AppendInt(7);
    const page_id_t before = disk_->AllocatePages(0);
    const ChunkMeta chunk = WriteChunk(disk_.get(), sevens);
    EXPECT_EQ(disk_->AllocatePages(0), before);
    EXPECT_EQ(chunk.encoding, Encoding::CONSTANT);
    EXPECT_EQ(chunk.page_count, 0u);
    EXPECT_EQ(chunk.first_page, INVALID_PAGE_ID);

    ColumnarTableMeta meta;
    meta.name = "sevens";
    meta.columns = {{"x", 1, ColumnType::BigInt(), false}};
    meta.row_count = 1000;
    meta.row_groups.push_back(RowGroupMeta{1000, {chunk}});
    auto bpm = MakePool(1);
    const ColumnarTable table(bpm.get(), meta);
    ColumnVector out(ColumnType::BigInt());
    table.ReadChunk(0, 0, out);
    EXPECT_TRUE(SameValues(out, sevens));
    ColumnVector one(ColumnType::BigInt());
    table.ReadValue(RowPosition{0, 999}, 0, one);
    EXPECT_EQ(one.GetInt(0), 7);
    EXPECT_EQ(table.GroupRanges(0).at("x").min, Value(int64_t{7}));
}

// A DOUBLE chunk holding NaN has no range, so nothing can skip its group.
TEST_F(ColumnarTableTest, NaNChunkHasNoRange) {
    ColumnVector doubles(ColumnType::Double());
    for (double x : {10.0, 20.0, std::nan("")}) doubles.AppendDouble(x);
    const ColumnVector ids = [] {
        ColumnVector v(ColumnType::BigInt());
        for (int64_t x : {1, 2, 3}) v.AppendInt(x);
        return v;
    }();

    ColumnarTableMeta meta;
    meta.name = "readings";
    meta.columns = {{"r_value", 1, ColumnType::Double(), false}, {"r_id", 2, ColumnType::BigInt(), false}};
    meta.row_count = 3;
    meta.row_groups.push_back(RowGroupMeta{3, {WriteChunk(disk_.get(), doubles), WriteChunk(disk_.get(), ids)}});
    auto bpm = MakePool(4);
    const ColumnarTable table(bpm.get(), meta);
    const auto ranges = table.GroupRanges(0);
    EXPECT_EQ(ranges.count("r_value"), 0u);
    ASSERT_EQ(ranges.count("r_id"), 1u);
    EXPECT_EQ(ranges.at("r_id").max, Value(int64_t{3}));

    ColumnVector out(ColumnType::Double());
    table.ReadChunk(0, 0, out);
    EXPECT_TRUE(SameValues(out, doubles));
}

TEST_F(ColumnarTableTest, WrongVectorTypeThrows) {
    auto bpm = MakePool(64);
    const ColumnarTable table(bpm.get(), meta_);
    ColumnVector dates(ColumnType::Date());
    EXPECT_THROW(table.ReadChunk(0, kBigInt, dates), DatabaseException);
    EXPECT_THROW(table.ReadValue(RowPosition{0, 0}, kBigInt, dates), DatabaseException);
}

TEST_F(ColumnarTableTest, MissingChunkThrows) {
    auto bpm = MakePool(64);
    ColumnarTableMeta meta = meta_;
    meta.row_groups[1].chunks.pop_back();
    EXPECT_THROW((void)ColumnarTable(bpm.get(), meta), DatabaseException);
}

TEST_F(ColumnarTableTest, FixtureEncodings) {
    // The reads above cover bit-packed and dictionary chunks.
    const auto& chunks = meta_.row_groups[0].chunks;
    EXPECT_EQ(chunks[kBigInt].encoding, Encoding::BITPACK);
    EXPECT_EQ(chunks[kNumeric].encoding, Encoding::BITPACK);
    EXPECT_EQ(chunks[kDate].encoding, Encoding::BITPACK);
    EXPECT_EQ(chunks[kText].encoding, Encoding::DICTIONARY);
    EXPECT_GT(chunks[kText].dict_page_count, 0u);
}

}  // namespace
}  // namespace shilmandb
