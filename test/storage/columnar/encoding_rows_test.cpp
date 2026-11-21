#include <gtest/gtest.h>
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "common/exception.hpp"
#include "storage/columnar/columnar_table.hpp"
#include "storage/columnar/encoding.hpp"
#include "storage/columnar/string_arena.hpp"
#include "storage/disk_manager.hpp"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace shilmandb {
namespace {

constexpr Encoding kAllEncodings[] = {Encoding::CONSTANT, Encoding::PLAIN, Encoding::BITPACK,
                                      Encoding::DICTIONARY};

// Hands out a chunk's pages one at a time and counts each page's pins.
// With poison on, an unpinned page's bytes are overwritten, so stale pointers show.
class CountingPages : public ChunkPages {
public:
    explicit CountingPages(const EncodedChunk& chunk, bool poison = false)
        : chunk_(chunk), poison_(poison), pins_(chunk.meta.page_count, 0), copy_(poison ? PAGE_SIZE : 0) {}
    ~CountingPages() override { EXPECT_EQ(pinned_, kNone) << "a page is still pinned"; }

    const char* Pin(uint32_t index) override {
        EXPECT_EQ(pinned_, kNone) << "two pages pinned at once";
        if (index >= chunk_.meta.page_count) {
            ADD_FAILURE() << "page " << index << " is past the chunk's " << chunk_.meta.page_count;
            throw DatabaseException("page past the chunk");
        }
        pinned_ = index;
        ++pins_[index];
        const char* page = chunk_.pages.data() + static_cast<size_t>(index) * PAGE_SIZE;
        if (!poison_) return page;
        std::memcpy(copy_.data(), page, PAGE_SIZE);
        return copy_.data();
    }
    void Unpin(uint32_t index) override {
        EXPECT_EQ(pinned_, index);
        pinned_ = kNone;
        if (poison_) std::memset(copy_.data(), 0xAB, PAGE_SIZE);
    }

    [[nodiscard]] const std::vector<uint32_t>& Pins() const { return pins_; }
    [[nodiscard]] uint32_t MaxPins() const { return pins_.empty() ? 0 : *std::max_element(pins_.begin(), pins_.end()); }
    [[nodiscard]] uint64_t TotalPins() const {
        uint64_t total = 0;
        for (uint32_t p : pins_) total += p;
        return total;
    }
    void ResetPins() { std::fill(pins_.begin(), pins_.end(), 0u); }

private:
    static constexpr uint32_t kNone = UINT32_MAX;
    const EncodedChunk& chunk_;
    bool poison_;
    std::vector<uint32_t> pins_;
    std::vector<char> copy_;  // the pinned page's bytes when poisoning
    uint32_t pinned_{kNone};
};

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

// Runs the DecodeRows overload for the type's kind and returns what it wrote.
ColumnVector DecodeRowsToVector(const EncodedChunk& chunk, const ColumnType& type, const uint32_t* rows, uint32_t n,
                                ChunkPages& pages, StringArena& arena) {
    ColumnVector out(type);
    switch (PhysicalKindOf(type)) {
        case PhysicalKind::INT64: {
            std::vector<int64_t> v(n);
            DecodeRows(chunk.meta, pages, type, rows, n, v.data());
            for (int64_t x : v) out.AppendInt(x);
            break;
        }
        case PhysicalKind::INT128: {
            std::vector<__int128> v(n);
            DecodeRows(chunk.meta, pages, type, rows, n, v.data());
            for (__int128 x : v) out.AppendWide(x);
            break;
        }
        case PhysicalKind::DOUBLE: {
            std::vector<double> v(n);
            DecodeRows(chunk.meta, pages, type, rows, n, v.data());
            for (double x : v) out.AppendDouble(x);
            break;
        }
        case PhysicalKind::STRING: {
            std::vector<StringRef> v(n);
            DecodeRows(chunk.meta, pages, type, rows, n, v.data(), arena);
            for (StringRef s : v) out.AppendString(View(s));
            break;
        }
    }
    return out;
}

// DecodeChunk's values, through pages of their own.
ColumnVector DecodeWhole(const EncodedChunk& chunk, const ColumnType& type) {
    CountingPages pages(chunk);
    ColumnVector out(type);
    DecodeChunk(chunk.meta, pages, out);
    return out;
}

// DecodeValue's value at each listed row, one call per row.
ColumnVector DecodeEach(const EncodedChunk& chunk, const ColumnType& type, const std::vector<uint32_t>& rows) {
    CountingPages pages(chunk);
    ColumnVector out(type);
    for (uint32_t r : rows) DecodeValue(chunk.meta, pages, r, out);
    return out;
}

// Rows 0 and count - 1, plus each other row with probability rate, ascending.
std::vector<uint32_t> RandomSubset(uint32_t count, double rate, std::mt19937_64& rng) {
    std::vector<uint32_t> rows;
    const bool all = rate >= 1.0;
    const uint64_t cut = all ? 0 : static_cast<uint64_t>(rate * 18446744073709551616.0);  // keep when rng() < cut
    for (uint32_t r = 0; r < count; ++r) {
        if (all || r == 0 || r + 1 == count || rng() < cut) rows.push_back(r);
    }
    return rows;
}

// 10 to the power n, as a 128-bit integer.
__int128 Pow10(int n) {
    __int128 v = 1;
    for (int i = 0; i < n; ++i) v *= 10;
    return v;
}

// A random value in [-(10^digits - 1), 10^digits - 1].
__int128 RandomScaled(int digits, std::mt19937_64& rng) {
    const __int128 limit = Pow10(digits);
    const auto raw = (static_cast<unsigned __int128>(rng()) << 64) | rng();
    const auto v = static_cast<__int128>(raw % static_cast<unsigned __int128>(limit));
    return (rng() & 1) ? -v : v;
}

// Lowercase letters, up to max_len of them.
std::string RandomString(size_t max_len, std::mt19937_64& rng) {
    std::string s(rng() % (max_len + 1), ' ');
    for (auto& c : s) c = static_cast<char>('a' + rng() % 26);
    return s;
}

// Appends one random value that fits the type.
void AppendRandom(ColumnVector& v, std::mt19937_64& rng) {
    const ColumnType& t = v.Type();
    switch (t.id) {
        case ColumnTypeId::BOOLEAN:   v.AppendInt(static_cast<int64_t>(rng() & 1)); break;
        case ColumnTypeId::SMALLINT:  v.AppendInt(static_cast<int16_t>(rng())); break;
        case ColumnTypeId::INTEGER:   v.AppendInt(static_cast<int32_t>(rng())); break;
        case ColumnTypeId::BIGINT:    v.AppendInt(static_cast<int64_t>(rng())); break;
        case ColumnTypeId::TIMESTAMP: v.AppendInt(static_cast<int64_t>(rng())); break;
        case ColumnTypeId::DATE:      v.AppendInt(-719528 + static_cast<int64_t>(rng() % 3652425)); break;
        case ColumnTypeId::NUMERIC:
            if (v.Kind() == PhysicalKind::INT64) {
                v.AppendInt(static_cast<int64_t>(RandomScaled(t.precision, rng)));
            } else {
                v.AppendWide(RandomScaled(t.precision, rng));
            }
            break;
        case ColumnTypeId::DOUBLE:
            v.AppendDouble(std::uniform_real_distribution<double>(-1e9, 1e9)(rng));
            break;
        case ColumnTypeId::VARCHAR: v.AppendString(RandomString(t.max_length, rng)); break;
        case ColumnTypeId::TEXT:    v.AppendString(RandomString(40, rng)); break;  // long ones have their own test
    }
}

// How a test vector's values are spread.
enum class Shape { RANDOM, FEW, NEAR, CONSTANT };

// The shape's name, for test traces.
const char* ShapeName(Shape s) {
    switch (s) {
        case Shape::RANDOM:   return "random";
        case Shape::FEW:      return "few";
        case Shape::NEAR:     return "near";
        case Shape::CONSTANT: return "constant";
    }
    return "?";
}

// n values of the type, no NULLs. FEW draws from 37 values; NEAR stays
// within 2^40 of -10^30 (wide NUMERIC only); CONSTANT repeats one value.
ColumnVector MakeVector(const ColumnType& type, size_t n, Shape shape, std::mt19937_64& rng) {
    ColumnVector v(type);
    v.Reserve(n);
    if (shape == Shape::NEAR) {
        for (size_t i = 0; i < n; ++i) v.AppendWide(-Pow10(30) + static_cast<__int128>(rng() % (uint64_t{1} << 40)));
        return v;
    }
    if (shape == Shape::RANDOM) {
        for (size_t i = 0; i < n; ++i) AppendRandom(v, rng);
        return v;
    }
    ColumnVector pool(type);
    const size_t distinct = shape == Shape::FEW ? 37 : 1;
    for (size_t i = 0; i < distinct; ++i) AppendRandom(pool, rng);
    for (size_t i = 0; i < n; ++i) AppendRow(pool, rng() % distinct, v);
    return v;
}

// One type of each storage kind and PLAIN width, wide NUMERIC included.
std::vector<ColumnType> EveryType() {
    return {ColumnType::Boolean(),      ColumnType::SmallInt(),      ColumnType::Integer(),
            ColumnType::BigInt(),       ColumnType::Numeric(15, 2),  ColumnType::Numeric(38, 6),
            ColumnType::Double(),       ColumnType::Date(),          ColumnType::Timestamp(),
            ColumnType::Varchar(25),    ColumnType::Text()};
}

// A copy of page index's header.
ColumnarPageHeader HeaderOf(const EncodedChunk& chunk, uint32_t index) {
    ColumnarPageHeader h;
    std::memcpy(&h, chunk.pages.data() + static_cast<size_t>(index) * PAGE_SIZE, sizeof(h));
    return h;
}

// The text of the DatabaseException f throws, or "" when it throws none.
template <typename F>
std::string ThrownText(F f) {
    try {
        f();
    } catch (const DatabaseException& e) {
        return e.what();
    }
    return "";
}

// DecodeRows gives DecodeChunk's values for all rows and a prefix, and DecodeValue's
// for random ascending subsets. No call pins a page twice.
void ExpectDecodeRowsMatches(const EncodedChunk& chunk, const ColumnType& type, std::mt19937_64& rng) {
    const uint32_t count = chunk.meta.value_count;
    const ColumnVector all = DecodeWhole(chunk, type);
    ASSERT_EQ(all.Size(), count);

    StringArena arena;
    CountingPages pages(chunk, true);
    EXPECT_TRUE(SameValues(DecodeRowsToVector(chunk, type, nullptr, count, pages, arena), all)) << "all rows";
    EXPECT_LE(pages.MaxPins(), 1u);

    // A prefix ends partway through a page.
    const uint32_t prefix = count / 2 + 1;
    ColumnVector head(type);
    for (uint32_t r = 0; r < prefix; ++r) AppendRow(all, r, head);
    pages.ResetPins();
    EXPECT_TRUE(SameValues(DecodeRowsToVector(chunk, type, nullptr, prefix, pages, arena), head)) << "prefix";
    EXPECT_LE(pages.MaxPins(), 1u);
    const ChunkMeta& meta = chunk.meta;
    for (size_t p = 0; p < meta.page_first_rows.size(); ++p) {
        const uint32_t want = meta.page_first_rows[p] < prefix ? 1 : 0;  // pages past the prefix stay unread
        EXPECT_EQ(pages.Pins()[meta.dict_page_count + p], want) << "data page " << p << " for a prefix of " << prefix;
    }

    for (double rate : {0.01, 0.2}) {
        const std::vector<uint32_t> rows = RandomSubset(count, rate, rng);
        const ColumnVector expected = DecodeEach(chunk, type, rows);
        pages.ResetPins();
        const auto n = static_cast<uint32_t>(rows.size());
        EXPECT_TRUE(SameValues(DecodeRowsToVector(chunk, type, rows.data(), n, pages, arena), expected)) << "rate " << rate;
        EXPECT_LE(pages.MaxPins(), 1u);
    }
}

// Runs the DecodeRows overload for the type's kind and throws the values away.
void DecodeRowsDiscard(const ChunkMeta& meta, ChunkPages& pages, const ColumnType& type, const uint32_t* rows, uint32_t n) {
    StringArena arena;
    switch (PhysicalKindOf(type)) {
        case PhysicalKind::INT64: {
            std::vector<int64_t> v(n);
            DecodeRows(meta, pages, type, rows, n, v.data());
            break;
        }
        case PhysicalKind::INT128: {
            std::vector<__int128> v(n);
            DecodeRows(meta, pages, type, rows, n, v.data());
            break;
        }
        case PhysicalKind::DOUBLE: {
            std::vector<double> v(n);
            DecodeRows(meta, pages, type, rows, n, v.data());
            break;
        }
        case PhysicalKind::STRING: {
            std::vector<StringRef> v(n);
            DecodeRows(meta, pages, type, rows, n, v.data(), arena);
            break;
        }
    }
}

// Whether a thrown text marks the chunk as damaged.
::testing::AssertionResult IsDamaged(const std::string& text) {
    if (text.rfind("damaged column", 0) == 0) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << (text.empty() ? "no throw" : text);
}

// DecodeRows refuses the chunk as damaged, for all rows and for just row,
// leaving no page pinned. With codes set, DecodeCodes refuses a DICTIONARY chunk too.
void ExpectRowsDamaged(const EncodedChunk& chunk, const ColumnType& type, uint32_t row, bool codes = true) {
    const ChunkMeta& meta = chunk.meta;
    const uint32_t one[] = {row};
    {
        CountingPages pages(chunk);
        EXPECT_TRUE(IsDamaged(ThrownText([&] { DecodeRowsDiscard(meta, pages, type, nullptr, meta.value_count); })))
            << "DecodeRows, all rows";
        EXPECT_TRUE(IsDamaged(ThrownText([&] { DecodeRowsDiscard(meta, pages, type, one, 1); }))) << "DecodeRows, row " << row;
    }
    if (codes && meta.encoding == Encoding::DICTIONARY) {
        CountingPages pages(chunk);
        std::vector<uint32_t> out(meta.value_count);
        EXPECT_TRUE(IsDamaged(ThrownText([&] { DecodeCodes(meta, pages, nullptr, meta.value_count, out.data()); })))
            << "DecodeCodes, all rows";
        EXPECT_TRUE(IsDamaged(ThrownText([&] { DecodeCodes(meta, pages, one, 1, out.data()); }))) << "DecodeCodes, row " << row;
    }
}

// Rewrites one page's value count and NULL flag.
void ForgeHeader(EncodedChunk& chunk, uint32_t index, uint16_t count, bool nulls) {
    char* page = chunk.pages.data() + static_cast<size_t>(index) * PAGE_SIZE;
    ColumnarPageHeader h;
    std::memcpy(&h, page, sizeof(h));
    h.value_count = count;
    if (nulls) h.flags = static_cast<uint8_t>(h.flags | kPageHasNulls);
    std::memcpy(page, &h, sizeof(h));
}

// Writes entry i of a PLAIN string page's offset table.
void SetStringOffset(EncodedChunk& chunk, uint32_t index, uint32_t i, uint16_t offset) {
    char* page = chunk.pages.data() + static_cast<size_t>(index) * PAGE_SIZE;
    std::memcpy(page + kPageHeaderSize + 2 * static_cast<size_t>(i), &offset, sizeof(offset));
}

// Reads entry i of a PLAIN string page's offset table.
uint16_t StringOffsetAt(const EncodedChunk& chunk, uint32_t index, uint32_t i) {
    uint16_t offset;
    std::memcpy(&offset, chunk.pages.data() + static_cast<size_t>(index) * PAGE_SIZE + kPageHeaderSize + 2 * static_cast<size_t>(i), sizeof(offset));
    return offset;
}

// Big BIGINTs that never repeat, so DICTIONARY has an entry per row.
ColumnVector DistinctBigInts(int n) {
    ColumnVector v(ColumnType::BigInt());
    for (int i = 0; i < n; ++i) v.AppendInt(i * 1000003LL * 1000003LL);
    return v;
}

// Records the order a decoder pins pages in; pages come from memory.
class OrderPages : public ChunkPages {
public:
    explicit OrderPages(const EncodedChunk& chunk) : chunk_(chunk) {}
    const char* Pin(uint32_t index) override {
        order_.push_back(index);
        return chunk_.pages.data() + static_cast<size_t>(index) * PAGE_SIZE;
    }
    void Unpin(uint32_t) override {}
    [[nodiscard]] const std::vector<uint32_t>& Order() const { return order_; }

private:
    const EncodedChunk& chunk_;
    std::vector<uint32_t> order_;
};

// A temp file name for this process.
std::string TempPath(const std::string& name) {
    return (std::filesystem::temp_directory_path() / ("shilmandb_encoding_rows_" + name + "_" + std::to_string(::getpid()))).string();
}

constexpr uint32_t kBatchRows = 100000;
constexpr uint32_t kBatchId = 0, kBatchWide = 1, kBatchRate = 2, kBatchNote = 3, kBatchWord = 4;
constexpr uint32_t kBatchNullRow = 7500;
constexpr size_t kBatchPool = 64 * kPinBatch;  // full batches, and the whole table fits

// One row group with chunks longer than a pin batch, and its values.
struct BatchTable {
    ColumnarTableMeta meta;
    std::vector<ColumnVector> values;
};

// Writes the table; with a NULL, b_rate holds one at kBatchNullRow.
BatchTable MakeBatchTable(DiskManager& disk, std::mt19937_64& rng, bool with_null = false) {
    BatchTable t;
    t.meta.oid = 1;
    t.meta.name = "batch";
    t.meta.columns = {{"b_id", 1, ColumnType::BigInt(), false},
                      {"b_wide", 2, ColumnType::Numeric(38, 6), false},
                      {"b_rate", 3, ColumnType::Double(), with_null},
                      {"b_note", 4, ColumnType::Text(), false},
                      {"b_word", 5, ColumnType::Text(), false}};
    for (uint32_t c = 0; c < kBatchWord; ++c) t.values.push_back(MakeVector(t.meta.columns[c].type, kBatchRows, Shape::RANDOM, rng));
    if (with_null) {
        ColumnVector rate(ColumnType::Double());
        for (uint32_t r = 0; r < kBatchRows; ++r) {
            if (r == kBatchNullRow) {
                rate.AppendNull();
            } else {
                AppendRow(t.values[kBatchRate], r, rate);
            }
        }
        t.values[kBatchRate] = std::move(rate);
    }

    // 4,000 long words: a dictionary over several pages
    std::vector<std::string> words;
    for (int i = 0; i < 4000; ++i) words.push_back("word_" + std::to_string(i) + std::string(25, static_cast<char>('a' + i % 26)));
    ColumnVector word(ColumnType::Text());
    for (uint32_t r = 0; r < kBatchRows; ++r) word.AppendString(words[rng() % words.size()]);
    t.values.push_back(std::move(word));

    RowGroupMeta group;
    group.row_count = kBatchRows;
    for (const ColumnVector& col : t.values) group.chunks.push_back(WriteChunk(&disk, col));
    t.meta.row_count = kBatchRows;
    t.meta.row_groups.push_back(std::move(group));
    return t;
}

// ReadRows for the column's kind, as a vector; rows null means 0..n-1.
ColumnVector ReadRowsToVector(const ColumnarTable& table, uint32_t column, const uint32_t* rows, uint32_t n, StringArena& arena) {
    const ColumnType& type = table.Meta().columns[column].type;
    ColumnVector got(type);
    switch (PhysicalKindOf(type)) {
        case PhysicalKind::INT64: {
            std::vector<int64_t> v(n);
            table.ReadRows(0, column, rows, n, v.data());
            for (int64_t x : v) got.AppendInt(x);
            break;
        }
        case PhysicalKind::INT128: {
            std::vector<__int128> v(n);
            table.ReadRows(0, column, rows, n, v.data());
            for (__int128 x : v) got.AppendWide(x);
            break;
        }
        case PhysicalKind::DOUBLE: {
            std::vector<double> v(n);
            table.ReadRows(0, column, rows, n, v.data());
            for (double x : v) got.AppendDouble(x);
            break;
        }
        case PhysicalKind::STRING: {
            std::vector<StringRef> v(n);
            table.ReadRows(0, column, rows, n, v.data(), arena);
            for (StringRef s : v) got.AppendString(View(s));
            break;
        }
    }
    return got;
}

// The listed rows of src, or all of them when rows is empty.
ColumnVector RowsOf(const ColumnVector& src, const std::vector<uint32_t>& rows) {
    ColumnVector out(src.Type());
    if (rows.empty()) {
        for (size_t r = 0; r < src.Size(); ++r) AppendRow(src, r, out);
    } else {
        for (uint32_t r : rows) AppendRow(src, r, out);
    }
    return out;
}

// The chunk page indexes a trace file lists, in order.
std::vector<uint32_t> TracedPages(const std::string& path, page_id_t first_page) {
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);  // the header
    std::vector<uint32_t> pages;
    while (std::getline(in, line)) {
        const auto pid = static_cast<page_id_t>(std::stoul(line.substr(line.find(',') + 1)));
        pages.push_back(pid - first_page);
    }
    return pages;
}

// True when no frame of the pool is pinned.
::testing::AssertionResult NothingPinned(const BufferPoolManager& bpm) {
    for (size_t f = 0; f < bpm.GetPoolSize(); ++f) {
        const Page& page = bpm.GetPage(static_cast<frame_id_t>(f));
        if (page.GetPinCount() != 0) {
            return ::testing::AssertionFailure() << "frame " << f << " holds page " << page.GetPageId() << " pinned " << page.GetPinCount() << " times";
        }
    }
    return ::testing::AssertionSuccess();
}

}  // namespace

TEST(EncodingRowsTest, DecodeRowsEqualsDecodeChunk) {
    std::mt19937_64 rng(2026);
    std::set<std::pair<int, int>> covered;  // (kind, encoding) seen on several pages, or CONSTANT
    for (const auto& type : EveryType()) {
        const PhysicalKind kind = PhysicalKindOf(type);
        for (size_t n : {1u, 2047u, 65536u}) {
            for (Shape shape : {Shape::RANDOM, Shape::FEW, Shape::NEAR, Shape::CONSTANT}) {
                if (shape == Shape::NEAR && kind != PhysicalKind::INT128) continue;
                const ColumnVector v = MakeVector(type, n, shape, rng);
                for (Encoding e : kAllEncodings) {
                    const auto chunk = TryEncodeChunk(v, e);
                    if (!chunk) continue;
                    SCOPED_TRACE(ColumnTypeName(type) + " n=" + std::to_string(n) + " " + ShapeName(shape) +
                                 " encoding " + std::to_string(static_cast<int>(e)));
                    ExpectDecodeRowsMatches(*chunk, type, rng);
                    if (e == Encoding::CONSTANT || chunk->meta.page_count - chunk->meta.dict_page_count > 1) {
                        covered.insert({static_cast<int>(kind), static_cast<int>(e)});
                    }
                }
            }
        }
    }

    // Each kind with each encoding that applies to it, CONSTANT aside, spans several data pages.
    const auto pair = [](PhysicalKind k, Encoding e) { return std::make_pair(static_cast<int>(k), static_cast<int>(e)); };
    const std::set<std::pair<int, int>> want = {
        pair(PhysicalKind::INT64, Encoding::CONSTANT),  pair(PhysicalKind::INT64, Encoding::PLAIN),
        pair(PhysicalKind::INT64, Encoding::BITPACK),   pair(PhysicalKind::INT64, Encoding::DICTIONARY),
        pair(PhysicalKind::INT128, Encoding::CONSTANT), pair(PhysicalKind::INT128, Encoding::PLAIN),
        pair(PhysicalKind::INT128, Encoding::BITPACK),  pair(PhysicalKind::INT128, Encoding::DICTIONARY),
        pair(PhysicalKind::DOUBLE, Encoding::CONSTANT), pair(PhysicalKind::DOUBLE, Encoding::PLAIN),
        pair(PhysicalKind::DOUBLE, Encoding::DICTIONARY),
        pair(PhysicalKind::STRING, Encoding::CONSTANT), pair(PhysicalKind::STRING, Encoding::PLAIN),
        pair(PhysicalKind::STRING, Encoding::DICTIONARY),
    };
    EXPECT_EQ(covered, want);
}

TEST(EncodingRowsTest, DecodeRowsEveryBitWidth) {
    std::mt19937_64 rng(64);
    for (uint32_t w = 0; w <= 64; ++w) {
        SCOPED_TRACE("width " + std::to_string(w));
        const uint64_t mask = w == 64 ? ~uint64_t{0} : (uint64_t{1} << w) - 1;
        const int64_t lo = w == 0 ? 42 : (w == 64 ? INT64_MIN : -static_cast<int64_t>(mask / 2) - 1);
        const auto hi = static_cast<int64_t>(static_cast<uint64_t>(lo) + mask);

        // Enough values for several pages at the wider widths.
        ColumnVector v(ColumnType::BigInt());
        v.AppendInt(lo);
        v.AppendInt(hi);
        for (int i = 0; i < 6000; ++i) {
            v.AppendInt(static_cast<int64_t>(static_cast<uint64_t>(lo) + (mask == 0 ? 0 : rng() % mask)));
        }
        const auto chunk = TryEncodeChunk(v, Encoding::BITPACK);
        ASSERT_TRUE(chunk.has_value());
        EXPECT_EQ(HeaderOf(*chunk, 0).bit_width, w);
        ExpectDecodeRowsMatches(*chunk, v.Type(), rng);

        // The same widths for a wide NUMERIC, offset far from zero.
        ColumnVector wide(ColumnType::Numeric(38, 0));
        const __int128 base = -Pow10(30);
        wide.AppendWide(base);
        wide.AppendWide(base + static_cast<__int128>(mask));
        for (int i = 0; i < 1500; ++i) {
            wide.AppendWide(base + static_cast<__int128>(mask == 0 ? 0 : rng() % mask));
        }
        const auto wide_chunk = TryEncodeChunk(wide, Encoding::BITPACK);
        ASSERT_TRUE(wide_chunk.has_value());
        EXPECT_EQ(HeaderOf(*wide_chunk, 0).bit_width, w);
        ExpectDecodeRowsMatches(*wide_chunk, wide.Type(), rng);
    }

    // INT64_MIN and INT64_MAX together, through every encoding that applies.
    ColumnVector extremes(ColumnType::BigInt());
    for (int i = 0; i < 3000; ++i) {
        extremes.AppendInt(i % 3 == 0 ? INT64_MIN : (i % 3 == 1 ? INT64_MAX : 0));
    }
    ColumnVector stamps(ColumnType::Timestamp());
    for (int i = 0; i < 3000; ++i) stamps.AppendInt(i % 2 == 0 ? INT64_MIN : INT64_MAX);
    for (const ColumnVector* v : {&extremes, &stamps}) {
        for (Encoding e : kAllEncodings) {
            const auto chunk = TryEncodeChunk(*v, e);
            if (!chunk) continue;
            SCOPED_TRACE(ColumnTypeName(v->Type()) + " extremes, encoding " + std::to_string(static_cast<int>(e)));
            if (e == Encoding::BITPACK) EXPECT_EQ(HeaderOf(*chunk, 0).bit_width, 64);
            ExpectDecodeRowsMatches(*chunk, v->Type(), rng);
        }
    }
}

TEST(EncodingRowsTest, EachPageFetchedOnce) {
    std::mt19937_64 rng(7);
    const std::pair<ColumnType, Shape> cases[] = {
        {ColumnType::BigInt(), Shape::RANDOM},
        {ColumnType::Numeric(38, 6), Shape::NEAR},
        {ColumnType::Double(), Shape::RANDOM},
        {ColumnType::Text(), Shape::RANDOM},
        {ColumnType::Text(), Shape::FEW},
    };
    for (const auto& [type, shape] : cases) {
        const ColumnVector v = MakeVector(type, 65536, shape, rng);
        for (Encoding e : {Encoding::PLAIN, Encoding::BITPACK, Encoding::DICTIONARY}) {
            const auto chunk = TryEncodeChunk(v, e);
            if (!chunk) continue;
            SCOPED_TRACE(ColumnTypeName(type) + " " + ShapeName(shape) + " encoding " + std::to_string(static_cast<int>(e)));
            const ChunkMeta& meta = chunk->meta;
            const uint32_t dict_pages = meta.dict_page_count;
            const auto data_pages = static_cast<uint32_t>(meta.page_first_rows.size());
            ASSERT_GE(data_pages, 4u);

            // Requested rows: the middle of every data page, or of every even one.
            for (bool every_page : {true, false}) {
                std::vector<uint32_t> rows;
                std::vector<bool> holds(data_pages, false);
                for (uint32_t p = 0; p < data_pages; ++p) {
                    if (!every_page && p % 2 == 1) continue;
                    const uint32_t first = meta.page_first_rows[p];
                    const uint32_t end = p + 1 < data_pages ? meta.page_first_rows[p + 1] : meta.value_count;
                    rows.push_back(first + (end - first) / 2);
                    if (end - first > 2) rows.push_back(end - 1);
                    holds[p] = true;
                }
                std::sort(rows.begin(), rows.end());
                const auto n = static_cast<uint32_t>(rows.size());

                StringArena arena;
                CountingPages pages(*chunk);
                (void)DecodeRowsToVector(*chunk, type, rows.data(), n, pages, arena);
                for (uint32_t p = 0; p < data_pages; ++p) {
                    EXPECT_EQ(pages.Pins()[dict_pages + p], holds[p] ? 1u : 0u) << "data page " << p;
                }

                // A dictionary page is read once when it holds an entry in use, else never.
                if (e == Encoding::DICTIONARY) {
                    std::vector<uint32_t> codes(n);
                    CountingPages code_pages(*chunk);
                    DecodeCodes(meta, code_pages, rows.data(), n, codes.data());
                    for (uint32_t j = 0; j < dict_pages; ++j) {
                        EXPECT_EQ(code_pages.Pins()[j], 0u) << "codes read dictionary page " << j;
                        const uint32_t first = meta.dict_page_first_entries[j];
                        const uint32_t end = j + 1 < dict_pages ? meta.dict_page_first_entries[j + 1] : meta.dict_size;
                        const bool used = std::any_of(codes.begin(), codes.end(), [&](uint32_t c) { return c >= first && c < end; });
                        EXPECT_EQ(pages.Pins()[j], used ? 1u : 0u) << "dictionary page " << j;
                    }
                    for (uint32_t p = 0; p < data_pages; ++p) {
                        EXPECT_EQ(code_pages.Pins()[dict_pages + p], holds[p] ? 1u : 0u) << "codes, data page " << p;
                    }
                    if (!every_page) {
                        EXPECT_LT(pages.TotalPins(), meta.page_count) << "some pages were never needed";
                    }
                }
            }

            // All rows: every page exactly once.
            StringArena arena;
            CountingPages pages(*chunk);
            (void)DecodeRowsToVector(*chunk, type, nullptr, meta.value_count, pages, arena);
            for (uint32_t i = 0; i < meta.page_count; ++i) EXPECT_EQ(pages.Pins()[i], 1u) << "page " << i;

            if (e == Encoding::DICTIONARY && PhysicalKindOf(type) == PhysicalKind::STRING) {
                CountingPages dict_only(*chunk);
                std::vector<StringRef> entries;
                DecodeDictionary(meta, dict_only, type, entries, arena);
                for (uint32_t i = 0; i < meta.page_count; ++i) {
                    EXPECT_EQ(dict_only.Pins()[i], i < dict_pages ? 1u : 0u) << "DecodeDictionary, page " << i;
                }
            }
        }
    }
}

TEST(EncodingRowsTest, CodesAndDictionaryRebuildValues) {
    std::mt19937_64 rng(11);
    const std::pair<ColumnType, Shape> string_cases[] = {
        {ColumnType::Text(), Shape::FEW},
        {ColumnType::Text(), Shape::RANDOM},
        {ColumnType::Varchar(25), Shape::FEW},
    };
    for (const auto& [type, shape] : string_cases) {
        SCOPED_TRACE(ColumnTypeName(type) + " " + ShapeName(shape));
        const ColumnVector v = MakeVector(type, 20000, shape, rng);
        const auto chunk = TryEncodeChunk(v, Encoding::DICTIONARY);
        ASSERT_TRUE(chunk.has_value());
        const uint32_t count = chunk->meta.value_count;
        const ColumnVector all = DecodeWhole(*chunk, type);

        StringArena arena;
        std::vector<StringRef> entries;
        std::vector<uint32_t> codes(count);
        {
            CountingPages pages(*chunk, true);
            DecodeDictionary(chunk->meta, pages, type, entries, arena);
            DecodeCodes(chunk->meta, pages, nullptr, count, codes.data());
        }
        ASSERT_EQ(entries.size(), chunk->meta.dict_size);
        std::set<std::string> distinct;
        for (StringRef s : entries) distinct.emplace(View(s));
        EXPECT_EQ(distinct.size(), entries.size()) << "entries are distinct";

        for (uint32_t i = 0; i < count; ++i) {
            ASSERT_LT(codes[i], entries.size());
            ASSERT_EQ(View(entries[codes[i]]), all.GetString(i)) << "row " << i;
        }

        // A subset's codes are the same rows' codes from the whole decode.
        const std::vector<uint32_t> rows = RandomSubset(count, 0.05, rng);
        std::vector<uint32_t> some(rows.size());
        CountingPages pages(*chunk);
        DecodeCodes(chunk->meta, pages, rows.data(), static_cast<uint32_t>(rows.size()), some.data());
        for (size_t k = 0; k < rows.size(); ++k) EXPECT_EQ(some[k], codes[rows[k]]) << "row " << rows[k];

        // DecodeDictionary replaces what out held.
        std::vector<StringRef> again(3);
        DecodeDictionary(chunk->meta, pages, type, again, arena);
        EXPECT_EQ(again.size(), entries.size());
    }

    // Numeric dictionaries have codes too: equal codes mean equal values.
    for (const ColumnType& type : {ColumnType::BigInt(), ColumnType::Numeric(38, 6), ColumnType::Double()}) {
        SCOPED_TRACE(ColumnTypeName(type));
        const ColumnVector v = MakeVector(type, 20000, Shape::FEW, rng);
        const auto chunk = TryEncodeChunk(v, Encoding::DICTIONARY);
        ASSERT_TRUE(chunk.has_value());
        const uint32_t count = chunk->meta.value_count;
        const ColumnVector all = DecodeWhole(*chunk, type);

        std::vector<uint32_t> codes(count);
        CountingPages pages(*chunk);
        DecodeCodes(chunk->meta, pages, nullptr, count, codes.data());
        std::vector<int64_t> first_row(chunk->meta.dict_size, -1);
        for (uint32_t i = 0; i < count; ++i) {
            ASSERT_LT(codes[i], chunk->meta.dict_size);
            if (first_row[codes[i]] < 0) {
                first_row[codes[i]] = i;
                continue;
            }
            ColumnVector a(type), b(type);
            AppendRow(all, static_cast<size_t>(first_row[codes[i]]), a);
            AppendRow(all, i, b);
            ASSERT_TRUE(SameValues(a, b)) << "row " << i;
        }
        EXPECT_TRUE(std::all_of(first_row.begin(), first_row.end(), [](int64_t r) { return r >= 0; })) << "every entry used";

        std::vector<StringRef> entries;
        StringArena arena;
        EXPECT_FALSE(ThrownText([&] { DecodeDictionary(chunk->meta, pages, type, entries, arena); }).empty())
            << "DecodeDictionary takes string chunks only";
    }

    // Codes and the dictionary need a DICTIONARY chunk.
    const ColumnVector text = MakeVector(ColumnType::Text(), 500, Shape::FEW, rng);
    const auto plain = TryEncodeChunk(text, Encoding::PLAIN);
    ASSERT_TRUE(plain.has_value());
    CountingPages pages(*plain);
    std::vector<uint32_t> codes(text.Size());
    std::vector<StringRef> entries;
    StringArena arena;
    EXPECT_FALSE(ThrownText([&] { DecodeCodes(plain->meta, pages, nullptr, 500, codes.data()); }).empty());
    EXPECT_FALSE(ThrownText([&] { DecodeDictionary(plain->meta, pages, ColumnType::Text(), entries, arena); }).empty());
    EXPECT_EQ(pages.TotalPins(), 0u);
}

TEST(EncodingRowsTest, LongAndMultibyteStrings) {
    // Long values, 2- to 4-byte UTF-8 and empty strings, several of each.
    const std::string two = "\xC3\xA9";            // e acute
    const std::string three = "\xE2\x82\xAC";      // euro sign
    const std::string four = "\xF0\x9F\x98\x80";   // grinning face
    std::vector<std::string> kinds = {"", "a", two, three, four, two + "x" + three + four + "", "plain ascii"};
    for (int i = 0; i < 6; ++i) {
        std::string long_value(8000, static_cast<char>('A' + i));
        kinds.push_back(long_value);
    }
    std::string emoji_run;
    for (int i = 0; i < 2000; ++i) emoji_run += (i % 3 == 0 ? four : (i % 3 == 1 ? three + "b" : two + "cd"));
    kinds.push_back(emoji_run.substr(0, 8000));
    kinds.push_back(std::string(kMaxValueBytes, 'z'));

    std::vector<std::string> values;
    for (int i = 0; i < 400; ++i) values.push_back(kinds[(i * 7 + i / 13) % kinds.size()]);
    ColumnVector v(ColumnType::Text());
    for (const auto& s : values) v.AppendString(s);

    std::mt19937_64 rng(5);
    for (Encoding e : {Encoding::PLAIN, Encoding::DICTIONARY}) {
        SCOPED_TRACE("encoding " + std::to_string(static_cast<int>(e)));
        StringArena arena;
        const auto n = static_cast<uint32_t>(values.size());
        std::vector<StringRef> all(n);
        const std::vector<uint32_t> rows = RandomSubset(n, 0.2, rng);
        std::vector<StringRef> some(rows.size());
        std::vector<StringRef> entries;
        {
            auto chunk = std::make_unique<EncodedChunk>(*TryEncodeChunk(v, e));
            ASSERT_GE(chunk->meta.page_count, 8u);
            CountingPages pages(*chunk, true);
            DecodeRows(chunk->meta, pages, ColumnType::Text(), nullptr, n, all.data(), arena);
            DecodeRows(chunk->meta, pages, ColumnType::Text(), rows.data(), static_cast<uint32_t>(rows.size()), some.data(), arena);
            if (e == Encoding::DICTIONARY) DecodeDictionary(chunk->meta, pages, ColumnType::Text(), entries, arena);
        }
        // Every page is unpinned, poisoned and freed; the refs still read their bytes.
        for (uint32_t i = 0; i < n; ++i) ASSERT_EQ(View(all[i]), values[i]) << "row " << i;
        for (size_t k = 0; k < rows.size(); ++k) ASSERT_EQ(View(some[k]), values[rows[k]]) << "row " << rows[k];
        if (e == Encoding::DICTIONARY) {
            std::set<std::string> want(kinds.begin(), kinds.end());
            std::set<std::string> got;
            for (StringRef s : entries) got.emplace(View(s));
            EXPECT_EQ(got, want);
        }
    }

    // A CONSTANT chunk's value is copied too.
    ColumnVector same(ColumnType::Text());
    for (int i = 0; i < 50; ++i) same.AppendString(emoji_run.substr(0, 8000));
    StringArena arena;
    std::vector<StringRef> out(50);
    {
        auto chunk = std::make_unique<EncodedChunk>(*TryEncodeChunk(same, Encoding::CONSTANT));
        CountingPages pages(*chunk, true);
        DecodeRows(chunk->meta, pages, ColumnType::Text(), nullptr, 50, out.data(), arena);
    }
    for (StringRef s : out) EXPECT_EQ(View(s), emoji_run.substr(0, 8000));
}

TEST(EncodingRowsTest, NullSlotThrows) {
    std::mt19937_64 rng(3);
    for (const ColumnType& type : {ColumnType::BigInt(), ColumnType::Date(), ColumnType::Numeric(38, 6), ColumnType::Double(), ColumnType::Text()}) {
        const ColumnVector values = MakeVector(type, 5000, Shape::FEW, rng);
        ColumnVector v(type);
        for (uint32_t r = 0; r < 5000; ++r) {
            if (r == 3000 || r == 4321) {
                v.AppendNull();
            } else {
                AppendRow(values, r, v);
            }
        }
        for (Encoding e : {Encoding::PLAIN, Encoding::BITPACK, Encoding::DICTIONARY}) {
            const auto chunk = TryEncodeChunk(v, e);
            if (!chunk) continue;
            SCOPED_TRACE(ColumnTypeName(type) + " encoding " + std::to_string(static_cast<int>(e)));
            StringArena arena;
            CountingPages pages(*chunk);
            auto decode = [&](std::vector<uint32_t> rows) {
                return ThrownText([&] {
                    (void)DecodeRowsToVector(*chunk, type, rows.data(), static_cast<uint32_t>(rows.size()), pages, arena);
                });
            };
            EXPECT_EQ(ThrownText([&] { (void)DecodeRowsToVector(*chunk, type, nullptr, 5000, pages, arena); }),
                      "damaged column chunk: row 3000 is NULL");
            EXPECT_EQ(decode({10, 3000}), "damaged column chunk: row 3000 is NULL");
            EXPECT_EQ(decode({4321, 4999}), "damaged column chunk: row 4321 is NULL");

            // Rows around the NULLs still decode.
            const std::vector<uint32_t> fine = {0, 2999, 3001, 4320, 4322, 4999};
            EXPECT_EQ(decode(fine), "");
            EXPECT_TRUE(SameValues(DecodeRowsToVector(*chunk, type, fine.data(), 6, pages, arena), DecodeEach(*chunk, type, fine)));
            EXPECT_TRUE(SameValues(DecodeRowsToVector(*chunk, type, nullptr, 3000, pages, arena),
                                   DecodeEach(*chunk, type, RandomSubset(3000, 1.0, rng))));

            if (e == Encoding::DICTIONARY) {
                std::vector<uint32_t> codes(2);
                const uint32_t rows[] = {1, 4321};
                EXPECT_EQ(ThrownText([&] { DecodeCodes(chunk->meta, pages, rows, 2, codes.data()); }),
                          "damaged column chunk: row 4321 is NULL");
            }
        }

        // An all-NULL chunk is CONSTANT with no value.
        ColumnVector nulls(type);
        for (int i = 0; i < 100; ++i) nulls.AppendNull();
        const auto chunk = TryEncodeChunk(nulls, Encoding::CONSTANT);
        ASSERT_TRUE(chunk.has_value());
        ASSERT_FALSE(chunk->meta.constant.has_value());
        StringArena arena;
        CountingPages pages(*chunk);
        const uint32_t rows[] = {7, 8};
        EXPECT_EQ(ThrownText([&] { (void)DecodeRowsToVector(*chunk, type, rows, 2, pages, arena); }),
                  "damaged column chunk: row 7 is NULL");
        EXPECT_EQ(ThrownText([&] { (void)DecodeRowsToVector(*chunk, type, nullptr, 100, pages, arena); }),
                  "damaged column chunk: row 0 is NULL");
        EXPECT_EQ(ThrownText([&] { (void)DecodeRowsToVector(*chunk, type, nullptr, 0, pages, arena); }), "");
    }
}

TEST(EncodingRowsTest, BadRequestsThrow) {
    std::mt19937_64 rng(9);
    const ColumnVector ints = MakeVector(ColumnType::BigInt(), 3000, Shape::RANDOM, rng);
    const auto chunk = TryEncodeChunk(ints, Encoding::PLAIN);
    ASSERT_TRUE(chunk.has_value());
    CountingPages pages(*chunk);
    StringArena arena;

    // An overload that doesn't match the type's physical kind.
    std::vector<__int128> wides(3);
    std::vector<double> doubles(3);
    std::vector<StringRef> strings(3);
    std::vector<int64_t> out(3000);
    EXPECT_FALSE(ThrownText([&] { DecodeRows(chunk->meta, pages, ColumnType::BigInt(), nullptr, 3, wides.data()); }).empty());
    EXPECT_FALSE(ThrownText([&] { DecodeRows(chunk->meta, pages, ColumnType::BigInt(), nullptr, 3, doubles.data()); }).empty());
    EXPECT_FALSE(ThrownText([&] { DecodeRows(chunk->meta, pages, ColumnType::BigInt(), nullptr, 3, strings.data(), arena); }).empty());
    EXPECT_FALSE(ThrownText([&] { DecodeRows(chunk->meta, pages, ColumnType::Text(), nullptr, 3, out.data()); }).empty());
    EXPECT_FALSE(ThrownText([&] { DecodeRows(chunk->meta, pages, ColumnType::Numeric(38, 2), nullptr, 3, out.data()); }).empty());

    // Rows past the chunk, out of order, or more than it holds.
    const uint32_t past[] = {5, 3000};
    const uint32_t backwards[] = {9, 4};
    EXPECT_EQ(ThrownText([&] { DecodeRows(chunk->meta, pages, ColumnType::BigInt(), past, 2, out.data()); }),
              "row 3000 is past the chunk's 3000 values");
    EXPECT_FALSE(ThrownText([&] { DecodeRows(chunk->meta, pages, ColumnType::BigInt(), backwards, 2, out.data()); }).empty());
    EXPECT_EQ(ThrownText([&] { DecodeRows(chunk->meta, pages, ColumnType::BigInt(), nullptr, 3001, out.data()); }),
              "row 3000 is past the chunk's 3000 values");
    EXPECT_EQ(pages.TotalPins(), 0u) << "a bad request pins nothing";

    // No rows pin nothing; a repeated row reads twice.
    DecodeRows(chunk->meta, pages, ColumnType::BigInt(), nullptr, 0, out.data());
    EXPECT_EQ(pages.TotalPins(), 0u);
    const uint32_t twice[] = {5, 5, 2999};
    DecodeRows(chunk->meta, pages, ColumnType::BigInt(), twice, 3, out.data());
    EXPECT_EQ(out[0], ints.GetInt(5));
    EXPECT_EQ(out[1], ints.GetInt(5));
    EXPECT_EQ(out[2], ints.GetInt(2999));

    // BITPACK on a DOUBLE column is a damaged chunk, as DecodeChunk says.
    ColumnVector widths(ColumnType::BigInt());
    for (int i = 0; i < 100; ++i) widths.AppendInt(i);
    const auto packed = TryEncodeChunk(widths, Encoding::BITPACK);
    ASSERT_TRUE(packed.has_value());
    CountingPages packed_pages(*packed);
    std::vector<double> hundred(100);
    EXPECT_EQ(ThrownText([&] { DecodeRows(packed->meta, packed_pages, ColumnType::Double(), nullptr, 100, hundred.data()); })
                  .rfind("damaged column page", 0),
              0u);
}

TEST(EncodingRowsTest, ColumnarTableReadRowsMatchReadChunk) {
    const std::string path = (std::filesystem::temp_directory_path() /
                              ("shilmandb_encoding_rows_" + std::to_string(::getpid()) + ".db"))
                                 .string();
    std::filesystem::remove(path);
    {
        DiskManager disk(path);
        ColumnarTableMeta meta;
        meta.oid = 1;
        meta.name = "orders";
        meta.columns = {{"o_id", 1, ColumnType::BigInt(), false},
                        {"o_price", 2, ColumnType::Numeric(15, 2), false},
                        {"o_big", 3, ColumnType::Numeric(38, 6), false},
                        {"o_rate", 4, ColumnType::Double(), false},
                        {"o_word", 5, ColumnType::Text(), false}};
        constexpr uint32_t kOid = 0, kWord = 4;
        const uint32_t group_rows[] = {9000, 37};
        const char* const words[] = {"alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"};

        std::mt19937_64 rng(42);
        for (uint32_t rows : group_rows) {
            std::vector<ColumnVector> cols;
            for (const ColumnMeta& c : meta.columns) cols.push_back(MakeVector(c.type, rows, Shape::RANDOM, rng));
            cols[kWord] = ColumnVector(ColumnType::Text());
            for (uint32_t r = 0; r < rows; ++r) cols[kWord].AppendString(words[rng() % 7]);
            RowGroupMeta group;
            group.row_count = rows;
            for (const ColumnVector& col : cols) group.chunks.push_back(WriteChunk(&disk, col));
            meta.row_count += rows;
            meta.row_groups.push_back(std::move(group));
        }

        BufferPoolManager bpm(16, &disk, std::make_unique<LRUEvictionPolicy>(16));
        const ColumnarTable table(&bpm, meta);
        StringArena arena;
        for (size_t g = 0; g < table.RowGroupCount(); ++g) {
            const uint32_t rows = table.RowGroupRows(g);
            const std::vector<uint32_t> subset = RandomSubset(rows, 0.1, rng);
            const auto n = static_cast<uint32_t>(subset.size());
            for (uint32_t c = 0; c < meta.columns.size(); ++c) {
                SCOPED_TRACE("group " + std::to_string(g) + " column " + meta.columns[c].name);
                const ColumnType& type = meta.columns[c].type;
                ColumnVector expected(type);
                table.ReadChunk(g, c, expected);
                ColumnVector expected_subset(type);
                for (uint32_t r : subset) AppendRow(expected, r, expected_subset);
                EXPECT_EQ(&table.ChunkAt(g, c), &table.Meta().row_groups[g].chunks[c]);

                for (int pass = 0; pass < 2; ++pass) {
                    const uint32_t* list = pass == 0 ? nullptr : subset.data();
                    const uint32_t count = pass == 0 ? rows : n;
                    ColumnVector got(type);
                    switch (PhysicalKindOf(type)) {
                        case PhysicalKind::INT64: {
                            std::vector<int64_t> v(count);
                            table.ReadRows(g, c, list, count, v.data());
                            for (int64_t x : v) got.AppendInt(x);
                            break;
                        }
                        case PhysicalKind::INT128: {
                            std::vector<__int128> v(count);
                            table.ReadRows(g, c, list, count, v.data());
                            for (__int128 x : v) got.AppendWide(x);
                            break;
                        }
                        case PhysicalKind::DOUBLE: {
                            std::vector<double> v(count);
                            table.ReadRows(g, c, list, count, v.data());
                            for (double x : v) got.AppendDouble(x);
                            break;
                        }
                        case PhysicalKind::STRING: {
                            std::vector<StringRef> v(count);
                            table.ReadRows(g, c, list, count, v.data(), arena);
                            for (StringRef s : v) got.AppendString(View(s));
                            break;
                        }
                    }
                    EXPECT_TRUE(SameValues(got, pass == 0 ? expected : expected_subset)) << "pass " << pass;
                }
            }

            // The big group's word column is a dictionary: its codes index its entries.
            if (g == 0) ASSERT_EQ(table.ChunkAt(g, kWord).encoding, Encoding::DICTIONARY);
            if (table.ChunkAt(g, kWord).encoding != Encoding::DICTIONARY) continue;
            ColumnVector words_read(ColumnType::Text());
            table.ReadChunk(g, kWord, words_read);
            std::vector<StringRef> entries;
            table.ReadDictionary(g, kWord, entries, arena);
            std::vector<uint32_t> codes(n);
            table.ReadCodes(g, kWord, subset.data(), n, codes.data());
            for (uint32_t k = 0; k < n; ++k) {
                ASSERT_LT(codes[k], entries.size());
                EXPECT_EQ(View(entries[codes[k]]), words_read.GetString(subset[k]));
            }
        }

        // The same checks as ReadChunk: group, column and the column's type.
        std::vector<int64_t> ints(4);
        std::vector<StringRef> strings(4);
        std::vector<StringRef> entries;
        EXPECT_EQ(ThrownText([&] { (void)table.ChunkAt(2, 0); }), "orders: no row group 2");
        EXPECT_EQ(ThrownText([&] { (void)table.ChunkAt(0, 9); }), "orders: no column 9");
        EXPECT_EQ(ThrownText([&] { table.ReadRows(2, kOid, nullptr, 4, ints.data()); }), "orders: no row group 2");
        EXPECT_EQ(ThrownText([&] { table.ReadRows(0, 9, nullptr, 4, ints.data()); }), "orders: no column 9");
        EXPECT_EQ(ThrownText([&] { table.ReadRows(0, kWord, nullptr, 4, ints.data()); }).rfind("orders: column o_word is TEXT", 0), 0u);
        EXPECT_EQ(ThrownText([&] { table.ReadRows(0, kOid, nullptr, 4, strings.data(), arena); }).rfind("orders: column o_id is BIGINT", 0), 0u);
        EXPECT_EQ(ThrownText([&] { table.ReadDictionary(0, kOid, entries, arena); }).rfind("orders: column o_id is BIGINT", 0), 0u);
        std::vector<uint32_t> codes(4);
        ASSERT_NE(table.ChunkAt(0, kOid).encoding, Encoding::DICTIONARY);
        ASSERT_NE(table.ChunkAt(1, kWord).encoding, Encoding::DICTIONARY);
        EXPECT_EQ(ThrownText([&] { table.ReadCodes(0, kOid, nullptr, 4, codes.data()); }), "orders: column o_id has no dictionary in row group 0");
        EXPECT_EQ(ThrownText([&] { table.ReadDictionary(1, kWord, entries, arena); }), "orders: column o_word has no dictionary in row group 1");

        // Every read unpinned its pages.
        EXPECT_NO_THROW(bpm.DropAllPages());
    }
    std::filesystem::remove(path);
}

// The damaged pages EncodingTest forges for DecodeChunk, through the new decoders.
TEST(EncodingRowsTest, DamagedPagesThrow) {
    std::mt19937_64 rng(1);

    // A page of the wrong kind.
    const ColumnVector text = MakeVector(ColumnType::Text(), 100, Shape::RANDOM, rng);
    EncodedChunk kind = *TryEncodeChunk(text, Encoding::PLAIN);
    kind.pages[offsetof(ColumnarPageHeader, kind)] = static_cast<char>(PageKind::DICTIONARY);
    ExpectRowsDamaged(kind, text.Type(), 0);

    // A bit width past 64.
    ColumnVector small(ColumnType::BigInt());
    for (int64_t x : {1, 5, 9, 200}) small.AppendInt(x);
    EncodedChunk wide = *TryEncodeChunk(small, Encoding::BITPACK);
    wide.pages[offsetof(ColumnarPageHeader, bit_width)] = 65;
    ExpectRowsDamaged(wide, small.Type(), 1);

    // A value count too big for the page, with and without NULL bits.
    ColumnVector ints(ColumnType::BigInt());
    for (int64_t x : {5, 900, 17, 5, 900, 17, 42, 5, 900, 17}) ints.AppendInt(x);
    ColumnVector words(ColumnType::Text());
    for (const char* w : {"alpha", "bravo", "charlie", "delta", "echo", "alpha", "bravo"}) words.AppendString(w);
    const std::pair<const ColumnVector*, Encoding> cases[] = {
        {&ints, Encoding::PLAIN}, {&words, Encoding::PLAIN}, {&ints, Encoding::BITPACK},
        {&ints, Encoding::DICTIONARY}, {&words, Encoding::DICTIONARY}};
    for (const auto& [values, encoding] : cases) {
        for (bool nulls : {false, true}) {
            SCOPED_TRACE(ColumnTypeName(values->Type()) + " encoding " + std::to_string(static_cast<int>(encoding)) +
                         (nulls ? " with" : " without") + " NULL bits");
            EncodedChunk chunk = *TryEncodeChunk(*values, encoding);
            ASSERT_EQ(chunk.meta.page_first_rows.size(), 1u);
            ForgeHeader(chunk, chunk.meta.dict_page_count, 65535, nulls);
            chunk.meta.value_count = 65535;
            ExpectRowsDamaged(chunk, values->Type(), 65534);
        }
    }

    // 1,010 BIGINTs fit a page alone, but not after their NULL bits.
    ColumnVector longs = DistinctBigInts(1010);
    EncodedChunk crowded = *TryEncodeChunk(longs, Encoding::PLAIN);
    ASSERT_EQ(crowded.meta.page_first_rows.size(), 1u);
    ForgeHeader(crowded, 0, 1010, true);
    ExpectRowsDamaged(crowded, longs.Type(), 1009);

    // A string starting inside the header or the offset table.
    ColumnVector five(ColumnType::Text());
    for (const char* w : {"alpha", "bravo", "charlie", "delta", "echo"}) five.AppendString(w);
    for (uint16_t from : {uint16_t{4}, uint16_t{kPageHeaderSize + 2}}) {
        SCOPED_TRACE("string starts at byte " + std::to_string(from));
        EncodedChunk chunk = *TryEncodeChunk(five, Encoding::PLAIN);
        SetStringOffset(chunk, 0, 0, from);
        ExpectRowsDamaged(chunk, five.Type(), 0);
    }

    // The catalog's page starts disagree with a data page's own header.
    const ColumnVector distinct = DistinctBigInts(1500);
    EncodedChunk shifted = *TryEncodeChunk(distinct, Encoding::PLAIN);
    ASSERT_EQ(shifted.meta.page_first_rows, (std::vector<uint32_t>{0, 1020}));
    shifted.meta.page_first_rows[1] = 1015;
    ExpectRowsDamaged(shifted, distinct.Type(), 1015);

    // The same for a dictionary page; codes never read the dictionary.
    EncodedChunk coded = *TryEncodeChunk(distinct, Encoding::DICTIONARY);
    ASSERT_EQ(coded.meta.dict_page_first_entries, (std::vector<uint32_t>{0, 1020}));
    coded.meta.dict_page_first_entries[1] = 1015;
    ExpectRowsDamaged(coded, distinct.Type(), 1015, false);
}

// A code past the dictionary among many codes, read in bulk and by row.
TEST(EncodingRowsTest, DamagedCodeThrows) {
    ColumnVector ints(ColumnType::BigInt());
    ColumnVector words(ColumnType::Text());
    const char* const five[] = {"alpha", "bravo", "charlie", "delta", "echo"};
    for (int i = 0; i < 3000; ++i) {
        ints.AppendInt((i % 5) * 1000003LL);
        words.AppendString(five[i % 5]);
    }
    constexpr uint32_t kRow = 1234;
    for (const ColumnVector* v : {&ints, &words}) {
        SCOPED_TRACE(ColumnTypeName(v->Type()));
        EncodedChunk chunk = *TryEncodeChunk(*v, Encoding::DICTIONARY);
        const uint32_t data_page = chunk.meta.dict_page_count;
        ASSERT_EQ(chunk.meta.dict_size, 5u);
        ASSERT_EQ(chunk.meta.page_first_rows.size(), 1u) << "one page holds every code";
        ASSERT_EQ(HeaderOf(chunk, data_page).bit_width, 3);

        // Code 7 at kRow: three set bits, each past the 5 entries.
        char* page = chunk.pages.data() + static_cast<size_t>(data_page) * PAGE_SIZE;
        const size_t bit = kPageHeaderSize * 8 + static_cast<size_t>(kRow) * 3;
        for (size_t b = bit; b < bit + 3; ++b) page[b / 8] = static_cast<char>(page[b / 8] | (1 << (b % 8)));

        // Bulk codes first: without its check DecodeRows would index past its tables.
        CountingPages pages(chunk);
        std::vector<uint32_t> codes(3000);
        const std::string bulk = ThrownText([&] { DecodeCodes(chunk.meta, pages, nullptr, 3000, codes.data()); });
        ASSERT_EQ(bulk, "damaged column page: dictionary code 7 past the dictionary");
        ExpectRowsDamaged(chunk, v->Type(), kRow);

        // Rows on either side of it still decode.
        const std::vector<uint32_t> around = {0, kRow - 1, kRow + 1, 2999};
        StringArena arena;
        EXPECT_TRUE(SameValues(DecodeRowsToVector(chunk, v->Type(), around.data(), 4, pages, arena), DecodeEach(chunk, v->Type(), around)));
    }
}

// String offsets that go backwards mid-page, or past the page, are refused.
TEST(EncodingRowsTest, DamagedStringOffsetsThrow) {
    ColumnVector words(ColumnType::Text());
    for (int i = 0; i < 2000; ++i) words.AppendString("word" + std::to_string(i));
    const EncodedChunk clean = *TryEncodeChunk(words, Encoding::PLAIN);
    ASSERT_GE(clean.meta.page_first_rows.size(), 2u);
    const uint32_t on_first_page = clean.meta.page_first_rows[1];
    ASSERT_GT(on_first_page, 10u);

    // Slot 5 ends before it starts.
    EncodedChunk backwards = clean;
    SetStringOffset(backwards, 0, 6, static_cast<uint16_t>(StringOffsetAt(clean, 0, 5) - 1));
    ExpectRowsDamaged(backwards, words.Type(), 5);

    // The first page's last string ends past the page; later pages keep reads in bounds.
    EncodedChunk past = clean;
    SetStringOffset(past, 0, on_first_page, static_cast<uint16_t>(PAGE_SIZE + 100));
    ExpectRowsDamaged(past, words.Type(), on_first_page - 1);

    // The same through a dictionary page.
    EncodedChunk coded = *TryEncodeChunk(words, Encoding::DICTIONARY);
    ASSERT_GE(coded.meta.dict_page_count, 2u);
    SetStringOffset(coded, 0, 6, static_cast<uint16_t>(StringOffsetAt(coded, 0, 5) - 1));
    ExpectRowsDamaged(coded, words.Type(), 5, false);
    CountingPages pages(coded);
    std::vector<StringRef> entries;
    StringArena arena;
    EXPECT_TRUE(IsDamaged(ThrownText([&] { DecodeDictionary(coded.meta, pages, words.Type(), entries, arena); })));
}

// A page holding fewer values than its page list says is refused.
TEST(EncodingRowsTest, PageValueCountMismatchThrows) {
    const ColumnVector distinct = DistinctBigInts(1500);

    // A data page: rows on it, and all rows, are refused.
    EncodedChunk plain = *TryEncodeChunk(distinct, Encoding::PLAIN);
    ASSERT_EQ(plain.meta.page_first_rows, (std::vector<uint32_t>{0, 1020}));
    ForgeHeader(plain, 0, 1019, false);
    ExpectRowsDamaged(plain, distinct.Type(), 5);

    // A page of codes.
    EncodedChunk codes = *TryEncodeChunk(distinct, Encoding::DICTIONARY);
    ASSERT_EQ(codes.meta.page_first_rows.size(), 1u);
    ForgeHeader(codes, codes.meta.dict_page_count, 1499, false);
    ExpectRowsDamaged(codes, distinct.Type(), 5);

    // A dictionary page; row 0 holds entry 0, on the first dictionary page.
    EncodedChunk entries = *TryEncodeChunk(distinct, Encoding::DICTIONARY);
    ASSERT_EQ(entries.meta.dict_page_first_entries, (std::vector<uint32_t>{0, 1020}));
    ForgeHeader(entries, 0, 1019, false);
    ExpectRowsDamaged(entries, distinct.Type(), 0, false);

    // A string dictionary page, through DecodeDictionary too.
    ColumnVector words(ColumnType::Text());
    for (int i = 0; i < 3000; ++i) words.AppendString("entry" + std::to_string(i));
    EncodedChunk text = *TryEncodeChunk(words, Encoding::DICTIONARY);
    ASSERT_GE(text.meta.dict_page_count, 2u);
    ForgeHeader(text, 0, static_cast<uint16_t>(text.meta.dict_page_first_entries[1] - 1), false);
    ExpectRowsDamaged(text, words.Type(), 0, false);
    CountingPages pages(text);
    std::vector<StringRef> out;
    StringArena arena;
    EXPECT_TRUE(IsDamaged(ThrownText([&] { DecodeDictionary(text.meta, pages, words.Type(), out, arena); })));
}

// Broken page lists throw before any pin: bad counts, or values without a page.
TEST(EncodingRowsTest, BrokenPageListThrows) {
    const ColumnVector distinct = DistinctBigInts(1500);
    const EncodedChunk plain = *TryEncodeChunk(distinct, Encoding::PLAIN);
    const EncodedChunk coded = *TryEncodeChunk(distinct, Encoding::DICTIONARY);
    ASSERT_EQ(coded.meta.dict_page_count, 2u);
    ASSERT_EQ(coded.meta.page_first_rows.size(), 1u);

    // No data pages for 1,500 values: nothing would be written.
    EncodedChunk no_pages = plain;
    no_pages.meta.page_first_rows.clear();
    no_pages.meta.page_count = 0;
    no_pages.pages.clear();
    ExpectRowsDamaged(no_pages, distinct.Type(), 7);

    // Fewer pages than the lists name.
    EncodedChunk short_count = plain;
    short_count.meta.page_count = 1;
    ExpectRowsDamaged(short_count, distinct.Type(), 1400);

    // A dictionary page list shorter than its page count.
    EncodedChunk lost_list = coded;
    lost_list.meta.dict_page_first_entries.clear();
    ExpectRowsDamaged(lost_list, distinct.Type(), 7);

    // No dictionary pages for 1,500 entries: the data page alone, its codes intact.
    EncodedChunk no_dictionary = coded;
    no_dictionary.pages.erase(no_dictionary.pages.begin(), no_dictionary.pages.begin() + 2 * PAGE_SIZE);
    no_dictionary.meta.dict_page_first_entries.clear();
    no_dictionary.meta.dict_page_count = 0;
    no_dictionary.meta.page_count = 1;
    ExpectRowsDamaged(no_dictionary, distinct.Type(), 7);

    // Page 0 forged to agree with a start of 1: only the begin-at-0 check stops
    // row 0 from reading slot -1, far outside the page.
    EncodedChunk agreeing = plain;
    agreeing.meta.page_first_rows = {1, 1020};
    ColumnarPageHeader h;
    std::memcpy(&h, agreeing.pages.data(), sizeof(h));
    h.first_row = 1;
    h.value_count = 1019;
    std::memcpy(agreeing.pages.data(), &h, sizeof(h));
    ExpectRowsDamaged(agreeing, distinct.Type(), 0);

    // Starts that don't begin at 0, don't rise, or run past the values.
    for (const std::vector<uint32_t>& starts : {std::vector<uint32_t>{1, 1020}, std::vector<uint32_t>{0, 0},
                                                std::vector<uint32_t>{0, 1600}}) {
        SCOPED_TRACE("starts " + std::to_string(starts[0]) + ", " + std::to_string(starts[1]));
        EncodedChunk chunk = plain;
        chunk.meta.page_first_rows = starts;
        ExpectRowsDamaged(chunk, distinct.Type(), 1100);
    }
}

// An encoding byte no decoder knows is refused, not left unwritten.
TEST(EncodingRowsTest, UnknownEncodingThrows) {
    std::mt19937_64 rng(13);
    for (const ColumnType& type : {ColumnType::BigInt(), ColumnType::Numeric(38, 6), ColumnType::Double(), ColumnType::Text()}) {
        SCOPED_TRACE(ColumnTypeName(type));
        const ColumnVector v = MakeVector(type, 3000, Shape::RANDOM, rng);
        EncodedChunk chunk = *TryEncodeChunk(v, Encoding::PLAIN);
        chunk.meta.encoding = static_cast<Encoding>(9);
        ExpectRowsDamaged(chunk, type, 10);
        CountingPages pages(chunk);
        DecodeRowsDiscard(chunk.meta, pages, type, nullptr, 0);  // no rows asked, nothing read
        EXPECT_EQ(pages.TotalPins(), 0u);
    }
}

// The batched readers give the stored values and pin exactly the walk's pages, in order.
TEST(EncodingRowsTest, BatchedReadsMatchThePageWalk) {
    const std::string path = TempPath("batch") + ".db";
    const std::string trace = TempPath("batch_trace") + ".csv";
    std::filesystem::remove(path);
    std::mt19937_64 rng(77);
    {
        DiskManager disk(path);
        const BatchTable t = MakeBatchTable(disk, rng);
        const ChunkMeta& words = t.meta.row_groups[0].chunks[kBatchWord];
        ASSERT_EQ(words.encoding, Encoding::DICTIONARY);
        ASSERT_GT(words.dict_page_count, 2u);

        // Every row, a sparse list skipping pages, a dense list, and a middle band
        std::vector<std::vector<uint32_t>> lists = {{}, RandomSubset(kBatchRows, 0.002, rng), RandomSubset(kBatchRows, 0.3, rng), {}};
        for (uint32_t r = 30000; r < 31500; r += 3) lists[3].push_back(r);

        // A pool with room for full batches, and one whose batches are 4 pages
        const std::pair<bool, size_t> setups[] = {{false, kBatchPool}, {true, kBatchPool}, {false, 8}, {true, 8}};
        for (const auto& [concurrent, frames] : setups) {
            SCOPED_TRACE(std::string(concurrent ? "concurrent" : "plain") + ", " + std::to_string(frames) + " frames");
            BufferPoolManager bpm(frames, &disk, std::make_unique<LRUEvictionPolicy>(frames));
            bpm.SetConcurrent(concurrent);
            const ColumnarTable table(&bpm, t.meta);
            StringArena arena;

            for (uint32_t c = 0; c < t.meta.columns.size(); ++c) {
                const ColumnType& type = t.meta.columns[c].type;
                const ChunkMeta& chunk = table.ChunkAt(0, c);
                const EncodedChunk memory = EncodeChunk(t.values[c]);
                ASSERT_EQ(memory.meta.page_count, chunk.page_count);
                if (c != kBatchWord) ASSERT_GT(chunk.page_count, kPinBatch) << t.meta.columns[c].name;
                for (size_t l = 0; l < lists.size(); ++l) {
                    SCOPED_TRACE(t.meta.columns[c].name + " list " + std::to_string(l));
                    const uint32_t* rows = lists[l].empty() ? nullptr : lists[l].data();
                    const auto n = lists[l].empty() ? kBatchRows : static_cast<uint32_t>(lists[l].size());

                    OrderPages walk(memory);
                    StringArena scratch;
                    (void)DecodeRowsToVector(memory, type, rows, n, walk, scratch);

                    bpm.EnableTracing(trace);
                    const ColumnVector got = ReadRowsToVector(table, c, rows, n, arena);
                    bpm.DisableTracing();
                    EXPECT_TRUE(SameValues(got, RowsOf(t.values[c], lists[l])));
                    EXPECT_EQ(TracedPages(trace, chunk.first_page), walk.Order());
                    EXPECT_TRUE(NothingPinned(bpm));
                }
            }

            // Codes and the dictionary, through the same batches
            const EncodedChunk memory = EncodeChunk(t.values[kBatchWord]);
            for (size_t l = 0; l < lists.size(); ++l) {
                SCOPED_TRACE("codes, list " + std::to_string(l));
                const uint32_t* rows = lists[l].empty() ? nullptr : lists[l].data();
                const auto n = lists[l].empty() ? kBatchRows : static_cast<uint32_t>(lists[l].size());
                OrderPages walk(memory);
                std::vector<uint32_t> want(n), got(n);
                DecodeCodes(memory.meta, walk, rows, n, want.data());
                bpm.EnableTracing(trace);
                table.ReadCodes(0, kBatchWord, rows, n, got.data());
                bpm.DisableTracing();
                EXPECT_EQ(got, want);
                EXPECT_EQ(TracedPages(trace, words.first_page), walk.Order());
            }
            OrderPages walk(memory);
            std::vector<StringRef> want, got;
            StringArena scratch;
            DecodeDictionary(memory.meta, walk, ColumnType::Text(), want, scratch);
            bpm.EnableTracing(trace);
            table.ReadDictionary(0, kBatchWord, got, arena);
            bpm.DisableTracing();
            ASSERT_EQ(got.size(), want.size());
            for (size_t e = 0; e < got.size(); ++e) EXPECT_EQ(View(got[e]), View(want[e])) << "entry " << e;
            EXPECT_EQ(TracedPages(trace, words.first_page), walk.Order());

            EXPECT_TRUE(NothingPinned(bpm));
            EXPECT_NO_THROW(bpm.DropAllPages());
            bpm.SetConcurrent(false);
        }
    }
    std::filesystem::remove(path);
    std::filesystem::remove(trace);
}

// A checksum failure or a NULL partway through a batch throws, leaving no pin.
TEST(EncodingRowsTest, BatchedReadsLeaveNoPinOnRefusal) {
    const std::string path = TempPath("batch_refusal") + ".db";
    std::filesystem::remove(path);
    std::mt19937_64 rng(78);
    {
        DiskManager disk(path);
        const BatchTable t = MakeBatchTable(disk, rng, true);

        // Damage data page 5 of b_id on disk, behind its checksum
        const ChunkMeta& ids = t.meta.row_groups[0].chunks[kBatchId];
        const page_id_t damaged = ids.first_page + ids.dict_page_count + 5;
        std::vector<char> page(PAGE_SIZE);
        disk.ReadPage(damaged, page.data());
        page[100] = static_cast<char>(page[100] ^ 0x5A);
        disk.WritePage(damaged, page.data());

        // The rows on every b_id page but the damaged one
        std::vector<uint32_t> around;
        const uint32_t skip_from = ids.page_first_rows[5], skip_to = ids.page_first_rows[6];
        for (uint32_t r = 0; r < kBatchRows; ++r) {
            if (r < skip_from || r >= skip_to) around.push_back(r);
        }

        for (const bool concurrent : {false, true}) {
            SCOPED_TRACE(concurrent ? "concurrent" : "plain");
            BufferPoolManager bpm(kBatchPool, &disk, std::make_unique<LRUEvictionPolicy>(kBatchPool));
            bpm.SetConcurrent(concurrent);
            const ColumnarTable table(&bpm, t.meta);
            StringArena arena;

            EXPECT_EQ(ThrownText([&] { (void)ReadRowsToVector(table, kBatchId, nullptr, kBatchRows, arena); }),
                      "batch: page " + std::to_string(damaged) + " failed its checksum");
            EXPECT_TRUE(NothingPinned(bpm));
            EXPECT_TRUE(SameValues(ReadRowsToVector(table, kBatchId, around.data(), static_cast<uint32_t>(around.size()), arena),
                                   RowsOf(t.values[kBatchId], around)));

            EXPECT_EQ(ThrownText([&] { (void)ReadRowsToVector(table, kBatchRate, nullptr, kBatchRows, arena); }),
                      "damaged column chunk: row " + std::to_string(kBatchNullRow) + " is NULL");
            EXPECT_TRUE(NothingPinned(bpm));
            EXPECT_NO_THROW(bpm.DropAllPages());
            bpm.SetConcurrent(false);
        }
    }
    std::filesystem::remove(path);
}

// A pool smaller than a batch, or nearly full of pins, still reads every row.
TEST(EncodingRowsTest, BatchedReadsInASmallPool) {
    const std::string path = TempPath("batch_small") + ".db";
    std::filesystem::remove(path);
    std::mt19937_64 rng(79);
    {
        DiskManager disk(path);
        const BatchTable t = MakeBatchTable(disk, rng);
        const ChunkMeta& notes = t.meta.row_groups[0].chunks[kBatchNote];
        const ChunkMeta& ids = t.meta.row_groups[0].chunks[kBatchId];

        for (const bool concurrent : {false, true}) {
            SCOPED_TRACE(concurrent ? "concurrent" : "plain");
            {
                BufferPoolManager bpm(8, &disk, std::make_unique<LRUEvictionPolicy>(8));
                bpm.SetConcurrent(concurrent);
                const ColumnarTable table(&bpm, t.meta);
                StringArena arena;
                for (const uint32_t c : {kBatchId, kBatchWord}) {
                    EXPECT_TRUE(SameValues(ReadRowsToVector(table, c, nullptr, kBatchRows, arena), t.values[c])) << "column " << c;
                }
                EXPECT_TRUE(NothingPinned(bpm));
                bpm.SetConcurrent(false);
            }

            // Two frames free: each batch is a short prefix; the reader takes the rest later
            const size_t frames = 128;
            ASSERT_GT(notes.page_count, frames);
            BufferPoolManager bpm(frames, &disk, std::make_unique<LRUEvictionPolicy>(frames));
            bpm.SetConcurrent(concurrent);
            const ColumnarTable table(&bpm, t.meta);
            StringArena arena;
            for (page_id_t i = 0; i + 2 < frames; ++i) ASSERT_NE(bpm.FetchPage(notes.first_page + i), nullptr);
            EXPECT_TRUE(SameValues(ReadRowsToVector(table, kBatchId, nullptr, kBatchRows, arena), t.values[kBatchId]));

            // No frame at all is today's error
            for (page_id_t i = frames - 2; i < frames; ++i) ASSERT_NE(bpm.FetchPage(notes.first_page + i), nullptr);
            EXPECT_EQ(ThrownText([&] { (void)ReadRowsToVector(table, kBatchId, nullptr, kBatchRows, arena); }),
                      "batch: no free frame for page " + std::to_string(ids.first_page));
            for (page_id_t i = 0; i < frames; ++i) EXPECT_TRUE(bpm.UnpinPage(notes.first_page + i, false));
            EXPECT_TRUE(NothingPinned(bpm));
            EXPECT_NO_THROW(bpm.DropAllPages());
            bpm.SetConcurrent(false);
        }
    }
    std::filesystem::remove(path);
}


// Many workers on a small pool still read:
// their batches together keep to half the pool.
TEST(EncodingRowsTest, BatchedReadsManyWorkersOnASmallPool) {
    constexpr size_t kFrames = 64;
    constexpr unsigned kWorkers = 48;  // one page each fits, two each would not
    const std::string path = TempPath("batch_workers") + ".db";
    std::filesystem::remove(path);
    std::mt19937_64 rng(80);
    {
        DiskManager disk(path);
        const BatchTable t = MakeBatchTable(disk, rng);
        BufferPoolManager bpm(kFrames, &disk, std::make_unique<LRUEvictionPolicy>(kFrames));
        const ColumnarTable table(&bpm, t.meta);
        const uint32_t columns[] = {kBatchId, kBatchWide, kBatchRate};

        std::atomic<int> refused{0}, wrong{0};
        bpm.SetConcurrent(true, kWorkers);
        EXPECT_EQ(bpm.ConcurrentWorkers(), kWorkers);
        std::vector<std::thread> threads;
        for (unsigned w = 0; w < kWorkers; ++w) {
            threads.emplace_back([&, w] {
                StringArena arena;
                for (unsigned round = 0; round < 3; ++round) {
                    const uint32_t c = columns[(w + round) % 3];
                    try {
                        if (!SameValues(ReadRowsToVector(table, c, nullptr, kBatchRows, arena), t.values[c])) wrong.fetch_add(1);
                    } catch (const DatabaseException&) {
                        refused.fetch_add(1);
                    }
                }
            });
        }
        for (auto& thread : threads) thread.join();
        bpm.SetConcurrent(false);

        EXPECT_EQ(bpm.ConcurrentWorkers(), 1u);
        EXPECT_EQ(refused.load(), 0);
        EXPECT_EQ(wrong.load(), 0);
        EXPECT_TRUE(NothingPinned(bpm));
        EXPECT_NO_THROW(bpm.DropAllPages());
    }
    std::filesystem::remove(path);
}

}  // namespace shilmandb
