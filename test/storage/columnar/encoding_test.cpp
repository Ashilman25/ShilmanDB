#include <gtest/gtest.h>
#include "common/exception.hpp"
#include "storage/columnar/crc32c.hpp"
#include "storage/columnar/encoding.hpp"
#include "storage/columnar/string_arena.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace shilmandb {
namespace {

constexpr Encoding kAllEncodings[] = {Encoding::CONSTANT, Encoding::PLAIN, Encoding::BITPACK,
                                      Encoding::DICTIONARY};

// Hands out a chunk's pages, checking pins are undone, one page at a time.
// A big-value page may be pinned beside its data page, one at a time.
class MemoryPages : public ChunkPages {
public:
    explicit MemoryPages(const EncodedChunk& chunk)
        : chunk_(chunk), first_big_(chunk.meta.big_page_count <= chunk.meta.page_count ? chunk.meta.page_count - chunk.meta.big_page_count : 0) {}
    ~MemoryPages() override {
        EXPECT_EQ(pins_, unpins_);
        EXPECT_EQ(pinned_, kNone);
        EXPECT_EQ(pinned_big_, kNone);
    }

    const char* Pin(uint32_t index) override {
        uint32_t& slot = index >= first_big_ ? pinned_big_ : pinned_;
        EXPECT_EQ(slot, kNone) << (index >= first_big_ ? "two big-value pages pinned at once" : "two pages pinned at once");
        EXPECT_LT(index, chunk_.meta.page_count);
        slot = index;
        ++pins_;
        if (index >= first_big_) ++big_pins_;
        return chunk_.pages.data() + static_cast<size_t>(index) * PAGE_SIZE;
    }
    void Unpin(uint32_t index) override {
        uint32_t& slot = index >= first_big_ ? pinned_big_ : pinned_;
        EXPECT_EQ(slot, index);
        slot = kNone;
        ++unpins_;
    }

    // How many times a big-value page was pinned.
    [[nodiscard]] uint64_t BigPins() const { return big_pins_; }

private:
    static constexpr uint32_t kNone = UINT32_MAX;
    const EncodedChunk& chunk_;
    uint32_t first_big_;
    uint32_t pinned_{kNone};
    uint32_t pinned_big_{kNone};
    uint64_t pins_{0};
    uint64_t unpins_{0};
    uint64_t big_pins_{0};
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

ColumnVector Decode(const EncodedChunk& chunk, const ColumnType& type) {
    ColumnVector out(type);
    MemoryPages pages(chunk);
    DecodeChunk(chunk.meta, pages, out);
    return out;
}

// Decodes every row on its own, in order.
ColumnVector DecodeEachRow(const EncodedChunk& chunk, const ColumnType& type) {
    ColumnVector out(type);
    MemoryPages pages(chunk);
    for (uint32_t row = 0; row < chunk.meta.value_count; ++row) DecodeValue(chunk.meta, pages, row, out);
    return out;
}

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
        case ColumnTypeId::TEXT:    v.AppendString(RandomString(300, rng)); break;
    }
}

ColumnVector RandomVector(const ColumnType& type, size_t n, double null_rate, std::mt19937_64& rng) {
    ColumnVector v(type);
    std::bernoulli_distribution is_null(null_rate);
    for (size_t i = 0; i < n; ++i) {
        if (is_null(rng)) {
            v.AppendNull();
        } else {
            AppendRandom(v, rng);
        }
    }
    return v;
}

std::vector<ColumnType> EveryType() {
    return {ColumnType::Boolean(),      ColumnType::SmallInt(),      ColumnType::Integer(),
            ColumnType::BigInt(),       ColumnType::Numeric(15, 2),  ColumnType::Numeric(38, 6),
            ColumnType::Double(),       ColumnType::Date(),          ColumnType::Timestamp(),
            ColumnType::Varchar(25),    ColumnType::Text()};
}

ColumnVector Ints(const ColumnType& type, std::initializer_list<int64_t> values) {
    ColumnVector v(type);
    for (int64_t x : values) v.AppendInt(x);
    return v;
}

ColumnVector Strings(std::initializer_list<std::string> values) {
    ColumnVector v(ColumnType::Text());
    for (const auto& s : values) v.AppendString(s);
    return v;
}

ColumnarPageHeader HeaderOf(const EncodedChunk& chunk, uint32_t index) {
    ColumnarPageHeader h;
    std::memcpy(&h, chunk.pages.data() + static_cast<size_t>(index) * PAGE_SIZE, sizeof(h));
    return h;
}

ColumnVector Wides(const ColumnType& type, std::initializer_list<__int128> values) {
    ColumnVector v(type);
    for (__int128 x : values) v.AppendWide(x);
    return v;
}

// Rewrites one page's value count and NULL flag, then reseals so checksums pass.
void ForgeHeader(EncodedChunk& chunk, uint32_t index, uint16_t count, bool nulls) {
    char* page = chunk.pages.data() + static_cast<size_t>(index) * PAGE_SIZE;
    ColumnarPageHeader h;
    std::memcpy(&h, page, sizeof(h));
    h.value_count = count;
    if (nulls) h.flags = static_cast<uint8_t>(h.flags | kPageHasNulls);
    std::memcpy(page, &h, sizeof(h));
    SealPages(chunk.pages.data(), chunk.meta.page_count, 7);
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

// Both the whole decode and one row's decode refuse the chunk as damaged.
void ExpectDamaged(const EncodedChunk& chunk, const ColumnType& type, uint32_t row) {
    MemoryPages pages(chunk);
    const std::string whole = ThrownText([&] {
        ColumnVector out(type);
        DecodeChunk(chunk.meta, pages, out);
    });
    EXPECT_EQ(whole.rfind("damaged column page", 0), 0u) << "DecodeChunk: " << whole;
    const std::string one = ThrownText([&] {
        ColumnVector out(type);
        DecodeValue(chunk.meta, pages, row, out);
    });
    EXPECT_EQ(one.rfind("damaged column page", 0), 0u) << "DecodeValue: " << one;
}

// Every encoding that applies gives back the input, whole and row by row.
void ExpectEveryEncodingRoundTrips(const ColumnVector& v) {
    for (Encoding e : kAllEncodings) {
        auto chunk = TryEncodeChunk(v, e);
        if (!chunk) continue;
        EXPECT_EQ(chunk->meta.encoding, e);
        EXPECT_TRUE(SameValues(Decode(*chunk, v.Type()), v)) << "encoding " << static_cast<int>(e);
        EXPECT_TRUE(SameValues(DecodeEachRow(*chunk, v.Type()), v)) << "encoding " << static_cast<int>(e);
    }
}

}  // namespace

TEST(EncodingTest, RoundTripEveryKind) {
    std::mt19937_64 rng(2026);
    for (const auto& type : EveryType()) {
        for (size_t n : {1u, 2u, 1000u, 65536u}) {
            for (double null_rate : {0.0, 0.1}) {
                SCOPED_TRACE(ColumnTypeName(type) + " n=" + std::to_string(n) +
                             " nulls=" + std::to_string(null_rate));
                const ColumnVector v = RandomVector(type, n, null_rate, rng);

                const EncodedChunk chunk = EncodeChunk(v);
                EXPECT_EQ(chunk.meta.value_count, n);
                EXPECT_EQ(chunk.meta.null_count, v.NullCount());
                EXPECT_EQ(chunk.pages.size(), static_cast<size_t>(chunk.meta.page_count) * PAGE_SIZE);
                EXPECT_EQ(chunk.meta.first_page, INVALID_PAGE_ID);
                EXPECT_TRUE(SameValues(Decode(chunk, type), v));

                for (Encoding e : kAllEncodings) {
                    auto forced = TryEncodeChunk(v, e);
                    if (!forced) continue;
                    EXPECT_LE(chunk.meta.page_count, forced->meta.page_count) << "not the smallest";
                    EXPECT_TRUE(SameValues(Decode(*forced, type), v)) << "encoding " << static_cast<int>(e);
                }
            }
        }
    }
}

TEST(EncodingTest, BitWidthsZeroTo64) {
    std::mt19937_64 rng(64);
    for (uint32_t w = 0; w <= 64; ++w) {
        SCOPED_TRACE("width " + std::to_string(w));
        const uint64_t mask = w == 64 ? ~uint64_t{0} : (uint64_t{1} << w) - 1;
        const int64_t lo = w == 0 ? 42 : (w == 64 ? INT64_MIN : -static_cast<int64_t>(mask / 2) - 1);
        const auto hi = static_cast<int64_t>(static_cast<uint64_t>(lo) + mask);

        ColumnVector v(ColumnType::BigInt());
        v.AppendInt(lo);
        v.AppendInt(hi);
        for (int i = 0; i < 3000; ++i) {
            v.AppendInt(static_cast<int64_t>(static_cast<uint64_t>(lo) + (mask == 0 ? 0 : rng() % mask)));
        }
        auto chunk = TryEncodeChunk(v, Encoding::BITPACK);
        ASSERT_TRUE(chunk.has_value());
        ASSERT_GE(chunk->meta.page_count, 1u);
        EXPECT_EQ(HeaderOf(*chunk, 0).bit_width, w);
        EXPECT_TRUE(SameValues(Decode(*chunk, v.Type()), v));
        EXPECT_TRUE(SameValues(DecodeEachRow(*chunk, v.Type()), v));

        // The same widths for a wide NUMERIC, offset far from zero.
        ColumnVector wide(ColumnType::Numeric(38, 0));
        const __int128 base = -Pow10(30);
        wide.AppendWide(base);
        wide.AppendWide(base + static_cast<__int128>(mask));
        for (int i = 0; i < 500; ++i) {
            wide.AppendWide(base + static_cast<__int128>(mask == 0 ? 0 : rng() % mask));
        }
        auto wide_chunk = TryEncodeChunk(wide, Encoding::BITPACK);
        ASSERT_TRUE(wide_chunk.has_value());
        EXPECT_EQ(HeaderOf(*wide_chunk, 0).bit_width, w);
        EXPECT_TRUE(SameValues(Decode(*wide_chunk, wide.Type()), wide));
        EXPECT_TRUE(SameValues(DecodeEachRow(*wide_chunk, wide.Type()), wide));
    }
}

TEST(EncodingTest, IntegerExtremes) {
    ExpectEveryEncodingRoundTrips(Ints(ColumnType::BigInt(), {INT64_MIN, INT64_MAX, 0}));
    ExpectEveryEncodingRoundTrips(Ints(ColumnType::Date(), {-1, -719528}));
    ExpectEveryEncodingRoundTrips(Ints(ColumnType::Numeric(15, 2), {-999999999999999, 999999999999999}));

    const auto extremes = Ints(ColumnType::BigInt(), {INT64_MIN, INT64_MAX, 0});
    auto packed = TryEncodeChunk(extremes, Encoding::BITPACK);
    ASSERT_TRUE(packed.has_value());
    EXPECT_EQ(HeaderOf(*packed, 0).bit_width, 64);
    EXPECT_EQ(packed->meta.page_count, 1u) << "the extremes share one page";
    EXPECT_TRUE(SameValues(Decode(EncodeChunk(extremes), extremes.Type()), extremes));

    ColumnVector wide(ColumnType::Numeric(38, 0));
    wide.AppendWide(-Pow10(37));
    wide.AppendWide(Pow10(37));
    ExpectEveryEncodingRoundTrips(wide);
    EXPECT_FALSE(TryEncodeChunk(wide, Encoding::BITPACK).has_value()) << "a range over 64 bits can't pack";
    EXPECT_TRUE(SameValues(Decode(EncodeChunk(wide), wide.Type()), wide));
}

// int128 NUMERIC at the edge of 38 digits, both too wide to pack and packing.
TEST(EncodingTest, WideNumericAtItsLimits) {
    const ColumnType type = ColumnType::Numeric(38, 0);
    const __int128 top = Pow10(38) - 1;
    const auto extremes = Wides(type, {-top, top, 0, -top});
    ExpectEveryEncodingRoundTrips(extremes);
    EXPECT_FALSE(TryEncodeChunk(extremes, Encoding::BITPACK).has_value());
    EXPECT_TRUE(SameValues(Decode(EncodeChunk(extremes), type), extremes));

    // Near either limit, a range under 2^64 packs against an int128 frame.
    std::mt19937_64 rng(38);
    for (const __int128 edge : {top, -top}) {
        ColumnVector near(type);
        near.AppendWide(edge);
        for (int i = 0; i < 2000; ++i) {
            const auto step = static_cast<__int128>(rng() >> 1);
            near.AppendWide(edge > 0 ? edge - step : edge + step);
        }
        auto packed = TryEncodeChunk(near, Encoding::BITPACK);
        ASSERT_TRUE(packed.has_value());
        EXPECT_EQ(HeaderOf(*packed, 0).bit_width, 63);
        ExpectEveryEncodingRoundTrips(near);
    }
}

// NUMERIC(18,s) is 64-bit; values near -(10^18 - 1) pack and come back exact.
TEST(EncodingTest, NegativeNumeric18Packs) {
    const ColumnType type = ColumnType::Numeric(18, 4);
    const auto low = static_cast<int64_t>(-(Pow10(18) - 1));
    std::mt19937_64 rng(18);
    ColumnVector near(type);
    near.AppendInt(low);
    for (int i = 0; i < 2000; ++i) near.AppendInt(low + static_cast<int64_t>(rng() % 1000000));
    ASSERT_TRUE(TryEncodeChunk(near, Encoding::BITPACK).has_value());
    ExpectEveryEncodingRoundTrips(near);

    const auto both = Ints(type, {low, -low, 0, low + 1});
    auto packed = TryEncodeChunk(both, Encoding::BITPACK);
    ASSERT_TRUE(packed.has_value());
    EXPECT_EQ(HeaderOf(*packed, 0).bit_width, 61);
    ExpectEveryEncodingRoundTrips(both);
}

TEST(EncodingTest, DecodeValueMatchesDecodeChunk) {
    std::mt19937_64 rng(5);
    bool seen[5] = {};

    // Few distinct values with NULLs, so PLAIN, BITPACK and DICTIONARY all apply.
    for (const auto& type : {ColumnType::BigInt(), ColumnType::Integer(), ColumnType::Numeric(38, 2),
                             ColumnType::Double(), ColumnType::Text()}) {
        SCOPED_TRACE(ColumnTypeName(type));
        ColumnVector pool = RandomVector(type, 40, 0.0, rng);
        if (type.id == ColumnTypeId::NUMERIC) {
            pool = ColumnVector(type);
            for (int i = 0; i < 40; ++i) pool.AppendWide(Pow10(25) + static_cast<__int128>(rng() % 1000000));
        }
        ColumnVector v(type);
        for (int i = 0; i < 5000; ++i) {
            const size_t pick = rng() % 44;
            if (pick >= 40) {
                v.AppendNull();
                continue;
            }
            switch (v.Kind()) {
                case PhysicalKind::INT64:  v.AppendInt(pool.GetInt(pick)); break;
                case PhysicalKind::INT128: v.AppendWide(pool.GetWide(pick)); break;
                case PhysicalKind::DOUBLE: v.AppendDouble(pool.GetDouble(pick)); break;
                case PhysicalKind::STRING: v.AppendString(pool.GetString(pick)); break;
            }
        }
        for (Encoding e : kAllEncodings) {
            auto chunk = TryEncodeChunk(v, e);
            if (!chunk) continue;
            seen[static_cast<int>(e)] = true;
            const ColumnVector whole = Decode(*chunk, type);
            EXPECT_TRUE(SameValues(whole, v));
            EXPECT_TRUE(SameValues(DecodeEachRow(*chunk, type), whole)) << "encoding " << static_cast<int>(e);
        }
    }

    // CONSTANT: one repeated value, and every row NULL.
    ColumnVector same(ColumnType::Text());
    for (int i = 0; i < 300; ++i) same.AppendString("same");
    ColumnVector nulls(ColumnType::Integer());
    for (int i = 0; i < 300; ++i) nulls.AppendNull();
    for (const auto* v : {&same, &nulls}) {
        auto chunk = TryEncodeChunk(*v, Encoding::CONSTANT);
        ASSERT_TRUE(chunk.has_value());
        seen[static_cast<int>(Encoding::CONSTANT)] = true;
        EXPECT_EQ(chunk->meta.page_count, 0u);
        EXPECT_TRUE(SameValues(DecodeEachRow(*chunk, v->Type()), Decode(*chunk, v->Type())));
        EXPECT_TRUE(SameValues(Decode(*chunk, v->Type()), *v));
    }
    for (Encoding e : kAllEncodings) EXPECT_TRUE(seen[static_cast<int>(e)]) << static_cast<int>(e);
}

TEST(EncodingTest, PicksSmallest) {
    ColumnVector equal(ColumnType::Integer());
    for (int i = 0; i < 1000; ++i) equal.AppendInt(7);
    const auto constant = EncodeChunk(equal);
    EXPECT_EQ(constant.meta.encoding, Encoding::CONSTANT);
    EXPECT_EQ(constant.meta.page_count, 0u);
    EXPECT_TRUE(constant.meta.constant.has_value());

    std::mt19937_64 rng(3);
    ColumnVector small(ColumnType::Integer());
    for (int i = 0; i < 65536; ++i) small.AppendInt(static_cast<int64_t>(rng() % 100));
    EXPECT_EQ(EncodeChunk(small).meta.encoding, Encoding::BITPACK);

    const char* words[] = {"alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"};
    ColumnVector seven(ColumnType::Text());
    for (int i = 0; i < 65536; ++i) seven.AppendString(words[rng() % 7]);
    const auto dict = EncodeChunk(seven);
    EXPECT_EQ(dict.meta.encoding, Encoding::DICTIONARY);
    EXPECT_EQ(dict.meta.dict_size, 7u);
    EXPECT_EQ(dict.meta.dict_page_count, 1u);

    ColumnVector unique(ColumnType::Text());
    for (int i = 0; i < 65536; ++i) {
        std::string s = std::to_string(i);
        s.resize(40, 'x');
        unique.AppendString(s);
    }
    EXPECT_EQ(EncodeChunk(unique).meta.encoding, Encoding::PLAIN);
}

// A page header counts values in 16 bits, so no page holds 65,536.
TEST(EncodingTest, PageHoldsAtMost65535Values) {
    ColumnVector zeros(ColumnType::BigInt());
    for (int i = 0; i < 65536; ++i) zeros.AppendInt(0);
    for (Encoding e : {Encoding::BITPACK, Encoding::DICTIONARY}) {
        auto chunk = TryEncodeChunk(zeros, e);
        ASSERT_TRUE(chunk.has_value());
        ASSERT_EQ(chunk->meta.page_first_rows.size(), 2u);
        EXPECT_EQ(chunk->meta.page_first_rows[1], 65535u);
        EXPECT_TRUE(SameValues(Decode(*chunk, zeros.Type()), zeros));
    }
}

TEST(EncodingTest, MinMaxSkipsNulls) {
    ColumnVector ints(ColumnType::BigInt());
    ints.AppendNull();
    ints.AppendInt(5);
    ints.AppendInt(-3);
    ints.AppendNull();
    ints.AppendInt(9);
    const auto meta = EncodeChunk(ints).meta;
    ASSERT_TRUE(meta.min.has_value() && meta.max.has_value());
    EXPECT_EQ(std::get<int64_t>(*meta.min), -3);
    EXPECT_EQ(std::get<int64_t>(*meta.max), 9);
    EXPECT_EQ(meta.null_count, 2u);

    ColumnVector text(ColumnType::Text());
    text.AppendString("b");
    text.AppendNull();
    text.AppendString("a");
    text.AppendString("c");
    const auto text_meta = EncodeChunk(text).meta;
    ASSERT_TRUE(text_meta.min.has_value() && text_meta.max.has_value());
    EXPECT_EQ(std::get<std::string>(*text_meta.min), "a");
    EXPECT_EQ(std::get<std::string>(*text_meta.max), "c");

    // Any NaN means no min or max; all NULLs means none either.
    ColumnVector doubles(ColumnType::Double());
    doubles.AppendDouble(std::nan(""));
    doubles.AppendDouble(2.5);
    doubles.AppendNull();
    doubles.AppendDouble(-1.0);
    const auto double_meta = EncodeChunk(doubles).meta;
    EXPECT_FALSE(double_meta.min.has_value());
    EXPECT_FALSE(double_meta.max.has_value());

    ColumnVector all_null(ColumnType::Integer());
    all_null.AppendNull();
    all_null.AppendNull();
    const auto null_meta = EncodeChunk(all_null).meta;
    EXPECT_EQ(null_meta.encoding, Encoding::CONSTANT);
    EXPECT_FALSE(null_meta.min.has_value());
    EXPECT_FALSE(null_meta.max.has_value());
    EXPECT_FALSE(null_meta.constant.has_value());
    EXPECT_TRUE(SameValues(Decode(EncodeChunk(all_null), all_null.Type()), all_null));
}

// Value's <= and >= match NaN, so a chunk holding one must never be skipped.
TEST(EncodingTest, NaNMeansNoMinMax) {
    ColumnVector with_nan(ColumnType::Double());
    for (double x : {10.0, 20.0, std::nan("")}) with_nan.AppendDouble(x);
    for (Encoding e : kAllEncodings) {
        auto chunk = TryEncodeChunk(with_nan, e);
        if (!chunk) continue;
        EXPECT_FALSE(chunk->meta.min.has_value()) << static_cast<int>(e);
        EXPECT_FALSE(chunk->meta.max.has_value()) << static_cast<int>(e);
    }
    ExpectEveryEncodingRoundTrips(with_nan);

    ColumnVector numbers(ColumnType::Double());
    for (double x : {10.0, 20.0, -0.5}) numbers.AppendDouble(x);
    const auto kept = EncodeChunk(numbers).meta;
    ASSERT_TRUE(kept.min.has_value() && kept.max.has_value());
    EXPECT_EQ(std::get<double>(*kept.min), -0.5);
    EXPECT_EQ(std::get<double>(*kept.max), 20.0);

    // One repeated value is still CONSTANT, with or without NaN.
    ColumnVector same(ColumnType::Double());
    for (int i = 0; i < 100; ++i) same.AppendDouble(2.5);
    const auto same_meta = EncodeChunk(same).meta;
    EXPECT_EQ(same_meta.encoding, Encoding::CONSTANT);
    ASSERT_TRUE(same_meta.min.has_value() && same_meta.max.has_value());
    EXPECT_EQ(std::get<double>(*same_meta.min), 2.5);
    EXPECT_EQ(std::get<double>(*same_meta.max), 2.5);

    ColumnVector all_nan(ColumnType::Double());
    for (int i = 0; i < 100; ++i) all_nan.AppendDouble(std::numeric_limits<double>::quiet_NaN());
    const auto nan_meta = EncodeChunk(all_nan).meta;
    EXPECT_EQ(nan_meta.encoding, Encoding::CONSTANT);
    EXPECT_FALSE(nan_meta.min.has_value());
    EXPECT_TRUE(SameValues(Decode(EncodeChunk(all_nan), all_nan.Type()), all_nan));
}

TEST(EncodingTest, LongStringsHaveNoMinMax) {
    const auto meta = EncodeChunk(Strings({"short", std::string(65, 'x')})).meta;
    EXPECT_FALSE(meta.min.has_value());
    EXPECT_FALSE(meta.max.has_value());

    const auto kept = EncodeChunk(Strings({"short", std::string(64, 'x')})).meta;
    ASSERT_TRUE(kept.min.has_value() && kept.max.has_value());
    EXPECT_EQ(std::get<std::string>(*kept.min), "short");
    EXPECT_EQ(std::get<std::string>(*kept.max), std::string(64, 'x'));
}

// Doubles count as distinct by their bits, so -0.0 and NaN payloads survive.
TEST(EncodingTest, DoubleBitsKept) {
    ColumnVector v(ColumnType::Double());
    for (int i = 0; i < 50; ++i) {
        v.AppendDouble(0.0);
        v.AppendDouble(-0.0);
        v.AppendDouble(std::numeric_limits<double>::quiet_NaN());
        v.AppendDouble(-std::numeric_limits<double>::infinity());
    }
    ExpectEveryEncodingRoundTrips(v);
    EXPECT_EQ(TryEncodeChunk(v, Encoding::DICTIONARY)->meta.dict_size, 4u);
    EXPECT_FALSE(TryEncodeChunk(v, Encoding::CONSTANT).has_value());
}

TEST(EncodingTest, Utf8RoundTrip) {
    ColumnVector v(ColumnType::Varchar(10000));
    const std::string samples[] = {"caf\xC3\xA9", "\xE2\x82\xAC" "5", "\xF0\x9D\x84\x9E", "\xD0\x96\xE2\x82\xAC\xF0\x9F\x98\x80"};
    for (int i = 0; i < 200; ++i) v.AppendString(samples[i % 4]);

    std::string big;
    while (big.size() + 3 <= 7998) big += "\xE2\x82\xAC";  // the euro sign, 3 bytes
    big += "\xC3\xA9";
    big.resize(8000, 'z');
    ASSERT_EQ(big.size(), 8000u);
    v.AppendString(big);
    v.AppendNull();
    v.AppendString(samples[2]);

    for (Encoding e : {Encoding::PLAIN, Encoding::DICTIONARY}) {
        auto chunk = TryEncodeChunk(v, e);
        ASSERT_TRUE(chunk.has_value());
        EXPECT_TRUE(SameValues(Decode(*chunk, v.Type()), v)) << static_cast<int>(e);
        EXPECT_TRUE(SameValues(DecodeEachRow(*chunk, v.Type()), v)) << static_cast<int>(e);
    }
}

namespace {

// A string of n bytes whose content depends on n, so two lengths never match.
std::string S(size_t n) {
    std::string s(n, ' ');
    for (size_t i = 0; i < n; ++i) s[i] = static_cast<char>('a' + (i * 7 + n) % 26);
    return s;
}

// A TEXT vector of these strings.
ColumnVector StringsOf(const std::vector<std::string>& values) {
    ColumnVector v(ColumnType::Text());
    for (const auto& s : values) v.AppendString(s);
    return v;
}

// The listed entries of values, in order.
std::vector<std::string> Pick(const std::vector<std::string>& values, const std::vector<uint32_t>& rows) {
    std::vector<std::string> out;
    for (uint32_t r : rows) out.push_back(values[r]);
    return out;
}

// Every string of a chunk without NULLs, by DecodeChunk.
std::vector<std::string> DecodeAll(const EncodedChunk& chunk) {
    const ColumnVector v = Decode(chunk, ColumnType::Text());
    std::vector<std::string> out;
    for (size_t i = 0; i < v.Size(); ++i) out.emplace_back(v.GetString(i));
    return out;
}

// The listed rows by DecodeRows; nullptr rows means all of them.
std::vector<std::string> DecodeRowsStrings(const EncodedChunk& chunk, const std::vector<uint32_t>* rows) {
    MemoryPages pages(chunk);
    StringArena arena;
    const uint32_t n = rows != nullptr ? static_cast<uint32_t>(rows->size()) : chunk.meta.value_count;
    std::vector<StringRef> refs(n);
    DecodeRows(chunk.meta, pages, ColumnType::Text(), rows != nullptr ? rows->data() : nullptr, n, refs.data(), arena);
    std::vector<std::string> out;
    for (StringRef s : refs) out.emplace_back(View(s));
    return out;
}
std::vector<std::string> DecodeRowsStrings(const EncodedChunk& chunk, const std::vector<uint32_t>& rows) {
    return DecodeRowsStrings(chunk, &rows);
}

// One row by DecodeValue.
std::string DecodeOne(const EncodedChunk& chunk, uint32_t row) {
    ColumnVector out(ColumnType::Text());
    MemoryPages pages(chunk);
    DecodeValue(chunk.meta, pages, row, out);
    return std::string(out.GetString(0));
}

// Bytes [at, at + n) of page index of the chunk.
std::string PageBytes(const EncodedChunk& chunk, uint32_t index, size_t at, size_t n) {
    return std::string(chunk.pages.data() + static_cast<size_t>(index) * PAGE_SIZE + at, n);
}

// Where slot i's bytes start, on a page with one big-bits byte and no NULLs.
size_t SlotStart(const EncodedChunk& chunk, uint32_t page, uint32_t i) {
    uint16_t offset = 0;
    std::memcpy(&offset, chunk.pages.data() + static_cast<size_t>(page) * PAGE_SIZE + kPageHeaderSize + 1 + 2 * i, sizeof(offset));
    return offset;
}

// Many copies of three words, and one big value.
ColumnVector RepeatedWithOneBig() {
    const char* words[] = {"red", "green", "blue"};
    ColumnVector v(ColumnType::Text());
    for (int i = 0; i < 5000; ++i) v.AppendString(i == 2500 ? S(10000) : words[i % 3]);
    return v;
}

}  // namespace

// Values up to 8,155 bytes stay in line; one more byte goes out of line.
TEST(EncodingTest, InLineLimitIs8155Bytes) {
    // The largest in-line value shares its page with NULL bits.
    ColumnVector edge(ColumnType::Text());
    edge.AppendString(S(8155));
    edge.AppendNull();
    ExpectEveryEncodingRoundTrips(edge);
    const EncodedChunk in_line = EncodeChunk(edge);
    EXPECT_EQ(in_line.meta.big_page_count, 0u);
    EXPECT_EQ(HeaderOf(in_line, 0).flags, kPageHasNulls);

    edge.AppendString(S(8156));
    const EncodedChunk out_of_line = EncodeChunk(edge);
    EXPECT_EQ(out_of_line.meta.encoding, Encoding::PLAIN);
    EXPECT_EQ(out_of_line.meta.big_page_count, 1u);
    ExpectEveryEncodingRoundTrips(edge);

    // 9,000 bytes, refused before big values, now round-trips.
    ExpectEveryEncodingRoundTrips(Strings({"fine", S(9000)}));
}

TEST(EncodingTest, BigValuesRoundTrip) {
    // short, 8,155 in line, 8,156 out of line, 24,481 (four pages), 3 MB, short
    const std::vector<std::string> input = {"a", S(8155), S(8156), S(3 * 8160 + 1), S(3 << 20), "b"};
    const EncodedChunk chunk = EncodeChunk(StringsOf(input));
    EXPECT_EQ(chunk.meta.encoding, Encoding::PLAIN);
    EXPECT_GT(chunk.meta.big_page_count, 0u);
    EXPECT_EQ(chunk.meta.big_page_count, 1u + 4u + 386u);  // 3 MB is 385.5 pages of 8,160 bytes
    EXPECT_EQ(chunk.meta.page_count, chunk.meta.page_first_rows.size() + chunk.meta.big_page_count);
    EXPECT_EQ(chunk.pages.size(), static_cast<size_t>(chunk.meta.page_count) * PAGE_SIZE);
    EXPECT_EQ(DecodeAll(chunk), input);
    EXPECT_EQ(DecodeRowsStrings(chunk, {2, 4}), Pick(input, {2, 4}));
    EXPECT_EQ(DecodeRowsStrings(chunk, nullptr), input);
    EXPECT_EQ(DecodeOne(chunk, 3), input[3]);
    for (uint32_t row = 0; row < input.size(); ++row) EXPECT_EQ(DecodeOne(chunk, row), input[row]) << "row " << row;

    // Many rows over several data pages, with NULLs and big values among them.
    std::mt19937_64 rng(31);
    ColumnVector mixed(ColumnType::Text());
    for (int i = 0; i < 3000; ++i) {
        if (i % 13 == 5) {
            mixed.AppendNull();
        } else if (i % 97 == 0) {
            mixed.AppendString(S(8156 + rng() % 20000));
        } else {
            mixed.AppendString(RandomString(200, rng));
        }
    }
    const EncodedChunk many = EncodeChunk(mixed);
    EXPECT_EQ(many.meta.encoding, Encoding::PLAIN);
    EXPECT_GT(many.meta.page_first_rows.size(), 10u);
    EXPECT_GE(many.meta.big_page_count, 28u);  // 28 big values, each on at least one page
    ExpectEveryEncodingRoundTrips(mixed);

    // A decode pins each big-value page once per value read.
    MemoryPages pages(many);
    ColumnVector out(ColumnType::Text());
    DecodeChunk(many.meta, pages, out);
    EXPECT_EQ(pages.BigPins(), many.meta.big_page_count);
}

// On disk: a flagged data page, its big bits and refs, then big-value pages.
TEST(EncodingTest, BigValueLayout) {
    const EncodedChunk chunk = EncodeChunk(Strings({"a", S(8156), S(9000)}));
    ASSERT_EQ(chunk.meta.page_first_rows.size(), 1u);
    EXPECT_EQ(chunk.meta.dict_page_count, 0u);
    EXPECT_EQ(chunk.meta.big_page_count, 3u);  // 8,156 bytes on one page, 9,000 on two
    ASSERT_EQ(chunk.meta.page_count, 4u);

    const ColumnarPageHeader data = HeaderOf(chunk, 0);
    EXPECT_EQ(data.kind, static_cast<uint8_t>(PageKind::DATA));
    EXPECT_EQ(data.encoding, static_cast<uint8_t>(Encoding::PLAIN));
    EXPECT_EQ(data.flags, kPageBigValues);
    EXPECT_EQ(data.value_count, 3u);
    EXPECT_EQ(chunk.pages[kPageHeaderSize], 0x06) << "slots 1 and 2 are big";

    // Offsets start after the big bits; each big slot holds an 8-byte reference.
    const size_t strings = kPageHeaderSize + 1 + 2 * 4;
    EXPECT_EQ(SlotStart(chunk, 0, 0), strings);
    EXPECT_EQ(SlotStart(chunk, 0, 1), strings + 1);
    EXPECT_EQ(SlotStart(chunk, 0, 2), strings + 1 + 8);
    EXPECT_EQ(SlotStart(chunk, 0, 3), strings + 1 + 16);
    EXPECT_EQ(PageBytes(chunk, 0, strings, 1), "a");
    BigValueRef first, second;
    std::memcpy(&first, chunk.pages.data() + strings + 1, sizeof(first));
    std::memcpy(&second, chunk.pages.data() + strings + 9, sizeof(second));
    EXPECT_EQ(first.page_index, 1u);
    EXPECT_EQ(first.length, 8156u);
    EXPECT_EQ(second.page_index, 2u);
    EXPECT_EQ(second.length, 9000u);

    // Each big-value page: its header, then a piece of the value, then zeros.
    const struct {
        uint32_t page, first_byte, bytes;
        std::string piece;
    } pieces[] = {{1, 0, 8156, S(8156)}, {2, 0, 8160, S(9000).substr(0, 8160)}, {3, 8160, 840, S(9000).substr(8160)}};
    for (const auto& p : pieces) {
        SCOPED_TRACE("page " + std::to_string(p.page));
        const ColumnarPageHeader h = HeaderOf(chunk, p.page);
        EXPECT_EQ(h.kind, static_cast<uint8_t>(PageKind::BIG_VALUE));
        EXPECT_EQ(h.first_row, p.first_byte);
        EXPECT_EQ(h.value_count, p.bytes);
        EXPECT_EQ(PageBytes(chunk, p.page, kPageHeaderSize, p.bytes), p.piece);
        const std::string rest = PageBytes(chunk, p.page, kPageHeaderSize + p.bytes, PAGE_SIZE - kPageHeaderSize - p.bytes);
        EXPECT_EQ(rest, std::string(rest.size(), '\0'));
    }

    // With NULL bits, the big bits follow them.
    ColumnVector nulls(ColumnType::Text());
    nulls.AppendString("a");
    nulls.AppendNull();
    nulls.AppendString(S(8156));
    const EncodedChunk with_nulls = EncodeChunk(nulls);
    EXPECT_EQ(HeaderOf(with_nulls, 0).flags, kPageHasNulls | kPageBigValues);
    EXPECT_EQ(with_nulls.pages[kPageHeaderSize], 0x02) << "slot 1 is NULL";
    EXPECT_EQ(with_nulls.pages[kPageHeaderSize + 1], 0x04) << "slot 2 is big";
}

TEST(EncodingTest, BigValueChunkIsNeverDictionary) {
    const ColumnVector v = RepeatedWithOneBig();
    EXPECT_FALSE(TryEncodeChunk(v, Encoding::DICTIONARY).has_value());
    const EncodedChunk chunk = EncodeChunk(v);
    EXPECT_EQ(chunk.meta.encoding, Encoding::PLAIN);
    EXPECT_TRUE(SameValues(Decode(chunk, v.Type()), v));

    // The same words without the big value pick DICTIONARY.
    ColumnVector small(ColumnType::Text());
    for (size_t i = 0; i < v.Size(); ++i) {
        if (i != 2500) small.AppendString(v.GetString(i));
    }
    EXPECT_EQ(EncodeChunk(small).meta.encoding, Encoding::DICTIONARY);
}

// One big value repeated isn't CONSTANT, so a chunk's meta never holds a big value.
TEST(EncodingTest, BigConstantStaysOnPages) {
    for (int rows : {1, 3}) {
        SCOPED_TRACE(std::to_string(rows) + " rows");
        ColumnVector same(ColumnType::Text());
        for (int i = 0; i < rows; ++i) same.AppendString(S(9000));
        EXPECT_FALSE(TryEncodeChunk(same, Encoding::CONSTANT).has_value());
        const EncodedChunk chunk = EncodeChunk(same);
        EXPECT_EQ(chunk.meta.encoding, Encoding::PLAIN);
        EXPECT_FALSE(chunk.meta.constant.has_value());
        EXPECT_TRUE(SameValues(Decode(chunk, same.Type()), same));
    }
    // An in-line value repeated is still CONSTANT.
    EXPECT_EQ(EncodeChunk(Strings({S(8155), S(8155)})).meta.encoding, Encoding::CONSTANT);
}

TEST(EncodingTest, ValueOverTheCapIsRefused) {
    EXPECT_THROW(CheckValueSize(1025, /*cap=*/1024), DatabaseException);
    EXPECT_NO_THROW(CheckValueSize(1024, 1024));
    EXPECT_EQ(ThrownText([] { CheckValueSize(1025, 1024); }), "value of 1025 bytes is over the 1 GiB limit for one value");

    static_assert(kMaxBigValueBytes == 1073741824, "the cap is 1 GiB");
    EXPECT_NO_THROW(CheckValueSize(kMaxBigValueBytes));
    EXPECT_EQ(ThrownText([] { CheckValueSize(kMaxBigValueBytes + 1); }), "value of 1073741825 bytes is over the 1 GiB limit for one value");
}

// Forged big-value pages and references are refused, even with good checksums.
TEST(EncodingTest, DamagedBigValuePageThrows) {
    const std::vector<std::string> input = {"a", S(9000), "b"};
    const EncodedChunk good = EncodeChunk(StringsOf(input));
    ASSERT_EQ(good.meta.page_count, 3u);  // the data page, then the value's two pages
    ASSERT_EQ(good.meta.big_page_count, 2u);
    const size_t ref_at = SlotStart(good, 0, 1);

    // Rewrites a page header field, then reseals.
    const auto forge_header = [&](uint32_t page, auto change) {
        EncodedChunk chunk = good;
        char* p = chunk.pages.data() + static_cast<size_t>(page) * PAGE_SIZE;
        ColumnarPageHeader h;
        std::memcpy(&h, p, sizeof(h));
        change(h);
        std::memcpy(p, &h, sizeof(h));
        SealPages(chunk.pages.data(), chunk.meta.page_count, 7);
        return chunk;
    };
    // Rewrites the big slot's reference, then reseals.
    const auto forge_ref = [&](uint32_t page_index, uint32_t length) {
        EncodedChunk chunk = good;
        const BigValueRef ref{page_index, length};
        std::memcpy(chunk.pages.data() + ref_at, &ref, sizeof(ref));
        SealPages(chunk.pages.data(), chunk.meta.page_count, 7);
        return chunk;
    };

    {
        SCOPED_TRACE("a big-value page of another kind");
        ExpectDamaged(forge_header(1, [](ColumnarPageHeader& h) { h.kind = static_cast<uint8_t>(PageKind::DATA); }), ColumnType::Text(), 1);
    }
    {
        SCOPED_TRACE("a piece starting at the wrong byte");
        ExpectDamaged(forge_header(2, [](ColumnarPageHeader& h) { h.first_row = 8000; }), ColumnType::Text(), 1);
    }
    {
        SCOPED_TRACE("a piece of the wrong length");
        ExpectDamaged(forge_header(2, [](ColumnarPageHeader& h) { h.value_count = 841; }), ColumnType::Text(), 1);
    }
    {
        SCOPED_TRACE("the big flag on a dictionary page");
        const ColumnVector words = Strings({"x", "y", "x", "y", "x"});
        EncodedChunk coded = *TryEncodeChunk(words, Encoding::DICTIONARY);
        coded.pages[offsetof(ColumnarPageHeader, flags)] = static_cast<char>(kPageBigValues);
        ExpectDamaged(coded, words.Type(), 0);
    }
    {
        SCOPED_TRACE("the big flag on an integer page");
        const ColumnVector ints = Ints(ColumnType::BigInt(), {5, 900, 17});
        EncodedChunk plain = *TryEncodeChunk(ints, Encoding::PLAIN);
        plain.pages[offsetof(ColumnarPageHeader, flags)] = static_cast<char>(kPageBigValues);
        ExpectDamaged(plain, ints.Type(), 0);
    }
    for (const auto& [page_index, length] : {std::pair<uint32_t, uint32_t>{0, 9000}, {2, 9000}, {1, 8000}, {1, 9001}, {1, UINT32_MAX}}) {
        SCOPED_TRACE("a reference to " + std::to_string(length) + " bytes at page " + std::to_string(page_index));
        ExpectDamaged(forge_ref(page_index, length), ColumnType::Text(), 1);
    }
    {
        SCOPED_TRACE("a chunk that doesn't count its big-value pages");
        EncodedChunk uncounted = good;
        uncounted.meta.big_page_count = 0;
        ExpectDamaged(uncounted, ColumnType::Text(), 1);
    }

    // The checksum covers big-value pages like any other.
    EncodedChunk sealed = good;
    SealPages(sealed.pages.data(), sealed.meta.page_count, 100);
    for (uint32_t i = 0; i < sealed.meta.page_count; ++i) {
        EXPECT_TRUE(VerifyPage(sealed.pages.data() + static_cast<size_t>(i) * PAGE_SIZE, 100 + i));
    }
    char* last = sealed.pages.data() + 2 * PAGE_SIZE;
    last[kPageHeaderSize + 500] = static_cast<char>(last[kPageHeaderSize + 500] ^ 1);
    EXPECT_FALSE(VerifyPage(last, 102));
    EXPECT_EQ(DecodeAll(EncodeChunk(StringsOf(input))), input);
}

TEST(EncodingTest, ChecksumCatchesFlippedBit) {
    std::mt19937_64 rng(9);
    const ColumnVector v = RandomVector(ColumnType::BigInt(), 3000, 0.1, rng);
    EncodedChunk chunk = EncodeChunk(v);
    ASSERT_GE(chunk.meta.page_count, 2u);

    const page_id_t first = 100;
    SealPages(chunk.pages.data(), chunk.meta.page_count, first);
    for (uint32_t i = 0; i < chunk.meta.page_count; ++i) {
        EXPECT_TRUE(VerifyPage(chunk.pages.data() + static_cast<size_t>(i) * PAGE_SIZE, first + i));
    }

    char* page = chunk.pages.data();
    for (size_t bit = 0; bit < PAGE_SIZE * 8; ++bit) {
        page[bit / 8] = static_cast<char>(page[bit / 8] ^ (1 << (bit % 8)));
        ASSERT_FALSE(VerifyPage(page, first)) << "bit " << bit;
        page[bit / 8] = static_cast<char>(page[bit / 8] ^ (1 << (bit % 8)));
    }
    EXPECT_TRUE(VerifyPage(page, first));
    EXPECT_FALSE(VerifyPage(page, first + 1));

    // Sealed pages still decode.
    EXPECT_TRUE(SameValues(Decode(chunk, v.Type()), v));
}

TEST(EncodingTest, DamagedPageKindThrows) {
    std::mt19937_64 rng(1);
    const ColumnVector v = RandomVector(ColumnType::Text(), 100, 0.0, rng);
    EncodedChunk chunk = *TryEncodeChunk(v, Encoding::PLAIN);
    chunk.pages[4] = static_cast<char>(PageKind::DICTIONARY);
    ColumnVector out(v.Type());
    MemoryPages pages(chunk);
    try {
        DecodeChunk(chunk.meta, pages, out);
        FAIL() << "expected a throw";
    } catch (const DatabaseException& e) {
        EXPECT_NE(std::string(e.what()).find("damaged column page"), std::string::npos) << e.what();
    }
    EXPECT_THROW(DecodeValue(chunk.meta, pages, 0, out), DatabaseException);
}

// A header whose checksum passes but whose bit width is past 64 is refused.
TEST(EncodingTest, BitWidthOver64Throws) {
    ColumnVector v = Ints(ColumnType::BigInt(), {1, 5, 9, 200});
    EncodedChunk chunk = *TryEncodeChunk(v, Encoding::BITPACK);
    chunk.pages[offsetof(ColumnarPageHeader, bit_width)] = 65;
    SealPages(chunk.pages.data(), chunk.meta.page_count, 7);
    ASSERT_TRUE(VerifyPage(chunk.pages.data(), 7));

    ColumnVector out(v.Type());
    MemoryPages pages(chunk);
    try {
        DecodeValue(chunk.meta, pages, 1, out);
        FAIL() << "expected a throw";
    } catch (const DatabaseException& e) {
        EXPECT_NE(std::string(e.what()).find("damaged column page"), std::string::npos) << e.what();
    }
    const std::string whole = ThrownText([&] { DecodeChunk(chunk.meta, pages, out); });
    EXPECT_EQ(whole, "damaged column page: bit width 65");
}

// A sealed page whose value count is too big is refused before any read.
TEST(EncodingTest, OversizedValueCountThrows) {
    const ColumnVector ints = Ints(ColumnType::BigInt(), {5, 900, 17, 5, 900, 17, 42, 5, 900, 17});
    const ColumnVector words = Strings({"alpha", "bravo", "charlie", "delta", "echo", "alpha", "bravo"});
    const std::pair<const ColumnVector*, Encoding> cases[] = {
        {&ints, Encoding::PLAIN}, {&words, Encoding::PLAIN}, {&ints, Encoding::BITPACK},
        {&ints, Encoding::DICTIONARY}, {&words, Encoding::DICTIONARY}};
    for (const auto& [values, encoding] : cases) {
        for (bool nulls : {false, true}) {
            SCOPED_TRACE(ColumnTypeName(values->Type()) + " encoding " + std::to_string(static_cast<int>(encoding)) +
                         (nulls ? " with" : " without") + " NULL bits");
            EncodedChunk chunk = *TryEncodeChunk(*values, encoding);
            ASSERT_EQ(chunk.meta.page_first_rows.size(), 1u);
            const uint32_t data_page = chunk.meta.dict_page_count;
            if (encoding != Encoding::PLAIN) {
                ASSERT_GT(HeaderOf(chunk, data_page).bit_width, 0);
            }
            ForgeHeader(chunk, data_page, 65535, nulls);
            chunk.meta.value_count = 65535;
            ExpectDamaged(chunk, values->Type(), 65534);
        }
    }

    // 1,010 BIGINTs fit a page alone, but not after their NULL bits.
    ColumnVector longs(ColumnType::BigInt());
    for (int i = 0; i < 1010; ++i) longs.AppendInt(i * 1000003LL * 1000003LL);
    EncodedChunk chunk = *TryEncodeChunk(longs, Encoding::PLAIN);
    ASSERT_EQ(chunk.meta.page_first_rows.size(), 1u);
    ForgeHeader(chunk, 0, 1010, true);
    ExpectDamaged(chunk, longs.Type(), 1009);
}

// A string's start offset pointing into the header or the offset table is refused.
TEST(EncodingTest, StringOffsetInsideTableThrows) {
    const ColumnVector words = Strings({"alpha", "bravo", "charlie", "delta", "echo"});
    for (uint16_t from : {uint16_t{4}, uint16_t{kPageHeaderSize + 2}}) {
        SCOPED_TRACE("from " + std::to_string(from));
        EncodedChunk chunk = *TryEncodeChunk(words, Encoding::PLAIN);
        std::memcpy(chunk.pages.data() + kPageHeaderSize, &from, sizeof(from));
        SealPages(chunk.pages.data(), chunk.meta.page_count, 7);
        ExpectDamaged(chunk, words.Type(), 0);
    }
}

// The catalog's page starts must match the pages' own headers.
TEST(EncodingTest, FirstRowMismatchThrows) {
    ColumnVector v(ColumnType::BigInt());
    for (int i = 0; i < 1500; ++i) v.AppendInt(i * 1000003LL * 1000003LL);
    EncodedChunk chunk = *TryEncodeChunk(v, Encoding::PLAIN);
    ASSERT_EQ(chunk.meta.page_first_rows, (std::vector<uint32_t>{0, 1020}));
    chunk.meta.page_first_rows[1] = 1015;
    ExpectDamaged(chunk, v.Type(), 1015);

    // The same for a dictionary page's first entry.
    ColumnVector distinct(ColumnType::BigInt());
    for (int i = 0; i < 1500; ++i) distinct.AppendInt(i * 1000003LL * 1000003LL);
    EncodedChunk coded = *TryEncodeChunk(distinct, Encoding::DICTIONARY);
    ASSERT_EQ(coded.meta.dict_page_first_entries, (std::vector<uint32_t>{0, 1020}));
    coded.meta.dict_page_first_entries[1] = 1015;
    ExpectDamaged(coded, distinct.Type(), 1015);
}

// Values outside a narrow type's range throw instead of being cut short.
TEST(EncodingTest, ValueOutsideTypeThrows) {
    auto expect_throw = [](const ColumnVector& v, const std::string& text) {
        try {
            (void)EncodeChunk(v);
            FAIL() << "expected a throw for " << text;
        } catch (const DatabaseException& e) {
            EXPECT_EQ(std::string(e.what()), text);
        }
    };
    expect_throw(Ints(ColumnType::Integer(), {1, int64_t{1} << 31}), "value 2147483648 does not fit INTEGER");
    expect_throw(Ints(ColumnType::Boolean(), {0, 1, 2}), "value 2 does not fit BOOLEAN");
    expect_throw(Ints(ColumnType::Boolean(), {-1, 1}), "value -1 does not fit BOOLEAN");

    // Forcing an encoding checks the values the same way.
    for (Encoding e : kAllEncodings) {
        SCOPED_TRACE("encoding " + std::to_string(static_cast<int>(e)));
        EXPECT_EQ(ThrownText([&] { (void)TryEncodeChunk(Ints(ColumnType::Integer(), {1, int64_t{1} << 31}), e); }),
                  "value 2147483648 does not fit INTEGER");
        EXPECT_EQ(ThrownText([&] { (void)TryEncodeChunk(Ints(ColumnType::Boolean(), {1, 2}), e); }),
                  "value 2 does not fit BOOLEAN");
    }

    // NULLs mixed with 0 and 1 are fine; a 2 among them still isn't.
    ColumnVector flags(ColumnType::Boolean());
    for (int x : {-1, 0, 1, -1, 1, 0, -1}) {
        if (x < 0) flags.AppendNull(); else flags.AppendInt(x);
    }
    ExpectEveryEncodingRoundTrips(flags);
    flags.AppendInt(2);
    expect_throw(flags, "value 2 does not fit BOOLEAN");

    ExpectEveryEncodingRoundTrips(Ints(ColumnType::Boolean(), {0, 1, 1, 0}));
    ExpectEveryEncodingRoundTrips(Ints(ColumnType::Integer(), {INT32_MIN, INT32_MAX}));
}

namespace {

// Appends a field's raw bytes, for hashing a chunk's meta.
template <typename T>
void PutRaw(std::vector<char>& out, const T& x) {
    const char* p = reinterpret_cast<const char*>(&x);
    out.insert(out.end(), p, p + sizeof(T));
}

// A present flag, then the scalar's kind and bytes.
void PutScalarBytes(std::vector<char>& out, const std::optional<Scalar>& s) {
    PutRaw(out, static_cast<uint8_t>(s.has_value()));
    if (!s) return;
    PutRaw(out, static_cast<uint8_t>(s->index()));
    switch (s->index()) {
        case 0: PutRaw(out, std::get<int64_t>(*s)); break;
        case 1: PutRaw(out, std::get<__int128>(*s)); break;
        case 2: PutRaw(out, std::get<double>(*s)); break;
        default: {
            const std::string& text = std::get<std::string>(*s);
            PutRaw(out, static_cast<uint32_t>(text.size()));
            out.insert(out.end(), text.begin(), text.end());
            break;
        }
    }
}

// The meta fields a chunk had before big values, as bytes.
std::vector<char> MetaBytes(const ChunkMeta& m) {
    std::vector<char> out;
    PutRaw(out, m.first_page);
    PutRaw(out, m.page_count);
    PutRaw(out, m.dict_page_count);
    PutRaw(out, static_cast<uint8_t>(m.encoding));
    PutRaw(out, m.value_count);
    PutRaw(out, m.null_count);
    PutRaw(out, static_cast<uint32_t>(m.page_first_rows.size()));
    for (uint32_t x : m.page_first_rows) PutRaw(out, x);
    PutRaw(out, static_cast<uint32_t>(m.dict_page_first_entries.size()));
    for (uint32_t x : m.dict_page_first_entries) PutRaw(out, x);
    PutRaw(out, m.dict_size);
    PutScalarBytes(out, m.min);
    PutScalarBytes(out, m.max);
    PutScalarBytes(out, m.constant);
    return out;
}

// One CRC32C over the picked chunk and every forced encoding: meta, then pages.
uint32_t ChunkHash(const ColumnVector& v) {
    std::vector<std::optional<EncodedChunk>> chunks;
    chunks.emplace_back(EncodeChunk(v));
    for (Encoding e : kAllEncodings) chunks.push_back(TryEncodeChunk(v, e));
    uint32_t crc = 0;
    for (const auto& chunk : chunks) {
        const uint8_t present = chunk.has_value() ? 1 : 0;
        crc = Crc32c(&present, 1, crc);
        if (!chunk) continue;
        const std::vector<char> meta = MetaBytes(chunk->meta);
        crc = Crc32c(meta.data(), meta.size(), crc);
        crc = Crc32c(chunk->pages.data(), chunk->pages.size(), crc);
    }
    return crc;
}

// Letters drawn straight from the engine, so every library gives the same string.
std::string Letters(size_t n, std::mt19937_64& rng) {
    std::string s(n, ' ');
    for (char& c : s) c = static_cast<char>('a' + rng() % 26);
    return s;
}

// The fixed chunks whose bytes must never change: every encoding, NULLs, long strings.
std::vector<std::pair<std::string, ColumnVector>> TodaysChunks() {
    std::vector<std::pair<std::string, ColumnVector>> cases;
    std::mt19937_64 rng(20261122);
    const size_t lengths[] = {0, 1, 7, 64, 65, 300, 4000, 8154, 8155};

    ColumnVector long_text(ColumnType::Text());
    for (int i = 0; i < 60; ++i) long_text.AppendString(Letters(lengths[i % 9], rng));
    cases.emplace_back("plain strings up to 8155", std::move(long_text));

    ColumnVector long_nulls(ColumnType::Text());
    for (int i = 0; i < 60; ++i) {
        if (i % 4 == 3) long_nulls.AppendNull(); else long_nulls.AppendString(Letters(lengths[i % 9], rng));
    }
    cases.emplace_back("plain strings with NULLs", std::move(long_nulls));

    ColumnVector varchar(ColumnType::Varchar(30));
    for (int i = 0; i < 5000; ++i) {
        if (i % 7 == 0) varchar.AppendNull(); else varchar.AppendString(std::to_string(i) + Letters(rng() % 20, rng));
    }
    cases.emplace_back("varchar with NULLs", std::move(varchar));

    const char* words[] = {"alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"};
    ColumnVector seven(ColumnType::Text());
    ColumnVector seven_nulls(ColumnType::Varchar(10));
    for (int i = 0; i < 65536; ++i) {
        seven.AppendString(words[rng() % 7]);
        if (rng() % 10 == 0) seven_nulls.AppendNull(); else seven_nulls.AppendString(words[rng() % 7]);
    }
    cases.emplace_back("dictionary strings", std::move(seven));
    cases.emplace_back("dictionary strings with NULLs", std::move(seven_nulls));

    ColumnVector small(ColumnType::BigInt());
    ColumnVector small_nulls(ColumnType::Integer());
    ColumnVector wide_ints(ColumnType::Integer());
    ColumnVector dates(ColumnType::Date());
    ColumnVector flags(ColumnType::Boolean());
    ColumnVector shorts(ColumnType::SmallInt());
    for (int i = 0; i < 65536; ++i) {
        small.AppendInt(static_cast<int64_t>(rng() % 100));
        if (i % 9 == 0) small_nulls.AppendNull(); else small_nulls.AppendInt(static_cast<int64_t>(rng() % 5000) - 2500);
        wide_ints.AppendInt(static_cast<int32_t>(rng()));
        dates.AppendInt(8000 + static_cast<int64_t>(rng() % 2557));
        flags.AppendInt(static_cast<int64_t>(rng() & 1));
        shorts.AppendInt(static_cast<int16_t>(rng()));
    }
    cases.emplace_back("bitpack integers", std::move(small));
    cases.emplace_back("bitpack integers with NULLs", std::move(small_nulls));
    cases.emplace_back("random integers", std::move(wide_ints));
    cases.emplace_back("dates", std::move(dates));
    cases.emplace_back("booleans", std::move(flags));
    cases.emplace_back("smallints", std::move(shorts));

    ColumnVector money(ColumnType::Numeric(15, 2));
    ColumnVector wide_money(ColumnType::Numeric(38, 6));
    ColumnVector doubles(ColumnType::Double());
    for (int i = 0; i < 20000; ++i) {
        money.AppendInt(static_cast<int64_t>(rng() % 2000000000000000ULL) - 1000000000000000LL);
        if (i % 11 == 0) wide_money.AppendNull(); else wide_money.AppendWide(static_cast<__int128>(static_cast<int64_t>(rng())) * 1000003);
        doubles.AppendDouble(static_cast<double>(static_cast<int64_t>(rng() >> 11)) / 1024.0);
    }
    cases.emplace_back("numeric 15,2", std::move(money));
    cases.emplace_back("numeric 38,6 with NULLs", std::move(wide_money));
    cases.emplace_back("doubles", std::move(doubles));

    ColumnVector same_text(ColumnType::Text());
    for (int i = 0; i < 1000; ++i) same_text.AppendString("the same words");
    ColumnVector same_int(ColumnType::BigInt());
    for (int i = 0; i < 1000; ++i) same_int.AppendInt(-42);
    ColumnVector all_null(ColumnType::Text());
    for (int i = 0; i < 1000; ++i) all_null.AppendNull();
    cases.emplace_back("constant string", std::move(same_text));
    cases.emplace_back("constant integer", std::move(same_int));
    cases.emplace_back("all NULL", std::move(all_null));
    return cases;
}

}  // namespace

// Values up to 8,155 bytes: pages and meta match the encoder before big values.
TEST(EncodingTest, NoBigValueKeepsTodaysBytes) {
    // Hashes taken from the encoder as it was before big values.
    struct Expected {
        const char* name;
        Encoding picked;
        uint32_t hash;
    };
    const Expected expected[] = {
        {"plain strings up to 8155", Encoding::PLAIN, 0xCC8660ABu},
        {"plain strings with NULLs", Encoding::PLAIN, 0xA8C89AA9u},
        {"varchar with NULLs", Encoding::PLAIN, 0x86E9B3ECu},
        {"dictionary strings", Encoding::DICTIONARY, 0x76DCDB30u},
        {"dictionary strings with NULLs", Encoding::DICTIONARY, 0x11C030D9u},
        {"bitpack integers", Encoding::BITPACK, 0x8D7413F8u},
        {"bitpack integers with NULLs", Encoding::BITPACK, 0x2BF841A1u},
        {"random integers", Encoding::PLAIN, 0x99D751A9u},
        {"dates", Encoding::BITPACK, 0x37CB91C1u},
        {"booleans", Encoding::BITPACK, 0x4FD21ABFu},
        {"smallints", Encoding::PLAIN, 0x3DA92A29u},
        {"numeric 15,2", Encoding::BITPACK, 0x20C0FFD1u},
        {"numeric 38,6 with NULLs", Encoding::PLAIN, 0xFD191D04u},
        {"doubles", Encoding::PLAIN, 0x30DDB14Fu},
        {"constant string", Encoding::CONSTANT, 0xDB2ACD32u},
        {"constant integer", Encoding::CONSTANT, 0x7771D54Du},
        {"all NULL", Encoding::CONSTANT, 0x5EE05D2Bu},
    };
    const auto cases = TodaysChunks();
    ASSERT_EQ(cases.size(), std::size(expected));
    for (size_t i = 0; i < cases.size(); ++i) {
        ASSERT_EQ(cases[i].first, expected[i].name);
        const EncodedChunk picked = EncodeChunk(cases[i].second);
        EXPECT_EQ(picked.meta.encoding, expected[i].picked) << cases[i].first;
        EXPECT_EQ(picked.meta.big_page_count, 0u) << cases[i].first;
        const uint32_t hash = ChunkHash(cases[i].second);
        char text[16];
        std::snprintf(text, sizeof(text), "0x%08Xu", hash);
        EXPECT_EQ(hash, expected[i].hash) << cases[i].first << ": " << text;
    }
}

}  // namespace shilmandb
