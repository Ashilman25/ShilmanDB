#include <gtest/gtest.h>
#include "common/exception.hpp"
#include "storage/columnar/encoding.hpp"
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace shilmandb {
namespace {

constexpr Encoding kAllEncodings[] = {Encoding::CONSTANT, Encoding::PLAIN, Encoding::BITPACK,
                                      Encoding::DICTIONARY};

// Hands out a chunk's pages, checking pins are undone, one page at a time.
class MemoryPages : public ChunkPages {
public:
    explicit MemoryPages(const EncodedChunk& chunk) : chunk_(chunk) {}
    ~MemoryPages() override {
        EXPECT_EQ(pins_, unpins_);
        EXPECT_EQ(pinned_, kNone);
    }

    const char* Pin(uint32_t index) override {
        EXPECT_EQ(pinned_, kNone) << "two pages pinned at once";
        EXPECT_LT(index, chunk_.meta.page_count);
        pinned_ = index;
        ++pins_;
        return chunk_.pages.data() + static_cast<size_t>(index) * PAGE_SIZE;
    }
    void Unpin(uint32_t index) override {
        EXPECT_EQ(pinned_, index);
        pinned_ = kNone;
        ++unpins_;
    }

private:
    static constexpr uint32_t kNone = UINT32_MAX;
    const EncodedChunk& chunk_;
    uint32_t pinned_{kNone};
    uint64_t pins_{0};
    uint64_t unpins_{0};
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

    // NaN is skipped too; all NULLs means no min or max at all.
    ColumnVector doubles(ColumnType::Double());
    doubles.AppendDouble(std::nan(""));
    doubles.AppendDouble(2.5);
    doubles.AppendNull();
    doubles.AppendDouble(-1.0);
    const auto double_meta = EncodeChunk(doubles).meta;
    ASSERT_TRUE(double_meta.min.has_value() && double_meta.max.has_value());
    EXPECT_EQ(std::get<double>(*double_meta.min), -1.0);
    EXPECT_EQ(std::get<double>(*double_meta.max), 2.5);

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

TEST(EncodingTest, ValueTooBigForPageThrows) {
    const auto v = Strings({"fine", std::string(9000, 'x')});
    try {
        (void)EncodeChunk(v);
        FAIL() << "expected a throw";
    } catch (const DatabaseException& e) {
        EXPECT_NE(std::string(e.what()).find("bigger than a page"), std::string::npos) << e.what();
        EXPECT_NE(std::string(e.what()).find("9000"), std::string::npos) << e.what();
    }
    EXPECT_THROW((void)TryEncodeChunk(v, Encoding::DICTIONARY), DatabaseException);

    // The largest value that fits shares its page with NULL bits.
    ColumnVector edge(ColumnType::Text());
    edge.AppendString(std::string(8155, 'y'));
    edge.AppendNull();
    ExpectEveryEncodingRoundTrips(edge);
    edge.AppendString(std::string(8156, 'y'));
    EXPECT_THROW((void)EncodeChunk(edge), DatabaseException);
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
    EXPECT_THROW(DecodeChunk(chunk.meta, pages, out), DatabaseException);
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

    ExpectEveryEncodingRoundTrips(Ints(ColumnType::Boolean(), {0, 1, 1, 0}));
    ExpectEveryEncodingRoundTrips(Ints(ColumnType::Integer(), {INT32_MIN, INT32_MAX}));
}

}  // namespace shilmandb
