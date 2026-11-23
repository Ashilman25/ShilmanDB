#include <gtest/gtest.h>
#include "common/exception.hpp"
#include "storage/columnar/catalog_codec.hpp"
#include "storage/columnar/crc32c.hpp"
#include "storage/columnar/superblock.hpp"
#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace shilmandb {
namespace {

constexpr Encoding kAllEncodings[] = {Encoding::CONSTANT, Encoding::PLAIN, Encoding::BITPACK,
                                      Encoding::DICTIONARY};
constexpr size_t kByteCountOffset = 4;  // u32 version, then the u64 byte count

// Builds random but well-formed catalogs, and notes which variants it used.
// With big_pages on, PLAIN chunks may count big-value pages too.
class CatalogMaker {
public:
    explicit CatalogMaker(uint32_t seed, bool big_pages = false) : rng_(seed), big_pages_(big_pages) {}

    std::set<Encoding> encodings_seen;
    std::set<size_t> scalars_seen;
    std::set<TypeId> values_seen;

    // A catalog with 0 to 3 tables and, when tables exist, 0 to 4 indexes.
    CatalogMeta Catalog() {
        CatalogMeta meta;
        const int tables = Int(0, 3);
        for (int t = 0; t < tables; ++t) meta.tables.push_back(Table(kFirstOid + t));
        const int indexes = meta.tables.empty() ? 0 : Int(0, 4);
        for (int i = 0; i < indexes; ++i) {
            const auto& table = meta.tables[Int(0, static_cast<int>(meta.tables.size()) - 1)];
            IndexMeta index;
            index.oid = kFirstOid + 100 + i;
            index.name = "idx" + std::to_string(i) + ":" + Text(0, 12);  // the prefix keeps names unique
            index.table_oid = table.oid;
            index.column_number = Int(1, static_cast<int>(table.columns.size()));
            index.key_type = ExecutorType(table.columns[index.column_number - 1].type);
            index.root_page = Int(0, 1) ? INVALID_PAGE_ID : static_cast<page_id_t>(Int(2, 1 << 20));
            index.unique = Int(0, 1) == 1;
            meta.indexes.push_back(std::move(index));
        }
        return meta;
    }

private:
    // An integer in [lo, hi], drawn straight from the engine so every library agrees.
    int Int(int lo, int hi) {
        const auto span = static_cast<uint64_t>(static_cast<int64_t>(hi) - lo + 1);
        return static_cast<int>(lo + static_cast<int64_t>(Raw() % span));
    }

    // The engine's next 32 bits; mt19937's output is the same on every library.
    uint64_t Raw() { return static_cast<uint64_t>(rng_()) & 0xFFFFFFFFu; }

    // Random bytes, any of the 256, of a length in [lo, hi].
    std::string Text(int lo, int hi) {
        std::string s(Int(lo, hi), ' ');
        for (char& c : s) c = static_cast<char>(Int(0, 255));
        return s;
    }

    // Any double, with NaN, -0.0 and infinity mixed in.
    double Double() {
        switch (Int(0, 4)) {
            case 0: return std::numeric_limits<double>::quiet_NaN();
            case 1: return -0.0;
            case 2: return -std::numeric_limits<double>::infinity();
            default: return -1e12 + 2e12 * (static_cast<double>(Raw()) / 4294967296.0);
        }
    }

    // One of the ten column types, built through its checking factory.
    ColumnType Type() {
        switch (Int(1, 10)) {
            case 1: return ColumnType::Boolean();
            case 2: return ColumnType::SmallInt();
            case 3: return ColumnType::Integer();
            case 4: return ColumnType::BigInt();
            case 5: {
                const int p = Int(1, 38);
                return ColumnType::Numeric(p, Int(0, p));
            }
            case 6: return ColumnType::Double();
            case 7: return ColumnType::Date();
            case 8: return ColumnType::Timestamp();
            case 9: return ColumnType::Varchar(static_cast<uint32_t>(Int(1, 100000)));
            default: return ColumnType::Text();
        }
    }

    // One of the four scalar kinds, picked at random.
    Scalar MakeScalar() {
        Scalar s;
        switch (Int(0, 3)) {
            case 0: {
                const uint64_t high = Raw();  // two statements, so the draw order is fixed
                s = static_cast<int64_t>(high << 32 | Raw());
                break;
            }
            case 1: s = static_cast<__int128>(static_cast<int64_t>(Raw())) * -1000000000000000LL; break;
            case 2: s = Double(); break;
            default: s = Text(0, 64); break;
        }
        scalars_seen.insert(s.index());
        return s;
    }

    // A scalar three times in four, else nothing.
    std::optional<Scalar> MaybeScalar() {
        if (Int(0, 3) == 0) return std::nullopt;
        return MakeScalar();
    }

    // One chunk of a group, with the page lists its encoding needs.
    ChunkMeta Chunk(Encoding encoding, uint32_t rows) {
        ChunkMeta c;
        c.encoding = encoding;
        encodings_seen.insert(encoding);
        c.value_count = rows;
        c.null_count = static_cast<uint32_t>(Int(0, static_cast<int>(rows)));
        c.min = MaybeScalar();
        c.max = MaybeScalar();
        if (encoding == Encoding::CONSTANT) {
            c.constant = MaybeScalar();
            return c;
        }
        // Page starts begin at 0 and rise strictly, all below the count they split.
        if (encoding == Encoding::DICTIONARY) c.dict_size = Int(0, static_cast<int>(rows - c.null_count));
        c.dict_page_count = c.dict_size == 0 ? 0 : Int(1, static_cast<int>(std::min<uint32_t>(3, c.dict_size)));
        const auto data_pages = static_cast<uint32_t>(Int(1, static_cast<int>(std::min<uint32_t>(4, rows))));
        if (big_pages_ && encoding == Encoding::PLAIN) c.big_page_count = static_cast<uint32_t>(Int(0, 5));
        c.page_count = c.dict_page_count + data_pages + c.big_page_count;
        c.first_page = static_cast<page_id_t>(Int(2, 1 << 24));
        for (uint32_t i = 0; i < data_pages; ++i) c.page_first_rows.push_back(i * (rows / data_pages));
        for (uint32_t i = 0; i < c.dict_page_count; ++i) {
            c.dict_page_first_entries.push_back(i * (c.dict_size / c.dict_page_count));
        }
        return c;
    }

    // One min / max value of each executor type in turn.
    Value MakeValue(TypeId type) {
        values_seen.insert(type);
        switch (type) {
            case TypeId::INTEGER: return Value(static_cast<int32_t>(Raw()));
            case TypeId::BIGINT:  return Value(static_cast<int64_t>(Raw()) << 20);
            case TypeId::DECIMAL: return Value(Double());
            case TypeId::VARCHAR: return Value(Text(0, 20));
            default:              return Value::MakeDate(Int(-30000, 30000));
        }
    }

    // A table with 1 to 16 columns, 0 to 3 row groups and random stats.
    ColumnarTableMeta Table(uint32_t oid) {
        ColumnarTableMeta t;
        t.oid = oid;
        t.name = "t" + std::to_string(oid) + ":" + Text(0, 20);  // the prefix keeps names unique
        const int columns = Int(1, 16);
        for (int i = 0; i < columns; ++i) {
            ColumnMeta col;
            col.name = "c" + std::to_string(i) + ":" + Text(0, 6);
            col.number = static_cast<uint32_t>(i + 1);
            col.type = Type();
            col.nullable = Int(0, 1) == 1;
            t.columns.push_back(std::move(col));
        }
        const int groups = Int(0, 3);
        for (int g = 0; g < groups; ++g) {
            RowGroupMeta group;
            group.row_count = static_cast<uint32_t>(Int(1, static_cast<int>(kRowGroupSize)));
            for (int i = 0; i < columns; ++i) {
                group.chunks.push_back(Chunk(kAllEncodings[(g + i + next_encoding_++) % 4], group.row_count));
            }
            t.row_count += group.row_count;
            t.row_groups.push_back(std::move(group));
        }
        t.stats.row_count = t.row_count;
        t.stats.page_count = static_cast<uint64_t>(Int(0, 100000));
        for (const auto& col : t.columns) {
            if (Int(0, 1)) t.stats.distinct_counts[col.name] = Raw();
            if (Int(0, 1)) {
                const auto type = static_cast<TypeId>(Int(1, 5));
                t.stats.min_max[col.name] = MinMax{MakeValue(type), MakeValue(type)};
            }
        }
        return t;
    }

    std::mt19937 rng_;
    bool big_pages_;
    int next_encoding_{0};
};

// A small catalog with one of everything, for the damage tests.
CatalogMeta SmallCatalog() {
    CatalogMeta meta;
    ColumnarTableMeta t;
    t.oid = kFirstOid;
    t.name = "orders";
    ColumnMeta key;
    key.name = "o_orderkey";
    key.number = 1;
    key.type = ColumnType::BigInt();
    ColumnMeta price;
    price.name = "o_totalprice";
    price.number = 2;
    price.type = ColumnType::Numeric(15, 2);
    price.nullable = true;
    t.columns = {key, price};

    RowGroupMeta group;
    group.row_count = 3000;
    ChunkMeta packed;
    packed.encoding = Encoding::BITPACK;
    packed.first_page = 10;
    packed.page_count = 2;
    packed.value_count = 3000;
    packed.page_first_rows = {0, 2000};
    packed.min = Scalar{int64_t{1}};
    packed.max = Scalar{int64_t{3000}};
    ChunkMeta dict;
    dict.encoding = Encoding::DICTIONARY;
    dict.first_page = 12;
    dict.page_count = 2;
    dict.dict_page_count = 1;
    dict.value_count = 3000;
    dict.null_count = 5;
    dict.page_first_rows = {0};
    dict.dict_page_first_entries = {0};
    dict.dict_size = 40;
    dict.min = Scalar{static_cast<__int128>(85700)};
    dict.max = Scalar{std::string("big")};
    group.chunks = {packed, dict};
    t.row_groups = {group};
    t.row_count = 3000;
    t.stats.row_count = 3000;
    t.stats.page_count = 4;
    t.stats.distinct_counts = {{"o_orderkey", 3000}, {"o_totalprice", 2990}};
    t.stats.min_max["o_orderkey"] = MinMax{Value(int64_t{1}), Value(int64_t{3000})};
    t.stats.min_max["o_totalprice"] = MinMax{Value(857.0), Value(555285.16)};
    meta.tables.push_back(std::move(t));

    IndexMeta index;
    index.oid = kFirstOid + 1;
    index.name = "idx_orders_key";
    index.table_oid = kFirstOid;
    index.column_number = 1;
    index.key_type = TypeId::BIGINT;
    index.root_page = 20;
    meta.indexes.push_back(std::move(index));
    return meta;
}

// Expects a throw whose text starts "damaged catalog" and holds the fragment.
void ExpectDamaged(const std::vector<char>& bytes, const std::string& what, const std::string& fragment = "") {
    try {
        (void)DecodeCatalog(bytes.data(), bytes.size());
        ADD_FAILURE() << what << ": decoded without a throw";
    } catch (const DatabaseException& e) {
        const std::string text = e.what();
        EXPECT_EQ(text.rfind("damaged catalog: ", 0), 0u) << what << ": " << text;
        EXPECT_NE(text.find(fragment), std::string::npos) << what << ": " << text;
    }
}

// Encodes the small catalog after one change, and expects the named damage.
void ExpectDamagedAfter(const std::function<void(CatalogMeta&)>& change, const std::string& fragment) {
    CatalogMeta meta = SmallCatalog();
    change(meta);
    ExpectDamaged(EncodeCatalog(meta), fragment, fragment);
}

TEST(CatalogCodecTest, CatalogRoundTrip) {
    CatalogMaker maker(20261116, true);
    for (int i = 0; i < 50; ++i) {
        const CatalogMeta meta = maker.Catalog();
        const std::vector<char> bytes = EncodeCatalog(meta);
        const CatalogMeta back = DecodeCatalog(bytes.data(), bytes.size());
        ASSERT_EQ(back.tables.size(), meta.tables.size()) << "catalog " << i;
        ASSERT_EQ(back.indexes.size(), meta.indexes.size()) << "catalog " << i;
        ASSERT_EQ(EncodeCatalog(back), bytes) << "catalog " << i;
    }
    EXPECT_EQ(maker.encodings_seen.size(), 4u);
    EXPECT_EQ(maker.scalars_seen.size(), 4u);
    EXPECT_EQ(maker.values_seen.size(), 5u);
}

TEST(CatalogCodecTest, CatalogKeepsFields) {
    const std::vector<char> bytes = EncodeCatalog(SmallCatalog());
    const CatalogMeta back = DecodeCatalog(bytes.data(), bytes.size());
    ASSERT_EQ(back.tables.size(), 1u);
    const auto& t = back.tables[0];
    EXPECT_EQ(t.oid, kFirstOid);
    EXPECT_EQ(t.name, "orders");
    ASSERT_EQ(t.columns.size(), 2u);
    EXPECT_EQ(t.columns[1].name, "o_totalprice");
    EXPECT_EQ(t.columns[1].number, 2u);
    EXPECT_EQ(t.columns[1].type, ColumnType::Numeric(15, 2));
    EXPECT_TRUE(t.columns[1].nullable);
    EXPECT_EQ(t.row_count, 3000u);
    ASSERT_EQ(t.row_groups.size(), 1u);
    const auto& dict = t.row_groups[0].chunks[1];
    EXPECT_EQ(dict.encoding, Encoding::DICTIONARY);
    EXPECT_EQ(dict.first_page, 12u);
    EXPECT_EQ(dict.null_count, 5u);
    EXPECT_EQ(dict.dict_size, 40u);
    EXPECT_EQ(std::get<__int128>(*dict.min), static_cast<__int128>(85700));
    EXPECT_EQ(std::get<std::string>(*dict.max), "big");
    EXPECT_FALSE(dict.constant.has_value());
    EXPECT_EQ(t.stats.page_count, 4u);
    EXPECT_EQ(t.stats.distinct_counts.at("o_totalprice"), 2990u);
    EXPECT_EQ(t.stats.min_max.at("o_totalprice").max.type_, TypeId::DECIMAL);
    EXPECT_DOUBLE_EQ(t.stats.min_max.at("o_totalprice").max.decimal_, 555285.16);
    ASSERT_EQ(back.indexes.size(), 1u);
    EXPECT_EQ(back.indexes[0].name, "idx_orders_key");
    EXPECT_EQ(back.indexes[0].key_type, TypeId::BIGINT);
    EXPECT_EQ(back.indexes[0].root_page, 20u);
    EXPECT_FALSE(back.indexes[0].unique);
}

// Version 1 bytes match the catalog codec from before big values, hashes taken from it.
TEST(CatalogCodecTest, VersionOneBytesMatchTheOldCodec) {
    const std::vector<char> small = EncodeCatalog(SmallCatalog(), 1);
    EXPECT_EQ(small.size(), 415u);
    EXPECT_EQ(Crc32c(small.data(), small.size()), 0xE7BB5C5Bu);

    CatalogMaker maker(20261122);
    uint32_t crc = 0;
    size_t bytes = 0;
    for (int i = 0; i < 20; ++i) {
        const std::vector<char> blob = EncodeCatalog(maker.Catalog(), 1);
        crc = Crc32c(blob.data(), blob.size(), crc);
        bytes += blob.size();
    }
    EXPECT_EQ(bytes, 39284u);
    EXPECT_EQ(crc, 0x8AF8C4A9u);
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

// Chunk 0 of the small catalog as PLAIN text, with big-value pages.
CatalogMeta CatalogWithBigPages() {
    CatalogMeta meta = SmallCatalog();
    meta.tables[0].columns[0].type = ColumnType::Text();
    meta.indexes.clear();
    ChunkMeta& chunk = meta.tables[0].row_groups[0].chunks[0];
    chunk.encoding = Encoding::PLAIN;
    chunk.big_page_count = 7;
    chunk.page_count = 2 + 7;
    chunk.min.reset();
    chunk.max.reset();
    return meta;
}

TEST(CatalogCodecTest, VersionOneCatalogHasNoBigPages) {
    EXPECT_EQ(kFormatVersion, 2u);
    EXPECT_EQ(kOldestFormatVersion, 1u);
    const CatalogMeta meta = SmallCatalog();
    const std::vector<char> v1 = EncodeCatalog(meta, 1);
    const std::vector<char> v2 = EncodeCatalog(meta);
    uint32_t version = 0;
    std::memcpy(&version, v1.data(), sizeof(version));
    EXPECT_EQ(version, 1u);
    std::memcpy(&version, v2.data(), sizeof(version));
    EXPECT_EQ(version, 2u);
    EXPECT_EQ(EncodeCatalog(meta, 2), v2);

    // Version 2 adds a 4-byte big page count to each of the two chunks.
    EXPECT_EQ(v2.size(), v1.size() + 2 * sizeof(uint32_t));
    const CatalogMeta back = DecodeCatalog(v1.data(), v1.size());
    for (const ChunkMeta& chunk : back.tables[0].row_groups[0].chunks) EXPECT_EQ(chunk.big_page_count, 0u);
    EXPECT_EQ(EncodeCatalog(back), v2) << "read as version 1, written as version 2";
    EXPECT_EQ(EncodeCatalog(back, 1), v1);

    // Version 1 can't hold big values, and no other version is written.
    EXPECT_EQ(ThrownText([] { (void)EncodeCatalog(CatalogWithBigPages(), 1); }),
              "catalog format 1 can't hold values bigger than a page: table orders, column o_orderkey");
    EXPECT_EQ(ThrownText([&] { (void)EncodeCatalog(meta, 0); }), "can't write catalog format 0");
    EXPECT_EQ(ThrownText([&] { (void)EncodeCatalog(meta, 3); }), "can't write catalog format 3");
}

TEST(CatalogCodecTest, BigPageCountRoundTrips) {
    const std::vector<char> bytes = EncodeCatalog(CatalogWithBigPages());
    const CatalogMeta back = DecodeCatalog(bytes.data(), bytes.size());
    const ChunkMeta& chunk = back.tables[0].row_groups[0].chunks[0];
    EXPECT_EQ(chunk.big_page_count, 7u);
    EXPECT_EQ(chunk.page_count, 9u);
    EXPECT_EQ(chunk.page_first_rows, (std::vector<uint32_t>{0, 2000}));
    EXPECT_EQ(back.tables[0].row_groups[0].chunks[1].big_page_count, 0u);
    EXPECT_EQ(EncodeCatalog(back), bytes);

    // Random catalogs with big pages round-trip too.
    CatalogMaker maker(7, true);
    for (int i = 0; i < 20; ++i) {
        const std::vector<char> blob = EncodeCatalog(maker.Catalog());
        const CatalogMeta again = DecodeCatalog(blob.data(), blob.size());
        ASSERT_EQ(EncodeCatalog(again), blob) << "catalog " << i;
    }
}

// Big-value pages only in PLAIN chunks, and inside the chunk's page count.
TEST(CatalogCodecTest, BigPageShapesThrow) {
    const auto damaged_after = [](const std::function<void(ChunkMeta&)>& change, const std::string& fragment) {
        CatalogMeta meta = CatalogWithBigPages();
        change(meta.tables[0].row_groups[0].chunks[0]);
        ExpectDamaged(EncodeCatalog(meta), fragment, fragment);
    };
    damaged_after([](ChunkMeta& c) { c.encoding = Encoding::BITPACK; }, "big-value pages in a chunk that isn't PLAIN");
    damaged_after([](ChunkMeta& c) { c.big_page_count = 10; }, "more dictionary and big-value pages than pages in a chunk");
    damaged_after([](ChunkMeta& c) { c.big_page_count = UINT32_MAX; }, "more dictionary and big-value pages than pages in a chunk");
    damaged_after([](ChunkMeta& c) { c.big_page_count = 6; }, "chunk data page list doesn't match its page count");
    damaged_after([](ChunkMeta& c) { c.big_page_count = 8; }, "chunk data page list doesn't match its page count");

    // A dictionary chunk with big pages is refused too.
    CatalogMeta coded = SmallCatalog();
    ChunkMeta& dict = coded.tables[0].row_groups[0].chunks[1];
    dict.big_page_count = 1;
    dict.page_count += 1;
    ExpectDamaged(EncodeCatalog(coded), "big pages on a dictionary chunk", "big-value pages in a chunk that isn't PLAIN");
}

TEST(CatalogCodecTest, EmptyCatalogRoundTrips) {
    const std::vector<char> bytes = EncodeCatalog(CatalogMeta{});
    EXPECT_EQ(bytes.size(), 20u);  // version, byte count, two empty counts
    const CatalogMeta back = DecodeCatalog(bytes.data(), bytes.size());
    EXPECT_TRUE(back.tables.empty());
    EXPECT_TRUE(back.indexes.empty());
}

TEST(CatalogCodecTest, CatalogTruncatedOrCorruptThrows) {
    const std::vector<char> bytes = EncodeCatalog(SmallCatalog());
    for (size_t len = 0; len < bytes.size(); ++len) {
        std::vector<char> prefix(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(len));
        ExpectDamaged(prefix, "prefix " + std::to_string(len));
        // With the byte count patched, only the field reads can catch it.
        if (len >= kByteCountOffset + sizeof(uint64_t)) {
            const uint64_t patched = len;
            std::memcpy(prefix.data() + kByteCountOffset, &patched, sizeof(patched));
            ExpectDamaged(prefix, "patched prefix " + std::to_string(len));
        }
    }

    // The table name's length sits right after the table count and oid.
    std::vector<char> long_name = bytes;
    const uint32_t huge = UINT32_MAX;
    std::memcpy(long_name.data() + 12 + 4 + 4, &huge, sizeof(huge));
    ExpectDamaged(long_name, "name length past the end");

    std::vector<char> many_tables = bytes;
    std::memcpy(many_tables.data() + 12, &huge, sizeof(huge));
    ExpectDamaged(many_tables, "table count past the end");

    // Versions 1 and 2 are read; anything else is refused.
    for (const char other : {char{0}, char{3}}) {
        std::vector<char> version = bytes;
        version[0] = other;
        ExpectDamaged(version, "another version", "version " + std::to_string(other) + ", expected 1 to 2");
    }

    std::vector<char> trailing = bytes;
    trailing.push_back(0);
    const uint64_t count = trailing.size();
    std::memcpy(trailing.data() + kByteCountOffset, &count, sizeof(count));
    ExpectDamaged(trailing, "trailing byte");
}

TEST(CatalogCodecTest, BadShapesThrow) {
    const auto chunks = [](CatalogMeta& m) -> std::vector<ChunkMeta>& { return m.tables[0].row_groups[0].chunks; };
    ExpectDamagedAfter([&](CatalogMeta& m) { chunks(m).pop_back(); }, "row group with 1 chunks for 2 columns");
    ExpectDamagedAfter([&](CatalogMeta& m) { chunks(m)[0].value_count = 2999; }, "chunk of 2999 values");
    ExpectDamagedAfter([](CatalogMeta& m) { m.tables[0].row_count = 3001; }, "row count doesn't match");
    ExpectDamagedAfter([](CatalogMeta& m) { m.tables[0].row_groups[0].row_count = kRowGroupSize + 1; },
                       "row group of 65537 rows");
    ExpectDamagedAfter([&](CatalogMeta& m) { chunks(m)[1].dict_page_count = 3; }, "more dictionary pages");
    ExpectDamagedAfter([&](CatalogMeta& m) { chunks(m)[0].page_first_rows.push_back(1); }, "data page list");
    ExpectDamagedAfter([&](CatalogMeta& m) { chunks(m)[1].dict_page_first_entries.clear(); }, "dictionary page list");
    ExpectDamagedAfter([&](CatalogMeta& m) { chunks(m)[1].null_count = 3001; }, "more NULLs than values");
    ExpectDamagedAfter([](CatalogMeta& m) { m.tables[0].columns[1].number = 3; }, "column 2 numbered 3");
    ExpectDamagedAfter([](CatalogMeta& m) { m.indexes[0].table_oid = 99; }, "missing table 99");
    ExpectDamagedAfter([](CatalogMeta& m) { m.indexes[0].column_number = 3; }, "missing column 3");
    ExpectDamagedAfter([](CatalogMeta& m) { m.indexes[0].column_number = 0; }, "missing column 0");
    ExpectDamagedAfter([](CatalogMeta& m) { m.indexes[0].key_type = TypeId::INVALID; }, "key type 0");
    ExpectDamagedAfter([](CatalogMeta& m) { m.tables[0].stats.min_max["o_orderkey"].min = Value(); }, "value type 0");
    ExpectDamagedAfter([](CatalogMeta& m) { m.tables[0].columns[1].type.precision = 39; }, "NUMERIC(39,2)");
    ExpectDamagedAfter([](CatalogMeta& m) { m.tables[0].columns[1].type.scale = 16; }, "NUMERIC(15,16)");
    ExpectDamagedAfter([](CatalogMeta& m) { m.tables[0].columns[0].type.max_length = 7; }, "stray fields");
    ExpectDamagedAfter([](CatalogMeta& m) { m.tables[0].columns[0].type = ColumnType{ColumnTypeId::VARCHAR, 0, 0, 0}; },
                       "VARCHAR needs a max length");
    ExpectDamagedAfter([](CatalogMeta& m) { m.tables[0].columns[0].type.id = static_cast<ColumnTypeId>(11); },
                       "column type id 11");
    ExpectDamagedAfter([&](CatalogMeta& m) { chunks(m)[0].encoding = static_cast<Encoding>(5); }, "encoding 5");

    // Duplicates, which later lookups by name or oid can't tell apart.
    ExpectDamagedAfter([](CatalogMeta& m) {
        m.tables.push_back(m.tables[0]);
        m.tables[1].oid += 50;
    }, "two tables named orders");
    ExpectDamagedAfter([](CatalogMeta& m) {
        m.tables.push_back(m.tables[0]);
        m.tables[1].name = "orders2";
    }, "two tables with oid 16384");
    ExpectDamagedAfter([](CatalogMeta& m) {
        m.indexes.push_back(m.indexes[0]);
        m.indexes[1].name = "idx2";
    }, "two indexes with oid 16385");
    ExpectDamagedAfter([](CatalogMeta& m) {
        m.indexes.push_back(m.indexes[0]);
        m.indexes[1].oid += 50;
    }, "two indexes named idx_orders_key");
    ExpectDamagedAfter([](CatalogMeta& m) { m.tables[0].columns[1].name = "o_orderkey"; },
                       "two columns named o_orderkey in table orders");

    ExpectDamagedAfter([](CatalogMeta& m) { m.indexes[0].key_type = TypeId::INTEGER; },
                       "key type doesn't match its column");

    // The dictionary chunk has 3000 values, 5 of them NULL, so at most 2995 entries.
    ExpectDamagedAfter([&](CatalogMeta& m) { chunks(m)[1].dict_size = 2996; },
                       "dictionary of 2996 entries in a chunk allowing 2995");
    ExpectDamagedAfter([&](CatalogMeta& m) { chunks(m)[1].dict_size = UINT32_MAX; },
                       "dictionary of 4294967295 entries");
    ExpectDamagedAfter([&](CatalogMeta& m) { chunks(m)[0].dict_size = 1; },
                       "dictionary of 1 entries in a chunk allowing 0");
    CatalogMeta full_dict = SmallCatalog();
    chunks(full_dict)[1].dict_size = 2995;
    const std::vector<char> full_bytes = EncodeCatalog(full_dict);
    EXPECT_NO_THROW((void)DecodeCatalog(full_bytes.data(), full_bytes.size()));
}

// Finds where the nth copy (0-based) of a string starts in the blob.
std::ptrdiff_t FindNth(const std::vector<char>& bytes, const std::string& s, int n) {
    auto it = bytes.begin();
    for (int i = 0; i <= n; ++i) {
        it = std::search(i == 0 ? it : it + 1, bytes.end(), s.begin(), s.end());
        if (it == bytes.end()) return -1;
    }
    return it - bytes.begin();
}

TEST(CatalogCodecTest, StatsKeysOutOfOrderThrow) {
    // "o_totalprice" is a column name, then a distinct-count key, then a min / max key.
    // Turning it into "o_aotalprice" puts it before "o_orderkey".
    const std::vector<char> bytes = EncodeCatalog(SmallCatalog());
    const std::string key = "o_totalprice";
    const auto distinct_at = FindNth(bytes, key, 1);
    const auto range_at = FindNth(bytes, key, 2);
    ASSERT_GT(distinct_at, 0);
    ASSERT_GT(range_at, distinct_at);
    ASSERT_EQ(FindNth(bytes, key, 3), -1);

    std::vector<char> distinct = bytes;
    distinct[distinct_at + 2] = 'a';
    ExpectDamaged(distinct, "distinct keys out of order", "distinct count keys out of order");

    std::vector<char> range = bytes;
    range[range_at + 2] = 'a';
    ExpectDamaged(range, "min / max keys out of order", "min / max keys out of order");

    // A repeated key counts as out of order: "key_b" patched to "key_a".
    CatalogMeta twin = SmallCatalog();
    twin.tables[0].stats.distinct_counts = {{"key_a", 1}, {"key_b", 2}};
    std::vector<char> repeated = EncodeCatalog(twin);
    const auto b_at = FindNth(repeated, "key_b", 0);
    ASSERT_GT(b_at, 0);
    repeated[b_at + 4] = 'a';
    ExpectDamaged(repeated, "repeated distinct key", "distinct count keys out of order");
}

TEST(CatalogCodecTest, PageListShapeThrows) {
    // Chunk 0 is BITPACK, 3000 values on pages {0, 2000}.
    // Chunk 1 has 40 dictionary entries.
    const auto chunks = [](CatalogMeta& m) -> std::vector<ChunkMeta>& { return m.tables[0].row_groups[0].chunks; };
    ExpectDamagedAfter([&](CatalogMeta& m) {
        chunks(m)[0].page_count = 3;
        chunks(m)[0].page_first_rows = {0, 2000, 1000};
    }, "chunk data page list doesn't rise at entry 2");
    ExpectDamagedAfter([&](CatalogMeta& m) { chunks(m)[0].page_first_rows = {0, 0}; },
                       "chunk data page list doesn't rise at entry 1");
    ExpectDamagedAfter([&](CatalogMeta& m) { chunks(m)[0].page_first_rows = {5, 2000}; },
                       "chunk data page list starts at 5, not 0");
    ExpectDamagedAfter([&](CatalogMeta& m) { chunks(m)[0].page_first_rows = {0, 3000}; },
                       "chunk data page list ends at 3000, not below 3000");

    // Several dictionary pages, so the order can be wrong; one data page follows them.
    const auto dict_pages = [&](CatalogMeta& m, std::vector<uint32_t> entries) {
        chunks(m)[1].dict_page_count = static_cast<uint32_t>(entries.size());
        chunks(m)[1].page_count = chunks(m)[1].dict_page_count + 1;
        chunks(m)[1].dict_page_first_entries = std::move(entries);
    };
    ExpectDamagedAfter([&](CatalogMeta& m) { dict_pages(m, {0, 30, 10}); },
                       "chunk dictionary page list doesn't rise at entry 2");
    ExpectDamagedAfter([&](CatalogMeta& m) { dict_pages(m, {0, 0}); },
                       "chunk dictionary page list doesn't rise at entry 1");
    ExpectDamagedAfter([&](CatalogMeta& m) { dict_pages(m, {1, 30}); },
                       "chunk dictionary page list starts at 1, not 0");
    ExpectDamagedAfter([&](CatalogMeta& m) { dict_pages(m, {0, 40}); },
                       "chunk dictionary page list ends at 40, not below 40");
    ExpectDamagedAfter([&](CatalogMeta& m) { chunks(m)[1].dict_size = 0; },
                       "chunk dictionary page list ends at 0, not below 0");

    // Well-formed lists still decode.
    CatalogMeta good = SmallCatalog();
    dict_pages(good, {0, 39});
    chunks(good)[0].page_first_rows = {0, 2999};
    const std::vector<char> bytes = EncodeCatalog(good);
    EXPECT_NO_THROW((void)DecodeCatalog(bytes.data(), bytes.size()));
}

TEST(CatalogCodecTest, BadEnumBytesThrow) {
    // The index's unique flag is the blob's last byte.
    // Its key type sits 5 bytes before it.
    const std::vector<char> bytes = EncodeCatalog(SmallCatalog());
    std::vector<char> bad_bool = bytes;
    bad_bool.back() = 2;
    ExpectDamaged(bad_bool, "unique flag 2", "flag byte 2");
    std::vector<char> bad_type = bytes;
    bad_type[bytes.size() - 6] = 6;
    ExpectDamaged(bad_type, "key type 6", "key type 6");

    // The max scalar "big" is a tag byte, a u32 length, then its 3 bytes.
    const std::string big = "big";
    const auto at = std::search(bytes.begin(), bytes.end(), big.begin(), big.end()) - bytes.begin();
    std::vector<char> bad_tag = bytes;
    ASSERT_EQ(bad_tag[at - 5], 3);
    bad_tag[at - 5] = 4;
    ExpectDamaged(bad_tag, "scalar tag 4", "scalar tag 4");
    std::vector<char> bad_optional = bytes;
    bad_optional[at - 6] = 2;
    ExpectDamaged(bad_optional, "optional flag 2", "flag byte 2");
}

}  // namespace
}  // namespace shilmandb
