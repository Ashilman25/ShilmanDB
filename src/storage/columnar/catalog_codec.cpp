#include "storage/columnar/catalog_codec.hpp"
#include "common/exception.hpp"
#include "storage/columnar/superblock.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace shilmandb {

namespace {

// Where the u64 byte count sits, right after the u32 version.
constexpr size_t kByteCountOffset = sizeof(uint32_t);

// Smallest encoded sizes, so a corrupt count can't ask for more items than bytes left.
constexpr size_t kMinString = 4;
constexpr size_t kMinOptional = 1;
constexpr size_t kMinColumnType = 1 + 1 + 1 + 4;
constexpr size_t kMinColumn = kMinString + 4 + kMinColumnType + 1;
constexpr size_t kMinChunk = 4 + 4 + 4 + 1 + 4 + 4 + 4 + 4 + 4 + 3 * kMinOptional;
constexpr size_t kMinRowGroup = 4 + 4;
constexpr size_t kMinStats = 8 + 8 + 4 + 4;
constexpr size_t kMinTable = 4 + kMinString + 4 + 8 + 4 + kMinStats;
constexpr size_t kMinValue = 1 + 4;
constexpr size_t kMinDistinct = kMinString + 8;
constexpr size_t kMinMinMax = kMinString + 2 * kMinValue;
constexpr size_t kMinIndex = 4 + kMinString + 4 + 4 + 1 + 4 + 1;

// Every decode error goes through here, so they share one prefix.
[[noreturn]] void Damaged(const std::string& why) {
    throw DatabaseException("damaged catalog: " + why);
}

// Throws when the key was already seen; later lookups go by name and oid.
template <typename Key>
void CheckUnique(std::unordered_set<Key>& seen, const Key& key, const std::string& what) {
    if (!seen.insert(key).second) Damaged("two " + what);
}

// Appends fixed-width little-endian fields to the blob.
class Writer {
public:
    // Copies a fixed-width field as its little-endian bytes.
    template <typename T>
    void Put(T value) {
        static_assert(std::is_trivially_copyable_v<T>, "only plain fields go in raw");
        const char* p = reinterpret_cast<const char*>(&value);
        out_.insert(out_.end(), p, p + sizeof(T));
    }

    // Doubles go in as their raw bits, so NaN and -0.0 come back the same.
    void PutDouble(double value) {
        uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        Put(bits);
    }

    // A u32 length; anything longer can't be stored.
    void PutCount(size_t n) {
        if (n > std::numeric_limits<uint32_t>::max()) {
            throw DatabaseException("catalog list or string too long: " + std::to_string(n));
        }
        Put(static_cast<uint32_t>(n));
    }

    // Length first, then the bytes.
    void PutString(const std::string& s) {
        PutCount(s.size());
        out_.insert(out_.end(), s.begin(), s.end());
    }

    // One byte, 0 or 1.
    void PutBool(bool b) { Put(static_cast<uint8_t>(b ? 1 : 0)); }

    // The blob so far.
    std::vector<char>& Bytes() { return out_; }

private:
    std::vector<char> out_;
};

// Reads fields with a bounds check on every read.
class Reader {
public:
    Reader(const char* data, size_t size) : data_(data), size_(size) {}

    // Reads one fixed-width field, or throws when it runs past the end.
    template <typename T>
    T Get() {
        static_assert(std::is_trivially_copyable_v<T>, "only plain fields come out raw");
        Need(sizeof(T));
        T value;
        std::memcpy(&value, data_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return value;
    }

    // Reads a double from its raw bits.
    double GetDouble() {
        const uint64_t bits = Get<uint64_t>();
        double value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    // A count of items, each at least min_bytes, that must fit in what's left.
    uint32_t GetCount(size_t min_bytes, const char* what) {
        const uint32_t n = Get<uint32_t>();
        if (n > Left() / min_bytes) Damaged(std::to_string(n) + " " + what + " run past the end");
        return n;
    }

    // A length, checked against what's left, then the bytes.
    std::string GetString() {
        const uint32_t n = GetCount(1, "string bytes");
        std::string s(data_ + pos_, n);
        pos_ += n;
        return s;
    }

    // Only 0 or 1 count as a flag.
    bool GetBool() {
        const uint8_t b = Get<uint8_t>();
        if (b > 1) Damaged("flag byte " + std::to_string(b));
        return b == 1;
    }

    // Bytes not yet read.
    [[nodiscard]] size_t Left() const { return size_ - pos_; }

private:
    // Throws unless n more bytes are there.
    void Need(size_t n) const {
        if (n > Left()) Damaged("cut short at byte " + std::to_string(pos_));
    }

    const char* data_;
    size_t size_;
    size_t pos_{0};
};

// ---- writing ----

// Id, precision, scale, then max length.
void PutType(Writer& w, const ColumnType& type) {
    w.Put(type.id);
    w.Put(type.precision);
    w.Put(type.scale);
    w.Put(type.max_length);
}

// A tag byte (the variant index), then the payload.
void PutScalar(Writer& w, const Scalar& s) {
    w.Put(static_cast<uint8_t>(s.index()));
    switch (s.index()) {
        case 0: w.Put(std::get<int64_t>(s)); break;
        case 1: w.Put(std::get<__int128>(s)); break;
        case 2: w.PutDouble(std::get<double>(s)); break;
        default: w.PutString(std::get<std::string>(s)); break;
    }
}

// A present flag, then the scalar if there is one.
void PutOptionalScalar(Writer& w, const std::optional<Scalar>& s) {
    w.PutBool(s.has_value());
    if (s) PutScalar(w, *s);
}

// A TypeId byte, then the payload; an INVALID value writes only its byte.
void PutValue(Writer& w, const Value& v) {
    w.Put(v.type_);
    switch (v.type_) {
        case TypeId::INTEGER: w.Put(v.integer_); break;
        case TypeId::BIGINT:  w.Put(v.bigint_); break;
        case TypeId::DECIMAL: w.PutDouble(v.decimal_); break;
        case TypeId::VARCHAR: w.PutString(v.varchar_); break;
        case TypeId::DATE:    w.Put(v.date_); break;
        case TypeId::INVALID: break;
    }
}

// A count, then each u32.
void PutU32List(Writer& w, const std::vector<uint32_t>& list) {
    w.PutCount(list.size());
    for (uint32_t x : list) w.Put(x);
}

// Fixed fields, page lists, then min, max and constant.
void PutChunk(Writer& w, const ChunkMeta& c) {
    w.Put(c.first_page);
    w.Put(c.page_count);
    w.Put(c.dict_page_count);
    w.Put(c.encoding);
    w.Put(c.value_count);
    w.Put(c.null_count);
    PutU32List(w, c.page_first_rows);
    PutU32List(w, c.dict_page_first_entries);
    w.Put(c.dict_size);
    PutOptionalScalar(w, c.min);
    PutOptionalScalar(w, c.max);
    PutOptionalScalar(w, c.constant);
}

// A map's entries sorted by key, so the bytes don't depend on hash order.
template <typename Map>
std::vector<const typename Map::value_type*> SortedEntries(const Map& map) {
    std::vector<const typename Map::value_type*> entries;
    entries.reserve(map.size());
    for (const auto& entry : map) entries.push_back(&entry);
    std::sort(entries.begin(), entries.end(), [](auto* a, auto* b) { return a->first < b->first; });
    return entries;
}

// Counts, then both maps in key order.
void PutStats(Writer& w, const TableStats& stats) {
    w.Put(stats.row_count);
    w.Put(stats.page_count);
    w.PutCount(stats.distinct_counts.size());
    for (const auto* entry : SortedEntries(stats.distinct_counts)) {
        w.PutString(entry->first);
        w.Put(entry->second);
    }
    w.PutCount(stats.min_max.size());
    for (const auto* entry : SortedEntries(stats.min_max)) {
        w.PutString(entry->first);
        PutValue(w, entry->second.min);
        PutValue(w, entry->second.max);
    }
}

// Name and columns, row groups with their chunks, then stats.
void PutTable(Writer& w, const ColumnarTableMeta& t) {
    w.Put(t.oid);
    w.PutString(t.name);
    w.PutCount(t.columns.size());
    for (const auto& col : t.columns) {
        w.PutString(col.name);
        w.Put(col.number);
        PutType(w, col.type);
        w.PutBool(col.nullable);
    }
    w.Put(t.row_count);
    w.PutCount(t.row_groups.size());
    for (const auto& group : t.row_groups) {
        w.Put(group.row_count);
        w.PutCount(group.chunks.size());
        for (const auto& chunk : group.chunks) PutChunk(w, chunk);
    }
    PutStats(w, t.stats);
}

// Every index field, in struct order.
void PutIndex(Writer& w, const IndexMeta& index) {
    w.Put(index.oid);
    w.PutString(index.name);
    w.Put(index.table_oid);
    w.Put(index.column_number);
    w.Put(index.key_type);
    w.Put(index.root_page);
    w.PutBool(index.unique);
}

// ---- reading ----

// Rebuilds the type through its checking factory, so bad limits throw here.
ColumnType BuildType(uint8_t id, uint8_t precision, uint8_t scale, uint32_t max_length) {
    try {
        switch (static_cast<ColumnTypeId>(id)) {
            case ColumnTypeId::BOOLEAN:   return ColumnType::Boolean();
            case ColumnTypeId::SMALLINT:  return ColumnType::SmallInt();
            case ColumnTypeId::INTEGER:   return ColumnType::Integer();
            case ColumnTypeId::BIGINT:    return ColumnType::BigInt();
            case ColumnTypeId::NUMERIC:   return ColumnType::Numeric(precision, scale);
            case ColumnTypeId::DOUBLE:    return ColumnType::Double();
            case ColumnTypeId::DATE:      return ColumnType::Date();
            case ColumnTypeId::TIMESTAMP: return ColumnType::Timestamp();
            case ColumnTypeId::VARCHAR:   return ColumnType::Varchar(max_length);
            case ColumnTypeId::TEXT:      return ColumnType::Text();
        }
    } catch (const DatabaseException& e) {
        Damaged(std::string("column type: ") + e.what());
    }
    Damaged("column type id " + std::to_string(id));
}

// Also refuses a type whose unused fields aren't zero.
ColumnType GetType(Reader& r) {
    const auto id = r.Get<uint8_t>();
    const auto precision = r.Get<uint8_t>();
    const auto scale = r.Get<uint8_t>();
    const auto max_length = r.Get<uint32_t>();
    const ColumnType type = BuildType(id, precision, scale, max_length);
    if (type.precision != precision || type.scale != scale || type.max_length != max_length) {
        Damaged("column type " + ColumnTypeName(type) + " has stray fields");
    }
    return type;
}

// A tag byte picks which of the four kinds follows.
Scalar GetScalar(Reader& r) {
    const auto tag = r.Get<uint8_t>();
    switch (tag) {
        case 0: return Scalar{r.Get<int64_t>()};
        case 1: return Scalar{r.Get<__int128>()};
        case 2: return Scalar{r.GetDouble()};
        case 3: return Scalar{r.GetString()};
        default: Damaged("scalar tag " + std::to_string(tag));
    }
}

// A present flag, then the scalar if set.
std::optional<Scalar> GetOptionalScalar(Reader& r) {
    if (!r.GetBool()) return std::nullopt;
    return GetScalar(r);
}

// A TypeId byte, then its payload; INVALID is refused.
Value GetValue(Reader& r) {
    const auto type = r.Get<uint8_t>();
    switch (static_cast<TypeId>(type)) {
        case TypeId::INTEGER: return Value(r.Get<int32_t>());
        case TypeId::BIGINT:  return Value(r.Get<int64_t>());
        case TypeId::DECIMAL: return Value(r.GetDouble());
        case TypeId::VARCHAR: return Value(r.GetString());
        case TypeId::DATE:    return Value::MakeDate(r.Get<int32_t>());
        case TypeId::INVALID: break;
    }
    Damaged("value type " + std::to_string(type));
}

// An index key type: any real TypeId.
TypeId GetTypeId(Reader& r) {
    const auto type = r.Get<uint8_t>();
    if (type < static_cast<uint8_t>(TypeId::INTEGER) || type > static_cast<uint8_t>(TypeId::DATE)) {
        Damaged("key type " + std::to_string(type));
    }
    return static_cast<TypeId>(type);
}

// One of the four encodings.
Encoding GetEncoding(Reader& r) {
    const auto e = r.Get<uint8_t>();
    if (e < static_cast<uint8_t>(Encoding::CONSTANT) || e > static_cast<uint8_t>(Encoding::DICTIONARY)) {
        Damaged("encoding " + std::to_string(e));
    }
    return static_cast<Encoding>(e);
}

// A count, checked before allocating, then each u32.
std::vector<uint32_t> GetU32List(Reader& r, const char* what) {
    std::vector<uint32_t> list(r.GetCount(sizeof(uint32_t), what));
    for (uint32_t& x : list) x = r.Get<uint32_t>();
    return list;
}

// Reads a chunk and checks the page lists readers index by.
ChunkMeta GetChunk(Reader& r, uint32_t group_rows) {
    ChunkMeta c;
    c.first_page = r.Get<uint32_t>();
    c.page_count = r.Get<uint32_t>();
    c.dict_page_count = r.Get<uint32_t>();
    c.encoding = GetEncoding(r);
    c.value_count = r.Get<uint32_t>();
    c.null_count = r.Get<uint32_t>();
    c.page_first_rows = GetU32List(r, "page first rows");
    c.dict_page_first_entries = GetU32List(r, "dictionary first entries");
    c.dict_size = r.Get<uint32_t>();
    c.min = GetOptionalScalar(r);
    c.max = GetOptionalScalar(r);
    c.constant = GetOptionalScalar(r);

    if (c.value_count != group_rows) {
        Damaged("chunk of " + std::to_string(c.value_count) + " values in a group of " + std::to_string(group_rows));
    }
    if (c.null_count > c.value_count) Damaged("more NULLs than values in a chunk");
    if (c.dict_page_count > c.page_count) Damaged("more dictionary pages than pages in a chunk");
    if (c.page_first_rows.size() != c.page_count - c.dict_page_count) {
        Damaged("chunk data page list doesn't match its page count");
    }
    if (c.dict_page_first_entries.size() != c.dict_page_count) {
        Damaged("chunk dictionary page list doesn't match its page count");
    }
    // A dictionary holds distinct non-NULL values; the reader reserves dict_size entries.
    const uint32_t max_dict = c.encoding == Encoding::DICTIONARY ? c.value_count - c.null_count : 0;
    if (c.dict_size > max_dict) {
        Damaged("dictionary of " + std::to_string(c.dict_size) + " entries in a chunk allowing " + std::to_string(max_dict));
    }
    return c;
}

// Both maps must come in strictly ascending key order.
void GetStats(Reader& r, TableStats& stats) {
    stats.row_count = r.Get<uint64_t>();
    stats.page_count = r.Get<uint64_t>();
    std::string last;
    const uint32_t distinct = r.GetCount(kMinDistinct, "distinct counts");
    for (uint32_t i = 0; i < distinct; ++i) {
        std::string key = r.GetString();
        if (i > 0 && key <= last) Damaged("distinct count keys out of order");
        stats.distinct_counts[key] = r.Get<uint64_t>();
        last = std::move(key);
    }
    const uint32_t ranges = r.GetCount(kMinMinMax, "min / max entries");
    for (uint32_t i = 0; i < ranges; ++i) {
        std::string key = r.GetString();
        if (i > 0 && key <= last) Damaged("min / max keys out of order");
        MinMax range;
        range.min = GetValue(r);
        range.max = GetValue(r);
        stats.min_max[key] = std::move(range);
        last = std::move(key);
    }
}

// Reads a table and checks its columns, groups and row counts line up.
ColumnarTableMeta GetTable(Reader& r) {
    ColumnarTableMeta t;
    t.oid = r.Get<uint32_t>();
    t.name = r.GetString();
    t.columns.resize(r.GetCount(kMinColumn, "columns"));
    std::unordered_set<std::string> names;
    for (size_t i = 0; i < t.columns.size(); ++i) {
        ColumnMeta& col = t.columns[i];
        col.name = r.GetString();
        CheckUnique(names, col.name, "columns named " + col.name + " in table " + t.name);
        col.number = r.Get<uint32_t>();
        if (col.number != i + 1) {
            Damaged("column " + std::to_string(i + 1) + " numbered " + std::to_string(col.number));
        }
        col.type = GetType(r);
        col.nullable = r.GetBool();
    }
    t.row_count = r.Get<uint64_t>();
    t.row_groups.resize(r.GetCount(kMinRowGroup, "row groups"));
    uint64_t rows = 0;
    for (RowGroupMeta& group : t.row_groups) {
        group.row_count = r.Get<uint32_t>();
        if (group.row_count > kRowGroupSize) {
            Damaged("row group of " + std::to_string(group.row_count) + " rows");
        }
        group.chunks.resize(r.GetCount(kMinChunk, "chunks"));
        if (group.chunks.size() != t.columns.size()) {
            Damaged("row group with " + std::to_string(group.chunks.size()) + " chunks for " + std::to_string(t.columns.size()) + " columns");
        }
        for (ChunkMeta& chunk : group.chunks) chunk = GetChunk(r, group.row_count);
        rows += group.row_count;
    }
    if (rows != t.row_count) Damaged("table " + t.name + " row count doesn't match its row groups");
    GetStats(r, t.stats);
    return t;
}

// Checks the index names a real table and column, with the column's key type.
IndexMeta GetIndex(Reader& r, const std::vector<ColumnarTableMeta>& tables) {
    IndexMeta index;
    index.oid = r.Get<uint32_t>();
    index.name = r.GetString();
    index.table_oid = r.Get<uint32_t>();
    index.column_number = r.Get<uint32_t>();
    index.key_type = GetTypeId(r);
    index.root_page = r.Get<uint32_t>();
    index.unique = r.GetBool();

    const auto table = std::find_if(tables.begin(), tables.end(), [&](const ColumnarTableMeta& t) { return t.oid == index.table_oid; });
    if (table == tables.end()) {
        Damaged("index " + index.name + " on missing table " + std::to_string(index.table_oid));
    }
    if (index.column_number < 1 || index.column_number > table->columns.size()) {
        Damaged("index " + index.name + " on missing column " + std::to_string(index.column_number));
    }
    if (index.key_type != ExecutorType(table->columns[index.column_number - 1].type)) {
        Damaged("index " + index.name + " key type doesn't match its column");
    }
    return index;
}

}  // namespace

// Version, byte count (filled in last), tables, then indexes.
std::vector<char> EncodeCatalog(const CatalogMeta& meta) {
    Writer w;
    w.Put(kFormatVersion);
    w.Put(uint64_t{0});
    w.PutCount(meta.tables.size());
    for (const auto& table : meta.tables) PutTable(w, table);
    w.PutCount(meta.indexes.size());
    for (const auto& index : meta.indexes) PutIndex(w, index);

    std::vector<char>& bytes = w.Bytes();
    const uint64_t total = bytes.size();
    std::memcpy(bytes.data() + kByteCountOffset, &total, sizeof(total));
    return std::move(bytes);
}

// Checks the version and byte count first, then every field and duplicate.
CatalogMeta DecodeCatalog(const char* data, size_t size) {
    Reader r(data, size);
    const auto version = r.Get<uint32_t>();
    if (version != kFormatVersion) {
        Damaged("version " + std::to_string(version) + ", expected " + std::to_string(kFormatVersion));
    }
    const auto total = r.Get<uint64_t>();
    if (total != size) Damaged("says " + std::to_string(total) + " bytes but has " + std::to_string(size));

    CatalogMeta meta;
    meta.tables.resize(r.GetCount(kMinTable, "tables"));
    std::unordered_set<std::string> table_names;
    std::unordered_set<uint32_t> table_oids;
    for (auto& table : meta.tables) {
        table = GetTable(r);
        CheckUnique(table_names, table.name, "tables named " + table.name);
        CheckUnique(table_oids, table.oid, "tables with oid " + std::to_string(table.oid));
    }
    meta.indexes.resize(r.GetCount(kMinIndex, "indexes"));
    std::unordered_set<std::string> index_names;
    std::unordered_set<uint32_t> index_oids;
    for (auto& index : meta.indexes) {
        index = GetIndex(r, meta.tables);
        CheckUnique(index_names, index.name, "indexes named " + index.name);
        CheckUnique(index_oids, index.oid, "indexes with oid " + std::to_string(index.oid));
    }
    if (r.Left() != 0) Damaged(std::to_string(r.Left()) + " bytes after the end");
    return meta;
}

}  // namespace shilmandb
