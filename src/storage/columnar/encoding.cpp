#include "storage/columnar/encoding.hpp"
#include "common/exception.hpp"
#include "storage/columnar/crc32c.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <functional>
#include <string_view>
#include <type_traits>

namespace shilmandb {
namespace {

constexpr size_t kDataBytes = PAGE_SIZE - kPageHeaderSize;  // room after the header
constexpr uint32_t kMaxPageValues = UINT16_MAX;             // a header counts values in 16 bits
static_assert(sizeof(ColumnarPageHeader::value_count) == 2, "so no header count passes kMaxPageValues");

// Facts about a chunk, gathered in one pass before any encoding is built.
struct Stats {
    uint32_t count{0};
    uint32_t nulls{0};
    bool constant{false};  // CONSTANT applies
    bool packs{false};     // BITPACK applies
    std::optional<Scalar> min, max;
    std::optional<Scalar> value;  // the constant, when no row is NULL
};

// Page layout picked for a chunk: where each page starts.
struct Layout {
    Encoding encoding{Encoding::PLAIN};
    std::vector<uint32_t> data_pages;  // first row per data page
    std::vector<uint32_t> dict_pages;  // first entry per dictionary page
    [[nodiscard]] size_t PageCount() const { return data_pages.size() + dict_pages.size(); }
};

// The distinct values of a chunk and each row's code.
struct Dictionary {
    std::vector<uint32_t> rows;   // entry k is the value at this row
    std::vector<uint32_t> codes;  // per row; 0 for a NULL row
};

//  sizes

// Bytes of NULL bits for n values; none when the page stores no NULLs.
size_t NullBytes(size_t n, bool nulls) { return nulls ? (n + 7) / 8 : 0; }

// Bits needed to hold every offset up to range.
uint32_t BitsFor(uint64_t range) { return range == 0 ? 0 : 64 - __builtin_clzll(range); }

// Whether n values of `bits` each fit on one page.
bool FitsPacked(size_t n, uint32_t bits, bool nulls) {
    return n <= kMaxPageValues && NullBytes(n, nulls) + (n * bits + 7) / 8 <= kDataBytes;
}

// The most values of `bits` each that fit on one page.
uint32_t PackedCapacity(uint32_t bits, bool nulls) {
    const size_t per_value = bits + (nulls ? 1 : 0);
    size_t n = per_value == 0 ? kMaxPageValues : std::min<size_t>(kMaxPageValues, kDataBytes * 8 / per_value);
    while (!FitsPacked(n, bits, nulls)) --n;  // rounding up the bytes can cost a value or two
    return static_cast<uint32_t>(n);
}

// Whether n strings of `bytes` in total fit on one page, offsets included.
bool FitsStrings(size_t n, size_t bytes, bool nulls) {
    return n <= kMaxPageValues && NullBytes(n, nulls) + 2 * (n + 1) + bytes <= kDataBytes;
}

// Page starts for n values at a fixed number per page.
std::vector<uint32_t> EvenPages(uint32_t n, uint32_t per_page) {
    std::vector<uint32_t> starts;
    starts.reserve((n + per_page - 1) / per_page);
    for (uint32_t r = 0; r < n; r += per_page) starts.push_back(r);
    return starts;
}

// Page starts for strings: each page takes values while they fit.
template <typename Length>
std::vector<uint32_t> StringPages(uint32_t n, bool nulls, Length length) {
    std::vector<uint32_t> starts;
    uint32_t start = 0;
    while (start < n) {
        starts.push_back(start);
        size_t bytes = 0;
        uint32_t count = 0;
        while (start + count < n && FitsStrings(count + 1, bytes + length(start + count), nulls)) {
            bytes += length(start + count);
            ++count;
        }
        start += count;  // every value fits a page alone, so count is at least 1
    }
    return starts;
}

// One past the last row of page p: the next page's start, or n.
uint32_t PageEnd(const std::vector<uint32_t>& starts, size_t p, uint32_t n) {
    return p + 1 < starts.size() ? starts[p + 1] : n;
}

// How far x sits above lo, in unsigned arithmetic so no range overflows.
uint64_t Offset(int64_t lo, int64_t x) { return static_cast<uint64_t>(x) - static_cast<uint64_t>(lo); }
uint64_t Offset(__int128 lo, __int128 x) {
    return static_cast<uint64_t>(static_cast<unsigned __int128>(x) - static_cast<unsigned __int128>(lo));
}

//  stats

// Refuses a value that no page could hold.
[[noreturn]] void ThrowTooBig(size_t bytes) {
    throw DatabaseException("value of " + std::to_string(bytes) + " bytes is bigger than a page");
}

// Refuses a page that doesn't match what its chunk says.
[[noreturn]] void ThrowDamaged(const std::string& what) {
    throw DatabaseException("damaged column page: " + what);
}

// Min, max and whether every non-NULL integer is the same; checks values fit the type.
template <typename T, typename Get>
void IntegerStats(const ColumnVector& v, Stats& s, Get get) {
    bool any = false;
    T lo{}, hi{};
    for (uint32_t i = 0; i < s.count; ++i) {
        if (v.IsNull(i)) continue;
        const T x = get(i);
        if (!any) {
            lo = hi = x;
            any = true;
        } else {
            lo = std::min(lo, x);
            hi = std::max(hi, x);
        }
    }
    s.constant = s.nulls == s.count || (s.nulls == 0 && lo == hi);
    s.packs = !any || static_cast<unsigned __int128>(hi) - static_cast<unsigned __int128>(lo) <= UINT64_MAX;
    if (!any) return;

    // Narrow types must fit their PLAIN width; a boolean is only 0 or 1.
    const uint32_t width = PlainWidth(v.Type());
    if (width < 8) {
        const bool boolean = v.Type().id == ColumnTypeId::BOOLEAN;
        const T limit = static_cast<T>(1) << (8 * width - 1);
        const T low = boolean ? 0 : -limit;
        const T high = boolean ? 1 : limit - 1;
        if (lo < low || hi > high) {
            const T bad = lo < low ? lo : hi;
            throw DatabaseException("value " + std::to_string(static_cast<int64_t>(bad)) + " does not fit " + ColumnTypeName(v.Type()));
        }
    }
    s.min = Scalar(lo);
    s.max = Scalar(hi);
    if (s.nulls == 0 && s.constant) s.value = Scalar(lo);
}

// Doubles count as equal only by their bits. Any NaN means no min or max,
// since Value's <= and >= treat NaN as matching every value.
void DoubleStats(const ColumnVector& v, Stats& s) {
    bool any = false, any_number = false, any_nan = false, same_bits = true;
    uint64_t first_bits = 0;
    double lo = 0, hi = 0;
    for (uint32_t i = 0; i < s.count; ++i) {
        if (v.IsNull(i)) continue;
        const double x = v.GetDouble(i);

        uint64_t bits;
        std::memcpy(&bits, &x, sizeof(bits));
        if (!any) first_bits = bits;

        same_bits = same_bits && bits == first_bits;
        any = true;
        if (std::isnan(x)) {
            any_nan = true;
            continue;
        }
        if (!any_number) {
            lo = hi = x;
            any_number = true;
        } else {
            lo = std::min(lo, x);
            hi = std::max(hi, x);
        }
    }
    s.constant = s.nulls == s.count || (s.nulls == 0 && same_bits);
    if (any_number && !any_nan) {
        s.min = Scalar(lo);
        s.max = Scalar(hi);
    }
    if (s.nulls == 0 && s.constant && any) s.value = Scalar(v.GetDouble(0));
}

// String min and max are kept only when both are at most 64 bytes.
void StringStats(const ColumnVector& v, Stats& s) {
    bool any = false;
    std::string_view lo, hi;
    for (uint32_t i = 0; i < s.count; ++i) {
        const std::string_view x = v.GetString(i);
        if (x.size() > kMaxValueBytes) ThrowTooBig(x.size());
        if (v.IsNull(i)) continue;
        if (!any) {
            lo = hi = x;
            any = true;
        } else if (x < lo) {
            lo = x;
        } else if (x > hi) {
            hi = x;
        }
    }
    s.constant = s.nulls == s.count || (s.nulls == 0 && lo == hi);
    if (any && lo.size() <= 64 && hi.size() <= 64) {
        s.min = Scalar(std::string(lo));
        s.max = Scalar(std::string(hi));
    }
    if (s.nulls == 0 && s.constant && any) s.value = Scalar(std::string(lo));
}

// Gathers a chunk's stats by its physical kind.
Stats ComputeStats(const ColumnVector& v) {
    if (v.Size() > UINT32_MAX) throw DatabaseException("a column chunk holds at most 2^32 - 1 values");
    Stats s;
    s.count = static_cast<uint32_t>(v.Size());
    s.nulls = static_cast<uint32_t>(v.NullCount());
    switch (v.Kind()) {
        case PhysicalKind::INT64:
            IntegerStats<int64_t>(v, s, [&](uint32_t i) { return v.GetInt(i); });
            break;
        case PhysicalKind::INT128:
            IntegerStats<__int128>(v, s, [&](uint32_t i) { return v.GetWide(i); });
            break;
        case PhysicalKind::DOUBLE: DoubleStats(v, s); break;
        case PhysicalKind::STRING: StringStats(v, s); break;
    }
    return s;
}

//  layouts

// PLAIN pages: strings fill each page, fixed widths take a set number.
Layout PlainLayout(const ColumnVector& v, const Stats& s) {
    Layout layout;
    const bool nulls = s.nulls > 0;
    if (v.Kind() == PhysicalKind::STRING) {
        layout.data_pages = StringPages(s.count, nulls, [&](uint32_t i) { return v.GetString(i).size(); });
    } else {
        layout.data_pages = EvenPages(s.count, PackedCapacity(8 * PlainWidth(v.Type()), nulls));
    }
    return layout;
}

// Page starts for BITPACK: each page takes values while their range still fits.
template <typename Get>
std::vector<uint32_t> PackedPages(const ColumnVector& v, const Stats& s, Get get) {
    using T = decltype(get(0u));
    const bool nulls = s.nulls > 0;
    std::vector<uint32_t> starts;
    uint32_t start = 0;

    while (start < s.count) {
        starts.push_back(start);
        bool any = false;
        T lo{}, hi{};
        uint32_t bits = 0, count = 0;
        for (uint32_t row = start; row < s.count; ++row, ++count) {
            if (v.IsNull(row)) {
                if (!FitsPacked(count + 1, bits, nulls)) break;
                continue;
            }
            const T x = get(row);
            const T new_lo = any ? std::min(lo, x) : x;
            const T new_hi = any ? std::max(hi, x) : x;
            const uint32_t new_bits = BitsFor(Offset(new_lo, new_hi));
            if (!FitsPacked(count + 1, new_bits, nulls)) break;
            lo = new_lo;
            hi = new_hi;
            bits = new_bits;
            any = true;
        }
        start += count;
    }
    return starts;
}

// BITPACK pages for a 64-bit or 128-bit integer chunk.
Layout PackedLayout(const ColumnVector& v, const Stats& s) {
    Layout layout;
    layout.encoding = Encoding::BITPACK;
    if (v.Kind() == PhysicalKind::INT64) {
        layout.data_pages = PackedPages(v, s, [&](uint32_t i) { return v.GetInt(i); });
    } else {
        layout.data_pages = PackedPages(v, s, [&](uint32_t i) { return v.GetWide(i); });
    }
    return layout;
}

// Bits per dictionary code; a single entry needs none.
uint32_t CodeBits(size_t dict_size) { return dict_size <= 1 ? 0 : BitsFor(dict_size - 1); }

// Dictionary pages for the entries, then data pages of packed codes.
Layout DictionaryLayout(const ColumnVector& v, const Stats& s, const Dictionary& dict) {
    Layout layout;
    layout.encoding = Encoding::DICTIONARY;
    const auto entries = static_cast<uint32_t>(dict.rows.size());

    if (v.Kind() == PhysicalKind::STRING) {
        layout.dict_pages = StringPages(entries, false, [&](uint32_t k) { return v.GetString(dict.rows[k]).size(); });
    } else {
        layout.dict_pages = EvenPages(entries, PackedCapacity(8 * PlainWidth(v.Type()), false));
    }
    layout.data_pages = EvenPages(s.count, PackedCapacity(CodeBits(entries), s.nulls > 0));
    return layout;
}

// Scrambles a 64-bit value so nearby keys spread across the table.
uint64_t Mix(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

// Gives each distinct value a code, using an open-addressing table of row numbers.
// Gives up (nullopt) once the dictionary can no longer beat `best_pages`.
template <typename Hash, typename Equal, typename Bound>
std::optional<Dictionary> CollectEntries(const ColumnVector& v, const Stats& s, size_t best_pages, Hash hash, Equal equal, Bound lower_bound) {
    Dictionary dict;
    dict.codes.assign(s.count, 0);
    std::vector<uint32_t> slots(1024, 0);  // entry index + 1; 0 is empty
    size_t mask = slots.size() - 1;
    size_t entry_bytes = 0;

    for (uint32_t row = 0; row < s.count; ++row) {
        if (v.IsNull(row)) continue;
        size_t i = hash(row) & mask;
        while (slots[i] != 0 && !equal(dict.rows[slots[i] - 1], row)) i = (i + 1) & mask;
        if (slots[i] != 0) {
            dict.codes[row] = slots[i] - 1;
            continue;
        }
        const auto code = static_cast<uint32_t>(dict.rows.size());
        dict.rows.push_back(row);
        slots[i] = code + 1;
        dict.codes[row] = code;
        if (v.Kind() == PhysicalKind::STRING) entry_bytes += v.GetString(row).size();
        if (lower_bound(dict.rows.size(), entry_bytes) >= best_pages) return std::nullopt;

        if (dict.rows.size() * 2 > slots.size()) {  // keep the table at most half full
            slots.assign(slots.size() * 2, 0);
            mask = slots.size() - 1;
            for (uint32_t k = 0; k < dict.rows.size(); ++k) {
                size_t j = hash(dict.rows[k]) & mask;
                while (slots[j] != 0) j = (j + 1) & mask;
                slots[j] = k + 1;
            }
        }
    }
    return dict;
}

// Builds the dictionary with a hash and equality that fit the kind.
std::optional<Dictionary> MakeDictionary(const ColumnVector& v, const Stats& s, size_t best_pages) {
    const bool nulls = s.nulls > 0;
    const uint32_t per_page = v.Kind() == PhysicalKind::STRING ? 0 : PackedCapacity(8 * PlainWidth(v.Type()), false);
    // The fewest pages a dictionary with at least d entries can take.
    auto lower_bound = [&](size_t d, size_t bytes) {
        const size_t dict_pages = per_page != 0 ? (d + per_page - 1) / per_page : (2 * d + bytes + kDataBytes - 3) / (kDataBytes - 2);
        const uint32_t cap = PackedCapacity(CodeBits(d), nulls);
        return dict_pages + (s.count + cap - 1) / cap;
    };
    switch (v.Kind()) {
        case PhysicalKind::INT64:
            return CollectEntries(
                v, s, best_pages, [&](uint32_t r) { return Mix(static_cast<uint64_t>(v.GetInt(r))); },
                [&](uint32_t a, uint32_t b) { return v.GetInt(a) == v.GetInt(b); }, lower_bound
            );
        case PhysicalKind::INT128:
            return CollectEntries(
                v, s, best_pages,
                [&](uint32_t r) {
                    const auto x = static_cast<unsigned __int128>(v.GetWide(r));
                    return Mix(static_cast<uint64_t>(x) ^ Mix(static_cast<uint64_t>(x >> 64)));
                },
                [&](uint32_t a, uint32_t b) { return v.GetWide(a) == v.GetWide(b); }, lower_bound
            );
        case PhysicalKind::DOUBLE: {
            auto bits = [&](uint32_t r) {
                const double x = v.GetDouble(r);
                uint64_t b;
                std::memcpy(&b, &x, sizeof(b));
                return b;
            };
            return CollectEntries(
                v, s, best_pages, [&](uint32_t r) { return Mix(bits(r)); },
                [&](uint32_t a, uint32_t b) { return bits(a) == bits(b); }, lower_bound
            );
        }
        case PhysicalKind::STRING:
            return CollectEntries(
                v, s, best_pages, [&](uint32_t r) { return Mix(std::hash<std::string_view>{}(v.GetString(r))); },
                [&](uint32_t a, uint32_t b) { return v.GetString(a) == v.GetString(b); }, lower_bound
            );
    }
    return std::nullopt;
}

//  writing pages

// Packs values of a fixed bit width, low bits first.
class BitWriter {
public:
    explicit BitWriter(char* out) : out_(out) {}

    void Put(uint64_t value, uint32_t bits) {
        if (bits == 0) return;
        buffer_ |= static_cast<unsigned __int128>(value) << filled_;
        filled_ += bits;
        if (filled_ >= 64) {
            const auto word = static_cast<uint64_t>(buffer_);
            std::memcpy(out_, &word, sizeof(word));
            out_ += sizeof(word);
            buffer_ >>= 64;
            filled_ -= 64;
        }
    }
    // Writes the last partial word, only as many bytes as it uses.
    void Flush() {
        const auto word = static_cast<uint64_t>(buffer_);
        std::memcpy(out_, &word, (filled_ + 7) / 8);
    }

private:
    char* out_;
    unsigned __int128 buffer_{0};
    uint32_t filled_{0};
};

// Writes a page's header and NULL bits; returns where its data starts.
char* StartPage(char* page, PageKind kind, Encoding encoding, uint32_t bits, uint32_t first, uint32_t count, const ColumnVector* nulls_of, const void* fields = nullptr, size_t fields_len = 0) {
    ColumnarPageHeader h;
    h.kind = static_cast<uint8_t>(kind);
    h.encoding = static_cast<uint8_t>(encoding);
    h.flags = nulls_of != nullptr ? kPageHasNulls : 0;
    h.bit_width = static_cast<uint8_t>(bits);
    h.value_count = static_cast<uint16_t>(count);
    h.first_row = first;
    if (fields_len > 0) std::memcpy(h.fields, fields, fields_len);
    std::memcpy(page, &h, sizeof(h));

    char* data = page + kPageHeaderSize;
    if (nulls_of != nullptr) {
        for (uint32_t i = 0; i < count; ++i) {
            if (nulls_of->IsNull(first + i)) data[i >> 3] = static_cast<char>(data[i >> 3] | (1 << (i & 7)));
        }
        data += NullBytes(count, true);
    }
    return data;
}

// Writes PLAIN slots; slot i holds rows[i], or row first + i without a list.
void WritePlain(char* page, char* data, const ColumnVector& v, const uint32_t* rows, uint32_t first, uint32_t count) {
    auto row_at = [&](uint32_t i) { return rows != nullptr ? rows[i] : first + i; };
    const uint32_t width = PlainWidth(v.Type());
    switch (v.Kind()) {
        case PhysicalKind::INT64:
            for (uint32_t i = 0; i < count; ++i) {
                const int64_t x = v.GetInt(row_at(i));
                std::memcpy(data + static_cast<size_t>(i) * width, &x, width);  // low bytes first
            }
            break;
        case PhysicalKind::INT128:
            for (uint32_t i = 0; i < count; ++i) {
                const __int128 x = v.GetWide(row_at(i));
                std::memcpy(data + static_cast<size_t>(i) * sizeof(x), &x, sizeof(x));
            }
            break;
        case PhysicalKind::DOUBLE:
            for (uint32_t i = 0; i < count; ++i) {
                const double x = v.GetDouble(row_at(i));
                std::memcpy(data + static_cast<size_t>(i) * sizeof(x), &x, sizeof(x));
            }
            break;
        case PhysicalKind::STRING: {
            // Offsets from the page start, then the bytes of every string.
            char* bytes = data + 2 * (static_cast<size_t>(count) + 1);
            auto offset = static_cast<uint16_t>(bytes - page);
            std::memcpy(data, &offset, sizeof(offset));
            for (uint32_t i = 0; i < count; ++i) {
                const std::string_view x = v.GetString(row_at(i));
                std::memcpy(page + offset, x.data(), x.size());
                offset = static_cast<uint16_t>(offset + x.size());
                std::memcpy(data + 2 * (static_cast<size_t>(i) + 1), &offset, sizeof(offset));
            }
            break;
        }
    }
}

// One BITPACK page: the page minimum as the frame, then each value's offset from it.
template <typename Get>
void WritePackedPage(char* page, const ColumnVector& v, bool nulls, uint32_t first, uint32_t end, Get get) {
    using T = decltype(get(0u));
    bool any = false;
    T lo{}, hi{};
    for (uint32_t row = first; row < end; ++row) {
        if (v.IsNull(row)) continue;
        const T x = get(row);
        lo = any ? std::min(lo, x) : x;
        hi = any ? std::max(hi, x) : x;
        any = true;
    }
    const uint32_t bits = BitsFor(Offset(lo, hi));
    char* data = StartPage(page, PageKind::DATA, Encoding::BITPACK, bits, first, end - first, nulls ? &v : nullptr, &lo, sizeof(lo));
    BitWriter writer(data);
    for (uint32_t row = first; row < end; ++row) writer.Put(v.IsNull(row) ? 0 : Offset(lo, get(row)), bits);
    writer.Flush();
}

// Writes the chunk's meta and every page of the chosen layout.
EncodedChunk Build(const ColumnVector& v, const Stats& s, Layout layout, const Dictionary* dict) {
    EncodedChunk chunk;
    ChunkMeta& meta = chunk.meta;
    meta.encoding = layout.encoding;
    meta.value_count = s.count;
    meta.null_count = s.nulls;
    meta.min = s.min;
    meta.max = s.max;
    if (layout.encoding == Encoding::CONSTANT) {
        meta.constant = s.value;
        return chunk;
    }

    const bool nulls = s.nulls > 0;
    const ColumnVector* nulls_of = nulls ? &v : nullptr;
    meta.page_count = static_cast<uint32_t>(layout.PageCount());
    meta.dict_page_count = static_cast<uint32_t>(layout.dict_pages.size());
    chunk.pages.assign(static_cast<size_t>(meta.page_count) * PAGE_SIZE, 0);
    char* page = chunk.pages.data();

    if (layout.encoding == Encoding::DICTIONARY) {
        const auto entries = static_cast<uint32_t>(dict->rows.size());
        meta.dict_size = entries;
        for (size_t j = 0; j < layout.dict_pages.size(); ++j, page += PAGE_SIZE) {
            const uint32_t first = layout.dict_pages[j];
            const uint32_t count = PageEnd(layout.dict_pages, j, entries) - first;
            char* data = StartPage(page, PageKind::DICTIONARY, Encoding::PLAIN, 0, first, count, nullptr);
            WritePlain(page, data, v, dict->rows.data() + first, first, count);
        }
    }

    const uint32_t code_bits = layout.encoding == Encoding::DICTIONARY ? CodeBits(dict->rows.size()) : 0;
    for (size_t p = 0; p < layout.data_pages.size(); ++p, page += PAGE_SIZE) {
        const uint32_t first = layout.data_pages[p];
        const uint32_t end = PageEnd(layout.data_pages, p, s.count);
        switch (layout.encoding) {
            case Encoding::PLAIN: {
                char* data = StartPage(page, PageKind::DATA, Encoding::PLAIN, 0, first, end - first, nulls_of);
                WritePlain(page, data, v, nullptr, first, end - first);
                break;
            }
            case Encoding::BITPACK:
                if (v.Kind() == PhysicalKind::INT64) {
                    WritePackedPage(page, v, nulls, first, end, [&](uint32_t i) { return v.GetInt(i); });
                } else {
                    WritePackedPage(page, v, nulls, first, end, [&](uint32_t i) { return v.GetWide(i); });
                }
                break;
            case Encoding::DICTIONARY: {
                char* data = StartPage(page, PageKind::DATA, Encoding::DICTIONARY, code_bits, first, end - first, nulls_of);
                BitWriter writer(data);
                for (uint32_t row = first; row < end; ++row) writer.Put(dict->codes[row], code_bits);
                writer.Flush();
                break;
            }
            case Encoding::CONSTANT: break;
        }
    }
    meta.page_first_rows = std::move(layout.data_pages);
    meta.dict_page_first_entries = std::move(layout.dict_pages);
    return chunk;
}

//  reading pages

// A pinned page, unpinned when this goes out of scope.
class PinnedPage {
public:
    PinnedPage(ChunkPages& pages, uint32_t index) : pages_(pages), index_(index), data_(pages.Pin(index)) {}
    ~PinnedPage() { pages_.Unpin(index_); }
    PinnedPage(const PinnedPage&) = delete;
    PinnedPage& operator=(const PinnedPage&) = delete;
    [[nodiscard]] const char* Data() const { return data_; }

private:
    ChunkPages& pages_;
    uint32_t index_;
    const char* data_;
};

// A page's header plus where its NULL bits and data sit.
struct PageView {
    const char* page{nullptr};
    ColumnarPageHeader header;
    const uint8_t* null_bits{nullptr};  // nullptr when the page stores none
    const char* data{nullptr};
    size_t strings_start{0};            // PLAIN strings: the byte after the offset table

    [[nodiscard]] bool IsNull(uint32_t i) const {
        return null_bits != nullptr && ((null_bits[i >> 3] >> (i & 7)) & 1) != 0;
    }
    [[nodiscard]] size_t DataBit() const { return static_cast<size_t>(data - page) * 8; }
};

// Reads a page's header, throwing when it isn't the page the chunk says
// or when its values would run past the page.
PageView OpenPage(const char* page, PageKind kind, Encoding encoding, uint32_t first, const ColumnType& type) {
    PageView view;
    view.page = page;
    std::memcpy(&view.header, page, sizeof(view.header));
    if (view.header.kind != static_cast<uint8_t>(kind) || view.header.encoding != static_cast<uint8_t>(encoding)) {
        ThrowDamaged(
            "kind " + std::to_string(view.header.kind) + ", encoding " +
            std::to_string(view.header.encoding) + "; expected kind " +
            std::to_string(static_cast<int>(kind)) + ", encoding " +
            std::to_string(static_cast<int>(encoding))
        );
    }
    if (view.header.bit_width > 64) ThrowDamaged("bit width " + std::to_string(view.header.bit_width));
    if (view.header.first_row != first) {
        const char* unit = kind == PageKind::DATA ? "row " : "entry ";
        ThrowDamaged("page starts at " + std::string(unit) + std::to_string(view.header.first_row) + ", the chunk says " + std::to_string(first));
    }

    // Every byte the header implies must sit inside the page.
    const size_t n = view.header.value_count;
    const size_t null_bytes = NullBytes(n, (view.header.flags & kPageHasNulls) != 0);
    if (null_bytes > kDataBytes) ThrowDamaged("NULL bits for " + std::to_string(n) + " values run past the page");
    const size_t room = kDataBytes - null_bytes;
    const uint32_t width = PlainWidth(type);
    if (encoding != Encoding::PLAIN) {
        if (n * view.header.bit_width > room * 8) {
            ThrowDamaged(std::to_string(n) + " values of " + std::to_string(view.header.bit_width) + " bits run past the page");
        }
    } else if (width == 0) {
        if (2 * (n + 1) > room) ThrowDamaged("offsets for " + std::to_string(n) + " strings run past the page");
    } else if (n * width > room) {
        ThrowDamaged(std::to_string(n) + " values of " + std::to_string(width) + " bytes run past the page");
    }

    const char* p = page + kPageHeaderSize;
    if (null_bytes > 0) view.null_bits = reinterpret_cast<const uint8_t*>(p);
    view.data = p + null_bytes;
    view.strings_start = kPageHeaderSize + null_bytes + 2 * (n + 1);
    return view;
}

// Reads `bits` bits at bit position `bit` of a page, never past its end.
uint64_t ReadPacked(const char* page, size_t bit, uint32_t bits) {
    if (bits == 0) return 0;
    if (bit + bits > PAGE_SIZE * 8) ThrowDamaged("packed value at bit " + std::to_string(bit) + " runs past the page");
    const size_t byte = bit >> 3;
    const uint32_t shift = bit & 7;
    uint64_t word = 0;
    if (byte + sizeof(word) <= PAGE_SIZE) {
        std::memcpy(&word, page + byte, sizeof(word));
    } else {
        std::memcpy(&word, page + byte, PAGE_SIZE - byte);
    }
    uint64_t value = word >> shift;
    if (shift + bits > 64 && byte + 8 < PAGE_SIZE) {  // the value spills into a ninth byte
        value |= static_cast<uint64_t>(static_cast<unsigned char>(page[byte + 8])) << (64 - shift);
    }
    return bits == 64 ? value : value & ((uint64_t{1} << bits) - 1);
}

// Reads a little-endian integer of 1, 2, 4 or 8 bytes, sign-extended.
int64_t ReadPlainInt(const char* p, uint32_t width) {
    switch (width) {
        case 1: { int8_t x; std::memcpy(&x, p, 1); return x; }
        case 2: { int16_t x; std::memcpy(&x, p, 2); return x; }
        case 4: { int32_t x; std::memcpy(&x, p, 4); return x; }
        default: { int64_t x; std::memcpy(&x, p, 8); return x; }
    }
}

// Appends PLAIN slot i of a data or dictionary page.
void AppendPlain(const PageView& view, uint32_t i, uint32_t width, ColumnVector& out) {
    switch (out.Kind()) {
        case PhysicalKind::INT64: out.AppendInt(ReadPlainInt(view.data + static_cast<size_t>(i) * width, width)); break;
        case PhysicalKind::INT128: {
            __int128 x;
            std::memcpy(&x, view.data + static_cast<size_t>(i) * sizeof(x), sizeof(x));
            out.AppendWide(x);
            break;
        }
        case PhysicalKind::DOUBLE: {
            double x;
            std::memcpy(&x, view.data + static_cast<size_t>(i) * sizeof(x), sizeof(x));
            out.AppendDouble(x);
            break;
        }
        case PhysicalKind::STRING: {
            uint16_t from, to;
            std::memcpy(&from, view.data + 2 * static_cast<size_t>(i), sizeof(from));
            std::memcpy(&to, view.data + 2 * (static_cast<size_t>(i) + 1), sizeof(to));
            if (from < view.strings_start) {
                ThrowDamaged("string starts at byte " + std::to_string(from) + ", before the strings at byte " + std::to_string(view.strings_start));
            }
            if (to < from || to > PAGE_SIZE) ThrowDamaged("string offsets out of order");
            out.AppendString(std::string_view(view.page + from, to - from));
            break;
        }
    }
}

// Appends BITPACK slot i: the page's frame plus the packed offset.
void AppendPacked(const PageView& view, uint32_t i, ColumnVector& out) {
    const uint32_t bits = view.header.bit_width;
    const uint64_t offset = ReadPacked(view.page, view.DataBit() + static_cast<size_t>(i) * bits, bits);
    if (out.Kind() == PhysicalKind::INT64) {
        int64_t frame;
        std::memcpy(&frame, view.header.fields, sizeof(frame));
        out.AppendInt(static_cast<int64_t>(static_cast<uint64_t>(frame) + offset));
    } else if (out.Kind() == PhysicalKind::INT128) {
        __int128 frame;
        std::memcpy(&frame, view.header.fields, sizeof(frame));
        out.AppendWide(static_cast<__int128>(static_cast<unsigned __int128>(frame) + offset));
    } else {
        ThrowDamaged("BITPACK on a column that isn't an integer");
    }
}

// Reads dictionary code i, refusing one past the dictionary.
uint32_t ReadCode(const PageView& view, uint32_t i, uint32_t dict_size) {
    const uint32_t bits = view.header.bit_width;
    const auto code = static_cast<uint32_t>(ReadPacked(view.page, view.DataBit() + static_cast<size_t>(i) * bits, bits));
    if (code >= dict_size) ThrowDamaged("dictionary code " + std::to_string(code) + " past the dictionary");
    return code;
}

// Appends value i of an in-memory vector of the same kind.
void AppendFrom(const ColumnVector& from, size_t i, ColumnVector& out) {
    switch (out.Kind()) {
        case PhysicalKind::INT64:  out.AppendInt(from.GetInt(i)); break;
        case PhysicalKind::INT128: out.AppendWide(from.GetWide(i)); break;
        case PhysicalKind::DOUBLE: out.AppendDouble(from.GetDouble(i)); break;
        case PhysicalKind::STRING: out.AppendString(from.GetString(i)); break;
    }
}

// Appends n copies of a CONSTANT chunk's value, or n NULLs.
void AppendConstant(const ChunkMeta& meta, uint32_t n, ColumnVector& out) {
    if (!meta.constant.has_value()) {
        for (uint32_t i = 0; i < n; ++i) out.AppendNull();
        return;
    }
    const Scalar& c = *meta.constant;
    const bool matches = (out.Kind() == PhysicalKind::INT64 && std::holds_alternative<int64_t>(c)) ||
                         (out.Kind() == PhysicalKind::INT128 && std::holds_alternative<__int128>(c)) ||
                         (out.Kind() == PhysicalKind::DOUBLE && std::holds_alternative<double>(c)) ||
                         (out.Kind() == PhysicalKind::STRING && std::holds_alternative<std::string>(c));
    if (!matches) throw DatabaseException("damaged column chunk: constant of the wrong kind");
    for (uint32_t i = 0; i < n; ++i) {
        switch (out.Kind()) {
            case PhysicalKind::INT64:  out.AppendInt(std::get<int64_t>(c)); break;
            case PhysicalKind::INT128: out.AppendWide(std::get<__int128>(c)); break;
            case PhysicalKind::DOUBLE: out.AppendDouble(std::get<double>(c)); break;
            case PhysicalKind::STRING: out.AppendString(std::get<std::string>(c)); break;
        }
    }
}

}  // namespace

// CONSTANT when it applies, else the smallest of the rest.
EncodedChunk EncodeChunk(const ColumnVector& values) {
    const Stats s = ComputeStats(values);
    if (s.constant) {
        Layout constant;
        constant.encoding = Encoding::CONSTANT;
        return Build(values, s, std::move(constant), nullptr);
    }
    // Ties keep the earlier candidate: PLAIN, then BITPACK, then DICTIONARY.
    Layout best = PlainLayout(values, s);
    if (s.packs) {
        Layout packed = PackedLayout(values, s);
        if (packed.PageCount() < best.PageCount()) best = std::move(packed);
    }
    const auto dict = MakeDictionary(values, s, best.PageCount());
    if (dict.has_value()) {
        Layout coded = DictionaryLayout(values, s, *dict);
        if (coded.PageCount() < best.PageCount()) best = std::move(coded);
    }
    return Build(values, s, std::move(best), dict ? &*dict : nullptr);
}

// Builds exactly the encoding asked for, when it applies.
std::optional<EncodedChunk> TryEncodeChunk(const ColumnVector& values, Encoding encoding) {
    const Stats s = ComputeStats(values);
    switch (encoding) {
        case Encoding::CONSTANT: {
            if (!s.constant) return std::nullopt;
            Layout constant;
            constant.encoding = Encoding::CONSTANT;
            return Build(values, s, std::move(constant), nullptr);
        }
        case Encoding::PLAIN: return Build(values, s, PlainLayout(values, s), nullptr);
        case Encoding::BITPACK:
            if (!s.packs) return std::nullopt;
            return Build(values, s, PackedLayout(values, s), nullptr);
        case Encoding::DICTIONARY: {
            const auto dict = MakeDictionary(values, s, SIZE_MAX);
            return Build(values, s, DictionaryLayout(values, s, *dict), &*dict);
        }
    }
    return std::nullopt;
}

// The checksum skips its own 4 bytes and is seeded with the page id.
void SealPages(char* pages, uint32_t page_count, page_id_t first_page) {
    for (uint32_t i = 0; i < page_count; ++i) {
        char* page = pages + static_cast<size_t>(i) * PAGE_SIZE;
        const uint32_t crc = Crc32c(page + 4, PAGE_SIZE - 4, first_page + i);
        std::memcpy(page, &crc, sizeof(crc));
    }
}

// True when the stored checksum matches the page and its id.
bool VerifyPage(const char* page, page_id_t page_id) {
    uint32_t stored;
    std::memcpy(&stored, page, sizeof(stored));
    return stored == Crc32c(page + 4, PAGE_SIZE - 4, page_id);
}

// Copies the dictionary into memory first, so only one page is ever pinned.
void DecodeChunk(const ChunkMeta& meta, ChunkPages& pages, ColumnVector& out) {
    const size_t start_size = out.Size();
    out.Reserve(start_size + meta.value_count);
    if (meta.encoding == Encoding::CONSTANT) {
        AppendConstant(meta, meta.value_count, out);
        return;
    }
    const uint32_t width = PlainWidth(out.Type());
    const auto& entries = meta.dict_page_first_entries;
    if (entries.size() != meta.dict_page_count) ThrowDamaged("dictionary page list doesn't match its page count");

    // Each page must hold exactly the values between its start and the next.
    auto check_count = [](const std::vector<uint32_t>& starts, size_t p, uint32_t total, uint32_t n) {
        const uint32_t end = PageEnd(starts, p, total);
        const uint32_t want = end > starts[p] ? end - starts[p] : 0;
        if (n != want) ThrowDamaged("page holds " + std::to_string(n) + " values, the chunk says " + std::to_string(want));
    };

    ColumnVector dict(out.Type());
    if (meta.encoding == Encoding::DICTIONARY) {
        dict.Reserve(meta.dict_size);
        for (uint32_t j = 0; j < meta.dict_page_count; ++j) {
            PinnedPage pin(pages, j);
            const PageView view = OpenPage(pin.Data(), PageKind::DICTIONARY, Encoding::PLAIN, entries[j], out.Type());
            check_count(entries, j, meta.dict_size, view.header.value_count);
            for (uint32_t i = 0; i < view.header.value_count; ++i) AppendPlain(view, i, width, dict);
        }
    }

    for (uint32_t p = 0; p < meta.page_first_rows.size(); ++p) {
        PinnedPage pin(pages, meta.dict_page_count + p);
        const PageView view = OpenPage(pin.Data(), PageKind::DATA, meta.encoding, meta.page_first_rows[p], out.Type());
        const uint32_t n = view.header.value_count;
        if (n > meta.value_count - (out.Size() - start_size)) {
            ThrowDamaged("pages hold more than the chunk's " + std::to_string(meta.value_count) + " values");
        }
        check_count(meta.page_first_rows, p, meta.value_count, n);
        switch (meta.encoding) {
            case Encoding::PLAIN:
                for (uint32_t i = 0; i < n; ++i) {
                    if (view.IsNull(i)) out.AppendNull(); else AppendPlain(view, i, width, out);
                }
                break;
            case Encoding::BITPACK:
                for (uint32_t i = 0; i < n; ++i) {
                    if (view.IsNull(i)) out.AppendNull(); else AppendPacked(view, i, out);
                }
                break;
            case Encoding::DICTIONARY:
                for (uint32_t i = 0; i < n; ++i) {
                    if (view.IsNull(i)) out.AppendNull(); else AppendFrom(dict, ReadCode(view, i, dict.Size()), out);
                }
                break;
            case Encoding::CONSTANT: break;
        }
    }
    if (out.Size() - start_size != meta.value_count) {
        ThrowDamaged("pages hold " + std::to_string(out.Size() - start_size) + " values, the chunk " + std::to_string(meta.value_count));
    }
}

// Finds the row's page by binary search, then reads one slot of it.
void DecodeValue(const ChunkMeta& meta, ChunkPages& pages, uint32_t row, ColumnVector& out) {
    if (row >= meta.value_count) {
        throw DatabaseException("row " + std::to_string(row) + " is past the chunk's " + std::to_string(meta.value_count) + " values");
    }
    if (meta.encoding == Encoding::CONSTANT) {
        AppendConstant(meta, 1, out);
        return;
    }
    const auto& firsts = meta.page_first_rows;
    const auto page_it = std::upper_bound(firsts.begin(), firsts.end(), row);
    if (page_it == firsts.begin()) ThrowDamaged("no page holds row " + std::to_string(row));
    const auto p = static_cast<uint32_t>(page_it - firsts.begin() - 1);
    const uint32_t slot = row - firsts[p];
    const uint32_t width = PlainWidth(out.Type());

    uint32_t code = 0;
    {
        PinnedPage pin(pages, meta.dict_page_count + p);
        const PageView view = OpenPage(pin.Data(), PageKind::DATA, meta.encoding, firsts[p], out.Type());
        if (slot >= view.header.value_count) ThrowDamaged("row " + std::to_string(row) + " past its page");
        if (view.IsNull(slot)) {
            out.AppendNull();
            return;
        }
        switch (meta.encoding) {
            case Encoding::PLAIN: AppendPlain(view, slot, width, out); return;
            case Encoding::BITPACK: AppendPacked(view, slot, out); return;
            case Encoding::DICTIONARY: code = ReadCode(view, slot, meta.dict_size); break;
            case Encoding::CONSTANT: return;
        }
    }

    // The data page is unpinned; now pin the one dictionary page holding the entry.
    const auto& entries = meta.dict_page_first_entries;
    const auto entry_it = std::upper_bound(entries.begin(), entries.end(), code);
    if (entry_it == entries.begin()) ThrowDamaged("no dictionary page holds entry " + std::to_string(code));
    const auto j = static_cast<uint32_t>(entry_it - entries.begin() - 1);
    PinnedPage pin(pages, j);
    const PageView view = OpenPage(pin.Data(), PageKind::DICTIONARY, Encoding::PLAIN, entries[j], out.Type());
    const uint32_t entry = code - entries[j];
    if (entry >= view.header.value_count) ThrowDamaged("entry " + std::to_string(code) + " past its page");
    AppendPlain(view, entry, width, out);
}

// A few words for a physical kind, for error messages.
const char* PhysicalKindWords(PhysicalKind kind) {
    switch (kind) {
        case PhysicalKind::INT64:  return "64-bit integers";
        case PhysicalKind::INT128: return "128-bit integers";
        case PhysicalKind::DOUBLE: return "doubles";
        case PhysicalKind::STRING: return "strings";
    }
    return "values";
}

namespace {

//  reading rows into typed arrays

// The physical kind each output array of DecodeRows holds.
template <typename T>
constexpr PhysicalKind OutputKind() {
    if constexpr (std::is_same_v<T, int64_t>) return PhysicalKind::INT64;
    else if constexpr (std::is_same_v<T, __int128>) return PhysicalKind::INT128;
    else if constexpr (std::is_same_v<T, double>) return PhysicalKind::DOUBLE;
    else return PhysicalKind::STRING;
}

// Refuses an output array that doesn't match the column type's kind.
void RequireKind(const ColumnType& type, PhysicalKind kind) {
    if (PhysicalKindOf(type) != kind) {
        throw DatabaseException(ColumnTypeName(type) + " doesn't decode as " + PhysicalKindWords(kind));
    }
}

// Callers refuse chunks with NULLs first, so reaching one means the chunk is wrong.
[[noreturn]] void ThrowNullRow(uint32_t row) {
    throw DatabaseException("damaged column chunk: row " + std::to_string(row) + " is NULL");
}

// Checks the requested rows are ascending and inside the chunk.
void CheckRows(const ChunkMeta& meta, const uint32_t* rows, uint32_t n) {
    if (n == 0) return;
    if (rows != nullptr) {
        for (uint32_t k = 1; k < n; ++k) {
            if (rows[k] < rows[k - 1]) {
                throw DatabaseException("rows must be ascending: row " + std::to_string(rows[k]) + " comes after row " + std::to_string(rows[k - 1]));
            }
        }
    }
    const uint32_t last = rows != nullptr ? rows[n - 1] : n - 1;
    if (last >= meta.value_count) {
        throw DatabaseException("row " + std::to_string(last) + " is past the chunk's " + std::to_string(meta.value_count) + " values");
    }
}

// Starts begin at 0, rise, and stay below total: every value on exactly one page.
void CheckStarts(const std::vector<uint32_t>& starts, uint32_t total, const char* what) {
    if (total > 0 && starts.empty()) ThrowDamaged(std::string("no ") + what + " pages for " + std::to_string(total) + " values");
    for (size_t p = 0; p < starts.size(); ++p) {
        const bool rises = p == 0 ? starts[0] == 0 : starts[p] > starts[p - 1];
        if (!rises || starts[p] >= total) ThrowDamaged(std::string(what) + " page starts are out of order");
    }
}

// The page lists add up to the chunk's pages, every value on exactly one page.
void CheckPageLists(const ChunkMeta& meta) {
    if (meta.dict_page_first_entries.size() != meta.dict_page_count) ThrowDamaged("dictionary page list doesn't match its page count");
    if (meta.dict_page_count + meta.page_first_rows.size() != meta.page_count) ThrowDamaged("page lists don't match the chunk's page count");
    CheckStarts(meta.page_first_rows, meta.value_count, "data");
    CheckStarts(meta.dict_page_first_entries, meta.dict_size, "dictionary");
}

// Calls fn(k, slot) for requested rows k0..k1-1 of a page starting at row first.
template <typename Fn>
void ForEachSlot(const uint32_t* rows, uint32_t first, uint32_t k0, uint32_t k1, Fn fn) {
    if (rows == nullptr) {
        for (uint32_t k = k0; k < k1; ++k) fn(k, k - first);
    } else {
        for (uint32_t k = k0; k < k1; ++k) fn(k, rows[k] - first);
    }
}

// Throws for the first requested row on this page that is NULL.
void RequireNoNulls(const PageView& view, const uint32_t* rows, uint32_t first, uint32_t k0, uint32_t k1) {
    if (view.null_bits == nullptr) return;
    ForEachSlot(rows, first, k0, k1, [&](uint32_t, uint32_t slot) {
        if (view.IsNull(slot)) ThrowNullRow(first + slot);
    });
}

// Walks the data pages holding a requested row, in order, pinning each once.
// visit(view, first, k0, k1) gets rows k0..k1-1 on that page; pages.Expect gets the list first.
template <typename Visit>
void WalkDataPages(const ChunkMeta& meta, ChunkPages& pages, const ColumnType& type, const uint32_t* rows, uint32_t n, Visit visit) {
    struct Step {
        uint32_t page, k0, k1;
    };
    const auto& firsts = meta.page_first_rows;
    std::vector<Step> steps;
    std::vector<uint32_t> indexes;
    uint32_t k = 0;
    for (uint32_t p = 0; p < firsts.size() && k < n; ++p) {
        const uint32_t end = PageEnd(firsts, p, meta.value_count);
        const uint32_t k0 = k;
        if (rows == nullptr) {
            k = std::min(end, n);
        } else {
            while (k < n && rows[k] < end) ++k;
        }
        if (k == k0) continue;  // no requested row here, so the page is never pinned
        steps.push_back({p, k0, k});
        indexes.push_back(meta.dict_page_count + p);
    }
    pages.Expect(indexes.data(), indexes.size());

    for (const Step& s : steps) {
        const uint32_t first = firsts[s.page];
        const uint32_t end = PageEnd(firsts, s.page, meta.value_count);
        PinnedPage pin(pages, meta.dict_page_count + s.page);
        const PageView view = OpenPage(pin.Data(), PageKind::DATA, meta.encoding, first, type);
        if (view.header.value_count != end - first) {
            ThrowDamaged("page holds " + std::to_string(view.header.value_count) + " values, the chunk says " + std::to_string(end - first));
        }
        RequireNoNulls(view, rows, first, s.k0, s.k1);
        visit(view, first, s.k0, s.k1);
    }
}

// Walks the dictionary pages where wanted(e0, e1) holds, pinning each once.
// visit(view, e0, e1) gets the page's entries e0..e1-1; pages.Expect gets the list first.
template <typename Wanted, typename Visit>
void WalkDictionaryPages(const ChunkMeta& meta, ChunkPages& pages, const ColumnType& type, Wanted wanted, Visit visit) {
    const auto& entries = meta.dict_page_first_entries;
    std::vector<uint32_t> indexes;
    for (uint32_t j = 0; j < entries.size(); ++j) {
        if (wanted(entries[j], PageEnd(entries, j, meta.dict_size))) indexes.push_back(j);
    }
    pages.Expect(indexes.data(), indexes.size());

    for (const uint32_t j : indexes) {
        const uint32_t e0 = entries[j];
        const uint32_t e1 = PageEnd(entries, j, meta.dict_size);
        PinnedPage pin(pages, j);
        const PageView view = OpenPage(pin.Data(), PageKind::DICTIONARY, Encoding::PLAIN, e0, type);
        if (view.header.value_count != e1 - e0) {
            ThrowDamaged("page holds " + std::to_string(view.header.value_count) + " values, the chunk says " + std::to_string(e1 - e0));
        }
        visit(view, e0, e1);
    }
}

// Unpacks the packed values of slots s0..s1-1 of a page, calling emit(slot, value).
// Slots whose nine bytes sit inside the page skip ReadPacked's bounds checks.
template <typename Emit>
void UnpackSlots(const PageView& view, uint32_t s0, uint32_t s1, Emit emit) {
    const uint32_t bits = view.header.bit_width;
    if (bits == 0) {
        for (uint32_t s = s0; s < s1; ++s) emit(s, uint64_t{0});
        return;
    }

    const size_t base = view.DataBit();
    const uint64_t mask = bits == 64 ? ~uint64_t{0} : (uint64_t{1} << bits) - 1;
    constexpr size_t kLastFastBit = (PAGE_SIZE - 9) * 8 + 7;  // its word and spill byte end inside the page
    uint32_t fast_end = s0;
    if (base + static_cast<size_t>(s0) * bits <= kLastFastBit) {
        fast_end = static_cast<uint32_t>(std::min<size_t>(s1, (kLastFastBit - base) / bits + 1));
    }

    for (uint32_t s = s0; s < fast_end; ++s) {
        const size_t bit = base + static_cast<size_t>(s) * bits;
        const size_t byte = bit >> 3;
        const uint32_t shift = bit & 7;
        assert(byte + 9 <= PAGE_SIZE && "the fast path reads nine bytes inside the page");

        uint64_t word;
        std::memcpy(&word, view.page + byte, sizeof(word));
        uint64_t value = word >> shift;
        if (shift + bits > 64) value |= static_cast<uint64_t>(static_cast<unsigned char>(view.page[byte + 8])) << (64 - shift);
        emit(s, value & mask);
    }
    for (uint32_t s = fast_end; s < s1; ++s) emit(s, ReadPacked(view.page, base + static_cast<size_t>(s) * bits, bits));
}

// PLAIN slots stored as Stored, widened into out[k].
template <typename Stored, typename T>
void ReadFixedSlots(const PageView& view, const uint32_t* rows, uint32_t first, uint32_t k0, uint32_t k1, T* out) {
    const char* data = view.data;
    ForEachSlot(rows, first, k0, k1, [&](uint32_t k, uint32_t slot) {
        Stored x;
        std::memcpy(&x, data + static_cast<size_t>(slot) * sizeof(Stored), sizeof(Stored));
        out[k] = static_cast<T>(x);
    });
}

// Entry i of a PLAIN string page's offset table.
uint16_t StringOffset(const PageView& view, uint32_t i) {
    uint16_t x;
    std::memcpy(&x, view.data + 2 * static_cast<size_t>(i), sizeof(x));
    return x;
}

// PLAIN string slots s0..s1-1: one arena copy of their bytes, then a ref each.
void CopyStringRun(const PageView& view, uint32_t s0, uint32_t s1, StringRef* out, StringArena& arena) {
    if (s0 == s1) return;
    const uint16_t start = StringOffset(view, s0);
    if (start < view.strings_start) {
        ThrowDamaged("string starts at byte " + std::to_string(start) + ", before the strings at byte " + std::to_string(view.strings_start));
    }
    uint16_t end = start;
    for (uint32_t s = s0; s < s1; ++s) {
        const uint16_t to = StringOffset(view, s + 1);
        if (to < end || to > PAGE_SIZE) ThrowDamaged("string offsets out of order");
        end = to;
    }
    const StringRef bytes = arena.Copy(std::string_view(view.page + start, end - start));
    uint16_t from = start;
    for (uint32_t s = s0; s < s1; ++s) {
        const uint16_t to = StringOffset(view, s + 1);
        out[s - s0] = StringRef{bytes.data + (from - start), static_cast<uint32_t>(to - from)};
        from = to;
    }
}

// PLAIN string slots into out[k]: a run copied at once, a list string by string.
void ReadStringSlots(const PageView& view, const uint32_t* rows, uint32_t first, uint32_t k0, uint32_t k1, StringRef* out, StringArena& arena) {
    if (rows == nullptr) {
        CopyStringRun(view, k0 - first, k1 - first, out + k0, arena);
        return;
    }
    for (uint32_t k = k0; k < k1; ++k) CopyStringRun(view, rows[k] - first, rows[k] - first + 1, out + k, arena);
}

// PLAIN slots of a data or dictionary page into out[k], for each output kind.
template <typename T>
void ReadPlainSlots(const PageView& view, const ColumnType& type, const uint32_t* rows, uint32_t first, uint32_t k0, uint32_t k1, T* out, StringArena* arena) {
    if constexpr (std::is_same_v<T, int64_t>) {
        switch (PlainWidth(type)) {
            case 1:  ReadFixedSlots<int8_t>(view, rows, first, k0, k1, out); break;
            case 2:  ReadFixedSlots<int16_t>(view, rows, first, k0, k1, out); break;
            case 4:  ReadFixedSlots<int32_t>(view, rows, first, k0, k1, out); break;
            default: ReadFixedSlots<int64_t>(view, rows, first, k0, k1, out); break;
        }
    } else if constexpr (std::is_same_v<T, StringRef>) {
        ReadStringSlots(view, rows, first, k0, k1, out, *arena);
    } else {
        ReadFixedSlots<T>(view, rows, first, k0, k1, out);
    }
}

// Adds a packed offset to the page's frame, wrapping as the writer's subtraction did.
int64_t AddOffset(int64_t frame, uint64_t offset) {
    return static_cast<int64_t>(static_cast<uint64_t>(frame) + offset);
}
__int128 AddOffset(__int128 frame, uint64_t offset) {
    return static_cast<__int128>(static_cast<unsigned __int128>(frame) + offset);
}

// BITPACK slots into out[k]: a run unpacked in one pass, a list one ReadPacked each.
template <typename T>
void ReadPackedSlots(const PageView& view, const uint32_t* rows, uint32_t first, uint32_t k0, uint32_t k1, T* out) {
    T frame;
    std::memcpy(&frame, view.header.fields, sizeof(frame));
    if (rows == nullptr) {
        UnpackSlots(view, k0 - first, k1 - first, [&](uint32_t slot, uint64_t offset) { out[first + slot] = AddOffset(frame, offset); });
        return;
    }
    const uint32_t bits = view.header.bit_width;
    for (uint32_t k = k0; k < k1; ++k) {
        const size_t bit = view.DataBit() + static_cast<size_t>(rows[k] - first) * bits;
        out[k] = AddOffset(frame, ReadPacked(view.page, bit, bits));
    }
}

// The codes of the requested rows; each must fall inside the dictionary.
void ReadAllCodes(const ChunkMeta& meta, ChunkPages& pages, const ColumnType& type, const uint32_t* rows, uint32_t n, uint32_t* out) {
    WalkDataPages(meta, pages, type, rows, n, [&](const PageView& view, uint32_t first, uint32_t k0, uint32_t k1) {
        if (rows != nullptr) {
            for (uint32_t k = k0; k < k1; ++k) out[k] = ReadCode(view, rows[k] - first, meta.dict_size);
            return;
        }
        UnpackSlots(view, k0 - first, k1 - first, [&](uint32_t slot, uint64_t code) {
            if (code >= meta.dict_size) ThrowDamaged("dictionary code " + std::to_string(code) + " past the dictionary");
            out[first + slot] = static_cast<uint32_t>(code);
        });
    });
}

// A DICTIONARY chunk's rows: their codes, then only the dictionary pages they use.
template <typename T>
void DecodeDictionaryRows(const ChunkMeta& meta, ChunkPages& pages, const ColumnType& type, const uint32_t* rows, uint32_t n, T* out, StringArena* arena) {
    std::vector<uint32_t> codes(n);
    ReadAllCodes(meta, pages, type, rows, n, codes.data());
    std::vector<uint8_t> used(meta.dict_size, 0);
    for (uint32_t code : codes) used[code] = 1;

    std::vector<T> entries(meta.dict_size);
    WalkDictionaryPages(
        meta, pages, type,
        [&](uint32_t e0, uint32_t e1) { return std::find(used.begin() + e0, used.begin() + e1, uint8_t{1}) != used.begin() + e1; },
        [&](const PageView& view, uint32_t e0, uint32_t e1) { ReadPlainSlots(view, type, nullptr, e0, e0, e1, entries.data(), arena); }
    );
    for (uint32_t k = 0; k < n; ++k) out[k] = entries[codes[k]];
}

// A CONSTANT chunk's value as one output element; a string is copied into arena.
template <typename T>
T ConstantValue(const ChunkMeta& meta, uint32_t first_row, StringArena* arena) {
    if (!meta.constant.has_value()) ThrowNullRow(first_row);
    if constexpr (std::is_same_v<T, StringRef>) {
        const auto* s = std::get_if<std::string>(&*meta.constant);
        if (s == nullptr) throw DatabaseException("damaged column chunk: constant of the wrong kind");
        return arena->Copy(*s);
    } else {
        const auto* v = std::get_if<T>(&*meta.constant);
        if (v == nullptr) throw DatabaseException("damaged column chunk: constant of the wrong kind");
        return *v;
    }
}

// The body of the four DecodeRows overloads; arena is null except for strings.
template <typename T>
void DecodeTypedRows(const ChunkMeta& meta, ChunkPages& pages, const ColumnType& type, const uint32_t* rows, uint32_t n, T* out, StringArena* arena) {
    RequireKind(type, OutputKind<T>());
    CheckRows(meta, rows, n);
    if (n == 0) return;
    if (meta.encoding == Encoding::CONSTANT) {
        std::fill(out, out + n, ConstantValue<T>(meta, rows != nullptr ? rows[0] : 0, arena));
        return;
    }
    CheckPageLists(meta);
    switch (meta.encoding) {
        case Encoding::PLAIN:
            WalkDataPages(meta, pages, type, rows, n, [&](const PageView& view, uint32_t first, uint32_t k0, uint32_t k1) {
                ReadPlainSlots(view, type, rows, first, k0, k1, out, arena);
            });
            return;
        case Encoding::BITPACK:
            if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, __int128>) {
                WalkDataPages(meta, pages, type, rows, n, [&](const PageView& view, uint32_t first, uint32_t k0, uint32_t k1) {
                    ReadPackedSlots(view, rows, first, k0, k1, out);
                });
                return;
            }
            ThrowDamaged("BITPACK on a column that isn't an integer");
        case Encoding::DICTIONARY:
            DecodeDictionaryRows(meta, pages, type, rows, n, out, arena);
            return;
        case Encoding::CONSTANT: return;  // filled above
    }
    ThrowDamaged("unknown encoding " + std::to_string(static_cast<int>(meta.encoding)));
}

}  // namespace

// The four overloads share one body; each picks its output kind.
void DecodeRows(const ChunkMeta& meta, ChunkPages& pages, const ColumnType& type, const uint32_t* rows, uint32_t n, int64_t* out) {
    DecodeTypedRows(meta, pages, type, rows, n, out, nullptr);
}

void DecodeRows(const ChunkMeta& meta, ChunkPages& pages, const ColumnType& type, const uint32_t* rows, uint32_t n, __int128* out) {
    DecodeTypedRows(meta, pages, type, rows, n, out, nullptr);
}

void DecodeRows(const ChunkMeta& meta, ChunkPages& pages, const ColumnType& type, const uint32_t* rows, uint32_t n, double* out) {
    DecodeTypedRows(meta, pages, type, rows, n, out, nullptr);
}

void DecodeRows(const ChunkMeta& meta, ChunkPages& pages, const ColumnType& type, const uint32_t* rows, uint32_t n, StringRef* out, StringArena& arena) {
    DecodeTypedRows(meta, pages, type, rows, n, out, &arena);
}

// Reads only the data pages; the dictionary is never pinned.
void DecodeCodes(const ChunkMeta& meta, ChunkPages& pages, const uint32_t* rows, uint32_t n, uint32_t* out) {
    if (meta.encoding != Encoding::DICTIONARY) throw DatabaseException("only a DICTIONARY chunk has codes");
    CheckRows(meta, rows, n);
    if (n == 0) return;
    CheckPageLists(meta);
    ReadAllCodes(meta, pages, ColumnType::BigInt(), rows, n, out);
}

// Reads each dictionary page once, in order; the data pages are never pinned.
void DecodeDictionary(const ChunkMeta& meta, ChunkPages& pages, const ColumnType& type, std::vector<StringRef>& out, StringArena& arena) {
    RequireKind(type, PhysicalKind::STRING);
    if (meta.encoding != Encoding::DICTIONARY) throw DatabaseException("only a DICTIONARY chunk has a dictionary");
    CheckPageLists(meta);
    out.assign(meta.dict_size, StringRef{});
    WalkDictionaryPages(
        meta, pages, type, [](uint32_t, uint32_t) { return true; },
        [&](const PageView& view, uint32_t e0, uint32_t e1) { CopyStringRun(view, 0, e1 - e0, out.data() + e0, arena); }
    );
}

}  // namespace shilmandb
