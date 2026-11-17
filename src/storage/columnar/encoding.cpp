#include "storage/columnar/encoding.hpp"
#include "common/exception.hpp"
#include "storage/columnar/crc32c.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <string_view>

namespace shilmandb {
namespace {

constexpr size_t kDataBytes = PAGE_SIZE - kPageHeaderSize;  // room after the header
constexpr uint32_t kMaxPageValues = UINT16_MAX;             // a header counts values in 16 bits
// The biggest string a page holds: one NULL-bits byte and two offsets sit beside it.
constexpr size_t kMaxValueBytes = kDataBytes - 1 - 2 * sizeof(uint16_t);

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

size_t NullBytes(size_t n, bool nulls) { return nulls ? (n + 7) / 8 : 0; }

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

uint32_t PageEnd(const std::vector<uint32_t>& starts, size_t p, uint32_t n) {
    return p + 1 < starts.size() ? starts[p + 1] : n;
}

// How far x sits above lo, in unsigned arithmetic so no range overflows.
uint64_t Offset(int64_t lo, int64_t x) { return static_cast<uint64_t>(x) - static_cast<uint64_t>(lo); }
uint64_t Offset(__int128 lo, __int128 x) {
    return static_cast<uint64_t>(static_cast<unsigned __int128>(x) - static_cast<unsigned __int128>(lo));
}

//  stats

[[noreturn]] void ThrowTooBig(size_t bytes) {
    throw DatabaseException("value of " + std::to_string(bytes) + " bytes is bigger than a page");
}

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

// Doubles count as equal only by their bits; NaN never becomes min or max.
void DoubleStats(const ColumnVector& v, Stats& s) {
    bool any = false, any_number = false, same_bits = true;
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
        if (std::isnan(x)) continue;
        if (!any_number) {
            lo = hi = x;
            any_number = true;
        } else {
            lo = std::min(lo, x);
            hi = std::max(hi, x);
        }
    }
    s.constant = s.nulls == s.count || (s.nulls == 0 && same_bits);
    if (any_number) {
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

uint32_t CodeBits(size_t dict_size) { return dict_size <= 1 ? 0 : BitsFor(dict_size - 1); }

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

    [[nodiscard]] bool IsNull(uint32_t i) const {
        return null_bits != nullptr && ((null_bits[i >> 3] >> (i & 7)) & 1) != 0;
    }
    [[nodiscard]] size_t DataBit() const { return static_cast<size_t>(data - page) * 8; }
};

// Reads a page's header, throwing when it isn't the page the chunk says.
PageView OpenPage(const char* page, PageKind kind, Encoding encoding) {
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
    const char* p = page + kPageHeaderSize;
    if ((view.header.flags & kPageHasNulls) != 0) {
        view.null_bits = reinterpret_cast<const uint8_t*>(p);
        p += NullBytes(view.header.value_count, true);
    }
    view.data = p;
    return view;
}

// Reads `bits` bits at bit position `bit` of a page, never past its end.
uint64_t ReadPacked(const char* page, size_t bit, uint32_t bits) {
    if (bits == 0) return 0;
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
    ColumnVector dict(out.Type());
    if (meta.encoding == Encoding::DICTIONARY) {
        dict.Reserve(meta.dict_size);
        for (uint32_t j = 0; j < meta.dict_page_count; ++j) {
            PinnedPage pin(pages, j);
            const PageView view = OpenPage(pin.Data(), PageKind::DICTIONARY, Encoding::PLAIN);
            for (uint32_t i = 0; i < view.header.value_count; ++i) AppendPlain(view, i, width, dict);
        }
    }

    for (uint32_t p = 0; p < meta.page_first_rows.size(); ++p) {
        PinnedPage pin(pages, meta.dict_page_count + p);
        const PageView view = OpenPage(pin.Data(), PageKind::DATA, meta.encoding);
        const uint32_t n = view.header.value_count;
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
        const PageView view = OpenPage(pin.Data(), PageKind::DATA, meta.encoding);
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
    const PageView view = OpenPage(pin.Data(), PageKind::DICTIONARY, Encoding::PLAIN);
    const uint32_t entry = code - entries[j];
    if (entry >= view.header.value_count) ThrowDamaged("entry " + std::to_string(code) + " past its page");
    AppendPlain(view, entry, width, out);
}

}  // namespace shilmandb
