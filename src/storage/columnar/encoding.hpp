#pragma once
#include "common/config.hpp"
#include "storage/columnar/column_vector.hpp"
#include "storage/columnar/string_arena.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "the columnar file format is little-endian only");

namespace shilmandb {

// What a page of the columnar file holds. The numbers are part of the format.
enum class PageKind : uint8_t { SUPERBLOCK = 1, CATALOG = 2, DICTIONARY = 3, DATA = 4 };

// How a column chunk is stored. The numbers are part of the format.
enum class Encoding : uint8_t { CONSTANT = 1, PLAIN = 2, BITPACK = 3, DICTIONARY = 4 };

inline constexpr size_t kPageHeaderSize = 32;
inline constexpr uint8_t kPageHasNulls = 0x01;     // NULL bits follow the header
inline constexpr uint8_t kPageBigValues = 0x02;    // reserved for values bigger than a page

// The biggest string a page holds: one NULL-bits byte and two offsets sit beside it.
inline constexpr size_t kMaxValueBytes = PAGE_SIZE - kPageHeaderSize - 1 - 2 * sizeof(uint16_t);
static_assert(kMaxValueBytes == 8155, "a stored string is at most 8,155 bytes");

// The first 32 bytes of every page, stored exactly as laid out here.
struct ColumnarPageHeader {
    uint32_t crc32c{0};         // over bytes 4 to the end, seeded with the page id
    uint8_t kind{0};            // a PageKind
    uint8_t encoding{0};        // an Encoding
    uint8_t flags{0};
    uint8_t bit_width{0};       // packed bits per value
    uint16_t value_count{0};
    uint16_t reserved{0};
    uint32_t first_row{0};      // a data page's first row; a dictionary page's first entry
    uint8_t fields[16]{};       // the encoding's own fields, like BITPACK's frame of reference
};
static_assert(sizeof(ColumnarPageHeader) == kPageHeaderSize, "the page header is 32 bytes on disk");
static_assert(offsetof(ColumnarPageHeader, kind) == 4 && offsetof(ColumnarPageHeader, bit_width) == 7, "page header byte layout");
static_assert(offsetof(ColumnarPageHeader, value_count) == 8 && offsetof(ColumnarPageHeader, first_row) == 12 && offsetof(ColumnarPageHeader, fields) == 16, "page header byte layout");

// One physical value: what min, max and a constant chunk keep.
using Scalar = std::variant<int64_t, __int128, double, std::string>;

struct ChunkMeta {
    page_id_t first_page{INVALID_PAGE_ID};           // INVALID for CONSTANT
    uint32_t page_count{0};                          // dictionary pages first, then data pages
    uint32_t dict_page_count{0};
    Encoding encoding{Encoding::PLAIN};
    uint32_t value_count{0};
    uint32_t null_count{0};
    std::vector<uint32_t> page_first_rows;           // per data page: first row in the group
    std::vector<uint32_t> dict_page_first_entries;   // per dictionary page: first entry index
    uint32_t dict_size{0};
    std::optional<Scalar> min, max;                  // absent when all NULL, any NaN, or a string over 64 bytes
    std::optional<Scalar> constant;                  // CONSTANT with no NULLs
};

// pages.size() == meta.page_count * PAGE_SIZE, checksums still zero.
struct EncodedChunk {
    ChunkMeta meta;
    std::vector<char> pages;
};

// Builds every candidate that applies and keeps the smallest.
[[nodiscard]] EncodedChunk EncodeChunk(const ColumnVector& values);

// Builds one encoding, or nothing when that encoding doesn't apply.
[[nodiscard]] std::optional<EncodedChunk> TryEncodeChunk(const ColumnVector& values, Encoding encoding);

// Writes each page's checksum; it covers the page bytes and its page id.
void SealPages(char* pages, uint32_t page_count, page_id_t first_page);
[[nodiscard]] bool VerifyPage(const char* page, page_id_t page_id);

// Hands out a chunk's pages by index; the reader backs it with the buffer pool.
class ChunkPages {
public:
    virtual ~ChunkPages() = default;
    [[nodiscard]] virtual const char* Pin(uint32_t index) = 0;
    virtual void Unpin(uint32_t index) = 0;

    // Before a walk: the page indexes it will pin, in order, one at a time.
    // Lets a reader pin them in batches; ignoring it is fine.
    virtual void Expect(const uint32_t* indexes, size_t n) {
        (void)indexes;
        (void)n;
    }
};

// Appends all value_count values, holding at most one page at a time.
void DecodeChunk(const ChunkMeta& meta, ChunkPages& pages, ColumnVector& out);

// Appends the value at one row of the chunk.
void DecodeValue(const ChunkMeta& meta, ChunkPages& pages, uint32_t row, ColumnVector& out);

// A few words for a physical kind, like "64-bit integers", for error messages.
[[nodiscard]] const char* PhysicalKindWords(PhysicalKind kind);

// Writes rows into out[0..n): rows lists ascending positions, or nullptr means rows 0..n-1.
// Pins each page holding a row once. A NULL row or a wrong-kind type throws.
void DecodeRows(const ChunkMeta& meta, ChunkPages& pages, const ColumnType& type, const uint32_t* rows, uint32_t n, int64_t* out);
void DecodeRows(const ChunkMeta& meta, ChunkPages& pages, const ColumnType& type, const uint32_t* rows, uint32_t n, __int128* out);
void DecodeRows(const ChunkMeta& meta, ChunkPages& pages, const ColumnType& type, const uint32_t* rows, uint32_t n, double* out);

// Strings are copied into arena, so they stay valid after the pages are unpinned.
void DecodeRows(const ChunkMeta& meta, ChunkPages& pages, const ColumnType& type, const uint32_t* rows, uint32_t n, StringRef* out, StringArena& arena);

// The dictionary codes of the listed rows, for any DICTIONARY chunk.
void DecodeCodes(const ChunkMeta& meta, ChunkPages& pages, const uint32_t* rows, uint32_t n, uint32_t* out);

// Replaces out with every dictionary entry, copied into arena. DICTIONARY string chunks only.
void DecodeDictionary(const ChunkMeta& meta, ChunkPages& pages, const ColumnType& type, std::vector<StringRef>& out, StringArena& arena);

}  // namespace shilmandb
