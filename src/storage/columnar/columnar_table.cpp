#include "storage/columnar/columnar_table.hpp"
#include "buffer/buffer_pool_manager.hpp"
#include "common/exception.hpp"
#include "storage/columnar/field_parser.hpp"
#include "storage/disk_manager.hpp"
#include "storage/page.hpp"
#include <utility>

namespace shilmandb {

namespace {

// Serves a chunk's pages from the buffer pool.
// Checks each page once per read from disk.
class PoolPages : public ChunkPages {
public:
    PoolPages(BufferPoolManager* bpm, const ChunkMeta& chunk, const std::string& table) : bpm_(bpm), chunk_(chunk), table_(table) {}

    const char* Pin(uint32_t index) override {
        if (index >= chunk_.page_count) {
            throw DatabaseException(table_ + ": page " + std::to_string(index) + " is past the chunk's " + std::to_string(chunk_.page_count));
        }
        const page_id_t pid = chunk_.first_page + index;
        Page* page = bpm_->FetchPage(pid);
        if (page == nullptr) throw DatabaseException(table_ + ": no free frame for page " + std::to_string(pid));
        if (!page->IsVerified()) {
            if (!VerifyPage(page->GetData(), pid)) {
                (void)bpm_->UnpinPage(pid, false);
                throw DatabaseException(table_ + ": page " + std::to_string(pid) + " failed its checksum");
            }
            page->MarkVerified();
        }
        return page->GetData();
    }

    void Unpin(uint32_t index) override { (void)bpm_->UnpinPage(chunk_.first_page + index, false); }

private:
    BufferPoolManager* bpm_;
    const ChunkMeta& chunk_;
    const std::string& table_;
};

}  // namespace

// Turns a stored min or max into an executor value, as a cell is read.
Value ScalarToValue(const ColumnType& type, const Scalar& s) {
    ColumnVector one(type);
    bool matches = false;
    switch (one.Kind()) {
        case PhysicalKind::INT64:
            if (const auto* v = std::get_if<int64_t>(&s)) { one.AppendInt(*v); matches = true; }
            break;
        case PhysicalKind::INT128:
            if (const auto* v = std::get_if<__int128>(&s)) { one.AppendWide(*v); matches = true; }
            break;
        case PhysicalKind::DOUBLE:
            if (const auto* v = std::get_if<double>(&s)) { one.AppendDouble(*v); matches = true; }
            break;
        case PhysicalKind::STRING:
            if (const auto* v = std::get_if<std::string>(&s)) { one.AppendString(*v); matches = true; }
            break;
    }
    if (!matches) throw DatabaseException("a chunk's min or max doesn't match its column type " + ColumnTypeName(type));
    return ToExecutorValue(one, 0);
}

// Every group must have one chunk per column, or reads would index past them.
ColumnarTable::ColumnarTable(BufferPoolManager* bpm, ColumnarTableMeta meta) : bpm_(bpm), meta_(std::move(meta)) {
    for (size_t g = 0; g < meta_.row_groups.size(); ++g) {
        if (meta_.row_groups[g].chunks.size() != meta_.columns.size()) {
            throw DatabaseException(
                meta_.name + ": row group " + std::to_string(g) + " has " +
                std::to_string(meta_.row_groups[g].chunks.size()) + " chunks for " +
                std::to_string(meta_.columns.size()) + " columns"
            );
        }
    }
}

uint32_t ColumnarTable::RowGroupRows(size_t group) const {
    if (group >= meta_.row_groups.size()) {
        throw DatabaseException(meta_.name + ": no row group " + std::to_string(group));
    }
    return meta_.row_groups[group].row_count;
}

const ChunkMeta& ColumnarTable::ChunkAt(size_t group, uint32_t column) const {
    if (group >= meta_.row_groups.size()) {
        throw DatabaseException(meta_.name + ": no row group " + std::to_string(group));
    }
    if (column >= meta_.columns.size()) {
        throw DatabaseException(meta_.name + ": no column " + std::to_string(column));
    }
    return meta_.row_groups[group].chunks[column];
}

const ChunkMeta& ColumnarTable::Chunk(size_t group, uint32_t column, const ColumnVector& out) const {
    const ChunkMeta& chunk = ChunkAt(group, column);
    if (out.Type() != meta_.columns[column].type) {
        throw DatabaseException(meta_.name + ": column " + meta_.columns[column].name + " is " + ColumnTypeName(meta_.columns[column].type) + ", not " + ColumnTypeName(out.Type()));
    }
    return chunk;
}

const ChunkMeta& ColumnarTable::ChunkOfKind(size_t group, uint32_t column, PhysicalKind kind) const {
    const ChunkMeta& chunk = ChunkAt(group, column);
    const ColumnMeta& c = meta_.columns[column];
    if (PhysicalKindOf(c.type) != kind) {
        throw DatabaseException(meta_.name + ": column " + c.name + " is " + ColumnTypeName(c.type) + ", which doesn't read as " + PhysicalKindWords(kind));
    }
    return chunk;
}

void ColumnarTable::ReadChunk(size_t group, uint32_t column, ColumnVector& out) const {
    const ChunkMeta& chunk = Chunk(group, column, out);
    PoolPages pages(bpm_, chunk, meta_.name);
    DecodeChunk(chunk, pages, out);
}

void ColumnarTable::ReadValue(RowPosition pos, uint32_t column, ColumnVector& out) const {
    const ChunkMeta& chunk = Chunk(pos.row_group, column, out);
    if (pos.offset >= meta_.row_groups[pos.row_group].row_count) {
        throw DatabaseException(meta_.name + ": row group " + std::to_string(pos.row_group) + " has no row " + std::to_string(pos.offset));
    }
    PoolPages pages(bpm_, chunk, meta_.name);
    DecodeValue(chunk, pages, pos.offset, out);
}

const ChunkMeta& ColumnarTable::DictionaryChunk(size_t group, uint32_t column, const ChunkMeta& chunk) const {
    if (chunk.encoding != Encoding::DICTIONARY) {
        throw DatabaseException(meta_.name + ": column " + meta_.columns[column].name + " has no dictionary in row group " + std::to_string(group));
    }
    return chunk;
}

void ColumnarTable::ReadRows(size_t group, uint32_t column, const uint32_t* rows, uint32_t n, int64_t* out) const {
    const ChunkMeta& chunk = ChunkOfKind(group, column, PhysicalKind::INT64);
    PoolPages pages(bpm_, chunk, meta_.name);
    DecodeRows(chunk, pages, meta_.columns[column].type, rows, n, out);
}

void ColumnarTable::ReadRows(size_t group, uint32_t column, const uint32_t* rows, uint32_t n, __int128* out) const {
    const ChunkMeta& chunk = ChunkOfKind(group, column, PhysicalKind::INT128);
    PoolPages pages(bpm_, chunk, meta_.name);
    DecodeRows(chunk, pages, meta_.columns[column].type, rows, n, out);
}

void ColumnarTable::ReadRows(size_t group, uint32_t column, const uint32_t* rows, uint32_t n, double* out) const {
    const ChunkMeta& chunk = ChunkOfKind(group, column, PhysicalKind::DOUBLE);
    PoolPages pages(bpm_, chunk, meta_.name);
    DecodeRows(chunk, pages, meta_.columns[column].type, rows, n, out);
}

void ColumnarTable::ReadRows(size_t group, uint32_t column, const uint32_t* rows, uint32_t n, StringRef* out, StringArena& arena) const {
    const ChunkMeta& chunk = ChunkOfKind(group, column, PhysicalKind::STRING);
    PoolPages pages(bpm_, chunk, meta_.name);
    DecodeRows(chunk, pages, meta_.columns[column].type, rows, n, out, arena);
}

void ColumnarTable::ReadCodes(size_t group, uint32_t column, const uint32_t* rows, uint32_t n, uint32_t* out) const {
    const ChunkMeta& chunk = DictionaryChunk(group, column, ChunkAt(group, column));
    PoolPages pages(bpm_, chunk, meta_.name);
    DecodeCodes(chunk, pages, rows, n, out);
}

void ColumnarTable::ReadDictionary(size_t group, uint32_t column, std::vector<StringRef>& out, StringArena& arena) const {
    const ChunkMeta& chunk = DictionaryChunk(group, column, ChunkOfKind(group, column, PhysicalKind::STRING));
    PoolPages pages(bpm_, chunk, meta_.name);
    DecodeDictionary(chunk, pages, meta_.columns[column].type, out, arena);
}

std::unordered_map<std::string, MinMax> ColumnarTable::GroupRanges(size_t group) const {
    if (group >= meta_.row_groups.size()) {
        throw DatabaseException(meta_.name + ": no row group " + std::to_string(group));
    }
    std::unordered_map<std::string, MinMax> ranges;
    const RowGroupMeta& g = meta_.row_groups[group];
    for (size_t c = 0; c < meta_.columns.size(); ++c) {
        const ChunkMeta& chunk = g.chunks[c];
        if (!chunk.min || !chunk.max) continue;
        const ColumnType& type = meta_.columns[c].type;
        ranges.emplace(meta_.columns[c].name, MinMax{ScalarToValue(type, *chunk.min), ScalarToValue(type, *chunk.max)});
    }
    return ranges;
}

// One fetch_add for the whole run, so concurrent writers never share a page.
ChunkMeta WriteChunk(DiskManager* disk, const ColumnVector& values) {
    EncodedChunk encoded = EncodeChunk(values);
    const uint32_t count = encoded.meta.page_count;
    if (count == 0) return std::move(encoded.meta);

    const page_id_t first = disk->AllocatePages(count);
    encoded.meta.first_page = first;
    SealPages(encoded.pages.data(), count, first);
    for (uint32_t i = 0; i < count; ++i) {
        disk->WritePage(first + i, encoded.pages.data() + static_cast<size_t>(i) * PAGE_SIZE);
    }
    return std::move(encoded.meta);
}

}  // namespace shilmandb
