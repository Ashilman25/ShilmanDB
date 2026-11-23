#include "storage/columnar/columnar_table.hpp"
#include "buffer/buffer_pool_manager.hpp"
#include "common/exception.hpp"
#include "storage/columnar/field_parser.hpp"
#include "storage/disk_manager.hpp"
#include "storage/page.hpp"
#include <algorithm>
#include <cassert>
#include <utility>

namespace shilmandb {

namespace {

// Unpins a page when it goes out of scope, unless told to keep the pin.
class PinGuard {
public:
    PinGuard(BufferPoolManager* bpm, page_id_t pid) : bpm_(bpm), pid_(pid) {}
    ~PinGuard() {
        if (bpm_ != nullptr) (void)bpm_->UnpinPage(pid_, false);
    }
    PinGuard(const PinGuard&) = delete;
    PinGuard& operator=(const PinGuard&) = delete;

    // The caller takes over the pin.
    void Keep() { bpm_ = nullptr; }

private:
    BufferPoolManager* bpm_;
    page_id_t pid_;
};

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
        PinGuard guard(bpm_, pid);
        if (!page->IsVerified()) {
            if (!VerifyPage(page->GetData(), pid)) throw DatabaseException(table_ + ": page " + std::to_string(pid) + " failed its checksum");
            page->MarkVerified();
        }
        guard.Keep();
        return page->GetData();
    }

    void Unpin(uint32_t index) override { (void)bpm_->UnpinPage(chunk_.first_page + index, false); }

private:
    BufferPoolManager* bpm_;
    const ChunkMeta& chunk_;
    const std::string& table_;
};

// Pins the pages a walk expects in batches, one pool call per batch.
// Pages are used one at a time; a batch stays pinned until the next one.
// A big-value page is pinned alone beside the batch, never in it.
class BatchPoolPages : public ChunkPages {
public:
    BatchPoolPages(BufferPoolManager* bpm, const ChunkMeta& chunk, const std::string& table)
        : bpm_(bpm), chunk_(chunk), table_(table), limit_(PinLimit(*bpm)),
          first_big_(chunk.big_page_count <= chunk.page_count ? chunk.page_count - chunk.big_page_count : chunk.page_count) {}
    // Each release has its own try, so one failing can't keep the other's pins.
    ~BatchPoolPages() override {
        if (big_ != INVALID_PAGE_ID) {
            try {
                (void)bpm_->UnpinPage(big_, false);
            } catch (...) {
                // nothing to do from a destructor
            }
        }
        try {
            Release();
        } catch (...) {
            // nothing to do from a destructor
        }
    }
    BatchPoolPages(const BatchPoolPages&) = delete;
    BatchPoolPages& operator=(const BatchPoolPages&) = delete;

    // A new walk: the last one is done with its pages.
    void Expect(const uint32_t* indexes, size_t n) override {
        assert(std::none_of(indexes, indexes + n, [&](uint32_t i) { return i >= first_big_; }) && "big-value pages are never batched");
        Release();
        expected_.assign(indexes, indexes + n);
        next_ = 0;
    }

    const char* Pin(uint32_t index) override {
        if (index >= chunk_.page_count) {
            throw DatabaseException(table_ + ": page " + std::to_string(index) + " is past the chunk's " + std::to_string(chunk_.page_count));
        }
        if (index >= first_big_) return PinBig(index);
        assert(!in_use_ && "one page at a time");
        const page_id_t pid = chunk_.first_page + index;
        if (cursor_ >= held_ || batch_[cursor_] != pid) Load(index);
        Page* page = pages_[cursor_];
        Verify(page, pid);
        in_use_ = true;
        return page->GetData();
    }

    // A batch page stays pinned with its batch; the walk moves on.
    void Unpin(uint32_t index) override {
        if (index >= first_big_) {
            assert(big_ == static_cast<page_id_t>(chunk_.first_page + index) && "Unpin of a big-value page that isn't pinned");
            const page_id_t pid = big_;
            big_ = INVALID_PAGE_ID;
            (void)bpm_->UnpinPage(pid, false);
            return;
        }
        assert(in_use_ && "Unpin without Pin");
        in_use_ = false;
        ++cursor_;
    }

private:
    // At most kPinBatch, and the workers' batches together never pass half the pool.
    static size_t PinLimit(const BufferPoolManager& bpm) {
        const size_t share = bpm.GetPoolSize() / (2 * static_cast<size_t>(bpm.ConcurrentWorkers()));
        return std::clamp<size_t>(share, 1, kPinBatch);
    }

    // Releases the held batch, then pins the expected pages from index on, or index alone.
    void Load(uint32_t index) {
        Release();
        const bool expected = next_ < expected_.size() && expected_[next_] == index;
        const size_t count = expected ? std::min(limit_, expected_.size() - next_) : 1;
        batch_.resize(count);
        pages_.resize(count);
        for (size_t i = 0; i < count; ++i) batch_[i] = chunk_.first_page + (expected ? expected_[next_ + i] : index);
        held_ = bpm_->FetchPages(batch_.data(), count, pages_.data());
        if (held_ == 0) throw DatabaseException(table_ + ": no free frame for page " + std::to_string(batch_[0]));
        if (expected) next_ += held_;
    }

    // Checks a page's checksum the first time it is read from disk.
    void Verify(Page* page, page_id_t pid) const {
        if (!page->IsVerified()) {
            if (!VerifyPage(page->GetData(), pid)) throw DatabaseException(table_ + ": page " + std::to_string(pid) + " failed its checksum");
            page->MarkVerified();
        }
    }

    // Pins one big-value page on its own, leaving the batch as it is.
    const char* PinBig(uint32_t index) {
        assert(big_ == INVALID_PAGE_ID && "one big-value page at a time");
        const page_id_t pid = chunk_.first_page + index;
        Page* page = bpm_->FetchPage(pid);
        if (page == nullptr) throw DatabaseException(table_ + ": no free frame for page " + std::to_string(pid));
        PinGuard guard(bpm_, pid);
        Verify(page, pid);
        guard.Keep();
        big_ = pid;
        return page->GetData();
    }

    // Unpins the held batch in one pool call.
    void Release() {
        const size_t held = held_;
        held_ = 0;
        cursor_ = 0;
        if (held > 0) bpm_->UnpinPages(batch_.data(), held);
    }

    BufferPoolManager* bpm_;
    const ChunkMeta& chunk_;
    const std::string& table_;
    const size_t limit_;
    const uint32_t first_big_;        // big-value pages are this index and after
    std::vector<uint32_t> expected_;  // the walk's page indexes
    size_t next_{0};                  // the first expected page not yet pinned
    std::vector<page_id_t> batch_;    // the batch asked for; the first held_ are pinned
    std::vector<Page*> pages_;
    size_t held_{0};
    size_t cursor_{0};                // the batch entry the walk is on
    bool in_use_{false};
    page_id_t big_{INVALID_PAGE_ID};  // the big-value page pinned now, if any
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
    BatchPoolPages pages(bpm_, chunk, meta_.name);
    DecodeRows(chunk, pages, meta_.columns[column].type, rows, n, out);
}

void ColumnarTable::ReadRows(size_t group, uint32_t column, const uint32_t* rows, uint32_t n, __int128* out) const {
    const ChunkMeta& chunk = ChunkOfKind(group, column, PhysicalKind::INT128);
    BatchPoolPages pages(bpm_, chunk, meta_.name);
    DecodeRows(chunk, pages, meta_.columns[column].type, rows, n, out);
}

void ColumnarTable::ReadRows(size_t group, uint32_t column, const uint32_t* rows, uint32_t n, double* out) const {
    const ChunkMeta& chunk = ChunkOfKind(group, column, PhysicalKind::DOUBLE);
    BatchPoolPages pages(bpm_, chunk, meta_.name);
    DecodeRows(chunk, pages, meta_.columns[column].type, rows, n, out);
}

void ColumnarTable::ReadRows(size_t group, uint32_t column, const uint32_t* rows, uint32_t n, StringRef* out, StringArena& arena) const {
    const ChunkMeta& chunk = ChunkOfKind(group, column, PhysicalKind::STRING);
    BatchPoolPages pages(bpm_, chunk, meta_.name);
    DecodeRows(chunk, pages, meta_.columns[column].type, rows, n, out, arena);
}

void ColumnarTable::ReadCodes(size_t group, uint32_t column, const uint32_t* rows, uint32_t n, uint32_t* out) const {
    const ChunkMeta& chunk = DictionaryChunk(group, column, ChunkAt(group, column));
    BatchPoolPages pages(bpm_, chunk, meta_.name);
    DecodeCodes(chunk, pages, rows, n, out);
}

void ColumnarTable::ReadDictionary(size_t group, uint32_t column, std::vector<StringRef>& out, StringArena& arena) const {
    const ChunkMeta& chunk = DictionaryChunk(group, column, ChunkOfKind(group, column, PhysicalKind::STRING));
    BatchPoolPages pages(bpm_, chunk, meta_.name);
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
