#pragma once
#include "catalog/table_stats.hpp"
#include "common/rid.hpp"
#include "storage/columnar/column_vector.hpp"
#include "storage/columnar/columnar_meta.hpp"
#include "storage/columnar/encoding.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace shilmandb {

class BufferPoolManager;
class DiskManager;

// Where a row sits in a columnar table: its row group and its offset there.
struct RowPosition {
    uint32_t row_group{0};
    uint16_t offset{0};
};

// An index keeps a row position in a RID: group as page, offset as slot.
[[nodiscard]] inline RID ToRid(RowPosition p) { return RID{p.row_group, p.offset}; }
[[nodiscard]] inline RowPosition FromRid(const RID& r) { return RowPosition{r.page_id, r.slot_id}; }

// Reads one stored columnar table through the buffer pool.
// Columns are numbered by their 0-based place in the table's meta.
class ColumnarTable {
public:
    ColumnarTable(BufferPoolManager* bpm, ColumnarTableMeta meta);

    [[nodiscard]] const ColumnarTableMeta& Meta() const { return meta_; }
    [[nodiscard]] size_t RowGroupCount() const { return meta_.row_groups.size(); }
    [[nodiscard]] uint32_t RowGroupRows(size_t group) const;

    // Appends one column of one row group to out; DatabaseException on a bad checksum.
    void ReadChunk(size_t group, uint32_t column, ColumnVector& out) const;

    // Appends the value at pos to out.
    void ReadValue(RowPosition pos, uint32_t column, ColumnVector& out) const;

    // The group's min and max per column name, as executor values.
    // A chunk without min and max has no entry.
    [[nodiscard]] std::unordered_map<std::string, MinMax> GroupRanges(size_t group) const;

private:
    // The chunk at group and column, after checking both and out's type.
    [[nodiscard]] const ChunkMeta& Chunk(size_t group, uint32_t column, const ColumnVector& out) const;

    BufferPoolManager* bpm_;
    ColumnarTableMeta meta_;
};

// Turns a stored min or max into an executor value, as a cell is read.
[[nodiscard]] Value ScalarToValue(const ColumnType& type, const Scalar& s);

// Encodes one chunk, takes a run of pages and writes them; never uses the pool.
// Safe to call from several threads at once. Returns the chunk's meta with first_page set.
[[nodiscard]] ChunkMeta WriteChunk(DiskManager* disk, const ColumnVector& values);

}  // namespace shilmandb
