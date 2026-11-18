#include "executor/parallel/scan_source.hpp"
#include "common/exception.hpp"
#include "executor/columnar_scan_executor.hpp"
#include "planner/cost_model.hpp"
#include "storage/columnar/field_parser.hpp"
#include "storage/columnar/string_arena.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <utility>

namespace shilmandb::parallel {

namespace {

// One column of the current row group, decoded into the array its type uses.
struct GroupColumn {
    std::vector<int64_t> i64;
    std::vector<__int128> i128;
    std::vector<double> f64;
    std::vector<StringRef> str;
};

// A worker's decoded row group and how far into it the batches have got.
struct ScanLocal : LocalState {
    std::vector<GroupColumn> columns;  // one per slot
    std::vector<int64_t> micros;       // TIMESTAMP values before they become text
    StringArena arena;                 // the group's strings; cleared when the next morsel starts
    uint32_t rows{0};
    uint32_t cursor{0};
};

// Makes v hold at least n values; it never shrinks, so later groups reuse it.
template <typename T>
T* Room(std::vector<T>& v, uint32_t n) {
    if (v.size() < n) v.resize(n);
    return v.data();
}

// Decodes every row of one chunk. TIMESTAMP becomes the text today's scan shows.
void ReadColumn(const ColumnarTable& table, size_t group, uint32_t column, uint32_t rows, GroupColumn& out, ScanLocal& l) {
    const ColumnType& type = table.Meta().columns[column].type;
    switch (PhysicalKindOf(type)) {
        case PhysicalKind::INT64:
            if (type.id == ColumnTypeId::TIMESTAMP) {
                table.ReadRows(group, column, nullptr, rows, Room(l.micros, rows));
                StringRef* text = Room(out.str, rows);
                for (uint32_t i = 0; i < rows; ++i) text[i] = l.arena.Copy(FormatTimestamp(l.micros[i]));
            } else {
                table.ReadRows(group, column, nullptr, rows, Room(out.i64, rows));
            }
            return;
        case PhysicalKind::INT128:
            table.ReadRows(group, column, nullptr, rows, Room(out.i128, rows));
            return;
        case PhysicalKind::DOUBLE:
            table.ReadRows(group, column, nullptr, rows, Room(out.f64, rows));
            return;
        case PhysicalKind::STRING:
            table.ReadRows(group, column, nullptr, rows, Room(out.str, rows), l.arena);
            return;
    }
}

// Copies n decoded values starting at row `from` into a batch slot.
void CopySlice(const GroupColumn& col, uint32_t from, uint32_t n, Vector& slot) {
    const VType& t = slot.Type();
    if (t.kind == Kind::STRING) {
        std::memcpy(slot.Str(), col.str.data() + from, n * sizeof(StringRef));
    } else if (t.kind == Kind::DOUBLE) {
        std::memcpy(slot.F64(), col.f64.data() + from, n * sizeof(double));
    } else if (t.kind == Kind::DECIMAL && t.wide) {
        std::memcpy(slot.I128(), col.i128.data() + from, n * sizeof(__int128));
    } else {
        std::memcpy(slot.I64(), col.i64.data() + from, n * sizeof(int64_t));
    }
}

}  // namespace

// Slot types come from the stored column types, as today's scan reads them.
ScanSource::ScanSource(const ColumnarTable* table, std::vector<uint32_t> columns, std::vector<const Expression*> skip_conjuncts)
    : table_(table), columns_(std::move(columns)), skip_conjuncts_(std::move(skip_conjuncts)) {
    const auto& meta = table_->Meta().columns;
    types_.reserve(columns_.size());
    for (uint32_t c : columns_) {
        assert(c < meta.size() && "scan column past the table");
        types_.push_back(VTypeOf(meta[c].type));
    }
}

// A skipped row group never becomes a morsel, so it costs no page read.
void ScanSource::Prepare(QueryContext&) {
    groups_.clear();
    for (size_t g = 0; g < table_->RowGroupCount(); ++g) {
        if (skip_conjuncts_.empty() || CostModel::MayMatch(skip_conjuncts_, table_->GroupRanges(g))) groups_.push_back(g);
    }
}

// One decoded column per slot, sized on the first morsel.
std::unique_ptr<LocalState> ScanSource::MakeLocal() const {
    auto local = std::make_unique<ScanLocal>();
    local->columns.resize(columns_.size());
    return local;
}

// The arena is cleared here, so the last morsel's strings die with it.
void ScanSource::StartMorsel(size_t morsel, LocalState& local) {
    auto& l = static_cast<ScanLocal&>(local);
    const size_t group = groups_[morsel];

    // Refused before any read, with today's message.
    for (uint32_t c : columns_) {
        if (table_->ChunkAt(group, c).null_count > 0) throw DatabaseException(ColumnarNullError(table_->Meta().columns[c].name));
    }

    l.arena.Clear();
    l.rows = table_->RowGroupRows(group);
    l.cursor = 0;
    for (size_t k = 0; k < columns_.size(); ++k) ReadColumn(*table_, group, columns_[k], l.rows, l.columns[k], l);
}

// With no columns to hand up, a batch still carries its row count.
bool ScanSource::NextBatch(LocalState& local, Batch& out) {
    auto& l = static_cast<ScanLocal&>(local);
    if (l.cursor >= l.rows) return false;
    const uint32_t n = std::min(kBatchSize, l.rows - l.cursor);
    ShapeBatch(out, types_);
    out.Reset(n);
    for (size_t k = 0; k < columns_.size(); ++k) CopySlice(l.columns[k], l.cursor, n, out.slots[k]);
    l.cursor += n;
    return true;
}

}  // namespace shilmandb::parallel
