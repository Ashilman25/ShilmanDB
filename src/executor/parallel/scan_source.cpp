#include "executor/parallel/scan_source.hpp"
#include "catalog/catalog.hpp"
#include "common/exception.hpp"
#include "executor/columnar_scan_executor.hpp"
#include "index/btree_index.hpp"
#include "planner/cost_model.hpp"
#include "storage/columnar/field_parser.hpp"
#include "storage/columnar/string_arena.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <numeric>
#include <utility>

namespace shilmandb::parallel {

namespace {

// The batch byte cap; 0 means kMaxBatchStringBytes. Only tests change it.
size_t test_batch_bytes = 0;

// One column's values for a morsel's survivors: entry i belongs to survivor i.
struct ColumnValues {
    std::vector<int64_t> i64;        // INTEGER, DATE and narrow DECIMAL
    std::vector<__int128> i128;      // wide DECIMAL
    std::vector<double> f64;
    std::vector<StringRef> str;      // plain strings, TIMESTAMP text included
    std::vector<uint32_t> codes;     // a DICTIONARY string chunk's codes
    std::vector<StringRef> entries;
    uint64_t str_bytes{0};           // at least the survivors' string bytes, each row counted
    bool coded{false};               // held as codes over entries
    bool ready{false};               // decoded and lined up with the survivors
};

// The longest entry's length.
uint64_t LongestEntry(const std::vector<StringRef>& entries) {
    uint32_t most = 0;
    for (const StringRef& e : entries) most = std::max(most, e.size);
    return most;
}

// Makes v hold at least n values; it never shrinks, so later groups reuse it.
template <typename T>
T* Room(std::vector<T>& v, uint32_t n) {
    if (v.size() < n) v.resize(n);
    return v.data();
}

// out[i] = from[rows[i]] for the first n survivors.
template <typename T>
void Gather(const std::vector<T>& from, const uint32_t* rows, uint32_t n, std::vector<T>& out) {
    T* o = Room(out, n);
    for (uint32_t i = 0; i < n; ++i) o[i] = from[rows[i]];
}

// Keeps v[kept[j]] as v[j]; kept is ascending, so in place is safe.
template <typename T>
void Keep(std::vector<T>& v, const std::vector<uint32_t>& kept) {
    for (size_t j = 0; j < kept.size(); ++j) v[j] = v[kept[j]];
}

// Calls fn on the array holding a column's values.
template <typename Fn>
void WithValues(ColumnValues& col, const VType& t, Fn&& fn) {
    if (col.coded) {
        fn(col.codes);
    } else if (t.kind == Kind::STRING) {
        fn(col.str);
    } else if (t.kind == Kind::DOUBLE) {
        fn(col.f64);
    } else if (t.kind == Kind::DECIMAL && t.wide) {
        fn(col.i128);
    } else {
        fn(col.i64);
    }
}

// Copies survivors [from, from + n) into a batch slot; coded strings stay coded.
void FillSlot(const ColumnValues& col, uint32_t from, uint32_t n, Vector& slot) {
    const VType& t = slot.Type();
    if (col.coded) {
        slot.SetDictionary(&col.entries);
        std::memcpy(slot.Codes(), col.codes.data() + from, n * sizeof(uint32_t));
    } else if (t.kind == Kind::STRING) {
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

// A worker's morsel: where the survivors sit and each column's values for them.
struct ScanSource::Local : LocalState {
    std::vector<ColumnValues> columns;  // one per table column
    ColumnValues whole;                 // a whole chunk, before the survivors are picked out
    std::vector<int64_t> micros;        // TIMESTAMP values before they become text
    std::vector<uint32_t> rows;         // survivors' positions in the group, ascending
    std::vector<uint32_t> kept;         // indexes into rows that passed the last conjunct
    std::vector<Batch> single;          // one per single-column conjunct
    std::vector<Batch> multi;           // one per multi-column conjunct
    std::vector<uint64_t> decoded;      // rows decoded per column in this morsel
    EvalScratch scratch;
    StringArena arena;                  // the morsel's strings; cleared when the next morsel starts
    size_t group{0};
    uint32_t group_rows{0};
    uint32_t count{0};                  // survivors; while it equals group_rows, rows is implied
    uint32_t cursor{0};                 // survivors handed up so far
    bool capped{false};                 // batches count string bytes against the cap
};

// Slot types come from the stored column types, as today's scan reads them.
InputLayout ScanLayout(const ColumnarTable& table, const std::vector<uint32_t>& columns) {
    const auto& meta = table.Meta().columns;
    InputLayout layout;
    layout.slot_of_column.assign(meta.size(), -1);
    for (size_t k = 0; k < columns.size(); ++k) {
        assert(columns[k] < meta.size() && "scan column past the table");
        const ColumnType& type = meta[columns[k]].type;
        layout.slot_of_column[columns[k]] = static_cast<int>(k);
        layout.slot_types.push_back(VTypeOf(type));
        layout.slot_precision.push_back(type.id == ColumnTypeId::NUMERIC ? type.precision : 0);
    }
    return layout;
}

// One column read: single. None or several: multi. Each list keeps written order.
void AddConjuncts(ScanSpec& spec, const ColumnarTable& table, const Expression& predicate) {
    std::vector<const Expression*> parts;
    SplitAnd(predicate, parts);
    for (const Expression* part : parts) {
        std::vector<uint32_t> columns;
        CollectColumns(*part, columns);
        auto compiled = CompiledExpr::Compile(*part, ScanLayout(table, columns));
        if (columns.size() == 1) {
            spec.single.push_back(std::move(compiled));
            spec.single_column.push_back(columns[0]);
        } else {
            spec.multi.push_back(std::move(compiled));
            spec.multi_columns.push_back(std::move(columns));
        }
    }
}

// The table is found in Prepare, where the catalog is known.
ScanSource::ScanSource(ScanSpec spec) : spec_(std::move(spec)) {
    assert(spec_.single.size() == spec_.single_column.size() && "one column per single conjunct");
    assert(spec_.multi.size() == spec_.multi_columns.size() && "one column list per multi conjunct");
}

// The index is found in Prepare too.
ScanSource::ScanSource(ScanSpec spec, IndexBounds bounds) : ScanSource(std::move(spec)) {
    bounds_ = std::move(bounds);
}

// Runs on the calling thread: per-column facts, zone maps, and fresh counts.
void ScanSource::Prepare(QueryContext& ctx) {
    const TableInfo* info = ctx.catalog->GetTable(spec_.table);
    if (info == nullptr) throw DatabaseException("Table not found: " + spec_.table);
    if (info->columnar == nullptr) throw DatabaseException("the parallel executor reads only columnar tables: " + spec_.table);
    table_ = info->columnar.get();

    const auto& meta = table_->Meta().columns;
    const size_t n = meta.size();
    types_.clear();
    for (const auto& column : meta) types_.push_back(VTypeOf(column.type));

    // After step s, a column is live if handed up or read later.
    handed_.assign(n, false);
    last_use_.assign(n, 0);
    slot_types_.clear();
    read_ = spec_.columns;
    for (uint32_t c : spec_.columns) {
        assert(c < n && "scan column past the table");
        handed_[c] = true;
        slot_types_.push_back(types_[c]);
    }
    for (size_t i = 0; i < spec_.single.size(); ++i) {
        last_use_[spec_.single_column[i]] = i + 1;
        read_.push_back(spec_.single_column[i]);
    }
    for (size_t i = 0; i < spec_.multi.size(); ++i) {
        for (uint32_t c : spec_.multi_columns[i]) {
            last_use_[c] = spec_.single.size() + i + 1;
            read_.push_back(c);
        }
    }
    std::sort(read_.begin(), read_.end());
    read_.erase(std::unique(read_.begin(), read_.end()), read_.end());
    string_slots_.clear();
    for (uint32_t c : spec_.columns) {
        if (types_[c].kind == Kind::STRING) string_slots_.push_back(c);
    }
    byte_cap_ = test_batch_bytes != 0 ? test_batch_bytes : kMaxBatchStringBytes;

    // The listed columns first, so the error names the column today's scan names.
    refused_ = spec_.refuse_nulls;
    for (uint32_t c : refused_) {
        if (c >= n) throw DatabaseException(spec_.table + ": no column " + std::to_string(c) + " to read");
    }
    for (uint32_t c : read_) {
        if (std::find(refused_.begin(), refused_.end(), c) == refused_.end()) refused_.push_back(c);
    }

    // A skipped row group never becomes a morsel, so it costs no page read.
    groups_.clear();
    if (bounds_) {
        FindMatches(ctx);
    } else {
        for (size_t g = 0; g < table_->RowGroupCount(); ++g) {
            if (spec_.skip_conjuncts.empty() || CostModel::MayMatch(spec_.skip_conjuncts, table_->GroupRanges(g))) groups_.push_back(g);
        }
    }

    groups_skipped_.store(table_->RowGroupCount() - groups_.size(), std::memory_order_relaxed);
    rows_considered_.store(0, std::memory_order_relaxed);
    rows_passed_.store(0, std::memory_order_relaxed);
    rows_decoded_ = std::make_unique<std::atomic<uint64_t>[]>(n);
    decoded_columns_ = n;
    for (size_t c = 0; c < n; ++c) rows_decoded_[c].store(0, std::memory_order_relaxed);
}

// Walks as today's index scan does: from low, stopping at the first key past high.
// With 2+ threads the range is cut into pieces, each walked by one pool task.
// Positions sort as (group, offset), so each group's pages are read in file order.
void ScanSource::FindMatches(QueryContext& ctx) {
    const IndexInfo* index = ctx.catalog->GetIndex(bounds_->index);
    if (index == nullptr) throw DatabaseException("IndexScanExecutor: index not found: " + bounds_->index);
    if (index->table_name != spec_.table) throw DatabaseException("index " + bounds_->index + " is not on table " + spec_.table);
    BTreeIndex& tree = *index->index;

    // Keeps an entry's position, group in the high bits and offset in the low 32.
    const auto keep = [&](const RID& rid, std::vector<uint64_t>& out) {
        const RowPosition pos = FromRid(rid);
        if (pos.row_group >= table_->RowGroupCount() || pos.offset >= table_->RowGroupRows(pos.row_group)) {
            throw DatabaseException("damaged index: " + bounds_->index + " points past table " + spec_.table);
        }
        out.push_back((static_cast<uint64_t>(pos.row_group) << 32) | pos.offset);
    };

    std::vector<Value> cuts;
    if (ctx.threads > 1 && ctx.pool != nullptr) cuts = tree.SplitKeys(bounds_->low.value_or(Value()), bounds_->high.value_or(Value()), ctx.threads);

    std::vector<uint64_t> found;
    if (cuts.empty()) {
        for (auto it = bounds_->low ? tree.Begin(*bounds_->low) : tree.Begin(); !it.IsEnd(); ++it) {
            const auto entry = *it;
            if (bounds_->high && entry.first > *bounds_->high) break;
            keep(entry.second, found);
        }
    } else {
        std::vector<std::vector<uint64_t>> pieces(cuts.size() + 1);
        ForEachPoolTask(ctx, pieces.size(), [&](size_t i, unsigned) {
            const Value& from = i == 0 ? bounds_->low.value_or(Value()) : cuts[i - 1];
            const Value* until = i < cuts.size() ? &cuts[i] : nullptr;
            tree.ForEachEntry(from, [&](const Value& key, const RID& rid) {
                if (until != nullptr ? key >= *until : bounds_->high && key > *bounds_->high) return false;
                keep(rid, pieces[i]);
                return true;
            });
        });
        size_t total = 0;
        for (const auto& piece : pieces) total += piece.size();
        found.reserve(total);
        for (const auto& piece : pieces) found.insert(found.end(), piece.begin(), piece.end());
    }
    std::sort(found.begin(), found.end());
    found.erase(std::unique(found.begin(), found.end()), found.end());

    offsets_.clear();
    first_.clear();
    for (uint64_t p : found) {
        const auto group = static_cast<size_t>(p >> 32);
        if (groups_.empty() || groups_.back() != group) {
            groups_.push_back(group);
            first_.push_back(offsets_.size());
        }
        offsets_.push_back(static_cast<uint32_t>(p));
    }
    first_.push_back(offsets_.size());
}

// Arrays grow on the first morsels and are reused after that.
std::unique_ptr<LocalState> ScanSource::MakeLocal() const {
    assert(table_ != nullptr && "MakeLocal before Prepare");
    auto l = std::make_unique<Local>();
    l->columns.resize(types_.size());
    l->decoded.assign(types_.size(), 0);
    l->single.resize(spec_.single.size());
    for (size_t i = 0; i < spec_.single.size(); ++i) ShapeBatch(l->single[i], {types_[spec_.single_column[i]]});
    l->multi.resize(spec_.multi.size());
    for (size_t i = 0; i < spec_.multi.size(); ++i) {
        std::vector<VType> types;
        for (uint32_t c : spec_.multi_columns[i]) types.push_back(types_[c]);
        ShapeBatch(l->multi[i], types);
    }
    return l;
}

// Every single-column conjunct runs before every multi-column one, whatever the
// written order; each narrows the survivors. Then the handed-up columns, survivors only.
// Counts are added once per morsel, so several workers can share them.
void ScanSource::StartMorsel(size_t morsel, LocalState& local) const {
    auto& l = static_cast<Local&>(local);
    const size_t group = groups_[morsel];

    // Refused before any read, with today's message, whether or not c is decoded.
    for (uint32_t c : refused_) {
        if (table_->ChunkAt(group, c).null_count > 0) throw DatabaseException(ColumnarNullError(table_->Meta().columns[c].name));
    }

    // The last morsel's batches are all consumed, so its strings can go.
    l.arena.Clear();
    for (uint32_t c : read_) l.columns[c].ready = false;
    std::fill(l.decoded.begin(), l.decoded.end(), 0);
    l.group = group;
    l.group_rows = table_->RowGroupRows(group);
    l.count = l.group_rows;
    l.cursor = 0;
    if (bounds_) {
        // Distinct and ascending, so a group matched whole is rows 0 .. n - 1.
        const auto from = offsets_.begin() + static_cast<std::ptrdiff_t>(first_[morsel]);
        const auto to = offsets_.begin() + static_cast<std::ptrdiff_t>(first_[morsel + 1]);
        l.count = static_cast<uint32_t>(to - from);
        std::copy(from, to, Room(l.rows, l.count));
    }
    const uint32_t considered = l.count;

    const size_t singles = spec_.single.size();
    for (size_t i = 0; i < singles && l.count > 0; ++i) {
        const uint32_t c = spec_.single_column[i];
        if (!l.columns[c].ready) Decode(l, c);
        Apply(l, *spec_.single[i], &c, 1, l.single[i], i + 1);
    }
    for (size_t i = 0; i < spec_.multi.size() && l.count > 0; ++i) {
        const std::vector<uint32_t>& columns = spec_.multi_columns[i];
        for (uint32_t c : columns) {
            if (!l.columns[c].ready) Decode(l, c);
        }
        Apply(l, *spec_.multi[i], columns.data(), columns.size(), l.multi[i], singles + i + 1);
    }
    l.capped = false;
    if (l.count > 0) {
        for (uint32_t c : spec_.columns) {
            if (!l.columns[c].ready) Decode(l, c);
        }
        // Batches count their string bytes only when the morsel's strings could pass the cap.
        uint64_t most = 0;
        for (uint32_t c : string_slots_) {
            const ColumnValues& col = l.columns[c];
            most += col.coded ? uint64_t{kBatchSize} * LongestEntry(col.entries) : col.str_bytes;
        }
        l.capped = most > byte_cap_;
    }

    rows_considered_.fetch_add(considered, std::memory_order_relaxed);
    rows_passed_.fetch_add(l.count, std::memory_order_relaxed);
    for (uint32_t c : read_) {
        if (l.decoded[c] > 0) rows_decoded_[c].fetch_add(l.decoded[c], std::memory_order_relaxed);
    }
}

// Dense: the whole chunk read, then the survivors picked out. Sparse: only their rows.
// A DICTIONARY string chunk gives its entries once and a code per row.
void ScanSource::Decode(Local& l, uint32_t column) const {
    ColumnValues& col = l.columns[column];
    const ColumnType& type = table_->Meta().columns[column].type;
    const uint32_t n = l.count;
    const bool whole = static_cast<double>(n) > spec_.dense_fraction * static_cast<double>(l.group_rows);
    const bool all = n == l.group_rows;
    if (!whole && all) {
        uint32_t* every = Room(l.rows, n);
        std::iota(every, every + n, 0u);
    }
    const uint32_t* rows = whole ? nullptr : l.rows.data();
    const uint32_t count = whole ? l.group_rows : n;
    const bool gather = whole && !all;

    const size_t arena_before = l.arena.BytesUsed();

    // read writes count values: straight into target, or into scratch to gather from.
    auto fill = [&](auto& target, auto& scratch, auto read) {
        if (!gather) {
            read(Room(target, count));
            return;
        }
        read(Room(scratch, count));
        Gather(scratch, l.rows.data(), n, target);
    };

    col.coded = false;
    switch (PhysicalKindOf(type)) {
        case PhysicalKind::INT64:
            if (type.id == ColumnTypeId::TIMESTAMP) {
                // Formatted as today's scan shows it, for the survivors only.
                table_->ReadRows(l.group, column, rows, count, Room(l.micros, count));
                StringRef* text = Room(col.str, n);
                for (uint32_t i = 0; i < n; ++i) text[i] = l.arena.Copy(FormatTimestamp(l.micros[gather ? l.rows[i] : i]));
            } else {
                fill(col.i64, l.whole.i64, [&](int64_t* out) { table_->ReadRows(l.group, column, rows, count, out); });
            }
            break;
        case PhysicalKind::INT128:
            fill(col.i128, l.whole.i128, [&](__int128* out) { table_->ReadRows(l.group, column, rows, count, out); });
            break;
        case PhysicalKind::DOUBLE:
            fill(col.f64, l.whole.f64, [&](double* out) { table_->ReadRows(l.group, column, rows, count, out); });
            break;
        case PhysicalKind::STRING:
            if (table_->ChunkAt(l.group, column).encoding == Encoding::DICTIONARY) {
                table_->ReadDictionary(l.group, column, col.entries, l.arena);
                fill(col.codes, l.whole.codes, [&](uint32_t* out) { table_->ReadCodes(l.group, column, rows, count, out); });
                col.coded = true;
            } else {
                fill(col.str, l.whole.str, [&](StringRef* out) { table_->ReadRows(l.group, column, rows, count, out, l.arena); });
            }
            break;
    }
    col.ready = true;
    // A CONSTANT chunk's rows all point at one copy, so it counts once per survivor.
    const bool shared = PhysicalKindOf(type) == PhysicalKind::STRING && table_->ChunkAt(l.group, column).encoding == Encoding::CONSTANT;
    col.str_bytes = shared && n > 0 ? uint64_t{n} * col.str[0].size : l.arena.BytesUsed() - arena_before;
    l.decoded[column] += count;
}

// Batches of up to kBatchSize survivors, every row active, so the kernels see dense input.
void ScanSource::Apply(Local& l, const CompiledExpr& expr, const uint32_t* columns, size_t width, Batch& batch, size_t step) const {
    l.kept.clear();
    for (uint32_t from = 0; from < l.count; from += kBatchSize) {
        const uint32_t n = std::min(kBatchSize, l.count - from);
        batch.Reset(n);
        for (size_t k = 0; k < width; ++k) FillSlot(l.columns[columns[k]], from, n, batch.slots[k]);
        expr.Select(batch, batch.sel, l.scratch);
        for (uint32_t k = 0; k < batch.sel.Size(); ++k) l.kept.push_back(from + batch.sel[k]);
    }
    if (l.kept.size() < l.count) Narrow(l, step);
}

// A column no later step reads is dropped rather than compacted.
void ScanSource::Narrow(Local& l, size_t step) const {
    const auto m = static_cast<uint32_t>(l.kept.size());
    uint32_t* rows = Room(l.rows, m);
    if (l.count == l.group_rows) {
        std::copy(l.kept.begin(), l.kept.end(), rows);
    } else {
        for (uint32_t j = 0; j < m; ++j) rows[j] = rows[l.kept[j]];
    }
    for (uint32_t c : read_) {
        ColumnValues& col = l.columns[c];
        if (!col.ready) continue;
        if (handed_[c] || last_use_[c] > step) {
            WithValues(col, types_[c], [&](auto& values) { Keep(values, l.kept); });
        } else {
            col.ready = false;
        }
    }
    l.count = m;
}

// With no columns to hand up, a batch still carries its row count.
// Rows the byte cap leaves out start the next batch.
bool ScanSource::NextBatch(LocalState& local, Batch& out) const {
    auto& l = static_cast<Local&>(local);
    if (l.cursor >= l.count) return false;
    uint32_t n = std::min(kBatchSize, l.count - l.cursor);
    if (l.capped) n = RowsWithinCap(l, n);
    ShapeBatch(out, slot_types_);
    out.Reset(n);
    for (size_t k = 0; k < spec_.columns.size(); ++k) FillSlot(l.columns[spec_.columns[k]], l.cursor, n, out.slots[k]);
    l.cursor += n;
    return true;
}

// A batch ends once its string bytes pass the cap, keeping the row that
// passed it, so it always holds at least one row.
uint32_t ScanSource::RowsWithinCap(const Local& l, uint32_t n) const {
    uint64_t bytes = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t row = l.cursor + i;
        for (uint32_t c : string_slots_) {
            const ColumnValues& col = l.columns[c];
            bytes += col.coded ? col.entries[col.codes[row]].size : col.str[row].size;
        }
        if (bytes > byte_cap_) return i + 1;
    }
    return n;
}

// Relaxed loads: a snapshot, exact once the workers are done.
ScanCounters ScanSource::Counters() const {
    ScanCounters k;
    k.groups_skipped = groups_skipped_.load(std::memory_order_relaxed);
    k.rows_considered = rows_considered_.load(std::memory_order_relaxed);
    k.rows_passed = rows_passed_.load(std::memory_order_relaxed);
    k.rows_decoded.reserve(decoded_columns_);
    for (size_t c = 0; c < decoded_columns_; ++c) k.rows_decoded.push_back(rows_decoded_[c].load(std::memory_order_relaxed));
    return k;
}

// Counted at each morsel's start, before any conjunct narrows it.
// A LIMIT that stops early still counts every morsel started, unlike tuple mode.
std::optional<uint64_t> ScanSource::RowsRead() const {
    return rows_considered_.load(std::memory_order_relaxed);
}

// Read by Prepare, on the calling thread.
void BatchStringBytesForTest(size_t bytes) { test_batch_bytes = bytes; }

// A ScanSource behind the Source interface.
std::unique_ptr<Source> MakeScanSource(ScanSpec spec) {
    return std::make_unique<ScanSource>(std::move(spec));
}

// The same scan, its morsels limited to the index's matches.
std::unique_ptr<Source> MakeIndexScanSource(ScanSpec spec, std::string index_name, std::optional<Value> low, std::optional<Value> high) {
    IndexBounds bounds;
    bounds.index = std::move(index_name);
    bounds.low = std::move(low);
    bounds.high = std::move(high);
    return std::make_unique<ScanSource>(std::move(spec), std::move(bounds));
}

}  // namespace shilmandb::parallel
