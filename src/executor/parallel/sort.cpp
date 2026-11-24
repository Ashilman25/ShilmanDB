#include "executor/parallel/sort.hpp"
#include "common/exception.hpp"
#include "executor/parallel/batch.hpp"
#include "executor/parallel/column_buffer.hpp"
#include "executor/parallel/memory_budget.hpp"
#include "executor/parallel/temp_file.hpp"
#include "storage/columnar/string_arena.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>
#include <utility>

namespace shilmandb::parallel {

namespace {

// A run block's rows by the row estimate, one oversized row excepted.
// An 8 MiB floor holds 32, so a tight merge still reads many runs.
constexpr size_t kRunBlockBytes = size_t{256} << 10;

// The most runs one merge reads at once: 16 MiB of blocks.
constexpr size_t kMaxFanIn = 64;

// No byte cap on a merge's picks.
constexpr size_t kNoByteLimit = std::numeric_limits<size_t>::max();

// Set only by tests, on the calling thread.
// The most runs one merge reads; 0 means no cap.
size_t test_fan_in = 0;

// -1, 0 or 1 using < alone, as Value does.
template <typename T>
int Order(const T& a, const T& b) {
    return a < b ? -1 : (b < a ? 1 : 0);
}

// Doubles as Value compares them, except a NaN: after every number, NaNs by bits.
// So the sorts always get a strict weak order.
int OrderDoubles(double a, double b) {
    return std::isnan(a) || std::isnan(b) ? TotalOrder(a, b) : Order(a, b);
}

// Row a of x against row b of y, as Value compares them.
int CompareAt(const ColumnBuffer& x, uint32_t a, const ColumnBuffer& y, uint32_t b, const VType& t) {
    switch (t.kind) {
        case Kind::STRING: {
            const int c = View(x.str[a]).compare(View(y.str[b]));  // unsigned bytes, then shorter first
            return (c > 0) - (c < 0);
        }
        case Kind::DOUBLE:
            return OrderDoubles(x.f64[a], y.f64[b]);
        case Kind::DECIMAL:
            return t.wide ? Order(x.i128[a], y.i128[b]) : Order(x.i64[a], y.i64[b]);
        case Kind::INTEGER:
        case Kind::DATE:
            return Order(x.i64[a], y.i64[b]);
    }
    return 0;
}

// CompareAt for breaking ties: doubles in a total order, so ties never keep input order.
int TieCompareAt(const ColumnBuffer& x, uint32_t a, const ColumnBuffer& y, uint32_t b, const VType& t) {
    return t.kind == Kind::DOUBLE ? TotalOrder(x.f64[a], y.f64[b]) : CompareAt(x, a, y, b, t);
}

// Bytes one value takes in its array; a string counts its reference, not its bytes.
size_t WidthOf(ValueArray a) {
    switch (a) {
        case ValueArray::I64: return sizeof(int64_t);
        case ValueArray::I128: return sizeof(__int128);
        case ValueArray::F64: return sizeof(double);
        case ValueArray::STR: return sizeof(StringRef);
    }
    return 0;
}

// Empties a buffer, keeping its memory for the next block.
void ClearBuffer(ColumnBuffer& b) {
    b.i64.clear();
    b.i128.clear();
    b.f64.clear();
    b.str.clear();
}

// to = from's listed rows, in that order; strings still point where from's do.
void GatherRows(const ColumnBuffer& from, ValueArray a, const uint32_t* rows, size_t n, ColumnBuffer& to) {
    ClearBuffer(to);
    switch (a) {
        case ValueArray::I64:
            for (size_t i = 0; i < n; ++i) to.i64.push_back(from.i64[rows[i]]);
            break;
        case ValueArray::I128:
            for (size_t i = 0; i < n; ++i) to.i128.push_back(from.i128[rows[i]]);
            break;
        case ValueArray::F64:
            for (size_t i = 0; i < n; ++i) to.f64.push_back(from.f64[rows[i]]);
            break;
        case ValueArray::STR:
            for (size_t i = 0; i < n; ++i) to.str.push_back(from.str[rows[i]]);
            break;
    }
}

// Appends the listed rows of from to to, strings copied into arena.
void CopyRows(const ColumnBuffer& from, const VType& t, const std::vector<uint32_t>& rows, ColumnBuffer& to, StringArena& arena) {
    Vector v(t);
    Selection all;
    for (size_t at = 0; at < rows.size(); at += kBatchSize) {
        const auto n = static_cast<uint32_t>(std::min<size_t>(kBatchSize, rows.size() - at));
        Gather(from, rows.data() + at, n, v);
        all.SetAll(n);
        Append(v, all, to, arena);
    }
}

// A worker's key and kept columns, or a run block's (keys, then kept).
struct SortColumns {
    const ColumnBuffer* keys;
    const ColumnBuffer* kept;
    size_t key_count;

    // Column c, counting the keys first.
    [[nodiscard]] const ColumnBuffer& operator[](size_t c) const { return c < key_count ? keys[c] : kept[c - key_count]; }
};

// What the sink and source agree on: types, directions, the limit and the row order.
// Row bytes are estimated: fixed widths, strings' bytes, and 4 for the order index.
struct SortLayout {
    std::vector<VType> key_types;
    std::vector<bool> ascending;         // one per key
    std::vector<VType> kept_types;
    std::optional<size_t> limit;         // the spec's limit, at least 0
    std::vector<ValueArray> arrays;      // a run block's columns: keys, then kept
    std::vector<size_t> string_columns;  // which of those hold strings
    size_t key_bytes{0};                 // the keys' fixed bytes per row
    size_t row_bytes{0};                 // 8 per 64-bit value, 16 per wide decimal or string ref, plus 4

    // Types and estimates from the spec.
    explicit SortLayout(const SortSpec& spec) : ascending(spec.ascending), kept_types(spec.types) {
        assert(spec.keys.size() == spec.ascending.size() && "one direction per key");
        assert(spec.columns.size() == spec.types.size() && "one type per kept column");
        for (const auto& key : spec.keys) key_types.push_back(key->Type());
        if (spec.limit) limit = static_cast<size_t>(std::max<int64_t>(*spec.limit, 0));
        for (const VType& t : key_types) arrays.push_back(ArrayOf(t));
        for (const VType& t : kept_types) arrays.push_back(ArrayOf(t));
        for (size_t c = 0; c < arrays.size(); ++c) {
            row_bytes += WidthOf(arrays[c]);
            if (c + 1 == key_types.size()) key_bytes = row_bytes;
            if (arrays[c] == ValueArray::STR) string_columns.push_back(c);
        }
        row_bytes += sizeof(uint32_t);
    }

    // True when row a of x sorts first: keys as Value orders them, NaN last,
    // then kept columns ascending. The in-memory sort and the merge both use it.
    [[nodiscard]] bool Before(const SortColumns& x, uint32_t a, const SortColumns& y, uint32_t b) const {
        for (size_t i = 0; i < key_types.size(); ++i) {
            const int c = CompareAt(x.keys[i], a, y.keys[i], b, key_types[i]);
            if (c != 0) return ascending[i] ? c < 0 : c > 0;
        }
        for (size_t k = 0; k < kept_types.size(); ++k) {
            const int c = TieCompareAt(x.kept[k], a, y.kept[k], b, kept_types[k]);
            if (c != 0) return c < 0;
        }
        return false;
    }

    // One row's estimated bytes.
    [[nodiscard]] size_t RowBytes(const SortColumns& cols, uint32_t row) const {
        size_t bytes = row_bytes;
        for (const size_t c : string_columns) bytes += cols[c].str[row].size;
        return bytes;
    }

    // A worker's two stores as one set of columns.
    [[nodiscard]] SortColumns Columns(const ColumnStore& keys, const ColumnStore& kept) const {
        return {keys.columns.data(), kept.columns.data(), key_types.size()};
    }

    // A run block's columns: keys, then kept.
    [[nodiscard]] SortColumns Columns(const std::vector<ColumnBuffer>& block) const {
        return {block.data(), block.data() + key_types.size(), key_types.size()};
    }
};

// Row numbers in output order, cut at the limit; partial_sort for a top n.
std::vector<uint32_t> SortedOrder(const SortLayout& layout, const SortColumns& cols, size_t rows) {
    if (rows > std::numeric_limits<uint32_t>::max()) throw DatabaseException("the parallel executor can't sort more than 4294967295 rows");
    std::vector<uint32_t> order(rows);
    std::iota(order.begin(), order.end(), 0u);
    const auto before = [&](uint32_t a, uint32_t b) { return layout.Before(cols, a, cols, b); };
    if (layout.limit && *layout.limit < rows) {
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(*layout.limit), order.end(), before);
        order.resize(*layout.limit);
    } else {
        std::stable_sort(order.begin(), order.end(), before);
    }
    return order;
}

// One block of a run: where it starts and its rows.
struct RunBlock {
    uint64_t offset;
    uint32_t rows;
};

// A sorted run in a temp file: its blocks, each keys then kept columns.
struct Run {
    const TempFile* file{nullptr};
    std::vector<RunBlock> blocks;
};

// Writes rows in this order as one run, in blocks of about kRunBlockBytes at most.
Run WriteRunBlocks(const SortLayout& layout, const SortColumns& cols, const std::vector<uint32_t>& order, TempFile& file) {
    std::vector<ColumnBuffer> block(layout.arrays.size());
    std::vector<const ColumnBuffer*> parts;
    for (const ColumnBuffer& b : block) parts.push_back(&b);
    Run run;
    run.file = &file;
    for (size_t at = 0; at < order.size();) {
        size_t end = at;
        for (size_t bytes = 0; end < order.size(); ++end) {
            const size_t row = layout.RowBytes(cols, order[end]);
            if (end > at && bytes + row > kRunBlockBytes) break;
            bytes += row;
        }
        for (size_t c = 0; c < block.size(); ++c) GatherRows(cols[c], layout.arrays[c], order.data() + at, end - at, block[c]);
        run.blocks.push_back(RunBlock{file.WriteBlock(parts, layout.arrays, 0, end - at), static_cast<uint32_t>(end - at)});
        at = end;
    }
    return run;
}

// Where a worker's spill files go, and the counters they add to.
struct SpillPlace {
    const std::string* dir{nullptr};
    SpillCounters* counters{nullptr};

    // A new temp file; spilling without a directory is a bug.
    [[nodiscard]] std::unique_ptr<TempFile> Open() const {
        assert(dir != nullptr && "spilling needs a temp directory");
        if (dir == nullptr) throw DatabaseException("spilling needs a temp directory");
        return std::make_unique<TempFile>(*dir, counters);
    }
};

// One row picked by a merge: a cursor and a row of its loaded block.
struct Pick {
    uint32_t cursor;
    uint32_t row;
};

// Picks rows in order from sorted runs with a heap, holding one block per run.
// A pick points into a loaded block, valid until the next call to Next.
class RunMerge {
public:
    // Reads each run's first block. Blocks are counted in the caller's slots;
    // an oversized block's excess goes in reservation when the budget allows.
    RunMerge(const SortLayout& layout, const std::vector<const Run*>& runs, MemoryReservation& reservation)
        : layout_(layout), reservation_(reservation), cursors_(runs.size()) {
        for (size_t i = 0; i < runs.size(); ++i) {
            Cursor& c = cursors_[i];
            c.run = runs[i];
            c.block.resize(layout_.arrays.size());
            if (c.run->blocks.empty()) continue;
            Load(c);
            heap_.push_back(static_cast<uint32_t>(i));
        }
        std::make_heap(heap_.begin(), heap_.end(), [this](uint32_t a, uint32_t b) { return After(a, b); });
    }

    // Gives back what oversized blocks counted.
    ~RunMerge() {
        for (Cursor& c : cursors_) Drop(c);
    }

    RunMerge(const RunMerge&) = delete;
    RunMerge& operator=(const RunMerge&) = delete;

    // The next rows in order: at most max_rows, and past max_bytes only for the first.
    // Stops after a block's last row; that block goes on the next call.
    void Next(size_t max_rows, size_t max_bytes, std::vector<Pick>& picks) {
        Refill();
        picks.clear();
        const auto after = [this](uint32_t a, uint32_t b) { return After(a, b); };
        size_t bytes = 0;
        while (picks.size() < max_rows && !heap_.empty()) {
            Cursor& c = cursors_[heap_.front()];
            if (max_bytes != kNoByteLimit) {
                const size_t row = layout_.RowBytes(layout_.Columns(c.block), c.row);
                if (!picks.empty() && bytes + row > max_bytes) break;
                bytes += row;
            }
            std::pop_heap(heap_.begin(), heap_.end(), after);
            const uint32_t id = heap_.back();
            picks.push_back(Pick{id, c.row});
            if (++c.row < c.rows) {
                std::push_heap(heap_.begin(), heap_.end(), after);
                continue;
            }
            heap_.pop_back();
            exhausted_.push_back(id);
            break;
        }
    }

    // Column c (keys, then kept) of a cursor's loaded block.
    [[nodiscard]] const ColumnBuffer& Column(uint32_t cursor, size_t c) const { return cursors_[cursor].block[c]; }

private:
    // One run being read: its loaded block and the next row in it.
    struct Cursor {
        const Run* run{nullptr};
        size_t next_block{0};
        std::vector<ColumnBuffer> block;  // keys, then kept
        StringArena arena;                // the block's strings
        uint32_t row{0};
        uint32_t rows{0};
        size_t extra{0};  // bytes past its slot, counted for an oversized row
    };

    // Reads the cursor's next block over its last one.
    void Load(Cursor& c) {
        for (ColumnBuffer& b : c.block) ClearBuffer(b);
        c.arena.Clear();
        if (c.extra > 0) reservation_.Shrink(std::exchange(c.extra, 0));
        const RunBlock& block = c.run->blocks[c.next_block++];
        uint64_t offset = block.offset;
        if (!c.run->file->ReadBlock(&offset, c.block, layout_.arrays, c.arena)) {
            assert(false && "a run block offset past the file's end");
            throw DatabaseException("damaged temp file: a run block past the file's end");
        }
        c.row = 0;
        c.rows = block.rows;
        const size_t bytes = c.rows * layout_.row_bytes + c.arena.BytesUsed();
        if (bytes > kRunBlockBytes && reservation_.Grow(bytes - kRunBlockBytes)) c.extra = bytes - kRunBlockBytes;
    }

    // Frees a cursor's block and what it counted.
    void Drop(Cursor& c) {
        c.block = std::vector<ColumnBuffer>();
        c.arena = StringArena();
        if (c.extra > 0) reservation_.Shrink(std::exchange(c.extra, 0));
    }

    // Loads the next block of each cursor emptied on the last call, or drops it.
    void Refill() {
        for (const uint32_t id : exhausted_) {
            Cursor& c = cursors_[id];
            if (c.next_block == c.run->blocks.size()) {
                Drop(c);
                continue;
            }
            Load(c);
            heap_.push_back(id);
            std::push_heap(heap_.begin(), heap_.end(), [this](uint32_t a, uint32_t b) { return After(a, b); });
        }
        exhausted_.clear();
    }

    // The heap's order: true when cursor a's row comes after cursor b's.
    [[nodiscard]] bool After(uint32_t a, uint32_t b) const {
        const Cursor& x = cursors_[a];
        const Cursor& y = cursors_[b];
        return layout_.Before(layout_.Columns(y.block), y.row, layout_.Columns(x.block), x.row);
    }

    const SortLayout& layout_;
    MemoryReservation& reservation_;
    std::vector<Cursor> cursors_;
    std::vector<uint32_t> heap_;       // cursors with rows left, earliest row on top
    std::vector<uint32_t> exhausted_;  // cursors whose block ran out on the last call
};

// out's rows 0..n-1 = column c of each pick; strings stay in the merge's blocks.
void GatherPicks(const RunMerge& merge, size_t c, const std::vector<Pick>& picks, Vector& out) {
    const size_t n = picks.size();
    switch (ArrayOf(out.Type())) {
        case ValueArray::STR: {
            out.SetDictionary(nullptr);
            StringRef* o = out.Str();
            for (size_t i = 0; i < n; ++i) o[i] = merge.Column(picks[i].cursor, c).str[picks[i].row];
            break;
        }
        case ValueArray::F64: {
            double* o = out.F64();
            for (size_t i = 0; i < n; ++i) o[i] = merge.Column(picks[i].cursor, c).f64[picks[i].row];
            break;
        }
        case ValueArray::I128: {
            __int128* o = out.I128();
            for (size_t i = 0; i < n; ++i) o[i] = merge.Column(picks[i].cursor, c).i128[picks[i].row];
            break;
        }
        case ValueArray::I64: {
            int64_t* o = out.I64();
            for (size_t i = 0; i < n; ++i) o[i] = merge.Column(picks[i].cursor, c).i64[picks[i].row];
            break;
        }
    }
}

// to = column c of each pick; strings stay in the merge's blocks.
void GatherPicks(const RunMerge& merge, size_t c, ValueArray a, const std::vector<Pick>& picks, ColumnBuffer& to) {
    ClearBuffer(to);
    const auto copy = [&](auto member) {
        auto& out = to.*member;
        for (const Pick& p : picks) out.push_back((merge.Column(p.cursor, c).*member)[p.row]);
    };
    switch (a) {
        case ValueArray::I64: copy(&ColumnBuffer::i64); break;
        case ValueArray::I128: copy(&ColumnBuffer::i128); break;
        case ValueArray::F64: copy(&ColumnBuffer::f64); break;
        case ValueArray::STR: copy(&ColumnBuffer::str); break;
    }
}

// The sorted rows: written by the sink's Finalize, read by the source.
// In memory when nothing spilled; otherwise every row is in a run on disk.
struct SortedRows {
    ColumnStore kept;                             // the kept columns, with every worker's strings
    std::vector<uint32_t> order;                  // rows in output order, already cut at the limit
    std::vector<MemoryReservation> reservations;  // the workers' counts for these rows
    std::vector<std::unique_ptr<TempFile>> files;
    std::vector<Run> runs;
};

// One worker's rows: each key's values and each kept column.
struct SortLocal : LocalState {
    ColumnStore keys;
    ColumnStore kept;
    std::vector<Vector> key_values;  // one batch's keys, before they're appended
    EvalScratch scratch;
    MemoryReservation reservation;   // the rows' estimated bytes, after every Consume
    SpillPlace place;
    bool spill_everything{false};
    std::unique_ptr<TempFile> file;  // made on the first spill
    std::vector<Run> runs;           // written so far, all in file
};

// Keeps every row it's given; with a limit, each worker keeps only its top n.
// Rows past the worker's budget go to disk as sorted runs.
class SortSink final : public Sink {
public:
    SortSink(SortSpec spec, std::shared_ptr<const SortLayout> layout, SharedOutput<SortedRows> out)
        : spec_(std::move(spec)), layout_(std::move(layout)), out_(std::move(out)) {}

    // A local without a budget, so it never spills.
    [[nodiscard]] std::unique_ptr<LocalState> MakeLocal() const override { return NewLocal(); }

    // A local counted against this worker's budget, spilling to the query's temp directory.
    [[nodiscard]] std::unique_ptr<LocalState> MakeWorkerLocal(const QueryContext& ctx, unsigned worker) const override {
        auto local = NewLocal();
        local->reservation = MemoryReservation(ctx.budget, worker);
        local->place = SpillPlace{ctx.temp_dir, ctx.spill};
        local->spill_everything = ctx.spill_everything;
        return local;
    }

    // Evaluates the keys, copies keys and kept columns for the active rows, and counts them.
    // Over budget: a top n trims first, then what won't fit becomes a run.
    void Consume(Batch& in, LocalState& local) const override {
        auto& l = static_cast<SortLocal&>(local);
        for (size_t i = 0; i < spec_.keys.size(); ++i) spec_.keys[i]->Evaluate(in, in.sel, l.key_values[i], l.scratch);
        if (layout_->limit == size_t{0}) return;  // LIMIT 0 keeps nothing
        for (size_t i = 0; i < spec_.keys.size(); ++i) Append(l.key_values[i], in.sel, l.keys.columns[i], l.keys.arena);
        for (size_t k = 0; k < spec_.columns.size(); ++k) {
            const Vector& v = in.slots[spec_.columns[k]];
            assert(v.Type() == spec_.types[k] && "kept column holds another type");
            Append(v, in.sel, l.kept.columns[k], l.kept.arena);
        }
        l.keys.rows += in.Active();
        l.kept.rows += in.Active();

        const std::optional<size_t>& limit = layout_->limit;
        if (l.spill_everything) {
            WriteRun(l);
            return;
        }
        if (Recount(l)) {
            // Trimming past n + max(n, a batch) rows keeps the cost per row low.
            if (limit && l.kept.rows >= *limit + std::max<size_t>(*limit, kBatchSize)) {
                Trim(l);
                (void)Recount(l);  // only shrinks
            }
            return;
        }
        if (limit && l.kept.rows > *limit) {
            Trim(l);
            if (Recount(l)) return;
        }
        WriteRun(l);
    }

    // Keeps the worker's buffers, arena and all, until Finalize.
    void Combine(std::unique_ptr<LocalState> local) override { parts_.push_back(std::move(local)); }

    // The rows outlive this pipeline, so each part leaves its worker's floor,
    // spilling what must stay there. With no runs anywhere, the in-memory sort runs.
    void Finalize(QueryContext& ctx) override {
        ForEachTask(ctx, parts_.size(), [this](size_t i) {
            SortLocal& p = Part(i);
            assert(p.reservation.Bytes() == HeldBytes(p) && "a part's count drifted from its rows");
            for (size_t left = p.reservation.LeaveFloor(); left > 0; left = p.reservation.LeaveFloor()) WriteRun(p);
        });
        const bool spilled = std::any_of(parts_.begin(), parts_.end(), [](const std::unique_ptr<LocalState>& part) {
            return !static_cast<const SortLocal&>(*part).runs.empty();
        });
        if (spilled) {
            FinishRuns(ctx);
        } else {
            SortInMemory();
        }
        parts_.clear();
    }

private:
    // Empty buffers, and one vector per key for a batch's values.
    [[nodiscard]] std::unique_ptr<SortLocal> NewLocal() const {
        auto local = std::make_unique<SortLocal>();
        local->keys = ColumnStore(layout_->key_types.size());
        local->kept = ColumnStore(layout_->kept_types.size());
        for (const VType& t : layout_->key_types) local->key_values.emplace_back(t);
        return local;
    }

    // Combined part i.
    [[nodiscard]] SortLocal& Part(size_t i) { return static_cast<SortLocal&>(*parts_[i]); }

    // The worker's rows by the row estimate: fixed bytes per row plus every string's bytes.
    [[nodiscard]] size_t HeldBytes(const SortLocal& l) const {
        return l.kept.rows * layout_->row_bytes + l.keys.arena.BytesUsed() + l.kept.arena.BytesUsed();
    }

    // Brings the reservation to what the worker holds; false when the budget refuses.
    [[nodiscard]] bool Recount(SortLocal& l) const {
        const size_t held = HeldBytes(l);
        const size_t counted = l.reservation.Bytes();
        if (held < counted) l.reservation.Shrink(counted - held);
        return held <= counted || l.reservation.Grow(held - counted);
    }

    // Sorts what the worker holds and writes it as one run, cut at the limit.
    // Its memory is uncounted before the writing starts.
    void WriteRun(SortLocal& l) const {
        const SortColumns cols = layout_->Columns(l.keys, l.kept);
        const std::vector<uint32_t> order = SortedOrder(*layout_, cols, l.kept.rows);
        l.reservation.Shrink(l.reservation.Bytes());
        if (!order.empty()) {
            if (!l.file) l.file = l.place.Open();
            l.runs.push_back(WriteRunBlocks(*layout_, cols, order, *l.file));
        }
        l.keys = ColumnStore(layout_->key_types.size());
        l.kept = ColumnStore(layout_->kept_types.size());
    }

    // Cuts a worker's rows to its top n, copying their strings into fresh arenas.
    void Trim(SortLocal& l) const {
        const std::vector<uint32_t> keep = SortedOrder(*layout_, layout_->Columns(l.keys, l.kept), l.kept.rows);
        ColumnStore keys(l.keys.columns.size());
        ColumnStore kept(l.kept.columns.size());
        for (size_t i = 0; i < keys.columns.size(); ++i) CopyRows(l.keys.columns[i], layout_->key_types[i], keep, keys.columns[i], keys.arena);
        for (size_t k = 0; k < kept.columns.size(); ++k) CopyRows(l.kept.columns[k], layout_->kept_types[k], keep, kept.columns[k], kept.arena);
        keys.rows = keep.size();
        kept.rows = keep.size();
        l.keys = std::move(keys);
        l.kept = std::move(kept);
    }

    // Every part's remainder becomes a run too, so the source merges runs alone.
    void FinishRuns(QueryContext& ctx) {
        ForEachTask(ctx, parts_.size(), [this](size_t i) {
            SortLocal& p = Part(i);
            WriteRun(p);
            if (p.file) p.file->FinishWriting();
        });
        SortedRows& out = out_.Mutable();
        for (size_t i = 0; i < parts_.size(); ++i) {
            SortLocal& p = Part(i);
            out.runs.insert(out.runs.end(), std::make_move_iterator(p.runs.begin()), std::make_move_iterator(p.runs.end()));
            if (p.file) out.files.push_back(std::move(p.file));
        }
    }

    // Joins every worker's rows and sorts them; the reservations move with the rows.
    // The key values and their strings go once the order is known.
    void SortInMemory() {
        SortedRows& out = out_.Mutable();
        out.kept = ColumnStore(layout_->kept_types.size());
        std::vector<size_t> key_bytes;
        {
            ColumnStore keys(layout_->key_types.size());
            for (size_t i = 0; i < parts_.size(); ++i) {
                SortLocal& p = Part(i);
                key_bytes.push_back(p.keys.rows * layout_->key_bytes + p.keys.arena.BytesUsed());
                keys.Absorb(std::move(p.keys));
                out.kept.Absorb(std::move(p.kept));
                out.reservations.push_back(std::move(p.reservation));
            }
            out.order = SortedOrder(*layout_, layout_->Columns(keys, out.kept), out.kept.rows);
        }
        for (size_t i = 0; i < key_bytes.size(); ++i) out.reservations[i].Shrink(key_bytes[i]);
    }

    SortSpec spec_;
    std::shared_ptr<const SortLayout> layout_;
    SharedOutput<SortedRows> out_;
    std::vector<std::unique_ptr<LocalState>> parts_;  // one per worker, after Combine
};

// Runs a merge pass wrote, in a file of their own.
struct MergedRuns {
    std::unique_ptr<TempFile> file;
    std::vector<Run> runs;
};

// Where the source is: a place in the in-memory order, or a merge of runs.
struct SortSourceLocal : LocalState {
    size_t next{0};                      // in memory: the next place in the order
    MemoryReservation reservation;       // the merge's blocks, kRunBlockBytes per slot
    size_t slots{0};                     // blocks the reservation holds room for
    SpillPlace place;
    std::unique_ptr<MergedRuns> merged;  // the newest merge pass, if runs were merged down
    std::unique_ptr<RunMerge> merge;     // the final merge; declared after what it reads
    std::vector<Pick> picks;
    size_t emitted{0};
};

// Hands out the sorted rows in order, kBatchSize at a time, merging any runs.
class SortSource final : public Source {
public:
    SortSource(std::shared_ptr<const SortLayout> layout, std::shared_ptr<const SortedRows> rows)
        : layout_(std::move(layout)), rows_(std::move(rows)) {}

    // The sink's Finalize did the sorting; the fan-in's test cap is read here.
    void Prepare(QueryContext&) override { fan_in_ = test_fan_in != 0 ? std::clamp<size_t>(test_fan_in, 2, kMaxFanIn) : kMaxFanIn; }

    // One morsel, so one worker emits every row in order.
    [[nodiscard]] size_t MorselCount() const override { return 1; }

    // A local without a budget: a merge's blocks go uncounted.
    [[nodiscard]] std::unique_ptr<LocalState> MakeLocal() const override { return std::make_unique<SortSourceLocal>(); }

    // A local whose merge blocks count against this worker's budget.
    [[nodiscard]] std::unique_ptr<LocalState> MakeWorkerLocal(const QueryContext& ctx, unsigned worker) const override {
        auto local = std::make_unique<SortSourceLocal>();
        local->reservation = MemoryReservation(ctx.budget, worker);
        local->place = SpillPlace{ctx.temp_dir, ctx.spill};
        return local;
    }

    // Starts from the first sorted row; with runs, sets up their merge.
    void StartMorsel(size_t, LocalState& local) const override {
        auto& l = static_cast<SortSourceLocal&>(local);
        l.next = 0;
        l.emitted = 0;
        if (!rows_->runs.empty()) StartMerge(l);
    }

    // The next rows in order, as plain vectors.
    bool NextBatch(LocalState& local, Batch& out) const override {
        auto& l = static_cast<SortSourceLocal&>(local);
        return rows_->runs.empty() ? NextInMemory(l, out) : NextMerged(l, out);
    }

private:
    // How many rows to hand out in all.
    [[nodiscard]] size_t RowsWanted() const { return layout_->limit.value_or(std::numeric_limits<size_t>::max()); }

    // Strings point into the sink's arenas.
    bool NextInMemory(SortSourceLocal& l, Batch& out) const {
        const std::vector<uint32_t>& order = rows_->order;
        if (l.next >= order.size()) return false;
        const auto n = static_cast<uint32_t>(std::min<size_t>(kBatchSize, order.size() - l.next));
        ShapeBatch(out, layout_->kept_types);
        out.Reset(n);
        for (size_t k = 0; k < layout_->kept_types.size(); ++k) Gather(rows_->kept.columns[k], order.data() + l.next, n, out.slots[k]);
        l.next += n;
        return true;
    }

    // Strings point into the merge's blocks, which last until the next call.
    // When done, the blocks, merged files and slots go back.
    bool NextMerged(SortSourceLocal& l, Batch& out) const {
        if (!l.merge) return false;
        const size_t left = RowsWanted() - l.emitted;
        if (left > 0) l.merge->Next(std::min<size_t>(kBatchSize, left), kNoByteLimit, l.picks);
        if (left == 0 || l.picks.empty()) {
            l.merge.reset();
            l.merged.reset();
            l.reservation = MemoryReservation();
            l.slots = 0;
            return false;
        }
        const auto n = static_cast<uint32_t>(l.picks.size());
        ShapeBatch(out, layout_->kept_types);
        out.Reset(n);
        for (size_t k = 0; k < layout_->kept_types.size(); ++k) GatherPicks(*l.merge, layout_->key_types.size() + k, l.picks, out.slots[k]);
        l.emitted += n;
        return true;
    }

    // Reserves a block slot per run, up to the fan-in, merging down any extra runs.
    // With under 2 slots it still reads 2 runs at once, uncounted.
    void StartMerge(SortSourceLocal& l) const {
        std::vector<const Run*> runs;
        for (const Run& run : rows_->runs) runs.push_back(&run);
        const size_t want = std::min(fan_in_, runs.size());
        while (l.slots < want && l.reservation.Grow(kRunBlockBytes)) ++l.slots;
        while (runs.size() > std::max<size_t>(2, l.slots)) runs = MergePass(l, runs);
        l.merge = std::make_unique<RunMerge>(*layout_, runs, l.reservation);
    }

    // Merges each group of runs into one new run, all in a new file.
    // One slot holds the output block. The last pass's file goes, all its runs read.
    std::vector<const Run*> MergePass(SortSourceLocal& l, const std::vector<const Run*>& runs) const {
        const size_t group = l.slots >= 3 ? l.slots - 1 : 2;
        const size_t groups = (runs.size() + group - 1) / group;
        auto pass = std::make_unique<MergedRuns>();
        pass->file = l.place.Open();
        std::vector<ColumnBuffer> block(layout_->arrays.size());
        std::vector<const ColumnBuffer*> parts;
        for (const ColumnBuffer& b : block) parts.push_back(&b);
        for (size_t g = 0; g < groups; ++g) {
            // Group sizes differ by at most one run.
            const auto first = runs.begin() + static_cast<std::ptrdiff_t>(runs.size() * g / groups);
            const auto last = runs.begin() + static_cast<std::ptrdiff_t>(runs.size() * (g + 1) / groups);
            RunMerge merge(*layout_, std::vector<const Run*>(first, last), l.reservation);
            Run run;
            run.file = pass->file.get();
            // A top n's run needs only its first n rows.
            for (size_t left = RowsWanted(); left > 0; left -= l.picks.size()) {
                merge.Next(left, kRunBlockBytes, l.picks);
                if (l.picks.empty()) break;
                for (size_t c = 0; c < block.size(); ++c) GatherPicks(merge, c, layout_->arrays[c], l.picks, block[c]);
                const uint64_t offset = pass->file->WriteBlock(parts, layout_->arrays, 0, l.picks.size());
                run.blocks.push_back(RunBlock{offset, static_cast<uint32_t>(l.picks.size())});
            }
            pass->runs.push_back(std::move(run));
        }
        pass->file->FinishWriting();
        l.merged = std::move(pass);
        std::vector<const Run*> next;
        for (const Run& run : l.merged->runs) next.push_back(&run);
        return next;
    }

    std::shared_ptr<const SortLayout> layout_;
    std::shared_ptr<const SortedRows> rows_;
    size_t fan_in_{kMaxFanIn};  // set by Prepare
};

}  // namespace

// The sink and source share the sorted rows, so neither outlives what it reads.
std::pair<std::unique_ptr<Sink>, std::unique_ptr<Source>> MakeSort(SortSpec spec) {
    auto layout = std::make_shared<const SortLayout>(spec);
    SharedOutput<SortedRows> rows;
    auto source = std::make_unique<SortSource>(layout, rows.Reader());
    auto sink = std::make_unique<SortSink>(std::move(spec), std::move(layout), std::move(rows));
    return {std::move(sink), std::move(source)};
}

// Read once per sort, by its source's Prepare.
void SetSortFanInForTest(size_t most) {
    test_fan_in = most;
}

}  // namespace shilmandb::parallel
