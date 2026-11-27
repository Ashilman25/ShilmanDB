#include "executor/parallel/hash_join.hpp"
#include "common/exception.hpp"
#include "executor/parallel/batch.hpp"
#include "executor/parallel/column_buffer.hpp"
#include "executor/parallel/decimal.hpp"
#include "executor/parallel/key_hash.hpp"
#include "executor/parallel/memory_budget.hpp"
#include "executor/parallel/spill_partitions.hpp"
#include "executor/parallel/temp_file.hpp"
#include "storage/columnar/string_arena.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <utility>

namespace shilmandb::parallel {

namespace {

using detail::DoubleBits;
using detail::FoldKey;
using detail::HashBytes;
using detail::WideBits;

// Build rows one table may hold; ids are 32 bits and 0 ends a chain.
constexpr size_t kMaxBuildRows = std::numeric_limits<uint32_t>::max() - 1;

// The most partitions a build splits into.
constexpr size_t kMaxJoinPartitions = 1024;

//  Keys

// The form both sides' keys meet in.
enum class KeyForm : uint8_t { I64, I128, F64, STR };

// How one side's key values become the shared form.
struct KeyRule {
    KeyForm form{KeyForm::I64};
    VType from;         // this side's key type
    uint8_t scale{0};   // I128 only: the scale both sides take
};

// A kind's name, for error messages.
const char* KindName(Kind kind) {
    switch (kind) {
        case Kind::INTEGER: return "INTEGER";
        case Kind::DATE: return "DATE";
        case Kind::DECIMAL: return "DECIMAL";
        case Kind::DOUBLE: return "DOUBLE";
        case Kind::STRING: return "STRING";
    }
    return "UNKNOWN";
}

// Integers are decimals of scale 0.
bool IsExactNumber(const VType& t) {
    return t.kind == Kind::INTEGER || t.kind == Kind::DECIMAL;
}

// A decimal's scale, 0 for an integer.
uint8_t ScaleOf(const VType& t) {
    return t.kind == Kind::DECIMAL ? t.scale : 0;
}

// Where two key types meet, by kind, scale and width; never by result type.
std::pair<KeyRule, KeyRule> KeyRules(const VType& left, const VType& right) {
    KeyRule l{KeyForm::I64, left, 0};
    KeyRule r{KeyForm::I64, right, 0};
    const auto set = [&](KeyForm form, uint8_t scale) {
        l.form = r.form = form;
        l.scale = r.scale = scale;
    };
    const bool numbers = (IsExactNumber(left) || left.kind == Kind::DOUBLE) && (IsExactNumber(right) || right.kind == Kind::DOUBLE);
    if (left.kind == Kind::STRING && right.kind == Kind::STRING) {
        set(KeyForm::STR, 0);
    } else if (left.kind == Kind::DATE && right.kind == Kind::DATE) {
        set(KeyForm::I64, 0);
    } else if (numbers && (left.kind == Kind::DOUBLE || right.kind == Kind::DOUBLE)) {
        set(KeyForm::F64, 0);
    } else if (numbers) {
        // Same scale and both 64-bit stays as is; anything else meets exactly in 128 bits.
        const bool as_is = ScaleOf(left) == ScaleOf(right) && !left.wide && !right.wide;
        set(as_is ? KeyForm::I64 : KeyForm::I128, std::max(ScaleOf(left), ScaleOf(right)));
    } else {
        throw DatabaseException(std::string("the parallel executor can't join keys of types ") + KindName(left.kind) + " and " + KindName(right.kind));
    }
    return {l, r};
}

// One batch's keys in the shared form, with their hashes, by active position.
struct BatchKeys {
    std::vector<int64_t> i64;
    std::vector<__int128> i128;
    std::vector<double> f64;
    std::vector<StringRef> str;
    std::vector<uint64_t> hashes;
    std::vector<uint8_t> valid;  // 0: the key equals nothing (a NaN, or past 38 digits once rescaled)
    bool all_valid{true};
    std::vector<uint64_t> entry_hashes;  // a string key's dictionary entries, hashed

    // Room for a batch in the form's array.
    explicit BatchKeys(KeyForm form) : hashes(kBatchSize), valid(kBatchSize, 1) {
        switch (form) {
            case KeyForm::I64: i64.resize(kBatchSize); break;
            case KeyForm::I128: i128.resize(kBatchSize); break;
            case KeyForm::F64: f64.resize(kBatchSize); break;
            case KeyForm::STR: str.resize(kBatchSize); break;
        }
    }

    // True when active row k's key can match.
    [[nodiscard]] bool Valid(uint32_t k) const { return all_valid || valid[k] != 0; }
};

// Row i of v as a double, the way the columnar scan shows a decimal.
double AsDouble(const Vector& v, uint32_t i) {
    switch (v.Type().kind) {
        case Kind::DOUBLE: return v.F64()[i];
        case Kind::DECIMAL: return v.Type().wide ? DecimalToDouble(v.I128()[i], v.Type().scale) : DecimalToDouble(v.I64()[i], v.Type().scale);
        default: return static_cast<double>(v.I64()[i]);
    }
}

// Moves v's active rows into the rule's form, hashed; position k is active row k.
// A small dictionary's entries are hashed once each.
void Normalize(const Vector& v, const Selection& sel, uint32_t n, const KeyRule& rule, BatchKeys& out) {
    out.all_valid = true;
    switch (rule.form) {
        case KeyForm::I64: {
            const int64_t* a = v.I64();
            for (uint32_t k = 0; k < n; ++k) {
                out.i64[k] = a[sel[k]];
                out.hashes[k] = FoldKey(0, static_cast<uint64_t>(out.i64[k]));
            }
            break;
        }
        case KeyForm::I128: {
            const __int128 factor = Pow10(static_cast<uint8_t>(rule.scale - ScaleOf(rule.from)));
            const bool wide = ArrayOf(rule.from) == ValueArray::I128;
            for (uint32_t k = 0; k < n; ++k) {
                const __int128 x = wide ? v.I128()[sel[k]] : v.I64()[sel[k]];
                __int128 y = 0;
                const bool fits = !detail::MulOverflow128(x, factor, &y) && detail::FitsDecimal(y);
                out.i128[k] = fits ? y : 0;
                out.valid[k] = fits ? 1 : 0;
                out.all_valid = out.all_valid && fits;
                out.hashes[k] = FoldKey(0, WideBits(out.i128[k]));
            }
            break;
        }
        case KeyForm::F64:
            for (uint32_t k = 0; k < n; ++k) {
                const double d = AsDouble(v, sel[k]);
                const bool number = !std::isnan(d);
                out.f64[k] = d;
                out.valid[k] = number ? 1 : 0;
                out.all_valid = out.all_valid && number;
                out.hashes[k] = FoldKey(0, DoubleBits(d));
            }
            break;
        case KeyForm::STR:
            if (detail::HashEntries(v, n, out.entry_hashes)) {
                for (uint64_t& h : out.entry_hashes) h = FoldKey(0, h);
                const uint32_t* codes = v.Codes();
                for (uint32_t k = 0; k < n; ++k) {
                    out.str[k] = v.StrAt(sel[k]);
                    out.hashes[k] = out.entry_hashes[codes[sel[k]]];
                }
                break;
            }
            for (uint32_t k = 0; k < n; ++k) {
                out.str[k] = v.StrAt(sel[k]);
                out.hashes[k] = FoldKey(0, HashBytes(View(out.str[k])));
            }
            break;
    }
}

// Appends the valid keys among positions 0..n-1; strings are copied into arena.
void AppendKeys(const BatchKeys& keys, uint32_t n, KeyForm form, ColumnBuffer& buf, StringArena& arena) {
    for (uint32_t k = 0; k < n; ++k) {
        if (!keys.Valid(k)) continue;
        switch (form) {
            case KeyForm::I64: buf.i64.push_back(keys.i64[k]); break;
            case KeyForm::I128: buf.i128.push_back(keys.i128[k]); break;
            case KeyForm::F64: buf.f64.push_back(keys.f64[k]); break;
            case KeyForm::STR: buf.str.push_back(arena.Copy(View(keys.str[k]))); break;
        }
    }
}

// Appends the keys at positions[0..n) in the shared form; strings are copied into arena.
void AppendKeysAt(const BatchKeys& keys, KeyForm form, const uint32_t* positions, uint32_t n, ColumnBuffer& buf, StringArena& arena) {
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t k = positions[i];
        switch (form) {
            case KeyForm::I64: buf.i64.push_back(keys.i64[k]); break;
            case KeyForm::I128: buf.i128.push_back(keys.i128[k]); break;
            case KeyForm::F64: buf.f64.push_back(keys.f64[k]); break;
            case KeyForm::STR: buf.str.push_back(arena.Copy(View(keys.str[k]))); break;
        }
    }
}

// The array a key form's values sit in.
ValueArray ArrayOfForm(KeyForm form) {
    switch (form) {
        case KeyForm::I64: return ValueArray::I64;
        case KeyForm::I128: return ValueArray::I128;
        case KeyForm::F64: return ValueArray::F64;
        case KeyForm::STR: return ValueArray::STR;
    }
    return ValueArray::I64;
}

//  Kept rows

// Built, spilled and loaded rows share one layout:
// the key, its hash, then the kept slots.
constexpr size_t kKeyColumn = 0;
constexpr size_t kHashColumn = 1;
constexpr size_t kFirstKept = 2;

// Per row of the built table, its index: row id
// 4, hash 8, chain 4, bucket at most 8.
constexpr size_t kTableIndexBytes = 24;

// Per row loaded from disk, its chain 4 and bucket at most 8.
constexpr size_t kChainBytes = 12;

// A string in memory is a 16-byte reference; on disk, a 4-byte length.
constexpr size_t kStringRefExtra = sizeof(StringRef) - sizeof(uint32_t);

// Every partition marked spilled.
constexpr uint64_t kAllPartitions = ~uint64_t{0};

// Set only by tests, on the calling thread. A spilled load past this many rows
// splits, or ends its chunk; 0 means no cap.
size_t test_load_rows = 0;

// The arrays of one side's rows: the key in
// the shared form, its hash, then the kept slots.
std::vector<ValueArray> RowArrays(KeyForm form, const std::vector<VType>& kept) {
    std::vector<ValueArray> arrays = {ArrayOfForm(form), ValueArray::I64};
    for (const VType& t : kept) arrays.push_back(ArrayOf(t));
    return arrays;
}

// Bytes a row takes in its arrays, strings counted by reference.
size_t FixedBytes(const std::vector<ValueArray>& arrays) {
    size_t bytes = 0;
    for (const ValueArray a : arrays) bytes += ArrayWidth(a);
    return bytes;
}

// How many of the arrays hold strings.
size_t StringColumns(const std::vector<ValueArray>& arrays) {
    return static_cast<size_t>(std::count(arrays.begin(), arrays.end(), ValueArray::STR));
}

// A store's columns as the pointers a writer takes.
std::vector<const ColumnBuffer*> ColumnsOf(const std::vector<ColumnBuffer>& columns) {
    std::vector<const ColumnBuffer*> out;
    for (const ColumnBuffer& c : columns) out.push_back(&c);
    return out;
}

// Empties a store, keeping its memory for the next batch.
void ClearStore(ColumnStore& store) {
    for (ColumnBuffer& c : store.columns) ClearBuffer(c);
    store.arena.Clear();
    store.rows = 0;
}

// Writes every row of columns (`rows` of them) to
// one partition, a batch of ids at a time.
void WriteAll(PartitionedWriter& writer, size_t partition, const std::vector<ColumnBuffer>& columns, size_t rows, std::vector<uint32_t>& ids) {
    const std::vector<const ColumnBuffer*> cols = ColumnsOf(columns);
    ids.resize(std::min<size_t>(kBatchSize, rows));
    for (size_t at = 0; at < rows; at += kBatchSize) {
        const size_t n = std::min<size_t>(kBatchSize, rows - at);
        for (size_t i = 0; i < n; ++i) ids[i] = static_cast<uint32_t>(at + i);
        writer.Add(partition, cols, ids.data(), n);
    }
}

//  The built table

// A hash's partition: its top bits, since buckets use the low ones.
inline size_t PartitionOf(uint64_t h, int bits) {
    return bits == 0 ? 0 : static_cast<size_t>(h >> (64 - bits));
}

// Chains by bucket over n rows: row j is entry j+1; 0 ends a chain.
struct ChainIndex {
    std::vector<uint32_t> buckets;  // a chain's first entry, 0 when empty; a power of two
    std::vector<uint32_t> next;     // per row: the next entry in its chain
    uint64_t mask{0};
};

// Links rows 0..n-1 into one chain per bucket, each in row order.
template <typename Hash>
void LinkChains(const Hash* hashes, size_t n, ChainIndex& c) {
    size_t buckets = 1;
    while (buckets < n) buckets *= 2;
    c.mask = buckets - 1;
    c.buckets.assign(buckets, 0);
    c.next.assign(n, 0);
    for (size_t j = n; j-- > 0;) {
        uint32_t& head = c.buckets[static_cast<uint64_t>(hashes[j]) & c.mask];
        c.next[j] = head;
        head = static_cast<uint32_t>(j + 1);
    }
}

// One hash partition: its rows and hashes, and their chains.
// It holds everything its build writes, so partitions can build at the same time.
struct Partition {
    std::vector<uint32_t> rows;    // build row ids, in arrival order
    std::vector<uint64_t> hashes;  // one per row here
    ChainIndex chains;             // over the rows here
};

// The finished build side: written by the sink's Finalize, read by the operator.
// Its reservations count what it keeps in memory, and go back when it's destroyed.
struct BuildTable {
    ColumnStore store;  // the key in the shared form, its hash (freed once built), the payload
    std::vector<Partition> partitions;
    int bits{0};        // a row's partition is its hash's top `bits` bits
    bool ready{false};
    std::vector<SpillFile> files;   // each spilling worker's rows of the spilled partitions
    std::vector<size_t> spilled;    // level-0 spill partitions on disk, ascending
    uint64_t spilled_mask{0};       // bit q set when partition q is on disk
    size_t spilled_rows{0};
    size_t load_rows{0};            // tests: the most rows a spilled load holds; 0 for no cap
    std::vector<MemoryReservation> reservations;  // the workers' counts of store and partitions
};

// Sends each build row to its partition, keeping arrival order within each.
void Scatter(const std::vector<int64_t>& hashes, BuildTable& t) {
    std::vector<size_t> counts(t.partitions.size(), 0);
    for (const int64_t h : hashes) ++counts[PartitionOf(static_cast<uint64_t>(h), t.bits)];
    for (size_t p = 0; p < counts.size(); ++p) {
        t.partitions[p].rows.reserve(counts[p]);
        t.partitions[p].hashes.reserve(counts[p]);
    }
    for (size_t g = 0; g < hashes.size(); ++g) {
        const auto h = static_cast<uint64_t>(hashes[g]);
        Partition& p = t.partitions[PartitionOf(h, t.bits)];
        p.rows.push_back(static_cast<uint32_t>(g));
        p.hashes.push_back(h);
    }
}

//  Sink

// One worker's rows of one spill partition, so it can be written and freed alone.
struct HeldRows {
    ColumnStore store;
    size_t bytes{0};  // the rows' estimated bytes, the arena's blocks included
};

// One worker's build rows: in arrival order while nothing spills, then by spill partition.
struct BuildLocal : LocalState {
    ColumnStore store;                // arrival order: key, hash, payload; emptied once partitioned
    std::vector<HeldRows> parts;      // one per spill partition, once spilling starts
    ColumnStore marked_rows;          // a batch's rows of marked partitions, on their way to disk
    Vector key_values;                // one batch's keys, when the key isn't a bare column
    BatchKeys keys;
    Selection keep;                   // the batch's rows whose key can match
    std::vector<uint8_t> part_of;     // per active row: its spill partition
    std::vector<uint32_t> order;      // the batch's valid rows, grouped by partition
    Selection pick;                   // one partition's rows of the batch
    std::vector<uint32_t> ids;        // row ids handed to the writer
    EvalScratch scratch;
    MemoryReservation reservation;    // the held rows, and the writer's buffers while it writes
    size_t held{0};                   // the rows' estimated bytes
    size_t counted{0};                // the part of the reservation that counts them
    bool pooled{false};               // Finalize moved that count into the table's pooled reservation
    const std::string* temp_dir{nullptr};
    SpillCounters* counters{nullptr};
    bool spill_everything{false};
    std::unique_ptr<PartitionedWriter> writer;  // made on the first spill; after the reservation it counts in

    BuildLocal(VType key_type, KeyForm form, size_t columns) : store(columns), marked_rows(columns), key_values(key_type), keys(form), part_of(kBatchSize), order(kBatchSize) {}

    // True once spilling started and the rows are kept by spill partition.
    [[nodiscard]] bool Partitioned() const { return !parts.empty(); }
};

// True when any of the first n keys can match.
bool AnyValid(const BatchKeys& keys, uint32_t n) {
    if (keys.all_valid) return n > 0;
    return std::any_of(keys.valid.begin(), keys.valid.begin() + n, [](uint8_t v) { return v != 0; });
}

// Keeps the build rows, strings copied, then builds the table. Past the budget,
// rows are kept by spill partition and whole partitions go to disk, largest first.
class BuildSink final : public Sink {
public:
    BuildSink(std::unique_ptr<CompiledExpr> key, KeyRule rule, std::vector<uint32_t> payload, std::vector<ValueArray> arrays, SharedOutput<BuildTable> table)
        : key_(std::move(key)), rule_(rule), payload_(std::move(payload)), arrays_(std::move(arrays)), row_bytes_(FixedBytes(arrays_) + kTableIndexBytes),
          table_(std::move(table)) {}

    // A local without a budget, so it never spills.
    [[nodiscard]] std::unique_ptr<LocalState> MakeLocal() const override {
        return std::make_unique<BuildLocal>(key_->Type(), rule_.form, arrays_.size());
    }

    // A local counted against this worker's budget, spilling to the query's temp directory.
    [[nodiscard]] std::unique_ptr<LocalState> MakeWorkerLocal(const QueryContext& ctx, unsigned worker) const override {
        auto local = std::make_unique<BuildLocal>(key_->Type(), rule_.form, arrays_.size());
        local->reservation = MemoryReservation(ctx.budget, worker);
        local->temp_dir = ctx.temp_dir;
        local->counters = ctx.spill;
        local->spill_everything = ctx.spill_everything;
        return local;
    }

    // Keeps rows whose key can match, in arrival order until spilling starts: a refused grant,
    // spill_everything, or a partition another worker marked. Then rows are kept by partition.
    void Consume(Batch& in, LocalState& local) const override {
        auto& l = static_cast<BuildLocal&>(local);
        const uint32_t n = in.Active();
        if (n == 0) return;
        Normalize(ValuesFor(*key_, in, l.key_values, l.scratch), in.sel, n, rule_, l.keys);

        if (!l.Partitioned()) {
            if (l.spill_everything && AnyValid(l.keys, n)) marked_.fetch_or(kAllPartitions, std::memory_order_relaxed);
            if (marked_.load(std::memory_order_relaxed) == 0) {
                AppendArrival(in, l, n);
                if (Recount(l)) return;
                SortByPartition(l);
                SpillUntilCounted(l);
                return;
            }
            SortByPartition(l);
        }
        // Rows held for newly marked partitions go out before the batch's rows of them.
        const uint64_t marked = marked_.load(std::memory_order_relaxed);
        SpillMarked(l, marked);
        AppendPartitioned(in, l, n, marked);
        SpillUntilCounted(l);
    }

    // Keeps the worker's rows, arena and all, until Finalize.
    void Combine(std::unique_ptr<LocalState> local) override { parts_.push_back(std::move(local)); }

    // Pools the rows, or leaves the floors and spills what can't, then builds by partition.
    // Rows join the table in worker order, then each part's arrival or partition order.
    void Finalize(QueryContext& ctx) override {
        std::optional<MemoryReservation> pooled = LeaveFloors(ctx);

        BuildTable& t = table_.Mutable();
        t.store = ColumnStore(arrays_.size());
        std::vector<ColumnStore*> stores;  // every part's rows, in table order
        std::vector<size_t> kept;          // per reservation in t, the rows it counts
        size_t pooled_rows = 0;
        for (size_t i = 0; i < parts_.size(); ++i) {
            BuildLocal& p = Part(i);
            if (p.writer) {
                std::optional<SpillFile> file = p.writer->TakeFile();
                p.writer.reset();  // before its reservation moves
                if (file) t.files.push_back(std::move(*file));
            }

            size_t rows = p.store.rows;
            if (p.store.rows > 0) stores.push_back(&p.store);
            for (HeldRows& held : p.parts) {
                if (held.store.rows == 0) continue;
                stores.push_back(&held.store);
                rows += held.store.rows;
            }
            kept.push_back(p.pooled ? 0 : rows);
            if (p.pooled) pooled_rows += rows;
            t.reservations.push_back(std::move(p.reservation));
        }
        if (pooled) {
            kept.push_back(pooled_rows);
            t.reservations.push_back(std::move(*pooled));
        }
        AbsorbAll(t.store, arrays_, stores);
        parts_.clear();
        // A marked partition with no rows on disk has nothing to join: not spilled.
        for (size_t q = 0; q < kSpillPartitions; ++q) {
            uint64_t rows = 0;
            for (const SpillFile& f : t.files) rows += f.rows[q];
            if (rows == 0) continue;
            t.spilled.push_back(q);
            t.spilled_mask |= uint64_t{1} << q;
            t.spilled_rows += rows;
        }
        if (t.store.rows > kMaxBuildRows) throw DatabaseException("the parallel executor can't build a hash join of more than 4294967294 rows");

        const size_t count = JoinPartitionCount(t.store.rows, ctx.threads);
        t.bits = __builtin_ctzll(count);
        t.partitions.assign(count, Partition{});
        Scatter(t.store.columns[kHashColumn].i64, t);
        // The partitions hold the hashes now, so the column and its count go.
        // LeaveFloor then gives back the spare grant Shrink keeps, for the table's whole life.
        t.store.columns[kHashColumn] = ColumnBuffer();
        for (size_t i = 0; i < kept.size(); ++i) {
            MemoryReservation& r = t.reservations[i];
            r.Shrink(std::min(r.Bytes(), kept[i] * sizeof(int64_t)));
            (void)r.LeaveFloor();
        }
        ForEachTask(ctx, count, [&t](size_t p) { LinkChains(t.partitions[p].hashes.data(), t.partitions[p].hashes.size(), t.partitions[p].chains); });
        t.load_rows = test_load_rows;
        t.ready = true;
        if (ctx.spill != nullptr) ctx.spill->partitions += t.spilled.size();
        SharedOutput<BuildTable> done = std::move(table_);  // the operator's reference is the last
    }

private:
    // Combined part i.
    [[nodiscard]] BuildLocal& Part(size_t i) { return static_cast<BuildLocal&>(*parts_[i]); }

    // The batch's rows whose key can match, after the rows held, as before spilling existed.
    void AppendArrival(const Batch& in, BuildLocal& l, uint32_t n) const {
        const Selection* keep = &in.sel;
        if (!l.keys.all_valid) {
            uint32_t kept = 0;
            for (uint32_t k = 0; k < n; ++k) {
                if (l.keys.valid[k] != 0) l.keep.Data()[kept++] = in.sel[k];
            }
            l.keep.Resize(kept);
            keep = &l.keep;
        }
        ColumnStore& s = l.store;
        AppendKeys(l.keys, n, rule_.form, s.columns[kKeyColumn], s.arena);
        std::vector<int64_t>& hashes = s.columns[kHashColumn].i64;
        for (uint32_t k = 0; k < n; ++k) {
            if (l.keys.Valid(k)) hashes.push_back(static_cast<int64_t>(l.keys.hashes[k]));
        }
        for (size_t p = 0; p < payload_.size(); ++p) Append(in.slots[payload_[p]], *keep, s.columns[kFirstKept + p], s.arena);
        s.rows += keep->Size();
        l.held = s.rows * row_bytes_ + s.arena.BytesHeld();
    }

    // Sorts the rows held into one store per spill partition, keeping arrival order within each.
    // Copies a column at a time, freeing each once copied.
    void SortByPartition(BuildLocal& l) const {
        ColumnStore& from = l.store;
        const std::vector<int64_t>& hashes = from.columns[kHashColumn].i64;
        std::array<size_t, kSpillPartitions + 1> offsets{};
        std::vector<uint8_t> part_of(from.rows);

        for (size_t r = 0; r < from.rows; ++r) {
            part_of[r] = static_cast<uint8_t>(SpillPartitionOf(static_cast<uint64_t>(hashes[r]), 0));
            ++offsets[part_of[r] + 1];
        }
        for (size_t q = 0; q < kSpillPartitions; ++q) offsets[q + 1] += offsets[q];
        std::array<size_t, kSpillPartitions> next{};
        std::copy(offsets.begin(), offsets.end() - 1, next.begin());
        std::vector<uint32_t> order(from.rows);
        for (size_t r = 0; r < from.rows; ++r) order[next[part_of[r]]++] = static_cast<uint32_t>(r);

        l.parts.resize(kSpillPartitions);
        for (HeldRows& p : l.parts) p.store = ColumnStore(arrays_.size());
        for (size_t c = 0; c < arrays_.size(); ++c) {
            for (size_t q = 0; q < kSpillPartitions; ++q) {
                ColumnStore& to = l.parts[q].store;
                CopyRows(from.columns[c], arrays_[c], order.data() + offsets[q], offsets[q + 1] - offsets[q], to.columns[c], to.arena);
            }
            from.columns[c] = ColumnBuffer();
        }
        l.held = 0;
        for (size_t q = 0; q < kSpillPartitions; ++q) {
            HeldRows& p = l.parts[q];
            p.store.rows = offsets[q + 1] - offsets[q];
            p.bytes = Bytes(p.store);
            l.held += p.bytes;
        }
        from = ColumnStore();
    }

    // The batch's valid rows by spill partition: kept, or written out when in marked.
    void AppendPartitioned(const Batch& in, BuildLocal& l, uint32_t n, uint64_t marked) const {
        std::array<uint32_t, kSpillPartitions + 1> offsets{};
        for (uint32_t k = 0; k < n; ++k) {
            if (!l.keys.Valid(k)) continue;
            l.part_of[k] = static_cast<uint8_t>(SpillPartitionOf(l.keys.hashes[k], 0));
            ++offsets[l.part_of[k] + 1];
        }

        for (size_t q = 0; q < kSpillPartitions; ++q) offsets[q + 1] += offsets[q];
        if (offsets[kSpillPartitions] == 0) return;
        std::array<uint32_t, kSpillPartitions> next{};
        std::copy(offsets.begin(), offsets.end() - 1, next.begin());
        for (uint32_t k = 0; k < n; ++k) {
            if (l.keys.Valid(k)) l.order[next[l.part_of[k]]++] = k;
        }

        for (size_t q = 0; q < kSpillPartitions; ++q) {
            const uint32_t count = offsets[q + 1] - offsets[q];
            if (count == 0) continue;
            const uint32_t* rows = l.order.data() + offsets[q];
            if ((marked >> q & 1) != 0) {
                AppendAt(in, l, rows, count, l.marked_rows);
                WriteAll(Writer(l), q, l.marked_rows.columns, l.marked_rows.rows, l.ids);
                ClearStore(l.marked_rows);
                continue;
            }
            HeldRows& part = l.parts[q];
            AppendAt(in, l, rows, count, part.store);
            const size_t bytes = Bytes(part.store);
            l.held += bytes - part.bytes;
            part.bytes = bytes;
        }
    }

    // A store's estimated bytes: its rows, and the blocks its arena holds.
    [[nodiscard]] size_t Bytes(const ColumnStore& s) const { return s.rows * row_bytes_ + s.arena.BytesHeld(); }

    // Appends the batch's rows at key positions rows[0..n) to store: key, hash, then payload.
    void AppendAt(const Batch& in, BuildLocal& l, const uint32_t* rows, uint32_t n, ColumnStore& store) const {
        AppendKeysAt(l.keys, rule_.form, rows, n, store.columns[kKeyColumn], store.arena);
        std::vector<int64_t>& hashes = store.columns[kHashColumn].i64;
        for (uint32_t i = 0; i < n; ++i) hashes.push_back(static_cast<int64_t>(l.keys.hashes[rows[i]]));
        for (uint32_t i = 0; i < n; ++i) l.pick.Data()[i] = in.sel[rows[i]];
        l.pick.Resize(n);
        for (size_t s = 0; s < payload_.size(); ++s) Append(in.slots[payload_[s]], l.pick, store.columns[kFirstKept + s], store.arena);
        store.rows += n;
    }

    // The worker's writer, made on its first spill; its buffers count in the same reservation.
    [[nodiscard]] PartitionedWriter& Writer(BuildLocal& l) const {
        if (!l.writer) l.writer = std::make_unique<PartitionedWriter>(arrays_, SpillDirectory(l.temp_dir), l.counters, &l.reservation);
        return *l.writer;
    }

    // Brings the reservation to the held bytes; false when the budget refuses.
    [[nodiscard]] static bool Recount(BuildLocal& l) {
        assert(!l.pooled && "pooled rows are counted in the table's reservation");
        if (l.held < l.counted) {
            l.reservation.Shrink(l.counted - l.held);
        } else if (l.held > l.counted && !l.reservation.Grow(l.held - l.counted)) {
            return false;
        }
        l.counted = l.held;
        return true;
    }

    // Marks this worker's largest unmarked partition for every worker; false when it holds none.
    [[nodiscard]] bool MarkLargest(BuildLocal& l) const {
        const uint64_t marked = marked_.load(std::memory_order_relaxed);
        size_t largest = kSpillPartitions;
        for (size_t q = 0; q < kSpillPartitions; ++q) {
            if ((marked >> q & 1) != 0 || l.parts[q].store.rows == 0) continue;
            if (largest == kSpillPartitions || l.parts[q].bytes > l.parts[largest].bytes) largest = q;
        }
        if (largest == kSpillPartitions) return false;
        marked_.fetch_or(uint64_t{1} << largest, std::memory_order_relaxed);
        return true;
    }

    // Writes and frees this worker's rows of the partitions in marked, then flushes the writer.
    // Each partition is uncounted first, so the writer finds room.
    void SpillMarked(BuildLocal& l, uint64_t marked) const {
        bool wrote = false;
        for (size_t q = 0; q < kSpillPartitions; ++q) {
            HeldRows& part = l.parts[q];
            if ((marked >> q & 1) == 0 || part.store.rows == 0) continue;
            l.held -= part.bytes;
            if (l.counted > l.held) {
                l.reservation.Shrink(l.counted - l.held);
                l.counted = l.held;
            }
            WriteAll(Writer(l), q, part.store.columns, part.store.rows, l.ids);
            part = HeldRows{ColumnStore(arrays_.size()), 0};
            wrote = true;
        }
        if (wrote) l.writer->Flush();
    }

    // The same for every partition marked now.
    void SpillMarked(BuildLocal& l) const { SpillMarked(l, marked_.load(std::memory_order_relaxed)); }

    // Writes out the marked partitions, then marks the largest until what's left is counted.
    void SpillUntilCounted(BuildLocal& l) const {
        SpillMarked(l);
        while (!Recount(l) && MarkLargest(l)) SpillMarked(l);
    }

    // The rows outlive this pipeline, so they leave the floors: this thread pools every part's
    // rows into one reservation, once marked rows are on disk. Refused, each part leaves its
    // floor; parts still stuck write marked partitions in parallel rounds, and after each
    // round the rows pool again or more are marked, largest across parts. Returns the pool.
    [[nodiscard]] std::optional<MemoryReservation> LeaveFloors(QueryContext& ctx) {
        // With nothing marked there's nothing to write first, so the rows pool straight away.
        const bool any_marked = marked_.load(std::memory_order_relaxed) != 0;
        std::optional<MemoryReservation> pooled;
        if (!any_marked) pooled = PoolRows(ctx);
        if (pooled) return pooled;

        std::vector<size_t> stuck(parts_.size(), 0);  // per part: bytes that can't leave its floor
        const auto round = [&](const std::vector<size_t>& busy) {
            ForEachTask(ctx, busy.size(), [&](size_t k) { stuck[busy[k]] = WriteMarked(Part(busy[k])); });
            // A part may have finished before another made room, so each stuck one tries again.
            for (size_t i = 0; i < parts_.size(); ++i) {
                if (stuck[i] > 0) stuck[i] = Stuck(Part(i));
            }
        };

        // With nothing marked, a part in arrival order needs no task.
        std::vector<size_t> busy;
        for (size_t i = 0; i < parts_.size(); ++i) {
            if (any_marked || Part(i).Partitioned()) {
                busy.push_back(i);
            } else {
                stuck[i] = Stuck(Part(i));
            }
        }
        if (!busy.empty()) round(busy);
        // The marked rows are on disk now, so the rest pools, stuck or not.
        if (any_marked) pooled = PoolRows(ctx);

        while (!pooled && std::any_of(stuck.begin(), stuck.end(), [](size_t s) { return s > 0; })) {
            // Rows still in arrival order sort by partition, so each partition's bytes are known.
            std::vector<size_t> arrival;
            for (size_t i = 0; i < parts_.size(); ++i) {
                if (!Part(i).Partitioned() && Part(i).store.rows > 0) arrival.push_back(i);
            }
            ForEachTask(ctx, arrival.size(), [&](size_t k) {
                SortByPartition(Part(arrival[k]));
                stuck[arrival[k]] = Stuck(Part(arrival[k]));
            });
            if (!MarkAcross(std::accumulate(stuck.begin(), stuck.end(), size_t{0}))) break;
            busy.clear();
            for (size_t i = 0; i < parts_.size(); ++i) {
                if (HoldsMarked(Part(i))) busy.push_back(i);
            }
            round(busy);
            pooled = PoolRows(ctx);
        }
        assert(std::none_of(parts_.begin(), parts_.end(), [this](const std::unique_ptr<LocalState>& p) { return HoldsMarked(static_cast<const BuildLocal&>(*p)); }) &&
               "every marked partition's rows are on disk");
        return pooled;
    }

    // Pools every part's counted rows; nullopt, with nothing changed, when refused or when
    // a part holds rows it couldn't count. A writer's buffers stay with the part.
    [[nodiscard]] std::optional<MemoryReservation> PoolRows(QueryContext& ctx) {
        for (size_t i = 0; i < parts_.size(); ++i) {
            if (Part(i).held > Part(i).counted) return std::nullopt;
        }
        return PoolParts(ctx.budget, parts_.size(), [this](size_t i) -> BuildLocal& { return Part(i); }, [](const BuildLocal& p) { return p.counted; });
    }

    // Writes the part's rows of marked partitions, sorting arrival-order rows by partition
    // first. Returns the bytes it can't take off its floor.
    [[nodiscard]] size_t WriteMarked(BuildLocal& p) const {
        if (!p.Partitioned() && p.store.rows > 0 && marked_.load(std::memory_order_relaxed) != 0) SortByPartition(p);
        if (p.Partitioned()) SpillMarked(p);
        if (p.writer) p.writer->Flush();  // its buffers aren't the table's floor bytes
        return Stuck(p);
    }

    // Floor bytes the part can't move to the shared pool, plus rows it doesn't count.
    // Recounts first, since sorting by partition can make the rows bigger.
    [[nodiscard]] static size_t Stuck(BuildLocal& p) {
        (void)Recount(p);
        const size_t uncounted = p.held > p.counted ? p.held - p.counted : 0;
        return p.reservation.LeaveFloor() + uncounted;
    }

    // Marks unmarked partitions, largest by bytes summed across parts, until they cover need.
    // False when there's none left to mark.
    [[nodiscard]] bool MarkAcross(size_t need) {
        std::array<size_t, kSpillPartitions> bytes{};
        for (size_t i = 0; i < parts_.size(); ++i) {
            const BuildLocal& p = Part(i);
            if (!p.Partitioned()) continue;
            for (size_t q = 0; q < kSpillPartitions; ++q) {
                if (p.parts[q].store.rows > 0) bytes[q] += p.parts[q].bytes;
            }
        }
        uint64_t marked = marked_.load(std::memory_order_relaxed);
        bool any = false;
        for (size_t covered = 0; covered < need;) {
            size_t largest = kSpillPartitions;
            for (size_t q = 0; q < kSpillPartitions; ++q) {
                if ((marked >> q & 1) != 0 || bytes[q] == 0) continue;
                if (largest == kSpillPartitions || bytes[q] > bytes[largest]) largest = q;
            }
            if (largest == kSpillPartitions) break;
            marked |= uint64_t{1} << largest;
            covered += bytes[largest];
            any = true;
        }
        marked_.store(marked, std::memory_order_relaxed);
        return any;
    }

    // True when the part still holds rows of a marked partition.
    [[nodiscard]] bool HoldsMarked(const BuildLocal& p) const {
        const uint64_t marked = marked_.load(std::memory_order_relaxed);
        if (!p.Partitioned()) return marked != 0 && p.store.rows > 0;
        for (size_t q = 0; q < kSpillPartitions; ++q) {
            if ((marked >> q & 1) != 0 && p.parts[q].store.rows > 0) return true;
        }
        return false;
    }

    std::unique_ptr<CompiledExpr> key_;
    KeyRule rule_;
    std::vector<uint32_t> payload_;
    std::vector<ValueArray> arrays_;  // the row layout: key, hash, payload
    size_t row_bytes_;                // a row's estimated bytes beside its strings
    SharedOutput<BuildTable> table_;
    std::vector<std::unique_ptr<LocalState>> parts_;  // one per worker, after Combine
    mutable std::atomic<uint64_t> marked_{0};         // spill partitions every worker writes out
};

//  Probe

// One piece of a spilled partition: its build and probe rows in some spill files.
struct SpillPiece {
    std::vector<const SpillFile*> build;
    std::vector<const SpillFile*> probe;
    size_t partition{0};  // within those files
    int level{0};
    bool chunked{false};  // loads its build rows a chunk at a time
};

// A split's two files, kept while pieces at `depth` or past it still read them.
struct SplitFiles {
    std::unique_ptr<SpillFile> build;
    std::unique_ptr<SpillFile> probe;
    size_t depth{0};
};

// The next block of one partition across a list of spill files.
struct BlockCursor {
    size_t file{0};
    size_t block{0};
};

// A worker's place in its probe batch, and its scratch; in the spill phase,
// the pieces left, the build rows loaded and the probe block streaming through them.
struct ProbeLocal : LocalState {
    Vector key_values;  // the batch's keys, when the key isn't a bare column
    BatchKeys keys;
    std::vector<uint32_t> probe_rows;  // per pair: its row in the probe batch, or the probe block
    std::vector<uint32_t> build_rows;  // per pair: its build row id, or its loaded row
    uint32_t next_active{0};           // the first active probe row not finished
    uint32_t resume{0};                // that row's next chain entry + 1, or 0 to start at its bucket
    bool pending{false};               // the current input still has rows to walk
    EvalScratch scratch;

    MemoryReservation reservation;  // the writer's buffers, then a spilled partition's loaded rows
    const std::string* temp_dir{nullptr};
    SpillCounters* counters{nullptr};
    unsigned worker{0};
    ColumnStore diverted;          // a batch's rows of spilled partitions, on their way to disk
    std::vector<uint8_t> part_of;  // per diverted row: its spill partition
    std::vector<uint32_t> order;   // the diverted rows' batch positions
    Selection pick;
    std::unique_ptr<PartitionedWriter> writer;  // made on the first diverted row; after the reservation

    std::vector<SpillPiece> pieces;  // this partition's pieces not started, the next one last
    std::vector<SplitFiles> splits;  // the splits those pieces read, the newest last
    SpillPiece piece;                // the piece being joined
    bool in_piece{false};
    std::vector<ColumnBuffer> table;  // its build rows loaded: key, hash, payload
    StringArena table_arena;
    ChainIndex chains;
    size_t counted{0};                // what the reservation holds for the loaded rows
    BlockCursor next_build;           // chunks: the next build block
    BlockCursor next_probe;
    std::vector<ColumnBuffer> block;  // the probe block being streamed
    StringArena block_arena;
    uint32_t block_rows{0};
    uint32_t block_next{0};           // its first row not finished
    uint32_t block_resume{0};         // that row's next chain entry + 1, or 0 to start at its bucket
    std::vector<ColumnBuffer> moving;  // a block on its way to a split
    StringArena moving_arena;

    ProbeLocal(VType key_type, KeyForm form, size_t columns) : key_values(key_type), keys(form), probe_rows(kBatchSize), build_rows(kBatchSize), diverted(columns), part_of(kBatchSize), order(kBatchSize) {}
};

// Pairs probe rows with matching build rows, up to a batch per call. Rows of
// spilled partitions go to disk instead; each spilled partition is joined in the spill phase.
class ProbeOperator final : public Operator {
public:
    ProbeOperator(JoinSpec spec, KeyRule rule, std::vector<ValueArray> build_arrays, std::vector<ValueArray> probe_arrays, std::shared_ptr<const BuildTable> table)
        : key_(std::move(spec.probe_key)), rule_(rule), probe_slots_(std::move(spec.probe_slots)), payload_count_(spec.build_payload.size()),
          types_(std::move(spec.types)), residual_(std::move(spec.residual)), build_left_(spec.build_left), build_arrays_(std::move(build_arrays)),
          probe_arrays_(std::move(probe_arrays)), loaded_row_bytes_(FixedBytes(build_arrays_) + kChainBytes),
          loaded_string_extra_(StringColumns(build_arrays_) * kStringRefExtra + kChainBytes), table_(std::move(table)) {}

    // Each worker keeps its own cursor, so workers never share one.
    [[nodiscard]] std::unique_ptr<LocalState> MakeLocal() const override {
        return std::make_unique<ProbeLocal>(key_->Type(), rule_.form, probe_arrays_.size());
    }

    // A local counted against this worker's budget, spilling to the temp directory.
    // Runs on the calling thread before any worker, so it may size the file slots.
    [[nodiscard]] std::unique_ptr<LocalState> MakeWorkerLocal(const QueryContext& ctx, unsigned worker) const override {
        auto local = std::make_unique<ProbeLocal>(key_->Type(), rule_.form, probe_arrays_.size());
        local->reservation = MemoryReservation(ctx.budget, worker);
        local->temp_dir = ctx.temp_dir;
        local->counters = ctx.spill;
        local->worker = worker;
        if (probe_files_.size() < ctx.threads) probe_files_.resize(ctx.threads);
        return local;
    }

    // HAVE_MORE_OUTPUT while pairs may remain for this input; in is left untouched.
    OpResult Execute(Batch& in, Batch& out, LocalState* local) const override {
        const BuildTable& t = *table_;
        assert(t.ready && "the build side must be finalized before the join runs");
        auto& l = static_cast<ProbeLocal&>(*local);
        ShapeBatch(out, types_);
        const uint32_t active = in.Active();
        if (!l.pending) {
            if (active == 0 || (t.store.rows == 0 && t.spilled_mask == 0)) {
                out.Reset(0);
                return OpResult::NEED_INPUT;
            }
            Normalize(ValuesFor(*key_, in, l.key_values, l.scratch), in.sel, active, rule_, l.keys);
            if (t.spilled_mask != 0) Divert(in, active, l);
            if (t.store.rows == 0) {
                out.Reset(0);
                return OpResult::NEED_INPUT;
            }
            l.next_active = 0;
            l.resume = 0;
            l.pending = true;
        }

        const uint32_t n = Pairs(in.sel, active, l);
        EmitPairs([&](size_t s) -> const Vector& { return in.slots[probe_slots_[s]]; }, table_->store.columns, out, l, n);
        if (l.next_active < active) return OpResult::HAVE_MORE_OUTPUT;
        l.pending = false;
        return OpResult::NEED_INPUT;
    }

    // Hands the worker's diverted rows, flushed, to its own file slot.
    void FinishMorsels(LocalState* local) const override {
        auto& l = static_cast<ProbeLocal&>(*local);
        if (!l.writer) return;
        std::optional<SpillFile> file = l.writer->TakeFile();
        if (!file) return;
        if (l.worker >= probe_files_.size()) throw DatabaseException("the hash join's probe has no spill file slot for worker " + std::to_string(l.worker));
        probe_files_[l.worker].push_back(std::move(*file));
    }

    [[nodiscard]] size_t SpilledPartitionCount() const override { return table_ != nullptr && table_->ready ? table_->spilled.size() : 0; }

    // Frees the last partition's state, then queues spilled partition p from every worker's files.
    // With no rows on either side it has no pairs, so nothing is queued.
    void StartSpilledPartition(size_t p, LocalState* local) const override {
        auto& l = static_cast<ProbeLocal&>(*local);
        const BuildTable& t = *table_;
        EndPiece(l);
        l.pieces.clear();
        l.splits.clear();

        partitions_started_.fetch_add(1, std::memory_order_relaxed);
        SpillPiece piece;
        piece.partition = t.spilled.at(p);

        for (const SpillFile& f : t.files) {
            if (f.rows[piece.partition] > 0) piece.build.push_back(&f);
        }
        for (const std::vector<SpillFile>& slot : probe_files_) {
            for (const SpillFile& f : slot) {
                if (f.rows[piece.partition] > 0) piece.probe.push_back(&f);
            }
        }
        if (!piece.build.empty() && !piece.probe.empty()) l.pieces.push_back(std::move(piece));
    }

    // Loads a piece's build rows (whole or a chunk), then streams its probe blocks through.
    // What a batch points into is freed only in a later call.
    bool NextSpilledBatch(LocalState* local, Batch& out) const override {
        auto& l = static_cast<ProbeLocal&>(*local);
        ShapeBatch(out, types_);
        for (;;) {
            if (!l.in_piece) {
                // Pieces run newest first, so a split whose pieces all ran is the newest.
                while (!l.splits.empty() && l.splits.back().depth >= l.pieces.size()) l.splits.pop_back();
                if (l.pieces.empty()) {
                    out.Reset(0);
                    return false;
                }
                SpillPiece piece = std::move(l.pieces.back());
                l.pieces.pop_back();
                StartPiece(std::move(piece), l);
                continue;
            }
            if (l.block_next >= l.block_rows) {
                if (NextProbeBlock(l)) continue;
                if (l.piece.chunked && NextBlock(l.piece.build, l.piece.partition, l.next_build) != nullptr) {
                    LoadChunk(l);
                    continue;
                }
                EndPiece(l);
                continue;
            }
            const uint32_t n = BlockPairs(l);
            if (n == 0) continue;  // the block ran out with no pair
            EmitPairs([&l](size_t s) -> const ColumnBuffer& { return l.block[kFirstKept + s]; }, l.table, out, l, n);
            return true;
        }
    }

    // The probe pipeline is done: the table and the probe
    // files go, and what tests may ask is kept.
    void PipelineDone() override {
        done_info_ = TableInfo();
        table_.reset();
        probe_files_.clear();
    }

    // What tests may check.
    [[nodiscard]] HashJoinInfo Info() const {
        HashJoinInfo info = table_ != nullptr ? TableInfo() : done_info_;
        info.partitions_started = partitions_started_.load(std::memory_order_relaxed);
        info.splits = splits_.load(std::memory_order_relaxed);
        info.chunked_pieces = chunked_pieces_.load(std::memory_order_relaxed);
        info.loads = loads_.load(std::memory_order_relaxed);
        info.deepest_level = deepest_level_.load(std::memory_order_relaxed);
        return info;
    }

private:
    // What the table says about the build side.
    [[nodiscard]] HashJoinInfo TableInfo() const {
        HashJoinInfo info;
        info.build_left = build_left_;
        info.has_residual = residual_ != nullptr;
        if (table_ == nullptr) return info;
        info.build_rows = table_->ready ? table_->store.rows + table_->spilled_rows : 0;
        info.partitions = table_->partitions.size();
        info.spilled_partitions = table_->spilled.size();
        info.spilled_mask = table_->spilled_mask;
        return info;
    }

    // Collects pairs from the cursor on, for the key form in use.
    uint32_t Pairs(const Selection& sel, uint32_t active, ProbeLocal& l) const {
        const ColumnBuffer& built = table_->store.columns[kKeyColumn];
        const BatchKeys& keys = l.keys;
        switch (rule_.form) {
            case KeyForm::I64: return Walk(sel, active, l, [&](uint32_t g, uint32_t k) { return built.i64[g] == keys.i64[k]; });
            case KeyForm::I128: return Walk(sel, active, l, [&](uint32_t g, uint32_t k) { return built.i128[g] == keys.i128[k]; });
            case KeyForm::F64: return Walk(sel, active, l, [&](uint32_t g, uint32_t k) { return built.f64[g] == keys.f64[k]; });
            case KeyForm::STR: return Walk(sel, active, l, [&](uint32_t g, uint32_t k) { return View(built.str[g]) == View(keys.str[k]); });
        }
        return 0;
    }

    // Walks each active row's chain from the cursor until a batch of pairs is full;
    // the cursor then points at the first entry not yet looked at.
    template <typename Equal>
    uint32_t Walk(const Selection& sel, uint32_t active, ProbeLocal& l, Equal equal) const {
        const BuildTable& t = *table_;
        const BatchKeys& keys = l.keys;
        uint32_t n = 0;
        for (; l.next_active < active; ++l.next_active) {
            const uint32_t k = l.next_active;
            if (!keys.Valid(k)) continue;
            const uint64_t h = keys.hashes[k];
            const Partition& p = t.partitions[PartitionOf(h, t.bits)];
            uint32_t entry = l.resume != 0 ? l.resume : p.chains.buckets[h & p.chains.mask];
            l.resume = 0;
            for (; entry != 0; entry = p.chains.next[entry - 1]) {
                if (n == kBatchSize) {
                    l.resume = entry;
                    return n;
                }
                const uint32_t j = entry - 1;
                if (p.hashes[j] == h && equal(p.rows[j], k)) {
                    l.probe_rows[n] = sel[k];
                    l.build_rows[n] = p.rows[j];
                    ++n;
                }
            }
        }
        return n;
    }

    // Writes n pairs in output order, left side's slots first, keeping those the residual passes.
    // probe(s) is probe slot s's column: a batch's vector, or a spilled block's.
    template <typename ProbeColumn>
    void EmitPairs(ProbeColumn probe, const std::vector<ColumnBuffer>& build, Batch& out, ProbeLocal& l, uint32_t n) const {
        out.Reset(n);
        const size_t first_probe = build_left_ ? payload_count_ : 0;
        const size_t first_payload = build_left_ ? 0 : probe_slots_.size();
        for (size_t s = 0; s < probe_slots_.size(); ++s) Gather(probe(s), l.probe_rows.data(), n, out.slots[first_probe + s]);
        for (size_t s = 0; s < payload_count_; ++s) Gather(build[kFirstKept + s], l.build_rows.data(), n, out.slots[first_payload + s]);
        if (residual_ != nullptr && n > 0) residual_->Select(out, out.sel, l.scratch);
    }

    // Writes the batch's rows of spilled partitions to disk (key, hash, probe slots),
    // then marks them so the walk skips them.
    void Divert(const Batch& in, uint32_t active, ProbeLocal& l) const {
        const uint64_t mask = table_->spilled_mask;
        BatchKeys& keys = l.keys;
        uint32_t n = 0;

        for (uint32_t k = 0; k < active; ++k) {
            if (!keys.Valid(k)) continue;
            const size_t q = SpillPartitionOf(keys.hashes[k], 0);
            if ((mask >> q & 1) == 0) continue;
            l.order[n] = k;
            l.part_of[n] = static_cast<uint8_t>(q);
            ++n;
        }
        if (n == 0) return;

        ColumnStore& d = l.diverted;
        ClearStore(d);
        AppendKeysAt(keys, rule_.form, l.order.data(), n, d.columns[kKeyColumn], d.arena);

        for (uint32_t i = 0; i < n; ++i) d.columns[kHashColumn].i64.push_back(static_cast<int64_t>(keys.hashes[l.order[i]]));
        for (uint32_t i = 0; i < n; ++i) l.pick.Data()[i] = in.sel[l.order[i]];
        l.pick.Resize(n);
        for (size_t s = 0; s < probe_slots_.size(); ++s) Append(in.slots[probe_slots_[s]], l.pick, d.columns[kFirstKept + s], d.arena);
        d.rows = n;
        if (!l.writer) l.writer = std::make_unique<PartitionedWriter>(probe_arrays_, SpillDirectory(l.temp_dir), l.counters, &l.reservation);
        l.writer->AddByPartition(ColumnsOf(d.columns), l.part_of.data(), n);

        // Normalize leaves valid[] alone when every key is valid, so it's filled first.
        if (keys.all_valid) std::fill(keys.valid.begin(), keys.valid.begin() + active, 1);
        keys.all_valid = false;
        for (uint32_t i = 0; i < n; ++i) keys.valid[l.order[i]] = 0;
    }

    // The next recorded block of one partition across
    // files, past the cursor; null at the end.
    static const uint64_t* NextBlock(const std::vector<const SpillFile*>& files, size_t partition, const BlockCursor& at) {
        BlockCursor c = at;
        while (c.file < files.size()) {
            const std::vector<uint64_t>& blocks = files[c.file]->blocks[partition];
            if (c.block < blocks.size()) return &blocks[c.block];
            ++c.file;
            c.block = 0;
        }
        return nullptr;
    }

    // Reads the block at the cursor into out
    // (appending), strings into arena, and moves past it.
    static void ReadNext(const std::vector<const SpillFile*>& files, size_t partition, BlockCursor& at, std::vector<ColumnBuffer>& out, StringArena& arena) {
        while (files[at.file]->blocks[partition].size() <= at.block) {
            ++at.file;
            at.block = 0;
        }
        const SpillFile& file = *files[at.file];
        file.ReadBlockAt(file.blocks[partition][at.block], out, arena);
        ++at.block;
    }

    // True when the budget can give the reservation a grant (no rows are loaded here).
    // Without one no split could fit, so the piece loads in chunks instead.
    [[nodiscard]] static bool CanHoldAGrant(ProbeLocal& l) {
        if (!l.reservation.Grow(1)) return false;
        l.reservation.Shrink(1);  // keeps the grant as a spare for the split
        return true;
    }

    // Loads the build rows whole if they fit, else splits at the next level.
    // At the deepest level, with no grant to spare, or after a one-part split: chunks.
    void StartPiece(SpillPiece piece, ProbeLocal& l) const {
        l.piece = std::move(piece);
        const size_t q = l.piece.partition;
        // Raises the deepest level seen to this piece's.
        for (int seen = deepest_level_.load(std::memory_order_relaxed); seen < l.piece.level;) {
            if (deepest_level_.compare_exchange_weak(seen, l.piece.level, std::memory_order_relaxed)) break;
        }
        uint64_t rows = 0;
        uint64_t bytes = 0;
        for (const SpillFile* f : l.piece.build) {
            rows += f->rows[q];
            bytes += f->bytes[q];
        }
        const size_t cap = table_->load_rows;
        const bool over_cap = cap != 0 && rows > cap;
        const size_t estimate = bytes + rows * loaded_string_extra_;
        bool fits = false;

        if (!l.piece.chunked && !over_cap && rows <= kMaxBuildRows) fits = l.reservation.Grow(estimate);
        if (fits) {
            l.counted = estimate;
            LoadWhole(l, rows);
        } else {
            if (!l.piece.chunked) {
                if (l.piece.level < kMaxSpillLevel && (over_cap || CanHoldAGrant(l))) {
                    Split(l);
                    return;
                }
                l.piece.chunked = true;
            }
            chunked_pieces_.fetch_add(1, std::memory_order_relaxed);
            l.next_build = BlockCursor{};
            LoadChunk(l);
        }
        l.in_piece = true;
    }

    // Every build block of the piece, into a table sized for them.
    void LoadWhole(ProbeLocal& l, uint64_t rows) const {
        l.table.assign(build_arrays_.size(), ColumnBuffer{});
        for (size_t c = 0; c < build_arrays_.size(); ++c) Reserve(l.table[c], build_arrays_[c], rows);
        BlockCursor at;
        while (NextBlock(l.piece.build, l.piece.partition, at) != nullptr) ReadNext(l.piece.build, l.piece.partition, at, l.table, l.table_arena);
        assert(l.table[kHashColumn].i64.size() == rows && "a piece loads every row its files recorded");
        Index(l);
    }

    // The next chunk of the piece's build rows: blocks while the reservation grows, at least
    // one even when it can't (that block stays uncounted), then the probe rows stream again.
    void LoadChunk(ProbeLocal& l) const {
        ClearTable(l);
        l.table.resize(build_arrays_.size());
        const size_t cap = table_->load_rows;
        while (NextBlock(l.piece.build, l.piece.partition, l.next_build) != nullptr) {
            ReadNext(l.piece.build, l.piece.partition, l.next_build, l.table, l.table_arena);
            const size_t rows = l.table[kHashColumn].i64.size();
            const size_t held = rows * loaded_row_bytes_ + l.table_arena.BytesHeld();
            if (held > l.counted) {
                if (!l.reservation.Grow(held - l.counted)) break;
                l.counted = held;
            }
            if ((cap != 0 && rows >= cap) || rows >= kMaxBuildRows / 2) break;
        }
        Index(l);
    }

    // Chains the loaded rows and counts the load.
    void Index(ProbeLocal& l) const {
        const std::vector<int64_t>& hashes = l.table[kHashColumn].i64;
        LinkChains(hashes.data(), hashes.size(), l.chains);
        loads_.fetch_add(1, std::memory_order_relaxed);
        RestartProbe(l);
    }

    // The piece's probe rows start again from their first block.
    static void RestartProbe(ProbeLocal& l) {
        l.next_probe = BlockCursor{};
        l.block_rows = 0;
        l.block_next = 0;
        l.block_resume = 0;
    }

    // Reads the piece's next probe block in place
    // of the last; false when none is left.
    static bool NextProbeBlock(ProbeLocal& l) {
        if (NextBlock(l.piece.probe, l.piece.partition, l.next_probe) == nullptr) return false;
        for (ColumnBuffer& c : l.block) ClearBuffer(c);
        l.block_arena.Clear();
        ReadNext(l.piece.probe, l.piece.partition, l.next_probe, l.block, l.block_arena);
        l.block_rows = static_cast<uint32_t>(l.block[kHashColumn].i64.size());
        l.block_next = 0;
        l.block_resume = 0;
        return true;
    }

    // Frees the loaded rows and uncounts them.
    static void ClearTable(ProbeLocal& l) {
        l.table.clear();
        l.table.shrink_to_fit();
        l.table_arena = StringArena();
        l.chains = ChainIndex();
        if (l.counted > 0) l.reservation.Shrink(std::exchange(l.counted, 0));
    }

    // The piece is done: its rows go, the probe block's memory stays for the next.
    static void EndPiece(ProbeLocal& l) {
        ClearTable(l);
        l.in_piece = false;
        RestartProbe(l);
    }

    // Collects pairs from the probe block's cursor on, for the key form in use.
    uint32_t BlockPairs(ProbeLocal& l) const {
        const ColumnBuffer& built = l.table[kKeyColumn];
        const ColumnBuffer& probe = l.block[kKeyColumn];
        switch (rule_.form) {
            case KeyForm::I64: return WalkBlock(l, [&](uint32_t g, uint32_t i) { return built.i64[g] == probe.i64[i]; });
            case KeyForm::I128: return WalkBlock(l, [&](uint32_t g, uint32_t i) { return built.i128[g] == probe.i128[i]; });
            case KeyForm::F64: return WalkBlock(l, [&](uint32_t g, uint32_t i) { return built.f64[g] == probe.f64[i]; });
            case KeyForm::STR: return WalkBlock(l, [&](uint32_t g, uint32_t i) { return View(built.str[g]) == View(probe.str[i]); });
        }
        return 0;
    }

    // Walks each probe block row's chain over the loaded rows until a batch of
    // pairs is full; the cursor then points at the first entry not yet looked at.
    template <typename Equal>
    uint32_t WalkBlock(ProbeLocal& l, Equal equal) const {
        const ChainIndex& c = l.chains;
        const std::vector<int64_t>& built = l.table[kHashColumn].i64;
        const std::vector<int64_t>& probe = l.block[kHashColumn].i64;
        uint32_t n = 0;
        for (; l.block_next < l.block_rows; ++l.block_next) {
            const uint32_t i = l.block_next;
            const auto h = static_cast<uint64_t>(probe[i]);
            uint32_t entry = l.block_resume != 0 ? l.block_resume : c.buckets[h & c.mask];
            l.block_resume = 0;
            for (; entry != 0; entry = c.next[entry - 1]) {
                if (n == kBatchSize) {
                    l.block_resume = entry;
                    return n;
                }

                const uint32_t g = entry - 1;
                if (static_cast<uint64_t>(built[g]) == h && equal(g, i)) {
                    l.probe_rows[n] = i;
                    l.build_rows[n] = g;
                    ++n;
                }
            }
        }
        return n;
    }

    // Writes both sides again by next-level partition and queues each part with rows on both;
    // a one-part split marks its part chunked, since splitting again can't help.
    void Split(ProbeLocal& l) const {
        const SpillPiece in = l.piece;
        const int level = in.level + 1;
        splits_.fetch_add(1, std::memory_order_relaxed);
        std::unique_ptr<SpillFile> build = Repartition(in.build, in.partition, level, build_arrays_, l);

        if (build == nullptr) return;
        std::unique_ptr<SpillFile> probe = Repartition(in.probe, in.partition, level, probe_arrays_, l);
        if (probe == nullptr) return;
        const SpillFile* b = build.get();
        const SpillFile* p = probe.get();

        l.splits.push_back(SplitFiles{std::move(build), std::move(probe), l.pieces.size()});
        const auto parts = std::count_if(b->rows.begin(), b->rows.end(), [](uint64_t rows) { return rows > 0; });
        for (size_t q = 0; q < kSpillPartitions; ++q) {
            if (b->rows[q] == 0 || p->rows[q] == 0) continue;
            SpillPiece next;
            next.build = {b};
            next.probe = {p};
            next.partition = q;
            next.level = level;
            next.chunked = parts == 1;
            l.pieces.push_back(std::move(next));
        }
    }

    // One side's rows of a partition, written again by
    // their partition at `level`, a block at a time.
    std::unique_ptr<SpillFile> Repartition(const std::vector<const SpillFile*>& files, size_t partition, int level, const std::vector<ValueArray>& arrays, ProbeLocal& l) const {
        PartitionedWriter writer(arrays, SpillDirectory(l.temp_dir), l.counters, &l.reservation);
        BlockCursor at;
        
        while (NextBlock(files, partition, at) != nullptr) {
            for (ColumnBuffer& c : l.moving) ClearBuffer(c);
            l.moving_arena.Clear();
            ReadNext(files, partition, at, l.moving, l.moving_arena);
            const std::vector<int64_t>& hashes = l.moving[kHashColumn].i64;
            l.part_of.resize(std::max<size_t>(l.part_of.size(), hashes.size()));
            for (size_t r = 0; r < hashes.size(); ++r) l.part_of[r] = static_cast<uint8_t>(SpillPartitionOf(static_cast<uint64_t>(hashes[r]), level));
            writer.AddByPartition(ColumnsOf(l.moving), l.part_of.data(), hashes.size());
        }
        std::optional<SpillFile> file = writer.TakeFile();
        return file ? std::make_unique<SpillFile>(std::move(*file)) : nullptr;
    }

    std::unique_ptr<CompiledExpr> key_;
    KeyRule rule_;
    std::vector<uint32_t> probe_slots_;
    size_t payload_count_;
    std::vector<VType> types_;
    std::unique_ptr<CompiledExpr> residual_;
    bool build_left_;
    std::vector<ValueArray> build_arrays_;  // spilled build rows: key, hash, payload
    std::vector<ValueArray> probe_arrays_;  // spilled probe rows: key, hash, probe slots
    size_t loaded_row_bytes_;               // a loaded row's bytes beside its strings
    size_t loaded_string_extra_;            // a loaded row's bytes past its spilled bytes
    std::shared_ptr<const BuildTable> table_;
    // Per worker, the files its diverted rows went to. Sized on the calling thread;
    // each worker appends only to its own, between phases.
    mutable std::vector<std::vector<SpillFile>> probe_files_;
    HashJoinInfo done_info_;  // what TableInfo said when the pipeline finished
    mutable std::atomic<size_t> partitions_started_{0};
    mutable std::atomic<size_t> splits_{0};
    mutable std::atomic<size_t> chunked_pieces_{0};
    mutable std::atomic<size_t> loads_{0};
    mutable std::atomic<int> deepest_level_{0};
};

}  // namespace

// The sink and operator share the built table, so neither outlives what it reads.
// Spilled rows share one layout: the key in the shared form, its hash, kept slots.
std::pair<std::unique_ptr<Sink>, std::unique_ptr<Operator>> MakeHashJoin(JoinSpec spec) {
    if (spec.build_key == nullptr || spec.probe_key == nullptr) throw DatabaseException("the parallel executor's hash join needs both keys");
    if (spec.types.size() != spec.build_payload.size() + spec.probe_slots.size()) {
        throw DatabaseException("the parallel executor's hash join spec has mismatched lists");
    }
    const VType build = spec.build_key->Type();
    const VType probe = spec.probe_key->Type();
    const auto rules = spec.build_left ? KeyRules(build, probe) : KeyRules(probe, build);
    const KeyRule build_rule = spec.build_left ? rules.first : rules.second;
    const KeyRule probe_rule = spec.build_left ? rules.second : rules.first;

    const size_t payload_count = spec.build_payload.size();
    const size_t probe_count = spec.probe_slots.size();
    const auto payload_at = spec.types.begin() + static_cast<std::ptrdiff_t>(spec.build_left ? 0 : probe_count);
    const auto probe_at = spec.types.begin() + static_cast<std::ptrdiff_t>(spec.build_left ? payload_count : 0);
    std::vector<ValueArray> build_arrays = RowArrays(build_rule.form, std::vector<VType>(payload_at, payload_at + static_cast<std::ptrdiff_t>(payload_count)));
    std::vector<ValueArray> probe_arrays = RowArrays(probe_rule.form, std::vector<VType>(probe_at, probe_at + static_cast<std::ptrdiff_t>(probe_count)));

    SharedOutput<BuildTable> table;
    std::shared_ptr<const BuildTable> reader = table.Reader();
    auto sink = std::make_unique<BuildSink>(std::move(spec.build_key), build_rule, spec.build_payload, build_arrays, std::move(table));
    auto op = std::make_unique<ProbeOperator>(std::move(spec), probe_rule, std::move(build_arrays), std::move(probe_arrays), std::move(reader));
    return {std::move(sink), std::move(op)};
}

// Enough partitions for about kJoinRowsPerPartition rows each, and at least one per thread.
size_t JoinPartitionCount(size_t rows, unsigned threads) {
    if (rows <= kJoinRowsPerPartition) return 1;
    const size_t want = std::max<size_t>({2, threads, (rows + kJoinRowsPerPartition - 1) / kJoinRowsPerPartition});
    size_t partitions = 2;
    while (partitions < want && partitions < kMaxJoinPartitions) partitions *= 2;
    return partitions;
}

// Only the probe operator knows its build side.
std::optional<HashJoinInfo> InspectHashJoin(const Operator& op) {
    const auto* probe = dynamic_cast<const ProbeOperator*>(&op);
    if (probe == nullptr) return std::nullopt;
    return probe->Info();
}

// Read once per join, by its build side's Finalize.
void SetHashJoinLoadRowsForTest(size_t most) {
    test_load_rows = most;
}

}  // namespace shilmandb::parallel
