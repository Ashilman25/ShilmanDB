#include "executor/parallel/hash_aggregate.hpp"
#include "common/exception.hpp"
#include "executor/parallel/batch.hpp"
#include "executor/parallel/column_buffer.hpp"
#include "executor/parallel/decimal.hpp"
#include "storage/columnar/string_arena.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <string_view>

namespace shilmandb::parallel {

namespace {

// Empty strings point here, so their data is never null.
constexpr char kEmpty[] = "";

// The smallest index a grouped table starts with.
constexpr size_t kMinIndexSlots = 1024;

// Groups one table may hold; ids are 32 bits and 0 marks an empty slot.
constexpr size_t kMaxGroups = std::numeric_limits<uint32_t>::max() - 1;

// About this many groups per partition once tables merge by partition.
constexpr size_t kGroupsPerPartition = 65536;

// The most partitions a merge splits into.
constexpr size_t kMaxPartitions = 1024;

// How an aggregate keeps its state.
enum class StateKind : uint8_t { COUNT, SUM_EXACT, SUM_DOUBLE, AVG_EXACT, AVG_DOUBLE, MIN, MAX, NOT_NUMERIC };

// The state kind for func over an argument of type arg.
StateKind StateKindOf(Aggregate::Func func, const VType& arg) {
    const bool exact = arg.kind == Kind::DECIMAL || arg.kind == Kind::INTEGER;
    switch (func) {
        case Aggregate::Func::COUNT: return StateKind::COUNT;
        case Aggregate::Func::SUM: return exact ? StateKind::SUM_EXACT : (arg.kind == Kind::DOUBLE ? StateKind::SUM_DOUBLE : StateKind::NOT_NUMERIC);
        case Aggregate::Func::AVG: return exact ? StateKind::AVG_EXACT : (arg.kind == Kind::DOUBLE ? StateKind::AVG_DOUBLE : StateKind::NOT_NUMERIC);
        case Aggregate::Func::MIN: return StateKind::MIN;
        case Aggregate::Func::MAX: return StateKind::MAX;
    }
    throw DatabaseException("Unknown aggregate function");
}

// One aggregate: how it keeps state, its argument's type, what it emits.
struct AggInfo {
    StateKind kind;
    VType arg;
    VType out;
};

// What every table of one aggregate shares: key types and the aggregates.
struct GroupShape {
    std::vector<VType> keys;
    std::vector<size_t> double_keys;  // indexes of the DOUBLE keys
    std::vector<AggInfo> aggs;
    bool global{false};
};

// One aggregate's state for every group, in the arrays its kind uses.
struct StateColumn {
    std::vector<int64_t> count;   // COUNT, AVG
    std::vector<__int128> exact;  // SUM / AVG of integers and decimals
    std::vector<double> real;     // SUM / AVG of doubles
    ColumnBuffer extreme;         // MIN / MAX
    std::vector<uint8_t> seen;    // MIN / MAX: 0 none yet, 1 kept, 2 a batch's string not yet copied
};

// Groups by id: keys, hashes and states, with an open-addressing index over them.
struct GroupTable {
    std::vector<ColumnBuffer> keys;   // one per key column
    std::vector<uint64_t> hashes;     // one per group
    std::vector<StateColumn> states;  // one per aggregate
    std::vector<uint32_t> index;      // group id + 1 per slot, 0 when empty; a power of two
    StringArena arena;                // strings this table copied in
    size_t size{0};
};

//  Hashing

using detail::DoubleBits;
using detail::FoldKey;
using detail::HashBytes;
using detail::WideBits;

// Equal as group keys: by value, NaN equal to NaN.
inline bool SameDouble(double a, double b) {
    return a == b || (std::isnan(a) && std::isnan(b));
}

// Of two equal DOUBLE keys a group keeps the smaller in a total order:
// -0.0 over +0.0, the NaN with the smallest bits. Any merge order agrees.
inline void KeepSmallerKey(double& kept, double seen) {
    if (TotalOrder(seen, kept) < 0) kept = seen;
}

// Folds key k of every active row into hashes[0..n).
void HashKey(const Vector& v, const Selection& sel, uint32_t n, uint64_t* hashes) {
    switch (ArrayOf(v.Type())) {
        case ValueArray::I64: {
            const int64_t* a = v.I64();
            for (uint32_t i = 0; i < n; ++i) hashes[i] = FoldKey(hashes[i], static_cast<uint64_t>(a[sel[i]]));
            break;
        }
        case ValueArray::I128: {
            const __int128* a = v.I128();
            for (uint32_t i = 0; i < n; ++i) hashes[i] = FoldKey(hashes[i], WideBits(a[sel[i]]));
            break;
        }
        case ValueArray::F64: {
            const double* a = v.F64();
            for (uint32_t i = 0; i < n; ++i) hashes[i] = FoldKey(hashes[i], DoubleBits(a[sel[i]]));
            break;
        }
        case ValueArray::STR:
            for (uint32_t i = 0; i < n; ++i) hashes[i] = FoldKey(hashes[i], HashBytes(View(v.StrAt(sel[i]))));
            break;
    }
}

//  Cells

// Appends the zero of type t: what an empty MIN / MAX shows.
void PushZero(ColumnBuffer& buf, const VType& t) {
    switch (ArrayOf(t)) {
        case ValueArray::I64: buf.i64.push_back(0); break;
        case ValueArray::I128: buf.i128.push_back(0); break;
        case ValueArray::F64: buf.f64.push_back(0.0); break;
        case ValueArray::STR: buf.str.push_back(StringRef{kEmpty, 0}); break;
    }
}

// Appends from's value g; strings keep pointing where from's do.
void PushCell(ColumnBuffer& to, const ColumnBuffer& from, uint32_t g, const VType& t) {
    switch (ArrayOf(t)) {
        case ValueArray::I64: to.i64.push_back(from.i64[g]); break;
        case ValueArray::I128: to.i128.push_back(from.i128[g]); break;
        case ValueArray::F64: to.f64.push_back(from.f64[g]); break;
        case ValueArray::STR: to.str.push_back(from.str[g]); break;
    }
}

// Appends a vector's row, strings copied into arena.
void PushRow(ColumnBuffer& to, const Vector& v, uint32_t row, StringArena& arena) {
    switch (ArrayOf(v.Type())) {
        case ValueArray::I64: to.i64.push_back(v.I64()[row]); break;
        case ValueArray::I128: to.i128.push_back(v.I128()[row]); break;
        case ValueArray::F64: to.f64.push_back(v.F64()[row]); break;
        case ValueArray::STR: to.str.push_back(arena.Copy(View(v.StrAt(row)))); break;
    }
}

// MIN / MAX order: Value's < for integers and decimals, a total order for doubles,
// so -0.0 against +0.0 and NaNs end the same in any merge order.
inline bool ExtremeLess(int64_t a, int64_t b) { return a < b; }
inline bool ExtremeLess(__int128 a, __int128 b) { return a < b; }
inline bool ExtremeLess(double a, double b) { return TotalOrder(a, b) < 0; }

// True when a's value x comes before b's value y in MIN / MAX order.
bool CellLess(const ColumnBuffer& a, uint32_t x, const ColumnBuffer& b, uint32_t y, const VType& t) {
    switch (ArrayOf(t)) {
        case ValueArray::I64: return ExtremeLess(a.i64[x], b.i64[y]);
        case ValueArray::I128: return ExtremeLess(a.i128[x], b.i128[y]);
        case ValueArray::F64: return ExtremeLess(a.f64[x], b.f64[y]);
        case ValueArray::STR: return View(a.str[x]) < View(b.str[y]);
    }
    return false;
}

// Sets to's value x to from's value y.
void SetCell(ColumnBuffer& to, uint32_t x, const ColumnBuffer& from, uint32_t y, const VType& t) {
    switch (ArrayOf(t)) {
        case ValueArray::I64: to.i64[x] = from.i64[y]; break;
        case ValueArray::I128: to.i128[x] = from.i128[y]; break;
        case ValueArray::F64: to.f64[x] = from.f64[y]; break;
        case ValueArray::STR: to.str[x] = from.str[y]; break;
    }
}

//  Group tables

// Empty keys and states for shape.
GroupTable MakeTable(const GroupShape& shape) {
    GroupTable t;
    t.keys.resize(shape.keys.size());
    t.states.resize(shape.aggs.size());
    return t;
}

// Appends a group's fresh states: zero counts and sums, MIN / MAX zero and unseen.
void AddFreshStates(GroupTable& t, const GroupShape& shape) {
    for (size_t a = 0; a < shape.aggs.size(); ++a) {
        StateColumn& s = t.states[a];
        switch (shape.aggs[a].kind) {
            case StateKind::COUNT: s.count.push_back(0); break;
            case StateKind::SUM_EXACT: s.exact.push_back(0); break;
            case StateKind::SUM_DOUBLE: s.real.push_back(0.0); break;
            case StateKind::AVG_EXACT:
                s.exact.push_back(0);
                s.count.push_back(0);
                break;
            case StateKind::AVG_DOUBLE:
                s.real.push_back(0.0);
                s.count.push_back(0);
                break;
            case StateKind::MIN:
            case StateKind::MAX:
                PushZero(s.extreme, shape.aggs[a].arg);
                s.seen.push_back(0);
                break;
            case StateKind::NOT_NUMERIC: break;
        }
    }
}

// Gives a global table its one group, fresh, so empty input still has a row.
void AddGlobalGroup(GroupTable& t, const GroupShape& shape) {
    AddFreshStates(t, shape);
    t.hashes.push_back(0);
    t.size = 1;
}

// Appends a copy of from's group g's states; strings keep pointing into from's arenas.
void CopyStates(GroupTable& t, const GroupShape& shape, const GroupTable& from, uint32_t g) {
    for (size_t a = 0; a < shape.aggs.size(); ++a) {
        StateColumn& s = t.states[a];
        const StateColumn& f = from.states[a];
        switch (shape.aggs[a].kind) {
            case StateKind::COUNT: s.count.push_back(f.count[g]); break;
            case StateKind::SUM_EXACT: s.exact.push_back(f.exact[g]); break;
            case StateKind::SUM_DOUBLE: s.real.push_back(f.real[g]); break;
            case StateKind::AVG_EXACT:
                s.exact.push_back(f.exact[g]);
                s.count.push_back(f.count[g]);
                break;
            case StateKind::AVG_DOUBLE:
                s.real.push_back(f.real[g]);
                s.count.push_back(f.count[g]);
                break;
            case StateKind::MIN:
            case StateKind::MAX:
                PushCell(s.extreme, f.extreme, g, shape.aggs[a].arg);
                s.seen.push_back(f.seen[g]);
                break;
            case StateKind::NOT_NUMERIC: break;
        }
    }
}

// Adds from's group g into t's group `into`; exact parts agree in any order.
void MergeStates(GroupTable& t, uint32_t into, const GroupShape& shape, const GroupTable& from, uint32_t g) {
    for (size_t a = 0; a < shape.aggs.size(); ++a) {
        StateColumn& s = t.states[a];
        const StateColumn& f = from.states[a];
        switch (shape.aggs[a].kind) {
            case StateKind::COUNT: s.count[into] += f.count[g]; break;
            case StateKind::SUM_EXACT: s.exact[into] = CheckedAdd(s.exact[into], f.exact[g]); break;
            case StateKind::SUM_DOUBLE: s.real[into] += f.real[g]; break;
            case StateKind::AVG_EXACT:
                s.exact[into] = CheckedAdd(s.exact[into], f.exact[g]);
                s.count[into] += f.count[g];
                break;
            case StateKind::AVG_DOUBLE:
                s.real[into] += f.real[g];
                s.count[into] += f.count[g];
                break;
            case StateKind::MIN:
            case StateKind::MAX: {
                if (f.seen[g] == 0) break;
                const VType& type = shape.aggs[a].arg;
                const bool min = shape.aggs[a].kind == StateKind::MIN;
                const bool better = min ? CellLess(f.extreme, g, s.extreme, into, type) : CellLess(s.extreme, into, f.extreme, g, type);
                if (s.seen[into] == 0 || better) {
                    SetCell(s.extreme, into, f.extreme, g, type);
                    s.seen[into] = 1;
                }
                break;
            }
            case StateKind::NOT_NUMERIC: break;
        }
    }
}

// Places group g in the index at the first free slot from its hash.
void Place(GroupTable& t, uint32_t g) {
    const size_t mask = t.index.size() - 1;
    size_t slot = t.hashes[g] & mask;
    while (t.index[slot] != 0) slot = (slot + 1) & mask;
    t.index[slot] = g + 1;
}

// Sizes the index so `groups` groups fill at most half of it.
void Reserve(GroupTable& t, size_t groups) {
    size_t slots = std::max(kMinIndexSlots, t.index.size());
    while (groups * 2 > slots) slots *= 2;
    if (slots == t.index.size()) return;
    t.index.assign(slots, 0);
    for (uint32_t g = 0; g < t.size; ++g) Place(t, g);
}

// Finds the group with hash h for which equal(id) holds, or adds one with add().
// Returns its id; added says which.
template <typename Equal, typename Add>
uint32_t FindOrAdd(GroupTable& t, uint64_t h, Equal equal, Add add, bool& added) {
    if ((t.size + 1) * 2 > t.index.size()) Reserve(t, t.size + 1);
    const size_t mask = t.index.size() - 1;
    for (size_t slot = h & mask;; slot = (slot + 1) & mask) {
        const uint32_t entry = t.index[slot];
        if (entry == 0) {
            if (t.size >= kMaxGroups) throw DatabaseException("the parallel executor can't group into more than 4294967294 groups");
            const auto g = static_cast<uint32_t>(t.size);
            add();
            t.hashes.push_back(h);
            t.index[slot] = g + 1;
            ++t.size;
            added = true;
            return g;
        }
        const uint32_t g = entry - 1;
        if (t.hashes[g] == h && equal(g)) {
            added = false;
            return g;
        }
    }
}

// True when t's group g has the keys of a batch row.
bool KeysMatchRow(const GroupTable& t, uint32_t g, const std::vector<Vector>& keys, uint32_t row) {
    for (size_t k = 0; k < keys.size(); ++k) {
        const ColumnBuffer& b = t.keys[k];
        const Vector& v = keys[k];
        switch (ArrayOf(v.Type())) {
            case ValueArray::I64:
                if (b.i64[g] != v.I64()[row]) return false;
                break;
            case ValueArray::I128:
                if (b.i128[g] != v.I128()[row]) return false;
                break;
            case ValueArray::F64:
                if (!SameDouble(b.f64[g], v.F64()[row])) return false;
                break;
            case ValueArray::STR:
                if (View(b.str[g]) != View(v.StrAt(row))) return false;
                break;
        }
    }
    return true;
}

// True when t's group g and from's group h have the same keys.
bool KeysMatchGroup(const GroupTable& t, uint32_t g, const GroupTable& from, uint32_t h, const GroupShape& shape) {
    for (size_t k = 0; k < shape.keys.size(); ++k) {
        const ColumnBuffer& a = t.keys[k];
        const ColumnBuffer& b = from.keys[k];
        switch (ArrayOf(shape.keys[k])) {
            case ValueArray::I64:
                if (a.i64[g] != b.i64[h]) return false;
                break;
            case ValueArray::I128:
                if (a.i128[g] != b.i128[h]) return false;
                break;
            case ValueArray::F64:
                if (!SameDouble(a.f64[g], b.f64[h])) return false;
                break;
            case ValueArray::STR:
                if (View(a.str[g]) != View(b.str[h])) return false;
                break;
        }
    }
    return true;
}

// Adds from's group g to t: merged into the group with its keys, else copied.
void MergeGroup(GroupTable& t, const GroupShape& shape, const GroupTable& from, uint32_t g) {
    bool added = false;
    const uint32_t into = FindOrAdd(
        t, from.hashes[g], [&](uint32_t id) { return KeysMatchGroup(t, id, from, g, shape); },
        [&] {
            for (size_t k = 0; k < shape.keys.size(); ++k) PushCell(t.keys[k], from.keys[k], g, shape.keys[k]);
            CopyStates(t, shape, from, g);
        },
        added);
    if (added) return;
    for (size_t k : shape.double_keys) KeepSmallerKey(t.keys[k].f64[into], from.keys[k].f64[g]);
    MergeStates(t, into, shape, from, g);
}

// Every table merged into the biggest, which keeps its own arena.
GroupTable MergeIntoOne(std::vector<GroupTable>& tables, const GroupShape& shape) {
    size_t biggest = 0;
    for (size_t i = 1; i < tables.size(); ++i) {
        if (tables[i].size > tables[biggest].size) biggest = i;
    }
    GroupTable base = std::move(tables[biggest]);
    size_t total = base.size;
    for (size_t i = 0; i < tables.size(); ++i) total += i == biggest ? 0 : tables[i].size;
    Reserve(base, total);
    for (size_t i = 0; i < tables.size(); ++i) {
        if (i == biggest) continue;
        for (uint32_t g = 0; g < tables[i].size; ++g) MergeGroup(base, shape, tables[i], g);
    }
    return base;
}

// A table's groups ordered by partition: the ids of partition p are
// order[offsets[p]] up to order[offsets[p + 1]].
struct Buckets {
    std::vector<uint32_t> order;
    std::vector<size_t> offsets;
};

// A hash's partition: its top bits, since the index uses the low ones.
inline size_t PartitionOf(uint64_t h, int bits) {
    return bits == 0 ? 0 : static_cast<size_t>(h >> (64 - bits));
}

// Counting-sorts one table's groups by partition.
Buckets BucketByPartition(const GroupTable& t, size_t partitions, int bits) {
    Buckets b;
    b.offsets.assign(partitions + 1, 0);
    for (uint32_t g = 0; g < t.size; ++g) ++b.offsets[PartitionOf(t.hashes[g], bits) + 1];
    for (size_t p = 0; p < partitions; ++p) b.offsets[p + 1] += b.offsets[p];
    std::vector<size_t> next(b.offsets.begin(), b.offsets.end() - 1);
    b.order.resize(t.size);
    for (uint32_t g = 0; g < t.size; ++g) b.order[next[PartitionOf(t.hashes[g], bits)]++] = g;
    return b;
}

// One partition's groups from every table, merged into a table of its own.
// It reads the tables and writes only its result, so partitions may merge at once.
GroupTable MergePartition(const std::vector<GroupTable>& tables, const std::vector<Buckets>& buckets, size_t p, const GroupShape& shape) {
    GroupTable out = MakeTable(shape);
    size_t total = 0;
    for (const Buckets& b : buckets) total += b.offsets[p + 1] - b.offsets[p];
    Reserve(out, total);
    for (size_t i = 0; i < tables.size(); ++i) {
        const Buckets& b = buckets[i];
        for (size_t at = b.offsets[p]; at < b.offsets[p + 1]; ++at) MergeGroup(out, shape, tables[i], b.order[at]);
    }
    return out;
}

// The groups once merged, written by the sink's Finalize and read by the source.
struct MergedGroups {
    std::vector<GroupTable> tables;   // one per partition, or one in all
    std::vector<StringArena> arenas;  // the workers' arenas their strings point into
};

//  Sink

// One worker's groups and its scratch for a batch.
struct AggLocal : LocalState {
    GroupTable table;
    std::vector<Vector> key_values;  // one batch's keys
    std::vector<Vector> arg_values;  // one batch's arguments, per aggregate
    std::vector<uint64_t> hashes;    // per active row
    std::vector<uint32_t> group_of;  // per active row
    std::vector<uint32_t> pending;   // groups holding a batch's MIN / MAX string
    EvalScratch scratch;
};

// Counts each active row in its group; a global aggregate has just group 0.
void CountRows(std::vector<int64_t>& count, const uint32_t* group, uint32_t n, bool global) {
    if (global) {
        count[0] += n;
        return;
    }
    for (uint32_t i = 0; i < n; ++i) ++count[group[i]];
}

// MIN or MAX of plain values over one batch; row i belongs to group group[i].
template <typename T>
void UpdateExtremes(const T* x, const Selection& sel, uint32_t n, const uint32_t* group, std::vector<T>& best, std::vector<uint8_t>& seen, bool min) {
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t g = group[i];
        const T& v = x[sel[i]];
        if (seen[g] == 0 || (min ? ExtremeLess(v, best[g]) : ExtremeLess(best[g], v))) {
            best[g] = v;
            seen[g] = 1;
        }
    }
}

// Groups rows per worker and merges the workers' tables in Finalize.
class AggregateSink final : public Sink {
public:
    AggregateSink(AggregateSpec spec, std::shared_ptr<const GroupShape> shape, std::shared_ptr<MergedGroups> out) : spec_(std::move(spec)), shape_(std::move(shape)), out_(std::move(out)) {}

    // A global local starts with its one group, so empty input still gives a row.
    [[nodiscard]] std::unique_ptr<LocalState> MakeLocal() const override {
        auto local = std::make_unique<AggLocal>();
        local->table = MakeTable(*shape_);
        for (const VType& t : shape_->keys) local->key_values.emplace_back(t);
        for (size_t a = 0; a < shape_->aggs.size(); ++a) local->arg_values.emplace_back(spec_.args[a] ? spec_.args[a]->Type() : VType{});
        local->hashes.resize(kBatchSize);
        local->group_of.assign(kBatchSize, 0);
        if (shape_->global) AddGlobalGroup(local->table, *shape_);
        return local;
    }

    // Finds each active row's group, then folds the rows into every aggregate.
    void Consume(Batch& in, LocalState& local) override {
        auto& l = static_cast<AggLocal&>(local);
        const uint32_t n = in.Active();
        if (n == 0) return;
        if (!shape_->global) AssignGroups(in, l, n);
        for (size_t a = 0; a < shape_->aggs.size(); ++a) Update(a, in, l, n);
    }

    // Keeps the worker's table, arena and all, until Finalize.
    void Combine(std::unique_ptr<LocalState> local) override { parts_.push_back(std::move(local)); }

    // Merges the workers' tables: into one up to kMergeIntoOneMaxGroups summed groups,
    // else into hash partitions, each merged on its own.
    void Finalize(QueryContext& ctx) override {
        std::vector<GroupTable> tables;
        for (auto& part : parts_) tables.push_back(std::move(static_cast<AggLocal&>(*part).table));
        parts_.clear();
        out_->tables.clear();

        const GroupShape& shape = *shape_;
        if (shape.global) {
            GroupTable merged = MakeTable(shape);
            AddGlobalGroup(merged, shape);
            for (const GroupTable& t : tables) MergeStates(merged, 0, shape, t, 0);
            out_->tables.push_back(std::move(merged));
        } else if (!tables.empty()) {
            size_t summed = 0;
            for (const GroupTable& t : tables) summed += t.size;
            const size_t partitions = MergePartitionCount(summed, ctx.threads);
            if (partitions == 1) {
                out_->tables.push_back(MergeIntoOne(tables, shape));
            } else {
                const int bits = __builtin_ctzll(partitions);
                std::vector<Buckets> buckets;
                for (const GroupTable& t : tables) buckets.push_back(BucketByPartition(t, partitions, bits));
                for (size_t p = 0; p < partitions; ++p) out_->tables.push_back(MergePartition(tables, buckets, p, shape));
            }
        }
        for (GroupTable& t : tables) out_->arenas.push_back(std::move(t.arena));
    }

private:
    // Evaluates the keys, hashes them, and finds or adds each active row's group.
    void AssignGroups(const Batch& in, AggLocal& l, uint32_t n) const {
        for (size_t k = 0; k < spec_.keys.size(); ++k) spec_.keys[k]->Evaluate(in, in.sel, l.key_values[k], l.scratch);
        std::fill(l.hashes.begin(), l.hashes.begin() + n, 0);
        for (const Vector& v : l.key_values) HashKey(v, in.sel, n, l.hashes.data());

        GroupTable& t = l.table;
        const GroupShape& shape = *shape_;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t row = in.sel[i];
            bool added = false;
            l.group_of[i] = FindOrAdd(
                t, l.hashes[i], [&](uint32_t g) { return KeysMatchRow(t, g, l.key_values, row); },
                [&] {
                    for (size_t k = 0; k < l.key_values.size(); ++k) PushRow(t.keys[k], l.key_values[k], row, t.arena);
                    AddFreshStates(t, shape);
                },
                added);
            if (added) continue;
            for (size_t k : shape.double_keys) KeepSmallerKey(t.keys[k].f64[l.group_of[i]], l.key_values[k].F64()[row]);
        }
    }

    // Folds one batch into aggregate a's states.
    void Update(size_t a, const Batch& in, AggLocal& l, uint32_t n) const {
        const AggInfo& info = shape_->aggs[a];
        StateColumn& s = l.table.states[a];
        const uint32_t* group = l.group_of.data();
        const bool global = shape_->global;

        if (info.kind == StateKind::COUNT) {
            CountRows(s.count, group, n, global);
            return;
        }

        Vector& v = l.arg_values[a];
        spec_.args[a]->Evaluate(in, in.sel, v, l.scratch);
        const Selection& sel = in.sel;
        switch (info.kind) {
            case StateKind::SUM_EXACT:
            case StateKind::AVG_EXACT:
                AddExact(v, sel, n, group, global, s.exact);
                break;
            case StateKind::SUM_DOUBLE:
            case StateKind::AVG_DOUBLE: {
                const double* x = v.F64();
                for (uint32_t i = 0; i < n; ++i) s.real[group[i]] += x[sel[i]];
                break;
            }
            case StateKind::MIN:
            case StateKind::MAX:
                UpdateExtreme(info, v, sel, n, group, s, l);
                break;
            case StateKind::NOT_NUMERIC:
                throw DatabaseException("Cannot aggregate non-numeric type");
            case StateKind::COUNT:
                break;
        }
        if (info.kind == StateKind::AVG_EXACT || info.kind == StateKind::AVG_DOUBLE) CountRows(s.count, group, n, global);
    }

    // Exact sums. A batch of 64-bit values can't pass 2^75, so a global sum
    // adds the batch first and checks once.
    static void AddExact(const Vector& v, const Selection& sel, uint32_t n, const uint32_t* group, bool global, std::vector<__int128>& sums) {
        if (ArrayOf(v.Type()) == ValueArray::I128) {
            const __int128* x = v.I128();
            for (uint32_t i = 0; i < n; ++i) sums[group[i]] = CheckedAdd(sums[group[i]], x[sel[i]]);
            return;
        }
        const int64_t* x = v.I64();
        if (global) {
            __int128 batch = 0;
            for (uint32_t i = 0; i < n; ++i) batch += x[sel[i]];
            sums[0] = CheckedAdd(sums[0], batch);
            return;
        }
        for (uint32_t i = 0; i < n; ++i) sums[group[i]] = CheckedAdd(sums[group[i]], x[sel[i]]);
    }

    // MIN / MAX. A winning string stays the batch's own bytes until the batch
    // ends, then is copied once into the worker's arena.
    static void UpdateExtreme(const AggInfo& info, const Vector& v, const Selection& sel, uint32_t n, const uint32_t* group, StateColumn& s, AggLocal& l) {
        const bool min = info.kind == StateKind::MIN;
        switch (ArrayOf(info.arg)) {
            case ValueArray::I64: UpdateExtremes(v.I64(), sel, n, group, s.extreme.i64, s.seen, min); break;
            case ValueArray::I128: UpdateExtremes(v.I128(), sel, n, group, s.extreme.i128, s.seen, min); break;
            case ValueArray::F64: UpdateExtremes(v.F64(), sel, n, group, s.extreme.f64, s.seen, min); break;
            case ValueArray::STR: {
                std::vector<StringRef>& best = s.extreme.str;
                for (uint32_t i = 0; i < n; ++i) {
                    const uint32_t g = group[i];
                    const StringRef x = v.StrAt(sel[i]);
                    if (s.seen[g] == 0 || (min ? View(x) < View(best[g]) : View(best[g]) < View(x))) {
                        best[g] = x;
                        if (s.seen[g] != 2) l.pending.push_back(g);
                        s.seen[g] = 2;
                    }
                }
                for (uint32_t g : l.pending) {
                    best[g] = l.table.arena.Copy(View(best[g]));
                    s.seen[g] = 1;
                }
                l.pending.clear();
                break;
            }
        }
    }

    AggregateSpec spec_;
    std::shared_ptr<const GroupShape> shape_;
    std::shared_ptr<MergedGroups> out_;
    std::vector<std::unique_ptr<LocalState>> parts_;  // one per worker, after Combine
};

//  Source

// One morsel: up to a batch of consecutive groups of one table.
struct GroupRange {
    uint32_t table;
    uint32_t first;
    uint32_t count;
};

// Which range a worker emits next, and the row ids to gather.
struct AggSourceLocal : LocalState {
    GroupRange range{0, 0, 0};
    bool done{true};
    std::vector<uint32_t> rows;
};

// Hands out the merged groups, a batch per morsel.
class AggregateSource final : public Source {
public:
    AggregateSource(std::shared_ptr<const GroupShape> shape, std::shared_ptr<const MergedGroups> groups)
        : shape_(std::move(shape)), groups_(std::move(groups)) {
        types_ = shape_->keys;
        for (const AggInfo& info : shape_->aggs) types_.push_back(info.out);
    }

    // Cuts every merged table into ranges of at most a batch.
    void Prepare(QueryContext&) override {
        ranges_.clear();
        for (size_t t = 0; t < groups_->tables.size(); ++t) {
            const size_t size = groups_->tables[t].size;
            for (size_t first = 0; first < size; first += kBatchSize) {
                const auto count = static_cast<uint32_t>(std::min<size_t>(kBatchSize, size - first));
                ranges_.push_back(GroupRange{static_cast<uint32_t>(t), static_cast<uint32_t>(first), count});
            }
        }
    }

    [[nodiscard]] size_t MorselCount() const override { return ranges_.size(); }

    // Room for a batch of row ids.
    [[nodiscard]] std::unique_ptr<LocalState> MakeLocal() const override {
        auto local = std::make_unique<AggSourceLocal>();
        local->rows.resize(kBatchSize);
        return local;
    }

    // Points the worker at one range.
    void StartMorsel(size_t morsel, LocalState& local) override {
        auto& l = static_cast<AggSourceLocal&>(local);
        l.range = ranges_[morsel];
        l.done = false;
        for (uint32_t i = 0; i < l.range.count; ++i) l.rows[i] = l.range.first + i;
    }

    // The range's keys, then each aggregate's value; strings point into the merged arenas.
    bool NextBatch(LocalState& local, Batch& out) override {
        auto& l = static_cast<AggSourceLocal&>(local);
        if (l.done) return false;
        l.done = true;
        const GroupTable& t = groups_->tables[l.range.table];
        const uint32_t n = l.range.count;
        ShapeBatch(out, types_);
        out.Reset(n);
        const size_t key_count = shape_->keys.size();
        for (size_t k = 0; k < key_count; ++k) Gather(t.keys[k], l.rows.data(), n, out.slots[k]);
        for (size_t a = 0; a < shape_->aggs.size(); ++a) {
            Emit(shape_->aggs[a], t.states[a], l.range.first, n, l.rows.data(), out.slots[key_count + a]);
        }
        return true;
    }

private:
    // Writes groups first..first+n-1 of one aggregate into out's rows 0..n-1.
    static void Emit(const AggInfo& info, const StateColumn& s, uint32_t first, uint32_t n, const uint32_t* rows, Vector& out) {
        switch (info.kind) {
            case StateKind::COUNT: {
                int64_t* o = out.I64();
                for (uint32_t i = 0; i < n; ++i) o[i] = static_cast<int32_t>(s.count[first + i]);  // wraps as today
                break;
            }
            case StateKind::SUM_EXACT: {
                __int128* o = out.I128();
                for (uint32_t i = 0; i < n; ++i) o[i] = s.exact[first + i];
                break;
            }
            case StateKind::SUM_DOUBLE: {
                double* o = out.F64();
                for (uint32_t i = 0; i < n; ++i) o[i] = s.real[first + i];
                break;
            }
            case StateKind::AVG_EXACT: {
                const uint8_t scale = info.arg.kind == Kind::DECIMAL ? info.arg.scale : 0;
                double* o = out.F64();
                for (uint32_t i = 0; i < n; ++i) {
                    const int64_t count = s.count[first + i];
                    o[i] = count > 0 ? DecimalToDouble(s.exact[first + i], scale) / static_cast<double>(count) : 0.0;
                }
                break;
            }
            case StateKind::AVG_DOUBLE: {
                double* o = out.F64();
                for (uint32_t i = 0; i < n; ++i) {
                    const int64_t count = s.count[first + i];
                    o[i] = count > 0 ? s.real[first + i] / static_cast<double>(count) : 0.0;
                }
                break;
            }
            case StateKind::MIN:
            case StateKind::MAX:
                Gather(s.extreme, rows, n, out);
                break;
            case StateKind::NOT_NUMERIC: {
                double* o = out.F64();
                for (uint32_t i = 0; i < n; ++i) o[i] = 0.0;
                break;
            }
        }
    }

    std::shared_ptr<const GroupShape> shape_;
    std::shared_ptr<const MergedGroups> groups_;
    std::vector<VType> types_;
    std::vector<GroupRange> ranges_;
};

}  // namespace

// The planner types COUNT INTEGER, SUM and AVG DECIMAL, MIN / MAX as the argument.
VType AggregateOutputType(Aggregate::Func func, const CompiledExpr* arg) {
    if (func == Aggregate::Func::COUNT) return VType{Kind::INTEGER, 0, false, TypeId::INTEGER};
    if (arg == nullptr) throw DatabaseException("the parallel executor needs an argument for this aggregate");
    const VType t = arg->Type();
    const VType real{Kind::DOUBLE, 0, false, TypeId::DECIMAL};
    switch (func) {
        case Aggregate::Func::SUM:
            if (t.kind == Kind::DECIMAL) return VType{Kind::DECIMAL, t.scale, true, TypeId::DECIMAL};
            if (t.kind == Kind::INTEGER) return VType{Kind::DECIMAL, 0, true, TypeId::DECIMAL};
            return real;
        case Aggregate::Func::AVG:
            return real;
        case Aggregate::Func::MIN:
        case Aggregate::Func::MAX:
            return t;
        case Aggregate::Func::COUNT:
            break;
    }
    return VType{Kind::INTEGER, 0, false, TypeId::INTEGER};
}

// Enough partitions for about kGroupsPerPartition groups each, and at least one per thread.
size_t MergePartitionCount(size_t summed_groups, unsigned threads) {
    if (summed_groups <= kMergeIntoOneMaxGroups) return 1;
    const size_t want = std::max<size_t>({2, threads, (summed_groups + kGroupsPerPartition - 1) / kGroupsPerPartition});
    size_t partitions = 2;
    while (partitions < want && partitions < kMaxPartitions) partitions *= 2;
    return partitions;
}

// Sink and source share the shape and merged groups, so neither outlives what it reads.
std::pair<std::unique_ptr<Sink>, std::unique_ptr<Source>> MakeHashAggregate(AggregateSpec spec) {
    if (spec.args.size() != spec.funcs.size() || spec.out_types.size() != spec.keys.size() + spec.funcs.size()) {
        throw DatabaseException("the parallel executor's aggregate spec has mismatched lists");
    }
    if (spec.global && !spec.keys.empty()) throw DatabaseException("the parallel executor's global aggregate has keys");

    auto shape = std::make_shared<GroupShape>();
    shape->global = spec.global;
    for (size_t k = 0; k < spec.keys.size(); ++k) {
        shape->keys.push_back(spec.keys[k]->Type());
        if (spec.out_types[k] != shape->keys.back()) throw DatabaseException("the parallel executor's aggregate key types don't match");
        if (ArrayOf(shape->keys.back()) == ValueArray::F64) shape->double_keys.push_back(k);
    }
    for (size_t a = 0; a < spec.funcs.size(); ++a) {
        const CompiledExpr* arg = spec.funcs[a] == Aggregate::Func::COUNT ? nullptr : spec.args[a].get();
        AggInfo info;
        info.out = AggregateOutputType(spec.funcs[a], arg);
        info.arg = arg != nullptr ? arg->Type() : VType{};
        info.kind = StateKindOf(spec.funcs[a], info.arg);
        if (spec.out_types[spec.keys.size() + a] != info.out) throw DatabaseException("the parallel executor's aggregate output types don't match");
        if (info.kind == StateKind::COUNT) spec.args[a].reset();  // COUNT never reads its argument
        shape->aggs.push_back(info);
    }

    auto groups = std::make_shared<MergedGroups>();
    auto source = std::make_unique<AggregateSource>(shape, groups);
    auto sink = std::make_unique<AggregateSink>(std::move(spec), std::move(shape), std::move(groups));
    return {std::move(sink), std::move(source)};
}

}  // namespace shilmandb::parallel
