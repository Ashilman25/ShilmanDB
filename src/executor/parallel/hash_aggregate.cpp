#include "executor/parallel/hash_aggregate.hpp"
#include "common/exception.hpp"
#include "executor/parallel/batch.hpp"
#include "executor/parallel/column_buffer.hpp"
#include "executor/parallel/decimal.hpp"
#include "executor/parallel/memory_budget.hpp"
#include "executor/parallel/spill_partitions.hpp"
#include "executor/parallel/temp_file.hpp"
#include "storage/columnar/string_arena.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
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

// Set only by tests, on the calling thread.
// A merge of spilled groups splits past this many groups; 0 means no cap.
size_t test_merge_groups = 0;

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
    std::vector<ValueArray> spill_arrays;  // a spilled group's columns: keys, hash, each state's fields
    size_t group_bytes{0};                 // one group's fixed bytes in a table, for the budget
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

// Counting-sorts one table's groups by partition_of(hash), each below `partitions`.
template <typename PartitionOfHash>
Buckets BucketBy(const GroupTable& t, size_t partitions, PartitionOfHash partition_of) {
    Buckets b;
    b.offsets.assign(partitions + 1, 0);
    for (uint32_t g = 0; g < t.size; ++g) ++b.offsets[partition_of(t.hashes[g]) + 1];
    for (size_t p = 0; p < partitions; ++p) b.offsets[p + 1] += b.offsets[p];
    std::vector<size_t> next(b.offsets.begin(), b.offsets.end() - 1);
    b.order.resize(t.size);
    for (uint32_t g = 0; g < t.size; ++g) b.order[next[partition_of(t.hashes[g])]++] = g;
    return b;
}

// Counting-sorts one table's groups by partition.
Buckets BucketByPartition(const GroupTable& t, size_t partitions, int bits) {
    return BucketBy(t, partitions, [bits](uint64_t h) { return PartitionOf(h, bits); });
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

//  Spilling

// A spilled group's columns: keys, the hash, then each state's fields as kept.
// Also one group's fixed bytes in memory, MIN / MAX's seen flag as one byte.
void FillSpillLayout(GroupShape& shape) {
    size_t bytes = 0;
    // A column kept in memory at its array's width.
    const auto column = [&](ValueArray a) {
        shape.spill_arrays.push_back(a);
        bytes += ArrayWidth(a);
    };

    for (const VType& t : shape.keys) column(ArrayOf(t));
    column(ValueArray::I64);  // the hash
    for (const AggInfo& info : shape.aggs) {
        switch (info.kind) {
            case StateKind::COUNT: column(ValueArray::I64); break;
            case StateKind::SUM_EXACT: column(ValueArray::I128); break;
            case StateKind::SUM_DOUBLE: column(ValueArray::F64); break;
            case StateKind::AVG_EXACT:
                column(ValueArray::I128);
                column(ValueArray::I64);
                break;
            case StateKind::AVG_DOUBLE:
                column(ValueArray::F64);
                column(ValueArray::I64);
                break;
            case StateKind::MIN:
            case StateKind::MAX:
                column(ArrayOf(info.arg));
                shape.spill_arrays.push_back(ValueArray::I64);  // the seen flag
                bytes += sizeof(uint8_t);
                break;
            case StateKind::NOT_NUMERIC: break;
        }
    }
    shape.group_bytes = bytes;
}

// A table's bytes by the estimate: fixed bytes per group, index slots, copied strings.
// Vectors' spare capacity isn't counted.
size_t TableBytes(const GroupTable& t, const GroupShape& shape) {
    return t.size * shape.group_bytes + t.index.size() * sizeof(uint32_t) + t.arena.BytesUsed();
}

// Appends group g as a spilled row: keys, hash, then each state's fields as kept.
// Strings still point into t's arenas; the writer copies them.
void AppendSpilledGroup(const GroupTable& t, uint32_t g, const GroupShape& shape, std::vector<ColumnBuffer>& cols) {
    size_t c = 0;
    for (size_t k = 0; k < shape.keys.size(); ++k) PushCell(cols[c++], t.keys[k], g, shape.keys[k]);
    cols[c++].i64.push_back(static_cast<int64_t>(t.hashes[g]));
    for (size_t a = 0; a < shape.aggs.size(); ++a) {
        const StateColumn& s = t.states[a];
        switch (shape.aggs[a].kind) {
            case StateKind::COUNT: cols[c++].i64.push_back(s.count[g]); break;
            case StateKind::SUM_EXACT: cols[c++].i128.push_back(s.exact[g]); break;
            case StateKind::SUM_DOUBLE: cols[c++].f64.push_back(s.real[g]); break;
            case StateKind::AVG_EXACT:
                cols[c++].i128.push_back(s.exact[g]);
                cols[c++].i64.push_back(s.count[g]);
                break;
            case StateKind::AVG_DOUBLE:
                cols[c++].f64.push_back(s.real[g]);
                cols[c++].i64.push_back(s.count[g]);
                break;
            case StateKind::MIN:
            case StateKind::MAX:
                // A batch's winning string is copied before Consume returns, so 2 never gets here.
                assert(s.seen[g] != 2 && "a MIN / MAX string not yet copied");
                PushCell(cols[c++], s.extreme, g, shape.aggs[a].arg);
                cols[c++].i64.push_back(s.seen[g]);
                break;
            case StateKind::NOT_NUMERIC: break;
        }
    }
}

// Writes n groups of t through the writer, each to its spill partition at level.
// id_at(i) is the i-th group's id. Rows are gathered a batch at a time.
template <typename IdAt>
void WriteGroups(const GroupTable& t, const GroupShape& shape, size_t n, IdAt id_at, int level, PartitionedWriter& writer) {
    std::vector<ColumnBuffer> cols(shape.spill_arrays.size());
    std::vector<const ColumnBuffer*> parts;
    for (const ColumnBuffer& col : cols) parts.push_back(&col);
    std::vector<uint8_t> partition_of;
    for (size_t at = 0; at < n; at += kBatchSize) {
        const size_t m = std::min<size_t>(kBatchSize, n - at);
        for (ColumnBuffer& col : cols) ClearBuffer(col);
        partition_of.resize(m);
        for (size_t i = 0; i < m; ++i) {
            const uint32_t g = id_at(at + i);
            AppendSpilledGroup(t, g, shape, cols);
            partition_of[i] = static_cast<uint8_t>(SpillPartitionOf(t.hashes[g], level));
        }
        writer.AddByPartition(parts, partition_of.data(), m);
    }
}

// A spilled block's rows as a table of groups, its columns moved out of block.
// The table has no index: it's only merged from.
GroupTable FromBlock(std::vector<ColumnBuffer>& block, const GroupShape& shape) {
    GroupTable t;
    size_t c = 0;
    for (size_t k = 0; k < shape.keys.size(); ++k) t.keys.push_back(std::move(block[c++]));
    for (const int64_t h : block[c++].i64) t.hashes.push_back(static_cast<uint64_t>(h));
    t.size = t.hashes.size();
    t.states.resize(shape.aggs.size());
    
    for (size_t a = 0; a < shape.aggs.size(); ++a) {
        StateColumn& s = t.states[a];
        switch (shape.aggs[a].kind) {
            case StateKind::COUNT: s.count = std::move(block[c++].i64); break;
            case StateKind::SUM_EXACT: s.exact = std::move(block[c++].i128); break;
            case StateKind::SUM_DOUBLE: s.real = std::move(block[c++].f64); break;
            case StateKind::AVG_EXACT:
                s.exact = std::move(block[c++].i128);
                s.count = std::move(block[c++].i64);
                break;
            case StateKind::AVG_DOUBLE:
                s.real = std::move(block[c++].f64);
                s.count = std::move(block[c++].i64);
                break;
            case StateKind::MIN:
            case StateKind::MAX:
                s.extreme = std::move(block[c++]);
                for (const int64_t seen : block[c++].i64) s.seen.push_back(static_cast<uint8_t>(seen));
                break;
            case StateKind::NOT_NUMERIC: break;
        }
    }
    return t;
}

// Reads the spilled block at `at` as a table of groups, strings into arena.
GroupTable ReadGroups(const SpillFile& file, uint64_t at, const GroupShape& shape, std::vector<ColumnBuffer>& block, StringArena& arena) {
    for (ColumnBuffer& b : block) ClearBuffer(b);
    file.ReadBlockAt(at, block, arena);
    return FromBlock(block, shape);
}

// The groups once merged, written by the sink's Finalize and read by the source.
// After a spill nothing is merged yet: the source merges one spill partition per morsel.
struct MergedGroups {
    std::vector<GroupTable> tables;               // one per partition, or one in all
    std::vector<StringArena> arenas;              // the workers' arenas their strings point into
    std::vector<MemoryReservation> reservations;  // the workers' counts, kept with their groups
    bool spilled{false};
    std::vector<GroupTable> parts;   // after a spill: each worker's groups still in memory
    std::vector<Buckets> buckets;    // each part's groups by level 0 spill partition
    std::vector<SpillFile> files;    // each spilling worker's partial states
    std::vector<size_t> morsels;     // the spill partitions holding any group
    int64_t count_offset{0};         // tests only: added to every COUNT the source emits
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
    MemoryReservation reservation;   // the table's estimated bytes, and the writer's buffers while it writes
    size_t counted{0};               // the part of the reservation that counts the table
    const std::string* temp_dir{nullptr};
    SpillCounters* counters{nullptr};
    bool spill_everything{false};
    std::unique_ptr<PartitionedWriter> writer;  // made on the first spill; after the reservation it counts in
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

// Adds offset to every COUNT state of one merged table.
void AddToCounts(GroupTable& t, const GroupShape& shape, int64_t offset) {
    for (size_t a = 0; a < shape.aggs.size(); ++a) {
        if (shape.aggs[a].kind != StateKind::COUNT) continue;
        for (int64_t& count : t.states[a].count) count += offset;
    }
}

// Groups rows per worker and merges the workers' tables in Finalize.
class AggregateSink final : public Sink {
public:
    AggregateSink(AggregateSpec spec, std::shared_ptr<const GroupShape> shape, SharedOutput<MergedGroups> out) : spec_(std::move(spec)), shape_(std::move(shape)), out_(std::move(out)) {}

    // A local without a budget, so it never spills.
    [[nodiscard]] std::unique_ptr<LocalState> MakeLocal() const override { return NewLocal(); }

    // A local counted against this worker's budget, spilling to the query's temp directory.
    // A fresh global group is counted from the first batch, so empty input never spills.
    [[nodiscard]] std::unique_ptr<LocalState> MakeWorkerLocal(const QueryContext& ctx, unsigned worker) const override {
        auto local = NewLocal();
        local->reservation = MemoryReservation(ctx.budget, worker);
        local->temp_dir = ctx.temp_dir;
        local->counters = ctx.spill;
        local->spill_everything = ctx.spill_everything;
        return local;
    }

    // Finds each active row's group, then folds the rows into every aggregate.
    // Past the budget, or with spill_everything, the table goes to disk.
    void Consume(Batch& in, LocalState& local) const override {
        auto& l = static_cast<AggLocal&>(local);
        const uint32_t n = in.Active();
        if (n == 0) return;
        if (!shape_->global) AssignGroups(in, l, n);
        for (size_t a = 0; a < shape_->aggs.size(); ++a) Update(a, in, l, n);
        if (l.spill_everything || !Recount(l)) SpillTable(l, true);
    }

    // Keeps the worker's table, arena and all, until Finalize.
    void Combine(std::unique_ptr<LocalState> local) override { parts_.push_back(std::move(local)); }

    // The tables outlive this pipeline, so each part leaves its worker's floor, spilling what
    // must stay there. With no spill anywhere they merge as before: into one up to
    // kMergeIntoOneMaxGroups summed groups, else by hash partition on the pool.
    // After a spill the source merges them with the runs, one spill partition per morsel.
    void Finalize(QueryContext& ctx) override {
        ForEachTask(ctx, parts_.size(), [this](size_t i) {
            AggLocal& p = Part(i);
            for (size_t left = p.reservation.LeaveFloor(); left > 0; left = p.reservation.LeaveFloor()) SpillTable(p, false);
        });
        const bool spilled = std::any_of(parts_.begin(), parts_.end(), [](const std::unique_ptr<LocalState>& part) {
            return static_cast<const AggLocal&>(*part).writer != nullptr;
        });
        if (spilled) {
            FinishSpilled(ctx);
            parts_.clear();
            return;
        }

        MergedGroups& out = out_.Mutable();
        std::vector<GroupTable> tables;
        for (auto& part : parts_) {
            AggLocal& p = static_cast<AggLocal&>(*part);
            tables.push_back(std::move(p.table));
            out.reservations.push_back(std::move(p.reservation));
        }
        parts_.clear();
        out.tables.clear();

        const GroupShape& shape = *shape_;
        if (shape.global) {
            GroupTable merged = MakeTable(shape);
            AddGlobalGroup(merged, shape);
            for (const GroupTable& t : tables) MergeStates(merged, 0, shape, t, 0);
            out.tables.push_back(std::move(merged));
        } else if (!tables.empty()) {
            size_t summed = 0;
            for (const GroupTable& t : tables) summed += t.size;
            const size_t partitions = MergePartitionCount(summed, ctx.threads);
            if (partitions == 1) {
                out.tables.push_back(MergeIntoOne(tables, shape));
            } else {
                const int bits = __builtin_ctzll(partitions);
                std::vector<Buckets> buckets;
                for (const GroupTable& t : tables) buckets.push_back(BucketByPartition(t, partitions, bits));
                // Each task writes only its own slot, so partitions merge at once.
                out.tables.resize(partitions);
                ForEachTask(ctx, partitions, [&](size_t p) { out.tables[p] = MergePartition(tables, buckets, p, shape); });
            }
        }
        if (count_offset_ != 0) {
            for (GroupTable& t : out.tables) AddToCounts(t, shape, count_offset_);
        }
        for (GroupTable& t : tables) out.arenas.push_back(std::move(t.arena));
    }

    // Tests only: what Finalize adds to every COUNT.
    void OffsetCounts(int64_t offset) { count_offset_ = offset; }

private:
    // A global local starts with its one group, so empty input still gives a row.
    [[nodiscard]] std::unique_ptr<AggLocal> NewLocal() const {
        auto local = std::make_unique<AggLocal>();
        local->table = MakeTable(*shape_);
        for (const VType& t : shape_->keys) local->key_values.emplace_back(t);
        for (size_t a = 0; a < shape_->aggs.size(); ++a) local->arg_values.emplace_back(spec_.args[a] ? spec_.args[a]->Type() : VType{});
        local->hashes.resize(kBatchSize);
        local->group_of.assign(kBatchSize, 0);
        if (shape_->global) AddGlobalGroup(local->table, *shape_);
        return local;
    }

    // Combined part i.
    [[nodiscard]] AggLocal& Part(size_t i) { return static_cast<AggLocal&>(*parts_[i]); }

    // Brings the table's share of the reservation to its bytes; false when the budget refuses.
    [[nodiscard]] bool Recount(AggLocal& l) const {
        const size_t held = TableBytes(l.table, *shape_);
        if (held < l.counted) {
            l.reservation.Shrink(l.counted - held);
        } else if (held > l.counted && !l.reservation.Grow(held - l.counted)) {
            return false;
        }
        l.counted = held;
        return true;
    }

    // Writes every group to disk by spill partition, then starts a fresh table; refill gives
    // a global table its one group back, counted from the next batch.
    // The table is uncounted first, so the writer finds room.
    void SpillTable(AggLocal& l, bool refill) const {
        const GroupShape& shape = *shape_;
        if (l.counted > 0) l.reservation.Shrink(std::exchange(l.counted, 0));
        if (l.table.size > 0) {
            if (!l.writer) l.writer = std::make_unique<PartitionedWriter>(shape.spill_arrays, SpillDirectory(l.temp_dir), l.counters, &l.reservation);
            WriteGroups(l.table, shape, l.table.size, [](size_t g) { return static_cast<uint32_t>(g); }, 0, *l.writer);
            l.writer->Flush();
        }
        l.table = MakeTable(shape);
        if (refill && shape.global) AddGlobalGroup(l.table, shape);
    }

    // Hands the source every worker's table, its spill file and its count, unmerged.
    // Each spill partition with runs counts once in the query's spill totals.
    void FinishSpilled(QueryContext& ctx) {
        MergedGroups& out = out_.Mutable();
        out.spilled = true;
        out.count_offset = count_offset_;
        for (size_t i = 0; i < parts_.size(); ++i) {
            AggLocal& p = Part(i);
            if (p.writer) {
                std::optional<SpillFile> file = p.writer->TakeFile();
                p.writer.reset();  // before its reservation moves
                if (file) out.files.push_back(std::move(*file));
            }
            out.parts.push_back(std::move(p.table));
            out.reservations.push_back(std::move(p.reservation));
        }
        // Each task writes only its own slot.
        out.buckets.resize(out.parts.size());
        ForEachTask(ctx, out.parts.size(), [&out](size_t i) {
            out.buckets[i] = BucketBy(out.parts[i], kSpillPartitions, [](uint64_t h) { return SpillPartitionOf(h, 0); });
        });
        uint64_t with_runs = 0;
        for (size_t p = 0; p < kSpillPartitions; ++p) {
            const bool runs = std::any_of(out.files.begin(), out.files.end(), [p](const SpillFile& f) { return f.rows[p] > 0; });
            const bool groups = std::any_of(out.buckets.begin(), out.buckets.end(), [p](const Buckets& b) { return b.offsets[p + 1] > b.offsets[p]; });
            with_runs += runs ? 1 : 0;
            if (runs || groups) out.morsels.push_back(p);
        }
        if (ctx.spill != nullptr) ctx.spill->partitions += with_runs;
    }

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
    SharedOutput<MergedGroups> out_;
    std::vector<std::unique_ptr<LocalState>> parts_;  // one per worker, after Combine
    int64_t count_offset_{0};                          // tests only
};

//  Source

// One morsel: up to a batch of consecutive groups of one table.
struct GroupRange {
    uint32_t table;
    uint32_t first;
    uint32_t count;
};

// One piece of a spill partition to merge: the partition itself, from the workers'
// tables and spill files, or a piece a split wrote to its own file.
struct MergeInput {
    const SpillFile* file{nullptr};  // null: the sink's partition
    size_t partition{0};
    int level{0};
    bool forced{false};  // merged even past the budget
};

// Which range a worker emits next, and the row ids to gather.
// After a spill: the partition this worker merges and emits, piece by piece.
struct AggSourceLocal : LocalState {
    GroupRange range{0, 0, 0};
    bool done{true};
    std::vector<uint32_t> rows;
    MemoryReservation reservation;  // the piece being merged, or a split's buffers
    const std::string* temp_dir{nullptr};
    SpillCounters* counters{nullptr};
    std::vector<MergeInput> pending;                // pieces of this morsel not merged yet
    std::vector<std::unique_ptr<SpillFile>> files;  // this morsel's splits
    GroupTable piece;                               // merged groups being emitted
    StringArena arena;                              // strings read from runs, which piece points into
    size_t counted{0};                              // what the reservation holds for piece and arena
    uint32_t next{0};                               // the piece's next group to emit
    std::vector<ColumnBuffer> block;                // one spilled block as read
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
    // After a spill the morsels are spill partitions. The merge's test cap is read here.
    void Prepare(QueryContext&) override {
        ranges_.clear();
        merge_cap_ = test_merge_groups;
        if (groups_->spilled) return;
        for (size_t t = 0; t < groups_->tables.size(); ++t) {
            const size_t size = groups_->tables[t].size;
            for (size_t first = 0; first < size; first += kBatchSize) {
                const auto count = static_cast<uint32_t>(std::min<size_t>(kBatchSize, size - first));
                ranges_.push_back(GroupRange{static_cast<uint32_t>(t), static_cast<uint32_t>(first), count});
            }
        }
    }

    [[nodiscard]] size_t MorselCount() const override { return groups_->spilled ? groups_->morsels.size() : ranges_.size(); }

    // Room for a batch of row ids; a merge after a spill goes uncounted.
    [[nodiscard]] std::unique_ptr<LocalState> MakeLocal() const override { return NewLocal(); }

    // A local whose merges count against this worker's budget and split to the temp directory.
    [[nodiscard]] std::unique_ptr<LocalState> MakeWorkerLocal(const QueryContext& ctx, unsigned worker) const override {
        auto local = NewLocal();
        local->reservation = MemoryReservation(ctx.budget, worker);
        local->temp_dir = ctx.temp_dir;
        local->counters = ctx.spill;
        return local;
    }

    // Points the worker at one range, or after a spill at one spill partition.
    void StartMorsel(size_t morsel, LocalState& local) const override {
        auto& l = static_cast<AggSourceLocal&>(local);
        if (groups_->spilled) {
            StartPartition(groups_->morsels[morsel], l);
            return;
        }
        l.range = ranges_[morsel];
        l.done = false;
        for (uint32_t i = 0; i < l.range.count; ++i) l.rows[i] = l.range.first + i;
    }

    // The range's keys, then each aggregate's value; strings point into the merged arenas.
    bool NextBatch(LocalState& local, Batch& out) const override {
        auto& l = static_cast<AggSourceLocal&>(local);
        if (groups_->spilled) return NextMerged(l, out);
        if (l.done) return false;
        l.done = true;
        FillBatch(groups_->tables[l.range.table], l.range.first, l.range.count, l.rows.data(), out);
        return true;
    }

private:
    // Room for a batch of row ids.
    [[nodiscard]] static std::unique_ptr<AggSourceLocal> NewLocal() {
        auto local = std::make_unique<AggSourceLocal>();
        local->rows.resize(kBatchSize);
        return local;
    }

    // Groups first..first+n-1 of t into out: keys, then each aggregate's value.
    void FillBatch(const GroupTable& t, uint32_t first, uint32_t n, const uint32_t* rows, Batch& out) const {
        ShapeBatch(out, types_);
        out.Reset(n);
        const size_t key_count = shape_->keys.size();
        for (size_t k = 0; k < key_count; ++k) Gather(t.keys[k], rows, n, out.slots[k]);
        for (size_t a = 0; a < shape_->aggs.size(); ++a) Emit(shape_->aggs[a], t.states[a], first, n, rows, out.slots[key_count + a]);
    }

    // Frees the last morsel's piece and splits, then queues partition p.
    // A global aggregate's rows share one hash, so no split could part them.
    void StartPartition(size_t p, AggSourceLocal& l) const {
        ResetPiece(l);
        l.pending.clear();
        l.files.clear();
        l.pending.push_back(MergeInput{nullptr, p, 0, shape_->global});
    }

    // The piece's next groups. A finished piece is freed and the next one merged,
    // so strings (in the workers' tables and this arena) stay valid until the next call.
    bool NextMerged(AggSourceLocal& l, Batch& out) const {
        while (l.next >= l.piece.size) {
            if (l.pending.empty()) return false;
            const MergeInput in = l.pending.back();
            l.pending.pop_back();
            MergePiece(in, l);
        }
        const auto n = static_cast<uint32_t>(std::min<size_t>(kBatchSize, l.piece.size - l.next));
        for (uint32_t i = 0; i < n; ++i) l.rows[i] = l.next + i;
        FillBatch(l.piece, l.next, n, l.rows.data(), out);
        l.next += n;
        return true;
    }

    // Frees the piece and the strings its runs brought in, and uncounts them.
    static void ResetPiece(AggSourceLocal& l) {
        l.piece = GroupTable();
        l.arena.Clear();
        l.next = 0;
        if (l.counted > 0) l.reservation.Shrink(std::exchange(l.counted, 0));
    }

    // Merges one piece; a piece that won't fit is split at the next level instead.
    void MergePiece(const MergeInput& in, AggSourceLocal& l) const {
        ResetPiece(l);
        if (MergeInto(in, l)) {
            if (groups_->count_offset != 0) AddToCounts(l.piece, *shape_, groups_->count_offset);
            return;
        }
        ResetPiece(l);  // uncounted before the split writes, so its buffers find room
        Split(in, l);
    }

    // Brings the reservation to the piece's bytes and its strings; false when the budget refuses.
    [[nodiscard]] bool RecountPiece(AggSourceLocal& l) const {
        const size_t held = TableBytes(l.piece, *shape_) + l.arena.BytesUsed();
        if (held <= l.counted) return true;
        if (!l.reservation.Grow(held - l.counted)) return false;
        l.counted = held;
        return true;
    }

    // True when the reservation holds a grant, or the budget can give one.
    // Without one no split could fit, so the piece merges anyway.
    [[nodiscard]] static bool CanHoldAGrant(AggSourceLocal& l) {
        if (l.counted > 0) return true;
        if (!l.reservation.Grow(1)) return false;
        l.reservation.Shrink(1);  // keeps the grant as a spare for the split
        return true;
    }

    // Merges the piece's groups into l.piece with MergeGroup, counting as it grows.
    // False when the budget or test cap refuses more and the piece may still split.
    bool MergeInto(const MergeInput& in, AggSourceLocal& l) const {
        const GroupShape& shape = *shape_;
        l.piece = MakeTable(shape);
        if (shape.global) AddGlobalGroup(l.piece, shape);
        bool forced = in.forced;
        // True while merging may go on.
        const auto room = [&] {
            if (!forced && merge_cap_ != 0 && l.piece.size > merge_cap_) return false;
            if (RecountPiece(l)) return true;
            if (!forced && !CanHoldAGrant(l)) forced = true;
            return forced;
        };
        if (in.file != nullptr) return MergeRuns(*in.file, in.partition, l, room);
        // Counted across every worker's part, so room is checked every batch of groups.
        size_t merged = 0;
        for (size_t i = 0; i < groups_->parts.size(); ++i) {
            const GroupTable& part = groups_->parts[i];
            const Buckets& b = groups_->buckets[i];
            for (size_t at = b.offsets[in.partition]; at < b.offsets[in.partition + 1]; ++at) {
                MergeGroup(l.piece, shape, part, b.order[at]);
                if (++merged % kBatchSize == 0 && !room()) return false;
            }
        }
        if (!room()) return false;
        for (const SpillFile& file : groups_->files) {
            if (!MergeRuns(file, in.partition, l, room)) return false;
        }
        return true;
    }

    // Merges each block of one spill file partition into l.piece, checking room after each.
    template <typename Room>
    bool MergeRuns(const SpillFile& file, size_t partition, AggSourceLocal& l, Room& room) const {
        for (const uint64_t at : file.blocks[partition]) {
            const GroupTable from = ReadGroups(file, at, *shape_, l.block, l.arena);
            for (uint32_t g = 0; g < from.size; ++g) MergeGroup(l.piece, *shape_, from, g);
            if (!room()) return false;
        }
        return true;
    }

    // Writes the piece's groups to a new file by next-level partition, queuing each part.
    // A part holding every group, or at the deepest level, merges even past the budget.
    void Split(const MergeInput& in, AggSourceLocal& l) const {
        const GroupShape& shape = *shape_;
        const int level = in.level + 1;
        std::optional<SpillFile> written;
        {
            PartitionedWriter writer(shape.spill_arrays, SpillDirectory(l.temp_dir), l.counters, &l.reservation);
            if (in.file != nullptr) {
                SplitRuns(*in.file, in.partition, level, writer, l);
            } else {
                for (size_t i = 0; i < groups_->parts.size(); ++i) {
                    const Buckets& b = groups_->buckets[i];
                    const size_t first = b.offsets[in.partition];
                    const auto id_at = [&b, first](size_t j) { return b.order[first + j]; };
                    WriteGroups(groups_->parts[i], shape, b.offsets[in.partition + 1] - first, id_at, level, writer);
                }
                for (const SpillFile& file : groups_->files) SplitRuns(file, in.partition, level, writer, l);
            }
            written = writer.TakeFile();
        }
        if (!written) return;
        l.files.push_back(std::make_unique<SpillFile>(std::move(*written)));
        const SpillFile& file = *l.files.back();
        const auto parts = std::count_if(file.rows.begin(), file.rows.end(), [](uint64_t rows) { return rows > 0; });
        for (size_t q = 0; q < kSpillPartitions; ++q) {
            if (file.rows[q] > 0) l.pending.push_back(MergeInput{&file, q, level, parts == 1 || level >= kMaxSpillLevel});
        }
    }

    // Writes one spill file partition through writer at level, a block at a time.
    void SplitRuns(const SpillFile& file, size_t partition, int level, PartitionedWriter& writer, AggSourceLocal& l) const {
        StringArena strings;
        for (const uint64_t at : file.blocks[partition]) {
            strings.Clear();
            const GroupTable from = ReadGroups(file, at, *shape_, l.block, strings);
            WriteGroups(from, *shape_, from.size, [](size_t g) { return static_cast<uint32_t>(g); }, level, writer);
        }
    }

    // Writes groups first..first+n-1 of one aggregate into out's rows 0..n-1.
    static void Emit(const AggInfo& info, const StateColumn& s, uint32_t first, uint32_t n, const uint32_t* rows, Vector& out) {
        switch (info.kind) {
            case StateKind::COUNT: {
                int64_t* o = out.I64();
                for (uint32_t i = 0; i < n; ++i) o[i] = s.count[first + i];
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
    size_t merge_cap_{0};  // set by Prepare
};

}  // namespace

// Parallel plans type COUNT BIGINT, SUM and AVG DECIMAL, MIN / MAX as the argument.
VType AggregateOutputType(Aggregate::Func func, const CompiledExpr* arg) {
    if (func == Aggregate::Func::COUNT) return VType{Kind::INTEGER, 0, false, TypeId::BIGINT};
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
    FillSpillLayout(*shape);

    SharedOutput<MergedGroups> groups;
    auto source = std::make_unique<AggregateSource>(shape, groups.Reader());
    auto sink = std::make_unique<AggregateSink>(std::move(spec), std::move(shape), std::move(groups));
    return {std::move(sink), std::move(source)};
}

// Only an aggregate's sink has counts to offset.
void OffsetCountsForTest(Sink& aggregate_sink, int64_t offset) {
    auto* sink = dynamic_cast<AggregateSink*>(&aggregate_sink);
    if (sink == nullptr) throw DatabaseException("OffsetCountsForTest needs an aggregate's sink");
    sink->OffsetCounts(offset);
}

// Read once per aggregate, by its source's Prepare.
void SetAggregateMergeGroupsForTest(size_t most) {
    test_merge_groups = most;
}

}  // namespace shilmandb::parallel
