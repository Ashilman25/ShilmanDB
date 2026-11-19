#include "executor/parallel/hash_join.hpp"
#include "common/exception.hpp"
#include "executor/parallel/batch.hpp"
#include "executor/parallel/column_buffer.hpp"
#include "executor/parallel/decimal.hpp"
#include "executor/parallel/key_hash.hpp"
#include "storage/columnar/string_arena.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <string>

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
                const bool fits = !__builtin_mul_overflow(x, factor, &y) && detail::FitsDecimal(y);
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

//  The built table

// A hash's partition: its top bits, since buckets use the low ones.
inline size_t PartitionOf(uint64_t h, int bits) {
    return bits == 0 ? 0 : static_cast<size_t>(h >> (64 - bits));
}

// One hash partition: its rows and hashes, a bucket array and a chain per row.
// It holds everything its build writes, so partitions can build at the same time.
struct Partition {
    std::vector<uint32_t> rows;     // build row ids, in arrival order
    std::vector<uint64_t> hashes;   // one per row here
    std::vector<uint32_t> buckets;  // the first row here + 1, 0 when empty; a power of two
    std::vector<uint32_t> next;     // per row here: the next row in its bucket + 1, 0 at the end
    uint64_t mask{0};
};

// Links a partition's rows into one chain per bucket, each in arrival order.
// Reads and writes only this partition.
void BuildPartition(Partition& p) {
    const size_t n = p.rows.size();
    size_t buckets = 1;
    while (buckets < n) buckets *= 2;
    p.mask = buckets - 1;
    p.buckets.assign(buckets, 0);
    p.next.assign(n, 0);
    for (size_t j = n; j-- > 0;) {
        uint32_t& head = p.buckets[p.hashes[j] & p.mask];
        p.next[j] = head;
        head = static_cast<uint32_t>(j + 1);
    }
}

// The finished build side: written by the sink's Finalize, read by the operator.
struct BuildTable {
    ColumnStore store;  // column 0 the keys in the shared form, then the payload; every worker's strings
    std::vector<Partition> partitions;
    int bits{0};        // a row's partition is its hash's top `bits` bits
    bool ready{false};
};

// Sends each build row to its partition, keeping arrival order within each.
void Scatter(const std::vector<uint64_t>& hashes, BuildTable& t) {
    std::vector<size_t> counts(t.partitions.size(), 0);
    for (uint64_t h : hashes) ++counts[PartitionOf(h, t.bits)];
    for (size_t p = 0; p < counts.size(); ++p) {
        t.partitions[p].rows.reserve(counts[p]);
        t.partitions[p].hashes.reserve(counts[p]);
    }
    for (size_t g = 0; g < hashes.size(); ++g) {
        Partition& p = t.partitions[PartitionOf(hashes[g], t.bits)];
        p.rows.push_back(static_cast<uint32_t>(g));
        p.hashes.push_back(hashes[g]);
    }
}

//  Sink

// One worker's build rows and its scratch for a batch.
struct BuildLocal : LocalState {
    ColumnStore store;             // column 0 the keys, then one per payload slot
    std::vector<uint64_t> hashes;  // one per kept row
    Vector key_values;             // one batch's keys as evaluated
    BatchKeys keys;
    Selection keep;                // the rows whose key can match
    EvalScratch scratch;

    BuildLocal(VType key_type, KeyForm form, size_t columns) : store(columns), key_values(key_type), keys(form) {}
};

// Keeps the build side's keys and payload, strings copied, then builds the table.
class BuildSink final : public Sink {
public:
    BuildSink(std::unique_ptr<CompiledExpr> key, KeyRule rule, std::vector<uint32_t> payload, SharedOutput<BuildTable> table) : key_(std::move(key)), rule_(rule), payload_(std::move(payload)), table_(std::move(table)) {}

    // An empty store: the key column, then one per payload slot.
    [[nodiscard]] std::unique_ptr<LocalState> MakeLocal() const override {
        return std::make_unique<BuildLocal>(key_->Type(), rule_.form, 1 + payload_.size());
    }

    // A row whose key equals nothing is dropped; every other row is kept whole.
    void Consume(Batch& in, LocalState& local) const override {
        auto& l = static_cast<BuildLocal&>(local);
        const uint32_t n = in.Active();
        if (n == 0) return;
        key_->Evaluate(in, in.sel, l.key_values, l.scratch);
        Normalize(l.key_values, in.sel, n, rule_, l.keys);

        const Selection* keep = &in.sel;
        if (!l.keys.all_valid) {
            uint32_t kept = 0;
            for (uint32_t k = 0; k < n; ++k) {
                if (l.keys.valid[k] != 0) l.keep.Data()[kept++] = in.sel[k];
            }
            l.keep.Resize(kept);
            keep = &l.keep;
        }
        AppendKeys(l.keys, n, rule_.form, l.store.columns[0], l.store.arena);
        for (uint32_t k = 0; k < n; ++k) {
            if (l.keys.Valid(k)) l.hashes.push_back(l.keys.hashes[k]);
        }
        for (size_t s = 0; s < payload_.size(); ++s) Append(in.slots[payload_[s]], *keep, l.store.columns[1 + s], l.store.arena);
        l.store.rows += keep->Size();
    }

    // Keeps the worker's rows, arena and all, until Finalize.
    void Combine(std::unique_ptr<LocalState> local) override { parts_.push_back(std::move(local)); }

    // Joins every worker's rows, sends them to hash partitions, then builds each partition,
    // on the pool when there is one.
    void Finalize(QueryContext& ctx) override {
        BuildTable& t = table_.Mutable();
        t.store = ColumnStore(1 + payload_.size());
        std::vector<uint64_t> hashes;
        for (auto& part : parts_) {
            auto& l = static_cast<BuildLocal&>(*part);
            t.store.Absorb(std::move(l.store));
            hashes.insert(hashes.end(), l.hashes.begin(), l.hashes.end());
        }
        parts_.clear();
        if (t.store.rows > kMaxBuildRows) throw DatabaseException("the parallel executor can't build a hash join of more than 4294967294 rows");
        assert(hashes.size() == t.store.rows && "one hash per build row");

        const size_t count = JoinPartitionCount(t.store.rows, ctx.threads);
        t.bits = __builtin_ctzll(count);
        t.partitions.assign(count, Partition{});
        Scatter(hashes, t);
        ForEachTask(ctx, count, [&t](size_t p) { BuildPartition(t.partitions[p]); });
        t.ready = true;
    }

private:
    std::unique_ptr<CompiledExpr> key_;
    KeyRule rule_;
    std::vector<uint32_t> payload_;
    SharedOutput<BuildTable> table_;
    std::vector<std::unique_ptr<LocalState>> parts_;  // one per worker, after Combine
};

//  Probe

// A worker's place in its current probe batch, and its scratch.
struct ProbeLocal : LocalState {
    Vector key_values;  // the batch's keys as evaluated
    BatchKeys keys;
    std::vector<uint32_t> probe_rows;  // per pair: its row in the probe batch
    std::vector<uint32_t> build_rows;  // per pair: its build row id
    uint32_t next_active{0};           // the first active probe row not finished
    uint32_t resume{0};                // that row's next chain entry + 1, or 0 to start at its bucket
    bool pending{false};               // the current input still has rows to walk
    EvalScratch scratch;

    ProbeLocal(VType key_type, KeyForm form) : key_values(key_type), keys(form), probe_rows(kBatchSize), build_rows(kBatchSize) {}
};

// Pairs probe rows with matching build rows, up to a batch per call.
class ProbeOperator final : public Operator {
public:
    ProbeOperator(JoinSpec spec, KeyRule rule, std::shared_ptr<const BuildTable> table)
        : key_(std::move(spec.probe_key)), rule_(rule), probe_slots_(std::move(spec.probe_slots)), payload_count_(spec.build_payload.size()),
          types_(std::move(spec.types)), residual_(std::move(spec.residual)), build_left_(spec.build_left), table_(std::move(table)) {}

    // Each worker keeps its own cursor, so workers never share one.
    [[nodiscard]] std::unique_ptr<LocalState> MakeLocal() const override { return std::make_unique<ProbeLocal>(key_->Type(), rule_.form); }

    // HAVE_MORE_OUTPUT while pairs may remain for this input; in is left untouched.
    OpResult Execute(Batch& in, Batch& out, LocalState* local) const override {
        assert(table_->ready && "the build side must be finalized before the join runs");
        auto& l = static_cast<ProbeLocal&>(*local);
        ShapeBatch(out, types_);
        const uint32_t active = in.Active();
        if (!l.pending) {
            if (active == 0 || table_->store.rows == 0) {
                out.Reset(0);
                return OpResult::NEED_INPUT;
            }
            key_->Evaluate(in, in.sel, l.key_values, l.scratch);
            Normalize(l.key_values, in.sel, active, rule_, l.keys);
            l.next_active = 0;
            l.resume = 0;
            l.pending = true;
        }

        const uint32_t n = Pairs(in.sel, active, l);
        Emit(in, out, l, n);
        if (l.next_active < active) return OpResult::HAVE_MORE_OUTPUT;
        l.pending = false;
        return OpResult::NEED_INPUT;
    }

    // What tests may check.
    [[nodiscard]] HashJoinInfo Info() const {
        HashJoinInfo info;
        info.build_left = build_left_;
        info.has_residual = residual_ != nullptr;
        info.build_rows = table_->ready ? table_->store.rows : 0;
        info.partitions = table_->partitions.size();
        return info;
    }

private:
    // Collects pairs from the cursor on, for the key form in use.
    uint32_t Pairs(const Selection& sel, uint32_t active, ProbeLocal& l) const {
        const ColumnBuffer& built = table_->store.columns[0];
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
            uint32_t entry = l.resume != 0 ? l.resume : p.buckets[h & p.mask];
            l.resume = 0;
            for (; entry != 0; entry = p.next[entry - 1]) {
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

    // Writes n pairs in output order (left side's slots, then the right side's), then keeps
    // the pairs the residual passes.
    void Emit(const Batch& in, Batch& out, ProbeLocal& l, uint32_t n) const {
        out.Reset(n);
        const size_t first_probe = build_left_ ? payload_count_ : 0;
        const size_t first_payload = build_left_ ? 0 : probe_slots_.size();
        for (size_t s = 0; s < probe_slots_.size(); ++s) Gather(in.slots[probe_slots_[s]], l.probe_rows.data(), n, out.slots[first_probe + s]);
        for (size_t s = 0; s < payload_count_; ++s) Gather(table_->store.columns[1 + s], l.build_rows.data(), n, out.slots[first_payload + s]);
        if (residual_ != nullptr && n > 0) residual_->Select(out, out.sel, l.scratch);
    }

    std::unique_ptr<CompiledExpr> key_;
    KeyRule rule_;
    std::vector<uint32_t> probe_slots_;
    size_t payload_count_;
    std::vector<VType> types_;
    std::unique_ptr<CompiledExpr> residual_;
    bool build_left_;
    std::shared_ptr<const BuildTable> table_;
};

}  // namespace

// The sink and operator share the built table, so neither outlives what it reads.
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

    SharedOutput<BuildTable> table;
    std::shared_ptr<const BuildTable> reader = table.Reader();
    auto sink = std::make_unique<BuildSink>(std::move(spec.build_key), build_rule, spec.build_payload, std::move(table));
    auto op = std::make_unique<ProbeOperator>(std::move(spec), probe_rule, std::move(reader));
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

}  // namespace shilmandb::parallel
