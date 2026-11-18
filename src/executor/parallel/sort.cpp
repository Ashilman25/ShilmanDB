#include "executor/parallel/sort.hpp"
#include "common/exception.hpp"
#include "executor/parallel/batch.hpp"
#include "executor/parallel/column_buffer.hpp"
#include "storage/columnar/string_arena.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <numeric>

namespace shilmandb::parallel {

namespace {

// The sorted rows: written by the sink's Finalize, read by the source.
struct SortedRows {
    ColumnStore kept;             // the kept columns, with every worker's strings
    std::vector<uint32_t> order;  // rows in output order, already cut at the limit
};

// -1, 0 or 1 using < alone, as Value does; NaN equals everything.
template <typename T>
int Order(const T& a, const T& b) {
    return a < b ? -1 : (b < a ? 1 : 0);
}

// Rows a and b of one buffer, compared as Value's < and > compare them.
int CompareAt(const ColumnBuffer& buf, const VType& t, uint32_t a, uint32_t b) {
    switch (t.kind) {
        case Kind::STRING: {
            const int c = View(buf.str[a]).compare(View(buf.str[b]));  // unsigned bytes, then shorter first
            return (c > 0) - (c < 0);
        }
        case Kind::DOUBLE:
            return Order(buf.f64[a], buf.f64[b]);
        case Kind::DECIMAL:
            return t.wide ? Order(buf.i128[a], buf.i128[b]) : Order(buf.i64[a], buf.i64[b]);
        case Kind::INTEGER:
        case Kind::DATE:
            return Order(buf.i64[a], buf.i64[b]);
    }
    return 0;
}

// A total order on doubles: as < where it decides, then -0.0 first, NaN last.
int TotalOrder(double a, double b) {
    const bool a_nan = std::isnan(a);
    const bool b_nan = std::isnan(b);
    if (a_nan != b_nan) return a_nan ? 1 : -1;
    if (a_nan) {
        // NaNs among themselves by their bits, so even they never tie by accident.
        uint64_t x = 0;
        uint64_t y = 0;
        std::memcpy(&x, &a, sizeof(a));
        std::memcpy(&y, &b, sizeof(b));
        return Order(x, y);
    }
    const int c = Order(a, b);
    return c != 0 ? c : Order(std::signbit(b), std::signbit(a));
}

// CompareAt for breaking ties: doubles in a total order, so ties never keep input order.
int TieCompareAt(const ColumnBuffer& buf, const VType& t, uint32_t a, uint32_t b) {
    return t.kind == Kind::DOUBLE ? TotalOrder(buf.f64[a], buf.f64[b]) : CompareAt(buf, t, a, b);
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

// One worker's rows: each key's values and each kept column.
struct SortLocal : LocalState {
    ColumnStore keys;
    ColumnStore kept;
    std::vector<Vector> key_values;  // one batch's keys, before they're appended
    EvalScratch scratch;
};

// Keeps every row it's given; with a limit, each worker keeps only its top n.
class SortSink final : public Sink {
public:
    SortSink(SortSpec spec, std::shared_ptr<SortedRows> out) : spec_(std::move(spec)), out_(std::move(out)) {
        assert(spec_.keys.size() == spec_.ascending.size() && "one direction per key");
        assert(spec_.columns.size() == spec_.types.size() && "one type per kept column");
        for (const auto& key : spec_.keys) key_types_.push_back(key->Type());
        if (spec_.limit) limit_ = static_cast<size_t>(std::max<int64_t>(*spec_.limit, 0));
    }

    // Empty buffers, and one vector per key for a batch's values.
    [[nodiscard]] std::unique_ptr<LocalState> MakeLocal() const override {
        auto local = std::make_unique<SortLocal>();
        local->keys = ColumnStore(key_types_.size());
        local->kept = ColumnStore(spec_.columns.size());
        for (const VType& t : key_types_) local->key_values.emplace_back(t);
        return local;
    }

    // Evaluates the keys, then copies keys and kept columns for the active rows.
    void Consume(Batch& in, LocalState& local) override {
        auto& l = static_cast<SortLocal&>(local);
        for (size_t i = 0; i < spec_.keys.size(); ++i) {
            spec_.keys[i]->Evaluate(in, in.sel, l.key_values[i], l.scratch);
            Append(l.key_values[i], in.sel, l.keys.columns[i], l.keys.arena);
        }
        for (size_t k = 0; k < spec_.columns.size(); ++k) {
            const Vector& v = in.slots[spec_.columns[k]];
            assert(v.Type() == spec_.types[k] && "kept column holds another type");
            Append(v, in.sel, l.kept.columns[k], l.kept.arena);
        }
        l.keys.rows += in.Active();
        l.kept.rows += in.Active();
        // Trimming past n + max(n, a batch) rows keeps the cost per row low.
        if (limit_ && l.kept.rows >= *limit_ + std::max<size_t>(*limit_, kBatchSize)) Trim(l);
    }

    // Keeps the worker's buffers, arena and all, until Finalize.
    void Combine(std::unique_ptr<LocalState> local) override { parts_.push_back(std::move(local)); }

    // Joins every worker's rows, sorts them, and keeps the strings for the source.
    // The key values and their strings go once the order is known.
    void Finalize(QueryContext&) override {
        ColumnStore keys(key_types_.size());
        out_->kept = ColumnStore(spec_.columns.size());
        for (auto& part : parts_) {
            auto& p = static_cast<SortLocal&>(*part);
            keys.Absorb(std::move(p.keys));
            out_->kept.Absorb(std::move(p.kept));
        }
        parts_.clear();
        out_->order = SortedOrder(keys, out_->kept);
    }

private:
    // True when row a comes before row b: keys first, then kept columns ascending.
    // Keys compare as Value does; the tie pass uses a total order on doubles.
    bool Before(const ColumnStore& keys, const ColumnStore& kept, uint32_t a, uint32_t b) const {
        for (size_t i = 0; i < keys.columns.size(); ++i) {
            const int c = CompareAt(keys.columns[i], key_types_[i], a, b);
            if (c != 0) return spec_.ascending[i] ? c < 0 : c > 0;
        }
        for (size_t k = 0; k < kept.columns.size(); ++k) {
            const int c = TieCompareAt(kept.columns[k], spec_.types[k], a, b);
            if (c != 0) return c < 0;
        }
        return false;
    }

    // Row numbers in output order, cut at the limit.
    // Merge sort and heap selection stay in bounds even if a NaN key breaks order.
    std::vector<uint32_t> SortedOrder(const ColumnStore& keys, const ColumnStore& kept) const {
        const size_t rows = kept.rows;
        if (rows > std::numeric_limits<uint32_t>::max()) throw DatabaseException("the parallel executor can't sort more than 4294967295 rows");
        std::vector<uint32_t> order(rows);
        std::iota(order.begin(), order.end(), 0u);
        const auto before = [&](uint32_t a, uint32_t b) { return Before(keys, kept, a, b); };
        if (limit_ && *limit_ < rows) {
            std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(*limit_), order.end(), before);
            order.resize(*limit_);
        } else {
            std::stable_sort(order.begin(), order.end(), before);
        }
        return order;
    }

    // Cuts a worker's rows to its top n, copying their strings into fresh arenas.
    void Trim(SortLocal& l) const {
        const std::vector<uint32_t> keep = SortedOrder(l.keys, l.kept);
        ColumnStore keys(l.keys.columns.size());
        ColumnStore kept(l.kept.columns.size());
        for (size_t i = 0; i < keys.columns.size(); ++i) CopyRows(l.keys.columns[i], key_types_[i], keep, keys.columns[i], keys.arena);
        for (size_t k = 0; k < kept.columns.size(); ++k) CopyRows(l.kept.columns[k], spec_.types[k], keep, kept.columns[k], kept.arena);
        keys.rows = keep.size();
        kept.rows = keep.size();
        l.keys = std::move(keys);
        l.kept = std::move(kept);
    }

    SortSpec spec_;
    std::vector<VType> key_types_;
    std::optional<size_t> limit_;  // the spec's limit, at least 0
    std::shared_ptr<SortedRows> out_;
    std::vector<std::unique_ptr<LocalState>> parts_;  // one per worker, after Combine
};

// Where a worker is in the sorted rows.
struct SortSourceLocal : LocalState {
    size_t next{0};
};

// Hands out the sorted rows in order, kBatchSize at a time.
class SortSource final : public Source {
public:
    SortSource(std::vector<VType> types, std::shared_ptr<const SortedRows> rows) : types_(std::move(types)), rows_(std::move(rows)) {}

    // The sink's Finalize already did the work.
    void Prepare(QueryContext&) override {}

    // One morsel, so one worker emits every row in order.
    [[nodiscard]] size_t MorselCount() const override { return 1; }

    // Only the read position, set by StartMorsel.
    [[nodiscard]] std::unique_ptr<LocalState> MakeLocal() const override { return std::make_unique<SortSourceLocal>(); }

    // Starts from the first sorted row.
    void StartMorsel(size_t, LocalState& local) override { static_cast<SortSourceLocal&>(local).next = 0; }

    // The next rows in order, as plain vectors; strings point into the sink's arenas.
    bool NextBatch(LocalState& local, Batch& out) override {
        auto& l = static_cast<SortSourceLocal&>(local);
        const std::vector<uint32_t>& order = rows_->order;
        if (l.next >= order.size()) return false;
        const auto n = static_cast<uint32_t>(std::min<size_t>(kBatchSize, order.size() - l.next));
        ShapeBatch(out, types_);
        out.Reset(n);
        for (size_t k = 0; k < types_.size(); ++k) Gather(rows_->kept.columns[k], order.data() + l.next, n, out.slots[k]);
        l.next += n;
        return true;
    }

private:
    std::vector<VType> types_;
    std::shared_ptr<const SortedRows> rows_;
};

}  // namespace

// The sink and source share the sorted rows, so neither outlives what it reads.
std::pair<std::unique_ptr<Sink>, std::unique_ptr<Source>> MakeSort(SortSpec spec) {
    auto rows = std::make_shared<SortedRows>();
    auto source = std::make_unique<SortSource>(spec.types, rows);
    auto sink = std::make_unique<SortSink>(std::move(spec), std::move(rows));
    return {std::move(sink), std::move(source)};
}

}  // namespace shilmandb::parallel
