#include "executor/parallel/nested_loop_join.hpp"
#include "common/exception.hpp"
#include "executor/parallel/batch.hpp"
#include "executor/parallel/column_buffer.hpp"
#include "storage/columnar/string_arena.hpp"
#include <algorithm>
#include <cassert>
#include <limits>

namespace shilmandb::parallel {

namespace {

// The right side: written by the sink's Finalize, read by the operator.
struct RightRows {
    ColumnStore store;  // one column per kept right slot, with every worker's strings
    bool ready{false};
};

// One worker's copy of the right rows it saw.
struct RightLocal : LocalState {
    ColumnStore store;
};

// Keeps the right side's kept slots, copying strings, since its morsels won't last.
class RightSink final : public Sink {
public:
    RightSink(std::vector<uint32_t> slots, std::shared_ptr<RightRows> out) : slots_(std::move(slots)), out_(std::move(out)) {}

    // One empty buffer per kept slot.
    [[nodiscard]] std::unique_ptr<LocalState> MakeLocal() const override {
        auto local = std::make_unique<RightLocal>();
        local->store = ColumnStore(slots_.size());
        return local;
    }

    // Rows count even with no slots kept, so a cross product still has its size.
    void Consume(Batch& in, LocalState& local) const override {
        auto& l = static_cast<RightLocal&>(local);
        for (size_t k = 0; k < slots_.size(); ++k) Append(in.slots[slots_[k]], in.sel, l.store.columns[k], l.store.arena);
        l.store.rows += in.Active();
    }

    // Keeps the worker's buffers, arena and all, until Finalize.
    void Combine(std::unique_ptr<LocalState> local) override { parts_.push_back(std::move(local)); }

    // Joins every worker's rows into one set the operator reads.
    void Finalize(QueryContext&) override {
        out_->store = ColumnStore(slots_.size());
        for (auto& part : parts_) out_->store.Absorb(std::move(static_cast<RightLocal&>(*part).store));
        parts_.clear();
        if (out_->store.rows > std::numeric_limits<uint32_t>::max()) {
            throw DatabaseException("the parallel executor can't join against more than 4294967295 rows");
        }
        out_->ready = true;
    }

private:
    std::vector<uint32_t> slots_;
    std::shared_ptr<RightRows> out_;
    std::vector<std::unique_ptr<LocalState>> parts_;  // one per worker, after Combine
};

// A worker's place in its current left batch, and its scratch.
struct JoinLocal : LocalState {
    size_t next_right{0};              // the first right row not yet paired with this batch
    std::vector<uint32_t> left_rows;   // per candidate: its left batch row
    std::vector<uint32_t> right_rows;  // per candidate: its right row
    EvalScratch scratch;

    JoinLocal() : left_rows(kBatchSize), right_rows(kBatchSize) {}
};

// Pairs left batches with right rows, a block at a time, keeping passing pairs.
class NestedLoopOperator final : public Operator {
public:
    NestedLoopOperator(NestedLoopSpec spec, std::shared_ptr<const RightRows> right) : spec_(std::move(spec)), right_(std::move(right)) {
        assert(spec_.right_materialized && "the right side is always kept whole first");
        assert(spec_.types.size() == spec_.left_slots.size() + spec_.right_slots.size() && "one type per candidate slot");
    }

    // Each worker keeps its own place, so workers never share one.
    [[nodiscard]] std::unique_ptr<LocalState> MakeLocal() const override { return std::make_unique<JoinLocal>(); }

    // Candidate b * active + k: active left row k, right row next + b.
    // HAVE_MORE_OUTPUT while right rows remain for this batch; in is left untouched.
    OpResult Execute(Batch& in, Batch& out, LocalState* local) const override {
        assert(right_->ready && "the right side must be finalized before the join runs");
        auto& l = static_cast<JoinLocal&>(*local);
        ShapeBatch(out, spec_.types);
        const uint32_t active = in.Active();
        const size_t right = right_->store.rows;
        if (active == 0 || right == 0) {
            out.Reset(0);
            l.next_right = 0;
            return OpResult::NEED_INPUT;
        }

        const auto block = static_cast<uint32_t>(std::min<size_t>(std::max(1u, kBatchSize / active), right - l.next_right));
        const uint32_t n = block * active;
        for (uint32_t b = 0; b < block; ++b) {
            const auto r = static_cast<uint32_t>(l.next_right + b);
            for (uint32_t k = 0; k < active; ++k) {
                l.left_rows[b * active + k] = in.sel[k];
                l.right_rows[b * active + k] = r;
            }
        }

        out.Reset(n);
        const size_t lefts = spec_.left_slots.size();
        for (size_t s = 0; s < lefts; ++s) Gather(in.slots[spec_.left_slots[s]], l.left_rows.data(), n, out.slots[s]);
        for (size_t s = 0; s < spec_.right_slots.size(); ++s) Gather(right_->store.columns[s], l.right_rows.data(), n, out.slots[lefts + s]);
        if (spec_.predicate) spec_.predicate->Select(out, out.sel, l.scratch);

        l.next_right += block;
        if (l.next_right < right) return OpResult::HAVE_MORE_OUTPUT;
        l.next_right = 0;
        return OpResult::NEED_INPUT;
    }

private:
    NestedLoopSpec spec_;
    std::shared_ptr<const RightRows> right_;
};

}  // namespace

// The sink and operator share the right rows, so neither outlives what it reads.
std::pair<std::unique_ptr<Sink>, std::unique_ptr<Operator>> MakeNestedLoopJoin(NestedLoopSpec spec) {
    auto right = std::make_shared<RightRows>();
    auto sink = std::make_unique<RightSink>(spec.right_slots, right);
    auto op = std::make_unique<NestedLoopOperator>(std::move(spec), std::move(right));
    return {std::move(sink), std::move(op)};
}

}  // namespace shilmandb::parallel
