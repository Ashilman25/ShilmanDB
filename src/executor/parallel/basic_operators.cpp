#include "executor/parallel/basic_operators.hpp"
#include <algorithm>
#include <cstring>
#include <utility>

namespace shilmandb::parallel {

namespace {

// A worker's scratch for evaluating compiled expressions.
struct ScratchLocal : LocalState {
    EvalScratch scratch;
};

// Copies the active rows of from into to, keeping the "all" flag.
void CopySelection(const Selection& from, Selection& to) {
    if (from.IsAll()) {
        to.SetAll(from.Size());
        return;
    }
    std::memcpy(to.Data(), from.Data(), from.Size() * sizeof(uint16_t));
    to.Resize(from.Size());
}

}  // namespace

FilterOperator::FilterOperator(std::unique_ptr<CompiledExpr> predicate) : predicate_(std::move(predicate)) {}

// Each worker needs its own scratch.
std::unique_ptr<LocalState> FilterOperator::MakeLocal() const {
    return std::make_unique<ScratchLocal>();
}

// Narrows the input's selection in place, then hands the batch on by swapping.
OpResult FilterOperator::Execute(Batch& in, Batch& out, LocalState* local) const {
    predicate_->Select(in, in.sel, static_cast<ScratchLocal*>(local)->scratch);
    std::swap(in, out);
    return OpResult::NEED_INPUT;
}

// Output types come from the compiled expressions, fixed once built.
ProjectionOperator::ProjectionOperator(std::vector<std::unique_ptr<CompiledExpr>> expressions) : expressions_(std::move(expressions)) {
    types_.reserve(expressions_.size());
    for (const auto& e : expressions_) types_.push_back(e->Type());
}

// Each worker needs its own scratch.
std::unique_ptr<LocalState> ProjectionOperator::MakeLocal() const {
    return std::make_unique<ScratchLocal>();
}

// Output rows keep the input's physical positions and selection.
OpResult ProjectionOperator::Execute(Batch& in, Batch& out, LocalState* local) const {
    EvalScratch& scratch = static_cast<ScratchLocal*>(local)->scratch;
    ShapeBatch(out, types_);
    out.Reset(in.rows);
    CopySelection(in.sel, out.sel);
    for (size_t k = 0; k < expressions_.size(); ++k) expressions_[k]->Evaluate(in, in.sel, out.slots[k], scratch);
    return OpResult::NEED_INPUT;
}

// A compare-and-swap loop, so racing workers never take more than the limit.
uint64_t LimitCounter::Reserve(uint64_t wanted) const {
    uint64_t taken = taken_.load(std::memory_order_relaxed);
    for (;;) {
        if (taken >= limit_) return 0;
        const uint64_t granted = std::min(wanted, limit_ - taken);
        if (taken_.compare_exchange_weak(taken, taken + granted, std::memory_order_relaxed)) return granted;
    }
}

// Keeps the first rows the counter grants, then hands the batch on by swapping.
OpResult LimitOperator::Execute(Batch& in, Batch& out, LocalState*) const {
    const uint32_t active = in.Active();
    const auto granted = static_cast<uint32_t>(counter_.Reserve(active));
    if (granted < active) {
        if (in.sel.IsAll()) {
            in.sel.SetAll(granted);
        } else {
            in.sel.Resize(granted);
        }
    }
    std::swap(in, out);
    return OpResult::NEED_INPUT;
}

}  // namespace shilmandb::parallel
