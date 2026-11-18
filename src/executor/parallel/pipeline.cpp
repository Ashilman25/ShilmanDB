#include "executor/parallel/pipeline.hpp"
#include <utility>

namespace shilmandb::parallel {

namespace {

// One worker's locals for a pipeline, and one batch per step.
struct Worker {
    std::unique_ptr<LocalState> source;
    std::vector<std::unique_ptr<LocalState>> operators;
    std::unique_ptr<LocalState> sink;
    std::vector<Batch> batches;  // batches[0] from the source, batches[i + 1] from operator i
};

// A fresh local for every step, and empty batches the steps shape on first use.
Worker MakeWorker(const Pipeline& p) {
    Worker w;
    w.source = p.source->MakeLocal();
    for (const Operator* op : p.operators) w.operators.push_back(op->MakeLocal());
    w.sink = p.sink->MakeLocal();
    w.batches.resize(p.operators.size() + 1);
    return w;
}

// Pushes batches[step] through operator `step` and everything after it.
void Push(Pipeline& p, Worker& w, size_t step) {
    Batch& in = w.batches[step];
    if (in.Active() == 0) return;
    if (step == p.operators.size()) {
        p.sink->Consume(in, *w.sink);
        return;
    }
    Batch& out = w.batches[step + 1];
    for (;;) {
        const OpResult result = p.operators[step]->Execute(in, out, w.operators[step].get());
        Push(p, w, step + 1);
        if (result == OpResult::NEED_INPUT || p.sink->Done()) return;
    }
}

}  // namespace

// Operators without per-worker state need nothing here.
std::unique_ptr<LocalState> Operator::MakeLocal() const {
    return nullptr;
}

// Batches are reused across morsels, so this rebuilds at most a few times per worker.
void ShapeBatch(Batch& batch, const std::vector<VType>& types) {
    bool same = batch.slots.size() == types.size();
    for (size_t i = 0; same && i < types.size(); ++i) same = batch.slots[i].Type() == types[i];
    if (same) return;
    batch.slots.clear();
    batch.slots.reserve(types.size());
    for (const VType& t : types) batch.slots.emplace_back(t);
}

// Morsels run in order on the calling thread; a reached limit stops taking more.
void RunPipeline(Pipeline& p, QueryContext& ctx) {
    p.source->Prepare(ctx);
    Worker w = MakeWorker(p);
    const size_t morsels = p.source->MorselCount();
    for (size_t m = 0; m < morsels && !p.sink->Done(); ++m) {
        p.source->StartMorsel(m, *w.source);
        while (!p.sink->Done() && p.source->NextBatch(*w.source, w.batches[0])) Push(p, w, 0);
    }
    p.sink->Combine(std::move(w.sink));
    p.sink->Finalize(ctx);
}

}  // namespace shilmandb::parallel
