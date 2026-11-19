#include "executor/parallel/pipeline.hpp"
#include "buffer/buffer_pool_manager.hpp"
#include "executor/parallel/worker_pool.hpp"
#include <algorithm>
#include <atomic>
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

// Takes morsels until none are left, the sink is done, or another worker failed.
// started counts this worker's morsels.
void RunMorsels(Pipeline& p, Worker& w, std::atomic<size_t>& next, const std::atomic<bool>& stop, size_t morsels, uint64_t& started) {
    while (!stop.load(std::memory_order_relaxed) && !p.sink->Done()) {
        const size_t m = next.fetch_add(1, std::memory_order_relaxed);
        if (m >= morsels) return;
        ++started;
        p.source->StartMorsel(m, *w.source);
        while (!stop.load(std::memory_order_relaxed) && !p.sink->Done() && p.source->NextBatch(*w.source, w.batches[0])) Push(p, w, 0);
    }
}

// Turns the buffer pool's concurrent mode on, and off again on every exit.
class ConcurrentScope {
public:
    ConcurrentScope(BufferPoolManager* bpm, unsigned workers) : bpm_(bpm) {
        if (bpm_ != nullptr) bpm_->SetConcurrent(true, workers);
    }
    ~ConcurrentScope() {
        if (bpm_ != nullptr) bpm_->SetConcurrent(false);
    }
    ConcurrentScope(const ConcurrentScope&) = delete;
    ConcurrentScope& operator=(const ConcurrentScope&) = delete;

private:
    BufferPoolManager* bpm_;
};

// One worker without a pool, a second thread or a second piece of work.
unsigned WorkersFor(const QueryContext& ctx, size_t count) {
    if (ctx.pool == nullptr || ctx.threads <= 1 || count <= 1) return 1;
    return static_cast<unsigned>(std::min<size_t>({ctx.threads, count, ctx.pool->Size()}));
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

// One worker runs on the calling thread with no latch, as a serial loop.
// Several share an atomic morsel counter; the first to throw stops the rest.
// Combine then goes in worker order, so exact results never depend on timing.
void RunPipeline(Pipeline& p, QueryContext& ctx) {
    p.source->Prepare(ctx);
    const size_t morsels = p.source->MorselCount();
    const unsigned workers = WorkersFor(ctx, morsels);
    std::vector<Worker> locals;
    locals.reserve(workers);
    for (unsigned w = 0; w < workers; ++w) locals.push_back(MakeWorker(p));

    std::atomic<size_t> next{0};
    std::atomic<bool> stop{false};
    std::vector<uint64_t> started(workers, 0);  // each worker writes only its own entry
    if (workers == 1) {
        RunMorsels(p, locals[0], next, stop, morsels, started[0]);
    } else {
        ConcurrentScope concurrent(ctx.bpm, workers);
        ctx.pool->Run(workers, [&](unsigned w) {
            try {
                RunMorsels(p, locals[w], next, stop, morsels, started[w]);
            } catch (...) {
                stop.store(true, std::memory_order_relaxed);
                throw;
            }
        });
    }

    for (uint64_t n : started) ctx.morsels += n;
    for (Worker& w : locals) p.sink->Combine(std::move(w.sink));
    p.sink->Finalize(ctx);
}

// Tasks are taken from a shared counter; a failed task stops the others taking more.
void ForEachTask(QueryContext& ctx, size_t count, const std::function<void(size_t)>& task) {
    const unsigned workers = WorkersFor(ctx, count);
    if (workers == 1) {
        for (size_t i = 0; i < count; ++i) task(i);
        return;
    }
    std::atomic<size_t> next{0};
    std::atomic<bool> stop{false};
    ctx.pool->Run(workers, [&](unsigned) {
        try {
            while (!stop.load(std::memory_order_relaxed)) {
                const size_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= count) return;
                task(i);
            }
        } catch (...) {
            stop.store(true, std::memory_order_relaxed);
            throw;
        }
    });
}

}  // namespace shilmandb::parallel
