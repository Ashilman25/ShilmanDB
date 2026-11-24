#include "executor/parallel/pipeline.hpp"
#include "buffer/buffer_pool_manager.hpp"
#include "executor/parallel/worker_pool.hpp"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <utility>

namespace shilmandb::parallel {

namespace {

using Clock = std::chrono::steady_clock;
using std::chrono::nanoseconds;

// One worker's time and rows per step: the source, each operator, then the sink.
struct StepProfile {
    std::vector<nanoseconds> time;
    std::vector<uint64_t> rows;  // rows each step handed on; the sink hands on none
};

// One worker's locals for a pipeline, and one batch per step.
struct Worker {
    std::unique_ptr<LocalState> source;
    std::vector<std::unique_ptr<LocalState>> operators;
    std::unique_ptr<LocalState> sink;
    std::vector<Batch> batches;            // batches[0] from the source, batches[i + 1] from operator i
    std::unique_ptr<StepProfile> profile;  // only when analyze is on
};

// A fresh local per step for this worker, and empty batches shaped on first use.
Worker MakeWorker(const Pipeline& p, const QueryContext& ctx, unsigned worker, bool profile) {
    Worker w;
    w.source = p.source->MakeWorkerLocal(ctx, worker);
    for (const Operator* op : p.operators) w.operators.push_back(op->MakeWorkerLocal(ctx, worker));
    w.sink = p.sink->MakeWorkerLocal(ctx, worker);
    w.batches.resize(p.operators.size() + 1);
    if (profile) {
        w.profile = std::make_unique<StepProfile>();
        w.profile->time.assign(p.operators.size() + 2, nanoseconds{0});
        w.profile->rows.assign(p.operators.size() + 1, 0);
    }
    return w;
}

// Pushes batches[step] through operator `step` and everything after it.
// With a profile, each call's own time and output rows go to its step.
void Push(Pipeline& p, Worker& w, size_t step) {
    Batch& in = w.batches[step];
    if (in.Active() == 0) return;
    StepProfile* profile = w.profile.get();
    
    if (step == p.operators.size()) {
        if (profile == nullptr) {
            p.sink->Consume(in, *w.sink);
        } else {
            const auto start = Clock::now();
            p.sink->Consume(in, *w.sink);
            profile->time[step + 1] += Clock::now() - start;
        }
        return;
    }
    Batch& out = w.batches[step + 1];
    const Operator& op = *p.operators[step];
    LocalState* local = w.operators[step].get();
    for (;;) {
        OpResult result;
        if (profile == nullptr) {
            result = op.Execute(in, out, local);
        } else {
            const auto start = Clock::now();
            result = op.Execute(in, out, local);
            profile->time[step + 1] += Clock::now() - start;
            profile->rows[step + 1] += out.Active();
        }
        Push(p, w, step + 1);
        if (result == OpResult::NEED_INPUT || p.sink->Done()) return;
    }
}

// Starts a morsel, timed into the source's step when profiling.
void StartMorsel(const Pipeline& p, Worker& w, size_t morsel) {
    if (w.profile == nullptr) {
        p.source->StartMorsel(morsel, *w.source);
        return;
    }
    const auto start = Clock::now();
    p.source->StartMorsel(morsel, *w.source);
    w.profile->time[0] += Clock::now() - start;
}

// The source's next batch, timed and counted when profiling.
bool NextBatch(const Pipeline& p, Worker& w) {
    if (w.profile == nullptr) return p.source->NextBatch(*w.source, w.batches[0]);
    const auto start = Clock::now();
    const bool more = p.source->NextBatch(*w.source, w.batches[0]);
    w.profile->time[0] += Clock::now() - start;
    if (more) w.profile->rows[0] += w.batches[0].Active();
    return more;
}

// Takes morsels until none are left, the sink is done, or another worker failed.
// started counts this worker's morsels.
void RunMorsels(Pipeline& p, Worker& w, std::atomic<size_t>& next, const std::atomic<bool>& stop, size_t morsels, uint64_t& started) {
    while (!stop.load(std::memory_order_relaxed) && !p.sink->Done()) {
        const size_t m = next.fetch_add(1, std::memory_order_relaxed);
        if (m >= morsels) return;
        ++started;
        StartMorsel(p, w, m);
        while (!stop.load(std::memory_order_relaxed) && !p.sink->Done() && NextBatch(p, w)) Push(p, w, 0);
    }
}

// Lets operators first.. flush what they buffered, timed into their steps when profiling.
// Skipped after a failure or once the sink is done, since nothing reads it then.
void FinishOperators(const Pipeline& p, Worker& w, size_t first, const std::atomic<bool>& stop) {
    if (stop.load(std::memory_order_relaxed) || p.sink->Done()) return;
    for (size_t i = first; i < p.operators.size(); ++i) {
        if (w.profile == nullptr) {
            p.operators[i]->FinishMorsels(w.operators[i].get());
            continue;
        }
        const auto start = Clock::now();
        p.operators[i]->FinishMorsels(w.operators[i].get());
        w.profile->time[i + 1] += Clock::now() - start;
    }
}

// Operator k's next spilled batch, timed and counted into its step when profiling.
bool NextSpilledBatch(const Pipeline& p, Worker& w, size_t k) {
    Batch& out = w.batches[k + 1];
    if (w.profile == nullptr) return p.operators[k]->NextSpilledBatch(w.operators[k].get(), out);
    const auto start = Clock::now();
    const bool more = p.operators[k]->NextSpilledBatch(w.operators[k].get(), out);
    w.profile->time[k + 1] += Clock::now() - start;
    if (more) w.profile->rows[k + 1] += out.Active();
    return more;
}

// Takes operator k's spilled partitions until none are left, the sink is done, or another
// worker failed. Each batch goes through the operators after k, as a morsel's would.
void RunSpilled(Pipeline& p, Worker& w, size_t k, std::atomic<size_t>& next, const std::atomic<bool>& stop, size_t parts) {
    const Operator& op = *p.operators[k];
    while (!stop.load(std::memory_order_relaxed) && !p.sink->Done()) {
        const size_t part = next.fetch_add(1, std::memory_order_relaxed);
        if (part >= parts) return;
        if (w.profile == nullptr) {
            op.StartSpilledPartition(part, w.operators[k].get());
        } else {
            const auto start = Clock::now();
            op.StartSpilledPartition(part, w.operators[k].get());
            w.profile->time[k + 1] += Clock::now() - start;
        }
        while (!stop.load(std::memory_order_relaxed) && !p.sink->Done() && NextSpilledBatch(p, w, k)) Push(p, w, k + 1);
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

// Adds to a node's own time; a null node or negative time adds nothing.
void ChargeTime(OperatorStatsMap& stats, const PlanNode* node, nanoseconds time) {
    if (node != nullptr) stats[node].time += std::max(time, nanoseconds{0});
}

// Sums the workers' steps in worker order and charges them to the steps' plan nodes.
// The source's node gets the rows read; fused nodes get the rows handed up.
// Under a LIMIT that stops early, workers read ahead, so rows can pass tuple mode's.
void ChargeWorkers(const Pipeline& p, const std::vector<Worker>& workers, OperatorStatsMap& stats) {
    const size_t n = p.operators.size();
    std::vector<nanoseconds> time(n + 2, nanoseconds{0});
    std::vector<uint64_t> rows(n + 1, 0);
    for (const Worker& w : workers) {
        for (size_t i = 0; i < time.size(); ++i) time[i] += w.profile->time[i];
        for (size_t i = 0; i < rows.size(); ++i) rows[i] += w.profile->rows[i];
    }
    if (p.source_node != nullptr) {
        ChargeTime(stats, p.source_node, time[0]);
        stats[p.source_node].rows += p.source->RowsRead().value_or(rows[0]);
    }
    for (const PlanNode* node : p.source_fused) stats[node].rows += rows[0];
    assert((p.operator_nodes.empty() || p.operator_nodes.size() == n) && "one plan node per operator, or none");
    if (p.operator_nodes.size() == n) {
        for (size_t i = 0; i < n; ++i) {
            if (p.operator_nodes[i] == nullptr) continue;
            ChargeTime(stats, p.operator_nodes[i], time[i + 1]);
            stats[p.operator_nodes[i]].rows += rows[i + 1];
        }
    }
    ChargeTime(stats, p.sink_node, time[n + 1]);
}

// Records how long a scope took into *out, when out is set.
class BusyTimer {
public:
    explicit BusyTimer(nanoseconds* out) : out_(out) {
        if (out_ != nullptr) start_ = Clock::now();
    }
    ~BusyTimer() {
        if (out_ != nullptr) *out_ = Clock::now() - start_;
    }
    BusyTimer(const BusyTimer&) = delete;
    BusyTimer& operator=(const BusyTimer&) = delete;

private:
    nanoseconds* out_;
    Clock::time_point start_;
};

// The time now when profiling, so analyze off reads no clock.
Clock::time_point NowIf(bool on) {
    return on ? Clock::now() : Clock::time_point{};
}

// One worker without a pool, a second thread or a second piece of work.
unsigned WorkersFor(const QueryContext& ctx, size_t count) {
    if (ctx.pool == nullptr || ctx.threads <= 1 || count <= 1) return 1;
    return static_cast<unsigned>(std::min<size_t>({ctx.threads, count, ctx.pool->Size()}));
}

// Runs fn(w) on every worker: on the calling thread for one, else on the pool.
// The first worker to throw stops the others; the pool rethrows it.
void OnWorkers(QueryContext& ctx, unsigned workers, std::atomic<bool>& stop, const std::function<void(unsigned)>& fn) {
    if (workers == 1) {
        fn(0);
        return;
    }
    
    ctx.pool->Run(workers, [&](unsigned w) {
        try {
            fn(w);
        } catch (...) {
            stop.store(true, std::memory_order_relaxed);
            throw;
        }
    });
}

}  // namespace

// Operators without per-worker state need nothing here.
std::unique_ptr<LocalState> Operator::MakeLocal() const {
    return nullptr;
}

// Most locals need neither the query nor the worker.
std::unique_ptr<LocalState> Source::MakeWorkerLocal(const QueryContext&, unsigned) const {
    return MakeLocal();
}

// Most locals need neither the query nor the worker.
std::unique_ptr<LocalState> Operator::MakeWorkerLocal(const QueryContext&, unsigned) const {
    return MakeLocal();
}

// Most locals need neither the query nor the worker.
std::unique_ptr<LocalState> Sink::MakeWorkerLocal(const QueryContext&, unsigned) const {
    return MakeLocal();
}

// Operators that never spill buffer nothing.
void Operator::FinishMorsels(LocalState*) const {}

// Operators that never spill have no partitions waiting.
size_t Operator::SpilledPartitionCount() const {
    return 0;
}

// Never called on an operator with no spilled partitions.
void Operator::StartSpilledPartition(size_t, LocalState*) const {}

// No spilled partitions, so no batches.
bool Operator::NextSpilledBatch(LocalState*, Batch&) const {
    return false;
}

// Most operators hold nothing past their pipeline.
void Operator::PipelineDone() {}

// Batches are reused across morsels, so this rebuilds at most a few times per worker.
void ShapeBatch(Batch& batch, const std::vector<VType>& types) {
    bool same = batch.slots.size() == types.size();
    for (size_t i = 0; same && i < types.size(); ++i) same = batch.slots[i].Type() == types[i];
    if (same) return;
    batch.slots.clear();
    batch.slots.reserve(types.size());
    for (const VType& t : types) batch.slots.emplace_back(t);
}

// Morsels, then each spilled operator's partitions, shared by atomic counters; a phase
// starts once every worker finishes the last. Combine goes in worker order, as before.
void RunPipeline(Pipeline& p, QueryContext& ctx) {
    OperatorStatsMap* stats = ctx.op_stats;
    auto start = NowIf(stats != nullptr);
    p.source->Prepare(ctx);
    if (stats != nullptr) ChargeTime(*stats, p.source_node, Clock::now() - start);

    // Spilled partitions are known up front, so the workers are sized once for both.
    const size_t morsels = p.source->MorselCount();
    std::vector<size_t> spilled(p.operators.size(), 0);
    size_t most = morsels;
    for (size_t k = 0; k < p.operators.size(); ++k) {
        spilled[k] = p.operators[k]->SpilledPartitionCount();
        most = std::max(most, spilled[k]);
    }
    const unsigned workers = WorkersFor(ctx, most);
    std::vector<Worker> locals;
    locals.reserve(workers);
    for (unsigned w = 0; w < workers; ++w) locals.push_back(MakeWorker(p, ctx, w, stats != nullptr));

    std::atomic<bool> stop{false};
    std::vector<uint64_t> started(workers, 0);  // each worker writes only its own entry
    {
        std::optional<ConcurrentScope> concurrent;
        if (workers > 1) concurrent.emplace(ctx.bpm, workers);
        std::atomic<size_t> next{0};
        OnWorkers(ctx, workers, stop, [&](unsigned w) {
            RunMorsels(p, locals[w], next, stop, morsels, started[w]);
            FinishOperators(p, locals[w], 0, stop);
        });
        for (size_t k = 0; k < p.operators.size(); ++k) {
            if (spilled[k] == 0) continue;
            std::atomic<size_t> next_part{0};
            OnWorkers(ctx, workers, stop, [&](unsigned w) {
                RunSpilled(p, locals[w], k, next_part, stop, spilled[k]);
                FinishOperators(p, locals[w], k + 1, stop);
            });
        }
    }

    for (uint64_t n : started) ctx.morsels += n;
    if (stats != nullptr) ChargeWorkers(p, locals, *stats);

    start = NowIf(stats != nullptr);
    for (Worker& w : locals) p.sink->Combine(std::move(w.sink));
    if (stats != nullptr) ChargeTime(*stats, p.sink_node, Clock::now() - start);

    // Pool tasks inside Finalize add their thread time beyond the caller's wall time.
    ctx.task_extra = nanoseconds{0};
    start = NowIf(stats != nullptr);
    p.sink->Finalize(ctx);
    if (stats != nullptr) ChargeTime(*stats, p.sink_node, Clock::now() - start + ctx.task_extra);
    for (Operator* op : p.operators) op->PipelineDone();
}

// Tasks that don't need the worker index.
void ForEachTask(QueryContext& ctx, size_t count, const std::function<void(size_t)>& task) {
    ForEachTask(ctx, count, [&task](size_t i, unsigned) { task(i); });
}

// Tasks are taken from a shared counter; a failed task stops the others taking more.
// With analyze on, every worker's busy time is summed into ctx.task_extra.
void ForEachTask(QueryContext& ctx, size_t count, const std::function<void(size_t, unsigned)>& task) {
    const unsigned workers = WorkersFor(ctx, count);
    if (workers == 1) {
        for (size_t i = 0; i < count; ++i) task(i, 0);
        return;
    }
    const bool timed = ctx.op_stats != nullptr;
    std::vector<nanoseconds> busy(timed ? workers : 0, nanoseconds{0});  // each worker writes only its own entry
    const auto start = NowIf(timed);
    std::atomic<size_t> next{0};
    std::atomic<bool> stop{false};
    ctx.pool->Run(workers, [&](unsigned w) {
        const BusyTimer timer(timed ? &busy[w] : nullptr);
        try {
            while (!stop.load(std::memory_order_relaxed)) {
                const size_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= count) return;
                task(i, w);
            }
        } catch (...) {
            stop.store(true, std::memory_order_relaxed);
            throw;
        }
    });
    if (!timed) return;
    nanoseconds total{0};
    for (nanoseconds b : busy) total += b;
    ctx.task_extra += total - (Clock::now() - start);
}

}  // namespace shilmandb::parallel
