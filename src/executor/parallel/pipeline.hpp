#pragma once
#include "executor/executor_context.hpp"
#include "executor/parallel/batch.hpp"
#include "executor/parallel/vector.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace shilmandb {
struct PlanNode;
}

namespace shilmandb::parallel {

class WorkerPool;
class MemoryBudget;
struct SpillCounters;

// What every part of a query can reach while it runs.
struct QueryContext {
    BufferPoolManager* bpm;
    Catalog* catalog;
    unsigned threads{1};                  // already resolved
    WorkerPool* pool{nullptr};            // null when threads is 1
    OperatorStatsMap* op_stats{nullptr};  // set when analyze is on: own times while running, totals after
    uint64_t morsels{0};                  // morsels started so far, added up by RunPipeline
    std::chrono::nanoseconds task_extra{0};  // analyze: pool tasks' thread time beyond the caller's wall time
    MemoryBudget* budget{nullptr};           // breakers' memory; null means no limit
    SpillCounters* spill{nullptr};           // spill totals, when someone wants them
    const std::string* temp_dir{nullptr};    // where spill files go
    bool spill_everything{false};            // tests: every breaker spills everything it holds
};

// A sink's output that only non-const code can change, so a const worker call can't.
template <typename T>
class SharedOutput {
public:
    SharedOutput() : value_(std::make_shared<T>()) {}
    SharedOutput(const SharedOutput&) = delete;
    SharedOutput& operator=(const SharedOutput&) = delete;
    SharedOutput(SharedOutput&&) noexcept = default;
    SharedOutput& operator=(SharedOutput&&) noexcept = default;

    [[nodiscard]] const T& Get() const { return *value_; }
    [[nodiscard]] T& Mutable() { return *value_; }

    // A read-only handle for whoever reads the output later.
    [[nodiscard]] std::shared_ptr<const T> Reader() const { return value_; }

private:
    std::shared_ptr<T> value_;
};

// One worker's own state for a source, operator or sink.
struct LocalState {
    virtual ~LocalState() = default;
};

enum class OpResult { NEED_INPUT, HAVE_MORE_OUTPUT };

// Hands out a pipeline's rows in morsels; several workers may read different morsels at once.
// Worker calls are const: a worker's own state goes in its LocalState.
class Source {
public:
    virtual ~Source() = default;

    // Runs on the calling thread, before any worker.
    virtual void Prepare(QueryContext& ctx) = 0;

    // How many morsels Prepare found.
    [[nodiscard]] virtual size_t MorselCount() const = 0;

    [[nodiscard]] virtual std::unique_ptr<LocalState> MakeLocal() const = 0;

    // The local for one worker of this query; MakeLocal() unless a source needs more.
    // worker is below ctx.threads, 0 when serial.
    [[nodiscard]] virtual std::unique_ptr<LocalState> MakeWorkerLocal(const QueryContext& ctx, unsigned worker) const;

    // Starts reading one morsel into this worker's state.
    virtual void StartMorsel(size_t morsel, LocalState& local) const = 0;

    // Fills out with the next rows of the morsel; false when it's done.
    virtual bool NextBatch(LocalState& local, Batch& out) const = 0;

    // For analyze: rows read before a fused filter, once the pipeline is done.
    // None means the rows handed up.
    [[nodiscard]] virtual std::optional<uint64_t> RowsRead() const { return std::nullopt; }

    // After Combine, before the sink's Finalize: frees what the source still holds (a sort's rows).
    virtual void PipelineDone();
};

// A streaming step between a source and a sink, shared by every worker.
// out may arrive empty or holding vectors from an earlier batch; shape it with ShapeBatch.
class Operator {
public:
    virtual ~Operator() = default;

    // Per-worker state; none by default.
    [[nodiscard]] virtual std::unique_ptr<LocalState> MakeLocal() const;

    // The local for one worker of this query; MakeLocal() unless overridden.
    [[nodiscard]] virtual std::unique_ptr<LocalState> MakeWorkerLocal(const QueryContext& ctx, unsigned worker) const;

    // HAVE_MORE_OUTPUT: call again with the same input after `out` is passed on.
    // An operator that answers NEED_INPUT may swap in and out instead of copying.
    // Const, since workers share the operator; their own state goes in local.
    virtual OpResult Execute(Batch& in, Batch& out, LocalState* local) const = 0;

    // After this worker's last morsel (and after each
    // spill phase): flush what it buffered for later.
    virtual void FinishMorsels(LocalState* local) const;

    // Spilled partitions waiting after the source's morsels and earlier spill phases.
    // Known before the pipeline starts.
    [[nodiscard]] virtual size_t SpilledPartitionCount() const;

    // Joins partition p for this worker; NextSpilledBatch then hands out its output batches.
    // Strings in a batch stay valid until this worker's next NextSpilledBatch or StartSpilledPartition.
    virtual void StartSpilledPartition(size_t p, LocalState* local) const;
    virtual bool NextSpilledBatch(LocalState* local, Batch& out) const;

    // After Combine, before the sink's Finalize: frees what the operator still holds (a build table).
    virtual void PipelineDone();
};

// Where a pipeline's rows end up: each worker fills its own local, then they're combined.
class Sink {
public:
    virtual ~Sink() = default;

    [[nodiscard]] virtual std::unique_ptr<LocalState> MakeLocal() const = 0;

    // The local for one worker of this query; MakeLocal() unless a sink needs more.
    // worker is below ctx.threads, 0 when serial.
    [[nodiscard]] virtual std::unique_ptr<LocalState> MakeWorkerLocal(const QueryContext& ctx, unsigned worker) const;

    // Workers call this at the same time, so it writes only their local.
    virtual void Consume(Batch& in, LocalState& local) const = 0;

    // Once per worker, on the calling thread, after the last morsel.
    virtual void Combine(std::unique_ptr<LocalState> local) = 0;

    // On the calling thread; may run parallel work. The source and operators
    // have freed their state, so Consume must copy whatever is kept.
    virtual void Finalize(QueryContext& ctx) = 0;

    // True once a limit is reached, so workers stop taking morsels.
    [[nodiscard]] virtual bool Done() const { return false; }
};

struct Pipeline {
    Source* source;
    std::vector<Operator*> operators;  // in the order rows pass through them
    Sink* sink;
    std::vector<size_t> depends_on;    // pipelines whose sinks must be finalized first

    // For analyze: the plan node each step's time and rows go to; null charges nothing.
    // Nodes fused into the source (a filter, a limit) get the rows it hands up.
    const PlanNode* source_node{nullptr};
    std::vector<const PlanNode*> source_fused;
    std::vector<const PlanNode*> operator_nodes;  // empty, or one per operator
    const PlanNode* sink_node{nullptr};
};

// Gives batch one slot per type, rebuilding the slots only when they differ.
void ShapeBatch(Batch& batch, const std::vector<VType>& types);

// Runs one pipeline: morsels through the operators into the sink, then each spilled operator's
// partitions, then Combine, the source's and every operator's PipelineDone, then Finalize.
// Several workers run with a pool, 2+ threads and 2+ morsels or spilled partitions; the
// buffer pool is concurrent only while they run. With ctx.op_stats set, steps charge their nodes.
void RunPipeline(Pipeline& p, QueryContext& ctx);

// Runs task(0..count-1) once each, across the pool's workers when there is one.
// For Finalize work that never touches the buffer pool. Rethrows the first exception.
// With analyze on, adds workers' time past the caller's wall time to task_extra.
void ForEachTask(QueryContext& ctx, size_t count, const std::function<void(size_t)>& task);

// The same, also passing the worker running task i: below ctx.threads, 0 when serial.
// One worker index runs on one thread at a time,
// so state indexed by it needs no lock.
void ForEachTask(QueryContext& ctx, size_t count, const std::function<void(size_t, unsigned)>& task);

}  // namespace shilmandb::parallel
