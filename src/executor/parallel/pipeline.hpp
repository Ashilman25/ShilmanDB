#pragma once
#include "executor/executor_context.hpp"
#include "executor/parallel/batch.hpp"
#include "executor/parallel/vector.hpp"
#include <cstddef>
#include <memory>
#include <vector>

namespace shilmandb::parallel {

class WorkerPool;

// What every part of a query can reach while it runs.
struct QueryContext {
    BufferPoolManager* bpm;
    Catalog* catalog;
    unsigned threads{1};                  // already resolved
    WorkerPool* pool{nullptr};
    OperatorStatsMap* op_stats{nullptr};  // set when analyze is on
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

    // Starts reading one morsel into this worker's state.
    virtual void StartMorsel(size_t morsel, LocalState& local) const = 0;

    // Fills out with the next rows of the morsel; false when it's done.
    virtual bool NextBatch(LocalState& local, Batch& out) const = 0;
};

// A streaming step between a source and a sink, shared by every worker.
// out may arrive empty or holding vectors from an earlier batch; shape it with ShapeBatch.
class Operator {
public:
    virtual ~Operator() = default;

    // Per-worker state; none by default.
    [[nodiscard]] virtual std::unique_ptr<LocalState> MakeLocal() const;

    // HAVE_MORE_OUTPUT: call again with the same input after `out` is passed on.
    // An operator that answers NEED_INPUT may swap in and out instead of copying.
    // Const, since workers share the operator; their own state goes in local.
    virtual OpResult Execute(Batch& in, Batch& out, LocalState* local) const = 0;
};

// Where a pipeline's rows end up: each worker fills its own local, then they're combined.
class Sink {
public:
    virtual ~Sink() = default;

    [[nodiscard]] virtual std::unique_ptr<LocalState> MakeLocal() const = 0;

    // Workers call this at the same time, so it writes only their local.
    virtual void Consume(Batch& in, LocalState& local) const = 0;

    // Once per worker, on the calling thread, after the last morsel.
    virtual void Combine(std::unique_ptr<LocalState> local) = 0;

    // On the calling thread; may run parallel work.
    virtual void Finalize(QueryContext& ctx) = 0;

    // True once a limit is reached, so workers stop taking morsels.
    [[nodiscard]] virtual bool Done() const { return false; }
};

struct Pipeline {
    Source* source;
    std::vector<Operator*> operators;  // in the order rows pass through them
    Sink* sink;
    std::vector<size_t> depends_on;    // pipelines whose sinks must be finalized first
};

// Gives batch one slot per type, rebuilding the slots only when they differ.
void ShapeBatch(Batch& batch, const std::vector<VType>& types);

// Runs one pipeline: every morsel through the operators into the sink, then Combine and Finalize.
void RunPipeline(Pipeline& p, QueryContext& ctx);

}  // namespace shilmandb::parallel
