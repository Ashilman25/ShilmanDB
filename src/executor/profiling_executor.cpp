#include "executor/profiling_executor.hpp"

#include <cassert>
#include <chrono>

namespace shilmandb {

namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] std::chrono::nanoseconds Since(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start);
}

}  // namespace

// Map nodes don't move when the map grows, so holding a pointer to our entry is safe.
ProfilingExecutor::ProfilingExecutor(const PlanNode* plan, ExecutorContext* ctx, std::unique_ptr<Executor> child) : Executor(plan, ctx), child_(std::move(child)) {
    assert(ctx->op_stats && "ProfilingExecutor needs an op_stats map");
    stats_ = &(*ctx->op_stats)[plan];
}

void ProfilingExecutor::Init() {
    const auto start = Clock::now();
    child_->Init();
    stats_->time += Since(start);
    initialized_ = true;
}

bool ProfilingExecutor::Next(Tuple* tuple) {
    assert(initialized_ && "Next() called before Init()");
    const auto start = Clock::now();
    const bool has_row = child_->Next(tuple);
    stats_->time += Since(start);
    if (has_row) ++stats_->rows;
    return has_row;
}

void ProfilingExecutor::Close() {
    const auto start = Clock::now();
    child_->Close();
    stats_->time += Since(start);
    initialized_ = false;
}

}  // namespace shilmandb
