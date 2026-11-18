#pragma once
#include "executor/parallel/pipeline.hpp"
#include "planner/plan_node.hpp"
#include "types/tuple.hpp"
#include <vector>

namespace shilmandb::parallel {

struct PhysicalPlan;

// The most worker threads one query may use.
inline constexpr unsigned kMaxThreads = 256;

// Compiles the plan, runs its pipelines in order and returns the result rows.
// Any exception reaches the caller unchanged, with no page left pinned.
[[nodiscard]] std::vector<Tuple> RunParallelQuery(const PlanNode& root, QueryContext& ctx);

// Runs a compiled plan's pipelines in list order and returns the result rows.
// Throws DatabaseException, before running any, if one depends on a pipeline not before it.
[[nodiscard]] std::vector<Tuple> RunPhysicalPlan(PhysicalPlan& plan, QueryContext& ctx);

// 0 means one thread per core (at least 1); never more than kMaxThreads.
[[nodiscard]] unsigned ResolveThreads(unsigned requested);

}  // namespace shilmandb::parallel
