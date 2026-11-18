#include "executor/parallel/parallel_executor.hpp"
#include "executor/parallel/compiler.hpp"
#include <algorithm>
#include <thread>

namespace shilmandb::parallel {

// Pipelines are stored in dependency order, so running them in turn is enough.
std::vector<Tuple> RunParallelQuery(const PlanNode& root, QueryContext& ctx) {
    PhysicalPlan plan = CompilePlan(root, ctx);
    for (Pipeline& p : plan.pipelines) RunPipeline(p, ctx);
    return plan.result->TakeTuples();
}

// hardware_concurrency may report 0 when it can't tell.
unsigned ResolveThreads(unsigned requested) {
    const unsigned threads = requested != 0 ? requested : std::max(1u, std::thread::hardware_concurrency());
    return std::min(threads, kMaxThreads);
}

}  // namespace shilmandb::parallel
