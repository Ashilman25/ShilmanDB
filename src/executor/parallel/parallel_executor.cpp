#include "executor/parallel/parallel_executor.hpp"
#include "common/exception.hpp"
#include "executor/parallel/compiler.hpp"
#include <algorithm>
#include <string>
#include <thread>

namespace shilmandb::parallel {

// Compiles, then runs.
std::vector<Tuple> RunParallelQuery(const PlanNode& root, QueryContext& ctx) {
    PhysicalPlan plan = CompilePlan(root, ctx);
    return RunPhysicalPlan(plan, ctx);
}

// Running in turn is enough once every dependency comes earlier in the list.
std::vector<Tuple> RunPhysicalPlan(PhysicalPlan& plan, QueryContext& ctx) {
    for (size_t i = 0; i < plan.pipelines.size(); ++i) {
        for (size_t d : plan.pipelines[i].depends_on) {
            if (d >= i) {
                throw DatabaseException("the parallel executor's pipeline " + std::to_string(i) + " depends on pipeline " + std::to_string(d) + ", which doesn't run before it");
            }
        }
    }
    for (Pipeline& p : plan.pipelines) RunPipeline(p, ctx);
    return plan.result->TakeTuples();
}

// hardware_concurrency may report 0 when it can't tell.
unsigned ResolveThreads(unsigned requested) {
    const unsigned threads = requested != 0 ? requested : std::max(1u, std::thread::hardware_concurrency());
    return std::min(threads, kMaxThreads);
}

}  // namespace shilmandb::parallel
