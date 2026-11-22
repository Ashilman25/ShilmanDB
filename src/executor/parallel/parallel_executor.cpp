#include "executor/parallel/parallel_executor.hpp"
#include "common/exception.hpp"
#include "executor/parallel/compiler.hpp"
#include <algorithm>
#include <chrono>
#include <string>
#include <thread>

namespace shilmandb::parallel {

namespace {

// Turns each node's own time into its subtree's total, children first; returns it.
std::chrono::nanoseconds AddSubtreeTimes(const PlanNode& node, OperatorStatsMap& stats) {
    std::chrono::nanoseconds children{0};
    for (const auto& child : node.children) children += AddSubtreeTimes(*child, stats);
    OperatorStats& own = stats[&node];
    own.time += children;
    return own.time;
}

}  // namespace

// Compiles, then runs. With analyze on, the own times become totals at the end,
// as tuple mode reports them.
std::vector<Tuple> RunParallelQuery(const PlanNode& root, QueryContext& ctx) {
    PhysicalPlan plan = CompilePlan(root, ctx);
    std::vector<Tuple> tuples = RunPhysicalPlan(plan, ctx);
    if (ctx.op_stats != nullptr) (void)AddSubtreeTimes(root, *ctx.op_stats);
    return tuples;
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
