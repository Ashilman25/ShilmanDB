#pragma once
#include "executor/parallel/pipeline.hpp"
#include "executor/parallel/result_sink.hpp"
#include "planner/plan_node.hpp"
#include <memory>
#include <vector>

namespace shilmandb::parallel {

// A compiled query: the parts it owns and the pipelines that wire them together.
struct PhysicalPlan {
    std::vector<std::unique_ptr<Source>> sources;
    std::vector<std::unique_ptr<Operator>> operators;
    std::vector<std::unique_ptr<Sink>> sinks;
    std::vector<Pipeline> pipelines;  // dependency order; the last ends in the result sink
    ResultSink* result{nullptr};
};

// Throws DatabaseException("the parallel executor reads only columnar tables: <name>")
// for a row-store table, checked first, and
// "the parallel executor can't run <TYPE> yet" for a node it doesn't handle.
[[nodiscard]] PhysicalPlan CompilePlan(const PlanNode& root, QueryContext& ctx);

}  // namespace shilmandb::parallel
