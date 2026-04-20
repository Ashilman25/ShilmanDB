#pragma once

#include "executor/executor.hpp"
#include "executor/executor_context.hpp"
#include "planner/plan_node.hpp"
#include <memory>

namespace shilmandb {

class ExecutorFactory {
public:
    [[nodiscard]] static std::unique_ptr<Executor> CreateExecutor(const PlanNode* plan, ExecutorContext* ctx, ExecutionMode mode = ExecutionMode::TUPLE);
};

}  // namespace shilmandb
