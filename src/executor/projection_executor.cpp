#include "executor/projection_executor.hpp"
#include "executor/expression_evaluator.hpp"
#include "planner/plan_node.hpp"
#include <cassert>

namespace shilmandb {

ProjectionExecutor::ProjectionExecutor(const PlanNode* plan, ExecutorContext* ctx, std::unique_ptr<Executor> child) : Executor(plan, ctx), child_(std::move(child)) {}

void ProjectionExecutor::Init() {
    child_->Init();
    initialized_ = true;
}

bool ProjectionExecutor::Next(Tuple* tuple) {
    assert(initialized_ && "Next() called before Init()");

    Tuple child_tuple;
    if (!child_->Next(&child_tuple)) {
        return false;
    }

    const auto* proj_plan = static_cast<const ProjectionPlanNode*>(plan_);
    const auto& child_schema = child_->GetOutputSchema();

    std::vector<Value> values;
    values.reserve(proj_plan->expressions.size());

    for (const auto& expr : proj_plan->expressions) {
        values.push_back(EvaluateExpression(expr.get(), child_tuple, child_schema));
    }

    *tuple = Tuple(std::move(values), GetOutputSchema());
    return true;
}

void ProjectionExecutor::Close() {
    child_->Close();
    initialized_ = false;
}

}  // namespace shilmandb
