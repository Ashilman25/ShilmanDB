#include "executor/filter_executor.hpp"
#include "executor/expression_evaluator.hpp"
#include "planner/plan_node.hpp"

namespace shilmandb {

void FilterExecutor::Init() {
    child_->Init();
}

bool FilterExecutor::Next(Tuple* tuple) {
    const auto* filter_plan = static_cast<const FilterPlanNode*>(plan_);

    while (child_->Next(tuple)) {
        auto result = EvaluateExpression(filter_plan->predicate.get(), *tuple, child_->GetOutputSchema());
        if (IsTruthy(result)) {
            return true;
        }
    }
    
    return false;
}

void FilterExecutor::Close() {
    child_->Close();
}

}  // namespace shilmandb
