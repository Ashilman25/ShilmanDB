#include "executor/nested_loop_join_executor.hpp"
#include "executor/expression_evaluator.hpp"
#include "planner/plan_node.hpp"

namespace shilmandb {

void NestedLoopJoinExecutor::Init() {
    right_child_->Init();
    right_tuples_.clear();

    Tuple t;
    while (right_child_->Next(&t)) {
        right_tuples_.push_back(t);
    }

    left_child_->Init();
    left_done_ = !left_child_->Next(&current_left_);
    right_idx_ = 0;
}

bool NestedLoopJoinExecutor::Next(Tuple* tuple) {
    const auto* nlj_plan = static_cast<const NestedLoopJoinPlanNode*>(plan_);
    const auto& left_schema = left_child_->GetOutputSchema();
    const auto& right_schema = right_child_->GetOutputSchema();
    const auto& output_schema = plan_->output_schema;

    while (!left_done_) {
        while (right_idx_ < right_tuples_.size()) {
            auto combined = CombineTuples(current_left_, left_schema, right_tuples_[right_idx_], right_schema, output_schema);
            ++right_idx_;

            auto result = EvaluateExpression(nlj_plan->predicate.get(), combined, output_schema);
            if (IsTruthy(result)) {
                *tuple = std::move(combined);
                return true;
            }
        }

        //go left, reset right
        right_idx_ = 0;
        left_done_ = !left_child_->Next(&current_left_);
    }

    return false;
}

void NestedLoopJoinExecutor::Close() {
    left_child_->Close();
    right_child_->Close();
}

}  // namespace shilmandb
