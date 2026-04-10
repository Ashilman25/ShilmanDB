#include "executor/sort_executor.hpp"
#include "executor/expression_evaluator.hpp"
#include "planner/plan_node.hpp"
#include <algorithm>
#include <cassert>

namespace shilmandb {

SortExecutor::SortExecutor(const PlanNode* plan, ExecutorContext* ctx, std::unique_ptr<Executor> child) : Executor(plan, ctx), child_(std::move(child)) {}

void SortExecutor::Init() {
    child_->Init();

    sorted_tuples_.clear();
    cursor_ = 0;

    Tuple tuple;
    while (child_->Next(&tuple)) {
        sorted_tuples_.push_back(std::move(tuple));
    }

    const auto* sort_plan = static_cast<const SortPlanNode*>(plan_);
    const auto& order_by = sort_plan->order_by;
    const auto& schema = child_->GetOutputSchema();

    std::stable_sort(sorted_tuples_.begin(), sorted_tuples_.end(), [&order_by, &schema](const Tuple& a, const Tuple& b) {
        for (const auto& item : order_by) {
            auto val_a = EvaluateExpression(item.expr.get(), a, schema);
            auto val_b = EvaluateExpression(item.expr.get(), b, schema);

            if (val_a < val_b) return item.ascending;
            if (val_a > val_b) return !item.ascending;
        }
        return false;  // All keys equal
    });
    initialized_ = true;
}

bool SortExecutor::Next(Tuple* tuple) {
    assert(initialized_ && "Next() called before Init()");
    if (cursor_ < sorted_tuples_.size()) {
        *tuple = std::move(sorted_tuples_[cursor_++]);
        return true;
    }
    return false;
}

void SortExecutor::Close() {
    child_->Close();
    sorted_tuples_.clear();
    cursor_ = 0;
    initialized_ = false;
}

}  // namespace shilmandb
