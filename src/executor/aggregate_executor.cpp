#include "executor/aggregate_executor.hpp"
#include "executor/expression_evaluator.hpp"
#include "planner/plan_node.hpp"
#include "common/exception.hpp"
#include <cassert>

namespace shilmandb {

AggregateExecutor::AggregateExecutor(const PlanNode* plan, ExecutorContext* ctx, std::unique_ptr<Executor> child) : Executor(plan, ctx), child_(std::move(child)) {}

void AggregateExecutor::Init() {
    child_->Init();
    groups_.clear();

    const auto* agg_plan = static_cast<const AggregatePlanNode*>(plan_);
    const auto& group_by_exprs = agg_plan->group_by_exprs;
    const auto& agg_exprs = agg_plan->aggregate_exprs;
    const auto& agg_funcs = agg_plan->aggregate_funcs;
    const auto& child_schema = child_->GetOutputSchema();
    const auto num_aggs = agg_funcs.size();

    Tuple tuple;
    while (child_->Next(&tuple)) {
        // Build group key
        std::vector<Value> key;
        key.reserve(group_by_exprs.size());
        for (const auto& gb_expr : group_by_exprs) {
            key.push_back(EvaluateExpression(gb_expr.get(), tuple, child_schema));
        }

        // Look up or create group
        auto& states = groups_[key];
        if (states.empty()) {
            states.resize(num_aggs);
        }

        // Update each aggregate
        for (size_t i = 0; i < num_aggs; ++i) {
            auto& state = states[i];

            switch (agg_funcs[i]) {
                case Aggregate::Func::COUNT: {
                    state.count++;
                    break;
                }
                case Aggregate::Func::SUM:
                case Aggregate::Func::AVG: {
                    auto val = EvaluateExpression(agg_exprs[i].get(), tuple, child_schema);
                    state.sum += ValueToDouble(val);
                    state.count++;
                    break;
                }
                case Aggregate::Func::MIN: {
                    auto val = EvaluateExpression(agg_exprs[i].get(), tuple, child_schema);
                    if (!state.has_value || val < state.min_val) {
                        state.min_val = val;
                        state.has_value = true;
                    }
                    break;
                }
                case Aggregate::Func::MAX: {
                    auto val = EvaluateExpression(agg_exprs[i].get(), tuple, child_schema);
                    if (!state.has_value || val > state.max_val) {
                        state.max_val = val;
                        state.has_value = true;
                    }
                    break;
                }
                default:
                    throw DatabaseException("Unknown aggregate function");
            }
        }
    }


    if (groups_.empty() && group_by_exprs.empty()) {
        groups_[std::vector<Value>{}].resize(num_aggs);
    }

    group_iter_ = groups_.begin();
    initialized_ = true;
}

bool AggregateExecutor::Next(Tuple* tuple) {
    assert(initialized_ && "Next() called before Init()");

    if (group_iter_ == groups_.end()) {
        return false;
    }

    const auto* agg_plan = static_cast<const AggregatePlanNode*>(plan_);
    const auto& agg_funcs = agg_plan->aggregate_funcs;
    const auto& key = group_iter_->first;
    const auto& states = group_iter_->second;

    std::vector<Value> values;
    values.reserve(key.size() + states.size());

    // Group-by columns first
    for (const auto& v : key) {
        values.push_back(v);
    }

    // Aggregate result columns
    for (size_t i = 0; i < agg_funcs.size(); ++i) {
        const auto& state = states[i];

        switch (agg_funcs[i]) {
            case Aggregate::Func::COUNT:
                values.push_back(Value(static_cast<int32_t>(state.count)));
                break;
            case Aggregate::Func::SUM:
                values.push_back(Value(state.sum));
                break;
            case Aggregate::Func::AVG:
                values.push_back(Value(state.count > 0
                    ? state.sum / static_cast<double>(state.count)
                    : 0.0));
                break;
            case Aggregate::Func::MIN:
                values.push_back(state.has_value
                    ? state.min_val
                    : Value(static_cast<int32_t>(0)));
                break;
            case Aggregate::Func::MAX:
                values.push_back(state.has_value
                    ? state.max_val
                    : Value(static_cast<int32_t>(0)));
                break;
            default:
                throw DatabaseException("Unknown aggregate function");
        }
    }

    *tuple = Tuple(std::move(values), GetOutputSchema());
    ++group_iter_;
    return true;
}

void AggregateExecutor::Close() {
    child_->Close();
    groups_.clear();
    initialized_ = false;
}

}  // namespace shilmandb
