#include "executor/vectorized_aggregate_executor.hpp"

#include "common/exception.hpp"
#include "executor/expression_evaluator.hpp"
#include "planner/column_usage.hpp"
#include "planner/plan_node.hpp"

#include <cassert>
#include <utility>
#include <vector>

namespace shilmandb {

VectorizedAggregateExecutor::VectorizedAggregateExecutor(const PlanNode* plan, ExecutorContext* ctx, std::unique_ptr<Executor> child) : Executor(plan, ctx), child_(std::move(child)), buffer_(plan->output_schema) {
    const auto* agg_plan = static_cast<const AggregatePlanNode*>(plan);

    group_by_exprs_.reserve(agg_plan->group_by_exprs.size());
    for (const auto& e : agg_plan->group_by_exprs) {
        group_by_exprs_.push_back(e.get());
    }

    aggregate_exprs_.reserve(agg_plan->aggregate_exprs.size());
    for (const auto& e : agg_plan->aggregate_exprs) {
        aggregate_exprs_.push_back(e.get());
    }

    aggregate_funcs_ = agg_plan->aggregate_funcs;
    cols_ = ColumnsReadBy(*plan);
}

void VectorizedAggregateExecutor::UpdateAggregates(std::vector<AggregateState>& states, const ChunkRow& row) {
    const auto num_aggs = aggregate_funcs_.size();
    for (size_t i = 0; i < num_aggs; ++i) {
        auto& state = states[i];
        switch (aggregate_funcs_[i]) {
            case Aggregate::Func::COUNT: {
                state.count++;
                break;
            }
            case Aggregate::Func::SUM:
            case Aggregate::Func::AVG: {
                auto val = EvaluateExpression(aggregate_exprs_[i], row);
                state.sum += ValueToDouble(val);
                state.count++;
                break;
            }
            case Aggregate::Func::MIN: {
                auto val = EvaluateExpression(aggregate_exprs_[i], row);
                if (!state.has_value || val < state.min_val) {
                    state.min_val = val;
                    state.has_value = true;
                }
                break;
            }
            case Aggregate::Func::MAX: {
                auto val = EvaluateExpression(aggregate_exprs_[i], row);
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

std::vector<Value> VectorizedAggregateExecutor::BuildResultValues(const std::vector<Value>& key, const std::vector<AggregateState>& states) const {
    std::vector<Value> values;
    values.reserve(key.size() + states.size());

    for (const auto& v : key) {
        values.push_back(v);
    }

    for (size_t i = 0; i < aggregate_funcs_.size(); ++i) {
        const auto& state = states[i];
        switch (aggregate_funcs_[i]) {
            case Aggregate::Func::COUNT:
                values.push_back(Value(static_cast<int32_t>(state.count)));
                break;
            case Aggregate::Func::SUM:
                values.push_back(Value(state.sum));
                break;
            case Aggregate::Func::AVG:
                values.push_back(Value(state.count > 0 ? state.sum / static_cast<double>(state.count) : 0.0));
                break;
            case Aggregate::Func::MIN:
                values.push_back(state.has_value ? state.min_val : Value(static_cast<int32_t>(0)));
                break;
            case Aggregate::Func::MAX:
                values.push_back(state.has_value ? state.max_val : Value(static_cast<int32_t>(0)));
                break;
            default:
                throw DatabaseException("Unknown aggregate function");
        }
    }

    return values;
}

void VectorizedAggregateExecutor::Init() {
    child_->Init();
    groups_.clear();

    const auto& child_schema = child_->GetOutputSchema();
    const auto num_aggs = aggregate_funcs_.size();

    DataChunk local_chunk(child_schema);
    while (child_->NextBatch(&local_chunk)) {
        local_chunk.EnsureDecoded(cols_);
        const size_t n_rows = local_chunk.size();
        const bool has_sel = local_chunk.HasSelectionVector();
        for (size_t i = 0; i < n_rows; ++i) {
            const size_t physical = has_sel ? local_chunk.GetSelectionVector()[i] : i;
            const ChunkRow row{local_chunk, physical};

            std::vector<Value> key;
            key.reserve(group_by_exprs_.size());
            for (const auto* gb_expr : group_by_exprs_) {
                key.push_back(EvaluateExpression(gb_expr, row));
            }

            auto& states = groups_[key];
            if (states.empty()) {
                states.resize(num_aggs);
            }
            UpdateAggregates(states, row);
        }
        local_chunk.Reset();
    }

    if (groups_.empty() && group_by_exprs_.empty()) {
        groups_[std::vector<Value>{}].resize(num_aggs);
    }

    emit_iter_ = groups_.begin();
    buffer_.Reset();
    buffer_cursor_ = 0;
    initialized_ = true;
}

bool VectorizedAggregateExecutor::NextBatch(DataChunk* chunk) {
    assert(initialized_ && "NextBatch() called before Init()");
    chunk->Reset();
    while (!chunk->IsFull() && emit_iter_ != groups_.end()) {
        chunk->AppendValues(BuildResultValues(emit_iter_->first, emit_iter_->second));
        ++emit_iter_;
    }
    return chunk->size() > 0;
}

bool VectorizedAggregateExecutor::Next(Tuple* tuple) {
    assert(initialized_ && "Next() called before Init()");
    while (buffer_cursor_ >= buffer_.size()) {
        buffer_cursor_ = 0;
        if (!NextBatch(&buffer_)) return false;
    }
    *tuple = buffer_.MaterializeTuple(buffer_cursor_++);
    return true;
}

void VectorizedAggregateExecutor::Close() {
    child_->Close();
    groups_.clear();
    buffer_.Reset();
    buffer_cursor_ = 0;
    initialized_ = false;
}

}  // namespace shilmandb
