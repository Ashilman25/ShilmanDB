#include "executor/vectorized_projection_executor.hpp"

#include "executor/expression_evaluator.hpp"
#include "planner/plan_node.hpp"

#include <cassert>
#include <utility>
#include <vector>

namespace shilmandb {

VectorizedProjectionExecutor::VectorizedProjectionExecutor(const PlanNode* plan, ExecutorContext* ctx, std::unique_ptr<Executor> child)
    : Executor(plan, ctx),
      child_(std::move(child)),
      child_chunk_(child_->GetOutputSchema()),
      buffer_(plan->output_schema) {
        
    const auto* proj_plan = static_cast<const ProjectionPlanNode*>(plan);
    expressions_.reserve(proj_plan->expressions.size());
    for (const auto& e : proj_plan->expressions) {
        expressions_.push_back(e.get());
    }
}

void VectorizedProjectionExecutor::Init() {
    child_->Init();
    child_chunk_.Reset();
    buffer_.Reset();
    buffer_cursor_ = 0;
    initialized_ = true;
}

bool VectorizedProjectionExecutor::NextBatch(DataChunk* chunk) {
    assert(initialized_ && "NextBatch() called before Init()");

    child_chunk_.Reset();
    if (!child_->NextBatch(&child_chunk_)) {
        return false;
    }

    chunk->Reset();  // clears any stale selection vector so AppendTuple is legal

    const auto& child_schema = child_chunk_.GetSchema();
    const auto& out_schema = GetOutputSchema();
    const size_t n_rows = child_chunk_.size();

    for (size_t i = 0; i < n_rows; ++i) {
        auto input_tuple = child_chunk_.MaterializeTuple(i);  // honors child's selection vector
        std::vector<Value> projected;
        projected.reserve(expressions_.size());
        for (const auto* expr : expressions_) {
            projected.push_back(EvaluateExpression(expr, input_tuple, child_schema));
        }
        chunk->AppendTuple(Tuple(std::move(projected), out_schema));
    }
    return chunk->size() > 0;
}

bool VectorizedProjectionExecutor::Next(Tuple* tuple) {
    assert(initialized_ && "Next() called before Init()");
    while (buffer_cursor_ >= buffer_.size()) {
        buffer_cursor_ = 0;
        if (!NextBatch(&buffer_)) return false;
    }
    *tuple = buffer_.MaterializeTuple(buffer_cursor_++);
    return true;
}

void VectorizedProjectionExecutor::Close() {
    child_->Close();
    child_chunk_.Reset();
    buffer_.Reset();
    buffer_cursor_ = 0;
    initialized_ = false;
}

}  // namespace shilmandb
