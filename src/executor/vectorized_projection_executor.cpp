#include "executor/vectorized_projection_executor.hpp"

#include "executor/expression_evaluator.hpp"
#include "planner/column_usage.hpp"
#include "planner/plan_node.hpp"

#include <cassert>
#include <utility>
#include <vector>

namespace shilmandb {

VectorizedProjectionExecutor::VectorizedProjectionExecutor(const PlanNode* plan, ExecutorContext* ctx, std::unique_ptr<Executor> child)
    : Executor(plan, ctx),
      child_(std::move(child)),
      cols_(ColumnsReadBy(*plan)),
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

    chunk->Reset();  // clears any stale selection vector so AppendValues is legal

    child_chunk_.EnsureDecoded(cols_);
    const size_t n_rows = child_chunk_.size();
    const bool has_sel = child_chunk_.HasSelectionVector();

    for (size_t i = 0; i < n_rows; ++i) {
        const size_t physical = has_sel ? child_chunk_.GetSelectionVector()[i] : i;
        const ChunkRow row{child_chunk_, physical};
        std::vector<Value> projected;
        projected.reserve(expressions_.size());
        for (const auto* expr : expressions_) {
            projected.push_back(EvaluateExpression(expr, row));
        }
        chunk->AppendValues(std::move(projected));
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
