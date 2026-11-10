#include "executor/vectorized_filter_executor.hpp"

#include "executor/expression_evaluator.hpp"
#include "planner/plan_node.hpp"

#include <cassert>
#include <vector>

namespace shilmandb {

VectorizedFilterExecutor::VectorizedFilterExecutor(const PlanNode* plan, ExecutorContext* ctx, std::unique_ptr<Executor> child) : Executor(plan, ctx),
      predicate_(static_cast<const FilterPlanNode*>(plan)->predicate.get()),
      child_(std::move(child)),
      buffer_(plan->output_schema) {}

void VectorizedFilterExecutor::Init() {
    child_->Init();
    buffer_.Reset();
    buffer_cursor_ = 0;
    initialized_ = true;
}

bool VectorizedFilterExecutor::NextBatch(DataChunk* chunk) {
    assert(initialized_ && "NextBatch() called before Init()");
    while (true) {
        if (!child_->NextBatch(chunk)) return false;

        const size_t n = chunk->size();
        std::vector<uint32_t> selected;
        selected.reserve(n);

        if (chunk->HasSelectionVector()) {
            const auto& prev = chunk->GetSelectionVector();
            for (size_t i = 0; i < n; ++i) {
                if (IsTruthy(EvaluateExpression(predicate_, ChunkRow{*chunk, prev[i]}))) {
                    selected.push_back(prev[i]);
                }
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                if (IsTruthy(EvaluateExpression(predicate_, ChunkRow{*chunk, i}))) {
                    selected.push_back(static_cast<uint32_t>(i));
                }
            }
        }

        if (!selected.empty()) {
            chunk->SetSelectionVector(std::move(selected));
            return true;
        }
       
    }
}

bool VectorizedFilterExecutor::Next(Tuple* tuple) {
    assert(initialized_ && "Next() called before Init()");
    while (buffer_cursor_ >= buffer_.size()) {
        buffer_cursor_ = 0;
        if (!NextBatch(&buffer_)) return false;
    }
    *tuple = buffer_.MaterializeTuple(buffer_cursor_++);
    return true;
}

void VectorizedFilterExecutor::Close() {
    child_->Close();
    buffer_.Reset();
    buffer_cursor_ = 0;
    initialized_ = false;
}

}  // namespace shilmandb
