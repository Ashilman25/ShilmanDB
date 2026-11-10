#include "executor/vectorized_seq_scan_executor.hpp"

#include "planner/plan_node.hpp"
#include "common/exception.hpp"

#include <cassert>

namespace shilmandb {

VectorizedSeqScanExecutor::VectorizedSeqScanExecutor(const PlanNode* plan, ExecutorContext* ctx) : Executor(plan, ctx), buffer_(plan->output_schema) {}

void VectorizedSeqScanExecutor::Init() {
    const auto* scan_plan = static_cast<const SeqScanPlanNode*>(plan_);
    auto* table_info = ctx_->catalog->GetTable(scan_plan->table_name);
    if (!table_info) {
        throw DatabaseException("VectorizedSeqScanExecutor: table not found: " + scan_plan->table_name);
    }
    table_ = table_info->table.get();
    pos_ = RID{table_->GetFirstPageId(), 0};

    buffer_.Reset();
    buffer_cursor_ = 0;
    initialized_ = true;
}

bool VectorizedSeqScanExecutor::NextBatch(DataChunk* chunk) {
    assert(initialized_ && "NextBatch() called before Init()");
    chunk->Reset();
    table_->ReadRows(&pos_, chunk->capacity(), [&](const char* data, uint32_t length) { chunk->AppendRow(data, length); });
    return chunk->size() > 0;
}

bool VectorizedSeqScanExecutor::Next(Tuple* tuple) {
    assert(initialized_ && "Next() called before Init()");
    while (buffer_cursor_ >= buffer_.size()) {
        buffer_cursor_ = 0;
        if (!NextBatch(&buffer_)) return false;
    }
    *tuple = buffer_.MaterializeTuple(buffer_cursor_++);
    return true;
}

void VectorizedSeqScanExecutor::Close() {
    table_ = nullptr;
    buffer_.Reset();
    buffer_cursor_ = 0;
    initialized_ = false;
}

}  // namespace shilmandb
