#include "executor/columnar_scan_executor.hpp"

#include "catalog/catalog.hpp"
#include "common/exception.hpp"
#include "planner/cost_model.hpp"
#include "planner/plan_node.hpp"
#include "storage/columnar/field_parser.hpp"

#include <algorithm>
#include <cassert>
#include <numeric>

namespace shilmandb {

// Every column when nothing is listed; a listed column must exist.
std::vector<uint32_t> ColumnarReadColumns(const TableInfo& table, const std::optional<std::vector<uint32_t>>& listed) {
    const auto n_cols = static_cast<uint32_t>(table.columnar->Meta().columns.size());
    if (!listed) {
        std::vector<uint32_t> all(n_cols);
        std::iota(all.begin(), all.end(), 0u);
        return all;
    }
    for (uint32_t c : *listed) {
        if (c >= n_cols || c >= table.schema.GetColumnCount()) {
            throw DatabaseException(table.name + ": no column " + std::to_string(c) + " to read");
        }
    }
    return *listed;
}

std::string ColumnarNullError(const std::string& column) {
    return "column '" + column + "' holds NULLs, which queries can't read yet";
}

ColumnarScanExecutor::ColumnarScanExecutor(const PlanNode* plan, ExecutorContext* ctx) : Executor(plan, ctx), buffer_(plan->output_schema) {}

// Finds the table and the columns to read, and starts at group 0.
void ColumnarScanExecutor::Init() {
    const auto* scan_plan = static_cast<const SeqScanPlanNode*>(plan_);
    auto* table_info = ctx_->catalog->GetTable(scan_plan->table_name);
    if (table_info == nullptr || table_info->storage != StorageKind::COLUMNAR) {
        throw DatabaseException("ColumnarScanExecutor: columnar table not found: " + scan_plan->table_name);
    }
    const Schema& plan_schema = plan_->output_schema;
    const Schema& table_schema = table_info->schema;
    if (plan_schema.GetColumnCount() != table_schema.GetColumnCount()) {
        throw DatabaseException("ColumnarScanExecutor: the plan's columns don't match table " + scan_plan->table_name);
    }
    for (uint32_t c = 0; c < plan_schema.GetColumnCount(); ++c) {
        if (plan_schema.GetColumn(c).type != table_schema.GetColumn(c).type) {
            throw DatabaseException("ColumnarScanExecutor: the plan's column " + std::to_string(c) + " has a different type than table " + scan_plan->table_name);
        }
    }
    table_ = table_info->columnar.get();
    read_columns_ = ColumnarReadColumns(*table_info, scan_plan->read_columns);

    const auto& columns = table_->Meta().columns;
    read_names_.clear();
    vectors_.clear();
    for (uint32_t c : read_columns_) {
        read_names_.push_back(columns[c].name);
        vectors_.emplace_back(columns[c].type);
    }

    skip_conjuncts_.clear();
    for (const auto& conj : scan_plan->skip_conjuncts) skip_conjuncts_.push_back(conj.get());
    skipped_groups_ = 0;

    next_group_ = 0;
    group_rows_ = 0;
    group_cursor_ = 0;
    buffer_.Reset();
    buffer_cursor_ = 0;
    initialized_ = true;
}

// Decodes the next group that may match; skipped groups cost no page read.
bool ColumnarScanExecutor::LoadGroup() {
    while (next_group_ < table_->RowGroupCount() && !skip_conjuncts_.empty() && !CostModel::MayMatch(skip_conjuncts_, table_->GroupRanges(next_group_))) {
        ++skipped_groups_;
        ++next_group_;
    }
    if (next_group_ >= table_->RowGroupCount()) return false;

    // Refuse NULLs before reading, so a refused chunk costs no page read.
    const auto& chunks = table_->Meta().row_groups[next_group_].chunks;
    for (size_t k = 0; k < read_columns_.size(); ++k) {
        if (chunks[read_columns_[k]].null_count > 0) throw DatabaseException(ColumnarNullError(read_names_[k]));
    }
    for (size_t k = 0; k < read_columns_.size(); ++k) {
        vectors_[k].Clear();
        table_->ReadChunk(next_group_, read_columns_[k], vectors_[k]);
    }
    group_rows_ = table_->RowGroupRows(next_group_);
    group_cursor_ = 0;
    ++next_group_;
    return true;
}

// Hands out up to a chunk's capacity of rows from the current group.
bool ColumnarScanExecutor::NextBatch(DataChunk* chunk) {
    assert(initialized_ && "NextBatch() called before Init()");
    chunk->Reset();
    while (group_cursor_ >= group_rows_) {
        if (!LoadGroup()) return false;
    }

    // A batch never crosses a row group.
    const size_t n = std::min(chunk->capacity(), group_rows_ - group_cursor_);
    chunk->StartColumnFill(n, read_columns_);
    for (size_t k = 0; k < read_columns_.size(); ++k) {
        auto& out = chunk->GetMutableColumn(read_columns_[k]);
        const ColumnVector& in = vectors_[k];
        for (size_t i = 0; i < n; ++i) out[i] = ToExecutorValue(in, group_cursor_ + i);
    }
    group_cursor_ += n;
    return true;
}

// One row at a time from the internal chunk; unread columns get placeholders.
bool ColumnarScanExecutor::Next(Tuple* tuple) {
    assert(initialized_ && "Next() called before Init()");
    while (buffer_cursor_ >= buffer_.size()) {
        buffer_cursor_ = 0;
        if (!NextBatch(&buffer_)) return false;
    }
    *tuple = buffer_.MaterializeTuple(buffer_cursor_++);
    return true;
}

void ColumnarScanExecutor::Close() {
    table_ = nullptr;
    vectors_.clear();
    skip_conjuncts_.clear();
    buffer_.Reset();
    buffer_cursor_ = 0;
    initialized_ = false;
}

}  // namespace shilmandb
