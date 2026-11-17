#include "executor/index_scan_executor.hpp"
#include "executor/columnar_scan_executor.hpp"
#include "storage/columnar/field_parser.hpp"
#include "planner/plan_node.hpp"
#include "common/exception.hpp"
#include <cassert>

namespace shilmandb {

void IndexScanExecutor::Init() {
    const auto* scan_plan = static_cast<const IndexScanPlanNode*>(plan_);

    auto* index_info = ctx_->catalog->GetIndex(scan_plan->index_name);
    if (!index_info) {
        throw DatabaseException("IndexScanExecutor: index not found: " + scan_plan->index_name);
    }

    auto* table_info = ctx_->catalog->GetTable(scan_plan->table_name);
    if (!table_info) {
        throw DatabaseException("IndexScanExecutor: table not found: " + scan_plan->table_name);
    }

    table_ = table_info->table.get();
    schema_ = &table_info->schema;
    high_key_ = scan_plan->high_key;

    columnar_ = nullptr;
    read_columns_.clear();
    read_names_.clear();
    scratch_.clear();
    defaults_.clear();
    if (table_info->storage == StorageKind::COLUMNAR) {
        const Schema& plan_schema = plan_->output_schema;
        if (plan_schema.GetColumnCount() != schema_->GetColumnCount()) {
            throw DatabaseException("IndexScanExecutor: the plan's columns don't match table " + scan_plan->table_name);
        }
        for (uint32_t c = 0; c < plan_schema.GetColumnCount(); ++c) {
            if (plan_schema.GetColumn(c).type != schema_->GetColumn(c).type) {
                throw DatabaseException("IndexScanExecutor: the plan's column " + std::to_string(c) + " has a different type than table " + scan_plan->table_name);
            }
        }
        for (const auto& col : schema_->GetColumns()) defaults_.push_back(Value::DefaultFor(col.type));
        columnar_ = table_info->columnar.get();
        read_columns_ = ColumnarReadColumns(*table_info, scan_plan->read_columns);
        for (uint32_t c : read_columns_) {
            read_names_.push_back(columnar_->Meta().columns[c].name);
            scratch_.emplace_back(columnar_->Meta().columns[c].type);
        }
    }

    if (scan_plan->low_key.has_value()) {
        iter_.emplace(index_info->index->Begin(scan_plan->low_key.value()));
    } else {
        iter_.emplace(index_info->index->Begin());
    }

    end_.emplace(index_info->index->End());
}

bool IndexScanExecutor::Next(Tuple* tuple) {
    assert(iter_.has_value() && "Next() called before Init()");

    while (!iter_->IsEnd()) {
        auto [key, rid] = **iter_;

        if (high_key_.has_value() && key > high_key_.value()) {
            return false;
        }

        ++(*iter_);

        if (columnar_ != nullptr) {
            *tuple = FetchColumnar(rid);
            return true;
        }
        if (table_->GetTuple(rid, tuple, *schema_)) {
            return true;
        }
    }
    return false;
}

Tuple IndexScanExecutor::FetchColumnar(const RID& rid) {
    const RowPosition pos = FromRid(rid);
    std::vector<Value> values = defaults_;

    for (size_t k = 0; k < read_columns_.size(); ++k) {
        ColumnVector& one = scratch_[k];
        one.Clear();
        columnar_->ReadValue(pos, read_columns_[k], one);
        if (one.IsNull(0)) throw DatabaseException(ColumnarNullError(read_names_[k]));
        values[read_columns_[k]] = ToExecutorValue(one, 0);
    }
    return Tuple(std::move(values), *schema_);
}

void IndexScanExecutor::Close() {}

}  // namespace shilmandb
