#include "executor/index_scan_executor.hpp"
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

        if (table_->GetTuple(rid, tuple, *schema_)) {
            return true;
        }
    }
    return false;
}

void IndexScanExecutor::Close() {}

}  // namespace shilmandb
