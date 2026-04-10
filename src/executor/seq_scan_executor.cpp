#include "executor/seq_scan_executor.hpp"
#include "planner/plan_node.hpp"
#include "common/exception.hpp"
#include <cassert>

namespace shilmandb {

void SeqScanExecutor::Init() {
    const auto* scan_plan = static_cast<const SeqScanPlanNode*>(plan_);
    auto* table_info = ctx_->catalog->GetTable(scan_plan->table_name);
    
    if (!table_info) {
        throw DatabaseException("SeqScanExecutor: table not found: " + scan_plan->table_name);
    }

    iter_.emplace(table_info->table->Begin(table_info->schema));
    end_.emplace(table_info->table->End());
}

bool SeqScanExecutor::Next(Tuple* tuple) {
    assert(iter_.has_value() && "Next() called before Init()");
    if (*iter_ != *end_) {
        *tuple = **iter_;
        ++(*iter_);
        return true;
    }
    return false;
}

void SeqScanExecutor::Close() {}

}  // namespace shilmandb
