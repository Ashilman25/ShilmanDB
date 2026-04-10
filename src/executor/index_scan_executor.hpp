#pragma once
#include "executor/executor.hpp"
#include "index/btree_index.hpp"
#include "storage/table_heap.hpp"
#include <optional>

namespace shilmandb {

class IndexScanExecutor : public Executor {
public:
    IndexScanExecutor(const PlanNode* plan, ExecutorContext* ctx) : Executor(plan, ctx) {}

    void Init() override;
    bool Next(Tuple* tuple) override;
    void Close() override;

private:
    std::optional<BTreeIndex::Iterator> iter_;
    std::optional<BTreeIndex::Iterator> end_;

    TableHeap* table_{nullptr};
    const Schema* schema_{nullptr};
    
    std::optional<Value> high_key_;
};

}  // namespace shilmandb
