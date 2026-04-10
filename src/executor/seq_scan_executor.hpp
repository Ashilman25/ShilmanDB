#pragma once
#include "executor/executor.hpp"
#include "storage/table_heap.hpp"
#include <optional>

namespace shilmandb {

class SeqScanExecutor : public Executor {
public:
    SeqScanExecutor(const PlanNode* plan, ExecutorContext* ctx) : Executor(plan, ctx) {}

    void Init() override;
    bool Next(Tuple* tuple) override;
    void Close() override;

private:
    std::optional<TableHeap::Iterator> iter_;
    std::optional<TableHeap::Iterator> end_;
};

}  // namespace shilmandb
