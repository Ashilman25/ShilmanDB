#pragma once
#include "executor/executor.hpp"
#include "parser/ast.hpp"

#include <memory>
#include <vector>

namespace shilmandb {

class VectorizedProjectionExecutor : public Executor {
public:
    VectorizedProjectionExecutor(const PlanNode* plan, ExecutorContext* ctx,
                                 std::unique_ptr<Executor> child);

    void Init() override;
    bool Next(Tuple* tuple) override;
    bool NextBatch(DataChunk* chunk) override;
    void Close() override;

private:
    std::vector<const Expression*> expressions_;  // borrowed from ProjectionPlanNode
    std::unique_ptr<Executor> child_;
    bool initialized_{false};

    DataChunk child_chunk_;     // pulls one batch from child per call
    DataChunk buffer_;          // tuple-mode drain buffer
    size_t buffer_cursor_{0};
};

}  // namespace shilmandb
