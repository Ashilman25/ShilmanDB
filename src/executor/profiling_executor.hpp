#pragma once
#include "executor/executor.hpp"

#include <memory>

namespace shilmandb {

// Wraps one executor and records the rows it emits and the time spent in it,
// including its children. Results go into ctx->op_stats under the wrapped plan node.
class ProfilingExecutor : public Executor {
public:
    ProfilingExecutor(const PlanNode* plan, ExecutorContext* ctx, std::unique_ptr<Executor> child);

    void Init() override;
    bool Next(Tuple* tuple) override;
    void Close() override;

private:
    std::unique_ptr<Executor> child_;
    OperatorStats* stats_;
    bool initialized_{false};
};

}  // namespace shilmandb
