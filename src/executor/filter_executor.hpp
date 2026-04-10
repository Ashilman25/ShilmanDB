#pragma once
#include "executor/executor.hpp"
#include <memory>

namespace shilmandb {

class FilterExecutor : public Executor {
public:
    FilterExecutor(const PlanNode* plan, ExecutorContext* ctx, std::unique_ptr<Executor> child) : Executor(plan, ctx), child_(std::move(child)) {}

    void Init() override;
    bool Next(Tuple* tuple) override;
    void Close() override;

private:
    std::unique_ptr<Executor> child_;
};

}  // namespace shilmandb
