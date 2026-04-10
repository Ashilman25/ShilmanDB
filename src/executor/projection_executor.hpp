#pragma once
#include "executor/executor.hpp"
#include <memory>

namespace shilmandb {

class ProjectionExecutor : public Executor {
public:
    ProjectionExecutor(const PlanNode* plan, ExecutorContext* ctx, std::unique_ptr<Executor> child);

    void Init() override;
    bool Next(Tuple* tuple) override;
    void Close() override;

private:
    std::unique_ptr<Executor> child_;
    bool initialized_{false};
};

}  // namespace shilmandb
