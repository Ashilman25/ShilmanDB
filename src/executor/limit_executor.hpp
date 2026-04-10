#pragma once
#include "executor/executor.hpp"
#include <memory>

namespace shilmandb {

class LimitExecutor : public Executor {
public:
    LimitExecutor(const PlanNode* plan, ExecutorContext* ctx, std::unique_ptr<Executor> child);

    void Init() override;
    bool Next(Tuple* tuple) override;
    void Close() override;

private:
    std::unique_ptr<Executor> child_;
    int64_t limit_{0};
    int64_t emitted_{0};
    bool initialized_{false};
};

}  // namespace shilmandb
