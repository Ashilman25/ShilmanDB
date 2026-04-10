#pragma once
#include "executor/executor.hpp"
#include <memory>
#include <vector>

namespace shilmandb {

class SortExecutor : public Executor {
public:
    SortExecutor(const PlanNode* plan, ExecutorContext* ctx, std::unique_ptr<Executor> child);

    void Init() override;
    bool Next(Tuple* tuple) override;
    void Close() override;

private:
    std::unique_ptr<Executor> child_;
    std::vector<Tuple> sorted_tuples_;
    size_t cursor_{0};
    bool initialized_{false};
};

}  // namespace shilmandb
