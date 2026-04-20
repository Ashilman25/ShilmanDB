#pragma once
#include "executor/aggregate_state.hpp"
#include "executor/executor.hpp"

#include <map>
#include <memory>
#include <vector>

namespace shilmandb {

class AggregateExecutor : public Executor {
public:
    AggregateExecutor(const PlanNode* plan, ExecutorContext* ctx, std::unique_ptr<Executor> child);

    void Init() override;
    bool Next(Tuple* tuple) override;
    void Close() override;

private:
    std::unique_ptr<Executor> child_;
    std::map<std::vector<Value>, std::vector<AggregateState>> groups_;
    std::map<std::vector<Value>, std::vector<AggregateState>>::const_iterator group_iter_;
    bool initialized_{false};
};

}  // namespace shilmandb
