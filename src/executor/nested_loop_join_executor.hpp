#pragma once
#include "executor/executor.hpp"
#include <memory>
#include <vector>

namespace shilmandb {

class NestedLoopJoinExecutor : public Executor {
public:
    NestedLoopJoinExecutor(const PlanNode* plan, ExecutorContext* ctx, std::unique_ptr<Executor> left, std::unique_ptr<Executor> right) : Executor(plan, ctx), left_child_(std::move(left)), right_child_(std::move(right)) {}

    void Init() override;
    bool Next(Tuple* tuple) override;
    void Close() override;

private:
    std::unique_ptr<Executor> left_child_;
    std::unique_ptr<Executor> right_child_;

    std::vector<Tuple> right_tuples_;
    Tuple current_left_;
    size_t right_idx_{0};
    bool left_done_{true};
};

}  // namespace shilmandb
