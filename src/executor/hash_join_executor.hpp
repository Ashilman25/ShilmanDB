#pragma once
#include "executor/executor.hpp"
#include "types/value.hpp"
#include <memory>
#include <unordered_map>
#include <vector>

namespace shilmandb {

class HashJoinExecutor : public Executor {
public:
    HashJoinExecutor(const PlanNode* plan, ExecutorContext* ctx, std::unique_ptr<Executor> left, std::unique_ptr<Executor> right) : Executor(plan, ctx), left_child_(std::move(left)), right_child_(std::move(right)) {}

    void Init() override;
    bool Next(Tuple* tuple) override;
    void Close() override;

private:
    std::unique_ptr<Executor> left_child_;
    std::unique_ptr<Executor> right_child_;

    std::unordered_map<Value, std::vector<Tuple>> hash_table_;
    uint32_t left_key_idx_{0};
    uint32_t right_key_idx_{0};

    const std::vector<Tuple>* current_bucket_{nullptr};
    size_t match_idx_{0};
    Tuple current_right_;

    void ExtractKeyIndices();
};

}  // namespace shilmandb
