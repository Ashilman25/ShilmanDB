#pragma once
#include "executor/aggregate_state.hpp"
#include "executor/executor.hpp"
#include "executor/expression_evaluator.hpp"
#include "parser/ast.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace shilmandb {

class VectorizedAggregateExecutor : public Executor {
public:
    VectorizedAggregateExecutor(const PlanNode* plan, ExecutorContext* ctx, std::unique_ptr<Executor> child);

    void Init() override;
    bool Next(Tuple* tuple) override;
    bool NextBatch(DataChunk* chunk) override;
    void Close() override;

private:
    void UpdateAggregates(std::vector<AggregateState>& states, const ChunkRow& row);
    std::vector<Value> BuildResultValues(const std::vector<Value>& key, const std::vector<AggregateState>& states) const;

    std::unique_ptr<Executor> child_;
    std::vector<const Expression*> group_by_exprs_;   // borrowed from AggregatePlanNode
    std::vector<const Expression*> aggregate_exprs_;  // borrowed from AggregatePlanNode
    std::vector<Aggregate::Func> aggregate_funcs_;    // copied (trivial enum)
    std::vector<uint32_t> cols_;                      // child columns the expressions read

    std::map<std::vector<Value>, std::vector<AggregateState>> groups_;
    std::map<std::vector<Value>, std::vector<AggregateState>>::iterator emit_iter_;

    bool initialized_{false};
    DataChunk buffer_;
    size_t buffer_cursor_{0};
};

}  // namespace shilmandb
