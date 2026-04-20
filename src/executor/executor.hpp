#pragma once
#include "catalog/schema.hpp"
#include "executor/data_chunk.hpp"
#include "executor/executor_context.hpp"
#include "planner/plan_node.hpp"
#include "types/tuple.hpp"

namespace shilmandb {

enum class ExecutionMode { 
    TUPLE, 
    VECTORIZED 
};

class Executor {
public:
    Executor(const PlanNode* plan, ExecutorContext* ctx) : plan_(plan), ctx_(ctx) {}
    virtual ~Executor() = default;

    virtual void Init() = 0;
    virtual bool Next(Tuple* tuple) = 0;
    virtual void Close() = 0;

    virtual bool NextBatch(DataChunk* chunk) {
        chunk->Reset();
        Tuple tuple;
        while (!chunk->IsFull()) {
            if (!Next(&tuple)) break;
            chunk->AppendTuple(tuple);
        }
        return chunk->size() > 0;
    }

    const Schema& GetOutputSchema() const { return plan_->output_schema; }

protected:
    const PlanNode* plan_;
    ExecutorContext* ctx_;
};

}  // namespace shilmandb
