#pragma once
#include "planner/plan_node.hpp"
#include "executor/executor_context.hpp"
#include "types/tuple.hpp"
#include "catalog/schema.hpp"

namespace shilmandb {

class Executor {

public:
    Executor(const PlanNode* plan, ExecutorContext* ctx) : plan_(plan), ctx_(ctx) {}
    virtual ~Executor() = default;

    virtual void Init() = 0;
    virtual bool Next(Tuple* tuple) = 0;
    virtual void Close() = 0;

    const Schema& GetOutputSchema() const {return plan_->output_schema;}

protected:
    const PlanNode* plan_;
    ExecutorContext* ctx_;

};
}