#include "executor/limit_executor.hpp"
#include "planner/plan_node.hpp"
#include <cassert>

namespace shilmandb {

LimitExecutor::LimitExecutor(const PlanNode* plan, ExecutorContext* ctx, std::unique_ptr<Executor> child) : Executor(plan, ctx), child_(std::move(child)), limit_(static_cast<const LimitPlanNode*>(plan)->limit) {}

void LimitExecutor::Init() {
    child_->Init();
    emitted_ = 0;
    initialized_ = true;
}

bool LimitExecutor::Next(Tuple* tuple) {
    assert(initialized_ && "Next() called before Init()");

    if (emitted_ >= limit_) {
        return false;
    }

    if (!child_->Next(tuple)) {
        return false;
    }

    ++emitted_;
    return true;
}

void LimitExecutor::Close() {
    child_->Close();
    emitted_ = 0;
    initialized_ = false;
}

}  // namespace shilmandb
