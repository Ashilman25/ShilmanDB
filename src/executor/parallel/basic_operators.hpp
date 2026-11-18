#pragma once
#include "executor/parallel/expression.hpp"
#include "executor/parallel/pipeline.hpp"
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace shilmandb::parallel {

// Keeps the rows where a predicate is true; the batch passes on without a copy.
class FilterOperator : public Operator {
public:
    explicit FilterOperator(std::unique_ptr<CompiledExpr> predicate);

    [[nodiscard]] std::unique_ptr<LocalState> MakeLocal() const override;
    OpResult Execute(Batch& in, Batch& out, LocalState* local) override;

private:
    std::unique_ptr<CompiledExpr> predicate_;
};

// Computes one output slot per expression, for the input's active rows.
class ProjectionOperator : public Operator {
public:
    explicit ProjectionOperator(std::vector<std::unique_ptr<CompiledExpr>> expressions);

    // The type of each output slot, in order.
    [[nodiscard]] const std::vector<VType>& OutputTypes() const { return types_; }

    [[nodiscard]] std::unique_ptr<LocalState> MakeLocal() const override;
    OpResult Execute(Batch& in, Batch& out, LocalState* local) override;

private:
    std::vector<std::unique_ptr<CompiledExpr>> expressions_;
    std::vector<VType> types_;
};

// Rows a LIMIT has let through, shared by every worker; never more than the limit.
class LimitCounter {
public:
    explicit LimitCounter(uint64_t limit) : limit_(limit) {}

    // Takes up to `wanted` of the rows left and returns how many it got.
    [[nodiscard]] uint64_t Reserve(uint64_t wanted);

    // True once every row the limit allows is taken.
    [[nodiscard]] bool Reached() const { return taken_.load(std::memory_order_relaxed) >= limit_; }

private:
    const uint64_t limit_;
    std::atomic<uint64_t> taken_{0};
};

// Passes on rows until its counter is used up; the pipeline's sink watches the counter.
class LimitOperator : public Operator {
public:
    explicit LimitOperator(uint64_t limit) : counter_(limit) {}

    [[nodiscard]] const LimitCounter& Counter() const { return counter_; }

    OpResult Execute(Batch& in, Batch& out, LocalState* local) override;

private:
    LimitCounter counter_;
};

}  // namespace shilmandb::parallel
