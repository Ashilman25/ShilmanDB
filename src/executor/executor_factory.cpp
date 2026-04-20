#include "executor/executor_factory.hpp"
#include "executor/seq_scan_executor.hpp"
#include "executor/index_scan_executor.hpp"
#include "executor/filter_executor.hpp"
#include "executor/hash_join_executor.hpp"
#include "executor/nested_loop_join_executor.hpp"
#include "executor/sort_executor.hpp"
#include "executor/aggregate_executor.hpp"
#include "executor/projection_executor.hpp"
#include "executor/limit_executor.hpp"
#include "executor/vectorized_seq_scan_executor.hpp"
#include "executor/vectorized_filter_executor.hpp"
#include "executor/vectorized_projection_executor.hpp"
#include "executor/vectorized_aggregate_executor.hpp"
#include "common/exception.hpp"

namespace shilmandb {

std::unique_ptr<Executor> ExecutorFactory::CreateExecutor(const PlanNode* plan, ExecutorContext* ctx, ExecutionMode mode) {
    switch (plan->type) {
        case PlanNodeType::SEQ_SCAN: {
            switch (mode) {
                case ExecutionMode::VECTORIZED:
                    return std::make_unique<VectorizedSeqScanExecutor>(plan, ctx);
                case ExecutionMode::TUPLE:
                    return std::make_unique<SeqScanExecutor>(plan, ctx);
            }
            throw DatabaseException("Unknown execution mode");
        }

        case PlanNodeType::INDEX_SCAN: {
            switch (mode) {
                case ExecutionMode::VECTORIZED:
                    [[fallthrough]];
                case ExecutionMode::TUPLE:
                    return std::make_unique<IndexScanExecutor>(plan, ctx);
            }
            throw DatabaseException("Unknown execution mode");
        }

        case PlanNodeType::FILTER: {
            auto child = CreateExecutor(plan->children[0].get(), ctx, mode);
            switch (mode) {
                case ExecutionMode::VECTORIZED:
                    return std::make_unique<VectorizedFilterExecutor>(plan, ctx, std::move(child));
                case ExecutionMode::TUPLE:
                    return std::make_unique<FilterExecutor>(plan, ctx, std::move(child));
            }
            throw DatabaseException("Unknown execution mode");
        }

        case PlanNodeType::HASH_JOIN: {
            auto left = CreateExecutor(plan->children[0].get(), ctx, mode);
            auto right = CreateExecutor(plan->children[1].get(), ctx, mode);
            switch (mode) {
                case ExecutionMode::VECTORIZED:
                    [[fallthrough]];
                case ExecutionMode::TUPLE:
                    return std::make_unique<HashJoinExecutor>(plan, ctx, std::move(left), std::move(right));
            }
            throw DatabaseException("Unknown execution mode");
        }

        case PlanNodeType::NESTED_LOOP_JOIN: {
            auto left = CreateExecutor(plan->children[0].get(), ctx, mode);
            auto right = CreateExecutor(plan->children[1].get(), ctx, mode);
            switch (mode) {
                case ExecutionMode::VECTORIZED:
                    [[fallthrough]];
                case ExecutionMode::TUPLE:
                    return std::make_unique<NestedLoopJoinExecutor>(plan, ctx, std::move(left), std::move(right));
            }
            throw DatabaseException("Unknown execution mode");
        }

        case PlanNodeType::SORT: {
            auto child = CreateExecutor(plan->children[0].get(), ctx, mode);
            switch (mode) {
                case ExecutionMode::VECTORIZED:
                    [[fallthrough]];
                case ExecutionMode::TUPLE:
                    return std::make_unique<SortExecutor>(plan, ctx, std::move(child));
            }
            throw DatabaseException("Unknown execution mode");
        }

        case PlanNodeType::AGGREGATE: {
            auto child = CreateExecutor(plan->children[0].get(), ctx, mode);
            switch (mode) {
                case ExecutionMode::VECTORIZED:
                    return std::make_unique<VectorizedAggregateExecutor>(plan, ctx, std::move(child));
                case ExecutionMode::TUPLE:
                    return std::make_unique<AggregateExecutor>(plan, ctx, std::move(child));
            }
            throw DatabaseException("Unknown execution mode");
        }

        case PlanNodeType::PROJECTION: {
            auto child = CreateExecutor(plan->children[0].get(), ctx, mode);
            switch (mode) {
                case ExecutionMode::VECTORIZED:
                    return std::make_unique<VectorizedProjectionExecutor>(plan, ctx, std::move(child));
                case ExecutionMode::TUPLE:
                    return std::make_unique<ProjectionExecutor>(plan, ctx, std::move(child));
            }
            throw DatabaseException("Unknown execution mode");
        }

        case PlanNodeType::LIMIT: {
            auto child = CreateExecutor(plan->children[0].get(), ctx, mode);
            switch (mode) {
                case ExecutionMode::VECTORIZED:
                    [[fallthrough]];
                case ExecutionMode::TUPLE:
                    return std::make_unique<LimitExecutor>(plan, ctx, std::move(child));
            }
            throw DatabaseException("Unknown execution mode");
        }

        default:
            throw DatabaseException("Unknown plan node type");
    }
}

}  // namespace shilmandb
