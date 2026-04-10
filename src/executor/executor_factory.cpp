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
#include "common/exception.hpp"

namespace shilmandb {

std::unique_ptr<Executor> ExecutorFactory::CreateExecutor(const PlanNode* plan, ExecutorContext* ctx) {
    switch (plan->type) {
        case PlanNodeType::SEQ_SCAN:
            return std::make_unique<SeqScanExecutor>(plan, ctx);

        case PlanNodeType::INDEX_SCAN:
            return std::make_unique<IndexScanExecutor>(plan, ctx);

        case PlanNodeType::FILTER: {
            auto child = CreateExecutor(plan->children[0].get(), ctx);
            return std::make_unique<FilterExecutor>(plan, ctx, std::move(child));
        }

        case PlanNodeType::HASH_JOIN: {
            auto left = CreateExecutor(plan->children[0].get(), ctx);
            auto right = CreateExecutor(plan->children[1].get(), ctx);
            return std::make_unique<HashJoinExecutor>(plan, ctx, std::move(left), std::move(right));
        }

        case PlanNodeType::NESTED_LOOP_JOIN: {
            auto left = CreateExecutor(plan->children[0].get(), ctx);
            auto right = CreateExecutor(plan->children[1].get(), ctx);
            return std::make_unique<NestedLoopJoinExecutor>(plan, ctx, std::move(left), std::move(right));
        }

        case PlanNodeType::SORT: {
            auto child = CreateExecutor(plan->children[0].get(), ctx);
            return std::make_unique<SortExecutor>(plan, ctx, std::move(child));
        }

        case PlanNodeType::AGGREGATE: {
            auto child = CreateExecutor(plan->children[0].get(), ctx);
            return std::make_unique<AggregateExecutor>(plan, ctx, std::move(child));
        }

        case PlanNodeType::PROJECTION: {
            auto child = CreateExecutor(plan->children[0].get(), ctx);
            return std::make_unique<ProjectionExecutor>(plan, ctx, std::move(child));
        }

        case PlanNodeType::LIMIT: {
            auto child = CreateExecutor(plan->children[0].get(), ctx);
            return std::make_unique<LimitExecutor>(plan, ctx, std::move(child));
        }

        default:
            throw DatabaseException("Unknown plan node type");
    }
}

}  // namespace shilmandb
