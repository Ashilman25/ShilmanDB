#pragma once
#include "buffer/buffer_pool_manager.hpp"
#include "catalog/catalog.hpp"
#include <chrono>
#include <cstdint>
#include <unordered_map>

namespace shilmandb {

struct PlanNode;

// What one operator did during a query. Time includes everything below it.
struct OperatorStats {
    uint64_t rows{0};
    std::chrono::nanoseconds time{0};
};

using OperatorStatsMap = std::unordered_map<const PlanNode*, OperatorStats>;

struct ExecutorContext {
    BufferPoolManager* bpm;
    Catalog* catalog;
    OperatorStatsMap* op_stats{nullptr};  // set only when profiling a query
};

} //shilmandb
