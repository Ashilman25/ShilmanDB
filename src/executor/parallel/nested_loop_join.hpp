#pragma once
#include "executor/parallel/expression.hpp"
#include "executor/parallel/pipeline.hpp"
#include "executor/parallel/vector.hpp"
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace shilmandb::parallel {

// A join that tests every left row against every right row.
// Candidate batches hold the left slots first, then the right ones,
// in the join's output column order.
struct NestedLoopSpec {
    std::unique_ptr<CompiledExpr> predicate;  // over a candidate batch; nullptr keeps every pair
    bool right_materialized{true};            // the right side always goes into the sink first
    std::vector<uint32_t> left_slots;         // left batch slots, copied to candidate slots 0..
    std::vector<uint32_t> right_slots;        // right batch slots the sink keeps, placed after the left ones
    std::vector<VType> types;                 // one per candidate slot
};

// The sink keeps the whole right side, strings in its own arenas. Once it's
// finalized, the operator pairs each left batch with it, one block per call.
[[nodiscard]] std::pair<std::unique_ptr<Sink>, std::unique_ptr<Operator>> MakeNestedLoopJoin(NestedLoopSpec spec);

}  // namespace shilmandb::parallel
