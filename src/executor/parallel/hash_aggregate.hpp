#pragma once
#include "executor/parallel/expression.hpp"
#include "executor/parallel/key_hash.hpp"
#include "executor/parallel/pipeline.hpp"
#include "executor/parallel/vector.hpp"
#include "parser/ast.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace shilmandb::parallel {

// Worker tables merge into one up to this many groups in all, else by partition.
inline constexpr size_t kMergeIntoOneMaxGroups = 100000;

// A GROUP BY and its aggregates, or one global row when there are no keys.
struct AggregateSpec {
    std::vector<std::unique_ptr<CompiledExpr>> keys;  // compiled over the input batch
    std::vector<Aggregate::Func> funcs;
    std::vector<std::unique_ptr<CompiledExpr>> args;  // one per func; nullptr for COUNT, which never reads it
    std::vector<VType> out_types;                     // keys, then aggregates, as the plan's output schema
    bool global{false};                               // no GROUP BY: one row, even on empty input
};

// What an aggregate emits. COUNT: BIGINT. SUM of an integer or decimal: an exact
// wide decimal at the argument's scale; of anything else: DOUBLE. AVG: DOUBLE.
// MIN / MAX: the argument's type. Throws when a non-COUNT has no argument.
[[nodiscard]] VType AggregateOutputType(Aggregate::Func func, const CompiledExpr* arg);

// Hash partitions Finalize merges into: 1 up to kMergeIntoOneMaxGroups, else a power of two.
[[nodiscard]] size_t MergePartitionCount(size_t summed_groups, unsigned threads);

// The sink groups each worker's rows in its own table; Finalize merges them.
// The source then hands out every group, keys then aggregates as slots 0..n-1,
// in several morsels and no set order. Throws when out_types disagree with the rest.
// Past the memory budget a worker's table goes to disk by spill partition,
// and the source merges one partition per morsel instead.
[[nodiscard]] std::pair<std::unique_ptr<Sink>, std::unique_ptr<Source>> MakeHashAggregate(AggregateSpec spec);

// Tests only: the sink adds offset to every COUNT state when it finalizes.
// Throws when the sink isn't an aggregate's.
void OffsetCountsForTest(Sink& aggregate_sink, int64_t offset);

// Tests only: a merge of spilled groups splits past `most` groups; 0 turns it off.
// Read by the aggregate's source when it prepares.
void SetAggregateMergeGroupsForTest(size_t most);

}  // namespace shilmandb::parallel
