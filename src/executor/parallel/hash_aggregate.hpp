#pragma once
#include "executor/parallel/expression.hpp"
#include "executor/parallel/pipeline.hpp"
#include "executor/parallel/vector.hpp"
#include "parser/ast.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace shilmandb::parallel {

namespace detail {

// Added before mixing each key, so a zero key still moves the hash.
inline constexpr uint64_t kKeySeed = 0x9e3779b97f4a7c15ULL;

// Spreads 64 bits over the whole word (murmur3's finalizer); a bijection.
[[nodiscard]] constexpr uint64_t HashMix(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

// Folds one key's bits into a row's hash, which starts at 0; key order matters.
[[nodiscard]] constexpr uint64_t FoldKey(uint64_t h, uint64_t bits) {
    return HashMix(h ^ (bits + kKeySeed));
}

}  // namespace detail

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

// What an aggregate emits. COUNT: INTEGER. SUM of an integer or decimal: an exact
// wide decimal at the argument's scale; of anything else: DOUBLE. AVG: DOUBLE.
// MIN / MAX: the argument's type. Throws when a non-COUNT has no argument.
[[nodiscard]] VType AggregateOutputType(Aggregate::Func func, const CompiledExpr* arg);

// Hash partitions Finalize merges into: 1 up to kMergeIntoOneMaxGroups, else a power of two.
[[nodiscard]] size_t MergePartitionCount(size_t summed_groups, unsigned threads);

// The sink groups each worker's rows in its own table; Finalize merges them.
// The source then hands out every group, keys then aggregates as slots 0..n-1,
// in several morsels and no set order. Throws when out_types disagree with the rest.
[[nodiscard]] std::pair<std::unique_ptr<Sink>, std::unique_ptr<Source>> MakeHashAggregate(AggregateSpec spec);

}  // namespace shilmandb::parallel
