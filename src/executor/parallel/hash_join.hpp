#pragma once
#include "executor/parallel/expression.hpp"
#include "executor/parallel/pipeline.hpp"
#include "executor/parallel/vector.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace shilmandb::parallel {

// About this many build rows per hash partition; a smaller build is one partition.
inline constexpr size_t kJoinRowsPerPartition = 65536;

// An inner equi-join on one key pair. The build side goes into the sink first;
// probe batches then stream through the operator, which writes the left side's
// slots, then the right side's, in the join's output column order.
struct JoinSpec {
    std::unique_ptr<CompiledExpr> build_key;  // over a build batch
    std::unique_ptr<CompiledExpr> probe_key;  // over a probe batch
    std::vector<uint32_t> build_payload;      // build batch slots kept for the output
    std::vector<uint32_t> probe_slots;        // probe batch slots copied to the output
    std::vector<VType> types;                 // one per output slot: the left side's, then the right side's
    std::unique_ptr<CompiledExpr> residual;   // the rest of the join predicate, over the output; may be null
    bool build_left{false};                   // the build side is the join's left child
};

// The sink keeps each worker's keys and payload, strings copied; Finalize builds one
// table by hash partition. The operator pairs probe rows with matching build rows.
// Keys compare by value: integers and decimals exactly, doubles as doubles. Other mixes
// throw "the parallel executor can't join keys of types <left> and <right>".
[[nodiscard]] std::pair<std::unique_ptr<Sink>, std::unique_ptr<Operator>> MakeHashJoin(JoinSpec spec);

// Partitions for this many build rows: 1 up to kJoinRowsPerPartition, else a power of two.
[[nodiscard]] size_t JoinPartitionCount(size_t rows, unsigned threads);

// What a hash join's probe operator knows about its build side.
struct HashJoinInfo {
    bool build_left{false};
    bool has_residual{false};
    size_t build_rows{0};  // 0 until the build side is finalized
    size_t partitions{0};
};

// For tests: a hash join probe operator's info; nullopt for any other operator.
[[nodiscard]] std::optional<HashJoinInfo> InspectHashJoin(const Operator& op);

}  // namespace shilmandb::parallel
