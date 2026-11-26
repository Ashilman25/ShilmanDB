#pragma once
#include "executor/parallel/expression.hpp"
#include "executor/parallel/pipeline.hpp"
#include "executor/parallel/vector.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace shilmandb::parallel {

// An ORDER BY, with a LIMIT right above it folded in as a top n.
// Rows equal on every key come out ordered by the kept columns, ascending in
// column order, so the visible result is the same at any thread count.
struct SortSpec {
    std::vector<std::unique_ptr<CompiledExpr>> keys;  // compiled over the input batch
    std::vector<bool> ascending;                      // one per key
    std::optional<int64_t> limit;                     // from a LIMIT directly above the SORT
    std::vector<uint32_t> columns;                    // input slots kept; the source emits them as slots 0..n-1
    std::vector<VType> types;                         // one per kept column
};

// The sink takes every row. After its Finalize the source hands them out sorted,
// at most max(limit, 0) of them, in one morsel so the order holds.
// Keys compare as Value's < and > do: integers and dates as integers,
// decimals exactly, doubles as doubles, strings bytewise.
// Rows past a worker's budget go to temp files as sorted runs;
// the source merges them into the same order, byte for byte.
[[nodiscard]] std::pair<std::unique_ptr<Sink>, std::unique_ptr<Source>> MakeSort(SortSpec spec);

// For tests: caps a merge's fan-in, at least 2; 0 turns the cap off.
// Read when a sort's source is prepared, on the calling thread.
void SetSortFanInForTest(size_t most);

// What a sort merge holds after it loads blocks, for tests.
struct SortMergeState {
    size_t runs{0};           // runs the merge reads at once
    size_t loaded_bytes{0};   // its loaded blocks, by the row estimate
    size_t counted_bytes{0};  // what the budget counts for those blocks
};

// For tests: called by a merge each time it loads blocks; empty turns it off.
// Set only while no sort runs.
void SetSortMergeHookForTest(std::function<void(const SortMergeState&)> hook);

}  // namespace shilmandb::parallel
