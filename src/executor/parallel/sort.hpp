#pragma once
#include "executor/parallel/expression.hpp"
#include "executor/parallel/pipeline.hpp"
#include "executor/parallel/vector.hpp"
#include <cstdint>
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
[[nodiscard]] std::pair<std::unique_ptr<Sink>, std::unique_ptr<Source>> MakeSort(SortSpec spec);

}  // namespace shilmandb::parallel
