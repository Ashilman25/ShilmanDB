#pragma once

#include "catalog/schema.hpp"
#include "catalog/table_stats.hpp"
#include "parser/ast.hpp"
#include "types/types.hpp"
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace shilmandb {

struct CostModel {
    static constexpr double kSeqPageReadCost = 1.0;         // sequential page read
    static constexpr double kRandomPageReadCost = 4.0;      // index scatter
    static constexpr double kTupleProcessCost = 0.01;       // Per tuple CPU cost
    static constexpr double kIndexTraversalCost = 2.0;      // Per Btree level descent

    static constexpr double kMinSelectivity = 0.001;        
    static constexpr double kOpenRangeSelectivity = 0.33;   // heuristic for col > v / col < v
    static constexpr double kDefaultSelectivity = 0.1;      
    static constexpr int kBTreeFanOut = 500;                // ~PAGE_SIZE / (key + pointer + overhead)

    [[nodiscard]] static double EstimateTablePages(uint64_t row_count, const Schema& schema);
    [[nodiscard]] static int EstimateTreeHeight(uint64_t row_count, TypeId key_type);

    // Fraction of rows that pass every predicate in `conjuncts` (an AND list). Range comparisons
    // on one column are merged into a single interval and priced from the column's min and max.
    // With `only_column`, predicates on other columns count as 1.0 (how one index is priced).
    [[nodiscard]] static double EstimateSelectivity(const std::vector<const Expression*>& conjuncts, const TableStats& stats, std::string_view only_column = {});

    // Rows a table's own filters keep: row_count * selectivity, at least 1 (0 for an empty table).
    [[nodiscard]] static double EstimateFilteredRows(const std::vector<const Expression*>& filters, const TableStats& stats);

    [[nodiscard]] static double SeqScanCost(uint64_t row_count, const Schema& schema);
    [[nodiscard]] static double IndexScanCost(uint64_t row_count, double selectivity, TypeId key_type);
};

}  // namespace shilmandb
