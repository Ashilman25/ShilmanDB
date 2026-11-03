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

// What a scan costs, in microseconds. The CPU parts were measured on this engine; the page
// reads depend on where pages come from, so they follow the storage.
struct ScanCosts {
    static constexpr double kCachedPageReadUs = 0.6;   // from the OS file cache
    static constexpr double kDirectPageReadUs = 50.0;  // macOS direct I/O, to the SSD
    static constexpr double kScanRowUs = 0.06;         // CPU per row a sequential scan reads and filters
    static constexpr double kIndexRowUs = 0.07;        // CPU per row an index scan returns

    double seq_page_us{kCachedPageReadUs};
    double random_page_us{kCachedPageReadUs};
    double scan_row_us{kScanRowUs};
    double index_row_us{kIndexRowUs};

    // Both kinds of page read cost `us`; the CPU parts keep their measured values.
    [[nodiscard]] static constexpr ScanCosts ForPageRead(double us) { return {us, us, kScanRowUs, kIndexRowUs}; }
};

struct CostModel {
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

    // The table's real page count when the stats have one, else an estimate from its row count.
    [[nodiscard]] static double TablePages(const TableStats& stats, const Schema& schema);

    [[nodiscard]] static double SeqScanCost(const TableStats& stats, const Schema& schema, const ScanCosts& costs = ScanCosts{});
    [[nodiscard]] static double IndexScanCost(uint64_t row_count, double selectivity, TypeId key_type, const ScanCosts& costs = ScanCosts{});
};

}  // namespace shilmandb
