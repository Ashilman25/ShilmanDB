#pragma once

#include "catalog/catalog.hpp"
#include "catalog/schema.hpp"
#include "parser/ast.hpp"
#include "planner/cost_model.hpp"
#include "planner/plan_node.hpp"
#include "types/value.hpp"
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace shilmandb {

struct ConjunctMatch {
    bool referenced_column{false};  // Predicate touches the indexed column
    bool contributes_low{false};
    bool contributes_high{false};
    Value low_value{};              
    Value high_value{};             
    bool fully_consumed{false};     
};

// What each way of scanning one table costs, in microseconds.
struct ScanCostEstimate {
    double seq{0.0};
    std::vector<std::pair<std::string, double>> indexes;  // index name and cost, for each index a predicate uses
};

class IndexSelector {
public:
    // Picks a table scan or an index. With `columnar` set (the parallel executor) a columnar
    // table gets columnar prices, and `read_columns` lists what it reads (null = every column).
    [[nodiscard]] static std::unique_ptr<PlanNode> SelectScanStrategy(
        const std::string& table_name, const Schema& table_schema, 
        std::vector<std::unique_ptr<Expression>>& predicates, 
        std::vector<std::unique_ptr<Expression>>& residual_out, 
        Catalog* catalog,
        const ScanCosts& costs = ScanCosts{},
        const ColumnarScanCosts* columnar = nullptr,
        const std::vector<uint32_t>* read_columns = nullptr
    );

    // The prices SelectScanStrategy compares, for a table's predicates (nested ANDs are split).
    [[nodiscard]] static ScanCostEstimate EstimateScanCosts(
        const std::string& table_name, const std::vector<const Expression*>& predicates,
        Catalog* catalog, const ScanCosts& costs = ScanCosts{},
        const ColumnarScanCosts* columnar = nullptr,
        const std::vector<uint32_t>* read_columns = nullptr
    );

    [[nodiscard]] static ConjunctMatch MatchSingleConjunct(const Expression* expr, const std::string& indexed_column);
};

}  // namespace shilmandb
