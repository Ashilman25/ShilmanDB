#pragma once
#include "catalog/table_stats.hpp"
#include "parser/ast.hpp"
#include <vector>

namespace shilmandb {

class JoinOrderOptimizer {
public:

    [[nodiscard]] static auto FindBestOrder(const std::vector<TableRef>& tables, const std::vector<JoinClause>& joins, const std::vector<TableStats>& stats) -> std::vector<int>;
    [[nodiscard]] static auto EstimateCost(const std::vector<int>& order, const std::vector<TableRef>& tables, const std::vector<JoinClause>& joins, const std::vector<TableStats>& stats) -> double;

    // Estimated row count after each table joins in. Element 0 is the first table's row count,
    // and EstimateCost is these values added up in order.
    [[nodiscard]] static auto EstimateStepRows(const std::vector<int>& order, const std::vector<TableRef>& tables, const std::vector<JoinClause>& joins, const std::vector<TableStats>& stats) -> std::vector<double>;

    [[nodiscard]] static auto BuildFeatureVector(const std::vector<TableRef>& tables, const std::vector<JoinClause>& joins, const std::vector<TableStats>& stats) -> std::vector<float>;
};

}  // namespace shilmandb
