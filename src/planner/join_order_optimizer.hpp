#pragma once
#include "catalog/table_stats.hpp"
#include "parser/ast.hpp"
#include "planner/join_graph.hpp"
#include <vector>

namespace shilmandb {

class JoinOrderOptimizer {
public:

    // With edges, only orders connected through them compete, so the search never picks a cross
    // product the joins don't ask for. If no order is connected (or edges is empty) it searches all.
    [[nodiscard]] static auto FindBestOrder(const std::vector<TableRef>& tables, const std::vector<JoinClause>& joins, const std::vector<TableStats>& stats, const std::vector<TableEdge>& edges = {}) -> std::vector<int>;
    [[nodiscard]] static auto EstimateCost(const std::vector<int>& order, const std::vector<TableRef>& tables, const std::vector<JoinClause>& joins, const std::vector<TableStats>& stats) -> double;

    // Estimated row count after each table joins in. Element 0 is the first table's row count,
    // and EstimateCost is these values added up in order.
    [[nodiscard]] static auto EstimateStepRows(const std::vector<int>& order, const std::vector<TableRef>& tables, const std::vector<JoinClause>& joins, const std::vector<TableStats>& stats) -> std::vector<double>;

    [[nodiscard]] static auto BuildFeatureVector(const std::vector<TableRef>& tables, const std::vector<JoinClause>& joins, const std::vector<TableStats>& stats) -> std::vector<float>;
};

}  // namespace shilmandb
