#pragma once
#include "catalog/table_stats.hpp"
#include "parser/ast.hpp"
#include <vector>

namespace shilmandb {

class JoinOrderOptimizer {
public:

    [[nodiscard]] static auto FindBestOrder(const std::vector<TableRef>& tables, const std::vector<JoinClause>& joins, const std::vector<TableStats>& stats) -> std::vector<int>;
    [[nodiscard]] static auto EstimateCost(const std::vector<int>& order, const std::vector<TableRef>& tables, const std::vector<JoinClause>& joins, const std::vector<TableStats>& stats) -> double;
    [[nodiscard]] static auto BuildFeatureVector(const std::vector<TableRef>& tables, const std::vector<JoinClause>& joins, const std::vector<TableStats>& stats) -> std::vector<float>;
};

}  // namespace shilmandb
