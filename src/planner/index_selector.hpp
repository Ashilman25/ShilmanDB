#pragma once

#include "catalog/catalog.hpp"
#include "catalog/schema.hpp"
#include "parser/ast.hpp"
#include "planner/plan_node.hpp"
#include "types/value.hpp"
#include <memory>
#include <optional>
#include <string>
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

class IndexSelector {
public:
    [[nodiscard]] static std::unique_ptr<PlanNode> SelectScanStrategy(const std::string& table_name, const Schema& table_schema, 
                                                                        std::vector<std::unique_ptr<Expression>>& predicates, 
                                                                        std::vector<std::unique_ptr<Expression>>& residual_out, 
                                                                        Catalog* catalog);

    [[nodiscard]] static ConjunctMatch MatchSingleConjunct(const Expression* expr, const std::string& indexed_column);
};

}  // namespace shilmandb
