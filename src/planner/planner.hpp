#pragma once
#include "catalog/catalog.hpp"
#include "parser/ast.hpp"
#include "planner/plan_node.hpp"
#include <memory>

namespace shilmandb {

class Planner {
public:
    explicit Planner(Catalog* catalog);

    [[nodiscard]] auto Plan(SelectStatement stmt) -> std::unique_ptr<PlanNode>;

private:
    Catalog* catalog_;
};

}  // namespace shilmandb
