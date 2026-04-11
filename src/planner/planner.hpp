#pragma once
#include "catalog/catalog.hpp"
#include "parser/ast.hpp"
#include "planner/plan_node.hpp"
#include <memory>

namespace shilmandb {

#ifdef SHILMANDB_HAS_LIBTORCH
class LearnedJoinOptimizer;
#endif

class Planner {
public:
    explicit Planner(Catalog* catalog);

#ifdef SHILMANDB_HAS_LIBTORCH
    Planner(Catalog* catalog, LearnedJoinOptimizer* learned_optimizer);
#endif

    [[nodiscard]] auto Plan(SelectStatement stmt) -> std::unique_ptr<PlanNode>;

private:
    Catalog* catalog_;

#ifdef SHILMANDB_HAS_LIBTORCH
    LearnedJoinOptimizer* learned_optimizer_{nullptr};
#endif

};

}  // namespace shilmandb
