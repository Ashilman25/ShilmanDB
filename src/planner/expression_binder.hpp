#pragma once
#include "catalog/schema.hpp"
#include "parser/ast.hpp"
#include "planner/plan_node.hpp"

namespace shilmandb {

// Stores each column reference's position in `schema`, so rows are not searched by name.
// A literal compared to a column is also converted to the column's type once.
void BindExpression(Expression* expr, const Schema& schema);

// Binds every expression in the plan against the schema its executor reads rows with.
void BindPlan(PlanNode* node);

}  // namespace shilmandb
