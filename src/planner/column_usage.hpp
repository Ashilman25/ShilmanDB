#pragma once
#include <cstdint>
#include <vector>

#include "catalog/schema.hpp"
#include "parser/ast.hpp"
#include "planner/plan_node.hpp"

namespace shilmandb {

// Adds every column the expression reads to out, kept sorted and unique.
void CollectColumns(const Expression* expr, const Schema& schema, std::vector<uint32_t>& out);

// Columns of the schema a node's expressions run against, sorted and unique.
[[nodiscard]] std::vector<uint32_t> ColumnsReadBy(const PlanNode& node);

}  // namespace shilmandb
