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

// A column an operator reads that comes from a scan which doesn't list it.
struct UnlistedRead {
    const PlanNode* reader;  // the node whose expression reads the column
    uint32_t index;          // the column's position in the row that node reads
    const PlanNode* scan;    // the scan the column comes from
    uint32_t scan_column;    // the column's index in that scan's table
};

// Every read of a scan column missing from that scan's read_columns.
[[nodiscard]] std::vector<UnlistedRead> FindUnlistedReads(const PlanNode& root);

}  // namespace shilmandb
