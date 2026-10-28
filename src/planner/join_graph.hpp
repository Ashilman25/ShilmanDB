#pragma once
#include <utility>
#include <vector>

namespace shilmandb {

// Two table slots tied by a join predicate. Direction doesn't matter.
using TableEdge = std::pair<int, int>;

// True if every table after the first shares an edge with a table already placed
[[nodiscard]] auto IsConnectedOrder(const std::vector<int>& order, const std::vector<TableEdge>& edges) -> bool;

// Every connected order of tables 0..num_tables-1 in alphabetical order.
[[nodiscard]] auto ConnectedOrders(int num_tables, const std::vector<TableEdge>& edges) -> std::vector<std::vector<int>>;

}  // namespace shilmandb
