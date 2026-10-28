#include "planner/join_graph.hpp"
#include <algorithm>
#include <cstddef>
#include <numeric>

namespace shilmandb {

auto IsConnectedOrder(const std::vector<int>& order, const std::vector<TableEdge>& edges) -> bool {
    // The placed tables are just the front of the order, so we search that instead of
    // building a lookup table. Odd slot values (negative, too large) can't go out of bounds.
    for (size_t k = 1; k < order.size(); ++k) {
        const auto placed_end = order.begin() + static_cast<std::ptrdiff_t>(k);
        const auto is_placed = [&](int slot) { return std::find(order.begin(), placed_end, slot) != placed_end; };

        const int next = order[k];
        const bool tied = std::any_of(edges.begin(), edges.end(), [&](const TableEdge& e) {
            return (e.first == next && is_placed(e.second)) || (e.second == next && is_placed(e.first));
        });
        if (!tied) return false;
    }
    return true;
}

auto ConnectedOrders(int num_tables, const std::vector<TableEdge>& edges) -> std::vector<std::vector<int>> {
    if (num_tables <= 0) return {};

    std::vector<int> order(static_cast<size_t>(num_tables));
    std::iota(order.begin(), order.end(), 0);

    std::vector<std::vector<int>> connected;
    do {
        if (IsConnectedOrder(order, edges)) connected.push_back(order);
    } while (std::next_permutation(order.begin(), order.end()));
    return connected;
}

}  // namespace shilmandb
