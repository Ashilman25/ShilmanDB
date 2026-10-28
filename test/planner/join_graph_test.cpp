#include <gtest/gtest.h>
#include <algorithm>
#include "planner/join_graph.hpp"

namespace shilmandb {

TEST(JoinGraphTest, SingleTable) {
    const std::vector<std::vector<int>> expected{{0}};
    EXPECT_EQ(ConnectedOrders(1, {}), expected);
    EXPECT_TRUE(IsConnectedOrder({0}, {}));
}

TEST(JoinGraphTest, ChainOfThree) {
    // 0 - 1 - 2
    const std::vector<TableEdge> edges{{0, 1}, {1, 2}};
    const std::vector<std::vector<int>> expected{{0, 1, 2}, {1, 0, 2}, {1, 2, 0}, {2, 1, 0}};
    EXPECT_EQ(ConnectedOrders(3, edges), expected);
}

TEST(JoinGraphTest, ChainOfFive) {
    // A chain of n tables has 2^(n-1) connected orders
    const std::vector<TableEdge> edges{{0, 1}, {1, 2}, {2, 3}, {3, 4}};
    const auto orders = ConnectedOrders(5, edges);
    EXPECT_EQ(orders.size(), 16u);
    EXPECT_TRUE(std::is_sorted(orders.begin(), orders.end()));
}

TEST(JoinGraphTest, Star) {
    // Table 0 in the middle: 6 orders start with it, and 6 start with a leaf then go to 0
    const std::vector<TableEdge> edges{{0, 1}, {0, 2}, {0, 3}};
    EXPECT_EQ(ConnectedOrders(4, edges).size(), 12u);
}

TEST(JoinGraphTest, Disconnected) {
    // Table 2 has no edge, so no order can reach it without a cross product
    const std::vector<TableEdge> edges{{0, 1}};
    EXPECT_TRUE(ConnectedOrders(3, edges).empty());
    EXPECT_FALSE(IsConnectedOrder({0, 1, 2}, edges));
}

TEST(JoinGraphTest, EdgeDirectionIgnored) {
    EXPECT_TRUE(IsConnectedOrder({0, 2, 1}, {{2, 0}, {1, 2}}));
}

TEST(JoinGraphTest, ZeroTables) {
    EXPECT_TRUE(ConnectedOrders(0, {}).empty());
}

}  // namespace shilmandb
