#include "planner/join_order_optimizer.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_set>

namespace shilmandb {

namespace {

struct JoinEdge {
    int table_a;  // index into tables[], always < table_b
    int table_b;
    std::vector<std::pair<std::string, std::string>> column_pairs;  // (col_a, col_b)
};


auto ResolveTableIndex(const std::optional<std::string>& col_table_name, const std::vector<TableRef>& tables) -> int {
    if (!col_table_name.has_value()) return -1;
    const auto& name = col_table_name.value();

    for (int i = 0; i < static_cast<int>(tables.size()); ++i) {
        if (tables[i].alias.has_value() && tables[i].alias.value() == name) return i;
        if (tables[i].table_name == name) return i;
    }
    return -1;
}


void ExtractEquiJoinPairs(const Expression* expr, const std::vector<TableRef>& tables, std::vector<std::tuple<int, std::string, int, std::string>>& pairs) {
    if (!expr) return;

    const auto* bin = dynamic_cast<const BinaryOp*>(expr);
    if (!bin) return;

    if (bin->op == BinaryOp::Op::AND) {
        ExtractEquiJoinPairs(bin->left.get(), tables, pairs);
        ExtractEquiJoinPairs(bin->right.get(), tables, pairs);
        return;
    }

    if (bin->op != BinaryOp::Op::EQ) return;

    const auto* left_col = dynamic_cast<const ColumnRef*>(bin->left.get());
    const auto* right_col = dynamic_cast<const ColumnRef*>(bin->right.get());
    if (!left_col || !right_col) return;

    int left_idx = ResolveTableIndex(left_col->table_name, tables);
    int right_idx = ResolveTableIndex(right_col->table_name, tables);
    if (left_idx < 0 || right_idx < 0 || left_idx == right_idx) return;


    if (left_idx < right_idx) {
        pairs.emplace_back(left_idx, left_col->column_name, right_idx, right_col->column_name);
    } else {
        pairs.emplace_back(right_idx, right_col->column_name, left_idx, left_col->column_name);
    }
}


auto ExtractJoinEdges(const std::vector<TableRef>& tables, const std::vector<JoinClause>& joins) -> std::vector<JoinEdge> {
    std::vector<std::tuple<int, std::string, int, std::string>> raw_pairs;

    for (const auto& join : joins) {
        ExtractEquiJoinPairs(join.on_condition.get(), tables, raw_pairs);
    }

    std::vector<JoinEdge> edges;
    for (const auto& raw : raw_pairs) {
        int ta = std::get<0>(raw);
        const std::string& ca = std::get<1>(raw);

        int tb = std::get<2>(raw);
        const std::string& cb = std::get<3>(raw);

        auto it = std::find_if(edges.begin(), edges.end(), [ta, tb](const JoinEdge& e) {
            return e.table_a == ta && e.table_b == tb;
        });

        if (it != edges.end()) {
            it->column_pairs.emplace_back(ca, cb);
        } else {
            edges.push_back({ta, tb, {{ca, cb}}});
        }
    }
    return edges;
}


auto LookupDistinct(const TableStats& stats, const std::string& col_name) -> uint64_t {
    auto it = stats.distinct_counts.find(col_name);
    return (it != stats.distinct_counts.end()) ? it->second : stats.row_count;
}


auto EstimateCostInternal(const std::vector<int>& order, const std::vector<JoinEdge>& edges, const std::vector<TableStats>& stats) -> double {
    if (order.empty()) return 0.0;

    double intermediate = static_cast<double>(stats[order[0]].row_count);
    double total_cost = intermediate;

    std::unordered_set<int> joined;
    joined.insert(order[0]);

    for (size_t k = 1; k < order.size(); ++k) {
        const int new_table = order[k];
        const double new_rows = static_cast<double>(stats[new_table].row_count);

        double selectivity = 1.0;
        bool has_join = false;

        for (const auto& edge : edges) {
            bool a_in_joined = joined.find(edge.table_a) != joined.end();
            bool b_in_joined = joined.find(edge.table_b) != joined.end();

            bool a_is_new = (edge.table_a == new_table);
            bool b_is_new = (edge.table_b == new_table);

            if ((a_in_joined && b_is_new) || (b_in_joined && a_is_new)) {
                has_join = true;
                for (const auto& [col_a, col_b] : edge.column_pairs) {
                    auto v_a = LookupDistinct(stats[edge.table_a], col_a);
                    auto v_b = LookupDistinct(stats[edge.table_b], col_b);
                    auto max_v = std::max(v_a, v_b);

                    if (max_v > 0) {
                        selectivity *= 1.0 / static_cast<double>(max_v);
                    }
                }
            }
        }

        intermediate = has_join ? intermediate * new_rows * selectivity : intermediate * new_rows;

        total_cost += intermediate;
        joined.insert(new_table);
    }

    return total_cost;
}

}  // anonymous namespace






auto JoinOrderOptimizer::FindBestOrder(const std::vector<TableRef>& tables, const std::vector<JoinClause>& joins, const std::vector<TableStats>& stats) -> std::vector<int> {
    const auto n = static_cast<int>(tables.size());

    if (n <= 1) {
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        return order;
    }

    if (n > 6) {
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        return order;
    }


    auto edges = ExtractJoinEdges(tables, joins);

    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);

    double best_cost = std::numeric_limits<double>::max();
    std::vector<int> best_order = perm;

    do {
        double cost = EstimateCostInternal(perm, edges, stats);
        if (cost < best_cost) {
            best_cost = cost;
            best_order = perm;
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    return best_order;
}

auto JoinOrderOptimizer::EstimateCost(const std::vector<int>& order, const std::vector<TableRef>& tables, const std::vector<JoinClause>& joins, const std::vector<TableStats>& stats) -> double {
    auto edges = ExtractJoinEdges(tables, joins);
    return EstimateCostInternal(order, edges, stats);
}

auto JoinOrderOptimizer::BuildFeatureVector(const std::vector<TableRef>& tables, const std::vector<JoinClause>& joins, const std::vector<TableStats>& stats) -> std::vector<float> {
    constexpr int kMaxTables = 6;
    constexpr int kTableFeatures = 3;
    constexpr int kPairFeatures = 2;
    constexpr int kNumPairs = 15;  // C(6, 2)
    constexpr int kTotalFeatures = kMaxTables * kTableFeatures + kNumPairs * kPairFeatures;

    std::vector<float> features(kTotalFeatures, 0.0f);
    const int n = std::min(static_cast<int>(tables.size()), kMaxTables);

    for (int i = 0; i < n; ++i) {
        features[i * kTableFeatures + 0] = static_cast<float>(std::log(static_cast<double>(stats[i].row_count) + 1.0));

        double avg_distinct = 0.0;
        if (!stats[i].distinct_counts.empty()) {
            double sum = 0.0;
            for (const auto& [col, cnt] : stats[i].distinct_counts) {
                sum += static_cast<double>(cnt);
            }

            avg_distinct = sum / static_cast<double>(stats[i].distinct_counts.size());
        }
        features[i * kTableFeatures + 1] = static_cast<float>(std::log(avg_distinct + 1.0));
        features[i * kTableFeatures + 2] = 1.0f;
    }


    auto edges = ExtractJoinEdges(tables, joins);
    int p = 0;

    for (int i = 0; i < kMaxTables; ++i) {
        for (int j = i + 1; j < kMaxTables; ++j) {
            const int offset = kMaxTables * kTableFeatures + p * kPairFeatures;

            for (const auto& edge : edges) {
                if ((edge.table_a == i && edge.table_b == j) || (edge.table_a == j && edge.table_b == i)) {

                    features[offset] = 1.0f;

                    if (!edge.column_pairs.empty() && i < n && j < n) {
                        const auto& [col_a, col_b] = edge.column_pairs[0];

                        auto v_a = LookupDistinct(stats[edge.table_a], col_a);
                        auto v_b = LookupDistinct(stats[edge.table_b], col_b);

                        auto max_v = std::max(v_a, v_b);
                        auto min_v = std::min(v_a, v_b);
                        if (max_v > 0) {
                            features[offset + 1] = static_cast<float>(static_cast<double>(min_v) / static_cast<double>(max_v));
                        }
                    }
                    break;
                }
            }
            ++p;
        }
    }

    return features;
}

}  // namespace shilmandb
