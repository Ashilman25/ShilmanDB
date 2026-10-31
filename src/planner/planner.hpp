#pragma once
#include "catalog/catalog.hpp"
#include "parser/ast.hpp"
#include "planner/join_graph.hpp"
#include "planner/plan_node.hpp"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace shilmandb {

#ifdef SHILMANDB_HAS_LIBTORCH
class LearnedJoinOptimizer;
#endif

// Who picked the join order. LEARNED_REJECTED means the model ran but its order wasn't used.
enum class JoinOrderSource { SINGLE_TABLE, CLASSICAL, LEARNED, LEARNED_REJECTED, FORCED };

[[nodiscard]] constexpr auto ToString(JoinOrderSource s) -> const char* {
    switch (s) {
        case JoinOrderSource::SINGLE_TABLE:     return "single_table";
        case JoinOrderSource::CLASSICAL:        return "classical";
        case JoinOrderSource::LEARNED:          return "learned";
        case JoinOrderSource::LEARNED_REJECTED: return "learned_rejected";
        case JoinOrderSource::FORCED:           return "forced";
    }
    return "unknown";
}

// How the last Plan() call ordered its joins. Table slots are indexes into `tables`.
struct JoinOrderInfo {
    std::vector<std::string> tables;    // table name per slot: FROM tables, then JOIN tables, as written
    std::vector<double> est_scan_rows;  // rows each table's own WHERE filters are expected to keep, one per slot
    std::vector<TableEdge> edges;       // tables tied by an ON condition or a two-table WHERE predicate - each pair is (low, high), sorted, no duplicates
    std::vector<int> order;
    JoinOrderSource source{JoinOrderSource::SINGLE_TABLE};
    std::vector<int> model_order;       // what the model proposed; empty if it didn't run or failed first
    std::vector<double> est_step_rows;  // cost formula's row estimate after each step of `order`
    double est_cost{0.0};               // est_step_rows added up
};

class Planner {
public:
    explicit Planner(Catalog* catalog);

#ifdef SHILMANDB_HAS_LIBTORCH
    Planner(Catalog* catalog, LearnedJoinOptimizer* learned_optimizer);
#endif

    [[nodiscard]] auto Plan(SelectStatement stmt) -> std::unique_ptr<PlanNode>;

    // Join tables in exactly this order instead of picking one. Plan() throws if the order
    // doesn't list every table once or would need a cross product.
    void SetForcedJoinOrder(std::vector<int> order) { forced_order_ = std::move(order); }

    [[nodiscard]] auto GetJoinOrderInfo() const -> const JoinOrderInfo& { return join_info_; }

    // Raw pointers into the last plan, so they're only valid while that plan lives.
    // Step k is the node whose output is the join result after k + 1 tables.
    [[nodiscard]] auto GetStepNodes() const -> const std::vector<const PlanNode*>& { return step_nodes_; }
    // One per table slot: the table's scan, or the filter sitting on top of it.
    [[nodiscard]] auto GetScanNodes() const -> const std::vector<const PlanNode*>& { return scan_nodes_; }

private:
    Catalog* catalog_;

#ifdef SHILMANDB_HAS_LIBTORCH
    LearnedJoinOptimizer* learned_optimizer_{nullptr};
#endif

    std::optional<std::vector<int>> forced_order_;
    JoinOrderInfo join_info_;
    std::vector<const PlanNode*> step_nodes_;
    std::vector<const PlanNode*> scan_nodes_;
};

}  // namespace shilmandb
