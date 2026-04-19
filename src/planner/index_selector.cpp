#include "planner/index_selector.hpp"
#include "planner/cost_model.hpp"
#include "common/exception.hpp"
#include <algorithm>
#include <limits>

namespace shilmandb {

namespace {

// Returns (column_ref, literal, effective_op) after literal-swap normalization.
// operator is mirrored: LT<->GT, LTE<->GTE, EQ<->EQ, NEQ<->NEQ.
struct NormalizedCompare {
    const ColumnRef* col{nullptr};
    const Literal* lit{nullptr};
    BinaryOp::Op op{};
};

BinaryOp::Op MirrorComparison(BinaryOp::Op op) {
    switch (op) {
        case BinaryOp::Op::LT:
            return BinaryOp::Op::GT;
        case BinaryOp::Op::GT:
            return BinaryOp::Op::LT;
        case BinaryOp::Op::LTE:
            return BinaryOp::Op::GTE;
        case BinaryOp::Op::GTE: 
            return BinaryOp::Op::LTE;
        default: 
            return op;
    }
}

NormalizedCompare NormalizeCompare(const BinaryOp* bin) {
    NormalizedCompare out;
    if (!bin) return out;

    const auto* lc = dynamic_cast<const ColumnRef*>(bin->left.get());
    const auto* rc = dynamic_cast<const ColumnRef*>(bin->right.get());
    const auto* ll = dynamic_cast<const Literal*>(bin->left.get());
    const auto* rl = dynamic_cast<const Literal*>(bin->right.get());

    if (lc && rl) {
        out.col = lc;
        out.lit = rl;
        out.op  = bin->op;
    } else if (rc && ll) {
        out.col = rc;
        out.lit = ll;
        out.op  = MirrorComparison(bin->op);
    }
    return out;
}

}  // namespace

ConjunctMatch IndexSelector::MatchSingleConjunct(const Expression* expr, const std::string& indexed_column) {
    ConjunctMatch m;
    const auto* bin = dynamic_cast<const BinaryOp*>(expr);
    if (!bin) return m;

    const NormalizedCompare n = NormalizeCompare(bin);
    if (!n.col) return m;
    if (n.col->column_name != indexed_column) return m;

    switch (n.op) {
        case BinaryOp::Op::EQ:
            m.referenced_column = true;
            m.contributes_low = true;
            m.contributes_high = true;
            m.low_value = n.lit->value;
            m.high_value = n.lit->value;
            m.fully_consumed = true;
            return m;
        case BinaryOp::Op::GTE:
            m.referenced_column = true;
            m.contributes_low = true;
            m.low_value = n.lit->value;
            m.fully_consumed = true;
            return m;
        case BinaryOp::Op::LTE:
            m.referenced_column = true;
            m.contributes_high = true;
            m.high_value = n.lit->value;
            m.fully_consumed = true;
            return m;
        case BinaryOp::Op::GT:
            m.referenced_column = true;
            m.contributes_low  = true;
            m.low_value = n.lit->value;
            m.fully_consumed = false;  // strict > stays in residual
            return m;
        case BinaryOp::Op::LT:
            m.referenced_column = true;
            m.contributes_high = true;
            m.high_value = n.lit->value;
            m.fully_consumed = false;  // strict < stays in residual
            return m;
        default:
            // NEQ, AND, OR, arithmetic
            return m;
    }
}

std::unique_ptr<PlanNode> IndexSelector::SelectScanStrategy(const std::string& table_name, const Schema& table_schema, std::vector<std::unique_ptr<Expression>>& predicates, std::vector<std::unique_ptr<Expression>>& residual_out, Catalog* catalog) {
    auto* table_info = catalog->GetTable(table_name);
    if (!table_info) {
        throw DatabaseException("IndexSelector: table not found: " + table_name);
    }

    const auto& stats = table_info->stats;
    const double seq_cost = CostModel::SeqScanCost(stats.row_count, table_schema);

    struct BestCandidate {
        IndexInfo* index{nullptr};
        std::optional<Value> low_key;
        std::optional<Value> high_key;
        double cost{std::numeric_limits<double>::infinity()};
        std::vector<size_t> consumed_indices;
    };
    BestCandidate best;

    const auto indexes = catalog->GetTableIndexes(table_name);
    for (auto* idx_info : indexes) {
        const TypeId key_type = table_schema.GetColumn(idx_info->col_idx).type;

        std::optional<Value> low;
        std::optional<Value> high;
        std::vector<size_t>  consumed;
        std::vector<const Expression*> matched_preds;

        for (size_t i = 0; i < predicates.size(); ++i) {
            const ConjunctMatch m = MatchSingleConjunct(predicates[i].get(), idx_info->column_name);
            if (!m.referenced_column) continue;
            matched_preds.push_back(predicates[i].get());

            if (m.contributes_low) {
                if (!low.has_value() || m.low_value > *low) low = m.low_value;
            }
            if (m.contributes_high) {
                if (!high.has_value() || m.high_value < *high) high = m.high_value;
            }
            if (m.fully_consumed) consumed.push_back(i);
        }

        if (matched_preds.empty()) continue;

        double sel = 1.0;
        for (const auto* pred : matched_preds) {
            sel *= CostModel::EstimateSelectivity(pred, idx_info->column_name, stats);
        }
        sel = std::max(CostModel::kMinSelectivity, std::min(1.0, sel));

        const double idx_cost = CostModel::IndexScanCost(stats.row_count, sel, key_type);
        if (idx_cost < best.cost) {
            best.index = idx_info;
            best.low_key = low;
            best.high_key = high;
            best.cost = idx_cost;
            best.consumed_indices = std::move(consumed);
        }
    }

    if (best.index != nullptr && best.cost < seq_cost) {
        auto scan = std::make_unique<IndexScanPlanNode>(table_schema, table_name, best.index->index_name);
        scan->low_key  = best.low_key;
        scan->high_key = best.high_key;

        const auto& consumed = best.consumed_indices;
        for (size_t i = 0; i < predicates.size(); ++i) {
            const bool is_consumed = std::find(consumed.begin(), consumed.end(), i) != consumed.end();
            if (is_consumed) {
                predicates[i].reset();  // destroy
            } else {
                residual_out.push_back(std::move(predicates[i]));
            }
        }
        predicates.clear();
        return scan;
    }

    for (auto& p : predicates) residual_out.push_back(std::move(p));
    predicates.clear();
    return std::make_unique<SeqScanPlanNode>(table_schema, table_name);
}

}  // namespace shilmandb
