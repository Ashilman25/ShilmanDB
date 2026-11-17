#include "planner/column_usage.hpp"

#include <algorithm>
#include <optional>
#include <utility>

namespace shilmandb {

namespace {

void Walk(const Expression* expr, const Schema& schema, std::vector<uint32_t>& out) {
    if (expr == nullptr) return;

    switch (expr->type) {
        case ExprType::COLUMN_REF: {
            const auto& col = static_cast<const ColumnRef&>(*expr);
            if (col.bound_index) {
                out.push_back(*col.bound_index);
                break;
            }

            const auto& columns = schema.GetColumns();
            for (uint32_t i = 0; i < columns.size(); ++i) {
                if (columns[i].name == col.column_name) {
                    out.push_back(i);
                    break;
                }
            }
            break;
        }
        case ExprType::BINARY_OP: {
            const auto& bin = static_cast<const BinaryOp&>(*expr);
            Walk(bin.left.get(), schema, out);
            Walk(bin.right.get(), schema, out);
            break;
        }
        case ExprType::UNARY_OP:
            Walk(static_cast<const UnaryOp&>(*expr).operand.get(), schema, out);
            break;
        case ExprType::AGGREGATE:
            Walk(static_cast<const Aggregate&>(*expr).arg.get(), schema, out);
            break;
        case ExprType::CASE: {
            const auto& c = static_cast<const CaseExpression&>(*expr);
            for (const auto& [cond, result] : c.when_clauses) {
                Walk(cond.get(), schema, out);
                Walk(result.get(), schema, out);
            }
            Walk(c.else_clause.get(), schema, out);
            break;
        }
        case ExprType::LITERAL:
        case ExprType::STAR:
            break;
    }
}

void SortUnique(std::vector<uint32_t>& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

}  // namespace

void CollectColumns(const Expression* expr, const Schema& schema, std::vector<uint32_t>& out) {
    Walk(expr, schema, out);
    SortUnique(out);
}

std::vector<uint32_t> ColumnsReadBy(const PlanNode& node) {
    std::vector<uint32_t> out;
    const Schema* child_schema = node.children.empty() ? nullptr : &node.children[0]->output_schema;

    switch (node.type) {
        case PlanNodeType::FILTER:
            if (child_schema) {
                CollectColumns(static_cast<const FilterPlanNode&>(node).predicate.get(), *child_schema, out);
            }
            break;
        case PlanNodeType::PROJECTION:
            if (child_schema) {
                for (const auto& e : static_cast<const ProjectionPlanNode&>(node).expressions) {
                    CollectColumns(e.get(), *child_schema, out);
                }
            }
            break;
        case PlanNodeType::AGGREGATE:
            if (child_schema) {
                const auto& agg = static_cast<const AggregatePlanNode&>(node);
                for (const auto& e : agg.group_by_exprs) CollectColumns(e.get(), *child_schema, out);
                for (const auto& e : agg.aggregate_exprs) CollectColumns(e.get(), *child_schema, out);
            }
            break;
        case PlanNodeType::SORT:
            if (child_schema) {
                for (const auto& item : static_cast<const SortPlanNode&>(node).order_by) {
                    CollectColumns(item.expr.get(), *child_schema, out);
                }
            }
            break;
        case PlanNodeType::HASH_JOIN:
            CollectColumns(static_cast<const HashJoinPlanNode&>(node).join_predicate.get(), node.output_schema, out);
            break;
        case PlanNodeType::NESTED_LOOP_JOIN:
            CollectColumns(static_cast<const NestedLoopJoinPlanNode&>(node).predicate.get(), node.output_schema, out);
            break;
        case PlanNodeType::SEQ_SCAN:
        case PlanNodeType::INDEX_SCAN:
        case PlanNodeType::LIMIT:
            break;
    }
    return out;
}

namespace {

// Where an output column comes from: a scan's table column, or nothing if derived.
using Source = std::optional<std::pair<const PlanNode*, uint32_t>>;

// The columns a scan lists; unset means every column.
const std::optional<std::vector<uint32_t>>& ListedColumns(const PlanNode& scan) {
    if (scan.type == PlanNodeType::SEQ_SCAN) return static_cast<const SeqScanPlanNode&>(scan).read_columns;
    return static_cast<const IndexScanPlanNode&>(scan).read_columns;
}

// Checks the node's reads against the scans below it and returns its output's sources.
std::vector<Source> CheckReads(const PlanNode& node, std::vector<UnlistedRead>& out) {
    std::vector<std::vector<Source>> below;
    for (const auto& child : node.children) below.push_back(CheckReads(*child, out));

    const size_t width = node.output_schema.GetColumnCount();
    std::vector<Source> sources(width);
    switch (node.type) {
        case PlanNodeType::SEQ_SCAN:
        case PlanNodeType::INDEX_SCAN:
            for (uint32_t c = 0; c < width; ++c) sources[c] = std::make_pair(&node, c);
            break;
        case PlanNodeType::FILTER:
        case PlanNodeType::SORT:
        case PlanNodeType::LIMIT:
            if (!below.empty()) sources = below[0];
            break;
        case PlanNodeType::HASH_JOIN:
        case PlanNodeType::NESTED_LOOP_JOIN:
            // Left columns, then right, as BuildJoinSchema lays them out.
            if (below.size() == 2) {
                sources = below[0];
                sources.insert(sources.end(), below[1].begin(), below[1].end());
            }
            break;
        case PlanNodeType::PROJECTION:
        case PlanNodeType::AGGREGATE:
            break;
    }

    // Joins read their own combined row; the others read their child's.
    const bool is_join = node.type == PlanNodeType::HASH_JOIN || node.type == PlanNodeType::NESTED_LOOP_JOIN;
    const std::vector<Source>* input = is_join ? &sources : (below.empty() ? nullptr : &below[0]);
    if (input == nullptr) return sources;

    for (const uint32_t idx : ColumnsReadBy(node)) {
        if (idx >= input->size() || !(*input)[idx]) continue;
        const auto [scan, column] = *(*input)[idx];
        const auto& listed = ListedColumns(*scan);
        if (!listed || std::find(listed->begin(), listed->end(), column) != listed->end()) continue;
        out.push_back(UnlistedRead{&node, idx, scan, column});
    }
    return sources;
}

}  // namespace

std::vector<UnlistedRead> FindUnlistedReads(const PlanNode& root) {
    std::vector<UnlistedRead> out;
    (void)CheckReads(root, out);
    return out;
}

}  // namespace shilmandb
