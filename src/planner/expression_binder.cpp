#include "planner/expression_binder.hpp"

#include <exception>

namespace shilmandb {

namespace {

constexpr bool IsComparison(BinaryOp::Op op) {
    return op == BinaryOp::Op::EQ || op == BinaryOp::Op::NEQ || op == BinaryOp::Op::LT ||
           op == BinaryOp::Op::GT || op == BinaryOp::Op::LTE || op == BinaryOp::Op::GTE;
}

// Converts a literal to the column's type once, so it is not converted again for every row
void ConvertLiteral(const Expression* column_side, Expression* literal_side, const Schema& schema) {
    if (column_side->type != ExprType::COLUMN_REF || literal_side->type != ExprType::LITERAL) return;
    const auto& col = static_cast<const ColumnRef&>(*column_side);
    if (!col.bound_index) return;

    const TypeId column_type = schema.GetColumns()[*col.bound_index].type;
    auto& value = static_cast<Literal*>(literal_side)->value;
    value = value.WidenTo(column_type);
}

}  // namespace

void BindExpression(Expression* expr, const Schema& schema) {
    if (expr == nullptr) return;

    switch (expr->type) {
        case ExprType::COLUMN_REF: {
            auto* col = static_cast<ColumnRef*>(expr);
            const auto& columns = schema.GetColumns();
            col->bound_index.reset();  // don't keep a position from an earlier bind

            for (uint32_t i = 0; i < columns.size(); ++i) {
                if (columns[i].name == col->column_name) {
                    col->bound_index = i;
                    break;
                }
            }
            break;
        }
        case ExprType::BINARY_OP: {
            auto* bin = static_cast<BinaryOp*>(expr);
            BindExpression(bin->left.get(), schema);
            BindExpression(bin->right.get(), schema);
            if (IsComparison(bin->op)) {
                ConvertLiteral(bin->left.get(), bin->right.get(), schema);
                ConvertLiteral(bin->right.get(), bin->left.get(), schema);
            }
            break;
        }
        case ExprType::UNARY_OP:
            BindExpression(static_cast<UnaryOp*>(expr)->operand.get(), schema);
            break;
        case ExprType::AGGREGATE:
            BindExpression(static_cast<Aggregate*>(expr)->arg.get(), schema);
            break;
        case ExprType::CASE: {
            auto* c = static_cast<CaseExpression*>(expr);
            for (auto& [cond, result] : c->when_clauses) {
                BindExpression(cond.get(), schema);
                BindExpression(result.get(), schema);
            }
            BindExpression(c->else_clause.get(), schema);
            break;
        }
        case ExprType::LITERAL:
        case ExprType::STAR:
            break;
    }
}

void BindPlan(PlanNode* node) {
    if (node == nullptr) return;

    // Executors read rows with their child's schema, except joins, which read the combined row.
    const Schema* child_schema = node->children.empty() ? nullptr : &node->children[0]->output_schema;

    // A hand-built node with no child has nothing to bind against; its expressions stay unbound.
    const bool needs_child = node->type == PlanNodeType::FILTER || node->type == PlanNodeType::AGGREGATE || node->type == PlanNodeType::PROJECTION || node->type == PlanNodeType::SORT;
    if (needs_child && child_schema == nullptr) {
        return;
    }

    switch (node->type) {
        case PlanNodeType::FILTER:
            BindExpression(static_cast<FilterPlanNode*>(node)->predicate.get(), *child_schema);
            break;
        case PlanNodeType::HASH_JOIN:
            BindExpression(static_cast<HashJoinPlanNode*>(node)->join_predicate.get(), node->output_schema);
            break;
        case PlanNodeType::NESTED_LOOP_JOIN:
            BindExpression(static_cast<NestedLoopJoinPlanNode*>(node)->predicate.get(), node->output_schema);
            break;
        case PlanNodeType::AGGREGATE: {
            auto* agg = static_cast<AggregatePlanNode*>(node);
            for (auto& e : agg->group_by_exprs) BindExpression(e.get(), *child_schema);
            for (auto& e : agg->aggregate_exprs) BindExpression(e.get(), *child_schema);
            break;
        }
        case PlanNodeType::PROJECTION:
            for (auto& e : static_cast<ProjectionPlanNode*>(node)->expressions) {
                BindExpression(e.get(), *child_schema);
            }
            break;
        case PlanNodeType::SORT:
            for (auto& item : static_cast<SortPlanNode*>(node)->order_by) {
                BindExpression(item.expr.get(), *child_schema);
            }
            break;
        case PlanNodeType::SEQ_SCAN:
        case PlanNodeType::INDEX_SCAN:
        case PlanNodeType::LIMIT:
            break;
    }

    for (auto& child : node->children) BindPlan(child.get());
}

}  // namespace shilmandb
