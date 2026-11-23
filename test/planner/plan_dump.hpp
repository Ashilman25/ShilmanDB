#pragma once
// Shared by the planner tests and the parallel tests that check COUNT types:
// a sanctioned exception to per-file test helpers.

#include "catalog/schema.hpp"
#include "parser/ast.hpp"
#include "planner/plan_node.hpp"
#include "types/value.hpp"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace shilmandb::test {

// The TypeId's name, as the dump prints it.
inline std::string TypeName(TypeId type) {
    switch (type) {
        case TypeId::INVALID: return "INVALID";
        case TypeId::INTEGER: return "INTEGER";
        case TypeId::BIGINT: return "BIGINT";
        case TypeId::DECIMAL: return "DECIMAL";
        case TypeId::VARCHAR: return "VARCHAR";
        case TypeId::DATE: return "DATE";
    }
    return "TYPE" + std::to_string(static_cast<int>(type));
}

// A value with its type, exact: doubles get all 17 digits.
inline std::string ValueText(const Value& v) {
    switch (v.type_) {
        case TypeId::INVALID: return "NULL";
        case TypeId::INTEGER: return "INTEGER:" + std::to_string(v.integer_);
        case TypeId::BIGINT: return "BIGINT:" + std::to_string(v.bigint_);
        case TypeId::DECIMAL: {
            char buf[40];
            std::snprintf(buf, sizeof(buf), "%.17g", v.decimal_);
            return std::string("DECIMAL:") + buf;
        }
        case TypeId::VARCHAR: return "VARCHAR:'" + v.varchar_ + "'";
        case TypeId::DATE: return "DATE:" + v.ToString();
    }
    return "?";
}

// The operator as SQL writes it.
inline std::string OpText(BinaryOp::Op op) {
    switch (op) {
        case BinaryOp::Op::EQ: return "=";
        case BinaryOp::Op::NEQ: return "<>";
        case BinaryOp::Op::LT: return "<";
        case BinaryOp::Op::GT: return ">";
        case BinaryOp::Op::LTE: return "<=";
        case BinaryOp::Op::GTE: return ">=";
        case BinaryOp::Op::AND: return "AND";
        case BinaryOp::Op::OR: return "OR";
        case BinaryOp::Op::ADD: return "+";
        case BinaryOp::Op::SUB: return "-";
        case BinaryOp::Op::MUL: return "*";
        case BinaryOp::Op::DIV: return "/";
        case BinaryOp::Op::LIKE: return "LIKE";
    }
    return "?";
}

// The aggregate function's SQL name.
inline std::string FuncText(Aggregate::Func func) {
    switch (func) {
        case Aggregate::Func::COUNT: return "COUNT";
        case Aggregate::Func::SUM: return "SUM";
        case Aggregate::Func::AVG: return "AVG";
        case Aggregate::Func::MIN: return "MIN";
        case Aggregate::Func::MAX: return "MAX";
    }
    return "?";
}

// An expression as text; a column shows its bound index after '@' ('@?' when unbound).
inline std::string ExprText(const Expression* expr) {
    if (expr == nullptr) return "none";
    switch (expr->type) {
        case ExprType::COLUMN_REF: {
            const auto* col = static_cast<const ColumnRef*>(expr);
            std::string out = col->table_name ? *col->table_name + "." : "";
            out += col->column_name + "@";
            out += col->bound_index ? std::to_string(*col->bound_index) : "?";
            return out;
        }
        case ExprType::LITERAL:
            return ValueText(static_cast<const Literal*>(expr)->value);
        case ExprType::BINARY_OP: {
            const auto* bin = static_cast<const BinaryOp*>(expr);
            return "(" + ExprText(bin->left.get()) + " " + OpText(bin->op) + " " + ExprText(bin->right.get()) + ")";
        }
        case ExprType::UNARY_OP: {
            const auto* un = static_cast<const UnaryOp*>(expr);
            const char* name = un->op == UnaryOp::Op::NOT ? "NOT" : "NEGATE";
            return std::string(name) + "(" + ExprText(un->operand.get()) + ")";
        }
        case ExprType::AGGREGATE: {
            const auto* agg = static_cast<const Aggregate*>(expr);
            return FuncText(agg->func) + "(" + (agg->arg ? ExprText(agg->arg.get()) : "*") + ")";
        }
        case ExprType::STAR:
            return "*";
        case ExprType::CASE: {
            const auto* ce = static_cast<const CaseExpression*>(expr);
            std::string out = "CASE";
            for (const auto& [cond, result] : ce->when_clauses) {
                out += " WHEN " + ExprText(cond.get()) + " THEN " + ExprText(result.get());
            }
            if (ce->else_clause) out += " ELSE " + ExprText(ce->else_clause.get());
            return out + " END";
        }
    }
    return "?";
}

// Expressions joined with ", ", or "none" for an empty list.
inline std::string ExprListText(const std::vector<std::unique_ptr<Expression>>& exprs) {
    if (exprs.empty()) return "none";
    std::string out;
    for (size_t i = 0; i < exprs.size(); ++i) out += (i ? ", " : "") + ExprText(exprs[i].get());
    return out;
}

// Each column as "name TYPE", joined with ", ".
inline std::string SchemaText(const Schema& schema) {
    std::string out;
    for (uint32_t i = 0; i < schema.GetColumnCount(); ++i) {
        const Column& c = schema.GetColumn(i);
        out += (i ? ", " : "") + c.name + " " + TypeName(c.type);
    }
    return out;
}

// "all" when unset, else the listed column indexes.
inline std::string ReadColumnsText(const std::optional<std::vector<uint32_t>>& cols) {
    if (!cols) return "all";
    std::string out = "[";
    for (size_t i = 0; i < cols->size(); ++i) out += (i ? ", " : "") + std::to_string((*cols)[i]);
    return out + "]";
}

// An index scan bound, or "none" when open.
inline std::string KeyText(const std::optional<Value>& key) { return key ? ValueText(*key) : "none"; }

// One node and its subtree: header, "| " detail lines, children two spaces deeper.
inline void DumpNode(const PlanNode& node, int depth, std::string& out) {
    const std::string pad(static_cast<size_t>(depth) * 2, ' ');
    const auto line = [&](const std::string& text) { out += pad + text + "\n"; };
    const auto detail = [&](const std::string& key, const std::string& value) { line("| " + key + ": " + value); };

    switch (node.type) {
        case PlanNodeType::SEQ_SCAN: {
            const auto& n = static_cast<const SeqScanPlanNode&>(node);
            line("SeqScan " + n.table_name);
            detail("schema", SchemaText(n.output_schema));
            detail("read_columns", ReadColumnsText(n.read_columns));
            detail("skip_conjuncts", ExprListText(n.skip_conjuncts));
            break;
        }
        case PlanNodeType::INDEX_SCAN: {
            const auto& n = static_cast<const IndexScanPlanNode&>(node);
            line("IndexScan " + n.table_name + " " + n.index_name);
            detail("schema", SchemaText(n.output_schema));
            detail("low", KeyText(n.low_key));
            detail("high", KeyText(n.high_key));
            detail("read_columns", ReadColumnsText(n.read_columns));
            break;
        }
        case PlanNodeType::FILTER: {
            const auto& n = static_cast<const FilterPlanNode&>(node);
            line("Filter");
            detail("schema", SchemaText(n.output_schema));
            detail("predicate", ExprText(n.predicate.get()));
            break;
        }
        case PlanNodeType::HASH_JOIN: {
            const auto& n = static_cast<const HashJoinPlanNode&>(node);
            line("HashJoin");
            detail("schema", SchemaText(n.output_schema));
            detail("predicate", ExprText(n.join_predicate.get()));
            break;
        }
        case PlanNodeType::NESTED_LOOP_JOIN: {
            const auto& n = static_cast<const NestedLoopJoinPlanNode&>(node);
            line("NestedLoopJoin");
            detail("schema", SchemaText(n.output_schema));
            detail("predicate", ExprText(n.predicate.get()));
            break;
        }
        case PlanNodeType::SORT: {
            const auto& n = static_cast<const SortPlanNode&>(node);
            line("Sort");
            detail("schema", SchemaText(n.output_schema));
            std::string keys;
            for (size_t i = 0; i < n.order_by.size(); ++i) {
                keys += (i ? ", " : "") + ExprText(n.order_by[i].expr.get()) + (n.order_by[i].ascending ? " ASC" : " DESC");
            }
            detail("keys", keys.empty() ? "none" : keys);
            break;
        }
        case PlanNodeType::AGGREGATE: {
            const auto& n = static_cast<const AggregatePlanNode&>(node);
            line("Aggregate");
            detail("schema", SchemaText(n.output_schema));
            detail("group_by", ExprListText(n.group_by_exprs));
            std::string aggs;
            for (size_t i = 0; i < n.aggregate_funcs.size(); ++i) {
                const Expression* arg = i < n.aggregate_exprs.size() ? n.aggregate_exprs[i].get() : nullptr;
                aggs += (i ? ", " : "") + FuncText(n.aggregate_funcs[i]) + "(" + (arg ? ExprText(arg) : "*") + ")";
            }
            detail("aggregates", aggs.empty() ? "none" : aggs);
            break;
        }
        case PlanNodeType::PROJECTION: {
            const auto& n = static_cast<const ProjectionPlanNode&>(node);
            line("Projection");
            detail("schema", SchemaText(n.output_schema));
            detail("exprs", ExprListText(n.expressions));
            break;
        }
        case PlanNodeType::LIMIT: {
            const auto& n = static_cast<const LimitPlanNode&>(node);
            line("Limit");
            detail("schema", SchemaText(n.output_schema));
            detail("limit", std::to_string(n.limit));
            break;
        }
        default:
            line("Unknown node type " + std::to_string(static_cast<int>(node.type)));
            break;
    }
    for (const auto& child : node.children) DumpNode(*child, depth + 1, out);
}

// The whole plan as text: shape, tables, indexes, keys, columns and expressions; never estimates.
[[nodiscard]] inline std::string DumpPlan(const PlanNode& root) {
    std::string out;
    DumpNode(root, 0, out);
    return out;
}

// The type a parallel plan gives expr over schema, whose counts columns are BIGINT there.
// Mirrors the planner's ResolveExprType, early returns included.
[[nodiscard]] inline TypeId ParallelExprType(const Expression* expr, const Schema& schema, const std::vector<bool>& counts) {
    if (const auto* col = dynamic_cast<const ColumnRef*>(expr)) {
        for (uint32_t c = 0; c < schema.GetColumnCount(); ++c) {
            if (schema.GetColumn(c).name != col->column_name) continue;
            return c < counts.size() && counts[c] ? TypeId::BIGINT : schema.GetColumn(c).type;
        }
        return TypeId::INTEGER;
    }
    if (const auto* lit = dynamic_cast<const Literal*>(expr)) return lit->value.type_;
    if (const auto* agg = dynamic_cast<const Aggregate*>(expr)) {
        switch (agg->func) {
            case Aggregate::Func::COUNT: return TypeId::BIGINT;
            case Aggregate::Func::SUM:
            case Aggregate::Func::AVG: return TypeId::DECIMAL;
            case Aggregate::Func::MIN:
            case Aggregate::Func::MAX: return agg->arg ? ParallelExprType(agg->arg.get(), schema, counts) : TypeId::INTEGER;
        }
    }
    if (const auto* ce = dynamic_cast<const CaseExpression*>(expr)) {
        if (ce->when_clauses.empty()) return TypeId::INTEGER;
        // A branch that can't merge gives the first branch's type, ELSE unread.
        const TypeId first = ParallelExprType(ce->when_clauses[0].second.get(), schema, counts);
        TypeId resolved = first;
        for (size_t i = 1; i < ce->when_clauses.size(); ++i) {
            const TypeId merged = CommonType(resolved, ParallelExprType(ce->when_clauses[i].second.get(), schema, counts));
            if (merged == TypeId::INVALID) return first;
            resolved = merged;
        }
        if (ce->else_clause) {
            const TypeId merged = CommonType(resolved, ParallelExprType(ce->else_clause.get(), schema, counts));
            if (merged != TypeId::INVALID) resolved = merged;
        }
        return resolved;
    }
    if (const auto* bin = dynamic_cast<const BinaryOp*>(expr)) {
        using Op = BinaryOp::Op;
        switch (bin->op) {
            case Op::EQ: case Op::NEQ: case Op::LT: case Op::GT:
            case Op::LTE: case Op::GTE: case Op::AND: case Op::OR:
            case Op::LIKE:
                return TypeId::INTEGER;
            case Op::ADD: case Op::SUB: case Op::MUL: case Op::DIV: {
                const TypeId lt = ParallelExprType(bin->left.get(), schema, counts);
                const TypeId merged = CommonType(lt, ParallelExprType(bin->right.get(), schema, counts));
                return merged == TypeId::INVALID ? lt : merged;
            }
        }
    }
    if (const auto* un = dynamic_cast<const UnaryOp*>(expr)) {
        if (un->op == UnaryOp::Op::NOT) return TypeId::INTEGER;
        return ParallelExprType(un->operand.get(), schema, counts);
    }
    return TypeId::INTEGER;
}

// True when expr comes from a COUNT column: INTEGER today, BIGINT in a parallel plan.
[[nodiscard]] inline bool ComputesFromCount(const Expression* expr, TypeId today, const Schema& schema, const std::vector<bool>& counts) {
    return today == TypeId::INTEGER && ParallelExprType(expr, schema, counts) == TypeId::BIGINT;
}

// One node's COUNT columns, given each child's: an aggregate's COUNTs and the INTEGER
// columns above that come from one. Parallel plans type these BIGINT.
[[nodiscard]] inline std::vector<bool> NodeCountColumns(const PlanNode& node, const std::vector<std::vector<bool>>& below) {
    const uint32_t width = node.output_schema.GetColumnCount();
    std::vector<bool> counts(width, false);
    switch (node.type) {
        case PlanNodeType::AGGREGATE: {
            // The aggregates are the last columns; a GROUP BY expression gets none.
            const auto& agg = static_cast<const AggregatePlanNode&>(node);
            const size_t funcs = agg.aggregate_funcs.size();
            const size_t first = width >= funcs ? width - funcs : 0;
            for (size_t i = 0; i < funcs && first + i < width; ++i) {
                counts[first + i] = agg.aggregate_funcs[i] == Aggregate::Func::COUNT;
            }
            break;
        }
        case PlanNodeType::PROJECTION: {
            const auto& proj = static_cast<const ProjectionPlanNode&>(node);
            const Schema& in = node.children[0]->output_schema;
            for (uint32_t c = 0; c < width && c < proj.expressions.size(); ++c) {
                counts[c] = ComputesFromCount(proj.expressions[c].get(), node.output_schema.GetColumn(c).type, in, below[0]);
            }
            break;
        }
        case PlanNodeType::FILTER:
        case PlanNodeType::SORT:
        case PlanNodeType::LIMIT:
            counts = below[0];
            break;
        case PlanNodeType::HASH_JOIN:
        case PlanNodeType::NESTED_LOOP_JOIN:
            counts = below[0];
            counts.insert(counts.end(), below[1].begin(), below[1].end());
            break;
        default:
            break;
    }
    counts.resize(width, false);
    for (uint32_t c = 0; c < width; ++c) counts[c] = counts[c] && node.output_schema.GetColumn(c).type == TypeId::INTEGER;
    return counts;
}

// The output columns of a TUPLE or VECTORIZED plan that a parallel plan types BIGINT.
[[nodiscard]] inline std::vector<bool> CountColumns(const PlanNode& node) {
    std::vector<std::vector<bool>> below;
    for (const auto& child : node.children) below.push_back(CountColumns(*child));
    return NodeCountColumns(node, below);
}

// Retypes node and its subtree's COUNT columns BIGINT; adds how many to retyped.
inline std::vector<bool> RetypeCountColumns(PlanNode& node, size_t& retyped) {
    std::vector<std::vector<bool>> below;
    for (auto& child : node.children) below.push_back(RetypeCountColumns(*child, retyped));
    const std::vector<bool> counts = NodeCountColumns(node, below);
    std::vector<Column> columns = node.output_schema.GetColumns();
    for (size_t c = 0; c < columns.size(); ++c) {
        if (!counts[c]) continue;
        columns[c] = Column(columns[c].name, TypeId::BIGINT);
        ++retyped;
    }
    node.output_schema = Schema(std::move(columns));
    return counts;
}

// Turns a TUPLE or VECTORIZED plan into the types a parallel plan gives it.
// Returns how many columns it retyped, across every node.
inline size_t RetypeCountColumns(PlanNode& root) {
    size_t retyped = 0;
    (void)RetypeCountColumns(root, retyped);
    return retyped;
}

}  // namespace shilmandb::test
