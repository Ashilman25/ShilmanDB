#include "planner/planner.hpp"
#include "planner/join_order_optimizer.hpp"
#include "planner/index_selector.hpp"
#include "common/exception.hpp"
#include "types/value.hpp"
#include <algorithm>
#include <unordered_set>

#ifdef SHILMANDB_HAS_LIBTORCH
#include "planner/learned_join_optimizer.hpp"
#endif

namespace shilmandb {

namespace {


auto SplitConjuncts(std::unique_ptr<Expression> expr) -> std::vector<std::unique_ptr<Expression>> {
    std::vector<std::unique_ptr<Expression>> result;
    if (!expr) return result;

    auto* bin = dynamic_cast<BinaryOp*>(expr.get());
    if (bin && bin->op == BinaryOp::Op::AND) {
        auto left_parts = SplitConjuncts(std::move(bin->left));
        auto right_parts = SplitConjuncts(std::move(bin->right));

        for (auto& p : left_parts) result.push_back(std::move(p));
        for (auto& p : right_parts) result.push_back(std::move(p));
    } else {
        result.push_back(std::move(expr));
    }
    return result;
}


auto ResolveUnqualifiedColumn(const std::string& col_name, const std::vector<TableRef>& tables, const std::vector<TableInfo*>& table_infos) -> int {
    int found = -1;
    for (size_t i = 0; i < tables.size(); ++i) {
        const auto& schema = table_infos[i]->schema;
        for (uint32_t c = 0; c < schema.GetColumnCount(); ++c) {
            if (schema.GetColumn(c).name == col_name) {
                if (found >= 0) return -1;  // ambiguous
                found = static_cast<int>(i);
                break;
            }
        }
    }
    return found;
}

void CollectTableIndices(const Expression* expr, const std::vector<TableRef>& tables, const std::vector<TableInfo*>& table_infos, std::unordered_set<int>& indices) {
    if (!expr) return;

    if (const auto* col = dynamic_cast<const ColumnRef*>(expr)) {
        if (col->table_name.has_value()) {
            const auto& name = col->table_name.value();

            for (size_t i = 0; i < tables.size(); ++i) {
                if ((tables[i].alias.has_value() && tables[i].alias.value() == name) || tables[i].table_name == name) {
                    indices.insert(static_cast<int>(i));
                    break;
                }
            }
        } else {
            int idx = ResolveUnqualifiedColumn(col->column_name, tables, table_infos);
            if (idx >= 0) indices.insert(idx);
        }
    } else if (const auto* bin = dynamic_cast<const BinaryOp*>(expr)) {
        CollectTableIndices(bin->left.get(), tables, table_infos, indices);
        CollectTableIndices(bin->right.get(), tables, table_infos, indices);

    } else if (const auto* un = dynamic_cast<const UnaryOp*>(expr)) {
        CollectTableIndices(un->operand.get(), tables, table_infos, indices);

    } else if (const auto* agg = dynamic_cast<const Aggregate*>(expr)) {
        CollectTableIndices(agg->arg.get(), tables, table_infos, indices);
    }
}

auto CombinePredicates(std::vector<std::unique_ptr<Expression>>& preds) -> std::unique_ptr<Expression> {
    if (preds.empty()) return nullptr;
    if (preds.size() == 1) return std::move(preds[0]);

    auto combined = std::move(preds.back());
    for (int i = static_cast<int>(preds.size()) - 2; i >= 0; --i) {
        auto and_node = std::make_unique<BinaryOp>();

        and_node->op = BinaryOp::Op::AND;
        and_node->left = std::move(preds[i]);
        and_node->right = std::move(combined);

        combined = std::move(and_node);
    }
    return combined;
}


auto BuildJoinSchema(const Schema& left, const Schema& right) -> Schema {
    std::vector<Column> columns;
    columns.reserve(left.GetColumnCount() + right.GetColumnCount());

    for (uint32_t i = 0; i < left.GetColumnCount(); ++i) {
        columns.push_back(left.GetColumn(i));
    }

    for (uint32_t i = 0; i < right.GetColumnCount(); ++i) {
        columns.push_back(right.GetColumn(i));
    }
    return Schema(std::move(columns));
}


auto IsEquiJoin(const Expression* expr) -> bool {
    if (!expr) return false;
    const auto* bin = dynamic_cast<const BinaryOp*>(expr);
    if (!bin) return false;

    if (bin->op == BinaryOp::Op::AND) {
        return IsEquiJoin(bin->left.get()) && IsEquiJoin(bin->right.get());
    }
    if (bin->op == BinaryOp::Op::EQ) {
        return dynamic_cast<const ColumnRef*>(bin->left.get()) != nullptr && dynamic_cast<const ColumnRef*>(bin->right.get()) != nullptr;
    }
    return false;
}


auto DeriveColumnName(const Expression* expr) -> std::string {
    if (const auto* col = dynamic_cast<const ColumnRef*>(expr)) {
        return col->column_name;
    }
    if (const auto* agg = dynamic_cast<const Aggregate*>(expr)) {
        std::string prefix;
        switch (agg->func) {
            case Aggregate::Func::COUNT: 
                prefix = "count"; 
                break;
            case Aggregate::Func::SUM:   
                prefix = "sum"; 
                break;
            case Aggregate::Func::AVG:   
                prefix = "avg"; 
                break;
            case Aggregate::Func::MIN:   
                prefix = "min"; 
                break;
            case Aggregate::Func::MAX:   
                prefix = "max"; 
                break;
        }

        if (!agg->arg) return prefix + "_star";
        if (const auto* arg_col = dynamic_cast<const ColumnRef*>(agg->arg.get())) {
            return prefix + "_" + arg_col->column_name;
        }
        return prefix + "_expr";
    }
    return "expr";
}


bool ExpressionEquals(const Expression* a, const Expression* b) {
    if (a == b) return true;
    if (!a || !b) return false;
    if (a->type != b->type) return false;

    switch (a->type) {
        case ExprType::COLUMN_REF: {
            const auto* ac = static_cast<const ColumnRef*>(a);
            const auto* bc = static_cast<const ColumnRef*>(b);
            return ac->table_name == bc->table_name && ac->column_name == bc->column_name;
        }
        case ExprType::LITERAL: {
            const auto* al = static_cast<const Literal*>(a);
            const auto* bl = static_cast<const Literal*>(b);
            return al->value == bl->value;
        }
        case ExprType::BINARY_OP: {
            const auto* ab = static_cast<const BinaryOp*>(a);
            const auto* bb = static_cast<const BinaryOp*>(b);
            return ab->op == bb->op && ExpressionEquals(ab->left.get(), bb->left.get()) && ExpressionEquals(ab->right.get(), bb->right.get());
        }
        case ExprType::UNARY_OP: {
            const auto* au = static_cast<const UnaryOp*>(a);
            const auto* bu = static_cast<const UnaryOp*>(b);
            return au->op == bu->op && ExpressionEquals(au->operand.get(), bu->operand.get()); 
        }
        case ExprType::AGGREGATE: {
            const auto* aa = static_cast<const Aggregate*>(a);
            const auto* ba = static_cast<const Aggregate*>(b);
            return aa->func == ba->func && ExpressionEquals(aa->arg.get(), ba->arg.get());
        }
        case ExprType::STAR:
            return true;
        case ExprType::CASE: {
            const auto* ac = static_cast<const CaseExpression*>(a);
            const auto* bc = static_cast<const CaseExpression*>(b);
            if (ac->when_clauses.size() != bc->when_clauses.size()) return false;
            for (size_t i = 0; i < ac->when_clauses.size(); ++i) {
                if (!ExpressionEquals(ac->when_clauses[i].first.get(), bc->when_clauses[i].first.get())) return false;
                if (!ExpressionEquals(ac->when_clauses[i].second.get(), bc->when_clauses[i].second.get())) return false;
            }
            return ExpressionEquals(ac->else_clause.get(), bc->else_clause.get());
        }
    }
    return false;
}

// Tree-walking aggregate detector. Returns true if `expr` contains an
// Aggregate node anywhere in its tree.
bool HasAggregate(const Expression* expr) {
    if (!expr) return false;
    if (dynamic_cast<const Aggregate*>(expr)) return true;

    if (const auto* bin = dynamic_cast<const BinaryOp*>(expr)) {
        return HasAggregate(bin->left.get()) || HasAggregate(bin->right.get());
    }
    if (const auto* un = dynamic_cast<const UnaryOp*>(expr)) {
        return HasAggregate(un->operand.get());
    }
    if (const auto* ce = dynamic_cast<const CaseExpression*>(expr)) {
        for (const auto& [cond, result] : ce->when_clauses) {
            if (HasAggregate(cond.get()) || HasAggregate(result.get())) return true;
        }
        return HasAggregate(ce->else_clause.get());
    }
    return false;
}

// In-place tree rewrite. For every Aggregate node reachable from `expr`
// so like select sum, count, etc
void ExtractAggregates(std::unique_ptr<Expression>& expr, std::vector<std::unique_ptr<Expression>>& agg_exprs, std::vector<Aggregate::Func>& agg_funcs, std::vector<std::string>& agg_out_names) {
    if (!expr) return;

    if (auto* agg = dynamic_cast<Aggregate*>(expr.get())) {
        for (size_t i = 0; i < agg_funcs.size(); ++i) {
            if (agg_funcs[i] == agg->func && ExpressionEquals(agg_exprs[i].get(), agg->arg.get())) {
                auto cr = std::make_unique<ColumnRef>();
                cr->column_name = agg_out_names[i];
                expr = std::move(cr);
                return;
            }
        }

        std::string prefix;
        switch (agg->func) {
            case Aggregate::Func::COUNT: 
                prefix = "count"; 
                break;
            case Aggregate::Func::SUM:   
                prefix = "sum";   
                break;
            case Aggregate::Func::AVG:   
                prefix = "avg";   
                break;
            case Aggregate::Func::MIN:   
                prefix = "min";   
                break;
            case Aggregate::Func::MAX:   
                prefix = "max";   
                break;
        }

        std::string out_name;
        if (!agg->arg) {
            out_name = prefix + "_star";
        } else if (const auto* col = dynamic_cast<const ColumnRef*>(agg->arg.get())) {
            out_name = prefix + "_" + col->column_name;
        } else {
            out_name = prefix + "_expr_" + std::to_string(agg_exprs.size());
        }

        agg_funcs.push_back(agg->func);
        agg_exprs.push_back(agg->arg ? agg->arg->Clone() : nullptr);
        agg_out_names.push_back(out_name);

        auto cr = std::make_unique<ColumnRef>();
        cr->column_name = out_name;
        expr = std::move(cr);
        return;
    }

    if (auto* bin = dynamic_cast<BinaryOp*>(expr.get())) {
        ExtractAggregates(bin->left, agg_exprs, agg_funcs, agg_out_names);
        ExtractAggregates(bin->right, agg_exprs, agg_funcs, agg_out_names);
        return;
    }
    if (auto* un = dynamic_cast<UnaryOp*>(expr.get())) {
        ExtractAggregates(un->operand, agg_exprs, agg_funcs, agg_out_names);
        return;
    }
    if (auto* ce = dynamic_cast<CaseExpression*>(expr.get())) {
        for (auto& [cond, result] : ce->when_clauses) {
            ExtractAggregates(cond, agg_exprs, agg_funcs, agg_out_names);
            ExtractAggregates(result, agg_exprs, agg_funcs, agg_out_names);
        }
        ExtractAggregates(ce->else_clause, agg_exprs, agg_funcs, agg_out_names);
        return;
    }
}

auto ResolveExprType(const Expression* expr, const Schema& schema) -> TypeId {
    if (const auto* col = dynamic_cast<const ColumnRef*>(expr)) {
        for (uint32_t i = 0; i < schema.GetColumnCount(); ++i) {
            if (schema.GetColumn(i).name == col->column_name) {
                return schema.GetColumn(i).type;
            }
        }
        return TypeId::INTEGER;
    }
    
    if (const auto* lit = dynamic_cast<const Literal*>(expr)) {
        return lit->value.type_;
    }
    if (const auto* agg = dynamic_cast<const Aggregate*>(expr)) {
        switch (agg->func) {
            case Aggregate::Func::COUNT:
                return TypeId::INTEGER;
            case Aggregate::Func::SUM:
                return TypeId::DECIMAL;
            case Aggregate::Func::AVG:
                return TypeId::DECIMAL;
            case Aggregate::Func::MIN:
            case Aggregate::Func::MAX:
                return agg->arg ? ResolveExprType(agg->arg.get(), schema) : TypeId::INTEGER;
        }
    }
    if (const auto* ce = dynamic_cast<const CaseExpression*>(expr)) {
        if (ce->when_clauses.empty()) return TypeId::INTEGER;  // guarded by parser

        TypeId resolved = ResolveExprType(ce->when_clauses[0].second.get(), schema);
        for (size_t i = 1; i < ce->when_clauses.size(); ++i) {
            auto t = ResolveExprType(ce->when_clauses[i].second.get(), schema);
            auto merged = CommonType(resolved, t);

            if (merged == TypeId::INVALID) {
                return ResolveExprType(ce->when_clauses[0].second.get(), schema);
            }
            resolved = merged;
        }

        if (ce->else_clause) {
            auto t = ResolveExprType(ce->else_clause.get(), schema);
            auto merged = CommonType(resolved, t);
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
                return TypeId::INTEGER;  // boolean result
            case Op::ADD: case Op::SUB: case Op::MUL: case Op::DIV: {
                auto lt = ResolveExprType(bin->left.get(), schema);
                auto rt = ResolveExprType(bin->right.get(), schema);
                auto merged = CommonType(lt, rt);
                return merged == TypeId::INVALID ? lt : merged;
            }
        }
    }
    if (const auto* un = dynamic_cast<const UnaryOp*>(expr)) {
        if (un->op == UnaryOp::Op::NOT) return TypeId::INTEGER;
        return ResolveExprType(un->operand.get(), schema);
    }
    return TypeId::INTEGER;
}

}  // anonymous namespace

Planner::Planner(Catalog* catalog) : catalog_(catalog) {}

#ifdef SHILMANDB_HAS_LIBTORCH
Planner::Planner(Catalog* catalog, LearnedJoinOptimizer* learned_optimizer) : catalog_(catalog), learned_optimizer_(learned_optimizer) {}
#endif

auto Planner::Plan(SelectStatement stmt) -> std::unique_ptr<PlanNode> {
    std::vector<TableRef> tables;
    std::vector<TableInfo*> table_infos;

    for (const auto& tref : stmt.from_clause) {
        tables.push_back({tref.table_name, tref.alias});
    }
    for (const auto& join : stmt.joins) {
        tables.push_back({join.right_table.table_name, join.right_table.alias});
    }

    for (const auto& tref : tables) {
        auto* info = catalog_->GetTable(tref.table_name);
        if (!info) {
            throw DatabaseException("Table not found: " + tref.table_name);
        }
        table_infos.push_back(info);
    }

    if (tables.empty()) {
        throw DatabaseException("No tables in FROM clause");
    }

    if (stmt.select_list.empty()) {
        throw DatabaseException("Empty select list");
    }

    std::vector<std::vector<std::unique_ptr<Expression>>> table_predicates(tables.size());
    std::vector<std::unique_ptr<Expression>> residual_predicates;

    if (stmt.where_clause) {
        auto conjuncts = SplitConjuncts(std::move(stmt.where_clause));

        for (auto& conj : conjuncts) {
            std::unordered_set<int> refs;
            CollectTableIndices(conj.get(), tables, table_infos, refs);

            if (refs.size() == 1) {
                table_predicates[*refs.begin()].push_back(std::move(conj));
            } else {
                residual_predicates.push_back(std::move(conj));
            }
        }
    }

    std::vector<std::unique_ptr<PlanNode>> scans;
    scans.reserve(tables.size());
    for (size_t i = 0; i < tables.size(); ++i) {
        std::vector<std::unique_ptr<Expression>> residual;
        auto scan = IndexSelector::SelectScanStrategy(tables[i].table_name, table_infos[i]->schema, table_predicates[i], residual, catalog_);

        if (!residual.empty()) {
            auto pred = CombinePredicates(residual);
            auto filter = std::make_unique<FilterPlanNode>(scan->output_schema, std::move(pred));
            filter->children.push_back(std::move(scan));
            scan = std::move(filter);
        }
        scans.push_back(std::move(scan));
    }

    //join tree
    std::unique_ptr<PlanNode> current;

    if (tables.size() <= 1) {
        current = std::move(scans[0]);
    } else {
        std::vector<TableStats> stats;
        stats.reserve(table_infos.size());

        for (const auto* info : table_infos) {
            stats.push_back(info->stats);
        }

        std::vector<int> order;

#ifdef SHILMANDB_HAS_LIBTORCH
        if (learned_optimizer_ != nullptr) {
            try {
                auto features = JoinOrderOptimizer::BuildFeatureVector(tables, stmt.joins, stats);
                auto learned_order = learned_optimizer_->PredictJoinOrder(features, static_cast<int>(tables.size()));

                double learned_cost = JoinOrderOptimizer::EstimateCost(learned_order, tables, stmt.joins, stats);
                auto exhaustive_order = JoinOrderOptimizer::FindBestOrder(tables, stmt.joins, stats);
                double exhaustive_cost = JoinOrderOptimizer::EstimateCost(exhaustive_order, tables, stmt.joins, stats);

                if (learned_cost <= 1.5 * exhaustive_cost) {
                    order = std::move(learned_order);
                } else {
                    order = std::move(exhaustive_order);
                }
            } catch (const std::exception&) {
                order = JoinOrderOptimizer::FindBestOrder(tables, stmt.joins, stats);
            }
        } else {
            order = JoinOrderOptimizer::FindBestOrder(tables, stmt.joins, stats);
        }
#else
        order = JoinOrderOptimizer::FindBestOrder(tables, stmt.joins, stats);
#endif

        current = std::move(scans[order[0]]);

        std::unordered_set<int> joined_set;
        joined_set.insert(order[0]);

        for (size_t k = 1; k < order.size(); ++k) {
            const int new_idx = order[k];

            std::vector<std::unique_ptr<Expression>> join_preds;
            for (auto& join : stmt.joins) {
                if (!join.on_condition) continue;

                std::unordered_set<int> refs;
                CollectTableIndices(join.on_condition.get(), tables, table_infos, refs);

                bool refs_new = refs.find(new_idx) != refs.end();
                bool refs_joined = std::any_of(refs.begin(), refs.end(), [&](int r) {
                    return joined_set.find(r) != joined_set.end();
                });

                if (refs_new && refs_joined) {
                    join_preds.push_back(std::move(join.on_condition));
                    break;
                }
            }

            for (auto it = residual_predicates.begin(); it != residual_predicates.end();) {
                std::unordered_set<int> refs;
                CollectTableIndices(it->get(), tables, table_infos, refs);

                bool refs_new = refs.find(new_idx) != refs.end();
                bool refs_joined = std::any_of(refs.begin(), refs.end(), [&](int r) {
                    return joined_set.find(r) != joined_set.end();
                });

                bool refs_only_joined_or_new = std::all_of(refs.begin(), refs.end(), [&](int r) {
                    return r == new_idx || joined_set.find(r) != joined_set.end();
                });

                if (refs_new && refs_joined && refs_only_joined_or_new) {
                    join_preds.push_back(std::move(*it));
                    it = residual_predicates.erase(it);
                } else {
                    ++it;
                }
            }

            joined_set.insert(new_idx);

            auto join_schema = BuildJoinSchema(current->output_schema, scans[new_idx]->output_schema);

            std::vector<std::unique_ptr<Expression>> equi_preds;
            std::vector<std::unique_ptr<Expression>> post_filter_preds;
            for (auto& p : join_preds) {
                if (IsEquiJoin(p.get())) {
                    equi_preds.push_back(std::move(p));
                } else {
                    post_filter_preds.push_back(std::move(p));
                }
            }

            std::unique_ptr<Expression> equi_pred = CombinePredicates(equi_preds);
            std::unique_ptr<PlanNode> join_node;
            auto post_filter = CombinePredicates(post_filter_preds);

            if (equi_pred) {
                auto schema_for_join = join_schema;
                join_node = std::make_unique<HashJoinPlanNode>(std::move(join_schema), std::move(equi_pred));
                join_node->children.push_back(std::move(current));
                join_node->children.push_back(std::move(scans[new_idx]));
                if (post_filter) {
                    auto filter = std::make_unique<FilterPlanNode>(std::move(schema_for_join), std::move(post_filter));
                    filter->children.push_back(std::move(join_node));
                    current = std::move(filter);
                } else {
                    current = std::move(join_node);
                }
            } else {
                join_node = std::make_unique<NestedLoopJoinPlanNode>(std::move(join_schema), std::move(post_filter));
                join_node->children.push_back(std::move(current));
                join_node->children.push_back(std::move(scans[new_idx]));
                current = std::move(join_node);
            }
        }

        if (!residual_predicates.empty()) {
            auto residual = CombinePredicates(residual_predicates);
            auto filter = std::make_unique<FilterPlanNode>(current->output_schema, std::move(residual));
            filter->children.push_back(std::move(current));
            current = std::move(filter);
        }
    }

    auto has_aggregate_in_select = [&]() {
        return std::any_of(stmt.select_list.begin(), stmt.select_list.end(),
            [](const SelectItem& item) {
                return HasAggregate(item.expr.get());
            });
    };
    auto has_aggregate_in_order_by = [&]() {
        return std::any_of(stmt.order_by.begin(), stmt.order_by.end(),
            [](const OrderByItem& item) {
                return HasAggregate(item.expr.get());
            });
    };

    if (!stmt.group_by.empty() || has_aggregate_in_select() || has_aggregate_in_order_by()) {
        std::vector<std::unique_ptr<Expression>> agg_exprs;
        std::vector<Aggregate::Func> agg_funcs;
        std::vector<std::string> agg_out_names;

        // In-place rewrite: aggregates inside SELECT/ORDER BY expressions are
        // replaced with ColumnRefs pointing at the AggregatePlanNode's output
        // columns. ORDER BY is walked here (not after projection) so the Sort
        // node above it can resolve the column names against the agg schema.
        for (auto& item : stmt.select_list) {
            ExtractAggregates(item.expr, agg_exprs, agg_funcs, agg_out_names);
        }
        for (auto& ob_item : stmt.order_by) {
            ExtractAggregates(ob_item.expr, agg_exprs, agg_funcs, agg_out_names);
        }

        std::vector<Column> agg_columns;

        for (const auto& gb_expr : stmt.group_by) {
            if (const auto* col = dynamic_cast<const ColumnRef*>(gb_expr.get())) {
                for (uint32_t c = 0; c < current->output_schema.GetColumnCount(); ++c) {
                    if (current->output_schema.GetColumn(c).name == col->column_name) {
                        agg_columns.push_back(current->output_schema.GetColumn(c));
                        break;
                    }
                }
            }
        }

        for (size_t i = 0; i < agg_funcs.size(); ++i) {
            TypeId col_type = TypeId::INTEGER;
            switch (agg_funcs[i]) {
                case Aggregate::Func::COUNT:
                    col_type = TypeId::INTEGER;
                    break;
                case Aggregate::Func::SUM:
                case Aggregate::Func::AVG:
                    col_type = TypeId::DECIMAL;
                    break;
                case Aggregate::Func::MIN:
                case Aggregate::Func::MAX:
                    col_type = agg_exprs[i]
                        ? ResolveExprType(agg_exprs[i].get(), current->output_schema)
                        : TypeId::INTEGER;
                    break;
            }
            agg_columns.emplace_back(agg_out_names[i], col_type);
        }

        Schema agg_schema(std::move(agg_columns));
        auto agg_node = std::make_unique<AggregatePlanNode>(std::move(agg_schema));

        agg_node->group_by_exprs = std::move(stmt.group_by);
        agg_node->aggregate_exprs = std::move(agg_exprs);
        agg_node->aggregate_funcs = std::move(agg_funcs);
        agg_node->children.push_back(std::move(current));

        current = std::move(agg_node);
    }

    if (stmt.having) {
        auto having_filter = std::make_unique<FilterPlanNode>(current->output_schema, std::move(stmt.having));
        having_filter->children.push_back(std::move(current));
        current = std::move(having_filter);
    }

    {
        std::vector<Column> proj_columns;
        std::vector<std::unique_ptr<Expression>> proj_exprs;

        for (auto& item : stmt.select_list) {
            if (dynamic_cast<StarExpr*>(item.expr.get())) {
                for (uint32_t c = 0; c < current->output_schema.GetColumnCount(); ++c) {
                    const auto& col = current->output_schema.GetColumn(c);
                    proj_columns.push_back(col);

                    auto cr = std::make_unique<ColumnRef>();
                    cr->column_name = col.name;
                    proj_exprs.push_back(std::move(cr));
                }

            } else {
                std::string col_name = item.alias.has_value()
                    ? item.alias.value()
                    : DeriveColumnName(item.expr.get());
                auto col_type = ResolveExprType(item.expr.get(), current->output_schema);

                proj_columns.emplace_back(col_name, col_type);
                proj_exprs.push_back(std::move(item.expr));
            }
        }

        Schema proj_schema(std::move(proj_columns));
        auto proj = std::make_unique<ProjectionPlanNode>(std::move(proj_schema), std::move(proj_exprs));

        proj->children.push_back(std::move(current));
        current = std::move(proj);
    }

    if (!stmt.order_by.empty()) {
        // ORDER BY aggregates were already extracted into ColumnRefs during
        // the aggregate-extraction pass above. Those ColumnRefs must resolve
        // to columns present in the projection output — when an ORDER BY
        // aggregate matches one in SELECT, the shared name ensures both
        // resolve to the same column.
        auto sort = std::make_unique<SortPlanNode>(current->output_schema, std::move(stmt.order_by));
        sort->children.push_back(std::move(current));
        current = std::move(sort);
    }


    if (stmt.limit.has_value()) {
        auto limit_node = std::make_unique<LimitPlanNode>(current->output_schema, stmt.limit.value());
        limit_node->children.push_back(std::move(current));
        current = std::move(limit_node);
    }

    return current;
}

}  // namespace shilmandb
