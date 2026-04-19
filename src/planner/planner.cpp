#include "planner/planner.hpp"
#include "planner/join_order_optimizer.hpp"
#include "planner/index_selector.hpp"
#include "common/exception.hpp"
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

            std::unique_ptr<Expression> join_pred;
            for (auto& join : stmt.joins) {
                if (!join.on_condition) continue;

                std::unordered_set<int> refs;
                CollectTableIndices(join.on_condition.get(), tables, table_infos, refs);

                bool refs_new = refs.find(new_idx) != refs.end();
                bool refs_joined = std::any_of(refs.begin(), refs.end(), [&](int r) {
                    return joined_set.find(r) != joined_set.end();
                });

                if (refs_new && refs_joined) {
                    join_pred = std::move(join.on_condition);
                    break;
                }
            }

            joined_set.insert(new_idx);

            auto join_schema = BuildJoinSchema(current->output_schema, scans[new_idx]->output_schema);

            std::unique_ptr<PlanNode> join_node;
            if (join_pred && IsEquiJoin(join_pred.get())) {
                join_node = std::make_unique<HashJoinPlanNode>(std::move(join_schema), std::move(join_pred));
            } else {
                join_node = std::make_unique<NestedLoopJoinPlanNode>(std::move(join_schema), std::move(join_pred));
            }

            join_node->children.push_back(std::move(current));
            join_node->children.push_back(std::move(scans[new_idx]));
            current = std::move(join_node);
        }

        if (!residual_predicates.empty()) {
            auto residual = CombinePredicates(residual_predicates);
            auto filter = std::make_unique<FilterPlanNode>(current->output_schema, std::move(residual));
            filter->children.push_back(std::move(current));
            current = std::move(filter);
        }
    }

    auto has_aggregate_in_select = [&]() {
        return std::any_of(stmt.select_list.begin(), stmt.select_list.end(),[](const SelectItem& item) {
            return dynamic_cast<const Aggregate*>(item.expr.get()) != nullptr;
        });
    };

    if (!stmt.group_by.empty() || has_aggregate_in_select()) {
        std::vector<std::unique_ptr<Expression>> agg_exprs;
        std::vector<Aggregate::Func> agg_funcs;

        for (const auto& item : stmt.select_list) {
            if (const auto* agg = dynamic_cast<const Aggregate*>(item.expr.get())) {
                agg_funcs.push_back(agg->func);

                if (agg->arg) {
                    agg_exprs.push_back(agg->arg->Clone());
                } else {
                    agg_exprs.push_back(nullptr);  // COUNT(*)
                }
            }
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
            std::string col_name;
            TypeId col_type = TypeId::INTEGER;

            switch (agg_funcs[i]) {
                case Aggregate::Func::COUNT:
                    col_type = TypeId::INTEGER;
                    if (agg_exprs[i]) {
                        col_name = "count_" + DeriveColumnName(agg_exprs[i].get());
                    } else {
                        col_name = "count_star";
                    }
                    break;
                case Aggregate::Func::SUM:
                    col_type = TypeId::DECIMAL;
                    col_name = "sum_" + (agg_exprs[i] ? DeriveColumnName(agg_exprs[i].get()) : "expr");
                    break;
                case Aggregate::Func::AVG:
                    col_type = TypeId::DECIMAL;
                    col_name = "avg_" + (agg_exprs[i] ? DeriveColumnName(agg_exprs[i].get()) : "expr");
                    break;
                case Aggregate::Func::MIN:
                    col_name = "min_" + (agg_exprs[i] ? DeriveColumnName(agg_exprs[i].get()) : "expr");
                    col_type = agg_exprs[i] ? ResolveExprType(agg_exprs[i].get(), current->output_schema) : TypeId::INTEGER;
                    break;
                case Aggregate::Func::MAX:
                    col_name = "max_" + (agg_exprs[i] ? DeriveColumnName(agg_exprs[i].get()) : "expr");
                    col_type = agg_exprs[i] ? ResolveExprType(agg_exprs[i].get(), current->output_schema) : TypeId::INTEGER;
                    break;
            }
            agg_columns.emplace_back(col_name, col_type);
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
                std::string col_name = item.alias.has_value() ? item.alias.value() : DeriveColumnName(item.expr.get());
                auto col_type = ResolveExprType(item.expr.get(), current->output_schema);

                proj_columns.emplace_back(col_name, col_type);

                if (dynamic_cast<Aggregate*>(item.expr.get())) {
                    auto cr = std::make_unique<ColumnRef>();
                    cr->column_name = DeriveColumnName(item.expr.get());
                    proj_exprs.push_back(std::move(cr));
                } else {
                    proj_exprs.push_back(std::move(item.expr));
                }
            }
        }

        Schema proj_schema(std::move(proj_columns));
        auto proj = std::make_unique<ProjectionPlanNode>(std::move(proj_schema), std::move(proj_exprs));

        proj->children.push_back(std::move(current));
        current = std::move(proj);
    }

    if (!stmt.order_by.empty()) {
        for (auto& ob_item : stmt.order_by) {
            if (const auto* agg = dynamic_cast<const Aggregate*>(ob_item.expr.get())) {
                auto cr = std::make_unique<ColumnRef>();
                cr->column_name = DeriveColumnName(agg);
                ob_item.expr = std::move(cr);
            }
        }
        
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
