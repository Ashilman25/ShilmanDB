#include "planner/or_factoring.hpp"

#include <algorithm>
#include <utility>

namespace shilmandb {

namespace {

using TablesOf = std::function<std::unordered_set<int>(const Expression&)>;
using Parts = std::vector<const Expression*>;

// Both missing, or both there and the same.
bool SameChild(const Expression* a, const Expression* b) {
    if (a == nullptr || b == nullptr) return a == b;
    return SameExpression(*a, *b);
}

bool IsOp(const Expression* expr, BinaryOp::Op op) {
    return expr != nullptr && expr->type == ExprType::BINARY_OP && static_cast<const BinaryOp*>(expr)->op == op;
}

// The leaves under a chain of `op` nodes, left first.
void Flatten(const Expression* expr, BinaryOp::Op op, Parts& out) {
    if (IsOp(expr, op)) {
        const auto* bin = static_cast<const BinaryOp*>(expr);
        Flatten(bin->left.get(), op, out);
        Flatten(bin->right.get(), op, out);
    } else if (expr != nullptr) {
        out.push_back(expr);
    }
}

// True when a CASE sits anywhere in the tree.
bool HoldsCase(const Expression* expr) {
    if (expr == nullptr) return false;
    switch (expr->type) {
        case ExprType::CASE:
            return true;
        case ExprType::BINARY_OP: {
            const auto* bin = static_cast<const BinaryOp*>(expr);
            return HoldsCase(bin->left.get()) || HoldsCase(bin->right.get());
        }
        case ExprType::UNARY_OP:
            return HoldsCase(static_cast<const UnaryOp*>(expr)->operand.get());
        case ExprType::AGGREGATE:
            return HoldsCase(static_cast<const Aggregate*>(expr)->arg.get());
        default:
            return false;
    }
}

// True when every column was placed on a table (tables_of marks the rest with -1).
bool AllPlaced(const std::unordered_set<int>& tables) {
    return std::none_of(tables.begin(), tables.end(), [](int t) { return t < 0; });
}

bool Contains(const Parts& parts, const Expression& expr) {
    return std::any_of(parts.begin(), parts.end(), [&](const Expression* p) { return SameExpression(*p, expr); });
}

// Clones of parts joined by op, nested to the left as the parser builds chains.
std::unique_ptr<Expression> Chain(std::vector<std::unique_ptr<Expression>> parts, BinaryOp::Op op) {
    std::unique_ptr<Expression> out = std::move(parts.front());
    for (size_t i = 1; i < parts.size(); ++i) {
        auto node = std::make_unique<BinaryOp>();
        node->op = op;
        node->left = std::move(out);
        node->right = std::move(parts[i]);
        out = std::move(node);
    }
    return out;
}

// What `conj` implies, passed to add: shared conjuncts, then one OR per table.
void AddImplied(const Expression& conj, const TablesOf& tables_of, const std::function<void(std::unique_ptr<Expression>)>& add) {
    Parts roots;
    Flatten(&conj, BinaryOp::Op::OR, roots);
    std::vector<Parts> branches(roots.size());
    for (size_t i = 0; i < roots.size(); ++i) Flatten(roots[i], BinaryOp::Op::AND, branches[i]);

    // Shared: in every branch, on placed tables, no CASE.
    // Once each, in the first branch's order.
    Parts shared;
    for (const Expression* part : branches.front()) {
        if (Contains(shared, *part) || HoldsCase(part)) continue;
        const auto tables = tables_of(*part);
        if (tables.empty() || !AllPlaced(tables)) continue;
        const bool everywhere = std::all_of(branches.begin() + 1, branches.end(), [&](const Parts& branch) { return Contains(branch, *part); });
        if (!everywhere) continue;
        shared.push_back(part);
        add(part->Clone());
    }

    // An OR on one table already filters that table; a weaker copy adds nothing.
    const auto or_tables = tables_of(conj);
    if (std::count_if(or_tables.begin(), or_tables.end(), [](int t) { return t >= 0; }) < 2) return;

    // The one placed table a non-shared, CASE-free part filters, or -1.
    const auto table_of = [&](const Expression& part) {
        if (Contains(shared, part) || HoldsCase(&part)) return -1;
        const auto tables = tables_of(part);
        return tables.size() == 1 && *tables.begin() >= 0 ? *tables.begin() : -1;
    };

    std::vector<int> candidates;
    for (const Expression* part : branches.front()) {
        const int table = table_of(*part);
        if (table >= 0 && std::find(candidates.begin(), candidates.end(), table) == candidates.end()) {
            candidates.push_back(table);
        }
    }

    // Per table: only when every branch filters it on its own.
    for (const int table : candidates) {
        std::vector<std::unique_ptr<Expression>> per_branch;
        for (const Parts& branch : branches) {
            std::vector<std::unique_ptr<Expression>> mine;
            for (const Expression* part : branch) {
                if (table_of(*part) == table) mine.push_back(part->Clone());
            }
            if (mine.empty()) break;
            per_branch.push_back(Chain(std::move(mine), BinaryOp::Op::AND));
        }
        if (per_branch.size() == branches.size()) add(Chain(std::move(per_branch), BinaryOp::Op::OR));
    }
}

}  // namespace

bool SameExpression(const Expression& a, const Expression& b) {
    if (&a == &b) return true;
    if (a.type != b.type) return false;

    switch (a.type) {
        case ExprType::COLUMN_REF: {
            const auto& ac = static_cast<const ColumnRef&>(a);
            const auto& bc = static_cast<const ColumnRef&>(b);
            return ac.table_name == bc.table_name && ac.column_name == bc.column_name;
        }
        case ExprType::LITERAL: {
            // Types first: Value's == throws across types it can't compare.
            const auto& al = static_cast<const Literal&>(a);
            const auto& bl = static_cast<const Literal&>(b);
            return al.value.type_ == bl.value.type_ && al.value == bl.value && al.text == bl.text;
        }
        case ExprType::BINARY_OP: {
            const auto& ab = static_cast<const BinaryOp&>(a);
            const auto& bb = static_cast<const BinaryOp&>(b);
            return ab.op == bb.op && SameChild(ab.left.get(), bb.left.get()) && SameChild(ab.right.get(), bb.right.get());
        }
        case ExprType::UNARY_OP: {
            const auto& au = static_cast<const UnaryOp&>(a);
            const auto& bu = static_cast<const UnaryOp&>(b);
            return au.op == bu.op && SameChild(au.operand.get(), bu.operand.get());
        }
        case ExprType::AGGREGATE: {
            const auto& aa = static_cast<const Aggregate&>(a);
            const auto& ba = static_cast<const Aggregate&>(b);
            return aa.func == ba.func && SameChild(aa.arg.get(), ba.arg.get());
        }
        case ExprType::STAR:
            return true;
        case ExprType::CASE: {
            const auto& ac = static_cast<const CaseExpression&>(a);
            const auto& bc = static_cast<const CaseExpression&>(b);
            if (ac.when_clauses.size() != bc.when_clauses.size()) return false;
            for (size_t i = 0; i < ac.when_clauses.size(); ++i) {
                if (!SameChild(ac.when_clauses[i].first.get(), bc.when_clauses[i].first.get())) return false;
                if (!SameChild(ac.when_clauses[i].second.get(), bc.when_clauses[i].second.get())) return false;
            }
            return SameChild(ac.else_clause.get(), bc.else_clause.get());
        }
    }
    return false;
}

std::vector<std::unique_ptr<Expression>> FactorOrConjuncts(std::vector<std::unique_ptr<Expression>> conjuncts, const TablesOf& tables_of) {
    // Every original conjunct and every addition so far, so nothing is added twice.
    Parts seen;
    for (const auto& conj : conjuncts) {
        if (conj) seen.push_back(conj.get());
    }

    std::vector<std::unique_ptr<Expression>> out;
    out.reserve(conjuncts.size());
    const auto add = [&](std::unique_ptr<Expression> expr) {
        if (Contains(seen, *expr)) return;
        seen.push_back(expr.get());
        out.push_back(std::move(expr));
    };
    for (auto& conj : conjuncts) {
        if (IsOp(conj.get(), BinaryOp::Op::OR)) AddImplied(*conj, tables_of, add);
        out.push_back(std::move(conj));
    }
    return out;
}

}  // namespace shilmandb
