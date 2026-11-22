#pragma once

#include "parser/ast.hpp"

#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

namespace shilmandb {

// True when two trees match in shape, names, operators, literal types, values and text.
[[nodiscard]] bool SameExpression(const Expression& a, const Expression& b);

// Before each OR of ANDs, adds clones of what every branch shares, and per table
// the OR of each branch's own filters on it. Every original conjunct stays.
// tables_of marks a column it can't place with -1; such parts stay put.
// Nothing equal to a conjunct already there is added again.
[[nodiscard]] std::vector<std::unique_ptr<Expression>> FactorOrConjuncts(
    std::vector<std::unique_ptr<Expression>> conjuncts,
    const std::function<std::unordered_set<int>(const Expression&)>& tables_of);

}  // namespace shilmandb
