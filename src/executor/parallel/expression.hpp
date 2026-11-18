#pragma once
#include "executor/parallel/batch.hpp"
#include "executor/parallel/vector.hpp"
#include "parser/ast.hpp"
#include <cstdint>
#include <memory>
#include <vector>

namespace shilmandb::parallel {

// Where each input column sits in a batch. slot_of_column[c] == -1: not present.
struct InputLayout {
    std::vector<int> slot_of_column;      // indexed by the plan schema's column index (bound_index)
    std::vector<VType> slot_types;        // indexed by slot
    std::vector<uint8_t> slot_precision;
};

// Per-worker temporaries (vectors, selections, lookup tables), reused across calls.
// Each thread keeps its own; compiled expressions hold no per-call state.
struct EvalScratch {
private:
    friend struct ScratchAccess;
    std::vector<std::unique_ptr<Vector>> vectors_[4];  // free vectors by storage: 64-bit, 128-bit, double, string
    std::vector<std::unique_ptr<Selection>> selections_;
    std::vector<std::unique_ptr<std::vector<uint8_t>>> tables_;
};

// A bound expression compiled into typed kernels over a batch's vectors.
// Immutable once built, so several workers may evaluate one at the same time.
class CompiledExpr {
public:
    // Throws DatabaseException for AGGREGATE, STAR, an unbound column, a column not in
    // the layout, and anything today's executor would reject on every row.
    [[nodiscard]] static std::unique_ptr<CompiledExpr> Compile(const Expression& expr, const InputLayout& in);

    virtual ~CompiledExpr() = default;

    // The type of every value this expression writes.
    [[nodiscard]] virtual VType Type() const = 0;

    // Writes a value for every row in `rows` into out at the same physical index.
    // Strings come out plain, pointing into the batch or this tree's constants.
    virtual void Evaluate(const Batch& in, const Selection& rows, Vector& out, EvalScratch& scratch) const = 0;

    // Narrows sel to the rows where this expression is true (a non-zero INTEGER).
    virtual void Select(const Batch& in, Selection& sel, EvalScratch& scratch) const = 0;
};

// Adds the columns (plan schema indexes) an expression reads to out, kept sorted and unique.
void CollectColumns(const Expression& expr, std::vector<uint32_t>& out);

}  // namespace shilmandb::parallel
