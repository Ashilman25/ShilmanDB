#pragma once
#include "executor/parallel/vector.hpp"
#include <cassert>
#include <cstdint>
#include <vector>

namespace shilmandb::parallel {

static_assert(kBatchSize <= 65536, "row indexes in a selection are 16 bits");

// The active rows of a batch, as ascending row indexes.
class Selection {
public:
    // Room for a full batch of indexes, sized once.
    Selection() : idx_(kBatchSize) {}

    // Rows 0..n-1, flagged as "all" so loops can skip the indirection.
    void SetAll(uint32_t n);

    // True after SetAll, false once Resize narrows it.
    [[nodiscard]] bool IsAll() const { return all_; }

    // How many rows are active.
    [[nodiscard]] uint32_t Size() const { return size_; }

    // The i-th active row.
    [[nodiscard]] uint16_t operator[](uint32_t i) const {
        assert(i < size_ && "selection index out of range");
        return idx_[i];
    }

    // Write up to kBatchSize ascending indexes here, then call Resize.
    uint16_t* Data() { return idx_.data(); }

    // The active rows as a plain array, for read-only loops.
    [[nodiscard]] const uint16_t* Data() const { return idx_.data(); }

    // Keeps the first n written indexes; the selection is no longer "all".
    void Resize(uint32_t n);

private:
    std::vector<uint16_t> idx_;
    uint32_t size_{0};
    bool all_{true};
};

// Vectors of one batch, a physical row count and the active rows.
struct Batch {
    std::vector<Vector> slots;
    uint32_t rows{0};  // physical rows 0..rows-1
    Selection sel;     // the active rows

    // How many rows are active.
    [[nodiscard]] uint32_t Active() const { return sel.Size(); }

    // rows = n, every row active, and every slot back to plain form.
    void Reset(uint32_t n);
};

}  // namespace shilmandb::parallel
