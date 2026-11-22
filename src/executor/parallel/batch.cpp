#include "executor/parallel/batch.hpp"
#include <numeric>

namespace shilmandb::parallel {

// Fills 0..n-1 too, so operator[] needs no branch.
void Selection::SetAll(uint32_t n) {
    assert(n <= kBatchSize && "selection bigger than a batch");
    std::iota(idx_.begin(), idx_.begin() + n, uint16_t{0});
    size_ = n;
    all_ = true;
}

// Debug builds check the indexes are strictly ascending and inside a batch.
void Selection::Resize(uint32_t n) {
    assert(n <= kBatchSize && "selection bigger than a batch");
#ifndef NDEBUG
    for (uint32_t i = 1; i < n; ++i) {
        assert(idx_[i - 1] < idx_[i] && "selection indexes must be ascending");
    }
#endif
    size_ = n;
    all_ = false;
}

// Drops any dictionary, so a reused batch never reads a stale one.
void Batch::Reset(uint32_t n) {
    assert(n <= kBatchSize && "batch bigger than kBatchSize");
    rows = n;
    sel.SetAll(n);
    for (Vector& slot : slots) {
        if (slot.IsDictionary()) {
            slot.SetDictionary(nullptr);
        }
    }
}

}  // namespace shilmandb::parallel
