#include "storage/columnar/column_vector.hpp"

namespace shilmandb {

// The kind is fixed by the type; strings start with the arena's first offset.
ColumnVector::ColumnVector(ColumnType type) : type_(type), kind_(PhysicalKindOf(type)) {
    if (kind_ == PhysicalKind::STRING) offsets_.push_back(0);
}

// Reserves only the array this kind uses, plus the NULL flags.
void ColumnVector::Reserve(size_t n) {
    nulls_.reserve(n);
    switch (kind_) {
        case PhysicalKind::INT64:  ints_.reserve(n); break;
        case PhysicalKind::INT128: wides_.reserve(n); break;
        case PhysicalKind::DOUBLE: doubles_.reserve(n); break;
        case PhysicalKind::STRING: offsets_.reserve(n + 1); break;
    }
}

// Empties every array, leaving the strings' first offset in place.
void ColumnVector::Clear() {
    ints_.clear();
    wides_.clear();
    doubles_.clear();
    bytes_.clear();
    nulls_.clear();
    null_count_ = 0;
    if (kind_ == PhysicalKind::STRING) offsets_.assign(1, 0);
}

// A NULL row still takes a slot, so row i is index i in every array.
void ColumnVector::AppendNull() {
    switch (kind_) {
        case PhysicalKind::INT64:  ints_.push_back(0); break;
        case PhysicalKind::INT128: wides_.push_back(0); break;
        case PhysicalKind::DOUBLE: doubles_.push_back(0.0); break;
        case PhysicalKind::STRING: offsets_.push_back(bytes_.size()); break;
    }
    nulls_.push_back(1);
    ++null_count_;
}

}  // namespace shilmandb
