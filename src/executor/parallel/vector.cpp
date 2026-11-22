#include "executor/parallel/vector.hpp"

namespace shilmandb::parallel {

// Compares every field.
bool VType::operator==(const VType& o) const {
    return kind == o.kind && scale == o.scale && wide == o.wide && result == o.result;
}

// The opposite of operator==.
bool VType::operator!=(const VType& o) const {
    return !(*this == o);
}

// The result type is what the columnar scan already shows, so the two agree.
VType VTypeOf(const ColumnType& storage) {
    VType t;
    t.result = ExecutorType(storage);  // throws for an unknown id
    switch (storage.id) {
        case ColumnTypeId::BOOLEAN:
        case ColumnTypeId::SMALLINT:
        case ColumnTypeId::INTEGER:
        case ColumnTypeId::BIGINT:
            t.kind = Kind::INTEGER;
            break;
        case ColumnTypeId::NUMERIC:
            t.kind = Kind::DECIMAL;
            t.scale = storage.scale;
            t.wide = PhysicalKindOf(storage) == PhysicalKind::INT128;
            break;
        case ColumnTypeId::DOUBLE:
            t.kind = Kind::DOUBLE;
            break;
        case ColumnTypeId::DATE:
            t.kind = Kind::DATE;
            break;
        case ColumnTypeId::TIMESTAMP:  // shown as text, like the columnar scan does
        case ColumnTypeId::VARCHAR:
        case ColumnTypeId::TEXT:
            t.kind = Kind::STRING;
            break;
    }
    return t;
}

// Sizes only the array this kind reads and writes.
Vector::Vector(VType type) : type_(type) {
    switch (type.kind) {
        case Kind::INTEGER:
        case Kind::DATE:
            i64_.resize(kBatchSize);
            break;
        case Kind::DECIMAL:
            if (type.wide) {
                i128_.resize(kBatchSize);
            } else {
                i64_.resize(kBatchSize);
            }
            break;
        case Kind::DOUBLE:
            f64_.resize(kBatchSize);
            break;
        case Kind::STRING:
            str_.resize(kBatchSize);
            codes_.resize(kBatchSize);
            break;
    }
}

// The entries belong to the caller and must outlive this form.
void Vector::SetDictionary(const std::vector<StringRef>* entries) {
    assert(type_.kind == Kind::STRING && "SetDictionary() on a vector that isn't strings");
    dict_ = entries;
}

}  // namespace shilmandb::parallel
