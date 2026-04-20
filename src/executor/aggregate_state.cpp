#include "executor/aggregate_state.hpp"

#include "common/exception.hpp"

namespace shilmandb {

double ValueToDouble(const Value& v) {
    switch (v.type_) {
        case TypeId::INTEGER:
            return static_cast<double>(v.integer_);
        case TypeId::BIGINT:
            return static_cast<double>(v.bigint_);
        case TypeId::DECIMAL:
            return v.decimal_;
        default:
            throw DatabaseException("Cannot aggregate non-numeric type");
    }
}

}  // namespace shilmandb
