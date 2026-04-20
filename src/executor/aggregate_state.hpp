#pragma once
#include "types/value.hpp"

#include <cstdint>

namespace shilmandb {

struct AggregateState {
    int64_t count{0};
    double sum{0.0};
    Value min_val;
    Value max_val;
    bool has_value{false};
};

// Throws DatabaseException on non-numeric Value.
double ValueToDouble(const Value& v);

}  // namespace shilmandb
