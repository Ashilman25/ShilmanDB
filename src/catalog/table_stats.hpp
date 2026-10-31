#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include "types/value.hpp"

namespace shilmandb {

struct MinMax {
    Value min;
    Value max;
};

struct TableStats {
    uint64_t row_count{0};

    //col name, distinct vals
    std::unordered_map<std::string, uint64_t> distinct_counts;

    //col name, smallest and largest value (INTEGER, BIGINT, DECIMAL and DATE columns only)
    std::unordered_map<std::string, MinMax> min_max;
};

}
