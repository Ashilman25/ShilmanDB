#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>

namespace shilmandb {

struct TableStats {
    uint64_t row_count{0};

    //col name, distinct vals
    std::unordered_map<std::string, uint64_t> distinct_counts;
};

}