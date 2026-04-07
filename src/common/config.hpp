#pragma once
#include <cstdint>
#include <cstddef>
#include <limits>

namespace learneddb {
    static constexpr size_t PAGE_SIZE = 8192;
    using page_id_t = uint32_t;
    using frame_id_t = uint32_t;
    static constexpr page_id_t INVALID_PAGE_ID = std::numeric_limits<page_id_t>::max();
}  // namespace learneddb
