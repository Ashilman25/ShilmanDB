#pragma once
#include <cstdint>
#include <cstddef>
#include <limits>

namespace shilmandb {
    using page_id_t = uint32_t;
    using frame_id_t = uint32_t;

    static constexpr size_t PAGE_SIZE = 8192; // 8KB
    static constexpr page_id_t INVALID_PAGE_ID = std::numeric_limits<page_id_t>::max();
    static constexpr size_t DEFAULT_BUFFER_POOL_SIZE = 1024; // 1024 frames = 8MB
    
}  // namespace shilmandb
