#pragma once
#include "common/config.hpp"

namespace shilmandb {

struct RID {
    page_id_t page_id{INVALID_PAGE_ID};
    uint16_t slot_id{0};

    RID() = default;
    RID(page_id_t pid, uint16_t sid) : page_id(pid), slot_id(sid) {}

    bool operator==(const RID& other) const {
        return page_id == other.page_id && slot_id == other.slot_id;
    }

    bool operator!=(const RID& other) const {
        return !(*this == other);
    }
};


} //namespace shilmandb