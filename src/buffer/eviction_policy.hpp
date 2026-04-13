#pragma once
#include "common/config.hpp"
#include <optional>
#include <cstddef>

namespace shilmandb {


class EvictionPolicy {

public:
    virtual ~EvictionPolicy() = default;

    //record that frame_id was accessed (move to MRU position in LRU)
    virtual void RecordAccess(frame_id_t frame_id) = 0;

    //record access with page_id context (learned policies need page identity)
    virtual void RecordAccess(frame_id_t frame_id, page_id_t page_id) {
        RecordAccess(frame_id);
    }

    //mark frame as evicitable or not
    virtual void SetEvictable(frame_id_t frame_id, bool evictable) = 0;

    //select and return chosen frame
    virtual std::optional<frame_id_t> Evict() = 0;

    //remove frame from tracking
    virtual void Remove(frame_id_t frame_id) = 0;

    //num of evictable frames tracked
    virtual size_t Size() const = 0;

};

}