#pragma once
#include "common/config.hpp"
#include <optional>
#include <cstddef>
#include <cstdint>

namespace shilmandb {

// How a fetch relates to its page's recent use. The buffer pool works it out.
enum class AccessType : uint8_t {
    NEW_USE,  // a miss, a new page, or a return after more than kVisitWindow page changes
    TOUCH,    // a return to a resident page within kVisitWindow page changes: the same visit
    REPEAT,   // the same page as the fetch just before
};

class EvictionPolicy {

public:
    virtual ~EvictionPolicy() = default;

    //record that frame_id was accessed (move to MRU position in LRU)
    virtual void RecordAccess(frame_id_t frame_id) = 0;

    //record access with page_id context (learned policies need page identity)
    virtual void RecordAccess(frame_id_t frame_id, page_id_t page_id) {
        RecordAccess(frame_id);
    }

    // Policies that count visits override this. The default ignores the type.
    virtual void RecordAccess(frame_id_t frame_id, page_id_t page_id, AccessType /*type*/) {
        RecordAccess(frame_id, page_id);
    }

    //mark frame as evicitable or not
    virtual void SetEvictable(frame_id_t frame_id, bool evictable) = 0;

    //select and return chosen frame
    virtual std::optional<frame_id_t> Evict() = 0;

    // Picks a victim for loading incoming_page (INVALID_PAGE_ID for a brand new page).
    // ARC needs it; the default ignores it.
    virtual std::optional<frame_id_t> Evict(page_id_t /*incoming_page*/) { return Evict(); }

    //remove frame from tracking
    virtual void Remove(frame_id_t frame_id) = 0;

    //num of evictable frames tracked
    virtual size_t Size() const = 0;

    // True if the buffer pool should work out an AccessType for every fetch.
    [[nodiscard]] virtual bool UsesAccessTypes() const { return false; }

    // Forgets every frame and page, as if just built.
    virtual void Reset() = 0;

};

}
