#pragma once
#include "common/config.hpp"
#include <cassert>
#include <cstring>

namespace shilmandb {

class Page {
public:
    void Pin() { ++pin_count_; }
    void Unpin() { assert(pin_count_ > 0 && "Unpin called on page with zero pin count"); --pin_count_; }

    void MarkDirty() { is_dirty_ = true; }
    void ClearDirty() { is_dirty_ = false; }

    [[nodiscard]] int GetPinCount() const { return pin_count_; }
    [[nodiscard]] bool IsDirty() const { return is_dirty_; }

    // A columnar page read from disk is checksummed once, then marked verified
    [[nodiscard]] bool IsVerified() const { return is_verified_; }
    void MarkVerified() { is_verified_ = true; }

    [[nodiscard]] page_id_t GetPageId() const { return page_id_; }
    void SetPageId(page_id_t id) { page_id_ = id; }

    char* GetData() { return data_; }
    const char* GetData() const { return data_; }

    // Forgets which page this is, its pin count and both flags.
    // The bytes stay as they are.
    void ClearMetadata() {
        page_id_ = INVALID_PAGE_ID;
        pin_count_ = 0;
        is_dirty_ = false;
        is_verified_ = false;
    }

    void ResetMemory() {
        std::memset(data_, 0, PAGE_SIZE);
        ClearMetadata();
    }

private:
    char data_[PAGE_SIZE]{};
    page_id_t page_id_{INVALID_PAGE_ID};
    int pin_count_{0};
    bool is_dirty_{false};
    bool is_verified_{false};
};

}  // namespace shilmandb
