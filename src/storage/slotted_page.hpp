#pragma once
#include "common/config.hpp"
#include <cstdint>
#include <cstring>

namespace shilmandb {

struct SlotEntry {
    uint16_t offset;  // byte offset from start of page to tuple data
    uint16_t length;  // tuple length in bytes; 0 = deleted
};

struct PageHeader {
    page_id_t page_id;
    uint16_t num_slots;
    uint16_t free_space_offset;  // byte offset to end of slot directory
    page_id_t next_page_id;     // linked list of heap pages
    uint32_t flags;              // bit 0: is_leaf - for B-tree
};

static constexpr size_t PAGE_HEADER_SIZE = sizeof(PageHeader);  // 16 bytes
static constexpr size_t SLOT_ENTRY_SIZE = sizeof(SlotEntry);   // 4 bytes

static_assert(sizeof(PageHeader) == 16, "PageHeader must be exactly 16 bytes for on-disk layout");
static_assert(sizeof(SlotEntry)  == 4, "SlotEntry must be exactly 4 bytes");

class SlottedPage {
public:
    // The biggest row an empty page holds: the page minus its header and one slot
    static constexpr uint16_t kMaxTupleLength = PAGE_SIZE - sizeof(PageHeader) - sizeof(SlotEntry);

    static void Init(char* page_data, page_id_t page_id);

    static PageHeader* GetHeader(char* page_data);
    static const PageHeader* GetHeader(const char* page_data);

    [[nodiscard]] static int  InsertTuple(char* page_data, const char* tuple_data, uint16_t tuple_length);
    [[nodiscard]] static bool GetTuple(const char* page_data, uint16_t slot_id, char* out_data, uint16_t* out_length);
    [[nodiscard]] static bool DeleteTuple(char* page_data, uint16_t slot_id);

    static void Compact(char* page_data);

    [[nodiscard]] static uint16_t GetFreeSpace(const char* page_data);

private:
    static SlotEntry* GetSlotEntry(char* page_data, uint16_t slot_id);
    static const SlotEntry* GetSlotEntry(const char* page_data, uint16_t slot_id);
    static uint16_t FindTupleDataStart(const char* page_data, uint16_t num_slots);
};

static_assert(SlottedPage::kMaxTupleLength == PAGE_SIZE - PAGE_HEADER_SIZE - SLOT_ENTRY_SIZE, "the biggest row must fill an empty page exactly and fit a uint16_t");

}  // namespace shilmandb
