#pragma once
#include "buffer/buffer_pool_manager.hpp"
#include "common/rid.hpp"
#include "types/tuple.hpp"
#include "catalog/schema.hpp"
#include "common/config.hpp"
#include "storage/slotted_page.hpp"

namespace shilmandb {

class TableHeap {
public:
    // Open an existing heap file by its first page id
    TableHeap(BufferPoolManager* bpm, page_id_t first_page_id);

    // Create a new empty heap
    explicit TableHeap(BufferPoolManager* bpm);

    [[nodiscard]] RID InsertTuple(const Tuple& tuple);
    [[nodiscard]] bool GetTuple(const RID& rid, Tuple* tuple, const Schema& schema);
    [[nodiscard]] bool DeleteTuple(const RID& rid);

    [[nodiscard]] page_id_t GetFirstPageId() const;

    class Iterator {
    public:
        Iterator(TableHeap* table, RID rid, const Schema* schema);

        Tuple operator*();
        Iterator& operator++();
        bool operator==(const Iterator& other) const;
        bool operator!=(const Iterator& other) const;
        [[nodiscard]] RID GetRID() const;

    private:
        TableHeap* table_;
        RID current_rid_;
        const Schema* schema_;

        void AdvancePastDeleted();
    };

    Iterator Begin(const Schema& schema);
    Iterator End();

    // Passes live rows from *pos onward to on_row(data, length), one pin per page.
    // Stops at max_rows or the end; *pos is left at the next unread slot.
    template <typename OnRow>
    size_t ReadRows(RID* pos, size_t max_rows, OnRow&& on_row) {
        size_t count = 0;
        while (count < max_rows && pos->page_id != INVALID_PAGE_ID) {
            const page_id_t page_id = pos->page_id;
            auto* page = bpm_->FetchPage(page_id);
            const char* data = page->GetData();
            const auto* header = SlottedPage::GetHeader(data);
            const size_t num_slots = header->num_slots;
            const page_id_t next_page_id = header->next_page_id;

            size_t slot_id = pos->slot_id;
            while (slot_id < num_slots && count < max_rows) {
                const auto* slot = reinterpret_cast<const SlotEntry*>(data + PAGE_HEADER_SIZE + slot_id * SLOT_ENTRY_SIZE);
                ++slot_id;
                
                if (slot->length == 0) { continue; }
                on_row(data + slot->offset, static_cast<uint32_t>(slot->length));
                ++count;
            }
            pos->slot_id = static_cast<decltype(pos->slot_id)>(slot_id);
            (void)bpm_->UnpinPage(page_id, false);

            if (slot_id >= num_slots && count < max_rows) {
                *pos = RID{next_page_id, 0};
            }
        }
        return count;
    }

private:
    BufferPoolManager* bpm_;
    page_id_t first_page_id_;
    page_id_t last_insert_page_id_;
};

}  // namespace shilmandb
