#include "storage/table_heap.hpp"
#include "storage/slotted_page.hpp"
#include <cassert>

namespace shilmandb {

//CONSTRUCTORS

// Existing heap — no I/O
TableHeap::TableHeap(BufferPoolManager* bpm, page_id_t first_page_id) : bpm_(bpm), first_page_id_(first_page_id), last_insert_page_id_(first_page_id) {}

// New empty heap — allocate first page
TableHeap::TableHeap(BufferPoolManager* bpm) : bpm_(bpm), first_page_id_(INVALID_PAGE_ID), last_insert_page_id_(INVALID_PAGE_ID) {
    auto* page = bpm_->NewPage(&first_page_id_);
    SlottedPage::Init(page->GetData(), first_page_id_);
    (void)bpm_->UnpinPage(first_page_id_, true);
    last_insert_page_id_ = first_page_id_;
}

page_id_t TableHeap::GetFirstPageId() const { return first_page_id_; }



RID TableHeap::InsertTuple(const Tuple& tuple) {
    if (tuple.GetLength() > static_cast<uint32_t>(UINT16_MAX)) {
        return RID{INVALID_PAGE_ID, 0};
    }

    auto tuple_len = static_cast<uint16_t>(tuple.GetLength());
    auto current_page_id = last_insert_page_id_;

    while (true) {
        auto* page = bpm_->FetchPage(current_page_id);
        if (page == nullptr) { return RID{INVALID_PAGE_ID, 0}; }

        auto slot_id = SlottedPage::InsertTuple(page->GetData(), tuple.GetData(), tuple_len);

        if (slot_id != -1) {
            (void)bpm_->UnpinPage(current_page_id, true);
            last_insert_page_id_ = current_page_id;
            return RID{current_page_id, static_cast<uint16_t>(slot_id)};
        }

        // No space, follow linked list or allocate new page
        auto* header = SlottedPage::GetHeader(page->GetData());
        auto next_page_id = header->next_page_id;

        if (next_page_id == INVALID_PAGE_ID) {
            page_id_t new_page_id;
            auto* new_page = bpm_->NewPage(&new_page_id);

            if (new_page == nullptr) {
                (void)bpm_->UnpinPage(current_page_id, false);
                return RID{INVALID_PAGE_ID, 0};
            }
            SlottedPage::Init(new_page->GetData(), new_page_id);

            //current -> new
            header->next_page_id = new_page_id;
            (void)bpm_->UnpinPage(current_page_id, true);
            (void)bpm_->UnpinPage(new_page_id, true);
            current_page_id = new_page_id;

        } else {
            (void)bpm_->UnpinPage(current_page_id, false);
            current_page_id = next_page_id;
        }
    }
}



bool TableHeap::GetTuple(const RID& rid, Tuple* tuple, const Schema& schema) {
    auto* page = bpm_->FetchPage(rid.page_id);
    if (page == nullptr) { return false; }

    char buf[PAGE_SIZE];
    uint16_t len = 0;
    bool ok = SlottedPage::GetTuple(page->GetData(), rid.slot_id, buf, &len);

    if (ok) {
        tuple->DeserializeFrom(buf, static_cast<uint32_t>(len), schema);
    }
    (void)bpm_->UnpinPage(rid.page_id, false);
    return ok;
}

bool TableHeap::DeleteTuple(const RID& rid) {
    auto* page = bpm_->FetchPage(rid.page_id);
    if (page == nullptr) { return false; }

    bool ok = SlottedPage::DeleteTuple(page->GetData(), rid.slot_id);
    (void)bpm_->UnpinPage(rid.page_id, ok);
    return ok;
}



TableHeap::Iterator::Iterator(TableHeap* table, RID rid, const Schema* schema) : table_(table), current_rid_(rid), schema_(schema) {
    //land on first valid tuple or get to end
    if (current_rid_.page_id != INVALID_PAGE_ID) {
        AdvancePastDeleted();
    }
}

Tuple TableHeap::Iterator::operator*() {
    assert(current_rid_.page_id != INVALID_PAGE_ID && "Dereferenced end iterator");
    auto* page = table_->bpm_->FetchPage(current_rid_.page_id);

    char buf[PAGE_SIZE];
    uint16_t len = 0;
    (void)SlottedPage::GetTuple(page->GetData(), current_rid_.slot_id, buf, &len);

    Tuple tuple;
    tuple.DeserializeFrom(buf, static_cast<uint32_t>(len), *schema_);
    (void)table_->bpm_->UnpinPage(current_rid_.page_id, false);
    return tuple;
}

TableHeap::Iterator& TableHeap::Iterator::operator++() {
    auto* page = table_->bpm_->FetchPage(current_rid_.page_id);
    auto* header = SlottedPage::GetHeader(page->GetData());

    current_rid_.slot_id++;

    if (current_rid_.slot_id >= header->num_slots) {
        auto next = header->next_page_id;
        (void)table_->bpm_->UnpinPage(current_rid_.page_id, false);

        if (next == INVALID_PAGE_ID) {
            current_rid_ = RID{INVALID_PAGE_ID, 0};
            return *this;
        }
        current_rid_ = RID{next, 0};
    } else {
        (void)table_->bpm_->UnpinPage(current_rid_.page_id, false);
    }

    AdvancePastDeleted();
    return *this;
}

bool TableHeap::Iterator::operator==(const Iterator& other) const {
    return current_rid_ == other.current_rid_;
}

bool TableHeap::Iterator::operator!=(const Iterator& other) const {
    return current_rid_ != other.current_rid_;
}

RID TableHeap::Iterator::GetRID() const {
    return current_rid_;
}

void TableHeap::Iterator::AdvancePastDeleted() {
    while (current_rid_.page_id != INVALID_PAGE_ID) {
        auto* page = table_->bpm_->FetchPage(current_rid_.page_id);
        auto* header = SlottedPage::GetHeader(page->GetData());

        while (current_rid_.slot_id < header->num_slots) {
            auto* slot = reinterpret_cast<const SlotEntry*>(page->GetData() + PAGE_HEADER_SIZE + current_rid_.slot_id * SLOT_ENTRY_SIZE);

            if (slot->length > 0) {
                (void)table_->bpm_->UnpinPage(current_rid_.page_id, false);
                return;
            }
            current_rid_.slot_id++;
        }

        // Past end of page, move to next
        auto next = header->next_page_id;
        (void)table_->bpm_->UnpinPage(current_rid_.page_id, false);

        if (next == INVALID_PAGE_ID) {
            current_rid_ = RID{INVALID_PAGE_ID, 0};
            return;
        }
        current_rid_ = RID{next, 0};
    }
}

TableHeap::Iterator TableHeap::Begin(const Schema& schema) {
    return Iterator(this, RID{first_page_id_, 0}, &schema);
}

TableHeap::Iterator TableHeap::End() {
    return Iterator(this, RID{INVALID_PAGE_ID, 0}, nullptr);
}

}  // namespace shilmandb
