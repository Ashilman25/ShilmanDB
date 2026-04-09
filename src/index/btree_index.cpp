#include "index/btree_index.hpp"
#include "storage/slotted_page.hpp"
#include <cassert>

namespace shilmandb {


BTreeIndex::BTreeIndex(BufferPoolManager* bpm, TypeId key_type, page_id_t root_page_id) : bpm_(bpm), root_page_id_(root_page_id), key_type_(key_type) {}


bool BTreeIndex::IsLeafPage(const char* page_data) {
    auto* hdr = SlottedPage::GetHeader(page_data);
    return (hdr->flags & 1u) != 0;
}

page_id_t BTreeIndex::GetRootPageId() const {
    return root_page_id_;
}

//Find leaf with path
page_id_t BTreeIndex::FindLeaf(const Value& key, std::vector<page_id_t>& path) {
    path.clear();
    auto current_id = root_page_id_;

    while (true) {
        auto* page = bpm_->FetchPage(current_id);
        auto* data = page->GetData();

        if (IsLeafPage(data)) {
            (void)bpm_->UnpinPage(current_id, false);
            return current_id;
        }

        path.push_back(current_id);
        auto child_id = BTreeInternalPage::Lookup(data, key, key_type_);
        (void)bpm_->UnpinPage(current_id, false);
        current_id = child_id;
    }
}


//find leaf no path
page_id_t BTreeIndex::FindLeaf(const Value& key) {
    auto current_id = root_page_id_;

    while (true) {
        auto* page = bpm_->FetchPage(current_id);
        auto* data = page->GetData();

        if (IsLeafPage(data)) {
            (void)bpm_->UnpinPage(current_id, false);
            return current_id;
        }

        auto child_id = BTreeInternalPage::Lookup(data, key, key_type_);
        (void)bpm_->UnpinPage(current_id, false);
        current_id = child_id;
    }
}


page_id_t BTreeIndex::FindLeftmostLeaf() {
    if (root_page_id_ == INVALID_PAGE_ID) {
        return INVALID_PAGE_ID;
    }

    auto current_id = root_page_id_;
    while (true) {
        auto* page = bpm_->FetchPage(current_id);
        auto* data = page->GetData();

        if (IsLeafPage(data)) {
            (void)bpm_->UnpinPage(current_id, false);
            return current_id;
        }

        auto child_id = BTreeInternalPage::GetChild(data, 0, key_type_);
        (void)bpm_->UnpinPage(current_id, false);
        current_id = child_id;
    }
}



void BTreeIndex::Insert(const Value& key, const RID& rid) {

    if (root_page_id_ == INVALID_PAGE_ID) {
        page_id_t new_id;
        auto* page = bpm_->NewPage(&new_id);
        auto* data = page->GetData();
        BTreeLeafPage::Init(data, new_id);
        (void)BTreeLeafPage::Insert(data, key, rid, key_type_);
        (void)bpm_->UnpinPage(new_id, true);
        root_page_id_ = new_id;
        return;
    }

    std::vector<page_id_t> path;
    auto leaf_id = FindLeaf(key, path);

    auto* page = bpm_->FetchPage(leaf_id);
    auto* data = page->GetData();
    bool is_full = (BTreeLeafPage::GetNumKeys(data) >= BTreeLeafPage::MaxKeys(key_type_));
    (void)bpm_->UnpinPage(leaf_id, false);

    if (is_full) {
        SplitLeaf(leaf_id, path);
        leaf_id = FindLeaf(key, path);
    }

    page = bpm_->FetchPage(leaf_id);
    data = page->GetData();
    (void)BTreeLeafPage::Insert(data, key, rid, key_type_);
    (void)bpm_->UnpinPage(leaf_id, true);
}



void BTreeIndex::Delete(const Value& key, const RID& rid) {
    if (root_page_id_ == INVALID_PAGE_ID) {
        return;
    }

    auto leaf_id = FindLeaf(key);
    auto* page = bpm_->FetchPage(leaf_id);
    auto* data = page->GetData();
    auto deleted = BTreeLeafPage::Delete(data, key, rid, key_type_);
    (void)bpm_->UnpinPage(leaf_id, deleted);
}



std::vector<RID> BTreeIndex::PointLookup(const Value& key) {
    if (root_page_id_ == INVALID_PAGE_ID) {
        return {};
    }

    auto leaf_id = FindLeaf(key);
    auto* page = bpm_->FetchPage(leaf_id);
    auto* data = page->GetData();
    auto result = BTreeLeafPage::Lookup(data, key, key_type_);
    (void)bpm_->UnpinPage(leaf_id, false);
    return result;
}



void BTreeIndex::SplitLeaf(page_id_t leaf_id, std::vector<page_id_t>& path) {

    auto* old_page = bpm_->FetchPage(leaf_id);
    auto* old_data = old_page->GetData();
    auto n = BTreeLeafPage::GetNumKeys(old_data);
    uint16_t split = n / 2;


    auto push_key = BTreeLeafPage::GetKey(old_data, split, key_type_);

    page_id_t new_id;
    auto* new_page = bpm_->NewPage(&new_id);
    auto* new_data = new_page->GetData();
    BTreeLeafPage::Init(new_data, new_id);


    for (uint16_t i = split; i < n; ++i) {
        auto key = BTreeLeafPage::GetKey(old_data, i, key_type_);
        auto rid = BTreeLeafPage::GetRID(old_data, i, key_type_);
        (void)BTreeLeafPage::Insert(new_data, key, rid, key_type_);
    }


    auto old_next = BTreeLeafPage::GetNextLeafPageId(old_data);
    BTreeLeafPage::SetNextLeafPageId(new_data, old_next);
    BTreeLeafPage::SetNextLeafPageId(old_data, new_id);

    BTreeLeafPage::SetNumKeys(old_data, split);

    (void)bpm_->UnpinPage(leaf_id, true);
    (void)bpm_->UnpinPage(new_id, true);

    InsertIntoParent(path, leaf_id, push_key, new_id);
}



void BTreeIndex::InsertIntoParent(std::vector<page_id_t>& path, page_id_t left_child, const Value& key, page_id_t right_child) {
    if (path.empty()) {
        CreateNewRoot(key, left_child, right_child);
        return;
    }

    auto parent_id = path.back();
    path.pop_back();

    auto* parent_page = bpm_->FetchPage(parent_id);
    auto* parent_data = parent_page->GetData();
    BTreeInternalPage::InsertAfter(parent_data, left_child, key, right_child, key_type_);

    if (BTreeInternalPage::NeedsSplit(parent_data, key_type_)) {
        auto n = BTreeInternalPage::GetNumKeys(parent_data);
        uint16_t mid = n / 2;

        auto push_key = BTreeInternalPage::GetKey(parent_data, mid, key_type_);

        page_id_t new_internal_id;
        auto* new_page = bpm_->NewPage(&new_internal_id);
        auto* new_data = new_page->GetData();
        BTreeInternalPage::Init(new_data, new_internal_id);

        auto first_right_child = BTreeInternalPage::GetChild(parent_data, mid + 1, key_type_);
        BTreeInternalPage::SetChild(new_data, 0, first_right_child, key_type_);

        uint16_t new_count = n - mid - 1;
        BTreeInternalPage::SetNumKeys(new_data, new_count);
        for (uint16_t i = 0; i < new_count; ++i) {
            auto k = BTreeInternalPage::GetKey(parent_data, mid + 1 + i, key_type_);
            BTreeInternalPage::SetKey(new_data, i, k, key_type_);
            auto c = BTreeInternalPage::GetChild(parent_data, mid + 2 + i, key_type_);
            BTreeInternalPage::SetChild(new_data, i + 1, c, key_type_);
        }

        BTreeInternalPage::SetNumKeys(parent_data, mid);

        (void)bpm_->UnpinPage(parent_id, true);
        (void)bpm_->UnpinPage(new_internal_id, true);

        InsertIntoParent(path, parent_id, push_key, new_internal_id);

    } else {
        (void)bpm_->UnpinPage(parent_id, true);
    }
}



void BTreeIndex::CreateNewRoot(const Value& key, page_id_t left_child, page_id_t right_child) {
    page_id_t root_id;
    auto* page = bpm_->NewPage(&root_id);
    auto* data = page->GetData();
    BTreeInternalPage::Init(data, root_id);

    BTreeInternalPage::SetChild(data, 0, left_child, key_type_);
    BTreeInternalPage::SetNumKeys(data, 1);
    BTreeInternalPage::SetKey(data, 0, key, key_type_);
    BTreeInternalPage::SetChild(data, 1, right_child, key_type_);

    (void)bpm_->UnpinPage(root_id, true);
    root_page_id_ = root_id;
}



BTreeIndex::Iterator::Iterator(BufferPoolManager* bpm, TypeId key_type, page_id_t page_id, uint16_t slot_idx) : bpm_(bpm), key_type_(key_type), current_page_id_(page_id), current_slot_(slot_idx) {}

std::pair<Value, RID> BTreeIndex::Iterator::operator*() {
    auto* page = bpm_->FetchPage(current_page_id_);
    auto* data = page->GetData();

    auto key = BTreeLeafPage::GetKey(data, current_slot_, key_type_);
    auto rid = BTreeLeafPage::GetRID(data, current_slot_, key_type_);
    
    (void)bpm_->UnpinPage(current_page_id_, false);
    return {key, rid};
}

BTreeIndex::Iterator& BTreeIndex::Iterator::operator++() {
    auto* page = bpm_->FetchPage(current_page_id_);
    auto* data = page->GetData();
    auto num_keys = BTreeLeafPage::GetNumKeys(data);

    if (current_slot_ + 1 < num_keys) {
        ++current_slot_;
        (void)bpm_->UnpinPage(current_page_id_, false);

    } else {
        auto next = BTreeLeafPage::GetNextLeafPageId(data);
        (void)bpm_->UnpinPage(current_page_id_, false);

        while (next != INVALID_PAGE_ID) {
            auto* next_page = bpm_->FetchPage(next);
            auto* next_data = next_page->GetData();
            auto next_num_keys = BTreeLeafPage::GetNumKeys(next_data);
            auto following = BTreeLeafPage::GetNextLeafPageId(next_data);
            (void)bpm_->UnpinPage(next, false);

            if (next_num_keys > 0) {
                current_page_id_ = next;
                current_slot_ = 0;
                return *this;
            }
            next = following;
        }

        current_page_id_ = INVALID_PAGE_ID;
        current_slot_ = 0;
    }
    return *this;
}

bool BTreeIndex::Iterator::operator!=(const Iterator& other) const {
    return current_page_id_ != other.current_page_id_ || current_slot_ != other.current_slot_;
}

bool BTreeIndex::Iterator::IsEnd() const {
    return current_page_id_ == INVALID_PAGE_ID;
}



BTreeIndex::Iterator BTreeIndex::Begin() {
    if (root_page_id_ == INVALID_PAGE_ID) {
        return End();
    }

    auto current_id = FindLeftmostLeaf();
    while (current_id != INVALID_PAGE_ID) {
        auto* page = bpm_->FetchPage(current_id);
        auto* data = page->GetData();
        auto num_keys = BTreeLeafPage::GetNumKeys(data);
        auto next_id = BTreeLeafPage::GetNextLeafPageId(data);
        (void)bpm_->UnpinPage(current_id, false);

        if (num_keys > 0) {
            return Iterator(bpm_, key_type_, current_id, 0);
        }
        current_id = next_id;
    }
    return End();
}

BTreeIndex::Iterator BTreeIndex::Begin(const Value& low_key) {
    if (root_page_id_ == INVALID_PAGE_ID) {
        return End();
    }

    auto leaf_id = FindLeaf(low_key);
    auto* page = bpm_->FetchPage(leaf_id);
    auto* data = page->GetData();
    auto pos = BTreeLeafPage::LowerBound(data, low_key, key_type_);
    auto num_keys = BTreeLeafPage::GetNumKeys(data);

    if (pos >= num_keys) {
        auto next = BTreeLeafPage::GetNextLeafPageId(data);
        (void)bpm_->UnpinPage(leaf_id, false);

        // Skip empty leaves left by deletions
        while (next != INVALID_PAGE_ID) {
            auto* next_page = bpm_->FetchPage(next);
            auto* next_data = next_page->GetData();
            auto next_num_keys = BTreeLeafPage::GetNumKeys(next_data);
            auto following = BTreeLeafPage::GetNextLeafPageId(next_data);
            (void)bpm_->UnpinPage(next, false);

            if (next_num_keys > 0) {
                return Iterator(bpm_, key_type_, next, 0);
            }
            next = following;
        }
        return End();
    }

    (void)bpm_->UnpinPage(leaf_id, false);
    return Iterator(bpm_, key_type_, leaf_id, pos);
}

BTreeIndex::Iterator BTreeIndex::End() {
    return Iterator(bpm_, key_type_, INVALID_PAGE_ID, 0);
}

}  // namespace shilmandb
