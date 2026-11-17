#include "index/btree_index.hpp"
#include "common/exception.hpp"
#include "storage/slotted_page.hpp"
#include <cassert>
#include <string>

namespace shilmandb {

namespace {

constexpr int kMaxDepth = 64;  // far deeper than any real tree, so a longer descent is a cycle

// Holds a fetched page and unpins it on scope exit, even after a throw.
class PagePin {
public:
    PagePin(BufferPoolManager* bpm, page_id_t page_id) : bpm_(bpm), page_id_(page_id), data_(bpm->FetchPage(page_id)->GetData()) {}
    ~PagePin() { (void)bpm_->UnpinPage(page_id_, dirty_); }
    PagePin(const PagePin&) = delete;
    PagePin& operator=(const PagePin&) = delete;

    [[nodiscard]] char* Data() const { return data_; }
    void MarkDirty() { dirty_ = true; }

private:
    BufferPoolManager* bpm_;
    page_id_t page_id_;
    char* data_;
    bool dirty_{false};
};

// Builds and throws the bad child pointer error, kept off the hot path.
[[noreturn]] __attribute__((cold, noinline)) void ThrowBadChild(page_id_t parent, page_id_t child, page_id_t num_pages) {
    const std::string at = "damaged index: page " + std::to_string(parent) + " has a child pointer ";
    if (child == INVALID_PAGE_ID) throw DatabaseException(at + "that is INVALID_PAGE_ID");
    throw DatabaseException(at + "to page " + std::to_string(child) + ", past the file's " + std::to_string(num_pages) + " pages");
}

// Builds and throws a leaf chain error, kept off the hot path.
[[noreturn]] __attribute__((cold, noinline)) void ThrowBadChain(page_id_t leaf, page_id_t pages, bool past_end) {
    if (past_end) {
        throw DatabaseException("damaged index: leaf chain goes to page " + std::to_string(leaf) + ", past the file's " + std::to_string(pages) + " pages");
    }
    throw DatabaseException("damaged index: leaf chain longer than the file's " + std::to_string(pages) + " pages");
}

// Builds and throws the error for a root past the end of the file.
[[noreturn]] __attribute__((cold, noinline)) void ThrowBadRoot(page_id_t root, page_id_t pages) {
    throw DatabaseException("damaged index: root page " + std::to_string(root) + " is past the file's " + std::to_string(pages) + " pages");
}

// Builds and throws the error for a descent that never reaches a leaf.
[[noreturn]] __attribute__((cold, noinline)) void ThrowTooDeep() {
    throw DatabaseException("damaged index: descent deeper than " + std::to_string(kMaxDepth) + " levels");
}

// Throws unless an internal page's child pointer names a page the file has.
// INVALID_PAGE_ID is the largest id, so one compare covers it.
inline void CheckChild(page_id_t parent, page_id_t child, page_id_t num_pages) {
    if (child >= num_pages) ThrowBadChild(parent, child, num_pages);
}

// Counts a leaf on a walk; refuses a page past the end, or a cycle.
inline void StepToLeaf(page_id_t leaf, uint64_t& walked, const BufferPoolManager& bpm) {
    const page_id_t pages = bpm.GetNumPages();
    if (leaf >= pages) ThrowBadChain(leaf, pages, true);
    if (++walked > pages) ThrowBadChain(leaf, pages, false);
}

// The first leaf with a key, from start along the chain, or INVALID_PAGE_ID.
page_id_t FirstNonEmptyLeaf(BufferPoolManager* bpm, TypeId key_type, page_id_t start, uint64_t& walked) {
    for (page_id_t leaf = start; leaf != INVALID_PAGE_ID;) {
        StepToLeaf(leaf, walked, *bpm);
        const PagePin pin(bpm, leaf);
        BTreeLeafPage::CheckKeyCount(pin.Data(), key_type);
        if (BTreeLeafPage::GetNumKeys(pin.Data()) > 0) return leaf;
        leaf = BTreeLeafPage::GetNextLeafPageId(pin.Data());
    }
    return INVALID_PAGE_ID;
}

// True when the leaf has no key above `key`, so more entries for `key` may start in the next leaf.
bool MayContinueInNextLeaf(const char* leaf_data, const Value& key, TypeId key_type) {
    const auto num_keys = BTreeLeafPage::GetNumKeys(leaf_data);
    return num_keys == 0 || BTreeLeafPage::GetKey(leaf_data, num_keys - 1, key_type) <= key;
}

}  // namespace


BTreeIndex::BTreeIndex(BufferPoolManager* bpm, TypeId key_type, page_id_t root_page_id) : bpm_(bpm), root_page_id_(root_page_id), key_type_(key_type) {}


bool BTreeIndex::IsLeafPage(const char* page_data) {
    auto* hdr = SlottedPage::GetHeader(page_data);
    return (hdr->flags & 1u) != 0;
}

page_id_t BTreeIndex::GetRootPageId() const {
    return root_page_id_;
}

page_id_t BTreeIndex::Descend(const Value* key, bool go_right_on_equal, std::vector<page_id_t>* path) {
    if (path != nullptr) path->clear();
    if (root_page_id_ >= bpm_->GetNumPages()) ThrowBadRoot(root_page_id_, bpm_->GetNumPages());

    auto current_id = root_page_id_;
    for (int depth = 1;; ++depth) {
        if (depth > kMaxDepth) ThrowTooDeep();
        page_id_t child = INVALID_PAGE_ID;
        {
            const PagePin pin(bpm_, current_id);
            const char* data = pin.Data();
            if (IsLeafPage(data)) return current_id;

            if (path != nullptr) path->push_back(current_id);
            if (key == nullptr) {
                BTreeInternalPage::CheckKeyCount(data, key_type_);
                child = BTreeInternalPage::GetChild(data, 0, key_type_);
            } else if (go_right_on_equal) {
                child = BTreeInternalPage::Lookup(data, *key, key_type_);
            } else {
                child = BTreeInternalPage::LookupLeftmost(data, *key, key_type_);
            }
        }
        CheckChild(current_id, child, bpm_->GetNumPages());
        current_id = child;
    }
}

page_id_t BTreeIndex::FindLeaf(const Value& key, std::vector<page_id_t>& path) {
    return Descend(&key, true, &path);
}

// Descends left of any separator equal to the key, so copies of it spread over several leaves are not skipped.
page_id_t BTreeIndex::FindLeftmostLeaf(const Value& key) {
    return Descend(&key, false, nullptr);
}

page_id_t BTreeIndex::FindLeftmostLeaf() {
    if (root_page_id_ == INVALID_PAGE_ID) {
        return INVALID_PAGE_ID;
    }
    return Descend(nullptr, false, nullptr);
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

    // The entry can be in any leaf that holds copies of the key, so walk them from the first.
    auto leaf_id = FindLeftmostLeaf(key);
    uint64_t walked = 0;
    while (leaf_id != INVALID_PAGE_ID) {
        StepToLeaf(leaf_id, walked, *bpm_);
        PagePin pin(bpm_, leaf_id);
        auto* data = pin.Data();
        BTreeLeafPage::CheckKeyCount(data, key_type_);

        const auto deleted = BTreeLeafPage::Delete(data, key, rid, key_type_);
        if (deleted) pin.MarkDirty();
        leaf_id = (!deleted && MayContinueInNextLeaf(data, key, key_type_)) ? BTreeLeafPage::GetNextLeafPageId(data) : INVALID_PAGE_ID;
    }
}



std::vector<RID> BTreeIndex::PointLookup(const Value& key) {
    if (root_page_id_ == INVALID_PAGE_ID) {
        return {};
    }

    std::vector<RID> result;
    auto leaf_id = FindLeftmostLeaf(key);
    uint64_t walked = 0;
    while (leaf_id != INVALID_PAGE_ID) {
        StepToLeaf(leaf_id, walked, *bpm_);
        const PagePin pin(bpm_, leaf_id);
        const char* data = pin.Data();
        BTreeLeafPage::CheckKeyCount(data, key_type_);

        const auto rids = BTreeLeafPage::Lookup(data, key, key_type_);
        result.insert(result.end(), rids.begin(), rids.end());
        leaf_id = MayContinueInNextLeaf(data, key, key_type_) ? BTreeLeafPage::GetNextLeafPageId(data) : INVALID_PAGE_ID;
    }
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
    const PagePin pin(bpm_, current_page_id_);
    const char* data = pin.Data();
    BTreeLeafPage::CheckKeyCount(data, key_type_);
    return {BTreeLeafPage::GetKey(data, current_slot_, key_type_), BTreeLeafPage::GetRID(data, current_slot_, key_type_)};
}

BTreeIndex::Iterator& BTreeIndex::Iterator::operator++() {
    page_id_t next = INVALID_PAGE_ID;
    {
        const PagePin pin(bpm_, current_page_id_);
        const char* data = pin.Data();
        BTreeLeafPage::CheckKeyCount(data, key_type_);
        if (current_slot_ + 1 < BTreeLeafPage::GetNumKeys(data)) {
            ++current_slot_;
            return *this;
        }
        next = BTreeLeafPage::GetNextLeafPageId(data);
    }

    // Skip empty leaves left by deletions
    current_page_id_ = FirstNonEmptyLeaf(bpm_, key_type_, next, leaves_walked_);
    current_slot_ = 0;
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

    uint64_t walked = 0;
    const auto leaf_id = FirstNonEmptyLeaf(bpm_, key_type_, FindLeftmostLeaf(), walked);
    return leaf_id == INVALID_PAGE_ID ? End() : Iterator(bpm_, key_type_, leaf_id, 0);
}

BTreeIndex::Iterator BTreeIndex::Begin(const Value& low_key) {
    if (root_page_id_ == INVALID_PAGE_ID) {
        return End();
    }

    const auto leaf_id = FindLeftmostLeaf(low_key);
    uint16_t pos = 0;
    uint16_t num_keys = 0;
    page_id_t next = INVALID_PAGE_ID;
    {
        const PagePin pin(bpm_, leaf_id);
        const char* data = pin.Data();
        BTreeLeafPage::CheckKeyCount(data, key_type_);
        pos = BTreeLeafPage::LowerBound(data, low_key, key_type_);
        num_keys = BTreeLeafPage::GetNumKeys(data);
        next = BTreeLeafPage::GetNextLeafPageId(data);
    }
    if (pos < num_keys) {
        return Iterator(bpm_, key_type_, leaf_id, pos);
    }

    // Skip empty leaves left by deletions
    uint64_t walked = 1;
    const auto next_id = FirstNonEmptyLeaf(bpm_, key_type_, next, walked);
    return next_id == INVALID_PAGE_ID ? End() : Iterator(bpm_, key_type_, next_id, 0);
}

BTreeIndex::Iterator BTreeIndex::End() {
    return Iterator(bpm_, key_type_, INVALID_PAGE_ID, 0);
}

}  // namespace shilmandb
