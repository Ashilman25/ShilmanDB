#pragma once
#include "buffer/buffer_pool_manager.hpp"
#include "index/btree_internal_page.hpp"
#include "index/btree_leaf_page.hpp"
#include "types/value.hpp"
#include "common/rid.hpp"
#include "common/config.hpp"
#include <functional>
#include <vector>
#include <utility>

namespace shilmandb {

class BTreeIndex {
public:
    BTreeIndex(BufferPoolManager* bpm, TypeId key_type, page_id_t root_page_id = INVALID_PAGE_ID);

    void Insert(const Value& key, const RID& rid);
    void Delete(const Value& key, const RID& rid);
    [[nodiscard]] std::vector<RID> PointLookup(const Value& key);
    [[nodiscard]] page_id_t GetRootPageId() const;

    class Iterator {
    public:
        Iterator(BufferPoolManager* bpm, TypeId key_type, page_id_t page_id, uint16_t slot_idx);

        std::pair<Value, RID> operator*();
        Iterator& operator++();
        bool operator!=(const Iterator& other) const;
        bool IsEnd() const;

    private:
        BufferPoolManager* bpm_;
        TypeId key_type_;
        page_id_t current_page_id_;
        uint16_t current_slot_;
        uint64_t leaves_walked_{0};  // leaf changes so far, to catch a cycle in the chain
    };

    [[nodiscard]] Iterator Begin();
    [[nodiscard]] Iterator Begin(const Value& low_key);
    [[nodiscard]] Iterator End();

    // Calls fn(key, rid) in key order from the first key >= from (Value(): the start)
    // until it returns false. One pin per leaf, so fn must not use the pool.
    void ForEachEntry(const Value& from, const std::function<bool(const Value&, const RID&)>& fn);

    // Up to pieces - 1 sorted, distinct keys strictly between low and high,
    // cutting the range into pieces of about equal leaves. Value() leaves an end open.
    [[nodiscard]] std::vector<Value> SplitKeys(const Value& low, const Value& high, size_t pieces);

private:
    BufferPoolManager* bpm_;
    page_id_t root_page_id_;
    TypeId key_type_;

    page_id_t FindLeaf(const Value& key, std::vector<page_id_t>& path);
    page_id_t FindLeftmostLeaf();
    // The first leaf that can hold `key`. It may hold only smaller keys, then the entries for `key` start in a later leaf.
    page_id_t FindLeftmostLeaf(const Value& key);
    // Walks down to a leaf by key, or by child 0 when key is null.
    // Throws "damaged index" instead of looping or reading past a page.
    page_id_t Descend(const Value* key, bool go_right_on_equal, std::vector<page_id_t>* path);
    // Adds, in key order, each separator strictly inside the bounds that splits two leaves.
    // Reads only internal pages whose keys can fall in [low, high].
    void CollectSeparators(page_id_t page, size_t below, const Value& low, const Value& high, std::vector<Value>& out, uint64_t& visited);

    void SplitLeaf(page_id_t leaf_id, std::vector<page_id_t>& path);
    void InsertIntoParent(std::vector<page_id_t>& path, page_id_t left_child, const Value& key, page_id_t right_child);
    void CreateNewRoot(const Value& key, page_id_t left_child, page_id_t right_child);

    static bool IsLeafPage(const char* page_data);
};

}  // namespace shilmandb
