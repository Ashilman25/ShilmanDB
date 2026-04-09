#pragma once
#include "common/config.hpp"
#include "common/rid.hpp"
#include "types/value.hpp"
#include "storage/slotted_page.hpp"
#include <vector>

namespace shilmandb {

class BTreeLeafPage {
public:
    static void Init(char* page_data, page_id_t page_id);

    static uint16_t GetNumKeys(const char* page_data);
    static void SetNumKeys(char* page_data, uint16_t n);

    static page_id_t GetNextLeafPageId(const char* page_data);
    static void SetNextLeafPageId(char* page_data, page_id_t next_leaf);

    static Value GetKey(const char* page_data, uint16_t key_idx, TypeId key_type);
    static RID GetRID(const char* page_data, uint16_t key_idx, TypeId key_type);

    static void SetKey(char* page_data, uint16_t key_idx, const Value& key, TypeId key_type);
    static void SetRID(char* page_data, uint16_t key_idx, const RID& rid, TypeId key_type);

    [[nodiscard]] static bool Insert(char* page_data, const Value& key, const RID& rid, TypeId key_type);
    [[nodiscard]] static bool Delete(char* page_data, const Value& key, const RID& rid, TypeId key_type);
    [[nodiscard]] static std::vector<RID> Lookup(const char* page_data, const Value& key, TypeId key_type);
    [[nodiscard]] static uint16_t LowerBound(const char* page_data, const Value& key, TypeId key_type);

    [[nodiscard]] static uint16_t MaxKeys(TypeId key_type);
    [[nodiscard]] static bool NeedsSplit(const char* page_data, TypeId key_type);

private:
    static uint16_t KeySize(TypeId key_type);
    static void WriteKeyRaw(char* dst, const Value& key, TypeId key_type);
    static constexpr size_t RID_SIZE = 6;  // page_id(4B) + slot_id(2B)
    static constexpr size_t DATA_START = PAGE_HEADER_SIZE + 2 + 4;  // after header + num_keys + next_leaf
    static size_t EntryOffset(uint16_t idx, TypeId key_type);
    
    static_assert(PAGE_HEADER_SIZE == 16, "BTreeLeafPage layout assumes 16-byte PageHeader");
    static_assert(RID_SIZE == 6, "BTreeLeafPage layout assumes 6-byte RID (page_id + slot_id)");
};

}  // namespace shilmandb
