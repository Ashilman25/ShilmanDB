#pragma once
#include "common/config.hpp"
#include "types/value.hpp"
#include "storage/slotted_page.hpp"  

namespace shilmandb {

class BTreeInternalPage {
public:
    static void Init(char* page_data, page_id_t page_id);

    static uint16_t GetNumKeys(const char* page_data);
    static void SetNumKeys(char* page_data, uint16_t n);

    static Value GetKey(const char* page_data, uint16_t key_idx, TypeId key_type);
    static void SetKey(char* page_data, uint16_t key_idx, const Value& key, TypeId key_type);

    static page_id_t GetChild(const char* page_data, uint16_t child_idx, TypeId key_type);
    static void SetChild(char* page_data, uint16_t child_idx, page_id_t child_page_id, TypeId key_type);

    [[nodiscard]] static page_id_t Lookup(const char* page_data, const Value& key, TypeId key_type);

    static void InsertAfter(char* page_data, page_id_t left_child, const Value& key, page_id_t right_child, TypeId key_type);

    [[nodiscard]] static uint16_t MaxKeys(TypeId key_type);
    [[nodiscard]] static bool NeedsSplit(const char* page_data, TypeId key_type);

private:
    static uint16_t KeySize(TypeId key_type);
    static size_t KeyOffset(uint16_t key_idx, TypeId key_type);
    static size_t ChildOffset(uint16_t child_idx, TypeId key_type);
    static void WriteKeyRaw(char* dst, const Value& key, TypeId key_type);
    static constexpr size_t DATA_START = PAGE_HEADER_SIZE + 2;  // after header + num_keys
    static_assert(PAGE_HEADER_SIZE == 16, "BTreeInternalPage layout assumes 16-byte PageHeader");
};

}  // namespace shilmandb
