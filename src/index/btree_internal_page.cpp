#include "index/btree_internal_page.hpp"
#include "common/exception.hpp"
#include <cassert>
#include <cstring>


namespace shilmandb {

uint16_t BTreeInternalPage::KeySize(TypeId key_type) {
    switch (key_type) {
        case TypeId::INTEGER: 
            return 4;
        case TypeId::BIGINT:  
            return 8;
        case TypeId::DECIMAL: 
            return 8;
        case TypeId::DATE:    
            return 4;
        default: 
            throw DatabaseException("BTreeInternalPage: unsupported key type");
    }
}

// Layout within the data region (after header + num_keys uint16_t):
// [ child_0 (4B) | key_0 (ks) | child_1 (4B) | key_1 (ks) | ... | child_n (4B) ]
size_t BTreeInternalPage::ChildOffset(uint16_t child_idx, TypeId key_type) {
    return DATA_START + child_idx * (static_cast<size_t>(KeySize(key_type)) + 4);
}

size_t BTreeInternalPage::KeyOffset(uint16_t key_idx, TypeId key_type) {
    return ChildOffset(key_idx, key_type) + 4;
}

void BTreeInternalPage::WriteKeyRaw(char* dst, const Value& key, TypeId key_type) {
    switch (key_type) {
        case TypeId::INTEGER:
            std::memcpy(dst, &key.integer_, sizeof(key.integer_)); break;
        case TypeId::BIGINT:  
            std::memcpy(dst, &key.bigint_,  sizeof(key.bigint_));  break;
        case TypeId::DECIMAL: 
            std::memcpy(dst, &key.decimal_, sizeof(key.decimal_)); break;
        case TypeId::DATE:    
            std::memcpy(dst, &key.date_,    sizeof(key.date_));    break;
        default: 
            throw DatabaseException("BTreeInternalPage: unsupported key type");
    }
}


void BTreeInternalPage::Init(char* page_data, page_id_t page_id) {
    std::memset(page_data, 0, PAGE_SIZE);
    auto* hdr = SlottedPage::GetHeader(page_data);
    hdr->page_id = page_id;
    hdr->flags = 0;  // bit 0 = 0 → non-leaf
}


uint16_t BTreeInternalPage::GetNumKeys(const char* page_data) {
    uint16_t n = 0;
    std::memcpy(&n, page_data + PAGE_HEADER_SIZE, sizeof(n));
    return n;
}

void BTreeInternalPage::SetNumKeys(char* page_data, uint16_t n) {
    std::memcpy(page_data + PAGE_HEADER_SIZE, &n, sizeof(n));
}


uint16_t BTreeInternalPage::MaxKeys(TypeId key_type) {
    const auto ks = static_cast<size_t>(KeySize(key_type));
    const size_t available = PAGE_SIZE - DATA_START;
    return static_cast<uint16_t>(((available - 4) / (ks + 4)) - 1);
}

bool BTreeInternalPage::NeedsSplit(const char* page_data, TypeId key_type) {
    return GetNumKeys(page_data) > MaxKeys(key_type);
}



Value BTreeInternalPage::GetKey(const char* page_data, uint16_t key_idx, TypeId key_type) {
    assert(key_idx < GetNumKeys(page_data));
    const char* src = page_data + KeyOffset(key_idx, key_type);

    switch (key_type) {
        case TypeId::INTEGER: {
            int32_t v = 0;
            std::memcpy(&v, src, sizeof(v));
            return Value(v);
        }
        case TypeId::BIGINT: {
            int64_t v = 0;
            std::memcpy(&v, src, sizeof(v));
            return Value(v);
        }
        case TypeId::DECIMAL: {
            double v = 0.0;
            std::memcpy(&v, src, sizeof(v));
            return Value(v);
        }
        case TypeId::DATE: {
            int32_t v = 0;
            std::memcpy(&v, src, sizeof(v));
            return Value::MakeDate(v);
        }
        default:
            throw DatabaseException("BTreeInternalPage: unsupported key type");
    }
}

void BTreeInternalPage::SetKey(char* page_data, uint16_t key_idx, const Value& key, TypeId key_type) {
    assert(key_idx < GetNumKeys(page_data));
    WriteKeyRaw(page_data + KeyOffset(key_idx, key_type), key, key_type);
}


page_id_t BTreeInternalPage::GetChild(const char* page_data, uint16_t child_idx, TypeId key_type) {
    assert(child_idx <= GetNumKeys(page_data));
    page_id_t child = INVALID_PAGE_ID;
    std::memcpy(&child, page_data + ChildOffset(child_idx, key_type), sizeof(child));
    return child;
}

void BTreeInternalPage::SetChild(char* page_data, uint16_t child_idx, page_id_t child_page_id, TypeId key_type) {
    assert(child_idx <= GetNumKeys(page_data));
    std::memcpy(page_data + ChildOffset(child_idx, key_type), &child_page_id, sizeof(child_page_id));
}



page_id_t BTreeInternalPage::Lookup(const char* page_data, const Value& key, TypeId key_type) {
    auto num_keys = GetNumKeys(page_data);
    assert(num_keys > 0 && "Lookup on empty internal node is undefined");

    uint16_t lo = 0;
    uint16_t hi = num_keys;

    while (lo < hi) {
        uint16_t mid = lo + (hi - lo) / 2;
        if (GetKey(page_data, mid, key_type) <= key) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return GetChild(page_data, lo, key_type);
}


void BTreeInternalPage::InsertAfter(char* page_data, page_id_t left_child, const Value& key, page_id_t right_child, TypeId key_type) {
    auto num_keys = GetNumKeys(page_data);
    assert(num_keys <= MaxKeys(key_type) && "InsertAfter on double-overfull page");
    auto ks = static_cast<size_t>(KeySize(key_type));
    auto stride = ks + 4;  // one key + one child pointer

    // Find the child index matching left_child
    uint16_t pos = 0;
    bool found = false;
    for (uint16_t i = 0; i <= num_keys; ++i) {
        if (GetChild(page_data, i, key_type) == left_child) {
            pos = i;
            found = true;
            break;
        }
    }
    assert(found && "InsertAfter: left_child not found in child array");

    auto shift_src = KeyOffset(pos, key_type);
    auto shift_end = ChildOffset(num_keys, key_type) + 4;  // end of last child pointer
    auto shift_len = shift_end - shift_src;

    if (shift_len > 0) {
        std::memmove(page_data + shift_src + stride, page_data + shift_src, shift_len);
    }


    WriteKeyRaw(page_data + shift_src, key, key_type);
    std::memcpy(page_data + shift_src + ks, &right_child, sizeof(right_child));
    SetNumKeys(page_data, num_keys + 1);
}

}  // namespace shilmandb
