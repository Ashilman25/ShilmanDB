#include "index/btree_leaf_page.hpp"
#include "common/exception.hpp"
#include <cassert>
#include <cstring>

namespace shilmandb {


static constexpr size_t NEXT_LEAF_OFFSET = PAGE_HEADER_SIZE + 2;  // after header + num_keys uint16_t


uint16_t BTreeLeafPage::KeySize(TypeId key_type) {
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
            throw DatabaseException("BTreeLeafPage: unsupported key type");
    }
}

// Layout of each entry in the data region:
//   [ key (ks bytes) | page_id (4B) | slot_id (2B) ]
//
// DATA_START = PAGE_HEADER_SIZE + 2 (num_keys) + 4 (next_leaf page_id)

size_t BTreeLeafPage::EntryOffset(uint16_t idx, TypeId key_type) {
    return DATA_START + idx * (static_cast<size_t>(KeySize(key_type)) + RID_SIZE);
}

void BTreeLeafPage::WriteKeyRaw(char* dst, const Value& key, TypeId key_type) {
    switch (key_type) {
        case TypeId::INTEGER: 
            std::memcpy(dst, &key.integer_,sizeof(key.integer_)); 
            break;
        case TypeId::BIGINT:  
            std::memcpy(dst, &key.bigint_,sizeof(key.bigint_));  
            break;
        case TypeId::DECIMAL: 
            std::memcpy(dst, &key.decimal_, sizeof(key.decimal_)); 
            break;
        case TypeId::DATE:    
            std::memcpy(dst, &key.date_, sizeof(key.date_));    
            break;
        default: 
            throw DatabaseException("BTreeLeafPage: unsupported key type");
    }
}



void BTreeLeafPage::Init(char* page_data, page_id_t page_id) {
    std::memset(page_data, 0, PAGE_SIZE);
    auto* hdr = SlottedPage::GetHeader(page_data);
    hdr->page_id = page_id;
    hdr->flags = 1;  // bit 0 = 1 → leaf page
    SetNextLeafPageId(page_data, INVALID_PAGE_ID);
}


uint16_t BTreeLeafPage::GetNumKeys(const char* page_data) {
    uint16_t n = 0;
    std::memcpy(&n, page_data + PAGE_HEADER_SIZE, sizeof(n));
    return n;
}

void BTreeLeafPage::SetNumKeys(char* page_data, uint16_t n) {
    std::memcpy(page_data + PAGE_HEADER_SIZE, &n, sizeof(n));
}


page_id_t BTreeLeafPage::GetNextLeafPageId(const char* page_data) {
    page_id_t next = INVALID_PAGE_ID;
    std::memcpy(&next, page_data + NEXT_LEAF_OFFSET, sizeof(next));
    return next;
}

void BTreeLeafPage::SetNextLeafPageId(char* page_data, page_id_t next_leaf) {
    std::memcpy(page_data + NEXT_LEAF_OFFSET, &next_leaf, sizeof(next_leaf));
}


uint16_t BTreeLeafPage::MaxKeys(TypeId key_type) {
    const auto ks = static_cast<size_t>(KeySize(key_type));
    return static_cast<uint16_t>((PAGE_SIZE - DATA_START) / (ks + RID_SIZE));
}

bool BTreeLeafPage::NeedsSplit(const char* page_data, TypeId key_type) {
    return GetNumKeys(page_data) > MaxKeys(key_type);
}



Value BTreeLeafPage::GetKey(const char* page_data, uint16_t key_idx, TypeId key_type) {
    assert(key_idx < GetNumKeys(page_data));
    const char* src = page_data + EntryOffset(key_idx, key_type);

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
            throw DatabaseException("BTreeLeafPage: unsupported key type");
    }
}

void BTreeLeafPage::SetKey(char* page_data, uint16_t key_idx, const Value& key, TypeId key_type) {
    assert(key_idx < GetNumKeys(page_data));
    WriteKeyRaw(page_data + EntryOffset(key_idx, key_type), key, key_type);
}


RID BTreeLeafPage::GetRID(const char* page_data, uint16_t key_idx, TypeId key_type) {
    assert(key_idx < GetNumKeys(page_data));
    const char* src = page_data + EntryOffset(key_idx, key_type) + KeySize(key_type);

    page_id_t pid = INVALID_PAGE_ID;
    uint16_t sid = 0;
    std::memcpy(&pid, src, sizeof(pid));
    std::memcpy(&sid, src + sizeof(pid), sizeof(sid));
    return RID(pid, sid);
}

void BTreeLeafPage::SetRID(char* page_data, uint16_t key_idx, const RID& rid, TypeId key_type) {
    assert(key_idx < GetNumKeys(page_data));
    char* dst = page_data + EntryOffset(key_idx, key_type) + KeySize(key_type);
    std::memcpy(dst, &rid.page_id, sizeof(rid.page_id));
    std::memcpy(dst + sizeof(rid.page_id), &rid.slot_id, sizeof(rid.slot_id));
}


uint16_t BTreeLeafPage::LowerBound(const char* page_data, const Value& key, TypeId key_type) {
    uint16_t lo = 0;
    uint16_t hi = GetNumKeys(page_data);
    while (lo < hi) {
        uint16_t mid = lo + (hi - lo) / 2;

        if (GetKey(page_data, mid, key_type) < key) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}



bool BTreeLeafPage::Insert(char* page_data, const Value& key, const RID& rid, TypeId key_type) {
    auto num_keys = GetNumKeys(page_data);
    assert(num_keys <= MaxKeys(key_type) && "Insert on overfull page — caller must split first");
    auto ks = static_cast<size_t>(KeySize(key_type));
    auto entry_size = ks + RID_SIZE;

    auto pos = LowerBound(page_data, key, key_type);
    while (pos < num_keys && GetKey(page_data, pos, key_type) == key) {
        ++pos;
    }

    if (pos < num_keys) {
        auto src = EntryOffset(pos, key_type);
        auto len = static_cast<size_t>(num_keys - pos) * entry_size;
        std::memmove(page_data + src + entry_size, page_data + src, len);
    }

    WriteKeyRaw(page_data + EntryOffset(pos, key_type), key, key_type);

    //write RID
    //page_Id then slot_id
    auto rid_offset = EntryOffset(pos, key_type) + ks;
    std::memcpy(page_data + rid_offset, &rid.page_id, sizeof(rid.page_id));
    std::memcpy(page_data + rid_offset + sizeof(rid.page_id), &rid.slot_id, sizeof(rid.slot_id));

    SetNumKeys(page_data, num_keys + 1);
    return true;
}


bool BTreeLeafPage::Delete(char* page_data, const Value& key, const RID& rid, TypeId key_type) {
    auto num_keys = GetNumKeys(page_data);
    auto ks = static_cast<size_t>(KeySize(key_type));
    auto entry_size = ks + RID_SIZE;

    auto pos = LowerBound(page_data, key, key_type);

    while (pos < num_keys && GetKey(page_data, pos, key_type) == key) {
        if (GetRID(page_data, pos, key_type) == rid) {

            auto next = pos + 1;
            if (next < num_keys) {
                auto src = EntryOffset(next, key_type);
                auto dst = EntryOffset(pos, key_type);
                auto len = static_cast<size_t>(num_keys - next) * entry_size;
                std::memmove(page_data + dst, page_data + src, len);
            }

            SetNumKeys(page_data, num_keys - 1);
            return true;
        }
        ++pos;
    }
    return false;
}



std::vector<RID> BTreeLeafPage::Lookup(const char* page_data, const Value& key, TypeId key_type) {
    auto num_keys = GetNumKeys(page_data);
    auto pos = LowerBound(page_data, key, key_type);

    std::vector<RID> result;
    while (pos < num_keys && GetKey(page_data, pos, key_type) == key) {
        result.push_back(GetRID(page_data, pos, key_type));
        ++pos;
    }
    return result;
}

}  // namespace shilmandb
