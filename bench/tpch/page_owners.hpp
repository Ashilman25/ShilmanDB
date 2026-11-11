#pragma once

#include <cstdint>
#include <fstream>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

#include "buffer/buffer_pool_manager.hpp"
#include "catalog/catalog.hpp"
#include "engine/database.hpp"
#include "index/btree_internal_page.hpp"
#include "storage/slotted_page.hpp"

namespace shilmandb::tpch {

enum class OwnedKind : uint8_t { TABLE, INNER, LEAF, OTHER };

struct PageOwner {
    OwnedKind kind{OwnedKind::OTHER};
    int object{-1};  // position in the tables, then tables.size() + position in the indexes
    std::string owner;
};

namespace detail {

// Pins a page, hands its bytes to fn, and unpins it.
template <typename Fn>
void WithPage(BufferPoolManager* bpm, page_id_t id, Fn&& fn) {
    Page* page = bpm->FetchPage(id);
    if (page == nullptr) throw std::runtime_error("page owners: cannot fetch page " + std::to_string(id));
    try {
        fn(page->GetData());
    } catch (...) {
        (void)bpm->UnpinPage(id, false);
        throw;
    }
    (void)bpm->UnpinPage(id, false);
}

// Records one page's owner; throws on a repeat or an id past the file.
inline void Claim(std::vector<PageOwner>& owners, page_id_t id, OwnedKind kind, int object, const std::string& name) {
    if (id >= owners.size()) throw std::runtime_error("page owners: page " + std::to_string(id) + " is past the file");
    if (owners[id].kind != OwnedKind::OTHER) {
        throw std::runtime_error("page owners: page " + std::to_string(id) + " reached twice");
    }
    owners[id] = {kind, object, name};
}

}  // namespace detail

// Names the table or index that owns every page of the file.
[[nodiscard]] inline std::vector<PageOwner> CollectPageOwners(Database& db, const std::vector<std::string>& tables, const std::vector<std::string>& indexes) {
    auto* bpm = db.GetBufferPoolManager();
    std::vector<PageOwner> owners(db.GetDiskManager()->GetNumPages());

    for (size_t t = 0; t < tables.size(); ++t) {
        auto* info = db.GetCatalog()->GetTable(tables[t]);
        if (info == nullptr) throw std::runtime_error("page owners: no table " + tables[t]);
        for (page_id_t id = info->first_page_id; id != INVALID_PAGE_ID;) {
            detail::Claim(owners, id, OwnedKind::TABLE, static_cast<int>(t), tables[t]);
            page_id_t next = INVALID_PAGE_ID;
            detail::WithPage(bpm, id, [&](const char* data) { next = SlottedPage::GetHeader(data)->next_page_id; });
            id = next;
        }
    }

    for (size_t x = 0; x < indexes.size(); ++x) {
        auto* info = db.GetCatalog()->GetIndex(indexes[x]);
        if (info == nullptr) throw std::runtime_error("page owners: no index " + indexes[x]);

        const auto* table = db.GetCatalog()->GetTable(info->table_name);
        if (table == nullptr) throw std::runtime_error("page owners: no table " + info->table_name + " for index " + indexes[x]);

        const TypeId key_type = table->schema.GetColumn(info->col_idx).type;
        const int object = static_cast<int>(tables.size() + x);

        std::queue<page_id_t> pending;
        pending.push(info->index->GetRootPageId());
        while (!pending.empty()) {
            const page_id_t id = pending.front();
            pending.pop();
            bool leaf = false;

            std::vector<page_id_t> children;
            detail::WithPage(bpm, id, [&](const char* data) {
                leaf = (SlottedPage::GetHeader(data)->flags & 1) != 0;
                if (leaf) return;
                const uint16_t keys = BTreeInternalPage::GetNumKeys(data);
                for (uint16_t i = 0; i <= keys; ++i) children.push_back(BTreeInternalPage::GetChild(data, i, key_type));
            });

            detail::Claim(owners, id, leaf ? OwnedKind::LEAF : OwnedKind::INNER, object, indexes[x]);
            for (const page_id_t child : children) pending.push(child);
        }
    }
    return owners;
}

// Writes page_id,kind,object,owner with one line per page.
inline void WritePageOwners(const std::vector<PageOwner>& owners, const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("page owners: cannot write " + path);
    out << "page_id,kind,object,owner\n";
    for (size_t i = 0; i < owners.size(); ++i) {
        const auto& o = owners[i];
        const char* kind = o.kind == OwnedKind::TABLE ? "table" : o.kind == OwnedKind::INNER ? "inner" : o.kind == OwnedKind::LEAF ? "leaf" : "other";
        out << i << ',' << kind << ',' << o.object << ',' << o.owner << '\n';
    }
}

}  // namespace shilmandb::tpch
