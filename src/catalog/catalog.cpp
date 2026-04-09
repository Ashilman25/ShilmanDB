#include "catalog/catalog.hpp"
#include "common/exception.hpp"
#include <unordered_set>

namespace shilmandb {

Catalog::Catalog(BufferPoolManager* bpm) : bpm_(bpm) {}

TableInfo* Catalog::CreateTable(const std::string& name, const Schema& schema) {
    if (tables_.find(name) != tables_.end()) {
        return nullptr;
    }

    auto heap = std::make_unique<TableHeap>(bpm_);
    auto info = std::make_unique<TableInfo>();
    info->name = name;
    info->schema = schema;
    info->first_page_id = heap->GetFirstPageId();
    info->table = std::move(heap);

    auto* ptr = info.get();
    tables_.emplace(name, std::move(info));
    return ptr;
}

TableInfo* Catalog::GetTable(const std::string& name) {
    auto it = tables_.find(name);
    if (it == tables_.end()) {
        return nullptr;
    }
    return it->second.get();
}

IndexInfo* Catalog::CreateIndex(const std::string& index_name, const std::string& table_name, const std::string& column_name) {
    if (indexes_.find(index_name) != indexes_.end()) {
        return nullptr;
    }

    auto* table_info = GetTable(table_name);
    if (table_info == nullptr) {
        return nullptr;
    }

    auto col_idx = table_info->schema.GetColumnIndex(column_name);
    auto key_type = table_info->schema.GetColumn(col_idx).type;

    if (key_type == TypeId::VARCHAR || key_type == TypeId::INVALID) {
        throw DatabaseException("Cannot create index on column '" + column_name + "': unsupported key type");
    }

    auto btree = std::make_unique<BTreeIndex>(bpm_, key_type);

    // populate index from existing data
    for (auto it = table_info->table->Begin(table_info->schema); it != table_info->table->End(); ++it) {
        auto tuple = *it;
        auto key = tuple.GetValue(table_info->schema, col_idx);
        btree->Insert(key, it.GetRID());
    }

    auto info = std::make_unique<IndexInfo>();
    info->index_name = index_name;
    info->table_name = table_name;
    info->column_name = column_name;
    info->col_idx = col_idx;
    info->root_page_id = btree->GetRootPageId();
    info->index = std::move(btree);

    auto* ptr = info.get();
    indexes_.emplace(index_name, std::move(info));
    return ptr;
}

IndexInfo* Catalog::GetIndex(const std::string& index_name) {
    auto it = indexes_.find(index_name);
    if (it == indexes_.end()) {
        return nullptr;
    }
    return it->second.get();
}

std::vector<IndexInfo*> Catalog::GetTableIndexes(const std::string& table_name) {
    std::vector<IndexInfo*> result;
    for (auto& [name, info] : indexes_) {
        if (info->table_name == table_name) {
            result.push_back(info.get());
        }
    }
    return result;
}

void Catalog::UpdateTableStats(const std::string& table_name) {
    auto* table_info = GetTable(table_name);
    if (table_info == nullptr) {
        return;
    }

    auto& schema = table_info->schema;
    auto col_count = schema.GetColumnCount();

    uint64_t row_count = 0;
    std::vector<std::unordered_set<Value>> distinct_sets(col_count);

    for (auto it = table_info->table->Begin(schema); it != table_info->table->End(); ++it) {
        auto tuple = *it;
        ++row_count;

        for (uint32_t i = 0; i < col_count; ++i) {
            distinct_sets[i].insert(tuple.GetValue(schema, i));
        }
    }

    table_info->stats.row_count = row_count;
    table_info->stats.distinct_counts.clear();
    for (uint32_t i = 0; i < col_count; ++i) {
        table_info->stats.distinct_counts[schema.GetColumn(i).name] = static_cast<uint64_t>(distinct_sets[i].size());
    }
}

}  // namespace shilmandb
