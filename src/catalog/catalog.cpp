#include "catalog/catalog.hpp"
#include "common/exception.hpp"
#include "storage/columnar/bulk_loader.hpp"
#include "storage/columnar/column_vector.hpp"
#include "storage/columnar/columnar_file.hpp"
#include "storage/columnar/field_parser.hpp"
#include <algorithm>
#include <optional>
#include <unordered_set>
#include <utility>

namespace shilmandb {

namespace {

// The executor schema a saved table's columns describe.
Schema SchemaOf(const ColumnarTableMeta& meta) {
    std::vector<Column> columns;
    columns.reserve(meta.columns.size());
    for (const ColumnMeta& col : meta.columns) {
        columns.emplace_back(col.name, col.type, col.nullable);
    }
    return Schema(std::move(columns));
}

// Same column count, names and storage types; nullability may differ.
bool SameColumns(const Schema& a, const Schema& b) {
    return std::equal(
        a.GetColumns().begin(), a.GetColumns().end(), b.GetColumns().begin(), b.GetColumns().end(),
        [](const Column& x, const Column& y) { return x.name == y.name && x.storage_type == y.storage_type; }
    );
}

}  // namespace

Catalog::Catalog(BufferPoolManager* bpm, ColumnarFile* file) : bpm_(bpm), file_(file) {}

void Catalog::RestoreFromFile() {
    if (file_ == nullptr) return;
    const CatalogMeta& saved = file_->Catalog();
    for (const ColumnarTableMeta& meta : saved.tables) {
        (void)AddColumnar(meta);
    }

    // The decoder already checked that each index's table and column exist
    for (const IndexMeta& meta : saved.indexes) {
        const auto table = std::find_if(saved.tables.begin(), saved.tables.end(), [&](const ColumnarTableMeta& t) { return t.oid == meta.table_oid; });
        auto info = std::make_unique<IndexInfo>();
        info->index_name = meta.name;
        info->table_name = table->name;
        info->col_idx = meta.column_number - 1;
        info->column_name = table->columns[info->col_idx].name;
        info->index = std::make_unique<BTreeIndex>(bpm_, meta.key_type, meta.root_page);
        info->root_page_id = meta.root_page;
        info->oid = meta.oid;
        indexes_[meta.name] = std::move(info);
    }
}

TableInfo* Catalog::LoadColumnarTable(const std::string& name, const Schema& schema, const std::string& path, const BulkLoadOptions& options) {
    if (file_ == nullptr) {
        throw DatabaseException("columnar tables need a database file");
    }
    if (auto it = tables_.find(name); it != tables_.end()) {
        if (it->second->storage == StorageKind::ROW) {
            throw DatabaseException("table " + name + " already exists as a row-store table");
        }
        return ReplaceColumnar(*it->second, schema, path, options);
    }

    // The OID is taken only after the load works, so a failed load wastes none
    ColumnarTableMeta meta = BulkLoad(file_->Disk(), name, schema, path, options);
    meta.oid = file_->AllocateOid();

    CatalogMeta next = file_->Catalog();
    next.tables.push_back(meta);
    file_->Commit(std::move(next));
    return AddColumnar(meta);
}

TableInfo* Catalog::ReplaceColumnar(TableInfo& info, const Schema& schema, const std::string& path, const BulkLoadOptions& options) {
    if (!SameColumns(info.schema, schema)) {
        throw DatabaseException("table " + info.name + " exists with a different schema");
    }

    ColumnarTableMeta meta = BulkLoad(file_->Disk(), info.name, schema, path, options);
    meta.oid = info.oid;
    auto table = std::make_unique<ColumnarTable>(bpm_, meta);

    // New trees on new pages: a committed tree's pages are never written again
    const std::vector<IndexInfo*> indexes = GetTableIndexes(info.name);
    std::vector<std::unique_ptr<BTreeIndex>> trees;
    trees.reserve(indexes.size());
    for (const IndexInfo* index : indexes) {
        trees.push_back(BuildColumnarIndex(*table, index->col_idx));
    }
    bpm_->FlushAllPages();

    // Built before the commit, so nothing after it can throw
    Schema new_schema = SchemaOf(meta);
    CatalogMeta next = file_->Catalog();
    for (ColumnarTableMeta& saved : next.tables) {
        if (saved.oid == meta.oid) saved = meta;
    }
    for (size_t i = 0; i < indexes.size(); ++i) {
        for (IndexMeta& saved : next.indexes) {
            if (saved.oid == indexes[i]->oid) saved.root_page = trees[i]->GetRootPageId();
        }
    }
    file_->Commit(std::move(next));

    // Only now, with the commit done, does anyone see the new data
    info.schema = std::move(new_schema);
    info.stats = std::move(meta.stats);
    info.columnar = std::move(table);
    for (size_t i = 0; i < indexes.size(); ++i) {
        indexes[i]->root_page_id = trees[i]->GetRootPageId();
        indexes[i]->index = std::move(trees[i]);
    }
    return &info;
}

TableInfo* Catalog::AddColumnar(const ColumnarTableMeta& meta) {
    auto info = std::make_unique<TableInfo>();
    info->name = meta.name;
    info->schema = SchemaOf(meta);
    info->first_page_id = INVALID_PAGE_ID;
    info->stats = meta.stats;
    info->storage = StorageKind::COLUMNAR;
    info->oid = meta.oid;
    info->columnar = std::make_unique<ColumnarTable>(bpm_, meta);

    auto* ptr = info.get();
    tables_[meta.name] = std::move(info);
    return ptr;
}

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
    if (table_info->storage == StorageKind::COLUMNAR) {
        return CreateColumnarIndex(index_name, *table_info, col_idx);
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

IndexInfo* Catalog::CreateColumnarIndex(const std::string& index_name, const TableInfo& table, uint32_t col_idx) {
    auto btree = BuildColumnarIndex(*table.columnar, col_idx);
    // The commit only counts pages already on disk
    bpm_->FlushAllPages();

    IndexMeta meta;
    meta.oid = file_->AllocateOid();
    meta.name = index_name;
    meta.table_oid = table.oid;
    meta.column_number = col_idx + 1;
    meta.key_type = table.schema.GetColumn(col_idx).type;
    meta.root_page = btree->GetRootPageId();

    CatalogMeta next = file_->Catalog();
    next.indexes.push_back(meta);
    file_->Commit(std::move(next));

    auto info = std::make_unique<IndexInfo>();
    info->index_name = index_name;
    info->table_name = table.name;
    info->column_name = table.schema.GetColumn(col_idx).name;
    info->col_idx = col_idx;
    info->root_page_id = meta.root_page;
    info->index = std::move(btree);
    info->oid = meta.oid;

    auto* ptr = info.get();
    indexes_.emplace(index_name, std::move(info));
    return ptr;
}

std::unique_ptr<BTreeIndex> Catalog::BuildColumnarIndex(const ColumnarTable& table, uint32_t col_idx) const {
    const ColumnMeta& column = table.Meta().columns.at(col_idx);
    auto btree = std::make_unique<BTreeIndex>(bpm_, ExecutorType(column.type));
    ColumnVector chunk(column.type);
    for (size_t group = 0; group < table.RowGroupCount(); ++group) {
        chunk.Clear();
        table.ReadChunk(group, col_idx, chunk);
        // The tree has no NULL keys
        if (chunk.NullCount() > 0) {
            throw DatabaseException("Cannot create index on column '" + column.name + "': it holds a NULL");
        }
        for (size_t row = 0; row < chunk.Size(); ++row) {
            const RowPosition pos{static_cast<uint32_t>(group), static_cast<uint16_t>(row)};
            btree->Insert(ToExecutorValue(chunk, row), ToRid(pos));
        }
    }
    return btree;
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
    // A columnar table's stats come from its loader; there is no heap to scan
    if (table_info == nullptr || table_info->storage == StorageKind::COLUMNAR) {
        return;
    }

    auto& schema = table_info->schema;
    auto col_count = schema.GetColumnCount();

    uint64_t row_count = 0;
    uint64_t page_count = 0;
    std::optional<page_id_t> last_page;
    std::vector<std::unordered_set<Value>> distinct_sets(col_count);
    std::vector<std::optional<MinMax>> ranges(col_count);

    for (auto it = table_info->table->Begin(schema); it != table_info->table->End(); ++it) {
        auto tuple = *it;
        ++row_count;

        const page_id_t page = it.GetRID().page_id;
        if (page != last_page) {
            ++page_count;
            last_page = page;
        }

        for (uint32_t i = 0; i < col_count; ++i) {
            auto value = tuple.GetValue(schema, i);

            if (schema.GetColumn(i).type != TypeId::VARCHAR) {
                if (!ranges[i]) {
                    ranges[i] = MinMax{value, value};
                } else if (value < ranges[i]->min) {
                    ranges[i]->min = value;
                } else if (ranges[i]->max < value) {
                    ranges[i]->max = value;
                }
            }

            distinct_sets[i].insert(std::move(value));
        }
    }

    table_info->stats.row_count = row_count;
    table_info->stats.page_count = page_count;
    table_info->stats.distinct_counts.clear();
    table_info->stats.min_max.clear();
    for (uint32_t i = 0; i < col_count; ++i) {
        if (ranges[i]) {
            table_info->stats.min_max[schema.GetColumn(i).name] = *ranges[i];
        }
        table_info->stats.distinct_counts[schema.GetColumn(i).name] = static_cast<uint64_t>(distinct_sets[i].size());
    }
}

}  // namespace shilmandb
