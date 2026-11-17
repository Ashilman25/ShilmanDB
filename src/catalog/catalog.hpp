#pragma once
#include "buffer/buffer_pool_manager.hpp"
#include "storage/table_heap.hpp"
#include "index/btree_index.hpp"
#include "catalog/schema.hpp"
#include "catalog/table_stats.hpp"
#include "storage/columnar/columnar_table.hpp"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>

namespace shilmandb {

class ColumnarFile;
struct BulkLoadOptions;

enum class StorageKind { ROW, COLUMNAR };

struct TableInfo {
    std::string name;
    Schema schema;
    std::unique_ptr<TableHeap> table;
    page_id_t first_page_id;
    TableStats stats;
    StorageKind storage{StorageKind::ROW};
    std::unique_ptr<ColumnarTable> columnar;  // set when COLUMNAR
    uint32_t oid{0};                          // 0 for row-store tables
};

struct IndexInfo {
    std::string index_name;
    std::string table_name;
    std::string column_name;
    uint32_t col_idx;
    std::unique_ptr<BTreeIndex> index;
    page_id_t root_page_id;
    uint32_t oid{0};  // 0 for row-store indexes
};

class Catalog {
public:
    explicit Catalog(BufferPoolManager* bpm, ColumnarFile* file = nullptr);

    // Registers the tables and indexes saved in the file; the database calls it once.
    void RestoreFromFile();

    // Loads a columnar table and commits it to the file.
    // An existing one with the same columns is replaced, indexes rebuilt, in one commit.
    TableInfo* LoadColumnarTable(const std::string& name, const Schema& schema, const std::string& path, const BulkLoadOptions& options);

    [[nodiscard]] TableInfo* CreateTable(const std::string& name, const Schema& schema);
    [[nodiscard]] TableInfo* GetTable(const std::string& name);

    [[nodiscard]] IndexInfo* CreateIndex(const std::string& index_name, const std::string& table_name, const std::string& column_name);
    [[nodiscard]] IndexInfo* GetIndex(const std::string& index_name);
    [[nodiscard]] std::vector<IndexInfo*> GetTableIndexes(const std::string& table_name);
    
    void UpdateTableStats(const std::string& table_name);

private:
    // Registers one columnar table from its saved meta.
    TableInfo* AddColumnar(const ColumnarTableMeta& meta);

    // Swaps in new data for a columnar table, with new trees for its indexes.
    TableInfo* ReplaceColumnar(TableInfo& info, const Schema& schema, const std::string& path, const BulkLoadOptions& options);

    // Indexes a columnar table's column and commits the index to the file.
    IndexInfo* CreateColumnarIndex(const std::string& index_name, const TableInfo& table, uint32_t col_idx);

    // Builds a new tree over one column, in row order, on fresh pages. Commits nothing.
    [[nodiscard]] std::unique_ptr<BTreeIndex> BuildColumnarIndex(const ColumnarTable& table, uint32_t col_idx) const;

    BufferPoolManager* bpm_;
    ColumnarFile* file_;
    std::unordered_map<std::string, std::unique_ptr<TableInfo>> tables_;
    std::unordered_map<std::string, std::unique_ptr<IndexInfo>> indexes_;
};

}  // namespace shilmandb
