#pragma once
#include "buffer/buffer_pool_manager.hpp"
#include "storage/table_heap.hpp"
#include "index/btree_index.hpp"
#include "catalog/schema.hpp"
#include "catalog/table_stats.hpp"
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>

namespace shilmandb {

struct TableInfo {
    std::string name;
    Schema schema;
    std::unique_ptr<TableHeap> table;
    page_id_t first_page_id;
    TableStats stats;
};

struct IndexInfo {
    std::string index_name;
    std::string table_name;
    std::string column_name;
    uint32_t col_idx;
    std::unique_ptr<BTreeIndex> index;
    page_id_t root_page_id;
};

class Catalog {
public:
    explicit Catalog(BufferPoolManager* bpm);

    [[nodiscard]] TableInfo* CreateTable(const std::string& name, const Schema& schema);
    [[nodiscard]] TableInfo* GetTable(const std::string& name);

    [[nodiscard]] IndexInfo* CreateIndex(const std::string& index_name, const std::string& table_name, const std::string& column_name);
    [[nodiscard]] IndexInfo* GetIndex(const std::string& index_name);
    [[nodiscard]] std::vector<IndexInfo*> GetTableIndexes(const std::string& table_name);
    
    void UpdateTableStats(const std::string& table_name);

private:
    BufferPoolManager* bpm_;
    std::unordered_map<std::string, std::unique_ptr<TableInfo>> tables_;
    std::unordered_map<std::string, std::unique_ptr<IndexInfo>> indexes_;
};

}  // namespace shilmandb
