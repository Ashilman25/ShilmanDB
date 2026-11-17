#pragma once
#include "catalog/table_stats.hpp"
#include "common/config.hpp"
#include "storage/columnar/encoding.hpp"
#include "types/column_type.hpp"
#include "types/types.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace shilmandb {

// Rows per row group; the last group of a table may be short.
inline constexpr uint32_t kRowGroupSize = 65536;

struct ColumnMeta {
    std::string name;
    uint32_t number{0};  // 1-based, in table order
    ColumnType type;
    bool nullable{false};
};

struct RowGroupMeta {
    uint32_t row_count{0};
    std::vector<ChunkMeta> chunks;  // one per column
};

struct ColumnarTableMeta {
    uint32_t oid{0};
    std::string name;
    std::vector<ColumnMeta> columns;
    uint64_t row_count{0};
    std::vector<RowGroupMeta> row_groups;
    TableStats stats;
};

struct IndexMeta {
    uint32_t oid{0};
    std::string name;
    uint32_t table_oid{0};
    uint32_t column_number{0};
    TypeId key_type{TypeId::INVALID};
    page_id_t root_page{INVALID_PAGE_ID};  // the current root, which moves when the root splits
    bool unique{false};
};

struct CatalogMeta {
    std::vector<ColumnarTableMeta> tables;
    std::vector<IndexMeta> indexes;
};

}  // namespace shilmandb
