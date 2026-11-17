#pragma once
#include "catalog/schema.hpp"
#include "storage/columnar/columnar_meta.hpp"
#include <optional>
#include <string>

namespace shilmandb {

class DiskManager;

struct BulkLoadOptions {
    char delimiter{'|'};
    unsigned threads{0};                    // 0 = std::thread::hardware_concurrency(); at most 256
    std::optional<std::string> null_text;   // a field equal to this is NULL; unset = no NULLs
};

// Reads a delimited text file into row groups written straight to disk. Nothing is committed.
// Fills row_groups, row_count, columns and stats; leaves oid 0. Throws DatabaseException on bad input.
[[nodiscard]] ColumnarTableMeta BulkLoad(DiskManager* disk, const std::string& table_name, const Schema& schema, const std::string& path, const BulkLoadOptions& options);

}  // namespace shilmandb
