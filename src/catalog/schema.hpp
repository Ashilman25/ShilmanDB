#pragma once
#include "types/types.hpp"
#include "types/column_type.hpp"
#include <string>
#include <vector>
#include <cstdint>

namespace shilmandb {


constexpr uint32_t FixedLengthForType(TypeId type) {
    switch (type) {
        case TypeId::INTEGER: 
            return 4;
        case TypeId::BIGINT: 
            return 8;
        case TypeId::DECIMAL: 
            return 8;
        case TypeId::DATE: 
            return 4;
        case TypeId::VARCHAR: 
            return 4;
        case TypeId::INVALID: 
            return 0;
    }
    return 0;
}

struct Column {
    std::string name;
    TypeId type;               // what the planner and executor see
    uint32_t fixed_length;
    bool nullable;
    ColumnType storage_type;   // what a columnar file stores

    // Storage type is the default for this TypeId (ColumnType{} for INVALID)
    Column(std::string name, TypeId type, bool nullable = false);

    // Executor type comes from the storage type
    Column(std::string name, ColumnType storage_type, bool nullable = false);
};

class Schema {
public:
    explicit Schema(std::vector<Column> columns);
    Schema() = default;

    uint32_t GetColumnCount() const;
    const Column& GetColumn(uint32_t col_idx) const;
    uint32_t GetColumnIndex(const std::string& col_name) const;
    uint32_t GetOffset(uint32_t col_idx) const;
    uint32_t GetFixedLength() const;
    const std::vector<Column>& GetColumns() const;

    // Fixed-part offset of each VARCHAR column's slot, in column order.
    const std::vector<uint32_t>& GetVarcharOffsets() const { return varchar_offsets_; }

private:
    std::vector<Column> columns_;
    std::vector<uint32_t> offsets_;
    std::vector<uint32_t> varchar_offsets_;
    uint32_t fixed_length_{0};

    void ComputeOffsets();
};

}  // namespace shilmandb
