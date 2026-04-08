#include "catalog/schema.hpp"
#include "common/exception.hpp"
#include <utility>

namespace shilmandb {



Column::Column(std::string name, TypeId type, bool nullable) : name{std::move(name)}, type{type}, fixed_length{FixedLengthForType(type)}, nullable{nullable} {}


Schema::Schema(std::vector<Column> columns) : columns_{std::move(columns)} {
    ComputeOffsets();
}

void Schema::ComputeOffsets() {
    offsets_.reserve(columns_.size());
    uint32_t offset = 0;

    for (const auto& col : columns_) {
        offsets_.push_back(offset);
        offset += col.fixed_length;
    }
    
    fixed_length_ = offset;
}

uint32_t Schema::GetColumnCount() const {
    return static_cast<uint32_t>(columns_.size());
}

const Column& Schema::GetColumn(uint32_t col_idx) const {
    if (col_idx >= columns_.size()) {
        throw DatabaseException("Column index " + std::to_string(col_idx) + " out of range");
    }
    return columns_[col_idx];
}

uint32_t Schema::GetColumnIndex(const std::string& col_name) const {
    for (uint32_t i = 0; i < columns_.size(); ++i) {
        if (columns_[i].name == col_name) {
            return i;
        }
    }
    throw DatabaseException("Column not found: " + col_name);
}

uint32_t Schema::GetOffset(uint32_t col_idx) const {
    if (col_idx >= offsets_.size()) {
        throw DatabaseException("Column index " + std::to_string(col_idx) + " out of range");
    }
    return offsets_[col_idx];
}

uint32_t Schema::GetFixedLength() const {
    return fixed_length_;
}

const std::vector<Column>& Schema::GetColumns() const {
    return columns_;
}

}  // namespace shilmandb
