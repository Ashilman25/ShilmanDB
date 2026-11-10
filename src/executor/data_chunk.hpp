#pragma once
#include "catalog/schema.hpp"
#include "types/tuple.hpp"
#include <cassert>
#include <cstdint>
#include <vector>

namespace shilmandb {

class DataChunk {
public:
    static constexpr size_t kDefaultBatchSize = 1024;

    explicit DataChunk(const Schema& schema, size_t capacity = kDefaultBatchSize);

    // Stores the row's bytes and decodes nothing.
    void AppendRow(const char* data, uint32_t length);

    // Stores the row's bytes and decodes every column.
    void AppendTuple(const Tuple& tuple);

    // Takes already decoded values for every column (no row bytes kept).
    void AppendValues(std::vector<Value>&& values);

    // Decodes these columns for the active rows; decoded columns are skipped.
    void EnsureDecoded(const std::vector<uint32_t>& cols);
    
    // Decodes one cell straight from the row bytes, without caching it.
    [[nodiscard]] Value DecodeCell(size_t col_idx, size_t physical_row) const;
    [[nodiscard]] bool IsDecoded(size_t col_idx) const { return decoded_[col_idx] != 0; }
    [[nodiscard]] bool HasRowStore() const { return !row_starts_.empty(); }
    [[nodiscard]] Tuple MaterializeTuple(size_t logical_row_idx) const;

    // column access
    [[nodiscard]] size_t ColumnCount() const { return columns_.size(); }
    [[nodiscard]] const std::vector<Value>& GetColumn(size_t col_idx) const {
        assert((size_ == 0 || IsDecoded(col_idx)) && "column is not decoded");
        return columns_[col_idx];
    }
    [[nodiscard]] std::vector<Value>& GetMutableColumn(size_t col_idx) {
        assert((size_ == 0 || IsDecoded(col_idx)) && "column is not decoded");
        return columns_[col_idx];
    }
    [[nodiscard]] const Value& GetValue(size_t col_idx, size_t physical_row_idx) const {
        assert((size_ == 0 || IsDecoded(col_idx)) && "column is not decoded");
        return columns_[col_idx][physical_row_idx];
    }

    // selection vector management
    void SetSelectionVector(std::vector<uint32_t> sel);
    [[nodiscard]] bool HasSelectionVector() const { return has_selection_; }
    [[nodiscard]] const std::vector<uint32_t>& GetSelectionVector() const { return selection_vector_; }

    void Flatten();

    //size management
    void Reset();
    [[nodiscard]] size_t size() const { return has_selection_ ? selection_vector_.size() : size_; }
    [[nodiscard]] size_t capacity() const { return capacity_; }
    [[nodiscard]] bool IsFull() const { return size_ >= capacity_; }
    [[nodiscard]] const Schema& GetSchema() const { return schema_; }

private:
    Schema schema_;
    std::vector<std::vector<Value>> columns_;  // columns_[col_idx][row_idx]
    size_t size_{0};
    size_t capacity_;
    std::vector<uint32_t> selection_vector_;
    bool has_selection_{false};
    std::vector<char> arena_;             // every row's bytes, back to back
    std::vector<uint32_t> row_starts_;    // start of each row in arena_
    std::vector<uint8_t> decoded_;        // per column: filled for the active rows
    bool from_values_{false};             // rows came from AppendValues
};

}  // namespace shilmandb
