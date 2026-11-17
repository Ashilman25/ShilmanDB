#pragma once
#include "types/column_type.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace shilmandb {

// One column's values in memory, with a NULL flag per row.
// Numbers sit in one typed array; strings share one byte arena plus offsets.
class ColumnVector {
public:
    explicit ColumnVector(ColumnType type);

    [[nodiscard]] const ColumnType& Type() const { return type_; }
    [[nodiscard]] PhysicalKind Kind() const { return kind_; }
    [[nodiscard]] size_t Size() const { return nulls_.size(); }

    // Makes room for n rows; string bytes grow as they come.
    void Reserve(size_t n);
    // Drops every row but keeps the memory.
    void Clear();

    // Stores a zero or empty value and marks the row NULL.
    void AppendNull();

    // The append for each physical kind; the wrong kind is a Debug assert.
    void AppendInt(int64_t v) {
        assert(kind_ == PhysicalKind::INT64 && "AppendInt on a column that isn't INT64");
        ints_.push_back(v);
        nulls_.push_back(0);
    }
    void AppendWide(__int128 v) {
        assert(kind_ == PhysicalKind::INT128 && "AppendWide on a column that isn't INT128");
        wides_.push_back(v);
        nulls_.push_back(0);
    }
    void AppendDouble(double v) {
        assert(kind_ == PhysicalKind::DOUBLE && "AppendDouble on a column that isn't DOUBLE");
        doubles_.push_back(v);
        nulls_.push_back(0);
    }
    void AppendString(std::string_view v) {
        assert(kind_ == PhysicalKind::STRING && "AppendString on a column that isn't STRING");
        bytes_.insert(bytes_.end(), v.begin(), v.end());
        offsets_.push_back(bytes_.size());
        nulls_.push_back(0);
    }

    [[nodiscard]] bool IsNull(size_t i) const {
        assert(i < Size());
        return nulls_[i] != 0;
    }
    [[nodiscard]] size_t NullCount() const { return null_count_; }

    // The getter for each physical kind; a NULL row reads as zero or empty.
    [[nodiscard]] int64_t GetInt(size_t i) const {
        assert(kind_ == PhysicalKind::INT64 && i < ints_.size());
        return ints_[i];
    }
    [[nodiscard]] __int128 GetWide(size_t i) const {
        assert(kind_ == PhysicalKind::INT128 && i < wides_.size());
        return wides_[i];
    }
    [[nodiscard]] double GetDouble(size_t i) const {
        assert(kind_ == PhysicalKind::DOUBLE && i < doubles_.size());
        return doubles_[i];
    }
    [[nodiscard]] std::string_view GetString(size_t i) const {
        assert(kind_ == PhysicalKind::STRING && i + 1 < offsets_.size());
        return {bytes_.data() + offsets_[i], offsets_[i + 1] - offsets_[i]};
    }

private:
    ColumnType type_;
    PhysicalKind kind_;
    std::vector<int64_t> ints_;
    std::vector<__int128> wides_;
    std::vector<double> doubles_;
    std::vector<char> bytes_;       // every string's bytes, back to back
    std::vector<size_t> offsets_;   // string i is bytes_[offsets_[i], offsets_[i + 1])
    std::vector<uint8_t> nulls_;    // 1 where the row is NULL
    size_t null_count_{0};
};

}  // namespace shilmandb
