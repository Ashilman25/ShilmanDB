#include "executor/data_chunk.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iterator>

#include "common/exception.hpp"

namespace shilmandb {

DataChunk::DataChunk(const Schema& schema, size_t capacity) : schema_(schema), capacity_(capacity) {
    const auto n_cols = schema_.GetColumnCount();
    columns_.resize(n_cols);
    for (auto& col : columns_) col.reserve(capacity_);
    selection_vector_.reserve(capacity_);
    row_starts_.reserve(capacity_);
    decoded_.assign(n_cols, 0);
}

void DataChunk::AppendRow(const char* data, uint32_t length) {
    assert(!has_selection_ && "AppendRow requires no active selection vector");
    assert(size_ < capacity_ && "AppendRow on full chunk");
    assert(!from_values_ && "a chunk holds row bytes or values, not both");

    if (size_ == 0) std::fill(decoded_.begin(), decoded_.end(), 0);
    assert(std::none_of(decoded_.begin(), decoded_.end(), [](uint8_t d) { return d != 0; }) && "AppendRow after columns were decoded");

    row_starts_.push_back(static_cast<uint32_t>(arena_.size()));
    arena_.insert(arena_.end(), data, data + length);
    ++size_;
}

void DataChunk::AppendTuple(const Tuple& tuple) {
    assert(!has_selection_ && "AppendTuple requires no active selection vector");
    assert(size_ < capacity_ && "AppendTuple on full chunk");
    assert(!from_values_ && "a chunk holds row bytes or values, not both");

    const auto n_cols = schema_.GetColumnCount();
    if (size_ == 0) std::fill(decoded_.begin(), decoded_.end(), 1);
    assert(std::all_of(decoded_.begin(), decoded_.end(), [](uint8_t d) { return d != 0; }) && "AppendTuple after AppendRow");

    row_starts_.push_back(static_cast<uint32_t>(arena_.size()));
    arena_.insert(arena_.end(), tuple.GetData(), tuple.GetData() + tuple.GetLength());
    for (uint32_t c = 0; c < n_cols; ++c) {
        columns_[c].push_back(tuple.GetValue(schema_, c));
    }
    ++size_;
}

void DataChunk::AppendValues(std::vector<Value>&& values) {
    assert(!has_selection_ && "AppendValues requires no active selection vector");
    assert(size_ < capacity_ && "AppendValues on full chunk");
    assert(row_starts_.empty() && "a chunk holds row bytes or values, not both");

    const auto n_cols = schema_.GetColumnCount();
    if (values.size() != n_cols) {
        Tuple throws(std::move(values), schema_);  // throws the usual count error
        return;
    }
    if (size_ == 0) std::fill(decoded_.begin(), decoded_.end(), 1);
    from_values_ = true;

    const bool types_match = [&] {
        for (uint32_t c = 0; c < n_cols; ++c) {
            if (values[c].type_ != schema_.GetColumn(c).type) return false;
        }
        return true;
    }();

    if (types_match) {
        for (uint32_t c = 0; c < n_cols; ++c) columns_[c].push_back(std::move(values[c]));
    } else {
        // Same result as encoding the row and reading it back.
        Tuple encoded(std::move(values), schema_);
        for (uint32_t c = 0; c < n_cols; ++c) columns_[c].push_back(encoded.GetValue(schema_, c));
    }
    ++size_;
}

namespace {

// Per-type cell readers shared by the batch and single-cell decoders.
template <typename T>
inline T ReadFixed(const char* row, uint32_t offset) {
    T v;
    std::memcpy(&v, row + offset, sizeof(T));
    return v;
}

inline Value ReadVarchar(const char* row, uint32_t offset) {
    const auto var_offset = ReadFixed<uint32_t>(row, offset);
    const auto len = ReadFixed<uint16_t>(row, var_offset);
    return Value(std::string(row + var_offset + 2, len));
}

}  // namespace

void DataChunk::EnsureDecoded(const std::vector<uint32_t>& cols) {
    for (uint32_t c : cols) {
        if (decoded_[c]) continue;
        assert(!from_values_ && "EnsureDecoded needs row bytes");
        decoded_[c] = 1;
        if (size_ == 0) continue;

        auto& out = columns_[c];
        out.resize(size_);
        const char* base = arena_.data();
        const uint32_t offset = schema_.GetOffset(c);
        const uint32_t* starts = row_starts_.data();

        // Visits each active row's index once.
        auto each_row = [&](auto&& fn) {
            if (has_selection_) {
                for (uint32_t r : selection_vector_) fn(r);
            } else {
                for (size_t r = 0; r < size_; ++r) fn(r);
            }
        };

        switch (schema_.GetColumn(c).type) {
            case TypeId::INTEGER:
                each_row([&](size_t r) { out[r] = Value(ReadFixed<int32_t>(base + starts[r], offset)); });
                break;
            case TypeId::BIGINT:
                each_row([&](size_t r) { out[r] = Value(ReadFixed<int64_t>(base + starts[r], offset)); });
                break;
            case TypeId::DECIMAL:
                each_row([&](size_t r) { out[r] = Value(ReadFixed<double>(base + starts[r], offset)); });
                break;
            case TypeId::DATE:
                each_row([&](size_t r) { out[r] = Value::MakeDate(ReadFixed<int32_t>(base + starts[r], offset)); });
                break;
            case TypeId::VARCHAR:
                each_row([&](size_t r) { out[r] = ReadVarchar(base + starts[r], offset); });
                break;
            default:
                throw DatabaseException("Unsupported type in DataChunk::EnsureDecoded");
        }
    }
}

Value DataChunk::DecodeCell(size_t col_idx, size_t physical_row) const {
    assert(HasRowStore() && "DecodeCell needs row bytes");
    assert(physical_row < size_ && "DecodeCell row out of range");
    const char* row = arena_.data() + row_starts_[physical_row];
    const uint32_t offset = schema_.GetOffset(col_idx);

    switch (schema_.GetColumn(col_idx).type) {
        case TypeId::INTEGER: return Value(ReadFixed<int32_t>(row, offset));
        case TypeId::BIGINT:  return Value(ReadFixed<int64_t>(row, offset));
        case TypeId::DECIMAL: return Value(ReadFixed<double>(row, offset));
        case TypeId::DATE:    return Value::MakeDate(ReadFixed<int32_t>(row, offset));
        case TypeId::VARCHAR: return ReadVarchar(row, offset);
        default:
            throw DatabaseException("Unsupported type in DataChunk::DecodeCell");
    }
}

Tuple DataChunk::MaterializeTuple(size_t logical_row_idx) const {
    assert(logical_row_idx < size() && "MaterializeTuple index out of range");
    const size_t physical = has_selection_ ? static_cast<size_t>(selection_vector_[logical_row_idx]) : logical_row_idx;

    if (HasRowStore()) {
        const size_t end = physical + 1 < size_ ? row_starts_[physical + 1] : arena_.size();
        Tuple t;
        t.DeserializeFrom(arena_.data() + row_starts_[physical], static_cast<uint32_t>(end - row_starts_[physical]), schema_);
        return t;
    }

    const auto n_cols = columns_.size();
    std::vector<Value> values;
    values.reserve(n_cols);
    for (size_t c = 0; c < n_cols; ++c) {
        assert(IsDecoded(c) && "column is not decoded");
        values.push_back(columns_[c][physical]);
    }
    return Tuple(std::move(values), schema_);
}

void DataChunk::SetSelectionVector(std::vector<uint32_t> sel) {
    assert((sel.empty() || *std::max_element(sel.begin(), sel.end()) < size_) && "SetSelectionVector: index out of range");
    assert((!has_selection_ || std::includes(selection_vector_.begin(), selection_vector_.end(), sel.begin(), sel.end())) && "SetSelectionVector: new selection must be a subset of the current one");
    selection_vector_ = std::move(sel);
    has_selection_ = true;
}

void DataChunk::Flatten() {
    if (!has_selection_) return;

    const size_t new_size = selection_vector_.size();
    for (size_t i = 0; i < new_size; ++i) {
        assert(selection_vector_[i] >= i && "Flatten requires selection_vector_[i] >= i for in-place rewrite");
    }
    for (size_t c = 0; c < columns_.size(); ++c) {
        if (!decoded_[c]) continue;
        auto& col = columns_[c];
        for (size_t i = 0; i < new_size; ++i) {
            col[i] = col[selection_vector_[i]];
        }
        col.resize(new_size);
    }
    if (HasRowStore()) {
        // Slide each kept row down over the dropped ones.
        size_t write = 0;
        for (size_t i = 0; i < new_size; ++i) {
            const uint32_t r = selection_vector_[i];
            const size_t start = row_starts_[r];
            const size_t end = r + 1 < size_ ? row_starts_[r + 1] : arena_.size();
            std::memmove(arena_.data() + write, arena_.data() + start, end - start);
            row_starts_[i] = static_cast<uint32_t>(write);
            write += end - start;
        }
        arena_.resize(write);
        row_starts_.resize(new_size);
    }
    size_ = new_size;
    selection_vector_.clear();
    has_selection_ = false;
}

void DataChunk::Reset() {
    for (auto& col : columns_) col.clear();   // retains capacity
    selection_vector_.clear();                // retains capacity
    arena_.clear();                           // retains capacity
    row_starts_.clear();
    std::fill(decoded_.begin(), decoded_.end(), 0);
    from_values_ = false;
    size_ = 0;
    has_selection_ = false;
}

}  // namespace shilmandb
