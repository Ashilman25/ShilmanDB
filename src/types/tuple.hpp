#pragma once
#include "types/value.hpp"
#include "catalog/schema.hpp"
#include <vector>
#include <cstdint>

namespace shilmandb {

class Tuple {
public:
    Tuple() = default;
    Tuple(std::vector<Value> values, const Schema& schema);

    // Joins two rows into one: the left row's columns, then the right row's
    static Tuple Concat(const Tuple& left, const Schema& left_schema, const Tuple& right, const Schema& right_schema);

    Value GetValue(const Schema& schema, uint32_t col_idx) const;

    void SerializeTo(char* storage) const;
    void DeserializeFrom(const char* storage, uint32_t length, const Schema& schema);

    uint32_t GetLength() const;
    const char* GetData() const;
    bool IsEmpty() const;

private:
    std::vector<char> data_;
};

}  // namespace shilmandb
