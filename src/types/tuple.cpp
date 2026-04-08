#include "types/tuple.hpp"
#include "common/exception.hpp"
#include <cstring>
#include <limits>

namespace shilmandb {

Tuple::Tuple(std::vector<Value> values, const Schema& schema) {
    if (values.size() != schema.GetColumnCount()) {
        throw DatabaseException("Value count (" + std::to_string(values.size()) + ") does not match schema column count (" + std::to_string(schema.GetColumnCount()) + ")");
    }

    // total size = fixed prefix + variable region
    uint32_t total_size = schema.GetFixedLength();
    for (uint32_t i = 0; i < schema.GetColumnCount(); ++i) {
        if (schema.GetColumn(i).type == TypeId::VARCHAR) {
            total_size += 2 + static_cast<uint32_t>(values[i].varchar_.size());
        }
    }

    data_.resize(total_size);

    //fixed length prefix
    uint32_t var_offset = schema.GetFixedLength();
    for (uint32_t i = 0; i < schema.GetColumnCount(); ++i) {
        const auto& col = schema.GetColumn(i);
        uint32_t offset = schema.GetOffset(i);

        switch (col.type) {
            case TypeId::INTEGER:
                std::memcpy(data_.data() + offset, &values[i].integer_, 4);
                break;
            case TypeId::BIGINT:
                std::memcpy(data_.data() + offset, &values[i].bigint_, 8);
                break;
            case TypeId::DECIMAL:
                std::memcpy(data_.data() + offset, &values[i].decimal_, 8);
                break;
            case TypeId::DATE:
                std::memcpy(data_.data() + offset, &values[i].date_, 4);
                break;
            case TypeId::VARCHAR: {
                std::memcpy(data_.data() + offset, &var_offset, 4);
                var_offset += 2 + static_cast<uint32_t>(values[i].varchar_.size());
                break;
            }
            default:
                throw DatabaseException("Unsupported type in Tuple constructor");
        }
    }

    uint32_t write_pos = schema.GetFixedLength();
    for (uint32_t i = 0; i < schema.GetColumnCount(); ++i) {
        if (schema.GetColumn(i).type == TypeId::VARCHAR) {
            if (values[i].varchar_.size() > std::numeric_limits<uint16_t>::max()) {
                throw DatabaseException("VARCHAR exceeds maximum length of 65535 bytes");
            }

            auto len = static_cast<uint16_t>(values[i].varchar_.size());
            std::memcpy(data_.data() + write_pos, &len, 2);
            write_pos += 2;
            
            if (len > 0) {
                std::memcpy(data_.data() + write_pos, values[i].varchar_.data(), len);
                write_pos += len;
            }
        }
    }
}

Value Tuple::GetValue(const Schema& schema, uint32_t col_idx) const {
    const auto& col = schema.GetColumn(col_idx);
    uint32_t offset = schema.GetOffset(col_idx);

    switch (col.type) {
        case TypeId::INTEGER: {
            int32_t val;
            std::memcpy(&val, data_.data() + offset, 4);
            return Value(val);
        }
        case TypeId::BIGINT: {
            int64_t val;
            std::memcpy(&val, data_.data() + offset, 8);
            return Value(val);
        }
        case TypeId::DECIMAL: {
            double val;
            std::memcpy(&val, data_.data() + offset, 8);
            return Value(val);
        }
        case TypeId::DATE: {
            int32_t val;
            std::memcpy(&val, data_.data() + offset, 4);
            return Value::MakeDate(val);
        }
        case TypeId::VARCHAR: {
            uint32_t var_offset;
            std::memcpy(&var_offset, data_.data() + offset, 4);
            uint16_t len;
            std::memcpy(&len, data_.data() + var_offset, 2);
            std::string str(data_.data() + var_offset + 2, len);
            return Value(str);
        }
        default:
            throw DatabaseException("Unsupported type in Tuple::GetValue");
    }
}

void Tuple::SerializeTo(char* storage) const {
    std::memcpy(storage, data_.data(), data_.size());
}

void Tuple::DeserializeFrom(const char* storage, uint32_t length, const Schema& /*schema*/) {
    data_.resize(length);
    std::memcpy(data_.data(), storage, length);
}

uint32_t Tuple::GetLength() const {
    return static_cast<uint32_t>(data_.size());
}

const char* Tuple::GetData() const {
    return data_.data();
}

bool Tuple::IsEmpty() const {
    return data_.empty();
}

}  // namespace shilmandb
