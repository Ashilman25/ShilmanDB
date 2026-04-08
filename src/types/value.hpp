#pragma once
#include "types/types.hpp"
#include <string>
#include <cstdint>
#include <functional>

namespace learneddb {

class Value {

public:
    TypeId type_{TypeId::INVALID};

    union {
        int32_t integer_;
        int64_t bigint_;
        double decimal_;
        int32_t date_;      
    };
    std::string varchar_;   

    Value();
    explicit Value(int32_t val);
    explicit Value(int64_t val);
    explicit Value(double val);
    explicit Value(const std::string& val);
    static Value MakeDate(int32_t days_since_epoch);

    bool operator==(const Value& other) const;
    bool operator!=(const Value& other) const;
    bool operator<(const Value& other) const;
    bool operator>(const Value& other) const;
    bool operator<=(const Value& other) const;
    bool operator>=(const Value& other) const;

    Value Add(const Value& other) const;
    Value Subtract(const Value& other) const;
    Value Multiply(const Value& other) const;
    Value Divide(const Value& other) const;

    std::string ToString() const;
    static Value FromString(TypeId type, const std::string& str);

    uint32_t GetFixedLength() const;
    size_t Hash() const;
    bool IsNull() const;
};

}  // namespace learneddb

namespace std {

template<>
struct hash<learneddb::Value> {
    size_t operator()(const learneddb::Value& v) const { return v.Hash(); }
};

}  // namespace std
