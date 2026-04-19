#pragma once
#include "types/types.hpp"
#include <string>
#include <cstdint>
#include <functional>

namespace shilmandb {

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

    [[nodiscard]] Value CastTo(TypeId target) const;
};

[[nodiscard]] constexpr TypeId CommonType(TypeId a, TypeId b) {
    if (a == b) return a;

    if ((a == TypeId::DATE && b == TypeId::VARCHAR) || (a == TypeId::VARCHAR && b == TypeId::DATE)) {
        return TypeId::DATE;
    }
        

    auto rank = [](TypeId t) -> int {
        switch (t) {
            case TypeId::INTEGER: 
                return 1;
            case TypeId::BIGINT:  
                return 2;
            case TypeId::DECIMAL: 
                return 3;
            default:              
                return 0;
        }
    };

    const int ra = rank(a);
    const int rb = rank(b);
    if (ra > 0 && rb > 0) return (ra >= rb) ? a : b;

    return TypeId::INVALID;
}

}  // namespace shilmandb

namespace std {

template<>
struct hash<shilmandb::Value> {
    size_t operator()(const shilmandb::Value& v) const { return v.Hash(); }
};

}  // namespace std
