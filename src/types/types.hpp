#pragma once
#include <cstdint>

namespace shilmandb {

enum class TypeId : uint8_t {
    INVALID = 0,    //null type
    INTEGER = 1,    //4 byte int32_t
    BIGINT = 2,     //8 byte int64_t
    DECIMAL = 3,    //8 byte double
    VARCHAR = 4,    //var length: uint16_t len prefix + chars
    DATE = 5        //4 bytes, int32_t, days since Jan 1, 1970
};

}