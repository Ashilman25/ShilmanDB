#pragma once
#include "types/types.hpp"
#include <cstdint>
#include <string>

namespace shilmandb {

// What a column stores on disk. The numbers are part of the file format.
enum class ColumnTypeId : uint8_t {
    BOOLEAN = 1,
    SMALLINT = 2,
    INTEGER = 3,
    BIGINT = 4,
    NUMERIC = 5,    // integer scaled by 10^scale
    DOUBLE = 6,
    DATE = 7,       // days since 1970-01-01
    TIMESTAMP = 8,  // microseconds since 2000-01-01
    VARCHAR = 9,
    TEXT = 10
};

struct ColumnType {
    ColumnTypeId id{ColumnTypeId::INTEGER};
    uint8_t precision{0};    // NUMERIC only
    uint8_t scale{0};        // NUMERIC only
    uint32_t max_length{0};  // VARCHAR only, in characters

    // One factory per type. Unused fields stay 0, so equal types compare equal.
    [[nodiscard]] static ColumnType Boolean();
    [[nodiscard]] static ColumnType SmallInt();
    [[nodiscard]] static ColumnType Integer();
    [[nodiscard]] static ColumnType BigInt();

    // Throws unless 1 <= precision <= 38 and 0 <= scale <= precision.
    [[nodiscard]] static ColumnType Numeric(int precision, int scale);
    [[nodiscard]] static ColumnType Double();
    [[nodiscard]] static ColumnType Date();
    [[nodiscard]] static ColumnType Timestamp();
    
    // Throws for a max length of 0.
    [[nodiscard]] static ColumnType Varchar(uint32_t max_length);
    [[nodiscard]] static ColumnType Text();

    // Equal when all four fields match.
    bool operator==(const ColumnType& o) const;
    bool operator!=(const ColumnType& o) const;
};

// How a value is held in memory while encoding.
enum class PhysicalKind : uint8_t { INT64, INT128, DOUBLE, STRING };

// The storage type a plain TypeId column gets. INVALID throws.
[[nodiscard]] ColumnType DefaultColumnType(TypeId type);

// The TypeId today's planner and executor see for a storage type.
[[nodiscard]] TypeId ExecutorType(const ColumnType& type);

// NUMERIC above 18 digits needs 128 bits; every other number fits in 64.
[[nodiscard]] PhysicalKind PhysicalKindOf(const ColumnType& type);

// Bytes per value in PLAIN encoding; 0 for strings.
[[nodiscard]] uint32_t PlainWidth(const ColumnType& type);

// SQL-style name, like "NUMERIC(15,2)" or "VARCHAR(25)".
[[nodiscard]] std::string ColumnTypeName(const ColumnType& type);

}  // namespace shilmandb
