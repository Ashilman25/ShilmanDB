#include "types/column_type.hpp"
#include "common/exception.hpp"

namespace shilmandb {

namespace {

constexpr int kMaxNumericPrecision = 38;
constexpr int kMaxInt64Precision = 18;

// A type with just its id set.
ColumnType Make(ColumnTypeId id) {
    ColumnType type;
    type.id = id;
    return type;
}

// True for NUMERIC that needs 128 bits.
bool IsWideNumeric(const ColumnType& type) {
    return type.id == ColumnTypeId::NUMERIC && type.precision > kMaxInt64Precision;
}

// For an id outside the enum, as from a corrupt file.
[[noreturn]] void ThrowUnknown(const ColumnType& type) {
    throw DatabaseException("Unknown column type id " + std::to_string(static_cast<int>(type.id)));
}

}  // namespace

// Factories for the types with no parameters.
ColumnType ColumnType::Boolean()   { return Make(ColumnTypeId::BOOLEAN); }
ColumnType ColumnType::SmallInt()  { return Make(ColumnTypeId::SMALLINT); }
ColumnType ColumnType::Integer()   { return Make(ColumnTypeId::INTEGER); }
ColumnType ColumnType::BigInt()    { return Make(ColumnTypeId::BIGINT); }
ColumnType ColumnType::Double()    { return Make(ColumnTypeId::DOUBLE); }
ColumnType ColumnType::Date()      { return Make(ColumnTypeId::DATE); }
ColumnType ColumnType::Timestamp() { return Make(ColumnTypeId::TIMESTAMP); }
ColumnType ColumnType::Text()      { return Make(ColumnTypeId::TEXT); }

// Checks the precision and scale before building.
ColumnType ColumnType::Numeric(int precision, int scale) {
    if (precision < 1 || precision > kMaxNumericPrecision || scale < 0 || scale > precision) {
        throw DatabaseException("Bad NUMERIC(" + std::to_string(precision) + "," + std::to_string(scale) + "): need 1 <= precision <= 38 and 0 <= scale <= precision");
    }
    ColumnType type = Make(ColumnTypeId::NUMERIC);
    type.precision = static_cast<uint8_t>(precision);
    type.scale = static_cast<uint8_t>(scale);
    return type;
}

// A VARCHAR must allow at least one character.
ColumnType ColumnType::Varchar(uint32_t max_length) {
    if (max_length == 0) {
        throw DatabaseException("VARCHAR needs a max length of at least 1");
    }
    ColumnType type = Make(ColumnTypeId::VARCHAR);
    type.max_length = max_length;
    return type;
}

// Compares every field; factories zero the unused ones.
bool ColumnType::operator==(const ColumnType& o) const {
    return id == o.id && precision == o.precision && scale == o.scale && max_length == o.max_length;
}

// The opposite of operator==.
bool ColumnType::operator!=(const ColumnType& o) const {
    return !(*this == o);
}

// DECIMAL keeps its double meaning, VARCHAR has no length limit.
ColumnType DefaultColumnType(TypeId type) {
    switch (type) {
        case TypeId::INTEGER: return ColumnType::Integer();
        case TypeId::BIGINT:  return ColumnType::BigInt();
        case TypeId::DECIMAL: return ColumnType::Double();
        case TypeId::VARCHAR: return ColumnType::Text();
        case TypeId::DATE:    return ColumnType::Date();
        case TypeId::INVALID: break;
    }
    throw DatabaseException("No storage type for an INVALID column");
}

// Maps each storage type onto the five TypeIds the executor knows.
TypeId ExecutorType(const ColumnType& type) {
    switch (type.id) {
        case ColumnTypeId::BOOLEAN:
        case ColumnTypeId::SMALLINT:
        case ColumnTypeId::INTEGER:
            return TypeId::INTEGER;
        case ColumnTypeId::BIGINT:
            return TypeId::BIGINT;
        case ColumnTypeId::NUMERIC:
        case ColumnTypeId::DOUBLE:
            return TypeId::DECIMAL;
        case ColumnTypeId::DATE:
            return TypeId::DATE;
        case ColumnTypeId::TIMESTAMP:  // shown as its text form for now
        case ColumnTypeId::VARCHAR:
        case ColumnTypeId::TEXT:
            return TypeId::VARCHAR;
    }
    ThrowUnknown(type);
}

// Every integer-like type is held as int64 except wide NUMERIC.
PhysicalKind PhysicalKindOf(const ColumnType& type) {
    switch (type.id) {
        case ColumnTypeId::NUMERIC:
            return IsWideNumeric(type) ? PhysicalKind::INT128 : PhysicalKind::INT64;
        case ColumnTypeId::DOUBLE:
            return PhysicalKind::DOUBLE;
        case ColumnTypeId::VARCHAR:
        case ColumnTypeId::TEXT:
            return PhysicalKind::STRING;
        case ColumnTypeId::BOOLEAN:
        case ColumnTypeId::SMALLINT:
        case ColumnTypeId::INTEGER:
        case ColumnTypeId::BIGINT:
        case ColumnTypeId::DATE:
        case ColumnTypeId::TIMESTAMP:
            return PhysicalKind::INT64;
    }
    ThrowUnknown(type);
}

// Fixed byte width per value; strings have none.
uint32_t PlainWidth(const ColumnType& type) {
    switch (type.id) {
        case ColumnTypeId::BOOLEAN:   return 1;
        case ColumnTypeId::SMALLINT:  return 2;
        case ColumnTypeId::INTEGER:   return 4;
        case ColumnTypeId::DATE:      return 4;
        case ColumnTypeId::BIGINT:    return 8;
        case ColumnTypeId::TIMESTAMP: return 8;
        case ColumnTypeId::DOUBLE:    return 8;
        case ColumnTypeId::NUMERIC:   return IsWideNumeric(type) ? 16 : 8;
        case ColumnTypeId::VARCHAR:
        case ColumnTypeId::TEXT:
            return 0;
    }
    ThrowUnknown(type);
}

// The name as SQL writes it, with no spaces inside the parentheses.
std::string ColumnTypeName(const ColumnType& type) {
    switch (type.id) {
        case ColumnTypeId::BOOLEAN:   return "BOOLEAN";
        case ColumnTypeId::SMALLINT:  return "SMALLINT";
        case ColumnTypeId::INTEGER:   return "INTEGER";
        case ColumnTypeId::BIGINT:    return "BIGINT";
        case ColumnTypeId::NUMERIC:
            return "NUMERIC(" + std::to_string(type.precision) + "," + std::to_string(type.scale) + ")";
        case ColumnTypeId::DOUBLE:    return "DOUBLE";
        case ColumnTypeId::DATE:      return "DATE";
        case ColumnTypeId::TIMESTAMP: return "TIMESTAMP";
        case ColumnTypeId::VARCHAR:   return "VARCHAR(" + std::to_string(type.max_length) + ")";
        case ColumnTypeId::TEXT:      return "TEXT";
    }
    ThrowUnknown(type);
}

}  // namespace shilmandb
