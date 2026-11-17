#pragma once
#include "storage/columnar/column_vector.hpp"
#include "types/value.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace shilmandb {

// Parses one text field and appends it; returns the reason on failure (nothing appended).
// NULL text is the caller's to spot: it calls AppendNull itself.
[[nodiscard]] std::optional<std::string> ParseField(std::string_view text, ColumnVector& col);

// Today's executor value for cell i; DatabaseException if it is NULL.
[[nodiscard]] Value ToExecutorValue(const ColumnVector& col, size_t i);

// "YYYY-MM-DD HH:MM:SS" plus ".ffffff" when the microseconds aren't zero.
[[nodiscard]] std::string FormatTimestamp(int64_t micros_since_2000);

}  // namespace shilmandb
