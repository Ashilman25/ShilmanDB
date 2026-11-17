#pragma once
#include <cstddef>
#include <cstdint>

namespace shilmandb {

// CRC-32C (Castagnoli). Seed 0 gives the standard value; a result seeds the next piece.
[[nodiscard]] uint32_t Crc32c(const void* data, size_t len, uint32_t seed = 0);

// The same checksum from a lookup table, for machines without the instruction.
[[nodiscard]] uint32_t Crc32cPortable(const void* data, size_t len, uint32_t seed = 0);

}  // namespace shilmandb
