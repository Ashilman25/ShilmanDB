#pragma once
#include "storage/columnar/columnar_meta.hpp"
#include "storage/columnar/superblock.hpp"
#include <cstddef>
#include <vector>

namespace shilmandb {

// The same catalog always gives the same bytes: map keys go in sorted order.
// Writes kFormatVersion unless given an older one; version 1 can't hold big values.
[[nodiscard]] std::vector<char> EncodeCatalog(const CatalogMeta& meta, uint32_t version = kFormatVersion);

// Reads versions kOldestFormatVersion to kFormatVersion.
// Throws DatabaseException("damaged catalog: ...") on any bad byte or shape.
[[nodiscard]] CatalogMeta DecodeCatalog(const char* data, size_t size);

}  // namespace shilmandb
