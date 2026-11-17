#pragma once
#include "storage/columnar/columnar_meta.hpp"
#include <cstddef>
#include <vector>

namespace shilmandb {

// The same catalog always gives the same bytes: map keys go in sorted order.
[[nodiscard]] std::vector<char> EncodeCatalog(const CatalogMeta& meta);

// Throws DatabaseException("damaged catalog: ...") on any bad byte or shape.
[[nodiscard]] CatalogMeta DecodeCatalog(const char* data, size_t size);

}  // namespace shilmandb
