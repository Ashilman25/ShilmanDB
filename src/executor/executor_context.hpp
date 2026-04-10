#pragma once
#include "buffer/buffer_pool_manager.hpp"
#include "catalog/catalog.hpp"

namespace shilmandb {

struct ExecutorContext {
    BufferPoolManager* bpm;
    Catalog* catalog;
};

} //shilmandb