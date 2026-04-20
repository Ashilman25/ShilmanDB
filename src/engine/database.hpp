#pragma once

#include "buffer/buffer_pool_manager.hpp"
#include "storage/disk_manager.hpp"
#include "catalog/catalog.hpp"
#include "types/tuple.hpp"
#include "catalog/schema.hpp"
#include "executor/executor.hpp"

#include <string>
#include <vector>
#include <memory>

#ifdef SHILMANDB_HAS_LIBTORCH
#include "planner/learned_join_optimizer.hpp"
#endif

namespace shilmandb {

struct QueryResult {
    Schema schema;
    std::vector<Tuple> tuples;
};

class Database {
public:
    explicit Database(const std::string& db_file, size_t buffer_pool_size = DEFAULT_BUFFER_POOL_SIZE);
    ~Database();

#ifdef SHILMANDB_HAS_LIBTORCH
    Database(const std::string& db_file, size_t buffer_pool_size, bool use_learned_join, const std::string& join_model_path,
             bool use_learned_eviction, const std::string& eviction_model_path,
             const std::string& eviction_version = "v2");
#endif

    Database(const Database&) = delete;
    Database& operator=(const Database&) = delete;
    Database(Database&&) = delete;
    Database& operator=(Database&&) = delete;

    [[nodiscard]] QueryResult ExecuteSQL(const std::string& sql, ExecutionMode mode = ExecutionMode::TUPLE);

    void LoadTable(const std::string& table_name, const Schema& schema, const std::string& tbl_file_path, char delimiter = '|');

    [[nodiscard]] Catalog* GetCatalog() const;
    [[nodiscard]] BufferPoolManager* GetBufferPoolManager() const;

private:
    std::unique_ptr<DiskManager> disk_manager_;
    std::unique_ptr<BufferPoolManager> bpm_;
    std::unique_ptr<Catalog> catalog_;

#ifdef SHILMANDB_HAS_LIBTORCH
    std::unique_ptr<LearnedJoinOptimizer> learned_join_optimizer_;
#endif
};

}  // namespace shilmandb
