#pragma once

#include "buffer/buffer_pool_manager.hpp"
#include "storage/disk_manager.hpp"
#include "catalog/catalog.hpp"
#include "types/tuple.hpp"
#include "catalog/schema.hpp"
#include "executor/executor.hpp"
#include "planner/planner.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <memory>

#ifdef SHILMANDB_HAS_LIBTORCH
#include "planner/learned_join_optimizer.hpp"
#endif

namespace shilmandb {

struct QueryOptions {
    ExecutionMode mode{ExecutionMode::TUPLE};
    std::optional<std::vector<int>> forced_join_order;  // plan the joins in exactly this order
    bool analyze{false};                                // count rows and time every operator (tuple mode only)
    bool use_learned_join{true};                        // use the learned join model if one is loaded
};

struct OperatorProfile {
    std::string name;     // "SeqScan lineitem", "Filter", "HashJoin +orders", "Aggregate", ...
    int depth{0};         // 0 = plan root
    uint64_t rows{0};
    double total_ms{0.0}; // this operator and everything under it
    double self_ms{0.0};  // total_ms minus its children's total_ms
};

struct QueryProfile {
    JoinOrderInfo join;                      // always filled
    std::vector<uint64_t> step_rows;         // analyze only: actual rows after each join step, in join order
    std::vector<uint64_t> scan_rows;         // analyze only: actual rows out of each table's scan + filter, in join order
    std::vector<OperatorProfile> operators;  // analyze only: whole tree, root first (pre-order)
};

struct QueryResult {
    Schema schema;
    std::vector<Tuple> tuples;
    QueryProfile profile;
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

    // Same as ExecuteSQL(sql, QueryOptions{mode}).
    [[nodiscard]] QueryResult ExecuteSQL(const std::string& sql, ExecutionMode mode = ExecutionMode::TUPLE);
    [[nodiscard]] QueryResult ExecuteSQL(const std::string& sql, const QueryOptions& options);

    // Parses and plans the query without running it, and returns how its joins were ordered.
    [[nodiscard]] JoinOrderInfo ExplainJoinOrder(const std::string& sql, const QueryOptions& options = {});

    void LoadTable(const std::string& table_name, const Schema& schema, const std::string& tbl_file_path, char delimiter = '|');

    [[nodiscard]] Catalog* GetCatalog() const;
    [[nodiscard]] BufferPoolManager* GetBufferPoolManager() const;

private:
    // A planner with the join model (if wanted and loaded) and any forced order set.
    [[nodiscard]] Planner MakePlanner(const QueryOptions& options) const;

    std::unique_ptr<DiskManager> disk_manager_;
    std::unique_ptr<BufferPoolManager> bpm_;
    std::unique_ptr<Catalog> catalog_;

#ifdef SHILMANDB_HAS_LIBTORCH
    std::unique_ptr<LearnedJoinOptimizer> learned_join_optimizer_;
#endif
};

}  // namespace shilmandb
