#pragma once

#include "buffer/buffer_pool_manager.hpp"
#include "buffer/eviction_policy_factory.hpp"
#include "storage/disk_manager.hpp"
#include "storage/columnar/columnar_file.hpp"
#include "catalog/catalog.hpp"
#include "types/tuple.hpp"
#include "catalog/schema.hpp"
#include "executor/executor.hpp"
#include "planner/planner.hpp"

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <memory>

#ifdef SHILMANDB_HAS_LIBTORCH
#include "planner/learned_join_optimizer.hpp"
#endif

namespace shilmandb {

namespace parallel {
class WorkerPool;
}

struct LoadOptions {
    StorageKind storage{StorageKind::ROW};
    char delimiter{'|'};
    unsigned threads{0};                    // columnar only; 0 = one per core
    std::optional<std::string> null_text;   // columnar only; a field equal to this is NULL
};

struct QueryOptions {
    ExecutionMode mode{ExecutionMode::TUPLE};
    std::optional<std::vector<int>> forced_join_order;  // plan the joins in exactly this order
    bool analyze{false};                                // count rows and time every operator (tuple mode only)
    bool use_learned_join{true};                        // use the learned join model if one is loaded
    unsigned threads{0};                                // PARALLEL only: 0 = one per core, capped at 256
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

// What the last parallel query used, for benchmark reports.
struct ParallelQueryStats {
    unsigned threads{1};                    // the resolved thread count
    uint64_t morsels{0};                    // morsels started across all its pipelines
    std::chrono::nanoseconds latch_wait{0}; // time workers waited for the buffer pool's latch
};

class Database {
public:
    // LRU eviction, no learned join
    explicit Database(const std::string& db_file, size_t buffer_pool_size = DEFAULT_BUFFER_POOL_SIZE);

    // The eviction policy to use (a learned one needs eviction_model_path). A non-empty
    // join_model_path loads the learned join model, which needs an ML build; without one
    // it throws DatabaseException.
    Database(const std::string& db_file, size_t buffer_pool_size, EvictionPolicyKind eviction, const std::string& eviction_model_path = "", const std::string& join_model_path = "");

    ~Database();

#ifdef SHILMANDB_HAS_LIBTORCH
    // Learned eviction is V2 when eviction_version is "v2", else V1; LRU when off.
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

    // Builds the planner ExecuteSQL and ExplainJoinOrder use for these options.
    [[nodiscard]] Planner MakePlanner(const QueryOptions& options) const;

    void LoadTable(const std::string& table_name, const Schema& schema, const std::string& tbl_file_path, char delimiter = '|');

    // Loads into the row store or, committed to the file, as a columnar table.
    void LoadTable(const std::string& table_name, const Schema& schema, const std::string& tbl_file_path, const LoadOptions& options);

    [[nodiscard]] Catalog* GetCatalog() const;
    [[nodiscard]] BufferPoolManager* GetBufferPoolManager() const;
    [[nodiscard]] DiskManager* GetDiskManager() const;

    // What the planner assumes a scan costs. Read from the disk manager on every call, so it
    // follows the read penalty and direct I/O, unless an override is set.
    [[nodiscard]] ScanCosts GetScanCosts() const;
    void SetScanCosts(std::optional<ScanCosts> costs) { scan_costs_ = costs; }

    // Set when a parallel query finishes; an error keeps the last value.
    [[nodiscard]] const ParallelQueryStats& LastParallelStats() const { return last_parallel_stats_; }

private:
    std::optional<ScanCosts> scan_costs_;
    std::unique_ptr<DiskManager> disk_manager_;
    std::unique_ptr<BufferPoolManager> bpm_;
    std::unique_ptr<ColumnarFile> columnar_file_;
    std::unique_ptr<Catalog> catalog_;

#ifdef SHILMANDB_HAS_LIBTORCH
    std::unique_ptr<LearnedJoinOptimizer> learned_join_optimizer_;
#endif
    ParallelQueryStats last_parallel_stats_;

    // Last, so its threads are joined before anything they could touch goes away.
    // Built on the first query wanting 2+ threads; rebuilt bigger when a query wants more.
    std::unique_ptr<parallel::WorkerPool> worker_pool_;
};

}  // namespace shilmandb
