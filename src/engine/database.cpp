#include "engine/database.hpp"

#include "buffer/lru_eviction_policy.hpp"
#include "parser/parser.hpp"
#include "planner/planner.hpp"
#include "executor/executor_factory.hpp"
#include "executor/executor_context.hpp"
#include "common/exception.hpp"

#ifdef SHILMANDB_HAS_LIBTORCH
#include "buffer/learned_eviction_policy.hpp"
#include "buffer/learned_eviction_policy_v2.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>

namespace shilmandb {

namespace {

[[nodiscard]] double ToMs(std::chrono::nanoseconds ns) {
    return std::chrono::duration<double, std::milli>(ns).count();
}

[[nodiscard]] OperatorStats StatsFor(const OperatorStatsMap& op_stats, const PlanNode* node) {
    auto it = op_stats.find(node);
    return it != op_stats.end() ? it->second : OperatorStats{};
}

[[nodiscard]] std::string ScanTable(const PlanNode* node) {
    switch (node->type) {
        case PlanNodeType::SEQ_SCAN: return static_cast<const SeqScanPlanNode*>(node)->table_name;
        case PlanNodeType::INDEX_SCAN: return static_cast<const IndexScanPlanNode*>(node)->table_name;
        default: return "";
    }
}

// The table a join adds: the scan at the bottom of its right side.
[[nodiscard]] std::string JoinedTable(const PlanNode* join) {
    if (join->children.size() < 2) return "";
    const PlanNode* node = join->children[1].get();
    while (!node->children.empty()) node = node->children[0].get();
    return ScanTable(node);
}

[[nodiscard]] std::string OperatorName(const PlanNode* node) {
    switch (node->type) {
        case PlanNodeType::SEQ_SCAN: return "SeqScan " + ScanTable(node);
        case PlanNodeType::INDEX_SCAN: return "IndexScan " + ScanTable(node);
        case PlanNodeType::FILTER: return "Filter";
        case PlanNodeType::HASH_JOIN: return "HashJoin +" + JoinedTable(node);
        case PlanNodeType::NESTED_LOOP_JOIN: return "NestedLoopJoin +" + JoinedTable(node);
        case PlanNodeType::SORT: return "Sort";
        case PlanNodeType::AGGREGATE: return "Aggregate";
        case PlanNodeType::PROJECTION: return "Projection";
        case PlanNodeType::LIMIT: return "Limit";
    }
    return "Unknown";
}

// Root first, then each child's whole subtree in order.
void CollectOperators(const PlanNode* node, int depth, const OperatorStatsMap& op_stats, std::vector<OperatorProfile>& out) {
    const auto stats = StatsFor(op_stats, node);
    const double total_ms = ToMs(stats.time);

    double children_ms = 0.0;
    for (const auto& child : node->children) {
        children_ms += ToMs(StatsFor(op_stats, child.get()).time);
    }

    // Clock reads inside the children can push their sum a hair past ours.
    out.push_back({OperatorName(node), depth, stats.rows, total_ms, std::max(0.0, total_ms - children_ms)});

    for (const auto& child : node->children) {
        CollectOperators(child.get(), depth + 1, op_stats, out);
    }
}

void FillAnalyzeProfile(const Planner& planner, const PlanNode* root, const OperatorStatsMap& op_stats, QueryProfile& profile) {
    for (const auto* step : planner.GetStepNodes()) {
        profile.step_rows.push_back(StatsFor(op_stats, step).rows);
    }

    const auto& scans = planner.GetScanNodes();
    for (int slot : profile.join.order) {
        profile.scan_rows.push_back(StatsFor(op_stats, scans[slot]).rows);
    }

    CollectOperators(root, 0, op_stats, profile.operators);
}

}  // namespace

Database::Database(const std::string& db_file, size_t buffer_pool_size) {
    disk_manager_ = std::make_unique<DiskManager>(db_file);
    auto eviction = std::make_unique<LRUEvictionPolicy>(buffer_pool_size);
    bpm_ = std::make_unique<BufferPoolManager>(buffer_pool_size, disk_manager_.get(), std::move(eviction));
    catalog_ = std::make_unique<Catalog>(bpm_.get());
}

Database::~Database() = default;

#ifdef SHILMANDB_HAS_LIBTORCH
Database::Database(const std::string& db_file, size_t buffer_pool_size, bool use_learned_join, const std::string& join_model_path, bool use_learned_eviction, const std::string& eviction_model_path, const std::string& eviction_version) {
    disk_manager_ = std::make_unique<DiskManager>(db_file);

    std::unique_ptr<EvictionPolicy> eviction;
    if (use_learned_eviction) {
        if (eviction_version == "v2") {
            eviction = std::make_unique<LearnedEvictionPolicyV2>(eviction_model_path);
        } else {
            eviction = std::make_unique<LearnedEvictionPolicy>(eviction_model_path);
        }
    } else {
        eviction = std::make_unique<LRUEvictionPolicy>(buffer_pool_size);
    }
    
    bpm_ = std::make_unique<BufferPoolManager>(buffer_pool_size, disk_manager_.get(), std::move(eviction));
    catalog_ = std::make_unique<Catalog>(bpm_.get());

    if (use_learned_join) {
        learned_join_optimizer_ = std::make_unique<LearnedJoinOptimizer>(join_model_path);
    }
}
#endif

QueryResult Database::ExecuteSQL(const std::string& sql, ExecutionMode mode) {
    return ExecuteSQL(sql, QueryOptions{mode});
}

QueryResult Database::ExecuteSQL(const std::string& sql, const QueryOptions& options) {
    Parser parser(sql);
    auto stmt = parser.Parse();

    auto planner = MakePlanner(options);
    auto plan = planner.Plan(std::move(*stmt));

    // Declared before the executor so the profilers' pointers into it outlive them.
    OperatorStatsMap op_stats;
    ExecutorContext ctx{bpm_.get(), catalog_.get(), options.analyze ? &op_stats : nullptr};
    auto executor = ExecutorFactory::CreateExecutor(plan.get(), &ctx, options.mode);

    executor->Init();
    std::vector<Tuple> tuples;
    Tuple tuple;
    try {
        while (executor->Next(&tuple)) {
            tuples.push_back(std::move(tuple));
        }
    } catch (...) {
        executor->Close();
        throw;
    }
    executor->Close();

    // Stats are final after Close, and the planner's node pointers are good while the plan lives.
    QueryProfile profile;
    profile.join = planner.GetJoinOrderInfo();
    if (options.analyze) {
        FillAnalyzeProfile(planner, plan.get(), op_stats, profile);
    }

    return QueryResult{std::move(plan->output_schema), std::move(tuples), std::move(profile)};
}

JoinOrderInfo Database::ExplainJoinOrder(const std::string& sql, const QueryOptions& options) {
    Parser parser(sql);
    auto stmt = parser.Parse();

    auto planner = MakePlanner(options);
    (void)planner.Plan(std::move(*stmt));
    return planner.GetJoinOrderInfo();
}

Planner Database::MakePlanner(const QueryOptions& options) const {
#ifdef SHILMANDB_HAS_LIBTORCH
    Planner planner(catalog_.get(), options.use_learned_join ? learned_join_optimizer_.get() : nullptr);
#else
    Planner planner(catalog_.get());
#endif
    if (options.forced_join_order) {
        planner.SetForcedJoinOrder(*options.forced_join_order);
    }
    return planner;
}

void Database::LoadTable(const std::string& table_name, const Schema& schema, const std::string& tbl_file_path, char delimiter) {
    auto* table_info = catalog_->CreateTable(table_name, schema);
    if (!table_info) {
        throw DatabaseException("Failed to create table: " + table_name);
    }

    std::ifstream file(tbl_file_path);
    if (!file.is_open()) {
        throw DatabaseException("Cannot open file: " + tbl_file_path);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::vector<Value> values;
        values.reserve(schema.GetColumnCount());
        std::istringstream ss(line);
        std::string field;
        uint32_t col_idx = 0;

        while (std::getline(ss, field, delimiter)) {
            if (col_idx >= schema.GetColumnCount()) break;
            values.push_back(Value::FromString(schema.GetColumn(col_idx).type, field));
            ++col_idx;
        }

        if (col_idx == schema.GetColumnCount()) {
            (void)table_info->table->InsertTuple(Tuple(std::move(values), schema));
        }
    }

    catalog_->UpdateTableStats(table_name);
}

Catalog* Database::GetCatalog() const { return catalog_.get(); }

BufferPoolManager* Database::GetBufferPoolManager() const { return bpm_.get(); }

}  // namespace shilmandb
