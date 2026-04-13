#include "engine/database.hpp"

#include "buffer/lru_eviction_policy.hpp"
#include "parser/parser.hpp"
#include "planner/planner.hpp"
#include "executor/executor_factory.hpp"
#include "executor/executor_context.hpp"
#include "common/exception.hpp"

#ifdef SHILMANDB_HAS_LIBTORCH
#include "buffer/learned_eviction_policy.hpp"
#endif

#include <fstream>
#include <sstream>

namespace shilmandb {

Database::Database(const std::string& db_file, size_t buffer_pool_size) {
    disk_manager_ = std::make_unique<DiskManager>(db_file);
    auto eviction = std::make_unique<LRUEvictionPolicy>(buffer_pool_size);
    bpm_ = std::make_unique<BufferPoolManager>(buffer_pool_size, disk_manager_.get(), std::move(eviction));
    catalog_ = std::make_unique<Catalog>(bpm_.get());
}

Database::~Database() = default;

#ifdef SHILMANDB_HAS_LIBTORCH
Database::Database(const std::string& db_file, size_t buffer_pool_size, bool use_learned_join, const std::string& join_model_path, bool use_learned_eviction, const std::string& eviction_model_path) {
    disk_manager_ = std::make_unique<DiskManager>(db_file);

    std::unique_ptr<EvictionPolicy> eviction;
    if (use_learned_eviction) {
        eviction = std::make_unique<LearnedEvictionPolicy>(eviction_model_path);
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

QueryResult Database::ExecuteSQL(const std::string& sql) {
    Parser parser(sql);
    auto stmt = parser.Parse();

#ifdef SHILMANDB_HAS_LIBTORCH
    Planner planner(catalog_.get(), learned_join_optimizer_.get());
#else
    Planner planner(catalog_.get());
#endif
    auto plan = planner.Plan(std::move(*stmt));

    ExecutorContext ctx{bpm_.get(), catalog_.get()};
    auto executor = ExecutorFactory::CreateExecutor(plan.get(), &ctx);

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

    return QueryResult{std::move(plan->output_schema), std::move(tuples)};
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
