#pragma once
#include "executor/executor.hpp"
#include "index/btree_index.hpp"
#include "storage/columnar/column_vector.hpp"
#include "storage/columnar/columnar_table.hpp"
#include "storage/table_heap.hpp"
#include <optional>
#include <string>
#include <vector>

namespace shilmandb {

class IndexScanExecutor : public Executor {
public:
    IndexScanExecutor(const PlanNode* plan, ExecutorContext* ctx) : Executor(plan, ctx) {}

    void Init() override;
    bool Next(Tuple* tuple) override;
    void Close() override;

private:
    // Builds the row at rid from a columnar table's read columns.
    [[nodiscard]] Tuple FetchColumnar(const RID& rid);

    std::optional<BTreeIndex::Iterator> iter_;
    std::optional<BTreeIndex::Iterator> end_;

    TableHeap* table_{nullptr};
    const Schema* schema_{nullptr};
    std::optional<Value> high_key_;

    // Set only for a columnar table.
    const ColumnarTable* columnar_{nullptr};
    std::vector<uint32_t> read_columns_;
    std::vector<std::string> read_names_;
    std::vector<ColumnVector> scratch_;  // one value per read column
    std::vector<Value> defaults_;        // a row of placeholders, copied per fetch
};

}  // namespace shilmandb
