#pragma once
#include "executor/executor.hpp"
#include "storage/columnar/column_vector.hpp"
#include "storage/columnar/columnar_table.hpp"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace shilmandb {

struct TableInfo;

// Scans a columnar table a row group at a time, decoding only the read columns.
// Unread columns come out as placeholder values.
class ColumnarScanExecutor : public Executor {
public:
    ColumnarScanExecutor(const PlanNode* plan, ExecutorContext* ctx);

    void Init() override;
    bool Next(Tuple* tuple) override;
    bool NextBatch(DataChunk* chunk) override;
    void Close() override;

    // Row groups this scan skipped by their min and max since Init.
    [[nodiscard]] size_t SkippedRowGroups() const { return skipped_groups_; }

private:
    // Decodes the read columns of the next row group; false when none is left.
    bool LoadGroup();

    const ColumnarTable* table_{nullptr};
    std::vector<uint32_t> read_columns_;
    std::vector<std::string> read_names_;
    std::vector<ColumnVector> vectors_;  // the current group, one per read column
    std::vector<const Expression*> skip_conjuncts_;  // owned by the plan
    size_t skipped_groups_{0};
    size_t next_group_{0};
    size_t group_rows_{0};
    size_t group_cursor_{0};
    bool initialized_{false};

    // Used only when a tuple-at-a-time parent drains via Next().
    DataChunk buffer_;
    size_t buffer_cursor_{0};
};

// The columns a columnar read covers: the plan's list, or every column.
// Throws DatabaseException for a column past the end.
[[nodiscard]] std::vector<uint32_t> ColumnarReadColumns(const TableInfo& table, const std::optional<std::vector<uint32_t>>& listed);

// The error for reading a column that holds NULLs.
[[nodiscard]] std::string ColumnarNullError(const std::string& column);

}  // namespace shilmandb
