#pragma once
#include "executor/parallel/pipeline.hpp"
#include "parser/ast.hpp"
#include "storage/columnar/columnar_table.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace shilmandb::parallel {

// Reads a columnar table one row group per morsel, handing up only the listed columns.
// Slot k of every batch holds columns[k].
class ScanSource : public Source {
public:
    // skip_conjuncts belong to the plan and must outlive the scan.
    ScanSource(const ColumnarTable* table, std::vector<uint32_t> columns, std::vector<const Expression*> skip_conjuncts);

    // The type of each slot, in the order of columns.
    [[nodiscard]] const std::vector<VType>& SlotTypes() const { return types_; }

    // Keeps the row groups whose min and max say a row may match.
    void Prepare(QueryContext& ctx) override;

    [[nodiscard]] size_t MorselCount() const override { return groups_.size(); }
    [[nodiscard]] std::unique_ptr<LocalState> MakeLocal() const override;

    // Refuses NULLs, then decodes each column of the morsel's row group once.
    void StartMorsel(size_t morsel, LocalState& local) override;

    // Copies the next kBatchSize rows of the decoded group into out.
    bool NextBatch(LocalState& local, Batch& out) override;

private:
    const ColumnarTable* table_;
    std::vector<uint32_t> columns_;
    std::vector<VType> types_;
    std::vector<const Expression*> skip_conjuncts_;
    std::vector<size_t> groups_;  // the row groups to read, one per morsel
};

}  // namespace shilmandb::parallel
