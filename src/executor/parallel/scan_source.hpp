#pragma once
#include "executor/parallel/expression.hpp"
#include "executor/parallel/pipeline.hpp"
#include "parser/ast.hpp"
#include "storage/columnar/columnar_table.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace shilmandb::parallel {

// Above this share of a row group's rows, a column is decoded whole.
inline constexpr double kDenseFraction = 0.125;

// One columnar table scan, with the filter right above it fused in.
// Conjunct i of single reads slot 0, holding single_column[i];
// conjunct i of multi reads slot k, holding multi_columns[i][k] (see ScanLayout).
struct ScanSpec {
    std::string table;
    std::vector<uint32_t> columns;                       // table columns to hand up; slot k holds columns[k]
    std::vector<std::unique_ptr<CompiledExpr>> single;   // single-column conjuncts, in written order
    std::vector<uint32_t> single_column;                 // the column each one reads
    std::vector<std::unique_ptr<CompiledExpr>> multi;    // conjuncts reading several columns, or none
    std::vector<std::vector<uint32_t>> multi_columns;    // the columns each one reads, sorted
    std::vector<const Expression*> skip_conjuncts;       // from the plan, which must outlive the scan
    std::vector<uint32_t> refuse_nulls;                  // columns whose NULLs are refused even if never read
    double dense_fraction{kDenseFraction};
};

// What a scan read and kept, summed over every morsel so far.
struct ScanCounters {
    uint64_t groups_skipped{0};
    uint64_t rows_considered{0};
    uint64_t rows_passed{0};
    std::vector<uint64_t> rows_decoded;  // per table column; a whole chunk counts all its rows
};

// Slot k holds table column columns[k]; NUMERIC slots carry their precision.
[[nodiscard]] InputLayout ScanLayout(const ColumnarTable& table, const std::vector<uint32_t>& columns);

// Splits a filter's AND tree, left first, and compiles each part into the spec.
void AddConjuncts(ScanSpec& spec, const ColumnarTable& table, const Expression& predicate);

// Reads a columnar table one row group per morsel. Each conjunct decodes its column
// for the rows still selected and narrows them; handed-up columns come last, for the
// survivors only. A morsel's batches stay valid until its local starts the next one.
class ScanSource : public Source {
public:
    explicit ScanSource(ScanSpec spec);

    // Finds the table; keeps the row groups whose min and max allow a match.
    void Prepare(QueryContext& ctx) override;

    [[nodiscard]] size_t MorselCount() const override { return groups_.size(); }

    // Needs Prepare first, which finds the table.
    [[nodiscard]] std::unique_ptr<LocalState> MakeLocal() const override;

    // Refuses NULLs from chunk metadata, then filters the row group and decodes what's handed up.
    void StartMorsel(size_t morsel, LocalState& local) const override;

    // Hands up the next kBatchSize survivors, every row active.
    bool NextBatch(LocalState& local, Batch& out) const override;

    // A snapshot of the counts since Prepare.
    [[nodiscard]] ScanCounters Counters() const;

private:
    struct Local;

    // Decodes one column for the survivors, whole while the selection is dense.
    void Decode(Local& l, uint32_t column) const;

    // Runs conjunct number `step` (from 1) over the survivors; keeps the rows it passes.
    void Apply(Local& l, const CompiledExpr& expr, const uint32_t* columns, size_t width, Batch& batch, size_t step) const;

    // Drops the failed survivors from the survivor list and every column read later.
    void Narrow(Local& l, size_t step) const;

    ScanSpec spec_;
    const ColumnarTable* table_{nullptr};
    std::vector<VType> types_;       // per table column
    std::vector<VType> slot_types_;  // per handed-up slot
    std::vector<uint32_t> read_;     // every column the scan reads, sorted
    std::vector<uint32_t> refused_;  // refuse_nulls in its order, then any other column read
    std::vector<bool> handed_;       // per table column: handed up
    std::vector<size_t> last_use_;   // per table column: the last conjunct step reading it, 0 for none
    std::vector<size_t> groups_;     // the row groups to read, one per morsel

    // Counts every worker adds to from const StartMorsel, so atomic and mutable.
    // Prepare allocates rows_decoded_; a const call can't reallocate it.
    mutable std::atomic<uint64_t> groups_skipped_{0};
    mutable std::atomic<uint64_t> rows_considered_{0};
    mutable std::atomic<uint64_t> rows_passed_{0};
    std::unique_ptr<std::atomic<uint64_t>[]> rows_decoded_;  // per table column
    size_t decoded_columns_{0};                             // rows_decoded_'s size
};

// The source for one scan.
[[nodiscard]] std::unique_ptr<Source> MakeScanSource(ScanSpec spec);

}  // namespace shilmandb::parallel
