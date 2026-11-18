#include "executor/parallel/compiler.hpp"
#include "catalog/catalog.hpp"
#include "common/exception.hpp"
#include "executor/parallel/basic_operators.hpp"
#include "executor/parallel/expression.hpp"
#include "executor/parallel/scan_source.hpp"
#include <algorithm>
#include <cassert>
#include <numeric>
#include <string>
#include <utility>

namespace shilmandb::parallel {

namespace {

// The enumerator's name, for error messages.
const char* PlanNodeTypeName(PlanNodeType type) {
    switch (type) {
        case PlanNodeType::SEQ_SCAN: return "SEQ_SCAN";
        case PlanNodeType::INDEX_SCAN: return "INDEX_SCAN";
        case PlanNodeType::FILTER: return "FILTER";
        case PlanNodeType::HASH_JOIN: return "HASH_JOIN";
        case PlanNodeType::NESTED_LOOP_JOIN: return "NESTED_LOOP_JOIN";
        case PlanNodeType::SORT: return "SORT";
        case PlanNodeType::AGGREGATE: return "AGGREGATE";
        case PlanNodeType::PROJECTION: return "PROJECTION";
        case PlanNodeType::LIMIT: return "LIMIT";
    }
    return "UNKNOWN";
}

// Every scan's table must be stored by column; checked before anything compiles.
void CheckTables(const PlanNode& node, Catalog& catalog) {
    const std::string* name = nullptr;
    if (node.type == PlanNodeType::SEQ_SCAN) name = &static_cast<const SeqScanPlanNode&>(node).table_name;
    if (node.type == PlanNodeType::INDEX_SCAN) name = &static_cast<const IndexScanPlanNode&>(node).table_name;
    if (name != nullptr) {
        const TableInfo* info = catalog.GetTable(*name);
        if (info == nullptr) throw DatabaseException("Table not found: " + *name);
        if (info->storage != StorageKind::COLUMNAR) throw DatabaseException("the parallel executor reads only columnar tables: " + *name);
    }
    for (const auto& child : node.children) CheckTables(*child, catalog);
}

// Builds one pipeline from a chain of streaming nodes. Each node asks its child
// only for the columns read above it, so batches never carry unread columns.
class PlanCompiler {
public:
    PlanCompiler(PhysicalPlan& plan, QueryContext& ctx) : plan_(plan), ctx_(ctx) {}

    // The root hands every output column to the result sink.
    void CompileRoot(const PlanNode& root) {
        std::vector<uint32_t> all(root.output_schema.GetColumnCount());
        std::iota(all.begin(), all.end(), 0u);
        const InputLayout layout = Build(root, all);

        auto sink = std::make_unique<ResultSink>(root.output_schema, layout);
        for (const LimitOperator* limit : limits_) sink->WatchLimit(&limit->Counter());
        plan_.result = sink.get();
        pipeline_.sink = sink.get();
        plan_.sinks.push_back(std::move(sink));
        plan_.pipelines.push_back(std::move(pipeline_));
    }

private:
    // Compiles node's subtree; returns where node's output columns sit in its batches.
    // needed lists node's output columns read above it, sorted.
    InputLayout Build(const PlanNode& node, const std::vector<uint32_t>& needed) {
        switch (node.type) {
            case PlanNodeType::SEQ_SCAN:
                return Scan(static_cast<const SeqScanPlanNode&>(node), needed);
            case PlanNodeType::FILTER:
                return Filter(static_cast<const FilterPlanNode&>(node), needed);
            case PlanNodeType::PROJECTION:
                return Projection(static_cast<const ProjectionPlanNode&>(node), needed);
            case PlanNodeType::LIMIT:
                return Limit(static_cast<const LimitPlanNode&>(node), needed);
            default:
                break;
        }
        throw DatabaseException(std::string("the parallel executor can't run ") + PlanNodeTypeName(node.type) + " yet");
    }

    // Slot k holds needed[k]; NUMERIC slots carry their precision so sums can stay narrow.
    InputLayout Scan(const SeqScanPlanNode& scan, const std::vector<uint32_t>& needed) {
        const ColumnarTable* table = ctx_.catalog->GetTable(scan.table_name)->columnar.get();
        const auto& columns = table->Meta().columns;
        if (scan.output_schema.GetColumnCount() != columns.size()) {
            throw DatabaseException("the parallel executor: the plan's columns don't match table " + scan.table_name);
        }
#ifndef NDEBUG
        if (scan.read_columns) {
            for (uint32_t c : needed) {
                assert(std::find(scan.read_columns->begin(), scan.read_columns->end(), c) != scan.read_columns->end() && "a scan hands up a column its plan doesn't list");
            }
        }
#endif

        std::vector<const Expression*> skip;
        for (const auto& conj : scan.skip_conjuncts) skip.push_back(conj.get());
        auto source = std::make_unique<ScanSource>(table, needed, std::move(skip));

        InputLayout layout;
        layout.slot_of_column.assign(columns.size(), -1);
        for (size_t k = 0; k < needed.size(); ++k) {
            const ColumnType& type = columns[needed[k]].type;
            layout.slot_of_column[needed[k]] = static_cast<int>(k);
            layout.slot_types.push_back(source->SlotTypes()[k]);
            layout.slot_precision.push_back(type.id == ColumnTypeId::NUMERIC ? type.precision : 0);
        }
        pipeline_.source = source.get();
        plan_.sources.push_back(std::move(source));
        return layout;
    }

    // The batch passes through, so the filter's layout is its child's.
    // Columns only the predicate reads ride upward too, so a sink that keeps
    // its input must copy just the columns it needs, never every slot.
    InputLayout Filter(const FilterPlanNode& filter, const std::vector<uint32_t>& needed) {
        std::vector<uint32_t> child_needed = needed;
        CollectColumns(*filter.predicate, child_needed);
        const InputLayout layout = Build(*filter.children[0], child_needed);
        AddOperator(std::make_unique<FilterOperator>(CompiledExpr::Compile(*filter.predicate, layout)));
        return layout;
    }

    // Computes only the needed outputs; slot k holds needed[k].
    InputLayout Projection(const ProjectionPlanNode& projection, const std::vector<uint32_t>& needed) {
        assert(projection.expressions.size() == projection.output_schema.GetColumnCount());
        std::vector<uint32_t> child_needed;
        for (uint32_t c : needed) CollectColumns(*projection.expressions[c], child_needed);
        const InputLayout in = Build(*projection.children[0], child_needed);

        std::vector<std::unique_ptr<CompiledExpr>> expressions;
        for (uint32_t c : needed) expressions.push_back(CompiledExpr::Compile(*projection.expressions[c], in));
        auto op = std::make_unique<ProjectionOperator>(std::move(expressions));

        InputLayout out;
        out.slot_of_column.assign(projection.output_schema.GetColumnCount(), -1);
        for (size_t k = 0; k < needed.size(); ++k) out.slot_of_column[needed[k]] = static_cast<int>(k);
        out.slot_types = op->OutputTypes();
        AddOperator(std::move(op));
        return out;
    }

    // The pipeline's sink watches the limit's counter, so workers stop once it's used up.
    InputLayout Limit(const LimitPlanNode& limit, const std::vector<uint32_t>& needed) {
        const InputLayout layout = Build(*limit.children[0], needed);
        auto op = std::make_unique<LimitOperator>(static_cast<uint64_t>(std::max<int64_t>(limit.limit, 0)));
        limits_.push_back(op.get());
        AddOperator(std::move(op));
        return layout;
    }

    // Children are built first, so operators land in the order rows pass through them.
    void AddOperator(std::unique_ptr<Operator> op) {
        pipeline_.operators.push_back(op.get());
        plan_.operators.push_back(std::move(op));
    }

    PhysicalPlan& plan_;
    QueryContext& ctx_;
    Pipeline pipeline_{nullptr, {}, nullptr, {}};
    std::vector<const LimitOperator*> limits_;
};

}  // namespace

// Two passes: every table checked, then the tree compiled.
PhysicalPlan CompilePlan(const PlanNode& root, QueryContext& ctx) {
    CheckTables(root, *ctx.catalog);
    PhysicalPlan plan;
    PlanCompiler(plan, ctx).CompileRoot(root);
    return plan;
}

}  // namespace shilmandb::parallel
