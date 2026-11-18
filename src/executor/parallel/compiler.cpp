#include "executor/parallel/compiler.hpp"
#include "catalog/catalog.hpp"
#include "common/exception.hpp"
#include "executor/parallel/basic_operators.hpp"
#include "executor/parallel/expression.hpp"
#include "executor/parallel/nested_loop_join.hpp"
#include "executor/parallel/scan_source.hpp"
#include "executor/parallel/sort.hpp"
#include <algorithm>
#include <cassert>
#include <numeric>
#include <optional>
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

// The slot holding column c; the child was asked for c, so it's there.
uint32_t SlotOf(const InputLayout& layout, uint32_t c) {
    const int slot = c < layout.slot_of_column.size() ? layout.slot_of_column[c] : -1;
    if (slot < 0 || static_cast<size_t>(slot) >= layout.slot_types.size()) {
        throw DatabaseException("the parallel executor lost column " + std::to_string(c) + " of a child");
    }
    return static_cast<uint32_t>(slot);
}

// Makes slot `slot` of from the next slot of out, holding out's column c.
void AddSlot(InputLayout& out, uint32_t c, const InputLayout& from, uint32_t slot) {
    out.slot_of_column[c] = static_cast<int>(out.slot_types.size());
    out.slot_types.push_back(from.slot_types[slot]);
    out.slot_precision.push_back(slot < from.slot_precision.size() ? from.slot_precision[slot] : 0);
}

// A pipeline being built, the limits inside it, and its operators that may
// answer HAVE_MORE_OUTPUT.
struct Building {
    Pipeline pipeline{nullptr, {}, nullptr, {}};
    std::vector<const LimitOperator*> limits;
    std::vector<const Operator*> expanding;
};

// Builds pipelines from the plan. Streaming nodes join the pipeline being built;
// a breaker ends it in a sink and starts the next one from its source.
// Each node asks its child only for the columns read above it.
class PlanCompiler {
public:
    PlanCompiler(PhysicalPlan& plan, QueryContext& ctx) : plan_(plan), ctx_(ctx) {}

    // The root hands every output column to the result sink, which watches
    // the limits in its own pipeline.
    void CompileRoot(const PlanNode& root) {
        std::vector<uint32_t> all(root.output_schema.GetColumnCount());
        std::iota(all.begin(), all.end(), 0u);
        const InputLayout layout = Build(root, all);

        auto sink = std::make_unique<ResultSink>(root.output_schema, layout);
        AssertLimitsAboveExpanding();
        for (const LimitOperator* limit : current_.limits) sink->WatchLimit(&limit->Counter());
        plan_.result = sink.get();
        Close(Own(std::move(sink)));
    }

private:
    // Done() stops a pipeline mid-batch. A watched limit below an operator still
    // expanding its batch would cut that batch short, so every limit must come after.
    // The planner puts LIMIT at the root, so this holds today.
    void AssertLimitsAboveExpanding() const {
#ifndef NDEBUG
        const auto& ops = current_.pipeline.operators;
        const auto position = [&](const Operator* op) { return std::find(ops.begin(), ops.end(), op) - ops.begin(); };
        for (const LimitOperator* limit : current_.limits) {
            for (const Operator* op : current_.expanding) {
                assert(position(limit) > position(op) && "a watched LIMIT sits below an operator that can answer HAVE_MORE_OUTPUT");
            }
        }
#endif
    }

    // Compiles node's subtree; returns where node's output columns sit in its batches.
    // needed lists node's output columns read above it, sorted.
    InputLayout Build(const PlanNode& node, const std::vector<uint32_t>& needed) {
        switch (node.type) {
            case PlanNodeType::SEQ_SCAN:
                return Scan(static_cast<const SeqScanPlanNode&>(node), nullptr, needed);
            case PlanNodeType::FILTER: {
                const auto& filter = static_cast<const FilterPlanNode&>(node);
                const PlanNode& child = *filter.children[0];
                if (child.type == PlanNodeType::SEQ_SCAN) {
                    return Scan(static_cast<const SeqScanPlanNode&>(child), filter.predicate.get(), needed);
                }
                return Filter(filter, needed);
            }
            case PlanNodeType::PROJECTION:
                return Projection(static_cast<const ProjectionPlanNode&>(node), needed);
            case PlanNodeType::LIMIT: {
                const auto& limit = static_cast<const LimitPlanNode&>(node);
                if (limit.children[0]->type == PlanNodeType::SORT) {
                    return Sort(static_cast<const SortPlanNode&>(*limit.children[0]), needed, limit.limit);
                }
                return Limit(limit, needed);
            }
            case PlanNodeType::SORT:
                return Sort(static_cast<const SortPlanNode&>(node), needed, std::nullopt);
            case PlanNodeType::NESTED_LOOP_JOIN:
                return NestedLoopJoin(static_cast<const NestedLoopJoinPlanNode&>(node), needed);
            default:
                break;
        }
        throw DatabaseException(std::string("the parallel executor can't run ") + PlanNodeTypeName(node.type) + " yet");
    }

    // Slot k holds needed[k]; NUMERIC slots carry their precision so sums can stay narrow.
    // A fused predicate's columns are read inside the scan and never handed up.
    InputLayout Scan(const SeqScanPlanNode& scan, const Expression* predicate, const std::vector<uint32_t>& needed) {
        const ColumnarTable* table = ctx_.catalog->GetTable(scan.table_name)->columnar.get();
        if (scan.output_schema.GetColumnCount() != table->Meta().columns.size()) {
            throw DatabaseException("the parallel executor: the plan's columns don't match table " + scan.table_name);
        }

        ScanSpec spec;
        spec.table = scan.table_name;
        spec.columns = needed;
        for (const auto& conj : scan.skip_conjuncts) spec.skip_conjuncts.push_back(conj.get());
        if (predicate != nullptr) AddConjuncts(spec, *table, *predicate);
#ifndef NDEBUG
        if (scan.read_columns) {
            std::vector<uint32_t> read = needed;
            read.insert(read.end(), spec.single_column.begin(), spec.single_column.end());
            for (const auto& columns : spec.multi_columns) read.insert(read.end(), columns.begin(), columns.end());
            for (uint32_t c : read) {
                const bool listed = std::find(scan.read_columns->begin(), scan.read_columns->end(), c) != scan.read_columns->end();
                assert(listed && "a scan reads a column its plan doesn't list");
            }
        }
#endif

        SetSource(MakeScanSource(std::move(spec)));
        return ScanLayout(*table, needed);
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

    // A result sink in this pipeline watches the counter, stopping workers at the limit.
    InputLayout Limit(const LimitPlanNode& limit, const std::vector<uint32_t>& needed) {
        const InputLayout layout = Build(*limit.children[0], needed);
        auto op = std::make_unique<LimitOperator>(static_cast<uint64_t>(std::max<int64_t>(limit.limit, 0)));
        current_.limits.push_back(op.get());
        AddOperator(std::move(op));
        return layout;
    }

    // A breaker: the child's pipeline ends in the sort's sink, and the next pipeline
    // starts from its source. Slot k holds needed[k]; a LIMIT above becomes its top n.
    InputLayout Sort(const SortPlanNode& sort, const std::vector<uint32_t>& needed, std::optional<int64_t> limit) {
        std::vector<uint32_t> child_needed = needed;
        for (const auto& item : sort.order_by) CollectColumns(*item.expr, child_needed);
        const InputLayout in = Build(*sort.children[0], child_needed);

        SortSpec spec;
        for (const auto& item : sort.order_by) {
            spec.keys.push_back(CompiledExpr::Compile(*item.expr, in));
            spec.ascending.push_back(item.ascending);
        }
        spec.limit = limit;
        InputLayout out;
        out.slot_of_column.assign(sort.output_schema.GetColumnCount(), -1);
        for (uint32_t c : needed) {
            const uint32_t slot = SlotOf(in, c);
            spec.columns.push_back(slot);
            spec.types.push_back(in.slot_types[slot]);
            AddSlot(out, c, in, slot);
        }

        auto parts = MakeSort(std::move(spec));
        const size_t sorted = Close(Own(std::move(parts.first)));
        SetSource(std::move(parts.second));
        current_.pipeline.depends_on.push_back(sorted);
        return out;
    }

    // The right side runs first, into the join's sink; left batches stream through
    // the join. Candidate slots: the needed and predicate columns, left ones first.
    InputLayout NestedLoopJoin(const NestedLoopJoinPlanNode& join, const std::vector<uint32_t>& needed) {
        const PlanNode& left = *join.children[0];
        const PlanNode& right = *join.children[1];
        const uint32_t left_count = left.output_schema.GetColumnCount();
        assert(join.output_schema.GetColumnCount() == left_count + right.output_schema.GetColumnCount());

        std::vector<uint32_t> columns = needed;
        if (join.predicate) CollectColumns(*join.predicate, columns);
        std::vector<uint32_t> left_needed;
        std::vector<uint32_t> right_needed;
        for (uint32_t c : columns) {
            if (c < left_count) {
                left_needed.push_back(c);
            } else {
                right_needed.push_back(c - left_count);
            }
        }

        // The right side gets a pipeline of its own; it closes once the join's
        // predicate, which needs both sides' layouts, is compiled.
        Building outer = std::exchange(current_, Building{});
        const InputLayout right_in = Build(right, right_needed);
        Building right_side = std::exchange(current_, std::move(outer));
        const InputLayout left_in = Build(left, left_needed);

        NestedLoopSpec spec;
        InputLayout out;
        out.slot_of_column.assign(join.output_schema.GetColumnCount(), -1);
        for (uint32_t c : left_needed) {
            const uint32_t slot = SlotOf(left_in, c);
            spec.left_slots.push_back(slot);
            AddSlot(out, c, left_in, slot);
        }
        for (uint32_t c : right_needed) {
            const uint32_t slot = SlotOf(right_in, c);
            spec.right_slots.push_back(slot);
            AddSlot(out, c + left_count, right_in, slot);
        }
        spec.types = out.slot_types;
        if (join.predicate) spec.predicate = CompiledExpr::Compile(*join.predicate, out);

        auto parts = MakeNestedLoopJoin(std::move(spec));
        current_.pipeline.depends_on.push_back(Close(std::move(right_side), Own(std::move(parts.first))));
        current_.expanding.push_back(parts.second.get());
        AddOperator(std::move(parts.second));
        return out;
    }

    // The pipeline being built gets its one source.
    void SetSource(std::unique_ptr<Source> source) {
        assert(current_.pipeline.source == nullptr && "a pipeline has one source");
        current_.pipeline.source = source.get();
        plan_.sources.push_back(std::move(source));
    }

    // Children are built first, so operators land in the order rows pass through them.
    void AddOperator(std::unique_ptr<Operator> op) {
        current_.pipeline.operators.push_back(op.get());
        plan_.operators.push_back(std::move(op));
    }

    // The plan owns every sink; returns it for a pipeline to point at.
    Sink* Own(std::unique_ptr<Sink> sink) {
        plan_.sinks.push_back(std::move(sink));
        return plan_.sinks.back().get();
    }

    // Ends a pipeline in sink and adds it to the plan; returns its index.
    // Every pipeline it depends on was added before it, so the list stays in order.
    size_t Close(Building building, Sink* sink) {
        assert(building.pipeline.source != nullptr && "a pipeline needs a source");
        building.pipeline.sink = sink;
        plan_.pipelines.push_back(std::move(building.pipeline));
        return plan_.pipelines.size() - 1;
    }

    // Ends the pipeline being built and starts an empty one.
    size_t Close(Sink* sink) { return Close(std::exchange(current_, Building{}), sink); }

    PhysicalPlan& plan_;
    QueryContext& ctx_;
    Building current_;
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
