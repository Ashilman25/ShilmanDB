#include "executor/parallel/compiler.hpp"
#include "catalog/catalog.hpp"
#include "common/exception.hpp"
#include "executor/columnar_scan_executor.hpp"
#include "executor/parallel/basic_operators.hpp"
#include "executor/parallel/expression.hpp"
#include "executor/parallel/hash_aggregate.hpp"
#include "executor/parallel/hash_join.hpp"
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

// A plain column's slot precision in the layout, else 0 (the widest its type allows).
uint8_t PrecisionOf(const Expression* expr, const InputLayout& in) {
    const auto* ref = dynamic_cast<const ColumnRef*>(expr);
    if (ref == nullptr || !ref->bound_index) return 0;
    const uint32_t c = *ref->bound_index;
    const int slot = c < in.slot_of_column.size() ? in.slot_of_column[c] : -1;
    if (slot < 0 || static_cast<size_t>(slot) >= in.slot_precision.size()) return 0;
    return in.slot_precision[static_cast<size_t>(slot)];
}

// A child's layout from its join's side: column c is child column c - offset.
InputLayout ShiftLayout(const InputLayout& child, uint32_t offset, uint32_t join_columns) {
    InputLayout out = child;
    out.slot_of_column.assign(join_columns, -1);
    for (size_t c = 0; c < child.slot_of_column.size() && c + offset < join_columns; ++c) out.slot_of_column[c + offset] = child.slot_of_column[c];
    return out;
}

// Adds c to a sorted list unless it's there.
void InsertSorted(std::vector<uint32_t>& columns, uint32_t c) {
    const auto at = std::lower_bound(columns.begin(), columns.end(), c);
    if (at == columns.end() || *at != c) columns.insert(at, c);
}

// A fused filter's predicate, or none.
const Expression* PredicateOf(const FilterPlanNode* filter) {
    return filter != nullptr ? filter->predicate.get() : nullptr;
}

// A hash join's key: its equality and the column it reads on each child.
struct JoinKey {
    const BinaryOp* eq{nullptr};
    uint32_t left{0};   // a left child column
    uint32_t right{0};  // a right child column, numbered as the right child does
};

// The first column with this name, as today's hash join looks it up.
std::optional<uint32_t> FirstNamed(const Schema& schema, const std::string& name) {
    for (uint32_t i = 0; i < schema.GetColumnCount(); ++i) {
        if (schema.GetColumn(i).name == name) return i;
    }
    return std::nullopt;
}

// Today's rule: a column found on one side only decides; else a qualified
// column, a before b, is taken as the left one; else a left, b right.
std::optional<std::pair<uint32_t, uint32_t>> ResolveKeyColumns(const ColumnRef& a, const ColumnRef& b, const Schema& left, const Schema& right) {
    const std::optional<uint32_t> a_left = FirstNamed(left, a.column_name);
    const std::optional<uint32_t> a_right = FirstNamed(right, a.column_name);
    const std::optional<uint32_t> b_left = FirstNamed(left, b.column_name);
    const std::optional<uint32_t> b_right = FirstNamed(right, b.column_name);
    if (a_left && !a_right && b_right && !b_left) return std::make_pair(*a_left, *b_right);
    if (b_left && !b_right && a_right && !a_left) return std::make_pair(*b_left, *a_right);
    if (a_left && b_right && a.table_name) return std::make_pair(*a_left, *b_right);
    if (b_left && a_right && b.table_name) return std::make_pair(*b_left, *a_right);
    if (a_left && b_right) return std::make_pair(*a_left, *b_right);
    if (b_left && a_right) return std::make_pair(*b_left, *a_right);
    return std::nullopt;
}

// The first EQ of two columns that resolves to one column per child, searching
// ANDs left first. Columns resolve by name in each child's schema, not by binding.
std::optional<JoinKey> FindJoinKey(const Expression& e, const Schema& left, const Schema& right) {
    if (e.type != ExprType::BINARY_OP) return std::nullopt;
    const auto& b = static_cast<const BinaryOp&>(e);
    if (b.op == BinaryOp::Op::AND) {
        if (std::optional<JoinKey> found = FindJoinKey(*b.left, left, right)) return found;
        return FindJoinKey(*b.right, left, right);
    }
    if (b.op != BinaryOp::Op::EQ || b.left->type != ExprType::COLUMN_REF || b.right->type != ExprType::COLUMN_REF) return std::nullopt;
    const auto columns = ResolveKeyColumns(static_cast<const ColumnRef&>(*b.left), static_cast<const ColumnRef&>(*b.right), left, right);
    if (!columns) return std::nullopt;
    return JoinKey{&b, columns->first, columns->second};
}

// A column reference bound to column c, for compiling a key found by name.
ColumnRef BoundColumn(const std::string& name, uint32_t c) {
    ColumnRef ref;
    ref.column_name = name;
    ref.bound_index = c;
    return ref;
}

// Makes slot `slot` of from the next slot of out, holding out's column c.
void AddSlot(InputLayout& out, uint32_t c, const InputLayout& from, uint32_t slot) {
    out.slot_of_column[c] = static_cast<int>(out.slot_types.size());
    out.slot_types.push_back(from.slot_types[slot]);
    out.slot_precision.push_back(slot < from.slot_precision.size() ? from.slot_precision[slot] : 0);
}

// A join's columns read above it or by its predicate, split by side.
struct JoinSides {
    std::vector<uint32_t> left;   // left child columns, sorted
    std::vector<uint32_t> right;  // right child columns, numbered as the right child does, sorted
};

// Splits sorted join columns at the left child's column count.
JoinSides SplitSides(const std::vector<uint32_t>& columns, uint32_t left_count) {
    JoinSides sides;
    for (uint32_t c : columns) {
        if (c < left_count) {
            sides.left.push_back(c);
        } else {
            sides.right.push_back(c - left_count);
        }
    }
    return sides;
}

// Where a join's output columns sit, and which child slot each one copies.
struct JoinOutput {
    InputLayout layout;
    std::vector<uint32_t> left_slots;   // per left column: its slot in left batches
    std::vector<uint32_t> right_slots;  // per right column: its slot in right batches
};

// Every join hands up the left side's columns first, then the right side's.
JoinOutput LayOutJoin(const JoinSides& sides, uint32_t join_count, uint32_t left_count, const InputLayout& left_in, const InputLayout& right_in) {
    JoinOutput out;
    out.layout.slot_of_column.assign(join_count, -1);
    for (uint32_t c : sides.left) {
        out.left_slots.push_back(SlotOf(left_in, c));
        AddSlot(out.layout, c, left_in, out.left_slots.back());
    }
    for (uint32_t c : sides.right) {
        out.right_slots.push_back(SlotOf(right_in, c));
        AddSlot(out.layout, c + left_count, right_in, out.right_slots.back());
    }
    return out;
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
        CheckLimitsAboveExpanding();
        for (const LimitOperator* limit : current_.limits) sink->WatchLimit(&limit->Counter());
        plan_.result = sink.get();
        Close(Own(std::move(sink)), &root);
    }

private:
    // Done() stops a pipeline mid-batch. A watched limit below an operator still
    // expanding its batch would cut that batch short, so every limit must come after.
    // The planner puts LIMIT at the root, so this holds today; every build checks it.
    void CheckLimitsAboveExpanding() const {
        const auto& ops = current_.pipeline.operators;
        const auto position = [&](const Operator* op) { return std::find(ops.begin(), ops.end(), op) - ops.begin(); };
        for (const LimitOperator* limit : current_.limits) {
            for (const Operator* op : current_.expanding) {
                if (position(limit) <= position(op)) throw DatabaseException("the parallel executor can't run a LIMIT below a join");
            }
        }
    }

    // Compiles node's subtree; returns where node's output columns sit in its batches.
    // needed lists node's output columns read above it, sorted.
    InputLayout Build(const PlanNode& node, const std::vector<uint32_t>& needed) {
        switch (node.type) {
            case PlanNodeType::SEQ_SCAN:
                return Scan(static_cast<const SeqScanPlanNode&>(node), nullptr, needed);
            case PlanNodeType::INDEX_SCAN:
                return IndexScan(static_cast<const IndexScanPlanNode&>(node), nullptr, needed);
            case PlanNodeType::FILTER: {
                const auto& filter = static_cast<const FilterPlanNode&>(node);
                const PlanNode& child = *filter.children[0];
                if (child.type == PlanNodeType::SEQ_SCAN) {
                    return Scan(static_cast<const SeqScanPlanNode&>(child), &filter, needed);
                }
                if (child.type == PlanNodeType::INDEX_SCAN) {
                    return IndexScan(static_cast<const IndexScanPlanNode&>(child), &filter, needed);
                }
                return Filter(filter, needed);
            }
            case PlanNodeType::PROJECTION:
                return Projection(static_cast<const ProjectionPlanNode&>(node), needed);
            case PlanNodeType::LIMIT: {
                const auto& limit = static_cast<const LimitPlanNode&>(node);
                if (limit.children[0]->type == PlanNodeType::SORT) {
                    return Sort(static_cast<const SortPlanNode&>(*limit.children[0]), needed, &limit);
                }
                return Limit(limit, needed);
            }
            case PlanNodeType::SORT:
                return Sort(static_cast<const SortPlanNode&>(node), needed, nullptr);
            case PlanNodeType::NESTED_LOOP_JOIN:
                return NestedLoopJoin(static_cast<const NestedLoopJoinPlanNode&>(node), needed);
            case PlanNodeType::HASH_JOIN:
                return HashJoin(static_cast<const HashJoinPlanNode&>(node), needed);
            case PlanNodeType::AGGREGATE:
                return HashAggregate(static_cast<const AggregatePlanNode&>(node));
            default:
                break;
        }
        throw DatabaseException(std::string("the parallel executor can't run ") + PlanNodeTypeName(node.type) + " yet");
    }

    // Slot k holds needed[k]; NUMERIC slots carry their precision so sums can stay narrow.
    // A fused predicate's columns are read inside the scan and never handed up.
    // NULLs are refused in every column the plan lists, as today's scan does.
    ScanSpec SpecFor(const std::string& table_name, const Schema& schema, const std::optional<std::vector<uint32_t>>& read_columns, const Expression* predicate, const std::vector<uint32_t>& needed) {
        const TableInfo& info = *ctx_.catalog->GetTable(table_name);
        const ColumnarTable* table = info.columnar.get();
        if (schema.GetColumnCount() != table->Meta().columns.size()) {
            throw DatabaseException("the parallel executor: the plan's columns don't match table " + table_name);
        }

        ScanSpec spec;
        spec.table = table_name;
        spec.columns = needed;
        spec.refuse_nulls = ColumnarReadColumns(info, read_columns);
        if (predicate != nullptr) AddConjuncts(spec, *table, *predicate);
#ifndef NDEBUG
        // An index key may be unlisted when its predicate was used up; it isn't read.
        if (read_columns) {
            std::vector<uint32_t> read = needed;
            read.insert(read.end(), spec.single_column.begin(), spec.single_column.end());
            for (const auto& columns : spec.multi_columns) read.insert(read.end(), columns.begin(), columns.end());
            for (uint32_t c : read) {
                const bool listed = std::find(read_columns->begin(), read_columns->end(), c) != read_columns->end();
                assert(listed && "a scan reads a column its plan doesn't list");
            }
        }
#endif
        return spec;
    }

    // Row groups whose min and max rule out a match are never read.
    // A filter right above runs inside; it has no time of its own.
    InputLayout Scan(const SeqScanPlanNode& scan, const FilterPlanNode* filter, const std::vector<uint32_t>& needed) {
        ScanSpec spec = SpecFor(scan.table_name, scan.output_schema, scan.read_columns, PredicateOf(filter), needed);
        for (const auto& conj : scan.skip_conjuncts) spec.skip_conjuncts.push_back(conj.get());
        SetSource(MakeScanSource(std::move(spec)), &scan, filter);
        return ScanLayout(*ctx_.catalog->GetTable(scan.table_name)->columnar, needed);
    }

    // The same scan over the index's matches, [low, high] inclusive as today's index scan.
    // Rows come in row-position order; a filter right above still runs inside.
    InputLayout IndexScan(const IndexScanPlanNode& scan, const FilterPlanNode* filter, const std::vector<uint32_t>& needed) {
        if (ctx_.catalog->GetIndex(scan.index_name) == nullptr) throw DatabaseException("IndexScanExecutor: index not found: " + scan.index_name);
        ScanSpec spec = SpecFor(scan.table_name, scan.output_schema, scan.read_columns, PredicateOf(filter), needed);
        SetSource(MakeIndexScanSource(std::move(spec), scan.index_name, scan.low_key, scan.high_key), &scan, filter);
        return ScanLayout(*ctx_.catalog->GetTable(scan.table_name)->columnar, needed);
    }

    // The batch passes through, so the filter's layout is its child's.
    // Columns only the predicate reads ride upward too, so a sink that keeps
    // its input must copy just the columns it needs, never every slot.
    InputLayout Filter(const FilterPlanNode& filter, const std::vector<uint32_t>& needed) {
        std::vector<uint32_t> child_needed = needed;
        CollectColumns(*filter.predicate, child_needed);
        const InputLayout layout = Build(*filter.children[0], child_needed);
        AddOperator(std::make_unique<FilterOperator>(CompiledExpr::Compile(*filter.predicate, layout)), &filter);
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
        AddOperator(std::move(op), &projection);
        return out;
    }

    // A result sink in this pipeline watches the counter, stopping workers at the limit.
    InputLayout Limit(const LimitPlanNode& limit, const std::vector<uint32_t>& needed) {
        const InputLayout layout = Build(*limit.children[0], needed);
        auto op = std::make_unique<LimitOperator>(static_cast<uint64_t>(std::max<int64_t>(limit.limit, 0)));
        current_.limits.push_back(op.get());
        AddOperator(std::move(op), &limit);
        return layout;
    }

    // A breaker: the child's pipeline ends in the sort's sink, and the next pipeline
    // starts from its source. Slot k holds needed[k]; a LIMIT above becomes its top n,
    // reporting the rows the top n hands up. Its own time is only the result
    // sink's, when it is the root.
    InputLayout Sort(const SortPlanNode& sort, const std::vector<uint32_t>& needed, const LimitPlanNode* limit) {
        std::vector<uint32_t> child_needed = needed;
        for (const auto& item : sort.order_by) CollectColumns(*item.expr, child_needed);
        const InputLayout in = Build(*sort.children[0], child_needed);

        SortSpec spec;
        for (const auto& item : sort.order_by) {
            spec.keys.push_back(CompiledExpr::Compile(*item.expr, in));
            spec.ascending.push_back(item.ascending);
        }
        if (limit != nullptr) spec.limit = limit->limit;
        InputLayout out;
        out.slot_of_column.assign(sort.output_schema.GetColumnCount(), -1);
        for (uint32_t c : needed) {
            const uint32_t slot = SlotOf(in, c);
            spec.columns.push_back(slot);
            spec.types.push_back(in.slot_types[slot]);
            AddSlot(out, c, in, slot);
        }

        auto parts = MakeSort(std::move(spec));
        const size_t sorted = Close(Own(std::move(parts.first)), &sort);
        SetSource(std::move(parts.second), &sort, limit);
        current_.pipeline.depends_on.push_back(sorted);
        return out;
    }

    // A breaker: the child's pipeline ends in the aggregate's sink, and the next
    // pipeline starts from its source, which emits output column c as slot c.
    // COUNT never reads its argument, so its columns aren't asked for.
    InputLayout HashAggregate(const AggregatePlanNode& agg) {
        const size_t key_count = agg.group_by_exprs.size();
        const size_t agg_count = agg.aggregate_funcs.size();
        if (agg.aggregate_exprs.size() != agg_count || agg.output_schema.GetColumnCount() != key_count + agg_count) {
            throw DatabaseException("the parallel executor groups only by plain columns");
        }

        const auto reads_arg = [&](size_t i) { return agg.aggregate_funcs[i] != Aggregate::Func::COUNT && agg.aggregate_exprs[i] != nullptr; };
        std::vector<uint32_t> child_needed;
        for (const auto& key : agg.group_by_exprs) CollectColumns(*key, child_needed);
        for (size_t i = 0; i < agg_count; ++i) {
            if (reads_arg(i)) CollectColumns(*agg.aggregate_exprs[i], child_needed);
        }
        const InputLayout in = Build(*agg.children[0], child_needed);

        // Plain-column keys and MIN / MAX keep their precision; sums are wide, checked.
        AggregateSpec spec;
        InputLayout out;
        for (const auto& key : agg.group_by_exprs) {
            spec.keys.push_back(CompiledExpr::Compile(*key, in));
            spec.out_types.push_back(spec.keys.back()->Type());
            out.slot_precision.push_back(PrecisionOf(key.get(), in));
        }

        for (size_t i = 0; i < agg_count; ++i) {
            const Aggregate::Func func = agg.aggregate_funcs[i];
            spec.funcs.push_back(func);
            spec.args.push_back(reads_arg(i) ? CompiledExpr::Compile(*agg.aggregate_exprs[i], in) : nullptr);
            spec.out_types.push_back(AggregateOutputType(func, spec.args.back().get()));
            const bool extreme = func == Aggregate::Func::MIN || func == Aggregate::Func::MAX;
            out.slot_precision.push_back(extreme ? PrecisionOf(agg.aggregate_exprs[i].get(), in) : 0);
        }
        
        spec.global = key_count == 0;
        out.slot_of_column.resize(key_count + agg_count);
        std::iota(out.slot_of_column.begin(), out.slot_of_column.end(), 0);
        out.slot_types = spec.out_types;

        auto parts = MakeHashAggregate(std::move(spec));
        const size_t grouped = Close(Own(std::move(parts.first)), &agg);
        SetSource(std::move(parts.second), &agg);
        current_.pipeline.depends_on.push_back(grouped);
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
        const JoinSides sides = SplitSides(columns, left_count);

        // The right side gets a pipeline of its own; it closes once the join's
        // predicate, which needs both sides' layouts, is compiled.
        Building outer = std::exchange(current_, Building{});
        const InputLayout right_in = Build(right, sides.right);
        Building right_side = std::exchange(current_, std::move(outer));
        const InputLayout left_in = Build(left, sides.left);

        JoinOutput out = LayOutJoin(sides, join.output_schema.GetColumnCount(), left_count, left_in, right_in);
        NestedLoopSpec spec;
        spec.left_slots = std::move(out.left_slots);
        spec.right_slots = std::move(out.right_slots);
        spec.types = out.layout.slot_types;
        if (join.predicate) spec.predicate = CompiledExpr::Compile(*join.predicate, out.layout);

        auto parts = MakeNestedLoopJoin(std::move(spec));
        current_.pipeline.depends_on.push_back(Close(std::move(right_side), Own(std::move(parts.first)), &join));
        current_.expanding.push_back(parts.second.get());
        AddOperator(std::move(parts.second), &join);
        return std::move(out.layout);
    }

    // The build child runs first, into the join's sink; probe batches stream through
    // the join. The smaller known estimate builds; a tie or an unknown builds the right.
    // Output slots: the needed and residual columns, left ones first.
    InputLayout HashJoin(const HashJoinPlanNode& join, const std::vector<uint32_t>& needed) {
        const PlanNode& left = *join.children[0];
        const PlanNode& right = *join.children[1];
        const uint32_t left_count = left.output_schema.GetColumnCount();
        const uint32_t join_count = join.output_schema.GetColumnCount();
        assert(join_count == left_count + right.output_schema.GetColumnCount());

        const std::optional<JoinKey> key = join.join_predicate ? FindJoinKey(*join.join_predicate, left.output_schema, right.output_schema) : std::nullopt;
        if (!key) throw DatabaseException("HashJoinExecutor: no equi-join predicate found");
        const ColumnRef left_key = BoundColumn(left.output_schema.GetColumn(key->left).name, key->left);
        const ColumnRef right_key = BoundColumn(right.output_schema.GetColumn(key->right).name, left_count + key->right);
        std::vector<const Expression*> rest;
        SplitAnd(*join.join_predicate, rest);
        rest.erase(std::remove(rest.begin(), rest.end(), key->eq), rest.end());

        // Each side hands up its output and residual columns, plus its key.
        std::vector<uint32_t> columns = needed;
        for (const Expression* e : rest) CollectColumns(*e, columns);
        const JoinSides sides = SplitSides(columns, left_count);
        std::vector<uint32_t> left_read = sides.left;
        std::vector<uint32_t> right_read = sides.right;
        InsertSorted(left_read, key->left);
        InsertSorted(right_read, key->right);

        // The build side gets a pipeline of its own, closed once the join is made.
        const bool build_left = join.est_left_rows >= 0 && join.est_right_rows >= 0 && join.est_left_rows < join.est_right_rows;
        Building outer = std::exchange(current_, Building{});
        const InputLayout build_in = Build(build_left ? left : right, build_left ? left_read : right_read);
        Building build_side = std::exchange(current_, std::move(outer));
        const InputLayout probe_in = Build(build_left ? right : left, build_left ? right_read : left_read);
        const InputLayout& left_in = build_left ? build_in : probe_in;
        const InputLayout& right_in = build_left ? probe_in : build_in;

        JoinOutput out = LayOutJoin(sides, join_count, left_count, left_in, right_in);
        JoinSpec spec;
        auto left_expr = CompiledExpr::Compile(left_key, ShiftLayout(left_in, 0, join_count));
        auto right_expr = CompiledExpr::Compile(right_key, ShiftLayout(right_in, left_count, join_count));
        spec.build_key = std::move(build_left ? left_expr : right_expr);
        spec.probe_key = std::move(build_left ? right_expr : left_expr);
        spec.build_payload = build_left ? out.left_slots : out.right_slots;
        spec.probe_slots = build_left ? out.right_slots : out.left_slots;
        spec.types = out.layout.slot_types;
        spec.residual = CompiledExpr::CompileAnd(rest, out.layout);
        spec.build_left = build_left;

        auto parts = MakeHashJoin(std::move(spec));
        current_.pipeline.depends_on.push_back(Close(std::move(build_side), Own(std::move(parts.first)), &join));
        current_.expanding.push_back(parts.second.get());
        AddOperator(std::move(parts.second), &join);
        return std::move(out.layout);
    }

    // The pipeline being built gets its one source, charged to node.
    // A node fused into the source (a filter, a limit) gets the rows it hands up.
    void SetSource(std::unique_ptr<Source> source, const PlanNode* node, const PlanNode* fused = nullptr) {
        assert(current_.pipeline.source == nullptr && "a pipeline has one source");
        current_.pipeline.source = source.get();
        current_.pipeline.source_node = node;
        if (fused != nullptr) current_.pipeline.source_fused.push_back(fused);
        plan_.sources.push_back(std::move(source));
    }

    // Children are built first, so operators land in the order rows pass through them.
    void AddOperator(std::unique_ptr<Operator> op, const PlanNode* node) {
        current_.pipeline.operators.push_back(op.get());
        current_.pipeline.operator_nodes.push_back(node);
        plan_.operators.push_back(std::move(op));
    }

    // The plan owns every sink; returns it for a pipeline to point at.
    Sink* Own(std::unique_ptr<Sink> sink) {
        plan_.sinks.push_back(std::move(sink));
        return plan_.sinks.back().get();
    }

    // Ends a pipeline in sink, charged to node; returns its index in the plan.
    // Every pipeline it depends on was added before it, so the list stays in order.
    size_t Close(Building building, Sink* sink, const PlanNode* node) {
        assert(building.pipeline.source != nullptr && "a pipeline needs a source");
        building.pipeline.sink = sink;
        building.pipeline.sink_node = node;
        plan_.pipelines.push_back(std::move(building.pipeline));
        return plan_.pipelines.size() - 1;
    }

    // Ends the pipeline being built and starts an empty one.
    size_t Close(Sink* sink, const PlanNode* node) { return Close(std::exchange(current_, Building{}), sink, node); }

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
