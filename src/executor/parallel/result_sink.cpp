#include "executor/parallel/result_sink.hpp"
#include "common/exception.hpp"
#include "executor/parallel/basic_operators.hpp"
#include "executor/parallel/column_buffer.hpp"
#include "executor/parallel/decimal.hpp"
#include "storage/columnar/string_arena.hpp"
#include <algorithm>
#include <string>
#include <utility>

namespace shilmandb::parallel {

namespace {

// A worker's copy of its result rows; its strings live in its own arena.
struct ResultLocal : LocalState {
    std::vector<ColumnBuffer> columns;
    StringArena arena;
    size_t rows{0};
};

// Row r as the Value today's columnar scan would give, built with its type's constructor.
Value MakeValue(const VType& t, const ColumnBuffer& buf, size_t r) {
    switch (t.kind) {
        case Kind::INTEGER:
            return t.result == TypeId::BIGINT ? Value(buf.i64[r]) : Value(static_cast<int32_t>(buf.i64[r]));
        case Kind::DATE:
            return Value::MakeDate(static_cast<int32_t>(buf.i64[r]));
        case Kind::DECIMAL:
            return Value(t.wide ? DecimalToDouble(buf.i128[r], t.scale) : DecimalToDouble(buf.i64[r], t.scale));
        case Kind::DOUBLE:
            return Value(buf.f64[r]);
        case Kind::STRING:
            return Value(std::string(buf.str[r].data, buf.str[r].size));
    }
    return Value();
}

}  // namespace

// Every output column must come from a slot, typed as the plan types it.
ResultSink::ResultSink(Schema schema, const InputLayout& layout) : schema_(std::move(schema)) {
    for (uint32_t c = 0; c < schema_.GetColumnCount(); ++c) {
        const Column& column = schema_.GetColumn(c);
        const int slot = c < layout.slot_of_column.size() ? layout.slot_of_column[c] : -1;
        if (slot < 0 || static_cast<size_t>(slot) >= layout.slot_types.size()) {
            throw DatabaseException("the parallel executor lost result column '" + column.name + "'");
        }
        const VType& t = layout.slot_types[static_cast<size_t>(slot)];
        if (t.result != column.type) {
            throw DatabaseException("the parallel executor types result column '" + column.name + "' differently from the plan");
        }
        slots_.push_back(static_cast<uint32_t>(slot));
        types_.push_back(t);
    }
}

// One buffer per output column.
std::unique_ptr<LocalState> ResultSink::MakeLocal() const {
    auto local = std::make_unique<ResultLocal>();
    local->columns.resize(slots_.size());
    return local;
}

// Copies the active rows into this worker's buffers.
void ResultSink::Consume(Batch& in, LocalState& local) const {
    auto& l = static_cast<ResultLocal&>(local);
    for (size_t c = 0; c < slots_.size(); ++c) Append(in.slots[slots_[c]], in.sel, l.columns[c], l.arena);
    l.rows += in.Active();
}

// Keeps the worker's buffers, arena and all, until Finalize.
void ResultSink::Combine(std::unique_ptr<LocalState> local) {
    parts_.push_back(std::move(local));
}

// Builds every Tuple here, on the calling thread, then frees the buffers.
void ResultSink::Finalize(QueryContext&) {
    size_t total = 0;
    for (const auto& part : parts_) total += static_cast<const ResultLocal&>(*part).rows;
    tuples_.reserve(tuples_.size() + total);
    for (const auto& part : parts_) {
        const auto& p = static_cast<const ResultLocal&>(*part);
        for (size_t r = 0; r < p.rows; ++r) {
            std::vector<Value> values;
            values.reserve(types_.size());
            for (size_t c = 0; c < types_.size(); ++c) values.push_back(MakeValue(types_[c], p.columns[c], r));
            tuples_.emplace_back(std::move(values), schema_);
        }
    }
    parts_.clear();
}

// Done when any limit above the rows is used up.
bool ResultSink::Done() const {
    return std::any_of(limits_.begin(), limits_.end(), [](const LimitCounter* l) { return l->Reached(); });
}

// Swapped out, so the sink holds nothing afterwards.
std::vector<Tuple> ResultSink::TakeTuples() {
    std::vector<Tuple> out;
    out.swap(tuples_);
    return out;
}

}  // namespace shilmandb::parallel
