#include "planner/index_selector.hpp"
#include "planner/cost_model.hpp"
#include "common/exception.hpp"
#include "storage/columnar/columnar_table.hpp"
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <unordered_set>

namespace shilmandb {

namespace {

// Returns (column_ref, literal, effective_op) after literal-swap normalization.
// operator is mirrored: LT<->GT, LTE<->GTE, EQ<->EQ, NEQ<->NEQ.
struct NormalizedCompare {
    const ColumnRef* col{nullptr};
    const Literal* lit{nullptr};
    BinaryOp::Op op{};
};

BinaryOp::Op MirrorComparison(BinaryOp::Op op) {
    switch (op) {
        case BinaryOp::Op::LT:
            return BinaryOp::Op::GT;
        case BinaryOp::Op::GT:
            return BinaryOp::Op::LT;
        case BinaryOp::Op::LTE:
            return BinaryOp::Op::GTE;
        case BinaryOp::Op::GTE: 
            return BinaryOp::Op::LTE;
        default: 
            return op;
    }
}

NormalizedCompare NormalizeCompare(const BinaryOp* bin) {
    NormalizedCompare out;
    if (!bin) return out;

    const auto* lc = dynamic_cast<const ColumnRef*>(bin->left.get());
    const auto* rc = dynamic_cast<const ColumnRef*>(bin->right.get());
    const auto* ll = dynamic_cast<const Literal*>(bin->left.get());
    const auto* rl = dynamic_cast<const Literal*>(bin->right.get());

    if (lc && rl) {
        out.col = lc;
        out.lit = rl;
        out.op  = bin->op;
    } else if (rc && ll) {
        out.col = rc;
        out.lit = ll;
        out.op  = MirrorComparison(bin->op);
    }
    return out;
}

}  // namespace

ConjunctMatch IndexSelector::MatchSingleConjunct(const Expression* expr, const std::string& indexed_column) {
    ConjunctMatch m;
    const auto* bin = dynamic_cast<const BinaryOp*>(expr);
    if (!bin) return m;

    const NormalizedCompare n = NormalizeCompare(bin);
    if (!n.col) return m;
    if (n.col->column_name != indexed_column) return m;

    switch (n.op) {
        case BinaryOp::Op::EQ:
            m.referenced_column = true;
            m.contributes_low = true;
            m.contributes_high = true;
            m.low_value = n.lit->value;
            m.high_value = n.lit->value;
            m.fully_consumed = true;
            return m;
        case BinaryOp::Op::GTE:
            m.referenced_column = true;
            m.contributes_low = true;
            m.low_value = n.lit->value;
            m.fully_consumed = true;
            return m;
        case BinaryOp::Op::LTE:
            m.referenced_column = true;
            m.contributes_high = true;
            m.high_value = n.lit->value;
            m.fully_consumed = true;
            return m;
        case BinaryOp::Op::GT:
            m.referenced_column = true;
            m.contributes_low  = true;
            m.low_value = n.lit->value;
            m.fully_consumed = false;  // strict > stays in residual
            return m;
        case BinaryOp::Op::LT:
            m.referenced_column = true;
            m.contributes_high = true;
            m.high_value = n.lit->value;
            m.fully_consumed = false;  // strict < stays in residual
            return m;
        default:
            // NEQ, AND, OR, arithmetic
            return m;
    }
}

namespace {

// Significant digits in a decimal literal's text: leading zeros don't count, trailing ones do.
int SignificantDigits(const std::string& text) {
    int digits = 0;
    bool started = false;
    for (const char ch : text) {
        if (ch == 'e' || ch == 'E') break;
        if (ch < '0' || ch > '9') continue;
        started = started || ch != '0';
        if (started) ++digits;
    }
    return digits;
}

// True when the comparison holds a literal written with more digits than a double keeps.
bool HasInexactLiteral(const Expression* expr) {
    const auto* bin = dynamic_cast<const BinaryOp*>(expr);
    if (!bin) return false;
    for (const Expression* side : {bin->left.get(), bin->right.get()}) {
        const auto* lit = dynamic_cast<const Literal*>(side);
        if (!lit || !lit->text || lit->value.type_ != TypeId::DECIMAL) continue;
        // Only text that still says what the value holds.
        char* end = nullptr;
        const double parsed = std::strtod(lit->text->c_str(), &end);
        if (end != lit->text->c_str() + lit->text->size() || parsed != lit->value.decimal_) continue;
        if (SignificantDigits(*lit->text) > kExactDoubleDigits) return true;
    }
    return false;
}

// True when the column's values can pass what a double holds exactly.
bool WiderThanDouble(const ColumnType& type) {
    return type.id == ColumnTypeId::BIGINT || (type.id == ColumnTypeId::NUMERIC && type.precision > kExactDoubleDigits);
}

// One index some predicate uses: its bounds, and the predicates it fully answers.
struct IndexCandidate {
    IndexInfo* index{nullptr};
    std::optional<Value> low;
    std::optional<Value> high;
    std::vector<size_t> consumed;
    std::vector<const Expression*> matched;
};

// Every index of the table a predicate uses, in catalog order. With exact_bounds_only,
// a bound isn't consumed when a double can't hold its literal or column exactly.
std::vector<IndexCandidate> MatchIndexes(const std::string& table_name, const Schema& schema, const std::vector<const Expression*>& predicates, Catalog* catalog, bool exact_bounds_only) {
    std::vector<IndexCandidate> out;
    for (auto* idx_info : catalog->GetTableIndexes(table_name)) {
        const TypeId key_type = schema.GetColumn(idx_info->col_idx).type;
        const bool inexact_key = exact_bounds_only && WiderThanDouble(schema.GetColumn(idx_info->col_idx).storage_type);
        IndexCandidate c;
        c.index = idx_info;
        for (size_t i = 0; i < predicates.size(); ++i) {
            const ConjunctMatch m = IndexSelector::MatchSingleConjunct(predicates[i], idx_info->column_name);
            if (!m.referenced_column) continue;
            c.matched.push_back(predicates[i]);

            // Convert bounds to the key's type once (a date string becomes a DATE).
            if (m.contributes_low) {
                const Value bound = m.low_value.WidenTo(key_type);
                if (!c.low.has_value() || bound > *c.low) c.low = bound;
            }
            if (m.contributes_high) {
                const Value bound = m.high_value.WidenTo(key_type);
                if (!c.high.has_value() || bound < *c.high) c.high = bound;
            }
            // The double bound still holds every match; the exact check stays above the scan.
            const bool inexact = inexact_key || (exact_bounds_only && HasInexactLiteral(predicates[i]));
            if (m.fully_consumed && !inexact) c.consumed.push_back(i);
        }
        if (!c.matched.empty()) out.push_back(std::move(c));
    }
    return out;
}

// Column names an expression reads.
void CollectNames(const Expression* expr, std::unordered_set<std::string>& names) {
    if (!expr) return;
    if (const auto* col = dynamic_cast<const ColumnRef*>(expr)) {
        names.insert(col->column_name);
    } else if (const auto* bin = dynamic_cast<const BinaryOp*>(expr)) {
        CollectNames(bin->left.get(), names);
        CollectNames(bin->right.get(), names);
    } else if (const auto* un = dynamic_cast<const UnaryOp*>(expr)) {
        CollectNames(un->operand.get(), names);
    } else if (const auto* ce = dynamic_cast<const CaseExpression*>(expr)) {
        for (const auto& [cond, result] : ce->when_clauses) {
            CollectNames(cond.get(), names);
            CollectNames(result.get(), names);
        }
        CollectNames(ce->else_clause.get(), names);
    }
}

// Prices one table's scans: today's row-store formulas, or columnar ones when given.
class ScanPricer {
public:
    ScanPricer(const TableInfo& table, const Schema& schema, const std::vector<const Expression*>& predicates, const ScanCosts& costs, const ColumnarScanCosts* columnar, const std::vector<uint32_t>* read_columns) : table_(table), schema_(schema), predicates_(predicates), costs_(costs) {
        if (columnar == nullptr || table.storage != StorageKind::COLUMNAR || !table.columnar) return;
        columnar_ = columnar;

        // The read columns, plus any a predicate names.
        std::unordered_set<std::string> names;
        for (const auto* p : predicates) CollectNames(p, names);
        const uint32_t count = schema.GetColumnCount();
        std::vector<bool> priced(count, read_columns == nullptr);
        if (read_columns != nullptr) {
            for (uint32_t c : *read_columns) {
                if (c < count) priced[c] = true;
            }
        }
        for (uint32_t c = 0; c < count; ++c) {
            const bool filtered = names.find(schema.GetColumn(c).name) != names.end();
            if (priced[c] || filtered) columns_.push_back({c, filtered});
        }
    }

    [[nodiscard]] double Seq() const {
        if (columnar_ == nullptr) return CostModel::SeqScanCost(table_.stats, schema_, costs_);
        const ColumnarTable& t = *table_.columnar;
        const ColumnarScanCosts& c = *columnar_;

        // The row groups the zone maps keep, as the scan will.
        std::vector<size_t> kept;
        double kept_rows = 0.0;
        for (size_t g = 0; g < t.RowGroupCount(); ++g) {
            if (!CostModel::MayMatch(predicates_, t.GroupRanges(g))) continue;
            kept.push_back(g);
            kept_rows += t.RowGroupRows(g);
        }

        // Filtered columns are decoded for every kept row; the rest only for rows that pass.
        const double passed = std::min(kept_rows, static_cast<double>(table_.stats.row_count) * CostModel::EstimateSelectivity(predicates_, table_.stats));
        double cost = 0.0;
        for (const PricedColumn& col : columns_) {
            double pages = 0.0;
            for (size_t g : kept) pages += t.ChunkAt(g, col.index).page_count;
            if (col.filtered) {
                cost += pages * c.page_us + kept_rows * c.value_us;
            } else {
                const double touched = passed >= kept_rows ? pages : ExpectedPagesTouched(pages, passed);
                cost += touched * c.page_us + passed * c.value_us;
            }
        }
        return cost;
    }

    [[nodiscard]] double Index(const IndexCandidate& candidate) const {
        const TypeId key_type = schema_.GetColumn(candidate.index->col_idx).type;
        const double sel = CostModel::EstimateSelectivity(candidate.matched, table_.stats, candidate.index->column_name);
        const uint64_t rows = table_.stats.row_count;
        if (columnar_ == nullptr) return CostModel::IndexScanCost(rows, sel, key_type, costs_);
        const ColumnarTable& t = *table_.columnar;
        const ColumnarScanCosts& c = *columnar_;

        // The descent and the entries walked, then position-ordered fetches per read column.
        const double matches = std::max(1.0, static_cast<double>(rows) * sel);
        double cost = CostModel::EstimateTreeHeight(rows, key_type) * c.page_us + matches * c.index_entry_us;
        for (const PricedColumn& col : columns_) {
            double pages = 0.0;
            for (size_t g = 0; g < t.RowGroupCount(); ++g) pages += t.ChunkAt(g, col.index).page_count;
            cost += ExpectedPagesTouched(pages, matches) * c.page_us + matches * c.position_us;
        }
        return cost;
    }

private:
    struct PricedColumn {
        uint32_t index;
        bool filtered;  // a predicate names it
    };

    const TableInfo& table_;
    const Schema& schema_;
    const std::vector<const Expression*>& predicates_;
    const ScanCosts& costs_;
    const ColumnarScanCosts* columnar_{nullptr};
    std::vector<PricedColumn> columns_;
};

// The parallel executor compares columnar values exactly, so its index bounds must be exact too.
bool ExactBoundsOnly(const TableInfo& table, const ColumnarScanCosts* columnar) {
    return columnar != nullptr && table.storage == StorageKind::COLUMNAR && table.columnar;
}

TableInfo& TableOrThrow(Catalog* catalog, const std::string& table_name) {
    auto* table_info = catalog->GetTable(table_name);
    if (!table_info) {
        throw DatabaseException("IndexSelector: table not found: " + table_name);
    }
    return *table_info;
}

// Splits nested ANDs into their leaves.
void SplitAnd(const Expression* e, std::vector<const Expression*>& out) {
    const auto* bin = dynamic_cast<const BinaryOp*>(e);
    if (bin && bin->op == BinaryOp::Op::AND) {
        SplitAnd(bin->left.get(), out);
        SplitAnd(bin->right.get(), out);
    } else if (e) {
        out.push_back(e);
    }
}

}  // namespace

ScanCostEstimate IndexSelector::EstimateScanCosts(const std::string& table_name, const std::vector<const Expression*>& predicates, Catalog* catalog, const ScanCosts& costs, const ColumnarScanCosts* columnar, const std::vector<uint32_t>* read_columns) {
    const TableInfo& table = TableOrThrow(catalog, table_name);
    std::vector<const Expression*> leaves;
    for (const auto* p : predicates) SplitAnd(p, leaves);

    const ScanPricer pricer(table, table.schema, leaves, costs, columnar, read_columns);
    ScanCostEstimate out;
    out.seq = pricer.Seq();
    for (const auto& candidate : MatchIndexes(table_name, table.schema, leaves, catalog, ExactBoundsOnly(table, columnar))) {
        out.indexes.emplace_back(candidate.index->index_name, pricer.Index(candidate));
    }
    return out;
}

std::unique_ptr<PlanNode> IndexSelector::SelectScanStrategy(const std::string& table_name, const Schema& table_schema, std::vector<std::unique_ptr<Expression>>& predicates,
                                                            std::vector<std::unique_ptr<Expression>>& residual_out, Catalog* catalog, const ScanCosts& costs,
                                                            const ColumnarScanCosts* columnar, const std::vector<uint32_t>* read_columns) {
    const TableInfo& table = TableOrThrow(catalog, table_name);
    std::vector<const Expression*> preds;
    preds.reserve(predicates.size());
    for (const auto& p : predicates) preds.push_back(p.get());

    const ScanPricer pricer(table, table_schema, preds, costs, columnar, read_columns);
    const double seq_cost = pricer.Seq();

    // The cheapest index; the first one wins a tie.
    std::optional<IndexCandidate> best;
    double best_cost = std::numeric_limits<double>::infinity();
    for (auto& candidate : MatchIndexes(table_name, table_schema, preds, catalog, ExactBoundsOnly(table, columnar))) {
        const double idx_cost = pricer.Index(candidate);
        if (idx_cost < best_cost) {
            best_cost = idx_cost;
            best = std::move(candidate);
        }
    }

    if (best && best_cost < seq_cost) {
        auto scan = std::make_unique<IndexScanPlanNode>(table_schema, table_name, best->index->index_name);
        scan->low_key  = best->low;
        scan->high_key = best->high;

        const auto& consumed = best->consumed;
        for (size_t i = 0; i < predicates.size(); ++i) {
            const bool is_consumed = std::find(consumed.begin(), consumed.end(), i) != consumed.end();
            if (is_consumed) {
                predicates[i].reset();  // destroy
            } else {
                residual_out.push_back(std::move(predicates[i]));
            }
        }
        predicates.clear();
        return scan;
    }

    for (auto& p : predicates) residual_out.push_back(std::move(p));
    predicates.clear();
    return std::make_unique<SeqScanPlanNode>(table_schema, table_name);
}

}  // namespace shilmandb
