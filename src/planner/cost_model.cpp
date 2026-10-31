// src/planner/cost_model.cpp
#include "planner/cost_model.hpp"
#include "common/config.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <optional>

namespace shilmandb {

namespace {

double Clamp(double x) {
    return std::max(CostModel::kMinSelectivity, std::min(1.0, x));
}

// Returns (column_ref, literal, effective_op) with literal-swap normalization.
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
            return op;  // EQ, NEQ, and non-comparison ops 
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

// Everything the range comparisons on one column say, merged into one interval.
struct Interval {
    double col_min{0};
    double col_max{0};
    bool whole_numbers{false};  // INTEGER, BIGINT, DATE
    std::optional<double> lo;
    std::optional<double> hi;

    void RaiseLow(double v) { lo = lo ? std::max(*lo, v) : v; }
    void LowerHigh(double v) { hi = hi ? std::min(*hi, v) : v; }

    [[nodiscard]] double Fraction() const {
        const double from = std::max(lo.value_or(col_min), col_min);
        const double to = std::min(hi.value_or(col_max), col_max);
        if (to < from) return 0.0;
        if (whole_numbers) return (to - from + 1.0) / (col_max - col_min + 1.0);
        if (col_max == col_min) return 1.0;
        return (to - from) / (col_max - col_min);
    }
};

bool IsNumeric(TypeId t) {
    return t == TypeId::INTEGER || t == TypeId::BIGINT || t == TypeId::DECIMAL;
}

std::optional<double> AsNumber(const Value& v) {
    switch (v.type_) {
        case TypeId::INTEGER:
            return v.integer_;
        case TypeId::BIGINT:
            return static_cast<double>(v.bigint_);
        case TypeId::DECIMAL:
            return v.decimal_;
        case TypeId::DATE:
            return v.date_;
        default:
            return std::nullopt;
    }
}

// Adds `col op lit` to the interval. False if the literal can't be compared with the column's range.
bool AddBound(Interval& iv, TypeId col_type, BinaryOp::Op op, const Value& lit) {
    const Value widened = lit.WidenTo(col_type);
    const bool comparable = widened.type_ == col_type || (IsNumeric(widened.type_) && IsNumeric(col_type));
    const auto num = comparable ? AsNumber(widened) : std::nullopt;
    if (!num) return false;

    const double v = *num;
    const bool whole = iv.whole_numbers;
    switch (op) {
        case BinaryOp::Op::GT:
            iv.RaiseLow(whole ? std::floor(v) + 1.0 : v);
            break;
        case BinaryOp::Op::GTE:
            iv.RaiseLow(whole ? std::ceil(v) : v);
            break;
        case BinaryOp::Op::LT:
            iv.LowerHigh(whole ? std::ceil(v) - 1.0 : v);
            break;
        case BinaryOp::Op::LTE:
            iv.LowerHigh(whole ? std::floor(v) : v);
            break;
        default:
            return false;
    }
    return true;
}

bool IsInequality(BinaryOp::Op op) {
    return op == BinaryOp::Op::LT || op == BinaryOp::Op::LTE || op == BinaryOp::Op::GT || op == BinaryOp::Op::GTE;
}

// True for `col < col` style comparisons (either inequality, both sides plain columns).
bool IsColumnVsColumnInequality(const BinaryOp* bin) {
    return bin && IsInequality(bin->op) && dynamic_cast<const ColumnRef*>(bin->left.get()) && dynamic_cast<const ColumnRef*>(bin->right.get());
}

// One matching value out of the column's distinct values.
double EqualitySelectivity(const TableStats& stats, const std::string& column) {
    const auto it = stats.distinct_counts.find(column);
    if (it == stats.distinct_counts.end() || it->second == 0) return CostModel::kDefaultSelectivity;
    return 1.0 / static_cast<double>(it->second);
}

// Splits nested ANDs into their leaves.
void Flatten(const Expression* e, std::vector<const Expression*>& out) {
    if (!e) return;
    const auto* bin = dynamic_cast<const BinaryOp*>(e);
    if (bin && bin->op == BinaryOp::Op::AND) {
        Flatten(bin->left.get(), out);
        Flatten(bin->right.get(), out);
    } else {
        out.push_back(e);
    }
}

}  // namespace

double CostModel::EstimateTablePages(uint64_t row_count, const Schema& schema) {
    const auto bytes_per_tuple = std::max<uint32_t>(1u, schema.GetFixedLength());
    const double total_bytes = static_cast<double>(row_count) * static_cast<double>(bytes_per_tuple);
    const double pages = std::ceil(total_bytes / static_cast<double>(PAGE_SIZE));
    return std::max(1.0, pages);
}

int CostModel::EstimateTreeHeight(uint64_t row_count, TypeId /*key_type*/) {
    if (row_count <= 1) return 1;
    const double height = std::ceil(std::log(static_cast<double>(row_count)) / std::log(static_cast<double>(kBTreeFanOut)));
    return std::max(1, static_cast<int>(height));
}

double CostModel::EstimateSelectivity(const std::vector<const Expression*>& conjuncts, const TableStats& stats, std::string_view only_column) {
    std::vector<const Expression*> leaves;
    for (const auto* c : conjuncts) Flatten(c, leaves);

    const bool only_one = !only_column.empty();
    double sel = 1.0;
    std::map<std::string, Interval> intervals;

    for (const auto* leaf : leaves) {
        const auto* bin = dynamic_cast<const BinaryOp*>(leaf);

        if (bin && bin->op == BinaryOp::Op::OR) {
            const double l = EstimateSelectivity({bin->left.get()}, stats, only_column);
            const double r = EstimateSelectivity({bin->right.get()}, stats, only_column);
            sel *= l + r - l * r;
            continue;
        }

        const NormalizedCompare n = NormalizeCompare(bin);
        if (!n.col) {
            if (only_one) continue;
            sel *= IsColumnVsColumnInequality(bin) ? kOpenRangeSelectivity : kDefaultSelectivity;
            continue;
        }

        const std::string& column = n.col->column_name;  // stats use the bare name, so the table qualifier is ignored
        if (only_one && column != only_column) continue;

        if (n.op == BinaryOp::Op::EQ) {
            sel *= EqualitySelectivity(stats, column);
        } else if (!IsInequality(n.op)) {
            sel *= kDefaultSelectivity;
        } else {
            const auto mm = stats.min_max.find(column);
            bool merged = false;
            if (mm != stats.min_max.end()) {
                const auto lo = AsNumber(mm->second.min);
                const auto hi = AsNumber(mm->second.max);
                
                if (lo && hi) {
                    auto [it, fresh] = intervals.try_emplace(column);
                    const TypeId col_type = mm->second.min.type_;
                    if (fresh) {
                        it->second.col_min = *lo;
                        it->second.col_max = *hi;
                        it->second.whole_numbers = col_type != TypeId::DECIMAL;
                    }
                    merged = AddBound(it->second, col_type, n.op, n.lit->value);
                }
            }
            if (!merged) sel *= kOpenRangeSelectivity;
        }
    }

    for (const auto& [column, interval] : intervals) sel *= interval.Fraction();
    return Clamp(sel);
}

double CostModel::EstimateFilteredRows(const std::vector<const Expression*>& filters, const TableStats& stats) {
    if (stats.row_count == 0) return 0.0;
    return std::max(1.0, static_cast<double>(stats.row_count) * EstimateSelectivity(filters, stats));
}

double CostModel::SeqScanCost(uint64_t row_count, const Schema& schema) {
    const double pages = EstimateTablePages(row_count, schema);
    return pages * kSeqPageReadCost + static_cast<double>(row_count) * kTupleProcessCost;
}

double CostModel::IndexScanCost(uint64_t row_count, double selectivity, TypeId key_type) {
    const int height = EstimateTreeHeight(row_count, key_type);
    const double matching = std::max(1.0, static_cast<double>(row_count) * selectivity);

    return static_cast<double>(height) * kIndexTraversalCost + matching * kRandomPageReadCost + matching * kTupleProcessCost;
}

}  // namespace shilmandb
