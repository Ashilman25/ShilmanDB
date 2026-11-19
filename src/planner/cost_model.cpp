// src/planner/cost_model.cpp
#include "planner/cost_model.hpp"
#include "common/config.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
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

    // True when no value of the column's range fits the bounds.
    [[nodiscard]] bool Empty() const {
        return std::min(hi.value_or(col_max), col_max) < std::max(lo.value_or(col_min), col_min);
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

// From 2^53 on, doubles can't hold every whole number, so I don't trust bounds.
constexpr double kExactDoubleLimit = 9007199254740992.0;

// A literal's value, also through a minus sign on a number (how -1 parses).
std::optional<Value> LiteralValue(const Expression* e) {
    if (const auto* lit = dynamic_cast<const Literal*>(e)) return lit->value;
    const auto* neg = dynamic_cast<const UnaryOp*>(e);
    if (!neg || neg->op != UnaryOp::Op::NEGATE) return std::nullopt;
    const auto* lit = dynamic_cast<const Literal*>(neg->operand.get());
    if (!lit) return std::nullopt;
    const Value& v = lit->value;
    switch (v.type_) {
        case TypeId::INTEGER:
            if (v.integer_ == std::numeric_limits<int32_t>::min()) return std::nullopt;
            return Value(-v.integer_);
        case TypeId::BIGINT:
            if (v.bigint_ == std::numeric_limits<int64_t>::min()) return std::nullopt;
            return Value(-v.bigint_);
        case TypeId::DECIMAL:
            return Value(-v.decimal_);
        default:
            return std::nullopt;
    }
}

// A column compared with a value on either side; the op reads col op value.
struct ValueCompare {
    const ColumnRef* col{nullptr};
    Value value;
    BinaryOp::Op op{};
};

std::optional<ValueCompare> CompareWithValue(const BinaryOp* bin) {
    const NormalizedCompare n = NormalizeCompare(bin);
    if (n.col) return ValueCompare{n.col, n.lit->value, n.op};

    const auto* lc = dynamic_cast<const ColumnRef*>(bin->left.get());
    const auto* rc = dynamic_cast<const ColumnRef*>(bin->right.get());
    if (lc) {
        if (auto v = LiteralValue(bin->right.get())) return ValueCompare{lc, *v, bin->op};
    } else if (rc) {
        if (auto v = LiteralValue(bin->left.get())) return ValueCompare{rc, *v, MirrorComparison(bin->op)};
    }
    return std::nullopt;
}

// A usable number, NaN never counts; whole-number columns also need exact doubles.
std::optional<double> SkipNumber(const Value& v, bool whole) {
    const auto num = AsNumber(v);
    if (!num || std::isnan(*num)) return std::nullopt;
    if (whole && std::fabs(*num) >= kExactDoubleLimit) return std::nullopt;
    return num;
}

}  // namespace

double ExpectedPagesTouched(double pages, double positions) {
    if (pages <= 0.0 || positions <= 0.0) return 0.0;
    if (pages <= 1.0) return pages;
    // (1 - 1/p)^k through log1p, so a large p keeps its precision.
    return pages * -std::expm1(positions * std::log1p(-1.0 / pages));
}

bool CostModel::MayMatch(const std::vector<const Expression*>& conjuncts, const std::unordered_map<std::string, MinMax>& ranges) {
    std::vector<const Expression*> leaves;
    for (const auto* c : conjuncts) Flatten(c, leaves);

    std::map<std::string, Interval> intervals;
    for (const auto* leaf : leaves) {
        const auto* bin = dynamic_cast<const BinaryOp*>(leaf);
        if (!bin) continue;

        if (bin->op == BinaryOp::Op::OR) {
            if (!MayMatch({bin->left.get()}, ranges) && !MayMatch({bin->right.get()}, ranges)) return false;
            continue;
        }
        if (bin->op != BinaryOp::Op::EQ && !IsInequality(bin->op)) continue;

        const auto cmp = CompareWithValue(bin);
        if (!cmp) continue;
        const auto range = ranges.find(cmp->col->column_name);
        if (range == ranges.end()) continue;

        // Only number and date ranges, both ends the same type.
        const TypeId col_type = range->second.min.type_;
        if (range->second.max.type_ != col_type) continue;
        if (!IsNumeric(col_type) && col_type != TypeId::DATE) continue;

        const bool whole = col_type != TypeId::DECIMAL;
        const auto lo = SkipNumber(range->second.min, whole);
        const auto hi = SkipNumber(range->second.max, whole);
        const auto lit = SkipNumber(cmp->value.WidenTo(col_type), whole);
        if (!lo || !hi || !lit || *hi < *lo) continue;

        auto [it, fresh] = intervals.try_emplace(cmp->col->column_name);
        if (fresh) {
            it->second.col_min = *lo;
            it->second.col_max = *hi;
            it->second.whole_numbers = whole;
        }
        // An equality is the closed interval [value, value].
        if (cmp->op == BinaryOp::Op::EQ) {
            (void)AddBound(it->second, col_type, BinaryOp::Op::GTE, cmp->value);
            (void)AddBound(it->second, col_type, BinaryOp::Op::LTE, cmp->value);
        } else {
            (void)AddBound(it->second, col_type, cmp->op, cmp->value);
        }
    }

    return std::none_of(intervals.begin(), intervals.end(), [](const auto& entry) { return entry.second.Empty(); });
}

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

double CostModel::TablePages(const TableStats& stats, const Schema& schema) {
    if (stats.page_count > 0) return static_cast<double>(stats.page_count);
    return EstimateTablePages(stats.row_count, schema);
}

double CostModel::SeqScanCost(const TableStats& stats, const Schema& schema, const ScanCosts& costs) {
    return TablePages(stats, schema) * costs.seq_page_us + static_cast<double>(stats.row_count) * costs.scan_row_us;
}

double CostModel::IndexScanCost(uint64_t row_count, double selectivity, TypeId key_type, const ScanCosts& costs) {
    const int height = EstimateTreeHeight(row_count, key_type);
    const double matching = std::max(1.0, static_cast<double>(row_count) * selectivity);

    // Descent, then one random page read plus the row's CPU for every match.
    return static_cast<double>(height) * costs.random_page_us + matching * (costs.random_page_us + costs.index_row_us);
}

}  // namespace shilmandb
