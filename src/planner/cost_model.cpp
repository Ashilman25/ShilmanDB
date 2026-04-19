// src/planner/cost_model.cpp
#include "planner/cost_model.hpp"
#include "common/config.hpp"
#include <algorithm>
#include <cmath>

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

double CostModel::EstimateSelectivity(const Expression* predicate, const std::string& column_name, const TableStats& stats) {
    if (!predicate) return kDefaultSelectivity;

    const auto* bin = dynamic_cast<const BinaryOp*>(predicate);
    if (!bin) return kDefaultSelectivity;

    if (bin->op == BinaryOp::Op::AND) {
        const double l = EstimateSelectivity(bin->left.get(),  column_name, stats);
        const double r = EstimateSelectivity(bin->right.get(), column_name, stats);
        return Clamp(l * r);
    }

    if (bin->op == BinaryOp::Op::OR) {
        const double l = EstimateSelectivity(bin->left.get(),  column_name, stats);
        const double r = EstimateSelectivity(bin->right.get(), column_name, stats);
        return Clamp(l + r - l * r);
    }

    const NormalizedCompare n = NormalizeCompare(bin);
    if (!n.col) return kDefaultSelectivity;
    if (n.col->column_name != column_name) return 1.0;

    switch (n.op) {
        case BinaryOp::Op::EQ: {
            auto it = stats.distinct_counts.find(column_name);
            if (it == stats.distinct_counts.end() || it->second == 0) {
                return kDefaultSelectivity;
            }
            return Clamp(1.0 / static_cast<double>(it->second));
        }
        case BinaryOp::Op::LT:
        case BinaryOp::Op::LTE:
        case BinaryOp::Op::GT:
        case BinaryOp::Op::GTE:
            return kOpenRangeSelectivity;
        default:
            return kDefaultSelectivity;
    }
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
