#include "executor/hash_join_executor.hpp"
#include "executor/expression_evaluator.hpp"
#include "planner/plan_node.hpp"
#include "common/exception.hpp"
#include <cassert>

namespace shilmandb {

namespace {


auto FindFirstEQ(const Expression* expr) -> const BinaryOp* {
    if (expr->type != ExprType::BINARY_OP) return nullptr;

    const auto* bin = static_cast<const BinaryOp*>(expr);
    if (bin->op == BinaryOp::Op::EQ && bin->left->type == ExprType::COLUMN_REF && bin->right->type == ExprType::COLUMN_REF) {
        return bin;
    }

    // AND
    if (bin->op == BinaryOp::Op::AND) {
        if (auto* found = FindFirstEQ(bin->left.get())) return found;
        return FindFirstEQ(bin->right.get());
    }
    return nullptr;
}


auto TryResolve(const Schema& schema, const std::string& col_name, uint32_t& out_idx) -> bool {
    for (uint32_t i = 0; i < schema.GetColumnCount(); ++i) {
        if (schema.GetColumn(i).name == col_name) {
            out_idx = i;
            return true;
        }
    }
    return false;
}

}  // namespace

void HashJoinExecutor::ExtractKeyIndices() {
    const auto* join_plan = static_cast<const HashJoinPlanNode*>(plan_);
    const auto* eq = FindFirstEQ(join_plan->join_predicate.get());
    if (!eq) {
        throw DatabaseException("HashJoinExecutor: no equi-join predicate found");
    }

    const auto* col_a = static_cast<const ColumnRef*>(eq->left.get());
    const auto* col_b = static_cast<const ColumnRef*>(eq->right.get());

    const auto& left_schema = left_child_->GetOutputSchema();
    const auto& right_schema = right_child_->GetOutputSchema();


    uint32_t a_left, a_right, b_left, b_right;
    bool a_in_left = TryResolve(left_schema, col_a->column_name, a_left);
    bool a_in_right = TryResolve(right_schema, col_a->column_name, a_right);
    bool b_in_left = TryResolve(left_schema, col_b->column_name, b_left);
    bool b_in_right = TryResolve(right_schema, col_b->column_name, b_right);

    if (a_in_left && !a_in_right && b_in_right && !b_in_left) {
        left_key_idx_ = a_left;
        right_key_idx_ = b_right;
        return;
    }
    if (b_in_left && !b_in_right && a_in_right && !a_in_left) {
        left_key_idx_ = b_left;
        right_key_idx_ = a_right;
        return;
    }

    if (a_in_left && b_in_right && col_a->table_name.has_value()) {
        left_key_idx_ = a_left;
        right_key_idx_ = b_right;
        return;
    }
    if (b_in_left && a_in_right && col_b->table_name.has_value()) {
        left_key_idx_ = b_left;
        right_key_idx_ = a_right;
        return;
    }

    if (a_in_left && b_in_right) {
        left_key_idx_ = a_left;
        right_key_idx_ = b_right;
        return;
    }
    if (b_in_left && a_in_right) {
        left_key_idx_ = b_left;
        right_key_idx_ = a_right;
        return;
    }

    throw DatabaseException("HashJoinExecutor: could not resolve join key columns");
}

void HashJoinExecutor::Init() {
    left_child_->Init();
    right_child_->Init();

    ExtractKeyIndices();

    hash_table_.clear();
    probe_buffer_.clear();
    probe_cursor_ = 0;
    current_bucket_ = nullptr;
    match_idx_ = 0;

    // Pull one row from each side in turn. The first side to run out is the smaller one,
    // and a tie builds from the left. Whatever the other side already gave us is kept to probe with.
    std::vector<Tuple> left_rows, right_rows;
    Tuple row;
    bool left_done = false, right_done = false;
    while (!left_done && !right_done) {
        if (left_child_->Next(&row)) left_rows.push_back(std::move(row)); else left_done = true;
        if (left_done) break;
        if (right_child_->Next(&row)) right_rows.push_back(std::move(row)); else right_done = true;
    }
    build_right_ = right_done;

    auto& build_rows = build_right_ ? right_rows : left_rows;
    const auto& build_schema = build_right_ ? right_child_->GetOutputSchema() : left_child_->GetOutputSchema();
    const uint32_t build_key_idx = build_right_ ? right_key_idx_ : left_key_idx_;

    hash_table_.reserve(build_rows.size());
    for (auto& r : build_rows) {
        auto key_val = r.GetValue(build_schema, build_key_idx);
        hash_table_[std::move(key_val)].push_back(std::move(r));
    }
    probe_buffer_ = std::move(build_right_ ? left_rows : right_rows);

    initialized_ = true;
}

bool HashJoinExecutor::NextProbe(Tuple* out) {
    if (probe_cursor_ < probe_buffer_.size()) {
        *out = std::move(probe_buffer_[probe_cursor_++]);
        return true;
    }
    return (build_right_ ? left_child_ : right_child_)->Next(out);
}

bool HashJoinExecutor::Next(Tuple* tuple) {
    assert(initialized_ && "Next() called before Init()");
    const auto* join_plan = static_cast<const HashJoinPlanNode*>(plan_);
    const auto& left_schema = left_child_->GetOutputSchema();
    const auto& right_schema = right_child_->GetOutputSchema();
    const auto& output_schema = plan_->output_schema;
    const auto& probe_schema = build_right_ ? left_schema : right_schema;
    const uint32_t probe_key_idx = build_right_ ? left_key_idx_ : right_key_idx_;

    while (true) {
        // try current bucket
        while (current_bucket_ && match_idx_ < current_bucket_->size()) {
            const Tuple& entry = (*current_bucket_)[match_idx_++];
            auto combined = build_right_
                ? CombineTuples(current_probe_, left_schema, entry, right_schema, output_schema)
                : CombineTuples(entry, left_schema, current_probe_, right_schema, output_schema);

            auto result = EvaluateExpression(join_plan->join_predicate.get(), combined, output_schema);
            if (IsTruthy(result)) {
                *tuple = std::move(combined);
                return true;
            }
        }

        // next probe tuple
        if (!NextProbe(&current_probe_)) {
            return false;
        }

        auto key_val = current_probe_.GetValue(probe_schema, probe_key_idx);
        auto it = hash_table_.find(key_val);

        if (it != hash_table_.end()) {
            current_bucket_ = &it->second;
            match_idx_ = 0;
        } else {
            current_bucket_ = nullptr;
        }
    }
}

void HashJoinExecutor::Close() {
    left_child_->Close();
    right_child_->Close();
    hash_table_.clear();
    probe_buffer_.clear();
    probe_cursor_ = 0;
    current_bucket_ = nullptr;
    initialized_ = false;
}

}  // namespace shilmandb
