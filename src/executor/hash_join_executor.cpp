#include "executor/hash_join_executor.hpp"
#include "executor/expression_evaluator.hpp"
#include "planner/plan_node.hpp"
#include "common/exception.hpp"

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
    const auto& left_schema = left_child_->GetOutputSchema();
    Tuple left_tuple;

    while (left_child_->Next(&left_tuple)) {
        auto key_val = left_tuple.GetValue(left_schema, left_key_idx_);
        hash_table_[key_val].push_back(left_tuple);
    }

    // reset for probing
    current_bucket_ = nullptr;
    match_idx_ = 0;
}

bool HashJoinExecutor::Next(Tuple* tuple) {
    const auto* join_plan = static_cast<const HashJoinPlanNode*>(plan_);
    const auto& left_schema = left_child_->GetOutputSchema();
    const auto& right_schema = right_child_->GetOutputSchema();
    const auto& output_schema = plan_->output_schema;

    while (true) {
        //try current bucket
        while (current_bucket_ && match_idx_ < current_bucket_->size()) {
            auto combined = CombineTuples((*current_bucket_)[match_idx_], left_schema, current_right_, right_schema, output_schema);
            ++match_idx_;

            auto result = EvaluateExpression(join_plan->join_predicate.get(), combined, output_schema);
            if (IsTruthy(result)) {
                *tuple = std::move(combined);
                return true;
            }
        }

        // next right tuple
        if (!right_child_->Next(&current_right_)) {
            return false;
        }

        // hash table
        auto key_val = current_right_.GetValue(right_schema, right_key_idx_);
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
}

}  // namespace shilmandb
