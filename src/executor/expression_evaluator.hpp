#pragma once
#include "parser/ast.hpp"
#include "types/tuple.hpp"
#include "catalog/schema.hpp"
#include "common/exception.hpp"
#include "executor/data_chunk.hpp"
#include <cassert>
#include <string>

namespace shilmandb {

inline bool IsTruthy(const Value& v) {
    return v.type_ == TypeId::INTEGER && v.integer_ != 0;
}

inline Value MakeBool(bool b) {
    return Value(static_cast<int32_t>(b ? 1 : 0));
}

inline bool MatchLike(const std::string& str, const std::string& pattern) {
    size_t si = 0, pi = 0;
    size_t star_p = std::string::npos, match_s = 0;

    while (si < str.size()) {
        if (pi < pattern.size() &&
            (pattern[pi] == '_' || pattern[pi] == str[si])) {
            ++si; ++pi;
        } else if (pi < pattern.size() && pattern[pi] == '%') {
            star_p = pi++;
            match_s = si;
        } else if (star_p != std::string::npos) {
            pi = star_p + 1;
            si = ++match_s;
        } else {
            return false;
        }
    }
    while (pi < pattern.size() && pattern[pi] == '%') ++pi;
    return pi == pattern.size();
}

// Reads a row's columns from a Tuple.
struct TupleRow {
    const Tuple& tuple;
    const Schema& schema;
    [[nodiscard]] Value Get(uint32_t idx) const { return tuple.GetValue(schema, idx); }
    [[nodiscard]] const Schema& GetSchema() const { return schema; }
};

// Reads a cell from the decoded column, or straight from the row bytes if not decoded.
struct ChunkRow {
    const DataChunk& chunk;
    size_t physical_row;
    [[nodiscard]] Value Get(uint32_t idx) const {
        return chunk.IsDecoded(idx) ? chunk.GetValue(idx, physical_row) : chunk.DecodeCell(idx, physical_row);
    }
    [[nodiscard]] const Schema& GetSchema() const { return chunk.GetSchema(); }
};

template <typename Row>
Value EvaluateExpression(const Expression* expr, const Row& row) {
    switch (expr->type) {
        case ExprType::COLUMN_REF: {
            const auto* col = static_cast<const ColumnRef*>(expr);
            if (col->bound_index) {
                assert(row.GetSchema().GetColumn(*col->bound_index).name == col->column_name && "column bound against a different schema");
                return row.Get(*col->bound_index);
            }
            return row.Get(row.GetSchema().GetColumnIndex(col->column_name));
        }

        case ExprType::LITERAL: {
            const auto* lit = static_cast<const Literal*>(expr);
            return lit->value;
        }

        case ExprType::BINARY_OP: {
            const auto* bin = static_cast<const BinaryOp*>(expr);

            if (bin->op == BinaryOp::Op::AND) {
                auto left = EvaluateExpression(bin->left.get(), row);
                if (!IsTruthy(left)) return MakeBool(false);
                return MakeBool(IsTruthy(EvaluateExpression(bin->right.get(), row)));
            }
            if (bin->op == BinaryOp::Op::OR) {
                auto left = EvaluateExpression(bin->left.get(), row);
                if (IsTruthy(left)) return MakeBool(true);
                return MakeBool(IsTruthy(EvaluateExpression(bin->right.get(), row)));
            }

            auto left = EvaluateExpression(bin->left.get(), row);
            auto right = EvaluateExpression(bin->right.get(), row);

            switch (bin->op) {
                case BinaryOp::Op::EQ:  
                    return MakeBool(left == right);
                case BinaryOp::Op::NEQ:     
                    return MakeBool(left != right);
                case BinaryOp::Op::LT:  
                    return MakeBool(left < right);
                case BinaryOp::Op::GT:  
                    return MakeBool(left > right);
                case BinaryOp::Op::LTE: 
                    return MakeBool(left <= right);
                case BinaryOp::Op::GTE:
                    return MakeBool(left >= right);
                case BinaryOp::Op::LIKE: {
                    auto lhs_str = left.ToString();
                    auto pattern = right.ToString();
                    return MakeBool(MatchLike(lhs_str, pattern));
                }
                case BinaryOp::Op::ADD:
                    return left.Add(right);
                case BinaryOp::Op::SUB: 
                    return left.Subtract(right);
                case BinaryOp::Op::MUL: 
                    return left.Multiply(right);
                case BinaryOp::Op::DIV: 
                    return left.Divide(right);
                default:
                    throw DatabaseException("Unknown binary operator");
            }
        }

        case ExprType::UNARY_OP: {
            const auto* unary = static_cast<const UnaryOp*>(expr);
            auto operand = EvaluateExpression(unary->operand.get(), row);

            switch (unary->op) {
                case UnaryOp::Op::NOT:
                    return MakeBool(!IsTruthy(operand));
                case UnaryOp::Op::NEGATE:
                    switch (operand.type_) {
                        case TypeId::INTEGER: 
                            return Value(-operand.integer_);
                        case TypeId::BIGINT:  
                            return Value(-operand.bigint_);
                        case TypeId::DECIMAL: 
                            return Value(-operand.decimal_);
                        default:
                            throw DatabaseException("NEGATE not supported for this type");
                    }
            }
            break;
        }

        case ExprType::CASE: {
            const auto* case_expr = static_cast<const CaseExpression*>(expr);
            for (const auto& [cond, result] : case_expr->when_clauses) {
                if (IsTruthy(EvaluateExpression(cond.get(), row))) {
                    return EvaluateExpression(result.get(), row);
                }
            }
            if (case_expr->else_clause) {
                return EvaluateExpression(case_expr->else_clause.get(), row);
            }
            return Value(int32_t{0});  // zero-default (no NULL)
        }

        case ExprType::AGGREGATE:
        case ExprType::STAR:
            throw DatabaseException("Cannot evaluate AGGREGATE or STAR expression in this context");
    }

    throw DatabaseException("Unknown expression type");
}

inline Value EvaluateExpression(const Expression* expr, const Tuple& tuple, const Schema& schema) {
    return EvaluateExpression(expr, TupleRow{tuple, schema});
}

inline auto CombineTuples(const Tuple& left, const Schema& left_schema, const Tuple& right, const Schema& right_schema, const Schema& output_schema) -> Tuple {
    assert(output_schema.GetColumnCount() == left_schema.GetColumnCount() + right_schema.GetColumnCount() && "join output schema must be the left columns then the right columns");
    assert(output_schema.GetFixedLength() == left_schema.GetFixedLength() + right_schema.GetFixedLength() && "join output schema must be the left columns then the right columns");
    (void)output_schema;

    // Copies the two rows' bytes; no column is decoded.
    return Tuple::Concat(left, left_schema, right, right_schema);
}

}  // namespace shilmandb
