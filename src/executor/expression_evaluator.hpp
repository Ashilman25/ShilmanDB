#pragma once
#include "parser/ast.hpp"
#include "types/tuple.hpp"
#include "catalog/schema.hpp"
#include "common/exception.hpp"

namespace shilmandb {

inline bool IsTruthy(const Value& v) {
    return v.type_ == TypeId::INTEGER && v.integer_ != 0;
}

inline Value MakeBool(bool b) {
    return Value(static_cast<int32_t>(b ? 1 : 0));
}

inline Value EvaluateExpression(const Expression* expr, const Tuple& tuple, const Schema& schema) {
    switch (expr->type) {
        case ExprType::COLUMN_REF: {
            const auto* col = static_cast<const ColumnRef*>(expr);
            auto idx = schema.GetColumnIndex(col->column_name);
            return tuple.GetValue(schema, idx);
        }

        case ExprType::LITERAL: {
            const auto* lit = static_cast<const Literal*>(expr);
            return lit->value;
        }

        case ExprType::BINARY_OP: {
            const auto* bin = static_cast<const BinaryOp*>(expr);

            if (bin->op == BinaryOp::Op::AND) {
                auto left = EvaluateExpression(bin->left.get(), tuple, schema);
                if (!IsTruthy(left)) return MakeBool(false);
                return MakeBool(IsTruthy(EvaluateExpression(bin->right.get(), tuple, schema)));
            }
            if (bin->op == BinaryOp::Op::OR) {
                auto left = EvaluateExpression(bin->left.get(), tuple, schema);
                if (IsTruthy(left)) return MakeBool(true);
                return MakeBool(IsTruthy(EvaluateExpression(bin->right.get(), tuple, schema)));
            }

            auto left = EvaluateExpression(bin->left.get(), tuple, schema);
            auto right = EvaluateExpression(bin->right.get(), tuple, schema);

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
            auto operand = EvaluateExpression(unary->operand.get(), tuple, schema);

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

        case ExprType::AGGREGATE:
        case ExprType::STAR:
            throw DatabaseException("Cannot evaluate AGGREGATE or STAR expression in this context");
    }

    throw DatabaseException("Unknown expression type");
}

inline auto CombineTuples(const Tuple& left, const Schema& left_schema, const Tuple& right, const Schema& right_schema, const Schema& output_schema) -> Tuple {
    std::vector<Value> values;
    values.reserve(output_schema.GetColumnCount());

    for (uint32_t i = 0; i < left_schema.GetColumnCount(); ++i) {
        values.push_back(left.GetValue(left_schema, i));
    }

    for (uint32_t i = 0; i < right_schema.GetColumnCount(); ++i) {
        values.push_back(right.GetValue(right_schema, i));
    }
    
    return Tuple(std::move(values), output_schema);
}

}  // namespace shilmandb
