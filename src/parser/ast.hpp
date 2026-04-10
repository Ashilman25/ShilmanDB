#pragma once
#include "types/value.hpp"
#include <string>
#include <vector>
#include <memory>
#include <optional>

namespace shilmandb {

enum class ExprType {
    COLUMN_REF, LITERAL, BINARY_OP, UNARY_OP, AGGREGATE, STAR
};

struct Expression {
    ExprType type;
    virtual ~Expression() = default;

    [[nodiscard]] virtual std::unique_ptr<Expression> Clone() const = 0;

protected:
    explicit Expression(ExprType t) : type(t) {}
};

struct ColumnRef : Expression {
    std::optional<std::string> table_name;
    std::string column_name;
    ColumnRef() : Expression(ExprType::COLUMN_REF) {}

    [[nodiscard]] std::unique_ptr<Expression> Clone() const override {
        auto c = std::make_unique<ColumnRef>();
        c->table_name = table_name;
        c->column_name = column_name;
        return c;
    }
};

struct Literal : Expression {
    Value value;
    Literal() : Expression(ExprType::LITERAL) {}

    [[nodiscard]] std::unique_ptr<Expression> Clone() const override {
        auto c = std::make_unique<Literal>();
        c->value = value;
        return c;
    }
};

struct BinaryOp : Expression {
    enum class Op {EQ, NEQ, LT, GT, LTE, GTE, AND, OR, ADD, SUB, MUL, DIV};
    Op op;

    std::unique_ptr<Expression> left;
    std::unique_ptr<Expression> right;
    BinaryOp() : Expression(ExprType::BINARY_OP) {}

    [[nodiscard]] std::unique_ptr<Expression> Clone() const override {
        auto c = std::make_unique<BinaryOp>();
        c->op = op;
        c->left = left ? left->Clone() : nullptr;
        c->right = right ? right->Clone() : nullptr;
        return c;
    }
};

struct UnaryOp : Expression {
    enum class Op {NOT, NEGATE};
    Op op;
    
    std::unique_ptr<Expression> operand;
    UnaryOp() : Expression(ExprType::UNARY_OP) {}

    [[nodiscard]] std::unique_ptr<Expression> Clone() const override {
        auto c = std::make_unique<UnaryOp>();
        c->op = op;
        c->operand = operand ? operand->Clone() : nullptr;
        return c;
    }
};

struct Aggregate : Expression {
    enum class Func {COUNT, SUM, AVG, MIN, MAX};
    Func func;

    std::unique_ptr<Expression> arg; //nullptr for COUNT(*)
    Aggregate() : Expression(ExprType::AGGREGATE) {}

    [[nodiscard]] std::unique_ptr<Expression> Clone() const override {
        auto c = std::make_unique<Aggregate>();
        c->func = func;
        c->arg = arg ? arg->Clone() : nullptr;
        return c;
    }
};


struct StarExpr : Expression {
    StarExpr() : Expression(ExprType::STAR) {}

    [[nodiscard]] std::unique_ptr<Expression> Clone() const override {
        return std::make_unique<StarExpr>();
    }
};


//statement components

struct SelectItem {
    std::unique_ptr<Expression> expr;
    std::optional<std::string> alias;
};

struct TableRef {
    std::string table_name;
    std::optional<std::string> alias;
};

enum class JoinType { INNER, LEFT };

struct JoinClause {
    JoinType join_type{JoinType::INNER};
    TableRef right_table;
    std::unique_ptr<Expression> on_condition;
};

struct OrderByItem {
    std::unique_ptr<Expression> expr;
    bool ascending{true}; //true = asc, false = desc
};


struct SelectStatement {
    std::vector<SelectItem> select_list;
    std::vector<TableRef> from_clause;
    std::vector<JoinClause> joins;
    std::unique_ptr<Expression> where_clause;
    std::vector<std::unique_ptr<Expression>> group_by;
    std::unique_ptr<Expression> having;
    std::vector<OrderByItem> order_by;
    std::optional<int64_t> limit;
};


} //namespace shilmandb