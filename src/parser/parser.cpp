#include "parser/parser.hpp"
#include "parser/parse_exception.hpp"
#include "types/value.hpp"
#include <stdexcept>

namespace shilmandb {


Parser::Parser(const std::string& sql) : lexer_(sql), current_token_{TokenType::INVALID, "", 0, 0} {}

void Parser::Advance() {
    current_token_ = lexer_.NextToken();
}

void Parser::Expect(TokenType type) {
    if (current_token_.type != type) {
        throw ParseException(
            "Expected token type " + std::to_string(static_cast<int>(type)) +
            " but got " + std::to_string(static_cast<int>(current_token_.type)) +
            " ('" + current_token_.value + "')",
            current_token_.line, current_token_.column);
    }
    Advance();
}

bool Parser::Match(TokenType type) {
    if (current_token_.type == type) {
        Advance();
        return true;
    }
    return false;
}


std::unique_ptr<SelectStatement> Parser::Parse() {
    Advance();  
    auto stmt = ParseSelect();

    if (current_token_.type != TokenType::END_OF_INPUT) {
        Expect(TokenType::SEMICOLON);
    }
    return stmt;
}


std::unique_ptr<SelectStatement> Parser::ParseSelect() {
    Expect(TokenType::SELECT);

    auto stmt = std::make_unique<SelectStatement>();
    stmt->select_list = ParseSelectList();

    Expect(TokenType::FROM);
    stmt->from_clause = ParseFromClause();
    stmt->joins = ParseJoins();

    if (current_token_.type == TokenType::WHERE) {
        stmt->where_clause = ParseWhereClause();
    }
    if (current_token_.type == TokenType::GROUP) {
        stmt->group_by = ParseGroupBy();
    }
    if (current_token_.type == TokenType::HAVING) {
        stmt->having = ParseHaving();
    }
    if (current_token_.type == TokenType::ORDER) {
        stmt->order_by = ParseOrderBy();
    }
    if (current_token_.type == TokenType::LIMIT) {
        stmt->limit = ParseLimit();
    }

    return stmt;
}

//comma separated expressions
std::vector<SelectItem> Parser::ParseSelectList() {
    std::vector<SelectItem> items;

    do {
        SelectItem item;

        if (current_token_.type == TokenType::STAR) {
            item.expr = std::make_unique<StarExpr>();
            Advance();
        } else {
            item.expr = ParseExpression();
        }

        if (Match(TokenType::AS)) {
            item.alias = current_token_.value;
            Advance();
        } else if (current_token_.type == TokenType::IDENTIFIER) {
            item.alias = current_token_.value;
            Advance();
        }

        items.push_back(std::move(item));
    } while (Match(TokenType::COMMA));

    return items;
}

//comma separate table refs
std::vector<TableRef> Parser::ParseFromClause() {
    std::vector<TableRef> tables;

    do {
        TableRef ref;
        ref.table_name = current_token_.value;
        Expect(TokenType::IDENTIFIER);

        if (Match(TokenType::AS)) {
            ref.alias = current_token_.value;
            Advance();
        } else if (current_token_.type == TokenType::IDENTIFIER) {
            ref.alias = current_token_.value;
            Advance();
        }

        tables.push_back(std::move(ref));
    } while (Match(TokenType::COMMA));

    return tables;
}


std::vector<JoinClause> Parser::ParseJoins() {
    std::vector<JoinClause> joins;

    while (current_token_.type == TokenType::JOIN || current_token_.type == TokenType::INNER || current_token_.type == TokenType::LEFT) {
        auto join_type = JoinType::INNER;

        if (current_token_.type == TokenType::LEFT) {
            join_type = JoinType::LEFT;
            Advance();
        } else if (current_token_.type == TokenType::INNER) {
            Advance();
        }

        Expect(TokenType::JOIN);

        JoinClause join;
        join.join_type = join_type;
        join.right_table.table_name = current_token_.value;
        Expect(TokenType::IDENTIFIER);

        if (Match(TokenType::AS)) {
            join.right_table.alias = current_token_.value;
            Advance();
        } else if (current_token_.type == TokenType::IDENTIFIER) {
            join.right_table.alias = current_token_.value;
            Advance();
        }

        Expect(TokenType::ON);
        join.on_condition = ParseExpression();
        joins.push_back(std::move(join));
    }

    return joins;
}


std::unique_ptr<Expression> Parser::ParseWhereClause() {
    Expect(TokenType::WHERE);
    return ParseExpression();
}


std::unique_ptr<Expression> Parser::ParseExpression() {
    auto left = ParseAndExpr();

    while (current_token_.type == TokenType::OR) {
        Advance();

        auto right = ParseAndExpr();
        auto node = std::make_unique<BinaryOp>();

        node->op = BinaryOp::Op::OR;
        node->left = std::move(left);
        node->right = std::move(right);
        left = std::move(node);
    }

    return left;
}

std::unique_ptr<Expression> Parser::ParseAndExpr() {
    auto left = ParseComparison();

    while (current_token_.type == TokenType::AND) {
        Advance();

        auto right = ParseComparison();
        auto node = std::make_unique<BinaryOp>();

        node->op = BinaryOp::Op::AND;
        node->left = std::move(left);
        node->right = std::move(right);
        left = std::move(node);
    }

    return left;
}

//comparisons like <, >, =, <=, >=, !=
std::unique_ptr<Expression> Parser::ParseComparison() {
    auto left = ParseAddSub();

    if (current_token_.type == TokenType::NOT) {
        Token peek = lexer_.PeekToken();
        if (peek.type == TokenType::BETWEEN) {
            Advance();  // consume NOT (returns cached BETWEEN)
            Advance();  // consume BETWEEN (reads next token from input)
            return ParseBetweenTail(std::move(left), /*negated=*/true);
        }
        if (peek.type == TokenType::IN) {
            Advance();  // consume NOT (returns cached IN)
            Advance();  // consume IN (reads next token from input)
            return ParseInTail(std::move(left), /*negated=*/true);
        }
        if (peek.type == TokenType::LIKE) {
            Advance();  // consume NOT (returns cached LIKE)
            Advance();  // consume LIKE (reads next token from input)
            return ParseLikeTail(std::move(left), /*negated=*/true);
        }

    }

    // BETWEEN —
    if (current_token_.type == TokenType::BETWEEN) {
        Advance();
        return ParseBetweenTail(std::move(left), /*negated=*/false);
    }

    // IN
    if (current_token_.type == TokenType::IN) {
        Advance();
        return ParseInTail(std::move(left), /*negated=*/false);
    }

    // LIKE
    if (current_token_.type == TokenType::LIKE) {
        Advance();
        return ParseLikeTail(std::move(left), /*negated=*/false);
    }

    while (true) {
        BinaryOp::Op op;
        switch (current_token_.type) {
            case TokenType::EQUALS:        
                op = BinaryOp::Op::EQ;  
                break;
            case TokenType::NOT_EQUALS:    
                op = BinaryOp::Op::NEQ; 
                break;
            case TokenType::LESS_THAN:     
                op = BinaryOp::Op::LT;  
                break;
            case TokenType::GREATER_THAN:  
                op = BinaryOp::Op::GT;  
                break;
            case TokenType::LESS_EQUAL:    
                op = BinaryOp::Op::LTE; 
                break;
            case TokenType::GREATER_EQUAL: 
                op = BinaryOp::Op::GTE; 
                break;
            default: 
                return left;
        }
        Advance();

        auto right = ParseAddSub();
        auto node = std::make_unique<BinaryOp>();

        node->op = op;
        node->left = std::move(left);
        node->right = std::move(right);
        left = std::move(node);
    }
}

//parse +, -
std::unique_ptr<Expression> Parser::ParseAddSub() {
    auto left = ParseMulDiv();

    while (current_token_.type == TokenType::PLUS || current_token_.type == TokenType::MINUS) {
        auto op = (current_token_.type == TokenType::PLUS) ? BinaryOp::Op::ADD : BinaryOp::Op::SUB;
        Advance();

        auto right = ParseMulDiv();
        auto node = std::make_unique<BinaryOp>();

        node->op = op;
        node->left = std::move(left);
        node->right = std::move(right);
        left = std::move(node);
    }

    return left;
}

//parse *, /
std::unique_ptr<Expression> Parser::ParseMulDiv() {
    auto left = ParseUnary();

    while (current_token_.type == TokenType::STAR || current_token_.type == TokenType::SLASH) {
        auto op = (current_token_.type == TokenType::STAR) ? BinaryOp::Op::MUL : BinaryOp::Op::DIV;
        Advance();

        auto right = ParseUnary();
        auto node = std::make_unique<BinaryOp>();

        node->op = op;
        node->left = std::move(left);
        node->right = std::move(right);
        left = std::move(node);
    }

    return left;
}

//parse NOT, unary minus
std::unique_ptr<Expression> Parser::ParseUnary() {
    if (current_token_.type == TokenType::NOT) {
        Advance();

        auto operand = ParseUnary();
        auto node = std::make_unique<UnaryOp>();

        node->op = UnaryOp::Op::NOT;
        node->operand = std::move(operand);
        return node;
    }

    if (current_token_.type == TokenType::MINUS) {
        Advance();

        auto operand = ParseUnary();
        auto node = std::make_unique<UnaryOp>();

        node->op = UnaryOp::Op::NEGATE;
        node->operand = std::move(operand);
        return node;
    }

    return ParsePrimary();
}


static bool IsAggregateFunc(TokenType type) {
    return type == TokenType::COUNT || type == TokenType::SUM || type == TokenType::AVG   || type == TokenType::MIN || type == TokenType::MAX;
}

static Aggregate::Func ToAggregateFunc(TokenType type) {
    switch (type) {
        case TokenType::COUNT: 
            return Aggregate::Func::COUNT;
        case TokenType::SUM:   
            return Aggregate::Func::SUM;
        case TokenType::AVG:   
            return Aggregate::Func::AVG;
        case TokenType::MIN:   
            return Aggregate::Func::MIN;
        case TokenType::MAX:   
            return Aggregate::Func::MAX;
        default: 
            throw std::logic_error("Not an aggregate function");
    }
}

//parse literals, parens, aggs, col refs, DATE
std::unique_ptr<Expression> Parser::ParsePrimary() {
    //int literal
    if (current_token_.type == TokenType::INTEGER_LITERAL) {
        auto node = std::make_unique<Literal>();

        try {
            node->value = Value(static_cast<int32_t>(std::stoi(current_token_.value)));
        } catch (const std::out_of_range&) {
            throw ParseException("Integer literal out of range: " + current_token_.value, current_token_.line, current_token_.column);
        }

        Advance();
        return node;
    }

    // Float literal
    if (current_token_.type == TokenType::FLOAT_LITERAL) {
        auto node = std::make_unique<Literal>();

        try {
            node->value = Value(std::stod(current_token_.value));
        } catch (const std::out_of_range&) {
            throw ParseException("Float literal out of range: " + current_token_.value, current_token_.line, current_token_.column);
        }
        
        Advance();
        return node;
    }

    // String literal
    if (current_token_.type == TokenType::STRING_LITERAL) {
        auto node = std::make_unique<Literal>();
        node->value = Value(current_token_.value);
        Advance();
        return node;
    }

    // DATE 'YYYY-MM-DD' -> date literal -> string literal
    if (current_token_.type == TokenType::DATE_LITERAL) {
        Advance();

        if (current_token_.type != TokenType::STRING_LITERAL) {
            throw ParseException("Expected date string after DATE keyword", current_token_.line, current_token_.column);
        }

        auto node = std::make_unique<Literal>();
        node->value = Value::FromString(TypeId::DATE, current_token_.value);
        Advance();
        return node;
    }

    // stuff in paranthetsis
    if (current_token_.type == TokenType::LPAREN) {
        Advance();
        auto expr = ParseExpression();
        Expect(TokenType::RPAREN);
        return expr;
    }

    // CASE WHEN <cond> THEN <result> [WHEN <cond> THEN <result> ...]
    // [ELSE <result>] END
    if (current_token_.type == TokenType::CASE) {
        Advance();  // consume CASE
        auto case_expr = std::make_unique<CaseExpression>();

        while (current_token_.type == TokenType::WHEN) {
            Advance();  // consume WHEN
            auto condition = ParseExpression();
            Expect(TokenType::THEN);
            auto result = ParseExpression();
            case_expr->when_clauses.emplace_back(
                std::move(condition), std::move(result));
        }

        if (case_expr->when_clauses.empty()) {
            throw ParseException(
                "CASE expression requires at least one WHEN clause",
                current_token_.line, current_token_.column);
        }

        if (Match(TokenType::ELSE)) {
            case_expr->else_clause = ParseExpression();
        }

        Expect(TokenType::END);
        return case_expr;
    }

    // agg funcs: COUNT, SUM, AVG, MIN, MAX
    if (IsAggregateFunc(current_token_.type)) {
        auto func = ToAggregateFunc(current_token_.type);
        Advance();
        Expect(TokenType::LPAREN);

        auto node = std::make_unique<Aggregate>();
        node->func = func;

        if (func == Aggregate::Func::COUNT && current_token_.type == TokenType::STAR) {
            Advance();
            node->arg = nullptr;
        } else {
            node->arg = ParseExpression();
        }

        Expect(TokenType::RPAREN);
        return node;
    }

    // col ref, identifier or table.column
    if (current_token_.type == TokenType::IDENTIFIER) {
        std::string name = current_token_.value;
        Advance();

        if (current_token_.type == TokenType::DOT) {
            Advance();

            auto node = std::make_unique<ColumnRef>();
            node->table_name = name;
            node->column_name = current_token_.value;

            Expect(TokenType::IDENTIFIER);
            return node;
        }

        auto node = std::make_unique<ColumnRef>();
        node->column_name = name;
        return node;
    }

    //star
    if (current_token_.type == TokenType::STAR) {
        Advance();
        return std::make_unique<StarExpr>();
    }

    throw ParseException("Unexpected token '" + current_token_.value + "'", current_token_.line, current_token_.column);
}

//parse group by expressions
std::vector<std::unique_ptr<Expression>> Parser::ParseGroupBy() {
    Expect(TokenType::GROUP);
    Expect(TokenType::BY);

    std::vector<std::unique_ptr<Expression>> exprs;
    do {
        exprs.push_back(ParseExpression());
    } while (Match(TokenType::COMMA));

    return exprs;
}

//parse having clause
std::unique_ptr<Expression> Parser::ParseHaving() {
    Expect(TokenType::HAVING);
    return ParseExpression();
}

//parse order by express
std::vector<OrderByItem> Parser::ParseOrderBy() {
    Expect(TokenType::ORDER);
    Expect(TokenType::BY);

    std::vector<OrderByItem> items;

    do {
        OrderByItem item;
        item.expr = ParseExpression();
        item.ascending = true;

        if (current_token_.type == TokenType::ASC) {
            Advance();
        } else if (current_token_.type == TokenType::DESC) {
            item.ascending = false;
            Advance();
        }

        items.push_back(std::move(item));
    } while (Match(TokenType::COMMA));

    return items;
}

//limit expr
std::optional<int64_t> Parser::ParseLimit() {
    Expect(TokenType::LIMIT);

    if (current_token_.type != TokenType::INTEGER_LITERAL) {
        throw ParseException("Expected integer after LIMIT", current_token_.line, current_token_.column);
    }

    int64_t val;
    try {
        val = std::stoll(current_token_.value);
    } catch (const std::out_of_range&) {
        throw ParseException("LIMIT value out of range: " + current_token_.value, current_token_.line, current_token_.column);
    }
    Advance();
    return val;
}

// Desugar `lhs BETWEEN low AND high` into `(lhs >= low) AND (lhs <= high)`.
// `low` / `high` are parsed via ParseAddSub so the BETWEEN's required AND
// is not greedily swallowed when the surrounding expression is an AND chain.
std::unique_ptr<Expression> Parser::ParseBetweenTail(std::unique_ptr<Expression> lhs, bool negated) {
    auto low = ParseAddSub();
    Expect(TokenType::AND);
    auto high = ParseAddSub();

    auto gte = std::make_unique<BinaryOp>();
    gte->op = BinaryOp::Op::GTE;
    gte->left = lhs->Clone();
    gte->right = std::move(low);

    auto lte = std::make_unique<BinaryOp>();
    lte->op = BinaryOp::Op::LTE;
    lte->left = std::move(lhs);
    lte->right = std::move(high);

    auto and_node = std::make_unique<BinaryOp>();
    and_node->op = BinaryOp::Op::AND;
    and_node->left = std::move(gte);
    and_node->right = std::move(lte);

    if (negated) {
        auto not_node = std::make_unique<UnaryOp>();
        not_node->op = UnaryOp::Op::NOT;
        not_node->operand = std::move(and_node);
        return not_node;
    }
    return and_node;
}

std::unique_ptr<Expression> Parser::ParseInTail(std::unique_ptr<Expression> lhs, bool negated) {
    Expect(TokenType::LPAREN);

    std::vector<std::unique_ptr<Expression>> values;
    values.push_back(ParseAddSub());
    while (Match(TokenType::COMMA)) {
        values.push_back(ParseAddSub());
    }
    Expect(TokenType::RPAREN);

    const BinaryOp::Op leaf = negated ? BinaryOp::Op::NEQ : BinaryOp::Op::EQ;
    const BinaryOp::Op glue = negated ? BinaryOp::Op::AND : BinaryOp::Op::OR;

    // Right-associative chain: leaf(lhs, v_last) glue (leaf(lhs, ...) glue (...))
    auto result = std::make_unique<BinaryOp>();
    result->op = leaf;
    result->left = lhs->Clone();
    result->right = std::move(values.back());
    values.pop_back();

    while (!values.empty()) {
        auto next = std::make_unique<BinaryOp>();
        next->op = leaf;
        next->left = lhs->Clone();
        next->right = std::move(values.back());
        values.pop_back();

        auto combined = std::make_unique<BinaryOp>();
        combined->op = glue;
        combined->left = std::move(next);
        combined->right = std::move(result);
        result = std::move(combined);
    }
    // `lhs` is cloned once per IN value; the original is never moved and is
    // destroyed via RAII when this function returns.
    return result;
}

std::unique_ptr<Expression> Parser::ParseLikeTail(std::unique_ptr<Expression> lhs, bool negated) {
    auto pattern = ParseAddSub();

    auto like = std::make_unique<BinaryOp>();
    like->op = BinaryOp::Op::LIKE;
    like->left = std::move(lhs);
    like->right = std::move(pattern);

    if (negated) {
        auto not_node = std::make_unique<UnaryOp>();
        not_node->op = UnaryOp::Op::NOT;
        not_node->operand = std::move(like);
        return not_node;
    }
    return like;
}

}  // namespace shilmandb
