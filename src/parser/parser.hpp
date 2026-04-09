#pragma once
#include "parser/lexer.hpp"
#include "parser/ast.hpp"
#include "parser/token.hpp"
#include <memory>
#include <string>

namespace shilmandb {

class Parser {
public:
    explicit Parser(const std::string& sql);

    [[nodiscard]] std::unique_ptr<SelectStatement> Parse();

private:
    Lexer lexer_;
    Token current_token_;

    void Advance();
    void Expect(TokenType type);
    [[nodiscard]] bool Match(TokenType type);

    [[nodiscard]] std::unique_ptr<SelectStatement> ParseSelect();
    [[nodiscard]] std::vector<SelectItem> ParseSelectList();
    [[nodiscard]] std::vector<TableRef> ParseFromClause();
    [[nodiscard]] std::vector<JoinClause> ParseJoins();
    [[nodiscard]] std::unique_ptr<Expression> ParseWhereClause();
    [[nodiscard]] std::unique_ptr<Expression> ParseExpression();
    [[nodiscard]] std::unique_ptr<Expression> ParseAndExpr();
    [[nodiscard]] std::unique_ptr<Expression> ParseComparison();
    [[nodiscard]] std::unique_ptr<Expression> ParseAddSub();
    [[nodiscard]] std::unique_ptr<Expression> ParseMulDiv();
    [[nodiscard]] std::unique_ptr<Expression> ParseUnary();
    [[nodiscard]] std::unique_ptr<Expression> ParsePrimary();
    [[nodiscard]] std::vector<std::unique_ptr<Expression>> ParseGroupBy();
    [[nodiscard]] std::unique_ptr<Expression> ParseHaving();
    [[nodiscard]] std::vector<OrderByItem> ParseOrderBy();
    [[nodiscard]] std::optional<int64_t> ParseLimit();
};

}  // namespace shilmandb
