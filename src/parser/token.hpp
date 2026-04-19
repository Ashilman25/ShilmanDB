#pragma once
#include <string>
#include <cstddef>

namespace shilmandb {


enum class TokenType {
    // Keywords
    SELECT, FROM, WHERE, JOIN, INNER, LEFT, ON, AND, OR, NOT,
    GROUP, BY, ORDER, ASC, DESC, LIMIT, AS, HAVING, BETWEEN, IN, LIKE,
    COUNT, SUM, AVG, MIN, MAX,
    CASE, WHEN, THEN, ELSE, END,

    // Literals
    INTEGER_LITERAL, FLOAT_LITERAL, STRING_LITERAL, DATE_LITERAL,

    // Identifiers
    IDENTIFIER,

    // Operators
    EQUALS, NOT_EQUALS, LESS_THAN, GREATER_THAN, LESS_EQUAL, GREATER_EQUAL,
    PLUS, MINUS, STAR, SLASH,

    // Punctuation
    COMMA, DOT, LPAREN, RPAREN, SEMICOLON,

    // Special
    END_OF_INPUT, INVALID
};

struct Token {
    TokenType type;
    std::string value;
    size_t line;
    size_t column;
};


} //namespace shilmandb