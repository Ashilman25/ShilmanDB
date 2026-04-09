#include "parser/lexer.hpp"
#include "parser/parse_exception.hpp"
#include <unordered_map>
#include <algorithm>
#include <cctype>

namespace shilmandb {

// Keyword table -- keys are UPPERCASE, maps to TokenType
static const std::unordered_map<std::string, TokenType> kKeywords = {
    {"SELECT", TokenType::SELECT},
    {"FROM", TokenType::FROM},
    {"WHERE", TokenType::WHERE},
    {"JOIN", TokenType::JOIN},
    {"INNER", TokenType::INNER},
    {"LEFT", TokenType::LEFT},
    {"ON", TokenType::ON},
    {"AND", TokenType::AND},
    {"OR", TokenType::OR},
    {"NOT", TokenType::NOT},
    {"GROUP", TokenType::GROUP},
    {"BY", TokenType::BY},
    {"ORDER", TokenType::ORDER},
    {"ASC", TokenType::ASC},
    {"DESC", TokenType::DESC},
    {"LIMIT", TokenType::LIMIT},
    {"AS", TokenType::AS},
    {"HAVING", TokenType::HAVING},
    {"BETWEEN", TokenType::BETWEEN},
    {"IN", TokenType::IN},
    {"LIKE", TokenType::LIKE},
    {"COUNT", TokenType::COUNT},
    {"SUM", TokenType::SUM},
    {"AVG", TokenType::AVG},
    {"MIN", TokenType::MIN},
    {"MAX", TokenType::MAX},
    {"DATE", TokenType::DATE_LITERAL},
};

Lexer::Lexer(const std::string& input) : input_(input) {}

char Lexer::CurrentChar() const {
    return IsAtEnd() ? '\0' : input_[pos_];
}

char Lexer::PeekChar() const {
    auto next = pos_ + 1;
    return (next < input_.size()) ? input_[next] : '\0';
}

void Lexer::Advance() {
    if (!IsAtEnd()) {
        if (input_[pos_] == '\n') {
            ++line_;
            column_ = 1;
        } else {
            ++column_;
        }
        ++pos_;
    }
}

bool Lexer::IsAtEnd() const {
    return pos_ >= input_.size();
}


void Lexer::SkipWhitespace() {
    while (!IsAtEnd()) {
        char c = CurrentChar();

        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            Advance();
        } else if (c == '-' && PeekChar() == '-') {
            while (!IsAtEnd() && CurrentChar() != '\n') {
                Advance();
            }
        } else {
            break;
        }
    }
}


//int literal or float literal
Token Lexer::ReadNumber() {
    size_t start_line = line_;
    size_t start_col = column_;
    std::string num;

    while (!IsAtEnd() && std::isdigit(static_cast<unsigned char>(CurrentChar()))) {
        num += CurrentChar();
        Advance();
    }

    if (!IsAtEnd() && CurrentChar() == '.' && std::isdigit(static_cast<unsigned char>(PeekChar()))) {
        num += '.';
        Advance();
        
        while (!IsAtEnd() && std::isdigit(static_cast<unsigned char>(CurrentChar()))) {
            num += CurrentChar();
            Advance();
        }
        return {TokenType::FLOAT_LITERAL, std::move(num), start_line, start_col};
    }

    return {TokenType::INTEGER_LITERAL, std::move(num), start_line, start_col};
}


//single quoted sql string, '' escpae for literal quote
Token Lexer::ReadString() {
    size_t start_line = line_;
    size_t start_col = column_;
    Advance();  

    std::string str;
    while (!IsAtEnd()) {
        char c = CurrentChar();

        if (c == '\'') {
            if (PeekChar() == '\'') {
                str += '\'';
                Advance();
                Advance();
            } else {
                Advance();  
                return {TokenType::STRING_LITERAL, std::move(str), start_line, start_col};
            }
        } else {
            str += c;
            Advance();
        }
    }

    throw ParseException("Unterminated string literal", start_line, start_col);
}


Token Lexer::ReadIdentifier() {
    size_t start_line = line_;
    size_t start_col = column_;
    std::string ident;

    while (!IsAtEnd() && (std::isalnum(static_cast<unsigned char>(CurrentChar())) || CurrentChar() == '_')) {
        ident += CurrentChar();
        Advance();
    }

    std::string upper = ident;
    std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char c) { return std::toupper(c); });

    auto it = kKeywords.find(upper);
    if (it != kKeywords.end()) {
        return {it->second, std::move(upper), start_line, start_col};
    }

    return {TokenType::IDENTIFIER, std::move(ident), start_line, start_col};
}


//one slot lookahead cache
Token Lexer::PeekToken() {
    if (!has_peeked_) {
        peeked_token_ = NextToken();
        has_peeked_ = true;
    }
    return peeked_token_;
}


Token Lexer::NextToken() {
    if (has_peeked_) {
        has_peeked_ = false;
        return std::move(peeked_token_);
    }

    SkipWhitespace();

    if (IsAtEnd()) {
        return {TokenType::END_OF_INPUT, "", line_, column_};
    }

    size_t start_line = line_;
    size_t start_col = column_;
    char c = CurrentChar();

    if (std::isdigit(static_cast<unsigned char>(c))) {
        return ReadNumber();
    }

    if (c == '\'') {
        return ReadString();
    }

    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
        return ReadIdentifier();
    }

    // Two character operators
    if (c == '!' && PeekChar() == '=') {
        Advance(); Advance();
        return {TokenType::NOT_EQUALS, "!=", start_line, start_col};
    }

    if (c == '<') {
        Advance();
        if (CurrentChar() == '=') {
            Advance();
            return {TokenType::LESS_EQUAL, "<=", start_line, start_col};
        }
        return {TokenType::LESS_THAN, "<", start_line, start_col};
    }
    
    if (c == '>') {
        Advance();
        if (CurrentChar() == '=') {
            Advance();
            return {TokenType::GREATER_EQUAL, ">=", start_line, start_col};
        }
        return {TokenType::GREATER_THAN, ">", start_line, start_col};
    }

    // Single-character tokens
    Advance();
    switch (c) {
        case '=': 
            return {TokenType::EQUALS, "=", start_line, start_col};
        case '+': 
            return {TokenType::PLUS, "+", start_line, start_col};
        case '-': 
            return {TokenType::MINUS, "-", start_line, start_col};
        case '*': 
            return {TokenType::STAR, "*", start_line, start_col};
        case '/': 
            return {TokenType::SLASH, "/", start_line, start_col};
        case ',':   
            return {TokenType::COMMA, ",", start_line, start_col};
        case '.': 
            return {TokenType::DOT, ".", start_line, start_col};
        case '(': 
            return {TokenType::LPAREN, "(", start_line, start_col};
        case ')': 
            return {TokenType::RPAREN, ")", start_line, start_col};
        case ';': 
            return {TokenType::SEMICOLON, ";", start_line, start_col};
        default: 
            break;
}

    return {TokenType::INVALID, std::string(1, c), start_line, start_col};
}

}  // namespace shilmandb
