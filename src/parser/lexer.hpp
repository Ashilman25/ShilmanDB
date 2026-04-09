#pragma once
#include "parser/token.hpp"
#include <string>

namespace shilmandb {

class Lexer {
public:
    explicit Lexer(const std::string& input);

    [[nodiscard]] Token NextToken();
    [[nodiscard]] Token PeekToken();

private:
    std::string input_;
    
    size_t pos_{0};
    size_t line_{1};
    size_t column_{1};

    bool has_peeked_{false};
    Token peeked_token_;

    void SkipWhitespace();
    void Advance();

    [[nodiscard]] Token ReadNumber();
    [[nodiscard]] Token ReadString();
    [[nodiscard]] Token ReadIdentifier();

    [[nodiscard]] char CurrentChar() const;
    [[nodiscard]] char PeekChar() const;

    [[nodiscard]] bool IsAtEnd() const;
};

}  // namespace shilmandb
