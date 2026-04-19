#include <gtest/gtest.h>
#include "parser/lexer.hpp"
#include "parser/parse_exception.hpp"
#include <vector>

namespace shilmandb {

// Helper: collect all tokens from input until END_OF_INPUT
static std::vector<Token> Tokenize(const std::string& input) {
    Lexer lexer(input);
    std::vector<Token> tokens;
    while (true) {
        auto tok = lexer.NextToken();
        tokens.push_back(tok);
        if (tok.type == TokenType::END_OF_INPUT) break;
    }
    return tokens;
}

// -----------------------------------------------------------------------
// Test 1: SimpleSelect
// -----------------------------------------------------------------------
TEST(LexerTest, SimpleSelect) {
    auto tokens = Tokenize("SELECT * FROM foo");
    ASSERT_EQ(tokens.size(), 5u);
    EXPECT_EQ(tokens[0].type, TokenType::SELECT);
    EXPECT_EQ(tokens[1].type, TokenType::STAR);
    EXPECT_EQ(tokens[2].type, TokenType::FROM);
    EXPECT_EQ(tokens[3].type, TokenType::IDENTIFIER);
    EXPECT_EQ(tokens[3].value, "foo");
    EXPECT_EQ(tokens[4].type, TokenType::END_OF_INPUT);
}

// -----------------------------------------------------------------------
// Test 2: IntegerAndFloat
// -----------------------------------------------------------------------
TEST(LexerTest, IntegerAndFloat) {
    auto tokens = Tokenize("42 3.14");
    ASSERT_EQ(tokens.size(), 3u);
    EXPECT_EQ(tokens[0].type, TokenType::INTEGER_LITERAL);
    EXPECT_EQ(tokens[0].value, "42");
    EXPECT_EQ(tokens[1].type, TokenType::FLOAT_LITERAL);
    EXPECT_EQ(tokens[1].value, "3.14");
    EXPECT_EQ(tokens[2].type, TokenType::END_OF_INPUT);
}

// -----------------------------------------------------------------------
// Test 3: StringLiteral
// -----------------------------------------------------------------------
TEST(LexerTest, StringLiteral) {
    auto tokens = Tokenize("'hello'");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0].type, TokenType::STRING_LITERAL);
    EXPECT_EQ(tokens[0].value, "hello");
    EXPECT_EQ(tokens[1].type, TokenType::END_OF_INPUT);
}

// -----------------------------------------------------------------------
// Test 4: EscapedQuote
// -----------------------------------------------------------------------
TEST(LexerTest, EscapedQuote) {
    auto tokens = Tokenize("'O''Brien'");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0].type, TokenType::STRING_LITERAL);
    EXPECT_EQ(tokens[0].value, "O'Brien");
    EXPECT_EQ(tokens[1].type, TokenType::END_OF_INPUT);
}

// -----------------------------------------------------------------------
// Test 5: Operators
// -----------------------------------------------------------------------
TEST(LexerTest, Operators) {
    auto tokens = Tokenize("= != < > <= >= + - * /");
    ASSERT_EQ(tokens.size(), 11u);  // 10 operators + END_OF_INPUT
    EXPECT_EQ(tokens[0].type, TokenType::EQUALS);
    EXPECT_EQ(tokens[1].type, TokenType::NOT_EQUALS);
    EXPECT_EQ(tokens[2].type, TokenType::LESS_THAN);
    EXPECT_EQ(tokens[3].type, TokenType::GREATER_THAN);
    EXPECT_EQ(tokens[4].type, TokenType::LESS_EQUAL);
    EXPECT_EQ(tokens[5].type, TokenType::GREATER_EQUAL);
    EXPECT_EQ(tokens[6].type, TokenType::PLUS);
    EXPECT_EQ(tokens[7].type, TokenType::MINUS);
    EXPECT_EQ(tokens[8].type, TokenType::STAR);
    EXPECT_EQ(tokens[9].type, TokenType::SLASH);
    EXPECT_EQ(tokens[10].type, TokenType::END_OF_INPUT);
}

// -----------------------------------------------------------------------
// Test 6: KeywordsCaseInsensitive
// -----------------------------------------------------------------------
TEST(LexerTest, KeywordsCaseInsensitive) {
    auto tokens = Tokenize("select FROM Where");
    ASSERT_EQ(tokens.size(), 4u);
    EXPECT_EQ(tokens[0].type, TokenType::SELECT);
    EXPECT_EQ(tokens[0].value, "SELECT");   // input was "select"
    EXPECT_EQ(tokens[1].type, TokenType::FROM);
    EXPECT_EQ(tokens[1].value, "FROM");
    EXPECT_EQ(tokens[2].type, TokenType::WHERE);
    EXPECT_EQ(tokens[2].value, "WHERE");    // input was "Where"
    EXPECT_EQ(tokens[3].type, TokenType::END_OF_INPUT);
}

// -----------------------------------------------------------------------
// Test 7: QualifiedColumn
// -----------------------------------------------------------------------
TEST(LexerTest, QualifiedColumn) {
    auto tokens = Tokenize("t.col");
    ASSERT_EQ(tokens.size(), 4u);
    EXPECT_EQ(tokens[0].type, TokenType::IDENTIFIER);
    EXPECT_EQ(tokens[0].value, "t");
    EXPECT_EQ(tokens[1].type, TokenType::DOT);
    EXPECT_EQ(tokens[2].type, TokenType::IDENTIFIER);
    EXPECT_EQ(tokens[2].value, "col");
    EXPECT_EQ(tokens[3].type, TokenType::END_OF_INPUT);
}

// -----------------------------------------------------------------------
// Test 8: AllTPCHTokens -- TPC-H Q1 tokenization
// -----------------------------------------------------------------------
TEST(LexerTest, AllTPCHTokens) {
    const std::string tpch_q1 = R"(
        SELECT
            l_returnflag,
            l_linestatus,
            SUM(l_quantity) AS sum_qty,
            SUM(l_extendedprice) AS sum_base_price,
            SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
            SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
            AVG(l_quantity) AS avg_qty,
            AVG(l_extendedprice) AS avg_price,
            AVG(l_discount) AS avg_disc,
            COUNT(*) AS count_order
        FROM
            lineitem
        WHERE
            l_shipdate <= DATE '1998-12-01'
        GROUP BY
            l_returnflag,
            l_linestatus
        ORDER BY
            l_returnflag,
            l_linestatus;
    )";

    auto tokens = Tokenize(tpch_q1);

    // Verify no INVALID tokens
    for (size_t i = 0; i < tokens.size(); ++i) {
        EXPECT_NE(tokens[i].type, TokenType::INVALID)
            << "INVALID token at index " << i
            << ": value='" << tokens[i].value << "'"
            << " line=" << tokens[i].line
            << " col=" << tokens[i].column;
    }

    // Verify stream ends with END_OF_INPUT
    ASSERT_FALSE(tokens.empty());
    EXPECT_EQ(tokens.back().type, TokenType::END_OF_INPUT);

    // Verify we got a reasonable number of tokens (Q1 has ~70+ tokens)
    EXPECT_GT(tokens.size(), 60u);

    // Spot-check key tokens
    EXPECT_EQ(tokens[0].type, TokenType::SELECT);
    EXPECT_EQ(tokens[1].type, TokenType::IDENTIFIER);
    EXPECT_EQ(tokens[1].value, "l_returnflag");
}

// -----------------------------------------------------------------------
// Bonus: PeekToken behavior
// -----------------------------------------------------------------------
TEST(LexerTest, PeekTokenDoesNotConsume) {
    Lexer lexer("SELECT *");
    Token peeked = lexer.PeekToken();
    EXPECT_EQ(peeked.type, TokenType::SELECT);

    // PeekToken again returns the same token
    Token peeked2 = lexer.PeekToken();
    EXPECT_EQ(peeked2.type, TokenType::SELECT);

    // NextToken consumes it
    Token next = lexer.NextToken();
    EXPECT_EQ(next.type, TokenType::SELECT);

    // Now the next token should be STAR
    Token star = lexer.NextToken();
    EXPECT_EQ(star.type, TokenType::STAR);
}

// -----------------------------------------------------------------------
// Bonus: Unterminated string throws ParseException
// -----------------------------------------------------------------------
TEST(LexerTest, UnterminatedStringThrows) {
    Lexer lexer("'hello");
    EXPECT_THROW((void)lexer.NextToken(), ParseException);
}

// -----------------------------------------------------------------------
// Bonus: SQL comments are skipped
// -----------------------------------------------------------------------
TEST(LexerTest, CommentsSkipped) {
    auto tokens = Tokenize("SELECT -- this is a comment\n*");
    ASSERT_EQ(tokens.size(), 3u);
    EXPECT_EQ(tokens[0].type, TokenType::SELECT);
    EXPECT_EQ(tokens[1].type, TokenType::STAR);
    EXPECT_EQ(tokens[1].line, 2u);    // * is on line 2 after the \n
    EXPECT_EQ(tokens[1].column, 1u);  // first column of line 2
    EXPECT_EQ(tokens[2].type, TokenType::END_OF_INPUT);
}

// -----------------------------------------------------------------------
// Test: EmptyInput
// -----------------------------------------------------------------------
TEST(LexerTest, EmptyInput) {
    auto tokens = Tokenize("");
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_EQ(tokens[0].type, TokenType::END_OF_INPUT);
}

// -----------------------------------------------------------------------
// Test: PunctuationTokens
// -----------------------------------------------------------------------
TEST(LexerTest, PunctuationTokens) {
    auto tokens = Tokenize("(a, b);");
    ASSERT_EQ(tokens.size(), 7u);
    EXPECT_EQ(tokens[0].type, TokenType::LPAREN);
    EXPECT_EQ(tokens[1].type, TokenType::IDENTIFIER);
    EXPECT_EQ(tokens[1].value, "a");
    EXPECT_EQ(tokens[2].type, TokenType::COMMA);
    EXPECT_EQ(tokens[3].type, TokenType::IDENTIFIER);
    EXPECT_EQ(tokens[3].value, "b");
    EXPECT_EQ(tokens[4].type, TokenType::RPAREN);
    EXPECT_EQ(tokens[5].type, TokenType::SEMICOLON);
    EXPECT_EQ(tokens[6].type, TokenType::END_OF_INPUT);
}

// -----------------------------------------------------------------------
// CaseKeywords: CASE / WHEN / THEN / ELSE / END recognised as keywords
// (case-insensitive)
// -----------------------------------------------------------------------
TEST(LexerTest, CaseKeywords) {
    auto tokens = Tokenize("CASE WHEN THEN ELSE END case when then else end");
    ASSERT_EQ(tokens.size(), 11u);
    EXPECT_EQ(tokens[0].type, TokenType::CASE);
    EXPECT_EQ(tokens[1].type, TokenType::WHEN);
    EXPECT_EQ(tokens[2].type, TokenType::THEN);
    EXPECT_EQ(tokens[3].type, TokenType::ELSE);
    EXPECT_EQ(tokens[4].type, TokenType::END);
    EXPECT_EQ(tokens[5].type, TokenType::CASE);
    EXPECT_EQ(tokens[6].type, TokenType::WHEN);
    EXPECT_EQ(tokens[7].type, TokenType::THEN);
    EXPECT_EQ(tokens[8].type, TokenType::ELSE);
    EXPECT_EQ(tokens[9].type, TokenType::END);
    EXPECT_EQ(tokens[10].type, TokenType::END_OF_INPUT);
}

}  // namespace shilmandb
