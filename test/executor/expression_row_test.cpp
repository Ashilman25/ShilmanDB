#include <gtest/gtest.h>
#include "catalog/schema.hpp"
#include "executor/data_chunk.hpp"
#include "executor/expression_evaluator.hpp"
#include "planner/expression_binder.hpp"
#include "types/tuple.hpp"
#include "types/value.hpp"

#include <memory>
#include <string>
#include <vector>

namespace shilmandb {

namespace {

using ExprPtr = std::unique_ptr<Expression>;

Schema MakeSchema() {
    return Schema({
        Column("i", TypeId::INTEGER),
        Column("b", TypeId::BIGINT),
        Column("x", TypeId::DECIMAL),
        Column("s", TypeId::VARCHAR),
        Column("d", TypeId::DATE),
    });
}

ExprPtr Col(const std::string& name) {
    auto c = std::make_unique<ColumnRef>();
    c->column_name = name;
    return c;
}

ExprPtr Lit(Value v) {
    auto l = std::make_unique<Literal>();
    l->value = std::move(v);
    return l;
}

ExprPtr Bin(BinaryOp::Op op, ExprPtr l, ExprPtr r) {
    auto b = std::make_unique<BinaryOp>();
    b->op = op;
    b->left = std::move(l);
    b->right = std::move(r);
    return b;
}

ExprPtr Un(UnaryOp::Op op, ExprPtr e) {
    auto u = std::make_unique<UnaryOp>();
    u->op = op;
    u->operand = std::move(e);
    return u;
}

ExprPtr Str(const std::string& s) { return Lit(Value(s)); }
ExprPtr Int(int32_t v) { return Lit(Value(v)); }

class ExpressionRowTest : public ::testing::Test {
protected:
    ExpressionRowTest() : schema_(MakeSchema()), chunk_(schema_) {
        const char* strs[] = {"PROMO x", "", "abc"};
        const char* dates[] = {"1995-03-15", "1994-01-01", "1996-12-31"};
        for (int r = 0; r < 3; ++r) {
            std::vector<Value> vals;
            vals.emplace_back(static_cast<int32_t>(r + 1));
            vals.emplace_back(static_cast<int64_t>(r * 10 + 5));
            vals.emplace_back(0.25 * (r + 1));
            vals.emplace_back(std::string(strs[r]));
            vals.push_back(Value::FromString(TypeId::DATE, dates[r]));
            tuples_.emplace_back(std::move(vals), schema_);
            chunk_.AppendRow(tuples_.back().GetData(), tuples_.back().GetLength());
        }
        chunk_.EnsureDecoded({0, 1, 2, 3, 4});
    }

    void Bind(Expression* e) { BindExpression(e, schema_); }

    void ExpectSame(const Expression* e) {
        for (size_t r = 0; r < tuples_.size(); ++r) {
            Value a = EvaluateExpression(e, tuples_[r], schema_);
            Value b = EvaluateExpression(e, ChunkRow{chunk_, r});
            EXPECT_EQ(a.type_, b.type_) << "row " << r;
            EXPECT_EQ(a.ToString(), b.ToString()) << "row " << r;
        }
    }

    void BindAndCheck(ExprPtr e) {
        Bind(e.get());
        ExpectSame(e.get());
    }

    Schema schema_;
    DataChunk chunk_;
    std::vector<Tuple> tuples_;
};

}  // namespace

TEST_F(ExpressionRowTest, ColumnAndLiteral) {
    for (const char* name : {"i", "b", "x", "s", "d"}) BindAndCheck(Col(name));
    BindAndCheck(Int(42));
}

TEST_F(ExpressionRowTest, ComparisonsWithPromotion) {
    BindAndCheck(Bin(BinaryOp::Op::LT, Col("i"), Col("b")));
    BindAndCheck(Bin(BinaryOp::Op::GTE, Col("x"), Col("i")));
    BindAndCheck(Bin(BinaryOp::Op::EQ, Col("d"), Str("1995-03-15")));
    BindAndCheck(Bin(BinaryOp::Op::NEQ, Col("s"), Str("abc")));
    // unbound: the date column is compared with a date string
    auto unbound = Bin(BinaryOp::Op::GT, Col("d"), Str("1995-01-01"));
    ExpectSame(unbound.get());
}

TEST_F(ExpressionRowTest, Arithmetic) {
    BindAndCheck(Bin(BinaryOp::Op::MUL, Col("x"),
                     Bin(BinaryOp::Op::SUB, Int(1), Col("x"))));
    BindAndCheck(Bin(BinaryOp::Op::ADD, Col("i"), Col("b")));
    BindAndCheck(Bin(BinaryOp::Op::DIV, Col("x"), Col("i")));
    BindAndCheck(Bin(BinaryOp::Op::SUB, Col("i"), Int(3)));
}

TEST_F(ExpressionRowTest, LikeAndLogic) {
    BindAndCheck(Bin(BinaryOp::Op::LIKE, Col("s"), Str("PRO%")));
    BindAndCheck(Bin(BinaryOp::Op::OR,
                     Bin(BinaryOp::Op::AND,
                         Bin(BinaryOp::Op::GT, Col("i"), Int(0)),
                         Bin(BinaryOp::Op::LIKE, Col("s"), Str("%x"))),
                     Bin(BinaryOp::Op::LT, Col("x"), Lit(Value(1.0)))));
    BindAndCheck(Un(UnaryOp::Op::NOT, Bin(BinaryOp::Op::EQ, Col("i"), Int(2))));
}

TEST_F(ExpressionRowTest, NegateAndCase) {
    BindAndCheck(Un(UnaryOp::Op::NEGATE, Col("i")));
    BindAndCheck(Un(UnaryOp::Op::NEGATE, Col("x")));

    auto with_else = std::make_unique<CaseExpression>();
    with_else->when_clauses.emplace_back(Bin(BinaryOp::Op::EQ, Col("i"), Int(1)), Col("x"));
    with_else->else_clause = Lit(Value(0.0));
    BindAndCheck(std::move(with_else));

    auto no_else = std::make_unique<CaseExpression>();
    no_else->when_clauses.emplace_back(Bin(BinaryOp::Op::EQ, Col("i"), Int(99)), Int(1));
    BindAndCheck(std::move(no_else));
}

TEST_F(ExpressionRowTest, UnboundNameLooksUpByName) {
    auto e = Col("s");
    ExpectSame(e.get());
}

TEST_F(ExpressionRowTest, MissingNameThrowsInBothModes) {
    auto e = Col("zzz");
    EXPECT_THROW((void)EvaluateExpression(e.get(), tuples_[0], schema_), DatabaseException);
    EXPECT_THROW((void)EvaluateExpression(e.get(), ChunkRow{chunk_, 0}), DatabaseException);
}

TEST_F(ExpressionRowTest, UndecodedChunkMatchesTuple) {
    DataChunk raw(schema_);
    for (const Tuple& t : tuples_) raw.AppendRow(t.GetData(), t.GetLength());

    std::vector<ExprPtr> exprs;
    for (const char* name : {"i", "b", "x", "s", "d"}) exprs.push_back(Col(name));
    exprs.push_back(Bin(BinaryOp::Op::LT, Col("i"), Col("b")));
    exprs.push_back(Bin(BinaryOp::Op::EQ, Col("d"), Str("1995-03-15")));
    exprs.push_back(Bin(BinaryOp::Op::LIKE, Col("s"), Str("PRO%")));
    exprs.push_back(Bin(BinaryOp::Op::OR,
                        Bin(BinaryOp::Op::AND, Bin(BinaryOp::Op::GT, Col("i"), Int(0)),
                            Bin(BinaryOp::Op::LIKE, Col("s"), Str("%x"))),
                        Bin(BinaryOp::Op::LT, Col("x"), Lit(Value(1.0)))));
    exprs.push_back(Bin(BinaryOp::Op::MUL, Col("x"), Bin(BinaryOp::Op::SUB, Int(1), Col("x"))));
    auto cs = std::make_unique<CaseExpression>();
    cs->when_clauses.emplace_back(Bin(BinaryOp::Op::EQ, Col("i"), Int(1)), Col("x"));
    cs->else_clause = Lit(Value(0.0));
    exprs.push_back(std::move(cs));

    for (auto& e : exprs) {
        Bind(e.get());
        for (size_t r = 0; r < tuples_.size(); ++r) {
            Value a = EvaluateExpression(e.get(), tuples_[r], schema_);
            Value b = EvaluateExpression(e.get(), ChunkRow{raw, r});
            EXPECT_EQ(a.type_, b.type_) << "row " << r;
            EXPECT_EQ(a.ToString(), b.ToString()) << "row " << r;
        }
    }
    for (size_t c = 0; c < 5; ++c) EXPECT_FALSE(raw.IsDecoded(c)) << "column " << c;
}

}  // namespace shilmandb
