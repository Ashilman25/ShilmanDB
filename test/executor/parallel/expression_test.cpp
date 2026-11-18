#include <gtest/gtest.h>
#include "catalog/schema.hpp"
#include "common/exception.hpp"
#include "executor/expression_evaluator.hpp"
#include "executor/parallel/batch.hpp"
#include "executor/parallel/decimal.hpp"
#include "executor/parallel/expression.hpp"
#include "executor/parallel/vector.hpp"
#include "parser/ast.hpp"
#include "types/column_type.hpp"
#include "types/tuple.hpp"
#include "types/value.hpp"
#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace shilmandb::parallel {

namespace {

using ExprPtr = std::unique_ptr<Expression>;
using Op = BinaryOp::Op;

// A column read by position, as the planner's binder leaves it.
ExprPtr Col(const std::string& name, uint32_t index) {
    auto c = std::make_unique<ColumnRef>();
    c->column_name = name;
    c->bound_index = index;
    return c;
}

// A literal holding v.
ExprPtr Lit(Value v) {
    auto l = std::make_unique<Literal>();
    l->value = std::move(v);
    return l;
}

// A decimal literal read the way the parser reads one.
ExprPtr Dec(const std::string& text) {
    return Lit(Value(std::stod(text)));
}

ExprPtr Int(int32_t v) { return Lit(Value(v)); }
ExprPtr Str(const std::string& s) { return Lit(Value(s)); }

ExprPtr Bin(Op op, ExprPtr l, ExprPtr r) {
    auto b = std::make_unique<BinaryOp>();
    b->op = op;
    b->left = std::move(l);
    b->right = std::move(r);
    return b;
}

ExprPtr Un(UnaryOp::Op op, ExprPtr x) {
    auto u = std::make_unique<UnaryOp>();
    u->op = op;
    u->operand = std::move(x);
    return u;
}

// `lhs IN (values...)` as the parser builds it: a right-nested OR of equalities.
ExprPtr In(const Expression& lhs, std::vector<ExprPtr> values) {
    ExprPtr result = Bin(Op::EQ, lhs.Clone(), std::move(values.back()));
    values.pop_back();
    while (!values.empty()) {
        ExprPtr next = Bin(Op::EQ, lhs.Clone(), std::move(values.back()));
        values.pop_back();
        result = Bin(Op::OR, std::move(next), std::move(result));
    }
    return result;
}

// Clone() drops bound_index, so IN's clones get it back here.
void Rebind(Expression& e, uint32_t index) {
    if (e.type == ExprType::COLUMN_REF) {
        static_cast<ColumnRef&>(e).bound_index = index;
    } else if (e.type == ExprType::BINARY_OP) {
        auto& b = static_cast<BinaryOp&>(e);
        Rebind(*b.left, index);
        Rebind(*b.right, index);
    }
}

// Columns of a hand-built batch plus the layout that maps them.
struct TestBatch {
    InputLayout layout;
    Batch batch;
    std::deque<std::string> strings;                 // bytes the string vectors point at
    std::deque<std::vector<StringRef>> dictionaries;  // entries the dictionary vectors read

    // Adds a column and returns its plan index (also its slot).
    uint32_t Add(const ColumnType& type) {
        const auto index = static_cast<uint32_t>(layout.slot_types.size());
        layout.slot_of_column.push_back(static_cast<int>(index));
        layout.slot_types.push_back(VTypeOf(type));
        layout.slot_precision.push_back(type.id == ColumnTypeId::NUMERIC ? type.precision : 0);
        batch.slots.emplace_back(VTypeOf(type));
        return index;
    }

    uint32_t AddInts(const ColumnType& type, const std::vector<int64_t>& values) {
        const uint32_t c = Add(type);
        for (size_t i = 0; i < values.size(); ++i) batch.slots[c].I64()[i] = values[i];
        return c;
    }

    uint32_t AddDecimals(int precision, int scale, const std::vector<__int128>& raw) {
        const uint32_t c = Add(ColumnType::Numeric(precision, scale));
        Vector& v = batch.slots[c];
        for (size_t i = 0; i < raw.size(); ++i) {
            if (v.Type().wide) {
                v.I128()[i] = raw[i];
            } else {
                v.I64()[i] = static_cast<int64_t>(raw[i]);
            }
        }
        return c;
    }

    uint32_t AddDoubles(const std::vector<double>& values) {
        const uint32_t c = Add(ColumnType::Double());
        for (size_t i = 0; i < values.size(); ++i) batch.slots[c].F64()[i] = values[i];
        return c;
    }

    // Plain strings, or a dictionary of the distinct values with one code per row.
    uint32_t AddStrings(const std::vector<std::string>& values, bool dictionary = false) {
        const uint32_t c = Add(ColumnType::Varchar(100));
        Vector& v = batch.slots[c];
        if (!dictionary) {
            for (size_t i = 0; i < values.size(); ++i) {
                strings.push_back(values[i]);
                v.Str()[i] = StringRef{strings.back().data(), static_cast<uint32_t>(strings.back().size())};
            }
            return c;
        }
        dictionaries.emplace_back();
        std::vector<StringRef>& entries = dictionaries.back();
        std::vector<std::string> seen;
        for (size_t i = 0; i < values.size(); ++i) {
            size_t code = 0;
            while (code < seen.size() && seen[code] != values[i]) ++code;
            if (code == seen.size()) {
                seen.push_back(values[i]);
                strings.push_back(values[i]);
                entries.push_back(StringRef{strings.back().data(), static_cast<uint32_t>(strings.back().size())});
            }
            v.Codes()[i] = static_cast<uint32_t>(code);
        }
        v.SetDictionary(&entries);
        return c;
    }

    // Sets the row count; every row starts active. Dictionaries stay.
    void Finish(uint32_t rows) {
        batch.rows = rows;
        batch.sel.SetAll(rows);
    }
};

std::unique_ptr<CompiledExpr> Compile(const Expression& e, const TestBatch& t) {
    return CompiledExpr::Compile(e, t.layout);
}

// The rows Select keeps, starting from every row.
std::vector<uint16_t> Selected(const CompiledExpr& e, const TestBatch& t) {
    EvalScratch scratch;
    Selection sel;
    sel.SetAll(t.batch.rows);
    e.Select(t.batch, sel, scratch);
    return std::vector<uint16_t>(sel.Data(), sel.Data() + sel.Size());
}

// The rows Select keeps, starting from `rows`.
std::vector<uint16_t> SelectedFrom(const CompiledExpr& e, const TestBatch& t, const std::vector<uint16_t>& rows) {
    EvalScratch scratch;
    Selection sel;
    for (size_t i = 0; i < rows.size(); ++i) sel.Data()[i] = rows[i];
    sel.Resize(static_cast<uint32_t>(rows.size()));
    e.Select(t.batch, sel, scratch);
    return std::vector<uint16_t>(sel.Data(), sel.Data() + sel.Size());
}

// Every row's value, in a fresh vector of the expression's type.
Vector EvaluateAll(const CompiledExpr& e, const TestBatch& t) {
    EvalScratch scratch;
    Vector out(e.Type());
    Selection rows;
    rows.SetAll(t.batch.rows);
    e.Evaluate(t.batch, rows, out, scratch);
    return out;
}

// One row's exact decimal, from whichever array the vector uses.
__int128 DecimalAt(const Vector& v, uint32_t row) {
    return v.Type().wide ? v.I128()[row] : static_cast<__int128>(v.I64()[row]);
}

// Runs fn and returns the DatabaseException text, or "" when nothing is thrown.
template <typename Fn>
std::string ErrorOf(Fn fn) {
    try {
        fn();
    } catch (const DatabaseException& e) {
        return e.what();
    }
    return "";
}

std::vector<uint16_t> Rows(std::initializer_list<uint16_t> r) { return std::vector<uint16_t>(r); }

}  // namespace

TEST(ParallelExpressionTest, CompareDecimalWithLiteral) {
    TestBatch t;
    const uint32_t disc = t.AddDecimals(15, 2, {3, 4, 5, 6, 7, 5});
    t.Finish(6);

    auto expr = Bin(Op::GTE, Col("l_discount", disc), Dec("0.05"));
    auto compiled = Compile(*expr, t);
    EXPECT_EQ(Selected(*compiled, t), Rows({2, 3, 4, 5}));

    const Vector out = EvaluateAll(*compiled, t);
    EXPECT_EQ(compiled->Type().kind, Kind::INTEGER);
    EXPECT_EQ(compiled->Type().result, TypeId::INTEGER);
    const std::vector<int64_t> expected = {0, 0, 1, 1, 1, 1};
    for (uint32_t i = 0; i < 6; ++i) EXPECT_EQ(out.I64()[i], expected[i]) << "row " << i;
}

TEST(ParallelExpressionTest, LiteralWithMoreDecimalsThanColumn) {
    TestBatch t;
    const uint32_t disc = t.AddDecimals(15, 2, {3, 4, 5, 6, 7, 5});
    t.Finish(6);
    auto ge = Bin(Op::GTE, Col("l_discount", disc), Dec("0.055"));
    EXPECT_EQ(Selected(*Compile(*ge, t), t), Rows({3, 4}));

    // -0.055 arrives as NEGATE(0.055) and folds to exactly -55 at scale 3.
    TestBatch neg;
    const uint32_t d2 = neg.AddDecimals(15, 2, {-7, -6, -5, -4, 5, 6});
    neg.Finish(6);
    auto gt = Bin(Op::GT, Col("l_discount", d2), Un(UnaryOp::Op::NEGATE, Dec("0.055")));
    EXPECT_EQ(Selected(*Compile(*gt, neg), neg), Rows({2, 3, 4, 5}));
    auto le = Bin(Op::LTE, Col("l_discount", d2), Un(UnaryOp::Op::NEGATE, Dec("0.055")));
    EXPECT_EQ(Selected(*Compile(*le, neg), neg), Rows({0, 1}));
}

TEST(ParallelExpressionTest, MixedScaleAddAndCompare) {
    TestBatch t;
    const uint32_t a = t.AddDecimals(15, 2, {12345, -100, 0, 99999, -5, 1});
    const uint32_t b = t.AddDecimals(12, 4, {12345, 1000000, -1, -99999999, 50000, 0});
    t.Finish(6);

    auto sum = Bin(Op::ADD, Col("a", a), Col("b", b));
    auto compiled = Compile(*sum, t);
    EXPECT_EQ(compiled->Type().kind, Kind::DECIMAL);
    EXPECT_EQ(compiled->Type().scale, 4);
    EXPECT_EQ(compiled->Type().result, TypeId::DECIMAL);
    const Vector out = EvaluateAll(*compiled, t);
    const std::vector<int64_t> expected = {1246845, 990000, -1, -90000099, 49500, 100};
    for (uint32_t i = 0; i < 6; ++i) EXPECT_TRUE(DecimalAt(out, i) == expected[i]) << "row " << i;

    auto gt = Bin(Op::GT, Col("a", a), Col("b", b));
    EXPECT_EQ(Selected(*Compile(*gt, t), t), Rows({0, 2, 3, 5}));

    auto eq99 = Bin(Op::EQ, Bin(Op::ADD, Col("a", a), Col("b", b)), Dec("99.0"));
    EXPECT_EQ(Selected(*Compile(*eq99, t), t), Rows({1}));

    auto eq_neg = Bin(Op::EQ, Bin(Op::ADD, Col("a", a), Col("b", b)), Un(UnaryOp::Op::NEGATE, Dec("9000.0099")));
    EXPECT_EQ(Selected(*Compile(*eq_neg, t), t), Rows({3}));

    auto sub = Bin(Op::SUB, Col("b", b), Col("a", a));
    const Vector diff = EvaluateAll(*Compile(*sub, t), t);
    const std::vector<int64_t> expected_diff = {12345 - 1234500, 1000000 + 10000, -1, -99999999 - 9999900, 50500, -100};
    for (uint32_t i = 0; i < 6; ++i) EXPECT_TRUE(DecimalAt(diff, i) == expected_diff[i]) << "row " << i;
}

TEST(ParallelExpressionTest, DoubleMeetsDecimal) {
    TestBatch t;
    const uint32_t d = t.AddDoubles({1.5, -2.25, 0.1, 3.0, 0.0, 1e10});
    const uint32_t n = t.AddDecimals(15, 2, {200, 1, 333, -7, 5, 10});
    t.Finish(6);

    auto product = Bin(Op::MUL, Col("d", d), Col("n", n));
    auto compiled = Compile(*product, t);
    EXPECT_EQ(compiled->Type().kind, Kind::DOUBLE);
    EXPECT_EQ(compiled->Type().result, TypeId::DECIMAL);
    const Vector out = EvaluateAll(*compiled, t);
    const std::vector<double> ds = {1.5, -2.25, 0.1, 3.0, 0.0, 1e10};
    const std::vector<int64_t> ns = {200, 1, 333, -7, 5, 10};
    for (uint32_t i = 0; i < 6; ++i) {
        EXPECT_EQ(out.F64()[i], ds[i] * DecimalToDouble(ns[i], 2)) << "row " << i;
    }

    // A decimal literal meeting a double is its original double again.
    auto plus = Bin(Op::ADD, Col("d", d), Dec("0.1"));
    const Vector sums = EvaluateAll(*Compile(*plus, t), t);
    for (uint32_t i = 0; i < 6; ++i) EXPECT_EQ(sums.F64()[i], ds[i] + 0.1) << "row " << i;

    auto lt = Bin(Op::LT, Col("d", d), Col("n", n));
    EXPECT_EQ(Selected(*Compile(*lt, t), t), Rows({0, 1, 2, 4}));
}

TEST(ParallelExpressionTest, RevenueIsExact) {
    TestBatch t;
    const std::vector<__int128> price = {1234567, 9999999999999, 100, 1, 4590000, 3333};
    const std::vector<__int128> disc = {5, 10, 0, 100, 7, 3};
    const uint32_t p = t.AddDecimals(15, 2, price);
    const uint32_t d = t.AddDecimals(15, 2, disc);
    t.Finish(6);

    auto revenue = Bin(Op::MUL, Col("l_extendedprice", p), Bin(Op::SUB, Int(1), Col("l_discount", d)));
    auto compiled = Compile(*revenue, t);
    EXPECT_EQ(compiled->Type().kind, Kind::DECIMAL);
    EXPECT_EQ(compiled->Type().scale, 4);
    const Vector out = EvaluateAll(*compiled, t);
    for (uint32_t i = 0; i < 6; ++i) {
        EXPECT_TRUE(DecimalAt(out, i) == price[i] * (100 - disc[i])) << "row " << i;
    }
}

TEST(ParallelExpressionTest, WideResultsStayExact) {
    TestBatch t;
    const __int128 nines = 999999999999999999;  // 18 digits, a full NUMERIC(18,0)
    const uint32_t big = t.AddDecimals(18, 0, {nines, -nines, 1, 0, 123, -5});
    const uint32_t wide = t.AddDecimals(38, 0, {nines * nines, 1, 2, 3, 4, 5});
    t.Finish(6);

    // Past INT64_MAX: held in 128 bits, still exact.
    auto times = Bin(Op::MUL, Col("big", big), Int(100));
    auto compiled = Compile(*times, t);
    EXPECT_EQ(compiled->Type().kind, Kind::DECIMAL);
    EXPECT_TRUE(compiled->Type().wide);
    const Vector out = EvaluateAll(*compiled, t);
    EXPECT_TRUE(DecimalAt(out, 0) == nines * 100);
    EXPECT_TRUE(DecimalAt(out, 1) == -nines * 100);

    auto plus = Bin(Op::ADD, Col("big", big), Col("big", big));
    const Vector sums = EvaluateAll(*Compile(*plus, t), t);
    EXPECT_TRUE(DecimalAt(sums, 0) == nines * 2);

    // Past 38 digits throws, only for the rows evaluated.
    auto square = Bin(Op::MUL, Col("wide", wide), Col("wide", wide));
    auto sq = Compile(*square, t);
    EXPECT_EQ(ErrorOf([&] { (void)EvaluateAll(*sq, t); }), "NUMERIC result is too wide");
    EvalScratch scratch;
    Vector small(sq->Type());
    Selection rest;
    for (uint16_t i = 1; i < 6; ++i) rest.Data()[i - 1] = i;
    rest.Resize(5);
    sq->Evaluate(t.batch, rest, small, scratch);
    EXPECT_TRUE(DecimalAt(small, 5) == 25);
}

TEST(ParallelExpressionTest, DivisionRules) {
    TestBatch t;
    const uint32_t x = t.AddInts(ColumnType::Integer(), {7, -7, 7, 0, 1, -1});
    const uint32_t y = t.AddInts(ColumnType::Integer(), {2, 2, -2, 5, 0, 3});
    const uint32_t n = t.AddDecimals(15, 2, {150, -150, 0, 1, -1, 99});
    const uint32_t d = t.AddDoubles({0.0, -0.0, 1.0, 2.0, 0.0, -0.0});
    const uint32_t b = t.AddInts(ColumnType::BigInt(), {-9, 9, 10, 11, 12, 13});
    t.Finish(6);

    auto int_div = Bin(Op::DIV, Col("x", x), Col("y", y));
    auto compiled = Compile(*int_div, t);
    EXPECT_EQ(compiled->Type().kind, Kind::INTEGER);
    EXPECT_EQ(compiled->Type().result, TypeId::INTEGER);
    EXPECT_EQ(ErrorOf([&] { (void)EvaluateAll(*compiled, t); }), "Division by zero");

    EvalScratch scratch;
    Vector out(compiled->Type());
    Selection nonzero;
    const std::vector<uint16_t> rows = {0, 1, 2, 3, 5};
    for (size_t i = 0; i < rows.size(); ++i) nonzero.Data()[i] = rows[i];
    nonzero.Resize(5);
    compiled->Evaluate(t.batch, nonzero, out, scratch);
    EXPECT_EQ(out.I64()[0], 3);
    EXPECT_EQ(out.I64()[1], -3);
    EXPECT_EQ(out.I64()[2], -3);
    EXPECT_EQ(out.I64()[3], 0);
    EXPECT_EQ(out.I64()[5], 0);

    auto by_zero = Bin(Op::DIV, Col("x", x), Int(0));
    EXPECT_EQ(ErrorOf([&] { (void)EvaluateAll(*Compile(*by_zero, t), t); }), "Division by zero");

    auto big_div = Bin(Op::DIV, Col("b", b), Int(4));
    auto big_compiled = Compile(*big_div, t);
    EXPECT_EQ(big_compiled->Type().result, TypeId::BIGINT);
    EXPECT_EQ(EvaluateAll(*big_compiled, t).I64()[0], -2);

    // A decimal or double side divides as doubles; any zero divisor gives +infinity.
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<ExprPtr> zero_divisors;
    zero_divisors.push_back(Int(0));
    zero_divisors.push_back(Dec("0.0"));
    zero_divisors.push_back(Un(UnaryOp::Op::NEGATE, Dec("0.0")));
    zero_divisors.push_back(Col("d", d));
    for (auto& divisor : zero_divisors) {
        auto dec_div = Bin(Op::DIV, Col("n", n), divisor->Clone());
        if (divisor->type == ExprType::COLUMN_REF) Rebind(*static_cast<BinaryOp&>(*dec_div).right, d);
        auto c = Compile(*dec_div, t);
        EXPECT_EQ(c->Type().kind, Kind::DOUBLE);
        EXPECT_EQ(c->Type().result, TypeId::DECIMAL);
        const Vector q = EvaluateAll(*c, t);
        const bool column = divisor->type == ExprType::COLUMN_REF;
        for (uint32_t i : {0u, 1u, 4u, 5u}) EXPECT_EQ(q.F64()[i], inf) << "row " << i;
        if (!column) {
            EXPECT_EQ(q.F64()[2], inf);  // 0 / 0 is +infinity too, as today
        }
    }

    auto dec_by_int = Bin(Op::DIV, Col("n", n), Col("y", y));
    const Vector q = EvaluateAll(*Compile(*dec_by_int, t), t);
    EXPECT_EQ(q.F64()[0], 1.5 / 2.0);
    EXPECT_EQ(q.F64()[4], inf);  // -0.01 / 0 is +infinity, not -infinity
}

TEST(ParallelExpressionTest, ShortCircuitSkipsErrors) {
    TestBatch t;
    const uint32_t x = t.AddInts(ColumnType::Integer(), {0, 5, 20, 0, 2, -10});
    t.Finish(6);

    auto and_expr = Bin(Op::AND, Bin(Op::NEQ, Col("x", x), Int(0)),
                        Bin(Op::GT, Bin(Op::DIV, Int(10), Col("x", x)), Int(1)));
    auto and_c = Compile(*and_expr, t);
    EXPECT_EQ(Selected(*and_c, t), Rows({1, 4}));
    const Vector and_v = EvaluateAll(*and_c, t);
    const std::vector<int64_t> and_expected = {0, 1, 0, 0, 1, 0};
    for (uint32_t i = 0; i < 6; ++i) EXPECT_EQ(and_v.I64()[i], and_expected[i]) << "row " << i;

    auto or_expr = Bin(Op::OR, Bin(Op::EQ, Col("x", x), Int(0)),
                       Bin(Op::GT, Bin(Op::DIV, Int(10), Col("x", x)), Int(1)));
    auto or_c = Compile(*or_expr, t);
    EXPECT_EQ(Selected(*or_c, t), Rows({0, 1, 3, 4}));
    const Vector or_v = EvaluateAll(*or_c, t);
    const std::vector<int64_t> or_expected = {1, 1, 0, 1, 1, 0};
    for (uint32_t i = 0; i < 6; ++i) EXPECT_EQ(or_v.I64()[i], or_expected[i]) << "row " << i;

    // From a partial selection the OR keeps only rows it was given.
    EXPECT_EQ(SelectedFrom(*or_c, t, {1, 2, 3}), Rows({1, 3}));

    // Without the guard, the zero rows throw.
    auto unguarded = Bin(Op::GT, Bin(Op::DIV, Int(10), Col("x", x)), Int(1));
    EXPECT_EQ(ErrorOf([&] { (void)Selected(*Compile(*unguarded, t), t); }), "Division by zero");
}

TEST(ParallelExpressionTest, InListIsOneProbe) {
    for (const bool dictionary : {false, true}) {
        TestBatch t;
        const uint32_t mode = t.AddStrings({"MAIL", "SHIP", "AIR", "TRUCK", "MAIL", "RAIL"}, dictionary);
        t.Finish(6);

        auto col = Col("l_shipmode", mode);
        std::vector<ExprPtr> values;
        values.push_back(Str("MAIL"));
        values.push_back(Str("SHIP"));
        auto in = In(*col, std::move(values));
        Rebind(*in, mode);
        const std::vector<uint16_t> probe = Selected(*Compile(*in, t), t);

        // The same terms evaluated one at a time, then joined.
        std::vector<bool> any(6, false);
        for (const char* v : {"MAIL", "SHIP"}) {
            auto eq = Bin(Op::EQ, Col("l_shipmode", mode), Str(v));
            for (uint16_t r : Selected(*Compile(*eq, t), t)) any[r] = true;
        }
        std::vector<uint16_t> by_terms;
        for (uint16_t r = 0; r < 6; ++r) {
            if (any[r]) by_terms.push_back(r);
        }
        EXPECT_EQ(probe, by_terms) << "dictionary " << dictionary;
        EXPECT_EQ(probe, Rows({0, 1, 4}));
    }

    // Numbers: a literal with more decimals than the column never matches.
    TestBatch t;
    const uint32_t n = t.AddDecimals(15, 2, {5, 700, 6, 55, 0, 700});
    const uint32_t i = t.AddInts(ColumnType::Integer(), {1, 2, 3, 4, 5, 6});
    t.Finish(6);
    std::vector<ExprPtr> nums;
    nums.push_back(Dec("0.05"));
    nums.push_back(Dec("0.055"));
    nums.push_back(Int(7));
    auto ncol = Col("n", n);
    auto nin = In(*ncol, std::move(nums));
    Rebind(*nin, n);
    EXPECT_EQ(Selected(*Compile(*nin, t), t), Rows({0, 1, 5}));

    std::vector<ExprPtr> ints;
    ints.push_back(Int(2));
    ints.push_back(Dec("4.0"));
    ints.push_back(Dec("5.5"));
    auto icol = Col("i", i);
    auto iin = In(*icol, std::move(ints));
    Rebind(*iin, i);
    EXPECT_EQ(Selected(*Compile(*iin, t), t), Rows({1, 3}));
}

TEST(ParallelExpressionTest, LikeFastPathsMatchMatchLike) {
    const std::vector<std::string> values = {
        "PROMO BURNISHED COPPER", "LARGE POLISHED BRASS", "forest green metal", "abcdef", "PROMO%x", "%BRASS",
    };
    const std::vector<std::string> patterns = {"PROMO%", "%BRASS", "%green%", "a_c%", "%", "PROMO", "%a%c%"};
    for (const bool dictionary : {false, true}) {
        TestBatch t;
        const uint32_t s = t.AddStrings(values, dictionary);
        t.Finish(6);
        for (const std::string& pattern : patterns) {
            auto like = Bin(Op::LIKE, Col("p_type", s), Str(pattern));
            std::vector<uint16_t> expected;
            for (uint16_t r = 0; r < 6; ++r) {
                if (MatchLike(values[r], pattern)) expected.push_back(r);
            }
            EXPECT_EQ(Selected(*Compile(*like, t), t), expected) << pattern << " dictionary " << dictionary;
        }
    }

    // A pattern that isn't a literal goes through MatchLike row by row.
    const std::vector<std::string> strs = {"abc", "abc", "xyz", "", "a", "%"};
    const std::vector<std::string> pats = {"a%", "_b_", "%z", "%", "", "%_"};
    TestBatch t;
    const uint32_t s = t.AddStrings(strs);
    const uint32_t p = t.AddStrings(pats);
    t.Finish(6);
    std::vector<uint16_t> expected;
    for (uint16_t r = 0; r < 6; ++r) {
        if (MatchLike(strs[r], pats[r])) expected.push_back(r);
    }
    auto like = Bin(Op::LIKE, Col("s", s), Col("p", p));
    EXPECT_EQ(Selected(*Compile(*like, t), t), expected);
    EXPECT_EQ(expected, Rows({0, 1, 2, 3}));
}

TEST(ParallelExpressionTest, CaseEvaluatesBranchesOnTheirRows) {
    TestBatch t;
    const uint32_t prio =
        t.AddStrings({"1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW", "1-URGENT"}, true);
    const uint32_t x = t.AddInts(ColumnType::Integer(), {0, 5, 20, 0, 2, -10});
    t.Finish(6);

    // Q12's two CASEs.
    auto high = std::make_unique<CaseExpression>();
    high->when_clauses.emplace_back(Bin(Op::OR, Bin(Op::EQ, Col("o_orderpriority", prio), Str("1-URGENT")),
                                        Bin(Op::EQ, Col("o_orderpriority", prio), Str("2-HIGH"))),
                                    Int(1));
    high->else_clause = Int(0);
    auto low = std::make_unique<CaseExpression>();
    low->when_clauses.emplace_back(Bin(Op::AND, Bin(Op::NEQ, Col("o_orderpriority", prio), Str("1-URGENT")),
                                       Bin(Op::NEQ, Col("o_orderpriority", prio), Str("2-HIGH"))),
                                   Int(1));
    low->else_clause = Int(0);

    const Vector hv = EvaluateAll(*Compile(*high, t), t);
    const Vector lv = EvaluateAll(*Compile(*low, t), t);
    const std::vector<int64_t> high_expected = {1, 1, 0, 0, 0, 1};
    for (uint32_t i = 0; i < 6; ++i) {
        EXPECT_EQ(hv.I64()[i], high_expected[i]) << "row " << i;
        EXPECT_EQ(lv.I64()[i], 1 - high_expected[i]) << "row " << i;
    }

    // A branch that would divide by zero runs only on its own rows.
    auto guarded = std::make_unique<CaseExpression>();
    guarded->when_clauses.emplace_back(Bin(Op::NEQ, Col("x", x), Int(0)), Bin(Op::DIV, Int(10), Col("x", x)));
    guarded->else_clause = Un(UnaryOp::Op::NEGATE, Int(1));
    const Vector gv = EvaluateAll(*Compile(*guarded, t), t);
    const std::vector<int64_t> guarded_expected = {-1, 2, 0, -1, 5, -1};
    for (uint32_t i = 0; i < 6; ++i) EXPECT_EQ(gv.I64()[i], guarded_expected[i]) << "row " << i;

    // Conditions run in order: a later WHEN never sees rows an earlier one took.
    auto ordered = std::make_unique<CaseExpression>();
    ordered->when_clauses.emplace_back(Bin(Op::EQ, Col("x", x), Int(0)), Int(100));
    ordered->when_clauses.emplace_back(Bin(Op::GT, Bin(Op::DIV, Int(10), Col("x", x)), Int(0)), Int(200));
    const Vector ov = EvaluateAll(*Compile(*ordered, t), t);
    const std::vector<int64_t> ordered_expected = {100, 200, 0, 100, 200, 0};  // no ELSE: zero
    for (uint32_t i = 0; i < 6; ++i) EXPECT_EQ(ov.I64()[i], ordered_expected[i]) << "row " << i;

    // Mixed branch types take the common type: exact decimal at the larger scale.
    auto mixed = std::make_unique<CaseExpression>();
    mixed->when_clauses.emplace_back(Bin(Op::GT, Col("x", x), Int(3)), Dec("2.5"));
    mixed->else_clause = Col("x", x);
    auto mc = Compile(*mixed, t);
    EXPECT_EQ(mc->Type().kind, Kind::DECIMAL);
    EXPECT_EQ(mc->Type().scale, 1);
    const Vector mv = EvaluateAll(*mc, t);
    const std::vector<int64_t> mixed_expected = {0, 25, 25, 0, 20, -100};
    for (uint32_t i = 0; i < 6; ++i) EXPECT_TRUE(DecimalAt(mv, i) == mixed_expected[i]) << "row " << i;

    // String results.
    auto label = std::make_unique<CaseExpression>();
    label->when_clauses.emplace_back(Bin(Op::LIKE, Col("o_orderpriority", prio), Str("1%")), Str("first"));
    label->else_clause = Col("o_orderpriority", prio);
    auto label_c = Compile(*label, t);  // string results may point into the tree's constants
    const Vector sv = EvaluateAll(*label_c, t);
    EXPECT_EQ(View(sv.StrAt(0)), "first");
    EXPECT_EQ(View(sv.StrAt(1)), "2-HIGH");
    EXPECT_EQ(View(sv.StrAt(5)), "first");
}

TEST(ParallelExpressionTest, DictionaryPredicateOncePerEntry) {
    const std::vector<std::string> entries = {"AIR", "MAIL", "SHIP"};
    std::vector<std::string> values(kBatchSize);
    for (uint32_t i = 0; i < kBatchSize; ++i) values[i] = entries[(i * 7 + i / 5) % 3];

    TestBatch dict;
    const uint32_t dc = dict.AddStrings(values, true);
    dict.Finish(kBatchSize);
    ASSERT_TRUE(dict.batch.slots[dc].IsDictionary());
    ASSERT_EQ(dict.batch.slots[dc].Entries().size(), 3u);
    TestBatch plain;
    const uint32_t pc = plain.AddStrings(values, false);
    plain.Finish(kBatchSize);

    // Every third row, so the partial path runs too.
    std::vector<uint16_t> some;
    for (uint16_t r = 0; r < kBatchSize; r += 3) some.push_back(r);

    auto make = [](uint32_t c, int which) -> ExprPtr {
        switch (which) {
            case 0:
                return Bin(Op::EQ, Col("m", c), Str("MAIL"));
            case 1:
                return Bin(Op::LT, Str("AIR"), Col("m", c));
            case 2: {
                auto col = Col("m", c);
                std::vector<ExprPtr> v;
                v.push_back(Str("MAIL"));
                v.push_back(Str("SHIP"));
                auto in = In(*col, std::move(v));
                Rebind(*in, c);
                return in;
            }
            case 3:
                return Bin(Op::LIKE, Col("m", c), Str("%AI%"));
            default:
                return Un(UnaryOp::Op::NOT, Bin(Op::LIKE, Col("m", c), Str("_HIP")));
        }
    };
    for (int which = 0; which < 5; ++which) {
        auto on_dict = Compile(*make(dc, which), dict);
        auto on_plain = Compile(*make(pc, which), plain);
        const std::vector<uint16_t> want = Selected(*on_plain, plain);
        EXPECT_EQ(Selected(*on_dict, dict), want) << "predicate " << which;
        EXPECT_FALSE(want.empty());
        EXPECT_EQ(SelectedFrom(*on_dict, dict, some), SelectedFrom(*on_plain, plain, some)) << "predicate " << which;
        const Vector dv = EvaluateAll(*on_dict, dict);
        const Vector pv = EvaluateAll(*on_plain, plain);
        for (uint32_t i = 0; i < kBatchSize; ++i) ASSERT_EQ(dv.I64()[i], pv.I64()[i]) << "row " << i;
    }
}

TEST(ParallelExpressionTest, CompileErrorsUseTodaysText) {
    TestBatch t;
    const uint32_t i = t.AddInts(ColumnType::Integer(), {1, 2, 3, 4, 5, 6});
    const uint32_t s = t.AddStrings({"a", "b", "c", "d", "e", "f"});
    const uint32_t dt = t.AddInts(ColumnType::Date(), {1, 2, 3, 4, 5, 6});
    t.Finish(6);

    auto error = [&](ExprPtr e) { return ErrorOf([&] { (void)Compile(*e, t); }); };
    EXPECT_EQ(error(Bin(Op::ADD, Col("s", s), Int(1))), "Type mismatch in arithmetic");
    EXPECT_EQ(error(Bin(Op::ADD, Col("dt", dt), Col("i", i))), "Type mismatch in arithmetic");
    EXPECT_EQ(error(Bin(Op::MUL, Col("s", s), Col("s", s))), "Arithmetic not supported for this type");
    EXPECT_EQ(error(Bin(Op::SUB, Col("dt", dt), Col("dt", dt))), "Arithmetic not supported for this type");
    EXPECT_EQ(error(Bin(Op::EQ, Col("s", s), Int(1))), "Type mismatch in comparison");
    EXPECT_EQ(error(Bin(Op::LT, Col("dt", dt), Int(1))), "Type mismatch in comparison");
    EXPECT_EQ(error(Bin(Op::GT, Col("s", s), Dec("1.5"))), "Type mismatch in comparison");
    EXPECT_EQ(error(Un(UnaryOp::Op::NEGATE, Col("dt", dt))), "NEGATE not supported for this type");
    EXPECT_EQ(error(Un(UnaryOp::Op::NEGATE, Col("s", s))), "NEGATE not supported for this type");

    auto agg = std::make_unique<Aggregate>();
    agg->func = Aggregate::Func::SUM;
    agg->arg = Col("i", i);
    EXPECT_EQ(error(std::move(agg)), "Cannot evaluate AGGREGATE or STAR expression in this context");
    EXPECT_EQ(error(std::make_unique<StarExpr>()), "Cannot evaluate AGGREGATE or STAR expression in this context");

    auto unbound = std::make_unique<ColumnRef>();
    unbound->column_name = "l_mystery";
    EXPECT_NE(error(std::move(unbound)).find("l_mystery"), std::string::npos);
    EXPECT_NE(error(Col("l_missing", 99)).find("l_missing"), std::string::npos);

    // A DATE against a string literal converts the literal once.
    auto date_eq = Bin(Op::EQ, Col("dt", dt), Str("1970-01-03"));
    EXPECT_EQ(Selected(*Compile(*date_eq, t), t), Rows({1}));
}

TEST(ParallelExpressionTest, IntendedCompileTimeRejections) {
    // These fail at compile time on purpose, even if no row would reach them.
    TestBatch t;
    const uint32_t i = t.AddInts(ColumnType::Integer(), {1, 2, 3, 4, 5, 6});
    const uint32_t s = t.AddStrings({"a", "b", "c", "d", "e", "f"});
    const uint32_t dt = t.AddInts(ColumnType::Date(), {1, 2, 3, 4, 5, 6});
    t.Finish(6);
    auto error = [&](ExprPtr e) { return ErrorOf([&] { (void)Compile(*e, t); }); };

    // LIKE on a side that isn't text (today compares ToString() texts).
    EXPECT_NE(error(Bin(Op::LIKE, Col("i", i), Str("1%"))).find("LIKE"), std::string::npos);

    // A DATE against a string that isn't a constant.
    EXPECT_EQ(error(Bin(Op::EQ, Col("dt", dt), Col("s", s))), "a DATE can only be compared with a constant string");

    // A malformed date literal throws today's parse error once, at compile time.
    auto bad_date = Bin(Op::LT, Col("dt", dt), Str("not-a-date"));
    EXPECT_THROW((void)Compile(*bad_date, t), std::invalid_argument);

    // A CASE mixing strings and numbers, or dates and a string that isn't a constant.
    auto mixed = std::make_unique<CaseExpression>();
    mixed->when_clauses.emplace_back(Bin(Op::GT, Col("i", i), Int(3)), Str("big"));
    mixed->else_clause = Int(0);
    EXPECT_EQ(error(std::move(mixed)), "CASE results have types that don't mix");
    auto dates = std::make_unique<CaseExpression>();
    dates->when_clauses.emplace_back(Bin(Op::GT, Col("i", i), Int(3)), Col("dt", dt));
    dates->else_clause = Col("s", s);
    EXPECT_EQ(error(std::move(dates)), "CASE results have types that don't mix");
}

TEST(ParallelExpressionTest, CaseReadsConstantStringNextToDates) {
    TestBatch t;
    const uint32_t x = t.AddInts(ColumnType::Integer(), {1, 5, 2, 9, 0, 4});
    const uint32_t dt = t.AddInts(ColumnType::Date(), {10, 11, 12, 13, 14, 15});
    t.Finish(6);

    auto c = std::make_unique<CaseExpression>();
    c->when_clauses.emplace_back(Bin(Op::GT, Col("x", x), Int(3)), Col("dt", dt));
    c->else_clause = Str("1970-01-05");
    auto compiled = Compile(*c, t);
    EXPECT_EQ(compiled->Type().kind, Kind::DATE);
    const Vector v = EvaluateAll(*compiled, t);
    const std::vector<int64_t> expected = {4, 11, 4, 13, 4, 15};
    for (uint32_t r = 0; r < 6; ++r) EXPECT_EQ(v.I64()[r], expected[r]) << "row " << r;
}

TEST(ParallelExpressionTest, IntendedDifferencesFromToday) {
    TestBatch t;
    const uint32_t x = t.AddInts(ColumnType::Integer(), {5, -1, 2, 0, 9, -4});
    t.Finish(6);
    const Schema schema({Column("x", TypeId::INTEGER)});
    const Tuple positive(std::vector<Value>{Value(int32_t{5})}, schema);
    const Tuple negative(std::vector<Value>{Value(int32_t{-1})}, schema);

    // A CASE has one static type: 7 is DECIMAL, so / 2 is 3.5 (today 3).
    auto c = std::make_unique<CaseExpression>();
    c->when_clauses.emplace_back(Bin(Op::GT, Col("x", x), Int(0)), Int(7));
    c->else_clause = Dec("0.5");
    auto half = Bin(Op::DIV, std::move(c), Int(2));
    auto compiled = Compile(*half, t);
    EXPECT_EQ(compiled->Type().kind, Kind::DOUBLE);
    const Vector v = EvaluateAll(*compiled, t);
    EXPECT_EQ(v.F64()[0], 3.5);
    EXPECT_EQ(v.F64()[1], 0.25);
    const Value today = EvaluateExpression(half.get(), positive, schema);
    EXPECT_EQ(today.type_, TypeId::INTEGER);
    EXPECT_EQ(today.integer_, 3);

    // No ELSE over text or dates gives '' or day 0 (today an INTEGER 0).
    auto text = std::make_unique<CaseExpression>();
    text->when_clauses.emplace_back(Bin(Op::GT, Col("x", x), Int(0)), Str("yes"));
    auto text_c = Compile(*text, t);
    const Vector tv = EvaluateAll(*text_c, t);
    EXPECT_EQ(View(tv.StrAt(0)), "yes");
    EXPECT_EQ(View(tv.StrAt(1)), "");
    EXPECT_EQ(EvaluateExpression(text.get(), negative, schema).type_, TypeId::INTEGER);

    auto day = std::make_unique<CaseExpression>();
    day->when_clauses.emplace_back(Bin(Op::GT, Col("x", x), Int(0)), Lit(Value::MakeDate(10)));
    auto day_c = Compile(*day, t);
    EXPECT_EQ(day_c->Type().kind, Kind::DATE);
    const Vector dv = EvaluateAll(*day_c, t);
    EXPECT_EQ(dv.I64()[0], 10);
    EXPECT_EQ(dv.I64()[1], 0);
    EXPECT_EQ(EvaluateExpression(day.get(), negative, schema).type_, TypeId::INTEGER);
}

TEST(ParallelExpressionTest, WideComparisonsAtExactTies) {
    // a is NUMERIC(38,2), held wide; b is NUMERIC(20,4). a needs two more
    // decimals, so `a op b` shifts the left side and `b op a` the right.
    const std::vector<__int128> as = {12345, 12345, 12345, -12345, -12345, -12345, 0, 7};
    const std::vector<__int128> bs = {1234500, 1234501, 1234499, -1234500, -1234499, -1234501, 0, 70001};
    TestBatch t;
    const uint32_t a = t.AddDecimals(38, 2, as);
    const uint32_t b = t.AddDecimals(20, 4, bs);
    t.Finish(8);

    for (const Op op : {Op::EQ, Op::NEQ, Op::LT, Op::LTE, Op::GT, Op::GTE}) {
        for (const bool a_left : {true, false}) {
            auto expr = a_left ? Bin(op, Col("a", a), Col("b", b)) : Bin(op, Col("b", b), Col("a", a));
            std::vector<uint16_t> expected;
            for (uint16_t r = 0; r < 8; ++r) {
                const __int128 l = a_left ? as[r] * 100 : bs[r];
                const __int128 rr = a_left ? bs[r] : as[r] * 100;
                const bool hit = op == Op::EQ ? l == rr : op == Op::NEQ ? l != rr : op == Op::LT ? l < rr
                               : op == Op::LTE ? l <= rr : op == Op::GT ? l > rr : l >= rr;
                if (hit) expected.push_back(r);
            }
            auto compiled = Compile(*expr, t);
            EXPECT_EQ(Selected(*compiled, t), expected) << "op " << static_cast<int>(op) << " a_left " << a_left;
            const Vector v = EvaluateAll(*compiled, t);
            for (uint16_t r = 0; r < 8; ++r) {
                const bool hit = std::find(expected.begin(), expected.end(), r) != expected.end();
                EXPECT_EQ(v.I64()[r], hit ? 1 : 0) << "op " << static_cast<int>(op) << " row " << r;
            }
        }
    }
}

TEST(ParallelExpressionTest, ProductPastScale38IsDouble) {
    TestBatch t;
    const std::vector<int64_t> xs = {0, 1, -3, 20, 5, -1};
    const uint32_t x = t.AddInts(ColumnType::Integer(), xs);
    t.Finish(6);

    // Ten multiplications by 1.0425 add up to 40 decimals: past 38, so doubles.
    auto chain = [&] {
        ExprPtr e = Col("x", x);
        for (int k = 0; k < 10; ++k) e = Bin(Op::MUL, std::move(e), Dec("1.0425"));
        return e;
    };
    auto close = [](double a, double b) {
        const double diff = a > b ? a - b : b - a;
        const double size = std::max({1.0, a > 0 ? a : -a, b > 0 ? b : -b});
        return diff <= 1e-12 * size;
    };
    auto expected = [&](int64_t v) {
        double d = static_cast<double>(v);
        for (int k = 0; k < 10; ++k) d *= 1.0425;
        return d;
    };

    auto plain = Compile(*chain(), t);
    EXPECT_EQ(plain->Type().kind, Kind::DOUBLE);
    EXPECT_EQ(plain->Type().result, TypeId::DECIMAL);
    const Vector pv = EvaluateAll(*plain, t);
    for (uint32_t r = 0; r < 6; ++r) EXPECT_TRUE(close(pv.F64()[r], expected(xs[r]))) << "row " << r;

    // Under a WHEN no row takes, and under one some rows take.
    for (const int limit : {100, 0}) {
        auto c = std::make_unique<CaseExpression>();
        c->when_clauses.emplace_back(Bin(Op::GT, Col("x", x), Int(limit)), chain());
        c->else_clause = Int(0);
        auto compiled = Compile(*c, t);
        EXPECT_EQ(compiled->Type().kind, Kind::DOUBLE);
        const Vector v = EvaluateAll(*compiled, t);
        for (uint32_t r = 0; r < 6; ++r) {
            const double want = xs[r] > limit ? expected(xs[r]) : 0.0;
            EXPECT_TRUE(close(v.F64()[r], want)) << "limit " << limit << " row " << r;
        }
    }
}

TEST(ParallelExpressionTest, InProbeMatchesOrChainOnWideColumn) {
    // 10^38 - 1 rounds to the double 1e38, so it equals 1e38.
    __int128 max = 1;
    for (int k = 0; k < 38; ++k) max *= 10;
    max -= 1;
    TestBatch t;
    const uint32_t c = t.AddDecimals(38, 0, {max, 1, 2, -max, 0, 5});
    t.Finish(6);

    const std::vector<std::vector<std::string>> lists = {{"1e38", "1"}, {"1e38", "1e40"}};
    const std::vector<std::vector<uint16_t>> want = {Rows({0, 1}), Rows({0})};
    for (size_t l = 0; l < lists.size(); ++l) {
        auto col = Col("c", c);
        std::vector<ExprPtr> values;
        for (const std::string& v : lists[l]) values.push_back(v == "1" ? Int(1) : Dec(v));
        auto in = In(*col, std::move(values));
        Rebind(*in, c);
        const std::vector<uint16_t> probe = Selected(*Compile(*in, t), t);

        std::vector<bool> any(6, false);
        for (const std::string& v : lists[l]) {
            auto eq = Bin(Op::EQ, Col("c", c), v == "1" ? Int(1) : Dec(v));
            for (uint16_t r : Selected(*Compile(*eq, t), t)) any[r] = true;
        }
        std::vector<uint16_t> chain;
        for (uint16_t r = 0; r < 6; ++r) {
            if (any[r]) chain.push_back(r);
        }
        EXPECT_EQ(probe, chain) << "list " << l;
        EXPECT_EQ(probe, want[l]) << "list " << l;
    }
}

TEST(ParallelExpressionTest, FastAndCheckedWideMathAgree) {
    // The same values as narrow NUMERIC(18,s) columns, whose digit bound lets the
    // kernels skip checks, and as wide NUMERIC(38,s) columns, which are checked.
    std::mt19937_64 rng(42);
    const int64_t limit = 999'999'999'999'999'999;
    std::uniform_int_distribution<int64_t> any(-limit, limit);
    std::vector<__int128> as(kBatchSize), bs(kBatchSize);
    for (uint32_t r = 0; r < kBatchSize; ++r) {
        as[r] = any(rng);
        bs[r] = any(rng);
    }
    TestBatch t;
    const uint32_t na = t.AddDecimals(18, 2, as);
    const uint32_t nb = t.AddDecimals(18, 4, bs);
    const uint32_t wa = t.AddDecimals(38, 2, as);
    const uint32_t wb = t.AddDecimals(38, 4, bs);
    t.Finish(kBatchSize);

    for (const Op op : {Op::ADD, Op::SUB, Op::MUL}) {
        auto fast = Compile(*Bin(op, Col("na", na), Col("nb", nb)), t);
        auto checked = Compile(*Bin(op, Col("wa", wa), Col("wb", wb)), t);
        auto mixed = Compile(*Bin(op, Col("na", na), Col("wb", wb)), t);
        ASSERT_TRUE(fast->Type().wide);
        ASSERT_EQ(fast->Type(), checked->Type());
        ASSERT_EQ(fast->Type(), mixed->Type());
        const Vector fv = EvaluateAll(*fast, t);
        const Vector cv = EvaluateAll(*checked, t);
        const Vector mv = EvaluateAll(*mixed, t);
        for (uint32_t r = 0; r < kBatchSize; ++r) {
            const __int128 want = op == Op::ADD ? as[r] * 100 + bs[r] : op == Op::SUB ? as[r] * 100 - bs[r] : as[r] * bs[r];
            ASSERT_TRUE(fv.I128()[r] == want) << "op " << static_cast<int>(op) << " row " << r;
            ASSERT_TRUE(cv.I128()[r] == want) << "op " << static_cast<int>(op) << " row " << r;
            ASSERT_TRUE(mv.I128()[r] == want) << "op " << static_cast<int>(op) << " row " << r;
        }
    }
}

TEST(ParallelExpressionTest, ConstantsFoldExactly) {
    TestBatch t;
    t.AddInts(ColumnType::Integer(), {1, 2, 3, 4, 5, 6});
    t.Finish(6);

    auto neg = Compile(*Un(UnaryOp::Op::NEGATE, Dec("0.055")), t);
    EXPECT_EQ(neg->Type().kind, Kind::DECIMAL);
    EXPECT_EQ(neg->Type().scale, 3);
    EXPECT_TRUE(DecimalAt(EvaluateAll(*neg, t), 4) == -55);

    auto three = Compile(*Bin(Op::ADD, Int(1), Int(2)), t);
    EXPECT_EQ(three->Type().kind, Kind::INTEGER);
    EXPECT_EQ(EvaluateAll(*three, t).I64()[5], 3);

    // A constant that throws waits for a row to throw on.
    auto bad = Compile(*Bin(Op::DIV, Int(1), Int(0)), t);
    EXPECT_EQ(ErrorOf([&] { (void)EvaluateAll(*bad, t); }), "Division by zero");
    EvalScratch scratch;
    Vector out(bad->Type());
    Selection none;
    none.Resize(0);
    bad->Evaluate(t.batch, none, out, scratch);

    // A wide product folds to a small value; a sum with it stays wide.
    auto big = [] { return Bin(Op::MUL, Dec("1000000000.5"), Dec("1000000000.5")); };
    auto wide_zero = Compile(*Bin(Op::ADD, Bin(Op::SUB, big(), big()), Dec("0.5")), t);
    EXPECT_EQ(wide_zero->Type().kind, Kind::DECIMAL);
    EXPECT_EQ(wide_zero->Type().scale, 2);
    EXPECT_TRUE(DecimalAt(EvaluateAll(*wide_zero, t), 0) == 50);

    auto truth = Compile(*Bin(Op::AND, Int(1), Bin(Op::LT, Int(2), Int(3))), t);
    EXPECT_EQ(Selected(*truth, t).size(), 6u);
    auto never = Compile(*Dec("1.5"), t);
    EXPECT_TRUE(Selected(*never, t).empty());  // only a non-zero INTEGER is true
}

TEST(ParallelExpressionTest, CollectColumnsSortedAndUnique) {
    auto c = std::make_unique<CaseExpression>();
    c->when_clauses.emplace_back(Bin(Op::GT, Col("e", 7), Col("b", 2)), Col("e", 7));
    c->else_clause = Bin(Op::ADD, Col("a", 0), Int(1));
    std::vector<uint32_t> cols;
    CollectColumns(*c, cols);
    EXPECT_EQ(cols, (std::vector<uint32_t>{0, 2, 7}));

    // A second expression adds to the same list.
    CollectColumns(*Bin(Op::EQ, Col("c", 5), Col("b", 2)), cols);
    EXPECT_EQ(cols, (std::vector<uint32_t>{0, 2, 5, 7}));
}

TEST(ParallelExpressionTest, ManyThreadsShareOneTree) {
    TestBatch t;
    std::vector<int64_t> xs(kBatchSize);
    std::vector<__int128> ns(kBatchSize);
    std::vector<std::string> ss(kBatchSize);
    for (uint32_t i = 0; i < kBatchSize; ++i) {
        xs[i] = static_cast<int64_t>(i % 17) - 8;
        ns[i] = static_cast<__int128>(i * 37 % 1000);
        ss[i] = (i % 3 == 0) ? "MAIL" : "SHIP";
    }
    const uint32_t x = t.AddInts(ColumnType::Integer(), xs);
    const uint32_t n = t.AddDecimals(15, 2, ns);
    const uint32_t s = t.AddStrings(ss, true);
    t.Finish(kBatchSize);

    auto cond = Bin(Op::AND, Bin(Op::NEQ, Col("x", x), Int(0)),
                    Bin(Op::OR, Bin(Op::EQ, Col("s", s), Str("MAIL")),
                        Bin(Op::GT, Bin(Op::MUL, Col("n", n), Bin(Op::DIV, Int(100), Col("x", x))), Dec("3.5"))));
    auto compiled = Compile(*cond, t);
    const std::vector<uint16_t> want = Selected(*compiled, t);

    std::vector<int> mismatches(4, 0);
    std::vector<std::thread> workers;
    for (int w = 0; w < 4; ++w) {
        workers.emplace_back([&, w] {
            EvalScratch scratch;
            for (int round = 0; round < 50; ++round) {
                Selection sel;
                sel.SetAll(kBatchSize);
                compiled->Select(t.batch, sel, scratch);
                if (std::vector<uint16_t>(sel.Data(), sel.Data() + sel.Size()) != want) ++mismatches[w];
            }
        });
    }
    for (std::thread& worker : workers) worker.join();
    EXPECT_EQ(mismatches, (std::vector<int>{0, 0, 0, 0}));
}

}  // namespace shilmandb::parallel
