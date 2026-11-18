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
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace shilmandb::parallel {

namespace {

using ExprPtr = std::unique_ptr<Expression>;
using Op = BinaryOp::Op;

constexpr uint32_t kSeeds = 200;
constexpr uint32_t kRows = 1000;
constexpr int kTreesPerSeed = 3;
constexpr int kRootDepth = 4;
constexpr double kTolerance = 1e-12;  // relative, against max(1, |a|, |b|)
constexpr double kUnit = 2.3e-16;     // a little over one rounding of a double
constexpr double kBudget = 2e-13;     // most any value may drift from the real one
constexpr int kMaxFailures = 25;      // stop printing after this many
constexpr uint64_t kMaxForgivenFlips = 20;  // near-tie flips forgiven over all seeds

// The schema's columns, in plan order.
enum ColumnId : uint32_t { kI, kJ, kB, kN, kM, kD, kDt, kS, kColumnCount };
const char* const kNames[kColumnCount] = {"i", "j", "b", "n", "m", "d", "dt", "s"};

// The storage type of each column.
ColumnType StorageOf(uint32_t c) {
    switch (c) {
        case kI:
        case kJ:
            return ColumnType::Integer();
        case kB:
            return ColumnType::BigInt();
        case kN:
            return ColumnType::Numeric(15, 2);
        case kM:
            return ColumnType::Numeric(12, 4);
        case kD:
            return ColumnType::Double();
        case kDt:
            return ColumnType::Date();
        default:
            return ColumnType::Varchar(40);
    }
}

Schema MakeSchema() {
    std::vector<Column> cols;
    for (uint32_t c = 0; c < kColumnCount; ++c) cols.emplace_back(kNames[c], StorageOf(c));
    return Schema(cols);
}

// ---- Expression builders ----

ExprPtr ColE(uint32_t c) {
    auto e = std::make_unique<ColumnRef>();
    e->column_name = kNames[c];
    e->bound_index = c;
    return e;
}

ExprPtr LitE(Value v) {
    auto e = std::make_unique<Literal>();
    e->value = std::move(v);
    return e;
}

ExprPtr BinE(Op op, ExprPtr l, ExprPtr r) {
    auto e = std::make_unique<BinaryOp>();
    e->op = op;
    e->left = std::move(l);
    e->right = std::move(r);
    return e;
}

ExprPtr UnE(UnaryOp::Op op, ExprPtr x) {
    auto e = std::make_unique<UnaryOp>();
    e->op = op;
    e->operand = std::move(x);
    return e;
}

// The binder's rule: a literal by a column takes its type when that widens.
void Widen(Expression& maybe_col, Expression& maybe_lit, const Schema& schema) {
    if (maybe_col.type != ExprType::COLUMN_REF || maybe_lit.type != ExprType::LITERAL) return;
    const auto& col = static_cast<const ColumnRef&>(maybe_col);
    auto& lit = static_cast<Literal&>(maybe_lit);
    lit.value = lit.value.WidenTo(schema.GetColumn(*col.bound_index).type);
}

// ---- Printing, for failure messages ----

std::string Show(const Expression& e) {
    switch (e.type) {
        case ExprType::COLUMN_REF:
            return static_cast<const ColumnRef&>(e).column_name;
        case ExprType::LITERAL: {
            const Value& v = static_cast<const Literal&>(e).value;
            char buf[64];
            switch (v.type_) {
                case TypeId::INTEGER:
                    return std::to_string(v.integer_);
                case TypeId::BIGINT:
                    return std::to_string(v.bigint_) + "L";
                case TypeId::DECIMAL:
                    std::snprintf(buf, sizeof(buf), "%.17gd", v.decimal_);
                    return buf;
                case TypeId::DATE:
                    return "DATE '" + v.ToString() + "'";
                case TypeId::VARCHAR:
                    return "'" + v.varchar_ + "'";
                case TypeId::INVALID:
                    return "NULL";
            }
            return "?";
        }
        case ExprType::BINARY_OP: {
            const auto& b = static_cast<const BinaryOp&>(e);
            static const char* const kOps[] = {"=", "<>", "<", ">", "<=", ">=", "AND", "OR", "+", "-", "*", "/", "LIKE"};
            return "(" + Show(*b.left) + " " + kOps[static_cast<int>(b.op)] + " " + Show(*b.right) + ")";
        }
        case ExprType::UNARY_OP: {
            const auto& u = static_cast<const UnaryOp&>(e);
            return (u.op == UnaryOp::Op::NOT ? "NOT " : "-") + Show(*u.operand);
        }
        case ExprType::CASE: {
            const auto& c = static_cast<const CaseExpression&>(e);
            std::string s = "CASE";
            for (const auto& [cond, result] : c.when_clauses) s += " WHEN " + Show(*cond) + " THEN " + Show(*result);
            if (c.else_clause) s += " ELSE " + Show(*c.else_clause);
            return s + " END";
        }
        default:
            return "?";
    }
}

// ---- The data: the same rows as Tuples and as one batch ----

struct Data {
    std::vector<int64_t> i, j, b, n, m, dt;
    std::vector<double> d;
    std::vector<std::string> s;
};

// Small magnitudes keep today's double rounding far below the tolerance.
Data MakeData(std::mt19937& rng) {
    auto between = [&rng](int64_t lo, int64_t hi) { return std::uniform_int_distribution<int64_t>(lo, hi)(rng); };
    static const char* const kPieces[] = {"a", "b", "c", "%", "_", "\xC3\xA9"};  // the last is a two-byte é
    Data data;
    for (uint32_t r = 0; r < kRows; ++r) {
        data.i.push_back(between(-20, 20));
        data.j.push_back(between(-20, 20));
        data.b.push_back(between(-500, 500));
        data.n.push_back(between(-1000, 1000));    // -10.00 .. 10.00
        data.m.push_back(between(-10000, 10000));  // -1.0000 .. 1.0000
        switch (between(0, 3)) {
            case 0:
                data.d.push_back(static_cast<double>(between(-80, 80)) / 8.0);
                break;
            case 1:
                data.d.push_back(static_cast<double>(between(-1000, 1000)) / 100.0);
                break;
            case 2:
                data.d.push_back(std::uniform_real_distribution<double>(-10.0, 10.0)(rng));
                break;
            default:
                data.d.push_back(between(0, 1) == 0 ? 0.0 : -0.0);
                break;
        }
        data.dt.push_back(between(9000, 9060));
        std::string s;
        const int64_t len = between(0, 5);
        for (int64_t k = 0; k < len; ++k) s += kPieces[between(0, 5)];
        data.s.push_back(s);
    }
    return data;
}

std::vector<Tuple> MakeTuples(const Data& data, const Schema& schema) {
    std::vector<Tuple> tuples;
    tuples.reserve(kRows);
    for (uint32_t r = 0; r < kRows; ++r) {
        std::vector<Value> v;
        v.emplace_back(static_cast<int32_t>(data.i[r]));
        v.emplace_back(static_cast<int32_t>(data.j[r]));
        v.emplace_back(data.b[r]);
        v.emplace_back(DecimalToDouble(data.n[r], 2));  // what the columnar scan shows today
        v.emplace_back(DecimalToDouble(data.m[r], 4));
        v.emplace_back(data.d[r]);
        v.push_back(Value::MakeDate(static_cast<int32_t>(data.dt[r])));
        v.emplace_back(data.s[r]);
        tuples.emplace_back(std::move(v), schema);
    }
    return tuples;
}

struct Fixture {
    InputLayout layout;
    Batch batch;
    std::vector<StringRef> dictionary;
};

// Columns go to shuffled slots; the string column is a dictionary when asked.
void MakeBatch(const Data& data, std::mt19937& rng, bool dictionary, Fixture& fx) {
    std::vector<uint32_t> slot_of(kColumnCount);
    std::iota(slot_of.begin(), slot_of.end(), 0u);
    std::shuffle(slot_of.begin(), slot_of.end(), rng);
    std::vector<uint32_t> column_at(kColumnCount);
    for (uint32_t c = 0; c < kColumnCount; ++c) column_at[slot_of[c]] = c;

    fx.layout.slot_of_column.assign(kColumnCount, -1);
    for (uint32_t c = 0; c < kColumnCount; ++c) fx.layout.slot_of_column[c] = static_cast<int>(slot_of[c]);
    fx.batch.slots.clear();
    for (uint32_t slot = 0; slot < kColumnCount; ++slot) {
        const ColumnType type = StorageOf(column_at[slot]);
        fx.layout.slot_types.push_back(VTypeOf(type));
        fx.layout.slot_precision.push_back(type.precision);
        fx.batch.slots.emplace_back(VTypeOf(type));
    }

    auto fill = [&](uint32_t c, const std::vector<int64_t>& values) {
        int64_t* out = fx.batch.slots[slot_of[c]].I64();
        for (uint32_t r = 0; r < kRows; ++r) out[r] = values[r];
    };
    fill(kI, data.i);
    fill(kJ, data.j);
    fill(kB, data.b);
    fill(kN, data.n);
    fill(kM, data.m);
    fill(kDt, data.dt);
    double* d = fx.batch.slots[slot_of[kD]].F64();
    for (uint32_t r = 0; r < kRows; ++r) d[r] = data.d[r];

    Vector& s = fx.batch.slots[slot_of[kS]];
    auto ref = [](const std::string& v) { return StringRef{v.data(), static_cast<uint32_t>(v.size())}; };
    if (!dictionary) {
        for (uint32_t r = 0; r < kRows; ++r) s.Str()[r] = ref(data.s[r]);
    } else {
        std::vector<const std::string*> seen;
        for (uint32_t r = 0; r < kRows; ++r) {
            size_t code = 0;
            while (code < seen.size() && *seen[code] != data.s[r]) ++code;
            if (code == seen.size()) {
                seen.push_back(&data.s[r]);
                fx.dictionary.push_back(ref(data.s[r]));
            }
            s.Codes()[r] = static_cast<uint32_t>(code);
        }
        s.SetDictionary(&fx.dictionary);
    }
    fx.batch.rows = kRows;
    fx.batch.sel.SetAll(kRows);
}

// ---- The tree generator ----

// What a generated value is, as today's executor types it.
enum class G { INT, BIG, NUM, DBL, DATE, STR };

bool IsIntFamily(G g) { return g == G::INT || g == G::BIG; }

// A generated tree plus what the generator knows about its values.
struct Gen {
    ExprPtr e;
    G type{G::INT};
    double bound{0};        // |value| is at most this; infinity for a fractional quotient
    double err{0};          // both engines stay within this of the real value
    bool same{true};        // the compiled double is today's double, bit for bit
    bool terminal{false};   // a fractional quotient: no more arithmetic on it
    int scale{0};           // NUM only, an upper bound
    bool div_safe{true};    // no CASE inside mixes integer and fractional rows
    bool truth_safe{true};  // no CASE inside mixes INTEGER rows with other types
    bool type_exact{true};  // every row's TypeId today is the compiled result TypeId
};

// The family the compiler gives a + - * / of two families.
G ArithFamily(Op op, G a, G b) {
    if (op == Op::DIV) {
        if (!IsIntFamily(a) || !IsIntFamily(b)) return G::DBL;
        return (a == G::BIG || b == G::BIG) ? G::BIG : G::INT;
    }
    if (a == G::DBL || b == G::DBL) return G::DBL;
    if (a == G::NUM || b == G::NUM) return G::NUM;
    return (a == G::BIG || b == G::BIG) ? G::BIG : G::INT;
}

// The TypeId today's executor gives a value of this family.
TypeId TypeIdOf(G g) {
    switch (g) {
        case G::INT:
            return TypeId::INTEGER;
        case G::BIG:
            return TypeId::BIGINT;
        case G::DATE:
            return TypeId::DATE;
        case G::STR:
            return TypeId::VARCHAR;
        default:
            return TypeId::DECIMAL;
    }
}

// Which branch types a numeric CASE may take.
enum class Branches { ANY, INT_FAMILY, FRACTIONAL, INT_ONLY, NOT_INT, BIG_ONLY };

bool Allowed(Branches b, G g) {
    switch (b) {
        case Branches::ANY:
            return true;
        case Branches::INT_FAMILY:
            return IsIntFamily(g);
        case Branches::FRACTIONAL:
            return g == G::NUM || g == G::DBL;
        case Branches::INT_ONLY:
            return g == G::INT;
        case Branches::NOT_INT:
            return g != G::INT;
        case Branches::BIG_ONLY:
            return g == G::BIG;
    }
    return false;
}

// Builds random, well-typed trees whose answers both engines agree on within the rules.
class TreeGen {
public:
    TreeGen(std::mt19937& rng, const Schema& schema) : rng_(rng), schema_(schema) {}

    Gen Root() {
        const int pick = Pick(20);
        if (pick < 9) return Numeric(kRootDepth, false, Chance(0.5));
        if (pick < 16) return Bool(kRootDepth);
        if (pick < 18) return Date(kRootDepth);
        return Str(kRootDepth);
    }

private:
    int Pick(int n) { return std::uniform_int_distribution<int>(0, n - 1)(rng_); }
    bool Chance(double p) { return std::uniform_real_distribution<double>(0.0, 1.0)(rng_) < p; }

    // A decimal literal such as 3.125, read the way the parser reads one.
    Gen DecimalLiteral(int max_scale, int64_t max_units) {
        const int scale = Pick(max_scale + 1);
        int64_t pow = 1;
        for (int k = 0; k < scale; ++k) pow *= 10;
        const int64_t units = std::uniform_int_distribution<int64_t>(0, max_units * pow)(rng_);
        std::string text = std::to_string(units / pow) + ".";
        if (scale == 0) {
            text += "0";
        } else {
            const std::string frac = std::to_string(units % pow);
            text += std::string(static_cast<size_t>(scale) - frac.size(), '0') + frac;
        }
        Gen g;
        g.e = LitE(Value(std::stod(text)));
        g.type = G::NUM;
        g.bound = static_cast<double>(units) / static_cast<double>(pow);
        g.err = g.bound * kUnit;
        g.scale = scale;
        return g;
    }

    Gen Leaf(uint32_t c) {
        Gen g;
        g.e = ColE(c);
        switch (c) {
            case kI:
            case kJ:
                g.type = G::INT;
                g.bound = 20;
                break;
            case kB:
                g.type = G::BIG;
                g.bound = 500;
                break;
            case kN:
                g.type = G::NUM;
                g.bound = 10;
                g.err = 10 * kUnit;
                g.scale = 2;
                break;
            case kM:
                g.type = G::NUM;
                g.bound = 1;
                g.err = kUnit;
                g.scale = 4;
                break;
            default:
                g.type = G::DBL;
                g.bound = 10;
                break;
        }
        return g;
    }

    Gen NumericLeaf() {
        switch (Pick(11)) {
            case 0:
                return Leaf(kI);
            case 1:
                return Leaf(kJ);
            case 2:
                return Leaf(kB);
            case 3:
                return Leaf(kN);
            case 4:
                return Leaf(kM);
            case 5:
                return Leaf(kD);
            case 6: {
                Gen g;
                const int v = Pick(21);
                g.e = LitE(Value(static_cast<int32_t>(v)));
                g.bound = v;
                return g;
            }
            case 7: {
                Gen g;
                const int v = Pick(501);
                g.e = LitE(Value(static_cast<int64_t>(v)));
                g.type = G::BIG;
                g.bound = v;
                return g;
            }
            case 8:
            case 9:
                return DecimalLiteral(3, 10);
            default: {
                if (!Chance(0.2)) return DecimalLiteral(2, 10);
                Gen g;  // a double no exact decimal holds: compiled as a double
                g.e = LitE(Value(1e-40));
                g.type = G::DBL;
                g.bound = 1e-40;
                return g;
            }
        }
    }

    // A leaf of the given type, for when a CASE branch has to fit.
    Gen LeafOf(G g) {
        switch (g) {
            case G::INT:
                return Leaf(kI);
            case G::BIG:
                return Leaf(kB);
            case G::NUM:
                return Leaf(kN);
            default:
                return Leaf(kD);
        }
    }

    Gen Numeric(int depth, bool need_div_safe, bool need_truth_safe) {
        if (depth <= 0 || Chance(0.2)) return NumericLeaf();
        const int pick = Pick(10);
        if (pick < 5) return Arithmetic(depth, need_div_safe, need_truth_safe);
        if (pick == 5) {
            Gen g = Numeric(depth - 1, need_div_safe, need_truth_safe);
            g.e = UnE(UnaryOp::Op::NEGATE, std::move(g.e));
            return g;
        }
        if (pick < 8) return NumericCase(depth, need_div_safe, need_truth_safe);
        if (pick == 8) {
            Gen g = Bool(depth - 1);
            g.bound = 1;
            return g;
        }
        return Chance(0.5) ? GuardedCase() : NumericLeaf();
    }

    // An integer column that holds zeros, and a numerator for dividing by it.
    uint32_t Divisor() { return Chance(0.5) ? kI : kJ; }
    ExprPtr Numerator(bool big) {
        if (big) return ColE(kB);
        return Chance(0.5) ? ColE(kI) : LitE(Value(static_cast<int32_t>(Pick(41) - 20)));
    }

    // A guard keeps zero rows from the division: `c <> 0 AND x / c > k`
    // or `c = 0 OR ...`. A guard placed second lets today throw.
    ExprPtr Guarded() {
        static const Op kCmp[] = {Op::EQ, Op::NEQ, Op::LT, Op::GT, Op::LTE, Op::GTE};
        const uint32_t c = Divisor();
        ExprPtr test = BinE(kCmp[Pick(6)], BinE(Op::DIV, Numerator(Chance(0.3)), ColE(c)),
                            LitE(Value(static_cast<int32_t>(Pick(9) - 4))));
        const bool use_and = Chance(0.5);
        ExprPtr guard = BinE(use_and ? Op::NEQ : Op::EQ, ColE(c), LitE(Value(int32_t{0})));
        if (Chance(0.2)) std::swap(guard, test);
        return BinE(use_and ? Op::AND : Op::OR, std::move(guard), std::move(test));
    }

    // CASE WHEN c <> 0 THEN x / c: the division sees only its rows.
    // Both results share one type, so truthiness and division see what today sees.
    Gen GuardedCase() {
        const uint32_t c = Divisor();
        const bool big = Chance(0.5);
        auto e = std::make_unique<CaseExpression>();
        if (Chance(0.5)) {
            e->when_clauses.emplace_back(BinE(Op::NEQ, ColE(c), LitE(Value(int32_t{0}))), BinE(Op::DIV, Numerator(big), ColE(c)));
            e->else_clause = ColE(big ? kB : kJ);
        } else {
            e->when_clauses.emplace_back(BinE(Op::EQ, ColE(c), LitE(Value(int32_t{0}))), ColE(big ? kB : kJ));
            e->else_clause = BinE(Op::DIV, Numerator(big), ColE(c));
        }
        Gen g;
        g.e = std::move(e);
        g.type = big ? G::BIG : G::INT;
        g.bound = big ? 500 : 20;
        return g;
    }

    // Tries a few random operands; falls back to a leaf when none fit the limits.
    Gen Arithmetic(int depth, bool need_div_safe, bool need_truth_safe) {
        static const Op kArith[] = {Op::ADD, Op::SUB, Op::MUL, Op::DIV};
        for (int attempt = 0; attempt < 4; ++attempt) {
            const Op op = kArith[Pick(4)];
            Gen a = Numeric(depth - 1, need_div_safe || op == Op::DIV, need_truth_safe);
            Gen b = Numeric(depth - 1, need_div_safe || op == Op::DIV, need_truth_safe);
            Gen out;
            if (Combine(op, a, b, out)) return out;
        }
        return NumericLeaf();
    }

    // a op b, or false when it could overflow or drift too far.
    static bool Combine(Op op, Gen& a, Gen& b, Gen& out) {
        if (a.terminal || b.terminal) return false;
        if (op == Op::DIV) {
            if (!a.div_safe || !b.div_safe) return false;
            if (IsIntFamily(a.type) && IsIntFamily(b.type)) {
                out.type = (a.type == G::BIG || b.type == G::BIG) ? G::BIG : G::INT;
                out.bound = a.bound;
            } else {
                if (!a.same || !b.same) return false;  // a quotient of drifted values could blow up
                out.type = G::DBL;
                out.bound = std::numeric_limits<double>::infinity();
                out.terminal = true;
            }
        } else {
            out.type = ArithFamily(op, a.type, b.type);
            out.bound = op == Op::MUL ? a.bound * b.bound : a.bound + b.bound;
            if (out.type == G::INT && out.bound > 1e9) return false;   // int32 must not overflow today
            if (out.type == G::BIG && out.bound > 1e17) return false;
            if (!IsIntFamily(out.type)) {
                out.err = op == Op::MUL ? a.bound * b.err + b.bound * a.err + a.err * b.err + out.bound * kUnit
                                        : a.err + b.err + out.bound * kUnit;
                if (out.err > kBudget) return false;
                out.same = out.type == G::DBL && a.same && b.same;
                out.scale = op == Op::MUL ? a.scale + b.scale : std::max(a.scale, b.scale);
                if (out.scale > 24 || out.bound * std::pow(10.0, out.scale) > 1e28) return false;
            }
        }
        out.div_safe = a.div_safe && b.div_safe;
        out.truth_safe = a.truth_safe && b.truth_safe;
        out.type_exact = a.type_exact && b.type_exact;
        out.e = BinE(op, std::move(a.e), std::move(b.e));
        return true;
    }

    Gen NumericCase(int depth, bool need_div_safe, bool need_truth_safe) {
        Branches allowed = Branches::ANY;
        if (need_div_safe && need_truth_safe) {
            const Branches options[] = {Branches::INT_ONLY, Branches::BIG_ONLY, Branches::FRACTIONAL};
            allowed = options[Pick(3)];
        } else if (need_div_safe) {
            allowed = Chance(0.5) ? Branches::INT_FAMILY : Branches::FRACTIONAL;
        } else if (need_truth_safe) {
            allowed = Chance(0.5) ? Branches::INT_ONLY : Branches::NOT_INT;
        }

        auto branch = [&]() {
            for (int attempt = 0; attempt < 6; ++attempt) {
                Gen g = Numeric(depth - 1, need_div_safe, need_truth_safe);
                if (Allowed(allowed, g.type)) return g;
            }
            for (G g : {G::INT, G::BIG, G::NUM, G::DBL}) {
                if (Allowed(allowed, g)) return LeafOf(g);
            }
            return LeafOf(G::INT);
        };

        auto c = std::make_unique<CaseExpression>();
        std::vector<Gen> results;
        const int whens = 1 + Pick(2);
        for (int w = 0; w < whens; ++w) {
            ExprPtr cond = Condition(depth - 1);
            results.push_back(branch());
            c->when_clauses.emplace_back(std::move(cond), std::move(results.back().e));
        }
        // A missing ELSE gives today's INTEGER 0, so a dividing fractional CASE keeps one.
        const bool has_else = allowed == Branches::FRACTIONAL || Chance(0.7);
        if (has_else) {
            results.push_back(branch());
            c->else_clause = std::move(results.back().e);
        }

        Gen out;
        bool any_int = false, any_big = false, any_num = false, any_dbl = false;
        for (const Gen& r : results) {
            any_int |= r.type == G::INT;
            any_big |= r.type == G::BIG;
            any_num |= r.type == G::NUM;
            any_dbl |= r.type == G::DBL;
            out.bound = std::max(out.bound, r.bound);
            out.err = std::max(out.err, r.err);
            out.scale = std::max(out.scale, r.scale);
            out.same = out.same && r.same;
            out.terminal = out.terminal || r.terminal;
            out.div_safe = out.div_safe && r.div_safe;
            out.truth_safe = out.truth_safe && r.truth_safe;
        }
        out.type = any_dbl ? G::DBL : any_num ? G::NUM : any_big ? G::BIG : G::INT;
        const bool int_rows = any_int || any_big || !has_else;
        if (int_rows && (any_num || any_dbl)) out.div_safe = false;
        if (any_int && out.type != G::INT) out.truth_safe = false;
        // Today each row keeps its result's type; a missing ELSE is an INTEGER 0.
        for (const Gen& r : results) {
            out.type_exact = out.type_exact && r.type_exact && TypeIdOf(r.type) == TypeIdOf(out.type);
        }
        if (!has_else && out.type != G::INT) out.type_exact = false;
        out.e = std::move(c);
        return out;
    }

    // A condition for AND, OR, NOT or WHEN: usually a predicate, sometimes a number.
    ExprPtr Condition(int depth) {
        if (Chance(0.8)) return Bool(depth).e;
        return Numeric(depth, false, true).e;
    }

    Gen Bool(int depth) {
        Gen g;
        if (depth <= 0) {
            g.e = Pick(4) == 0 ? InList() : Comparison(0);
            return g;
        }
        switch (Pick(10)) {
            case 0:
            case 1:
            case 2:
            case 3:
                g.e = Comparison(depth);
                break;
            case 4:
                g.e = Like(depth);
                break;
            case 5:
                g.e = Chance(0.5) ? InList() : Guarded();
                break;
            case 6:
            case 7:
                g.e = BinE(Chance(0.5) ? Op::AND : Op::OR, Condition(depth - 1), Condition(depth - 1));
                break;
            default:
                g.e = UnE(UnaryOp::Op::NOT, Condition(depth - 1));
                break;
        }
        return g;
    }

    ExprPtr Comparison(int depth) {
        static const Op kCmp[] = {Op::EQ, Op::NEQ, Op::LT, Op::GT, Op::LTE, Op::GTE};
        const Op op = kCmp[Pick(6)];
        const int family = Pick(10);
        ExprPtr l, r;
        if (family < 6) {
            l = Numeric(depth - 1, false, false).e;
            r = Numeric(depth - 1, false, false).e;
        } else if (family < 8) {
            l = Date(depth - 1).e;
            if (Chance(0.2)) {
                // A date written as text; the compiler converts it once.
                r = LitE(Value(Value::MakeDate(9000 + Pick(61)).ToString()));
            } else {
                r = Date(depth - 1).e;
            }
        } else {
            l = Str(depth - 1).e;
            r = Str(depth - 1).e;
        }
        if (Chance(0.5)) std::swap(l, r);
        Widen(*l, *r, schema_);
        Widen(*r, *l, schema_);
        return BinE(op, std::move(l), std::move(r));
    }

    std::string Piece() {
        static const char* const kPieces[] = {"a", "b", "c", "\xC3\xA9"};
        return kPieces[Pick(4)];
    }

    std::string Pattern() {
        std::string x;
        const int len = Pick(3);
        for (int k = 0; k < len; ++k) x += Piece();
        switch (Pick(5)) {
            case 0:
                return x + "%";
            case 1:
                return "%" + x;
            case 2:
                return "%" + x + "%";
            case 3:
                return x;
            default: {
                static const char* const kAny[] = {"a", "b", "%", "_", "\xC3\xA9"};
                std::string p;
                const int n = Pick(5);
                for (int k = 0; k < n; ++k) p += kAny[Pick(5)];
                return p;
            }
        }
    }

    ExprPtr Like(int depth) {
        ExprPtr l = Chance(0.6) ? ColE(kS) : Str(depth - 1).e;
        ExprPtr r = Chance(0.85) ? LitE(Value(Pattern())) : Str(depth - 1).e;
        return BinE(Op::LIKE, std::move(l), std::move(r));
    }

    // `column IN (...)` as the parser builds it, with the binder's literal widening.
    ExprPtr InList() {
        static const uint32_t kColumns[] = {kI, kB, kN, kD, kDt, kS};
        const uint32_t c = kColumns[Pick(6)];
        std::vector<ExprPtr> values;
        const int count = 2 + Pick(3);
        for (int k = 0; k < count; ++k) {
            switch (c) {
                case kI:
                    values.push_back(LitE(Value(static_cast<int32_t>(Pick(30) - 5))));
                    break;
                case kB:
                    values.push_back(Chance(0.5) ? LitE(Value(static_cast<int32_t>(Pick(1001) - 500)))
                                                 : LitE(Value(static_cast<int64_t>(Pick(1001) - 500))));
                    break;
                case kN:
                    if (Chance(0.3)) {
                        values.push_back(LitE(Value(static_cast<int32_t>(Pick(21) - 10))));
                    } else {
                        const double units = static_cast<double>(Pick(2001) - 1000);
                        values.push_back(LitE(Value(Chance(0.2) ? units / 1000.0 : units / 100.0)));
                    }
                    break;
                case kD:
                    values.push_back(LitE(Value(static_cast<double>(Pick(161) - 80) / 8.0)));
                    break;
                case kDt:
                    values.push_back(LitE(Value::MakeDate(9000 + Pick(61))));
                    break;
                default: {
                    std::string s;
                    const int len = Pick(3);
                    for (int k2 = 0; k2 < len; ++k2) s += Piece();
                    values.push_back(LitE(Value(s)));
                    break;
                }
            }
        }
        auto leaf = [&](ExprPtr v) {
            ExprPtr col = ColE(c);
            Widen(*col, *v, schema_);
            return BinE(Op::EQ, std::move(col), std::move(v));
        };
        ExprPtr result = leaf(std::move(values.back()));
        values.pop_back();
        while (!values.empty()) {
            ExprPtr next = leaf(std::move(values.back()));
            values.pop_back();
            result = BinE(Op::OR, std::move(next), std::move(result));
        }
        return result;
    }

    Gen Date(int depth) {
        Gen g;
        g.type = G::DATE;
        if (depth <= 0 || Chance(0.6)) {
            g.e = Chance(0.5) ? ColE(kDt) : LitE(Value::MakeDate(9000 + Pick(61)));
            return g;
        }
        auto c = std::make_unique<CaseExpression>();
        const int whens = 1 + Pick(2);
        for (int w = 0; w < whens; ++w) c->when_clauses.emplace_back(Condition(depth - 1), Date(depth - 1).e);
        c->else_clause = Date(depth - 1).e;
        g.e = std::move(c);
        return g;
    }

    Gen Str(int depth) {
        Gen g;
        g.type = G::STR;
        if (depth <= 0 || Chance(0.6)) {
            if (Chance(0.6)) {
                g.e = ColE(kS);
            } else {
                std::string s;
                const int len = Pick(3);
                for (int k = 0; k < len; ++k) s += Piece();
                g.e = LitE(Value(s));
            }
            return g;
        }
        auto c = std::make_unique<CaseExpression>();
        const int whens = 1 + Pick(2);
        for (int w = 0; w < whens; ++w) c->when_clauses.emplace_back(Condition(depth - 1), Str(depth - 1).e);
        c->else_clause = Str(depth - 1).e;
        g.e = std::move(c);
        return g;
    }

    std::mt19937& rng_;
    const Schema& schema_;
};

// ---- Checking one tree ----

// True when the tree reads a NUMERIC column or holds a decimal literal.
bool TouchesDecimal(const Expression& e) {
    switch (e.type) {
        case ExprType::COLUMN_REF: {
            const auto c = *static_cast<const ColumnRef&>(e).bound_index;
            return c == kN || c == kM;
        }
        case ExprType::LITERAL:
            return static_cast<const Literal&>(e).value.type_ == TypeId::DECIMAL;
        case ExprType::BINARY_OP: {
            const auto& b = static_cast<const BinaryOp&>(e);
            return TouchesDecimal(*b.left) || TouchesDecimal(*b.right);
        }
        case ExprType::UNARY_OP:
            return TouchesDecimal(*static_cast<const UnaryOp&>(e).operand);
        case ExprType::CASE: {
            const auto& c = static_cast<const CaseExpression&>(e);
            for (const auto& [cond, result] : c.when_clauses) {
                if (TouchesDecimal(*cond) || TouchesDecimal(*result)) return true;
            }
            return c.else_clause && TouchesDecimal(*c.else_clause);
        }
        default:
            return false;
    }
}

bool IsNumber(const Value& v) {
    return v.type_ == TypeId::INTEGER || v.type_ == TypeId::BIGINT || v.type_ == TypeId::DECIMAL;
}

double AsDouble(const Value& v) {
    switch (v.type_) {
        case TypeId::INTEGER:
            return v.integer_;
        case TypeId::BIGINT:
            return static_cast<double>(v.bigint_);
        default:
            return v.decimal_;
    }
}

int64_t AsInt(const Value& v) {
    return v.type_ == TypeId::INTEGER ? v.integer_ : v.bigint_;
}

// Within 1e-12 relative to max(1, |a|, |b|). Infinities and NaN only match themselves.
bool Close(double a, double b) {
    if (!std::isfinite(a) || !std::isfinite(b)) return a == b || (std::isnan(a) && std::isnan(b));
    return std::fabs(a - b) <= kTolerance * std::max({1.0, std::fabs(a), std::fabs(b)});
}

bool IsComparison(Op op) {
    return op == Op::EQ || op == Op::NEQ || op == Op::LT || op == Op::GT || op == Op::LTE || op == Op::GTE;
}

bool IsArithmetic(Op op) {
    return op == Op::ADD || op == Op::SUB || op == Op::MUL || op == Op::DIV;
}

// The family the compiler types an expression as.
G FamilyOf(const Expression& e) {
    switch (e.type) {
        case ExprType::COLUMN_REF:
            switch (*static_cast<const ColumnRef&>(e).bound_index) {
                case kI:
                case kJ:
                    return G::INT;
                case kB:
                    return G::BIG;
                case kN:
                case kM:
                    return G::NUM;
                case kD:
                    return G::DBL;
                case kDt:
                    return G::DATE;
                default:
                    return G::STR;
            }
        case ExprType::LITERAL: {
            const Value& v = static_cast<const Literal&>(e).value;
            switch (v.type_) {
                case TypeId::INTEGER:
                    return G::INT;
                case TypeId::BIGINT:
                    return G::BIG;
                case TypeId::DECIMAL:
                    return DecimalFromDouble(v.decimal_) ? G::NUM : G::DBL;
                case TypeId::DATE:
                    return G::DATE;
                default:
                    return G::STR;
            }
        }
        case ExprType::BINARY_OP: {
            const auto& b = static_cast<const BinaryOp&>(e);
            if (!IsArithmetic(b.op)) return G::INT;
            return ArithFamily(b.op, FamilyOf(*b.left), FamilyOf(*b.right));
        }
        case ExprType::UNARY_OP: {
            const auto& u = static_cast<const UnaryOp&>(e);
            return u.op == UnaryOp::Op::NOT ? G::INT : FamilyOf(*u.operand);
        }
        case ExprType::CASE: {
            const auto& c = static_cast<const CaseExpression&>(e);
            std::vector<G> results;
            for (const auto& [cond, result] : c.when_clauses) results.push_back(FamilyOf(*result));
            if (c.else_clause) results.push_back(FamilyOf(*c.else_clause));
            for (G g : {G::DBL, G::NUM, G::BIG, G::DATE, G::STR}) {
                if (std::find(results.begin(), results.end(), g) != results.end()) return g;
            }
            return G::INT;
        }
        default:
            return G::INT;
    }
}

// True when the compiled value can differ from today's by rounding: it holds an
// exact decimal sum or product somewhere that today computes in doubles.
bool Drifts(const Expression& e) {
    switch (e.type) {
        case ExprType::BINARY_OP: {
            const auto& b = static_cast<const BinaryOp&>(e);
            if (!IsArithmetic(b.op)) return false;  // 1 or 0; a flip is judged at its own comparison
            if (b.op != Op::DIV && FamilyOf(e) == G::NUM) return true;
            return Drifts(*b.left) || Drifts(*b.right);
        }
        case ExprType::UNARY_OP: {
            const auto& u = static_cast<const UnaryOp&>(e);
            return u.op == UnaryOp::Op::NEGATE && Drifts(*u.operand);
        }
        case ExprType::CASE: {
            const auto& c = static_cast<const CaseExpression&>(e);
            for (const auto& [cond, result] : c.when_clauses) {
                if (Drifts(*result)) return true;
            }
            return c.else_clause && Drifts(*c.else_clause);
        }
        default:
            return false;
    }
}

// The compiled value of one row, as the Value today's executor would hold.
Value OurValue(const Vector& out, uint32_t row) {
    const VType t = out.Type();
    switch (t.kind) {
        case Kind::INTEGER:
            return t.result == TypeId::INTEGER ? Value(static_cast<int32_t>(out.I64()[row])) : Value(out.I64()[row]);
        case Kind::DATE:
            return Value::MakeDate(static_cast<int32_t>(out.I64()[row]));
        case Kind::DECIMAL:
            return Value(DecimalToDouble(t.wide ? out.I128()[row] : out.I64()[row], t.scale));
        case Kind::DOUBLE:
            return Value(out.F64()[row]);
        case Kind::STRING:
            return Value(std::string(View(out.StrAt(row))));
    }
    return Value();
}

std::string Describe(const Value& v) {
    char buf[64];
    if (v.type_ == TypeId::DECIMAL) {
        std::snprintf(buf, sizeof(buf), "%.17g (DECIMAL)", v.decimal_);
        return buf;
    }
    return v.ToString() + " (type " + std::to_string(static_cast<int>(v.type_)) + ")";
}

struct Stats {
    uint64_t trees{0};
    uint64_t compared{0};         // row checks, values and errors
    uint64_t errors{0};           // rows where today throws
    uint64_t selects{0};          // rows checked through Select
    uint64_t tolerant_values{0};  // doubles accepted as close rather than equal
    uint64_t forgiven_flips{0};   // mismatches explained by a comparison flipped at a near tie
    double max_gap{0};            // the largest |compiled - today| accepted
    double max_gap_share{0};      // the largest accepted gap over its tree's drift bound
    int failures{0};
};

struct Outcome {
    bool threw{false};
    std::string error;
    Value value;
};

Selection MakeSelection(const std::vector<uint16_t>& rows, bool all) {
    Selection sel;
    if (all) {
        sel.SetAll(static_cast<uint32_t>(rows.size()));
        return sel;
    }
    std::copy(rows.begin(), rows.end(), sel.Data());
    sel.Resize(static_cast<uint32_t>(rows.size()));
    return sel;
}

class TreeChecker {
public:
    TreeChecker(const Schema& schema, const std::vector<Tuple>& tuples, const Fixture& fx, Stats& stats)
        : schema_(schema), tuples_(tuples), fx_(fx), stats_(stats) {}

    void Check(const Gen& tree, const std::vector<uint16_t>& subset, const std::string& label) {
        ++stats_.trees;
        expr_ = tree.e.get();
        label_ = label;
        today_.assign(kRows, Outcome{});
        for (uint32_t r = 0; r < kRows; ++r) {
            try {
                today_[r].value = EvaluateExpression(expr_, tuples_[r], schema_);
            } catch (const std::exception& e) {
                today_[r].threw = true;
                today_[r].error = e.what();
            }
        }
        try {
            compiled_ = CompiledExpr::Compile(*expr_, fx_.layout);
        } catch (const std::exception& e) {
            Fail(0, std::string("compile threw: ") + e.what());
            return;
        }
        tolerant_ = TouchesDecimal(*expr_);
        same_ = tree.same;
        drift_ = 2 * tree.err;
        type_exact_ = tree.type_exact;

        std::vector<uint16_t> all(kRows);
        std::iota(all.begin(), all.end(), uint16_t{0});
        CheckRows(all, true, tree.truth_safe);
        CheckRows(subset, false, tree.truth_safe);
    }

private:
    void CheckRows(const std::vector<uint16_t>& rows, bool all, bool check_select) {
        std::vector<uint16_t> ok, bad;
        for (uint16_t r : rows) (today_[r].threw ? bad : ok).push_back(r);

        // Only DatabaseException is expected from the compiled side; anything else fails the test.
        EvalScratch scratch;
        Vector out(compiled_->Type());
        std::vector<std::string> ours_error(kRows);
        std::vector<bool> ours_threw(kRows, false);
        try {
            const Selection sel = MakeSelection(ok, all && bad.empty());
            compiled_->Evaluate(fx_.batch, sel, out, scratch);
        } catch (const DatabaseException&) {
            // Find the rows that throw, one at a time.
            std::vector<uint16_t> rest;
            for (uint16_t r : ok) {
                try {
                    compiled_->Evaluate(fx_.batch, MakeSelection({r}, false), out, scratch);
                    rest.push_back(r);
                } catch (const DatabaseException& e) {
                    ours_threw[r] = true;
                    ours_error[r] = e.what();
                }
            }
            // Without them the batch must not throw: kernels throw only for their rows.
            try {
                compiled_->Evaluate(fx_.batch, MakeSelection(rest, false), out, scratch);
            } catch (const DatabaseException& e) {
                Fail(0, std::string("the batch threw '") + e.what() + "' though no row alone does");
            }
        }

        std::vector<bool> forgiven(kRows, false);
        auto mismatch = [&](uint16_t r, const std::string& what) {
            if (!forgiven[r] && FlippedTie(*expr_, r)) {
                forgiven[r] = true;
                ++stats_.forgiven_flips;
            }
            if (!forgiven[r]) Fail(r, what);
        };

        for (uint16_t r : ok) {
            ++stats_.compared;
            if (ours_threw[r]) {
                mismatch(r, "compiled threw '" + ours_error[r] + "', today gave " + Describe(today_[r].value));
                continue;
            }
            const Value got = OurValue(out, r);
            if (!Matches(today_[r].value, got)) {
                mismatch(r, "compiled " + Describe(got) + ", today " + Describe(today_[r].value));
            }
        }

        for (uint16_t r : bad) {
            ++stats_.compared;
            ++stats_.errors;
            std::string error;
            try {
                compiled_->Evaluate(fx_.batch, MakeSelection({r}, false), out, scratch);
            } catch (const DatabaseException& e) {
                error = e.what();
            }
            if (error != today_[r].error) {
                mismatch(r, "compiled error '" + error + "', today '" + today_[r].error + "'");
            }
        }

        if (!check_select) return;
        std::vector<uint16_t> live;
        for (uint16_t r : ok) {
            if (!ours_threw[r]) live.push_back(r);
        }
        Selection sel = MakeSelection(live, all && live.size() == kRows);
        try {
            compiled_->Select(fx_.batch, sel, scratch);
        } catch (const DatabaseException& e) {
            Fail(0, std::string("Select threw: ") + e.what());
            return;
        }
        std::vector<bool> kept(kRows, false);
        for (uint32_t k = 0; k < sel.Size(); ++k) kept[sel[k]] = true;
        for (uint16_t r : live) {
            ++stats_.selects;
            if (kept[r] != IsTruthy(today_[r].value)) {
                mismatch(r, std::string("Select ") + (kept[r] ? "kept" : "dropped") + " a row today calls " +
                                Describe(today_[r].value));
            }
        }
    }

    // Same TypeId (unless a CASE's static type is in play) and same value. Doubles may
    // differ only where an exact decimal met today's doubles, within the tree's drift bound.
    bool Matches(const Value& want, const Value& got) {
        if (type_exact_ && want.type_ != got.type_) return false;
        if (IsNumber(want) && IsNumber(got)) {
            if (want.type_ != TypeId::DECIMAL && got.type_ != TypeId::DECIMAL) return AsInt(want) == AsInt(got);
            const double a = AsDouble(want);
            const double b = AsDouble(got);
            if (a == b || (std::isnan(a) && std::isnan(b))) return true;
            if (!tolerant_ || same_ || !Close(a, b) || std::fabs(a - b) > drift_) return false;
            ++stats_.tolerant_values;
            stats_.max_gap = std::max(stats_.max_gap, std::fabs(a - b));
            stats_.max_gap_share = std::max(stats_.max_gap_share, std::fabs(a - b) / drift_);
            return true;
        }
        if (want.type_ != got.type_) return false;
        if (want.type_ == TypeId::DATE) return want.date_ == got.date_;
        return want.varchar_ == got.varchar_;
    }

    // True when some comparison flipped at a near tie on this row: today sees its
    // sides within the tolerance, a side carries decimal rounding, and the results differ.
    bool FlippedTie(const Expression& e, uint16_t row) {
        switch (e.type) {
            case ExprType::BINARY_OP: {
                const auto& b = static_cast<const BinaryOp&>(e);
                if (IsComparison(b.op) && (Drifts(*b.left) || Drifts(*b.right)) && Flipped(b, row)) return true;
                return FlippedTie(*b.left, row) || FlippedTie(*b.right, row);
            }
            case ExprType::UNARY_OP:
                return FlippedTie(*static_cast<const UnaryOp&>(e).operand, row);
            case ExprType::CASE: {
                const auto& c = static_cast<const CaseExpression&>(e);
                for (const auto& [cond, result] : c.when_clauses) {
                    if (FlippedTie(*cond, row) || FlippedTie(*result, row)) return true;
                }
                return c.else_clause && FlippedTie(*c.else_clause, row);
            }
            default:
                return false;
        }
    }

    // One comparison, near-tied today, whose compiled answer on this row differs.
    bool Flipped(const BinaryOp& cmp, uint16_t row) {
        Value l, r, whole;
        try {
            l = EvaluateExpression(cmp.left.get(), tuples_[row], schema_);
            r = EvaluateExpression(cmp.right.get(), tuples_[row], schema_);
            whole = EvaluateExpression(&cmp, tuples_[row], schema_);
        } catch (const std::exception&) {
            return false;  // a side today never reaches on this row
        }
        if (!IsNumber(l) || !IsNumber(r) || !Close(AsDouble(l), AsDouble(r))) return false;
        const std::unique_ptr<CompiledExpr> compiled = CompiledExpr::Compile(cmp, fx_.layout);
        EvalScratch scratch;
        Vector out(compiled->Type());
        try {
            compiled->Evaluate(fx_.batch, MakeSelection({row}, false), out, scratch);
        } catch (const DatabaseException&) {
            return false;
        }
        return (out.I64()[row] != 0) != IsTruthy(whole);
    }

    void Fail(uint16_t row, const std::string& what) {
        if (++stats_.failures > kMaxFailures) return;
        ADD_FAILURE() << label_ << " row " << row << ": " << what << "\n  tree: " << Show(*expr_);
    }

    const Schema& schema_;
    const std::vector<Tuple>& tuples_;
    const Fixture& fx_;
    Stats& stats_;
    const Expression* expr_{nullptr};
    std::string label_;
    std::vector<Outcome> today_;
    std::unique_ptr<CompiledExpr> compiled_;
    bool tolerant_{false};    // the tree touches a decimal
    bool same_{true};         // the compiled double must equal today's exactly
    double drift_{0};         // most the root may differ from today's
    bool type_exact_{true};   // TypeIds must match too
};

}  // namespace

TEST(ParallelExpressionDifferentialTest, CompiledMatchesEvaluateExpression) {
    const Schema schema = MakeSchema();
    Stats stats;
    for (uint32_t seed = 1; seed <= kSeeds; ++seed) {
        std::mt19937 rng(seed);
        const Data data = MakeData(rng);
        const std::vector<Tuple> tuples = MakeTuples(data, schema);
        Fixture fx;
        MakeBatch(data, rng, seed % 2 == 0, fx);

        std::vector<uint16_t> subset;
        for (uint16_t r = 0; r < kRows; ++r) {
            if (rng() % 2 == 0) subset.push_back(r);
        }

        TreeGen gen(rng, schema);
        TreeChecker checker(schema, tuples, fx, stats);
        for (int t = 0; t < kTreesPerSeed; ++t) {
            const Gen tree = gen.Root();
            checker.Check(tree, subset, "seed " + std::to_string(seed) + " tree " + std::to_string(t));
        }
    }

    std::printf("[ differential ] %llu trees, %llu row checks (%llu today errors), %llu Select rows, "
                "%llu tolerant values (largest gap %.3g, %.3g of its bound), %llu forgiven flips\n",
                static_cast<unsigned long long>(stats.trees), static_cast<unsigned long long>(stats.compared),
                static_cast<unsigned long long>(stats.errors), static_cast<unsigned long long>(stats.selects),
                static_cast<unsigned long long>(stats.tolerant_values), stats.max_gap, stats.max_gap_share,
                static_cast<unsigned long long>(stats.forgiven_flips));
    EXPECT_EQ(stats.failures, 0);
    EXPECT_GT(stats.errors, 0u);  // the error path ran
    EXPECT_LE(stats.forgiven_flips, kMaxForgivenFlips);
}

}  // namespace shilmandb::parallel
