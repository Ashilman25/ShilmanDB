#include "executor/parallel/expression.hpp"
#include "common/exception.hpp"
#include "executor/parallel/decimal.hpp"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace shilmandb::parallel {


// Scratch: temporaries borrowed for one call and handed back after

namespace {

// Which array a vector keeps its values in; scratch vectors are shared by storage.
enum Storage : int { 
    kI64 = 0, 
    kI128 = 1, 
    kF64 = 2, 
    kStr = 3 
};

// INTEGER, DATE and narrow DECIMAL share the 64-bit array.
Storage StorageOf(const VType& t) {
    switch (t.kind) {
        case Kind::INTEGER:
        case Kind::DATE:
            return kI64;
        case Kind::DECIMAL:
            return t.wide ? kI128 : kI64;
        case Kind::DOUBLE:
            return kF64;
        case Kind::STRING:
            return kStr;
    }
    return kI64;
}

}  // namespace

// Hands EvalScratch's temporaries out and takes them back.
struct ScratchAccess {
    // A free vector with the same storage as t, or a new one.
    static std::unique_ptr<Vector> TakeVector(EvalScratch& s, const VType& t) {
        auto& pool = s.vectors_[StorageOf(t)];
        if (pool.empty()) return std::make_unique<Vector>(t);
        std::unique_ptr<Vector> v = std::move(pool.back());
        pool.pop_back();
        return v;
    }

    // Takes a vector back, in plain form.
    static void GiveVector(EvalScratch& s, std::unique_ptr<Vector> v) {
        if (v->Type().kind == Kind::STRING) v->SetDictionary(nullptr);
        s.vectors_[StorageOf(v->Type())].push_back(std::move(v));
    }

    // Selections and tables work the same way.
    static std::unique_ptr<Selection> TakeSelection(EvalScratch& s) {
        if (s.selections_.empty()) return std::make_unique<Selection>();
        std::unique_ptr<Selection> sel = std::move(s.selections_.back());
        s.selections_.pop_back();
        return sel;
    }

    static void GiveSelection(EvalScratch& s, std::unique_ptr<Selection> sel) {
        s.selections_.push_back(std::move(sel));
    }

    static std::unique_ptr<std::vector<uint8_t>> TakeTable(EvalScratch& s) {
        if (s.tables_.empty()) return std::make_unique<std::vector<uint8_t>>();
        std::unique_ptr<std::vector<uint8_t>> t = std::move(s.tables_.back());
        s.tables_.pop_back();
        return t;
    }

    static void GiveTable(EvalScratch& s, std::unique_ptr<std::vector<uint8_t>> t) {
        s.tables_.push_back(std::move(t));
    }
};

namespace {

// A scratch vector borrowed on first use, handed back when the scope ends.
class TempVector {
public:
    explicit TempVector(EvalScratch& s) : s_(s) {}
    ~TempVector() {
        if (v_) ScratchAccess::GiveVector(s_, std::move(v_));
    }
    TempVector(const TempVector&) = delete;
    TempVector& operator=(const TempVector&) = delete;

    // The borrowed vector; its storage matches t.
    Vector& Get(const VType& t) {
        if (!v_) v_ = ScratchAccess::TakeVector(s_, t);
        assert(StorageOf(v_->Type()) == StorageOf(t));
        return *v_;
    }

private:
    EvalScratch& s_;
    std::unique_ptr<Vector> v_;
};

// A scratch selection for the length of a scope.
class TempSelection {
public:
    explicit TempSelection(EvalScratch& s) : s_(s), sel_(ScratchAccess::TakeSelection(s)) {}
    ~TempSelection() { ScratchAccess::GiveSelection(s_, std::move(sel_)); }
    TempSelection(const TempSelection&) = delete;
    TempSelection& operator=(const TempSelection&) = delete;

    Selection& operator*() { return *sel_; }

private:
    EvalScratch& s_;
    std::unique_ptr<Selection> sel_;
};

// A scratch byte table (one entry per dictionary entry) for the length of a scope.
class TempTable {
public:
    explicit TempTable(EvalScratch& s) : s_(s), t_(ScratchAccess::TakeTable(s)) {}
    ~TempTable() { ScratchAccess::GiveTable(s_, std::move(t_)); }
    TempTable(const TempTable&) = delete;
    TempTable& operator=(const TempTable&) = delete;

    std::vector<uint8_t>& operator*() { return *t_; }

private:
    EvalScratch& s_;
    std::unique_ptr<std::vector<uint8_t>> t_;
};


// Selection helpers

// Calls fn(row) for every active row, skipping the index array for "all".
template <typename Fn>
void ForRows(const Selection& rows, Fn fn) {
    const uint32_t n = rows.Size();
    if (rows.IsAll()) {
        for (uint32_t i = 0; i < n; ++i) fn(i);
    } else {
        const uint16_t* idx = rows.Data();
        for (uint32_t k = 0; k < n; ++k) fn(static_cast<uint32_t>(idx[k]));
    }
}

// Keeps the first `kept` rows; an "all" selection nothing left stays "all".
void Narrow(Selection& sel, uint32_t kept) {
    if (!sel.IsAll() || kept != sel.Size()) sel.Resize(kept);
}

// Keeps the rows where keep(row) is true, compacting in place.
template <typename Keep>
void FilterRows(Selection& sel, Keep keep) {
    uint16_t* idx = sel.Data();
    const uint32_t n = sel.Size();
    uint32_t kept = 0;
    if (sel.IsAll()) {
        for (uint32_t i = 0; i < n; ++i) {
            idx[kept] = static_cast<uint16_t>(i);
            kept += keep(i) ? 1 : 0;
        }
    } else {
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t i = idx[k];
            idx[kept] = static_cast<uint16_t>(i);
            kept += keep(i) ? 1 : 0;
        }
    }
    Narrow(sel, kept);
}

// Copies from into to, keeping the "all" flag.
void CopySelection(const Selection& from, Selection& to) {
    if (from.IsAll()) {
        to.SetAll(from.Size());
        return;
    }
    std::memcpy(to.Data(), from.Data(), from.Size() * sizeof(uint16_t));
    to.Resize(from.Size());
}

// out = all minus part, where part is inside all. out may be all itself.
void Subtract(const Selection& all, const Selection& part, Selection& out) {
    const uint16_t* p = part.Data();
    const uint32_t pn = part.Size();
    const uint32_t total = all.Size();
    const bool was_all = all.IsAll();
    uint16_t* o = out.Data();
    uint32_t n = 0;
    uint32_t j = 0;

    ForRows(all, [&](uint32_t i) {
        if (j < pn && p[j] == i) {
            ++j;
        } else {
            o[n++] = static_cast<uint16_t>(i);
        }
    });

    if (was_all && n == total) {
        out.SetAll(n);
    } else {
        out.Resize(n);
    }
}

// Writes the union of two disjoint ascending selections into sel, which held both.
void MergeInto(const Selection& a, const Selection& b, Selection& sel) {
    const uint16_t* pa = a.Data();
    const uint16_t* pb = b.Data();
    const uint32_t na = a.Size();
    const uint32_t nb = b.Size();
    uint16_t* o = sel.Data();
    uint32_t i = 0;
    uint32_t j = 0;
    uint32_t n = 0;
    while (i < na && j < nb) o[n++] = pa[i] < pb[j] ? pa[i++] : pb[j++];
    while (i < na) o[n++] = pa[i++];
    while (j < nb) o[n++] = pb[j++];
    Narrow(sel, n);
}

// Small helpers

// Only an INTEGER whose result is INTEGER can be true, as IsTruthy says.
bool CanBeTrue(const VType& t) {
    return t.kind == Kind::INTEGER && t.result == TypeId::INTEGER;
}

// Digits in |v|, at least 1.
uint8_t Digits(__int128 v) {
    unsigned __int128 m = v < 0 ? -static_cast<unsigned __int128>(v) : static_cast<unsigned __int128>(v);
    uint8_t d = 1;
    while (m >= 10) {
        m /= 10;
        ++d;
    }
    return d;
}

// Digits an integer that isn't a constant may need when it meets a decimal.
uint8_t IntegerPrecision(TypeId result) {
    return result == TypeId::BIGINT ? 19 : 10;
}

// Signed arithmetic that wraps instead of overflowing.
int64_t WrapAdd(int64_t a, int64_t b) {
    return static_cast<int64_t>(static_cast<uint64_t>(a) + static_cast<uint64_t>(b));
}
int64_t WrapSub(int64_t a, int64_t b) {
    return static_cast<int64_t>(static_cast<uint64_t>(a) - static_cast<uint64_t>(b));
}
int64_t WrapMul(int64_t a, int64_t b) {
    return static_cast<int64_t>(static_cast<uint64_t>(a) * static_cast<uint64_t>(b));
}
int64_t WrapNeg(int64_t a) {
    return static_cast<int64_t>(0 - static_cast<uint64_t>(a));
}

// a - b; throws past 38 digits.
__int128 CheckedSub(__int128 a, __int128 b) {
    __int128 d;
    if (__builtin_sub_overflow(a, b, &d) || !detail::FitsDecimal(d)) detail::ThrowTooWide();
    return d;
}

// True when both inputs are 64-bit, so a 128-bit result can't overflow.
template <typename X, typename Y>
constexpr bool kBothNarrow = std::is_same<X, int64_t>::value && std::is_same<Y, int64_t>::value;

// Wide sums and products whose digit bound says they fit 38 digits, unchecked.
// Two 64-bit inputs widen and multiply directly; wider ones wrap, never undefined.
template <typename X, typename Y>
__int128 FastAdd(X x, Y y) {
    if constexpr (kBothNarrow<X, Y>) return static_cast<__int128>(x) + y;
    return static_cast<__int128>(static_cast<unsigned __int128>(x) + static_cast<unsigned __int128>(y));
}
template <typename X, typename Y>
__int128 FastSub(X x, Y y) {
    if constexpr (kBothNarrow<X, Y>) return static_cast<__int128>(x) - y;
    return static_cast<__int128>(static_cast<unsigned __int128>(x) - static_cast<unsigned __int128>(y));
}
template <typename X, typename Y>
__int128 FastMul(X x, Y y) {
    if constexpr (kBothNarrow<X, Y>) return static_cast<__int128>(x) * y;
    return static_cast<__int128>(static_cast<unsigned __int128>(x) * static_cast<unsigned __int128>(y));
}

// Debug check that a trusted digit bound held.
__int128 Fits(__int128 v) {
    assert(detail::FitsDecimal(v) && "a column passed its declared precision");
    return v;
}

// The sign of a * 10^d - b, without overflowing.
int CompareShifted(__int128 a, __int128 b, uint8_t d) {
    const __int128 p = Pow10(d);
    const __int128 q = b / p;
    const __int128 r = b % p;
    if (a != q) return a < q ? -1 : 1;
    return r > 0 ? -1 : (r < 0 ? 1 : 0);
}

// Bytewise order, the same as std::string's.
int CompareBytes(StringRef a, StringRef b) {
    return View(a).compare(View(b));
}

// MatchLike over views: `_` is one byte, `%` any run, no escape.
bool MatchLikeView(std::string_view str, std::string_view pattern) {
    size_t si = 0;
    size_t pi = 0;
    size_t star_p = std::string_view::npos;
    size_t match_s = 0;
    while (si < str.size()) {
        if (pi < pattern.size() && (pattern[pi] == '_' || pattern[pi] == str[si])) {
            ++si;
            ++pi;
        } else if (pi < pattern.size() && pattern[pi] == '%') {
            star_p = pi++;
            match_s = si;
        } else if (star_p != std::string_view::npos) {
            pi = star_p + 1;
            si = ++match_s;
        } else {
            return false;
        }
    }
    while (pi < pattern.size() && pattern[pi] == '%') ++pi;
    return pi == pattern.size();
}

// A VType from its four fields.
constexpr VType MakeType(Kind kind, TypeId result, uint8_t scale = 0, bool wide = false) {
    VType t;
    t.kind = kind;
    t.scale = scale;
    t.wide = wide;
    t.result = result;
    return t;
}

constexpr VType kBoolType = MakeType(Kind::INTEGER, TypeId::INTEGER);
constexpr VType kDoubleType = MakeType(Kind::DOUBLE, TypeId::DECIMAL);

// The comparison operators, written with < and == only, as Value does (NaN included).
struct CmpEq {
    template <typename T>
    bool operator()(T a, T b) const { return a == b; }
};
struct CmpNe {
    template <typename T>
    bool operator()(T a, T b) const { return !(a == b); }
};
struct CmpLt {
    template <typename T>
    bool operator()(T a, T b) const { return a < b; }
};
struct CmpGt {
    template <typename T>
    bool operator()(T a, T b) const { return b < a; }
};
struct CmpLe {
    template <typename T>
    bool operator()(T a, T b) const { return !(b < a); }
};
struct CmpGe {
    template <typename T>
    bool operator()(T a, T b) const { return !(a < b); }
};

// Calls fn with the functor for a comparison operator.
template <typename Fn>
void WithCmp(BinaryOp::Op op, Fn&& fn) {
    switch (op) {
        case BinaryOp::Op::EQ:
            fn(CmpEq{});
            return;
        case BinaryOp::Op::NEQ:
            fn(CmpNe{});
            return;
        case BinaryOp::Op::LT:
            fn(CmpLt{});
            return;
        case BinaryOp::Op::GT:
            fn(CmpGt{});
            return;
        case BinaryOp::Op::LTE:
            fn(CmpLe{});
            return;
        case BinaryOp::Op::GTE:
            fn(CmpGe{});
            return;
        default:
            assert(false && "not a comparison");
    }
}

// How the two sides of a comparison or arithmetic node are read.
enum class Domain : uint8_t { I64, WIDE, F64, STR };

// A child's values: an array indexed by row, or one constant for every row.
template <typename T>
struct VecArg {
    const T* p;
    T operator[](uint32_t i) const { return p[i]; }
};
template <typename T>
struct ConstArg {
    T v;
    T operator[](uint32_t) const { return v; }
};

// The array a vector keeps T in.
template <typename T>
const T* DataOf(const Vector& v);
template <>
const int64_t* DataOf<int64_t>(const Vector& v) { return v.I64(); }
template <>
const __int128* DataOf<__int128>(const Vector& v) { return v.I128(); }
template <>
const double* DataOf<double>(const Vector& v) { return v.F64(); }
template <typename T>
T* MutableDataOf(Vector& v);
template <>
int64_t* MutableDataOf<int64_t>(Vector& v) { return v.I64(); }
template <>
__int128* MutableDataOf<__int128>(Vector& v) { return v.I128(); }
template <>
double* MutableDataOf<double>(Vector& v) { return v.F64(); }

// out[i] = f(a[i], b[i]) for every active row.
template <typename A, typename B, typename O, typename F>
void Map2(const Selection& rows, const A& a, const B& b, O* out, F f) {
    const uint32_t n = rows.Size();
    if (rows.IsAll()) {
        for (uint32_t i = 0; i < n; ++i) out[i] = f(a[i], b[i]);
    } else {
        const uint16_t* idx = rows.Data();
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t i = idx[k];
            out[i] = f(a[i], b[i]);
        }
    }
}

// Nodes


class Node;
using NodePtr = std::unique_ptr<Node>;

// A compiled node: its type, its digit bound, and how to read its values.
class Node : public CompiledExpr {
public:
    Node(VType type, uint8_t precision) : type_(type), precision_(precision) {}

    [[nodiscard]] VType Type() const override { return type_; }

    // Most digits a DECIMAL, or an integer meeting one, can hold.
    [[nodiscard]] uint8_t Precision() const { return precision_; }

    // True for a folded constant.
    [[nodiscard]] virtual bool IsConstant() const { return false; }

    // True when every child is a constant, so this node can fold.
    [[nodiscard]] virtual bool ChildrenConstant() const = 0;

    // Evaluates, then keeps the rows whose value is a true INTEGER.
    void Select(const Batch& in, Selection& sel, EvalScratch& scratch) const override;

protected:
    // Debug check that out keeps values where this node writes them.
    void CheckOut(const Vector& out) const {
        assert(StorageOf(out.Type()) == StorageOf(type_) && "output vector has the wrong storage");
        (void)out;
    }

private:
    VType type_;
    uint8_t precision_;
};

// A child's values for `rows`: its batch vector, or a scratch vector it fills.
const Vector& ValuesOf(const Node& n, const Batch& in, const Selection& rows, EvalScratch& s, TempVector& hold) {
    if (const Vector* v = n.Direct(in)) return *v;
    Vector& out = hold.Get(n.Type());
    n.Evaluate(in, rows, out, s);
    return out;
}

void Node::Select(const Batch& in, Selection& sel, EvalScratch& scratch) const {
    TempVector hold(scratch);
    const Vector& v = ValuesOf(*this, in, sel, scratch, hold);  // runs even when nothing can be true, for errors
    if (!CanBeTrue(type_)) {
        sel.Resize(0);
        return;
    }
    const int64_t* a = v.I64();
    FilterRows(sel, [a](uint32_t i) { return a[i] != 0; });
}

// A node whose value is 1 or 0; Evaluate is built from Select.
class PredicateNode : public Node {
public:
    PredicateNode() : Node(kBoolType, 1) {}

    // 1 for the rows Select keeps, 0 for the others.
    void Evaluate(const Batch& in, const Selection& rows, Vector& out, EvalScratch& scratch) const override {
        CheckOut(out);
        TempSelection keep(scratch);
        CopySelection(rows, *keep);
        Select(in, *keep, scratch);
        int64_t* o = out.I64();
        ForRows(rows, [o](uint32_t i) { o[i] = 0; });
        ForRows(*keep, [o](uint32_t i) { o[i] = 1; });
    }
};

// Column 

// Reads one slot of the batch.
class ColumnNode final : public Node {
public:
    ColumnNode(uint32_t slot, VType type, uint8_t precision) : Node(type, precision), slot_(slot) {}

    bool ChildrenConstant() const override { return false; }

    const Vector* Direct(const Batch& in) const override {
        assert(slot_ < in.slots.size() && "column slot past the batch");
        assert(StorageOf(in.slots[slot_].Type()) == StorageOf(Type()) && "slot holds another type");
        return &in.slots[slot_];
    }

    // Copies the rows' values; strings come out plain.
    void Evaluate(const Batch& in, const Selection& rows, Vector& out, EvalScratch&) const override {
        CheckOut(out);
        const Vector& src = *Direct(in);
        switch (StorageOf(Type())) {
            case kI64: {
                const int64_t* a = src.I64();
                int64_t* o = out.I64();
                ForRows(rows, [&](uint32_t i) { o[i] = a[i]; });
                break;
            }
            case kI128: {
                const __int128* a = src.I128();
                __int128* o = out.I128();
                ForRows(rows, [&](uint32_t i) { o[i] = a[i]; });
                break;
            }
            case kF64: {
                const double* a = src.F64();
                double* o = out.F64();
                ForRows(rows, [&](uint32_t i) { o[i] = a[i]; });
                break;
            }
            case kStr: {
                out.SetDictionary(nullptr);
                StringRef* o = out.Str();
                ForRows(rows, [&](uint32_t i) { o[i] = src.StrAt(i); });
                break;
            }
        }
    }

private:
    uint32_t slot_;
};

//  Constant 

// A literal or a folded sub-expression: one value for every row.
class ConstantNode final : public Node {
public:
    ConstantNode(VType type, uint8_t precision, int64_t i64, __int128 i128, double f64, std::string str) : Node(type, precision), i64_(i64), i128_(i128), f64_(f64), str_(std::move(str)) {}

    bool IsConstant() const override { return true; }
    bool ChildrenConstant() const override { return false; }

    int64_t I64() const { return i64_; }
    __int128 I128() const { return Type().wide ? i128_ : i64_; }
    double F64() const { return f64_; }
    StringRef Str() const { return StringRef{str_.data(), static_cast<uint32_t>(str_.size())}; }

    // The constant as the type T a kernel reads.
    template <typename T>
    T As() const;

    void Evaluate(const Batch&, const Selection& rows, Vector& out, EvalScratch&) const override {
        CheckOut(out);
        switch (StorageOf(Type())) {
            case kI64: {
                int64_t* o = out.I64();
                ForRows(rows, [&](uint32_t i) { o[i] = i64_; });
                break;
            }
            case kI128: {
                __int128* o = out.I128();
                ForRows(rows, [&](uint32_t i) { o[i] = i128_; });
                break;
            }
            case kF64: {
                double* o = out.F64();
                ForRows(rows, [&](uint32_t i) { o[i] = f64_; });
                break;
            }
            case kStr: {
                out.SetDictionary(nullptr);
                StringRef* o = out.Str();
                const StringRef s = Str();
                ForRows(rows, [&](uint32_t i) { o[i] = s; });
                break;
            }
        }
    }

    void Select(const Batch&, Selection& sel, EvalScratch&) const override {
        if (!(CanBeTrue(Type()) && i64_ != 0)) sel.Resize(0);
    }

private:
    int64_t i64_;
    __int128 i128_;
    double f64_;
    std::string str_;  // owns a string constant's bytes
};

template <>
int64_t ConstantNode::As<int64_t>() const {
    assert(StorageOf(Type()) == kI64);
    return i64_;
}
template <>
__int128 ConstantNode::As<__int128>() const { return I128(); }
template <>
double ConstantNode::As<double>() const { return f64_; }

// A node already known to be a constant.
const ConstantNode& AsConstant(const Node& n) {
    assert(n.IsConstant());
    return static_cast<const ConstantNode&>(n);
}

// Constants of each kind; an integer's digits are its own.
NodePtr MakeInteger(int64_t v, TypeId result) {
    return std::make_unique<ConstantNode>(MakeType(Kind::INTEGER, result), Digits(v), v, 0, 0.0, std::string());
}

// An exact decimal constant, narrow while its digits fit 18.
NodePtr MakeDecimal(__int128 v, uint8_t scale) {
    const uint8_t p = std::max<uint8_t>(Digits(v), std::max<uint8_t>(scale, 1));
    const bool wide = p > 18;
    return std::make_unique<ConstantNode>(MakeType(Kind::DECIMAL, TypeId::DECIMAL, scale, wide), p, wide ? 0 : static_cast<int64_t>(v), wide ? v : 0, 0.0, std::string());
}

NodePtr MakeDouble(double v) {
    return std::make_unique<ConstantNode>(kDoubleType, 0, 0, 0, v, std::string());
}

NodePtr MakeDate(int64_t days) {
    return std::make_unique<ConstantNode>(MakeType(Kind::DATE, TypeId::DATE), 0, days, 0, 0.0, std::string());
}

NodePtr MakeString(std::string s) {
    return std::make_unique<ConstantNode>(MakeType(Kind::STRING, TypeId::VARCHAR), 0, 0, 0, 0.0, std::move(s));
}

// Digits for a decimal: at least its scale and 1, and 19 when wide,
// so a wide value never meets a sum or product that narrows it.
uint8_t DecimalDigits(int digits, uint8_t scale, bool wide) {
    const int p = std::max({digits, static_cast<int>(scale), 1, wide ? 19 : 0});
    return static_cast<uint8_t>(std::min<int>(p, kMaxDecimalDigits));
}

// Row `row` of v as a constant of type t, the replaced node's type.
NodePtr ConstantFrom(const VType& t, const Vector& v, uint32_t row) {
    switch (t.kind) {
        case Kind::INTEGER:
            return MakeInteger(v.I64()[row], t.result);
        case Kind::DATE:
            return MakeDate(v.I64()[row]);
        case Kind::DECIMAL: {
            const __int128 x = t.wide ? v.I128()[row] : v.I64()[row];
            const uint8_t p = DecimalDigits(Digits(x), t.scale, t.wide);
            return std::make_unique<ConstantNode>(t, p, t.wide ? 0 : static_cast<int64_t>(x), t.wide ? x : 0, 0.0, std::string());
        }
        case Kind::DOUBLE:
            return MakeDouble(v.F64()[row]);
        case Kind::STRING:
            return MakeString(std::string(View(v.StrAt(row))));
    }
    return nullptr;
}

// Calls fn with the child as a VecArg or, for a constant, a ConstArg.
template <typename T, typename Fn>
void WithArg(const Node& n, const Batch& in, const Selection& rows, EvalScratch& s, TempVector& hold, Fn&& fn) {
    if (n.IsConstant()) {
        fn(ConstArg<T>{AsConstant(n).As<T>()});
    } else {
        fn(VecArg<T>{DataOf<T>(ValuesOf(n, in, rows, s, hold))});
    }
}

// The same for a decimal, read in whichever width it is held.
template <typename Fn>
void WithDecimalArg(const Node& n, const Batch& in, const Selection& rows, EvalScratch& s, TempVector& hold, Fn&& fn) {
    if (StorageOf(n.Type()) == kI128) {
        WithArg<__int128>(n, in, rows, s, hold, fn);
    } else {
        WithArg<int64_t>(n, in, rows, s, hold, fn);
    }
}

// Calls fn(a, b) with both children read as T, the left one first.
template <typename T, typename Fn>
void WithArgs(const Node& l, const Node& r, const Batch& in, const Selection& rows, EvalScratch& s, Fn&& fn) {
    TempVector hl(s);
    TempVector hr(s);
    WithArg<T>(l, in, rows, s, hl, [&](const auto& a) {
        WithArg<T>(r, in, rows, s, hr, [&](const auto& b) { fn(a, b); });
    });
}

// ---- Cast ----

enum class CastOp : uint8_t { INT_TO_DOUBLE, DECIMAL_TO_DOUBLE, TO_DECIMAL };

// Converts a child to a double, or to a wider or finer exact decimal.
class CastNode final : public Node {
public:
    // checked: a wide result may pass 38 digits, so each value is checked.
    CastNode(CastOp op, NodePtr child, VType to, uint8_t precision, bool checked = false) : Node(to, precision), op_(op), child_(std::move(child)), checked_(checked) {}

    bool ChildrenConstant() const override { return child_->IsConstant(); }

    void Evaluate(const Batch& in, const Selection& rows, Vector& out, EvalScratch& s) const override {
        CheckOut(out);
        TempVector hold(s);
        const Vector& src = ValuesOf(*child_, in, rows, s, hold);
        const VType from = child_->Type();
        switch (op_) {
            case CastOp::INT_TO_DOUBLE: {
                const int64_t* a = src.I64();
                double* o = out.F64();
                ForRows(rows, [&](uint32_t i) { o[i] = static_cast<double>(a[i]); });
                break;
            }
            case CastOp::DECIMAL_TO_DOUBLE: {
                double* o = out.F64();
                if (from.wide) {
                    const __int128* a = src.I128();
                    ForRows(rows, [&](uint32_t i) { o[i] = DecimalToDouble(a[i], from.scale); });
                } else {
                    const int64_t* a = src.I64();
                    ForRows(rows, [&](uint32_t i) { o[i] = DecimalToDouble(a[i], from.scale); });
                }
                break;
            }
            case CastOp::TO_DECIMAL:
                ToDecimal(src, from, rows, out);
                break;
        }
    }

private:
    // Raises the scale (and maybe the width). Wide results are checked only when
    // the digit bound passes 38; otherwise they can't overflow.
    void ToDecimal(const Vector& src, const VType& from, const Selection& rows, Vector& out) const {
        const uint8_t from_scale = from.kind == Kind::DECIMAL ? from.scale : 0;
        const bool from_wide = from.kind == Kind::DECIMAL && from.wide;
        const uint8_t diff = static_cast<uint8_t>(Type().scale - from_scale);
        if (Type().wide) {
            __int128* o = out.I128();
            if (from_wide) {
                const __int128* a = src.I128();
                Widen(rows, a, diff, o);
            } else {
                const int64_t* a = src.I64();
                Widen(rows, a, diff, o);
            }
            return;
        }
        // Narrow to narrow: the digit bound says the product fits in 64 bits.
        assert(!from_wide && diff <= 18);
        const auto factor = static_cast<int64_t>(Pow10(diff));
        const int64_t* a = src.I64();
        int64_t* o = out.I64();
        ForRows(rows, [&](uint32_t i) { o[i] = WrapMul(a[i], factor); });
    }

    // o = a * 10^diff in 128 bits.
    template <typename T>
    void Widen(const Selection& rows, const T* a, uint8_t diff, __int128* o) const {
        if (diff == 0) {
            ForRows(rows, [&](uint32_t i) { o[i] = a[i]; });
        } else if (checked_) {
            const __int128 factor = Pow10(diff);
            ForRows(rows, [&](uint32_t i) { o[i] = CheckedMul(a[i], factor); });
        } else if (diff <= 18) {
            const auto factor = static_cast<int64_t>(Pow10(diff));
            ForRows(rows, [&](uint32_t i) { o[i] = Fits(FastMul(a[i], factor)); });
        } else {
            const __int128 factor = Pow10(diff);
            ForRows(rows, [&](uint32_t i) { o[i] = Fits(FastMul(a[i], factor)); });
        }
    }

    CastOp op_;
    NodePtr child_;
    bool checked_;
};

// ---- Arithmetic ----

// + - * / on two children already in one domain.
class ArithmeticNode final : public Node {
public:
    // WIDE takes either width on each side; checked: the result may pass 38 digits.
    ArithmeticNode(BinaryOp::Op op, Domain domain, NodePtr l, NodePtr r, VType type, uint8_t precision, bool checked = false) : Node(type, precision), op_(op), domain_(domain), l_(std::move(l)), r_(std::move(r)), checked_(checked) {}

    bool ChildrenConstant() const override { return l_->IsConstant() && r_->IsConstant(); }

    void Evaluate(const Batch& in, const Selection& rows, Vector& out, EvalScratch& s) const override {
        CheckOut(out);
        switch (domain_) {
            case Domain::I64:
                WithArgs<int64_t>(*l_, *r_, in, rows, s, [&](const auto& a, const auto& b) { RunI64(rows, a, b, out.I64()); });
                break;
            case Domain::WIDE: {
                TempVector hl(s);
                TempVector hr(s);
                WithDecimalArg(*l_, in, rows, s, hl, [&](const auto& a) {
                    WithDecimalArg(*r_, in, rows, s, hr, [&](const auto& b) { RunWide(rows, a, b, out.I128()); });
                });
                break;
            }
            case Domain::F64:
                WithArgs<double>(*l_, *r_, in, rows, s, [&](const auto& a, const auto& b) { RunF64(rows, a, b, out.F64()); });
                break;
            case Domain::STR:
                assert(false && "no string arithmetic");
                break;
        }
    }

private:
    // Integers and narrow decimals wrap; division truncates and throws on zero.
    template <typename A, typename B>
    void RunI64(const Selection& rows, const A& a, const B& b, int64_t* o) const {
        switch (op_) {
            case BinaryOp::Op::ADD:
                Map2(rows, a, b, o, WrapAdd);
                break;
            case BinaryOp::Op::SUB:
                Map2(rows, a, b, o, WrapSub);
                break;
            case BinaryOp::Op::MUL:
                Map2(rows, a, b, o, WrapMul);
                break;
            default:
                ForRows(rows, [&](uint32_t i) {
                    if (b[i] == 0) throw DatabaseException("Division by zero");
                });
                Map2(rows, a, b, o, [](int64_t x, int64_t y) { return y == -1 ? WrapNeg(x) : x / y; });
                break;
        }
    }

    // Wide decimals: checked past 38 digits only when the digit bound allows it.
    template <typename A, typename B>
    void RunWide(const Selection& rows, const A& a, const B& b, __int128* o) const {
        if (checked_) {
            switch (op_) {
                case BinaryOp::Op::ADD:
                    Map2(rows, a, b, o, [](__int128 x, __int128 y) { return CheckedAdd(x, y); });
                    break;
                case BinaryOp::Op::SUB:
                    Map2(rows, a, b, o, [](__int128 x, __int128 y) { return CheckedSub(x, y); });
                    break;
                default:
                    Map2(rows, a, b, o, [](__int128 x, __int128 y) { return CheckedMul(x, y); });
                    break;
            }
            return;
        }
        switch (op_) {
            case BinaryOp::Op::ADD:
                Map2(rows, a, b, o, [](auto x, auto y) { return Fits(FastAdd(x, y)); });
                break;
            case BinaryOp::Op::SUB:
                Map2(rows, a, b, o, [](auto x, auto y) { return Fits(FastSub(x, y)); });
                break;
            default:
                Map2(rows, a, b, o, [](auto x, auto y) { return Fits(FastMul(x, y)); });
                break;
        }
    }

    // Doubles as today; any zero divisor gives +infinity.
    template <typename A, typename B>
    void RunF64(const Selection& rows, const A& a, const B& b, double* o) const {
        switch (op_) {
            case BinaryOp::Op::ADD:
                Map2(rows, a, b, o, [](double x, double y) { return x + y; });
                break;
            case BinaryOp::Op::SUB:
                Map2(rows, a, b, o, [](double x, double y) { return x - y; });
                break;
            case BinaryOp::Op::MUL:
                Map2(rows, a, b, o, [](double x, double y) { return x * y; });
                break;
            default:
                Map2(rows, a, b, o, [](double x, double y) {
                    return y == 0.0 ? std::numeric_limits<double>::infinity() : x / y;
                });
                break;
        }
    }

    BinaryOp::Op op_;
    Domain domain_;
    NodePtr l_;
    NodePtr r_;
    bool checked_;
};

// Negate 

class NegateNode final : public Node {
public:
    explicit NegateNode(NodePtr child) : Node(child->Type(), child->Precision()), child_(std::move(child)) {}

    bool ChildrenConstant() const override { return child_->IsConstant(); }

    void Evaluate(const Batch& in, const Selection& rows, Vector& out, EvalScratch& s) const override {
        CheckOut(out);
        TempVector hold(s);
        const Vector& src = ValuesOf(*child_, in, rows, s, hold);
        switch (StorageOf(Type())) {
            case kI64: {
                const int64_t* a = src.I64();
                int64_t* o = out.I64();
                ForRows(rows, [&](uint32_t i) { o[i] = WrapNeg(a[i]); });
                break;
            }
            case kI128: {
                const __int128* a = src.I128();  // within 38 digits, so negating is safe
                __int128* o = out.I128();
                ForRows(rows, [&](uint32_t i) { o[i] = -a[i]; });
                break;
            }
            case kF64: {
                const double* a = src.F64();
                double* o = out.F64();
                ForRows(rows, [&](uint32_t i) { o[i] = -a[i]; });
                break;
            }
            case kStr:
                assert(false && "no string negation");
                break;
        }
    }

private:
    NodePtr child_;
};

// Keeps the rows where pred(string) is true. On a dictionary no bigger than the
// selection, pred runs once per entry and rows look up their code.
// Testing entries no row uses is safe only because string predicates never throw.
template <typename Pred>
void FilterStrings(const Vector& v, Selection& sel, EvalScratch& s, Pred pred) {
    if (v.IsDictionary() && v.Entries().size() <= sel.Size()) {
        const std::vector<StringRef>& entries = v.Entries();
        TempTable table(s);
        std::vector<uint8_t>& t = *table;
        t.resize(entries.size());
        for (size_t e = 0; e < entries.size(); ++e) t[e] = pred(entries[e]) ? 1 : 0;
        const uint32_t* codes = v.Codes();
        const uint8_t* hit = t.data();
        FilterRows(sel, [&](uint32_t i) { return hit[codes[i]] != 0; });
        return;
    }
    FilterRows(sel, [&](uint32_t i) { return pred(v.StrAt(i)); });
}

//  Compare 

// = <> < > <= >= on two children already in one domain.
class CompareNode final : public PredicateNode {
public:
    CompareNode(BinaryOp::Op op, Domain domain, NodePtr l, NodePtr r, uint8_t shift_l, uint8_t shift_r) : op_(op), domain_(domain), l_(std::move(l)), r_(std::move(r)), shift_l_(shift_l), shift_r_(shift_r) {}

    bool ChildrenConstant() const override { return l_->IsConstant() && r_->IsConstant(); }

    void Select(const Batch& in, Selection& sel, EvalScratch& s) const override {
        switch (domain_) {
            case Domain::I64:
                WithArgs<int64_t>(*l_, *r_, in, sel, s, [&](const auto& a, const auto& b) {
                    WithCmp(op_, [&](auto cmp) { FilterRows(sel, [&](uint32_t i) { return cmp(a[i], b[i]); }); });
                });
                break;
            case Domain::F64:
                WithArgs<double>(*l_, *r_, in, sel, s, [&](const auto& a, const auto& b) {
                    WithCmp(op_, [&](auto cmp) { FilterRows(sel, [&](uint32_t i) { return cmp(a[i], b[i]); }); });
                });
                break;
            case Domain::WIDE:
                WithArgs<__int128>(*l_, *r_, in, sel, s, [&](const auto& a, const auto& b) {
                    WithCmp(op_, [&](auto cmp) { FilterRows(sel, [&](uint32_t i) { return cmp(Wide(a[i], b[i]), 0); }); });
                });
                break;
            case Domain::STR:
                SelectStrings(in, sel, s);
                break;
        }
    }

    // Numbers write 1 / 0 directly; strings go through Select.
    void Evaluate(const Batch& in, const Selection& rows, Vector& out, EvalScratch& s) const override {
        CheckOut(out);
        int64_t* o = out.I64();
        switch (domain_) {
            case Domain::I64:
                WithArgs<int64_t>(*l_, *r_, in, rows, s, [&](const auto& a, const auto& b) {
                    WithCmp(op_, [&](auto cmp) { Map2(rows, a, b, o, [cmp](int64_t x, int64_t y) -> int64_t { return cmp(x, y) ? 1 : 0; }); });
                });
                break;
            case Domain::F64:
                WithArgs<double>(*l_, *r_, in, rows, s, [&](const auto& a, const auto& b) {
                    WithCmp(op_, [&](auto cmp) { Map2(rows, a, b, o, [cmp](double x, double y) -> int64_t { return cmp(x, y) ? 1 : 0; }); });
                });
                break;
            case Domain::WIDE:
                WithArgs<__int128>(*l_, *r_, in, rows, s, [&](const auto& a, const auto& b) {
                    WithCmp(op_, [&](auto cmp) {
                        Map2(rows, a, b, o, [&](__int128 x, __int128 y) -> int64_t { return cmp(Wide(x, y), 0) ? 1 : 0; });
                    });
                });
                break;
            case Domain::STR:
                PredicateNode::Evaluate(in, rows, out, s);
                break;
        }
    }

private:
    // The sign of l - r once both sides sit at the common scale.
    int Wide(__int128 a, __int128 b) const {
        if (shift_l_ > 0) return CompareShifted(a, b, shift_l_);
        if (shift_r_ > 0) return -CompareShifted(b, a, shift_r_);
        return a < b ? -1 : (b < a ? 1 : 0);
    }

    // Bytewise; a dictionary side against a constant is tested once per entry.
    void SelectStrings(const Batch& in, Selection& sel, EvalScratch& s) const {
        TempVector hl(s);
        TempVector hr(s);
        WithCmp(op_, [&](auto cmp) {
            if (r_->IsConstant()) {
                const StringRef c = AsConstant(*r_).Str();
                const Vector& v = ValuesOf(*l_, in, sel, s, hl);
                FilterStrings(v, sel, s, [&](StringRef x) { return cmp(CompareBytes(x, c), 0); });
            } else if (l_->IsConstant()) {
                const StringRef c = AsConstant(*l_).Str();
                const Vector& v = ValuesOf(*r_, in, sel, s, hr);
                FilterStrings(v, sel, s, [&](StringRef x) { return cmp(CompareBytes(c, x), 0); });
            } else {
                const Vector& lv = ValuesOf(*l_, in, sel, s, hl);
                const Vector& rv = ValuesOf(*r_, in, sel, s, hr);
                FilterRows(sel, [&](uint32_t i) { return cmp(CompareBytes(lv.StrAt(i), rv.StrAt(i)), 0); });
            }
        });
    }

    BinaryOp::Op op_;
    Domain domain_;
    NodePtr l_;
    NodePtr r_;
    uint8_t shift_l_;  // WIDE only: powers of ten the left side is behind
    uint8_t shift_r_;
};

// ---- AND / OR / NOT ----

// The right side runs only on rows the left side left true.
class AndNode final : public PredicateNode {
public:
    AndNode(NodePtr l, NodePtr r) : l_(std::move(l)), r_(std::move(r)) {}

    bool ChildrenConstant() const override { return l_->IsConstant() && r_->IsConstant(); }

    void Select(const Batch& in, Selection& sel, EvalScratch& s) const override {
        l_->Select(in, sel, s);
        if (sel.Size() > 0) r_->Select(in, sel, s);
    }

private:
    NodePtr l_;
    NodePtr r_;
};

// The right side runs only on rows the left side left false.
class OrNode final : public PredicateNode {
public:
    OrNode(NodePtr l, NodePtr r) : l_(std::move(l)), r_(std::move(r)) {}

    bool ChildrenConstant() const override { return l_->IsConstant() && r_->IsConstant(); }

    void Select(const Batch& in, Selection& sel, EvalScratch& s) const override {
        if (sel.Size() == 0) return;
        TempSelection taken(s);
        CopySelection(sel, *taken);
        l_->Select(in, *taken, s);
        if ((*taken).Size() == sel.Size()) return;  // every row is already true

        TempSelection rest(s);
        Subtract(sel, *taken, *rest);
        r_->Select(in, *rest, s);
        MergeInto(*taken, *rest, sel);
    }

private:
    NodePtr l_;
    NodePtr r_;
};

class NotNode final : public PredicateNode {
public:
    explicit NotNode(NodePtr child) : child_(std::move(child)) {}

    bool ChildrenConstant() const override { return child_->IsConstant(); }

    void Select(const Batch& in, Selection& sel, EvalScratch& s) const override {
        TempSelection yes(s);
        CopySelection(sel, *yes);
        child_->Select(in, *yes, s);
        Subtract(sel, *yes, sel);
    }

private:
    NodePtr child_;
};

// ---- IN ----

// `column IN (constants)` as one probe against a set.
class InNode final : public PredicateNode {
public:
    InNode(NodePtr operand, Domain domain, std::vector<int64_t> ints, std::vector<double> doubles, std::vector<std::string> strings)
        : operand_(std::move(operand)), domain_(domain), ints_(std::move(ints)), doubles_(std::move(doubles)), strings_(std::move(strings)) {
        std::sort(ints_.begin(), ints_.end());
    }

    bool ChildrenConstant() const override { return false; }

    void Select(const Batch& in, Selection& sel, EvalScratch& s) const override {
        TempVector hold(s);
        const Vector& v = ValuesOf(*operand_, in, sel, s, hold);
        switch (domain_) {
            case Domain::I64: {
                const int64_t* a = v.I64();
                FilterRows(sel, [&](uint32_t i) { return std::binary_search(ints_.begin(), ints_.end(), a[i]); });
                break;
            }
            case Domain::F64: {
                const double* a = v.F64();
                FilterRows(sel, [&](uint32_t i) {
                    return std::any_of(doubles_.begin(), doubles_.end(), [&](double d) { return a[i] == d; });
                });
                break;
            }
            case Domain::STR:
                FilterStrings(v, sel, s, [&](StringRef x) {
                    const std::string_view sv = View(x);
                    return std::any_of(strings_.begin(), strings_.end(), [&](const std::string& c) { return sv == c; });
                });
                break;
            case Domain::WIDE:
                assert(false && "wide IN lists compile as OR chains");
                break;
        }
    }

private:
    NodePtr operand_;
    Domain domain_;
    std::vector<int64_t> ints_;  // sorted
    std::vector<double> doubles_;
    std::vector<std::string> strings_;
};

//  LIKE 

// A literal pattern's shape; only GENERAL needs the full matcher.
enum class LikeShape : uint8_t { PREFIX, SUFFIX, CONTAINS, GENERAL };

// The shape of a literal pattern, and its fixed text when it has one.
LikeShape ShapeOf(const std::string& p, std::string& needle) {
    auto plain = [](std::string_view s) { return s.find_first_of("%_") == std::string_view::npos; };
    const std::string_view v(p);
    if (!v.empty() && v.back() == '%' && plain(v.substr(0, v.size() - 1))) {
        needle = std::string(v.substr(0, v.size() - 1));
        return LikeShape::PREFIX;
    }
    if (!v.empty() && v.front() == '%' && plain(v.substr(1))) {
        needle = std::string(v.substr(1));
        return LikeShape::SUFFIX;
    }
    if (v.size() >= 2 && v.front() == '%' && v.back() == '%' && plain(v.substr(1, v.size() - 2))) {
        needle = std::string(v.substr(1, v.size() - 2));
        return LikeShape::CONTAINS;
    }
    return LikeShape::GENERAL;
}

class LikeNode final : public PredicateNode {
public:
    // pattern is nullptr when the pattern is the literal `text`.
    LikeNode(NodePtr str, NodePtr pattern, std::string text) : str_(std::move(str)), pattern_(std::move(pattern)), text_(std::move(text)) {
        shape_ = pattern_ ? LikeShape::GENERAL : ShapeOf(text_, needle_);
    }

    bool ChildrenConstant() const override { return str_->IsConstant() && (!pattern_ || pattern_->IsConstant()); }

    void Select(const Batch& in, Selection& sel, EvalScratch& s) const override {
        TempVector hs(s);
        const Vector& v = ValuesOf(*str_, in, sel, s, hs);
        if (!pattern_) {
            FilterStrings(v, sel, s, [this](StringRef x) { return Matches(View(x)); });
            return;
        }
        TempVector hp(s);
        const Vector& p = ValuesOf(*pattern_, in, sel, s, hp);
        FilterRows(sel, [&](uint32_t i) { return MatchLikeView(View(v.StrAt(i)), View(p.StrAt(i))); });
    }

private:
    // Fast paths give MatchLike's answers only when the text has no '%', since
    // MatchLike lets a pattern '%' match a text '%' literally first.
    bool Matches(std::string_view x) const {
        if (shape_ == LikeShape::GENERAL || x.find('%') != std::string_view::npos) return MatchLikeView(x, text_);
        const size_t n = needle_.size();
        switch (shape_) {
            case LikeShape::PREFIX:
                return x.size() >= n && x.compare(0, n, needle_) == 0;
            case LikeShape::SUFFIX:
                return x.size() >= n && x.compare(x.size() - n, n, needle_) == 0;
            case LikeShape::CONTAINS:
                return x.find(needle_) != std::string_view::npos;
            case LikeShape::GENERAL:
                break;
        }
        return MatchLikeView(x, text_);
    }

    NodePtr str_;
    NodePtr pattern_;
    std::string text_;    // the literal pattern
    std::string needle_;  // its fixed text, for the fast paths
    LikeShape shape_{LikeShape::GENERAL};
};

// ---- CASE ----

// WHENs run in order on rows still open; each THEN only on its rows.
class CaseNode final : public Node {
public:
    CaseNode(VType type, uint8_t precision, std::vector<std::pair<NodePtr, NodePtr>> whens, NodePtr otherwise) : Node(type, precision), whens_(std::move(whens)), else_(std::move(otherwise)) {}

    bool ChildrenConstant() const override {
        for (const auto& [cond, result] : whens_) {
            if (!cond->IsConstant() || !result->IsConstant()) return false;
        }
        return !else_ || else_->IsConstant();
    }

    void Evaluate(const Batch& in, const Selection& rows, Vector& out, EvalScratch& s) const override {
        CheckOut(out);
        if (Type().kind == Kind::STRING) out.SetDictionary(nullptr);
        TempSelection open(s);
        TempSelection taken(s);
        CopySelection(rows, *open);

        for (const auto& [cond, result] : whens_) {
            if ((*open).Size() == 0) return;
            CopySelection(*open, *taken);
            cond->Select(in, *taken, s);
            if ((*taken).Size() == 0) continue;
            result->Evaluate(in, *taken, out, s);
            Subtract(*open, *taken, *open);
        }
        if ((*open).Size() == 0) return;
        if (else_) {
            else_->Evaluate(in, *open, out, s);
        } else {
            FillZero(*open, out);
        }
    }

private:
    // No ELSE: zero of the CASE's type (today gives an INTEGER 0).
    void FillZero(const Selection& rows, Vector& out) const {
        switch (StorageOf(Type())) {
            case kI64: {
                int64_t* o = out.I64();
                ForRows(rows, [o](uint32_t i) { o[i] = 0; });
                break;
            }
            case kI128: {
                __int128* o = out.I128();
                ForRows(rows, [o](uint32_t i) { o[i] = 0; });
                break;
            }
            case kF64: {
                double* o = out.F64();
                ForRows(rows, [o](uint32_t i) { o[i] = 0.0; });
                break;
            }
            case kStr: {
                StringRef* o = out.Str();
                ForRows(rows, [o](uint32_t i) { o[i] = StringRef{"", 0}; });
                break;
            }
        }
    }

    std::vector<std::pair<NodePtr, NodePtr>> whens_;
    NodePtr else_;  // nullptr when absent
};

// Typing and folding

// Runs a node with all-constant children once, on one row, keeping the value.
// A node that throws stays, so it throws only for rows that reach it.
NodePtr Fold(NodePtr n) {
    if (n->IsConstant() || !n->ChildrenConstant()) return n;
    try {
        Batch one;
        one.rows = 1;
        one.sel.SetAll(1);
        Vector out(n->Type());
        EvalScratch scratch;
        n->Evaluate(one, one.sel, out, scratch);
        return ConstantFrom(n->Type(), out, 0);
    } catch (const std::exception&) {
        return n;
    }
}

// True for a decimal held in 128 bits.
bool IsWide(const VType& t) {
    return t.kind == Kind::DECIMAL && t.wide;
}

// A node's digits and scale, treating an integer as a decimal at scale 0.
struct DecimalInfo {
    int precision;
    int scale;
};

// The node's digits and scale.
DecimalInfo InfoOf(const Node& n) {
    const VType t = n.Type();
    return DecimalInfo{n.Precision(), t.kind == Kind::DECIMAL ? t.scale : 0};
}

// Digits capped at 38.
uint8_t CapDigits(int p) {
    return static_cast<uint8_t>(std::min<int>(p, kMaxDecimalDigits));
}

// The node as a double; integers and decimals convert, doubles pass through.
NodePtr ToDouble(NodePtr n) {
    switch (n->Type().kind) {
        case Kind::DOUBLE:
            return n;
        case Kind::INTEGER:
            return Fold(std::make_unique<CastNode>(CastOp::INT_TO_DOUBLE, std::move(n), kDoubleType, 0));
        case Kind::DECIMAL:
            return Fold(std::make_unique<CastNode>(CastOp::DECIMAL_TO_DOUBLE, std::move(n), kDoubleType, 0));
        default:
            assert(false && "only numbers convert to double");
            return n;
    }
}

// The node as an exact decimal at `scale` (never lower than its own) and width.
// An integer read at scale 0 in 64 bits needs no conversion.
NodePtr ToDecimal(NodePtr n, uint8_t scale, bool wide) {
    const VType t = n->Type();
    const DecimalInfo info = InfoOf(*n);
    const bool from_wide = IsWide(t);

    assert(info.scale <= scale && (!from_wide || wide));
    if (info.scale == scale && from_wide == wide) return n;
    const int bound = info.precision - info.scale + scale;
    const uint8_t p = DecimalDigits(bound, scale, wide);

    return Fold(std::make_unique<CastNode>(CastOp::TO_DECIMAL, std::move(n), MakeType(Kind::DECIMAL, TypeId::DECIMAL, scale, wide), p, bound > kMaxDecimalDigits));
}

// The node at `scale` for a sum: unchanged when already there, else narrow while
// the rescaled digits fit 18, else wide.
NodePtr RescaleForSum(NodePtr n, uint8_t scale) {
    const DecimalInfo info = InfoOf(*n);
    if (info.scale == scale) return n;
    const bool narrow = !IsWide(n->Type()) && info.precision - info.scale + scale <= 18;
    return ToDecimal(std::move(n), scale, !narrow);
}

// + - * /: a double side makes doubles, then exact decimals, then integers.
NodePtr MakeArithmetic(BinaryOp::Op op, NodePtr l, NodePtr r) {
    const VType lt = l->Type();
    const VType rt = r->Type();
    const TypeId common = CommonType(lt.result, rt.result);
    if (common == TypeId::INVALID) throw DatabaseException("Type mismatch in arithmetic");
    if (common == TypeId::VARCHAR || common == TypeId::DATE) {
        throw DatabaseException("Arithmetic not supported for this type");
    }
    const bool both_int = lt.kind == Kind::INTEGER && rt.kind == Kind::INTEGER;
    const VType int_type = MakeType(Kind::INTEGER, common);

    if (op == BinaryOp::Op::DIV) {
        if (both_int) {
            return Fold(std::make_unique<ArithmeticNode>(op, Domain::I64, std::move(l), std::move(r), int_type, IntegerPrecision(common)));
        }
        return Fold(std::make_unique<ArithmeticNode>(op, Domain::F64, ToDouble(std::move(l)), ToDouble(std::move(r)), kDoubleType, 0));
    }
    if (lt.kind == Kind::DOUBLE || rt.kind == Kind::DOUBLE) {
        return Fold(std::make_unique<ArithmeticNode>(op, Domain::F64, ToDouble(std::move(l)), ToDouble(std::move(r)), kDoubleType, 0));
    }
    if (both_int) {
        return Fold(std::make_unique<ArithmeticNode>(op, Domain::I64, std::move(l), std::move(r), int_type, IntegerPrecision(common)));
    }

    // Exact decimals: products add scales, sums and differences use the larger one.
    // Each side keeps its width; values are checked only when the bound passes 38.
    const DecimalInfo a = InfoOf(*l);
    const DecimalInfo b = InfoOf(*r);
    if (op == BinaryOp::Op::MUL) {
        const int scale = a.scale + b.scale;
        if (scale > kMaxDecimalDigits) {
            // Too many decimals to hold exactly: doubles, as today computes it.
            return Fold(std::make_unique<ArithmeticNode>(op, Domain::F64, ToDouble(std::move(l)), ToDouble(std::move(r)), kDoubleType, 0));
        }
        const int bound = a.precision + b.precision;
        const uint8_t p = CapDigits(bound);
        const bool wide = p > 18;
        return Fold(std::make_unique<ArithmeticNode>(op, wide ? Domain::WIDE : Domain::I64, std::move(l), std::move(r), MakeType(Kind::DECIMAL, TypeId::DECIMAL, static_cast<uint8_t>(scale), wide), p, bound > kMaxDecimalDigits));
    }
    const int scale = std::max(a.scale, b.scale);
    const int bound = std::max(a.precision - a.scale, b.precision - b.scale) + scale + 1;
    const uint8_t p = CapDigits(bound);
    const bool wide = p > 18;
    NodePtr lw = RescaleForSum(std::move(l), static_cast<uint8_t>(scale));
    NodePtr rw = RescaleForSum(std::move(r), static_cast<uint8_t>(scale));
    return Fold(std::make_unique<ArithmeticNode>(op, wide ? Domain::WIDE : Domain::I64, std::move(lw), std::move(rw), MakeType(Kind::DECIMAL, TypeId::DECIMAL, static_cast<uint8_t>(scale), wide), p, bound > kMaxDecimalDigits));
}

// A string constant read as a DATE by today's cast; its exceptions are today's.
NodePtr ParseDate(const Node& n) {
    const Value v = Value::FromString(TypeId::DATE, std::string(View(AsConstant(n).Str())));
    return MakeDate(v.date_);
}

// A string constant facing a DATE becomes a DATE, as today's cast reads it.
NodePtr DateFromString(NodePtr n, const VType& other) {
    if (n->Type().kind != Kind::STRING || other.kind != Kind::DATE) return n;
    if (!n->IsConstant()) throw DatabaseException("a DATE can only be compared with a constant string");
    return ParseDate(*n);
}

// Two sides moved into one comparison domain.
struct Prepared {
    Domain domain{Domain::I64};
    NodePtr l;
    NodePtr r;
    uint8_t shift_l{0};
    uint8_t shift_r{0};
};

// Types a comparison: exact for decimals and integers, doubles when a double is there.
Prepared PrepareCompare(NodePtr l, NodePtr r) {
    l = DateFromString(std::move(l), r->Type());
    r = DateFromString(std::move(r), l->Type());
    const VType lt = l->Type();
    const VType rt = r->Type();
    if (CommonType(lt.result, rt.result) == TypeId::INVALID) throw DatabaseException("Type mismatch in comparison");

    Prepared p;
    if (lt.kind == Kind::STRING) {
        p.domain = Domain::STR;
    } else if (lt.kind == Kind::DATE) {
        p.domain = Domain::I64;
    } else if (lt.kind == Kind::DOUBLE || rt.kind == Kind::DOUBLE) {
        p.domain = Domain::F64;
        l = ToDouble(std::move(l));
        r = ToDouble(std::move(r));
    } else if (lt.kind == Kind::INTEGER && rt.kind == Kind::INTEGER) {
        p.domain = Domain::I64;
    } else {
        const DecimalInfo a = InfoOf(*l);
        const DecimalInfo b = InfoOf(*r);
        const int scale = std::max(a.scale, b.scale);
        const bool narrow = !IsWide(lt) && !IsWide(rt) && a.precision - a.scale + scale <= 18 && b.precision - b.scale + scale <= 18;
        if (narrow) {
            p.domain = Domain::I64;
            l = ToDecimal(std::move(l), static_cast<uint8_t>(scale), false);
            r = ToDecimal(std::move(r), static_cast<uint8_t>(scale), false);
        } else {
            // Rescaling could pass 38 digits, so the kernel shifts while comparing.
            p.domain = Domain::WIDE;
            l = ToDecimal(std::move(l), static_cast<uint8_t>(a.scale), true);
            r = ToDecimal(std::move(r), static_cast<uint8_t>(b.scale), true);
            p.shift_l = static_cast<uint8_t>(scale - a.scale);
            p.shift_r = static_cast<uint8_t>(scale - b.scale);
        }
    }
    p.l = std::move(l);
    p.r = std::move(r);
    return p;
}

// One `column = literal` leaf of an IN list.
struct InLeaf {
    const ColumnRef* column;
    const Literal* literal;
};

// True for an OR tree whose leaves are all `column = literal`, one column.
bool CollectInLeaves(const Expression& e, std::vector<InLeaf>& leaves) {
    if (e.type != ExprType::BINARY_OP) return false;
    const auto& b = static_cast<const BinaryOp&>(e);
    if (b.op == BinaryOp::Op::OR) return CollectInLeaves(*b.left, leaves) && CollectInLeaves(*b.right, leaves);
    if (b.op != BinaryOp::Op::EQ) return false;

    const Expression* col = b.left.get();
    const Expression* lit = b.right.get();
    if (col->type == ExprType::LITERAL) std::swap(col, lit);
    if (col->type != ExprType::COLUMN_REF || lit->type != ExprType::LITERAL) return false;

    const auto* c = static_cast<const ColumnRef*>(col);
    if (!c->bound_index) return false;
    if (!leaves.empty() && *leaves.front().column->bound_index != *c->bound_index) return false;
    leaves.push_back(InLeaf{c, static_cast<const Literal*>(lit)});
    return true;
}

// Builds compiled nodes from bound expressions.
class Compiler {
public:
    explicit Compiler(const InputLayout& in) : in_(in) {}

    // Compiles one expression and everything under it.
    NodePtr Build(const Expression& e) {
        switch (e.type) {
            case ExprType::COLUMN_REF:
                return Column(static_cast<const ColumnRef&>(e));
            case ExprType::LITERAL:
                return Constant(static_cast<const Literal&>(e));
            case ExprType::BINARY_OP:
                return Binary(static_cast<const BinaryOp&>(e));
            case ExprType::UNARY_OP:
                return Unary(static_cast<const UnaryOp&>(e));
            case ExprType::CASE:
                return Case(static_cast<const CaseExpression&>(e));
            case ExprType::AGGREGATE:
            case ExprType::STAR:
                break;
        }
        throw DatabaseException("Cannot evaluate AGGREGATE or STAR expression in this context");
    }

private:
    // A bound column found in the layout; throws when it isn't there.
    NodePtr Column(const ColumnRef& c) {
        if (!c.bound_index) throw DatabaseException("column '" + c.column_name + "' is not bound");
        const uint32_t index = *c.bound_index;
        const int slot = index < in_.slot_of_column.size() ? in_.slot_of_column[index] : -1;
        if (slot < 0 || static_cast<size_t>(slot) >= in_.slot_types.size()) {
            throw DatabaseException("column '" + c.column_name + "' is not in the batch");
        }

        const VType t = in_.slot_types[static_cast<size_t>(slot)];
        uint8_t p = 0;
        if (t.kind == Kind::DECIMAL) {
            const size_t s = static_cast<size_t>(slot);
            const int given = s < in_.slot_precision.size() ? in_.slot_precision[s] : 0;
            assert((given == 0 || (given >= t.scale && given <= kMaxDecimalDigits && (t.wide || given <= 18))) && "slot precision doesn't fit the slot's type");
            p = given == 0 ? (t.wide ? 38 : 18) : DecimalDigits(t.wide ? given : std::min(given, 18), t.scale, t.wide);
        } else if (t.kind == Kind::INTEGER) {
            p = IntegerPrecision(t.result);
        }
        return std::make_unique<ColumnNode>(static_cast<uint32_t>(slot), t, p);
    }

    // A decimal literal is exact from its written text while its value still matches it,
    // else exact from a round-trip text of at most 15 digits, else a double.
    static NodePtr Constant(const Literal& lit) {
        const Value& v = lit.value;
        switch (v.type_) {
            case TypeId::INTEGER:
                return MakeInteger(v.integer_, TypeId::INTEGER);
            case TypeId::BIGINT:
                return MakeInteger(v.bigint_, TypeId::BIGINT);
            case TypeId::DECIMAL: {
                const std::optional<DecimalLiteral> d =
                    WrittenTextMatches(lit) ? DecimalFromText(*lit.text) : DecimalFromDouble(v.decimal_, kExactDoubleDigits);
                return d ? MakeDecimal(d->value, d->scale) : MakeDouble(v.decimal_);
            }
            case TypeId::VARCHAR:
                return MakeString(v.varchar_);
            case TypeId::DATE:
                return MakeDate(v.date_);
            case TypeId::INVALID:
                break;
        }
        throw DatabaseException("NULL literals aren't supported here");
    }

    // True when the literal's text is set and the binder left its value alone.
    static bool WrittenTextMatches(const Literal& lit) {
        if (!lit.text || lit.value.type_ != TypeId::DECIMAL) return false;
        const char* begin = lit.text->c_str();
        char* end = nullptr;
        const double parsed = std::strtod(begin, &end);
        return end == begin + lit.text->size() && parsed == lit.value.decimal_;
    }

    // AND, OR (maybe an IN list), LIKE, arithmetic and comparisons.
    NodePtr Binary(const BinaryOp& b) {
        switch (b.op) {
            case BinaryOp::Op::AND:
                return Fold(std::make_unique<AndNode>(Build(*b.left), Build(*b.right)));
            case BinaryOp::Op::OR: {
                std::vector<InLeaf> leaves;
                if (CollectInLeaves(b, leaves)) {
                    if (NodePtr in = InList(leaves)) return in;
                }
                return Fold(std::make_unique<OrNode>(Build(*b.left), Build(*b.right)));
            }
            case BinaryOp::Op::LIKE:
                return Like(Build(*b.left), Build(*b.right));
            case BinaryOp::Op::ADD:
            case BinaryOp::Op::SUB:
            case BinaryOp::Op::MUL:
            case BinaryOp::Op::DIV:
                return MakeArithmetic(b.op, Build(*b.left), Build(*b.right));
            default: {  // = <> < > <= >=
                Prepared p = PrepareCompare(Build(*b.left), Build(*b.right));
                return Fold(std::make_unique<CompareNode>(b.op, p.domain, std::move(p.l), std::move(p.r), p.shift_l, p.shift_r));
            }
        }
    }

    // One probe for an IN list; nullptr when the plain OR chain should run instead.
    NodePtr InList(const std::vector<InLeaf>& leaves) {
        NodePtr operand;
        Domain domain = Domain::I64;
        std::vector<int64_t> ints;
        std::vector<double> doubles;
        std::vector<std::string> strings;

        for (const InLeaf& leaf : leaves) {
            NodePtr col = Column(*leaf.column);
            const Node* raw = col.get();
            Prepared p = PrepareCompare(std::move(col), Constant(*leaf.literal));  // today's type errors
            if (p.domain == Domain::WIDE || !p.r->IsConstant()) return nullptr;
            if (p.l.get() != raw) {
                // An exact literal with more decimals than the column can never match.
                // A double rounds the column too (10^38 - 1 equals 1e38), so the chain decides.
                if (p.domain == Domain::I64) continue;
                return nullptr;
            }
            const ConstantNode& c = AsConstant(*p.r);
            switch (p.domain) {
                case Domain::I64:
                    ints.push_back(c.I64());
                    break;
                case Domain::F64:
                    doubles.push_back(c.F64());
                    break;
                default:
                    strings.emplace_back(View(c.Str()));
                    break;
            }
            if (!operand) {
                operand = std::move(p.l);
                domain = p.domain;
            }
        }
        if (!operand) return MakeInteger(0, TypeId::INTEGER);  // nothing can match
        return std::make_unique<InNode>(std::move(operand), domain, std::move(ints), std::move(doubles), std::move(strings));
    }

    // LIKE on strings only; a literal pattern is classified once.
    static NodePtr Like(NodePtr str, NodePtr pattern) {
        if (str->Type().kind != Kind::STRING || pattern->Type().kind != Kind::STRING) {
            throw DatabaseException("LIKE needs text on both sides");
        }
        if (pattern->IsConstant()) {
            std::string text(View(AsConstant(*pattern).Str()));
            return Fold(std::make_unique<LikeNode>(std::move(str), nullptr, std::move(text)));
        }
        return Fold(std::make_unique<LikeNode>(std::move(str), std::move(pattern), std::string()));
    }

    // NOT any value; NEGATE numbers only, as today.
    NodePtr Unary(const UnaryOp& u) {
        NodePtr child = Build(*u.operand);
        if (u.op == UnaryOp::Op::NOT) return Fold(std::make_unique<NotNode>(std::move(child)));
        const Kind k = child->Type().kind;
        if (k == Kind::DATE || k == Kind::STRING) throw DatabaseException("NEGATE not supported for this type");
        return Fold(std::make_unique<NegateNode>(std::move(child)));
    }

    // The CASE's type is the common type of its results, as the planner types it.
    NodePtr Case(const CaseExpression& c) {
        std::vector<std::pair<NodePtr, NodePtr>> whens;
        for (const auto& [cond, result] : c.when_clauses) whens.emplace_back(Build(*cond), Build(*result));
        NodePtr otherwise = c.else_clause ? Build(*c.else_clause) : nullptr;
        if (whens.empty()) {
            if (otherwise) return otherwise;
            throw DatabaseException("CASE needs a WHEN");
        }

        std::vector<NodePtr*> results;
        for (auto& w : whens) results.push_back(&w.second);
        if (otherwise) results.push_back(&otherwise);

        // Next to DATE results, a constant string reads as a DATE, as comparisons do.
        const bool any_date = std::any_of(results.begin(), results.end(), [](NodePtr* r) { return (*r)->Type().kind == Kind::DATE; });
        for (NodePtr* r : results) {
            if (!any_date || (*r)->Type().kind != Kind::STRING) continue;
            if (!(*r)->IsConstant()) throw DatabaseException("CASE results have types that don't mix");
            *r = ParseDate(**r);
        }

        bool strings = true, dates = true, numbers = true, doubles = false, decimals = false;
        TypeId int_result = (*results.front())->Type().result;
        int int_digits = 0, scale = 0;
        for (NodePtr* r : results) {
            const VType t = (*r)->Type();
            strings = strings && t.kind == Kind::STRING;
            dates = dates && t.kind == Kind::DATE;
            numbers = numbers && (t.kind == Kind::INTEGER || t.kind == Kind::DECIMAL || t.kind == Kind::DOUBLE);
            doubles = doubles || t.kind == Kind::DOUBLE;
            decimals = decimals || t.kind == Kind::DECIMAL;
            
            if (t.kind == Kind::INTEGER) int_result = CommonType(int_result, t.result);
            if (t.kind == Kind::INTEGER || t.kind == Kind::DECIMAL) {
                const DecimalInfo info = InfoOf(**r);
                int_digits = std::max(int_digits, info.precision - info.scale);
                scale = std::max(scale, info.scale);
            }
        }

        VType type;
        uint8_t precision = 0;
        if (strings || dates) {
            type = (*results.front())->Type();
        } else if (!numbers) {
            throw DatabaseException("CASE results have types that don't mix");
        } else if (doubles) {
            type = kDoubleType;
            for (NodePtr* r : results) *r = ToDouble(std::move(*r));
        } else if (decimals) {
            precision = CapDigits(int_digits + scale);
            const bool wide = precision > 18;
            type = MakeType(Kind::DECIMAL, TypeId::DECIMAL, static_cast<uint8_t>(scale), wide);
            for (NodePtr* r : results) *r = ToDecimal(std::move(*r), static_cast<uint8_t>(scale), wide);
        } else {
            type = MakeType(Kind::INTEGER, int_result);
            precision = IntegerPrecision(int_result);
        }
        return Fold(std::make_unique<CaseNode>(type, precision, std::move(whens), std::move(otherwise)));
    }

    const InputLayout& in_;
};

// Adds every bound column the expression reads.
void Collect(const Expression& e, std::vector<uint32_t>& out) {
    switch (e.type) {
        case ExprType::COLUMN_REF: {
            const auto& c = static_cast<const ColumnRef&>(e);
            if (c.bound_index) out.push_back(*c.bound_index);
            break;
        }
        case ExprType::BINARY_OP: {
            const auto& b = static_cast<const BinaryOp&>(e);
            Collect(*b.left, out);
            Collect(*b.right, out);
            break;
        }
        case ExprType::UNARY_OP:
            Collect(*static_cast<const UnaryOp&>(e).operand, out);
            break;
        case ExprType::AGGREGATE: {
            const auto& a = static_cast<const Aggregate&>(e);
            if (a.arg) Collect(*a.arg, out);
            break;
        }
        case ExprType::CASE: {
            const auto& c = static_cast<const CaseExpression&>(e);
            for (const auto& [cond, result] : c.when_clauses) {
                Collect(*cond, out);
                Collect(*result, out);
            }
            if (c.else_clause) Collect(*c.else_clause, out);
            break;
        }
        case ExprType::LITERAL:
        case ExprType::STAR:
            break;
    }
}

}  // namespace

// Typing, folding and node building all happen here, once per expression.
std::unique_ptr<CompiledExpr> CompiledExpr::Compile(const Expression& expr, const InputLayout& in) {
    return Compiler(in).Build(expr);
}

// A left-deep chain of AND nodes:
// each part runs only on rows the ones before passed.
std::unique_ptr<CompiledExpr> CompiledExpr::CompileAnd(const std::vector<const Expression*>& parts, const InputLayout& in) {
    if (parts.empty()) return nullptr;
    Compiler compiler(in);
    NodePtr tree = compiler.Build(*parts[0]);
    for (size_t i = 1; i < parts.size(); ++i) tree = Fold(std::make_unique<AndNode>(std::move(tree), compiler.Build(*parts[i])));
    return tree;
}

// A bare column's batch vector as is, else expr evaluated into out.
const Vector& ValuesFor(const CompiledExpr& expr, const Batch& in, Vector& out, EvalScratch& scratch) {
    if (const Vector* v = expr.Direct(in)) {
        assert(v->Type().kind == expr.Type().kind && v->Type().scale == expr.Type().scale && v->Type().wide == expr.Type().wide &&
               "a column's batch vector holds another type");
        return *v;
    }
    expr.Evaluate(in, in.sel, out, scratch);
    return out;
}

// Collects, then sorts and drops repeats across everything in out.
void CollectColumns(const Expression& expr, std::vector<uint32_t>& out) {
    Collect(expr, out);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
}

// An AND missing a side stays whole.
void SplitAnd(const Expression& expr, std::vector<const Expression*>& out) {
    if (expr.type == ExprType::BINARY_OP) {
        const auto& b = static_cast<const BinaryOp&>(expr);
        if (b.op == BinaryOp::Op::AND && b.left && b.right) {
            SplitAnd(*b.left, out);
            SplitAnd(*b.right, out);
            return;
        }
    }
    out.push_back(&expr);
}

}  // namespace shilmandb::parallel
