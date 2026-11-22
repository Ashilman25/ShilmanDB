#include "executor/parallel/column_buffer.hpp"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iterator>
#include <utility>

namespace shilmandb::parallel {

namespace {

// Calls fn(row) for every active row.
template <typename Fn>
void ForActive(const Selection& sel, Fn fn) {
    for (uint32_t k = 0; k < sel.Size(); ++k) fn(static_cast<uint32_t>(sel[k]));
}

// out[i] = from[rows[i]] for the first n rows; from holds `size` values.
template <typename T>
void GatherArray(const T* from, size_t size, const uint32_t* rows, uint32_t n, T* out) {
    for (uint32_t i = 0; i < n; ++i) {
        assert(rows[i] < size && "gathered row past the end");
        out[i] = from[rows[i]];
    }
    (void)size;
}

// Moves from's items after to's, or takes from whole when to is empty.
template <typename T>
void MoveAfter(std::vector<T>& to, std::vector<T>& from) {
    if (to.empty()) {
        to = std::move(from);
    } else {
        to.insert(to.end(), std::make_move_iterator(from.begin()), std::make_move_iterator(from.end()));
    }
    from.clear();
}

}  // namespace

// Strings go through StrAt, so plain and dictionary vectors both work.
void Append(const Vector& v, const Selection& sel, ColumnBuffer& buf, StringArena& arena) {
    switch (ArrayOf(v.Type())) {
        case ValueArray::STR:
            ForActive(sel, [&](uint32_t i) { buf.str.push_back(arena.Copy(View(v.StrAt(i)))); });
            break;
        case ValueArray::F64: {
            const double* a = v.F64();
            ForActive(sel, [&](uint32_t i) { buf.f64.push_back(a[i]); });
            break;
        }
        case ValueArray::I128: {
            const __int128* a = v.I128();
            ForActive(sel, [&](uint32_t i) { buf.i128.push_back(a[i]); });
            break;
        }
        case ValueArray::I64: {
            const int64_t* a = v.I64();
            ForActive(sel, [&](uint32_t i) { buf.i64.push_back(a[i]); });
            break;
        }
    }
}

// The vector ends up plain, so a dictionary left from an earlier batch is dropped.
void Gather(const ColumnBuffer& buf, const uint32_t* rows, uint32_t n, Vector& out) {
    assert(n <= kBatchSize && "gathered more rows than a vector holds");
    switch (ArrayOf(out.Type())) {
        case ValueArray::STR:
            out.SetDictionary(nullptr);
            GatherArray(buf.str.data(), buf.str.size(), rows, n, out.Str());
            break;
        case ValueArray::F64: GatherArray(buf.f64.data(), buf.f64.size(), rows, n, out.F64()); break;
        case ValueArray::I128: GatherArray(buf.i128.data(), buf.i128.size(), rows, n, out.I128()); break;
        case ValueArray::I64: GatherArray(buf.i64.data(), buf.i64.size(), rows, n, out.I64()); break;
    }
}

// Reads strings through StrAt, so a dictionary vector gives plain strings.
void Gather(const Vector& from, const uint32_t* rows, uint32_t n, Vector& out) {
    assert(n <= kBatchSize && "gathered more rows than a vector holds");
    assert(from.Type() == out.Type() && "gathering into a vector of another type");
    switch (ArrayOf(from.Type())) {
        case ValueArray::STR: {
            out.SetDictionary(nullptr);
            StringRef* o = out.Str();
            for (uint32_t i = 0; i < n; ++i) o[i] = from.StrAt(rows[i]);
            break;
        }
        case ValueArray::F64: GatherArray(from.F64(), kBatchSize, rows, n, out.F64()); break;
        case ValueArray::I128: GatherArray(from.I128(), kBatchSize, rows, n, out.I128()); break;
        case ValueArray::I64: GatherArray(from.I64(), kBatchSize, rows, n, out.I64()); break;
    }
}

// Only one array is in use; moving all four is just as simple.
void Extend(ColumnBuffer& to, ColumnBuffer&& from) {
    MoveAfter(to.i64, from.i64);
    MoveAfter(to.i128, from.i128);
    MoveAfter(to.f64, from.f64);
    MoveAfter(to.str, from.str);
}

// Signs, then NaN bits, break the ties that < leaves.
int TotalOrder(double a, double b) {
    const auto order = [](auto x, auto y) { return x < y ? -1 : (y < x ? 1 : 0); };
    const bool a_nan = std::isnan(a);
    const bool b_nan = std::isnan(b);
    
    if (a_nan != b_nan) return a_nan ? 1 : -1;
    if (a_nan) {
        uint64_t x = 0;
        uint64_t y = 0;
        std::memcpy(&x, &a, sizeof(a));
        std::memcpy(&y, &b, sizeof(b));
        return order(x, y);
    }
    const int c = order(a, b);
    return c != 0 ? c : order(std::signbit(b), std::signbit(a));
}

// A StringArena's blocks never move, so moving the arenas keeps every string in place.
void ColumnStore::Absorb(ColumnStore&& other) {
    assert(other.columns.size() == columns.size() && "absorbing a store with other columns");
    for (size_t k = 0; k < columns.size(); ++k) Extend(columns[k], std::move(other.columns[k]));
    absorbed.push_back(std::move(other.arena));
    for (StringArena& a : other.absorbed) absorbed.push_back(std::move(a));
    other.absorbed.clear();
    rows += other.rows;
    other.rows = 0;
}

}  // namespace shilmandb::parallel
