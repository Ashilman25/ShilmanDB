#pragma once
#include "executor/parallel/batch.hpp"
#include "executor/parallel/vector.hpp"
#include "storage/columnar/string_arena.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace shilmandb::parallel {

// One column's values kept past their batch, in the array its type uses.
// Strings point into an arena that whoever owns the buffer keeps alive.
struct ColumnBuffer {
    std::vector<int64_t> i64;    // INTEGER, DATE and narrow DECIMAL
    std::vector<__int128> i128;  // wide DECIMAL
    std::vector<double> f64;     // DOUBLE
    std::vector<StringRef> str;  // STRING
};

// Appends v's active rows to buf. Strings are copied into arena,
// since a morsel's bytes die when its worker starts the next one.
void Append(const Vector& v, const Selection& sel, ColumnBuffer& buf, StringArena& arena);

// Writes rows[0..n) of buf into out's rows 0..n-1, in that order.
// out's type picks the array; strings come out plain, pointing into buf's arena.
void Gather(const ColumnBuffer& buf, const uint32_t* rows, uint32_t n, Vector& out);

// The same from a batch's vector: rows index from, plain or dictionary.
// Strings come out plain, pointing where from's strings point.
void Gather(const Vector& from, const uint32_t* rows, uint32_t n, Vector& out);

// Moves from's values after to's; strings keep pointing where they did.
void Extend(ColumnBuffer& to, ColumnBuffer&& from);

// Rows kept past their batches: one buffer per column. New strings go into arena;
// arenas taken from other stores keep those stores' strings valid.
struct ColumnStore {
    std::vector<ColumnBuffer> columns;
    StringArena arena;
    std::vector<StringArena> absorbed;
    size_t rows{0};

    // n empty columns.
    explicit ColumnStore(size_t n = 0) : columns(n) {}

    // Moves other's rows after these; its arenas come along, so its strings stay valid.
    void Absorb(ColumnStore&& other);
};

}  // namespace shilmandb::parallel
