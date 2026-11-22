#pragma once
#include "storage/columnar/string_arena.hpp"
#include "types/column_type.hpp"
#include "types/types.hpp"
#include <cassert>
#include <cstdint>
#include <vector>

namespace shilmandb::parallel {

// Values per vector and rows per batch.
inline constexpr uint32_t kBatchSize = 2048;

// How a vector holds its values in memory.
enum class Kind : uint8_t { INTEGER, DATE, DECIMAL, DOUBLE, STRING };

// A value's in-memory form and the TypeId it becomes on the way out.
struct VType {
    Kind kind{Kind::INTEGER};
    uint8_t scale{0};                // DECIMAL only
    bool wide{false};                // DECIMAL held in 128 bits
    TypeId result{TypeId::INTEGER};  // what the value becomes when it leaves the executor

    // Equal when all four fields match.
    bool operator==(const VType& o) const;
    bool operator!=(const VType& o) const;
};

// The in-memory form for a column's storage type. Unknown ids throw.
[[nodiscard]] VType VTypeOf(const ColumnType& storage);

// Up to kBatchSize values of one column, in the array its kind uses.
// A STRING vector is plain (one StringRef per row) or a dictionary (codes per row).
class Vector {
public:
    // Room for kBatchSize values of the given type.
    explicit Vector(VType type);

    // The type the vector was built with.
    [[nodiscard]] const VType& Type() const { return type_; }

    // The array for each kind; the wrong kind is a Debug assert.
    int64_t* I64() {
        assert(HoldsI64() && "I64() on a vector that isn't 64-bit integers");
        return i64_.data();
    }
    const int64_t* I64() const {
        assert(HoldsI64() && "I64() on a vector that isn't 64-bit integers");
        return i64_.data();
    }
    __int128* I128() {
        assert(HoldsI128() && "I128() on a vector that isn't wide decimals");
        return i128_.data();
    }
    const __int128* I128() const {
        assert(HoldsI128() && "I128() on a vector that isn't wide decimals");
        return i128_.data();
    }
    double* F64() {
        assert(type_.kind == Kind::DOUBLE && "F64() on a vector that isn't doubles");
        return f64_.data();
    }
    const double* F64() const {
        assert(type_.kind == Kind::DOUBLE && "F64() on a vector that isn't doubles");
        return f64_.data();
    }
    StringRef* Str() {
        assert(type_.kind == Kind::STRING && "Str() on a vector that isn't strings");
        return str_.data();
    }
    const StringRef* Str() const {
        assert(type_.kind == Kind::STRING && "Str() on a vector that isn't strings");
        return str_.data();
    }

    // Switches to dictionary form over shared, read-only entries; nullptr goes back to plain.
    void SetDictionary(const std::vector<StringRef>* entries);

    // True while the vector reads its strings through a dictionary.
    [[nodiscard]] bool IsDictionary() const { return dict_ != nullptr; }

    // One code per row, an index into Entries().
    uint32_t* Codes() {
        assert(type_.kind == Kind::STRING && "Codes() on a vector that isn't strings");
        return codes_.data();
    }
    const uint32_t* Codes() const {
        assert(type_.kind == Kind::STRING && "Codes() on a vector that isn't strings");
        return codes_.data();
    }

    // The dictionary's entries; only in dictionary form.
    [[nodiscard]] const std::vector<StringRef>& Entries() const {
        assert(IsDictionary() && "Entries() on a plain vector");
        return *dict_;
    }

    // A row's string in either form.
    [[nodiscard]] StringRef StrAt(uint32_t row) const {
        assert(type_.kind == Kind::STRING && row < kBatchSize);
        if (dict_ != nullptr) {
            assert(codes_[row] < dict_->size() && "dictionary code out of range");
            return (*dict_)[codes_[row]];
        }
        return str_[row];
    }

private:
    // INTEGER, DATE and narrow DECIMAL share the 64-bit array.
    [[nodiscard]] bool HoldsI64() const {
        return type_.kind == Kind::INTEGER || type_.kind == Kind::DATE || (type_.kind == Kind::DECIMAL && !type_.wide);
    }

    // Only wide DECIMAL uses the 128-bit array.
    [[nodiscard]] bool HoldsI128() const { return type_.kind == Kind::DECIMAL && type_.wide; }

    VType type_;
    // Only the arrays this kind uses are sized; the rest stay empty.
    std::vector<int64_t> i64_;
    std::vector<__int128> i128_;
    std::vector<double> f64_;
    std::vector<StringRef> str_;
    std::vector<uint32_t> codes_;
    const std::vector<StringRef>* dict_{nullptr};  // not owned
};

}  // namespace shilmandb::parallel
