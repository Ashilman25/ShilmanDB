#include <gtest/gtest.h>
#include "executor/parallel/batch.hpp"
#include "executor/parallel/vector.hpp"
#include "storage/columnar/string_arena.hpp"
#include "types/column_type.hpp"
#include <string>
#include <vector>

namespace shilmandb::parallel {

namespace {

// A string of len bytes that names its index at both ends, so overlaps show.
std::string MakeString(uint32_t i, size_t len) {
    std::string s(len, static_cast<char>('a' + i % 26));
    const std::string tag = std::to_string(i);
    if (len >= 2 * tag.size()) {
        s.replace(0, tag.size(), tag);
        s.replace(len - tag.size(), tag.size(), tag);
    }
    return s;
}

// Counts the refs whose bytes no longer match their original string.
size_t CountMismatches(const std::vector<StringRef>& refs, const std::vector<std::string>& originals) {
    size_t mismatches = 0;
    for (size_t i = 0; i < refs.size(); ++i) {
        if (View(refs[i]) != originals[i]) {
            ++mismatches;
        }
    }
    return mismatches;
}

// Builds a VType by hand for the expected side of a comparison.
VType Expect(Kind kind, TypeId result, uint8_t scale = 0, bool wide = false) {
    VType t;
    t.kind = kind;
    t.scale = scale;
    t.wide = wide;
    t.result = result;
    return t;
}

}  // namespace

TEST(ParallelVectorTest, StringArenaKeepsBytesUntilClear) {
    StringArena arena;
    std::vector<std::string> originals;
    std::vector<StringRef> refs;
    originals.reserve(10'000);
    refs.reserve(10'000);

    // 9,001 is prime, so the lengths cover every size from 0 to 9,000 bytes.
    size_t total = 0;
    for (uint32_t i = 0; i < 10'000; ++i) {
        originals.push_back(MakeString(i, (i * 7919u) % 9001u));
        refs.push_back(arena.Copy(originals.back()));
        total += originals.back().size();
    }
    EXPECT_EQ(CountMismatches(refs, originals), 0u);
    EXPECT_GE(arena.BytesUsed(), total);

    // A value bigger than a block gets its own, and nothing earlier moves.
    originals.push_back(MakeString(10'000, 100'000));
    refs.push_back(arena.Copy(originals.back()));
    total += originals.back().size();
    EXPECT_EQ(CountMismatches(refs, originals), 0u);
    EXPECT_GE(arena.BytesUsed(), total);

    arena.Clear();
    EXPECT_EQ(arena.BytesUsed(), 0u);
    const StringRef again = arena.Copy("after clear");
    EXPECT_EQ(View(again), "after clear");
    EXPECT_EQ(View(arena.Copy("")), "");
    EXPECT_NE(arena.Copy("").data, nullptr);

    // A later copy that needs a new block leaves the reused first block alone.
    const std::string fill = MakeString(20'000, StringArena::kBlockSize);
    const StringRef full = arena.Copy(fill);
    EXPECT_EQ(View(again), "after clear");
    EXPECT_EQ(View(full), fill);
}

TEST(ParallelVectorTest, StringArenaBytesHeldCountsBlocks) {
    StringArena arena;
    EXPECT_EQ(arena.BytesHeld(), 0u);
    EXPECT_EQ(View(arena.Copy("")), "");
    EXPECT_EQ(arena.BytesHeld(), 0u) << "an empty copy takes no block";

    // One short string takes a whole block; a big value takes exactly its bytes.
    (void)arena.Copy("abc");
    EXPECT_EQ(arena.BytesUsed(), 3u);
    EXPECT_EQ(arena.BytesHeld(), StringArena::kBlockSize);
    const std::string big(100'000, 'b');
    (void)arena.Copy(big);
    EXPECT_EQ(arena.BytesHeld(), StringArena::kBlockSize + big.size());

    // A string that won't fit the rest of the block starts a second one.
    (void)arena.Copy(std::string(StringArena::kBlockSize - 2, 'c'));
    EXPECT_EQ(arena.BytesHeld(), 2 * StringArena::kBlockSize + big.size());

    // Clear keeps the first block only.
    arena.Clear();
    EXPECT_EQ(arena.BytesHeld(), StringArena::kBlockSize);
}

TEST(ParallelVectorTest, VTypeOfCoversEveryStorageType) {
    struct Case {
        ColumnType storage;
        VType expected;
    };
    const std::vector<Case> cases = {
        {ColumnType::Boolean(), Expect(Kind::INTEGER, TypeId::INTEGER)},
        {ColumnType::SmallInt(), Expect(Kind::INTEGER, TypeId::INTEGER)},
        {ColumnType::Integer(), Expect(Kind::INTEGER, TypeId::INTEGER)},
        {ColumnType::BigInt(), Expect(Kind::INTEGER, TypeId::BIGINT)},
        {ColumnType::Numeric(15, 2), Expect(Kind::DECIMAL, TypeId::DECIMAL, 2, false)},
        {ColumnType::Numeric(18, 4), Expect(Kind::DECIMAL, TypeId::DECIMAL, 4, false)},
        {ColumnType::Numeric(19, 0), Expect(Kind::DECIMAL, TypeId::DECIMAL, 0, true)},
        {ColumnType::Numeric(20, 2), Expect(Kind::DECIMAL, TypeId::DECIMAL, 2, true)},
        {ColumnType::Numeric(38, 10), Expect(Kind::DECIMAL, TypeId::DECIMAL, 10, true)},
        {ColumnType::Double(), Expect(Kind::DOUBLE, TypeId::DECIMAL)},
        {ColumnType::Date(), Expect(Kind::DATE, TypeId::DATE)},
        {ColumnType::Timestamp(), Expect(Kind::STRING, TypeId::VARCHAR)},
        {ColumnType::Varchar(25), Expect(Kind::STRING, TypeId::VARCHAR)},
        {ColumnType::Text(), Expect(Kind::STRING, TypeId::VARCHAR)},
    };

    for (const Case& c : cases) {
        SCOPED_TRACE(ColumnTypeName(c.storage));
        const VType got = VTypeOf(c.storage);
        EXPECT_TRUE(got == c.expected);
        EXPECT_EQ(got.kind, c.expected.kind);
        EXPECT_EQ(got.scale, c.expected.scale);
        EXPECT_EQ(got.wide, c.expected.wide);
        EXPECT_EQ(got.result, c.expected.result);

        // Must agree with what the columnar scan hands the executor today.
        EXPECT_EQ(got.result, ExecutorType(c.storage));
        EXPECT_EQ(got.wide, PhysicalKindOf(c.storage) == PhysicalKind::INT128);
    }

    // Every one of the ten storage ids is in the table above.
    std::vector<bool> seen(11, false);
    for (const Case& c : cases) {
        seen[static_cast<size_t>(c.storage.id)] = true;
    }
    for (int id = 1; id <= 10; ++id) {
        EXPECT_TRUE(seen[id]) << "storage id " << id << " has no case";
    }

    EXPECT_FALSE(VTypeOf(ColumnType::Numeric(15, 2)) == VTypeOf(ColumnType::Numeric(20, 2)));
    EXPECT_FALSE(VTypeOf(ColumnType::Numeric(15, 2)) == VTypeOf(ColumnType::Numeric(15, 3)));
    EXPECT_FALSE(VTypeOf(ColumnType::Integer()) == VTypeOf(ColumnType::BigInt()));
}

TEST(ParallelVectorTest, VectorDictionaryAndPlainReadTheSame) {
    StringArena arena;
    const std::vector<StringRef> entries = {arena.Copy("AIR"), arena.Copy("MAIL"), arena.Copy(""),
                                            arena.Copy("REG AIR")};
    const VType text = VTypeOf(ColumnType::Text());
    Vector dict(text);
    Vector plain(text);
    EXPECT_TRUE(dict.Type() == text);

    dict.SetDictionary(&entries);
    EXPECT_TRUE(dict.IsDictionary());
    EXPECT_FALSE(plain.IsDictionary());
    EXPECT_EQ(&dict.Entries(), &entries);

    for (uint32_t row = 0; row < kBatchSize; ++row) {
        const auto code = static_cast<uint32_t>((row * 5 + 1) % entries.size());
        dict.Codes()[row] = code;
        plain.Str()[row] = entries[code];
    }

    size_t mismatches = 0;
    for (uint32_t row = 0; row < kBatchSize; ++row) {
        const uint32_t code = dict.Codes()[row];
        if (View(dict.StrAt(row)) != View(plain.StrAt(row)) || View(dict.StrAt(row)) != View(entries[code])) {
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0u);

    dict.SetDictionary(nullptr);
    EXPECT_FALSE(dict.IsDictionary());
}

TEST(ParallelVectorTest, SelectionSetAllAndNarrow) {
    Selection sel;
    sel.SetAll(5);
    EXPECT_TRUE(sel.IsAll());
    EXPECT_EQ(sel.Size(), 5u);
    for (uint32_t i = 0; i < 5; ++i) {
        EXPECT_EQ(sel[i], i);
    }

    sel.Data()[0] = 1;
    sel.Data()[1] = 3;
    sel.Resize(2);
    EXPECT_FALSE(sel.IsAll());
    EXPECT_EQ(sel.Size(), 2u);
    EXPECT_EQ(sel[0], 1u);
    EXPECT_EQ(sel[1], 3u);

    // Reset makes every row active and returns dictionary slots to plain form.
    StringArena arena;
    const std::vector<StringRef> entries = {arena.Copy("x")};
    Batch batch;
    batch.slots.emplace_back(VTypeOf(ColumnType::Integer()));
    batch.slots.emplace_back(VTypeOf(ColumnType::Text()));
    batch.slots[1].SetDictionary(&entries);
    batch.sel.Resize(0);
    batch.Reset(7);
    EXPECT_EQ(batch.rows, 7u);
    EXPECT_EQ(batch.Active(), 7u);
    EXPECT_TRUE(batch.sel.IsAll());
    EXPECT_FALSE(batch.slots[1].IsDictionary());
}

}  // namespace shilmandb::parallel
