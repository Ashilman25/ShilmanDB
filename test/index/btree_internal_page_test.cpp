#include <gtest/gtest.h>
#include "index/btree_internal_page.hpp"
#include "storage/slotted_page.hpp"
#include "common/exception.hpp"
#include <array>
#include <cstring>

namespace shilmandb {

class BTreeInternalPageTest : public ::testing::Test {
protected:
    std::array<char, PAGE_SIZE> page_{};

    void SetUp() override {
        BTreeInternalPage::Init(page_.data(), 42);
    }
};

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
TEST_F(BTreeInternalPageTest, InitSetsHeaderAndNumKeys) {
    auto* hdr = SlottedPage::GetHeader(page_.data());
    EXPECT_EQ(hdr->page_id, 42u);
    EXPECT_EQ(hdr->flags & 1u, 0u);  // bit 0 = 0 means non-leaf
    EXPECT_EQ(BTreeInternalPage::GetNumKeys(page_.data()), 0);
}

TEST_F(BTreeInternalPageTest, InitZerosPage) {
    // Dirty the page first, then re-init
    std::memset(page_.data(), 0xFF, PAGE_SIZE);
    BTreeInternalPage::Init(page_.data(), 7);

    auto* hdr = SlottedPage::GetHeader(page_.data());
    EXPECT_EQ(hdr->page_id, 7u);
    EXPECT_EQ(BTreeInternalPage::GetNumKeys(page_.data()), 0);

    // Spot-check that data region is zeroed
    EXPECT_EQ(page_[PAGE_HEADER_SIZE + 10], 0);
    EXPECT_EQ(page_[PAGE_SIZE - 1], 0);
}

// ---------------------------------------------------------------------------
// MaxKeys
// ---------------------------------------------------------------------------
TEST_F(BTreeInternalPageTest, MaxKeysInteger) {
    // 8170 / 8 - 1 = 1020 (one slot reserved for temporary overfull state)
    EXPECT_EQ(BTreeInternalPage::MaxKeys(TypeId::INTEGER), 1020);
}

TEST_F(BTreeInternalPageTest, MaxKeysBigint) {
    // 8170 / 12 - 1 = 679 (one slot reserved for temporary overfull state)
    EXPECT_EQ(BTreeInternalPage::MaxKeys(TypeId::BIGINT), 679);
}

TEST_F(BTreeInternalPageTest, MaxKeysDecimal) {
    EXPECT_EQ(BTreeInternalPage::MaxKeys(TypeId::DECIMAL), 679);
}

TEST_F(BTreeInternalPageTest, MaxKeysDate) {
    EXPECT_EQ(BTreeInternalPage::MaxKeys(TypeId::DATE), 1020);
}

// ---------------------------------------------------------------------------
// SetKey / GetKey / SetChild / GetChild — manual layout
// ---------------------------------------------------------------------------
TEST_F(BTreeInternalPageTest, SetAndGetKeysInteger) {
    // Manually build a node with 3 keys: [child0][10][child1][20][child2][30][child3]
    BTreeInternalPage::SetNumKeys(page_.data(), 3);

    BTreeInternalPage::SetChild(page_.data(), 0, 100, TypeId::INTEGER);
    BTreeInternalPage::SetKey(page_.data(), 0, Value(10), TypeId::INTEGER);
    BTreeInternalPage::SetChild(page_.data(), 1, 101, TypeId::INTEGER);
    BTreeInternalPage::SetKey(page_.data(), 1, Value(20), TypeId::INTEGER);
    BTreeInternalPage::SetChild(page_.data(), 2, 102, TypeId::INTEGER);
    BTreeInternalPage::SetKey(page_.data(), 2, Value(30), TypeId::INTEGER);
    BTreeInternalPage::SetChild(page_.data(), 3, 103, TypeId::INTEGER);

    EXPECT_EQ(BTreeInternalPage::GetKey(page_.data(), 0, TypeId::INTEGER), Value(10));
    EXPECT_EQ(BTreeInternalPage::GetKey(page_.data(), 1, TypeId::INTEGER), Value(20));
    EXPECT_EQ(BTreeInternalPage::GetKey(page_.data(), 2, TypeId::INTEGER), Value(30));

    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 0, TypeId::INTEGER), 100u);
    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 1, TypeId::INTEGER), 101u);
    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 2, TypeId::INTEGER), 102u);
    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 3, TypeId::INTEGER), 103u);
}

TEST_F(BTreeInternalPageTest, SetAndGetKeysBigint) {
    BTreeInternalPage::SetNumKeys(page_.data(), 2);

    BTreeInternalPage::SetChild(page_.data(), 0, 200, TypeId::BIGINT);
    BTreeInternalPage::SetKey(page_.data(), 0, Value(int64_t{1000000}), TypeId::BIGINT);
    BTreeInternalPage::SetChild(page_.data(), 1, 201, TypeId::BIGINT);
    BTreeInternalPage::SetKey(page_.data(), 1, Value(int64_t{2000000}), TypeId::BIGINT);
    BTreeInternalPage::SetChild(page_.data(), 2, 202, TypeId::BIGINT);

    EXPECT_EQ(BTreeInternalPage::GetKey(page_.data(), 0, TypeId::BIGINT), Value(int64_t{1000000}));
    EXPECT_EQ(BTreeInternalPage::GetKey(page_.data(), 1, TypeId::BIGINT), Value(int64_t{2000000}));
    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 0, TypeId::BIGINT), 200u);
    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 1, TypeId::BIGINT), 201u);
    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 2, TypeId::BIGINT), 202u);
}

TEST_F(BTreeInternalPageTest, SetAndGetKeysDecimal) {
    BTreeInternalPage::SetNumKeys(page_.data(), 1);

    BTreeInternalPage::SetChild(page_.data(), 0, 300, TypeId::DECIMAL);
    BTreeInternalPage::SetKey(page_.data(), 0, Value(3.14), TypeId::DECIMAL);
    BTreeInternalPage::SetChild(page_.data(), 1, 301, TypeId::DECIMAL);

    EXPECT_EQ(BTreeInternalPage::GetKey(page_.data(), 0, TypeId::DECIMAL), Value(3.14));
    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 0, TypeId::DECIMAL), 300u);
    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 1, TypeId::DECIMAL), 301u);
}

TEST_F(BTreeInternalPageTest, SetAndGetKeysDate) {
    BTreeInternalPage::SetNumKeys(page_.data(), 2);

    auto d1 = Value::MakeDate(100);
    auto d2 = Value::MakeDate(200);
    BTreeInternalPage::SetChild(page_.data(), 0, 400, TypeId::DATE);
    BTreeInternalPage::SetKey(page_.data(), 0, d1, TypeId::DATE);
    BTreeInternalPage::SetChild(page_.data(), 1, 401, TypeId::DATE);
    BTreeInternalPage::SetKey(page_.data(), 1, d2, TypeId::DATE);
    BTreeInternalPage::SetChild(page_.data(), 2, 402, TypeId::DATE);

    EXPECT_EQ(BTreeInternalPage::GetKey(page_.data(), 0, TypeId::DATE), d1);
    EXPECT_EQ(BTreeInternalPage::GetKey(page_.data(), 1, TypeId::DATE), d2);
    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 0, TypeId::DATE), 400u);
    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 2, TypeId::DATE), 402u);
}

// ---------------------------------------------------------------------------
// Lookup — binary search
// ---------------------------------------------------------------------------
TEST_F(BTreeInternalPageTest, LookupFindsCorrectChild) {
    // Node: [child0=10][key=20][child1=11][key=40][child2=12][key=60][child3=13]
    BTreeInternalPage::SetNumKeys(page_.data(), 3);
    BTreeInternalPage::SetChild(page_.data(), 0, 10, TypeId::INTEGER);
    BTreeInternalPage::SetKey(page_.data(), 0, Value(20), TypeId::INTEGER);
    BTreeInternalPage::SetChild(page_.data(), 1, 11, TypeId::INTEGER);
    BTreeInternalPage::SetKey(page_.data(), 1, Value(40), TypeId::INTEGER);
    BTreeInternalPage::SetChild(page_.data(), 2, 12, TypeId::INTEGER);
    BTreeInternalPage::SetKey(page_.data(), 2, Value(60), TypeId::INTEGER);
    BTreeInternalPage::SetChild(page_.data(), 3, 13, TypeId::INTEGER);

    // key < 20 → child0
    EXPECT_EQ(BTreeInternalPage::Lookup(page_.data(), Value(5), TypeId::INTEGER), 10u);
    // key == 20 → child1 (20 is not > 20, so we go right of key[0])
    EXPECT_EQ(BTreeInternalPage::Lookup(page_.data(), Value(20), TypeId::INTEGER), 11u);
    // key between 20 and 40 → child1
    EXPECT_EQ(BTreeInternalPage::Lookup(page_.data(), Value(30), TypeId::INTEGER), 11u);
    // key == 40 → child2
    EXPECT_EQ(BTreeInternalPage::Lookup(page_.data(), Value(40), TypeId::INTEGER), 12u);
    // key between 40 and 60 → child2
    EXPECT_EQ(BTreeInternalPage::Lookup(page_.data(), Value(50), TypeId::INTEGER), 12u);
    // key == 60 → child3
    EXPECT_EQ(BTreeInternalPage::Lookup(page_.data(), Value(60), TypeId::INTEGER), 13u);
    // key > 60 → child3
    EXPECT_EQ(BTreeInternalPage::Lookup(page_.data(), Value(100), TypeId::INTEGER), 13u);
}

TEST_F(BTreeInternalPageTest, LookupSingleKey) {
    // Node: [child0=5][key=50][child1=6]
    BTreeInternalPage::SetNumKeys(page_.data(), 1);
    BTreeInternalPage::SetChild(page_.data(), 0, 5, TypeId::INTEGER);
    BTreeInternalPage::SetKey(page_.data(), 0, Value(50), TypeId::INTEGER);
    BTreeInternalPage::SetChild(page_.data(), 1, 6, TypeId::INTEGER);

    EXPECT_EQ(BTreeInternalPage::Lookup(page_.data(), Value(10), TypeId::INTEGER), 5u);
    EXPECT_EQ(BTreeInternalPage::Lookup(page_.data(), Value(50), TypeId::INTEGER), 6u);
    EXPECT_EQ(BTreeInternalPage::Lookup(page_.data(), Value(99), TypeId::INTEGER), 6u);
}

TEST_F(BTreeInternalPageTest, LookupBigintKeys) {
    BTreeInternalPage::SetNumKeys(page_.data(), 2);
    BTreeInternalPage::SetChild(page_.data(), 0, 10, TypeId::BIGINT);
    BTreeInternalPage::SetKey(page_.data(), 0, Value(int64_t{1000}), TypeId::BIGINT);
    BTreeInternalPage::SetChild(page_.data(), 1, 11, TypeId::BIGINT);
    BTreeInternalPage::SetKey(page_.data(), 1, Value(int64_t{2000}), TypeId::BIGINT);
    BTreeInternalPage::SetChild(page_.data(), 2, 12, TypeId::BIGINT);

    EXPECT_EQ(BTreeInternalPage::Lookup(page_.data(), Value(int64_t{500}), TypeId::BIGINT), 10u);
    EXPECT_EQ(BTreeInternalPage::Lookup(page_.data(), Value(int64_t{1000}), TypeId::BIGINT), 11u);
    EXPECT_EQ(BTreeInternalPage::Lookup(page_.data(), Value(int64_t{1500}), TypeId::BIGINT), 11u);
    EXPECT_EQ(BTreeInternalPage::Lookup(page_.data(), Value(int64_t{2000}), TypeId::BIGINT), 12u);
    EXPECT_EQ(BTreeInternalPage::Lookup(page_.data(), Value(int64_t{9999}), TypeId::BIGINT), 12u);
}

// ---------------------------------------------------------------------------
// InsertAfter
// ---------------------------------------------------------------------------
TEST_F(BTreeInternalPageTest, InsertAfterBuildsNode) {
    // Start with root: [child0=10][key=50][child1=11]
    BTreeInternalPage::SetNumKeys(page_.data(), 1);
    BTreeInternalPage::SetChild(page_.data(), 0, 10, TypeId::INTEGER);
    BTreeInternalPage::SetKey(page_.data(), 0, Value(50), TypeId::INTEGER);
    BTreeInternalPage::SetChild(page_.data(), 1, 11, TypeId::INTEGER);

    // Insert (key=25, right_child=12) after left_child=10
    // Result: [10][25][12][50][11]
    BTreeInternalPage::InsertAfter(page_.data(), 10, Value(25), 12, TypeId::INTEGER);

    EXPECT_EQ(BTreeInternalPage::GetNumKeys(page_.data()), 2);
    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 0, TypeId::INTEGER), 10u);
    EXPECT_EQ(BTreeInternalPage::GetKey(page_.data(), 0, TypeId::INTEGER), Value(25));
    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 1, TypeId::INTEGER), 12u);
    EXPECT_EQ(BTreeInternalPage::GetKey(page_.data(), 1, TypeId::INTEGER), Value(50));
    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 2, TypeId::INTEGER), 11u);
}

TEST_F(BTreeInternalPageTest, InsertAfterAtEnd) {
    // Start with: [10][50][11]
    BTreeInternalPage::SetNumKeys(page_.data(), 1);
    BTreeInternalPage::SetChild(page_.data(), 0, 10, TypeId::INTEGER);
    BTreeInternalPage::SetKey(page_.data(), 0, Value(50), TypeId::INTEGER);
    BTreeInternalPage::SetChild(page_.data(), 1, 11, TypeId::INTEGER);

    // Insert (key=75, right_child=12) after left_child=11
    // Result: [10][50][11][75][12]
    BTreeInternalPage::InsertAfter(page_.data(), 11, Value(75), 12, TypeId::INTEGER);

    EXPECT_EQ(BTreeInternalPage::GetNumKeys(page_.data()), 2);
    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 0, TypeId::INTEGER), 10u);
    EXPECT_EQ(BTreeInternalPage::GetKey(page_.data(), 0, TypeId::INTEGER), Value(50));
    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 1, TypeId::INTEGER), 11u);
    EXPECT_EQ(BTreeInternalPage::GetKey(page_.data(), 1, TypeId::INTEGER), Value(75));
    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 2, TypeId::INTEGER), 12u);
}

TEST_F(BTreeInternalPageTest, InsertAfterMultiple) {
    // Build up: start with [10][50][11], insert 3 more keys
    BTreeInternalPage::SetNumKeys(page_.data(), 1);
    BTreeInternalPage::SetChild(page_.data(), 0, 10, TypeId::INTEGER);
    BTreeInternalPage::SetKey(page_.data(), 0, Value(50), TypeId::INTEGER);
    BTreeInternalPage::SetChild(page_.data(), 1, 11, TypeId::INTEGER);

    BTreeInternalPage::InsertAfter(page_.data(), 10, Value(25), 12, TypeId::INTEGER);
    BTreeInternalPage::InsertAfter(page_.data(), 12, Value(37), 13, TypeId::INTEGER);
    BTreeInternalPage::InsertAfter(page_.data(), 11, Value(75), 14, TypeId::INTEGER);

    // Expected: [10][25][12][37][13][50][11][75][14]
    EXPECT_EQ(BTreeInternalPage::GetNumKeys(page_.data()), 4);
    EXPECT_EQ(BTreeInternalPage::GetKey(page_.data(), 0, TypeId::INTEGER), Value(25));
    EXPECT_EQ(BTreeInternalPage::GetKey(page_.data(), 1, TypeId::INTEGER), Value(37));
    EXPECT_EQ(BTreeInternalPage::GetKey(page_.data(), 2, TypeId::INTEGER), Value(50));
    EXPECT_EQ(BTreeInternalPage::GetKey(page_.data(), 3, TypeId::INTEGER), Value(75));

    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 0, TypeId::INTEGER), 10u);
    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 1, TypeId::INTEGER), 12u);
    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 2, TypeId::INTEGER), 13u);
    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 3, TypeId::INTEGER), 11u);
    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 4, TypeId::INTEGER), 14u);
}

TEST_F(BTreeInternalPageTest, InsertAfterThenLookup) {
    // Build a node via InsertAfter, then verify Lookup routes correctly
    BTreeInternalPage::SetNumKeys(page_.data(), 1);
    BTreeInternalPage::SetChild(page_.data(), 0, 10, TypeId::INTEGER);
    BTreeInternalPage::SetKey(page_.data(), 0, Value(50), TypeId::INTEGER);
    BTreeInternalPage::SetChild(page_.data(), 1, 11, TypeId::INTEGER);

    BTreeInternalPage::InsertAfter(page_.data(), 10, Value(25), 12, TypeId::INTEGER);
    BTreeInternalPage::InsertAfter(page_.data(), 11, Value(75), 13, TypeId::INTEGER);

    // Node: [10][25][12][50][11][75][13]
    EXPECT_EQ(BTreeInternalPage::Lookup(page_.data(), Value(1), TypeId::INTEGER), 10u);
    EXPECT_EQ(BTreeInternalPage::Lookup(page_.data(), Value(25), TypeId::INTEGER), 12u);
    EXPECT_EQ(BTreeInternalPage::Lookup(page_.data(), Value(30), TypeId::INTEGER), 12u);
    EXPECT_EQ(BTreeInternalPage::Lookup(page_.data(), Value(50), TypeId::INTEGER), 11u);
    EXPECT_EQ(BTreeInternalPage::Lookup(page_.data(), Value(60), TypeId::INTEGER), 11u);
    EXPECT_EQ(BTreeInternalPage::Lookup(page_.data(), Value(75), TypeId::INTEGER), 13u);
    EXPECT_EQ(BTreeInternalPage::Lookup(page_.data(), Value(100), TypeId::INTEGER), 13u);
}

// ---------------------------------------------------------------------------
// NeedsSplit
// ---------------------------------------------------------------------------
TEST_F(BTreeInternalPageTest, NeedsSplitFalseWhenUnderCapacity) {
    BTreeInternalPage::SetNumKeys(page_.data(), 5);
    EXPECT_FALSE(BTreeInternalPage::NeedsSplit(page_.data(), TypeId::INTEGER));
}

TEST_F(BTreeInternalPageTest, NeedsSplitFalseAtExactCapacity) {
    auto max_keys = BTreeInternalPage::MaxKeys(TypeId::INTEGER);
    BTreeInternalPage::SetNumKeys(page_.data(), max_keys);
    EXPECT_FALSE(BTreeInternalPage::NeedsSplit(page_.data(), TypeId::INTEGER));
}

TEST_F(BTreeInternalPageTest, NeedsSplitTrueWhenOverfull) {
    auto max_keys = BTreeInternalPage::MaxKeys(TypeId::INTEGER);
    BTreeInternalPage::SetNumKeys(page_.data(), max_keys + 1);
    EXPECT_TRUE(BTreeInternalPage::NeedsSplit(page_.data(), TypeId::INTEGER));
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------
TEST_F(BTreeInternalPageTest, KeySizeThrowsOnVarchar) {
    EXPECT_THROW((void)BTreeInternalPage::MaxKeys(TypeId::VARCHAR), DatabaseException);
}

TEST_F(BTreeInternalPageTest, KeySizeThrowsOnInvalid) {
    EXPECT_THROW((void)BTreeInternalPage::MaxKeys(TypeId::INVALID), DatabaseException);
}

TEST_F(BTreeInternalPageTest, InsertAfterBigintKey) {
    BTreeInternalPage::SetNumKeys(page_.data(), 1);
    BTreeInternalPage::SetChild(page_.data(), 0, 10, TypeId::BIGINT);
    BTreeInternalPage::SetKey(page_.data(), 0, Value(int64_t{500}), TypeId::BIGINT);
    BTreeInternalPage::SetChild(page_.data(), 1, 11, TypeId::BIGINT);

    BTreeInternalPage::InsertAfter(page_.data(), 10, Value(int64_t{250}), 12, TypeId::BIGINT);

    EXPECT_EQ(BTreeInternalPage::GetNumKeys(page_.data()), 2);
    EXPECT_EQ(BTreeInternalPage::GetKey(page_.data(), 0, TypeId::BIGINT), Value(int64_t{250}));
    EXPECT_EQ(BTreeInternalPage::GetKey(page_.data(), 1, TypeId::BIGINT), Value(int64_t{500}));
    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 0, TypeId::BIGINT), 10u);
    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 1, TypeId::BIGINT), 12u);
    EXPECT_EQ(BTreeInternalPage::GetChild(page_.data(), 2, TypeId::BIGINT), 11u);
}

}  // namespace shilmandb
