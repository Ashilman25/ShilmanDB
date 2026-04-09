#include <gtest/gtest.h>
#include "index/btree_leaf_page.hpp"
#include "storage/slotted_page.hpp"
#include "common/exception.hpp"
#include <array>
#include <cstring>

namespace shilmandb {

class BTreeLeafPageTest : public ::testing::Test {
protected:
    std::array<char, PAGE_SIZE> page_{};

    void SetUp() override {
        BTreeLeafPage::Init(page_.data(), 42);
    }
};

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
TEST_F(BTreeLeafPageTest, InitSetsHeaderAndLeafFlag) {
    auto* hdr = SlottedPage::GetHeader(page_.data());
    EXPECT_EQ(hdr->page_id, 42u);
    EXPECT_EQ(hdr->flags & 1u, 1u);  // bit 0 = 1 means leaf
    EXPECT_EQ(BTreeLeafPage::GetNumKeys(page_.data()), 0);
    EXPECT_EQ(BTreeLeafPage::GetNextLeafPageId(page_.data()), INVALID_PAGE_ID);
}

TEST_F(BTreeLeafPageTest, InitZerosPageAndSetsNextLeaf) {
    std::memset(page_.data(), 0xFF, PAGE_SIZE);
    BTreeLeafPage::Init(page_.data(), 7);

    auto* hdr = SlottedPage::GetHeader(page_.data());
    EXPECT_EQ(hdr->page_id, 7u);
    EXPECT_EQ(hdr->flags & 1u, 1u);
    EXPECT_EQ(BTreeLeafPage::GetNumKeys(page_.data()), 0);
    EXPECT_EQ(BTreeLeafPage::GetNextLeafPageId(page_.data()), INVALID_PAGE_ID);
    EXPECT_EQ(page_[PAGE_SIZE - 1], 0);
}

TEST_F(BTreeLeafPageTest, SetAndGetNextLeafPageId) {
    BTreeLeafPage::SetNextLeafPageId(page_.data(), 99);
    EXPECT_EQ(BTreeLeafPage::GetNextLeafPageId(page_.data()), 99u);

    BTreeLeafPage::SetNextLeafPageId(page_.data(), INVALID_PAGE_ID);
    EXPECT_EQ(BTreeLeafPage::GetNextLeafPageId(page_.data()), INVALID_PAGE_ID);
}

// ---------------------------------------------------------------------------
// MaxKeys
// ---------------------------------------------------------------------------
TEST_F(BTreeLeafPageTest, MaxKeysInteger) {
    // (8192 - 22) / (4 + 6) = 8170 / 10 = 817
    EXPECT_EQ(BTreeLeafPage::MaxKeys(TypeId::INTEGER), 817);
}

TEST_F(BTreeLeafPageTest, MaxKeysBigint) {
    // (8192 - 22) / (8 + 6) = 8170 / 14 = 583
    EXPECT_EQ(BTreeLeafPage::MaxKeys(TypeId::BIGINT), 583);
}

TEST_F(BTreeLeafPageTest, MaxKeysDecimal) {
    EXPECT_EQ(BTreeLeafPage::MaxKeys(TypeId::DECIMAL), 583);
}

TEST_F(BTreeLeafPageTest, MaxKeysDate) {
    EXPECT_EQ(BTreeLeafPage::MaxKeys(TypeId::DATE), 817);
}

TEST_F(BTreeLeafPageTest, KeySizeThrowsOnVarchar) {
    EXPECT_THROW((void)BTreeLeafPage::MaxKeys(TypeId::VARCHAR), DatabaseException);
}

TEST_F(BTreeLeafPageTest, KeySizeThrowsOnInvalid) {
    EXPECT_THROW((void)BTreeLeafPage::MaxKeys(TypeId::INVALID), DatabaseException);
}

// ---------------------------------------------------------------------------
// SetKey / GetKey / SetRID / GetRID — manual layout
// ---------------------------------------------------------------------------
TEST_F(BTreeLeafPageTest, SetAndGetKeysAndRIDsInteger) {
    BTreeLeafPage::SetNumKeys(page_.data(), 3);

    BTreeLeafPage::SetKey(page_.data(), 0, Value(10), TypeId::INTEGER);
    BTreeLeafPage::SetRID(page_.data(), 0, RID(1, 0), TypeId::INTEGER);
    BTreeLeafPage::SetKey(page_.data(), 1, Value(20), TypeId::INTEGER);
    BTreeLeafPage::SetRID(page_.data(), 1, RID(2, 1), TypeId::INTEGER);
    BTreeLeafPage::SetKey(page_.data(), 2, Value(30), TypeId::INTEGER);
    BTreeLeafPage::SetRID(page_.data(), 2, RID(3, 2), TypeId::INTEGER);

    EXPECT_EQ(BTreeLeafPage::GetKey(page_.data(), 0, TypeId::INTEGER), Value(10));
    EXPECT_EQ(BTreeLeafPage::GetKey(page_.data(), 1, TypeId::INTEGER), Value(20));
    EXPECT_EQ(BTreeLeafPage::GetKey(page_.data(), 2, TypeId::INTEGER), Value(30));

    EXPECT_EQ(BTreeLeafPage::GetRID(page_.data(), 0, TypeId::INTEGER), RID(1, 0));
    EXPECT_EQ(BTreeLeafPage::GetRID(page_.data(), 1, TypeId::INTEGER), RID(2, 1));
    EXPECT_EQ(BTreeLeafPage::GetRID(page_.data(), 2, TypeId::INTEGER), RID(3, 2));
}

TEST_F(BTreeLeafPageTest, SetAndGetKeysAndRIDsBigint) {
    BTreeLeafPage::SetNumKeys(page_.data(), 2);

    BTreeLeafPage::SetKey(page_.data(), 0, Value(int64_t{1000}), TypeId::BIGINT);
    BTreeLeafPage::SetRID(page_.data(), 0, RID(10, 5), TypeId::BIGINT);
    BTreeLeafPage::SetKey(page_.data(), 1, Value(int64_t{2000}), TypeId::BIGINT);
    BTreeLeafPage::SetRID(page_.data(), 1, RID(11, 6), TypeId::BIGINT);

    EXPECT_EQ(BTreeLeafPage::GetKey(page_.data(), 0, TypeId::BIGINT), Value(int64_t{1000}));
    EXPECT_EQ(BTreeLeafPage::GetKey(page_.data(), 1, TypeId::BIGINT), Value(int64_t{2000}));
    EXPECT_EQ(BTreeLeafPage::GetRID(page_.data(), 0, TypeId::BIGINT), RID(10, 5));
    EXPECT_EQ(BTreeLeafPage::GetRID(page_.data(), 1, TypeId::BIGINT), RID(11, 6));
}

// ---------------------------------------------------------------------------
// LowerBound
// ---------------------------------------------------------------------------
TEST_F(BTreeLeafPageTest, LowerBoundFindsPosition) {
    BTreeLeafPage::SetNumKeys(page_.data(), 3);
    BTreeLeafPage::SetKey(page_.data(), 0, Value(10), TypeId::INTEGER);
    BTreeLeafPage::SetRID(page_.data(), 0, RID(1, 0), TypeId::INTEGER);
    BTreeLeafPage::SetKey(page_.data(), 1, Value(20), TypeId::INTEGER);
    BTreeLeafPage::SetRID(page_.data(), 1, RID(2, 0), TypeId::INTEGER);
    BTreeLeafPage::SetKey(page_.data(), 2, Value(30), TypeId::INTEGER);
    BTreeLeafPage::SetRID(page_.data(), 2, RID(3, 0), TypeId::INTEGER);

    EXPECT_EQ(BTreeLeafPage::LowerBound(page_.data(), Value(5), TypeId::INTEGER), 0);
    EXPECT_EQ(BTreeLeafPage::LowerBound(page_.data(), Value(10), TypeId::INTEGER), 0);
    EXPECT_EQ(BTreeLeafPage::LowerBound(page_.data(), Value(15), TypeId::INTEGER), 1);
    EXPECT_EQ(BTreeLeafPage::LowerBound(page_.data(), Value(20), TypeId::INTEGER), 1);
    EXPECT_EQ(BTreeLeafPage::LowerBound(page_.data(), Value(25), TypeId::INTEGER), 2);
    EXPECT_EQ(BTreeLeafPage::LowerBound(page_.data(), Value(30), TypeId::INTEGER), 2);
    EXPECT_EQ(BTreeLeafPage::LowerBound(page_.data(), Value(99), TypeId::INTEGER), 3);
}

TEST_F(BTreeLeafPageTest, LowerBoundEmptyPage) {
    EXPECT_EQ(BTreeLeafPage::LowerBound(page_.data(), Value(10), TypeId::INTEGER), 0);
}

// ---------------------------------------------------------------------------
// Insert
// ---------------------------------------------------------------------------
TEST_F(BTreeLeafPageTest, InsertSingleEntry) {
    EXPECT_TRUE(BTreeLeafPage::Insert(page_.data(), Value(50), RID(1, 0), TypeId::INTEGER));
    EXPECT_EQ(BTreeLeafPage::GetNumKeys(page_.data()), 1);
    EXPECT_EQ(BTreeLeafPage::GetKey(page_.data(), 0, TypeId::INTEGER), Value(50));
    EXPECT_EQ(BTreeLeafPage::GetRID(page_.data(), 0, TypeId::INTEGER), RID(1, 0));
}

TEST_F(BTreeLeafPageTest, InsertMaintainsSortOrder) {
    EXPECT_TRUE(BTreeLeafPage::Insert(page_.data(), Value(30), RID(3, 0), TypeId::INTEGER));
    EXPECT_TRUE(BTreeLeafPage::Insert(page_.data(), Value(10), RID(1, 0), TypeId::INTEGER));
    EXPECT_TRUE(BTreeLeafPage::Insert(page_.data(), Value(20), RID(2, 0), TypeId::INTEGER));
    EXPECT_TRUE(BTreeLeafPage::Insert(page_.data(), Value(40), RID(4, 0), TypeId::INTEGER));

    EXPECT_EQ(BTreeLeafPage::GetNumKeys(page_.data()), 4);
    EXPECT_EQ(BTreeLeafPage::GetKey(page_.data(), 0, TypeId::INTEGER), Value(10));
    EXPECT_EQ(BTreeLeafPage::GetKey(page_.data(), 1, TypeId::INTEGER), Value(20));
    EXPECT_EQ(BTreeLeafPage::GetKey(page_.data(), 2, TypeId::INTEGER), Value(30));
    EXPECT_EQ(BTreeLeafPage::GetKey(page_.data(), 3, TypeId::INTEGER), Value(40));

    EXPECT_EQ(BTreeLeafPage::GetRID(page_.data(), 0, TypeId::INTEGER), RID(1, 0));
    EXPECT_EQ(BTreeLeafPage::GetRID(page_.data(), 1, TypeId::INTEGER), RID(2, 0));
    EXPECT_EQ(BTreeLeafPage::GetRID(page_.data(), 2, TypeId::INTEGER), RID(3, 0));
    EXPECT_EQ(BTreeLeafPage::GetRID(page_.data(), 3, TypeId::INTEGER), RID(4, 0));
}

TEST_F(BTreeLeafPageTest, InsertDuplicateKeys) {
    EXPECT_TRUE(BTreeLeafPage::Insert(page_.data(), Value(10), RID(1, 0), TypeId::INTEGER));
    EXPECT_TRUE(BTreeLeafPage::Insert(page_.data(), Value(10), RID(2, 0), TypeId::INTEGER));
    EXPECT_TRUE(BTreeLeafPage::Insert(page_.data(), Value(10), RID(3, 0), TypeId::INTEGER));

    EXPECT_EQ(BTreeLeafPage::GetNumKeys(page_.data()), 3);
    EXPECT_EQ(BTreeLeafPage::GetKey(page_.data(), 0, TypeId::INTEGER), Value(10));
    EXPECT_EQ(BTreeLeafPage::GetKey(page_.data(), 1, TypeId::INTEGER), Value(10));
    EXPECT_EQ(BTreeLeafPage::GetKey(page_.data(), 2, TypeId::INTEGER), Value(10));
}

TEST_F(BTreeLeafPageTest, InsertBigintKeys) {
    EXPECT_TRUE(BTreeLeafPage::Insert(page_.data(), Value(int64_t{200}), RID(2, 0), TypeId::BIGINT));
    EXPECT_TRUE(BTreeLeafPage::Insert(page_.data(), Value(int64_t{100}), RID(1, 0), TypeId::BIGINT));

    EXPECT_EQ(BTreeLeafPage::GetNumKeys(page_.data()), 2);
    EXPECT_EQ(BTreeLeafPage::GetKey(page_.data(), 0, TypeId::BIGINT), Value(int64_t{100}));
    EXPECT_EQ(BTreeLeafPage::GetKey(page_.data(), 1, TypeId::BIGINT), Value(int64_t{200}));
    EXPECT_EQ(BTreeLeafPage::GetRID(page_.data(), 0, TypeId::BIGINT), RID(1, 0));
    EXPECT_EQ(BTreeLeafPage::GetRID(page_.data(), 1, TypeId::BIGINT), RID(2, 0));
}

TEST_F(BTreeLeafPageTest, NeedsSplitAfterOverfull) {
    BTreeLeafPage::SetNumKeys(page_.data(), BTreeLeafPage::MaxKeys(TypeId::INTEGER));
    EXPECT_FALSE(BTreeLeafPage::NeedsSplit(page_.data(), TypeId::INTEGER));

    BTreeLeafPage::SetNumKeys(page_.data(), BTreeLeafPage::MaxKeys(TypeId::INTEGER) + 1);
    EXPECT_TRUE(BTreeLeafPage::NeedsSplit(page_.data(), TypeId::INTEGER));
}

// ---------------------------------------------------------------------------
// Lookup
// ---------------------------------------------------------------------------
TEST_F(BTreeLeafPageTest, LookupFindsMatchingRIDs) {
    (void)BTreeLeafPage::Insert(page_.data(), Value(10), RID(1, 0), TypeId::INTEGER);
    (void)BTreeLeafPage::Insert(page_.data(), Value(20), RID(2, 0), TypeId::INTEGER);
    (void)BTreeLeafPage::Insert(page_.data(), Value(30), RID(3, 0), TypeId::INTEGER);

    auto rids = BTreeLeafPage::Lookup(page_.data(), Value(20), TypeId::INTEGER);
    ASSERT_EQ(rids.size(), 1u);
    EXPECT_EQ(rids[0], RID(2, 0));
}

TEST_F(BTreeLeafPageTest, LookupNotFound) {
    (void)BTreeLeafPage::Insert(page_.data(), Value(10), RID(1, 0), TypeId::INTEGER);
    (void)BTreeLeafPage::Insert(page_.data(), Value(30), RID(3, 0), TypeId::INTEGER);

    auto rids = BTreeLeafPage::Lookup(page_.data(), Value(20), TypeId::INTEGER);
    EXPECT_TRUE(rids.empty());
}

TEST_F(BTreeLeafPageTest, LookupDuplicateKeys) {
    (void)BTreeLeafPage::Insert(page_.data(), Value(10), RID(1, 0), TypeId::INTEGER);
    (void)BTreeLeafPage::Insert(page_.data(), Value(10), RID(2, 0), TypeId::INTEGER);
    (void)BTreeLeafPage::Insert(page_.data(), Value(10), RID(3, 0), TypeId::INTEGER);
    (void)BTreeLeafPage::Insert(page_.data(), Value(20), RID(4, 0), TypeId::INTEGER);

    auto rids = BTreeLeafPage::Lookup(page_.data(), Value(10), TypeId::INTEGER);
    ASSERT_EQ(rids.size(), 3u);
    EXPECT_EQ(rids[0], RID(1, 0));
    EXPECT_EQ(rids[1], RID(2, 0));
    EXPECT_EQ(rids[2], RID(3, 0));
}

TEST_F(BTreeLeafPageTest, LookupEmptyPage) {
    auto rids = BTreeLeafPage::Lookup(page_.data(), Value(10), TypeId::INTEGER);
    EXPECT_TRUE(rids.empty());
}

// ---------------------------------------------------------------------------
// Delete
// ---------------------------------------------------------------------------
TEST_F(BTreeLeafPageTest, DeleteSingleEntry) {
    (void)BTreeLeafPage::Insert(page_.data(), Value(10), RID(1, 0), TypeId::INTEGER);
    (void)BTreeLeafPage::Insert(page_.data(), Value(20), RID(2, 0), TypeId::INTEGER);
    (void)BTreeLeafPage::Insert(page_.data(), Value(30), RID(3, 0), TypeId::INTEGER);

    EXPECT_TRUE(BTreeLeafPage::Delete(page_.data(), Value(20), RID(2, 0), TypeId::INTEGER));
    EXPECT_EQ(BTreeLeafPage::GetNumKeys(page_.data()), 2);
    EXPECT_EQ(BTreeLeafPage::GetKey(page_.data(), 0, TypeId::INTEGER), Value(10));
    EXPECT_EQ(BTreeLeafPage::GetKey(page_.data(), 1, TypeId::INTEGER), Value(30));
}

TEST_F(BTreeLeafPageTest, DeleteNotFound) {
    (void)BTreeLeafPage::Insert(page_.data(), Value(10), RID(1, 0), TypeId::INTEGER);
    EXPECT_FALSE(BTreeLeafPage::Delete(page_.data(), Value(99), RID(1, 0), TypeId::INTEGER));
    EXPECT_EQ(BTreeLeafPage::GetNumKeys(page_.data()), 1);
}

TEST_F(BTreeLeafPageTest, DeleteMatchesBothKeyAndRID) {
    (void)BTreeLeafPage::Insert(page_.data(), Value(10), RID(1, 0), TypeId::INTEGER);
    (void)BTreeLeafPage::Insert(page_.data(), Value(10), RID(2, 0), TypeId::INTEGER);
    (void)BTreeLeafPage::Insert(page_.data(), Value(10), RID(3, 0), TypeId::INTEGER);

    EXPECT_TRUE(BTreeLeafPage::Delete(page_.data(), Value(10), RID(2, 0), TypeId::INTEGER));
    EXPECT_EQ(BTreeLeafPage::GetNumKeys(page_.data()), 2);
    EXPECT_EQ(BTreeLeafPage::GetRID(page_.data(), 0, TypeId::INTEGER), RID(1, 0));
    EXPECT_EQ(BTreeLeafPage::GetRID(page_.data(), 1, TypeId::INTEGER), RID(3, 0));
}

TEST_F(BTreeLeafPageTest, DeleteWrongRIDReturnsFalse) {
    (void)BTreeLeafPage::Insert(page_.data(), Value(10), RID(1, 0), TypeId::INTEGER);
    EXPECT_FALSE(BTreeLeafPage::Delete(page_.data(), Value(10), RID(99, 0), TypeId::INTEGER));
    EXPECT_EQ(BTreeLeafPage::GetNumKeys(page_.data()), 1);
}

TEST_F(BTreeLeafPageTest, DeleteFirstAndLastEntries) {
    (void)BTreeLeafPage::Insert(page_.data(), Value(10), RID(1, 0), TypeId::INTEGER);
    (void)BTreeLeafPage::Insert(page_.data(), Value(20), RID(2, 0), TypeId::INTEGER);
    (void)BTreeLeafPage::Insert(page_.data(), Value(30), RID(3, 0), TypeId::INTEGER);

    EXPECT_TRUE(BTreeLeafPage::Delete(page_.data(), Value(10), RID(1, 0), TypeId::INTEGER));
    EXPECT_EQ(BTreeLeafPage::GetNumKeys(page_.data()), 2);
    EXPECT_EQ(BTreeLeafPage::GetKey(page_.data(), 0, TypeId::INTEGER), Value(20));

    EXPECT_TRUE(BTreeLeafPage::Delete(page_.data(), Value(30), RID(3, 0), TypeId::INTEGER));
    EXPECT_EQ(BTreeLeafPage::GetNumKeys(page_.data()), 1);
    EXPECT_EQ(BTreeLeafPage::GetKey(page_.data(), 0, TypeId::INTEGER), Value(20));
}

TEST_F(BTreeLeafPageTest, InsertAfterDelete) {
    (void)BTreeLeafPage::Insert(page_.data(), Value(10), RID(1, 0), TypeId::INTEGER);
    (void)BTreeLeafPage::Insert(page_.data(), Value(30), RID(3, 0), TypeId::INTEGER);
    (void)BTreeLeafPage::Delete(page_.data(), Value(10), RID(1, 0), TypeId::INTEGER);
    (void)BTreeLeafPage::Insert(page_.data(), Value(20), RID(2, 0), TypeId::INTEGER);

    EXPECT_EQ(BTreeLeafPage::GetNumKeys(page_.data()), 2);
    EXPECT_EQ(BTreeLeafPage::GetKey(page_.data(), 0, TypeId::INTEGER), Value(20));
    EXPECT_EQ(BTreeLeafPage::GetKey(page_.data(), 1, TypeId::INTEGER), Value(30));
}

}  // namespace shilmandb
