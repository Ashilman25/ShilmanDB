#include <gtest/gtest.h>
#include "storage/slotted_page.hpp"
#include <array>
#include <cstring>
#include <vector>

namespace shilmandb {

// Fresh page buffer for each test
class SlottedPageTest : public ::testing::Test {
protected:
    std::array<char, PAGE_SIZE> page_{};

    void SetUp() override {
        SlottedPage::Init(page_.data(), 42);
    }
};

// ---------------------------------------------------------------------------
// Test 1: InitSetsHeader
// ---------------------------------------------------------------------------
TEST_F(SlottedPageTest, InitSetsHeader) {
    auto* hdr = SlottedPage::GetHeader(page_.data());
    EXPECT_EQ(hdr->page_id, 42u);
    EXPECT_EQ(hdr->num_slots, 0);
    EXPECT_EQ(hdr->free_space_offset, static_cast<uint16_t>(PAGE_HEADER_SIZE));
    EXPECT_EQ(hdr->next_page_id, INVALID_PAGE_ID);
    EXPECT_EQ(hdr->flags, 0u);
}

// ---------------------------------------------------------------------------
// Test 2: InsertAndRetrieve
// ---------------------------------------------------------------------------
TEST_F(SlottedPageTest, InsertAndRetrieve) {
    std::array<char, 50> tuple{};
    std::memset(tuple.data(), 0xAA, tuple.size());

    int slot = SlottedPage::InsertTuple(page_.data(), tuple.data(), 50);
    ASSERT_EQ(slot, 0);

    std::array<char, PAGE_SIZE> out{};
    uint16_t out_len = 0;
    ASSERT_TRUE(SlottedPage::GetTuple(page_.data(), 0, out.data(), &out_len));
    EXPECT_EQ(out_len, 50);
    EXPECT_EQ(std::memcmp(out.data(), tuple.data(), 50), 0);
}

// ---------------------------------------------------------------------------
// Test 3: InsertMultipleVariableLength
// ---------------------------------------------------------------------------
TEST_F(SlottedPageTest, InsertMultipleVariableLength) {
    constexpr int kCount = 100;
    std::vector<std::vector<char>> tuples(kCount);

    for (int i = 0; i < kCount; ++i) {
        uint16_t len = static_cast<uint16_t>(20 + (i % 10) * 20);  // 20..200
        tuples[i].assign(len, static_cast<char>(i + 1));

        int slot = SlottedPage::InsertTuple(page_.data(), tuples[i].data(), len);
        if (slot == -1) {
            // Page full — stop inserting, verify what we have
            tuples.resize(i);
            break;
        }
        EXPECT_EQ(slot, i);
    }

    // Retrieve and verify each inserted tuple
    for (int i = 0; i < static_cast<int>(tuples.size()); ++i) {
        std::array<char, PAGE_SIZE> out{};
        uint16_t out_len = 0;
        ASSERT_TRUE(SlottedPage::GetTuple(page_.data(), static_cast<uint16_t>(i),
                                          out.data(), &out_len))
            << "Failed to get tuple " << i;
        EXPECT_EQ(out_len, tuples[i].size());
        EXPECT_EQ(std::memcmp(out.data(), tuples[i].data(), out_len), 0)
            << "Data mismatch on tuple " << i;
    }
}

// ---------------------------------------------------------------------------
// Test 4: DeleteAndSkip
// ---------------------------------------------------------------------------
TEST_F(SlottedPageTest, DeleteAndSkip) {
    constexpr int kCount = 10;
    std::array<std::array<char, 40>, kCount> tuples{};
    for (int i = 0; i < kCount; ++i) {
        std::memset(tuples[i].data(), i + 1, 40);
        ASSERT_NE(SlottedPage::InsertTuple(page_.data(), tuples[i].data(), 40), -1);
    }

    // Delete slots 2, 5, 7
    EXPECT_TRUE(SlottedPage::DeleteTuple(page_.data(), 2));
    EXPECT_TRUE(SlottedPage::DeleteTuple(page_.data(), 5));
    EXPECT_TRUE(SlottedPage::DeleteTuple(page_.data(), 7));

    // Verify: deleted slots return false, others return correct data
    for (int i = 0; i < kCount; ++i) {
        std::array<char, PAGE_SIZE> out{};
        uint16_t out_len = 0;
        bool found = SlottedPage::GetTuple(page_.data(), static_cast<uint16_t>(i),
                                           out.data(), &out_len);
        if (i == 2 || i == 5 || i == 7) {
            EXPECT_FALSE(found) << "Deleted slot " << i << " should not be found";
        } else {
            ASSERT_TRUE(found) << "Live slot " << i << " should be found";
            EXPECT_EQ(out_len, 40);
            EXPECT_EQ(std::memcmp(out.data(), tuples[i].data(), 40), 0)
                << "Data mismatch on slot " << i;
        }
    }

    // Double-delete returns false
    EXPECT_FALSE(SlottedPage::DeleteTuple(page_.data(), 2));
}

// ---------------------------------------------------------------------------
// Test 5: CompactReclaimsSpace
// ---------------------------------------------------------------------------
TEST_F(SlottedPageTest, CompactReclaimsSpace) {
    constexpr int kCount = 10;
    for (int i = 0; i < kCount; ++i) {
        std::array<char, 60> t{};
        std::memset(t.data(), i + 1, 60);
        ASSERT_NE(SlottedPage::InsertTuple(page_.data(), t.data(), 60), -1);
    }

    uint16_t free_before = SlottedPage::GetFreeSpace(page_.data());

    // Delete even slots (0, 2, 4, 6, 8)
    for (int i = 0; i < kCount; i += 2) {
        ASSERT_TRUE(SlottedPage::DeleteTuple(page_.data(), static_cast<uint16_t>(i)));
    }

    SlottedPage::Compact(page_.data());

    uint16_t free_after = SlottedPage::GetFreeSpace(page_.data());
    EXPECT_GT(free_after, free_before);

    // Remaining odd slots still retrievable with correct data
    for (int i = 1; i < kCount; i += 2) {
        std::array<char, PAGE_SIZE> out{};
        uint16_t out_len = 0;
        ASSERT_TRUE(SlottedPage::GetTuple(page_.data(), static_cast<uint16_t>(i),
                                          out.data(), &out_len))
            << "Live slot " << i << " missing after compact";
        EXPECT_EQ(out_len, 60);
        std::array<char, 60> expected{};
        std::memset(expected.data(), i + 1, 60);
        EXPECT_EQ(std::memcmp(out.data(), expected.data(), 60), 0);
    }
}

// ---------------------------------------------------------------------------
// Test 6: OverflowRejects
// ---------------------------------------------------------------------------
TEST_F(SlottedPageTest, OverflowRejects) {
    std::array<char, PAGE_SIZE> big{};
    int slot = SlottedPage::InsertTuple(page_.data(), big.data(),
                                        static_cast<uint16_t>(PAGE_SIZE));
    EXPECT_EQ(slot, -1);
}

// ---------------------------------------------------------------------------
// Test 7: FillPageCompletely
// ---------------------------------------------------------------------------
TEST_F(SlottedPageTest, FillPageCompletely) {
    constexpr uint16_t kTupleLen = 64;
    int count = 0;
    int slot  = 0;

    while (true) {
        std::array<char, kTupleLen> t{};
        std::memset(t.data(), count + 1, kTupleLen);
        slot = SlottedPage::InsertTuple(page_.data(), t.data(), kTupleLen);
        if (slot == -1) { break; }
        ++count;
    }

    EXPECT_GT(count, 0);
    EXPECT_EQ(slot, -1);  // last insert failed

    // Verify last successful tuple
    std::array<char, PAGE_SIZE> out{};
    uint16_t out_len = 0;
    ASSERT_TRUE(SlottedPage::GetTuple(page_.data(), static_cast<uint16_t>(count - 1),
                                      out.data(), &out_len));
    EXPECT_EQ(out_len, kTupleLen);
}

// ---------------------------------------------------------------------------
// Test 8: InsertAfterCompact
// ---------------------------------------------------------------------------
TEST_F(SlottedPageTest, InsertAfterCompact) {
    constexpr uint16_t kTupleLen = 80;

    // Fill the page
    std::vector<int> inserted;
    while (true) {
        std::array<char, kTupleLen> t{};
        std::memset(t.data(), static_cast<int>(inserted.size()) + 1, kTupleLen);
        int slot = SlottedPage::InsertTuple(page_.data(), t.data(), kTupleLen);
        if (slot == -1) { break; }
        inserted.push_back(slot);
    }

    int total_before = static_cast<int>(inserted.size());
    ASSERT_GT(total_before, 4);

    // Delete first half
    int half = total_before / 2;
    for (int i = 0; i < half; ++i) {
        ASSERT_TRUE(SlottedPage::DeleteTuple(page_.data(), static_cast<uint16_t>(i)));
    }

    SlottedPage::Compact(page_.data());

    // Insert new tuples into reclaimed space
    std::vector<int> new_slots;
    while (true) {
        auto fill = static_cast<char>(200 + static_cast<int>(new_slots.size()));
        std::array<char, kTupleLen> t{};
        std::memset(t.data(), fill, kTupleLen);
        int slot = SlottedPage::InsertTuple(page_.data(), t.data(), kTupleLen);
        if (slot == -1) { break; }
        new_slots.push_back(slot);
    }

    EXPECT_GT(static_cast<int>(new_slots.size()), 0);

    // Verify surviving original tuples (second half)
    for (int i = half; i < total_before; ++i) {
        std::array<char, PAGE_SIZE> out{};
        uint16_t out_len = 0;
        ASSERT_TRUE(SlottedPage::GetTuple(page_.data(), static_cast<uint16_t>(i),
                                          out.data(), &out_len))
            << "Original slot " << i << " missing after compact+reinsert";
        EXPECT_EQ(out_len, kTupleLen);
        std::array<char, kTupleLen> expected{};
        std::memset(expected.data(), i + 1, kTupleLen);
        EXPECT_EQ(std::memcmp(out.data(), expected.data(), kTupleLen), 0);
    }

    // Verify newly inserted tuples
    for (int i = 0; i < static_cast<int>(new_slots.size()); ++i) {
        std::array<char, PAGE_SIZE> out{};
        uint16_t out_len = 0;
        auto fill = static_cast<char>(200 + i);
        ASSERT_TRUE(SlottedPage::GetTuple(page_.data(), static_cast<uint16_t>(new_slots[i]),
                                          out.data(), &out_len));
        EXPECT_EQ(out_len, kTupleLen);
        std::array<char, kTupleLen> expected{};
        std::memset(expected.data(), fill, kTupleLen);
        EXPECT_EQ(std::memcmp(out.data(), expected.data(), kTupleLen), 0);
    }
}

// ---------------------------------------------------------------------------
// Test 9: ZeroLengthTupleRejected
// ---------------------------------------------------------------------------
TEST_F(SlottedPageTest, ZeroLengthTupleRejected) {
    char dummy = 'x';
    int slot = SlottedPage::InsertTuple(page_.data(), &dummy, 0);
    EXPECT_EQ(slot, -1);

    // Page should be unchanged
    auto* hdr = SlottedPage::GetHeader(page_.data());
    EXPECT_EQ(hdr->num_slots, 0);
}

// ---------------------------------------------------------------------------
// Test 10: GetFreeSpaceLifecycle
// ---------------------------------------------------------------------------
TEST_F(SlottedPageTest, GetFreeSpaceLifecycle) {
    // Fresh page: full capacity minus one slot entry overhead
    uint16_t initial = SlottedPage::GetFreeSpace(page_.data());
    EXPECT_EQ(initial,
              static_cast<uint16_t>(PAGE_SIZE - PAGE_HEADER_SIZE - SLOT_ENTRY_SIZE));

    // After inserting 100 bytes: loses 100 (tuple) + 4 (slot entry for next insert)
    std::array<char, 100> t{};
    ASSERT_NE(SlottedPage::InsertTuple(page_.data(), t.data(), 100), -1);
    uint16_t after_insert = SlottedPage::GetFreeSpace(page_.data());
    EXPECT_EQ(after_insert, initial - 100 - static_cast<uint16_t>(SLOT_ENTRY_SIZE));

    // After delete: free space increases (FindTupleDataStart skips deleted slots)
    ASSERT_TRUE(SlottedPage::DeleteTuple(page_.data(), 0));
    uint16_t after_delete = SlottedPage::GetFreeSpace(page_.data());
    EXPECT_GT(after_delete, after_insert);

    // After compact: free space >= after_delete (gap zeroed, tuples repacked)
    SlottedPage::Compact(page_.data());
    uint16_t after_compact = SlottedPage::GetFreeSpace(page_.data());
    EXPECT_GE(after_compact, after_delete);
}

// ---------------------------------------------------------------------------
// Test 11: GetTupleOutOfRangeReturnsFalse
// ---------------------------------------------------------------------------
TEST_F(SlottedPageTest, GetTupleOutOfRangeReturnsFalse) {
    std::array<char, 40> tuple{};
    ASSERT_NE(SlottedPage::InsertTuple(page_.data(), tuple.data(), 40), -1);

    std::array<char, PAGE_SIZE> out{};
    uint16_t out_len = 0;
    EXPECT_FALSE(SlottedPage::GetTuple(page_.data(), 999, out.data(), &out_len));
}

// ---------------------------------------------------------------------------
// Test 12: DeleteTupleOutOfRangeReturnsFalse
// ---------------------------------------------------------------------------
TEST_F(SlottedPageTest, DeleteTupleOutOfRangeReturnsFalse) {
    std::array<char, 40> tuple{};
    ASSERT_NE(SlottedPage::InsertTuple(page_.data(), tuple.data(), 40), -1);

    EXPECT_FALSE(SlottedPage::DeleteTuple(page_.data(), 999));
}

// ---------------------------------------------------------------------------
// Test 13: FillPageExactCount
// ---------------------------------------------------------------------------
TEST_F(SlottedPageTest, FillPageExactCount) {
    constexpr uint16_t kTupleLen = 64;
    // Expected: (PAGE_SIZE - PAGE_HEADER_SIZE) / (kTupleLen + SLOT_ENTRY_SIZE)
    //         = (8192 - 16) / (64 + 4) = 8176 / 68 = 120
    constexpr int kExpected = (PAGE_SIZE - PAGE_HEADER_SIZE) / (kTupleLen + SLOT_ENTRY_SIZE);

    int count = 0;
    while (true) {
        std::array<char, kTupleLen> t{};
        std::memset(t.data(), count + 1, kTupleLen);
        if (SlottedPage::InsertTuple(page_.data(), t.data(), kTupleLen) == -1) { break; }
        ++count;
    }

    EXPECT_EQ(count, kExpected);
}

}  // namespace shilmandb
