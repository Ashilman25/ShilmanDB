#include <gtest/gtest.h>
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include <cstring>
#include <filesystem>
#include <memory>

namespace shilmandb {

class BufferPoolManagerTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() / "shilmandb_bpm_test.db").string();
        std::filesystem::remove(test_file_);
    }

    void TearDown() override {
        std::filesystem::remove(test_file_);
    }

    // Helper: create a BPM with a given pool size
    struct BPMBundle {
        std::unique_ptr<DiskManager> disk_manager;
        std::unique_ptr<BufferPoolManager> bpm;
    };

    BPMBundle MakeBPM(size_t pool_size) {
        auto dm = std::make_unique<DiskManager>(test_file_);
        auto eviction = std::make_unique<LRUEvictionPolicy>(pool_size);
        auto bpm = std::make_unique<BufferPoolManager>(
            pool_size, dm.get(), std::move(eviction));
        return {std::move(dm), std::move(bpm)};
    }
};

// ---------------------------------------------------------------------------
// Test 1: FetchAndUnpin
// ---------------------------------------------------------------------------
TEST_F(BufferPoolManagerTest, FetchAndUnpin) {
    auto [dm, bpm] = MakeBPM(10);

    // Create a new page and write a pattern
    page_id_t pid;
    auto* page = bpm->NewPage(&pid);
    ASSERT_NE(page, nullptr);
    std::memset(page->GetData(), 0xAB, PAGE_SIZE);
    (void)bpm->UnpinPage(pid, true);

    // Fetch same page — should be a cache hit
    auto* fetched = bpm->FetchPage(pid);
    ASSERT_NE(fetched, nullptr);

    // Verify data intact
    for (size_t i = 0; i < PAGE_SIZE; ++i) {
        EXPECT_EQ(static_cast<unsigned char>(fetched->GetData()[i]), 0xAB)
            << "Mismatch at byte " << i;
    }
    EXPECT_EQ(bpm->GetHitCount(), 1u);
    (void)bpm->UnpinPage(pid, false);
}

// ---------------------------------------------------------------------------
// Test 2: EvictionWritesDirtyPages
// ---------------------------------------------------------------------------
TEST_F(BufferPoolManagerTest, EvictionWritesDirtyPages) {
    auto [dm, bpm] = MakeBPM(3);

    // Fill pool with 3 pages, each with distinct data
    page_id_t pids[3];
    for (int i = 0; i < 3; ++i) {
        auto* page = bpm->NewPage(&pids[i]);
        ASSERT_NE(page, nullptr);
        std::memset(page->GetData(), 0x10 + i, PAGE_SIZE);
        (void)bpm->UnpinPage(pids[i], true);
    }

    // Fetch a 4th page — forces eviction of LRU (pids[0])
    page_id_t pid3;
    auto* page3 = bpm->NewPage(&pid3);
    ASSERT_NE(page3, nullptr);
    (void)bpm->UnpinPage(pid3, false);

    // Fetch the evicted page back — data should have survived via disk
    auto* refetched = bpm->FetchPage(pids[0]);
    ASSERT_NE(refetched, nullptr);
    for (size_t i = 0; i < PAGE_SIZE; ++i) {
        EXPECT_EQ(static_cast<unsigned char>(refetched->GetData()[i]), 0x10)
            << "Mismatch at byte " << i;
    }
    (void)bpm->UnpinPage(pids[0], false);
}

// ---------------------------------------------------------------------------
// Test 3: PinnedPagesNotEvicted
// ---------------------------------------------------------------------------
TEST_F(BufferPoolManagerTest, PinnedPagesNotEvicted) {
    auto [dm, bpm] = MakeBPM(2);

    // Create 2 pages, keep both pinned
    page_id_t pid0, pid1;
    ASSERT_NE(bpm->NewPage(&pid0), nullptr);
    ASSERT_NE(bpm->NewPage(&pid1), nullptr);

    // Pool full, all pinned — cannot create another
    page_id_t pid2;
    EXPECT_EQ(bpm->NewPage(&pid2), nullptr);
}

// ---------------------------------------------------------------------------
// Test 4: LRUEvictionOrder
// ---------------------------------------------------------------------------
TEST_F(BufferPoolManagerTest, LRUEvictionOrder) {
    auto [dm, bpm] = MakeBPM(3);

    // Create pages A=0, B=1, C=2. Unpin each.
    page_id_t pids[3];
    for (int i = 0; i < 3; ++i) {
        auto* p = bpm->NewPage(&pids[i]);
        ASSERT_NE(p, nullptr);
        std::memset(p->GetData(), 'A' + i, PAGE_SIZE);
        (void)bpm->UnpinPage(pids[i], true);
    }

    // Fetch D — evicts A (LRU). Pool: B, C, D.
    page_id_t pid_d;
    auto* page_d = bpm->NewPage(&pid_d);
    ASSERT_NE(page_d, nullptr);
    (void)bpm->UnpinPage(pid_d, false);

    bpm->ResetStats();

    // Fetch A — must be a miss (was evicted). Evicts B (now LRU). Pool: C, D, A.
    auto* a = bpm->FetchPage(pids[0]);
    ASSERT_NE(a, nullptr);
    EXPECT_EQ(bpm->GetMissCount(), 1u);
    (void)bpm->UnpinPage(pids[0], false);

    // Fetch C — should be a hit (still in pool)
    auto* c = bpm->FetchPage(pids[2]);
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(bpm->GetHitCount(), 1u);
    (void)bpm->UnpinPage(pids[2], false);
}

// ---------------------------------------------------------------------------
// Test 5: HitMissCounters
// ---------------------------------------------------------------------------
TEST_F(BufferPoolManagerTest, HitMissCounters) {
    auto [dm, bpm] = MakeBPM(10);

    EXPECT_EQ(bpm->GetHitCount(), 0u);
    EXPECT_EQ(bpm->GetMissCount(), 0u);

    // NewPage doesn't count as hit or miss
    page_id_t pid;
    auto* page = bpm->NewPage(&pid);
    ASSERT_NE(page, nullptr);
    (void)bpm->UnpinPage(pid, false);

    // FetchPage on page already in pool — hit
    auto* f1 = bpm->FetchPage(pid);
    ASSERT_NE(f1, nullptr);
    EXPECT_EQ(bpm->GetHitCount(), 1u);
    EXPECT_EQ(bpm->GetMissCount(), 0u);
    (void)bpm->UnpinPage(pid, false);

    bpm->ResetStats();
    EXPECT_EQ(bpm->GetHitCount(), 0u);
    EXPECT_EQ(bpm->GetMissCount(), 0u);
}

// ---------------------------------------------------------------------------
// Test 6: DeletePage
// ---------------------------------------------------------------------------
TEST_F(BufferPoolManagerTest, DeletePage) {
    auto [dm, bpm] = MakeBPM(10);

    page_id_t pid;
    auto* page = bpm->NewPage(&pid);
    ASSERT_NE(page, nullptr);
    std::memset(page->GetData(), 0xCC, PAGE_SIZE);
    (void)bpm->UnpinPage(pid, true);

    // Delete the page
    EXPECT_TRUE(bpm->DeletePage(pid));

    // Fetch same page_id — reads from disk (never flushed, so all zeros)
    auto* refetched = bpm->FetchPage(pid);
    ASSERT_NE(refetched, nullptr);
    for (size_t i = 0; i < PAGE_SIZE; ++i) {
        EXPECT_EQ(refetched->GetData()[i], 0) << "Expected zero at byte " << i;
    }
    (void)bpm->UnpinPage(pid, false);
}

// ---------------------------------------------------------------------------
// Test 7: FlushPage
// ---------------------------------------------------------------------------
TEST_F(BufferPoolManagerTest, FlushPage) {
    {
        auto [dm, bpm] = MakeBPM(10);

        page_id_t pid;
        auto* page = bpm->NewPage(&pid);
        ASSERT_NE(page, nullptr);
        EXPECT_EQ(pid, 0u);
        std::memset(page->GetData(), 0xDD, PAGE_SIZE);
        (void)bpm->UnpinPage(pid, true);  // mark dirty

        // Re-fetch to pin, verify dirty
        page = bpm->FetchPage(pid);
        ASSERT_NE(page, nullptr);
        EXPECT_TRUE(page->IsDirty());

        // Flush — should write to disk and clear dirty flag
        EXPECT_TRUE(bpm->FlushPage(pid));
        EXPECT_FALSE(page->IsDirty());
        (void)bpm->UnpinPage(pid, false);
    }
    // BPM and DiskManager destroyed — file persists

    // Re-open and verify data survived
    {
        auto [dm2, bpm2] = MakeBPM(10);
        auto* page = bpm2->FetchPage(0);
        ASSERT_NE(page, nullptr);
        for (size_t i = 0; i < PAGE_SIZE; ++i) {
            EXPECT_EQ(static_cast<unsigned char>(page->GetData()[i]), 0xDD)
                << "Mismatch at byte " << i;
        }
        (void)bpm2->UnpinPage(0, false);
    }
}

// ---------------------------------------------------------------------------
// Test 8: ConcurrentPinUnpin
// ---------------------------------------------------------------------------
TEST_F(BufferPoolManagerTest, ConcurrentPinUnpin) {
    auto [dm, bpm] = MakeBPM(10);

    page_id_t pid;
    auto* page = bpm->NewPage(&pid);
    ASSERT_NE(page, nullptr);
    EXPECT_EQ(page->GetPinCount(), 1);

    // Fetch again — pin_count should be 2
    auto* page2 = bpm->FetchPage(pid);
    ASSERT_NE(page2, nullptr);
    EXPECT_EQ(page->GetPinCount(), 2);

    // Unpin twice
    EXPECT_TRUE(bpm->UnpinPage(pid, false));
    EXPECT_EQ(page->GetPinCount(), 1);

    EXPECT_TRUE(bpm->UnpinPage(pid, false));
    EXPECT_EQ(page->GetPinCount(), 0);

    // Third unpin should fail (already 0)
    EXPECT_FALSE(bpm->UnpinPage(pid, false));
}

// ---------------------------------------------------------------------------
// Test 9: NewPageReturnsSequentialIds
// ---------------------------------------------------------------------------
TEST_F(BufferPoolManagerTest, NewPageReturnsSequentialIds) {
    auto [dm, bpm] = MakeBPM(10);

    for (page_id_t expected = 0; expected < 5; ++expected) {
        page_id_t pid;
        auto* page = bpm->NewPage(&pid);
        ASSERT_NE(page, nullptr);
        EXPECT_EQ(pid, expected);
        (void)bpm->UnpinPage(pid, false);
    }
}

// ---------------------------------------------------------------------------
// Test 10: FetchPageNullptrWhenExhausted
// ---------------------------------------------------------------------------
TEST_F(BufferPoolManagerTest, FetchPageNullptrWhenExhausted) {
    auto [dm, bpm] = MakeBPM(1);

    page_id_t pid;
    auto* page = bpm->NewPage(&pid);
    ASSERT_NE(page, nullptr);
    // Don't unpin — page is pinned

    // Pool exhausted, try to fetch a different page
    EXPECT_EQ(bpm->FetchPage(999), nullptr);
}

// ---------------------------------------------------------------------------
// Test 11: UnpinPageNotInPool
// ---------------------------------------------------------------------------
TEST_F(BufferPoolManagerTest, UnpinPageNotInPool) {
    auto [dm, bpm] = MakeBPM(10);
    EXPECT_FALSE(bpm->UnpinPage(42, false));
}

// ---------------------------------------------------------------------------
// Test 12: FlushAllPages
// ---------------------------------------------------------------------------
TEST_F(BufferPoolManagerTest, FlushAllPages) {
    page_id_t pids[5];

    {
        auto [dm, bpm] = MakeBPM(10);

        for (int i = 0; i < 5; ++i) {
            auto* page = bpm->NewPage(&pids[i]);
            ASSERT_NE(page, nullptr);
            std::memset(page->GetData(), 0x30 + i, PAGE_SIZE);
            (void)bpm->UnpinPage(pids[i], true);
        }

        bpm->FlushAllPages();
    }
    // BPM destroyed

    // Re-open and verify all 5 pages
    {
        auto [dm2, bpm2] = MakeBPM(10);
        for (int i = 0; i < 5; ++i) {
            auto* page = bpm2->FetchPage(pids[i]);
            ASSERT_NE(page, nullptr);
            for (size_t j = 0; j < PAGE_SIZE; ++j) {
                EXPECT_EQ(static_cast<unsigned char>(page->GetData()[j]),
                           static_cast<unsigned char>(0x30 + i))
                    << "Page " << pids[i] << " mismatch at byte " << j;
            }
            (void)bpm2->UnpinPage(pids[i], false);
        }
    }
}

// ---------------------------------------------------------------------------
// Test 13: UnpinDirtyFlagIsNotClearedByFalse
// ---------------------------------------------------------------------------
TEST_F(BufferPoolManagerTest, UnpinDirtyFlagIsNotClearedByFalse) {
    auto [dm, bpm] = MakeBPM(10);

    page_id_t pid;
    auto* page = bpm->NewPage(&pid);
    ASSERT_NE(page, nullptr);
    std::memset(page->GetData(), 0xEE, PAGE_SIZE);

    // Mark dirty via unpin
    (void)bpm->UnpinPage(pid, true);

    // Re-fetch and unpin with is_dirty=false — must NOT clear the dirty flag
    page = bpm->FetchPage(pid);
    ASSERT_NE(page, nullptr);
    EXPECT_TRUE(page->IsDirty());

    (void)bpm->UnpinPage(pid, false);
    // Page should still be dirty — false does not clear an already-dirty page
    EXPECT_TRUE(page->IsDirty());
}

// ---------------------------------------------------------------------------
// Test 14: DeletePinnedPageRefused
// ---------------------------------------------------------------------------
TEST_F(BufferPoolManagerTest, DeletePinnedPageRefused) {
    auto [dm, bpm] = MakeBPM(10);

    page_id_t pid;
    auto* page = bpm->NewPage(&pid);
    ASSERT_NE(page, nullptr);

    // Page is pinned (pin_count=1) — delete must fail
    EXPECT_FALSE(bpm->DeletePage(pid));

    // Unpin, then delete should succeed
    (void)bpm->UnpinPage(pid, false);
    EXPECT_TRUE(bpm->DeletePage(pid));
}

// ---------------------------------------------------------------------------
// Test 15: DeletePageNotInPoolReturnsFalse
// ---------------------------------------------------------------------------
TEST_F(BufferPoolManagerTest, DeletePageNotInPoolReturnsFalse) {
    auto [dm, bpm] = MakeBPM(10);
    EXPECT_FALSE(bpm->DeletePage(999));
}

// ---------------------------------------------------------------------------
// Test 16: FlushPageNotInPoolReturnsFalse
// ---------------------------------------------------------------------------
TEST_F(BufferPoolManagerTest, FlushPageNotInPoolReturnsFalse) {
    auto [dm, bpm] = MakeBPM(10);
    EXPECT_FALSE(bpm->FlushPage(999));
}

// ---------------------------------------------------------------------------
// Test 17: NewPageViaEvictionPath
// ---------------------------------------------------------------------------
TEST_F(BufferPoolManagerTest, NewPageViaEvictionPath) {
    auto [dm, bpm] = MakeBPM(2);

    // Fill pool — exhausts free list
    page_id_t pid0, pid1;
    auto* p0 = bpm->NewPage(&pid0);
    ASSERT_NE(p0, nullptr);
    std::memset(p0->GetData(), 0xAA, PAGE_SIZE);
    (void)bpm->UnpinPage(pid0, true);

    auto* p1 = bpm->NewPage(&pid1);
    ASSERT_NE(p1, nullptr);
    (void)bpm->UnpinPage(pid1, false);

    // Free list empty — NewPage must go through eviction
    page_id_t pid2;
    auto* p2 = bpm->NewPage(&pid2);
    ASSERT_NE(p2, nullptr);
    EXPECT_EQ(p2->GetPinCount(), 1);

    // Evicted page's dirty data should survive on disk
    (void)bpm->UnpinPage(pid2, false);
    auto* refetched = bpm->FetchPage(pid0);
    ASSERT_NE(refetched, nullptr);
    for (size_t i = 0; i < PAGE_SIZE; ++i) {
        EXPECT_EQ(static_cast<unsigned char>(refetched->GetData()[i]), 0xAA)
            << "Mismatch at byte " << i;
    }
    (void)bpm->UnpinPage(pid0, false);
}

}  // namespace shilmandb
