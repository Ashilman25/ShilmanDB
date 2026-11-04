#include <gtest/gtest.h>
#include <string>
#include <unistd.h>
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/eviction_policy_factory.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "common/exception.hpp"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <utility>
#include <vector>

namespace shilmandb {

class BufferPoolManagerTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() /
                      ("shilmandb_bpm_test_" + std::to_string(::getpid()) + ".db")).string();
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

    // A fresh file and a BPM with the given policy from the factory
    BPMBundle MakeBPM(size_t pool_size, EvictionPolicyKind kind) {
        std::filesystem::remove(test_file_);
        auto dm = std::make_unique<DiskManager>(test_file_);
        auto bpm = std::make_unique<BufferPoolManager>(pool_size, dm.get(), MakeEvictionPolicy(kind, pool_size));
        return {std::move(dm), std::move(bpm)};
    }

    // The policies the engine can run without a model
    static constexpr EvictionPolicyKind kEngineKinds[] = {
        EvictionPolicyKind::LRU, EvictionPolicyKind::CLOCK, EvictionPolicyKind::LRU2,
        EvictionPolicyKind::ARC, EvictionPolicyKind::LIRS,
    };
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

// ---------------------------------------------------------------------------
// Test 18: TracingWritesCorrectEntries
// ---------------------------------------------------------------------------
TEST_F(BufferPoolManagerTest, TracingWritesCorrectEntries) {
    auto trace_path = (std::filesystem::temp_directory_path() /
                       ("shilmandb_trace_test_" + std::to_string(::getpid()) + ".csv")).string();
    auto [dm, bpm] = MakeBPM(10);

    bpm->EnableTracing(trace_path);

    // NewPage should NOT produce trace entries
    page_id_t pid0, pid1, pid2;
    auto* p0 = bpm->NewPage(&pid0);
    ASSERT_NE(p0, nullptr);
    (void)bpm->UnpinPage(pid0, false);

    auto* p1 = bpm->NewPage(&pid1);
    ASSERT_NE(p1, nullptr);
    (void)bpm->UnpinPage(pid1, false);

    auto* p2 = bpm->NewPage(&pid2);
    ASSERT_NE(p2, nullptr);
    (void)bpm->UnpinPage(pid2, false);

    // FetchPage SHOULD produce trace entries
    auto* f0 = bpm->FetchPage(pid0);
    ASSERT_NE(f0, nullptr);
    (void)bpm->UnpinPage(pid0, false);

    auto* f1 = bpm->FetchPage(pid1);
    ASSERT_NE(f1, nullptr);
    (void)bpm->UnpinPage(pid1, false);

    auto* f2 = bpm->FetchPage(pid2);
    ASSERT_NE(f2, nullptr);
    (void)bpm->UnpinPage(pid2, false);

    // Fetch pid0 again — second access
    auto* f0b = bpm->FetchPage(pid0);
    ASSERT_NE(f0b, nullptr);
    (void)bpm->UnpinPage(pid0, false);

    bpm->DisableTracing();

    // Read and verify trace file
    std::ifstream trace(trace_path);
    ASSERT_TRUE(trace.is_open());

    std::string header;
    std::getline(trace, header);
    EXPECT_EQ(header, "timestamp,page_id");

    // Should have 4 entries (4 FetchPage calls, 0 for NewPage)
    std::vector<std::pair<uint64_t, page_id_t>> entries;
    std::string line;
    while (std::getline(trace, line)) {
        auto comma = line.find(',');
        ASSERT_NE(comma, std::string::npos);
        auto ts = std::stoull(line.substr(0, comma));
        auto pid = static_cast<page_id_t>(std::stoul(line.substr(comma + 1)));
        entries.emplace_back(ts, pid);
    }

    ASSERT_EQ(entries.size(), 4u);

    // Timestamps monotonically increasing
    for (size_t i = 1; i < entries.size(); ++i) {
        EXPECT_GT(entries[i].first, entries[i - 1].first);
    }

    // Page IDs match fetch order: pid0, pid1, pid2, pid0
    EXPECT_EQ(entries[0].second, pid0);
    EXPECT_EQ(entries[1].second, pid1);
    EXPECT_EQ(entries[2].second, pid2);
    EXPECT_EQ(entries[3].second, pid0);

    std::filesystem::remove(trace_path);
}

// ---------------------------------------------------------------------------
// Collapsed traces and the trace counter
// ---------------------------------------------------------------------------
namespace {

std::string TraceTestPath() {
    return (std::filesystem::temp_directory_path() /
            ("shilmandb_trace_collapse_test_" + std::to_string(::getpid()) + ".csv")).string();
}

std::vector<std::string> ReadLines(const std::string& path) {
    std::ifstream in(path);
    std::vector<std::string> lines;
    for (std::string line; std::getline(in, line);) {
        lines.push_back(line);
    }
    return lines;
}

}  // namespace

TEST_F(BufferPoolManagerTest, TracingCollapseDropsRepeats) {
    const auto trace_path = TraceTestPath();
    auto [dm, bpm] = MakeBPM(4);

    page_id_t a, b;
    ASSERT_NE(bpm->NewPage(&a), nullptr);
    (void)bpm->UnpinPage(a, false);
    ASSERT_NE(bpm->NewPage(&b), nullptr);
    (void)bpm->UnpinPage(b, false);

    bpm->EnableTracing(trace_path, true);
    for (const page_id_t pid : {a, a, a, b, b, a}) {
        ASSERT_NE(bpm->FetchPage(pid), nullptr);
        (void)bpm->UnpinPage(pid, false);
    }
    bpm->DisableTracing();

    const std::vector<std::string> expected = {
        "timestamp,page_id",
        "0," + std::to_string(a),
        "3," + std::to_string(b),
        "5," + std::to_string(a),
    };
    EXPECT_EQ(ReadLines(trace_path), expected);
    EXPECT_EQ(bpm->GetTraceCount(), 6u);

    std::filesystem::remove(trace_path);
}

TEST_F(BufferPoolManagerTest, TracingWithoutCollapseKeepsEveryFetch) {
    const auto trace_path = TraceTestPath();
    auto [dm, bpm] = MakeBPM(4);

    page_id_t a, b;
    ASSERT_NE(bpm->NewPage(&a), nullptr);
    (void)bpm->UnpinPage(a, false);
    ASSERT_NE(bpm->NewPage(&b), nullptr);
    (void)bpm->UnpinPage(b, false);

    bpm->EnableTracing(trace_path);
    for (const page_id_t pid : {a, a, a, b, b, a}) {
        ASSERT_NE(bpm->FetchPage(pid), nullptr);
        (void)bpm->UnpinPage(pid, false);
    }
    bpm->DisableTracing();

    const std::vector<std::string> expected = {
        "timestamp,page_id",
        "0," + std::to_string(a),
        "1," + std::to_string(a),
        "2," + std::to_string(a),
        "3," + std::to_string(b),
        "4," + std::to_string(b),
        "5," + std::to_string(a),
    };
    EXPECT_EQ(ReadLines(trace_path), expected);
    EXPECT_EQ(bpm->GetTraceCount(), 6u);

    std::filesystem::remove(trace_path);
}

TEST_F(BufferPoolManagerTest, TraceCountStartsAtZero) {
    const auto trace_path = TraceTestPath();
    auto [dm, bpm] = MakeBPM(4);

    page_id_t a;
    ASSERT_NE(bpm->NewPage(&a), nullptr);
    (void)bpm->UnpinPage(a, false);

    // not tracing: fetches are not counted
    for (int i = 0; i < 2; ++i) {
        ASSERT_NE(bpm->FetchPage(a), nullptr);
        (void)bpm->UnpinPage(a, false);
    }
    EXPECT_EQ(bpm->GetTraceCount(), 0u);

    bpm->EnableTracing(trace_path);
    ASSERT_NE(bpm->FetchPage(a), nullptr);
    (void)bpm->UnpinPage(a, false);
    EXPECT_EQ(bpm->GetTraceCount(), 1u);

    // turning tracing on again starts over
    bpm->EnableTracing(trace_path);
    EXPECT_EQ(bpm->GetTraceCount(), 0u);

    bpm->DisableTracing();
    std::filesystem::remove(trace_path);
}

// ---------------------------------------------------------------------------
// A failed read must not lose the frame
// ---------------------------------------------------------------------------
TEST_F(BufferPoolManagerTest, FailedReadKeepsFrame) {
    auto [dm, bpm] = MakeBPM(2);

    EXPECT_THROW((void)bpm->FetchPage(INVALID_PAGE_ID), IOException);
    EXPECT_EQ(bpm->GetMissCount(), 0u);
    EXPECT_EQ(bpm->GetHitCount(), 0u);

    // both frames are still usable
    page_id_t pid0, pid1;
    EXPECT_NE(bpm->NewPage(&pid0), nullptr);
    EXPECT_NE(bpm->NewPage(&pid1), nullptr);
}

TEST_F(BufferPoolManagerTest, FailedReadAfterEvictionKeepsFrame) {
    auto [dm, bpm] = MakeBPM(1);

    page_id_t pid0;
    auto* p0 = bpm->NewPage(&pid0);
    ASSERT_NE(p0, nullptr);
    std::memcpy(p0->GetData(), "hello", 6);
    (void)bpm->UnpinPage(pid0, true);

    // the only frame is taken from p0, then the read fails
    EXPECT_THROW((void)bpm->FetchPage(INVALID_PAGE_ID), IOException);

    // p0 was written out before the read, so it comes back intact
    auto* again = bpm->FetchPage(pid0);
    ASSERT_NE(again, nullptr);
    EXPECT_EQ(std::memcmp(again->GetData(), "hello", 6), 0);
    (void)bpm->UnpinPage(pid0, false);

    page_id_t pid1;
    EXPECT_NE(bpm->NewPage(&pid1), nullptr);
}

TEST_F(BufferPoolManagerTest, FailedReadKeepsFrameEveryPolicy) {
    for (const auto kind : kEngineKinds) {
        SCOPED_TRACE(std::string(EvictionPolicyKindName(kind)));
        auto [dm, bpm] = MakeBPM(2, kind);

        EXPECT_THROW((void)bpm->FetchPage(INVALID_PAGE_ID), IOException);
        EXPECT_EQ(bpm->GetMissCount(), 0u);
        EXPECT_EQ(bpm->GetHitCount(), 0u);

        // both frames are still usable, at the same time
        page_id_t pid0, pid1;
        ASSERT_NE(bpm->NewPage(&pid0), nullptr);
        ASSERT_NE(bpm->NewPage(&pid1), nullptr);
        (void)bpm->UnpinPage(pid0, false);
        (void)bpm->UnpinPage(pid1, false);
    }
}

TEST_F(BufferPoolManagerTest, FailedReadAfterEvictionKeepsFrameEveryPolicy) {
    for (const auto kind : kEngineKinds) {
        SCOPED_TRACE(std::string(EvictionPolicyKindName(kind)));
        auto [dm, bpm] = MakeBPM(2, kind);

        page_id_t pids[2];
        const char* texts[2] = {"page 0", "page 1"};
        for (int i = 0; i < 2; ++i) {
            auto* page = bpm->NewPage(&pids[i]);
            ASSERT_NE(page, nullptr);
            std::memcpy(page->GetData(), texts[i], 7);
            (void)bpm->UnpinPage(pids[i], true);
        }

        // the pool is full, so a frame is taken from one page, then the read fails
        EXPECT_THROW((void)bpm->FetchPage(INVALID_PAGE_ID), IOException);

        // both pages come back intact and fit at once, so no frame was lost
        auto* first = bpm->FetchPage(pids[0]);
        auto* second = bpm->FetchPage(pids[1]);
        ASSERT_NE(first, nullptr);
        ASSERT_NE(second, nullptr);
        EXPECT_EQ(std::memcmp(first->GetData(), texts[0], 7), 0);
        EXPECT_EQ(std::memcmp(second->GetData(), texts[1], 7), 0);
        EXPECT_EQ(bpm->GetMissCount(), 1u);  // the page that lost its frame
        EXPECT_EQ(bpm->GetHitCount(), 1u);
        (void)bpm->UnpinPage(pids[0], false);
        (void)bpm->UnpinPage(pids[1], false);

        page_id_t pid2;
        ASSERT_NE(bpm->NewPage(&pid2), nullptr);
        (void)bpm->UnpinPage(pid2, false);
    }
}

TEST_F(BufferPoolManagerTest, EveryPolicyReturnsNullWhenAllPinned) {
    for (const auto kind : kEngineKinds) {
        for (page_id_t released = 0; released < 3; ++released) {
            SCOPED_TRACE(std::string(EvictionPolicyKindName(kind)) + ", unpinning page " + std::to_string(released));
            auto [dm, bpm] = MakeBPM(3, kind);

            // pages 0-4 on disk, then an empty pool
            for (page_id_t i = 0; i < 5; ++i) {
                page_id_t pid;
                ASSERT_NE(bpm->NewPage(&pid), nullptr);
                ASSERT_EQ(pid, i);
                (void)bpm->UnpinPage(pid, true);
            }
            bpm->DropAllPages();

            // three pages held pinned fill the pool
            for (page_id_t p = 0; p < 3; ++p) ASSERT_NE(bpm->FetchPage(p), nullptr);
            EXPECT_EQ(bpm->FetchPage(4), nullptr);

            // nothing was evicted: the three pages still hit
            bpm->ResetStats();
            for (page_id_t p = 0; p < 3; ++p) {
                ASSERT_NE(bpm->FetchPage(p), nullptr);
                (void)bpm->UnpinPage(p, false);
            }
            EXPECT_EQ(bpm->GetHitCount(), 3u);
            EXPECT_EQ(bpm->GetMissCount(), 0u);

            // unpinning one makes room, and that page is the one evicted
            ASSERT_TRUE(bpm->UnpinPage(released, false));
            ASSERT_NE(bpm->FetchPage(4), nullptr);
            std::set<page_id_t> resident;
            for (frame_id_t f = 0; f < 3; ++f) resident.insert(bpm->GetPage(f).GetPageId());
            std::set<page_id_t> expected = {0, 1, 2, 4};
            expected.erase(released);
            EXPECT_EQ(resident, expected);

            for (page_id_t p = 0; p < 3; ++p) {
                if (p != released) (void)bpm->UnpinPage(p, false);
            }
            (void)bpm->UnpinPage(4, false);
        }
    }
}

// ---------------------------------------------------------------------------
// Page visits
// ---------------------------------------------------------------------------
TEST(PageTest, ClearMetadataKeepsData) {
    auto page = std::make_unique<Page>();
    std::memcpy(page->GetData(), "keep me", 8);
    page->SetPageId(7);
    page->Pin();
    page->MarkDirty();

    page->ClearMetadata();

    EXPECT_EQ(page->GetPageId(), INVALID_PAGE_ID);
    EXPECT_EQ(page->GetPinCount(), 0);
    EXPECT_FALSE(page->IsDirty());
    EXPECT_EQ(std::memcmp(page->GetData(), "keep me", 8), 0);
}

TEST_F(BufferPoolManagerTest, SinglePageOneVisit) {
    auto [dm, bpm] = MakeBPM(4);
    page_id_t p0;
    ASSERT_NE(bpm->NewPage(&p0), nullptr);
    (void)bpm->UnpinPage(p0, false);

    bpm->ResetStats();
    for (int i = 0; i < 3; ++i) {
        ASSERT_NE(bpm->FetchPage(p0), nullptr);
        (void)bpm->UnpinPage(p0, false);
    }

    EXPECT_EQ(bpm->GetVisitCount(), 1u);
    EXPECT_EQ(bpm->GetMissCount(), 0u);
}

TEST_F(BufferPoolManagerTest, InterleavedPagesOneVisitEach) {
    auto [dm, bpm] = MakeBPM(8);
    page_id_t a, b;
    ASSERT_NE(bpm->NewPage(&a), nullptr);
    (void)bpm->UnpinPage(a, false);
    ASSERT_NE(bpm->NewPage(&b), nullptr);
    (void)bpm->UnpinPage(b, false);

    bpm->ResetStats();
    for (int round = 0; round < 200; ++round) {
        for (page_id_t pid : {a, b}) {
            for (int i = 0; i < 3; ++i) {
                ASSERT_NE(bpm->FetchPage(pid), nullptr);
                (void)bpm->UnpinPage(pid, false);
            }
        }
    }

    EXPECT_EQ(bpm->GetVisitCount(), 2u);
}

TEST_F(BufferPoolManagerTest, ReuseAfterWindowIsNewVisit) {
    auto [dm, bpm] = MakeBPM(128);
    std::vector<page_id_t> pids(101);
    for (auto& pid : pids) {
        ASSERT_NE(bpm->NewPage(&pid), nullptr);
        (void)bpm->UnpinPage(pid, false);
    }

    bpm->ResetStats();
    for (page_id_t pid : pids) {
        ASSERT_NE(bpm->FetchPage(pid), nullptr);
        (void)bpm->UnpinPage(pid, false);
    }
    // p0 again, after 100 other pages
    ASSERT_NE(bpm->FetchPage(pids[0]), nullptr);
    (void)bpm->UnpinPage(pids[0], false);

    EXPECT_EQ(bpm->GetVisitCount(), 102u);
    EXPECT_EQ(bpm->GetMissCount(), 0u);
}

TEST_F(BufferPoolManagerTest, MissIsAlwaysVisit) {
    auto [dm, bpm] = MakeBPM(1);
    page_id_t a, b;
    ASSERT_NE(bpm->NewPage(&a), nullptr);
    (void)bpm->UnpinPage(a, false);
    ASSERT_NE(bpm->NewPage(&b), nullptr);  // evicts a
    (void)bpm->UnpinPage(b, false);

    bpm->ResetStats();
    for (page_id_t pid : {a, b, a}) {
        ASSERT_NE(bpm->FetchPage(pid), nullptr);
        (void)bpm->UnpinPage(pid, false);
    }

    EXPECT_EQ(bpm->GetMissCount(), 3u);
    EXPECT_EQ(bpm->GetVisitCount(), 3u);
}

TEST_F(BufferPoolManagerTest, ResetStatsStartsFreshVisits) {
    auto [dm, bpm] = MakeBPM(4);
    page_id_t p0;
    ASSERT_NE(bpm->NewPage(&p0), nullptr);
    (void)bpm->UnpinPage(p0, false);

    for (int i = 0; i < 3; ++i) {
        ASSERT_NE(bpm->FetchPage(p0), nullptr);
        (void)bpm->UnpinPage(p0, false);
    }
    bpm->ResetStats();
    ASSERT_NE(bpm->FetchPage(p0), nullptr);
    (void)bpm->UnpinPage(p0, false);

    EXPECT_EQ(bpm->GetVisitCount(), 1u);
}

// ---------------------------------------------------------------------------
// Policy timing
// ---------------------------------------------------------------------------
TEST_F(BufferPoolManagerTest, PolicyTimingCountsCalls) {
    auto [dm, bpm] = MakeBPM(1);
    bpm->SetPolicyTiming(true);

    page_id_t p0, p1;
    ASSERT_NE(bpm->NewPage(&p0), nullptr);
    (void)bpm->UnpinPage(p0, false);
    ASSERT_NE(bpm->NewPage(&p1), nullptr);  // evicts p0
    (void)bpm->UnpinPage(p1, false);
    ASSERT_TRUE(bpm->DeletePage(p1));

    PolicyStats stats = bpm->GetPolicyStats();
    EXPECT_EQ(stats.record_calls, 2u);
    EXPECT_EQ(stats.set_evictable_calls, 4u);
    EXPECT_EQ(stats.evict_calls, 1u);
    EXPECT_EQ(stats.remove_calls, 1u);
}

TEST_F(BufferPoolManagerTest, PolicyTimingOffAfterDisable) {
    auto [dm, bpm] = MakeBPM(1);

    page_id_t p0, p1;
    ASSERT_NE(bpm->NewPage(&p0), nullptr);
    (void)bpm->UnpinPage(p0, false);
    ASSERT_NE(bpm->NewPage(&p1), nullptr);
    (void)bpm->UnpinPage(p1, false);
    ASSERT_TRUE(bpm->DeletePage(p1));

    // Timing was never on, so nothing was counted
    PolicyStats never_on = bpm->GetPolicyStats();
    EXPECT_EQ(never_on.TotalCalls(), 0u);

    ASSERT_NE(bpm->NewPage(&p0), nullptr);
    (void)bpm->UnpinPage(p0, false);

    bpm->SetPolicyTiming(true);
    ASSERT_NE(bpm->FetchPage(p0), nullptr);
    (void)bpm->UnpinPage(p0, false);
    bpm->SetPolicyTiming(false);
    ASSERT_NE(bpm->FetchPage(p0), nullptr);
    (void)bpm->UnpinPage(p0, false);

    PolicyStats stats = bpm->GetPolicyStats();
    EXPECT_EQ(stats.record_calls, 1u);
    EXPECT_EQ(stats.set_evictable_calls, 2u);
}

TEST_F(BufferPoolManagerTest, PolicyTimingKeepsSameMisses) {
    constexpr int kPages = 10;
    constexpr int kFrames = 4;
    constexpr int kFetches = 200;

    struct Outcome {
        uint64_t hits;
        uint64_t misses;
        std::vector<page_id_t> frame_pages;
    };

    // Same fetch pattern on two pools; only one has timing on
    auto run = [&](bool timing, const std::string& file) {
        DiskManager dm(file);
        BufferPoolManager bpm(kFrames, &dm, std::make_unique<LRUEvictionPolicy>(kFrames));
        bpm.SetPolicyTiming(timing);

        std::vector<page_id_t> pids(kPages);
        for (auto& pid : pids) {
            EXPECT_NE(bpm.NewPage(&pid), nullptr);
            (void)bpm.UnpinPage(pid, false);
        }
        bpm.ResetStats();
        for (int i = 0; i < kFetches; ++i) {
            // Every other fetch goes back to page 0, so there is reuse
            page_id_t pid = (i % 2 == 0) ? pids[0] : pids[(i * 7) % kPages];
            EXPECT_NE(bpm.FetchPage(pid), nullptr);
            (void)bpm.UnpinPage(pid, false);
        }

        Outcome out{bpm.GetHitCount(), bpm.GetMissCount(), {}};
        for (frame_id_t f = 0; f < kFrames; ++f) {
            out.frame_pages.push_back(bpm.GetPage(f).GetPageId());
        }
        return out;
    };

    const Outcome plain = run(false, test_file_);
    std::filesystem::remove(test_file_);
    const Outcome timed = run(true, test_file_);

    EXPECT_GT(plain.hits, 0u);
    EXPECT_GT(plain.misses, 0u);
    EXPECT_EQ(plain.hits, timed.hits);
    EXPECT_EQ(plain.misses, timed.misses);
    EXPECT_EQ(plain.frame_pages, timed.frame_pages);
}

// ---------------------------------------------------------------------------
// What the buffer pool tells the policy
// ---------------------------------------------------------------------------

// So a failed check prints the type's name
void PrintTo(AccessType type, std::ostream* os) {
    switch (type) {
        case AccessType::NEW_USE: *os << "NEW_USE"; break;
        case AccessType::TOUCH: *os << "TOUCH"; break;
        case AccessType::REPEAT: *os << "REPEAT"; break;
    }
}

namespace {

struct TypedCall {
    frame_id_t frame;
    page_id_t page;
    AccessType type;
};

// Decides like LRU and remembers every call the buffer pool makes
class RecordingPolicy : public EvictionPolicy {
public:
    RecordingPolicy(size_t num_frames, bool uses_types) : lru_(num_frames), uses_types_(uses_types) {}

    void RecordAccess(frame_id_t frame_id) override { lru_.RecordAccess(frame_id); }
    void RecordAccess(frame_id_t frame_id, page_id_t page_id) override {
        plain_calls.emplace_back(frame_id, page_id);
        lru_.RecordAccess(frame_id);
    }
    void RecordAccess(frame_id_t frame_id, page_id_t page_id, AccessType type) override {
        typed_calls.push_back({frame_id, page_id, type});
        lru_.RecordAccess(frame_id);
    }
    void SetEvictable(frame_id_t frame_id, bool evictable) override { lru_.SetEvictable(frame_id, evictable); }
    std::optional<frame_id_t> Evict() override {
        ++plain_evicts;
        return lru_.Evict();
    }
    std::optional<frame_id_t> Evict(page_id_t incoming_page) override {
        evict_pages.push_back(incoming_page);
        return lru_.Evict();
    }
    void Remove(frame_id_t frame_id) override { lru_.Remove(frame_id); }
    size_t Size() const override { return lru_.Size(); }
    bool UsesAccessTypes() const override { return uses_types_; }
    void Reset() override {
        ++reset_calls;
        lru_.Reset();
    }

    std::vector<AccessType> Types() const {
        std::vector<AccessType> types;
        for (const auto& call : typed_calls) {
            types.push_back(call.type);
        }
        return types;
    }

    std::vector<TypedCall> typed_calls;
    std::vector<std::pair<frame_id_t, page_id_t>> plain_calls;
    std::vector<page_id_t> evict_pages;
    int plain_evicts{0};
    int reset_calls{0};

private:
    LRUEvictionPolicy lru_;
    bool uses_types_;
};

struct RecordingBundle {
    std::unique_ptr<DiskManager> disk_manager;
    std::unique_ptr<BufferPoolManager> bpm;
    RecordingPolicy* policy;  // owned by bpm
};

RecordingBundle MakeRecordingBPM(const std::string& file, size_t pool_size, bool uses_types) {
    auto dm = std::make_unique<DiskManager>(file);
    auto policy = std::make_unique<RecordingPolicy>(pool_size, uses_types);
    RecordingPolicy* raw = policy.get();
    auto bpm = std::make_unique<BufferPoolManager>(pool_size, dm.get(), std::move(policy));
    return {std::move(dm), std::move(bpm), raw};
}

// Fetches and unpins each page in turn
void FetchAll(BufferPoolManager& bpm, std::initializer_list<page_id_t> pages) {
    for (const page_id_t pid : pages) {
        ASSERT_NE(bpm.FetchPage(pid), nullptr);
        (void)bpm.UnpinPage(pid, false);
    }
}

void FetchRange(BufferPoolManager& bpm, page_id_t first, page_id_t last) {
    for (page_id_t pid = first; pid <= last; ++pid) {
        ASSERT_NE(bpm.FetchPage(pid), nullptr);
        (void)bpm.UnpinPage(pid, false);
    }
}

}  // namespace

TEST_F(BufferPoolManagerTest, AccessTypesForVisitPolicy) {
    auto [dm, bpm, policy] = MakeRecordingBPM(test_file_, 8, true);

    FetchAll(*bpm, {0, 0, 1, 0});

    const std::vector<AccessType> expected = {AccessType::NEW_USE, AccessType::REPEAT, AccessType::NEW_USE, AccessType::TOUCH};
    EXPECT_EQ(policy->Types(), expected);
    EXPECT_TRUE(policy->plain_calls.empty());

    // frames and pages come through too
    ASSERT_EQ(policy->typed_calls.size(), 4u);
    EXPECT_EQ(policy->typed_calls[2].page, 1u);
    EXPECT_EQ(policy->typed_calls[3].page, 0u);
    EXPECT_EQ(policy->typed_calls[3].frame, policy->typed_calls[0].frame);
}

TEST_F(BufferPoolManagerTest, ReturnAfter64RunsIsTouch) {
    auto [dm, bpm, policy] = MakeRecordingBPM(test_file_, 80, true);

    FetchAll(*bpm, {0});
    FetchRange(*bpm, 1, 63);  // 63 other pages, so page 0 comes back 64 runs later
    FetchAll(*bpm, {0});

    EXPECT_EQ(bpm->GetMissCount(), 64u);
    EXPECT_EQ(policy->Types().back(), AccessType::TOUCH);
}

TEST_F(BufferPoolManagerTest, ReturnAfter65RunsIsNewUse) {
    auto [dm, bpm, policy] = MakeRecordingBPM(test_file_, 80, true);

    FetchAll(*bpm, {0});
    FetchRange(*bpm, 1, 64);
    FetchAll(*bpm, {0});

    EXPECT_EQ(bpm->GetMissCount(), 65u);  // page 0 is still resident: a hit, but a new use
    EXPECT_EQ(policy->Types().back(), AccessType::NEW_USE);
}

TEST_F(BufferPoolManagerTest, ResetStatsDoesNotChangeAccessTypes) {
    auto [dm, bpm, policy] = MakeRecordingBPM(test_file_, 80, true);

    // ResetStats doesn't count as a page change
    FetchAll(*bpm, {0});
    bpm->ResetStats();
    FetchAll(*bpm, {0});
    EXPECT_EQ(policy->Types().back(), AccessType::REPEAT);

    // and doesn't start a new visit
    FetchAll(*bpm, {0});
    FetchRange(*bpm, 1, 63);
    bpm->ResetStats();
    FetchAll(*bpm, {0});
    EXPECT_EQ(policy->Types().back(), AccessType::TOUCH);
}

TEST_F(BufferPoolManagerTest, NewPageIsNewUseAndAPageChange) {
    auto [dm, bpm, policy] = MakeRecordingBPM(test_file_, 8, true);

    page_id_t a, p;
    ASSERT_NE(bpm->NewPage(&a), nullptr);
    (void)bpm->UnpinPage(a, false);
    ASSERT_NE(bpm->NewPage(&p), nullptr);
    (void)bpm->UnpinPage(p, false);
    ASSERT_EQ(policy->typed_calls.size(), 2u);
    EXPECT_EQ(policy->typed_calls[1].page, p);
    EXPECT_EQ(policy->typed_calls[1].type, AccessType::NEW_USE);

    // p was the last page, so fetching it is a repeat; a is a return within the window
    FetchAll(*bpm, {p, a});
    const std::vector<AccessType> expected = {AccessType::NEW_USE, AccessType::NEW_USE, AccessType::REPEAT, AccessType::TOUCH};
    EXPECT_EQ(policy->Types(), expected);
}

TEST_F(BufferPoolManagerTest, PolicyWithoutTypesGetsOldCall) {
    auto [dm, bpm, policy] = MakeRecordingBPM(test_file_, 8, false);

    page_id_t a;
    ASSERT_NE(bpm->NewPage(&a), nullptr);
    (void)bpm->UnpinPage(a, false);
    FetchAll(*bpm, {a, a, 5, a});

    EXPECT_TRUE(policy->typed_calls.empty());
    const std::vector<std::pair<frame_id_t, page_id_t>> expected = {{0, a}, {0, a}, {0, a}, {1, 5}, {0, a}};
    EXPECT_EQ(policy->plain_calls, expected);
}

TEST_F(BufferPoolManagerTest, EvictGetsIncomingPage) {
    auto [dm, bpm, policy] = MakeRecordingBPM(test_file_, 2, false);

    FetchAll(*bpm, {1, 2});
    EXPECT_TRUE(policy->evict_pages.empty());  // free frames first

    FetchAll(*bpm, {5});
    page_id_t fresh;
    ASSERT_NE(bpm->NewPage(&fresh), nullptr);
    (void)bpm->UnpinPage(fresh, false);

    const std::vector<page_id_t> expected = {5, INVALID_PAGE_ID};
    EXPECT_EQ(policy->evict_pages, expected);
    EXPECT_EQ(policy->plain_evicts, 0);
}

TEST_F(BufferPoolManagerTest, DropAllPagesWritesAndEmpties) {
    auto [dm, bpm, policy] = MakeRecordingBPM(test_file_, 3, true);

    page_id_t a, b, c;
    ASSERT_NE(bpm->NewPage(&a), nullptr);
    (void)bpm->UnpinPage(a, false);
    ASSERT_NE(bpm->NewPage(&b), nullptr);
    (void)bpm->UnpinPage(b, false);
    auto* page_c = bpm->NewPage(&c);
    ASSERT_NE(page_c, nullptr);
    std::memcpy(page_c->GetData(), "drop me", 8);
    (void)bpm->UnpinPage(c, true);

    // leave the free list out of order: frames 1 then 0, with c still in frame 2
    ASSERT_TRUE(bpm->DeletePage(b));
    ASSERT_TRUE(bpm->DeletePage(a));
    FetchAll(*bpm, {c});
    const uint64_t hits = bpm->GetHitCount();
    const uint64_t misses = bpm->GetMissCount();

    bpm->DropAllPages();

    for (frame_id_t f = 0; f < 3; ++f) {
        EXPECT_EQ(bpm->GetPage(f).GetPageId(), INVALID_PAGE_ID) << "frame " << f;
    }
    EXPECT_EQ(policy->reset_calls, 1);
    EXPECT_EQ(bpm->GetHitCount(), hits);  // stats are left alone
    EXPECT_EQ(bpm->GetMissCount(), misses);

    // c was written out, so it reads back from disk
    auto* again = bpm->FetchPage(c);
    ASSERT_NE(again, nullptr);
    EXPECT_EQ(bpm->GetMissCount(), misses + 1);
    EXPECT_EQ(std::memcmp(again->GetData(), "drop me", 8), 0);
    (void)bpm->UnpinPage(c, false);

    // frames are handed out from 0 again, like a new pool
    FetchAll(*bpm, {a, b});
    EXPECT_EQ(bpm->GetPage(0).GetPageId(), c);
    EXPECT_EQ(bpm->GetPage(1).GetPageId(), a);
    EXPECT_EQ(bpm->GetPage(2).GetPageId(), b);
}

TEST_F(BufferPoolManagerTest, DropAllPagesThrowsOnPinnedPage) {
    auto [dm, bpm, policy] = MakeRecordingBPM(test_file_, 4, true);

    page_id_t a, b;
    ASSERT_NE(bpm->NewPage(&a), nullptr);
    (void)bpm->UnpinPage(a, true);
    ASSERT_NE(bpm->NewPage(&b), nullptr);  // stays pinned

    EXPECT_THROW(bpm->DropAllPages(), DatabaseException);

    // nothing changed: both pages are resident, a is still dirty, the policy kept its state
    EXPECT_EQ(policy->reset_calls, 0);
    EXPECT_TRUE(bpm->GetPage(0).IsDirty());
    EXPECT_EQ(bpm->GetPage(1).GetPinCount(), 1);
    bpm->ResetStats();
    FetchAll(*bpm, {a, b});
    EXPECT_EQ(bpm->GetHitCount(), 2u);
    EXPECT_EQ(bpm->GetMissCount(), 0u);
    (void)bpm->UnpinPage(b, false);
}

TEST_F(BufferPoolManagerTest, DropAllPagesOnEmptyPool) {
    auto [dm, bpm, policy] = MakeRecordingBPM(test_file_, 4, true);

    EXPECT_NO_THROW(bpm->DropAllPages());
    EXPECT_NO_THROW(bpm->DropAllPages());
    EXPECT_EQ(policy->reset_calls, 2);

    page_id_t a;
    ASSERT_NE(bpm->NewPage(&a), nullptr);
    EXPECT_EQ(bpm->GetPage(0).GetPageId(), a);
    (void)bpm->UnpinPage(a, false);
}

TEST_F(BufferPoolManagerTest, PolicyTimingKeepsAccessTypes) {
    auto run = [&](bool timing) {
        std::filesystem::remove(test_file_);
        auto [dm, bpm, policy] = MakeRecordingBPM(test_file_, 80, true);
        bpm->SetPolicyTiming(timing);

        page_id_t fresh;
        EXPECT_NE(bpm->NewPage(&fresh), nullptr);
        (void)bpm->UnpinPage(fresh, false);
        FetchAll(*bpm, {0, 0, 1, 0});
        FetchRange(*bpm, 2, 66);
        FetchAll(*bpm, {1, 1});
        bpm->DropAllPages();
        FetchAll(*bpm, {1, 0});

        std::vector<std::pair<frame_id_t, page_id_t>> where;
        for (const auto& call : policy->typed_calls) {
            where.emplace_back(call.frame, call.page);
        }
        if (timing) {
            EXPECT_EQ(bpm->GetPolicyStats().record_calls, policy->typed_calls.size());
        }
        return std::make_pair(policy->Types(), where);
    };

    const auto plain = run(false);
    const auto timed = run(true);

    ASSERT_EQ(plain.first.size(), 74u);
    EXPECT_EQ(plain.first, timed.first);
    EXPECT_EQ(plain.second, timed.second);
}

}  // namespace shilmandb
