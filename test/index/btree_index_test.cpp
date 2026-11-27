#include <gtest/gtest.h>
#include <string>
#include <unistd.h>
#include "index/btree_index.hpp"
#include "common/exception.hpp"
#include "storage/slotted_page.hpp"
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "storage/disk_manager.hpp"
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

namespace shilmandb {

class BTreeIndexTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() /
                      ("shilmandb_btree_index_test_" + std::to_string(::getpid()) + ".db")).string();
        std::filesystem::remove(test_file_);
    }

    void TearDown() override {
        std::filesystem::remove(test_file_);
    }

    struct BPMBundle {
        std::unique_ptr<DiskManager> disk_manager;
        std::unique_ptr<BufferPoolManager> bpm;
    };

    BPMBundle MakeBPM(size_t pool_size = 1000) {
        auto dm = std::make_unique<DiskManager>(test_file_);
        auto eviction = std::make_unique<LRUEvictionPolicy>(pool_size);
        auto bpm = std::make_unique<BufferPoolManager>(
            pool_size, dm.get(), std::move(eviction));
        return {std::move(dm), std::move(bpm)};
    }
};

// ---------------------------------------------------------------------------
// Empty tree
// ---------------------------------------------------------------------------
TEST_F(BTreeIndexTest, EmptyTree) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::INTEGER);

    // PointLookup on empty tree returns empty
    auto rids = index.PointLookup(Value(42));
    EXPECT_TRUE(rids.empty());

    // Begin() == End()
    auto it = index.Begin();
    EXPECT_TRUE(it.IsEnd());
    EXPECT_TRUE(!(it != index.End()));

    // Range scan on empty tree returns nothing
    auto it2 = index.Begin(Value(0));
    EXPECT_TRUE(it2.IsEnd());

    // Root is INVALID
    EXPECT_EQ(index.GetRootPageId(), INVALID_PAGE_ID);
}

// ---------------------------------------------------------------------------
// Insert + PointLookup (no splits needed for small counts)
// ---------------------------------------------------------------------------
TEST_F(BTreeIndexTest, InsertAndPointLookupSmall) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::INTEGER);

    // Insert 100 keys — well within a single leaf (MaxKeys=817 for INTEGER)
    for (int32_t i = 0; i < 100; ++i) {
        index.Insert(Value(i), RID(i, 0));
    }

    EXPECT_NE(index.GetRootPageId(), INVALID_PAGE_ID);

    // PointLookup each
    for (int32_t i = 0; i < 100; ++i) {
        auto rids = index.PointLookup(Value(i));
        ASSERT_EQ(rids.size(), 1u) << "key=" << i;
        EXPECT_EQ(rids[0], RID(i, 0));
    }

    // Lookup non-existent key
    auto rids = index.PointLookup(Value(999));
    EXPECT_TRUE(rids.empty());
}

// ---------------------------------------------------------------------------
// Duplicate keys
// ---------------------------------------------------------------------------
TEST_F(BTreeIndexTest, DuplicateKeys) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::INTEGER);

    for (int32_t i = 0; i < 100; ++i) {
        index.Insert(Value(42), RID(i, 0));
    }

    auto rids = index.PointLookup(Value(42));
    ASSERT_EQ(rids.size(), 100u);
    // Verify all RIDs present (order may vary within duplicates)
    for (int32_t i = 0; i < 100; ++i) {
        EXPECT_NE(std::find(rids.begin(), rids.end(), RID(i, 0)), rids.end())
            << "Missing RID(" << i << ", 0)";
    }
}

// ---------------------------------------------------------------------------
// Delete basic
// ---------------------------------------------------------------------------
TEST_F(BTreeIndexTest, DeleteBasic) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::INTEGER);

    for (int32_t i = 0; i < 100; ++i) {
        index.Insert(Value(i), RID(i, 0));
    }

    // Delete even keys
    for (int32_t i = 0; i < 100; i += 2) {
        index.Delete(Value(i), RID(i, 0));
    }

    // Verify deleted keys are gone
    for (int32_t i = 0; i < 100; i += 2) {
        auto rids = index.PointLookup(Value(i));
        EXPECT_TRUE(rids.empty()) << "key=" << i << " should be deleted";
    }

    // Verify remaining keys still present
    for (int32_t i = 1; i < 100; i += 2) {
        auto rids = index.PointLookup(Value(i));
        ASSERT_EQ(rids.size(), 1u) << "key=" << i;
        EXPECT_EQ(rids[0], RID(i, 0));
    }
}

// ---------------------------------------------------------------------------
// Delete all
// ---------------------------------------------------------------------------
TEST_F(BTreeIndexTest, DeleteAll) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::INTEGER);

    for (int32_t i = 0; i < 100; ++i) {
        index.Insert(Value(i), RID(i, 0));
    }

    for (int32_t i = 0; i < 100; ++i) {
        index.Delete(Value(i), RID(i, 0));
    }

    // All lookups return empty
    for (int32_t i = 0; i < 100; ++i) {
        auto rids = index.PointLookup(Value(i));
        EXPECT_TRUE(rids.empty());
    }
}

// ---------------------------------------------------------------------------
// Mixed insert and lookup
// ---------------------------------------------------------------------------
TEST_F(BTreeIndexTest, MixedInsertAndLookup) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::INTEGER);

    for (int32_t i = 0; i < 500; ++i) {
        index.Insert(Value(i), RID(i, 0));
        // Immediately verify it's findable
        auto rids = index.PointLookup(Value(i));
        ASSERT_EQ(rids.size(), 1u) << "key=" << i;
        EXPECT_EQ(rids[0], RID(i, 0));
    }
}

// ---------------------------------------------------------------------------
// Split occurs
// ---------------------------------------------------------------------------
TEST_F(BTreeIndexTest, SplitOccurs) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::INTEGER);

    // INTEGER MaxKeys = 817, so 900 inserts must trigger at least one split
    for (int32_t i = 0; i < 900; ++i) {
        index.Insert(Value(i), RID(i, 0));
    }

    // All keys still findable after split
    for (int32_t i = 0; i < 900; ++i) {
        auto rids = index.PointLookup(Value(i));
        ASSERT_EQ(rids.size(), 1u) << "key=" << i;
        EXPECT_EQ(rids[0], RID(i, 0));
    }

    // Root should now be an internal node (not a leaf)
    auto* root_page = bundle.bpm->FetchPage(index.GetRootPageId());
    auto* hdr = SlottedPage::GetHeader(root_page->GetData());
    EXPECT_EQ(hdr->flags & 1u, 0u) << "Root should be internal after split";
    (void)bundle.bpm->UnpinPage(index.GetRootPageId(), false);
}

// ---------------------------------------------------------------------------
// Insert 1000 sequential keys
// ---------------------------------------------------------------------------
TEST_F(BTreeIndexTest, InsertAndPointLookup) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::INTEGER);

    for (int32_t i = 0; i < 1000; ++i) {
        index.Insert(Value(i), RID(i, 0));
    }

    for (int32_t i = 0; i < 1000; ++i) {
        auto rids = index.PointLookup(Value(i));
        ASSERT_EQ(rids.size(), 1u) << "key=" << i;
        EXPECT_EQ(rids[0], RID(i, 0));
    }
}

// ---------------------------------------------------------------------------
// Insert 10,000 random keys
// ---------------------------------------------------------------------------
TEST_F(BTreeIndexTest, InsertRandomKeys) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::INTEGER);

    // Generate keys 0..9999 in random order
    std::vector<int32_t> keys(10000);
    std::iota(keys.begin(), keys.end(), 0);
    std::mt19937 rng(42);  // fixed seed for reproducibility
    std::shuffle(keys.begin(), keys.end(), rng);

    for (auto k : keys) {
        index.Insert(Value(k), RID(k, 0));
    }

    for (int32_t i = 0; i < 10000; ++i) {
        auto rids = index.PointLookup(Value(i));
        ASSERT_EQ(rids.size(), 1u) << "key=" << i;
        EXPECT_EQ(rids[0], RID(i, 0));
    }
}

// ---------------------------------------------------------------------------
// Tree height growth
// ---------------------------------------------------------------------------
TEST_F(BTreeIndexTest, TreeHeightGrowth) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::INTEGER);

    for (int32_t i = 0; i < 10000; ++i) {
        index.Insert(Value(i), RID(i, 0));
    }

    // Root must be an internal node (height >= 2)
    auto* root_page = bundle.bpm->FetchPage(index.GetRootPageId());
    auto* hdr = SlottedPage::GetHeader(root_page->GetData());
    EXPECT_EQ(hdr->flags & 1u, 0u) << "Root should be internal with 10k keys";
    (void)bundle.bpm->UnpinPage(index.GetRootPageId(), false);
}

// ---------------------------------------------------------------------------
// Insert sequential and reverse
// ---------------------------------------------------------------------------
TEST_F(BTreeIndexTest, InsertSequentialAndReverse) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::INTEGER);

    // Insert 0..4999 forward
    for (int32_t i = 0; i < 5000; ++i) {
        index.Insert(Value(i), RID(i, 0));
    }
    // Insert 9999..5000 reverse
    for (int32_t i = 9999; i >= 5000; --i) {
        index.Insert(Value(i), RID(i, 0));
    }

    // All 10,000 findable
    for (int32_t i = 0; i < 10000; ++i) {
        auto rids = index.PointLookup(Value(i));
        ASSERT_EQ(rids.size(), 1u) << "key=" << i;
        EXPECT_EQ(rids[0], RID(i, 0));
    }
}

// ---------------------------------------------------------------------------
// Full scan
// ---------------------------------------------------------------------------
TEST_F(BTreeIndexTest, FullScan) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::INTEGER);

    // Insert 1000 keys in random order
    std::vector<int32_t> keys(1000);
    std::iota(keys.begin(), keys.end(), 0);
    std::mt19937 rng(123);
    std::shuffle(keys.begin(), keys.end(), rng);

    for (auto k : keys) {
        index.Insert(Value(k), RID(k, 0));
    }

    // Full scan: collect all (key, rid) pairs
    std::vector<int32_t> scanned;
    for (auto it = index.Begin(); it != index.End(); ++it) {
        auto [key, rid] = *it;
        scanned.push_back(key.integer_);
    }

    // Verify count
    ASSERT_EQ(scanned.size(), 1000u);

    // Verify sorted order
    for (size_t i = 1; i < scanned.size(); ++i) {
        EXPECT_LE(scanned[i - 1], scanned[i]) << "Not sorted at index " << i;
    }

    // Verify all keys present
    std::sort(scanned.begin(), scanned.end());
    for (int32_t i = 0; i < 1000; ++i) {
        EXPECT_EQ(scanned[i], i);
    }
}

// ---------------------------------------------------------------------------
// Range scan
// ---------------------------------------------------------------------------
TEST_F(BTreeIndexTest, RangeScan) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::INTEGER);

    for (int32_t i = 0; i < 1000; ++i) {
        index.Insert(Value(i), RID(i, 0));
    }

    // Range scan [100, 200]
    std::vector<int32_t> scanned;
    for (auto it = index.Begin(Value(100)); it != index.End(); ++it) {
        auto [key, rid] = *it;
        if (key.integer_ > 200) break;
        scanned.push_back(key.integer_);
    }

    ASSERT_EQ(scanned.size(), 101u);  // 100, 101, ..., 200
    for (int32_t i = 0; i < 101; ++i) {
        EXPECT_EQ(scanned[i], 100 + i);
    }
}

// ---------------------------------------------------------------------------
// Range scan empty range
// ---------------------------------------------------------------------------
TEST_F(BTreeIndexTest, RangeScanEmptyRange) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::INTEGER);

    for (int32_t i = 0; i < 100; ++i) {
        index.Insert(Value(i), RID(i, 0));
    }

    // Range scan [200, 300] — no keys in this range
    auto it = index.Begin(Value(200));
    EXPECT_TRUE(it.IsEnd());
}

// ---------------------------------------------------------------------------
// Large-scale insert
// ---------------------------------------------------------------------------
TEST_F(BTreeIndexTest, LargeScaleInsert) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::INTEGER);

    for (int32_t i = 0; i < 10000; ++i) {
        index.Insert(Value(i), RID(i, 0));
    }

    // Point lookup every key
    for (int32_t i = 0; i < 10000; ++i) {
        auto rids = index.PointLookup(Value(i));
        ASSERT_EQ(rids.size(), 1u) << "key=" << i;
    }

    // Range scan [0, 9999] returns all 10,000
    int count = 0;
    for (auto it = index.Begin(Value(0)); it != index.End(); ++it) {
        auto [key, rid] = *it;
        if (key.integer_ > 9999) break;
        ++count;
    }
    EXPECT_EQ(count, 10000);
}

// ---------------------------------------------------------------------------
// Delete without rebalance
// ---------------------------------------------------------------------------
TEST_F(BTreeIndexTest, DeleteWithoutRebalance) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::INTEGER);

    for (int32_t i = 0; i < 1000; ++i) {
        index.Insert(Value(i), RID(i, 0));
    }

    // Delete first 500
    for (int32_t i = 0; i < 500; ++i) {
        index.Delete(Value(i), RID(i, 0));
    }

    // Deleted keys are gone
    for (int32_t i = 0; i < 500; ++i) {
        auto rids = index.PointLookup(Value(i));
        EXPECT_TRUE(rids.empty()) << "key=" << i;
    }

    // Remaining 500 still found
    for (int32_t i = 500; i < 1000; ++i) {
        auto rids = index.PointLookup(Value(i));
        ASSERT_EQ(rids.size(), 1u) << "key=" << i;
    }

    // Full scan returns exactly 500 keys, all sorted
    std::vector<int32_t> scanned;
    for (auto it = index.Begin(); it != index.End(); ++it) {
        auto [key, rid] = *it;
        scanned.push_back(key.integer_);
    }
    ASSERT_EQ(scanned.size(), 500u);
    for (size_t i = 1; i < scanned.size(); ++i) {
        EXPECT_LE(scanned[i - 1], scanned[i]);
    }
}

// ---------------------------------------------------------------------------
// One key with thousands of copies, spread over several leaves
// ---------------------------------------------------------------------------
namespace {

constexpr int32_t kManyKey = 500;
constexpr int32_t kManyCopies = 3000;

// Keys 1..300, then kManyCopies of kManyKey, then 600..900. The copies start in a leaf that
// also holds smaller keys and run on through the leaves after it.
void FillWithManyCopies(BTreeIndex& index) {
    // More copies than three leaves can hold, so they sit in at least four.
    ASSERT_GT(kManyCopies, 3 * BTreeLeafPage::MaxKeys(TypeId::INTEGER));

    for (int32_t i = 1; i <= 300; ++i) {
        index.Insert(Value(i), RID(i, 0));
    }
    for (int32_t i = 0; i < kManyCopies; ++i) {
        index.Insert(Value(kManyKey), RID(i, 1));
    }
    for (int32_t i = 600; i <= 900; ++i) {
        index.Insert(Value(i), RID(i, 0));
    }
}

// Counts the entries equal to kManyKey from `it` on. Sets `first_rid` to the first one seen.
int32_t CountManyKey(BTreeIndex::Iterator it, RID* first_rid = nullptr) {
    int32_t count = 0;
    for (; !it.IsEnd(); ++it) {
        auto [key, rid] = *it;
        if (key.integer_ != kManyKey) break;
        if (count == 0 && first_rid != nullptr) *first_rid = rid;
        ++count;
    }
    return count;
}

}  // namespace

TEST_F(BTreeIndexTest, DuplicatesAcrossLeavesRangeFromKey) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::INTEGER);
    ASSERT_NO_FATAL_FAILURE(FillWithManyCopies(index));

    RID first(0, 0);
    EXPECT_EQ(CountManyKey(index.Begin(Value(kManyKey)), &first), kManyCopies);
    EXPECT_EQ(first, RID(0, 1));
}

TEST_F(BTreeIndexTest, DuplicatesAcrossLeavesRangeFromBelow) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::INTEGER);
    ASSERT_NO_FATAL_FAILURE(FillWithManyCopies(index));

    auto it = index.Begin(Value(450));
    ASSERT_FALSE(it.IsEnd());
    EXPECT_EQ((*it).first.integer_, kManyKey);
    EXPECT_EQ(CountManyKey(it), kManyCopies);
}

TEST_F(BTreeIndexTest, DuplicatesAcrossLeavesPointLookup) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::INTEGER);
    ASSERT_NO_FATAL_FAILURE(FillWithManyCopies(index));

    const auto rids = index.PointLookup(Value(kManyKey));
    ASSERT_EQ(rids.size(), static_cast<size_t>(kManyCopies));
    for (int32_t i = 0; i < kManyCopies; ++i) {
        EXPECT_NE(std::find(rids.begin(), rids.end(), RID(i, 1)), rids.end()) << "Missing RID(" << i << ", 1)";
    }
}

TEST_F(BTreeIndexTest, DuplicatesAcrossLeavesDelete) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::INTEGER);
    ASSERT_NO_FATAL_FAILURE(FillWithManyCopies(index));

    // One entry from the first leaf, one from a middle one, one from the last copy.
    // The later two sit in leaves after the first, so Delete has to walk on to find them.
    index.Delete(Value(kManyKey), RID(0, 1));
    EXPECT_EQ(index.PointLookup(Value(kManyKey)).size(), static_cast<size_t>(kManyCopies - 1));

    index.Delete(Value(kManyKey), RID(1500, 1));
    EXPECT_EQ(index.PointLookup(Value(kManyKey)).size(), static_cast<size_t>(kManyCopies - 2));

    index.Delete(Value(kManyKey), RID(kManyCopies - 1, 1));
    EXPECT_EQ(index.PointLookup(Value(kManyKey)).size(), static_cast<size_t>(kManyCopies - 3));

    // An entry that is not there changes nothing and ends the walk.
    index.Delete(Value(kManyKey), RID(5000, 1));
    const auto rids = index.PointLookup(Value(kManyKey));
    EXPECT_EQ(rids.size(), static_cast<size_t>(kManyCopies - 3));

    for (const RID& gone : {RID(0, 1), RID(1500, 1), RID(kManyCopies - 1, 1)}) {
        EXPECT_EQ(std::find(rids.begin(), rids.end(), gone), rids.end()) << "RID(" << gone.page_id << ", 1) is still there";
    }
    EXPECT_NE(std::find(rids.begin(), rids.end(), RID(1, 1)), rids.end());
}


namespace {

constexpr int64_t kDamageKeys = 5000;  // sorted keys: a root over about 17 BIGINT leaves

// A two-level tree of keys 0 to kDamageKeys - 1.
void FillTwoLevels(BTreeIndex& index, BufferPoolManager& bpm) {
    for (int64_t k = 0; k < kDamageKeys; ++k) index.Insert(Value(k), RID(static_cast<page_id_t>(k), 0));
    const page_id_t root = index.GetRootPageId();
    const char* data = bpm.FetchPage(root)->GetData();
    const bool internal = (SlottedPage::GetHeader(data)->flags & 1u) == 0;
    (void)bpm.UnpinPage(root, false);
    ASSERT_TRUE(internal);
}

// Changes one page in place, through the pool.
void Damage(BufferPoolManager& bpm, page_id_t pid, const std::function<void(char*)>& change) {
    change(bpm.FetchPage(pid)->GetData());
    (void)bpm.UnpinPage(pid, true);
}

// Child i of an internal page.
page_id_t ChildOf(BufferPoolManager& bpm, page_id_t pid, uint16_t i) {
    const page_id_t child = BTreeInternalPage::GetChild(bpm.FetchPage(pid)->GetData(), i, TypeId::BIGINT);
    (void)bpm.UnpinPage(pid, false);
    return child;
}

// Expects every frame of the pool to be unpinned.
void ExpectNoPins(const BufferPoolManager& bpm, const std::string& what) {
    for (size_t f = 0; f < bpm.GetPoolSize(); ++f) {
        ASSERT_EQ(bpm.GetPage(static_cast<frame_id_t>(f)).GetPinCount(), 0) << what << ": frame " << f << " left pinned";
    }
}

// Expects a point lookup, a range scan and a full scan to throw this text.
void ExpectDamaged(BTreeIndex& index, const BufferPoolManager& bpm, int64_t key, const std::string& text) {
    const std::vector<std::pair<std::string, std::function<void()>>> ops = {
        {"point lookup", [&] { (void)index.PointLookup(Value(key)); }},
        {"range scan", [&] { for (auto it = index.Begin(Value(key)); !it.IsEnd(); ++it) (void)*it; }},
        {"full scan", [&] { for (auto it = index.Begin(); !it.IsEnd(); ++it) (void)*it; }},
    };
    for (const auto& [name, op] : ops) {
        std::string error = "no throw";
        try {
            op();
        } catch (const std::exception& e) {
            error = e.what();
        }
        EXPECT_EQ(error, text) << name;
        ExpectNoPins(bpm, name);
    }
}

}  // namespace

TEST_F(BTreeIndexTest, DamagedInternalKeyCountThrows) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::BIGINT);
    ASSERT_NO_FATAL_FAILURE(FillTwoLevels(index, *bundle.bpm));
    const page_id_t root = index.GetRootPageId();

    Damage(*bundle.bpm, root, [](char* data) { BTreeInternalPage::SetNumKeys(data, 0); });
    ExpectDamaged(index, *bundle.bpm, 10, "damaged index: internal page has 0 keys");

    const auto too_many = static_cast<uint16_t>(BTreeInternalPage::MaxKeys(TypeId::BIGINT) + 2);
    Damage(*bundle.bpm, root, [&](char* data) { BTreeInternalPage::SetNumKeys(data, too_many); });
    ExpectDamaged(index, *bundle.bpm, 10,
                  "damaged index: internal page has " + std::to_string(too_many) + " keys, more than " +
                      std::to_string(too_many - 1) + " fit");
}

TEST_F(BTreeIndexTest, DamagedLeafKeyCountThrows) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::BIGINT);
    ASSERT_NO_FATAL_FAILURE(FillTwoLevels(index, *bundle.bpm));
    const page_id_t leaf = ChildOf(*bundle.bpm, index.GetRootPageId(), 0);

    const uint16_t max = BTreeLeafPage::MaxKeys(TypeId::BIGINT);
    Damage(*bundle.bpm, leaf, [&](char* data) { BTreeLeafPage::SetNumKeys(data, static_cast<uint16_t>(max + 1)); });
    ExpectDamaged(index, *bundle.bpm, 10,
                  "damaged index: leaf page has " + std::to_string(max + 1) + " keys, more than " + std::to_string(max) +
                      " fit");
}

TEST_F(BTreeIndexTest, DamagedChildPointerThrows) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::BIGINT);
    ASSERT_NO_FATAL_FAILURE(FillTwoLevels(index, *bundle.bpm));
    const page_id_t root = index.GetRootPageId();
    const std::string at = "damaged index: page " + std::to_string(root) + " has a child pointer ";

    Damage(*bundle.bpm, root, [](char* data) { BTreeInternalPage::SetChild(data, 0, INVALID_PAGE_ID, TypeId::BIGINT); });
    ExpectDamaged(index, *bundle.bpm, 10, at + "that is INVALID_PAGE_ID");

    const page_id_t pages = bundle.bpm->GetNumPages();
    Damage(*bundle.bpm, root, [&](char* data) { BTreeInternalPage::SetChild(data, 0, pages, TypeId::BIGINT); });
    ExpectDamaged(index, *bundle.bpm, 10,
                  at + "to page " + std::to_string(pages) + ", past the file's " + std::to_string(pages) + " pages");
}

TEST_F(BTreeIndexTest, LeafCycleThrows) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::BIGINT);
    ASSERT_NO_FATAL_FAILURE(FillTwoLevels(index, *bundle.bpm));
    const page_id_t first = ChildOf(*bundle.bpm, index.GetRootPageId(), 0);
    const page_id_t second = ChildOf(*bundle.bpm, index.GetRootPageId(), 1);

    // The second leaf points back at the first; the walk would never end
    Damage(*bundle.bpm, second, [&](char* data) { BTreeLeafPage::SetNextLeafPageId(data, first); });
    const std::string text =
        "damaged index: leaf chain longer than the file's " + std::to_string(bundle.bpm->GetNumPages()) + " pages";
    std::string error = "no throw";
    size_t rows = 0;
    try {
        for (auto it = index.Begin(); !it.IsEnd() && rows <= 10 * static_cast<size_t>(kDamageKeys); ++it) ++rows;
    } catch (const std::exception& e) {
        error = e.what();
    }
    EXPECT_EQ(error, text) << "after " << rows << " rows";
    ExpectNoPins(*bundle.bpm, "leaf cycle");
}

TEST_F(BTreeIndexTest, LeafPointerPastTheEndThrows) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::BIGINT);
    ASSERT_NO_FATAL_FAILURE(FillTwoLevels(index, *bundle.bpm));
    const page_id_t first = ChildOf(*bundle.bpm, index.GetRootPageId(), 0);
    const page_id_t pages = bundle.bpm->GetNumPages();

    Damage(*bundle.bpm, first, [&](char* data) { BTreeLeafPage::SetNextLeafPageId(data, pages); });
    std::string error = "no throw";
    try {
        for (auto it = index.Begin(); !it.IsEnd(); ++it) (void)*it;
    } catch (const std::exception& e) {
        error = e.what();
    }
    EXPECT_EQ(error, "damaged index: leaf chain goes to page " + std::to_string(pages) + ", past the file's " +
                         std::to_string(pages) + " pages");
    ExpectNoPins(*bundle.bpm, "leaf pointer past the end");
}

// ---------------------------------------------------------------------------
// Split keys: cutting a key range into pieces of about equal leaves
// ---------------------------------------------------------------------------
namespace {

// A BIGINT bound, or Value() for an open end.
Value Bound(const std::optional<int64_t>& key) {
    return key ? Value(*key) : Value();
}

// Levels from the root down child 0 to a leaf, the leaf included.
int Levels(const BTreeIndex& index, BufferPoolManager& bpm) {
    int levels = 1;
    for (page_id_t pid = index.GetRootPageId();; ++levels) {
        const char* data = bpm.FetchPage(pid)->GetData();
        const bool leaf = (SlottedPage::GetHeader(data)->flags & 1u) != 0;
        const page_id_t child = leaf ? INVALID_PAGE_ID : BTreeInternalPage::GetChild(data, 0, TypeId::BIGINT);
        (void)bpm.UnpinPage(pid, false);
        if (leaf) return levels;
        pid = child;
    }
}

// Each leaf's first key, left to right along the leaf chain.
std::vector<int64_t> LeafFirstKeys(const BTreeIndex& index, BufferPoolManager& bpm) {
    page_id_t pid = index.GetRootPageId();
    for (;;) {
        const char* data = bpm.FetchPage(pid)->GetData();
        const bool leaf = (SlottedPage::GetHeader(data)->flags & 1u) != 0;
        const page_id_t child = leaf ? pid : BTreeInternalPage::GetChild(data, 0, TypeId::BIGINT);
        (void)bpm.UnpinPage(pid, false);
        if (leaf) break;
        pid = child;
    }
    std::vector<int64_t> firsts;
    while (pid != INVALID_PAGE_ID) {
        const char* data = bpm.FetchPage(pid)->GetData();
        if (BTreeLeafPage::GetNumKeys(data) > 0) firsts.push_back(BTreeLeafPage::GetKey(data, 0, TypeId::BIGINT).bigint_);
        const page_id_t next = BTreeLeafPage::GetNextLeafPageId(data);
        (void)bpm.UnpinPage(pid, false);
        pid = next;
    }
    return firsts;
}

// Expects keys sorted, distinct, strictly between the bounds, and at most pieces - 1.
void ExpectSplitKeys(const std::vector<Value>& keys, const std::optional<int64_t>& low, const std::optional<int64_t>& high, size_t pieces) {
    EXPECT_LE(keys.size() + 1, std::max<size_t>(pieces, 1));
    for (size_t i = 0; i < keys.size(); ++i) {
        ASSERT_EQ(keys[i].type_, TypeId::BIGINT);
        if (low) EXPECT_GT(keys[i].bigint_, *low) << "key " << i;
        if (high) EXPECT_LT(keys[i].bigint_, *high) << "key " << i;
        if (i > 0) EXPECT_LT(keys[i - 1].bigint_, keys[i].bigint_) << "key " << i;
    }
}

using Entry = std::pair<int64_t, RID>;

// The entries of [low, high] in key order, as a range scan reads them.
std::vector<Entry> ScanRange(BTreeIndex& index, const std::optional<int64_t>& low, const std::optional<int64_t>& high) {
    std::vector<Entry> out;
    for (auto it = low ? index.Begin(Value(*low)) : index.Begin(); !it.IsEnd(); ++it) {
        const auto [key, rid] = *it;
        if (high && key.bigint_ > *high) break;
        out.emplace_back(key.bigint_, rid);
    }
    return out;
}

// Piece i walks from Begin(keys[i - 1]) up to the first key at or past keys[i]; the first
// starts at low, the last stops past high. Returns every piece's entries, in piece order.
std::vector<Entry> ScanPieces(BTreeIndex& index, const std::optional<int64_t>& low, const std::optional<int64_t>& high, const std::vector<Value>& keys) {
    std::vector<Entry> out;
    for (size_t i = 0; i <= keys.size(); ++i) {
        const std::optional<int64_t> from = i == 0 ? low : std::optional<int64_t>(keys[i - 1].bigint_);
        for (auto it = from ? index.Begin(Value(*from)) : index.Begin(); !it.IsEnd(); ++it) {
            const auto [key, rid] = *it;
            if (i < keys.size() ? key.bigint_ >= keys[i].bigint_ : (high && key.bigint_ > *high)) break;
            out.emplace_back(key.bigint_, rid);
        }
    }
    return out;
}

}  // namespace

TEST_F(BTreeIndexTest, SplitKeysCutTheRangeIntoEqualPieces) {
    // Sorted keys leave every leaf about half full: three levels, about 860 leaves.
    constexpr int64_t kKeys = 250000;
    auto bundle = MakeBPM(4096);
    BTreeIndex index(bundle.bpm.get(), TypeId::BIGINT);
    for (int64_t k = 0; k < kKeys; ++k) index.Insert(Value(k), RID(static_cast<page_id_t>(k), 0));
    ASSERT_GE(Levels(index, *bundle.bpm), 3);
    const std::vector<int64_t> firsts = LeafFirstKeys(index, *bundle.bpm);
    ASSERT_GT(firsts.size(), 800u);

    // The whole tree: every key starts a leaf, and pieces differ by at most one leaf.
    for (const size_t pieces : {size_t{2}, size_t{4}, size_t{8}, size_t{18}, size_t{256}}) {
        SCOPED_TRACE(std::to_string(pieces) + " pieces");
        const std::vector<Value> keys = index.SplitKeys(Value(), Value(), pieces);
        ASSERT_EQ(keys.size(), pieces - 1);
        ExpectSplitKeys(keys, std::nullopt, std::nullopt, pieces);
        std::vector<size_t> starts = {0};
        for (const Value& key : keys) {
            const auto at = std::lower_bound(firsts.begin(), firsts.end(), key.bigint_);
            ASSERT_TRUE(at != firsts.end() && *at == key.bigint_) << key.bigint_ << " starts no leaf";
            starts.push_back(static_cast<size_t>(at - firsts.begin()));
        }
        starts.push_back(firsts.size());
        size_t fewest = firsts.size();
        size_t most = 0;
        for (size_t i = 0; i + 1 < starts.size(); ++i) {
            fewest = std::min(fewest, starts[i + 1] - starts[i]);
            most = std::max(most, starts[i + 1] - starts[i]);
        }
        EXPECT_LE(most - fewest, 1u) << "leaves per piece from " << fewest << " to " << most;
    }

    // Bounded and half-open ranges: inside the bounds. Asked for more pieces than leaves,
    // every leaf start strictly inside comes back.
    struct Range {
        std::optional<int64_t> low;
        std::optional<int64_t> high;
    };
    for (const Range& r : {Range{1000, 200000}, Range{123457, 128457}, Range{std::nullopt, 50000}, Range{150000, std::nullopt}}) {
        for (const size_t pieces : {size_t{2}, size_t{4}, size_t{18}, size_t{100000}}) {
            SCOPED_TRACE(std::to_string(r.low.value_or(-1)) + " .. " + std::to_string(r.high.value_or(-1)) + ", " + std::to_string(pieces) + " pieces");
            const std::vector<Value> keys = index.SplitKeys(Bound(r.low), Bound(r.high), pieces);
            ExpectSplitKeys(keys, r.low, r.high, pieces);
            size_t inside = 0;  // the first leaf's first key is no separator
            for (size_t i = 1; i < firsts.size(); ++i) inside += (!r.low || firsts[i] > *r.low) && (!r.high || firsts[i] < *r.high) ? 1 : 0;
            EXPECT_EQ(keys.size(), std::min(inside, pieces - 1));
            EXPECT_EQ(ScanPieces(index, r.low, r.high, keys), ScanRange(index, r.low, r.high));
        }
    }

    // Nothing to cut: one piece, a range inside one leaf, bounds on two leaf starts,
    // an empty range, and ranges past either end.
    EXPECT_TRUE(index.SplitKeys(Value(), Value(), 1).empty());
    EXPECT_TRUE(index.SplitKeys(Value(), Value(), 0).empty());
    const std::vector<Range> nothing = {Range{firsts[10] + 1, firsts[11] - 1}, Range{firsts[10], firsts[11]}, Range{5000, 4000},
                                        Range{kKeys + 10, std::nullopt}, Range{std::nullopt, -5}};
    for (const Range& r : nothing) {
        EXPECT_TRUE(index.SplitKeys(Bound(r.low), Bound(r.high), 8).empty()) << r.low.value_or(-1) << " .. " << r.high.value_or(-1);
    }
    ExpectNoPins(*bundle.bpm, "split keys");

    // A bad child pointer off the leftmost path is found on the way down to the cuts.
    const page_id_t root = index.GetRootPageId();
    uint16_t last = 0;
    {
        const char* data = bundle.bpm->FetchPage(root)->GetData();
        last = BTreeInternalPage::GetNumKeys(data);
        (void)bundle.bpm->UnpinPage(root, false);
    }
    ASSERT_GT(last, 0);
    Damage(*bundle.bpm, root, [&](char* data) { BTreeInternalPage::SetChild(data, last, INVALID_PAGE_ID, TypeId::BIGINT); });
    std::string error = "no throw";
    try {
        (void)index.SplitKeys(Value(), Value(), 8);
    } catch (const std::exception& e) {
        error = e.what();
    }
    EXPECT_EQ(error, "damaged index: page " + std::to_string(root) + " has a child pointer that is INVALID_PAGE_ID");
    ExpectNoPins(*bundle.bpm, "damaged split keys");
}

TEST_F(BTreeIndexTest, SplitKeysWithCopiesAcrossLeaves) {
    // 600 keys with 400 copies each, inserted in order: each key's copies cross a leaf split.
    constexpr int64_t kKeys = 600;
    constexpr int64_t kCopies = 400;
    ASSERT_GT(kCopies, BTreeLeafPage::MaxKeys(TypeId::BIGINT) / 2);
    auto bundle = MakeBPM(4096);
    BTreeIndex index(bundle.bpm.get(), TypeId::BIGINT);
    for (int64_t k = 0; k < kKeys; ++k) {
        for (int64_t c = 0; c < kCopies; ++c) index.Insert(Value(k), RID(static_cast<page_id_t>(k), static_cast<uint16_t>(c)));
    }
    ASSERT_GE(Levels(index, *bundle.bpm), 3);

    // Every entry lands in exactly one piece, in key order, so pieces read like the whole range.
    struct Range {
        std::optional<int64_t> low;
        std::optional<int64_t> high;
    };
    for (const Range& r : {Range{std::nullopt, std::nullopt}, Range{100, 200}, Range{0, kKeys - 1}, Range{std::nullopt, 300}, Range{450, std::nullopt},
                           Range{100, 102}}) {
        for (const size_t pieces : {size_t{2}, size_t{4}, size_t{8}, size_t{18}}) {
            SCOPED_TRACE(std::to_string(r.low.value_or(-1)) + " .. " + std::to_string(r.high.value_or(-1)) + ", " + std::to_string(pieces) + " pieces");
            const std::vector<Value> keys = index.SplitKeys(Bound(r.low), Bound(r.high), pieces);
            ExpectSplitKeys(keys, r.low, r.high, pieces);
            EXPECT_FALSE(keys.empty());
            const std::vector<Entry> want = ScanRange(index, r.low, r.high);
            EXPECT_EQ(want.size(), static_cast<size_t>((r.high.value_or(kKeys - 1) - r.low.value_or(0) + 1) * kCopies));
            EXPECT_EQ(ScanPieces(index, r.low, r.high, keys), want);
        }
    }

    // The whole tree has enough distinct separators for every cut.
    EXPECT_EQ(index.SplitKeys(Value(), Value(), 8).size(), 7u);

    // One key's copies span several leaves, but no separator sits strictly inside.
    EXPECT_TRUE(index.SplitKeys(Value(int64_t{300}), Value(int64_t{300}), 8).empty());
    ExpectNoPins(*bundle.bpm, "split keys with copies");
}

TEST_F(BTreeIndexTest, SplitKeysOnSmallAndDamagedTrees) {
    auto bundle = MakeBPM();
    {
        BTreeIndex index(bundle.bpm.get(), TypeId::BIGINT);
        EXPECT_TRUE(index.SplitKeys(Value(), Value(), 4).empty()) << "empty tree";
        for (int64_t k = 0; k < 100; ++k) index.Insert(Value(k), RID(static_cast<page_id_t>(k), 0));
        EXPECT_TRUE(index.SplitKeys(Value(), Value(), 4).empty()) << "one leaf";
    }

    // Two levels: the root is the level above the leaves, so its separators are the cuts.
    BTreeIndex index(bundle.bpm.get(), TypeId::BIGINT);
    ASSERT_NO_FATAL_FAILURE(FillTwoLevels(index, *bundle.bpm));
    const page_id_t root = index.GetRootPageId();
    std::vector<int64_t> separators;
    {
        const char* data = bundle.bpm->FetchPage(root)->GetData();
        for (uint16_t i = 0; i < BTreeInternalPage::GetNumKeys(data); ++i) separators.push_back(BTreeInternalPage::GetKey(data, i, TypeId::BIGINT).bigint_);
        (void)bundle.bpm->UnpinPage(root, false);
    }
    const std::vector<Value> keys = index.SplitKeys(Value(), Value(), 4);
    ASSERT_EQ(keys.size(), 3u);
    for (const Value& key : keys) EXPECT_NE(std::find(separators.begin(), separators.end(), key.bigint_), separators.end()) << key.bigint_;
    ExpectNoPins(*bundle.bpm, "two levels");

    // Damage is refused with the walk's own text, nothing left pinned.
    auto expect_refused = [&](const std::string& text) {
        std::string error = "no throw";
        try {
            (void)index.SplitKeys(Value(), Value(), 4);
        } catch (const std::exception& e) {
            error = e.what();
        }
        EXPECT_EQ(error, text);
        ExpectNoPins(*bundle.bpm, text);
    };
    Damage(*bundle.bpm, root, [](char* data) { BTreeInternalPage::SetChild(data, 0, INVALID_PAGE_ID, TypeId::BIGINT); });
    expect_refused("damaged index: page " + std::to_string(root) + " has a child pointer that is INVALID_PAGE_ID");
    Damage(*bundle.bpm, root, [](char* data) { BTreeInternalPage::SetNumKeys(data, 0); });
    expect_refused("damaged index: internal page has 0 keys");
}

TEST_F(BTreeIndexTest, ForEachEntryReadsWhatTheIteratorReads) {
    // Copies of 500 cross leaves, so a start on 500 must find its first copy.
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::INTEGER);
    ASSERT_NO_FATAL_FAILURE(FillWithManyCopies(index));
    using Pair = std::pair<int32_t, RID>;

    // Up to `limit` entries from a start, by the iterator and by ForEachEntry.
    const auto by_iterator = [&](const Value& from, size_t limit) {
        std::vector<Pair> out;
        for (auto it = from.IsNull() ? index.Begin() : index.Begin(from); !it.IsEnd() && out.size() < limit; ++it) {
            const auto [key, rid] = *it;
            out.emplace_back(key.integer_, rid);
        }
        return out;
    };
    const auto by_leaves = [&](const Value& from, size_t limit) {
        std::vector<Pair> out;
        index.ForEachEntry(from, [&](const Value& key, const RID& rid) {
            if (out.size() >= limit) return false;
            out.emplace_back(key.integer_, rid);
            return true;
        });
        return out;
    };
    for (const Value& from : {Value(), Value(int32_t(0)), Value(int32_t(1)), Value(int32_t(450)), Value(int32_t(kManyKey)), Value(int32_t(501)),
                              Value(int32_t(750)), Value(int32_t(900)), Value(int32_t(901))}) {
        for (const size_t limit : {size_t{0}, size_t{1}, size_t{2000}, size_t{1000000}}) {
            SCOPED_TRACE((from.IsNull() ? std::string("start") : std::to_string(from.integer_)) + ", up to " + std::to_string(limit));
            EXPECT_EQ(by_leaves(from, limit), by_iterator(from, limit));
        }
    }
    EXPECT_EQ(by_leaves(Value(), 1000000).size(), static_cast<size_t>(300 + kManyCopies + 301));
    ExpectNoPins(*bundle.bpm, "for each entry");

    // An empty tree has nothing; a throw from fn unpins its leaf.
    BTreeIndex empty(bundle.bpm.get(), TypeId::INTEGER);
    empty.ForEachEntry(Value(), [](const Value&, const RID&) {
        ADD_FAILURE() << "an entry in an empty tree";
        return true;
    });
    EXPECT_THROW(index.ForEachEntry(Value(int32_t(600)), [](const Value&, const RID&) -> bool { throw DatabaseException("stop"); }), DatabaseException);
    ExpectNoPins(*bundle.bpm, "a throwing fn");
}

TEST_F(BTreeIndexTest, ForEachEntryRefusesADamagedChain) {
    auto bundle = MakeBPM();
    BTreeIndex index(bundle.bpm.get(), TypeId::BIGINT);
    ASSERT_NO_FATAL_FAILURE(FillTwoLevels(index, *bundle.bpm));
    const page_id_t first = ChildOf(*bundle.bpm, index.GetRootPageId(), 0);
    const page_id_t second = ChildOf(*bundle.bpm, index.GetRootPageId(), 1);
    const auto error_of = [&](const Value& from) {
        std::string error = "no throw";
        try {
            index.ForEachEntry(from, [](const Value&, const RID&) { return true; });
        } catch (const std::exception& e) {
            error = e.what();
        }
        return error;
    };

    // The same texts the iterator throws, nothing left pinned.
    Damage(*bundle.bpm, second, [&](char* data) { BTreeLeafPage::SetNextLeafPageId(data, first); });
    const std::string cycle = "damaged index: leaf chain longer than the file's " + std::to_string(bundle.bpm->GetNumPages()) + " pages";
    EXPECT_EQ(error_of(Value()), cycle);
    EXPECT_EQ(error_of(Value(int64_t{10})), cycle);
    ExpectNoPins(*bundle.bpm, "leaf cycle");

    const page_id_t pages = bundle.bpm->GetNumPages();
    Damage(*bundle.bpm, first, [&](char* data) { BTreeLeafPage::SetNextLeafPageId(data, pages); });
    EXPECT_EQ(error_of(Value()), "damaged index: leaf chain goes to page " + std::to_string(pages) + ", past the file's " + std::to_string(pages) + " pages");
    ExpectNoPins(*bundle.bpm, "leaf pointer past the end");

    const uint16_t max = BTreeLeafPage::MaxKeys(TypeId::BIGINT);
    Damage(*bundle.bpm, first, [&](char* data) { BTreeLeafPage::SetNumKeys(data, static_cast<uint16_t>(max + 1)); });
    EXPECT_EQ(error_of(Value(int64_t{10})), "damaged index: leaf page has " + std::to_string(max + 1) + " keys, more than " + std::to_string(max) + " fit");
    ExpectNoPins(*bundle.bpm, "leaf key count");
}

TEST_F(BTreeIndexTest, ForEachEntryOverLeavesEmptiedByDeletes) {
    // Shuffled inserts, some keys with 1,000 copies, then deletes that empty whole leaves
    // at the start, in the middle and at the end of the chain.
    auto bundle = MakeBPM(2048);
    BTreeIndex index(bundle.bpm.get(), TypeId::BIGINT);
    std::mt19937 rng(7);
    std::vector<Entry> all;
    for (int64_t k = 0; k < 1500; ++k) {
        const int copies = k % 97 == 0 ? 1000 : (k % 13 == 0 ? 20 : 2);
        for (int c = 0; c < copies; ++c) all.emplace_back(k, RID(static_cast<page_id_t>(k), static_cast<uint16_t>(c)));
    }
    std::shuffle(all.begin(), all.end(), rng);
    for (const Entry& e : all) index.Insert(Value(e.first), e.second);
    const auto deleted = [](int64_t k) { return k < 40 || (k >= 500 && k < 800) || k >= 1450; };
    for (const Entry& e : all) {
        if (deleted(e.first)) index.Delete(Value(e.first), e.second);
    }

    // Leaves left with no key, counted along the chain.
    size_t empty = 0;
    {
        page_id_t pid = index.GetRootPageId();
        for (bool leaf = false; !leaf;) {
            const char* data = bundle.bpm->FetchPage(pid)->GetData();
            leaf = (SlottedPage::GetHeader(data)->flags & 1u) != 0;
            const page_id_t child = leaf ? pid : BTreeInternalPage::GetChild(data, 0, TypeId::BIGINT);
            (void)bundle.bpm->UnpinPage(pid, false);
            pid = child;
        }
        while (pid != INVALID_PAGE_ID) {
            const char* data = bundle.bpm->FetchPage(pid)->GetData();
            empty += BTreeLeafPage::GetNumKeys(data) == 0 ? 1 : 0;
            const page_id_t next = BTreeLeafPage::GetNextLeafPageId(data);
            (void)bundle.bpm->UnpinPage(pid, false);
            pid = next;
        }
    }
    ASSERT_GE(empty, 3u) << "the deletes should empty whole leaves";
    const size_t kept = static_cast<size_t>(std::count_if(all.begin(), all.end(), [&](const Entry& e) { return !deleted(e.first); }));
    ASSERT_EQ(ScanRange(index, std::nullopt, std::nullopt).size(), kept);

    // Random ranges: SplitKeys' pieces read by ForEachEntry give the iterator's range scan,
    // and ForEachEntry from each start gives the iterator's walk from there.
    std::uniform_int_distribution<int64_t> key(-10, 1510);
    for (int trial = 0; trial < 300; ++trial) {
        std::optional<int64_t> low;
        std::optional<int64_t> high;
        if (rng() % 5 != 0) low = key(rng);
        if (rng() % 5 != 0) high = key(rng);
        if (rng() % 7 == 0 && low) high = low;
        const size_t pieces = 1 + rng() % 20;
        SCOPED_TRACE(std::to_string(low.value_or(-999)) + " .. " + std::to_string(high.value_or(-999)) + ", " + std::to_string(pieces) + " pieces");

        const std::vector<Value> cuts = index.SplitKeys(Bound(low), Bound(high), pieces);
        ExpectSplitKeys(cuts, low, high, pieces);
        std::vector<Entry> got;
        for (size_t i = 0; i <= cuts.size(); ++i) {
            const Value* until = i < cuts.size() ? &cuts[i] : nullptr;
            index.ForEachEntry(i == 0 ? Bound(low) : cuts[i - 1], [&](const Value& k, const RID& rid) {
                if (until != nullptr ? k >= *until : high && k.bigint_ > *high) return false;
                got.emplace_back(k.bigint_, rid);
                return true;
            });
        }
        EXPECT_EQ(got, ScanRange(index, low, high));

        std::vector<Entry> from;
        index.ForEachEntry(Bound(low), [&](const Value& k, const RID& rid) {
            from.emplace_back(k.bigint_, rid);
            return true;
        });
        EXPECT_EQ(from, ScanRange(index, low, std::nullopt));
    }
    ExpectNoPins(*bundle.bpm, "for each entry after deletes");
}

}  // namespace shilmandb
