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

}  // namespace shilmandb
