#include <gtest/gtest.h>
#include <string>
#include <unistd.h>
#include "storage/table_heap.hpp"
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "catalog/schema.hpp"
#include "types/value.hpp"
#include <filesystem>
#include <memory>
#include <set>
#include <vector>

namespace shilmandb {

class TableHeapTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() /
                      ("shilmandb_table_heap_test_" + std::to_string(::getpid()) + ".db")).string();
        std::filesystem::remove(test_file_);
    }

    void TearDown() override {
        std::filesystem::remove(test_file_);
    }

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

    // Schema: (id INTEGER, name VARCHAR)
    Schema MakeSchema() {
        return Schema({Column("id", TypeId::INTEGER), Column("name", TypeId::VARCHAR)});
    }

    // Build a tuple: (id, "row_<id>")
    Tuple MakeTuple(int32_t id, const Schema& schema) {
        return Tuple(
            {Value(id), Value(std::string("row_") + std::to_string(id))},
            schema);
    }
};

// ---------------------------------------------------------------------------
// Test 1: InsertAndScan
// ---------------------------------------------------------------------------
TEST_F(TableHeapTest, InsertAndScan) {
    auto [dm, bpm] = MakeBPM(64);
    auto schema = MakeSchema();
    TableHeap heap(bpm.get());

    // Insert 100 tuples
    for (int32_t i = 0; i < 100; ++i) {
        auto rid = heap.InsertTuple(MakeTuple(i, schema));
        ASSERT_NE(rid.page_id, INVALID_PAGE_ID) << "Insert failed at i=" << i;
    }

    // Scan and collect
    std::vector<Tuple> results;
    for (auto it = heap.Begin(schema); it != heap.End(); ++it) {
        results.push_back(*it);
    }

    EXPECT_EQ(results.size(), 100u);

    // Verify values (tuples come back in insertion order for a heap)
    for (int32_t i = 0; i < 100; ++i) {
        auto id_val = results[i].GetValue(schema, 0);
        auto name_val = results[i].GetValue(schema, 1);
        EXPECT_EQ(id_val, Value(i));
        EXPECT_EQ(name_val, Value(std::string("row_") + std::to_string(i)));
    }
}

// ---------------------------------------------------------------------------
// Test 2: InsertSpansMultiplePages
// ---------------------------------------------------------------------------
TEST_F(TableHeapTest, InsertSpansMultiplePages) {
    auto [dm, bpm] = MakeBPM(128);
    auto schema = MakeSchema();
    TableHeap heap(bpm.get());

    constexpr int32_t count = 10000;
    std::set<page_id_t> pages_used;
    for (int32_t i = 0; i < count; ++i) {
        auto rid = heap.InsertTuple(MakeTuple(i, schema));
        ASSERT_NE(rid.page_id, INVALID_PAGE_ID) << "Insert failed at i=" << i;
        pages_used.insert(rid.page_id);
    }

    // 10000 tuples cannot fit on a single 8KB page — multiple pages required
    EXPECT_GT(pages_used.size(), 1u);

    // Full scan
    int32_t scanned = 0;
    for (auto it = heap.Begin(schema); it != heap.End(); ++it) {
        ++scanned;
    }
    EXPECT_EQ(scanned, count);
}

// ---------------------------------------------------------------------------
// Test 3: DeleteAndScan
// ---------------------------------------------------------------------------
TEST_F(TableHeapTest, DeleteAndScan) {
    auto [dm, bpm] = MakeBPM(64);
    auto schema = MakeSchema();
    TableHeap heap(bpm.get());

    // Insert 100 tuples, save RIDs
    std::vector<RID> rids;
    for (int32_t i = 0; i < 100; ++i) {
        rids.push_back(heap.InsertTuple(MakeTuple(i, schema)));
    }

    // Delete even-indexed
    for (int32_t i = 0; i < 100; i += 2) {
        EXPECT_TRUE(heap.DeleteTuple(rids[i]));
    }

    // Scan — should see only odd-indexed tuples
    std::vector<Tuple> results;
    for (auto it = heap.Begin(schema); it != heap.End(); ++it) {
        results.push_back(*it);
    }

    EXPECT_EQ(results.size(), 50u);

    // Verify all remaining tuples have odd ids
    for (const auto& t : results) {
        auto id_val = t.GetValue(schema, 0);
        EXPECT_EQ(id_val.integer_ % 2, 1) << "Even id found after delete: " << id_val.integer_;
    }
}

// ---------------------------------------------------------------------------
// Test 4: GetTupleByRID
// ---------------------------------------------------------------------------
TEST_F(TableHeapTest, GetTupleByRID) {
    auto [dm, bpm] = MakeBPM(64);
    auto schema = MakeSchema();
    TableHeap heap(bpm.get());

    auto rid = heap.InsertTuple(MakeTuple(42, schema));
    ASSERT_NE(rid.page_id, INVALID_PAGE_ID);

    Tuple out;
    ASSERT_TRUE(heap.GetTuple(rid, &out, schema));

    EXPECT_EQ(out.GetValue(schema, 0), Value(static_cast<int32_t>(42)));
    EXPECT_EQ(out.GetValue(schema, 1), Value(std::string("row_42")));
}

// ---------------------------------------------------------------------------
// Test 5: NoPinnedPagesAfterScan
// ---------------------------------------------------------------------------
TEST_F(TableHeapTest, NoPinnedPagesAfterScan) {
    auto [dm, bpm] = MakeBPM(64);
    auto schema = MakeSchema();
    TableHeap heap(bpm.get());

    for (int32_t i = 0; i < 200; ++i) {
        auto rid = heap.InsertTuple(MakeTuple(i, schema));
        ASSERT_NE(rid.page_id, INVALID_PAGE_ID);
    }

    // Complete a full scan
    for (auto it = heap.Begin(schema); it != heap.End(); ++it) {
        (void)*it;
    }

    // Verify no pages are pinned
    for (size_t i = 0; i < bpm->GetPoolSize(); ++i) {
        EXPECT_EQ(bpm->GetPage(static_cast<frame_id_t>(i)).GetPinCount(), 0)
            << "Frame " << i << " still pinned after full scan";
    }
}

}  // namespace shilmandb
