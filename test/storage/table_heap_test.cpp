#include <gtest/gtest.h>
#include <cstring>
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

// ---------------------------------------------------------------------------
// ReadRows: page-at-a-time reads
// ---------------------------------------------------------------------------
namespace {

// Reads every row with the given batch size, collecting each row as a Tuple.
size_t ReadAllRows(TableHeap& heap, RID* pos, size_t max_rows, const Schema& schema,
                   std::vector<Tuple>* out) {
    size_t total = 0;
    while (true) {
        size_t n = heap.ReadRows(pos, max_rows, [&](const char* data, uint32_t length) {
            Tuple t;
            t.DeserializeFrom(data, length, schema);
            out->push_back(std::move(t));
        });
        if (n == 0) { break; }
        total += n;
    }
    return total;
}

}  // namespace

TEST_F(TableHeapTest, ReadRowsMatchesIterator) {
    auto [dm, bpm] = MakeBPM(64);
    auto schema = MakeSchema();
    TableHeap heap(bpm.get());
    for (int32_t i = 0; i < 1000; ++i) {
        ASSERT_NE(heap.InsertTuple(MakeTuple(i, schema)).page_id, INVALID_PAGE_ID);
    }

    RID pos{heap.GetFirstPageId(), 0};
    std::vector<Tuple> rows;
    EXPECT_EQ(ReadAllRows(heap, &pos, 250, schema, &rows), 1000u);
    EXPECT_EQ(pos.page_id, INVALID_PAGE_ID);

    size_t i = 0;
    for (auto it = heap.Begin(schema); it != heap.End(); ++it, ++i) {
        Tuple expected = *it;
        ASSERT_LT(i, rows.size());
        ASSERT_EQ(rows[i].GetLength(), expected.GetLength()) << "row " << i;
        EXPECT_EQ(std::memcmp(rows[i].GetData(), expected.GetData(), expected.GetLength()), 0)
            << "row " << i;
    }
    EXPECT_EQ(i, rows.size());
}

TEST_F(TableHeapTest, ReadRowsStopsMidPage) {
    auto [dm, bpm] = MakeBPM(64);
    auto schema = MakeSchema();
    TableHeap heap(bpm.get());
    for (int32_t i = 0; i < 1000; ++i) {
        ASSERT_NE(heap.InsertTuple(MakeTuple(i, schema)).page_id, INVALID_PAGE_ID);
    }

    RID pos{heap.GetFirstPageId(), 0};
    std::vector<Tuple> rows;
    auto collect = [&](const char* data, uint32_t length) {
        Tuple t;
        t.DeserializeFrom(data, length, schema);
        rows.push_back(std::move(t));
    };
    EXPECT_EQ(heap.ReadRows(&pos, 7, collect), 7u);
    EXPECT_EQ(pos.page_id, heap.GetFirstPageId());
    EXPECT_EQ(pos.slot_id, 7);

    rows.clear();
    EXPECT_EQ(heap.ReadRows(&pos, 1, collect), 1u);
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].GetValue(schema, 0).integer_, 7);
}

TEST_F(TableHeapTest, ReadRowsSkipsDeleted) {
    auto [dm, bpm] = MakeBPM(64);
    auto schema = MakeSchema();
    TableHeap heap(bpm.get());
    std::vector<RID> rids;
    for (int32_t i = 0; i < 100; ++i) {
        rids.push_back(heap.InsertTuple(MakeTuple(i, schema)));
    }
    for (int32_t i = 0; i < 100; i += 2) {
        ASSERT_TRUE(heap.DeleteTuple(rids[i]));
    }

    RID pos{heap.GetFirstPageId(), 0};
    std::vector<Tuple> rows;
    EXPECT_EQ(ReadAllRows(heap, &pos, 100000, schema, &rows), 50u);
    ASSERT_EQ(rows.size(), 50u);
    for (size_t i = 0; i < rows.size(); ++i) {
        EXPECT_EQ(rows[i].GetValue(schema, 0).integer_, static_cast<int32_t>(2 * i + 1));
    }
}

TEST_F(TableHeapTest, ReadRowsEmptyHeap) {
    auto [dm, bpm] = MakeBPM(64);
    TableHeap heap(bpm.get());

    RID pos{heap.GetFirstPageId(), 0};
    size_t calls = 0;
    EXPECT_EQ(heap.ReadRows(&pos, 10, [&](const char*, uint32_t) { ++calls; }), 0u);
    EXPECT_EQ(calls, 0u);
    EXPECT_EQ(pos.page_id, INVALID_PAGE_ID);
}

TEST_F(TableHeapTest, ReadRowsOneFetchPerPage) {
    auto [dm, bpm] = MakeBPM(64);
    auto schema = MakeSchema();
    TableHeap heap(bpm.get());
    for (int32_t i = 0; i < 1000; ++i) {
        ASSERT_NE(heap.InsertTuple(MakeTuple(i, schema)).page_id, INVALID_PAGE_ID);
    }

    // Count heap pages by walking the chain once.
    size_t page_count = 0;
    for (page_id_t pid = heap.GetFirstPageId(); pid != INVALID_PAGE_ID;) {
        auto* page = bpm->FetchPage(pid);
        ASSERT_NE(page, nullptr);
        auto next = SlottedPage::GetHeader(page->GetData())->next_page_id;
        (void)bpm->UnpinPage(pid, false);
        pid = next;
        ++page_count;
    }
    ASSERT_GT(page_count, 1u);

    bpm->ResetStats();
    RID pos{heap.GetFirstPageId(), 0};
    size_t rows = heap.ReadRows(&pos, 100000, [](const char*, uint32_t) {});
    EXPECT_EQ(rows, 1000u);
    EXPECT_EQ(bpm->GetHitCount() + bpm->GetMissCount(), page_count);

    for (size_t i = 0; i < bpm->GetPoolSize(); ++i) {
        EXPECT_EQ(bpm->GetPage(static_cast<frame_id_t>(i)).GetPinCount(), 0)
            << "Frame " << i << " still pinned after ReadRows";
    }
}

}  // namespace shilmandb
