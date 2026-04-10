#include <gtest/gtest.h>
#include "executor/sort_executor.hpp"
#include "executor/seq_scan_executor.hpp"
#include "planner/plan_node.hpp"
#include "catalog/catalog.hpp"
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "storage/disk_manager.hpp"
#include <filesystem>
#include <memory>

namespace shilmandb {

// --- AST builder helpers ---

static auto MakeColRef(const std::string& col) {
    auto e = std::make_unique<ColumnRef>();
    e->column_name = col;
    return e;
}

// --- Test fixture ---

class SortExecutorTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() / "shilmandb_sort_test.db").string();
        std::filesystem::remove(test_file_);
    }

    void TearDown() override {
        std::filesystem::remove(test_file_);
    }

    struct BPMBundle {
        std::unique_ptr<DiskManager> disk_manager;
        std::unique_ptr<BufferPoolManager> bpm;
    };

    static BPMBundle MakeBPM(const std::string& path, size_t pool_size = 1000) {
        auto dm = std::make_unique<DiskManager>(path);
        auto eviction = std::make_unique<LRUEvictionPolicy>(pool_size);
        auto bpm = std::make_unique<BufferPoolManager>(
            pool_size, dm.get(), std::move(eviction));
        return {std::move(dm), std::move(bpm)};
    }

    static Schema MakeSchema() {
        return Schema({
            Column("id", TypeId::INTEGER),
            Column("name", TypeId::VARCHAR),
            Column("score", TypeId::INTEGER)
        });
    }

    struct TestEnv {
        BPMBundle bundle;
        std::unique_ptr<Catalog> catalog;
        ExecutorContext ctx;
        Schema schema;
    };

    TestEnv SetUpEnv() {
        auto bundle = MakeBPM(test_file_);
        auto catalog = std::make_unique<Catalog>(bundle.bpm.get());
        auto schema = MakeSchema();
        (void)catalog->CreateTable("t", schema);
        ExecutorContext ctx{bundle.bpm.get(), catalog.get()};
        return {std::move(bundle), std::move(catalog), ctx, schema};
    }

    void InsertRow(TestEnv& env, int32_t id, const std::string& name, int32_t score) {
        auto* table_info = env.catalog->GetTable("t");
        std::vector<Value> vals = {Value(id), Value(name), Value(score)};
        (void)table_info->table->InsertTuple(Tuple(vals, env.schema));
    }

    // Build SeqScan -> Sort pipeline, collect all output tuples
    std::vector<Tuple> RunSort(TestEnv& env, std::vector<OrderByItem> order_by) {
        auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");
        auto sort_plan = std::make_unique<SortPlanNode>(env.schema, std::move(order_by));

        auto child = std::make_unique<SeqScanExecutor>(scan_plan.get(), &env.ctx);
        SortExecutor sorter(sort_plan.get(), &env.ctx, std::move(child));
        sorter.Init();

        std::vector<Tuple> results;
        Tuple result;
        while (sorter.Next(&result)) {
            results.push_back(result);
        }
        sorter.Close();
        return results;
    }
};

// --- Tests ---

TEST_F(SortExecutorTest, SingleColumnAsc) {
    auto env = SetUpEnv();
    InsertRow(env, 3, "c", 30);
    InsertRow(env, 1, "a", 10);
    InsertRow(env, 2, "b", 20);

    std::vector<OrderByItem> ob;
    ob.push_back({MakeColRef("id"), true});

    auto results = RunSort(env, std::move(ob));
    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0].GetValue(env.schema, 0).integer_, 1);
    EXPECT_EQ(results[1].GetValue(env.schema, 0).integer_, 2);
    EXPECT_EQ(results[2].GetValue(env.schema, 0).integer_, 3);
}

TEST_F(SortExecutorTest, SingleColumnDesc) {
    auto env = SetUpEnv();
    InsertRow(env, 3, "c", 30);
    InsertRow(env, 1, "a", 10);
    InsertRow(env, 2, "b", 20);

    std::vector<OrderByItem> ob;
    ob.push_back({MakeColRef("id"), false});

    auto results = RunSort(env, std::move(ob));
    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0].GetValue(env.schema, 0).integer_, 3);
    EXPECT_EQ(results[1].GetValue(env.schema, 0).integer_, 2);
    EXPECT_EQ(results[2].GetValue(env.schema, 0).integer_, 1);
}

TEST_F(SortExecutorTest, MultiColumnIntThenVarchar) {
    auto env = SetUpEnv();
    InsertRow(env, 1, "b", 10);
    InsertRow(env, 2, "a", 20);
    InsertRow(env, 1, "a", 30);

    std::vector<OrderByItem> ob;
    ob.push_back({MakeColRef("id"), true});    // ASC
    ob.push_back({MakeColRef("name"), true});  // ASC tiebreaker

    auto results = RunSort(env, std::move(ob));
    ASSERT_EQ(results.size(), 3u);

    // (1, "a", 30), (1, "b", 10), (2, "a", 20)
    EXPECT_EQ(results[0].GetValue(env.schema, 0).integer_, 1);
    EXPECT_EQ(results[0].GetValue(env.schema, 1).varchar_, "a");
    EXPECT_EQ(results[0].GetValue(env.schema, 2).integer_, 30);

    EXPECT_EQ(results[1].GetValue(env.schema, 0).integer_, 1);
    EXPECT_EQ(results[1].GetValue(env.schema, 1).varchar_, "b");
    EXPECT_EQ(results[1].GetValue(env.schema, 2).integer_, 10);

    EXPECT_EQ(results[2].GetValue(env.schema, 0).integer_, 2);
    EXPECT_EQ(results[2].GetValue(env.schema, 1).varchar_, "a");
    EXPECT_EQ(results[2].GetValue(env.schema, 2).integer_, 20);
}

TEST_F(SortExecutorTest, MultiColumnVarcharThenInt) {
    auto env = SetUpEnv();
    InsertRow(env, 3, "a", 30);
    InsertRow(env, 1, "b", 10);
    InsertRow(env, 2, "a", 20);

    std::vector<OrderByItem> ob;
    ob.push_back({MakeColRef("name"), true});  // ASC
    ob.push_back({MakeColRef("id"), true});    // ASC tiebreaker

    auto results = RunSort(env, std::move(ob));
    ASSERT_EQ(results.size(), 3u);

    // (2, "a", 20), (3, "a", 30), (1, "b", 10)
    EXPECT_EQ(results[0].GetValue(env.schema, 0).integer_, 2);
    EXPECT_EQ(results[0].GetValue(env.schema, 1).varchar_, "a");

    EXPECT_EQ(results[1].GetValue(env.schema, 0).integer_, 3);
    EXPECT_EQ(results[1].GetValue(env.schema, 1).varchar_, "a");

    EXPECT_EQ(results[2].GetValue(env.schema, 0).integer_, 1);
    EXPECT_EQ(results[2].GetValue(env.schema, 1).varchar_, "b");
}

TEST_F(SortExecutorTest, MultiColumnMixedAscDesc) {
    auto env = SetUpEnv();
    InsertRow(env, 1, "b", 20);
    InsertRow(env, 1, "a", 30);
    InsertRow(env, 2, "c", 10);
    InsertRow(env, 1, "c", 10);

    std::vector<OrderByItem> ob;
    ob.push_back({MakeColRef("id"), true});     // ASC
    ob.push_back({MakeColRef("score"), false});  // DESC tiebreaker

    auto results = RunSort(env, std::move(ob));
    ASSERT_EQ(results.size(), 4u);

    // id ASC, then score DESC within each group:
    // (1, "a", 30), (1, "b", 20), (1, "c", 10), (2, "c", 10)
    EXPECT_EQ(results[0].GetValue(env.schema, 0).integer_, 1);
    EXPECT_EQ(results[0].GetValue(env.schema, 2).integer_, 30);

    EXPECT_EQ(results[1].GetValue(env.schema, 0).integer_, 1);
    EXPECT_EQ(results[1].GetValue(env.schema, 2).integer_, 20);

    EXPECT_EQ(results[2].GetValue(env.schema, 0).integer_, 1);
    EXPECT_EQ(results[2].GetValue(env.schema, 2).integer_, 10);

    EXPECT_EQ(results[3].GetValue(env.schema, 0).integer_, 2);
    EXPECT_EQ(results[3].GetValue(env.schema, 2).integer_, 10);
}

TEST_F(SortExecutorTest, EmptyInput) {
    auto env = SetUpEnv();
    // No rows inserted

    std::vector<OrderByItem> ob;
    ob.push_back({MakeColRef("id"), true});

    auto results = RunSort(env, std::move(ob));
    EXPECT_TRUE(results.empty());
}

TEST_F(SortExecutorTest, SingleRow) {
    auto env = SetUpEnv();
    InsertRow(env, 1, "a", 10);

    std::vector<OrderByItem> ob;
    ob.push_back({MakeColRef("id"), true});

    auto results = RunSort(env, std::move(ob));
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].GetValue(env.schema, 0).integer_, 1);
    EXPECT_EQ(results[0].GetValue(env.schema, 1).varchar_, "a");
    EXPECT_EQ(results[0].GetValue(env.schema, 2).integer_, 10);
}

TEST_F(SortExecutorTest, AlreadySorted) {
    auto env = SetUpEnv();
    InsertRow(env, 1, "a", 10);
    InsertRow(env, 2, "b", 20);
    InsertRow(env, 3, "c", 30);

    std::vector<OrderByItem> ob;
    ob.push_back({MakeColRef("id"), true});

    auto results = RunSort(env, std::move(ob));
    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0].GetValue(env.schema, 0).integer_, 1);
    EXPECT_EQ(results[1].GetValue(env.schema, 0).integer_, 2);
    EXPECT_EQ(results[2].GetValue(env.schema, 0).integer_, 3);
}

TEST_F(SortExecutorTest, ReinitResetsSortedOutput) {
    auto env = SetUpEnv();
    InsertRow(env, 3, "c", 30);
    InsertRow(env, 1, "a", 10);
    InsertRow(env, 2, "b", 20);

    auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");
    std::vector<OrderByItem> ob;
    ob.push_back({MakeColRef("id"), true});
    auto sort_plan = std::make_unique<SortPlanNode>(env.schema, std::move(ob));

    auto child = std::make_unique<SeqScanExecutor>(scan_plan.get(), &env.ctx);
    SortExecutor sorter(sort_plan.get(), &env.ctx, std::move(child));

    // First pass
    sorter.Init();
    Tuple result;
    int count1 = 0;
    while (sorter.Next(&result)) { ++count1; }
    EXPECT_EQ(count1, 3);

    // Re-init and drain again — should yield same results
    sorter.Init();
    int count2 = 0;
    while (sorter.Next(&result)) { ++count2; }
    EXPECT_EQ(count2, 3);
    sorter.Close();
}

TEST_F(SortExecutorTest, NextBeforeInitAsserts) {
    auto env = SetUpEnv();
    auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");
    std::vector<OrderByItem> ob;
    ob.push_back({MakeColRef("id"), true});
    auto sort_plan = std::make_unique<SortPlanNode>(env.schema, std::move(ob));

    auto child = std::make_unique<SeqScanExecutor>(scan_plan.get(), &env.ctx);
    SortExecutor sorter(sort_plan.get(), &env.ctx, std::move(child));

    Tuple t;
    EXPECT_DEATH(sorter.Next(&t), "Next.*called before Init");
}

}  // namespace shilmandb
