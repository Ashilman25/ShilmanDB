#include <gtest/gtest.h>
#include "executor/hash_join_executor.hpp"
#include "executor/seq_scan_executor.hpp"
#include "planner/plan_node.hpp"
#include "catalog/catalog.hpp"
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "storage/disk_manager.hpp"
#include <filesystem>
#include <memory>
#include <set>
#include <string>

namespace shilmandb {

// --- AST builder helpers ---

static auto MakeColRef(const std::string& col) {
    auto e = std::make_unique<ColumnRef>();
    e->column_name = col;
    return e;
}

static auto MakeEqPred(const std::string& left_col, const std::string& right_col) {
    auto e = std::make_unique<BinaryOp>();
    e->op = BinaryOp::Op::EQ;
    e->left = MakeColRef(left_col);
    e->right = MakeColRef(right_col);
    return e;
}

// --- Test fixture ---

class HashJoinExecutorTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() / "shilmandb_hash_join_test.db").string();
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
};

TEST_F(HashJoinExecutorTest, BasicJoin) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    Schema left_schema({Column("lid", TypeId::INTEGER), Column("lval", TypeId::VARCHAR)});
    Schema right_schema({Column("rid", TypeId::INTEGER), Column("rval", TypeId::VARCHAR)});

    auto* left_info = catalog.CreateTable("left_t", left_schema);
    auto* right_info = catalog.CreateTable("right_t", right_schema);

    // Left: (1,"a"), (2,"b"), (3,"c")
    for (auto& [id, v] : std::vector<std::pair<int, std::string>>{{1,"a"},{2,"b"},{3,"c"}}) {
        (void)left_info->table->InsertTuple(Tuple({Value(id), Value(v)}, left_schema));
    }
    // Right: (2,"x"), (3,"y"), (4,"z")
    for (auto& [id, v] : std::vector<std::pair<int, std::string>>{{2,"x"},{3,"y"},{4,"z"}}) {
        (void)right_info->table->InsertTuple(Tuple({Value(id), Value(v)}, right_schema));
    }

    // Build join plan: join on lid = rid
    Schema output_schema({Column("lid", TypeId::INTEGER), Column("lval", TypeId::VARCHAR),
                          Column("rid", TypeId::INTEGER), Column("rval", TypeId::VARCHAR)});

    auto left_scan = std::make_unique<SeqScanPlanNode>(left_schema, "left_t");
    auto right_scan = std::make_unique<SeqScanPlanNode>(right_schema, "right_t");
    auto join_pred = MakeEqPred("lid", "rid");
    auto join_plan = std::make_unique<HashJoinPlanNode>(output_schema, std::move(join_pred));

    ExecutorContext ctx{bundle.bpm.get(), &catalog};
    auto left_exec = std::make_unique<SeqScanExecutor>(left_scan.get(), &ctx);
    auto right_exec = std::make_unique<SeqScanExecutor>(right_scan.get(), &ctx);
    HashJoinExecutor join_exec(join_plan.get(), &ctx, std::move(left_exec), std::move(right_exec));
    join_exec.Init();

    std::set<int> matched_keys;
    Tuple result;
    int count = 0;
    while (join_exec.Next(&result)) {
        auto lid = result.GetValue(output_schema, 0).integer_;
        auto lval = result.GetValue(output_schema, 1).varchar_;
        auto rid = result.GetValue(output_schema, 2).integer_;
        auto rval = result.GetValue(output_schema, 3).varchar_;

        EXPECT_EQ(lid, rid);
        matched_keys.insert(lid);

        if (lid == 2) { EXPECT_EQ(lval, "b"); EXPECT_EQ(rval, "x"); }
        if (lid == 3) { EXPECT_EQ(lval, "c"); EXPECT_EQ(rval, "y"); }
        ++count;
    }
    EXPECT_EQ(count, 2);
    EXPECT_TRUE(matched_keys.count(2));
    EXPECT_TRUE(matched_keys.count(3));
    join_exec.Close();
}

TEST_F(HashJoinExecutorTest, DuplicateKeysOnLeft) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    Schema left_schema({Column("lid", TypeId::INTEGER), Column("lval", TypeId::VARCHAR)});
    Schema right_schema({Column("rid", TypeId::INTEGER), Column("rval", TypeId::VARCHAR)});

    auto* left_info = catalog.CreateTable("left_t", left_schema);
    auto* right_info = catalog.CreateTable("right_t", right_schema);

    // Left: two rows with key=2
    (void)left_info->table->InsertTuple(Tuple({Value(2), Value("b1")}, left_schema));
    (void)left_info->table->InsertTuple(Tuple({Value(2), Value("b2")}, left_schema));

    // Right: one row with key=2
    (void)right_info->table->InsertTuple(Tuple({Value(2), Value("x")}, right_schema));

    Schema output_schema({Column("lid", TypeId::INTEGER), Column("lval", TypeId::VARCHAR),
                          Column("rid", TypeId::INTEGER), Column("rval", TypeId::VARCHAR)});

    auto left_scan = std::make_unique<SeqScanPlanNode>(left_schema, "left_t");
    auto right_scan = std::make_unique<SeqScanPlanNode>(right_schema, "right_t");
    auto join_plan = std::make_unique<HashJoinPlanNode>(output_schema, MakeEqPred("lid", "rid"));

    ExecutorContext ctx{bundle.bpm.get(), &catalog};
    auto left_exec = std::make_unique<SeqScanExecutor>(left_scan.get(), &ctx);
    auto right_exec = std::make_unique<SeqScanExecutor>(right_scan.get(), &ctx);
    HashJoinExecutor join_exec(join_plan.get(), &ctx, std::move(left_exec), std::move(right_exec));
    join_exec.Init();

    std::set<std::string> lvals;
    Tuple result;
    int count = 0;
    while (join_exec.Next(&result)) {
        EXPECT_EQ(result.GetValue(output_schema, 0).integer_, 2);
        lvals.insert(result.GetValue(output_schema, 1).varchar_);
        ++count;
    }
    EXPECT_EQ(count, 2);
    EXPECT_TRUE(lvals.count("b1"));
    EXPECT_TRUE(lvals.count("b2"));
    join_exec.Close();
}

TEST_F(HashJoinExecutorTest, EmptyRightTable) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    Schema left_schema({Column("lid", TypeId::INTEGER)});
    Schema right_schema({Column("rid", TypeId::INTEGER)});

    auto* left_info = catalog.CreateTable("left_t", left_schema);
    (void)catalog.CreateTable("right_t", right_schema);

    (void)left_info->table->InsertTuple(Tuple({Value(1)}, left_schema));

    Schema output_schema({Column("lid", TypeId::INTEGER), Column("rid", TypeId::INTEGER)});

    auto left_scan = std::make_unique<SeqScanPlanNode>(left_schema, "left_t");
    auto right_scan = std::make_unique<SeqScanPlanNode>(right_schema, "right_t");
    auto join_plan = std::make_unique<HashJoinPlanNode>(output_schema, MakeEqPred("lid", "rid"));

    ExecutorContext ctx{bundle.bpm.get(), &catalog};
    auto left_exec = std::make_unique<SeqScanExecutor>(left_scan.get(), &ctx);
    auto right_exec = std::make_unique<SeqScanExecutor>(right_scan.get(), &ctx);
    HashJoinExecutor join_exec(join_plan.get(), &ctx, std::move(left_exec), std::move(right_exec));
    join_exec.Init();

    Tuple result;
    EXPECT_FALSE(join_exec.Next(&result));
    join_exec.Close();
}

TEST_F(HashJoinExecutorTest, NoMatchingKeys) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    Schema left_schema({Column("lid", TypeId::INTEGER)});
    Schema right_schema({Column("rid", TypeId::INTEGER)});

    auto* left_info = catalog.CreateTable("left_t", left_schema);
    auto* right_info = catalog.CreateTable("right_t", right_schema);

    for (int i = 1; i <= 3; ++i)
        (void)left_info->table->InsertTuple(Tuple({Value(i)}, left_schema));
    for (int i = 4; i <= 6; ++i)
        (void)right_info->table->InsertTuple(Tuple({Value(i)}, right_schema));

    Schema output_schema({Column("lid", TypeId::INTEGER), Column("rid", TypeId::INTEGER)});

    auto left_scan = std::make_unique<SeqScanPlanNode>(left_schema, "left_t");
    auto right_scan = std::make_unique<SeqScanPlanNode>(right_schema, "right_t");
    auto join_plan = std::make_unique<HashJoinPlanNode>(output_schema, MakeEqPred("lid", "rid"));

    ExecutorContext ctx{bundle.bpm.get(), &catalog};
    auto left_exec = std::make_unique<SeqScanExecutor>(left_scan.get(), &ctx);
    auto right_exec = std::make_unique<SeqScanExecutor>(right_scan.get(), &ctx);
    HashJoinExecutor join_exec(join_plan.get(), &ctx, std::move(left_exec), std::move(right_exec));
    join_exec.Init();

    Tuple result;
    EXPECT_FALSE(join_exec.Next(&result));
    join_exec.Close();
}

TEST_F(HashJoinExecutorTest, LargeJoin) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    Schema left_schema({Column("lid", TypeId::INTEGER), Column("lval", TypeId::INTEGER)});
    Schema right_schema({Column("rid", TypeId::INTEGER), Column("rval", TypeId::INTEGER)});

    auto* left_info = catalog.CreateTable("left_t", left_schema);
    auto* right_info = catalog.CreateTable("right_t", right_schema);

    for (int i = 0; i < 100; ++i) {
        (void)left_info->table->InsertTuple(
            Tuple({Value(i), Value(i * 10)}, left_schema));
        (void)right_info->table->InsertTuple(
            Tuple({Value(i), Value(i * 100)}, right_schema));
    }

    Schema output_schema({Column("lid", TypeId::INTEGER), Column("lval", TypeId::INTEGER),
                          Column("rid", TypeId::INTEGER), Column("rval", TypeId::INTEGER)});

    auto left_scan = std::make_unique<SeqScanPlanNode>(left_schema, "left_t");
    auto right_scan = std::make_unique<SeqScanPlanNode>(right_schema, "right_t");
    auto join_plan = std::make_unique<HashJoinPlanNode>(output_schema, MakeEqPred("lid", "rid"));

    ExecutorContext ctx{bundle.bpm.get(), &catalog};
    auto left_exec = std::make_unique<SeqScanExecutor>(left_scan.get(), &ctx);
    auto right_exec = std::make_unique<SeqScanExecutor>(right_scan.get(), &ctx);
    HashJoinExecutor join_exec(join_plan.get(), &ctx, std::move(left_exec), std::move(right_exec));
    join_exec.Init();

    int count = 0;
    Tuple result;
    while (join_exec.Next(&result)) {
        auto lid = result.GetValue(output_schema, 0).integer_;
        auto rid = result.GetValue(output_schema, 2).integer_;
        EXPECT_EQ(lid, rid);

        auto lval = result.GetValue(output_schema, 1).integer_;
        auto rval = result.GetValue(output_schema, 3).integer_;
        EXPECT_EQ(lval, lid * 10);
        EXPECT_EQ(rval, rid * 100);
        ++count;
    }
    EXPECT_EQ(count, 100);
    join_exec.Close();
}

TEST_F(HashJoinExecutorTest, ManyToManyDuplicateKeys) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    Schema left_schema({Column("lid", TypeId::INTEGER), Column("lval", TypeId::VARCHAR)});
    Schema right_schema({Column("rid", TypeId::INTEGER), Column("rval", TypeId::VARCHAR)});

    auto* left_info = catalog.CreateTable("left_t", left_schema);
    auto* right_info = catalog.CreateTable("right_t", right_schema);

    // Left: two rows with key=2, right: two rows with key=2 → 4 output rows
    (void)left_info->table->InsertTuple(Tuple({Value(2), Value("L1")}, left_schema));
    (void)left_info->table->InsertTuple(Tuple({Value(2), Value("L2")}, left_schema));
    (void)right_info->table->InsertTuple(Tuple({Value(2), Value("R1")}, right_schema));
    (void)right_info->table->InsertTuple(Tuple({Value(2), Value("R2")}, right_schema));

    Schema output_schema({Column("lid", TypeId::INTEGER), Column("lval", TypeId::VARCHAR),
                          Column("rid", TypeId::INTEGER), Column("rval", TypeId::VARCHAR)});

    auto left_scan = std::make_unique<SeqScanPlanNode>(left_schema, "left_t");
    auto right_scan = std::make_unique<SeqScanPlanNode>(right_schema, "right_t");
    auto join_plan = std::make_unique<HashJoinPlanNode>(output_schema, MakeEqPred("lid", "rid"));

    ExecutorContext ctx{bundle.bpm.get(), &catalog};
    auto left_exec = std::make_unique<SeqScanExecutor>(left_scan.get(), &ctx);
    auto right_exec = std::make_unique<SeqScanExecutor>(right_scan.get(), &ctx);
    HashJoinExecutor join_exec(join_plan.get(), &ctx, std::move(left_exec), std::move(right_exec));
    join_exec.Init();

    std::set<std::string> combos;
    Tuple result;
    int count = 0;
    while (join_exec.Next(&result)) {
        auto lval = result.GetValue(output_schema, 1).varchar_;
        auto rval = result.GetValue(output_schema, 3).varchar_;
        combos.insert(lval + "+" + rval);
        ++count;
    }
    EXPECT_EQ(count, 4);
    EXPECT_TRUE(combos.count("L1+R1"));
    EXPECT_TRUE(combos.count("L1+R2"));
    EXPECT_TRUE(combos.count("L2+R1"));
    EXPECT_TRUE(combos.count("L2+R2"));
    join_exec.Close();
}

TEST_F(HashJoinExecutorTest, EmptyLeftTable) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    Schema left_schema({Column("lid", TypeId::INTEGER)});
    Schema right_schema({Column("rid", TypeId::INTEGER)});

    (void)catalog.CreateTable("left_t", left_schema);
    auto* right_info = catalog.CreateTable("right_t", right_schema);
    (void)right_info->table->InsertTuple(Tuple({Value(1)}, right_schema));

    Schema output_schema({Column("lid", TypeId::INTEGER), Column("rid", TypeId::INTEGER)});

    auto left_scan = std::make_unique<SeqScanPlanNode>(left_schema, "left_t");
    auto right_scan = std::make_unique<SeqScanPlanNode>(right_schema, "right_t");
    auto join_plan = std::make_unique<HashJoinPlanNode>(output_schema, MakeEqPred("lid", "rid"));

    ExecutorContext ctx{bundle.bpm.get(), &catalog};
    auto left_exec = std::make_unique<SeqScanExecutor>(left_scan.get(), &ctx);
    auto right_exec = std::make_unique<SeqScanExecutor>(right_scan.get(), &ctx);
    HashJoinExecutor join_exec(join_plan.get(), &ctx, std::move(left_exec), std::move(right_exec));
    join_exec.Init();

    Tuple result;
    EXPECT_FALSE(join_exec.Next(&result));
    join_exec.Close();
}

}  // namespace shilmandb
