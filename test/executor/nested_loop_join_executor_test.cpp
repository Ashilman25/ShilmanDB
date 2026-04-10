#include <gtest/gtest.h>
#include "executor/nested_loop_join_executor.hpp"
#include "executor/seq_scan_executor.hpp"
#include "planner/plan_node.hpp"
#include "catalog/catalog.hpp"
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "storage/disk_manager.hpp"
#include <filesystem>
#include <memory>
#include <set>

namespace shilmandb {

// --- AST helpers ---

static auto MakeColRef(const std::string& col) {
    auto e = std::make_unique<ColumnRef>();
    e->column_name = col;
    return e;
}

static auto MakeBinOp(BinaryOp::Op op,
                      std::unique_ptr<Expression> lhs,
                      std::unique_ptr<Expression> rhs) {
    auto e = std::make_unique<BinaryOp>();
    e->op = op;
    e->left = std::move(lhs);
    e->right = std::move(rhs);
    return e;
}

static auto MakeLiteral(int32_t v) {
    auto e = std::make_unique<Literal>();
    e->value = Value(v);
    return e;
}

// --- Test fixture ---

class NestedLoopJoinExecutorTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() / "shilmandb_nlj_test.db").string();
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

TEST_F(NestedLoopJoinExecutorTest, NonEqualityPredicate) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    Schema left_schema({Column("lid", TypeId::INTEGER)});
    Schema right_schema({Column("rid", TypeId::INTEGER)});

    auto* left_info = catalog.CreateTable("left_t", left_schema);
    auto* right_info = catalog.CreateTable("right_t", right_schema);

    // Left: 1, 2, 3.  Right: 1, 2, 3.
    for (int i = 1; i <= 3; ++i) {
        (void)left_info->table->InsertTuple(Tuple({Value(i)}, left_schema));
        (void)right_info->table->InsertTuple(Tuple({Value(i)}, right_schema));
    }

    // lid < rid  → (1,2), (1,3), (2,3) = 3 results
    Schema output_schema({Column("lid", TypeId::INTEGER), Column("rid", TypeId::INTEGER)});

    auto left_scan = std::make_unique<SeqScanPlanNode>(left_schema, "left_t");
    auto right_scan = std::make_unique<SeqScanPlanNode>(right_schema, "right_t");
    auto pred = MakeBinOp(BinaryOp::Op::LT, MakeColRef("lid"), MakeColRef("rid"));
    auto join_plan = std::make_unique<NestedLoopJoinPlanNode>(output_schema, std::move(pred));

    ExecutorContext ctx{bundle.bpm.get(), &catalog};
    auto left_exec = std::make_unique<SeqScanExecutor>(left_scan.get(), &ctx);
    auto right_exec = std::make_unique<SeqScanExecutor>(right_scan.get(), &ctx);
    NestedLoopJoinExecutor nlj(join_plan.get(), &ctx, std::move(left_exec), std::move(right_exec));
    nlj.Init();

    int count = 0;
    Tuple result;
    while (nlj.Next(&result)) {
        auto lid = result.GetValue(output_schema, 0).integer_;
        auto rid = result.GetValue(output_schema, 1).integer_;
        EXPECT_LT(lid, rid);
        ++count;
    }
    EXPECT_EQ(count, 3);
    nlj.Close();
}

TEST_F(NestedLoopJoinExecutorTest, EqualityPredicate) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    Schema left_schema({Column("lid", TypeId::INTEGER), Column("lval", TypeId::VARCHAR)});
    Schema right_schema({Column("rid", TypeId::INTEGER), Column("rval", TypeId::VARCHAR)});

    auto* left_info = catalog.CreateTable("left_t", left_schema);
    auto* right_info = catalog.CreateTable("right_t", right_schema);

    for (auto& [id, v] : std::vector<std::pair<int, std::string>>{{1,"a"},{2,"b"},{3,"c"}})
        (void)left_info->table->InsertTuple(Tuple({Value(id), Value(v)}, left_schema));
    for (auto& [id, v] : std::vector<std::pair<int, std::string>>{{2,"x"},{3,"y"},{4,"z"}})
        (void)right_info->table->InsertTuple(Tuple({Value(id), Value(v)}, right_schema));

    Schema output_schema({Column("lid", TypeId::INTEGER), Column("lval", TypeId::VARCHAR),
                          Column("rid", TypeId::INTEGER), Column("rval", TypeId::VARCHAR)});

    auto left_scan = std::make_unique<SeqScanPlanNode>(left_schema, "left_t");
    auto right_scan = std::make_unique<SeqScanPlanNode>(right_schema, "right_t");
    auto pred = MakeBinOp(BinaryOp::Op::EQ, MakeColRef("lid"), MakeColRef("rid"));
    auto join_plan = std::make_unique<NestedLoopJoinPlanNode>(output_schema, std::move(pred));

    ExecutorContext ctx{bundle.bpm.get(), &catalog};
    auto left_exec = std::make_unique<SeqScanExecutor>(left_scan.get(), &ctx);
    auto right_exec = std::make_unique<SeqScanExecutor>(right_scan.get(), &ctx);
    NestedLoopJoinExecutor nlj(join_plan.get(), &ctx, std::move(left_exec), std::move(right_exec));
    nlj.Init();

    std::set<int> matched;
    Tuple result;
    int count = 0;
    while (nlj.Next(&result)) {
        auto lid = result.GetValue(output_schema, 0).integer_;
        auto rid = result.GetValue(output_schema, 2).integer_;
        EXPECT_EQ(lid, rid);
        matched.insert(lid);
        ++count;
    }
    EXPECT_EQ(count, 2);
    EXPECT_TRUE(matched.count(2));
    EXPECT_TRUE(matched.count(3));
    nlj.Close();
}

TEST_F(NestedLoopJoinExecutorTest, EmptyRightTable) {
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
    auto pred = MakeBinOp(BinaryOp::Op::EQ, MakeColRef("lid"), MakeColRef("rid"));
    auto join_plan = std::make_unique<NestedLoopJoinPlanNode>(output_schema, std::move(pred));

    ExecutorContext ctx{bundle.bpm.get(), &catalog};
    auto left_exec = std::make_unique<SeqScanExecutor>(left_scan.get(), &ctx);
    auto right_exec = std::make_unique<SeqScanExecutor>(right_scan.get(), &ctx);
    NestedLoopJoinExecutor nlj(join_plan.get(), &ctx, std::move(left_exec), std::move(right_exec));
    nlj.Init();

    Tuple result;
    EXPECT_FALSE(nlj.Next(&result));
    nlj.Close();
}

TEST_F(NestedLoopJoinExecutorTest, CrossJoin) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());

    Schema left_schema({Column("lid", TypeId::INTEGER)});
    Schema right_schema({Column("rid", TypeId::INTEGER)});

    auto* left_info = catalog.CreateTable("left_t", left_schema);
    auto* right_info = catalog.CreateTable("right_t", right_schema);

    constexpr int kLeft = 5, kRight = 4;
    for (int i = 0; i < kLeft; ++i)
        (void)left_info->table->InsertTuple(Tuple({Value(i)}, left_schema));
    for (int i = 0; i < kRight; ++i)
        (void)right_info->table->InsertTuple(Tuple({Value(i)}, right_schema));

    // Predicate always true: 1 = 1
    Schema output_schema({Column("lid", TypeId::INTEGER), Column("rid", TypeId::INTEGER)});

    auto left_scan = std::make_unique<SeqScanPlanNode>(left_schema, "left_t");
    auto right_scan = std::make_unique<SeqScanPlanNode>(right_schema, "right_t");
    auto pred = MakeBinOp(BinaryOp::Op::EQ, MakeLiteral(1), MakeLiteral(1));
    auto join_plan = std::make_unique<NestedLoopJoinPlanNode>(output_schema, std::move(pred));

    ExecutorContext ctx{bundle.bpm.get(), &catalog};
    auto left_exec = std::make_unique<SeqScanExecutor>(left_scan.get(), &ctx);
    auto right_exec = std::make_unique<SeqScanExecutor>(right_scan.get(), &ctx);
    NestedLoopJoinExecutor nlj(join_plan.get(), &ctx, std::move(left_exec), std::move(right_exec));
    nlj.Init();

    int count = 0;
    Tuple result;
    while (nlj.Next(&result)) { ++count; }
    EXPECT_EQ(count, kLeft * kRight);
    nlj.Close();
}

TEST_F(NestedLoopJoinExecutorTest, EmptyLeftTable) {
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
    auto pred = MakeBinOp(BinaryOp::Op::EQ, MakeColRef("lid"), MakeColRef("rid"));
    auto join_plan = std::make_unique<NestedLoopJoinPlanNode>(output_schema, std::move(pred));

    ExecutorContext ctx{bundle.bpm.get(), &catalog};
    auto left_exec = std::make_unique<SeqScanExecutor>(left_scan.get(), &ctx);
    auto right_exec = std::make_unique<SeqScanExecutor>(right_scan.get(), &ctx);
    NestedLoopJoinExecutor nlj(join_plan.get(), &ctx, std::move(left_exec), std::move(right_exec));
    nlj.Init();

    Tuple result;
    EXPECT_FALSE(nlj.Next(&result));
    nlj.Close();
}

}  // namespace shilmandb
