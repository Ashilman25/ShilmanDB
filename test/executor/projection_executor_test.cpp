#include <gtest/gtest.h>
#include "executor/projection_executor.hpp"
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

static auto MakeLiteral(int32_t v) {
    auto e = std::make_unique<Literal>();
    e->value = Value(v);
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

// --- Test fixture ---

class ProjectionExecutorTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() / "shilmandb_proj_test.db").string();
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

    static Schema MakeInputSchema() {
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
        auto schema = MakeInputSchema();
        (void)catalog->CreateTable("t", schema);
        ExecutorContext ctx{bundle.bpm.get(), catalog.get()};
        return {std::move(bundle), std::move(catalog), ctx, schema};
    }

    void InsertRow(TestEnv& env, int32_t id, const std::string& name, int32_t score) {
        auto* table_info = env.catalog->GetTable("t");
        std::vector<Value> vals = {Value(id), Value(name), Value(score)};
        (void)table_info->table->InsertTuple(Tuple(vals, env.schema));
    }

    std::vector<Tuple> RunProjection(TestEnv& env,
                                     std::vector<std::unique_ptr<Expression>> exprs,
                                     Schema output_schema) {
        auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");
        auto proj_plan = std::make_unique<ProjectionPlanNode>(
            std::move(output_schema), std::move(exprs));

        auto child = std::make_unique<SeqScanExecutor>(scan_plan.get(), &env.ctx);
        ProjectionExecutor proj(proj_plan.get(), &env.ctx, std::move(child));
        proj.Init();

        std::vector<Tuple> results;
        Tuple result;
        while (proj.Next(&result)) {
            results.push_back(result);
        }
        proj.Close();
        return results;
    }
};

// --- Tests ---

TEST_F(ProjectionExecutorTest, ColumnSubset) {
    auto env = SetUpEnv();
    InsertRow(env, 1, "a", 10);
    InsertRow(env, 2, "b", 20);
    InsertRow(env, 3, "c", 30);

    std::vector<std::unique_ptr<Expression>> exprs;
    exprs.push_back(MakeColRef("id"));
    exprs.push_back(MakeColRef("score"));

    Schema out_schema({Column("id", TypeId::INTEGER), Column("score", TypeId::INTEGER)});
    Schema result_schema({Column("id", TypeId::INTEGER), Column("score", TypeId::INTEGER)});

    auto results = RunProjection(env, std::move(exprs), std::move(out_schema));
    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0].GetValue(result_schema, 0).integer_, 1);
    EXPECT_EQ(results[0].GetValue(result_schema, 1).integer_, 10);
    EXPECT_EQ(results[1].GetValue(result_schema, 0).integer_, 2);
    EXPECT_EQ(results[1].GetValue(result_schema, 1).integer_, 20);
    EXPECT_EQ(results[2].GetValue(result_schema, 0).integer_, 3);
    EXPECT_EQ(results[2].GetValue(result_schema, 1).integer_, 30);
}

TEST_F(ProjectionExecutorTest, ArithmeticProjection) {
    auto env = SetUpEnv();
    InsertRow(env, 1, "a", 10);
    InsertRow(env, 2, "b", 20);
    InsertRow(env, 3, "c", 30);

    std::vector<std::unique_ptr<Expression>> exprs;
    exprs.push_back(MakeBinOp(BinaryOp::Op::ADD, MakeColRef("id"), MakeColRef("score")));

    Schema out_schema({Column("sum", TypeId::INTEGER)});
    Schema result_schema({Column("sum", TypeId::INTEGER)});

    auto results = RunProjection(env, std::move(exprs), std::move(out_schema));
    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0].GetValue(result_schema, 0).integer_, 11);
    EXPECT_EQ(results[1].GetValue(result_schema, 0).integer_, 22);
    EXPECT_EQ(results[2].GetValue(result_schema, 0).integer_, 33);
}

TEST_F(ProjectionExecutorTest, LiteralProjection) {
    auto env = SetUpEnv();
    InsertRow(env, 1, "a", 10);
    InsertRow(env, 2, "b", 20);
    InsertRow(env, 3, "c", 30);

    std::vector<std::unique_ptr<Expression>> exprs;
    exprs.push_back(MakeLiteral(42));

    Schema out_schema({Column("lit", TypeId::INTEGER)});
    Schema result_schema({Column("lit", TypeId::INTEGER)});

    auto results = RunProjection(env, std::move(exprs), std::move(out_schema));
    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0].GetValue(result_schema, 0).integer_, 42);
    EXPECT_EQ(results[1].GetValue(result_schema, 0).integer_, 42);
    EXPECT_EQ(results[2].GetValue(result_schema, 0).integer_, 42);
}

TEST_F(ProjectionExecutorTest, EmptyInput) {
    auto env = SetUpEnv();

    std::vector<std::unique_ptr<Expression>> exprs;
    exprs.push_back(MakeColRef("id"));

    Schema out_schema({Column("id", TypeId::INTEGER)});

    auto results = RunProjection(env, std::move(exprs), std::move(out_schema));
    EXPECT_TRUE(results.empty());
}

TEST_F(ProjectionExecutorTest, SingleRow) {
    auto env = SetUpEnv();
    InsertRow(env, 1, "a", 10);

    std::vector<std::unique_ptr<Expression>> exprs;
    exprs.push_back(MakeColRef("id"));
    exprs.push_back(MakeColRef("name"));

    Schema out_schema({Column("id", TypeId::INTEGER), Column("name", TypeId::VARCHAR)});
    Schema result_schema({Column("id", TypeId::INTEGER), Column("name", TypeId::VARCHAR)});

    auto results = RunProjection(env, std::move(exprs), std::move(out_schema));
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].GetValue(result_schema, 0).integer_, 1);
    EXPECT_EQ(results[0].GetValue(result_schema, 1).varchar_, "a");
}

TEST_F(ProjectionExecutorTest, IdentityProjection) {
    auto env = SetUpEnv();
    InsertRow(env, 1, "a", 10);
    InsertRow(env, 2, "b", 20);

    std::vector<std::unique_ptr<Expression>> exprs;
    exprs.push_back(MakeColRef("id"));
    exprs.push_back(MakeColRef("name"));
    exprs.push_back(MakeColRef("score"));

    Schema out_schema({
        Column("id", TypeId::INTEGER),
        Column("name", TypeId::VARCHAR),
        Column("score", TypeId::INTEGER)
    });
    Schema result_schema({
        Column("id", TypeId::INTEGER),
        Column("name", TypeId::VARCHAR),
        Column("score", TypeId::INTEGER)
    });

    auto results = RunProjection(env, std::move(exprs), std::move(out_schema));
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0].GetValue(result_schema, 0).integer_, 1);
    EXPECT_EQ(results[0].GetValue(result_schema, 1).varchar_, "a");
    EXPECT_EQ(results[0].GetValue(result_schema, 2).integer_, 10);
    EXPECT_EQ(results[1].GetValue(result_schema, 0).integer_, 2);
    EXPECT_EQ(results[1].GetValue(result_schema, 1).varchar_, "b");
    EXPECT_EQ(results[1].GetValue(result_schema, 2).integer_, 20);
}

TEST_F(ProjectionExecutorTest, ReinitResetsProjection) {
    auto env = SetUpEnv();
    InsertRow(env, 1, "a", 10);
    InsertRow(env, 2, "b", 20);

    auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");

    std::vector<std::unique_ptr<Expression>> exprs;
    exprs.push_back(MakeColRef("id"));
    Schema out_schema({Column("id", TypeId::INTEGER)});
    auto proj_plan = std::make_unique<ProjectionPlanNode>(std::move(out_schema), std::move(exprs));

    auto child = std::make_unique<SeqScanExecutor>(scan_plan.get(), &env.ctx);
    ProjectionExecutor proj(proj_plan.get(), &env.ctx, std::move(child));

    proj.Init();
    Tuple result;
    int count1 = 0;
    while (proj.Next(&result)) { ++count1; }
    EXPECT_EQ(count1, 2);

    proj.Init();
    int count2 = 0;
    while (proj.Next(&result)) { ++count2; }
    EXPECT_EQ(count2, 2);
    proj.Close();
}

TEST_F(ProjectionExecutorTest, NextBeforeInitAsserts) {
    auto env = SetUpEnv();
    auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");

    std::vector<std::unique_ptr<Expression>> exprs;
    exprs.push_back(MakeColRef("id"));
    Schema out_schema({Column("id", TypeId::INTEGER)});
    auto proj_plan = std::make_unique<ProjectionPlanNode>(std::move(out_schema), std::move(exprs));

    auto child = std::make_unique<SeqScanExecutor>(scan_plan.get(), &env.ctx);
    ProjectionExecutor proj(proj_plan.get(), &env.ctx, std::move(child));

    Tuple t;
    EXPECT_DEATH(proj.Next(&t), "Next.*called before Init");
}

}  // namespace shilmandb
