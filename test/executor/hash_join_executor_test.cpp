#include <gtest/gtest.h>
#include <unistd.h>
#include "executor/hash_join_executor.hpp"
#include "executor/profiling_executor.hpp"
#include "executor/seq_scan_executor.hpp"
#include "planner/plan_node.hpp"
#include "catalog/catalog.hpp"
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "storage/disk_manager.hpp"
#include <filesystem>
#include <memory>
#include <algorithm>
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

static auto MakeAndPred(std::unique_ptr<Expression> lhs, std::unique_ptr<Expression> rhs) {
    auto e = std::make_unique<BinaryOp>();
    e->op = BinaryOp::Op::AND;
    e->left = std::move(lhs);
    e->right = std::move(rhs);
    return e;
}

// --- Test fixture ---

class HashJoinExecutorTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() /
                      ("shilmandb_hash_join_test_" + std::to_string(::getpid()) + ".db")).string();
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

    using Rows = std::vector<std::vector<Value>>;

    struct JoinRun {
        Rows rows;
        bool builds_right;
    };

    // Loads both tables, joins them with scan children and drains every output row.
    JoinRun RunJoin(const Schema& left_schema, const Rows& left_rows,
                    const Schema& right_schema, const Rows& right_rows,
                    std::unique_ptr<Expression> pred) {
        auto bundle = MakeBPM(test_file_);
        Catalog catalog(bundle.bpm.get());
        auto* left_info = catalog.CreateTable("left_t", left_schema);
        auto* right_info = catalog.CreateTable("right_t", right_schema);
        for (const auto& r : left_rows) (void)left_info->table->InsertTuple(Tuple(r, left_schema));
        for (const auto& r : right_rows) (void)right_info->table->InsertTuple(Tuple(r, right_schema));

        std::vector<Column> out_cols;
        for (uint32_t i = 0; i < left_schema.GetColumnCount(); ++i) out_cols.push_back(left_schema.GetColumn(i));
        for (uint32_t i = 0; i < right_schema.GetColumnCount(); ++i) out_cols.push_back(right_schema.GetColumn(i));
        Schema output_schema(out_cols);

        auto left_scan = std::make_unique<SeqScanPlanNode>(left_schema, "left_t");
        auto right_scan = std::make_unique<SeqScanPlanNode>(right_schema, "right_t");
        auto join_plan = std::make_unique<HashJoinPlanNode>(output_schema, std::move(pred));

        ExecutorContext ctx{bundle.bpm.get(), &catalog};
        HashJoinExecutor join_exec(join_plan.get(), &ctx,
                                   std::make_unique<SeqScanExecutor>(left_scan.get(), &ctx),
                                   std::make_unique<SeqScanExecutor>(right_scan.get(), &ctx));
        join_exec.Init();
        JoinRun run{{}, join_exec.BuildsFromRight()};
        Tuple t;
        while (join_exec.Next(&t)) {
            std::vector<Value> row;
            for (uint32_t i = 0; i < output_schema.GetColumnCount(); ++i) row.push_back(t.GetValue(output_schema, i));
            run.rows.push_back(std::move(row));
        }
        join_exec.Close();
        return run;
    }

    // Rows as sorted strings, so tests don't depend on output order.
    static std::vector<std::string> Render(const Rows& rows) {
        std::vector<std::string> out;
        for (const auto& r : rows) {
            std::string s;
            for (const auto& v : r) s += v.ToString() + "|";
            out.push_back(std::move(s));
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    static Schema LSchema() { return Schema({Column("lid", TypeId::INTEGER), Column("lval", TypeId::VARCHAR)}); }
    static Schema RSchema() { return Schema({Column("rid", TypeId::INTEGER), Column("rval", TypeId::VARCHAR)}); }

    static Rows LRows(std::initializer_list<int> keys) {
        Rows rows;
        for (int k : keys) rows.push_back({Value(k), Value("l" + std::to_string(k))});
        return rows;
    }
    static Rows RRows(std::initializer_list<int> keys) {
        Rows rows;
        for (int k : keys) rows.push_back({Value(k), Value("r" + std::to_string(k))});
        return rows;
    }
    static Rows Range(int lo, int hi, bool left) {
        Rows rows;
        for (int k = lo; k <= hi; ++k) {
            rows.push_back({Value(k), Value((left ? "l" : "r") + std::to_string(k))});
        }
        return rows;
    }

    static std::string Expected(int k) {
        return std::to_string(k) + "|l" + std::to_string(k) + "|" + std::to_string(k) + "|r" + std::to_string(k) + "|";
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

// --- Build side follows the smaller input ---

TEST_F(HashJoinExecutorTest, BuildsFromRightWhenRightIsSmaller) {
    auto run = RunJoin(LSchema(), Range(1, 10, true), RSchema(), RRows({3, 7}), MakeEqPred("lid", "rid"));
    EXPECT_TRUE(run.builds_right);
    EXPECT_EQ(Render(run.rows), (std::vector<std::string>{Expected(3), Expected(7)}));
}

TEST_F(HashJoinExecutorTest, BuildsFromLeftWhenLeftIsSmaller) {
    auto run = RunJoin(LSchema(), LRows({3, 7}), RSchema(), Range(1, 10, false), MakeEqPred("lid", "rid"));
    EXPECT_FALSE(run.builds_right);
    EXPECT_EQ(Render(run.rows), (std::vector<std::string>{Expected(3), Expected(7)}));
}

TEST_F(HashJoinExecutorTest, EqualSizesBuildFromLeft) {
    auto run = RunJoin(LSchema(), Range(1, 3, true), RSchema(), Range(2, 4, false), MakeEqPred("lid", "rid"));
    EXPECT_FALSE(run.builds_right);
    EXPECT_EQ(Render(run.rows), (std::vector<std::string>{Expected(2), Expected(3)}));
}

TEST_F(HashJoinExecutorTest, DuplicateKeysOnRightWhenBuildingRight) {
    // Same key, different second column, so every output pair is told apart.
    Rows left = {{Value(1), Value("La")}, {Value(1), Value("Lb")}, {Value(2), Value("Lc")},
                 {Value(2), Value("Ld")}, {Value(3), Value("Le")}, {Value(4), Value("Lf")}};
    Rows right = {{Value(1), Value("Ra")}, {Value(1), Value("Rb")}, {Value(2), Value("Rc")}};

    auto run = RunJoin(LSchema(), left, RSchema(), right, MakeEqPred("lid", "rid"));
    EXPECT_TRUE(run.builds_right);
    ASSERT_EQ(run.rows.size(), 6u);
    EXPECT_EQ(Render(run.rows), (std::vector<std::string>{
        "1|La|1|Ra|", "1|La|1|Rb|", "1|Lb|1|Ra|", "1|Lb|1|Rb|", "2|Lc|2|Rc|", "2|Ld|2|Rc|"}));
}

TEST_F(HashJoinExecutorTest, EmptyRightSideBuildsRight) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());
    auto ls = LSchema();
    auto rs = RSchema();
    auto* left_info = catalog.CreateTable("left_t", ls);
    (void)catalog.CreateTable("right_t", rs);
    for (const auto& r : Range(1, 3, true)) (void)left_info->table->InsertTuple(Tuple(r, ls));

    Schema out({Column("lid", TypeId::INTEGER), Column("lval", TypeId::VARCHAR),
                Column("rid", TypeId::INTEGER), Column("rval", TypeId::VARCHAR)});
    auto left_scan = std::make_unique<SeqScanPlanNode>(ls, "left_t");
    auto right_scan = std::make_unique<SeqScanPlanNode>(rs, "right_t");
    auto join_plan = std::make_unique<HashJoinPlanNode>(out, MakeEqPred("lid", "rid"));
    OperatorStatsMap stats;
    ExecutorContext ctx{bundle.bpm.get(), &catalog, &stats};
    HashJoinExecutor join_exec(join_plan.get(), &ctx,
                               std::make_unique<ProfilingExecutor>(
                                   left_scan.get(), &ctx, std::make_unique<SeqScanExecutor>(left_scan.get(), &ctx)),
                               std::make_unique<SeqScanExecutor>(right_scan.get(), &ctx));
    join_exec.Init();
    EXPECT_TRUE(join_exec.BuildsFromRight());
    Tuple t;
    int count = 0;
    while (join_exec.Next(&t)) ++count;
    join_exec.Close();
    EXPECT_EQ(count, 0);
    // The probe side is still read to the end.
    EXPECT_EQ(stats[left_scan.get()].rows, 3u);
}

TEST_F(HashJoinExecutorTest, EmptyLeftSideBuildsLeft) {
    auto run = RunJoin(LSchema(), {}, RSchema(), Range(1, 3, false), MakeEqPred("lid", "rid"));
    EXPECT_FALSE(run.builds_right);
    EXPECT_TRUE(run.rows.empty());
}

TEST_F(HashJoinExecutorTest, ProbeBufferHandoffBuildingRight) {
    auto run = RunJoin(LSchema(), Range(1, 100, true), RSchema(), RRows({2, 50, 99}), MakeEqPred("lid", "rid"));
    EXPECT_TRUE(run.builds_right);
    EXPECT_EQ(Render(run.rows), (std::vector<std::string>{Expected(2), Expected(50), Expected(99)}));
}

TEST_F(HashJoinExecutorTest, ProbeBufferHandoffBuildingLeft) {
    auto run = RunJoin(LSchema(), LRows({2, 50, 99}), RSchema(), Range(1, 100, false), MakeEqPred("lid", "rid"));
    EXPECT_FALSE(run.builds_right);
    EXPECT_EQ(Render(run.rows), (std::vector<std::string>{Expected(2), Expected(50), Expected(99)}));
}

TEST_F(HashJoinExecutorTest, TwoEqualitiesBuildingRight) {
    // The left side lists b before a, so the two key columns sit at different positions.
    Schema ls({Column("b", TypeId::INTEGER), Column("a", TypeId::INTEGER)});
    Schema rs({Column("c", TypeId::INTEGER), Column("d", TypeId::INTEGER)});
    Rows left, right;
    for (auto [b, a] : std::vector<std::pair<int, int>>{{1, 1}, {2, 1}, {1, 2}, {2, 2}, {3, 3}, {4, 4}}) {
        left.push_back({Value(b), Value(a)});
    }
    for (auto [c, d] : std::vector<std::pair<int, int>>{{1, 2}, {2, 2}}) right.push_back({Value(c), Value(d)});

    auto run = RunJoin(ls, left, rs, right, MakeAndPred(MakeEqPred("c", "a"), MakeEqPred("b", "d")));
    EXPECT_TRUE(run.builds_right);
    EXPECT_EQ(Render(run.rows), (std::vector<std::string>{"2|1|1|2|", "2|2|2|2|"}));
}

TEST_F(HashJoinExecutorTest, VarcharKeysBuildingRight) {
    Schema ls({Column("lname", TypeId::VARCHAR), Column("lid", TypeId::INTEGER)});
    Schema rs({Column("rid", TypeId::INTEGER), Column("rname", TypeId::VARCHAR)});
    Rows left, right;
    int id = 1;
    for (const char* n : {"ann", "bob", "cat", "dan"}) left.push_back({Value(std::string(n)), Value(id++)});
    right.push_back({Value(20), Value(std::string("bob"))});
    right.push_back({Value(40), Value(std::string("dan"))});

    auto run = RunJoin(ls, left, rs, right, MakeEqPred("lname", "rname"));
    EXPECT_TRUE(run.builds_right);
    EXPECT_EQ(Render(run.rows), (std::vector<std::string>{"bob|2|20|bob|", "dan|4|40|dan|"}));
}

// --- Lifecycle ---

TEST_F(HashJoinExecutorTest, ReinitAfterCloseGivesSameRows) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());
    auto ls = LSchema();
    auto rs = RSchema();
    auto* left_info = catalog.CreateTable("left_t", ls);
    auto* right_info = catalog.CreateTable("right_t", rs);
    for (const auto& r : Range(1, 10, true)) (void)left_info->table->InsertTuple(Tuple(r, ls));
    for (const auto& r : RRows({3, 7})) (void)right_info->table->InsertTuple(Tuple(r, rs));

    Schema out({Column("lid", TypeId::INTEGER), Column("lval", TypeId::VARCHAR),
                Column("rid", TypeId::INTEGER), Column("rval", TypeId::VARCHAR)});
    auto left_scan = std::make_unique<SeqScanPlanNode>(ls, "left_t");
    auto right_scan = std::make_unique<SeqScanPlanNode>(rs, "right_t");
    auto join_plan = std::make_unique<HashJoinPlanNode>(out, MakeEqPred("lid", "rid"));
    ExecutorContext ctx{bundle.bpm.get(), &catalog};
    HashJoinExecutor join_exec(join_plan.get(), &ctx,
                               std::make_unique<SeqScanExecutor>(left_scan.get(), &ctx),
                               std::make_unique<SeqScanExecutor>(right_scan.get(), &ctx));

    for (int round = 0; round < 2; ++round) {
        join_exec.Init();
        EXPECT_TRUE(join_exec.BuildsFromRight());
        Rows rows;
        Tuple t;
        while (join_exec.Next(&t)) {
            rows.push_back({t.GetValue(out, 0), t.GetValue(out, 1), t.GetValue(out, 2), t.GetValue(out, 3)});
        }
        join_exec.Close();
        EXPECT_EQ(Render(rows), (std::vector<std::string>{Expected(3), Expected(7)}));
    }
}

TEST_F(HashJoinExecutorTest, CloseMidStreamIsSafe) {
    auto bundle = MakeBPM(test_file_);
    Catalog catalog(bundle.bpm.get());
    auto ls = LSchema();
    auto rs = RSchema();
    auto* left_info = catalog.CreateTable("left_t", ls);
    auto* right_info = catalog.CreateTable("right_t", rs);
    for (const auto& r : Range(1, 100, true)) (void)left_info->table->InsertTuple(Tuple(r, ls));
    for (const auto& r : Range(1, 50, false)) (void)right_info->table->InsertTuple(Tuple(r, rs));

    Schema out({Column("lid", TypeId::INTEGER), Column("lval", TypeId::VARCHAR),
                Column("rid", TypeId::INTEGER), Column("rval", TypeId::VARCHAR)});
    auto left_scan = std::make_unique<SeqScanPlanNode>(ls, "left_t");
    auto right_scan = std::make_unique<SeqScanPlanNode>(rs, "right_t");
    auto join_plan = std::make_unique<HashJoinPlanNode>(out, MakeEqPred("lid", "rid"));
    ExecutorContext ctx{bundle.bpm.get(), &catalog};
    HashJoinExecutor join_exec(join_plan.get(), &ctx,
                               std::make_unique<SeqScanExecutor>(left_scan.get(), &ctx),
                               std::make_unique<SeqScanExecutor>(right_scan.get(), &ctx));
    join_exec.Init();
    Tuple t;
    EXPECT_TRUE(join_exec.Next(&t));
    EXPECT_TRUE(join_exec.BuildsFromRight());
    join_exec.Close();
}

}  // namespace shilmandb
