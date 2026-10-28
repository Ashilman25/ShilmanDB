#include <gtest/gtest.h>
#include <unistd.h>

#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "catalog/catalog.hpp"
#include "executor/batch_to_tuple_adapter.hpp"
#include "executor/filter_executor.hpp"
#include "executor/seq_scan_executor.hpp"
#include "executor/vectorized_filter_executor.hpp"
#include "executor/vectorized_seq_scan_executor.hpp"
#include "parser/ast.hpp"
#include "planner/plan_node.hpp"
#include "storage/disk_manager.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace shilmandb {

// --- AST builder helpers (duplicated per project convention) ---

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

// --- Fixture ---

class BatchToTupleAdapterTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() /
                      ("shilmandb_batch_to_tuple_adapter_test_" + std::to_string(::getpid()) + ".db")).string();
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
            Column("val", TypeId::INTEGER),
        });
    }

    struct TestEnv {
        BPMBundle bundle;
        std::unique_ptr<Catalog> catalog;
        ExecutorContext ctx;
        Schema schema;
    };

    TestEnv SetUpEnv(int num_rows) {
        auto bundle = MakeBPM(test_file_);
        auto catalog = std::make_unique<Catalog>(bundle.bpm.get());
        auto schema = MakeInputSchema();
        auto* table_info = catalog->CreateTable("t", schema);
        for (int i = 0; i < num_rows; ++i) {
            std::vector<Value> vals = {
                Value(static_cast<int32_t>(i)),
                Value(static_cast<int32_t>(i * 10)),
            };
            (void)table_info->table->InsertTuple(Tuple(vals, schema));
        }
        ExecutorContext ctx{bundle.bpm.get(), catalog.get()};
        return {std::move(bundle), std::move(catalog), ctx, schema};
    }
};

// --- Tests ---

TEST_F(BatchToTupleAdapterTest, RoundTripOrder) {
    auto env = SetUpEnv(/*num_rows=*/16);
    auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");

    auto vec_scan = std::make_unique<VectorizedSeqScanExecutor>(scan_plan.get(), &env.ctx);
    auto adapter = std::make_unique<BatchToTupleAdapter>(
        scan_plan.get(), &env.ctx, std::move(vec_scan));
    adapter->Init();
    std::vector<Tuple> via_adapter;
    Tuple t;
    while (adapter->Next(&t)) via_adapter.push_back(t);
    adapter->Close();

    auto tuple_scan = std::make_unique<SeqScanExecutor>(scan_plan.get(), &env.ctx);
    tuple_scan->Init();
    std::vector<Tuple> via_tuple;
    while (tuple_scan->Next(&t)) via_tuple.push_back(t);
    tuple_scan->Close();

    ASSERT_EQ(via_adapter.size(), via_tuple.size());
    ASSERT_EQ(via_adapter.size(), 16u);
    for (size_t r = 0; r < via_adapter.size(); ++r) {
        for (uint32_t c = 0; c < env.schema.GetColumnCount(); ++c) {
            EXPECT_TRUE(via_adapter[r].GetValue(env.schema, c) ==
                        via_tuple[r].GetValue(env.schema, c))
                << "row " << r << " col " << c;
        }
    }
}

TEST_F(BatchToTupleAdapterTest, HonorsSelectionVector) {
    auto env = SetUpEnv(/*num_rows=*/100);

    auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");
    // predicate: id > 49 — survivors: 50 rows (ids 50..99)
    auto pred = MakeBinOp(BinaryOp::Op::GT, MakeColRef("id"), MakeLiteral(49));
    auto filter_plan = std::make_unique<FilterPlanNode>(env.schema, std::move(pred));
    filter_plan->children.push_back(std::move(scan_plan));

    auto vec_scan = std::make_unique<VectorizedSeqScanExecutor>(
        filter_plan->children[0].get(), &env.ctx);
    auto vec_filter = std::make_unique<VectorizedFilterExecutor>(
        filter_plan.get(), &env.ctx, std::move(vec_scan));
    auto adapter = std::make_unique<BatchToTupleAdapter>(
        filter_plan.get(), &env.ctx, std::move(vec_filter));
    adapter->Init();
    std::vector<Tuple> via_adapter;
    Tuple t;
    while (adapter->Next(&t)) via_adapter.push_back(t);
    adapter->Close();

    // Tuple-mode baseline over the same plan
    auto tuple_scan = std::make_unique<SeqScanExecutor>(
        filter_plan->children[0].get(), &env.ctx);
    auto tuple_filter = std::make_unique<FilterExecutor>(
        filter_plan.get(), &env.ctx, std::move(tuple_scan));
    tuple_filter->Init();
    std::vector<Tuple> via_tuple;
    while (tuple_filter->Next(&t)) via_tuple.push_back(t);
    tuple_filter->Close();

    ASSERT_EQ(via_adapter.size(), via_tuple.size());
    ASSERT_EQ(via_adapter.size(), 50u);
    for (size_t r = 0; r < via_adapter.size(); ++r) {
        for (uint32_t c = 0; c < env.schema.GetColumnCount(); ++c) {
            EXPECT_TRUE(via_adapter[r].GetValue(env.schema, c) ==
                        via_tuple[r].GetValue(env.schema, c))
                << "row " << r << " col " << c;
        }
    }
}

TEST_F(BatchToTupleAdapterTest, EmptyChild) {
    auto env = SetUpEnv(/*num_rows=*/0);
    auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");

    auto vec_scan = std::make_unique<VectorizedSeqScanExecutor>(scan_plan.get(), &env.ctx);
    auto adapter = std::make_unique<BatchToTupleAdapter>(
        scan_plan.get(), &env.ctx, std::move(vec_scan));
    adapter->Init();
    Tuple t;
    EXPECT_FALSE(adapter->Next(&t));
    EXPECT_FALSE(adapter->Next(&t));  // idempotent EOF
    adapter->Close();
}

TEST_F(BatchToTupleAdapterTest, ExactlyOneBatch) {
    auto env = SetUpEnv(/*num_rows=*/1024);  // DataChunk::kDefaultBatchSize
    auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");

    auto vec_scan = std::make_unique<VectorizedSeqScanExecutor>(scan_plan.get(), &env.ctx);
    auto adapter = std::make_unique<BatchToTupleAdapter>(
        scan_plan.get(), &env.ctx, std::move(vec_scan));
    adapter->Init();
    size_t count = 0;
    Tuple t;
    while (adapter->Next(&t)) ++count;
    adapter->Close();
    EXPECT_EQ(count, 1024u);
}

TEST_F(BatchToTupleAdapterTest, MultipleBatches) {
    auto env = SetUpEnv(/*num_rows=*/2500);  // spans 3 batches (1024 + 1024 + 452)
    auto scan_plan = std::make_unique<SeqScanPlanNode>(env.schema, "t");

    auto vec_scan = std::make_unique<VectorizedSeqScanExecutor>(scan_plan.get(), &env.ctx);
    auto adapter = std::make_unique<BatchToTupleAdapter>(
        scan_plan.get(), &env.ctx, std::move(vec_scan));
    adapter->Init();
    std::vector<Tuple> via_adapter;
    Tuple t;
    while (adapter->Next(&t)) via_adapter.push_back(t);
    adapter->Close();
    ASSERT_EQ(via_adapter.size(), 2500u);

    auto tuple_scan = std::make_unique<SeqScanExecutor>(scan_plan.get(), &env.ctx);
    tuple_scan->Init();
    std::vector<Tuple> via_tuple;
    while (tuple_scan->Next(&t)) via_tuple.push_back(t);
    tuple_scan->Close();
    ASSERT_EQ(via_tuple.size(), 2500u);
    for (size_t r = 0; r < 2500u; ++r) {
        EXPECT_TRUE(via_adapter[r].GetValue(env.schema, 0) ==
                    via_tuple[r].GetValue(env.schema, 0))
            << "row " << r;
    }
}

}  // namespace shilmandb
