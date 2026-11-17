#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "buffer/buffer_pool_manager.hpp"
#include "catalog/catalog.hpp"
#include "engine/database.hpp"
#include "executor/columnar_scan_executor.hpp"
#include "executor/data_chunk.hpp"
#include "executor/executor_context.hpp"
#include "executor/filter_executor.hpp"
#include "executor/vectorized_filter_executor.hpp"
#include "parser/parser.hpp"
#include "planner/plan_node.hpp"
#include "planner/planner.hpp"

namespace shilmandb {
namespace {

constexpr size_t kGroupRows = 65536;
constexpr size_t kGroups = 5;
constexpr size_t kRows = kGroupRows * kGroups;  // 327,680

using Op = BinaryOp::Op;

std::unique_ptr<Expression> Col(const std::string& name) {
    auto c = std::make_unique<ColumnRef>();
    c->column_name = name;
    return c;
}

std::unique_ptr<Expression> Lit(Value v) {
    auto l = std::make_unique<Literal>();
    l->value = std::move(v);
    return l;
}

// col op v, with an INTEGER literal.
std::unique_ptr<Expression> Cmp(const std::string& col, Op op, int32_t v) {
    auto b = std::make_unique<BinaryOp>();
    b->op = op;
    b->left = Col(col);
    b->right = Lit(Value(v));
    return b;
}

// Splits nested ANDs into their leaves.
void Leaves(const Expression* e, std::vector<const Expression*>& out) {
    const auto* bin = dynamic_cast<const BinaryOp*>(e);
    if (bin && bin->op == Op::AND) {
        Leaves(bin->left.get(), out);
        Leaves(bin->right.get(), out);
    } else {
        out.push_back(e);
    }
}

// Same shape, names and values; literals compare as the engine compares them.
bool SameExpr(const Expression* a, const Expression* b) {
    if (a == nullptr || b == nullptr) return a == b;
    if (a->type != b->type) return false;
    switch (a->type) {
        case ExprType::COLUMN_REF: {
            const auto& x = static_cast<const ColumnRef&>(*a);
            const auto& y = static_cast<const ColumnRef&>(*b);
            return x.table_name == y.table_name && x.column_name == y.column_name;
        }
        case ExprType::LITERAL:
            return static_cast<const Literal&>(*a).value == static_cast<const Literal&>(*b).value;
        case ExprType::BINARY_OP: {
            const auto& x = static_cast<const BinaryOp&>(*a);
            const auto& y = static_cast<const BinaryOp&>(*b);
            return x.op == y.op && SameExpr(x.left.get(), y.left.get()) && SameExpr(x.right.get(), y.right.get());
        }
        case ExprType::UNARY_OP: {
            const auto& x = static_cast<const UnaryOp&>(*a);
            const auto& y = static_cast<const UnaryOp&>(*b);
            return x.op == y.op && SameExpr(x.operand.get(), y.operand.get());
        }
        default:
            return false;
    }
}

class ColumnarSkipTest : public ::testing::Test {
protected:
    std::string db_file_;
    std::vector<std::string> tbl_files_;
    std::unique_ptr<Database> db_;
    std::unique_ptr<ExecutorContext> ctx_;

    void SetUp() override {
        db_file_ = TempPath(".db");
        std::filesystem::remove(db_file_);

        // k is the line number minus 1: each row group holds one stretch of k.
        std::string text;
        text.reserve(kRows * 16);
        for (size_t k = 0; k < kRows; ++k) text += std::to_string(k) + "|" + std::to_string(k % 100) + "\n";
        const std::string tbl = WriteTbl("t", text);

        LoadOptions columnar;
        columnar.storage = StorageKind::COLUMNAR;
        columnar.threads = 2;

        db_ = std::make_unique<Database>(db_file_);
        db_->LoadTable("t", Schema({Column("k", ColumnType::BigInt()), Column("v", ColumnType::Integer())}), tbl, columnar);
        ctx_ = std::make_unique<ExecutorContext>(ExecutorContext{db_->GetBufferPoolManager(), db_->GetCatalog()});
        ASSERT_EQ(db_->GetCatalog()->GetTable("t")->columnar->RowGroupCount(), kGroups);
    }

    void TearDown() override {
        ctx_.reset();
        db_.reset();
        std::filesystem::remove(db_file_);
        for (const auto& f : tbl_files_) std::filesystem::remove(f);
    }

    static std::string TempPath(const std::string& suffix) {
        return (std::filesystem::temp_directory_path() /
                ("shilmandb_columnar_skip_test_" + std::to_string(::getpid()) + suffix)).string();
    }

    std::string WriteTbl(const std::string& what, const std::string& text) {
        const std::string path = TempPath("_" + what + ".tbl");
        std::ofstream(path, std::ios::binary) << text;
        tbl_files_.push_back(path);
        return path;
    }

    const Schema& SchemaOf(const std::string& table) { return db_->GetCatalog()->GetTable(table)->schema; }

    // Pages of column col in the given row groups.
    uint64_t PagesOf(uint32_t col, size_t from_group, size_t to_group) {
        const auto& meta = db_->GetCatalog()->GetTable("t")->columnar->Meta();
        uint64_t pages = 0;
        for (size_t g = from_group; g < to_group; ++g) pages += meta.row_groups[g].chunks[col].page_count;
        return pages;
    }

    struct Run {
        size_t rows{0};
        size_t skipped{0};
        uint64_t misses{0};
    };

    // A Filter over a cold scan of t reading only k, skip hints if asked.
    Run Count(Op op, int32_t v, bool skip, ExecutionMode mode) {
        auto scan = std::make_unique<SeqScanPlanNode>(SchemaOf("t"), "t");
        scan->read_columns = std::vector<uint32_t>{0};
        if (skip) scan->skip_conjuncts.push_back(Cmp("k", op, v));
        FilterPlanNode filter(SchemaOf("t"), Cmp("k", op, v));
        filter.children.push_back(std::move(scan));

        auto* bpm = db_->GetBufferPoolManager();
        bpm->DropAllPages();
        bpm->ResetStats();

        auto scan_exec = std::make_unique<ColumnarScanExecutor>(filter.children[0].get(), ctx_.get());
        const ColumnarScanExecutor* raw = scan_exec.get();
        Run run;
        if (mode == ExecutionMode::TUPLE) {
            FilterExecutor exec(&filter, ctx_.get(), std::move(scan_exec));
            exec.Init();
            Tuple t;
            while (exec.Next(&t)) ++run.rows;
            run.skipped = raw->SkippedRowGroups();
            exec.Close();
        } else {
            VectorizedFilterExecutor exec(&filter, ctx_.get(), std::move(scan_exec));
            exec.Init();
            DataChunk chunk(SchemaOf("t"));
            while (exec.NextBatch(&chunk)) run.rows += chunk.size();
            run.skipped = raw->SkippedRowGroups();
            exec.Close();
        }
        run.misses = bpm->GetMissCount();
        return run;
    }

    std::unique_ptr<PlanNode> PlanSql(const std::string& sql) {
        Parser parser(sql);
        auto stmt = parser.Parse();
        Planner planner(db_->GetCatalog());
        return planner.Plan(std::move(*stmt));
    }
};

constexpr ExecutionMode kModes[] = {ExecutionMode::TUPLE, ExecutionMode::VECTORIZED};

TEST_F(ColumnarSkipTest, SkipsGroupsBelowARange) {
    for (const auto mode : kModes) {
        const Run skipping = Count(Op::GTE, 200000, true, mode);
        const Run full = Count(Op::GTE, 200000, false, mode);
        EXPECT_EQ(skipping.rows, 127680u);
        EXPECT_EQ(full.rows, skipping.rows);
        EXPECT_EQ(skipping.skipped, 3u);
        EXPECT_EQ(full.skipped, 0u);
        // A skipped group reads no page.
        EXPECT_EQ(skipping.misses, PagesOf(0, 3, kGroups));
        EXPECT_EQ(full.misses, PagesOf(0, 0, kGroups));
        EXPECT_LT(skipping.misses, full.misses);
    }
}

TEST_F(ColumnarSkipTest, SkipsEveryGroupForAMissingKey) {
    for (const auto mode : kModes) {
        const Run skipping = Count(Op::EQ, -1, true, mode);
        EXPECT_EQ(skipping.rows, 0u);
        EXPECT_EQ(skipping.skipped, kGroups);
        EXPECT_EQ(skipping.misses, 0u);
        EXPECT_EQ(Count(Op::EQ, -1, false, mode).rows, 0u);
    }
}

TEST_F(ColumnarSkipTest, KeepsEveryGroupThatMayMatch) {
    for (const auto mode : kModes) {
        const Run skipping = Count(Op::LT, 300000, true, mode);
        EXPECT_EQ(skipping.rows, 300000u);
        EXPECT_EQ(skipping.skipped, 0u);
    }
}

TEST_F(ColumnarSkipTest, SqlSkipsThroughThePlanner) {
    const auto plan = PlanSql("SELECT COUNT(*) FROM t WHERE k >= 200000");
    const PlanNode* node = plan.get();
    while (!node->children.empty()) node = node->children[0].get();
    ASSERT_EQ(node->type, PlanNodeType::SEQ_SCAN);
    const auto* scan = static_cast<const SeqScanPlanNode*>(node);
    EXPECT_FALSE(scan->skip_conjuncts.empty());
    ASSERT_TRUE(scan->read_columns.has_value());
    EXPECT_EQ(*scan->read_columns, std::vector<uint32_t>{0});

    auto* bpm = db_->GetBufferPoolManager();
    for (const auto mode : kModes) {
        bpm->DropAllPages();
        bpm->ResetStats();
        const auto count = db_->ExecuteSQL("SELECT COUNT(*) FROM t WHERE k >= 200000", mode);
        ASSERT_EQ(count.tuples.size(), 1u);
        EXPECT_EQ(count.tuples[0].GetValue(count.schema, 0).integer_, 127680);
        EXPECT_EQ(bpm->GetMissCount(), PagesOf(0, 3, kGroups));

        // -1 parses as a minus sign on 1; every group is still skipped.
        bpm->DropAllPages();
        bpm->ResetStats();
        const auto none = db_->ExecuteSQL("SELECT COUNT(*) FROM t WHERE k = -1", mode);
        ASSERT_EQ(none.tuples.size(), 1u);
        EXPECT_EQ(none.tuples[0].GetValue(none.schema, 0).integer_, 0);
        EXPECT_EQ(bpm->GetMissCount(), 0u);

        const auto sum = db_->ExecuteSQL("SELECT SUM(v), COUNT(*) FROM t WHERE k >= 65530 AND k < 65542", mode);
        ASSERT_EQ(sum.tuples.size(), 1u);
        EXPECT_EQ(sum.tuples[0].GetValue(sum.schema, 1).integer_, 12);
    }
}

TEST_F(ColumnarSkipTest, FilterStaysAboveSkippingScan) {
    std::string text;
    for (int i = 0; i < 1000; ++i) text += std::to_string(i * 300) + "|" + std::to_string(i % 70) + "\n";
    db_->LoadTable("h", Schema({Column("hk", ColumnType::BigInt()), Column("hv", ColumnType::Integer())}), WriteTbl("h", text));

    const auto plan = PlanSql(
        "SELECT COUNT(*) FROM t JOIN h ON t.k = h.hk "
        "WHERE t.k >= 200000 AND h.hv < 50 AND t.v < 3 AND h.hk BETWEEN 1000 AND 280000");

    size_t skipping_scans = 0;
    std::vector<const PlanNode*> stack = {plan.get()};
    while (!stack.empty()) {
        const PlanNode* node = stack.back();
        stack.pop_back();
        ASSERT_NE(node->type, PlanNodeType::SEQ_SCAN) << "the root is a scan";
        for (const auto& child : node->children) {
            if (child->type == PlanNodeType::SEQ_SCAN) {
                const auto* scan = static_cast<const SeqScanPlanNode*>(child.get());
                if (scan->skip_conjuncts.empty()) continue;
                ++skipping_scans;
                ASSERT_EQ(node->type, PlanNodeType::FILTER) << scan->table_name;
                std::vector<const Expression*> filter_leaves;
                Leaves(static_cast<const FilterPlanNode*>(node)->predicate.get(), filter_leaves);
                for (const auto& conj : scan->skip_conjuncts) {
                    std::vector<const Expression*> skip_leaves;
                    Leaves(conj.get(), skip_leaves);
                    for (const auto* leaf : skip_leaves) {
                        const bool found = std::any_of(filter_leaves.begin(), filter_leaves.end(),
                                                       [&](const Expression* f) { return SameExpr(leaf, f); });
                        EXPECT_TRUE(found) << scan->table_name << ": a skip hint is not in the filter above";
                    }
                }
            } else {
                stack.push_back(child.get());
            }
        }
    }
    EXPECT_EQ(skipping_scans, 2u);

    for (const auto mode : kModes) {
        const auto result = db_->ExecuteSQL(
            "SELECT COUNT(*) FROM t JOIN h ON t.k = h.hk "
            "WHERE t.k >= 200000 AND h.hv < 50 AND t.v < 3 AND h.hk BETWEEN 1000 AND 280000",
            mode);
        ASSERT_EQ(result.tuples.size(), 1u);
        // Matches: hk = 300 i, i in [667, 933], i % 70 < 50.
        // Every such t.v is 0, so t.v < 3 keeps them all.
        EXPECT_EQ(result.tuples[0].GetValue(result.schema, 0).integer_, 187);
    }
}

}  // namespace
}  // namespace shilmandb
