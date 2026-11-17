#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "buffer/buffer_pool_manager.hpp"
#include "catalog/catalog.hpp"
#include "common/exception.hpp"
#include "engine/database.hpp"
#include "executor/data_chunk.hpp"
#include "executor/executor_context.hpp"
#include "executor/executor_factory.hpp"
#include "planner/plan_node.hpp"

namespace shilmandb {
namespace {

constexpr size_t kRows = 3000;

// Runs fn and returns what it threw, or "" if it didn't throw.
template <typename Fn>
std::string ErrorOf(Fn&& fn) {
    try {
        fn();
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

// One column per executor type; price is a DOUBLE, cost a NUMERIC(15,2).
Schema SixColumns() {
    return Schema({Column("id", ColumnType::BigInt()), Column("qty", ColumnType::Integer()),
                   Column("price", ColumnType::Double()), Column("cost", ColumnType::Numeric(15, 2)),
                   Column("ship", ColumnType::Date()), Column("label", ColumnType::Text())});
}

// Row n (1-based) of the shared file: unique ids, negatives, repeated labels.
std::string SixColumnLine(size_t n) {
    static const char* const kLabels[] = {"alpha", "beta", "gamma", "delta"};
    const long long id = static_cast<long long>(n * 7919 % kRows) - 500;
    const int qty = static_cast<int>(n % 97) - 40;
    const int price_whole = static_cast<int>(n * 13 % 2000) - 1000;
    const int price_frac = static_cast<int>(n % 999) + 1;
    const int cost_whole = static_cast<int>(n * 31 % 5000) - 2500;
    const int cost_cents = static_cast<int>(n % 99) + 1;
    const int year = 1992 + static_cast<int>(n % 7);
    const int month = 1 + static_cast<int>(n % 12);
    const int day = 1 + static_cast<int>(n % 28);
    std::string label = kLabels[n % 4];
    if (n % 5 != 0) label += std::to_string(n % 11);

    char buf[160];
    std::snprintf(buf, sizeof(buf), "%lld|%d|%d.%03d|%d.%02d|%04d-%02d-%02d|", id, qty, price_whole, price_frac,
                  cost_whole, cost_cents, year, month, day);
    return std::string(buf) + label + "\n";
}

// Columnar storage on two threads.
LoadOptions Columnar() {
    LoadOptions options;
    options.storage = StorageKind::COLUMNAR;
    options.threads = 2;
    return options;
}

// Checks two values match exactly; doubles compare by their bits.
void ExpectSameValue(const Value& a, const Value& b, const std::string& where) {
    ASSERT_EQ(a.type_, b.type_) << where;
    switch (a.type_) {
        case TypeId::INTEGER: EXPECT_EQ(a.integer_, b.integer_) << where; break;
        case TypeId::BIGINT:  EXPECT_EQ(a.bigint_, b.bigint_) << where; break;
        case TypeId::DATE:    EXPECT_EQ(a.date_, b.date_) << where; break;
        case TypeId::DECIMAL: EXPECT_EQ(std::memcmp(&a.decimal_, &b.decimal_, sizeof(double)), 0) << where; break;
        case TypeId::VARCHAR: EXPECT_EQ(a.varchar_, b.varchar_) << where; break;
        case TypeId::INVALID: ADD_FAILURE() << where << ": INVALID value"; break;
    }
}

// Checks two row lists match row by row, value by value.
void ExpectSameRows(const std::vector<Tuple>& a, const std::vector<Tuple>& b, const Schema& schema) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t r = 0; r < a.size(); ++r) {
        for (uint32_t c = 0; c < schema.GetColumnCount(); ++c) {
            ExpectSameValue(a[r].GetValue(schema, c), b[r].GetValue(schema, c),
                            "row " + std::to_string(r) + " column " + std::to_string(c));
            if (::testing::Test::HasFailure()) return;
        }
    }
}

// Checks two query results hold the same rows in the same order.
void ExpectSameResult(const QueryResult& a, const QueryResult& b) {
    ASSERT_EQ(a.schema.GetColumnCount(), b.schema.GetColumnCount());
    ExpectSameRows(a.tuples, b.tuples, a.schema);
}

class ColumnarScanExecutorTest : public ::testing::Test {
protected:
    std::string db_file_;
    std::vector<std::string> tbl_files_;
    std::unique_ptr<Database> db_;
    std::unique_ptr<ExecutorContext> ctx_;

    void SetUp() override {
        db_file_ = TempPath(".db");
        std::filesystem::remove(db_file_);

        std::string text;
        for (size_t n = 1; n <= kRows; ++n) text += SixColumnLine(n);
        const std::string tbl = WriteTbl("six", text);

        db_ = std::make_unique<Database>(db_file_);
        db_->LoadTable("r", SixColumns(), tbl);
        db_->LoadTable("c", SixColumns(), tbl, Columnar());
        ctx_ = std::make_unique<ExecutorContext>(ExecutorContext{db_->GetBufferPoolManager(), db_->GetCatalog()});
    }

    void TearDown() override {
        ctx_.reset();
        db_.reset();
        std::filesystem::remove(db_file_);
        for (const auto& f : tbl_files_) std::filesystem::remove(f);
    }

    // A temp path unique to this process.
    static std::string TempPath(const std::string& suffix) {
        return (std::filesystem::temp_directory_path() /
                ("shilmandb_columnar_scan_test_" + std::to_string(::getpid()) + suffix)).string();
    }

    // Writes text to a per-test .tbl file and returns its path.
    std::string WriteTbl(const std::string& what, const std::string& text) {
        const std::string path = TempPath("_" + what + ".tbl");
        std::ofstream(path, std::ios::binary) << text;
        tbl_files_.push_back(path);
        return path;
    }

    const Schema& SchemaOf(const std::string& table) { return db_->GetCatalog()->GetTable(table)->schema; }

    // A hand-built scan of table, reading only cols when given.
    std::unique_ptr<SeqScanPlanNode> ScanPlan(const std::string& table, std::optional<std::vector<uint32_t>> cols) {
        auto plan = std::make_unique<SeqScanPlanNode>(SchemaOf(table), table);
        plan->read_columns = std::move(cols);
        return plan;
    }

    // Runs a plan tuple by tuple, or batch by batch in vectorized mode.
    std::vector<Tuple> Drain(const PlanNode* plan, ExecutionMode mode) {
        auto exec = ExecutorFactory::CreateExecutor(plan, ctx_.get(), mode);
        exec->Init();
        std::vector<Tuple> rows;
        if (mode == ExecutionMode::TUPLE) {
            Tuple t;
            while (exec->Next(&t)) rows.push_back(t);
        } else {
            DataChunk chunk(plan->output_schema);
            while (exec->NextBatch(&chunk)) {
                for (size_t i = 0; i < chunk.size(); ++i) rows.push_back(chunk.MaterializeTuple(i));
            }
        }
        exec->Close();
        return rows;
    }
};

constexpr ExecutionMode kModes[] = {ExecutionMode::TUPLE, ExecutionMode::VECTORIZED};

TEST_F(ColumnarScanExecutorTest, SameValuesAsRowStoreTuple) {
    const auto row = db_->ExecuteSQL("SELECT * FROM r");
    const auto col = db_->ExecuteSQL("SELECT * FROM c");
    ASSERT_EQ(row.tuples.size(), kRows);
    ExpectSameResult(row, col);
}

TEST_F(ColumnarScanExecutorTest, SameValuesAsRowStoreVectorized) {
    const auto row = db_->ExecuteSQL("SELECT * FROM r", ExecutionMode::VECTORIZED);
    const auto col = db_->ExecuteSQL("SELECT * FROM c", ExecutionMode::VECTORIZED);
    ASSERT_EQ(row.tuples.size(), kRows);
    ExpectSameResult(row, col);
}

TEST_F(ColumnarScanExecutorTest, ReadColumnsSubsetGivesDefaults) {
    const Schema& schema = SchemaOf("c");
    const auto full = Drain(ScanPlan("r", std::nullopt).get(), ExecutionMode::TUPLE);
    ASSERT_EQ(full.size(), kRows);

    for (const auto mode : kModes) {
        const auto plan = ScanPlan("c", std::vector<uint32_t>{0});
        const auto rows = Drain(plan.get(), mode);
        ASSERT_EQ(rows.size(), kRows);
        for (size_t r = 0; r < kRows; ++r) {
            ExpectSameValue(rows[r].GetValue(schema, 0), full[r].GetValue(schema, 0), "id row " + std::to_string(r));
            for (uint32_t c = 1; c < schema.GetColumnCount(); ++c) {
                ExpectSameValue(rows[r].GetValue(schema, c), Value::DefaultFor(schema.GetColumn(c).type),
                                "default row " + std::to_string(r) + " column " + std::to_string(c));
            }
            if (HasFailure()) return;
        }
    }
}

TEST_F(ColumnarScanExecutorTest, ReadColumnPastTheEndThrows) {
    const auto plan = ScanPlan("c", std::vector<uint32_t>{0, 6});
    auto exec = ExecutorFactory::CreateExecutor(plan.get(), ctx_.get(), ExecutionMode::TUPLE);
    EXPECT_THROW(exec->Init(), DatabaseException);
}

TEST_F(ColumnarScanExecutorTest, NullInReadColumnThrows) {
    const Schema schema({Column("a", ColumnType::Integer()), Column("b", ColumnType::Integer(), true)});
    LoadOptions options = Columnar();
    options.null_text = "NULL";
    db_->LoadTable("n", schema, WriteTbl("nulls", "1|NULL\n2|5\n"), options);

    const std::string want = "column 'b' holds NULLs, which queries can't read yet";
    for (const auto mode : kModes) {
        const auto plan = ScanPlan("n", std::vector<uint32_t>{0, 1});
        EXPECT_EQ(ErrorOf([&] { (void)Drain(plan.get(), mode); }), want);
        EXPECT_EQ(ErrorOf([&] { (void)db_->ExecuteSQL("SELECT * FROM n", mode); }), want);
    }
}

TEST_F(ColumnarScanExecutorTest, NullInUnreadColumnIsFine) {
    const Schema schema({Column("a", ColumnType::Integer()), Column("b", ColumnType::Integer(), true)});
    LoadOptions options = Columnar();
    options.null_text = "NULL";
    db_->LoadTable("n", schema, WriteTbl("nulls", "1|NULL\n2|5\n"), options);

    for (const auto mode : kModes) {
        const auto plan = ScanPlan("n", std::vector<uint32_t>{0});
        const auto rows = Drain(plan.get(), mode);
        ASSERT_EQ(rows.size(), 2u);
        EXPECT_EQ(rows[0].GetValue(SchemaOf("n"), 0).integer_, 1);
        EXPECT_EQ(rows[1].GetValue(SchemaOf("n"), 0).integer_, 2);
        EXPECT_EQ(rows[0].GetValue(SchemaOf("n"), 1).integer_, 0);
        EXPECT_EQ(rows[1].GetValue(SchemaOf("n"), 1).integer_, 0);
    }
}

TEST_F(ColumnarScanExecutorTest, IndexScanOnColumnar) {
    ASSERT_NE(db_->GetCatalog()->CreateIndex("idx_r_id", "r", "id"), nullptr);
    ASSERT_NE(db_->GetCatalog()->CreateIndex("idx_c_id", "c", "id"), nullptr);

    // Same key range on both tables.
    auto range = [&](const std::string& table, const std::string& index) {
        auto plan = std::make_unique<IndexScanPlanNode>(SchemaOf(table), table, index);
        plan->low_key = Value(int64_t{-100});
        plan->high_key = Value(int64_t{400});
        return plan;
    };

    const auto row_plan = range("r", "idx_r_id");
    const auto col_plan = range("c", "idx_c_id");
    for (const auto mode : kModes) {
        const auto want = Drain(row_plan.get(), mode);
        const auto got = Drain(col_plan.get(), mode);
        ASSERT_EQ(want.size(), 501u);
        ExpectSameRows(want, got, SchemaOf("c"));
    }

    // Reading only the key gives defaults elsewhere.
    col_plan->read_columns = std::vector<uint32_t>{0};
    const auto want = Drain(row_plan.get(), ExecutionMode::TUPLE);
    const auto keys = Drain(col_plan.get(), ExecutionMode::TUPLE);
    ASSERT_EQ(keys.size(), want.size());
    const Schema& schema = SchemaOf("c");
    for (size_t r = 0; r < keys.size(); ++r) {
        ExpectSameValue(keys[r].GetValue(schema, 0), want[r].GetValue(schema, 0), "key row " + std::to_string(r));
        EXPECT_EQ(keys[r].GetValue(schema, 5).varchar_, "");
        EXPECT_EQ(keys[r].GetValue(schema, 2).decimal_, 0.0);
    }

    // Throws on a pinned page, so this shows every pin was released.
    EXPECT_NO_THROW(db_->GetBufferPoolManager()->DropAllPages());
}

TEST_F(ColumnarScanExecutorTest, IndexScanOnColumnarRefusesNulls) {
    const Schema schema({Column("a", ColumnType::Integer()), Column("b", ColumnType::Integer(), true)});
    LoadOptions options = Columnar();
    options.null_text = "NULL";
    db_->LoadTable("n", schema, WriteTbl("nulls", "1|NULL\n2|5\n"), options);
    ASSERT_NE(db_->GetCatalog()->CreateIndex("idx_n_a", "n", "a"), nullptr);

    auto plan = std::make_unique<IndexScanPlanNode>(SchemaOf("n"), "n", "idx_n_a");
    EXPECT_EQ(ErrorOf([&] { (void)Drain(plan.get(), ExecutionMode::TUPLE); }),
              "column 'b' holds NULLs, which queries can't read yet");

    // Row 2's b isn't NULL, so a range over it alone reads fine.
    plan->low_key = Value(int32_t{2});
    const auto rows = Drain(plan.get(), ExecutionMode::TUPLE);
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].GetValue(SchemaOf("n"), 1).integer_, 5);

    // Throws on a pinned page, so this shows the refused read released its pins too.
    EXPECT_NO_THROW(db_->GetBufferPoolManager()->DropAllPages());
}

TEST_F(ColumnarScanExecutorTest, AnalyzeNamesColumnarScan) {
    QueryOptions options;
    options.analyze = true;
    const auto result = db_->ExecuteSQL("SELECT * FROM c", options);
    EXPECT_EQ(result.tuples.size(), kRows);
    const auto& ops = result.profile.operators;
    const auto scan = std::find_if(ops.begin(), ops.end(), [](const OperatorProfile& op) { return op.name == "SeqScan c"; });
    ASSERT_NE(scan, ops.end());
    EXPECT_EQ(scan->rows, kRows);
}

TEST_F(ColumnarScanExecutorTest, ScanFetchesOnlyReadColumns) {
    const auto& meta = db_->GetCatalog()->GetTable("c")->columnar->Meta();
    auto pages_of = [&](uint32_t col) {
        uint64_t pages = 0;
        for (const auto& group : meta.row_groups) pages += group.chunks[col].page_count;
        return pages;
    };
    const uint64_t first = pages_of(0);
    ASSERT_GT(first, 0u);

    // A second column with a different page count, so a match can't be luck.
    uint32_t other = 1;
    while (other < meta.columns.size() && pages_of(other) == first) ++other;
    ASSERT_LT(other, meta.columns.size()) << "every column has column 0's page count";

    auto* bpm = db_->GetBufferPoolManager();
    for (const uint32_t col : {0u, other}) {
        for (const auto mode : kModes) {
            bpm->DropAllPages();
            bpm->ResetStats();
            const auto plan = ScanPlan("c", std::vector<uint32_t>{col});
            EXPECT_EQ(Drain(plan.get(), mode).size(), kRows);
            EXPECT_EQ(bpm->GetMissCount(), pages_of(col)) << "column " << col;
        }
    }
    EXPECT_NE(pages_of(other), first);
}

TEST_F(ColumnarScanExecutorTest, PlanColumnTypesMustMatch) {
    // Same names and count as c, but price read as an integer.
    Schema wrong({Column("id", TypeId::BIGINT), Column("qty", TypeId::INTEGER), Column("price", TypeId::INTEGER),
                  Column("cost", TypeId::DECIMAL), Column("ship", TypeId::DATE), Column("label", TypeId::VARCHAR)});
    SeqScanPlanNode plan(wrong, "c");
    auto exec = ExecutorFactory::CreateExecutor(&plan, ctx_.get(), ExecutionMode::TUPLE);
    EXPECT_EQ(ErrorOf([&] { exec->Init(); }), "ColumnarScanExecutor: the plan's column 2 has a different type than table c");
}

TEST_F(ColumnarScanExecutorTest, FilteredQueriesMatchRowStore) {
    const std::vector<std::string> queries = {
        "SELECT id, qty, price FROM T WHERE qty > 0 AND price < 500.0 AND label = 'beta3'",
        "SELECT label, COUNT(*), SUM(cost), MIN(ship), MAX(qty) FROM T "
        "WHERE qty >= -10 AND ship >= '1994-01-01' GROUP BY label ORDER BY label",
        "SELECT id, cost, ship FROM T WHERE cost > 0.0 AND ship < '1996-06-01' AND id < 1000 ORDER BY cost, id",
    };
    for (const auto mode : kModes) {
        for (const auto& q : queries) {
            std::string on_r = q, on_c = q;
            on_r.replace(on_r.find("FROM T"), 6, "FROM r");
            on_c.replace(on_c.find("FROM T"), 6, "FROM c");
            const auto want = db_->ExecuteSQL(on_r, mode);
            const auto got = db_->ExecuteSQL(on_c, mode);
            EXPECT_FALSE(want.tuples.empty()) << q;
            SCOPED_TRACE(q);
            ExpectSameResult(want, got);
        }
    }
}

TEST_F(ColumnarScanExecutorTest, EmptyColumnarTable) {
    db_->LoadTable("e", SixColumns(), WriteTbl("empty", ""), Columnar());
    for (const auto mode : kModes) {
        const auto count = db_->ExecuteSQL("SELECT COUNT(*) FROM e", mode);
        ASSERT_EQ(count.tuples.size(), 1u);
        EXPECT_EQ(count.tuples[0].GetValue(count.schema, 0).integer_, 0);
        EXPECT_TRUE(db_->ExecuteSQL("SELECT * FROM e", mode).tuples.empty());
    }
}

}  // namespace
}  // namespace shilmandb
