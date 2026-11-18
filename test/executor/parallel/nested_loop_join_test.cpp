#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "catalog/catalog.hpp"
#include "common/exception.hpp"
#include "engine/database.hpp"
#include "executor/parallel/basic_operators.hpp"
#include "executor/parallel/batch.hpp"
#include "executor/parallel/compiler.hpp"
#include "executor/parallel/nested_loop_join.hpp"
#include "executor/parallel/pipeline.hpp"
#include "parser/parser.hpp"
#include "planner/planner.hpp"

namespace shilmandb::parallel {
namespace {

constexpr uint32_t kBigRows = 70000;  // two row groups

const char* const kWords[] = {"alpha", "beta", "", "gamma delta", "épée"};

const VType kInt{};
const VType kStr{Kind::STRING, 0, false, TypeId::VARCHAR};

// A result row: its non-decimal cells as text and its decimals as numbers.
struct Row {
    std::vector<std::string> text;
    std::vector<double> decimals;

    bool operator<(const Row& o) const { return text != o.text ? text < o.text : decimals < o.decimals; }
};

// Row i of a result, split into text cells and decimals.
Row RowOf(const QueryResult& r, size_t i) {
    Row row;
    for (uint32_t c = 0; c < r.schema.GetColumnCount(); ++c) {
        const Value v = r.tuples[i].GetValue(r.schema, c);
        if (v.type_ == TypeId::DECIMAL) {
            row.decimals.push_back(v.decimal_);
        } else {
            row.text.push_back(v.ToString());
        }
    }
    return row;
}

// Exact text cells, decimals within SameResult's 1e-6 relative.
bool SameRow(const Row& x, const Row& y) {
    if (x.text != y.text || x.decimals.size() != y.decimals.size()) return false;
    for (size_t d = 0; d < x.decimals.size(); ++d) {
        const double a = x.decimals[d];
        const double b = y.decimals[d];
        if (std::abs(a - b) > 1e-6 * std::max({1.0, std::abs(a), std::abs(b)})) return false;
    }
    return true;
}

// A row as one line of text, for failure messages.
std::string Describe(const Row& row) {
    std::string out;
    for (const auto& t : row.text) out += t + " | ";
    for (double d : row.decimals) out += std::to_string(d) + " | ";
    return out;
}

// Every row of a result, sorted so results compare as multisets.
std::vector<Row> SortedRows(const QueryResult& r) {
    std::vector<Row> rows;
    rows.reserve(r.tuples.size());
    for (size_t i = 0; i < r.tuples.size(); ++i) rows.push_back(RowOf(r, i));
    std::sort(rows.begin(), rows.end());
    return rows;
}

// How got and want first differ as multisets of rows, or "" when they match.
std::string Difference(const QueryResult& got, const QueryResult& want) {
    if (got.schema.GetColumnCount() != want.schema.GetColumnCount()) return "column counts differ";
    for (uint32_t c = 0; c < got.schema.GetColumnCount(); ++c) {
        if (got.schema.GetColumn(c).type != want.schema.GetColumn(c).type) return "column " + std::to_string(c) + " has another type";
    }
    if (got.tuples.size() != want.tuples.size()) {
        return std::to_string(got.tuples.size()) + " rows, expected " + std::to_string(want.tuples.size());
    }
    const std::vector<Row> x = SortedRows(got);
    const std::vector<Row> y = SortedRows(want);
    for (size_t i = 0; i < x.size(); ++i) {
        if (!SameRow(x[i], y[i])) return "row " + std::to_string(i) + ": got " + Describe(x[i]) + "expected " + Describe(y[i]);
    }
    return "";
}

// Signed cents as text with two decimals.
std::string Money(int64_t cents) {
    const int64_t abs = cents < 0 ? -cents : cents;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s%lld.%02lld", cents < 0 ? "-" : "", static_cast<long long>(abs / 100),
                  static_cast<long long>(abs % 100));
    return buf;
}

// A batch of n integers from first, one slot, every row active.
Batch IntBatch(int64_t first, uint32_t n) {
    Batch b;
    ShapeBatch(b, {kInt});
    b.Reset(n);
    for (uint32_t i = 0; i < n; ++i) b.slots[0].I64()[i] = first + i;
    return b;
}

// What one worker saw come out of the join.
struct Seen {
    std::vector<std::pair<int64_t, int64_t>> pairs;  // (left value, right value)
    std::vector<std::string> names;                  // the right side's string per pair
    int calls{0};
};

// Reads a join output's active rows: left int, right int, right string.
void Collect(const Batch& out, Seen& seen) {
    for (uint32_t k = 0; k < out.Active(); ++k) {
        const uint32_t r = out.sel[k];
        seen.pairs.emplace_back(out.slots[0].I64()[r], out.slots[1].I64()[r]);
        seen.names.emplace_back(View(out.slots[2].StrAt(r)));
    }
}

class ParallelNestedLoopJoinTest : public ::testing::Test {
protected:
    std::string db_file_;
    std::vector<std::string> tbl_files_;
    std::unique_ptr<Database> db_;

    void SetUp() override {
        db_file_ = TempPath(".db");
        std::filesystem::remove(db_file_);
        db_ = std::make_unique<Database>(db_file_);

        // Binding ignores table names, so every table's columns have their own names.
        std::string a;
        for (uint32_t r = 0; r < 300; ++r) {
            a += std::to_string((r * 37) % 500) + "|" + Money(static_cast<int64_t>((r * 7919) % 100000) - 50000) + "|" +
                 kWords[r % 5] + "|a-note-" + std::to_string(r) + (r % 9 == 0 ? " ünï" : "") + "\n";
        }
        Load("a", Schema({Column("a_x", ColumnType::Integer()), Column("a_m", ColumnType::Numeric(15, 2)),
                          Column("a_s", ColumnType::Varchar(16)), Column("a_n", ColumnType::Text())}),
             a);

        std::string b;
        char buf[128];
        for (uint32_t r = 0; r < 200; ++r) {
            // The trailing delimiter keeps an empty last field.
            std::snprintf(buf, sizeof(buf), "%u|%.2f|%s|\n", (r * 53) % 400, r / 4.0 - 20.5, kWords[r % 4]);
            b += buf;
        }
        Load("b", Schema({Column("b_y", ColumnType::Integer()), Column("b_f", ColumnType::Double()),
                          Column("b_t", ColumnType::Varchar(16))}),
             b);

        std::string c;
        for (uint32_t r = 0; r < 50; ++r) c += std::to_string((r * 9) % 450) + "|c-" + std::to_string(r) + "\n";
        Load("c", Schema({Column("c_z", ColumnType::Integer()), Column("c_w", ColumnType::Text())}), c);
    }

    void TearDown() override {
        db_.reset();
        std::filesystem::remove(db_file_);
        for (const auto& f : tbl_files_) std::filesystem::remove(f);
    }

    // A temp path unique to this process.
    static std::string TempPath(const std::string& suffix) {
        return (std::filesystem::temp_directory_path() /
                ("shilmandb_parallel_nested_loop_join_test_" + std::to_string(::getpid()) + suffix)).string();
    }

    // Loads text into a columnar table on two threads.
    void Load(const std::string& name, const Schema& schema, const std::string& text) {
        const std::string path = TempPath("_" + name + ".tbl");
        std::ofstream(path, std::ios::binary) << text;
        tbl_files_.push_back(path);
        LoadOptions options;
        options.storage = StorageKind::COLUMNAR;
        options.threads = 2;
        db_->LoadTable(name, schema, path, options);
    }

    // Runs sql in the given mode; threads matters only for PARALLEL.
    QueryResult Run(const std::string& sql, ExecutionMode mode, unsigned threads = 1) {
        QueryOptions options;
        options.mode = mode;
        options.threads = threads;
        return db_->ExecuteSQL(sql, options);
    }

    // PARALLEL against VECTORIZED as multisets; returns the row count.
    size_t Check(const std::string& sql) {
        const QueryResult want = Run(sql, ExecutionMode::VECTORIZED);
        EXPECT_EQ(Difference(Run(sql, ExecutionMode::PARALLEL), want), "") << sql;
        return want.tuples.size();
    }
};

TEST_F(ParallelNestedLoopJoinTest, NestedLoopMatchesReference) {
    // a.x < b.y over 300 x 200 rows, written as ON and as WHERE.
    const size_t on = Check("SELECT a_x, b_y, a_s, b_t FROM a JOIN b ON a_x < b_y");
    EXPECT_GT(on, 1000u);
    EXPECT_LT(on, 60000u);
    EXPECT_EQ(Check("SELECT * FROM a, b WHERE a_x < b_y"), on);

    // A column only the predicate reads; mixed types; strings on both sides.
    EXPECT_EQ(Check("SELECT a_x FROM a JOIN b ON a_x < b_y"), on);
    EXPECT_GT(Check("SELECT a_x, b_y, a_m, b_f, a_n FROM a, b WHERE a_m < b_f AND a_s <> b_t"), 0u);

    // No predicate: every pair, also when one side hands up no column.
    EXPECT_EQ(Check("SELECT a_x, b_y FROM a, b"), 60000u);
    size_t small_b = 0;
    for (uint32_t r = 0; r < 200; ++r) small_b += (r * 53) % 400 < 40 ? 1 : 0;
    EXPECT_EQ(Check("SELECT a_s FROM a, b WHERE b_y < 40"), 300u * small_b);

    // An empty side, on the right and on the left, with and without a predicate.
    EXPECT_EQ(Check("SELECT a_x, b_y FROM a JOIN b ON a_x < b_y WHERE b_y > 100000"), 0u);
    EXPECT_EQ(Check("SELECT a_x, b_y FROM a JOIN b ON a_x < b_y WHERE a_x > 100000"), 0u);
    EXPECT_EQ(Check("SELECT a_x, b_y FROM a, b WHERE b_y > 100000"), 0u);
    EXPECT_EQ(Check("SELECT a_x, b_y FROM a, b WHERE a_x > 100000"), 0u);

    // Two joins: the second one's left batches arrive with narrowed selections.
    EXPECT_GT(Check("SELECT a_x, b_y, c_z, c_w FROM a, b, c WHERE a_x < b_y AND b_y < c_z AND c_z < 100"), 100u);

    // A sort above the join, compared in order (the sort keys are unique).
    const std::string sorted = "SELECT a_x, b_y, a_n FROM a JOIN b ON a_x < b_y WHERE a_x + b_y > 600 ORDER BY a_n, b_y";
    const QueryResult want = Run(sorted, ExecutionMode::VECTORIZED);
    const QueryResult got = Run(sorted, ExecutionMode::PARALLEL);
    ASSERT_EQ(got.tuples.size(), want.tuples.size());
    ASSERT_GT(got.tuples.size(), 0u);
    for (size_t i = 0; i < got.tuples.size(); ++i) {
        ASSERT_TRUE(SameRow(RowOf(got, i), RowOf(want, i))) << "row " << i;
    }
}

TEST_F(ParallelNestedLoopJoinTest, ManyMorselsOnOneSide) {
    std::string big;
    for (uint32_t r = 0; r < kBigRows; ++r) big += std::to_string(r) + "|" + kWords[r % 5] + "|\n";
    Load("big", Schema({Column("big_x", ColumnType::Integer()), Column("big_s", ColumnType::Varchar(16))}), big);
    ASSERT_EQ(db_->GetCatalog()->GetTable("big")->columnar->RowGroupCount(), 2u);

    // Five right rows; the big table on each side of the join in turn.
    const std::string sql = "SELECT big_x, b_y, big_s, b_t FROM big, b WHERE big_x + b_y > 69000 AND b_y < 10";
    const QueryResult want = Run(sql, ExecutionMode::VECTORIZED);
    EXPECT_GT(want.tuples.size(), 4000u);
    for (const std::vector<int>& order : {std::vector<int>{0, 1}, std::vector<int>{1, 0}}) {
        QueryOptions options;
        options.mode = ExecutionMode::PARALLEL;
        options.threads = 1;
        options.forced_join_order = order;
        const QueryResult got = db_->ExecuteSQL(sql, options);
        EXPECT_EQ(got.profile.join.order, order);
        EXPECT_EQ(Difference(got, want), "") << "order " << order[0] << ", " << order[1];
    }
}

TEST_F(ParallelNestedLoopJoinTest, PipelinesInDependencyOrder) {
    // A plan and what it compiled to; the pipelines point into the plan.
    struct Compiled {
        std::unique_ptr<PlanNode> root;
        PhysicalPlan physical;
    };
    QueryContext ctx{db_->GetBufferPoolManager(), db_->GetCatalog(), 1};
    const auto compile = [&](const std::string& sql) {
        Compiled c;
        Planner planner(db_->GetCatalog());
        c.root = planner.Plan(std::move(*Parser(sql).Parse()));
        c.physical = CompilePlan(*c.root, ctx);
        return c;
    };
    // Every dependency comes first, and only the last pipeline ends in the result.
    const auto check_order = [](const PhysicalPlan& plan, const std::string& sql) {
        for (size_t i = 0; i < plan.pipelines.size(); ++i) {
            const Pipeline& p = plan.pipelines[i];
            EXPECT_NE(p.source, nullptr) << sql << ", pipeline " << i;
            for (size_t d : p.depends_on) EXPECT_LT(d, i) << sql << ", pipeline " << i;
            EXPECT_EQ(p.sink == plan.result, i + 1 == plan.pipelines.size()) << sql << ", pipeline " << i;
        }
    };
    const auto has_limit_operator = [](const PhysicalPlan& plan) {
        return std::any_of(plan.operators.begin(), plan.operators.end(),
                           [](const auto& op) { return dynamic_cast<const LimitOperator*>(op.get()) != nullptr; });
    };

    // The right side's pipeline runs first, into the join's sink.
    const std::string join = "SELECT a_x, b_y FROM a JOIN b ON a_x < b_y";
    const Compiled one = compile(join);
    check_order(one.physical, join);
    ASSERT_EQ(one.physical.pipelines.size(), 2u);
    EXPECT_EQ(one.physical.pipelines[1].depends_on, std::vector<size_t>{0});

    // Two joins and a sort: both right sides, joins into the sort, then the result.
    // The LIMIT becomes the sort's top n, so no limit operator is left.
    const std::string two = "SELECT a_x, b_y, c_z FROM a, b, c WHERE a_x < b_y AND b_y < c_z ORDER BY a_x, b_y, c_z LIMIT 5";
    const Compiled sorted = compile(two);
    check_order(sorted.physical, two);
    ASSERT_EQ(sorted.physical.pipelines.size(), 4u);
    std::vector<size_t> joins_depend = sorted.physical.pipelines[2].depends_on;
    std::sort(joins_depend.begin(), joins_depend.end());
    EXPECT_EQ(joins_depend, (std::vector<size_t>{0, 1}));
    EXPECT_EQ(sorted.physical.pipelines[3].depends_on, std::vector<size_t>{2});
    EXPECT_TRUE(sorted.physical.pipelines[3].operators.empty());
    EXPECT_FALSE(has_limit_operator(sorted.physical));

    // A LIMIT with no sort stays an operator; the result's pipeline stops at it.
    const std::string limited = join + " LIMIT 5";
    Compiled cut = compile(limited);
    check_order(cut.physical, limited);
    ASSERT_TRUE(has_limit_operator(cut.physical));
    for (Pipeline& p : cut.physical.pipelines) RunPipeline(p, ctx);
    EXPECT_TRUE(cut.physical.result->Done());
    EXPECT_EQ(cut.physical.result->TakeTuples().size(), 5u);
}

TEST_F(ParallelNestedLoopJoinTest, LimitBelowAJoinIsRefused) {
    // No planner builds this: a LIMIT under the join, in the result's own pipeline.
    // Its counter would cut the join's output short, so every build refuses it.
    const Schema& a = db_->GetCatalog()->GetTable("a")->schema;
    const Schema& b = db_->GetCatalog()->GetTable("b")->schema;
    std::vector<Column> joined = a.GetColumns();
    joined.insert(joined.end(), b.GetColumns().begin(), b.GetColumns().end());
    auto limit = std::make_unique<LimitPlanNode>(a, 5);
    limit->children.push_back(std::make_unique<SeqScanPlanNode>(a, "a"));
    NestedLoopJoinPlanNode join(Schema(joined), nullptr);
    join.children.push_back(std::move(limit));
    join.children.push_back(std::make_unique<SeqScanPlanNode>(b, "b"));

    QueryContext ctx{db_->GetBufferPoolManager(), db_->GetCatalog(), 1};
    try {
        (void)CompilePlan(join, ctx);
        ADD_FAILURE() << "a LIMIT below a join compiled";
    } catch (const DatabaseException& e) {
        EXPECT_STREQ(e.what(), "the parallel executor can't run a LIMIT below a join");
    }
}

TEST_F(ParallelNestedLoopJoinTest, CursorLivesInEachWorkersState) {
    // Every pair (no predicate): left int, then the right side's int and string.
    NestedLoopSpec spec;
    spec.left_slots = {0};
    spec.right_slots = {0, 1};
    spec.types = {kInt, kInt, kStr};
    auto made = MakeNestedLoopJoin(std::move(spec));
    Sink& sink = *made.first;
    Operator& join = *made.second;
    QueryContext ctx{nullptr, nullptr};

    // Three right rows; their strings are wiped once consumed, so the sink must copy them.
    std::vector<std::string> names = {"first", "ünï", ""};
    Batch right;
    ShapeBatch(right, {kInt, kStr});
    right.Reset(3);
    for (uint32_t i = 0; i < 3; ++i) {
        right.slots[0].I64()[i] = 100 * (i + 1);
        right.slots[1].Str()[i] = StringRef{names[i].data(), static_cast<uint32_t>(names[i].size())};
    }
    auto right_local = sink.MakeLocal();
    sink.Consume(right, *right_local);
    sink.Combine(std::move(right_local));
    sink.Combine(sink.MakeLocal());  // a worker that saw nothing
    for (std::string& s : names) std::fill(s.begin(), s.end(), '#');
    sink.Finalize(ctx);

    // Two workers take turns; a full batch needs one call per right row.
    Batch left1 = IntBatch(0, kBatchSize);
    Batch left2 = IntBatch(10000, 1000);
    // A narrowed selection: rows 5..504 of the second batch.
    for (uint32_t i = 0; i < 500; ++i) left2.sel.Data()[i] = static_cast<uint16_t>(5 + i);
    left2.sel.Resize(500);

    auto w1 = join.MakeLocal();
    auto w2 = join.MakeLocal();
    Batch out1;
    Batch out2;
    Seen seen1;
    Seen seen2;
    bool more1 = true;
    bool more2 = true;
    // A cursor two workers shared would never finish, so the loop is bounded.
    for (int guard = 0; (more1 || more2) && guard < 100; ++guard) {
        if (more1) {
            more1 = join.Execute(left1, out1, w1.get()) == OpResult::HAVE_MORE_OUTPUT;
            Collect(out1, seen1);
            ++seen1.calls;
        }
        if (more2) {
            more2 = join.Execute(left2, out2, w2.get()) == OpResult::HAVE_MORE_OUTPUT;
            Collect(out2, seen2);
            ++seen2.calls;
        }
    }
    ASSERT_FALSE(more1 || more2) << "a worker never finished its batch";
    EXPECT_EQ(seen1.calls, 3);
    EXPECT_EQ(seen2.calls, 1);  // 500 rows x 3 right rows fit one batch

    const auto expect = [](const Seen& seen, int64_t first, uint32_t n) {
        std::vector<std::pair<int64_t, int64_t>> want;
        std::vector<std::pair<int64_t, int64_t>> got = seen.pairs;
        for (uint32_t i = 0; i < n; ++i) {
            for (int64_t r = 1; r <= 3; ++r) want.emplace_back(first + i, 100 * r);
        }
        std::sort(want.begin(), want.end());
        std::sort(got.begin(), got.end());
        EXPECT_EQ(got, want);
        for (size_t i = 0; i < seen.pairs.size(); ++i) {
            const int64_t r = seen.pairs[i].second;
            EXPECT_EQ(seen.names[i], r == 100 ? "first" : (r == 200 ? "ünï" : "")) << "pair " << i;
        }
    };
    expect(seen1, 0, kBatchSize);
    expect(seen2, 10005, 500);

    // The next batch on a worker starts from the first right row again.
    Batch left3 = IntBatch(-7, 2);
    Seen seen3;
    EXPECT_EQ(join.Execute(left3, out1, w1.get()), OpResult::NEED_INPUT);
    Collect(out1, seen3);
    expect(seen3, -7, 2);
}

TEST_F(ParallelNestedLoopJoinTest, EmptyRightSideGivesNothing) {
    NestedLoopSpec spec;
    spec.left_slots = {0};
    spec.right_slots = {0, 1};
    spec.types = {kInt, kInt, kStr};
    auto made = MakeNestedLoopJoin(std::move(spec));
    QueryContext ctx{nullptr, nullptr};
    made.first->Combine(made.first->MakeLocal());
    made.first->Finalize(ctx);

    Batch left = IntBatch(0, 10);
    Batch out;
    auto local = made.second->MakeLocal();
    EXPECT_EQ(made.second->Execute(left, out, local.get()), OpResult::NEED_INPUT);
    EXPECT_EQ(out.Active(), 0u);
}

}  // namespace
}  // namespace shilmandb::parallel
