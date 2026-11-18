#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "catalog/catalog.hpp"
#include "engine/database.hpp"
#include "executor/parallel/batch.hpp"
#include "executor/parallel/compiler.hpp"
#include "executor/parallel/expression.hpp"
#include "executor/parallel/hash_join.hpp"
#include "executor/parallel/parallel_executor.hpp"
#include "executor/parallel/pipeline.hpp"
#include "parser/ast.hpp"
#include "parser/parser.hpp"
#include "planner/planner.hpp"
#include "tpch/tpch_workload.hpp"

namespace shilmandb::parallel {
namespace {

constexpr uint32_t kBigRows = 70000;  // two row groups, so two morsels

const char* const kWords[] = {"alpha", "beta", "", "gamma delta", "épée"};

const VType kInt{};
const VType kStr{Kind::STRING, 0, false, TypeId::VARCHAR};

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

// How two sorted row lists first differ, or "" when they match.
std::string RowsDifference(const std::vector<Row>& x, const std::vector<Row>& y) {
    if (x.size() != y.size()) return std::to_string(x.size()) + " rows, expected " + std::to_string(y.size());
    for (size_t i = 0; i < x.size(); ++i) {
        if (!SameRow(x[i], y[i])) return "row " + std::to_string(i) + ": got " + Describe(x[i]) + "expected " + Describe(y[i]);
    }
    return "";
}

// How got and want first differ as multisets of rows, or "" when they match.
std::string Difference(const QueryResult& got, const QueryResult& want) {
    if (got.schema.GetColumnCount() != want.schema.GetColumnCount()) return "column counts differ";
    for (uint32_t c = 0; c < got.schema.GetColumnCount(); ++c) {
        if (got.schema.GetColumn(c).type != want.schema.GetColumn(c).type) return "column " + std::to_string(c) + " has another type";
    }
    return RowsDifference(SortedRows(got), SortedRows(want));
}

// Signed cents as text with two decimals.
std::string Money(int64_t cents) {
    const int64_t abs = cents < 0 ? -cents : cents;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s%lld.%02lld", cents < 0 ? "-" : "", static_cast<long long>(abs / 100),
                  static_cast<long long>(abs % 100));
    return buf;
}

// An 8,000-byte string ending in multibyte text; the filler changes with r.
std::string BigText(uint32_t r) {
    return std::string(7994, static_cast<char>('a' + r % 26)) + "épée";
}

// A column reference bound to plan column c.
std::unique_ptr<CompiledExpr> Col(uint32_t c, const InputLayout& in) {
    ColumnRef ref;
    ref.column_name = "c" + std::to_string(c);
    ref.bound_index = c;
    return CompiledExpr::Compile(ref, in);
}

// A layout where column k sits in slot k.
InputLayout LayoutOf(const std::vector<VType>& types) {
    InputLayout in;
    for (size_t k = 0; k < types.size(); ++k) in.slot_of_column.push_back(static_cast<int>(k));
    in.slot_types = types;
    in.slot_precision.assign(types.size(), 0);
    return in;
}

// A string the batch points at; the caller keeps it alive.
StringRef Ref(const std::string& s) {
    return StringRef{s.data(), static_cast<uint32_t>(s.size())};
}

// The topmost hash join of a plan, or nullptr.
HashJoinPlanNode* TopHashJoin(PlanNode* node) {
    if (node->type == PlanNodeType::HASH_JOIN) return static_cast<HashJoinPlanNode*>(node);
    for (auto& child : node->children) {
        if (auto* found = TopHashJoin(child.get())) return found;
    }
    return nullptr;
}

// True when a scan of table sits somewhere under node.
bool Holds(const PlanNode& node, const std::string& table) {
    if (node.type == PlanNodeType::SEQ_SCAN && static_cast<const SeqScanPlanNode&>(node).table_name == table) return true;
    return std::any_of(node.children.begin(), node.children.end(), [&](const auto& c) { return Holds(*c, table); });
}

// A hand-fed join over int keys. Build slots: key, payload; probe slots: key, value.
// The output is the probe value, then the payload, when the right side builds.
std::pair<std::unique_ptr<Sink>, std::unique_ptr<Operator>> IntJoin(bool build_left) {
    const InputLayout two = LayoutOf({kInt, kInt});
    JoinSpec spec;
    spec.build_key = Col(0, two);
    spec.probe_key = Col(0, two);
    spec.build_payload = {1};
    spec.probe_slots = {1};
    spec.types = {kInt, kInt};
    spec.build_left = build_left;
    return MakeHashJoin(std::move(spec));
}

// A batch of (key, value) rows, every row active.
Batch PairBatch(const std::vector<std::pair<int64_t, int64_t>>& rows) {
    Batch b;
    ShapeBatch(b, {kInt, kInt});
    b.Reset(static_cast<uint32_t>(rows.size()));
    for (size_t i = 0; i < rows.size(); ++i) {
        b.slots[0].I64()[i] = rows[i].first;
        b.slots[1].I64()[i] = rows[i].second;
    }
    return b;
}

// Runs one probe batch through the join until it asks for input.
// Returns (slot 0, slot 1) of every output row; counts the calls.
std::vector<std::pair<int64_t, int64_t>> Drain(Operator& op, Batch& in, LocalState* local, int* calls = nullptr,
                                               std::vector<uint32_t>* sizes = nullptr) {
    std::vector<std::pair<int64_t, int64_t>> got;
    Batch out;
    for (int guard = 0; guard < 100000; ++guard) {
        const OpResult result = op.Execute(in, out, local);
        if (calls != nullptr) ++*calls;
        if (sizes != nullptr) sizes->push_back(out.Active());
        for (uint32_t k = 0; k < out.Active(); ++k) {
            const uint32_t r = out.sel[k];
            got.emplace_back(out.slots[0].I64()[r], out.slots[1].I64()[r]);
        }
        if (result == OpResult::NEED_INPUT) return got;
    }
    ADD_FAILURE() << "the join never asked for input";
    return got;
}

class ParallelHashJoinTest : public ::testing::Test {
protected:
    std::string db_file_;
    std::vector<std::string> extra_files_;
    std::unique_ptr<Database> db_;

    // Key k in 1..50 has k copies in l and 51 - k in r.
    // Keys 51..60 are only in l, 61..75 only in r.
    struct LRow {
        int64_t k, j, m, v;
        std::string s;
    };
    struct RRow {
        int64_t k, j, y, cents;
        std::string t;
    };
    std::vector<LRow> l_rows_;
    std::vector<RRow> r_rows_;

    void SetUp() override {
        db_file_ = TempPath(".db");
        std::filesystem::remove(db_file_);
        db_ = std::make_unique<Database>(db_file_);

        for (int64_t k = 1; k <= 60; ++k) {
            const int64_t copies = k <= 50 ? k : 1;
            for (int64_t c = 0; c < copies; ++c) l_rows_.push_back({k, (k * 100 + c) % 3, c % 4, k * 100 + c, kWords[(k + c) % 5]});
        }
        for (int64_t k = 1; k <= 75; ++k) {
            if (k > 50 && k <= 60) continue;
            const int64_t copies = k <= 50 ? 51 - k : 1;
            for (int64_t c = 0; c < copies; ++c) {
                r_rows_.push_back({k, c % 3, k * 100 + 20 - c, k * 1000 + c * 7 - 30000,
                                   "t" + std::to_string(k) + "-" + std::to_string(c) + (c % 3 == 0 ? " ünï" : "")});
            }
        }
        // Scatter the keys, so equal keys don't sit together.
        std::vector<LRow> l_scattered;
        for (size_t i = 0; i < l_rows_.size(); ++i) l_scattered.push_back(l_rows_[(i * 37) % l_rows_.size()]);
        std::string l;
        for (const LRow& r : l_scattered) {
            l += std::to_string(r.k) + "|" + std::to_string(r.j) + "|" + std::to_string(r.m) + "|" + std::to_string(r.v) + "|" + r.s + "|\n";
        }
        Load("l", Schema({Column("l_k", ColumnType::Integer()), Column("l_j", ColumnType::Integer()), Column("l_m", ColumnType::Integer()),
                          Column("l_v", ColumnType::Integer()), Column("l_s", ColumnType::Varchar(16))}),
             l);
        std::string r;
        for (const RRow& x : r_rows_) {
            r += std::to_string(x.k) + "|" + std::to_string(x.j) + "|" + std::to_string(x.y) + "|" + Money(x.cents) + "|" + x.t + "|\n";
        }
        Load("r", Schema({Column("r_k", ColumnType::Integer()), Column("r_j", ColumnType::Integer()), Column("r_y", ColumnType::Integer()),
                          Column("r_w", ColumnType::Numeric(15, 2)), Column("r_t", ColumnType::Text())}),
             r);
    }

    void TearDown() override {
        db_.reset();
        std::filesystem::remove(db_file_);
        for (const auto& f : extra_files_) std::filesystem::remove(f);
    }

    // A temp path unique to this process.
    static std::string TempPath(const std::string& suffix) {
        return (std::filesystem::temp_directory_path() /
                ("shilmandb_parallel_hash_join_test_" + std::to_string(::getpid()) + suffix)).string();
    }

    // Loads text into a columnar table on two threads.
    void Load(const std::string& name, const Schema& schema, const std::string& text) {
        const std::string path = TempPath("_" + name + ".tbl");
        std::ofstream(path, std::ios::binary) << text;
        extra_files_.push_back(path);
        LoadOptions options;
        options.storage = StorageKind::COLUMNAR;
        options.threads = 2;
        db_->LoadTable(name, schema, path, options);
    }

    // Runs sql on db in one mode, maybe with a forced join order.
    static QueryResult RunOn(Database& db, const std::string& sql, ExecutionMode mode, unsigned threads = 1,
                             std::optional<std::vector<int>> order = std::nullopt) {
        QueryOptions options;
        options.mode = mode;
        options.threads = threads;
        options.forced_join_order = std::move(order);
        return db.ExecuteSQL(sql, options);
    }

    // The same on this test's database.
    QueryResult Run(const std::string& sql, ExecutionMode mode, unsigned threads = 1, std::optional<std::vector<int>> order = std::nullopt) {
        return RunOn(*db_, sql, mode, threads, std::move(order));
    }

    // PARALLEL against VECTORIZED as multisets, in both join orders; returns the row count.
    size_t Check(const std::string& sql) {
        const QueryResult want = Run(sql, ExecutionMode::VECTORIZED);
        for (const std::vector<int>& order : {std::vector<int>{0, 1}, std::vector<int>{1, 0}}) {
            for (const unsigned threads : {1u, 18u}) {
                EXPECT_EQ(Difference(Run(sql, ExecutionMode::PARALLEL, threads, order), want), "") << sql << ", order " << order[0];
            }
        }
        return want.tuples.size();
    }

    // The plan ExecuteSQL would run in PARALLEL mode.
    static std::unique_ptr<PlanNode> PlanOn(Database& db, const std::string& sql, std::optional<std::vector<int>> order = std::nullopt) {
        QueryOptions options;
        options.mode = ExecutionMode::PARALLEL;
        options.forced_join_order = std::move(order);
        Planner planner = db.MakePlanner(options);
        return planner.Plan(std::move(*Parser(sql).Parse()));
    }

    // The same on this test's database.
    std::unique_ptr<PlanNode> Plan(const std::string& sql, std::optional<std::vector<int>> order = std::nullopt) {
        return PlanOn(*db_, sql, std::move(order));
    }

    // A plan whose topmost hash join builds the side asked for, through its estimates.
    std::unique_ptr<PlanNode> PlanBuilding(const std::string& sql, bool build_left, std::optional<std::vector<int>> order = std::nullopt) {
        auto root = Plan(sql, std::move(order));
        HashJoinPlanNode* join = TopHashJoin(root.get());
        EXPECT_NE(join, nullptr) << sql;
        if (join != nullptr) {
            join->est_left_rows = build_left ? 10.0 : 20.0;
            join->est_right_rows = build_left ? 20.0 : 10.0;
        }
        return root;
    }

    // A compiled plan's rows, and what its topmost hash join saw.
    struct Compiled {
        QueryResult result;
        HashJoinInfo top;  // the last hash join the compiler added, which is the topmost
        size_t joins{0};
        size_t pipelines{0};
    };

    // Compiles a plan, runs its pipelines in order and inspects its hash joins.
    static Compiled RunPlanOn(Database& db, const PlanNode& root, unsigned threads = 1) {
        QueryContext ctx{db.GetBufferPoolManager(), db.GetCatalog(), threads};
        PhysicalPlan physical = CompilePlan(root, ctx);
        for (Pipeline& p : physical.pipelines) RunPipeline(p, ctx);
        Compiled c{QueryResult{root.output_schema, physical.result->TakeTuples(), {}}, {}, 0, physical.pipelines.size()};
        for (const auto& op : physical.operators) {
            if (const auto info = InspectHashJoin(*op)) {
                c.top = *info;
                ++c.joins;
            }
        }
        return c;
    }

    // The same on this test's database.
    Compiled RunPlan(const PlanNode& root, unsigned threads = 1) { return RunPlanOn(*db_, root, threads); }

    // The SF=0.01 TPC-H data, or "" when it hasn't been generated.
    static std::string FindTpchDir() {
        for (const std::string dir : {"bench/tpch_data/sf0.01/", "../bench/tpch_data/sf0.01/", "../../bench/tpch_data/sf0.01/"}) {
            if (std::filesystem::exists(dir + "lineitem.tbl")) return dir;
        }
        return "";
    }

    // All of SF=0.01 TPC-H, stored by column, no indexes, in its own file.
    std::unique_ptr<Database> LoadTpch(const std::string& dir) {
        const std::string file = TempPath("_tpch.db");
        std::filesystem::remove(file);
        extra_files_.push_back(file);
        auto tpch = std::make_unique<Database>(file, 4096);
        LoadOptions options;
        options.storage = StorageKind::COLUMNAR;
        for (const auto& td : tpch::TableDescs()) tpch->LoadTable(td.name, td.schema, dir + td.tbl_file, options);
        return tpch;
    }
};

//  Answers against references

TEST_F(ParallelHashJoinTest, DuplicatesOnBothSides) {
    // A nested loop over the generated rows is the reference.
    std::vector<Row> want;
    for (const LRow& a : l_rows_) {
        for (const RRow& b : r_rows_) {
            if (a.k != b.k) continue;
            want.push_back(Row{{std::to_string(a.k), std::to_string(a.v), a.s, std::to_string(b.k), b.t}, {static_cast<double>(b.cents) / 100.0}});
        }
    }
    std::sort(want.begin(), want.end());
    ASSERT_EQ(want.size(), 22100u);  // the sum of k * (51 - k) for k = 1..50

    const std::string sql = "SELECT l_k, l_v, l_s, r_k, r_w, r_t FROM l JOIN r ON l_k = r_k";
    EXPECT_EQ(RowsDifference(SortedRows(Run(sql, ExecutionMode::VECTORIZED)), want), "");
    for (const std::vector<int>& order : {std::vector<int>{0, 1}, std::vector<int>{1, 0}}) {
        for (const unsigned threads : {1u, 3u, 18u}) {
            const QueryResult got = Run(sql, ExecutionMode::PARALLEL, threads, order);
            EXPECT_EQ(got.profile.join.order, order);
            EXPECT_EQ(RowsDifference(SortedRows(got), want), "") << "order " << order[0] << ", threads " << threads;
        }
        // Both build sides, forced through the estimates.
        for (const bool build_left : {false, true}) {
            const Compiled c = RunPlan(*PlanBuilding(sql, build_left, order));
            EXPECT_EQ(c.top.build_left, build_left);
            EXPECT_EQ(c.top.build_rows, build_left == (order[0] == 0) ? l_rows_.size() : r_rows_.size());
            EXPECT_EQ(RowsDifference(SortedRows(c.result), want), "") << "order " << order[0] << ", build_left " << build_left;
        }
    }

    // A LIMIT above the join keeps rows the join really makes.
    const QueryResult some = Run(sql + " LIMIT 7", ExecutionMode::PARALLEL, 1);
    ASSERT_EQ(some.tuples.size(), 7u);
    for (size_t i = 0; i < some.tuples.size(); ++i) {
        const Row row = RowOf(some, i);
        EXPECT_TRUE(std::any_of(want.begin(), want.end(), [&](const Row& w) { return SameRow(w, row); })) << Describe(row);
    }
}

TEST_F(ParallelHashJoinTest, NoMatches) {
    // l's keys stop at 60 and r's only-right keys start at 61.
    EXPECT_EQ(Check("SELECT l_k, r_k, r_t FROM l JOIN r ON l_k = r_k WHERE l_k > 50 AND r_k > 50"), 0u);
    // Every row has a key, just never a matching one.
    EXPECT_EQ(Check("SELECT l_k, r_k FROM l JOIN r ON l_v = r_k"), 0u);
}

TEST_F(ParallelHashJoinTest, EmptyBuildSide) {
    for (const bool build_left : {false, true}) {
        const std::string empty_side = build_left ? "l_k" : "r_k";
        const std::string sql = "SELECT l_k, l_s, r_k, r_t FROM l JOIN r ON l_k = r_k WHERE " + empty_side + " > 1000";
        const Compiled c = RunPlan(*PlanBuilding(sql, build_left, std::vector<int>{0, 1}));
        EXPECT_EQ(c.top.build_left, build_left);
        EXPECT_EQ(c.top.build_rows, 0u);
        EXPECT_TRUE(c.result.tuples.empty());
        EXPECT_EQ(Difference(c.result, Run(sql, ExecutionMode::VECTORIZED)), "");
    }

    // A hand-fed join: nothing built, then a full probe batch.
    auto made = IntJoin(false);
    QueryContext ctx{nullptr, nullptr};
    made.first->Combine(made.first->MakeLocal());
    made.first->Finalize(ctx);
    Batch probe = PairBatch({{1, 10}, {2, 20}, {3, 30}});
    auto local = made.second->MakeLocal();
    int calls = 0;
    EXPECT_TRUE(Drain(*made.second, probe, local.get(), &calls).empty());
    EXPECT_EQ(calls, 1);
}

TEST_F(ParallelHashJoinTest, EmptyProbeSide) {
    for (const bool build_left : {false, true}) {
        const std::string empty_side = build_left ? "r_k" : "l_k";
        const std::string sql = "SELECT l_k, l_s, r_k, r_t FROM l JOIN r ON l_k = r_k WHERE " + empty_side + " > 1000";
        const Compiled c = RunPlan(*PlanBuilding(sql, build_left, std::vector<int>{0, 1}));
        EXPECT_EQ(c.top.build_left, build_left);
        EXPECT_EQ(c.top.build_rows, build_left ? l_rows_.size() : r_rows_.size());
        EXPECT_TRUE(c.result.tuples.empty());
    }

    // A hand-fed join: rows built, then a probe batch with no active rows.
    auto made = IntJoin(false);
    QueryContext ctx{nullptr, nullptr};
    auto build = made.first->MakeLocal();
    Batch rows = PairBatch({{1, 100}, {2, 200}});
    made.first->Consume(rows, *build);
    made.first->Combine(std::move(build));
    made.first->Finalize(ctx);
    Batch probe = PairBatch({{1, 10}, {2, 20}});
    probe.sel.Resize(0);
    auto local = made.second->MakeLocal();
    EXPECT_TRUE(Drain(*made.second, probe, local.get()).empty());
    // The local is ready for the next batch.
    Batch next = PairBatch({{2, 20}});
    EXPECT_EQ(Drain(*made.second, next, local.get()), (std::vector<std::pair<int64_t, int64_t>>{{20, 200}}));
}

TEST_F(ParallelHashJoinTest, BuildEitherSide) {
    const std::string sql = "SELECT l_k, l_v, l_s, r_w, r_t FROM l JOIN r ON l_k = r_k WHERE l_v < 3000";
    const QueryResult want = Run(sql, ExecutionMode::VECTORIZED);
    ASSERT_GT(want.tuples.size(), 1000u);
    for (const bool build_left : {false, true}) {
        const Compiled c = RunPlan(*PlanBuilding(sql, build_left));
        EXPECT_EQ(c.joins, 1u);
        EXPECT_EQ(c.pipelines, 2u);  // the build side, then the probe side into the result
        EXPECT_EQ(c.top.build_left, build_left);
        EXPECT_FALSE(c.top.has_residual);
        EXPECT_EQ(Difference(c.result, want), "") << "build_left " << build_left;
    }

    // The estimates decide: the smaller known side builds; a tie or an unknown builds right.
    const auto builds_left = [&](double left, double right) {
        auto root = Plan(sql);
        HashJoinPlanNode* join = TopHashJoin(root.get());
        join->est_left_rows = left;
        join->est_right_rows = right;
        return RunPlan(*root).top.build_left;
    };
    EXPECT_TRUE(builds_left(5.0, 6.0));
    EXPECT_FALSE(builds_left(6.0, 5.0));
    EXPECT_FALSE(builds_left(5.0, 5.0));
    EXPECT_FALSE(builds_left(-1.0, 5.0));
    EXPECT_FALSE(builds_left(5.0, -1.0));

    // Hand-fed: the same pairs whichever side builds; the left side's slots come first.
    const std::vector<std::pair<int64_t, int64_t>> build_rows = {{1, 100}, {2, 200}, {2, 201}, {4, 400}};
    const std::vector<std::pair<int64_t, int64_t>> probe_rows = {{2, 20}, {3, 30}, {1, 10}, {2, 21}};
    for (const bool build_left : {false, true}) {
        auto made = IntJoin(build_left);
        QueryContext ctx{nullptr, nullptr};
        auto build = made.first->MakeLocal();
        Batch rows = PairBatch(build_rows);
        made.first->Consume(rows, *build);
        made.first->Combine(std::move(build));
        made.first->Finalize(ctx);
        Batch probe = PairBatch(probe_rows);
        auto local = made.second->MakeLocal();
        auto got = Drain(*made.second, probe, local.get());
        if (build_left) {
            for (auto& p : got) std::swap(p.first, p.second);  // payload first when the left side builds
        }
        std::sort(got.begin(), got.end());
        EXPECT_EQ(got, (std::vector<std::pair<int64_t, int64_t>>{{10, 100}, {20, 200}, {20, 201}, {21, 200}, {21, 201}}))
            << "build_left " << build_left;
    }
}

TEST_F(ParallelHashJoinTest, ManyOutputPairsPerBatch) {
    // 100,000 build rows share key 7, ten have key 8.
    auto made = IntJoin(false);
    Sink& sink = *made.first;
    Operator& join = *made.second;
    QueryContext ctx{nullptr, nullptr};
    auto w1 = sink.MakeLocal();
    auto w2 = sink.MakeLocal();
    for (int64_t first = 0; first < 100000; first += kBatchSize) {
        std::vector<std::pair<int64_t, int64_t>> rows;
        for (int64_t i = first; i < std::min<int64_t>(first + kBatchSize, 100000); ++i) rows.emplace_back(7, i);
        Batch b = PairBatch(rows);
        sink.Consume(b, (first / kBatchSize) % 2 == 0 ? *w1 : *w2);
    }
    std::vector<std::pair<int64_t, int64_t>> eights;
    for (int64_t i = 0; i < 10; ++i) eights.emplace_back(8, 1000000 + i);
    Batch e = PairBatch(eights);
    sink.Consume(e, *w2);
    sink.Combine(std::move(w1));
    sink.Combine(std::move(w2));
    sink.Finalize(ctx);
    EXPECT_EQ(InspectHashJoin(join)->build_rows, 100010u);

    // One probe batch: key 7 once, key 8 once, key 9 not built.
    Batch probe = PairBatch({{9, -9}, {7, -7}, {5, -5}, {8, -8}});
    probe.sel.Data()[0] = 0;
    probe.sel.Data()[1] = 1;
    probe.sel.Data()[2] = 3;  // key 5 is not active
    probe.sel.Resize(3);
    auto local = join.MakeLocal();
    int calls = 0;
    std::vector<uint32_t> sizes;
    const auto got = Drain(join, probe, local.get(), &calls, &sizes);
    ASSERT_EQ(got.size(), 100010u);
    EXPECT_EQ(calls, 49);  // 100,010 pairs, 2,048 per batch
    for (size_t i = 0; i + 1 < sizes.size(); ++i) EXPECT_EQ(sizes[i], kBatchSize) << "batch " << i;
    std::vector<int> seen(100000, 0);
    size_t eight_pairs = 0;
    for (const auto& p : got) {
        if (p.first == -7) {
            ASSERT_GE(p.second, 0);
            ASSERT_LT(p.second, 100000);
            ++seen[static_cast<size_t>(p.second)];
        } else {
            EXPECT_EQ(p.first, -8);
            EXPECT_GE(p.second, 1000000);
            ++eight_pairs;
        }
    }
    EXPECT_EQ(eight_pairs, 10u);
    EXPECT_TRUE(std::all_of(seen.begin(), seen.end(), [](int n) { return n == 1; }));

    // Two workers take turns, each with its own cursor.
    Batch p1 = PairBatch({{7, 1}});
    Batch p2 = PairBatch({{8, 2}, {7, 3}});
    auto l1 = join.MakeLocal();
    auto l2 = join.MakeLocal();
    Batch o1;
    Batch o2;
    size_t n1 = 0;
    size_t n2 = 0;
    bool more1 = true;
    bool more2 = true;
    for (int guard = 0; (more1 || more2) && guard < 1000; ++guard) {
        if (more1) {
            more1 = join.Execute(p1, o1, l1.get()) == OpResult::HAVE_MORE_OUTPUT;
            n1 += o1.Active();
        }
        if (more2) {
            more2 = join.Execute(p2, o2, l2.get()) == OpResult::HAVE_MORE_OUTPUT;
            n2 += o2.Active();
        }
    }
    EXPECT_FALSE(more1 || more2);
    EXPECT_EQ(n1, 100000u);
    EXPECT_EQ(n2, 100010u);
}

TEST_F(ParallelHashJoinTest, ResidualPredicate) {
    // The planner puts a.x < b.y in a filter above the join.
    const std::string sql = "SELECT l_k, l_v, r_y, r_t FROM l JOIN r ON l_k = r_k WHERE l_v < r_y";
    const QueryResult want = Run(sql, ExecutionMode::VECTORIZED);
    ASSERT_GT(want.tuples.size(), 100u);
    ASSERT_LT(want.tuples.size(), 22100u);
    EXPECT_EQ(Check(sql), want.tuples.size());

    // Moved into the join's own predicate, on either side of the key, it's the residual.
    for (const bool key_first : {true, false}) {
        auto root = Plan(sql);
        ASSERT_EQ(root->children[0]->type, PlanNodeType::FILTER);
        auto* filter = static_cast<FilterPlanNode*>(root->children[0].get());
        ASSERT_EQ(filter->children[0]->type, PlanNodeType::HASH_JOIN);
        std::unique_ptr<PlanNode> join_node = std::move(filter->children[0]);
        auto* join = static_cast<HashJoinPlanNode*>(join_node.get());
        auto both = std::make_unique<BinaryOp>();
        both->op = BinaryOp::Op::AND;
        both->left = key_first ? std::move(join->join_predicate) : std::move(filter->predicate);
        both->right = key_first ? std::move(filter->predicate) : std::move(join->join_predicate);
        join->join_predicate = std::move(both);
        root->children[0] = std::move(join_node);
        for (const bool build_left : {false, true}) {
            join->est_left_rows = build_left ? 1.0 : 2.0;
            join->est_right_rows = build_left ? 2.0 : 1.0;
            const Compiled c = RunPlan(*root);
            EXPECT_TRUE(c.top.has_residual);
            EXPECT_EQ(c.top.build_left, build_left);
            EXPECT_EQ(Difference(c.result, want), "") << "key_first " << key_first << ", build_left " << build_left;
        }
    }

    // A second key equality stays in the join, as its residual.
    const std::string two_keys = "SELECT l_k, l_j, r_j, r_t FROM l JOIN r ON l_k = r_k AND l_j = r_j";
    EXPECT_GT(Check(two_keys), 1000u);
    EXPECT_TRUE(RunPlan(*Plan(two_keys)).top.has_residual);
    // One key alone leaves no residual: the key's equality is never tested twice.
    EXPECT_FALSE(RunPlan(*Plan("SELECT l_k, r_t FROM l JOIN r ON l_k = r_k")).top.has_residual);
}

TEST_F(ParallelHashJoinTest, SameSideEqualityIsSkippedAnIntendedDifference) {
    // The ON's first equality names two l columns. Today's executor can't pick
    // a key from it and throws; this one skips it, joins on l_k = r_k
    // and keeps it in the residual.
    const std::string sql = "SELECT l_k, l_j, l_m, r_t FROM l JOIN r ON l_j = l_m AND l_k = r_k";
    EXPECT_EQ(ErrorOf([&] { (void)Run(sql, ExecutionMode::VECTORIZED); }), "HashJoinExecutor: could not resolve join key columns");
    const QueryResult want = Run("SELECT l_k, l_j, l_m, r_t FROM l JOIN r ON l_k = r_k WHERE l_j = l_m", ExecutionMode::VECTORIZED);
    ASSERT_GT(want.tuples.size(), 100u);
    for (const std::vector<int>& order : {std::vector<int>{0, 1}, std::vector<int>{1, 0}}) {
        EXPECT_EQ(Difference(Run(sql, ExecutionMode::PARALLEL, 1, order), want), "") << "order " << order[0];
    }
    EXPECT_TRUE(RunPlan(*Plan(sql)).top.has_residual);
}

TEST_F(ParallelHashJoinTest, SharedColumnNameKeysResolveByName) {
    // Both tables have a column k. Binding ignores qualifiers, so ka.k and kb.k
    // both bind to the left child's k; the key is found by name in each child.
    Load("ka", Schema({Column("k", ColumnType::Integer()), Column("x", ColumnType::Integer())}), "1|5|\n2|5|\n3|6|\n");
    Load("kb", Schema({Column("k", ColumnType::Integer()), Column("y", ColumnType::Integer())}), "1|5|\n3|6|\n4|5|\n");
    struct Case {
        std::string sql;
        size_t rows;
    };
    const std::vector<Case> cases = {
        {"SELECT x, y FROM ka JOIN kb ON ka.k = kb.k AND ka.x = kb.y", 2},
        {"SELECT x, y FROM ka JOIN kb ON ka.k = kb.k", 2},
        {"SELECT x, y FROM ka, kb WHERE ka.k = kb.k AND x = y", 2},
        // All modes give 5, the same wrong answer: x = y is the key, and the
        // residual's ka.k = kb.k reads one column twice (binding ignores qualifiers).
        {"SELECT x, y FROM ka JOIN kb ON ka.x = kb.y AND ka.k = kb.k", 5},
    };
    for (const Case& c : cases) {
        SCOPED_TRACE(c.sql);
        const QueryResult want = Run(c.sql, ExecutionMode::VECTORIZED, 1);
        EXPECT_EQ(want.tuples.size(), c.rows);
        // How a run differs from want, or what it threw.
        const auto differs = [&](auto run) {
            try {
                return Difference(run(), want);
            } catch (const std::exception& e) {
                return std::string("threw: ") + e.what();
            }
        };
        for (const std::vector<int>& order : {std::vector<int>{0, 1}, std::vector<int>{1, 0}}) {
            EXPECT_EQ(differs([&] { return Run(c.sql, ExecutionMode::PARALLEL, 1, order); }), "") << "order " << order[0];
        }
        for (const bool build_left : {false, true}) {
            EXPECT_EQ(differs([&] { return RunPlan(*PlanBuilding(c.sql, build_left)).result; }), "") << "build_left " << build_left;
        }
    }
}

TEST_F(ParallelHashJoinTest, NoEquiKeyIsRefused) {
    // Only the same-side equality left: no key, today's error text.
    auto root = Plan("SELECT l_k, r_k FROM l JOIN r ON l_j = l_m AND l_k = r_k");
    HashJoinPlanNode* join = TopHashJoin(root.get());
    ASSERT_NE(join, nullptr);
    auto* both = static_cast<BinaryOp*>(join->join_predicate.get());
    ASSERT_EQ(both->op, BinaryOp::Op::AND);
    join->join_predicate = std::move(both->left);
    QueryContext ctx{db_->GetBufferPoolManager(), db_->GetCatalog(), 1};
    EXPECT_EQ(ErrorOf([&] { (void)CompilePlan(*root, ctx); }), "HashJoinExecutor: no equi-join predicate found");
}

//  Key types

TEST_F(ParallelHashJoinTest, MixedIntegerKeysMatchByValueAnIntendedDifference) {
    // ka_i = r and kb_b = kb_i = 2r: twenty keys meet.
    std::string a;
    std::string b;
    for (uint32_t r = 0; r < 40; ++r) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%u|%.1f|\n", r, r / 2.0);
        a += buf;
        std::snprintf(buf, sizeof(buf), "%u|%u|%u.%s|\n", 2 * r, 2 * r, r, r % 2 == 0 ? "00" : "25");
        b += buf;
    }
    Load("ka", Schema({Column("ka_i", ColumnType::Integer()), Column("ka_f", ColumnType::Double())}), a);
    Load("kb", Schema({Column("kb_i", ColumnType::Integer()), Column("kb_b", ColumnType::BigInt()), Column("kb_m", ColumnType::Numeric(15, 2))}), b);

    // Today's hash mixes in the type, so INTEGER and BIGINT keys never meet,
    // a silent wrong answer. This executor compares keys by value.
    const std::string mixed = "SELECT ka_i, kb_i FROM ka JOIN kb ON ka_i = kb_b";
    EXPECT_TRUE(Run(mixed, ExecutionMode::VECTORIZED).tuples.empty());
    const QueryResult same_type = Run("SELECT ka_i, kb_i FROM ka JOIN kb ON ka_i = kb_i", ExecutionMode::VECTORIZED);
    ASSERT_EQ(same_type.tuples.size(), 20u);
    for (const std::vector<int>& order : {std::vector<int>{0, 1}, std::vector<int>{1, 0}}) {
        EXPECT_EQ(Difference(Run(mixed, ExecutionMode::PARALLEL, 1, order), same_type), "") << "order " << order[0];
    }

    // An integer against a decimal meets r.00, by value (today: never).
    const std::string to_decimal = "SELECT ka_i, kb_m FROM ka JOIN kb ON ka_i = kb_m";
    EXPECT_TRUE(Run(to_decimal, ExecutionMode::VECTORIZED).tuples.empty());
    for (const std::vector<int>& order : {std::vector<int>{0, 1}, std::vector<int>{1, 0}}) {
        const QueryResult got = Run(to_decimal, ExecutionMode::PARALLEL, 1, order);
        ASSERT_EQ(got.tuples.size(), 20u) << "order " << order[0];
        for (const Tuple& t : got.tuples) EXPECT_EQ(static_cast<double>(t.GetValue(got.schema, 0).integer_), t.GetValue(got.schema, 1).decimal_);
    }

    // A double against an integer compares as doubles (today: never).
    // ka_f = r / 2 meets 0, 2, ..., 18.
    const std::string to_double = "SELECT ka_f, kb_i FROM ka JOIN kb ON ka_f = kb_i";
    EXPECT_TRUE(Run(to_double, ExecutionMode::VECTORIZED).tuples.empty());
    for (const std::vector<int>& order : {std::vector<int>{0, 1}, std::vector<int>{1, 0}}) {
        const QueryResult got = Run(to_double, ExecutionMode::PARALLEL, 1, order);
        ASSERT_EQ(got.tuples.size(), 10u) << "order " << order[0];
        for (const Tuple& t : got.tuples) EXPECT_EQ(t.GetValue(got.schema, 0).decimal_, static_cast<double>(t.GetValue(got.schema, 1).integer_));
    }
}

TEST_F(ParallelHashJoinTest, DecimalDoubleDateAndStringKeys) {
    std::string a;
    std::string b;
    for (uint32_t r = 0; r < 40; ++r) {
        char buf[256];
        // ka_f starts at -0.0, which must meet 0.00.
        std::snprintf(buf, sizeof(buf), "%u.25|%.1f|1995-01-%02u|s%u|%u|%u.00|\n", r, r == 0 ? -0.0 : r / 2.0, r % 28 + 1, r % 13, r, r);
        a += buf;
        // kb_w's last row has 38 digits: rescaled to two decimals it would pass 38.
        const std::string w = r == 39 ? std::string(38, '9') : std::to_string(r);
        std::snprintf(buf, sizeof(buf), "%u.%s|%u.%s|%u.2500|1995-01-%02u|s%u|%u|%s|\n", r, r % 2 == 0 ? "00" : "25", r, r % 3 == 0 ? "250" : "255", r,
                      r % 20 + 1, r % 7, r, w.c_str());
        b += buf;
    }
    Load("ka", Schema({Column("ka_m", ColumnType::Numeric(15, 2)), Column("ka_f", ColumnType::Double()), Column("ka_d", ColumnType::Date()),
                       Column("ka_s", ColumnType::Varchar(16)), Column("ka_i", ColumnType::Integer()), Column("ka_c", ColumnType::Numeric(15, 2))}),
         a);
    Load("kb", Schema({Column("kb_m", ColumnType::Numeric(15, 2)), Column("kb_n", ColumnType::Numeric(25, 3)), Column("kb_q", ColumnType::Numeric(12, 4)),
                       Column("kb_d", ColumnType::Date()), Column("kb_s", ColumnType::Text()), Column("kb_i", ColumnType::Integer()),
                       Column("kb_w", ColumnType::Numeric(38, 0))}),
         b);

    // Scales 2 against 3 (wide) and 4 meet exactly; 2 against 2; doubles against decimals.
    EXPECT_EQ(Check("SELECT ka_m, kb_n, ka_i, kb_i FROM ka JOIN kb ON ka_m = kb_n"), 14u);
    EXPECT_EQ(Check("SELECT ka_m, kb_q, ka_i, kb_i FROM ka JOIN kb ON ka_m = kb_q"), 40u);
    EXPECT_EQ(Check("SELECT ka_m, kb_m, ka_i, kb_i FROM ka JOIN kb ON ka_m = kb_m"), 20u);
    EXPECT_EQ(Check("SELECT ka_f, kb_m, ka_i, kb_i FROM ka JOIN kb ON ka_f = kb_m"), 10u);
    // A key that would pass 38 digits once rescaled matches nothing, and nothing throws.
    EXPECT_EQ(Check("SELECT ka_c, kb_w, ka_i, kb_i FROM ka JOIN kb ON ka_c = kb_w"), 39u);
    EXPECT_GT(Check("SELECT ka_d, kb_d, ka_i, kb_i FROM ka JOIN kb ON ka_d = kb_d"), 30u);
    EXPECT_GT(Check("SELECT ka_s, kb_s, ka_i, kb_i FROM ka JOIN kb ON ka_s = kb_s"), 30u);

    // Keys that can't compare are refused before anything runs, left side's type first.
    EXPECT_EQ(ErrorOf([&] { (void)Run("SELECT ka_i FROM ka JOIN kb ON ka_d = kb_s", ExecutionMode::PARALLEL, 1, std::vector<int>{0, 1}); }),
              "the parallel executor can't join keys of types DATE and STRING");
    EXPECT_EQ(ErrorOf([&] { (void)Run("SELECT ka_i FROM ka JOIN kb ON ka_d = kb_s", ExecutionMode::PARALLEL, 1, std::vector<int>{1, 0}); }),
              "the parallel executor can't join keys of types STRING and DATE");
    EXPECT_EQ(ErrorOf([&] { (void)Run("SELECT ka_i FROM ka JOIN kb ON ka_s = kb_i", ExecutionMode::PARALLEL, 1, std::vector<int>{0, 1}); }),
              "the parallel executor can't join keys of types STRING and INTEGER");
    EXPECT_EQ(ErrorOf([&] { (void)Run("SELECT ka_i FROM ka JOIN kb ON ka_d = kb_i", ExecutionMode::PARALLEL, 1, std::vector<int>{0, 1}); }),
              "the parallel executor can't join keys of types DATE and INTEGER");
}

//  Strings and partitions

TEST_F(ParallelHashJoinTest, PayloadStringsOutliveMorsel) {
    // Hand-fed: string keys and payloads whose bytes are wiped once consumed.
    const InputLayout in = LayoutOf({kStr, kStr, kInt});
    JoinSpec spec;
    spec.build_key = Col(0, in);
    spec.probe_key = Col(0, LayoutOf({kStr}));
    spec.build_payload = {1, 2};
    spec.probe_slots = {0};
    spec.types = {kStr, kStr, kInt};
    auto made = MakeHashJoin(std::move(spec));
    QueryContext ctx{nullptr, nullptr};

    std::vector<std::string> keys = {"k-big", "k-ünï", "k-empty", "k-dict"};
    std::vector<std::string> payloads = {BigText(3), "ünï épée", "", "from a dictionary"};
    const std::vector<std::string> want_payloads = payloads;
    auto build_local = made.first->MakeLocal();

    // Rows 0..2 come plain.
    Batch plain;
    ShapeBatch(plain, {kStr, kStr, kInt});
    plain.Reset(3);
    for (uint32_t i = 0; i < 3; ++i) {
        plain.slots[0].Str()[i] = Ref(keys[i]);
        plain.slots[1].Str()[i] = Ref(payloads[i]);
        plain.slots[2].I64()[i] = i;
    }
    made.first->Consume(plain, *build_local);

    // Row 3 comes through dictionaries, as a scan hands them up.
    std::vector<StringRef> key_entries = {Ref(keys[0]), Ref(keys[3])};
    std::vector<StringRef> payload_entries = {Ref(payloads[3])};
    Batch dict;
    ShapeBatch(dict, {kStr, kStr, kInt});
    dict.Reset(1);
    dict.slots[0].SetDictionary(&key_entries);
    dict.slots[0].Codes()[0] = 1;
    dict.slots[1].SetDictionary(&payload_entries);
    dict.slots[1].Codes()[0] = 0;
    dict.slots[2].I64()[0] = 3;
    made.first->Consume(dict, *build_local);
    made.first->Combine(std::move(build_local));

    // The batches' bytes and dictionaries die with their morsel.
    for (std::string& s : keys) std::fill(s.begin(), s.end(), '#');
    for (std::string& s : payloads) std::fill(s.begin(), s.end(), '#');
    key_entries.assign(key_entries.size(), StringRef{});
    payload_entries.assign(payload_entries.size(), StringRef{});
    made.first->Finalize(ctx);

    const std::vector<std::string> probe_keys = {"k-dict", "k-big", "nope", "k-empty", "k-ünï"};
    Batch probe;
    ShapeBatch(probe, {kStr});
    probe.Reset(static_cast<uint32_t>(probe_keys.size()));
    for (uint32_t i = 0; i < probe_keys.size(); ++i) probe.slots[0].Str()[i] = Ref(probe_keys[i]);
    auto local = made.second->MakeLocal();
    Batch out;
    ASSERT_EQ(made.second->Execute(probe, out, local.get()), OpResult::NEED_INPUT);
    ASSERT_EQ(out.Active(), 4u);
    std::map<std::string, std::pair<std::string, int64_t>> got;
    for (uint32_t k = 0; k < out.Active(); ++k) {
        const uint32_t r = out.sel[k];
        got[std::string(View(out.slots[0].StrAt(r)))] = {std::string(View(out.slots[1].StrAt(r))), out.slots[2].I64()[r]};
    }
    EXPECT_EQ(got["k-big"], std::make_pair(want_payloads[0], int64_t{0}));
    EXPECT_EQ(got["k-ünï"], std::make_pair(want_payloads[1], int64_t{1}));
    EXPECT_EQ(got["k-empty"], std::make_pair(want_payloads[2], int64_t{2}));
    EXPECT_EQ(got["k-dict"], std::make_pair(want_payloads[3], int64_t{3}));

    // Through SQL: two morsels on each side, 8,000-byte and multibyte strings on one.
    const auto pl_text = [](uint32_t r) { return r % 5000 == 7 ? BigText(r) : "w" + std::to_string(r) + (r % 9 == 0 ? " ünï" : ""); };
    std::string pl;
    std::string pr;
    for (uint32_t r = 0; r < kBigRows; ++r) {
        pl += std::to_string(r) + "|" + pl_text(r) + "|\n";
        pr += std::to_string(kBigRows - 1 - r) + "|" + kWords[r % 5] + "|\n";
    }
    Load("pl", Schema({Column("pl_k", ColumnType::Integer()), Column("pl_s", ColumnType::Text())}), pl);
    Load("pr", Schema({Column("pr_k", ColumnType::Integer()), Column("pr_t", ColumnType::Varchar(16))}), pr);
    ASSERT_EQ(db_->GetCatalog()->GetTable("pl")->columnar->RowGroupCount(), 2u);

    const std::string sql = "SELECT pl_k, pl_s, pr_t FROM pl JOIN pr ON pl_k = pr_k";
    for (const std::vector<int>& order : {std::vector<int>{0, 1}, std::vector<int>{1, 0}}) {
        for (const bool build_left : {false, true}) {
            const Compiled c = RunPlan(*PlanBuilding(sql, build_left, order));
            ASSERT_EQ(c.result.tuples.size(), kBigRows);
            size_t big = 0;
            for (const Tuple& t : c.result.tuples) {
                const auto k = static_cast<uint32_t>(t.GetValue(c.result.schema, 0).integer_);
                const std::string s = t.GetValue(c.result.schema, 1).varchar_;
                ASSERT_EQ(s, pl_text(k)) << "key " << k;
                ASSERT_EQ(t.GetValue(c.result.schema, 2).varchar_, kWords[(kBigRows - 1 - k) % 5]) << "key " << k;
                big += s.size() == 8000 ? 1 : 0;
            }
            EXPECT_EQ(big, 14u);
        }
    }
}

TEST_F(ParallelHashJoinTest, ManyPartitions) {
    // 300,000 build rows from two workers, key 3i + 1 and payload i.
    constexpr int64_t kRows = 300000;
    ASSERT_GT(JoinPartitionCount(kRows, 1), 1u);
    EXPECT_EQ(JoinPartitionCount(kJoinRowsPerPartition, 18), 1u);
    auto made = IntJoin(false);
    QueryContext ctx{nullptr, nullptr};
    auto w1 = made.first->MakeLocal();
    auto w2 = made.first->MakeLocal();
    for (int64_t first = 0; first < kRows; first += kBatchSize) {
        std::vector<std::pair<int64_t, int64_t>> rows;
        for (int64_t i = first; i < std::min<int64_t>(first + kBatchSize, kRows); ++i) rows.emplace_back(3 * i + 1, i);
        Batch b = PairBatch(rows);
        made.first->Consume(b, (first / kBatchSize) % 3 == 0 ? *w1 : *w2);
    }
    made.first->Combine(std::move(w1));
    made.first->Combine(std::move(w2));
    made.first->Finalize(ctx);
    const HashJoinInfo info = *InspectHashJoin(*made.second);
    EXPECT_EQ(info.build_rows, static_cast<size_t>(kRows));
    EXPECT_EQ(info.partitions, JoinPartitionCount(kRows, 1));

    // Probes keys 0 to 3 * kRows; one in three matches exactly its row.
    std::vector<int> seen(kRows, 0);
    auto local = made.second->MakeLocal();
    size_t pairs = 0;
    for (int64_t first = 0; first < 3 * kRows; first += kBatchSize) {
        std::vector<std::pair<int64_t, int64_t>> rows;
        for (int64_t key = first; key < std::min<int64_t>(first + kBatchSize, 3 * kRows); ++key) rows.emplace_back(key, key);
        Batch probe = PairBatch(rows);
        for (const auto& p : Drain(*made.second, probe, local.get())) {
            ASSERT_EQ(p.first, 3 * p.second + 1);
            ++seen[static_cast<size_t>(p.second)];
            ++pairs;
        }
    }
    EXPECT_EQ(pairs, static_cast<size_t>(kRows));
    EXPECT_TRUE(std::all_of(seen.begin(), seen.end(), [](int n) { return n == 1; }));
}

TEST_F(ParallelHashJoinTest, DictionaryProbeKeys) {
    // Built from plain strings, probed through a dictionary, as a scan hands strings up.
    const InputLayout two = LayoutOf({kStr, kInt});
    JoinSpec spec;
    spec.build_key = Col(0, two);
    spec.probe_key = Col(0, two);
    spec.build_payload = {1};
    spec.probe_slots = {0, 1};
    spec.types = {kStr, kInt, kInt};
    auto made = MakeHashJoin(std::move(spec));
    QueryContext ctx{nullptr, nullptr};
    const std::vector<std::string> keys = {"alpha", "épée", "", "beta"};
    Batch build;
    ShapeBatch(build, {kStr, kInt});
    build.Reset(4);
    for (uint32_t i = 0; i < 4; ++i) {
        build.slots[0].Str()[i] = Ref(keys[i]);
        build.slots[1].I64()[i] = 100 + i;
    }
    auto build_local = made.first->MakeLocal();
    made.first->Consume(build, *build_local);
    made.first->Combine(std::move(build_local));
    made.first->Finalize(ctx);

    // Five entries, one never built; codes repeat and skip entries; row 2 is inactive.
    const std::vector<std::string> words = {"beta", "gamma", "", "épée", "alpha"};
    std::vector<StringRef> entries;
    for (const std::string& w : words) entries.push_back(Ref(w));
    const uint32_t codes[] = {3, 1, 0, 4, 2, 3};
    Batch probe;
    ShapeBatch(probe, {kStr, kInt});
    probe.Reset(6);
    probe.slots[0].SetDictionary(&entries);
    for (uint32_t i = 0; i < 6; ++i) {
        probe.slots[0].Codes()[i] = codes[i];
        probe.slots[1].I64()[i] = i;
    }
    const uint16_t active[] = {0, 1, 3, 4, 5};
    std::copy(std::begin(active), std::end(active), probe.sel.Data());
    probe.sel.Resize(5);

    auto local = made.second->MakeLocal();
    Batch out;
    ASSERT_EQ(made.second->Execute(probe, out, local.get()), OpResult::NEED_INPUT);
    EXPECT_FALSE(out.slots[0].IsDictionary());
    std::vector<std::tuple<std::string, int64_t, int64_t>> got;
    for (uint32_t k = 0; k < out.Active(); ++k) {
        const uint32_t r = out.sel[k];
        got.emplace_back(std::string(View(out.slots[0].StrAt(r))), out.slots[1].I64()[r], out.slots[2].I64()[r]);
    }
    std::sort(got.begin(), got.end());
    EXPECT_EQ(got, (std::vector<std::tuple<std::string, int64_t, int64_t>>{{"", 4, 102}, {"alpha", 3, 100}, {"épée", 0, 101}, {"épée", 5, 101}}));
}

TEST_F(ParallelHashJoinTest, NanKeysNeverMatch) {
    // Hand-fed doubles: a NaN equals nothing, as Value's == says, whatever its bits.
    const VType kDouble{Kind::DOUBLE, 0, false, TypeId::DECIMAL};
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const InputLayout two = LayoutOf({kDouble, kInt});
    JoinSpec spec;
    spec.build_key = Col(0, two);
    spec.probe_key = Col(0, two);
    spec.build_payload = {1};
    spec.probe_slots = {1};
    spec.types = {kInt, kInt};
    auto made = MakeHashJoin(std::move(spec));
    QueryContext ctx{nullptr, nullptr};
    const auto doubles = [&](const std::vector<double>& keys, int64_t first) {
        Batch b;
        ShapeBatch(b, {kDouble, kInt});
        b.Reset(static_cast<uint32_t>(keys.size()));
        for (size_t i = 0; i < keys.size(); ++i) {
            b.slots[0].F64()[i] = keys[i];
            b.slots[1].I64()[i] = first + static_cast<int64_t>(i);
        }
        return b;
    };
    Batch build = doubles({nan, 1.5, -nan, 2.0}, 0);
    auto build_local = made.first->MakeLocal();
    made.first->Consume(build, *build_local);
    made.first->Combine(std::move(build_local));
    made.first->Finalize(ctx);
    EXPECT_EQ(InspectHashJoin(*made.second)->build_rows, 2u);  // the NaN rows can never match

    Batch probe = doubles({nan, 1.5, -nan, 2.0, -0.0}, 10);
    auto local = made.second->MakeLocal();
    auto got = Drain(*made.second, probe, local.get());
    std::sort(got.begin(), got.end());
    EXPECT_EQ(got, (std::vector<std::pair<int64_t, int64_t>>{{11, 1}, {13, 3}}));

    // Through SQL, against today's executor, which never matches a NaN either.
    Load("na", Schema({Column("na_f", ColumnType::Double()), Column("na_i", ColumnType::Integer())}), "nan|0\n1.5|1\nNaN|2\n-nan|3\n");
    Load("nb", Schema({Column("nb_f", ColumnType::Double()), Column("nb_i", ColumnType::Integer())}), "nan|0\n1.5|1\nNAN|2\n");
    const std::string sql = "SELECT na_i, nb_i FROM na JOIN nb ON na_f = nb_f";
    EXPECT_EQ(Run(sql, ExecutionMode::VECTORIZED).tuples.size(), 1u);
    EXPECT_EQ(Check(sql), 1u);
}

//  TPC-H shapes

TEST_F(ParallelHashJoinTest, Q3AndQ5ShapeAnswers) {
    const std::string dir = FindTpchDir();
    if (dir.empty()) GTEST_SKIP() << "TPC-H SF=0.01 data not found. Run bench/tpch/generate_tpch_data.sh first.";
    const auto tpch = LoadTpch(dir);
    for (const char* tag : {"Q3", "Q5"}) {
        const std::string sql = tpch::FindQuery(tag)->sql;
        const QueryResult want = RunOn(*tpch, sql, ExecutionMode::VECTORIZED);
        ASSERT_GT(want.tuples.size(), 0u) << tag;
        for (const unsigned threads : {1u, 18u}) {
            const QueryResult got = RunOn(*tpch, sql, ExecutionMode::PARALLEL, threads);
            EXPECT_TRUE(tpch::SameResult(got, want)) << tag << ", threads " << threads << ": " << Difference(got, want);
        }
    }
}

TEST_F(ParallelHashJoinTest, Q10BuildsOrdersSide) {
    const std::string dir = FindTpchDir();
    if (dir.empty()) GTEST_SKIP() << "TPC-H SF=0.01 data not found. Run bench/tpch/generate_tpch_data.sh first.";
    const auto tpch = LoadTpch(dir);
    const std::string q10 = tpch::FindQuery("Q10")->sql;

    // The last join adds lineitem to a side holding orders.
    const auto root = PlanOn(*tpch, q10);
    const HashJoinPlanNode* last = TopHashJoin(root.get());
    ASSERT_NE(last, nullptr);
    const bool orders_left = Holds(*last->children[0], "orders");
    ASSERT_NE(orders_left, Holds(*last->children[1], "orders"));
    ASSERT_TRUE(Holds(*last->children[orders_left ? 1 : 0], "lineitem"));
    ASSERT_FALSE(Holds(*last->children[orders_left ? 0 : 1], "lineitem"));

    // It builds the orders side: the window's 611 orders, never the 14,902 returned lineitems.
    const auto count = [&](const std::string& sql) {
        const QueryResult r = RunOn(*tpch, sql, ExecutionMode::VECTORIZED);
        return static_cast<int64_t>(r.tuples.at(0).GetValue(r.schema, 0).integer_);
    };
    const int64_t window = count("SELECT COUNT(*) FROM orders WHERE o_orderdate >= '1993-10-01' AND o_orderdate < '1994-01-01'");
    const int64_t returned = count("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'");
    ASSERT_GT(window, 0);
    ASSERT_GT(returned, window);
    const Compiled c = RunPlanOn(*tpch, *root);
    EXPECT_EQ(c.top.build_left, orders_left);
    EXPECT_EQ(c.top.build_rows, static_cast<size_t>(window));

    // And its answer is today's.
    const QueryResult want = RunOn(*tpch, q10, ExecutionMode::VECTORIZED);
    ASSERT_EQ(want.tuples.size(), 20u);
    EXPECT_TRUE(tpch::SameResult(c.result, want)) << Difference(c.result, want);
    for (const unsigned threads : {1u, 18u}) {
        const QueryResult got = RunOn(*tpch, q10, ExecutionMode::PARALLEL, threads);
        EXPECT_TRUE(tpch::SameResult(got, want)) << "threads " << threads << ": " << Difference(got, want);
    }
}

}  // namespace
}  // namespace shilmandb::parallel
