#include <gtest/gtest.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "catalog/catalog.hpp"
#include "engine/database.hpp"
#include "executor/parallel/basic_operators.hpp"
#include "executor/parallel/batch.hpp"
#include "executor/parallel/column_buffer.hpp"
#include "executor/parallel/compiler.hpp"
#include "executor/parallel/expression.hpp"
#include "executor/parallel/hash_aggregate.hpp"
#include "executor/parallel/hash_join.hpp"
#include "executor/parallel/key_hash.hpp"
#include "executor/parallel/memory_budget.hpp"
#include "executor/parallel/parallel_executor.hpp"
#include "executor/parallel/pipeline.hpp"
#include "executor/parallel/spill_partitions.hpp"
#include "executor/parallel/temp_file.hpp"
#include "executor/parallel/worker_pool.hpp"
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

// How a hand-fed batch holds its strings: plain, one dictionary entry per row,
// or shared entries (each distinct string once, after one no row uses).
enum class Coding { PLAIN, ENTRY_PER_ROW, SHARED_ENTRIES };

// Points slot's rows at strs in the given coding; entries backs a dictionary.
void FillStrings(Vector& slot, const std::vector<std::string>& strs, Coding coding, std::vector<StringRef>& entries) {
    entries.clear();
    if (coding == Coding::PLAIN) {
        for (size_t i = 0; i < strs.size(); ++i) slot.Str()[i] = Ref(strs[i]);
        return;
    }
    if (coding == Coding::SHARED_ENTRIES) entries.push_back(StringRef{"never used", 10});
    std::map<std::string, uint32_t> code_of;
    for (size_t i = 0; i < strs.size(); ++i) {
        const auto found = code_of.find(strs[i]);
        if (coding == Coding::SHARED_ENTRIES && found != code_of.end()) {
            slot.Codes()[i] = found->second;
            continue;
        }
        code_of[strs[i]] = static_cast<uint32_t>(entries.size());
        slot.Codes()[i] = static_cast<uint32_t>(entries.size());
        entries.push_back(Ref(strs[i]));
    }
    slot.SetDictionary(&entries);
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
    // Both tables have a column k. Binding ignores qualifiers,
    // so ka.k and kb.k both bind to the left child's k;
    // the key is found by name in each child.
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

// Slot 0 behind an expression, never read in place:
// CASE WHEN c1 = c1 THEN c0 ELSE c0 END.
std::unique_ptr<CompiledExpr> ComputedStringKey(const InputLayout& in) {
    const auto ref = [](uint32_t c) {
        auto r = std::make_unique<ColumnRef>();
        r->column_name = "c" + std::to_string(c);
        r->bound_index = c;
        return r;
    };
    auto same = std::make_unique<BinaryOp>();
    same->op = BinaryOp::Op::EQ;
    same->left = ref(1);
    same->right = ref(1);
    CaseExpression pick;
    pick.when_clauses.emplace_back(std::move(same), ref(0));
    pick.else_clause = ref(0);
    return CompiledExpr::Compile(pick, in);
}

TEST_F(ParallelHashJoinTest, DictionaryAndPlainKeysPairAlike) {
    // Key m as text: long, empty and multibyte ones among them.
    const auto key = [](uint32_t m) -> std::string {
        if (m == 5) return std::string(300, 'x') + "-5";
        if (m == 7) return "";
        if (m == 11) return "épée";
        return "k-" + std::to_string(m);
    };
    constexpr uint32_t kBuildRows = 3000;
    constexpr uint32_t kProbeRows = 2500;
    constexpr uint32_t kBuildBatch = 500;
    constexpr uint32_t kProbeBatch = 400;
    const Coding cycle[] = {Coding::PLAIN, Coding::SHARED_ENTRIES, Coding::ENTRY_PER_ROW, Coding::SHARED_ENTRIES};
    using Pick = std::function<Coding(size_t)>;
    const Pick cycled = [&cycle](size_t b) { return cycle[b % 4]; };
    const Pick shifted = [&cycle](size_t b) { return cycle[(b + 1) % 4]; };
    const Pick shared = [](size_t) { return Coding::SHARED_ENTRIES; };
    const Pick plain = [](size_t) { return Coding::PLAIN; };
    // Each side's coding, and whether its key is computed rather than read in place.
    struct Run {
        Pick build;
        Pick probe;
        bool computed_build;
        bool computed_probe;
    };
    const std::vector<Run> runs = {
        {cycled, shifted, false, false},
        {shared, plain, false, false},
        {plain, shared, false, false},
        {cycled, shifted, true, false},
        {shifted, cycled, false, true},
    };

    // Build row x has key x % 37 and payload x;
    // probe row j has key j % 45 and value j.
    // Every fifth build row and every fourth probe row is inactive.
    const InputLayout two = LayoutOf({kStr, kInt});
    for (size_t r = 0; r < runs.size(); ++r) {
        JoinSpec spec;
        spec.build_key = runs[r].computed_build ? ComputedStringKey(two) : Col(0, two);
        spec.probe_key = runs[r].computed_probe ? ComputedStringKey(two) : Col(0, two);
        spec.build_payload = {1};
        spec.probe_slots = {1};
        spec.types = {kInt, kInt};
        Batch shape;
        ShapeBatch(shape, {kStr, kInt});
        EXPECT_EQ(spec.build_key->Direct(shape) == nullptr, runs[r].computed_build) << "run " << r;
        EXPECT_EQ(spec.probe_key->Direct(shape) == nullptr, runs[r].computed_probe) << "run " << r;
        auto made = MakeHashJoin(std::move(spec));
        QueryContext ctx{nullptr, nullptr};

        auto w0 = made.first->MakeLocal();
        auto w1 = made.first->MakeLocal();
        for (uint32_t at = 0; at < kBuildRows; at += kBuildBatch) {
            const size_t b = at / kBuildBatch;
            std::vector<std::string> strs;
            std::vector<StringRef> entries;
            Batch build;
            ShapeBatch(build, {kStr, kInt});
            build.Reset(kBuildBatch);
            uint32_t kept = 0;
            for (uint32_t i = 0; i < kBuildBatch; ++i) {
                strs.push_back(key((at + i) % 37));
                build.slots[1].I64()[i] = at + i;
                if ((at + i) % 5 != 4) build.sel.Data()[kept++] = static_cast<uint16_t>(i);
            }
            build.sel.Resize(kept);
            FillStrings(build.slots[0], strs, runs[r].build(b), entries);
            made.first->Consume(build, b % 2 == 0 ? *w0 : *w1);
        }
        made.first->Combine(std::move(w0));
        made.first->Combine(std::move(w1));
        made.first->Finalize(ctx);

        auto local = made.second->MakeLocal();
        std::vector<std::pair<int64_t, int64_t>> got;
        std::vector<std::pair<int64_t, int64_t>> want;
        for (uint32_t at = 0; at < kProbeRows; at += kProbeBatch) {
            const uint32_t n = std::min(kProbeBatch, kProbeRows - at);
            std::vector<std::string> strs;
            std::vector<StringRef> entries;
            Batch probe;
            ShapeBatch(probe, {kStr, kInt});
            probe.Reset(n);
            uint32_t kept = 0;
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t j = at + i;
                strs.push_back(key(j % 45));
                probe.slots[1].I64()[i] = j;
                if (i % 4 == 3) continue;
                probe.sel.Data()[kept++] = static_cast<uint16_t>(i);
                for (uint32_t x = j % 45; j % 45 < 37 && x < kBuildRows; x += 37) {
                    if (x % 5 != 4) want.emplace_back(j, x);
                }
            }
            probe.sel.Resize(kept);
            FillStrings(probe.slots[0], strs, runs[r].probe(at / kProbeBatch), entries);
            const auto pairs = Drain(*made.second, probe, local.get());
            got.insert(got.end(), pairs.begin(), pairs.end());
        }
        std::sort(got.begin(), got.end());
        std::sort(want.begin(), want.end());
        EXPECT_GT(want.size(), 100000u);
        EXPECT_EQ(got.size(), want.size()) << "run " << r;
        EXPECT_TRUE(got == want) << "run " << r;
    }
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

//  Spilling

constexpr unsigned kSpillThreads = 4;
constexpr size_t kSpillMorsels = 8;  // enough for every worker

const VType kDouble{Kind::DOUBLE, 0, false, TypeId::DECIMAL};
const VType kWide3{Kind::DECIMAL, 3, true, TypeId::DECIMAL};

// One row of a hand-fed join side: slots k, s, v, f, w.
struct JoinRow {
    int64_t k{0};
    std::string s;
    int64_t v{0};
    double f{0.0};
    __int128 w{0};
};

using JoinRowMaker = std::function<JoinRow(uint32_t)>;

// Slots: k INTEGER, s VARCHAR, v INTEGER, f DOUBLE, w NUMERIC(30,3).
const std::vector<VType>& SideTypes() {
    static const std::vector<VType> types = {kInt, kStr, kInt, kDouble, kWide3};
    return types;
}

// One side's rows: how many, made how, and whether s is dictionary-coded.
struct SideData {
    uint32_t rows{0};
    JoinRowMaker make;
    bool dictionary{false};
    size_t morsels{kSpillMorsels};
};

// Hands out a side's rows in morsels, s plain or dictionary-coded.
// Batches of 2,048, 1,000, 7 and 333 rows in turn.
// A morsel's strings are wiped when its worker starts another or finishes,
// so a join that kept a pointer instead of a copy reads '#'.
class SideInput : public Source {
public:
    struct Local : LocalState {
        std::vector<JoinRow> rows;
        std::vector<std::string> distinct;
        std::vector<StringRef> entries;
        std::vector<uint32_t> codes;
        size_t next{0};
        size_t batch{0};

        ~Local() override { Wipe(); }

        void Wipe() {
            for (JoinRow& r : rows) std::fill(r.s.begin(), r.s.end(), '#');
            for (std::string& s : distinct) std::fill(s.begin(), s.end(), '#');
        }
    };

    explicit SideInput(SideData data) : data_(std::move(data)) {}

    void Prepare(QueryContext&) override {}
    size_t MorselCount() const override { return data_.morsels; }
    std::unique_ptr<LocalState> MakeLocal() const override { return std::make_unique<Local>(); }

    void StartMorsel(size_t morsel, LocalState& local) const override {
        auto& l = static_cast<Local&>(local);
        l.Wipe();
        l.rows.clear();
        l.distinct.clear();
        l.entries.clear();
        l.codes.clear();
        const auto begin = static_cast<uint32_t>(uint64_t{data_.rows} * morsel / data_.morsels);
        const auto end = static_cast<uint32_t>(uint64_t{data_.rows} * (morsel + 1) / data_.morsels);
        for (uint32_t i = begin; i < end; ++i) l.rows.push_back(data_.make(i));
        if (data_.dictionary) {
            std::unordered_map<std::string, uint32_t> code_of;
            for (const JoinRow& r : l.rows) {
                const auto found = code_of.find(r.s);
                if (found != code_of.end()) {
                    l.codes.push_back(found->second);
                    continue;
                }
                const auto code = static_cast<uint32_t>(l.distinct.size());
                code_of.emplace(r.s, code);
                l.distinct.push_back(r.s);
                l.codes.push_back(code);
            }
            for (const std::string& s : l.distinct) l.entries.push_back(Ref(s));
        }
        l.next = 0;
        l.batch = 0;
    }

    bool NextBatch(LocalState& local, Batch& out) const override {
        auto& l = static_cast<Local&>(local);
        if (l.next >= l.rows.size()) return false;
        static constexpr uint32_t kSizes[] = {kBatchSize, 1000, 7, 333};
        const auto n = static_cast<uint32_t>(std::min<size_t>(kSizes[l.batch++ % 4], l.rows.size() - l.next));
        ShapeBatch(out, SideTypes());
        out.Reset(n);
        for (uint32_t i = 0; i < n; ++i) {
            const JoinRow& r = l.rows[l.next + i];
            out.slots[0].I64()[i] = r.k;
            out.slots[2].I64()[i] = r.v;
            out.slots[3].F64()[i] = r.f;
            out.slots[4].I128()[i] = r.w;
            if (!data_.dictionary) out.slots[1].Str()[i] = Ref(r.s);
        }
        if (data_.dictionary) {
            out.slots[1].SetDictionary(&l.entries);
            for (uint32_t i = 0; i < n; ++i) out.slots[1].Codes()[i] = l.codes[l.next + i];
        }
        l.next += n;
        return true;
    }

private:
    SideData data_;
};

// Appends raw bytes.
void AppendRaw(std::string& out, const void* data, size_t n) {
    out.append(static_cast<const char*>(data), n);
}

// Appends a cell's exact bytes: numbers raw, strings length-prefixed.
void AppendCell(const Vector& v, uint32_t row, std::string& out) {
    switch (ArrayOf(v.Type())) {
        case ValueArray::I64: AppendRaw(out, &v.I64()[row], sizeof(int64_t)); break;
        case ValueArray::I128: AppendRaw(out, &v.I128()[row], sizeof(__int128)); break;
        case ValueArray::F64: AppendRaw(out, &v.F64()[row], sizeof(double)); break;
        case ValueArray::STR: {
            const StringRef s = v.StrAt(row);
            AppendRaw(out, &s.size, sizeof(s.size));
            AppendRaw(out, s.data, s.size);
            break;
        }
    }
}

// Keeps every row as its cells' exact bytes;
// done once a watched limit is used up.
class BytesSink : public Sink {
public:
    struct Local : LocalState {
        std::vector<std::string> rows;
    };

    explicit BytesSink(const LimitCounter* limit = nullptr) : limit_(limit) {}

    std::unique_ptr<LocalState> MakeLocal() const override { return std::make_unique<Local>(); }

    void Consume(Batch& in, LocalState& local) const override {
        auto& l = static_cast<Local&>(local);
        for (uint32_t i = 0; i < in.Active(); ++i) {
            std::string row;
            for (const Vector& v : in.slots) AppendCell(v, in.sel[i], row);
            l.rows.push_back(std::move(row));
        }
    }

    void Combine(std::unique_ptr<LocalState> local) override {
        auto& l = static_cast<Local&>(*local);
        rows.insert(rows.end(), std::make_move_iterator(l.rows.begin()), std::make_move_iterator(l.rows.end()));
    }

    void Finalize(QueryContext&) override {}

    bool Done() const override { return limit_ != nullptr && limit_->Reached(); }

    std::vector<std::string> rows;

private:
    const LimitCounter* limit_;
};

// "" when both hold the same rows, byte for byte, in any order.
std::string BytesDifference(std::vector<std::string> got, std::vector<std::string> want) {
    if (got.size() != want.size()) return std::to_string(got.size()) + " rows, expected " + std::to_string(want.size());
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != want[i]) return "sorted row " + std::to_string(i) + " differs";
    }
    return "";
}

// "" when both results have the same types and tuples byte for byte, any order.
std::string ExactMultisetDifference(const QueryResult& got, const QueryResult& want) {
    if (got.schema.GetColumnCount() != want.schema.GetColumnCount()) return "column counts differ";
    for (uint32_t c = 0; c < got.schema.GetColumnCount(); ++c) {
        if (got.schema.GetColumn(c).type != want.schema.GetColumn(c).type) return "column " + std::to_string(c) + " has another type";
    }
    const auto bytes = [](const QueryResult& r) {
        std::vector<std::string> out;
        for (const Tuple& t : r.tuples) out.emplace_back(t.GetData(), t.GetLength());
        return out;
    };
    return BytesDifference(bytes(got), bytes(want));
}

// A column reference bound to plan column c, uncompiled.
std::unique_ptr<Expression> RefTo(uint32_t c) {
    auto ref = std::make_unique<ColumnRef>();
    ref->column_name = "c" + std::to_string(c);
    ref->bound_index = c;
    return ref;
}

// An integer literal.
std::unique_ptr<Expression> IntLiteral(int32_t v) {
    auto lit = std::make_unique<Literal>();
    lit->value = Value(v);
    return lit;
}

// left op right over the layout.
std::unique_ptr<CompiledExpr> Compare(BinaryOp::Op op, std::unique_ptr<Expression> left, std::unique_ptr<Expression> right, const InputLayout& in) {
    BinaryOp cmp;
    cmp.op = op;
    cmp.left = std::move(left);
    cmp.right = std::move(right);
    return CompiledExpr::Compile(cmp, in);
}

// A join of two hand-fed sides, on slot 0 by default, keeping every slot.
struct JoinCase {
    SideData build;
    SideData probe;
    bool build_left{false};
    bool residual{false};  // keeps pairs whose left v is below their right v
    std::string label;
    uint32_t build_key_slot{0};
    uint32_t probe_key_slot{0};
};

// The join's output types: the left side's slots, then the right side's.
std::vector<VType> JoinTypes() {
    std::vector<VType> types = SideTypes();
    types.insert(types.end(), SideTypes().begin(), SideTypes().end());
    return types;
}

// The case's join spec over SideInput batches.
JoinSpec SpecOf(const JoinCase& c) {
    const InputLayout side = LayoutOf(SideTypes());
    JoinSpec spec;
    spec.build_key = Col(c.build_key_slot, side);
    spec.probe_key = Col(c.probe_key_slot, side);
    spec.build_payload = {0, 1, 2, 3, 4};
    spec.probe_slots = {0, 1, 2, 3, 4};
    spec.types = JoinTypes();
    spec.build_left = c.build_left;
    if (c.residual) spec.residual = Compare(BinaryOp::Op::LT, RefTo(2), RefTo(7), LayoutOf(spec.types));
    return spec;
}

// How one spill run is set up.
struct JoinSpillConfig {
    unsigned threads{1};
    bool spill_everything{false};
    std::optional<size_t> limit;  // the budget's limit; none means no budget
    size_t held_build{0};         // bytes the test holds on every worker's floor while the builds run
    size_t held_probe{0};         // the same while the probe pipelines run
    int fail_writes{0};           // errno every temp write fails with; 0 for none
    size_t fail_from{0};          // the pipeline whose writes fail first
    bool expect_error{false};     // the run should throw even without failing writes
    size_t load_cap{0};           // most build rows a spilled load holds; 0 for no cap
};

// No budget at all.
JoinSpillConfig Unspilled(unsigned threads) {
    JoinSpillConfig c;
    c.threads = threads;
    return c;
}

// Every breaker spills everything it holds; a budget of just the floors.
JoinSpillConfig Everything(unsigned threads) {
    JoinSpillConfig c = Unspilled(threads);
    c.spill_everything = true;
    c.limit = threads * kWorkerMinimumBytes;
    return c;
}

// The builds get `grants` of shared room between
// them: every floor is held until the probe.
JoinSpillConfig Squeezed(unsigned threads, size_t grants) {
    JoinSpillConfig c = Unspilled(threads);
    c.limit = threads * kWorkerMinimumBytes + grants * kGrantBytes;
    c.held_build = kWorkerMinimumBytes;
    return c;
}

// The pipelines one run executes in order, and what owns their parts.
struct SpillPlan {
    std::vector<std::unique_ptr<Source>> sources;
    std::vector<std::unique_ptr<Operator>> operators;
    std::vector<std::unique_ptr<Sink>> sinks;
    std::vector<Pipeline> pipelines;
    size_t first_probe{1};               // pipelines before this one are builds
    std::vector<const Operator*> joins;  // probe operators, inspected after the run
    BytesSink* out{nullptr};
};

// What a run gave and spilled.
struct JoinSpillResult {
    std::vector<std::string> rows;
    std::vector<HashJoinInfo> joins;  // one per probe operator, in plan order
    uint64_t files{0};
    uint64_t written{0};
    uint64_t read{0};
    uint64_t partitions{0};
    size_t held_after_finalize{0};  // hand-fed runs: the budget's held bytes right after Finalize
    bool floors_free{false};        // hand-fed runs: every worker could take its whole floor then
    std::string error;              // what the run threw, if it did

    [[nodiscard]] const HashJoinInfo& Join() const { return joins.at(0); }
};

// Descriptors this process has open, counted from /dev/fd.
size_t OpenDescriptors() {
    std::error_code ec;
    size_t n = 0;
    for (std::filesystem::directory_iterator it("/dev/fd", ec), end; !ec && it != end; it.increment(ec)) ++n;
    return n;
}

// Makes every temp write fail with this errno while alive.
class FailWritesScope {
public:
    explicit FailWritesScope(int error_number) { TempFile::FailWritesForTest(error_number); }
    ~FailWritesScope() { TempFile::FailWritesForTest(0); }
    FailWritesScope(const FailWritesScope&) = delete;
    FailWritesScope& operator=(const FailWritesScope&) = delete;
};

// Caps the build rows a spilled load holds, while alive.
class LoadCapScope {
public:
    explicit LoadCapScope(size_t most) { SetHashJoinLoadRowsForTest(most); }
    ~LoadCapScope() { SetHashJoinLoadRowsForTest(0); }
    LoadCapScope(const LoadCapScope&) = delete;
    LoadCapScope& operator=(const LoadCapScope&) = delete;
};

// Runs a hook with each new temp file's path while alive.
class OpenHookScope {
public:
    explicit OpenHookScope(std::function<void(const std::string&)> hook) { TempFile::SetOpenHookForTest(std::move(hook)); }
    ~OpenHookScope() { TempFile::SetOpenHookForTest(nullptr); }
    OpenHookScope(const OpenHookScope&) = delete;
    OpenHookScope& operator=(const OpenHookScope&) = delete;
};

// Opens a second descriptor on every new temp file while alive; closes them when done.
class SecondDescriptors {
public:
    SecondDescriptors() {
        TempFile::SetOpenHookForTest([this](const std::string& path) {
            const int fd = ::open(path.c_str(), O_RDWR);
            const std::lock_guard<std::mutex> lock(mutex_);
            fds_.push_back(fd);
        });
    }
    ~SecondDescriptors() { Close(); }
    SecondDescriptors(const SecondDescriptors&) = delete;
    SecondDescriptors& operator=(const SecondDescriptors&) = delete;

    // Flips every bit of the byte at offset in
    // each file open so far; how many it flipped.
    size_t FlipByte(uint64_t offset) {
        const std::lock_guard<std::mutex> lock(mutex_);
        size_t flipped = 0;
        for (const int fd : fds_) {
            unsigned char b = 0;
            if (fd < 0 || ::pread(fd, &b, 1, static_cast<off_t>(offset)) != 1) continue;
            b = static_cast<unsigned char>(~b);
            if (::pwrite(fd, &b, 1, static_cast<off_t>(offset)) == 1) ++flipped;
        }
        return flipped;
    }

    // Stops watching and closes every descriptor.
    void Close() {
        TempFile::SetOpenHookForTest(nullptr);
        const std::lock_guard<std::mutex> lock(mutex_);
        for (const int fd : fds_) {
            if (fd >= 0) ::close(fd);
        }
        fds_.clear();
    }

private:
    std::mutex mutex_;
    std::vector<int> fds_;
};

// The spill partition of a lone INTEGER key at one level.
size_t KeyPartition(int64_t k, int level = 0) {
    return SpillPartitionOf(detail::FoldKey(0, static_cast<uint64_t>(k)), level);
}

// The first `per` keys from 0 up that land in each level-0 partition.
std::vector<std::vector<int64_t>> KeysByPartition(size_t per) {
    std::vector<std::vector<int64_t>> keys(kSpillPartitions);
    size_t full = 0;
    for (int64_t k = 0; full < kSpillPartitions; ++k) {
        std::vector<int64_t>& mine = keys[KeyPartition(k)];
        if (mine.size() == per) continue;
        mine.push_back(k);
        if (mine.size() == per) ++full;
    }
    return keys;
}

// The first n keys from 0 up that land in level-0 partition q.
std::vector<int64_t> KeysIn(size_t q, size_t n) {
    std::vector<int64_t> keys;
    for (int64_t k = 0; keys.size() < n; ++k) {
        if (KeyPartition(k) == q) keys.push_back(k);
    }
    return keys;
}

// A plain side row: strings of `len` bytes that change
// with i, v = i, f and w from i.
JoinRow PlainRow(int64_t k, uint32_t i, size_t len = 12) {
    std::string s = "s" + std::to_string(i) + "-";
    s.resize(len, static_cast<char>('a' + i % 26));
    return JoinRow{k, s, static_cast<int64_t>(i), i * 0.25 - 7.5, static_cast<__int128>(i) * 1000003 - 4000000};
}

class ParallelHashJoinSpillTest : public ParallelHashJoinTest {
protected:
    std::string temp_dir_;
    std::unique_ptr<WorkerPool> pool_;

    void SetUp() override {
        ParallelHashJoinTest::SetUp();
        temp_dir_ = TempPath("_spill_dir");
        std::filesystem::remove_all(temp_dir_);
        db_->SetTempDirectory(temp_dir_);
        pool_ = std::make_unique<WorkerPool>(kSpillThreads);
    }

    void TearDown() override {
        pool_.reset();
        ParallelHashJoinTest::TearDown();
        std::filesystem::remove_all(temp_dir_);
    }

    // The temp directory has no entries, or doesn't exist yet.
    bool TempDirEmpty() const {
        std::error_code ec;
        return !std::filesystem::exists(temp_dir_, ec) || std::filesystem::is_empty(temp_dir_, ec);
    }

    // Runs a plan's pipelines in order under the config; between(i, budget) runs before
    // pipeline i > 0. Afterwards no grant, temp file or descriptor is left.
    JoinSpillResult RunSpill(const std::string& label, const JoinSpillConfig& cfg, const std::function<SpillPlan()>& make,
                             const std::function<void(size_t, MemoryBudget*)>& between = {}) {
        const std::string where = label + ", " + std::to_string(cfg.threads) + " threads" + (cfg.spill_everything ? ", everything" : "") +
                                  (cfg.limit ? ", limit " + std::to_string(*cfg.limit) : "") + ", held " + std::to_string(cfg.held_build) + " / " +
                                  std::to_string(cfg.held_probe);
        const size_t descriptors = OpenDescriptors();
        EXPECT_GT(descriptors, 0u) << "can't list /dev/fd";
        // The budget comes first, so everything counted in it goes before it.
        std::optional<MemoryBudget> budget;
        if (cfg.limit) budget.emplace(*cfg.limit, cfg.threads);
        MemoryBudget* b = budget ? &*budget : nullptr;
        std::vector<MemoryReservation> held;
        const auto hold = [&](size_t bytes) {
            held.clear();
            for (unsigned w = 0; w < cfg.threads && bytes > 0; ++w) {
                held.emplace_back(b, w);
                EXPECT_TRUE(held.back().Grow(bytes)) << where;
            }
        };
        SpillCounters counters;
        JoinSpillResult result;
        {
            const LoadCapScope cap(cfg.load_cap);
            SpillPlan plan = make();
            QueryContext ctx{nullptr, nullptr, cfg.threads, cfg.threads > 1 ? pool_.get() : nullptr};
            ctx.budget = b;
            ctx.spill = &counters;
            ctx.temp_dir = &temp_dir_;
            ctx.spill_everything = cfg.spill_everything;
            try {
                std::optional<FailWritesScope> fail;
                hold(cfg.held_build);
                for (size_t i = 0; i < plan.pipelines.size(); ++i) {
                    if (i == plan.first_probe) hold(cfg.held_probe);
                    if (i > 0 && between) between(i, b);
                    if (cfg.fail_writes != 0 && i == cfg.fail_from) fail.emplace(cfg.fail_writes);
                    RunPipeline(plan.pipelines[i], ctx);
                }
            } catch (const std::exception& e) {
                result.error = e.what();
            }
            for (const Operator* op : plan.joins) result.joins.push_back(InspectHashJoin(*op).value_or(HashJoinInfo{}));
            if (plan.out != nullptr) result.rows = std::move(plan.out->rows);
        }
        held.clear();
        EXPECT_EQ(result.error.empty(), cfg.fail_writes == 0 && !cfg.expect_error) << where << ": " << result.error;
        if (!cfg.limit && !cfg.spill_everything) EXPECT_EQ(counters.files.load(), 0u) << where;
        if (b != nullptr) EXPECT_EQ(b->HeldBytes(), 0u) << where;
        EXPECT_TRUE(TempDirEmpty()) << where;
        EXPECT_EQ(OpenDescriptors(), descriptors) << where;
        result.files = counters.files.load();
        result.written = counters.bytes_written.load();
        result.read = counters.bytes_read.load();
        result.partitions = counters.partitions.load();
        return result;
    }

    // The build pipeline, then the probe pipeline through `above` into a bytes sink.
    static SpillPlan OneJoin(const JoinCase& c, std::vector<std::unique_ptr<Operator>> above = {}, const LimitCounter* limit = nullptr) {
        SpillPlan plan;
        auto made = MakeHashJoin(SpecOf(c));
        plan.sources.push_back(std::make_unique<SideInput>(c.build));
        plan.sources.push_back(std::make_unique<SideInput>(c.probe));
        plan.joins.push_back(made.second.get());
        auto out = std::make_unique<BytesSink>(limit);
        plan.out = out.get();
        plan.pipelines.push_back(Pipeline{plan.sources[0].get(), {}, made.first.get(), {}});
        Pipeline probe{plan.sources[1].get(), {made.second.get()}, out.get(), {0}};
        for (auto& op : above) probe.operators.push_back(op.get());
        plan.pipelines.push_back(std::move(probe));
        plan.sinks.push_back(std::move(made.first));
        plan.sinks.push_back(std::move(out));
        plan.operators.push_back(std::move(made.second));
        for (auto& op : above) plan.operators.push_back(std::move(op));
        return plan;
    }

    // One join's run with no extra steps.
    JoinSpillResult RunJoin(const JoinCase& c, const JoinSpillConfig& cfg, const std::function<void(size_t, MemoryBudget*)>& between = {}) {
        return RunSpill(c.label, cfg, [&c] { return OneJoin(c); }, between);
    }

    // A query with these threads, spilling everything or not.
    QueryResult RunWith(const std::string& sql, unsigned threads, bool spill_everything, bool analyze = false) {
        QueryOptions options;
        options.mode = ExecutionMode::PARALLEL;
        options.threads = threads;
        options.spill_everything = spill_everything;
        options.analyze = analyze;
        return db_->ExecuteSQL(sql, options);
    }

    // A join on k keeping only each side's v, so its rows hold no strings.
    // The output is the probe's v, then the build's.
    static JoinSpec NarrowSpec() {
        const InputLayout side = LayoutOf(SideTypes());
        JoinSpec spec;
        spec.build_key = Col(0, side);
        spec.probe_key = Col(0, side);
        spec.build_payload = {2};
        spec.probe_slots = {2};
        spec.types = {kInt, kInt};
        return spec;
    }

    // True when a fresh reservation on every worker can take its whole floor.
    // The shared pool is filled first, so a probe can't borrow shared room
    // to cover a grant left on a floor.
    static bool FloorsFree(MemoryBudget& budget, unsigned workers) {
        MemoryReservation filler(&budget, 0);
        if (!filler.Pool({})) return false;  // shared grants only from now on
        while (filler.Grow(kGrantBytes)) {}
        for (unsigned w = 0; w < workers; ++w) {
            MemoryReservation probe(&budget, w);
            if (!probe.Grow(kWorkerMinimumBytes)) return false;
        }
        return true;
    }

    // Feeds one morsel of a side to a build worker's local.
    static void Feed(const Sink& sink, LocalState& local, const SideInput& in, size_t morsel) {
        auto source = in.MakeLocal();
        in.StartMorsel(morsel, *source);
        Batch b;
        while (in.NextBatch(*source, b)) sink.Consume(b, local);
    }

    // Probes a finalized join through a pipeline, so spilled partitions join in its spill phase.
    static std::vector<std::string> Probe(Operator& join, const SideData& probe, QueryContext& ctx) {
        SideInput in(probe);
        BytesSink out;
        Pipeline p{&in, {&join}, &out, {}};
        RunPipeline(p, ctx);
        return std::move(out.rows);
    }

    // A narrow join fed by hand, morsel m to worker m % workers, then probed.
    // Finalize uses pool when set. Afterwards no grant or temp file is left.
    JoinSpillResult HandFedJoin(const SideData& build, const SideData& probe, std::optional<size_t> limit, unsigned workers, WorkerPool* pool) {
        std::optional<MemoryBudget> budget;
        if (limit) budget.emplace(*limit, workers);
        SpillCounters counters;
        JoinSpillResult result;
        {
            auto made = MakeHashJoin(NarrowSpec());
            QueryContext ctx{nullptr, nullptr, workers, pool};
            ctx.budget = budget ? &*budget : nullptr;
            ctx.spill = &counters;
            ctx.temp_dir = &temp_dir_;
            const SideInput in(build);
            for (unsigned w = 0; w < workers; ++w) {
                auto local = made.first->MakeWorkerLocal(ctx, w);
                for (size_t m = w; m < build.morsels; m += workers) Feed(*made.first, *local, in, m);
                made.first->Combine(std::move(local));
            }
            made.first->Finalize(ctx);
            if (budget) {
                result.held_after_finalize = budget->HeldBytes();
                result.floors_free = FloorsFree(*budget, workers);
            }
            result.rows = Probe(*made.second, probe, ctx);
            result.joins.push_back(*InspectHashJoin(*made.second));
        }
        if (budget) EXPECT_EQ(budget->HeldBytes(), 0u);
        EXPECT_TRUE(TempDirEmpty());
        result.files = counters.files.load();
        result.written = counters.bytes_written.load();
        result.partitions = counters.partitions.load();
        return result;
    }
};

TEST_F(ParallelHashJoinSpillTest, BuildSpillsLargestPartitionsFirst) {
    // Partition q holds (q + 1) * 40 build rows over 10 keys. Each partition's
    // rows arrive at its share's pace, so at any moment the higher partition holds more.
    const auto keys = KeysByPartition(10);
    struct Placed {
        double at;
        int64_t k;
    };
    std::vector<Placed> placed;
    for (size_t q = 0; q < kSpillPartitions; ++q) {
        const size_t n = (q + 1) * 40;
        for (size_t j = 0; j < n; ++j) placed.push_back(Placed{(static_cast<double>(j) + 0.5) / static_cast<double>(n), keys[q][j % 10]});
    }
    std::stable_sort(placed.begin(), placed.end(), [](const Placed& a, const Placed& b) { return a.at < b.at; });
    std::vector<int64_t> build_keys;
    for (const Placed& p : placed) build_keys.push_back(p.k);
    std::vector<int64_t> probe_keys;
    for (const auto& mine : keys) probe_keys.insert(probe_keys.end(), mine.begin(), mine.end());

    JoinCase c;
    c.label = "largest first";
    c.build = SideData{static_cast<uint32_t>(build_keys.size()), [&build_keys](uint32_t i) { return PlainRow(build_keys[i], i, 24); }};
    c.probe = SideData{static_cast<uint32_t>(probe_keys.size()), [&probe_keys](uint32_t i) { return PlainRow(probe_keys[i], i); }};
    const JoinSpillResult want = RunJoin(c, Unspilled(1));
    ASSERT_EQ(want.rows.size(), build_keys.size());

    for (const unsigned threads : {1u, kSpillThreads}) {
        // The builds get one shared grant between them.
        const JoinSpillResult got = RunJoin(c, Squeezed(threads, 1));
        EXPECT_EQ(BytesDifference(got.rows, want.rows), "") << threads << " threads";
        EXPECT_GT(got.files, 0u);
        EXPECT_GT(got.written, 0u);
        const HashJoinInfo& info = got.Join();
        EXPECT_GT(info.spilled_partitions, 0u);
        EXPECT_EQ(got.partitions, info.spilled_partitions);
        EXPECT_EQ(static_cast<size_t>(__builtin_popcountll(info.spilled_mask)), info.spilled_partitions);
        EXPECT_EQ(info.build_rows, build_keys.size());
        // Several workers sharing one grant may each spill all they hold; one worker keeps some.
        if (threads > 1) continue;
        EXPECT_LT(info.spilled_partitions, kSpillPartitions);
        // One worker marks the largest partition each time, so the spilled ones are the highest.
        const size_t kept = kSpillPartitions - info.spilled_partitions;
        EXPECT_EQ(info.spilled_mask, ~uint64_t{0} << kept) << std::hex << info.spilled_mask;
    }
}

TEST_F(ParallelHashJoinSpillTest, ProbeRowsOfSpilledPartitionsReenter) {
    // 600 keys, 12 build rows of about 1,100 bytes
    // and 3 probe rows each, in every partition.
    JoinCase c;
    c.label = "reenter";
    c.build = SideData{7200, [](uint32_t i) { return PlainRow(i % 600, i, 1000); }};
    c.probe = SideData{1800, [](uint32_t i) { return PlainRow(i % 600, i, 30); }, true};

    // A filter on the build side's v, then COUNT,
    // SUM(v), MIN(s), SUM(w) of it by the probe key.
    const auto make = [&c] {
        SpillPlan plan;
        auto join = MakeHashJoin(SpecOf(c));
        const InputLayout out = LayoutOf(JoinTypes());
        auto filter = std::make_unique<FilterOperator>(Compare(BinaryOp::Op::LT, RefTo(7), IntLiteral(5000), out));
        AggregateSpec agg;
        agg.keys.push_back(Col(0, out));
        agg.out_types.push_back(kInt);
        for (const auto& [func, col] : std::vector<std::pair<Aggregate::Func, int>>{
                 {Aggregate::Func::COUNT, -1}, {Aggregate::Func::SUM, 7}, {Aggregate::Func::MIN, 6}, {Aggregate::Func::SUM, 9}}) {
            agg.funcs.push_back(func);
            agg.args.push_back(col < 0 ? nullptr : Col(static_cast<uint32_t>(col), out));
            agg.out_types.push_back(AggregateOutputType(func, agg.args.back().get()));
        }
        auto grouped = MakeHashAggregate(std::move(agg));
        auto rows = std::make_unique<BytesSink>();
        plan.out = rows.get();
        plan.sources.push_back(std::make_unique<SideInput>(c.build));
        plan.sources.push_back(std::make_unique<SideInput>(c.probe));
        plan.joins.push_back(join.second.get());
        plan.pipelines.push_back(Pipeline{plan.sources[0].get(), {}, join.first.get(), {}});
        plan.pipelines.push_back(Pipeline{plan.sources[1].get(), {join.second.get(), filter.get()}, grouped.first.get(), {0}});
        plan.pipelines.push_back(Pipeline{grouped.second.get(), {}, rows.get(), {1}});
        plan.sources.push_back(std::move(grouped.second));
        plan.sinks.push_back(std::move(join.first));
        plan.sinks.push_back(std::move(grouped.first));
        plan.sinks.push_back(std::move(rows));
        plan.operators.push_back(std::move(join.second));
        plan.operators.push_back(std::move(filter));
        return plan;
    };
    const JoinSpillResult want = RunSpill(c.label, Unspilled(1), make);
    ASSERT_EQ(want.rows.size(), 600u);

    for (const unsigned threads : {1u, kSpillThreads}) {
        EXPECT_EQ(BytesDifference(RunSpill(c.label, Unspilled(threads), make).rows, want.rows), "") << threads << " threads";
        for (const JoinSpillConfig& cfg : {Everything(threads), Squeezed(threads, 1)}) {
            const JoinSpillResult got = RunSpill(c.label, cfg, make);
            EXPECT_EQ(BytesDifference(got.rows, want.rows), "") << threads << " threads, everything " << cfg.spill_everything;
            EXPECT_GT(got.files, 0u);
            EXPECT_GT(got.written, 0u);
            EXPECT_GT(got.Join().spilled_partitions, 0u);
            EXPECT_GT(got.Join().partitions_started, 0u);
        }
    }
}

TEST_F(ParallelHashJoinSpillTest, TwoSpilledJoinsInOnePipeline) {
    // The first join's output meets the second build on the first build's v.
    const SideData first_build{6000, [](uint32_t i) { return JoinRow{i % 500, "a" + std::to_string(i), i % 300, i * 0.5, i}; }};
    const SideData second_build{600, [](uint32_t i) { return PlainRow(i % 300, i, 20); }, true};
    const SideData probe{2000, [](uint32_t i) { return PlainRow(i % 600, i); }};
    const auto make = [&] {
        SpillPlan plan;
        JoinCase first;
        first.build = first_build;
        auto a = MakeHashJoin(SpecOf(first));
        std::vector<VType> a_out = JoinTypes();
        JoinSpec spec;
        spec.build_key = Col(0, LayoutOf(SideTypes()));
        spec.probe_key = Col(7, LayoutOf(a_out));
        spec.build_payload = {0, 1, 2, 3, 4};
        spec.probe_slots = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        spec.types = a_out;
        spec.types.insert(spec.types.end(), SideTypes().begin(), SideTypes().end());
        auto b = MakeHashJoin(std::move(spec));
        auto rows = std::make_unique<BytesSink>();
        plan.out = rows.get();
        plan.sources.push_back(std::make_unique<SideInput>(first_build));
        plan.sources.push_back(std::make_unique<SideInput>(second_build));
        plan.sources.push_back(std::make_unique<SideInput>(probe));
        plan.joins = {a.second.get(), b.second.get()};
        plan.pipelines.push_back(Pipeline{plan.sources[0].get(), {}, a.first.get(), {}});
        plan.pipelines.push_back(Pipeline{plan.sources[1].get(), {}, b.first.get(), {}});
        plan.pipelines.push_back(Pipeline{plan.sources[2].get(), {a.second.get(), b.second.get()}, rows.get(), {0, 1}});
        plan.first_probe = 2;
        plan.sinks.push_back(std::move(a.first));
        plan.sinks.push_back(std::move(b.first));
        plan.sinks.push_back(std::move(rows));
        plan.operators.push_back(std::move(a.second));
        plan.operators.push_back(std::move(b.second));
        return plan;
    };
    const JoinSpillResult want = RunSpill("two joins", Unspilled(1), make);
    ASSERT_GT(want.rows.size(), 10000u);

    for (const unsigned threads : {1u, kSpillThreads}) {
        EXPECT_EQ(BytesDifference(RunSpill("two joins", Unspilled(threads), make).rows, want.rows), "") << threads << " threads";
        // Both builds spill everything, so every row the second join sees came out of the
        // first join's spill phase, and the second join spills those in turn.
        const JoinSpillResult got = RunSpill("two joins", Everything(threads), make);
        EXPECT_EQ(BytesDifference(got.rows, want.rows), "") << threads << " threads";
        ASSERT_EQ(got.joins.size(), 2u);
        for (const HashJoinInfo& info : got.joins) {
            EXPECT_GT(info.spilled_partitions, 0u);
            EXPECT_EQ(info.spilled_partitions, static_cast<size_t>(__builtin_popcountll(info.spilled_mask)));
            EXPECT_GT(info.partitions_started, 0u);
        }
        EXPECT_GT(got.files, 0u);
        EXPECT_GT(got.written, 0u);
        EXPECT_EQ(got.partitions, got.joins[0].spilled_partitions + got.joins[1].spilled_partitions);
    }
}

TEST_F(ParallelHashJoinSpillTest, RecursiveSplit) {
    // 2,000 keys of one level-0 partition, 4 build rows and 2 probe rows each.
    const std::vector<int64_t> keys = KeysIn(5, 2000);
    JoinCase c;
    c.label = "split";
    c.build = SideData{8000, [&keys](uint32_t i) { return PlainRow(keys[i % 2000], i); }};
    c.probe = SideData{4000, [&keys](uint32_t i) { return PlainRow(keys[(i * 7) % 2000], i); }};
    c.residual = true;
    const JoinSpillResult want = RunJoin(c, Unspilled(1));
    ASSERT_GT(want.rows.size(), 1000u);

    for (const unsigned threads : {1u, kSpillThreads}) {
        // The test cap: the partition's 8,000 rows won't load whole, so it splits once.
        JoinSpillConfig capped = Everything(threads);
        capped.load_cap = 1000;
        const JoinSpillResult got = RunJoin(c, capped);
        EXPECT_EQ(BytesDifference(got.rows, want.rows), "") << threads << " threads";
        EXPECT_EQ(got.Join().spilled_partitions, 1u);
        EXPECT_EQ(got.Join().partitions_started, 1u);
        EXPECT_EQ(got.Join().splits, 1u);
        EXPECT_EQ(got.Join().chunked_pieces, 0u);
        EXPECT_EQ(got.partitions, 1u);
        EXPECT_GT(got.files, 0u);
        EXPECT_GT(got.written, 0u);
        if (threads == 1) EXPECT_EQ(got.files, 4u) << "a build and a probe file, then one of each from the split";

        // The budget: the probe's worker has one grant, less than the partition needs,
        // but enough for each piece of a split.
        JoinSpillConfig tight = Everything(threads);
        tight.held_probe = kWorkerMinimumBytes - kGrantBytes;
        JoinCase wide = c;
        wide.build = SideData{40000, [&keys](uint32_t i) { return PlainRow(keys[i % 2000], i, 40); }};
        const JoinSpillResult wide_want = RunJoin(wide, Unspilled(1));
        const JoinSpillResult split = RunJoin(wide, tight);
        EXPECT_EQ(BytesDifference(split.rows, wide_want.rows), "") << threads << " threads";
        EXPECT_EQ(split.Join().splits, 1u);
        EXPECT_EQ(split.Join().chunked_pieces, 0u);
        EXPECT_GT(split.written, 0u);
    }
}

TEST_F(ParallelHashJoinSpillTest, OneKeyBiggerThanTheBudget) {
    // Key 7 holds 30,000 of 33,334 build rows (90%), about 9 MB, past one grant.
    // The other keys share its level-0 partition.
    std::vector<int64_t> others;
    for (const int64_t k : KeysIn(KeyPartition(7), 601)) {
        if (k != 7) others.push_back(k);
    }
    ASSERT_GE(others.size(), 600u);
    JoinCase c;
    c.label = "one big key";
    c.build = SideData{33334, [&others](uint32_t i) { return PlainRow(i < 30000 ? 7 : others[i % 500], i, 200); }};
    c.probe = SideData{2002, [&others](uint32_t i) { return PlainRow(i < 2 ? 7 : others[i % 600], i); }};
    const JoinSpillResult want = RunJoin(c, Unspilled(1));
    ASSERT_GT(want.rows.size(), 60000u);

    for (const unsigned threads : {1u, kSpillThreads}) {
        // Each probe worker has one grant, so key 7's rows load in chunks.
        JoinSpillConfig cfg = Everything(threads);
        cfg.held_probe = kWorkerMinimumBytes - kGrantBytes;
        const JoinSpillResult got = RunJoin(c, cfg);
        EXPECT_EQ(BytesDifference(got.rows, want.rows), "") << threads << " threads";
        EXPECT_EQ(got.Join().spilled_partitions, 1u);
        EXPECT_GE(got.Join().chunked_pieces, 1u);
        EXPECT_GE(got.Join().loads, got.Join().chunked_pieces + 2) << "key 7's piece loads in several chunks";
        EXPECT_GT(got.files, 0u);
        EXPECT_GT(got.written, 0u);

        // With the test cap instead of the budget: a
        // split that leaves every row in one piece.
        JoinCase alone = c;
        alone.build = SideData{3000, [](uint32_t i) { return PlainRow(7, i); }};
        alone.probe = SideData{50, [](uint32_t i) { return PlainRow(i % 2 == 0 ? 7 : 8, i); }};
        const JoinSpillResult alone_want = RunJoin(alone, Unspilled(1));
        JoinSpillConfig capped = Everything(threads);
        capped.load_cap = 300;
        const JoinSpillResult chunked = RunJoin(alone, capped);
        EXPECT_EQ(BytesDifference(chunked.rows, alone_want.rows), "") << threads << " threads";
        EXPECT_EQ(chunked.Join().splits, 1u);
        EXPECT_EQ(chunked.Join().chunked_pieces, 1u);
        EXPECT_GE(chunked.Join().loads, 3u) << "a block or two per chunk";
        EXPECT_GT(chunked.files, 0u);
        EXPECT_GT(chunked.written, 0u);
    }
}

TEST_F(ParallelHashJoinSpillTest, EmptyBuildAndAlmostNoBudget) {
    const SideData probe{5000, [](uint32_t i) { return PlainRow(i % 700, i); }};
    for (const unsigned threads : {1u, kSpillThreads}) {
        // A budget whose floors are all held, from the first row to the last.
        JoinSpillConfig none = Unspilled(threads);
        none.limit = threads * kWorkerMinimumBytes;
        none.held_build = kWorkerMinimumBytes;
        none.held_probe = kWorkerMinimumBytes;
        JoinSpillConfig none_everything = none;
        none_everything.spill_everything = true;

        // An empty build side never spills, whatever the budget.
        JoinCase empty;
        empty.label = "empty build";
        empty.build = SideData{0, [](uint32_t i) { return PlainRow(0, i); }};
        empty.probe = probe;
        for (const JoinSpillConfig& cfg : {Unspilled(threads), Everything(threads), none, none_everything}) {
            const JoinSpillResult got = RunJoin(empty, cfg);
            EXPECT_TRUE(got.rows.empty()) << threads << " threads";
            EXPECT_EQ(got.files, 0u);
            EXPECT_EQ(got.Join().spilled_partitions, 0u);
            EXPECT_EQ(got.Join().build_rows, 0u);
        }

        // A few build rows with no memory at all: every row
        // spills, and the phase loads one block at a time.
        JoinCase few = empty;
        few.label = "few rows";
        few.build = SideData{50, [](uint32_t i) { return PlainRow(i * 13 % 700, i); }};
        const JoinSpillResult want = RunJoin(few, Unspilled(1));
        ASSERT_GT(want.rows.size(), 0u);
        for (const JoinSpillConfig& cfg : {none, none_everything}) {
            const JoinSpillResult got = RunJoin(few, cfg);
            EXPECT_EQ(BytesDifference(got.rows, want.rows), "") << threads << " threads";
            EXPECT_GT(got.files, 0u);
            EXPECT_GT(got.written, 0u);
            EXPECT_GT(got.Join().spilled_partitions, 0u);
        }
    }
}

TEST_F(ParallelHashJoinSpillTest, PayloadStringsSurvive) {
    // Strings on both sides: empty, multibyte, 8,000 bytes and short, plain and coded.
    const auto text = [](uint32_t i) -> std::string {
        switch (i % 5) {
            case 0: return "";
            case 1: return "épée ünï " + std::to_string(i % 7);
            case 2: return BigText(i % 3);
            case 3: return "w" + std::to_string(i % 11);
            default: return kWords[i % 5];
        }
    };
    for (const bool dictionary : {false, true}) {
        for (const bool build_left : {false, true}) {
            JoinCase c;
            c.label = std::string("strings") + (dictionary ? ", coded" : "") + (build_left ? ", build left" : "");
            c.build_left = build_left;
            c.build = SideData{3000, [&text](uint32_t i) { return JoinRow{i % 300, text(i), i, -0.0, -static_cast<__int128>(i)}; }, dictionary};
            c.probe = SideData{3000, [&text](uint32_t i) { return JoinRow{(i * 7) % 350, text(i + 2), i, i * 1.5, i}; }, dictionary};
            const JoinSpillResult want = RunJoin(c, Unspilled(1));
            ASSERT_GT(want.rows.size(), 1000u);
            for (const unsigned threads : {1u, kSpillThreads}) {
                for (const JoinSpillConfig& cfg : {Everything(threads), Squeezed(threads, 1)}) {
                    const JoinSpillResult got = RunJoin(c, cfg);
                    EXPECT_EQ(BytesDifference(got.rows, want.rows), "") << c.label << ", " << threads << " threads";
                    EXPECT_GT(got.files, 0u);
                    EXPECT_GT(got.written, 0u);
                    EXPECT_GT(got.Join().spilled_partitions, 0u);
                }
            }
        }
    }

    // Through SQL: the l and r tables, spilled against unspilled, byte for byte.
    const std::string sql = "SELECT l_k, l_v, l_s, r_k, r_w, r_t FROM l JOIN r ON l_k = r_k";
    const QueryResult want = RunWith(sql, 1, false, true);
    for (const unsigned threads : {1u, 2u}) {
        const QueryResult got = RunWith(sql, threads, true, true);
        EXPECT_EQ(ExactMultisetDifference(got, want), "") << threads << " threads";
        const ParallelQueryStats& stats = db_->LastParallelStats();
        EXPECT_GT(stats.spill_files, 0u);
        EXPECT_GT(stats.spill_bytes_written, 0u);
        EXPECT_GT(stats.spill_partitions, 0u);
        // Analyze charges the spill phase's rows to the join, as morsels' are.
        ASSERT_EQ(got.profile.operators.size(), want.profile.operators.size());
        for (size_t i = 0; i < got.profile.operators.size(); ++i) {
            EXPECT_EQ(got.profile.operators[i].name, want.profile.operators[i].name);
            EXPECT_EQ(got.profile.operators[i].rows, want.profile.operators[i].rows) << got.profile.operators[i].name;
        }
    }
    EXPECT_TRUE(TempDirEmpty());
}

TEST_F(ParallelHashJoinSpillTest, LimitStopsTheSpillPhase) {
    // 640 keys over every partition, 10 build rows each; a LIMIT 5 above the join.
    const auto keys = KeysByPartition(10);
    std::vector<int64_t> all;
    for (const auto& mine : keys) all.insert(all.end(), mine.begin(), mine.end());
    JoinCase c;
    c.label = "limit";
    c.build = SideData{6400, [&all](uint32_t i) { return PlainRow(all[i % 640], i); }};
    c.probe = SideData{640, [&all](uint32_t i) { return PlainRow(all[i], i); }};
    const JoinSpillResult every = RunJoin(c, Unspilled(1));
    ASSERT_EQ(every.rows.size(), 6400u);

    const auto make = [&c] {
        std::vector<std::unique_ptr<Operator>> above;
        auto limit = std::make_unique<LimitOperator>(5);
        const LimitCounter* counter = &limit->Counter();
        above.push_back(std::move(limit));
        return OneJoin(c, std::move(above), counter);
    };
    for (const unsigned threads : {1u, kSpillThreads}) {
        const JoinSpillResult got = RunSpill(c.label, Everything(threads), make);
        ASSERT_EQ(got.rows.size(), 5u) << threads << " threads";
        for (const std::string& row : got.rows) EXPECT_NE(std::find(every.rows.begin(), every.rows.end(), row), every.rows.end());
        EXPECT_EQ(got.Join().spilled_partitions, kSpillPartitions);
        EXPECT_GT(got.Join().partitions_started, 0u);
        EXPECT_LT(got.Join().partitions_started, kSpillPartitions / 2) << threads << " threads";
        EXPECT_LT(got.read, got.written);
    }
}

TEST_F(ParallelHashJoinSpillTest, FailedWriteLeavesNothingBehind) {
    const std::string reason = "out of temp space in " + temp_dir_ + " while spilling: " + std::strerror(ENOSPC);
    JoinCase c;
    c.label = "failing";
    c.build = SideData{20000, [](uint32_t i) { return PlainRow(i % 700, i, 300); }};
    c.probe = SideData{3000, [](uint32_t i) { return PlainRow(i % 700, i, 300); }};
    const JoinSpillResult want = RunJoin(c, Unspilled(1));

    // One partition and one morsel per side: the
    // third temp file is the spill phase's split.
    const std::vector<int64_t> keys = KeysIn(9, 500);
    JoinCase one = c;
    one.build = SideData{20000, [&keys](uint32_t i) { return PlainRow(keys[i % 500], i, 300); }, false, 1};
    one.probe = SideData{3000, [&keys](uint32_t i) { return PlainRow(keys[i % 500], i, 300); }, false, 1};

    for (const unsigned threads : {1u, kSpillThreads}) {
        // In the build, then in the probe's morsels.
        for (const size_t from : {size_t{0}, size_t{1}}) {
            JoinSpillConfig cfg = Everything(threads);
            cfg.fail_writes = ENOSPC;
            cfg.fail_from = from;
            const JoinSpillResult failed = RunJoin(c, cfg);
            EXPECT_EQ(failed.error, reason) << threads << " threads, from " << from;
            EXPECT_TRUE(failed.rows.empty());
        }
        // In the spill phase, splitting a partition past the cap.
        {
            JoinSpillConfig cfg = Everything(threads);
            cfg.expect_error = true;
            cfg.load_cap = 2000;
            std::atomic<int> opened{0};
            const OpenHookScope hook([&opened](const std::string&) {
                if (opened.fetch_add(1) == 2) TempFile::FailWritesForTest(ENOSPC);
            });
            const JoinSpillResult failed = RunJoin(one, cfg);
            TempFile::FailWritesForTest(0);
            EXPECT_GE(opened.load(), 3) << threads << " threads";
            EXPECT_EQ(failed.error, reason) << threads << " threads, in the split";
        }
        // The next join answers.
        const JoinSpillResult next = RunJoin(c, Everything(threads));
        EXPECT_EQ(BytesDifference(next.rows, want.rows), "") << threads << " threads";
        EXPECT_GT(next.files, 0u);
    }
}

TEST_F(ParallelHashJoinSpillTest, DamagedSpillIsRefused) {
    JoinCase c;
    c.label = "damaged";
    c.build = SideData{4000, [](uint32_t i) { return PlainRow(i % 400, i); }};
    c.probe = SideData{1000, [](uint32_t i) { return PlainRow(i % 400, i); }};
    for (const unsigned threads : {1u, kSpillThreads}) {
        SecondDescriptors second;
        size_t flipped = 0;
        JoinSpillConfig cfg = Everything(threads);
        cfg.expect_error = true;
        // Between the pipelines, a byte of every build file's first block's payload flips.
        const JoinSpillResult got = RunSpill(c.label, cfg, [&c] { return OneJoin(c); }, [&](size_t, MemoryBudget*) {
            flipped = second.FlipByte(sizeof(TempBlockHeader) + 3);
            second.Close();
        });
        EXPECT_GT(flipped, 0u);
        EXPECT_EQ(got.error, "damaged temp file: checksum mismatch") << threads << " threads";
    }
}

TEST_F(ParallelHashJoinSpillTest, BuildStaysInMemoryOffTheFloor) {
    JoinCase c;
    c.label = "off the floor";
    c.build = SideData{16000, [](uint32_t i) { return PlainRow(i % 2000, i); }};
    c.probe = SideData{2000, [](uint32_t i) { return PlainRow(i, i); }};
    const JoinSpillResult want = RunJoin(c, Unspilled(1));
    for (const unsigned threads : {1u, kSpillThreads}) {
        // Each worker's rows fit one grant, and there
        // is a grant of shared room per worker.
        JoinSpillConfig cfg = Unspilled(threads);
        cfg.limit = threads * (kWorkerMinimumBytes + kGrantBytes);
        bool checked = false;
        const JoinSpillResult got = RunJoin(c, cfg, [&](size_t, MemoryBudget* b) {
            EXPECT_GT(b->HeldBytes(), 0u);
            EXPECT_GT(b->PeakBytes(), threads * kWorkerMinimumBytes) << "LeaveFloor moved the table to the shared pool";
            checked = true;
        });
        EXPECT_TRUE(checked);
        EXPECT_EQ(BytesDifference(got.rows, want.rows), "") << threads << " threads";
        EXPECT_EQ(got.files, 0u);
        EXPECT_EQ(got.Join().spilled_partitions, 0u);
    }
}

TEST_F(ParallelHashJoinSpillTest, SpilledKeysOfEveryForm) {
    // Keys: strings (some multibyte), doubles (-0.0 and +0.0 meet; NaN meets nothing),
    // wide decimals, and integers against wide decimals.
    const auto make = [](uint32_t i, uint32_t keys) {
        const uint32_t k = i % keys;
        JoinRow r;
        r.k = k;
        r.s = (k % 3 == 0 ? "ключ " : "key ") + std::to_string(k);
        r.v = i;
        r.f = k == 0 ? (i % 2 == 0 ? -0.0 : 0.0) : (k == 1 ? std::nan("") : k * 0.5);
        r.w = static_cast<__int128>(k) * 1000;
        return r;
    };
    struct Form {
        uint32_t build_slot;
        uint32_t probe_slot;
        const char* label;
    };
    for (const Form& form : {Form{1, 1, "string keys"}, Form{3, 3, "double keys"}, Form{4, 4, "wide decimal keys"},
                             Form{0, 4, "integer against wide decimal keys"}}) {
        JoinCase c;
        c.label = form.label;
        c.build_key_slot = form.build_slot;
        c.probe_key_slot = form.probe_slot;
        c.build = SideData{6000, [make](uint32_t i) { return make(i, 400); }, true};
        c.probe = SideData{1500, [make](uint32_t i) { return make(i * 7, 450); }};
        const JoinSpillResult want = RunJoin(c, Unspilled(1));
        ASSERT_GT(want.rows.size(), 1000u) << c.label;
        for (const unsigned threads : {1u, kSpillThreads}) {
            // Everything spilled, or no room at all while building.
            for (const JoinSpillConfig& cfg : {Everything(threads), Squeezed(threads, 0)}) {
                const JoinSpillResult got = RunJoin(c, cfg);
                EXPECT_EQ(BytesDifference(got.rows, want.rows), "") << c.label << ", " << threads << " threads";
                EXPECT_GT(got.files, 0u);
                EXPECT_GT(got.written, 0u);
                EXPECT_GT(got.Join().spilled_partitions, 0u);
            }
        }
    }
}

TEST_F(ParallelHashJoinSpillTest, TablesKeepWorkerAndArrivalOrder) {
    // Nothing spills: key 7's pairs come in worker order, then each worker's arrival order.
    {
        MemoryBudget budget(2 * (kWorkerMinimumBytes + kGrantBytes), 2);
        SpillCounters counters;
        {
            auto made = IntJoin(false);
            QueryContext ctx{nullptr, nullptr, 2};
            ctx.budget = &budget;
            ctx.spill = &counters;
            ctx.temp_dir = &temp_dir_;
            auto w0 = made.first->MakeWorkerLocal(ctx, 0);
            auto w1 = made.first->MakeWorkerLocal(ctx, 1);
            Batch a = PairBatch({{7, 10}, {8, 1}, {7, 11}});
            Batch b = PairBatch({{7, 20}, {7, 21}});
            Batch c = PairBatch({{7, 12}});
            made.first->Consume(a, *w0);
            made.first->Consume(b, *w1);
            made.first->Consume(c, *w0);
            made.first->Combine(std::move(w0));
            made.first->Combine(std::move(w1));
            made.first->Finalize(ctx);
            EXPECT_GT(budget.HeldBytes(), 0u);
            Batch probe = PairBatch({{7, -7}});
            auto local = made.second->MakeLocal();
            EXPECT_EQ(Drain(*made.second, probe, local.get()), (std::vector<std::pair<int64_t, int64_t>>{{-7, 10}, {-7, 11}, {-7, 12}, {-7, 20}, {-7, 21}}));
            EXPECT_EQ(InspectHashJoin(*made.second)->spilled_partitions, 0u);
        }
        EXPECT_EQ(budget.HeldBytes(), 0u);
        EXPECT_EQ(counters.files.load(), 0u);
    }

    // One worker whose build spills switches to rows by partition; key 7's partition stays
    // in memory, and its rows keep their arrival order.
    MemoryBudget budget(kWorkerMinimumBytes + 2 * kGrantBytes, 1);
    SpillCounters counters;
    {
        MemoryReservation floor(&budget, 0);
        ASSERT_TRUE(floor.Grow(kWorkerMinimumBytes));
        auto made = IntJoin(false);
        QueryContext ctx{nullptr, nullptr, 1};
        ctx.budget = &budget;
        ctx.spill = &counters;
        ctx.temp_dir = &temp_dir_;
        auto w = made.first->MakeWorkerLocal(ctx, 0);
        const size_t seven = KeyPartition(7);
        int64_t filler = 1000;
        for (int64_t batch = 0; batch < 150; ++batch) {
            std::vector<std::pair<int64_t, int64_t>> rows;
            rows.emplace_back(7, batch);
            while (rows.size() < kBatchSize) {
                if (KeyPartition(++filler) != seven) rows.emplace_back(filler, filler);
            }
            Batch b = PairBatch(rows);
            made.first->Consume(b, *w);
        }
        made.first->Combine(std::move(w));
        made.first->Finalize(ctx);
        const HashJoinInfo info = *InspectHashJoin(*made.second);
        EXPECT_GT(info.spilled_partitions, 0u);
        ASSERT_EQ(info.spilled_mask >> seven & 1, 0u) << "key 7's partition stays in memory";
        Batch probe = PairBatch({{7, -7}});
        auto local = made.second->MakeLocal();
        std::vector<std::pair<int64_t, int64_t>> want;
        for (int64_t batch = 0; batch < 150; ++batch) want.emplace_back(-7, batch);
        EXPECT_EQ(Drain(*made.second, probe, local.get()), want);
        EXPECT_GT(counters.files.load(), 0u);
    }
    EXPECT_EQ(budget.HeldBytes(), 0u);
    EXPECT_TRUE(TempDirEmpty());
}

TEST_F(ParallelHashJoinSpillTest, FinalizeSpillsWhatCantLeaveTheFloor) {
    // Each worker's rows fit its floor, so Consume never spills; with no shared room
    // they can't leave the floor at Finalize, so they go to disk there.
    JoinCase c;
    c.label = "finalize spill";
    c.build = SideData{20000, [](uint32_t i) { return PlainRow(i % 2000, i); }};
    c.probe = SideData{3000, [](uint32_t i) { return PlainRow(i % 2500, i); }};
    const JoinSpillResult want = RunJoin(c, Unspilled(1));
    ASSERT_GT(want.rows.size(), 10000u);
    for (const unsigned threads : {1u, kSpillThreads}) {
        JoinSpillConfig cfg = Unspilled(threads);
        cfg.limit = threads * kWorkerMinimumBytes;
        const JoinSpillResult got = RunJoin(c, cfg);
        EXPECT_EQ(BytesDifference(got.rows, want.rows), "") << threads << " threads";
        EXPECT_GT(got.files, 0u);
        EXPECT_GT(got.written, 0u);
        EXPECT_GT(got.Join().spilled_partitions, 0u);
    }
}

TEST_F(ParallelHashJoinSpillTest, FinishedTableHoldsNoSpareGrant) {
    // 95,000 rows of 48 bytes need two grants; without their 8-byte hashes, one.
    MemoryBudget budget(kWorkerMinimumBytes + 2 * kGrantBytes, 1);
    {
        auto made = IntJoin(false);
        QueryContext ctx{nullptr, nullptr, 1};
        ctx.budget = &budget;
        ctx.temp_dir = &temp_dir_;
        auto w = made.first->MakeWorkerLocal(ctx, 0);
        for (int64_t first = 0; first < 95000; first += kBatchSize) {
            std::vector<std::pair<int64_t, int64_t>> rows;
            for (int64_t i = first; i < std::min<int64_t>(first + kBatchSize, 95000); ++i) rows.emplace_back(i, i);
            Batch b = PairBatch(rows);
            made.first->Consume(b, *w);
        }
        made.first->Combine(std::move(w));
        made.first->Finalize(ctx);
        EXPECT_EQ(InspectHashJoin(*made.second)->build_rows, 95000u);
        EXPECT_EQ(InspectHashJoin(*made.second)->spilled_partitions, 0u);
        EXPECT_EQ(budget.HeldBytes(), kGrantBytes) << "the table keeps the one grant it uses, no spare";
    }
    EXPECT_EQ(budget.HeldBytes(), 0u);
}

TEST_F(ParallelHashJoinSpillTest, FinalizeSpillsOnlyWhatTheFloorsNeed) {
    // Four workers each keep 160,000 rows of 48 bytes, two grants, on their floors.
    // Shared room holds three workers' grants: pooled, the rows need eight, so they
    // can't pool at first. A quarter of the rows go, then the rest pool.
    constexpr unsigned kWorkers = 4;
    constexpr uint32_t kPerWorker = 160000;
    const SideData build{kPerWorker * kWorkers, [](uint32_t i) { return PlainRow(i % 150000, i); }, false, kWorkers};
    const SideData probe{30000, [](uint32_t i) { return PlainRow(i * 7 % 200000, i); }};
    const JoinSpillResult want = HandFedJoin(build, probe, std::nullopt, kWorkers, nullptr);
    ASSERT_GT(want.rows.size(), 10000u);

    for (WorkerPool* pool : {static_cast<WorkerPool*>(nullptr), pool_.get()}) {
        const JoinSpillResult got = HandFedJoin(build, probe, kWorkers * kWorkerMinimumBytes + 6 * kGrantBytes, kWorkers, pool);
        const char* where = pool != nullptr ? "on the pool" : "serial";
        EXPECT_EQ(BytesDifference(got.rows, want.rows), "") << where;
        EXPECT_GT(got.files, 0u) << where;
        EXPECT_GT(got.written, 0u) << where;
        EXPECT_GT(got.Join().spilled_partitions, 0u) << where;
        EXPECT_LT(got.Join().spilled_partitions, kSpillPartitions / 2) << where << ": only enough partitions to free the floors";
        EXPECT_EQ(got.partitions, got.Join().spilled_partitions) << where;
        EXPECT_TRUE(got.floors_free) << where << ": every floor is free";
    }
}

TEST_F(ParallelHashJoinSpillTest, FinalizePoolsSmallPartsInsteadOfSpilling) {
    // Each worker keeps 1,500 rows of 48 bytes, under one grant, on its floor.
    // With one grant of shared room, one part could move alone; pooled, all fit.
    for (const unsigned workers : {4u, 18u}) {
        const SideData build{workers * 1500, [](uint32_t i) { return PlainRow(i % 5000, i); }, false, workers};
        const SideData probe{6000, [](uint32_t i) { return PlainRow(i * 7 % 5500, i); }};
        const JoinSpillResult want = HandFedJoin(build, probe, std::nullopt, workers, nullptr);
        ASSERT_GT(want.rows.size(), 1000u);
        const size_t floors = workers * kWorkerMinimumBytes;
        for (WorkerPool* pool : {static_cast<WorkerPool*>(nullptr), pool_.get()}) {
            const std::string where = std::to_string(workers) + " workers" + (pool != nullptr ? ", on the pool" : ", serial");
            const JoinSpillResult got = HandFedJoin(build, probe, floors + kGrantBytes, workers, pool);
            EXPECT_EQ(BytesDifference(got.rows, want.rows), "") << where;
            EXPECT_EQ(got.files, 0u) << where;
            EXPECT_EQ(got.written, 0u) << where;
            EXPECT_EQ(got.partitions, 0u) << where;
            EXPECT_EQ(got.Join().spilled_partitions, 0u) << where;
            EXPECT_EQ(got.held_after_finalize, kGrantBytes) << where << ": the table holds one grant";
            EXPECT_TRUE(got.floors_free) << where;
        }

        // Just the floors: no shared room to pool into, so the rows spill at Finalize.
        const JoinSpillResult tight = HandFedJoin(build, probe, floors, workers, pool_.get());
        EXPECT_EQ(BytesDifference(tight.rows, want.rows), "") << workers << " workers, floors only";
        EXPECT_GT(tight.files, 0u) << workers << " workers, floors only";
        EXPECT_GT(tight.Join().spilled_partitions, 0u) << workers << " workers, floors only";
        EXPECT_EQ(tight.held_after_finalize, 0u) << workers << " workers, floors only";
    }
}

TEST_F(ParallelHashJoinSpillTest, FinishedTablePoolsSoTheNextBreakerFindsRoom) {
    // Four workers keep 1,500 rows each, one floor grant apiece. With a shared grant
    // per worker, each part could leave its floor alone; pooled, they take one grant.
    // A second build on worker 0 then needs two shared grants beside its floor,
    // so it fits only if the first table left the room.
    constexpr unsigned kWorkers = 4;
    const SideData build{kWorkers * 1500, [](uint32_t i) { return PlainRow(i % 5000, i); }, false, kWorkers};
    const SideData probe{6000, [](uint32_t i) { return PlainRow(i * 7 % 5500, i); }};
    const JoinSpillResult want = HandFedJoin(build, probe, std::nullopt, kWorkers, nullptr);
    MemoryBudget budget(kWorkers * (kWorkerMinimumBytes + kGrantBytes), kWorkers);
    SpillCounters counters;
    {
        QueryContext ctx{nullptr, nullptr, kWorkers, pool_.get()};
        ctx.budget = &budget;
        ctx.spill = &counters;
        ctx.temp_dir = &temp_dir_;
        auto first = MakeHashJoin(NarrowSpec());
        const SideInput small(build);
        for (unsigned w = 0; w < kWorkers; ++w) {
            auto local = first.first->MakeWorkerLocal(ctx, w);
            Feed(*first.first, *local, small, w);
            first.first->Combine(std::move(local));
        }
        first.first->Finalize(ctx);
        EXPECT_EQ(budget.HeldBytes(), kGrantBytes) << "the first table holds one grant, not one per part";

        // 300,000 rows of 48 bytes: two floor grants and two shared ones.
        auto second = MakeHashJoin(NarrowSpec());
        const SideInput big(SideData{300000, [](uint32_t i) { return PlainRow(i, i); }, false, 1});
        auto local = second.first->MakeWorkerLocal(ctx, 0);
        Feed(*second.first, *local, big, 0);
        EXPECT_EQ(counters.files.load(), 0u) << "the second build found its shared grants";
        EXPECT_EQ(budget.HeldBytes(), 5 * kGrantBytes);

        // The first table still answers.
        EXPECT_EQ(BytesDifference(Probe(*first.second, probe, ctx), want.rows), "");
    }
    EXPECT_EQ(budget.HeldBytes(), 0u);
    EXPECT_TRUE(TempDirEmpty());
}

TEST_F(ParallelHashJoinSpillTest, SpilledRowsKeepArrivalOrder) {
    // Worker 1 holds key 7's rows 0 and 1 when worker 0, out of room,
    // marks its partition. Worker 1's later rows 4 and 5 must reach disk after them.
    MemoryBudget budget(2 * kWorkerMinimumBytes, 2);
    SpillCounters counters;
    std::vector<std::string> rows;
    {
        MemoryReservation floor(&budget, 0);
        ASSERT_TRUE(floor.Grow(kWorkerMinimumBytes));
        auto made = MakeHashJoin(NarrowSpec());
        QueryContext ctx{nullptr, nullptr, 2};
        ctx.budget = &budget;
        ctx.spill = &counters;
        ctx.temp_dir = &temp_dir_;
        const SideInput in(SideData{6, [](uint32_t i) { return PlainRow(7, i); }, false, 3});
        auto w0 = made.first->MakeWorkerLocal(ctx, 0);
        auto w1 = made.first->MakeWorkerLocal(ctx, 1);
        Feed(*made.first, *w1, in, 0);
        Feed(*made.first, *w0, in, 1);
        Feed(*made.first, *w1, in, 2);
        made.first->Combine(std::move(w0));
        made.first->Combine(std::move(w1));
        made.first->Finalize(ctx);
        EXPECT_EQ(InspectHashJoin(*made.second)->spilled_mask, uint64_t{1} << KeyPartition(7));
        floor = MemoryReservation();  // the probe gets worker 0's floor
        ctx.threads = 1;
        rows = Probe(*made.second, SideData{1, [](uint32_t) { return PlainRow(7, 1000); }, false, 1}, ctx);
    }
    // Worker 0's file is read first, then worker 1's, each in the order written.
    std::vector<int64_t> build_v;
    for (const std::string& row : rows) {
        ASSERT_EQ(row.size(), 2 * sizeof(int64_t));
        int64_t v = 0;
        std::memcpy(&v, row.data() + sizeof(int64_t), sizeof(v));
        build_v.push_back(v);
    }
    EXPECT_EQ(build_v, (std::vector<int64_t>{2, 3, 0, 1, 4, 5}));
    EXPECT_GT(counters.files.load(), 0u);
    EXPECT_EQ(budget.HeldBytes(), 0u);
    EXPECT_TRUE(TempDirEmpty());
}

TEST(ParallelColumnStoreTest, AbsorbAllSizesOnceAndFreesEachStore) {
    // Stores of 5,000, 3 and 7,000 rows, each an integer and a string.
    const std::vector<ValueArray> arrays = {ValueArray::I64, ValueArray::STR};
    std::vector<ColumnStore> parts;
    std::vector<std::pair<int64_t, std::string>> want;
    for (const size_t n : {size_t{5000}, size_t{3}, size_t{7000}}) {
        ColumnStore& s = parts.emplace_back(arrays.size());
        for (size_t i = 0; i < n; ++i) {
            const auto v = static_cast<int64_t>(want.size());
            want.emplace_back(v, "row " + std::to_string(v));
            s.columns[0].i64.push_back(v);
            s.columns[1].str.push_back(s.arena.Copy(want.back().second));
        }
        s.rows = n;
    }

    ColumnStore to(arrays.size());
    AbsorbAll(to, arrays, {&parts[0], &parts[1], &parts[2]});
    ASSERT_EQ(to.rows, want.size());
    EXPECT_EQ(to.columns[0].i64.capacity(), want.size()) << "sized once, with no room to spare";
    EXPECT_EQ(to.columns[1].str.capacity(), want.size());
    for (const ColumnStore& s : parts) {
        EXPECT_EQ(s.rows, 0u);
        EXPECT_EQ(s.columns[0].i64.capacity(), 0u) << "each store's copy is freed";
        EXPECT_EQ(s.columns[1].str.capacity(), 0u);
    }
    for (size_t i = 0; i < want.size(); ++i) {
        ASSERT_EQ(to.columns[0].i64[i], want[i].first);
        ASSERT_EQ(View(to.columns[1].str[i]), want[i].second) << "strings stay valid";
    }

    // A lone store into an empty one moves whole: nothing is copied.
    ColumnStore lone(arrays.size());
    lone.columns[0].i64 = {4, 5, 6};
    lone.columns[1].str.resize(3);
    lone.rows = 3;
    const int64_t* values = lone.columns[0].i64.data();
    ColumnStore into(arrays.size());
    AbsorbAll(into, arrays, {&lone});
    EXPECT_EQ(into.rows, 3u);
    EXPECT_EQ(into.columns[0].i64.data(), values);
}

TEST_F(ParallelHashJoinSpillTest, SplitsSplitAgainDownToTheDeepestLevel) {
    // 2,000 keys in one level-0 partition, 4 build and 2 coded probe rows each.
    // Small load caps make pieces split again, so several splits' files are live at once.
    const std::vector<int64_t> keys = KeysIn(5, 2000);
    for (const bool build_left : {false, true}) {
        JoinCase c;
        c.label = build_left ? "nested splits, build left" : "nested splits";
        c.build_left = build_left;
        c.residual = true;
        c.build = SideData{8000, [&keys](uint32_t i) { return PlainRow(keys[i % 2000], i); }};
        c.probe = SideData{4000, [&keys](uint32_t i) { return PlainRow(keys[(i * 7) % 2000], i); }, true};
        const JoinSpillResult want = RunJoin(c, Unspilled(1));
        ASSERT_GT(want.rows.size(), 1000u);

        for (const unsigned threads : {1u, kSpillThreads}) {
            // A cap of 60: each level-1 piece (about 124 rows) splits again.
            JoinSpillConfig cfg = Everything(threads);
            cfg.load_cap = 60;
            const JoinSpillResult twice = RunJoin(c, cfg);
            EXPECT_EQ(BytesDifference(twice.rows, want.rows), "") << c.label << ", " << threads << " threads, cap 60";
            EXPECT_GT(twice.Join().splits, kSpillPartitions) << "the partition's split, then one per level-1 piece";
            EXPECT_EQ(twice.Join().deepest_level, 2);
            EXPECT_EQ(twice.Join().chunked_pieces, 0u);

            // Cap 3, below one key's rows: pieces split to the deepest level, then chunk.
            cfg.load_cap = 3;
            const JoinSpillResult deep = RunJoin(c, cfg);
            EXPECT_EQ(BytesDifference(deep.rows, want.rows), "") << c.label << ", " << threads << " threads, cap 3";
            EXPECT_EQ(deep.Join().deepest_level, kMaxSpillLevel) << "no piece splits past the deepest level";
            EXPECT_GT(deep.Join().chunked_pieces, 0u);
            EXPECT_GT(deep.Join().splits, twice.Join().splits);
            EXPECT_GT(deep.files, twice.files);
            EXPECT_GT(deep.written, 0u);
        }
    }
}

TEST_F(ParallelHashJoinSpillTest, ValuesOver64KiBSpill) {
    // Values past the writer's 64 KiB buffers go to disk as blocks of their own,
    // then through whole loads, splits and chunks.
    const std::vector<int64_t> keys = KeysIn(9, 40);
    JoinCase payloads;
    payloads.label = "big payloads";
    payloads.build = SideData{240, [&keys](uint32_t i) { return PlainRow(keys[i % 40], i, i % 10 == 0 ? 100000 + i : 20); }};
    payloads.probe = SideData{80, [&keys](uint32_t i) { return PlainRow(keys[(i * 3) % 40], i, i % 7 == 0 ? 70000 + i : 10); }};

    // Every key string is over 64 KiB, the probe's dictionary-coded.
    const auto key_text = [](uint32_t k) { return std::string(80000 + k % 5, static_cast<char>('a' + k % 26)) + std::to_string(k); };
    JoinCase keyed;
    keyed.label = "big string keys";
    keyed.build_key_slot = 1;
    keyed.probe_key_slot = 1;
    keyed.build = SideData{200, [&key_text](uint32_t i) {
                               JoinRow r = PlainRow(i % 25, i);
                               r.s = key_text(i % 25);
                               return r;
                           }};
    keyed.probe = SideData{60, [&key_text](uint32_t i) {
                               JoinRow r = PlainRow(i % 30, i);
                               r.s = key_text(i % 30);
                               return r;
                           }, true};

    for (const JoinCase* c : {&payloads, &keyed}) {
        const JoinSpillResult want = RunJoin(*c, Unspilled(1));
        ASSERT_GT(want.rows.size(), 100u) << c->label;
        for (const unsigned threads : {1u, kSpillThreads}) {
            JoinSpillConfig cfg = Everything(threads);
            const JoinSpillResult whole = RunJoin(*c, cfg);
            EXPECT_EQ(BytesDifference(whole.rows, want.rows), "") << c->label << ", " << threads << " threads, whole loads";
            EXPECT_EQ(whole.Join().splits, 0u);
            EXPECT_GT(whole.written, 0u);

            cfg.load_cap = 3;
            const JoinSpillResult capped = RunJoin(*c, cfg);
            EXPECT_EQ(BytesDifference(capped.rows, want.rows), "") << c->label << ", " << threads << " threads, capped loads";
            EXPECT_GT(capped.Join().splits, 0u);
            EXPECT_GT(capped.Join().chunked_pieces, 0u);
        }
    }
}

}  // namespace
}  // namespace shilmandb::parallel
