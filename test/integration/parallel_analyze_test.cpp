#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "catalog/catalog.hpp"
#include "engine/database.hpp"
#include "executor/executor_context.hpp"
#include "executor/executor_factory.hpp"
#include "parser/parser.hpp"
#include "planner/planner.hpp"
#include "tpch/tpch_workload.hpp"

// analyze in the parallel executor: tuple mode's names, depths and rows on the
// same plan, and per-operator thread times that add up.
namespace shilmandb {
namespace {

constexpr size_t kWideGroups = 40;
constexpr size_t kWideRows = (kWideGroups - 1) * 65536 + 1000;  // 40 row groups
constexpr size_t kSideRows = 70000;
constexpr double kMsTol = 1e-9;  // rounding when nanoseconds become milliseconds
constexpr unsigned kThreadCounts[] = {2, 4, 8};

const char* const kTags[] = {"amber", "blue", "coral", "dune", "ember", "fern", "gold"};

// The SF=0.01 data directory, or "" when it hasn't been generated.
std::string FindDataDir() {
    for (const std::string dir : {"bench/tpch_data/sf0.01/", "../bench/tpch_data/sf0.01/", "../../bench/tpch_data/sf0.01/"}) {
        if (std::filesystem::exists(dir + "lineitem.tbl")) return dir;
    }
    return {};
}

// Row r of wide40: w_id, w_grp, w_amt, w_tag.
void WriteWide(const std::string& path) {
    std::string out;
    out.reserve(kWideRows * 28);
    char buf[96];
    for (size_t r = 0; r < kWideRows; ++r) {
        const unsigned cents = static_cast<unsigned>((r * 7919) % 100000);
        std::snprintf(buf, sizeof(buf), "%zu|%zu|%u.%02u|%s|\n", r, r / 65536, cents / 100, cents % 100, kTags[r % 7]);
        out += buf;
    }
    std::ofstream(path, std::ios::binary) << out;
}

// Row r of side70: s_id = 31 r, so every row meets wide40.
void WriteSide(const std::string& path) {
    std::string out;
    char buf[96];
    for (size_t r = 0; r < kSideRows; ++r) {
        const unsigned cents = static_cast<unsigned>((r * 104729) % 50000);
        std::snprintf(buf, sizeof(buf), "%zu|%u.%02u|%s|\n", r * 31, cents / 100, cents % 100, kTags[(r * 3) % 7]);
        out += buf;
    }
    std::ofstream(path, std::ios::binary) << out;
}

// The scan's table, as the profile names it.
std::string ScanTable(const PlanNode* node) {
    if (node->type == PlanNodeType::SEQ_SCAN) return static_cast<const SeqScanPlanNode*>(node)->table_name;
    if (node->type == PlanNodeType::INDEX_SCAN) return static_cast<const IndexScanPlanNode*>(node)->table_name;
    return "";
}

// The profile's operator name: a join is named by the leaf scan on its right.
std::string NameOf(const PlanNode* node) {
    const auto joined = [&] {
        const PlanNode* leaf = node->children[1].get();
        while (!leaf->children.empty()) leaf = leaf->children[0].get();
        return ScanTable(leaf);
    };
    switch (node->type) {
        case PlanNodeType::SEQ_SCAN: return "SeqScan " + ScanTable(node);
        case PlanNodeType::INDEX_SCAN: return "IndexScan " + ScanTable(node);
        case PlanNodeType::FILTER: return "Filter";
        case PlanNodeType::HASH_JOIN: return "HashJoin +" + joined();
        case PlanNodeType::NESTED_LOOP_JOIN: return "NestedLoopJoin +" + joined();
        case PlanNodeType::SORT: return "Sort";
        case PlanNodeType::AGGREGATE: return "Aggregate";
        case PlanNodeType::PROJECTION: return "Projection";
        case PlanNodeType::LIMIT: return "Limit";
    }
    return "Unknown";
}

// Tuple mode's analyze, without its times.
struct Reference {
    std::vector<OperatorProfile> operators;
    std::vector<uint64_t> step_rows;
    std::vector<uint64_t> scan_rows;
    size_t result_rows{0};
};

// Root first, then each child's subtree, as the profile lists operators.
void Collect(const PlanNode* node, int depth, const OperatorStatsMap& stats, std::vector<OperatorProfile>& out) {
    const auto it = stats.find(node);
    OperatorProfile p;
    p.name = NameOf(node);
    p.depth = depth;
    p.rows = it != stats.end() ? it->second.rows : 0;
    out.push_back(p);
    for (const auto& child : node->children) Collect(child.get(), depth + 1, stats, out);
}

// Tuple mode's executors with analyze on, run over the plan the parallel mode makes.
// The parallel planner may order joins or pick scans differently from tuple mode's.
Reference TupleAnalyzeOnParallelPlan(const Database& db, const std::string& sql) {
    QueryOptions options;
    options.mode = ExecutionMode::PARALLEL;
    Planner planner = db.MakePlanner(options);
    Parser parser(sql);
    auto stmt = parser.Parse();
    const auto plan = planner.Plan(std::move(*stmt));

    OperatorStatsMap stats;
    ExecutorContext ctx{db.GetBufferPoolManager(), db.GetCatalog(), &stats};
    auto executor = ExecutorFactory::CreateExecutor(plan.get(), &ctx, ExecutionMode::TUPLE);
    Reference ref;
    executor->Init();
    Tuple tuple;
    while (executor->Next(&tuple)) ++ref.result_rows;
    executor->Close();

    Collect(plan.get(), 0, stats, ref.operators);
    const auto rows_of = [&](const PlanNode* node) {
        const auto it = stats.find(node);
        return it != stats.end() ? it->second.rows : 0;
    };
    for (const PlanNode* step : planner.GetStepNodes()) ref.step_rows.push_back(rows_of(step));
    for (int slot : planner.GetJoinOrderInfo().order) ref.scan_rows.push_back(rows_of(planner.GetScanNodes()[slot]));
    return ref;
}

// One line per operator, indented by depth, for failure messages.
std::string Describe(const std::vector<OperatorProfile>& ops) {
    std::ostringstream out;
    for (const auto& op : ops) out << std::string(2 * static_cast<size_t>(op.depth), ' ') << op.name << " rows=" << op.rows << "\n";
    return out.str();
}

// The same operators in the same places with the same rows.
void ExpectSameShapeAndRows(const std::vector<OperatorProfile>& got, const std::vector<OperatorProfile>& want) {
    ASSERT_EQ(got.size(), want.size()) << "got:\n" << Describe(got) << "want:\n" << Describe(want);
    for (size_t i = 0; i < want.size(); ++i) {
        EXPECT_EQ(got[i].name, want[i].name) << "operator " << i;
        EXPECT_EQ(got[i].depth, want[i].depth) << "operator " << i << " " << want[i].name;
        EXPECT_EQ(got[i].rows, want[i].rows) << "operator " << i << " " << want[i].name << "\ngot:\n" << Describe(got) << "want:\n" << Describe(want);
    }
}

// Self times are never negative, and each total covers its children's totals.
void ExpectTimesAddUp(const std::vector<OperatorProfile>& ops) {
    ASSERT_FALSE(ops.empty());
    EXPECT_GT(ops[0].total_ms, 0.0) << "the root measured nothing";
    for (size_t i = 0; i < ops.size(); ++i) {
        EXPECT_GE(ops[i].self_ms, 0.0) << ops[i].name;
        EXPECT_GE(ops[i].total_ms, 0.0) << ops[i].name;
        double children = 0.0;
        for (size_t j = i + 1; j < ops.size() && ops[j].depth > ops[i].depth; ++j) {
            if (ops[j].depth == ops[i].depth + 1) children += ops[j].total_ms;
        }
        EXPECT_GE(ops[i].total_ms + kMsTol, children) << "operator " << i << " " << ops[i].name;
        EXPECT_NEAR(ops[i].self_ms, ops[i].total_ms - children, 1e-6) << "operator " << i << " " << ops[i].name;
    }
}

// TPC-H SF=0.01 columnar with the 8 indexes, and wide40 / side70, loaded once.
class ParallelAnalyzeTest : public ::testing::Test {
protected:
    static std::unique_ptr<Database> tpch_db_;
    static std::unique_ptr<Database> wide_db_;
    static std::string tpch_file_;
    static std::string wide_file_;
    static std::string data_dir_;

    static void SetUpTestSuite() {
        const auto temp = std::filesystem::temp_directory_path();
        const std::string pid = std::to_string(::getpid());
        tpch_file_ = (temp / ("shilmandb_parallel_analyze_test_tpch_" + pid + ".db")).string();
        wide_file_ = (temp / ("shilmandb_parallel_analyze_test_wide_" + pid + ".db")).string();
        data_dir_ = FindDataDir();
    }

    static void TearDownTestSuite() {
        tpch_db_.reset();
        wide_db_.reset();
        std::filesystem::remove(tpch_file_);
        std::filesystem::remove(wide_file_);
    }

    // Skips without the data; the skip ends only this helper, so callers check IsSkipped().
    static void NeedTpch() {
        if (data_dir_.empty()) {
            GTEST_SKIP() << "TPC-H SF=0.01 data not found. Run bench/tpch/generate_tpch_data.sh first.";
        }
        if (!tpch_db_) {
            std::filesystem::remove(tpch_file_);
            tpch_db_ = std::make_unique<Database>(tpch_file_, 4096);
            LoadOptions options;
            options.storage = StorageKind::COLUMNAR;
            for (const auto& td : tpch::TableDescs()) tpch_db_->LoadTable(td.name, td.schema, data_dir_ + td.tbl_file, options);
            auto* catalog = tpch_db_->GetCatalog();
            for (const auto& idx : tpch::IndexDescs()) (void)catalog->CreateIndex(idx.index_name, idx.table_name, idx.column_name);
            for (const auto& td : tpch::TableDescs()) catalog->UpdateTableStats(td.name);
        }
    }

    // Generated, so it never skips.
    static Database& NeedWide() {
        if (!wide_db_) {
            std::filesystem::remove(wide_file_);
            wide_db_ = std::make_unique<Database>(wide_file_, 1024);
            LoadOptions options;
            options.storage = StorageKind::COLUMNAR;
            const std::string wide_path = wide_file_ + "_wide40.tbl";
            const std::string side_path = wide_file_ + "_side70.tbl";
            WriteWide(wide_path);
            WriteSide(side_path);
            wide_db_->LoadTable("wide40",
                                Schema({Column("w_id", ColumnType::BigInt()), Column("w_grp", ColumnType::Integer()),
                                        Column("w_amt", ColumnType::Numeric(12, 2)), Column("w_tag", ColumnType::Varchar(8))}),
                                wide_path, options);
            wide_db_->LoadTable("side70",
                                Schema({Column("s_id", ColumnType::BigInt()), Column("s_val", ColumnType::Numeric(10, 2)), Column("s_tag", ColumnType::Varchar(8))}),
                                side_path, options);
            std::filesystem::remove(wide_path);
            std::filesystem::remove(side_path);
            wide_db_->GetCatalog()->UpdateTableStats("wide40");
            wide_db_->GetCatalog()->UpdateTableStats("side70");
        }
        return *wide_db_;
    }

    // The parallel executor with analyze on.
    static QueryResult Analyze(Database& db, const std::string& sql, unsigned threads) {
        QueryOptions options;
        options.mode = ExecutionMode::PARALLEL;
        options.threads = threads;
        options.analyze = true;
        return db.ExecuteSQL(sql, options);
    }
};

std::unique_ptr<Database> ParallelAnalyzeTest::tpch_db_;
std::unique_ptr<Database> ParallelAnalyzeTest::wide_db_;
std::string ParallelAnalyzeTest::tpch_file_;
std::string ParallelAnalyzeTest::wide_file_;
std::string ParallelAnalyzeTest::data_dir_;

// Queries over 40 row groups whose rows pass through operators on every worker:
// a filter above a join, a HAVING filter, a computed projection, a top n.
const std::vector<std::pair<std::string, std::string>>& WideQueries() {
    static const std::vector<std::pair<std::string, std::string>> queries = {
        {"grouped", "SELECT w_grp, COUNT(*), SUM(w_amt) FROM wide40 WHERE w_amt > 200.00 GROUP BY w_grp HAVING w_grp < 30 ORDER BY w_grp"},
        {"join filter", "SELECT COUNT(*), SUM(s_val) FROM wide40 JOIN side70 ON w_id = s_id WHERE w_amt > s_val"},
        {"projection", "SELECT w_id, w_amt * 2 + 1 FROM wide40 WHERE w_tag = 'blue' AND w_amt < 50.00"},
        {"top n", "SELECT w_id, w_amt, w_tag FROM wide40 WHERE w_amt > 990.00 ORDER BY w_amt DESC, w_id LIMIT 30"},
    };
    return queries;
}

}  // namespace

// A LIMIT over a SORT runs as one top n. Its Sort reports the rows
// it hands up, what tuple mode's Limit pulls from its Sort,
// so rows match there too.
// The extra queries add a HAVING filter, a join's filter and a nested-loop join.
TEST_F(ParallelAnalyzeTest, NamesAndRowsMatchTupleMode) {
    NeedTpch();
    if (IsSkipped()) return;
    Database* db = tpch_db_.get();
    std::vector<std::pair<std::string, std::string>> queries;
    for (const auto& q : tpch::Queries()) queries.emplace_back(q.tag, q.sql);
    queries.emplace_back("having", "SELECT n_regionkey, COUNT(*) FROM nation GROUP BY n_regionkey HAVING n_regionkey > 1 ORDER BY n_regionkey");
    queries.emplace_back("join filter", "SELECT COUNT(*), SUM(o_totalprice) FROM orders JOIN customer ON o_custkey = c_custkey WHERE o_totalprice > c_acctbal * 30");
    queries.emplace_back("nested loop", "SELECT r_name, n_name FROM region JOIN nation ON r_regionkey < n_regionkey ORDER BY r_name, n_name");
    for (const auto& [tag, sql] : queries) {
        SCOPED_TRACE(tag);
        const Reference want = TupleAnalyzeOnParallelPlan(*db, sql);
        const QueryResult got = Analyze(*db, sql, 1);
        EXPECT_EQ(got.tuples.size(), want.result_rows);
        ExpectSameShapeAndRows(got.profile.operators, want.operators);
        EXPECT_EQ(got.profile.step_rows, want.step_rows);
        EXPECT_EQ(got.profile.scan_rows, want.scan_rows);
        EXPECT_EQ(got.profile.scan_rows.size(), got.profile.join.order.size());
    }
}

TEST_F(ParallelAnalyzeTest, TimesAreNonNegativeAndAddUp) {
    NeedTpch();
    if (IsSkipped()) return;
    Database* db = tpch_db_.get();
    for (const auto& q : tpch::Queries()) {
        SCOPED_TRACE(q.tag);
        ExpectTimesAddUp(Analyze(*db, q.sql, 1).profile.operators);
    }
}

// Rows equal one thread's at every count; times are thread time, summed over
// workers, so they may pass the query's wall time. Tuple mode is too slow
// here in Debug, so the extra queries above compare these operators with it.
TEST_F(ParallelAnalyzeTest, SeveralThreadsOnFortyRowGroups) {
    Database& db = NeedWide();
    for (const auto& [tag, sql] : WideQueries()) {
        SCOPED_TRACE(tag);
        const QueryResult one = Analyze(db, sql, 1);
        ExpectTimesAddUp(one.profile.operators);
        for (unsigned threads : kThreadCounts) {
            SCOPED_TRACE(std::to_string(threads) + " threads");
            const QueryResult many = Analyze(db, sql, threads);
            EXPECT_EQ(db.LastParallelStats().threads, threads);
            EXPECT_GT(db.LastParallelStats().morsels, 30u) << "the scan spans many row groups";
            EXPECT_EQ(many.tuples.size(), one.tuples.size());
            ExpectSameShapeAndRows(many.profile.operators, one.profile.operators);
            EXPECT_EQ(many.profile.step_rows, one.profile.step_rows);
            EXPECT_EQ(many.profile.scan_rows, one.profile.scan_rows);
            ExpectTimesAddUp(many.profile.operators);
        }
    }
}

// A LIMIT with no ORDER BY stops early. Workers read ahead of it,
// and a scan counts whole morsels, so rows below the limit may pass tuple mode's.
TEST_F(ParallelAnalyzeTest, EarlyLimitRowsAtLeastTupleMode) {
    Database& db = NeedWide();
    const std::string sql = "SELECT w_id, w_tag FROM wide40 WHERE w_amt > 100.00 LIMIT 50";
    const Reference want = TupleAnalyzeOnParallelPlan(db, sql);
    ASSERT_EQ(want.result_rows, 50u);
    for (unsigned threads : {1u, 4u, 8u}) {
        SCOPED_TRACE(std::to_string(threads) + " threads");
        const QueryResult got = Analyze(db, sql, threads);
        EXPECT_EQ(got.tuples.size(), 50u);
        const auto& ops = got.profile.operators;
        ASSERT_EQ(ops.size(), want.operators.size()) << "got:\n" << Describe(ops) << "want:\n" << Describe(want.operators);
        EXPECT_EQ(ops[0].name, "Limit");
        EXPECT_EQ(ops[0].rows, 50u);
        for (size_t i = 0; i < ops.size(); ++i) {
            EXPECT_EQ(ops[i].name, want.operators[i].name);
            EXPECT_EQ(ops[i].depth, want.operators[i].depth);
            EXPECT_GE(ops[i].rows, want.operators[i].rows) << ops[i].name;
        }
        EXPECT_GT(ops.back().rows, want.operators.back().rows) << "the scan counts its whole first morsel";
        ExpectTimesAddUp(ops);
    }
}

// Without analyze the profile stays empty, apart from the join decision.
TEST_F(ParallelAnalyzeTest, OffLeavesTheProfileEmpty) {
    Database& db = NeedWide();
    QueryOptions options;
    options.mode = ExecutionMode::PARALLEL;
    options.threads = 4;
    const QueryResult r = db.ExecuteSQL(WideQueries()[1].second, options);
    ASSERT_EQ(r.tuples.size(), 1u);
    EXPECT_TRUE(r.profile.operators.empty());
    EXPECT_TRUE(r.profile.step_rows.empty());
    EXPECT_TRUE(r.profile.scan_rows.empty());
    EXPECT_EQ(r.profile.join.tables.size(), 2u);
}

}  // namespace shilmandb
