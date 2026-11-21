#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "catalog/catalog.hpp"
#include "engine/database.hpp"
#include "parser/parser.hpp"
#include "planner/cost_model.hpp"
#include "planner/index_selector.hpp"
#include "planner/plan_dump.hpp"
#include "planner/plan_node.hpp"
#include "planner/planner.hpp"
#include "storage/columnar/columnar_table.hpp"

// The parallel executor's columnar scan costs: a table scan against an index,
// priced with zone maps, late columns and position-ordered fetches.
namespace shilmandb {
namespace {

constexpr uint32_t kGroupRows = 65536;
constexpr uint32_t kRows = 4 * kGroupRows;  // cs: 4 row groups
constexpr uint32_t kWideRows = 20000;       // w18
constexpr uint32_t kW19Rows = 2000;         // w19
constexpr uint32_t kRowStoreRows = 5000;    // rs

// cs's columns, by number.
constexpr uint32_t kId = 0, kK = 1, kAmt = 2, kNote = 3, kTag = 4;

// k scatters the row number over the whole table, so its index jumps between groups.
uint32_t K(uint32_t r) {
    return static_cast<uint32_t>((static_cast<uint64_t>(r) * 7919u) % kRows);
}

// A negative entry cost makes any usable index win;
// a positive one with no other cost loses.
ColumnarScanCosts ForceIndex() { return ColumnarScanCosts{0.0, 0.0, -1.0, 0.0}; }
ColumnarScanCosts ForceSeq() { return ColumnarScanCosts{0.0, 0.0, 1.0, 0.0}; }

// The first node of this type in the plan, or null.
const PlanNode* FindNode(const PlanNode& root, PlanNodeType type) {
    if (root.type == type) return &root;
    for (const auto& child : root.children) {
        if (const PlanNode* found = FindNode(*child, type)) return found;
    }
    return nullptr;
}

// The filter right above the plan's index scan, or null.
const FilterPlanNode* FilterAboveIndexScan(const PlanNode& root) {
    if (root.type == PlanNodeType::FILTER && root.children[0]->type == PlanNodeType::INDEX_SCAN) return static_cast<const FilterPlanNode*>(&root);
    for (const auto& child : root.children) {
        if (const FilterPlanNode* found = FilterAboveIndexScan(*child)) return found;
    }
    return nullptr;
}

// The conjuncts of sql's WHERE, nested ANDs split; the statement owns them.
struct ParsedWhere {
    std::unique_ptr<SelectStatement> stmt;
    std::vector<const Expression*> conjuncts;
};

void SplitAnd(const Expression* e, std::vector<const Expression*>& out) {
    const auto* bin = dynamic_cast<const BinaryOp*>(e);
    if (bin && bin->op == BinaryOp::Op::AND) {
        SplitAnd(bin->left.get(), out);
        SplitAnd(bin->right.get(), out);
    } else if (e) {
        out.push_back(e);
    }
}

ParsedWhere Where(const std::string& sql) {
    ParsedWhere out;
    out.stmt = Parser(sql).Parse();
    SplitAnd(out.stmt->where_clause.get(), out.conjuncts);
    return out;
}

class ColumnarScanCostTest : public ::testing::Test {
protected:
    static std::unique_ptr<Database> db_;
    static std::string file_;

    static void SetUpTestSuite() {
        file_ = (std::filesystem::temp_directory_path() / ("shilmandb_columnar_scan_cost_test_" + std::to_string(::getpid()) + ".db")).string();
    }

    static void TearDownTestSuite() {
        db_.reset();
        std::filesystem::remove(file_);
    }

    // cs (columnar, indexes on id and k), w18 (columnar, a wide NUMERIC with an index)
    // and rs (row store, an index on k).
    static Database& Db() {
        if (db_) return *db_;
        std::filesystem::remove(file_);
        db_ = std::make_unique<Database>(file_, 4096);
        LoadOptions columnar;
        columnar.storage = StorageKind::COLUMNAR;

        const std::string cs = file_ + "_cs.tbl";
        {
            std::string out;
            out.reserve(static_cast<size_t>(kRows) * 40);
            char buf[96];
            for (uint32_t r = 0; r < kRows; ++r) {
                const uint32_t cents = static_cast<uint32_t>((static_cast<uint64_t>(r) * 104729u) % 1000000u);
                std::snprintf(buf, sizeof(buf), "%u|%u|%u.%02u|n%u|t%u\n", r, K(r), cents / 100, cents % 100, r, r % 7);
                out += buf;
            }
            std::ofstream(cs, std::ios::binary) << out;
        }
        db_->LoadTable("cs",
                       Schema({Column("id", ColumnType::Integer()), Column("k", ColumnType::Integer()), Column("amt", ColumnType::Numeric(12, 2)),
                               Column("note", ColumnType::Text()), Column("tag", ColumnType::Varchar(8))}),
                       cs, columnar);
        std::filesystem::remove(cs);

        // Row 1 holds a value past double precision; the others are small.
        const std::string w18 = file_ + "_w18.tbl";
        {
            std::ofstream f(w18, std::ios::binary);
            f << "1|1234567890123456.78\n";
            for (uint32_t i = 2; i <= kWideRows; ++i) f << i << "|" << i << ".25\n";
        }
        db_->LoadTable("w18", Schema({Column("w_id", ColumnType::Integer()), Column("m18", ColumnType::Numeric(18, 2))}), w18, columnar);
        std::filesystem::remove(w18);

        // Row 1 holds values a double can't tell from their neighbours; the others are small.
        const std::string w19 = file_ + "_w19.tbl";
        {
            std::ofstream f(w19, std::ios::binary);
            f << "1|1000000000000.500001|9007199254740993|1234567890123.45\n";
            for (uint32_t i = 2; i <= kW19Rows; ++i) f << i << "|" << i << ".25|" << i << "|" << i << ".25\n";
        }
        db_->LoadTable("w19",
                       Schema({Column("v_id", ColumnType::Integer()), Column("m", ColumnType::Numeric(19, 6)), Column("big", ColumnType::BigInt()),
                               Column("n15", ColumnType::Numeric(15, 2))}),
                       w19, columnar);
        std::filesystem::remove(w19);

        const std::string rs = file_ + "_rs.tbl";
        {
            std::ofstream f(rs, std::ios::binary);
            for (uint32_t r = 0; r < kRowStoreRows; ++r) f << r << "|" << (r * 7919u) % kRowStoreRows << "\n";
        }
        db_->LoadTable("rs", Schema({Column("rs_id", TypeId::INTEGER), Column("rs_k", TypeId::INTEGER)}), rs);
        std::filesystem::remove(rs);

        auto* catalog = db_->GetCatalog();
        (void)catalog->CreateIndex("cs_id", "cs", "id");
        (void)catalog->CreateIndex("cs_k", "cs", "k");
        (void)catalog->CreateIndex("w18_m18", "w18", "m18");
        (void)catalog->CreateIndex("rs_k", "rs", "rs_k");
        (void)catalog->CreateIndex("w19_m", "w19", "m");
        (void)catalog->CreateIndex("w19_big", "w19", "big");
        (void)catalog->CreateIndex("w19_n15", "w19", "n15");
        for (const std::string t : {"cs", "w18", "w19", "rs"}) catalog->UpdateTableStats(t);
        return *db_;
    }

    static std::unique_ptr<PlanNode> PlanFor(const std::string& sql, ExecutionMode mode) {
        QueryOptions options;
        options.mode = mode;
        return Db().MakePlanner(options).Plan(std::move(*Parser(sql).Parse()));
    }

    static bool UsesIndex(const std::string& sql, ExecutionMode mode = ExecutionMode::PARALLEL) {
        const auto plan = PlanFor(sql, mode);
        return FindNode(*plan, PlanNodeType::INDEX_SCAN) != nullptr;
    }

    // The first column of every row, as text, sorted.
    static std::vector<std::string> FirstColumn(const std::string& sql) {
        QueryOptions options;
        options.mode = ExecutionMode::PARALLEL;
        options.threads = 1;
        const QueryResult r = Db().ExecuteSQL(sql, options);
        std::vector<std::string> out;
        for (const Tuple& t : r.tuples) out.push_back(t.GetValue(r.schema, 0).ToString());
        std::sort(out.begin(), out.end());
        return out;
    }

    static const ColumnarTable& Cs() { return *Db().GetCatalog()->GetTable("cs")->columnar; }

    // Pages of column c over the listed row groups.
    static double Pages(uint32_t c, const std::vector<size_t>& groups) {
        double pages = 0;
        for (size_t g : groups) pages += Cs().ChunkAt(g, c).page_count;
        return pages;
    }
};

std::unique_ptr<Database> ColumnarScanCostTest::db_;
std::string ColumnarScanCostTest::file_;

TEST_F(ColumnarScanCostTest, ExpectedPagesTouchedBounds) {
    EXPECT_EQ(ExpectedPagesTouched(100.0, 0.0), 0.0);
    EXPECT_EQ(ExpectedPagesTouched(0.0, 50.0), 0.0);
    EXPECT_NEAR(ExpectedPagesTouched(100.0, 1.0), 1.0, 1e-9);
    EXPECT_NEAR(ExpectedPagesTouched(1.0, 7.0), 1.0, 1e-9);
    EXPECT_NEAR(ExpectedPagesTouched(100.0, 100000.0), 100.0, 1e-6);
    // Two positions on 2 pages share one half the time: 1.5 pages.
    EXPECT_NEAR(ExpectedPagesTouched(2.0, 2.0), 1.5, 1e-9);

    // Never more than the pages or the positions, and growing with the positions.
    double last = 0.0;
    for (double k : {1.0, 10.0, 100.0, 1000.0, 10000.0}) {
        const double touched = ExpectedPagesTouched(500.0, k);
        EXPECT_LE(touched, std::min(500.0, k) + 1e-9) << k;
        EXPECT_GT(touched, last) << k;
        last = touched;
    }
}

TEST_F(ColumnarScanCostTest, RowModesIgnoreColumnarCosts) {
    // Today's plans don't move whatever the override says; nor do row-store tables' in any mode.
    Database& db = Db();
    const std::vector<std::string> queries = {
        "SELECT id, k, amt FROM cs WHERE k >= 1000 AND k <= 1261",
        "SELECT id, k, amt FROM cs WHERE k >= 1000 AND k <= 132071",
        "SELECT note FROM cs WHERE id >= 70000 AND id <= 70500 AND amt < 50.00",
        "SELECT w_id FROM w18 WHERE m18 = 1234567890123456.78",
        "SELECT rs_id FROM rs WHERE rs_k >= 10 AND rs_k <= 20",
        "SELECT rs_id FROM rs WHERE rs_k >= 10",
    };
    const std::vector<std::optional<ColumnarScanCosts>> overrides = {std::nullopt, ForceSeq(), ForceIndex(), ColumnarScanCosts{5.0, 1.0, 3.0, 2.0}};
    for (const std::string& sql : queries) {
        SCOPED_TRACE(sql);
        const bool row_store = sql.find("FROM rs") != std::string::npos;
        std::vector<ExecutionMode> modes = {ExecutionMode::TUPLE, ExecutionMode::VECTORIZED};
        if (row_store) modes.push_back(ExecutionMode::PARALLEL);
        for (ExecutionMode mode : modes) {
            db.SetColumnarScanCosts(std::nullopt);
            const std::string want = test::DumpPlan(*PlanFor(sql, mode));
            for (const auto& o : overrides) {
                db.SetColumnarScanCosts(o);
                EXPECT_EQ(test::DumpPlan(*PlanFor(sql, mode)), want) << "mode " << static_cast<int>(mode);
            }
        }
    }
    db.SetColumnarScanCosts(std::nullopt);
    // The row store picks its index for a narrow range
    // and a table scan for a wide one, as today.
    EXPECT_TRUE(UsesIndex("SELECT rs_id FROM rs WHERE rs_k >= 10 AND rs_k <= 20"));
    EXPECT_FALSE(UsesIndex("SELECT rs_id FROM rs WHERE rs_k >= 10"));
}

TEST_F(ColumnarScanCostTest, PicksScanWhenIndexTouchesEveryPage) {
    // Half of k, scattered over every group: the index fetches touch every page anyway.
    Database& db = Db();
    db.SetColumnarScanCosts(std::nullopt);
    const std::string sql = "SELECT id, k, amt, note FROM cs WHERE k >= 1000 AND k <= 132071";
    EXPECT_FALSE(UsesIndex(sql));
    EXPECT_NE(FindNode(*PlanFor(sql, ExecutionMode::PARALLEL), PlanNodeType::SEQ_SCAN), nullptr);

    const ParsedWhere where = Where(sql);
    const std::vector<uint32_t> read = {kId, kK, kAmt, kNote};
    const ColumnarScanCosts defaults;
    const ScanCostEstimate est = IndexSelector::EstimateScanCosts("cs", where.conjuncts, db.GetCatalog(), ScanCosts{}, &defaults, &read);
    ASSERT_EQ(est.indexes.size(), 1u);
    EXPECT_EQ(est.indexes[0].first, "cs_k");
    EXPECT_GT(est.indexes[0].second, est.seq);
}

TEST_F(ColumnarScanCostTest, PicksIndexForANarrowRange) {
    // 262 rows of k (0.1%), scattered: the index beats decoding k for every row.
    Database& db = Db();
    db.SetColumnarScanCosts(std::nullopt);
    for (const std::string& select : {std::string("id, k, amt, note"), std::string("*"), std::string("COUNT(*)")}) {
        const std::string sql = "SELECT " + select + " FROM cs WHERE k >= 1000 AND k <= 1261";
        SCOPED_TRACE(sql);
        const auto plan = PlanFor(sql, ExecutionMode::PARALLEL);
        const auto* scan = static_cast<const IndexScanPlanNode*>(FindNode(*plan, PlanNodeType::INDEX_SCAN));
        ASSERT_NE(scan, nullptr);
        EXPECT_EQ(scan->index_name, "cs_k");
    }
    const std::vector<std::string> got = FirstColumn("SELECT k FROM cs WHERE k >= 1000 AND k <= 1261");
    EXPECT_EQ(got.size(), 262u);
}

TEST_F(ColumnarScanCostTest, CostsFollowTheFormula) {
    // Unit constants give each term alone, which is also how a fit gets its inputs.
    Database& db = Db();
    const ParsedWhere where = Where("SELECT id FROM cs WHERE id >= 70000 AND id <= 70500");
    const TableStats& stats = db.GetCatalog()->GetTable("cs")->stats;
    const std::vector<uint32_t> read = {kId, kAmt};
    const auto price = [&](const ColumnarScanCosts& c) {
        return IndexSelector::EstimateScanCosts("cs", where.conjuncts, db.GetCatalog(), ScanCosts{}, &c, &read);
    };

    // The zone maps keep group 1 alone;
    // amt is read only for the rows the filter keeps.
    const double out = std::min<double>(kGroupRows, kRows * CostModel::EstimateSelectivity(where.conjuncts, stats));
    const ScanCostEstimate pages = price(ColumnarScanCosts{1.0, 0.0, 0.0, 0.0});
    EXPECT_NEAR(pages.seq, Pages(kId, {1}) + ExpectedPagesTouched(Pages(kAmt, {1}), out), 1e-9);
    const ScanCostEstimate values = price(ColumnarScanCosts{0.0, 1.0, 0.0, 0.0});
    EXPECT_NEAR(values.seq, kGroupRows + out, 1e-9);

    // The index: a descent, one entry per match, and position-ordered fetches per read column.
    const double matches = std::max(1.0, kRows * CostModel::EstimateSelectivity(where.conjuncts, stats, "id"));
    const int height = CostModel::EstimateTreeHeight(stats.row_count, TypeId::INTEGER);
    ASSERT_EQ(pages.indexes.size(), 1u);
    EXPECT_EQ(pages.indexes[0].first, "cs_id");
    const std::vector<size_t> all = {0, 1, 2, 3};
    EXPECT_NEAR(pages.indexes[0].second, height + ExpectedPagesTouched(Pages(kId, all), matches) + ExpectedPagesTouched(Pages(kAmt, all), matches), 1e-9);
    EXPECT_NEAR(values.indexes[0].second, 0.0, 1e-9);
    EXPECT_NEAR(price(ColumnarScanCosts{0.0, 0.0, 1.0, 0.0}).indexes[0].second, matches, 1e-9);
    EXPECT_NEAR(price(ColumnarScanCosts{0.0, 0.0, 0.0, 1.0}).indexes[0].second, 2 * matches, 1e-9);

    // Every column when no list is given; row-store prices when no columnar costs are given.
    const ScanCostEstimate every = IndexSelector::EstimateScanCosts("cs", where.conjuncts, db.GetCatalog(), ScanCosts{}, nullptr, nullptr);
    EXPECT_NEAR(every.seq, CostModel::SeqScanCost(stats, db.GetCatalog()->GetTable("cs")->schema), 1e-9);
    const ColumnarScanCosts unit_values{0.0, 1.0, 0.0, 0.0};
    const ScanCostEstimate wide = IndexSelector::EstimateScanCosts("cs", where.conjuncts, db.GetCatalog(), ScanCosts{}, &unit_values, nullptr);
    EXPECT_NEAR(wide.seq, kGroupRows + 4 * out, 1e-9);
}

TEST_F(ColumnarScanCostTest, ReadColumnsComeFromTheStatement) {
    // With these constants a 10% range on k
    // picks the index only when few columns are read.
    Database& db = Db();
    db.SetColumnarScanCosts(ColumnarScanCosts{0.0, 0.25, 0.0, 1.0});
    const std::string range = " WHERE k >= 100000 AND k <= 126213";
    EXPECT_TRUE(UsesIndex("SELECT k FROM cs" + range));
    EXPECT_TRUE(UsesIndex("SELECT COUNT(*) FROM cs" + range));
    EXPECT_TRUE(UsesIndex("SELECT c.k FROM cs c WHERE c.k >= 100000 AND c.k <= 126213"));
    EXPECT_FALSE(UsesIndex("SELECT * FROM cs" + range));
    EXPECT_FALSE(UsesIndex("SELECT id, k, amt, note, tag FROM cs" + range));
    EXPECT_FALSE(UsesIndex("SELECT k FROM cs" + range + " ORDER BY id, amt, note, tag"));
    db.SetColumnarScanCosts(std::nullopt);
}

TEST_F(ColumnarScanCostTest, OverrideForcesEitherPlan) {
    Database& db = Db();
    const ColumnarScanCosts defaults;
    EXPECT_EQ(defaults.page_us, ColumnarScanCosts::kPageUs);
    EXPECT_EQ(db.GetColumnarScanCosts().value_us, ColumnarScanCosts::kValueUs);

    const std::string sql = "SELECT id, amt FROM cs WHERE id >= 70000 AND id <= 70500";
    db.SetColumnarScanCosts(ForceIndex());
    EXPECT_EQ(db.GetColumnarScanCosts().index_entry_us, -1.0);
    EXPECT_TRUE(UsesIndex(sql));
    const std::vector<std::string> by_index = FirstColumn(sql);
    // Even a range the zone maps rule out entirely.
    EXPECT_TRUE(UsesIndex("SELECT id FROM cs WHERE id >= 10 AND id <= 5"));

    db.SetColumnarScanCosts(ForceSeq());
    EXPECT_FALSE(UsesIndex(sql));
    EXPECT_EQ(FirstColumn(sql), by_index);
    EXPECT_EQ(by_index.size(), 501u);

    db.SetColumnarScanCosts(std::nullopt);
    EXPECT_EQ(db.GetColumnarScanCosts().index_entry_us, ColumnarScanCosts::kIndexEntryUs);
}

TEST_F(ColumnarScanCostTest, WideLiteralsStayAsAFilter) {
    // A literal past 15 significant digits, or a column whose values pass them, isn't
    // exact as a double: the index range only narrows the rows and the comparison
    // runs again, exactly, above the scan.
    Database& db = Db();
    std::vector<std::string> small_rows;  // w19's rows 2 .. 2000
    for (uint32_t i = 2; i <= kW19Rows; ++i) small_rows.push_back(std::to_string(i));
    std::sort(small_rows.begin(), small_rows.end());
    struct Case {
        std::string sql;
        std::vector<std::string> want;
        bool today_uses_index;  // whether today's row-store prices pick the index
    };
    // Today's prices pick the index for a few matches,
    // and a table scan for a range that keeps nearly every row.
    const std::vector<Case> cases = {
        {"SELECT w_id FROM w18 WHERE m18 = 1234567890123456.78", {"1"}, true},
        {"SELECT w_id FROM w18 WHERE m18 = 1234567890123456.77", {}, true},
        {"SELECT w_id FROM w18 WHERE m18 >= 1234567890123456.79", {}, true},
        // Short literals on wide columns: NUMERIC(18,2), NUMERIC(19,6) and BIGINT.
        {"SELECT w_id FROM w18 WHERE m18 = 19999.25", {"19999"}, true},
        {"SELECT w_id FROM w18 WHERE m18 >= 123456789012.25", {"1"}, false},
        {"SELECT v_id FROM w19 WHERE m = 1000000000000.5", {}, true},
        {"SELECT v_id FROM w19 WHERE m <= 1000000000000.5", small_rows, false},
        {"SELECT v_id FROM w19 WHERE big >= 2000", {"1", "2000"}, false},
    };
    for (const Case& c : cases) {
        SCOPED_TRACE(c.sql);
        db.SetColumnarScanCosts(ForceIndex());
        const auto plan = PlanFor(c.sql, ExecutionMode::PARALLEL);
        ASSERT_NE(FindNode(*plan, PlanNodeType::INDEX_SCAN), nullptr);
        EXPECT_NE(FilterAboveIndexScan(*plan), nullptr) << "the comparison was consumed by the index";
        EXPECT_EQ(FirstColumn(c.sql), c.want);

        db.SetColumnarScanCosts(ForceSeq());
        ASSERT_EQ(FindNode(*PlanFor(c.sql, ExecutionMode::PARALLEL), PlanNodeType::INDEX_SCAN), nullptr);
        EXPECT_EQ(FirstColumn(c.sql), c.want);

        // Today's modes plan as before, and their index still consumes the bound.
        const auto today = PlanFor(c.sql, ExecutionMode::VECTORIZED);
        ASSERT_EQ(FindNode(*today, PlanNodeType::INDEX_SCAN) != nullptr, c.today_uses_index);
        EXPECT_EQ(FilterAboveIndexScan(*today), nullptr);
    }

    // On a NUMERIC(15,2) column, literals of 15 digits or fewer are exact,
    // so the index consumes them.
    db.SetColumnarScanCosts(ForceIndex());
    const std::vector<Case> exact = {
        {"SELECT v_id FROM w19 WHERE n15 = 1999.25", {"1999"}, true},
        {"SELECT v_id FROM w19 WHERE n15 >= 1234567890123.45", {"1"}, true},
    };
    for (const Case& c : exact) {
        SCOPED_TRACE(c.sql);
        const auto plan = PlanFor(c.sql, ExecutionMode::PARALLEL);
        ASSERT_NE(FindNode(*plan, PlanNodeType::INDEX_SCAN), nullptr);
        EXPECT_EQ(FilterAboveIndexScan(*plan), nullptr);
        EXPECT_EQ(FirstColumn(c.sql), c.want);
    }
    db.SetColumnarScanCosts(std::nullopt);
}

}  // namespace
}  // namespace shilmandb
