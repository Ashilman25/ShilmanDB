#include <gtest/gtest.h>
#include <unistd.h>

#include "engine/database.hpp"
#include "tpch/tpch_workload.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace shilmandb {

// ── Index descriptors (this suite's own 6, not the shared 8) ────────

struct IndexDesc {
    std::string index_name;
    std::string table_name;
    std::string column_name;
};

static std::vector<IndexDesc> BuildIndexDescs() {
    return {
        {"idx_orders_custkey",    "orders",   "o_custkey"},
        {"idx_lineitem_orderkey", "lineitem", "l_orderkey"},
        {"idx_customer_custkey",  "customer", "c_custkey"},
        {"idx_supplier_nationkey","supplier", "s_nationkey"},
        {"idx_nation_nationkey",  "nation",   "n_nationkey"},
        {"idx_part_partkey",      "part",     "p_partkey"},
    };
}

// ── Fixture: one database per storage, loaded once for the suite ────

// A storage and the mode every query of the test runs in.
struct StorageMode {
    StorageKind storage;
    ExecutionMode mode;
};

// Runs each query in one mode, so the bodies keep calling db_->ExecuteSQL(sql).
class ModeDatabase {
public:
    ModeDatabase() = default;
    ModeDatabase(Database* db, ExecutionMode mode) : db_(db), mode_(mode) {}

    // The query in this test's mode.
    [[nodiscard]] QueryResult ExecuteSQL(const std::string& sql) const { return db_->ExecuteSQL(sql, mode_); }

    // Lets db_->ExecuteSQL(sql) read as it did with a plain Database pointer.
    const ModeDatabase* operator->() const { return this; }

private:
    Database* db_{nullptr};
    ExecutionMode mode_{ExecutionMode::TUPLE};
};

class SqliteComparisonTest : public ::testing::TestWithParam<StorageMode> {
protected:
    static std::unique_ptr<Database> row_db_;
    static std::unique_ptr<Database> columnar_db_;
    static std::string row_file_;
    static std::string columnar_file_;
    static std::string data_dir_;
    ModeDatabase db_;

    // Try several candidate paths for the SF=0.01 data directory.
    // CTest runs from build/test/, so ../../bench/... is the typical relative path.
    static std::string FindDataDir() {
        std::vector<std::string> candidates = {
            "bench/tpch_data/sf0.01/",
            "../bench/tpch_data/sf0.01/",
            "../../bench/tpch_data/sf0.01/",
        };
        for (const auto& dir : candidates) {
            if (std::filesystem::exists(dir + "lineitem.tbl")) {
                return dir;
            }
        }
        return {};
    }

    // Loads the TPC-H tables into one storage, then adds the indexes and stats.
    static std::unique_ptr<Database> Load(const std::string& file, const std::string& data_dir,
                                          StorageKind storage) {
        std::filesystem::remove(file);

        // Large pool so eviction does not interfere with correctness.
        constexpr size_t kPoolSize = 4096;
        auto db = std::make_unique<Database>(file, kPoolSize);

        LoadOptions options;
        options.storage = storage;
        for (const auto& td : tpch::TableDescs()) {
            db->LoadTable(td.name, td.schema, data_dir + td.tbl_file, options);
        }

        auto* catalog = db->GetCatalog();
        for (const auto& idx : BuildIndexDescs()) {
            (void)catalog->CreateIndex(idx.index_name, idx.table_name, idx.column_name);
        }
        for (const auto& td : tpch::TableDescs()) {
            catalog->UpdateTableStats(td.name);
        }
        return db;
    }

    static void SetUpTestSuite() {
        const auto temp = std::filesystem::temp_directory_path();
        const std::string pid = std::to_string(::getpid());
        row_file_ = (temp / ("shilmandb_sqlite_cmp_test_row_" + pid + ".db")).string();
        columnar_file_ = (temp / ("shilmandb_sqlite_cmp_test_col_" + pid + ".db")).string();
        data_dir_ = FindDataDir();
    }

    static void TearDownTestSuite() {
        row_db_.reset();
        columnar_db_.reset();
        std::filesystem::remove(row_file_);
        std::filesystem::remove(columnar_file_);
    }

    // Loads only this test's storage, the first time a test asks for it.
    void SetUp() override {
        if (data_dir_.empty()) {
            GTEST_SKIP() << "TPC-H SF=0.01 data not found. "
                            "Run bench/tpch/generate_tpch_data.sh first.";
        }
        const bool row = GetParam().storage == StorageKind::ROW;
        auto& slot = row ? row_db_ : columnar_db_;
        if (!slot) slot = Load(row ? row_file_ : columnar_file_, data_dir_, GetParam().storage);
        db_ = ModeDatabase(slot.get(), GetParam().mode);
    }
};

std::unique_ptr<Database> SqliteComparisonTest::row_db_;
std::unique_ptr<Database> SqliteComparisonTest::columnar_db_;
std::string SqliteComparisonTest::row_file_;
std::string SqliteComparisonTest::columnar_file_;
std::string SqliteComparisonTest::data_dir_;

// Row, Columnar or Columnar_Parallel; the row store never runs in parallel mode.
static std::string NameOf(const StorageMode& p) {
    const std::string storage = p.storage == StorageKind::ROW ? "Row" : "Columnar";
    return p.mode == ExecutionMode::PARALLEL ? storage + "_Parallel" : storage;
}

// Names the pair in test output, so ctest shows /Row, /Columnar and /Columnar_Parallel.
static void PrintTo(const StorageMode& p, std::ostream* os) {
    *os << NameOf(p);
}

INSTANTIATE_TEST_SUITE_P(Storage, SqliteComparisonTest,
                         ::testing::Values(StorageMode{StorageKind::ROW, ExecutionMode::TUPLE},
                                           StorageMode{StorageKind::COLUMNAR, ExecutionMode::TUPLE},
                                           StorageMode{StorageKind::COLUMNAR, ExecutionMode::PARALLEL}),
                         [](const ::testing::TestParamInfo<StorageMode>& info) { return NameOf(info.param); });

// ── Q1: Pricing Summary ─────────────────────────────────────────────

TEST_P(SqliteComparisonTest, Q1PricingSummary) {
    const std::string sql =
        "SELECT l_returnflag, l_linestatus, SUM(l_quantity), SUM(l_extendedprice), "
        "SUM(l_discount), COUNT(*) "
        "FROM lineitem "
        "WHERE l_shipdate <= '1998-09-02' "
        "GROUP BY l_returnflag, l_linestatus "
        "ORDER BY l_returnflag, l_linestatus";

    auto result = db_->ExecuteSQL(sql);
    ASSERT_EQ(result.tuples.size(), 4u);

    // Expected rows ordered by (l_returnflag, l_linestatus):
    //   A|F, N|F, N|O, R|F
    struct Q1Row {
        std::string returnflag;
        std::string linestatus;
        double sum_quantity;
        double sum_extendedprice;
        double sum_discount;
        int32_t count;
    };

    const std::vector<Q1Row> expected = {
        {"A", "F", 380456.0,  532348211.65, 745.01, 14876},
        {"N", "F",   8971.0,   12384801.37,  16.62,   348},
        {"N", "O", 742802.0, 1041502841.45, 1457.04, 29181},
        {"R", "F", 381449.0,  534594445.35, 742.53, 14902},
    };

    for (size_t i = 0; i < expected.size(); ++i) {
        const auto& row = expected[i];
        const auto& tuple = result.tuples[i];

        EXPECT_EQ(tuple.GetValue(result.schema, 0).varchar_, row.returnflag)
            << "Row " << i << " l_returnflag";
        EXPECT_EQ(tuple.GetValue(result.schema, 1).varchar_, row.linestatus)
            << "Row " << i << " l_linestatus";
        EXPECT_NEAR(tuple.GetValue(result.schema, 2).decimal_, row.sum_quantity, 0.01)
            << "Row " << i << " SUM(l_quantity)";
        EXPECT_NEAR(tuple.GetValue(result.schema, 3).decimal_, row.sum_extendedprice, 0.01)
            << "Row " << i << " SUM(l_extendedprice)";
        EXPECT_NEAR(tuple.GetValue(result.schema, 4).decimal_, row.sum_discount, 0.01)
            << "Row " << i << " SUM(l_discount)";
        EXPECT_EQ(tuple.GetValue(result.schema, 5).integer_, row.count)
            << "Row " << i << " COUNT(*)";
    }
}

// ── Q6: Revenue Forecast ────────────────────────────────────────────

TEST_P(SqliteComparisonTest, Q6RevenueForecast) {
    const std::string sql =
        "SELECT SUM(l_extendedprice * l_discount) "
        "FROM lineitem "
        "WHERE l_shipdate >= '1994-01-01' AND l_shipdate < '1995-01-01' "
        "AND l_discount >= 0.05 AND l_discount <= 0.07 "
        "AND l_quantity < 24";

    auto result = db_->ExecuteSQL(sql);
    ASSERT_EQ(result.tuples.size(), 1u);

    EXPECT_NEAR(result.tuples[0].GetValue(result.schema, 0).decimal_,
                1193053.23, 0.01);
}

// ── Q3: Shipping Priority ───────────────────────────────────────────

TEST_P(SqliteComparisonTest, Q3ShippingPriority) {
    const std::string sql =
        "SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)), "
        "o_orderdate, o_shippriority "
        "FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey "
        "JOIN lineitem l ON l.l_orderkey = o.o_orderkey "
        "WHERE c.c_mktsegment = 'BUILDING' "
        "AND o.o_orderdate < '1995-03-15' "
        "AND l.l_shipdate > '1995-03-15' "
        "GROUP BY l_orderkey, o_orderdate, o_shippriority "
        "ORDER BY SUM(l_extendedprice * (1 - l_discount)) DESC "
        "LIMIT 10";

    auto result = db_->ExecuteSQL(sql);
    ASSERT_EQ(result.tuples.size(), 10u);

    // Spot-check first row.
    auto row0_orderkey = result.tuples[0].GetValue(result.schema, 0).bigint_;
    EXPECT_EQ(row0_orderkey, 47714)
        << "Q3 first-row l_orderkey";

    auto row0_disc_price = result.tuples[0].GetValue(result.schema, 1).decimal_;
    EXPECT_NEAR(row0_disc_price, 267010.59, 0.01)
        << "Q3 first-row SUM(disc_price)";

    auto row0_date = result.tuples[0].GetValue(result.schema, 2).ToString();
    EXPECT_EQ(row0_date, "1995-03-11")
        << "Q3 first-row o_orderdate";

    auto row0_priority = result.tuples[0].GetValue(result.schema, 3).integer_;
    EXPECT_EQ(row0_priority, 0)
        << "Q3 first-row o_shippriority";
}

// ── Q5: Local Supplier Volume ───────────────────────────────────────

TEST_P(SqliteComparisonTest, Q5LocalSupplierVolume) {
    const std::string sql =
        "SELECT n_name, SUM(l_extendedprice * (1 - l_discount)) "
        "FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey "
        "JOIN lineitem l ON l.l_orderkey = o.o_orderkey "
        "JOIN supplier s ON l.l_suppkey = s.s_suppkey "
        "JOIN nation n ON s.s_nationkey = n.n_nationkey "
        "WHERE o.o_orderdate >= '1994-01-01' AND o.o_orderdate < '1995-01-01' "
        "GROUP BY n_name "
        "ORDER BY SUM(l_extendedprice * (1 - l_discount)) DESC";

    auto result = db_->ExecuteSQL(sql);
    ASSERT_EQ(result.tuples.size(), 25u);

    // Spot-check first row.
    auto row0_name = result.tuples[0].GetValue(result.schema, 0).varchar_;
    EXPECT_EQ(row0_name, "UNITED STATES")
        << "Q5 first-row n_name";

    auto row0_revenue = result.tuples[0].GetValue(result.schema, 1).decimal_;
    EXPECT_NEAR(row0_revenue, 26931889.33, 0.01)
        << "Q5 first-row SUM(revenue)";
}

// ── Q10: Returned Item Reporting ────────────────────────────────────

TEST_P(SqliteComparisonTest, Q10ReturnedItemReporting) {
    const std::string sql =
        "SELECT c_custkey, c_name, SUM(l_extendedprice * (1 - l_discount)), "
        "c_acctbal, n_name, c_address, c_phone, c_comment "
        "FROM customer, orders, lineitem, nation "
        "WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey "
        "AND o_orderdate >= '1993-10-01' AND o_orderdate < '1994-01-01' "
        "AND l_returnflag = 'R' AND c_nationkey = n_nationkey "
        "GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment "
        "ORDER BY SUM(l_extendedprice * (1 - l_discount)) DESC "
        "LIMIT 20";

    auto result = db_->ExecuteSQL(sql);
    ASSERT_EQ(result.tuples.size(), 20u);

    // First row: c_custkey=679, c_name="Customer#000000679",
    //            sum_disc_price≈378211.33, c_acctbal=1394.44, n_name="IRAN".
    EXPECT_EQ(result.tuples[0].GetValue(result.schema, 0).bigint_, 679)
        << "Q10 first-row c_custkey";
    EXPECT_EQ(result.tuples[0].GetValue(result.schema, 1).varchar_,
              "Customer#000000679")
        << "Q10 first-row c_name";
    EXPECT_NEAR(result.tuples[0].GetValue(result.schema, 2).decimal_,
                378211.33, 0.01)
        << "Q10 first-row SUM(disc_price)";
    EXPECT_NEAR(result.tuples[0].GetValue(result.schema, 3).decimal_,
                1394.44, 0.01)
        << "Q10 first-row c_acctbal";
    EXPECT_EQ(result.tuples[0].GetValue(result.schema, 4).varchar_, "IRAN")
        << "Q10 first-row n_name";
}

// ── Q12: Shipping Modes and Order Priority ──────────────────────────

TEST_P(SqliteComparisonTest, Q12ShippingModes) {
    const std::string sql =
        "SELECT l_shipmode, "
        "SUM(CASE WHEN o_orderpriority = '1-URGENT' OR "
        "              o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND "
        "              o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) "
        "FROM orders, lineitem "
        "WHERE o_orderkey = l_orderkey "
        "AND l_shipmode IN ('MAIL', 'SHIP') "
        "AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate "
        "AND l_receiptdate >= '1994-01-01' AND l_receiptdate < '1995-01-01' "
        "GROUP BY l_shipmode "
        "ORDER BY l_shipmode";

    auto result = db_->ExecuteSQL(sql);
    ASSERT_EQ(result.tuples.size(), 2u);

    // Row 0: MAIL, 64, 86
    EXPECT_EQ(result.tuples[0].GetValue(result.schema, 0).varchar_, "MAIL");
    EXPECT_NEAR(result.tuples[0].GetValue(result.schema, 1).decimal_, 64.0, 0.01)
        << "Q12 MAIL high_line_count";
    EXPECT_NEAR(result.tuples[0].GetValue(result.schema, 2).decimal_, 86.0, 0.01)
        << "Q12 MAIL low_line_count";

    // Row 1: SHIP, 61, 96
    EXPECT_EQ(result.tuples[1].GetValue(result.schema, 0).varchar_, "SHIP");
    EXPECT_NEAR(result.tuples[1].GetValue(result.schema, 1).decimal_, 61.0, 0.01)
        << "Q12 SHIP high_line_count";
    EXPECT_NEAR(result.tuples[1].GetValue(result.schema, 2).decimal_, 96.0, 0.01)
        << "Q12 SHIP low_line_count";
}

// ── Q14: Promotion Effect ───────────────────────────────────────────

TEST_P(SqliteComparisonTest, Q14PromotionEffect) {
    const std::string sql =
        "SELECT 100.00 * SUM(CASE WHEN p_type LIKE 'PROMO%' "
        "       THEN l_extendedprice * (1 - l_discount) ELSE 0 END) "
        "       / SUM(l_extendedprice * (1 - l_discount)) "
        "FROM lineitem, part "
        "WHERE l_partkey = p_partkey "
        "AND l_shipdate >= '1995-09-01' AND l_shipdate < '1995-10-01'";

    auto result = db_->ExecuteSQL(sql);
    ASSERT_EQ(result.tuples.size(), 1u);

    EXPECT_NEAR(result.tuples[0].GetValue(result.schema, 0).decimal_,
                15.48654581228407, 0.01)
        << "Q14 promo_revenue";
}

// ── Q19: Discounted Revenue ─────────────────────────────────────────

TEST_P(SqliteComparisonTest, Q19DiscountedRevenue) {
    const std::string sql =
        "SELECT SUM(l_extendedprice * (1 - l_discount)) "
        "FROM lineitem, part "
        "WHERE p_partkey = l_partkey "
        "AND ((p_brand = 'Brand#12' "
        "  AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG') "
        "  AND l_quantity >= 1 AND l_quantity <= 11 "
        "  AND p_size BETWEEN 1 AND 5 "
        "  AND l_shipmode IN ('AIR', 'AIR REG') "
        "  AND l_shipinstruct = 'DELIVER IN PERSON') "
        "OR (p_brand = 'Brand#23' "
        "  AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK') "
        "  AND l_quantity >= 10 AND l_quantity <= 20 "
        "  AND p_size BETWEEN 1 AND 10 "
        "  AND l_shipmode IN ('AIR', 'AIR REG') "
        "  AND l_shipinstruct = 'DELIVER IN PERSON') "
        "OR (p_brand = 'Brand#34' "
        "  AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG') "
        "  AND l_quantity >= 20 AND l_quantity <= 30 "
        "  AND p_size BETWEEN 1 AND 15 "
        "  AND l_shipmode IN ('AIR', 'AIR REG') "
        "  AND l_shipinstruct = 'DELIVER IN PERSON'))";

    auto result = db_->ExecuteSQL(sql);
    ASSERT_EQ(result.tuples.size(), 1u);

    EXPECT_NEAR(result.tuples[0].GetValue(result.schema, 0).decimal_,
                22923.028, 0.01)
        << "Q19 revenue";
}

}  // namespace shilmandb
