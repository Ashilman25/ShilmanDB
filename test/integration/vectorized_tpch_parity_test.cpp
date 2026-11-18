#include <gtest/gtest.h>
#include <unistd.h>

#include "engine/database.hpp"
#include "tpch/tpch_workload.hpp"
#include "executor/vectorized_parity_harness.hpp"

#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace shilmandb {

// This suite's own 6 indexes, not the shared 8.
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

// ── Fixture: load TPC-H SF=0.01 once per storage for the whole suite ─

// A storage and the mode each query is compared with tuple mode in.
struct StorageMode {
    StorageKind storage;
    ExecutionMode mode;
};

class VectorizedTpchParityTest : public ::testing::TestWithParam<StorageMode> {
protected:
    static std::unique_ptr<Database> row_db_;
    static std::unique_ptr<Database> columnar_db_;
    static std::string row_file_;
    static std::string columnar_file_;
    static std::string data_dir_;
    Database* db_{nullptr};

    // CTest runs from build/, so ../bench/... is the typical relative path.
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
        row_file_ = (temp / ("shilmandb_vec_tpch_parity_test_row_" + pid + ".db")).string();
        columnar_file_ = (temp / ("shilmandb_vec_tpch_parity_test_col_" + pid + ".db")).string();
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
        db_ = slot.get();
    }

    // Tuple mode against this test's mode: exact for vectorized; parallel sums
    // exactly, so its DECIMAL cells get SameResult's 1e-6 and the rest stay exact.
    void RunParity(const std::string& sql) {
        auto tuple_rs = db_->ExecuteSQL(sql, ExecutionMode::TUPLE);
        auto other_rs = db_->ExecuteSQL(sql, GetParam().mode);
        ASSERT_EQ(tuple_rs.tuples.size(), other_rs.tuples.size())
            << "TPC-H parity: row-count mismatch";
        if (GetParam().mode == ExecutionMode::VECTORIZED) {
            test::ExpectRowsEqual(tuple_rs.tuples, other_rs.tuples, tuple_rs.schema);
            return;
        }
        ASSERT_EQ(tuple_rs.schema.GetColumnCount(), other_rs.schema.GetColumnCount());
        for (uint32_t c = 0; c < tuple_rs.schema.GetColumnCount(); ++c) {
            ASSERT_EQ(tuple_rs.schema.GetColumn(c).type, other_rs.schema.GetColumn(c).type) << "parity: col " << c << " type";
        }
        EXPECT_TRUE(tpch::SameResult(tuple_rs, other_rs)) << "parity: " << FirstDifference(tuple_rs, other_rs);
    }

    // The first row SameResult rejects, both sides as text, for the failure message.
    static std::string FirstDifference(const QueryResult& a, const QueryResult& b) {
        for (size_t r = 0; r < a.tuples.size(); ++r) {
            const QueryResult x{a.schema, {a.tuples[r]}, {}};
            const QueryResult y{b.schema, {b.tuples[r]}, {}};
            if (!tpch::SameResult(x, y)) return "row " + std::to_string(r) + ": " + RowText(x) + "vs " + RowText(y);
        }
        return "none";
    }

    // A one-row result's cells; decimals with every digit.
    static std::string RowText(const QueryResult& one) {
        std::string out;
        for (uint32_t c = 0; c < one.schema.GetColumnCount(); ++c) {
            const Value v = one.tuples[0].GetValue(one.schema, c);
            if (v.type_ == TypeId::DECIMAL) {
                char buf[40];
                std::snprintf(buf, sizeof(buf), "%.17g", v.decimal_);
                out += buf;
            } else {
                out += v.ToString();
            }
            out += " | ";
        }
        return out;
    }
};

std::unique_ptr<Database> VectorizedTpchParityTest::row_db_;
std::unique_ptr<Database> VectorizedTpchParityTest::columnar_db_;
std::string VectorizedTpchParityTest::row_file_;
std::string VectorizedTpchParityTest::columnar_file_;
std::string VectorizedTpchParityTest::data_dir_;

// Row, Columnar or Columnar_Parallel; the row store never runs in parallel mode.
static std::string NameOf(const StorageMode& p) {
    const std::string storage = p.storage == StorageKind::ROW ? "Row" : "Columnar";
    return p.mode == ExecutionMode::PARALLEL ? storage + "_Parallel" : storage;
}

// Names the pair in test output, so ctest shows /Row, /Columnar and /Columnar_Parallel.
static void PrintTo(const StorageMode& p, std::ostream* os) {
    *os << NameOf(p);
}

INSTANTIATE_TEST_SUITE_P(Storage, VectorizedTpchParityTest,
                         ::testing::Values(StorageMode{StorageKind::ROW, ExecutionMode::VECTORIZED},
                                           StorageMode{StorageKind::COLUMNAR, ExecutionMode::VECTORIZED},
                                           StorageMode{StorageKind::COLUMNAR, ExecutionMode::PARALLEL}),
                         [](const ::testing::TestParamInfo<StorageMode>& info) { return NameOf(info.param); });

// ── Q1: Pricing Summary ─────────────────────────────────────────────

TEST_P(VectorizedTpchParityTest, Q1Parity) {
    RunParity(
        "SELECT l_returnflag, l_linestatus, SUM(l_quantity), SUM(l_extendedprice), "
        "SUM(l_discount), COUNT(*) "
        "FROM lineitem "
        "WHERE l_shipdate <= '1998-09-02' "
        "GROUP BY l_returnflag, l_linestatus "
        "ORDER BY l_returnflag, l_linestatus");
}

// ── Q3: Shipping Priority ───────────────────────────────────────────

TEST_P(VectorizedTpchParityTest, Q3Parity) {
    RunParity(
        "SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)), o_orderdate, o_shippriority "
        "FROM customer c "
        "JOIN orders o ON c.c_custkey = o.o_custkey "
        "JOIN lineitem l ON l.l_orderkey = o.o_orderkey "
        "WHERE c.c_mktsegment = 'BUILDING' "
        "AND o.o_orderdate < '1995-03-15' "
        "AND l.l_shipdate > '1995-03-15' "
        "GROUP BY l_orderkey, o_orderdate, o_shippriority "
        "ORDER BY SUM(l_extendedprice * (1 - l_discount)) DESC "
        "LIMIT 10");
}

// ── Q5: Local Supplier Volume ───────────────────────────────────────

TEST_P(VectorizedTpchParityTest, Q5Parity) {
    RunParity(
        "SELECT n_name, SUM(l_extendedprice * (1 - l_discount)) "
        "FROM customer c "
        "JOIN orders o ON c.c_custkey = o.o_custkey "
        "JOIN lineitem l ON l.l_orderkey = o.o_orderkey "
        "JOIN supplier s ON l.l_suppkey = s.s_suppkey "
        "JOIN nation n ON s.s_nationkey = n.n_nationkey "
        "WHERE o.o_orderdate >= '1994-01-01' AND o.o_orderdate < '1995-01-01' "
        "GROUP BY n_name "
        "ORDER BY SUM(l_extendedprice * (1 - l_discount)) DESC");
}

// ── Q6: Revenue Forecast ────────────────────────────────────────────

TEST_P(VectorizedTpchParityTest, Q6Parity) {
    RunParity(
        "SELECT SUM(l_extendedprice * l_discount) "
        "FROM lineitem "
        "WHERE l_shipdate >= '1994-01-01' AND l_shipdate < '1995-01-01' "
        "AND l_discount >= 0.05 AND l_discount <= 0.07 AND l_quantity < 24");
}

// ── Q10: Returned Item Reporting ────────────────────────────────────

TEST_P(VectorizedTpchParityTest, Q10Parity) {
    RunParity(
        "SELECT c_custkey, c_name, SUM(l_extendedprice * (1 - l_discount)), "
        "c_acctbal, n_name, c_address, c_phone, c_comment "
        "FROM customer, orders, lineitem, nation "
        "WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey "
        "AND o_orderdate >= '1993-10-01' AND o_orderdate < '1994-01-01' "
        "AND l_returnflag = 'R' AND c_nationkey = n_nationkey "
        "GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment "
        "ORDER BY SUM(l_extendedprice * (1 - l_discount)) DESC "
        "LIMIT 20");
}

// ── Q12: Shipping Modes and Order Priority ──────────────────────────

TEST_P(VectorizedTpchParityTest, Q12Parity) {
    RunParity(
        "SELECT l_shipmode, "
        "SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) "
        "FROM orders, lineitem "
        "WHERE o_orderkey = l_orderkey "
        "AND l_shipmode IN ('MAIL', 'SHIP') "
        "AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate "
        "AND l_receiptdate >= '1994-01-01' AND l_receiptdate < '1995-01-01' "
        "GROUP BY l_shipmode "
        "ORDER BY l_shipmode");
}

// ── Q14: Promotion Effect ───────────────────────────────────────────

TEST_P(VectorizedTpchParityTest, Q14Parity) {
    RunParity(
        "SELECT 100.00 * SUM(CASE WHEN p_type LIKE 'PROMO%' "
        "THEN l_extendedprice * (1 - l_discount) ELSE 0 END) "
        "/ SUM(l_extendedprice * (1 - l_discount)) "
        "FROM lineitem, part "
        "WHERE l_partkey = p_partkey "
        "AND l_shipdate >= '1995-09-01' AND l_shipdate < '1995-10-01'");
}

// ── Q19: Discounted Revenue ─────────────────────────────────────────

TEST_P(VectorizedTpchParityTest, Q19Parity) {
    RunParity(
        "SELECT SUM(l_extendedprice * (1 - l_discount)) "
        "FROM lineitem, part "
        "WHERE p_partkey = l_partkey "
        "AND ((p_brand = 'Brand#12' "
        "AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG') "
        "AND l_quantity >= 1 AND l_quantity <= 11 "
        "AND p_size BETWEEN 1 AND 5 "
        "AND l_shipmode IN ('AIR', 'AIR REG') "
        "AND l_shipinstruct = 'DELIVER IN PERSON') "
        "OR (p_brand = 'Brand#23' "
        "AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK') "
        "AND l_quantity >= 10 AND l_quantity <= 20 "
        "AND p_size BETWEEN 1 AND 10 "
        "AND l_shipmode IN ('AIR', 'AIR REG') "
        "AND l_shipinstruct = 'DELIVER IN PERSON') "
        "OR (p_brand = 'Brand#34' "
        "AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG') "
        "AND l_quantity >= 20 AND l_quantity <= 30 "
        "AND p_size BETWEEN 1 AND 15 "
        "AND l_shipmode IN ('AIR', 'AIR REG') "
        "AND l_shipinstruct = 'DELIVER IN PERSON'))");
}

}  // namespace shilmandb
