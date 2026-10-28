#include <gtest/gtest.h>
#include <unistd.h>

#include "engine/database.hpp"
#include "executor/vectorized_parity_harness.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace shilmandb {

// ── TPC-H schema helpers (mirrors bench/load_tpch.cpp) ──────────────

static Schema RegionSchema() {
    return Schema({
        {"r_regionkey", TypeId::BIGINT},
        {"r_name", TypeId::VARCHAR},
        {"r_comment", TypeId::VARCHAR},
    });
}

static Schema NationSchema() {
    return Schema({
        {"n_nationkey", TypeId::BIGINT},
        {"n_name", TypeId::VARCHAR},
        {"n_regionkey", TypeId::BIGINT},
        {"n_comment", TypeId::VARCHAR},
    });
}

static Schema PartSchema() {
    return Schema({
        {"p_partkey", TypeId::BIGINT},
        {"p_name", TypeId::VARCHAR},
        {"p_mfgr", TypeId::VARCHAR},
        {"p_brand", TypeId::VARCHAR},
        {"p_type", TypeId::VARCHAR},
        {"p_size", TypeId::INTEGER},
        {"p_container", TypeId::VARCHAR},
        {"p_retailprice", TypeId::DECIMAL},
        {"p_comment", TypeId::VARCHAR},
    });
}

static Schema SupplierSchema() {
    return Schema({
        {"s_suppkey", TypeId::BIGINT},
        {"s_name", TypeId::VARCHAR},
        {"s_address", TypeId::VARCHAR},
        {"s_nationkey", TypeId::BIGINT},
        {"s_phone", TypeId::VARCHAR},
        {"s_acctbal", TypeId::DECIMAL},
        {"s_comment", TypeId::VARCHAR},
    });
}

static Schema CustomerSchema() {
    return Schema({
        {"c_custkey", TypeId::BIGINT},
        {"c_name", TypeId::VARCHAR},
        {"c_address", TypeId::VARCHAR},
        {"c_nationkey", TypeId::BIGINT},
        {"c_phone", TypeId::VARCHAR},
        {"c_acctbal", TypeId::DECIMAL},
        {"c_mktsegment", TypeId::VARCHAR},
        {"c_comment", TypeId::VARCHAR},
    });
}

static Schema OrdersSchema() {
    return Schema({
        {"o_orderkey", TypeId::BIGINT},
        {"o_custkey", TypeId::BIGINT},
        {"o_orderstatus", TypeId::VARCHAR},
        {"o_totalprice", TypeId::DECIMAL},
        {"o_orderdate", TypeId::DATE},
        {"o_orderpriority", TypeId::VARCHAR},
        {"o_clerk", TypeId::VARCHAR},
        {"o_shippriority", TypeId::INTEGER},
        {"o_comment", TypeId::VARCHAR},
    });
}

static Schema LineitemSchema() {
    return Schema({
        {"l_orderkey", TypeId::BIGINT},
        {"l_partkey", TypeId::BIGINT},
        {"l_suppkey", TypeId::BIGINT},
        {"l_linenumber", TypeId::INTEGER},
        {"l_quantity", TypeId::DECIMAL},
        {"l_extendedprice", TypeId::DECIMAL},
        {"l_discount", TypeId::DECIMAL},
        {"l_tax", TypeId::DECIMAL},
        {"l_returnflag", TypeId::VARCHAR},
        {"l_linestatus", TypeId::VARCHAR},
        {"l_shipdate", TypeId::DATE},
        {"l_commitdate", TypeId::DATE},
        {"l_receiptdate", TypeId::DATE},
        {"l_shipinstruct", TypeId::VARCHAR},
        {"l_shipmode", TypeId::VARCHAR},
        {"l_comment", TypeId::VARCHAR},
    });
}

struct TableDesc {
    std::string name;
    Schema schema;
    std::string tbl_file;
};

struct IndexDesc {
    std::string index_name;
    std::string table_name;
    std::string column_name;
};

static std::vector<TableDesc> BuildTableDescs() {
    return {
        {"region",   RegionSchema(),   "region.tbl"},
        {"nation",   NationSchema(),   "nation.tbl"},
        {"part",     PartSchema(),     "part.tbl"},
        {"supplier", SupplierSchema(), "supplier.tbl"},
        {"customer", CustomerSchema(), "customer.tbl"},
        {"orders",   OrdersSchema(),   "orders.tbl"},
        {"lineitem", LineitemSchema(), "lineitem.tbl"},
    };
}

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

// ── Fixture: load TPC-H SF=0.01 once for the whole suite ────────────

class VectorizedTpchParityTest : public ::testing::Test {
protected:
    static std::unique_ptr<Database> db_;
    static std::string test_file_;

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

    static void SetUpTestSuite() {
        test_file_ = (std::filesystem::temp_directory_path() /
                      ("shilmandb_vec_tpch_parity_test_" + std::to_string(::getpid()) + ".db")).string();
        std::filesystem::remove(test_file_);

        auto data_dir = FindDataDir();
        if (data_dir.empty()) {
            // Leave db_ null; every test skips via RunParity.
            return;
        }

        // Large pool so eviction does not interfere with correctness.
        constexpr size_t kPoolSize = 4096;
        db_ = std::make_unique<Database>(test_file_, kPoolSize);

        for (const auto& td : BuildTableDescs()) {
            db_->LoadTable(td.name, td.schema, data_dir + td.tbl_file);
        }
        auto* catalog = db_->GetCatalog();
        for (const auto& idx : BuildIndexDescs()) {
            (void)catalog->CreateIndex(idx.index_name, idx.table_name, idx.column_name);
        }
        for (const auto& td : BuildTableDescs()) {
            catalog->UpdateTableStats(td.name);
        }
    }

    static void TearDownTestSuite() {
        db_.reset();
        if (!test_file_.empty()) {
            std::filesystem::remove(test_file_);
        }
    }

    static void RunParity(const std::string& sql) {
        if (!db_) {
            GTEST_SKIP() << "TPC-H SF=0.01 data not found. "
                            "Run bench/generate_tpch_data.sh first.";
        }
        auto tuple_rs = db_->ExecuteSQL(sql, ExecutionMode::TUPLE);
        auto vec_rs   = db_->ExecuteSQL(sql, ExecutionMode::VECTORIZED);
        ASSERT_EQ(tuple_rs.tuples.size(), vec_rs.tuples.size())
            << "TPC-H parity: row-count mismatch";
        test::ExpectRowsEqual(tuple_rs.tuples, vec_rs.tuples, tuple_rs.schema);
    }
};

std::unique_ptr<Database> VectorizedTpchParityTest::db_;
std::string VectorizedTpchParityTest::test_file_;

// ── Q1: Pricing Summary ─────────────────────────────────────────────

TEST_F(VectorizedTpchParityTest, Q1Parity) {
    RunParity(
        "SELECT l_returnflag, l_linestatus, SUM(l_quantity), SUM(l_extendedprice), "
        "SUM(l_discount), COUNT(*) "
        "FROM lineitem "
        "WHERE l_shipdate <= '1998-09-02' "
        "GROUP BY l_returnflag, l_linestatus "
        "ORDER BY l_returnflag, l_linestatus");
}

// ── Q3: Shipping Priority ───────────────────────────────────────────

TEST_F(VectorizedTpchParityTest, Q3Parity) {
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

TEST_F(VectorizedTpchParityTest, Q5Parity) {
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

TEST_F(VectorizedTpchParityTest, Q6Parity) {
    RunParity(
        "SELECT SUM(l_extendedprice * l_discount) "
        "FROM lineitem "
        "WHERE l_shipdate >= '1994-01-01' AND l_shipdate < '1995-01-01' "
        "AND l_discount >= 0.05 AND l_discount <= 0.07 AND l_quantity < 24");
}

// ── Q10: Returned Item Reporting ────────────────────────────────────

TEST_F(VectorizedTpchParityTest, Q10Parity) {
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

TEST_F(VectorizedTpchParityTest, Q12Parity) {
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

TEST_F(VectorizedTpchParityTest, Q14Parity) {
    RunParity(
        "SELECT 100.00 * SUM(CASE WHEN p_type LIKE 'PROMO%' "
        "THEN l_extendedprice * (1 - l_discount) ELSE 0 END) "
        "/ SUM(l_extendedprice * (1 - l_discount)) "
        "FROM lineitem, part "
        "WHERE l_partkey = p_partkey "
        "AND l_shipdate >= '1995-09-01' AND l_shipdate < '1995-10-01'");
}

// ── Q19: Discounted Revenue ─────────────────────────────────────────

TEST_F(VectorizedTpchParityTest, Q19Parity) {
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
