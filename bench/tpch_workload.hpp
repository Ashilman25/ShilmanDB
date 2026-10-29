#pragma once

#include "engine/database.hpp"
#include "common/exception.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

// The TPC-H tables, indexes and queries every bench tool and test runs.
namespace shilmandb::tpch {

// Schemas

[[nodiscard]] inline Schema LineitemSchema() {
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

[[nodiscard]] inline Schema OrdersSchema() {
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

[[nodiscard]] inline Schema CustomerSchema() {
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

[[nodiscard]] inline Schema SupplierSchema() {
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

[[nodiscard]] inline Schema NationSchema() {
    return Schema({
        {"n_nationkey", TypeId::BIGINT},
        {"n_name", TypeId::VARCHAR},
        {"n_regionkey", TypeId::BIGINT},
        {"n_comment", TypeId::VARCHAR},
    });
}

[[nodiscard]] inline Schema RegionSchema() {
    return Schema({
        {"r_regionkey", TypeId::BIGINT},
        {"r_name", TypeId::VARCHAR},
        {"r_comment", TypeId::VARCHAR},
    });
}

[[nodiscard]] inline Schema PartSchema() {
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

// Tables

struct TableDesc {
    std::string name;
    Schema schema;
    std::string tbl_file;  // e.g. "lineitem.tbl"
};

// In load order.
[[nodiscard]] inline auto TableDescs() -> std::vector<TableDesc> {
    return {
        {"region", RegionSchema(), "region.tbl"},
        {"nation", NationSchema(), "nation.tbl"},
        {"part", PartSchema(), "part.tbl"},
        {"supplier", SupplierSchema(), "supplier.tbl"},
        {"customer", CustomerSchema(), "customer.tbl"},
        {"orders", OrdersSchema(), "orders.tbl"},
        {"lineitem", LineitemSchema(), "lineitem.tbl"},
    };
}

//  Indexes (table, index_name, column)

struct IndexDesc {
    std::string table_name;
    std::string index_name;
    std::string column_name;
};

[[nodiscard]] inline auto IndexDescs() -> std::vector<IndexDesc> {
    return {
        {"orders", "idx_orders_custkey", "o_custkey"},
        {"lineitem", "idx_lineitem_orderkey", "l_orderkey"},
        {"customer", "idx_customer_custkey", "c_custkey"},
        {"supplier", "idx_supplier_nationkey", "s_nationkey"},
        {"nation", "idx_nation_nationkey", "n_nationkey"},
        {"lineitem", "idx_lineitem_shipdate", "l_shipdate"},
        {"orders", "idx_orders_orderdate", "o_orderdate"},
        {"part", "idx_part_partkey", "p_partkey"},
    };
}

//  Queries 

struct Query {
    std::string name;
    std::string tag;   // short identifier: "Q1", "Q3", etc.
    std::string sql;
    std::string canonical_header;
};

// Canonical headers match bench/run_sqlite_benchmarks.py QUERY_HEADERS
// (hardcoded rather than using planner-derived schema names for stable cross-engine comparison)
[[nodiscard]] inline auto Queries() -> std::vector<Query> {
    return {
        {"Q1 (Pricing Summary)", "Q1",
         "SELECT l_returnflag, l_linestatus, SUM(l_quantity), SUM(l_extendedprice), "
         "SUM(l_discount), COUNT(*) "
         "FROM lineitem "
         "WHERE l_shipdate <= '1998-09-02' "
         "GROUP BY l_returnflag, l_linestatus "
         "ORDER BY l_returnflag, l_linestatus",
         "l_returnflag,l_linestatus,sum_quantity,sum_extendedprice,sum_discount,count_star"},

        {"Q6 (Revenue Forecast)", "Q6",
         "SELECT SUM(l_extendedprice * l_discount) "
         "FROM lineitem "
         "WHERE l_shipdate >= '1994-01-01' AND l_shipdate < '1995-01-01' "
         "AND l_discount >= 0.05 AND l_discount <= 0.07 AND l_quantity < 24",
         "sum_disc_price"},

        {"Q3 (Shipping Priority)", "Q3",
         "SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)), o_orderdate, o_shippriority "
         "FROM customer c "
         "JOIN orders o ON c.c_custkey = o.o_custkey "
         "JOIN lineitem l ON l.l_orderkey = o.o_orderkey "
         "WHERE c.c_mktsegment = 'BUILDING' "
         "AND o.o_orderdate < '1995-03-15' "
         "AND l.l_shipdate > '1995-03-15' "
         "GROUP BY l_orderkey, o_orderdate, o_shippriority "
         "ORDER BY SUM(l_extendedprice * (1 - l_discount)) DESC "
         "LIMIT 10",
         "l_orderkey,sum_disc_price,o_orderdate,o_shippriority"},

        {"Q5 (Local Supplier Volume)", "Q5",
         "SELECT n_name, SUM(l_extendedprice * (1 - l_discount)) "
         "FROM customer c "
         "JOIN orders o ON c.c_custkey = o.o_custkey "
         "JOIN lineitem l ON l.l_orderkey = o.o_orderkey "
         "JOIN supplier s ON l.l_suppkey = s.s_suppkey "
         "JOIN nation n ON s.s_nationkey = n.n_nationkey "
         "WHERE o.o_orderdate >= '1994-01-01' AND o.o_orderdate < '1995-01-01' "
         "GROUP BY n_name "
         "ORDER BY SUM(l_extendedprice * (1 - l_discount)) DESC",
         "n_name,sum_disc_price"},

        {"Q10 (Returned Item Reporting)", "Q10",
         "SELECT c_custkey, c_name, SUM(l_extendedprice * (1 - l_discount)), "
         "c_acctbal, n_name, c_address, c_phone, c_comment "
         "FROM customer, orders, lineitem, nation "
         "WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey "
         "AND o_orderdate >= '1993-10-01' AND o_orderdate < '1994-01-01' "
         "AND l_returnflag = 'R' AND c_nationkey = n_nationkey "
         "GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment "
         "ORDER BY SUM(l_extendedprice * (1 - l_discount)) DESC "
         "LIMIT 20",
         "c_custkey,c_name,sum_disc_price,c_acctbal,n_name,c_address,c_phone,c_comment"},

        {"Q12 (Shipping Modes and Order Priority)", "Q12",
         "SELECT l_shipmode, "
         "SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END), "
         "SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) "
         "FROM orders, lineitem "
         "WHERE o_orderkey = l_orderkey "
         "AND l_shipmode IN ('MAIL', 'SHIP') "
         "AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate "
         "AND l_receiptdate >= '1994-01-01' AND l_receiptdate < '1995-01-01' "
         "GROUP BY l_shipmode "
         "ORDER BY l_shipmode",
         "l_shipmode,high_line_count,low_line_count"},

        {"Q14 (Promotion Effect)", "Q14",
         "SELECT 100.00 * SUM(CASE WHEN p_type LIKE 'PROMO%' "
         "THEN l_extendedprice * (1 - l_discount) ELSE 0 END) "
         "/ SUM(l_extendedprice * (1 - l_discount)) "
         "FROM lineitem, part "
         "WHERE l_partkey = p_partkey "
         "AND l_shipdate >= '1995-09-01' AND l_shipdate < '1995-10-01'",
         "promo_revenue"},

        {"Q19 (Discounted Revenue)", "Q19",
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
         "AND l_shipinstruct = 'DELIVER IN PERSON'))",
         "revenue"},
    };
}

// The query with this tag ("Q5"), or nothing if there isn't one.
[[nodiscard]] inline auto FindQuery(const std::string& tag) -> std::optional<Query> {
    auto queries = Queries();
    auto it = std::find_if(queries.begin(), queries.end(), [&](const Query& q) { return q.tag == tag; });
    if (it == queries.end()) return std::nullopt;
    return std::move(*it);
}

//  Loading and comparing

// Loads all 7 tables from data_dir (which must end with '/'), printing one line per table.
inline void LoadTables(Database& db, const std::string& data_dir) {
    for (const auto& td : TableDescs()) {
        std::string path = data_dir + td.tbl_file;
        std::cout << "Loading " << td.name << " from " << path << " ... " << std::flush;

        auto start = std::chrono::steady_clock::now();
        try {
            db.LoadTable(td.name, td.schema, path);
        } catch (const std::exception& e) {
            throw DatabaseException("Error loading " + td.name + ": " + e.what());
        }
        auto elapsed = std::chrono::steady_clock::now() - start;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

        auto* info = db.GetCatalog()->GetTable(td.name);
        std::cout << info->stats.row_count << " rows (" << ms << " ms)\n";
    }
}

// True when both results have the same rows in the same order
[[nodiscard]] inline auto SameResult(const QueryResult& a, const QueryResult& b, double rel_tol = 1e-6) -> bool {
    if (a.tuples.size() != b.tuples.size()) return false;
    const uint32_t cols = a.schema.GetColumnCount();
    if (cols != b.schema.GetColumnCount()) return false;

    for (size_t r = 0; r < a.tuples.size(); ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            const Value x = a.tuples[r].GetValue(a.schema, c);
            const Value y = b.tuples[r].GetValue(b.schema, c);

            const bool same = (x.type_ == TypeId::DECIMAL && y.type_ == TypeId::DECIMAL)
                ? std::abs(x.decimal_ - y.decimal_) <= rel_tol * std::max({1.0, std::abs(x.decimal_), std::abs(y.decimal_)})
                : x.ToString() == y.ToString();
            if (!same) return false;
        }
    }
    return true;
}

}  // namespace shilmandb::tpch
