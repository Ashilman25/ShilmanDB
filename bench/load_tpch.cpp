#include "engine/database.hpp"
#include "common/exception.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <chrono>

using namespace shilmandb;


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

static Schema NationSchema() {
    return Schema({
        {"n_nationkey", TypeId::BIGINT},
        {"n_name", TypeId::VARCHAR},
        {"n_regionkey", TypeId::BIGINT},
        {"n_comment", TypeId::VARCHAR},
    });
}

static Schema RegionSchema() {
    return Schema({
        {"r_regionkey", TypeId::BIGINT},
        {"r_name", TypeId::VARCHAR},
        {"r_comment", TypeId::VARCHAR},
    });
}

// ── Table descriptor ─────────────────────────────────────────────────

struct TableDesc {
    std::string name;
    Schema schema;
    std::string tbl_file;  // e.g. "lineitem.tbl"
};

static std::vector<TableDesc> BuildTableDescs() {
    return {
        {"region", RegionSchema(), "region.tbl"},
        {"nation", NationSchema(), "nation.tbl"},
        {"supplier", SupplierSchema(), "supplier.tbl"},
        {"customer", CustomerSchema(), "customer.tbl"},
        {"orders", OrdersSchema(), "orders.tbl"},
        {"lineitem", LineitemSchema(), "lineitem.tbl"},
    };
}

// ── Index descriptors (table, index_name, column) ────────────────────

struct IndexDesc {
    std::string table_name;
    std::string index_name;
    std::string column_name;
};

static std::vector<IndexDesc> BuildIndexDescs() {
    return {
        {"orders",   "idx_orders_custkey",    "o_custkey"},
        {"lineitem", "idx_lineitem_orderkey", "l_orderkey"},
        {"customer", "idx_customer_custkey",  "c_custkey"},
        {"supplier", "idx_supplier_nationkey","s_nationkey"},
        {"nation",   "idx_nation_nationkey",  "n_nationkey"},
    };
}

// ── Argument parsing ─────────────────────────────────────────────────

struct Args {
    std::string sf;
    std::string db_file;
    std::string data_dir;
    bool verify{true};
#ifdef SHILMANDB_HAS_LIBTORCH
    bool use_learned_join{false};
    std::string join_model_path;
#endif
};

static void PrintUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " --sf <scale_factor> --db-file <path> --data-dir <path> [--no-verify]"
#ifdef SHILMANDB_HAS_LIBTORCH
              << " [--use-learned-join --join-model-path <path>]"
#endif
              << "\n";
}

static Args ParseArgs(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--sf" && i + 1 < argc) {
            args.sf = argv[++i];
        } else if (arg == "--db-file" && i + 1 < argc) {
            args.db_file = argv[++i];
        } else if (arg == "--data-dir" && i + 1 < argc) {
            args.data_dir = argv[++i];
        } else if (arg == "--no-verify") {
            args.verify = false;
#ifdef SHILMANDB_HAS_LIBTORCH
        } else if (arg == "--use-learned-join") {
            args.use_learned_join = true;
        } else if (arg == "--join-model-path" && i + 1 < argc) {
            args.join_model_path = argv[++i];
#endif
        } else {
            PrintUsage(argv[0]);
            std::exit(1);
        }
    }
    if (args.sf.empty() || args.db_file.empty() || args.data_dir.empty()) {
        PrintUsage(argv[0]);
        std::exit(1);
    }
#ifdef SHILMANDB_HAS_LIBTORCH
    if (args.use_learned_join && args.join_model_path.empty()) {
        std::cerr << "Error: --use-learned-join requires --join-model-path\n";
        std::exit(1);
    }
#endif
    // Ensure data_dir ends with '/'
    if (args.data_dir.back() != '/') {
        args.data_dir += '/';
    }
    return args;
}

// ── Main ─────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    auto args = ParseArgs(argc, argv);

    std::cout << "=== ShilmanDB TPC-H Loader (SF=" << args.sf << ") ===\n";

    constexpr size_t kLoadBufferPoolSize = 4096;
#ifdef SHILMANDB_HAS_LIBTORCH
    Database db(args.db_file, kLoadBufferPoolSize,
                args.use_learned_join, args.join_model_path);
    if (args.use_learned_join) {
        std::cout << "Learned join optimizer: ON (model: " << args.join_model_path << ")\n";
    }
#else
    Database db(args.db_file, kLoadBufferPoolSize);
#endif


    auto tables = BuildTableDescs();
    for (const auto& td : tables) {
        std::string path = args.data_dir + td.tbl_file;
        std::cout << "Loading " << td.name << " from " << path << " ... " << std::flush;

        auto start = std::chrono::steady_clock::now();
        try {
            db.LoadTable(td.name, td.schema, path);
        } catch (const std::exception& e) {
            std::cerr << "\nError loading " << td.name << ": " << e.what() << "\n";
            return 1;
        }
        auto elapsed = std::chrono::steady_clock::now() - start;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

        auto* info = db.GetCatalog()->GetTable(td.name);
        std::cout << info->stats.row_count << " rows (" << ms << " ms)\n";
    }



    std::cout << "\nCreating indexes ...\n";
    auto indexes = BuildIndexDescs();
    for (const auto& idx : indexes) {
        std::cout << "  " << idx.index_name << " on " << idx.table_name << "(" << idx.column_name << ") ... " << std::flush;
        auto start = std::chrono::steady_clock::now();
        auto* index_info = db.GetCatalog()->CreateIndex(idx.index_name, idx.table_name, idx.column_name);
        auto elapsed = std::chrono::steady_clock::now() - start;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

        if (index_info) {
            std::cout << "done (" << ms << " ms)\n";
        } else {
            std::cout << "FAILED\n";
        }
    }



    if (args.verify) {
        std::cout << "\n=== Row Counts ===\n";
        std::vector<std::string> verify_tables = {
            "region", "nation", "supplier", "customer", "orders", "lineitem"
        };
        for (const auto& name : verify_tables) {
            std::string sql = "SELECT COUNT(*) FROM " + name;
            try {
                auto result = db.ExecuteSQL(sql);
                if (!result.tuples.empty()) {
                    std::cout << "  " << name << ": " << result.tuples[0].GetValue(result.schema, 0).ToString() << " rows\n";
                }
            } catch (const std::exception& e) {
                std::cerr << "  " << name << ": verification failed — " << e.what() << "\n";
            }
        }

        // ── TPC-H target queries ─────────────────────────────────────

        std::cout << "\n=== TPC-H Query Verification ===\n";

        struct TpchQuery {
            std::string name;
            std::string sql;
        };

        std::vector<TpchQuery> queries = {
            {"Q1 (Pricing Summary)",
             "SELECT l_returnflag, l_linestatus, SUM(l_quantity), SUM(l_extendedprice), "
             "SUM(l_discount), COUNT(*) "
             "FROM lineitem "
             "WHERE l_shipdate <= '1998-09-02' "
             "GROUP BY l_returnflag, l_linestatus "
             "ORDER BY l_returnflag, l_linestatus"},

            {"Q6 (Revenue Forecast)",
             "SELECT SUM(l_extendedprice * l_discount) "
             "FROM lineitem "
             "WHERE l_shipdate >= '1994-01-01' AND l_shipdate < '1995-01-01' "
             "AND l_discount >= 0.05 AND l_discount <= 0.07 AND l_quantity < 24"},

            {"Q3 (Shipping Priority)",
             "SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)), o_orderdate, o_shippriority "
             "FROM customer c "
             "JOIN orders o ON c.c_custkey = o.o_custkey "
             "JOIN lineitem l ON l.l_orderkey = o.o_orderkey "
             "WHERE c.c_mktsegment = 'BUILDING' "
             "AND o.o_orderdate < '1995-03-15' "
             "AND l.l_shipdate > '1995-03-15' "
             "GROUP BY l_orderkey, o_orderdate, o_shippriority "
             "ORDER BY SUM(l_extendedprice * (1 - l_discount)) DESC "
             "LIMIT 10"},

            {"Q5 (Local Supplier Volume)",
             "SELECT n_name, SUM(l_extendedprice * (1 - l_discount)) "
             "FROM customer c "
             "JOIN orders o ON c.c_custkey = o.o_custkey "
             "JOIN lineitem l ON l.l_orderkey = o.o_orderkey "
             "JOIN supplier s ON l.l_suppkey = s.s_suppkey "
             "JOIN nation n ON s.s_nationkey = n.n_nationkey "
             "WHERE o.o_orderdate >= '1994-01-01' AND o.o_orderdate < '1995-01-01' "
             "GROUP BY n_name "
             "ORDER BY SUM(l_extendedprice * (1 - l_discount)) DESC"},
        };

        int passed = 0;
        for (const auto& q : queries) {
            std::cout << "\n  " << q.name << " ... " << std::flush;
            auto start = std::chrono::steady_clock::now();
            try {
                auto result = db.ExecuteSQL(q.sql);
                auto elapsed = std::chrono::steady_clock::now() - start;
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

                std::cout << "OK — " << result.tuples.size() << " rows (" << ms << " ms)\n";

                // Print first few rows for inspection
                const auto& schema = result.schema;
                std::cout << "    Columns: ";
                for (uint32_t c = 0; c < schema.GetColumnCount(); ++c) {
                    if (c > 0) std::cout << " | ";
                    std::cout << schema.GetColumn(c).name;
                }
                std::cout << "\n";

                size_t show = std::min(result.tuples.size(), static_cast<size_t>(5));
                for (size_t r = 0; r < show; ++r) {
                    std::cout << "    ";
                    for (uint32_t c = 0; c < schema.GetColumnCount(); ++c) {
                        if (c > 0) std::cout << " | ";
                        std::cout << result.tuples[r].GetValue(schema, c).ToString();
                    }
                    std::cout << "\n";
                }
                if (result.tuples.size() > show) {
                    std::cout << "    ... (" << result.tuples.size() - show << " more rows)\n";
                }
                ++passed;
            } catch (const std::exception& e) {
                auto elapsed = std::chrono::steady_clock::now() - start;
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
                std::cerr << "FAILED (" << ms << " ms) — " << e.what() << "\n";
            }
        }

        std::cout << "\n=== " << passed << "/" << queries.size() << " TPC-H queries passed ===\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
