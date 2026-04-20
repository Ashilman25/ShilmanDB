#include "engine/database.hpp"
#include "executor/executor.hpp"
#include "common/exception.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>

using namespace shilmandb;
namespace fs = std::filesystem;


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
        {"part", PartSchema(), "part.tbl"},
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
        {"lineitem", "idx_lineitem_shipdate", "l_shipdate"},
        {"orders",   "idx_orders_orderdate",  "o_orderdate"},
        {"part",     "idx_part_partkey",      "p_partkey"},
    };
}

// ── Argument parsing ─────────────────────────────────────────────────

struct Args {
    std::string sf;
    std::string db_file;
    std::string data_dir;
    bool verify{true};
    std::string trace_path;
    std::string results_dir;
    size_t pool_size{4096};
    ExecutionMode exec_mode{ExecutionMode::TUPLE};
#ifdef SHILMANDB_HAS_LIBTORCH
    bool use_learned_join{false};
    std::string join_model_path;
    bool use_learned_eviction{false};
    std::string eviction_model_path;
    std::string eviction_version{"v2"};
#endif
};

static void PrintUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " --sf <scale_factor> --db-file <path> --data-dir <path> [--no-verify]"
              << " [--enable-tracing <path>] [--pool-size <N>] [--results-dir <path>]"
              << " [--execution-mode <tuple|vectorized>]"
#ifdef SHILMANDB_HAS_LIBTORCH
              << " [--use-learned-join --join-model-path <path>]"
              << " [--use-learned-eviction --eviction-model-path <path>]"
              << " [--eviction-model-version <v1|v2>]"
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
        } else if (arg == "--enable-tracing" && i + 1 < argc) {
            args.trace_path = argv[++i];
        } else if (arg == "--pool-size" && i + 1 < argc) {
            args.pool_size = std::stoul(argv[++i]);
        } else if (arg == "--results-dir" && i + 1 < argc) {
            args.results_dir = argv[++i];
        } else if (arg == "--execution-mode" && i + 1 < argc) {
            std::string mode_str = argv[++i];
            if (mode_str == "vectorized") {
                args.exec_mode = ExecutionMode::VECTORIZED;
            } else if (mode_str == "tuple") {
                args.exec_mode = ExecutionMode::TUPLE;
            } else {
                std::cerr << "Error: unknown --execution-mode '" << mode_str
                          << "' (expected 'tuple' or 'vectorized')\n";
                std::exit(1);
            }
#ifdef SHILMANDB_HAS_LIBTORCH
        } else if (arg == "--use-learned-join") {
            args.use_learned_join = true;
        } else if (arg == "--join-model-path" && i + 1 < argc) {
            args.join_model_path = argv[++i];
        } else if (arg == "--use-learned-eviction") {
            args.use_learned_eviction = true;
        } else if (arg == "--eviction-model-path" && i + 1 < argc) {
            args.eviction_model_path = argv[++i];
        } else if (arg == "--eviction-model-version" && i + 1 < argc) {
            args.eviction_version = argv[++i];
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
    if (args.use_learned_eviction && args.eviction_model_path.empty()) {
        std::cerr << "Error: --use-learned-eviction requires --eviction-model-path\n";
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
    std::cout << "Execution mode: "
              << (args.exec_mode == ExecutionMode::VECTORIZED ? "VECTORIZED" : "TUPLE")
              << "\n";

#ifdef SHILMANDB_HAS_LIBTORCH
    Database db(args.db_file, args.pool_size,
                args.use_learned_join, args.join_model_path,
                args.use_learned_eviction, args.eviction_model_path,
                args.eviction_version);
    if (args.use_learned_join) {
        std::cout << "Learned join optimizer: ON (model: " << args.join_model_path << ")\n";
    }
    if (args.use_learned_eviction) {
        std::cout << "Learned eviction policy: ON (" << args.eviction_version << ", model: " << args.eviction_model_path << ")\n";
    }
#else
    Database db(args.db_file, args.pool_size);
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



    bool run_queries = args.verify || !args.trace_path.empty() || !args.results_dir.empty();

    if (!args.trace_path.empty()) {
        if (!args.verify) {
            std::cout << "\nTracing enabled — running verification queries for trace generation\n";
        }
        db.GetBufferPoolManager()->EnableTracing(args.trace_path);
    }

    if (run_queries) {
        std::cout << "\n=== Row Counts ===\n";
        std::vector<std::string> verify_tables = {
            "region", "nation", "part", "supplier", "customer", "orders", "lineitem"
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
            std::string tag;   // short identifier: "Q1", "Q3", etc.
            std::string sql;
            std::string canonical_header;
        };

        // Canonical headers match bench/run_sqlite_benchmarks.py QUERY_HEADERS
        // (hardcoded rather than using planner-derived schema names for stable cross-engine comparison)
        std::vector<TpchQuery> queries = {
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

        // Create results directory if requested
        const bool export_csv = !args.results_dir.empty();
        if (export_csv) {
            std::error_code ec;
            fs::create_directories(args.results_dir, ec);
            if (ec) {
                std::cerr << "Error: could not create results directory '"
                          << args.results_dir << "': " << ec.message() << "\n";
                return 1;
            }
        }

        // CSV value formatter: RFC 4180 quoting for VARCHAR fields
        auto csv_value = [](const Value& v, const Schema& schema, uint32_t col) -> std::string {
            auto str = v.ToString();
            if (schema.GetColumn(col).type == TypeId::VARCHAR) {
                bool needs_quoting = str.find(',')  != std::string::npos
                                  || str.find('"')  != std::string::npos
                                  || str.find('\n') != std::string::npos;
                if (needs_quoting) {
                    std::string escaped;
                    escaped.reserve(str.size() + 2);
                    for (char ch : str) {
                        if (ch == '"') escaped += '"';
                        escaped += ch;
                    }
                    return "\"" + escaped + "\"";
                }
            }
            return str;
        };

        // Latency records for latencies.csv
        struct LatencyRecord {
            std::string tag;
            size_t rows;
            double latency_ms;
        };
        std::vector<LatencyRecord> latencies;

        int passed = 0;
        for (const auto& q : queries) {
            std::cout << "\n  " << q.name << " ... " << std::flush;
            auto start = std::chrono::steady_clock::now();
            try {
                auto result = db.ExecuteSQL(q.sql, args.exec_mode);
                auto elapsed = std::chrono::steady_clock::now() - start;
                double ms = std::chrono::duration<double, std::milli>(elapsed).count();

                std::cout << "OK — " << result.tuples.size() << " rows (" << static_cast<int64_t>(ms) << " ms)\n";

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

                // Export query results to CSV
                if (export_csv) {
                    auto csv_path = fs::path(args.results_dir) / (q.tag + ".csv");
                    std::ofstream ofs(csv_path);
                    if (ofs) {
                        ofs << q.canonical_header << "\n";
                        for (const auto& tuple : result.tuples) {
                            for (uint32_t c = 0; c < schema.GetColumnCount(); ++c) {
                                if (c > 0) ofs << ",";
                                ofs << csv_value(tuple.GetValue(schema, c), schema, c);
                            }
                            ofs << "\n";
                        }
                        std::cout << "    -> Exported to " << csv_path.string() << "\n";
                    } else {
                        std::cerr << "    Warning: could not open " << csv_path.string() << " for writing\n";
                    }
                }

                latencies.push_back({q.tag, result.tuples.size(), ms});
                ++passed;
            } catch (const std::exception& e) {
                auto elapsed = std::chrono::steady_clock::now() - start;
                double ms = std::chrono::duration<double, std::milli>(elapsed).count();
                std::cerr << "FAILED (" << static_cast<int64_t>(ms) << " ms) — " << e.what() << "\n";
            }
        }

        std::cout << "\n=== " << passed << "/" << queries.size() << " TPC-H queries passed ===\n";

        // Write latencies.csv
        if (export_csv && !latencies.empty()) {
            auto lat_path = fs::path(args.results_dir) / "latencies.csv";
            std::ofstream ofs(lat_path);
            if (ofs) {
                ofs << "query,rows,latency_ms\n";
                ofs << std::fixed << std::setprecision(4);
                for (const auto& rec : latencies) {
                    ofs << rec.tag << "," << rec.rows << "," << rec.latency_ms << "\n";
                }
                std::cout << "\nLatencies written to " << lat_path.string() << "\n";
            } else {
                std::cerr << "Warning: could not open " << lat_path.string() << " for writing\n";
            }
        }
    }

    if (!args.trace_path.empty()) {
        db.GetBufferPoolManager()->DisableTracing();
        std::cout << "\nTrace written to " << args.trace_path << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
