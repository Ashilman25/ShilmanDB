#include "engine/database.hpp"
#include "executor/executor.hpp"
#include "common/exception.hpp"
#include "tpch_workload.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

using namespace shilmandb;
namespace fs = std::filesystem;

// ── Argument parsing ─────────────────────────────────────────────────

struct Args {
    std::string sf;
    std::string db_file;
    std::string data_dir;
    bool verify{true};
    std::string trace_path;
    std::string results_dir;
    size_t pool_size{4096};
    bool create_indexes{true};
    int warmup_runs{1};
    int timed_runs{5};
    ExecutionMode exec_mode{ExecutionMode::TUPLE};
    int64_t miss_penalty_us{0};
    bool direct_io{false};
    bool policy_timing{false};
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
              << " [--enable-tracing <path>] [--pool-size <N>] [--create-indexes <true|false>]"
              << " [--warmup <N>] [--runs <N>]"
              << " [--results-dir <path>]"
              << " [--execution-mode <tuple|vectorized>]"
              << " [--miss-penalty-us <N>] [--direct-io] [--policy-timing]"
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
        } else if (arg == "--create-indexes" && i + 1 < argc) {
            std::string val = argv[++i];
            if (val == "true" || val == "1") {
                args.create_indexes = true;
            } else if (val == "false" || val == "0") {
                args.create_indexes = false;
            } else {
                std::cerr << "Error: --create-indexes expects 'true' or 'false', got '" << val << "'\n";
                std::exit(1);
            }
        } else if (arg == "--warmup" && i + 1 < argc) {
            args.warmup_runs = std::stoi(argv[++i]);
            if (args.warmup_runs < 0) {
                std::cerr << "Error: --warmup must be >= 0\n";
                std::exit(1);
            }
        } else if (arg == "--runs" && i + 1 < argc) {
            args.timed_runs = std::stoi(argv[++i]);
            if (args.timed_runs < 1) {
                std::cerr << "Error: --runs must be >= 1\n";
                std::exit(1);
            }
        } else if (arg == "--results-dir" && i + 1 < argc) {
            args.results_dir = argv[++i];
        } else if (arg == "--execution-mode" && i + 1 < argc) {
            std::string mode_str = argv[++i];
            if (mode_str == "vectorized") {
                args.exec_mode = ExecutionMode::VECTORIZED;
            } else if (mode_str == "tuple") {
                args.exec_mode = ExecutionMode::TUPLE;
            } else {
                std::cerr << "Error: unknown --execution-mode '" << mode_str << "' (expected 'tuple' or 'vectorized')\n";
                std::exit(1);
            }
        } else if (arg == "--miss-penalty-us" && i + 1 < argc) {
            std::string val = argv[++i];
            try {
                size_t used = 0;
                args.miss_penalty_us = std::stoll(val, &used);
                if (used != val.size() || args.miss_penalty_us < 0) {
                    throw std::invalid_argument(val);
                }
            } catch (const std::exception&) {
                std::cerr << "Error: --miss-penalty-us expects a non-negative integer, got '" << val << "'\n";
                PrintUsage(argv[0]);
                std::exit(1);
            }
        } else if (arg == "--direct-io") {
            args.direct_io = true;
        } else if (arg == "--policy-timing") {
            args.policy_timing = true;
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

    if (args.direct_io) {
        try {
            db.GetDiskManager()->SetDirectIO(true);
        } catch (const std::exception& e) {
            std::cerr << "Error: --direct-io failed: " << e.what() << "\n";
            return 1;
        }
        std::cout << "Direct I/O: ON\n";
    }

    std::string eviction_name = "lru";
#ifdef SHILMANDB_HAS_LIBTORCH
    // Database runs V2 only for exactly "v2"; any other string gets V1
    if (args.use_learned_eviction) eviction_name = (args.eviction_version == "v2") ? "v2" : "v1";
#endif

    try {
        tpch::LoadTables(db, args.data_dir);
    } catch (const std::exception& e) {
        std::cerr << "\n" << e.what() << "\n";
        return 1;
    }
    const page_id_t db_pages = db.GetDiskManager()->GetNumPages();



    if (args.create_indexes) {
        std::cout << "\nCreating indexes ...\n";
        auto indexes = tpch::IndexDescs();
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
    } else {
        std::cout << "\nSkipping index creation (--create-indexes false)\n";
    }
    const page_id_t db_pages_total = db.GetDiskManager()->GetNumPages();

    // The penalty starts after the load, so only the queries pay it
    if (args.miss_penalty_us > 0) {
        db.GetDiskManager()->SetReadPenalty(std::chrono::microseconds{args.miss_penalty_us});
        std::cout << "Miss penalty: " << args.miss_penalty_us << " us per page read\n";
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

        const auto queries = tpch::Queries();

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

        // Latency records for latencies.csv — one per (query, run)
        struct LatencyRecord {
            std::string tag;
            int run;              // 0-indexed timed run number
            size_t rows;
            double latency_ms;
            uint64_t hits;
            uint64_t misses;
            uint64_t visits;
            uint64_t reads;
            double read_ms;
        };
        std::vector<LatencyRecord> latencies;

        // One extra run per query with the policy timer on
        struct PolicyRecord {
            std::string tag;
            size_t rows;
            double latency_ms;
            PolicyStats stats;
        };
        std::vector<PolicyRecord> policy_records;

        auto* bpm = db.GetBufferPoolManager();
        auto* disk = db.GetDiskManager();
        uint64_t total_reads = 0;
        std::chrono::nanoseconds total_read_time{0};

        // Warmup runs — timings discarded, CSVs not overwritten
        if (args.warmup_runs > 0) {
            std::cout << "\n--- Warmup (" << args.warmup_runs << " run" << (args.warmup_runs == 1 ? "" : "s") << ") ---\n";
            for (int w = 0; w < args.warmup_runs; ++w) {
                for (const auto& q : queries) {
                    try {
                        (void)db.ExecuteSQL(q.sql, args.exec_mode);
                    } catch (const std::exception& e) {
                        std::cerr << "  warmup " << q.tag << " failed: " << e.what() << "\n";
                    }
                }
            }
        }

        std::cout << "\n--- Timed runs (" << args.timed_runs << ") ---\n";
        int passed = 0;
        for (const auto& q : queries) {
            std::cout << "\n  " << q.name << " ...\n";
            size_t rows = 0;
            for (int run = 0; run < args.timed_runs; ++run) {
                bpm->ResetStats();
                disk->ResetReadStats();
                auto start = std::chrono::steady_clock::now();
                try {
                    auto result = db.ExecuteSQL(q.sql, args.exec_mode);
                    auto elapsed = std::chrono::steady_clock::now() - start;
                    double ms = std::chrono::duration<double, std::milli>(elapsed).count();
                    uint64_t hits = bpm->GetHitCount();
                    uint64_t misses = bpm->GetMissCount();
                    uint64_t visits = bpm->GetVisitCount();
                    const auto read_stats = disk->GetReadStats();
                    double read_ms = std::chrono::duration<double, std::milli>(read_stats.time).count();
                    rows = result.tuples.size();
                    total_reads += read_stats.reads;
                    total_read_time += read_stats.time;

                    std::cout << "    run " << run << ": "
                              << static_cast<int64_t>(ms) << " ms, "
                              << rows << " rows, "
                              << hits << " hits, "
                              << misses << " misses, "
                              << visits << " visits, "
                              << read_stats.reads << " reads\n";

                    // Export query results CSV ONLY on run 0 (stable output)
                    if (export_csv && run == 0) {
                        const auto& schema = result.schema;
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

                    latencies.push_back({q.tag, run, rows, ms, hits, misses, visits,
                                         read_stats.reads, read_ms});
                } catch (const std::exception& e) {
                    auto elapsed = std::chrono::steady_clock::now() - start;
                    double ms = std::chrono::duration<double, std::milli>(elapsed).count();
                    std::cerr << "    run " << run << ": FAILED (" << static_cast<int64_t>(ms) << " ms) — " << e.what() << "\n";
                }
            }

            // One more run with the policy timer on. Timed runs above never include it.
            if (args.policy_timing) {
                bpm->ResetStats();
                bpm->SetPolicyTiming(true);
                auto start = std::chrono::steady_clock::now();

                try {
                    auto result = db.ExecuteSQL(q.sql, args.exec_mode);
                    auto elapsed = std::chrono::steady_clock::now() - start;
                    double ms = std::chrono::duration<double, std::milli>(elapsed).count();
                    policy_records.push_back({q.tag, result.tuples.size(), ms, bpm->GetPolicyStats()});
                    std::cout << "    policy run: " << static_cast<int64_t>(ms) << " ms, " << policy_records.back().stats.TotalCalls() << " policy calls\n";
                } catch (const std::exception& e) {
                    std::cerr << "    policy run FAILED — " << e.what() << "\n";
                }
                bpm->SetPolicyTiming(false);
            }
            if (!latencies.empty() && latencies.back().tag == q.tag) {
                ++passed;
            }
        }

        std::cout << "\n=== " << passed << "/" << queries.size() << " TPC-H queries passed ===\n";

        // Write latencies.csv — one row per (query, run)
        if (export_csv && !latencies.empty()) {
            auto lat_path = fs::path(args.results_dir) / "latencies.csv";
            std::ofstream ofs(lat_path);
            if (ofs) {
                ofs << "query,run,rows,latency_ms,hits,misses,visits,reads,read_ms\n";
                ofs << std::fixed << std::setprecision(4);
                for (const auto& rec : latencies) {
                    ofs << rec.tag << ","
                        << rec.run << ","
                        << rec.rows << ","
                        << rec.latency_ms << ","
                        << rec.hits << ","
                        << rec.misses << ","
                        << rec.visits << ","
                        << rec.reads << ","
                        << rec.read_ms << "\n";
                }
                std::cout << "\nLatencies written to " << lat_path.string() << "\n";
            } else {
                std::cerr << "Warning: could not open " << lat_path.string() << " for writing\n";
            }
        }

        // Settings and totals for this run
        if (export_csv) {
            auto info_path = fs::path(args.results_dir) / "run_info.csv";
            std::ofstream ofs(info_path);
            if (ofs) {
                auto flag = [](bool b) { return b ? "true" : "false"; };
                ofs << "key,value\n"
                    << "sf," << args.sf << "\n"
                    << "pool_size," << args.pool_size << "\n"
                    << "db_pages," << db_pages << "\n"
                    << "db_pages_total," << db_pages_total << "\n"
                    << "miss_penalty_us," << args.miss_penalty_us << "\n"
                    << "direct_io," << flag(args.direct_io) << "\n"
                    << "eviction," << eviction_name << "\n"
                    << "execution_mode," << (args.exec_mode == ExecutionMode::VECTORIZED ? "vectorized" : "tuple") << "\n"
                    << "policy_timing," << flag(args.policy_timing) << "\n"
                    << "us_per_read,";
                if (total_reads > 0) {
                    const double read_us = std::chrono::duration<double, std::micro>(total_read_time).count();
                    ofs << std::fixed << std::setprecision(3) << read_us / static_cast<double>(total_reads);
                }
                ofs << "\n";
                std::cout << "Run info written to " << info_path.string() << "\n";
            } else {
                std::cerr << "Warning: could not open " << info_path.string() << " for writing\n";
            }
        }

        if (export_csv && args.policy_timing && !policy_records.empty()) {
            auto pol_path = fs::path(args.results_dir) / "policy_time.csv";
            std::ofstream ofs(pol_path);
            if (ofs) {
                ofs << "query,rows,latency_ms,policy_calls,policy_ms,policy_raw_ms,record_ms,set_evictable_ms,evict_ms,remove_ms\n";
                ofs << std::fixed << std::setprecision(4);
                auto ms = [](std::chrono::nanoseconds t) {
                    return std::chrono::duration<double, std::milli>(t).count();
                };

                for (const auto& rec : policy_records) {
                    const auto& st = rec.stats;
                    // Each method's time minus what reading the clock around its calls cost
                    auto net = [&](std::chrono::nanoseconds raw, uint64_t calls) {
                        return ms(std::max(std::chrono::nanoseconds{0}, raw - st.timer_cost * static_cast<std::chrono::nanoseconds::rep>(calls)));
                    };

                    ofs << rec.tag << ","
                        << rec.rows << ","
                        << rec.latency_ms << ","
                        << st.TotalCalls() << ","
                        << ms(st.NetTime()) << ","
                        << ms(st.RawTime()) << ","
                        << net(st.record_time, st.record_calls) << ","
                        << net(st.set_evictable_time, st.set_evictable_calls) << ","
                        << net(st.evict_time, st.evict_calls) << ","
                        << net(st.remove_time, st.remove_calls) << "\n";
                }
                std::cout << "Policy time written to " << pol_path.string() << "\n";
            } else {
                std::cerr << "Warning: could not open " << pol_path.string() << " for writing\n";
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
