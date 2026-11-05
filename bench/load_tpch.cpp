#include "engine/database.hpp"
#include "buffer/eviction_policy_factory.hpp"
#include "executor/executor.hpp"
#include "common/exception.hpp"
#include "tpch_streams.hpp"
#include "tpch_workload.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_set>

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
    std::optional<double> planner_read_us;  // overrides the page read cost the planner prices scans with
    tpch::Workload workload{tpch::Workload::SINGLE};
    uint64_t seed{1};
    std::optional<size_t> stream_length;  // set to the scale factor's default when not given
    bool trace_collapse{false};
    EvictionPolicyKind eviction{EvictionPolicyKind::LRU};
    std::string eviction_model_path;
    bool keep_load_pool{false};  // a workload starts with the pool the load left, not an empty one
#ifdef SHILMANDB_HAS_LIBTORCH
    bool use_learned_join{false};
    std::string join_model_path;
#endif
};

static void PrintUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " --sf <scale_factor> --db-file <path> --data-dir <path> [--no-verify]"
              << " [--enable-tracing <path> [--trace-collapse]] [--pool-size <N>] [--create-indexes <true|false>]"
              << " [--warmup <N>] [--runs <N>]"
              << " [--results-dir <path>]"
              << " [--execution-mode <tuple|vectorized>]"
              << " [--miss-penalty-us <N>] [--direct-io] [--policy-timing] [--planner-read-us <X>]"
              << " [--workload <single|stream|lookup|mixed>] [--stream-length <N>] [--seed <S>] [--keep-load-pool]"
              << " [--eviction-policy <lru|clock|lru2|arc|lirs|learned-v2|learned-v1>] [--eviction-model-path <path>]"
#ifdef SHILMANDB_HAS_LIBTORCH
              << " [--use-learned-join --join-model-path <path>]"
              << " [--use-learned-eviction (same as --eviction-policy learned-v2)] [--eviction-model-version <v1|v2>]"
#endif
              << "\n";
}

[[noreturn]] static void ExitWithError(const std::string& message) {
    std::cerr << "Error: " << message << "\n";
    std::exit(1);
}

static bool IsLearned(EvictionPolicyKind kind) {
    return kind == EvictionPolicyKind::LEARNED_V1 || kind == EvictionPolicyKind::LEARNED_V2;
}

// The name run_info.csv records. The learned ones keep their old names, v2 and v1.
static std::string RunInfoEvictionName(EvictionPolicyKind kind) {
    if (kind == EvictionPolicyKind::LEARNED_V2) return "v2";
    if (kind == EvictionPolicyKind::LEARNED_V1) return "v1";
    return std::string(EvictionPolicyKindName(kind));
}

// A string made only of digits, as a number. Nothing for anything else, or a number too big.
static std::optional<uint64_t> ParseUnsigned(const std::string& text) {
    const bool digits = !text.empty() &&
        std::all_of(text.begin(), text.end(), [](unsigned char ch) { return std::isdigit(ch) != 0; });
    if (!digits) return std::nullopt;
    try {
        return std::stoull(text);
    } catch (const std::out_of_range&) {
        return std::nullopt;
    }
}

static Args ParseArgs(int argc, char* argv[]) {
    Args args;
    bool seed_given = false;
    std::string policy_name;  // what --eviction-policy said, empty if not given
    bool use_learned_eviction = false;
    std::string eviction_version{"v2"};
    bool version_given = false;

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
        } else if (arg == "--planner-read-us" && i + 1 < argc) {
            const std::string val = argv[++i];
            std::optional<double> parsed;
            try {
                size_t used = 0;
                const double number = std::stod(val, &used);
                if (used == val.size() && std::isfinite(number) && number > 0) parsed = number;
            } catch (const std::exception&) {
                // not a number; reported below
            }
            if (!parsed) ExitWithError("--planner-read-us expects a number > 0, got '" + val + "'");
            args.planner_read_us = parsed;
        } else if (arg == "--workload" && i + 1 < argc) {
            const std::string val = argv[++i];
            const auto workload = tpch::ParseWorkload(val);
            if (!workload) ExitWithError("unknown --workload '" + val + "' (expected single, stream, lookup or mixed)");
            args.workload = *workload;
        } else if (arg == "--stream-length" && i + 1 < argc) {
            const std::string val = argv[++i];
            const auto length = ParseUnsigned(val);
            if (!length || *length < 1) ExitWithError("--stream-length expects an integer >= 1, got '" + val + "'");
            args.stream_length = static_cast<size_t>(*length);
        } else if (arg == "--seed" && i + 1 < argc) {
            const std::string val = argv[++i];
            const auto seed = ParseUnsigned(val);
            if (!seed) ExitWithError("--seed expects a non-negative integer, got '" + val + "'");
            args.seed = *seed;
            seed_given = true;
        } else if (arg == "--trace-collapse") {
            args.trace_collapse = true;
        } else if (arg == "--keep-load-pool") {
            args.keep_load_pool = true;
        } else if (arg == "--eviction-policy" && i + 1 < argc) {
            policy_name = argv[++i];
            const auto kind = ParseEvictionPolicyKind(policy_name);
            if (!kind) {
                ExitWithError("unknown --eviction-policy '" + policy_name + "' (expected lru, clock, lru2, arc, lirs, learned-v2 or learned-v1)");
            }
            args.eviction = *kind;
        } else if (arg == "--use-learned-eviction") {
            use_learned_eviction = true;
        } else if (arg == "--eviction-model-path" && i + 1 < argc) {
            args.eviction_model_path = argv[++i];
        } else if (arg == "--eviction-model-version" && i + 1 < argc) {
            eviction_version = argv[++i];
            version_given = true;
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
    if (use_learned_eviction && !policy_name.empty()) {
        ExitWithError("--eviction-policy and --use-learned-eviction can't be used together "
                      "(--use-learned-eviction means --eviction-policy learned-v2, or learned-v1 with --eviction-model-version v1)");
    }
    if (version_given && !policy_name.empty()) {
        std::cerr << "Warning: --eviction-model-version is ignored with --eviction-policy (learned-v1 or learned-v2 names the version)\n";
    }
    // Only exactly "v2" means V2, as the database has always read it
    if (use_learned_eviction) {
        args.eviction = eviction_version == "v2" ? EvictionPolicyKind::LEARNED_V2 : EvictionPolicyKind::LEARNED_V1;
    }
    if (!args.eviction_model_path.empty() && !IsLearned(args.eviction)) {
        std::cerr << "Warning: --eviction-model-path is ignored with eviction policy " << EvictionPolicyKindName(args.eviction)
                  << " (only learned-v2 and learned-v1 use a model)\n";
    }
    if (IsLearned(args.eviction)) {
        const std::string flag = use_learned_eviction ? "--use-learned-eviction" : "--eviction-policy " + policy_name;
#ifdef SHILMANDB_HAS_LIBTORCH
        if (args.eviction_model_path.empty()) ExitWithError(flag + " requires --eviction-model-path");
#else
        ExitWithError(flag + " needs a build with LibTorch (SHILMANDB_ENABLE_ML=ON)");
#endif
    }
    const bool single = args.workload == tpch::Workload::SINGLE;
    const std::string workload_name = tpch::WorkloadName(args.workload);
    if (single && args.stream_length) ExitWithError("--stream-length needs --workload stream, lookup or mixed");
    if (single && seed_given) ExitWithError("--seed needs --workload stream, lookup or mixed");
    if (single && args.keep_load_pool) ExitWithError("--keep-load-pool needs --workload stream, lookup or mixed");
    if (args.trace_collapse && args.trace_path.empty()) ExitWithError("--trace-collapse needs --enable-tracing <path>");
    if (!single && !args.stream_length) {
        args.stream_length = tpch::DefaultStreamLength(args.workload, args.sf);
        if (!args.stream_length) {
            ExitWithError("--workload " + workload_name + " has no default length at --sf " + args.sf +
                          "; pass --stream-length <N>");
        }
    }
    // Ensure data_dir ends with '/'
    if (args.data_dir.back() != '/') {
        args.data_dir += '/';
    }
    return args;
}

// Output helpers

// A CSV field in quotes, with any quote inside it doubled.
static std::string QuoteCsv(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 2);
    out += '"';
    for (char ch : text) {
        if (ch == '"') out += '"';
        out += ch;
    }
    out += '"';
    return out;
}

// A result value as a CSV field. VARCHARs are quoted when they hold a comma, a quote or a newline.
static std::string CsvValue(const Value& v, const Schema& schema, uint32_t col) {
    auto str = v.ToString();
    const bool needs_quoting = schema.GetColumn(col).type == TypeId::VARCHAR && str.find_first_of(",\"\n") != std::string::npos;
    return needs_quoting ? QuoteCsv(str) : str;
}

static double ToMs(std::chrono::nanoseconds t) {
    return std::chrono::duration<double, std::milli>(t).count();
}

static const char* BoolText(bool b) { return b ? "true" : "false"; }

// A number with a fixed count of decimals.
static std::string Fixed(double value, int decimals) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(decimals) << value;
    return out.str();
}

// 81140 -> "81,140"
static std::string WithCommas(uint64_t n) {
    std::string text = std::to_string(n);
    for (auto i = static_cast<std::ptrdiff_t>(text.size()) - 3; i > 0; i -= 3) {
        text.insert(static_cast<size_t>(i), ",");
    }
    return text;
}

static void AddReads(ReadStats& total, const ReadStats& more) {
    total.reads += more.reads;
    total.time += more.time;
}

// Makes the results folder if it isn't there yet. Says why if it can't.
[[nodiscard]] static bool MakeResultsDir(const std::string& dir) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        std::cerr << "Error: could not create results directory '" << dir << "': " << ec.message() << "\n";
        return false;
    }
    return true;
}

// Writes the header line, then whatever `write_rows` adds. Says so if the file won't open.
template <typename WriteRows>
[[nodiscard]] static bool WriteCsv(const fs::path& path, const std::string& header, WriteRows&& write_rows) {
    std::ofstream ofs(path);
    if (!ofs) {
        std::cerr << "Error: could not open " << path.string() << " for writing\n";
        return false;
    }
    ofs << header << "\n";
    write_rows(ofs);
    return true;
}

// One run with the policy timer on.
struct PolicyRecord {
    std::string tag;
    size_t rows;
    double latency_ms;
    PolicyStats stats;
};

constexpr const char* kPolicyColumns = "query,rows,latency_ms,policy_calls,policy_ms,policy_raw_ms,record_ms,set_evictable_ms,evict_ms,remove_ms";

// One policy_time.csv row, from the query column on.
static void WritePolicyColumns(std::ostream& ofs, const PolicyRecord& rec) {
    const auto& st = rec.stats;
    auto net = [&](std::chrono::nanoseconds raw, uint64_t calls) {
        return ToMs(std::max(std::chrono::nanoseconds{0}, raw - st.timer_cost * static_cast<std::chrono::nanoseconds::rep>(calls)));
    };

    ofs << rec.tag << ","
        << rec.rows << ","
        << rec.latency_ms << ","
        << st.TotalCalls() << ","
        << ToMs(st.NetTime()) << ","
        << ToMs(st.RawTime()) << ","
        << net(st.record_time, st.record_calls) << ","
        << net(st.set_evictable_time, st.set_evictable_calls) << ","
        << net(st.evict_time, st.evict_calls) << ","
        << net(st.remove_time, st.remove_calls) << "\n";
}

// What every run_info.csv records about the database.
struct RunInfo {
    page_id_t db_pages;        // tables only, right after the load
    page_id_t db_pages_total;  // after the indexes
    std::string eviction;
    double planner_read_us;  // the page read cost the planner priced scans with
};

// The keys every run_info.csv starts with. `reads` are the timed runs' page reads.
static void WriteRunInfo(std::ostream& ofs, const Args& args, const RunInfo& info, const ReadStats& reads) {
    ofs << "sf," << args.sf << "\n"
        << "pool_size," << args.pool_size << "\n"
        << "db_pages," << info.db_pages << "\n"
        << "db_pages_total," << info.db_pages_total << "\n"
        << "miss_penalty_us," << args.miss_penalty_us << "\n"
        << "direct_io," << BoolText(args.direct_io) << "\n"
        << "eviction," << info.eviction << "\n"
        << "execution_mode," << (args.exec_mode == ExecutionMode::VECTORIZED ? "vectorized" : "tuple") << "\n"
        << "policy_timing," << BoolText(args.policy_timing) << "\n"
        << "us_per_read,";
    if (reads.reads > 0) {
        const double read_us = std::chrono::duration<double, std::micro>(reads.time).count();
        ofs << Fixed(read_us / static_cast<double>(reads.reads), 3);
    }
    ofs << "\n"
        << "planner_read_us," << Fixed(info.planner_read_us, 2) << "\n";
}

//Workloads

// One item of one pass: its result size, its time and what the pool did.
struct ItemRecord {
    int pass{0};
    size_t seq{0};
    size_t rows{0};
    double latency_ms{0};
    uint64_t hits{0};
    uint64_t misses{0};
    uint64_t visits{0};
    ReadStats reads;
    uint64_t trace_start{0};  // fetches traced before the item
    uint64_t trace_end{0};    // and after it
};

// A pass's items added up.
struct PassTotals {
    size_t items{0};
    double latency_ms{0};
    uint64_t hits{0};
    uint64_t misses{0};
    uint64_t visits{0};
    ReadStats reads;
    uint64_t query_misses{0};
    uint64_t lookup_misses{0};

    void Add(const ItemRecord& r, tpch::StreamItem::Kind kind) {
        ++items;
        latency_ms += r.latency_ms;
        hits += r.hits;
        misses += r.misses;
        visits += r.visits;
        AddReads(reads, r.reads);
        (kind == tpch::StreamItem::Kind::LOOKUP ? lookup_misses : query_misses) += r.misses;
    }
};

struct ItemRun {
    ItemRecord record;
    QueryResult result;
};

static const char* KindName(tpch::StreamItem::Kind kind) {
    return kind == tpch::StreamItem::Kind::LOOKUP ? "lookup" : "tpch";
}

// Runs one item with fresh counts, the way each timed run of a single query starts.
static ItemRun RunItem(Database& db, const tpch::StreamItem& item, ExecutionMode mode) {
    auto* bpm = db.GetBufferPoolManager();
    auto* disk = db.GetDiskManager();
    bpm->ResetStats();
    disk->ResetReadStats();

    ItemRun run;
    auto& rec = run.record;
    rec.trace_start = bpm->GetTraceCount();
    const auto start = std::chrono::steady_clock::now();
    auto result = db.ExecuteSQL(item.sql, mode);
    rec.latency_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    rec.trace_end = bpm->GetTraceCount();

    rec.rows = result.tuples.size();
    rec.hits = bpm->GetHitCount();
    rec.misses = bpm->GetMissCount();
    rec.visits = bpm->GetVisitCount();
    rec.reads = disk->GetReadStats();
    run.result = std::move(result);
    return run;
}

static void PrintPass(const std::string& label, const PassTotals& t) {
    std::cout << "  " << label << ": " << t.items << " items, " << Fixed(t.latency_ms, 1) << " ms, "
              << WithCommas(t.misses) << " misses, " << WithCommas(t.visits) << " visits\n";
}

// Runs the workload's warmup passes, its timed passes and, if asked, one pass with the policy
// timer on. The CSVs are written only once everything ran. Returns the exit code.
static int RunWorkload(Database& db, const Args& args, const RunInfo& info) {
    using Kind = tpch::StreamItem::Kind;
    const auto items = tpch::MakeWorkload(args.workload, args.seed, *args.stream_length);
    const bool has_lookups = args.workload == tpch::Workload::LOOKUP || args.workload == tpch::Workload::MIXED;

    std::cout << "\n=== Workload: " << tpch::WorkloadName(args.workload) << " (seed " << args.seed << ", " << items.size() << " items per pass) ===\n";

    // A lookup that scans instead of reading through its index would measure the wrong thing
    if (has_lookups && args.create_indexes) {
        std::unordered_set<std::string> checked;
        for (const auto& item : items) {
            if (item.kind != Kind::LOOKUP || !checked.insert(item.sql).second) continue;
            if (!tpch::PlanUsesIndex(db, item.sql, item.index_name)) {
                std::cerr << "Error: lookup does not use " << item.index_name << ": " << item.sql << "\n";
                return 1;
            }
        }
        std::cout << "Plan check: " << checked.size() << " distinct lookups use their index\n";
    } else if (has_lookups) {
        std::cout << "Plan check: skipped, no indexes (every lookup scans its table)\n";
    }

    const bool export_csv = !args.results_dir.empty();
    if (export_csv && !MakeResultsDir(args.results_dir)) return 1;

    auto* bpm = db.GetBufferPoolManager();
    const bool tracing = !args.trace_path.empty();

    // Gives nothing back if the item failed, after saying which one
    auto run_item = [&](size_t seq) -> std::optional<ItemRun> {
        try {
            return RunItem(db, items[seq], args.exec_mode);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << items[seq].tag << " seq " << seq << ": " << e.what() << "\n";
            return std::nullopt;
        }
    };

    // Every pass runs the same items in the same order
    const int passes = args.warmup_runs + args.timed_runs;
    std::vector<ItemRecord> records;
    records.reserve(items.size() * static_cast<size_t>(passes));
    std::vector<PassTotals> totals;
    std::vector<QueryResult> first_timed_results;
    ReadStats timed_reads;

    std::cout << "\n--- Passes (" << args.warmup_runs << " warmup, " << args.timed_runs << " timed) ---\n";
    for (int pass = 0; pass < passes; ++pass) {
        const bool warmup = pass < args.warmup_runs;
        PassTotals pass_totals;
        for (size_t seq = 0; seq < items.size(); ++seq) {
            auto run = run_item(seq);
            if (!run) return 1;
            run->record.pass = pass;
            run->record.seq = seq;
            pass_totals.Add(run->record, items[seq].kind);
            records.push_back(run->record);
            if (pass == args.warmup_runs) first_timed_results.push_back(std::move(run->result));
        }
        if (!warmup) AddReads(timed_reads, pass_totals.reads);
        PrintPass("pass " + std::to_string(pass) + (warmup ? " (warmup)" : ""), pass_totals);
        totals.push_back(pass_totals);
    }

    // The trace stops here, so stream.csv's trace columns cover all of it after the row counts
    if (tracing) bpm->DisableTracing();

    // One more pass with the policy timer on. The timed passes never include it.
    std::vector<PolicyRecord> policy_records;
    if (args.policy_timing) {
        bpm->SetPolicyTiming(true);
        double total_ms = 0;
        uint64_t total_calls = 0;
        for (size_t seq = 0; seq < items.size(); ++seq) {
            auto run = run_item(seq);
            if (!run) return 1;
            policy_records.push_back({items[seq].tag, run->record.rows, run->record.latency_ms, bpm->GetPolicyStats()});
            total_ms += run->record.latency_ms;
            total_calls += policy_records.back().stats.TotalCalls();
        }
        bpm->SetPolicyTiming(false);
        std::cout << "  policy pass: " << items.size() << " items, " << Fixed(total_ms, 1) << " ms, " << WithCommas(total_calls) << " policy calls\n";
    }

    if (!export_csv) return 0;
    const fs::path dir(args.results_dir);

    const bool stream_ok = WriteCsv(
        dir / "stream.csv",
        "pass,warmup,seq,kind,query,rows,latency_ms,hits,misses,visits,reads,read_ms,trace_start,trace_end",
        [&](std::ostream& ofs) {
            ofs << std::fixed << std::setprecision(4);
            for (const auto& r : records) {
                const auto& item = items[r.seq];
                ofs << r.pass << ","
                    << (r.pass < args.warmup_runs ? 1 : 0) << ","
                    << r.seq << ","
                    << KindName(item.kind) << ","
                    << item.tag << ","
                    << r.rows << ","
                    << r.latency_ms << ","
                    << r.hits << ","
                    << r.misses << ","
                    << r.visits << ","
                    << r.reads.reads << ","
                    << ToMs(r.reads.time) << ",";
                // Left empty when nothing was traced
                if (tracing) ofs << r.trace_start;
                ofs << ",";
                if (tracing) ofs << r.trace_end;
                ofs << "\n";
            }
        });
    if (!stream_ok) return 1;

    const bool totals_ok = WriteCsv(
        dir / "stream_totals.csv",
        "pass,warmup,items,latency_ms,hits,misses,visits,reads,read_ms,query_misses,lookup_misses",
        [&](std::ostream& ofs) {
            ofs << std::fixed << std::setprecision(4);
            for (size_t pass = 0; pass < totals.size(); ++pass) {
                const auto& t = totals[pass];
                ofs << pass << ","
                    << (pass < static_cast<size_t>(args.warmup_runs) ? 1 : 0) << ","
                    << t.items << ","
                    << t.latency_ms << ","
                    << t.hits << ","
                    << t.misses << ","
                    << t.visits << ","
                    << t.reads.reads << ","
                    << ToMs(t.reads.time) << ","
                    << t.query_misses << ","
                    << t.lookup_misses << "\n";
            }
        });
    if (!totals_ok) return 1;

    const bool queries_ok = WriteCsv(dir / "stream_queries.csv", "seq,kind,query,sql", [&](std::ostream& ofs) {
        for (size_t seq = 0; seq < items.size(); ++seq) {
            const auto& item = items[seq];
            ofs << seq << "," << KindName(item.kind) << "," << item.tag << "," << QuoteCsv(item.sql) << "\n";
        }
    });
    if (!queries_ok) return 1;

    // Every result cell of the first timed pass, one per line, since items have different columns
    const bool results_ok = WriteCsv(dir / "stream_results.csv", "seq,row,col,value", [&](std::ostream& ofs) {
        for (size_t seq = 0; seq < first_timed_results.size(); ++seq) {
            const auto& schema = first_timed_results[seq].schema;
            const auto& tuples = first_timed_results[seq].tuples;
            for (size_t row = 0; row < tuples.size(); ++row) {
                for (uint32_t col = 0; col < schema.GetColumnCount(); ++col) {
                    ofs << seq << "," << row << "," << col << "," << CsvValue(tuples[row].GetValue(schema, col), schema, col) << "\n";
                }
            }
        }
    });
    if (!results_ok) return 1;

    const bool info_ok = WriteCsv(dir / "run_info.csv", "key,value", [&](std::ostream& ofs) {
        WriteRunInfo(ofs, args, info, timed_reads);
        ofs << "workload," << tpch::WorkloadName(args.workload) << "\n"
            << "seed," << args.seed << "\n"
            << "stream_length," << *args.stream_length << "\n"
            << "items," << items.size() << "\n"
            << "warmup_passes," << args.warmup_runs << "\n"
            << "timed_passes," << args.timed_runs << "\n";
        if (args.workload == tpch::Workload::MIXED) ofs << "lookups_per_query," << tpch::kMixedLookupsPerQuery << "\n";
        if (has_lookups) ofs << "zipf_s," << tpch::kLookupZipfS << "\n";
        ofs << "trace_collapse," << BoolText(args.trace_collapse) << "\n"
            << "cold_start," << BoolText(!args.keep_load_pool) << "\n";
    });
    if (!info_ok) return 1;

    if (args.policy_timing) {
        const bool policy_ok =
            WriteCsv(dir / "policy_time.csv", std::string("seq,") + kPolicyColumns, [&](std::ostream& ofs) {
                ofs << std::fixed << std::setprecision(4);
                for (size_t seq = 0; seq < policy_records.size(); ++seq) {
                    ofs << seq << ",";
                    WritePolicyColumns(ofs, policy_records[seq]);
                }
            });
        if (!policy_ok) return 1;
    }

    std::cout << "\nWorkload results written to " << dir.string() << "\n";
    return 0;
}

// ── Main ─────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    auto args = ParseArgs(argc, argv);

    std::cout << "=== ShilmanDB TPC-H Loader (SF=" << args.sf << ") ===\n";
    std::cout << "Execution mode: "
              << (args.exec_mode == ExecutionMode::VECTORIZED ? "VECTORIZED" : "TUPLE")
              << "\n";
    std::cout << "Eviction policy: " << EvictionPolicyKindName(args.eviction) << "\n";
    const std::string eviction_name = RunInfoEvictionName(args.eviction);

    std::string join_model_path;
#ifdef SHILMANDB_HAS_LIBTORCH
    if (args.use_learned_join) join_model_path = args.join_model_path;
#endif
    std::unique_ptr<Database> db_owner;
    try {
        db_owner = std::make_unique<Database>(args.db_file, args.pool_size, args.eviction, args.eviction_model_path, join_model_path);
    } catch (const std::exception& e) {
        ExitWithError(std::string("could not open the database: ") + e.what());
    }
    Database& db = *db_owner;
#ifdef SHILMANDB_HAS_LIBTORCH
    if (args.use_learned_join) {
        std::cout << "Learned join optimizer: ON (model: " << args.join_model_path << ")\n";
    }
#endif
    if (IsLearned(args.eviction)) {
        std::cout << "Learned eviction policy: ON (" << eviction_name << ", model: " << args.eviction_model_path << ")\n";
    }

    if (args.direct_io) {
        try {
            db.GetDiskManager()->SetDirectIO(true);
        } catch (const std::exception& e) {
            std::cerr << "Error: --direct-io failed: " << e.what() << "\n";
            return 1;
        }
        std::cout << "Direct I/O: ON\n";
    }

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

    if (args.planner_read_us) db.SetScanCosts(ScanCosts::ForPageRead(*args.planner_read_us));
    const double planner_read_us = db.GetScanCosts().random_page_us;
    // Same rule GetScanCosts uses: the largest of the file cache, direct I/O and the penalty wins
    const bool penalty_leads = static_cast<double>(args.miss_penalty_us) > ScanCosts::kCachedPageReadUs && (!args.direct_io || static_cast<double>(args.miss_penalty_us) >= ScanCosts::kDirectPageReadUs);
    const char* planner_why = args.planner_read_us ? "set by --planner-read-us" : penalty_leads ? "follows the read penalty" : args.direct_io ? "follows direct I/O" : "file cache";
    std::cout << "Planner page read cost: " << Fixed(planner_read_us, 2) << " us (" << planner_why << ")\n";
    const RunInfo run_info{db_pages, db_pages_total, eviction_name, planner_read_us};

    const bool single = args.workload == tpch::Workload::SINGLE;
    bool run_queries = !single || args.verify || !args.trace_path.empty() || !args.results_dir.empty();

    // A workload starts from an empty pool and a fresh policy, like a restart after the load.
    // Done before tracing starts, so the trace and the pool start at the same point.
    if (!single && !args.keep_load_pool) {
        try {
            db.GetBufferPoolManager()->DropAllPages();
        } catch (const std::exception& e) {
            std::cerr << "Error: could not empty the pool: " << e.what() << "\n";
            return 1;
        }
        std::cout << "Pool emptied before the workload (cold start)\n";
    }

    if (!args.trace_path.empty()) {
        if (!args.verify) {
            std::cout << "\nTracing enabled — running verification queries for trace generation\n";
        }
        db.GetBufferPoolManager()->EnableTracing(args.trace_path, args.trace_collapse);
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
    }

    if (!single) {
        try {
            if (RunWorkload(db, args, run_info) != 0) return 1;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            return 1;
        }
    } else if (run_queries) {
        // ── TPC-H target queries ─────────────────────────────────────

        std::cout << "\n=== TPC-H Query Verification ===\n";

        const auto queries = tpch::Queries();

        // Create results directory if requested
        const bool export_csv = !args.results_dir.empty();
        if (export_csv && !MakeResultsDir(args.results_dir)) return 1;

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
                                    ofs << CsvValue(tuple.GetValue(schema, c), schema, c);
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
                ofs << "key,value\n";
                WriteRunInfo(ofs, args, run_info, ReadStats{total_reads, total_read_time});
                if (args.trace_collapse) ofs << "trace_collapse,true\n";
                std::cout << "Run info written to " << info_path.string() << "\n";
            } else {
                std::cerr << "Warning: could not open " << info_path.string() << " for writing\n";
            }
        }

        if (export_csv && args.policy_timing && !policy_records.empty()) {
            auto pol_path = fs::path(args.results_dir) / "policy_time.csv";
            std::ofstream ofs(pol_path);
            if (ofs) {
                ofs << kPolicyColumns << "\n";
                ofs << std::fixed << std::setprecision(4);
                for (const auto& rec : policy_records) {
                    WritePolicyColumns(ofs, rec);
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
