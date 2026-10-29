#include "engine/database.hpp"
#include "common/exception.hpp"
#include "planner/join_graph.hpp"
#include "tpch_workload.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

// Times every connected join order of the TPC-H join queries and writes raw CSVs:
// one row per order, one row per operator of each order's analyze run, and what
// the classical (and learned) planner would have picked.

using namespace shilmandb;
namespace fs = std::filesystem;

namespace {

constexpr std::array<std::string_view, 6> kJoinQueries{"Q3", "Q5", "Q10", "Q12", "Q14", "Q19"};

using Clock = std::chrono::steady_clock;

//  Small helpers

[[nodiscard]] std::vector<std::string> SplitCommas(const std::string& s) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (true) {
        const size_t comma = s.find(',', start);
        parts.push_back(s.substr(start, comma - start));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return parts;
}

// The whole string as a number, or nothing if any of it isn't.
template <typename T>
[[nodiscard]] std::optional<T> ParseNumber(std::string_view s) {
    T value{};
    const char* end = s.data() + s.size();
    const auto [ptr, ec] = std::from_chars(s.data(), end, value);
    if (ec != std::errc() || ptr != end) return std::nullopt;
    return value;
}

// Items formatted by fmt and joined with sep.
template <typename Range, typename Fmt>
[[nodiscard]] std::string Join(const Range& items, std::string_view sep, Fmt fmt) {
    std::string out;
    bool first = true;
    for (const auto& item : items) {
        if (!first) out += sep;
        first = false;
        out += fmt(item);
    }
    return out;
}

[[nodiscard]] std::string FormatEst(double v) {
    std::ostringstream os;
    os << std::setprecision(12) << v;
    return os.str();
}

[[nodiscard]] std::string FormatMs(double v, int decimals = 3) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(decimals) << v;
    return os.str();
}

const auto kToString = [](const auto& v) { return std::to_string(v); };
const auto kSame = [](const std::string& s) { return s; };

// Fields joined with commas. Doubles must be formatted first, so every number is written on purpose.
template <typename... Fields>
[[nodiscard]] std::string CsvRow(const Fields&... fields) {
    static_assert((!std::is_floating_point_v<Fields> && ...), "format doubles with FormatEst or FormatMs");
    std::ostringstream os;
    const char* sep = "";
    ((os << sep << fields, sep = ","), ...);
    return os.str();
}

[[nodiscard]] double Median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t mid = v.size() / 2;
    return v.size() % 2 == 1 ? v[mid] : (v[mid - 1] + v[mid]) / 2.0;
}

//  Arguments

struct Args {
    std::string sf;
    std::string data_dir;
    std::string db_file;
    std::string out_dir{"bench/results/headroom/join"};
    std::vector<std::string> queries;
    int warmup{1};
    int runs{5};
    size_t pool_size{0};  // 0 = pick from the scale factor
    std::optional<std::vector<int>> orders;
    std::string join_model_path;  // only settable in ML builds
};

void PrintUsage(std::ostream& out, const char* prog) {
    out << "Usage: " << prog << " --sf <sf> --data-dir <dir> --db-file <path>\n"
        << "         [--out-dir bench/results/headroom/join]\n"
        << "         [--queries Q3,Q5,Q10,Q12,Q14,Q19] [--warmup 1] [--runs 5] [--pool-size N]\n"
        << "         [--orders 4,3,2,1,0]          time just this order (needs exactly one --queries tag)\n"
#ifdef SHILMANDB_HAS_LIBTORCH
        << "         [--join-model-path <path>]    also record the learned optimizer's choice\n"
#endif
        ;
}

[[noreturn]] void Fail(const std::string& message) {
    std::cerr << "Error: " << message << "\n";
    std::exit(1);
}

[[nodiscard]] std::string ValidTags() {
    return Join(kJoinQueries, ",", [](std::string_view t) { return std::string(t); });
}

[[nodiscard]] bool IsJoinQuery(const std::string& tag) {
    return std::find(kJoinQueries.begin(), kJoinQueries.end(), tag) != kJoinQueries.end();
}

[[nodiscard]] std::vector<std::string> ParseQueries(const std::string& value) {
    std::vector<std::string> tags;
    for (const auto& tag : SplitCommas(value)) {
        if (!IsJoinQuery(tag)) {
            std::string upper = tag;
            std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char c) { return std::toupper(c); });
            const std::string hint = IsJoinQuery(upper) ? " (tags are case-sensitive: did you mean '" + upper + "'?)" : "";
            Fail("--queries: '" + tag + "' is not one of the join queries " + ValidTags() + hint);
        }
        if (std::find(tags.begin(), tags.end(), tag) != tags.end()) {
            Fail("--queries: '" + tag + "' is listed twice");
        }
        tags.push_back(tag);
    }
    return tags;
}

[[nodiscard]] std::vector<int> ParseOrder(const std::string& value) {
    std::vector<int> order;
    for (const auto& slot : SplitCommas(value)) {
        const auto n = ParseNumber<int>(slot);
        if (!n || *n < 0) {
            Fail("--orders: '" + slot + "' is not a non-negative integer (expected table slots like 4,3,2,1,0)");
        }
        order.push_back(*n);
    }
    return order;
}

template <typename T>
[[nodiscard]] T ParseCount(const std::string& flag, const std::string& value, T min) {
    const auto n = ParseNumber<T>(value);
    if (!n || *n < min) Fail(flag + " expects an integer >= " + std::to_string(min) + ", got '" + value + "'");
    return *n;
}

// Checks everything that can be checked without the data, so a typo fails in
// seconds instead of after a long load.
[[nodiscard]] Args ParseArgs(int argc, char* argv[]) {
    Args args;
    std::optional<std::string> queries_arg;
    std::optional<std::string> orders_arg;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) Fail(arg + " needs a value");
            return argv[++i];
        };

        if (arg == "--sf") args.sf = value();
        else if (arg == "--data-dir") args.data_dir = value();
        else if (arg == "--db-file") args.db_file = value();
        else if (arg == "--out-dir") args.out_dir = value();
        else if (arg == "--queries") queries_arg = value();
        else if (arg == "--orders") orders_arg = value();
        else if (arg == "--warmup") args.warmup = ParseCount<int>(arg, value(), 0);
        else if (arg == "--runs") args.runs = ParseCount<int>(arg, value(), 1);
        else if (arg == "--pool-size") args.pool_size = ParseCount<size_t>(arg, value(), 1);
#ifdef SHILMANDB_HAS_LIBTORCH
        else if (arg == "--join-model-path") args.join_model_path = value();
#endif
        else if (arg == "--help" || arg == "-h") {
            PrintUsage(std::cout, argv[0]);
            std::exit(0);
        } else {
            PrintUsage(std::cerr, argv[0]);
            Fail("unknown argument '" + arg + "'");
        }
    }

    if (args.sf.empty() || args.data_dir.empty() || args.db_file.empty()) {
        PrintUsage(std::cerr, argv[0]);
        Fail("--sf, --data-dir and --db-file are required");
    }

    char* sf_end = nullptr;
    const double sf = std::strtod(args.sf.c_str(), &sf_end);
    if (sf_end != args.sf.c_str() + args.sf.size() || !(sf > 0.0)) {
        Fail("--sf expects a positive number, got '" + args.sf + "'");
    }

    if (queries_arg) {
        args.queries = ParseQueries(*queries_arg);
    } else {
        for (auto tag : kJoinQueries) args.queries.emplace_back(tag);
    }

    if (orders_arg) {
        if (!queries_arg || args.queries.size() != 1) {
            Fail("--orders needs exactly one --queries tag, e.g. --queries Q5 --orders 4,3,2,1,0");
        }
        args.orders = ParseOrder(*orders_arg);
    }

    // A file that exists but isn't a model still fails when the Database loads it, before the data.
    if (!args.join_model_path.empty() && !fs::is_regular_file(args.join_model_path)) {
        Fail("--join-model-path: " + args.join_model_path + " not found");
    }

    if (args.data_dir.back() != '/') args.data_dir += '/';
    for (const auto& td : tpch::TableDescs()) {
        if (!fs::is_regular_file(args.data_dir + td.tbl_file)) {
            Fail("--data-dir: " + args.data_dir + td.tbl_file + " not found");
        }
    }

    std::error_code ec;
    fs::create_directories(args.out_dir, ec);
    if (ec || ::access(args.out_dir.c_str(), W_OK) != 0) {
        Fail("--out-dir: cannot write to '" + args.out_dir + "'" + (ec ? " (" + ec.message() + ")" : ""));
    }

    return args;
}

[[nodiscard]] size_t DefaultPoolSize(const std::string& sf) {
    if (sf == "0.01") return 128;
    if (sf == "0.1") return 512;
    if (sf == "1.0" || sf == "1") return 2048;
    return 4096;
}

// A full run (every join query, no --orders) gets no suffix. Anything smaller is
// tagged, so a quick check never overwrites a full sweep.
[[nodiscard]] std::string FileSuffix(const Args& args) {
    if (args.queries.size() == kJoinQueries.size() && !args.orders) return "";
    std::string suffix = "_" + Join(args.queries, "-", kSame);
    if (args.orders) suffix += "_o" + Join(*args.orders, "-", kToString);
    return suffix;
}

//  Output

// One output CSV. Every row is flushed right away, so a killed run keeps what it finished.
class CsvFile {
public:
    CsvFile(const fs::path& path, const std::string& header) : path_(path), out_(path) {
        if (!out_) throw DatabaseException("cannot open " + path_.string() + " for writing");
        Write(header);
    }

    void Write(const std::string& row) {
        out_ << row << '\n';
        out_.flush();
        if (!out_) throw DatabaseException("write to " + path_.string() + " failed");
    }

    [[nodiscard]] const fs::path& Path() const { return path_; }

private:
    fs::path path_;
    std::ofstream out_;
};

//  The sweep

// One order to time. Connected orders are forced; a planner's pick that isn't
// connected (it can't see WHERE joins) runs unforced, as that planner plans it.
struct Candidate {
    std::vector<int> order;
    bool connected{true};
    bool is_classical{false};
    bool is_learned{false};
    bool use_model{false};  // unforced runs only

    std::optional<double> median_ms;  // set once timed
    bool result_ok{false};
};

struct QueryJob {
    tpch::Query query;
    JoinOrderInfo classical;
    std::optional<JoinOrderInfo> learned;
    std::vector<Candidate> candidates;
};

[[nodiscard]] QueryOptions OptionsFor(const Candidate& c, bool analyze) {
    QueryOptions options;
    options.analyze = analyze;
    options.use_learned_join = c.use_model;
    if (c.connected) options.forced_join_order = c.order;
    return options;
}

[[nodiscard]] std::string OrderNames(const std::vector<int>& order, const std::vector<std::string>& tables) {
    return Join(order, ">", [&](int slot) {
        return slot >= 0 && static_cast<size_t>(slot) < tables.size() ? tables[slot] : std::to_string(slot);
    });
}

// Plans the query (no running), picks the orders to time and checks a forced --orders.
[[nodiscard]] QueryJob Prepare(Database& db, const tpch::Query& query, const Args& args) {
    QueryJob job{query, {}, std::nullopt, {}};

    QueryOptions classical;
    classical.use_learned_join = false;
    job.classical = db.ExplainJoinOrder(query.sql, classical);

    if (!args.join_model_path.empty()) {
        QueryOptions learned;
        learned.use_learned_join = true;
        job.learned = db.ExplainJoinOrder(query.sql, learned);

        // The planner quietly falls back when the model throws, which looks just like a rejection.
        // An empty model order means it never ran, so stop before any timing.
        if (job.learned->model_order.empty() && job.classical.tables.size() > 1) {
            Fail("--join-model-path: the model in " + args.join_model_path + " did not run on " + query.tag + " (is it the join order model?)");
        }
    }

    std::vector<std::vector<int>> orders;
    if (args.orders) {
        QueryOptions forced;
        forced.forced_join_order = *args.orders;
        try {
            (void)db.ExplainJoinOrder(query.sql, forced);
        } catch (const DatabaseException& e) {
            Fail(query.tag + " --orders " + Join(*args.orders, ",", kToString) + ": " + e.what());
        }
        orders.push_back(*args.orders);
    } else {
        orders = ConnectedOrders(static_cast<int>(job.classical.tables.size()), job.classical.edges);
    }

    const auto learned_pick = [&](const std::vector<int>& order) { return job.learned && order == job.learned->order; };

    for (auto& order : orders) {
        Candidate c;
        c.is_classical = order == job.classical.order;
        c.is_learned = learned_pick(order);
        c.order = std::move(order);
        job.candidates.push_back(std::move(c));
    }

    if (!args.orders) {
        const auto has = [&](bool Candidate::*flag) {
            return std::any_of(job.candidates.begin(), job.candidates.end(), [&](const Candidate& c) { return c.*flag; });
        };
        if (!has(&Candidate::is_classical)) {
            Candidate c;
            c.order = job.classical.order;
            c.connected = false;
            c.is_classical = true;
            c.is_learned = learned_pick(c.order);
            job.candidates.push_back(std::move(c));
        }
        if (job.learned && !has(&Candidate::is_learned)) {
            Candidate c;
            c.order = job.learned->order;
            c.connected = false;
            c.is_learned = true;
            c.use_model = true;
            job.candidates.push_back(std::move(c));
        }
    }
    return job;
}

struct Outputs {
    std::string sf;
    CsvFile& orders;
    CsvFile& operators;
    CsvFile& choices;
};

void WriteChoices(const QueryJob& job, Outputs& out) {
    const auto& tables = job.classical.tables;
    out.choices.Write(CsvRow(
        out.sf, job.query.tag, "classical", OrderNames(job.classical.order, tables),
        ToString(job.classical.source), OrderNames(job.classical.model_order, tables)
    ));
    if (job.learned) {
        out.choices.Write(CsvRow(
            out.sf, job.query.tag, "learned", OrderNames(job.learned->order, tables),
            ToString(job.learned->source), OrderNames(job.learned->model_order, tables)
        ));
    }
}

// Warmup, timed runs and one analyze run for one order, then its CSV rows.
void TimeCandidate(Database& db, const QueryJob& job, Candidate& c, const QueryResult& reference, const Args& args, Outputs& out) {
    const auto& sql = job.query.sql;
    const auto options = OptionsFor(c, false);

    for (int i = 0; i < args.warmup; ++i) {
        (void)db.ExecuteSQL(sql, options);
    }

    std::vector<double> times;
    times.reserve(args.runs);
    for (int i = 0; i < args.runs; ++i) {
        const auto start = Clock::now();
        const auto result = db.ExecuteSQL(sql, options);
        times.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
        if (i == 0) c.result_ok = tpch::SameResult(reference, result);
    }
    c.median_ms = Median(times);

    // Kept apart from the timed runs, so profiling never touches the latencies.
    const auto analyzed = db.ExecuteSQL(sql, OptionsFor(c, true));
    const auto& profile = analyzed.profile;
    const auto& tables = job.classical.tables;
    if (profile.join.order != c.order) {
        std::cerr << "  warning: " << job.query.tag << " planned " << OrderNames(profile.join.order, tables) << " instead of " << OrderNames(c.order, tables) << "\n";
    }

    const auto order = OrderNames(c.order, tables);
    for (size_t i = 0; i < profile.operators.size(); ++i) {
        const auto& op = profile.operators[i];
        out.operators.Write(CsvRow(
            out.sf, job.query.tag, order, i, op.depth, op.name, op.rows, 
            FormatMs(op.total_ms), FormatMs(op.self_ms)
        ));
    }

    const auto table_rows = Join(c.order, ";", [&](int slot) {
        return std::to_string(db.GetCatalog()->GetTable(tables[slot])->stats.row_count);
    });
    const auto [min_it, max_it] = std::minmax_element(times.begin(), times.end());

    out.orders.Write(CsvRow(
        out.sf, job.query.tag, order, Join(c.order, " ", kToString), int{c.connected},
        int{c.is_classical}, int{c.is_learned}, FormatEst(profile.join.est_cost),
        Join(profile.join.est_step_rows, ";", FormatEst), Join(profile.step_rows, ";", kToString),
        table_rows, Join(profile.scan_rows, ";", kToString), FormatMs(*c.median_ms),
        FormatMs(*min_it), FormatMs(*max_it), args.runs, int{c.result_ok}
    ));
}

// Returns how many orders failed to run.
[[nodiscard]] int SweepQuery(Database& db, QueryJob& job, const Args& args, Outputs& out) {
    const auto& tag = job.query.tag;
    const auto& tables = job.classical.tables;

    std::cout << "\n" << tag << ": " << job.candidates.size() << " order(s) | classical " << OrderNames(job.classical.order, tables);
    if (job.learned) {
        std::cout << " | learned " << OrderNames(job.learned->order, tables) << " (" << ToString(job.learned->source) << ")";
    }
    std::cout << "\n";

    WriteChoices(job, out);

    // The default plan with the model off is the answer every order must match.
    QueryOptions ref_options;
    ref_options.use_learned_join = false;
    const auto reference = db.ExecuteSQL(job.query.sql, ref_options);

    int failures = 0;
    const size_t total = job.candidates.size();
    for (size_t i = 0; i < total; ++i) {
        auto& c = job.candidates[i];
        std::cout << tag << " [" << i + 1 << "/" << total << "] " << OrderNames(c.order, tables) << std::flush;
        try {
            TimeCandidate(db, job, c, reference, args, out);
            std::cout << "  median " << FormatMs(*c.median_ms, 1) << " ms  " << (c.result_ok ? "ok" : "RESULT DIFFERS");
        } catch (const std::exception& e) {
            ++failures;
            c.median_ms.reset();  // no row was written, so leave it out of the summary
            std::cout << "  FAILED: " << e.what();
        }
        if (c.is_classical) std::cout << "  [classical]";
        if (c.is_learned) std::cout << "  [learned]";
        if (!c.connected) std::cout << "  (not connected, run unforced)";
        std::cout << std::endl;
    }
    return failures;
}

void PrintSummary(const std::vector<QueryJob>& jobs) {
    std::cout << "\n=== Summary (median ms) ===\n";
    for (const auto& job : jobs) {
        const auto& cands = job.candidates;
        const auto& tables = job.classical.tables;

        const Candidate* best = nullptr;
        for (const auto& c : cands) {
            if (c.median_ms && c.result_ok && (!best || *c.median_ms < *best->median_ms)) best = &c;
        }
        std::cout << std::left << std::setw(4) << job.query.tag << std::right;
        if (!best) {
            std::cout << "  no order finished with the right answer\n";
            continue;
        }
        std::cout << "  best " << OrderNames(best->order, tables) << " " << FormatMs(*best->median_ms, 1) << " ms";

        const auto pick = [&](const char* label, bool Candidate::*flag) {
            const auto it = std::find_if(cands.begin(), cands.end(), [&](const Candidate& c) { return c.*flag; });
            std::cout << " | " << label << " ";
            if (it == cands.end() || !it->median_ms) {
                std::cout << "not timed";
            } else {
                std::cout << FormatMs(*it->median_ms, 1) << " ms (" << FormatMs(*it->median_ms / *best->median_ms, 2) << "x)";
            }
        };
        pick("classical", &Candidate::is_classical);
        if (job.learned) pick("learned", &Candidate::is_learned);
        std::cout << "\n";
    }
}

int Run(const Args& args) {
    const auto run_start = Clock::now();
    const size_t pool_size = args.pool_size ? args.pool_size : DefaultPoolSize(args.sf);

    // Built before the load: with a model path this loads the model, so a bad path fails fast.
    std::unique_ptr<Database> db;
    try {
#ifdef SHILMANDB_HAS_LIBTORCH
        const bool with_model = !args.join_model_path.empty();
        db = std::make_unique<Database>(args.db_file, pool_size, with_model, args.join_model_path, /*use_learned_eviction=*/false, /*eviction_model_path=*/"");
#else
        db = std::make_unique<Database>(args.db_file, pool_size);
#endif
    } catch (const std::exception& e) {
        // LibTorch tacks a C++ backtrace onto its message; the first line says what went wrong.
        const std::string what = e.what();
        Fail(what.substr(0, what.find('\n')));
    }

    std::cout << "=== Join headroom (SF=" << args.sf << ", pool " << pool_size << " pages, LRU, no indexes) ===\n"
              << "Queries: " << Join(args.queries, ",", kSame) << " | warmup " << args.warmup << ", runs "
              << args.runs;
    if (args.orders) std::cout << " | forced order " << Join(*args.orders, ",", kToString);
    if (!args.join_model_path.empty()) std::cout << " | join model " << args.join_model_path;
    std::cout << "\n\n";

    try {
        tpch::LoadTables(*db, args.data_dir);
    } catch (const std::exception& e) {
        Fail(e.what());
    }

    // Plan everything before opening any CSV, so a rejected --orders leaves no files behind.
    std::vector<QueryJob> jobs;
    for (const auto& tag : args.queries) {
        jobs.push_back(Prepare(*db, *tpch::FindQuery(tag), args));
    }

    const auto suffix = FileSuffix(args);
    const fs::path dir = args.out_dir;
    CsvFile orders(
        dir / ("orders_sf" + args.sf + suffix + ".csv"),
        "sf,query,order,order_slots,connected,is_classical,is_learned,est_cost,est_step_rows,"
        "actual_step_rows,table_rows,scan_rows,median_ms,min_ms,max_ms,runs,result_ok"
    );

    CsvFile operators(dir / ("operators_sf" + args.sf + suffix + ".csv"), "sf,query,order,op_index,depth,operator,rows,total_ms,self_ms");
    CsvFile choices(dir / ("choices_sf" + args.sf + suffix + ".csv"), "sf,query,chooser,order,source,model_order");
    Outputs out{args.sf, orders, operators, choices};

    int failures = 0;
    for (auto& job : jobs) {
        failures += SweepQuery(*db, job, args, out);
    }

    PrintSummary(jobs);

    const double secs = std::chrono::duration<double>(Clock::now() - run_start).count();
    std::cout << "\nWrote:\n";
    for (const auto* file : {&orders, &operators, &choices}) std::cout << "  " << file->Path().string() << "\n";
    std::cout << "Done in " << FormatMs(secs, 1) << " s\n";

    if (failures > 0) {
        std::cerr << failures << " order(s) failed to run; see FAILED lines above\n";
        return 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    const auto args = ParseArgs(argc, argv);
    try {
        return Run(args);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
