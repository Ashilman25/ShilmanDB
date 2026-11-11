#include "eviction_sim.hpp"
#include "eviction_switch.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// Replays a joined phases trace under every fixed policy and several switchers,
// and writes the misses, phase splits, occupancy and switches as CSVs.

using namespace shilmandb::evsim;

namespace {

using Clock = std::chrono::steady_clock;

struct Window {
    int k;
    size_t size;
};

struct HedgePair {
    double eta;
    double alpha;
};

struct Options {
    std::string traces_dir, name, out_prefix, start{"lirs_visit"};
    std::vector<std::string> components;
    std::vector<size_t> pools;
    std::vector<Window> windows;
    std::vector<HedgePair> hedges;
    double margin{2.0}, threshold{0.05};
    uint32_t visit_window{64};
    unsigned jobs{0};
    bool describe{false}, validate{false};
};

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

[[nodiscard]] std::pair<std::string, std::string> SplitColon(const std::string& flag, const std::string& s) {
    const size_t colon = s.find(':');
    if (colon == std::string::npos) throw std::runtime_error(flag + " expects A:B, got '" + s + "'");
    return {s.substr(0, colon), s.substr(colon + 1)};
}

[[nodiscard]] double ParseDouble(const std::string& flag, const std::string& s) {
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (s.empty() || end != s.c_str() + s.size() || !std::isfinite(v)) {
        throw std::runtime_error(flag + " expects a number, got '" + s + "'");
    }
    return v;
}

[[nodiscard]] size_t ParseCount(const std::string& flag, const std::string& s) {
    const bool digits = !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); });
    errno = 0;
    const unsigned long long v = digits ? std::strtoull(s.c_str(), nullptr, 10) : 0;
    if (!digits || errno == ERANGE || v > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error(flag + " expects a whole number, got '" + s + "'");
    }
    return static_cast<size_t>(v);
}

// A whole number that must also fit in T.
template <typename T>
[[nodiscard]] T ParseCountAs(const std::string& flag, const std::string& s) {
    const size_t v = ParseCount(flag, s);
    if (v > static_cast<size_t>(std::numeric_limits<T>::max())) {
        throw std::runtime_error(flag + " value '" + s + "' is too large");
    }
    return static_cast<T>(v);
}

// Reads the flags and checks them before any trace is loaded.
[[nodiscard]] Options ParseArgs(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--describe") { o.describe = true; continue; }
        if (flag == "--validate") { o.validate = true; continue; }
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + flag);

        const std::string value = argv[++i];
        if (flag == "--traces-dir") o.traces_dir = value;
        else if (flag == "--components") o.components = SplitCommas(value);
        else if (flag == "--name") o.name = value;
        else if (flag == "--out-prefix") o.out_prefix = value;
        else if (flag == "--start") o.start = value;
        else if (flag == "--pools") {
            for (const auto& p : SplitCommas(value)) {
                const size_t pool = ParseCount(flag, p);
                if (pool == 0) throw std::runtime_error("a pool must be at least 1");
                o.pools.push_back(pool);
            }
        } else if (flag == "--windows") {
            for (const auto& w : SplitCommas(value)) {
                const auto [k, size] = SplitColon(flag, w);
                const int share = ParseCountAs<int>(flag, k);
                if (share == 0) throw std::runtime_error("a window share must be at least 1");
                o.windows.push_back(Window{share, ParseCount(flag, size)});
            }
        } else if (flag == "--hedge") {
            for (const auto& h : SplitCommas(value)) {
                const auto [eta, alpha] = SplitColon(flag, h);
                const HedgePair pair{ParseDouble(flag, eta), ParseDouble(flag, alpha)};
                if (!(pair.eta > 0)) throw std::runtime_error("hedge eta must be above 0, got '" + eta + "'");
                if (!(pair.alpha >= 0 && pair.alpha < 1)) {
                    throw std::runtime_error("hedge alpha must be in [0, 1), got '" + alpha + "'");
                }
                o.hedges.push_back(pair);
            }
        } else if (flag == "--margin") o.margin = ParseDouble(flag, value);
        else if (flag == "--threshold") o.threshold = ParseDouble(flag, value);
        else if (flag == "--visit-window") o.visit_window = ParseCountAs<uint32_t>(flag, value);
        else if (flag == "--jobs") o.jobs = ParseCountAs<unsigned>(flag, value);
        else throw std::runtime_error("unknown flag " + flag);
    }
    if (o.traces_dir.empty() || o.components.empty() || o.name.empty() || o.out_prefix.empty()) {
        throw std::runtime_error("--traces-dir, --components, --name and --out-prefix are required");
    }
    if (o.describe && o.validate) throw std::runtime_error("--describe and --validate don't go together");
    if (!o.describe && o.pools.empty()) throw std::runtime_error("--pools is required");
    if (!o.describe && !o.validate && o.windows.empty()) throw std::runtime_error("replay needs --windows");
    if (!(o.margin >= 1)) throw std::runtime_error("--margin must be at least 1");
    if (!(o.threshold >= 0 && o.threshold < 1)) throw std::runtime_error("--threshold must be in [0, 1)");
    const auto& names = SwitchCandidates();
    if (std::find(names.begin(), names.end(), o.start) == names.end()) {
        throw std::runtime_error("--start " + o.start + " is not a candidate");
    }
    if (o.jobs == 0) o.jobs = std::max(1u, std::thread::hardware_concurrency());
    return o;
}

// Runs fn on every index across jobs threads.
// The first failure is rethrown once they stop.
void RunUnits(size_t count, unsigned jobs, const std::function<void(size_t)>& fn) {
    std::atomic<size_t> next{0};
    std::exception_ptr failure;
    std::mutex failure_mutex;

    const auto worker = [&] {
        while (true) {
            const size_t i = next.fetch_add(1);
            if (i >= count) return;
            try {
                fn(i);
            } catch (...) {
                const std::lock_guard<std::mutex> lock(failure_mutex);
                if (!failure) failure = std::current_exception();
                next = count;
                return;
            }
        }
    };

    std::vector<std::thread> threads;
    const unsigned n = static_cast<unsigned>(std::min<size_t>(jobs, std::max<size_t>(1, count)));
    for (unsigned t = 0; t < n; ++t) threads.emplace_back(worker);
    for (auto& t : threads) t.join();
    if (failure) std::rethrow_exception(failure);
}

std::mutex g_print_mutex;

// Prints one finished unit and how long it took.
void Progress(const std::string& label, Clock::time_point start) {
    const double s = std::chrono::duration<double>(Clock::now() - start).count();
    const std::lock_guard<std::mutex> lock(g_print_mutex);
    std::cout << label << ": " << std::fixed << std::setprecision(2) << s << " s\n" << std::defaultfloat;
}

// Shortest readable form of a number, empty when it doesn't apply.
template <typename T>
[[nodiscard]] std::string Fmt(const std::optional<T>& v) {
    if (!v) return "";
    std::ostringstream out;
    out << *v;
    return out.str();
}

// One line of the results table, plus what the other CSVs need.
struct ResultRow {
    std::string switcher, policy;
    size_t pool{0};
    std::optional<int> window_k;
    std::optional<size_t> window;
    std::optional<double> eta, alpha, margin, threshold;
    uint64_t misses{0};
    std::optional<size_t> switches;
    std::optional<SwitchRun> run;  // phase misses and occupancy
    Schedule schedule;             // only for switchers
    bool is_switcher{false};
};

// Refuses a window that isn't a multiple of 4 or leaves no decision.
void CheckWindows(const Options& o, const PhasesTrace& phases) {
    for (const auto& w : o.windows) {
        if (w.size == 0 || w.size % 4 != 0) {
            throw std::runtime_error("window " + std::to_string(w.size) + " must be a positive multiple of 4");
        }
        if (w.size >= phases.trace.runs.size()) {
            throw std::runtime_error("window " + std::to_string(w.size) + " leaves no decision");
        }
    }
}

// Loads each named component from the traces folder.
[[nodiscard]] std::vector<LoadedTrace> LoadComponents(const Options& o, const std::vector<std::string>& names) {
    std::vector<LoadedTrace> loaded;
    for (const auto& name : names) loaded.push_back(LoadRecorded(o.traces_dir, name));
    return loaded;
}

// Writes where each phase starts and how long it is.
void WriteLayout(const Options& o, const PhasesTrace& phases) {
    std::ofstream out(o.out_prefix + "_layout.csv");
    if (!out) throw std::runtime_error("cannot write under " + o.out_prefix);
    out << "trace,phase,component,first_run,runs\n";

    for (size_t p = 0; p < phases.phase_start.size(); ++p) {
        out << o.name << ',' << p << ',' << phases.names[p] << ',' << phases.phase_start[p] << ','
            << phases.PhaseEnd(p) - phases.phase_start[p] << '\n';
    }
    out.close();
    if (!out) throw std::runtime_error("failed writing " + o.out_prefix + "_layout.csv");
}

// Each distinct component alone, through every candidate and Belady at every pool.
void Validate(const Options& o) {
    std::vector<std::string> distinct;
    for (const auto& c : o.components) {
        if (std::find(distinct.begin(), distinct.end(), c) == distinct.end()) distinct.push_back(c);
    }
    const auto loaded = LoadComponents(o, distinct);
    std::vector<std::vector<uint32_t>> next_use;
    for (const auto& t : loaded) next_use.push_back(NextUse(t.trace));

    std::vector<std::string> policies = SwitchCandidates();
    policies.push_back("belady");
    const size_t per_component = o.pools.size() * policies.size();
    std::vector<std::string> lines(loaded.size() * per_component);

    const auto total_start = Clock::now();
    RunUnits(lines.size(), o.jobs, [&](size_t u) {
        const auto start = Clock::now();
        const size_t c = u / per_component, rest = u % per_component;
        const size_t pool = o.pools[rest / policies.size()];
        const std::string& policy = policies[rest % policies.size()];
        SimContext ctx;
        ctx.trace = &loaded[c].trace;
        ctx.next_use = &next_use[c];
        ctx.visit_window = o.visit_window;
        auto p = MakePolicy(policy, pool, ctx);
        const SimResult result = Simulate(*p, loaded[c].trace, loaded[c].items);

        std::map<int, std::pair<bool, uint64_t>> passes;
        for (size_t i = 0; i < loaded[c].items.size(); ++i) {
            auto& entry = passes[loaded[c].items[i].pass];
            entry.first = loaded[c].items[i].warmup;
            entry.second += result.item_misses[i];
        }
        std::ostringstream out;
        for (const auto& [pass, info] : passes) {
            out << loaded[c].name << ',' << policy << ',' << pool << ',' << pass << ',' << (info.first ? 1 : 0) << ',' << info.second << '\n';
        }
        lines[u] = out.str();
        Progress("validate " + loaded[c].name + " " + policy + " pool " + std::to_string(pool), start);
    });

    std::ofstream out(o.out_prefix + "_validation.csv");
    if (!out) throw std::runtime_error("cannot write under " + o.out_prefix);
    out << "trace,policy,pool,pass,warmup,misses\n";
    for (const auto& l : lines) out << l;
    out.close();
    if (!out) throw std::runtime_error("failed writing " + o.out_prefix + "_validation.csv");
    Progress("total", total_start);
}

[[nodiscard]] ResultRow FromRun(const std::string& switcher, const std::string& policy, size_t pool, SwitchRun run) {
    ResultRow row;
    row.switcher = switcher;
    row.policy = policy;
    row.pool = pool;
    row.misses = run.misses;
    row.run = std::move(run);
    return row;
}

// The rows for one pool and window: free bound, hindsight, two-window, then each hedge pair.
[[nodiscard]] std::vector<ResultRow> WindowRows(const Options& o, const PhasesTrace& phases, const SimContext& ctx, size_t pool, const Window& w, const std::vector<const std::vector<uint8_t>*>& flags) {
    const auto& names = SwitchCandidates();
    const int start = static_cast<int>(std::find(names.begin(), names.end(), o.start) - names.begin());
    const size_t step = w.size / 2;
    std::vector<ResultRow> rows;

    const auto tag = [&](ResultRow& row) {
        row.window_k = w.k;
        row.window = w.size;
    };
    const auto live = [&](const std::string& switcher, const Schedule& schedule) {
        ResultRow row = FromRun(switcher, "", pool, LiveReplay(phases, pool, ctx, names, schedule));
        row.schedule = schedule;
        row.is_switcher = true;
        row.switches = row.run->switches;
        tag(row);
        return row;
    };

    ResultRow bound;
    bound.switcher = "free_bound";
    bound.pool = pool;
    bound.misses = FreeSwitchBound(flags, step);
    tag(bound);
    rows.push_back(std::move(bound));

    const HindsightResult best = Hindsight(phases.trace, pool, step, ctx, names);
    ResultRow hindsight = live("hindsight", best.schedule);
    if (hindsight.misses != best.misses) throw std::runtime_error("hindsight bookkeeping mismatch");
    rows.push_back(std::move(hindsight));

    const Evidence evidence = BuildEvidence(phases.trace, pool, w.size, ctx, names);
    ResultRow two = live("two_window", TwoWindowRule(evidence, start, o.threshold));
    two.threshold = o.threshold;
    rows.push_back(std::move(two));

    for (const auto& h : o.hedges) {
        ResultRow row = live(h.alpha > 0 ? "fixed_share" : "hedge", HedgeRule(evidence, start, HedgeSettings{h.eta, h.alpha, o.margin}));
        row.eta = h.eta;
        row.alpha = h.alpha;
        row.margin = o.margin;
        rows.push_back(std::move(row));
    }
    return rows;
}

// Writes the results, phase, occupancy and switch tables.
void WriteReplay(const Options& o, const PhasesTrace& phases, const std::vector<ResultRow>& rows) {
    const auto& names = SwitchCandidates();
    std::ofstream results(o.out_prefix + "_results.csv"), by_phase(o.out_prefix + "_phases.csv"), occupancy(o.out_prefix + "_occupancy.csv"), switches(o.out_prefix + "_switches.csv");
    
    if (!results || !by_phase || !occupancy || !switches) throw std::runtime_error("cannot write under " + o.out_prefix);
    results << "trace,switcher,policy,pool,window_k,window,eta,alpha,margin,threshold,misses,switches\n";

    const std::string head = "trace,switcher,policy,pool,window_k,eta,alpha";
    by_phase << head << ",phase,component,misses\n";
    occupancy << head << ",phase,candidate,runs\n";
    switches << head << ",run,from,to\n";

    for (const ResultRow& r : rows) {
        results << o.name << ',' << r.switcher << ',' << r.policy << ',' << r.pool << ',' << Fmt(r.window_k) << ','
                << Fmt(r.window) << ',' << Fmt(r.eta) << ',' << Fmt(r.alpha) << ',' << Fmt(r.margin) << ','
                << Fmt(r.threshold) << ',' << r.misses << ',' << Fmt(r.switches) << '\n';
        if (!r.run) continue;
        const std::string key = o.name + ',' + r.switcher + ',' + r.policy + ',' + std::to_string(r.pool) + ',' + Fmt(r.window_k) + ',' + Fmt(r.eta) + ',' + Fmt(r.alpha);
        for (size_t p = 0; p < r.run->phase_misses.size(); ++p) {
            by_phase << key << ',' << p << ',' << phases.names[p] << ',' << r.run->phase_misses[p] << '\n';
        }
        if (!r.is_switcher) continue;
        for (size_t p = 0; p < r.run->occupancy.size(); ++p) {
            for (size_t c = 0; c < names.size(); ++c) {
                occupancy << key << ',' << p << ',' << names[c] << ',' << r.run->occupancy[p][c] << '\n';
            }
        }
        int current = r.schedule.start;
        for (const auto& [run, to] : r.schedule.switches) {
            switches << key << ',' << run << ',' << names[static_cast<size_t>(current)] << ',' << names[static_cast<size_t>(to)] << '\n';
            current = to;
        }
    }
    for (auto* out : {&results, &by_phase, &occupancy, &switches}) out->close();
    if (!results || !by_phase || !occupancy || !switches) throw std::runtime_error("failed writing under " + o.out_prefix);
}

void Replay(const Options& o, const PhasesTrace& phases) {
    CheckWindows(o, phases);
    const auto& names = SwitchCandidates();
    const size_t n_pools = o.pools.size(), n_cand = names.size();
    const auto next_use = NextUse(phases.trace);

    SimContext ctx;
    ctx.trace = &phases.trace;
    ctx.next_use = &next_use;
    ctx.visit_window = o.visit_window;
    const auto total_start = Clock::now();

    // Fixed replays first, keeping their missed-run flags for the free bound.
    std::vector<std::vector<uint8_t>> flags(n_pools * n_cand);
    std::vector<ResultRow> fixed(n_pools * (n_cand + 1));
    RunUnits(fixed.size(), o.jobs, [&](size_t u) {
        const auto start = Clock::now();
        const size_t pool_i = u / (n_cand + 1), c = u % (n_cand + 1);
        const bool belady = c == n_cand;
        const std::string policy = belady ? "belady" : names[c];
        SwitchRun run = FixedReplay(phases, policy, o.pools[pool_i], ctx, belady ? nullptr : &flags[pool_i * n_cand + c]);
        fixed[u] = FromRun(belady ? "belady" : "fixed", policy, o.pools[pool_i], std::move(run));
        Progress("fixed " + policy + " pool " + std::to_string(o.pools[pool_i]), start);
    });

    std::vector<std::vector<ResultRow>> windowed(n_pools * o.windows.size());
    RunUnits(windowed.size(), o.jobs, [&](size_t u) {
        const auto start = Clock::now();
        const size_t pool_i = u / o.windows.size();
        const Window& w = o.windows[u % o.windows.size()];
        std::vector<const std::vector<uint8_t>*> pool_flags;
        for (size_t c = 0; c < n_cand; ++c) pool_flags.push_back(&flags[pool_i * n_cand + c]);
        windowed[u] = WindowRows(o, phases, ctx, o.pools[pool_i], w, pool_flags);
        Progress("window " + std::to_string(w.k) + ":" + std::to_string(w.size) + " pool " + std::to_string(o.pools[pool_i]), start);
    });

    std::vector<ResultRow> all;
    for (size_t p = 0; p < n_pools; ++p) {
        for (size_t c = 0; c <= n_cand; ++c) all.push_back(std::move(fixed[p * (n_cand + 1) + c]));
        for (size_t w = 0; w < o.windows.size(); ++w) {
            for (auto& r : windowed[p * o.windows.size() + w]) all.push_back(std::move(r));
        }
    }
    WriteReplay(o, phases, all);
    Progress("total", total_start);
}

int Main(int argc, char** argv) {
    const Options o = ParseArgs(argc, argv);
    if (o.validate) {
        Validate(o);
        return 0;
    }
    // The loaded traces are freed once joined; only the phases trace is kept.
    const PhasesTrace phases = [&] {
        const auto loaded = LoadComponents(o, o.components);
        std::vector<Piece> pieces;
        for (const auto& t : loaded) pieces.push_back(BodyOf(t));
        return JoinPhases(pieces);
    }();
    
    std::cout << "phases trace: " << phases.trace.runs.size() << " runs, " << phases.phase_start.size() << " phases\n";
    WriteLayout(o, phases);
    if (!o.describe) Replay(o, phases);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return Main(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "eviction_switch: " << e.what() << "\n";
        return 1;
    }
}
