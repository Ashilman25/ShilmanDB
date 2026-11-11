#include "eviction_learned.hpp"
#include "eviction_sim.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <charconv>
#include <chrono>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// Replays one recorded trace under many policies and pool sizes, and writes the
// misses of every pass and every item as CSVs.

using namespace shilmandb::evsim;

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string trace, items, name, out_prefix, v2_weights, model, learned_name{"learned"}, page_owners, check_model;
    std::vector<size_t> pools;
    std::vector<std::string> policies;
    bool raw{false};
    uint32_t visit_window{64};
    size_t v2_max_frames{2048};
    uint64_t v2_clock_start{0};
    unsigned jobs{0};
    size_t sample_k{32};
    uint64_t sample_seed{1};
    bool time_decisions{false};
};

struct Cell {
    std::string policy;
    size_t pool{0};
    bool skipped{false};
    SimResult result;
    double seconds{0.0};
    std::vector<uint32_t> decision_ns;
    uint64_t decisions{0}, macs_per_candidate{0}, nodes_visited{0};
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

template <typename T>
[[nodiscard]] T ParseNumber(const std::string& flag, std::string_view s) {
    T value{};
    const char* end = s.data() + s.size();
    const auto [ptr, ec] = std::from_chars(s.data(), end, value);
    if (s.empty() || ec != std::errc() || ptr != end) {
        throw std::runtime_error(flag + " expects a number, got '" + std::string(s) + "'");
    }
    return value;
}

[[nodiscard]] Options ParseArgs(int argc, char** argv) {
    Options o;
    std::optional<std::string> policies;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--raw") { o.raw = true; continue; }
        if (flag == "--time-decisions") { o.time_decisions = true; continue; }
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + flag);
        const std::string value = argv[++i];
        if (flag == "--trace") o.trace = value;
        else if (flag == "--items") o.items = value;
        else if (flag == "--name") o.name = value;
        else if (flag == "--out-prefix") o.out_prefix = value;
        else if (flag == "--v2-weights") o.v2_weights = value;
        else if (flag == "--model") o.model = value;
        else if (flag == "--learned-name") o.learned_name = value;
        else if (flag == "--page-owners") o.page_owners = value;
        else if (flag == "--check-model") o.check_model = value;
        else if (flag == "--sample-k") {
            o.sample_k = ParseNumber<size_t>(flag, value);
            if (o.sample_k == 0) throw std::runtime_error("--sample-k must be at least 1");
        }
        else if (flag == "--sample-seed") o.sample_seed = ParseNumber<uint64_t>(flag, value);
        else if (flag == "--policies") policies = value;
        else if (flag == "--pools") {
            for (const auto& p : SplitCommas(value)) {
                const auto pool = ParseNumber<size_t>(flag, p);
                if (pool == 0) throw std::runtime_error("a pool must be at least 1");
                o.pools.push_back(pool);
            }
        }
        else if (flag == "--visit-window") o.visit_window = ParseNumber<uint32_t>(flag, value);
        else if (flag == "--v2-max-frames") o.v2_max_frames = ParseNumber<size_t>(flag, value);
        else if (flag == "--v2-clock-start") o.v2_clock_start = ParseNumber<uint64_t>(flag, value);
        else if (flag == "--jobs") o.jobs = ParseNumber<unsigned>(flag, value);
        else throw std::runtime_error("unknown flag " + flag);
    }
    if (!o.check_model.empty()) return o;
    for (const auto* required : {&o.trace, &o.items, &o.name, &o.out_prefix}) {
        if (required->empty()) throw std::runtime_error("--trace, --items, --name, --pools and --out-prefix are required");
    }
    if (o.pools.empty()) throw std::runtime_error("--trace, --items, --name, --pools and --out-prefix are required");

    std::vector<std::string> all = PolicyNames();
    all.insert(all.end(), LearnedPolicyNames().begin(), LearnedPolicyNames().end());
    if (policies) {
        o.policies = SplitCommas(*policies);
    } else {
        for (const auto& n : PolicyNames()) {
            if (n != "v2" || !o.v2_weights.empty()) o.policies.push_back(n);
        }
    }
    for (const auto& n : o.policies) {
        if (std::find(all.begin(), all.end(), n) == all.end()) throw std::runtime_error("unknown policy " + n);
    }
    if (std::find(o.policies.begin(), o.policies.end(), "v2") != o.policies.end() && o.v2_weights.empty()) {
        throw std::runtime_error("v2 needs --v2-weights");
    }
    if (o.jobs == 0) o.jobs = std::max(1u, std::thread::hardware_concurrency());
    return o;
}

// The cell as a table of passes: every pass of the items file, in order.
struct PassRow {
    int pass{0};
    bool warmup{false};
    uint64_t items{0}, fetches{0}, runs{0}, misses{0}, query_misses{0}, lookup_misses{0};
};

[[nodiscard]] std::vector<PassRow> SumPasses(const std::vector<Item>& items, const SimResult& r) {
    std::vector<PassRow> rows;
    std::map<int, size_t> index;
    for (size_t i = 0; i < items.size(); ++i) {
        const auto [it, fresh] = index.emplace(items[i].pass, rows.size());
        if (fresh) {
            rows.emplace_back();
            rows.back().pass = items[i].pass;
            rows.back().warmup = items[i].warmup;
        }
        PassRow& row = rows[it->second];
        ++row.items;
        row.fetches += items[i].trace_end - items[i].trace_start;
        row.runs += items[i].end_run - items[i].first_run;
        row.misses += r.item_misses[i];
        (items[i].kind == "lookup" ? row.lookup_misses : row.query_misses) += r.item_misses[i];
    }
    return rows;
}

// Median, mean and 90th percentile of the decision times.
struct DecisionStats {
    double median{0}, mean{0}, p90{0};
};

[[nodiscard]] DecisionStats SummarizeDecisions(std::vector<uint32_t> ns) {
    DecisionStats s;
    if (ns.empty()) return s;
    std::sort(ns.begin(), ns.end());
    const size_t n = ns.size();
    s.median = n % 2 == 1 ? ns[n / 2] : (static_cast<double>(ns[n / 2 - 1]) + ns[n / 2]) / 2.0;
    s.p90 = ns[static_cast<size_t>(0.9 * static_cast<double>(n - 1))];
    double sum = 0;
    for (const uint32_t v : ns) sum += v;
    s.mean = sum / static_cast<double>(n);
    return s;
}

const char* FeatureSetName(FeatureSet set) {
    return set == FeatureSet::BASIC ? "basic" : set == FeatureSet::RICHER_NO_OWNER ? "richer_no_owner" : "richer";
}

int Main(int argc, char** argv) {
    const Options o = ParseArgs(argc, argv);

    if (!o.check_model.empty()) {
        const LearnedModel model = LoadLearnedModel(o.check_model);
        const ModelScorer scorer(model);
        std::cout << "max ref error " << ModelMaxRefError(model) << "\n"
                  << "macs per candidate " << scorer.MacsPerCandidate() << "\n"
                  << "inputs " << model.inputs << "\n"
                  << "pool " << model.pool << "\n";
        return 0;
    }

    const auto has = [&](const char* name) { return std::find(o.policies.begin(), o.policies.end(), name) != o.policies.end(); };
    if (o.time_decisions && !has("learned")) throw std::runtime_error("--time-decisions needs the learned policy (nothing to time)");

    // The model and the owner map are read before anything else, so a bad input stops the run early.
    LearnedModel model;
    if (has("learned")) {
        if (o.model.empty()) throw std::runtime_error("learned needs --model");
        model = LoadLearnedModel(o.model);
        if (o.pools.size() != 1 || o.pools[0] != model.pool) {
            throw std::runtime_error("the model is for pool " + std::to_string(model.pool) + ", not " + (o.pools.size() == 1 ? std::to_string(o.pools[0]) : "several pools"));
        }
        if (model.features == FeatureSet::RICHER && o.page_owners.empty()) {
            throw std::runtime_error("a richer model needs --page-owners");
        }
    }
    OwnerMap owners;
    if (!o.page_owners.empty()) owners = LoadPageOwners(o.page_owners);

    auto items = LoadItems(o.items);
    if (items.empty()) throw std::runtime_error("no items in " + o.items);

    const auto load_start = Clock::now();
    const Trace trace = LoadTrace(o.trace, EndFetch(items), o.raw);
    MapItems(trace, items);
    std::vector<uint32_t> next_use;
    if (!o.page_owners.empty()) CheckTraceCovered(owners, trace);
    if (has("belady") || has("sampled_belady") || has("sampled_label")) next_use = NextUse(trace);
    const double load_s = std::chrono::duration<double>(Clock::now() - load_start).count();
    std::cout << "trace: " << trace.runs.size() << " runs, " << trace.end_fetch << " fetches, " << items.size() << " items, loaded in " << std::fixed << std::setprecision(2) << load_s << " s\n";

    V2Weights v2;
    const bool has_v2 = std::find(o.policies.begin(), o.policies.end(), "v2") != o.policies.end();
    if (has_v2) {
        v2 = LoadV2Weights(o.v2_weights);
        const double err = V2MaxRefError(v2);
        if (err >= 1e-4) throw std::runtime_error("v2 weights disagree with their reference scores, max error " + std::to_string(err));
    }

    SimContext ctx;
    ctx.trace = &trace;
    ctx.next_use = next_use.empty() ? nullptr : &next_use;
    ctx.v2 = has_v2 ? &v2 : nullptr;
    ctx.v2_clock_start = o.v2_clock_start;
    ctx.visit_window = o.visit_window;

    LearnedContext lctx;
    lctx.owners = o.page_owners.empty() ? nullptr : &owners;
    lctx.model = has("learned") ? &model : nullptr;
    lctx.config.k = o.sample_k;
    lctx.config.seed = o.sample_seed;
    lctx.config.time_decisions = o.time_decisions;
    if (has("sampled_label")) {
        for (const size_t pool : o.pools) lctx.belady_hits[pool] = BeladyHits(trace, pool, next_use);
    }

    // Policies in PolicyNames() order, then the learned ones, then pool order.
    std::vector<std::string> order = PolicyNames();
    order.insert(order.end(), LearnedPolicyNames().begin(), LearnedPolicyNames().end());
    std::vector<Cell> cells;
    for (const auto& name : order) {
        if (!has(name.c_str())) continue;
        for (const size_t pool : o.pools) {
            Cell c;
            c.policy = name;
            c.pool = pool;
            c.skipped = name == "v2" && pool > o.v2_max_frames;
            cells.push_back(std::move(c));
        }
    }

    std::atomic<size_t> next_cell{0};
    std::exception_ptr failure;
    std::mutex failure_mutex;
    const auto worker = [&] {
        while (true) {
            const size_t i = next_cell.fetch_add(1);
            if (i >= cells.size()) return;
            Cell& c = cells[i];
            if (c.skipped) continue;
            try {
                const auto start = Clock::now();
                auto policy = MakeAnyPolicy(c.policy, c.pool, ctx, lctx);
                c.result = Simulate(*policy, trace, items);
                if (const auto* sampled = dynamic_cast<const SampledPolicy*>(policy.get())) {
                    c.decision_ns = sampled->DecisionNs();
                    c.decisions = sampled->Decisions();
                    if (const ModelScorer* scorer = sampled->Scorer()) {
                        c.macs_per_candidate = scorer->MacsPerCandidate();
                        c.nodes_visited = scorer->NodesVisited();
                    }
                }
                c.seconds = std::chrono::duration<double>(Clock::now() - start).count();
            } catch (...) {
                const std::lock_guard<std::mutex> lock(failure_mutex);
                if (!failure) failure = std::current_exception();
                next_cell = cells.size();
                return;
            }
        }
    };
    std::vector<std::thread> threads;
    const unsigned n_threads = static_cast<unsigned>(std::min<size_t>(o.jobs, std::max<size_t>(1, cells.size())));
    for (unsigned t = 0; t < n_threads; ++t) threads.emplace_back(worker);
    for (auto& t : threads) t.join();
    if (failure) std::rethrow_exception(failure);

    std::ofstream passes(o.out_prefix + "_passes.csv"), per_item(o.out_prefix + "_items.csv"),
        cell_csv(o.out_prefix + "_cells.csv");
    if (!passes || !per_item || !cell_csv) throw std::runtime_error("cannot write under " + o.out_prefix);
    passes << "trace,policy,pool,pass,warmup,items,fetches,runs,misses,query_misses,lookup_misses\n";
    per_item << "trace,policy,pool,pass,seq,kind,query,misses\n";
    cell_csv << "trace,policy,pool,evictions,seconds\n";
    const std::string learned_label = o.learned_name;
    const auto shown = [&](const std::string& policy) { return policy == "learned" ? learned_label : policy; };

    int first_timed_pass = -1;
    for (const auto& item : items) {
        if (!item.warmup && (first_timed_pass < 0 || item.pass < first_timed_pass)) first_timed_pass = item.pass;
    }

    for (const Cell& c : cells) {
        if (c.skipped) {
            std::cout << "skip v2 at pool " << c.pool << " (over --v2-max-frames)\n";
            continue;
        }
        for (const PassRow& r : SumPasses(items, c.result)) {
            passes << o.name << ',' << shown(c.policy) << ',' << c.pool << ',' << r.pass << ',' << (r.warmup ? 1 : 0) << ','
                   << r.items << ',' << r.fetches << ',' << r.runs << ',' << r.misses << ','
                   << r.query_misses << ',' << r.lookup_misses << '\n';
        }
        uint64_t first_timed_misses = 0;
        for (size_t i = 0; i < items.size(); ++i) {
            if (items[i].warmup) continue;
            per_item << o.name << ',' << shown(c.policy) << ',' << c.pool << ',' << items[i].pass << ',' << items[i].seq << ',' << items[i].kind << ',' << items[i].query << ',' << c.result.item_misses[i] << '\n';
            if (items[i].pass == first_timed_pass) first_timed_misses += c.result.item_misses[i];
        }
        cell_csv << o.name << ',' << shown(c.policy) << ',' << c.pool << ',' << c.result.evictions << ',' << std::fixed << std::setprecision(4) << c.seconds << '\n';
        std::cout << shown(c.policy) << ' ' << c.pool << ": " << first_timed_misses << " misses, " << std::setprecision(2) << c.seconds << " s\n";
    }
    if (o.time_decisions) {
        std::ofstream timing(o.out_prefix + "_timing.csv");
        if (!timing) throw std::runtime_error("cannot write under " + o.out_prefix);
        timing << "trace,policy,pool,k,decisions,median_ns,mean_ns,p90_ns,macs_per_decision,nodes_per_decision,features,model\n";

        for (const Cell& c : cells) {
            if (c.policy != "learned") continue;
            const DecisionStats st = SummarizeDecisions(c.decision_ns);
            const double nodes = c.decisions == 0 ? 0.0 : static_cast<double>(c.nodes_visited) / static_cast<double>(c.decisions);
            timing << o.name << ',' << shown(c.policy) << ',' << c.pool << ',' << o.sample_k << ',' << c.decisions << ','
                   << std::fixed << std::setprecision(1) << st.median << ',' << st.mean << ',' << st.p90 << ','
                   << std::setprecision(0) << static_cast<double>(c.macs_per_candidate * std::min<uint64_t>(o.sample_k, c.pool)) << ','
                   << std::setprecision(3) << nodes << ',' << FeatureSetName(model.features) << ',' << o.model << '\n';
        }
        timing.close();
        if (!timing) throw std::runtime_error("failed writing " + o.out_prefix + "_timing.csv");
    }
    for (auto* out : {&passes, &per_item, &cell_csv}) out->close();
    for (const auto& [stream, suffix] : {std::pair<const std::ofstream*, const char*>{&passes, "_passes.csv"}, {&per_item, "_items.csv"}, {&cell_csv, "_cells.csv"}}) {
        if (!*stream) throw std::runtime_error("failed writing " + o.out_prefix + suffix);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return Main(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "eviction_sim: " << e.what() << "\n";
        return 1;
    }
}
