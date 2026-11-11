#pragma once

// Joins recorded traces into one phases trace for checking policy switching.
// Header-only, and it does not use any engine code.

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "eviction_sim.hpp"

namespace shilmandb::evsim {

// The policies a switcher may pick, in tie order.
[[nodiscard]] inline const std::vector<std::string>& SwitchCandidates() {
    static const std::vector<std::string> names = {"lru", "clock", "lru2_visit", "arc_visit", "lirs_visit"};
    return names;
}

struct LoadedTrace {
    std::string name;
    Trace trace;
    std::vector<Item> items;
};

// Reads dir/name.csv and dir/name/stream.csv, then maps the items onto the runs.
[[nodiscard]] inline LoadedTrace LoadRecorded(const std::string& dir, const std::string& name) {
    LoadedTrace loaded;
    loaded.name = name;
    loaded.items = LoadItems(dir + "/" + name + "/stream.csv");

    if (loaded.items.empty()) throw std::runtime_error(name + " has no items");
    loaded.trace = LoadTrace(dir + "/" + name + ".csv", EndFetch(loaded.items));

    MapItems(loaded.trace, loaded.items);
    return loaded;
}

// A slice of one trace's runs: the prefix before body_begin, then the body.
struct Piece {
    const Trace* trace;
    size_t body_begin;
    size_t body_end;
    std::string name;
};

// The runs from the first item to the end of the last item.
[[nodiscard]] inline Piece BodyOf(const LoadedTrace& t) {
    if (t.items.empty()) throw std::invalid_argument("trace has no items");
    return Piece{&t.trace, t.items.front().first_run, t.items.back().end_run, t.name};
}

struct PhasesTrace {
    Trace trace;
    std::vector<size_t> phase_start;  // first run of each phase
    std::vector<std::string> names;

    // The phase a run belongs to.
    [[nodiscard]] size_t PhaseOf(size_t run) const {
        const auto it = std::upper_bound(phase_start.begin(), phase_start.end(), run);
        return static_cast<size_t>(it - phase_start.begin()) - 1;
    }

    // The run where a phase ends (the next phase's start).
    [[nodiscard]] size_t PhaseEnd(size_t phase) const {
        return phase + 1 < phase_start.size() ? phase_start[phase + 1] : trace.runs.size();
    }
};

// Joins the bodies of pieces that share one prefix; refuses a piece whose prefix differs.
[[nodiscard]] inline PhasesTrace JoinPhases(const std::vector<Piece>& pieces) {
    if (pieces.empty()) throw std::invalid_argument("no pieces to join");
    for (const auto& piece : pieces) {
        if (piece.body_begin >= piece.body_end || piece.body_end > piece.trace->runs.size()) {
            throw std::invalid_argument("empty or bad body in " + piece.name);
        }
    }

    const Piece& first = pieces.front();
    PhasesTrace joined;
    size_t total_runs = 0;
    for (const auto& piece : pieces) total_runs += piece.body_end - piece.body_begin;
    joined.trace.runs.reserve(total_runs);
    uint64_t fetch = 0;

    const auto add = [&](const Run& run) {
        if (!joined.trace.runs.empty() && joined.trace.runs.back().page == run.page) {
            Run& last = joined.trace.runs.back();
            if (uint64_t{last.length} + run.length > UINT32_MAX) throw std::runtime_error("run is too long");
            last.length += run.length;
        } else {
            joined.trace.runs.push_back(Run{run.page, fetch, run.length});
        }
        fetch += run.length;
        joined.trace.page_limit = std::max(joined.trace.page_limit, run.page + 1);
    };

    for (const auto& piece : pieces) {
        const auto& runs = piece.trace->runs;
        bool same = piece.body_begin == first.body_begin;
        for (size_t i = 0; same && i < piece.body_begin; ++i) {
            same = runs[i].page == first.trace->runs[i].page && runs[i].length == first.trace->runs[i].length;
        }
        if (!same) throw std::runtime_error("prefix of " + piece.name + " differs from " + first.name);

        joined.phase_start.push_back(joined.trace.runs.size());
        joined.names.push_back(piece.name);
        for (size_t i = piece.body_begin; i < piece.body_end; ++i) add(runs[i]);
    }
    if (joined.trace.runs.size() >= UINT32_MAX) throw std::runtime_error("trace has too many runs");
    joined.trace.end_fetch = fetch;
    return joined;
}

// Runs p over runs [begin, end) and returns the misses; flags marks each missed run.
inline uint64_t ReplayRange(Policy& p, const Trace& t, size_t begin, size_t end, std::vector<uint8_t>* flags = nullptr) {
    uint64_t misses = 0;
    for (size_t r = begin; r < end; ++r) {
        if (p.Access(r, t.runs[r])) {
            ++misses;
            if (flags != nullptr) (*flags)[r] = 1;
        }
    }
    return misses;
}

// Records the latest run each page was seen in.
inline void NoteRuns(std::vector<uint32_t>& last_run, const Trace& t, size_t begin, size_t end) {
    for (size_t r = begin; r < end; ++r) last_run[t.runs[r].page] = static_cast<uint32_t>(r);
}

// A fresh policy that has seen the resident pages, least recently used first.
[[nodiscard]] inline std::unique_ptr<Policy> HandOver(const std::string& name, size_t pool, const SimContext& ctx, std::vector<uint32_t> residents, const std::vector<uint32_t>& last_run) {
    auto policy = MakePolicy(name, pool, ctx);
    std::sort(residents.begin(), residents.end(), [&](uint32_t a, uint32_t b) { return last_run[a] != last_run[b] ? last_run[a] < last_run[b] : a < b; });
    for (const uint32_t page : residents) {
        const uint32_t run = last_run[page];
        (void)policy->Access(run, ctx.trace->runs[run]);
    }
    return policy;
}

// Which candidate starts, and the (run, candidate) switches that follow.
struct Schedule {
    int start{0};
    std::vector<std::pair<size_t, int>> switches;
};

struct SwitchRun {
    uint64_t misses{0};
    std::vector<uint64_t> phase_misses;
    std::vector<std::vector<uint64_t>> occupancy;  // runs per [phase][candidate]
    size_t switches{0};
};

// Throws unless the schedule is in order, in range, and always changes candidate.
inline void CheckSchedule(const Schedule& schedule, size_t run_count, size_t candidate_count) {
    const auto in_range = [&](int c) { return c >= 0 && static_cast<size_t>(c) < candidate_count; };
    if (!in_range(schedule.start)) throw std::invalid_argument("start candidate out of range");
    int current = schedule.start;
    size_t previous = 0;
    bool first = true;

    for (const auto& [run, to] : schedule.switches) {
        if (run >= run_count) throw std::invalid_argument("switch run past the end of the trace");
        if (!first && run <= previous) throw std::invalid_argument("switch runs must strictly increase");
        if (!in_range(to)) throw std::invalid_argument("switch candidate out of range");
        if (to == current) throw std::invalid_argument("switch to the candidate already running");
        current = to;
        previous = run;
        first = false;
    }
}

// Replays the trace, handing the pool to a new candidate at each scheduled run.
[[nodiscard]] inline SwitchRun LiveReplay(const PhasesTrace& phases, size_t pool, const SimContext& ctx, const std::vector<std::string>& candidates, const Schedule& schedule) {
    if (ctx.trace != &phases.trace) throw std::invalid_argument("context trace is not the phases trace");
    const auto& runs = phases.trace.runs;
    CheckSchedule(schedule, runs.size(), candidates.size());

    SwitchRun out;
    out.phase_misses.assign(phases.phase_start.size(), 0);
    out.occupancy.assign(phases.phase_start.size(), std::vector<uint64_t>(candidates.size(), 0));

    int current = schedule.start;
    auto policy = MakePolicy(candidates[static_cast<size_t>(current)], pool, ctx);
    std::vector<uint32_t> last_run(phases.trace.page_limit, kNever);
    size_t next_switch = 0;
    for (size_t r = 0; r < runs.size(); ++r) {
        if (next_switch < schedule.switches.size() && schedule.switches[next_switch].first == r) {
            current = schedule.switches[next_switch].second;
            policy = HandOver(candidates[static_cast<size_t>(current)], pool, ctx, policy->ResidentPages(), last_run);
            ++next_switch;
            ++out.switches;
        }
        const size_t phase = phases.PhaseOf(r);
        if (policy->Access(r, runs[r])) {
            ++out.misses;
            ++out.phase_misses[phase];
        }
        ++out.occupancy[phase][static_cast<size_t>(current)];
        last_run[runs[r].page] = static_cast<uint32_t>(r);
    }
    return out;
}

// One policy over the whole trace; flags marks each missed run.
[[nodiscard]] inline SwitchRun FixedReplay(const PhasesTrace& phases, const std::string& name, size_t pool, const SimContext& ctx, std::vector<uint8_t>* flags = nullptr) {
    if (ctx.trace != &phases.trace) throw std::invalid_argument("context trace is not the phases trace");
    auto policy = MakePolicy(name, pool, ctx);
    if (flags != nullptr) flags->assign(phases.trace.runs.size(), 0);
    SwitchRun out;

    for (size_t phase = 0; phase < phases.phase_start.size(); ++phase) {
        out.phase_misses.push_back(ReplayRange(*policy, phases.trace, phases.phase_start[phase], phases.PhaseEnd(phase), flags));
        out.misses += out.phase_misses.back();
    }
    return out;
}


// What each candidate missed over the window before every decision.
struct Evidence {
    size_t window{0}, step{0};
    std::vector<size_t> at;                       // runs where a decision is made
    std::vector<std::vector<uint64_t>> misses;    // [decision][candidate]
};

// Replays each candidate cold over the window before every decision, counting its last three quarters.
[[nodiscard]] inline Evidence BuildEvidence(const Trace& trace, size_t pool, size_t window, const SimContext& ctx, const std::vector<std::string>& candidates) {
    if (window == 0 || window % 4 != 0) {
        throw std::invalid_argument("window " + std::to_string(window) + " must be a positive multiple of 4");
    }
    const size_t run_count = trace.runs.size();
    if (window >= run_count) throw std::invalid_argument("window " + std::to_string(window) + " leaves no decision");

    Evidence evidence;
    evidence.window = window;
    evidence.step = window / 2;
    const size_t warm = window / 4;

    for (size_t at = window; at < run_count; at += evidence.step) {
        std::vector<uint64_t> row;
        for (const auto& name : candidates) {
            auto policy = MakePolicy(name, pool, ctx);
            (void)ReplayRange(*policy, trace, at - window, at - window + warm);
            row.push_back(ReplayRange(*policy, trace, at - window + warm, at));
        }
        evidence.at.push_back(at);
        evidence.misses.push_back(std::move(row));
    }
    return evidence;
}

// The candidate with the fewest misses; ties go to current, then candidate order.
// Throws unless start is one of the evidence's candidates.
inline void CheckStart(const Evidence& evidence, int start) {
    const bool known = evidence.misses.empty() || static_cast<size_t>(start) < evidence.misses.front().size();
    if (start < 0 || !known) throw std::invalid_argument("start is not a candidate");
}

[[nodiscard]] inline size_t LeastMisses(const std::vector<uint64_t>& m, size_t current) {
    size_t best = current;
    for (size_t c = 0; c < m.size(); ++c) {
        if (m[c] < m[best]) best = c;
    }
    return best;
}

// Switches once the same other candidate beats the current one
// by threshold twice in a row.
[[nodiscard]] inline Schedule TwoWindowRule(const Evidence& evidence, int start, double threshold, int min_gap = 2) {
    CheckStart(evidence, start);
    Schedule schedule;
    schedule.start = start;

    size_t current = static_cast<size_t>(start);
    long previous_lead = -1;
    long last_switch = -static_cast<long>(min_gap);  // far enough back for a switch at decision 1

    for (size_t d = 0; d < evidence.at.size(); ++d) {
        const auto& m = evidence.misses[d];
        const size_t best = LeastMisses(m, current);
        const bool leads = best != current && static_cast<double>(m[best]) <= (1.0 - threshold) * static_cast<double>(m[current]);
        if (!leads) {
            previous_lead = -1;
            continue;
        }
        if (previous_lead == static_cast<long>(best) && static_cast<long>(d) - last_switch >= min_gap) {
            schedule.switches.emplace_back(evidence.at[d], static_cast<int>(best));
            current = best;
            previous_lead = -1;
            last_switch = static_cast<long>(d);
        } else {
            previous_lead = static_cast<long>(best);
        }
    }
    return schedule;
}

struct HedgeSettings {
    double eta;
    double alpha;
    double margin;
};

// One Hedge update of the log weights from a decision's misses.
// alpha mixes in a share of the uniform weights.
inline void HedgeStep(std::vector<double>& log_w, const std::vector<uint64_t>& m, double eta, double alpha) {
    const double lowest = static_cast<double>(*std::min_element(m.begin(), m.end()));
    const double scale = std::max(lowest, 1.0);
    for (size_t c = 0; c < log_w.size(); ++c) {
        const double loss = std::min(1.0, (static_cast<double>(m[c]) - lowest) / scale);
        log_w[c] -= eta * loss;
    }
    if (alpha > 0) {
        const double top = *std::max_element(log_w.begin(), log_w.end());
        double sum = 0;
        for (const double lw : log_w) sum += std::exp(lw - top);
        const double n = static_cast<double>(log_w.size());
        for (double& lw : log_w) lw = std::log((1.0 - alpha) * std::exp(lw - top) / sum + alpha / n);
    }
}

// Linear weights that add up to 1, from log weights.
[[nodiscard]] inline std::vector<double> Normalized(const std::vector<double>& log_w) {
    const double top = *std::max_element(log_w.begin(), log_w.end());
    std::vector<double> w;
    double sum = 0;
    for (const double lw : log_w) {
        w.push_back(std::exp(lw - top));
        sum += w.back();
    }
    for (double& x : w) x /= sum;
    return w;
}

// Normalized Hedge weights after each decision; alpha > 0 is Fixed Share.
[[nodiscard]] inline std::vector<std::vector<double>> HedgeWeights(const Evidence& evidence, double eta, double alpha) {
    std::vector<std::vector<double>> rows;
    if (evidence.misses.empty()) return rows;
    std::vector<double> log_w(evidence.misses.front().size(), 0.0);
    for (const auto& m : evidence.misses) {
        HedgeStep(log_w, m, eta, alpha);
        rows.push_back(Normalized(log_w));
    }
    return rows;
}

// Switches when the heaviest candidate's log weight leads the current one by log(margin).
[[nodiscard]] inline Schedule HedgeRule(const Evidence& evidence, int start, const HedgeSettings& settings) {
    CheckStart(evidence, start);
    Schedule schedule;
    schedule.start = start;

    if (evidence.misses.empty()) return schedule;
    size_t current = static_cast<size_t>(start);
    std::vector<double> log_w(evidence.misses.front().size(), 0.0);

    const double needed = std::log(settings.margin);
    for (size_t d = 0; d < evidence.at.size(); ++d) {
        HedgeStep(log_w, evidence.misses[d], settings.eta, settings.alpha);
        size_t leader = current;
        for (size_t c = 0; c < log_w.size(); ++c) {
            if (log_w[c] > log_w[leader]) leader = c;
        }
        if (leader != current && log_w[leader] - log_w[current] >= needed) {
            schedule.switches.emplace_back(evidence.at[d], static_cast<int>(leader));
            current = leader;
        }
    }
    return schedule;
}

struct HindsightResult {
    Schedule schedule;
    uint64_t misses{0};
};

// The cheapest schedule of whole segments, paying the real handover cost of each switch.
[[nodiscard]] inline HindsightResult Hindsight(const Trace& trace, size_t pool, size_t step, const SimContext& ctx, const std::vector<std::string>& candidates) {
    if (ctx.trace != &trace) throw std::invalid_argument("context trace is not the trace");
    if (step == 0) throw std::invalid_argument("step must be at least 1");
    if (candidates.empty()) throw std::invalid_argument("no candidates");

    struct State {
        uint64_t misses{0};
        Schedule schedule;
        std::unique_ptr<Policy> policy;
    };
    const size_t run_count = trace.runs.size();
    const size_t n = candidates.size();
    std::vector<uint32_t> last_run(trace.page_limit, kNever);

    std::vector<State> states(n);
    const size_t first_end = std::min(step, run_count);
    for (size_t c = 0; c < n; ++c) {
        states[c].schedule.start = static_cast<int>(c);
        states[c].policy = MakePolicy(candidates[c], pool, ctx);
        states[c].misses = ReplayRange(*states[c].policy, trace, 0, first_end);
    }
    NoteRuns(last_run, trace, 0, first_end);

    for (size_t begin = step; begin < run_count; begin += step) {
        const size_t end = std::min(begin + step, run_count);
        std::vector<std::vector<uint32_t>> residents(n);
        for (size_t p = 0; p < n; ++p) residents[p] = states[p].policy->ResidentPages();

        std::vector<State> next(n);
        for (size_t q = 0; q < n; ++q) {
            bool have = false;
            bool best_stays = false;
            for (size_t p = 0; p < n; ++p) {
                const bool stays = p == q;
                State candidate;
                candidate.policy = stays ? std::move(states[q].policy) : HandOver(candidates[q], pool, ctx, residents[p], last_run);
                candidate.misses = states[p].misses + ReplayRange(*candidate.policy, trace, begin, end);

                if (have) {
                    const State& best = next[q];
                    const size_t best_switches = best.schedule.switches.size();
                    const size_t switches = states[p].schedule.switches.size() + (stays ? 0 : 1);
                    bool better = candidate.misses < best.misses;
                    if (candidate.misses == best.misses) {
                        if (stays != best_stays) better = stays;
                        else better = switches < best_switches;
                    }
                    if (!better) continue;
                }
                candidate.schedule = states[p].schedule;
                if (!stays) candidate.schedule.switches.emplace_back(begin, static_cast<int>(q));
                next[q] = std::move(candidate);
                best_stays = stays;
                have = true;
            }
        }
        states = std::move(next);
        NoteRuns(last_run, trace, begin, end);
    }

    size_t winner = 0;
    for (size_t c = 1; c < n; ++c) {
        const bool fewer = states[c].misses < states[winner].misses;
        const bool tied_fewer_switches = states[c].misses == states[winner].misses && states[c].schedule.switches.size() < states[winner].schedule.switches.size();
        if (fewer || tied_fewer_switches) winner = c;
    }
    return HindsightResult{std::move(states[winner].schedule), states[winner].misses};
}

// Misses if every segment could use its best candidate for free; unreachable, only reported.
[[nodiscard]] inline uint64_t FreeSwitchBound(const std::vector<const std::vector<uint8_t>*>& flags, size_t step) {
    if (flags.empty()) throw std::invalid_argument("no candidates");
    if (step == 0) throw std::invalid_argument("step must be at least 1");
    const size_t length = flags.front()->size();
    for (const auto* f : flags) {
        if (f->size() != length) throw std::invalid_argument("flag lists differ in length");
    }
    uint64_t total = 0;
    for (size_t begin = 0; begin < length; begin += step) {
        const size_t end = std::min(begin + step, length);
        uint64_t best = UINT64_MAX;
        for (const auto* f : flags) {
            uint64_t m = 0;
            for (size_t r = begin; r < end; ++r) m += (*f)[r];
            best = std::min(best, m);
        }
        total += best;
    }
    return total;
}

}  // namespace shilmandb::evsim
