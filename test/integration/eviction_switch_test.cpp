#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "eviction_switch.hpp"

namespace shilmandb::evsim {
namespace {

using RunList = std::vector<std::pair<uint32_t, uint32_t>>;  // (page, fetches)

std::string TempPath(const std::string& name) {
    return (std::filesystem::temp_directory_path() /
            ("shilmandb_evswitch_" + name + "_" + std::to_string(::getpid())))
        .string();
}

// Runs back to back from fetch 0.
Trace TraceOfRuns(const RunList& runs) {
    std::vector<std::pair<uint64_t, uint32_t>> lines;
    uint64_t ts = 0;
    for (const auto& [page, length] : runs) {
        lines.emplace_back(ts, page);
        ts += length;
    }
    return MakeTrace(lines, ts);
}

RunList Concat(RunList a, const RunList& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

const RunList kPrefix = {{0, 1}, {1, 1}, {2, 1}, {3, 1}};

TEST(EvictionSwitchTest, JoinKeepsRunsAndRenumbersFetches) {
    const Trace a = TraceOfRuns(Concat(kPrefix, {{5, 2}, {6, 3}}));
    const Trace b = TraceOfRuns(Concat(kPrefix, {{7, 1}, {5, 4}}));
    const auto joined = JoinPhases({{&a, 4, 6, "A"}, {&b, 4, 6, "B"}});

    const auto& runs = joined.trace.runs;
    ASSERT_EQ(runs.size(), 4u);
    const uint32_t pages[] = {5, 6, 7, 5};
    const uint64_t firsts[] = {0, 2, 5, 6};
    const uint32_t lengths[] = {2, 3, 1, 4};
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(runs[i].page, pages[i]);
        EXPECT_EQ(runs[i].first_fetch, firsts[i]);
        EXPECT_EQ(runs[i].length, lengths[i]);
    }
    EXPECT_EQ(joined.trace.end_fetch, 10u);
    EXPECT_EQ(joined.trace.page_limit, 8u);
    EXPECT_EQ(joined.phase_start, (std::vector<size_t>{0, 2}));
    EXPECT_EQ(joined.names, (std::vector<std::string>{"A", "B"}));
    EXPECT_EQ(joined.PhaseOf(1), 0u);
    EXPECT_EQ(joined.PhaseOf(2), 1u);
    EXPECT_EQ(joined.PhaseEnd(0), 2u);
    EXPECT_EQ(joined.PhaseEnd(1), 4u);
}

TEST(EvictionSwitchTest, JoinMergesASharedBoundaryPage) {
    const Trace a = TraceOfRuns(Concat(kPrefix, {{6, 3}}));
    const Trace b = TraceOfRuns(Concat(kPrefix, {{6, 2}, {8, 1}}));
    const auto joined = JoinPhases({{&a, 4, 5, "A"}, {&b, 4, 6, "B"}});

    ASSERT_EQ(joined.trace.runs.size(), 2u);
    EXPECT_EQ(joined.trace.runs[0].page, 6u);
    EXPECT_EQ(joined.trace.runs[0].length, 5u);
    EXPECT_EQ(joined.trace.runs[1].page, 8u);
    EXPECT_EQ(joined.trace.runs[1].first_fetch, 5u);
    EXPECT_EQ(joined.phase_start, (std::vector<size_t>{0, 1}));
    EXPECT_EQ(joined.PhaseOf(0), 0u);
    EXPECT_EQ(joined.PhaseOf(1), 1u);
}

TEST(EvictionSwitchTest, JoinRefusesADifferentPrefix) {
    const Trace a = TraceOfRuns(Concat(kPrefix, {{5, 2}}));
    const Trace other_page = TraceOfRuns(Concat({{0, 1}, {1, 1}, {9, 1}, {3, 1}}, {{5, 2}}));
    const Trace other_length = TraceOfRuns(Concat({{0, 1}, {1, 2}, {2, 1}, {3, 1}}, {{5, 2}}));

    for (const Trace* bad : {&other_page, &other_length}) {
        try {
            (void)JoinPhases({{&a, 4, 5, "A"}, {bad, 4, 5, "B"}});
            FAIL() << "expected a throw";
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find("B"), std::string::npos);
        }
    }
}

TEST(EvictionSwitchTest, JoinRefusesAMergedRunTooLong) {
    const Trace a = TraceOfRuns({{6, UINT32_MAX}});
    const Trace b = TraceOfRuns({{6, 1}});
    EXPECT_THROW((void)JoinPhases({{&a, 0, 1, "A"}, {&b, 0, 1, "B"}}), std::runtime_error);
}

TEST(EvictionSwitchTest, JoinRefusesEmptyInput) {
    const Trace a = TraceOfRuns(Concat(kPrefix, {{5, 2}}));
    EXPECT_THROW((void)JoinPhases({}), std::invalid_argument);
    EXPECT_THROW((void)JoinPhases({{&a, 4, 4, "A"}}), std::invalid_argument);
}

// Removes a temp directory when the test ends, pass or fail.
struct TempDir {
    std::filesystem::path path;
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

TEST(EvictionSwitchTest, LoadRecordedAndBodyOfUseTheItems) {
    const TempDir temp{TempPath("rec")};
    const std::filesystem::path& dir = temp.path;
    std::filesystem::create_directories(dir / "t1");
    {
        std::ofstream trace(dir / "t1.csv");
        trace << "timestamp,page_id\n0,10\n1,11\n2,12\n3,13\n4,14\n5,15\n";
        std::ofstream items(dir / "t1" / "stream.csv");
        items << "pass,warmup,seq,kind,query,trace_start,trace_end\n"
              << "0,0,0,tpch,Q1,3,5\n"
              << "0,0,1,tpch,Q6,5,6\n";
    }

    const LoadedTrace loaded = LoadRecorded(dir.string(), "t1");
    EXPECT_EQ(loaded.name, "t1");
    EXPECT_EQ(loaded.trace.end_fetch, 6u);
    ASSERT_EQ(loaded.items.size(), 2u);
    const Piece body = BodyOf(loaded);
    EXPECT_EQ(body.trace, &loaded.trace);
    EXPECT_EQ(body.body_begin, 3u);
    EXPECT_EQ(body.body_end, 6u);
    EXPECT_EQ(body.name, "t1");
}

TEST(EvictionSwitchTest, CandidatesAreInTieOrder) {
    EXPECT_EQ(SwitchCandidates(),
              (std::vector<std::string>{"lru", "clock", "lru2_visit", "arc_visit", "lirs_visit"}));
}

RunList RandomRuns(std::mt19937& rng, size_t count, uint32_t pages, uint32_t max_length) {
    RunList runs;
    while (runs.size() < count) {
        const uint32_t page = rng() % pages;
        if (!runs.empty() && runs.back().first == page) continue;
        runs.emplace_back(page, 1 + rng() % max_length);
    }
    return runs;
}

SimContext ContextOf(const Trace& trace) {
    SimContext ctx;
    ctx.trace = &trace;
    return ctx;
}

std::set<uint32_t> SetOf(const std::vector<uint32_t>& pages) { return {pages.begin(), pages.end()}; }

TEST(EvictionSwitchTest, ResidentPagesAreExactlyThePagesThatHit) {
    std::mt19937 rng(11);
    const Trace trace = TraceOfRuns(RandomRuns(rng, 600, 30, 3));
    const SimContext ctx = ContextOf(trace);
    for (const std::string name : {"lru", "clock", "lru2", "lru2_visit", "lru2_full", "arc", "arc_visit", "arc_full",
                                   "lirs", "lirs_visit", "lirs_full"}) {
        for (const size_t r : {50u, 200u, 599u}) {
            auto policy = MakePolicy(name, 8, ctx);
            (void)ReplayRange(*policy, trace, 0, r);
            const auto resident = policy->ResidentPages();
            const auto set = SetOf(resident);
            EXPECT_EQ(set.size(), resident.size()) << name << " has duplicates";
            EXPECT_LE(resident.size(), 8u) << name;
            const evsim::Run& last = trace.runs[r - 1];
            for (uint32_t p = 0; p < trace.page_limit; ++p) {
                auto probe = MakePolicy(name, 8, ctx);
                (void)ReplayRange(*probe, trace, 0, r);
                const bool hit = !probe->Access(r, evsim::Run{p, last.first_fetch + last.length, 1});
                EXPECT_EQ(hit, set.count(p) == 1) << name << " r=" << r << " page " << p;
            }
        }
    }
}

TEST(EvictionSwitchTest, ResidentPagesThrowsForOtherPolicies) {
    std::mt19937 rng(3);
    const Trace trace = TraceOfRuns(RandomRuns(rng, 100, 20, 2));
    const auto next = NextUse(trace);
    SimContext ctx = ContextOf(trace);
    ctx.next_use = &next;
    auto policy = MakePolicy("belady", 8, ctx);
    EXPECT_THROW((void)policy->ResidentPages(), std::logic_error);
}

TEST(EvictionSwitchTest, HandOverToLruChangesNothing) {
    std::mt19937 rng(5);
    const Trace trace = TraceOfRuns(RandomRuns(rng, 600, 30, 3));
    const SimContext ctx = ContextOf(trace);
    const size_t n = trace.runs.size();

    auto straight = MakePolicy("lru", 8, ctx);
    const uint64_t expected = ReplayRange(*straight, trace, 0, n);

    std::unique_ptr<Policy> policy = MakePolicy("lru", 8, ctx);
    std::vector<uint32_t> last_run(trace.page_limit, kNever);
    uint64_t misses = 0;
    size_t from = 0;
    for (const size_t stop : {size_t{100}, size_t{250}, size_t{400}, n}) {
        misses += ReplayRange(*policy, trace, from, stop);
        NoteRuns(last_run, trace, from, stop);
        from = stop;
        if (stop < n) policy = HandOver("lru", 8, ctx, policy->ResidentPages(), last_run);
    }
    EXPECT_EQ(misses, expected);
}

TEST(EvictionSwitchTest, HandOverGivesPagesOldestFirst) {
    std::mt19937 rng(9);
    RunList runs = RandomRuns(rng, 300, 30, 3);
    for (uint32_t p = 30; p < 38; ++p) runs.emplace_back(p, 1);
    const Trace trace = TraceOfRuns(runs);
    const SimContext ctx = ContextOf(trace);

    auto arc = MakePolicy("arc_visit", 8, ctx);
    (void)ReplayRange(*arc, trace, 0, 300);
    std::vector<uint32_t> last_run(trace.page_limit, kNever);
    NoteRuns(last_run, trace, 0, 300);

    auto lru = HandOver("lru", 8, ctx, arc->ResidentPages(), last_run);
    auto before = SetOf(lru->ResidentPages());
    ASSERT_EQ(before, SetOf(arc->ResidentPages()));

    std::set<uint32_t> old_left = before;
    for (size_t r = 300; r < 308; ++r) {
        const uint32_t expected = *std::min_element(old_left.begin(), old_left.end(), [&](uint32_t a, uint32_t b) {
            return last_run[a] < last_run[b];
        });
        ASSERT_TRUE(lru->Access(r, trace.runs[r]));
        const auto after = SetOf(lru->ResidentPages());
        std::set<uint32_t> gone;
        std::set_difference(before.begin(), before.end(), after.begin(), after.end(), std::inserter(gone, gone.end()));
        ASSERT_EQ(gone, std::set<uint32_t>{expected}) << "run " << r;
        old_left.erase(expected);
        before = after;
    }
}

PhasesTrace TwoPhases(uint32_t seed) {
    std::mt19937 rng(seed);
    PhasesTrace phases;
    phases.trace = TraceOfRuns(RandomRuns(rng, 600, 40, 3));
    phases.phase_start = {0, 300};
    phases.names = {"A", "B"};
    return phases;
}

TEST(EvictionSwitchTest, LiveReplayWithoutSwitchesIsTheFixedReplay) {
    const PhasesTrace phases = TwoPhases(21);
    const SimContext ctx = ContextOf(phases.trace);
    const auto& names = SwitchCandidates();
    for (size_t c = 0; c < names.size(); ++c) {
        const SwitchRun live = LiveReplay(phases, 8, ctx, names, Schedule{static_cast<int>(c), {}});
        const SwitchRun fixed = FixedReplay(phases, names[c], 8, ctx);
        EXPECT_EQ(live.misses, fixed.misses) << names[c];
        EXPECT_EQ(live.phase_misses, fixed.phase_misses) << names[c];
        EXPECT_EQ(live.switches, 0u);
    }
}

TEST(EvictionSwitchTest, FixedReplayFlagsTheMisses) {
    const PhasesTrace phases = TwoPhases(22);
    const SimContext ctx = ContextOf(phases.trace);
    std::vector<uint8_t> flags;
    const SwitchRun fixed = FixedReplay(phases, "lru", 8, ctx, &flags);
    ASSERT_EQ(flags.size(), phases.trace.runs.size());
    EXPECT_EQ(static_cast<uint64_t>(std::count(flags.begin(), flags.end(), 1)), fixed.misses);
}

TEST(EvictionSwitchTest, LiveReplayCountsPhasesAndOccupancy) {
    const PhasesTrace phases = TwoPhases(23);
    const SimContext ctx = ContextOf(phases.trace);
    const SwitchRun run = LiveReplay(phases, 8, ctx, SwitchCandidates(), Schedule{4, {{450, 3}}});
    EXPECT_EQ(run.occupancy[0][4], 300u);
    EXPECT_EQ(run.occupancy[1][4], 150u);
    EXPECT_EQ(run.occupancy[1][3], 150u);
    EXPECT_EQ(run.switches, 1u);
    EXPECT_EQ(run.phase_misses[0] + run.phase_misses[1], run.misses);
}

TEST(EvictionSwitchTest, LiveReplayRefusesABadSchedule) {
    const PhasesTrace phases = TwoPhases(24);
    const SimContext ctx = ContextOf(phases.trace);
    const auto& names = SwitchCandidates();
    const std::vector<Schedule> bad = {
        Schedule{0, {{300, 1}, {200, 2}}},  // out of order
        Schedule{0, {{300, 1}, {300, 2}}},  // same run twice
        Schedule{0, {{100, 0}}},            // already running
        Schedule{0, {{100, 1}, {200, 1}}},  // already running after a switch
        Schedule{0, {{600, 1}}},            // past the end
        Schedule{0, {{100, 5}}},            // no such candidate
        Schedule{7, {}},                    // no such start
    };
    for (const auto& schedule : bad) {
        EXPECT_THROW((void)LiveReplay(phases, 8, ctx, names, schedule), std::invalid_argument);
    }
}


// An evidence matrix with a decision every 100 runs.
Evidence EvidenceOf(const std::vector<std::vector<uint64_t>>& rows) {
    Evidence evidence;
    evidence.window = 40;
    evidence.step = 20;
    for (size_t d = 0; d < rows.size(); ++d) evidence.at.push_back(100 * (d + 1));
    evidence.misses = rows;
    return evidence;
}

std::vector<std::vector<uint64_t>> Repeat(std::vector<std::vector<uint64_t>> rows, size_t times) {
    std::vector<std::vector<uint64_t>> out;
    for (size_t i = 0; i < times; ++i) out.insert(out.end(), rows.begin(), rows.end());
    return out;
}

std::vector<std::vector<uint64_t>> Join(std::vector<std::vector<uint64_t>> a,
                                        const std::vector<std::vector<uint64_t>>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

using Switches = std::vector<std::pair<size_t, int>>;

TEST(EvictionSwitchTest, EvidenceEqualsDirectReplay) {
    std::mt19937 rng(5);
    const Trace trace = TraceOfRuns(RandomRuns(rng, 300, 30, 3));
    const SimContext ctx = ContextOf(trace);
    const auto& names = SwitchCandidates();
    const auto evidence = BuildEvidence(trace, 8, 40, ctx, names);

    EXPECT_EQ(evidence.window, 40u);
    EXPECT_EQ(evidence.step, 20u);
    ASSERT_GT(evidence.at.size(), 5u);
    EXPECT_EQ(evidence.at.front(), 40u);
    EXPECT_EQ(evidence.at[1], 60u);
    EXPECT_LT(evidence.at.back(), trace.runs.size());
    ASSERT_EQ(evidence.misses.size(), evidence.at.size());
    for (size_t d = 0; d < evidence.at.size(); ++d) {
        ASSERT_EQ(evidence.misses[d].size(), names.size());
        for (size_t c = 0; c < names.size(); ++c) {
            auto policy = MakePolicy(names[c], 8, ctx);
            uint64_t misses = 0;
            for (size_t r = evidence.at[d] - 40; r < evidence.at[d]; ++r) {
                const bool missed = policy->Access(r, trace.runs[r]);
                if (r >= evidence.at[d] - 30 && missed) ++misses;
            }
            EXPECT_EQ(evidence.misses[d][c], misses) << "d=" << d << " c=" << c;
        }
    }
}

TEST(EvictionSwitchTest, BuildEvidenceRefusesBadWindows) {
    std::mt19937 rng(6);
    const Trace trace = TraceOfRuns(RandomRuns(rng, 100, 20, 2));
    const SimContext ctx = ContextOf(trace);
    const auto& names = SwitchCandidates();
    EXPECT_THROW((void)BuildEvidence(trace, 8, 0, ctx, names), std::invalid_argument);
    EXPECT_THROW((void)BuildEvidence(trace, 8, 42, ctx, names), std::invalid_argument);
    EXPECT_THROW((void)BuildEvidence(trace, 8, 100, ctx, names), std::invalid_argument);
    EXPECT_THROW((void)BuildEvidence(trace, 8, 104, ctx, names), std::invalid_argument);
    try {
        (void)BuildEvidence(trace, 8, 100, ctx, names);
        FAIL() << "expected a throw";
    } catch (const std::invalid_argument& e) {
        EXPECT_STREQ(e.what(), "window 100 leaves no decision");
    }
}

TEST(EvictionSwitchTest, TwoWindowNeedsTheSameLeaderTwice) {
    const auto twice = TwoWindowRule(EvidenceOf(Repeat({{100, 90}}, 2)), 0, 0.05);
    EXPECT_EQ(twice.start, 0);
    EXPECT_EQ(twice.switches, (Switches{{200, 1}}));

    const auto differ = TwoWindowRule(EvidenceOf({{100, 90, 100}, {100, 100, 90}}), 0, 0.05);
    EXPECT_TRUE(differ.switches.empty());
}

TEST(EvictionSwitchTest, TwoWindowThresholdIsInclusive) {
    EXPECT_TRUE(TwoWindowRule(EvidenceOf(Repeat({{100, 96}}, 3)), 0, 0.05).switches.empty());
    EXPECT_EQ(TwoWindowRule(EvidenceOf(Repeat({{100, 95}}, 2)), 0, 0.05).switches, (Switches{{200, 1}}));
}

TEST(EvictionSwitchTest, TwoWindowSwitchesBackAfterTheGap) {
    const auto rows = Join(Repeat({{100, 90}}, 2), Repeat({{80, 100}}, 2));
    EXPECT_EQ(TwoWindowRule(EvidenceOf(rows), 0, 0.05).switches, (Switches{{200, 1}, {400, 0}}));
}

TEST(EvictionSwitchTest, HedgeWeightsMatchHandValues) {
    const auto evidence = EvidenceOf(Repeat({{100, 110}}, 2));
    const auto plain = HedgeWeights(evidence, 10, 0);
    ASSERT_EQ(plain.size(), 2u);
    EXPECT_NEAR(plain[0][0], 0.7311, 1e-4);
    EXPECT_NEAR(plain[0][1], 0.2689, 1e-4);
    EXPECT_NEAR(plain[1][0], 0.8808, 1e-4);
    EXPECT_NEAR(plain[1][1], 0.1192, 1e-4);

    const auto shared = HedgeWeights(evidence, 10, 0.1);
    EXPECT_NEAR(shared[0][0], 0.7080, 1e-4);
    EXPECT_NEAR(shared[0][1], 0.2920, 1e-4);
}

TEST(EvictionSwitchTest, HedgeSwitchesWhenLeaderDoublesCurrent) {
    const auto evidence = EvidenceOf(Repeat({{100, 110}}, 2));
    EXPECT_EQ(HedgeRule(evidence, 1, HedgeSettings{10, 0, 2}).switches, (Switches{{100, 0}}));
    EXPECT_EQ(HedgeRule(evidence, 1, HedgeSettings{5, 0, 2}).switches, (Switches{{200, 0}}));
}

TEST(EvictionSwitchTest, FixedShareRecoversFasterThanPlainHedge) {
    const auto evidence = EvidenceOf(Join(Repeat({{100, 110}}, 10), Repeat({{110, 100}}, 15)));
    const auto plain = HedgeRule(evidence, 0, HedgeSettings{10, 0, 1});
    ASSERT_EQ(plain.switches.size(), 1u);
    EXPECT_EQ(plain.switches[0].first, evidence.at[20]);
    EXPECT_EQ(plain.switches[0].second, 1);

    const auto shared = HedgeRule(evidence, 0, HedgeSettings{10, 0.1, 1});
    ASSERT_FALSE(shared.switches.empty());
    EXPECT_EQ(shared.switches[0].first, evidence.at[12]);  // the 13th decision
    EXPECT_EQ(shared.switches[0].second, 1);
}

TEST(EvictionSwitchTest, RulesRefuseAStartThatIsNotACandidate) {
    const auto evidence = EvidenceOf(Repeat({{100, 90}}, 2));
    for (const int start : {-1, 2}) {
        EXPECT_THROW((void)TwoWindowRule(evidence, start, 0.05), std::invalid_argument);
        EXPECT_THROW((void)HedgeRule(evidence, start, HedgeSettings{10, 0, 2}), std::invalid_argument);
    }
}

TEST(EvictionSwitchTest, RulesStayPutOnFlatOrZeroEvidence) {
    for (const uint64_t level : {uint64_t{0}, uint64_t{50}}) {
        const auto evidence = EvidenceOf(Repeat({{level, level, level}}, 6));
        EXPECT_TRUE(TwoWindowRule(evidence, 2, 0.05).switches.empty());
        for (const double alpha : {0.0, 0.1}) {
            EXPECT_TRUE(HedgeRule(evidence, 2, HedgeSettings{20, alpha, 2}).switches.empty());
            for (const auto& row : HedgeWeights(evidence, 20, alpha)) {
                for (const double w : row) EXPECT_TRUE(std::isfinite(w));
            }
        }
    }
}

PhasesTrace SinglePhase(const RunList& runs) {
    PhasesTrace phases;
    phases.trace = TraceOfRuns(runs);
    phases.phase_start = {0};
    phases.names = {"A"};
    return phases;
}

TEST(EvictionSwitchTest, HindsightWithOneCandidateIsThatPolicy) {
    const PhasesTrace phases = TwoPhases(31);
    const SimContext ctx = ContextOf(phases.trace);
    const auto result = Hindsight(phases.trace, 8, 50, ctx, {"arc_visit"});
    EXPECT_EQ(result.misses, FixedReplay(phases, "arc_visit", 8, ctx).misses);
    EXPECT_EQ(result.schedule.start, 0);
    EXPECT_TRUE(result.schedule.switches.empty());
}

TEST(EvictionSwitchTest, HindsightPathReplaysToItsMisses) {
    const PhasesTrace phases = TwoPhases(32);
    const SimContext ctx = ContextOf(phases.trace);
    const auto& names = SwitchCandidates();
    for (const size_t step : {size_t{50}, size_t{77}}) {
        const auto result = Hindsight(phases.trace, 8, step, ctx, names);
        const SwitchRun live = LiveReplay(phases, 8, ctx, names, result.schedule);
        EXPECT_EQ(live.misses, result.misses) << "step " << step;
        EXPECT_EQ(live.switches, result.schedule.switches.size());
        EXPECT_GE(result.schedule.switches.size(), 1u) << "step " << step;
        uint64_t used = 0;
        for (const auto& row : live.occupancy) {
            for (const uint64_t n : row) used += n;
        }
        EXPECT_EQ(used, phases.trace.runs.size());
    }
}

TEST(EvictionSwitchTest, HindsightSwitchesWhenHalvesDiffer) {
    // First half: 8 new pages, round-robin 3 rounds, shifted 20 times. Second half: a 12-page loop.
    RunList first;
    uint32_t next_page = 0;
    for (int shift = 0; shift < 20; ++shift) {
        for (int round = 0; round < 3; ++round) {
            for (uint32_t i = 0; i < 8; ++i) first.emplace_back(next_page + i, 1);
        }
        next_page += 8;
    }
    RunList second;
    for (int round = 0; round < 30; ++round) {
        for (uint32_t i = 0; i < 12; ++i) second.emplace_back(next_page + i, 1);
    }
    const size_t pool = 10;
    const std::vector<std::string> names = {"lru", "lirs"};

    const auto fixed = [&](const RunList& runs, const std::string& name) {
        const PhasesTrace phases = SinglePhase(runs);
        const SimContext ctx = ContextOf(phases.trace);
        return FixedReplay(phases, name, pool, ctx).misses;
    };
    ASSERT_LT(fixed(first, "lru"), fixed(first, "lirs"));
    ASSERT_LT(fixed(second, "lirs"), fixed(second, "lru"));

    PhasesTrace both;
    both.trace = TraceOfRuns(Concat(first, second));
    both.phase_start = {0, first.size()};
    both.names = {"A", "B"};
    const SimContext ctx = ContextOf(both.trace);
    const size_t step = first.size() / 12;  // 480 / 12 = 40, so a segment ends where the halves meet
    ASSERT_EQ(first.size() % step, 0u);

    const auto result = Hindsight(both.trace, pool, step, ctx, names);
    EXPECT_LT(result.misses, FixedReplay(both, "lru", pool, ctx).misses);
    EXPECT_LT(result.misses, FixedReplay(both, "lirs", pool, ctx).misses);
    EXPECT_GE(result.schedule.switches.size(), 1u);
    EXPECT_EQ(LiveReplay(both, pool, ctx, names, result.schedule).misses, result.misses);
}

TEST(EvictionSwitchTest, FreeSwitchBoundHandExample) {
    const std::vector<uint8_t> a = {1, 0, 1, 0};
    const std::vector<uint8_t> b = {0, 1, 0, 1};
    EXPECT_EQ(FreeSwitchBound({&a, &b}, 1), 0u);
    EXPECT_EQ(FreeSwitchBound({&a, &b}, 2), 2u);
    EXPECT_EQ(FreeSwitchBound({&a, &b}, 4), 2u);
    EXPECT_EQ(FreeSwitchBound({&a}, 2), 2u);
    EXPECT_EQ(FreeSwitchBound({&a}, 3), 2u);
    EXPECT_THROW((void)FreeSwitchBound({&a, &b}, 0), std::invalid_argument);
    EXPECT_THROW((void)FreeSwitchBound({}, 1), std::invalid_argument);
    const std::vector<uint8_t> shorter = {1, 0};
    EXPECT_THROW((void)FreeSwitchBound({&a, &shorter}, 1), std::invalid_argument);
}

}  // namespace
}  // namespace shilmandb::evsim
