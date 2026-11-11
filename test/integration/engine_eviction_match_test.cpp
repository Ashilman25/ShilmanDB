#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <ostream>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "buffer/buffer_pool_manager.hpp"
#include "buffer/eviction_policy_factory.hpp"
#include "eviction/sim/eviction_sim.hpp"
#include "storage/disk_manager.hpp"

// Drives a real BufferPoolManager and the trace simulator with the same fetches, and checks
// that every fetch hits or misses in both.

namespace shilmandb {
namespace {

constexpr uint32_t kPages = 300;
// the buffer pool's visit window, which the simulator is given too
constexpr auto kWindow = static_cast<uint32_t>(BufferPoolManager::kVisitWindow);

// A fetch sequence, with ResetStats() calls between some fetches.
struct Sequence {
    std::vector<uint32_t> pages;     // one entry per fetch
    std::vector<bool> reset_before;  // call ResetStats() right before this fetch

    void Add(uint32_t page, bool reset = false) {
        pages.push_back(page);
        reset_before.push_back(reset);
    }
};

// Scans (some looped, so pages come back within and past the window), two scans
// interleaved page by page, hot pages and random pages. Every page is fetched 1-5 times in a row.
Sequence RandomSequence(uint64_t seed, size_t fetches) {
    std::mt19937_64 rng(seed);
    const auto below = [&](uint64_t n) { return static_cast<uint32_t>(rng() % n); };

    Sequence seq;
    const auto emit = [&](uint32_t page) {
        const uint32_t times = 1 + below(5);
        for (uint32_t k = 0; k < times; ++k) seq.Add(page, below(400) == 0);
    };

    std::vector<uint32_t> hot(12);
    for (auto& page : hot) page = below(kPages);

    while (seq.pages.size() < fetches) {
        switch (below(4)) {
            case 0: {  // a scan, looped up to 3 times
                const uint32_t start = below(kPages), length = 20 + below(110), loops = 1 + below(3);
                for (uint32_t l = 0; l < loops; ++l) {
                    for (uint32_t i = 0; i < length; ++i) emit((start + i) % kPages);
                }
                break;
            }
            case 1: {  // two scans, interleaved
                const uint32_t a = below(kPages), b = below(kPages), length = 10 + below(70);
                for (uint32_t i = 0; i < length; ++i) {
                    emit((a + i) % kPages);
                    emit((b + i) % kPages);
                }
                break;
            }
            case 2: {  // hot pages
                const uint32_t count = 20 + below(80);
                for (uint32_t i = 0; i < count; ++i) emit(hot[below(static_cast<uint32_t>(hot.size()))]);
                break;
            }
            default: {  // random pages
                const uint32_t count = 20 + below(80);
                for (uint32_t i = 0; i < count; ++i) emit(below(kPages));
                break;
            }
        }
    }
    seq.pages.resize(fetches);
    seq.reset_before.resize(fetches);
    return seq;
}

// Hot pages come back exactly 64 or 65 runs after their last run (the fillers between are
// cold pages), right after a ResetStats(), then repeat with another ResetStats() inside the repeats.
Sequence WindowEdgeSequence() {
    constexpr uint32_t kHot = 16, kCold = 60;
    std::mt19937_64 rng(1);
    Sequence seq;
    for (uint32_t block = 0; block < 300; ++block) {
        const auto page = static_cast<uint32_t>(rng() % kHot);
        const uint32_t fillers = rng() % 2 == 0 ? kWindow - 1 : kWindow;
        seq.Add(page);
        uint32_t previous = page;
        for (uint32_t i = 0; i < fillers; ++i) {
            uint32_t filler;
            do {
                filler = kHot + static_cast<uint32_t>(rng() % kCold);
            } while (filler == previous);
            seq.Add(filler);
            previous = filler;
        }
        seq.Add(page, true);  // fillers + 1 runs after its last run
        seq.Add(page);
        seq.Add(page, true);
    }
    return seq;
}

// Every fetch is one trace line, so the simulator merges repeats into runs itself.
evsim::Trace TraceOf(const Sequence& seq) {
    std::vector<std::pair<uint64_t, uint32_t>> lines;
    lines.reserve(seq.pages.size());
    for (size_t i = 0; i < seq.pages.size(); ++i) lines.emplace_back(i, seq.pages[i]);
    return evsim::MakeTrace(lines, seq.pages.size());
}

// The simulator's miss flag for every run.
std::vector<bool> SimMisses(const std::string& name, size_t pool, const evsim::Trace& trace, uint32_t window = kWindow) {
    evsim::SimContext ctx;
    ctx.trace = &trace;
    ctx.visit_window = window;
    auto policy = evsim::MakePolicy(name, pool, ctx);
    std::vector<bool> misses;
    misses.reserve(trace.runs.size());
    for (size_t r = 0; r < trace.runs.size(); ++r) misses.push_back(policy->Access(r, trace.runs[r]));
    return misses;
}

uint64_t Count(const std::vector<bool>& flags) {
    uint64_t n = 0;
    for (const bool f : flags) n += f ? 1 : 0;
    return n;
}

struct KindCase {
    EvictionPolicyKind kind;
    const char* sim_name;
};

// So test names and failures show the policy
void PrintTo(const KindCase& c, std::ostream* os) { *os << EvictionPolicyKindName(c.kind); }

class EngineEvictionMatchTest : public ::testing::TestWithParam<KindCase> {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() /
                      ("shilmandb_engine_match_" + std::to_string(::getpid()) + ".db")).string();
        std::filesystem::remove(test_file_);
    }

    void TearDown() override { std::filesystem::remove(test_file_); }

    // Runs the sequence through a real buffer pool and records whether each fetch missed.
    // Every fetch is unpinned right away, so nothing is pinned when the pool evicts.
    void RunEngine(size_t pool, const Sequence& seq, bool timing, std::vector<bool>* misses) {
        std::filesystem::remove(test_file_);
        DiskManager dm(test_file_);
        BufferPoolManager bpm(pool, &dm, MakeEvictionPolicy(GetParam().kind, pool));
        if (timing) bpm.SetPolicyTiming(true);

        for (uint32_t i = 0; i < kPages; ++i) {
            page_id_t pid;
            ASSERT_NE(bpm.NewPage(&pid), nullptr);
            ASSERT_EQ(pid, i);
            ASSERT_TRUE(bpm.UnpinPage(pid, true));
        }
        bpm.DropAllPages();

        misses->clear();
        misses->reserve(seq.pages.size());
        for (size_t i = 0; i < seq.pages.size(); ++i) {
            if (seq.reset_before[i]) bpm.ResetStats();
            const uint64_t before = bpm.GetMissCount();
            ASSERT_NE(bpm.FetchPage(seq.pages[i]), nullptr) << "fetch " << i;
            misses->push_back(bpm.GetMissCount() != before);
            ASSERT_TRUE(bpm.UnpinPage(seq.pages[i], false));
        }

        if (timing) {
            EXPECT_GT(bpm.GetPolicyStats().record_calls, 0u);
        }
    }

    // Checks the engine's flag on each run's first fetch against the simulator's, and that
    // the rest of every run hits. Returns the number of fetches compared.
    size_t ExpectSameMisses(const std::vector<bool>& engine, const evsim::Trace& trace, const std::vector<bool>& sim, const std::string& label) {
        EXPECT_EQ(engine.size(), trace.end_fetch) << label;
        EXPECT_EQ(sim.size(), trace.runs.size()) << label;
        if (engine.size() != trace.end_fetch || sim.size() != trace.runs.size()) return 0;

        size_t mismatches = 0;
        std::string first;
        for (size_t r = 0; r < trace.runs.size(); ++r) {
            const evsim::Run& run = trace.runs[r];
            if (engine[run.first_fetch] != sim[r]) {
                if (mismatches++ == 0) {
                    std::ostringstream out;
                    out << "run " << r << " (fetch " << run.first_fetch << ", page " << run.page << "): engine "
                        << (engine[run.first_fetch] ? "missed" : "hit") << ", simulator " << (sim[r] ? "missed" : "hit");
                    first = out.str();
                }
            }
            for (uint32_t k = 1; k < run.length; ++k) {
                if (engine[run.first_fetch + k] && mismatches++ == 0) {
                    first = "fetch " + std::to_string(run.first_fetch + k) + " repeats page " + std::to_string(run.page) + " but missed";
                }
            }
        }
        EXPECT_EQ(mismatches, 0u) << label << ": first difference at " << first;
        EXPECT_EQ(Count(engine), Count(sim)) << label;
        return engine.size();
    }

    std::string Label(size_t pool, const std::string& what) const {
        return std::string(EvictionPolicyKindName(GetParam().kind)) + " pool " + std::to_string(pool) + " " + what;
    }
};

TEST_P(EngineEvictionMatchTest, EngineMatchesSimOnRandomSequences) {
    constexpr size_t kFetches = 30000;
    size_t compared = 0;
    for (const uint64_t seed : {1u, 2u, 3u}) {
        const Sequence seq = RandomSequence(seed, kFetches);
        const evsim::Trace trace = TraceOf(seq);

        // The sequence has what the test needs: repeats, returns on both sides of the
        // window, and resets
        size_t repeats = 0, within = 0, past = 0;
        std::vector<int64_t> last_run(kPages, -1);
        for (size_t r = 0; r < trace.runs.size(); ++r) {
            const evsim::Run& run = trace.runs[r];
            repeats += run.length - 1;
            if (last_run[run.page] >= 0) {
                (r - static_cast<size_t>(last_run[run.page]) <= kWindow ? within : past) += 1;
            }
            last_run[run.page] = static_cast<int64_t>(r);
        }
        ASSERT_GT(repeats, 10000u) << "seed " << seed;
        ASSERT_GT(within, 1000u) << "seed " << seed;
        ASSERT_GT(past, 1000u) << "seed " << seed;
        ASSERT_GT(Count(seq.reset_before), 20u) << "seed " << seed;

        for (const size_t pool : {2u, 3u, 8u, 40u, 100u}) {
            std::vector<bool> engine;
            RunEngine(pool, seq, false, &engine);
            ASSERT_FALSE(HasFatalFailure());
            compared += ExpectSameMisses(engine, trace, SimMisses(GetParam().sim_name, pool, trace),
                                         Label(pool, "seed " + std::to_string(seed)));
        }
    }
    EXPECT_EQ(compared, 3 * 5 * kFetches);
}

TEST_P(EngineEvictionMatchTest, EngineMatchesSimAcrossResetAtWindowEdge) {
    const Sequence seq = WindowEdgeSequence();
    const evsim::Trace trace = TraceOf(seq);

    for (const size_t pool : {8u, 40u, 70u}) {
        std::vector<bool> engine;
        RunEngine(pool, seq, false, &engine);
        ASSERT_FALSE(HasFatalFailure());
        const std::vector<bool> sim = SimMisses(GetParam().sim_name, pool, trace);
        (void)ExpectSameMisses(engine, trace, sim, Label(pool, "window edge"));

        // For the policies that count visits, the edge matters: a window one run
        // shorter or longer gives different misses
        const std::string name = GetParam().sim_name;
        if (pool == 70 && name != "lru" && name != "clock") {
            EXPECT_NE(Count(SimMisses(name, pool, trace, kWindow - 1)), Count(sim)) << name;
            EXPECT_NE(Count(SimMisses(name, pool, trace, kWindow + 1)), Count(sim)) << name;
        }
    }
}

TEST_P(EngineEvictionMatchTest, EngineMatchesSimWithTimingOn) {
    constexpr size_t kPool = 40;
    const Sequence seq = RandomSequence(1, 30000);
    const evsim::Trace trace = TraceOf(seq);

    std::vector<bool> engine;
    RunEngine(kPool, seq, true, &engine);
    ASSERT_FALSE(HasFatalFailure());
    (void)ExpectSameMisses(engine, trace, SimMisses(GetParam().sim_name, kPool, trace), Label(kPool, "timed"));
}

INSTANTIATE_TEST_SUITE_P(Kinds, EngineEvictionMatchTest,
                         ::testing::Values(KindCase{EvictionPolicyKind::LRU, "lru"},
                                           KindCase{EvictionPolicyKind::CLOCK, "clock"},
                                           KindCase{EvictionPolicyKind::LRU2, "lru2_visit"},
                                           KindCase{EvictionPolicyKind::ARC, "arc_visit"},
                                           KindCase{EvictionPolicyKind::LIRS, "lirs_visit"}),
                         [](const ::testing::TestParamInfo<KindCase>& info) {
                             return std::string(EvictionPolicyKindName(info.param.kind));
                         });

}  // namespace
}  // namespace shilmandb
