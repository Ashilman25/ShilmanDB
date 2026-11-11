#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "eviction/sim/eviction_sim.hpp"
#include "storage/disk_manager.hpp"

namespace shilmandb::evsim {
namespace {

using Lines = std::vector<std::pair<uint64_t, uint32_t>>;

std::string TempPath(const std::string& name) {
    return (std::filesystem::temp_directory_path() /
            ("shilmandb_evsim_" + name + "_" + std::to_string(::getpid())))
        .string();
}

void WriteFile(const std::string& path, const std::string& text) {
    std::ofstream out(path);
    out << text;
}

// One fetch per page, timestamps 0..n-1.
Trace TraceOfPages(const std::vector<uint32_t>& pages) {
    Lines lines;
    for (size_t i = 0; i < pages.size(); ++i) lines.emplace_back(i, pages[i]);
    return MakeTrace(lines, pages.size());
}

using RunList = std::vector<std::pair<uint32_t, uint32_t>>;  // (page, fetches)

// Runs back to back from fetch 0.
Trace TraceOfRuns(const RunList& runs) {
    Lines lines;
    uint64_t ts = 0;
    for (const auto& [page, length] : runs) {
        lines.emplace_back(ts, page);
        ts += length;
    }
    return MakeTrace(lines, ts);
}

// The same runs with every fetch as its own run.
Trace RawTraceOfRuns(const RunList& runs) {
    Lines lines;
    uint64_t ts = 0;
    for (const auto& [page, length] : runs) {
        for (uint32_t k = 0; k < length; ++k) lines.emplace_back(ts++, page);
    }
    return MakeTrace(lines, ts, true);
}

// Random pages, each run 1..max_length fetches; a page never follows itself.
RunList RandomRuns(std::mt19937& rng, size_t count, uint32_t pages, uint32_t max_length) {
    RunList runs;
    while (runs.size() < count) {
        const uint32_t page = rng() % pages;
        if (!runs.empty() && runs.back().first == page) continue;
        runs.emplace_back(page, 1 + rng() % max_length);
    }
    return runs;
}

// Half the runs come from a pair of pages that changes every 20 runs, so pages come back
// after one or two other runs, the way a hash join reads two tables in turn.
RunList InterleavedRuns(std::mt19937& rng, size_t count, uint32_t pages) {
    RunList runs;
    uint32_t pair[2] = {0, 1};
    for (size_t r = 0; r < count; ++r) {
        if (r % 20 == 0) {
            pair[0] = rng() % pages;
            pair[1] = (pair[0] + 1 + rng() % (pages - 1)) % pages;
        }
        uint32_t page;
        do {
            page = rng() % 2 == 0 ? pair[rng() % 2] : rng() % pages;
        } while (!runs.empty() && runs.back().first == page);
        runs.emplace_back(page, 1 + rng() % 4);
    }
    return runs;
}

uint64_t RunPolicy(const std::string& name, size_t pool, const Trace& trace, uint32_t visit_window = 64) {
    const auto next_use = NextUse(trace);
    SimContext ctx;
    ctx.trace = &trace;
    ctx.next_use = &next_use;
    ctx.visit_window = visit_window;
    auto policy = MakePolicy(name, pool, ctx);
    return Simulate(*policy, trace, {}).misses;
}

// The miss flag of every run.
std::vector<bool> FastFlags(const std::string& name, size_t pool, const Trace& trace, uint32_t visit_window) {
    SimContext ctx;
    ctx.trace = &trace;
    ctx.visit_window = visit_window;
    auto policy = MakePolicy(name, pool, ctx);
    std::vector<bool> flags;
    for (size_t r = 0; r < trace.runs.size(); ++r) flags.push_back(policy->Access(r, trace.runs[r]));
    return flags;
}

// Slow, literal versions of the policies, to check the fast ones against. Their lists are
// plain vectors, front = oldest end, searched linearly.

using PageVec = std::vector<uint32_t>;

// Position of the page in the list, or the list's size if it is not there. A plain loop:
// in Debug it runs several times faster than std::find.
size_t Find(const PageVec& list, uint32_t page) {
    const uint32_t* data = list.data();
    const size_t n = list.size();
    size_t i = 0;
    while (i < n && data[i] != page) ++i;
    return i;
}

bool Has(const PageVec& list, uint32_t page) { return Find(list, page) < list.size(); }

// Removes the page if it is there.
void Erase(PageVec& list, uint32_t page) {
    const size_t i = Find(list, page);
    if (i < list.size()) list.erase(list.begin() + static_cast<std::ptrdiff_t>(i));
}

uint32_t PopFront(PageVec& list) {
    const uint32_t page = list.front();
    list.erase(list.begin());
    return page;
}

struct SlowClock {
    explicit SlowClock(size_t c) : slots(c, -1), referenced(c, false) {}

    bool Resident(uint32_t page) const {
        return std::find(slots.begin(), slots.end(), int64_t{page}) != slots.end();
    }
    void Touch(uint32_t, size_t) {}
    bool Use(uint32_t page, int64_t) {
        const auto it = std::find(slots.begin(), slots.end(), int64_t{page});
        if (it != slots.end()) {
            referenced[static_cast<size_t>(it - slots.begin())] = true;
            return false;
        }
        size_t slot;
        if (filled < slots.size()) {
            slot = filled++;
        } else {
            while (referenced[hand]) {
                referenced[hand] = false;
                hand = (hand + 1) % slots.size();
            }
            slot = hand;
            hand = (hand + 1) % slots.size();
        }
        slots[slot] = page;
        referenced[slot] = true;
        return true;
    }

    std::vector<int64_t> slots;  // page in each slot, -1 for empty
    std::vector<bool> referenced;
    size_t filled{0};
    size_t hand{0};
};

struct SlowLru2 {
    static constexpr int64_t kNever = -1;

    explicit SlowLru2(size_t c) : c(c) {}

    bool Resident(uint32_t page) const { return Has(resident, page); }
    void Touch(uint32_t page, size_t run) { last[page] = static_cast<int64_t>(run); }
    bool Use(uint32_t page, int64_t time) {
        if (page >= last.size()) {
            last.resize(page + 1, kNever);
            second.resize(page + 1, kNever);
        }
        const bool miss = !Resident(page);
        if (miss) {
            if (resident.size() == c) {
                auto victim = resident.begin();
                for (auto it = resident.begin(); it != resident.end(); ++it) {
                    if (EvictsBefore(*it, *victim)) victim = it;
                }
                resident.erase(victim);
            }
            resident.push_back(page);
        }
        if (last[page] != kNever) second[page] = last[page];
        last[page] = time;
        return miss;
    }
    // A page with no second-to-last use goes before one with it (oldest last use first);
    // between two that have one, the older second-to-last use goes first.
    bool EvictsBefore(uint32_t a, uint32_t b) const {
        const bool a_twice = second[a] != kNever, b_twice = second[b] != kNever;
        if (a_twice != b_twice) return !a_twice;
        if (!a_twice) return last[a] < last[b];
        return second[a] < second[b];
    }

    size_t c;
    PageVec resident;
    std::vector<int64_t> last, second;  // use times per page
};

// Figure 4 of Megiddo & Modha, case by case.
struct SlowArc {
    explicit SlowArc(size_t c) : c(c) {}

    bool Resident(uint32_t x) const { return Has(t1, x) || Has(t2, x); }
    void Touch(uint32_t x, size_t) {
        auto& list = Has(t1, x) ? t1 : t2;
        Erase(list, x);
        list.push_back(x);
    }
    bool Use(uint32_t x, int64_t) {
        if (Has(t1, x) || Has(t2, x)) {  // case I
            Erase(t1, x);
            Erase(t2, x);
            t2.push_back(x);
            return false;
        }
        const auto size_b1 = static_cast<double>(b1.size()), size_b2 = static_cast<double>(b2.size());
        if (Has(b1, x)) {  // case II
            p = std::min(static_cast<double>(c), p + (size_b1 >= size_b2 ? 1.0 : size_b2 / size_b1));
            Replace(x);
            Erase(b1, x);
            t2.push_back(x);
            return true;
        }
        if (Has(b2, x)) {  // case III
            p = std::max(0.0, p - (size_b2 >= size_b1 ? 1.0 : size_b1 / size_b2));
            Replace(x);
            Erase(b2, x);
            t2.push_back(x);
            return true;
        }
        const size_t l1 = t1.size() + b1.size();  // case IV
        const size_t total = l1 + t2.size() + b2.size();
        if (l1 == c) {
            if (t1.size() < c) {
                PopFront(b1);
                Replace(x);
            } else {
                PopFront(t1);
            }
        } else if (total >= c) {
            if (total == 2 * c) PopFront(b2);
            Replace(x);
        }
        t1.push_back(x);
        return true;
    }
    void Replace(uint32_t x) {
        const auto size_t1 = static_cast<double>(t1.size());
        if (!t1.empty() && ((Has(b2, x) && size_t1 == p) || size_t1 > p)) {
            b1.push_back(PopFront(t1));
        } else {
            if (t2.empty()) throw std::logic_error("slow arc: T2 empty");
            b2.push_back(PopFront(t2));
        }
    }

    size_t c;
    double p{0};
    PageVec t1, t2, b1, b2;
};

struct SlowLirs {
    explicit SlowLirs(size_t c)
        : c(c), lir_limit(c - static_cast<size_t>(std::max(1L, std::lround(0.01 * static_cast<double>(c))))) {}

    bool Resident(uint32_t x) const { return Has(resident, x); }
    void Touch(uint32_t x, size_t) {
        if (Has(lir, x)) {
            ToTop(x);
            Prune();
        } else {
            Erase(q, x);
            q.push_back(x);
        }
    }
    bool Use(uint32_t x, int64_t) {
        if (Has(lir, x)) {
            ToTop(x);
            Prune();
            return false;
        }
        if (Has(resident, x)) {
            if (Has(s, x)) {
                ToTop(x);
                Erase(q, x);
                lir.push_back(x);
                DemoteBottom();
            } else {
                s.push_back(x);
                Erase(q, x);
                q.push_back(x);
            }
            return false;
        }
        if (resident.size() == c) Erase(resident, PopFront(q));
        resident.push_back(x);
        if (lir.size() < lir_limit) {
            ToTop(x);
            lir.push_back(x);
        } else if (Has(s, x)) {
            ToTop(x);
            lir.push_back(x);
            DemoteBottom();
        } else {
            s.push_back(x);
            q.push_back(x);
        }
        return true;
    }
    void ToTop(uint32_t x) {
        Erase(s, x);
        s.push_back(x);
    }
    void Prune() {
        while (!s.empty() && !Has(lir, s.front())) PopFront(s);
    }
    void DemoteBottom() {
        const uint32_t bottom = PopFront(s);
        Erase(lir, bottom);
        q.push_back(bottom);
        Prune();
    }

    size_t c;
    size_t lir_limit;
    PageVec s;  // front is the bottom of the stack
    PageVec q;  // front is the next victim
    PageVec lir, resident;
};

// Feeds runs to a slow policy in the given mode and returns each run's miss flag. Visit:
// a run of a resident page within `window` runs of that page's previous run is a touch.
// Every fetch: a run of L fetches is L uses at fetch numbers first_fetch.. (only the
// first can miss).
template <typename Slow>
std::vector<bool> SlowFlags(Slow slow, const std::vector<Run>& runs, Mode mode, uint32_t window) {
    std::vector<int64_t> last_run;  // per page, -1 before its first run
    std::vector<bool> flags;
    for (size_t i = 0; i < runs.size(); ++i) {
        const Run& run = runs[i];
        if (run.page >= last_run.size()) last_run.resize(run.page + 1, -1);
        const int64_t prev = last_run[run.page];
        const bool touch = mode == Mode::VISIT && slow.Resident(run.page) && prev >= 0 &&
                           static_cast<int64_t>(i) - prev <= static_cast<int64_t>(window);
        last_run[run.page] = static_cast<int64_t>(i);
        if (touch) {
            slow.Touch(run.page, i);
            flags.push_back(false);
            continue;
        }
        const uint32_t uses = mode == Mode::EVERY_FETCH ? run.length : 1;
        bool miss = false;
        for (uint32_t k = 0; k < uses; ++k) {
            const auto time = mode == Mode::EVERY_FETCH ? static_cast<int64_t>(run.first_fetch + k)
                                                        : static_cast<int64_t>(i);
            const bool m = slow.Use(run.page, time);
            if (k == 0) miss = m;
            else if (m) throw std::logic_error("a repeat fetch missed");
        }
        flags.push_back(miss);
    }
    return flags;
}

bool SameRuns(const Trace& a, const Trace& b) {
    if (a.runs.size() != b.runs.size()) return false;
    for (size_t i = 0; i < a.runs.size(); ++i) {
        if (a.runs[i].page != b.runs[i].page || a.runs[i].first_fetch != b.runs[i].first_fetch ||
            a.runs[i].length != b.runs[i].length) {
            return false;
        }
    }
    return true;
}

TEST(EvictionSimTest, LoadMergesRepeatsIntoRuns) {
    const Lines lines = {{0, 5}, {1, 5}, {2, 7}, {5, 5}, {6, 9}};
    const Trace trace = MakeTrace(lines, 9);
    ASSERT_EQ(trace.runs.size(), 4u);
    const evsim::Run expected[] = {{5, 0, 2}, {7, 2, 3}, {5, 5, 1}, {9, 6, 3}};
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(trace.runs[i].page, expected[i].page);
        EXPECT_EQ(trace.runs[i].first_fetch, expected[i].first_fetch);
        EXPECT_EQ(trace.runs[i].length, expected[i].length);
    }
    EXPECT_EQ(trace.page_limit, 10u);

    const std::string path = TempPath("load.csv");
    WriteFile(path, "timestamp,page_id\n0,5\n1,5\n2,7\n5,5\n6,9\n");
    const Trace loaded = LoadTrace(path, 9);
    std::filesystem::remove(path);
    EXPECT_TRUE(SameRuns(trace, loaded));
    EXPECT_EQ(loaded.page_limit, 10u);
}

TEST(EvictionSimTest, CollapsedAndFullTracesGiveSameRuns) {
    std::mt19937 rng(7);
    Lines full, collapsed;
    uint64_t ts = 0;
    uint32_t last = UINT32_MAX;
    while (ts < 3000) {
        uint32_t page;
        do { page = rng() % 40; } while (page == last);
        const int len = 1 + static_cast<int>(rng() % 5);
        for (int i = 0; i < len; ++i) {
            full.emplace_back(ts + i, page);
        }
        collapsed.emplace_back(ts, page);
        ts += len;
        last = page;
    }
    const Trace a = MakeTrace(full, ts);
    const Trace b = MakeTrace(collapsed, ts);
    EXPECT_TRUE(SameRuns(a, b));

    const Trace raw = MakeTrace(full, ts, true);
    EXPECT_EQ(raw.runs.size(), ts);
    for (const auto& run : raw.runs) EXPECT_EQ(run.length, 1u);
}

TEST(EvictionSimTest, BadTraceThrows) {
    EXPECT_THROW((void)MakeTrace({{3, 1}, {2, 1}}, 10), std::runtime_error);
    EXPECT_THROW((void)MakeTrace({{0, 1}, {4, 2}}, 4), std::runtime_error);
    EXPECT_THROW((void)MakeTrace({}, 4), std::runtime_error);

    const std::string path = TempPath("bad.csv");
    WriteFile(path, "ts,page\n0,1\n");
    EXPECT_THROW((void)LoadTrace(path, 5), std::runtime_error);
    WriteFile(path, "timestamp,page_id\n0,3000000000\n");
    EXPECT_THROW((void)LoadTrace(path, 5), std::runtime_error);
    WriteFile(path, "pass,warmup,seq,kind,query\n0,0,0,tpch,Q1\n");
    EXPECT_THROW((void)LoadItems(path), std::runtime_error);
    WriteFile(path, "pass,warmup,seq,kind,query,trace_start,trace_end\n0,0,0,tpch,Q1,,\n");
    EXPECT_THROW((void)LoadItems(path), std::runtime_error);
    std::filesystem::remove(path);

    const Trace trace = TraceOfPages({0, 1, 2});
    SimContext ctx;
    ctx.trace = &trace;
    EXPECT_THROW((void)MakePolicy("nope", 2, ctx), std::invalid_argument);
    EXPECT_THROW((void)MakePolicy("lru", 0, ctx), std::invalid_argument);
    EXPECT_THROW((void)MakePolicy("belady", 2, ctx), std::invalid_argument);
    EXPECT_THROW((void)MakePolicy("lru", 2, SimContext{}), std::invalid_argument);
}

TEST(EvictionSimTest, CutShortTraceAndMisorderedItemsThrow) {
    const Trace trace = MakeTrace({{0, 1}, {3, 2}, {6, 3}}, 10);
    std::vector<Item> items(2);
    items[0].trace_start = 0;
    items[0].trace_end = 5;
    items[1].trace_start = 5;
    items[1].trace_end = 10;
    EXPECT_NO_THROW(MapItems(trace, items));

    // The last item starts after the trace's last line: the file was cut short.
    items[1].trace_start = 8;
    EXPECT_THROW(MapItems(trace, items), std::runtime_error);

    // An item that starts before the one before it ended.
    items[1].trace_start = 4;
    EXPECT_THROW(MapItems(trace, items), std::runtime_error);

    // An item that ends before it starts.
    items[1].trace_start = 7;
    items[1].trace_end = 6;
    EXPECT_THROW(MapItems(trace, items), std::runtime_error);
}

TEST(EvictionSimTest, ItemsCountMissesPerItem) {
    // Runs: P1 @0 (3), P2 @3 (3), P3 @6 (6, crosses into item 1), P1 @12 (3), P2 @15 (5)
    const Trace trace = MakeTrace({{0, 1}, {3, 2}, {6, 3}, {12, 1}, {15, 2}}, 20);
    std::vector<Item> items(2);
    items[0].trace_start = 3;
    items[0].trace_end = 10;
    items[1].trace_start = 10;
    items[1].trace_end = 20;
    MapItems(trace, items);
    EXPECT_EQ(items[0].first_run, 1u);
    EXPECT_EQ(items[0].end_run, 3u);
    EXPECT_EQ(items[1].first_run, 3u);
    EXPECT_EQ(items[1].end_run, 5u);
    EXPECT_EQ(EndFetch(items), 20u);

    SimContext ctx;
    ctx.trace = &trace;
    auto lru = MakePolicy("lru", 2, ctx);
    const SimResult result = Simulate(*lru, trace, items);
    EXPECT_EQ(result.item_misses[0], 2u);  // P2 and P3 (P3 counts here though it runs on)
    EXPECT_EQ(result.item_misses[1], 2u);
    EXPECT_EQ(result.misses, 5u);          // plus the miss on P1 before item 0
    EXPECT_EQ(result.item_misses[0] + result.item_misses[1] + 1, result.misses);
}

TEST(EvictionSimTest, LruMatchesBufferPool) {
    const std::string path = TempPath("bpm.db");
    std::filesystem::remove(path);
    constexpr int kPages = 300;
    constexpr size_t kPool = 40;
    constexpr int kFetches = 20000;

    std::vector<uint32_t> sequence;
    uint64_t bpm_misses = 0;
    {
        DiskManager dm(path);
        const std::vector<char> blank(PAGE_SIZE, 0);
        for (int i = 0; i < kPages; ++i) dm.WritePage(dm.AllocatePage(), blank.data());
        BufferPoolManager bpm(kPool, &dm, std::make_unique<LRUEvictionPolicy>(kPool));

        std::mt19937 rng(11);
        for (int i = 0; i < kFetches; ++i) {
            const uint32_t pid = (rng() % 100 < 70) ? rng() % 30 : rng() % kPages;
            sequence.push_back(pid);
            ASSERT_NE(bpm.FetchPage(pid), nullptr);
            (void)bpm.UnpinPage(pid, false);
        }
        bpm_misses = bpm.GetMissCount();
    }
    std::filesystem::remove(path);

    EXPECT_EQ(RunPolicy("lru", kPool, TraceOfPages(sequence)), bpm_misses);
}

TEST(EvictionSimTest, TeachingsExampleLruAndBelady) {
    // A B C D A B E A B C D E
    const Trace trace = TraceOfPages({0, 1, 2, 3, 0, 1, 4, 0, 1, 2, 3, 4});
    EXPECT_EQ(RunPolicy("lru", 3, trace), 10u);
    EXPECT_EQ(RunPolicy("belady", 3, trace), 7u);
}

TEST(EvictionSimTest, BeladyMatchesBruteForce) {
    std::mt19937 rng(3);
    for (int round = 0; round < 200; ++round) {
        const size_t len = 10 + rng() % 3;
        std::vector<uint32_t> pages;
        for (size_t i = 0; i < len; ++i) {
            const uint32_t page = rng() % 5;
            if (pages.empty() || pages.back() != page) pages.push_back(page);  // repeats would merge into one run
        }
        const Trace trace = TraceOfPages(pages);

        for (const size_t pool : {2u, 3u}) {
            // Fewest misses over every victim choice, memoized on (position, resident set).
            std::map<std::pair<size_t, unsigned>, uint64_t> memo;
            std::function<uint64_t(size_t, unsigned)> best = [&](size_t pos, unsigned set) -> uint64_t {
                if (pos == pages.size()) return 0;
                const auto key = std::make_pair(pos, set);
                if (const auto it = memo.find(key); it != memo.end()) return it->second;
                const unsigned bit = 1u << pages[pos];
                uint64_t result;
                if (set & bit) {
                    result = best(pos + 1, set);
                } else if (static_cast<size_t>(__builtin_popcount(set)) < pool) {
                    result = 1 + best(pos + 1, set | bit);
                } else {
                    result = UINT64_MAX;
                    for (unsigned v = 0; v < 5; ++v) {
                        if (set & (1u << v)) result = std::min(result, 1 + best(pos + 1, (set & ~(1u << v)) | bit));
                    }
                }
                return memo[key] = result;
            };
            const uint64_t expected = best(0, 0);
            EXPECT_EQ(RunPolicy("belady", pool, trace), expected) << "round " << round << " pool " << pool;
        }
    }
}

TEST(EvictionSimTest, BeladyNeverMissesMoreThanLru) {
    std::mt19937 rng(5);
    for (int round = 0; round < 50; ++round) {
        std::vector<uint32_t> pages;
        for (int i = 0; i < 2000; ++i) pages.push_back(rng() % 200);
        const Trace trace = TraceOfPages(pages);
        for (const size_t pool : {10u, 50u, 150u}) {
            EXPECT_LE(RunPolicy("belady", pool, trace), RunPolicy("lru", pool, trace));
        }
    }
}

TEST(EvictionSimTest, TeachingsExampleOtherPolicies) {
    // A B C D A B E A B C D E
    const Trace trace = TraceOfPages({0, 1, 2, 3, 0, 1, 4, 0, 1, 2, 3, 4});
    EXPECT_EQ(RunPolicy("clock", 3, trace), 9u);
    EXPECT_EQ(RunPolicy("lru2", 3, trace), 10u);
    EXPECT_EQ(RunPolicy("arc", 3, trace), 10u);
    EXPECT_EQ(RunPolicy("lirs", 3, trace), 8u);
}

TEST(EvictionSimTest, ScanAfterHotPair) {
    // The pair 1 2 twice, a scan of pages 10..19, then the pair again.
    std::vector<uint32_t> pages = {1, 2, 1, 2};
    for (uint32_t p = 10; p < 20; ++p) pages.push_back(p);
    pages.push_back(1);
    pages.push_back(2);
    const Trace trace = TraceOfPages(pages);
    EXPECT_EQ(RunPolicy("lru", 4, trace), 14u);
    EXPECT_EQ(RunPolicy("clock", 4, trace), 14u);
    EXPECT_EQ(RunPolicy("lru2", 4, trace), 12u);
    EXPECT_EQ(RunPolicy("arc", 4, trace), 12u);
    EXPECT_EQ(RunPolicy("lirs", 4, trace), 12u);
    EXPECT_EQ(RunPolicy("belady", 4, trace), 12u);
}

TEST(EvictionSimTest, EveryFetchLosesTheHotPair) {
    // As above, but each scanned page is fetched twice in a row: counted fetch by fetch,
    // the scan looks used twice and pushes the pair out.
    RunList runs = {{1, 1}, {2, 1}, {1, 1}, {2, 1}};
    for (uint32_t p = 10; p < 20; ++p) runs.emplace_back(p, 2);
    runs.emplace_back(1, 1);
    runs.emplace_back(2, 1);
    const Trace trace = TraceOfRuns(runs);
    for (const std::string base : {"lru2", "arc", "lirs"}) {
        EXPECT_EQ(RunPolicy(base + "_full", 4, trace), 14u) << base;
        EXPECT_EQ(RunPolicy(base, 4, trace), 12u) << base;
    }
}

TEST(EvictionSimTest, LirsKeepsALoop) {
    // Pages 0..10 five times over: a loop one page bigger than the pool.
    std::vector<uint32_t> pages;
    for (int loop = 0; loop < 5; ++loop) {
        for (uint32_t p = 0; p <= 10; ++p) pages.push_back(p);
    }
    const Trace trace = TraceOfPages(pages);
    EXPECT_EQ(RunPolicy("lru", 10, trace), 55u);
    EXPECT_EQ(RunPolicy("clock", 10, trace), 55u);
    EXPECT_EQ(RunPolicy("lru2", 10, trace), 55u);
    EXPECT_EQ(RunPolicy("arc", 10, trace), 55u);
    EXPECT_EQ(RunPolicy("lirs", 10, trace), 19u);
    EXPECT_EQ(RunPolicy("belady", 10, trace), 15u);
}

TEST(EvictionSimTest, VisitModeSkipsInterleavedReturns) {
    // Pairs read in turn (10 20 10 20 ...) look used twice unless a quick return is a touch.
    std::vector<uint32_t> pages = {1, 2, 5, 6, 5, 6, 1, 2};
    for (uint32_t k = 0; k < 4; ++k) {
        for (const uint32_t p : {10 + k, 20 + k, 10 + k, 20 + k}) pages.push_back(p);
    }
    pages.push_back(1);
    pages.push_back(2);
    const Trace trace = TraceOfPages(pages);
    EXPECT_EQ(RunPolicy("lru2", 4, trace, 3), 22u);
    EXPECT_EQ(RunPolicy("lru2_visit", 4, trace, 3), 12u);
    EXPECT_EQ(RunPolicy("arc", 4, trace, 3), 15u);
    EXPECT_EQ(RunPolicy("arc_visit", 4, trace, 3), 12u);

    // With a window of 0 no run is a touch, so visit mode is plain mode.
    for (const std::string base : {"lru2", "arc", "lirs"}) {
        EXPECT_EQ(FastFlags(base + "_visit", 4, trace, 0), FastFlags(base, 4, trace, 0)) << base;
    }
}

TEST(EvictionSimTest, FastMatchesLiteralReference) {
    const auto check = [](const std::string& name, size_t pool, const Trace& trace, uint32_t window,
                          const std::vector<bool>& expected, int round) {
        const auto actual = FastFlags(name, pool, trace, window);
        const auto diff = std::mismatch(expected.begin(), expected.end(), actual.begin(), actual.end());
        EXPECT_TRUE(diff.first == expected.end() && diff.second == actual.end())
            << name << " window " << window << " pool " << pool << " round " << round
            << ": first difference at run " << (diff.first - expected.begin());
    };

    std::mt19937 rng(17);
    for (int round = 0; round < 30; ++round) {
        const Trace trace = TraceOfRuns(InterleavedRuns(rng, 3000, 120));
        ASSERT_EQ(trace.runs.size(), 3000u);
        const auto& runs = trace.runs;
        for (const size_t pool : {2u, 7u, 40u, 100u}) {
            check("clock", pool, trace, 64, SlowFlags(SlowClock(pool), runs, Mode::PLAIN, 64), round);
            for (const auto& [suffix, mode] : {std::make_pair(std::string(), Mode::PLAIN),
                                               std::make_pair(std::string("_full"), Mode::EVERY_FETCH)}) {
                check("lru2" + suffix, pool, trace, 64, SlowFlags(SlowLru2(pool), runs, mode, 64), round);
                check("arc" + suffix, pool, trace, 64, SlowFlags(SlowArc(pool), runs, mode, 64), round);
                check("lirs" + suffix, pool, trace, 64, SlowFlags(SlowLirs(pool), runs, mode, 64), round);
            }
            for (const uint32_t window : {3u, 64u}) {
                check("lru2_visit", pool, trace, window, SlowFlags(SlowLru2(pool), runs, Mode::VISIT, window), round);
                check("arc_visit", pool, trace, window, SlowFlags(SlowArc(pool), runs, Mode::VISIT, window), round);
                check("lirs_visit", pool, trace, window, SlowFlags(SlowLirs(pool), runs, Mode::VISIT, window), round);
            }
        }
    }

    // Pools under 150 give LIRS a single HIR page, so the order of Q never matters there.
    // These two do: at 140 rounding 1% up would give 2 HIR pages, and at 160 lround gives 2
    // where rounding down gives 1. Its own seed, so these traces don't depend on the loop above.
    std::mt19937 lirs_rng(31);
    for (int round = 0; round < 10; ++round) {
        const Trace trace = TraceOfRuns(InterleavedRuns(lirs_rng, 3000, 400));
        const auto& runs = trace.runs;
        for (const size_t pool : {140u, 160u}) {
            check("lirs", pool, trace, 64, SlowFlags(SlowLirs(pool), runs, Mode::PLAIN, 64), round);
            check("lirs_full", pool, trace, 64, SlowFlags(SlowLirs(pool), runs, Mode::EVERY_FETCH, 64), round);
            for (const uint32_t window : {3u, 64u}) {
                check("lirs_visit", pool, trace, window, SlowFlags(SlowLirs(pool), runs, Mode::VISIT, window), round);
            }
        }
    }
}

TEST(EvictionSimTest, ArcKeepsInvariants) {
    std::mt19937 rng(23);
    for (int round = 0; round < 20; ++round) {
        // A hot set of 8 pages mixed with scans of 20-150 pages from a random start.
        RunList runs;
        uint32_t scan_page = 0, scan_left = 0;
        while (runs.size() < 4000) {
            uint32_t page;
            if (rng() % 2 == 0) {
                page = rng() % 8;
            } else {
                if (scan_left == 0) {
                    scan_page = 8 + rng() % 300;
                    scan_left = 20 + rng() % 131;
                }
                page = scan_page++;
                --scan_left;
            }
            if (!runs.empty() && runs.back().first == page) continue;
            runs.emplace_back(page, 1 + rng() % 3);
        }
        const Trace trace = TraceOfRuns(runs);

        for (const size_t c : {1u, 5u, 64u}) {
            for (const Mode mode : {Mode::PLAIN, Mode::VISIT, Mode::EVERY_FETCH}) {
                ArcPolicy arc(c, trace.page_limit, mode, 64);
                for (size_t r = 0; r < trace.runs.size(); ++r) {
                    (void)arc.Access(r, trace.runs[r]);
                    const ArcSizes s = arc.Sizes();
                    const size_t cache = s.t1 + s.t2, total = cache + s.b1 + s.b2;
                    const bool ok = cache <= c && s.t1 + s.b1 <= c && total <= 2 * c && s.p >= 0.0 &&
                                    s.p <= static_cast<double>(c) &&
                                    (total >= c || (s.b1 == 0 && s.b2 == 0)) &&  // no ghosts before the cache fills
                                    (total < c || cache == c);                   // and full from then on
                    if (!ok) {
                        ADD_FAILURE() << "round " << round << " c " << c << " mode " << static_cast<int>(mode)
                                      << " run " << r << ": T1 " << s.t1 << " T2 " << s.t2 << " B1 " << s.b1
                                      << " B2 " << s.b2 << " p " << s.p;
                        break;
                    }
                }
            }
        }
    }
}

TEST(EvictionSimTest, PlainAndEveryFetchAgreeForLruClockBelady) {
    std::mt19937 rng(29);
    for (int round = 0; round < 20; ++round) {
        const RunList runs = RandomRuns(rng, 2000, 150, 5);
        const Trace merged = TraceOfRuns(runs);
        const Trace raw = RawTraceOfRuns(runs);
        ASSERT_EQ(merged.runs.size(), runs.size());
        ASSERT_GT(raw.runs.size(), merged.runs.size());
        for (const std::string name : {"lru", "clock", "belady"}) {
            for (const size_t pool : {10u, 50u, 120u}) {
                EXPECT_EQ(RunPolicy(name, pool, merged), RunPolicy(name, pool, raw))
                    << name << " pool " << pool << " round " << round;
            }
        }
    }
}

TEST(EvictionSimTest, BeladyNeverMissesMoreThanAnyPolicy) {
    std::mt19937 rng(5);
    for (int round = 0; round < 50; ++round) {
        const Trace trace = TraceOfRuns(RandomRuns(rng, 2000, 200, 4));
        for (const size_t pool : {10u, 50u, 150u}) {
            const uint64_t belady = RunPolicy("belady", pool, trace);
            for (const auto& name : PolicyNames()) {
                if (name == "v2" || name == "belady") continue;
                EXPECT_LE(belady, RunPolicy(name, pool, trace)) << name << " pool " << pool << " round " << round;
            }
        }
    }
}

// Hand weights: h1 = 2, h2 = 2.
V2Weights TinyWeights() {
    V2Weights w;
    w.mean = {1.0f, 0.0f, 0.0f, 0.0f};
    w.std = {2.0f, 1.0f, 1.0f, 1.0f};
    w.h1 = 2;
    w.h2 = 2;
    w.w1 = {1, -1, 0, 0, 0, 0, 1, 0};
    w.b1 = {0.0f, 0.5f};
    w.w2 = {1, 1, -1, 2};
    w.b2 = {0.1f, -0.2f};
    w.w3 = {2.0f, 3.0f};
    w.b3 = 0.5f;
    return w;
}

// Scores x0 and nothing else, so the page with the largest recency goes first.
V2Weights RecencyWeights() {
    V2Weights w;
    w.std = {1.0f, 1.0f, 1.0f, 1.0f};
    w.h1 = 2;
    w.h2 = 2;
    w.w1 = {1, 0, 0, 0, 0, 1, 0, 0};
    w.b1 = {0.0f, 0.0f};
    w.w2 = {1, 0, 0, 1};
    w.b2 = {0.0f, 0.0f};
    w.w3 = {1.0f, 0.0f};
    w.b3 = 0.0f;
    return w;
}

TEST(EvictionSimTest, V2TinyNetworkScore) {
    const V2Weights w = TinyWeights();
    const float x[8] = {2, 1, 3, 9, 1, 0, 0, 0};
    float out[2] = {0, 0};
    std::vector<float> scratch;
    V2Score(w, x, 2, out, scratch);
    EXPECT_NEAR(out[0], 28.1f, 1e-5);
    EXPECT_NEAR(out[1], 4.1f, 1e-5);

    // The v2 policy needs weights in its context.
    const Trace trace = TraceOfPages({1, 2, 3});
    SimContext ctx;
    ctx.trace = &trace;
    EXPECT_THROW((void)MakePolicy("v2", 2, ctx), std::invalid_argument);
}

TEST(EvictionSimTest, V2FeaturesMatchEngineArithmetic) {
    // Loaded by a run of 3 at fetch 10 (last 12, second 11), hit by one fetch at 20,
    // evicted at fetch 30: r1 = 9, n = 4, r2 = 30 - 12 - 1 = 17.
    const auto a = V2Features(30, 20, true, 12, 4, 1000);
    EXPECT_EQ(a[0], std::log(1.0f + 9.0f));
    EXPECT_EQ(a[1], std::log(1.0f + 4.0f));
    EXPECT_EQ(a[2], std::log(1.0f + 17.0f));
    EXPECT_EQ(a[3], 0.0f);

    // One fetch at 25, no second access: r2 = clock start + 30.
    const auto b = V2Features(30, 25, false, 0, 1, 1000);
    EXPECT_EQ(b[0], std::log(1.0f + 4.0f));
    EXPECT_EQ(b[1], std::log(1.0f + 1.0f));
    EXPECT_EQ(b[2], std::log(1.0f + 1030.0f));
}

TEST(EvictionSimTest, V2WithRecencyNetworkIsLru) {
    const V2Weights w = RecencyWeights();
    std::mt19937 rng(11);
    for (int round = 0; round < 20; ++round) {
        const Trace trace = TraceOfRuns(RandomRuns(rng, 2000, 120, 5));
        for (const size_t pool : {8u, 30u, 90u}) {
            SimContext ctx;
            ctx.trace = &trace;
            ctx.v2 = &w;
            ctx.v2_clock_start = 7;
            auto v2 = MakePolicy("v2", pool, ctx);
            EXPECT_EQ(Simulate(*v2, trace, {}).misses, RunPolicy("lru", pool, trace))
                << "pool " << pool << " round " << round;
        }
    }
}

TEST(EvictionSimTest, V2WeightsNeedRefLines) {
    const std::string path = TempPath("v2tiny.txt");
    const std::string body =
        "mean 4 0 0 0 0\nstd 4 1 1 1 1\nw1 1 4 1 0 0 0\nb1 1 0\nw2 1 1 1\nb2 1 0\nw3 1 1 1\nb3 1 0\n";
    WriteFile(path, body);
    EXPECT_THROW((void)LoadV2Weights(path), std::runtime_error);
    WriteFile(path, body + "ref 1 2 0 0 0 2\n");
    const V2Weights w = LoadV2Weights(path);
    EXPECT_EQ(w.refs.size(), 1u);
    EXPECT_LT(V2MaxRefError(w), 1e-6);
    std::filesystem::remove(path);
}

TEST(EvictionSimTest, V2WeightsRejectNonFiniteValues) {
    const std::string path = TempPath("v2bad.txt");
    const std::string tail = "w1 1 4 1 0 0 0\nb1 1 0\nw2 1 1 1\nb2 1 0\nw3 1 1 1\nb3 1 0\nref 1 2 0 0 0 2\n";
    for (const std::string head : {"mean 4 nan 0 0 0\nstd 4 1 1 1 1\n", "mean 4 0 0 0 0\nstd 4 1 inf 1 1\n",
                                   "mean 4 0 0 0 0\nstd 4 0 1 1 1\n"}) {
        WriteFile(path, head + tail);
        EXPECT_THROW((void)LoadV2Weights(path), std::runtime_error) << head;
    }

    // A reference score of NaN can't be matched, so the check reports an infinite gap.
    WriteFile(path, "mean 4 0 0 0 0\nstd 4 1 1 1 1\n" + tail);
    V2Weights w = LoadV2Weights(path);
    w.refs[0][4] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_TRUE(std::isinf(V2MaxRefError(w)));
    std::filesystem::remove(path);
}

TEST(EvictionSimTest, V2ExportedWeightsMatchTorch) {
    std::string path;
    if (const char* env = std::getenv("SHILMANDB_V2_WEIGHTS")) {
        path = env;
    } else {
        for (const char* rel : {"bench/results/headroom/eviction/v2_weights.txt",
                                "../bench/results/headroom/eviction/v2_weights.txt",
                                "../../bench/results/headroom/eviction/v2_weights.txt"}) {
            if (std::filesystem::exists(rel)) {
                path = rel;
                break;
            }
        }
    }
    if (path.empty() || !std::filesystem::exists(path)) GTEST_SKIP() << "no exported V2 weights";
    const V2Weights w = LoadV2Weights(path);
    EXPECT_EQ(w.h1, 128u);
    EXPECT_EQ(w.h2, 64u);
    EXPECT_EQ(w.refs.size(), 256u);
    EXPECT_LT(V2MaxRefError(w), 1e-4);
}

}  // namespace
}  // namespace shilmandb::evsim
