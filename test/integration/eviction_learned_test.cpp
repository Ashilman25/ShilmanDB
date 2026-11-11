#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "eviction/sim/eviction_learned.hpp"

namespace shilmandb::evsim {
namespace {

using Lines = std::vector<std::pair<uint64_t, uint32_t>>;
using RunList = std::vector<std::pair<uint32_t, uint32_t>>;  // (page, fetches)

std::string TempPath(const std::string& name) {
    return (std::filesystem::temp_directory_path() /
            ("shilmandb_evlearned_" + name + "_" + std::to_string(::getpid())))
        .string();
}

void WriteFile(const std::string& path, const std::string& text) {
    std::ofstream out(path);
    out << text;
}

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

// Cuts the trace into `count` items of whole runs, the way a workload's items cover it.
std::vector<Item> ItemsOver(const Trace& trace, size_t count) {
    std::vector<Item> items(count);
    const size_t n = trace.runs.size();
    const auto fetch_at = [&](size_t run) { return run < n ? trace.runs[run].first_fetch : trace.end_fetch; };
    for (size_t i = 0; i < count; ++i) {
        items[i].seq = static_cast<int>(i);
        items[i].trace_start = fetch_at(i * n / count);
        items[i].trace_end = fetch_at((i + 1) * n / count);
    }
    MapItems(trace, items);
    return items;
}

SampledConfig ConfigWithK(size_t k) {
    SampledConfig config;
    config.k = k;
    return config;
}

float Lg(uint64_t x) { return static_cast<float>(std::log1p(static_cast<double>(x))); }

// The what() of whatever fn throws, or "" if it doesn't throw.
template <typename Fn>
std::string ErrorOf(Fn&& fn) {
    try {
        fn();
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

// A basic 3 -> 2 -> 1 network I can score by hand.
const char* const kTinyMlp =
    "model mlp\n"
    "features basic\n"
    "inputs 3\n"
    "pool 9\n"
    "sf 0.01\n"
    "mean 3 0 0 0\n"
    "std 3 1 1 1\n"
    "layer 2 3 1 0 0 0 1 -1 0 0.5\n"
    "layer 1 2 2 -1 0.25\n"
    "ref 2\n"
    "1 2 3 2.25\n"
    "0 3 1 -2.25\n";

// Two one-split trees: feature 0 at 0.5 (-1 / +1), feature 2 at 1.0 (0.25 / 0.75).
const char* const kTinyGbdt =
    "model gbdt\n"
    "features basic\n"
    "inputs 3\n"
    "pool 9\n"
    "sf 0.01\n"
    "trees 2\n"
    "tree 3\n"
    "0 0.5 1 2 0\n"
    "-1 0 -1 -1 -1\n"
    "-1 0 -1 -1 1\n"
    "tree 3\n"
    "2 1 1 2 0\n"
    "-1 0 -1 -1 0.25\n"
    "-1 0 -1 -1 0.75\n"
    "ref 2\n"
    "0.5 0 1 -0.75\n"
    "0.6 0 1.1 1.75\n";

// Scores a page by minus log1p(fetches since its last fetch): the least recently used is lowest.
const char* const kRecencyMlp =
    "model mlp\n"
    "features basic\n"
    "inputs 3\n"
    "pool 9\n"
    "sf 0.01\n"
    "mean 3 0 0 0\n"
    "std 3 1 1 1\n"
    "layer 1 3 1 0 0 0\n"
    "layer 1 1 -1 0\n"
    "ref 2\n"
    "0 0 0 0\n"
    "1.5 2 3 -1.5\n";

LearnedModel LoadText(const std::string& name, const std::string& text) {
    const std::string path = TempPath(name);
    WriteFile(path, text);
    try {
        LearnedModel model = LoadLearnedModel(path);
        std::remove(path.c_str());
        return model;
    } catch (...) {
        std::remove(path.c_str());
        throw;
    }
}

// The text with its one `from` swapped for `to`.
std::string Swapped(std::string text, const std::string& from, const std::string& to) {
    const size_t at = text.find(from);
    EXPECT_NE(at, std::string::npos) << from;
    EXPECT_EQ(text.find(from, at + 1), std::string::npos) << from;
    if (at != std::string::npos) text.replace(at, from.size(), to);
    return text;
}

float RandomWeight(std::mt19937& rng) { return static_cast<float>(rng() % 2001) / 1000.0f - 1.0f; }

// A network with random weights; outs lists each layer's width, the last one 1.
LearnedModel RandomMlp(std::mt19937& rng, FeatureSet features, const std::vector<size_t>& outs) {
    LearnedModel model;
    model.type = LearnedModel::Type::MLP;
    model.features = features;
    model.inputs = FeatureCount(features);
    model.pool = 9;
    for (size_t i = 0; i < model.inputs; ++i) {
        model.mean.push_back(RandomWeight(rng));
        model.std.push_back(0.5f + static_cast<float>(rng() % 1000) / 500.0f);
    }
    size_t in = model.inputs;
    for (const size_t out : outs) {
        LearnedModel::Layer layer;
        layer.out = out;
        layer.in = in;
        for (size_t i = 0; i < out * in; ++i) layer.w.push_back(RandomWeight(rng));
        for (size_t i = 0; i < out; ++i) layer.b.push_back(RandomWeight(rng));
        model.layers.push_back(std::move(layer));
        in = out;
    }
    return model;
}

// Full trees three splits deep: nodes 0-6 split, 7-14 are leaves.
LearnedModel RandomGbdt(std::mt19937& rng, FeatureSet features, size_t trees) {
    LearnedModel model;
    model.type = LearnedModel::Type::GBDT;
    model.features = features;
    model.inputs = FeatureCount(features);
    model.pool = 9;
    for (size_t t = 0; t < trees; ++t) {
        std::vector<LearnedModel::Node> nodes(15);
        for (int32_t i = 0; i < 15; ++i) {
            if (i < 7) {
                nodes[i].feature = static_cast<int32_t>(rng() % model.inputs);
                nodes[i].threshold = static_cast<double>(rng() % 5000) / 1000.0;
                nodes[i].left = 2 * i + 1;
                nodes[i].right = 2 * i + 2;
            } else {
                nodes[i].value = RandomWeight(rng);
            }
        }
        model.trees.push_back(std::move(nodes));
    }
    return model;
}


TEST(EvictionLearnedTest, FeatureSetsAndNames) {
    EXPECT_EQ(FeatureCount(FeatureSet::BASIC), 3u);
    EXPECT_EQ(FeatureCount(FeatureSet::RICHER_NO_OWNER), 11u);
    EXPECT_EQ(FeatureCount(FeatureSet::RICHER), 29u);
    EXPECT_EQ(kMaxFeatures, 29u);

    EXPECT_EQ(ParseFeatureSet("basic"), FeatureSet::BASIC);
    EXPECT_EQ(ParseFeatureSet("richer_no_owner"), FeatureSet::RICHER_NO_OWNER);
    EXPECT_EQ(ParseFeatureSet("richer"), FeatureSet::RICHER);
    EXPECT_FALSE(ParseFeatureSet("Richer").has_value());
    EXPECT_FALSE(ParseFeatureSet("").has_value());

    const std::vector<std::string> expected = {
        "b_since_fetch", "b_fetches", "b_since_second",
        "h_since_run", "h_visits", "h_since_visit", "h_gap1", "h_gap2", "h_gap3", "h_last_run_len", "h_age",
        "o_table", "o_inner", "o_leaf",
        "o_obj0", "o_obj1", "o_obj2", "o_obj3", "o_obj4", "o_obj5", "o_obj6", "o_obj7",
        "o_obj8", "o_obj9", "o_obj10", "o_obj11", "o_obj12", "o_obj13", "o_obj14"};
    const auto& names = FeatureNames();
    ASSERT_EQ(names.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) EXPECT_EQ(names[i], expected[i]) << i;
}

TEST(EvictionLearnedTest, FeaturesMatchHandComputedValues) {
    // A x3, B x1, A x2, then 70 one-fetch runs of new pages, then A x1, D x1.
    const uint32_t A = 0, B = 1, D = 2, C0 = 3;
    RunList runs = {{A, 3}, {B, 1}, {A, 2}};
    for (uint32_t i = 0; i < 70; ++i) runs.emplace_back(C0 + i, 1);
    runs.emplace_back(A, 1);
    runs.emplace_back(D, 1);
    const Trace trace = TraceOfRuns(runs);
    ASSERT_EQ(trace.runs.size(), 75u);
    ASSERT_EQ(trace.runs[73].page, A);
    ASSERT_EQ(trace.runs[73].first_fetch, 76u);
    ASSERT_EQ(trace.runs[74].first_fetch, 77u);

    OwnerMap owners;
    owners.kind = {PageKind::LEAF, PageKind::TABLE};
    owners.object = {9, 6};

    PageHistory history(trace.page_limit, &owners);
    EXPECT_EQ(history.LastRun(A), kNever);
    for (size_t r = 0; r <= 73; ++r) history.Update(r, trace.runs[r]);
    EXPECT_EQ(history.LastRun(A), 73u);
    EXPECT_EQ(history.LastRun(B), 1u);
    EXPECT_EQ(history.LastRun(D), kNever);

    const auto expect = [&](uint32_t page, const std::array<uint64_t, 11>& numbers, size_t kind_hot, size_t object_hot) {
        std::array<float, kMaxFeatures> out;
        out.fill(-1.0f);
        history.Features(page, 74, 77, kMaxFeatures, out.data());
        for (size_t i = 0; i < 11; ++i) EXPECT_FLOAT_EQ(out[i], Lg(numbers[i])) << "page " << page << " " << FeatureNames()[i];
        for (size_t i = 11; i < kMaxFeatures; ++i) {
            const float want = i == 11 + kind_hot || i == 14 + object_hot ? 1.0f : 0.0f;
            EXPECT_EQ(out[i], want) << "page " << page << " " << FeatureNames()[i];
        }
    };
    //          since_fetch fetches since_second since_run visits since_visit gap1 gap2 gap3 run_len age
    expect(A, {0, 6, 71, 0, 2, 0, 73, 0, 0, 1, 73}, 2, 9);
    expect(B, {73, 1, 77, 72, 1, 72, 0, 0, 0, 1, 72}, 0, 6);

    // A page past the end of the owner map gets no one-hot at all.
    std::array<float, kMaxFeatures> out;
    out.fill(-1.0f);
    history.Features(C0, 74, 77, kMaxFeatures, out.data());
    EXPECT_FLOAT_EQ(out[0], Lg(77 - 6 - 1));
    for (size_t i = 11; i < kMaxFeatures; ++i) EXPECT_EQ(out[i], 0.0f) << i;
}

TEST(EvictionLearnedTest, VisitsAndGapsFollowTheWindow) {
    // Page 0 at runs 0, 64, 129, 200, 300, 302: a break of more than 64 runs starts a visit.
    const std::vector<size_t> at = {0, 64, 129, 200, 300, 302};
    RunList runs;
    uint32_t filler = 1;
    for (size_t r = 0; r <= 302; ++r) {
        const bool mine = std::find(at.begin(), at.end(), r) != at.end();
        runs.emplace_back(mine ? 0 : filler++, r == 302 ? 5 : 1);
    }
    const Trace trace = TraceOfRuns(runs);
    ASSERT_EQ(trace.runs.size(), 303u);

    PageHistory history(trace.page_limit, nullptr);
    for (size_t r = 0; r < trace.runs.size(); ++r) history.Update(r, trace.runs[r]);

    // Visits start at runs 0 (with 64), 129, 200, 300 (with 302).
    std::array<float, 11> out;
    const size_t r = 303;
    const uint64_t t = trace.end_fetch;
    history.Features(0, r, t, 11, out.data());
    EXPECT_FLOAT_EQ(out[0], Lg(0));           // last fetch is the trace's last
    EXPECT_FLOAT_EQ(out[1], Lg(5 + 5));       // five one-fetch runs and one of five
    EXPECT_FLOAT_EQ(out[2], Lg(1));           // the fetch before the last one
    EXPECT_FLOAT_EQ(out[3], Lg(0));
    EXPECT_FLOAT_EQ(out[4], Lg(4));
    EXPECT_FLOAT_EQ(out[5], Lg(303 - 300 - 1));
    EXPECT_FLOAT_EQ(out[6], Lg(100));
    EXPECT_FLOAT_EQ(out[7], Lg(71));
    EXPECT_FLOAT_EQ(out[8], Lg(129));
    EXPECT_FLOAT_EQ(out[9], Lg(5));
    EXPECT_FLOAT_EQ(out[10], Lg(302));
}

TEST(EvictionLearnedTest, FeaturePrefixesMatch) {
    std::mt19937 rng(11);
    const uint32_t pages = 40;
    const Trace trace = TraceOfRuns(RandomRuns(rng, 3000, pages, 4));

    OwnerMap owners;
    for (uint32_t p = 0; p < pages; ++p) {
        const auto kind = static_cast<PageKind>(rng() % 4);
        owners.kind.push_back(kind);
        const int object = kind == PageKind::OTHER ? -1 : kind == PageKind::TABLE ? static_cast<int>(rng() % 7) : 7 + static_cast<int>(rng() % 8);
        owners.object.push_back(static_cast<int8_t>(object));
    }

    PageHistory with_owners(trace.page_limit, &owners);
    PageHistory without(trace.page_limit, nullptr);
    size_t checked = 0;
    for (size_t r = 0; r < trace.runs.size(); ++r) {
        const uint64_t t = trace.runs[r].first_fetch;
        for (uint32_t p = 0; p < trace.page_limit && r % 7 == 0; ++p) {
            if (with_owners.LastRun(p) == kNever) continue;
            std::array<float, kMaxFeatures> all, some, bare;
            with_owners.Features(p, r, t, 29, all.data());
            for (const size_t n : {size_t{3}, size_t{11}}) {
                some.fill(-7.0f);
                bare.fill(-7.0f);
                with_owners.Features(p, r, t, n, some.data());
                without.Features(p, r, t, n, bare.data());
                for (size_t i = 0; i < n; ++i) {
                    ASSERT_EQ(some[i], all[i]) << "run " << r << " page " << p << " n " << n << " feature " << i;
                    ASSERT_EQ(bare[i], all[i]) << "run " << r << " page " << p << " n " << n << " feature " << i;
                }
                // Nothing past n is written.
                for (size_t i = n; i < kMaxFeatures; ++i) ASSERT_EQ(some[i], -7.0f) << i;
            }
            // Exactly one kind and one object bit, unless nothing owns the page.
            const float kinds = std::accumulate(all.begin() + 11, all.begin() + 14, 0.0f);
            const float objects = std::accumulate(all.begin() + 14, all.end(), 0.0f);
            const float want = owners.kind[p] == PageKind::OTHER ? 0.0f : 1.0f;
            ASSERT_EQ(kinds, want);
            ASSERT_EQ(objects, want);
            ++checked;
        }
        with_owners.Update(r, trace.runs[r]);
        without.Update(r, trace.runs[r]);
    }
    EXPECT_GT(checked, 10000u);
}

TEST(EvictionLearnedTest, OwnerMapLoadsAndCoverageThrowsWithPageId) {
    const std::string path = TempPath("owners.csv");
    WriteFile(path, "page_id,kind,object,owner\n"
                    "0,table,6,lineitem\n"
                    "2,inner,7,idx_orders_custkey\n"
                    "1,leaf,14,idx_part_partkey\n"
                    "3,other,-1,\n");
    const OwnerMap owners = LoadPageOwners(path);
    ASSERT_EQ(owners.kind.size(), 4u);
    ASSERT_EQ(owners.object.size(), 4u);
    EXPECT_EQ(owners.kind[0], PageKind::TABLE);
    EXPECT_EQ(owners.kind[1], PageKind::LEAF);
    EXPECT_EQ(owners.kind[2], PageKind::INNER);
    EXPECT_EQ(owners.kind[3], PageKind::OTHER);
    EXPECT_EQ(owners.object[0], 6);
    EXPECT_EQ(owners.object[1], 14);
    EXPECT_EQ(owners.object[2], 7);
    EXPECT_EQ(owners.object[3], -1);

    // Every page owned: fine.
    EXPECT_NO_THROW(CheckTraceCovered(owners, TraceOfRuns({{0, 2}, {1, 1}, {2, 3}, {0, 1}})));

    // A page past the end of the map.
    const std::string past = ErrorOf([&] { CheckTraceCovered(owners, TraceOfRuns({{0, 1}, {1, 1}, {17, 1}, {5, 1}})); });
    EXPECT_NE(past.find("page 17"), std::string::npos) << past;
    EXPECT_EQ(past.find("page 5"), std::string::npos) << past;

    // A page the map knows only as "other".
    const std::string other = ErrorOf([&] { CheckTraceCovered(owners, TraceOfRuns({{0, 1}, {3, 1}, {2, 1}})); });
    EXPECT_NE(other.find("page 3"), std::string::npos) << other;

    // A page id the file skips is "other" too.
    WriteFile(path, "page_id,kind,object,owner\n0,table,0,region\n2,table,1,nation\n");
    const OwnerMap gap = LoadPageOwners(path);
    ASSERT_EQ(gap.kind.size(), 3u);
    EXPECT_EQ(gap.kind[1], PageKind::OTHER);
    EXPECT_EQ(gap.object[1], -1);
    const std::string skipped = ErrorOf([&] { CheckTraceCovered(gap, TraceOfRuns({{0, 1}, {1, 1}})); });
    EXPECT_NE(skipped.find("page 1"), std::string::npos) << skipped;

    // Files that must not load.
    const std::string header = "page_id,kind,object,owner\n";
    const std::vector<std::string> bad_files = {
        "",                                          // empty
        "page_id,owner,kind\n0,region,table\n",      // another header
        header,                                      // no pages
        header + "0,table,0\n",                      // too few fields
        header + "0,table,0,region,extra\n",         // too many
        header + "x,table,0,region\n",               // page id
        header + "-1,table,0,region\n",
        header + "0,heap,0,region\n",                // kind
        header + "0,table,zero,region\n",            // object
        header + "0,table,,region\n",
        header + "0,table,15,region\n",              // object out of range
        header + "0,table,7,region\n",               // an index number on a table page
        header + "0,leaf,6,idx\n",                   // a table number on an index page
        header + "0,other,3,\n",
        header + "0,table,0,region\n0,table,0,region\n",  // listed twice
    };
    for (size_t i = 0; i < bad_files.size(); ++i) {
        WriteFile(path, bad_files[i]);
        EXPECT_THROW((void)LoadPageOwners(path), std::runtime_error) << "file " << i;
    }

    std::remove(path.c_str());
    EXPECT_THROW((void)LoadPageOwners(path), std::runtime_error);  // no such file
}

TEST(EvictionLearnedTest, BeladyHitsMatchBeladyPolicy) {
    std::mt19937 rng(5);
    for (int round = 0; round < 50; ++round) {
        const Trace trace = TraceOfRuns(RandomRuns(rng, 300 + rng() % 200, 5 + rng() % 40, 3));
        const auto next_use = NextUse(trace);
        for (const size_t pool : {size_t{1}, size_t{2}, size_t{5}, size_t{17}}) {
            const auto hits = BeladyHits(trace, pool, next_use);
            ASSERT_EQ(hits.size(), trace.runs.size());
            const uint64_t hit_count = std::accumulate(hits.begin(), hits.end(), uint64_t{0});

            BeladyPolicy belady(pool, trace.page_limit, next_use);
            const uint64_t misses = Simulate(belady, trace, {}).misses;
            EXPECT_EQ(trace.runs.size() - hit_count, misses) << "round " << round << " pool " << pool;
        }
    }
}

TEST(EvictionLearnedTest, LabelsMatchBruteForce) {
    std::mt19937 rng(9);
    for (int round = 0; round < 30; ++round) {
        const Trace trace = TraceOfRuns(RandomRuns(rng, 60 + rng() % 60, 4 + rng() % 9, 3));
        const auto next_use = NextUse(trace);
        for (const size_t pool : {size_t{2}, size_t{3}, size_t{7}}) {
            // Belady by hand: the pool is a plain list of (page, the run it was last used at).
            // On a full miss the page used furthest ahead goes (the bigger page id on a tie),
            // and the run that last used it did not get kept until its next use.
            std::vector<uint8_t> kept(trace.runs.size(), 0);
            std::vector<std::pair<uint32_t, uint32_t>> resident;
            for (size_t r = 0; r < trace.runs.size(); ++r) {
                const uint32_t page = trace.runs[r].page;
                auto it = std::find_if(resident.begin(), resident.end(), [&](const auto& e) { return e.first == page; });
                if (it == resident.end()) {
                    if (resident.size() == pool) {
                        const auto victim = std::max_element(resident.begin(), resident.end(), [&](const auto& a, const auto& b) {
                            return std::make_pair(next_use[a.second], a.first) < std::make_pair(next_use[b.second], b.first);
                        });
                        kept[victim->second] = 0;
                        resident.erase(victim);
                    }
                    resident.emplace_back(page, static_cast<uint32_t>(r));
                } else {
                    it->second = static_cast<uint32_t>(r);
                }
                kept[r] = next_use[r] != kNever ? 1 : 0;
            }

            const auto hits = BeladyHits(trace, pool, next_use);
            size_t ones = 0;
            for (size_t i = 0; i < trace.runs.size(); ++i) {
                const uint8_t label = RelaxedBeladyLabel(static_cast<uint32_t>(i), next_use, hits);
                ASSERT_EQ(label, kept[i]) << "round " << round << " pool " << pool << " run " << i;
                if (next_use[i] == kNever) ASSERT_EQ(label, 0);
                ones += label;
            }
            // Every hit is some earlier run's kept page, so the two counts are the same.
            EXPECT_EQ(ones, static_cast<size_t>(std::accumulate(hits.begin(), hits.end(), 0)));
        }
    }
}

TEST(EvictionLearnedTest, SampleRngDrawsLikeTheStreamGenerator) {
    // The stream generator's rule: a 64-bit Mersenne Twister draw, redrawn while it is under
    // (2^64 mod span), then taken mod span.
    std::mt19937_64 engine(7);
    SampleRng rng(7);
    for (int i = 0; i < 2000; ++i) {
        const int64_t lo = i % 5, hi = lo + (i * 37) % 1000;
        const uint64_t span = static_cast<uint64_t>(hi - lo) + 1;
        const uint64_t too_low = (0 - span) % span;
        uint64_t draw = engine();
        while (draw < too_low) draw = engine();
        ASSERT_EQ(rng.Uniform(lo, hi), lo + static_cast<int64_t>(draw % span)) << i;
    }
    EXPECT_EQ(rng.Uniform(4, 4), 4);
}

TEST(EvictionLearnedTest, SampledLruWithFullSampleIsLru) {
    std::mt19937 rng(21);
    for (int round = 0; round < 40; ++round) {
        const Trace trace = TraceOfRuns(RandomRuns(rng, 300 + rng() % 200, 5 + rng() % 40, 3));
        const auto items = ItemsOver(trace, 7);
        for (const size_t pool : {size_t{1}, size_t{2}, size_t{5}, size_t{17}}) {
            LruPolicy lru(pool, trace.page_limit);
            const SimResult want = Simulate(lru, trace, items);

            SampledPolicy sampled(pool, trace, SampledScore::LRU, ConfigWithK(std::numeric_limits<size_t>::max()), {});
            const SimResult got = Simulate(sampled, trace, items);
            ASSERT_EQ(got.misses, want.misses) << "round " << round << " pool " << pool;
            ASSERT_EQ(got.item_misses, want.item_misses) << "round " << round << " pool " << pool;
            ASSERT_EQ(got.evictions, want.evictions);
            ASSERT_EQ(sampled.Decisions(), want.evictions);
        }
    }
}

TEST(EvictionLearnedTest, SampledBeladyWithFullSampleIsBelady) {
    std::mt19937 rng(22);
    for (int round = 0; round < 40; ++round) {
        const Trace trace = TraceOfRuns(RandomRuns(rng, 300 + rng() % 200, 5 + rng() % 40, 3));
        const auto items = ItemsOver(trace, 7);
        const auto next_use = NextUse(trace);
        SampledInputs inputs;
        inputs.next_use = &next_use;
        for (const size_t pool : {size_t{1}, size_t{2}, size_t{5}, size_t{17}}) {
            BeladyPolicy belady(pool, trace.page_limit, next_use);
            const SimResult want = Simulate(belady, trace, items);

            SampledPolicy sampled(pool, trace, SampledScore::BELADY, ConfigWithK(std::numeric_limits<size_t>::max()), inputs);
            const SimResult got = Simulate(sampled, trace, items);
            ASSERT_EQ(got.misses, want.misses) << "round " << round << " pool " << pool;
            ASSERT_EQ(got.item_misses, want.item_misses) << "round " << round << " pool " << pool;
        }
    }
}

TEST(EvictionLearnedTest, SamplerDrawsDistinctResidentPages) {
    std::mt19937 rng(23);
    const Trace trace = TraceOfRuns(RandomRuns(rng, 6000, 200, 3));
    SampledPolicy policy(40, trace, SampledScore::RANDOM, ConfigWithK(32), {});
    EXPECT_TRUE(policy.LastCandidates().empty());

    std::set<uint32_t> victims;
    uint64_t decisions = 0;
    for (size_t r = 0; r < trace.runs.size(); ++r) {
        const std::vector<uint32_t> before = policy.Resident();
        const bool miss = policy.Access(r, trace.runs[r]);
        ASSERT_EQ(policy.Evictions(), policy.Decisions());
        if (policy.Decisions() == decisions) {
            ASSERT_TRUE(!miss || before.size() < 40);
            continue;
        }
        ASSERT_EQ(++decisions, policy.Decisions());
        ASSERT_TRUE(miss);
        ASSERT_EQ(before.size(), 40u);

        const std::vector<uint32_t>& drawn = policy.LastCandidates();
        ASSERT_EQ(drawn.size(), 32u) << "run " << r;
        const std::set<uint32_t> distinct(drawn.begin(), drawn.end());
        ASSERT_EQ(distinct.size(), 32u) << "run " << r;
        const std::set<uint32_t> was(before.begin(), before.end());
        for (const uint32_t page : drawn) ASSERT_EQ(was.count(page), 1u) << "run " << r << " page " << page;

        // Exactly one page left the pool, it was drawn, and the new page took its place.
        const std::set<uint32_t> now(policy.Resident().begin(), policy.Resident().end());
        ASSERT_EQ(now.size(), 40u);
        ASSERT_EQ(now.count(trace.runs[r].page), 1u);
        std::vector<uint32_t> gone;
        for (const uint32_t page : before) {
            if (now.count(page) == 0) gone.push_back(page);
        }
        ASSERT_EQ(gone.size(), 1u);
        ASSERT_EQ(distinct.count(gone[0]), 1u) << "run " << r;
        // With every score equal, the first page drawn goes.
        ASSERT_EQ(gone[0], drawn[0]) << "run " << r;
        victims.insert(gone[0]);
    }
    EXPECT_GT(decisions, 3000u);
    EXPECT_GT(victims.size(), 150u);  // the draw moves around the pool
}

TEST(EvictionLearnedTest, FewerResidentThanKUsesAll) {
    std::mt19937 rng(24);
    const Trace trace = TraceOfRuns(RandomRuns(rng, 2000, 30, 3));
    // The same seed with a sample that is exactly the pool, or far bigger: no draw either way.
    SampledPolicy policy(5, trace, SampledScore::LRU, ConfigWithK(32), {});
    SampledPolicy exact(5, trace, SampledScore::LRU, ConfigWithK(5), {});
    LruPolicy lru(5, trace.page_limit);

    uint64_t decisions = 0;
    for (size_t r = 0; r < trace.runs.size(); ++r) {
        const std::vector<uint32_t> before = policy.Resident();
        const bool miss = policy.Access(r, trace.runs[r]);
        ASSERT_EQ(miss, lru.Access(r, trace.runs[r])) << "run " << r;
        ASSERT_EQ(miss, exact.Access(r, trace.runs[r])) << "run " << r;
        if (policy.Decisions() == decisions) continue;
        ++decisions;
        // All five, in the pool's own order.
        ASSERT_EQ(policy.LastCandidates(), before) << "run " << r;
        ASSERT_EQ(exact.LastCandidates(), before) << "run " << r;
    }
    EXPECT_GT(decisions, 1000u);
}

TEST(EvictionLearnedTest, SamplingIsDeterministicForASeed) {
    std::mt19937 rng(25);
    const Trace trace = TraceOfRuns(RandomRuns(rng, 5000, 300, 3));
    const auto items = ItemsOver(trace, 20);
    const auto run = [&](uint64_t seed) {
        SampledConfig config;
        config.seed = seed;
        SampledPolicy policy(64, trace, SampledScore::RANDOM, config, {});
        return Simulate(policy, trace, items);
    };
    const SimResult a = run(1), b = run(1);
    EXPECT_EQ(a.misses, b.misses);
    EXPECT_EQ(a.item_misses, b.item_misses);
    EXPECT_GT(a.evictions, 1000u);

    // The default is seed 1 and 32 candidates.
    SampledPolicy plain(64, trace, SampledScore::RANDOM, SampledConfig{}, {});
    EXPECT_EQ(Simulate(plain, trace, items).item_misses, a.item_misses);
    EXPECT_EQ(SampledConfig{}.k, 32u);
    EXPECT_EQ(SampledConfig{}.seed, 1u);
    EXPECT_FALSE(SampledConfig{}.time_decisions);
}

TEST(EvictionLearnedTest, SampledLabelEvictsOldestLabelZero) {
    const uint32_t A = 0, B = 1, C = 2;
    const Trace trace = TraceOfRuns({{A, 1}, {B, 1}, {A, 1}, {C, 1}});
    const auto next_use = NextUse(trace);
    const auto hits = BeladyHits(trace, 2, next_use);
    SampledInputs inputs;
    inputs.next_use = &next_use;
    inputs.belady_hits = &hits;

    SampledPolicy policy(2, trace, SampledScore::LABEL, SampledConfig{}, inputs);
    EXPECT_TRUE(policy.Access(0, trace.runs[0]));
    EXPECT_TRUE(policy.Access(1, trace.runs[1]));
    EXPECT_FALSE(policy.Access(2, trace.runs[2]));
    EXPECT_EQ(policy.Resident(), (std::vector<uint32_t>{A, B}));
    EXPECT_EQ(policy.Decisions(), 0u);

    // Neither page is used again, so both have label 0 and the older one (B) goes.
    EXPECT_TRUE(policy.Access(3, trace.runs[3]));
    EXPECT_EQ(policy.Decisions(), 1u);
    EXPECT_EQ(policy.LastCandidates(), (std::vector<uint32_t>{A, B}));
    std::vector<uint32_t> now = policy.Resident();
    std::sort(now.begin(), now.end());
    EXPECT_EQ(now, (std::vector<uint32_t>{A, C}));
    EXPECT_EQ(policy.History().LastRun(A), 2u);
    EXPECT_EQ(policy.History().LastRun(C), 3u);
}

TEST(EvictionLearnedTest, SampledLabelKeepsLabelOneOverNewerLabelZero) {
    // Pool 2, runs A, B, C, A. At C: A is older, but Belady keeps it until run 3 (label 1),
    // and B is never used again (label 0). So B goes, though plain LRU would evict A.
    const uint32_t A = 0, B = 1, C = 2;
    const Trace trace = TraceOfRuns({{A, 1}, {B, 1}, {C, 1}, {A, 1}});
    const auto next_use = NextUse(trace);
    const auto hits = BeladyHits(trace, 2, next_use);
    ASSERT_EQ(hits[3], 1);
    SampledInputs inputs;
    inputs.next_use = &next_use;
    inputs.belady_hits = &hits;

    SampledPolicy policy(2, trace, SampledScore::LABEL, SampledConfig{}, inputs);
    for (size_t r = 0; r < 3; ++r) EXPECT_TRUE(policy.Access(r, trace.runs[r]));
    EXPECT_FALSE(policy.Access(3, trace.runs[3]));
}

TEST(EvictionLearnedTest, HookRunsBeforeEachEvictionAndTimingIsOptIn) {
    std::mt19937 rng(26);
    const Trace trace = TraceOfRuns(RandomRuns(rng, 3000, 100, 3));

    SampledConfig timed;
    timed.time_decisions = true;
    SampledPolicy policy(20, trace, SampledScore::LRU, timed, {});
    SampledPolicy untimed(20, trace, SampledScore::LRU, SampledConfig{}, {});

    uint64_t calls = 0;
    policy.SetDecisionHook([&](size_t run_index, const evsim::Run& run) {
        // The pool is still whole, the missing page is not in it, and nothing was counted yet.
        EXPECT_EQ(policy.Resident().size(), 20u);
        EXPECT_EQ(policy.Decisions(), calls);
        EXPECT_EQ(run.page, trace.runs[run_index].page);
        EXPECT_EQ(std::count(policy.Resident().begin(), policy.Resident().end(), run.page), 0);
        EXPECT_NE(policy.History().LastRun(run.page), static_cast<uint32_t>(run_index));
        ++calls;
    });
    for (size_t r = 0; r < trace.runs.size(); ++r) {
        ASSERT_EQ(policy.Access(r, trace.runs[r]), untimed.Access(r, trace.runs[r]));
    }
    EXPECT_GT(calls, 1000u);
    EXPECT_EQ(calls, policy.Decisions());
    EXPECT_EQ(policy.DecisionNs().size(), policy.Decisions());
    EXPECT_TRUE(untimed.DecisionNs().empty());
    EXPECT_EQ(untimed.Decisions(), policy.Decisions());
}

TEST(EvictionLearnedTest, SampledPolicyRefusesMissingInputs) {
    const Trace trace = TraceOfRuns({{0, 1}, {1, 1}, {0, 1}});
    const auto next_use = NextUse(trace);
    const auto hits = BeladyHits(trace, 2, next_use);
    const std::vector<uint32_t> short_next(2, kNever);
    const SampledConfig config;
    const auto make = [&](size_t pool, SampledScore score, const SampledConfig& c, const std::vector<uint32_t>* next, const std::vector<uint8_t>* h) {
        SampledInputs inputs;
        inputs.next_use = next;
        inputs.belady_hits = h;
        return std::make_unique<SampledPolicy>(pool, trace, score, c, inputs);
    };
    EXPECT_NO_THROW(make(2, SampledScore::LABEL, config, &next_use, &hits));
    EXPECT_THROW(make(0, SampledScore::LRU, config, nullptr, nullptr), std::invalid_argument);
    EXPECT_THROW(make(2, SampledScore::LRU, ConfigWithK(0), nullptr, nullptr), std::invalid_argument);
    EXPECT_THROW(make(2, SampledScore::BELADY, config, nullptr, nullptr), std::invalid_argument);
    EXPECT_THROW(make(2, SampledScore::BELADY, config, &short_next, nullptr), std::invalid_argument);
    EXPECT_THROW(make(2, SampledScore::LABEL, config, &next_use, nullptr), std::invalid_argument);
    EXPECT_THROW(make(2, SampledScore::LABEL, config, nullptr, &hits), std::invalid_argument);
    EXPECT_THROW(make(2, SampledScore::MODEL, config, nullptr, nullptr), std::invalid_argument);
}

TEST(EvictionLearnedTest, MakeAnyPolicyKnowsEveryName) {
    EXPECT_EQ(LearnedPolicyNames(), (std::vector<std::string>{"learned", "sampled_random", "sampled_lru", "sampled_label", "sampled_belady"}));

    std::mt19937 rng(27);
    const Trace trace = TraceOfRuns(RandomRuns(rng, 800, 60, 3));
    const auto items = ItemsOver(trace, 5);
    const auto next_use = NextUse(trace);
    const size_t pool = 8;

    SimContext ctx;
    ctx.trace = &trace;
    ctx.next_use = &next_use;
    LearnedContext lctx;
    lctx.belady_hits[pool] = BeladyHits(trace, pool, next_use);

    // Every name builds and replays. The learned one needs a model first.
    const LearnedModel model = LoadText("factory_model", kRecencyMlp);
    for (const std::string& name : LearnedPolicyNames()) {
        if (name == "learned") {
            EXPECT_THROW((void)MakeAnyPolicy(name, pool, ctx, lctx), std::invalid_argument);
            lctx.model = &model;
        }
        const auto policy = MakeAnyPolicy(name, pool, ctx, lctx);
        ASSERT_NE(policy, nullptr) << name;
        EXPECT_NE(dynamic_cast<SampledPolicy*>(policy.get()), nullptr) << name;
        const SimResult result = Simulate(*policy, trace, items);
        EXPECT_GT(result.evictions, 0u) << name;
        EXPECT_EQ(result.misses, result.evictions + pool) << name;
    }

    // The names mean what they say: with a sample of the whole pool they are the exact policies.
    lctx.config.k = std::numeric_limits<size_t>::max();
    const auto misses_of = [&](const std::string& name) { return Simulate(*MakeAnyPolicy(name, pool, ctx, lctx), trace, items).item_misses; };
    EXPECT_EQ(misses_of("sampled_lru"), Simulate(*MakePolicy("lru", pool, ctx), trace, items).item_misses);
    EXPECT_EQ(misses_of("sampled_belady"), Simulate(*MakePolicy("belady", pool, ctx), trace, items).item_misses);
    EXPECT_EQ(misses_of("learned"), misses_of("sampled_lru"));  // this model scores by recency
    lctx.config.k = 32;

    // Every other name goes to the simulator's own factory.
    for (const std::string& name : PolicyNames()) {
        if (name == "v2") continue;  // needs weights
        EXPECT_EQ(Simulate(*MakeAnyPolicy(name, pool, ctx, lctx), trace, items).item_misses,
                  Simulate(*MakePolicy(name, pool, ctx), trace, items).item_misses)
            << name;
    }
    EXPECT_THROW((void)MakeAnyPolicy("nope", pool, ctx, lctx), std::invalid_argument);
    EXPECT_THROW((void)MakeAnyPolicy("sampled_lru", 0, ctx, lctx), std::invalid_argument);

    // The label needs Belady's hits for this very pool, and both look-ahead names need next_use.
    EXPECT_THROW((void)MakeAnyPolicy("sampled_label", pool + 1, ctx, lctx), std::invalid_argument);
    LearnedContext no_hits;
    EXPECT_THROW((void)MakeAnyPolicy("sampled_label", pool, ctx, no_hits), std::invalid_argument);
    SimContext no_next;
    no_next.trace = &trace;
    EXPECT_THROW((void)MakeAnyPolicy("sampled_label", pool, no_next, lctx), std::invalid_argument);
    EXPECT_THROW((void)MakeAnyPolicy("sampled_belady", pool, no_next, lctx), std::invalid_argument);
    EXPECT_NO_THROW((void)MakeAnyPolicy("sampled_lru", pool, no_next, lctx));
    SimContext no_trace;
    EXPECT_THROW((void)MakeAnyPolicy("sampled_lru", pool, no_trace, lctx), std::invalid_argument);
}

TEST(EvictionLearnedTest, MlpTinyNetworkScore) {
    const LearnedModel model = LoadText("tiny_mlp", kTinyMlp);
    EXPECT_EQ(model.type, LearnedModel::Type::MLP);
    EXPECT_EQ(model.features, FeatureSet::BASIC);
    EXPECT_EQ(model.inputs, 3u);
    EXPECT_EQ(model.pool, 9u);
    EXPECT_EQ(model.sf, "0.01");
    EXPECT_EQ(model.mean, (std::vector<float>{0, 0, 0}));
    EXPECT_EQ(model.std, (std::vector<float>{1, 1, 1}));
    ASSERT_EQ(model.layers.size(), 2u);
    EXPECT_EQ(model.layers[0].out, 2u);
    EXPECT_EQ(model.layers[0].in, 3u);
    EXPECT_EQ(model.layers[0].w, (std::vector<float>{1, 0, 0, 0, 1, -1}));
    EXPECT_EQ(model.layers[0].b, (std::vector<float>{0, 0.5f}));
    EXPECT_EQ(model.layers[1].w, (std::vector<float>{2, -1}));
    EXPECT_EQ(model.layers[1].b, (std::vector<float>{0.25f}));
    ASSERT_EQ(model.refs.size(), 2u);
    EXPECT_EQ(model.refs[1], (std::vector<float>{0, 3, 1, -2.25f}));
    EXPECT_EQ(ModelMaxRefError(model), 0.0);

    // Candidates (1, 2, 3) and (0, 3, 1), one feature after another.
    ModelScorer scorer(model);
    const std::vector<float> x = {1, 0, 2, 3, 3, 1};
    std::array<float, 2> out{};
    scorer.Score(x.data(), 2, out.data());
    EXPECT_EQ(out[0], 2.25f);
    EXPECT_EQ(out[1], -2.25f);
    EXPECT_EQ(scorer.MacsPerCandidate(), 8u);
    EXPECT_EQ(scorer.NodesVisited(), 0u);

    // Standardizing happens first: with mean 1 and std 2, (3, 5, 7) is (1, 2, 3) again.
    LearnedModel shifted = model;
    shifted.mean = {1, 1, 1};
    shifted.std = {2, 2, 2};
    ModelScorer shifted_scorer(shifted);
    const std::vector<float> y = {3, 5, 7};
    shifted_scorer.Score(y.data(), 1, out.data());
    EXPECT_EQ(out[0], 2.25f);
}

TEST(EvictionLearnedTest, GbdtTinyTreesScore) {
    const LearnedModel model = LoadText("tiny_gbdt", kTinyGbdt);
    EXPECT_EQ(model.type, LearnedModel::Type::GBDT);
    ASSERT_EQ(model.trees.size(), 2u);
    ASSERT_EQ(model.trees[1].size(), 3u);
    EXPECT_EQ(model.trees[1][0].feature, 2);
    EXPECT_EQ(model.trees[1][0].threshold, 1.0);
    EXPECT_EQ(model.trees[1][2].value, 0.75f);
    EXPECT_EQ(model.trees[1][2].left, -1);

    // (0.5, 0, 1.0): both tests are "<=", so both go left. (0.6, 0, 1.1): both right.
    ModelScorer scorer(model);
    const std::vector<float> x = {0.5f, 0.6f, 0, 0, 1.0f, 1.1f};
    std::array<float, 2> out{};
    scorer.Score(x.data(), 2, out.data());
    EXPECT_EQ(out[0], -0.75f);
    EXPECT_EQ(out[1], 1.75f);
    EXPECT_EQ(scorer.MacsPerCandidate(), 0u);
    EXPECT_EQ(scorer.NodesVisited(), 4u);  // one split per tree per candidate
    scorer.Score(x.data(), 1, out.data());
    EXPECT_EQ(scorer.NodesVisited(), 6u);

    // The threshold is a double: 0.1 as a float is a little above 0.1, so it goes right.
    LearnedModel fine = model;
    fine.trees.resize(1);
    fine.trees[0][0].threshold = 0.1;
    ModelScorer fine_scorer(fine);
    const std::vector<float> z = {0.1f, 0, 0};
    fine_scorer.Score(z.data(), 1, out.data());
    EXPECT_EQ(out[0], 1.0f);
}

TEST(EvictionLearnedTest, ModelFilesAreChecked) {
    const std::string mlp = kTinyMlp, gbdt = kTinyGbdt;
    const std::string path = TempPath("checked_model");

    // A ref score a little off still loads; the limit is 1e-4 of the score.
    WriteFile(path, Swapped(mlp, "3 2.25\n", "3 2.2501\n"));
    EXPECT_NO_THROW((void)LoadLearnedModel(path));

    const std::vector<std::pair<std::string, std::string>> bad = {
        {"no ref", mlp.substr(0, mlp.find("ref 2"))},
        {"ref with no rows", mlp.substr(0, mlp.find("ref 2")) + "ref 0\n"},
        {"inf weight", Swapped(mlp, "layer 2 3 1 0", "layer 2 3 inf 0")},
        {"nan bias", Swapped(mlp, "-1 0.25", "-1 nan")},
        {"weight too big for a float", Swapped(mlp, "layer 2 3 1 0", "layer 2 3 1e39 0")},
        {"inputs do not match features", Swapped(mlp, "inputs 3", "inputs 4")},
        {"ref score off", Swapped(mlp, "3 2.25\n", "3 2.251\n")},
        {"truncated layer", Swapped(mlp, "layer 1 2 2 -1 0.25", "layer 1 2 2 -1")},
        {"file ends in a ref row", mlp.substr(0, mlp.size() - 6)},
        {"junk after the refs", mlp + "layer 1 1 1 0\n"},
        {"unknown model", Swapped(mlp, "model mlp", "model cnn")},
        {"unknown features", Swapped(mlp, "features basic", "features huge")},
        {"pool 0", Swapped(mlp, "pool 9", "pool 0")},
        {"std of 0", Swapped(mlp, "std 3 1 1 1", "std 3 1 0 1")},
        {"mean too short", Swapped(mlp, "mean 3 0 0 0", "mean 2 0 0")},
        {"lines out of order", Swapped(Swapped(mlp, "mean 3 0 0 0", "std 3 1 1 1 "), "\nstd 3 1 1 1\n", "\nmean 3 0 0 0\n")},
        {"no layers", mlp.substr(0, mlp.find("layer 2")) + "ref 1\n1 2 3 0\n"},
        {"layers do not chain", Swapped(mlp, "layer 1 2 2 -1 0.25", "layer 1 3 2 -1 0 0.25")},
        {"last layer is not 1 wide", Swapped(mlp, "layer 1 2 2 -1 0.25", "layer 2 2 2 -1 2 -1 0.25 0.25")},
        {"trees in an mlp", Swapped(mlp, "layer 1 2 2 -1 0.25", "trees 0")},
        {"gbdt ref score off", Swapped(gbdt, "1.1 1.75", "1.1 1.5")},
        {"no trees", Swapped(gbdt.substr(0, gbdt.find("tree 3")) + "ref 1\n0 0 0 0\n", "trees 2", "trees 0")},
        {"fewer trees than promised", Swapped(gbdt, "trees 2", "trees 3")},
        {"child past the end", Swapped(gbdt, "0 0.5 1 2 0", "0 0.5 1 3 0")},
        {"only one child", Swapped(gbdt, "0 0.5 1 2 0", "0 0.5 1 -1 0")},
        {"a loop", Swapped(gbdt, "0 0.5 1 2 0", "0 0.5 0 2 0")},
        {"a node used twice", Swapped(gbdt, "0 0.5 1 2 0", "0 0.5 1 1 0")},
        {"feature past the inputs", Swapped(gbdt, "0 0.5 1 2 0", "3 0.5 1 2 0")},
        {"inf threshold", Swapped(gbdt, "0 0.5 1 2 0", "0 inf 1 2 0")},
        {"child is not a whole number", Swapped(gbdt, "0 0.5 1 2 0", "0 0.5 1.5 2 0")},
        {"empty tree", Swapped(gbdt, "tree 3\n2 1 1 2 0\n-1 0 -1 -1 0.25\n-1 0 -1 -1 0.75\n", "tree 0\n")},
        {"empty file", ""},
    };
    for (const auto& [what, text] : bad) {
        WriteFile(path, text);
        const std::string error = ErrorOf([&] { (void)LoadLearnedModel(path); });
        EXPECT_NE(error.find(path), std::string::npos) << what << ": '" << error << "'";
        EXPECT_THROW((void)LoadLearnedModel(path), std::runtime_error) << what;
    }

    std::remove(path.c_str());
    EXPECT_THROW((void)LoadLearnedModel(path), std::runtime_error);  // no such file

    // A model put together in memory is checked too, when a scorer is built for it.
    std::mt19937 rng(3);
    LearnedModel broken = RandomMlp(rng, FeatureSet::BASIC, {4, 1});
    EXPECT_NO_THROW(ModelScorer{broken});
    broken.layers[1].in = 5;
    EXPECT_THROW(ModelScorer{broken}, std::invalid_argument);
    LearnedModel looped = RandomGbdt(rng, FeatureSet::BASIC, 1);
    EXPECT_NO_THROW(ModelScorer{looped});
    looped.trees[0][6].left = 0;
    EXPECT_THROW(ModelScorer{looped}, std::invalid_argument);
}

TEST(EvictionLearnedTest, ScoringIsBatchIndependent) {
    std::mt19937 rng(41);
    const size_t inputs = kMaxFeatures;
    const std::vector<LearnedModel> models = {RandomMlp(rng, FeatureSet::RICHER, {24, 12, 1}), RandomGbdt(rng, FeatureSet::RICHER, 3)};

    for (const LearnedModel& model : models) {
        const size_t n = 32;
        std::vector<float> x(inputs * n);
        for (float& v : x) v = static_cast<float>(rng() % 5000) / 1000.0f;

        ModelScorer scorer(model);
        std::vector<float> together(n);
        scorer.Score(x.data(), n, together.data());

        // Each one alone, then the first 13 (a batch that is not a round size).
        ModelScorer other(model);
        std::set<float> distinct;
        for (size_t c = 0; c < n; ++c) {
            std::vector<float> one(inputs);
            for (size_t f = 0; f < inputs; ++f) one[f] = x[f * n + c];
            float alone = 0;
            other.Score(one.data(), 1, &alone);
            ASSERT_EQ(alone, together[c]) << "candidate " << c;
            ASSERT_TRUE(std::isfinite(alone));
            distinct.insert(alone);
        }
        EXPECT_GT(distinct.size(), 4u);  // the scores do depend on the inputs

        const size_t m = 13;
        std::vector<float> part(inputs * m), part_out(m);
        for (size_t f = 0; f < inputs; ++f) {
            for (size_t c = 0; c < m; ++c) part[f * m + c] = x[f * n + c];
        }
        scorer.Score(part.data(), m, part_out.data());
        for (size_t c = 0; c < m; ++c) ASSERT_EQ(part_out[c], together[c]) << "candidate " << c;

        // A bigger batch after a smaller one works too.
        scorer.Score(x.data(), n, together.data());
        for (size_t c = 0; c < m; ++c) ASSERT_EQ(part_out[c], together[c]) << "candidate " << c;
    }
}

TEST(EvictionLearnedTest, MacsPerCandidate) {
    std::mt19937 rng(42);
    const LearnedModel mlp = RandomMlp(rng, FeatureSet::RICHER, {24, 12, 1});
    EXPECT_EQ(ModelScorer(mlp).MacsPerCandidate(), 996u);  // 29x24 + 24x12 + 12x1
    const LearnedModel gbdt = RandomGbdt(rng, FeatureSet::RICHER, 3);
    EXPECT_EQ(ModelScorer(gbdt).MacsPerCandidate(), 0u);
}

TEST(EvictionLearnedTest, LearnedRecencyModelEqualsSampledLru) {
    const LearnedModel model = LoadText("recency_model", kRecencyMlp);
    SampledInputs inputs;
    inputs.model = &model;

    std::mt19937 rng(43);
    uint64_t evictions = 0;
    for (int round = 0; round < 20; ++round) {
        const Trace trace = TraceOfRuns(RandomRuns(rng, 1500 + rng() % 1000, 20 + rng() % 40, 3));
        ASSERT_LT(trace.end_fetch, 10000u);  // so every log1p(fetches) is a different float
        const auto items = ItemsOver(trace, 9);

        SampledPolicy learned(9, trace, SampledScore::MODEL, ConfigWithK(4), inputs);
        SampledPolicy lru(9, trace, SampledScore::LRU, ConfigWithK(4), {});
        const SimResult got = Simulate(learned, trace, items);
        const SimResult want = Simulate(lru, trace, items);
        ASSERT_EQ(got.item_misses, want.item_misses) << "round " << round;
        ASSERT_EQ(got.misses, want.misses) << "round " << round;
        ASSERT_EQ(learned.Decisions(), want.evictions);
        evictions += want.evictions;
    }
    EXPECT_GT(evictions, 10000u);
}

TEST(EvictionLearnedTest, RicherModelNeedsOwners) {
    std::mt19937 rng(44);
    const Trace trace = TraceOfRuns(RandomRuns(rng, 600, 30, 3));
    const auto items = ItemsOver(trace, 4);
    OwnerMap owners;
    for (uint32_t p = 0; p < trace.page_limit; ++p) {
        owners.kind.push_back(p % 2 == 0 ? PageKind::TABLE : PageKind::LEAF);
        owners.object.push_back(static_cast<int8_t>(p % 2 == 0 ? p % 7 : 7 + p % 8));
    }

    const LearnedModel richer = RandomMlp(rng, FeatureSet::RICHER, {8, 1});
    const LearnedModel no_owner = RandomGbdt(rng, FeatureSet::RICHER_NO_OWNER, 2);
    const auto make = [&](const LearnedModel& model, const OwnerMap* map) {
        SampledInputs inputs;
        inputs.model = &model;
        inputs.owners = map;
        return std::make_unique<SampledPolicy>(8, trace, SampledScore::MODEL, SampledConfig{}, inputs);
    };
    EXPECT_THROW(make(richer, nullptr), std::invalid_argument);

    // With owners it runs, and a model without owner inputs never needed them.
    for (const auto& policy : {make(richer, &owners), make(no_owner, nullptr), make(no_owner, &owners)}) {
        const SimResult result = Simulate(*policy, trace, items);
        EXPECT_GT(result.evictions, 100u);
        EXPECT_EQ(result.misses, result.evictions + 8);
    }

    // The factory refuses it the same way.
    SimContext ctx;
    ctx.trace = &trace;
    LearnedContext lctx;
    lctx.model = &richer;
    EXPECT_THROW((void)MakeAnyPolicy("learned", 8, ctx, lctx), std::invalid_argument);
    lctx.owners = &owners;
    EXPECT_NO_THROW((void)MakeAnyPolicy("learned", 8, ctx, lctx));
}

TEST(EvictionLearnedTest, LearnedPolicyRefusesScoresThatAreNotNumbers) {
    // Weights big enough that the score overflows to infinity.
    LearnedModel model = LoadText("overflow_model", kRecencyMlp);
    model.layers[0].w = {0, 3e38f, 0};
    model.layers[1].w = {3e38f};
    SampledInputs inputs;
    inputs.model = &model;

    const Trace trace = TraceOfRuns({{0, 1}, {1, 1}, {2, 1}});
    SampledPolicy policy(2, trace, SampledScore::MODEL, SampledConfig{}, inputs);
    EXPECT_TRUE(policy.Access(0, trace.runs[0]));
    EXPECT_TRUE(policy.Access(1, trace.runs[1]));
    EXPECT_THROW(policy.Access(2, trace.runs[2]), std::runtime_error);
}


// Owners that vary by page, so every feature column gets exercised.
OwnerMap RandomOwners(uint32_t pages, std::mt19937& rng) {
    OwnerMap owners;
    owners.kind.resize(pages);
    owners.object.resize(pages);
    for (uint32_t p = 0; p < pages; ++p) {
        const int object = static_cast<int>(rng() % kNumObjects);
        owners.object[p] = static_cast<int8_t>(object);
        owners.kind[p] = object < kNumTables ? PageKind::TABLE : (rng() % 2 == 0 ? PageKind::LEAF : PageKind::INNER);
    }
    return owners;
}

struct SampleCase {
    Trace trace;
    OwnerMap owners;
    std::vector<uint32_t> next_use;
    std::vector<uint8_t> hits;
};

SampleCase MakeSampleCase(uint32_t seed, size_t pool) {
    std::mt19937 rng(seed);
    SampleCase c;
    c.trace = TraceOfRuns(RandomRuns(rng, 2600, 40, 3));
    c.owners = RandomOwners(c.trace.page_limit, rng);
    c.next_use = NextUse(c.trace);
    c.hits = BeladyHits(c.trace, pool, c.next_use);
    return c;
}

TEST(EvictionLearnedTest, CollectSamplesKeepsExactlyNDecisions) {
    const SampleCase c = MakeSampleCase(31, 8);
    for (const SampledScore replay : {SampledScore::LRU, SampledScore::BELADY}) {
        const SampleSet some = CollectSamples(c.trace, 8, replay, c.owners, c.next_use, c.hits, 4, 100, 1);
        EXPECT_EQ(some.decisions, 100u);
        EXPECT_EQ(some.rows.size(), size_t{100} * 4 * kSampleRowWidth);
        EXPECT_EQ(some.run_of_row.size(), 400u);
        EXPECT_GT(some.evictions, 1000u);

        const SampleSet all = CollectSamples(c.trace, 8, replay, c.owners, c.next_use, c.hits, 4, 1000000, 1);
        EXPECT_EQ(all.decisions, all.evictions);
        EXPECT_EQ(all.rows.size(), all.evictions * 4 * kSampleRowWidth);
    }
}

TEST(EvictionLearnedTest, FewerResidentThanKGivesFewerRows) {
    const SampleCase c = MakeSampleCase(32, 3);
    const SampleSet set = CollectSamples(c.trace, 3, SampledScore::LRU, c.owners, c.next_use, c.hits, 10, 50, 1);
    EXPECT_EQ(set.decisions, 50u);
    EXPECT_EQ(set.rows.size(), size_t{50} * 3 * kSampleRowWidth);
}

TEST(EvictionLearnedTest, CollectedRowsMatchFeaturesAndLabels) {
    const SampleCase c = MakeSampleCase(33, 8);
    for (const SampledScore replay : {SampledScore::LRU, SampledScore::BELADY}) {
        const SampleSet set = CollectSamples(c.trace, 8, replay, c.owners, c.next_use, c.hits, 5, 60, 7);
        ASSERT_EQ(set.rows.size() / kSampleRowWidth, set.run_of_row.size());

        // Walk the runs once with a fresh history; check each row when its run comes up.
        std::vector<size_t> by_run(set.run_of_row.size());
        std::iota(by_run.begin(), by_run.end(), size_t{0});
        std::stable_sort(by_run.begin(), by_run.end(), [&](size_t a, size_t b) { return set.run_of_row[a] < set.run_of_row[b]; });

        PageHistory history(c.trace.page_limit, &c.owners);
        size_t next_row = 0, run = 0;
        for (; run < c.trace.runs.size() && next_row < by_run.size(); ++run) {
            while (next_row < by_run.size() && set.run_of_row[by_run[next_row]] == run) {
                const size_t row = by_run[next_row++];
                const uint32_t page = set.page_of_row[row];
                std::array<float, kMaxFeatures> want;
                history.Features(page, run, c.trace.runs[run].first_fetch, kMaxFeatures, want.data());
                const float* got = set.rows.data() + row * kSampleRowWidth;
                for (size_t f = 0; f < kMaxFeatures; ++f) ASSERT_EQ(got[1 + f], want[f]) << "row " << row << " feature " << f;
                ASSERT_EQ(got[0], static_cast<float>(RelaxedBeladyLabel(history.LastRun(page), c.next_use, c.hits))) << "row " << row;
            }
            history.Update(run, c.trace.runs[run]);
        }
        EXPECT_EQ(next_row, by_run.size());
    }
}

TEST(EvictionLearnedTest, CollectorReplayIsExactlyLruAndBelady) {
    const SampleCase c = MakeSampleCase(34, 8);
    uint64_t lru_misses = 0, belady_misses = 0;
    LruPolicy lru(8, c.trace.page_limit);
    BeladyPolicy belady(8, c.trace.page_limit, c.next_use);
    for (size_t i = 0; i < c.trace.runs.size(); ++i) {
        lru_misses += lru.Access(i, c.trace.runs[i]) ? 1 : 0;
        belady_misses += belady.Access(i, c.trace.runs[i]) ? 1 : 0;
    }
    const SampleSet a = CollectSamples(c.trace, 8, SampledScore::LRU, c.owners, c.next_use, c.hits, 4, 30, 1);
    const SampleSet b = CollectSamples(c.trace, 8, SampledScore::BELADY, c.owners, c.next_use, c.hits, 4, 30, 1);
    EXPECT_EQ(a.replay_misses, lru_misses);
    EXPECT_EQ(a.evictions, lru.Evictions());
    EXPECT_EQ(b.replay_misses, belady_misses);
    EXPECT_EQ(b.evictions, belady.Evictions());
}

TEST(EvictionLearnedTest, CollectSamplesIsDeterministic) {
    const SampleCase c = MakeSampleCase(35, 8);
    const SampleSet a = CollectSamples(c.trace, 8, SampledScore::LRU, c.owners, c.next_use, c.hits, 4, 40, 1);
    const SampleSet b = CollectSamples(c.trace, 8, SampledScore::LRU, c.owners, c.next_use, c.hits, 4, 40, 1);
    const SampleSet other = CollectSamples(c.trace, 8, SampledScore::LRU, c.owners, c.next_use, c.hits, 4, 40, 2);
    EXPECT_EQ(a.rows, b.rows);
    EXPECT_NE(a.rows, other.rows);
}

TEST(EvictionLearnedTest, WriteSamplesWritesBinAndJson) {
    const SampleCase c = MakeSampleCase(36, 8);
    const SampleSet set = CollectSamples(c.trace, 8, SampledScore::LRU, c.owners, c.next_use, c.hits, 4, 20, 1);
    SampleMeta meta;
    meta.sf = "0.01";
    meta.trace = "mytrace";
    meta.replay = "lru";
    meta.pool = 8;
    meta.k = 4;
    meta.seed = 1;

    const std::string prefix = TempPath("samples");
    WriteSamples(set, meta, prefix);
    const size_t rows = set.rows.size() / kSampleRowWidth;
    EXPECT_EQ(std::filesystem::file_size(prefix + ".bin"), set.rows.size() * 4);

    std::ifstream bin(prefix + ".bin", std::ios::binary);
    std::vector<float> back(set.rows.size());
    bin.read(reinterpret_cast<char*>(back.data()), static_cast<std::streamsize>(back.size() * sizeof(float)));
    EXPECT_EQ(back, set.rows);

    std::ifstream in(prefix + ".json");
    const std::string json((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_NE(json.find("\"rows\": " + std::to_string(rows)), std::string::npos);
    EXPECT_NE(json.find("\"trace\": \"mytrace\""), std::string::npos);
    EXPECT_NE(json.find("\"columns\": [\"label\", \"b_since_fetch\""), std::string::npos);
    EXPECT_NE(json.find("\"replay_misses\": " + std::to_string(set.replay_misses)), std::string::npos);
    std::remove((prefix + ".bin").c_str());
    std::remove((prefix + ".json").c_str());
}

}  // namespace
}  // namespace shilmandb::evsim
