#include "eviction_learned.hpp"
#include "eviction_sim.hpp"

#include <charconv>
#include <exception>
#include <iostream>
#include <string>
#include <string_view>

// Replays one recorded trace as LRU or Belady and writes training rows for the learned policy:
// at sampled evictions, a few candidates from the pool with their features and labels.

using namespace shilmandb::evsim;

namespace {

struct Options {
    std::string trace, items, name, sf, replay, page_owners, out;
    size_t pool{0}, decisions{16384}, sample_k{32};
    uint64_t seed{1};
};

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
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + flag);
        const std::string value = argv[++i];
        if (flag == "--trace") o.trace = value;
        else if (flag == "--items") o.items = value;
        else if (flag == "--name") o.name = value;
        else if (flag == "--sf") o.sf = value;
        else if (flag == "--replay") o.replay = value;
        else if (flag == "--page-owners") o.page_owners = value;
        else if (flag == "--out") o.out = value;
        else if (flag == "--pool") o.pool = ParseNumber<size_t>(flag, value);
        else if (flag == "--decisions") o.decisions = ParseNumber<size_t>(flag, value);
        else if (flag == "--sample-k") o.sample_k = ParseNumber<size_t>(flag, value);
        else if (flag == "--seed") o.seed = ParseNumber<uint64_t>(flag, value);
        else throw std::runtime_error("unknown flag " + flag);
    }

    for (const auto* required : {&o.trace, &o.items, &o.name, &o.sf, &o.replay, &o.page_owners, &o.out}) {
        if (required->empty()) throw std::runtime_error("--trace, --items, --name, --sf, --pool, --replay, --page-owners and --out are required");
    }

    if (o.pool == 0) throw std::runtime_error("--pool must be at least 1");
    if (o.sample_k == 0) throw std::runtime_error("--sample-k must be at least 1");
    if (o.decisions == 0) throw std::runtime_error("--decisions must be at least 1");
    if (o.replay != "lru" && o.replay != "belady") throw std::runtime_error("--replay must be lru or belady");
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = ParseArgs(argc, argv);
        const OwnerMap owners = LoadPageOwners(o.page_owners);

        auto items = LoadItems(o.items);
        if (items.empty()) throw std::runtime_error("no items in " + o.items);

        const Trace trace = LoadTrace(o.trace, EndFetch(items));
        MapItems(trace, items);
        CheckTraceCovered(owners, trace);

        const auto next_use = NextUse(trace);
        const auto belady_hits = BeladyHits(trace, o.pool, next_use);
        const SampledScore replay = o.replay == "lru" ? SampledScore::LRU : SampledScore::BELADY;
        const SampleSet set = CollectSamples(trace, o.pool, replay, owners, next_use, belady_hits, o.sample_k, o.decisions, o.seed);

        SampleMeta meta;
        meta.sf = o.sf;
        meta.trace = o.name;
        meta.replay = o.replay;
        meta.pool = o.pool;
        meta.k = o.sample_k;
        meta.seed = o.seed;

        WriteSamples(set, meta, o.out);
        std::cout << o.name << " " << o.replay << " pool " << o.pool << ": " << set.evictions << " evictions, " << set.decisions << " decisions, " << set.rows.size() / kSampleRowWidth << " rows\n";
    } catch (const std::exception& e) {
        std::cerr << "eviction_samples: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
