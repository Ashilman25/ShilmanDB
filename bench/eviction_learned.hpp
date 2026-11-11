#pragma once

// What a learned eviction policy sees and learns from, for the trace simulator:
// the page owner map, each page's access history, the model's inputs, and the labels.
// Also the policy itself: it evicts the lowest-scoring of a few pages drawn at random.
// Header-only, and it does not use any engine code.

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "eviction_sim.hpp"

namespace shilmandb::evsim {

enum class PageKind : uint8_t { TABLE, INNER, LEAF, OTHER };

constexpr int kNumObjects = 15;  // the 7 tables, then the 8 indexes
constexpr int kNumTables = 7;

// What each page belongs to, indexed by page id.
struct OwnerMap {
    std::vector<PageKind> kind;
    std::vector<int8_t> object;  // 0..14, or -1 for a page nothing owns
};

// Reads a "page_id,kind,object,owner" file. A page id the file leaves out is OTHER.
[[nodiscard]] inline OwnerMap LoadPageOwners(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);

    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("empty " + path);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "page_id,kind,object,owner") {
        throw std::runtime_error(path + ": header must be page_id,kind,object,owner");
    }

    OwnerMap owners;
    std::vector<bool> seen;
    size_t line_no = 1;
    while (std::getline(in, line)) {
        ++line_no;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        const auto bad = [&](const std::string& why) {
            return std::runtime_error(path + " line " + std::to_string(line_no) + ": " + why + " in '" + line + "'");
        };

        // Exactly four fields; the owner name is only for people and may be empty.
        const size_t c1 = line.find(',');
        const size_t c2 = c1 == std::string::npos ? c1 : line.find(',', c1 + 1);
        const size_t c3 = c2 == std::string::npos ? c2 : line.find(',', c2 + 1);
        if (c3 == std::string::npos || line.find(',', c3 + 1) != std::string::npos) throw bad("expected 4 fields");

        const std::string id_text = line.substr(0, c1);
        const std::string kind_text = line.substr(c1 + 1, c2 - c1 - 1);
        const std::string object_text = line.substr(c2 + 1, c3 - c2 - 1);

        if (id_text.empty() || id_text.size() > 9 || id_text.find_first_not_of("0123456789") != std::string::npos) {
            throw bad("bad page id");
        }
        const size_t page = std::stoul(id_text);

        PageKind kind;
        if (kind_text == "table") kind = PageKind::TABLE;
        else if (kind_text == "inner") kind = PageKind::INNER;
        else if (kind_text == "leaf") kind = PageKind::LEAF;
        else if (kind_text == "other") kind = PageKind::OTHER;
        else throw bad("bad kind");

        const bool negative = !object_text.empty() && object_text[0] == '-';
        const std::string digits = object_text.substr(negative ? 1 : 0);
        if (digits.empty() || digits.size() > 3 || digits.find_first_not_of("0123456789") != std::string::npos) {
            throw bad("bad object");
        }
        const int object = (negative ? -1 : 1) * std::stoi(digits);

        // The object must fit the kind: tables are 0-6, indexes 7-14, nothing is -1.
        const bool fits = kind == PageKind::OTHER   ? object == -1 : kind == PageKind::TABLE ? object >= 0 && object < kNumTables : object >= kNumTables && object < kNumObjects;
        if (!fits) throw bad("object does not fit the kind");

        if (page >= owners.kind.size()) {
            owners.kind.resize(page + 1, PageKind::OTHER);
            owners.object.resize(page + 1, -1);
            seen.resize(page + 1, false);
        }
        if (seen[page]) throw bad("page listed twice");
        seen[page] = true;
        owners.kind[page] = kind;
        owners.object[page] = static_cast<int8_t>(object);
    }
    if (owners.kind.empty()) throw std::runtime_error(path + " lists no pages");
    return owners;
}

// Throws, naming the page, at the first trace page the map has no owner for.
// That is what a map from a different database layout looks like.
inline void CheckTraceCovered(const OwnerMap& owners, const Trace& trace) {
    for (size_t r = 0; r < trace.runs.size(); ++r) {
        const uint32_t page = trace.runs[r].page;
        if (page < owners.kind.size() && owners.kind[page] != PageKind::OTHER) continue;
        throw std::runtime_error("page " + std::to_string(page) + " (run " + std::to_string(r) + ") is in the trace but has no owner in the page map" + (page < owners.kind.size() ? " (kind other)" : " (the map has " + std::to_string(owners.kind.size()) + " pages)"));
    }
}

// The model's inputs. Each set is a prefix of the next one.
enum class FeatureSet { BASIC, RICHER_NO_OWNER, RICHER };

constexpr size_t kBasicFeatures = 3;
constexpr size_t kHistoryFeatures = 11;  // basic + history
constexpr size_t kMaxFeatures = 29;      // + kind and object one-hots

[[nodiscard]] constexpr size_t FeatureCount(FeatureSet set) {
    return set == FeatureSet::BASIC ? kBasicFeatures : set == FeatureSet::RICHER_NO_OWNER ? kHistoryFeatures : kMaxFeatures;
}

[[nodiscard]] inline std::optional<FeatureSet> ParseFeatureSet(const std::string& name) {
    if (name == "basic") return FeatureSet::BASIC;
    if (name == "richer_no_owner") return FeatureSet::RICHER_NO_OWNER;
    if (name == "richer") return FeatureSet::RICHER;
    return std::nullopt;
}

[[nodiscard]] inline const std::array<const char*, kMaxFeatures>& FeatureNames() {
    static const std::array<const char*, kMaxFeatures> names = {
        "b_since_fetch", "b_fetches", "b_since_second",
        "h_since_run", "h_visits", "h_since_visit", "h_gap1", "h_gap2", "h_gap3", "h_last_run_len", "h_age",
        "o_table", "o_inner", "o_leaf",
        "o_obj0", "o_obj1", "o_obj2", "o_obj3", "o_obj4", "o_obj5", "o_obj6", "o_obj7",
        "o_obj8", "o_obj9", "o_obj10", "o_obj11", "o_obj12", "o_obj13", "o_obj14"};
    return names;
}

// Everything the features need to know about each page's past, whether or not the page is
// in the pool. Fixed-size state per page id, so nothing is hashed or allocated per run.
class PageHistory {
public:
    static constexpr uint32_t kVisitWindow = 64;  // runs; a longer break starts a new visit

    // owners may be null; then only the first 11 features can be asked for.
    PageHistory(uint32_t page_limit, const OwnerMap* owners) : owners_(owners), pages_(page_limit) {}

    // Call once per run, in order, after any decision made for that run.
    void Update(size_t run_index, const Run& run) {
        State& s = pages_[run.page];
        const auto r = static_cast<uint32_t>(run_index);
        const uint64_t t = run.first_fetch, len = run.length;

        // The second-to-last fetch first: a one-fetch run needs the old last fetch.
        if (len >= 2) {
            s.second_fetch = t + len - 2;
            s.has_second = true;
        } else if (s.last_run != kNever) {
            s.second_fetch = s.last_fetch;
            s.has_second = true;
        }
        s.last_fetch = t + len - 1;
        s.fetches += len;

        if (s.last_run == kNever) {
            s.first_run = r;
            s.visits = 1;
            s.visit_start = r;
        } else if (r - s.last_run > kVisitWindow) {
            s.gaps[2] = s.gaps[1];
            s.gaps[1] = s.gaps[0];
            s.gaps[0] = r - s.visit_start;
            ++s.visits;
            s.visit_start = r;
        }
        s.last_run = r;
        s.last_run_len = run.length;
    }

    // Writes the first n features (3, 11 or 29) of a page that has been seen, for a decision
    // at run r whose first fetch is t. Only the asked-for part is worked out.
    void Features(uint32_t page, size_t r, uint64_t t, size_t n, float* out) const {
        assert(n == kBasicFeatures || n == kHistoryFeatures || n == kMaxFeatures);
        const State& s = pages_[page];
        assert(s.last_run != kNever && s.last_run < r && s.last_fetch < t);
        const auto lg = [](uint64_t x) { return static_cast<float>(std::log1p(static_cast<double>(x))); };

        out[0] = lg(t - s.last_fetch - 1);
        out[1] = lg(s.fetches);
        out[2] = lg(s.has_second ? t - s.second_fetch - 1 : t);
        if (n <= kBasicFeatures) return;

        out[3] = lg(r - s.last_run - 1);
        out[4] = lg(s.visits);
        out[5] = lg(r - s.visit_start - 1);
        out[6] = lg(s.gaps[0]);
        out[7] = lg(s.gaps[1]);
        out[8] = lg(s.gaps[2]);
        out[9] = lg(s.last_run_len);
        out[10] = lg(r - s.first_run - 1);
        if (n <= kHistoryFeatures) return;

        assert(owners_ != nullptr);
        for (size_t i = kHistoryFeatures; i < kMaxFeatures; ++i) out[i] = 0.0f;
        if (page >= owners_->kind.size()) return;

        const PageKind kind = owners_->kind[page];
        if (kind != PageKind::OTHER) out[kHistoryFeatures + static_cast<size_t>(kind)] = 1.0f;
        
        const int object = owners_->object[page];
        if (object >= 0 && object < kNumObjects) out[kHistoryFeatures + 3 + static_cast<size_t>(object)] = 1.0f;
    }

    // The page's latest run so far, kNever if it has not been seen.
    [[nodiscard]] uint32_t LastRun(uint32_t page) const { return pages_[page].last_run; }

private:
    struct State {
        uint64_t last_fetch{0};
        uint64_t second_fetch{0};
        uint64_t fetches{0};
        uint32_t last_run{kNever};
        uint32_t first_run{0};
        uint32_t visits{0};
        uint32_t visit_start{0};       // run that started the latest visit
        std::array<uint32_t, 3> gaps{};  // runs between visit starts, newest first
        uint32_t last_run_len{0};
        bool has_second{false};
    };

    const OwnerMap* owners_;
    std::vector<State> pages_;
};

// belady_hits[i] = 1 when run i's first fetch hits under Belady at this pool.
[[nodiscard]] inline std::vector<uint8_t> BeladyHits(const Trace& trace, size_t pool, const std::vector<uint32_t>& next_use) {
    BeladyPolicy belady(pool, trace.page_limit, next_use);
    std::vector<uint8_t> hits(trace.runs.size());
    for (size_t i = 0; i < trace.runs.size(); ++i) hits[i] = belady.Access(i, trace.runs[i]) ? 0 : 1;
    return hits;
}

// The label of a page whose latest run is last_run: 1 if Belady keeps it until its next
// use (that use hits), 0 if Belady evicts it first or it is never used again.
[[nodiscard]] inline uint8_t RelaxedBeladyLabel(uint32_t last_run, const std::vector<uint32_t>& next_use, const std::vector<uint8_t>& belady_hits) {
    const uint32_t next = next_use[last_run];
    return next == kNever ? 0 : belady_hits[next];
}

// Whole numbers from a seed, drawn the way the workload generator draws them (tpch::Rng),
// so a seed means the same sample on any standard library.
class SampleRng {
public:
    explicit SampleRng(uint64_t seed) : engine_(seed) {}

    // A whole number from lo to hi, both included.
    [[nodiscard]] int64_t Uniform(int64_t lo, int64_t hi) {
        assert(lo <= hi);
        const uint64_t span = static_cast<uint64_t>(hi) - static_cast<uint64_t>(lo) + 1;
        if (span == 0) return static_cast<int64_t>(engine_());  // the whole 64-bit range

        // Throw away the few low draws that would make some results more likely than others.
        const uint64_t too_low = (0 - span) % span;
        uint64_t draw = engine_();
        while (draw < too_low) draw = engine_();
        return static_cast<int64_t>(static_cast<uint64_t>(lo) + draw % span);
    }

private:
    std::mt19937_64 engine_;
};

// A trained model as its text file holds it
// a small network or a set of boosted trees
struct LearnedModel {
    enum class Type { MLP, GBDT };
    Type type{Type::MLP};
    FeatureSet features{FeatureSet::BASIC};
    size_t inputs{0}, pool{0};
    std::string sf;

    // Network: inputs are standardized, then go through the layers (ReLU between them).
    std::vector<float> mean, std;
    struct Layer {
        size_t out{0}, in{0};
        std::vector<float> w, b;  // w is [out][in], row after row
    };
    std::vector<Layer> layers;

    // Trees: node 0 is the root; a node with left and right of -1 is a leaf.
    struct Node {
        int32_t feature{-1};
        double threshold{0};
        int32_t left{-1}, right{-1};
        float value{0};
    };
    std::vector<std::vector<Node>> trees;

    std::vector<std::vector<float>> refs;  // N inputs then the score training worked out
};

// Throws std::invalid_argument if the model's parts don't fit together.
inline void CheckModelShape(const LearnedModel& model) {
    const auto bad = [](const std::string& why) { throw std::invalid_argument(why); };

    if (model.inputs != FeatureCount(model.features)) {
        bad("inputs " + std::to_string(model.inputs) + " does not match its features (" + std::to_string(FeatureCount(model.features)) + ")");
    }
    for (const auto& row : model.refs) {
        if (row.size() != model.inputs + 1) bad("a ref row must have every input and one score");
    }

    if (model.type == LearnedModel::Type::MLP) {
        if (model.mean.size() != model.inputs || model.std.size() != model.inputs) bad("mean and std must have one value per input");
        if (std::any_of(model.std.begin(), model.std.end(), [](float v) { return !(v > 0); })) bad("std values must be above 0");
        if (model.layers.empty()) bad("the network has no layers");
        size_t in = model.inputs;
        for (const auto& layer : model.layers) {
            if (layer.out == 0 || layer.in != in) bad("a layer does not fit the one before it");
            if (layer.w.size() != layer.out * layer.in || layer.b.size() != layer.out) bad("a layer has the wrong number of weights");
            in = layer.out;
        }
        if (in != 1) bad("the last layer must have 1 output");
        return;
    }

    if (model.trees.empty()) bad("the model has no trees");
    std::vector<int32_t> todo;
    std::vector<uint8_t> seen;
    for (size_t t = 0; t < model.trees.size(); ++t) {
        const auto& nodes = model.trees[t];
        const std::string name = "tree " + std::to_string(t);
        if (nodes.empty()) bad(name + " has no nodes");

        // Walk down from the root: every node must be reached exactly once, so a walk always ends.
        const auto size = static_cast<int64_t>(nodes.size());
        seen.assign(nodes.size(), 0);
        todo.assign(1, 0);
        size_t reached = 0;
        while (!todo.empty()) {
            const int32_t at = todo.back();
            todo.pop_back();
            if (seen[at] != 0) bad(name + " reaches node " + std::to_string(at) + " twice");
            seen[at] = 1;
            ++reached;

            const auto& node = nodes[at];
            if (node.left == -1 && node.right == -1) continue;
            if (node.left < 0 || node.left >= size || node.right < 0 || node.right >= size) bad(name + " node " + std::to_string(at) + " has a bad child");
            if (node.feature < 0 || static_cast<size_t>(node.feature) >= model.inputs) bad(name + " node " + std::to_string(at) + " has a bad feature");
            todo.push_back(node.left);
            todo.push_back(node.right);
        }
        if (reached != nodes.size()) bad(name + " has nodes the root never reaches");
    }
}

// Scores candidates with a model, in plain loops. It keeps its own working memory, so every
// user (and every thread) needs its own; the model is only read and must outlive it.
class ModelScorer {
public:
    explicit ModelScorer(const LearnedModel& model) : model_(&model) {
        CheckModelShape(model);
        if (model.type == LearnedModel::Type::MLP) {
            width_ = model.inputs;
            for (const auto& layer : model.layers) {
                width_ = std::max(width_, layer.out);
                macs_ += layer.in * layer.out;
            }
            return;
        }
        // All the trees in one array, each child index moved to its place in it.
        for (const auto& tree : model.trees) {
            const auto base = static_cast<int32_t>(nodes_.size());
            roots_.push_back(base);
            for (const auto& node : tree) {
                const bool leaf = node.left < 0;
                nodes_.push_back({node.threshold, leaf ? 0 : node.feature, leaf ? -1 : base + node.left, leaf ? -1 : base + node.right, node.value});
            }
        }
    }

    // Makes room for n candidates now, so Score doesn't have to later.
    void Reserve(size_t n) {
        if (n <= capacity_) return;
        capacity_ = n;
        front_.resize(width_ * n);
        back_.resize(width_ * n);
    }

    // x is feature-major: x[f * n + c] is feature f of candidate c. Writes n scores.
    // A candidate's score never depends on what else is in the batch.
    void Score(const float* x, size_t n, float* out) {
        if (model_->type == LearnedModel::Type::MLP) ScoreMlp(x, n, out);
        else ScoreTrees(x, n, out);
    }

    // Multiply-adds per candidate: the sum of in x out over the layers. 0 for trees.
    [[nodiscard]] uint64_t MacsPerCandidate() const { return macs_; }

    // Tree splits walked through so far, over every Score call. 0 for a network.
    [[nodiscard]] uint64_t NodesVisited() const { return nodes_visited_; }

private:
    struct FlatNode {
        double threshold;
        int32_t feature, left, right;  // left of -1 marks a leaf
        float value;
    };

    void ScoreMlp(const float* x, size_t n, float* out) {
        Reserve(n);
        float* cur = front_.data();
        float* next = back_.data();

        const size_t inputs = model_->inputs;
        for (size_t f = 0; f < inputs; ++f) {
            const float mean = model_->mean[f], scale = model_->std[f];
            const float* src = x + f * n;
            float* dst = cur + f * n;
            for (size_t c = 0; c < n; ++c) dst[c] = (src[c] - mean) / scale;
        }

        // One unit at a time, with the candidates in the inner loop: each candidate's sum
        // is added up in the same order however many there are.
        const size_t layers = model_->layers.size();
        for (size_t l = 0; l < layers; ++l) {
            const LearnedModel::Layer& layer = model_->layers[l];
            const bool last = l + 1 == layers;
            float* dst = last ? out : next;
            for (size_t j = 0; j < layer.out; ++j) {
                float* acc = dst + j * n;
                const float bias = layer.b[j];
                for (size_t c = 0; c < n; ++c) acc[c] = bias;

                const float* w = layer.w.data() + j * layer.in;
                for (size_t i = 0; i < layer.in; ++i) {
                    const float wi = w[i];
                    const float* xi = cur + i * n;
                    for (size_t c = 0; c < n; ++c) acc[c] += wi * xi[c];
                }
                if (last) continue;
                // Written this way so a NaN stays a NaN and gets caught later.
                for (size_t c = 0; c < n; ++c) acc[c] = acc[c] < 0.0f ? 0.0f : acc[c];
            }
            std::swap(cur, next);
        }
    }

    void ScoreTrees(const float* x, size_t n, float* out) {
        const FlatNode* nodes = nodes_.data();
        uint64_t visited = 0;
        for (size_t c = 0; c < n; ++c) {
            float sum = 0.0f;
            for (const int32_t root : roots_) {
                const FlatNode* node = nodes + root;

                while (node->left >= 0) {
                    const bool go_left = static_cast<double>(x[static_cast<size_t>(node->feature) * n + c]) <= node->threshold;
                    node = nodes + (go_left ? node->left : node->right);
                    ++visited;
                }
                sum += node->value;
            }
            out[c] = sum;
        }
        nodes_visited_ += visited;
    }

    const LearnedModel* model_;
    uint64_t macs_{0};
    uint64_t nodes_visited_{0};

    // Network: two buffers of [unit][candidate] that the layers go back and forth between.
    size_t width_{0};
    size_t capacity_{0};
    std::vector<float> front_, back_;

    std::vector<FlatNode> nodes_;
    std::vector<int32_t> roots_;
};

// How far my scores for the model's ref rows are from the scores in the file, at worst:
// |mine - ref| / max(1, |ref|). Infinity if any is not a number.
[[nodiscard]] inline double ModelMaxRefError(const LearnedModel& model) {
    const size_t n = model.refs.size();
    if (n == 0) return 0.0;

    std::vector<float> x(model.inputs * n), out(n);
    for (size_t r = 0; r < n; ++r) {
        for (size_t f = 0; f < model.inputs; ++f) x[f * n + r] = model.refs[r][f];
    }
    ModelScorer scorer(model);
    scorer.Score(x.data(), n, out.data());

    double worst = 0;
    for (size_t r = 0; r < n; ++r) {
        const double ref = model.refs[r][model.inputs];
        const double gap = std::abs(out[r] - ref) / std::max(1.0, std::abs(ref));
        if (!std::isfinite(gap)) return std::numeric_limits<double>::infinity();
        worst = std::max(worst, gap);
    }
    return worst;
}

constexpr double kModelRefTolerance = 1e-4;

// Reads a model text file. Throws, naming the file, unless it fits
// and its reference scores match.
[[nodiscard]] inline LearnedModel LoadLearnedModel(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);

    const auto fail = [&](const std::string& why) { throw std::runtime_error(path + ": " + why); };
    const auto token = [&]() {
        std::string text;
        if (!(in >> text)) fail("file ends early");
        return text;
    };
    const auto expect = [&](const std::string& word) {
        const std::string text = token();
        if (text != word) fail("expected '" + word + "', got '" + text + "'");
    };
    const auto count = [&]() {
        const std::string text = token();
        if (text.size() > 9 || text.find_first_not_of("0123456789") != std::string::npos) fail("bad count '" + text + "'");
        return static_cast<size_t>(std::stoul(text));
    };
    // A whole number that may be negative (-1 marks a leaf).
    const auto index = [&]() {
        const std::string text = token();
        const std::string digits = text.substr(!text.empty() && text[0] == '-' ? 1 : 0);
        if (digits.empty() || digits.size() > 9 || digits.find_first_not_of("0123456789") != std::string::npos) fail("bad whole number '" + text + "'");
        return static_cast<int32_t>(std::stol(text));
    };
    const auto number = [&]() {
        const std::string text = token();
        char* end = nullptr;
        const float value = std::strtof(text.c_str(), &end);
        if (end == text.c_str() || *end != '\0' || !std::isfinite(value)) fail("bad number '" + text + "'");
        return value;
    };
    const auto wide_number = [&]() {
        const std::string text = token();
        char* end = nullptr;
        const double value = std::strtod(text.c_str(), &end);
        if (end == text.c_str() || *end != '\0' || !std::isfinite(value)) fail("bad number '" + text + "'");
        return value;
    };
    // Reads n numbers one by one, so a wrong count in the file can't make me ask for huge memory.
    const auto numbers = [&](size_t n) {
        std::vector<float> values;
        for (size_t i = 0; i < n; ++i) values.push_back(number());
        return values;
    };

    LearnedModel model;
    expect("model");
    const std::string type = token();
    if (type == "mlp") model.type = LearnedModel::Type::MLP;
    else if (type == "gbdt") model.type = LearnedModel::Type::GBDT;
    else fail("unknown model '" + type + "'");

    expect("features");
    const std::string set = token();
    const auto features = ParseFeatureSet(set);
    if (!features) fail("unknown features '" + set + "'");
    model.features = *features;

    expect("inputs");
    model.inputs = count();
    if (model.inputs != FeatureCount(model.features)) {
        fail("inputs " + std::to_string(model.inputs) + " does not match features " + set + " (" + std::to_string(FeatureCount(model.features)) + ")");
    }
    expect("pool");
    model.pool = count();
    if (model.pool == 0) fail("pool must be at least 1");
    expect("sf");
    model.sf = token();

    // The next word after the model's own lines; it must be "ref".
    std::string after;
    if (model.type == LearnedModel::Type::MLP) {
        expect("mean");
        if (count() != model.inputs) fail("mean must have one value per input");
        model.mean = numbers(model.inputs);
        expect("std");
        if (count() != model.inputs) fail("std must have one value per input");
        model.std = numbers(model.inputs);

        while (in >> after && after == "layer") {
            LearnedModel::Layer layer;
            layer.out = count();
            layer.in = count();
            layer.w = numbers(layer.out * layer.in);
            layer.b = numbers(layer.out);
            model.layers.push_back(std::move(layer));
            after.clear();
        }
    } else {
        expect("trees");
        const size_t trees = count();
        for (size_t t = 0; t < trees; ++t) {
            expect("tree");
            const size_t nodes = count();
            std::vector<LearnedModel::Node> tree;
            for (size_t i = 0; i < nodes; ++i) {
                LearnedModel::Node node;
                node.feature = index();
                node.threshold = wide_number();
                node.left = index();
                node.right = index();
                node.value = number();
                tree.push_back(node);
            }
            model.trees.push_back(std::move(tree));
        }
        in >> after;
    }

    if (after.empty()) fail("no ref lines, so the scores can't be checked against training's");
    if (after != "ref") fail("expected 'ref', got '" + after + "'");
    const size_t refs = count();
    if (refs == 0) fail("no ref rows, so the scores can't be checked against training's");
    for (size_t r = 0; r < refs; ++r) model.refs.push_back(numbers(model.inputs + 1));
    if (in >> after) fail("unexpected '" + after + "' after the ref rows");

    try {
        CheckModelShape(model);
    } catch (const std::invalid_argument& e) {
        fail(e.what());
    }
    const double error = ModelMaxRefError(model);
    if (!(error < kModelRefTolerance)) {
        fail("my scores for the ref rows are off by " + std::to_string(error) + " (limit " + std::to_string(kModelRefTolerance) + ")");
    }
    return model;
}

// What a sampled policy ranks its candidates by. The lowest score is evicted.
enum class SampledScore {
    RANDOM,  // every score equal: the first page drawn goes
    LRU,     // the least recently used candidate
    LABEL,   // a page Belady would not keep first, then the least recently used
    BELADY,  // the candidate used furthest ahead
    MODEL    // the learned model's score
};

// Shared, read-only data a sampled policy looks at. It must outlive the policy.
struct SampledInputs {
    const OwnerMap* owners{nullptr};
    const std::vector<uint32_t>* next_use{nullptr};    // BELADY, LABEL
    const std::vector<uint8_t>* belady_hits{nullptr};  // LABEL
    const LearnedModel* model{nullptr};                // MODEL
};

struct SampledConfig {
    size_t k{32};  // candidates per eviction; anything at or above the pool means all of it
    uint64_t seed{1};
    bool time_decisions{false};
};

// On a miss with a full pool, draws k resident pages at random and evicts the one with the
// lowest score. Everything that changes lives in the object, so many can run at once on
// different threads over the same trace and inputs.
class SampledPolicy final : public Policy {
public:
    using DecisionHook = std::function<void(size_t run_index, const Run& run)>;

    SampledPolicy(size_t pool, const Trace& trace, SampledScore score, const SampledConfig& config, const SampledInputs& inputs) : pool_(pool), score_kind_(score), config_(config), inputs_(inputs), rng_(config.seed), history_(trace.page_limit, inputs.owners), in_pool_(trace.page_limit, 0) {
        if (pool == 0) throw std::invalid_argument("pool must be at least 1");
        if (config.k == 0) throw std::invalid_argument("a sampled policy needs at least 1 candidate");

        const bool looks_ahead = score == SampledScore::BELADY || score == SampledScore::LABEL;
        if (looks_ahead && (inputs.next_use == nullptr || inputs.next_use->size() != trace.runs.size())) {
            throw std::invalid_argument("this sampled policy needs next_use for every run of the trace");
        }
        if (score == SampledScore::LABEL && (inputs.belady_hits == nullptr || inputs.belady_hits->size() != trace.runs.size())) {
            throw std::invalid_argument("the label score needs Belady's hits for every run of the trace");
        }

        resident_.reserve(std::min<size_t>(pool, trace.page_limit));
        if (score != SampledScore::MODEL) {
            page_score_.assign(trace.page_limit, 0.0);
            return;
        }

        if (inputs.model == nullptr) throw std::invalid_argument("the learned policy needs a model");
        if (inputs.model->features == FeatureSet::RICHER && inputs.owners == nullptr) {
            throw std::invalid_argument("a model with owner inputs needs the page owner map");
        }
        // Room for the most candidates an eviction can have, so no eviction allocates.
        const size_t most = std::min(config.k, pool);
        scorer_.emplace(*inputs.model);
        scorer_->Reserve(most);
        model_inputs_.resize(inputs.model->inputs * most);
        model_scores_.resize(most);
    }

    bool Access(size_t run_index, const Run& run) override {
        const uint32_t page = run.page;
        const bool miss = in_pool_[page] == 0;
        if (miss) {
            if (resident_.size() == pool_) EvictOne(run_index, run);
            in_pool_[page] = 1;
            resident_.push_back(page);
        }

        // The page's score only changes when it is used, so I work it out here, once.
        switch (score_kind_) {
            case SampledScore::RANDOM:
            case SampledScore::MODEL:
                break;
            case SampledScore::LRU:
                page_score_[page] = static_cast<double>(run_index);
                break;
            case SampledScore::BELADY:
                // kNever is the biggest number, so a page never used again scores lowest.
                page_score_[page] = -static_cast<double>((*inputs_.next_use)[run_index]);
                break;
            case SampledScore::LABEL: {
                const uint8_t label = RelaxedBeladyLabel(static_cast<uint32_t>(run_index), *inputs_.next_use, *inputs_.belady_hits);
                page_score_[page] = static_cast<double>(run_index) + (label != 0 ? kLabelOne : 0.0);
                break;
            }
        }
        history_.Update(run_index, run);
        return miss;
    }

    // The pages in the pool, packed, in no particular order.
    [[nodiscard]] const std::vector<uint32_t>& Resident() const { return resident_; }

    // The pages the latest eviction chose from, in the order drawn. Empty before the first one.
    [[nodiscard]] const std::vector<uint32_t>& LastCandidates() const {
        if (candidates_stale_) {
            const size_t last = resident_.size() - 1;
            candidates_.resize(last_count_);
            for (size_t i = 0; i < last_count_; ++i) {
                candidates_[i] = i == last_victim_slot_ ? last_victim_ : i == last ? resident_[last_victim_slot_] : resident_[i];
            }
            candidates_stale_ = false;
        }
        return candidates_;
    }

    [[nodiscard]] const PageHistory& History() const { return history_; }

    // The model's scorer (its work counters), null unless this policy scores with a model.
    [[nodiscard]] const ModelScorer* Scorer() const { return scorer_ ? &*scorer_ : nullptr; }

    // Called before each eviction, while the pool is still whole and nothing has been drawn.
    void SetDecisionHook(DecisionHook hook) { hook_ = std::move(hook); }

    // Nanoseconds each eviction took, from before the draw to the pick. Only with time_decisions.
    [[nodiscard]] const std::vector<uint32_t>& DecisionNs() const { return decision_ns_; }
    [[nodiscard]] uint64_t Decisions() const { return evictions_; }

private:
    static constexpr double kLabelOne = 1099511627776.0;  // 2^40, above any run index

    void EvictOne(size_t run_index, const Run& run) {
        if (hook_) hook_(run_index, run);

        using Clock = std::chrono::steady_clock;
        const Clock::time_point start = config_.time_decisions ? Clock::now() : Clock::time_point{};

        // The candidates end up at the front of the pool array. With no more pages than
        // candidates that is the whole array as it stands, and nothing is drawn.
        const size_t size = resident_.size();
        const size_t count = std::min(config_.k, size);
        if (count < size) {
            for (size_t i = 0; i < count; ++i) {
                const auto j = static_cast<size_t>(rng_.Uniform(static_cast<int64_t>(i), static_cast<int64_t>(size) - 1));
                std::swap(resident_[i], resident_[j]);
            }
        }

        // Lowest score goes; on a tie, the one drawn first.
        size_t slot = 0;
        if (score_kind_ == SampledScore::MODEL) {
            slot = LowestModelScore(count, run_index, run.first_fetch);
        } else {
            double lowest = page_score_[resident_[0]];
            for (size_t i = 1; i < count; ++i) {
                const double score = page_score_[resident_[i]];
                if (score < lowest) {
                    lowest = score;
                    slot = i;
                }
            }
        }

        if (config_.time_decisions) {
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
            decision_ns_.push_back(static_cast<uint32_t>(std::clamp<int64_t>(ns, 0, std::numeric_limits<uint32_t>::max())));
        }

        last_victim_ = resident_[slot];
        last_victim_slot_ = slot;
        last_count_ = count;
        candidates_stale_ = true;

        in_pool_[last_victim_] = 0;
        resident_[slot] = resident_.back();
        resident_.pop_back();
        ++evictions_;
    }

    // Scores the first `count` pages of the pool with the model, for a decision at run r whose
    // first fetch is t. Returns the slot of the lowest score, the first one on a tie.
    [[nodiscard]] size_t LowestModelScore(size_t count, size_t r, uint64_t t) {
        const size_t inputs = inputs_.model->inputs;
        std::array<float, kMaxFeatures> row;
        for (size_t c = 0; c < count; ++c) {
            history_.Features(resident_[c], r, t, inputs, row.data());
            for (size_t f = 0; f < inputs; ++f) model_inputs_[f * count + c] = row[f];
        }
        scorer_->Score(model_inputs_.data(), count, model_scores_.data());

        // A score that is not a number would win or lose every comparison for no reason.
        size_t slot = 0;
        for (size_t c = 0; c < count; ++c) {
            const float score = model_scores_[c];
            if (!std::isfinite(score)) {
                throw std::runtime_error("the model's score for page " + std::to_string(resident_[c]) + " at run " + std::to_string(r) + " is not a number");
            }
            if (score < model_scores_[slot]) slot = c;
        }
        return slot;
    }

    size_t pool_;
    SampledScore score_kind_;
    SampledConfig config_;
    SampledInputs inputs_;
    SampleRng rng_;
    PageHistory history_;
    DecisionHook hook_;

    std::vector<uint32_t> resident_;
    std::vector<uint8_t> in_pool_;    // by page id
    std::vector<double> page_score_;  // by page id; not used by the model

    // Only with a model.
    std::optional<ModelScorer> scorer_;
    std::vector<float> model_inputs_;  // [feature][candidate]
    std::vector<float> model_scores_;
    std::vector<uint32_t> decision_ns_;

    // Enough to rebuild the latest candidates when asked, so an eviction never copies them.
    mutable std::vector<uint32_t> candidates_;
    mutable bool candidates_stale_{false};
    uint32_t last_victim_{0};
    size_t last_victim_slot_{0};
    size_t last_count_{0};
};

// What the extra policy names need on top of the simulator's own context.
// It must outlive the policies made from it.
struct LearnedContext {
    const OwnerMap* owners{nullptr};
    const LearnedModel* model{nullptr};
    std::map<size_t, std::vector<uint8_t>> belady_hits;  // by pool, for sampled_label
    SampledConfig config;
};

[[nodiscard]] inline const std::vector<std::string>& LearnedPolicyNames() {
    static const std::vector<std::string> names = {"learned", "sampled_random", "sampled_lru", "sampled_label", "sampled_belady"};
    return names;
}

// Builds the names above; every other name goes to MakePolicy unchanged.
[[nodiscard]] inline std::unique_ptr<Policy> MakeAnyPolicy(const std::string& name, size_t pool, const SimContext& ctx, const LearnedContext& lctx) {
    SampledScore score;
    if (name == "learned") score = SampledScore::MODEL;
    else if (name == "sampled_random") score = SampledScore::RANDOM;
    else if (name == "sampled_lru") score = SampledScore::LRU;
    else if (name == "sampled_label") score = SampledScore::LABEL;
    else if (name == "sampled_belady") score = SampledScore::BELADY;
    else return MakePolicy(name, pool, ctx);

    if (pool == 0) throw std::invalid_argument("pool must be at least 1");
    if (ctx.trace == nullptr) throw std::invalid_argument("context has no trace");

    SampledInputs inputs;
    inputs.owners = lctx.owners;
    inputs.model = lctx.model;
    if (score == SampledScore::MODEL && lctx.model == nullptr) throw std::invalid_argument(name + " needs a model");
    if (score == SampledScore::BELADY || score == SampledScore::LABEL) {
        if (ctx.next_use == nullptr) throw std::invalid_argument(name + " needs next_use");
        inputs.next_use = ctx.next_use;
    }
    if (score == SampledScore::LABEL) {
        const auto hits = lctx.belady_hits.find(pool);
        if (hits == lctx.belady_hits.end()) throw std::invalid_argument(name + " needs Belady's hits at pool " + std::to_string(pool));
        inputs.belady_hits = &hits->second;
    }
    return std::make_unique<SampledPolicy>(pool, *ctx.trace, score, lctx.config, inputs);
}


// A sampled policy that looks at every page in the pool,
// so it is exactly LRU or Belady.
[[nodiscard]] inline SampledConfig SampleAllConfig() {
    SampledConfig config;
    config.k = std::numeric_limits<size_t>::max();
    return config;
}

// One row per recorded candidate: the label, then the 29 features.
constexpr size_t kSampleRowWidth = 1 + kMaxFeatures;

struct SampleSet {
    uint64_t evictions{0}, replay_misses{0};
    size_t decisions{0};
    std::vector<float> rows;                        // per row: label, then the 29 features
    std::vector<uint32_t> run_of_row, page_of_row;  // for checks; not written
};

// Replays the trace as exactly LRU or Belady. At a reservoir sample of `decisions` of its
// evictions, it records k candidates drawn uniformly from the pages in the pool.
[[nodiscard]] inline SampleSet CollectSamples(const Trace& trace, size_t pool, SampledScore replay, const OwnerMap& owners, const std::vector<uint32_t>& next_use, const std::vector<uint8_t>& belady_hits, size_t k, size_t decisions, uint64_t seed) {
    if (replay != SampledScore::LRU && replay != SampledScore::BELADY) throw std::invalid_argument("the replay must be LRU or Belady");
    if (k == 0 || decisions == 0) throw std::invalid_argument("k and decisions must be at least 1");

    SampledInputs inputs;
    inputs.owners = &owners;
    inputs.next_use = &next_use;
    SampledPolicy policy(pool, trace, replay, SampleAllConfig(), inputs);

    struct Slot {
        std::vector<float> rows;
        std::vector<uint32_t> pages;
        uint32_t run{0};
    };
    std::vector<Slot> slots;
    slots.reserve(std::min<size_t>(decisions, 1u << 20));

    SampleRng rng(seed);
    uint64_t evictions = 0;
    std::vector<uint32_t> copy;
    policy.SetDecisionHook([&](size_t run_index, const Run& run) {
        const uint64_t e = evictions++;
        size_t slot;
        if (e < decisions) {
            slot = static_cast<size_t>(e);
            slots.emplace_back();
        } else {
            const auto j = static_cast<size_t>(rng.Uniform(0, static_cast<int64_t>(e)));
            if (j >= decisions) return;
            slot = j;
        }

        // Partial shuffle of a copy, so the policy's own array stays as it is.
        const std::vector<uint32_t>& resident = policy.Resident();
        copy.assign(resident.begin(), resident.end());
        const size_t count = std::min(k, copy.size());
        Slot& out = slots[slot];
        out.rows.assign(count * kSampleRowWidth, 0.0f);
        out.pages.resize(count);
        out.run = static_cast<uint32_t>(run_index);

        for (size_t i = 0; i < count; ++i) {
            const auto j = static_cast<size_t>(rng.Uniform(static_cast<int64_t>(i), static_cast<int64_t>(copy.size()) - 1));
            std::swap(copy[i], copy[j]);
            const uint32_t page = copy[i];
            float* row = out.rows.data() + i * kSampleRowWidth;
            row[0] = static_cast<float>(RelaxedBeladyLabel(policy.History().LastRun(page), next_use, belady_hits));
            policy.History().Features(page, run_index, run.first_fetch, kMaxFeatures, row + 1);
            out.pages[i] = page;
        }
    });

    SampleSet set;
    for (size_t i = 0; i < trace.runs.size(); ++i) {
        if (policy.Access(i, trace.runs[i])) ++set.replay_misses;
    }
    set.evictions = policy.Evictions();
    set.decisions = slots.size();
    for (const Slot& slot : slots) {
        set.rows.insert(set.rows.end(), slot.rows.begin(), slot.rows.end());
        set.page_of_row.insert(set.page_of_row.end(), slot.pages.begin(), slot.pages.end());
        set.run_of_row.insert(set.run_of_row.end(), slot.pages.size(), slot.run);
    }
    return set;
}

struct SampleMeta {
    std::string sf, trace, replay;
    size_t pool{0}, k{0};
    uint64_t seed{0};
};

// Writes <prefix>.bin (little-endian float32 rows) and <prefix>.json (what the rows are).
inline void WriteSamples(const SampleSet& set, const SampleMeta& meta, const std::string& prefix) {
    const uint16_t probe = 1;
    if (*reinterpret_cast<const uint8_t*>(&probe) != 1) throw std::runtime_error("sample files need a little-endian machine");
    static_assert(sizeof(float) == 4, "float32 expected");

    std::ofstream bin(prefix + ".bin", std::ios::binary);
    if (!bin) throw std::runtime_error("cannot write " + prefix + ".bin");
    bin.write(reinterpret_cast<const char*>(set.rows.data()), static_cast<std::streamsize>(set.rows.size() * sizeof(float)));
    bin.close();
    if (!bin) throw std::runtime_error("failed writing " + prefix + ".bin");

    std::ofstream json(prefix + ".json");
    if (!json) throw std::runtime_error("cannot write " + prefix + ".json");
    json << "{\n  \"format\": 1,\n  \"sf\": \"" << meta.sf << "\",\n  \"trace\": \"" << meta.trace << "\",\n  \"replay\": \"" << meta.replay
         << "\",\n  \"pool\": " << meta.pool << ",\n  \"k\": " << meta.k << ",\n  \"seed\": " << meta.seed
         << ",\n  \"decisions\": " << set.decisions << ",\n  \"evictions\": " << set.evictions
         << ",\n  \"replay_misses\": " << set.replay_misses << ",\n  \"rows\": " << set.rows.size() / kSampleRowWidth
         << ",\n  \"columns\": [\"label\"";
         
    for (const char* name : FeatureNames()) json << ", \"" << name << "\"";
    json << "]\n}\n";
    json.close();
    if (!json) throw std::runtime_error("failed writing " + prefix + ".json");
}

}  // namespace shilmandb::evsim
