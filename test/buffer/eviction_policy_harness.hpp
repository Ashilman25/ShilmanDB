#pragma once
// Shared by the eviction policy tests: a sanctioned exception to "each test file defines its own helpers".

#include "buffer/eviction_policy.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace shilmandb::test {

// A page-level buffer pool over one policy, with no disk. It calls the policy the way
// BufferPoolManager does: free frames go out in order 0, 1, 2, ...; a miss with no free frame
// calls Evict(page); then RecordAccess(frame, page, type), SetEvictable(frame, false), and the
// unpin (SetEvictable(frame, true)) unless the page is held.
class PolicyHarness {
public:
    PolicyHarness(EvictionPolicy& policy, size_t pool) : policy_(policy), page_in_(pool, INVALID_PAGE_ID) {
        for (size_t f = 0; f < pool; ++f) free_.push_back(static_cast<frame_id_t>(f));
    }

    // Returns true on a miss. Throws std::runtime_error if every frame is pinned, and
    // std::logic_error for a repeat that misses or if the policy picks an empty or pinned frame.
    bool Access(page_id_t page, AccessType type) {
        last_victim_.reset();
        frame_id_t frame;
        bool miss = false;
        if (const auto it = frame_of_.find(page); it != frame_of_.end()) {
            frame = it->second;
        } else {
            if (type == AccessType::REPEAT) throw std::logic_error("a miss can't be a repeat");
            miss = true;
            frame = TakeFrame(page);
            frame_of_[page] = frame;
            page_in_[frame] = page;
        }
        policy_.RecordAccess(frame, page, type);
        policy_.SetEvictable(frame, false);
        if (!IsHeld(page)) policy_.SetEvictable(frame, true);
        return miss;
    }

    // A held page stays pinned whenever it is resident: now, and after every access or load.
    void Hold(page_id_t page) {
        if (!held_.insert(page).second) return;
        if (const auto it = frame_of_.find(page); it != frame_of_.end()) policy_.SetEvictable(it->second, false);
    }

    void Release(page_id_t page) {
        if (held_.erase(page) == 0) return;
        if (const auto it = frame_of_.find(page); it != frame_of_.end()) policy_.SetEvictable(it->second, true);
    }

    // Deletes a resident page (calls Remove); its frame goes to the back of the free list.
    // A page that isn't resident is ignored; a held one throws std::logic_error.
    void Drop(page_id_t page) {
        const auto it = frame_of_.find(page);
        if (it == frame_of_.end()) return;
        if (IsHeld(page)) throw std::logic_error("can't drop a held page");
        const frame_id_t frame = it->second;
        policy_.Remove(frame);
        frame_of_.erase(it);
        page_in_[frame] = INVALID_PAGE_ID;
        free_.push_back(frame);
    }

    // The page the last Access evicted, if it evicted one.
    [[nodiscard]] std::optional<page_id_t> LastVictim() const { return last_victim_; }

    [[nodiscard]] bool Resident(page_id_t page) const { return frame_of_.find(page) != frame_of_.end(); }

private:
    [[nodiscard]] bool IsHeld(page_id_t page) const { return held_.find(page) != held_.end(); }

    frame_id_t TakeFrame(page_id_t page) {
        if (!free_.empty()) {
            const frame_id_t frame = free_.front();
            free_.pop_front();
            return frame;
        }
        const auto victim = policy_.Evict(page);
        if (!victim.has_value()) throw std::runtime_error("every frame is pinned");
        const frame_id_t frame = *victim;
        if (frame >= page_in_.size() || page_in_[frame] == INVALID_PAGE_ID) {
            throw std::logic_error("policy evicted an empty frame " + std::to_string(frame));
        }
        const page_id_t old = page_in_[frame];
        if (IsHeld(old)) throw std::logic_error("policy evicted pinned page " + std::to_string(old));
        frame_of_.erase(old);
        last_victim_ = old;
        return frame;
    }

    EvictionPolicy& policy_;
    std::vector<page_id_t> page_in_;  // page in each frame, INVALID_PAGE_ID if free
    std::unordered_map<page_id_t, frame_id_t> frame_of_;
    std::deque<frame_id_t> free_;
    std::unordered_set<page_id_t> held_;
    std::optional<page_id_t> last_victim_;
};

// The buffer pool's access-type rule at any window. A run is a stretch of accesses to one page.
// A miss is a new use; a hit on the same page as the access before is a repeat; any other hit
// is a touch if it comes within `window` runs of that page's last run, else a new use.
class VisitTypes {
public:
    explicit VisitTypes(uint64_t window) : window_(window) {}

    AccessType Next(page_id_t page, bool resident) {
        const bool page_change = page != last_page_;
        if (page_change) {
            ++run_;
            last_page_ = page;
        }
        const auto [it, first_time] = last_run_.try_emplace(page, run_);
        const uint64_t previous = it->second;
        it->second = run_;

        if (!resident) return AccessType::NEW_USE;
        if (!page_change) return AccessType::REPEAT;
        return !first_time && run_ - previous <= window_ ? AccessType::TOUCH : AccessType::NEW_USE;
    }

private:
    uint64_t window_;
    uint64_t run_{0};
    page_id_t last_page_{INVALID_PAGE_ID};
    std::unordered_map<page_id_t, uint64_t> last_run_;
};

// Misses for a page sequence on a new pool. With no window every access is a NEW_USE (plain
// feeding); with one the types come from VisitTypes.
inline uint64_t CountMisses(EvictionPolicy& policy, size_t pool, const std::vector<page_id_t>& pages,
                            std::optional<uint64_t> window = std::nullopt) {
    PolicyHarness harness(policy, pool);
    std::optional<VisitTypes> visits;
    if (window.has_value()) visits.emplace(*window);
    uint64_t misses = 0;
    for (const page_id_t page : pages) {
        const AccessType type = visits ? visits->Next(page, harness.Resident(page)) : AccessType::NEW_USE;
        if (harness.Access(page, type)) ++misses;
    }
    return misses;
}

// Checking a policy against a slow reference written in the test file. A Reference has:
//   explicit Reference(size_t num_frames);
//   void RecordAccess(frame_id_t, page_id_t, AccessType);
//   void SetEvictable(frame_id_t, bool);
//   std::optional<frame_id_t> Evict(page_id_t incoming_page);
//   void Remove(frame_id_t);
//   size_t Size() const;

// Calls the policy and the reference with the same arguments. Keeps the first call where they
// disagree on the victim or on Size(), or where the after-call check (if set) reports a problem.
template <typename Reference>
class Twin : public EvictionPolicy {
public:
    Twin(EvictionPolicy& fast, size_t num_frames) : fast_(fast), ref_(num_frames) {}

    void RecordAccess(frame_id_t) override { throw std::logic_error("not used"); }
    void RecordAccess(frame_id_t frame_id, page_id_t page_id) override { RecordAccess(frame_id, page_id, AccessType::NEW_USE); }
    void RecordAccess(frame_id_t frame_id, page_id_t page_id, AccessType type) override {
        fast_.RecordAccess(frame_id, page_id, type);
        ref_.RecordAccess(frame_id, page_id, type);
        Check("RecordAccess");
    }
    void SetEvictable(frame_id_t frame_id, bool evictable) override {
        fast_.SetEvictable(frame_id, evictable);
        ref_.SetEvictable(frame_id, evictable);
        Check("SetEvictable");
    }
    std::optional<frame_id_t> Evict() override { return Evict(INVALID_PAGE_ID); }
    std::optional<frame_id_t> Evict(page_id_t incoming_page) override {
        const auto fast = fast_.Evict(incoming_page);
        const auto ref = ref_.Evict(incoming_page);
        if (fast != ref && error_.empty()) error_ = "Evict picked frame " + Name(fast) + ", reference " + Name(ref);
        Check("Evict");
        return fast;
    }
    void Remove(frame_id_t frame_id) override {
        fast_.Remove(frame_id);
        ref_.Remove(frame_id);
        Check("Remove");
    }
    size_t Size() const override { return fast_.Size(); }
    bool UsesAccessTypes() const override { return true; }
    void Reset() override { throw std::logic_error("not used"); }

    // Runs after every call; returns an empty string when all is well
    void SetAfterCall(std::function<std::string()> check) { after_call_ = std::move(check); }

    [[nodiscard]] const Reference& GetReference() const { return ref_; }
    [[nodiscard]] const std::string& Error() const { return error_; }

private:
    static std::string Name(std::optional<frame_id_t> frame) { return frame ? std::to_string(*frame) : "none"; }

    void Check(const char* call) {
        if (!error_.empty()) return;
        if (fast_.Size() != ref_.Size()) {
            error_ = std::string(call) + ": Size " + std::to_string(fast_.Size()) + ", reference " + std::to_string(ref_.Size());
        } else if (after_call_) {
            if (std::string problem = after_call_(); !problem.empty()) error_ = std::string(call) + ": " + problem;
        }
    }

    EvictionPolicy& fast_;
    Reference ref_;
    std::string error_;
    std::function<std::string()> after_call_;
};

// A random workload through the harness: runs of 1-5 repeats; a quarter of the runs come from
// a pair of pages that changes every 20 runs, the rest go 70% to 30 hot pages and 30% to the
// other 270; about 2% holds or releases (never more than pool - 1 held) and 0.5% drops.
// Types come from the visit rule at `window`. Returns the twin's first error with where it
// happened, or an empty string.
template <typename Reference>
std::string RandomOps(Twin<Reference>& twin, size_t pool, uint64_t window, std::mt19937& rng, int count) {
    constexpr uint32_t kPages = 300;
    constexpr uint32_t kHot = 30;

    PolicyHarness harness(twin, pool);
    VisitTypes visits(window);
    std::vector<page_id_t> held;
    const auto is_held = [&](page_id_t p) { return std::find(held.begin(), held.end(), p) != held.end(); };

    page_id_t page = 0;
    page_id_t last = INVALID_PAGE_ID;
    uint32_t run_left = 0;
    page_id_t pair[2] = {0, 1};
    int pair_runs = 0;

    for (int i = 0; i < count; ++i) {
        const uint32_t roll = rng() % 1000;
        if (roll < 20) {
            if (!held.empty() && (rng() % 2 == 0 || held.size() + 2 > pool)) {
                const size_t k = rng() % held.size();
                harness.Release(held[k]);
                held.erase(held.begin() + static_cast<std::ptrdiff_t>(k));
            } else if (held.size() + 2 <= pool) {
                const page_id_t p = rng() % kHot;
                if (!is_held(p)) {
                    harness.Hold(p);
                    held.push_back(p);
                }
            }
        } else if (roll < 25) {
            const page_id_t p = rng() % 2 == 0 ? last : rng() % kHot;
            if (p != INVALID_PAGE_ID && harness.Resident(p) && !is_held(p)) harness.Drop(p);
        } else {
            if (run_left == 0) {
                if (pair_runs == 0) {
                    pair[0] = rng() % kPages;
                    pair[1] = (pair[0] + 1 + rng() % (kPages - 1)) % kPages;
                    pair_runs = 20;
                }
                --pair_runs;
                if (rng() % 4 == 0) {
                    page = pair[rng() % 2];
                } else {
                    page = rng() % 10 < 7 ? rng() % kHot : kHot + rng() % (kPages - kHot);
                }
                run_left = 1 + rng() % 5;
            }
            --run_left;
            harness.Access(page, visits.Next(page, harness.Resident(page)));
            last = page;
        }
        if (!twin.Error().empty()) {
            return twin.Error() + " (op " + std::to_string(i) + ", pool " + std::to_string(pool) + ", window " + std::to_string(window) + ")";
        }
    }
    return "";
}

struct RandomOpsResult {
    uint64_t ops{0};
    std::string error;  // empty when every call matched
};

// Seeds 1-3, windows 3 and 64, every pool in `pools`, 4,200 ops each, one random generator per
// seed. `check` (optional) runs after every call with the policy and the reference, and returns
// an empty string when all is well. Stops at the first problem.
template <typename Policy, typename Reference>
RandomOpsResult MatchReferenceOnRandomOps(const std::vector<size_t>& pools,
                                          const std::function<std::string(const Policy&, const Reference&)>& check = {}) {
    constexpr int kOpsPerRound = 4200;
    RandomOpsResult result;
    for (const unsigned seed : {1u, 2u, 3u}) {
        std::mt19937 rng(seed);
        for (const uint64_t window : {3u, 64u}) {
            for (const size_t pool : pools) {
                Policy policy(pool);
                Twin<Reference> twin(policy, pool);
                if (check) twin.SetAfterCall([&] { return check(policy, twin.GetReference()); });
                result.error = RandomOps(twin, pool, window, rng, kOpsPerRound);
                if (!result.error.empty()) {
                    result.error += " seed " + std::to_string(seed);
                    return result;
                }
                result.ops += kOpsPerRound;
            }
        }
    }
    return result;
}

}  // namespace shilmandb::test
