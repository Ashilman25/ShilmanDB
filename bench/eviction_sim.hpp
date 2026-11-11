#pragma once

// Replays a recorded buffer-pool trace through eviction policies and counts misses.
// Header-only, and it does not use any engine code.

#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace shilmandb::evsim {

// One page fetched `length` times in a row, starting at fetch number `first_fetch`.
struct Run {
    uint32_t page;
    uint64_t first_fetch;
    uint32_t length;
};

struct Trace {
    std::vector<Run> runs;
    uint64_t end_fetch{0};   // fetch number just past the last fetch
    uint32_t page_limit{0};  // largest page id + 1
};

// Lines are (timestamp, page) in file order. Without `raw`, consecutive equal pages merge.
[[nodiscard]] inline Trace MakeTrace(const std::vector<std::pair<uint64_t, uint32_t>>& lines, uint64_t end_fetch, bool raw = false) {
    if (lines.empty()) throw std::runtime_error("trace has no lines");
    for (size_t i = 1; i < lines.size(); ++i) {
        if (lines[i].first <= lines[i - 1].first) {
            throw std::runtime_error("trace timestamps must strictly increase");
        }
    }
    if (end_fetch <= lines.back().first) {
        throw std::runtime_error("end_fetch must be past the last timestamp");
    }

    Trace trace;
    trace.end_fetch = end_fetch;
    for (const auto& [ts, page] : lines) {
        if (!raw && !trace.runs.empty() && trace.runs.back().page == page) continue;
        trace.runs.push_back(Run{page, ts, 0});
        trace.page_limit = std::max(trace.page_limit, page + 1);
    }
    if (trace.runs.size() >= UINT32_MAX) throw std::runtime_error("trace has too many runs");

    for (size_t i = 0; i < trace.runs.size(); ++i) {
        const uint64_t next = i + 1 < trace.runs.size() ? trace.runs[i + 1].first_fetch : end_fetch;
        const uint64_t length = next - trace.runs[i].first_fetch;
        if (length > UINT32_MAX) throw std::runtime_error("run is too long");
        trace.runs[i].length = static_cast<uint32_t>(length);
    }
    return trace;
}

namespace detail {

// Reads an unsigned number at p, stops at `stop`, and moves p past it.
inline uint64_t ParseNumber(const char*& p, const char* end, char stop, const char* what) {
    if (p == end || *p < '0' || *p > '9') throw std::runtime_error(std::string("bad ") + what + " in trace");
    uint64_t value = 0;
    while (p != end && *p >= '0' && *p <= '9') {
        value = value * 10 + static_cast<uint64_t>(*p - '0');
        if (value > 1000000000000000ULL) throw std::runtime_error("number too big in trace");
        ++p;
    }
    if (p != end && *p != stop && *p != '\r') throw std::runtime_error(std::string("bad ") + what + " in trace");
    return value;
}

inline std::string ReadFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

}  // namespace detail

// Loads a "timestamp,page_id" file.
[[nodiscard]] inline Trace LoadTrace(const std::string& path, uint64_t end_fetch, bool raw = false) {
    const std::string text = detail::ReadFile(path);
    const char* p = text.data();
    const char* const end = p + text.size();

    const char* eol = std::find(p, end, '\n');
    std::string header(p, eol);
    if (!header.empty() && header.back() == '\r') header.pop_back();
    if (header != "timestamp,page_id") throw std::runtime_error("trace header must be timestamp,page_id");
    p = eol == end ? end : eol + 1;

    std::vector<std::pair<uint64_t, uint32_t>> lines;
    while (p != end) {
        if (*p == '\n' || *p == '\r') { ++p; continue; }
        const uint64_t ts = detail::ParseNumber(p, end, ',', "timestamp");
        if (p == end || *p != ',') throw std::runtime_error("bad line in trace");
        ++p;
        const uint64_t page = detail::ParseNumber(p, end, '\n', "page id");
        if (page > INT32_MAX) throw std::runtime_error("page id too big in trace");
        lines.emplace_back(ts, static_cast<uint32_t>(page));
    }
    return MakeTrace(lines, end_fetch, raw);
}

// One query or lookup of a workload run, and the runs of the trace it covers.
struct Item {
    int pass{0};
    bool warmup{false};
    int seq{0};
    std::string kind;
    std::string query;
    uint64_t trace_start{0};
    uint64_t trace_end{0};
    size_t first_run{0};
    size_t end_run{0};
};

[[nodiscard]] inline std::vector<Item> LoadItems(const std::string& stream_csv) {
    std::ifstream in(stream_csv);
    if (!in) throw std::runtime_error("cannot open " + stream_csv);

    const auto split = [](const std::string& line) {
        std::vector<std::string> fields;
        std::string field;
        std::stringstream ss(line);
        while (std::getline(ss, field, ',')) fields.push_back(field);
        if (!line.empty() && line.back() == ',') fields.emplace_back();
        return fields;
    };

    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("empty " + stream_csv);
    const auto header = split(line);
    const auto column = [&](const std::string& name) {
        const auto it = std::find(header.begin(), header.end(), name);
        if (it == header.end()) throw std::runtime_error(stream_csv + " has no column " + name);
        return static_cast<size_t>(it - header.begin());
    };
    const size_t c_pass = column("pass"), c_warmup = column("warmup"), c_seq = column("seq");
    const size_t c_kind = column("kind"), c_query = column("query");
    const size_t c_start = column("trace_start"), c_end = column("trace_end");

    std::vector<Item> items;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const auto f = split(line);
        if (f.size() <= std::max({c_pass, c_warmup, c_seq, c_kind, c_query, c_start, c_end})) {
            throw std::runtime_error("short row in " + stream_csv);
        }
        if (f[c_start].empty() || f[c_end].empty()) {
            throw std::runtime_error(stream_csv + " has no trace range (run without tracing)");
        }

        Item item;
        item.pass = std::stoi(f[c_pass]);
        item.warmup = f[c_warmup] == "1" || f[c_warmup] == "true";
        item.seq = std::stoi(f[c_seq]);
        item.kind = f[c_kind];
        item.query = f[c_query];
        item.trace_start = std::stoull(f[c_start]);
        item.trace_end = std::stoull(f[c_end]);
        items.push_back(std::move(item));
    }
    return items;
}

[[nodiscard]] inline uint64_t EndFetch(const std::vector<Item>& items) {
    uint64_t end = 0;
    for (const auto& item : items) end = std::max(end, item.trace_end);
    return end;
}

// Finds which runs each item covers. Throws if the items are out of order or the trace is cut short.
inline void MapItems(const Trace& trace, std::vector<Item>& items) {
    for (size_t i = 0; i < items.size(); ++i) {
        if (items[i].trace_end < items[i].trace_start ||
            (i > 0 && items[i].trace_start < items[i - 1].trace_end)) {
            throw std::runtime_error("items are not in increasing, non-overlapping trace order (item " + std::to_string(i) + ")");
        }
    }
    if (!items.empty() && trace.runs.back().first_fetch < items.back().trace_start) {
        throw std::runtime_error("the trace ends before the last item starts (is the trace file cut short?)");
    }
    const auto first_at_or_after = [&](uint64_t fetch) {
        const auto it = std::lower_bound(trace.runs.begin(), trace.runs.end(), fetch, [](const Run& run, uint64_t f) { return run.first_fetch < f; });
        return static_cast<size_t>(it - trace.runs.begin());
    };
    for (auto& item : items) {
        item.first_run = first_at_or_after(item.trace_start);
        item.end_run = first_at_or_after(item.trace_end);
    }
}

constexpr uint32_t kNever = UINT32_MAX;

// For each run, the index of the next run of the same page (kNever if none).
[[nodiscard]] inline std::vector<uint32_t> NextUse(const Trace& trace) {
    std::vector<uint32_t> next(trace.runs.size(), kNever);
    std::vector<uint32_t> last(trace.page_limit, kNever);
    for (size_t i = trace.runs.size(); i-- > 0;) {
        next[i] = last[trace.runs[i].page];
        last[trace.runs[i].page] = static_cast<uint32_t>(i);
    }
    return next;
}

class Policy {
public:
    virtual ~Policy() = default;

    // Returns true when the run's first fetch misses.
    virtual bool Access(size_t run_index, const Run& run) = 0;
    [[nodiscard]] uint64_t Evictions() const { return evictions_; }

    // The pages now in the pool, in no particular order.
    [[nodiscard]] virtual std::vector<uint32_t> ResidentPages() const {
        throw std::logic_error("policy cannot list its resident pages");
    }

protected:
    uint64_t evictions_{0};
};

// The shipped learned policy's network (4 -> h1 -> h2 -> 1), read from an exported text file.
struct V2Weights {
    std::array<float, 4> mean{}, std{};
    size_t h1{0}, h2{0};
    std::vector<float> w1, b1, w2, b2, w3;  // w1 is [h1][4], w2 is [h2][h1], w3 is [h2]
    float b3{0};
    std::vector<std::array<float, 5>> refs;  // x0 x1 x2 x3 score, as scored by TorchScript
};

struct SimContext {
    const Trace* trace{nullptr};
    const std::vector<uint32_t>* next_use{nullptr};
    const V2Weights* v2{nullptr};
    uint64_t v2_clock_start{0};
    uint32_t visit_window{64};
};

enum class Mode { PLAIN, VISIT, EVERY_FETCH };

// Least recently used, as a linked list threaded through page ids.
class LruPolicy final : public Policy {
public:
    LruPolicy(size_t pool, uint32_t page_limit) : pool_(pool), prev_(page_limit, kNone), next_(page_limit, kNone), resident_(page_limit, false) {}

    bool Access(size_t, const Run& run) override {
        const uint32_t page = run.page;
        if (resident_[page]) {
            Unlink(page);
            PushBack(page);
            return false;
        }
        if (size_ == pool_) {
            const uint32_t victim = head_;
            Unlink(victim);
            resident_[victim] = false;
            --size_;
            ++evictions_;
        }
        resident_[page] = true;
        ++size_;
        PushBack(page);
        return true;
    }

    [[nodiscard]] std::vector<uint32_t> ResidentPages() const override {
        std::vector<uint32_t> pages;
        for (uint32_t p = head_; p != kNone; p = next_[p]) pages.push_back(p);
        return pages;
    }

private:
    static constexpr uint32_t kNone = UINT32_MAX;

    void Unlink(uint32_t page) {
        const uint32_t p = prev_[page], n = next_[page];
        if (p == kNone) head_ = n; else next_[p] = n;
        if (n == kNone) tail_ = p; else prev_[n] = p;
    }
    void PushBack(uint32_t page) {
        prev_[page] = tail_;
        next_[page] = kNone;
        if (tail_ == kNone) head_ = page; else next_[tail_] = page;
        tail_ = page;
    }

    size_t pool_;
    size_t size_{0};
    uint32_t head_{kNone};
    uint32_t tail_{kNone};
    std::vector<uint32_t> prev_, next_;
    std::vector<bool> resident_;
};

// Evicts the page whose next use is furthest away. Needs the whole trace up front.
class BeladyPolicy final : public Policy {
public:
    BeladyPolicy(size_t pool, uint32_t page_limit, const std::vector<uint32_t>& next_use) : pool_(pool), next_use_(next_use), current_(page_limit, kNever), resident_(page_limit, false) {}

    bool Access(size_t run_index, const Run& run) override {
        const uint32_t page = run.page;
        const uint32_t next = next_use_[run_index];
        bool miss = false;
        if (!resident_[page]) {
            miss = true;
            if (size_ == pool_) EvictFurthest();
            resident_[page] = true;
            ++size_;
        }
        current_[page] = next;
        heap_.emplace(next, page);
        return miss;
    }

private:
    void EvictFurthest() {
        // Entries for evicted pages or older next uses are stale; skip them.
        while (!heap_.empty()) {
            const auto [next, page] = heap_.top();
            heap_.pop();
            if (resident_[page] && current_[page] == next) {
                resident_[page] = false;
                --size_;
                ++evictions_;
                return;
            }
        }
        throw std::logic_error("belady heap ran empty");
    }

    size_t pool_;
    size_t size_{0};
    const std::vector<uint32_t>& next_use_;
    std::vector<uint32_t> current_;
    std::vector<bool> resident_;
    std::priority_queue<std::pair<uint32_t, uint32_t>> heap_;
};

namespace detail {

// Doubly linked lists threaded through page ids. A page is in at most one of the lists at
// a time. The front of a list is its oldest end.
class PageLists {
public:
    explicit PageLists(uint32_t page_limit, size_t count = 1) : prev_(page_limit, kNone), next_(page_limit, kNone), ends_(count) {}

    void PushBack(uint32_t page, size_t list = 0) {
        Ends& e = ends_[list];
        prev_[page] = e.tail;
        next_[page] = kNone;
        if (e.tail == kNone) e.head = page; else next_[e.tail] = page;
        e.tail = page;
        ++e.size;
    }
    void Remove(uint32_t page, size_t list = 0) {
        Ends& e = ends_[list];
        const uint32_t p = prev_[page], n = next_[page];
        if (p == kNone) e.head = n; else next_[p] = n;
        if (n == kNone) e.tail = p; else prev_[n] = p;
        --e.size;
    }
    void MoveToBack(uint32_t page, size_t list = 0) {
        Remove(page, list);
        PushBack(page, list);
    }
    uint32_t PopFront(size_t list = 0) {
        const uint32_t page = ends_[list].head;
        Remove(page, list);
        return page;
    }
    [[nodiscard]] uint32_t Front(size_t list = 0) const { return ends_[list].head; }
    [[nodiscard]] size_t Size(size_t list = 0) const { return ends_[list].size; }
    [[nodiscard]] bool Empty(size_t list = 0) const { return ends_[list].size == 0; }

private:
    static constexpr uint32_t kNone = UINT32_MAX;
    struct Ends {
        uint32_t head{kNone};
        uint32_t tail{kNone};
        size_t size{0};
    };
    std::vector<uint32_t> prev_, next_;
    std::vector<Ends> ends_;
};

// Visit mode: a run of a resident page that comes within `window` runs of that page's
// previous run is a touch, not a new use. Off in the other modes.
class VisitTracker {
public:
    VisitTracker(Mode mode, uint32_t window, uint32_t page_limit) : on_(mode == Mode::VISIT), window_(window), last_run_(on_ ? page_limit : 0, kNever) {}

    // Call once for every run, before the policy changes anything.
    [[nodiscard]] bool IsTouch(size_t run_index, uint32_t page, bool resident) {
        if (!on_) return false;
        const uint32_t prev = last_run_[page];
        last_run_[page] = static_cast<uint32_t>(run_index);
        return resident && prev != kNever && run_index - prev <= window_;
    }

private:
    bool on_;
    uint32_t window_;
    std::vector<uint32_t> last_run_;
};

}  // namespace detail

// Pool slots in a circle, each with a reference bit set on load and on every hit. 
// On a miss the hand clears set bits until it finds a clear one, evicts that page, and stops
// one slot past it.
class ClockPolicy final : public Policy {
public:
    ClockPolicy(size_t pool, uint32_t page_limit) : slot_of_(page_limit, kEmpty), page_in_(pool, kEmpty), referenced_(pool, false) {}

    bool Access(size_t, const Run& run) override {
        const uint32_t page = run.page;
        if (slot_of_[page] != kEmpty) {
            referenced_[slot_of_[page]] = true;
            return false;
        }

        size_t slot = filled_;
        if (filled_ < page_in_.size()) {
            ++filled_;
        } else {
            while (referenced_[hand_]) {
                referenced_[hand_] = false;
                hand_ = NextSlot(hand_);
            }
            slot = hand_;
            slot_of_[page_in_[slot]] = kEmpty;
            ++evictions_;
            hand_ = NextSlot(hand_);
        }

        page_in_[slot] = page;
        slot_of_[page] = static_cast<uint32_t>(slot);
        referenced_[slot] = true;
        return true;
    }

    [[nodiscard]] std::vector<uint32_t> ResidentPages() const override {
        return std::vector<uint32_t>(page_in_.begin(), page_in_.begin() + static_cast<std::ptrdiff_t>(filled_));
    }

private:
    static constexpr uint32_t kEmpty = UINT32_MAX;

    [[nodiscard]] size_t NextSlot(size_t slot) const { return slot + 1 == page_in_.size() ? 0 : slot + 1; }

    std::vector<uint32_t> slot_of_;  // slot of each resident page
    std::vector<uint32_t> page_in_;  // page in each slot
    std::vector<bool> referenced_;
    size_t filled_{0};
    size_t hand_{0};
};

// LRU-2: evicts the resident page whose second-to-last use is oldest. Pages used only once
// go first, oldest last use first. History is kept for every page ever seen.
class Lru2Policy final : public Policy {
public:
    Lru2Policy(size_t pool, uint32_t page_limit, Mode mode, uint32_t visit_window) : pool_(pool), every_fetch_(mode == Mode::EVERY_FETCH), visits_(mode, visit_window, page_limit), last_(page_limit, kNone), second_(page_limit, kNone), resident_(page_limit, false) {}

    bool Access(size_t run_index, const Run& run) override {
        const uint32_t page = run.page;
        const bool touch = visits_.IsTouch(run_index, page, resident_[page]);
        const bool miss = !resident_[page];
        if (miss) {
            if (order_.size() == pool_) {
                const uint32_t victim = std::get<2>(*order_.begin());
                order_.erase(order_.begin());
                resident_[victim] = false;
                ++evictions_;
            }
            resident_[page] = true;
        } else {
            order_.erase(KeyOf(page));
        }

        if (touch) {
            last_[page] = static_cast<int64_t>(run_index);  // a touch is not a second use
        } else if (every_fetch_) {
            // L fetches are L uses, at fetch numbers t .. t + L - 1.
            const auto t = static_cast<int64_t>(run.first_fetch);
            const auto len = static_cast<int64_t>(run.length);
            second_[page] = len >= 2 ? t + len - 2 : last_[page];
            last_[page] = t + len - 1;
        } else {
            second_[page] = last_[page];
            last_[page] = static_cast<int64_t>(run_index);
        }
        order_.insert(KeyOf(page));
        return miss;
    }

    [[nodiscard]] std::vector<uint32_t> ResidentPages() const override {
        std::vector<uint32_t> pages;
        for (const auto& key : order_) pages.push_back(std::get<2>(key));
        return pages;
    }

private:
    static constexpr int64_t kNone = -1;
    using Key = std::tuple<int64_t, int64_t, uint32_t>;  // (second-to-last use, last use, page)

    [[nodiscard]] Key KeyOf(uint32_t page) const { return {second_[page], last_[page], page}; }

    size_t pool_;
    bool every_fetch_;
    detail::VisitTracker visits_;
    std::vector<int64_t> last_, second_;
    std::vector<bool> resident_;
    std::set<Key> order_;  // resident pages, next victim first
};

struct ArcSizes {
    size_t t1{0}, t2{0}, b1{0}, b2{0};
    double p{0};
};

// ARC (Megiddo & Modha, FAST 2003, Figure 4). T1 holds pages used once lately, T2 pages
// used at least twice; the ghost lists B1 and B2 remember pages just evicted from each.
// The target size p of T1 grows on a B1 hit and shrinks on a B2 hit.
class ArcPolicy final : public Policy {
public:
    ArcPolicy(size_t pool, uint32_t page_limit, Mode mode, uint32_t visit_window) : c_(pool), every_fetch_(mode == Mode::EVERY_FETCH), visits_(mode, visit_window, page_limit), lists_(page_limit, 4), where_(page_limit, kOut) {}

    bool Access(size_t run_index, const Run& run) override {
        const uint32_t x = run.page;
        const List where = where_[x];
        if (visits_.IsTouch(run_index, x, where == kT1 || where == kT2)) {
            lists_.MoveToBack(x, where);  // stays in its own list
            CheckInvariants();
            return false;
        }

        bool miss = true;
        switch (where) {
            case kT1:
            case kT2:
                MoveTo(x, kT2);
                miss = false;
                break;
            case kB1: {
                const auto b1 = static_cast<double>(Size(kB1)), b2 = static_cast<double>(Size(kB2));
                p_ = std::min(static_cast<double>(c_), p_ + (b1 >= b2 ? 1.0 : b2 / b1));
                Replace(false);
                MoveTo(x, kT2);
                break;
            }
            case kB2: {
                const auto b1 = static_cast<double>(Size(kB1)), b2 = static_cast<double>(Size(kB2));
                p_ = std::max(0.0, p_ - (b2 >= b1 ? 1.0 : b1 / b2));
                Replace(true);
                MoveTo(x, kT2);
                break;
            }
            case kOut: {
                const size_t l1 = Size(kT1) + Size(kB1);
                const size_t total = l1 + Size(kT2) + Size(kB2);
                if (l1 == c_) {
                    if (Size(kT1) < c_) {
                        Drop(lists_.Front(kB1));
                        Replace(false);
                    } else {
                        Drop(lists_.Front(kT1));  // evicted without a ghost
                        ++evictions_;
                    }
                } else if (total >= c_) {
                    if (total == 2 * c_) Drop(lists_.Front(kB2));
                    Replace(false);
                }
                MoveTo(x, kT1);
                break;
            }
        }
        // Every fetch after the first is a hit, which puts the page at the MRU end of T2.
        if (every_fetch_ && run.length >= 2) MoveTo(x, kT2);
        CheckInvariants();
        return miss;
    }

    [[nodiscard]] ArcSizes Sizes() const { return {Size(kT1), Size(kT2), Size(kB1), Size(kB2), p_}; }

    [[nodiscard]] std::vector<uint32_t> ResidentPages() const override {
        std::vector<uint32_t> pages;
        for (size_t p = 0; p < where_.size(); ++p) {
            if (where_[p] == kT1 || where_[p] == kT2) pages.push_back(static_cast<uint32_t>(p));
        }
        return pages;
    }

private:
    enum List : uint8_t { kT1, kT2, kB1, kB2, kOut };

    [[nodiscard]] size_t Size(List list) const { return lists_.Size(list); }

    void MoveTo(uint32_t page, List to) {
        if (where_[page] != kOut) lists_.Remove(page, where_[page]);
        lists_.PushBack(page, to);
        where_[page] = to;
    }
    void Drop(uint32_t page) {
        lists_.Remove(page, where_[page]);
        where_[page] = kOut;
    }
    // Evicts the LRU page of T1 or T2 into its ghost list.
    void Replace(bool x_in_b2) {
        const auto t1 = static_cast<double>(Size(kT1));
        if (Size(kT1) >= 1 && ((x_in_b2 && t1 == p_) || t1 > p_)) {
            MoveTo(lists_.Front(kT1), kB1);
        } else {
            if (lists_.Empty(kT2)) throw std::logic_error("arc: REPLACE found T2 empty");
            MoveTo(lists_.Front(kT2), kB2);
        }
        ++evictions_;
    }
    void CheckInvariants() const {
        assert(Size(kT1) + Size(kT2) <= c_);
        assert(Size(kT1) + Size(kB1) <= c_);
        assert(Size(kT1) + Size(kT2) + Size(kB1) + Size(kB2) <= 2 * c_);
        assert(p_ >= 0.0 && p_ <= static_cast<double>(c_));
    }

    size_t c_;
    bool every_fetch_;
    detail::VisitTracker visits_;
    detail::PageLists lists_;
    std::vector<List> where_;
    double p_{0};
};

class LirsPolicy final : public Policy {
public:
    LirsPolicy(size_t pool, uint32_t page_limit, Mode mode, uint32_t visit_window) : pool_(pool), lir_limit_(LirLimit(pool)), every_fetch_(mode == Mode::EVERY_FETCH), visits_(mode, visit_window, page_limit), s_(page_limit), q_(page_limit), resident_(page_limit, false), lir_(page_limit, false), in_s_(page_limit, false) {}

    bool Access(size_t run_index, const Run& run) override {
        const uint32_t x = run.page;
        if (visits_.IsTouch(run_index, x, resident_[x])) {
            if (lir_[x]) {
                MoveToTop(x);
                Prune();
            } else {
                q_.MoveToBack(x);  // S stays as it is
            }
            return false;
        }
        const bool miss = Use(x);
        if (every_fetch_) {
            // Once the page is LIR at the top of S, more uses change nothing.
            for (uint32_t k = 1; k < run.length && !lir_[x]; ++k) (void)Use(x);
        }
        return miss;
    }

    [[nodiscard]] std::vector<uint32_t> ResidentPages() const override {
        std::vector<uint32_t> pages;
        for (size_t p = 0; p < resident_.size(); ++p) {
            if (resident_[p]) pages.push_back(static_cast<uint32_t>(p));
        }
        return pages;
    }

private:
    static size_t LirLimit(size_t pool) {
        if (pool < 2) throw std::invalid_argument("lirs needs a pool of at least 2");
        const long hir = std::max(1L, std::lround(0.01 * static_cast<double>(pool)));
        return pool - static_cast<size_t>(hir);
    }

    bool Use(uint32_t x) {
        if (lir_[x]) {
            MoveToTop(x);
            Prune();
            return false;
        }
        if (resident_[x]) {
            if (in_s_[x]) {
                MoveToTop(x);
                RemoveFromQ(x);
                lir_[x] = true;
                ++lir_count_;
                DemoteBottom();
            } else {
                MoveToTop(x);
                q_.MoveToBack(x);
            }
            return false;
        }

        if (resident_count_ == pool_) Evict();
        resident_[x] = true;
        ++resident_count_;
        if (lir_count_ < lir_limit_) {
            MoveToTop(x);
            lir_[x] = true;
            ++lir_count_;
        } else if (in_s_[x]) {
            MoveToTop(x);
            lir_[x] = true;
            ++lir_count_;
            DemoteBottom();
        } else {
            MoveToTop(x);
            q_.PushBack(x);
        }
        return true;
    }

    void MoveToTop(uint32_t x) {
        if (in_s_[x]) s_.Remove(x);
        s_.PushBack(x);
        in_s_[x] = true;
    }
    void RemoveFromQ(uint32_t x) {
        q_.Remove(x);
    }
    // Drops non-LIR entries from the bottom of S until an LIR page is there.
    void Prune() {
        while (!s_.Empty() && !lir_[s_.Front()]) in_s_[s_.PopFront()] = false;
    }
    // The LIR page at the bottom of S becomes a resident HIR page at the end of Q.
    void DemoteBottom() {
        const uint32_t bottom = s_.PopFront();
        assert(lir_[bottom]);
        in_s_[bottom] = false;
        lir_[bottom] = false;
        --lir_count_;
        q_.PushBack(bottom);
        Prune();
    }
    // Evicts the front of Q; it stays in S if it is there.
    void Evict() {
        if (q_.Empty()) throw std::logic_error("lirs: no resident HIR page to evict");
        const uint32_t victim = q_.Front();
        RemoveFromQ(victim);
        resident_[victim] = false;
        --resident_count_;
        ++evictions_;
    }

    size_t pool_;
    size_t lir_limit_;
    bool every_fetch_;
    detail::VisitTracker visits_;
    detail::PageLists s_;  // front is the bottom of the stack
    detail::PageLists q_;  // front is the next victim
    std::vector<bool> resident_, lir_, in_s_;
    size_t resident_count_{0};
    size_t lir_count_{0};
};

[[nodiscard]] inline V2Weights LoadV2Weights(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);

    const auto fail = [&](const std::string& why) { throw std::runtime_error(path + ": " + why); };
    const auto number = [&]() {
        std::string token;
        if (!(in >> token)) fail("file ends early");
        char* end = nullptr;
        const float value = std::strtof(token.c_str(), &end);
        if (end == token.c_str() || *end != '\0' || !std::isfinite(value)) fail("bad number '" + token + "'");
        return value;
    };
    const auto count = [&]() {
        const float v = number();
        if (v < 0 || v != std::floor(v)) fail("bad count");
        return static_cast<size_t>(v);
    };
    const auto vector_of = [&](size_t n) {
        std::vector<float> v(n);
        for (auto& x : v) x = number();
        return v;
    };

    V2Weights w;
    bool has[8] = {false, false, false, false, false, false, false, false};
    std::string tag;
    while (in >> tag) {
        if (tag == "mean" || tag == "std") {
            if (count() != 4) fail(tag + " must have 4 values");
            auto& dst = tag == "mean" ? w.mean : w.std;
            for (auto& x : dst) x = number();
            if (tag == "std" && std::any_of(dst.begin(), dst.end(), [](float x) { return x <= 0; })) fail("std values must be above 0");
            has[tag == "mean" ? 0 : 1] = true;

        } else if (tag == "w1" || tag == "w2" || tag == "w3") {
            const size_t out = count(), inn = count();
            if (tag == "w1") {
                if (inn != 4) fail("w1 must have 4 inputs");
                w.h1 = out;
                w.w1 = vector_of(out * inn);
                has[2] = true;
            } else if (tag == "w2") {
                if (inn != w.h1 || !has[2]) fail("w2 does not match w1");
                w.h2 = out;
                w.w2 = vector_of(out * inn);
                has[4] = true;
            } else {
                if (out != 1 || inn != w.h2 || !has[4]) fail("w3 must be 1 x h2");
                w.w3 = vector_of(inn);
                has[6] = true;
            }

        } else if (tag == "b1" || tag == "b2" || tag == "b3") {
            const size_t n = count();
            if (tag == "b1") {
                if (n != w.h1 || !has[2]) fail("b1 does not match w1");
                w.b1 = vector_of(n);
                has[3] = true;
            } else if (tag == "b2") {
                if (n != w.h2 || !has[4]) fail("b2 does not match w2");
                w.b2 = vector_of(n);
                has[5] = true;
            } else {
                if (n != 1) fail("b3 must have 1 value");
                w.b3 = number();
                has[7] = true;
            }

        } else if (tag == "ref") {
            const size_t n = count();
            w.refs.resize(n);
            for (auto& row : w.refs) {
                for (auto& x : row) x = number();
            }

        } else {
            fail("unknown line '" + tag + "'");
        }
    }
    for (const bool h : has) {
        if (!h) fail("a weight line is missing");
    }
    if (w.h1 == 0 || w.h2 == 0) fail("empty layer");
    if (w.refs.empty()) fail("no ref lines, so the scores can't be checked against TorchScript");
    return w;
}

// Scores n rows of 4 raw log features. scratch is working space the caller can reuse.
inline void V2Score(const V2Weights& w, const float* x, size_t n, float* out, std::vector<float>& scratch) {
    const size_t h1 = w.h1, h2 = w.h2;
    scratch.resize(n * (h1 + h2));
    float* a1 = scratch.data();
    float* a2 = a1 + n * h1;

    for (size_t i = 0; i < n; ++i) {
        float f[4];
        for (size_t k = 0; k < 4; ++k) f[k] = (x[i * 4 + k] - w.mean[k]) / w.std[k];
        float* row = a1 + i * h1;
        for (size_t j = 0; j < h1; ++j) {
            const float* wj = &w.w1[j * 4];
            const float s = w.b1[j] + wj[0] * f[0] + wj[1] * f[1] + wj[2] * f[2] + wj[3] * f[3];
            row[j] = s > 0 ? s : 0;
        }
    }

#if defined(__APPLE__)
    // a2 (n x h2) = a1 (n x h1) * w2^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(n), static_cast<int>(h2),
                static_cast<int>(h1), 1.0f, a1, static_cast<int>(h1), w.w2.data(), static_cast<int>(h1), 0.0f, a2,
                static_cast<int>(h2));
#else
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < h2; ++k) {
            float s = 0;
            for (size_t j = 0; j < h1; ++j) s += a1[i * h1 + j] * w.w2[k * h1 + j];
            a2[i * h2 + k] = s;
        }
    }
#endif

    for (size_t i = 0; i < n; ++i) {
        float o = w.b3;
        for (size_t k = 0; k < h2; ++k) {
            const float v = a2[i * h2 + k] + w.b2[k];
            o += w.w3[k] * (v > 0 ? v : 0);
        }
        out[i] = o;
    }
}

// Largest relative gap between our scores and TorchScript's on the exported check inputs.
// A score that isn't a finite number counts as an infinite gap, so it can never pass.
[[nodiscard]] inline double V2MaxRefError(const V2Weights& w) {
    std::vector<float> x, out(w.refs.size()), scratch;
    for (const auto& r : w.refs) x.insert(x.end(), r.begin(), r.begin() + 4);

    V2Score(w, x.data(), w.refs.size(), out.data(), scratch);
    double worst = 0;

    for (size_t i = 0; i < w.refs.size(); ++i) {
        const double ref = w.refs[i][4];
        const double gap = std::abs(out[i] - ref) / std::max(1.0, std::abs(ref));
        if (!std::isfinite(gap)) return std::numeric_limits<double>::infinity();
        worst = std::max(worst, gap);
    }
    return worst;
}

// The three features the engine gives its network, for an eviction at fetch t. Fetch
// numbers count from 0, and the engine's clock ticks once per fetch, so a page last
// fetched at t_last has been idle for t - t_last - 1 ticks. Without a second access the
// engine uses timestamp 0, which is clock_start + t ticks ago.
[[nodiscard]] inline std::array<float, 4> V2Features(uint64_t t, uint64_t t_last, bool has_second, uint64_t t_second, uint64_t fetches, uint64_t clock_start) {
    const uint64_t r1 = t - t_last - 1;
    const uint64_t r2 = has_second ? t - t_second - 1 : clock_start + t;
    const auto lg = [](uint64_t v) { return std::log(1.0f + static_cast<float>(v)); };
    return {lg(r1), lg(fetches), lg(r2), 0.0f};
}

// Evicts the page the network scores highest (the first one on a tie).
class V2Policy final : public Policy {
public:
    V2Policy(size_t pool, uint32_t page_limit, const V2Weights& w, uint64_t clock_start) : w_(w), clock_start_(clock_start), slot_of_(page_limit, kEmpty), slots_(pool) {}

    bool Access(size_t, const Run& run) override {
        const uint64_t t = run.first_fetch, len = run.length;
        const uint32_t slot_hit = slot_of_[run.page];
        if (slot_hit != kEmpty) {
            Slot& s = slots_[slot_hit];
            s.second = len >= 2 ? t + len - 2 : s.last;
            s.has_second = true;
            s.last = t + len - 1;
            s.fetches += len;
            return false;
        }
        size_t slot = filled_;
        if (filled_ < slots_.size()) {
            ++filled_;
        } else {
            slot = Victim(t);
            slot_of_[slots_[slot].page] = kEmpty;
            ++evictions_;
        }
        slots_[slot] = Slot{run.page, t + len - 1, t + len - 2, len >= 2, len};
        slot_of_[run.page] = static_cast<uint32_t>(slot);
        return true;
    }

private:
    static constexpr uint32_t kEmpty = UINT32_MAX;

    struct Slot {
        uint32_t page{0};
        uint64_t last{0}, second{0};
        bool has_second{false};
        uint64_t fetches{0};
    };

    [[nodiscard]] size_t Victim(uint64_t t) {
        const size_t n = slots_.size();
        features_.resize(n * 4);
        scores_.resize(n);
        for (size_t i = 0; i < n; ++i) {
            const Slot& s = slots_[i];
            const auto f = V2Features(t, s.last, s.has_second, s.second, s.fetches, clock_start_);
            std::copy(f.begin(), f.end(), features_.begin() + static_cast<std::ptrdiff_t>(i * 4));
        }
        V2Score(w_, features_.data(), n, scores_.data(), scratch_);
        return static_cast<size_t>(std::max_element(scores_.begin(), scores_.end()) - scores_.begin());
    }

    const V2Weights& w_;
    uint64_t clock_start_;
    std::vector<uint32_t> slot_of_;
    std::vector<Slot> slots_;
    size_t filled_{0};
    std::vector<float> features_, scores_, scratch_;
};

namespace detail {

// "lru2_visit" -> ("lru2", VISIT), "arc_full" -> ("arc", EVERY_FETCH), anything else plain.
inline std::pair<std::string, Mode> SplitMode(const std::string& name) {
    const auto ends_with = [&](const std::string& suffix) {
        return name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    if (ends_with("_visit")) return {name.substr(0, name.size() - 6), Mode::VISIT};
    if (ends_with("_full")) return {name.substr(0, name.size() - 5), Mode::EVERY_FETCH};
    return {name, Mode::PLAIN};
}

}  // namespace detail

[[nodiscard]] inline std::unique_ptr<Policy> MakePolicy(const std::string& name, size_t pool, const SimContext& ctx) {
    if (pool == 0) throw std::invalid_argument("pool must be at least 1");
    if (ctx.trace == nullptr) throw std::invalid_argument("context has no trace");
    const uint32_t pages = ctx.trace->page_limit;
    if (name == "lru") return std::make_unique<LruPolicy>(pool, pages);
    if (name == "clock") return std::make_unique<ClockPolicy>(pool, pages);
    if (name == "belady") {
        if (ctx.next_use == nullptr) throw std::invalid_argument("belady needs next_use");
        return std::make_unique<BeladyPolicy>(pool, pages, *ctx.next_use);
    }
    if (name == "v2") {
        if (ctx.v2 == nullptr) throw std::invalid_argument("v2 needs weights");
        return std::make_unique<V2Policy>(pool, pages, *ctx.v2, ctx.v2_clock_start);
    }
    // LRU-2, ARC and LIRS also come in visit (_visit) and every-fetch (_full) versions.
    const auto [base, mode] = detail::SplitMode(name);
    const uint32_t window = ctx.visit_window;
    if (base == "lru2") return std::make_unique<Lru2Policy>(pool, pages, mode, window);
    if (base == "arc") return std::make_unique<ArcPolicy>(pool, pages, mode, window);
    if (base == "lirs") return std::make_unique<LirsPolicy>(pool, pages, mode, window);
    throw std::invalid_argument("unknown policy " + name);
}

[[nodiscard]] inline const std::vector<std::string>& PolicyNames() {
    static const std::vector<std::string> names = {
        "lru", "clock", "lru2", "arc", "lirs", "lru2_visit", "arc_visit", "lirs_visit",
        "lru2_full", "arc_full", "lirs_full", "v2", "belady"
    };
    return names;
}

struct SimResult {
    std::vector<uint64_t> item_misses;
    uint64_t misses{0};
    uint64_t evictions{0};
};

// Replays every run in order. A run counts for the item that covers it; runs outside
// every item count only in the total.
[[nodiscard]] inline SimResult Simulate(Policy& policy, const Trace& trace, const std::vector<Item>& items) {
    SimResult result;
    result.item_misses.assign(items.size(), 0);
    size_t cur = 0;
    for (size_t r = 0; r < trace.runs.size(); ++r) {
        while (cur < items.size() && r >= items[cur].end_run) ++cur;
        if (!policy.Access(r, trace.runs[r])) continue;
        ++result.misses;
        if (cur < items.size() && r >= items[cur].first_run) ++result.item_misses[cur];
    }
    result.evictions = policy.Evictions();
    return result;
}

}  // namespace shilmandb::evsim
