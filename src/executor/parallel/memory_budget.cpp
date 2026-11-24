#include "executor/parallel/memory_budget.hpp"
#include "common/exception.hpp"
#include <algorithm>
#include <cassert>
#include <limits>
#include <string>
#include <utility>

#ifdef __APPLE__
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

namespace shilmandb::parallel {

namespace {

constexpr auto kRelaxed = std::memory_order_relaxed;

// Raises peak to at least value.
void RaiseTo(std::atomic<size_t>& peak, size_t value) {
    size_t cur = peak.load(kRelaxed);
    while (cur < value && !peak.compare_exchange_weak(cur, value, kRelaxed)) {}
}

}  // namespace

// Refuses a limit that can't hold every worker's floor.
MemoryBudget::MemoryBudget(size_t limit_bytes, unsigned workers) : limit_(limit_bytes), workers_(workers), floor_used_(std::make_unique<std::atomic<size_t>[]>(workers)) {
    const size_t floors = size_t{workers} * kWorkerMinimumBytes;
    if (limit_bytes < floors) {
        throw DatabaseException("memory limit " + std::to_string(limit_bytes) + " is too small for " + std::to_string(workers) + " threads (needs at least " + std::to_string(floors) + ")");
    }
    shared_capacity_ = limit_bytes - floors;
    for (unsigned w = 0; w < workers; ++w) floor_used_[w].store(0, kRelaxed);
}

// Every reservation must go before its budget; Debug checks that all grants came back.
MemoryBudget::~MemoryBudget() {
    assert(HeldBytes() == 0 && "a reservation outlived its memory budget");
}

// Each part stays within its share of the limit, so the sum can't pass it.
size_t MemoryBudget::HeldBytes() const {
    size_t held = shared_used_.load(kRelaxed);
    for (unsigned w = 0; w < workers_; ++w) held += floor_used_[w].load(kRelaxed);
    return held;
}

// The floors are counted whether or not they're used.
size_t MemoryBudget::PeakBytes() const {
    return size_t{workers_} * kWorkerMinimumBytes + shared_peak_.load(kRelaxed);
}

// The floor's free grants first, the rest from the shared counter; undone if that's full.
bool MemoryBudget::Take(unsigned worker, size_t bytes, size_t& from_floor, size_t& from_shared) {
    std::atomic<size_t>& floor = floor_used_[worker];
    size_t used = floor.load(kRelaxed);
    size_t take = 0;
    do {
        take = std::min(bytes, kWorkerMinimumBytes - used);
    } while (take > 0 && !floor.compare_exchange_weak(used, used + take, kRelaxed));

    const size_t rest = bytes - take;
    if (rest > 0) {
        size_t shared = shared_used_.load(kRelaxed);
        do {
            if (rest > shared_capacity_ - shared) {
                if (take > 0) floor.fetch_sub(take, kRelaxed);
                return false;
            }
        } while (!shared_used_.compare_exchange_weak(shared, shared + rest, kRelaxed));
        RaiseTo(shared_peak_, shared + rest);
    }
    from_floor = take;
    from_shared = rest;
    return true;
}

// Lowers the counters by what a reservation hands back.
void MemoryBudget::Give(unsigned worker, size_t from_floor, size_t from_shared) {
    if (from_floor > 0) floor_used_[worker].fetch_sub(from_floor, kRelaxed);
    if (from_shared > 0) shared_used_.fetch_sub(from_shared, kRelaxed);
}

// Whole grants, as room allows. The shared count goes up before the floor's goes down,
// so another reservation can't take the floor bytes while the move can still fail.
size_t MemoryBudget::MoveToShared(unsigned worker, size_t bytes) {
    size_t shared = shared_used_.load(kRelaxed);
    size_t move = 0;
    do {
        const size_t room = (shared_capacity_ - shared) / kGrantBytes * kGrantBytes;
        move = std::min(bytes, room);
    } while (move > 0 && !shared_used_.compare_exchange_weak(shared, shared + move, kRelaxed));

    if (move == 0) return 0;
    RaiseTo(shared_peak_, shared + move);
    floor_used_[worker].fetch_sub(move, kRelaxed);
    return move;
}

// The worker index picks the floor this reservation draws on.
MemoryReservation::MemoryReservation(MemoryBudget* budget, unsigned worker) : budget_(budget), worker_(worker) {
    if (budget_ != nullptr && worker >= budget_->workers_) {
        throw DatabaseException("worker " + std::to_string(worker) + " is past the budget's " + std::to_string(budget_->workers_) + " workers");
    }
}

// Gives back every grant, spare included.
MemoryReservation::~MemoryReservation() {
    bytes_ = 0;
    ReleaseSlack(false);
}

// Takes over other's grants; other is left holding nothing.
MemoryReservation::MemoryReservation(MemoryReservation&& other) noexcept : budget_(std::exchange(other.budget_, nullptr)),
      worker_(std::exchange(other.worker_, 0)), bytes_(std::exchange(other.bytes_, 0)), from_floor_(std::exchange(other.from_floor_, 0)), from_shared_(std::exchange(other.from_shared_, 0)), left_floor_(std::exchange(other.left_floor_, false)) {}

// Returns what this held, then takes over other's bytes.
MemoryReservation& MemoryReservation::operator=(MemoryReservation&& other) noexcept {
    if (this == &other) return *this;
    bytes_ = 0;
    ReleaseSlack(false);

    budget_ = std::exchange(other.budget_, nullptr);
    worker_ = std::exchange(other.worker_, 0);

    bytes_ = std::exchange(other.bytes_, 0);
    from_floor_ = std::exchange(other.from_floor_, 0);
    from_shared_ = std::exchange(other.from_shared_, 0);
    left_floor_ = std::exchange(other.left_floor_, false);
    return *this;
}

// Fits in grants already held without touching the budget; else takes whole grants.
bool MemoryReservation::Grow(size_t bytes) {
    if (budget_ == nullptr) {
        bytes_ += bytes;
        return true;
    }
    const size_t limit = budget_->Limit();
    if (bytes > limit - bytes_) return false;
    const size_t total = bytes_ + bytes;
    const size_t held = from_floor_ + from_shared_;
    if (total <= held) {
        bytes_ = total;
        return true;
    }

    const size_t grants = total / kGrantBytes + (total % kGrantBytes != 0 ? 1 : 0);
    if (grants > std::numeric_limits<size_t>::max() / kGrantBytes) return false;
    size_t floor = 0;
    size_t shared = 0;
    if (!budget_->Take(worker_, grants * kGrantBytes - held, floor, shared)) return false;
    from_floor_ += floor;
    from_shared_ += shared;
    bytes_ = total;
    return true;
}

// Keeps a spare grant, so alternating Grow and Shrink on a boundary stays local.
// After LeaveFloor it frees floor grants first and keeps no spare on the floor.
void MemoryReservation::Shrink(size_t bytes) {
    assert(bytes <= bytes_ && "Shrink past what the reservation holds");
    bytes_ -= std::min(bytes, bytes_);
    ReleaseSlack(true);
}

// Frees unused grants, floor first, then moves floor grants in use to the shared pool.
// Its own split follows the counters, so later releases go to the right one.
size_t MemoryReservation::LeaveFloor() {
    if (budget_ == nullptr) return 0;
    left_floor_ = true;
    ReleaseSlack(false);
    if (from_floor_ == 0) return 0;
    const size_t moved = budget_->MoveToShared(worker_, from_floor_);
    from_floor_ -= moved;
    from_shared_ += moved;
    return bytes_ > from_shared_ ? bytes_ - from_shared_ : 0;
}

// Keeps the grants bytes_ still needs. Shared grants go back first, for others,
// except after LeaveFloor, when the floor must empty for the next pipeline.
void MemoryReservation::ReleaseSlack(bool keep_spare) {
    if (budget_ == nullptr) return;
    const size_t needed = (bytes_ / kGrantBytes + (bytes_ % kGrantBytes != 0 ? 1 : 0)) * kGrantBytes;
    size_t excess = from_floor_ + from_shared_ - needed;
    size_t floor = 0;
    size_t shared = 0;
    
    if (left_floor_) {
        floor = std::min(excess, from_floor_);
        shared = excess - floor;
        if (keep_spare) shared -= std::min(shared, kGrantBytes);
    } else {
        if (keep_spare) excess -= std::min(excess, kGrantBytes);
        shared = std::min(excess, from_shared_);
        floor = excess - shared;
    }
    if (floor == 0 && shared == 0) return;
    budget_->Give(worker_, floor, shared);
    from_floor_ -= floor;
    from_shared_ -= shared;
}

// A quarter of the machine's memory; 1 GiB if the system won't say.
size_t DefaultMemoryLimit() {
    uint64_t physical = 0;
#ifdef __APPLE__
    size_t size = sizeof(physical);
    if (sysctlbyname("hw.memsize", &physical, &size, nullptr, 0) != 0) physical = 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0) physical = static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
#endif
    if (physical == 0) return size_t{1} << 30;
    return static_cast<size_t>(physical / 4);
}

}  // namespace shilmandb::parallel
