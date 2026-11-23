#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace shilmandb::parallel {

// Memory beyond the floors is handed out in steps of this size.
inline constexpr size_t kGrantBytes = size_t{4} << 20;

// Memory held back for each worker, so one worker can't starve the rest.
inline constexpr size_t kWorkerMinimumBytes = size_t{8} << 20;

static_assert(kWorkerMinimumBytes % kGrantBytes == 0, "a floor is whole grants");

// Spill totals for a query, added to by every worker.
struct SpillCounters {
    std::atomic<uint64_t> bytes_written{0}, bytes_read{0}, files{0}, partitions{0};
    std::atomic<int64_t> nanos{0};  // time writing and reading, summed over workers
};

// One query's operator memory: a ceiling, a minimum held back per worker, grants beyond it.
// Safe to use from any thread; held bytes never pass the limit.
class MemoryBudget {
public:
    MemoryBudget(size_t limit_bytes, unsigned workers);  // throws the too-small error

    MemoryBudget(const MemoryBudget&) = delete;
    MemoryBudget& operator=(const MemoryBudget&) = delete;

    [[nodiscard]] size_t Limit() const { return limit_; }

    // Bytes granted to live reservations right now, floors and beyond; 0 once all are gone.
    [[nodiscard]] size_t HeldBytes() const;

    // Every floor plus the most granted beyond them, so it starts at the floors.
    // Bytes LeaveFloor moves to the shared pool count again here.
    [[nodiscard]] size_t PeakBytes() const;  // floors included

private:
    friend class MemoryReservation;

    // Takes `bytes` (whole grants) for a worker, floor first; all or nothing.
    [[nodiscard]] bool Take(unsigned worker, size_t bytes, size_t& from_floor, size_t& from_shared);

    // Gives back what Take handed out.
    void Give(unsigned worker, size_t from_floor, size_t from_shared);

    // Moves up to `bytes` of a worker's floor use to the shared pool; returns it.
    [[nodiscard]] size_t MoveToShared(unsigned worker, size_t bytes);

    size_t limit_;
    unsigned workers_;
    size_t shared_capacity_;                            // the limit minus every floor
    std::unique_ptr<std::atomic<size_t>[]> floor_used_;  // per worker, shared by its reservations
    std::atomic<size_t> shared_used_{0};
    std::atomic<size_t> shared_peak_{0};
};

// What one worker holds for one breaker; returns everything when destroyed. Movable, not copyable.
// One reservation is used by one thread at a time.
class MemoryReservation {
public:
    MemoryReservation() = default;  // no budget: every Grow succeeds
    MemoryReservation(MemoryBudget* budget, unsigned worker);
    ~MemoryReservation();

    MemoryReservation(MemoryReservation&& other) noexcept;
    MemoryReservation& operator=(MemoryReservation&& other) noexcept;
    MemoryReservation(const MemoryReservation&) = delete;
    MemoryReservation& operator=(const MemoryReservation&) = delete;

    [[nodiscard]] bool Grow(size_t bytes);  // false: over the budget, so spill

    // Stops using bytes; keeps a spare grant so the next Grow needn't touch the budget.
    void Shrink(size_t bytes);

    [[nodiscard]] size_t Bytes() const { return bytes_; }

    // At Finalize, for state outliving its pipeline: frees slack, moves floor bytes to shared.
    // Returns floor bytes in use; loop spill, Shrink(spilled), LeaveFloor() until it returns 0.
    [[nodiscard]] size_t LeaveFloor();

private:
    // Hands back whole grants this reservation no longer needs, past one spare if asked.
    // After LeaveFloor, floor grants go first and the spare is only ever a shared grant.
    void ReleaseSlack(bool keep_spare);

    MemoryBudget* budget_{nullptr};
    unsigned worker_{0};
    size_t bytes_{0};        // what the owner says it uses
    size_t from_floor_{0};   // grants taken from the worker's floor
    size_t from_shared_{0};  // grants taken beyond the floors
    bool left_floor_{false}; // LeaveFloor was called: give floor grants back first
};

[[nodiscard]] size_t DefaultMemoryLimit();  // physical memory / 4

}  // namespace shilmandb::parallel
