#include <gtest/gtest.h>
#include "common/exception.hpp"
#include "engine/database.hpp"
#include "executor/parallel/memory_budget.hpp"
#include <unistd.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace shilmandb::parallel {

namespace {

constexpr size_t kMiB = size_t{1} << 20;

// Runs fn and returns what it threw, or "" if it didn't throw.
template <typename Fn>
std::string ErrorOf(Fn&& fn) {
    try {
        fn();
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

}  // namespace

// The budget on its own, plus a small columnar table for the database tests.
class ParallelMemoryBudgetTest : public ::testing::Test {
protected:
    std::string db_file_;
    std::string tbl_file_;
    std::unique_ptr<Database> db_;

    void TearDown() override {
        db_.reset();
        if (!db_file_.empty()) std::filesystem::remove(db_file_);
        if (!tbl_file_.empty()) std::filesystem::remove(tbl_file_);
    }

    // A temp path unique to this process.
    static std::string TempPath(const std::string& suffix) {
        return (std::filesystem::temp_directory_path() / ("shilmandb_memory_budget_test_" + std::to_string(::getpid()) + suffix)).string();
    }

    // A database holding columnar table t(k, v): 5,000 rows, 50 values of k.
    void OpenDatabase() {
        db_file_ = TempPath(".db");
        tbl_file_ = TempPath("_t.tbl");
        std::filesystem::remove(db_file_);
        {
            std::ofstream out(tbl_file_, std::ios::binary);
            for (int r = 0; r < 5000; ++r) out << r % 50 << '|' << r << '\n';
        }
        db_ = std::make_unique<Database>(db_file_);
        LoadOptions load;
        load.storage = StorageKind::COLUMNAR;
        load.threads = 2;
        db_->LoadTable("t", Schema({Column("k", ColumnType::Integer()), Column("v", ColumnType::Integer())}), tbl_file_, load);
    }

    // Runs the aggregate query in this mode, with these threads and this limit.
    QueryResult Run(ExecutionMode mode, unsigned threads, std::optional<size_t> limit, bool spill_everything = false) {
        QueryOptions options;
        options.mode = mode;
        options.threads = threads;
        options.memory_limit = limit;
        options.spill_everything = spill_everything;
        return db_->ExecuteSQL("SELECT k, SUM(v), COUNT(*) FROM t GROUP BY k", options);
    }
};

TEST_F(ParallelMemoryBudgetTest, FloorsAreHeldBack) {
    MemoryBudget b(4 * kWorkerMinimumBytes, 4);
    MemoryReservation hog(&b, 0);
    while (hog.Grow(kGrantBytes)) {}
    EXPECT_EQ(hog.Bytes(), kWorkerMinimumBytes) << "only its own floor: nothing is left beyond the floors";
    MemoryReservation other(&b, 3);
    EXPECT_TRUE(other.Grow(kWorkerMinimumBytes));  // its floor is still there
    EXPECT_FALSE(other.Grow(1));
    EXPECT_EQ(other.Bytes(), kWorkerMinimumBytes) << "a refused Grow changes nothing";
    EXPECT_LE(b.PeakBytes(), b.Limit());
    EXPECT_EQ(b.HeldBytes(), 2 * kWorkerMinimumBytes);
}

TEST_F(ParallelMemoryBudgetTest, GrantsBeyondTheFloorsAreShared) {
    MemoryBudget b(2 * kWorkerMinimumBytes + 3 * kGrantBytes, 2);
    MemoryReservation hog(&b, 0);
    while (hog.Grow(kGrantBytes)) {}
    EXPECT_EQ(hog.Bytes(), kWorkerMinimumBytes + 3 * kGrantBytes);
    MemoryReservation other(&b, 1);
    EXPECT_TRUE(other.Grow(kWorkerMinimumBytes));
    EXPECT_FALSE(other.Grow(1));
    EXPECT_EQ(b.PeakBytes(), b.Limit());

    // Shrinking the hog hands its shared grants back to everyone; it keeps one spare.
    hog.Shrink(hog.Bytes());
    EXPECT_EQ(hog.Bytes(), 0u);
    EXPECT_TRUE(other.Grow(3 * kGrantBytes));
    EXPECT_EQ(b.HeldBytes(), kWorkerMinimumBytes + 3 * kGrantBytes + kGrantBytes);
}

TEST_F(ParallelMemoryBudgetTest, GrowsInWholeGrants) {
    MemoryBudget b(64 * kMiB, 1);
    MemoryReservation r(&b, 0);
    EXPECT_TRUE(r.Grow(1));
    EXPECT_EQ(r.Bytes(), 1u);
    EXPECT_EQ(b.HeldBytes(), kGrantBytes);
    EXPECT_TRUE(r.Grow(kGrantBytes - 1));
    EXPECT_EQ(b.HeldBytes(), kGrantBytes) << "fits the grant already held";
    EXPECT_TRUE(r.Grow(1));
    EXPECT_EQ(b.HeldBytes(), 2 * kGrantBytes);

    // A request bigger than a grant rounds up to whole grants.
    EXPECT_TRUE(r.Grow(3 * kGrantBytes));
    EXPECT_EQ(r.Bytes(), 4 * kGrantBytes + 1);
    EXPECT_EQ(b.HeldBytes(), 5 * kGrantBytes);

    // Too big for the whole budget: refused, nothing changes.
    EXPECT_FALSE(r.Grow(std::numeric_limits<size_t>::max()));
    EXPECT_FALSE(r.Grow(64 * kMiB));
    EXPECT_EQ(r.Bytes(), 4 * kGrantBytes + 1);
    EXPECT_EQ(b.HeldBytes(), 5 * kGrantBytes);

    // Shrinking keeps the grants still needed plus one spare.
    r.Shrink(2 * kGrantBytes);
    EXPECT_EQ(b.HeldBytes(), 4 * kGrantBytes);
    r.Shrink(r.Bytes());
    EXPECT_EQ(b.HeldBytes(), kGrantBytes);
    r = MemoryReservation();
    EXPECT_EQ(b.HeldBytes(), 0u) << "destruction returns the spare too";
    EXPECT_EQ(b.PeakBytes(), kWorkerMinimumBytes + 3 * kGrantBytes) << "floors plus the most granted beyond them";
}

TEST_F(ParallelMemoryBudgetTest, SpareGrantStopsChurnOnABoundary) {
    MemoryBudget b(64 * kMiB, 1);
    MemoryReservation r(&b, 0);
    ASSERT_TRUE(r.Grow(3 * kGrantBytes));
    ASSERT_TRUE(r.Grow(1));
    const size_t held = b.HeldBytes();
    EXPECT_EQ(held, 4 * kGrantBytes);
    for (int i = 0; i < 100; ++i) {
        r.Shrink(1);
        EXPECT_EQ(b.HeldBytes(), held) << "round " << i;
        ASSERT_TRUE(r.Grow(1));
        EXPECT_EQ(b.HeldBytes(), held) << "round " << i;
    }
    // More than one grant of slack goes back, down to one spare.
    r.Shrink(2 * kGrantBytes + 1);
    EXPECT_EQ(b.HeldBytes(), 2 * kGrantBytes);
    r = MemoryReservation();
    EXPECT_EQ(b.HeldBytes(), 0u);
}

TEST_F(ParallelMemoryBudgetTest, LeaveFloorMovesEverythingWhenThereIsRoom) {
    MemoryBudget b(2 * kWorkerMinimumBytes + 40 * kMiB, 2);
    {
        MemoryReservation kept(&b, 0);
        ASSERT_TRUE(kept.Grow(6 * kMiB));
        EXPECT_EQ(b.HeldBytes(), kWorkerMinimumBytes) << "all on worker 0's floor";
        EXPECT_EQ(kept.LeaveFloor(), 0u);
        EXPECT_EQ(b.HeldBytes(), kWorkerMinimumBytes) << "moved, not freed";
        EXPECT_EQ(kept.Bytes(), 6 * kMiB);

        // The next breaker on worker 0 gets its whole floor.
        MemoryReservation next(&b, 0);
        EXPECT_TRUE(next.Grow(kWorkerMinimumBytes));
        EXPECT_EQ(b.HeldBytes(), 2 * kWorkerMinimumBytes);

        // Shrinking and growing after the move keep the counts right.
        kept.Shrink(kept.Bytes());
        EXPECT_EQ(b.HeldBytes(), kWorkerMinimumBytes + kGrantBytes);
        ASSERT_TRUE(kept.Grow(3 * kGrantBytes));
        EXPECT_EQ(kept.LeaveFloor(), 0u) << "worker 0's floor is taken, so these came from the shared pool";
        EXPECT_LE(b.HeldBytes(), b.Limit());
    }
    EXPECT_EQ(b.HeldBytes(), 0u);
    EXPECT_LE(b.PeakBytes(), b.Limit());

    // Nothing to move without a budget.
    MemoryReservation none;
    ASSERT_TRUE(none.Grow(kMiB));
    EXPECT_EQ(none.LeaveFloor(), 0u);
    MemoryReservation null_budget(nullptr, 3);
    EXPECT_EQ(null_budget.LeaveFloor(), 0u);
}

TEST_F(ParallelMemoryBudgetTest, LeaveFloorMovesWhatFitsWhenSharedIsNearlyFull) {
    // One grant of room beyond the floors.
    MemoryBudget b(2 * kWorkerMinimumBytes + kGrantBytes, 2);
    {
        MemoryReservation kept(&b, 0);
        ASSERT_TRUE(kept.Grow(kWorkerMinimumBytes));
        const size_t held = b.HeldBytes();
        EXPECT_EQ(kept.LeaveFloor(), kWorkerMinimumBytes - kGrantBytes) << "one grant fits";
        EXPECT_EQ(b.HeldBytes(), held);
        EXPECT_LE(b.HeldBytes(), b.Limit());
        EXPECT_EQ(kept.LeaveFloor(), kWorkerMinimumBytes - kGrantBytes) << "no room left";

        // Worker 0 has one grant of floor back; the shared pool is full.
        MemoryReservation next(&b, 0);
        EXPECT_TRUE(next.Grow(kGrantBytes));
        EXPECT_FALSE(next.Grow(1));
        EXPECT_EQ(b.HeldBytes(), b.Limit() - kWorkerMinimumBytes);
        EXPECT_EQ(b.PeakBytes(), b.Limit());
    }
    EXPECT_EQ(b.HeldBytes(), 0u);
}

TEST_F(ParallelMemoryBudgetTest, SpillShrinkLeaveFloorLoopFreesTheFloor) {
    // One grant of room beyond the floors: LeaveFloor can't move everything.
    MemoryBudget b(2 * kWorkerMinimumBytes + kGrantBytes, 2);
    {
        MemoryReservation kept(&b, 0);
        ASSERT_TRUE(kept.Grow(kWorkerMinimumBytes));
        size_t left = kept.LeaveFloor();
        EXPECT_EQ(left, kGrantBytes);
        int rounds = 0;
        while (left > 0 && rounds++ < 10) {
            kept.Shrink(left);  // the caller spilled this much
            left = kept.LeaveFloor();
        }
        EXPECT_EQ(left, 0u);
        EXPECT_EQ(kept.Bytes(), kGrantBytes) << "what moved to the shared pool stays in memory";
        EXPECT_LE(b.HeldBytes(), b.Limit());

        // The next breaker on worker 0 gets its whole floor.
        MemoryReservation next(&b, 0);
        EXPECT_TRUE(next.Grow(kWorkerMinimumBytes));
    }
    EXPECT_EQ(b.HeldBytes(), 0u);

    // Unused floor slack goes back before anything moves.
    {
        MemoryReservation kept(&b, 0);
        ASSERT_TRUE(kept.Grow(kWorkerMinimumBytes));
        kept.Shrink(kGrantBytes);  // a spare grant of floor kept
        EXPECT_EQ(b.HeldBytes(), kWorkerMinimumBytes);
        EXPECT_EQ(kept.LeaveFloor(), 0u);
        EXPECT_EQ(b.HeldBytes(), kGrantBytes);

        MemoryReservation next(&b, 0);
        EXPECT_TRUE(next.Grow(kWorkerMinimumBytes));
    }
    EXPECT_EQ(b.HeldBytes(), 0u);
}

namespace {

// Runs the caller's loop on 6 MiB (not whole grants) with this much shared room.
void LeaveFloorOnPartialGrants(size_t shared_room, size_t first_left, size_t stays) {
    MemoryBudget b(2 * kWorkerMinimumBytes + shared_room, 2);
    {
        MemoryReservation kept(&b, 0);
        ASSERT_TRUE(kept.Grow(6 * kMiB));
        size_t left = kept.LeaveFloor();
        EXPECT_EQ(left, first_left) << "exactly the floor bytes in use";
        int rounds = 0;
        while (left > 0 && rounds++ < 10) {
            ASSERT_LE(left, kept.Bytes()) << "never more than the reservation uses";
            kept.Shrink(left);  // the caller spilled this much
            left = kept.LeaveFloor();
        }
        EXPECT_EQ(left, 0u);
        EXPECT_EQ(rounds, 1) << "one spill is enough";
        EXPECT_EQ(kept.Bytes(), stays);
        EXPECT_LE(b.HeldBytes(), b.Limit());

        MemoryReservation next(&b, 0);
        EXPECT_TRUE(next.Grow(kWorkerMinimumBytes)) << "the next breaker on worker 0 gets its whole floor";
    }
    EXPECT_EQ(b.HeldBytes(), 0u);
}

}  // namespace

TEST_F(ParallelMemoryBudgetTest, LeaveFloorOnPartialGrantsWithNoSharedRoom) {
    LeaveFloorOnPartialGrants(0, 6 * kMiB, 0);
}

TEST_F(ParallelMemoryBudgetTest, LeaveFloorOnPartialGrantsWithOneSharedGrant) {
    LeaveFloorOnPartialGrants(kGrantBytes, 2 * kMiB, kGrantBytes);
}

TEST_F(ParallelMemoryBudgetTest, TooSmallForThreadsThrows) {
    EXPECT_THROW(MemoryBudget(18 * kWorkerMinimumBytes - 1, 18), DatabaseException);
    EXPECT_EQ(ErrorOf([] { MemoryBudget b(18 * kWorkerMinimumBytes - 1, 18); }),
              "memory limit 150994943 is too small for 18 threads (needs at least 150994944)");
    MemoryBudget exact(18 * kWorkerMinimumBytes, 18);
    EXPECT_EQ(exact.Limit(), 18 * kWorkerMinimumBytes);
    EXPECT_EQ(exact.HeldBytes(), 0u);
    EXPECT_EQ(exact.PeakBytes(), 18 * kWorkerMinimumBytes) << "the floors count from the start";
}

TEST_F(ParallelMemoryBudgetTest, DestructionReturnsEverything) {
    MemoryBudget b(64 * kMiB, 2);
    {
        MemoryReservation a(&b, 0);
        MemoryReservation c(&b, 1);
        ASSERT_TRUE(a.Grow(20 * kMiB));
        ASSERT_TRUE(c.Grow(5 * kMiB));
        EXPECT_GT(b.HeldBytes(), 0u);
    }
    EXPECT_EQ(b.HeldBytes(), 0u);

    // Also when a throw leaves the scope.
    try {
        MemoryReservation a(&b, 0);
        ASSERT_TRUE(a.Grow(30 * kMiB));
        throw DatabaseException("unwind");
    } catch (const DatabaseException&) {
    }
    EXPECT_EQ(b.HeldBytes(), 0u);
}

TEST_F(ParallelMemoryBudgetTest, MovesHandOverTheBytes) {
    MemoryBudget b(64 * kMiB, 1);
    MemoryReservation a(&b, 0);
    ASSERT_TRUE(a.Grow(10 * kMiB));
    const size_t held = b.HeldBytes();

    MemoryReservation moved(std::move(a));
    EXPECT_EQ(moved.Bytes(), 10 * kMiB);
    EXPECT_EQ(a.Bytes(), 0u);  // NOLINT: a moved-from reservation holds nothing
    EXPECT_EQ(b.HeldBytes(), held);

    // Assigning over a reservation returns what it held first.
    MemoryReservation other(&b, 0);
    ASSERT_TRUE(other.Grow(20 * kMiB));
    other = std::move(moved);
    EXPECT_EQ(other.Bytes(), 10 * kMiB);
    EXPECT_EQ(b.HeldBytes(), held);
    other = MemoryReservation();
    EXPECT_EQ(b.HeldBytes(), 0u);
}

TEST_F(ParallelMemoryBudgetTest, NoBudgetAlwaysGrows) {
    MemoryReservation r;
    EXPECT_TRUE(r.Grow(std::numeric_limits<size_t>::max() / 2));
    EXPECT_TRUE(r.Grow(5));
    r.Shrink(5);
    EXPECT_EQ(r.Bytes(), std::numeric_limits<size_t>::max() / 2);
}

TEST_F(ParallelMemoryBudgetTest, ManyThreadsNeverPassTheCeiling) {
    // 16 threads, two per worker index, so workers' floors are shared.
    constexpr unsigned kThreads = 16;
    constexpr unsigned kWorkers = 8;
    MemoryBudget b(kWorkers * kWorkerMinimumBytes + 40 * kMiB, kWorkers);
    std::atomic<bool> done{false};
    std::atomic<bool> over{false};
    std::atomic<uint64_t> refused{0};

    std::thread watcher([&] {
        while (!done.load()) {
            if (b.HeldBytes() > b.Limit()) over.store(true);
        }
    });
    std::vector<std::thread> threads;
    for (unsigned t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            std::mt19937_64 rng(t + 1);
            MemoryReservation r[2] = {MemoryReservation(&b, t % kWorkers), MemoryReservation(&b, t % kWorkers)};
            for (int i = 0; i < 20000; ++i) {
                MemoryReservation& one = r[rng() % 2];
                if (rng() % 20 == 0) {
                    (void)one.LeaveFloor();
                } else if (rng() % 3 != 0) {
                    if (!one.Grow(rng() % (3 * kMiB))) refused.fetch_add(1);
                } else if (one.Bytes() > 0) {
                    one.Shrink(rng() % (one.Bytes() + 1));
                }
                if (b.HeldBytes() > b.Limit()) over.store(true);
            }
        });
    }
    for (auto& t : threads) t.join();
    done.store(true);
    watcher.join();

    EXPECT_FALSE(over.load()) << "held bytes passed the limit";
    EXPECT_GT(refused.load(), 0u) << "the budget was never full, so the test proved little";
    EXPECT_LE(b.PeakBytes(), b.Limit());
    EXPECT_EQ(b.HeldBytes(), 0u);
}

TEST_F(ParallelMemoryBudgetTest, DefaultsComeFromTheMachineAndTheFile) {
    EXPECT_GT(DefaultMemoryLimit(), 0u);
    OpenDatabase();
    EXPECT_EQ(db_->GetDefaultMemoryLimit(), DefaultMemoryLimit());
    EXPECT_EQ(db_->GetTempDirectory(), db_file_ + ".tmp");
    db_->SetDefaultMemoryLimit(64 * kMiB);
    EXPECT_EQ(db_->GetDefaultMemoryLimit(), 64 * kMiB);
    db_->SetTempDirectory("/some/where");
    EXPECT_EQ(db_->GetTempDirectory(), "/some/where");
}

TEST_F(ParallelMemoryBudgetTest, StatsReportTheLimit) {
    OpenDatabase();
    for (unsigned threads : {2u, 1u}) {
        const QueryResult r = Run(ExecutionMode::PARALLEL, threads, 256 * kMiB);
        EXPECT_EQ(r.tuples.size(), 50u);
        const ParallelQueryStats& s = db_->LastParallelStats();
        EXPECT_EQ(s.threads, threads);
        EXPECT_EQ(s.memory_limit, 256 * kMiB);
        EXPECT_GT(s.peak_tracked_bytes, 0u);
        EXPECT_GE(s.peak_tracked_bytes, threads * kWorkerMinimumBytes) << "the floors count";
        EXPECT_LE(s.peak_tracked_bytes, s.memory_limit);
        EXPECT_EQ(s.spill_bytes_written, 0u);
        EXPECT_EQ(s.spill_bytes_read, 0u);
        EXPECT_EQ(s.spill_files, 0u);
        EXPECT_EQ(s.spill_partitions, 0u);
        EXPECT_EQ(s.spill_time.count(), 0);
    }

    // Without a limit the query takes the database's default.
    (void)Run(ExecutionMode::PARALLEL, 2, std::nullopt);
    EXPECT_EQ(db_->LastParallelStats().memory_limit, db_->GetDefaultMemoryLimit());
    db_->SetDefaultMemoryLimit(64 * kMiB);
    (void)Run(ExecutionMode::PARALLEL, 2, std::nullopt);
    EXPECT_EQ(db_->LastParallelStats().memory_limit, 64 * kMiB);

    // Tuple and vectorized modes ignore the limit and spill_everything, and leave the stats alone.
    for (ExecutionMode mode : {ExecutionMode::TUPLE, ExecutionMode::VECTORIZED}) {
        const QueryResult want = Run(mode, 0, size_t{1}, true);
        EXPECT_EQ(want.tuples.size(), 50u);
        EXPECT_EQ(db_->LastParallelStats().memory_limit, 64 * kMiB);
    }
}

TEST_F(ParallelMemoryBudgetTest, TooSmallLimitFailsTheQueryOnly) {
    OpenDatabase();
    (void)Run(ExecutionMode::PARALLEL, 2, 256 * kMiB);
    const ParallelQueryStats before = db_->LastParallelStats();

    const size_t limit = 2 * kWorkerMinimumBytes - 1;
    EXPECT_EQ(ErrorOf([&] { (void)Run(ExecutionMode::PARALLEL, 2, limit); }),
              "memory limit 16777215 is too small for 2 threads (needs at least 16777216)");
    // The error keeps the last stats.
    EXPECT_EQ(db_->LastParallelStats().memory_limit, before.memory_limit);
    EXPECT_EQ(db_->LastParallelStats().morsels, before.morsels);
    EXPECT_EQ(db_->LastParallelStats().peak_tracked_bytes, before.peak_tracked_bytes);

    // One thread fits the same limit, and the database still works.
    const QueryResult r = Run(ExecutionMode::PARALLEL, 1, limit);
    EXPECT_EQ(r.tuples.size(), 50u);
    EXPECT_EQ(db_->LastParallelStats().memory_limit, limit);
}

#ifndef NDEBUG
TEST(ParallelMemoryBudgetDeathTest, ShrinkingMoreThanHeldAsserts) {
    MemoryBudget b(64 * kMiB, 1);
    MemoryReservation r(&b, 0);
    ASSERT_TRUE(r.Grow(10));
    EXPECT_DEATH(r.Shrink(11), "Shrink past what the reservation holds");
}
#endif

}  // namespace shilmandb::parallel
