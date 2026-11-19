#include <gtest/gtest.h>
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/eviction_policy_factory.hpp"
#include "common/exception.hpp"
#include "executor/parallel/pipeline.hpp"
#include "executor/parallel/worker_pool.hpp"
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace shilmandb::parallel {

namespace {

constexpr uint32_t kRows = 10;  // rows per fake batch

// Hands out `morsels` morsels of one batch each, all values 1.
// Records which thread started each morsel; may sleep or throw on request.
class CountingSource : public Source {
public:
    struct Local : LocalState {
        bool pending{false};
    };

    explicit CountingSource(size_t morsels) : morsels_(morsels), starts_(morsels) {}

    void Prepare(QueryContext& ctx) override {
        bpm_ = ctx.bpm;
        prepare_concurrent = bpm_ != nullptr && bpm_->IsConcurrent();
        caller = std::this_thread::get_id();
    }

    size_t MorselCount() const override { return morsels_; }

    std::unique_ptr<LocalState> MakeLocal() const override {
        if (bpm_ != nullptr && bpm_->IsConcurrent()) make_local_concurrent = true;
        return std::make_unique<Local>();
    }

    void StartMorsel(size_t morsel, LocalState& local) const override {
        starts_[morsel].fetch_add(1);
        started.fetch_add(1);
        if (bpm_ != nullptr && !bpm_->IsConcurrent()) worker_saw_off = true;
        if (std::this_thread::get_id() != caller) off_caller.fetch_add(1);
        if (sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        if (throw_at >= 0 && morsel == static_cast<size_t>(throw_at)) throw std::invalid_argument("bad date in morsel " + std::to_string(morsel));
        static_cast<Local&>(local).pending = true;
    }

    bool NextBatch(LocalState& local, Batch& out) const override {
        auto& l = static_cast<Local&>(local);
        if (!l.pending) return false;
        l.pending = false;
        ShapeBatch(out, {VType{}});
        out.Reset(kRows);
        int64_t* v = out.slots[0].I64();
        for (uint32_t i = 0; i < kRows; ++i) v[i] = 1;
        return true;
    }

    // How many times morsel m was started.
    [[nodiscard]] int Starts(size_t m) const { return starts_[m].load(); }

    // Settings, set before the run.
    int sleep_ms{0};
    long throw_at{-1};

    // Records; workers write them, so atomic.
    mutable std::atomic<int> started{0};
    mutable std::atomic<int> off_caller{0};
    mutable std::atomic<bool> worker_saw_off{false};
    mutable std::atomic<bool> make_local_concurrent{false};
    bool prepare_concurrent{false};
    std::thread::id caller;

private:
    size_t morsels_;
    mutable std::vector<std::atomic<int>> starts_;
    BufferPoolManager* bpm_{nullptr};
};

// Sums the rows per worker; locals are tagged in the order MakeLocal made them.
class TaggingSink : public Sink {
public:
    struct Local : LocalState {
        int tag{0};
        int64_t sum{0};
        bool slept{false};
    };

    explicit TaggingSink(BufferPoolManager* bpm) : bpm_(bpm) {}

    std::unique_ptr<LocalState> MakeLocal() const override {
        auto l = std::make_unique<Local>();
        l->tag = made.fetch_add(1);
        return l;
    }

    // Local 0 sleeps on its first batch, so its worker finishes last.
    void Consume(Batch& in, LocalState& local) const override {
        auto& l = static_cast<Local&>(local);
        if (l.tag == 0 && slow_first && !l.slept) {
            l.slept = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        const int64_t* v = in.slots[0].I64();
        for (uint32_t k = 0; k < in.Active(); ++k) l.sum += v[in.sel[k]];
    }

    void Combine(std::unique_ptr<LocalState> local) override {
        const auto& l = static_cast<const Local&>(*local);
        order.push_back(l.tag);
        total += l.sum;
        if (bpm_ != nullptr && bpm_->IsConcurrent()) combine_concurrent = true;
    }

    void Finalize(QueryContext&) override {
        ++finalizes;
        if (bpm_ != nullptr && bpm_->IsConcurrent()) finalize_concurrent = true;
    }

    bool slow_first{false};
    mutable std::atomic<int> made{0};
    std::vector<int> order;
    int64_t total{0};
    int finalizes{0};
    bool combine_concurrent{false};
    bool finalize_concurrent{false};

private:
    BufferPoolManager* bpm_;
};

// A real pool on a temp file, so the concurrent mode can be watched.
class ParallelWorkerPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        file_ = (std::filesystem::temp_directory_path() / ("shilmandb_worker_pool_test_" + std::to_string(::getpid()) + ".db")).string();
        std::filesystem::remove(file_);
        disk_ = std::make_unique<DiskManager>(file_);
        bpm_ = std::make_unique<BufferPoolManager>(16, disk_.get(), MakeEvictionPolicy(EvictionPolicyKind::LRU, 16));
    }

    void TearDown() override {
        bpm_.reset();
        disk_.reset();
        std::filesystem::remove(file_);
    }

    // A context on the real pool with this many threads and a worker pool.
    QueryContext Context(unsigned threads, WorkerPool* pool) { return QueryContext{bpm_.get(), nullptr, threads, pool}; }

    std::string file_;
    std::unique_ptr<DiskManager> disk_;
    std::unique_ptr<BufferPoolManager> bpm_;
};

}  // namespace

TEST_F(ParallelWorkerPoolTest, RunsEveryWorker) {
    WorkerPool pool(6);
    EXPECT_EQ(pool.Size(), 6u);
    for (unsigned workers = 1; workers <= pool.Size(); ++workers) {
        std::vector<std::atomic<int>> calls(pool.Size());
        std::vector<std::thread::id> ids(pool.Size());
        pool.Run(workers, [&](unsigned w) {
            calls[w].fetch_add(1);
            ids[w] = std::this_thread::get_id();
        });
        for (unsigned w = 0; w < pool.Size(); ++w) EXPECT_EQ(calls[w].load(), w < workers ? 1 : 0) << workers << " workers, worker " << w;
        EXPECT_EQ(ids[0], std::this_thread::get_id()) << "worker 0 is the caller";
        for (unsigned w = 1; w < workers; ++w) EXPECT_NE(ids[w], std::this_thread::get_id());
    }
    WorkerPool single(1);
    EXPECT_EQ(single.Size(), 1u);
    int ran = 0;
    single.Run(1, [&](unsigned w) { ran += static_cast<int>(w) + 1; });
    EXPECT_EQ(ran, 1);
    EXPECT_THROW(single.Run(2, [](unsigned) {}), DatabaseException);
}

TEST_F(ParallelWorkerPoolTest, RethrowsFirstException) {
    WorkerPool pool(4);
    // One worker throws a type that isn't DatabaseException; the rest finish first.
    std::vector<std::atomic<bool>> returned(4);
    try {
        pool.Run(4, [&](unsigned w) {
            if (w == 2) throw std::invalid_argument("stoi");
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            returned[w] = true;
        });
        FAIL() << "Run should rethrow";
    } catch (const std::invalid_argument& e) {
        EXPECT_STREQ(e.what(), "stoi");
    }
    for (unsigned w : {0u, 1u, 3u}) EXPECT_TRUE(returned[w].load()) << "worker " << w << " returned before the rethrow";

    // Worker 1 throws at once; the others throw much later and lose.
    try {
        pool.Run(4, [&](unsigned w) {
            if (w == 1) throw DatabaseException("first");
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            throw std::runtime_error("later");
        });
        FAIL() << "Run should rethrow";
    } catch (const DatabaseException& e) {
        EXPECT_STREQ(e.what(), "first");
    }

    // The error is cleared: the next run is clean.
    std::atomic<int> calls{0};
    EXPECT_NO_THROW(pool.Run(4, [&](unsigned) { calls.fetch_add(1); }));
    EXPECT_EQ(calls.load(), 4);
}

TEST_F(ParallelWorkerPoolTest, ReusableAcrossRuns) {
    WorkerPool pool(8);
    std::vector<int> counts(pool.Size(), 0);  // each worker writes only its own entry
    uint64_t expected = 0;
    for (int run = 0; run < 1000; ++run) {
        const unsigned workers = 1 + static_cast<unsigned>(run) % pool.Size();
        expected += workers;
        std::atomic<int> calls{0};
        pool.Run(workers, [&](unsigned w) {
            ++counts[w];
            calls.fetch_add(1);
        });
        ASSERT_EQ(calls.load(), static_cast<int>(workers)) << "run " << run;
    }
    uint64_t total = 0;
    for (int c : counts) total += static_cast<uint64_t>(c);
    EXPECT_EQ(total, expected);
    EXPECT_EQ(counts[0], 1000);
}

TEST_F(ParallelWorkerPoolTest, PipelineCombinesInWorkerOrder) {
    WorkerPool pool(4);
    CountingSource source(64);
    TaggingSink sink(bpm_.get());
    sink.slow_first = true;
    Pipeline p{&source, {}, &sink, {}};
    QueryContext ctx = Context(4, &pool);
    RunPipeline(p, ctx);

    for (size_t m = 0; m < 64; ++m) EXPECT_EQ(source.Starts(m), 1) << "morsel " << m;
    EXPECT_EQ(sink.total, 64 * static_cast<int64_t>(kRows));
    EXPECT_EQ(sink.order, (std::vector<int>{0, 1, 2, 3}));
    EXPECT_EQ(sink.finalizes, 1);
    EXPECT_GT(source.off_caller.load(), 0) << "background workers took morsels";
    EXPECT_EQ(ctx.morsels, 64u);
}

TEST_F(ParallelWorkerPoolTest, ConcurrentModeOnlyWhileWorkersRun) {
    WorkerPool pool(4);
    CountingSource source(32);
    TaggingSink sink(bpm_.get());
    Pipeline p{&source, {}, &sink, {}};
    QueryContext ctx = Context(4, &pool);
    RunPipeline(p, ctx);
    EXPECT_FALSE(source.prepare_concurrent);
    EXPECT_FALSE(source.make_local_concurrent.load());
    EXPECT_FALSE(source.worker_saw_off.load()) << "every morsel ran with the mode on";
    EXPECT_FALSE(sink.combine_concurrent);
    EXPECT_FALSE(sink.finalize_concurrent);
    EXPECT_FALSE(bpm_->IsConcurrent());
}

TEST_F(ParallelWorkerPoolTest, PipelineStopsOtherWorkersAfterAnError) {
    WorkerPool pool(4);
    CountingSource source(400);
    source.sleep_ms = 2;
    source.throw_at = 5;
    TaggingSink sink(bpm_.get());
    Pipeline p{&source, {}, &sink, {}};
    QueryContext ctx = Context(4, &pool);
    try {
        RunPipeline(p, ctx);
        FAIL() << "the morsel's exception should reach the caller";
    } catch (const std::invalid_argument& e) {
        EXPECT_STREQ(e.what(), "bad date in morsel 5");
    }
    EXPECT_LT(source.started.load(), 100) << "workers kept taking morsels after the error";
    EXPECT_FALSE(bpm_->IsConcurrent()) << "the mode is off after an error";
    EXPECT_TRUE(sink.order.empty()) << "nothing is combined after an error";
    EXPECT_EQ(sink.finalizes, 0);

    // The pool and the buffer pool both work afterwards.
    CountingSource again(16);
    TaggingSink sink2(bpm_.get());
    Pipeline p2{&again, {}, &sink2, {}};
    QueryContext ctx2 = Context(4, &pool);
    RunPipeline(p2, ctx2);
    EXPECT_EQ(sink2.total, 16 * static_cast<int64_t>(kRows));
    EXPECT_NO_THROW(bpm_->DropAllPages());
}

TEST_F(ParallelWorkerPoolTest, OneMorselRunsOnTheCallingThread) {
    WorkerPool pool(4);
    CountingSource source(1);
    TaggingSink sink(bpm_.get());
    Pipeline p{&source, {}, &sink, {}};
    QueryContext ctx = Context(4, &pool);
    RunPipeline(p, ctx);
    EXPECT_EQ(source.off_caller.load(), 0);
    EXPECT_TRUE(source.worker_saw_off.load()) << "one worker runs without the latch";
    EXPECT_EQ(sink.made.load(), 1) << "one worker, one local";
    EXPECT_EQ(sink.total, static_cast<int64_t>(kRows));

    // One thread never needs a pool.
    CountingSource many(8);
    TaggingSink sink2(bpm_.get());
    Pipeline p2{&many, {}, &sink2, {}};
    QueryContext ctx2 = Context(1, nullptr);
    RunPipeline(p2, ctx2);
    EXPECT_EQ(many.off_caller.load(), 0);
    EXPECT_EQ(sink2.made.load(), 1);
    EXPECT_EQ(ctx2.morsels, 8u);
}

TEST_F(ParallelWorkerPoolTest, ForEachTaskRunsEveryIndexOnce) {
    WorkerPool pool(4);
    for (size_t count : {0u, 1u, 2u, 7u, 64u}) {
        std::vector<std::atomic<int>> hits(count);
        std::atomic<int> off_caller{0};
        const auto caller = std::this_thread::get_id();
        QueryContext ctx = Context(4, &pool);
        ForEachTask(ctx, count, [&](size_t i) {
            hits[i].fetch_add(1);
            if (std::this_thread::get_id() != caller) off_caller.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        });
        for (size_t i = 0; i < count; ++i) EXPECT_EQ(hits[i].load(), 1) << count << " tasks, task " << i;
        if (count >= 7) EXPECT_GT(off_caller.load(), 0) << count << " tasks used the pool";
        EXPECT_FALSE(bpm_->IsConcurrent()) << "tasks never touch the buffer pool";
    }
    // A throwing task reaches the caller.
    QueryContext ctx = Context(4, &pool);
    EXPECT_THROW(ForEachTask(ctx, 16, [](size_t i) {
                     if (i == 3) throw DatabaseException("task 3");
                 }),
                 DatabaseException);
}

}  // namespace shilmandb::parallel
