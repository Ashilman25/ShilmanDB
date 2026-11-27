#include <gtest/gtest.h>
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "common/exception.hpp"
#include "executor/parallel/batch.hpp"
#include "executor/parallel/compiler.hpp"
#include "executor/parallel/expression.hpp"
#include "executor/parallel/hash_aggregate.hpp"
#include "executor/parallel/hash_join.hpp"
#include "executor/parallel/memory_budget.hpp"
#include "executor/parallel/parallel_executor.hpp"
#include "executor/parallel/pipeline.hpp"
#include "executor/parallel/sort.hpp"
#include "executor/parallel/vector.hpp"
#include "executor/parallel/worker_pool.hpp"
#include "parser/ast.hpp"
#include "storage/disk_manager.hpp"
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace shilmandb::parallel {

namespace {

constexpr uint32_t kRowsPerBatch = 100;

// Morsel m gives `batches` batches of one integer slot.
// Row i of the morsel holds m * 1000 + i.
class FakeSource : public Source {
public:
    struct Local : LocalState {
        size_t morsel{0};
        size_t emitted{0};
    };

    FakeSource(size_t morsels, size_t batches) : morsels_(morsels), batches_(batches) {}

    void Prepare(QueryContext&) override { ++prepared; }

    size_t MorselCount() const override {
        EXPECT_EQ(prepared, 1) << "MorselCount before Prepare";
        return morsels_;
    }

    std::unique_ptr<LocalState> MakeLocal() const override { return std::make_unique<Local>(); }

    void StartMorsel(size_t morsel, LocalState& local) const override {
        auto& l = static_cast<Local&>(local);
        l.morsel = morsel;
        l.emitted = 0;
        started.push_back(morsel);
    }

    bool NextBatch(LocalState& local, Batch& out) const override {
        auto& l = static_cast<Local&>(local);
        if (l.emitted == batches_) return false;
        ShapeBatch(out, {VType{}});
        out.Reset(kRowsPerBatch);
        int64_t* v = out.slots[0].I64();
        for (uint32_t i = 0; i < kRowsPerBatch; ++i) {
            v[i] = static_cast<int64_t>(l.morsel * 1000 + l.emitted * kRowsPerBatch + i);
        }
        ++l.emitted;
        return true;
    }

    // Test records, written from const calls; these fakes run on one thread.
    int prepared{0};
    mutable std::vector<size_t> started;

private:
    size_t morsels_;
    size_t batches_;
};

// Emits each input batch twice: HAVE_MORE_OUTPUT, then NEED_INPUT.
class Doubler : public Operator {
public:
    struct Local : LocalState {
        bool second{false};
    };

    std::unique_ptr<LocalState> MakeLocal() const override { return std::make_unique<Local>(); }

    OpResult Execute(Batch& in, Batch& out, LocalState* local) const override {
        auto& l = static_cast<Local&>(*local);
        ShapeBatch(out, {VType{}});
        out.Reset(in.Active());
        const int64_t* src = in.slots[0].I64();
        int64_t* dst = out.slots[0].I64();
        for (uint32_t k = 0; k < in.Active(); ++k) dst[k] = src[in.sel[k]];
        ++calls;
        l.second = !l.second;
        return l.second ? OpResult::HAVE_MORE_OUTPUT : OpResult::NEED_INPUT;
    }

    mutable int calls{0};  // written from const Execute; one thread only
};

// Keeps every value it sees and counts how often each step runs.
class FakeSink : public Sink {
public:
    struct Local : LocalState {
        std::vector<int64_t> values;
    };

    std::unique_ptr<LocalState> MakeLocal() const override {
        ++locals;
        return std::make_unique<Local>();
    }

    void Consume(Batch& in, LocalState& local) const override {
        EXPECT_EQ(combines, 0) << "Consume after Combine";
        auto& l = static_cast<Local&>(local);
        const int64_t* v = in.slots[0].I64();
        for (uint32_t k = 0; k < in.Active(); ++k) l.values.push_back(v[in.sel[k]]);
        ++batches;
    }

    void Combine(std::unique_ptr<LocalState> local) override {
        EXPECT_EQ(finalizes, 0) << "Combine after Finalize";
        ++combines;
        const auto& l = static_cast<const Local&>(*local);
        values.insert(values.end(), l.values.begin(), l.values.end());
    }

    void Finalize(QueryContext&) override { ++finalizes; }

    bool Done() const override { return stop_after > 0 && batches >= stop_after; }

    mutable int locals{0};
    int combines{0};
    int finalizes{0};
    mutable size_t batches{0};  // written from const Consume; one thread only
    size_t stop_after{0};  // 0: never done
    std::vector<int64_t> values;
};

// Every value a FakeSource with these sizes gives, sorted.
std::vector<int64_t> SourceValues(size_t morsels, size_t batches) {
    std::vector<int64_t> out;
    for (size_t m = 0; m < morsels; ++m) {
        for (size_t i = 0; i < batches * kRowsPerBatch; ++i) out.push_back(static_cast<int64_t>(m * 1000 + i));
    }
    return out;
}

// The values sorted, so runs compare as multisets.
std::vector<int64_t> Sorted(std::vector<int64_t> v) {
    std::sort(v.begin(), v.end());
    return v;
}

// A context for fakes that never touch the pool or the catalog.
QueryContext NoDatabase() {
    return QueryContext{nullptr, nullptr};
}

// The worker and context each local was made for; safe from any thread.
struct LocalLog {
    std::mutex mutex;
    std::vector<unsigned> workers;
    std::vector<const QueryContext*> contexts;

    void Add(unsigned worker, const QueryContext& ctx) {
        std::lock_guard<std::mutex> lock(mutex);
        workers.push_back(worker);
        contexts.push_back(&ctx);
    }

    // The workers seen, sorted.
    std::vector<unsigned> Workers() {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<unsigned> sorted = workers;
        std::sort(sorted.begin(), sorted.end());
        return sorted;
    }
};

// A local that knows its worker and the threads that used it.
struct WorkerLocal : LocalState {
    unsigned worker{0};
    bool batch_left{false};
    std::vector<std::thread::id> threads;  // one thread uses a local at a time

    void Touch() {
        const auto id = std::this_thread::get_id();
        if (std::find(threads.begin(), threads.end(), id) == threads.end()) threads.push_back(id);
    }
};

// A local for this worker, logged.
std::unique_ptr<LocalState> LoggedLocal(LocalLog& log, const QueryContext& ctx, unsigned worker) {
    log.Add(worker, ctx);
    auto local = std::make_unique<WorkerLocal>();
    local->worker = worker;
    return local;
}

// One batch of one row per morsel.
class WorkerSource : public Source {
public:
    explicit WorkerSource(size_t morsels) : morsels_(morsels) {}
    void Prepare(QueryContext&) override {}
    size_t MorselCount() const override { return morsels_; }
    std::unique_ptr<LocalState> MakeLocal() const override { return std::make_unique<WorkerLocal>(); }
    std::unique_ptr<LocalState> MakeWorkerLocal(const QueryContext& ctx, unsigned worker) const override {
        return LoggedLocal(log, ctx, worker);
    }
    void StartMorsel(size_t, LocalState& local) const override {
        auto& l = static_cast<WorkerLocal&>(local);
        l.Touch();
        l.batch_left = true;
    }
    bool NextBatch(LocalState& local, Batch& out) const override {
        auto& l = static_cast<WorkerLocal&>(local);
        if (!l.batch_left) return false;
        l.batch_left = false;
        ShapeBatch(out, {VType{}});
        out.Reset(1);
        out.slots[0].I64()[0] = 1;
        return true;
    }
    mutable LocalLog log;

private:
    size_t morsels_;
};

// Passes each batch on.
class WorkerOperator : public Operator {
public:
    std::unique_ptr<LocalState> MakeWorkerLocal(const QueryContext& ctx, unsigned worker) const override {
        return LoggedLocal(log, ctx, worker);
    }
    OpResult Execute(Batch& in, Batch& out, LocalState* local) const override {
        if (local != nullptr) static_cast<WorkerLocal*>(local)->Touch();
        std::swap(in, out);
        return OpResult::NEED_INPUT;
    }
    mutable LocalLog log;
};

// Counts rows and keeps each local's worker and threads.
class WorkerSink : public Sink {
public:
    std::unique_ptr<LocalState> MakeLocal() const override { return std::make_unique<WorkerLocal>(); }
    std::unique_ptr<LocalState> MakeWorkerLocal(const QueryContext& ctx, unsigned worker) const override {
        return LoggedLocal(log, ctx, worker);
    }
    void Consume(Batch& in, LocalState& local) const override {
        static_cast<WorkerLocal&>(local).Touch();
        rows.fetch_add(in.Active());
    }
    void Combine(std::unique_ptr<LocalState> local) override {
        const auto& l = static_cast<const WorkerLocal&>(*local);
        combined.emplace_back(l.worker, l.threads);
    }
    void Finalize(QueryContext&) override {}
    mutable LocalLog log;
    mutable std::atomic<uint64_t> rows{0};
    std::vector<std::pair<unsigned, std::vector<std::thread::id>>> combined;
};

// Passes batches on; after the morsels hands out `parts` spilled partitions of
// one batch each. Partition p's batch holds base + 10 * p
// and base + 10 * p + 1. Safe from any thread.
class SpillingOperator : public Operator {
public:
    struct Local : LocalState {
        unsigned worker{0};
        size_t part{0};
        bool batch_left{false};
    };

    SpillingOperator(size_t parts, int64_t base, std::atomic<int>& clock) : parts_(parts), base_(base), clock_(clock) {}

    std::unique_ptr<LocalState> MakeWorkerLocal(const QueryContext&, unsigned worker) const override {
        auto local = std::make_unique<Local>();
        local->worker = worker;
        return local;
    }
    OpResult Execute(Batch& in, Batch& out, LocalState*) const override {
        std::swap(in, out);
        return OpResult::NEED_INPUT;
    }
    void FinishMorsels(LocalState* local) const override {
        const std::lock_guard<std::mutex> lock(mutex_);
        finishes.push_back({static_cast<Local*>(local)->worker, clock_.fetch_add(1)});
    }
    size_t SpilledPartitionCount() const override { return parts_; }
    void StartSpilledPartition(size_t p, LocalState* local) const override {
        auto& l = *static_cast<Local*>(local);
        l.part = p;
        l.batch_left = true;
        const std::lock_guard<std::mutex> lock(mutex_);
        starts.push_back({static_cast<unsigned>(p), clock_.fetch_add(1)});
    }
    bool NextSpilledBatch(LocalState* local, Batch& out) const override {
        auto& l = *static_cast<Local*>(local);
        if (!l.batch_left) return false;
        l.batch_left = false;
        ShapeBatch(out, {VType{}});
        out.Reset(2);
        out.slots[0].I64()[0] = base_ + static_cast<int64_t>(10 * l.part);
        out.slots[0].I64()[1] = base_ + static_cast<int64_t>(10 * l.part) + 1;
        return true;
    }
    void PipelineDone() override { done_at.push_back(clock_.fetch_add(1)); }

    // (worker or partition, time) per call, in the order they happened.
    mutable std::vector<std::pair<unsigned, int>> finishes;
    mutable std::vector<std::pair<unsigned, int>> starts;
    std::vector<int> done_at;

private:
    size_t parts_;
    int64_t base_;
    std::atomic<int>& clock_;
    mutable std::mutex mutex_;
};

// Keeps every value; safe from any thread. Records when Finalize ran.
class ValueSink : public Sink {
public:
    struct Local : LocalState {
        std::vector<int64_t> values;
    };

    explicit ValueSink(std::atomic<int>& clock) : clock_(clock) {}

    std::unique_ptr<LocalState> MakeLocal() const override { return std::make_unique<Local>(); }
    void Consume(Batch& in, LocalState& local) const override {
        auto& l = static_cast<Local&>(local);
        const int64_t* v = in.slots[0].I64();
        for (uint32_t k = 0; k < in.Active(); ++k) l.values.push_back(v[in.sel[k]]);
    }
    void Combine(std::unique_ptr<LocalState> local) override {
        ++combines;
        const auto& l = static_cast<const Local&>(*local);
        values.insert(values.end(), l.values.begin(), l.values.end());
    }
    void Finalize(QueryContext&) override { finalized_at = clock_.fetch_add(1); }

    int combines{0};
    int finalized_at{-1};
    std::vector<int64_t> values;

private:
    std::atomic<int>& clock_;
};

// A column reference bound to plan column c.
std::unique_ptr<CompiledExpr> Col(uint32_t c, const InputLayout& in) {
    ColumnRef ref;
    ref.column_name = "c" + std::to_string(c);
    ref.bound_index = c;
    return CompiledExpr::Compile(ref, in);
}

// One integer column in slot 0.
InputLayout OneInt() {
    InputLayout in;
    in.slot_of_column = {0};
    in.slot_types = {VType{}};
    in.slot_precision = {0};
    return in;
}

// The times in a call log.
std::vector<int> Times(const std::vector<std::pair<unsigned, int>>& log) {
    std::vector<int> out;
    for (const auto& entry : log) out.push_back(entry.second);
    return out;
}

// The partitions in a start log, sorted.
std::vector<unsigned> Parts(const std::vector<std::pair<unsigned, int>>& log) {
    std::vector<unsigned> out;
    for (const auto& entry : log) out.push_back(entry.first);
    std::sort(out.begin(), out.end());
    return out;
}

// Descriptors this process has open, counted from /dev/fd.
size_t OpenDescriptors() {
    std::error_code ec;
    size_t n = 0;
    for (std::filesystem::directory_iterator it("/dev/fd", ec), end; !ec && it != end; it.increment(ec)) ++n;
    return n;
}

// A breaker's sink and the source that reads what it kept.
using Breaker = std::pair<std::unique_ptr<Sink>, std::unique_ptr<Source>>;

// What a breaker held between its two pipelines, and right after the second.
struct HeldAcross {
    size_t bytes_between{0};
    size_t bytes_after{0};
    size_t descriptors_before{0};
    size_t descriptors_between{0};
    size_t descriptors_after{0};
    std::vector<int64_t> values;  // slot 0 of what the source handed out
};

// A temp directory for spills, unique to this process.
std::string SpillDir() {
    return (std::filesystem::temp_directory_path() / ("shilmandb_parallel_pipeline_test_" + std::to_string(::getpid()))).string();
}

// Feeds FakeSource(5, 3) through a breaker on one thread, as a query would.
// Notes what's held while the breaker is still alive, then destroys it before its budget.
HeldAcross RunBreaker(const std::function<Breaker()>& make, bool spill_everything) {
    const std::string dir = SpillDir();
    std::filesystem::remove_all(dir);
    HeldAcross held;
    held.descriptors_before = OpenDescriptors();
    {
        MemoryBudget budget(kWorkerMinimumBytes + 2 * kGrantBytes, 1);
        SpillCounters counters;
        std::atomic<int> clock{0};
        Breaker made = make();
        FakeSource input(5, 3);
        ValueSink out(clock);
        QueryContext ctx = NoDatabase();
        ctx.budget = &budget;
        ctx.spill = &counters;
        ctx.temp_dir = &dir;
        ctx.spill_everything = spill_everything;

        Pipeline in{&input, {}, made.first.get(), {}};
        RunPipeline(in, ctx);
        held.bytes_between = budget.HeldBytes();
        held.descriptors_between = OpenDescriptors();

        Pipeline read{made.second.get(), {}, &out, {0}};
        RunPipeline(read, ctx);
        held.bytes_after = budget.HeldBytes();
        held.descriptors_after = OpenDescriptors();
        held.values = std::move(out.values);
        EXPECT_EQ(spill_everything, counters.files.load() > 0);
        made.first.reset();
        made.second.reset();
    }
    std::filesystem::remove_all(dir);
    return held;
}

// GROUP BY slot 0 with a COUNT(*).
Breaker CountByKey() {
    const InputLayout one = OneInt();
    AggregateSpec spec;
    spec.keys.push_back(Col(0, one));
    spec.funcs = {Aggregate::Func::COUNT};
    spec.args.push_back(nullptr);
    spec.out_types = {spec.keys[0]->Type(), AggregateOutputType(Aggregate::Func::COUNT, nullptr)};
    return MakeHashAggregate(std::move(spec));
}

// ORDER BY slot 0, keeping it.
Breaker SortByKey() {
    const InputLayout one = OneInt();
    SortSpec spec;
    spec.keys.push_back(Col(0, one));
    spec.ascending = {true};
    spec.columns = {0};
    spec.types = {VType{}};
    return MakeSort(std::move(spec));
}

}  // namespace

TEST(ParallelPipelineTest, RunPipelineVisitsEveryMorselOnce) {
    FakeSource source(5, 3);
    FakeSink sink;
    Pipeline p{&source, {}, &sink, {}};
    QueryContext ctx = NoDatabase();
    RunPipeline(p, ctx);

    EXPECT_EQ(source.prepared, 1);
    EXPECT_EQ(source.started, (std::vector<size_t>{0, 1, 2, 3, 4}));
    EXPECT_EQ(sink.batches, 15u);
    EXPECT_EQ(Sorted(sink.values), SourceValues(5, 3));
}

TEST(ParallelPipelineTest, HaveMoreOutputIsDrained) {
    FakeSource source(2, 2);
    Doubler doubler;
    FakeSink sink;
    Pipeline p{&source, {&doubler}, &sink, {}};
    QueryContext ctx = NoDatabase();
    RunPipeline(p, ctx);

    // Four input batches, each passed on twice.
    EXPECT_EQ(doubler.calls, 8);
    EXPECT_EQ(sink.batches, 8u);
    std::vector<int64_t> twice = SourceValues(2, 2);
    const std::vector<int64_t> once = twice;
    twice.insert(twice.end(), once.begin(), once.end());
    EXPECT_EQ(Sorted(sink.values), Sorted(twice));
}

TEST(ParallelPipelineTest, HaveMoreOutputIsDrainedThroughTwoOperators) {
    FakeSource source(1, 3);
    Doubler first;
    Doubler second;
    FakeSink sink;
    Pipeline p{&source, {&first, &second}, &sink, {}};
    QueryContext ctx = NoDatabase();
    RunPipeline(p, ctx);

    // Each source batch reaches the sink four times.
    EXPECT_EQ(first.calls, 6);
    EXPECT_EQ(second.calls, 12);
    EXPECT_EQ(sink.batches, 12u);
    EXPECT_EQ(sink.values.size(), 4 * SourceValues(1, 3).size());
}

TEST(ParallelPipelineTest, CombineAndFinalizeRunOnce) {
    for (const size_t morsels : {size_t{0}, size_t{1}, size_t{6}}) {
        FakeSource source(morsels, 2);
        FakeSink sink;
        Pipeline p{&source, {}, &sink, {}};
        QueryContext ctx = NoDatabase();
        RunPipeline(p, ctx);

        EXPECT_EQ(source.prepared, 1) << morsels << " morsels";
        EXPECT_EQ(sink.locals, 1) << morsels << " morsels";
        EXPECT_EQ(sink.combines, 1) << morsels << " morsels";
        EXPECT_EQ(sink.finalizes, 1) << morsels << " morsels";
        EXPECT_EQ(Sorted(sink.values), SourceValues(morsels, 2)) << morsels << " morsels";
    }
}

TEST(ParallelPipelineTest, DoneStopsTakingMorsels) {
    FakeSource source(4, 3);
    FakeSink sink;
    sink.stop_after = 1;
    Pipeline p{&source, {}, &sink, {}};
    QueryContext ctx = NoDatabase();
    RunPipeline(p, ctx);

    EXPECT_EQ(source.started, std::vector<size_t>{0});
    EXPECT_EQ(sink.batches, 1u);
    EXPECT_EQ(sink.values.size(), kRowsPerBatch);
    EXPECT_EQ(sink.combines, 1);
    EXPECT_EQ(sink.finalizes, 1);
}

TEST(ParallelPipelineTest, DependencyOnALaterPipelineIsRefused) {
    // Pipelines run in list order, so each may depend only on earlier ones:
    // not a later one, not itself, not one past the end.
    struct Bad {
        size_t pipeline;
        std::vector<size_t> depends_on;
    };
    for (const Bad& bad : {Bad{0, {1}}, Bad{0, {0}}, Bad{1, {1}}, Bad{1, {0, 2}}}) {
        FakeSource first(1, 1);
        FakeSource second(1, 1);
        FakeSink sink;
        ResultSink result(Schema(std::vector<Column>{}), InputLayout{});  // never fed; there if the run goes ahead
        PhysicalPlan plan;
        plan.pipelines.push_back(Pipeline{&first, {}, &sink, {}});
        plan.pipelines.push_back(Pipeline{&second, {}, &sink, {}});
        plan.pipelines[bad.pipeline].depends_on = bad.depends_on;
        plan.result = &result;
        QueryContext ctx = NoDatabase();
        try {
            (void)RunPhysicalPlan(plan, ctx);
            ADD_FAILURE() << "pipeline " << bad.pipeline << " ran with a bad dependency";
        } catch (const DatabaseException& e) {
            EXPECT_NE(std::string(e.what()).find("which doesn't run before it"), std::string::npos) << e.what();
        }
        // Refused before anything ran.
        EXPECT_EQ(first.prepared, 0);
        EXPECT_EQ(second.prepared, 0);
    }
}

TEST(ParallelPipelineTest, ResolveThreadsDefaultsAndCaps) {
    const unsigned cores = std::max(1u, std::thread::hardware_concurrency());
    EXPECT_EQ(ResolveThreads(0), std::min(cores, kMaxThreads));
    EXPECT_EQ(ResolveThreads(1), 1u);
    EXPECT_EQ(ResolveThreads(7), 7u);
    EXPECT_EQ(ResolveThreads(256), 256u);
    EXPECT_EQ(ResolveThreads(257), 256u);
    EXPECT_EQ(ResolveThreads(100000), 256u);
}

TEST(ParallelPipelineTest, ForEachTaskPassesTheWorker) {
    // Serial: every task runs as worker 0.
    QueryContext serial = NoDatabase();
    std::vector<unsigned> seen;
    ForEachTask(serial, 5, [&](size_t, unsigned worker) { seen.push_back(worker); });
    EXPECT_EQ(seen, std::vector<unsigned>(5, 0u));

    // Pooled: each worker index is below threads and runs on one thread at a time.
    constexpr unsigned kThreads = 4;
    WorkerPool pool(kThreads);
    QueryContext ctx = NoDatabase();
    ctx.threads = kThreads;
    ctx.pool = &pool;
    constexpr size_t kTasks = 64;
    std::vector<std::atomic<int>> hits(kTasks);
    std::vector<std::atomic<int>> inside(kThreads);
    std::vector<std::atomic<int>> used(kThreads);
    std::atomic<bool> bad_worker{false};
    std::atomic<bool> overlapped{false};
    ForEachTask(ctx, kTasks, [&](size_t i, unsigned worker) {
        hits[i].fetch_add(1);
        if (worker >= kThreads) {
            bad_worker.store(true);
            return;
        }
        if (inside[worker].fetch_add(1) != 0) overlapped.store(true);
        used[worker].fetch_add(1);
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        inside[worker].fetch_sub(1);
    });
    EXPECT_FALSE(bad_worker.load()) << "a worker index past threads";
    EXPECT_FALSE(overlapped.load()) << "one worker index ran on two threads at once";
    for (size_t i = 0; i < kTasks; ++i) EXPECT_EQ(hits[i].load(), 1) << "task " << i;
    int workers_used = 0;
    for (const auto& u : used) workers_used += u.load() > 0 ? 1 : 0;
    EXPECT_GT(workers_used, 1) << "the pool ran the tasks";
}

TEST(ParallelPipelineTest, ForEachPoolTaskTurnsTheLatchOnWhileTasksRun) {
    const std::string file = (std::filesystem::temp_directory_path() / ("shilmandb_parallel_pipeline_pool_task_" + std::to_string(::getpid()) + ".db")).string();
    std::filesystem::remove(file);
    {
        constexpr size_t kFrames = 32;
        constexpr page_id_t kPages = 8;
        DiskManager disk(file);
        BufferPoolManager bpm(kFrames, &disk, std::make_unique<LRUEvictionPolicy>(kFrames));
        for (page_id_t i = 0; i < kPages; ++i) {
            page_id_t id = INVALID_PAGE_ID;
            ASSERT_NE(bpm.NewPage(&id), nullptr);
            (void)bpm.UnpinPage(id, true);
        }
        constexpr unsigned kThreads = 4;
        WorkerPool pool(kThreads);
        QueryContext ctx{&bpm, nullptr};
        ctx.threads = kThreads;
        ctx.pool = &pool;
        const std::thread::id caller = std::this_thread::get_id();

        // Tasks that pin pages from several threads, each seeing the mode on for the workers that run.
        for (const size_t count : {size_t{64}, size_t{2}}) {
            const unsigned workers = static_cast<unsigned>(std::min<size_t>(count, kThreads));
            std::atomic<int> ran{0};
            std::atomic<int> off{0};
            ForEachPoolTask(ctx, count, [&](size_t i, unsigned) {
                if (!bpm.IsConcurrent() || bpm.ConcurrentWorkers() != workers) off.fetch_add(1);
                const auto id = static_cast<page_id_t>(i % kPages);
                if (bpm.FetchPage(id) == nullptr) {
                    off.fetch_add(1);
                    return;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                (void)bpm.UnpinPage(id, false);
                ran.fetch_add(1);
            });
            EXPECT_EQ(ran.load(), static_cast<int>(count));
            EXPECT_EQ(off.load(), 0) << count << " tasks";
            EXPECT_FALSE(bpm.IsConcurrent()) << "left on after " << count << " tasks";
            EXPECT_EQ(bpm.ConcurrentWorkers(), 1u);
        }

        // A failed task reaches the caller, and the mode is off again.
        std::atomic<int> started{0};
        try {
            ForEachPoolTask(ctx, 64, [&](size_t i, unsigned) {
                started.fetch_add(1);
                if (i == 5) throw DatabaseException("task 5 failed");
            });
            ADD_FAILURE() << "no throw";
        } catch (const DatabaseException& e) {
            EXPECT_STREQ(e.what(), "task 5 failed");
        }
        EXPECT_GT(started.load(), 0);
        EXPECT_FALSE(bpm.IsConcurrent()) << "left on after a failed task";

        // One worker (no pool, or one task): the caller runs it as worker 0 with the mode off.
        for (const bool pooled : {false, true}) {
            QueryContext one{&bpm, nullptr};
            one.threads = pooled ? kThreads : 1;
            one.pool = pooled ? &pool : nullptr;
            const size_t count = pooled ? 1 : 5;
            size_t ran = 0;
            ForEachPoolTask(one, count, [&](size_t i, unsigned worker) {
                EXPECT_FALSE(bpm.IsConcurrent());
                EXPECT_EQ(worker, 0u);
                EXPECT_EQ(std::this_thread::get_id(), caller);
                const auto id = static_cast<page_id_t>(i % kPages);
                ASSERT_NE(bpm.FetchPage(id), nullptr);
                (void)bpm.UnpinPage(id, false);
                ++ran;
            });
            EXPECT_EQ(ran, count) << (pooled ? "one task" : "no pool");
        }

        for (size_t f = 0; f < bpm.GetPoolSize(); ++f) EXPECT_EQ(bpm.GetPage(static_cast<frame_id_t>(f)).GetPinCount(), 0) << "frame " << f;
        EXPECT_NO_THROW(bpm.DropAllPages());
    }
    std::filesystem::remove(file);
}

#ifndef NDEBUG
TEST(ParallelPipelineDeathTest, ForEachPoolTaskInsideARunningPipelineAsserts) {
    // A nested call would switch the mode off under the outer scope's workers.
    const std::string file = (std::filesystem::temp_directory_path() / ("shilmandb_parallel_pipeline_nested_" + std::to_string(::getpid()) + ".db")).string();
    std::filesystem::remove(file);
    {
        DiskManager disk(file);
        BufferPoolManager bpm(8, &disk, std::make_unique<LRUEvictionPolicy>(8));
        QueryContext ctx{&bpm, nullptr};
        bpm.SetConcurrent(true, 2);
        EXPECT_DEATH(ForEachPoolTask(ctx, 4, [](size_t, unsigned) {}), "ForEachPoolTask inside a running pipeline");
        bpm.SetConcurrent(false);
    }
    std::filesystem::remove(file);
}
#endif

TEST(ParallelPipelineTest, LocalsKnowTheirWorkerAndContext) {
    constexpr unsigned kThreads = 4;
    WorkerPool pool(kThreads);
    for (const unsigned threads : {1u, kThreads}) {
        WorkerSource source(8);
        WorkerOperator op;
        WorkerSink sink;
        Pipeline p{&source, {&op}, &sink, {}};
        QueryContext ctx = NoDatabase();
        ctx.threads = threads;
        ctx.pool = threads > 1 ? &pool : nullptr;
        RunPipeline(p, ctx);

        // Every step made one local per worker, each told its worker and this context.
        std::vector<unsigned> want(threads);
        std::iota(want.begin(), want.end(), 0u);
        for (LocalLog* log : {&source.log, &op.log, &sink.log}) {
            EXPECT_EQ(log->Workers(), want) << threads << " threads";
            for (const QueryContext* seen : log->contexts) EXPECT_EQ(seen, &ctx) << threads << " threads";
        }

        // Each local ran on one thread: worker 0 on the caller's, no two sharing one.
        EXPECT_EQ(sink.rows.load(), 8u);
        ASSERT_EQ(sink.combined.size(), threads);
        std::vector<std::thread::id> used;
        for (const auto& [worker, seen] : sink.combined) {
            ASSERT_LE(seen.size(), 1u) << "worker " << worker;
            if (seen.empty()) continue;
            if (worker == 0) EXPECT_EQ(seen[0], std::this_thread::get_id());
            EXPECT_EQ(std::count(used.begin(), used.end(), seen[0]), 0) << "worker " << worker;
            used.push_back(seen[0]);
        }
    }
}

TEST(ParallelPipelineTest, SpillPhasesRunAfterTheMorsels) {
    constexpr unsigned kThreads = 4;
    WorkerPool pool(kThreads);
    for (const unsigned threads : {1u, kThreads}) {
        std::atomic<int> clock{0};
        FakeSource source(1, 2);
        SpillingOperator first(3, 1000000, clock);
        SpillingOperator second(2, 2000000, clock);
        ValueSink sink(clock);
        Pipeline p{&source, {&first, &second}, &sink, {}};
        QueryContext ctx = NoDatabase();
        ctx.threads = threads;
        ctx.pool = threads > 1 ? &pool : nullptr;
        RunPipeline(p, ctx);

        // One morsel, but three spilled partitions: three workers run, and all are combined.
        const size_t workers = threads == 1 ? 1 : 3;
        EXPECT_EQ(sink.combines, static_cast<int>(workers)) << threads << " threads";

        // The first operator's partitions pass through the second; the second's go to the sink.
        std::vector<int64_t> want = SourceValues(1, 2);
        for (int64_t p0 = 0; p0 < 3; ++p0) want.insert(want.end(), {1000000 + 10 * p0, 1000001 + 10 * p0});
        for (int64_t p1 = 0; p1 < 2; ++p1) want.insert(want.end(), {2000000 + 10 * p1, 2000001 + 10 * p1});
        EXPECT_EQ(Sorted(sink.values), Sorted(want)) << threads << " threads";
        EXPECT_EQ(Parts(first.starts), (std::vector<unsigned>{0, 1, 2}));
        EXPECT_EQ(Parts(second.starts), (std::vector<unsigned>{0, 1}));

        // Every worker finishes its morsels on both operators,
        // then the second again after the first phase.
        ASSERT_EQ(first.finishes.size(), workers);
        ASSERT_EQ(second.finishes.size(), 2 * workers);
        const std::vector<int> first_finishes = Times(first.finishes);
        const std::vector<int> first_starts = Times(first.starts);
        const std::vector<int> second_finishes = Times(second.finishes);
        const std::vector<int> second_starts = Times(second.starts);
        const int first_phase = *std::min_element(first_starts.begin(), first_starts.end());
        EXPECT_GT(first_phase, *std::max_element(first_finishes.begin(), first_finishes.end()));
        const auto before_first_phase = std::count_if(second_finishes.begin(), second_finishes.end(), [&](int t) { return t < first_phase; });
        EXPECT_EQ(static_cast<size_t>(before_first_phase), workers);
        EXPECT_GT(*std::min_element(second_starts.begin(), second_starts.end()), *std::max_element(second_finishes.begin(), second_finishes.end()));

        // PipelineDone once per operator, before Finalize, so its memory is back for the sink.
        ASSERT_EQ(first.done_at.size(), 1u);
        ASSERT_EQ(second.done_at.size(), 1u);
        EXPECT_LT(first.done_at[0], sink.finalized_at);
        EXPECT_LT(second.done_at[0], sink.finalized_at);
    }
}

TEST(ParallelPipelineTest, FinishedBuildFreedWhenItsProbeEnds) {
    MemoryBudget budget(kWorkerMinimumBytes + 2 * kGrantBytes, 1);
    std::atomic<int> clock{0};
    {
        const InputLayout one = OneInt();
        JoinSpec spec;
        spec.build_key = Col(0, one);
        spec.probe_key = Col(0, one);
        spec.build_payload = {0};
        spec.probe_slots = {0};
        spec.types = {VType{}, VType{}};
        auto made = MakeHashJoin(std::move(spec));

        // Build keys m * 1000 + 0..299 for
        // m = 0..4; probe keys 0..99 and 1000..1099.
        FakeSource build(5, 3);
        FakeSource probe(2, 1);
        ValueSink out(clock);
        QueryContext ctx = NoDatabase();
        ctx.budget = &budget;
        Pipeline build_side{&build, {}, made.first.get(), {}};
        RunPipeline(build_side, ctx);
        EXPECT_GT(budget.HeldBytes(), 0u) << "the finished table keeps its memory until it's probed";

        Pipeline probe_side{&probe, {made.second.get()}, &out, {0}};
        RunPipeline(probe_side, ctx);
        EXPECT_EQ(budget.HeldBytes(), 0u) << "the table goes once its probe pipeline ends";
        EXPECT_EQ(out.values.size(), 200u);

        // The operator still answers for tests.
        const std::optional<HashJoinInfo> info = InspectHashJoin(*made.second);
        ASSERT_TRUE(info.has_value());
        EXPECT_EQ(info->build_rows, 1500u);
        EXPECT_EQ(info->spilled_partitions, 0u);
    }
    EXPECT_EQ(budget.HeldBytes(), 0u);
}

TEST(ParallelPipelineTest, FinishedGroupsFreedWhenTheirSourceEnds) {
    // In memory, the merged groups keep their counts until they're read.
    const HeldAcross merged = RunBreaker(CountByKey, false);
    EXPECT_GT(merged.bytes_between, 0u) << "the merged groups keep their memory until they're read";
    EXPECT_EQ(merged.bytes_after, 0u) << "the groups go once their source's pipeline ends";
    EXPECT_EQ(merged.descriptors_after, merged.descriptors_before);
    EXPECT_EQ(Sorted(merged.values), SourceValues(5, 3));

    // Spilled, the workers' run files stay open until the source's merges read them.
    const HeldAcross spilled = RunBreaker(CountByKey, true);
    EXPECT_GT(spilled.descriptors_between, spilled.descriptors_before) << "the run files wait for the source";
    EXPECT_EQ(spilled.descriptors_after, spilled.descriptors_before) << "the run files go once the source's pipeline ends";
    EXPECT_EQ(spilled.bytes_after, 0u);
    EXPECT_EQ(Sorted(spilled.values), SourceValues(5, 3));
}

TEST(ParallelPipelineTest, FinishedSortFreedWhenItsSourceEnds) {
    // In memory, the sorted rows keep their counts until they're read.
    const HeldAcross sorted = RunBreaker(SortByKey, false);
    EXPECT_GT(sorted.bytes_between, 0u) << "the sorted rows keep their memory until they're read";
    EXPECT_EQ(sorted.bytes_after, 0u) << "the rows go once their source's pipeline ends";
    EXPECT_EQ(sorted.descriptors_after, sorted.descriptors_before);
    EXPECT_EQ(sorted.values, SourceValues(5, 3));

    // Spilled, the run files stay open until the source's merge reads them.
    const HeldAcross spilled = RunBreaker(SortByKey, true);
    EXPECT_GT(spilled.descriptors_between, spilled.descriptors_before) << "the run files wait for the source";
    EXPECT_EQ(spilled.descriptors_after, spilled.descriptors_before) << "the run files go once the source's pipeline ends";
    EXPECT_EQ(spilled.bytes_after, 0u);
    EXPECT_EQ(spilled.values, SourceValues(5, 3));
}

TEST(ParallelPipelineTest, FreedBuildReachesTheNextSinksFinalize) {
    // One worker: a floor of two grants and one shared grant, which the finished build takes.
    MemoryBudget budget(kWorkerMinimumBytes + kGrantBytes, 1);
    const std::string dir = SpillDir();
    std::filesystem::remove_all(dir);
    {
        const InputLayout one = OneInt();
        JoinSpec spec;
        spec.build_key = Col(0, one);
        spec.probe_key = Col(0, one);
        spec.build_payload = {0};
        spec.probe_slots = {0};
        spec.types = {VType{}, VType{}};
        auto join = MakeHashJoin(std::move(spec));
        Breaker groups = CountByKey();

        // Build keys 0..99, probe keys 0..199: keys 0..99 match once each.
        FakeSource build(1, 1);
        FakeSource probe(1, 2);
        std::atomic<int> clock{0};
        ValueSink out(clock);
        SpillCounters counters;
        QueryContext ctx = NoDatabase();
        ctx.budget = &budget;
        ctx.spill = &counters;
        ctx.temp_dir = &dir;

        Pipeline build_side{&build, {}, join.first.get(), {}};
        RunPipeline(build_side, ctx);
        ASSERT_GT(budget.HeldBytes(), 0u) << "the finished table holds the shared grant";

        // The aggregate's part can leave its floor only once the table's grant is back.
        Pipeline probe_side{&probe, {join.second.get()}, groups.first.get(), {0}};
        RunPipeline(probe_side, ctx);
        EXPECT_EQ(counters.files.load(), 0u) << "the probe's table was still counted when the aggregate finalized";

        Pipeline read{groups.second.get(), {}, &out, {1}};
        RunPipeline(read, ctx);
        EXPECT_EQ(Sorted(out.values), SourceValues(1, 1));
        join.first.reset();
        join.second.reset();
        groups.first.reset();
        groups.second.reset();
    }
    EXPECT_EQ(budget.HeldBytes(), 0u);
    std::filesystem::remove_all(dir);
}

}  // namespace shilmandb::parallel
