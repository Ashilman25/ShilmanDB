#include <gtest/gtest.h>
#include "common/exception.hpp"
#include "executor/parallel/batch.hpp"
#include "executor/parallel/compiler.hpp"
#include "executor/parallel/parallel_executor.hpp"
#include "executor/parallel/pipeline.hpp"
#include "executor/parallel/vector.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
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

    void StartMorsel(size_t morsel, LocalState& local) override {
        auto& l = static_cast<Local&>(local);
        l.morsel = morsel;
        l.emitted = 0;
        started.push_back(morsel);
    }

    bool NextBatch(LocalState& local, Batch& out) override {
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

    int prepared{0};
    std::vector<size_t> started;

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

    OpResult Execute(Batch& in, Batch& out, LocalState* local) override {
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

    int calls{0};
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

    void Consume(Batch& in, LocalState& local) override {
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
    size_t batches{0};
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

}  // namespace shilmandb::parallel
