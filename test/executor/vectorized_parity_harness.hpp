#pragma once
#include <gtest/gtest.h>

#include "catalog/schema.hpp"
#include "executor/executor_factory.hpp"
#include "types/tuple.hpp"

#include <vector>

namespace shilmandb::test {

struct ParityRuns {
    std::vector<Tuple> tuple_mode;
    std::vector<Tuple> vectorized_mode;
    Schema schema;
};

// Builds both executors from the same plan, drives them to completion,
// and returns the collected row sets. Both runs share the supplied ctx
// (catalog + BPM); scan / filter only read catalog state, so reusing the
// context between runs is safe.
inline ParityRuns RunBothModes(const PlanNode* plan, ExecutorContext* ctx) {
    auto drain = [&](ExecutionMode mode) {
        auto exec = ExecutorFactory::CreateExecutor(plan, ctx, mode);
        exec->Init();
        std::vector<Tuple> rows;
        Tuple t;
        while (exec->Next(&t)) rows.push_back(t);
        exec->Close();
        return rows;
    };
    return {drain(ExecutionMode::TUPLE),
            drain(ExecutionMode::VECTORIZED),
            plan->output_schema};
}

// Row-for-row comparator — assumes identical emission order, which holds
// for SeqScan + Filter (both iterate the heap in physical order). 19.5 /
// 19.6 (Aggregate / Sort) will need a sort-then-compare overload.
inline void ExpectRowsEqual(const std::vector<Tuple>& a,
                            const std::vector<Tuple>& b,
                            const Schema& schema) {
    ASSERT_EQ(a.size(), b.size()) << "parity: row-count mismatch";
    for (size_t r = 0; r < a.size(); ++r) {
        for (uint32_t c = 0; c < schema.GetColumnCount(); ++c) {
            EXPECT_TRUE(a[r].GetValue(schema, c) == b[r].GetValue(schema, c))
                << "parity: row " << r << " col " << c << " differs";
        }
    }
}

}  // namespace shilmandb::test
