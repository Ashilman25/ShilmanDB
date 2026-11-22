#pragma once
#include "catalog/schema.hpp"
#include "executor/parallel/expression.hpp"
#include "executor/parallel/pipeline.hpp"
#include "types/tuple.hpp"
#include <cstdint>
#include <memory>
#include <vector>

namespace shilmandb::parallel {

class LimitCounter;

// Collects a query's result rows and turns them into Tuples on the calling thread.
// Workers copy their rows into their own buffers; no Value is built until Finalize.
class ResultSink : public Sink {
public:
    // Throws DatabaseException when a column's type differs from the plan's.
    ResultSink(Schema schema, const InputLayout& layout);

    // Done once this counter is used up.
    void WatchLimit(const LimitCounter* counter) { limits_.push_back(counter); }

    [[nodiscard]] std::unique_ptr<LocalState> MakeLocal() const override;
    void Consume(Batch& in, LocalState& local) const override;
    void Combine(std::unique_ptr<LocalState> local) override;
    void Finalize(QueryContext& ctx) override;
    [[nodiscard]] bool Done() const override;

    // The rows Finalize built, moved out.
    [[nodiscard]] std::vector<Tuple> TakeTuples();

private:
    Schema schema_;
    std::vector<uint32_t> slots_;  // per output column, the batch slot holding it
    std::vector<VType> types_;     // per output column
    std::vector<const LimitCounter*> limits_;
    std::vector<std::unique_ptr<LocalState>> parts_;  // one per worker, after Combine
    std::vector<Tuple> tuples_;
};

}  // namespace shilmandb::parallel
