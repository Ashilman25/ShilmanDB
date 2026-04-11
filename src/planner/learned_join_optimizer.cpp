#ifdef SHILMANDB_HAS_LIBTORCH

#include "planner/learned_join_optimizer.hpp"
#include <cassert>

namespace shilmandb {

LearnedJoinOptimizer::LearnedJoinOptimizer(const std::string& model_path) {
    model_ = torch::jit::load(model_path);
    model_.eval();
}

auto LearnedJoinOptimizer::PredictJoinOrder(const std::vector<float>& features, int num_tables) -> std::vector<int> {
    assert(num_tables > 0 && num_tables <= 6 && "PredictJoinOrder supports 1-6 tables");

    auto input = torch::from_blob(
        const_cast<float*>(features.data()),
        {1, static_cast<int64_t>(features.size())},
        torch::kFloat32
    ).clone();

    c10::InferenceMode guard;
    auto output = model_.forward({input}).toTensor();

    auto scores = output.slice(/*dim=*/1, /*start=*/0, /*end=*/num_tables);
    auto sorted_indices = scores.argsort(/*dim=*/1, /*descending=*/true);
    auto accessor = sorted_indices.accessor<int64_t, 2>();

    std::vector<int> order;
    order.reserve(num_tables);
    for (int i = 0; i < num_tables; ++i) {
        order.push_back(static_cast<int>(accessor[0][i]));
    }
    return order;
}

}  // namespace shilmandb

#endif  // SHILMANDB_HAS_LIBTORCH
