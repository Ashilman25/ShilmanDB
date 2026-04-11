#pragma once

#ifdef SHILMANDB_HAS_LIBTORCH

#include <torch/script.h>
#include <string>
#include <vector>

namespace shilmandb {

class LearnedJoinOptimizer {
public:
    explicit LearnedJoinOptimizer(const std::string& model_path);

    [[nodiscard]] auto PredictJoinOrder(const std::vector<float>& features, int num_tables) -> std::vector<int>;

private:
    torch::jit::script::Module model_;
};

}  // namespace shilmandb

#endif  // SHILMANDB_HAS_LIBTORCH
