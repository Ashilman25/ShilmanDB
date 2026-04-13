#pragma once

#ifdef SHILMANDB_HAS_LIBTORCH

#include "buffer/eviction_policy.hpp"
#include <torch/script.h>
#include <string>
#include <deque>
#include <unordered_map>
#include <unordered_set>

namespace shilmandb {

class LearnedEvictionPolicy : public EvictionPolicy {
public:
    explicit LearnedEvictionPolicy(const std::string& model_path);

    void RecordAccess(frame_id_t frame_id) override;
    void RecordAccess(frame_id_t frame_id, page_id_t page_id) override;
    void SetEvictable(frame_id_t frame_id, bool evictable) override;

    [[nodiscard]] std::optional<frame_id_t> Evict() override;
    void Remove(frame_id_t frame_id) override;

    [[nodiscard]] size_t Size() const override;

private:
    static constexpr int kWindowSize = 64;

    torch::jit::script::Module model_;
    std::deque<int64_t> access_history_;
    std::unordered_map<frame_id_t, page_id_t> frame_to_page_;
    std::unordered_map<page_id_t, frame_id_t> page_to_frame_;
    std::unordered_set<frame_id_t> evictable_frames_;
};

}  // namespace shilmandb

#endif  // SHILMANDB_HAS_LIBTORCH
