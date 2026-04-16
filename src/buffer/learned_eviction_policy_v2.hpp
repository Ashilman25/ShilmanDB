#pragma once

#ifdef SHILMANDB_HAS_LIBTORCH

#include "buffer/eviction_policy.hpp"
#include <torch/script.h>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

namespace shilmandb {

class LearnedEvictionPolicyV2 : public EvictionPolicy {
public:
    explicit LearnedEvictionPolicyV2(const std::string& model_path);

    void RecordAccess(frame_id_t frame_id) override;
    void RecordAccess(frame_id_t frame_id, page_id_t page_id) override;
    void SetEvictable(frame_id_t frame_id, bool evictable) override;

    [[nodiscard]] std::optional<frame_id_t> Evict() override;
    void Remove(frame_id_t frame_id) override;

    [[nodiscard]] size_t Size() const override;

private:
    static constexpr int kNumFeatures = 4;

    torch::jit::script::Module model_;
    uint64_t global_clock_{0};

    // Per-frame behavioral tracking
    std::unordered_map<frame_id_t, uint64_t> last_access_ts_;
    std::unordered_map<frame_id_t, uint64_t> second_last_access_ts_;
    std::unordered_map<frame_id_t, uint32_t> access_count_;
    std::unordered_set<frame_id_t> evictable_frames_;
};

}  // namespace shilmandb

#endif  // SHILMANDB_HAS_LIBTORCH
