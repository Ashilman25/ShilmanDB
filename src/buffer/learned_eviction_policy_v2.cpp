#ifdef SHILMANDB_HAS_LIBTORCH

#include "buffer/learned_eviction_policy_v2.hpp"
#include "common/exception.hpp"
#include <cmath>
#include <vector>

namespace shilmandb {

LearnedEvictionPolicyV2::LearnedEvictionPolicyV2(const std::string& model_path) {
    try {
        model_ = torch::jit::load(model_path);
        model_.eval();
    } catch (const std::exception& e) {
        throw DatabaseException("Failed to load V2 eviction model from '" + model_path + "': " + e.what());
    }
}

void LearnedEvictionPolicyV2::RecordAccess(frame_id_t frame_id) {
    ++global_clock_;

    auto it = last_access_ts_.find(frame_id);
    if (it != last_access_ts_.end()) {
        second_last_access_ts_[frame_id] = it->second;
    }
    // else: first access — no second_last_access_ts_ entry,

    last_access_ts_[frame_id] = global_clock_;
    access_count_[frame_id]++;
}

void LearnedEvictionPolicyV2::RecordAccess(frame_id_t frame_id, page_id_t /*page_id*/) {
    RecordAccess(frame_id);
}

void LearnedEvictionPolicyV2::SetEvictable(frame_id_t frame_id, bool evictable) {
    if (evictable) {
        evictable_frames_.insert(frame_id);
    } else {
        evictable_frames_.erase(frame_id);
    }
}

auto LearnedEvictionPolicyV2::Evict() -> std::optional<frame_id_t> {
    if (evictable_frames_.empty()) {
        return std::nullopt;
    }

    std::vector<frame_id_t> candidates(evictable_frames_.begin(), evictable_frames_.end());
    auto n = static_cast<int64_t>(candidates.size());

    try {
        c10::InferenceMode guard;
        auto features = torch::zeros({n, kNumFeatures}, torch::kFloat32);
        auto acc = features.accessor<float, 2>();

        for (int64_t i = 0; i < n; ++i) {
            auto fid = candidates[static_cast<size_t>(i)];

            auto last_it = last_access_ts_.find(fid);
            uint64_t last_ts = (last_it != last_access_ts_.end()) ? last_it->second : 0ULL;

            auto second_it = second_last_access_ts_.find(fid);
            uint64_t second_ts = (second_it != second_last_access_ts_.end()) ? second_it->second : 0ULL;

            auto count_it = access_count_.find(fid);
            uint32_t count = (count_it != access_count_.end()) ? count_it->second : 0U;

            acc[i][0] = std::log(1.0f + static_cast<float>(global_clock_ - last_ts));     // log_recency
            acc[i][1] = std::log(1.0f + static_cast<float>(count));                        // log_frequency
            acc[i][2] = std::log(1.0f + static_cast<float>(global_clock_ - second_ts));    // log_recency_2
            acc[i][3] = 0.0f;  // is_dirty (read-only workload assumption)
        }

        auto output = model_.forward({features}).toTensor().flatten();
        auto victim_idx = output.argmax(0).item<int64_t>();
        auto victim = candidates[static_cast<size_t>(victim_idx)];

        evictable_frames_.erase(victim);
        last_access_ts_.erase(victim);
        second_last_access_ts_.erase(victim);
        access_count_.erase(victim);
        return victim;

    } catch (const std::exception& /*e*/) {
        // Fallback: evict first evictable frame
        auto victim = *evictable_frames_.begin();
        evictable_frames_.erase(evictable_frames_.begin());
        last_access_ts_.erase(victim);
        second_last_access_ts_.erase(victim);
        access_count_.erase(victim);
        return victim;
    }
}

void LearnedEvictionPolicyV2::Remove(frame_id_t frame_id) {
    evictable_frames_.erase(frame_id);
    last_access_ts_.erase(frame_id);
    second_last_access_ts_.erase(frame_id);
    access_count_.erase(frame_id);
}

auto LearnedEvictionPolicyV2::Size() const -> size_t {
    return evictable_frames_.size();
}

}  // namespace shilmandb

#endif  // SHILMANDB_HAS_LIBTORCH
