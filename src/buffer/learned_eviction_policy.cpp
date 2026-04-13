#ifdef SHILMANDB_HAS_LIBTORCH

#include "buffer/learned_eviction_policy.hpp"
#include "common/exception.hpp"
#include <algorithm>
#include <cassert>
#include <vector>

namespace shilmandb {

LearnedEvictionPolicy::LearnedEvictionPolicy(const std::string& model_path) {
    try {
        model_ = torch::jit::load(model_path);
        model_.eval();
    } catch (const std::exception& e) {
        throw DatabaseException(
            "Failed to load eviction model from '" + model_path + "': " + e.what());
    }
}

void LearnedEvictionPolicy::RecordAccess(frame_id_t /*frame_id*/) {
    //override func
}

void LearnedEvictionPolicy::RecordAccess(frame_id_t frame_id, page_id_t page_id) {
    access_history_.push_back(static_cast<int64_t>(page_id));
    if (access_history_.size() > static_cast<size_t>(kWindowSize)) {
        access_history_.pop_front();
    }

    frame_to_page_[frame_id] = page_id;
    page_to_frame_[page_id] = frame_id;
}

void LearnedEvictionPolicy::SetEvictable(frame_id_t frame_id, bool evictable) {
    if (evictable) {
        evictable_frames_.insert(frame_id);
    } else {
        evictable_frames_.erase(frame_id);
    }
}

auto LearnedEvictionPolicy::Evict() -> std::optional<frame_id_t> {
    if (evictable_frames_.empty()) {
        return std::nullopt;
    }

    //evictable candidates
    std::vector<frame_id_t> candidates(evictable_frames_.begin(), evictable_frames_.end());
    auto n = static_cast<int64_t>(candidates.size());

    constexpr int prefix_len = kWindowSize - 1;
    std::vector<int64_t> prefix(prefix_len, 0);

    auto hist_size = static_cast<int>(access_history_.size());
    int copy_count = std::min(hist_size, prefix_len);
    int prefix_offset = prefix_len - copy_count;
    int hist_offset = hist_size - copy_count;

    for (int j = 0; j < copy_count; ++j) {
        prefix[static_cast<size_t>(prefix_offset + j)] = access_history_[static_cast<size_t>(hist_offset + j)];
    }

    auto batch = torch::zeros({n, kWindowSize}, torch::kInt64);
    auto accessor = batch.accessor<int64_t, 2>();

    for (int64_t i = 0; i < n; ++i) {
        for (int j = 0; j < prefix_len; ++j) {
            accessor[i][j] = prefix[static_cast<size_t>(j)];
        }

        auto it = frame_to_page_.find(candidates[static_cast<size_t>(i)]);
        assert(it != frame_to_page_.end() && "Evictable frame missing from frame_to_page mapping");
        accessor[i][prefix_len] = static_cast<int64_t>(it->second);
    }


    try {
        c10::InferenceMode guard;
        auto output = model_.forward({batch}).toTensor().flatten();
        auto best_idx = output.argmax(0).item<int64_t>();

        auto victim = candidates[static_cast<size_t>(best_idx)];
        evictable_frames_.erase(victim);

        auto map_it = frame_to_page_.find(victim);
        if (map_it != frame_to_page_.end()) {
            page_to_frame_.erase(map_it->second);
            frame_to_page_.erase(map_it);
        }
        return victim;

    } catch (const std::exception& /*e*/) {

        auto victim = *evictable_frames_.begin();
        evictable_frames_.erase(evictable_frames_.begin());

        auto map_it = frame_to_page_.find(victim);
        if (map_it != frame_to_page_.end()) {
            page_to_frame_.erase(map_it->second);
            frame_to_page_.erase(map_it);
        }
        return victim;
    }
}

void LearnedEvictionPolicy::Remove(frame_id_t frame_id) {
    evictable_frames_.erase(frame_id);

    auto it = frame_to_page_.find(frame_id);
    if (it != frame_to_page_.end()) {
        page_to_frame_.erase(it->second);
        frame_to_page_.erase(it);
    }
}

auto LearnedEvictionPolicy::Size() const -> size_t {
    return evictable_frames_.size();
}

}  // namespace shilmandb

#endif  // SHILMANDB_HAS_LIBTORCH
