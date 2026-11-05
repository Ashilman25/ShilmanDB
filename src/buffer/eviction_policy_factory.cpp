#include "buffer/eviction_policy_factory.hpp"

#include "buffer/arc_eviction_policy.hpp"
#include "buffer/clock_eviction_policy.hpp"
#include "buffer/lirs_eviction_policy.hpp"
#include "buffer/lru2_eviction_policy.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "common/exception.hpp"

#ifdef SHILMANDB_HAS_LIBTORCH
#include "buffer/learned_eviction_policy.hpp"
#include "buffer/learned_eviction_policy_v2.hpp"
#endif

#include <algorithm>
#include <array>
#include <utility>

namespace shilmandb {

namespace {

constexpr std::array<std::pair<EvictionPolicyKind, std::string_view>, 7> kNames{{
    {EvictionPolicyKind::LRU, "lru"},
    {EvictionPolicyKind::CLOCK, "clock"},
    {EvictionPolicyKind::LRU2, "lru2"},
    {EvictionPolicyKind::ARC, "arc"},
    {EvictionPolicyKind::LIRS, "lirs"},
    {EvictionPolicyKind::LEARNED_V1, "learned-v1"},
    {EvictionPolicyKind::LEARNED_V2, "learned-v2"},
}};

std::unique_ptr<EvictionPolicy> MakeLearnedPolicy(EvictionPolicyKind kind, const std::string& model_path) {
    const std::string name(EvictionPolicyKindName(kind));
#ifdef SHILMANDB_HAS_LIBTORCH
    if (model_path.empty()) {
        throw DatabaseException(name + " eviction needs a model path");
    }
    if (kind == EvictionPolicyKind::LEARNED_V2) {
        return std::make_unique<LearnedEvictionPolicyV2>(model_path);
    }
    return std::make_unique<LearnedEvictionPolicy>(model_path);
#else
    (void)model_path;
    throw DatabaseException(name + " eviction needs a build with LibTorch (SHILMANDB_ENABLE_ML=ON)");
#endif
}

}  // namespace

std::optional<EvictionPolicyKind> ParseEvictionPolicyKind(std::string_view name) {
    const auto it = std::find_if(kNames.begin(), kNames.end(), [&](const auto& entry) { return entry.second == name; });
    if (it == kNames.end()) return std::nullopt;
    return it->first;
}

std::string_view EvictionPolicyKindName(EvictionPolicyKind kind) {
    const auto it = std::find_if(kNames.begin(), kNames.end(), [&](const auto& entry) { return entry.first == kind; });
    return it != kNames.end() ? it->second : "unknown";
}

std::unique_ptr<EvictionPolicy> MakeEvictionPolicy(EvictionPolicyKind kind, size_t pool_size, const std::string& model_path) {
    switch (kind) {
        case EvictionPolicyKind::LRU: return std::make_unique<LRUEvictionPolicy>(pool_size);
        case EvictionPolicyKind::CLOCK: return std::make_unique<ClockEvictionPolicy>(pool_size);
        case EvictionPolicyKind::LRU2: return std::make_unique<Lru2EvictionPolicy>(pool_size);
        case EvictionPolicyKind::ARC: return std::make_unique<ArcEvictionPolicy>(pool_size);
        case EvictionPolicyKind::LIRS: return std::make_unique<LirsEvictionPolicy>(pool_size);
        case EvictionPolicyKind::LEARNED_V1:
        case EvictionPolicyKind::LEARNED_V2: return MakeLearnedPolicy(kind, model_path);
    }
    throw DatabaseException("unknown eviction policy kind");
}

}  // namespace shilmandb
