#pragma once
#include "buffer/eviction_policy.hpp"
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace shilmandb {

enum class EvictionPolicyKind { LRU, CLOCK, LRU2, ARC, LIRS, LEARNED_V1, LEARNED_V2 };

// "lru", "clock", "lru2", "arc", "lirs", "learned-v1", "learned-v2" (exact, lower case).
// Anything else gives nullopt.
[[nodiscard]] std::optional<EvictionPolicyKind> ParseEvictionPolicyKind(std::string_view name);
[[nodiscard]] std::string_view EvictionPolicyKindName(EvictionPolicyKind kind);

// Builds a policy for a pool of pool_size frames.
// Learned kinds need an ML build and a model path; otherwise DatabaseException.
[[nodiscard]] std::unique_ptr<EvictionPolicy> MakeEvictionPolicy(EvictionPolicyKind kind, size_t pool_size, const std::string& model_path = "");

}  // namespace shilmandb
