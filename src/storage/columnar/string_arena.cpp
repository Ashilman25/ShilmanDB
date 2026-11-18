#include "storage/columnar/string_arena.hpp"
#include "common/exception.hpp"
#include <cstring>
#include <limits>
#include <string>
#include <utility>

namespace shilmandb {

namespace {

// Empty copies point here, so their data is never null.
constexpr char kEmpty[] = "";

}  // namespace

// Copies the bytes into stable storage; empty strings take no space.
StringRef StringArena::Copy(std::string_view s) {
    if (s.empty()) {
        return {kEmpty, 0};
    }
    if (s.size() > std::numeric_limits<uint32_t>::max()) {
        throw DatabaseException("String of " + std::to_string(s.size()) + " bytes is too long to copy");
    }
    char* dest = Allocate(s.size());
    std::memcpy(dest, s.data(), s.size());
    bytes_used_ += s.size();
    return {dest, static_cast<uint32_t>(s.size())};
}

// New blocks are added, never grown, so earlier copies keep their address.
char* StringArena::Allocate(size_t n) {
    if (n > kBlockSize) {
        std::unique_ptr<char[]> own(new char[n]);
        large_.push_back(std::move(own));
        return large_.back().get();
    }
    if (blocks_.empty() || kBlockSize - block_used_ < n) {
        std::unique_ptr<char[]> block(new char[kBlockSize]);
        blocks_.push_back(std::move(block));
        block_used_ = 0;
    }
    char* dest = blocks_.back().get() + block_used_;
    block_used_ += n;
    return dest;
}

// Frees every block but the first, which starts filling again from the top.
void StringArena::Clear() {
    large_.clear();
    if (blocks_.size() > 1) {
        blocks_.erase(blocks_.begin() + 1, blocks_.end());
    }
    block_used_ = 0;
    bytes_used_ = 0;
}

}  // namespace shilmandb
