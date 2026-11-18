#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace shilmandb {

// A string's bytes and length. It doesn't own the bytes.
struct StringRef {
    const char* data{nullptr};
    uint32_t size{0};
};

// The bytes a StringRef points at.
[[nodiscard]] inline std::string_view View(StringRef s) {
    return {s.data, s.size};
}

// Owns copied strings. A copy's bytes never move until Clear().
// Small values share 64 KB blocks; a bigger value gets a block of its own.
class StringArena {
public:
    static constexpr size_t kBlockSize = 64 * 1024;

    // Copies s into the arena and returns where it now lives.
    [[nodiscard]] StringRef Copy(std::string_view s);

    // Drops every copy, keeping the first block for reuse.
    void Clear();

    // Bytes copied in since the last Clear(), not the memory the blocks hold.
    [[nodiscard]] size_t BytesUsed() const { return bytes_used_; }

private:
    // Room for n bytes that stays put until Clear().
    char* Allocate(size_t n);

    std::vector<std::unique_ptr<char[]>> blocks_;  // kBlockSize each; the last one is being filled
    std::vector<std::unique_ptr<char[]>> large_;   // one per value over kBlockSize
    size_t block_used_{0};                         // bytes taken in blocks_.back()
    size_t bytes_used_{0};
};

}  // namespace shilmandb
