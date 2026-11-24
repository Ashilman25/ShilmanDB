#pragma once
#include "executor/parallel/column_buffer.hpp"
#include "storage/columnar/string_arena.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace shilmandb::parallel {

struct SpillCounters;

// Starts every block. The CRC32C covers the rest of the header, then the payload.
// The payload is each column in turn: I64, I128 and F64 as raw arrays,
// strings as their uint32_t lengths, then their bytes.
struct TempBlockHeader {
    uint32_t crc;
    uint32_t columns;
    uint64_t rows;
    uint64_t payload_bytes;
};
static_assert(sizeof(TempBlockHeader) == 24, "temp block header is 24 bytes");
static_assert(offsetof(TempBlockHeader, columns) == 4 && offsetof(TempBlockHeader, rows) == 8 && offsetof(TempBlockHeader, payload_bytes) == 16, "temp block header layout");

// A spill file in dir: unlinked once opened, so it vanishes when closed. Never synced.
// One thread writes it; once writing is finished, any number of threads may read it.
class TempFile {
public:
    static constexpr size_t kWriteBufferBytes = 64 * 1024;

    // Creates dir if it's missing; throws the out-of-temp-space error if it can't.
    TempFile(const std::string& dir, SpillCounters* counters);
    ~TempFile() = default;

    TempFile(TempFile&&) noexcept = default;
    TempFile& operator=(TempFile&&) noexcept = default;
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    // Writes rows [first, first + n) as one block; returns the block's offset.
    // Strings are copied, so the columns may die once this returns.
    uint64_t WriteBlock(const std::vector<const ColumnBuffer*>& columns, const std::vector<ValueArray>& arrays, size_t first, size_t n);

    // Writes out the write buffer and frees it. Call before reading.
    void FinishWriting();

    // Appends the block at *offset to out (a buffer per column); moves *offset past it.
    // Strings go into arena. False at the end; bad blocks throw "damaged temp file: ...".
    [[nodiscard]] bool ReadBlock(uint64_t* offset, std::vector<ColumnBuffer>& out, const std::vector<ValueArray>& arrays, StringArena& arena) const;

    // Bytes written so far, buffered ones included.
    [[nodiscard]] uint64_t Bytes() const { return flushed_ + buffer_.size(); }

    // Sets the write buffer's size (default kWriteBufferBytes); 0 writes every piece straight out.
    void SetWriteBuffer(size_t bytes);

    // Makes every write fail with this errno until set back to 0; for tests.
    static void FailWritesForTest(int error_number);

    // Called with each new file's path before it's unlinked; empty turns it off. For tests.
    static void SetOpenHookForTest(std::function<void(const std::string& path)> hook);

private:
    friend class PartitionedWriter;

    // Closes its descriptor when destroyed.
    class Descriptor {
    public:
        Descriptor() = default;
        explicit Descriptor(int fd) : fd_(fd) {}
        ~Descriptor();
        Descriptor(Descriptor&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
        Descriptor& operator=(Descriptor&& other) noexcept;
        Descriptor(const Descriptor&) = delete;
        Descriptor& operator=(const Descriptor&) = delete;
        [[nodiscard]] int Get() const { return fd_; }

    private:
        int fd_{-1};
    };

    // Writes one block from pieces already in payload form; returns where it starts.
    uint64_t WriteEncodedBlock(uint32_t columns, uint64_t rows, const std::vector<std::vector<char>>& pieces);

    // A block is written as BeginBlock, Put for each payload piece, then EndBlock.
    void BeginBlock(uint32_t columns, uint64_t rows, uint64_t payload_bytes);
    void Put(const void* data, size_t n);
    void EndBlock();

    void FlushBuffer();
    void WriteAt(const void* data, size_t n, uint64_t offset);
    void ReadAt(void* data, size_t n, uint64_t offset) const;

    std::string dir_;
    SpillCounters* counters_{nullptr};
    Descriptor fd_;
    uint64_t flushed_{0};     // bytes on disk; the buffer follows them
    std::vector<char> buffer_;
    size_t buffer_bytes_{kWriteBufferBytes};
    uint64_t block_start_{0};  // the block being written
    uint64_t block_left_{0};   // its payload bytes still to come
    uint32_t block_crc_{0};
    bool in_block_{false};
};

// The query's temp directory for a spill; spilling without one is a bug
// (a Debug assert, else DatabaseException).
[[nodiscard]] const std::string& SpillDirectory(const std::string* dir);

}  // namespace shilmandb::parallel
