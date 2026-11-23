#pragma once
#include "executor/parallel/column_buffer.hpp"
#include "executor/parallel/key_hash.hpp"
#include "executor/parallel/temp_file.hpp"
#include "storage/columnar/string_arena.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace shilmandb::parallel {

class MemoryReservation;
struct SpillCounters;

inline constexpr int kSpillPartitionBits = 6;
inline constexpr size_t kSpillPartitions = size_t{1} << kSpillPartitionBits;

// A hash's spill partition: 6 bits of a remix keyed by the level.
// Independent of the in-memory tables' bits and of every other level.
[[nodiscard]] constexpr size_t SpillPartitionOf(uint64_t hash, int level) {
    constexpr uint64_t kSpillSeed = 0xd6e8feb86659fd93ULL;
    const uint64_t mixed = detail::HashMix(hash + kSpillSeed * static_cast<uint64_t>(level + 1));
    return static_cast<size_t>(mixed >> (64 - kSpillPartitionBits));
}

// One writer's spilled rows: its file and where each partition's blocks start.
// Reading is const, so different threads may read different partitions at once.
struct SpillFile {
    TempFile file;
    std::vector<ValueArray> arrays;                             // one per column
    std::array<std::vector<uint64_t>, kSpillPartitions> blocks;  // block offsets, in write order
    std::array<uint64_t, kSpillPartitions> rows{};
    std::array<uint64_t, kSpillPartitions> bytes{};              // block bytes, headers included

    SpillFile(TempFile f, std::vector<ValueArray> a) : file(std::move(f)), arrays(std::move(a)) {}

    // Appends a partition's rows to out (one buffer per column), strings into arena.
    void ReadPartition(size_t partition, std::vector<ColumnBuffer>& out, StringArena& arena) const;
};

// One worker's rows for kSpillPartitions partitions, buffered and written to one file.
// Each block holds rows of one partition. Used by one thread at a time.
class PartitionedWriter {
public:
    static constexpr size_t kPartitionBufferBytes = 64 * 1024;

    // reservation counts the buffers (null: not counted); it and counters must outlive the writer.
    PartitionedWriter(std::vector<ValueArray> arrays, std::string dir, SpillCounters* counters, MemoryReservation* reservation);
    ~PartitionedWriter();

    PartitionedWriter(const PartitionedWriter&) = delete;
    PartitionedWriter& operator=(const PartitionedWriter&) = delete;

    // Copies rows[0..n) of these columns into a partition's buffer; strings are copied too.
    // Full or refused buffers are written out; an oversized row is its own block.
    void Add(size_t partition, const std::vector<const ColumnBuffer*>& columns, const uint32_t* rows, size_t n);

    // Writes out every buffered row and frees the buffers; more rows may follow.
    void Flush();

    // Flushes, then hands over the file; nullopt if no row was ever added.
    [[nodiscard]] std::optional<SpillFile> TakeFile();

private:
    // One partition's rows in payload form, waiting to be written as one block.
    struct Buffer {
        std::vector<std::vector<char>> streams;  // a fixed column's values; a string column's lengths, then bytes
        uint64_t rows{0};
        size_t used{0};      // payload bytes held
        size_t capacity{0};  // room the streams hold, all of it counted
    };

    enum class Room { OK, FULL, REFUSED };

    void AddRow(size_t partition, const std::vector<const ColumnBuffer*>& columns, uint32_t row);
    [[nodiscard]] Room MakeRoom(Buffer& b);
    void WriteRowAlone(size_t partition, const std::vector<const ColumnBuffer*>& columns, uint32_t row);
    void FlushPartition(size_t partition, bool release);
    void Record(size_t partition, uint64_t at, uint64_t rows);
    TempFile& File();
    [[nodiscard]] bool Charge(size_t bytes);
    void Uncharge(size_t bytes);

    std::vector<ValueArray> arrays_;
    std::vector<size_t> first_stream_;  // per column, its first stream
    size_t stream_count_{0};
    std::string dir_;
    SpillCounters* counters_;
    MemoryReservation* reservation_;
    std::array<Buffer, kSpillPartitions> buffers_;
    std::vector<size_t> adds_;  // per stream, the bytes the current row adds
    std::vector<size_t> plan_;  // per stream, the room MakeRoom plans
    std::optional<SpillFile> spilled_;
    bool file_buffer_charged_{false};
    size_t charged_{0};
};

}  // namespace shilmandb::parallel
