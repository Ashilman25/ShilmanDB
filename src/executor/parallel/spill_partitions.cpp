#include "executor/parallel/spill_partitions.hpp"
#include "common/exception.hpp"
#include "executor/parallel/memory_budget.hpp"
#include <algorithm>
#include <cassert>
#include <utility>

namespace shilmandb::parallel {

namespace {

// The smallest room a stream grows to, so tiny rows don't reallocate often.
constexpr size_t kMinStreamBytes = 64;

// Appends n raw bytes to a stream.
void AppendBytes(std::vector<char>& stream, const void* data, size_t n) {
    const char* p = static_cast<const char*>(data);
    stream.insert(stream.end(), p, p + n);
}

// How many values a column holds, in the array its kind uses.
[[maybe_unused]] size_t RowsIn(const ColumnBuffer& col, ValueArray a) {
    switch (a) {
        case ValueArray::I64: return col.i64.size();
        case ValueArray::I128: return col.i128.size();
        case ValueArray::F64: return col.f64.size();
        case ValueArray::STR: return col.str.size();
    }
    return 0;
}

}  // namespace

// Reads the partition's blocks in the order they were written.
void SpillFile::ReadPartition(size_t partition, std::vector<ColumnBuffer>& out, StringArena& arena) const {
    out.resize(arrays.size());
    for (const uint64_t at : blocks.at(partition)) {
        uint64_t offset = at;
        const bool read = file.ReadBlock(&offset, out, arrays, arena);
        assert(read && "a recorded block offset is past the file's end");
        (void)read;
    }
}

// A fixed column has one stream; a string column has two, lengths then bytes.
PartitionedWriter::PartitionedWriter(std::vector<ValueArray> arrays, std::string dir, SpillCounters* counters, MemoryReservation* reservation)
    : arrays_(std::move(arrays)), dir_(std::move(dir)), counters_(counters), reservation_(reservation) {
    for (const ValueArray a : arrays_) {
        first_stream_.push_back(stream_count_);
        stream_count_ += a == ValueArray::STR ? 2 : 1;
    }
    for (Buffer& b : buffers_) b.streams.resize(stream_count_);
    adds_.resize(stream_count_);
}

// Gives back every byte the buffers still count; buffered rows are dropped.
PartitionedWriter::~PartitionedWriter() {
    if (reservation_ != nullptr && charged_ > 0) reservation_->Shrink(charged_);
}

// Checks the call once, then adds each row.
void PartitionedWriter::Add(size_t partition, const std::vector<const ColumnBuffer*>& columns, const uint32_t* rows, size_t n) {
    if (partition >= kSpillPartitions) throw DatabaseException("spill partition " + std::to_string(partition) + " is out of range");
    if (columns.size() != arrays_.size()) throw DatabaseException("a spilled row needs one column per array kind");
    for (size_t i = 0; i < n; ++i) AddRow(partition, columns, rows[i]);
}

// Sizes the row per stream, finds it room, then appends its values.
void PartitionedWriter::AddRow(size_t partition, const std::vector<const ColumnBuffer*>& columns, uint32_t row) {
    size_t row_bytes = 0;
    for (size_t c = 0; c < arrays_.size(); ++c) {
        const size_t s = first_stream_[c];
        assert(row < RowsIn(*columns[c], arrays_[c]) && "spilled row past the column");
        switch (arrays_[c]) {
            case ValueArray::I64: adds_[s] = sizeof(int64_t); break;
            case ValueArray::I128: adds_[s] = sizeof(__int128); break;
            case ValueArray::F64: adds_[s] = sizeof(double); break;
            case ValueArray::STR:
                adds_[s] = sizeof(uint32_t);
                adds_[s + 1] = columns[c]->str[row].size;
                row_bytes += adds_[s + 1];
                break;
        }
        row_bytes += adds_[s];
    }

    Buffer& b = buffers_[partition];
    if (row_bytes > kPartitionBufferBytes) {
        FlushPartition(partition, false);
        WriteRowAlone(partition, columns, row);
        return;
    }
    if (b.used + row_bytes > kPartitionBufferBytes) FlushPartition(partition, false);
    if (MakeRoom(b) != Room::OK) {
        // Full by fragmentation or refused: write it out, free its room, try once more.
        FlushPartition(partition, true);
        if (MakeRoom(b) != Room::OK) {
            WriteRowAlone(partition, columns, row);
            return;
        }
    }

    for (size_t c = 0; c < arrays_.size(); ++c) {
        const ColumnBuffer& col = *columns[c];
        std::vector<char>& stream = b.streams[first_stream_[c]];
        switch (arrays_[c]) {
            case ValueArray::I64: AppendBytes(stream, &col.i64[row], sizeof(int64_t)); break;
            case ValueArray::I128: AppendBytes(stream, &col.i128[row], sizeof(__int128)); break;
            case ValueArray::F64: AppendBytes(stream, &col.f64[row], sizeof(double)); break;
            case ValueArray::STR: {
                const StringRef s = col.str[row];
                AppendBytes(stream, &s.size, sizeof(uint32_t));
                AppendBytes(b.streams[first_stream_[c] + 1], s.data, s.size);
                break;
            }
        }
    }
    ++b.rows;
    b.used += row_bytes;
}

// Grows streams to fit the current row, doubling where the buffer's 64 KiB allows.
// FULL: the exact room needed won't fit beside the other streams' room.
PartitionedWriter::Room PartitionedWriter::MakeRoom(Buffer& b) {
    size_t exact = 0;
    bool grows = false;
    for (size_t s = 0; s < stream_count_; ++s) {
        const size_t need = b.streams[s].size() + adds_[s];
        exact += std::max(b.streams[s].capacity(), need);
        grows = grows || need > b.streams[s].capacity();
    }
    if (!grows) return Room::OK;
    if (exact > kPartitionBufferBytes) return Room::FULL;

    // Room for doubling comes from what's left under the cap, stream by stream.
    plan_.resize(stream_count_);
    size_t spare = kPartitionBufferBytes - exact;
    for (size_t s = 0; s < stream_count_; ++s) {
        const size_t cap = b.streams[s].capacity();
        const size_t need = b.streams[s].size() + adds_[s];
        plan_[s] = std::max(cap, need);
        if (need <= cap) continue;
        const size_t want = std::max({need, 2 * cap, kMinStreamBytes});
        const size_t extra = std::min(want - need, spare);
        plan_[s] += extra;
        spare -= extra;
    }
    const size_t total = kPartitionBufferBytes - spare;
    if (total > b.capacity && !Charge(total - b.capacity)) return Room::REFUSED;
    for (size_t s = 0; s < stream_count_; ++s) b.streams[s].reserve(plan_[s]);
    b.capacity = std::max(total, b.capacity);
    return Room::OK;
}

// A row too big for a buffer, or with no room granted; its partition's flushed.
void PartitionedWriter::WriteRowAlone(size_t partition, const std::vector<const ColumnBuffer*>& columns, uint32_t row) {
    TempFile& file = File();
    const uint64_t at = file.WriteBlock(columns, arrays_, row, 1);
    Record(partition, at, 1);
}

// Writes the partition's rows as one block; release also frees and uncounts its room.
void PartitionedWriter::FlushPartition(size_t partition, bool release) {
    Buffer& b = buffers_[partition];
    if (b.rows > 0) {
        TempFile& file = File();
        const uint64_t at = file.WriteEncodedBlock(static_cast<uint32_t>(arrays_.size()), b.rows, b.streams);
        Record(partition, at, b.rows);
        for (auto& stream : b.streams) stream.clear();
        b.rows = 0;
        b.used = 0;
    }
    if (release && b.capacity > 0) {
        for (auto& stream : b.streams) std::vector<char>().swap(stream);
        Uncharge(b.capacity);
        b.capacity = 0;
    }
}

// Notes a block just written for a partition.
void PartitionedWriter::Record(size_t partition, uint64_t at, uint64_t rows) {
    spilled_->blocks[partition].push_back(at);
    spilled_->rows[partition] += rows;
    spilled_->bytes[partition] += spilled_->file.Bytes() - at;
}

// Opens the file on first use; its write buffer is counted, or it writes unbuffered.
TempFile& PartitionedWriter::File() {
    if (!spilled_) spilled_.emplace(TempFile(dir_, counters_), arrays_);
    if (!file_buffer_charged_) {
        file_buffer_charged_ = Charge(TempFile::kWriteBufferBytes);
        spilled_->file.SetWriteBuffer(file_buffer_charged_ ? TempFile::kWriteBufferBytes : 0);
    }
    return spilled_->file;
}

// Writes every partition's rows, frees every buffer and the file's write buffer.
void PartitionedWriter::Flush() {
    for (size_t p = 0; p < kSpillPartitions; ++p) FlushPartition(p, true);
    if (!spilled_) return;
    spilled_->file.FinishWriting();
    if (file_buffer_charged_) {
        Uncharge(TempFile::kWriteBufferBytes);
        file_buffer_charged_ = false;
    }
}

// Leaves the writer empty, ready to spill into a new file.
std::optional<SpillFile> PartitionedWriter::TakeFile() {
    Flush();
    std::optional<SpillFile> out = std::move(spilled_);
    spilled_.reset();
    return out;
}

// True if the reservation took the bytes, or there's no reservation.
bool PartitionedWriter::Charge(size_t bytes) {
    if (reservation_ == nullptr) return true;
    if (!reservation_->Grow(bytes)) return false;
    charged_ += bytes;
    return true;
}

// Hands bytes back to the reservation.
void PartitionedWriter::Uncharge(size_t bytes) {
    if (reservation_ == nullptr) return;
    reservation_->Shrink(bytes);
    charged_ -= bytes;
}

}  // namespace shilmandb::parallel
