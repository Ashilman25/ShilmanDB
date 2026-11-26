#include "executor/parallel/temp_file.hpp"
#include "common/exception.hpp"
#include "executor/parallel/memory_budget.hpp"
#include "storage/columnar/crc32c.hpp"
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <utility>

namespace shilmandb::parallel {

namespace {

// Set only by tests: every write fails with this errno while it's not 0.
std::atomic<int> test_write_errno{0};

// Set only by tests, while no other thread opens a temp file.
std::function<void(const std::string&)> test_open_hook;

// Makes each file name in this process unique.
std::atomic<uint64_t> next_file_id{0};

// The most one pwrite or pread asks for; macOS refuses calls past INT_MAX bytes.
constexpr size_t kMaxIoBytes = size_t{1} << 30;

// How many string lengths are encoded at a time.
constexpr size_t kLengthChunk = 1024;

// The error for any failure to make, open or write a temp file.
[[noreturn]] void ThrowNoSpace(const std::string& dir, int error_number) {
    throw DatabaseException("out of temp space in " + dir + " while spilling: " + std::strerror(error_number));
}

// The error for a block that fails its checks.
[[noreturn]] void ThrowDamaged(const std::string& what) {
    throw DatabaseException("damaged temp file: " + what);
}

// Makes one directory; fine if it's already one, even one another worker just made.
void MakeOneDirectory(const std::string& path, const std::string& dir) {
    struct stat st {};
    if (::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) return;
    if (::mkdir(path.c_str(), 0700) == 0) return;
    const int error_number = errno;
    if (error_number != EEXIST) ThrowNoSpace(dir, error_number);
    if (::stat(path.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) ThrowNoSpace(dir, ENOTDIR);
}

// Makes dir and any missing parents.
void MakeDirectory(const std::string& dir) {
    for (size_t slash = dir.find('/', 1); slash != std::string::npos; slash = dir.find('/', slash + 1)) {
        MakeOneDirectory(dir.substr(0, slash), dir);
    }
    MakeOneDirectory(dir, dir);
}

// Adds the time it's alive to the counters' nanos, if there are counters.
class SpillTimer {
public:
    explicit SpillTimer(SpillCounters* counters) : counters_(counters) {
        if (counters_ != nullptr) start_ = std::chrono::steady_clock::now();
    }
    ~SpillTimer() {
        if (counters_ == nullptr) return;
        const auto spent = std::chrono::steady_clock::now() - start_;
        counters_->nanos.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(spent).count(), std::memory_order_relaxed);
    }
    SpillTimer(const SpillTimer&) = delete;
    SpillTimer& operator=(const SpillTimer&) = delete;

private:
    SpillCounters* counters_;
    std::chrono::steady_clock::time_point start_{};
};

// Bytes per value in a fixed-width array.
size_t WidthOf(ValueArray a) {
    switch (a) {
        case ValueArray::I128: return sizeof(__int128);
        case ValueArray::I64: return sizeof(int64_t);
        case ValueArray::F64: return sizeof(double);
        case ValueArray::STR: break;
    }
    return 0;
}

// Appends rows values of type T, copied from bytes, to v.
template <typename T>
void AppendRaw(std::vector<T>& v, const char* bytes, uint64_t rows) {
    if (rows == 0) return;
    const size_t old = v.size();
    v.resize(old + rows);
    std::memcpy(v.data() + old, bytes, rows * sizeof(T));
}

}  // namespace

// Closing also frees the unlinked file's space.
TempFile::Descriptor::~Descriptor() {
    if (fd_ >= 0) ::close(fd_);
}

// Closes what it held, then takes other's descriptor.
TempFile::Descriptor& TempFile::Descriptor::operator=(Descriptor&& other) noexcept {
    if (this != &other) {
        if (fd_ >= 0) ::close(fd_);
        fd_ = std::exchange(other.fd_, -1);
    }
    return *this;
}

// Opens a new file only this process can name, then unlinks it before any write.
// If that fails, unlink is tried once more so no named file is left.
TempFile::TempFile(const std::string& dir, SpillCounters* counters) : dir_(dir), counters_(counters) {
    MakeDirectory(dir_);
    std::string path;
    int fd = -1;
    for (int attempt = 0; fd < 0; ++attempt) {
        path = dir_ + "/shilmandb_spill_" + std::to_string(::getpid()) + "_" + std::to_string(next_file_id.fetch_add(1)) + ".tmp";
        fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd >= 0) break;
        const int error_number = errno;
        const bool retry = error_number == EINTR || (error_number == EEXIST && attempt < 100);
        if (!retry) ThrowNoSpace(dir_, error_number);
    }
    fd_ = Descriptor(fd);
    try {
        if (test_open_hook) test_open_hook(path);
        if (::unlink(path.c_str()) != 0) {
            const int error_number = errno;
            ThrowNoSpace(dir_, error_number);
        }
    } catch (...) {
        (void)::unlink(path.c_str());
        throw;
    }
    if (counters_ != nullptr) counters_->files.fetch_add(1, std::memory_order_relaxed);
}

// Sizes the payload first, since the header carries it, then streams each column.
uint64_t TempFile::WriteBlock(const std::vector<const ColumnBuffer*>& columns, const std::vector<ValueArray>& arrays, size_t first, size_t n) {
    if (columns.size() != arrays.size()) throw DatabaseException("a temp block needs one array kind per column");
    uint64_t payload = 0;
    for (size_t c = 0; c < columns.size(); ++c) {
        if (arrays[c] != ValueArray::STR) {
            payload += WidthOf(arrays[c]) * n;
            continue;
        }
        assert(first + n <= columns[c]->str.size() && "temp block rows past the column");
        payload += sizeof(uint32_t) * n;
        for (size_t r = first; r < first + n; ++r) payload += columns[c]->str[r].size;
    }

    const uint64_t start = Bytes();
    BeginBlock(static_cast<uint32_t>(columns.size()), n, payload);
    for (size_t c = 0; c < columns.size(); ++c) {
        const ColumnBuffer& col = *columns[c];
        switch (arrays[c]) {
            case ValueArray::I64:
                assert(first + n <= col.i64.size() && "temp block rows past the column");
                Put(col.i64.data() + first, n * sizeof(int64_t));
                break;
            case ValueArray::I128:
                assert(first + n <= col.i128.size() && "temp block rows past the column");
                Put(col.i128.data() + first, n * sizeof(__int128));
                break;
            case ValueArray::F64:
                assert(first + n <= col.f64.size() && "temp block rows past the column");
                Put(col.f64.data() + first, n * sizeof(double));
                break;
            case ValueArray::STR: {
                uint32_t lengths[kLengthChunk];
                for (size_t r = first; r < first + n; r += kLengthChunk) {
                    const size_t k = std::min(kLengthChunk, first + n - r);
                    for (size_t i = 0; i < k; ++i) lengths[i] = col.str[r + i].size;
                    Put(lengths, k * sizeof(uint32_t));
                }
                for (size_t r = first; r < first + n; ++r) Put(col.str[r].data, col.str[r].size);
                break;
            }
        }
    }
    EndBlock();
    return start;
}

// The pieces are already in payload order, so they go out as they are.
uint64_t TempFile::WriteEncodedBlock(uint32_t columns, uint64_t rows, const std::vector<std::vector<char>>& pieces) {
    uint64_t payload = 0;
    for (const auto& piece : pieces) payload += piece.size();
    const uint64_t start = Bytes();
    BeginBlock(columns, rows, payload);
    for (const auto& piece : pieces) Put(piece.data(), piece.size());
    EndBlock();
    return start;
}

// The header goes out with a zero CRC; EndBlock fills it in.
void TempFile::BeginBlock(uint32_t columns, uint64_t rows, uint64_t payload_bytes) {
    assert(!in_block_ && "a temp block is already open");
    TempBlockHeader header{0, columns, rows, payload_bytes};
    block_start_ = Bytes();
    block_left_ = payload_bytes;
    block_crc_ = Crc32c(reinterpret_cast<const char*>(&header) + sizeof(header.crc), sizeof(header) - sizeof(header.crc));
    in_block_ = true;

    if (buffer_.size() + sizeof(header) > buffer_bytes_) FlushBuffer();
    if (sizeof(header) <= buffer_bytes_) {
        if (buffer_.capacity() < buffer_bytes_) buffer_.reserve(buffer_bytes_);
        const char* p = reinterpret_cast<const char*>(&header);
        buffer_.insert(buffer_.end(), p, p + sizeof(header));
    } else {
        WriteAt(&header, sizeof(header), flushed_);
        flushed_ += sizeof(header);
    }
}

// Adds payload bytes to the checksum and the buffer; bigger pieces go straight out.
void TempFile::Put(const void* data, size_t n) {
    assert(in_block_ && n <= block_left_ && "temp block payload longer than its header says");
    if (n == 0) return;
    block_crc_ = Crc32c(data, n, block_crc_);
    block_left_ -= n;
    if (buffer_.size() + n > buffer_bytes_) FlushBuffer();
    if (n <= buffer_bytes_) {
        if (buffer_.capacity() < buffer_bytes_) buffer_.reserve(buffer_bytes_);
        const char* p = static_cast<const char*>(data);
        buffer_.insert(buffer_.end(), p, p + n);
    } else {
        WriteAt(data, n, flushed_);
        flushed_ += n;
    }
}

// Fills in the header's CRC, in the buffer if it's still there, else on disk.
void TempFile::EndBlock() {
    assert(in_block_ && block_left_ == 0 && "temp block payload shorter than its header says");
    in_block_ = false;
    if (block_start_ >= flushed_) {
        std::memcpy(buffer_.data() + (block_start_ - flushed_), &block_crc_, sizeof(block_crc_));
    } else {
        WriteAt(&block_crc_, sizeof(block_crc_), block_start_);
    }
    if (counters_ != nullptr) counters_->bytes_written.fetch_add(Bytes() - block_start_, std::memory_order_relaxed);
}

// Writing may start again later; the buffer comes back when needed.
void TempFile::FinishWriting() {
    assert(!in_block_ && "a temp block is still open");
    FlushBuffer();
    std::vector<char>().swap(buffer_);
}

// Writes out what's buffered first, so no bytes are lost.
void TempFile::SetWriteBuffer(size_t bytes) {
    assert(!in_block_ && "a temp block is still open");
    FlushBuffer();
    std::vector<char>().swap(buffer_);
    buffer_bytes_ = bytes;
}

// Writes the buffer at the end of the file and empties it.
void TempFile::FlushBuffer() {
    if (buffer_.empty()) return;
    WriteAt(buffer_.data(), buffer_.size(), flushed_);
    flushed_ += buffer_.size();
    buffer_.clear();
}

// Writes every byte, past short writes and interrupts; any failure is the out-of-space error.
void TempFile::WriteAt(const void* data, size_t n, uint64_t offset) {
    SpillTimer timer(counters_);
    const char* p = static_cast<const char*>(data);
    while (n > 0) {
        const int fail = test_write_errno.load(std::memory_order_relaxed);
        ssize_t wrote = -1;
        if (fail != 0) {
            errno = fail;
        } else {
            wrote = ::pwrite(fd_.Get(), p, std::min(n, kMaxIoBytes), static_cast<off_t>(offset));
        }
        if (wrote < 0) {
            const int error_number = errno;
            if (error_number == EINTR) continue;
            ThrowNoSpace(dir_, error_number);
        }
        if (wrote == 0) ThrowNoSpace(dir_, ENOSPC);
        p += wrote;
        n -= static_cast<size_t>(wrote);
        offset += static_cast<uint64_t>(wrote);
    }
}

// Reads every byte, past short reads and interrupts; reaching the end is damage.
void TempFile::ReadAt(void* data, size_t n, uint64_t offset) const {
    SpillTimer timer(counters_);
    char* p = static_cast<char*>(data);
    while (n > 0) {
        const ssize_t got = ::pread(fd_.Get(), p, std::min(n, kMaxIoBytes), static_cast<off_t>(offset));
        if (got < 0) {
            const int error_number = errno;
            if (error_number == EINTR) continue;
            ThrowDamaged(std::string("can't read: ") + std::strerror(error_number));
        }
        if (got == 0) ThrowDamaged("short read");
        p += got;
        n -= static_cast<size_t>(got);
        offset += static_cast<uint64_t>(got);
    }
}

// Checks the header against the file before allocating, then the CRC, then the layout.
bool TempFile::ReadBlock(uint64_t* offset, std::vector<ColumnBuffer>& out, const std::vector<ValueArray>& arrays, StringArena& arena) const {
    if (in_block_ || !buffer_.empty()) throw DatabaseException("temp file read before its writes were finished");
    const uint64_t size = flushed_;
    if (*offset == size) return false;
    if (*offset > size || size - *offset < sizeof(TempBlockHeader)) ThrowDamaged("block header past the end");

    TempBlockHeader header{};
    ReadAt(&header, sizeof(header), *offset);
    if (header.payload_bytes > size - *offset - sizeof(header)) ThrowDamaged("block past the end");
    std::vector<char> payload(header.payload_bytes);
    ReadAt(payload.data(), payload.size(), *offset + sizeof(header));

    uint32_t crc = Crc32c(reinterpret_cast<const char*>(&header) + sizeof(header.crc), sizeof(header) - sizeof(header.crc));
    crc = Crc32c(payload.data(), payload.size(), crc);
    if (crc != header.crc) ThrowDamaged("checksum mismatch");
    if (header.columns != arrays.size()) {
        ThrowDamaged("block has " + std::to_string(header.columns) + " columns, expected " + std::to_string(arrays.size()));
    }

    // Walk the layout once before touching out, so a bad block appends nothing.
    const uint64_t rows = header.rows;
    const uint64_t total = header.payload_bytes;
    const char* p = payload.data();
    uint64_t pos = 0;
    
    for (const ValueArray a : arrays) {
        const size_t width = a == ValueArray::STR ? sizeof(uint32_t) : WidthOf(a);
        if (rows > (total - pos) / width) ThrowDamaged("column past the block");
        const char* lengths = p + pos;
        pos += rows * width;
        if (a != ValueArray::STR) continue;
        uint64_t bytes = 0;
        for (uint64_t r = 0; r < rows; ++r) {
            uint32_t len = 0;
            std::memcpy(&len, lengths + r * sizeof(uint32_t), sizeof(len));
            bytes += len;
        }
        if (bytes > total - pos) ThrowDamaged("strings past the block");
        pos += bytes;
    }
    if (pos != total) ThrowDamaged("block has extra bytes");

    out.resize(arrays.size());
    for (size_t c = 0; c < arrays.size(); ++c) {
        ColumnBuffer& col = out[c];
        switch (arrays[c]) {
            case ValueArray::I64: AppendRaw(col.i64, p, rows); break;
            case ValueArray::I128: AppendRaw(col.i128, p, rows); break;
            case ValueArray::F64: AppendRaw(col.f64, p, rows); break;
            case ValueArray::STR: {
                const char* lengths = p;
                const char* bytes = p + rows * sizeof(uint32_t);
                // Doubling, since a caller may append many blocks to one buffer.
                if (col.str.capacity() < col.str.size() + rows) col.str.reserve(std::max<size_t>(col.str.size() + rows, 2 * col.str.capacity()));
                for (uint64_t r = 0; r < rows; ++r) {
                    uint32_t len = 0;
                    std::memcpy(&len, lengths + r * sizeof(uint32_t), sizeof(len));
                    col.str.push_back(arena.Copy(std::string_view(bytes, len)));
                    bytes += len;
                }
                p = bytes;
                continue;
            }
        }
        p += rows * WidthOf(arrays[c]);
    }

    if (counters_ != nullptr) counters_->bytes_read.fetch_add(sizeof(header) + header.payload_bytes, std::memory_order_relaxed);
    *offset += sizeof(header) + header.payload_bytes;
    return true;
}

// Read by every write, from any thread.
void TempFile::FailWritesForTest(int error_number) {
    test_write_errno.store(error_number, std::memory_order_relaxed);
}

// Set only while no other thread opens a temp file.
void TempFile::SetOpenHookForTest(std::function<void(const std::string& path)> hook) {
    test_open_hook = std::move(hook);
}

// Breakers call this before opening their first spill file.
const std::string& SpillDirectory(const std::string* dir) {
    assert(dir != nullptr && "spilling needs a temp directory");
    if (dir == nullptr) throw DatabaseException("spilling needs a temp directory");
    return *dir;
}

}  // namespace shilmandb::parallel
