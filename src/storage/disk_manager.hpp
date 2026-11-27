#pragma once
#include "common/config.hpp"
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

namespace shilmandb {

// How many page reads succeeded and the total time spent in them.
struct ReadStats {
    uint64_t reads{0};
    std::chrono::nanoseconds time{0};
};

// Hears about every page write and sync, so crash tests can replay them.
// OnWrite can come from several loader threads at once, so it must be thread-safe.
class WriteObserver {
public:
    virtual ~WriteObserver() = default;
    virtual void OnWrite(page_id_t page_id, const char* data) = 0;  // called after a successful write
    virtual void OnSync() = 0;                                      // called after a successful sync
};

class DiskManager {
public:
    explicit DiskManager(const std::string& db_file);
    ~DiskManager();

    DiskManager(const DiskManager&) = delete;
    DiskManager& operator=(const DiskManager&) = delete;

    void ReadPage(page_id_t page_id, char* page_data);

    // Reads pages first .. first + n - 1 into frames[0 .. n - 1] with preadv.
    // Counts n reads; throws IOException as ReadPage does.
    void ReadPages(page_id_t first, size_t n, char* const* frames);

    void WritePage(page_id_t page_id, const char* page_data);
    page_id_t AllocatePage();
    void DeallocatePage(page_id_t page_id);

    // Hands out count pages in a row and returns the first id; safe across threads
    [[nodiscard]] page_id_t AllocatePages(uint32_t count) { return next_page_id_.fetch_add(count); }

    // Flushes every write so far to the drive: F_FULLFSYNC on macOS, fdatasync on Linux.
    // Without F_FULLFSYNC support it falls back to fsync, which may stop at the drive's cache.
    // Throws IOException; after a failure, treat every write since the last sync as lost.
    void Sync();

    // Cuts the file to num_pages pages, so the next new page is num_pages.
    // Drop buffer pool frames past the cut first, or a flush writes them back.
    void Truncate(page_id_t num_pages);

    // Tells the observer about every write and sync. nullptr (the default) is none.
    // Set it only while no writes are running; the pointer isn't guarded.
    void SetWriteObserver(WriteObserver* observer) { observer_ = observer; }

    // The file's real size in bytes, partial last page included
    [[nodiscard]] uint64_t GetFileBytes() const;

    // Reads can come from several threads at once, so the counts are atomic
    [[nodiscard]] ReadStats GetReadStats() const {
        return {read_count_.load(std::memory_order_relaxed), std::chrono::nanoseconds(read_nanos_.load(std::memory_order_relaxed))};
    }
    void ResetReadStats() {
        read_count_.store(0, std::memory_order_relaxed);
        read_nanos_.store(0, std::memory_order_relaxed);
        batch_reads_.store(0, std::memory_order_relaxed);
    }

    // How many ReadPages calls succeeded; their pages are in GetReadStats too
    [[nodiscard]] uint64_t GetBatchReadCount() const { return batch_reads_.load(std::memory_order_relaxed); }

    // Makes every read take at least this long, to stand in for slower storage. Zero is off.
    void SetReadPenalty(std::chrono::microseconds penalty) { read_penalty_ = penalty; }
    [[nodiscard]] std::chrono::microseconds GetReadPenalty() const { return read_penalty_; }

    // Skips the OS file cache for this file - macOS only, throws if it cant
    void SetDirectIO(bool on);
    [[nodiscard]] bool GetDirectIO() const { return direct_io_; }

    [[nodiscard]] page_id_t GetNumPages() const { return next_page_id_.load(); }

    const std::string& GetFileName() const { return file_name_; }

private:
    int fd_{-1};
    std::string file_name_;
    std::atomic<page_id_t> next_page_id_{0};
    std::atomic<uint64_t> read_count_{0};
    std::atomic<uint64_t> read_nanos_{0};
    std::atomic<uint64_t> batch_reads_{0};
    std::chrono::microseconds read_penalty_{0};
    bool direct_io_{false};
    WriteObserver* observer_{nullptr};
};

}  // namespace shilmandb
