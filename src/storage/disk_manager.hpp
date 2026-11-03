#pragma once
#include "common/config.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>

namespace shilmandb {

// How many page reads succeeded and the total time spent in them.
struct ReadStats {
    uint64_t reads{0};
    std::chrono::nanoseconds time{0};
};

class DiskManager {
public:
    explicit DiskManager(const std::string& db_file);
    ~DiskManager();

    DiskManager(const DiskManager&) = delete;
    DiskManager& operator=(const DiskManager&) = delete;

    void ReadPage(page_id_t page_id, char* page_data);
    void WritePage(page_id_t page_id, const char* page_data);
    page_id_t AllocatePage();
    void DeallocatePage(page_id_t page_id);

    [[nodiscard]] ReadStats GetReadStats() const { return read_stats_; }
    void ResetReadStats() { read_stats_ = {}; }

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
    ReadStats read_stats_{};
    std::chrono::microseconds read_penalty_{0};
    bool direct_io_{false};
};

}  // namespace shilmandb
