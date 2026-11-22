#include "storage/disk_manager.hpp"
#include "common/exception.hpp"
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace shilmandb {

namespace {

constexpr off_t PageOffset(page_id_t page_id) {
    return static_cast<off_t>(page_id) * static_cast<off_t>(PAGE_SIZE);
}

}  // namespace

// Plain fd I/O: a reopened std::fstream read pages ~250x slower.
DiskManager::DiskManager(const std::string& db_file) : file_name_(db_file) {
    fd_ = ::open(db_file.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_ < 0) {
        const int err = errno;
        throw IOException("Cannot open database file: " + db_file + " (" + std::strerror(err) + ")");
    }

    struct stat st{};
    if (::fstat(fd_, &st) < 0) {
        auto err = errno;
        ::close(fd_);
        fd_ = -1;
        throw IOException("Cannot stat database file: " + db_file + " (" + std::strerror(err) + ")");
    }

    // next page id comes from the file size
    next_page_id_.store(static_cast<page_id_t>(static_cast<size_t>(st.st_size) / PAGE_SIZE));
}

DiskManager::~DiskManager() {
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

void DiskManager::ReadPage(page_id_t page_id, char* page_data) {
    if (page_id == INVALID_PAGE_ID) {
        throw IOException("Cannot read INVALID_PAGE_ID");
    }

    const auto start = std::chrono::steady_clock::now();
    size_t bytes_read = 0;
    while (bytes_read < PAGE_SIZE) {
        auto n = ::pread(fd_, page_data + bytes_read, PAGE_SIZE - bytes_read, PageOffset(page_id) + static_cast<off_t>(bytes_read));
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n < 0) {
            const int err = errno;
            throw IOException("Failed to read page " + std::to_string(page_id) + ": " + std::strerror(err));
        }
        if (n == 0) {
            break;  // end of file
        }
        bytes_read += static_cast<size_t>(n);
    }

    // anything past the end of the file reads as zeros
    std::memset(page_data + bytes_read, 0, PAGE_SIZE - bytes_read);

    // busy-wait so the read lasts at least the penalty
    if (read_penalty_ > std::chrono::microseconds{0}) {
        while (std::chrono::steady_clock::now() - start < read_penalty_) {
        }
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start);
    read_count_.fetch_add(1, std::memory_order_relaxed);
    read_nanos_.fetch_add(static_cast<uint64_t>(elapsed.count()), std::memory_order_relaxed);
}

void DiskManager::SetDirectIO(bool on) {
#ifdef __APPLE__
    if (::fcntl(fd_, F_NOCACHE, on ? 1 : 0) == -1) {
        const int err = errno;
        throw IOException(std::string("Cannot change direct I/O: ") + std::strerror(err));
    }
    direct_io_ = on;
#else
    (void)on;
    throw IOException("direct I/O is only supported on macOS");
#endif
}

void DiskManager::WritePage(page_id_t page_id, const char* page_data) {
    if (page_id == INVALID_PAGE_ID) {
        throw IOException("Cannot write to INVALID_PAGE_ID");
    }

    size_t bytes_written = 0;
    while (bytes_written < PAGE_SIZE) {
        auto n = ::pwrite(fd_, page_data + bytes_written, PAGE_SIZE - bytes_written, PageOffset(page_id) + static_cast<off_t>(bytes_written));
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n < 0) {
            const int err = errno;
            throw IOException("Failed to write page " + std::to_string(page_id) + ": " + std::strerror(err));
        }
        if (n == 0) {
            throw IOException("Failed to write page " + std::to_string(page_id) + ": no bytes written");
        }
        bytes_written += static_cast<size_t>(n);
    }

    if (observer_) {
        observer_->OnWrite(page_id, page_data);
    }
}

page_id_t DiskManager::AllocatePage() {
    return next_page_id_++;
}

void DiskManager::DeallocatePage([[maybe_unused]] page_id_t page_id) {
    // for other classes to use and such
}

// F_FULLFSYNC flushes the drive's own cache too; a plain fsync on macOS doesnt
void DiskManager::Sync() {
#ifdef __APPLE__
    int rc = ::fcntl(fd_, F_FULLFSYNC);
    if (rc == -1 && errno == ENOTSUP) {
        rc = ::fsync(fd_);  // some file systems dont support a full sync
    }
#else
    int rc = ::fdatasync(fd_);
#endif
    if (rc == -1) {
        const int err = errno;
        throw IOException("Failed to sync " + file_name_ + ": " + std::strerror(err));
    }

    if (observer_) {
        observer_->OnSync();
    }
}

// Cuts the file to num_pages pages and hands out num_pages next
void DiskManager::Truncate(page_id_t num_pages) {
    if (num_pages == INVALID_PAGE_ID) {
        throw IOException("Cannot truncate to INVALID_PAGE_ID pages");
    }
    if (::ftruncate(fd_, PageOffset(num_pages)) == -1) {
        const int err = errno;
        throw IOException("Failed to truncate " + file_name_ + " to " + std::to_string(num_pages) + " pages: " + std::strerror(err));
    }
    next_page_id_.store(num_pages);
}

// Asks the OS for the file size, so it counts a partial last page
uint64_t DiskManager::GetFileBytes() const {
    struct stat st{};
    if (::fstat(fd_, &st) < 0) {
        const int err = errno;
        throw IOException("Cannot stat database file: " + file_name_ + " (" + std::strerror(err) + ")");
    }
    return static_cast<uint64_t>(st.st_size);
}

}  // namespace shilmandb
