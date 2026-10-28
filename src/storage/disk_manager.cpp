#include "storage/disk_manager.hpp"
#include "common/exception.hpp"
#include <cerrno>
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
}

page_id_t DiskManager::AllocatePage() {
    return next_page_id_++;
}

void DiskManager::DeallocatePage([[maybe_unused]] page_id_t page_id) {
    // for other classes to use and such
}

}  // namespace shilmandb
