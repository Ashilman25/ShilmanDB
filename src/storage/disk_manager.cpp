#include "storage/disk_manager.hpp"
#include "common/exception.hpp"
#include <cstring>

namespace shilmandb {

DiskManager::DiskManager(const std::string& db_file) : file_name_(db_file) {
    db_io_.open(db_file, std::ios::in | std::ios::out | std::ios::binary);

    if (!db_io_.is_open()) {

        db_io_.clear();
        db_io_.open(db_file, std::ios::out | std::ios::binary);
        db_io_.close();
        db_io_.open(db_file, std::ios::in | std::ios::out | std::ios::binary);

        if (!db_io_.is_open()) {
            throw IOException("Cannot open database file: " + db_file);
        }
    }

    // get next page id from file size
    db_io_.seekg(0, std::ios::end);
    auto file_size = static_cast<size_t>(db_io_.tellg());
    next_page_id_.store(static_cast<page_id_t>(file_size / PAGE_SIZE));
}

DiskManager::~DiskManager() {
    if (db_io_.is_open()) {
        db_io_.flush();
        db_io_.close();
    }
}

void DiskManager::ReadPage(page_id_t page_id, char* page_data) {
    auto offset = static_cast<std::streamoff>(page_id) * PAGE_SIZE;

    db_io_.seekg(0, std::ios::end);
    auto file_end = db_io_.tellg();

    if (offset >= file_end) {
        std::memset(page_data, 0, PAGE_SIZE);
        return;
    }

    db_io_.seekg(offset);
    db_io_.read(page_data, PAGE_SIZE);

    if (auto bytes_read = db_io_.gcount(); bytes_read < static_cast<std::streamsize>(PAGE_SIZE)) {
        std::memset(page_data + bytes_read, 0, PAGE_SIZE - bytes_read);
    }
}

void DiskManager::WritePage(page_id_t page_id, const char* page_data) {
    if (page_id == INVALID_PAGE_ID) {
        throw IOException("Cannot write to INVALID_PAGE_ID");
    }

    auto offset = static_cast<std::streamoff>(page_id) * PAGE_SIZE;
    db_io_.seekp(offset);
    db_io_.write(page_data, PAGE_SIZE);

    if (db_io_.bad()) {
        throw IOException("Failed to write page " + std::to_string(page_id));
    }

    db_io_.flush();
}

page_id_t DiskManager::AllocatePage() {
    return next_page_id_++;
}

void DiskManager::DeallocatePage([[maybe_unused]] page_id_t page_id) {
    // for other classes to use and such
}

}  // namespace shilmandb
