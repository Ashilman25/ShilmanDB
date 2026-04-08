#pragma once
#include "common/config.hpp"
#include <atomic>
#include <fstream>
#include <string>

namespace shilmandb {

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

    const std::string& GetFileName() const { return file_name_; }

private:
    std::fstream db_io_;
    std::string file_name_;
    std::atomic<page_id_t> next_page_id_{0};
};

}  // namespace shilmandb
