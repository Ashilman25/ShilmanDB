#pragma once
#include "common/config.hpp"
#include "storage/columnar/columnar_meta.hpp"
#include "storage/columnar/superblock.hpp"
#include "storage/disk_manager.hpp"
#include <cstdint>
#include <string>

namespace shilmandb {

// Owns the two superblock slots and the catalog, and commits a new catalog crash-safely.
class ColumnarFile {
public:
    // An empty file gets a fresh header; otherwise the header is checked and the catalog read.
    // Pages past the last commit are cut off. Throws DatabaseException naming the file.
    explicit ColumnarFile(DiskManager* disk);

    [[nodiscard]] const CatalogMeta& Catalog() const { return catalog_; }
    [[nodiscard]] const Superblock& Current() const { return current_; }
    [[nodiscard]] DiskManager* Disk() const { return disk_; }

    // Hands out the next OID; the next commit saves the counter
    [[nodiscard]] uint32_t AllocateOid() { return next_oid_++; }

    // New data pages must already be written. Writes next to fresh pages, syncs,
    // writes the other slot, syncs, and only then switches to it.
    void Commit(CatalogMeta next);

private:
    void Create();
    void Open();
    [[nodiscard]] CatalogMeta ReadCatalog(const Superblock& sb) const;
    [[noreturn]] void Refuse(const std::string& why) const;

    DiskManager* disk_;
    CatalogMeta catalog_;
    Superblock current_;
    page_id_t current_slot_{0};
    uint32_t next_oid_{kFirstOid};
};

}  // namespace shilmandb
