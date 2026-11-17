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
    // Creates the header in an empty file, else checks it and reads the catalog.
    // A file a crash left half created, with nothing ever committed, is created again.
    // Pages past the last commit are cut off. Throws DatabaseException naming the file.
    explicit ColumnarFile(DiskManager* disk);

    [[nodiscard]] const CatalogMeta& Catalog() const { return catalog_; }
    [[nodiscard]] const Superblock& Current() const { return current_; }
    [[nodiscard]] DiskManager* Disk() const { return disk_; }

    // Hands out the next OID; the next commit saves the counter
    [[nodiscard]] uint32_t AllocateOid() { return next_oid_++; }

    // New data pages must already be written. Writes next to fresh pages, syncs,
    // writes the other slot, syncs, and only then switches to it.
    // Throws before writing anything if the file wouldn't reopen with this catalog.
    // After a failed write or sync, reload: don't retry over the same data pages.
    void Commit(CatalogMeta next);

private:
    void Create();
    void Open();
    void OpenWinner(uint64_t bytes, const SlotRead* slots);
    [[nodiscard]] CatalogMeta ReadCatalog(const Superblock& sb) const;
    [[noreturn]] void Refuse(const std::string& why) const;

    DiskManager* disk_;
    CatalogMeta catalog_;
    Superblock current_;
    page_id_t current_slot_{0};
    uint32_t next_oid_{kFirstOid};
};

}  // namespace shilmandb
