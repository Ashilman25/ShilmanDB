#include "storage/columnar/columnar_file.hpp"
#include "common/exception.hpp"
#include "storage/columnar/catalog_codec.hpp"
#include "storage/columnar/encoding.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace shilmandb {

namespace {

constexpr size_t kCatalogSlice = PAGE_SIZE - kPageHeaderSize;  // blob bytes per catalog page
constexpr const char* kNotOurs = "not a ShilmanDB file, or an older format; reload it";

}  // namespace

ColumnarFile::ColumnarFile(DiskManager* disk) : disk_(disk) {
    if (disk_->GetFileBytes() == 0) {
        Create();
    } else {
        Open();
    }
    disk_->ResetReadStats();  // the header and catalog reads don't count
}

// Slot 0 at generation 1 and slot 1 at 0, both with an empty catalog.
void ColumnarFile::Create() {
    [[maybe_unused]] const page_id_t first = disk_->AllocatePages(kHeaderPages);
    assert(first == 0 && "an empty file should hand out page 0 first");
    std::vector<char> page(PAGE_SIZE);
    for (page_id_t slot = 0; slot < kHeaderPages; ++slot) {
        Superblock sb;
        sb.generation = slot == 0 ? 1 : 0;
        EncodeSuperblock(sb, slot, page.data());
        disk_->WritePage(slot, page.data());
        if (slot == 0) current_ = sb;
    }
    disk_->Sync();
    current_slot_ = 0;
}

// Checks every refusal rule before anything is written, then cuts off pages past the commit.
void ColumnarFile::Open() {
    const uint64_t bytes = disk_->GetFileBytes();
    if (bytes < PAGE_SIZE) Refuse(kNotOurs);

    SlotRead slots[kHeaderPages];
    std::vector<char> page(PAGE_SIZE);
    for (page_id_t slot = 0; slot < kHeaderPages; ++slot) {
        disk_->ReadPage(slot, page.data());
        slots[slot] = DecodeSuperblock(page.data(), slot);
    }

    // Another version in either slot refuses, so I never cut pages a newer build committed
    std::optional<uint32_t> other;
    for (const SlotRead& s : slots) {
        if (s.state == SlotState::OTHER_VERSION) {
            other = std::max(other.value_or(0), s.block.format_version);
        }
    }
    if (other) {
        const std::string theirs = std::to_string(*other);
        const std::string ours = std::to_string(kFormatVersion);
        if (*other < kFormatVersion) {
            Refuse("written in format " + theirs + ", this build reads format " + ours + "; reload it");
        }
        Refuse("written by a newer ShilmanDB (format " + theirs + "; this build reads format " + ours + ")");
    }

    // A usable slot checks out and has this build's page size; the higher generation wins
    std::optional<page_id_t> winner;
    for (page_id_t slot = 0; slot < kHeaderPages; ++slot) {
        const SlotRead& s = slots[slot];
        if (s.state != SlotState::VALID || s.block.page_size != PAGE_SIZE) continue;
        if (!winner || s.block.generation > slots[*winner].block.generation) winner = slot;
    }
    if (!winner) {
        const bool magic = slots[0].state != SlotState::NO_MAGIC || slots[1].state != SlotState::NO_MAGIC;
        Refuse(magic ? "the file header is damaged" : kNotOurs);
    }

    const Superblock& sb = slots[*winner].block;
    if (sb.committed_pages < kHeaderPages) Refuse("the file header is damaged");
    const uint64_t file_pages = bytes / PAGE_SIZE;
    if (sb.committed_pages > file_pages) {
        Refuse("the file is shorter than its last commit (" + std::to_string(file_pages) + " of " + std::to_string(sb.committed_pages) + " pages)");
    }
    catalog_ = ReadCatalog(sb);

    if (bytes != static_cast<uint64_t>(sb.committed_pages) * PAGE_SIZE) disk_->Truncate(sb.committed_pages);
    current_ = sb;
    current_slot_ = *winner;
    next_oid_ = sb.next_oid;
}

// Reads and checks the catalog pages the slot names, then decodes the joined blob.
CatalogMeta ColumnarFile::ReadCatalog(const Superblock& sb) const {
    const std::string pages_text = std::to_string(sb.catalog_page_count) + " pages from page " + std::to_string(sb.catalog_first_page);
    if (sb.catalog_page_count == 0) {
        if (sb.catalog_bytes != 0) Refuse("damaged catalog: no pages but " + std::to_string(sb.catalog_bytes) + " bytes");
        return {};
    }
    if (sb.catalog_first_page < kHeaderPages || sb.catalog_first_page > sb.committed_pages ||
        sb.catalog_page_count > sb.committed_pages - sb.catalog_first_page) {
        Refuse("damaged catalog: " + pages_text + " lie outside the committed " + std::to_string(sb.committed_pages) + " pages");
    }
    if (sb.catalog_bytes > static_cast<uint64_t>(sb.catalog_page_count) * kCatalogSlice) {
        Refuse("damaged catalog: " + std::to_string(sb.catalog_bytes) + " bytes can't fit in " + pages_text);
    }

    std::vector<char> blob;
    blob.reserve(sb.catalog_bytes);
    std::vector<char> page(PAGE_SIZE);
    for (uint32_t i = 0; i < sb.catalog_page_count; ++i) {
        const page_id_t id = sb.catalog_first_page + i;
        const std::string where = "page " + std::to_string(id);
        disk_->ReadPage(id, page.data());
        if (!VerifyPage(page.data(), id)) Refuse("damaged catalog: " + where + " fails its checksum");
        ColumnarPageHeader header;
        std::memcpy(&header, page.data(), sizeof(header));
        if (header.kind != static_cast<uint8_t>(PageKind::CATALOG)) {
            Refuse("damaged catalog: " + where + " is not a catalog page");
        }
        if (header.first_row != blob.size()) {
            Refuse("damaged catalog: " + where + " starts at byte " + std::to_string(header.first_row) + ", expected " + std::to_string(blob.size()));
        }
        if (header.value_count > kCatalogSlice) {
            Refuse("damaged catalog: " + where + " says it holds " + std::to_string(header.value_count) + " bytes");
        }
        blob.insert(blob.end(), page.data() + kPageHeaderSize, page.data() + kPageHeaderSize + header.value_count);
    }
    if (blob.size() != sb.catalog_bytes) {
        Refuse("damaged catalog: the pages hold " + std::to_string(blob.size()) + " bytes, the header says " + std::to_string(sb.catalog_bytes));
    }

    try {
        return DecodeCatalog(blob.data(), blob.size());
    } catch (const DatabaseException& e) {
        Refuse(e.what());
    }
}

// Catalog pages, sync, the other slot, sync, then the in-memory switch.
void ColumnarFile::Commit(CatalogMeta next) {
    const std::vector<char> blob = EncodeCatalog(next);
    if (blob.size() > std::numeric_limits<uint32_t>::max()) {
        throw DatabaseException("the catalog is too big to commit: " + std::to_string(blob.size()) + " bytes");
    }
    const auto count = static_cast<uint32_t>((blob.size() + kCatalogSlice - 1) / kCatalogSlice);
    const page_id_t first = disk_->AllocatePages(count);

    std::vector<char> pages(static_cast<size_t>(count) * PAGE_SIZE, 0);
    for (uint32_t i = 0; i < count; ++i) {
        const size_t offset = static_cast<size_t>(i) * kCatalogSlice;
        const size_t length = std::min(kCatalogSlice, blob.size() - offset);
        ColumnarPageHeader header;
        header.kind = static_cast<uint8_t>(PageKind::CATALOG);
        header.first_row = static_cast<uint32_t>(offset);
        header.value_count = static_cast<uint16_t>(length);
        char* page = pages.data() + static_cast<size_t>(i) * PAGE_SIZE;
        std::memcpy(page, &header, sizeof(header));
        std::memcpy(page + kPageHeaderSize, blob.data() + offset, length);
    }
    SealPages(pages.data(), count, first);
    for (uint32_t i = 0; i < count; ++i) {
        disk_->WritePage(first + i, pages.data() + static_cast<size_t>(i) * PAGE_SIZE);
    }
    disk_->Sync();

    Superblock sb = current_;
    sb.generation += 1;
    sb.catalog_first_page = first;
    sb.catalog_page_count = count;
    sb.catalog_bytes = blob.size();
    sb.committed_pages = disk_->GetNumPages();  // covers the data pages the caller wrote too
    sb.next_oid = next_oid_;
    const page_id_t slot = current_slot_ == 0 ? 1 : 0;
    std::vector<char> slot_page(PAGE_SIZE);
    EncodeSuperblock(sb, slot, slot_page.data());
    disk_->WritePage(slot, slot_page.data());
    disk_->Sync();

    catalog_ = std::move(next);
    current_ = sb;
    current_slot_ = slot;
}

// Throws with the file's path in front of the reason.
void ColumnarFile::Refuse(const std::string& why) const {
    throw DatabaseException(disk_->GetFileName() + ": " + why);
}

}  // namespace shilmandb
