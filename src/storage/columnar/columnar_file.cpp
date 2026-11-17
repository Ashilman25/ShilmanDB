#include "storage/columnar/columnar_file.hpp"
#include "common/exception.hpp"
#include "storage/columnar/catalog_codec.hpp"
#include "storage/columnar/encoding.hpp"
#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace shilmandb {

namespace {

constexpr size_t kCatalogSlice = PAGE_SIZE - kPageHeaderSize;  // blob bytes per catalog page
constexpr const char* kNotOurs = "not a ShilmanDB file, or an older format; reload it";

// What the open refuses with, so a half-created file can be created again instead.
class Refusal : public DatabaseException {
public:
    using DatabaseException::DatabaseException;
};

// Why the catalog doesn't fit below page end or next_oid; empty if it fits.
std::string Misfit(const CatalogMeta& meta, uint64_t end, uint32_t next_oid) {
    const std::string range = end > kHeaderPages ? "pages " + std::to_string(kHeaderPages) + " to " + std::to_string(end - 1) : "a file with no data pages";
    const auto oid_text = [&](uint32_t oid) {
        return " has oid " + std::to_string(oid) + ", not below the next oid " + std::to_string(next_oid);
    };
    for (const ColumnarTableMeta& table : meta.tables) {
        if (table.oid >= next_oid) return "table " + table.name + oid_text(table.oid);
        for (size_t g = 0; g < table.row_groups.size(); ++g) {
            const std::vector<ChunkMeta>& chunks = table.row_groups[g].chunks;
            for (size_t c = 0; c < chunks.size(); ++c) {
                const ChunkMeta& chunk = chunks[c];
                if (chunk.page_count == 0) continue;
                if (chunk.first_page >= kHeaderPages && static_cast<uint64_t>(chunk.first_page) + chunk.page_count <= end) continue;
                return "table " + table.name + " column " + table.columns.at(c).name + " row group " + std::to_string(g) + ": " + std::to_string(chunk.page_count) + " pages from page " + std::to_string(chunk.first_page) + " lie outside " + range;
            }
        }
    }
    for (const IndexMeta& index : meta.indexes) {
        if (index.oid >= next_oid) return "index " + index.name + oid_text(index.oid);
        if (index.root_page == INVALID_PAGE_ID || (index.root_page >= kHeaderPages && index.root_page < end)) continue;
        return "index " + index.name + " root page " + std::to_string(index.root_page) + " lies outside " + range;
    }
    return "";
}

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
    if (disk_->GetNumPages() != 0) {
        throw DatabaseException(disk_->GetFileName() + ": can't create the header, " + std::to_string(disk_->GetNumPages()) + " pages were already handed out");
    }
    (void)disk_->AllocatePages(kHeaderPages);
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

// Refuses before writing anything, or creates again a file a crash left half created.
// Then cuts off pages past the commit.
void ColumnarFile::Open() {
    const uint64_t bytes = disk_->GetFileBytes();

    // Past the end reads as zeros, so a short file still gives both slots
    SlotRead slots[kHeaderPages];
    std::vector<char> header(static_cast<size_t>(kHeaderPages) * PAGE_SIZE);
    for (page_id_t slot = 0; slot < kHeaderPages; ++slot) {
        char* page = header.data() + static_cast<size_t>(slot) * PAGE_SIZE;
        disk_->ReadPage(slot, page);
        slots[slot] = DecodeSuperblock(page, slot);
    }
    const bool all_zero = std::all_of(header.begin(), header.end(), [](char c) { return c == 0; });
    if (bytes < PAGE_SIZE && !all_zero) Refuse(kNotOurs);

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

    // A good slot for another page size means another build; refuse rather than fall back
    for (const SlotRead& s : slots) {
        if (s.state == SlotState::VALID && s.block.page_size != PAGE_SIZE) {
            Refuse("written with " + std::to_string(s.block.page_size) + "-byte pages; this build uses " + std::to_string(PAGE_SIZE) + "-byte pages");
        }
    }

    // Every commit adds a catalog page, so at most 2 pages means never committed.
    // A slot counts as a commit by what it points at, not only its generation.
    const bool magic = std::any_of(std::begin(slots), std::end(slots), [](const SlotRead& s) { return s.state != SlotState::NO_MAGIC; });
    const bool committed = std::any_of(std::begin(slots), std::end(slots), [](const SlotRead& s) {
        return s.state == SlotState::VALID && (s.block.generation >= 2 || s.block.committed_pages > kHeaderPages || s.block.catalog_page_count > 0);
    });
    const bool never_committed = bytes <= static_cast<uint64_t>(kHeaderPages) * PAGE_SIZE && !committed && (all_zero || magic);
    try {
        OpenWinner(bytes, slots);
    } catch (const Refusal&) {
        if (!never_committed) throw;
        disk_->Truncate(0);  // a crash during create left this; nothing in it was ever committed
        Create();
    }
}

// Opens the newer valid slot, or refuses; nothing is written before the last check.
void ColumnarFile::OpenWinner(uint64_t bytes, const SlotRead* slots) {
    std::optional<page_id_t> winner;
    for (page_id_t slot = 0; slot < kHeaderPages; ++slot) {
        const SlotRead& s = slots[slot];
        if (s.state != SlotState::VALID) continue;
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
    CatalogMeta catalog = ReadCatalog(sb);

    // Everything the catalog names must survive the cut below
    const std::string misfit = Misfit(catalog, sb.committed_pages, sb.next_oid);
    if (!misfit.empty()) Refuse("damaged catalog: " + misfit);

    if (bytes != static_cast<uint64_t>(sb.committed_pages) * PAGE_SIZE) disk_->Truncate(sb.committed_pages);
    catalog_ = std::move(catalog);
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
    // Checked as the open would check it, so a bad catalog never reaches the disk
    (void)DecodeCatalog(blob.data(), blob.size());
    const std::string misfit = Misfit(next, disk_->GetNumPages(), next_oid_);
    if (!misfit.empty()) throw DatabaseException("damaged catalog: " + misfit);

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
    throw Refusal(disk_->GetFileName() + ": " + why);
}

}  // namespace shilmandb
