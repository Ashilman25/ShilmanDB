#include "storage/columnar/superblock.hpp"
#include "storage/columnar/encoding.hpp"
#include <cstddef>
#include <cstring>

namespace shilmandb {

namespace {

// The bytes after the page header, stored exactly as laid out here.
// The magic and version offsets never change, so any build can read a newer version.
struct SlotLayout {
    char magic[16];
    uint32_t format_version;
    uint32_t page_size;
    uint64_t generation;
    uint32_t catalog_first_page;
    uint32_t catalog_page_count;
    uint64_t catalog_bytes;
    uint32_t committed_pages;
    uint32_t next_oid;
    uint32_t free_list_root;
    uint32_t reserved;
};

constexpr size_t kLayoutOffset = kPageHeaderSize;
static_assert(sizeof(SlotLayout) == 64, "the superblock fields are 64 bytes on disk");
static_assert(kLayoutOffset + offsetof(SlotLayout, magic) == 32 && kLayoutOffset + offsetof(SlotLayout, format_version) == 48, "the magic and version offsets are fixed for every version");
static_assert(offsetof(SlotLayout, page_size) == 20 && offsetof(SlotLayout, generation) == 24 && offsetof(SlotLayout, catalog_first_page) == 32 && offsetof(SlotLayout, catalog_page_count) == 36, "superblock byte layout");
static_assert(offsetof(SlotLayout, catalog_bytes) == 40 && offsetof(SlotLayout, committed_pages) == 48 && offsetof(SlotLayout, next_oid) == 52 && offsetof(SlotLayout, free_list_root) == 56 && offsetof(SlotLayout, reserved) == 60, "superblock byte layout");

}  // namespace

// Zeros the page first, so unused bytes are always zero.
void EncodeSuperblock(const Superblock& sb, page_id_t slot, char* page) {
    std::memset(page, 0, PAGE_SIZE);
    ColumnarPageHeader header;
    header.kind = static_cast<uint8_t>(PageKind::SUPERBLOCK);
    std::memcpy(page, &header, sizeof(header));

    SlotLayout layout{};
    std::memcpy(layout.magic, kMagic, sizeof(kMagic));
    layout.format_version = sb.format_version;
    layout.page_size = sb.page_size;
    layout.generation = sb.generation;
    layout.catalog_first_page = sb.catalog_first_page;
    layout.catalog_page_count = sb.catalog_page_count;
    layout.catalog_bytes = sb.catalog_bytes;
    layout.committed_pages = sb.committed_pages;
    layout.next_oid = sb.next_oid;
    layout.free_list_root = sb.free_list_root;
    std::memcpy(page + kLayoutOffset, &layout, sizeof(layout));

    SealPages(page, 1, slot);
}

// Checks the magic, then the version, then the checksum and kind.
// An unreadable version may checksum differently, so its checksum is never looked at.
SlotRead DecodeSuperblock(const char* page, page_id_t slot) {
    SlotRead read;
    SlotLayout layout;
    std::memcpy(&layout, page + kLayoutOffset, sizeof(layout));
    if (std::memcmp(layout.magic, kMagic, sizeof(kMagic)) != 0) {
        read.state = SlotState::NO_MAGIC;
        return read;
    }
    read.block.format_version = layout.format_version;
    if (layout.format_version < kOldestFormatVersion || layout.format_version > kFormatVersion) {
        read.state = SlotState::OTHER_VERSION;
        return read;
    }
    ColumnarPageHeader header;
    std::memcpy(&header, page, sizeof(header));
    if (!VerifyPage(page, slot) || header.kind != static_cast<uint8_t>(PageKind::SUPERBLOCK)) {
        read.state = SlotState::BAD_CHECKSUM;
        return read;
    }
    read.block.page_size = layout.page_size;
    read.block.generation = layout.generation;
    read.block.catalog_first_page = layout.catalog_first_page;
    read.block.catalog_page_count = layout.catalog_page_count;
    read.block.catalog_bytes = layout.catalog_bytes;
    read.block.committed_pages = layout.committed_pages;
    read.block.next_oid = layout.next_oid;
    read.block.free_list_root = layout.free_list_root;
    read.state = SlotState::VALID;
    return read;
}

}  // namespace shilmandb
