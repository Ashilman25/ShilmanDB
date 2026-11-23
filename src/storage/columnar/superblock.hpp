#pragma once
#include "common/config.hpp"
#include <cstdint>

namespace shilmandb {

// Every commit writes kFormatVersion; this build reads every version from the oldest up.
inline constexpr uint32_t kFormatVersion = 2;
inline constexpr uint32_t kOldestFormatVersion = 1;
inline constexpr page_id_t kHeaderPages = 2;  // pages 0 and 1 are the two superblock slots
inline constexpr uint32_t kFirstOid = 16384;
inline constexpr char kMagic[16] = {'S', 'h', 'i', 'l', 'm', 'a', 'n', 'D', 'B', 0, 0, 0, 0, 0, 0, 0};

struct Superblock {
    uint32_t format_version{kFormatVersion};
    uint32_t page_size{PAGE_SIZE};
    uint64_t generation{0};  // +1 per commit; the valid slot with the higher one wins
    page_id_t catalog_first_page{INVALID_PAGE_ID};
    uint32_t catalog_page_count{0};
    uint64_t catalog_bytes{0};
    page_id_t committed_pages{kHeaderPages};  // the file's page count at commit
    uint32_t next_oid{kFirstOid};
    page_id_t free_list_root{INVALID_PAGE_ID};  // reserved
};

enum class SlotState { VALID, NO_MAGIC, BAD_CHECKSUM, OTHER_VERSION };

// OTHER_VERSION is a version this build can't read; only block.format_version is filled in.
struct SlotRead {
    SlotState state{SlotState::NO_MAGIC};
    Superblock block;
};

// Fills a whole page: header, magic, fields, zeros, then the checksum.
void EncodeSuperblock(const Superblock& sb, page_id_t slot, char* page);

// Never throws: garbage comes back as a state.
[[nodiscard]] SlotRead DecodeSuperblock(const char* page, page_id_t slot);

}  // namespace shilmandb
