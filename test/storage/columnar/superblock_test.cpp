#include <gtest/gtest.h>
#include "storage/columnar/encoding.hpp"
#include "storage/columnar/superblock.hpp"
#include <cstring>
#include <string>
#include <vector>

namespace shilmandb {
namespace {

// A superblock with every field away from its default.
Superblock Sample() {
    Superblock sb;
    sb.generation = 0x0123456789ABCDEFULL;
    sb.catalog_first_page = 4242;
    sb.catalog_page_count = 7;
    sb.catalog_bytes = 0x00000001FFFFFFFFULL;
    sb.committed_pages = 99999;
    sb.next_oid = kFirstOid + 12;
    sb.free_list_root = 31;
    return sb;
}

TEST(SuperblockTest, SuperblockRoundTrip) {
    for (page_id_t slot : {page_id_t{0}, page_id_t{1}}) {
        std::vector<char> page(PAGE_SIZE, 'x');
        const Superblock sb = Sample();
        EncodeSuperblock(sb, slot, page.data());
        const SlotRead read = DecodeSuperblock(page.data(), slot);
        ASSERT_EQ(read.state, SlotState::VALID) << "slot " << slot;
        EXPECT_EQ(read.block.format_version, kFormatVersion);
        EXPECT_EQ(read.block.page_size, PAGE_SIZE);
        EXPECT_EQ(read.block.generation, sb.generation);
        EXPECT_EQ(read.block.catalog_first_page, sb.catalog_first_page);
        EXPECT_EQ(read.block.catalog_page_count, sb.catalog_page_count);
        EXPECT_EQ(read.block.catalog_bytes, sb.catalog_bytes);
        EXPECT_EQ(read.block.committed_pages, sb.committed_pages);
        EXPECT_EQ(read.block.next_oid, sb.next_oid);
        EXPECT_EQ(read.block.free_list_root, sb.free_list_root);
    }
}

TEST(SuperblockTest, SuperblockLayout) {
    std::vector<char> page(PAGE_SIZE, 'x');
    Superblock sb = Sample();
    sb.page_size = 4096;
    EncodeSuperblock(sb, 0, page.data());
    EXPECT_EQ(page[4], static_cast<char>(PageKind::SUPERBLOCK));
    EXPECT_EQ(std::memcmp(page.data() + 32, kMagic, sizeof(kMagic)), 0);
    uint32_t version = 0;
    std::memcpy(&version, page.data() + 48, sizeof(version));
    EXPECT_EQ(version, kFormatVersion);
    EXPECT_EQ(DecodeSuperblock(page.data(), 0).block.page_size, 4096u);
    // Everything past the fields is zero.
    for (size_t i = 96; i < PAGE_SIZE; ++i) ASSERT_EQ(page[i], 0) << "byte " << i;
}

TEST(SuperblockTest, SuperblockStates) {
    const std::vector<char> zeros(PAGE_SIZE, 0);
    EXPECT_EQ(DecodeSuperblock(zeros.data(), 0).state, SlotState::NO_MAGIC);

    std::vector<char> good(PAGE_SIZE);
    EncodeSuperblock(Sample(), 0, good.data());

    std::vector<char> flipped = good;
    flipped[56] ^= 0x10;  // a bit of the generation
    EXPECT_EQ(DecodeSuperblock(flipped.data(), 0).state, SlotState::BAD_CHECKSUM);

    EXPECT_EQ(DecodeSuperblock(good.data(), 1).state, SlotState::BAD_CHECKSUM);

    Superblock newer = Sample();
    newer.format_version = 3;
    std::vector<char> page(PAGE_SIZE);
    EncodeSuperblock(newer, 1, page.data());
    const SlotRead read = DecodeSuperblock(page.data(), 1);
    EXPECT_EQ(read.state, SlotState::OTHER_VERSION);
    EXPECT_EQ(read.block.format_version, 3u);

    // A sealed page of another kind with the magic in place is not a superblock.
    std::vector<char> other_kind = good;
    other_kind[4] = static_cast<char>(PageKind::CATALOG);
    SealPages(other_kind.data(), 1, 0);
    EXPECT_EQ(DecodeSuperblock(other_kind.data(), 0).state, SlotState::BAD_CHECKSUM);

    std::vector<char> no_magic = good;
    no_magic[33] ^= 1;
    EXPECT_EQ(DecodeSuperblock(no_magic.data(), 0).state, SlotState::NO_MAGIC);
}

TEST(SuperblockTest, OtherVersionWhateverTheChecksum) {
    // A newer format may change the checksum, so the version is read first.
    Superblock newer = Sample();
    newer.format_version = 3;
    std::vector<char> page(PAGE_SIZE);
    EncodeSuperblock(newer, 1, page.data());
    page[56] ^= 0x10;  // a bit of the generation, so the checksum fails
    const SlotRead read = DecodeSuperblock(page.data(), 1);
    EXPECT_EQ(read.state, SlotState::OTHER_VERSION);
    EXPECT_EQ(read.block.format_version, 3u);

    // Also with the wrong slot's seed.
    EncodeSuperblock(newer, 1, page.data());
    EXPECT_EQ(DecodeSuperblock(page.data(), 0).state, SlotState::OTHER_VERSION);

    // A readable version's slot with a bad checksum is still just damaged.
    for (uint32_t version : {1u, 2u}) {
        Superblock readable = Sample();
        readable.format_version = version;
        std::vector<char> torn(PAGE_SIZE);
        EncodeSuperblock(readable, 0, torn.data());
        torn[56] ^= 0x10;
        EXPECT_EQ(DecodeSuperblock(torn.data(), 0).state, SlotState::BAD_CHECKSUM) << "version " << version;
    }
}

TEST(SuperblockTest, ReadsVersionsOneAndTwo) {
    EXPECT_EQ(kFormatVersion, 2u);
    EXPECT_EQ(kOldestFormatVersion, 1u);
    EXPECT_EQ(Superblock{}.format_version, 2u);
    for (uint32_t version : {0u, 1u, 2u, 3u, 7u}) {
        SCOPED_TRACE("version " + std::to_string(version));
        Superblock sb = Sample();
        sb.format_version = version;
        std::vector<char> page(PAGE_SIZE);
        EncodeSuperblock(sb, 1, page.data());
        const SlotRead read = DecodeSuperblock(page.data(), 1);
        EXPECT_EQ(read.block.format_version, version);
        if (version == 1 || version == 2) {
            ASSERT_EQ(read.state, SlotState::VALID);
            EXPECT_EQ(read.block.generation, sb.generation);
            EXPECT_EQ(read.block.catalog_bytes, sb.catalog_bytes);
        } else {
            EXPECT_EQ(read.state, SlotState::OTHER_VERSION);
        }
    }
}

}  // namespace
}  // namespace shilmandb
