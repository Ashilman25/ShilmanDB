#include <gtest/gtest.h>
#include <string>
#include <unistd.h>
#include "storage/disk_manager.hpp"
#include "common/exception.hpp"
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>

namespace shilmandb {

class DiskManagerTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() /
                      ("shilmandb_dm_test_" + std::to_string(::getpid()) + ".db")).string();
        std::filesystem::remove(test_file_);
    }

    void TearDown() override {
        std::filesystem::remove(test_file_);
    }
};

TEST_F(DiskManagerTest, ReadWriteRoundTrip) {
    DiskManager dm(test_file_);
    auto page_id = dm.AllocatePage();

    std::array<char, PAGE_SIZE> write_buf{};
    std::memset(write_buf.data(), 0xAB, PAGE_SIZE);
    dm.WritePage(page_id, write_buf.data());

    std::array<char, PAGE_SIZE> read_buf{};
    dm.ReadPage(page_id, read_buf.data());

    EXPECT_EQ(std::memcmp(write_buf.data(), read_buf.data(), PAGE_SIZE), 0);
}

TEST_F(DiskManagerTest, AllocateSequentialPages) {
    DiskManager dm(test_file_);

    for (page_id_t i = 0; i < 10; ++i) {
        EXPECT_EQ(dm.AllocatePage(), i);
    }
}

TEST_F(DiskManagerTest, ReadUnwrittenPageReturnsZeros) {
    DiskManager dm(test_file_);
    dm.AllocatePage();  // page 0 — allocated but never written

    std::array<char, PAGE_SIZE> buf{};
    std::memset(buf.data(), 0xFF, PAGE_SIZE);  // fill non-zero to verify zeroing
    dm.ReadPage(0, buf.data());

    std::array<char, PAGE_SIZE> zeros{};
    EXPECT_EQ(std::memcmp(buf.data(), zeros.data(), PAGE_SIZE), 0);
}

TEST_F(DiskManagerTest, WriteMultiplePages) {
    DiskManager dm(test_file_);

    constexpr int kNumPages = 5;
    std::array<std::array<char, PAGE_SIZE>, kNumPages> buffers{};

    for (int i = 0; i < kNumPages; ++i) {
        auto pid = dm.AllocatePage();
        std::memset(buffers[i].data(), i + 1, PAGE_SIZE);
        dm.WritePage(pid, buffers[i].data());
    }

    std::array<char, PAGE_SIZE> read_buf{};
    for (int i = 0; i < kNumPages; ++i) {
        dm.ReadPage(static_cast<page_id_t>(i), read_buf.data());
        EXPECT_EQ(std::memcmp(read_buf.data(), buffers[i].data(), PAGE_SIZE), 0)
            << "Mismatch on page " << i;
    }
}

TEST_F(DiskManagerTest, WriteInvalidPageThrows) {
    DiskManager dm(test_file_);

    std::array<char, PAGE_SIZE> buf{};
    EXPECT_THROW(dm.WritePage(INVALID_PAGE_ID, buf.data()), std::runtime_error);
}

TEST_F(DiskManagerTest, PersistenceAcrossInstances) {
    std::array<char, PAGE_SIZE> write_buf{};
    std::memset(write_buf.data(), 0xCD, PAGE_SIZE);

    {
        DiskManager dm(test_file_);
        auto pid = dm.AllocatePage();
        dm.WritePage(pid, write_buf.data());
    }  // dm destroyed, file flushed and closed

    // Reopen — constructor should recover next_page_id_ from file size
    DiskManager dm2(test_file_);
    EXPECT_EQ(dm2.AllocatePage(), 1);  // next id should be 1, not 0

    std::array<char, PAGE_SIZE> read_buf{};
    dm2.ReadPage(0, read_buf.data());
    EXPECT_EQ(std::memcmp(read_buf.data(), write_buf.data(), PAGE_SIZE), 0);
}

TEST_F(DiskManagerTest, PartialReadZeroPads) {
    {
        DiskManager dm(test_file_);
        auto pid = dm.AllocatePage();
        std::array<char, PAGE_SIZE> buf{};
        std::memset(buf.data(), 0xEE, PAGE_SIZE);
        dm.WritePage(pid, buf.data());
    }

    // Truncate file to half a page
    std::filesystem::resize_file(test_file_, PAGE_SIZE / 2);

    DiskManager dm2(test_file_);
    std::array<char, PAGE_SIZE> read_buf{};
    dm2.ReadPage(0, read_buf.data());

    // First half should be the original pattern
    for (size_t i = 0; i < PAGE_SIZE / 2; ++i) {
        EXPECT_EQ(static_cast<unsigned char>(read_buf[i]), 0xEE) << "byte " << i;
    }
    // Second half should be zero-padded
    for (size_t i = PAGE_SIZE / 2; i < PAGE_SIZE; ++i) {
        EXPECT_EQ(read_buf[i], 0) << "byte " << i;
    }
}

TEST_F(DiskManagerTest, ReadBeyondAllocatedReturnsZeros) {
    DiskManager dm(test_file_);

    // Read page 999 — never allocated, file is empty
    std::array<char, PAGE_SIZE> buf{};
    std::memset(buf.data(), 0xFF, PAGE_SIZE);
    dm.ReadPage(999, buf.data());

    std::array<char, PAGE_SIZE> zeros{};
    EXPECT_EQ(std::memcmp(buf.data(), zeros.data(), PAGE_SIZE), 0);
}

TEST_F(DiskManagerTest, FreshFileReadIsFast) {
    // SetUp removed the file, so this goes through the create-new-file path
    DiskManager dm(test_file_);
    constexpr int kNumPages = 1000;

    std::array<char, PAGE_SIZE> buf{};
    for (int i = 0; i < kNumPages; ++i) {
        std::memset(buf.data(), i % 251, PAGE_SIZE);
        dm.WritePage(dm.AllocatePage(), buf.data());
    }

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kNumPages; ++i) {
        dm.ReadPage(static_cast<page_id_t>(i), buf.data());
        ASSERT_EQ(static_cast<unsigned char>(buf[0]), i % 251) << "page " << i;
        ASSERT_EQ(static_cast<unsigned char>(buf[PAGE_SIZE - 1]), i % 251) << "page " << i;
    }
    auto elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();

    // A healthy read is about 1 us per page; the old reopened-stream bug was ~650 us
    EXPECT_LT(elapsed_ms, 100.0) << "1000 page reads took " << elapsed_ms << " ms";
}

TEST_F(DiskManagerTest, OpenInvalidPathThrows) {
    auto bad_path = (std::filesystem::temp_directory_path() /
                     "shilmandb_no_such_dir" / "x.db").string();
    EXPECT_THROW(DiskManager dm(bad_path), IOException);
}

TEST_F(DiskManagerTest, WriteBeyondEndLeavesZeroGap) {
    std::array<char, PAGE_SIZE> pattern{};
    std::memset(pattern.data(), 0x5A, PAGE_SIZE);
    std::array<char, PAGE_SIZE> zeros{};

    {
        DiskManager dm(test_file_);
        dm.WritePage(5, pattern.data());

        std::array<char, PAGE_SIZE> buf{};
        std::memset(buf.data(), 0xFF, PAGE_SIZE);
        dm.ReadPage(2, buf.data());
        EXPECT_EQ(std::memcmp(buf.data(), zeros.data(), PAGE_SIZE), 0);

        dm.ReadPage(5, buf.data());
        EXPECT_EQ(std::memcmp(buf.data(), pattern.data(), PAGE_SIZE), 0);
    }

    // File is now 6 pages long, so a reopen hands out page 6 next
    DiskManager dm2(test_file_);
    EXPECT_EQ(dm2.AllocatePage(), 6u);
}

}  // namespace shilmandb
