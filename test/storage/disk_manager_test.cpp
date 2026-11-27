#include <gtest/gtest.h>
#include <string>
#include <unistd.h>
#include "storage/disk_manager.hpp"
#include "common/exception.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

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

TEST_F(DiskManagerTest, ReadStatsCountReadsAndTime) {
    DiskManager dm(test_file_);
    std::array<char, PAGE_SIZE> buf{};
    dm.WritePage(dm.AllocatePage(), buf.data());

    dm.ResetReadStats();
    for (int i = 0; i < 3; ++i) {
        dm.ReadPage(0, buf.data());
    }

    const auto stats = dm.GetReadStats();
    EXPECT_EQ(stats.reads, 3u);
    EXPECT_GT(stats.time, std::chrono::nanoseconds{0});
}

TEST_F(DiskManagerTest, ResetReadStatsClears) {
    DiskManager dm(test_file_);
    std::array<char, PAGE_SIZE> buf{};
    dm.ReadPage(0, buf.data());
    ASSERT_EQ(dm.GetReadStats().reads, 1u);

    dm.ResetReadStats();

    const auto stats = dm.GetReadStats();
    EXPECT_EQ(stats.reads, 0u);
    EXPECT_EQ(stats.time, std::chrono::nanoseconds{0});
}

TEST_F(DiskManagerTest, ReadInvalidPageThrows) {
    DiskManager dm(test_file_);

    std::array<char, PAGE_SIZE> buf{};
    EXPECT_THROW(dm.ReadPage(INVALID_PAGE_ID, buf.data()), IOException);
    EXPECT_EQ(dm.GetReadStats().reads, 0u);
}

TEST_F(DiskManagerTest, ReadPenaltyOffByDefault) {
    DiskManager dm(test_file_);
    EXPECT_EQ(dm.GetReadPenalty(), std::chrono::microseconds{0});
}

TEST_F(DiskManagerTest, ReadPenaltyWaits) {
    using namespace std::chrono_literals;
    DiskManager dm(test_file_);

    std::array<char, PAGE_SIZE> pattern{};
    std::memset(pattern.data(), 0x3C, PAGE_SIZE);
    dm.WritePage(dm.AllocatePage(), pattern.data());

    dm.SetReadPenalty(300us);
    EXPECT_EQ(dm.GetReadPenalty(), 300us);
    dm.ResetReadStats();

    std::array<char, PAGE_SIZE> buf{};
    const auto start = std::chrono::steady_clock::now();
    dm.ReadPage(0, buf.data());
    const auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_GE(elapsed, 300us);
    EXPECT_GE(dm.GetReadStats().time, 300us);
    EXPECT_EQ(std::memcmp(buf.data(), pattern.data(), PAGE_SIZE), 0);
}

TEST_F(DiskManagerTest, DirectIORoundTrip) {
#ifndef __APPLE__
    GTEST_SKIP() << "direct I/O is macOS only";
#endif
    DiskManager dm(test_file_);
    dm.SetDirectIO(true);

    std::array<char, PAGE_SIZE> pattern{};
    for (size_t i = 0; i < PAGE_SIZE; ++i) {
        pattern[i] = static_cast<char>(i % 251);
    }
    const auto pid = dm.AllocatePage();
    dm.WritePage(pid, pattern.data());

    std::array<char, PAGE_SIZE> buf{};
    dm.ReadPage(pid, buf.data());
    EXPECT_EQ(std::memcmp(buf.data(), pattern.data(), PAGE_SIZE), 0);

    EXPECT_NO_THROW(dm.SetDirectIO(false));
}

TEST_F(DiskManagerTest, DirectIOFlagIsReported) {
    DiskManager dm(test_file_);
    EXPECT_FALSE(dm.GetDirectIO());
#ifdef __APPLE__
    dm.SetDirectIO(true);
    EXPECT_TRUE(dm.GetDirectIO());
    dm.SetDirectIO(false);
    EXPECT_FALSE(dm.GetDirectIO());
#else
    EXPECT_THROW(dm.SetDirectIO(true), IOException);
    EXPECT_FALSE(dm.GetDirectIO());
#endif
}

TEST_F(DiskManagerTest, NumPagesFollowsAllocation) {
    DiskManager dm(test_file_);
    EXPECT_EQ(dm.GetNumPages(), 0u);

    for (int i = 0; i < 3; ++i) {
        dm.AllocatePage();
    }
    EXPECT_EQ(dm.GetNumPages(), 3u);
}

TEST_F(DiskManagerTest, AllocatePagesGivesContiguousRun) {
    DiskManager dm(test_file_);

    EXPECT_EQ(dm.AllocatePage(), 0u);
    EXPECT_EQ(dm.AllocatePages(5), 1u);
    EXPECT_EQ(dm.AllocatePage(), 6u);
    EXPECT_EQ(dm.GetNumPages(), 7u);

    // a run of zero pages moves nothing
    EXPECT_EQ(dm.AllocatePages(0), 7u);
    EXPECT_EQ(dm.GetNumPages(), 7u);
}

TEST_F(DiskManagerTest, TruncateCutsFileAndResetsNextPage) {
    DiskManager dm(test_file_);

    std::array<char, PAGE_SIZE> buf{};
    for (int i = 0; i < 5; ++i) {
        std::memset(buf.data(), i + 1, PAGE_SIZE);
        dm.WritePage(dm.AllocatePage(), buf.data());
    }
    ASSERT_EQ(dm.GetFileBytes(), 5 * PAGE_SIZE);
    EXPECT_THROW(dm.Truncate(INVALID_PAGE_ID), IOException);

    dm.Truncate(2);

    EXPECT_EQ(dm.GetNumPages(), 2u);
    EXPECT_EQ(dm.GetFileBytes(), 2 * PAGE_SIZE);
    EXPECT_EQ(dm.AllocatePage(), 2u);

    // the kept pages keep their bytes, a cut page reads as zeros
    dm.ReadPage(1, buf.data());
    EXPECT_EQ(static_cast<unsigned char>(buf[0]), 2);
    EXPECT_EQ(static_cast<unsigned char>(buf[PAGE_SIZE - 1]), 2);
    dm.ReadPage(3, buf.data());
    std::array<char, PAGE_SIZE> zeros{};
    EXPECT_EQ(std::memcmp(buf.data(), zeros.data(), PAGE_SIZE), 0);
}

TEST_F(DiskManagerTest, SyncSucceeds) {
    DiskManager dm(test_file_);
    std::array<char, PAGE_SIZE> buf{};
    std::memset(buf.data(), 0x42, PAGE_SIZE);
    dm.WritePage(dm.AllocatePage(), buf.data());

    EXPECT_NO_THROW(dm.Sync());
}

// Writes down every write and sync it hears about, in order
class RecordingObserver : public WriteObserver {
public:
    std::vector<std::string> events;
    std::vector<char> first_bytes;

    void OnWrite(page_id_t page_id, const char* data) override {
        events.push_back("W" + std::to_string(page_id));
        first_bytes.push_back(data[0]);
    }
    void OnSync() override { events.push_back("S"); }
};

TEST_F(DiskManagerTest, ObserverSeesWritesAndSyncsInOrder) {
    DiskManager dm(test_file_);
    RecordingObserver observer;
    dm.SetWriteObserver(&observer);

    std::array<char, PAGE_SIZE> buf{};
    buf[0] = 'a';
    dm.WritePage(0, buf.data());
    buf[0] = 'b';
    dm.WritePage(1, buf.data());
    dm.Sync();
    buf[0] = 'c';
    dm.WritePage(0, buf.data());

    // a failed write is not reported
    EXPECT_THROW(dm.WritePage(INVALID_PAGE_ID, buf.data()), IOException);

    const std::vector<std::string> expected{"W0", "W1", "S", "W0"};
    EXPECT_EQ(observer.events, expected);
    EXPECT_EQ(observer.first_bytes, (std::vector<char>{'a', 'b', 'c'}));

    // nullptr turns it off
    dm.SetWriteObserver(nullptr);
    dm.WritePage(2, buf.data());
    dm.Sync();
    EXPECT_EQ(observer.events, expected);
}

namespace {

// The byte at offset i of page pid, different on every page
char PatternByte(page_id_t pid, size_t i) {
    return static_cast<char>((pid * 131u + i * 7u) & 0xFFu);
}

// Writes count pages, each filled with its own pattern
void WritePatternPages(DiskManager& dm, page_id_t count) {
    std::vector<char> page(PAGE_SIZE);
    for (page_id_t pid = 0; pid < count; ++pid) {
        for (size_t i = 0; i < PAGE_SIZE; ++i) page[i] = PatternByte(pid, i);
        dm.WritePage(dm.AllocatePage(), page.data());
    }
}

// Reads n pages from first with ReadPages, then checks each against ReadPage
void ExpectReadPagesMatches(DiskManager& dm, page_id_t first, size_t n) {
    SCOPED_TRACE("first " + std::to_string(first) + ", n " + std::to_string(n));
    std::vector<std::vector<char>> frames(n, std::vector<char>(PAGE_SIZE, 0x7F));
    std::vector<char*> ptrs;
    for (auto& frame : frames) ptrs.push_back(frame.data());

    const auto before = dm.GetReadStats();
    const uint64_t batches_before = dm.GetBatchReadCount();
    dm.ReadPages(first, n, ptrs.data());
    EXPECT_EQ(dm.GetReadStats().reads - before.reads, n);
    EXPECT_EQ(dm.GetBatchReadCount() - batches_before, 1u);

    std::vector<char> one(PAGE_SIZE);
    for (size_t i = 0; i < n; ++i) {
        dm.ReadPage(first + static_cast<page_id_t>(i), one.data());
        EXPECT_EQ(std::memcmp(frames[i].data(), one.data(), PAGE_SIZE), 0) << "page " << first + i;
    }
}

}  // namespace

TEST_F(DiskManagerTest, ReadPagesMatchesReadPage) {
    constexpr page_id_t kPages = 40;
    DiskManager dm(test_file_);
    WritePatternPages(dm, kPages);

    ExpectReadPagesMatches(dm, 3, 1);
    ExpectReadPagesMatches(dm, 7, 2);
    ExpectReadPagesMatches(dm, 20, 16);
    EXPECT_GT(dm.GetReadStats().time, std::chrono::nanoseconds{0});

    // A run past the end of the file reads zeros there
    ExpectReadPagesMatches(dm, kPages - 2, 6);
    std::vector<char> frame(PAGE_SIZE, 0x7F);
    char* one_frame[] = {frame.data()};
    dm.ReadPages(kPages + 3, 1, one_frame);
    EXPECT_TRUE(std::all_of(frame.begin(), frame.end(), [](char c) { return c == 0; }));

    // A last page cut in half reads its first half, then zeros
    std::filesystem::resize_file(test_file_, (kPages - 1) * PAGE_SIZE + PAGE_SIZE / 2);
    ExpectReadPagesMatches(dm, kPages - 3, 4);
    std::vector<std::vector<char>> frames(2, std::vector<char>(PAGE_SIZE, 0x7F));
    char* two_frames[] = {frames[0].data(), frames[1].data()};
    dm.ReadPages(kPages - 1, 2, two_frames);
    for (size_t i = 0; i < PAGE_SIZE; ++i) {
        const char expected = i < PAGE_SIZE / 2 ? PatternByte(kPages - 1, i) : 0;
        ASSERT_EQ(frames[0][i], expected) << "byte " << i;
        ASSERT_EQ(frames[1][i], 0) << "byte " << i;
    }

    // INVALID_PAGE_ID throws, even as the end of a run; nothing is counted
    const auto reads = dm.GetReadStats().reads;
    const uint64_t batches = dm.GetBatchReadCount();
    EXPECT_THROW(dm.ReadPages(INVALID_PAGE_ID, 1, one_frame), IOException);
    EXPECT_THROW(dm.ReadPages(INVALID_PAGE_ID - 1, 2, two_frames), IOException);
    EXPECT_EQ(dm.GetReadStats().reads, reads);
    EXPECT_EQ(dm.GetBatchReadCount(), batches);

    // No pages is nothing at all
    dm.ReadPages(0, 0, nullptr);
    EXPECT_EQ(dm.GetReadStats().reads, reads);
    EXPECT_EQ(dm.GetBatchReadCount(), batches);

    dm.ResetReadStats();
    EXPECT_EQ(dm.GetBatchReadCount(), 0u);
}

TEST_F(DiskManagerTest, ReadPagesSplitsLongRuns) {
    constexpr page_id_t kPages = 1030;  // more than IOV_MAX, 1024 on macOS and Linux
    DiskManager dm(test_file_);
    WritePatternPages(dm, kPages);
    ExpectReadPagesMatches(dm, 0, kPages);
}

TEST_F(DiskManagerTest, ReadPagesPenaltyCoversEachPage) {
    using namespace std::chrono_literals;
    DiskManager dm(test_file_);
    WritePatternPages(dm, 4);
    dm.SetReadPenalty(200us);
    dm.ResetReadStats();

    std::vector<std::vector<char>> frames(4, std::vector<char>(PAGE_SIZE));
    char* ptrs[] = {frames[0].data(), frames[1].data(), frames[2].data(), frames[3].data()};
    const auto start = std::chrono::steady_clock::now();
    dm.ReadPages(0, 4, ptrs);
    const auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_GE(elapsed, 800us);
    EXPECT_GE(dm.GetReadStats().time, 800us);
    EXPECT_EQ(dm.GetReadStats().reads, 4u);
}

TEST_F(DiskManagerTest, GetFileBytesCountsPartialPage) {
    {
        std::ofstream out(test_file_, std::ios::binary);
        const std::string bytes(100, 'x');
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    DiskManager dm(test_file_);
    EXPECT_EQ(dm.GetFileBytes(), 100u);
    EXPECT_EQ(dm.GetNumPages(), 0u);
}

}  // namespace shilmandb
