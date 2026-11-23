#include <gtest/gtest.h>
#include "common/exception.hpp"
#include "executor/parallel/column_buffer.hpp"
#include "executor/parallel/memory_budget.hpp"
#include "executor/parallel/spill_partitions.hpp"
#include "executor/parallel/temp_file.hpp"
#include "storage/columnar/string_arena.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace shilmandb::parallel {

namespace {

// Runs fn and returns what it threw, or "" if it didn't throw.
template <typename Fn>
std::string ErrorOf(Fn&& fn) {
    try {
        fn();
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

// A double's bits, so NaN payloads and -0.0 compare exactly.
uint64_t BitsOf(double d) {
    uint64_t bits = 0;
    std::memcpy(&bits, &d, sizeof(bits));
    return bits;
}

// A double with exactly these bits.
double FromBits(uint64_t bits) {
    double d = 0;
    std::memcpy(&d, &bits, sizeof(d));
    return d;
}

// Turns write failures on for one scope, so a failed assertion can't leak them.
struct FailWritesScope {
    explicit FailWritesScope(int error_number) { TempFile::FailWritesForTest(error_number); }
    ~FailWritesScope() { TempFile::FailWritesForTest(0); }
};

// Keeps a second descriptor on the next temp file opened, before it's unlinked.
struct SecondDescriptorScope {
    int fd{-1};
    SecondDescriptorScope() {
        TempFile::SetOpenHookForTest([this](const std::string& path) { fd = ::open(path.c_str(), O_RDWR); });
    }
    ~SecondDescriptorScope() {
        TempFile::SetOpenHookForTest(nullptr);
        if (fd >= 0) ::close(fd);
    }
    // Flips every bit of the byte at offset; false if it couldn't.
    [[nodiscard]] bool FlipByte(uint64_t offset) const {
        unsigned char b = 0;
        if (::pread(fd, &b, 1, static_cast<off_t>(offset)) != 1) return false;
        b = static_cast<unsigned char>(~b);
        return ::pwrite(fd, &b, 1, static_cast<off_t>(offset)) == 1;
    }
};

// One row of the mixed test table, owned.
using Row = std::tuple<int64_t, std::string, uint64_t>;  // id, text, double bits

const std::vector<ValueArray> kRowArrays = {ValueArray::I64, ValueArray::STR, ValueArray::F64};

}  // namespace

class ParallelTempFileTest : public ::testing::Test {
protected:
    std::string dir_;

    void SetUp() override {
        dir_ = (std::filesystem::temp_directory_path() / ("shilmandb_temp_file_test_" + std::to_string(::getpid()))).string();
        std::filesystem::remove_all(dir_);
    }

    void TearDown() override { std::filesystem::remove_all(dir_); }

    // How many entries the directory holds right now.
    size_t EntriesIn(const std::string& dir) const {
        size_t n = 0;
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            (void)entry;
            ++n;
        }
        return n;
    }

    // Writes one small block of one I64 column.
    static uint64_t WriteSome(TempFile& f) {
        ColumnBuffer col;
        col.i64 = {1, 2, 3, 4};
        const uint64_t at = f.WriteBlock({&col}, {ValueArray::I64}, 0, col.i64.size());
        f.FinishWriting();
        return at;
    }

    // Reads every block from offset 0 into one set of buffers; returns the block count.
    static size_t ReadAll(const TempFile& f, std::vector<ColumnBuffer>& out, const std::vector<ValueArray>& arrays, StringArena& arena) {
        uint64_t offset = 0;
        size_t blocks = 0;
        while (f.ReadBlock(&offset, out, arrays, arena)) ++blocks;
        EXPECT_EQ(offset, f.Bytes());
        return blocks;
    }
};

TEST_F(ParallelTempFileTest, RoundTripsEveryKind) {
    const std::vector<ValueArray> arrays = {ValueArray::I64, ValueArray::I128, ValueArray::F64, ValueArray::STR};
    const __int128 big = static_cast<__int128>(std::numeric_limits<int64_t>::max()) * 1000 + 7;
    const uint64_t nan_bits = 0x7ff8000000001234ULL;

    std::vector<int64_t> want_i64;
    std::vector<__int128> want_i128;
    std::vector<uint64_t> want_f64;
    std::vector<std::string> want_str;
    std::vector<uint64_t> offsets;

    SpillCounters counters;
    TempFile f(dir_, &counters);
    {
        // The source buffers and their arena die before the read.
        StringArena arena;
        ColumnBuffer c0, c1, c2, c3;
        const std::vector<std::string> texts = {
            "", "caf\xc3\xa9 \xe6\x97\xa5\xe6\x9c\xac \xf0\x9f\x98\x80", std::string(8000, 'x'), std::string(3 << 20, 'y'), "plain", "z"};
        const std::vector<double> doubles = {FromBits(nan_bits), -0.0, 0.0, 1.5, -2.25e300, std::numeric_limits<double>::infinity()};
        for (int i = 0; i < 12; ++i) {
            const int64_t v64 = i % 2 == 0 ? std::numeric_limits<int64_t>::min() + i : int64_t{i} * 1'000'000'007;
            const __int128 v128 = i % 3 == 0 ? -big - i : big + i;
            const double d = doubles[static_cast<size_t>(i) % doubles.size()];
            std::string s = texts[static_cast<size_t>(i) % texts.size()];
            if (!s.empty()) s[0] = static_cast<char>('a' + i);
            c0.i64.push_back(v64);
            c1.i128.push_back(v128);
            c2.f64.push_back(d);
            c3.str.push_back(arena.Copy(s));
            want_i64.push_back(v64);
            want_i128.push_back(v128);
            want_f64.push_back(BitsOf(d));
            want_str.push_back(s);
        }
        const std::vector<const ColumnBuffer*> cols = {&c0, &c1, &c2, &c3};
        offsets.push_back(f.WriteBlock(cols, arrays, 0, 4));
        offsets.push_back(f.WriteBlock(cols, arrays, 4, 5));
        offsets.push_back(f.WriteBlock(cols, arrays, 9, 3));
        f.FinishWriting();
    }
    EXPECT_EQ(offsets[0], 0u);
    EXPECT_LT(offsets[0], offsets[1]);
    EXPECT_LT(offsets[1], offsets[2]);
    EXPECT_LT(offsets[2], f.Bytes());
    EXPECT_EQ(counters.files.load(), 1u);
    EXPECT_EQ(counters.bytes_written.load(), f.Bytes());
    EXPECT_GT(f.Bytes(), uint64_t{3} << 20);

    StringArena arena;
    std::vector<ColumnBuffer> out;
    ASSERT_EQ(ReadAll(f, out, arrays, arena), 3u);
    ASSERT_EQ(out.size(), 4u);
    EXPECT_EQ(out[0].i64, want_i64);
    EXPECT_TRUE(out[1].i128 == want_i128);
    ASSERT_EQ(out[2].f64.size(), want_f64.size());
    for (size_t i = 0; i < want_f64.size(); ++i) EXPECT_EQ(BitsOf(out[2].f64[i]), want_f64[i]) << i;
    ASSERT_EQ(out[3].str.size(), want_str.size());
    for (size_t i = 0; i < want_str.size(); ++i) EXPECT_TRUE(View(out[3].str[i]) == want_str[i]) << i;
    EXPECT_EQ(counters.bytes_read.load(), f.Bytes());
    EXPECT_GT(counters.nanos.load(), 0);

    // A block read from its own offset gives just its rows.
    std::vector<ColumnBuffer> second;
    uint64_t at = offsets[1];
    ASSERT_TRUE(f.ReadBlock(&at, second, arrays, arena));
    EXPECT_EQ(at, offsets[2]);
    ASSERT_EQ(second[0].i64.size(), 5u);
    EXPECT_EQ(second[0].i64[0], want_i64[4]);
}

TEST_F(ParallelTempFileTest, DamagedBlockIsRefused) {
    const std::vector<ValueArray> arrays = {ValueArray::I64, ValueArray::STR};
    StringArena src;
    ColumnBuffer ids, texts;
    for (int i = 0; i < 100; ++i) {
        ids.i64.push_back(i);
        texts.str.push_back(src.Copy("row " + std::to_string(i)));
    }
    // Bytes to flip: the row count, the payload length's top byte, a payload byte.
    struct Case {
        const char* what;
        uint64_t byte;
        const char* error;
    };
    const std::vector<Case> cases = {{"row count", 8, "damaged temp file: checksum mismatch"},
                                     {"payload length", 23, "damaged temp file: block past the end"},
                                     {"payload", 24 + 50, "damaged temp file: checksum mismatch"}};
    for (const auto& [what, byte, want] : cases) {
        SCOPED_TRACE(what);
        SecondDescriptorScope second;
        TempFile f(dir_, nullptr);
        ASSERT_GE(second.fd, 0);
        ASSERT_EQ(f.WriteBlock({&ids, &texts}, arrays, 0, 100), 0u);
        f.FinishWriting();
        ASSERT_TRUE(second.FlipByte(byte));
        StringArena arena;
        std::vector<ColumnBuffer> out;
        uint64_t offset = 0;
        EXPECT_EQ(ErrorOf([&] { (void)f.ReadBlock(&offset, out, arrays, arena); }), want);
        EXPECT_TRUE(out.empty());
    }

    // The wrong column count is refused too.
    TempFile f(dir_, nullptr);
    (void)f.WriteBlock({&ids, &texts}, arrays, 0, 100);
    f.FinishWriting();
    StringArena arena;
    std::vector<ColumnBuffer> out;
    uint64_t offset = 0;
    EXPECT_EQ(ErrorOf([&] { (void)f.ReadBlock(&offset, out, {ValueArray::I64}, arena); }),
              "damaged temp file: block has 2 columns, expected 1");
}

TEST_F(ParallelTempFileTest, FullDiskFailsCleanly) {
    {
        FailWritesScope fail(ENOSPC);
        TempFile f(dir_, nullptr);
        EXPECT_THROW(WriteSome(f), DatabaseException);
        TempFile g(dir_, nullptr);
        EXPECT_EQ(ErrorOf([&] { (void)WriteSome(g); }), "out of temp space in " + dir_ + " while spilling: No space left on device");

        // A big piece skips the write buffer and fails the same way.
        StringArena arena;
        ColumnBuffer big;
        big.str.push_back(arena.Copy(std::string(1 << 20, 'b')));
        TempFile h(dir_, nullptr);
        EXPECT_EQ(ErrorOf([&] { (void)h.WriteBlock({&big}, {ValueArray::STR}, 0, 1); }),
                  "out of temp space in " + dir_ + " while spilling: No space left on device");

        // Through a partitioned writer.
        PartitionedWriter writer({ValueArray::I64}, dir_, nullptr, nullptr);
        ColumnBuffer col;
        col.i64 = {7, 8, 9};
        const std::vector<uint32_t> rows = {0, 1, 2};
        writer.Add(3, {&col}, rows.data(), rows.size());
        EXPECT_EQ(ErrorOf([&] { writer.Flush(); }), "out of temp space in " + dir_ + " while spilling: No space left on device");
        EXPECT_EQ(EntriesIn(dir_), 0u);
    }
    EXPECT_EQ(EntriesIn(dir_), 0u);

    // Writes work again once the scope ends.
    TempFile f(dir_, nullptr);
    EXPECT_NO_THROW(WriteSome(f));

    // A directory that can't be made: its parent is a plain file.
    const std::string file = dir_ + "/plain";
    ASSERT_GE(::close(::open(file.c_str(), O_CREAT | O_WRONLY, 0600)), 0);
    const std::string bad = file + "/spill";
    EXPECT_EQ(ErrorOf([&] { TempFile t(bad, nullptr); }), "out of temp space in " + bad + " while spilling: Not a directory");
}

TEST_F(ParallelTempFileTest, NothingLeftOnDisk) {
    ASSERT_FALSE(std::filesystem::exists(dir_));
    {
        TempFile f(dir_, nullptr);
        ASSERT_TRUE(std::filesystem::is_directory(dir_));
        EXPECT_EQ(EntriesIn(dir_), 0u);
        ColumnBuffer col;
        for (int i = 0; i < 100'000; ++i) col.i64.push_back(i);
        (void)f.WriteBlock({&col}, {ValueArray::I64}, 0, col.i64.size());
        EXPECT_EQ(EntriesIn(dir_), 0u);
        f.FinishWriting();
        EXPECT_EQ(EntriesIn(dir_), 0u);

        PartitionedWriter writer({ValueArray::I64}, dir_, nullptr, nullptr);
        std::vector<uint32_t> rows(col.i64.size());
        for (uint32_t i = 0; i < rows.size(); ++i) rows[i] = i;
        writer.Add(0, {&col}, rows.data(), rows.size());
        writer.Flush();
        EXPECT_EQ(EntriesIn(dir_), 0u);
        std::optional<SpillFile> spilled = writer.TakeFile();
        ASSERT_TRUE(spilled.has_value());
        EXPECT_EQ(EntriesIn(dir_), 0u);
    }
    EXPECT_EQ(EntriesIn(dir_), 0u);
}

TEST_F(ParallelTempFileTest, WorkersMakeTheDirectoryTogether) {
    std::vector<std::thread> threads;
    std::vector<std::string> errors(8);
    for (size_t t = 0; t < errors.size(); ++t) {
        threads.emplace_back([&, t] { errors[t] = ErrorOf([&] { TempFile f(dir_ + "/shared", nullptr); }); });
    }
    for (auto& th : threads) th.join();
    for (const auto& e : errors) EXPECT_EQ(e, "");
    EXPECT_EQ(EntriesIn(dir_ + "/shared"), 0u);
}

TEST_F(ParallelTempFileTest, PartitionsAreStable) {
    std::mt19937_64 rng(42);
    std::vector<uint64_t> level0(kSpillPartitions), level1(kSpillPartitions), level2(kSpillPartitions);
    std::set<size_t> spread_of_one;  // level-1 partitions of hashes in level-0 partition 5
    size_t same = 0;
    constexpr int kHashes = 200'000;
    for (int i = 0; i < kHashes; ++i) {
        const uint64_t h = rng();
        const size_t p0 = SpillPartitionOf(h, 0);
        const size_t p1 = SpillPartitionOf(h, 1);
        ASSERT_LT(p0, kSpillPartitions);
        ASSERT_LT(p1, kSpillPartitions);
        EXPECT_EQ(SpillPartitionOf(h, 0), p0);
        ++level0[p0];
        ++level1[p1];
        ++level2[SpillPartitionOf(h, 2)];
        if (p0 == 5) spread_of_one.insert(p1);
        if (p0 == p1) ++same;
    }
    EXPECT_EQ(spread_of_one.size(), kSpillPartitions);
    // Levels agree about 1 time in 64, as independent picks would.
    EXPECT_LT(same, static_cast<size_t>(kHashes / 64 * 1.3));
    for (size_t p = 0; p < kSpillPartitions; ++p) {
        EXPECT_GT(level0[p], 0u) << p;
        EXPECT_GT(level1[p], 0u) << p;
        EXPECT_GT(level2[p], 0u) << p;
    }

    // Hashes that share the in-memory tables' top and low bits still spread.
    std::set<size_t> spread_of_shared;
    for (uint64_t mid = 0; mid < 4096; ++mid) {
        const uint64_t h = (uint64_t{0x2a} << 58) | (mid << 20) | 0x5;
        spread_of_shared.insert(SpillPartitionOf(h, 0));
    }
    EXPECT_EQ(spread_of_shared.size(), kSpillPartitions);
}

namespace {

// Random rows: ids, texts of 0-100 bytes with a few past a partition's buffer, doubles.
struct RandomTable {
    ColumnBuffer ids, texts, doubles;
    StringArena arena;
    std::vector<Row> rows;

    RandomTable(size_t n, uint64_t seed, size_t big_every) {
        std::mt19937_64 rng(seed);
        for (size_t i = 0; i < n; ++i) {
            const auto id = static_cast<int64_t>(rng());
            const size_t len = big_every != 0 && i % big_every == 0 ? 100'000 : rng() % 101;
            std::string s(len, static_cast<char>('a' + i % 26));
            if (len > 3) s[1] = static_cast<char>(rng());
            const double d = i % 97 == 0 ? FromBits(0x7ff800000000beefULL) : (i % 89 == 0 ? -0.0 : static_cast<double>(id) / 3);
            ids.i64.push_back(id);
            texts.str.push_back(arena.Copy(s));
            doubles.f64.push_back(d);
            rows.emplace_back(id, s, BitsOf(d));
        }
    }

    [[nodiscard]] std::vector<const ColumnBuffer*> Columns() const { return {&ids, &texts, &doubles}; }

    [[nodiscard]] size_t PartitionOfRow(size_t i) const {
        return SpillPartitionOf(static_cast<uint64_t>(ids.i64[i]) * 0x9e3779b97f4a7c15ULL, 0);
    }
};

// Adds every row to its partition: some one at a time, some grouped by partition.
void AddRows(PartitionedWriter& writer, const RandomTable& table) {
    const auto cols = table.Columns();
    const size_t n = table.rows.size();
    std::vector<std::vector<uint32_t>> grouped(kSpillPartitions);
    for (uint32_t i = 0; i < n; ++i) {
        if (i % 2 == 0) {
            writer.Add(table.PartitionOfRow(i), cols, &i, 1);
        } else {
            grouped[table.PartitionOfRow(i)].push_back(i);
        }
    }
    for (size_t p = 0; p < kSpillPartitions; ++p) writer.Add(p, cols, grouped[p].data(), grouped[p].size());
}

// Checks each partition reads back exactly its rows, as multisets.
void ExpectPartitionsMatch(const SpillFile& spilled, const RandomTable& table) {
    std::vector<std::multiset<Row>> want(kSpillPartitions);
    for (size_t i = 0; i < table.rows.size(); ++i) want[table.PartitionOfRow(i)].insert(table.rows[i]);
    size_t used = 0;
    for (size_t p = 0; p < kSpillPartitions; ++p) {
        StringArena arena;
        std::vector<ColumnBuffer> out;
        spilled.ReadPartition(p, out, arena);
        ASSERT_EQ(out.size(), 3u);
        std::multiset<Row> got;
        for (size_t r = 0; r < out[0].i64.size(); ++r) {
            got.emplace(out[0].i64[r], std::string(View(out[1].str[r])), BitsOf(out[2].f64[r]));
        }
        EXPECT_EQ(spilled.rows[p], want[p].size()) << p;
        EXPECT_TRUE(got == want[p]) << "partition " << p;
        if (!want[p].empty()) ++used;
    }
    EXPECT_EQ(used, kSpillPartitions);
}

}  // namespace

TEST_F(ParallelTempFileTest, PartitionedWriterRoundTrips) {
    const RandomTable table(100'000, 7, 25'000);
    SpillCounters counters;
    MemoryReservation reservation;
    PartitionedWriter writer(kRowArrays, dir_, &counters, &reservation);
    AddRows(writer, table);
    EXPECT_LE(reservation.Bytes(), kSpillPartitions * PartitionedWriter::kPartitionBufferBytes + TempFile::kWriteBufferBytes);
    EXPECT_GT(reservation.Bytes(), 0u);
    std::optional<SpillFile> spilled = writer.TakeFile();
    ASSERT_TRUE(spilled.has_value());
    EXPECT_EQ(reservation.Bytes(), 0u);
    EXPECT_EQ(counters.files.load(), 1u);
    EXPECT_EQ(counters.bytes_written.load(), spilled->file.Bytes());

    uint64_t bytes = 0;
    size_t many_blocks = 0;
    for (size_t p = 0; p < kSpillPartitions; ++p) {
        bytes += spilled->bytes[p];
        if (spilled->blocks[p].size() > 1) ++many_blocks;
    }
    EXPECT_EQ(bytes, spilled->file.Bytes());
    EXPECT_GT(many_blocks, kSpillPartitions / 2);
    ExpectPartitionsMatch(*spilled, table);
    EXPECT_EQ(counters.bytes_read.load(), spilled->file.Bytes());

    // A writer that got no rows has no file.
    PartitionedWriter idle(kRowArrays, dir_, &counters, &reservation);
    EXPECT_FALSE(idle.TakeFile().has_value());
    EXPECT_EQ(counters.files.load(), 1u);
}

TEST_F(ParallelTempFileTest, PartitionedWriterFlushesWhenRefused) {
    // The worker's whole floor is the only memory; the reservation already uses most of it.
    MemoryBudget budget(kWorkerMinimumBytes, 1);
    for (const size_t room : {size_t{100} << 10, size_t{0}}) {
        SCOPED_TRACE(room);
        const RandomTable table(room == 0 ? 3'000 : 40'000, 11, 10'000);
        MemoryReservation reservation(&budget, 0);
        const size_t held = kWorkerMinimumBytes - room;
        ASSERT_TRUE(reservation.Grow(held));
        PartitionedWriter writer(kRowArrays, dir_, nullptr, &reservation);
        AddRows(writer, table);
        EXPECT_LE(reservation.Bytes(), kWorkerMinimumBytes);
        std::optional<SpillFile> spilled = writer.TakeFile();
        ASSERT_TRUE(spilled.has_value());
        EXPECT_EQ(reservation.Bytes(), held);
        ExpectPartitionsMatch(*spilled, table);
        // With no room, every row is its own block; with a little, buffers still fill.
        size_t blocks = 0;
        for (size_t p = 0; p < kSpillPartitions; ++p) blocks += spilled->blocks[p].size();
        if (room == 0) {
            EXPECT_EQ(blocks, table.rows.size());
        } else {
            EXPECT_LT(blocks, table.rows.size() / 4);
        }
    }
    EXPECT_EQ(budget.HeldBytes(), 0u);
}

}  // namespace shilmandb::parallel
