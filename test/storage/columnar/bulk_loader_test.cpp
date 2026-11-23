#include <gtest/gtest.h>
#include "buffer/buffer_pool_manager.hpp"
#include "buffer/lru_eviction_policy.hpp"
#include "catalog/catalog.hpp"
#include "catalog/schema.hpp"
#include "common/exception.hpp"
#include "engine/database.hpp"
#include "storage/columnar/bulk_loader.hpp"
#include "storage/columnar/columnar_table.hpp"
#include "storage/columnar/field_parser.hpp"
#include "storage/disk_manager.hpp"
#include "types/value.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

namespace shilmandb {
namespace {

constexpr size_t kRows = 150000;

// Compares two vectors row by row; doubles by their bits.
::testing::AssertionResult SameValues(const ColumnVector& a, const ColumnVector& b) {
    if (a.Size() != b.Size()) {
        return ::testing::AssertionFailure() << "sizes " << a.Size() << " vs " << b.Size();
    }
    for (size_t i = 0; i < a.Size(); ++i) {
        if (a.IsNull(i) != b.IsNull(i)) return ::testing::AssertionFailure() << "NULL flag at row " << i;
        if (a.IsNull(i)) continue;
        bool same = true;
        switch (a.Kind()) {
            case PhysicalKind::INT64:  same = a.GetInt(i) == b.GetInt(i); break;
            case PhysicalKind::INT128: same = a.GetWide(i) == b.GetWide(i); break;
            case PhysicalKind::DOUBLE: {
                const double x = a.GetDouble(i), y = b.GetDouble(i);
                same = std::memcmp(&x, &y, sizeof(double)) == 0;
                break;
            }
            case PhysicalKind::STRING: same = a.GetString(i) == b.GetString(i); break;
        }
        if (!same) return ::testing::AssertionFailure() << "value at row " << i;
    }
    return ::testing::AssertionSuccess();
}

// Everything about two chunks except where their pages sit.
::testing::AssertionResult SameChunk(const ChunkMeta& a, const ChunkMeta& b) {
    if (a.encoding != b.encoding) return ::testing::AssertionFailure() << "encoding";
    if (a.page_count != b.page_count) return ::testing::AssertionFailure() << "page_count";
    if (a.dict_page_count != b.dict_page_count) return ::testing::AssertionFailure() << "dict_page_count";
    if (a.value_count != b.value_count) return ::testing::AssertionFailure() << "value_count";
    if (a.null_count != b.null_count) return ::testing::AssertionFailure() << "null_count";
    if (a.page_first_rows != b.page_first_rows) return ::testing::AssertionFailure() << "page_first_rows";
    if (a.dict_page_first_entries != b.dict_page_first_entries) return ::testing::AssertionFailure() << "dict entries";
    if (a.dict_size != b.dict_size) return ::testing::AssertionFailure() << "dict_size";
    if (a.min != b.min || a.max != b.max) return ::testing::AssertionFailure() << "min / max";
    if (a.constant != b.constant) return ::testing::AssertionFailure() << "constant";
    return ::testing::AssertionSuccess();
}

// The five-column table the bigger tests load.
Schema MixedSchema() {
    return Schema({Column("a", ColumnType::Integer()), Column("b", ColumnType::Numeric(15, 2)),
                   Column("c", ColumnType::Date()), Column("d", ColumnType::Varchar(10)),
                   Column("e", ColumnType::Boolean())});
}

// Row i's fields: a unique int, cents, a date, a word and a flag.
std::vector<std::string> MixedFields(size_t i) {
    char buf[64];
    std::vector<std::string> f;
    f.push_back(std::to_string(static_cast<long long>(i) * 3 - 200000));
    const long long cents = static_cast<long long>((i * 37) % 10000000);
    std::snprintf(buf, sizeof(buf), "%s%lld.%02lld", i % 7 == 0 ? "-" : "", cents / 100, cents % 100);
    f.emplace_back(buf);
    std::snprintf(buf, sizeof(buf), "%04zu-%02zu-%02zu", 1992 + i % 7, 1 + i % 12, 1 + i % 28);
    f.emplace_back(buf);
    f.push_back("w" + std::to_string(i % 5000));
    f.emplace_back(i % 3 == 0 ? "true" : "f");
    return f;
}

// Joins fields into one line.
std::string Join(const std::vector<std::string>& fields, char delim = '|') {
    std::string line;
    for (size_t c = 0; c < fields.size(); ++c) {
        if (c > 0) line += delim;
        line += fields[c];
    }
    return line;
}

// A file of two integer columns, one row per line, with chosen lines replaced.
std::string IntLines(size_t lines, const std::vector<std::pair<size_t, std::string>>& replace = {}) {
    std::string text;
    for (size_t n = 1; n <= lines; ++n) {
        std::string line = std::to_string(n) + "|" + std::to_string(n * 2);
        for (const auto& r : replace) {
            if (r.first == n) line = r.second;
        }
        text += line + "\n";
    }
    return text;
}

// Two plain integer columns, a and b.
Schema IntSchema() { return Schema({Column("a", ColumnType::Integer()), Column("b", ColumnType::Integer())}); }

// An integer and a text column, for lines of any length.
Schema TextSchema() { return Schema({Column("a", ColumnType::Integer()), Column("b", ColumnType::Text())}); }

constexpr size_t kBlock = size_t{1} << 20;  // the reader's block size

// Lowers the loader's value cap for one test, and a text schema's line limit.
class ValueCapScope {
public:
    explicit ValueCapScope(size_t bytes) { BulkLoadValueCapForTest(bytes); }
    ~ValueCapScope() { BulkLoadValueCapForTest(0); }
    ValueCapScope(const ValueCapScope&) = delete;
    ValueCapScope& operator=(const ValueCapScope&) = delete;
};

// The line limit for a cap and a column count, as the loader sets it.
// A schema with no string column gets cap 0.
size_t LineLimit(size_t cap, size_t columns) { return cap + columns * (kMaxValueBytes + 1); }

// n letters that never repeat in a short cycle, set by seed.
std::string Letters(size_t n, uint32_t seed) {
    std::string s(n, 'a');
    uint32_t x = seed * 2654435761u + 1;
    for (char& c : s) {
        x = x * 1664525u + 1013904223u;
        c = static_cast<char>('a' + (x >> 24) % 26);
    }
    return s;
}

// Valid TextSchema lines totalling exactly `bytes` bytes (at least 4).
std::string FillerLines(size_t bytes, size_t& lines) {
    std::string text;
    while (bytes - text.size() > 300) {
        text += "0|" + std::string(200, 's') + "\n";
        ++lines;
    }
    text += "0|" + std::string(bytes - text.size() - 3, 's') + "\n";
    ++lines;
    return text;
}

// Throws on every page write, as a failing disk would.
class FailingWrites : public WriteObserver {
public:
    explicit FailingWrites(int kind) : kind_(kind) {}
    void OnWrite(page_id_t, const char*) override {
        if (kind_ == 0) throw IOException("disk full");
        if (kind_ == 1) throw DatabaseException("bad page");
        throw std::runtime_error("not ours");
    }
    void OnSync() override {}

private:
    int kind_;
};

class BulkLoaderTest : public ::testing::Test {
protected:
    std::filesystem::path dir_;
    std::unique_ptr<DiskManager> disk_;

    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() / ("shilmandb_bulkload_" + std::to_string(::getpid()));
        std::filesystem::remove_all(dir_);
        std::filesystem::create_directories(dir_);
        disk_ = std::make_unique<DiskManager>((dir_ / "t.db").string());
    }

    void TearDown() override {
        disk_.reset();
        std::filesystem::remove_all(dir_);
    }

    // Writes text to a file in the temp dir and returns its path.
    std::string WriteFile(const std::string& name, const std::string& text) {
        const std::string path = (dir_ / name).string();
        std::ofstream out(path, std::ios::binary);
        out << text;
        return path;
    }

    // The mixed table's text, plus each column's expected values parsed from it.
    std::string MixedFile(std::vector<ColumnVector>& expected) {
        const Schema schema = MixedSchema();
        for (const Column& c : schema.GetColumns()) expected.emplace_back(c.storage_type);
        std::string text;
        for (size_t i = 0; i < kRows; ++i) {
            const auto fields = MixedFields(i);
            for (size_t c = 0; c < fields.size(); ++c) {
                const auto err = ParseField(fields[c], expected[c]);
                EXPECT_FALSE(err.has_value()) << fields[c];
            }
            text += Join(fields) + "\n";
        }
        return WriteFile("mixed.tbl", text);
    }

    std::unique_ptr<BufferPoolManager> MakePool(size_t frames = 64) {
        return std::make_unique<BufferPoolManager>(frames, disk_.get(), std::make_unique<LRUEvictionPolicy>(frames));
    }

    // Reads every column, with its row groups back to back.
    std::vector<ColumnVector> ReadAll(const ColumnarTableMeta& meta) {
        auto pool = MakePool();
        ColumnarTable table(pool.get(), meta);
        std::vector<ColumnVector> out;
        for (uint32_t c = 0; c < meta.columns.size(); ++c) {
            out.emplace_back(meta.columns[c].type);
            for (size_t g = 0; g < table.RowGroupCount(); ++g) table.ReadChunk(g, c, out.back());
        }
        return out;
    }

    // The message BulkLoad throws, or "" when it doesn't.
    std::string LoadError(const Schema& schema, const std::string& path, const BulkLoadOptions& options) {
        try {
            (void)BulkLoad(disk_.get(), "t", schema, path, options);
        } catch (const DatabaseException& e) {
            return e.what();
        }
        return "";
    }
};

// Default options with a set thread count.
BulkLoadOptions Threads(unsigned n) {
    BulkLoadOptions o;
    o.threads = n;
    return o;
}

TEST_F(BulkLoaderTest, LoadsAndReadsBack) {
    std::vector<ColumnVector> expected;
    const std::string path = MixedFile(expected);
    const ColumnarTableMeta meta = BulkLoad(disk_.get(), "t", MixedSchema(), path, Threads(4));

    EXPECT_EQ(meta.oid, 0u);
    EXPECT_EQ(meta.name, "t");
    ASSERT_EQ(meta.columns.size(), 5u);
    EXPECT_EQ(meta.columns[1].name, "b");
    EXPECT_EQ(meta.columns[1].number, 2u);
    EXPECT_EQ(meta.columns[1].type, ColumnType::Numeric(15, 2));
    EXPECT_EQ(meta.row_count, kRows);
    EXPECT_EQ(meta.stats.row_count, kRows);
    ASSERT_EQ(meta.row_groups.size(), 3u);
    EXPECT_EQ(meta.row_groups[0].row_count, 65536u);
    EXPECT_EQ(meta.row_groups[1].row_count, 65536u);
    EXPECT_EQ(meta.row_groups[2].row_count, 18928u);
    for (const RowGroupMeta& g : meta.row_groups) {
        ASSERT_EQ(g.chunks.size(), 5u);
        for (const ChunkMeta& chunk : g.chunks) EXPECT_EQ(chunk.value_count, g.row_count);
    }

    const auto got = ReadAll(meta);
    for (size_t c = 0; c < expected.size(); ++c) EXPECT_TRUE(SameValues(got[c], expected[c])) << "column " << c;
}

TEST_F(BulkLoaderTest, ThreadCountDoesNotChangeTheTable) {
    std::vector<ColumnVector> expected;
    const std::string path = MixedFile(expected);
    const ColumnarTableMeta one = BulkLoad(disk_.get(), "t", MixedSchema(), path, Threads(1));
    const ColumnarTableMeta eight = BulkLoad(disk_.get(), "t", MixedSchema(), path, Threads(8));

    EXPECT_EQ(one.row_count, eight.row_count);
    ASSERT_EQ(one.row_groups.size(), eight.row_groups.size());
    for (size_t g = 0; g < one.row_groups.size(); ++g) {
        EXPECT_EQ(one.row_groups[g].row_count, eight.row_groups[g].row_count);
        for (size_t c = 0; c < one.columns.size(); ++c) {
            EXPECT_TRUE(SameChunk(one.row_groups[g].chunks[c], eight.row_groups[g].chunks[c]))
                << "group " << g << " column " << c;
        }
    }

    EXPECT_EQ(one.stats.row_count, eight.stats.row_count);
    EXPECT_EQ(one.stats.page_count, eight.stats.page_count);
    EXPECT_EQ(one.stats.distinct_counts, eight.stats.distinct_counts);
    ASSERT_EQ(one.stats.min_max.size(), eight.stats.min_max.size());
    for (const auto& [name, range] : one.stats.min_max) {
        ASSERT_EQ(eight.stats.min_max.count(name), 1u) << name;
        const MinMax& other = eight.stats.min_max.at(name);
        EXPECT_EQ(range.min.type_, other.min.type_) << name;
        EXPECT_TRUE(range.min == other.min) << name;
        EXPECT_TRUE(range.max == other.max) << name;
    }

    const auto a = ReadAll(one);
    const auto b = ReadAll(eight);
    for (size_t c = 0; c < a.size(); ++c) {
        EXPECT_TRUE(SameValues(a[c], b[c])) << "column " << c;
        EXPECT_TRUE(SameValues(a[c], expected[c])) << "column " << c;
    }
}

TEST_F(BulkLoaderTest, TrailingDelimiterAndEmptyLines) {
    const Schema schema({Column("a", ColumnType::Integer()), Column("b", ColumnType::Varchar(10))});
    const auto meta = BulkLoad(disk_.get(), "t", schema, WriteFile("t.tbl", "1|a|\n\n2|b|\n"), Threads(2));
    ASSERT_EQ(meta.row_count, 2u);
    const auto got = ReadAll(meta);
    EXPECT_EQ(got[0].GetInt(0), 1);
    EXPECT_EQ(got[0].GetInt(1), 2);
    EXPECT_EQ(got[1].GetString(0), "a");
    EXPECT_EQ(got[1].GetString(1), "b");
}

TEST_F(BulkLoaderTest, LineEndings) {
    const Schema schema({Column("a", ColumnType::Integer()), Column("b", ColumnType::Varchar(10))});
    const auto plain = BulkLoad(disk_.get(), "t", schema, WriteFile("lf.tbl", "1|x\n2|yy|\n3|zzz\n"), Threads(1));
    const auto crlf = BulkLoad(disk_.get(), "t", schema, WriteFile("crlf.tbl", "1|x\r\n2|yy|\r\n\r\n3|zzz"), Threads(1));
    const auto bare = BulkLoad(disk_.get(), "t", schema, WriteFile("bare.tbl", "1|x\n2|yy|\n3|zzz"), Threads(1));

    ASSERT_EQ(plain.row_count, 3u);
    EXPECT_EQ(crlf.row_count, 3u);
    EXPECT_EQ(bare.row_count, 3u);
    const auto p = ReadAll(plain), c = ReadAll(crlf), b = ReadAll(bare);
    for (size_t col = 0; col < 2; ++col) {
        EXPECT_TRUE(SameValues(p[col], c[col])) << "column " << col;
        EXPECT_TRUE(SameValues(p[col], b[col])) << "column " << col;
    }
    EXPECT_EQ(p[1].GetString(2), "zzz");
}

TEST_F(BulkLoaderTest, BadFieldNamesLineAndColumn) {
    const std::string path = WriteFile("bad.tbl", IntLines(100000, {{70001, "abc|5"}}));
    const std::string msg = LoadError(IntSchema(), path, Threads(4));
    EXPECT_EQ(msg.rfind("t: line 70001, column a: ", 0), 0u) << msg;
}

TEST_F(BulkLoaderTest, WrongFieldCount) {
    EXPECT_EQ(LoadError(IntSchema(), WriteFile("long.tbl", IntLines(5, {{3, "1|2|3"}})), Threads(2)),
              "t: line 3: expected 2 fields, got 3");
    EXPECT_EQ(LoadError(IntSchema(), WriteFile("short.tbl", IntLines(5, {{4, "7"}})), Threads(2)),
              "t: line 4: expected 2 fields, got 1");
    // Only one trailing delimiter is dropped, so the second makes an empty field.
    EXPECT_EQ(LoadError(IntSchema(), WriteFile("two.tbl", IntLines(5, {{2, "1|2||"}})), Threads(2)),
              "t: line 2: expected 2 fields, got 3");
}

TEST_F(BulkLoaderTest, FirstErrorWins) {
    const std::string path = WriteFile("two_errors.tbl", IntLines(100000, {{5, "x|1"}, {100000, "y|1"}}));
    for (int attempt = 0; attempt < 3; ++attempt) {
        const std::string msg = LoadError(IntSchema(), path, Threads(8));
        EXPECT_EQ(msg.rfind("t: line 5, column a: ", 0), 0u) << msg;
    }
}

// Batch 0 fails on its last row, later batches on their first row.
TEST_F(BulkLoaderTest, FirstErrorWinsOutOfOrder) {
    constexpr size_t kGroups = 8;
    std::vector<std::pair<size_t, std::string>> bad = {{kRowGroupSize, "x|1"}};
    for (size_t g = 1; g < kGroups; ++g) bad.emplace_back(g * kRowGroupSize + 1, "y|1");
    const std::string path = WriteFile("late_errors.tbl", IntLines(kGroups * kRowGroupSize, bad));
    for (int attempt = 0; attempt < 10; ++attempt) {
        const std::string msg = LoadError(IntSchema(), path, Threads(8));
        EXPECT_EQ(msg.rfind("t: line 65536, column a: ", 0), 0u) << msg;
    }
}

// A \r\n split across the reader's 1 MiB blocks, and lines that straddle them.
TEST_F(BulkLoaderTest, LinesAcrossReadBlocks) {
    const Schema schema({Column("a", ColumnType::Integer()), Column("b", ColumnType::Text())});
    std::vector<std::string> rows;
    size_t crlf_bytes = 0;
    while (crlf_bytes + 400 < kBlock) {
        rows.push_back(std::to_string(rows.size()) + "|" + std::string(200, 's'));
        crlf_bytes += rows.back().size() + 2;
    }
    // This row's \r is the block's last byte; its \n starts the next block.
    const std::string head = std::to_string(rows.size()) + "|";
    rows.push_back(head + std::string(kBlock - 1 - crlf_bytes - head.size(), 'p'));
    for (size_t i = 0; i < 500; ++i) rows.push_back(std::to_string(rows.size()) + "|" + std::string(3000, 'L'));

    std::string crlf, lf;
    std::vector<ColumnVector> expected;
    for (const Column& c : schema.GetColumns()) expected.emplace_back(c.storage_type);
    for (const std::string& row : rows) {
        crlf += row + "\r\n";
        lf += row + "\n";
        const size_t bar = row.find('|');
        ASSERT_FALSE(ParseField(row.substr(0, bar), expected[0]).has_value());
        ASSERT_FALSE(ParseField(row.substr(bar + 1), expected[1]).has_value());
    }
    ASSERT_EQ(crlf[kBlock - 1], '\r');
    ASSERT_EQ(crlf[kBlock], '\n');
    ASSERT_NE(lf[kBlock - 1], '\n');  // the \n file has a line across the block boundary
    ASSERT_NE(lf[kBlock], '\n');

    for (const std::string* text : {&crlf, &lf}) {
        const auto meta = BulkLoad(disk_.get(), "t", schema, WriteFile("blocks.tbl", *text), Threads(2));
        ASSERT_EQ(meta.row_count, rows.size());
        const auto got = ReadAll(meta);
        EXPECT_TRUE(SameValues(got[0], expected[0]));
        EXPECT_TRUE(SameValues(got[1], expected[1]));
    }
}

// A one-block file whose last line has no newline, with and without \r.
TEST_F(BulkLoaderTest, LastLineEndsAtBlockEdge) {
    for (const std::string tail : {"", "\r"}) {
        SCOPED_TRACE(tail.empty() ? "no \\r" : "with \\r");
        size_t lines = 0;
        const std::string last = "77|" + std::string(5000, 'e');
        const std::string text = FillerLines(kBlock - last.size() - tail.size(), lines) + last + tail;
        ASSERT_EQ(text.size(), kBlock);
        const auto meta = BulkLoad(disk_.get(), "t", TextSchema(), WriteFile("edge.tbl", text), Threads(2));
        ASSERT_EQ(meta.row_count, lines + 1);
        const auto got = ReadAll(meta);
        EXPECT_EQ(got[0].GetInt(lines), 77);
        EXPECT_EQ(got[1].GetString(lines), std::string(5000, 'e'));
        EXPECT_EQ(got[1].GetString(lines - 1).substr(0, 1), "s");
    }
}

// A line of exactly the limit whose \r ends a block still reaches the parser.
TEST_F(BulkLoaderTest, CarriageReturnAllowanceAtBlockEdge) {
    const ValueCapScope cap(kMaxValueBytes);
    const size_t limit = LineLimit(kMaxValueBytes, 2);
    size_t lines = 0;
    const std::string line = "1|" + std::string(limit - 2, 'x');
    const std::string at_limit = FillerLines(kBlock - line.size() - 1, lines) + line + "\r\n2|end\n";
    ASSERT_EQ(at_limit[kBlock - 1], '\r');
    ASSERT_EQ(at_limit[kBlock], '\n');
    const std::string msg = LoadError(TextSchema(), WriteFile("cr_edge.tbl", at_limit), Threads(2));
    const std::string want = "t: line " + std::to_string(lines + 1) + ", column b: value of " +
                             std::to_string(limit - 2) + " bytes is over the 1 GiB limit for one value";
    EXPECT_EQ(msg, want);

    // One byte longer, and the block's end already holds too much.
    lines = 0;
    const std::string longer = FillerLines(kBlock - line.size() - 2, lines) + line + "x\r\n2|end\n";
    ASSERT_EQ(longer[kBlock - 1], '\r');
    EXPECT_EQ(LoadError(TextSchema(), WriteFile("cr_over.tbl", longer), Threads(2)),
              "t: line " + std::to_string(lines + 1) + ": line is longer than " + std::to_string(limit) + " bytes");
}

TEST_F(BulkLoaderTest, LongLineRefused) {
    // Numbers only, so no value can be big: an in-line value per column.
    const size_t limit = LineLimit(0, 2);
    const std::string refused = " line is longer than " + std::to_string(limit) + " bytes";
    // A file with no newline at all is refused, not read whole into memory.
    EXPECT_EQ(LoadError(IntSchema(), WriteFile("one_line.tbl", std::string(size_t{3} << 20, '7')), Threads(2)),
              "t: line 1:" + refused);
    EXPECT_EQ(LoadError(IntSchema(), WriteFile("long.tbl", "1|2\n\n" + std::string(limit + 1, '9') + "\n5|6\n"),
                        Threads(2)),
              "t: line 3:" + refused);

    // An earlier bad line in the same batch still wins.
    const std::string earlier = LoadError(IntSchema(), WriteFile("earlier.tbl", "1|2\nx|4\n" + std::string(limit + 1, '9')),
                                          Threads(2));
    EXPECT_EQ(earlier.rfind("t: line 2, column a: ", 0), 0u) << earlier;

    // A line of exactly the limit, even with \r\n, reaches the parser.
    for (const std::string end : {"\n", "\r\n"}) {
        const std::string text = "1|2\n1|" + std::string(limit - 2, '9') + end;
        const std::string msg = LoadError(IntSchema(), WriteFile("at_limit.tbl", text), Threads(2));
        EXPECT_EQ(msg.rfind("t: line 2, column b: ", 0), 0u) << msg;
    }
}

TEST_F(BulkLoaderTest, NumbersOnlyKeepTheInLineLimit) {
    const size_t in_line = LineLimit(0, 2);
    const std::string over = "1|2\n" + std::string(in_line + 1, '9') + "\n";
    const std::string refused = "t: line 2: line is longer than " + std::to_string(in_line) + " bytes";
    // The value cap, default or lowered, never widens a numbers-only line.
    EXPECT_EQ(LoadError(IntSchema(), WriteFile("numbers.tbl", over), Threads(1)), refused);
    {
        const ValueCapScope cap(20000);
        EXPECT_EQ(LoadError(IntSchema(), WriteFile("numbers.tbl", over), Threads(1)), refused);
    }
    // A text column gets room for one value at the cap.
    const std::string text = "1|" + std::string(in_line + 1, 't') + "\n";
    EXPECT_EQ(LoadError(TextSchema(), WriteFile("text.tbl", text), Threads(1)), "");
}

TEST_F(BulkLoaderTest, HugeThreadCountIsCapped) {
    const auto meta = BulkLoad(disk_.get(), "t", IntSchema(), WriteFile("tiny.tbl", "1|2\n3|4\n"), Threads(100000));
    EXPECT_EQ(meta.row_count, 2u);
}

TEST_F(BulkLoaderTest, LineNumbersCountSkippedEmptyLines) {
    const std::string msg = LoadError(IntSchema(), WriteFile("gaps.tbl", "1|2\n\n\r\n3|q\n"), Threads(1));
    EXPECT_EQ(msg.rfind("t: line 4, column b: ", 0), 0u) << msg;
}

TEST_F(BulkLoaderTest, NullText) {
    const Schema schema({Column("a", ColumnType::Integer(), true), Column("d", ColumnType::Varchar(10), true),
                         Column("n", ColumnType::BigInt(), true)});
    const std::string text = "5|x|\\N\n\\N|\\N|\\N\n10|y|\\N\n\\N|x|\\N\n1|\\N|\\N\n";
    BulkLoadOptions options = Threads(2);
    options.null_text = "\\N";
    const auto meta = BulkLoad(disk_.get(), "t", schema, WriteFile("nulls.tbl", text), options);

    ASSERT_EQ(meta.row_count, 5u);
    ASSERT_EQ(meta.row_groups.size(), 1u);
    EXPECT_EQ(meta.row_groups[0].chunks[0].null_count, 2u);
    EXPECT_EQ(meta.row_groups[0].chunks[1].null_count, 2u);
    EXPECT_EQ(meta.row_groups[0].chunks[2].null_count, 5u);

    const auto got = ReadAll(meta);
    EXPECT_FALSE(got[0].IsNull(0));
    EXPECT_TRUE(got[0].IsNull(1));
    EXPECT_TRUE(got[0].IsNull(3));
    EXPECT_EQ(got[0].GetInt(2), 10);
    EXPECT_TRUE(got[1].IsNull(1));
    EXPECT_TRUE(got[1].IsNull(4));
    EXPECT_EQ(got[1].GetString(3), "x");

    // NULLs stay out of the range and the distinct counts.
    ASSERT_EQ(meta.stats.min_max.count("a"), 1u);
    EXPECT_EQ(meta.stats.min_max.at("a").min.integer_, 1);
    EXPECT_EQ(meta.stats.min_max.at("a").max.integer_, 10);
    EXPECT_EQ(meta.stats.min_max.count("n"), 0u);
    EXPECT_EQ(meta.stats.distinct_counts.at("a"), 3u);
    EXPECT_EQ(meta.stats.distinct_counts.at("d"), 2u);
    EXPECT_EQ(meta.stats.distinct_counts.at("n"), 0u);
}

TEST_F(BulkLoaderTest, NullInColumnThatIsNotNullable) {
    BulkLoadOptions options = Threads(1);
    options.null_text = "\\N";
    EXPECT_EQ(LoadError(IntSchema(), WriteFile("n.tbl", "1|2\n3|\\N\n"), options),
              "t: line 2, column b: NULL in a column that isn't nullable");
}

TEST_F(BulkLoaderTest, LoadsValuesBiggerThanAPage) {
    const Schema schema({Column("id", ColumnType::Integer()), Column("body", ColumnType::Text()),
                         Column("tag", ColumnType::Varchar(8))});
    // Sizes around the in-line limit and values over many pages, beside short ones.
    const std::vector<std::pair<size_t, size_t>> sized = {{10, 0}, {20, 7}, {30, kMaxValueBytes}, {40, kMaxValueBytes + 1},
                                                         {50, size_t{2} << 20}, {51, kMaxValueBytes + 1}, {52, 9000},
                                                         {299, size_t{2} << 20}};
    std::vector<std::string> bodies(300);
    for (size_t i = 0; i < bodies.size(); ++i) bodies[i] = "b" + std::to_string(i);
    for (const auto& [row, bytes] : sized) bodies[row] = Letters(bytes, static_cast<uint32_t>(row));
    std::string text;
    for (size_t i = 0; i < bodies.size(); ++i) text += std::to_string(i) + "|" + bodies[i] + "|t" + std::to_string(i % 3) + "\n";
    const std::string path = WriteFile("big.tbl", text);

    // Read back exactly, and the same table at any thread count.
    const ColumnarTableMeta one = BulkLoad(disk_.get(), "t", schema, path, Threads(1));
    const ColumnarTableMeta four = BulkLoad(disk_.get(), "t", schema, path, Threads(4));
    ASSERT_EQ(one.row_groups.size(), 1u);
    const ChunkMeta& body = one.row_groups[0].chunks[1];
    EXPECT_EQ(body.encoding, Encoding::PLAIN);
    EXPECT_GT(body.big_page_count, 2u * ((size_t{2} << 20) / PAGE_SIZE));
    for (size_t c = 0; c < schema.GetColumnCount(); ++c) {
        EXPECT_TRUE(SameChunk(one.row_groups[0].chunks[c], four.row_groups[0].chunks[c])) << "column " << c;
        EXPECT_EQ(one.row_groups[0].chunks[c].big_page_count, four.row_groups[0].chunks[c].big_page_count);
    }
    for (const ColumnarTableMeta* meta : {&one, &four}) {
        const auto got = ReadAll(*meta);
        ASSERT_EQ(got[1].Size(), bodies.size());
        for (size_t i = 0; i < bodies.size(); ++i) ASSERT_EQ(got[1].GetString(i), bodies[i]) << "row " << i;
        EXPECT_EQ(got[2].GetString(299), "t2");
    }

    // Committed by a database, closed, and read back after the reopen.
    const std::string file = (dir_ / "big.db").string();
    {
        Database db(file, 64);
        LoadOptions options;
        options.storage = StorageKind::COLUMNAR;
        options.threads = 2;
        db.LoadTable("t", schema, path, options);
    }
    Database db(file, 64);
    const ColumnarTable& table = *db.GetCatalog()->GetTable("t")->columnar;
    ColumnVector whole(ColumnType::Text());
    table.ReadChunk(0, 1, whole);
    std::vector<StringRef> refs(bodies.size());
    StringArena arena;
    table.ReadRows(0, 1, nullptr, static_cast<uint32_t>(bodies.size()), refs.data(), arena);
    for (size_t i = 0; i < bodies.size(); ++i) {
        ASSERT_EQ(whole.GetString(i), bodies[i]) << "row " << i;
        ASSERT_EQ(View(refs[i]), bodies[i]) << "row " << i;
    }
    for (const auto& [row, bytes] : sized) {
        ColumnVector one_value(ColumnType::Text());
        table.ReadValue(RowPosition{0, static_cast<uint16_t>(row)}, 1, one_value);
        EXPECT_EQ(one_value.GetString(0), bodies[row]) << "row " << row;
    }
}

TEST_F(BulkLoaderTest, RefusesAValueOverTheCap) {
    const Schema schema({Column("id", ColumnType::Integer()), Column("body", ColumnType::Text())});
    const std::string over = "1|short\n2|" + std::string(20001, 'z') + "\n3|end\n";
    {
        const ValueCapScope cap(20000);
        // A value of exactly the cap loads.
        const auto meta = BulkLoad(disk_.get(), "t", schema, WriteFile("at_cap.tbl", "1|" + std::string(20000, 'z') + "\n"), Threads(1));
        EXPECT_EQ(meta.row_count, 1u);
        // One byte more is refused, naming the table, line and column.
        EXPECT_EQ(LoadError(schema, WriteFile("over.tbl", over), Threads(2)),
                  "t: line 2, column body: value of 20001 bytes is over the 1 GiB limit for one value");
    }
    // The cap is back to 1 GiB once the test is done with it.
    EXPECT_EQ(LoadError(schema, WriteFile("over.tbl", over), Threads(2)), "");
}

TEST_F(BulkLoaderTest, StatsMatchRowStoreRules) {
    std::vector<ColumnVector> expected;
    const std::string path = MixedFile(expected);
    const ColumnarTableMeta meta = BulkLoad(disk_.get(), "t", MixedSchema(), path, Threads(4));
    const TableStats& stats = meta.stats;

    std::set<std::string> keys;
    for (const auto& kv : stats.min_max) keys.insert(kv.first);
    EXPECT_EQ(keys, (std::set<std::string>{"a", "b", "c", "e"}));

    // Each range is the smallest and largest executor value of the parsed text.
    for (size_t c = 0; c < expected.size(); ++c) {
        const std::string name = MixedSchema().GetColumn(static_cast<uint32_t>(c)).name;
        if (keys.count(name) == 0) continue;
        Value lo = ToExecutorValue(expected[c], 0), hi = lo;
        for (size_t i = 1; i < expected[c].Size(); ++i) {
            const Value v = ToExecutorValue(expected[c], i);
            if (v < lo) lo = v;
            if (hi < v) hi = v;
        }
        const MinMax& range = stats.min_max.at(name);
        EXPECT_EQ(range.min.type_, lo.type_) << name;
        EXPECT_TRUE(range.min == lo) << name;
        EXPECT_TRUE(range.max == hi) << name;
    }
    EXPECT_EQ(stats.min_max.at("e").min.type_, TypeId::INTEGER);
    EXPECT_EQ(stats.min_max.at("b").min.type_, TypeId::DECIMAL);
    EXPECT_EQ(stats.min_max.at("c").min.type_, TypeId::DATE);

    EXPECT_EQ(stats.distinct_counts.size(), 5u);
    const double a = static_cast<double>(stats.distinct_counts.at("a"));
    std::printf("distinct estimate for a: %.0f of %zu\n", a, kRows);
    EXPECT_LE(std::fabs(a - static_cast<double>(kRows)), 0.02 * static_cast<double>(kRows));
    EXPECT_EQ(stats.distinct_counts.at("e"), 2u);

    uint64_t pages = 0;
    for (const RowGroupMeta& g : meta.row_groups) {
        for (const ChunkMeta& chunk : g.chunks) pages += chunk.page_count;
    }
    EXPECT_EQ(stats.page_count, pages);
    EXPECT_GT(pages, 0u);
}

// Any NaN in a DOUBLE chunk keeps that chunk out of the table's range.
TEST_F(BulkLoaderTest, NaNChunksGiveNoRange) {
    const Schema schema({Column("a", ColumnType::Integer()), Column("x", ColumnType::Double()),
                         Column("y", ColumnType::Double())});
    constexpr size_t kLines = kRowGroupSize + 1000;
    std::string text;
    for (size_t n = 1; n <= kLines; ++n) {
        const bool second_group = n > kRowGroupSize;
        const std::string x = (n == 1 || n == kRowGroupSize + 1) ? "nan" : std::to_string(n) + ".5";
        const std::string y = n == 7 ? "nan" : std::to_string(second_group ? n : n * 10) + ".25";
        text += std::to_string(n) + "|" + x + "|" + y + "\n";
    }
    for (unsigned threads : {1u, 4u}) {
        const auto meta = BulkLoad(disk_.get(), "t", schema, WriteFile("nan.tbl", text), Threads(threads));
        ASSERT_EQ(meta.row_groups.size(), 2u);
        EXPECT_EQ(meta.stats.min_max.count("x"), 0u) << "every chunk of x holds a NaN";

        // y's range comes only from the second group, the one without NaN.
        ASSERT_EQ(meta.stats.min_max.count("y"), 1u);
        EXPECT_DOUBLE_EQ(meta.stats.min_max.at("y").min.decimal_, static_cast<double>(kRowGroupSize + 1) + 0.25);
        EXPECT_DOUBLE_EQ(meta.stats.min_max.at("y").max.decimal_, static_cast<double>(kLines) + 0.25);
        ASSERT_EQ(meta.stats.min_max.count("a"), 1u);
    }
}

// A failed page write names the table and the batch's lines, keeping its class.
TEST_F(BulkLoaderTest, WriteFailureNamesTable) {
    const std::string path = WriteFile("w.tbl", "1|2\n\n3|4\n5|6\n");
    for (unsigned threads : {1u, 4u}) {
        SCOPED_TRACE(std::to_string(threads) + " threads");
        FailingWrites io(0);
        disk_->SetWriteObserver(&io);
        try {
            (void)BulkLoad(disk_.get(), "t", IntSchema(), path, Threads(threads));
            ADD_FAILURE() << "expected a throw";
        } catch (const IOException& e) {
            EXPECT_EQ(std::string(e.what()), "t: writing lines 1-4: disk full");
        }

        FailingWrites db(1);
        disk_->SetWriteObserver(&db);
        try {
            (void)BulkLoad(disk_.get(), "t", IntSchema(), path, Threads(threads));
            ADD_FAILURE() << "expected a throw";
        } catch (const IOException& e) {
            ADD_FAILURE() << "became an IOException: " << e.what();
        } catch (const DatabaseException& e) {
            EXPECT_EQ(std::string(e.what()), "t: writing lines 1-4: bad page");
        }

        FailingWrites other(2);
        disk_->SetWriteObserver(&other);
        try {
            (void)BulkLoad(disk_.get(), "t", IntSchema(), path, Threads(threads));
            ADD_FAILURE() << "expected a throw";
        } catch (const DatabaseException& e) {
            ADD_FAILURE() << "became a DatabaseException: " << e.what();
        } catch (const std::runtime_error& e) {
            EXPECT_EQ(std::string(e.what()), "not ours");
        }
        disk_->SetWriteObserver(nullptr);
    }
}

TEST_F(BulkLoaderTest, EmptyFileGivesNoRowGroups) {
    for (const std::string text : {"", "\n\n\r\n"}) {
        const auto meta = BulkLoad(disk_.get(), "t", IntSchema(), WriteFile("empty.tbl", text), Threads(3));
        EXPECT_EQ(meta.row_groups.size(), 0u);
        EXPECT_EQ(meta.row_count, 0u);
        EXPECT_EQ(meta.stats.row_count, 0u);
        EXPECT_EQ(meta.stats.page_count, 0u);
        EXPECT_TRUE(meta.stats.min_max.empty());
        ASSERT_EQ(meta.stats.distinct_counts.size(), 2u);
        EXPECT_EQ(meta.stats.distinct_counts.at("a"), 0u);
        EXPECT_EQ(meta.stats.distinct_counts.at("b"), 0u);
        EXPECT_EQ(meta.columns.size(), 2u);
    }
}

TEST_F(BulkLoaderTest, RefusesBadSchemaAndMissingFile) {
    Schema mismatched({Column("a", ColumnType::Integer()), Column("b", ColumnType::Integer())});
    std::vector<Column> cols = mismatched.GetColumns();
    cols[1].type = TypeId::VARCHAR;
    // The type check comes first, so the missing file isn't reported.
    EXPECT_EQ(LoadError(Schema(cols), (dir_ / "nope.tbl").string(), Threads(1)),
              "t: column b: its storage type doesn't match its type");

    const std::string missing = (dir_ / "nope.tbl").string();
    EXPECT_EQ(LoadError(IntSchema(), missing, Threads(1)), "t: cannot open " + missing);
}

}  // namespace
}  // namespace shilmandb
