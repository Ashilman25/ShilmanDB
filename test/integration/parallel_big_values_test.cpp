#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "catalog/catalog.hpp"
#include "common/exception.hpp"
#include "engine/database.hpp"
#include "executor/parallel/batch.hpp"
#include "executor/parallel/pipeline.hpp"
#include "executor/parallel/scan_source.hpp"
#include "storage/columnar/columnar_table.hpp"
#include "storage/columnar/encoding.hpp"

namespace shilmandb::parallel {
namespace {

constexpr uint32_t kBigRows = 2 * kRowGroupSize + 1000;  // three row groups
constexpr uint32_t kGroup1 = kRowGroupSize;
constexpr uint32_t kGroup2 = 2 * kRowGroupSize;
constexpr size_t kThreeMiB = size_t{3} << 20;

// n letters from a seeded generator, so no short cycle repeats.
std::string Letters(size_t n, uint32_t seed) {
    std::string s(n, 'a');
    uint32_t x = seed * 2654435761u + 1;
    for (char& c : s) {
        x = x * 1664525u + 1013904223u;
        c = static_cast<char>('a' + (x >> 24) % 26);
    }
    return s;
}

// The same text with byte i changed to another letter.
std::string ByteChanged(std::string s, size_t i) {
    s[i] = s[i] == 'a' ? 'b' : 'a';
    return s;
}

// The values the tests place among short ones. Those past 65,535 bytes start
// with 'z', so they sort after every short value.
struct Special {
    std::string empty, seven, v8155, v8156, v8156x, v65535, v70000, v70000x, v3m, v3mx, v3ms, other3m;
};

// Built once per process.
const Special& Specials() {
    static const Special s = [] {
        Special v;
        v.seven = "seven77";
        v.v8155 = Letters(kMaxValueBytes, 1);
        v.v8156 = Letters(kMaxValueBytes + 1, 2);
        v.v8156x = ByteChanged(v.v8156, v.v8156.size() - 1);
        v.v65535 = Letters(65535, 6);
        v.v70000 = "z" + Letters(69999, 3);
        v.v70000x = ByteChanged(v.v70000, 1);
        v.v3m = "z" + Letters(2999999, 4);
        v.v3mx = ByteChanged(v.v3m, v.v3m.size() - 1);
        v.v3ms = v.v3m.substr(0, v.v3m.size() - 1);
        v.other3m = "z" + Letters(2999999, 5);
        return v;
    }();
    return s;
}

// The big table's text column. Group 0 has no big value: a dictionary.
// Groups 1 and 2 hold big values beside short ones, so they are plain.
std::vector<std::string> BigColumn() {
    const Special& v = Specials();
    std::vector<std::string> s(kBigRows);
    for (uint32_t r = 0; r < kBigRows; ++r) s[r] = "w" + std::to_string(r % 40);
    s[100] = v.empty;
    s[200] = v.seven;
    s[300] = v.v8155;
    s[kGroup1 + 64] = v.empty;
    s[kGroup1 + 164] = v.seven;
    s[kGroup1 + 264] = v.v8155;
    s[kGroup1 + 364] = v.v8156;
    s[kGroup1 + 464] = v.v8156x;
    s[kGroup1 + 564] = v.v70000;
    s[kGroup1 + 664] = v.v3m;
    s[kGroup1 + 764] = v.v3ms;
    s[kGroup2 + 28] = v.v3m;
    s[kGroup2 + 128] = v.v3mx;
    s[kGroup2 + 228] = v.v70000;
    s[kGroup2 + 328] = v.v8156;
    return s;
}

// The probe table's text column: exact copies, near misses, and one short word.
std::vector<std::string> ProbeColumn() {
    const Special& v = Specials();
    return {v.v3m, v.v3mx, v.v3ms, v.v70000, v.v70000x, v.v8156, v.v8156x, v.v8155, v.empty, v.seven, v.other3m, "w3"};
}

// An integer and a text column, named for the table.
Schema TextSchema(const std::string& id, const std::string& text) {
    return Schema({Column(id, ColumnType::Integer()), Column(text, ColumnType::Text())});
}

// The value an integer cell holds, INTEGER or BIGINT.
int64_t IntCell(const QueryResult& r, size_t row, uint32_t col) {
    const Value v = r.tuples[row].GetValue(r.schema, col);
    return v.type_ == TypeId::BIGINT ? v.bigint_ : v.integer_;
}

// The bytes a text cell holds.
std::string TextCell(const QueryResult& r, size_t row, uint32_t col) {
    return r.tuples[row].GetValue(r.schema, col).varchar_;
}

// Every row's integer cells, in result order.
std::vector<std::vector<int64_t>> IntRows(const QueryResult& r) {
    std::vector<std::vector<int64_t>> rows(r.tuples.size());
    for (size_t i = 0; i < r.tuples.size(); ++i) {
        for (uint32_t c = 0; c < r.schema.GetColumnCount(); ++c) rows[i].push_back(IntCell(r, i, c));
    }
    return rows;
}

// The same rows sorted, so results compare as multisets.
std::vector<std::vector<int64_t>> Sorted(std::vector<std::vector<int64_t>> rows) {
    std::sort(rows.begin(), rows.end());
    return rows;
}

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

// Sets the scan's batch byte cap for one test and puts it back after.
class BatchBytesScope {
public:
    explicit BatchBytesScope(size_t bytes) { BatchStringBytesForTest(bytes); }
    ~BatchBytesScope() { BatchStringBytesForTest(0); }
    BatchBytesScope(const BatchBytesScope&) = delete;
    BatchBytesScope& operator=(const BatchBytesScope&) = delete;
};

// One batch a scan handed up: its ids, and each row's string bytes.
struct ScanBatch {
    std::vector<int64_t> ids;
    std::vector<size_t> bytes;       // summed over the row's string slots
    std::vector<std::string> heads;  // slot 1's first and last byte
};

class ParallelBigValueTest : public ::testing::Test {
protected:
    std::string db_file_;
    std::vector<std::string> files_;
    std::unique_ptr<Database> db_;

    void SetUp() override {
        db_file_ = TempPath(".db");
        std::filesystem::remove(db_file_);
        db_ = std::make_unique<Database>(db_file_, 64);
    }

    void TearDown() override {
        db_.reset();
        std::filesystem::remove(db_file_);
        for (const std::string& f : files_) std::filesystem::remove(f);
    }

    // A temp path unique to this process.
    static std::string TempPath(const std::string& suffix) {
        return (std::filesystem::temp_directory_path() /
                ("shilmandb_parallel_big_values_" + std::to_string(::getpid()) + suffix)).string();
    }

    // Writes rows "i|text[i]|" and loads them as a columnar table.
    void Load(const std::string& table, const Schema& schema, const std::vector<std::string>& text) {
        std::vector<std::string> lines(text.size());
        for (size_t i = 0; i < text.size(); ++i) lines[i] = std::to_string(i) + "|" + text[i] + "|";
        LoadLines(table, schema, lines);
    }

    // Writes the lines and loads them as a columnar table.
    void LoadLines(const std::string& table, const Schema& schema, const std::vector<std::string>& lines) {
        const std::string path = TempPath("_" + table + ".tbl");
        files_.push_back(path);
        {
            std::ofstream out(path, std::ios::binary);
            for (const std::string& line : lines) out << line << '\n';
        }
        LoadOptions options;
        options.storage = StorageKind::COLUMNAR;
        options.threads = 2;
        db_->LoadTable(table, schema, path, options);
    }

    // The loaded table's columnar reader.
    const ColumnarTable& Table(const std::string& name) const {
        return *db_->GetCatalog()->GetTable(name)->columnar;
    }

    // Runs sql in one mode and thread count.
    QueryResult Run(const std::string& sql, ExecutionMode mode, unsigned threads = 1) {
        QueryOptions options;
        options.mode = mode;
        options.threads = threads;
        return db_->ExecuteSQL(sql, options);
    }

    // Scans every row of table, handing up these columns (id first), and keeps each batch.
    std::vector<ScanBatch> DrainScan(const std::string& table, std::vector<uint32_t> columns = {0, 1}) {
        ScanSpec spec;
        spec.table = table;
        spec.columns = std::move(columns);
        ScanSource source(std::move(spec));
        QueryContext ctx{db_->GetBufferPoolManager(), db_->GetCatalog()};
        source.Prepare(ctx);
        auto local = source.MakeLocal();
        Batch batch;
        std::vector<ScanBatch> out;
        for (size_t m = 0; m < source.MorselCount(); ++m) {
            source.StartMorsel(m, *local);
            while (source.NextBatch(*local, batch)) {
                ScanBatch b;
                for (uint32_t r = 0; r < batch.rows; ++r) {
                    size_t bytes = 0;
                    for (const Vector& slot : batch.slots) {
                        if (slot.Type().kind == Kind::STRING) bytes += slot.StrAt(r).size;
                    }
                    const std::string_view s = View(batch.slots[1].StrAt(r));
                    b.ids.push_back(batch.slots[0].I64()[r]);
                    b.bytes.push_back(bytes);
                    b.heads.push_back(s.empty() ? "" : std::string{s.front(), s.back()});
                }
                out.push_back(std::move(b));
            }
        }
        return out;
    }
};

// Batch sizes the cap rule gives: a batch ends once its bytes pass cap,
// keeping the row that passed it, or at kBatchSize rows.
std::vector<size_t> ExpectedBatchSizes(const std::vector<size_t>& row_bytes, size_t cap) {
    std::vector<size_t> sizes;
    size_t rows = 0;
    size_t bytes = 0;
    for (size_t b : row_bytes) {
        ++rows;
        bytes += b;
        if (bytes > cap || rows == kBatchSize) {
            sizes.push_back(rows);
            rows = 0;
            bytes = 0;
        }
    }
    if (rows > 0) sizes.push_back(rows);
    return sizes;
}

TEST_F(ParallelBigValueTest, ScanJoinGroupSortKeepBytes) {
    const Special& v = Specials();
    const std::vector<std::string> s = BigColumn();
    const std::vector<std::string> ps = ProbeColumn();
    Load("big", TextSchema("id", "s"), s);
    Load("probe", TextSchema("pid", "ps"), ps);

    // Group 0's text is a dictionary; groups 1 and 2 are plain with big-value pages.
    const ColumnarTable& big = Table("big");
    ASSERT_EQ(big.RowGroupCount(), 3u);
    EXPECT_EQ(big.ChunkAt(0, 1).encoding, Encoding::DICTIONARY);
    for (size_t g = 1; g < 3; ++g) {
        EXPECT_EQ(big.ChunkAt(g, 1).encoding, Encoding::PLAIN) << "group " << g;
        EXPECT_GT(big.ChunkAt(g, 1).big_page_count, 0u) << "group " << g;
    }

    // Expected answers, worked out from the text itself.
    const uint32_t n = kBigRows;
    std::vector<std::vector<int64_t>> long_ids, v3m_ids, v70000_ids, sparse_ids, pairs, groups, small_order, top;
    const std::string head = v.v3m.substr(0, 24);
    const std::string tail = v.v3m.substr(v.v3m.size() - 24);
    const std::string head70 = v.v70000.substr(0, 24);
    const std::string tail70 = v.v70000.substr(v.v70000.size() - 24);
    auto starts = [](const std::string& x, const std::string& p) { return x.compare(0, p.size(), p) == 0; };
    auto ends = [](const std::string& x, const std::string& p) { return x.size() >= p.size() && x.compare(x.size() - p.size(), p.size(), p) == 0; };
    std::map<std::string, std::vector<int64_t>> by_text;
    for (uint32_t id = 0; id < n; ++id) {
        const std::string& x = s[id];
        if (x.size() > kMaxValueBytes) long_ids.push_back({id});
        if (x.size() >= 48 && starts(x, head) && ends(x, tail)) v3m_ids.push_back({id});
        if (x.size() >= 48 && starts(x, head70) && ends(x, tail70)) v70000_ids.push_back({id});
        if (id >= 66150 && id < 66350 && starts(x, head)) sparse_ids.push_back({id});
        by_text[x].push_back(id);
        for (size_t pid = 0; pid < ps.size(); ++pid) {
            if (ps[pid] == x) pairs.push_back({id, static_cast<int64_t>(pid)});
        }
    }
    for (const auto& [text, ids] : by_text) groups.push_back({static_cast<int64_t>(ids.size()), ids.front(), ids.back()});
    std::vector<uint32_t> order(n);
    for (uint32_t id = 0; id < n; ++id) order[id] = id;
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return s[a] != s[b] ? s[a] < s[b] : a < b; });
    for (uint32_t id : order) {
        if (s[id].size() <= 65535) small_order.push_back({id});
    }
    for (size_t k = 0; k < 25; ++k) top.push_back({order[k]});
    ASSERT_EQ(long_ids.size(), 9u);
    ASSERT_EQ(v3m_ids.size(), 2u);
    ASSERT_EQ(v70000_ids.size(), 2u);
    ASSERT_EQ(sparse_ids.size(), 2u);

    // Every row that holds more than 65,535 bytes, so the sort can leave them out.
    std::string not_long;
    for (const auto& row : long_ids) {
        if (s[row[0]].size() > 65535) not_long += (not_long.empty() ? "" : " AND ") + std::string("id <> ") + std::to_string(row[0]);
    }

    const std::string like_long = "SELECT id FROM big WHERE s LIKE '" + std::string(kMaxValueBytes + 1, '_') + "%'";
    const std::string like_v3m = "SELECT id FROM big WHERE s LIKE '" + head + "%" + tail + "'";
    const std::string like_v70000 = "SELECT id FROM big WHERE s LIKE '" + head70 + "%" + tail70 + "'";
    const std::string sparse = "SELECT id FROM big WHERE id >= 66150 AND id < 66350 AND s LIKE '" + head + "%'";
    const std::string join = "SELECT id, pid FROM big JOIN probe ON s = ps";
    const std::string group = "SELECT COUNT(*), MIN(id), MAX(id) FROM big GROUP BY s";
    const std::string sort_small = "SELECT id, s FROM big WHERE " + not_long + " ORDER BY s, id";
    const std::string sort_top = "SELECT id, s FROM big ORDER BY s, id LIMIT 25";

    // At 1 and 4 threads, then 4 again under a small cap that cuts batches.
    for (const size_t cap : {size_t{0}, size_t{100000}}) {
        const BatchBytesScope scope(cap);
        for (const unsigned threads : {1u, 4u}) {
            if (cap != 0 && threads == 1) continue;
            SCOPED_TRACE("threads " + std::to_string(threads) + ", cap " + std::to_string(cap));
            EXPECT_EQ(Sorted(IntRows(Run(like_long, ExecutionMode::PARALLEL, threads))), long_ids);
            EXPECT_EQ(Sorted(IntRows(Run(like_v3m, ExecutionMode::PARALLEL, threads))), v3m_ids);
            EXPECT_EQ(Sorted(IntRows(Run(like_v70000, ExecutionMode::PARALLEL, threads))), v70000_ids);
            EXPECT_EQ(Sorted(IntRows(Run(sparse, ExecutionMode::PARALLEL, threads))), sparse_ids);
            EXPECT_EQ(Sorted(IntRows(Run(join, ExecutionMode::PARALLEL, threads))), Sorted(pairs));
            EXPECT_EQ(Sorted(IntRows(Run(group, ExecutionMode::PARALLEL, threads))), Sorted(groups));

            // Sorted rows come back in order with their exact bytes.
            const QueryResult sorted = Run(sort_small, ExecutionMode::PARALLEL, threads);
            ASSERT_EQ(sorted.tuples.size(), small_order.size());
            for (size_t i = 0; i < sorted.tuples.size(); ++i) {
                ASSERT_EQ(IntCell(sorted, i, 0), small_order[i][0]) << "row " << i;
                ASSERT_EQ(TextCell(sorted, i, 1), s[small_order[i][0]]) << "row " << i;
            }
            // The 3 MB values compete for the top rows but sort after them.
            const QueryResult first = Run(sort_top, ExecutionMode::PARALLEL, threads);
            ASSERT_EQ(first.tuples.size(), top.size());
            for (size_t i = 0; i < top.size(); ++i) {
                EXPECT_EQ(IntCell(first, i, 0), top[i][0]) << "row " << i;
                EXPECT_EQ(TextCell(first, i, 1), s[top[i][0]]) << "row " << i;
            }
        }
    }

    // Read again after closing and reopening the file.
    db_.reset();
    db_ = std::make_unique<Database>(db_file_, 64);
    EXPECT_EQ(Sorted(IntRows(Run(join, ExecutionMode::PARALLEL, 4))), Sorted(pairs));
    EXPECT_EQ(Sorted(IntRows(Run(group, ExecutionMode::PARALLEL, 4))), Sorted(groups));
    ColumnVector whole(ColumnType::Text());
    Table("big").ReadChunk(1, 1, whole);
    for (uint32_t r = 0; r < kRowGroupSize; ++r) ASSERT_EQ(whole.GetString(r), s[kGroup1 + r]) << "row " << kGroup1 + r;
}

TEST_F(ParallelBigValueTest, OtherModesStopAt65535) {
    const Special& v = Specials();
    std::vector<std::string> mid(300);
    for (size_t i = 0; i < mid.size(); ++i) mid[i] = "m" + std::to_string(i);
    mid[10] = v.empty;
    mid[20] = v.seven;
    mid[30] = v.v8155;
    mid[40] = v.v8156;
    mid[50] = v.v65535;
    mid[51] = v.v8156x;
    Load("mid", TextSchema("id", "s"), mid);
    std::vector<std::string> over(10, "short");
    over[3] = v.v70000;
    Load("over", TextSchema("oid", "os"), over);
    ASSERT_GT(Table("mid").ChunkAt(0, 1).big_page_count, 0u);

    std::vector<size_t> order(mid.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return mid[a] != mid[b] ? mid[a] < mid[b] : a < b; });

    for (const ExecutionMode mode : {ExecutionMode::TUPLE, ExecutionMode::VECTORIZED, ExecutionMode::PARALLEL}) {
        SCOPED_TRACE("mode " + std::to_string(static_cast<int>(mode)));
        // Values up to 65,535 bytes come back exactly, in any order and sorted.
        const QueryResult all = Run("SELECT id, s FROM mid", mode);
        ASSERT_EQ(all.tuples.size(), mid.size());
        for (size_t i = 0; i < all.tuples.size(); ++i) {
            const auto id = static_cast<size_t>(IntCell(all, i, 0));
            ASSERT_LT(id, mid.size());
            EXPECT_EQ(TextCell(all, i, 1), mid[id]) << "id " << id;
        }
        const QueryResult sorted = Run("SELECT id, s FROM mid ORDER BY s, id", mode);
        ASSERT_EQ(sorted.tuples.size(), mid.size());
        for (size_t i = 0; i < sorted.tuples.size(); ++i) {
            EXPECT_EQ(IntCell(sorted, i, 0), static_cast<int64_t>(order[i])) << "row " << i;
            EXPECT_EQ(TextCell(sorted, i, 1), mid[order[i]]) << "row " << i;
        }

        // A result row can't hold more than 65,535 bytes of text in any mode.
        EXPECT_EQ(ErrorOf([&] { (void)Run("SELECT oid, os FROM over WHERE oid = 3", mode); }),
                  "VARCHAR exceeds maximum length of 65535 bytes");
        EXPECT_EQ(Run("SELECT oid FROM over", mode).tuples.size(), over.size());
    }
}

TEST_F(ParallelBigValueTest, BatchEndsAtTheByteCap) {
    // 40 values of 3 MiB in one row group, with short rows between them.
    std::vector<std::string> huge(100);
    std::vector<size_t> row_bytes(huge.size());
    for (size_t i = 0; i < huge.size(); ++i) {
        huge[i] = i % 2 == 0 && i < 80 ? std::string(kThreeMiB, static_cast<char>('a' + i % 26)) : "short" + std::to_string(i);
        huge[i].back() = 'Z';
        row_bytes[i] = huge[i].size();
    }
    Load("huge", TextSchema("id", "s"), huge);
    ASSERT_EQ(Table("huge").RowGroupCount(), 1u);

    // The real cap first; then a cap below one value, so each big value
    // ends its batch, and a batch starting with one holds just that row.
    for (const size_t cap : {kMaxBatchStringBytes, size_t{1} << 20}) {
        SCOPED_TRACE("cap " + std::to_string(cap));
        const BatchBytesScope scope(cap == kMaxBatchStringBytes ? 0 : cap);
        const std::vector<ScanBatch> batches = DrainScan("huge");
        std::vector<size_t> sizes;
        int64_t next = 0;
        for (const ScanBatch& b : batches) {
            ASSERT_FALSE(b.ids.empty());
            sizes.push_back(b.ids.size());
            size_t bytes = 0;
            for (size_t r = 0; r < b.ids.size(); ++r) {
                // Every row once, in order, with its own bytes.
                ASSERT_EQ(b.ids[r], next) << "row " << r;
                const auto i = static_cast<size_t>(next++);
                EXPECT_EQ(b.bytes[r], huge[i].size()) << "id " << i;
                EXPECT_EQ(b.heads[r], (std::string{huge[i].front(), huge[i].back()})) << "id " << i;
                bytes += b.bytes[r];
            }
            // No batch holds more than the cap plus its last value.
            EXPECT_LE(bytes - b.bytes.back(), cap);
        }
        EXPECT_EQ(next, static_cast<int64_t>(huge.size()));
        EXPECT_EQ(sizes, ExpectedBatchSizes(row_bytes, cap));
        ASSERT_GT(sizes.size(), 1u);
        if (cap < kThreeMiB) {
            // A value over the cap still makes a batch: just that row.
            EXPECT_EQ(sizes[0], 1u);
            EXPECT_EQ(batches[0].bytes[0], kThreeMiB);
            EXPECT_EQ(sizes[1], 2u);
        }
    }
}

TEST_F(ParallelBigValueTest, ConstantStringsCountEveryRow) {
    // A CONSTANT column of long strings beside a plain column of short ones.
    // Every row of the constant points at one copy, but each counts.
    const std::string same = Letters(5000, 7);
    std::vector<std::string> lines(3000);
    std::vector<size_t> row_bytes(lines.size());
    for (size_t i = 0; i < lines.size(); ++i) {
        const std::string mine = "row-" + std::to_string(i);
        lines[i] = std::to_string(i) + "|" + same + "|" + mine + "|";
        row_bytes[i] = same.size() + mine.size();
    }
    LoadLines("con", Schema({Column("id", ColumnType::Integer()), Column("k", ColumnType::Text()), Column("note", ColumnType::Text())}), lines);
    ASSERT_EQ(Table("con").ChunkAt(0, 1).encoding, Encoding::CONSTANT);
    ASSERT_EQ(Table("con").ChunkAt(0, 2).encoding, Encoding::PLAIN);

    const size_t cap = 100000;
    const BatchBytesScope scope(cap);
    const std::vector<ScanBatch> batches = DrainScan("con", {0, 1, 2});
    std::vector<size_t> sizes;
    int64_t next = 0;
    for (const ScanBatch& b : batches) {
        ASSERT_FALSE(b.ids.empty());
        sizes.push_back(b.ids.size());
        size_t bytes = 0;
        for (size_t r = 0; r < b.ids.size(); ++r) {
            ASSERT_EQ(b.ids[r], next++) << "row " << r;
            bytes += b.bytes[r];
        }
        EXPECT_LE(bytes - b.bytes.back(), cap);
    }
    EXPECT_EQ(next, static_cast<int64_t>(lines.size()));
    EXPECT_EQ(sizes, ExpectedBatchSizes(row_bytes, cap));
}

}  // namespace
}  // namespace shilmandb::parallel
