#include "storage/columnar/bulk_loader.hpp"
#include "common/exception.hpp"
#include "storage/columnar/column_vector.hpp"
#include "storage/columnar/columnar_table.hpp"
#include "storage/columnar/encoding.hpp"
#include "storage/columnar/field_parser.hpp"
#include "storage/columnar/hyperloglog.hpp"
#include "storage/disk_manager.hpp"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace shilmandb {
namespace {

constexpr size_t kReadBlockBytes = size_t{1} << 20;               // the reader takes the file 1 MiB at a time
constexpr uint64_t kNoFailure = std::numeric_limits<uint64_t>::max();
constexpr size_t kAbortCheckRows = 4096;                          // how often a worker looks for a failure

// One row group's lines, cut by the reader and parsed by one worker.
struct Batch {
    uint64_t number{0};
    std::string text;                    // the lines back to back, line ends removed
    std::vector<size_t> ends;            // line i is text[ends[i - 1], ends[i])
    std::vector<uint64_t> line_numbers;  // each line's 1-based place in the file
    std::exception_ptr tail_error;       // thrown once these lines parse cleanly
};

// What a worker hands back for one batch.
struct BatchResult {
    RowGroupMeta group;
    std::vector<HyperLogLog> sketches;  // one per column
};

// Splits a line on every delimiter, after dropping one trailing delimiter.
void SplitLine(std::string_view line, char delim, std::vector<std::string_view>& fields) {
    fields.clear();
    if (!line.empty() && line.back() == delim) line.remove_suffix(1);
    size_t start = 0;
    while (true) {
        const size_t at = line.find(delim, start);
        if (at == std::string_view::npos) {
            fields.push_back(line.substr(start));
            return;
        }
        fields.push_back(line.substr(start, at - start));
        start = at + 1;
    }
}

// Adds each non-NULL stored value's hash to the sketch.
void AddToSketch(const ColumnVector& col, HyperLogLog& sketch) {
    for (size_t i = 0; i < col.Size(); ++i) {
        if (col.IsNull(i)) continue;
        switch (col.Kind()) {
            case PhysicalKind::INT64:  sketch.Add(HashInt(col.GetInt(i))); break;
            case PhysicalKind::INT128: sketch.Add(HashWide(col.GetWide(i))); break;
            case PhysicalKind::DOUBLE: sketch.Add(HashDouble(col.GetDouble(i))); break;
            case PhysicalKind::STRING: sketch.Add(HashString(col.GetString(i))); break;
        }
    }
}

// Only these executor types get a range, as in the row store's stats.
bool KeepsRange(TypeId type) {
    return type == TypeId::INTEGER || type == TypeId::BIGINT || type == TypeId::DECIMAL || type == TypeId::DATE;
}

// Builds the table's meta from batch results, taken in batch order. Coordinator only.
class TableBuilder {
public:
    TableBuilder(const std::string& table, const Schema& schema) : schema_(schema), sketches_(schema.GetColumnCount()), lo_(schema.GetColumnCount()), hi_(schema.GetColumnCount()) {
        meta_.name = table;
        for (uint32_t i = 0; i < schema.GetColumnCount(); ++i) {
            const Column& c = schema.GetColumn(i);
            meta_.columns.push_back(ColumnMeta{c.name, i + 1, c.storage_type, c.nullable});
        }
    }

    // Places one row group and folds its stats in.
    void Add(BatchResult&& result) {
        meta_.row_count += result.group.row_count;
        for (size_t c = 0; c < sketches_.size(); ++c) {
            const ChunkMeta& chunk = result.group.chunks[c];
            page_count_ += chunk.page_count;
            sketches_[c].Merge(result.sketches[c]);
            // A DOUBLE chunk holding any NaN has no min or max, so no range.
            if (!KeepsRange(schema_.GetColumn(static_cast<uint32_t>(c)).type) || !chunk.min || !chunk.max) continue;
            if (!lo_[c] || *chunk.min < *lo_[c]) lo_[c] = chunk.min;
            if (!hi_[c] || *hi_[c] < *chunk.max) hi_[c] = chunk.max;
        }
        meta_.row_groups.push_back(std::move(result.group));
    }

    // Fills the table stats and hands the meta over.
    [[nodiscard]] ColumnarTableMeta Finish() {
        TableStats& stats = meta_.stats;
        stats.row_count = meta_.row_count;
        stats.page_count = page_count_;
        for (size_t c = 0; c < sketches_.size(); ++c) {
            const ColumnMeta& col = meta_.columns[c];
            stats.distinct_counts[col.name] = sketches_[c].Estimate();
            if (lo_[c] && hi_[c]) {
                stats.min_max[col.name] = MinMax{ScalarToValue(col.type, *lo_[c]), ScalarToValue(col.type, *hi_[c])};
            }
        }
        return std::move(meta_);
    }

private:
    const Schema& schema_;
    ColumnarTableMeta meta_;
    uint64_t page_count_{0};
    std::vector<HyperLogLog> sketches_;
    std::vector<std::optional<Scalar>> lo_, hi_;
};

// One reader cuts batches, workers parse and write them, the calling thread merges.
// Every field below the mutex is shared and guarded by it.
// Memory: at most 2 x workers batches in flight, plus the one being read.
// Each holds up to 65,536 lines of at most the line cap.
// There is no byte budget.
class Pipeline {
public:
    Pipeline(DiskManager* disk, const std::string& table, const std::string& path, const Schema& schema, const BulkLoadOptions& options, std::ifstream file, unsigned workers)
        : disk_(disk), table_(table), path_(path), schema_(schema), options_(options), file_(std::move(file)),
          workers_(workers), max_in_flight_(2 * static_cast<uint64_t>(workers)),
          max_line_bytes_(static_cast<uint64_t>(schema.GetColumnCount()) * (kMaxValueBytes + 1)) {}

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    // Stops and joins every thread, whatever way Run left.
    ~Pipeline() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        WakeAll();
        for (std::thread& t : threads_) {
            if (t.joinable()) t.join();
        }
    }

    // Starts the threads, merges results in batch order, rethrows the first failure.
    void Run(TableBuilder& builder) {
        threads_.reserve(workers_ + 1);
        threads_.emplace_back([this] { ReadLoop(); });
        for (unsigned i = 0; i < workers_; ++i) threads_.emplace_back([this] { WorkLoop(); });

        for (uint64_t next = 0;; ++next) {
            BatchResult result;
            {
                std::unique_lock<std::mutex> lock(mu_);
                coord_cv_.wait(lock, [&] {
                    return done_.count(next) > 0 || next >= fail_batch_ || (reader_done_ && next == batches_read_);
                });
                auto it = done_.find(next);
                if (it == done_.end()) break;
                result = std::move(it->second);
                done_.erase(it);
                --in_flight_;
            }
            reader_cv_.notify_one();
            builder.Add(std::move(result));
        }

        std::exception_ptr error;
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
            error = error_;
        }
        WakeAll();
        for (std::thread& t : threads_) t.join();
        threads_.clear();
        if (error) std::rethrow_exception(error);
    }

private:
    // Wakes every waiting thread so each rechecks its condition.
    void WakeAll() {
        reader_cv_.notify_all();
        worker_cv_.notify_all();
        coord_cv_.notify_all();
    }

    // Keeps the failure with the lowest batch number.
    void Fail(uint64_t batch, std::exception_ptr error) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (batch < fail_batch_) {
                fail_batch_ = batch;
                error_ = std::move(error);
            }
        }
        WakeAll();
    }

    // True once this batch can't matter: something earlier failed, or I'm stopping.
    [[nodiscard]] bool Abandoned(uint64_t batch) const { return stop_ || fail_batch_ < batch; }

    // Queues a full batch, waiting while too many are in flight. False means stop reading.
    bool Push(Batch&& batch) {
        {
            std::unique_lock<std::mutex> lock(mu_);
            reader_cv_.wait(lock, [&] { return in_flight_ < max_in_flight_ || stop_ || fail_batch_ != kNoFailure; });
            if (stop_ || fail_batch_ != kNoFailure) return false;
            work_.push_back(std::move(batch));
            ++in_flight_;
            ++batches_read_;
        }
        worker_cv_.notify_one();
        return true;
    }

    // Reads the file in blocks and cuts it into numbered batches of whole lines.
    void ReadLoop() {
        Batch batch;
        uint64_t line = 0;
        size_t row_start = 0;

        // Queues the whole lines read so far with an error thrown after they parse.
        // An earlier bad line in the batch still wins.
        auto push_with_error = [&](std::exception_ptr error) {
            batch.text.resize(row_start);
            batch.tail_error = std::move(error);
            (void)Push(std::move(batch));
        };

        // Queues the lines before a too-long line, carrying its error, then stops.
        auto refuse_long_line = [&](uint64_t at) {
            push_with_error(std::make_exception_ptr(DatabaseException(table_ + ": line " + std::to_string(at) + ": line is longer than " + std::to_string(max_line_bytes_) + " bytes")));
        };

        // Ends the current line; false when the pipeline wants the reader to stop.
        auto end_line = [&]() -> bool {
            ++line;
            if (batch.text.size() > row_start && batch.text.back() == '\r') batch.text.pop_back();
            if (batch.text.size() == row_start) return true;
            if (batch.text.size() - row_start > max_line_bytes_) {
                refuse_long_line(line);
                return false;
            }
            batch.ends.push_back(batch.text.size());
            batch.line_numbers.push_back(line);
            row_start = batch.text.size();
            if (batch.ends.size() < kRowGroupSize) return true;

            const uint64_t number = batch.number;
            if (!Push(std::move(batch))) return false;
            batch = Batch{};
            batch.number = number + 1;
            row_start = 0;
            return true;
        };

        try {
            std::vector<char> block(kReadBlockBytes);
            bool reading = true;
            while (reading) {
                file_.read(block.data(), static_cast<std::streamsize>(block.size()));
                if (file_.bad()) {
                    push_with_error(std::make_exception_ptr(IOException(table_ + ": cannot read " + path_)));
                    reading = false;
                    break;
                }
                const auto n = static_cast<size_t>(file_.gcount());
                if (n == 0) break;

                const char* p = block.data();
                const char* end = p + n;
                while (p < end) {
                    const auto* nl = static_cast<const char*>(std::memchr(p, '\n', static_cast<size_t>(end - p)));
                    if (nl == nullptr) {
                        batch.text.append(p, end);
                        // One more byte may be a \r that the line end strips.
                        if (batch.text.size() - row_start > max_line_bytes_ + 1) {
                            refuse_long_line(line + 1);
                            reading = false;
                        }
                        break;
                    }
                    batch.text.append(p, nl);
                    p = nl + 1;
                    if (!end_line()) {
                        reading = false;
                        break;
                    }
                }
            }
            // A last line with no newline still counts.
            if (reading && batch.text.size() > row_start) reading = end_line();
            if (reading && !batch.ends.empty()) (void)Push(std::move(batch));
        } catch (...) {
            Fail(batch.number, std::current_exception());
        }

        {
            std::lock_guard<std::mutex> lock(mu_);
            reader_done_ = true;
        }
        WakeAll();
    }

    // Takes batches until the reader is done or the pipeline stops.
    void WorkLoop() {
        std::vector<ColumnVector> cols;  // reused from batch to batch
        while (true) {
            Batch batch;
            {
                std::unique_lock<std::mutex> lock(mu_);
                worker_cv_.wait(lock, [&] { return stop_ || reader_done_ || !work_.empty(); });
                if (stop_ || work_.empty()) return;
                batch = std::move(work_.front());
                work_.pop_front();
            }
            if (Abandoned(batch.number)) continue;

            try {
                if (cols.empty()) {
                    for (const Column& c : schema_.GetColumns()) cols.emplace_back(c.storage_type);
                }
                std::optional<BatchResult> result = Process(batch, cols);
                if (!result) continue;
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    done_.emplace(batch.number, std::move(*result));
                }
                coord_cv_.notify_one();
            } catch (...) {
                Fail(batch.number, std::current_exception());
            }
        }
    }

    // The error for one bad field, naming the table, line and column.
    [[nodiscard]] DatabaseException LineError(uint64_t line, const std::string& column, const std::string& reason) const {
        return DatabaseException(table_ + ": line " + std::to_string(line) + ", column " + column + ": " + reason);
    }

    // Parses one field into its column; throws the line's error.
    void ParseCell(std::string_view field, uint32_t c, uint64_t line, ColumnVector& col) const {
        const Column& column = schema_.GetColumn(c);
        if (options_.null_text && field == *options_.null_text) {
            if (!column.nullable) throw LineError(line, column.name, "NULL in a column that isn't nullable");
            col.AppendNull();
            return;
        }
        if (auto reason = ParseField(field, col)) throw LineError(line, column.name, *reason);
        if (col.Kind() == PhysicalKind::STRING) {
            const size_t bytes = col.GetString(col.Size() - 1).size();
            if (bytes > kMaxValueBytes) {
                throw LineError(line, column.name, "value of " + std::to_string(bytes) + " bytes is bigger than a page");
            }
        }
    }

    // Parses, sketches and writes one batch; nothing when it was abandoned midway.
    [[nodiscard]] std::optional<BatchResult> Process(const Batch& batch, std::vector<ColumnVector>& cols) const {
        const size_t rows = batch.ends.size();
        const auto width = static_cast<uint32_t>(cols.size());
        for (ColumnVector& col : cols) {
            col.Clear();
            col.Reserve(rows);
        }

        std::vector<std::string_view> fields;
        fields.reserve(width + 1);
        size_t start = 0;
        for (size_t r = 0; r < rows; ++r) {
            if (r % kAbortCheckRows == 0 && Abandoned(batch.number)) return std::nullopt;
            const std::string_view line(batch.text.data() + start, batch.ends[r] - start);
            start = batch.ends[r];
            SplitLine(line, options_.delimiter, fields);
            if (fields.size() != width) {
                throw DatabaseException(table_ + ": line " + std::to_string(batch.line_numbers[r]) + ": expected " + std::to_string(width) + " fields, got " + std::to_string(fields.size()));
            }
            for (uint32_t c = 0; c < width; ++c) ParseCell(fields[c], c, batch.line_numbers[r], cols[c]);
        }
        if (batch.tail_error) std::rethrow_exception(batch.tail_error);

        BatchResult result;
        result.group.row_count = static_cast<uint32_t>(rows);
        result.sketches.resize(width);
        for (uint32_t c = 0; c < width; ++c) AddToSketch(cols[c], result.sketches[c]);
        if (Abandoned(batch.number)) return std::nullopt;

        result.group.chunks.reserve(width);
        try {
            for (const ColumnVector& col : cols) result.group.chunks.push_back(WriteChunk(disk_, col));
        } catch (const IOException& e) {
            throw IOException(WriteContext(batch) + e.what());
        } catch (const DatabaseException& e) {
            throw DatabaseException(WriteContext(batch) + e.what());
        }
        return result;
    }

    // "<table>: writing lines <first>-<last>: ", naming a batch whose write failed.
    [[nodiscard]] std::string WriteContext(const Batch& batch) const {
        return table_ + ": writing lines " + std::to_string(batch.line_numbers.front()) + "-" + std::to_string(batch.line_numbers.back()) + ": ";
    }

    DiskManager* disk_;
    const std::string& table_;
    const std::string& path_;
    const Schema& schema_;
    const BulkLoadOptions& options_;
    std::ifstream file_;  // the reader's alone once Run starts
    const unsigned workers_;
    const uint64_t max_in_flight_;
    // Longer lines are refused. Such a line has a field bigger than a page,
    // or absurdly padded number fields; the cap refuses those too.
    const uint64_t max_line_bytes_;
    std::vector<std::thread> threads_;  // the calling thread's alone

    std::mutex mu_;
    std::condition_variable reader_cv_, worker_cv_, coord_cv_;
    std::deque<Batch> work_;                  // read, waiting for a worker
    std::map<uint64_t, BatchResult> done_;    // finished, waiting to be merged in order
    uint64_t in_flight_{0};                   // read but not yet merged
    uint64_t batches_read_{0};
    bool reader_done_{false};
    std::atomic<bool> stop_{false};           // written under mu_; workers peek without it
    std::atomic<uint64_t> fail_batch_{kNoFailure};
    std::exception_ptr error_;
};

}  // namespace

// Checks the schema and opens the file before any thread starts.
ColumnarTableMeta BulkLoad(DiskManager* disk, const std::string& table_name, const Schema& schema, const std::string& path, const BulkLoadOptions& options) {
    for (const Column& c : schema.GetColumns()) {
        if (ExecutorType(c.storage_type) != c.type) {
            throw DatabaseException(table_name + ": column " + c.name + ": its storage type doesn't match its type");
        }
    }
    std::unordered_set<std::string> names;
    for (const Column& c : schema.GetColumns()) {
        if (!names.insert(c.name).second) throw DatabaseException(table_name + ": column " + c.name + " appears twice");
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) throw DatabaseException(table_name + ": cannot open " + path);

    unsigned workers = options.threads;
    if (workers == 0) workers = std::max(1u, std::thread::hardware_concurrency());
    workers = std::min(workers, kMaxLoadThreads);

    TableBuilder builder(table_name, schema);
    {
        Pipeline pipeline(disk, table_name, path, schema, options, std::move(file), workers);
        pipeline.Run(builder);
    }
    return builder.Finish();
}

}  // namespace shilmandb
