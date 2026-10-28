#include <gtest/gtest.h>
#include <unistd.h>

#include "catalog/catalog.hpp"
#include "catalog/schema.hpp"
#include "engine/database.hpp"
#include "executor/vectorized_parity_harness.hpp"
#include "types/tuple.hpp"
#include "types/value.hpp"

#include <filesystem>
#include <memory>
#include <string>

namespace shilmandb {

class VectorizedDatabaseParityTest : public ::testing::Test {
protected:
    std::string test_file_;
    std::unique_ptr<Database> db_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() /
                      ("shilmandb_vec_db_parity_test_" + std::to_string(::getpid()) + ".db")).string();
        std::filesystem::remove(test_file_);
        db_ = std::make_unique<Database>(test_file_);

        // Populate a small table: t(id, val) with id = 0..499, val = id * 2.
        Schema schema({Column("id", TypeId::INTEGER),
                       Column("val", TypeId::INTEGER)});
        auto* table_info = db_->GetCatalog()->CreateTable("t", schema);
        for (int i = 0; i < 500; ++i) {
            (void)table_info->table->InsertTuple(
                Tuple({Value(i), Value(i * 2)}, schema));
        }
        db_->GetCatalog()->UpdateTableStats("t");
    }

    void TearDown() override {
        db_.reset();
        std::filesystem::remove(test_file_);
    }

    void RunParity(const std::string& sql) {
        auto tuple_rs = db_->ExecuteSQL(sql, ExecutionMode::TUPLE);
        auto vec_rs   = db_->ExecuteSQL(sql, ExecutionMode::VECTORIZED);
        test::ExpectRowsEqual(tuple_rs.tuples, vec_rs.tuples, tuple_rs.schema);
    }
};

// Pure SeqScan — both modes iterate the heap in physical order.
TEST_F(VectorizedDatabaseParityTest, SeqScanOnly) {
    RunParity("SELECT id, val FROM t");
}

// Filter over SeqScan — selection vector on the vectorized side.
TEST_F(VectorizedDatabaseParityTest, FilterOverSeqScan) {
    RunParity("SELECT id, val FROM t WHERE val >= 200");
}

// Projection with arithmetic — exercises per-batch column materialization.
TEST_F(VectorizedDatabaseParityTest, ProjectionWithArithmetic) {
    RunParity("SELECT id, val * 2 FROM t");
}

// Aggregate + GROUP BY — ORDER BY ensures deterministic emission across modes
// (tuple mode uses unordered_map; vectorized mode uses std::map).
TEST_F(VectorizedDatabaseParityTest, AggregateGroupBy) {
    RunParity("SELECT id, COUNT(*), SUM(val) FROM t GROUP BY id ORDER BY id");
}

// Full pipeline — scan + filter + aggregate + sort.
TEST_F(VectorizedDatabaseParityTest, FullPipeline) {
    RunParity("SELECT id, SUM(val) FROM t WHERE val >= 100 "
              "GROUP BY id ORDER BY id");
}

}  // namespace shilmandb
