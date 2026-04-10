#include <gtest/gtest.h>

#include "engine/database.hpp"
#include "common/exception.hpp"

#include <filesystem>
#include <set>

namespace shilmandb {

class EndToEndTest : public ::testing::Test {
protected:
    std::string test_file_;

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() /
                      "shilmandb_e2e_test.db").string();
        std::filesystem::remove(test_file_);
    }

    void TearDown() override {
        std::filesystem::remove(test_file_);
    }

    void LoadSimpleTable(Database& db, const std::string& name,
                         const Schema& schema,
                         const std::vector<std::vector<Value>>& rows) {
        auto* table_info = db.GetCatalog()->CreateTable(name, schema);
        for (const auto& row : rows) {
            (void)table_info->table->InsertTuple(Tuple(row, schema));
        }
        db.GetCatalog()->UpdateTableStats(name);
    }
};

// ---------------------------------------------------------------------------
// Basic SELECT
// ---------------------------------------------------------------------------

TEST_F(EndToEndTest, SimpleSelectStar) {
    Database db(test_file_);
    Schema schema({Column("id", TypeId::INTEGER),
                   Column("name", TypeId::VARCHAR),
                   Column("score", TypeId::INTEGER)});
    std::vector<std::vector<Value>> rows = {
        {Value(1), Value(std::string("alice")), Value(10)},
        {Value(2), Value(std::string("bob")),   Value(20)},
        {Value(3), Value(std::string("carol")), Value(30)}
    };
    LoadSimpleTable(db, "t", schema, rows);

    auto result = db.ExecuteSQL("SELECT id, name, score FROM t");
    ASSERT_EQ(result.tuples.size(), 3u);
}

// ---------------------------------------------------------------------------
// WHERE filter
// ---------------------------------------------------------------------------

TEST_F(EndToEndTest, FilterQuery) {
    Database db(test_file_);
    Schema schema({Column("id", TypeId::INTEGER),
                   Column("score", TypeId::INTEGER)});
    std::vector<std::vector<Value>> rows = {
        {Value(1), Value(10)},
        {Value(2), Value(20)},
        {Value(3), Value(30)},
        {Value(4), Value(5)}
    };
    LoadSimpleTable(db, "t", schema, rows);

    auto result = db.ExecuteSQL("SELECT id, score FROM t WHERE score > 15");
    ASSERT_EQ(result.tuples.size(), 2u);
    for (const auto& tuple : result.tuples) {
        auto score = tuple.GetValue(result.schema, 1).integer_;
        EXPECT_GT(score, 15);
    }
}

// ---------------------------------------------------------------------------
// JOIN
// ---------------------------------------------------------------------------

TEST_F(EndToEndTest, JoinQuery) {
    Database db(test_file_);
    Schema s1({Column("id", TypeId::INTEGER),
               Column("name", TypeId::VARCHAR)});
    Schema s2({Column("tid", TypeId::INTEGER),
               Column("val", TypeId::INTEGER)});

    std::vector<std::vector<Value>> rows1 = {
        {Value(1), Value(std::string("alice"))},
        {Value(2), Value(std::string("bob"))}
    };
    std::vector<std::vector<Value>> rows2 = {
        {Value(1), Value(100)},
        {Value(2), Value(200)},
        {Value(1), Value(150)}
    };
    LoadSimpleTable(db, "t1", s1, rows1);
    LoadSimpleTable(db, "t2", s2, rows2);

    auto result = db.ExecuteSQL(
        "SELECT t1.id, name, val FROM t1 JOIN t2 ON t1.id = t2.tid");
    ASSERT_EQ(result.tuples.size(), 3u);

    // Verify join pairings: collect all (id, val) pairs
    std::set<std::pair<int32_t, int32_t>> pairs;
    for (const auto& tuple : result.tuples) {
        auto id = tuple.GetValue(result.schema, 0).integer_;
        auto val = tuple.GetValue(result.schema, 2).integer_;
        pairs.insert({id, val});
    }
    // alice(1) matches val 100 and 150, bob(2) matches val 200
    EXPECT_EQ(pairs.count({1, 100}), 1u);
    EXPECT_EQ(pairs.count({1, 150}), 1u);
    EXPECT_EQ(pairs.count({2, 200}), 1u);
}

// ---------------------------------------------------------------------------
// GROUP BY with aggregates
// ---------------------------------------------------------------------------

TEST_F(EndToEndTest, AggregateQuery) {
    Database db(test_file_);
    Schema schema({Column("id", TypeId::INTEGER),
                   Column("val", TypeId::INTEGER)});
    std::vector<std::vector<Value>> rows = {
        {Value(1), Value(10)},
        {Value(1), Value(20)},
        {Value(2), Value(30)}
    };
    LoadSimpleTable(db, "t", schema, rows);

    auto result = db.ExecuteSQL(
        "SELECT id, COUNT(*), SUM(val) FROM t GROUP BY id");
    ASSERT_EQ(result.tuples.size(), 2u);

    // std::map orders by key, so group 1 before group 2
    EXPECT_EQ(result.tuples[0].GetValue(result.schema, 0).integer_, 1);
    EXPECT_EQ(result.tuples[0].GetValue(result.schema, 1).integer_, 2);
    EXPECT_DOUBLE_EQ(result.tuples[0].GetValue(result.schema, 2).decimal_, 30.0);

    EXPECT_EQ(result.tuples[1].GetValue(result.schema, 0).integer_, 2);
    EXPECT_EQ(result.tuples[1].GetValue(result.schema, 1).integer_, 1);
    EXPECT_DOUBLE_EQ(result.tuples[1].GetValue(result.schema, 2).decimal_, 30.0);
}

// ---------------------------------------------------------------------------
// ORDER BY + LIMIT
// ---------------------------------------------------------------------------

TEST_F(EndToEndTest, OrderByAndLimit) {
    Database db(test_file_);
    Schema schema({Column("id", TypeId::INTEGER),
                   Column("score", TypeId::INTEGER)});
    std::vector<std::vector<Value>> rows;
    for (int i = 1; i <= 10; ++i) {
        rows.push_back({Value(i), Value(i * 10)});
    }
    LoadSimpleTable(db, "t", schema, rows);

    auto result = db.ExecuteSQL(
        "SELECT id, score FROM t ORDER BY score DESC LIMIT 3");
    ASSERT_EQ(result.tuples.size(), 3u);
    EXPECT_EQ(result.tuples[0].GetValue(result.schema, 1).integer_, 100);
    EXPECT_EQ(result.tuples[1].GetValue(result.schema, 1).integer_, 90);
    EXPECT_EQ(result.tuples[2].GetValue(result.schema, 1).integer_, 80);
}

// ---------------------------------------------------------------------------
// Error path
// ---------------------------------------------------------------------------

TEST_F(EndToEndTest, NonexistentTableThrows) {
    Database db(test_file_);
    EXPECT_THROW((void)db.ExecuteSQL("SELECT id FROM nonexistent"), DatabaseException);
}

}  // namespace shilmandb
