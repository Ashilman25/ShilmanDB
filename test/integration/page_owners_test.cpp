#include <gtest/gtest.h>

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "engine/database.hpp"
#include "page_owners.hpp"

namespace shilmandb {

class PageOwnersTest : public ::testing::Test {
protected:
    std::string db_file_;
    std::string csv_file_;

    void SetUp() override {
        const auto tmp = std::filesystem::temp_directory_path();
        db_file_ = (tmp / ("shilmandb_owners_test_" + std::to_string(::getpid()) + ".db")).string();
        csv_file_ = (tmp / ("shilmandb_owners_test_" + std::to_string(::getpid()) + ".csv")).string();
        std::filesystem::remove(db_file_);
        std::filesystem::remove(csv_file_);
    }

    void TearDown() override {
        std::filesystem::remove(db_file_);
        std::filesystem::remove(csv_file_);
    }

    // Builds table t with 3,000 rows, plus an index on k when asked.
    static void Build(Database& db, bool with_index) {
        const Schema schema({Column("k", TypeId::INTEGER), Column("v", TypeId::VARCHAR)});
        auto* info = db.GetCatalog()->CreateTable("t", schema);
        ASSERT_NE(info, nullptr);
        for (int32_t i = 0; i < 3000; ++i) {
            Tuple tuple({Value(i), Value(std::string(40, 'x'))}, schema);
            ASSERT_NE(info->table->InsertTuple(tuple).page_id, INVALID_PAGE_ID);
        }
        if (with_index) ASSERT_NE(db.GetCatalog()->CreateIndex("idx_t_k", "t", "k"), nullptr);
    }
};

TEST_F(PageOwnersTest, CollectPageOwnersCoversEveryPage) {
    Database db(db_file_);
    Build(db, true);
    const auto owners = tpch::CollectPageOwners(db, {"t"}, {"idx_t_k"});
    EXPECT_EQ(owners.size(), db.GetDiskManager()->GetNumPages());

    size_t tables = 0, inner = 0, leaves = 0;
    for (const auto& o : owners) {
        ASSERT_NE(o.kind, tpch::OwnedKind::OTHER);
        if (o.kind == tpch::OwnedKind::TABLE) {
            ++tables;
            EXPECT_EQ(o.object, 0);
            EXPECT_EQ(o.owner, "t");
        } else {
            (o.kind == tpch::OwnedKind::INNER ? inner : leaves)++;
            EXPECT_EQ(o.object, 1);
            EXPECT_EQ(o.owner, "idx_t_k");
        }
    }
    EXPECT_GE(tables, 2u);
    EXPECT_GE(inner, 1u);
    EXPECT_GE(leaves, 2u);
}

TEST_F(PageOwnersTest, WritePageOwnersWritesHeaderAndOneLinePerPage) {
    Database db(db_file_);
    Build(db, true);
    const auto owners = tpch::CollectPageOwners(db, {"t"}, {"idx_t_k"});
    tpch::WritePageOwners(owners, csv_file_);

    std::ifstream in(csv_file_);
    std::vector<std::string> lines;
    for (std::string line; std::getline(in, line);) lines.push_back(line);
    ASSERT_EQ(lines.size(), owners.size() + 1);
    EXPECT_EQ(lines[0], "page_id,kind,object,owner");
    for (size_t i = 0; i < owners.size(); ++i) {
        EXPECT_EQ(lines[i + 1].rfind(std::to_string(i) + ",", 0), 0u);
    }
}

TEST_F(PageOwnersTest, TablesOnlyWithoutIndexes) {
    Database db(db_file_);
    Build(db, false);
    const auto owners = tpch::CollectPageOwners(db, {"t"}, {});
    EXPECT_EQ(owners.size(), db.GetDiskManager()->GetNumPages());
    for (const auto& o : owners) EXPECT_EQ(o.kind, tpch::OwnedKind::TABLE);
}

}  // namespace shilmandb
