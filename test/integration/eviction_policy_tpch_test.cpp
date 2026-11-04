#include <gtest/gtest.h>

#include <unistd.h>

#include <filesystem>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "buffer/eviction_policy_factory.hpp"
#include "common/exception.hpp"
#include "engine/database.hpp"
#include "tpch_workload.hpp"

namespace shilmandb {
namespace {

// CTest runs from build/test/, so the data is usually two levels up.
std::string FindDataDir() {
    const std::vector<std::string> candidates = {
        "bench/tpch_data/sf0.01/",
        "../bench/tpch_data/sf0.01/",
        "../../bench/tpch_data/sf0.01/",
    };
    for (const auto& dir : candidates) {
        if (std::filesystem::exists(dir + "lineitem.tbl")) return dir;
    }
    return {};
}

std::string TempDb(const std::string& what) {
    return (std::filesystem::temp_directory_path() /
            ("shilmandb_policy_tpch_" + what + "_" + std::to_string(::getpid()) + ".db")).string();
}

// Loads SF=0.01 into a database with the given policy and runs the 8 TPC-H queries.
std::vector<QueryResult> RunQueries(EvictionPolicyKind kind, const std::string& data_dir) {
    constexpr size_t kPoolSize = 128;  // small, so the queries evict
    const std::string file = TempDb(std::string(EvictionPolicyKindName(kind)));
    std::filesystem::remove(file);

    std::vector<QueryResult> results;
    {
        Database db(file, kPoolSize, kind);
        tpch::LoadTables(db, data_dir);
        for (const auto& query : tpch::Queries()) {
            results.push_back(db.ExecuteSQL(query.sql));
        }
    }
    std::filesystem::remove(file);
    return results;
}

class EvictionPolicyTpchTest : public ::testing::TestWithParam<EvictionPolicyKind> {};

}  // namespace

// So test names and failures show the policy
static void PrintTo(EvictionPolicyKind kind, std::ostream* os) { *os << EvictionPolicyKindName(kind); }

TEST_P(EvictionPolicyTpchTest, EveryPolicyGivesLruAnswers) {
    const std::string data_dir = FindDataDir();
    if (data_dir.empty()) {
        GTEST_SKIP() << "TPC-H SF=0.01 data not found. Run bench/generate_tpch_data.sh first.";
    }

    const auto queries = tpch::Queries();
    const auto expected = RunQueries(EvictionPolicyKind::LRU, data_dir);
    const auto actual = RunQueries(GetParam(), data_dir);
    ASSERT_EQ(expected.size(), queries.size());
    ASSERT_EQ(actual.size(), queries.size());
    for (size_t i = 0; i < queries.size(); ++i) {
        EXPECT_GT(expected[i].tuples.size(), 0u) << queries[i].tag;
        EXPECT_TRUE(tpch::SameResult(actual[i], expected[i])) << queries[i].tag;
    }
}

INSTANTIATE_TEST_SUITE_P(Kinds, EvictionPolicyTpchTest,
                         ::testing::Values(EvictionPolicyKind::CLOCK, EvictionPolicyKind::LRU2,
                                           EvictionPolicyKind::ARC, EvictionPolicyKind::LIRS),
                         [](const ::testing::TestParamInfo<EvictionPolicyKind>& info) {
                             return std::string(EvictionPolicyKindName(info.param));
                         });

// The policy kind reaches the buffer pool: LIRS refuses a pool of one, a learned policy
// needs a model path.
TEST(EvictionPolicyDatabaseTest, KindReachesThePolicy) {
    const std::string file = TempDb("kind");
    std::filesystem::remove(file);
    EXPECT_THROW({ Database db(file, 1, EvictionPolicyKind::LIRS); }, DatabaseException);
    EXPECT_NO_THROW({ Database db(file, 1, EvictionPolicyKind::LRU); });
    EXPECT_THROW({ Database db(file, 16, EvictionPolicyKind::LEARNED_V2); }, DatabaseException);
    std::filesystem::remove(file);
}

// A policy that can't be built fails before the file is opened, so no file is left behind
TEST(EvictionPolicyDatabaseTest, BadPolicyCreatesNoFile) {
    const std::string file = TempDb("nofile");
    std::filesystem::remove(file);
    EXPECT_THROW({ Database db(file, 1, EvictionPolicyKind::LIRS); }, DatabaseException);
    EXPECT_FALSE(std::filesystem::exists(file));
    EXPECT_THROW({ Database db(file, 16, EvictionPolicyKind::LEARNED_V2); }, DatabaseException);
    EXPECT_FALSE(std::filesystem::exists(file));
    std::filesystem::remove(file);
}

#ifndef SHILMANDB_HAS_LIBTORCH
TEST(EvictionPolicyDatabaseTest, JoinModelNeedsMlBuild) {
    const std::string file = TempDb("join");
    std::filesystem::remove(file);
    EXPECT_THROW({ Database db(file, 16, EvictionPolicyKind::LRU, "", "join_model.pt"); }, DatabaseException);
    std::filesystem::remove(file);
}
#else
TEST(EvictionPolicyDatabaseTest, BadJoinModelPathThrows) {
    const std::string file = TempDb("join");
    std::filesystem::remove(file);
    EXPECT_THROW({ Database db(file, 16, EvictionPolicyKind::LRU, "", "no_such_join_model.pt"); }, DatabaseException);
    // the old constructor still refuses learned join without a path
    EXPECT_THROW({ Database db(file, 16, true, "", false, ""); }, DatabaseException);
    std::filesystem::remove(file);
}
#endif

}  // namespace shilmandb
