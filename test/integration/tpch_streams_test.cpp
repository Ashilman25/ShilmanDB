#include <gtest/gtest.h>
#include <unistd.h>

#include "common/exception.hpp"
#include "engine/database.hpp"
#include "tpch/tpch_streams.hpp"
#include "types/value.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace shilmandb::tpch {

namespace {

// The text a placeholder was given.
std::string Get(const Params& params, const std::string& name) {
    for (const auto& [key, value] : params) {
        if (key == name) return value;
    }
    ADD_FAILURE() << "missing placeholder " << name;
    return {};
}

int32_t Day(const std::string& date) { return Value::FromString(TypeId::DATE, date).date_; }

int YearOf(const std::string& date) { return std::stoi(date.substr(0, 4)); }

// A first-of-month date moved by a number of months ("1995-11-01" + 3 = "1996-02-01").
std::string AddMonths(const std::string& date, int months) {
    const int total = YearOf(date) * 12 + std::stoi(date.substr(5, 2)) - 1 + months;
    char buf[16];
    std::snprintf(buf, sizeof buf, "%04d-%02d-01", total / 12, total % 12 + 1);
    return buf;
}

std::vector<std::string> Tags(const std::vector<StreamItem>& items, size_t from, size_t to) {
    std::vector<std::string> tags;
    for (size_t i = from; i < to; ++i) tags.push_back(items[i].tag);
    return tags;
}

// The two quoted dates of a lookup's SQL, as day numbers.
std::pair<int32_t, int32_t> Window(const std::string& sql) {
    static const std::regex date("'(\\d{4}-\\d{2}-\\d{2})'");
    std::vector<int32_t> days;
    for (auto it = std::sregex_iterator(sql.begin(), sql.end(), date); it != std::sregex_iterator(); ++it) {
        days.push_back(Day((*it)[1].str()));
    }
    EXPECT_EQ(days.size(), 2u) << sql;
    return days.size() == 2 ? std::make_pair(days[0], days[1]) : std::make_pair(0, 0);
}

}  // namespace

TEST(TpchStreamsTest, TemplatesWithDefaultsMatchFixedQueries) {
    const auto templates = QueryTemplates();
    const auto queries = Queries();
    ASSERT_EQ(templates.size(), 8u);
    ASSERT_EQ(templates.size(), queries.size());

    for (size_t i = 0; i < templates.size(); ++i) {
        EXPECT_EQ(templates[i].tag, queries[i].tag);
        EXPECT_EQ(Render(templates[i].sql, DefaultParams(templates[i].tag)), queries[i].sql)
            << templates[i].tag;
    }
}

TEST(TpchStreamsTest, RenderRejectsLeftovers) {
    EXPECT_THROW((void)Render("a {x} b", {}), DatabaseException);
    EXPECT_THROW((void)Render("a b", {{"x", "1"}}), DatabaseException);
    EXPECT_EQ(Render("{x}{x}", {{"x", "1"}}), "11");
}

TEST(TpchStreamsTest, RngIsDeterministicAndInRange) {
    Rng a(42);
    Rng b(42);
    for (int i = 0; i < 100; ++i) EXPECT_EQ(a.Uniform(0, 1000), b.Uniform(0, 1000));

    Rng rng(1);
    std::set<int64_t> seen;
    for (int i = 0; i < 10'000; ++i) {
        const int64_t v = rng.Uniform(1, 3);
        EXPECT_TRUE(v >= 1 && v <= 3) << v;
        seen.insert(v);
    }
    EXPECT_EQ(seen, (std::set<int64_t>{1, 2, 3}));

    for (int i = 0; i < 10'000; ++i) {
        const double d = rng.UniformDouble();
        EXPECT_TRUE(d >= 0.0 && d < 1.0) << d;
    }
}

TEST(TpchStreamsTest, StreamDependsOnlyOnSeed) {
    const auto a = MakeStream(7, 24);
    const auto b = MakeStream(7, 24);
    ASSERT_EQ(a.size(), 24u);
    ASSERT_EQ(b.size(), 24u);
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].tag, b[i].tag);
        EXPECT_EQ(a[i].sql, b[i].sql);
    }

    const auto c = MakeStream(8, 24);
    ASSERT_EQ(c.size(), 24u);
    bool differs = false;
    for (size_t i = 0; i < a.size(); ++i) differs = differs || a[i].sql != c[i].sql;
    EXPECT_TRUE(differs);
}

TEST(TpchStreamsTest, StreamRoundsArePermutations) {
    const auto items = MakeStream(3, 20);
    ASSERT_EQ(items.size(), 20u);

    std::set<std::string> all_tags;
    for (const auto& q : Queries()) all_tags.insert(q.tag);
    ASSERT_EQ(all_tags.size(), 8u);

    for (size_t round_start : {size_t{0}, size_t{8}}) {
        const auto tags = Tags(items, round_start, round_start + 8);
        EXPECT_EQ(std::set<std::string>(tags.begin(), tags.end()), all_tags) << "round at " << round_start;
    }

    // The last round is cut short, so its 4 tags can't repeat.
    const auto last = Tags(items, 16, 20);
    EXPECT_EQ(std::set<std::string>(last.begin(), last.end()).size(), 4u);

    for (const auto& item : items) {
        EXPECT_EQ(item.kind, StreamItem::Kind::TPCH);
        EXPECT_TRUE(item.index_name.empty());
        EXPECT_FALSE(item.sql.empty());
    }
}

// Every round starts from the queries in their usual order, shuffles them, and draws each
// query's constants in that order. Replays that by hand and compares all 20 items.
TEST(TpchStreamsTest, EachRoundShufflesQueriesOrder) {
    constexpr size_t kItems = 20;
    const auto templates = QueryTemplates();
    const auto items = MakeStream(3, kItems);
    ASSERT_EQ(items.size(), kItems);

    Rng rng(3);
    size_t next = 0;
    while (next < kItems) {
        std::vector<std::string> tags;
        for (const auto& q : Queries()) tags.push_back(q.tag);
        rng.Shuffle(tags);

        for (const auto& tag : tags) {
            if (next == kItems) break;
            const auto t = std::find_if(templates.begin(), templates.end(),
                                        [&](const QueryTemplate& q) { return q.tag == tag; });
            ASSERT_NE(t, templates.end());

            EXPECT_EQ(items[next].tag, tag) << "item " << next;
            EXPECT_EQ(items[next].sql, Render(t->sql, RandomParams(tag, rng))) << "item " << next;
            ++next;
        }
    }
}

TEST(TpchStreamsTest, RandomParamsStayInRange) {
    constexpr int kDraws = 1000;
    Rng rng(5);

    const std::set<std::string> segments{"AUTOMOBILE", "BUILDING", "FURNITURE", "HOUSEHOLD", "MACHINERY"};
    const std::set<std::string> modes{"REG AIR", "AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB"};
    const std::regex brand("Brand#[1-5][1-5]");

    // Q5, Q6 and Q12 take a whole year: Jan 1 of 1993..1997 up to the next Jan 1.
    const auto expect_year = [](const Params& p) {
        const std::string date = Get(p, "date");
        const int year = YearOf(date);
        EXPECT_TRUE(year >= 1993 && year <= 1997) << date;
        EXPECT_EQ(date, std::to_string(year) + "-01-01");
        EXPECT_EQ(Get(p, "date_end"), std::to_string(year + 1) + "-01-01");
    };

    std::set<int> q6_quantities;
    std::set<int> q19_q1_values;
    std::set<std::string> q12_seen_modes;

    for (int i = 0; i < kDraws; ++i) {
        {  // Q1: 1998-12-01 minus 60 to 120 days
            const int32_t date = Day(Get(RandomParams("Q1", rng), "date"));
            EXPECT_TRUE(date >= Day("1998-08-03") && date <= Day("1998-10-02"));
        }
        {  // Q3
            const auto p = RandomParams("Q3", rng);
            EXPECT_EQ(segments.count(Get(p, "segment")), 1u) << Get(p, "segment");
            const int32_t date = Day(Get(p, "date"));
            EXPECT_TRUE(date >= Day("1995-03-01") && date <= Day("1995-03-31"));
        }
        expect_year(RandomParams("Q5", rng));
        {  // Q6
            const auto p = RandomParams("Q6", rng);
            expect_year(p);
            const double lo = std::stod(Get(p, "disc_lo"));
            const double hi = std::stod(Get(p, "disc_hi"));
            EXPECT_NEAR(hi - lo, 0.02, 1e-9);
            EXPECT_TRUE(lo >= 0.01 - 1e-9 && lo <= 0.08 + 1e-9) << lo;
            const int qty = std::stoi(Get(p, "qty"));
            EXPECT_TRUE(qty == 24 || qty == 25) << qty;
            q6_quantities.insert(qty);
        }
        {  // Q10: a first of month, three-month window
            const auto p = RandomParams("Q10", rng);
            const std::string date = Get(p, "date");
            EXPECT_EQ(date.substr(8), "01");
            EXPECT_TRUE(Day(date) >= Day("1993-02-01") && Day(date) <= Day("1995-01-01")) << date;
            EXPECT_EQ(Get(p, "date_end"), AddMonths(date, 3));
        }
        {  // Q12
            const auto p = RandomParams("Q12", rng);
            expect_year(p);
            const std::string m1 = Get(p, "mode1");
            const std::string m2 = Get(p, "mode2");
            EXPECT_NE(m1, m2);
            EXPECT_EQ(modes.count(m1), 1u) << m1;
            EXPECT_EQ(modes.count(m2), 1u) << m2;
            q12_seen_modes.insert({m1, m2});
        }
        {  // Q14: a first of month, one-month window
            const auto p = RandomParams("Q14", rng);
            const std::string date = Get(p, "date");
            EXPECT_EQ(date.substr(8), "01");
            EXPECT_TRUE(Day(date) >= Day("1993-01-01") && Day(date) <= Day("1997-12-01")) << date;
            EXPECT_EQ(Get(p, "date_end"), AddMonths(date, 1));
        }
        {  // Q19
            const auto p = RandomParams("Q19", rng);
            for (const char* name : {"brand1", "brand2", "brand3"}) {
                EXPECT_TRUE(std::regex_match(Get(p, name), brand)) << Get(p, name);
            }
            const struct { const char* lo; const char* hi; int min; int max; } ranges[] = {
                {"q1", "q1_hi", 1, 10}, {"q2", "q2_hi", 10, 20}, {"q3", "q3_hi", 20, 30}};
            for (const auto& r : ranges) {
                const int lo = std::stoi(Get(p, r.lo));
                EXPECT_TRUE(lo >= r.min && lo <= r.max) << r.lo << " = " << lo;
                EXPECT_EQ(std::stoi(Get(p, r.hi)), lo + 10) << r.hi;
            }
            q19_q1_values.insert(std::stoi(Get(p, "q1")));
        }
    }

    EXPECT_EQ(q6_quantities, (std::set<int>{24, 25}));
    EXPECT_EQ(q12_seen_modes, modes);
    EXPECT_EQ(*q19_q1_values.begin(), 1);
    EXPECT_EQ(*q19_q1_values.rbegin(), 10);
}

TEST(TpchStreamsTest, DefaultLengths) {
    EXPECT_EQ(DefaultStreamLength(Workload::STREAM, "0.01"), 64u);
    EXPECT_EQ(DefaultStreamLength(Workload::STREAM, "0.10"), 32u);
    EXPECT_EQ(DefaultStreamLength(Workload::STREAM, "1"), 16u);

    EXPECT_EQ(DefaultStreamLength(Workload::LOOKUP, "0.01"), 2000u);
    EXPECT_EQ(DefaultStreamLength(Workload::LOOKUP, "0.10"), 1000u);
    EXPECT_EQ(DefaultStreamLength(Workload::LOOKUP, "1"), 500u);

    EXPECT_EQ(DefaultStreamLength(Workload::MIXED, "0.01"), 32u);
    EXPECT_EQ(DefaultStreamLength(Workload::MIXED, "0.10"), 16u);
    EXPECT_EQ(DefaultStreamLength(Workload::MIXED, "1"), 8u);

    EXPECT_EQ(DefaultStreamLength(Workload::STREAM, "0.5"), std::nullopt);
    EXPECT_EQ(DefaultStreamLength(Workload::SINGLE, "0.01"), std::nullopt);
}

TEST(TpchStreamsTest, ZipfFavorsLowRanks) {
    constexpr size_t kRanks = 2526;
    constexpr int kDraws = 200'000;
    const Zipf zipf(kRanks, 1.0);
    Rng rng(11);

    std::vector<int> count(kRanks, 0);
    for (int i = 0; i < kDraws; ++i) {
        const size_t rank = zipf.Draw(rng);
        ASSERT_LT(rank, kRanks);
        ++count[rank];
    }

    EXPECT_GT(count[0], count[9]);
    EXPECT_GT(count[9], count[99]);

    // Rank 0 has weight 1 out of the harmonic number H(2526).
    double harmonic = 0;
    for (size_t k = 1; k <= kRanks; ++k) harmonic += 1.0 / static_cast<double>(k);
    const double expected = kDraws / harmonic;
    EXPECT_NEAR(count[0], expected, expected * 0.10);
}

TEST(TpchStreamsTest, LookupsAreShortRangesOnBothTables) {
    const auto items = MakeLookups(1, 2000);
    ASSERT_EQ(items.size(), 2000u);

    struct Table {
        const char* name;
        const char* index_name;
        const char* from;
        const char* first_day;
        const char* last_day;
        int count = 0;
    };
    std::map<std::string, Table> tables{
        {"LS", {"lineitem", "idx_lineitem_shipdate", "l_shipdate", "1992-01-02", "1998-12-01"}},
        {"LO", {"orders", "idx_orders_orderdate", "o_orderdate", "1992-01-01", "1998-08-02"}},
    };

    for (const auto& item : items) {
        EXPECT_EQ(item.kind, StreamItem::Kind::LOOKUP);
        const auto found = tables.find(item.tag);
        ASSERT_NE(found, tables.end()) << item.tag;
        auto& table = found->second;
        ++table.count;

        EXPECT_EQ(item.index_name, table.index_name);
        EXPECT_NE(item.sql.find(std::string("FROM ") + table.name + " WHERE " + table.from + " >= '"),
                  std::string::npos) << item.sql;

        const auto [d1, d2] = Window(item.sql);
        EXPECT_TRUE(d2 - d1 >= 1 && d2 - d1 <= 3) << item.sql;
        EXPECT_TRUE(d1 >= Day(table.first_day) && d1 <= Day(table.last_day)) << item.sql;
    }

    EXPECT_GE(tables.at("LS").count, 800);
    EXPECT_GE(tables.at("LO").count, 800);

    const auto again = MakeLookups(1, 2000);
    ASSERT_EQ(again.size(), items.size());
    for (size_t i = 0; i < items.size(); ++i) {
        EXPECT_EQ(items[i].tag, again[i].tag);
        EXPECT_EQ(items[i].sql, again[i].sql);
        EXPECT_EQ(items[i].index_name, again[i].index_name);
    }
}

TEST(TpchStreamsTest, MixedInterleavesStreamAndLookups) {
    constexpr size_t kQueries = 5;
    constexpr size_t kPerQuery = kMixedLookupsPerQuery;
    const auto mixed = MakeMixed(4, kQueries);
    const auto stream = MakeStream(4, kQueries);
    const auto lookups = MakeLookups(4, kQueries * kPerQuery);

    ASSERT_EQ(mixed.size(), kQueries * (1 + kPerQuery));
    for (size_t i = 0; i < kQueries; ++i) {
        const auto& query = mixed[i * (1 + kPerQuery)];
        EXPECT_EQ(query.kind, StreamItem::Kind::TPCH);
        EXPECT_EQ(query.tag, stream[i].tag);
        EXPECT_EQ(query.sql, stream[i].sql);

        for (size_t j = 0; j < kPerQuery; ++j) {
            const auto& lookup = mixed[i * (1 + kPerQuery) + 1 + j];
            EXPECT_EQ(lookup.kind, StreamItem::Kind::LOOKUP);
            EXPECT_EQ(lookup.tag, lookups[i * kPerQuery + j].tag);
            EXPECT_EQ(lookup.sql, lookups[i * kPerQuery + j].sql);
        }
    }

    // MakeWorkload hands each workload to its generator; the single workload has none.
    EXPECT_EQ(MakeWorkload(Workload::MIXED, 4, kQueries).back().sql, mixed.back().sql);
    EXPECT_EQ(MakeWorkload(Workload::STREAM, 4, kQueries).back().sql, stream.back().sql);
    EXPECT_EQ(MakeWorkload(Workload::LOOKUP, 4, 10).back().sql, MakeLookups(4, 10).back().sql);
    EXPECT_THROW((void)MakeWorkload(Workload::SINGLE, 4, kQueries), DatabaseException);
}

// SF=0.01 data with every TPC-H index built.
class TpchStreamsDataTest : public ::testing::Test {
protected:
    std::string test_file_;
    std::unique_ptr<Database> db_;

    // CTest runs from build/test/, so the data is usually two levels up.
    static std::string FindDataDir() {
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

    void SetUp() override {
        test_file_ = (std::filesystem::temp_directory_path() /
                      ("shilmandb_tpch_streams_test_" + std::to_string(::getpid()) + ".db")).string();
        std::filesystem::remove(test_file_);

        const auto data_dir = FindDataDir();
        if (data_dir.empty()) {
            GTEST_SKIP() << "TPC-H SF=0.01 data not found. Run bench/tpch/generate_tpch_data.sh first.";
        }

        // 32 MB, enough to hold all of SF=0.01.
        constexpr size_t kPoolSize = 4096;
        db_ = std::make_unique<Database>(test_file_, kPoolSize);

        for (const auto& td : TableDescs()) {
            db_->LoadTable(td.name, td.schema, data_dir + td.tbl_file);
        }
        for (const auto& idx : IndexDescs()) {
            (void)db_->GetCatalog()->CreateIndex(idx.index_name, idx.table_name, idx.column_name);
        }
    }

    void TearDown() override {
        db_.reset();
        std::filesystem::remove(test_file_);
    }
};

TEST_F(TpchStreamsDataTest, LookupPlansUseTheirIndex) {
    std::map<std::string, std::string> index_of_sql;
    for (const auto& item : MakeLookups(1, 500)) index_of_sql.emplace(item.sql, item.index_name);
    ASSERT_GT(index_of_sql.size(), 100u);

    for (const double page_us : {0.6, 50.0, 100.0}) {
        db_->SetScanCosts(ScanCosts::ForPageRead(page_us));
        for (const auto& [sql, index_name] : index_of_sql) {
            EXPECT_TRUE(PlanUsesIndex(*db_, sql, index_name)) << page_us << " us: " << sql;
        }
    }
    db_->SetScanCosts(std::nullopt);
}

TEST_F(TpchStreamsDataTest, PlanUsesIndexFollowsDatabaseCosts) {
    const std::string sql =
        "SELECT COUNT(*) FROM lineitem WHERE l_shipdate >= '1995-01-01' AND l_shipdate < '1995-04-01'";
    EXPECT_TRUE(PlanUsesIndex(*db_, sql, "idx_lineitem_shipdate"));

    db_->GetDiskManager()->SetReadPenalty(std::chrono::microseconds{100});
    EXPECT_FALSE(PlanUsesIndex(*db_, sql, "idx_lineitem_shipdate"));
    db_->GetDiskManager()->SetReadPenalty(std::chrono::microseconds{0});
    EXPECT_TRUE(PlanUsesIndex(*db_, sql, "idx_lineitem_shipdate"));

    db_->SetScanCosts(ScanCosts::ForPageRead(100));
    EXPECT_FALSE(PlanUsesIndex(*db_, sql, "idx_lineitem_shipdate"));
    db_->SetScanCosts(std::nullopt);
    EXPECT_TRUE(PlanUsesIndex(*db_, sql, "idx_lineitem_shipdate"));
}

TEST_F(TpchStreamsDataTest, ExecuteSqlFollowsDatabaseCosts) {
    const std::string sql =
        "SELECT COUNT(*) FROM lineitem WHERE l_shipdate >= '1995-01-01' AND l_shipdate < '1995-04-01'";
    QueryOptions options;
    options.analyze = true;

    const auto has_op = [](const QueryResult& r, const std::string& prefix) {
        return std::any_of(r.profile.operators.begin(), r.profile.operators.end(),
                           [&](const auto& op) { return op.name.rfind(prefix, 0) == 0; });
    };

    const auto cheap = db_->ExecuteSQL(sql, options);
    EXPECT_TRUE(has_op(cheap, "IndexScan lineitem"));

    db_->SetScanCosts(ScanCosts::ForPageRead(100));
    const auto slow = db_->ExecuteSQL(sql, options);
    EXPECT_TRUE(has_op(slow, "SeqScan lineitem"));
    EXPECT_FALSE(has_op(slow, "IndexScan"));
    db_->SetScanCosts(std::nullopt);

    ASSERT_EQ(cheap.tuples.size(), 1u);
    ASSERT_EQ(slow.tuples.size(), 1u);
    EXPECT_EQ(cheap.tuples[0].GetValue(cheap.schema, 0).integer_, slow.tuples[0].GetValue(slow.schema, 0).integer_);
}

TEST_F(TpchStreamsDataTest, PlanUsesIndexIsFalseForScans) {
    const auto q6 = FindQuery("Q6");
    ASSERT_TRUE(q6.has_value());
    EXPECT_FALSE(PlanUsesIndex(*db_, q6->sql, "idx_lineitem_shipdate"));

    // A one-day lineitem lookup does use its own index, but not the orders one.
    const std::string one_day =
        "SELECT COUNT(*), SUM(l_extendedprice) FROM lineitem "
        "WHERE l_shipdate >= '1995-03-15' AND l_shipdate < '1995-03-16'";
    EXPECT_TRUE(PlanUsesIndex(*db_, one_day, "idx_lineitem_shipdate"));
    EXPECT_FALSE(PlanUsesIndex(*db_, one_day, "idx_orders_orderdate"));
}

TEST_F(TpchStreamsDataTest, PlanCheckLeavesPoolAlone) {
    auto* bpm = db_->GetBufferPoolManager();
    bpm->ResetStats();

    for (const auto& item : MakeLookups(1, 200)) {
        (void)PlanUsesIndex(*db_, item.sql, item.index_name);
    }

    EXPECT_EQ(bpm->GetHitCount(), 0u);
    EXPECT_EQ(bpm->GetMissCount(), 0u);
    EXPECT_EQ(bpm->GetVisitCount(), 0u);
}

TEST_F(TpchStreamsDataTest, WorkloadItemsRun) {
    for (const auto& item : MakeWorkload(Workload::STREAM, 1, 16)) {
        EXPECT_NO_THROW((void)db_->ExecuteSQL(item.sql)) << item.tag;
    }

    // A lookup is one COUNT and SUM, so it always comes back as one row.
    for (const auto& item : MakeLookups(1, 50)) {
        const auto result = db_->ExecuteSQL(item.sql);
        EXPECT_EQ(result.tuples.size(), 1u) << item.sql;
    }
}

}  // namespace shilmandb::tpch
