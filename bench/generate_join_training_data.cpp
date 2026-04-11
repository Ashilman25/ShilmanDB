#include "engine/database.hpp"
#include "planner/join_order_optimizer.hpp"
#include "parser/ast.hpp"
#include "catalog/table_stats.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace shilmandb;



static auto RegionSchema() -> Schema {
    return Schema({
        {"r_regionkey", TypeId::BIGINT},
        {"r_name", TypeId::VARCHAR},
        {"r_comment", TypeId::VARCHAR},
    });
}

static auto NationSchema() -> Schema {
    return Schema({
        {"n_nationkey", TypeId::BIGINT},
        {"n_name", TypeId::VARCHAR},
        {"n_regionkey", TypeId::BIGINT},
        {"n_comment", TypeId::VARCHAR},
    });
}

static auto SupplierSchema() -> Schema {
    return Schema({
        {"s_suppkey", TypeId::BIGINT},
        {"s_name", TypeId::VARCHAR},
        {"s_address", TypeId::VARCHAR},
        {"s_nationkey", TypeId::BIGINT},
        {"s_phone", TypeId::VARCHAR},
        {"s_acctbal", TypeId::DECIMAL},
        {"s_comment", TypeId::VARCHAR},
    });
}

static auto CustomerSchema() -> Schema {
    return Schema({
        {"c_custkey", TypeId::BIGINT},
        {"c_name", TypeId::VARCHAR},
        {"c_address", TypeId::VARCHAR},
        {"c_nationkey", TypeId::BIGINT},
        {"c_phone", TypeId::VARCHAR},
        {"c_acctbal", TypeId::DECIMAL},
        {"c_mktsegment", TypeId::VARCHAR},
        {"c_comment", TypeId::VARCHAR},
    });
}

static auto OrdersSchema() -> Schema {
    return Schema({
        {"o_orderkey", TypeId::BIGINT},
        {"o_custkey", TypeId::BIGINT},
        {"o_orderstatus", TypeId::VARCHAR},
        {"o_totalprice", TypeId::DECIMAL},
        {"o_orderdate", TypeId::DATE},
        {"o_orderpriority", TypeId::VARCHAR},
        {"o_clerk", TypeId::VARCHAR},
        {"o_shippriority", TypeId::INTEGER},
        {"o_comment", TypeId::VARCHAR},
    });
}

static auto LineitemSchema() -> Schema {
    return Schema({
        {"l_orderkey", TypeId::BIGINT},
        {"l_partkey", TypeId::BIGINT},
        {"l_suppkey", TypeId::BIGINT},
        {"l_linenumber", TypeId::INTEGER},
        {"l_quantity", TypeId::DECIMAL},
        {"l_extendedprice", TypeId::DECIMAL},
        {"l_discount", TypeId::DECIMAL},
        {"l_tax", TypeId::DECIMAL},
        {"l_returnflag", TypeId::VARCHAR},
        {"l_linestatus", TypeId::VARCHAR},
        {"l_shipdate", TypeId::DATE},
        {"l_commitdate", TypeId::DATE},
        {"l_receiptdate", TypeId::DATE},
        {"l_shipinstruct", TypeId::VARCHAR},
        {"l_shipmode", TypeId::VARCHAR},
        {"l_comment", TypeId::VARCHAR},
    });
}

struct TableDesc {
    std::string name;
    Schema schema;
    std::string tbl_file;
};

static auto BuildTableDescs() -> std::vector<TableDesc> {
    return {
        {"region",   RegionSchema(),   "region.tbl"},
        {"nation",   NationSchema(),   "nation.tbl"},
        {"supplier", SupplierSchema(), "supplier.tbl"},
        {"customer", CustomerSchema(), "customer.tbl"},
        {"orders",   OrdersSchema(),   "orders.tbl"},
        {"lineitem", LineitemSchema(),  "lineitem.tbl"},
    };
}



static constexpr int kNumTables = 6;
// Table indices: 0=region, 1=nation, 2=supplier, 3=customer, 4=orders, 5=lineitem

static const std::vector<std::string> kTableNames = {
    "region", "nation", "supplier", "customer", "orders", "lineitem"
};

struct JoinEdgeDef {
    int table_a;
    std::string col_a;
    int table_b;
    std::string col_b;
};

static auto BuildJoinEdges() -> std::vector<JoinEdgeDef> {
    return {
        {0, "r_regionkey", 1, "n_regionkey"},   // region — nation
        {1, "n_nationkey", 2, "s_nationkey"},    // nation — supplier
        {1, "n_nationkey", 3, "c_nationkey"},    // nation — customer
        {3, "c_custkey",   4, "o_custkey"},      // customer — orders
        {4, "o_orderkey",  5, "l_orderkey"},     // orders — lineitem
        {2, "s_suppkey",   5, "l_suppkey"},      // supplier — lineitem
    };
}


static auto BuildConnectedSubsets() -> std::vector<std::vector<int>> {
    return {

        {0, 1}, {1, 2}, {1, 3}, {3, 4}, {4, 5}, {2, 5},

        {0, 1, 2}, {0, 1, 3}, {1, 2, 3}, {1, 2, 5},
        {1, 3, 4}, {2, 4, 5}, {3, 4, 5},

        {0, 1, 2, 3}, {0, 1, 2, 5}, {0, 1, 3, 4}, {1, 2, 3, 4},
        {1, 2, 3, 5}, {1, 2, 4, 5}, {1, 3, 4, 5}, {2, 3, 4, 5},

        {0, 1, 2, 3, 4}, {0, 1, 2, 3, 5}, {0, 1, 2, 4, 5},
        {0, 1, 3, 4, 5}, {1, 2, 3, 4, 5},

        {0, 1, 2, 3, 4, 5},
    };
}



static auto MakeEquiJoinClause(const std::string& left_table, const std::string& left_col, const std::string& right_table, const std::string& right_col) -> JoinClause {
    JoinClause jc;
    jc.join_type = JoinType::INNER;
    jc.right_table.table_name = right_table;

    auto left = std::make_unique<ColumnRef>();
    left->table_name = left_table;
    left->column_name = left_col;

    auto right = std::make_unique<ColumnRef>();
    right->table_name = right_table;
    right->column_name = right_col;

    auto eq = std::make_unique<BinaryOp>();
    eq->op = BinaryOp::Op::EQ;
    eq->left = std::move(left);
    eq->right = std::move(right);

    jc.on_condition = std::move(eq);
    return jc;
}


//subset data builders

static auto BuildSubsetTableRefs(const std::vector<int>& subset) -> std::vector<TableRef> {
    std::vector<TableRef> refs;
    refs.reserve(subset.size());
    for (int idx : subset) {
        refs.push_back({kTableNames[idx], std::nullopt});
    }
    return refs;
}

static auto BuildSubsetJoinClauses(const std::vector<int>& subset, const std::vector<JoinEdgeDef>& all_edges) -> std::vector<JoinClause> {
    std::vector<bool> in_subset(kNumTables, false);
    for (int idx : subset) {
        in_subset[idx] = true;
    }

    std::vector<JoinClause> clauses;
    for (const auto& edge : all_edges) {
        if (in_subset[edge.table_a] && in_subset[edge.table_b]) {
            clauses.push_back(MakeEquiJoinClause(kTableNames[edge.table_a], edge.col_a, kTableNames[edge.table_b], edge.col_b));
        }
    }
    return clauses;
}

static auto BuildSubsetStats(const std::vector<TableStats>& all_stats, const std::vector<int>& subset) -> std::vector<TableStats> {
    std::vector<TableStats> result;
    result.reserve(subset.size());
    for (int idx : subset) {
        result.push_back(all_stats[idx]);
    }
    return result;
}





static auto PerturbStats(const std::vector<TableStats>& baseline, const std::vector<int>& subset, std::mt19937& rng) -> std::vector<TableStats> {
    std::vector<TableStats> result = baseline;

    std::uniform_real_distribution<double> log_dist(-std::log(10.0), std::log(100.0));

    for (int idx : subset) {
        double scale = std::exp(log_dist(rng));
        auto& stats = result[idx];
        auto row_scaled = static_cast<uint64_t>(std::round(static_cast<double>(baseline[idx].row_count) * scale));
        stats.row_count = std::max(uint64_t{1}, row_scaled);

        for (auto& [col, distinct] : stats.distinct_counts) {
            auto scaled = static_cast<uint64_t>(std::round(static_cast<double>(distinct) * scale));
            distinct = std::max(uint64_t{1}, std::min(scaled, stats.row_count));
        }
    }
    return result;
}



struct TrainingSample {
    std::vector<float> features;     
    int num_tables;
    std::vector<int> optimal_order;  
    double best_cost;
};

static auto GenerateSample(const std::vector<TableRef>& subset_refs, const std::vector<JoinClause>& subset_joins, const std::vector<TableStats>& subset_stats) -> TrainingSample {
    TrainingSample sample;
    sample.num_tables = static_cast<int>(subset_refs.size());
    sample.features = JoinOrderOptimizer::BuildFeatureVector(subset_refs, subset_joins, subset_stats);

    const int n = sample.num_tables;
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);

    double best_cost = std::numeric_limits<double>::max();
    std::vector<int> best_order = perm;

    do {
        double cost = JoinOrderOptimizer::EstimateCost(perm, subset_refs, subset_joins, subset_stats);
        if (cost < best_cost) {
            best_cost = cost;
            best_order = perm;
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    sample.best_cost = best_cost;


    sample.optimal_order.resize(6);
    for (int i = 0; i < 6; ++i) {
        sample.optimal_order[i] = (i < n) ? best_order[i] : i;
    }

    return sample;
}



static void WriteCsv(const std::string& path, const std::vector<TrainingSample>& samples) {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Error: cannot open output file: " << path << "\n";
        std::exit(1);
    }

    // Header
    for (int i = 0; i < 48; ++i) {
        out << "feature_" << i << ",";
    }
    out << "num_tables,";
    for (int i = 0; i < 6; ++i) {
        out << "order_" << i << ",";
    }
    out << "best_cost\n";

    // Data rows
    out << std::fixed << std::setprecision(6);
    for (const auto& s : samples) {
        for (int i = 0; i < 48; ++i) {
            out << s.features[i] << ",";
        }
        out << s.num_tables << ",";
        for (int i = 0; i < 6; ++i) {
            out << s.optimal_order[i] << ",";
        }
        out << s.best_cost << "\n";
    }

    out.close();
}



struct Args {
    std::string sf;
    std::string db_file;
    std::string data_dir;
    std::string output = "bench/tpch_data/join_training_data.csv";
    int perturbations = 40;
    uint32_t seed = 42;
};

static void PrintUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " --sf <scale_factor> --db-file <path> --data-dir <path>" << " [--output <path>] [--perturbations <N>] [--seed <N>]\n";
}

static auto ParseArgs(int argc, char* argv[]) -> Args {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--sf" && i + 1 < argc) {
            args.sf = argv[++i];
        } else if (arg == "--db-file" && i + 1 < argc) {
            args.db_file = argv[++i];
        } else if (arg == "--data-dir" && i + 1 < argc) {
            args.data_dir = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            args.output = argv[++i];
        } else if (arg == "--perturbations" && i + 1 < argc) {
            args.perturbations = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            args.seed = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else {
            PrintUsage(argv[0]);
            std::exit(1);
        }
    }

    if (args.sf.empty() || args.db_file.empty() || args.data_dir.empty()) {
        PrintUsage(argv[0]);
        std::exit(1);
    }

    if (args.data_dir.back() != '/') {
        args.data_dir += '/';
    }

    return args;
}



int main(int argc, char* argv[]) {
    auto args = ParseArgs(argc, argv);

    std::cout << "=== Join Training Data Generator ===\n";
    std::cout << "SF=" << args.sf << "  perturbations=" << args.perturbations << "  seed=" << args.seed << "\n\n";


    constexpr size_t kBufferPoolSize = 4096;
    Database db(args.db_file, kBufferPoolSize);

    auto table_descs = BuildTableDescs();
    for (const auto& td : table_descs) {
        std::string path = args.data_dir + td.tbl_file;
        std::cout << "Loading " << td.name << " ... " << std::flush;

        auto start = std::chrono::steady_clock::now();
        db.LoadTable(td.name, td.schema, path);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();

        auto* info = db.GetCatalog()->GetTable(td.name);
        std::cout << info->stats.row_count << " rows (" << ms << " ms)\n";
    }



    std::vector<TableStats> baseline_stats;
    baseline_stats.reserve(kNumTables);
    for (const auto& name : kTableNames) {
        auto* info = db.GetCatalog()->GetTable(name);
        baseline_stats.push_back(info->stats);
    }



    auto all_edges = BuildJoinEdges();
    auto subsets = BuildConnectedSubsets();
    std::mt19937 rng(args.seed);

    std::vector<TrainingSample> samples;
    samples.reserve(subsets.size() * (1 + args.perturbations));

    std::vector<int> size_counts(7, 0);  // index 0-6, we use 2-6

    std::cout << "\nGenerating samples ...\n";

    for (const auto& subset : subsets) {
        auto subset_refs = BuildSubsetTableRefs(subset);
        auto subset_joins = BuildSubsetJoinClauses(subset, all_edges);

        auto real_stats = BuildSubsetStats(baseline_stats, subset);
        samples.push_back(GenerateSample(subset_refs, subset_joins, real_stats));
        size_counts[subset.size()]++;

        for (int p = 0; p < args.perturbations; ++p) {
            auto perturbed_all = PerturbStats(baseline_stats, subset, rng);
            auto perturbed_subset = BuildSubsetStats(perturbed_all, subset);
            samples.push_back(GenerateSample(subset_refs, subset_joins, perturbed_subset));
            size_counts[subset.size()]++;
        }
    }

    // Count unique orderings (for diversity check)
    std::set<std::vector<int>> seen;
    for (const auto& s : samples) {
        seen.insert(s.optimal_order);
    }
    int unique_orderings = static_cast<int>(seen.size());

    WriteCsv(args.output, samples);

    //summary
    std::cout << "\nConnected subsets: " << subsets.size() << "\n";
    std::cout << "Perturbations per subset: " << args.perturbations << "\n";
    std::cout << "Total samples: " << samples.size() << "\n";
    for (int sz = 2; sz <= 6; ++sz) {
        if (size_counts[sz] > 0) {
            std::cout << "  Size " << sz << ": " << size_counts[sz] << " samples\n";
        }
    }
    std::cout << "Unique optimal orderings: " << unique_orderings << "\n";
    std::cout << "Output: " << args.output << "\n";
    std::cout << "\nDone.\n";

    return 0;
}
