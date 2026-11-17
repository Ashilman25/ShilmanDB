// Loads three TPC-H tables columnar, indexes supplier, then replaces supplier over and over.
// Odd rounds load the first 50 lines (copied to <db_file>.half.tbl), even rounds the full file.
// A crash test kills it partway and checks the file still opens.
// Usage: columnar_crash_helper <db_file> <data_dir> <rounds>

#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>

#include "engine/database.hpp"
#include "tpch/tpch_workload.hpp"

int main(int argc, char** argv) {
    using namespace shilmandb;
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <db_file> <data_dir> <rounds>\n", argv[0]);
        return 2;
    }
    const std::string file = argv[1];
    std::string dir = argv[2];
    if (!dir.empty() && dir.back() != '/') dir += '/';
    const int rounds = std::stoi(argv[3]);
    const std::string full = dir + "supplier.tbl";
    const std::string half = file + ".half.tbl";

    try {
        {
            std::ifstream in(full);
            std::ofstream out(half, std::ios::trunc);
            std::string line;
            for (int n = 0; n < 50 && std::getline(in, line); ++n) out << line << '\n';
        }

        LoadOptions options;
        options.storage = StorageKind::COLUMNAR;
        Database db(file, 256);
        db.LoadTable("nation", tpch::NationSchema(), dir + "nation.tbl", options);
        db.LoadTable("region", tpch::RegionSchema(), dir + "region.tbl", options);
        db.LoadTable("supplier", tpch::SupplierSchema(), full, options);
        if (db.GetCatalog()->CreateIndex("idx_supplier_nationkey", "supplier", "s_nationkey") == nullptr) {
            std::fprintf(stderr, "can't create idx_supplier_nationkey\n");
            return 1;
        }
        std::printf("ready\n");
        std::fflush(stdout);

        for (int round = 1; round <= rounds; ++round) {
            db.LoadTable("supplier", tpch::SupplierSchema(), round % 2 == 1 ? half : full, options);
            std::printf("round %d committed\n", round);
            std::fflush(stdout);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "columnar_crash_helper: %s\n", e.what());
        std::filesystem::remove(half);
        return 1;
    }
    std::filesystem::remove(half);
    return 0;
}
