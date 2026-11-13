// Loads a dataset into ShilmanDB once, then answers "<id>\t<sql>" lines on stdin
// with one JSON line each on stdout: rows, column types and the query profile.

#include "engine/database.hpp"
#include "parser/parse_exception.hpp"

#include <unistd.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

using namespace shilmandb;
namespace fs = std::filesystem;

namespace {

constexpr size_t kRowCap = 10000;
constexpr size_t kDefaultPoolSize = 32768;
constexpr std::string_view kUsage = "usage: analyst_bridge --tables <tables.tsv> [--db-file PATH] [--pool-size N]";

using Clock = std::chrono::steady_clock;

// Milliseconds since start.
[[nodiscard]] double MsSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// The type's name, as tables.tsv and the replies spell it.
[[nodiscard]] constexpr std::string_view TypeName(TypeId type) {
    switch (type) {
        case TypeId::INTEGER: return "INTEGER";
        case TypeId::BIGINT:  return "BIGINT";
        case TypeId::DECIMAL: return "DECIMAL";
        case TypeId::VARCHAR: return "VARCHAR";
        case TypeId::DATE:    return "DATE";
        case TypeId::INVALID: break;
    }
    return "INVALID";
}

// JSON, written by hand

// Appends s as a JSON string. Control bytes become \u00XX; UTF-8 passes through.
void AppendString(std::string& out, std::string_view s) {
    out += '"';
    for (const char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (const auto byte = static_cast<unsigned char>(c); byte < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", byte);
            out += buf;
        } else {
            out += c;
        }
    }
    out += '"';
}

// Appends a double at full precision; inf and nan aren't JSON, so they become strings.
void AppendNumber(std::string& out, double d) {
    if (std::isnan(d)) {
        out += "\"nan\"";
    } else if (std::isinf(d)) {
        out += d > 0 ? "\"inf\"" : "\"-inf\"";
    } else {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.17g", d);
        out += buf;
    }
}

// Appends a whole number as is.
template <typename Int, std::enable_if_t<std::is_integral_v<Int>, int> = 0>
void AppendNumber(std::string& out, Int n) {
    out += std::to_string(n);
}

// Appends items as a JSON array, writing each one with append_item.
template <typename Items, typename AppendItem>
void AppendList(std::string& out, const Items& items, AppendItem append_item) {
    out += '[';
    bool first = true;
    for (const auto& item : items) {
        if (!first) out += ',';
        first = false;
        append_item(item);
    }
    out += ']';
}

// Appends a list of numbers as a JSON array.
template <typename T>
void AppendNumbers(std::string& out, const std::vector<T>& numbers) {
    AppendList(out, numbers, [&out](T n) { AppendNumber(out, n); });
}

// Numbers stay numbers; dates and text become strings.
void AppendValue(std::string& out, const Value& v) {
    switch (v.type_) {
        case TypeId::INTEGER: AppendNumber(out, v.integer_); return;
        case TypeId::BIGINT:  AppendNumber(out, v.bigint_); return;
        case TypeId::DECIMAL: AppendNumber(out, v.decimal_); return;
        case TypeId::DATE:
        case TypeId::VARCHAR: AppendString(out, v.ToString()); return;
        case TypeId::INVALID: break;
    }
    out += "null";
}

// The planner's join decision, then the measured rows and operator times.
void AppendProfile(std::string& out, const QueryProfile& profile) {
    const auto& join = profile.join;
    out += R"({"join":{"tables":)";
    AppendList(out, join.tables, [&out](const std::string& name) { AppendString(out, name); });
    out += R"(,"order":)";
    AppendNumbers(out, join.order);
    out += R"(,"source":)";
    AppendString(out, ToString(join.source));
    out += R"(,"edges":)";
    AppendList(out, join.edges, [&out](const TableEdge& edge) {
        out += '[';
        AppendNumber(out, edge.first);
        out += ',';
        AppendNumber(out, edge.second);
        out += ']';
    });
    out += R"(,"est_step_rows":)";
    AppendNumbers(out, join.est_step_rows);
    out += R"(,"est_scan_rows":)";
    AppendNumbers(out, join.est_scan_rows);

    out += R"(},"step_rows":)";
    AppendNumbers(out, profile.step_rows);
    out += R"(,"scan_rows":)";
    AppendNumbers(out, profile.scan_rows);
    out += R"(,"operators":)";
    AppendList(out, profile.operators, [&out](const OperatorProfile& op) {
        out += R"({"name":)";
        AppendString(out, op.name);
        out += R"(,"depth":)";
        AppendNumber(out, op.depth);
        out += R"(,"rows":)";
        AppendNumber(out, op.rows);
        out += R"(,"total_ms":)";
        AppendNumber(out, op.total_ms);
        out += R"(,"self_ms":)";
        AppendNumber(out, op.self_ms);
        out += '}';
    });
    out += '}';
}

// The reply for a query that ran: columns, up to kRowCap rows, profile and time.
[[nodiscard]] std::string OkReply(std::string_view id, const QueryResult& result, double ms) {
    const auto& schema = result.schema;
    const size_t shown = std::min(result.tuples.size(), kRowCap);

    std::string out = R"({"id":)";
    AppendString(out, id);
    out += R"(,"ok":true,"columns":)";
    AppendList(out, schema.GetColumns(), [&out](const Column& col) {
        out += R"({"name":)";
        AppendString(out, col.name);
        out += R"(,"type":)";
        AppendString(out, TypeName(col.type));
        out += '}';
    });

    out += R"(,"rows":[)";
    for (size_t r = 0; r < shown; ++r) {
        if (r > 0) out += ',';
        out += '[';
        for (uint32_t c = 0; c < schema.GetColumnCount(); ++c) {
            if (c > 0) out += ',';
            AppendValue(out, result.tuples[r].GetValue(schema, c));
        }
        out += ']';
    }
    out += R"(],"row_count":)";
    AppendNumber(out, result.tuples.size());
    out += R"(,"truncated":)";
    out += shown < result.tuples.size() ? "true" : "false";

    out += R"(,"profile":)";
    AppendProfile(out, result.profile);
    out += R"(,"ms":)";
    AppendNumber(out, ms);
    out += '}';
    return out;
}

// The reply for a request that failed: what kind of failure, and the engine's message.
[[nodiscard]] std::string ErrorReply(std::string_view id, std::string_view kind, std::string_view message, double ms) {
    std::string out = R"({"id":)";
    AppendString(out, id);
    out += R"(,"ok":false,"error":{"kind":)";
    AppendString(out, kind);
    out += R"(,"message":)";
    AppendString(out, message);
    out += R"(},"ms":)";
    AppendNumber(out, ms);
    out += '}';
    return out;
}

// Requests

// Plans, then runs, one "<id>\t<sql>" line.
// Planning first tells a bad query from a failed run.
[[nodiscard]] std::string HandleRequest(Database& db, std::string_view line) {
    const auto tab = line.find('\t');
    if (tab == std::string_view::npos) {
        return ErrorReply("", "request", "request line has no tab, expected <id><TAB><sql>", 0.0);
    }

    const auto id = line.substr(0, tab);
    const std::string sql{line.substr(tab + 1)};
    const auto start = Clock::now();

    try {
        (void)db.ExplainJoinOrder(sql);
    } catch (const ParseException& e) {
        return ErrorReply(id, "parse", e.what(), MsSince(start));
    } catch (const std::exception& e) {
        return ErrorReply(id, "plan", e.what(), MsSince(start));
    } catch (...) {
        return ErrorReply(id, "plan", "unknown error", MsSince(start));
    }

    QueryOptions options;
    options.analyze = true;
    try {  // a throw while writing the reply is a run error too
        const QueryResult result = db.ExecuteSQL(sql, options);
        return OkReply(id, result, MsSince(start));
    } catch (const std::exception& e) {
        return ErrorReply(id, "run", e.what(), MsSince(start));
    } catch (...) {
        return ErrorReply(id, "run", "unknown error", MsSince(start));
    }
}

// Answers one request per stdin line until stdin closes, flushing every reply.
void ServeRequests(Database& db) {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::cout << HandleRequest(db, line) << std::endl;
    }
}

// Arguments and tables.tsv

struct Args {
    fs::path tables;
    fs::path db_file;
    size_t pool_size{kDefaultPoolSize};
};

struct TableSpec {
    std::string name;
    fs::path file;
    Schema schema;
};

// A positive page count, or std::runtime_error.
[[nodiscard]] size_t ParsePoolSize(const std::string& text) {
    size_t n = 0;
    const char* end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(text.data(), end, n);
    if (ec != std::errc{} || ptr != end || n == 0) {
        throw std::runtime_error("--pool-size must be a positive whole number, got \"" + text + "\"");
    }
    return n;
}

// Reads the flags; anything unknown or missing throws std::runtime_error with the usage line.
[[nodiscard]] Args ParseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(flag + " needs a value\n" + std::string(kUsage));
            return argv[++i];
        };

        if (flag == "--tables") {
            args.tables = value();
        } else if (flag == "--db-file") {
            args.db_file = value();
        } else if (flag == "--pool-size") {
            args.pool_size = ParsePoolSize(value());
        } else {
            throw std::runtime_error("unknown argument " + flag + "\n" + std::string(kUsage));
        }
    }

    if (args.tables.empty()) throw std::runtime_error("--tables is required\n" + std::string(kUsage));
    if (args.db_file.empty()) args.db_file = "/tmp/analyst_bridge_" + std::to_string(::getpid()) + ".db";
    return args;
}

// Splits s at every sep, keeping empty pieces.
[[nodiscard]] std::vector<std::string_view> Split(std::string_view s, char sep) {
    std::vector<std::string_view> parts;
    for (size_t start = 0;;) {
        const auto end = s.find(sep, start);
        parts.push_back(s.substr(start, end - start));
        if (end == std::string_view::npos) return parts;
        start = end + 1;
    }
}

// The column type a tables.tsv names, or std::runtime_error.
[[nodiscard]] TypeId ParseType(std::string_view name) {
    for (const auto type : {TypeId::INTEGER, TypeId::BIGINT, TypeId::DECIMAL, TypeId::VARCHAR, TypeId::DATE}) {
        if (TypeName(type) == name) return type;
    }
    throw std::runtime_error("unknown column type \"" + std::string(name) + "\"");
}

// Builds a schema from "col:TYPE,col:TYPE,...".
[[nodiscard]] Schema ParseColumns(std::string_view spec) {
    std::vector<Column> columns;
    for (const auto part : Split(spec, ',')) {
        const auto colon = part.find(':');
        if (colon == std::string_view::npos || colon == 0) {
            throw std::runtime_error("bad column \"" + std::string(part) + "\", expected name:TYPE");
        }
        columns.emplace_back(std::string(part.substr(0, colon)), ParseType(part.substr(colon + 1)));
    }
    return Schema(std::move(columns));
}

// One "name<TAB>file<TAB>col:TYPE,..." line per table.
// A relative file resolves against the tsv's folder.
[[nodiscard]] std::vector<TableSpec> ReadTables(const fs::path& tsv) {
    std::ifstream in(tsv);
    if (!in) throw std::runtime_error("can't open " + tsv.string());

    std::vector<TableSpec> tables;
    std::string line;
    for (int n = 1; std::getline(in, line); ++n) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        const std::string where = tsv.string() + ":" + std::to_string(n) + ": ";
        const auto fields = Split(line, '\t');
        if (fields.size() != 3 || fields[0].empty() || fields[1].empty()) {
            throw std::runtime_error(where + "expected name<TAB>file<TAB>col:TYPE,...");
        }

        fs::path file{std::string(fields[1])};
        if (file.is_relative()) file = tsv.parent_path() / file;
        try {
            tables.push_back({std::string(fields[0]), std::move(file), ParseColumns(fields[2])});
        } catch (const std::runtime_error& e) {
            throw std::runtime_error(where + e.what());
        }
    }

    if (tables.empty()) throw std::runtime_error("no tables in " + tsv.string());
    return tables;
}

// Loads every table, naming the table and file in any error.
void LoadTables(Database& db, const std::vector<TableSpec>& tables) {
    for (const auto& table : tables) {
        if (!fs::is_regular_file(table.file)) {
            throw std::runtime_error("can't read " + table.file.string() + " for table " + table.name);
        }
        try {
            db.LoadTable(table.name, table.schema, table.file.string());
        } catch (const std::exception& e) {
            throw std::runtime_error("loading " + table.name + " from " + table.file.string() + ": " + e.what());
        }
    }
}

// The ready line, with each table's row count so a short load shows.
[[nodiscard]] std::string ReadyLine(Database& db, const std::vector<TableSpec>& tables, double load_ms) {
    std::string out = R"({"ready":true,"tables":)";
    AppendNumber(out, tables.size());
    out += R"(,"load_ms":)";
    AppendNumber(out, load_ms);
    out += R"(,"rows":{)";
    for (size_t i = 0; i < tables.size(); ++i) {
        if (i > 0) out += ',';
        AppendString(out, tables[i].name);
        out += ':';
        AppendNumber(out, db.GetCatalog()->GetTable(tables[i].name)->stats.row_count);
    }
    out += "}}";
    return out;
}

// Removes the file when it goes out of scope.
struct RemoveOnExit {
    fs::path path;
    ~RemoveOnExit() {
        std::error_code ec;
        fs::remove(path, ec);
    }
};

// Prints why the bridge can't go on and returns its exit code.
[[nodiscard]] int Fail(std::string_view message) {
    std::cerr << "analyst_bridge: " << message << std::endl;
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);

    Args args;
    try {
        args = ParseArgs(argc, argv);
    } catch (const std::exception& e) {
        return Fail(e.what());
    }

    // Declared before the database, so the file is removed after the database closes
    const RemoveOnExit remove_db{args.db_file};
    std::unique_ptr<Database> db;
    try {
        std::error_code ec;
        fs::remove(args.db_file, ec);

        const auto tables = ReadTables(args.tables);
        const auto start = Clock::now();
        db = std::make_unique<Database>(args.db_file.string(), args.pool_size);
        LoadTables(*db, tables);
        std::cout << ReadyLine(*db, tables, MsSince(start)) << std::endl;
    } catch (const std::exception& e) {
        return Fail(e.what());
    } catch (...) {
        return Fail("unknown error while loading");
    }

    ServeRequests(*db);
    return 0;
}
