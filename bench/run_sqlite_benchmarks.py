#!/usr/bin/env python3

import argparse
import csv
import os
import sqlite3
import statistics
import sys
import time
from typing import Dict, List, Tuple


TABLE_SCHEMAS: Dict[str, List[Tuple[str, str]]] = {
    "lineitem": [
        ("l_orderkey", "INTEGER"),
        ("l_partkey", "INTEGER"),
        ("l_suppkey", "INTEGER"),
        ("l_linenumber", "INTEGER"),
        ("l_quantity", "REAL"),
        ("l_extendedprice", "REAL"),
        ("l_discount", "REAL"),
        ("l_tax", "REAL"),
        ("l_returnflag", "TEXT"),
        ("l_linestatus", "TEXT"),
        ("l_shipdate", "TEXT"),
        ("l_commitdate", "TEXT"),
        ("l_receiptdate", "TEXT"),
        ("l_shipinstruct", "TEXT"),
        ("l_shipmode", "TEXT"),
        ("l_comment", "TEXT"),
    ],
    "orders": [
        ("o_orderkey", "INTEGER"),
        ("o_custkey", "INTEGER"),
        ("o_orderstatus", "TEXT"),
        ("o_totalprice", "REAL"),
        ("o_orderdate", "TEXT"),
        ("o_orderpriority", "TEXT"),
        ("o_clerk", "TEXT"),
        ("o_shippriority", "INTEGER"),
        ("o_comment", "TEXT"),
    ],
    "customer": [
        ("c_custkey", "INTEGER"),
        ("c_name", "TEXT"),
        ("c_address", "TEXT"),
        ("c_nationkey", "INTEGER"),
        ("c_phone", "TEXT"),
        ("c_acctbal", "REAL"),
        ("c_mktsegment", "TEXT"),
        ("c_comment", "TEXT"),
    ],
    "supplier": [
        ("s_suppkey", "INTEGER"),
        ("s_name", "TEXT"),
        ("s_address", "TEXT"),
        ("s_nationkey", "INTEGER"),
        ("s_phone", "TEXT"),
        ("s_acctbal", "REAL"),
        ("s_comment", "TEXT"),
    ],
    "nation": [
        ("n_nationkey", "INTEGER"),
        ("n_name", "TEXT"),
        ("n_regionkey", "INTEGER"),
        ("n_comment", "TEXT"),
    ],
    "region": [
        ("r_regionkey", "INTEGER"),
        ("r_name", "TEXT"),
        ("r_comment", "TEXT"),
    ],
    "part": [
        ("p_partkey", "INTEGER"),
        ("p_name", "TEXT"),
        ("p_mfgr", "TEXT"),
        ("p_brand", "TEXT"),
        ("p_type", "TEXT"),
        ("p_size", "INTEGER"),
        ("p_container", "TEXT"),
        ("p_retailprice", "REAL"),
        ("p_comment", "TEXT"),
    ],
}

# Load order matters: region/nation before supplier/customer before orders/lineitem
TABLE_LOAD_ORDER = ["region", "nation", "part", "supplier", "customer", "orders", "lineitem"]

INDEX_DEFS = [
    ("idx_orders_custkey", "orders", "o_custkey"),
    ("idx_lineitem_orderkey", "lineitem", "l_orderkey"),
    ("idx_customer_custkey", "customer", "c_custkey"),
    ("idx_supplier_nationkey", "supplier", "s_nationkey"),
    ("idx_nation_nationkey", "nation", "n_nationkey"),
    ("idx_part_partkey", "part", "p_partkey"),
]



QUERIES: Dict[str, str] = {
    "Q1": (
        "SELECT l_returnflag, l_linestatus, SUM(l_quantity), SUM(l_extendedprice), "
        "SUM(l_discount), COUNT(*) "
        "FROM lineitem "
        "WHERE l_shipdate <= '1998-09-02' "
        "GROUP BY l_returnflag, l_linestatus "
        "ORDER BY l_returnflag, l_linestatus"
    ),
    "Q6": (
        "SELECT SUM(l_extendedprice * l_discount) "
        "FROM lineitem "
        "WHERE l_shipdate >= '1994-01-01' AND l_shipdate < '1995-01-01' "
        "AND l_discount >= 0.05 AND l_discount <= 0.07 AND l_quantity < 24"
    ),
    "Q3": (
        "SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)), o_orderdate, o_shippriority "
        "FROM customer c "
        "JOIN orders o ON c.c_custkey = o.o_custkey "
        "JOIN lineitem l ON l.l_orderkey = o.o_orderkey "
        "WHERE c.c_mktsegment = 'BUILDING' "
        "AND o.o_orderdate < '1995-03-15' "
        "AND l.l_shipdate > '1995-03-15' "
        "GROUP BY l_orderkey, o_orderdate, o_shippriority "
        "ORDER BY SUM(l_extendedprice * (1 - l_discount)) DESC "
        "LIMIT 10"
    ),
    "Q5": (
        "SELECT n_name, SUM(l_extendedprice * (1 - l_discount)) "
        "FROM customer c "
        "JOIN orders o ON c.c_custkey = o.o_custkey "
        "JOIN lineitem l ON l.l_orderkey = o.o_orderkey "
        "JOIN supplier s ON l.l_suppkey = s.s_suppkey "
        "JOIN nation n ON s.s_nationkey = n.n_nationkey "
        "WHERE o.o_orderdate >= '1994-01-01' AND o.o_orderdate < '1995-01-01' "
        "GROUP BY n_name "
        "ORDER BY SUM(l_extendedprice * (1 - l_discount)) DESC"
    ),
    "Q10": (
        "SELECT c_custkey, c_name, SUM(l_extendedprice * (1 - l_discount)), "
        "c_acctbal, n_name, c_address, c_phone, c_comment "
        "FROM customer, orders, lineitem, nation "
        "WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey "
        "AND o_orderdate >= '1993-10-01' AND o_orderdate < '1994-01-01' "
        "AND l_returnflag = 'R' AND c_nationkey = n_nationkey "
        "GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment "
        "ORDER BY SUM(l_extendedprice * (1 - l_discount)) DESC "
        "LIMIT 20"
    ),
    "Q12": (
        "SELECT l_shipmode, "
        "SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) "
        "FROM orders, lineitem "
        "WHERE o_orderkey = l_orderkey "
        "AND l_shipmode IN ('MAIL', 'SHIP') "
        "AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate "
        "AND l_receiptdate >= '1994-01-01' AND l_receiptdate < '1995-01-01' "
        "GROUP BY l_shipmode "
        "ORDER BY l_shipmode"
    ),
    "Q14": (
        "SELECT 100.00 * SUM(CASE WHEN p_type LIKE 'PROMO%' "
        "THEN l_extendedprice * (1 - l_discount) ELSE 0 END) "
        "/ SUM(l_extendedprice * (1 - l_discount)) "
        "FROM lineitem, part "
        "WHERE l_partkey = p_partkey "
        "AND l_shipdate >= '1995-09-01' AND l_shipdate < '1995-10-01'"
    ),
    "Q19": (
        "SELECT SUM(l_extendedprice * (1 - l_discount)) "
        "FROM lineitem, part "
        "WHERE p_partkey = l_partkey "
        "AND ((p_brand = 'Brand#12' "
        "AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG') "
        "AND l_quantity >= 1 AND l_quantity <= 11 "
        "AND p_size BETWEEN 1 AND 5 "
        "AND l_shipmode IN ('AIR', 'AIR REG') "
        "AND l_shipinstruct = 'DELIVER IN PERSON') "
        "OR (p_brand = 'Brand#23' "
        "AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK') "
        "AND l_quantity >= 10 AND l_quantity <= 20 "
        "AND p_size BETWEEN 1 AND 10 "
        "AND l_shipmode IN ('AIR', 'AIR REG') "
        "AND l_shipinstruct = 'DELIVER IN PERSON') "
        "OR (p_brand = 'Brand#34' "
        "AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG') "
        "AND l_quantity >= 20 AND l_quantity <= 30 "
        "AND p_size BETWEEN 1 AND 15 "
        "AND l_shipmode IN ('AIR', 'AIR REG') "
        "AND l_shipinstruct = 'DELIVER IN PERSON'))"
    ),
}

# Canonical column headers for result CSVs
QUERY_HEADERS: Dict[str, List[str]] = {
    "Q1": ["l_returnflag", "l_linestatus", "sum_quantity", "sum_extendedprice",
            "sum_discount", "count_star"],
    "Q3": ["l_orderkey", "sum_disc_price", "o_orderdate", "o_shippriority"],
    "Q5": ["n_name", "sum_disc_price"],
    "Q6": ["sum_disc_price"],
    "Q10": ["c_custkey", "c_name", "sum_disc_price", "c_acctbal", "n_name",
            "c_address", "c_phone", "c_comment"],
    "Q12": ["l_shipmode", "high_line_count", "low_line_count"],
    "Q14": ["promo_revenue"],
    "Q19": ["revenue"],
}




def _parse_tbl_line(line: str, schema: List[Tuple[str, str]]) -> List:
    stripped = line.rstrip("\n").rstrip("|")
    fields = stripped.split("|")
    row = []
    for (_, col_type), raw in zip(schema, fields):
        if col_type == "INTEGER":
            row.append(int(raw))
        elif col_type == "REAL":
            row.append(float(raw))
        else:  # TEXT
            row.append(raw)
    return row


def load_table(conn: sqlite3.Connection, table_name: str, tbl_path: str) -> int:
    schema = TABLE_SCHEMAS[table_name]
    col_defs = ", ".join(f"{name} {typ}" for name, typ in schema)
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({col_defs})")

    placeholders = ", ".join("?" for _ in schema)
    insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"

    rows = []
    with open(tbl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(_parse_tbl_line(line, schema))

    conn.executemany(insert_sql, rows)
    conn.commit()
    return len(rows)


def create_indexes(conn: sqlite3.Connection) -> None:
    for idx_name, table_name, col_name in INDEX_DEFS:
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name}({col_name})"
        )
    conn.commit()


def setup_database(conn: sqlite3.Connection, data_dir: str) -> Dict[str, int]:
    row_counts = {}
    for table_name in TABLE_LOAD_ORDER:
        tbl_path = os.path.join(data_dir, f"{table_name}.tbl")
        if not os.path.exists(tbl_path):
            print(f"  WARNING: {tbl_path} not found, skipping {table_name}")
            continue
        count = load_table(conn, table_name, tbl_path)
        row_counts[table_name] = count
    create_indexes(conn)
    return row_counts




def run_query(conn: sqlite3.Connection, sql: str) -> List[tuple]:
    cursor = conn.execute(sql)
    return cursor.fetchall()


def benchmark_query(conn: sqlite3.Connection, query_name: str, sql: str, num_runs: int) -> Tuple[List[float], List[tuple]]:
    # Warmup
    _ = run_query(conn, sql)

    latencies = []
    result = None
    for _ in range(num_runs):
        t0 = time.perf_counter()
        result = run_query(conn, sql)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    return latencies, result




def export_query_results(rows: List[tuple], headers: List[str], output_path: str) -> None:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
        
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def write_latencies_csv(all_latencies: List[Dict], output_path: str) -> None:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
        
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sf", "query", "mode", "run", "latency_ms"])
        for entry in all_latencies:
            writer.writerow([
                entry["sf"], entry["query"], entry["mode"],
                entry["run"], f"{entry['latency_ms']:.4f}",
            ])




def run_benchmarks_for_sf(sf: str, data_dir: str, output_dir: str, num_runs: int) -> List[Dict]:
    sf_data_dir = os.path.join(data_dir, f"sf{sf}")
    if not os.path.isdir(sf_data_dir):
        print(f"  Data directory {sf_data_dir} not found, skipping SF={sf}")
        return []

    results_dir = os.path.join(output_dir, "sqlite_results")
    all_latencies = []
    query_order = ["Q1", "Q6", "Q3", "Q5", "Q10", "Q12", "Q14", "Q19"]

    for mode in ["memory", "disk"]:
        print(f"\n  --- Mode: {mode} (SF={sf}) ---")

        # Open connection
        if mode == "memory":
            db_path = ":memory:"
        else:
            db_path = f"/tmp/tpch_sqlite_sf{sf}.db"
            # Remove stale file so we start fresh
            if os.path.exists(db_path):
                os.remove(db_path)

        conn = sqlite3.connect(db_path)
        # Enable WAL for on-disk mode (typical production config)
        if mode == "disk":
            conn.execute("PRAGMA journal_mode=WAL")

        # Load data
        print(f"  Loading tables from {sf_data_dir} ...")
        t0 = time.perf_counter()
        row_counts = setup_database(conn, sf_data_dir)
        load_ms = (time.perf_counter() - t0) * 1000.0
        total_rows = sum(row_counts.values())
        print(f"  Loaded {total_rows:,} rows across {len(row_counts)} tables ({load_ms:.0f} ms)")
        for tbl, cnt in row_counts.items():
            print(f"    {tbl}: {cnt:,}")

        # Run queries
        for qname in query_order:
            sql = QUERIES[qname]
            print(f"  {qname} ({num_runs} runs) ... ", end="", flush=True)

            latencies, result_rows = benchmark_query(conn, qname, sql, num_runs)
            median = statistics.median(latencies)
            print(f"median={median:.2f} ms  (min={min(latencies):.2f}, max={max(latencies):.2f})")

            # Record per-run latencies
            for i, lat in enumerate(latencies, start=1):
                all_latencies.append({
                    "sf": sf,
                    "query": qname,
                    "mode": mode,
                    "run": i,
                    "latency_ms": lat,
                })

            # Export results once per SF — memory mode runs first in the loop,
            # so we export here to avoid writing duplicate CSVs from disk mode.
            if mode == "memory" and result_rows is not None:
                csv_path = os.path.join(results_dir, f"{qname}_sf{sf}.csv")
                export_query_results(result_rows, QUERY_HEADERS[qname], csv_path)

        conn.close()

        # Clean up on-disk file and WAL/SHM sidecars
        if mode == "disk":
            for suffix in ["", "-wal", "-shm"]:
                path = db_path + suffix
                if os.path.exists(path):
                    os.remove(path)

    return all_latencies


def print_summary(all_latencies: List[Dict]) -> None:
    groups: Dict[tuple, List[float]] = {}
    for entry in all_latencies:
        key = (entry["sf"], entry["query"], entry["mode"])
        groups.setdefault(key, []).append(entry["latency_ms"])

    print("\n" + "=" * 70)
    print(f"{'SF':<8} {'Query':<6} {'Mode':<8} {'Median (ms)':>12} {'Min':>10} {'Max':>10}")
    print("-" * 70)
    for (sf, query, mode), latencies in sorted(groups.items()):
        median = statistics.median(latencies)
        print(f"{sf:<8} {query:<6} {mode:<8} {median:>12.2f} {min(latencies):>10.2f} {max(latencies):>10.2f}")
    print("=" * 70)


#cli args and commands

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TPC-H benchmarks on SQLite for ShilmanDB comparison."
    )
    parser.add_argument(
        "--data-dir",
        default="bench/tpch_data",
        help="Base directory containing sf0.01/, sf0.1/, sf1.0/ (default: bench/tpch_data)",
    )
    parser.add_argument(
        "--sf",
        nargs="+",
        default=["all"],
        help="Scale factors to run: 0.01, 0.1, 1.0, or 'all' (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="bench/results",
        help="Directory for output files (default: bench/results)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of timed runs per query (default: 5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_sfs = ["0.01", "0.1", "1.0"]
    if args.runs < 1:
        print("Error: --runs must be at least 1.")
        sys.exit(1)

    if "all" in args.sf:
        scale_factors = all_sfs
    else:
        scale_factors = []
        for sf in args.sf:
            if sf not in all_sfs:
                print(f"Error: unknown scale factor '{sf}'. Choose from {all_sfs} or 'all'.")
                sys.exit(1)
            scale_factors.append(sf)

    print(f"SQLite TPC-H Benchmark")
    print(f"  Scale factors: {scale_factors}")
    print(f"  Runs per query: {args.runs}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  SQLite version: {sqlite3.sqlite_version}")

    all_latencies = []
    for sf in scale_factors:
        print(f"\n{'='*60}")
        print(f"  Scale Factor: {sf}")
        print(f"{'='*60}")
        latencies = run_benchmarks_for_sf(sf, args.data_dir, args.output_dir, args.runs)
        all_latencies.extend(latencies)

    if not all_latencies:
        print("\nNo benchmarks ran. Check that data files exist.")
        sys.exit(1)

    # Write latencies CSV
    latencies_path = os.path.join(args.output_dir, "sqlite_latencies.csv")
    write_latencies_csv(all_latencies, latencies_path)
    print(f"\nLatencies written to {latencies_path}")

    # Print summary
    print_summary(all_latencies)

    # List exported result files
    results_dir = os.path.join(args.output_dir, "sqlite_results")
    if os.path.isdir(results_dir):
        files = sorted(os.listdir(results_dir))
        print(f"\nQuery result CSVs in {results_dir}/:")
        for fname in files:
            fpath = os.path.join(results_dir, fname)
            size = os.path.getsize(fpath)
            print(f"  {fname} ({size:,} bytes)")


if __name__ == "__main__":
    main()
