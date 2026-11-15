#!/usr/bin/env python3
# DuckDB and Postgres TPC-H baseline runners
from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, Iterator, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "bench" / "tpch_data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "bench" / "results"


# Load a script next to this one by path, once per process.
def load_sibling(name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[name]
        raise
    return module


_sqlite_runner = load_sibling("run_sqlite_benchmarks")
ALL_SCALE_FACTORS: List[str] = _sqlite_runner.ALL_SCALE_FACTORS
KNOWN_SCALE_FACTORS: List[str] = _sqlite_runner.KNOWN_SCALE_FACTORS
QUERIES: Dict[str, str] = _sqlite_runner.QUERIES
QUERY_HEADERS: Dict[str, List[str]] = _sqlite_runner.QUERY_HEADERS
TABLE_SCHEMAS: Dict[str, List[Tuple[str, str]]] = _sqlite_runner.TABLE_SCHEMAS
TABLE_LOAD_ORDER: List[str] = _sqlite_runner.TABLE_LOAD_ORDER
export_query_results = _sqlite_runner.export_query_results

# The SQLite runner's order
QUERY_ORDER = ["Q1", "Q6", "Q3", "Q5", "Q10", "Q12", "Q14", "Q19"]

# SQLite keeps dates as TEXT; these are the ones the queries compare as dates.
DATE_COLUMNS = {"l_shipdate", "l_commitdate", "l_receiptdate", "o_orderdate"}

# The 8 indexes of tpch::IndexDescs() in tpch_workload.hpp: (index, table, column)
INDEX_DEFS_8: List[Tuple[str, str, str]] = [
    ("idx_orders_custkey", "orders", "o_custkey"),
    ("idx_lineitem_orderkey", "lineitem", "l_orderkey"),
    ("idx_customer_custkey", "customer", "c_custkey"),
    ("idx_supplier_nationkey", "supplier", "s_nationkey"),
    ("idx_nation_nationkey", "nation", "n_nationkey"),
    ("idx_lineitem_shipdate", "lineitem", "l_shipdate"),
    ("idx_orders_orderdate", "orders", "o_orderdate"),
    ("idx_part_partkey", "part", "p_partkey"),
]

# Per-engine query texts that differ from the SQLite runner's, each with its reason.
QUERY_OVERRIDES: Dict[str, Dict[str, str]] = {
    "duckdb": {},
    "postgres": {},
}


# A table's columns with typed SQL types instead of SQLite's REAL and TEXT.
def typed_schema(table: str) -> List[Tuple[str, str]]:
    typed = []
    for name, sqlite_type in TABLE_SCHEMAS[table]:
        if sqlite_type == "INTEGER":
            typed.append((name, "INTEGER"))
        elif sqlite_type == "REAL":
            typed.append((name, "DECIMAL(15,2)"))
        elif name in DATE_COLUMNS:
            typed.append((name, "DATE"))
        else:
            typed.append((name, "VARCHAR"))
    return typed


def query_text(engine: str, tag: str) -> str:
    return QUERY_OVERRIDES.get(engine, {}).get(tag, QUERIES[tag])


# Each .tbl line without its line ending and one trailing "|"; blank lines skipped.
def stripped_lines(tbl_path: Path) -> Iterator[str]:
    with open(tbl_path, "r", newline="") as f:
        for line in f:
            line = line.removesuffix("\n").removesuffix("\r").removesuffix("|")
            if line.strip():
                yield line


# Non-blank lines per .tbl file, for the tables the queries use.
def tbl_row_counts(data_dir: Path) -> Dict[str, int]:
    counts = {}
    for table in TABLE_LOAD_ORDER:
        path = Path(data_dir) / f"{table}.tbl"
        if path.exists():
            with open(path, "rb") as f:
                counts[table] = sum(1 for line in f if line.strip())
    return counts


# `warmup` untimed calls, then `runs` timed calls (execute + fetch).
# Returns the latencies in ms and the last call's rows.
def time_query(run: Callable[[], list], runs: int, warmup: int = 1) -> Tuple[List[float], list]:
    for _ in range(warmup):
        run()
    latencies = []
    rows: list = []
    for _ in range(runs):
        t0 = time.perf_counter()
        rows = run()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies, rows


def write_engine_latencies(rows: List[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sf", "query", "config", "run", "latency_ms"])
        for r in rows:
            writer.writerow([r["sf"], r["query"], r["config"], r["run"], f"{r['latency_ms']:.4f}"])


def write_info(info: Dict[str, str], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key, value in info.items():
            writer.writerow([key, value])


# Reuse a loaded database only if its marker equals today's row counts.
def marker_matches(stored: Dict[str, int], expected: Dict[str, int]) -> bool:
    return bool(expected) and stored == expected


# The flags every engine runner takes.
def add_baseline_args(parser: argparse.ArgumentParser, configs: Dict[str, List[str]]) -> None:
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="folder holding sf<SF>/*.tbl (default: bench/tpch_data)")
    parser.add_argument("--sf", action="extend", nargs="+", default=None, help="scale factors, space or comma separated: 0.01, 0.1, 1.0, 10, or 'all' (0.01, 0.1, 1.0; the default)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="where results go (default: bench/results)")
    parser.add_argument("--runs", type=int, default=5, help="timed runs per query (default: 5)")
    parser.add_argument("--warmup", type=int, default=1, help="untimed runs per query before the timed ones (default: 1)")
    parser.add_argument("--configs", action="extend", nargs="+", default=None, help=f"configs to run, space or comma separated (default: {','.join(configs)})")


def _split_commas(names: List[str]) -> List[str]:
    return [part for name in names for part in name.split(",") if part]


def resolve_scale_factors(names: Optional[List[str]]) -> Optional[List[str]]:
    names = _split_commas(names or [])
    if not names or "all" in names:
        return list(ALL_SCALE_FACTORS)
    
    for sf in names:
        if sf not in KNOWN_SCALE_FACTORS:
            print(f"Error: unknown scale factor '{sf}'. Choose from {KNOWN_SCALE_FACTORS} or 'all'.")
            return None
    return list(dict.fromkeys(names))


def resolve_configs(names: Optional[List[str]], configs: Dict[str, List[str]]) -> Optional[List[str]]:
    if not names:
        return list(configs)
    
    chosen = _split_commas(names)
    for config in chosen:
        if config not in configs:
            print(f"Error: unknown config '{config}'. Choose from {list(configs)}.")
            return None
    return list(dict.fromkeys(chosen))


# The checked --runs, --warmup, scale factors and configs; None after printing what's wrong.
def resolve_run_args(args: argparse.Namespace, configs: Dict[str, List[str]]) -> Optional[Tuple[List[str], List[str]]]:
    if args.runs < 1:
        print("Error: --runs must be at least 1.")
        return None
    if args.warmup < 0:
        print("Error: --warmup can't be negative.")
        return None
    
    scale_factors = resolve_scale_factors(args.sf)
    chosen = resolve_configs(args.configs, configs)
    if scale_factors is None or chosen is None:
        return None
    return scale_factors, chosen


# Reuse a loaded database when its marker matches the data, else load it.
# detail follows the Loaded line. Returns the load seconds, or None when reused.
def ensure_loaded(label: str, sf_data_dir: Path, reload: bool, read_marker: Callable[[], Dict[str, int]], load: Callable[[Dict[str, int]], float], detail: str = "") -> Optional[float]:
    expected = tbl_row_counts(sf_data_dir)
    if not reload and marker_matches(read_marker(), expected):
        print(f"  Reusing {label} (row counts match)")
        return None
    
    print(f"  Loading {sf_data_dir} into {label} ...")
    seconds = load(expected)
    print(f"  Loaded {sum(expected.values()):,} rows in {seconds:.2f} s{detail}")
    return seconds


# Time the 8 queries through execute(sql) and write each one's last rows to results_dir.
def run_queries(engine: str, execute: Callable[[str], list], sf: str, config: str, runs: int, results_dir: Path, warmup: int = 1) -> List[dict]:
    rows: List[dict] = []
    print(f"\n  --- Config: {config} (SF={sf}) ---")
    for tag in QUERY_ORDER:
        sql = query_text(engine, tag)
        latencies, result = time_query(lambda: execute(sql), runs, warmup)
        print(f"  {tag}: median={statistics.median(latencies):.2f} ms "
              f"(min={min(latencies):.2f}, max={max(latencies):.2f})")
        
        rows.extend({"sf": sf, "query": tag, "config": config, "run": i, "latency_ms": lat} for i, lat in enumerate(latencies, start=1))
        export_query_results(result, QUERY_HEADERS[tag], str(Path(results_dir) / f"{tag}_sf{sf}.csv"))
    return rows


# Every SF and config, then {engine}_latencies.csv and {engine}_info.csv. Returns the exit code.
# ensure(sf, sf_data_dir) gives the load seconds or None; run(sf, config) gives (rows, settings).
def run_all(engine: str, data_dir: Path, output_dir: Path, scale_factors: List[str], configs: List[str], info: Dict[str, str], ensure: Callable[[str, Path], Optional[float]], run: Callable[[str, str], Tuple[List[dict], Dict[str, str]]]) -> int:
    all_rows: List[dict] = []
    for sf in scale_factors:
        sf_data_dir = Path(data_dir) / f"sf{sf}"
        if not sf_data_dir.is_dir():
            print(f"  Data directory {sf_data_dir} not found, skipping SF={sf}")
            continue
        
        print(f"\n{'=' * 60}\n  Scale Factor: {sf}\n{'=' * 60}")
        seconds = ensure(sf, sf_data_dir)
        if seconds is not None:
            info[f"load_seconds_sf{sf}"] = f"{seconds:.3f}"
            
        for config in configs:
            rows, settings = run(sf, config)
            all_rows.extend(rows)
            info.update(settings)

    if not all_rows:
        print("\nNo benchmarks ran. Check that data files exist.")
        return 1

    output_dir = Path(output_dir)
    latencies_path = output_dir / f"{engine}_latencies.csv"
    info_path = output_dir / f"{engine}_info.csv"
    write_engine_latencies(all_rows, latencies_path)
    write_info(info, info_path)
    print(f"\nWrote {latencies_path}, {info_path} and {output_dir / f'{engine}_results'}/")
    return 0
