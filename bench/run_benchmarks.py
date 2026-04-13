#!/usr/bin/env python3

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple



ALL_SCALE_FACTORS = ["0.01", "0.1", "1.0"]
QUERIES = ["Q1", "Q3", "Q5", "Q6"]

SHILMANDB_CONFIGS: Dict[str, Dict[str, str]] = {
    "lru_heuristic": {},
    "lru_learned_join": {
        "--use-learned-join": "",
        "--join-model-path": "ml/join_optimizer/models/join_order_model.pt",
    },
    "learned_eviction_heuristic": {
        "--use-learned-eviction": "",
        "--eviction-model-path": "ml/eviction_policy/models/eviction_model.pt",
    },
    "learned_all": {
        "--use-learned-join": "",
        "--join-model-path": "ml/join_optimizer/models/join_order_model.pt",
        "--use-learned-eviction": "",
        "--eviction-model-path": "ml/eviction_policy/models/eviction_model.pt",
    },
}

CONFIG_GROUPS = {
    "baseline": ["lru_heuristic"],
    "learned": ["lru_learned_join", "learned_eviction_heuristic", "learned_all"],
    "all": list(SHILMANDB_CONFIGS.keys()),
}

POOL_SIZES: Dict[str, int] = {
    "0.01": 128,
    "0.1": 512,
    "1.0": 2048,
}



def elapsed_str(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds) // 60
    secs = seconds - minutes * 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def section_header(title: str, width: int = 70) -> None:
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def sub_header(title: str, width: int = 60) -> None:
    print(f"\n  --- {title} {'-' * max(1, width - len(title) - 7)}")




def check_tpch_data(data_dir: str, sf: str) -> bool:
    sf_dir = os.path.join(data_dir, f"sf{sf}")
    marker = os.path.join(sf_dir, "lineitem.tbl")
    return os.path.isfile(marker)


def generate_tpch_data(data_dir: str) -> bool:
    script = os.path.join("bench", "generate_tpch_data.sh")
    if not os.path.isfile(script):
        print(f"    ERROR: Data generation script not found: {script}")
        return False

    print(f"    Generating TPC-H data (all scale factors) ...")
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            ["bash", script],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        print(f"    ERROR: Data generation timed out after 600s")
        return False
    except OSError as e:
        print(f"    ERROR: Could not run data generation script: {e}")
        return False

    if result.returncode != 0:
        print(f"    ERROR: Data generation failed (exit {result.returncode})")
        if result.stderr:
            for line in result.stderr.strip().split("\n")[-5:]:
                print(f"      | {line}")
        return False

    duration = time.perf_counter() - t0
    print(f"    Data generation completed ({elapsed_str(duration)})")
    return True




def build_load_tpch_cmd(binary: str, sf: str, config_name: str, data_dir: str, results_dir: str, db_file: str) -> List[str]:
    pool_size = POOL_SIZES[sf]
    sf_data_dir = os.path.join(data_dir, f"sf{sf}")

    cmd = [
        binary,
        "--sf", sf,
        "--db-file", db_file,
        "--data-dir", sf_data_dir,
        "--pool-size", str(pool_size),
        "--results-dir", results_dir,
    ]

    for flag, value in SHILMANDB_CONFIGS[config_name].items():
        cmd.append(flag)
        if value:
            cmd.append(value)

    return cmd



def parse_load_tpch_latencies(latencies_path: str) -> Dict[str, Tuple[float, int]]:
    result = {}
    if not os.path.isfile(latencies_path):
        return result

    with open(latencies_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                query = row["query"]
                latency = float(row["latency_ms"])
                rows = int(row["rows"])
                result[query] = (latency, rows)
            except (KeyError, ValueError, TypeError) as e:
                print(f"    WARNING: Skipping malformed latency row: {e}")
                continue

    return result


def run_load_tpch(binary: str, sf: str, config_name: str, data_dir: str, results_dir: str, db_file: str, timeout_seconds: int = 3600) -> Tuple[bool, Optional[str]]:
    cmd = build_load_tpch_cmd(binary, sf, config_name, data_dir, results_dir, db_file)

    os.makedirs(results_dir, exist_ok=True)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout_seconds}s"
    except OSError as e:
        return False, f"Failed to execute binary: {e}"

    if proc.returncode != 0:
        stderr_tail = proc.stderr[-1500:] if proc.stderr else "(empty)"
        stdout_tail = proc.stdout[-500:] if proc.stdout else "(empty)"
        return False, (
            f"Exit code {proc.returncode}\n"
            f"  stderr: {stderr_tail}\n"
            f"  stdout: {stdout_tail}"
        )

    return True, None




def benchmark_shilmandb_config(binary: str, sf: str, config_name: str, data_dir: str, output_dir: str, num_runs: int, purge: bool) -> Tuple[bool, Optional[str], Dict[str, List[float]], Dict[str, int]]:
    per_query_latencies: Dict[str, List[float]] = {}
    per_query_rows: Dict[str, int] = {}


    warmup_results = os.path.join(output_dir, "shilmandb_results", config_name)
    warmup_db = f"/tmp/shilmandb_warmup_{config_name}_sf{sf}.db"


    if os.path.exists(warmup_db):
        os.remove(warmup_db)

    print(f"    Run 0 (warmup) ... ", end="", flush=True)
    t0 = time.perf_counter()
    success, error = run_load_tpch(
        binary, sf, config_name, data_dir, warmup_results, warmup_db,
    )
    warmup_time = time.perf_counter() - t0

    # Clean up warmup db
    _safe_remove(warmup_db)

    if not success:
        print(f"FAILED ({elapsed_str(warmup_time)})")
        return False, f"Warmup failed: {error}", {}, {}

    print(f"done ({elapsed_str(warmup_time)})")

    # Rename warmup query CSVs to include SF (Q1.csv -> Q1_sf0.01.csv)
    for tag in QUERIES:
        src = os.path.join(warmup_results, f"{tag}.csv")
        dst = os.path.join(warmup_results, f"{tag}_sf{sf}.csv")
        if os.path.isfile(src):
            shutil.move(src, dst)

    # Parse warmup latencies for row counts (timing discarded)
    warmup_lat_path = os.path.join(warmup_results, "latencies.csv")
    warmup_lat = parse_load_tpch_latencies(warmup_lat_path)
    for query, (_, rows) in warmup_lat.items():
        per_query_rows[query] = rows
    # Remove warmup latencies file (we don't want it in permanent results)
    _safe_remove(warmup_lat_path)

    # ---- Timed runs (1..N) ----
    for run_idx in range(1, num_runs + 1):
        if purge:
            _run_purge()

        with tempfile.TemporaryDirectory(prefix=f"shilmandb_run{run_idx}_") as tmp_dir:
            run_db = os.path.join(tmp_dir, f"shilmandb_{config_name}_sf{sf}.db")

            print(f"    Run {run_idx}/{num_runs} ... ", end="", flush=True)
            t0 = time.perf_counter()
            success, error = run_load_tpch(
                binary, sf, config_name, data_dir, tmp_dir, run_db,
            )
            run_time = time.perf_counter() - t0

            if not success:
                print(f"FAILED ({elapsed_str(run_time)})")
                print(f"      Error: {error.split(chr(10))[0][:100]}")
                # Continue with remaining runs rather than aborting
                continue

            # Parse latencies from this run
            lat_path = os.path.join(tmp_dir, "latencies.csv")
            run_latencies = parse_load_tpch_latencies(lat_path)

            query_summary = []
            for query, (latency, rows) in run_latencies.items():
                per_query_latencies.setdefault(query, []).append(latency)
                per_query_rows.setdefault(query, rows)
                query_summary.append(f"{query}={latency:.1f}ms")

            print(f"done ({elapsed_str(run_time)})  [{', '.join(query_summary)}]")
            # tmp_dir auto-cleaned by TemporaryDirectory context manager

    if not per_query_latencies:
        return False, "All timed runs failed", {}, {}

    return True, None, per_query_latencies, per_query_rows




def run_sqlite_benchmarks(data_dir: str, output_dir: str, num_runs: int, scale_factors: List[str]) -> Tuple[bool, Optional[str]]:
    script = os.path.join("bench", "run_sqlite_benchmarks.py")
    if not os.path.isfile(script):
        return False, f"SQLite benchmark script not found: {script}"

    cmd = [
        sys.executable, script,
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--runs", str(num_runs),
        "--sf",
    ] + scale_factors

    print(f"    Command: {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=False,  # Let SQLite script output directly to terminal
            timeout=7200,
        )
    except subprocess.TimeoutExpired:
        return False, "SQLite benchmarks timed out after 7200s"
    except OSError as e:
        return False, f"Failed to run SQLite script: {e}"

    if proc.returncode != 0:
        return False, f"SQLite benchmarks exited with code {proc.returncode}"

    return True, None




def run_correctness_comparison(sf: str, configs: List[str], output_dir: str) -> Tuple[bool, Optional[str]]:
    script = os.path.join("scripts", "compare_results.py")
    if not os.path.isfile(script):
        return False, f"Comparison script not found: {script}"

    shilmandb_dir = os.path.join(output_dir, "shilmandb_results")
    sqlite_dir = os.path.join(output_dir, "sqlite_results")

    cmd = [
        sys.executable, script,
        "--shilmandb-dir", shilmandb_dir,
        "--sqlite-dir", sqlite_dir,
        "--sf", sf,
        "--configs",
    ] + configs

    print(f"    Command: {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=False,  # Let comparison output directly to terminal
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return False, "Correctness comparison timed out"
    except OSError as e:
        return False, f"Failed to run comparison script: {e}"

    if proc.returncode != 0:
        return False, "Correctness comparison found mismatches (see output above)"

    return True, None




def parse_sqlite_latencies(output_dir: str, scale_factors: List[str]) -> List[Dict]:
    lat_path = os.path.join(output_dir, "sqlite_latencies.csv")
    if not os.path.isfile(lat_path):
        return []

    # Accumulate per-run latencies grouped by (sf, query, mode)
    groups: Dict[Tuple[str, str, str], List[float]] = {}
    with open(lat_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sf = row["sf"]
                if sf not in scale_factors:
                    continue
                key = (sf, row["query"], row["mode"])
                groups.setdefault(key, []).append(float(row["latency_ms"]))
            except (KeyError, ValueError, TypeError) as e:
                print(f"    WARNING: Skipping malformed SQLite latency row: {e}")
                continue

    # Count rows from result CSVs
    row_counts: Dict[Tuple[str, str], int] = {}
    sqlite_results_dir = os.path.join(output_dir, "sqlite_results")
    for sf in scale_factors:
        for query in QUERIES:
            csv_path = os.path.join(sqlite_results_dir, f"{query}_sf{sf}.csv")
            if os.path.isfile(csv_path):
                with open(csv_path, "r") as f:
                    # Count data rows (subtract header)
                    count = sum(1 for line in f if line.strip()) - 1
                    row_counts[(sf, query)] = max(0, count)

    records = []
    for (sf, query, mode), latencies in sorted(groups.items()):
        config_name = f"sqlite_{mode}"
        median = statistics.median(latencies)
        rows = row_counts.get((sf, query), 0)
        records.append({
            "sf": sf,
            "query": query,
            "config": config_name,
            "median_latency_ms": median,
            "rows": rows,
        })

    return records




def write_merged_latencies(records: List[Dict], output_path: str) -> None:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sf", "query", "config", "median_latency_ms", "rows"])
        for rec in sorted(records, key=lambda r: (r["sf"], r["query"], r["config"])):
            writer.writerow([
                rec["sf"],
                rec["query"],
                rec["config"],
                f"{rec['median_latency_ms']:.4f}",
                rec["rows"],
            ])




def _safe_remove(path: str) -> None:
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def _run_purge() -> None:
    try:
        result = subprocess.run(["sudo", "purge"], capture_output=True, timeout=30)
        if result.returncode != 0:
            print("    WARNING: 'sudo purge' failed -- cache may be warm")
    except (subprocess.TimeoutExpired, OSError):
        print("    WARNING: 'sudo purge' could not be executed -- cache may be warm")


def trigger_build() -> bool:
    print("  Building (Release + ML) ...")
    try:
        proc = subprocess.run(
            ["bash", "scripts/build.sh", "Release", "ON"],
            capture_output=False,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        print("  ERROR: Build timed out after 600s")
        return False
    except OSError as e:
        print(f"  ERROR: Could not run build script: {e}")
        return False

    if proc.returncode != 0:
        print("  ERROR: Build failed.")
        return False

    print("  Build succeeded.")
    return True




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Master benchmark orchestration for ShilmanDB vs SQLite.\n"
            "Runs multi-run benchmarks, correctness verification, and produces\n"
            "a merged latency CSV covering all configurations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python bench/run_benchmarks.py                              # all configs, all SFs\n"
            "  python bench/run_benchmarks.py --config baseline --sf 0.01  # quick baseline test\n"
            "  python bench/run_benchmarks.py --config learned --runs 10   # learned configs, 10 runs\n"
            "  python bench/run_benchmarks.py --purge --runs 5             # cold-start measurements\n"
        ),
    )
    parser.add_argument(
        "--config",
        choices=["baseline", "learned", "all"],
        default="all",
        help="ShilmanDB configs to benchmark (default: all)",
    )
    parser.add_argument(
        "--sf",
        nargs="+",
        default=["all"],
        help="Scale factors: 0.01, 0.1, 1.0, or 'all' (default: all)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of timed runs per config (default: 5)",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Trigger ML release build before running benchmarks",
    )
    parser.add_argument(
        "--binary",
        default="build/bench/load_tpch",
        help="Path to load_tpch binary (default: build/bench/load_tpch)",
    )
    parser.add_argument(
        "--data-dir",
        default="bench/tpch_data",
        help="TPC-H data directory (default: bench/tpch_data)",
    )
    parser.add_argument(
        "--output-dir",
        default="bench/results",
        help="Output directory (default: bench/results)",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Run 'sudo purge' between timed runs for cold-start measurements",
    )
    parser.add_argument(
        "--skip-sqlite",
        action="store_true",
        help="Skip SQLite benchmarks (use existing results if available)",
    )
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip correctness comparison step",
    )
    return parser.parse_args()




def main() -> None:
    args = parse_args()

    if args.runs < 1:
        print("Error: --runs must be at least 1.")
        sys.exit(1)

    # Resolve scale factors
    if "all" in args.sf:
        scale_factors = ALL_SCALE_FACTORS
    else:
        scale_factors = []
        for sf in args.sf:
            if sf not in ALL_SCALE_FACTORS:
                print(f"Error: unknown scale factor '{sf}'. Choose from {ALL_SCALE_FACTORS} or 'all'.")
                sys.exit(1)
            scale_factors.append(sf)
        # Deduplicate while preserving order
        scale_factors = list(dict.fromkeys(scale_factors))

    # Resolve ShilmanDB configs
    shilmandb_configs = CONFIG_GROUPS[args.config]

    overall_t0 = time.perf_counter()
    print("=" * 70)
    print("  ShilmanDB vs SQLite -- Master Benchmark Orchestration")
    print("=" * 70)
    print(f"  Timestamp:      {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Config group:   {args.config} -> {shilmandb_configs}")
    print(f"  Scale factors:  {scale_factors}")
    print(f"  Timed runs:     {args.runs}")
    print(f"  Binary:         {args.binary}")
    print(f"  Data dir:       {args.data_dir}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Purge caches:   {args.purge}")
    print(f"  Skip SQLite:    {args.skip_sqlite}")
    print(f"  Skip correct.:  {args.skip_correctness}")

    if args.build:
        section_header("BUILD")
        if not trigger_build():
            sys.exit(1)


    if not os.path.isfile(args.binary):
        print(f"\nError: binary not found at '{args.binary}'.")
        print("  Run with --build to build it, or pass --binary <path>.")
        sys.exit(1)


    shilmandb_results: List[Dict] = []  # merged latency records for ShilmanDB
    sqlite_ok = False
    correctness_results: List[Tuple[str, List[str], bool, str]] = []
    failures: List[str] = []

    section_header("DATA CHECK")
    valid_sfs: List[str] = []
    missing_sfs: List[str] = []
    for sf in scale_factors:
        if check_tpch_data(args.data_dir, sf):
            sf_dir = os.path.join(args.data_dir, f"sf{sf}")
            print(f"  SF={sf}: data found ({sf_dir})")
            valid_sfs.append(sf)
        else:
            print(f"  SF={sf}: data not found")
            missing_sfs.append(sf)

    # generate_tpch_data.sh always produces all SFs in one invocation
    if missing_sfs:
        print(f"  Missing SFs: {missing_sfs}. Running data generation ...")
        if generate_tpch_data(args.data_dir):
            for sf in missing_sfs:
                if check_tpch_data(args.data_dir, sf):
                    valid_sfs.append(sf)
                else:
                    msg = f"SF={sf}: still missing after generation, skipping"
                    print(f"  ERROR: {msg}")
                    failures.append(msg)
        else:
            for sf in missing_sfs:
                msg = f"SF={sf}: data generation failed, skipping"
                print(f"  ERROR: {msg}")
                failures.append(msg)

    if not valid_sfs:
        print("\nError: no valid scale factors with data available.")
        sys.exit(1)


    if not args.skip_sqlite:
        section_header(f"SQLITE BENCHMARKS (SFs: {valid_sfs}, {args.runs} runs)")
        sq_ok, sq_err = run_sqlite_benchmarks(
            args.data_dir, args.output_dir, args.runs, valid_sfs,
        )
        if sq_ok:
            sqlite_ok = True
            print(f"\n  SQLite benchmarks completed for all requested SFs.")
        else:
            msg = f"SQLite benchmarks failed: {sq_err}"
            print(f"\n  WARNING: {msg}")
            failures.append(msg)
    else:
        print("\n  Skipping SQLite benchmarks (--skip-sqlite).")
        sqlite_lat_path = os.path.join(args.output_dir, "sqlite_latencies.csv")
        if os.path.isfile(sqlite_lat_path):
            sqlite_ok = True
            print(f"  Using existing SQLite results from {sqlite_lat_path}")


    for sf in valid_sfs:
        section_header(f"SHILMANDB BENCHMARKS -- SF={sf}")

        configs_that_ran: List[str] = []

        for config_name in shilmandb_configs:
            sub_header(
                f"ShilmanDB [{config_name}] SF={sf}, "
                f"pool={POOL_SIZES[sf]}, {args.runs} timed runs"
            )

            t0 = time.perf_counter()
            success, error, per_query_lat, per_query_rows = benchmark_shilmandb_config(
                binary=args.binary,
                sf=sf,
                config_name=config_name,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                num_runs=args.runs,
                purge=args.purge,
            )
            config_time = time.perf_counter() - t0

            if success:
                configs_that_ran.append(config_name)

                # Compute medians and accumulate records
                for query in QUERIES:
                    latencies = per_query_lat.get(query, [])
                    rows = per_query_rows.get(query, 0)
                    if latencies:
                        median = statistics.median(latencies)
                        shilmandb_results.append({
                            "sf": sf,
                            "query": query,
                            "config": config_name,
                            "median_latency_ms": median,
                            "rows": rows,
                        })

                print(
                    f"    Completed {config_name} SF={sf} "
                    f"({elapsed_str(config_time)})"
                )
            else:
                msg = f"{config_name} SF={sf}: {error}"
                print(f"    FAILED: {msg}")
                failures.append(msg)

        # ---- Correctness comparison for this SF ----
        if not args.skip_correctness and configs_that_ran:
            sub_header(f"Correctness comparison (SF={sf})")
            cmp_ok, cmp_err = run_correctness_comparison(
                sf, configs_that_ran, args.output_dir,
            )
            correctness_results.append((sf, configs_that_ran, cmp_ok, cmp_err or ""))
            if not cmp_ok:
                print(f"  WARNING: Correctness issues detected for SF={sf}")
                # Do NOT abort -- keep benchmarking
        elif args.skip_correctness:
            print("\n  Skipping correctness comparison (--skip-correctness).")


    section_header("MERGE RESULTS")

    all_merged: List[Dict] = list(shilmandb_results)

    # Add SQLite medians
    if sqlite_ok:
        sqlite_records = parse_sqlite_latencies(args.output_dir, valid_sfs)
        all_merged.extend(sqlite_records)
        print(f"  SQLite: {len(sqlite_records)} median records")
    else:
        print("  SQLite: no results available")

    print(f"  ShilmanDB: {len(shilmandb_results)} median records")

    if all_merged:
        merged_path = os.path.join(args.output_dir, "latencies.csv")
        write_merged_latencies(all_merged, merged_path)
        print(f"  Merged latencies written to {merged_path}")
    else:
        print("  WARNING: No latency data to merge.")


    overall_time = time.perf_counter() - overall_t0
    section_header("SUMMARY")

    # Latency summary table
    if all_merged:
        print(f"\n  {'SF':<8} {'Query':<6} {'Config':<30} {'Median (ms)':>12} {'Rows':>6}")
        print(f"  {'-' * 66}")
        for rec in sorted(all_merged, key=lambda r: (r["sf"], r["query"], r["config"])):
            print(
                f"  {rec['sf']:<8} {rec['query']:<6} {rec['config']:<30} "
                f"{rec['median_latency_ms']:>12.2f} {rec['rows']:>6}"
            )

    # Correctness summary
    if correctness_results:
        print(f"\n  Correctness checks:")
        for sf, configs, passed, err in correctness_results:
            status = "PASS" if passed else "FAIL"
            print(f"    SF={sf} [{', '.join(configs)}]: {status}")
            if not passed and err:
                print(f"      {err[:120]}")

    # Failures
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for msg in failures:
            short = msg.split("\n")[0][:120]
            print(f"    - {short}")
    else:
        print(f"\n  All benchmarks completed successfully.")

    print(f"\n  Total elapsed: {elapsed_str(overall_time)}")
    print(f"  Output directory: {args.output_dir}")
    if all_merged:
        print(f"  Merged CSV: {os.path.join(args.output_dir, 'latencies.csv')}")

    # Exit with error if there were failures (but we ran everything we could)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
