#!/usr/bin/env python3


import argparse
import csv
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Tuple, Union



CONFIGS: Dict[str, Dict[str, str]] = {
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

ALL_CONFIGS = list(CONFIGS.keys())

ALL_SCALE_FACTORS = ["0.01", "0.1", "1.0"]

POOL_SIZES: Dict[str, int] = {
    "0.01": 128,
    "0.1": 512,
    "1.0": 2048,
}

QUERY_TAGS = ["Q1", "Q3", "Q5", "Q6", "Q10", "Q12", "Q14", "Q19"]




def build_command(binary: str, sf: str, config_name: str, data_dir: str, results_dir: str, db_file: str, execution_mode: str) -> List[str]:
    pool_size = POOL_SIZES[sf]
    sf_data_dir = os.path.join(data_dir, f"sf{sf}")

    cmd = [
        binary,
        "--sf", sf,
        "--db-file", db_file,
        "--data-dir", sf_data_dir,
        "--pool-size", str(pool_size),
        "--results-dir", results_dir,
        "--execution-mode", execution_mode,
    ]

    for flag, value in CONFIGS[config_name].items():
        cmd.append(flag)
        if value:
            cmd.append(value)

    return cmd


def parse_latencies_csv(path: str) -> List[Dict[str, str]]:
    records = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def write_combined_latencies(records: List[Dict], output_path: str) -> None:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sf", "query", "config", "latency_ms", "rows"])
        for rec in records:
            writer.writerow([
                rec["sf"], rec["query"], rec["config"],
                rec["latency_ms"], rec["rows"],
            ])




def run_single(binary: str, sf: str, config_name: str, data_dir: str, output_dir: str, execution_mode: str) -> Tuple[bool, Union[str, List[Dict]]]:
    temp_results = f"/tmp/shilmandb_results_{config_name}_{execution_mode}_{sf}"
    db_file = f"/tmp/shilmandb_{config_name}_{execution_mode}_sf{sf}.db"

    # Clean up any stale files from a previous run
    if os.path.exists(temp_results):
        shutil.rmtree(temp_results)

    if os.path.exists(db_file):
        os.remove(db_file)

    os.makedirs(temp_results, exist_ok=True)

    cmd = build_command(binary, sf, config_name, data_dir, temp_results, db_file, execution_mode)
    print(f"    Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout per run
        )
    except subprocess.TimeoutExpired:
        _cleanup(temp_results, db_file)
        return False, "Timed out after 1800 seconds"
    except OSError as e:
        _cleanup(temp_results, db_file)
        return False, f"Failed to execute binary: {e}"

    if result.returncode != 0:
        error_msg = (
            f"Non-zero exit code: {result.returncode}\n"
            f"--- stdout ---\n{result.stdout[-2000:] if result.stdout else '(empty)'}\n"
            f"--- stderr ---\n{result.stderr[-2000:] if result.stderr else '(empty)'}"
        )
        _cleanup(temp_results, db_file)
        return False, error_msg

    # Print stdout summary (last few lines)
    if result.stdout:
        lines = result.stdout.strip().split("\n")
        summary_lines = lines[-5:] if len(lines) > 5 else lines
        for line in summary_lines:
            print(f"    | {line}")

    # Copy per-query CSV results
    config_results_dir = os.path.join(output_dir, "shilmandb_results", f"{config_name}_{execution_mode}")
    os.makedirs(config_results_dir, exist_ok=True)

    for tag in QUERY_TAGS:
        src = os.path.join(temp_results, f"{tag}.csv")
        dst = os.path.join(config_results_dir, f"{tag}_sf{sf}.csv")
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"    WARNING: Expected {src} not found")

    # Parse latencies and check for partial query failures
    latencies_path = os.path.join(temp_results, "latencies.csv")
    latency_records = []
    if os.path.exists(latencies_path):
        latency_records = parse_latencies_csv(latencies_path)
        if len(latency_records) < len(QUERY_TAGS):
            recorded = {r["query"] for r in latency_records}
            missing = [t for t in QUERY_TAGS if t not in recorded]
            print(f"    WARNING: Only {len(latency_records)}/{len(QUERY_TAGS)} queries succeeded. "
                  f"Missing: {missing}")
    else:
        print(f"    WARNING: {latencies_path} not found")

    _cleanup(temp_results, db_file)
    return True, latency_records


def _cleanup(temp_results: str, db_file: str) -> None:
    if os.path.exists(temp_results):
        shutil.rmtree(temp_results, ignore_errors=True)
        
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except OSError:
            pass




def print_summary(all_latencies: List[Dict]) -> None:
    if not all_latencies:
        return

    print(f"\n{'=' * 72}")
    print(f"{'SF':<8} {'Query':<6} {'Config':<30} {'Latency (ms)':>12} {'Rows':>8}")
    print(f"{'-' * 72}")
    for rec in sorted(all_latencies, key=lambda r: (r["sf"], r["query"], r["config"])):
        print(
            f"{rec['sf']:<8} {rec['query']:<6} {rec['config']:<30} "
            f"{float(rec['latency_ms']):>12.2f} {rec['rows']:>8}"
        )
    print(f"{'=' * 72}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TPC-H benchmarks on ShilmanDB across multiple configurations."
    )
    parser.add_argument(
        "--binary",
        default="build/bench/load_tpch",
        help="Path to load_tpch binary (default: build/bench/load_tpch)",
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
        help="Scale factors: 0.01, 0.1, 1.0, or 'all' (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="bench/results",
        help="Directory for output files (default: bench/results)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["all"],
        help=(
            "Configs to run: lru_heuristic, lru_learned_join, "
            "learned_eviction_heuristic, learned_all, or 'all' (default: all)"
        ),
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Trigger 'bash scripts/build.sh Release ON' before running",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["tuple", "vectorized"],
        default="tuple",
        help="Execution mode passed to load_tpch --execution-mode (default: tuple)",
    )
    return parser.parse_args()




def main() -> None:
    args = parse_args()

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

    # Resolve configs
    if "all" in args.configs:
        configs = ALL_CONFIGS
    else:
        configs = []
        for c in args.configs:
            if c not in ALL_CONFIGS:
                print(f"Error: unknown config '{c}'. Choose from {ALL_CONFIGS} or 'all'.")
                sys.exit(1)
            configs.append(c)

    print("ShilmanDB TPC-H Benchmark Runner")
    print(f"  Scale factors: {scale_factors}")
    print(f"  Configs: {configs}")
    print(f"  Binary: {args.binary}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Execution mode: {args.execution_mode}")

    # Optional build step
    if args.build:
        print("\n--- Building (Release + ML) ---")
        build_result = subprocess.run(
            ["bash", "scripts/build.sh", "Release", "ON"],
            capture_output=False,
        )
        if build_result.returncode != 0:
            print("Error: build failed. Aborting.")
            sys.exit(1)
        print("Build succeeded.\n")

    # Check binary exists
    if not os.path.isfile(args.binary):
        print(f"Error: binary not found at '{args.binary}'.")
        print("  Run with --build to build it, or pass --binary <path>.")
        sys.exit(1)

    # Run benchmarks
    all_latencies: List[Dict] = []
    failures: List[Tuple[str, str, str]] = []  # (config, sf, error)
    total_runs = len(configs) * len(scale_factors)
    run_num = 0

    for sf in scale_factors:
        sf_data_dir = os.path.join(args.data_dir, f"sf{sf}")
        if not os.path.isdir(sf_data_dir):
            print(f"\n  WARNING: Data directory {sf_data_dir} not found, skipping SF={sf}")
            for config_name in configs:
                failures.append((config_name, sf, f"Data directory {sf_data_dir} not found"))
                run_num += 1
            continue

        for config_name in configs:
            run_num += 1
            print(f"\n{'=' * 60}")
            print(f"  [{run_num}/{total_runs}] Config: {config_name}, Mode: {args.execution_mode}, SF={sf}, Pool={POOL_SIZES[sf]}")
            print(f"{'=' * 60}")

            success, result = run_single(
                args.binary, sf, config_name, args.data_dir, args.output_dir, args.execution_mode
            )

            if success:
                # result is the list of latency records from latencies.csv
                for rec in result:
                    all_latencies.append({
                        "sf": sf,
                        "query": rec["query"],
                        "config": f"{config_name}_{args.execution_mode}",
                        "latency_ms": rec["latency_ms"],
                        "rows": rec["rows"],
                    })
                print(f"    OK ({len(result)} queries recorded)")
            else:
                print(f"    FAILED: {result}")
                failures.append((config_name, sf, result))

    # Write combined latencies
    if all_latencies:
        latencies_path = os.path.join(args.output_dir, "shilmandb_latencies.csv")
        write_combined_latencies(all_latencies, latencies_path)
        print(f"\nLatencies written to {latencies_path}")

    # Print summary
    print_summary(all_latencies)

    # List exported result files
    results_base = os.path.join(args.output_dir, "shilmandb_results")
    if os.path.isdir(results_base):
        print(f"\nQuery result CSVs in {results_base}/:")
        for config_name in sorted(os.listdir(results_base)):
            config_dir = os.path.join(results_base, config_name)
            if not os.path.isdir(config_dir):
                continue
            files = sorted(os.listdir(config_dir))
            for fname in files:
                fpath = os.path.join(config_dir, fname)
                size = os.path.getsize(fpath)
                print(f"  {config_name}/{fname} ({size:,} bytes)")

    # Report failures
    if failures:
        print(f"\n{'!' * 60}")
        print(f"  {len(failures)} run(s) FAILED:")
        for config_name, sf, error in failures:
            # Truncate long errors for the summary
            short_error = error.split("\n")[0][:120]
            print(f"    {config_name} SF={sf}: {short_error}")
        print(f"{'!' * 60}")
        sys.exit(1)

    if not all_latencies:
        print("\nNo benchmarks ran. Check that data files exist and binary is built.")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
