#!/usr/bin/env python3
import argparse
import csv
import math
import os
import subprocess
import sys
import tempfile


def discover_scale_factors(data_base_dir: str) -> list[str]:
    sfs = []
    for entry in sorted(os.listdir(data_base_dir)):
        if entry.startswith("sf"):
            sf_dir = os.path.join(data_base_dir, entry)
            
            if os.path.isdir(sf_dir) and os.path.isfile(os.path.join(sf_dir, "lineitem.tbl")):
                sfs.append(entry[2:])  
    return sfs


def run_cpp_binary(binary: str, sf: str, data_dir: str, output_csv: str, perturbations: int, seed: int,) -> int:
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    cmd = [
        binary,
        "--sf", sf,
        "--db-file", db_path,
        "--data-dir", data_dir,
        "--output", output_csv,
        "--perturbations", str(perturbations),
        "--seed", str(seed),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: C++ generator failed for SF={sf}", file=sys.stderr)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
                
            raise subprocess.CalledProcessError(result.returncode, cmd)
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


    with open(output_csv, "r") as f:
        row_count = sum(1 for _ in f) - 1

    return row_count


def concatenate_csvs(csv_paths: list[str], output_path: str) -> int:
    total_rows = 0
    with open(output_path, "w", newline="") as out_f:
        writer = None
        for i, path in enumerate(csv_paths):
            with open(path, "r", newline="") as in_f:
                reader = csv.reader(in_f)
                
                try:
                    header = next(reader)
                except StopIteration:
                    continue
                
                if i == 0:
                    writer = csv.writer(out_f)
                    writer.writerow(header)
                    
                for row in reader:
                    writer.writerow(row)
                    total_rows += 1
                    
    return total_rows


def validate_csv(path: str) -> bool:
    passed = True
    warnings = 0

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    row_count = len(rows)
    print(f"\n=== Validation ===")

    # Row count
    if row_count >= 500:
        print(f"Row count: {row_count} (>= 500) OK")
    else:
        print(f"Row count: {row_count} (< 500) ERROR")
        passed = False


    bad_features = 0
    bad_num_tables = 0
    bad_orders = 0
    num_tables_dist: dict[int, int] = {}
    unique_orderings: set[tuple[int, ...]] = set()

    for row_idx, row in enumerate(rows):
        for j in range(48):
            try:
                val = float(row[j])
                if not math.isfinite(val):
                    bad_features += 1
                    
            except (ValueError, IndexError):
                bad_features += 1


        try:
            nt = int(row[48])
            if nt not in (2, 3, 4, 5, 6):
                bad_num_tables += 1
                
            num_tables_dist[nt] = num_tables_dist.get(nt, 0) + 1
            
        except (ValueError, IndexError):
            bad_num_tables += 1
            continue

        try:
            order = tuple(int(row[49 + k]) for k in range(6))
            active = order[:nt]
            
            if any(v < 0 or v > 5 for v in order):
                bad_orders += 1
            elif len(set(active)) != len(active):
                bad_orders += 1
                
            unique_orderings.add(order)
        except (ValueError, IndexError):
            bad_orders += 1

    if bad_features > 0:
        print(f"Feature values: {bad_features} bad values WARNING")
        warnings += 1
    else:
        print("Feature ranges: all finite OK")

    if bad_num_tables > 0:
        print(f"num_tables: {bad_num_tables} invalid rows WARNING")
        warnings += 1

    print(f"num_tables distribution: {dict(sorted(num_tables_dist.items()))}")

    if bad_orders > 0:
        print(f"Order values: {bad_orders} invalid rows WARNING")
        warnings += 1

    n_unique = len(unique_orderings)
    if n_unique > 10:
        print(f"Unique optimal orderings: {n_unique}")
    else:
        print(f"Unique optimal orderings: {n_unique} (< 10) WARNING")
        warnings += 1

    if passed and warnings == 0:
        print("Validation: PASSED")
    elif passed:
        print(f"Validation: PASSED with {warnings} warning(s)")
    else:
        print("Validation: FAILED")

    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate join training data at multiple scale factors")
    parser.add_argument("--binary", default="build/bench/generate_join_training_data", help="Path to C++ generator binary")
    parser.add_argument("--data-base-dir", default="bench/tpch_data", help="Base directory containing sf*/ data dirs")
    parser.add_argument("--output", default="bench/tpch_data/join_training_data.csv", help="Output CSV path")
    parser.add_argument("--perturbations", type=int, default=40, help="Perturbations per subset")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    print("=== Join Training Data Generator (Python Orchestrator) ===")

    sfs = discover_scale_factors(args.data_base_dir)
    if not sfs:
        print(f"ERROR: No scale factor directories found in {args.data_base_dir}")
        sys.exit(1)
        
    print(f"Available scale factors: {sfs}")

    # Run C++ binary per SF
    tmp_csvs: list[str] = []
    try:
        for sf_idx, sf in enumerate(sfs):
            seed = args.seed + sf_idx
            print(f"\nRunning SF={sf} (seed={seed}) ...")
            data_dir = os.path.join(args.data_base_dir, f"sf{sf}") + "/"

            tmp_fd, tmp_csv = tempfile.mkstemp(suffix=".csv")
            os.close(tmp_fd)
            row_count = run_cpp_binary(args.binary, sf, data_dir, tmp_csv, args.perturbations, seed)
            print(f"  -> {row_count} rows generated")
            tmp_csvs.append(tmp_csv)

        total = concatenate_csvs(tmp_csvs, args.output)
        print(f"\nConcatenated: {total} total rows")
        print(f"Output: {args.output}")
        
    finally:
        for path in tmp_csvs:
            if os.path.exists(path):
                os.unlink(path)

    # Validate
    ok = validate_csv(args.output)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
