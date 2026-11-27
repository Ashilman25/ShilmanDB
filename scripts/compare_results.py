#!/usr/bin/env python3


import argparse
import csv
import math
import os
import sys
from typing import List, Optional, Tuple




ALL_SCALE_FACTORS = ["0.01", "0.1", "1.0"]
KNOWN_SCALE_FACTORS = [*ALL_SCALE_FACTORS, "10", "100"]  # 10 and 100 run only when named
ALL_QUERIES = ["Q1", "Q3", "Q5", "Q6", "Q10", "Q12", "Q14", "Q19"]

DEFAULT_SHILMANDB_DIR = "bench/results/shilmandb_results"
DEFAULT_SQLITE_DIR = "bench/results/sqlite_results"
DEFAULT_REFERENCE_NAME = "SQLite"
DEFAULT_TOLERANCE = 0.01

# A decimal that misses the tolerance but agrees to this relative difference is DRIFT.
# 1e-10 of a 1e12 sum is $100, under the ~$810 the smallest TPC-H row adds, so a lost row still fails.
DRIFT_RELATIVE = 1e-10
STATUS_RANK = {"PASS": 0, "DRIFT": 1, "FAIL": 2}



def load_csv(path: str) -> Tuple[List[str], List[List[str]]]:
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = [row for row in reader if any(cell.strip() for cell in row)]
        
    return headers, rows




def _row_sort_key(row: List[str]) -> tuple:

    parts = []
    for cell in row:
        f = try_parse_float(cell)
        parts.append((0, f) if f is not None else (1, cell))
        
    return tuple(parts)


def sort_rows(rows: List[List[str]]) -> List[List[str]]:
    return sorted(rows, key=_row_sort_key)




def try_parse_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# Grades one cell as PASS, DRIFT or FAIL, with its absolute and relative difference.
# Text, dates and non-finite numbers compare exactly; only two decimals can drift.
def cell_status(expected: str, actual: str, tolerance: float) -> Tuple[str, float, float]:
    expected_f = try_parse_float(expected)
    actual_f = try_parse_float(actual)

    numeric = (expected_f is not None and actual_f is not None
               and math.isfinite(expected_f) and math.isfinite(actual_f))
    if not numeric:
        return ("PASS" if expected == actual else "FAIL"), 0.0, 0.0

    abs_diff = abs(expected_f - actual_f)
    rel_diff = abs_diff / max(abs(expected_f), abs(actual_f), 1e-300)

    if abs_diff <= tolerance:
        return "PASS", abs_diff, rel_diff

    both_decimal = "." in expected and "." in actual
    if both_decimal and rel_diff <= DRIFT_RELATIVE:
        return "DRIFT", abs_diff, rel_diff

    return "FAIL", abs_diff, rel_diff


def cells_match(expected: str, actual: str, tolerance: float) -> bool:
    return cell_status(expected, actual, tolerance)[0] == "PASS"




class ComparisonResult:

    __slots__ = ("config", "sf", "query", "status", "mismatches", "drifts",
                 "max_abs_diff", "max_rel_diff", "error")

    def __init__(self, config: str, sf: str, query: str):
        self.config = config
        self.sf = sf
        self.query = query
        self.status: str = "FAIL"  # PASS, DRIFT, FAIL or SKIP (a missing file)
        self.mismatches: List[str] = []  # FAIL cells
        self.drifts: List[str] = []      # DRIFT cells
        # largest differences among the cells that missed the tolerance
        self.max_abs_diff: float = 0.0
        self.max_rel_diff: float = 0.0
        self.error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


# Compares one answer file with its reference. no_drift counts a would-be DRIFT cell as FAIL.
def compare_one(reference_path: str, candidate_path: str, config: str, sf: str, query: str,
                tolerance: float, reference_name: str = DEFAULT_REFERENCE_NAME,
                no_drift: bool = False) -> ComparisonResult:
    result = ComparisonResult(config, sf, query)

    if not os.path.exists(reference_path):
        result.status = "SKIP"
        result.error = f"{reference_name} reference not found: {reference_path}"
        return result
    
    if not os.path.exists(candidate_path):
        result.status = "SKIP"
        result.error = f"ShilmanDB result not found: {candidate_path}"
        return result

    ref_headers, ref_rows = load_csv(reference_path)
    cand_headers, cand_rows = load_csv(candidate_path)


    if ref_headers != cand_headers:
        result.error = (
            f"Header mismatch: {reference_name}={ref_headers}, ShilmanDB={cand_headers}"
        )
        return result


    if len(ref_rows) != len(cand_rows):
        result.error = (
            f"Row count mismatch: {reference_name}={len(ref_rows)}, ShilmanDB={len(cand_rows)}"
        )
        return result


    num_cols = len(ref_headers)
    for i, row in enumerate(ref_rows):
        if len(row) != num_cols:
            result.error = f"{reference_name} row {i} has {len(row)} columns, expected {num_cols}"
            return result
        
    for i, row in enumerate(cand_rows):
        if len(row) != num_cols:
            result.error = f"ShilmanDB row {i} has {len(row)} columns, expected {num_cols}"
            return result


    ref_sorted = sort_rows(ref_rows)
    cand_sorted = sort_rows(cand_rows)


    worst = "PASS"
    for row_idx, (ref_row, cand_row) in enumerate(zip(ref_sorted, cand_sorted)):
        for col_idx, (ref_val, cand_val) in enumerate(zip(ref_row, cand_row)):
            status, abs_diff, rel_diff = cell_status(ref_val, cand_val, tolerance)
            if status == "PASS":
                continue

            col_name = ref_headers[col_idx]
            detail = (
                f"  row {row_idx}, col '{col_name}': "
                f"expected={ref_val!r}  actual={cand_val!r}"
            )
            if status == "DRIFT":
                detail += f"  (abs {abs_diff:.3g}, rel {rel_diff:.3g})"
                if no_drift:
                    status = "FAIL"
            if status == "FAIL":
                result.mismatches.append(detail)
            else:
                result.drifts.append(detail)
            result.max_abs_diff = max(result.max_abs_diff, abs_diff)
            result.max_rel_diff = max(result.max_rel_diff, rel_diff)
            worst = max(worst, status, key=STATUS_RANK.__getitem__)

    result.status = worst
    return result


def discover_configs(shilmandb_dir: str) -> List[str]:
    if not os.path.isdir(shilmandb_dir):
        return []
    
    return sorted(
        d for d in os.listdir(shilmandb_dir)
        if os.path.isdir(os.path.join(shilmandb_dir, d))
    )


#main pipeline

# Reference files sit in reference_dir, or in reference_dir/reference_config when one is given.
def run_comparisons(shilmandb_dir: str, reference_dir: str, configs: List[str], scale_factors: List[str],
                    queries: List[str], tolerance: float, reference_config: Optional[str] = None,
                    reference_name: str = DEFAULT_REFERENCE_NAME, no_drift: bool = False) -> List[ComparisonResult]:
    results: List[ComparisonResult] = []
    if reference_config:
        reference_dir = os.path.join(reference_dir, reference_config)

    for config in configs:
        for sf in scale_factors:
            for query in queries:
                filename = f"{query}_sf{sf}.csv"
                reference_path = os.path.join(reference_dir, filename)
                candidate_path = os.path.join(shilmandb_dir, config, filename)

                cr = compare_one(
                    reference_path, candidate_path, config, sf, query, tolerance, reference_name, no_drift
                )
                results.append(cr)

    return results


# Prints one line per comparison and a summary; returns the FAIL + SKIP count.
def print_results(results: List[ComparisonResult]) -> int:
    passed = 0
    drifted = 0
    failed = 0

    for cr in results:
        label = f"[{cr.config}] {cr.query} SF={cr.sf}"

        if cr.status == "SKIP":
            print(f"  SKIP  {label}")
            print(f"        {cr.error}")
            failed += 1
        elif cr.error:
            print(f"  FAIL  {label}")
            print(f"        {cr.error}")
            failed += 1
        elif cr.status == "PASS":
            print(f"  PASS  {label}")
            passed += 1
        elif cr.status == "DRIFT":
            print(f"  DRIFT {label}  (max abs {cr.max_abs_diff:.3g}, max rel {cr.max_rel_diff:.3g})")
            for line in cr.drifts:
                print(f"       {line}")
            drifted += 1
        else:
            print(f"  FAIL  {label}  ({len(cr.mismatches)} mismatch(es))")
            for line in cr.mismatches:
                print(f"       {line}")
            failed += 1

    total = passed + drifted + failed
    print()
    print("-" * 60)
    print(f"  {passed}/{total} comparisons passed", end="")
    notes = ([f"{drifted} drift"] if drifted else []) + ([f"{failed} failed"] if failed else [])
    if notes:
        print(f"  ({', '.join(notes)})")
    else:
        print()
    print("-" * 60)

    return failed


#cli commands and args

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ShilmanDB results against reference results (SQLite by default)."
    )
    parser.add_argument(
        "--shilmandb-dir",
        default=DEFAULT_SHILMANDB_DIR,
        help=f"ShilmanDB results directory (default: {DEFAULT_SHILMANDB_DIR})",
    )
    parser.add_argument(
        "--reference-dir", "--sqlite-dir",
        dest="reference_dir",
        default=DEFAULT_SQLITE_DIR,
        help=f"Reference results directory (default: {DEFAULT_SQLITE_DIR})",
    )
    parser.add_argument(
        "--reference-config",
        default=None,
        help="Read reference files from reference-dir/NAME instead of reference-dir",
    )
    parser.add_argument(
        "--reference-name",
        default=DEFAULT_REFERENCE_NAME,
        help=f"Reference name used in messages (default: {DEFAULT_REFERENCE_NAME})",
    )
    parser.add_argument(
        "--sf",
        nargs="+",
        default=["all"],
        help="Scale factors: 0.01, 0.1, 1.0, 10, 100, or 'all' = 0.01 0.1 1.0 (default: all)",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=["all"],
        help="Queries: Q1, Q3, Q5, Q6, Q10, Q12, Q14, Q19, or 'all' (default: all)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="ShilmanDB configs to compare (default: all found in shilmandb-dir)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Float comparison tolerance (default: {DEFAULT_TOLERANCE})",
    )
    parser.add_argument(
        "--no-drift",
        action="store_true",
        help="Count a cell that would be DRIFT as FAIL (the runner passes this at SF=100)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)


    if "all" in args.sf:
        scale_factors = ALL_SCALE_FACTORS
    else:
        scale_factors = []
        for sf in args.sf:
            if sf not in KNOWN_SCALE_FACTORS:
                print(f"Error: unknown scale factor '{sf}'. Choose from {KNOWN_SCALE_FACTORS} or 'all'.")
                sys.exit(1)
            scale_factors.append(sf)


    if "all" in args.queries:
        queries = ALL_QUERIES
    else:
        queries = []
        for q in args.queries:
            q_upper = q.upper()
            if q_upper not in ALL_QUERIES:
                print(f"Error: unknown query '{q}'. Choose from {ALL_QUERIES} or 'all'.")
                sys.exit(1)
            queries.append(q_upper)


    if args.configs is not None:
        configs = args.configs
    else:
        configs = discover_configs(args.shilmandb_dir)
        if not configs:
            print(f"Error: no config directories found in {args.shilmandb_dir}")
            sys.exit(1)


    reference_dir = args.reference_dir
    if args.reference_config:
        reference_dir = os.path.join(reference_dir, args.reference_config)

    print(f"ShilmanDB vs {args.reference_name} Result Comparison")
    print(f"  ShilmanDB dir: {args.shilmandb_dir}")
    print(f"  {args.reference_name + ' dir:':<14} {reference_dir}")
    print(f"  Configs:       {configs}")
    print(f"  Scale factors: {scale_factors}")
    print(f"  Queries:       {queries}")
    print(f"  Tolerance:     {args.tolerance}")
    if args.no_drift:
        print("  No drift:      on (a cell that would be DRIFT is FAIL)")
    print()

    results = run_comparisons(
        args.shilmandb_dir, args.reference_dir,
        configs, scale_factors, queries, args.tolerance,
        args.reference_config, args.reference_name, args.no_drift,
    )

    failures = print_results(results)
    sys.exit(1 if failures > 0 else 0)


if __name__ == "__main__":
    main()
