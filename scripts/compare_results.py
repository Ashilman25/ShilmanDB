#!/usr/bin/env python3


import argparse
import csv
import os
import sys
from typing import List, Optional, Tuple




ALL_SCALE_FACTORS = ["0.01", "0.1", "1.0"]
ALL_QUERIES = ["Q1", "Q3", "Q5", "Q6", "Q10", "Q12", "Q14", "Q19"]

DEFAULT_SHILMANDB_DIR = "bench/results/shilmandb_results"
DEFAULT_SQLITE_DIR = "bench/results/sqlite_results"
DEFAULT_TOLERANCE = 0.01



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


def cells_match(expected: str, actual: str, tolerance: float) -> bool:
    expected_f = try_parse_float(expected)
    actual_f = try_parse_float(actual)

    if expected_f is not None and actual_f is not None:
        return abs(expected_f - actual_f) <= tolerance

    return expected == actual




class ComparisonResult:

    __slots__ = ("config", "sf", "query", "passed", "mismatches", "error")

    def __init__(self, config: str, sf: str, query: str):
        self.config = config
        self.sf = sf
        self.query = query
        self.passed: bool = False
        self.mismatches: List[str] = []
        self.error: Optional[str] = None


def compare_one(sqlite_path: str, shilmandb_path: str, config: str, sf: str, query: str, tolerance: float) -> ComparisonResult:
    result = ComparisonResult(config, sf, query)

    if not os.path.exists(sqlite_path):
        result.error = f"SQLite reference not found: {sqlite_path}"
        return result
    
    if not os.path.exists(shilmandb_path):
        result.error = f"ShilmanDB result not found: {shilmandb_path}"
        return result

    ref_headers, ref_rows = load_csv(sqlite_path)
    cand_headers, cand_rows = load_csv(shilmandb_path)


    if ref_headers != cand_headers:
        result.error = (
            f"Header mismatch: SQLite={ref_headers}, ShilmanDB={cand_headers}"
        )
        return result


    if len(ref_rows) != len(cand_rows):
        result.error = (
            f"Row count mismatch: SQLite={len(ref_rows)}, ShilmanDB={len(cand_rows)}"
        )
        return result


    num_cols = len(ref_headers)
    for i, row in enumerate(ref_rows):
        if len(row) != num_cols:
            result.error = f"SQLite row {i} has {len(row)} columns, expected {num_cols}"
            return result
        
    for i, row in enumerate(cand_rows):
        if len(row) != num_cols:
            result.error = f"ShilmanDB row {i} has {len(row)} columns, expected {num_cols}"
            return result


    ref_sorted = sort_rows(ref_rows)
    cand_sorted = sort_rows(cand_rows)


    for row_idx, (ref_row, cand_row) in enumerate(zip(ref_sorted, cand_sorted)):
        for col_idx, (ref_val, cand_val) in enumerate(zip(ref_row, cand_row)):
            if not cells_match(ref_val, cand_val, tolerance):
                col_name = ref_headers[col_idx]
                result.mismatches.append(
                    f"  row {row_idx}, col '{col_name}': "
                    f"expected={ref_val!r}  actual={cand_val!r}"
                )

    result.passed = len(result.mismatches) == 0
    return result


def discover_configs(shilmandb_dir: str) -> List[str]:
    if not os.path.isdir(shilmandb_dir):
        return []
    
    return sorted(
        d for d in os.listdir(shilmandb_dir)
        if os.path.isdir(os.path.join(shilmandb_dir, d))
    )


#main pipeline

def run_comparisons(shilmandb_dir: str, sqlite_dir: str, configs: List[str], scale_factors: List[str], queries: List[str], tolerance: float) -> List[ComparisonResult]:
    results: List[ComparisonResult] = []

    for config in configs:
        for sf in scale_factors:
            for query in queries:
                filename = f"{query}_sf{sf}.csv"
                sqlite_path = os.path.join(sqlite_dir, filename)
                shilmandb_path = os.path.join(shilmandb_dir, config, filename)

                cr = compare_one(
                    sqlite_path, shilmandb_path, config, sf, query, tolerance
                )
                results.append(cr)

    return results


def print_results(results: List[ComparisonResult]) -> int:
    passed = 0
    failed = 0

    for cr in results:
        label = f"[{cr.config}] {cr.query} SF={cr.sf}"

        if cr.error:
            print(f"  SKIP  {label}")
            print(f"        {cr.error}")
            failed += 1
            continue

        if cr.passed:
            print(f"  PASS  {label}")
            passed += 1
        else:
            print(f"  FAIL  {label}  ({len(cr.mismatches)} mismatch(es))")
            for line in cr.mismatches:
                print(f"       {line}")
            failed += 1

    total = passed + failed
    print()
    print("-" * 60)
    print(f"  {passed}/{total} comparisons passed", end="")
    if failed > 0:
        print(f"  ({failed} failed)")
    else:
        print()
    print("-" * 60)

    return failed


#cli commands and args

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ShilmanDB results against SQLite reference results."
    )
    parser.add_argument(
        "--shilmandb-dir",
        default=DEFAULT_SHILMANDB_DIR,
        help=f"ShilmanDB results directory (default: {DEFAULT_SHILMANDB_DIR})",
    )
    parser.add_argument(
        "--sqlite-dir",
        default=DEFAULT_SQLITE_DIR,
        help=f"SQLite results directory (default: {DEFAULT_SQLITE_DIR})",
    )
    parser.add_argument(
        "--sf",
        nargs="+",
        default=["all"],
        help="Scale factors: 0.01, 0.1, 1.0, or 'all' (default: all)",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()


    if "all" in args.sf:
        scale_factors = ALL_SCALE_FACTORS
    else:
        scale_factors = []
        for sf in args.sf:
            if sf not in ALL_SCALE_FACTORS:
                print(f"Error: unknown scale factor '{sf}'. Choose from {ALL_SCALE_FACTORS} or 'all'.")
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


    print("ShilmanDB vs SQLite Result Comparison")
    print(f"  ShilmanDB dir: {args.shilmandb_dir}")
    print(f"  SQLite dir:    {args.sqlite_dir}")
    print(f"  Configs:       {configs}")
    print(f"  Scale factors: {scale_factors}")
    print(f"  Queries:       {queries}")
    print(f"  Tolerance:     {args.tolerance}")
    print()

    results = run_comparisons(
        args.shilmandb_dir, args.sqlite_dir,
        configs, scale_factors, queries, args.tolerance,
    )

    failures = print_results(results)
    sys.exit(1 if failures > 0 else 0)


if __name__ == "__main__":
    main()
