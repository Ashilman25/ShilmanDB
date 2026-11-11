#!/usr/bin/env python3
#Check workload answers against SQLite.
from __future__ import annotations

import argparse
import csv
import importlib.util
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, List, Optional, Sequence, Union

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SCALE_FACTORS = ["0.01", "0.1", "1.0"]
DEFAULT_TOLERANCE = 0.01


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_sqlite_bench = _load_module("run_sqlite_benchmarks", HERE / "run_sqlite_benchmarks.py")
_compare = _load_module("compare_results", REPO_ROOT / "scripts" / "compare_results.py")


@dataclass
class Item:
    seq: int
    query: str
    sql: str
    rows: List[List[str]]  # ShilmanDB's cells, by row then column


@dataclass
class CheckRow:
    sf: str
    config: str  # the folder name
    seq: int
    query: str
    status: str  # "PASS" or "FAIL"
    detail: str = ""


#The scale-factor copy the runner makes if there is one, else the plain file
def _find_csv(folder: Path, stem: str, sf: str) -> Path:
    for name in (f"{stem}_sf{sf}.csv", f"{stem}.csv"):
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(f"{folder}: no {stem}_sf{sf}.csv or {stem}.csv")


def load_items(folder: Path, sf: str) -> List[Item]:
    with open(_find_csv(folder, "stream_queries", sf), newline="") as f:
        items = {int(r["seq"]): Item(int(r["seq"]), r["query"], r["sql"], []) for r in csv.DictReader(f)}

    # Results are in long form: one line per cell.
    cells: Dict[int, Dict[int, Dict[int, str]]] = {}
    with open(_find_csv(folder, "stream_results", sf), newline="") as f:
        for r in csv.DictReader(f):
            seq = int(r["seq"])
            if seq not in items:
                raise ValueError(f"{folder}: result for seq {seq} has no query")
            cells.setdefault(seq, {}).setdefault(int(r["row"]), {})[int(r["col"])] = r["value"]

    for seq, rows in cells.items():
        items[seq].rows = [[row[c] for c in sorted(row)] for _, row in sorted(rows.items())]
    return [items[seq] for seq in sorted(items)]


def sqlite_cells(row: tuple) -> List[str]:
    # ShilmanDB has no NULL: an aggregate over no rows gives 0.
    return ["0" if value is None else str(value) for value in row]


#None when the two answers match, else a short reason
def compare_rows(expected: List[List[str]], actual: List[List[str]], tolerance: float = DEFAULT_TOLERANCE) -> Optional[str]:
    if len(expected) != len(actual):
        return f"row count: expected {len(expected)}, got {len(actual)}"

    bad: List[str] = []
    for i, (want, got) in enumerate(zip(_compare.sort_rows(expected), _compare.sort_rows(actual))):
        if len(want) != len(got):
            return f"row {i}: expected {len(want)} columns, got {len(got)}"
        
        bad += [
            f"row {i} col {j}: expected {w!r}, got {g!r}"
            for j, (w, g) in enumerate(zip(want, got))
            if not _compare.cells_match(w, g, tolerance)
        ]
    return f"{len(bad)} cell(s) differ, first at {bad[0]}" if bad else None


def check_folders(sf: str, folders: Sequence[Path], run_sql: Callable[[str], List[tuple]]) -> List[CheckRow]:
    answers: Dict[str, Union[List[List[str]], str]] = {}  # SQLite's rows, or its error

    def answer(sql: str) -> Union[List[List[str]], str]:
        if sql not in answers:
            try:
                answers[sql] = [sqlite_cells(row) for row in run_sql(sql)]
            except sqlite3.Error as e:
                answers[sql] = f"sqlite error: {e}"
        return answers[sql]

    results: List[CheckRow] = []
    for folder in folders:
        for item in load_items(folder, sf):
            expected = answer(item.sql)
            detail = expected if isinstance(expected, str) else compare_rows(expected, item.rows)
            results.append(CheckRow(sf, folder.name, item.seq, item.query, "PASS" if detail is None else "FAIL", detail or ""))
    return results


def open_sqlite(data_dir: Path, sf: str) -> sqlite3.Connection:
    sf_dir = data_dir / f"sf{sf}"
    if not sf_dir.is_dir():
        raise FileNotFoundError(f"no TPC-H data folder {sf_dir}")

    conn = sqlite3.connect(":memory:")
    _sqlite_bench.setup_database(conn, str(sf_dir))
    # The lookup workloads filter on these two dates.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_lineitem_shipdate ON lineitem(l_shipdate)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_orderdate ON orders(o_orderdate)")
    conn.commit()
    return conn


def write_check_csv(path: Path, rows: Sequence[CheckRow]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sf", "config", "seq", "query", "status"])
        writer.writerows([r.sf, r.config, r.seq, r.query, r.status] for r in rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check workload answers against SQLite.")
    parser.add_argument("--sf", required=True, choices=SCALE_FACTORS)
    parser.add_argument("--data-dir", type=Path, default=Path("bench/tpch_data"), help="TPC-H data root holding sf0.01/, sf0.1/, sf1.0/")
    parser.add_argument("--output", type=Path, required=True, help="check CSV to write")
    parser.add_argument("folders", nargs="+", type=Path, metavar="FOLDER", help="workload result folders")
    args = parser.parse_args(argv)
    if len({folder.name for folder in args.folders}) != len(args.folders):
        parser.error("folder names must be different (the check CSV names a run by its folder)")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    conn = open_sqlite(args.data_dir, args.sf)
    rows = check_folders(args.sf, args.folders, lambda sql: conn.execute(sql).fetchall())
    write_check_csv(args.output, rows)

    failed = False
    for folder in args.folders:
        mine = [r for r in rows if r.config == folder.name]
        print(f"PASS {sum(r.status == 'PASS' for r in mine)}/{len(mine)} [{folder.name}]")
        for r in (r for r in mine if r.status == "FAIL"):
            failed = True
            print(f"  FAIL seq {r.seq} {r.query}: {r.detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
