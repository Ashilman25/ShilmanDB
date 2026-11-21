#!/usr/bin/env python3
#The bar report: ShilmanDB's parallel executor against DuckDB and Postgres, per query.

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
CHECK_TOLERANCE = 1e-9      # relative
P, P1 = "columnar+parallel", "columnar+parallel_1thread"
SWEEP_RE = re.compile(r"columnar\+parallel_(\d+)thread")
HEADER: List[str] = ["query", "P", "P1", "D", "D1", "PG", "PG1", "P_over_PG", "P_over_D", "P1_over_D1", "ours_speedup", "duckdb_speedup", "check", "P_over_PG1", "P1_over_PG1"]


# Loads run_benchmarks.py by path, for its scale factors and median reader.
def _load_runner() -> ModuleType:
    name = "run_benchmarks"
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


runner = _load_runner()


# A needed file or config is missing, or a median is zero: exit 2.
class MissingInput(Exception):
    pass


@dataclass
class BarRow:
    query: str
    p: float
    p1: float
    d: float
    d1: float
    pg: float
    pg1: float
    sweep: Dict[int, float] = field(default_factory=dict)

    @property
    def p_over_pg(self) -> float:
        return self.p / self.pg

    @property
    def p_over_d(self) -> float:
        return self.p / self.d

    @property
    def p1_over_d1(self) -> float:
        return self.p1 / self.d1

    @property
    def ours_speedup(self) -> float:
        return self.p1 / self.p

    @property
    def duckdb_speedup(self) -> float:
        return self.d1 / self.d

    # The ratio to DuckDB, multiplied back from its parts.
    @property
    def check(self) -> float:
        return self.p1_over_d1 * self.duckdb_speedup / self.ours_speedup

    # All our threads against one Postgres worker.
    @property
    def p_over_pg1(self) -> float:
        return self.p / self.pg1

    # One thread each: speed per core against Postgres.
    @property
    def p1_over_pg1(self) -> float:
        return self.p1 / self.pg1


@dataclass
class BarTable:
    rows: List[BarRow]
    sweep_threads: List[int]    # ascending


# ShilmanDB medians at one SF from latencies.csv: {(query, config): ms}, plus the query order.
def read_ours(path: Path, sf: str) -> Tuple[Dict[Tuple[str, str], float], List[str]]:
    if not path.exists():
        raise MissingInput(f"missing {path}")
    
    medians: Dict[Tuple[str, str], float] = {}
    queries: List[str] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["sf"] != sf:
                continue
            if row["query"] not in queries:
                queries.append(row["query"])
            if row["median_latency_ms"]:
                medians[(row["query"], row["config"])] = float(row["median_latency_ms"])
    return medians, queries


# Medians of the per-run rows of a duckdb_ or postgres_latencies.csv: {(query, config): ms}.
def read_engine(path: Path, sf: str) -> Dict[Tuple[str, str], float]:
    if not path.exists():
        raise MissingInput(f"missing {path}")
    stats = runner.parse_engine_latencies(path, [sf])
    return {(query, config): qs.median_latency_ms for (_sf, config, query), qs in stats.items()}


# The bar rows for one SF, in latencies.csv's query order. Raises MissingInput.
def build_table(results_dir: Path, sf: str) -> BarTable:
    ours, queries = read_ours(results_dir / "latencies.csv", sf)
    if not queries:
        raise MissingInput(f"latencies.csv has no rows at SF={sf}")
    medians = dict(ours)
    medians.update(read_engine(results_dir / "duckdb_latencies.csv", sf))
    medians.update(read_engine(results_dir / "postgres_latencies.csv", sf))

    sweep_threads = sorted({int(m.group(1)) for (_q, config) in ours if (m := SWEEP_RE.fullmatch(config)) and config != P1})

    needed = [P, P1, "duckdb", "duckdb_1thread", "postgres", "postgres_1worker"]
    missing = [f"{config} for {q}" for q in queries for config in needed if (q, config) not in medians]
    if missing:
        raise MissingInput("missing " + ", ".join(missing))

    # A 4-decimal median can round to zero on a tiny run, and the ratios divide by it
    swept = [f"columnar+parallel_{n}thread" for n in sweep_threads]
    zeros = [f"{config} for {q}" for q in queries for config in needed + swept if medians.get((q, config)) == 0.0]
    if zeros:
        raise MissingInput("zero median for " + ", ".join(zeros))

    rows = []
    for q in queries:
        p, p1, d, d1, pg, pg1 = (medians[(q, config)] for config in needed)
        sweep = {n: medians[(q, f"columnar+parallel_{n}thread")] for n in sweep_threads if (q, f"columnar+parallel_{n}thread") in medians}
        rows.append(BarRow(q, p, p1, d, d1, pg, pg1, sweep))
    return BarTable(rows, sweep_threads)


# A median in ms with 4 decimals, empty when there is none.
def _ms(x: Optional[float]) -> str:
    return "" if x is None else f"{x:.4f}"


# A ratio with 6 decimals.
def _ratio(x: float) -> str:
    return f"{x:.6f}"


# The CSV's cells, header first.
def table_cells(table: BarTable) -> List[List[str]]:
    cells = [HEADER + [f"P{n}" for n in table.sweep_threads]]
    for r in table.rows:
        cells.append([
            r.query, _ms(r.p), _ms(r.p1), _ms(r.d), _ms(r.d1), _ms(r.pg), _ms(r.pg1),
            _ratio(r.p_over_pg), _ratio(r.p_over_d), _ratio(r.p1_over_d1),
            _ratio(r.ours_speedup), _ratio(r.duckdb_speedup), _ratio(r.check),
            _ratio(r.p_over_pg1), _ratio(r.p1_over_pg1)
            ] + [_ms(r.sweep.get(n)) for n in table.sweep_threads])
    return cells


# The queries whose check doesn't multiply back to P_over_D.
def bad_checks(table: BarTable) -> List[str]:
    return [r.query for r in table.rows if abs(r.check - r.p_over_d) > CHECK_TOLERANCE * abs(r.p_over_d)]


# Prints the cells as right-aligned columns.
def print_table(cells: List[List[str]]) -> None:
    widths = [max(len(row[i]) for row in cells) for i in range(len(cells[0]))]
    for row in cells:
        print("  ".join(cell.rjust(width) for cell, width in zip(row, widths)))


# The command line, or argv when a test passes one.
def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="The bar table: parallel ShilmanDB against DuckDB and Postgres", epilog="Exit 2: a needed config or file is missing. Exit 1: check differs from P_over_D.")
    p.add_argument("--results-dir", type=Path, required=True, help="A run_benchmarks.py output folder")
    p.add_argument("--sf", required=True, choices=runner.KNOWN_SCALE_FACTORS, help="Scale factor, as the runner spells it")
    p.add_argument("--out", type=Path, default=None, help="Output CSV (default: bar_sf<SF>.csv in --results-dir)")
    return p.parse_args(argv)


# Builds, checks, prints and writes the table. Returns the exit code.
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        table = build_table(args.results_dir, args.sf)
    except MissingInput as e:
        print(f"! {e}", file=sys.stderr)
        return 2

    cells = table_cells(table)
    print_table(cells)
    bad = bad_checks(table)
    if bad:
        print(f"! check differs from P_over_D for {', '.join(bad)}", file=sys.stderr)
        return 1

    out = args.out or args.results_dir / f"bar_sf{args.sf}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        csv.writer(f).writerows(cells)
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
