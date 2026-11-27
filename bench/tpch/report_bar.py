#!/usr/bin/env python3
#The bar report: ShilmanDB's parallel executor against DuckDB and Postgres, per query.

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
CHECK_TOLERANCE = 1e-9      # relative
P, P1 = "columnar+parallel", "columnar+parallel_1thread"
PS, PD, D20 = "columnar+parallel_spill", "columnar+parallel_direct", "duckdb_20g"
SWEEP_RE = re.compile(r"columnar\+parallel_(\d+)thread")
HEADER: List[str] = ["query", "P", "P1", "D", "D1", "PG", "PG1", "P_over_PG", "P_over_D", "P1_over_D1", "ours_speedup", "duckdb_speedup", "check", "P_over_PG1", "P1_over_PG1"]

# Where the report also checks answers and adds spill, direct, 20g and memory columns
FULL_REPORT_SFS: List[str] = ["100"]
EXTRA_HEADER: List[str] = ["PS", "PS_over_P", "PS_spill_bytes", "PD", "PD_over_P", "D20", "P_over_D20"]
MEMORY_HEADER: List[str] = ["config", "pool_size", "memory_limit", "peak_rss_bytes", "spill_bytes_max"]


# Loads a script by path once, keeping it in sys.modules for its dataclasses.
def _load_module(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[name]
        raise
    return module


# The runner gives the scale factors and median reader; compare_results the answer rules
runner = _load_module("run_benchmarks", HERE / "run_benchmarks.py")
compare = _load_module("compare_results", HERE.parents[1] / "scripts" / "compare_results.py")


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
    # Only at SF=100, and None where the config didn't run
    ps: Optional[float] = None
    ps_spill: Optional[int] = None
    pd: Optional[float] = None
    d20: Optional[float] = None

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

    # The ShilmanDB configs this row shows a time for, in column order.
    def our_configs(self) -> List[str]:
        swept = [f"columnar+parallel_{n}thread" for n in sorted(self.sweep)]
        return [P, P1] + swept + [c for c, ms in ((PS, self.ps), (PD, self.pd)) if ms is not None]


@dataclass
class BarTable:
    rows: List[BarRow]
    sweep_threads: List[int]    # ascending
    extras: bool = False        # the SF=100 columns
    notes: List[str] = field(default_factory=list)  # what the optional configs lack


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


# Each config's pool_size at one SF from latencies.csv, in file order.
def read_pools(path: Path, sf: str) -> Dict[str, str]:
    pools: Dict[str, str] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["sf"] == sf and not pools.get(row["config"]):
                pools[row["config"]] = row.get("pool_size") or ""
    return pools


# spill_bytes_written per query from a parallel_stats file, one value per timed run.
def read_spills(path: Path) -> Dict[str, List[int]]:
    spills: Dict[str, List[int]] = {}
    if not path.exists():
        return spills
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("spill_bytes_written"):
                spills.setdefault(row["query"], []).append(int(row["spill_bytes_written"]))
    return spills


# What the optional configs lack, one note per config.
def optional_notes(medians: Dict[Tuple[str, str], float], queries: List[str], sf: str, spill_path: Path, spills: Dict[str, List[int]]) -> List[str]:
    notes = []
    for config in (PS, PD, D20):
        absent = [q for q in queries if (q, config) not in medians]
        if len(absent) == len(queries):
            notes.append(f"no {config} at SF={sf}; its columns are empty")
        elif absent:
            notes.append(f"no {config} for {', '.join(absent)}; those cells are empty")
    unspilled = [q for q in queries if (q, PS) in medians and q not in spills]
    if unspilled:
        notes.append(f"{spill_path.name} has no spill_bytes_written for {', '.join(unspilled)}; PS_spill_bytes is empty there")
    return notes


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
    extras = [PS, PD, D20] if sf in FULL_REPORT_SFS else []
    zeros = [f"{config} for {q}" for q in queries for config in needed + swept + extras if medians.get((q, config)) == 0.0]
    if zeros:
        raise MissingInput("zero median for " + ", ".join(zeros))

    spill_path = results_dir / f"parallel_stats_{PS}_sf{sf}.csv"
    spills = read_spills(spill_path) if extras else {}
    rows = []
    for q in queries:
        p, p1, d, d1, pg, pg1 = (medians[(q, config)] for config in needed)
        sweep = {n: medians[(q, f"columnar+parallel_{n}thread")] for n in sweep_threads if (q, f"columnar+parallel_{n}thread") in medians}
        row = BarRow(q, p, p1, d, d1, pg, pg1, sweep)
        if extras:
            row.ps, row.pd, row.d20 = (medians.get((q, config)) for config in extras)
            # The lower middle on an even count, so the bytes stay a whole number
            if row.ps is not None and q in spills:
                row.ps_spill = statistics.median_low(spills[q])
        rows.append(row)

    notes = optional_notes(medians, queries, sf, spill_path, spills) if extras else []
    return BarTable(rows, sweep_threads, bool(extras), notes)


# A median in ms with 4 decimals, empty when there is none.
def _ms(x: Optional[float]) -> str:
    return "" if x is None else f"{x:.4f}"


# A ratio with 6 decimals.
def _ratio(x: float) -> str:
    return f"{x:.6f}"


# a / b with 6 decimals, empty when a is missing.
def _opt_ratio(a: Optional[float], b: float) -> str:
    return "" if a is None else _ratio(a / b)


# A whole number, empty when there is none.
def _count(x: Optional[int]) -> str:
    return "" if x is None else str(x)


# The SF=100 cells of one row, in EXTRA_HEADER's order.
def _extra_cells(r: BarRow) -> List[str]:
    p_over_d20 = "" if r.d20 is None else _ratio(r.p / r.d20)
    return [_ms(r.ps), _opt_ratio(r.ps, r.p), _count(r.ps_spill), _ms(r.pd), _opt_ratio(r.pd, r.p), _ms(r.d20), p_over_d20]


# The CSV's cells, header first.
def table_cells(table: BarTable) -> List[List[str]]:
    cells = [HEADER + [f"P{n}" for n in table.sweep_threads] + (EXTRA_HEADER if table.extras else [])]
    for r in table.rows:
        cells.append([
            r.query, _ms(r.p), _ms(r.p1), _ms(r.d), _ms(r.d1), _ms(r.pg), _ms(r.pg1),
            _ratio(r.p_over_pg), _ratio(r.p_over_d), _ratio(r.p1_over_d1),
            _ratio(r.ours_speedup), _ratio(r.duckdb_speedup), _ratio(r.check),
            _ratio(r.p_over_pg1), _ratio(r.p1_over_pg1)
            ] + [_ms(r.sweep.get(n)) for n in table.sweep_threads] + (_extra_cells(r) if table.extras else []))
    return cells


# Memory table order: P, P1, the sweep, spill, direct, then the rest.
def _memory_order(config: str) -> Tuple[int, int]:
    named = {P: (0, 0), P1: (1, 0), PS: (3, 0), PD: (4, 0)}
    m = SWEEP_RE.fullmatch(config)
    return named.get(config) or ((2, int(m.group(1))) if m else (5, 0))


# One row per ShilmanDB config: pool, memory limit, peak RSS and largest spill.
def memory_cells(results_dir: Path, sf: str) -> List[List[str]]:
    pools = read_pools(results_dir / "latencies.csv", sf)
    ours = sorted((c for c in pools if runner.engine_of(c) == "shilmandb"), key=_memory_order)
    cells = [MEMORY_HEADER]
    for config in ours:
        info = runner.read_run_info(results_dir / runner.RESULTS_DIRS["shilmandb"] / config / f"run_info_sf{sf}.csv")
        spilled = [b for runs in read_spills(results_dir / f"parallel_stats_{config}_sf{sf}.csv").values() for b in runs]
        cells.append([config, pools[config], info.get("memory_limit", ""), info.get("peak_rss_bytes", ""),
                      str(max(spilled)) if spilled else ""])
    return cells


# What reading an answer file can raise; an empty one raises StopIteration
READ_ERRORS = (StopIteration, OSError, ValueError, csv.Error)


# The first unreadable file of paths and why, or None when all read.
def _unreadable(paths: Sequence[Path]) -> Optional[str]:
    for path in paths:
        try:
            compare.load_csv(str(path))
        except READ_ERRORS as e:
            why = "it's empty" if isinstance(e, StopIteration) else repr(e)
            return f"couldn't read {path}: {why}"
    return None


# One check, with no drift allowed. An unreadable file is a FAIL, not a crash.
def _check_answer(ref_dir: Path, cand_dir: Path, config: str, sf: str, query: str, ref_name: str) -> Optional[str]:
    name = f"{query}_sf{sf}.csv"
    cand, ref = cand_dir / name, ref_dir / name
    try:
        cr = compare.compare_one(str(ref), str(cand), config, sf, query, compare.DEFAULT_TOLERANCE, ref_name, no_drift=True)
    except READ_ERRORS:
        why = _unreadable([cand, ref]) or f"couldn't compare {cand} with {ref}"
        return f"{query} {config} against {ref_name}: FAIL ({why})"
    if cr.passed:
        return None
    why = cr.error or (f"{len(cr.mismatches)} cell(s) differ, max abs {cr.max_abs_diff:.3g}, max rel {cr.max_rel_diff:.3g}")
    return f"{query} {config} against {ref_name}: {cr.status} ({why})"


# Our shown configs against DuckDB and Postgres; duckdb and duckdb_20g against Postgres.
# D1, PG1 and leftover folders aren't checked. One line per check that isn't PASS.
def answer_problems(results_dir: Path, sf: str, table: BarTable) -> List[str]:
    refs = [(runner.REFERENCE_NAMES[e], results_dir / runner.RESULTS_DIRS[e] / e) for e in ("duckdb", "postgres")]
    ours_dir = results_dir / runner.RESULTS_DIRS["shilmandb"]
    duck_dir = results_dir / runner.RESULTS_DIRS["duckdb"]
    problems: List[str] = []
    for row in table.rows:
        checks = [(ours_dir, c, refs) for c in row.our_configs()]
        checks += [(duck_dir, c, refs[1:]) for c in ["duckdb"] + ([D20] if row.d20 is not None else [])]
        for folder, config, against in checks:
            for ref_name, ref_dir in against:
                problem = _check_answer(ref_dir, folder / config, config, sf, row.query, ref_name)
                if problem:
                    problems.append(problem)
    return problems


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
    p = argparse.ArgumentParser(description="The bar table: parallel ShilmanDB against DuckDB and Postgres", epilog="Exit 2: a needed config or file is missing. Exit 1: check differs from P_over_D, or at SF=100 an answer isn't PASS against DuckDB and Postgres.")
    p.add_argument("--results-dir", type=Path, required=True, help="A run_benchmarks.py output folder")
    p.add_argument("--sf", required=True, choices=runner.KNOWN_SCALE_FACTORS, help="Scale factor, as the runner spells it")
    p.add_argument("--out", type=Path, default=None, help="Output CSV (default: bar_sf<SF>.csv in --results-dir); at SF=100 the memory table goes beside it as <name>_memory.csv")
    return p.parse_args(argv)


# Builds, checks, prints and writes the table. Returns the exit code.
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        table = build_table(args.results_dir, args.sf)
    except MissingInput as e:
        print(f"! {e}", file=sys.stderr)
        return 2

    for note in table.notes:
        print(f"note: {note}", file=sys.stderr)
    memory: List[List[str]] = []
    if table.extras:
        problems = answer_problems(args.results_dir, args.sf, table)
        if problems:
            for problem in problems:
                print(f"! {problem}", file=sys.stderr)
            print(f"! {len(problems)} answer check(s) aren't PASS at SF={args.sf}, so nothing was written", file=sys.stderr)
            return 1
        memory = memory_cells(args.results_dir, args.sf)

    cells = table_cells(table)
    print_table(cells)
    if memory:
        print()
        print_table(memory)
    bad = bad_checks(table)
    if bad:
        print(f"! check differs from P_over_D for {', '.join(bad)}", file=sys.stderr)
        return 1

    out = args.out or args.results_dir / f"bar_sf{args.sf}.csv"
    outputs = [(out, cells)] + ([(out.with_name(f"{out.stem}_memory{out.suffix}"), memory)] if memory else [])
    for path, rows in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
