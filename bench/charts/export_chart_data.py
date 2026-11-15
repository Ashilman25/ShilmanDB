#!/usr/bin/env python3
"""Export small chart-input CSVs from result folders into docs/data/.

Reads only: nothing under the results folders is ever written. Each export runs
only when its input exists, so a fresh clone without bench/results/ skips them.

  baseline_sf{1|10}.csv, environment_sf{1|10}.csv  from run_benchmarks.py output
  eviction_misses.csv   SF=1.0 +index stream/mixed misses (LRU, ARC, LIRS, V2, Belady)
  ceilings.csv          join order ÷ best (SF=1.0) and eviction ÷ Belady (SF=0.01)
  analyst_bars.csv      the three bars from the analyst check's report.md

Usage:
  .venv/bin/python bench/charts/export_chart_data.py --out docs/data \
      [--results-dir bench/results] [--join-dir DIR] \
      [--baseline-dir-sf1 DIR] [--baseline-dir-sf10 DIR]
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"

BASELINE_COLUMNS = ["query", "engine", "config", "median_ms", "status", "max_abs_diff", "max_rel_diff"]
EVICTION_COLUMNS = ["sf", "kind", "pool_fraction", "policy", "misses", "vs_belady"]
CEILING_COLUMNS = ["part", "sf", "item", "method", "ratio"]
ANALYST_COLUMNS = ["bar", "count", "total", "rate", "wilson_low", "wilson_high", "needs"]

# The engine's own ARC and LIRS (counting page visits), as run in the engine.
EVICTION_POLICIES = ["lru", "arc_visit", "lirs_visit", "v2", "belady"]
EVICTION_WORKLOADS = ["stream", "mixed"]
DECISION_POOLS = ["10%", "25%"]
SHILMANDB_CONFIGS = {"baseline", "+index", "+vectorized", "+learned"}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, columns: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def same_sf(a: str, b: str) -> bool:
    return float(a) == float(b)


# Which engine a run_benchmarks config belongs to.
def engine_of(config: str) -> str:
    if config in SHILMANDB_CONFIGS:
        return "shilmandb"
    for engine in ("sqlite", "duckdb", "postgres"):
        if config.startswith(engine):
            return engine
    return "other"


# A time is published only if its answer was checked: PASS, DRIFT, or SQLite's own SKIP.
# SQLite is the reference only at SF <= 1; at SF=10 its SKIP was never checked.
def publishable(config: str, status: str, sf: str) -> bool:
    sqlite_reference = engine_of(config) == "sqlite" and float(sf) <= 1
    return status in ("PASS", "DRIFT") or (status == "SKIP" and sqlite_reference)


class StaleResultsError(ValueError):
    """A baseline folder whose answer statuses don't belong to its times."""


# Refuses a folder whose correctness.csv is missing or older than its latencies.csv.
def check_statuses_current(results_dir: Path) -> None:
    latencies, correctness = results_dir / "latencies.csv", results_dir / "correctness.csv"
    if not correctness.exists():
        raise StaleResultsError(f"{results_dir}: correctness.csv is missing, so no time can be checked")
    
    if correctness.stat().st_mtime < latencies.stat().st_mtime:
        raise StaleResultsError(
            f"{results_dir}: correctness.csv is older than latencies.csv "
            "(left from an earlier run? copy result folders with cp -p, "
            "which keeps file times)"
        )


def largest(values: Iterable[str]) -> str:
    filled = [v for v in values if v not in ("", None)]
    return max(filled, key=float) if filled else ""


# One row per query and config: time, answer status and the diffs behind that status.
def export_baseline(results_dir: Path, sf: str, out: Path) -> Path:
    check_statuses_current(results_dir)
    latencies = [r for r in read_csv(results_dir / "latencies.csv") if same_sf(r["sf"], sf)]
    status = {(r["query"], r["config"]): r["status"] for r in read_csv(results_dir / "correctness.csv") if same_sf(r["sf"], sf)}
    checks_path = results_dir / "reference_checks.csv"
    checks = [r for r in read_csv(checks_path) if same_sf(r["sf"], sf)] if checks_path.exists() else []

    rows = []
    for r in latencies:
        key = (r["query"], r["config"])
        st = status.get(key, "")
        matching = [c for c in checks if (c["query"], c["config"]) == key and c["status"] == st]
        rows.append({
            "query": r["query"],
            "engine": engine_of(r["config"]),
            "config": r["config"],
            "median_ms": r["median_latency_ms"] if publishable(r["config"], st, sf) else "",
            "status": st,
            "max_abs_diff": largest(c["max_abs_diff"] for c in matching),
            "max_rel_diff": largest(c["max_rel_diff"] for c in matching),
        })

    path = out / f"baseline_sf{sf}.csv"
    write_csv(path, BASELINE_COLUMNS, rows)
    env = results_dir / "environment.csv"
    if env.exists():
        shutil.copyfile(env, out / f"environment_sf{sf}.csv")
    return path


def eviction_csv(results_dir: Path) -> Path:
    return results_dir / "headroom" / "eviction" / "eviction_headroom.csv"


# SF=1.0 +index misses per pool size for the five policies, stream and mixed.
def export_eviction(out: Path, results_dir: Path = RESULTS_DIR) -> Path:
    rows = []
    for r in read_csv(eviction_csv(results_dir)):
        if (same_sf(r["sf"], "1.0") and r["config"] == "+index" and r["workload"] in EVICTION_WORKLOADS and r["policy"] in EVICTION_POLICIES):
            rows.append({
                "sf": "1.0",
                "kind": f"index_{r['workload']}",
                "pool_fraction": r["pool_fraction"],
                "policy": r["policy"],
                "misses": r["misses"],
                "vs_belady": r["vs_belady"],
            })
    path = out / "eviction_misses.csv"
    write_csv(path, EVICTION_COLUMNS, rows)
    return path


# Join: chosen ÷ best order at SF=1.0. Eviction: best classical and V2 ÷ Belady at SF=0.01.
def export_ceilings(out: Path, results_dir: Path = RESULTS_DIR, join_dir: Optional[Path] = None) -> Path:
    join_dir = join_dir or results_dir / "headroom" / "join_output_after"
    rows: List[Dict[str, object]] = []
    for r in read_csv(join_dir / "summary.csv"):
        if not same_sf(r["sf"], "1.0"):
            continue
        rows.append({
            "part": "join", 
            "sf": "1.0", 
            "item": r["query"], 
            "method": "classical",
            "ratio": r["classical_over_best"]
        })
        
        rows.append({
            "part": "join", 
            "sf": "1.0", 
            "item": r["query"], 
            "method": "learned",
            "ratio": r["learned_over_best"]
        })

    # SF=0.01 is the only size where V2 ran on all four decision cells.
    cells: Dict[tuple, Dict[str, Dict[str, str]]] = {}
    ev_path = eviction_csv(results_dir)
    for r in read_csv(ev_path) if ev_path.exists() else []:
        if (same_sf(r["sf"], "0.01") and r["config"] == "+index" and r["workload"] in EVICTION_WORKLOADS and r["pool_label"] in DECISION_POOLS):
            cells.setdefault((r["workload"], r["pool_label"]), {})[r["policy"]] = r

    for workload in EVICTION_WORKLOADS:
        for pool in DECISION_POOLS:
            policies = cells.get((workload, pool))
            if not policies:
                continue
            
            any_row = next(iter(policies.values()))
            best = policies.get(any_row["best_classical"])
            item = f"{workload} {pool}"
            if best:
                rows.append({
                    "part": "eviction", 
                    "sf": "0.01", "item": item,
                    "method": "best_classical", "ratio": best["vs_belady"]
                })
                
            if "v2" in policies:
                rows.append({
                    "part": "eviction", "sf": "0.01", "item": item,
                    "method": "v2", "ratio": policies["v2"]["vs_belady"]
                })
    path = out / "ceilings.csv"
    write_csv(path, CEILING_COLUMNS, rows)
    return path


BAR_ROW = re.compile(
    r"^\|\s*(?P<bar>[^|]+?)\s*\|\s*(?P<count>\d+) of (?P<total>\d+)\s*\|\s*(?P<rate>[\d.]+)\s*"
    r"\|\s*(?P<low>[\d.]+)-(?P<high>[\d.]+)\s*\|\s*(?P<needs>[^|]+?)\s*\|")


# The three bars from the "## The bars" table of the analyst check's report.
def export_analyst(out: Path, results_dir: Path = RESULTS_DIR) -> Path:
    text = (results_dir / "analyst_check" / "report.md").read_text()
    section = text.split("## The bars", 1)[1].split("\n#", 1)[0]
    rows = []
    for line in section.splitlines():
        m = BAR_ROW.match(line)
        if m:
            rows.append({
                "bar": m["bar"], "count": m["count"], "total": m["total"],
                "rate": m["rate"], "wilson_low": m["low"], "wilson_high": m["high"],
                "needs": m["needs"]
            })
            
    if len(rows) != 3:
        raise ValueError(f"expected 3 bars in the analyst report, found {len(rows)}")
    path = out / "analyst_bars.csv"
    write_csv(path, ANALYST_COLUMNS, rows)
    return path


# The input folder that out falls inside (results, bench/results/ or an extra), else None.
def out_under_results(out: Path, results_dir: Path, extra: Sequence[Path] = ()) -> Optional[Path]:
    target = out.resolve()
    for folder in (results_dir, RESULTS_DIR, *extra):
        folder = folder.resolve()
        if target == folder or folder in target.parents:
            return folder
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--join-dir", type=Path, default=None, help="join_headroom output (default <results-dir>/headroom/join_output_after)")
    parser.add_argument("--baseline-dir-sf1", type=Path, default=None)
    parser.add_argument("--baseline-dir-sf10", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "data")
    args = parser.parse_args(argv)

    results = args.results_dir
    inputs = [d for d in (args.baseline_dir_sf1, args.baseline_dir_sf10, args.join_dir) if d]
    bad = out_under_results(args.out, results, inputs)
    if bad:
        print(f"error: --out {args.out} is under {bad}; I never write into result folders", file=sys.stderr)
        return 2

    # Stale or missing statuses stop everything before any file is written.
    for folder in (args.baseline_dir_sf1, args.baseline_dir_sf10):
        if folder and (folder / "latencies.csv").exists():
            try:
                check_statuses_current(folder)
            except StaleResultsError as e:
                print(f"error: {e}", file=sys.stderr)
                return 2

    join_dir = args.join_dir or results / "headroom" / "join_output_after"
    jobs = [
        ("baseline SF=1", args.baseline_dir_sf1 and args.baseline_dir_sf1 / "latencies.csv", lambda: export_baseline(args.baseline_dir_sf1, "1", args.out)),
        ("baseline SF=10", args.baseline_dir_sf10 and args.baseline_dir_sf10 / "latencies.csv", lambda: export_baseline(args.baseline_dir_sf10, "10", args.out)),
        ("eviction", eviction_csv(results), lambda: export_eviction(args.out, results)),
        ("ceilings", join_dir / "summary.csv", lambda: export_ceilings(args.out, results, join_dir)),
        ("analyst", results / "analyst_check" / "report.md", lambda: export_analyst(args.out, results)),
    ]
    for name, needed, run in jobs:
        if not needed:
            continue
        if not needed.exists():
            print(f"skip {name}: {needed} not found")
            continue
        print(f"{name} -> {run()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
