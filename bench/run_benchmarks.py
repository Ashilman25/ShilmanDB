#!/usr/bin/env python3
"""
Benchmark runner

Runs a matrix of (scale factor x configuration x query) benchmarks, then
emits three unified CSVs under bench/results/:

- latencies.csv: (sf, query, config, median_latency_ms, rows, hits, misses)
- hit_rates.csv: (sf, config, pool_size, hit_rate, improvement_over_lru)
- correctness.csv: (sf, query, config, status)

"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Constants 

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BINARY = REPO_ROOT / "build" / "bench" / "load_tpch"
DEFAULT_DATA_DIR = REPO_ROOT / "bench" / "tpch_data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "bench" / "results"

ALL_SCALE_FACTORS: List[str] = ["0.01", "0.1", "1.0"]
QUERIES: List[str] = ["Q1", "Q3", "Q5", "Q6", "Q10", "Q12", "Q14", "Q19"]

POOL_SIZES: Dict[str, int] = {
    "0.01": 128,
    "0.1": 512,
    "1.0": 2048,
}

POOL_SWEEP_SIZES: List[int] = [128, 256, 512, 1024, 2048]
POOL_SWEEP_SF: str = "0.1"

JOIN_MODEL_PATH = "ml/join_optimizer/models/join_order_model.pt"
EVICTION_V2_MODEL_PATH = "ml/eviction_policy/models/eviction_model_v2.pt"
EVICTION_V1_MODEL_PATH = "ml/eviction_policy/models/eviction_model.pt"

# Config name -> list of CLI flags to pass to load_tpch.
PRIMARY_CONFIGS: Dict[str, List[str]] = {
    "baseline": [
        "--create-indexes", "false",
        "--execution-mode", "tuple",
    ],
    "+index": [
        "--create-indexes", "true",
        "--execution-mode", "tuple",
    ],
    "+learned": [
        "--create-indexes", "true",
        "--execution-mode", "tuple",
        "--use-learned-join", "--join-model-path", JOIN_MODEL_PATH,
        "--use-learned-eviction",
        "--eviction-model-path", EVICTION_V2_MODEL_PATH,
        "--eviction-model-version", "v2",
    ],
    "+vectorized": [
        "--create-indexes", "true",
        "--execution-mode", "vectorized",
    ],
}

SHILMANDB_CONFIG_NAMES: List[str] = list(PRIMARY_CONFIGS.keys())
SQLITE_CONFIG_NAMES: List[str] = ["sqlite_disk", "sqlite_memory"]
ALL_CONFIG_NAMES: List[str] = SHILMANDB_CONFIG_NAMES + SQLITE_CONFIG_NAMES

#Helpers

#format seconds as Hh Mm Ss / Mm Ss / Ss.S.
def elapsed_str(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins, secs = divmod(int(seconds), 60)
    if mins < 60:
        return f"{mins}m {secs}s"
    hrs, mins = divmod(mins, 60)
    return f"{hrs}h {mins}m {secs}s"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


def subsection(title: str) -> None:
    print(f"\n--- {title} ---", flush=True)


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# Per-query aggregated results

@dataclass
class QueryStats:
    sf: str
    config: str
    query: str
    median_latency_ms: Optional[float]   # None => all runs failed
    rows: Optional[int]
    median_hits: Optional[int]
    median_misses: Optional[int]


@dataclass
class RunOutput:
    """All stats from a single load_tpch invocation."""
    per_query: Dict[str, QueryStats] = field(default_factory=dict)


def _median_int(values: Sequence[int]) -> int:
    """Median rounded to int (used for hit/miss counts)."""
    return int(round(statistics.median(values)))


#Parse the per-run latencies.csv written by load_tpch
# Schema: query,run,rows,latency_ms,hits,misses
# Returns: {query_tag: QueryStats}
def parse_shilmandb_latencies(path: Path, sf: str, config: str) -> Dict[str, QueryStats]:
    per_query_runs: Dict[str, List[Tuple[float, int, int, int]]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["query"]
            per_query_runs.setdefault(q, []).append((
                float(row["latency_ms"]),
                int(row["rows"]),
                int(row["hits"]),
                int(row["misses"]),
            ))

    result: Dict[str, QueryStats] = {}
    for q, runs in per_query_runs.items():
        if not runs:
            continue
        latencies = [r[0] for r in runs]
        rows_vals = [r[1] for r in runs]
        hits_vals = [r[2] for r in runs]
        misses_vals = [r[3] for r in runs]
        result[q] = QueryStats(
            sf=sf,
            config=config,
            query=q,
            median_latency_ms=statistics.median(latencies),
            rows=rows_vals[0],                     # stable across runs
            median_hits=_median_int(hits_vals),
            median_misses=_median_int(misses_vals),
        )
    return result


# ShilmanDB runner 
#Build the load_tpch CLI for a given config.
def build_command(binary: Path, sf: str, config_name: str, data_dir: Path, results_dir: Path,
                  db_file: Path, pool_size: int, warmup_runs: int, timed_runs: int) -> List[str]:
    
    if config_name not in PRIMARY_CONFIGS:
        raise ValueError(f"unknown ShilmanDB config: {config_name}")

    cmd: List[str] = [
        str(binary),
        "--sf", sf,
        "--db-file", str(db_file),
        "--data-dir", str(data_dir / f"sf{sf}"),
        "--pool-size", str(pool_size),
        "--warmup", str(warmup_runs),
        "--runs", str(timed_runs),
        "--results-dir", str(results_dir),
        "--no-verify",
    ]
    cmd += PRIMARY_CONFIGS[config_name]
    return cmd


#spawn load_tpch for one (sf, config) combo
#returns per query stats
def run_shilmandb_config(binary: Path, sf: str, config_name: str, data_dir: Path, output_dir: Path, pool_size: int, 
                         warmup_runs: int, timed_runs: int, timeout_seconds: Optional[int] = None) -> Optional[Dict[str, QueryStats]]:
    
    results_parent = output_dir / "shilmandb_results" / config_name
    results_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"shilmandb_{config_name}_sf{sf}_") as tmp:
        tmp_path = Path(tmp)
        db_file = tmp_path / f"tpch_sf{sf}.db"

        cmd = build_command(
            binary=binary,
            sf=sf,
            config_name=config_name,
            data_dir=data_dir,
            results_dir=tmp_path,
            db_file=db_file,
            pool_size=pool_size,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
        )

        print(f"  $ {' '.join(cmd)}", flush=True)
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            eprint(f"  ! TIMEOUT after {timeout_seconds}s")
            return None

        wall = time.time() - t0
        print(f"  (wall: {elapsed_str(wall)}, exit: {proc.returncode})", flush=True)

        if proc.returncode != 0:
            eprint(f"  ! load_tpch exited {proc.returncode}")
            eprint("  --- last 40 lines of stderr ---")
            for line in proc.stderr.splitlines()[-40:]:
                eprint(f"    {line}")
            return None

        lat_csv = tmp_path / "latencies.csv"
        if not lat_csv.exists():
            eprint(f"  ! missing {lat_csv}")
            return None

        # Copy per-query CSVs to the canonical config dir with the
        # sf-suffixed filename expected by compare_results.py.
        for query in QUERIES:
            src = tmp_path / f"{query}.csv"
            if src.exists():
                dst = results_parent / f"{query}_sf{sf}.csv"
                shutil.copyfile(src, dst)

        return parse_shilmandb_latencies(lat_csv, sf=sf, config=config_name)


# SQLite runner adapter
#Invoke bench/run_sqlite_benchmarks.py as a subprocess
def run_sqlite_benchmarks(scale_factors: Sequence[str], data_dir: Path, output_dir: Path, timed_runs: int) -> bool:
    subsection(f"SQLite benchmarks (SFs: {', '.join(scale_factors)})")
    script = REPO_ROOT / "bench" / "run_sqlite_benchmarks.py"
    if not script.exists():
        eprint(f"! missing {script}")
        return False

    cmd: List[str] = [
        sys.executable, str(script),
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
        "--runs", str(timed_runs),
    ]
    for sf in scale_factors:
        cmd += ["--sf", sf]

    print(f"  $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    print(f"  (wall: {elapsed_str(time.time() - t0)}, exit: {proc.returncode})", flush=True)
    return proc.returncode == 0


#Parse {output_dir}/sqlite_latencies.csv and produce a dict keyed by (sf, config, query)
def parse_sqlite_latencies(output_dir: Path, scale_factors: Sequence[str]) -> Dict[Tuple[str, str, str], QueryStats]:
    lat_path = output_dir / "sqlite_latencies.csv"
    if not lat_path.exists():
        eprint(f"! missing {lat_path}")
        return {}

    buckets: Dict[Tuple[str, str, str], List[float]] = {}
    with lat_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            sf = row["sf"]
            if sf not in scale_factors:
                continue
            mode = row["mode"]                 # 'memory' | 'disk'
            config = "sqlite_memory" if mode == "memory" else "sqlite_disk"
            key = (sf, config, row["query"])
            buckets.setdefault(key, []).append(float(row["latency_ms"]))

    out: Dict[Tuple[str, str, str], QueryStats] = {}
    for (sf, config, query), lats in buckets.items():
        out[(sf, config, query)] = QueryStats(
            sf=sf, config=config, query=query,
            median_latency_ms=statistics.median(lats),
            rows=None,
            median_hits=None,
            median_misses=None,
        )
    return out


#Secondary sweeps
#run baseline config at POOL_SWEEP_SF with each pool size
def run_pool_size_sweep(binary: Path, data_dir: Path, output_dir: Path, 
                        pool_sizes: Sequence[int], warmup_runs: int, timed_runs: int) -> Dict[Tuple[str, str, str], QueryStats]:
    
    sf = POOL_SWEEP_SF
    section(f"Pool-size sweep (SF={sf}, pool_sizes={list(pool_sizes)})")
    baseline_flags = list(PRIMARY_CONFIGS["baseline"])
    results: Dict[Tuple[str, str, str], QueryStats] = {}
    
    for ps in pool_sizes:
        config_label = f"baseline_pool{ps}"
        subsection(f"{config_label}")
        PRIMARY_CONFIGS[config_label] = baseline_flags  # shares flags; pool_size differs via --pool-size
        try:
            stats = run_shilmandb_config(
                binary=binary, sf=sf, config_name=config_label,
                data_dir=data_dir, output_dir=output_dir,
                pool_size=ps, warmup_runs=warmup_runs, timed_runs=timed_runs,
            )
            
        finally:
            del PRIMARY_CONFIGS[config_label]
            
        if stats is None:
            eprint(f"  pool-sweep {ps} failed — skipping")
            continue
        for query, qs in stats.items():
            results[(sf, config_label, query)] = qs
            
    return results


#run a +learned -like config but with V1 eviction model at each SF
def run_v1_eviction_sweep(binary: Path, data_dir: Path, output_dir: Path, 
                          scale_factors: Sequence[str], warmup_runs: int, timed_runs: int) -> Dict[Tuple[str, str, str], QueryStats]:
    
    section(f"V1 eviction sweep (SFs: {', '.join(scale_factors)})")
    v1_flags = [
        "--create-indexes", "true",
        "--execution-mode", "tuple",
        "--use-learned-join", "--join-model-path", JOIN_MODEL_PATH,
        "--use-learned-eviction",
        "--eviction-model-path", EVICTION_V1_MODEL_PATH,
        "--eviction-model-version", "v1",
    ]
    PRIMARY_CONFIGS["+learned_v1"] = v1_flags
    try:
        results: Dict[Tuple[str, str, str], QueryStats] = {}
        for sf in scale_factors:
            subsection(f"+learned_v1 @ SF={sf}")
            stats = run_shilmandb_config(
                binary=binary, sf=sf, config_name="+learned_v1",
                data_dir=data_dir, output_dir=output_dir,
                pool_size=POOL_SIZES[sf], warmup_runs=warmup_runs, timed_runs=timed_runs,
            )
            if stats is None:
                eprint(f"  +learned_v1 @ SF={sf} failed — skipping")
                continue
            for query, qs in stats.items():
                results[(sf, "+learned_v1", query)] = qs
    finally:
        del PRIMARY_CONFIGS["+learned_v1"]
    return results


#  Output writers

def _fmt_float(x: Optional[float]) -> str:
    return "" if x is None else f"{x:.4f}"


def _fmt_int(x: Optional[int]) -> str:
    return "" if x is None else str(x)

#write latencies.csv with schema: sf, query, config, median_latency_ms, rows, hits, misses
def write_latencies_csv(path: Path, records: Dict[Tuple[str, str, str], QueryStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    query_order = {q: i for i, q in enumerate(QUERIES)}
    keys = sorted(records.keys(), key=lambda k: (k[0], query_order.get(k[2], 999), k[1]))
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sf", "query", "config", "median_latency_ms", "rows", "hits", "misses"])
        for key in keys:
            qs = records[key]
            w.writerow([
                qs.sf, qs.query, qs.config,
                _fmt_float(qs.median_latency_ms),
                _fmt_int(qs.rows),
                _fmt_int(qs.median_hits),
                _fmt_int(qs.median_misses),
            ])
    print(f"  -> {path}")


#Aggregate per-query hit/miss counts into per-(sf,config) hit_rate and improvement_over_lru
def write_hit_rates_csv(path: Path, records: Dict[Tuple[str, str, str], QueryStats], pool_sizes_by_config: Dict[Tuple[str, str], int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    by_sc: Dict[Tuple[str, str], List[QueryStats]] = {}
    for (sf, config, _q), qs in records.items():
        by_sc.setdefault((sf, config), []).append(qs)

    rates: Dict[Tuple[str, str], Optional[float]] = {}
    for (sf, config), stats_list in by_sc.items():
        is_sqlite = config.startswith("sqlite_")
        if is_sqlite:
            rates[(sf, config)] = None
            continue
        
        total_hits = sum(s.median_hits for s in stats_list if s.median_hits is not None)
        total_misses = sum(s.median_misses for s in stats_list if s.median_misses is not None)
        total = total_hits + total_misses
        rates[(sf, config)] = (total_hits / total) if total > 0 else 0.0

    baseline_rates: Dict[str, Optional[float]] = {
        sf: rates.get((sf, "baseline")) for sf in {k[0] for k in rates}
    }

    def _config_rank(name: str) -> int:
        if name == "baseline": return 0
        if name.startswith("+"): return 1
        if name.startswith("baseline_pool"): return 2
        if name.startswith("sqlite_"): return 3
        return 4

    keys = sorted(rates.keys(), key=lambda k: (k[0], _config_rank(k[1]), k[1]))

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sf", "config", "pool_size", "hit_rate", "improvement_over_lru"])
        
        for (sf, config) in keys:
            rate = rates[(sf, config)]
            ps = pool_sizes_by_config.get((sf, config), "")
            if rate is None:
                w.writerow([sf, config, ps, "", ""])
                continue
            if config == "baseline":
                improvement = ""
            else:
                base = baseline_rates.get(sf)
                improvement = "" if base is None else f"{rate - base:.4f}"
            w.writerow([sf, config, ps, f"{rate:.4f}", improvement])

    print(f"  -> {path}")


#For each ShilmanDB config in configs, invoke scripts/compare_results.py and capture per-query PASS/FAIL/SKIP.
def write_correctness_csv(path: Path, sfs: Sequence[str], configs: Sequence[str], output_dir: Path, queries: Sequence[str] = tuple(QUERIES)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    comparable = [c for c in configs if not c.startswith("sqlite_")]
    sqlite_configs = [c for c in configs if c.startswith("sqlite_")]

    rows: List[Tuple[str, str, str, str]] = []

    for sf in sfs:
        for q in queries:
            for c in sqlite_configs:
                rows.append((sf, q, c, "SKIP"))

    script = REPO_ROOT / "scripts" / "compare_results.py"
    # compare_results.py print_results() emits per-comparison lines of the form:
    #   "  PASS  [<config>] <Qtag> SF=<sf>"
    #   "  FAIL  [<config>] <Qtag> SF=<sf>  (N mismatch(es))"
    #   "  SKIP  [<config>] <Qtag> SF=<sf>"
    # followed by indented mismatch/error detail lines. We scan each line,
    # strip leading whitespace, match the first token against {PASS,FAIL,SKIP},
    # then extract the Qtag (the token after the [config] bracket).
    import re
    line_re = re.compile(r"^(PASS|FAIL|SKIP)\s+\[(?P<config>[^\]]+)\]\s+(?P<query>Q\d+)\s+SF=(?P<sf>\S+)")

    for sf in sfs:
        for cfg in comparable:
            shilmandb_dir = output_dir / "shilmandb_results"
            sqlite_dir = output_dir / "sqlite_results"
            cmd = [
                sys.executable, str(script),
                "--shilmandb-dir", str(shilmandb_dir),
                "--sqlite-dir", str(sqlite_dir),
                "--sf", sf,
                "--configs", cfg,
            ]
            print(f"  $ {' '.join(cmd)}", flush=True)
            proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False,)
            seen: Dict[str, str] = {}
            
            for raw_line in proc.stdout.splitlines():
                m = line_re.match(raw_line.strip())
                if not m:
                    continue
                status = m.group(1)
                query = m.group("query")

                if query not in seen:
                    seen[query] = status
                    
            for q in queries:
                rows.append((sf, q, cfg, seen.get(q, "SKIP")))

    query_order = {q: i for i, q in enumerate(QUERIES)}
    rows.sort(key=lambda r: (r[0], query_order.get(r[1], 999), r[2]))

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sf", "query", "config", "status"])
        for row in rows:
            w.writerow(row)
    print(f"  -> {path}")


# ── CLI + main ────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 20.1 benchmark runner")
    p.add_argument("--binary", type=Path, default=DEFAULT_BINARY, help=f"Path to load_tpch binary (default: {DEFAULT_BINARY})")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help=f"TPC-H data dir (default: {DEFAULT_DATA_DIR})")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Output dir (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--sf", action="append", choices=ALL_SCALE_FACTORS + ["all"], help="Scale factor(s). Repeat flag or use 'all'. Default: 'all'.")
    p.add_argument("--configs", default="all", help=("Comma-separated configs. Accepts 'all' or a subset of: " + ",".join(ALL_CONFIG_NAMES)))
    p.add_argument("--warmup", type=int, default=1, help="Warmup runs per query (default 1)")
    p.add_argument("--runs", type=int, default=5, help="Timed runs per query (default 5)")
    p.add_argument("--pool-sweep", action="store_true", help=f"Also run baseline pool-size sweep at SF={POOL_SWEEP_SF}")
    p.add_argument("--v1-eviction", action="store_true", help="Also run +learned with V1 eviction model per SF")
    p.add_argument("--skip-sqlite", action="store_true", help="Skip the SQLite benchmark invocation (reuse existing output)")
    p.add_argument("--skip-correctness", action="store_true", help="Skip the correctness comparison step")
    p.add_argument("--timeout", type=int, default=None, help="Per-config subprocess timeout in seconds (default: none)")
    return p.parse_args()


def resolve_sfs(arg: Optional[Sequence[str]]) -> List[str]:
    if not arg or "all" in arg:
        return list(ALL_SCALE_FACTORS)
    s = set(arg)
    return [sf for sf in ALL_SCALE_FACTORS if sf in s]


def resolve_configs(arg: str) -> List[str]:
    if arg == "all":
        return list(ALL_CONFIG_NAMES)
    requested = [c.strip() for c in arg.split(",") if c.strip()]
    for c in requested:
        if c not in ALL_CONFIG_NAMES:
            raise SystemExit(f"unknown config: {c!r}. "
                             f"Choose from: {','.join(ALL_CONFIG_NAMES)} or 'all'.")
    return requested


def main() -> int:
    args = parse_args()
    sfs = resolve_sfs(args.sf)
    configs = resolve_configs(args.configs)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    section(f"Phase 20.1 benchmark runner | SFs={sfs} | configs={configs}")
    print(f"  warmup={args.warmup}, runs={args.runs}, "
          f"pool_sweep={args.pool_sweep}, v1_eviction={args.v1_eviction}")

    shilmandb_configs = [c for c in configs if c in PRIMARY_CONFIGS]
    need_sqlite = any(c.startswith("sqlite_") for c in configs)

    if need_sqlite and not args.skip_sqlite:
        ok = run_sqlite_benchmarks(sfs, args.data_dir, output_dir, args.runs)
        if not ok:
            eprint("! sqlite benchmarks failed — continuing (SQLite rows will be empty)")
    elif need_sqlite and args.skip_sqlite:
        print("  (--skip-sqlite: reusing existing sqlite_latencies.csv)")

    all_records: Dict[Tuple[str, str, str], QueryStats] = {}
    pool_sizes_by_config: Dict[Tuple[str, str], int] = {}

    for sf in sfs:
        for cfg in shilmandb_configs:
            section(f"ShilmanDB {cfg} @ SF={sf}")
            pool = POOL_SIZES[sf]
            stats = run_shilmandb_config(
                binary=args.binary, sf=sf, config_name=cfg,
                data_dir=args.data_dir, output_dir=output_dir,
                pool_size=pool, warmup_runs=args.warmup, timed_runs=args.runs,
                timeout_seconds=args.timeout,
            )
            pool_sizes_by_config[(sf, cfg)] = pool
            if stats is None:
                continue
            for q, qs in stats.items():
                all_records[(sf, cfg, q)] = qs

    if args.pool_sweep:
        sweep_records = run_pool_size_sweep(
            binary=args.binary, data_dir=args.data_dir, output_dir=output_dir,
            pool_sizes=POOL_SWEEP_SIZES, warmup_runs=args.warmup, timed_runs=args.runs,
        )
        all_records.update(sweep_records)
        for (sf, cfg, _q) in sweep_records:
            ps = int(cfg.removeprefix("baseline_pool"))
            pool_sizes_by_config[(sf, cfg)] = ps

    if args.v1_eviction:
        v1_records = run_v1_eviction_sweep(
            binary=args.binary, data_dir=args.data_dir, output_dir=output_dir,
            scale_factors=sfs, warmup_runs=args.warmup, timed_runs=args.runs,
        )
        all_records.update(v1_records)
        for (sf, cfg, _q) in v1_records:
            pool_sizes_by_config[(sf, cfg)] = POOL_SIZES[sf]

    if need_sqlite:
        sqlite_records = parse_sqlite_latencies(output_dir, sfs)
        for key, qs in sqlite_records.items():
            all_records[key] = qs
            pool_sizes_by_config[(key[0], key[1])] = ""  # type: ignore[assignment]

    section("Writing outputs")
    write_latencies_csv(output_dir / "latencies.csv", all_records)
    write_hit_rates_csv(output_dir / "hit_rates.csv", all_records, pool_sizes_by_config)

    if not args.skip_correctness:
        write_correctness_csv(
            output_dir / "correctness.csv",
            sfs=sfs,
            configs=configs + (["+learned_v1"] if args.v1_eviction else []) + ([f"baseline_pool{ps}" for ps in POOL_SWEEP_SIZES] if args.pool_sweep else []),
            output_dir=output_dir,
        )

    section("Done")
    print(f"  latencies  -> {output_dir / 'latencies.csv'}")
    print(f"  hit_rates  -> {output_dir / 'hit_rates.csv'}")
    if not args.skip_correctness:
        print(f"  correctness -> {output_dir / 'correctness.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
