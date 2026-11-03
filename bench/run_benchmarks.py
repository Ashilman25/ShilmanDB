#!/usr/bin/env python3
"""
Benchmark runner

Runs a matrix of (scale factor x configuration x query) benchmarks, then
emits three unified CSVs under bench/results/:

- latencies.csv: (sf, query, config, median_latency_ms, rows, hits, misses,
  visits, page_hit_rate, reads, read_ms, us_per_read, policy_ms, pool_size,
  miss_penalty_us, direct_io)
- hit_rates.csv: (sf, config, pool_size, pool_fraction, miss_penalty_us,
  direct_io, hit_rate, raw_hit_rate, improvement_over_lru)
- correctness.csv: (sf, query, config, status)

A config label is the config name plus a suffix for each non-default option,
for example baseline_pen100, baseline_direct, baseline_frac0.25. Columns an
older load_tpch binary doesn't write are left empty.

With --workloads (stream, lookup, mixed) it also writes workloads.csv (one row
per run), workload_queries.csv (one row per run and query tag) and
stream_correctness.csv (every workload item checked against SQLite). A
workload label adds _<workload>_s<seed> after the config name, for example
baseline_stream_s1. The three files above only cover the single workload.

"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

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

# Pages in the tables alone (no indexes), measured by loading each scale factor.
DB_PAGES: Dict[str, int] = {
    "0.01": 1448,
    "0.1": 14417,
    "1.0": 144032,
}
MIN_POOL_PAGES: int = 16

# What load_tpch runs: "single" is one query at a time, the rest are streams of queries.
WORKLOADS: List[str] = ["single", "stream", "lookup", "mixed"]
INDEXED_WORKLOADS = {"lookup", "mixed"}     # fraction runs use a config with indexes for these

# CSVs a workload run writes (policy_time only with --policy-timing).
WORKLOAD_FILES: List[str] = ["stream", "stream_totals", "stream_queries", "stream_results", "run_info", "policy_time"]
WORKLOAD_REQUIRED_FILES: List[str] = ["stream", "stream_totals", "stream_queries", "stream_results"]

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


# Buffer pool size (in pages) that is the given fraction of the tables' size.
def pool_for_fraction(sf: str, fraction: float) -> int:
    if fraction <= 0:
        raise ValueError(f"pool fraction must be positive, got {fraction}")
    return max(MIN_POOL_PAGES, round(fraction * DB_PAGES[sf]))


# Config name, then _<workload>_s<seed> for a workload, then _frac, _pen and _direct when set.
def config_label(base: str, miss_penalty_us: int = 0, direct_io: bool = False, fraction: Optional[float] = None, workload: str = "single", seed: int = 1) -> str:
    label = base
    if workload != "single":
        label += f"_{workload}_s{seed}"
    if fraction is not None:
        label += f"_frac{float(fraction)}"
    if miss_penalty_us > 0:
        label += f"_pen{miss_penalty_us}"
    if direct_io:
        label += "_direct"
    return label


_LABEL_SUFFIX_RE = re.compile(
    rf"(?:_(?P<workload>{'|'.join(WORKLOADS[1:])})_s(?P<seed>\d+))?"
    r"(?:_frac(?P<frac>[0-9.eE+-]+?))?(?:_pen(?P<pen>\d+))?(?P<direct>_direct)?$"
)


@dataclass
class LabelOptions:
    base: str                       # the label without its suffix
    fraction: Optional[float]
    miss_penalty_us: int
    direct_io: bool
    workload: str = "single"
    seed: Optional[int] = None      # only a workload label has one


# Split a label made by config_label back into the config name and its options.
def parse_label(label: str) -> LabelOptions:
    m = _LABEL_SUFFIX_RE.search(label)
    assert m is not None            # every part is optional, so this always matches
    frac = m.group("frac")
    return LabelOptions(
        base=label[:m.start()],
        fraction=float(frac) if frac else None,
        miss_penalty_us=int(m.group("pen") or 0),
        direct_io=m.group("direct") is not None,
        workload=m.group("workload") or "single",
        seed=int(m.group("seed")) if m.group("seed") else None,
    )


# The load_tpch flags for the realistic-I/O options (only the ones that are set).
def option_flags(miss_penalty_us: int = 0, direct_io: bool = False, policy_timing: bool = False) -> List[str]:
    flags: List[str] = []
    if miss_penalty_us > 0:
        flags += ["--miss-penalty-us", str(miss_penalty_us)]
    if direct_io:
        flags.append("--direct-io")
    if policy_timing:
        flags.append("--policy-timing")
    return flags


# The load_tpch flags that pick a workload. Nothing for single, so older binaries still run it.
def workload_flags(workload: str, seed: int, stream_length: Optional[int]) -> List[str]:
    if workload == "single":
        return []
    flags = ["--workload", workload, "--seed", str(seed)]
    if stream_length is not None:
        flags += ["--stream-length", str(stream_length)]
    return flags


# The config the pool-fraction runs use for a workload.
def fraction_config(workload: str) -> str:
    return "+index" if workload in INDEXED_WORKLOADS else "baseline"


@dataclass(frozen=True)
class RunSpec:
    sf: str
    base: str                       # config name, without any suffix
    workload: str
    pool: int                       # buffer pool pages
    penalty: int                    # miss penalty in microseconds
    fraction: Optional[float]       # pool fraction, when this run is part of a fraction sweep


# Every ShilmanDB run to make, in order. Per scale factor and workload: each config
# (once per penalty), then the pool-fraction runs (once per penalty).
def planned_runs(sfs: Sequence[str], configs: Sequence[str], workloads: Sequence[str], penalties: Sequence[int], fractions: Sequence[float]) -> List[RunSpec]:
    runs: List[RunSpec] = []
    for sf in sfs:
        for workload in workloads:
            for config in configs:
                runs += [RunSpec(sf, config, workload, POOL_SIZES[sf], penalty, None) for penalty in penalties]

            for fraction in fractions:
                runs += [RunSpec(sf, fraction_config(workload), workload, pool_for_fraction(sf, fraction), penalty, fraction) for penalty in penalties]
    return runs


# Read a key,value CSV (run_info.csv). Empty dict when the file isn't there.
def read_run_info(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)          # header
        return {row[0]: (row[1] if len(row) > 1 else "") for row in reader if row}


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

    median_visits: Optional[int] = None
    median_reads: Optional[int] = None
    median_read_ms: Optional[float] = None
    policy_ms: Optional[float] = None
    pool_size: Optional[int] = None
    miss_penalty_us: Optional[int] = None
    direct_io: Optional[bool] = None

    # 1 - misses / visits: how often a page visit found the page in the pool.
    @property
    def page_hit_rate(self) -> Optional[float]:
        if not self.median_visits or self.median_misses is None:
            return None
        return 1.0 - self.median_misses / self.median_visits

    @property
    def us_per_read(self) -> Optional[float]:
        if not self.median_reads or self.median_read_ms is None:
            return None
        return self.median_read_ms * 1000.0 / self.median_reads


@dataclass
class RunOutput:
    """All stats from a single load_tpch invocation."""
    per_query: Dict[str, QueryStats] = field(default_factory=dict)


def _median_int(values: Sequence[int]) -> int:
    """Median rounded to int (used for hit/miss counts)."""
    return int(round(statistics.median(values)))


# Parse the per-run latencies.csv written by load_tpch
# Schema: query,run,rows,latency_ms,hits,misses[,visits,reads,read_ms]
# The last three are missing from older binaries; those fields stay None.
# Returns: {query_tag: QueryStats}
def parse_shilmandb_latencies(path: Path, sf: str, config: str) -> Dict[str, QueryStats]:
    per_query_rows: Dict[str, List[Dict[str, str]]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            per_query_rows.setdefault(row["query"], []).append(row)

    # Values of an optional column, or None unless every run has one.
    def column(runs: List[Dict[str, str]], name: str, cast) -> Optional[List]:
        raw = [r.get(name) for r in runs]
        if any(v is None or v == "" for v in raw):
            return None
        return [cast(v) for v in raw]

    result: Dict[str, QueryStats] = {}
    for q, runs in per_query_rows.items():
        latencies = [float(r["latency_ms"]) for r in runs]
        rows_vals = [int(r["rows"]) for r in runs]
        hits_vals = [int(r["hits"]) for r in runs]
        misses_vals = [int(r["misses"]) for r in runs]
        visits = column(runs, "visits", int)
        reads = column(runs, "reads", int)
        read_ms = column(runs, "read_ms", float)
        result[q] = QueryStats(
            sf=sf,
            config=config,
            query=q,
            median_latency_ms=statistics.median(latencies),
            rows=rows_vals[0],                     # stable across runs
            median_hits=_median_int(hits_vals),
            median_misses=_median_int(misses_vals),
            median_visits=_median_int(visits) if visits else None,
            median_reads=_median_int(reads) if reads else None,
            median_read_ms=statistics.median(read_ms) if read_ms else None,
        )
    return result


# Per-query policy time (ms) from policy_time.csv. Empty dict when it isn't there.
def read_policy_ms(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["query"]: float(row["policy_ms"])for row in csv.DictReader(f) if row.get("policy_ms")}


# Results of one workload run

@dataclass
class WorkloadStats:
    sf: str
    config: str                     # the run's label
    workload: str
    seed: int
    items: int                      # queries in one pass
    timed_passes: int
    pool_size: int
    miss_penalty_us: int
    direct_io: bool
    median_total_ms: Optional[float]    # median over the timed passes, like the counts below
    misses: Optional[int]
    visits: Optional[int]
    reads: Optional[int]
    read_ms: Optional[float]
    query_misses: Optional[int]
    lookup_misses: Optional[int]
    policy_ms: Optional[float] = None
    correct: str = ""               # "passed/total" against SQLite, empty if not checked

    # 1 - misses / visits: how often a page visit found the page in the pool.
    @property
    def page_hit_rate(self) -> Optional[float]:
        if not self.visits or self.misses is None:
            return None
        return 1.0 - self.misses / self.visits

    @property
    def us_per_read(self) -> Optional[float]:
        if not self.reads or self.read_ms is None:
            return None
        return self.read_ms * 1000.0 / self.reads


@dataclass
class WorkloadQueryRow:
    sf: str
    config: str
    workload: str
    query: str
    items: int
    median_latency_ms: float
    median_misses: float
    median_visits: float


# Medians over the timed passes of stream_totals.csv; warmup passes are left out.
# The keys are WorkloadStats field names.
def parse_stream_totals(path: Path) -> Dict[str, object]:
    with path.open(newline="") as f:
        timed = [row for row in csv.DictReader(f) if row["warmup"] == "0"]
    if not timed:
        raise ValueError(f"{path}: no timed passes")

    def int_median(column: str) -> int:
        return _median_int([int(row[column]) for row in timed])

    def float_median(column: str) -> float:
        return statistics.median([float(row[column]) for row in timed])

    return {
        "items": int_median("items"),
        "timed_passes": len(timed),
        "median_total_ms": float_median("latency_ms"),
        "misses": int_median("misses"),
        "visits": int_median("visits"),
        "reads": int_median("reads"),
        "read_ms": float_median("read_ms"),
        "query_misses": int_median("query_misses"),
        "lookup_misses": int_median("lookup_misses"),
    }


# Streams list the fixed queries first (in QUERIES order), then the lookup tags.
def _query_rank(tag: str) -> Tuple[int, str]:
    return (QUERIES.index(tag) if tag in QUERIES else len(QUERIES), tag)


# Per query tag, over every occurrence in the timed passes of stream.csv:
# {tag: {items, median_latency_ms, median_misses, median_visits}}.
def parse_stream_items(path: Path) -> Dict[str, Dict[str, float]]:
    by_tag: Dict[str, List[Dict[str, str]]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["warmup"] == "0":
                by_tag.setdefault(row["query"], []).append(row)

    def median_of(rows: List[Dict[str, str]], column: str) -> float:
        return statistics.median(float(r[column]) for r in rows)

    return {
        tag: {
            "items": len(rows),
            "median_latency_ms": median_of(rows, "latency_ms"),
            "median_misses": median_of(rows, "misses"),
            "median_visits": median_of(rows, "visits"),
        }
        for tag, rows in sorted(by_tag.items(), key=lambda entry: _query_rank(entry[0]))
    }


# Total policy time (ms) in a workload's policy_time.csv, one row per item. None when it isn't there.
def read_policy_total_ms(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    with path.open(newline="") as f:
        return sum(float(row["policy_ms"]) for row in csv.DictReader(f) if row.get("policy_ms"))


# ShilmanDB runner 
#Build the load_tpch CLI for a given config.
def build_command(binary: Path, sf: str, config_name: str, data_dir: Path, results_dir: Path,
                  db_file: Path, pool_size: int, warmup_runs: int, timed_runs: int,
                  extra_flags: Sequence[str] = ()) -> List[str]:

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
    cmd += extra_flags
    return cmd


# Warn (never fail) when the loaded database isn't the size DB_PAGES says it is.
def warn_on_db_pages_mismatch(info: Dict[str, str], sf: str) -> None:
    reported = info.get("db_pages")
    if reported is None or sf not in DB_PAGES:
        return
    if not reported.isdigit() or int(reported) != DB_PAGES[sf]:
        eprint(f"  ! warning: run_info.csv db_pages={reported} but DB_PAGES[{sf}]={DB_PAGES[sf]}; "
               "pool fractions may be off")


# Run load_tpch once for a (sf, config) in a temp folder and hand that folder to the caller.
# Gives None instead when the run timed out or failed (the reason is printed).
@contextmanager
def load_tpch_run(binary: Path, sf: str, config_name: str, data_dir: Path, pool_size: int, warmup_runs: int,
                  timed_runs: int, timeout_seconds: Optional[int], label: str,
                  extra_flags: Sequence[str]) -> Iterator[Optional[Path]]:
    with tempfile.TemporaryDirectory(prefix=f"shilmandb_{label}_sf{sf}_") as tmp:
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
            extra_flags=extra_flags,
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
            yield None
            return

        wall = time.time() - t0
        print(f"  (wall: {elapsed_str(wall)}, exit: {proc.returncode})", flush=True)

        if proc.returncode != 0:
            eprint(f"  ! load_tpch exited {proc.returncode}")
            eprint("  --- last 40 lines of stderr ---")
            for line in proc.stderr.splitlines()[-40:]:
                eprint(f"    {line}")
            yield None
            return

        yield tmp_path


#spawn load_tpch for one (sf, config) combo
#returns per query stats; label names the results folder and the config column (default: config_name)
def run_shilmandb_config(binary: Path, sf: str, config_name: str, data_dir: Path, output_dir: Path, pool_size: int, 
                         warmup_runs: int, timed_runs: int, timeout_seconds: Optional[int] = None,
                         label: Optional[str] = None,
                         extra_flags: Sequence[str] = ()) -> Optional[Dict[str, QueryStats]]:

    label = label or config_name
    results_parent = output_dir / "shilmandb_results" / label
    results_parent.mkdir(parents=True, exist_ok=True)

    with load_tpch_run(binary, sf, config_name, data_dir, pool_size, warmup_runs, timed_runs, timeout_seconds, label, extra_flags) as tmp_path:
        if tmp_path is None:
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

        # Keep the run settings and policy timing next to the query results.
        for name in ("run_info", "policy_time"):
            src = tmp_path / f"{name}.csv"
            if src.exists():
                shutil.copyfile(src, results_parent / f"{name}_sf{sf}.csv")

        info = read_run_info(tmp_path / "run_info.csv")
        warn_on_db_pages_mismatch(info, sf)
        policy_ms = read_policy_ms(tmp_path / "policy_time.csv")

        options = parse_label(label)
        stats = parse_shilmandb_latencies(lat_csv, sf=sf, config=label)
        for q, qs in stats.items():
            qs.policy_ms = policy_ms.get(q)
            qs.pool_size = pool_size
            qs.miss_penalty_us = options.miss_penalty_us
            qs.direct_io = options.direct_io
        return stats


# Same for a stream, lookup or mixed workload (the label says which): the stream CSVs are copied
# to shilmandb_results/<label>/<name>_sf<sf>.csv. None if the run failed or its files are missing.
def run_workload_config(binary: Path, sf: str, config_name: str, data_dir: Path, output_dir: Path, pool_size: int,
                        warmup_runs: int, timed_runs: int, timeout_seconds: Optional[int], label: str,
                        extra_flags: Sequence[str] = ()) -> Optional[WorkloadStats]:

    results_parent = output_dir / "shilmandb_results" / label
    results_parent.mkdir(parents=True, exist_ok=True)

    with load_tpch_run(binary, sf, config_name, data_dir, pool_size, warmup_runs, timed_runs, timeout_seconds, label, extra_flags) as tmp_path:
        if tmp_path is None:
            return None

        missing = [f"{name}.csv" for name in WORKLOAD_REQUIRED_FILES if not (tmp_path / f"{name}.csv").exists()]
        if missing:
            eprint(f"  ! missing {', '.join(missing)}")
            return None

        for name in WORKLOAD_FILES:
            src = tmp_path / f"{name}.csv"
            if src.exists():
                shutil.copyfile(src, results_parent / f"{name}_sf{sf}.csv")

        warn_on_db_pages_mismatch(read_run_info(tmp_path / "run_info.csv"), sf)
        try:
            totals = parse_stream_totals(tmp_path / "stream_totals.csv")
            policy_ms = read_policy_total_ms(tmp_path / "policy_time.csv")
        except (OSError, ValueError, KeyError) as e:
            eprint(f"  ! could not read the workload results: {e}")
            return None

        options = parse_label(label)
        return WorkloadStats(
            sf=sf,
            config=label,
            workload=options.workload,
            seed=options.seed if options.seed is not None else 0,
            pool_size=pool_size,
            miss_penalty_us=options.miss_penalty_us,
            direct_io=options.direct_io,
            policy_ms=policy_ms,
            **totals,
        )


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
        sweep_label = f"baseline_pool{ps}"
        subsection(f"{sweep_label}")
        PRIMARY_CONFIGS[sweep_label] = baseline_flags  # shares flags; pool_size differs via --pool-size
        try:
            stats = run_shilmandb_config(
                binary=binary, sf=sf, config_name=sweep_label,
                data_dir=data_dir, output_dir=output_dir,
                pool_size=ps, warmup_runs=warmup_runs, timed_runs=timed_runs,
            )
            
        finally:
            del PRIMARY_CONFIGS[sweep_label]
            
        if stats is None:
            eprint(f"  pool-sweep {ps} failed — skipping")
            continue
        for query, qs in stats.items():
            results[(sf, sweep_label, query)] = qs
            
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

def _fmt_flag(x: Optional[bool]) -> str:
    return "" if x is None else ("true" if x else "false")


LATENCIES_HEADER: List[str] = [
    "sf", "query", "config", "median_latency_ms", "rows", "hits", "misses",
    "visits", "page_hit_rate", "reads", "read_ms", "us_per_read", "policy_ms",
    "pool_size", "miss_penalty_us", "direct_io",
]

#write latencies.csv with the LATENCIES_HEADER columns; fields a binary didn't write stay empty
def write_latencies_csv(path: Path, records: Dict[Tuple[str, str, str], QueryStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    query_order = {q: i for i, q in enumerate(QUERIES)}
    keys = sorted(records.keys(), key=lambda k: (k[0], query_order.get(k[2], 999), k[1]))
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(LATENCIES_HEADER)
        for key in keys:
            qs = records[key]
            w.writerow([
                qs.sf, qs.query, qs.config,
                _fmt_float(qs.median_latency_ms),
                _fmt_int(qs.rows),
                _fmt_int(qs.median_hits),
                _fmt_int(qs.median_misses),
                _fmt_int(qs.median_visits),
                _fmt_float(qs.page_hit_rate),
                _fmt_int(qs.median_reads),
                _fmt_float(qs.median_read_ms),
                _fmt_float(qs.us_per_read),
                _fmt_float(qs.policy_ms),
                _fmt_int(qs.pool_size),
                _fmt_int(qs.miss_penalty_us),
                _fmt_flag(qs.direct_io),
            ])
    print(f"  -> {path}")


# A run's pool as a share of the tables' pages: the label's fraction if it has one, else pool_size / DB_PAGES.
# Empty when neither is known.
def pool_fraction_text(label: str, sf: str, pool_size) -> str:
    fraction = parse_label(label).fraction
    if fraction is not None:
        return f"{fraction:.4f}"
    if pool_size != "" and sf in DB_PAGES:
        return f"{pool_size / DB_PAGES[sf]:.4f}"
    return ""


#Aggregate per-query counts into per-(sf,config) hit rates.
# hit_rate is page level: 1 - sum(misses) / sum(visits). raw_hit_rate is the old fetch-level
# hits / (hits + misses). improvement_over_lru compares with the baseline row that has the same
# option suffix (_frac, _pen, _direct); empty when that row is missing or has no page-level rate.
def write_hit_rates_csv(path: Path, records: Dict[Tuple[str, str, str], QueryStats], pool_sizes_by_config: Dict[Tuple[str, str], int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    by_sc: Dict[Tuple[str, str], List[QueryStats]] = {}
    for (sf, config, _q), qs in records.items():
        by_sc.setdefault((sf, config), []).append(qs)

    page_rates: Dict[Tuple[str, str], Optional[float]] = {}
    raw_rates: Dict[Tuple[str, str], Optional[float]] = {}
    for (sf, config), stats_list in by_sc.items():
        if config.startswith("sqlite_"):
            page_rates[(sf, config)] = raw_rates[(sf, config)] = None
            continue

        total_hits = sum(s.median_hits for s in stats_list if s.median_hits is not None)
        total_misses = sum(s.median_misses for s in stats_list if s.median_misses is not None)
        total = total_hits + total_misses
        raw_rates[(sf, config)] = (total_hits / total) if total > 0 else 0.0

        have_visits = all(s.median_visits is not None and s.median_misses is not None for s in stats_list)
        total_visits = sum(s.median_visits for s in stats_list if s.median_visits is not None)
        page_rates[(sf, config)] = (1.0 - total_misses / total_visits) if have_visits and total_visits > 0 else None

    def _config_rank(name: str) -> int:
        base = parse_label(name).base
        if base == "baseline": return 0
        if base.startswith("+"): return 1
        if base.startswith("baseline_pool"): return 2
        if base.startswith("sqlite_"): return 3
        return 4

    keys = sorted(raw_rates.keys(), key=lambda k: (k[0], _config_rank(k[1]), k[1]))

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sf", "config", "pool_size", "pool_fraction", "miss_penalty_us",
                    "direct_io", "hit_rate", "raw_hit_rate", "improvement_over_lru"])

        for (sf, config) in keys:
            ps = pool_sizes_by_config.get((sf, config), "")
            if config.startswith("sqlite_"):
                w.writerow([sf, config, ps, "", "", "", "", "", ""])
                continue

            options = parse_label(config)
            rate = page_rates[(sf, config)]
            baseline_label = "baseline" + config[len(options.base):]
            base_rate = page_rates.get((sf, baseline_label))
            if rate is None or base_rate is None or config == baseline_label:
                improvement = ""
            else:
                improvement = f"{rate - base_rate:.4f}"

            w.writerow([
                sf, config, ps, pool_fraction_text(config, sf, ps), options.miss_penalty_us,
                "true" if options.direct_io else "false",
                "" if rate is None else f"{rate:.4f}",
                f"{raw_rates[(sf, config)]:.4f}",
                improvement,
            ])

    print(f"  -> {path}")


WORKLOADS_HEADER: List[str] = [
    "sf", "config", "workload", "seed", "items", "timed_passes", "pool_size", "pool_fraction",
    "miss_penalty_us", "direct_io", "median_total_ms", "misses", "visits", "page_hit_rate",
    "reads", "read_ms", "us_per_read", "query_misses", "lookup_misses", "policy_ms", "correct",
]

#write workloads.csv, one row per workload run, in the order given
def write_workloads_csv(path: Path, records: Sequence[WorkloadStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(WORKLOADS_HEADER)
        for r in records:
            w.writerow([
                r.sf, r.config, r.workload, r.seed, r.items, r.timed_passes, r.pool_size,
                pool_fraction_text(r.config, r.sf, r.pool_size),
                r.miss_penalty_us, _fmt_flag(r.direct_io),
                _fmt_float(r.median_total_ms),
                _fmt_int(r.misses), _fmt_int(r.visits), _fmt_float(r.page_hit_rate),
                _fmt_int(r.reads), _fmt_float(r.read_ms), _fmt_float(r.us_per_read),
                _fmt_int(r.query_misses), _fmt_int(r.lookup_misses),
                _fmt_float(r.policy_ms), r.correct,
            ])
    print(f"  -> {path}")


# A median of whole numbers is a whole number or ends in .5.
def _fmt_median(x: float) -> str:
    return str(int(x)) if x == int(x) else f"{x:.1f}"


#write workload_queries.csv, one row per (run, query tag)
def write_workload_queries_csv(path: Path, rows: Sequence[WorkloadQueryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sf", "config", "workload", "query", "items", "median_latency_ms", "median_misses", "median_visits"])
        for r in rows:
            w.writerow([r.sf, r.config, r.workload, r.query, r.items, _fmt_float(r.median_latency_ms), _fmt_median(r.median_misses), _fmt_median(r.median_visits)])
    print(f"  -> {path}")


# Per-query rows for one workload run, read back from the stream.csv copied into its results folder.
def workload_query_rows(output_dir: Path, record: WorkloadStats) -> List[WorkloadQueryRow]:
    path = output_dir / "shilmandb_results" / record.config / f"stream_sf{record.sf}.csv"
    try:
        items = parse_stream_items(path)
    except (OSError, ValueError, KeyError) as e:
        eprint(f"  ! could not read {path}: {e}")
        return []
    return [
        WorkloadQueryRow(record.sf, record.config, record.workload, tag, int(v["items"]), v["median_latency_ms"], v["median_misses"], v["median_visits"])
        for tag, v in items.items()
    ]


# Load bench/check_stream_results.py by path. It is only needed for workload runs.
def load_stream_checker() -> ModuleType:
    name = "check_stream_results"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "bench" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module      # its dataclasses look their module up here while it loads
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[name]
        raise
    return module


# Check every item of every workload run against SQLite, one SQLite copy per scale factor.
# Writes stream_correctness.csv and returns "passed/total" per (sf, label).
# A scale factor that can't be checked is reported and left out.
def check_workload_runs(records: Sequence[WorkloadStats], data_dir: Path, output_dir: Path) -> Dict[Tuple[str, str], str]:
    checker = load_stream_checker()
    labels_by_sf: Dict[str, List[str]] = {}
    for r in records:
        labels_by_sf.setdefault(r.sf, []).append(r.config)

    all_rows: list = []
    verdicts: Dict[Tuple[str, str], str] = {}
    for sf, labels in labels_by_sf.items():
        subsection(f"Workload correctness @ SF={sf}")
        folders = [output_dir / "shilmandb_results" / label for label in labels]
        try:
            conn = checker.open_sqlite(data_dir, sf)
            try:
                rows = checker.check_folders(sf, folders, lambda sql: conn.execute(sql).fetchall())
            finally:
                conn.close()
        except (OSError, ValueError, KeyError, sqlite3.Error) as e:
            eprint(f"  ! could not check SF={sf}: {e}")
            continue

        all_rows += rows
        for label in labels:
            mine = [row for row in rows if row.config == label]
            passed = sum(row.status == "PASS" for row in mine)
            verdicts[(sf, label)] = f"{passed}/{len(mine)}"
            print(f"  PASS {passed}/{len(mine)} [{label}]")
            failed = [row for row in mine if row.status == "FAIL"]
            for row in failed[:5]:
                print(f"    FAIL seq {row.seq} {row.query}: {row.detail}")
            if len(failed) > 5:
                print(f"    ... and {len(failed) - 5} more")

    path = output_dir / "stream_correctness.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    checker.write_check_csv(path, all_rows)
    print(f"  -> {path}")
    return verdicts


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


# CLI + main 

def _comma_list(text: str, cast, what: str) -> list:
    try:
        return [cast(v) for v in text.split(",") if v.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"{what}: bad value in {text!r}")


def penalty_list(text: str) -> List[int]:
    values = _comma_list(text, int, "--miss-penalty-us expects integers")
    if not values or any(v < 0 for v in values):
        raise argparse.ArgumentTypeError("--miss-penalty-us expects non-negative integers, e.g. 0,10,100")
    return list(dict.fromkeys(values))


def fraction_list(text: str) -> List[float]:
    values = _comma_list(text, float, "--pool-fractions expects numbers")
    if not values or any(not 0 < v <= 1 for v in values):
        raise argparse.ArgumentTypeError("--pool-fractions expects numbers in (0, 1], e.g. 0.05,0.25,1.0")
    return list(dict.fromkeys(values))


def workload_list(text: str) -> List[str]:
    values = _comma_list(text, str.strip, "--workloads expects names")
    if not values or any(v not in WORKLOADS for v in values):
        raise argparse.ArgumentTypeError(f"--workloads expects names from {','.join(WORKLOADS)}, e.g. single,stream")
    return list(dict.fromkeys(values))


def _int_at_least(text: str, minimum: int, what: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{what} expects an integer, got {text!r}")
    if value < minimum:
        raise argparse.ArgumentTypeError(f"{what} expects an integer of at least {minimum}, got {value}")
    return value


def seed_value(text: str) -> int:
    return _int_at_least(text, 0, "--seed")


def stream_length_value(text: str) -> int:
    return _int_at_least(text, 1, "--stream-length")


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
    p.add_argument("--miss-penalty-us", type=penalty_list, default=[0], help="Comma list of simulated per-read delays in microseconds; every ShilmanDB config runs once per value (default 0)")
    p.add_argument("--direct-io", action="store_true", help="Read pages around the OS file cache (adds _direct to labels)")
    p.add_argument("--pool-fractions", type=fraction_list, default=[], help="Comma list of pool sizes as fractions of the tables' pages (0 to 1]; runs baseline once per fraction per SF (+index for the lookup and mixed workloads)")
    p.add_argument("--policy-timing", action="store_true", help="Also time the eviction policy per query (fills policy_ms)")
    p.add_argument("--workloads", type=workload_list, default=["single"], help=f"Comma list of workloads from {','.join(WORKLOADS)}; single is one query at a time (default single). lookup and mixed run on every config; without indexes each lookup scans its table")
    p.add_argument("--seed", type=seed_value, default=1, help="Seed for the random queries of the non-single workloads (default 1)")
    p.add_argument("--stream-length", type=stream_length_value, default=None, help="Items per pass for every non-single workload (default: load_tpch's default for the scale factor)")
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
    if args.miss_penalty_us != [0] or args.direct_io or args.pool_fractions or args.policy_timing:
        print(f"  miss_penalty_us={args.miss_penalty_us}, direct_io={args.direct_io}, "
              f"pool_fractions={args.pool_fractions}, policy_timing={args.policy_timing}")
    has_workloads = any(w != "single" for w in args.workloads)
    if has_workloads:
        print(f"  workloads={args.workloads}, seed={args.seed}, stream_length={args.stream_length}")

    shilmandb_configs = [c for c in configs if c in PRIMARY_CONFIGS]
    # latencies.csv, hit_rates.csv and correctness.csv describe single-query runs only, so a call
    # that makes none (workloads only) must leave them alone.
    runs_single = "single" in args.workloads
    writes_single_files = runs_single or args.pool_sweep or args.v1_eviction
    sqlite_configs_requested = any(c.startswith("sqlite_") for c in configs)
    need_sqlite = runs_single and sqlite_configs_requested
    if sqlite_configs_requested and not runs_single:
        print("  (skip: the SQLite configs only run with the single workload)")

    if need_sqlite and not args.skip_sqlite:
        ok = run_sqlite_benchmarks(sfs, args.data_dir, output_dir, args.runs)
        if not ok:
            eprint("! sqlite benchmarks failed — continuing (SQLite rows will be empty)")
    elif need_sqlite and args.skip_sqlite:
        print("  (--skip-sqlite: reusing existing sqlite_latencies.csv)")

    all_records: Dict[Tuple[str, str, str], QueryStats] = {}
    pool_sizes_by_config: Dict[Tuple[str, str], int] = {}

    ran_labels: List[str] = []      # every single-query ShilmanDB label that was run, for the correctness check
    workload_records: List[WorkloadStats] = []

    def run_single(spec: RunSpec) -> None:
        label = config_label(spec.base, spec.penalty, args.direct_io, spec.fraction)
        section(f"ShilmanDB {label} @ SF={spec.sf}")
        stats = run_shilmandb_config(
            binary=args.binary, sf=spec.sf, config_name=spec.base,
            data_dir=args.data_dir, output_dir=output_dir,
            pool_size=spec.pool, warmup_runs=args.warmup, timed_runs=args.runs,
            timeout_seconds=args.timeout, label=label,
            extra_flags=option_flags(spec.penalty, args.direct_io, args.policy_timing),
        )
        pool_sizes_by_config[(spec.sf, label)] = spec.pool
        if label not in ran_labels:
            ran_labels.append(label)
        if stats is None:
            return
        for q, qs in stats.items():
            all_records[(spec.sf, label, q)] = qs

    def run_workload(spec: RunSpec) -> None:
        label = config_label(spec.base, spec.penalty, args.direct_io, spec.fraction, spec.workload, args.seed)
        section(f"ShilmanDB {label} @ SF={spec.sf}")
        stats = run_workload_config(
            binary=args.binary, sf=spec.sf, config_name=spec.base,
            data_dir=args.data_dir, output_dir=output_dir,
            pool_size=spec.pool, warmup_runs=args.warmup, timed_runs=args.runs,
            timeout_seconds=args.timeout, label=label,
            extra_flags=(workload_flags(spec.workload, args.seed, args.stream_length) + option_flags(spec.penalty, args.direct_io, args.policy_timing)),
        )
        if stats is None:
            eprint(f"  ! {label} @ SF={spec.sf} failed - skipping")
            return
        workload_records.append(stats)

    for spec in planned_runs(sfs, shilmandb_configs, args.workloads, args.miss_penalty_us, args.pool_fractions):
        (run_single if spec.workload == "single" else run_workload)(spec)

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

    if writes_single_files:
        section("Writing outputs")
        write_latencies_csv(output_dir / "latencies.csv", all_records)
        write_hit_rates_csv(output_dir / "hit_rates.csv", all_records, pool_sizes_by_config)

        if not args.skip_correctness:
            write_correctness_csv(
                output_dir / "correctness.csv",
                sfs=sfs,
                configs=ran_labels + [c for c in configs if c.startswith("sqlite_")]
                        + (["+learned_v1"] if args.v1_eviction else [])
                        + ([f"baseline_pool{ps}" for ps in POOL_SWEEP_SIZES] if args.pool_sweep else []),
                output_dir=output_dir,
            )

    if has_workloads:
        section("Writing workload outputs")
        if not args.skip_correctness and workload_records:
            verdicts = check_workload_runs(workload_records, args.data_dir, output_dir)
            for record in workload_records:
                record.correct = verdicts.get((record.sf, record.config), "")
        write_workloads_csv(output_dir / "workloads.csv", workload_records)
        write_workload_queries_csv(
            output_dir / "workload_queries.csv",
            [row for record in workload_records for row in workload_query_rows(output_dir, record)],
        )

    section("Done")
    if writes_single_files:
        print(f"  latencies  -> {output_dir / 'latencies.csv'}")
        print(f"  hit_rates  -> {output_dir / 'hit_rates.csv'}")
        if not args.skip_correctness:
            print(f"  correctness -> {output_dir / 'correctness.csv'}")
    if has_workloads:
        print(f"  workloads  -> {output_dir / 'workloads.csv'}")
        print(f"  workload_queries -> {output_dir / 'workload_queries.csv'}")
        if not args.skip_correctness:
            print(f"  stream_correctness -> {output_dir / 'stream_correctness.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
