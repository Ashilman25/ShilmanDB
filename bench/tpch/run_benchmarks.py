#!/usr/bin/env python3
"""
Benchmark runner

Runs a matrix of (scale factor x configuration x query) benchmarks, then
emits three unified CSVs under bench/results/:

- latencies.csv: (sf, query, config, median_latency_ms, rows, hits, misses,
  visits, page_hit_rate, reads, read_ms, us_per_read, policy_ms, pool_size,
  miss_penalty_us, direct_io, eviction)
- hit_rates.csv: (sf, config, pool_size, pool_fraction, miss_penalty_us,
  direct_io, hit_rate, raw_hit_rate, improvement_over_lru, eviction)
- correctness.csv: (sf, query, config, status)

With --duckdb and --postgres it also runs those engines' runners (after ShilmanDB,
Postgres last) and adds their configs to the three files, then writes
reference_checks.csv (every config against every reference it is checked against)
and environment.csv (machine, build and engine versions).

A config label is the config name plus a suffix for each non-default option,
for example baseline_pen100, baseline_direct, baseline_frac0.25. Columns an
older load_tpch binary doesn't write are left empty.

With --workloads (stream, lookup, mixed) it also writes workloads.csv (one row
per run), workload_queries.csv (one row per run and query tag) and
stream_correctness.csv (every workload item checked against SQLite). A
workload label adds _<workload>_s<seed> after the config name, for example
baseline_stream_s1. The three files above only cover the single workload.

The columnar and parallel configs run only when named. Within one call each
columnar file is loaded once per scale factor and reused by later runs:
columnar has its own, and the indexed configs share another. --columnar-dir
keeps the files for later calls, reused while their markers match the data.
Every run that loads its tables leaves load_times_<label>_sf<sf>.csv in the
output folder. SF=100 runs only the parallel configs, with a 4 GiB memory limit.

--parallel-threads 2,4 adds columnar+parallel_2thread and columnar+parallel_4thread
after the named configs. Every parallel run leaves
parallel_stats_<label>_sf<sf>.csv in the output folder.

"""

from __future__ import annotations

import argparse
import csv
import datetime
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

# Constants 

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_BINARY = REPO_ROOT / "build" / "bench" / "load_tpch"
DEFAULT_DATA_DIR = REPO_ROOT / "bench" / "tpch_data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "bench" / "results"

ALL_SCALE_FACTORS: List[str] = ["0.01", "0.1", "1.0"]
KNOWN_SCALE_FACTORS: List[str] = [*ALL_SCALE_FACTORS, "10", "100"]  # 10 and 100 run only when named
# Only the parallel configs have a memory limit, so nothing else runs at these
PARALLEL_ONLY_SCALE_FACTORS: List[str] = ["100"]
QUERIES: List[str] = ["Q1", "Q3", "Q5", "Q6", "Q10", "Q12", "Q14", "Q19"]

POOL_SIZES: Dict[str, int] = {
    "0.01": 128,
    "0.1": 512,
    "1.0": 2048,
    "10": 8192,
    "100": 2097152,     # 16 GiB of 8 KB pages
}

# --memory-limit in bytes for every parallel config at a scale factor,
# unless the config sets its own
MEMORY_LIMITS: Dict[str, int] = {"100": 4 << 30}

# Scale factors where an answer may not drift: compare_results.py gets --no-drift
NO_DRIFT_SCALE_FACTORS: List[str] = ["100"]

# Pages in the tables alone (no indexes), measured by loading each scale factor.
DB_PAGES: Dict[str, int] = {
    "0.01": 1448,
    "0.1": 14417,
    "1.0": 144032,
    "10": 1439816,
}
MIN_POOL_PAGES: int = 16
# The file's two header pages, counted in db_pages since the columnar format
HEADER_PAGES: int = 2

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

# The parallel executor on the indexed columnar file
PARALLEL_FLAGS: List[str] = ["--create-indexes", "true", "--execution-mode", "parallel", "--storage", "columnar"]

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
    # baseline, +index and +vectorized again, on columnar storage
    "columnar": [
        "--create-indexes", "false",
        "--execution-mode", "tuple",
        "--storage", "columnar",
    ],
    "columnar+index": [
        "--create-indexes", "true",
        "--execution-mode", "tuple",
        "--storage", "columnar",
    ],
    "columnar+vectorized": [
        "--create-indexes", "true",
        "--execution-mode", "vectorized",
        "--storage", "columnar",
    ],
    # The parallel executor on the indexed columnar file, on every core or one
    "columnar+parallel": [*PARALLEL_FLAGS],
    "columnar+parallel_1thread": [*PARALLEL_FLAGS, "--threads", "1"],
    # Spilling under a small limit, and cold reads around the file cache
    "columnar+parallel_spill": [*PARALLEL_FLAGS, "--memory-limit", str(256 << 20)],
    "columnar+parallel_direct": [*PARALLEL_FLAGS, "--direct-io"],
    # SF=100's settings at any scale factor (its pool is in POOL_OVERRIDES)
    "columnar+parallel_20g": [*PARALLEL_FLAGS, "--memory-limit", str(4 << 30)],
}

# Configs whose pool is the same at every scale factor, beating POOL_SIZES.
POOL_OVERRIDES: Dict[str, int] = {"columnar+parallel_20g": 2097152}

# The most threads one parallel query may use (kMaxThreads in the engine).
MAX_PARALLEL_THREADS: int = 256
_SWEEP_CONFIG_RE = re.compile(r"columnar\+parallel_(\d+)thread")

# What --configs all runs. The columnar configs run only when named.
SHILMANDB_CONFIG_NAMES: List[str] = ["baseline", "+index", "+learned", "+vectorized"]
PARALLEL_CONFIG_NAMES: List[str] = ["columnar+parallel", "columnar+parallel_1thread", "columnar+parallel_spill",
                                     "columnar+parallel_direct", "columnar+parallel_20g"]
COLUMNAR_CONFIG_NAMES: List[str] = ["columnar", "columnar+index", "columnar+vectorized", *PARALLEL_CONFIG_NAMES]
SQLITE_CONFIG_NAMES: List[str] = ["sqlite_disk", "sqlite_memory"]
ALL_CONFIG_NAMES: List[str] = SHILMANDB_CONFIG_NAMES + SQLITE_CONFIG_NAMES

# Other engines, added to a run only by --duckdb and --postgres. The named ones
# run only when --configs names them, and only with their engine's flag.
DUCKDB_CONFIG_NAMES: List[str] = ["duckdb", "duckdb_1thread"]
POSTGRES_CONFIG_NAMES: List[str] = ["postgres", "postgres_1worker"]
ENGINE_CONFIGS: Dict[str, List[str]] = {"duckdb": DUCKDB_CONFIG_NAMES, "postgres": POSTGRES_CONFIG_NAMES}
NAMED_ENGINE_CONFIGS: Dict[str, List[str]] = {"duckdb": ["duckdb_20g"], "postgres": []}

KNOWN_CONFIG_NAMES: List[str] = (ALL_CONFIG_NAMES + COLUMNAR_CONFIG_NAMES
                                 + [c for names in NAMED_ENGINE_CONFIGS.values() for c in names])


# The sweep config for n threads; 1 thread is the named columnar+parallel_1thread.
def sweep_config_name(threads: int) -> str:
    return f"columnar+parallel_{threads}thread"


# The sweep configs --parallel-threads adds, in its order, skipping any already named.
def sweep_configs(threads: Sequence[int], named: Sequence[str]) -> List[str]:
    return [name for name in map(sweep_config_name, threads) if name not in named]


# A config's load_tpch flags; a sweep config is columnar+parallel plus --threads N.
def config_flags(config_name: str) -> List[str]:
    if config_name in PRIMARY_CONFIGS:
        return PRIMARY_CONFIGS[config_name]
    m = _SWEEP_CONFIG_RE.fullmatch(config_name)
    if m is None:
        raise ValueError(f"unknown ShilmanDB config: {config_name}")
    return PRIMARY_CONFIGS["columnar+parallel"] + ["--threads", m.group(1)]


# True for a config whose tables are columnar, sweep configs included.
def is_columnar_config(config_name: str) -> bool:
    return config_name in COLUMNAR_CONFIG_NAMES or _SWEEP_CONFIG_RE.fullmatch(config_name) is not None


# True for a config that runs the parallel executor, sweep configs included.
def is_parallel_config(config_name: str) -> bool:
    return config_name in PARALLEL_CONFIG_NAMES or _SWEEP_CONFIG_RE.fullmatch(config_name) is not None


# --memory-limit for a parallel config at a scale factor in MEMORY_LIMITS,
# unless its flags set one.
def memory_limit_flags(sf: str, config_name: str) -> List[str]:
    if sf not in MEMORY_LIMITS or not is_parallel_config(config_name) or "--memory-limit" in config_flags(config_name):
        return []
    return ["--memory-limit", str(MEMORY_LIMITS[sf])]


# --cold for a parallel single-query run that reads around the file cache.
# Workloads never get it: they start from an empty pool already.
def cold_flags(config_name: str, direct_io: bool) -> List[str]:
    if not is_parallel_config(config_name):
        return []
    return ["--cold"] if direct_io or "--direct-io" in config_flags(config_name) else []


REFERENCE_NAMES: Dict[str, str] = {"sqlite": "SQLite", "duckdb": "DuckDB", "postgres": "Postgres"}
RESULTS_DIRS: Dict[str, str] = {"shilmandb": "shilmandb_results", "duckdb": "duckdb_results", "postgres": "postgres_results"}

# The server settings the Postgres runner records, copied into environment.csv.
POSTGRES_SETTING_NAMES: List[str] = [
    "shared_buffers", "effective_cache_size", "work_mem", "maintenance_work_mem",
    "max_parallel_workers_per_gather", "max_parallel_workers", "max_worker_processes", "jit",
]

#Helpers

# Load a script next to this one by path, once per process.
def load_sibling(name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module      # its dataclasses look their module up here while it loads
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[name]
        raise
    return module


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
    if sf not in DB_PAGES:
        raise ValueError(f"DB_PAGES has no entry for SF={sf} yet")
    return max(MIN_POOL_PAGES, round(fraction * DB_PAGES[sf]))


# The eviction policies the runner can compare. The learned ones come with the +learned config.
EVICTION_POLICIES: List[str] = ["lru", "clock", "lru2", "arc", "lirs"]


# Config name, then _<workload>_s<seed> for a workload, then _<policy> unless it is lru,
# then _frac, _pen and _direct when set. A config named ..._direct reads as direct I/O.
def config_label(base: str, miss_penalty_us: int = 0, direct_io: bool = False, fraction: Optional[float] = None, workload: str = "single", seed: int = 1, eviction: str = "lru") -> str:
    if base.endswith("_direct"):
        base, direct_io = base.removesuffix("_direct"), True
    label = base
    if workload != "single":
        label += f"_{workload}_s{seed}"
    if eviction != "lru":
        label += f"_{eviction}"
    if fraction is not None:
        label += f"_frac{float(fraction)}"
    if miss_penalty_us > 0:
        label += f"_pen{miss_penalty_us}"
    if direct_io:
        label += "_direct"
    return label


_LABEL_SUFFIX_RE = re.compile(
    rf"(?:_(?P<workload>{'|'.join(WORKLOADS[1:])})_s(?P<seed>\d+))?"
    rf"(?:_(?P<eviction>{'|'.join(EVICTION_POLICIES[1:])}))?"
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
    eviction: str = "lru"


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
        eviction=m.group("eviction") or "lru",
    )


# The load_tpch flags for the realistic-I/O options (only the ones that are set).
def option_flags(miss_penalty_us: int = 0, direct_io: bool = False, policy_timing: bool = False, eviction: str = "lru") -> List[str]:
    flags: List[str] = []
    if eviction != "lru":
        flags += ["--eviction-policy", eviction]
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
    eviction: str = "lru"


# Every ShilmanDB run to make, in order. Per scale factor and workload: each config
# (once per policy and penalty), then the pool-fraction runs (same). +learned is V2 eviction,
# so it runs once per penalty, as an lru run, whatever the policy list says.
def planned_runs(sfs: Sequence[str], configs: Sequence[str], workloads: Sequence[str], penalties: Sequence[int], fractions: Sequence[float], evictions: Sequence[str] = ("lru",)) -> List[RunSpec]:
    runs: List[RunSpec] = []
    for sf in sfs:
        for workload in workloads:
            for config in configs:
                pool = POOL_OVERRIDES.get(config, POOL_SIZES[sf])
                for eviction in (("lru",) if config == "+learned" else evictions):
                    runs +=[RunSpec(sf, config, workload, pool, penalty, None, eviction) for penalty in penalties]

            for fraction in fractions:
                for eviction in evictions:
                    runs += [RunSpec(sf, fraction_config(workload), workload, pool_for_fraction(sf, fraction), penalty, fraction, eviction) for penalty in penalties]
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
    eviction: str = ""              # from run_info.csv; empty if the binary didn't write it

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
    eviction: str = ""              # from run_info.csv; empty if the binary didn't write it

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
                  extra_flags: Sequence[str] = (), reuse_db: bool = False) -> List[str]:

    flags = config_flags(config_name)
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
    cmd += flags
    cmd += memory_limit_flags(sf, config_name)
    if reuse_db:
        cmd.append("--reuse-db")
    cmd += extra_flags
    return cmd


# Warn (never fail) when the loaded database isn't the size DB_PAGES says it is.
# Older binaries report heap pages only; newer ones add the header pages.
def warn_on_db_pages_mismatch(info: Dict[str, str], sf: str) -> None:
    reported = info.get("db_pages")
    if reported is None or sf not in DB_PAGES:
        return
    if not reported.isdigit() or int(reported) not in (DB_PAGES[sf], DB_PAGES[sf] + HEADER_PAGES):
        eprint(f"  ! warning: run_info.csv db_pages={reported} but DB_PAGES[{sf}]={DB_PAGES[sf]}; "
               "pool fractions may be off")


# The database files the columnar configs share, all in one folder.
# Per scale factor, columnar has a file and the indexed configs share another.
# A kept folder (--columnar-dir) outlives the call: each file has a marker,
# tpch_sf<sf>_<kind>.loaded, with the row counts it was loaded from.
@dataclass
class ColumnarFiles:
    root: Path
    kept: bool = False                          # True: files are reused across calls through markers
    loaded: set = field(default_factory=set)    # files a run that exited 0 loaded
    counts: Dict[Path, Dict[str, int]] = field(default_factory=dict)   # row counts per data folder

    # The file this config uses at this scale factor; None for a row config.
    def db_file(self, sf: str, config_name: str) -> Optional[Path]:
        if not is_columnar_config(config_name):
            return None
        kind = "plain" if config_name == "columnar" else "indexed"
        return self.root / f"tpch_sf{sf}_{kind}.db"

    # The marker beside a kept file.
    @staticmethod
    def marker(db_file: Path) -> Path:
        return db_file.with_suffix(".loaded")

    # The data folder's row counts, read once per call.
    def row_counts(self, sf_data_dir: Path) -> Dict[str, int]:
        if sf_data_dir not in self.counts:
            self.counts[sf_data_dir] = load_sibling("baseline_common").tbl_row_counts(sf_data_dir)
        return self.counts[sf_data_dir]

    # True when a run can open the file instead of loading it.
    # A kept file needs a marker saying it loaded (status 0) from today's row counts.
    def can_reuse(self, db_file: Path, sf_data_dir: Path) -> bool:
        if not self.kept:
            return db_file in self.loaded
        if not db_file.exists():
            return False
        try:
            stored = json.loads(self.marker(db_file).read_text())
            status, counts = stored["exit_status"], stored["row_counts"]
        except (OSError, ValueError, TypeError, KeyError):
            return False
        expected = self.row_counts(sf_data_dir)
        return status == 0 and bool(expected) and counts == expected

    # Before a run loads the file: its old marker no longer holds.
    def start_load(self, db_file: Path) -> None:
        if self.kept:
            self.marker(db_file).unlink(missing_ok=True)

    # After a run that loaded the file exited 0.
    def finish_load(self, db_file: Path, sf_data_dir: Path, exit_status: int) -> None:
        self.loaded.add(db_file)
        if self.kept:
            marker = {"exit_status": exit_status, "row_counts": self.row_counts(sf_data_dir)}
            self.marker(db_file).write_text(json.dumps(marker, indent=1) + "\n")


# Copy a run's load_times.csv, when it wrote one, to <output>/load_times_<label>_sf<sf>.csv.
def copy_load_times(tmp_path: Path, output_dir: Path, label: str, sf: str) -> None:
    src = tmp_path / "load_times.csv"
    if src.exists():
        shutil.copyfile(src, output_dir / f"load_times_{label}_sf{sf}.csv")


# Same for a parallel run's parallel_stats.csv: <output>/parallel_stats_<label>_sf<sf>.csv.
def copy_parallel_stats(tmp_path: Path, output_dir: Path, label: str, sf: str) -> None:
    src = tmp_path / "parallel_stats.csv"
    if src.exists():
        shutil.copyfile(src, output_dir / f"parallel_stats_{label}_sf{sf}.csv")


# Run load_tpch once for a (sf, config) in a temp folder and hand that folder to the caller.
# Gives None instead when the run timed out or failed (the reason is printed).
@contextmanager
def load_tpch_run(binary: Path, sf: str, config_name: str, data_dir: Path, pool_size: int, warmup_runs: int,
                  timed_runs: int, timeout_seconds: Optional[int], label: str,
                  extra_flags: Sequence[str],
                  columnar_files: Optional[ColumnarFiles] = None) -> Iterator[Optional[Path]]:
    
    with tempfile.TemporaryDirectory(prefix=f"shilmandb_{label}_sf{sf}_") as tmp:
        tmp_path = Path(tmp)
        db_file = tmp_path / f"tpch_sf{sf}.db"
        sf_data_dir = data_dir / f"sf{sf}"
        shared_file = columnar_files.db_file(sf, config_name) if columnar_files else None
        reuse_db = False
        if shared_file is not None:
            db_file = shared_file
            reuse_db = columnar_files.can_reuse(shared_file, sf_data_dir)
            if not reuse_db:
                columnar_files.start_load(shared_file)

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
            reuse_db=reuse_db,
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

        if shared_file is not None and not reuse_db:
            columnar_files.finish_load(shared_file, sf_data_dir, proc.returncode)
        yield tmp_path


#spawn load_tpch for one (sf, config) combo
#returns per query stats; label names the results folder and the config column (default: config_name)
def run_shilmandb_config(binary: Path, sf: str, config_name: str, data_dir: Path, output_dir: Path, pool_size: int, 
                         warmup_runs: int, timed_runs: int, timeout_seconds: Optional[int] = None,
                         label: Optional[str] = None,
                         extra_flags: Sequence[str] = (),
                         columnar_files: Optional[ColumnarFiles] = None) -> Optional[Dict[str, QueryStats]]:

    label = label or config_name
    results_parent = output_dir / "shilmandb_results" / label
    results_parent.mkdir(parents=True, exist_ok=True)

    with load_tpch_run(binary, sf, config_name, data_dir, pool_size, warmup_runs, timed_runs, timeout_seconds, label, extra_flags, columnar_files) as tmp_path:
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
        copy_load_times(tmp_path, output_dir, label, sf)
        copy_parallel_stats(tmp_path, output_dir, label, sf)

        info = read_run_info(tmp_path / "run_info.csv")
        # DB_PAGES counts heap pages, so only a row run can be checked against it
        if not is_columnar_config(config_name):
            warn_on_db_pages_mismatch(info, sf)
        policy_ms = read_policy_ms(tmp_path / "policy_time.csv")

        options = parse_label(label)
        stats = parse_shilmandb_latencies(lat_csv, sf=sf, config=label)
        for q, qs in stats.items():
            qs.policy_ms = policy_ms.get(q)
            qs.pool_size = pool_size
            qs.miss_penalty_us = options.miss_penalty_us
            qs.direct_io = options.direct_io
            qs.eviction = info.get("eviction", "")
        return stats


# Same for a stream, lookup or mixed workload (the label says which): the stream CSVs are copied
# to shilmandb_results/<label>/<name>_sf<sf>.csv. None if the run failed or its files are missing.
def run_workload_config(binary: Path, sf: str, config_name: str, data_dir: Path, output_dir: Path, pool_size: int,
                        warmup_runs: int, timed_runs: int, timeout_seconds: Optional[int], label: str,
                        extra_flags: Sequence[str] = (),
                        columnar_files: Optional[ColumnarFiles] = None) -> Optional[WorkloadStats]:

    results_parent = output_dir / "shilmandb_results" / label
    results_parent.mkdir(parents=True, exist_ok=True)

    with load_tpch_run(binary, sf, config_name, data_dir, pool_size, warmup_runs, timed_runs, timeout_seconds, label, extra_flags,
                       columnar_files) as tmp_path:
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
        copy_load_times(tmp_path, output_dir, label, sf)

        info = read_run_info(tmp_path / "run_info.csv")
        if not is_columnar_config(config_name):
            warn_on_db_pages_mismatch(info, sf)
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
            eviction=info.get("eviction", ""),
            **totals,
        )


# SQLite runner adapter
#Invoke bench/tpch/run_sqlite_benchmarks.py as a subprocess
def run_sqlite_benchmarks(scale_factors: Sequence[str], data_dir: Path, output_dir: Path, timed_runs: int) -> bool:
    subsection(f"SQLite benchmarks (SFs: {', '.join(scale_factors)})")
    script = HERE / "run_sqlite_benchmarks.py"
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


# Which engine a config belongs to: shilmandb, sqlite, duckdb or postgres.
def engine_of(config: str) -> str:
    if config.startswith("sqlite_"):
        return "sqlite"
    
    for engine, names in ENGINE_CONFIGS.items():
        if config in names or config in NAMED_ENGINE_CONFIGS[engine]:
            return engine
        
    return "shilmandb"


# Run bench/tpch/run_<engine>_benchmarks.py as a subprocess. False if it failed (it says why).
# configs, when given, goes to it as --configs; else it runs its default configs.
def run_engine_benchmarks(engine: str, scale_factors: Sequence[str], data_dir: Path, output_dir: Path, timed_runs: int, warmup_runs: int = 1,
                          configs: Optional[Sequence[str]] = None) -> bool:
    subsection(f"{REFERENCE_NAMES[engine]} benchmarks (SFs: {', '.join(scale_factors)})")
    script = HERE / f"run_{engine}_benchmarks.py"
    cmd: List[str] = [
        sys.executable, str(script),
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
        "--runs", str(timed_runs),
        "--warmup", str(warmup_runs),
    ]
    for sf in scale_factors:
        cmd += ["--sf", sf]
    if configs:
        cmd += ["--configs", ",".join(configs)]

    print(f"  $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    print(f"  (wall: {elapsed_str(time.time() - t0)}, exit: {proc.returncode})", flush=True)
    return proc.returncode == 0


# Median latency per (sf, config, query) from a duckdb_ or postgres_latencies.csv.
def parse_engine_latencies(path: Path, scale_factors: Sequence[str]) -> Dict[Tuple[str, str, str], QueryStats]:
    if not path.exists():
        eprint(f"! missing {path}")
        return {}

    buckets: Dict[Tuple[str, str, str], List[float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["sf"] in scale_factors:
                buckets.setdefault((row["sf"], row["config"], row["query"]), []).append(float(row["latency_ms"]))

    return {
        (sf, config, query): QueryStats(sf=sf, config=config, query=query, median_latency_ms=statistics.median(lats), rows=None, median_hits=None, median_misses=None)
        for (sf, config, query), lats in buckets.items()
    }


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
    "pool_size", "miss_penalty_us", "direct_io", "eviction",
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
                qs.eviction,
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
        if engine_of(config) != "shilmandb":
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
        if engine_of(name) != "shilmandb": return 3
        return 4

    keys = sorted(raw_rates.keys(), key=lambda k: (k[0], _config_rank(k[1]), k[1]))
    eviction_by_config = {(sf, config): stats_list[0].eviction for (sf, config), stats_list in by_sc.items()}

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sf", "config", "pool_size", "pool_fraction", "miss_penalty_us", "direct_io", "hit_rate", "raw_hit_rate", "improvement_over_lru", "eviction"])

        for (sf, config) in keys:
            ps = pool_sizes_by_config.get((sf, config), "")
            if engine_of(config) != "shilmandb":
                w.writerow([sf, config, ps, "", "", "", "", "", "", ""])
                continue

            options = parse_label(config)
            rate = page_rates[(sf, config)]
            baseline_label = config_label("baseline", options.miss_penalty_us, options.direct_io, options.fraction, options.workload, options.seed or 1)
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
                eviction_by_config.get((sf, config), ""),
            ])

    print(f"  -> {path}")


WORKLOADS_HEADER: List[str] = [
    "sf", "config", "workload", "seed", "items", "timed_passes", "pool_size", "pool_fraction",
    "miss_penalty_us", "direct_io", "median_total_ms", "misses", "visits", "page_hit_rate",
    "reads", "read_ms", "us_per_read", "query_misses", "lookup_misses", "policy_ms", "correct", "eviction",
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
                _fmt_float(r.policy_ms), r.correct, r.eviction,
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


# Load bench/tpch/check_stream_results.py by path. It is only needed for workload runs.
def load_stream_checker() -> ModuleType:
    return load_sibling("check_stream_results")


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


# compare_results.py prints one line per comparison, then indented detail lines:
#   "  PASS  [<config>] <Qtag> SF=<sf>"
#   "  DRIFT [<config>] <Qtag> SF=<sf>  (max abs <x>, max rel <y>)"
#   "  FAIL  [<config>] <Qtag> SF=<sf>  (N mismatch(es))"
#   "  SKIP  [<config>] <Qtag> SF=<sf>"
COMPARE_LINE_RE = re.compile(
    r"^(?P<status>PASS|DRIFT|FAIL|SKIP)\s+\[(?P<config>[^\]]+)\]\s+(?P<query>Q\d+)\s+SF=(?P<sf>\S+)"
    r"(?:\s+\(max abs (?P<abs>[^,\s]+), max rel (?P<rel>[^)\s]+)\))?"
)
STATUS_ORDER: List[str] = ["SKIP", "PASS", "DRIFT", "FAIL"]     # best to worst


# The worst of a config's statuses over its references; SKIP when it has none.
def worst_status(statuses: Sequence[str]) -> str:
    return max(statuses, key=STATUS_ORDER.index, default="SKIP")


# Run compare_results.py; returns its stdout and stderr, and repeats any stderr on ours.
# A normal run writes nothing to stderr, so anything there means it crashed.
def compare_output(cmd: List[str]) -> Tuple[str, str]:
    print(f"  $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if proc.stderr.strip():
        eprint(f"! compare_results.py wrote to stderr (exit {proc.returncode}):\n{proc.stderr.rstrip()}")
    return proc.stdout, proc.stderr


# {query: (status, max_abs, max_rel)} from compare_results.py's output; the diffs only for DRIFT.
def parse_compare_output(text: str) -> Dict[str, Tuple[str, str, str]]:
    seen: Dict[str, Tuple[str, str, str]] = {}
    for raw_line in text.splitlines():
        m = COMPARE_LINE_RE.match(raw_line.strip())
        if not m or m.group("query") in seen:
            continue
        drift = m.group("status") == "DRIFT"
        seen[m.group("query")] = (m.group("status"), (m.group("abs") or "") if drift else "", (m.group("rel") or "") if drift else "")
    return seen


# The answers configs are checked against at one SF: (name, folder, config folder in it).
# SF=10 uses DuckDB and Postgres when their flags are on; with neither, SQLite as before.
def references_for(sf: str, duckdb: bool, postgres: bool, output_dir: Path = DEFAULT_OUTPUT_DIR) -> List[Tuple[str, Path, Optional[str]]]:
    sqlite = [(REFERENCE_NAMES["sqlite"], output_dir / "sqlite_results", None)]
    if sf in ALL_SCALE_FACTORS or not (duckdb or postgres):
        return sqlite
    refs: List[Tuple[str, Path, Optional[str]]] = []
    if duckdb:
        refs.append((REFERENCE_NAMES["duckdb"], output_dir / "duckdb_results", "duckdb"))
    if postgres:
        refs.append((REFERENCE_NAMES["postgres"], output_dir / "postgres_results", "postgres"))
    return refs


# Check every config with scripts/compare_results.py against each of its references (never its own engine).
# correctness.csv gets the worst status per query; checks_path, when given, gets every check.
# SQLite configs, configs of a failed engine and configs left with no reference are SKIP.
# A query with no line from a compare that crashed is FAIL, else SKIP.
# At SF=100 a cell that would be DRIFT is FAIL (NO_DRIFT_SCALE_FACTORS).
def write_correctness_csv(path: Path, sfs: Sequence[str], configs: Sequence[str], output_dir: Path,
                          queries: Sequence[str] = tuple(QUERIES), duckdb: bool = False, postgres: bool = False,
                          failed_engines: Sequence[str] = (), checks_path: Optional[Path] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    script = REPO_ROOT / "scripts" / "compare_results.py"
    failed_names = {REFERENCE_NAMES[e] for e in failed_engines}

    rows: List[Tuple[str, str, str, str]] = []
    checks: List[Tuple[str, str, str, str, str, str, str]] = []

    for sf in sfs:
        for q in queries:
            for c in configs:
                if engine_of(c) == "sqlite":
                    rows.append((sf, q, c, "SKIP"))

    for sf in sfs:
        refs = [r for r in references_for(sf, duckdb, postgres, output_dir) if r[0] not in failed_names]
        for cfg in configs:
            engine = engine_of(cfg)
            if engine == "sqlite":
                continue
            
            statuses: Dict[str, List[str]] = {q: [] for q in queries}
            if engine not in failed_engines:
                for name, ref_dir, ref_config in refs:
                    if name == REFERENCE_NAMES.get(engine):
                        continue
                    cmd = [
                        sys.executable, str(script),
                        "--shilmandb-dir", str(output_dir / RESULTS_DIRS[engine]),
                        "--reference-dir", str(ref_dir),
                        "--reference-name", name,
                        "--sf", sf,
                        "--configs", cfg,
                    ] + (["--reference-config", ref_config] if ref_config else [])
                    if sf in NO_DRIFT_SCALE_FACTORS:
                        cmd.append("--no-drift")
                    stdout, stderr = compare_output(cmd)
                    seen = parse_compare_output(stdout)
                    missing = "FAIL" if stderr.strip() else "SKIP"
                    for q in queries:
                        status, max_abs, max_rel = seen.get(q, (missing, "", ""))
                        statuses[q].append(status)
                        checks.append((sf, q, cfg, name, status, max_abs, max_rel))
            for q in queries:
                rows.append((sf, q, cfg, worst_status(statuses[q])))

    query_order = {q: i for i, q in enumerate(QUERIES)}
    rows.sort(key=lambda r: (r[0], query_order.get(r[1], 999), r[2]))

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sf", "query", "config", "status"])
        for row in rows:
            w.writerow(row)
    print(f"  -> {path}")

    if checks_path is not None:
        checks.sort(key=lambda r: (r[0], query_order.get(r[1], 999), r[2], r[3]))
        with checks_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sf", "query", "config", "reference", "status", "max_abs_diff", "max_rel_diff"])
            for check in checks:
                w.writerow(check)
        print(f"  -> {checks_path}")


# Environment record

# One command's trimmed output, or "" when it can't run or fails.
def command_output(cmd: Sequence[str]) -> str:
    try:
        proc = subprocess.run(list(cmd), cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


# An installed package's version, or "" when it isn't installed.
def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


# The C++ compiler (path, id and version) and build type of the Release build in build/.
def build_info(build_dir: Path = REPO_ROOT / "build") -> Tuple[str, str]:
    cache: Dict[str, str] = {}
    cache_path = build_dir / "CMakeCache.txt"
    if cache_path.exists():
        for line in cache_path.read_text(errors="replace").splitlines():
            m = re.match(r"^(CMAKE_CXX_COMPILER|CMAKE_BUILD_TYPE|CMAKE_CACHE_(?:MAJOR|MINOR|PATCH)_VERSION):[A-Z]+=(.*)$", line)
            if m:
                cache[m.group(1)] = m.group(2)

    compiler = cache.get("CMAKE_CXX_COMPILER", "")
    cmake_file = _compiler_file(build_dir, cache)
    if cmake_file is not None and compiler:
        text = cmake_file.read_text(errors="replace")
        found = [re.search(rf'set\({name} "([^"]*)"\)', text) for name in ("CMAKE_CXX_COMPILER_ID", "CMAKE_CXX_COMPILER_VERSION")]
        if all(found):
            compiler += f" ({found[0].group(1)} {found[1].group(1)})"
    return compiler, cache.get("CMAKE_BUILD_TYPE", "")


# CMake's compiler file for the CMake version the cache records, else the newest one.
# An older CMake can leave its own version folder behind.
def _compiler_file(build_dir: Path, cache: Dict[str, str]) -> Optional[Path]:
    version = ".".join(cache.get(f"CMAKE_CACHE_{part}_VERSION", "") for part in ("MAJOR", "MINOR", "PATCH"))
    exact = build_dir / "CMakeFiles" / version / "CMakeCXXCompiler.cmake"
    if exact.exists():
        return exact
    found = list(build_dir.glob("CMakeFiles/*/CMakeCXXCompiler.cmake"))
    return max(found, key=lambda f: f.stat().st_mtime) if found else None


# Machine, build and engine versions for environment.csv; anything unknown is "".
def environment_record(duckdb_info: Dict[str, str], postgres_info: Dict[str, str]) -> Dict[str, str]:
    compiler, build_type = build_info()
    libtorch_file = REPO_ROOT / "third_party" / "libtorch" / "build-version"
    record = {
        "date": datetime.date.today().isoformat(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "chip": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memory_bytes": command_output(["sysctl", "-n", "hw.memsize"]),
        "macos": command_output(["sw_vers", "-productVersion"]),
        "compiler": compiler,
        "build_type": build_type,
        "sqlite": sqlite3.sqlite_version,
        "duckdb": duckdb_info.get("duckdb_version") or package_version("duckdb"),
        "postgres": postgres_info.get("postgres_version", ""),
        "psycopg": package_version("psycopg"),
        "libtorch": libtorch_file.read_text().strip() if libtorch_file.exists() else "",
        "pytorch": package_version("torch"),
    }
    record.update({name: postgres_info.get(name, "") for name in POSTGRES_SETTING_NAMES})
    return record


def write_environment_csv(path: Path, record: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for key, value in record.items():
            w.writerow([key, value])
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


# --parallel-threads: whole numbers from 1 to 256, duplicates dropped, order kept.
def parallel_threads_list(text: str) -> List[int]:
    message = f"--parallel-threads expects whole numbers from 1 to {MAX_PARALLEL_THREADS}, e.g. 2,4,6,12"
    values = [v.strip() for v in text.split(",") if v.strip()]
    if not values or not all(v.isascii() and v.isdigit() and 1 <= int(v) <= MAX_PARALLEL_THREADS for v in values):
        raise argparse.ArgumentTypeError(message)
    return list(dict.fromkeys(int(v) for v in values))


def eviction_policy_list(text: str) -> List[str]:
    values = _comma_list(text, str.strip, "--eviction-policies expects names")
    if not values or any(v not in EVICTION_POLICIES for v in values):
        raise argparse.ArgumentTypeError(f"--eviction-policies expects names from {','.join(EVICTION_POLICIES)}, e.g. lru,arc")
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


# Why these arguments can't run together, or None. Checked before anything runs.
def argument_refusal(sfs: Sequence[str], configs: Sequence[str], direct_io: bool, engines: Sequence[str],
                     pool_fractions: Sequence[float], v1_eviction: bool, workloads: Sequence[str]) -> Optional[str]:
    for config in configs:
        engine = engine_of(config)
        if engine in ENGINE_CONFIGS and engine not in engines:
            return f"{config} runs only with --{engine}"
    if direct_io and "columnar+parallel_direct" in configs:
        return "columnar+parallel_direct already reads with direct I/O; drop --direct-io or the config"

    big = [sf for sf in sfs if sf in PARALLEL_ONLY_SCALE_FACTORS]
    if not big:
        return None
    # Nothing but the parallel executor has a memory limit, so the rest would swap
    refused = [c for c in configs if engine_of(c) in ("shilmandb", "sqlite") and not is_parallel_config(c)]
    refused += ["--pool-fractions"] if pool_fractions else []
    refused += ["--v1-eviction"] if v1_eviction else []
    refused += [f"--workloads {w}" for w in workloads if w != "single"]
    if refused:
        return f"only the parallel configs run at SF={big[0]}; refused: {', '.join(refused)}"
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TPC-H benchmark runner", epilog="Exit 3: --duckdb or --postgres was asked for and its runner failed; every other output is still written.")
    p.add_argument("--binary", type=Path, default=DEFAULT_BINARY, help=f"Path to load_tpch binary (default: {DEFAULT_BINARY})")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help=f"TPC-H data dir (default: {DEFAULT_DATA_DIR})")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Output dir (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--sf", action="append", choices=KNOWN_SCALE_FACTORS + ["all"], help="Scale factor(s). Repeat flag or use 'all'. Default: 'all'.")
    p.add_argument("--configs", default="all", help=("Comma-separated configs. Accepts 'all' (" + ",".join(ALL_CONFIG_NAMES) + "), 'none' (no ShilmanDB or SQLite config) or a subset of: " + ",".join(KNOWN_CONFIG_NAMES) + ". The columnar configs run only when named; columnar+index loads an indexed file that later columnar+vectorized and parallel runs (and the --parallel-threads sweep) reuse. duckdb_20g runs only with --duckdb. At SF=100 only the parallel configs run"))
    p.add_argument("--parallel-threads", type=parallel_threads_list, default=[], help=f"Comma list of thread counts (1 to {MAX_PARALLEL_THREADS}); each adds columnar+parallel_<N>thread (columnar+parallel with --threads N) after the named configs. 1 is columnar+parallel_1thread")
    p.add_argument("--duckdb", action="store_true", help="Also run DuckDB (duckdb, duckdb_1thread, and duckdb_20g when --configs names it) through run_duckdb_benchmarks.py, after ShilmanDB; exit 3 if it fails")
    p.add_argument("--postgres", action="store_true", help="Also run Postgres (postgres, postgres_1worker) through run_postgres_benchmarks.py, last; needs the server on port 5434; exit 3 if it fails")
    p.add_argument("--warmup", type=int, default=1, help="Warmup runs per query, DuckDB and Postgres included (default 1)")
    p.add_argument("--runs", type=int, default=5, help="Timed runs per query (default 5)")
    p.add_argument("--pool-sweep", action="store_true", help=f"Also run baseline pool-size sweep at SF={POOL_SWEEP_SF}")
    p.add_argument("--v1-eviction", action="store_true", help="Also run +learned with V1 eviction model per SF")
    p.add_argument("--miss-penalty-us", type=penalty_list, default=[0], help="Comma list of simulated per-read delays in microseconds; every ShilmanDB config runs once per value (default 0)")
    p.add_argument("--direct-io", action="store_true", help="Read pages around the OS file cache (adds _direct to labels); parallel single-query runs also get --cold. Not with the columnar+parallel_direct config, which does the same")
    p.add_argument("--pool-fractions", type=fraction_list, default=[], help="Comma list of pool sizes as fractions of the tables' pages (0 to 1]; runs baseline once per fraction per SF (+index for the lookup and mixed workloads)")
    p.add_argument("--policy-timing", action="store_true", help="Also time the eviction policy per query (fills policy_ms)")
    p.add_argument("--eviction-policies", type=eviction_policy_list, default=["lru"], help=f"Comma list of eviction policies from {','.join(EVICTION_POLICIES)}; every ShilmanDB config runs once per policy (default lru). +learned is V2 eviction, so it runs once, not once per policy. single runs start from the pool the load left (warm); the other workloads start from an empty pool (cold)")
    p.add_argument("--workloads", type=workload_list, default=["single"], help=f"Comma list of workloads from {','.join(WORKLOADS)}; single is one query at a time (default single). lookup and mixed run on every config; without indexes each lookup scans its table")
    p.add_argument("--seed", type=seed_value, default=1, help="Seed for the random queries of the non-single workloads (default 1)")
    p.add_argument("--stream-length", type=stream_length_value, default=None, help="Items per pass for every non-single workload (default: load_tpch's default for the scale factor)")
    p.add_argument("--skip-sqlite", action="store_true", help="Skip the SQLite benchmark invocation (reuse existing output)")
    p.add_argument("--skip-correctness", action="store_true", help="Skip the correctness comparison step")
    p.add_argument("--timeout", type=int, default=None, help="Per-config subprocess timeout in seconds (default: none)")
    p.add_argument("--columnar-dir", type=Path, default=None, help="Keep the shared columnar files here (made if missing, never deleted); a later call reuses a file whose .loaded marker matches the data's row counts (default: a temp folder for this call)")
    return p.parse_args()


def resolve_sfs(arg: Optional[Sequence[str]]) -> List[str]:
    if not arg or "all" in arg:
        return list(ALL_SCALE_FACTORS)
    s = set(arg)
    return [sf for sf in KNOWN_SCALE_FACTORS if sf in s]


def resolve_configs(arg: str) -> List[str]:
    if arg == "all":
        return list(ALL_CONFIG_NAMES)
    requested = [c.strip() for c in arg.split(",") if c.strip()]
    if "none" in requested:
        if requested != ["none"]:
            raise SystemExit("'none' can't be mixed with other configs.")
        return []
    for c in requested:
        if c not in KNOWN_CONFIG_NAMES:
            raise SystemExit(f"unknown config: {c!r}. Choose from: {','.join(KNOWN_CONFIG_NAMES)} or 'all'.")
    return requested


def main() -> int:
    args = parse_args()
    sfs = resolve_sfs(args.sf)
    configs = resolve_configs(args.configs)
    engines = [e for e, on in (("duckdb", args.duckdb), ("postgres", args.postgres)) if on]
    refusal = argument_refusal(sfs, configs, args.direct_io, engines, args.pool_fractions, args.v1_eviction, args.workloads)
    if refusal is not None:
        raise SystemExit(f"Error: {refusal}")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    section(f"TPC-H benchmark runner | SFs={sfs} | configs={configs}")
    print(f"  warmup={args.warmup}, runs={args.runs}, "
          f"pool_sweep={args.pool_sweep}, v1_eviction={args.v1_eviction}")
    if args.miss_penalty_us != [0] or args.direct_io or args.pool_fractions or args.policy_timing:
        print(f"  miss_penalty_us={args.miss_penalty_us}, direct_io={args.direct_io}, "
              f"pool_fractions={args.pool_fractions}, policy_timing={args.policy_timing}")
    if args.parallel_threads:
        print(f"  parallel_threads={args.parallel_threads}")
    has_workloads = any(w != "single" for w in args.workloads)
    if has_workloads:
        print(f"  workloads={args.workloads}, seed={args.seed}, stream_length={args.stream_length}")
    if engines:
        print(f"  engines={engines}")

    shilmandb_configs = [c for c in configs if c in PRIMARY_CONFIGS]
    shilmandb_configs += sweep_configs(args.parallel_threads, shilmandb_configs)
    if "+learned" in shilmandb_configs and args.eviction_policies != ["lru"]:
        print("  note: +learned is V2 eviction, so it runs once, not once per policy")
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

    # DuckDB and Postgres run after ShilmanDB, Postgres last.
    # A runner that fails is reported, and its configs show SKIP.
    # A call without the single workload skips them, as if their flags were off.
    tried_engines = engines if runs_single else []
    engine_configs: List[str] = []
    failed_engines: List[str] = []
    engine_infos: Dict[str, Dict[str, str]] = {}          # only engines that ran in this call
    if engines and not runs_single:
        print("  (skip: DuckDB and Postgres only run with the single workload)")

    ran_labels: List[str] = []      # every single-query ShilmanDB label that was run, for the correctness check
    workload_records: List[WorkloadStats] = []

    def run_single(spec: RunSpec, columnar_files: Optional[ColumnarFiles]) -> None:
        label = config_label(spec.base, spec.penalty, args.direct_io, spec.fraction, eviction=spec.eviction)
        section(f"ShilmanDB {label} @ SF={spec.sf}")
        stats = run_shilmandb_config(
            binary=args.binary, sf=spec.sf, config_name=spec.base,
            data_dir=args.data_dir, output_dir=output_dir,
            pool_size=spec.pool, warmup_runs=args.warmup, timed_runs=args.runs,
            timeout_seconds=args.timeout, label=label,
            extra_flags=option_flags(spec.penalty, args.direct_io, args.policy_timing, spec.eviction) + cold_flags(spec.base, args.direct_io),
            columnar_files=columnar_files,
        )
        pool_sizes_by_config[(spec.sf, label)] = spec.pool
        if label not in ran_labels:
            ran_labels.append(label)
        if stats is None:
            return
        for q, qs in stats.items():
            all_records[(spec.sf, label, q)] = qs

    def run_workload(spec: RunSpec, columnar_files: Optional[ColumnarFiles]) -> None:
        label = config_label(spec.base, spec.penalty, args.direct_io, spec.fraction, spec.workload, args.seed, spec.eviction)
        section(f"ShilmanDB {label} @ SF={spec.sf}")
        stats = run_workload_config(
            binary=args.binary, sf=spec.sf, config_name=spec.base,
            data_dir=args.data_dir, output_dir=output_dir,
            pool_size=spec.pool, warmup_runs=args.warmup, timed_runs=args.runs,
            timeout_seconds=args.timeout, label=label,
            extra_flags=(workload_flags(spec.workload, args.seed, args.stream_length) + option_flags(spec.penalty, args.direct_io, args.policy_timing, spec.eviction)),
            columnar_files=columnar_files,
        )
        if stats is None:
            eprint(f"  ! {label} @ SF={spec.sf} failed - skipping")
            return
        workload_records.append(stats)

    # Columnar files last this call only, removed at the end even on error,
    # unless --columnar-dir keeps them
    uses_columnar = any(is_columnar_config(c) for c in shilmandb_configs)
    kept = uses_columnar and args.columnar_dir is not None
    if kept:
        args.columnar_dir.mkdir(parents=True, exist_ok=True)
    shared = (nullcontext(str(args.columnar_dir)) if kept else
              tempfile.TemporaryDirectory(prefix="shilmandb_columnar_") if uses_columnar else nullcontext())
    with shared as shared_dir:
        columnar_files = ColumnarFiles(Path(shared_dir), kept=kept) if shared_dir else None
        for spec in planned_runs(sfs, shilmandb_configs, args.workloads, args.miss_penalty_us, args.pool_fractions, args.eviction_policies):
            (run_single if spec.workload == "single" else run_workload)(spec, columnar_files)

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

    for engine in tried_engines:
        names = ENGINE_CONFIGS[engine] + [c for c in NAMED_ENGINE_CONFIGS[engine] if c in configs]
        engine_configs += names
        if not run_engine_benchmarks(engine, sfs, args.data_dir, output_dir, args.runs, args.warmup, names):
            eprint(f"! {engine} benchmarks failed - continuing without its rows")
            failed_engines.append(engine)
            continue
        engine_infos[engine] = read_run_info(output_dir / f"{engine}_info.csv")
        for key, qs in parse_engine_latencies(output_dir / f"{engine}_latencies.csv", sfs).items():
            all_records[key] = qs
            pool_sizes_by_config[(key[0], key[1])] = ""  # type: ignore[assignment]

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
                        + ([f"baseline_pool{ps}" for ps in POOL_SWEEP_SIZES] if args.pool_sweep else [])
                        + engine_configs,
                output_dir=output_dir,
                duckdb="duckdb" in tried_engines,
                postgres="postgres" in tried_engines,
                failed_engines=failed_engines,
                checks_path=(output_dir / "reference_checks.csv") if engine_infos else None,
            )

    if engine_infos:
        write_environment_csv(output_dir / "environment.csv", environment_record(engine_infos.get("duckdb", {}), engine_infos.get("postgres", {})))

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
            if engine_infos:
                print(f"  reference_checks -> {output_dir / 'reference_checks.csv'}")
                
    if engine_infos:
        print(f"  environment -> {output_dir / 'environment.csv'}")
    if has_workloads:
        print(f"  workloads  -> {output_dir / 'workloads.csv'}")
        print(f"  workload_queries -> {output_dir / 'workload_queries.csv'}")
        if not args.skip_correctness:
            print(f"  stream_correctness -> {output_dir / 'stream_correctness.csv'}")
    if failed_engines:
        eprint(f"! failed: {', '.join(failed_engines)} (exit 3)")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
