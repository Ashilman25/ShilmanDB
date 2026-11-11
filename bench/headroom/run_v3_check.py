#!/usr/bin/env python3
#Runs every step of the learned-eviction check for one scale factor.

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_rb = _load("run_benchmarks_for_v3", REPO_ROOT / "bench" / "run_benchmarks.py")
pool_for_fraction = _rb.pool_for_fraction
_train = _load("train_v3_check_for_driver", HERE / "train_v3_check.py")
TRAIN_TRACES, REPLAYS, CELLS = _train.TRAIN_TRACES, _train.REPLAYS, _train.CELLS
ablation_cell, sample_prefix, model_path = _train.ablation_cell, _train.sample_prefix, _train.model_path

TEST_TRACES = ("index_mixed_s1", "index_mixed_s2", "index_mixed_s3", "index_stream_s3", "index_lookup_s3")
MIXED = TEST_TRACES[:3]
ALL_STEPS = ("owners", "samples", "train", "validate", "replay", "time", "ablation", "analyze")
SF_DEFAULTS = {
    "0.01": {"fractions": (0.1, 0.25, 0.5), "tests": TEST_TRACES, "steps": ALL_STEPS},
    "0.1": {"fractions": (0.1, 0.25, 0.5), "tests": TEST_TRACES, "steps": ALL_STEPS},
    "1.0": {"fractions": (0.25,), "tests": MIXED, "steps": tuple(s for s in ALL_STEPS if s != "ablation")},
}
VALIDATE_POLICIES = "lru,arc_visit,lirs_visit,belady,sampled_random,sampled_lru,sampled_label,sampled_belady"
SAMPLE_K = 32
SAMPLE_SEED = 1


@dataclass(frozen=True)
class Paths:
    load_tpch: Path
    sim: Path
    samples: Path
    python: Path
    traces: Path
    sim_dir: Path
    out: Path
    data_dir: Path

    def sf_dir(self, sf: str) -> Path:
        return self.out / f"sf{sf}"

    def trace_dir(self, sf: str) -> Path:
        return self.traces / f"sf{sf}"

    def owners(self, sf: str) -> Path:
        return self.sf_dir(sf) / "page_owners.csv"


def pools(sf: str, fractions) -> List[int]:
    return [pool_for_fraction(sf, f) for f in fractions]


def _trace_args(sf: str, paths: Paths, trace: str) -> List[str]:
    td = paths.trace_dir(sf)
    return ["--trace", str(td / f"{trace}.csv"), "--items", str(td / trace / "stream.csv"), "--name", trace]


def owners_db(sf: str) -> str:
    return f"/tmp/v3_owners_sf{sf}.db"


def owners_cmds(sf: str, paths: Paths) -> List[List[str]]:
    return [
        ["rm", "-f", owners_db(sf)],
        [str(paths.load_tpch), "--sf", sf, "--db-file", owners_db(sf),
         "--data-dir", str(paths.data_dir / f"sf{sf}") + "/", "--create-indexes", "true",
         "--dump-page-owners", str(paths.owners(sf))],
    ]


# Fail if the page map's total differs from the traces' database,
# or if any page has no owner.
def check_owner_dump(csv_path: Path, expected_total: int) -> None:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
        
    if len(rows) != expected_total:
        raise ValueError(f"page map {csv_path} has {len(rows)} pages, expected {expected_total}")
    
    other = [r["page_id"] for r in rows if r["kind"] == "other"]
    if other:
        raise ValueError(f"page map {csv_path} has {len(other)} 'other' pages, first is page {other[0]}")


def expected_total(sf: str, paths: Paths) -> int:
    info = paths.trace_dir(sf) / "index_stream_s1" / "run_info.csv"
    with open(info, newline="") as f:
        for row in csv.reader(f):
            if row and row[0] == "db_pages_total":
                return int(row[1])
            
    raise ValueError(f"no db_pages_total in {info}")


def samples_cmds(sf: str, paths: Paths, pool_list, decisions: int) -> List[List[str]]:
    samples_dir = paths.sf_dir(sf) / "samples"
    cmds = []
    for pool in pool_list:
        for trace in TRAIN_TRACES:
            for replay in REPLAYS:
                cmds.append([
                    str(paths.samples), *_trace_args(sf, paths, trace), "--sf", sf, "--pool", str(pool),
                    "--replay", replay, "--page-owners", str(paths.owners(sf)),
                    "--decisions", str(decisions), "--sample-k", str(SAMPLE_K), "--seed", str(SAMPLE_SEED),
                    "--out", str(sample_prefix(samples_dir, trace, replay, pool)),
                ])
    return cmds


def train_cmds(sf: str, paths: Paths, pool_list, cells) -> List[List[str]]:
    d = paths.sf_dir(sf)
    return [
        [str(paths.python), str(HERE / "train_v3_check.py"), "--samples-dir", str(d / "samples"), "--sf", sf,
         "--pool", str(pool), "--out-dir", str(d / "models"), "--cells", ",".join(cells)]
        for pool in pool_list
    ]


def validate_cmds(sf: str, paths: Paths, pool_list, tests) -> List[List[str]]:
    replay = paths.sf_dir(sf) / "replay"
    return [
        [str(paths.sim), *_trace_args(sf, paths, t), "--pools", ",".join(map(str, pool_list)),
         "--policies", VALIDATE_POLICIES, "--sample-k", str(SAMPLE_K), "--sample-seed", str(SAMPLE_SEED),
         "--jobs", "1", "--out-prefix", str(replay / f"{t}_classical")]
        for t in tests
    ]


def replay_cmds(sf: str, paths: Paths, pool_list, tests, cells) -> List[List[str]]:
    d = paths.sf_dir(sf)
    return [
        [str(paths.sim), *_trace_args(sf, paths, t), "--pools", str(pool), "--policies", "learned",
         "--model", str(model_path(d / "models", cell, pool)), "--learned-name", cell,
         "--page-owners", str(paths.owners(sf)), "--sample-k", str(SAMPLE_K), "--sample-seed", str(SAMPLE_SEED),
         "--jobs", "1", "--out-prefix", str(d / "replay" / f"{t}_{cell}_p{pool}")]
        for cell in cells for pool in pool_list for t in tests
    ]


def time_cmds(sf: str, paths: Paths, pool_list, cells) -> List[List[str]]:
    d = paths.sf_dir(sf)
    pool = min(pool_list)
    return [
        [str(paths.sim), *_trace_args(sf, paths, "index_mixed_s1"), "--pools", str(pool), "--policies", "learned",
         "--model", str(model_path(d / "models", cell, pool)), "--learned-name", cell,
         "--page-owners", str(paths.owners(sf)), "--sample-k", str(SAMPLE_K), "--sample-seed", str(SAMPLE_SEED),
         "--jobs", "1", "--time-decisions", "--out-prefix", str(d / "timing" / cell)]
        for cell in cells
    ]


def analyze_cmd(paths: Paths) -> List[str]:
    return [str(paths.python), str(HERE / "analyze_v3_check.py"), "--out", str(paths.out), "--sim-dir", str(paths.sim_dir)]


class Driver:
    def __init__(self, dry_run: bool, jobs: int, log_dir: Path):
        self.dry_run, self.jobs, self.log_dir = dry_run, max(1, jobs), log_dir
        self.count = 0

    @staticmethod
    def show(cmd: List[str]) -> None:
        print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)

    def run(self, cmd: List[str]) -> None:
        self.show(cmd)
        if self.dry_run:
            return
        
        code = subprocess.run(cmd).returncode
        if code != 0:
            print(f"command failed with exit code {code}", file=sys.stderr)
            raise SystemExit(code)

    def _logged(self, cmd: List[str], log: Path) -> int:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as f:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode

    # Runs up to `jobs` at once; after a failure nothing new starts, running ones finish.
    def run_parallel(self, cmds: List[List[str]], label: str) -> None:
        if self.dry_run:
            for c in cmds:
                self.show(c)
            return
        
        pending, running, failed = list(enumerate(cmds)), {}, None
        with ThreadPoolExecutor(max_workers=self.jobs) as pool:
            while pending or running:
                while pending and failed is None and len(running) < self.jobs:
                    i, c = pending.pop(0)
                    self.show(c)
                    log = self.log_dir / f"{label}_{i:04d}.log"
                    running[pool.submit(self._logged, c, log)] = (c, log)
                    
                if not running:
                    break
                
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for fut in done:
                    c, log = running.pop(fut)
                    code = fut.result()
                    if code != 0 and failed is None:
                        failed = code
                        print(f"command failed with exit code {code}, output in {log}", file=sys.stderr)
                        
        if failed is not None:
            raise SystemExit(failed)


def step_owners(d: Driver, args, paths: Paths) -> None:
    if not d.dry_run:
        paths.sf_dir(args.sf).mkdir(parents=True, exist_ok=True)
        
    for c in owners_cmds(args.sf, paths):
        d.run(c)
        
    if d.dry_run:
        return

    try:
        check_owner_dump(paths.owners(args.sf), expected_total(args.sf, paths))
    except (ValueError, OSError) as e:
        raise SystemExit(f"page map check failed: {e}")
    finally:
        Path(owners_db(args.sf)).unlink(missing_ok=True)


def step_samples(d: Driver, args, paths: Paths) -> None:
    if not d.dry_run:
        (paths.sf_dir(args.sf) / "samples").mkdir(parents=True, exist_ok=True)
    d.run_parallel(samples_cmds(args.sf, paths, args.pools, args.decisions), "samples")


def step_train(d: Driver, args, paths: Paths) -> None:
    for c in train_cmds(args.sf, paths, args.pools, args.cells):
        d.run(c)


def step_validate(d: Driver, args, paths: Paths) -> None:
    if not d.dry_run:
        (paths.sf_dir(args.sf) / "replay").mkdir(parents=True, exist_ok=True)
    d.run_parallel(validate_cmds(args.sf, paths, args.pools, args.tests), "validate")


def step_replay(d: Driver, args, paths: Paths) -> None:
    if not d.dry_run:
        (paths.sf_dir(args.sf) / "replay").mkdir(parents=True, exist_ok=True)
    d.run_parallel(replay_cmds(args.sf, paths, args.pools, args.tests, args.cells), "replay")


def step_ablation(d: Driver, args, paths: Paths) -> None:
    if d.dry_run:
        best = "<best richer cell>"
        name = f"{best}_no_owner"
    else:
        analyze = _load("analyze_v3_check_for_driver", HERE / "analyze_v3_check.py")
        best = analyze.best_richer_cell(analyze.load_gates(paths.out, args.sf), analyze.load_timings(paths.out, args.sf))
        if best is None:
            raise SystemExit("ablation needs a complete richer cell; run replay for the richer cells first")
        name = ablation_cell(best).name
        (paths.sf_dir(args.sf) / "replay").mkdir(parents=True, exist_ok=True)
    print(f"ablation on {best}")
    
    for c in train_cmds(args.sf, paths, args.pools, [name]):
        d.run(c)
    d.run_parallel(replay_cmds(args.sf, paths, args.pools, args.tests, [name]), "ablation")


def step_time(d: Driver, args, paths: Paths) -> None:
    if not d.dry_run:
        (paths.sf_dir(args.sf) / "timing").mkdir(parents=True, exist_ok=True)
    for c in time_cmds(args.sf, paths, args.pools, args.cells):
        d.run(c)


def step_analyze(d: Driver, args, paths: Paths) -> None:
    d.run(analyze_cmd(paths))


STEP_FUNCS = {
    "owners": step_owners, "samples": step_samples, "train": step_train, "validate": step_validate,
    "replay": step_replay, "ablation": step_ablation, "time": step_time, "analyze": step_analyze,
}


def _list(text: str) -> List[str]:
    return [x for x in text.split(",") if x]


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sf", required=True, choices=list(SF_DEFAULTS))
    p.add_argument("--steps", type=_list, default=None, help=f"From {','.join(ALL_STEPS)} (default: all for the SF)")
    p.add_argument("--cells", type=_list, default=None, help="Model cells (default all 8; required for SF=1.0)")
    p.add_argument("--fractions", type=lambda t: [float(x) for x in _list(t)], default=None)
    p.add_argument("--test-traces", type=_list, default=None)
    p.add_argument("--decisions", type=int, default=16384)
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    p.add_argument("--dry-run", action="store_true", help="Print every command and run nothing")
    p.add_argument("--load-tpch", type=Path, default=REPO_ROOT / "build" / "bench" / "load_tpch")
    p.add_argument("--sim", type=Path, default=REPO_ROOT / "build" / "bench" / "eviction_sim")
    p.add_argument("--samples-bin", type=Path, default=REPO_ROOT / "build" / "bench" / "eviction_samples")
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--traces-dir", type=Path, default=REPO_ROOT / "bench" / "results" / "traces")
    p.add_argument("--sim-dir", type=Path, default=REPO_ROOT / "bench" / "results" / "headroom" / "eviction")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "bench" / "results" / "v3_check")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "bench" / "tpch_data")
    args = p.parse_args(argv)

    if args.sf == "1.0" and not args.cells:
        p.error("--cells is required for SF=1.0 (name the model cells to run, e.g. --cells gbdt_richer)")
        
    d = SF_DEFAULTS[args.sf]
    args.steps = args.steps or list(d["steps"])
    args.cells = args.cells or list(CELLS)
    args.tests = args.test_traces or list(d["tests"])
    bad = [s for s in args.steps if s not in ALL_STEPS] + [c for c in args.cells if c not in CELLS]
    if bad:
        p.error(f"unknown step or cell: {bad}")
        
    args.pools = pools(args.sf, args.fractions or d["fractions"])
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    paths = Paths(args.load_tpch, args.sim, args.samples_bin, args.python, args.traces_dir, args.sim_dir, args.out_dir, args.data_dir)
    driver = Driver(args.dry_run, args.jobs, paths.sf_dir(args.sf) / "logs")
    for step in ALL_STEPS:
        if step not in args.steps:
            continue
        
        started = time.time()
        print(f"== {step}: start {time.strftime('%H:%M:%S')}", flush=True)
        STEP_FUNCS[step](driver, args, paths)
        print(f"== {step}: done in {time.time() - started:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
