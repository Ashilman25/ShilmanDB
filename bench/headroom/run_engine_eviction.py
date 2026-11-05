#!/usr/bin/env python3
#Run the engine's eviction policies over the eviction-headroom grid, one load_tpch run per cell.
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

# Trace kinds, pools and the recording flags come from the eviction-headroom driver, so both always agree.
_evh_spec = importlib.util.spec_from_file_location("run_eviction_headroom_for_engine", REPO_ROOT / "bench" / "headroom" / "run_eviction_headroom.py")
assert _evh_spec and _evh_spec.loader
_evh = importlib.util.module_from_spec(_evh_spec)
sys.modules["run_eviction_headroom_for_engine"] = _evh
_evh_spec.loader.exec_module(_evh)
KINDS, MAIN_KINDS, PLAN_CHECK_KINDS, FRACTIONS = _evh.KINDS, _evh.MAIN_KINDS, _evh.PLAN_CHECK_KINDS, _evh.FRACTIONS
TraceSpec, grid_pools = _evh.TraceSpec, _evh.grid_pools
POOL_SIZES, DB_PAGES, pool_for_fraction = _evh.POOL_SIZES, _evh.DB_PAGES, _evh.pool_for_fraction

POLICIES = ["lru", "clock", "lru2", "arc", "lirs"]
MODELS = {
    "learned-v2": Path(_evh.V2_MODEL),
    "learned-v1": REPO_ROOT / "ml" / "eviction_policy" / "models" / "eviction_model.pt",
}

# What run_info.csv's eviction key says for each --eviction-policy name
RUN_INFO_NAME = {**{p: p for p in POLICIES}, "learned-v2": "v2", "learned-v1": "v1"}
RUN_STEPS = ["cached", "penalty", "timing", "learned", "warm"]
ALL_STEPS = RUN_STEPS + ["check", "analyze"]
SMALL_ONLY_STEPS = {"learned", "warm"}
PENALTY_US = 100
SF1_CACHED_JOBS = 4
TRI = (0.1, 0.25, 0.5)

# The traces and pools the eviction-headroom driver checked V2 on
LEARNED_KINDS = tuple(name.rsplit("_s", 1)[0] for name in _evh.V2CHECK_TRACES)
LEARNED_POOLS = tuple(_evh.V2CHECK_POOLS)
WARM_KINDS = ("baseline_stream", "index_lookup")
CHECK_LINKS = "check_folders"

# Everything a run leaves in its folder; cleared before the run starts
RESULT_FILES = ("driver_info.csv", "load_tpch.log", "run_info.csv", "stream.csv", "stream_totals.csv", "stream_queries.csv", "stream_results.csv", "policy_time.csv")


@dataclass(frozen=True)
class Group:
    kinds: Tuple[str, ...]
    seeds: Tuple[int, ...]
    pools: Tuple[int, ...]
    policies: Tuple[str, ...]
    runs: int = 2  # timed passes


@dataclass(frozen=True)
class Run:
    step: str
    spec: TraceSpec
    policy: str
    pool: int
    runs: int = 2


@dataclass(frozen=True)
class Paths:
    binary: Path
    out: Path
    sim: Path
    data_dir: Path
    db_dir: Path

    def sf_dir(self, sf: str) -> Path:
        return self.out / f"sf{sf}"

    def db_file(self, sf: str, job: int) -> Path:
        return self.db_dir / f"eve_sf{sf}_{job}.db"


def _pools(sf: str, fractions: Sequence[float]) -> Tuple[int, ...]:
    return tuple(pool_for_fraction(sf, f) for f in fractions)


#The grid each step runs at one SF before any option narrows or replaces it
def default_groups(step: str, sf: str) -> List[Group]:
    main, plan, every = tuple(MAIN_KINDS), tuple(PLAN_CHECK_KINDS), tuple(POLICIES)
    grid, tri, quarter = tuple(grid_pools(sf)), _pools(sf, TRI), _pools(sf, (0.25,))
    
    if step == "cached":
        if sf == "0.01":
            return [Group(main, (1, 2, 3), grid, every), Group(plan, (1,), grid, every)]
        if sf == "0.1":
            return [Group(main, (1,), grid, every), Group(plan, (1,), tri, every)]
        return [Group(main, (1,), tri, every)]
    
    if step == "penalty":
        if sf == "0.01":
            return [Group(main, (1,), tri, every)]
        if sf == "0.1":
            return [Group(main, (1,), quarter, ("lru", "arc", "lirs"), runs=1)]
        # SF=1.0 only when asked for: LIRS where it wins (stream) and ARC where it wins (mixed)
        return [Group(("index_stream",), (1,), quarter, ("lru", "lirs"), runs=1), Group(("index_mixed",), (1,), quarter, ("lru", "arc"), runs=1)]

    if step == "timing":
        if sf == "0.01":
            return [Group(main, (1,), (POOL_SIZES[sf],) + quarter, every)]
        if sf == "0.1":
            return [Group(main, (1,), quarter, every)]
        return [Group(("index_stream",), (1,), quarter, every)]
    
    if step == "learned":
        return [Group(LEARNED_KINDS, (1,), LEARNED_POOLS, ("learned-v2",)), Group(LEARNED_KINDS[:1], (1,), LEARNED_POOLS[:1], ("learned-v1",))]
    
    if step == "warm":
        return [Group(WARM_KINDS, (1,), grid, ("clock", "lru2", "arc", "lirs"))]
    raise ValueError(f"not a run step: {step}")


def default_steps(sf: str) -> List[str]:
    skip = set() if sf == "0.01" else SMALL_ONLY_STEPS
    if sf == "1.0":
        skip = skip | {"penalty"}  # optional at SF=1.0: ask for it with --steps
    return [s for s in ALL_STEPS if s not in skip]


def jobs_for(args: argparse.Namespace, step: str) -> int:
    if args.jobs is not None:
        return args.jobs
    return SF1_CACHED_JOBS if step == "cached" and args.sf == "1.0" else 1


def plan_runs(args: argparse.Namespace, step: str) -> List[Run]:
    runs: Dict[Run, None] = {}
    for g in default_groups(step, args.sf):
        kinds = [k for k in g.kinds if args.kinds is None or k in args.kinds]
        seeds = g.seeds if args.seeds is None else tuple(args.seeds)
        pools = g.pools if args.pools is None else tuple(args.pools)
        policies = g.policies if args.policies is None or step == "learned" else tuple(args.policies)
        
        for kind in kinds:
            for seed in seeds:
                if kind in PLAN_CHECK_KINDS and seed != 1:
                    continue  # only seed 1 was recorded for these
                
                spec = TraceSpec(args.sf, kind, seed)
                for policy in policies:
                    for pool in pools:
                        runs[Run(step, spec, policy, pool, g.runs)] = None
    return list(runs)


def run_folder(run: Run, paths: Paths) -> Path:
    return paths.sf_dir(run.spec.sf) / run.step / run.spec.name / run.policy / f"pool{run.pool}"


#load_tpch with the trace's recording flags, this run's pool and policy, and the step's extras
def run_cmd(run: Run, paths: Paths, job: int) -> List[str]:
    recording = _evh.Paths(paths.binary, Path(), Path(), Path(), paths.data_dir, paths.db_file(run.spec.sf, job))
    cmd = _evh._load_cmd(run.spec, run.pool, recording, run_folder(run, paths))
    cmd[cmd.index("--runs") + 1] = str(run.runs)
    extra = ["--eviction-policy", run.policy]
    
    if run.policy in MODELS:
        extra += ["--eviction-model-path", str(MODELS[run.policy])]
        
    if run.step == "penalty":
        extra += ["--miss-penalty-us", str(PENALTY_US)]
        
    if run.step == "timing":
        extra.append("--policy-timing")
        
    if run.step == "warm":
        extra.append("--keep-load-pool")
        
    at = cmd.index("--results-dir")
    return cmd[:at] + extra + cmd[at:]


def check_link_name(folder: Path) -> str:
    return "-".join(folder.parts[-4:])  # <step>-<trace>-<policy>-pool<P>


def check_cmd(sf: str, paths: Paths, links: Sequence[Path]) -> List[str]:
    return [sys.executable, str(REPO_ROOT / "bench" / "check_stream_results.py"), "--sf", sf, "--data-dir", str(paths.data_dir), "--output", str(paths.sf_dir(sf) / "stream_correctness.csv")] + [str(l) for l in links]


def analyze_cmd(paths: Paths) -> List[str]:
    return [sys.executable, str(REPO_ROOT / "bench" / "headroom" / "analyze_engine_eviction.py"), "--engine-dir", str(paths.out), "--sim-dir", str(paths.sim)]


#Run folders of this SF the driver finished, with answers to check
def finished_folders(paths: Paths, sf: str) -> List[Path]:
    root = paths.sf_dir(sf)
    found = []
    for step in RUN_STEPS:
        for f in sorted((root / step).glob("*/*/pool*")):
            if all((f / name).is_file() for name in ("driver_info.csv", "stream_queries.csv", "stream_results.csv")):
                found.append(f)
    return found


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_kv(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    with open(path, newline="") as f:
        return {r["key"]: r["value"] for r in csv.DictReader(f) if "key" in r and "value" in r}


def write_driver_info(folder: Path, step: str, jobs: int, binary: Path, binary_hash: Optional[str]) -> None:
    with open(folder / "driver_info.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerows([["jobs", jobs], ["step", step], ["binary", binary], ["binary_sha256", binary_hash or ""]])


#True if the folder holds this run's full result from the same binary
def is_complete(folder: Path, run: Run, binary_hash: Optional[str]) -> bool:
    if not binary_hash:
        return False
    
    info = read_kv(folder / "driver_info.csv")
    if info.get("binary_sha256") != binary_hash or info.get("step") != run.step:
        return False
    
    run_info = read_kv(folder / "run_info.csv")
    if run_info.get("pool_size") != str(run.pool) or run_info.get("eviction") != RUN_INFO_NAME[run.policy]:
        return False
    
    needed = ["stream.csv", "stream_queries.csv", "stream_results.csv"] + (["policy_time.csv"] if run.step == "timing" else [])
    if not all((folder / n).is_file() for n in needed) or not (folder / "stream_totals.csv").is_file():
        return False
    
    with open(folder / "stream_totals.csv") as f:
        passes = sum(1 for line in f if line.strip()) - 1
    return passes == 1 + run.runs  # one warmup pass plus the timed ones


def _took(seconds: float) -> str:
    return f"{seconds:.1f} s ({seconds / 60:.1f} min)"


class Driver:
    def __init__(self, paths: Paths, dry_run: bool, redo: bool, binary_hash: Optional[str]):
        self.paths, self.dry_run, self.redo, self.binary_hash = paths, dry_run, redo, binary_hash
        self.lock = threading.Lock()

    def say(self, text: str) -> None:
        with self.lock:
            print(text, flush=True)

    #A command that isn't an engine run; its output goes to the terminal
    def command(self, cmd: List[str]) -> None:
        self.say("$ " + shlex.join(cmd))
        if self.dry_run:
            return
        
        code = subprocess.run(cmd).returncode
        if code != 0:
            print(f"command failed with exit code {code}", file=sys.stderr)
            raise SystemExit(code)

    def one(self, run: Run, job: int, jobs: int, number: int, total: int) -> int:
        folder = run_folder(run, self.paths)
        if not self.redo and is_complete(folder, run, self.binary_hash):
            self.say(f"skip {folder}: already done with this binary ({number}/{total})")
            return 0
        
        cmd = run_cmd(run, self.paths, job)
        log = folder / "load_tpch.log"
        self.say("$ " + shlex.join(cmd) + f"\n  (run {number}/{total}, output to {log})")
        if self.dry_run:
            return 0
        
        db = self.paths.db_file(run.spec.sf, job)
        db.unlink(missing_ok=True)
        folder.mkdir(parents=True, exist_ok=True)
        
        for name in RESULT_FILES:
            (folder / name).unlink(missing_ok=True)
        with open(log, "w") as f:
            code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
            
        db.unlink(missing_ok=True)
        if code == 0:
            write_driver_info(folder, run.step, jobs, self.paths.binary, self.binary_hash)
        else:
            self.say(f"failed with exit code {code}: {folder} (log: {log})")
        return code

    def engine_runs(self, runs: List[Run], jobs: int) -> None:
        total = len(runs)
        if not self.dry_run:
            self.paths.db_dir.mkdir(parents=True, exist_ok=True)
            
        if jobs <= 1 or self.dry_run:
            for number, run in enumerate(runs, 1):
                code = self.one(run, (number - 1) % jobs, jobs, number, total)
                if code:
                    raise SystemExit(code)
            return

        # Each worker holds one database slot while its run lasts
        slots: Queue = Queue()
        for job in range(jobs):
            slots.put(job)
        stop = threading.Event()
        codes: List[int] = []

        def task(number: int, run: Run) -> None:
            if stop.is_set():
                return
            job = slots.get()
            try:
                if stop.is_set():
                    return
                code = self.one(run, job, jobs, number, total)
                if code:
                    codes.append(code)
                    stop.set()  # finish what is running, start nothing new
            finally:
                slots.put(job)

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(task, number, run) for number, run in enumerate(runs, 1)]
        for f in futures:
            f.result()
        if codes:
            raise SystemExit(codes[0])


def step_check(d: Driver, args: argparse.Namespace, paths: Paths, planned: List[Path]) -> None:
    folders = sorted(set(finished_folders(paths, args.sf)) | set(planned))
    if not folders:
        print(f"no finished run folders at SF={args.sf}: nothing to check")
        return
    
    # check_stream_results.py names each run by its folder, and every run folder is called pool<P>
    links_dir = paths.sf_dir(args.sf) / CHECK_LINKS
    links = [links_dir / check_link_name(f) for f in folders]
    print(f"link {len(folders)} run folders into {links_dir}/ as <step>-<trace>-<policy>-pool<P>")
    if not d.dry_run:
        links_dir.mkdir(parents=True, exist_ok=True)
        for old in links_dir.iterdir():
            if old.is_symlink():
                old.unlink()
                
        for link, folder in zip(links, folders):
            link.symlink_to(folder.resolve(), target_is_directory=True)
    d.command(check_cmd(args.sf, paths, links))


def _str_list(text: str) -> List[str]:
    return [x for x in text.split(",") if x]


def _int_list(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x]


#Pools for --fractions: 'default' is the benchmark's default pool, a number in (0, 1] a share of the tables' pages
def parse_fractions(text: str, sf: str) -> List[int]:
    pools = []
    for token in _str_list(text):
        if token == "default":
            pools.append(POOL_SIZES[sf])
            continue
        
        f = float(token)
        if not 0 < f <= 1:
            raise ValueError(f"fraction {token} is not in (0, 1]")
        pools.append(pool_for_fraction(sf, f))
    if not pools:
        raise ValueError("no fractions given")
    return list(dict.fromkeys(pools))


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sf", required=True, choices=["0.01", "0.1", "1.0"])
    p.add_argument("--steps", type=_str_list, default=None, help=f"Steps from {','.join(ALL_STEPS)}")
    p.add_argument("--policies", type=_str_list, default=None, help=f"Policies from {','.join(POLICIES)}")
    p.add_argument("--kinds", type=_str_list, default=None, help=f"Trace kinds from {','.join(KINDS)}")
    p.add_argument("--seeds", type=_int_list, default=None, help="Trace seeds")
    p.add_argument("--fractions", default=None, help="Pools as shares of the tables' pages, e.g. 0.1,0.25 or default")
    p.add_argument("--jobs", type=int, default=None, help=f"Engine runs at once (default 1; {SF1_CACHED_JOBS} for cached at SF=1.0). The analyzer reports no times from runs made in parallel")
    p.add_argument("--binary", type=Path, default=REPO_ROOT / "build" / "bench" / "load_tpch")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "bench" / "results" / "classical_eviction")
    p.add_argument("--sim-dir", type=Path, default=REPO_ROOT / "bench" / "results" / "headroom" / "eviction")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "bench" / "tpch_data")
    p.add_argument("--db-dir", type=Path, default=Path("/tmp"))
    p.add_argument("--redo", action="store_true", help="Run again even where a folder already has a full result")
    p.add_argument("--dry-run", action="store_true", help="Print every command and run nothing")

    args = p.parse_args(argv)
    args.steps = args.steps or default_steps(args.sf)
    bad = [s for s in args.steps if s not in ALL_STEPS]
    bad += [k for k in (args.kinds or []) if k not in KINDS]
    bad += [x for x in (args.policies or []) if x not in POLICIES]
    if bad:
        p.error(f"unknown step, kind or policy: {', '.join(bad)}")
        
    if args.sf != "0.01" and SMALL_ONLY_STEPS & set(args.steps):
        p.error("the learned and warm steps run at SF=0.01 only")
        
    for name in ("kinds", "policies", "seeds"):
        if getattr(args, name) == []:
            p.error(f"--{name} is empty")
            
    if args.jobs is not None and args.jobs < 1:
        p.error("--jobs must be at least 1")
    args.pools = None
    if args.fractions is not None:
        try:
            args.pools = parse_fractions(args.fractions, args.sf)
        except ValueError as e:
            p.error(f"--fractions: {e}")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    paths = Paths(args.binary, args.out_dir, args.sim_dir, args.data_dir, args.db_dir)
    binary_hash = file_sha256(args.binary) if args.binary.is_file() else None
    if binary_hash is None and not args.dry_run and set(args.steps) & set(RUN_STEPS):
        print(f"Error: no load_tpch binary at {args.binary}", file=sys.stderr)
        return 1

    driver = Driver(paths, args.dry_run, args.redo, binary_hash)
    began = time.time()
    planned: List[Path] = []
    for step in ALL_STEPS:
        if step not in args.steps:
            continue
        
        started = time.time()
        if step in RUN_STEPS:
            runs, jobs = plan_runs(args, step), jobs_for(args, step)
            print(f"== {step}: start {time.strftime('%H:%M:%S')} ({len(runs)} runs, {jobs} at a time)", flush=True)
            driver.engine_runs(runs, jobs)
            planned += [run_folder(r, paths) for r in runs]
        else:
            print(f"== {step}: start {time.strftime('%H:%M:%S')}", flush=True)
            if step == "check":
                step_check(driver, args, paths, planned)
            else:
                driver.command(analyze_cmd(paths))
                
        print(f"== {step}: done in {_took(time.time() - started)}", flush=True)
    print(f"== total: {_took(time.time() - began)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        code = main()
        sys.stdout.flush()
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        code = 1
    sys.exit(code)
