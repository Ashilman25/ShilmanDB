#!/usr/bin/env python3
#Run the whole eviction-headroom measurement for one scale factor
from __future__ import annotations

import argparse
import importlib.util
import itertools
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

_rb_spec = importlib.util.spec_from_file_location("run_benchmarks_for_evh", REPO_ROOT / "bench" / "run_benchmarks.py")
assert _rb_spec and _rb_spec.loader
_rb = importlib.util.module_from_spec(_rb_spec)
sys.modules["run_benchmarks_for_evh"] = _rb
_rb_spec.loader.exec_module(_rb)
POOL_SIZES, DB_PAGES, pool_for_fraction = _rb.POOL_SIZES, _rb.DB_PAGES, _rb.pool_for_fraction

# kind -> (workload, indexes, --planner-read-us)
KINDS = {
    "baseline_stream": ("stream", False, None),
    "index_stream": ("stream", True, None),
    "index_lookup": ("lookup", True, None),
    "index_mixed": ("mixed", True, None),
    "index100_stream": ("stream", True, 100),
    "index100_mixed": ("mixed", True, 100),
}
MAIN_KINDS = list(KINDS)[:4]
PLAN_CHECK_KINDS = list(KINDS)[4:]
FRACTIONS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
WINDOW_CHECK = {
    "traces": ["baseline_stream_s1", "index_mixed_s1"],
    "windows": [8, 512],
    "policies": "lru2_visit,arc_visit,lirs_visit",
}
DEFAULTS = {
    "0.01": {"v2_cap": 1448, "validate_seeds": [1, 2, 3], "validate_fractions": FRACTIONS},
    "0.1": {"v2_cap": 2048, "validate_seeds": [1], "validate_fractions": FRACTIONS},
    "1.0": {"v2_cap": 0, "validate_seeds": [1], "validate_fractions": [0.1, 0.5, 0.9]},
}
ALL_STEPS = ["export", "record", "validate", "fullcheck", "v2check", "simulate", "analyze"]
SMALL_ONLY_STEPS = {"fullcheck", "v2check"}
V2_MODEL = str(REPO_ROOT / "ml" / "eviction_policy" / "models" / "eviction_model_v2.pt")
FULLCHECK_TRACE = "baseline_stream_s1"
V2CHECK_TRACES = ["baseline_stream_s1", "index_lookup_s1"]
V2CHECK_POOLS = [72, 128]
CHECK_POLICIES = "lru,clock,lru2,arc,lirs,belady"


@dataclass(frozen=True)
class TraceSpec:
    sf: str
    kind: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.kind}_s{self.seed}"

    @property
    def workload(self) -> str:
        return KINDS[self.kind][0]

    @property
    def indexes(self) -> bool:
        return KINDS[self.kind][1]

    @property
    def planner_read_us(self) -> Optional[int]:
        return KINDS[self.kind][2]


@dataclass(frozen=True)
class Paths:
    binary: Path
    sim: Path
    traces: Path
    out: Path
    data_dir: Path
    db_file: Path

    def trace_dir(self, sf: str) -> Path:
        return self.traces / f"sf{sf}"

    def sim_dir(self, sf: str) -> Path:
        return self.out / f"sf{sf}"

    @property
    def v2_weights(self) -> Path:
        return self.out / "v2_weights.txt"


def grid_pools(sf: str) -> List[int]:
    return sorted({pool_for_fraction(sf, f) for f in FRACTIONS} | {POOL_SIZES[sf]})


def trace_specs(sf: str, seeds: Sequence[int], kinds: Optional[Sequence[str]]) -> List[TraceSpec]:
    if kinds is None:
        specs = [TraceSpec(sf, k, s) for s in seeds for k in MAIN_KINDS]
        specs += [TraceSpec(sf, k, 1) for k in PLAN_CHECK_KINDS]
        return specs
    return [TraceSpec(sf, k, s) for s in seeds for k in kinds]


def default_steps(sf: str) -> List[str]:
    return [s for s in ALL_STEPS if sf == "0.01" or s not in SMALL_ONLY_STEPS]


def _load_cmd(spec: TraceSpec, pool: int, paths: Paths, results_dir: Path) -> List[str]:
    cmd = [
        str(paths.binary), "--sf", spec.sf, "--db-file", str(paths.db_file),
        "--data-dir", str(paths.data_dir / f"sf{spec.sf}") + "/",
        "--pool-size", str(pool), "--create-indexes", "true" if spec.indexes else "false",
        "--workload", spec.workload, "--seed", str(spec.seed), "--warmup", "1", "--runs", "2",
    ]
    if spec.planner_read_us is not None:
        cmd += ["--planner-read-us", str(spec.planner_read_us)]
    return cmd + ["--results-dir", str(results_dir) + "/"]


def trace_file(spec: TraceSpec, paths: Paths) -> Path:
    return paths.trace_dir(spec.sf) / f"{spec.name}.csv"


def record_cmd(spec: TraceSpec, paths: Paths) -> List[str]:
    results = paths.trace_dir(spec.sf) / spec.name
    cmd = _load_cmd(spec, POOL_SIZES[spec.sf], paths, results)
    return cmd + ["--enable-tracing", str(trace_file(spec, paths)), "--trace-collapse"]


def validate_dir(spec: TraceSpec, pool: int, paths: Paths) -> Path:
    return paths.sim_dir(spec.sf) / "validate" / spec.name / f"pool{pool}"


def validate_cmd(spec: TraceSpec, pool: int, paths: Paths) -> List[str]:
    return _load_cmd(spec, pool, paths, validate_dir(spec, pool, paths))


def v2check_dir(spec: TraceSpec, pool: int, paths: Paths) -> Path:
    return paths.sim_dir(spec.sf) / "v2check" / spec.name / f"pool{pool}"


def v2check_cmd(spec: TraceSpec, pool: int, paths: Paths) -> List[str]:
    cmd = _load_cmd(spec, pool, paths, v2check_dir(spec, pool, paths))
    return cmd + ["--use-learned-eviction", "--eviction-model-path", V2_MODEL]


def sim_cmd(spec: TraceSpec, paths: Paths, v2_cap: int, raw: bool = False, window: Optional[int] = None, policies: Optional[str] = None, jobs: Optional[int] = None) -> List[str]:
    name = spec.name if window is None else f"{spec.name}_w{window}"
    cmd = [
        str(paths.sim), "--trace", str(trace_file(spec, paths)),
        "--items", str(paths.trace_dir(spec.sf) / spec.name / "stream.csv"),
        "--name", name, "--pools", ",".join(map(str, grid_pools(spec.sf))),
        "--out-prefix", str(paths.sim_dir(spec.sf) / name),
    ]
    if raw:
        cmd.append("--raw")
    if window is not None:
        cmd += ["--visit-window", str(window)]
    if policies:
        cmd += ["--policies", policies]
    if v2_cap > 0 and window is None:
        cmd += ["--v2-weights", str(paths.v2_weights), "--v2-max-frames", str(v2_cap)]
    if jobs:
        cmd += ["--jobs", str(jobs)]
    return cmd


def fullcheck_spec(sf: str) -> TraceSpec:
    return TraceSpec(sf, "baseline_stream", 1)


def full_trace_file(sf: str, paths: Paths) -> Path:
    return paths.trace_dir(sf) / f"{FULLCHECK_TRACE}_full.csv"


def full_record_cmd(sf: str, paths: Paths) -> List[str]:
    spec = fullcheck_spec(sf)
    results = paths.sim_dir(sf) / "fullcheck" / f"{FULLCHECK_TRACE}_full"
    cmd = _load_cmd(spec, POOL_SIZES[sf], paths, results)
    return cmd + ["--enable-tracing", str(full_trace_file(sf, paths))]


def full_sim_cmd(sf: str, paths: Paths, jobs: Optional[int]) -> List[str]:
    spec = fullcheck_spec(sf)
    results = paths.sim_dir(sf) / "fullcheck" / f"{FULLCHECK_TRACE}_full"
    cmd = [
        str(paths.sim), "--raw", "--trace", str(full_trace_file(sf, paths)),
        "--items", str(results / "stream.csv"), "--name", spec.name,
        "--pools", ",".join(map(str, grid_pools(sf))), "--policies", CHECK_POLICIES,
        "--out-prefix", str(paths.sim_dir(sf) / "fullcheck" / f"{FULLCHECK_TRACE}_raw"),
    ]
    if jobs:
        cmd += ["--jobs", str(jobs)]
    return cmd


#Lines of a trace with a line dropped when its page repeats the line before's page
def _collapsed_lines(path: Path) -> Iterator[str]:
    with open(path) as f:
        yield next(f, "").rstrip("\n")
        previous: Optional[str] = None
        for line in f:
            line = line.rstrip("\n")
            page = line.split(",")[1] if "," in line else line
            if page != previous:
                yield line
            previous = page


#'identical', or the first line where collapsing `full` differs from `collapsed`
def compare_collapsed(full: Path, collapsed: Path) -> str:
    with open(collapsed) as f:
        want = (line.rstrip("\n") for line in f)
        sentinel = object()
        for number, (a, b) in enumerate(itertools.zip_longest(_collapsed_lines(full), want, fillvalue=sentinel), 1):
            if a != b:
                show = lambda x: "<end of file>" if x is sentinel else x
                return f"differs at line {number}\ncollapsed full trace: {show(a)}\nrecorded trace: {show(b)}"
    return "identical"


class Driver:
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run

    def run(self, cmd: List[str], log: Optional[Path] = None, rm: Optional[Path] = None) -> None:
        print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
        if log:
            print(f"  (output to {log})")
        if self.dry_run:
            return
        if rm:
            rm.unlink(missing_ok=True)
        if log:
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "w") as f:
                code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        else:
            code = subprocess.run(cmd).returncode
        if code != 0:
            print(f"command failed with exit code {code}", file=sys.stderr)
            raise SystemExit(code)

    def mkdir(self, path: Path) -> None:
        if not self.dry_run:
            path.mkdir(parents=True, exist_ok=True)


def _log_for(results_dir: Path) -> Path:
    return results_dir.parent / f"{results_dir.name}.log"


def step_export(d: Driver, args, paths: Paths) -> None:
    d.mkdir(paths.out)
    d.run([sys.executable, str(REPO_ROOT / "bench" / "headroom" / "export_v2_weights.py"), "--model", V2_MODEL, "--out", str(paths.v2_weights)])


def step_record(d: Driver, args, paths: Paths) -> None:
    for spec in args.specs:
        d.mkdir(paths.trace_dir(args.sf))
        d.run(record_cmd(spec, paths), log=paths.trace_dir(args.sf) / f"{spec.name}.log", rm=paths.db_file)


def step_validate(d: Driver, args, paths: Paths) -> None:
    for spec in args.specs:
        if spec.kind not in MAIN_KINDS or spec.seed not in args.validate_seeds:
            continue
        for fraction in args.validate_fractions:
            pool = pool_for_fraction(args.sf, fraction)
            d.run(validate_cmd(spec, pool, paths), log=_log_for(validate_dir(spec, pool, paths)), rm=paths.db_file)


def step_fullcheck(d: Driver, args, paths: Paths) -> None:
    sf = args.sf
    full = full_trace_file(sf, paths)
    
    d.mkdir(paths.trace_dir(sf))
    d.mkdir(paths.sim_dir(sf) / "fullcheck")
    d.run(full_record_cmd(sf, paths), log=paths.trace_dir(sf) / f"{FULLCHECK_TRACE}_full.log", rm=paths.db_file)
    
    collapsed = paths.trace_dir(sf) / f"{FULLCHECK_TRACE}.csv"
    verdict_file = paths.sim_dir(sf) / "fullcheck" / "collapse_check.txt"
    print(f"compare {full} (collapsed on the fly) with {collapsed} -> {verdict_file}")
    verdict = "identical"
    
    if not d.dry_run:
        verdict = compare_collapsed(full, collapsed)
        verdict_file.write_text(verdict + "\n")
        print(verdict)
        
    d.run(full_sim_cmd(sf, paths, args.jobs))
    if args.keep_full_trace:
        print(f"keeping {full}")
    elif verdict != "identical":
        # Keep the evidence; the analyzer reports the difference and exits 1.
        print(f"keeping {full}: the collapse check failed")
    else:
        print(f"delete {full}")
        if not d.dry_run:
            full.unlink(missing_ok=True)


def step_v2check(d: Driver, args, paths: Paths) -> None:
    for name in V2CHECK_TRACES:
        spec = next((s for s in trace_specs(args.sf, [1], MAIN_KINDS) if s.name == name))
        for pool in V2CHECK_POOLS:
            d.run(v2check_cmd(spec, pool, paths), log=_log_for(v2check_dir(spec, pool, paths)), rm=paths.db_file)


def step_simulate(d: Driver, args, paths: Paths) -> None:
    d.mkdir(paths.sim_dir(args.sf))
    for spec in args.specs:
        if not d.dry_run and not trace_file(spec, paths).exists():
            raise SystemExit(f"missing trace {trace_file(spec, paths)}: run the record step first")
        d.run(sim_cmd(spec, paths, args.v2_max_frames, jobs=args.jobs))
        
    if args.sf != "0.01":
        return
    for name in WINDOW_CHECK["traces"]:
        spec = next((s for s in args.specs if s.name == name), None)
        if spec is None:
            continue
        for window in WINDOW_CHECK["windows"]:
            d.run(sim_cmd(spec, paths, 0, window=window, policies=WINDOW_CHECK["policies"], jobs=args.jobs))


def step_analyze(d: Driver, args, paths: Paths) -> None:
    d.run([
        sys.executable, str(REPO_ROOT / "bench" / "headroom" / "analyze_eviction_headroom.py"),
        "--sim-dir", str(paths.out), "--traces-dir", str(paths.traces)
    ])


STEP_FUNCS = {
    "export": step_export, "record": step_record, "validate": step_validate, "fullcheck": step_fullcheck,
    "v2check": step_v2check, "simulate": step_simulate, "analyze": step_analyze,
}


def _int_list(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x]


def _float_list(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x]


def _str_list(text: str) -> List[str]:
    return [x for x in text.split(",") if x]


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sf", required=True, choices=["0.01", "0.1", "1.0"])
    p.add_argument("--seeds", type=_int_list, default=[1, 2, 3], help="Trace seeds (default 1,2,3)")
    p.add_argument("--kinds", type=_str_list, default=None, help=f"Trace kinds from {','.join(KINDS)} (default: the four main kinds)")
    p.add_argument("--steps", type=_str_list, default=None, help=f"Steps from {','.join(ALL_STEPS)} (default: all; fullcheck and v2check only at SF=0.01)")
    p.add_argument("--binary", type=Path, default=REPO_ROOT / "build" / "bench" / "load_tpch")
    p.add_argument("--sim", type=Path, default=REPO_ROOT / "build" / "bench" / "eviction_sim")
    p.add_argument("--traces-dir", type=Path, default=REPO_ROOT / "bench" / "results" / "traces")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "bench" / "results" / "headroom" / "eviction")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "bench" / "tpch_data")
    p.add_argument("--db-file", type=Path, default=None, help="Default /tmp/evh_sf<sf>.db")
    p.add_argument("--v2-max-frames", type=int, default=None, help="V2 cap for the simulation (0 = no V2)")
    p.add_argument("--validate-seeds", type=_int_list, default=None)
    p.add_argument("--validate-fractions", type=_float_list, default=None)
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument("--keep-full-trace", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print every command and run nothing")
    
    args = p.parse_args(argv)
    d = DEFAULTS[args.sf]
    steps = args.steps or default_steps(args.sf)
    unknown = [s for s in steps if s not in ALL_STEPS]
    bad_kinds = [k for k in (args.kinds or []) if k not in KINDS]
    if unknown or bad_kinds:
        p.error(f"unknown step or kind: {unknown + bad_kinds}")
        
    args.steps = steps
    args.db_file = args.db_file or Path(f"/tmp/evh_sf{args.sf}.db")
    args.v2_max_frames = d["v2_cap"] if args.v2_max_frames is None else args.v2_max_frames
    args.validate_seeds = args.validate_seeds or d["validate_seeds"]
    args.validate_fractions = args.validate_fractions or d["validate_fractions"]
    args.specs = trace_specs(args.sf, args.seeds, args.kinds)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    paths = Paths(args.binary, args.sim, args.traces_dir, args.out_dir, args.data_dir, args.db_file)
    driver = Driver(args.dry_run)
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
