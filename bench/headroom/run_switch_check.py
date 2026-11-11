#!/usr/bin/env python3
#Runs every step of the policy-switching check for one scale factor
from __future__ import annotations

import argparse
import csv
import importlib.util
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_an = _load("analyze_switch_check_for_driver", HERE / "analyze_switch_check.py")
read_chosen, ChosenError, pool_for_fraction = _an.read_chosen, _an.ChosenError, _an.pool_for_fraction

ALL_STEPS = ("layout", "validate", "replay", "analyze")
PHASES = {K: (f"index_stream_s{K}", f"index_lookup_s{K}", f"index_mixed_s{K}", f"index_stream_s{K % 3 + 1}") for K in (1, 2, 3)}
WINDOW_KS = (64, 32, 16, 8, 4)
FRACTIONS = (0.10, 0.25, 0.50)
ETAS = (5, 10, 20)
ALPHAS = (0.01, 0.03, 0.1)
MARGIN = 2.0
THRESHOLD = 0.05
START = "lirs_visit"
REPLAY_OUTPUTS = ("results", "phases", "occupancy", "switches")


@dataclass(frozen=True)
class Paths:
    tool: Path
    python: Path
    traces: Path
    sim_dir: Path
    out: Path

    def sf_dir(self, sf: str) -> Path:
        return self.out / f"sf{sf}"

    def trace_dir(self, sf: str) -> Path:
        return self.traces / f"sf{sf}"

    def prefix(self, sf: str, k: int) -> Path:
        return self.sf_dir(sf) / phases_name(k)

    def windows(self, sf: str) -> Path:
        return self.sf_dir(sf) / "windows.csv"

    def chosen(self) -> Path:
        return self.out / "chosen.json"


def pools(sf: str) -> List[int]:
    return [pool_for_fraction(sf, f) for f in FRACTIONS]


def phases_name(k: int) -> str:
    return f"phases_s{k}"


# A window is a multiple of 4 that is about 1/k of a phase.
def window_for(L: float, k: int) -> int:
    return 4 * round(L / (4 * k))


# Mean run count of the 12 phases in the three layout files.
def mean_phase_length(layout_csvs: Sequence[Path]) -> float:
    if len(layout_csvs) != 3:
        raise ValueError(f"need 3 layout files, got {len(layout_csvs)}")

    runs: List[int] = []
    for path in layout_csvs:
        try:
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
        except OSError as e:
            raise ValueError(f"cannot read layout file {path}: {e}") from e
        
        if len(rows) != 4:
            raise ValueError(f"{path} has {len(rows)} phases, expected 4")
        runs += [int(r["runs"]) for r in rows]
    return sum(runs) / len(runs)


def write_windows(path: Path, L: float) -> Dict[int, int]:
    windows = {k: window_for(L, k) for k in WINDOW_KS}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_k", "window", "mean_phase_runs"])
        for k, size in windows.items():
            w.writerow([k, size, f"{L:g}"])
    return windows


def hedge_settings(sf: str, chosen: Optional[dict]) -> List[Tuple[float, float]]:
    if sf == "0.01":
        return [(e, 0) for e in ETAS] + [(e, a) for e in ETAS for a in ALPHAS]
    assert chosen is not None
    return [(chosen["hedge"]["eta"], 0), (chosen["fixed_share"]["eta"], chosen["fixed_share"]["alpha"])]


def _base_cmd(sf: str, paths: Paths, k: int) -> List[str]:
    return [
        str(paths.tool), 
        "--traces-dir", str(paths.trace_dir(sf)), 
        "--components", ",".join(PHASES[k]), 
        "--name", phases_name(k), 
        "--out-prefix", str(paths.prefix(sf, k))
    ]


def describe_cmd(sf: str, paths: Paths, k: int) -> List[str]:
    return _base_cmd(sf, paths, k) + ["--describe"]


def validate_cmd(sf: str, paths: Paths, k: int, jobs: Optional[int] = None) -> List[str]:
    cmd = _base_cmd(sf, paths, k) + ["--pools", ",".join(map(str, pools(sf))), "--validate"]
    return cmd + (["--jobs", str(jobs)] if jobs else [])


def replay_cmd(sf: str, paths: Paths, k: int, windows, hedge, jobs: Optional[int]) -> List[str]:
    if isinstance(windows, str):
        window_text = windows
    else:
        window_text = ",".join(f"{wk}:{windows[wk]}" for wk in WINDOW_KS)
        
    hedge_text = ",".join(f"{format(e, 'g')}:{format(a, 'g')}" for e, a in hedge)
    cmd = _base_cmd(sf, paths, k) + [
        "--pools", ",".join(map(str, pools(sf))), "--windows", window_text, "--hedge", hedge_text,
        "--margin", format(MARGIN, "g"), "--threshold", format(THRESHOLD, "g"), "--start", START,
    ]
    return cmd + (["--jobs", str(jobs)] if jobs else [])


def analyze_cmd(sf: str, paths: Paths) -> List[str]:
    return [str(paths.python), str(HERE / "analyze_switch_check.py"), "--sf", sf, "--out", str(paths.out), "--sim-dir", str(paths.sim_dir)]


class Driver:
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run

    def run(self, cmd: List[str]) -> None:
        print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
        if self.dry_run:
            return

        code = subprocess.run(cmd).returncode
        if code != 0:
            print(f"command failed with exit code {code}", file=sys.stderr)
            raise SystemExit(code)


def step_layout(d: Driver, args, paths: Paths) -> None:
    if not d.dry_run:
        paths.sf_dir(args.sf).mkdir(parents=True, exist_ok=True)
    for k in PHASES:
        d.run(describe_cmd(args.sf, paths, k))
    if d.dry_run:
        return

    try:
        L = mean_phase_length([Path(f"{paths.prefix(args.sf, k)}_layout.csv") for k in PHASES])
    except ValueError as e:
        raise SystemExit(str(e))
    windows = write_windows(paths.windows(args.sf), L)
    print(f"mean phase length L = {L:g} runs")
    for k, w in windows.items():
        print(f"  window share {k}: W = {w}")


def step_validate(d: Driver, args, paths: Paths) -> None:
    if not d.dry_run:
        paths.sf_dir(args.sf).mkdir(parents=True, exist_ok=True)
    for k in PHASES:
        d.run(validate_cmd(args.sf, paths, k, args.jobs))


def step_replay(d: Driver, args, paths: Paths) -> None:
    chosen = None
    if args.sf == "0.1":
        chosen = read_chosen(paths.chosen())

    win_path = paths.windows(args.sf)
    if win_path.exists():
        with open(win_path, newline="") as f:
            windows = {int(r["window_k"]): int(r["window"]) for r in csv.DictReader(f)}
    elif d.dry_run:
        windows = "<from windows.csv>"
    else:
        raise SystemExit(f"{win_path} is missing: run the layout step first")

    hedge = hedge_settings(args.sf, chosen)
    if not d.dry_run:
        paths.sf_dir(args.sf).mkdir(parents=True, exist_ok=True)
        # A replay that stops part way must not leave an older run's files to be summed with the new ones.
        for k in PHASES:
            for kind in REPLAY_OUTPUTS:
                Path(f"{paths.prefix(args.sf, k)}_{kind}.csv").unlink(missing_ok=True)
    for k in PHASES:
        d.run(replay_cmd(args.sf, paths, k, windows, hedge, args.jobs))


def step_analyze(d: Driver, args, paths: Paths) -> None:
    d.run(analyze_cmd(args.sf, paths))


STEP_FUNCS = {
    "layout": step_layout, 
    "validate": step_validate, 
    "replay": step_replay, 
    "analyze": step_analyze
}


def _list(text: str) -> List[str]:
    return [x for x in text.split(",") if x]


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sf", required=True, choices=("0.01", "0.1"))
    p.add_argument("--steps", type=_list, default=list(ALL_STEPS), help=f"From {','.join(ALL_STEPS)} (default: all)")
    p.add_argument("--jobs", type=int, default=None, help="Threads per tool call (default: every core)")
    p.add_argument("--dry-run", action="store_true", help="Print every command and run nothing")
    p.add_argument("--tool", type=Path, default=REPO_ROOT / "build" / "bench" / "eviction_switch")
    p.add_argument("--python", type=Path, default=REPO_ROOT / ".venv" / "bin" / "python")
    p.add_argument("--traces-dir", type=Path, default=REPO_ROOT / "bench" / "results" / "traces")
    p.add_argument("--sim-dir", type=Path, default=REPO_ROOT / "bench" / "results" / "headroom" / "eviction")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "bench" / "results" / "switch_check")
    args = p.parse_args(argv)

    bad = [s for s in args.steps if s not in ALL_STEPS]
    if bad:
        p.error(f"unknown step: {bad}")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    paths = Paths(args.tool, args.python, args.traces_dir, args.sim_dir, args.out_dir)

    needs_tool = any(s in args.steps for s in ("layout", "validate", "replay"))
    if needs_tool and not args.dry_run and not args.tool.exists():
        print(f"{args.tool} not found: build it with cmake --build build --target eviction_switch", file=sys.stderr)
        return 1

    if args.sf == "0.1" and "replay" in args.steps:
        try:
            read_chosen(paths.chosen())
        except ChosenError as e:
            print(e, file=sys.stderr)
            return 1

    driver = Driver(args.dry_run)
    try:
        for step in ALL_STEPS:
            if step not in args.steps:
                continue

            started = time.time()
            print(f"== {step}: start {time.strftime('%H:%M:%S')}", flush=True)
            STEP_FUNCS[step](driver, args, paths)
            print(f"== {step}: done in {time.time() - started:.1f} s", flush=True)
    except ChosenError as e:
        print(e, file=sys.stderr)
        return 1
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
            return 1
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
