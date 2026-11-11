#!/usr/bin/env python3
#Compare the engine's eviction runs (run_engine_eviction.py) with the simulated cells item by item

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The driver's names and grid, and the headroom analyzer's helpers, so all three agree
_driver = _load("run_engine_eviction_for_analyzer", REPO_ROOT / "bench" / "eviction" / "engine" / "run_engine_eviction.py")
_evh = _load("analyze_eviction_headroom_for_engine", REPO_ROOT / "bench" / "eviction" / "headroom" / "analyze_eviction_headroom.py")
POLICIES, RUN_STEPS, KINDS, RUN_INFO_NAME = _driver.POLICIES, _driver.RUN_STEPS, list(_driver.KINDS), _driver.RUN_INFO_NAME
PENALTY_US, DB_PAGES, pool_for_fraction, read_kv = _driver.PENALTY_US, _driver.DB_PAGES, _driver.pool_for_fraction, _driver.read_kv
AnalyzerError, _read, parse_name, pool_label = _evh.AnalyzerError, _evh._read, _evh.parse_name, _evh.pool_label
md_table, fnum, ROOM, CLOSE = _evh.md_table, _evh.fnum, _evh.ROOM, _evh.CLOSE
DECISION_KINDS, DECISION_FRACTIONS = _evh.DECISION_KINDS, _evh.DECISION_FRACTIONS

# Engine policy -> the simulator's policy that decides the same way
SIM_POLICY = {"lru": "lru", "clock": "clock", "lru2": "lru2_visit", "arc": "arc_visit", "lirs": "lirs_visit", "v2": "v2"}

# Kinds whose plans change when reads cost 100 µs, and the traces recorded with those plans
INDEX100 = {
    "index_stream": "index100_stream", 
    "index_mixed": "index100_mixed"
}
GATED_STEPS = ("cached", "penalty", "timing")  # also the order of preference when a cell has several runs
PLAN100_KINDS = tuple(INDEX100.values())
DASH = "—"

VALIDATION_COLUMNS = ["sf", "step", "trace", "policy", "sim_policy", "pool", "pass", "seq", "engine_misses", "sim_misses", "match", "gated"]
EVICTION_COLUMNS = [
    "sf", "kind", "pool_label", "pool", "pool_fraction", "policy", "penalty_us", "seeds", "misses", "vs_lru", "vs_belady",
    "pass_ms", "pass_ms_vs_lru", "policy_ns_per_call", "policy_ns_per_fetch", "policy_share",
]
RUN_INFO_KEYS = ("pool_size", "eviction", "miss_penalty_us", "planner_read_us", "cold_start")


def sim_trace(trace: str, planner_read_us: float) -> str:
    kind, _, _, seed, _ = parse_name(trace)
    if kind in INDEX100 and abs(planner_read_us - PENALTY_US) < 1e-6:
        return f"{INDEX100[kind]}_s{seed}"
    return trace


@dataclass
class EngineRun:
    sf: str
    step: str
    trace: str
    policy_dir: str
    folder: Path
    pool: int
    eviction: str
    penalty_us: int
    planner_read_us: float
    jobs: int
    binary_sha256: str              # from driver_info.csv, empty if it has none
    stream: pd.DataFrame            # timed items: pass, seq, misses
    totals: pd.DataFrame            # timed passes: pass, latency_ms, hits, misses, read_ms
    policy: Optional[pd.DataFrame]  # policy_time.csv, from timing runs

    @property
    def kind(self) -> str:
        return parse_name(self.trace)[0]

    @property
    def seed(self) -> int:
        return parse_name(self.trace)[3]

    @property
    def sim_trace(self) -> str:
        return sim_trace(self.trace, self.planner_read_us)

    @property
    def gated(self) -> bool:
        return self.step in GATED_STEPS and self.eviction in POLICIES

    @property
    def link_name(self) -> str:
        return _driver.check_link_name(self.folder)


#  loading

def load_run(folder: Path, sf: str, step: str) -> Tuple[EngineRun, List[str]]:
    trace, policy_dir = folder.parent.parent.name, folder.parent.name
    try:
        parse_name(trace)
    except ValueError:
        raise AnalyzerError(f"{trace} is not a trace name")
    
    if not (folder / "driver_info.csv").is_file():
        raise AnalyzerError("unfinished: no driver_info.csv (the driver writes it only after a run succeeds)")
    
    driver = read_kv(folder / "driver_info.csv")
    info = read_kv(folder / "run_info.csv")
    missing = [k for k in RUN_INFO_KEYS if k not in info] + ([] if "jobs" in driver else ["jobs (driver_info.csv)"])
    if missing:
        raise AnalyzerError(f"unfinished: no {', '.join(missing)}")

    stream = _read(folder / "stream.csv", ["pass", "warmup", "seq", "misses"])
    totals = _read(folder / "stream_totals.csv", ["pass", "warmup", "latency_ms", "hits", "misses", "read_ms"])
    timed = totals[totals["warmup"] == 0].sort_values("pass")
    if timed.empty:
        raise AnalyzerError("stream_totals.csv has no timed pass")
    
    policy_file = folder / "policy_time.csv"
    policy = _read(policy_file, ["policy_calls", "policy_ms"]) if policy_file.is_file() else None

    run = EngineRun(
        sf=sf, step=step, trace=trace, policy_dir=policy_dir, folder=folder, pool=int(info["pool_size"]),
        eviction=info["eviction"], penalty_us=int(float(info["miss_penalty_us"])), planner_read_us=float(info["planner_read_us"]),
        jobs=int(driver["jobs"]), binary_sha256=driver.get("binary_sha256", ""),
        stream=stream[stream["warmup"] == 0][["pass", "seq", "misses"]].reset_index(drop=True),
        totals=timed[["pass", "latency_ms", "hits", "misses", "read_ms"]].reset_index(drop=True), policy=policy,
    )

    odd = []
    if folder.name != f"pool{run.pool}":
        odd.append(f"run_info.csv says pool_size={run.pool}")
    if RUN_INFO_NAME.get(policy_dir) != run.eviction:
        odd.append(f"run_info.csv says eviction={run.eviction} in a {policy_dir} folder")
        
    cold = "false" if step == "warm" else "true"
    if info["cold_start"] != cold:
        odd.append(f"run_info.csv says cold_start={info['cold_start']}, expected {cold} for {step}")
        
    penalty = PENALTY_US if step == "penalty" else 0
    if run.penalty_us != penalty:
        odd.append(f"run_info.csv says miss_penalty_us={run.penalty_us}, expected {penalty} for {step}")
    if step == "timing" and policy is None:
        odd.append("a timing run without policy_time.csv")
    return run, [f"{folder}: {o}" for o in odd]


def load_runs(engine_dir: Path) -> Tuple[List[EngineRun], List[str]]:
    runs, problems = [], []
    for sf, d in _evh._sf_dirs(engine_dir):
        for step in RUN_STEPS:
            for folder in sorted((d / step).glob("*/*/pool*")):
                if not folder.is_dir():
                    continue
                
                try:
                    run, odd = load_run(folder, sf, step)
                except AnalyzerError as e:
                    problems.append(f"{folder}: {e}")
                    continue
                runs.append(run)
                problems += odd
    return runs, problems


#Reads the simulator's CSVs one file at a time, when first needed, and keeps what it read
class Sim:
    def __init__(self, root: Path):
        self.root = root
        self._items: Dict[Tuple[str, str], Optional[dict]] = {}
        self._passes: Dict[Tuple[str, str], Optional[pd.DataFrame]] = {}

    def _path(self, sf: str, trace: str, what: str) -> Path:
        return self.root / f"sf{sf}" / f"{trace}_{what}.csv"

    #{(policy, pool): timed items (pass, seq, misses)}, or None without the file
    def items(self, sf: str, trace: str) -> Optional[dict]:
        key = (sf, trace)
        if key not in self._items:
            path = self._path(sf, trace, "items")
            if not path.is_file():
                self._items[key] = None
            else:
                df = pd.read_csv(path, usecols=["policy", "pool", "pass", "seq", "misses"])
                df = df[df["policy"].isin(list(SIM_POLICY.values()))]
                self._items[key] = {(str(p), int(q)): g[["pass", "seq", "misses"]].reset_index(drop=True) for (p, q), g in df.groupby(["policy", "pool"])}
        return self._items[key]

    def cell(self, sf: str, trace: str, policy: str, pool: int) -> Optional[pd.DataFrame]:
        cells = self.items(sf, trace)
        return None if cells is None else cells.get((policy, pool))

    #Misses of the first timed pass, NaN if the cell isn't there
    def first_pass(self, sf: str, trace: str, policy: str, pool: int) -> float:
        key = (sf, trace)
        if key not in self._passes:
            path = self._path(sf, trace, "passes")
            self._passes[key] = pd.read_csv(path, usecols=["policy", "pool", "pass", "warmup", "misses"]) if path.is_file() else None
            
        df = self._passes[key]
        if df is None:
            return float("nan")
        sub = df[(df["policy"] == policy) & (df["pool"] == pool) & (df["warmup"] == 0)].sort_values("pass")
        return float(sub["misses"].iloc[0]) if len(sub) else float("nan")


#  validation

def validate_run(r: EngineRun, sim: Sim) -> pd.DataFrame:
    eng = r.stream.rename(columns={"misses": "engine_misses"})
    sim_policy = SIM_POLICY[r.eviction]
    cell = sim.cell(r.sf, r.sim_trace, sim_policy, r.pool)
    if cell is None:
        out = eng.assign(sim_misses=np.nan)
    else:
        cell = cell[cell["pass"].isin(eng["pass"].unique())].rename(columns={"misses": "sim_misses"})
        out = eng.merge(cell, on=["pass", "seq"], how="outer")
        
    out["engine_misses"] = out["engine_misses"].astype("Int64")
    out["sim_misses"] = out["sim_misses"].astype("Float64").round().astype("Int64")
    out["match"] = (out["engine_misses"] == out["sim_misses"]).fillna(False).astype(int)
    out = out.assign(sf=r.sf, step=r.step, trace=r.trace, policy=r.eviction, sim_policy=sim_policy, pool=r.pool, gated=int(r.gated), has_cell=cell is not None, sim_trace=r.sim_trace)
    return out.sort_values(["pass", "seq"])


def validate(runs: List[EngineRun], sim: Sim) -> pd.DataFrame:
    frames = [validate_run(r, sim) for r in runs if r.eviction in SIM_POLICY]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=VALIDATION_COLUMNS + ["has_cell", "sim_trace"])
    return pd.concat(frames, ignore_index=True)


#  per-run numbers

def run_numbers(r: EngineRun) -> dict:
    t = r.totals
    return {
        "first_misses": float(t["misses"].iloc[0]),
        "median_misses": float(t["misses"].median()),
        "pass_ms": float(t["latency_ms"].median()),
        "cpu_ms": float((t["latency_ms"] - t["read_ms"]).median()),
        "fetches": float((t["hits"] + t["misses"]).median()),
    }


#Net policy time per call, per fetch and as a share of a normal timed pass.
#The timed pass's own time includes the timer's cost, so it is never the denominator.
def policy_cost(r: EngineRun) -> dict:
    p, nums = r.policy, run_numbers(r)
    calls, net = float(p["policy_calls"].sum()), float(p["policy_ms"].sum())
    return {
        "calls": calls, "net_ms": net,
        "ns_per_call": net * 1e6 / calls if calls else np.nan,
        "ns_per_fetch": net * 1e6 / nums["fetches"] if nums["fetches"] else np.nan,
        "share": _ratio(net, nums["pass_ms"]),
    }


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values and all(np.isfinite(values)) else np.nan


def _ratio(a: float, b: float) -> float:
    return a / b if np.isfinite(a) and np.isfinite(b) and b > 0 else np.nan


#One row per (sf, kind, pool, policy, penalty): seed means of misses and pass times, policy cost, the ratios
def summarize(runs: List[EngineRun], sim: Sim) -> Tuple[pd.DataFrame, List[str]]:
    order = {s: i for i, s in enumerate(GATED_STEPS)}
    cells: Dict[tuple, Dict[int, List[EngineRun]]] = {}
    for r in sorted((r for r in runs if r.gated), key=lambda r: order[r.step]):
        cells.setdefault((r.sf, r.kind, r.pool, r.eviction, r.penalty_us), {}).setdefault(r.seed, []).append(r)

    rows = []
    for (sf, kind, pool, policy, penalty), by_seed in cells.items():
        seeds = sorted(by_seed)
        firsts = [by_seed[s][0] for s in seeds]
        sequential = [next((x for x in by_seed[s] if x.jobs == 1), None) for s in seeds]
        timing = [x for s in seeds for x in by_seed[s] if x.step == "timing" and x.jobs == 1 and x.policy is not None]
        costs = [policy_cost(x) for x in timing]
        rows.append({
            "sf": sf, "kind": kind, "pool": pool, "policy": policy, "penalty_us": penalty,
            "seeds": len(seeds), "seed_set": tuple(seeds),
            "misses": _mean([run_numbers(x)["first_misses"] for x in firsts]),
            "belady": _mean([sim.first_pass(sf, x.sim_trace, "belady", pool) for x in firsts]),
            "pass_ms": _mean([run_numbers(x)["pass_ms"] for x in sequential]) if all(sequential) else np.nan,
            "policy_ns_per_call": _mean([c["ns_per_call"] for c in costs]),
            "policy_ns_per_fetch": _mean([c["ns_per_fetch"] for c in costs]),
            "policy_share": _mean([c["share"] for c in costs]),
        })
    if not rows:
        return pd.DataFrame(columns=EVICTION_COLUMNS + ["belady", "seed_set"]), []

    df = pd.DataFrame(rows)
    lru = {(r["sf"], r["kind"], r["pool"], r["penalty_us"]): r for r in rows if r["policy"] == "lru"}
    warns = []
    vs_lru, time_vs_lru = [], []
    for r in rows:
        base = lru.get((r["sf"], r["kind"], r["pool"], r["penalty_us"]))
        vs_lru.append(_ratio(r["misses"], base["misses"]) if base else np.nan)
        time_vs_lru.append(_ratio(r["pass_ms"], base["pass_ms"]) if base else np.nan)
        if base and base["seed_set"] != r["seed_set"]:
            warns.append(f"SF={r['sf']}, {r['kind']}, pool {r['pool']}, {r['penalty_us']} µs: {r['policy']} has seeds "
                         f"{','.join(map(str, r['seed_set']))} but LRU has {','.join(map(str, base['seed_set']))} (ratios are over different seeds)")
    df["vs_lru"] = vs_lru
    df["pass_ms_vs_lru"] = time_vs_lru
    df["vs_belady"] = [_ratio(m, b) for m, b in zip(df["misses"], df["belady"])]
    df["pool_label"] = [pool_label(sf, p) for sf, p in zip(df["sf"], df["pool"])]
    df["pool_fraction"] = [p / DB_PAGES[sf] for sf, p in zip(df["sf"], df["pool"])]
    df["_k"] = df["kind"].map(lambda k: KINDS.index(k) if k in KINDS else len(KINDS))
    df["_p"] = df["policy"].map(POLICIES.index)
    df["_sf"] = df["sf"].astype(float)
    df = df.sort_values(["_sf", "_k", "penalty_us", "pool", "_p"]).drop(columns=["_k", "_p", "_sf"]).reset_index(drop=True)
    return df, warns


#How much two runs of the same cell differ, per SF
def noise_bands(runs: List[EngineRun]) -> Dict[str, dict]:
    timing = {(r.sf, r.trace, r.eviction, r.pool): r for r in runs if r.step == "timing" and r.gated and r.jobs == 1}
    diffs: Dict[str, List[float]] = {}
    
    for r in runs:
        twin = timing.get((r.sf, r.trace, r.eviction, r.pool)) if r.step == "cached" and r.gated and r.jobs == 1 else None
        if twin is None:
            continue
        
        cached_ms, twin_ms = run_numbers(r)["pass_ms"], run_numbers(twin)["pass_ms"]
        if np.isfinite(cached_ms) and np.isfinite(twin_ms) and cached_ms > 0:
            diffs.setdefault(r.sf, []).append(abs(cached_ms - twin_ms) / cached_ms)
            
    return {
        sf: {
            "cells": len(d), 
            "median": float(np.median(d)), 
            "max": float(np.max(d))
        } for sf, d in diffs.items()
    }


#A warning per SF whose gated runs come from more than one load_tpch build
def mixed_builds(runs: List[EngineRun]) -> List[str]:
    warns = []
    for sf in sorted({r.sf for r in runs}, key=float):
        counts: Dict[str, int] = {}
        for r in runs:
            if r.sf == sf and r.gated and r.binary_sha256:
                counts[r.binary_sha256] = counts.get(r.binary_sha256, 0) + 1
                
        if len(counts) > 1:
            shown = ", ".join(f"{h[:12]}: {n} runs" for h, n in sorted(counts.items(), key=lambda kv: -kv[1]))
            warns.append(f"SF={sf}: the gated runs come from {len(counts)} different load_tpch builds (binary_sha256 {shown}), so misses and times may mix builds. Running the driver again with one binary reruns every run made with another.")
    return warns


#  SQLite check

def load_correctness(engine_dir: Path, runs: List[EngineRun]) -> Dict[str, dict]:
    out = {}
    for sf in sorted({r.sf for r in runs} | {sf for sf, _ in _evh._sf_dirs(engine_dir)}, key=float):
        path = engine_dir / f"sf{sf}" / "stream_correctness.csv"
        folders = {r.link_name for r in runs if r.sf == sf}
        if not path.is_file():
            out[sf] = {"checked": False, "folders": len(folders)}
            continue
        
        df = pd.read_csv(path, usecols=["config", "status"])
        out[sf] = {
            "checked": True, "answers": len(df), "fail": int((df["status"] != "PASS").sum()),
            "configs": df["config"].nunique(), "folders": len(folders), "unchecked": len(folders - set(df["config"].astype(str))),
        }
    return out


#  report

def _pct(a: float, b: float) -> str:
    r = _ratio(a, b)
    return DASH if np.isnan(r) else f"{(r - 1) * 100:+.1f}%"


def _label(sf: str, pool: int) -> str:
    lab = pool_label(sf, pool)
    return lab if lab == str(pool) else f"{lab} ({pool})"


def _where(penalty: int) -> str:
    return "cached reads" if penalty == 0 else f"{penalty} µs per read"


def _cells(df: pd.DataFrame, sf: str):
    sub = df[df["sf"] == sf]
    for kind in KINDS:
        for penalty in sorted(sub[sub["kind"] == kind]["penalty_us"].unique()):
            yield kind, int(penalty), sub[(sub["kind"] == kind) & (sub["penalty_us"] == penalty)]


#The policy with the fewest misses in one (sf, kind, pool, penalty) group; ties go to the earlier policy
def _best(group: pd.DataFrame) -> Optional[pd.Series]:
    if group.empty:
        return None
    g = group.assign(_p=group["policy"].map(POLICIES.index)).sort_values(["misses", "_p"])
    return g.iloc[0]


def validation_section(val: pd.DataFrame, runs: List[EngineRun], sim: Sim) -> str:
    s = ["## Does the engine match the simulator?\n",
         "Every timed item of every engine run against the simulator's misses for the same trace, policy and pool "
         "(`lru2` = `lru2_visit`, `arc` = `arc_visit`, `lirs` = `lirs_visit`, `v2` = `v2`). Gated rows (cached, penalty and timing "
         "runs of the five classical policies) must all be equal; warm and learned-v2 runs are only reported. *measured*\n"]
    
    gated = val[val["gated"] == 1]
    if len(gated):
        equal = int(gated["match"].sum())
        n_runs = len({r.folder for r in runs if r.gated})
        verdict = "all equal" if equal == len(gated) else f"{len(gated) - equal} differ"
        s.append(f"**Gated: {equal:,} of {len(gated):,} items equal over {n_runs} runs ({verdict}).**\n")
        
    body = []
    if len(val):
        keys = ["sf", "step", "policy", "sim_policy"]
        step_order = {st: i for i, st in enumerate(RUN_STEPS)}
        for k, g in sorted(val.groupby(keys), key=lambda kv: (float(kv[0][0]), step_order.get(kv[0][1], 9), kv[0][2])):
            n_runs = len(g.groupby(["trace", "pool"]))
            no_cell = int((~g["has_cell"]).sum())
            differ = int(((g["match"] == 0) & g["has_cell"]).sum())
            body.append([k[0], k[1], k[2], k[3], n_runs, f"{len(g):,}", f"{int(g['match'].sum()):,}", f"{differ:,}", f"{no_cell:,}", "yes" if g["gated"].iloc[0] else "no"])
            
        s.append(md_table(["SF", "step", "policy", "sim policy", "runs", "items", "equal", "differ", "no sim cell", "gated"], body))
        mapped = [r for r in runs if r.sim_trace != r.trace]
        if mapped:
            s.append(f"{len(mapped)} run(s) at {PENALTY_US} µs per read on `index_stream` / `index_mixed` price their plans at {PENALTY_US} µs, so they are compared with the `index100_*` cells; `baseline_stream` and `index_lookup` keep their own.\n")
    else:
        s.append("No engine items to compare.\n")

    # V2
    s.append("### V2 (`learned-v2`, not gated)\n")
    s.append("V2 breaks score ties by hash-set order in the engine and by slot order in the simulator, so a difference is reported, not failed.\n")
    s.append(_side_table(val, runs, sim, lambda r: r.eviction == "v2", "No learned-v2 runs.", with_policy=False))

    # warm start
    s.append("### Warm start (`--keep-load-pool`, not gated)\n")
    s.append("The same policies starting from the pool the load left instead of an empty one, against the cold simulation: how much the warm start changes. *measured*\n")
    s.append(_side_table(val, runs, sim, lambda r: r.step == "warm", "No warm runs.", with_policy=True))

    v1 = [r for r in runs if r.eviction == "v1"]
    if v1:
        s.append(f"`learned-v1`: {len(v1)} run(s), not compared (the simulator has no V1). It is a smoke check: it ran, and its answers are in the SQLite check.\n")
    return "\n".join(s)


def _side_table(val: pd.DataFrame, runs: List[EngineRun], sim: Sim, pick, empty: str, with_policy: bool) -> str:
    chosen = [r for r in runs if pick(r)]
    if not chosen:
        return empty + "\n"
    
    body = []
    for r in sorted(chosen, key=lambda r: (float(r.sf), r.trace, POLICIES.index(r.eviction) if r.eviction in POLICIES else 9, r.pool)):
        g = val[(val["sf"] == r.sf) & (val["step"] == r.step) & (val["trace"] == r.trace) & (val["policy"] == r.eviction) & (val["pool"] == r.pool)]
        eng = run_numbers(r)["first_misses"]
        cold = sim.first_pass(r.sf, r.sim_trace, SIM_POLICY[r.eviction], r.pool)
        row = [r.sf, r.trace] + ([r.eviction] if with_policy else []) + [_label(r.sf, r.pool), f"{len(g):,}", f"{int((g['match'] == 0).sum()):,}", fnum(eng), fnum(cold), _pct(eng, cold)]
        body.append(row)
        
    heads = ["SF", "trace"] + (["policy"] if with_policy else []) + ["pool", "items", "items that differ", "engine misses (first timed pass)", "sim misses (cold)", "difference"]
    return md_table(heads, body)


def correctness_section(checks: Dict[str, dict]) -> str:
    s = ["## Answers against SQLite\n", "`check_stream_results.py` over every run folder, from `sf<sf>/stream_correctness.csv`. *measured*\n"]
    if not checks:
        s.append("No scale factors.\n")
        
    for sf, c in checks.items():
        if not c["checked"]:
            s.append(f"- SF={sf}: no check yet (run the driver's check step).")
            continue
        verdict = "all PASS" if c["fail"] == 0 else f"{c['fail']:,} of {c['answers']:,} answers FAIL"
        line = f"- SF={sf}: {c['answers']:,} answers from {c['configs']} run folders, {verdict}."
        if c["unchecked"]:
            line += f" {c['unchecked']} of {c['folders']} run folders were not in the check."
        s.append(line)
    return "\n".join(s) + "\n"


def misses_section(df: pd.DataFrame, sf: str) -> str:
    s = [f"## SF={sf}: misses per timed pass\n",
         "Seed means of the first timed pass (*measured*). best = the engine policy with the fewest misses; "
         "Belady (the floor no policy can beat) is from the simulator for the same trace and pool.\n"]
    
    for kind, penalty, sub in _cells(df, sf):
        seeds = sorted(set(sub["seeds"]))
        s.append(f"### {kind}, {_where(penalty)} (seeds per cell: {', '.join(map(str, seeds))})\n")
        body = []
        
        for pool in sorted(sub["pool"].unique()):
            g = sub[sub["pool"] == pool].set_index("policy")
            best = _best(sub[sub["pool"] == pool].reset_index(drop=True))
            lru = g.loc["lru", "misses"] if "lru" in g.index else np.nan
            belady = float(best["belady"])
            body.append([_label(sf, pool)] + [fnum(g.loc[p, "misses"]) if p in g.index else DASH for p in POLICIES] + [best["policy"], fnum(_ratio(best["misses"], lru), 3), fnum(belady), fnum(_ratio(best["misses"], belady), 3)])
            
        s.append(md_table(["pool"] + POLICIES + ["best", "best ÷ LRU", "Belady", "best ÷ Belady"], body))
    return "\n".join(s)


def _band_line(sf: str, band: Optional[dict]) -> str:
    if band is None:
        return f"- SF={sf}: no noise band (no twin runs)."
    return f"- SF={sf}: {band['cells']} cells, median {band['median'] * 100:.1f}%, max {band['max'] * 100:.1f}%."


def times_section(df: pd.DataFrame, sfs: List[str], bands: Dict[str, dict]) -> str:
    s = ["## Pass times\n",
         "Median timed pass in ms, from runs made one at a time (*measured*); in brackets, the change against LRU (negative = faster). "
         "Runs made in parallel (`--jobs` above 1) have no time.\n",
         "**Noise band:** how much two runs of the same cell differ (*measured*). A `timing` run is its `cached` run again plus "
         "`--policy-timing`, whose extra pass comes after the normal timed passes and is not in `stream_totals.csv`, so its normal "
         "passes are a second cached run. Per cell with both made one at a time: the gap between the two pass times ÷ the cached one. "
         "A cached-read change against LRU no bigger than the SF's max is marked *within noise*. The twins read from the file cache, "
         "so the 100 µs tables are not marked.\n",
         "\n".join(_band_line(sf, bands.get(sf)) for sf in sfs) + "\n"]
    
    any_time = False
    for sf in sfs:
        for kind, penalty, sub in _cells(df, sf):
            if sub["pass_ms"].isna().all():
                continue
            
            any_time = True
            s.append(f"### SF={sf}, {kind}, {_where(penalty)}\n")
            body = []
            band = bands.get(sf) if penalty == 0 else None
            
            for pool in sorted(sub["pool"].unique()):
                g = sub[sub["pool"] == pool].set_index("policy")
                lru = g.loc["lru", "pass_ms"] if "lru" in g.index else np.nan
                cells = []
                for p in POLICIES:
                    if p not in g.index or np.isnan(g.loc[p, "pass_ms"]):
                        cells.append(DASH)
                    elif p == "lru":
                        cells.append(fnum(g.loc[p, "pass_ms"], 1))
                    else:
                        ms = g.loc[p, "pass_ms"]
                        change = _ratio(ms, lru)
                        noise = ", within noise" if band and np.isfinite(change) and abs(change - 1) <= band["max"] else ""
                        cells.append(f"{fnum(ms, 1)} ({_pct(ms, lru)}{noise})")
                        
                body.append([_label(sf, pool)] + cells)
            s.append(md_table(["pool"] + [f"{p} (ms)" for p in POLICIES], body))
    if not any_time:
        s.append("No times (no run was made one at a time).\n")
    return "\n".join(s)


def hundred_section(runs: List[EngineRun]) -> str:
    s = [f"## At {PENALTY_US} µs per read: measured against cpu + misses x {PENALTY_US} µs\n",
         f"cpu = the cached run's median timed pass minus its read time, for the same trace, plans, policy and pool (*measured*); "
         f"predicted = cpu + median timed misses x {PENALTY_US} µs (*inferred*); measured = the {PENALTY_US} µs run's median timed pass (*measured*).\n"]
    
    twins = {(r.sf, r.trace, r.eviction, r.pool): r for r in runs if r.step == "cached" and r.jobs == 1}
    body = []
    for r in sorted((r for r in runs if r.step == "penalty" and r.gated), key=lambda r: (float(r.sf), KINDS.index(r.kind), r.pool, POLICIES.index(r.eviction), r.seed)):
        nums = run_numbers(r)
        measured = nums["pass_ms"] if r.jobs == 1 else np.nan
        twin = twins.get((r.sf, r.sim_trace, r.eviction, r.pool))
        cpu = run_numbers(twin)["cpu_ms"] if twin else np.nan
        predicted = cpu + nums["median_misses"] * r.penalty_us / 1000.0
        body.append([r.sf, r.kind, _label(r.sf, r.pool), r.eviction, r.seed, fnum(measured, 1), fnum(cpu, 1), fnum(nums["median_misses"]), fnum(predicted, 1), fnum(_ratio(measured, predicted), 3)])
        
    if not body:
        s.append("No penalty runs.\n")
    else:
        s.append(md_table(["SF", "kind", "pool", "policy", "seed", "measured (ms)", "cpu (ms)", "misses", "predicted (ms)", "measured ÷ predicted"], body))
        s.append("A dash in cpu means there is no cached run of that trace, policy and pool made one at a time.\n")
    return "\n".join(s)


def cost_section(runs: List[EngineRun]) -> str:
    s = ["## Each policy's own cost (an upper bound)\n",
         "From the extra pass with every policy call timed (`--policy-timing`), net of the timer's own cost. An upper bound: timing a call "
         "of a few ns between two clock reads overstates it (finding 29). Per fetch = net time ÷ the hits + misses of a normal timed pass; "
         "share = net time ÷ the median normal timed pass (from `stream_totals.csv`; the timed pass itself is slower because of the timer). "
         "Parallel runs are left out. *measured*\n"]
    
    body = []
    for r in sorted((r for r in runs if r.policy is not None and r.jobs == 1 and r.gated), key=lambda r: (float(r.sf), KINDS.index(r.kind), r.pool, POLICIES.index(r.eviction), r.seed)):
        c = policy_cost(r)
        body.append([r.sf, r.kind, _label(r.sf, r.pool), r.eviction, f"{c['calls']:,.0f}", fnum(c["net_ms"], 2), fnum(c["ns_per_call"], 1), fnum(c["ns_per_fetch"], 2), "—" if np.isnan(c["share"]) else f"{c['share'] * 100:.1f}%"])
    if not body:
        s.append("No timing runs.\n")
    else:
        s.append(md_table(["SF", "kind", "pool", "policy", "calls", "net (ms)", "ns per call", "ns per fetch", "share of the pass"], body))
    return "\n".join(s)


def _outcome(ratios: List[float]) -> str:
    if not ratios or any(np.isnan(r) for r in ratios):
        return "unknown (a cell is missing)"
    if all(r >= ROOM for r in ratios):
        return "room"
    if all(r <= CLOSE for r in ratios):
        return "close"
    return "between the rows"


def bar_section(df: pd.DataFrame, sfs: List[str]) -> str:
    s = ["## The new bar for eviction ML\n",
         "For each decision cell (`index_stream` and `index_mixed` at the 10% and 25% pools, seed means), the engine policy with the fewest "
         f"misses and its misses ÷ Belady: the number a learned policy has to beat. room = best ≥ {ROOM:.2f} x Belady in all four cells, "
         f"close = ≤ {CLOSE:.2f} x in all four, else between the rows (the thresholds the simulator's report used). *measured*\n"]
    
    for sf in sfs:
        for label, kinds, penalty in (("cached reads, cached-read plans", DECISION_KINDS, 0), ("cached reads, 100 µs plans (`index100_*`)", PLAN100_KINDS, 0)):
            sub = df[(df["sf"] == sf) & (df["penalty_us"] == penalty)]
            if not sub["kind"].isin(kinds).any():
                continue
            
            body, ratios = [], []
            for kind in kinds:
                for f in DECISION_FRACTIONS:
                    pool = pool_for_fraction(sf, f)
                    best = _best(sub[(sub["kind"] == kind) & (sub["pool"] == pool)])
                    if best is None:
                        body.append([kind, _label(sf, pool), DASH, DASH, DASH, DASH, DASH])
                        ratios.append(np.nan)
                        continue

                    lru = sub[(sub["kind"] == kind) & (sub["pool"] == pool) & (sub["policy"] == "lru")]
                    ratio = _ratio(best["misses"], best["belady"])
                    ratios.append(ratio)
                    body.append([kind, _label(sf, pool), best["policy"], fnum(best["misses"]), fnum(best["belady"]), fnum(ratio, 3), fnum(_ratio(float(lru["misses"].iloc[0]), best["belady"]), 3) if len(lru) else DASH])
            s.append(f"**SF={sf}, {label}: {_outcome(ratios)}.**\n")
            s.append(md_table(["kind", "pool", "best engine policy", "misses", "Belady", "best ÷ Belady", "LRU ÷ Belady"], body))
    if len(s) == 2:
        s.append("No decision cells measured.\n")
    return "\n".join(s)


HOW_TO_READ = (
    "## How to read this\n\n"
    "- Misses are what the engine's buffer pool counted in each timed pass, starting from an empty pool (the driver's runs start cold, "
    "as the simulator does). Fewer is better; Belady is the floor.\n"
    "- The engine matches the simulator when every timed item has the same misses. That is what lets the simulator's numbers stand for the engine.\n"
    "- `default` is the benchmark's default pool; the other pools are shares of the database's table pages.\n"
    "- With cached reads a miss costs about 0.6 µs, so misses barely move the time and the pass time is mostly CPU, the policy's included. "
    f"At {PENALTY_US} µs per read the misses dominate.\n"
    "- Policy cost is an upper bound; a profile gives a better estimate.\n"
)


def build_report(runs: List[EngineRun], val: pd.DataFrame, table: pd.DataFrame, checks: Dict[str, dict], warnings: List[str], sim: Sim) -> str:
    sfs = sorted({r.sf for r in runs}, key=float)
    sections = [validation_section(val, runs, sim), correctness_section(checks)]
    sections += [misses_section(table, sf) for sf in sfs if (table["sf"] == sf).any()]
    sections += [times_section(table, sfs, noise_bands(runs)), hundred_section(runs), cost_section(runs), bar_section(table, sfs), HOW_TO_READ]
    head = ""
    if warnings:
        head = "**WARNING**\n\n" + "\n".join(f"- {w}" for w in warnings) + "\n\n"
    return "# Classical eviction in the engine\n\n" + head + "\n\n".join(sections)


#  CLI

def run(engine_dir: Path, sim_dir: Path) -> int:
    if not engine_dir.is_dir():
        raise AnalyzerError(f"no engine folder {engine_dir}")
    
    sim = Sim(sim_dir)
    runs, problems = load_runs(engine_dir)
    val = validate(runs, sim)
    table, seed_warns = summarize(runs, sim)
    build_warns = mixed_builds(runs)
    checks = load_correctness(engine_dir, runs)

    errors = list(problems)
    if not runs:
        errors.append(f"no engine runs found in {engine_dir}")
        
    if len(val):
        gated = val[val["gated"] == 1]
        differ = int(((gated["match"] == 0) & gated["has_cell"].astype(bool)).sum())
        if differ:
            errors.append(f"validation: {differ:,} of {len(gated):,} checked items differ from the simulator (see engine_validation.csv)")
            
    no_cell = [r for r in runs if r.gated and sim.cell(r.sf, r.sim_trace, SIM_POLICY[r.eviction], r.pool) is None]
    if no_cell:
        shown = "; ".join(f"SF={r.sf} {r.step} {r.trace} {r.eviction} pool {r.pool} (looked in {r.sim_trace}, {SIM_POLICY[r.eviction]})" for r in no_cell[:5])
        errors.append(f"no sim cell for {len(no_cell)} checked run(s): {shown}" + (" …" if len(no_cell) > 5 else ""))
        
    for sf, c in checks.items():
        if c["checked"] and c["fail"]:
            errors.append(f"SF={sf}: {c['fail']:,} of {c['answers']:,} answers FAIL the SQLite check (see sf{sf}/stream_correctness.csv)")

    val[VALIDATION_COLUMNS].to_csv(engine_dir / "engine_validation.csv", index=False)
    table.reindex(columns=EVICTION_COLUMNS).to_csv(engine_dir / "engine_eviction.csv", index=False, float_format="%.10g")
    (engine_dir / "report.md").write_text(build_report(runs, val, table, checks, errors + build_warns + seed_warns, sim))

    print("Wrote:")
    for name in ("engine_validation.csv", "engine_eviction.csv", "report.md"):
        print(f"  {engine_dir / name}")
    for w in build_warns:
        print(f"Warning: {w}", file=sys.stderr)
    for e in errors:
        print(f"Error: {e}", file=sys.stderr)
    return 1 if errors else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare the engine's eviction runs with the simulator and write a report.")
    parser.add_argument("--engine-dir", type=Path, default=REPO_ROOT / "bench" / "results" / "classical_eviction")
    parser.add_argument("--sim-dir", type=Path, default=REPO_ROOT / "bench" / "results" / "headroom" / "eviction")
    args = parser.parse_args(argv)
    try:
        return run(args.engine_dir, args.sim_dir)
    except AnalyzerError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
