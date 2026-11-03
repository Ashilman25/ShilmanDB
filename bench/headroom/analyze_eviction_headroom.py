#!/usr/bin/env python3
# Turns eviction_sim's CSVs into validation.csv, eviction_headroom.csv, report.md and one chart per scale factor.

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

matplotlib.use("Agg")  # writes files only, never opens a window

REPO_ROOT = Path(__file__).resolve().parents[2]

# Paths, trace kinds and pool fractions come from the driver, so both always agree.
_spec = importlib.util.spec_from_file_location("run_eviction_headroom_for_analyzer", REPO_ROOT / "bench" / "headroom" / "run_eviction_headroom.py")
assert _spec and _spec.loader
_driver = importlib.util.module_from_spec(_spec)
sys.modules["run_eviction_headroom_for_analyzer"] = _driver
_spec.loader.exec_module(_driver)
FRACTIONS, DB_PAGES, POOL_SIZES, pool_for_fraction = _driver.FRACTIONS, _driver.DB_PAGES, _driver.POOL_SIZES, _driver.pool_for_fraction
MAIN_KINDS, PLAN_CHECK_KINDS, WINDOW_CHECK = _driver.MAIN_KINDS, _driver.PLAN_CHECK_KINDS, _driver.WINDOW_CHECK

# Draft thresholds for the decision: best classical / Belady
ROOM = 1.10
CLOSE = 1.05
DECISION_KINDS = ("index_stream", "index_mixed")
DECISION_FRACTIONS = (0.10, 0.25)
CLASSICAL = ("lru", "clock", "lru2", "arc", "lirs", "lru2_visit", "arc_visit", "lirs_visit")
MODES = {
    "lru2": ("lru2", "lru2_visit", "lru2_full"),
    "arc": ("arc", "arc_visit", "arc_full"),
    "lirs": ("lirs", "lirs_visit", "lirs_full"),
}
MISS_US = 100.0
HINT_TOLERANCE = 1.02  # "within 2% of the best"
HINT_BETTER = 0.98     # and at least 2% fewer misses than LRU
SEEDED = CLASSICAL + ("belady",)  # the policies whose seed sets must agree
PLAN_FRACTIONS = (0.10, 0.25, 0.50)
PLAN_MOVE = 0.02       # a plan-check cell moving more than this is noted
V2_GAP = 5.0           # engine V2 vs simulated V2, percent

CONFIGS = {
    "baseline": "baseline", 
    "index": "+index", 
    "index100": "+index (100 µs plans)"
}
NAME_RE = re.compile(r"^(baseline|index100|index)_(stream|lookup|mixed)_s(\d+)(?:_w(\d+))?$")
DASH = "—"

CSV_COLUMNS = [
    "sf", "workload", "config", "pool_label", "pool", "pool_fraction", "policy", "seeds", "misses",
    "misses_min", "misses_max", "vs_lru", "vs_belady", "best_classical", "miss_s_100us", "est_pass_s_100us",
]
VALIDATION_COLUMNS = ["sf", "trace", "pool", "source", "pass", "seq", "engine_misses", "sim_misses", "match"]
PASSES_COLUMNS = ["trace", "policy", "pool", "pass", "warmup", "misses"]
ITEMS_COLUMNS = ["trace", "policy", "pool", "pass", "seq", "misses"]
STREAM_COLUMNS = ["pass", "warmup", "seq", "misses"]
TOTALS_COLUMNS = ["pass", "warmup", "latency_ms", "misses", "read_ms"]

# Chart style: same surface, grid and ink as analyze_join_headroom.py
SURFACE, GRID, MUTED, INK = "#fcfcfb", "#e4e3df", "#52514e", "#0b0b0b"
FAMILY_COLORS = {
    "lru": "#2a78d6", 
    "lirs": "#eb6834", 
    "arc": "#1baf7a", 
    "lru2": "#eda100", 
    "clock": "#e87ba4", 
    "v2": "#008300"
}
FAMILY_LABELS = {
    "lru": "LRU", 
    "lirs": "LIRS (better mode)", 
    "arc": "ARC (better mode)", 
    "lru2": "LRU-2 (better mode)",
    "clock": "CLOCK", 
    "v2": "V2 (shipped model)",
}


class AnalyzerError(Exception):
    pass


def parse_name(name: str) -> tuple[str, str, str, int, int | None]:
    m = NAME_RE.match(name)
    if not m:
        raise ValueError(f"not a trace name: {name}")
    
    prefix, workload, seed, window = m.groups()
    return f"{prefix}_{workload}", CONFIGS[prefix], workload, int(seed), None if window is None else int(window)


def pool_label(sf: str, pool: int) -> str:
    for f in FRACTIONS:
        if pool == pool_for_fraction(sf, f):
            return f"{round(f * 100)}%"
    return "default" if pool == POOL_SIZES[sf] else str(pool)


def _read(path: Path, needed: list[str] | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise AnalyzerError(f"missing file: {path}")
    
    df = pd.read_csv(path)
    missing = [c for c in (needed or []) if c not in df.columns]
    if missing:
        raise AnalyzerError(f"{path} is missing column(s): {', '.join(missing)}")
    return df


def _sf_dirs(root: Path) -> list[tuple[str, Path]]:
    if not root.is_dir():
        return []
    return sorted((d.name[2:], d) for d in root.iterdir() if d.is_dir() and d.name.startswith("sf"))


def _sf_key(sf: str) -> float:
    return float(sf)


# loading

#Every sim passes file (with sf, kind, seed, window columns) and the LRU rows of every items file
def load_sim(sim_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    passes, items = [], []
    for sf, d in _sf_dirs(sim_dir):
        for f in sorted(d.glob("*_passes.csv")):
            passes.append(_read(f, PASSES_COLUMNS).assign(sf=sf, source_file=str(f)))
        for f in sorted(d.glob("*_items.csv")):
            df = _read(f, ITEMS_COLUMNS)
            items.append(df[df["policy"] == "lru"].assign(sf=sf))
            
    if not passes:
        raise AnalyzerError(f"no sim files (sf*/*_passes.csv) in {sim_dir}")
    allp = pd.concat(passes, ignore_index=True)
    dup = allp[allp.duplicated(["sf", "trace", "policy", "pool", "pass"], keep=False)]
    
    if len(dup):
        first = dup.iloc[0]
        same = dup[(dup["sf"] == first["sf"]) & (dup["trace"] == first["trace"]) & (dup["policy"] == first["policy"]) & (dup["pool"] == first["pool"]) & (dup["pass"] == first["pass"])]
        files = sorted(set(same["source_file"]))
        raise AnalyzerError(f"duplicate rows for trace {first['trace']}, policy {first['policy']}, pool {first['pool']}, "
                            f"pass {first['pass']} in {' and '.join(files)}")
        
    allp = allp.drop(columns="source_file")
    names = {t: parse_name(t) for t in allp["trace"].unique()}
    allp["kind"] = allp["trace"].map(lambda t: names[t][0])
    allp["seed"] = allp["trace"].map(lambda t: names[t][3])
    allp["window"] = allp["trace"].map(lambda t: names[t][4])
    empty = pd.DataFrame(columns=["trace", "policy", "pool", "pass", "seq", "misses", "sf"])
    return allp, (pd.concat(items, ignore_index=True) if items else empty)


#Misses of the first timed pass for every (sf, trace, policy, pool)
def headlines(passes: pd.DataFrame) -> pd.DataFrame:
    timed = passes[passes["warmup"] == 0].sort_values("pass")
    keys = ["sf", "trace", "policy", "pool"]
    first = timed.groupby(keys, as_index=False).first()
    return first[keys + ["kind", "seed", "window", "pass", "misses"]]


def _run_info_pool(folder: Path) -> int:
    info = _read(folder / "run_info.csv", ["key", "value"])
    row = info[info["key"] == "pool_size"]
    if row.empty:
        raise AnalyzerError(f"{folder / 'run_info.csv'} has no pool_size")
    return int(row["value"].iloc[0])


def _cpu_seconds(totals: pd.DataFrame) -> float:
    timed = totals[totals["warmup"] == 0]
    return float((timed["latency_ms"] - timed["read_ms"]).median() / 1000.0)


def load_engine_run(folder: Path, sf: str, source: str) -> dict:
    stream = _read(folder / "stream.csv", STREAM_COLUMNS)
    pool = _run_info_pool(folder)
    totals = _read(folder / "stream_totals.csv", TOTALS_COLUMNS) if source != "validate" else None
    return {
        "sf": sf, 
        "trace": folder.parent.name if source != "recording" else folder.name, 
        "pool": pool,
        "source": source, 
        "stream": stream, 
        "totals": totals, 
        "folder": folder
    }


def load_engine_runs(sim_dir: Path, traces_dir: Path) -> dict[str, list[dict]]:
    out = {"recording": [], "validate": [], "v2check": []}
    for sf, d in _sf_dirs(traces_dir):
        for f in sorted(x for x in d.iterdir() if x.is_dir()):
            out["recording"].append(load_engine_run(f, sf, "recording"))
            
    for sf, d in _sf_dirs(sim_dir):
        for source in ("validate", "v2check"):
            for f in sorted((d / source).glob("*/pool*")):
                if f.is_dir():
                    run = load_engine_run(f, sf, source)
                    run["trace"] = f.parent.name
                    out[source].append(run)
    return out


#  validation 

def validate(sim_items: pd.DataFrame, runs: list[dict]) -> pd.DataFrame:
    sim = sim_items[["sf", "trace", "pool", "pass", "seq", "misses"]].rename(columns={"misses": "sim_misses"})
    frames = []
    for r in runs:
        timed = r["stream"][r["stream"]["warmup"] == 0]
        frames.append(pd.DataFrame({
            "sf": r["sf"], "trace": r["trace"], "pool": r["pool"], "source": r["source"],
            "pass": timed["pass"].to_numpy(), "seq": timed["seq"].to_numpy(), "engine_misses": timed["misses"].to_numpy()
        }))
    if not frames:
        return pd.DataFrame(columns=VALIDATION_COLUMNS)
    
    eng = pd.concat(frames, ignore_index=True)
    sim = sim.astype({"pool": int, "pass": int, "seq": int})
    eng = eng.astype({"pool": int, "pass": int, "seq": int})
    out = eng.merge(sim.drop_duplicates(["sf", "trace", "pool", "pass", "seq"]), on=["sf", "trace", "pool", "pass", "seq"], how="left")
    out["match"] = (out["sim_misses"] == out["engine_misses"]).astype(int)
    out["sim_misses"] = out["sim_misses"].astype("Int64")
    return out[VALIDATION_COLUMNS]


#  aggregation 

def cpu_table(recordings: list[dict]) -> dict[tuple[str, str], float]:
    return {(r["sf"], r["trace"]): _cpu_seconds(r["totals"]) for r in recordings}


#Seed means per (sf, kind, pool, policy), the ratios, best_classical and the 100 µs times
def aggregate(heads: pd.DataFrame, cpu: dict[tuple[str, str], float]) -> pd.DataFrame:
    h = heads.copy()
    h["cpu"] = [cpu.get((sf, t), np.nan) for sf, t in zip(h["sf"], h["trace"])]
    h["miss_s"] = h["misses"] * MISS_US * 1e-6
    h["est_s"] = h["cpu"] + h["miss_s"]
    keys = ["sf", "kind", "pool", "policy"]
    
    # est_s keeps NaN for a seed with no recording, so the mean is NaN rather than a mean over fewer seeds
    g = h.groupby(keys).agg(
        seeds=("misses", "size"), misses=("misses", "mean"), misses_min=("misses", "min"),
        misses_max=("misses", "max"), miss_s_100us=("miss_s", "mean"),
        est_pass_s_100us=("est_s", lambda x: x.mean(skipna=False)),
        seed_set=("seed", lambda x: tuple(sorted(x)))
    ).reset_index()
    
    group = ["sf", "kind", "pool"]
    for name, col in (("belady", "vs_belady"), ("lru", "vs_lru")):
        base = g[g["policy"] == name].set_index(group)["misses"]
        denom = pd.Series([base.get(tuple(k), np.nan) for k in g[group].itertuples(index=False)], index=g.index)
        g[col] = (g["misses"] / denom.where(denom > 0)).astype(float)
        
    best = {}
    for k, sub in g[g["policy"].isin(CLASSICAL)].groupby(group):
        by = sub.set_index("policy")["misses"]
        best[k] = min((p for p in CLASSICAL if p in by.index), key=lambda p: (by[p], CLASSICAL.index(p)))
        
    g["best_classical"] = [best.get(tuple(k), "") for k in g[group].itertuples(index=False)]
    g["workload"] = g["kind"].str.split("_").str[-1]
    g["config"] = g["kind"].map(lambda k: CONFIGS[k.split("_")[0]])
    g["pool_label"] = [pool_label(sf, p) for sf, p in zip(g["sf"], g["pool"])]
    g["pool_fraction"] = [p / DB_PAGES[sf] for sf, p in zip(g["sf"], g["pool"])]
    return g


def lookup(agg: pd.DataFrame, sf: str, kind: str, pool: int, policy: str) -> float:
    row = agg[(agg["sf"] == sf) & (agg["kind"] == kind) & (agg["pool"] == pool) & (agg["policy"] == policy)]
    return float(row["misses"].iloc[0]) if len(row) else float("nan")


def best_over_belady(agg: pd.DataFrame, sf: str, kind: str, pool: int) -> tuple[float, str]:
    sub = agg[(agg["sf"] == sf) & (agg["kind"] == kind) & (agg["pool"] == pool) & (agg["policy"] == "belady")]
    best = agg[(agg["sf"] == sf) & (agg["kind"] == kind) & (agg["pool"] == pool)]["best_classical"]
    if sub.empty or best.empty or not best.iloc[0] or sub["misses"].iloc[0] <= 0:
        return float("nan"), ""
    name = best.iloc[0]
    return lookup(agg, sf, kind, pool, name) / float(sub["misses"].iloc[0]), name


#<family>_full ÷ the better of <family> and <family>_visit
def full_penalty(agg: pd.DataFrame, sf: str, kind: str, pool: int, family: str) -> float:
    plain, visit, full = (lookup(agg, sf, kind, pool, p) for p in MODES[family])
    best = np.nanmin([plain, visit]) if not (np.isnan(plain) and np.isnan(visit)) else float("nan")
    return full / best if best > 0 else float("nan")


# decision 

#Why a cell can't be trusted: classical policies or Belady missing, or policies averaged over different seeds
def cell_problems(agg: pd.DataFrame, sf: str, kind: str, pool: int) -> list[str]:
    sub = agg[(agg["sf"] == sf) & (agg["kind"] == kind) & (agg["pool"] == pool)]
    if sub.empty:
        return ["no data"]
    
    have = set(sub["policy"])
    out = []
    missing = [p for p in CLASSICAL + ("belady",) if p not in have]
    if missing:
        out.append("missing " + ", ".join(missing))
        
    seeded = sub[sub["policy"].isin(SEEDED)]
    sets = {tuple(r.seed_set) for r in seeded.itertuples()}
    if len(sets) > 1:
        detail = "; ".join(f"{p}: seeds {','.join(map(str, t))}" for p, t in sorted((r.policy, tuple(r.seed_set)) for r in seeded.itertuples()))
        out.append("seed sets differ (" + detail + ")")
    return out


def seed_mismatch_warnings(agg: pd.DataFrame) -> list[str]:
    out = []
    for (sf, kind, pool), sub in agg.groupby(["sf", "kind", "pool"]):
        if len({tuple(t) for t in sub[sub["policy"].isin(SEEDED)]["seed_set"]}) > 1:
            out.append(f"seed sets differ between policies at SF={sf}, {kind}, pool {pool} (means are over different seeds)")
    return out


#(kind, pool label, best ÷ Belady, best policy, problem). A cell with a problem has a NaN ratio
def decision_cells(agg: pd.DataFrame, sf: str, kinds: tuple[str, ...] = DECISION_KINDS) -> list[tuple[str, str, float, str, str]]:
    out = []
    for kind in kinds:
        for f in DECISION_FRACTIONS:
            pool = pool_for_fraction(sf, f)
            problems = cell_problems(agg, sf, kind, pool)
            ratio, name = (float("nan"), "") if problems else best_over_belady(agg, sf, kind, pool)
            if not problems and np.isnan(ratio):
                problems = ["Belady has 0 misses"]
            out.append((kind, f"{round(f * 100)}%", ratio, name, "; ".join(problems)))
    return out


#room, close, between or unknown
def outcome(cells: list[tuple[str, str, float, str, str]]) -> str:
    ratios = [c[2] for c in cells]
    if not ratios or any(np.isnan(r) for r in ratios):
        return "unknown"
    if all(r >= ROOM for r in ratios):
        return "room"
    if all(r <= CLOSE for r in ratios):
        return "close"
    return "between"


def decide(cells: list[tuple[str, str, float, str, str]]) -> str:
    result = outcome(cells)
    if result == "unknown":
        missing = "; ".join(f"{c[0]} {c[1]}: {c[4] or 'no data'}" for c in cells if np.isnan(c[2]))
        return f"Unknown (cells without a trustworthy value: {missing})."
    if result == "room":
        return "Eviction V3 (04) has real room: build it."
    if result == "close":
        return "Classical (ARC/LRU-2) is already within a few % of Belady: make workloads harder first."
    return "Between the rows: " + "; ".join(f"{k} {p} {r:.3f}" for k, p, r, _, _ in cells) + "."


CACHED_KINDS = ("index_stream", "index_mixed")
PLAN100_KINDS = ("index100_stream", "index100_mixed")


#Seed-1 decision on the cached-read plans and on the 100 µs plans, or None without index100_* traces
def plan_decisions(heads: pd.DataFrame, sf: str):
    sub = heads[(heads["sf"] == sf) & heads["window"].isna()]
    if not sub["kind"].isin(PLAN_CHECK_KINDS).any():
        return None
    
    seed1 = aggregate(sub[(sub["seed"] == 1) & sub["kind"].isin(CACHED_KINDS + PLAN100_KINDS)], {})
    cached = decision_cells(seed1, sf, CACHED_KINDS)
    plan100 = decision_cells(seed1, sf, PLAN100_KINDS)
    a, b = outcome(cached), outcome(plan100)
    return cached, plan100, a, b

#Problems with validation coverage: main-kind traces with no engine item checked, or nothing checked at all
def unvalidated(heads: pd.DataFrame, val: pd.DataFrame) -> list[str]:
    traces = {(sf, t) for sf, t, k, w in zip(heads["sf"], heads["trace"], heads["kind"], heads["window"]) if k in MAIN_KINDS and pd.isna(w)}
    done = set(zip(val["sf"], val["trace"])) if len(val) else set()
    out = [f"unvalidated: {t} at SF={sf} (no engine run to compare its simulated LRU with)" for sf, t in sorted(traces - done)]
    if len(val) == 0:
        out.append("validation: no engine items were checked at all")
    return out


# fullcheck, v2check 

#Raw replay against the collapsed sim. Returns the table, warnings, and the collapse verdict text
def fullcheck(sim_dir: Path, sf: str, plain: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    d = sim_dir / f"sf{sf}" / "fullcheck"
    warns: list[str] = []
    verdict = ""
    
    if not d.is_dir():
        return pd.DataFrame(), warns, verdict
    if (d / "collapse_check.txt").is_file():
        verdict = (d / "collapse_check.txt").read_text().strip()
        if verdict != "identical":
            warns.append(f"fullcheck SF={sf}: collapse check says: {verdict or '(empty)'}")
    else:
        warns.append(f"fullcheck SF={sf}: collapse_check.txt is missing")
        
    raw_file = d / "baseline_stream_s1_raw_passes.csv"
    if not raw_file.is_file():
        warns.append(f"fullcheck SF={sf}: baseline_stream_s1_raw_passes.csv is missing")
        return pd.DataFrame(), warns, verdict
    raw = headlines(_with_names(_read(raw_file, PASSES_COLUMNS).assign(sf=sf)))
    out = []
    raw_at = raw.set_index(["policy", "pool"])["misses"]
    
    for pool in sorted(raw["pool"].unique()):
        for policy in ("lru", "clock", "belady", "lru2", "arc", "lirs"):
            expect_name = policy if policy in ("lru", "clock", "belady") else f"{policy}_full"
            e = plain[(plain["trace"] == "baseline_stream_s1") & (plain["policy"] == expect_name) & (plain["pool"] == pool)]
            expected = int(e["misses"].iloc[0]) if len(e) else None
            
            if (policy, pool) not in raw_at.index:
                out.append({"policy": policy, "pool": pool, "raw": None, "compared with": expect_name, "expected": expected, "same": False})
                warns.append(f"fullcheck SF={sf}: raw replay has no {policy} at pool {pool}")
                continue
            
            got = int(raw_at[(policy, pool)])
            ok = expected == got
            out.append({
                "policy": policy, "pool": pool, 
                "raw": got, "compared with": expect_name, 
                "expected": expected, "same": ok
            })
            if not ok:
                warns.append(f"fullcheck SF={sf}: raw {policy} at pool {pool} is {got}, expected {expected} ({expect_name})")
    return pd.DataFrame(out), warns, verdict


def _with_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["kind"], df["seed"], df["window"] = "", 0, None
    return df


def v2check(runs: list[dict], plain: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in runs:
        timed = r["totals"][r["totals"]["warmup"] == 0].sort_values("pass")
        if timed.empty:
            raise AnalyzerError(f"{r['folder'] / 'stream_totals.csv'} has no timed rows")
        
        eng = int(timed["misses"].iloc[0])
        s = plain[(plain["sf"] == r["sf"]) & (plain["trace"] == r["trace"]) & (plain["policy"] == "v2") & (plain["pool"] == r["pool"])]
        sim = int(s["misses"].iloc[0]) if len(s) else None
        gap = (sim - eng) / eng * 100.0 if sim is not None and eng else float("nan")
        out.append({"sf": r["sf"], "trace": r["trace"], "pool": r["pool"], "engine V2": eng, "sim V2": sim, "difference %": gap})
    return pd.DataFrame(out)


#  report 

def fnum(x, digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NA:
        return DASH
    return f"{x:,.{digits}f}"


def md_table(headers: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(headers))) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(lines) + "\n"


def _pools(agg: pd.DataFrame, sf: str, kind: str) -> list[int]:
    return sorted(agg[(agg["sf"] == sf) & (agg["kind"] == kind)]["pool"].unique())


def _label(sf: str, pool: int) -> str:
    return pool_label(sf, pool) + (f" ({pool})" if pool_label(sf, pool) != str(pool) else "")


def _seed_count(agg: pd.DataFrame, sf: str, kind: str, pool: int, policy: str) -> int:
    row = agg[(agg["sf"] == sf) & (agg["kind"] == kind) & (agg["pool"] == pool) & (agg["policy"] == policy)]
    return int(row["seeds"].iloc[0]) if len(row) else 0


def kind_table(agg: pd.DataFrame, sf: str, kind: str) -> str:
    cols = ["lru", "clock", "lru2", "arc", "lirs", "lru2_visit", "arc_visit", "lirs_visit", "v2", "belady"]
    body, notes = [], []
    
    for pool in _pools(agg, sf, kind):
        ratio, name = best_over_belady(agg, sf, kind, pool)
        missing = [p for p in CLASSICAL if np.isnan(lookup(agg, sf, kind, pool, p))]
        mark = "*" if missing else ""
        if missing:
            notes.append(f"\\* {_label(sf, pool)}: classical policies missing ({', '.join(missing)}), so the best is only over the others.")
            
        v2n, bn = _seed_count(agg, sf, kind, pool, "v2"), _seed_count(agg, sf, kind, pool, "belady")
        if v2n and bn and v2n != bn:
            notes.append(f"v2 at {_label(sf, pool)} is the mean of {v2n} seed(s); the other columns are means over {bn}.")
            
        body.append([_label(sf, pool)] + [fnum(lookup(agg, sf, kind, pool, p)) for p in cols] + [fnum(ratio, 3) + (mark if not np.isnan(ratio) else ""), (name + mark) if name else DASH])
    return md_table(["pool"] + cols + ["best ÷ Belady", "best"], body) + ("\n" + "\n\n".join(notes) + "\n" if notes else "")


def modes_table(agg: pd.DataFrame, sf: str, kind: str) -> str:
    body = []
    for pool in _pools(agg, sf, kind):
        row = [_label(sf, pool)]
        for fam in MODES:
            row += [fnum(lookup(agg, sf, kind, pool, p)) for p in MODES[fam]] + [fnum(full_penalty(agg, sf, kind, pool, fam), 3)]
        body.append(row)
        
    heads = ["pool"]
    for fam in MODES:
        heads += list(MODES[fam]) + [f"{fam} full ÷ best"]
    return md_table(heads, body)


def time_table(agg: pd.DataFrame, sf: str, kind: str) -> str:
    body = []
    for pool in _pools(agg, sf, kind):
        sub = agg[(agg["sf"] == sf) & (agg["kind"] == kind) & (agg["pool"] == pool)].set_index("policy")
        best = sub["best_classical"].iloc[0] if len(sub) else ""
        def est(p):
            return fnum(sub.loc[p, "est_pass_s_100us"], 1) if p in sub.index else DASH
        
        body.append([_label(sf, pool), est("lru"), f"{est(best) if best else DASH} ({best or DASH})", est("belady"), fnum(sub.loc["lru", "miss_s_100us"], 1) if "lru" in sub.index else DASH])
    return md_table(["pool", "LRU pass (s)", "best classical pass (s)", "Belady pass (s)", "LRU misses alone (s)"], body)


def window_table(passes: pd.DataFrame, heads: pd.DataFrame, sf: str) -> str:
    win = heads[(heads["sf"] == sf) & heads["window"].notna()]
    if win.empty:
        return ""
    
    body = []
    for name in sorted(win["trace"].str.replace(r"_w\d+$", "", regex=True).unique()):
        for policy in WINDOW_CHECK["policies"].split(","):
            for pool in sorted(win[win["trace"].str.startswith(name + "_w")]["pool"].unique()):
                row = [name, policy, str(pool)]
                for w in (8, 64, 512):
                    t = name if w == 64 else f"{name}_w{w}"
                    m = heads[(heads["sf"] == sf) & (heads["trace"] == t) & (heads["policy"] == policy) & (heads["pool"] == pool)]
                    row.append(fnum(m["misses"].iloc[0]) if len(m) else DASH)
                body.append(row)
    return md_table(["trace", "policy", "pool", "W = 8", "W = 64", "W = 512"], body)


def plan_table(heads: pd.DataFrame, agg_seed1: pd.DataFrame, sf: str) -> tuple[str, bool, list[str]]:
    body, moved, missing = [], False, []
    for workload in ("stream", "mixed"):
        a, b = f"index_{workload}", f"index100_{workload}"
        
        for f in PLAN_FRACTIONS:
            pool = pool_for_fraction(sf, f)
            ra, _ = best_over_belady(agg_seed1, sf, a, pool)
            rb, _ = best_over_belady(agg_seed1, sf, b, pool)
            delta = rb - ra
            for label, r, k in (("+index", ra, a), ("100 µs plans", rb, b)):
                if np.isnan(r):
                    missing.append(f"{workload} {round(f * 100)}% ({label}: {k})")
                    
            moved |= (not np.isnan(delta)) and abs(delta) > PLAN_MOVE
            body.append([workload, f"{round(f * 100)}%", fnum(ra, 3), fnum(rb, 3), fnum(delta, 3)])
    return md_table(["workload", "pool", "best ÷ Belady, +index", "best ÷ Belady, 100 µs plans", "change"], body), moved, missing


def _mode_tag(policy: str) -> str:
    if policy.endswith("_visit"):
        return "visit"
    return "plain" if policy in MODES else "any"


HINT_LEGEND = ("Mode tags: any = no special mode; plain = a repeat of the page just used counts once; visit = the buffer pool's visit flag marks a repeat soon after.")


#Ratio table (5-90% pools), the within-2% table (5-75%), and the policies worth building with where they qualify
def hint(agg: pd.DataFrame, sf: str) -> tuple[str, str, dict[str, str]]:
    fraction_of = {pool_for_fraction(sf, f): f for f in FRACTIONS}
    ratios, near_rows, cands = [], [], {}
    
    for kind in MAIN_KINDS:
        for pool in _pools(agg, sf, kind):
            frac = fraction_of.get(pool, pool / DB_PAGES[sf])
            if not (0.05 - 1e-9 <= frac <= 0.90 + 1e-9):
                continue
            
            sub = agg[(agg["sf"] == sf) & (agg["kind"] == kind) & (agg["pool"] == pool) & (agg["policy"].isin(CLASSICAL))].set_index("policy")["misses"]
            if sub.empty:
                continue
            
            best, lru = sub.min(), (sub["lru"] if "lru" in sub.index else float("nan"))
            def rat(p, d):
                return f"{sub[p] / d:.3f}" if p in sub.index and d > 0 else DASH
            
            ratios.append([kind, _label(sf, pool)] + [f"{rat(p, lru)} / {rat(p, best)}" for p in CLASSICAL])
            if frac <= 0.75 + 1e-9:
                near = [p for p in CLASSICAL if p in sub.index and sub[p] <= best * HINT_TOLERANCE]
                # worth it only if it also beats LRU by at least 2% here
                wins = [p for p in near if p != "lru" and lru > 0 and sub[p] / lru <= HINT_BETTER + 1e-9]
                for p in wins:
                    cands.setdefault(p, {}).setdefault(kind, []).append(pool_label(sf, pool))
                tag = lambda ps: ", ".join(f"{p} [{_mode_tag(p)}]" for p in ps) or DASH
                near_rows.append([kind, _label(sf, pool), tag(near), tag(wins)])
                
    if not ratios:
        return "No data.\n", "", {}
    t1 = md_table(["trace kind", "pool"] + [f"{p} (÷ LRU / ÷ best)" for p in CLASSICAL], ratios)
    t2 = md_table(["trace kind", "pool (5-75%)", "within 2% of the best", "and 2% or more below LRU"], near_rows)
    
    lines = {
        p: "; ".join(f"{k} {', '.join(pools)}" for k, pools in by_kind.items())
        for p, by_kind in sorted(cands.items(), key=lambda kv: CLASSICAL.index(kv[0]))
    }
    return t1, t2, lines


def build_report(sfs: list[str], agg: pd.DataFrame, heads: pd.DataFrame, passes: pd.DataFrame, val: pd.DataFrame, sim_dir: Path, engine: dict[str, list[dict]]) -> str:
    plain = heads[heads["window"].isna()]
    warns: list[str] = []
    bad = val[val["match"] == 0]
    if len(bad):
        warns.append(f"validation: {len(bad)} of {len(val)} engine items do not match the simulated LRU (see validation.csv)")
    warns += unvalidated(heads, val)
    warns += seed_mismatch_warnings(agg)
    plan_dec = {sf: plan_decisions(heads, sf) for sf in sfs}
    
    for sf, pd_ in plan_dec.items():
        if pd_ and "unknown" not in pd_[2:] and pd_[2] != pd_[3]:
            warns.append(f"SF={sf}: the plan check CHANGES the decision (seed 1: cached-read plans say {pd_[2]}, 100 µs plans say {pd_[3]}); see the plan check and the draft decision")
            
    have_cpu = {(r["sf"], r["trace"]) for r in engine["recording"]}
    for sf, trace in sorted({(a, b) for a, b, k, w in zip(heads["sf"], heads["trace"], heads["kind"], heads["window"]) if k in MAIN_KINDS and pd.isna(w)}):
        if (sf, trace) not in have_cpu:
            warns.append(f"no recording folder for {trace} at SF={sf}: its 100 µs pass times are empty")
    sections = []

    # Is the simulator right?
    s = ["## Is the simulator right?\n"]
    s.append(f"Simulated LRU against the engine's own misses, item by item: {len(val) - len(bad)} of {len(val)} timed items match "
             f"({(val['source'] == 'recording').sum()} from recording runs, {(val['source'] == 'validate').sum()} from validation runs at other pools). *measured*\n")
    
    lru = passes[(passes["policy"] == "lru") & (passes["warmup"] == 0)]
    if len(lru):
        per = lru.groupby(["sf", "trace", "pool"])["misses"].nunique()
        s.append(f"LRU's timed passes give equal misses in {int((per == 1).sum())} of {len(per)} (trace, pool) cells. *measured*\n")
        
    for sf in sfs:
        table, fwarns, verdict = fullcheck(sim_dir, sf, plain[plain["sf"] == sf])
        warns += fwarns
        if verdict or not table.empty:
            s.append(f"**Uncollapsed check, SF={sf}.** Collapse check: {verdict or DASH}. Raw replay of `baseline_stream_s1` against the collapsed sim "
                     "(LRU, CLOCK, Belady unchanged; LRU-2, ARC, LIRS against their every-fetch versions):\n")
            if not table.empty:
                s.append(md_table(list(table.columns), [[str(v) if not isinstance(v, (bool, np.bool_)) else ("yes" if v else "NO") for v in r] for r in table.values.tolist()]))
                
        v2 = v2check(engine["v2check"], plain)
        v2 = v2[v2["sf"] == sf] if len(v2) else v2
        if len(v2):
            s.append(f"**Simulated V2 against the engine's V2, SF={sf}** (*measured*):\n")
            s.append(md_table(
                ["trace", "pool", "engine V2", "sim V2", "difference %"], 
                [[r["trace"], r["pool"], fnum(r["engine V2"]), "no sim cell" if pd.isna(r["sim V2"]) else fnum(r["sim V2"]), fnum(r["difference %"], 1)] for _, r in v2.iterrows()]
            ))
            if v2["sim V2"].isna().any():
                s.append("Some rows have no simulated V2 cell (V2 only runs at pools up to its cap), so they are not compared.\n")
            if (v2["difference %"].abs() > V2_GAP).any():
                s.append("A gap over 5% is expected, not a failure. Three known causes, all from the replay starting cold: "
                         "the engine's clock also counts the load's accesses; the engine's pool is already full when tracing starts, with frames that carry the load's counts; "
                         "the engine breaks score ties by hash-set order.\n")
                
    sections.append("\n".join(s))

    for sf in sfs:
        s = [f"## SF={sf}: simulated misses per timed pass (seed means, *measured*)\n"]
        for kind in MAIN_KINDS:
            if not _pools(agg, sf, kind):
                s.append(f"### {kind}\nNo data.\n")
                continue
            s.append(f"### {kind}\n")
            s.append(kind_table(agg, sf, kind))
        sections.append("\n".join(s))

    s = [
        "## LRU-2, ARC and LIRS: plain, visit and every fetch (misses)\n",
         "Plain counts a run of repeats once, visit treats a repeat soon after as a touch, every fetch feeds the policy as the engine does today. "
         "The last column of each group is every fetch ÷ the better of plain and visit.\n"
        ]
    
    for sf in sfs:
        for kind in MAIN_KINDS:
            if _pools(agg, sf, kind):
                s.append(f"### SF={sf}, {kind}\n")
                s.append(modes_table(agg, sf, kind))
    sections.append("\n".join(s))

    for sf in sfs:
        w = window_table(passes, heads, sf)
        if w:
            sections.append(f"## Visit window check (SF={sf}, seed 1)\n\nThe visit policies with the window at 8, 64 (the default) and 512 runs. *measured*\n\n{w}")
            
        if any(k in set(heads[heads["sf"] == sf]["kind"]) for k in PLAN_CHECK_KINDS):
            seed1 = aggregate(heads[(heads["sf"] == sf) & (heads["seed"] == 1) & heads["window"].isna()], {})
            table, moved, missing = plan_table(heads, seed1, sf)
            note = ""
            
            if missing:
                note += "Missing cells (left out of the check): " + ", ".join(missing) + ".\n\n"
            if moved:
                note += "Some cell moved by more than 0.02: the plans change which pages are read, so the traces differ.\n"
            elif missing:
                note += "No cell with data moved by more than 0.02, but cells are missing, so this is not a clean pass.\n"
            else:
                note += "No cell moved by more than 0.02.\n"
                
            if plan_dec.get(sf):
                _, _, ca, cb = plan_dec[sf]
                if "unknown" in (ca, cb):
                    note += f"\nDecision on seed 1: cached-read plans say {ca}, 100 µs plans say {cb}, so the two can't be compared.\n"
                elif ca == cb:
                    note += f"\nDecision on seed 1 is the same on both plan sets: {ca}.\n"
                else:
                    note += f"\n**Decision on seed 1 is different on the two plan sets: cached-read plans say {ca}, 100 µs plans say {cb}.**\n"
                    
            sections.append(f"## Plan check (SF={sf}, seed 1)\n\n`index100_*` traces (plans priced at 100 µs per read) against `index_*`, best classical ÷ Belady. *measured*\n\n{table}\n{note}")

    s = [
        "## Time at 100 µs per read (*inferred*)\n",
         "Pass time = the recording's CPU time (median timed pass minus read time) + misses x 100 µs. Policy CPU cost is not included.\n"
        ]
    for sf in sfs:
        for kind in MAIN_KINDS:
            if _pools(agg, sf, kind):
                s.append(f"### SF={sf}, {kind}\n")
                s.append(time_table(agg, sf, kind))
    sections.append("\n".join(s))

    s = [
        "## Draft decision\n", "For a person to check. Uses `index_stream` and `index_mixed` at the 10% and 25% pools, seed means; room = best classical ≥ "
         f"{ROOM:.2f} x Belady in all four cells, close = ≤ {CLOSE:.2f} x in all four.\n"
    ]
    
    for sf in sfs:
        cells = decision_cells(agg, sf)
        tag = " (provisional: SF=0.01 is small; the call is made on SF=0.1 and SF=1.0)" if sf == "0.01" else ""
        s.append(f"**SF={sf}{tag}:** {decide(cells)}\n")
        s.append(md_table(["trace kind", "pool", "best classical", "best ÷ Belady"], [[k, p, n or DASH, fnum(r, 3) if not pr else f"{DASH} ({pr})"] for k, p, r, n, pr in cells]))
        
        if plan_dec.get(sf):
            cached, plan100, ca, cb = plan_dec[sf]
            s.append("Plan check, seed 1 only (the main answer above is over the seed means of the cached-read plans):\n")
            for label, cs in (("cached-read plans (`index_*`)", cached), ("100 µs plans (`index100_*`)", plan100)):
                s.append(f"{label}: {decide(cs)}\n")
                s.append(md_table(["trace kind", "pool", "best classical", "best ÷ Belady"], [[k, p, n or DASH, fnum(r, 3) if not pr else f"{DASH} ({pr})"] for k, p, r, n, pr in cs]))
            if "unknown" in (ca, cb):
                s.append("The two plan sets can't be compared (a cell is missing).\n")
            elif ca != cb:
                s.append(f"**The plan check changes the answer: {ca} with cached-read plans, {cb} with 100 µs plans.**\n")
            else:
                s.append(f"Same outcome on both plan sets ({ca}).\n")
                
        extra = []
        for kind in ("baseline_stream",) + DECISION_KINDS:
            for f in (0.10, 0.25, 0.50):
                r, n = best_over_belady(agg, sf, kind, pool_for_fraction(sf, f))
                extra.append([kind, f"{round(f * 100)}%", fnum(r, 3), n or DASH])
                
        s.append("Next to it, `baseline_stream` and the 50% pool:\n")
        s.append(md_table(["trace kind", "pool", "best ÷ Belady", "best"], extra))
    sections.append("\n".join(s))

    s = ["## Which classical policies are worth building in the engine\n"]
    for sf in sfs:
        text, near, cands = hint(agg, sf)
        s.append(f"### SF={sf}\n")
        s.append("Misses of each classical policy ÷ LRU and ÷ the best at that pool (seed means, pools 5-90%):\n")
        s.append(text)
        s.append(near)
        s.append(HINT_LEGEND + "\n")
        s.append("Worth building: a non-LRU policy counts at a pool only if it is within 2% of the best there and at least 2% below LRU.\n")
        
        if cands:
            s.append("\n".join(f"- {p} [{_mode_tag(p)}]: {where}" for p, where in cands.items()) + "\n")
        else:
            s.append("None: at every 5-75% pool no policy is both near the best and 2% below LRU.\n")
            
        pen = [[kind, _label(sf, pool)] + [fnum(full_penalty(agg, sf, kind, pool, fam), 3) for fam in MODES] for kind in MAIN_KINDS for pool in _pools(agg, sf, kind)]
        s.append("Every-fetch penalty (`<name>_full` ÷ the better of `<name>` and `<name>_visit`; above 1 means feeding the engine's every access loses misses):\n")
        s.append(md_table(["trace kind", "pool"] + list(MODES), pen))
    sections.append("\n".join(s))

    sections.append("## How to read this\n\n"
                    "Misses are simulated on the recorded page-fetch traces (cold start, no pins). Belady is the floor no policy can beat. "
                    "`best ÷ Belady` of 1.00 means the best classical policy loses nothing to the floor. Seed means average the seeds' first timed pass. "
                    "`default` is the benchmark's default pool; the other pools are shares of the database's table pages.\n")

    head = ""
    if warns:
        head = "**WARNING**\n\n" + "\n".join(f"- {w}" for w in warns) + "\n\n"
    return "# Eviction headroom\n\n" + head + "\n\n".join(sections)


# chart 

def _family_series(agg: pd.DataFrame, sf: str, kind: str, family: str) -> pd.Series:
    names = {
        "lirs": ("lirs", "lirs_visit"),
        "arc": ("arc", "arc_visit"),
        "lru2": ("lru2", "lru2_visit")
    }.get(family, (family,))
    sub = agg[(agg["sf"] == sf) & (agg["kind"] == kind) & agg["policy"].isin(names)]
    sub = sub[sub["pool_label"].str.endswith("%")]
    return sub.groupby("pool")["misses"].min()


def plot_misses(sf: str, agg: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor=SURFACE)
    for ax, kind in zip(axes.flat, MAIN_KINDS):
        ax.set_facecolor(SURFACE)
        ax.set_title(kind, color=INK)
        drew = False
        for family, color in list(FAMILY_COLORS.items()) + [("belady", INK)]:
            series = _family_series(agg, sf, kind, family)
            if series.empty:
                continue
            drew = True
            x = [p / DB_PAGES[sf] * 100 for p in series.index]
            ax.plot(x, series.values, color=color, linewidth=2, marker="o", markersize=8, linestyle="--" if family == "belady" else "-", markeredgecolor=SURFACE)
            
        if not drew:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", color=MUTED, transform=ax.transAxes)
            
        ax.set_xlabel("pool, % of the database's pages", color=MUTED)
        ax.set_ylabel("misses per timed pass", color=MUTED)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED)
        
    handles = [Line2D([], [], color=c, linewidth=2, marker="o", markersize=8, label=FAMILY_LABELS[f]) for f, c in FAMILY_COLORS.items()]
    handles.append(Line2D([], [], color=INK, linewidth=2, marker="o", markersize=8, linestyle="--", label="Belady (floor)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8)
    fig.suptitle(f"Simulated misses at SF {sf} (lower is better; Belady is the floor)", color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


#  CLI 

def run(sim_dir: Path, traces_dir: Path) -> int:
    passes, sim_items = load_sim(sim_dir)
    engine = load_engine_runs(sim_dir, traces_dir)
    heads = headlines(passes)
    sfs = sorted(heads["sf"].unique(), key=_sf_key)

    val = validate(sim_items, engine["recording"] + engine["validate"])
    cpu = cpu_table(engine["recording"])
    main_heads = heads[heads["window"].isna() & heads["kind"].isin(MAIN_KINDS)]
    agg = aggregate(main_heads, cpu)

    val.to_csv(sim_dir / "validation.csv", index=False)
    out = agg.sort_values(["sf", "kind", "pool", "policy"], key=lambda c: c.map(_sf_key) if c.name == "sf" else c)
    out[CSV_COLUMNS].to_csv(sim_dir / "eviction_headroom.csv", index=False, float_format="%.10g")
    (sim_dir / "report.md").write_text(build_report(sfs, agg, heads, passes, val, sim_dir, engine))
    written = [sim_dir / "validation.csv", sim_dir / "eviction_headroom.csv", sim_dir / "report.md"]
    for sf in sfs:
        path = sim_dir / f"misses_sf{sf}.png"
        plot_misses(sf, agg, path)
        written.append(path)

    print("Wrote:")
    for p in written:
        print(f"  {p}")
    bad = int((val["match"] == 0).sum()) if len(val) else 0
    code = 0
    if bad:
        print(f"Error: {bad} engine items do not match the simulated LRU (see validation.csv)", file=sys.stderr)
        code = 1
        
    for problem in unvalidated(heads, val):
        print(f"Error: {problem}", file=sys.stderr)
        code = 1
        
    plain = heads[heads["window"].isna()]
    for sf in sfs:
        for problem in fullcheck(sim_dir, sf, plain[plain["sf"] == sf])[1]:
            print(f"Error: {problem}", file=sys.stderr)
            code = 1
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize eviction_sim CSVs into a report and charts.")
    parser.add_argument("--sim-dir", type=Path, default=Path("bench/results/headroom/eviction"))
    parser.add_argument("--traces-dir", type=Path, default=Path("bench/results/traces"))
    args = parser.parse_args(argv)
    try:
        return run(args.sim_dir, args.traces_dir)
    except AnalyzerError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
