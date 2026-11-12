from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


# Load vc_common from next to this file, reusing it if it's already loaded.
def _load_common():
    path = Path(__file__).resolve().parent / "vc_common.py"
    loaded = sys.modules.get("vc_common")
    if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() == path:
        return loaded

    spec = importlib.util.spec_from_file_location("vc_common", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"can't load {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["vc_common"] = module
    spec.loader.exec_module(module)
    return module


vc = _load_common()
vf = vc.load_sibling("vc_filters", "vector_filters.py")

# The bars, fixed before any result; a test pins every value.
BANDS = (("A", 0.0, 0.005), ("B", 0.005, 0.05), ("C", 0.05, 0.5), ("D", 0.5, float("inf")))
RECALL_BAR = 0.95
ROOM_BAR = 2.0
KEEP_SHARE = 0.5
RULE_EDGES = (0.0, 0.001, 0.003, 0.01, 0.03, 0.10, 0.30, 1.0)
TARGETS = (0.90, 0.95, 0.99)
WORK_BAR = 0.70
RECALL_TOL = 0.005
QUESTION_TOL = 0.01
E_RECALL = 0.995
E_CHOICES = (500, 1000, 2000)
GATE_SIZE = "500k"
MIN_EXT = (0, 8, 0)
MIN_GATE_QUERIES = 30
EXACT_OPTIONS = ("exact_seq", "exact_bitmap", "set_noindex")
ITERATIVE_COUNT = 10
MEMCHECK_CELLS = 20
DETERMINISM_QUERIES = 20
WARMUP = 50
RUNS = 3
GATE_TEST_QUERIES = 1000
GATE_QUESTIONS = 210
GATE_TRAIN_QUERIES = 8000
TREES = 100
LEARNING_RATE = 0.1
CHARGE_DIVISOR = 25
CHARGE_FACTORS = (0, 1, 4)
EPS = 1e-12  # slack for binary rounding at a bar

# My own copy of the grid's 33 options; a test checks it against the grid.
CONFIGS = (
    "off_ef40", "off_ef100", "off_ef200", "off_ef400", "off_ef1000",
    "strict_ef40", "strict_ef100", "strict_ef200", "strict_ef400",
    "relaxed_ef40", "relaxed_ef100", "relaxed_ef200", "relaxed_ef400",
    "relaxed100k_ef40", "relaxed100k_ef100"
)

FORCED = ("exact_seq", "exact_bitmap", *(f"hnsw_{c}" for c in CONFIGS))
SETTINGS = ("set_default", *(f"set_{c}" for c in CONFIGS[1:]), "set_noindex")
OPTIONS = FORCED + SETTINGS
INTENDED = {"exact_seq": "seq", "exact_bitmap": "bitmap", **{f"hnsw_{c}": "hnsw" for c in CONFIGS}}
ITERATIVE = tuple(f"hnsw_{c}" for c in CONFIGS if not c.startswith("off_"))
SCANS = ("seq", "bitmap", "hnsw", "btree_index")
FAMILY_COLUMNS = {
    "date": "review_date", "product": "product_id", "rating": "rating",
    "category": "category", "verified": "verified",
}

INPUTS = ("grid.csv", "memcheck.csv", "grid_info.json", "pg_load.json", "rem.csv", "determinism.csv", "plain_E.json", "train_info.json", "stopping.csv", "stopping_queries.csv")
GRID_NEEDS = ("qid", "half", "filter", "family", "n_pass", "share", "est_share", "option", "kind", "intended", "scan", "index", "recall", "time_ms", "timed_out", "deterministic")
GRID_NUMBERS = ("n_pass", "share", "est_share", "recall", "time_ms")
STOP_NUMBERS = ("target", "mean_recall", "mean_ndis", "mean_calls", "work_1x", "below_target_share", "n_queries")
QUERY_NUMBERS = ("target", "recall", "ndis", "calls")
METHODS = ("rem", "learned", "plain")
BAR_SETS = ("test", "questions")
CHOICES = ("ceiling", "baseline", "outofbox", "rule")
CELL_COLUMNS = ["qid", "group", "half", "filter", "family", "band", "n_pass", "share", "est_share", "bucket"]
COLORS = {"outofbox": "#2a78d6", "baseline": "#eb6834", "best": "#eda100", "rule": "#e87ba4", "ceiling": "#1baf7a"}


# x >= bar, inclusive, with EPS slack scaled to the numbers.
def at_least(x: float, bar: float) -> bool:
    return x >= bar - EPS * max(1.0, abs(x), abs(bar))


# x <= bar, inclusive, with the same slack.
def at_most(x: float, bar: float) -> bool:
    return at_least(bar, x)


# The band of a true passing share; lower edges are inclusive.
def band_of(share: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= share < hi:
            return name
    raise ValueError(f"share must be a number >= 0, got {share}")


# The rule bucket of an estimated share; any share >= 1.0 goes last.
def bucket_of(est_share: float) -> int:
    if not est_share >= 0:
        raise ValueError(f"estimated share must be a number >= 0, got {est_share}")
    return min(bisect.bisect_right(RULE_EDGES, est_share) - 1, len(RULE_EDGES) - 2)


# The grid with band and bucket columns, added from share and est_share when missing.
def with_bands(grid: pd.DataFrame) -> pd.DataFrame:
    g = grid
    if "band" not in g:
        g = g.assign(band=[band_of(s) for s in g["share"]])
    if "bucket" not in g:
        g = g.assign(bucket=[bucket_of(s) for s in g["est_share"]])
    return g


# The setting with the least total time whose mean recall meets the bar everywhere.
# Ties go to the first name; None if no setting qualifies.
def best_single_setting(grid: pd.DataFrame) -> str | None:
    g = with_bands(grid)
    bands = set(g["band"])
    good = []
    for option, rows in g[g["option"].isin(SETTINGS)].groupby("option"):
        means = rows.groupby("band")["recall"].mean()
        if all(b in means.index and at_least(means[b], RECALL_BAR) for b in bands):
            good.append((rows["time_ms"].sum(), option))
    return min(good)[1] if good else None


# Per cell, the fastest option with recall at the bar, ties by name.
# A cell with none takes its best-recall option (a warning case).
def ceiling_choice(grid: pd.DataFrame) -> pd.DataFrame:
    ok = grid["recall"] >= RECALL_BAR - EPS
    g = grid.assign(ok=ok, rank=np.where(ok, 0.0, grid["recall"]))
    g = g.sort_values(["ok", "rank", "time_ms", "option"], ascending=[False, False, True, True])
    picked = g.drop_duplicates(["qid", "filter"])
    return picked[["qid", "filter", "option", "time_ms", "recall"]].reset_index(drop=True)


# Rows the rule sees: a none cell's exact_bitmap is its exact_seq row.
def rule_rows(grid: pd.DataFrame) -> pd.DataFrame:
    bitmap = grid[grid["option"] == "exact_bitmap"]
    have = set(zip(bitmap["qid"], bitmap["filter"]))
    seq = grid[(grid["filter"] == "none") & (grid["option"] == "exact_seq")]
    seq = seq[[(q, f) not in have for q, f in zip(seq["qid"], seq["filter"])]]
    return pd.concat([grid, seq.assign(option="exact_bitmap")], ignore_index=True)


# Per bucket, the option with the least training time among those meeting the bar.
# An empty bucket copies the nearest pick, the lower bucket on a tie.
def fit_rule(train_cells: pd.DataFrame) -> dict[int, str]:
    if train_cells.empty:
        return {}
    picks = {}
    for bucket, rows in with_bands(train_cells).groupby("bucket"):
        good = [(r["time_ms"].sum(), o) for o, r in rows.groupby("option") if at_least(r["recall"].mean(), RECALL_BAR)]
        if good:
            picks[int(bucket)] = min(good)[1]
            
    if not picks:
        return {}
    return {i: picks[min(picks, key=lambda j: (abs(i - j), j))] for i in range(len(RULE_EDGES) - 1)}


# Each cell's rule option with that option's time and recall there.
def apply_rule(rule: dict[int, str], cells: pd.DataFrame) -> pd.DataFrame:
    g = with_bands(cells)
    keys = g.drop_duplicates(["qid", "filter"])[["qid", "filter", "bucket"]]
    keys = keys.assign(option=[rule.get(int(b), "") for b in keys["bucket"]])
    rows = g[["qid", "filter", "option", "time_ms", "recall"]]
    return keys.merge(rows, on=["qid", "filter", "option"], how="left").reset_index(drop=True)


# Total ms and mean recall of a choice over cells; None if any is missing.
def _total(cells: pd.DataFrame, label: str) -> tuple:
    ms = cells[f"{label}_ms"]
    if cells.empty or ms.isna().any():
        return None, None
    return float(ms.sum()), float(cells[f"{label}_recall"].mean())


# The timed-out rows behind these choices' totals, as (qid, filter, option).
def _timed_out_picks(cells: pd.DataFrame, labels) -> set:
    found = set()
    for label in labels:
        hit = cells[cells[f"{label}_timed_out"].eq(True)]
        found |= set(zip(hit["qid"], hit["filter"], hit[f"{label}_option"]))
    return found


# Part 1 per band: out of box, baseline, per-band best, ceiling, the rule, both clauses.
def part1(grid: pd.DataFrame) -> dict:
    g = with_bands(grid)
    if "timed_out" not in g:
        g = g.assign(timed_out=False)
    baseline = best_single_setting(g)
    filled = rule_rows(g)
    rule = fit_rule(filled[filled["half"] == "train"])
    picks = {
        "ceiling": ceiling_choice(g), 
        "rule": apply_rule(rule, filled),
        "baseline": g[g["option"] == baseline], 
        "outofbox": g[g["option"] == "set_default"]
    }
    
    cells = g.drop_duplicates(["qid", "filter"])[[c for c in CELL_COLUMNS if c in g]].reset_index(drop=True)
    flags = filled[["qid", "filter", "option", "timed_out"]].drop_duplicates(["qid", "filter", "option"])
    for label in CHOICES:
        frame = picks[label][["qid", "filter", "option", "time_ms", "recall"]].merge(flags, on=["qid", "filter", "option"], how="left")
        names = {
            "option": f"{label}_option", 
            "time_ms": f"{label}_ms", 
            "recall": f"{label}_recall",
            "timed_out": f"{label}_timed_out"
        }
        cells = cells.merge(frame.rename(columns=names), on=["qid", "filter"], how="left")

    bands, passing = [], None
    for name, _, _ in BANDS:
        c = cells[cells["band"] == name]
        t = c[c["half"] == "test"]
        row = {
            "band": name, 
            "n_cells": len(c), 
            "n_test": len(t)
        }
        for label in ("outofbox", "baseline", "ceiling"):
            row[f"{label}_ms"], row[f"{label}_recall"] = _total(c, label)
            
        best = best_single_setting(g[g["band"] == name]) if len(c) else None
        rows = g[(g["band"] == name) & (g["option"] == best)]
        row["best_setting"] = best
        row["best_ms"], row["best_recall"] = (None, None)
        if best:
            row["best_ms"], row["best_recall"] = float(rows["time_ms"].sum()), float(rows["recall"].mean())

        b, cc = row["baseline_ms"], row["ceiling_ms"]
        row["room"] = b / cc if b is not None and cc else None
        row["room_ok"] = b is not None and cc is not None and at_least(b, ROOM_BAR * cc)
        for label in ("baseline", "rule", "ceiling"):
            row[f"test_{label}_ms"], row[f"test_{label}_recall"] = _total(t, label)
            
        row["rule_ms"], row["rule_recall"] = row.pop("test_rule_ms"), row.pop("test_rule_recall")
        tb, tr, tc = row["test_baseline_ms"], row["rule_ms"], row["test_ceiling_ms"]
        known = None not in (tb, tr, tc)
        row["rule_keep"] = (tb - tr) / (tb - tc) if known and tb != tc else None
        row["rule_ok"] = (known and at_least(row["rule_recall"], RECALL_BAR) and at_least(tb - tr, KEEP_SHARE * (tb - tc)))

        # Room totals cover every cell, the rule clause's only the test half.
        behind = set()
        if row["room_ok"]:
            behind |= _timed_out_picks(c, ("baseline", "ceiling"))
        if row["rule_ok"]:
            behind |= _timed_out_picks(t, ("baseline", "rule", "ceiling"))
        row["timeouts_behind"] = len(behind)
        
        if passing is None and row["room_ok"] and row["rule_ok"]:
            passing = row
        bands.append(row)

    return {
        "bands": bands, "verdict": "GO" if passing else "NO-GO", "band": passing["band"] if passing else None,
        "rule_ratio": passing["test_baseline_ms"] / passing["rule_ms"] if passing else None,
        "baseline": baseline, "rule": rule, "cells": cells,
    }


# Work in distance computations, each predictor call charged trees / 25 times factor.
def work(ndis: float, calls: float, factor: float = 1) -> float:
    return ndis + calls * TREES / CHARGE_DIVISOR * factor


# One stopping.csv row as numbers, with work at every charge factor.
def _method_row(r: dict) -> dict:
    ndis, calls = float(r["mean_ndis"]), float(r["mean_calls"])
    return {
        "param": r["param"], "margin_ok": r["margin_ok"], "recall": float(r["mean_recall"]),
        "ndis": ndis, "calls": calls, "below": float(r["below_target_share"]),
        "work": {f: work(ndis, calls, f) for f in CHARGE_FACTORS},
    }


# Part 2 per target: REM, learned and plain on both sets, and the three clauses.
def part2(stopping: pd.DataFrame) -> dict:
    found = {(r["set"], f"{float(r['target']):g}", r["method"]): r for r in stopping.to_dict("records")}
    targets = []
    for target in TARGETS:
        entry = {"target": target}
        for s in BAR_SETS:
            rows = {m: found.get((s, f"{target:g}", m)) for m in METHODS}
            entry[s] = {m: None if r is None else _method_row(r) for m, r in rows.items()}
            
        complete = all(entry[s][m] is not None for s in BAR_SETS for m in METHODS)
        rem, learned, plain = (entry["test"][m] for m in METHODS)
        for key, other in (("vs_rem", rem), ("vs_plain", plain)):
            entry[key] = {f: learned["work"][f] / other["work"][f] if complete and other["work"][f] else None for f in CHARGE_FACTORS}
            
        q_rem, q_learned = entry["questions"]["rem"], entry["questions"]["learned"]
        entry["recall_ok"] = complete and at_least(learned["recall"], target - RECALL_TOL)
        entry["work_ok"] = complete and at_most(learned["work"][1], WORK_BAR * rem["work"][1])
        entry["questions_ok"] = complete and at_least(q_learned["recall"], q_rem["recall"] - QUESTION_TOL)
        targets.append(entry)
        
    go = all(t["recall_ok"] and t["work_ok"] and t["questions_ok"] for t in targets)
    return {
        "targets": targets, 
        "verdict": "GO" if go else "NO-GO"
    }


# A CSV as strings, or None after a warning: missing, unreadable or short of columns.
def _csv(path: Path, needs, warn) -> pd.DataFrame | None:
    if not path.exists():
        warn(f"{path.name} is missing")
        return None
    
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (ValueError, OSError) as e:
        warn(f"{path.name} can't be read ({e})")
        return None
    
    missing = [c for c in needs if c not in df.columns]
    if missing:
        warn(f"{path.name} lacks columns {', '.join(missing)}")
        return None
    return df


# A JSON object, or None after a warning.
def _json(path: Path, warn) -> dict | None:
    if not path.exists():
        warn(f"{path.name} is missing")
        return None
    
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError) as e:
        warn(f"{path.name} can't be read ({e})")
        return None
    
    if not isinstance(data, dict):
        warn(f"{path.name} isn't a JSON object")
        return None
    return data


# Parse number columns in place; False after a warning if a value isn't a number.
def _numbers(df: pd.DataFrame, cols, name: str, warn) -> bool:
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.isna().any():
            warn(f"{name}: a {c} value isn't a number ({df[c][v.isna()].iloc[0]!r})")
            return False
        
        df[c] = v.astype(float)
    return True


# Parse True / False columns in place; False after a warning on anything else.
def _bools(df: pd.DataFrame, cols, name: str, warn) -> bool:
    for c in cols:
        bad = ~df[c].isin(["True", "False"])
        if bad.any():
            warn(f"{name}: a {c} value isn't True or False ({df[c][bad].iloc[0]!r})")
            return False
        
        df[c] = (df[c] == "True").to_numpy()
    return True


# One warning for a list of problems: how many, and the first.
def _many(warn, items: list, what: str) -> None:
    if items:
        warn(f"{len(items)} {what} (first: {items[0]})")


# A version string as three numbers, or None.
def _version(text) -> tuple | None:
    try:
        parts = tuple(int(x) for x in str(text).split("."))
    except ValueError:
        return None
    return (parts + (0, 0, 0))[:3]


# The current cell set: (qid, filter) -> (group, half), built as the grid builds it.
def _cell_set(results: Path, size: str, info: dict, warn) -> dict | None:
    n = info.get("n_per_group")
    if not isinstance(n, int) or isinstance(n, bool):
        warn(f"grid_info.json n_per_group {n!r} isn't a whole number")
        return None
    
    if size == GATE_SIZE and n < MIN_GATE_QUERIES:
        warn(f"grid_info.json n_per_group {n} is below the {MIN_GATE_QUERIES} queries per group the gate needs")
    reviews = _csv(results / "data" / "query_reviews.csv", ("qid", "split"), warn)
    if reviews is None:
        return None
    
    q = vc.read_questions()
    tests = reviews.loc[reviews["split"] == "test", "qid"].tolist()
    written = q.loc[q["group"] == "written", "qid"].tolist()
    ai53 = q.loc[q["group"] == "ai53", "qid"].tolist()
    
    try:
        queries = vf.part1_queries(tests, written, ai53, size, n_per_group=n)
    except ValueError as e:
        warn(f"grid_info.json n_per_group {n} gives no query set ({e})")
        return None
    
    return {(qid, f): (group, half) for qid, group, half in queries for f in vf.FILTER_NAMES}


# "qid filter option" for each row, to name examples in warnings.
def _names(rows: pd.DataFrame) -> list[str]:
    return [f"{q} {f} {o}" for q, f, o in zip(rows["qid"], rows["filter"], rows["option"])]


# The index a forced option must read: hnsw's, the filter column's, or none.
def _forced_index(size: str, option: str, family: str) -> str:
    if option.startswith("hnsw_"):
        return f"reviews_{size}_hnsw_idx"
    if option == "exact_bitmap":
        return f"reviews_{size}_{FAMILY_COLUMNS.get(family, '?')}_idx"
    return ""


# The grid's no-verdict checks: options per cell, exact recall, HNSW repeats, forced plans.
def _check_grid(g: pd.DataFrame, cells: dict, size: str, warn) -> None:
    have = g.groupby(["qid", "filter"])["option"].apply(list).to_dict()
    missing, twice, extra = [], [], []
    for q, f in cells:
        opts = have.get((q, f), [])
        want = set(OPTIONS) - ({"exact_bitmap"} if f == "none" else set())
        if want - set(opts):
            missing.append(f"{q} {f} lacks {', '.join(sorted(want - set(opts))[:3])}")
        if len(opts) != len(set(opts)):
            twice.append(f"{q} {f}")
        if set(opts) - want:
            extra.append(f"{q} {f} has {', '.join(sorted(set(opts) - want))}")
            
    ok = (g["recall"] >= RECALL_BAR - EPS).groupby([g["qid"], g["filter"]]).any()
    no_ceiling = [f"{q} {f}" for (q, f), v in ok.items() if not v]
    _many(warn, missing, "cells miss an option")
    _many(warn, twice, "cells have an option twice")
    _many(warn, extra, "cells have an unexpected option")
    _many(warn, no_ceiling, "cells have no option reaching recall 0.95")

    exact_short = g["option"].isin(EXACT_OPTIONS) & ~g["timed_out"] & (g["recall"] < 1.0)
    _many(warn, _names(g[exact_short]), "exact option rows have recall below 1.0")
    changed = (g["scan"] == "hnsw") & ~g["timed_out"] & ~g["deterministic"]
    _many(warn, _names(g[changed]), "HNSW rows changed between their timed runs")
    known = g[g["option"].isin(OPTIONS)]
    kind = np.where(known["option"].isin(FORCED), "forced", "setting")
    intended = [INTENDED.get(o, "") for o in known["option"]]
    wrong = (known["kind"] != kind) | (known["intended"] != intended)
    _many(warn, _names(known[wrong]), "rows have the wrong kind or intended scan")

    f = g[g["option"].isin(FORCED)]
    want_index = [_forced_index(size, o, fam) for o, fam in zip(f["option"], f["family"])]
    off = (f["scan"] != f["option"].map(INTENDED)) | (f["index"] != want_index)
    _many(warn, _names(f[off]), "forced rows don't run their intended plan")


# The grid cut to the current cells, parsed and checked; grid is None when malformed.
def _grid(grid: pd.DataFrame, cells: dict, size: str, warn) -> dict:
    inside = np.array([k in cells for k in zip(grid["qid"], grid["filter"])], dtype=bool)
    g = grid[inside].copy()
    out = {"dropped": int((~inside).sum()), "grid": None, "timeouts": 0}
    meta = [cells[k] for k in zip(g["qid"], g["filter"])]
    
    halves = np.array([m[1] for m in meta], dtype=object)
    bad = g[g["half"].to_numpy() != halves]
    _many(warn, [f"{q} {f}" for q, f in zip(bad["qid"], bad["filter"])], "grid rows have a half that differs from the query split")
    g["group"], g["half"] = [m[0] for m in meta], halves
    parsed = _numbers(g, GRID_NUMBERS, "grid.csv", warn)
    
    if not (parsed and _bools(g, ("timed_out", "deterministic"), "grid.csv", warn)):
        return out
    if (g["n_pass"] % 1 == 0).all():
        g["n_pass"] = g["n_pass"].astype(np.int64)
        
    try:
        g = with_bands(g)
    except ValueError as e:
        warn(f"grid.csv: {e}")
        return out
    
    _check_grid(g, cells, size, warn)
    out["timeouts"] = int(g["timed_out"].sum())
    out["grid"] = g.drop_duplicates(["qid", "filter", "option"]).reset_index(drop=True)
    if len(out["grid"]) and best_single_setting(out["grid"]) is None:
        warn("no setting reaches mean recall 0.95 in every band, so there's no baseline")
    return out


# Memory check: every ids_equal true, the right band A cells, the 10 iterative options each.
def _check_memcheck(mem: pd.DataFrame, band_a: set | None, warn) -> None:
    if not _bools(mem, ("ids_equal",), "memcheck.csv", warn):
        return
    _many(warn, _names(mem[~mem["ids_equal"]]), "memcheck rows return different ids at 512MB")
    per = mem.groupby(["qid", "filter"])["option"].apply(list)
    _many(warn, [f"{q} {f}" for (q, f), opts in per.items() if sorted(opts) != sorted(ITERATIVE)], f"memcheck cells don't have exactly the {ITERATIVE_COUNT} iterative options")
    
    if band_a is not None:
        want = min(MEMCHECK_CELLS, len(band_a))
        outside = [k for k in per.index if k not in band_a]
        if len(per) != want or outside:
            warn(f"memcheck.csv covers {len(per)} cells ({len(outside)} not band A cells of the set); want {want} band A cells")


# A whole number, not a bool.
def _whole(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


# grid_info.json's grid runs, or None. At the gate a missing history warns,
# and so do another warm-up or run count and N changed after a finished grid.
def _history(info: dict, gate: bool, warn) -> list[dict] | None:
    runs = info.get("history")
    if not (isinstance(runs, list) and runs and all(isinstance(r, dict) for r in runs)):
        if gate:
            warn("grid_info.json has no history of grid runs")
        return None
    
    if not gate:
        return runs

    odd = [f"run {i}: warmup {r.get('warmup')!r}, runs {r.get('runs')!r}" for i, r in enumerate(runs, 1) if r.get("warmup") != WARMUP or r.get("runs") != RUNS]
    _many(warn, odd, f"grid runs in grid_info.json's history don't use warmup {WARMUP} and runs {RUNS}")
    changed = []
    for i, done in enumerate(runs):
        n = done.get("n_per_group")
        if done.get("complete") is True and _whole(n):
            changed += [(n, r["n_per_group"]) for r in runs[i + 1:] if _whole(r.get("n_per_group")) and r["n_per_group"] != n]
            
    if changed:
        warn(f"grid_info.json's history changes N from {changed[0][0]} to {changed[0][1]} after a finished grid")
    return runs


# Read and check one size folder; every problem becomes a warning naming the size.
def load_size(size_dir: Path, size: str, other_ext: str | None) -> dict:
    d = {"size": size, "warnings": [], "grid": None, "dropped": 0, "timeouts": 0, "n": None, "stopping": None, "history": None}

    # Add a warning that names this size.
    def warn(message: str) -> None:
        d["warnings"].append(f"{size}: {message}")

    gate = size == GATE_SIZE
    load, info = _json(size_dir / "pg_load.json", warn), _json(size_dir / "grid_info.json", warn)
    ext = (load or {}).get("extversion")
    d["server_version"], d["extversion"] = (load or {}).get("server_version"), ext
    if load is not None:
        parsed = _version(ext)
        if parsed is None or parsed < MIN_EXT:
            warn(f"pgvector extversion {ext!r} is below 0.8.0")
            
        if other_ext is not None and ext != other_ext:
            warn(f"pgvector extversion {ext!r} differs from the other size's {other_ext!r}")

    cells = None
    if info is not None:
        if info.get("size") != size:
            warn(f"grid_info.json size {info.get('size')!r} isn't {size!r}")
        if load is not None and info.get("extversion") != ext:
            warn(f"grid_info.json extversion {info.get('extversion')!r} differs from pg_load.json's {ext!r}")

        # The grid records sha256 of pg_load.json's bytes; a reload writes a new file.
        if load is not None and info.get("load_sha256") != hashlib.sha256((size_dir / "pg_load.json").read_bytes()).hexdigest():
            warn("grid_info.json load_sha256 doesn't match pg_load.json, so the grid was measured on another table load")

        d["history"] = _history(info, gate, warn)
        d["n"] = info.get("n_per_group")
        cells = _cell_set(size_dir.parent, size, info, warn)
        
    grid = _csv(size_dir / "grid.csv", GRID_NEEDS, warn)
    if grid is not None and cells is not None:
        d.update(_grid(grid, cells, size, warn))
    d["n_cells"] = len(cells) if cells else 0

    mem = _csv(size_dir / "memcheck.csv", ("qid", "filter", "option", "ids_equal"), warn)
    if mem is not None:
        band_a = None
        if d["grid"] is not None:
            a = d["grid"][d["grid"]["band"] == "A"]
            band_a = set(zip(a["qid"], a["filter"]))
        _check_memcheck(mem, band_a, warn)

    rem = _csv(size_dir / "rem.csv", ("set", "qid", "ef", "recall", "ndis"), warn)
    d["rem"] = rem if rem is not None and _numbers(rem, ("ef", "recall", "ndis"), "rem.csv", warn) else None
    det = _csv(size_dir / "determinism.csv", ("qid", "same_ids", "same_ndis"), warn)
    if det is not None and _bools(det, ("same_ids", "same_ndis"), "determinism.csv", warn):
        n_val = DETERMINISM_QUERIES
        if d["rem"] is not None:
            n_val = d["rem"].loc[d["rem"]["set"] == "validation", "qid"].nunique()
            
        if len(det) < min(DETERMINISM_QUERIES, n_val):
            warn(f"determinism.csv has {len(det)} queries; want {min(DETERMINISM_QUERIES, n_val)}")
            
        changed = det.loc[~det["same_ids"] | ~det["same_ndis"], "qid"]
        _many(warn, list(changed), "determinism.csv queries changed on a re-run")

    plain, train = _json(size_dir / "plain_E.json", warn), _json(size_dir / "train_info.json", warn)
    d["limits"] = (plain or {}).get("limits")
    if plain is not None:
        e = plain.get("E")
        if e is None:
            warn("plain_E.json: E is null (no E reached validation recall 0.995)")
        elif e not in E_CHOICES:
            warn(f"plain_E.json: E {e!r} is not one of 500, 1000, 2000")
        else:
            v = (plain.get("val_recall") or {}).get(str(e))
            if not isinstance(v, (int, float)) or not at_least(v, E_RECALL):
                warn(f"plain_E.json: validation recall at E {e} is {v!r}, below 0.995")
                
    if train is not None and (train.get("trees") != TREES or train.get("learning_rate") != LEARNING_RATE):
        warn(f"train_info.json: trees {train.get('trees')!r} and learning_rate {train.get('learning_rate')!r} aren't 100 and 0.1")
    if gate and train is not None and train.get("queries") != GATE_TRAIN_QUERIES:
        warn(f"train_info.json: queries {train.get('queries')!r}, want {GATE_TRAIN_QUERIES} at the gate size")
    for name, data in (("plain_E.json", plain), ("train_info.json", train)):
        if gate and data is not None and data.get("limits"):
            warn(f"{name}: limits {data.get('limits')} cut the query sets at the gate size")
    d["targets_info"] = (train or {}).get("targets") or {}

    stop = _csv(size_dir / "stopping.csv", ("set", "method", "param", "margin_ok", *STOP_NUMBERS), warn)
    d["stopping_raw"] = None if stop is None else stop.copy()
    if stop is not None and _numbers(stop, STOP_NUMBERS, "stopping.csv", warn):
        keys = [(s, f"{t:g}", m) for s, t, m in zip(stop["set"], stop["target"], stop["method"])]
        counts = Counter(k for k in keys if k[0] in BAR_SETS)
        want = [(s, f"{t:g}", m) for s in BAR_SETS for t in TARGETS for m in METHODS]
        bad = [" ".join(k) for k in want if counts[k] != 1] + [" ".join(k) for k in counts if k not in want]
        _many(warn, bad, "stopping.csv rows for the bars aren't there exactly once")
        mine = [work(n, c, 1) for n, c in zip(stop["mean_ndis"], stop["mean_calls"])]
        off = [" ".join(k) for k, w, x in zip(keys, stop["work_1x"], mine) if abs(w - x) > 1e-9 * max(1.0, abs(x))]
        _many(warn, off, "stopping.csv work_1x values differ from ndis + calls x 4")
        for s, want in (("test", GATE_TEST_QUERIES), ("questions", GATE_QUESTIONS)):
            wrong = stop.loc[(stop["set"] == s) & (stop["n_queries"] != want), "n_queries"]
            if gate and len(wrong):
                warn(f"stopping.csv: {s} rows have n_queries {wrong.iloc[0]:g}, want {want} at the gate size")
        d["stopping"] = stop
        
    sq = _csv(size_dir / "stopping_queries.csv", ("set", "qid", "method", *QUERY_NUMBERS), warn)
    parsed = sq is not None and _numbers(sq, QUERY_NUMBERS, "stopping_queries.csv", warn)
    d["queries"] = sq if parsed else None
    return d


# Every no-verdict warning for one size folder.
def validate(size_dir: Path, size: str, other_ext: str | None) -> list[str]:
    return load_size(size_dir, size, other_ext)["warnings"]


# A size folder is present when it holds any analyzer input.
def is_present(size_dir: Path) -> bool:
    return any((size_dir / name).exists() for name in INPUTS)


# The extversion in a folder's pg_load.json, or None.
def _extversion(size_dir: Path):
    try:
        return json.loads((size_dir / "pg_load.json").read_text()).get("extversion")
    except (OSError, ValueError, AttributeError):
        return None


# Short number formats for the report; "-" for a missing value.
def _f(v, spec: str = ",.1f", suffix: str = "") -> str:
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:{spec}}{suffix}"


# A total with its mean recall, like "1,234.5 (0.9812)".
def _mr(ms, recall) -> str:
    return "-" if ms is None else f"{ms:,.1f} ({recall:.4f})"


# "yes" or "no" for a clause.
def _yes(v) -> str:
    return "yes" if v else "no"


# How a part's reading prints: bold only in a section without warnings.
def _reading(text: str, dev: bool, warned: bool) -> str:
    label = "Development reading, not a verdict" if dev else "Reading"
    if not warned:
        return f"{label}: **{text}**."
    
    return f"{label} ({'see the warnings' if dev else 'no verdict: see the warnings'}): {text}."


# Part 1's report lines for one size.
def part1_lines(p1: dict, dev: bool, warned: bool = False) -> list[str]:
    reading = p1["verdict"]
    if p1["band"]:
        reading += f" in band {p1['band']}, rule {p1['rule_ratio']:.2f}x faster than the baseline"
    last = len(RULE_EDGES) - 2
    spans = [f">= {RULE_EDGES[i]:g}" if i == last else f"[{RULE_EDGES[i]:g}, {RULE_EDGES[i + 1]:g})" for i in range(last + 1)]
    rule = "; ".join(f"{spans[i]} {o}" for i, o in sorted(p1["rule"].items()))
    lines = [
        "### Part 1: filtered-search choice", "", _reading(reading, dev, warned), "",
        f"Baseline (pgvector's best single setting): {p1['baseline'] or 'none qualifies'}. "
        f"Rule by estimated share (fit on the training half): {rule or 'none (no training cells)'}.", "",
        "All cells; room clause: baseline total >= 2.0x ceiling total.", "",
        "| band | cells | out of box ms (recall) | baseline ms (recall) | per-band best setting ms (recall) "
        "| ceiling ms (recall) | room | ok |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    
    for b in p1["bands"]:
        best = "-" if b["best_setting"] is None else f"{b['best_setting']} {_mr(b['best_ms'], b['best_recall'])}"
        cols = [
            b["band"], b["n_cells"], _mr(b["outofbox_ms"], b["outofbox_recall"]),
            _mr(b["baseline_ms"], b["baseline_recall"]), best, _mr(b["ceiling_ms"], b["ceiling_recall"]),
            _f(b["room"], ".2f", "x"), _yes(b["room_ok"])
        ]
        
        lines.append("| " + " | ".join(map(str, cols)) + " |")
    lines += [
        "", "Test half; rule clause: rule recall >= 0.95 and it keeps >= 50% of the baseline-to-ceiling saving.",
        "", "| band | test cells | baseline ms (recall) | rule ms (recall) | ceiling ms (recall) | keep | ok |",
        "|---|---:|---:|---:|---:|---:|---|"
    ]
    
    for b in p1["bands"]:
        cols = [
            b["band"], b["n_test"], _mr(b["test_baseline_ms"], b["test_baseline_recall"]),
            _mr(b["rule_ms"], b["rule_recall"]), _mr(b["test_ceiling_ms"], b["test_ceiling_recall"]),
            _f(b["rule_keep"], ".0%"), _yes(b["rule_ok"])
        ]
        lines.append("| " + " | ".join(map(str, cols)) + " |")
    for b in p1["bands"]:
        if b["timeouts_behind"]:
            lines += ["", f"Timed-out rows behind band {b['band']}'s totals: {b['timeouts_behind']} (a timeout counts as {vc.TIMEOUT_MS:,.0f} ms and recall 0)."]
            
    lines += ["", "The ceiling's most common pick by filter family:", "", "| family | cells | option | share |", "|---|---:|---|---:|"]
    for fam, rows in p1["cells"].groupby("family", sort=False):
        counts = rows["ceiling_option"].value_counts()
        option, count = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        lines.append(f"| {fam} | {len(rows)} | {option} | {count / len(rows):.0%} |")
    return lines + [""]


# Planner choice per setting: the share of cells whose plan was each scan.
# A scan outside the four known ones counts as other.
def planner_lines(grid: pd.DataFrame) -> list[str]:
    heads = (*SCANS, "other")
    lines = ["Planner choice per setting (share of cells by scan):", "", "| setting | " + " | ".join(heads) + " |", "|---|" + "---:|" * len(heads)]
    for o in SETTINGS:
        scans = grid.loc[grid["option"] == o, "scan"]
        if len(scans):
            shares = [(scans == s).mean() for s in SCANS] + [(~scans.isin(SCANS)).mean()]
            lines.append(f"| {o} | " + " | ".join(f"{v:.0%}" for v in shares) + " |")
    return lines + [""]


# One method's row in part 2's table.
def _method_line(target: float, m: str, r: dict | None, q: dict | None) -> str:
    if r is None:
        return f"| {target:g} | {m} | missing | | | | | | | |"
    if m == "learned":
        param = f"margin {r['param']} ({'ok' if r['margin_ok'] == 'true' else 'none reached the target'})"
    else:
        param = f"{'ef' if m == 'rem' else 'E'} {r['param']}"
        
    works = " | ".join(_f(r["work"][f]) for f in CHARGE_FACTORS)
    q_recall = _f(None if q is None else q["recall"], ".4f")
    return (f"| {target:g} | {m} | {param} | {r['recall']:.4f} | {works} | {r['calls']:.1f} | {r['below']:.1%} | {q_recall} |")


# Part 2's report lines for one size.
def part2_lines(p2: dict, d: dict, dev: bool, warned: bool = False) -> list[str]:
    lines = [
        "### Part 2: learned stopping", "", _reading(p2["verdict"], dev, warned), "",
        "Work = distance computations + predictor calls x trees / 25 (x0, x1, x4).", "",
        "| target | method | param | test recall | work 0x | work 1x | work 4x | calls | below target "
        "| question recall |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"
    ]
    
    for t in p2["targets"]:
        lines += [_method_line(t["target"], m, t["test"][m], t["questions"][m]) for m in METHODS]
    lines += ["", "| target | learned / REM 0x | 1x | 4x | learned / plain 0x | 1x | 4x | recall ok | work ok " "| questions ok |", "|---|---:|---:|---:|---:|---:|---:|---|---|---|"]
    
    for t in p2["targets"]:
        ratios = " | ".join(_f(t[k][f], ".3f") for k in ("vs_rem", "vs_plain") for f in CHARGE_FACTORS)
        clauses = " | ".join(_yes(t[k]) for k in ("recall_ok", "work_ok", "questions_ok"))
        lines.append(f"| {t['target']:g} | {ratios} | {clauses} |")
    sq = d.get("queries")
    if sq is not None:
        lines += ["", "Per-query work on the test reviews (ndis + calls x 4):", "", "| target | method | p10 | median | p90 |", "|---|---|---:|---:|---:|"]
        test = sq[sq["set"] == "test"]
        for (t, m), rows in test.groupby(["target", "method"]):
            p = np.percentile(work(rows["ndis"], rows["calls"]), [10, 50, 90])
            lines.append(f"| {t:g} | {m} | {p[0]:,.0f} | {p[1]:,.0f} | {p[2]:,.0f} |")
            
    info = d.get("targets_info", {})
    if "0.95" in info and "0.99" in info and info["0.95"].get("dists_t") == info["0.99"].get("dists_t"):
        lines += ["", "At k = 10, targets 0.95 and 0.99 both need 10 of 10 per query, so they share " "dists_t / ipi / mpi; the learned rows differ only through the margin."]
    return lines + [""]


# One grid run as text, like "N = 35, kept 380, measured 0 cells, complete".
def _run_text(r: dict) -> str:
    parts = [f"N = {r.get('n_per_group')}"]
    if r.get("kept") or r.get("dropped"):
        parts += [f"kept {r.get('kept')}", f"dropped {r.get('dropped')}"]
        
    parts.append(f"measured {r.get('measured')} cells")
    return ", ".join(parts + (["complete"] if r.get("complete") is True else []))


# One size's section: its facts, then part 1 and part 2.
# warned marks a section whose readings can't be a verdict.
def size_section(d: dict, warned: bool = False) -> list[str]:
    dev = d["size"] != GATE_SIZE
    note = " (development, not a verdict)" if dev else " (no verdict: see the warnings)" if warned else ""
    limits = d["limits"]
    shown = ", ".join(f"{k} {v}" for k, v in limits.items()) if isinstance(limits, dict) and limits else limits or "none"
    lines = [
        f"## {d['size'].upper()}{note}", "",
        f"N = {d['n']} queries per group, {d['n_cells']} cells, {d['dropped']} grid rows dropped "
        f"(outside the current cell set), {d['timeouts']} timed-out rows. Postgres {d['server_version']}, "
        f"pgvector {d['extversion']}. Part 2 limits: {shown}.", ""
    ]
    if d.get("history"):
        lines += [f"Grid runs: {'; '.join(_run_text(r) for r in d['history'])}.", ""]
    if d["part1"]:
        lines += part1_lines(d["part1"], dev, warned) + planner_lines(d["grid"])
    else:
        lines += ["Part 1: not computed (see the warnings).", ""]
    if d["part2"]:
        lines += part2_lines(d["part2"], d, dev, warned)
    else:
        lines += ["Part 2: not computed (see the warnings).", ""]
    return lines


# The constants table at the end of the report.
def constants_lines() -> list[str]:
    names = (
        "BANDS", "RECALL_BAR", "ROOM_BAR", "KEEP_SHARE", "RULE_EDGES", "TARGETS", "WORK_BAR", "RECALL_TOL",
        "QUESTION_TOL", "E_RECALL", "E_CHOICES", "GATE_SIZE", "MIN_EXT", "MIN_GATE_QUERIES", "EXACT_OPTIONS",
        "ITERATIVE_COUNT", "MEMCHECK_CELLS", "DETERMINISM_QUERIES", "WARMUP", "RUNS", "GATE_TEST_QUERIES",
        "GATE_QUESTIONS", "GATE_TRAIN_QUERIES", "TREES", "LEARNING_RATE", "CHARGE_DIVISOR", "CHARGE_FACTORS", "EPS"
    )
    rows = [f"| {n} | {globals()[n]} |" for n in names]
    return ["## Constants", "", "| constant | value |", "|---|---|"] + rows + [""]


# Part 1 chart: total time per band, all cells left, the test half right.
def part1_chart(path: Path, p1: dict, size: str) -> None:
    names = [b["band"] for b in p1["bands"]]
    x = np.arange(len(names))
    panels = (
        ("All cells", (
            ("out of box", "outofbox_ms", "outofbox"), ("baseline", "baseline_ms", "baseline"),
            ("per-band best", "best_ms", "best"), ("ceiling", "ceiling_ms", "ceiling")
        )),
        ("Test half", (
            ("baseline", "test_baseline_ms", "baseline"), ("rule", "rule_ms", "rule"),
            ("ceiling", "test_ceiling_ms", "ceiling")
        )),
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (title, series) in zip(axes, panels):
        w = 0.8 / len(series)
        values = [[np.nan if b[key] is None else b[key] for b in p1["bands"]] for _, key, _ in series]
        
        for i, ((label, _, color), vals) in enumerate(zip(series, values)):
            ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w * 0.9, label=label, color=COLORS[color])
            
        if np.nanmax(np.array(values, dtype=float), initial=0) > 0:
            ax.set_yscale("log")
        ax.set_xticks(x, names)
        ax.set_title(title)
        ax.set_xlabel("band of true passing share")
        ax.set_ylabel("total ms")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(frameon=False)
        
    fig.suptitle(f"Part 1 at {size}: total time per band")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# Part 2 chart: REM's work-versus-recall curve over ef, learned and plain per target.
def part2_chart(path: Path, p2: dict, rem: pd.DataFrame | None, size: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if rem is not None and (rem["set"] == "test").any():
        curve = rem[rem["set"] == "test"].groupby("ef")[["recall", "ndis"]].mean()
        ax.plot(curve["recall"], curve["ndis"], "o-", lw=2, ms=4, color=COLORS["outofbox"], label="REM (fixed ef)")
        
    for m, marker, color in (("learned", "s", COLORS["baseline"]), ("plain", "^", COLORS["ceiling"])):
        pts = [(t["test"][m]["recall"], t["test"][m]["work"][1]) for t in p2["targets"] if t["test"][m]]
        if pts:
            ax.plot(*zip(*pts), marker, ms=8, color=color, label=f"{m} (per target)")
            
    for t in TARGETS:
        ax.axvline(t, color="0.6", lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("mean recall@10 (test reviews)")
    ax.set_ylabel("mean work (distance computations)")
    ax.set_title(f"Part 2 at {size}: work versus recall")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# argparse type for --failed-step: SIZE:STEP, with a known size and any step name.
def _failed_step(text: str) -> tuple[str, str]:
    size, _, step = text.partition(":")
    if size not in vc.SIZES or not step:
        raise argparse.ArgumentTypeError(f"wants SIZE:STEP with SIZE one of {', '.join(vc.SIZES)}, got {text!r}")
    return size, step


# Both headlines: NO VERDICT on any warning, PENDING 500K without 500K, else 500K's readings.
def headlines(sizes: dict, warnings: list) -> tuple[str, str]:
    if warnings:
        return "NO VERDICT", "NO VERDICT"
    
    gate = sizes.get(GATE_SIZE)
    if gate is None:
        return "PENDING 500K", "PENDING 500K"
    return (gate["part1"] or {}).get("verdict", "NO VERDICT"), (gate["part2"] or {}).get("verdict", "NO VERDICT")


# Read every present size, apply both bars, write the report, cells, stopping and charts.
# Returns 1 after writing everything when any warning exists.
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply the vector search check's bars and write report.md.")
    parser.add_argument("--results", type=Path, default=vc.RESULTS)
    parser.add_argument("--failed-step", type=_failed_step, default=None, metavar="SIZE:STEP", help="a step that failed before this run; gives NO VERDICT")
    args = parser.parse_args(argv)
    results = args.results
    dirs = {s: results / f"size{s}" for s in vc.SIZES}
    present = [s for s in vc.SIZES if is_present(dirs[s])]
    exts = {s: _extversion(dirs[s]) for s in present}

    sizes = {}
    for s in reversed(present):  # 500K first in the report
        d = load_size(dirs[s], s, next((exts[o] for o in present if o != s), None))
        d["part1"] = part1(d["grid"]) if d["grid"] is not None and len(d["grid"]) else None
        d["part2"] = part2(d["stopping"]) if d["stopping"] is not None else None
        sizes[s] = d

    if GATE_SIZE in sizes and "70k" not in present:
        sizes[GATE_SIZE]["warnings"].append(f"{GATE_SIZE}: no 70K run, so the pgvector version can't be compared across sizes")
    failed = []
    if args.failed_step:
        size, step = args.failed_step
        message = f"{size}: the {step} step failed, so the steps after it didn't run; their files are missing or from an older run"
        (sizes[size]["warnings"] if size in sizes else failed).append(message)
    warnings = [w for d in sizes.values() for w in d["warnings"]] + failed
    h1, h2 = headlines(sizes, warnings)

    lines = ["# Vector-search check", ""] + [x for w in warnings for x in (f"**WARNING** {w}", "")]
    lines += [f"**Part 1: {h1}**", "", f"**Part 2: {h2}**", "", "Verdicts come only from the 500K run; the 70K run is development, not a verdict.", ""]
    if not sizes:
        lines += ["No size folder has results yet.", ""]
        
    for s, d in sizes.items():
        # Any warning voids the 500K verdict; 70K's reading only follows its own warnings.
        lines += size_section(d, bool(warnings) if s == GATE_SIZE else bool(d["warnings"]))
        
    results.mkdir(parents=True, exist_ok=True)
    cells = [d["part1"]["cells"].assign(size=s) for s, d in sizes.items() if d["part1"]]
    columns = ["size", *CELL_COLUMNS, *(f"{c}_{k}" for c in CHOICES for k in ("option", "ms", "recall"))]
    frame = pd.concat(cells, ignore_index=True) if cells else pd.DataFrame()
    frame.reindex(columns=columns).to_csv(results / "cells.csv", index=False)
    stops = [d["stopping_raw"].assign(size=s) for s, d in sizes.items() if d["stopping_raw"] is not None]
    stop = pd.concat(stops, ignore_index=True) if stops else pd.DataFrame(columns=["size"])
    stop[["size", *(c for c in stop.columns if c != "size")]].to_csv(results / "stopping.csv", index=False)
    for s, d in sizes.items():
        if d["part1"]:
            part1_chart(results / f"part1_{s}.png", d["part1"], s)
        if d["part2"]:
            part2_chart(results / f"part2_{s}.png", d["part2"], d["rem"], s)

    # The report goes last, so a crash above leaves none and can't pass for warnings.
    (results / "report.md").write_text("\n".join(lines + constants_lines()))
    return 1 if warnings else 0


if __name__ == "__main__":
    sys.exit(main())
