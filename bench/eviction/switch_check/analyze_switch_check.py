from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

_rb_spec = importlib.util.spec_from_file_location("run_benchmarks_for_switch", REPO_ROOT / "bench" / "tpch" / "run_benchmarks.py")
assert _rb_spec and _rb_spec.loader
_rb = importlib.util.module_from_spec(_rb_spec)
sys.modules["run_benchmarks_for_switch"] = _rb
_rb_spec.loader.exec_module(_rb)
pool_for_fraction = _rb.pool_for_fraction

# The bar, fixed before any result was seen.
ROOM_BAR = 0.05
KEEP_SHARE = 0.5
HARM_LIMIT = 1.01
DECISION_FRACTION = 0.25
HARM_FRACTIONS = (0.10, 0.50)
REPORT_FRACTIONS = (0.10, 0.25, 0.50)
BAR_FIXED_ON = "2026-11-10"
K_ORDER = (16, 8, 32, 4, 64)
FAMILIES = ("two_window", "hedge", "fixed_share")
CANDIDATES = ("lru", "clock", "lru2_visit", "arc_visit", "lirs_visit")
PHASES_TRACES = ("phases_s1", "phases_s2", "phases_s3")
KEY = ["switcher", "policy", "pool", "window_k", "eta", "alpha"]
SENTINEL = -1.0
EPS = 1e-12
SWITCHER_KEY = ["trace", "switcher", "policy", "pool", "window_k", "eta", "alpha"]


class ChosenError(Exception):
    pass


def sf_dir_name(sf: str) -> str:
    return f"sf{sf}"


def pools_for(sf: str) -> dict:
    return {f: pool_for_fraction(sf, f) for f in REPORT_FRACTIONS}


# Fills the empty key fields with a sentinel so grouping and matching are plain.
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["policy"] = df["policy"].fillna("").astype(str)
    for col in ("window_k", "eta", "alpha"):
        df[col] = df[col].astype(float).fillna(SENTINEL)
    return df


def missing_traces(sf_dir: Path) -> list:
    return [t for t in PHASES_TRACES if not (sf_dir / f"{t}_results.csv").exists()]


# Reads every phases trace's results; raises ValueError if one is missing.
def read_results(sf_dir: Path, allow_missing: bool = False) -> pd.DataFrame:
    missing = missing_traces(sf_dir)
    if missing and not allow_missing:
        raise ValueError(f"missing phases trace results in {sf_dir}: {', '.join(missing)}")
    
    frames = [pd.read_csv(sf_dir / f"{t}_results.csv") for t in PHASES_TRACES if t not in missing]
    if not frames:
        return normalize(pd.DataFrame(columns=["trace", "switcher", "policy", "pool", "window_k", "window", "eta", "alpha", "margin", "threshold", "misses", "switches"]))
    return normalize(pd.concat(frames, ignore_index=True))


def read_aux(sf_dir: Path, suffix: str) -> pd.DataFrame:
    frames = [pd.read_csv(sf_dir / f"{t}_{suffix}.csv") for t in PHASES_TRACES if (sf_dir / f"{t}_{suffix}.csv").exists()]
    if not frames:
        return pd.DataFrame()
    return normalize(pd.concat(frames, ignore_index=True))


# Misses and switches summed over the traces per setting, with the trace count.
def summed(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["switches"] = d["switches"].fillna(0)
    g = d.groupby(KEY, sort=False)

    out = g.agg(
        misses=("misses", "sum"), switches=("switches", "sum"), n=("trace", "nunique"),
        window=("window", "first"), margin=("margin", "first"),
        threshold=("threshold", "first")
    ).reset_index()
    
    out["misses"] = out["misses"].astype(int)
    out["switches"] = out["switches"].astype(int)
    return out


def num_match(col: pd.Series, value: Optional[float]) -> pd.Series:
    target = SENTINEL if value is None else float(value)
    return col.apply(lambda v: math.isclose(v, target, rel_tol=1e-9, abs_tol=1e-12))


# Rows of a summed (or raw) frame matching a setting; None means the empty field.
def select(s: pd.DataFrame, switcher: str, pool: int, window_k=None, eta=None, alpha=None, policy: str = "") -> pd.DataFrame:
    m = (s["switcher"] == switcher) & (s["pool"] == pool) & (s["policy"] == policy)
    m &= num_match(s["window_k"], window_k) & num_match(s["eta"], eta) & num_match(s["alpha"], alpha)
    return s[m]


def lookup(s: pd.DataFrame, switcher: str, pool: int, window_k=None, eta=None, alpha=None, policy: str = ""):
    rows = select(s, switcher, pool, window_k, eta, alpha, policy)
    return None if rows.empty else rows.iloc[0]


def best_fixed(s: pd.DataFrame, pool: int) -> tuple:
    best = None
    for c in CANDIDATES:
        row = lookup(s, "fixed", pool, policy=c)
        if row is not None and (best is None or int(row["misses"]) < best[1]):
            best = (c, int(row["misses"]))
            
    if best is None:
        raise KeyError(f"no fixed rows at pool {pool}")
    return best


def belady_misses(s: pd.DataFrame, pool: int) -> Optional[int]:
    row = lookup(s, "belady", pool, policy="belady")
    return None if row is None else int(row["misses"])


def saving(misses: float, best: float) -> float:
    return 1.0 - misses / best


def gap_share(misses: float, best: float, belady: Optional[float]) -> Optional[float]:
    if belady is None or best - belady <= 0:
        return None
    return (best - misses) / (best - belady)


def at_least(value: float, bar: float) -> bool:
    return value >= bar - EPS


def none_if_sentinel(v: float):
    return None if v == SENTINEL else float(v)


def k_rank(k: int) -> int:
    return K_ORDER.index(k) if k in K_ORDER else len(K_ORDER)


# Per family, the best setting at one pool: misses, switches, k order, eta, alpha.
def pick_settings(s: pd.DataFrame, pool: int) -> dict:
    picks = {}
    for fam in FAMILIES:
        rows = s[(s["switcher"] == fam) & (s["pool"] == pool) & (s["n"] == len(PHASES_TRACES))]
        if rows.empty:
            continue
        ranked = sorted(rows.itertuples(index=False), key=lambda r: (r.misses, r.switches, k_rank(int(r.window_k)), r.eta, r.alpha))
        r = ranked[0]
        picks[fam] = dict(
            window_k=int(r.window_k), eta=none_if_sentinel(r.eta), alpha=none_if_sentinel(r.alpha),
            margin=None if pd.isna(r.margin) else float(r.margin),
            threshold=None if pd.isna(r.threshold) else float(r.threshold),
            misses=int(r.misses), switches=int(r.switches)
        )
    return picks


def write_chosen(path: Path, picks: dict) -> None:
    doc = {
        "sf": "0.01", 
        "bar_fixed_on": BAR_FIXED_ON, 
        "picked_on": date.today().isoformat(), 
        "families": picks
    }
    Path(path).write_text(json.dumps(doc, indent=2) + "\n")


# Fields each family's frozen setting must hold as numbers.
NEEDED_FIELDS = {
    "two_window": ("window_k", "threshold"), 
    "hedge": ("window_k", "eta", "alpha", "margin"),
    "fixed_share": ("window_k", "eta", "alpha", "margin")
}


def is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def read_chosen(path: Path) -> dict:
    path = Path(path)
    try:
        doc = json.loads(path.read_text())
    except OSError as e:
        raise ChosenError(f"cannot read {path}: {e}") from e
    except ValueError as e:
        raise ChosenError(f"{path} is not valid JSON: {e}") from e
    
    if not isinstance(doc, dict) or doc.get("sf") != "0.01":
        raise ChosenError(f"{path} was not picked on SF=0.01")
    fams = doc.get("families")
    if not isinstance(fams, dict):
        raise ChosenError(f"{path} has no families")
    
    for fam in FAMILIES:
        if not isinstance(fams.get(fam), dict) or "window_k" not in fams[fam]:
            raise ChosenError(f"{path} is missing the {fam} setting")
        
        bad = [f for f in NEEDED_FIELDS[fam] if not is_number(fams[fam].get(f))]
        if bad:
            raise ChosenError(f"{path}: the {fam} setting has no usable {', '.join(bad)}")
    return {fam: fams[fam] for fam in FAMILIES}


def family_row(s: pd.DataFrame, fam: str, pick: dict, pool: int):
    return lookup(s, fam, pool, pick["window_k"], pick.get("eta"), pick.get("alpha"))


# Three clauses per family at its chosen setting, plus go and handed-over family.
def verdict(s: pd.DataFrame, picks: dict, pools: dict) -> dict:
    p25 = pools[DECISION_FRACTION]
    best25 = best_fixed(s, p25)[1]
    out = {}
    for fam in FAMILIES:
        pick = picks[fam]
        entry = dict(missing=False, window_k=pick["window_k"], room=None, keep=None, harm={}, room_pass=False, keep_pass=False, harm_pass=False, passed=False, saving=None, total_misses=None)
        hind = lookup(s, "hindsight", p25, pick["window_k"])
        row = family_row(s, fam, pick, p25)
        harm_rows = {f: family_row(s, fam, pick, pools[f]) for f in HARM_FRACTIONS}
        
        if hind is None or row is None or any(r is None for r in harm_rows.values()):
            entry["missing"] = True
            out[fam] = entry
            continue
        
        room = saving(int(hind["misses"]), best25)
        sv = saving(int(row["misses"]), best25)
        entry.update(room=room, saving=sv, keep=sv)
        entry["room_pass"] = at_least(room, ROOM_BAR)
        entry["keep_pass"] = at_least(sv, KEEP_SHARE * room) if room > 0 else False
        
        harm_ok = True
        for f, r in harm_rows.items():
            ratio = int(r["misses"]) / best_fixed(s, pools[f])[1]
            entry["harm"][f] = ratio
            harm_ok &= ratio <= HARM_LIMIT + EPS
            
        entry["harm_pass"] = harm_ok
        entry["passed"] = entry["room_pass"] and entry["keep_pass"] and harm_ok
        entry["total_misses"] = sum(int(family_row(s, fam, pick, pools[f])["misses"]) for f in REPORT_FRACTIONS if family_row(s, fam, pick, pools[f]) is not None)
        out[fam] = entry

    passing = [f for f in FAMILIES if out[f]["passed"]]
    handed = None
    if passing:
        handed = sorted(passing, key=lambda f: (-out[f]["saving"], out[f]["total_misses"]))[0]
    return {
        "families": out, 
        "go": bool(passing), 
        "handed_over": handed
    }


# Warnings for results rows whose margin or threshold differ from the chosen file.
def chosen_mismatches(s: pd.DataFrame, picks: dict, pool: int) -> list:
    msgs = []
    for fam in FAMILIES:
        row = family_row(s, fam, picks[fam], pool)
        if row is None:
            continue
        
        for field in ("margin", "threshold"):
            want = picks[fam].get(field)
            have = None if pd.isna(row[field]) else float(row[field])
            same = want is None and have is None or (want is not None and have is not None and math.isclose(have, float(want), rel_tol=1e-9))
            
            if not same:
                msgs.append(f"the {fam} rows have {field} {have}, but chosen.json says {want}")
    return msgs


# Whether two picks for one family agree on every frozen field.
def same_pick(a: dict, b: dict) -> bool:
    for field in ("window_k", "eta", "alpha", "margin", "threshold", "misses", "switches"):
        x, y = a.get(field), b.get(field)
        if x is None or y is None:
            if x is not y:
                return False
            
        elif not math.isclose(float(x), float(y), rel_tol=1e-9, abs_tol=1e-12):
            return False
    return True


# One message per validation row missing from or different to the reference passes.
def check_validation(sf_dir: Path, sim_dir: Path) -> list:
    sf_dir, sim_dir = Path(sf_dir), Path(sim_dir)
    msgs: list = []
    seen: dict = {}
    for t in PHASES_TRACES:
        path = sf_dir / f"{t}_validation.csv"
        if not path.exists():
            msgs.append(f"validation file missing: {path}")
            continue
        
        df = pd.read_csv(path)
        if df.empty:
            msgs.append(f"validation file has no rows: {path}")
            
        for r in df.itertuples(index=False):
            key = (str(r.trace), str(r.policy), int(r.pool), int(r._3))
            miss = int(r.misses)
            if key in seen and seen[key] != miss:
                msgs.append(f"validation rows disagree for {key}: {seen[key]} vs {miss}")
                
            seen.setdefault(key, miss)
    refs: dict = {}
    for comp in sorted({k[0] for k in seen}):
        ref_path = sim_dir / sf_dir.name / f"{comp}_passes.csv"
        if not ref_path.exists():
            msgs.append(f"reference file missing: {ref_path}")
            continue
        ref = pd.read_csv(ref_path)
        refs[comp] = {(str(r.trace), str(r.policy), int(r.pool), int(r._3)): int(r.misses) for r in ref.itertuples(index=False)}
        
    for key, miss in sorted(seen.items()):
        table = refs.get(key[0])
        if table is None:
            continue
        if key not in table:
            msgs.append(f"{key} is not in the reference passes")
        elif table[key] != miss:
            msgs.append(f"{key}: replay {miss} misses, reference {table[key]}")
    return msgs


def windows_problems(df: pd.DataFrame, sf_dir: Path) -> list:
    path = sf_dir / "windows.csv"
    if not path.exists():
        return [f"{path} is missing"]
    
    w = pd.read_csv(path)
    known = {(int(a), int(b)) for a, b in zip(w["window_k"], w["window"])}
    used = df.dropna(subset=["window"])
    pairs = {(int(a), int(b)) for a, b in zip(used["window_k"], used["window"])}
    return [f"window_k {a} with window {b} is not in windows.csv" for a, b in sorted(pairs - known)]


def fmt_pct(v: Optional[float]) -> str:
    return "-" if v is None else f"{100 * v:.2f}%"


def ratio_text(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.4f}"


def setting_text(fam: str, pick: dict) -> str:
    parts = [f"k={pick['window_k']}"]
    if pick.get("eta") is not None:
        parts.append(f"eta={pick['eta']:g}")
    if pick.get("alpha") is not None and fam != "two_window":
        parts.append(f"alpha={pick['alpha']:g}")
    return ", ".join(parts)


def result_line(label: str, row, best: int, bel: Optional[int]) -> str:
    sw = "" if row is None or row["switches"] == 0 and row["switcher"] in ("fixed", "belady", "free_bound") else int(row["switches"])
    return (f"| {label} | {int(row['misses']):,} | {fmt_pct(saving(int(row['misses']), best))} | {ratio_text(gap_share(int(row['misses']), best, bel))} | {sw} |")


def pool_section(s: pd.DataFrame, frac: float, pool: int, settings: dict, grid: bool) -> list:
    lines = [f"### Pool {int(frac * 100)}% ({pool:,} frames)", "", "| policy | misses | saving | gap share | switches |", "|---|---:|---:|---:|---:|"]
    best_name, best = best_fixed(s, pool)
    bel = belady_misses(s, pool)
    for c in CANDIDATES:
        row = lookup(s, "fixed", pool, policy=c)
        if row is not None:
            lines.append(result_line(f"{c}{' (best fixed)' if c == best_name else ''}", row, best, bel))
            
    brow = lookup(s, "belady", pool, policy="belady")
    if brow is not None:
        lines.append(result_line("belady (reported only)", brow, best, bel))
        
    ks = sorted({int(k) for k in s[s["switcher"].isin(("hindsight", "free_bound"))]["window_k"]}, key=k_rank)
    for k in ks:
        for sw, label in (("free_bound", "free bound"), ("hindsight", "hindsight")):
            row = lookup(s, sw, pool, k)
            if row is not None:
                lines.append(result_line(f"k={k} {label}", row, best, bel))
                
        for fam in FAMILIES:
            pick = settings.get(fam)
            if pick is None:
                continue
            
            row = lookup(s, fam, pool, k, pick.get("eta"), pick.get("alpha"))
            if row is not None:
                lines.append(result_line(f"k={k} {fam} ({setting_text(fam, dict(pick, window_k=k))})", row, best, bel))
    lines.append("")
    if grid:
        lines += [f"Every grid setting at {int(frac * 100)}%:", "", "| family | setting | misses | saving | switches |", "|---|---|---:|---:|---:|"]
        rows = s[(s["pool"] == pool) & s["switcher"].isin(FAMILIES)]
        for r in sorted(rows.itertuples(index=False), key=lambda r: (FAMILIES.index(r.switcher), r.misses)):
            pk = dict(window_k=int(r.window_k), eta=none_if_sentinel(r.eta), alpha=none_if_sentinel(r.alpha))
            lines.append(f"| {r.switcher} | {setting_text(r.switcher, pk)} | {r.misses:,} | {fmt_pct(saving(r.misses, best))} | {r.switches} |")
            
        lines.append("")
    return lines


def phase_sections(sf_dir: Path, settings: dict, pool: int) -> list:
    ph = read_aux(sf_dir, "phases")
    occ = read_aux(sf_dir, "occupancy")
    if ph.empty:
        return ["## Per-phase misses", "", "No phase files.", ""]
    
    keys = [("hindsight", k, None, None) for k in sorted({p["window_k"] for p in settings.values()}, key=k_rank)]
    keys += [(f, p["window_k"], p.get("eta"), p.get("alpha")) for f, p in settings.items()]
    
    ph = ph[ph["pool"] == pool]
    psum = ph.groupby(KEY + ["phase"], sort=False)["misses"].sum().reset_index()
    phases = sorted(psum["phase"].unique())
    
    fixed_phase = {}
    for ph_i in phases:
        best = None
        for c in CANDIDATES:
            rows = psum[(psum["switcher"] == "fixed") & (psum["policy"] == c) & (psum["phase"] == ph_i)]
            if not rows.empty and (best is None or int(rows["misses"].iloc[0]) < best[1]):
                best = (c, int(rows["misses"].iloc[0]))
                
        fixed_phase[ph_i] = best
    lines = ["## Per-phase misses", "", f"Pool {int(DECISION_FRACTION * 100)}%, summed over the traces.", "", "| setting | " + " | ".join(f"phase {p}" for p in phases) + " |", "|---|" + "---:|" * len(phases)]
    lines.append("| best fixed per phase | " + " | ".join(f"{fixed_phase[p][1]:,} ({fixed_phase[p][0]})" if fixed_phase[p] else "-" for p in phases) + " |")
    
    for sw, k, eta, alpha in keys:
        m = (psum["switcher"] == sw) & num_match(psum["window_k"], k) & num_match(psum["eta"], eta) & num_match(psum["alpha"], alpha)
        sub = psum[m]
        cells = []
        for p in phases:
            r = sub[sub["phase"] == p]
            cells.append(f"{int(r['misses'].iloc[0]):,}" if not r.empty else "-")
            
        lines.append(f"| {sw} k={k}{'' if eta is None else f', eta={eta:g}'}{'' if not alpha else f', alpha={alpha:g}'} | " + " | ".join(cells) + " |")
    lines += [
        "", "## Time spent on each phase's best fixed policy", "",
        "Share of the phase's runs, summed over the traces.", "",
        "| setting | " + " | ".join(f"phase {p}" for p in phases) + " |", "|---|" + "---:|" * len(phases)
    ]
    occ = occ[occ["pool"] == pool] if not occ.empty else occ
    for sw, k, eta, alpha in keys:
        cells = []
        for p in phases:
            if occ.empty or fixed_phase[p] is None:
                cells.append("-")
                continue
            
            m = (occ["switcher"] == sw) & num_match(occ["window_k"], k) & num_match(occ["eta"], eta) & num_match(occ["alpha"], alpha) & (occ["phase"] == p)
            sub = occ[m]
            total = int(sub["runs"].sum())
            on = int(sub[sub["candidate"] == fixed_phase[p][0]]["runs"].sum())
            cells.append(f"{100 * on / total:.1f}%" if total else "-")
            
        lines.append(f"| {sw} k={k}{'' if eta is None else f', eta={eta:g}'}{'' if not alpha else f', alpha={alpha:g}'} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def verdict_lines(v: dict, picks: dict, pools: dict, dev: bool, warned: bool = False) -> list:
    head = "GO" if v["go"] else "NO-GO"
    lines = []
    if dev:
        lines += [
            "## Verdict: development run, not a decision", "",
            f"The clauses below are informative only ({head} on the picks at SF=0.01)"
            f"{' (not valid: see warnings)' if warned else ''}.", ""
        ]
    else:
        lines += [f"**{head}**" + (" (not valid: see warnings)" if warned else ""), ""]
    lines += [
        f"Bar fixed on {BAR_FIXED_ON}: room >= {ROOM_BAR:.0%}, keep >= {KEEP_SHARE:.0%} of the room, "
        f"harm <= {HARM_LIMIT}x best fixed at 10% and 50%. Misses summed over the three traces.", "",
        "| family | setting | room (hindsight) | keep (saving) | harm 10% | harm 50% | passes |",
        "|---|---|---|---|---|---|---|"
    ]
    
    for fam in FAMILIES:
        e = v["families"][fam]
        if e["missing"]:
            lines.append(f"| {fam} | {setting_text(fam, picks[fam])} | no rows for this setting | | | | no |")
            continue
        
        h10, h50 = (e["harm"][f] for f in HARM_FRACTIONS)
        lines.append(
            f"| {fam} | {setting_text(fam, picks[fam])} | {fmt_pct(e['room'])} {'ok' if e['room_pass'] else 'FAIL'} | "
            f"{fmt_pct(e['keep'])} {'ok' if e['keep_pass'] else 'FAIL'} | "
            f"{h10:.4f} {'ok' if h10 <= HARM_LIMIT + EPS else 'FAIL'} | "
            f"{h50:.4f} {'ok' if h50 <= HARM_LIMIT + EPS else 'FAIL'} | {'yes' if e['passed'] else 'no'} |"
        )
    handed = v["handed_over"]
    lines += ["", f"Handed over: {handed + ' (' + setting_text(handed, picks[handed]) + ')' if handed else 'none'}", ""]
    return lines


# A small table of each window share k and its W in runs, from windows.csv.
def window_lines(sf_dir: Path) -> list:
    path = sf_dir / "windows.csv"
    if not path.exists():
        return []
    
    w = pd.read_csv(path)
    lines = ["## Windows", "", "| k | W (runs) |", "|---|---:|"]
    for k, size in sorted(zip(w["window_k"], w["window"]), key=lambda kv: k_rank(int(kv[0]))):
        lines.append(f"| {int(k)} | {int(size):,} |")

    return lines + [""]


def write_chart(path: Path, s: pd.DataFrame, settings: dict, pool: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    best = best_fixed(s, pool)[1]
    ks = sorted({int(k) for k in s[s["switcher"] == "hindsight"]["window_k"]}, reverse=True)
    series = [("hindsight", "hindsight", None, None), ("free_bound", "free bound", None, None)]
    series += [(f, f, p.get("eta"), p.get("alpha")) for f, p in settings.items()]
    for sw, label, eta, alpha in series:
        xs, ys = [], []
        for k in ks:
            row = lookup(s, sw, pool, k, eta, alpha)
            if row is not None:
                xs.append(str(k))
                ys.append(100 * saving(int(row["misses"]), best))
        if xs:
            ax.plot(xs, ys, marker="o", label=label)
            
    ax.axhline(100 * ROOM_BAR, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("window share k")
    ax.set_ylabel("saving vs best fixed, 25% pool (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_cells(path: Path, s: pd.DataFrame, pools: dict) -> None:
    frac_of = {p: f for f, p in pools.items()}
    rows = []
    for r in s.itertuples(index=False):
        try:
            best = best_fixed(s, r.pool)[1]
        except KeyError:
            continue
        bel = belady_misses(s, r.pool)
        
        rows.append(dict(
            switcher=r.switcher, policy=r.policy, pool=r.pool, pool_fraction=frac_of.get(r.pool),
            window_k=none_if_sentinel(r.window_k),
            window=None if pd.isna(r.window) else int(r.window), eta=none_if_sentinel(r.eta),
            alpha=none_if_sentinel(r.alpha), misses=r.misses, switches=r.switches, traces=r.n,
            saving=saving(r.misses, best), gap_share=gap_share(r.misses, best, bel)
        ))
    pd.DataFrame(rows).to_csv(path, index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sf", required=True, choices=("0.01", "0.1"))
    ap.add_argument("--out", default=str(REPO_ROOT / "bench" / "results" / "switch_check"))
    ap.add_argument("--sim-dir", default=str(REPO_ROOT / "bench" / "results" / "headroom" / "eviction"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    sf_dir = out / sf_dir_name(args.sf)
    sim_dir = Path(args.sim_dir)
    dev = args.sf == "0.01"
    pools = pools_for(args.sf)
    p25 = pools[DECISION_FRACTION]

    warnings = [f"missing phases trace: {t}" for t in missing_traces(sf_dir)]
    df = read_results(sf_dir, allow_missing=True)
    warnings += windows_problems(df, sf_dir)
    val_msgs = check_validation(sf_dir, sim_dir)
    warnings += val_msgs
    s = summed(df) if len(df) else df
    
    if len(df):
        short = s[s["n"] < len(PHASES_TRACES)]
        if not short.empty and not missing_traces(sf_dir):
            warnings.append(f"{len(short)} settings appear in fewer than {len(PHASES_TRACES)} traces")
        elif not short.empty:
            warnings.append(f"{len(short)} settings are not summed over all {len(PHASES_TRACES)} traces")

    picks: dict = {}
    no_verdict = False
    if len(df):
        try:
            if dev:
                picks = pick_settings(s, p25)
                lost = [f for f in FAMILIES if f not in picks]
                if lost:
                    warnings.append(f"no rows to pick a setting for: {', '.join(lost)}")
            else:
                picks = read_chosen(out / "chosen.json")
                mismatched = chosen_mismatches(s, picks, p25)
                if mismatched:
                    no_verdict = True
                    warnings += mismatched
        except ChosenError as e:
            no_verdict = True
            warnings.append(str(e))

    v = None
    if len(picks) == len(FAMILIES) and not no_verdict:
        try:
            v = verdict(s, picks, pools)
            for f in FAMILIES:
                if v["families"][f]["missing"]:
                    warnings.append(f"the {f} setting {setting_text(f, picks[f])} has no rows")
        except KeyError as e:
            warnings.append(f"cannot judge: {e}")

    chosen_note = None
    if dev:
        frozen_path = out / "chosen.json"
        frozen = None
        if frozen_path.exists():
            try:
                frozen = read_chosen(frozen_path)
            except ChosenError:
                frozen = None

        if warnings or len(picks) != len(FAMILIES):
            chosen_note = "chosen.json was not written: this run has warnings."
            if frozen_path.exists():
                chosen_note += " The existing chosen.json was left as it is."
        elif frozen is None:
            write_chosen(frozen_path, picks)
        elif changed := [f for f in FAMILIES if not same_pick(frozen[f], picks[f])]:
            warnings.append(f"the picks changed for {', '.join(changed)}; chosen.json was kept. Delete chosen.json by hand to pick again.")
            chosen_note = "chosen.json was not written: the new picks differ from the frozen file."

    lines: list = []
    if warnings:
        lines += ["**WARNING**", ""] + [f"- {w}" for w in warnings] + [""]
        if chosen_note:
            lines += [chosen_note, ""]
            
    if no_verdict:
        lines += ["**NO VERDICT**", ""]
    title = [f"# Switching headroom, SF={args.sf}", ""]
    if v is not None and not dev:
        lines += verdict_lines(v, picks, pools, dev, bool(warnings)) + title
    else:
        lines += title
        if v is not None:
            lines += verdict_lines(v, picks, pools, dev, bool(warnings))
        elif not no_verdict:
            lines += ["No verdict could be formed from these results.", ""]
    lines += window_lines(sf_dir)
    if len(df):
        lines += ["## Results", ""]
        for frac, pool in pools.items():
            try:
                lines += pool_section(s, frac, pool, picks, dev and frac == DECISION_FRACTION)
            except KeyError as e:
                lines += [f"Pool {int(frac * 100)}%: {e}", ""]
        lines += phase_sections(sf_dir, picks, p25) if picks else []
        
    lines += ["## Validation", "", "Replay misses against the earlier reference passes, exact on trace, policy, pool and pass.", ""]
    lines += ["All rows match."] if not val_msgs else [f"- {m}" for m in val_msgs]
    lines.append("")

    sf_dir.mkdir(parents=True, exist_ok=True)
    (sf_dir / "report.md").write_text("\n".join(lines))
    if len(df):
        write_cells(sf_dir / "cells.csv", s, pools)
        try:
            write_chart(sf_dir / f"misses_sf{args.sf}.png", s, picks, p25)
        except KeyError:
            pass
    return 1 if warnings else 0


if __name__ == "__main__":
    sys.exit(main())
