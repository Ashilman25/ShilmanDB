from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from train_v3_check import CELLS, ablation_cell, param_count  # noqa: E402

_rb_spec = importlib.util.spec_from_file_location("run_benchmarks_for_v3", REPO_ROOT / "bench" / "tpch" / "run_benchmarks.py")
assert _rb_spec and _rb_spec.loader
_rb = importlib.util.module_from_spec(_rb_spec)
sys.modules["run_benchmarks_for_v3"] = _rb
_rb_spec.loader.exec_module(_rb)
pool_for_fraction = _rb.pool_for_fraction

# The bar, fixed before any result was seen.
MISS_BAR = 0.05
COST_BAR_US = 5.00
GATE_SEED_SETS = ((1, 2, 3), (3,))
GATE_FRACTIONS = (0.10, 0.25)
REPORT_FRACTIONS = (0.10, 0.25, 0.50)
BAR_FIXED_ON = "2026-11-10"
CLASSICAL = ("arc_visit", "lirs_visit")
CHECKED = ("lru", "arc_visit", "lirs_visit", "belady")
REFERENCES = ("sampled_random", "sampled_lru", "sampled_label", "sampled_belady")
TEST_TRACES = ("index_mixed_s1", "index_mixed_s2", "index_mixed_s3", "index_stream_s3", "index_lookup_s3")
EXTRA_GROUPS = {"stream_s3": ("index_stream_s3",), "lookup_s3": ("index_lookup_s3",)}
SF1_FRACTION = 0.25
EPS = 1e-12

Items = dict  # (trace, policy, pool) -> {(pass, seq): misses}


def seed_label(seeds) -> str:
    return "+".join(str(s) for s in seeds)


def mixed_traces(seeds) -> tuple:
    return tuple(f"index_mixed_s{s}" for s in seeds)


# Reads item CSVs into {(trace, policy, pool): {(pass, seq): misses}}.
def read_items(paths) -> Items:
    out: Items = {}
    for path in paths:
        df = pd.read_csv(path)
        for (trace, policy, pool), g in df.groupby(["trace", "policy", "pool"]):
            d = out.setdefault((str(trace), str(policy), int(pool)), {})
            d.update({(int(p), int(s)): int(m) for p, s, m in zip(g["pass"], g["seq"], g["misses"])})
            
    return out


# Misses in the lowest timed pass present.
def first_pass(items_by_key: dict) -> int:
    lowest = min(p for p, _ in items_by_key)
    return sum(m for (p, _), m in items_by_key.items() if p == lowest)


def first_pass_table(items: Items) -> dict:
    return {key: first_pass(v) for key, v in items.items() if v}


# The (pass, seq) keys of the lowest timed pass present.
def first_pass_keys(items_by_key: dict) -> set:
    lowest = min(p for p, _ in items_by_key)
    return {k for k in items_by_key if k[0] == lowest}


# A learned, reference or ablation replay counts only if its first pass
# covers exactly the items arc_visit's does.
def complete_replays(items: Items) -> Items:
    out: Items = {}
    for (trace, policy, pool), v in items.items():
        if v and policy not in CHECKED:
            ref = items.get((trace, "arc_visit", pool))
            if not ref or first_pass_keys(v) != first_pass_keys(ref):
                continue
            
        out[(trace, policy, pool)] = v
    return out


def saving(learned: int, classical: int) -> float:
    return 1.0 - learned / classical if classical > 0 else 0.0


def meets_bar(s: Optional[float]) -> bool:
    return s is not None and s >= MISS_BAR - EPS


def gap_captured(learned: int, classical: int, belady: int) -> float:
    room = classical - belady
    return (classical - learned) / room if room > 0 else 0.0


def passes_cost(timing_row) -> bool:
    return timing_row is not None and timing_row["median_ns"] / 1000.0 <= COST_BAR_US + EPS


# Better of arc / lirs summed over the given traces; a tie goes to arc.
def classical_over(misses: dict, pool: int, traces) -> tuple:
    sums = {p: sum(misses[(t, p, pool)] for t in traces) for p in CLASSICAL}
    best = min(CLASSICAL, key=lambda p: (sums[p], CLASSICAL.index(p)))
    return best, sums[best]


def classical_for(misses: dict, pool: int, seeds) -> tuple:
    return classical_over(misses, pool, mixed_traces(seeds))


def group_defs() -> dict:
    groups = {seed_label(s): mixed_traces(s) for s in GATE_SEED_SETS}
    groups.update(EXTRA_GROUPS)
    return groups


# Savings per (fraction, group) for one policy.
# Complete means every replay the bar needs is present.
def gate(misses: dict, cell: str, pools: dict, required=GATE_FRACTIONS) -> dict:
    savings, rows, missing = {}, [], []
    need_groups = {seed_label(s) for s in GATE_SEED_SETS}
    
    for fraction, pool in pools.items():
        for label, traces in group_defs().items():
            keys = [(t, p, pool) for t in traces for p in (cell, *CLASSICAL, "belady")]
            gone = [k for k in keys if k not in misses]
            required_here = fraction in required and label in need_groups
            
            if gone:
                savings[(fraction, label)] = None
                if required_here:
                    named = [k for k in gone if k[1] != cell] or gone
                    missing += [f"{k[1]} {k[0]} pool {k[2]}" for k in named]
                continue
            
            learned = sum(misses[(t, cell, pool)] for t in traces)
            cpol, cmiss = classical_over(misses, pool, traces)
            belady = sum(misses[(t, "belady", pool)] for t in traces)
            s = saving(learned, cmiss)
            savings[(fraction, label)] = s
            
            rows.append(dict(
                fraction=fraction, pool=pool, seeds=label, misses=learned, classical_policy=cpol,
                classical_misses=cmiss, saving=s, belady_misses=belady,
                gap_captured=gap_captured(learned, cmiss, belady)
            ))
            
    for fraction in required:
        if fraction not in pools:
            missing.append(f"no pool for fraction {fraction}")
            
    complete = not missing
    wanted = [(f, seed_label(s)) for f in required for s in GATE_SEED_SETS]
    passes = complete and all(meets_bar(savings.get((f, lbl))) for f, lbl in wanted)
    return dict(cell=cell, savings=savings, rows=rows, complete=complete, passes_miss=passes, missing=missing)


def mean_saving(g: dict) -> Optional[float]:
    vals = [g["savings"].get((f, seed_label(GATE_SEED_SETS[0]))) for f in GATE_FRACTIONS]
    return None if any(v is None for v in vals) else sum(vals) / len(vals)


def best_cell(gates: dict, timings: dict) -> Optional[str]:
    ok = [c for c, g in gates.items() if g["passes_miss"] and passes_cost(timings.get(c))]
    return max(ok, key=lambda c: (mean_saving(gates[c]) or 0.0), default=None)


# Complete cell with the highest mean saving, whether or not it passes either bar.
def best_on_misses(gates: dict) -> Optional[str]:
    ok = [c for c, g in gates.items() if g["complete"] and mean_saving(g) is not None]
    return max(ok, key=lambda c: mean_saving(gates[c]), default=None)


# The ablation's cell: the best richer cell passing both bars,
# else the best richer cell on misses alone.
def best_richer_cell(gates: dict, timings: dict) -> Optional[str]:
    richer = [c for c, g in gates.items() if c in CELLS and CELLS[c].features == "richer" and g["complete"] and mean_saving(g) is not None]
    both = [c for c in richer if gates[c]["passes_miss"] and passes_cost(timings.get(c))]
    return max(both or richer, key=lambda c: mean_saving(gates[c]), default=None)


def verdict(sf01: dict, sf1: Optional[dict]) -> str:
    if sf01.get("best") is None:
        return "NO-GO"
    if sf1 is None:
        return "PENDING SF=1.0"
    return "GO" if sf1["holds"] else "NO-GO"


def pools_for(sf: str) -> dict:
    fractions = (SF1_FRACTION,) if sf == "1.0" else REPORT_FRACTIONS
    return {f: pool_for_fraction(sf, f) for f in fractions}


def required_for(sf: str) -> tuple:
    return (SF1_FRACTION,) if sf == "1.0" else GATE_FRACTIONS


def replay_dir(out: Path, sf: str) -> Path:
    return out / f"sf{sf}" / "replay"


def load_misses(out: Path, sf: str) -> dict:
    return first_pass_table(complete_replays(read_items(sorted(replay_dir(out, sf).glob("*_items.csv")))))


# Gate result for every cell that has replays under <out>/sf<sf>/replay/.
def load_gates(out: Path, sf: str) -> dict:
    misses = load_misses(out, sf)
    present = {k[1] for k in misses}
    return {c: gate(misses, c, pools_for(sf), required_for(sf)) for c in CELLS if c in present}


TIMING_TRACE = "index_mixed_s1"
TIMING_K = 32


def timing_pool(sf: str) -> int:
    return pool_for_fraction(sf, SF1_FRACTION if sf == "1.0" else GATE_FRACTIONS[0])


# Timing rows measured on the trace, pool and sample count the bar names.
# Reasons for the rest go to notes.
def load_timings(out: Path, sf: str, notes: Optional[list] = None) -> dict:
    timings = {}
    for name in [*CELLS, *(ablation_cell(c).name for c in CELLS if CELLS[c].features == "richer")]:
        path = out / f"sf{sf}" / "timing" / f"{name}_timing.csv"
        
        if path.exists():
            df = pd.read_csv(path)
            if not len(df):
                continue
            
            row = df.iloc[0].to_dict()
            want = (TIMING_TRACE, timing_pool(sf), TIMING_K)
            got = (str(row.get("trace")), int(row.get("pool", -1)), int(row.get("k", -1)))
            if got == want:
                timings[name] = row
            elif notes is not None:
                notes.append(f"{name} timing was measured on {got[0]} pool {got[1]} k {got[2]}, not {want[0]} pool {want[1]} k {want[2]}")
                
    return timings


# One row per (trace, policy, pool): the replay's classical items against the recorded ones.
def validate(replay: Items, reference: Items, traces, pools) -> list:
    rows = []
    for trace in traces:
        for policy in CHECKED:
            for pool in pools:
                a, b = replay.get((trace, policy, pool)), reference.get((trace, policy, pool))
                
                if not a:
                    status, bad = "missing_replay", 0
                elif not b:
                    status, bad = "missing_reference", 0
                else:
                    bad = sum(1 for k in set(a) | set(b) if a.get(k) != b.get(k))
                    status = "match" if bad == 0 else "mismatch"
                    
                rows.append(dict(
                    trace=trace, policy=policy, pool=pool, status=status,
                    items=len(a or b or {}), bad_items=bad
                ))
    return rows


def sf_folders(out: Path) -> list:
    sfs = [p.name[2:] for p in out.glob("sf*") if (p / "replay").is_dir()]
    return sorted(sfs, key=float)


def validate_sf(out: Path, sim_dir: Path, sf: str, gate_sf: str) -> list:
    classical_files = {t: replay_dir(out, sf) / f"{t}_classical_items.csv" for t in TEST_TRACES}
    traces = [t for t in TEST_TRACES if sf == gate_sf or classical_files[t].exists()]
    replay = read_items([p for p in classical_files.values() if p.exists()])
    
    ref_files = [sim_dir / f"sf{sf}" / f"{t}_items.csv" for t in traces]
    reference = read_items([p for p in ref_files if p.exists()])
    rows = validate(replay, reference, traces, pools_for(sf).values())
    return [dict(sf=sf, **r) for r in rows]


def pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.2f}%"


def us(row, key="median_ns") -> str:
    return "not timed" if row is None else f"{row[key] / 1000:.2f}"


def table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def gate_cols(g: Optional[dict]) -> list:
    if g is None:
        return ["n/a"] * 4
    return [pct(g["savings"].get((f, seed_label(s)))) for f in GATE_FRACTIONS for s in GATE_SEED_SETS]


def params_of(spec) -> Optional[int]:
    return param_count(spec) if spec.kind == "mlp" else None


def cells_rows(sf: str, gates: dict, timings: dict) -> list:
    rows = []
    for cell, g in gates.items():
        spec = CELLS[cell]
        t = timings.get(cell)
        
        for r in g["rows"]:
            rows.append(dict(
                sf=sf, cell=cell, 
                features=spec.features, kind=spec.kind, params=params_of(spec),
                pool_fraction=r["fraction"], pool=r["pool"], seeds=r["seeds"], misses=r["misses"],
                classical_policy=r["classical_policy"], classical_misses=r["classical_misses"],
                saving=r["saving"], belady_misses=r["belady_misses"], gap_captured=r["gap_captured"],
                median_us=None if t is None else t["median_ns"] / 1000,
                macs_per_decision=None if t is None else t["macs_per_decision"],
                nodes_per_decision=None if t is None else t["nodes_per_decision"],
                complete=g["complete"], passes_miss=g["passes_miss"], passes_cost=passes_cost(t)
            ))
    return rows


def plot(path: Path, misses: dict, pools: dict, names) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = sorted(pools.values())
    for name in names:
        ys = []
        for pool in xs:
            keys = [(t, name, pool) for t in mixed_traces((1, 2, 3))]
            ys.append(sum(misses[k] for k in keys) if all(k in misses for k in keys) else float("nan"))
        ax.plot(xs, ys, marker="o", label=name, linestyle="--" if name in CHECKED else "-")
        
    ax.set_xlabel("pool (frames)")
    ax.set_ylabel("misses, index_mixed seeds 1-3")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "bench/results/v3_check")
    ap.add_argument("--sim-dir", type=Path, default=REPO_ROOT / "bench/results/headroom/eviction")
    args = ap.parse_args(argv)
    out: Path = args.out
    sfs = sf_folders(out)
    if not sfs:
        print(f"no sf*/replay folder under {out}", file=sys.stderr)
        return 1
    
    gate_sf = "0.1" if "0.1" in sfs else sfs[0]
    warnings: list = []

    vrows = [r for sf in sfs for r in validate_sf(out, args.sim_dir, sf, gate_sf)]
    for r in vrows:
        if r["status"] != "match":
            warnings.append(f"validation {r['status']}: sf{r['sf']} {r['trace']} {r['policy']} pool {r['pool']}" + (f" ({r['bad_items']} items differ)" if r["status"] == "mismatch" else ""))
    pd.DataFrame(vrows).to_csv(out / "validation.csv", index=False)

    gates = load_gates(out, gate_sf)
    timing_notes: list = []
    timings = load_timings(out, gate_sf, timing_notes)
    warnings += [f"incomplete: {n}" for n in timing_notes]
    misses = load_misses(out, gate_sf)
    
    for cell in CELLS:
        if cell not in gates:
            warnings.append(f"incomplete: {cell} has no replays at SF={gate_sf}")
        elif not gates[cell]["complete"]:
            warnings.append(f"incomplete: {cell} is missing " + ", ".join(gates[cell]["missing"]))
        if cell not in timings and not any(n.startswith(cell + " timing") for n in timing_notes):
            warnings.append(f"incomplete: {cell} is not timed")
            
    best = best_cell(gates, timings)

    sf1 = None
    sf1_note = ""
    if best is not None:
        g1s = load_gates(out, "1.0") if "1.0" in sfs else {}
        if best in g1s:
            g1 = g1s[best]
            if not g1["complete"]:
                warnings.append(f"incomplete: {best} is missing at SF=1.0: " + ", ".join(g1["missing"]))
                
            sf1 = {
                "holds": g1["passes_miss"], 
                "gate": g1
            }
        else:
            sf1_note = f"{best} has no replays at SF=1.0"
            if g1s:
                sf1_note += f" (other cells do: {', '.join(g1s)})"
    result = verdict({"best": best}, sf1)

    cells_all = []
    for sf in sfs:
        cells_all += cells_rows(sf, gates if sf == gate_sf else load_gates(out, sf), timings if sf == gate_sf else load_timings(out, sf))
    pd.DataFrame(cells_all, columns=[
        "sf", "cell", "features", "kind", "params", "pool_fraction", "pool", "seeds", "misses", "classical_policy",
        "classical_misses", "saving", "belady_misses", "gap_captured", "median_us", "macs_per_decision",
        "nodes_per_decision", "complete", "passes_miss", "passes_cost"
    ]).to_csv(out / "cells.csv", index=False)
    for sf in sfs:
        m = misses if sf == gate_sf else load_misses(out, sf)
        present = {k[1] for k in m}
        plot(out / f"misses_sf{sf}.png", m, pools_for(sf), [c for c in CELLS if c in present] + list(CHECKED))

    md = []
    if warnings:
        md.append("**WARNING**: this run is not a decision.\n\n" + "\n".join(f"- {w}" for w in warnings))
        
    dev = "" if gate_sf == "0.1" else f" (development run at SF={gate_sf}, not a decision)"
    validation_failed = any(r["status"] != "match" for r in vrows)
    incomplete = any(w.startswith("incomplete:") for w in warnings)
    why = " and ".join(x for x, on in (("validation failed", validation_failed), ("incomplete", incomplete)) if on)
    shown = f"NO VERDICT** ({why}; computed: {result})" if why else f"{result}**"
    
    md.append(
        f"## Verdict\n\n**{shown}{dev}\n\n"
        f"Bar fixed on {BAR_FIXED_ON}: at least {MISS_BAR * 100:.0f}% fewer misses than the better of "
        f"arc_visit / lirs_visit on index_mixed seeds {{1,2,3}} summed and seed {{3}} alone, at the 10% and "
        f"25% pools, and a median of at most {COST_BAR_US:.2f} us per decision. SF=1.0 check: same two seed "
        f"sets at 36,008 frames.\n\nGate SF: {gate_sf}. Best passing cell: {best or 'none'}."
    )
    if best is None:
        alt = best_on_misses(gates)
        if alt is None:
            md[-1] += "\n\nBest on misses alone (no-go): none, no cell is complete."
        else:
            ga, ta = gates[alt], timings.get(alt)
            md[-1] += (
                f"\n\nBest on misses alone (no-go): {alt}, mean saving {pct(mean_saving(ga))} "
                f"(10% / 25% {{1,2,3}}: {pct(ga['savings'].get((0.10, '1+2+3')))} / "
                f"{pct(ga['savings'].get((0.25, '1+2+3')))}), median {us(ta)} us, "
                f"miss bar {'passed' if ga['passes_miss'] else 'not passed'}, "
                f"cost bar {'passed' if passes_cost(ta) else 'not passed'}."
            )
    else:
        if sf1 is not None:
            g1 = sf1["gate"]
            md[-1] += (
                f"\n\nSF=1.0, {best}: saving at 36,008 frames "
                f"{pct(g1['savings'].get((SF1_FRACTION, '1+2+3')))} ({{1,2,3}}), "
                f"{pct(g1['savings'].get((SF1_FRACTION, '3')))} ({{3}}); "
                f"{'holds' if sf1['holds'] else 'does not hold'}."
            )
        else:
            md[-1] += f"\n\nSF=1.0: {sf1_note}. Pending."

    gate_header = [f"{int(f * 100)}% {{{seed_label(s)}}}" for f in GATE_FRACTIONS for s in GATE_SEED_SETS]
    rows = []
    for cell in CELLS:
        g, t = gates.get(cell), timings.get(cell)
        spec = CELLS[cell]
        rows.append([cell, spec.features, spec.kind, "n/a" if spec.kind == "gbdt" else param_count(spec), *gate_cols(g), us(t),
                     "n/a" if g is None else ("yes" if g["complete"] else "NO"),
                     "yes" if g and g["passes_miss"] else "no", "yes" if passes_cost(t) else "no"])
        
    md.append("## The 8 cells (saving against the better of arc_visit / lirs_visit)\n\n" + table(["cell", "features", "kind", "params", *gate_header, "median us", "complete", "miss bar", "cost bar"], rows))

    def misses_table() -> str:
        head = ["pool", "seeds", "cell", "misses", "classical", "classical misses", "lru misses", "belady misses", "gap captured"]
        mrows = []
        
        for fraction in GATE_FRACTIONS:
            for seeds in GATE_SEED_SETS:
                label = seed_label(seeds)
                
                for cell in CELLS:
                    g = gates.get(cell)
                    r = next((x for x in g["rows"] if x["fraction"] == fraction and x["seeds"] == label), None) if g else None
                    pool = pools_for(gate_sf)[fraction]
                    lru = [misses.get((t, "lru", pool)) for t in mixed_traces(seeds)]
                    lru_text = "n/a" if None in lru else sum(lru)
                    
                    if r is None:
                        mrows.append([f"{int(fraction * 100)}% ({pool})", label, cell, "n/a", "", "", lru_text, "", "n/a"])
                    else:
                        mrows.append([
                            f"{int(fraction * 100)}% ({pool})", 
                            label, cell, r["misses"], 
                            r["classical_policy"], r["classical_misses"], 
                            lru_text, r["belady_misses"], pct(r["gap_captured"])
                        ])
                        
        return "## Misses and share of the Belady gap captured\n\n" + table(head, mrows)

    md.append(misses_table())

    ref_rows = []
    for ref in REFERENCES:
        g = gate(misses, ref, pools_for(gate_sf), required_for(gate_sf)) if any(k[1] == ref for k in misses) else None
        ref_rows.append([ref, *gate_cols(g)])
        
    ref_md = "## Reference rows (report-only)\n\n" + table(["policy", *gate_header], ref_rows)
    rb = gate(misses, "sampled_belady", pools_for(gate_sf), required_for(gate_sf))
    if any(k[1] == "sampled_belady" for k in misses):
        clears = rb["complete"] and rb["passes_miss"]
        ref_md += (f"\n\nsampled_belady {'clears' if clears else 'does not clear'} the 5% bar on all four. " + ("" if clears else "If it doesn't, no learned scorer with 32 samples can, so the no-go would be structural."))
    else:
        ref_md += "\n\nsampled_belady was not replayed."
    md.append(ref_md)

    def extra(title, label_list, fractions):
        rr = []
        for cell in CELLS:
            g = gates.get(cell)
            rr.append([cell, *[pct(None if g is None else g["savings"].get((f, lb))) for f in fractions for lb in label_list]])
        head = [f"{int(f * 100)}% {lb}" for f in fractions for lb in label_list]
        return f"## {title}\n\n" + table(["cell", *head], rr)

    md.append(extra("Held-out stream_s3 and lookup_s3", ["stream_s3", "lookup_s3"], GATE_FRACTIONS))
    md.append(extra("The 50% pool (report-only)", ["1+2+3", "3"], (0.50,)))

    abl_md = "## Ablation (report-only)\n\n"
    richer = best_richer_cell(gates, timings)
    if richer is None:
        abl_md += "No complete richer cell yet."
    else:
        abl_md += (
            f"Ablation cell: {richer}, the best richer cell"
            f"{' passing both bars' if gates[richer]['passes_miss'] and passes_cost(timings.get(richer)) else ' on misses alone (none passes both bars)'}.\n\n"
        )
        
        abl_rows = []
        for base in [richer, *[c for c in CELLS if CELLS[c].features == "richer" and c != richer]]:
            abl = ablation_cell(base).name
            if base != richer and not any(k[1] == abl for k in misses):
                continue
            
            ag = gate(misses, abl, pools_for(gate_sf), required_for(gate_sf)) if any(k[1] == abl for k in misses) else None
            abl_rows += [[base, *gate_cols(gates.get(base))], [abl, *gate_cols(ag)]]
            if ag is None:
                abl_rows[-1][0] += " (not replayed)"
                
        abl_md += table(["cell", *gate_header], abl_rows)
    md.append(abl_md)

    trows = []
    for cell in CELLS:
        t = timings.get(cell)
        if t is None:
            trows.append([cell, "not timed", "", "", "", "", "", ""])
            continue
        
        macs, nodes = float(t["macs_per_decision"]), float(t["nodes_per_decision"])
        trows.append([cell, us(t), us(t, "mean_ns"), us(t, "p90_ns"), f"{macs / 2:.0f}", f"{macs:.0f}", f"{macs * 2:.0f}", f"{nodes / 2:.0f} / {nodes:.0f} / {nodes * 2:.0f}"])
        
    md.append(
        "## Timing\n\nMultiply-adds at K = 16 and 64 are *inferred* (linear in K); K = 32 is measured. "
        "Nodes are for trees, in the same order.\n\n" + table(["cell", "median us", "mean us", "p90 us", "macs K=16 (*inferred*)", "macs K=32", "macs K=64 (*inferred*)", "nodes K=16 / 32 / 64"], trows)
    )

    counts = pd.DataFrame(vrows).groupby(["sf", "status"]).size().reset_index(name="n") if vrows else pd.DataFrame()
    md.append("## Validation against the recorded classical replays\n\n" + (table(["sf", "status", "cells"], counts.values.tolist()) if len(counts) else "nothing validated"))

    (out / "report.md").write_text("\n\n".join(md) + "\n")
    return 1 if warnings else 0


if __name__ == "__main__":
    sys.exit(main())
