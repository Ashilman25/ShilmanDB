"""Tests for analyze_v3_check.py on synthetic CSVs."""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

SPEC_PATH = Path(__file__).with_name("analyze_v3_check.py")
spec = importlib.util.spec_from_file_location("analyze_v3_check", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["analyze_v3_check"] = mod
spec.loader.exec_module(mod)

POOLS = {0.10: 145, 0.25: 362}
MIXED = ("index_mixed_s1", "index_mixed_s2", "index_mixed_s3")
SF = "0.01"
ITEM_HEADER = ["trace", "policy", "pool", "pass", "seq", "kind", "query", "misses"]


def fill(misses, cell, learned, classical=1000, belady=800, pools=POOLS, arc=None, lirs=None):
    for pool in pools.values():
        for trace in MIXED:
            misses[(trace, cell, pool)] = learned
            misses[(trace, "arc_visit", pool)] = classical if arc is None else arc
            misses[(trace, "lirs_visit", pool)] = classical + 100 if lirs is None else lirs
            misses[(trace, "belady", pool)] = belady
    return misses


def test_saving_exact_bar():
    assert mod.saving(95_000, 100_000) >= mod.MISS_BAR
    assert mod.meets_bar(mod.saving(95_000, 100_000))
    assert not mod.meets_bar(mod.saving(95_001, 100_000))


def test_gap_captured():
    assert mod.gap_captured(110, 120, 100) == 0.5


def test_seed3_alone_can_fail_a_cell():
    misses = fill({}, "c", 940)  # 6% saving at every pool and seed
    misses[("index_mixed_s3", "c", 362)] = 951  # seed 3 alone: 4.9% at 25%
    # the {1,2,3} sum still saves over 6% at 25%
    misses[("index_mixed_s1", "c", 362)] = 929
    misses[("index_mixed_s2", "c", 362)] = 929
    g = mod.gate(misses, "c", POOLS)
    assert g["savings"][(0.25, "1+2+3")] >= 0.05
    assert g["savings"][(0.25, "3")] < 0.05
    assert g["complete"] and not g["passes_miss"]


def test_classical_is_better_of_arc_and_lirs():
    m = {}
    for trace, arc, lirs in (("index_mixed_s1", 500, 400), ("index_mixed_s2", 500, 400), ("index_mixed_s3", 100, 300)):
        m[(trace, "arc_visit", 362)] = arc
        m[(trace, "lirs_visit", 362)] = lirs
    assert mod.classical_for(m, 362, (1, 2, 3)) == ("arc_visit", 1100)  # 1,100 each: tie goes to arc
    assert mod.classical_for(m, 362, (1, 2)) == ("lirs_visit", 800)
    assert mod.classical_for(m, 362, (3,)) == ("arc_visit", 100)
    m[("index_mixed_s3", "lirs_visit", 362)] = 50
    assert mod.classical_for(m, 362, (3,)) == ("lirs_visit", 50)


def test_cost_bar():
    assert mod.passes_cost({"median_ns": 5000})
    assert not mod.passes_cost({"median_ns": 5010})
    assert not mod.passes_cost(None)


def test_best_cell_prefers_passing_cells_by_mean_saving():
    misses = {}
    fill(misses, "mlp_small_basic", 900)   # 10%
    fill(misses, "gbdt_basic", 800)        # 20%, but too slow
    fill(misses, "gbdt_richer", 850)       # 15%
    gates = {c: mod.gate(misses, c, POOLS) for c in ("mlp_small_basic", "gbdt_basic", "gbdt_richer")}
    timings = {"mlp_small_basic": {"median_ns": 1000}, "gbdt_basic": {"median_ns": 9000},
               "gbdt_richer": {"median_ns": 4000}}
    assert mod.best_cell(gates, timings) == "gbdt_richer"
    timings["gbdt_richer"] = None
    assert mod.best_cell(gates, timings) == "mlp_small_basic"
    assert mod.best_richer_cell(gates, timings) == "gbdt_richer"
    timings["mlp_small_basic"] = {"median_ns": 6000}
    assert mod.best_cell(gates, timings) is None


def test_verdict_rules():
    assert mod.verdict({"best": None}, None) == "NO-GO"
    assert mod.verdict({"best": "c"}, None) == "PENDING SF=1.0"
    assert mod.verdict({"best": "c"}, {"holds": True}) == "GO"
    assert mod.verdict({"best": "c"}, {"holds": False}) == "NO-GO"


def test_missing_replay_is_incomplete_not_go():
    misses = fill({}, "c", 800)
    del misses[("index_mixed_s3", "c", 362)]
    g = mod.gate(misses, "c", POOLS)
    assert not g["complete"] and not g["passes_miss"]
    assert mod.best_cell({"c": g}, {"c": {"median_ns": 1000}}) is None


def test_missing_classical_replay_is_named_not_the_cell():
    misses = fill({}, "c", 800)
    del misses[("index_mixed_s3", "arc_visit", 362)]
    g = mod.gate(misses, "c", POOLS)
    assert not g["complete"]
    assert "arc_visit index_mixed_s3 pool 362" in g["missing"]
    assert not any(m.startswith("c ") for m in g["missing"])


# ---- main() on a synthetic output folder ----

def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def item(trace, policy, pool, misses):
    return [trace, policy, pool, 1, 0, "tpch", "Q1", misses]


TRACES = MIXED + ("index_stream_s3", "index_lookup_s3")
ALL_POOLS = (145, 362, 724)
BASE = {"lru": 1200, "arc_visit": 1000, "lirs_visit": 1100, "belady": 800}


def build(tmp_path, skip=None, ref_tweak=None, learned=900, timed=True):
    out, ref = tmp_path / "out", tmp_path / "ref"
    for trace in TRACES:
        rows, rows5 = [], []
        for pool in ALL_POOLS:
            for pol, m in BASE.items():
                rows.append(item(trace, pol, pool, m))
                rows5.append(item(trace, pol, pool, m + 7))  # the recorded items are shifted below
        rows5 = [r[:7] + [r[7] - 7] for r in rows5]
        if ref_tweak:
            rows5 = ref_tweak(trace, rows5)
        write_csv(out / f"sf{SF}" / "replay" / f"{trace}_classical_items.csv", ITEM_HEADER, rows)
        write_csv(ref / f"sf{SF}" / f"{trace}_items.csv", ITEM_HEADER, rows5)
    for cell in mod.CELLS:
        for trace in TRACES:
            for pool in ALL_POOLS:
                if skip and skip(cell, trace, pool):
                    continue
                write_csv(out / f"sf{SF}" / "replay" / f"{trace}_{cell}_p{pool}_items.csv", ITEM_HEADER,
                          [item(trace, cell, pool, learned)])
        if timed:
            write_csv(out / f"sf{SF}" / "timing" / f"{cell}_timing.csv",
                      ["trace", "policy", "pool", "k", "decisions", "median_ns", "mean_ns", "p90_ns",
                       "macs_per_decision", "nodes_per_decision", "features", "model"],
                      [["index_mixed_s1", cell, 145, 32, 16384, 1000, 1100, 1500, 3000, 0, 3, "m"]])
    return out, ref


def run(tmp_path, **kw):
    out, ref = build(tmp_path, **kw)
    code = mod.main(["--out", str(out), "--sim-dir", str(ref)])
    return code, (out / "report.md").read_text()


def test_clean_run_is_a_development_pending(tmp_path):
    code, report = run(tmp_path)
    assert code == 0
    assert "development run at SF=0.01, not a decision" in report
    assert "PENDING SF=1.0" in report
    assert not report.startswith("**WARNING**")
    assert (tmp_path / "out" / "cells.csv").exists() and (tmp_path / "out" / "validation.csv").exists()


def test_validation_mismatch_exits_1_after_writing(tmp_path):
    def tweak(trace, rows):
        if trace == "index_mixed_s1":
            for r in rows:
                if r[1] == "arc_visit" and r[2] == 145:
                    r[7] += 1
        return rows
    code, report = run(tmp_path, ref_tweak=tweak)
    assert code == 1
    assert report.startswith("**WARNING**")
    assert "mismatch" in report
    assert "**NO VERDICT** (validation failed; computed: PENDING SF=1.0)" in report


def test_validation_missing_ref_cell_fails(tmp_path):
    def tweak(trace, rows):
        if trace == "index_mixed_s2":
            rows = [r for r in rows if r[2] != 362]
        return rows
    code, report = run(tmp_path, ref_tweak=tweak)
    assert code == 1
    assert report.startswith("**WARNING**")
    assert "index_mixed_s2 arc_visit pool 362" in report
    assert "**NO VERDICT** (validation failed; computed:" in report


def test_main_missing_replay_fails(tmp_path):
    code, report = run(tmp_path, skip=lambda c, t, p: c == "gbdt_basic" and t == "index_mixed_s3" and p == 362)
    assert code == 1
    assert report.startswith("**WARNING**")
    assert "gbdt_basic" in report
    assert "**NO VERDICT** (incomplete; computed:" in report


def test_missing_timing_is_incomplete(tmp_path):
    code, report = run(tmp_path, timed=False)
    assert code == 1
    assert "not timed" in report
    assert "**NO VERDICT** (incomplete; computed: NO-GO)" in report


def test_no_go_names_best_on_misses(tmp_path):
    code, report = run(tmp_path, learned=990)  # 1% saving: nothing passes
    assert code == 0
    assert "**NO-GO**" in report
    assert "Best on misses alone (no-go): mlp_small_basic" in report
    assert "miss bar not passed" in report


def build_sf1(tmp_path, cells, learned=900, skip_trace=None):
    out, ref = tmp_path / "out", tmp_path / "ref"
    for trace in MIXED:
        rows = [item(trace, p, 36008, m) for p, m in BASE.items()]
        write_csv(out / "sf1.0" / "replay" / f"{trace}_classical_items.csv", ITEM_HEADER, rows)
        write_csv(ref / "sf1.0" / f"{trace}_items.csv", ITEM_HEADER, rows)
        for cell in cells:
            if trace != skip_trace:
                write_csv(out / "sf1.0" / "replay" / f"{trace}_{cell}_p36008_items.csv", ITEM_HEADER,
                          [item(trace, cell, 36008, learned)])


def run_sf1(tmp_path, cells, **kw):
    out, ref = build(tmp_path)
    build_sf1(tmp_path, cells, **kw)
    code = mod.main(["--out", str(out), "--sim-dir", str(ref)])
    return code, (out / "report.md").read_text()


def test_sf1_holds_is_go(tmp_path):
    code, report = run_sf1(tmp_path, ["mlp_small_basic"])
    assert code == 0 and "**GO**" in report and "holds" in report


def test_sf1_not_holding_is_no_go(tmp_path):
    code, report = run_sf1(tmp_path, ["mlp_small_basic"], learned=990)
    assert code == 0 and "**NO-GO**" in report and "does not hold" in report


def test_sf1_partial_is_incomplete(tmp_path):
    code, report = run_sf1(tmp_path, ["mlp_small_basic"], skip_trace="index_mixed_s3")
    assert code == 1
    assert report.startswith("**WARNING**") and "missing at SF=1.0" in report
    assert "**NO VERDICT** (incomplete; computed: NO-GO)" in report


def test_sf1_pending_when_only_other_cells_have_replays(tmp_path):
    code, report = run_sf1(tmp_path, ["gbdt_basic"])
    assert code == 0
    assert "PENDING SF=1.0" in report and "other cells do: gbdt_basic" in report


def richer_gates(timings_ns):
    misses = {}
    fill(misses, "mlp_small_richer", 800)   # 20%
    fill(misses, "mlp_large_richer", 900)   # 10%
    fill(misses, "gbdt_richer", 850)        # 15%
    cells = ("mlp_small_richer", "mlp_large_richer", "gbdt_richer")
    gates = {c: mod.gate(misses, c, POOLS) for c in cells}
    return gates, {c: {"median_ns": ns} for c, ns in zip(cells, timings_ns)}


def test_best_richer_cell_needs_both_bars():
    gates, timings = richer_gates((9000, 4000, 6000))  # best on misses is too slow
    assert mod.best_richer_cell(gates, timings) == "mlp_large_richer"


def test_best_richer_cell_falls_back_to_misses_alone():
    gates, timings = richer_gates((9000, 9000, 9000))
    assert mod.best_richer_cell(gates, timings) == "mlp_small_richer"
    assert mod.best_richer_cell(gates, {}) == "mlp_small_richer"


# ---- replay completeness, timing rows ----

def write_replay(out, trace, policy, pool, items):
    rows = [[trace, policy, pool, 1, seq, "tpch", "Q1", m] for seq, m in items]
    write_csv(out / f"sf{SF}" / "replay" / f"{trace}_{policy}_p{pool}_items.csv", ITEM_HEADER, rows)


def test_truncated_learned_replay_counts_as_missing(tmp_path):
    out = tmp_path
    for t in MIXED:
        write_replay(out, t, "arc_visit", 362, [(0, 500), (1, 500)])
        write_replay(out, t, "learned_a", 362, [(0, 100), (1, 100)] if t != MIXED[1] else [(0, 100)])
        write_replay(out, t, "lru", 362, [(0, 500)])  # classical files are checked by validation, not here
    m = mod.load_misses(out, SF)
    assert (MIXED[0], "learned_a", 362) in m and (MIXED[2], "learned_a", 362) in m
    assert (MIXED[1], "learned_a", 362) not in m
    assert (MIXED[1], "lru", 362) in m


def test_timing_row_must_match_trace_pool_and_k(tmp_path):
    cols = ["trace", "policy", "pool", "k", "decisions", "median_ns", "mean_ns", "p90_ns",
            "macs_per_decision", "nodes_per_decision", "features", "model"]
    good = ["index_mixed_s1", "c", 145, 32, 1, 1000, 1, 1, 1, 0, 3, "m"]
    for name, row in (("mlp_small_basic", good), ("gbdt_basic", ["index_mixed_s2", *good[1:]]),
                      ("gbdt_richer", [*good[:2], 362, *good[3:]]), ("mlp_large_basic", [*good[:3], 16, *good[4:]])):
        write_csv(tmp_path / f"sf{SF}" / "timing" / f"{name}_timing.csv", cols, [row])
    notes = []
    t = mod.load_timings(tmp_path, SF, notes)
    assert list(t) == ["mlp_small_basic"]
    assert len(notes) == 3 and any("gbdt_basic" in n for n in notes)


def test_report_has_misses_table_and_n_a_params(tmp_path):
    code, report = run(tmp_path)
    assert "## Misses and share of the Belady gap captured" in report
    assert "| 10% (145) | 1+2+3 | mlp_small_basic | 2700 | arc_visit | 3000 | 3600 | 2400 | 50.00% |" in report
    assert "| gbdt_basic | basic | gbdt | n/a |" in report
    assert "Ablation cell:" in report
