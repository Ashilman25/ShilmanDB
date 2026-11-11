"""Unit tests for bench/eviction/headroom/analyze_eviction_headroom.py on small handwritten CSVs."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SPEC_PATH = Path(__file__).resolve().parents[1] / "headroom" / "analyze_eviction_headroom.py"
spec = importlib.util.spec_from_file_location("analyze_eviction_headroom", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["analyze_eviction_headroom"] = mod
spec.loader.exec_module(mod)

SF = "0.01"
P5, P10, P25, P100 = 72, 145, 362, 1448  # the 5%, 10%, 25% and 100% pools at SF=0.01
DEFAULT = 128
PLAIN = ["lru", "clock", "lru2", "arc", "lirs", "lru2_visit", "arc_visit", "lirs_visit", "lru2_full", "arc_full", "lirs_full", "v2", "belady"]


def policy_misses(lru=1000, best=None, belady=500, **over):
    """Misses for every policy at one pool: LRU, Belady, and optional overrides."""
    out = {p: lru for p in PLAIN}
    out["belady"] = belady
    if best is not None:
        out["arc_visit"] = best
    out.update(over)
    return out


def item_misses(total):
    return [total // 2, total - total // 2]


def write_sim(sim, trace, per_pool, sf=SF):
    """per_pool: {pool: {policy: misses}}. Pass 0 is warmup (+1000), pass 1 is the value, pass 2 is value + 1 (LRU stays equal)."""
    d = Path(sim) / f"sf{sf}"
    d.mkdir(parents=True, exist_ok=True)
    passes, items = [], []
    for pool, pols in per_pool.items():
        for pol, m in pols.items():
            for p, warm, extra in ((0, 1, 1000), (1, 0, 0), (2, 0, 0 if pol == "lru" else 1)):
                passes.append(f"{trace},{pol},{pool},{p},{warm},2,100,10,{m + extra},{m + extra},0")
            if pol == "lru":
                for p in (1, 2):
                    for seq, mm in enumerate(item_misses(m)):
                        items.append(f"{trace},{pol},{pool},{p},{seq},tpch,Q1,{mm}")
    (d / f"{trace}_passes.csv").write_text(
        "trace,policy,pool,pass,warmup,items,fetches,runs,misses,query_misses,lookup_misses\n" + "\n".join(passes) + "\n")
    (d / f"{trace}_items.csv").write_text("trace,policy,pool,pass,seq,kind,query,misses\n" + "\n".join(items) + "\n")
    (d / f"{trace}_cells.csv").write_text("trace,policy,pool,evictions,seconds\n")


def write_engine(folder, pool, misses, latency_ms=2000.0, read_ms=500.0, drop_col=None, bump_item=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cols = "pass,warmup,seq,kind,query,rows,latency_ms,hits,misses,visits,reads,read_ms,trace_start,trace_end".split(",")
    rows = []
    for p in (0, 1, 2):
        for seq, mm in enumerate(item_misses(misses)):
            if bump_item == (p, seq):
                mm += 3
            vals = [p, 1 if p == 0 else 0, seq, "tpch", "Q1", 1, 1.0, 0, mm, mm, mm, 0.5, 0, 10]
            rows.append(dict(zip(cols, vals)))
    df = pd.DataFrame(rows)
    if drop_col:
        df = df.drop(columns=[drop_col])
    df.to_csv(folder / "stream.csv", index=False)
    (folder / "run_info.csv").write_text(f"key,value\nsf,{SF}\npool_size,{pool}\n")
    totals = ["pass,warmup,items,latency_ms,hits,misses,visits,reads,read_ms,query_misses,lookup_misses"]
    for p in (0, 1, 2):
        totals.append(f"{p},{1 if p == 0 else 0},2,{latency_ms},0,{misses},{misses},{misses},{read_ms},{misses},0")
    (folder / "stream_totals.csv").write_text("\n".join(totals) + "\n")


def build(tmp_path, kinds=("index_stream", "index_mixed"), seeds=(1, 2), pools=(P5, P10, P25), best=None, belady=500, lru=1000, recordings=True, **over):
    sim, traces = tmp_path / "sim", tmp_path / "traces"
    for kind in kinds:
        for seed in seeds:
            name = f"{kind}_s{seed}"
            write_sim(sim, name, {p: policy_misses(lru=lru + 100 * (seed - 1), best=best, belady=belady, **over) for p in pools})
            if recordings:
                write_engine(traces / f"sf{SF}" / name, DEFAULT, lru + 100 * (seed - 1))
    return sim, traces


def run(tmp_path, sim, traces):
    code = mod.main(["--sim-dir", str(sim), "--traces-dir", str(traces)])
    return code, (sim / "report.md").read_text()


def rows(sim):
    return pd.read_csv(sim / "eviction_headroom.csv")


def test_parse_name():
    assert mod.parse_name("index_mixed_s2") == ("index_mixed", "+index", "mixed", 2, None)
    assert mod.parse_name("index100_stream_s1") == ("index100_stream", "+index (100 µs plans)", "stream", 1, None)
    assert mod.parse_name("baseline_stream_s1_w8") == ("baseline_stream", "baseline", "stream", 1, 8)
    assert mod.CLASSICAL[0] == "lru" and mod.ROOM == 1.10 and mod.CLOSE == 1.05


def test_seed_means_best_classical_and_vs_belady(tmp_path):
    sim, traces = tmp_path / "sim", tmp_path / "traces"
    for seed, (lru, best) in {1: (1000, 800), 2: (1200, 1000)}.items():
        write_sim(sim, f"index_stream_s{seed}", {P10: policy_misses(lru=lru, best=best, belady=500, lru2_full=100)})
    code, _ = run(tmp_path, sim, traces)
    df = rows(sim)
    lru = df[(df.policy == "lru") & (df.pool == P10)].iloc[0]
    assert (lru.misses, lru.misses_min, lru.misses_max, lru.seeds) == (1100, 1000, 1200, 2)
    arc = df[(df.policy == "arc_visit") & (df.pool == P10)].iloc[0]
    assert arc.misses == 900
    assert arc.vs_lru == pytest.approx(900 / 1100) and arc.vs_belady == pytest.approx(1.8)
    group = df[df.pool == P10]
    # lru2_full has the fewest misses but is never "best classical"
    assert set(group.best_classical) == {"arc_visit"}
    assert lru.pool_label == "10%" and lru.pool_fraction == pytest.approx(P10 / 1448)


def test_headline_uses_first_timed_pass(tmp_path):
    sim, traces = build(tmp_path, seeds=(1,), pools=(P10,))
    run(tmp_path, sim, traces)
    df = rows(sim)
    assert df[(df.policy == "lru") & (df.workload == "stream")].misses.iloc[0] == 1000  # warmup row is 2000


def test_default_pool_label(tmp_path):
    sim, traces = build(tmp_path, seeds=(1,), pools=(DEFAULT, P100))
    run(tmp_path, sim, traces)
    labels = dict(zip(rows(sim).pool, rows(sim).pool_label))
    assert labels[DEFAULT] == "default" and labels[P100] == "100%"


def test_zero_belady_leaves_ratios_empty(tmp_path):
    sim, traces = build(tmp_path, seeds=(1,), pools=(P100,), lru=0, belady=0)
    code, report = run(tmp_path, sim, traces)
    df = rows(sim)
    assert df.vs_lru.isna().all() and df.vs_belady.isna().all()
    assert "—" in report


def test_validation_all_match(tmp_path):
    sim, traces = build(tmp_path, seeds=(1,), pools=(DEFAULT, P10))
    code, report = run(tmp_path, sim, traces)
    val = pd.read_csv(sim / "validation.csv")
    assert code == 0 and (val.match == 1).all() and len(val) == 8  # 2 kinds x 2 passes x 2 items, at pool 128
    assert list(val.columns) == ["sf", "trace", "pool", "source", "pass", "seq", "engine_misses", "sim_misses", "match"]
    assert set(val.source) == {"recording"}
    assert "WARNING" not in report


def test_validation_mismatch_exits_1_with_warning_on_top(tmp_path):
    sim, traces = build(tmp_path, seeds=(1,), pools=(DEFAULT,), kinds=("index_stream",))
    write_engine(traces / f"sf{SF}" / "index_stream_s1", DEFAULT, 1000, bump_item=(1, 0))
    code, report = run(tmp_path, sim, traces)
    val = pd.read_csv(sim / "validation.csv")
    assert code == 1 and (val.match == 0).sum() == 1
    assert (sim / "eviction_headroom.csv").exists() and (sim / f"misses_sf{SF}.png").exists()
    assert "WARNING" in report.split("\n## ")[0]


def test_engine_item_without_sim_cell_is_a_mismatch(tmp_path):
    sim, traces = build(tmp_path, seeds=(1,), pools=(DEFAULT,), kinds=("index_stream",))
    write_engine(traces / f"sf{SF}" / "index_stream_s1", 999, 1000)  # no sim cell at pool 999
    code, _ = run(tmp_path, sim, traces)
    val = pd.read_csv(sim / "validation.csv")
    assert code == 1 and (val.match == 0).all() and val.sim_misses.isna().all() and len(val) == 4


def test_validate_folders_are_checked_too(tmp_path):
    sim, traces = build(tmp_path, seeds=(1,), pools=(DEFAULT, P10), kinds=("index_stream",))
    write_engine(sim / f"sf{SF}" / "validate" / "index_stream_s1" / f"pool{P10}", P10, 1000)
    code, _ = run(tmp_path, sim, traces)
    val = pd.read_csv(sim / "validation.csv")
    assert code == 0 and set(val.source) == {"recording", "validate"} and (val.match == 1).all()


def test_engine_file_missing_column_is_an_error(tmp_path, capsys):
    sim, traces = build(tmp_path, seeds=(1,), pools=(DEFAULT,), kinds=("index_stream",))
    write_engine(traces / f"sf{SF}" / "index_stream_s1", DEFAULT, 1000, drop_col="misses")
    assert run_code(sim, traces) == 1
    assert "stream.csv" in capsys.readouterr().err


def run_code(sim, traces):
    return mod.main(["--sim-dir", str(sim), "--traces-dir", str(traces)])


def test_no_sim_files_is_an_error(tmp_path, capsys):
    assert run_code(tmp_path / "sim", tmp_path / "traces") == 1
    assert "sim" in capsys.readouterr().err


def cells(tmp_path, a, b, kinds=("index_stream", "index_mixed")):
    sim, traces = tmp_path / "sim", tmp_path / "traces"
    for kind in kinds:
        write_sim(sim, f"{kind}_s1", {P10: policy_misses(lru=2000, best=a, belady=1000), P25: policy_misses(lru=2000, best=b, belady=1000)})
    return run(tmp_path, sim, traces)[1]


def test_decision_room(tmp_path):
    report = cells(tmp_path, 1200, 1150)
    assert "has real room: build it." in report and "provisional" in report


def test_decision_close(tmp_path):
    assert "within a few %" in cells(tmp_path, 1040, 1000)


def test_decision_between(tmp_path):
    report = cells(tmp_path, 1080, 1150)
    assert "Between the rows" in report and "has real room" not in report and "within a few %" not in report


def test_decision_unknown_when_a_cell_is_missing(tmp_path):
    report = cells(tmp_path, 1200, 1200, kinds=("index_stream",))
    assert "Unknown" in report and "has real room" not in report


def test_best_classical_can_be_a_visit_policy_and_never_full(tmp_path):
    sim, traces = build(tmp_path, seeds=(1,), pools=(P10,), kinds=("index_stream",), lirs_visit=700, arc_full=1)
    run(tmp_path, sim, traces)
    assert set(rows(sim).best_classical) == {"lirs_visit"}


def test_window_and_plan_check_stay_out_of_main_tables(tmp_path):
    sim, traces = build(tmp_path, seeds=(1,), pools=(DEFAULT, P10), kinds=("index_stream",))
    write_sim(sim, "index100_stream_s1", {P10: policy_misses(lru=1000, best=900, belady=500)})
    write_sim(sim, "index_stream_s1_w8", {P10: {"lru2_visit": 123, "arc_visit": 124, "lirs_visit": 125}})
    code, report = run(tmp_path, sim, traces)
    df = rows(sim)
    assert set(df.config) == {"+index"} and not (df.misses == 123).any()
    assert "Visit window" in report and "Plan check" in report


def test_every_fetch_penalty(tmp_path):
    sim, traces = build(tmp_path, seeds=(1,), pools=(P10,), kinds=("index_stream",), lru2=1000, lru2_visit=900, lru2_full=1500)
    _, report = run(tmp_path, sim, traces)
    assert "lru2 full ÷ best" in report and "1.667" in report


def test_estimated_pass_time_from_stream_totals(tmp_path):
    sim, traces = build(tmp_path, seeds=(1,), pools=(P10,), kinds=("index_stream",))
    run(tmp_path, sim, traces)
    lru = rows(sim).query("policy == 'lru'").iloc[0]
    assert lru.miss_s_100us == pytest.approx(0.1)
    assert lru.est_pass_s_100us == pytest.approx(1.5 + 0.1)  # (2000 - 500) ms of cpu


def test_fullcheck_difference_is_flagged_and_exits_1(tmp_path):
    sim, traces = tmp_path / "sim", tmp_path / "traces"
    write_sim(sim, "baseline_stream_s1", {P10: policy_misses(lru=1000, belady=500), DEFAULT: policy_misses(lru=1000, belady=500)})
    write_engine(traces / f"sf{SF}" / "baseline_stream_s1", DEFAULT, 1000)
    fc = sim / f"sf{SF}" / "fullcheck"
    fc.mkdir()
    (fc / "collapse_check.txt").write_text("identical\n")
    raw = policy_misses(lru=1000, belady=500)
    raw["lru"] = 1001  # differs from plain
    for pol in ("lru2", "arc", "lirs"):
        raw[pol] = raw[f"{pol}_full"]
    raw = {k: v for k, v in raw.items() if k in ("lru", "clock", "lru2", "arc", "lirs", "belady")}
    write_sim(tmp_path / "raw", "baseline_stream_s1", {P10: raw})  # same trace name, as the driver's raw files
    for suffix in ("passes", "items", "cells"):
        (tmp_path / "raw" / f"sf{SF}" / f"baseline_stream_s1_{suffix}.csv").rename(fc / f"baseline_stream_s1_raw_{suffix}.csv")
    code, report = run(tmp_path, sim, traces)
    top = report.split("\n## ")[0]
    assert code == 1 and "WARNING" in top and "fullcheck" in top


def test_fullcheck_clean_has_no_warning(tmp_path):
    sim, traces = tmp_path / "sim", tmp_path / "traces"
    write_sim(sim, "baseline_stream_s1", {P10: policy_misses(lru=1000, belady=500), DEFAULT: policy_misses(lru=1000, belady=500)})
    write_engine(traces / f"sf{SF}" / "baseline_stream_s1", DEFAULT, 1000)
    fc = sim / f"sf{SF}" / "fullcheck"
    fc.mkdir()
    (fc / "collapse_check.txt").write_text("identical\n")
    raw = {k: v for k, v in policy_misses(lru=1000, belady=500).items() if k in ("lru", "clock", "belady", "lru2_full", "arc_full", "lirs_full")}
    raw.update({"lru2": raw.pop("lru2_full"), "arc": raw.pop("arc_full"), "lirs": raw.pop("lirs_full")})
    write_engine(traces / f"sf{SF}" / "baseline_stream_s1", P10, 1000)
    write_sim(tmp_path / "raw", "baseline_stream_s1", {P10: raw})
    for suffix in ("passes", "items", "cells"):
        (tmp_path / "raw" / f"sf{SF}" / f"baseline_stream_s1_{suffix}.csv").rename(fc / f"baseline_stream_s1_raw_{suffix}.csv")
    code, report = run(tmp_path, sim, traces)
    assert code == 0 and "WARNING" not in report


def test_png_per_scale_factor(tmp_path):
    sim, traces = build(tmp_path, seeds=(1,), pools=(P5, P10, P25))
    write_sim(sim, "index_stream_s1", {P10: policy_misses()}, sf="0.1")
    run(tmp_path, sim, traces)
    assert (sim / "misses_sf0.01.png").stat().st_size > 1000
    assert (sim / "misses_sf0.1.png").stat().st_size > 1000


def drop_policy(sim, trace, policy, sf=SF):
    d = Path(sim) / f"sf{sf}"
    for suffix in ("passes", "items"):
        f = d / f"{trace}_{suffix}.csv"
        lines = f.read_text().splitlines()
        f.write_text("\n".join(l for i, l in enumerate(lines) if i == 0 or l.split(",")[1] != policy) + "\n")


def test_plan_check_ignores_75_percent(tmp_path):
    sim, traces = build(tmp_path, seeds=(1,), pools=(P10, 1086), kinds=("index_stream", "index_mixed"))
    for k in ("index100_stream_s1", "index100_mixed_s1"):
        write_sim(sim, k, {P10: policy_misses(best=None), 1086: policy_misses(best=600)})  # 75% moves far
    _, report = run(tmp_path, sim, traces)
    plan = report.split("## Plan check")[1].split("\n## ")[0]
    assert "75%" not in plan and "No cell" in plan


def test_plan_check_names_missing_cells(tmp_path):
    sim, traces = build(tmp_path, seeds=(1,), pools=(P10,), kinds=("index_stream",))
    write_sim(sim, "index100_stream_s1", {P10: policy_misses()})
    _, report = run(tmp_path, sim, traces)
    plan = report.split("## Plan check")[1].split("\n## ")[0]
    assert "Missing cells" in plan and "index_mixed" in plan and "not a clean pass" in plan


def test_decision_cell_missing_classical_policy_is_unknown(tmp_path):
    sim, traces = tmp_path / "sim", tmp_path / "traces"
    for kind in ("index_stream", "index_mixed"):
        write_sim(sim, f"{kind}_s1", {P10: policy_misses(lru=2000, best=1300, belady=1000), P25: policy_misses(lru=2000, best=1300, belady=1000)})
    drop_policy(sim, "index_mixed_s1", "arc_visit")
    _, report = run(tmp_path, sim, traces)
    decision = report.split("## Draft decision")[1].split("\n## ")[0]
    assert "Unknown" in decision and "arc_visit" in decision and "has real room" not in report


def test_decision_cell_with_different_seed_sets_is_unknown(tmp_path):
    sim, traces = tmp_path / "sim", tmp_path / "traces"
    for kind in ("index_stream", "index_mixed"):
        for seed in (1, 2):
            write_sim(sim, f"{kind}_s{seed}", {P10: policy_misses(lru=2000, best=1300, belady=1000), P25: policy_misses(lru=2000, best=1300, belady=1000)})
    drop_policy(sim, "index_stream_s2", "clock")
    _, report = run(tmp_path, sim, traces)
    top = report.split("\n## ")[0]
    decision = report.split("## Draft decision")[1].split("\n## ")[0]
    assert "Unknown" in decision and "seed sets differ" in decision and "seed sets differ" in top


def test_missing_recording_gives_nan_estimate_and_warning(tmp_path):
    sim, traces = build(tmp_path, seeds=(1, 2), pools=(DEFAULT,), kinds=("index_stream",))
    import shutil
    shutil.rmtree(traces / f"sf{SF}" / "index_stream_s2")
    _, report = run(tmp_path, sim, traces)
    lru = rows(sim).query("policy == 'lru'").iloc[0]
    assert pd.isna(lru.est_pass_s_100us) and lru.miss_s_100us == pytest.approx(0.105)
    assert "no recording folder for index_stream_s2" in report.split("\n## ")[0]


def test_fullcheck_missing_policy_in_raw_file_is_flagged(tmp_path):
    sim, traces = tmp_path / "sim", tmp_path / "traces"
    write_sim(sim, "baseline_stream_s1", {P10: policy_misses(lru=1000, belady=500), DEFAULT: policy_misses(lru=1000, belady=500)})
    write_engine(traces / f"sf{SF}" / "baseline_stream_s1", P10, 1000)
    fc = sim / f"sf{SF}" / "fullcheck"
    fc.mkdir()
    (fc / "collapse_check.txt").write_text("identical\n")
    pm = policy_misses(lru=1000, belady=500)
    raw = {"lru": 1000, "clock": 1000, "belady": 500, "lru2": pm["lru2_full"], "arc": pm["arc_full"]}  # no lirs
    write_sim(tmp_path / "raw", "baseline_stream_s1", {P10: raw})
    for suffix in ("passes", "items", "cells"):
        (tmp_path / "raw" / f"sf{SF}" / f"baseline_stream_s1_{suffix}.csv").rename(fc / f"baseline_stream_s1_raw_{suffix}.csv")
    code, report = run(tmp_path, sim, traces)
    assert code == 1 and "no lirs at pool" in report.split("\n## ")[0]


def test_v2check_notes_missing_sim_cell_and_rejects_empty_totals(tmp_path, capsys):
    sim, traces = build(tmp_path, seeds=(1,), pools=(DEFAULT,), kinds=("index_stream",))
    v2 = sim / f"sf{SF}" / "v2check" / "index_stream_s1" / f"pool{P10}"
    write_engine(v2, P10, 1000)  # sim has no cell at this pool
    _, report = run(tmp_path, sim, traces)
    assert "no sim cell" in report
    (v2 / "stream_totals.csv").write_text("pass,warmup,items,latency_ms,hits,misses,visits,reads,read_ms,query_misses,lookup_misses\n0,1,2,1,0,5,5,5,1,5,0\n")
    assert run_code(sim, traces) == 1
    assert "stream_totals.csv" in capsys.readouterr().err


def test_short_sim_file_is_a_named_error(tmp_path, capsys):
    sim, traces = build(tmp_path, seeds=(1,), pools=(DEFAULT,), kinds=("index_stream",))
    (sim / f"sf{SF}" / "index_stream_s1_passes.csv").write_text("trace,policy\nx,lru\n")
    assert run_code(sim, traces) == 1
    assert "index_stream_s1_passes.csv" in capsys.readouterr().err


def test_hint_covers_90_percent_and_ratios(tmp_path):
    sim, traces = build(tmp_path, seeds=(1,), pools=(1303,), kinds=("index_stream",), best=800)
    _, report = run(tmp_path, sim, traces)
    hint = report.split("## Which classical")[1]
    assert "90% (1303)" in hint and "0.800 / 1.000" in hint


def test_hint_keeps_the_rounded_fraction_pools_on_every_sf(tmp_path):
    for sf, fraction in (("0.01", 0.05), ("0.1", 0.75), ("1.0", 0.9)):
        pool = mod.pool_for_fraction(sf, fraction)
        sim = tmp_path / sf / "sim"
        write_sim(sim, "index_stream_s1", {pool: policy_misses(lru=1000, best=700, belady=500)}, sf=sf)
        passes, _ = mod.load_sim(sim)
        agg = mod.aggregate(mod.headlines(passes), {})
        ratios, near, cands = mod.hint(agg, sf)
        label = f"{round(fraction * 100)}% ({pool})"
        assert label in ratios, (sf, ratios)
        if fraction <= 0.75:
            assert label in near and "arc_visit" in cands, (sf, near, cands)
        else:
            assert label not in near  # the 90% pool is in the ratio table only


def test_hint_judges_the_default_pool_by_its_own_fraction(tmp_path):
    sim = tmp_path / "sim"
    write_sim(sim, "index_stream_s1", {DEFAULT: policy_misses(lru=1000, best=700, belady=500)})
    passes, _ = mod.load_sim(sim)
    agg = mod.aggregate(mod.headlines(passes), {})
    _, near, cands = mod.hint(agg, SF)
    assert "default (128)" in near and "arc_visit" in cands
    sim = tmp_path / "big"
    write_sim(sim, "index_stream_s1", {512: policy_misses(lru=1000, best=700, belady=500)}, sf="0.1")
    passes, _ = mod.load_sim(sim)
    agg = mod.aggregate(mod.headlines(passes), {})
    assert mod.hint(agg, "0.1") == ("No data.\n", "", {})  # 512 pages is 3.6%, under 5%


def test_worth_building_needs_a_win_over_lru(tmp_path):
    sim = tmp_path / "sim"
    # every policy within 2% of LRU: nothing qualifies
    write_sim(sim, "index_stream_s1", {P10: policy_misses(lru=1000, best=990, belady=500)})
    # arc_visit clearly better at the 25% pool only
    write_sim(sim, "index_mixed_s1", {P10: policy_misses(lru=1000, belady=500), P25: policy_misses(lru=1000, best=800, belady=500)})
    passes, _ = mod.load_sim(sim)
    agg = mod.aggregate(mod.headlines(passes), {})
    _, near, cands = mod.hint(agg, SF)
    assert cands == {"arc_visit": "index_mixed 25%"}
    assert "[visit]" in near and "must be" not in near


def test_report_lists_candidates_with_short_tags_and_legend(tmp_path):
    sim, traces = build(tmp_path, seeds=(1,), pools=(P25,), kinds=("index_stream",), best=800)
    _, report = run(tmp_path, sim, traces)
    hint = report.split("## Which classical")[1]
    assert "- arc_visit [visit]: index_stream 25%" in hint and "Mode tags:" in hint
    assert "the buffer pool's visit flag)" not in hint.split("Mode tags:")[0]


def _plan_check(tmp_path, cached, plan100):
    sim, traces = tmp_path / "sim", tmp_path / "traces"
    for kind in ("index_stream", "index_mixed"):
        write_sim(sim, f"{kind}_s1", {P10: policy_misses(**cached[0]), P25: policy_misses(**cached[1]), DEFAULT: policy_misses(lru=1000)})
        write_engine(traces / f"sf{SF}" / f"{kind}_s1", DEFAULT, 1000)
    for kind in ("index100_stream", "index100_mixed"):
        write_sim(sim, f"{kind}_s1", {P10: policy_misses(**plan100[0]), P25: policy_misses(**plan100[1])})
    return run(tmp_path, sim, traces)


def test_plan_check_that_flips_the_decision_is_shown_and_flagged(tmp_path):
    room = dict(lru=2000, best=1300, belady=1000)
    close = dict(lru=2000, best=1000, belady=1000)
    code, report = _plan_check(tmp_path, (room, room), (room, close))
    assert code == 0
    top = report.split("\n## ")[0]
    assert "CHANGES the decision" in top and "room" in top and "between" in top
    decision = report.split("## Draft decision")[1].split("\n## ")[0]
    assert "cached-read plans" in decision and "100 µs plans" in decision
    assert "**The plan check changes the answer: room with cached-read plans, between with 100 µs plans.**" in decision
    plan = report.split("## Plan check")[1].split("\n## ")[0]
    assert "different on the two plan sets" in plan
    assert "plan choices don't change" not in report


def test_plan_check_with_the_same_decision_says_so(tmp_path):
    room = dict(lru=2000, best=1300, belady=1000)
    _, report = _plan_check(tmp_path, (room, room), (room, room))
    assert "CHANGES" not in report.split("\n## ")[0]
    plan = report.split("## Plan check")[1].split("\n## ")[0]
    assert "same on both plan sets: room" in plan


def test_trace_without_a_recording_or_validation_run_exits_1(tmp_path, capsys):
    sim, traces = build(tmp_path, seeds=(1,), pools=(P10,), kinds=("index_stream",), recordings=False)
    code, report = run(tmp_path, sim, traces)
    assert code == 1 and "unvalidated: index_stream_s1" in report.split("\n## ")[0]
    assert "validation: no engine items were checked at all" in capsys.readouterr().err
    assert (sim / "validation.csv").is_file() and (sim / "eviction_headroom.csv").is_file()


def test_duplicate_rows_from_two_files_are_an_error(tmp_path, capsys):
    sim, traces = build(tmp_path, seeds=(1,), pools=(P10,), kinds=("index_stream",))
    d = sim / f"sf{SF}"
    (d / "index_stream_s1_again_passes.csv").write_text((d / "index_stream_s1_passes.csv").read_text())
    code = mod.main(["--sim-dir", str(sim), "--traces-dir", str(traces)])
    err = capsys.readouterr().err
    assert code == 1 and "duplicate rows" in err and "index_stream_s1_passes.csv" in err and "index_stream_s1_again_passes.csv" in err


def test_v2_seed_count_does_not_trigger_the_seed_set_warning(tmp_path):
    sim, traces = build(tmp_path, seeds=(1, 2), pools=(P10, P25), kinds=("index_stream", "index_mixed"))
    for kind in ("index_stream", "index_mixed"):
        f = sim / f"sf{SF}" / f"{kind}_s2_passes.csv"
        lines = f.read_text().splitlines()
        f.write_text("\n".join(l for i, l in enumerate(lines) if i == 0 or l.split(",")[1] != "v2") + "\n")
    _, report = run(tmp_path, sim, traces)
    assert "seed sets differ" not in report
    assert "v2 at 10%" in report and "mean of 1 seed(s)" in report
    decision = report.split("## Draft decision")[1]
    assert "Unknown" not in decision.split("\n## ")[0]
