"""Unit tests for bench/eviction/engine/analyze_engine_eviction.py on small handwritten CSVs."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SPEC_PATH = Path(__file__).resolve().parents[1] / "engine" / "analyze_engine_eviction.py"
spec = importlib.util.spec_from_file_location("analyze_engine_eviction", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["analyze_engine_eviction"] = mod
spec.loader.exec_module(mod)

SF = "0.01"
P25 = 362  # the 25% pool at SF=0.01
EVICTION = {"learned-v2": "v2", "learned-v1": "v1"}
VALIDATION_COLUMNS = "sf,step,trace,policy,sim_policy,pool,pass,seq,engine_misses,sim_misses,match,gated".split(",")
EVICTION_COLUMNS = ("sf,kind,pool_label,pool,pool_fraction,policy,penalty_us,seeds,misses,vs_lru,vs_belady,"
                    "pass_ms,pass_ms_vs_lru,policy_ns_per_call,policy_ns_per_fetch,policy_share").split(",")


def write_sim(sim: Path, trace: str, cells: dict, sf: str = SF) -> None:
    """cells: {(policy, pool): [misses per item]}, the same in the warmup pass and both timed passes."""
    d = sim / f"sf{sf}"
    d.mkdir(parents=True, exist_ok=True)
    passes, items = [], []
    for (policy, pool), per_item in cells.items():
        for p in (0, 1, 2):
            passes.append(f"{trace},{policy},{pool},{p},{int(p == 0)},{len(per_item)},1000,100,{sum(per_item)},{sum(per_item)},0")
            if p:
                items += [f"{trace},{policy},{pool},{p},{seq},tpch,Q1,{m}" for seq, m in enumerate(per_item)]
    (d / f"{trace}_passes.csv").write_text("trace,policy,pool,pass,warmup,items,fetches,runs,misses,query_misses,lookup_misses\n" + "\n".join(passes) + "\n")
    (d / f"{trace}_items.csv").write_text("trace,policy,pool,pass,seq,kind,query,misses\n" + "\n".join(items) + "\n")


def write_run(engine: Path, step: str, trace: str, policy: str, pool: int, per_item: list[int], *, sf: str = SF,
              jobs: int = 1, penalty: int = 0, planner_read_us: str = "0.60", latency: tuple = (100.0, 100.0, 100.0),
              read_ms: float = 10.0, hits: int = 1000, timing: list | None = None, cold_start: str | None = None,
              eviction: str | None = None, driver_info: bool = True, passes: tuple = (0, 1, 2),
              binary_sha256: str = "a" * 64) -> Path:
    folder = engine / f"sf{sf}" / step / trace / policy / f"pool{pool}"
    folder.mkdir(parents=True, exist_ok=True)
    stream = ["pass,warmup,seq,kind,query,rows,latency_ms,hits,misses,visits,reads,read_ms,trace_start,trace_end"]
    totals = ["pass,warmup,items,latency_ms,hits,misses,visits,reads,read_ms,query_misses,lookup_misses"]
    for p in passes:
        for seq, m in enumerate(per_item):
            stream.append(f"{p},{int(p == 0)},{seq},tpch,Q1,1,1.0,10,{m},{m},{m},0.5,,")
        total = sum(per_item)
        totals.append(f"{p},{int(p == 0)},{len(per_item)},{latency[p]},{hits},{total},{total},{total},{read_ms},{total},0")
    (folder / "stream.csv").write_text("\n".join(stream) + "\n")
    (folder / "stream_totals.csv").write_text("\n".join(totals) + "\n")
    cold = cold_start or ("false" if step == "warm" else "true")
    (folder / "run_info.csv").write_text(
        f"key,value\nsf,{sf}\npool_size,{pool}\nmiss_penalty_us,{penalty}\neviction,{eviction or EVICTION.get(policy, policy)}\n"
        f"policy_timing,{'true' if timing else 'false'}\nplanner_read_us,{planner_read_us}\nworkload,stream\ncold_start,{cold}\n")
    if driver_info:
        (folder / "driver_info.csv").write_text(f"key,value\njobs,{jobs}\nstep,{step}\nbinary,load_tpch\nbinary_sha256,{binary_sha256}\n")
    if timing:
        rows = ["seq,query,rows,latency_ms,policy_calls,policy_ms,policy_raw_ms,record_ms,set_evictable_ms,evict_ms,remove_ms"]
        rows += [f"{seq},Q1,1,{lat},{calls},{ms},{ms},0,0,0,0" for seq, (calls, ms, lat) in enumerate(timing)]
        (folder / "policy_time.csv").write_text("\n".join(rows) + "\n")
    return folder


def basic_sim(sim: Path) -> None:
    write_sim(sim, "baseline_stream_s1", {("lru", P25): [5, 7], ("lirs_visit", P25): [3, 4], ("lirs", P25): [9, 9],
                                          ("arc", P25): [9, 9], ("arc_visit", P25): [6, 6], ("belady", P25): [2, 2]})


def run(tmp_path: Path) -> tuple[int, str]:
    code = mod.main(["--engine-dir", str(tmp_path / "e"), "--sim-dir", str(tmp_path / "s")])
    return code, (tmp_path / "e" / "report.md").read_text()


def validation(tmp_path: Path) -> pd.DataFrame:
    return pd.read_csv(tmp_path / "e" / "engine_validation.csv")


def table(tmp_path: Path) -> pd.DataFrame:
    return pd.read_csv(tmp_path / "e" / "engine_eviction.csv")


def row(df: pd.DataFrame, **match) -> pd.Series:
    sub = df
    for k, v in match.items():
        sub = sub[sub[k] == v]
    assert len(sub) == 1, f"{len(sub)} rows for {match}"
    return sub.iloc[0]


def test_all_match_exits_0(tmp_path):
    basic_sim(tmp_path / "s")
    write_run(tmp_path / "e", "cached", "baseline_stream_s1", "lru", P25, [5, 7])
    write_run(tmp_path / "e", "cached", "baseline_stream_s1", "lirs", P25, [3, 4], latency=(90.0, 80.0, 70.0))
    code, report = run(tmp_path)
    assert code == 0 and "WARNING" not in report
    val = validation(tmp_path)
    assert list(val.columns) == VALIDATION_COLUMNS
    assert len(val) == 8 and val["match"].eq(1).all() and val["gated"].eq(1).all()
    assert set(val["pass"]) == {1, 2}  # the warmup pass is not compared
    df = table(tmp_path)
    assert list(df.columns) == EVICTION_COLUMNS
    lirs = row(df, policy="lirs")
    assert (lirs.misses, lirs.seeds, lirs.penalty_us, lirs.pool_label) == (7, 1, 0, "25%")
    assert lirs.vs_lru == pytest.approx(7 / 12) and lirs.vs_belady == pytest.approx(7 / 4)
    assert lirs.pool_fraction == pytest.approx(P25 / 1448)
    assert lirs.pass_ms == pytest.approx(75.0) and lirs.pass_ms_vs_lru == pytest.approx(0.75)


def test_mismatch_exits_1_and_warns(tmp_path):
    basic_sim(tmp_path / "s")
    write_run(tmp_path / "e", "cached", "baseline_stream_s1", "lirs", P25, [3, 5])
    code, report = run(tmp_path)
    assert code == 1
    assert report.splitlines()[2] == "**WARNING**"
    assert "2 of 4 checked items differ" in report
    val = validation(tmp_path)
    bad = val[val["match"] == 0]
    assert len(bad) == 2 and set(bad["seq"]) == {1} and set(bad["engine_misses"]) == {5} and set(bad["sim_misses"]) == {4}
    assert (tmp_path / "e" / "engine_eviction.csv").exists()  # everything is still written


def test_policy_mapping(tmp_path):
    assert mod.SIM_POLICY == {"lru": "lru", "clock": "clock", "lru2": "lru2_visit", "arc": "arc_visit", "lirs": "lirs_visit", "v2": "v2"}
    basic_sim(tmp_path / "s")
    write_run(tmp_path / "e", "cached", "baseline_stream_s1", "arc", P25, [6, 6])  # sim arc has 9, arc_visit 6
    code, _ = run(tmp_path)
    val = validation(tmp_path)
    assert code == 0 and set(val["sim_policy"]) == {"arc_visit"} and set(val["policy"]) == {"arc"}


def test_index100_cells_for_penalty_runs(tmp_path):
    sim = tmp_path / "s"
    write_sim(sim, "index_stream_s1", {("lru", P25): [10, 10], ("belady", P25): [5, 5]})
    write_sim(sim, "index100_stream_s1", {("lru", P25): [20, 20], ("belady", P25): [8, 8]})
    write_sim(sim, "index_mixed_s1", {("lru", P25): [11, 11]})
    write_sim(sim, "index100_mixed_s1", {("lru", P25): [21, 21]})
    write_sim(sim, "baseline_stream_s1", {("lru", P25): [30, 30]})
    write_sim(sim, "index_lookup_s1", {("lru", P25): [40, 40]})
    e = tmp_path / "e"
    hundred = dict(penalty=100, planner_read_us="100.00")
    write_run(e, "penalty", "index_stream_s1", "lru", P25, [20, 20], **hundred)
    write_run(e, "penalty", "index_mixed_s1", "lru", P25, [21, 21], **hundred)
    write_run(e, "penalty", "baseline_stream_s1", "lru", P25, [30, 30], **hundred)
    write_run(e, "penalty", "index_lookup_s1", "lru", P25, [40, 40], **hundred)
    write_run(e, "cached", "index_stream_s1", "lru", P25, [10, 10])
    code, report = run(tmp_path)
    assert code == 0, report
    assert validation(tmp_path)["match"].eq(1).all()
    assert mod.sim_trace("index_stream_s1", 100.0) == "index100_stream_s1"
    assert mod.sim_trace("index_mixed_s1", 100.0) == "index100_mixed_s1"
    assert mod.sim_trace("index_stream_s1", 0.6) == "index_stream_s1"
    assert mod.sim_trace("baseline_stream_s1", 100.0) == "baseline_stream_s1"
    assert mod.sim_trace("index_lookup_s1", 100.0) == "index_lookup_s1"
    # Belady for the 100 µs row comes from the index100 cells too
    hot = row(table(tmp_path), kind="index_stream", penalty_us=100)
    assert hot.vs_belady == pytest.approx(40 / 16)


def test_one_timed_pass_compares_pass_1_only(tmp_path):
    basic_sim(tmp_path / "s")
    write_run(tmp_path / "e", "penalty", "baseline_stream_s1", "lirs", P25, [3, 4], penalty=100, planner_read_us="100.00", passes=(0, 1))
    code, report = run(tmp_path)
    val = validation(tmp_path)
    assert code == 0, report
    assert len(val) == 2 and set(val["pass"]) == {1} and val["match"].eq(1).all()


def test_missing_sim_cell_fails(tmp_path):
    basic_sim(tmp_path / "s")
    write_run(tmp_path / "e", "cached", "baseline_stream_s1", "clock", P25, [5, 7])  # the sim has no clock cell
    code, report = run(tmp_path)
    assert code == 1 and "no sim cell" in report
    val = validation(tmp_path)
    assert len(val) == 4 and val["sim_misses"].isna().all() and val["match"].eq(0).all()


def test_warm_and_v2_not_gated(tmp_path):
    sim = tmp_path / "s"
    basic_sim(sim)
    write_sim(sim, "index_lookup_s1", {("v2", 72): [5, 6], ("lru", 72): [5, 5]})
    e = tmp_path / "e"
    write_run(e, "warm", "baseline_stream_s1", "lirs", P25, [9, 4])
    write_run(e, "learned", "index_lookup_s1", "learned-v2", 72, [5, 5])
    code, report = run(tmp_path)
    assert code == 0 and "WARNING" not in report
    val = validation(tmp_path)
    assert val["gated"].eq(0).all()
    assert set(val[val["step"] == "learned"]["sim_policy"]) == {"v2"}
    assert (val["match"] == 0).sum() == 4  # 2 warm items and 2 V2 items, both timed passes
    assert "Warm start" in report and "V2" in report
    assert "not gated" in report


def test_v1_left_out(tmp_path):
    basic_sim(tmp_path / "s")
    write_run(tmp_path / "e", "cached", "baseline_stream_s1", "lru", P25, [5, 7])
    write_run(tmp_path / "e", "learned", "baseline_stream_s1", "learned-v1", 72, [50, 50])
    code, report = run(tmp_path)
    val = validation(tmp_path)
    assert code == 0 and "learned" not in set(val["step"]) and len(val) == 4
    assert "learned-v1" in report and "not compared" in report
    assert "v1" not in set(table(tmp_path)["policy"])


def test_parallel_times_left_out(tmp_path):
    basic_sim(tmp_path / "s")
    write_run(tmp_path / "e", "cached", "baseline_stream_s1", "lru", P25, [5, 7], jobs=4)
    write_run(tmp_path / "e", "cached", "baseline_stream_s1", "lirs", P25, [3, 4], jobs=1)
    code, _ = run(tmp_path)
    df = table(tmp_path)
    assert code == 0
    assert pd.isna(row(df, policy="lru").pass_ms)
    lirs = row(df, policy="lirs")
    assert lirs.pass_ms == pytest.approx(100.0) and pd.isna(lirs.pass_ms_vs_lru)
    assert row(df, policy="lru").misses == 12  # misses are kept either way


def test_policy_cost_columns(tmp_path):
    write_sim(tmp_path / "s", "index_stream_s1", {("lirs_visit", 128): [500, 500], ("lru", 128): [600, 600], ("lirs_visit", P25): [5, 5]})
    e = tmp_path / "e"
    # The timed pass is much slower than a normal one (the timer's own cost), so the share must use the normal pass
    write_run(e, "timing", "index_stream_s1", "lirs", 128, [500, 500], hits=9000, latency=(120.0, 90.0, 110.0),
              timing=[(1000, 0.5, 300.0), (3000, 1.5, 500.0)])
    write_run(e, "timing", "index_stream_s1", "lirs", P25, [5, 5], hits=9000, jobs=2, timing=[(1000, 0.5, 40.0)])
    code, _ = run(tmp_path)
    df = table(tmp_path)
    assert code == 0
    lirs = row(df, policy="lirs", pool=128)
    assert lirs.misses == 1000  # a cell with only a timing run takes its misses from it
    assert lirs.policy_ns_per_call == pytest.approx(2.0e6 / 4000)
    assert lirs.policy_ns_per_fetch == pytest.approx(2.0e6 / (9000 + 1000))
    assert lirs.pass_ms == pytest.approx(100.0)
    assert lirs.policy_share == pytest.approx(2.0 / 100.0)  # net ÷ the median normal timed pass, not ÷ 800 ms
    parallel = row(df, policy="lirs", pool=P25)
    assert pd.isna(parallel.policy_ns_per_call) and pd.isna(parallel.policy_share) and pd.isna(parallel.pass_ms)


def test_report_sections(tmp_path):
    basic_sim(tmp_path / "s")
    write_sim(tmp_path / "s", "index_stream_s1", {("lru", P25): [10, 10], ("lirs_visit", P25): [6, 6], ("belady", P25): [5, 5]})
    write_sim(tmp_path / "s", "index100_stream_s1", {("lru", P25): [20, 20]})
    e = tmp_path / "e"
    write_run(e, "cached", "baseline_stream_s1", "lru", P25, [5, 7])
    write_run(e, "cached", "baseline_stream_s1", "lirs", P25, [3, 4])
    write_run(e, "cached", "index_stream_s1", "lru", P25, [10, 10])
    write_run(e, "cached", "index_stream_s1", "lirs", P25, [6, 6])
    write_run(e, "cached", "index100_stream_s1", "lru", P25, [20, 20], planner_read_us="100.00", read_ms=1.0)
    write_run(e, "penalty", "index_stream_s1", "lru", P25, [20, 20], penalty=100, planner_read_us="100.00",
              latency=(4100.0, 4100.0, 4100.0), read_ms=4000.0)
    write_run(e, "timing", "index_stream_s1", "lirs", P25, [6, 6], timing=[(10, 0.01, 50.0)])
    (e / f"sf{SF}" / "stream_correctness.csv").write_text("sf,config,seq,query,status\n0.01,cached-baseline_stream_s1-lru-pool362,0,Q1,PASS\n")
    code, report = run(tmp_path)
    assert code == 0, report
    for heading in ("# Classical eviction in the engine", "## Does the engine match the simulator?", "## Answers against SQLite",
                    f"## SF={SF}: misses per timed pass", "## Pass times", "## At 100 µs per read: measured against cpu + misses x 100 µs",
                    "## Each policy's own cost", "## The new bar for eviction ML", "## How to read this"):
        assert heading in report, heading
    assert "*measured*" in report
    # 4,100 ms measured against the cached 100 µs-plan run's CPU (99 ms) + 40 misses x 0.1 ms
    assert "103.0" in report and "39.806" in report


def test_correctness_fail_exits_1(tmp_path):
    basic_sim(tmp_path / "s")
    write_run(tmp_path / "e", "cached", "baseline_stream_s1", "lru", P25, [5, 7])
    (tmp_path / "e" / f"sf{SF}" / "stream_correctness.csv").write_text("sf,config,seq,query,status\n0.01,x,0,Q1,FAIL\n0.01,x,1,Q1,PASS\n")
    code, report = run(tmp_path)
    assert code == 1 and "1 of 2 answers FAIL" in report


def test_unfinished_or_odd_folders_fail(tmp_path):
    basic_sim(tmp_path / "s")
    write_run(tmp_path / "e", "cached", "baseline_stream_s1", "lru", P25, [5, 7], driver_info=False)
    code, report = run(tmp_path)
    assert code == 1 and "unfinished" in report
    write_run(tmp_path / "e", "cached", "baseline_stream_s1", "lru", P25, [5, 7], cold_start="false")
    code, report = run(tmp_path)
    assert code == 1 and "cold_start" in report


def test_no_runs_fails(tmp_path):
    basic_sim(tmp_path / "s")
    (tmp_path / "e").mkdir()
    code, report = run(tmp_path)
    assert code == 1 and "no engine runs" in report


TIMING = [(10, 0.01, 50.0), (10, 0.01, 50.0)]  # policy_time.csv rows for a two-item run


def noise_sim(sim: Path, sf: str = SF, pool: int = P25) -> None:
    write_sim(sim, "baseline_stream_s1", {("lru", pool): [5, 7], ("lirs_visit", pool): [3, 4], ("clock", pool): [5, 6],
                                          ("belady", pool): [2, 2]}, sf=sf)


def times_table(report: str, heading: str) -> list[str]:
    lines = report.splitlines()
    start = lines.index(heading)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("#")), len(lines))
    return [l for l in lines[start:end] if l.startswith("| ")]


def test_noise_band_marks_changes_inside_it(tmp_path):
    noise_sim(tmp_path / "s")
    e = tmp_path / "e"
    # twins: LRU's two runs differ by 3%, LIRS's by 0%, so the band is 2 cells, median 1.5%, max 3.0%
    write_run(e, "cached", "baseline_stream_s1", "lru", P25, [5, 7], latency=(100.0, 100.0, 100.0))
    write_run(e, "timing", "baseline_stream_s1", "lru", P25, [5, 7], latency=(100.0, 103.0, 103.0), timing=TIMING)
    write_run(e, "cached", "baseline_stream_s1", "lirs", P25, [3, 4], latency=(100.0, 102.0, 102.0))
    write_run(e, "timing", "baseline_stream_s1", "lirs", P25, [3, 4], latency=(100.0, 102.0, 102.0), timing=TIMING)
    write_run(e, "cached", "baseline_stream_s1", "clock", P25, [5, 6], latency=(90.0, 90.0, 90.0))  # no twin
    # at 100 µs per read the cached band doesn't apply, however small the change
    hundred = dict(penalty=100, planner_read_us="100.00")
    write_run(e, "penalty", "baseline_stream_s1", "lru", P25, [5, 7], latency=(5000.0, 5000.0, 5000.0), **hundred)
    write_run(e, "penalty", "baseline_stream_s1", "lirs", P25, [3, 4], latency=(5010.0, 5010.0, 5010.0), **hundred)
    code, report = run(tmp_path)
    assert code == 0, report
    assert f"- SF={SF}: 2 cells, median 1.5%, max 3.0%." in report

    cached = times_table(report, f"### SF={SF}, baseline_stream, cached reads")
    assert len(cached) == 2  # head and one pool
    cells = [c.strip() for c in cached[1].strip("|").split("|")]
    assert cells[0] == "25% (362)"
    assert cells[1] == "100.0"                          # lru, read from the cached run
    assert cells[2] == "90.0 (-10.0%)"                  # clock: outside the band
    assert cells[5] == "102.0 (+2.0%, within noise)"   # lirs: inside it

    hot = times_table(report, f"### SF={SF}, baseline_stream, 100 µs per read")
    assert "5,010.0 (+0.2%)" in hot[1] and "within noise" not in hot[1]


def test_no_noise_band_without_sequential_twins(tmp_path):
    p25_01 = mod.pool_for_fraction("0.1", 0.25)
    noise_sim(tmp_path / "s")
    noise_sim(tmp_path / "s", sf="0.1", pool=p25_01)
    e = tmp_path / "e"
    # SF=0.01: the cached run was made in parallel, so it is no twin of the timing run
    write_run(e, "cached", "baseline_stream_s1", "lru", P25, [5, 7], jobs=4)
    write_run(e, "timing", "baseline_stream_s1", "lru", P25, [5, 7], latency=(100.0, 100.0, 100.0), timing=TIMING)
    write_run(e, "cached", "baseline_stream_s1", "lirs", P25, [3, 4], latency=(100.0, 100.5, 100.5))
    # SF=0.1: cached runs only
    write_run(e, "cached", "baseline_stream_s1", "lru", p25_01, [5, 7], sf="0.1")
    write_run(e, "cached", "baseline_stream_s1", "lirs", p25_01, [3, 4], sf="0.1", latency=(100.0, 100.5, 100.5))
    code, report = run(tmp_path)
    assert code == 0, report
    assert f"- SF={SF}: no noise band (no twin runs)." in report
    assert "- SF=0.1: no noise band (no twin runs)." in report
    assert "within noise" not in report.replace("*within noise*", "")
    assert "100.5 (+0.5%)" in report


def test_mixed_builds_warn_without_failing(tmp_path):
    basic_sim(tmp_path / "s")
    e = tmp_path / "e"
    write_run(e, "cached", "baseline_stream_s1", "lru", P25, [5, 7], binary_sha256="1" * 64)
    write_run(e, "cached", "baseline_stream_s1", "lirs", P25, [3, 4], binary_sha256="2" * 64)
    write_run(e, "warm", "baseline_stream_s1", "lirs", P25, [9, 4], binary_sha256="3" * 64)  # not gated, not counted
    code, report = run(tmp_path)
    assert code == 0
    assert report.splitlines()[2] == "**WARNING**"
    assert f"SF={SF}: the gated runs come from 2 different load_tpch builds" in report
    assert "111111111111: 1 runs" in report and "222222222222: 1 runs" in report and "333333333333" not in report

    # one build: no warning
    write_run(e, "cached", "baseline_stream_s1", "lirs", P25, [3, 4], binary_sha256="1" * 64)
    code, report = run(tmp_path)
    assert code == 0 and "WARNING" not in report
