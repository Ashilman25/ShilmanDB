"""Unit tests for bench/charts/export_chart_data.py."""
from __future__ import annotations

import csv
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, List

import pytest

SPEC_PATH = Path(__file__).resolve().parents[1] / "export_chart_data.py"
spec = importlib.util.spec_from_file_location("export_chart_data", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["export_chart_data"] = mod
spec.loader.exec_module(mod)

LAT_HEADER = "sf,query,config,median_latency_ms,rows,hits,misses\n"

REPORT_MD = """# Analyst check

**GO**

## The bars

| Bar | Count | Rate | 95% Wilson | Needs | Holds |
|---|---:|---:|---:|---:|---:|
| catch (bar 1): wrong shown answers warned or refused | 20 of 35 | 0.571 | 0.409-0.720 | at least 0.50 | yes |
| keep, refusals (bar 2): right shown answers refused | 0 of 227 | 0.000 | 0.000-0.017 | at most 0.10 | yes |
| keep, warnings (bar 3): right shown answers warned or refused | 18 of 227 | 0.079 | 0.051-0.122 | at most 0.25 | yes |

### Bar 1 by cause, and without engine_traps

| Wrong shown answers | Caught | Rate | 95% Wilson |
|---|---:|---:|---:|
| cause model | 20 of 35 | 0.571 | 0.409-0.720 |
"""

EVICTION_HEADER = ("sf,workload,config,pool_label,pool,pool_fraction,policy,seeds,misses,"
                   "misses_min,misses_max,vs_lru,vs_belady,best_classical\n")

SUMMARY_HEADER = ("sf,query,num_orders,best_order,best_ms,classical_order,classical_ms,"
                  "classical_over_best,learned_order,learned_ms,learned_over_best\n")


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


# A baseline results folder: baseline PASS, duckdb DRIFT, postgres FAIL, sqlite SKIP.
def make_baseline_dir(root: Path) -> Path:
    d = root / "baseline_sf1"
    d.mkdir(parents=True)
    (d / "latencies.csv").write_text(
        LAT_HEADER
        + "1.0,Q1,baseline,800.0,4,1,1\n"
        + "1.0,Q1,duckdb,40.0,4,,\n"
        + "1.0,Q1,postgres,900.0,4,,\n"
        + "1.0,Q1,sqlite_disk,2000.0,4,,\n"
        + "0.01,Q1,baseline,7.0,4,1,1\n"
    )
    (d / "correctness.csv").write_text(
        "sf,query,config,status\n"
        "1.0,Q1,baseline,PASS\n"
        "1.0,Q1,duckdb,DRIFT\n"
        "1.0,Q1,postgres,FAIL\n"
        "1.0,Q1,sqlite_disk,SKIP\n"
    )
    (d / "reference_checks.csv").write_text(
        "sf,query,config,reference,status,max_abs_diff,max_rel_diff\n"
        "1.0,Q1,baseline,SQLite,PASS,,\n"
        "1.0,Q1,duckdb,SQLite,DRIFT,1e-09,2e-12\n"
        "1.0,Q1,duckdb,DuckDB,DRIFT,3e-09,1e-12\n"
        "1.0,Q1,postgres,SQLite,FAIL,,\n"
    )
    (d / "environment.csv").write_text("key,value\nchip,Apple M5 Max\n")
    return d


def make_results_dir(root: Path) -> Path:
    r = root / "results"
    ev = r / "headroom" / "eviction"
    ev.mkdir(parents=True)
    rows = []
    for sf in ("0.01", "1.0"):
        for wl in ("stream", "mixed", "lookup"):
            for label, frac in (("default", "0.0142"), ("10%", "0.1"), ("25%", "0.25")):
                for pol, misses, vsb in (("lru", 130, 1.3), ("arc", 125, 1.25),
                                         ("arc_visit", 120, 1.2), ("lirs_visit", 115, 1.15),
                                         ("v2", 140, 1.4), ("belady", 100, 1.0)):
                    rows.append(f"{sf},{wl},+index,{label},10,{frac},{pol},3,{misses},"
                                f"{misses},{misses},1,{vsb},lirs_visit\n")
                rows.append(f"{sf},{wl},baseline,{label},10,{frac},lru,3,999,999,999,1,9.9,lru\n")
    (ev / "eviction_headroom.csv").write_text(EVICTION_HEADER + "".join(rows))
    join = r / "headroom" / "join_output_after"
    join.mkdir(parents=True)
    (join / "summary.csv").write_text(
        SUMMARY_HEADER
        + "0.1,Q3,4,a,1,a,1,1.5,a,1,1.5\n"
        + "1.0,Q3,4,a,1,b,1,1.0104,b,1,1.0104\n"
        + "1.0,Q10,8,a,1,b,1,1.01948,c,1,1.03\n"
    )
    an = r / "analyst_check"
    an.mkdir(parents=True)
    (an / "report.md").write_text(REPORT_MD)
    return r


def test_export_baseline_fail_has_no_time(tmp_path: Path) -> None:
    src = make_baseline_dir(tmp_path)
    out = tmp_path / "out"
    mod.export_baseline(src, "1", out)
    rows = {r["config"]: r for r in read_rows(out / "baseline_sf1.csv")}
    assert set(rows) == {"baseline", "duckdb", "postgres", "sqlite_disk"}   # SF=0.01 row dropped
    assert rows["postgres"]["status"] == "FAIL"
    assert rows["postgres"]["median_ms"] == ""
    assert rows["baseline"]["median_ms"] == "800.0"
    assert rows["sqlite_disk"]["status"] == "SKIP"
    assert rows["sqlite_disk"]["median_ms"] == "2000.0"
    assert rows["baseline"]["engine"] == "shilmandb"
    assert rows["sqlite_disk"]["engine"] == "sqlite"
    assert (out / "environment_sf1.csv").read_text() == "key,value\nchip,Apple M5 Max\n"


def test_export_baseline_keeps_drift_diffs(tmp_path: Path) -> None:
    src = make_baseline_dir(tmp_path)
    out = tmp_path / "out"
    mod.export_baseline(src, "1", out)
    rows = {r["config"]: r for r in read_rows(out / "baseline_sf1.csv")}
    duck = rows["duckdb"]
    assert duck["engine"] == "duckdb"
    assert duck["status"] == "DRIFT"
    assert duck["median_ms"] == "40.0"
    assert float(duck["max_abs_diff"]) == 3e-09     # largest over both references
    assert float(duck["max_rel_diff"]) == 2e-12
    assert rows["baseline"]["max_abs_diff"] == ""
    assert list(read_rows(out / "baseline_sf1.csv")[0]) == [
        "query", "engine", "config", "median_ms", "status", "max_abs_diff", "max_rel_diff"]


def test_export_analyst_reads_counts(tmp_path: Path) -> None:
    results = make_results_dir(tmp_path)
    out = tmp_path / "out"
    mod.export_analyst(out, results_dir=results)
    rows = read_rows(out / "analyst_bars.csv")
    assert [(r["count"], r["total"]) for r in rows] == [("20", "35"), ("0", "227"), ("18", "227")]
    assert rows[0]["wilson_low"] == "0.409"
    assert rows[0]["wilson_high"] == "0.720"
    assert rows[0]["rate"] == "0.571"
    assert rows[0]["needs"] == "at least 0.50"
    assert rows[0]["bar"].startswith("catch (bar 1)")


def test_export_eviction_picks_sf1_index_policies(tmp_path: Path) -> None:
    results = make_results_dir(tmp_path)
    out = tmp_path / "out"
    mod.export_eviction(out, results_dir=results)
    rows = read_rows(out / "eviction_misses.csv")
    assert {r["sf"] for r in rows} == {"1.0"}
    assert {r["kind"] for r in rows} == {"index_stream", "index_mixed"}
    assert {r["policy"] for r in rows} == {"lru", "arc_visit", "lirs_visit", "v2", "belady"}
    assert {r["pool_fraction"] for r in rows} == {"0.0142", "0.1", "0.25"}
    assert len(rows) == 2 * 3 * 5


def test_export_ceilings_join_and_eviction(tmp_path: Path) -> None:
    results = make_results_dir(tmp_path)
    out = tmp_path / "out"
    mod.export_ceilings(out, results_dir=results)
    rows = read_rows(out / "ceilings.csv")
    join = [r for r in rows if r["part"] == "join"]
    assert {(r["item"], r["method"]) for r in join} == {
        ("Q3", "classical"), ("Q3", "learned"), ("Q10", "classical"), ("Q10", "learned")}
    assert {r["sf"] for r in join} == {"1.0"}
    ev = [r for r in rows if r["part"] == "eviction"]
    assert {r["sf"] for r in ev} == {"0.01"}
    assert len(ev) == 8
    best = {r["item"]: float(r["ratio"]) for r in ev if r["method"] == "best_classical"}
    assert best == {"stream 10%": 1.15, "stream 25%": 1.15, "mixed 10%": 1.15, "mixed 25%": 1.15}
    v2 = {r["item"]: float(r["ratio"]) for r in ev if r["method"] == "v2"}
    assert set(v2.values()) == {1.4}


def set_tree_mode(root: Path, writable: bool) -> None:
    for dirpath, _, files in os.walk(root):
        mode = 0o755 if writable else 0o555
        os.chmod(dirpath, mode)
        for name in files:
            os.chmod(Path(dirpath) / name, 0o644 if writable else 0o444)


def test_export_never_writes_under_results(tmp_path: Path) -> None:
    results = make_results_dir(tmp_path)
    baseline = make_baseline_dir(results)
    out = tmp_path / "out"
    set_tree_mode(results, writable=False)
    try:
        mod.export_baseline(baseline, "1", out)
        mod.export_eviction(out, results_dir=results)
        mod.export_ceilings(out, results_dir=results)
        mod.export_analyst(out, results_dir=results)
        mod.main(["--results-dir", str(results), "--baseline-dir-sf1", str(baseline),
                  "--out", str(out)])
    finally:
        set_tree_mode(results, writable=True)
    assert sorted(p.name for p in out.iterdir()) == [
        "analyst_bars.csv", "baseline_sf1.csv", "ceilings.csv", "environment_sf1.csv",
        "eviction_misses.csv"]


def test_main_skips_missing_inputs(tmp_path: Path) -> None:
    out = tmp_path / "out"
    mod.main(["--results-dir", str(tmp_path / "nothing"), "--out", str(out)])
    assert not out.exists() or list(out.iterdir()) == []


def test_main_refuses_out_under_results_dir(tmp_path: Path, capsys) -> None:
    results = make_results_dir(tmp_path)
    out = results / "x"
    assert mod.main(["--results-dir", str(results), "--out", str(out)]) != 0
    assert not out.exists()
    assert "under" in capsys.readouterr().err


def test_main_refuses_out_under_repo_results(tmp_path: Path) -> None:
    results = make_results_dir(tmp_path)
    out = mod.RESULTS_DIR / "chart_export_refusal_test"
    assert mod.main(["--results-dir", str(results), "--out", str(out)]) != 0
    assert not out.exists()
