"""Tests for report_bar.py: the bar table from one run's latency files."""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List

import pytest

SPEC_PATH = Path(__file__).resolve().parents[1] / "report_bar.py"
spec = importlib.util.spec_from_file_location("report_bar_under_test", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["report_bar_under_test"] = mod
spec.loader.exec_module(mod)

LATENCIES_HEADER = ("sf,query,config,median_latency_ms,rows,hits,misses,visits,page_hit_rate,reads,read_ms,"
                    "us_per_read,policy_ms,pool_size,miss_penalty_us,direct_io,eviction\n")
BAR_HEADER = ["query", "P", "P1", "D", "D1", "PG", "PG1", "P_over_PG", "P_over_D", "P1_over_D1",
              "ours_speedup", "duckdb_speedup", "check", "P_over_PG1", "P1_over_PG1"]


# One latencies.csv row as run_benchmarks writes it (only the median matters here).
def _ours(sf: str, query: str, config: str, ms: float) -> str:
    return f"{sf},{query},{config},{ms:.4f},1,,,,,,,,,8192,0,false,lru\n"


# Per-run rows as the DuckDB and Postgres runners write them: sf,query,config,run,latency_ms.
def _runs(sf: str, query: str, config: str, values: List[float]) -> str:
    return "".join(f"{sf},{query},{config},{i},{v:.4f}\n" for i, v in enumerate(values, start=1))


# A results folder with DuckDB's measured SF=10 Q6 numbers and made-up ShilmanDB ones, plus Q1.
def _write_fixture(folder: Path, skip: str = "") -> None:
    ours = LATENCIES_HEADER
    ours += _ours("10", "Q1", "columnar+parallel", 100.0)
    ours += _ours("10", "Q1", "columnar+parallel_1thread", 900.0)
    ours += _ours("10", "Q1", "columnar+parallel_2thread", 480.0)
    ours += _ours("10", "Q1", "columnar+vectorized", 6000.0)
    if skip != "P1":
        ours += _ours("10", "Q6", "columnar+parallel_1thread", 300.0)
    ours += _ours("10", "Q6", "columnar+parallel", 40.0)
    ours += _ours("10", "Q6", "columnar+parallel_2thread", 160.0)
    # Another scale factor is left out
    ours += _ours("1.0", "Q6", "columnar+parallel", 4.0)
    (folder / "latencies.csv").write_text(ours)

    duck = "sf,query,config,run,latency_ms\n"
    duck += _runs("10", "Q1", "duckdb", [34.0, 34.2, 35.0])
    duck += _runs("10", "Q1", "duckdb_1thread", [393.0, 390.0, 400.0])
    duck += _runs("10", "Q6", "duckdb", [10.0, 10.4, 11.0])
    duck += _runs("10", "Q6", "duckdb_1thread", [112.0, 112.8, 113.5])
    duck += _runs("1.0", "Q6", "duckdb", [1.0, 1.0, 1.0])
    (folder / "duckdb_latencies.csv").write_text(duck)

    if skip != "postgres":
        pg = "sf,query,config,run,latency_ms\n"
        pg += _runs("10", "Q1", "postgres", [2857.4])
        pg += _runs("10", "Q1", "postgres_1worker", [7153.6])
        pg += _runs("10", "Q6", "postgres", [1100.0, 1127.4, 1200.0])
        pg += _runs("10", "Q6", "postgres_1worker", [1312.3, 1300.0, 1400.0])
        (folder / "postgres_latencies.csv").write_text(pg)


def _read_bar(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_worked_example(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _write_fixture(tmp_path)
    assert mod.main(["--results-dir", str(tmp_path), "--sf", "10"]) == 0

    out = tmp_path / "bar_sf10.csv"
    with out.open(newline="") as f:
        assert next(csv.reader(f)) == BAR_HEADER + ["P2"]
    rows = _read_bar(out)
    assert [r["query"] for r in rows] == ["Q1", "Q6"]
    q6 = rows[1]
    assert float(q6["P"]) == 40.0
    assert float(q6["P1"]) == 300.0
    assert float(q6["D"]) == 10.4
    assert float(q6["D1"]) == 112.8
    assert float(q6["PG"]) == 1127.4
    assert float(q6["PG1"]) == 1312.3
    assert float(q6["P_over_D"]) == pytest.approx(3.846, abs=5e-4)
    assert float(q6["P1_over_D1"]) == pytest.approx(2.660, abs=5e-4)
    assert float(q6["ours_speedup"]) == pytest.approx(7.5, abs=1e-9)
    assert float(q6["duckdb_speedup"]) == pytest.approx(10.846, abs=5e-4)
    assert float(q6["P_over_PG"]) == pytest.approx(40.0 / 1127.4, abs=1e-6)
    assert q6["check"] == q6["P_over_D"]
    # Against one Postgres worker: PG1 1312.3, not PG 1127.4
    assert q6["P_over_PG1"] == f"{40.0 / 1312.3:.6f}"
    assert q6["P1_over_PG1"] == f"{300.0 / 1312.3:.6f}"
    assert rows[0]["P_over_PG1"] == f"{100.0 / 7153.6:.6f}"
    assert rows[0]["P1_over_PG1"] == f"{900.0 / 7153.6:.6f}"
    assert float(q6["P2"]) == 160.0
    # Ratios carry 6 decimals
    assert q6["P_over_D"] == f"{40.0 / 10.4:.6f}"
    printed = capsys.readouterr().out
    assert "Q6" in printed and "P1_over_PG1" in printed and f"{300.0 / 1312.3:.6f}" in printed


def test_check_multiplies_back(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    table = mod.build_table(tmp_path, "10")
    for row in table.rows:
        assert abs(row.check - row.p_over_d) <= 1e-9 * row.p_over_d
    q6 = table.rows[1]
    assert q6.p_over_d == pytest.approx(40.0 / 10.4, rel=1e-12)
    assert q6.duckdb_speedup == pytest.approx(112.8 / 10.4, rel=1e-12)
    assert q6.p_over_pg1 == pytest.approx(40.0 / 1312.3, rel=1e-12)
    assert q6.p1_over_pg1 == pytest.approx(300.0 / 1312.3, rel=1e-12)


def test_out_flag(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    target = tmp_path / "elsewhere.csv"
    assert mod.main(["--results-dir", str(tmp_path), "--sf", "10", "--out", str(target)]) == 0
    assert target.exists()
    assert not (tmp_path / "bar_sf10.csv").exists()


def test_missing_engine_file_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _write_fixture(tmp_path, skip="postgres")
    assert mod.main(["--results-dir", str(tmp_path), "--sf", "10"]) == 2
    assert "postgres" in capsys.readouterr().err
    assert not (tmp_path / "bar_sf10.csv").exists()


def test_missing_config_for_a_query_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _write_fixture(tmp_path, skip="P1")
    assert mod.main(["--results-dir", str(tmp_path), "--sf", "10"]) == 2
    err = capsys.readouterr().err
    assert "columnar+parallel_1thread" in err and "Q6" in err


def test_no_rows_at_that_scale_factor_exits_2(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    assert mod.main(["--results-dir", str(tmp_path), "--sf", "0.1"]) == 2


def test_check_mismatch_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                capsys: pytest.CaptureFixture) -> None:
    _write_fixture(tmp_path)
    # Stands in for an arithmetic bug in the multiply-back
    monkeypatch.setattr(mod.BarRow, "check", property(lambda row: row.p_over_d * (1 + 1e-6)))
    assert mod.main(["--results-dir", str(tmp_path), "--sf", "10"]) == 1
    assert "Q1, Q6" in capsys.readouterr().err
    assert not (tmp_path / "bar_sf10.csv").exists()


@pytest.mark.parametrize("config,query", [("columnar+parallel", "Q6"), ("duckdb", "Q1"),
                                          ("columnar+parallel_2thread", "Q6")])
def test_zero_median_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture, config: str, query: str) -> None:
    _write_fixture(tmp_path)
    name = "duckdb_latencies.csv" if config == "duckdb" else "latencies.csv"
    lines = (tmp_path / name).read_text().splitlines(keepends=True)
    zeroed = []
    for line in lines:
        fields = line.split(",")
        if fields[0] == "10" and fields[1] == query and fields[2] == config:
            # The median column for ours, the per-run latency for an engine
            fields[3 if name == "latencies.csv" else 4] = "0.0000" + ("\n" if name != "latencies.csv" else "")
        zeroed.append(",".join(fields))
    (tmp_path / name).write_text("".join(zeroed))
    assert mod.main(["--results-dir", str(tmp_path), "--sf", "10"]) == 2
    assert f"zero median for {config} for {query}" in capsys.readouterr().err
    assert not (tmp_path / "bar_sf10.csv").exists()
