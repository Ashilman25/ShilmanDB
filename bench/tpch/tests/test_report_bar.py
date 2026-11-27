"""Tests for report_bar.py: the bar table from one run's latency files."""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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


# One latencies.csv row as run_benchmarks writes it (the median, and the pool at SF=100).
def _ours(sf: str, query: str, config: str, ms: float, pool: int = 8192) -> str:
    return f"{sf},{query},{config},{ms:.4f},1,,,,,,,,,{pool},0,false,lru\n"


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


# SF=100: the spill, direct and duckdb_20g columns, the memory table and the answer check
SPILL, DIRECT, SWEEP2 = "columnar+parallel_spill", "columnar+parallel_direct", "columnar+parallel_2thread"
EXTRA_HEADER = ["PS", "PS_over_P", "PS_spill_bytes", "PD", "PD_over_P", "D20", "P_over_D20"]
MEMORY_HEADER = ["config", "pool_size", "memory_limit", "peak_rss_bytes", "spill_bytes_max"]
STATS_HEADER = ("query,run,threads,morsels,latch_wait_ms,memory_limit,peak_tracked_bytes,"
                "spill_bytes_written,spill_bytes_read,spill_files,spill_partitions,spill_ms\n")
POOL_100 = 2097152
LIMIT_4G, LIMIT_256M = 4 << 30, 256 << 20

# Made-up medians per config and query
OURS_100 = {
    "columnar+parallel": {"Q1": 1000.0, "Q6": 400.0},
    "columnar+parallel_1thread": {"Q1": 12000.0, "Q6": 3000.0},
    SWEEP2: {"Q1": 6000.0, "Q6": 1600.0},
    SPILL: {"Q1": 1300.0, "Q6": 520.0},
    DIRECT: {"Q1": 1500.0, "Q6": 700.0},
}
ENGINES_100 = {
    "duckdb": {"Q1": 250.0, "Q6": 100.0},
    "duckdb_1thread": {"Q1": 2500.0, "Q6": 1200.0},
    "duckdb_20g": {"Q1": 260.0, "Q6": 125.0},
    "postgres": {"Q1": 20000.0, "Q6": 8000.0},
    "postgres_1worker": {"Q1": 50000.0, "Q6": 9000.0},
}
# spill_bytes_written per timed run; Q6 has an even count on purpose
SPILLS_100 = {
    "columnar+parallel": {"Q1": [0, 0, 0], "Q6": [0, 0, 0]},
    "columnar+parallel_1thread": {"Q1": [0, 0, 0], "Q6": [0, 0, 0]},
    SPILL: {"Q1": [4096, 1024, 2048], "Q6": [300, 100, 200, 400]},
    DIRECT: {"Q1": [0, 0, 0], "Q6": [0, 0, 0]},
}
PEAK_RSS_100 = {"columnar+parallel": 6000000000, "columnar+parallel_1thread": 5000000000,
                SWEEP2: 5500000000, SPILL: 1500000000, DIRECT: 6100000000}
# Every engine's answer, so each check passes; Q1's big sum can drift
ANSWERS = {"Q1": ["l_returnflag,sum_charge", "A,1000000000000.00", "N,2500.50"],
           "Q6": ["revenue", "123456789.12"]}


# One answer file per query in folder, as the runners name them.
def _write_answers(folder: Path, sf: str, answers: Dict[str, List[str]] = ANSWERS) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for query, lines in answers.items():
        (folder / f"{query}_sf{sf}.csv").write_text("\n".join(lines) + "\n")


# A parallel_stats file: one row per timed run.
def _stats(spills: Dict[str, List[int]], limit: int) -> str:
    text = STATS_HEADER
    for query, values in spills.items():
        for run, spilled in enumerate(values):
            text += f"{query},{run},18,100,1.000,{limit},{1 << 20},{spilled},{spilled},0,0,0.000\n"
    return text


# A results folder at SF=100 as the runner leaves it. skip leaves configs out entirely.
def _write_sf100(folder: Path, skip: Sequence[str] = ()) -> None:
    sf = "100"
    ours = LATENCIES_HEADER
    for query in ("Q1", "Q6"):
        for config, medians in OURS_100.items():
            if config not in skip:
                ours += _ours(sf, query, config, medians[query], pool=POOL_100)
        # The runner adds the engines' medians too; they are not ShilmanDB configs
        ours += f"{sf},{query},duckdb,{ENGINES_100['duckdb'][query]:.4f},,,,,,,,,,,,,\n"
    (folder / "latencies.csv").write_text(ours)

    for engine in ("duckdb", "postgres"):
        text = "sf,query,config,run,latency_ms\n"
        for config, medians in ENGINES_100.items():
            if config.startswith(engine) and config not in skip:
                text += "".join(_runs(sf, query, config, [ms]) for query, ms in medians.items())
        (folder / f"{engine}_latencies.csv").write_text(text)

    for config in OURS_100:
        if config in skip:
            continue
        results = folder / "shilmandb_results" / config
        _write_answers(results, sf)
        limit = LIMIT_256M if config == SPILL else LIMIT_4G
        (results / f"run_info_sf{sf}.csv").write_text(
            f"key,value\nsf,{sf}\npool_size,{POOL_100}\nexecution_mode,parallel\n"
            f"peak_rss_bytes,{PEAK_RSS_100[config]}\nmemory_limit,{limit}\n")
        if config in SPILLS_100:
            (folder / f"parallel_stats_{config}_sf{sf}.csv").write_text(_stats(SPILLS_100[config], limit))

    for config in ("duckdb", "duckdb_20g"):
        if config not in skip:
            _write_answers(folder / "duckdb_results" / config, sf)
    _write_answers(folder / "postgres_results" / "postgres", sf)


def test_bar_sf100_has_spill_direct_and_20g(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _write_sf100(tmp_path)
    assert mod.main(["--results-dir", str(tmp_path), "--sf", "100"]) == 0

    out = tmp_path / "bar_sf100.csv"
    with out.open(newline="") as f:
        assert next(csv.reader(f)) == BAR_HEADER + ["P2"] + EXTRA_HEADER
    rows = _read_bar(out)
    assert [r["query"] for r in rows] == ["Q1", "Q6"]
    q6 = rows[1]
    assert (q6["P"], q6["P1"], q6["P2"]) == ("400.0000", "3000.0000", "1600.0000")
    assert (q6["D"], q6["PG1"]) == ("100.0000", "9000.0000")
    assert q6["PS"] == "520.0000"
    assert q6["PS_over_P"] == f"{520.0 / 400.0:.6f}"
    # The lower middle of four runs, so the count stays a whole number
    assert q6["PS_spill_bytes"] == "200"
    assert q6["PD"] == "700.0000"
    assert q6["PD_over_P"] == f"{700.0 / 400.0:.6f}"
    assert q6["D20"] == "125.0000"
    assert q6["P_over_D20"] == f"{400.0 / 125.0:.6f}"
    q1 = rows[0]
    assert (q1["PS_spill_bytes"], q1["PS_over_P"], q1["P_over_D20"]) == ("2048", "1.300000", f"{1000.0 / 260.0:.6f}")
    # The SF=10 columns are unchanged
    assert q6["P_over_D"] == f"{400.0 / 100.0:.6f}"
    assert q6["check"] == q6["P_over_D"]
    captured = capsys.readouterr()
    assert "PS_spill_bytes" in captured.out
    assert captured.err == ""


def test_bar_reports_peak_rss_and_limit(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _write_sf100(tmp_path)
    # A config with no stats file and one with no run info leave those cells empty
    (tmp_path / "parallel_stats_columnar+parallel_1thread_sf100.csv").unlink()
    (tmp_path / "shilmandb_results" / DIRECT / "run_info_sf100.csv").unlink()
    assert mod.main(["--results-dir", str(tmp_path), "--sf", "100"]) == 0

    memory = tmp_path / "bar_sf100_memory.csv"
    with memory.open(newline="") as f:
        assert list(csv.reader(f)) == [
            MEMORY_HEADER,
            ["columnar+parallel", str(POOL_100), str(LIMIT_4G), "6000000000", "0"],
            ["columnar+parallel_1thread", str(POOL_100), str(LIMIT_4G), "5000000000", ""],
            [SWEEP2, str(POOL_100), str(LIMIT_4G), "5500000000", ""],
            [SPILL, str(POOL_100), str(LIMIT_256M), "1500000000", "4096"],
            [DIRECT, str(POOL_100), "", "", "0"],
        ]
    printed = capsys.readouterr().out
    # Printed after the bar table
    assert printed.index("peak_rss_bytes") > printed.index("PS_spill_bytes")
    assert "1500000000" in printed and str(LIMIT_256M) in printed


def test_bar_sf100_memory_file_follows_out(tmp_path: Path) -> None:
    _write_sf100(tmp_path)
    target = tmp_path / "elsewhere" / "bar.csv"
    assert mod.main(["--results-dir", str(tmp_path), "--sf", "100", "--out", str(target)]) == 0
    assert target.exists() and (tmp_path / "elsewhere" / "bar_memory.csv").exists()
    assert not (tmp_path / "bar_sf100.csv").exists()
    assert not (tmp_path / "bar_sf100_memory.csv").exists()


def test_bar_sf100_optional_configs_leave_empty_cells(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _write_sf100(tmp_path, skip=(SPILL, DIRECT, "duckdb_20g"))
    assert mod.main(["--results-dir", str(tmp_path), "--sf", "100"]) == 0
    for row in _read_bar(tmp_path / "bar_sf100.csv"):
        assert [row[name] for name in EXTRA_HEADER] == [""] * len(EXTRA_HEADER)
        assert row["P"] != ""
    err = capsys.readouterr().err
    for config in (SPILL, DIRECT, "duckdb_20g"):
        assert f"no {config} at SF=100" in err


def test_bar_sf100_spill_missing_for_one_query(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _write_sf100(tmp_path)
    lines = (tmp_path / "latencies.csv").read_text().splitlines(keepends=True)
    (tmp_path / "latencies.csv").write_text("".join(l for l in lines if not l.startswith(f"100,Q1,{SPILL},")))
    assert mod.main(["--results-dir", str(tmp_path), "--sf", "100"]) == 0
    q1, q6 = _read_bar(tmp_path / "bar_sf100.csv")
    assert (q1["PS"], q1["PS_over_P"], q1["PS_spill_bytes"]) == ("", "", "")
    assert q6["PS"] == "520.0000"
    assert f"no {SPILL} for Q1" in capsys.readouterr().err


def test_bar_sf100_still_needs_the_six_configs(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _write_sf100(tmp_path, skip=("postgres_1worker",))
    assert mod.main(["--results-dir", str(tmp_path), "--sf", "100"]) == 2
    assert "postgres_1worker for Q1" in capsys.readouterr().err
    assert not (tmp_path / "bar_sf100.csv").exists()
    assert not (tmp_path / "bar_sf100_memory.csv").exists()


def test_bar_sf100_zero_median_of_an_optional_config_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _write_sf100(tmp_path)
    text = (tmp_path / "duckdb_latencies.csv").read_text()
    (tmp_path / "duckdb_latencies.csv").write_text(text.replace("100,Q6,duckdb_20g,1,125.0000", "100,Q6,duckdb_20g,1,0.0000"))
    assert mod.main(["--results-dir", str(tmp_path), "--sf", "100"]) == 2
    assert "zero median for duckdb_20g for Q6" in capsys.readouterr().err


# Wrong answers to put in a candidate or reference file.
WRONG_Q1 = ["l_returnflag,sum_charge", "A,1000000000001.00", "N,2500.50"]   # 1e-12 relative: drift
WRONG_Q6 = ["revenue", "123456790.12"]                                    # off by 1: FAIL


@pytest.mark.parametrize("folder,query,lines,expect", [
    ("shilmandb_results/columnar+parallel", "Q6", WRONG_Q6,
     ["Q6 columnar+parallel against DuckDB: FAIL", "Q6 columnar+parallel against Postgres: FAIL"]),
    # Drift is not allowed at SF=100
    ("shilmandb_results/columnar+parallel_1thread", "Q1", WRONG_Q1,
     ["Q1 columnar+parallel_1thread against DuckDB: FAIL", "max rel 1e-12"]),
    ("shilmandb_results/columnar+parallel_2thread", "Q6", WRONG_Q6, ["Q6 columnar+parallel_2thread against DuckDB: FAIL"]),
    (f"shilmandb_results/{SPILL}", "Q6", None, [f"Q6 {SPILL} against DuckDB: SKIP", f"Q6 {SPILL} against Postgres: SKIP"]),
    # An empty file is named, whether it is the candidate or the reference
    (f"shilmandb_results/{DIRECT}", "Q1", [], [f"Q1 {DIRECT} against DuckDB: FAIL (couldn't read {{path}}: it's empty)"]),
    ("duckdb_results/duckdb", "Q1", [], ["Q1 columnar+parallel against DuckDB: FAIL (couldn't read {path}: it's empty)",
                                          "Q1 duckdb against Postgres: FAIL (couldn't read {path}: it's empty)"]),
    ("duckdb_results/duckdb_20g", "Q6", WRONG_Q6, ["Q6 duckdb_20g against Postgres: FAIL"]),
    ("duckdb_results/duckdb", "Q6", WRONG_Q6, ["Q6 duckdb against Postgres: FAIL"]),
    # A wrong reference fails every config checked against it
    ("postgres_results/postgres", "Q6", WRONG_Q6, ["Q6 columnar+parallel against Postgres: FAIL", "Q6 duckdb against Postgres: FAIL"]),
])
def test_bar_sf100_refuses_an_answer_that_is_not_pass(tmp_path: Path, capsys: pytest.CaptureFixture, folder: str,
                                                     query: str, lines: Optional[List[str]], expect: List[str]) -> None:
    _write_sf100(tmp_path)
    path = tmp_path / folder / f"{query}_sf100.csv"
    if lines is None:
        path.unlink()                               # a missing file of a reported config
    else:
        path.write_text("\n".join(lines) + "\n" if lines else "")   # [] = an empty file
    assert mod.main(["--results-dir", str(tmp_path), "--sf", "100"]) == 1
    captured = capsys.readouterr()
    for text in expect:
        assert text.replace("{path}", str(path)) in captured.err
    assert "nothing was written" in captured.err
    assert captured.out == ""
    assert not (tmp_path / "bar_sf100.csv").exists()
    assert not (tmp_path / "bar_sf100_memory.csv").exists()


def test_bar_sf100_skips_checks_of_configs_it_does_not_report(tmp_path: Path) -> None:
    _write_sf100(tmp_path, skip=(SPILL,))
    # Answers left from another run of an unreported config are not checked
    _write_answers(tmp_path / "shilmandb_results" / SPILL, "100", {"Q6": WRONG_Q6})
    _write_answers(tmp_path / "duckdb_results" / "duckdb_1thread", "100", {"Q6": WRONG_Q6})
    assert mod.main(["--results-dir", str(tmp_path), "--sf", "100"]) == 0


# What the report printed and wrote for _write_sf10_extras before SF=100 existed.
SF10_BAR_CSV = (
    "query,P,P1,D,D1,PG,PG1,P_over_PG,P_over_D,P1_over_D1,ours_speedup,duckdb_speedup,check,P_over_PG1,P1_over_PG1,P2\r\n"
    "Q1,100.0000,900.0000,34.2000,393.0000,2857.4000,7153.6000,0.034997,2.923977,2.290076,9.000000,11.491228,"
    "2.923977,0.013979,0.125811,480.0000\r\n"
    "Q6,40.0000,300.0000,10.4000,112.8000,1127.4000,1312.3000,0.035480,3.846154,2.659574,7.500000,10.846154,"
    "3.846154,0.030481,0.228606,160.0000\r\n")
SF10_STDOUT = (
    "query         P        P1        D        D1         PG        PG1  P_over_PG  P_over_D  P1_over_D1  "
    "ours_speedup  duckdb_speedup     check  P_over_PG1  P1_over_PG1        P2\n"
    "   Q1  100.0000  900.0000  34.2000  393.0000  2857.4000  7153.6000   0.034997  2.923977    2.290076      "
    "9.000000       11.491228  2.923977    0.013979     0.125811  480.0000\n"
    "   Q6   40.0000  300.0000  10.4000  112.8000  1127.4000  1312.3000   0.035480  3.846154    2.659574      "
    "7.500000       10.846154  3.846154    0.030481     0.228606  160.0000\n"
    "  -> {OUT}\n")


# The SF=10 fixture plus SF=100-style files, and answers that disagree.
def _write_sf10_extras(folder: Path) -> None:
    _write_fixture(folder)
    with (folder / "latencies.csv").open("a") as f:
        f.write(_ours("10", "Q1", SPILL, 130.0) + _ours("10", "Q6", DIRECT, 70.0))
    with (folder / "duckdb_latencies.csv").open("a") as f:
        f.write(_runs("10", "Q6", "duckdb_20g", [12.5]))
    _write_answers(folder / "shilmandb_results" / "columnar+parallel", "10", {"Q6": WRONG_Q6})
    _write_answers(folder / "duckdb_results" / "duckdb", "10")
    (folder / "shilmandb_results" / "columnar+parallel" / "run_info_sf10.csv").write_text(
        "key,value\npeak_rss_bytes,123\nmemory_limit,0\n")
    (folder / f"parallel_stats_{SPILL}_sf10.csv").write_text(_stats({"Q1": [5, 6, 7]}, LIMIT_256M))


def test_bar_sf10_output_is_unchanged(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _write_sf10_extras(tmp_path)
    assert mod.main(["--results-dir", str(tmp_path), "--sf", "10"]) == 0
    assert (tmp_path / "bar_sf10.csv").read_bytes() == SF10_BAR_CSV.encode()
    captured = capsys.readouterr()
    assert captured.out == SF10_STDOUT.replace("{OUT}", str(tmp_path / "bar_sf10.csv"))
    assert captured.err == ""
    assert sorted(p.name for p in tmp_path.glob("bar_*")) == ["bar_sf10.csv"]
