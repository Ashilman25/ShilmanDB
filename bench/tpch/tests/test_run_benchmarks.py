"""Unit tests for bench/tpch/run_benchmarks.py helpers."""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Optional

import pytest

SPEC_PATH = Path(__file__).resolve().parents[1] / "run_benchmarks.py"
spec = importlib.util.spec_from_file_location("run_benchmarks", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)

sys.modules["run_benchmarks"] = mod
spec.loader.exec_module(mod)

LATENCIES_HEADER = [
    "sf", "query", "config", "median_latency_ms", "rows", "hits", "misses",
    "visits", "page_hit_rate", "reads", "read_ms", "us_per_read", "policy_ms",
    "pool_size", "miss_penalty_us", "direct_io", "eviction",
]
HIT_RATES_HEADER = [
    "sf", "config", "pool_size", "pool_fraction", "miss_penalty_us", "direct_io",
    "hit_rate", "raw_hit_rate", "improvement_over_lru", "eviction",
]


# parse_shilmandb_latencies
def test_parse_shilmandb_latencies_medians(tmp_path: Path) -> None:
    csv_path = tmp_path / "latencies.csv"
    csv_path.write_text(
        "query,run,rows,latency_ms,hits,misses\n"
        "Q1,0,4,10.0,100,5\n"
        "Q1,1,4,12.0,102,6\n"
        "Q1,2,4,14.0,104,7\n"
        "Q6,0,1,5.0,50,0\n"
        "Q6,1,1,7.0,55,1\n"
        "Q6,2,1,9.0,60,2\n"
    )
    stats = mod.parse_shilmandb_latencies(csv_path, sf="0.01", config="baseline")
    assert set(stats) == {"Q1", "Q6"}
    assert stats["Q1"].median_latency_ms == 12.0
    assert stats["Q1"].median_hits == 102
    assert stats["Q1"].median_misses == 6
    assert stats["Q1"].rows == 4
    assert stats["Q1"].sf == "0.01"
    assert stats["Q1"].config == "baseline"
    assert stats["Q6"].median_latency_ms == 7.0


# write_latencies_csv
def test_write_latencies_csv_handles_missing_hits(tmp_path: Path) -> None:
    records = {
        ("0.01", "baseline", "Q1"): mod.QueryStats(
            sf="0.01", config="baseline", query="Q1",
            median_latency_ms=123.4567, rows=4,
            median_hits=100, median_misses=5,
        ),
        ("0.01", "sqlite_disk", "Q1"): mod.QueryStats(
            sf="0.01", config="sqlite_disk", query="Q1",
            median_latency_ms=12.34, rows=None,
            median_hits=None, median_misses=None,
        ),
    }
    out = tmp_path / "latencies.csv"
    mod.write_latencies_csv(out, records)
    with out.open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == LATENCIES_HEADER
    assert len(rows) == 3
    shilmandb_row = next(r for r in rows[1:] if r[2] == "baseline")
    assert shilmandb_row[3] == "123.4567"
    assert shilmandb_row[5] == "100"
    sqlite_row = next(r for r in rows[1:] if r[2] == "sqlite_disk")
    assert sqlite_row[4] == ""
    assert sqlite_row[5] == ""
    assert sqlite_row[6] == ""


# write_hit_rates_csv
def _stats(sf, cfg, q, hits, misses, visits=None):
    return mod.QueryStats(
        sf=sf, config=cfg, query=q, median_latency_ms=1.0, rows=1,
        median_hits=hits, median_misses=misses, median_visits=visits,
    )


def _read_dicts(path: Path) -> list:
    with path.open() as f:
        return list(csv.DictReader(f))


def test_write_hit_rates_csv_improvement_vs_baseline(tmp_path: Path) -> None:
    records = {
        ("0.01", "baseline", "Q1"): _stats("0.01", "baseline", "Q1", 90, 10, 20),
        ("0.01", "baseline", "Q6"): _stats("0.01", "baseline", "Q6", 80, 20, 40),
        ("0.01", "+learned", "Q1"): _stats("0.01", "+learned", "Q1", 95, 5, 20),
        ("0.01", "+learned", "Q6"): _stats("0.01", "+learned", "Q6", 85, 15, 40),
        ("0.01", "sqlite_disk", "Q1"): mod.QueryStats(
            sf="0.01", config="sqlite_disk", query="Q1",
            median_latency_ms=1.0, rows=None,
            median_hits=None, median_misses=None,
        ),
    }
    pool_sizes = {
        ("0.01", "baseline"): 128,
        ("0.01", "+learned"): 128,
        ("0.01", "sqlite_disk"): "",
    }
    out = tmp_path / "hit_rates.csv"
    mod.write_hit_rates_csv(out, records, pool_sizes)
    with out.open() as f:
        header = next(csv.reader(f))
    assert header == HIT_RATES_HEADER
    by_cfg = {r["config"]: r for r in _read_dicts(out)}
    baseline_row, learned_row, sqlite_row = by_cfg["baseline"], by_cfg["+learned"], by_cfg["sqlite_disk"]
    assert baseline_row["hit_rate"] == "0.5000"
    assert baseline_row["raw_hit_rate"] == "0.8500"
    assert baseline_row["improvement_over_lru"] == ""
    assert baseline_row["pool_fraction"] == "0.0884"      # 128 / 1448
    assert learned_row["hit_rate"] == "0.6667"
    assert learned_row["raw_hit_rate"] == "0.9000"
    assert learned_row["improvement_over_lru"] == "0.1667"
    assert sqlite_row["hit_rate"] == ""
    assert sqlite_row["raw_hit_rate"] == ""
    assert sqlite_row["improvement_over_lru"] == ""
    assert sqlite_row["pool_fraction"] == ""


# resolve_sfs / resolve_configs
def test_resolve_sfs_defaults_to_all() -> None:
    assert mod.resolve_sfs(None) == ["0.01", "0.1", "1.0"]
    assert mod.resolve_sfs(["all"]) == ["0.01", "0.1", "1.0"]
    assert mod.resolve_sfs(["0.1", "0.01"]) == ["0.01", "0.1"]


def test_sf_all_is_still_three(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--sf", "all"])
    assert mod.resolve_sfs(mod.parse_args().sf) == ["0.01", "0.1", "1.0"]


def test_sf10_accepted_when_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--sf", "10"])
    assert mod.resolve_sfs(mod.parse_args().sf) == ["10"]
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--sf", "10", "--sf", "0.01"])
    assert mod.resolve_sfs(mod.parse_args().sf) == ["0.01", "10"]
    assert mod.POOL_SIZES["10"] == 8192


def test_resolve_configs_rejects_unknown() -> None:
    assert mod.resolve_configs("all") == mod.ALL_CONFIG_NAMES
    assert mod.resolve_configs("baseline,+learned") == ["baseline", "+learned"]
    with pytest.raises(SystemExit):
        mod.resolve_configs("baseline,nonsense")


# build_command
def test_build_command_includes_expected_flags(tmp_path: Path) -> None:
    cmd = mod.build_command(
        binary=Path("/bin/fake"), sf="0.01", config_name="+learned",
        data_dir=tmp_path, results_dir=tmp_path, db_file=tmp_path / "db",
        pool_size=128, warmup_runs=1, timed_runs=5,
    )
    assert "--sf" in cmd
    assert "--use-learned-join" in cmd
    assert "--use-learned-eviction" in cmd
    assert "--eviction-model-version" in cmd
    idx = cmd.index("--eviction-model-version")
    assert cmd[idx + 1] == "v2"


def test_build_command_rejects_unknown_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        mod.build_command(
            binary=Path("/bin/fake"), sf="0.01", config_name="bogus",
            data_dir=tmp_path, results_dir=tmp_path, db_file=tmp_path / "db",
            pool_size=128, warmup_runs=1, timed_runs=5,
        )


def test_build_command_default_has_no_new_flags(tmp_path: Path) -> None:
    cmd = mod.build_command(
        binary=Path("/bin/fake"), sf="0.01", config_name="baseline",
        data_dir=tmp_path, results_dir=tmp_path, db_file=tmp_path / "db",
        pool_size=128, warmup_runs=1, timed_runs=5,
    )
    for flag in ("--miss-penalty-us", "--direct-io", "--policy-timing"):
        assert flag not in cmd


def test_build_command_passes_extra_flags(tmp_path: Path) -> None:
    cmd = mod.build_command(
        binary=Path("/bin/fake"), sf="0.01", config_name="baseline",
        data_dir=tmp_path, results_dir=tmp_path, db_file=tmp_path / "db",
        pool_size=128, warmup_runs=1, timed_runs=5,
        extra_flags=["--miss-penalty-us", "100", "--direct-io"],
    )
    assert cmd[-3:] == ["--miss-penalty-us", "100", "--direct-io"]


# parse_shilmandb_latencies: new and old formats
def test_parse_latencies_new_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "latencies.csv"
    csv_path.write_text(
        "query,run,rows,latency_ms,hits,misses,visits,reads,read_ms\n"
        "Q1,0,4,10.0,100,5,20,5,0.5\n"
        "Q1,1,4,12.0,102,6,22,6,0.6\n"
        "Q1,2,4,14.0,104,7,24,7,0.7\n"
    )
    qs = mod.parse_shilmandb_latencies(csv_path, sf="0.01", config="baseline")["Q1"]
    assert qs.median_visits == 22
    assert qs.median_reads == 6
    assert qs.median_read_ms == pytest.approx(0.6)
    assert qs.page_hit_rate == pytest.approx(1 - 6 / 22)
    assert qs.us_per_read == pytest.approx(0.6 * 1000 / 6)


def test_parse_latencies_old_format(tmp_path: Path) -> None:
    csv_path = tmp_path / "latencies.csv"
    csv_path.write_text(
        "query,run,rows,latency_ms,hits,misses\n"
        "Q1,0,4,10.0,100,5\n"
        "Q1,1,4,12.0,102,6\n"
    )
    qs = mod.parse_shilmandb_latencies(csv_path, sf="0.01", config="baseline")["Q1"]
    assert qs.median_misses == 6
    assert qs.median_visits is None
    assert qs.median_reads is None
    assert qs.median_read_ms is None
    assert qs.policy_ms is None
    assert qs.page_hit_rate is None
    assert qs.us_per_read is None


# pool_for_fraction / config_label
def test_pool_for_fraction() -> None:
    assert mod.DB_PAGES == {"0.01": 1448, "0.1": 14417, "1.0": 144032, "10": 1439816}
    assert mod.pool_for_fraction("0.01", 0.05) == 72
    assert mod.pool_for_fraction("1.0", 0.01) == 1440
    assert mod.pool_for_fraction("0.01", 0.001) == 16
    assert mod.pool_for_fraction("0.01", 1.0) == 1448


def test_pool_for_fraction_sf10() -> None:
    assert mod.pool_for_fraction("10", 0.25) == 359954
    assert mod.pool_for_fraction("10", 0.01) == 14398  # 14,398.16 rounds down
    assert mod.pool_for_fraction("10", 0.001) == 1440  # 1,439.816 rounds up


def test_every_known_sf_has_db_pages() -> None:
    # SF=100 runs only the parallel configs, which never use DB_PAGES
    assert mod.PARALLEL_ONLY_SCALE_FACTORS == ["100"]
    assert set(mod.KNOWN_SCALE_FACTORS) - set(mod.PARALLEL_ONLY_SCALE_FACTORS) <= set(mod.DB_PAGES)


def test_pool_for_fraction_unknown_sf_needs_db_pages() -> None:
    with pytest.raises(ValueError, match="DB_PAGES has no entry for SF=100 yet"):
        mod.pool_for_fraction("100", 0.25)


def test_config_label() -> None:
    assert mod.config_label("baseline") == "baseline"
    assert mod.config_label("baseline", 100) == "baseline_pen100"
    assert mod.config_label("baseline", 0, True) == "baseline_direct"
    assert mod.config_label("baseline", fraction=0.25) == "baseline_frac0.25"
    assert mod.config_label("+index", 10, True) == "+index_pen10_direct"
    assert mod.config_label("baseline", 10, True, 0.5) == "baseline_frac0.5_pen10_direct"
    assert mod.config_label("baseline", fraction=1.0) == "baseline_frac1.0"
    # parse_label undoes it
    opts = mod.parse_label(mod.config_label("+learned", 100, True, 0.25))
    assert (opts.base, opts.fraction, opts.miss_penalty_us, opts.direct_io) == ("+learned", 0.25, 100, True)
    opts = mod.parse_label("baseline_pool512")
    assert (opts.base, opts.fraction, opts.miss_penalty_us, opts.direct_io) == ("baseline_pool512", None, 0, False)


# read_run_info
def test_read_run_info(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = tmp_path / "run_info.csv"
    path.write_text(
        "key,value\n"
        "sf,0.01\n"
        "pool_size,128\n"
        "db_pages,1448\n"
        "direct_io,false\n"
        "us_per_read,\n"
    )
    info = mod.read_run_info(path)
    assert info["sf"] == "0.01"
    assert info["pool_size"] == "128"
    assert info["db_pages"] == "1448"
    assert info["direct_io"] == "false"
    assert info["us_per_read"] == ""

    # db_pages is checked against the expected table size: a warning, never a failure
    mod.warn_on_db_pages_mismatch(info, "0.01")
    assert capsys.readouterr().err == ""
    mod.warn_on_db_pages_mismatch({"db_pages": "1500"}, "0.01")
    assert "db_pages=1500" in capsys.readouterr().err
    mod.warn_on_db_pages_mismatch({}, "0.01")
    assert capsys.readouterr().err == ""


def test_read_run_info_missing(tmp_path: Path) -> None:
    assert mod.read_run_info(tmp_path / "nope.csv") == {}


# hit rates: page level
def _check_matching_baselines(tmp_path: Path) -> None:
    pen = mod.config_label("baseline", 10)
    learned_pen = mod.config_label("+learned", 10)
    frac = mod.config_label("baseline", fraction=0.25)
    records = {}
    for cfg, misses in [("baseline", 10), (pen, 20), (learned_pen, 15), (frac, 30), ("+learned", 5)]:
        records[("0.01", cfg, "Q1")] = _stats("0.01", cfg, "Q1", 100, misses, 100)
    pool_sizes = {("0.01", c): 128 for c in ("baseline", pen, learned_pen, "+learned")}
    pool_sizes[("0.01", frac)] = 362
    out = tmp_path / "hit_rates.csv"
    mod.write_hit_rates_csv(out, records, pool_sizes)
    by_cfg = {r["config"]: r for r in _read_dicts(out)}
    assert by_cfg["+learned"]["improvement_over_lru"] == "0.0500"       # vs baseline
    assert by_cfg[learned_pen]["improvement_over_lru"] == "0.0500"      # vs baseline_pen10, not baseline
    assert by_cfg[pen]["improvement_over_lru"] == ""                    # it is its own baseline
    assert by_cfg[frac]["improvement_over_lru"] == ""
    assert by_cfg[frac]["pool_fraction"] == "0.2500"                    # the requested fraction
    assert by_cfg[pen]["miss_penalty_us"] == "10"
    assert by_cfg[pen]["direct_io"] == "false"


def test_hit_rates_page_level(tmp_path: Path) -> None:
    records = {
        ("0.01", "baseline", "Q1"): _stats("0.01", "baseline", "Q1", 900, 10, 20),
        ("0.01", "baseline", "Q6"): _stats("0.01", "baseline", "Q6", 700, 30, 40),
        # an older binary: no visits, so no page-level rate
        ("0.01", "+index", "Q1"): _stats("0.01", "+index", "Q1", 900, 10),
        ("0.01", "+index", "Q6"): _stats("0.01", "+index", "Q6", 700, 30, 40),
    }
    pool_sizes = {("0.01", "baseline"): 128, ("0.01", "+index"): 128}
    out = tmp_path / "hit_rates.csv"
    mod.write_hit_rates_csv(out, records, pool_sizes)
    by_cfg = {r["config"]: r for r in _read_dicts(out)}
    base = by_cfg["baseline"]
    assert float(base["hit_rate"]) == pytest.approx(1 - 40 / 60, abs=1e-4)
    assert float(base["raw_hit_rate"]) == pytest.approx(1600 / 1640, abs=1e-4)
    assert by_cfg["+index"]["hit_rate"] == ""
    assert by_cfg["+index"]["improvement_over_lru"] == ""
    assert by_cfg["+index"]["raw_hit_rate"] != ""

    _check_matching_baselines(tmp_path)


# write_latencies_csv: new header and columns
def test_write_latencies_new_header(tmp_path: Path) -> None:
    records = {
        ("0.01", "baseline_pen10", "Q1"): mod.QueryStats(
            sf="0.01", config="baseline_pen10", query="Q1",
            median_latency_ms=12.5, rows=4, median_hits=100, median_misses=5,
            median_visits=20, median_reads=5, median_read_ms=0.5,
            policy_ms=1.25, pool_size=128, miss_penalty_us=10, direct_io=False,
        ),
        ("0.01", "baseline", "Q1"): mod.QueryStats(
            sf="0.01", config="baseline", query="Q1",
            median_latency_ms=12.0, rows=4, median_hits=100, median_misses=5,
        ),
    }
    out = tmp_path / "latencies.csv"
    mod.write_latencies_csv(out, records)
    with out.open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == LATENCIES_HEADER
    full = dict(zip(LATENCIES_HEADER, next(r for r in rows[1:] if r[2] == "baseline_pen10")))
    assert full["visits"] == "20"
    assert full["page_hit_rate"] == "0.7500"
    assert full["reads"] == "5"
    assert full["read_ms"] == "0.5000"
    assert full["us_per_read"] == "100.0000"
    assert full["policy_ms"] == "1.2500"
    assert full["pool_size"] == "128"
    assert full["miss_penalty_us"] == "10"
    assert full["direct_io"] == "false"
    old = dict(zip(LATENCIES_HEADER, next(r for r in rows[1:] if r[2] == "baseline")))
    for col in LATENCIES_HEADER[7:]:
        assert old[col] == ""


# workload labels, flags and the run plan
def test_config_label_workload() -> None:
    assert mod.config_label("baseline", workload="stream", seed=1) == "baseline_stream_s1"
    assert mod.config_label("baseline", 100, False, 0.25, "stream", 1) == "baseline_stream_s1_frac0.25_pen100"
    assert mod.config_label("+index", 0, True, None, "lookup", 7) == "+index_lookup_s7_direct"
    # single keeps today's labels, whatever the seed
    assert mod.config_label("baseline", fraction=0.25) == "baseline_frac0.25"
    assert mod.config_label("baseline", seed=5) == "baseline"


def test_parse_label_workload_round_trip() -> None:
    opts = mod.parse_label("+index_mixed_s7_frac0.5_pen10_direct")
    assert (opts.base, opts.workload, opts.seed) == ("+index", "mixed", 7)
    assert (opts.fraction, opts.miss_penalty_us, opts.direct_io) == (0.5, 10, True)

    opts = mod.parse_label("baseline_pen100")
    assert (opts.base, opts.workload, opts.seed) == ("baseline", "single", None)
    assert opts.miss_penalty_us == 100

    label = mod.config_label("baseline", 100, True, 0.25, "stream", 3)
    opts = mod.parse_label(label)
    assert (opts.base, opts.workload, opts.seed, opts.fraction) == ("baseline", "stream", 3, 0.25)


def test_workload_flags_single_is_empty() -> None:
    assert mod.workload_flags("single", 1, None) == []
    assert mod.workload_flags("stream", 3, None) == ["--workload", "stream", "--seed", "3"]
    assert mod.workload_flags("lookup", 1, 16) == [
        "--workload", "lookup", "--seed", "1", "--stream-length", "16",
    ]


def test_fraction_config() -> None:
    assert mod.fraction_config("single") == "baseline"
    assert mod.fraction_config("stream") == "baseline"
    assert mod.fraction_config("lookup") == "+index"
    assert mod.fraction_config("mixed") == "+index"


def test_planned_runs_lookup_runs_on_every_config() -> None:
    runs = mod.planned_runs(["0.01"], ["baseline", "+index"], ["lookup", "mixed"], [0], [])
    assert runs == [
        mod.RunSpec("0.01", c, w, 128, 0, None)
        for w in ("lookup", "mixed")
        for c in ("baseline", "+index")
    ]


def test_planned_runs_single_matches_today() -> None:
    runs = mod.planned_runs(["0.01"], ["baseline", "+index"], ["single"], [0, 100], [0.5])
    assert runs == [
        mod.RunSpec("0.01", "baseline", "single", 128, 0, None),
        mod.RunSpec("0.01", "baseline", "single", 128, 100, None),
        mod.RunSpec("0.01", "+index", "single", 128, 0, None),
        mod.RunSpec("0.01", "+index", "single", 128, 100, None),
        mod.RunSpec("0.01", "baseline", "single", 724, 0, 0.5),
        mod.RunSpec("0.01", "baseline", "single", 724, 100, 0.5),
    ]


def test_planned_runs_every_config_and_fraction_configs() -> None:
    runs = mod.planned_runs(["0.01"], ["baseline", "+index"], ["stream", "lookup"], [0], [0.25])
    assert runs == [
        mod.RunSpec("0.01", "baseline", "stream", 128, 0, None),
        mod.RunSpec("0.01", "+index", "stream", 128, 0, None),
        mod.RunSpec("0.01", "baseline", "stream", 362, 0, 0.25),
        mod.RunSpec("0.01", "baseline", "lookup", 128, 0, None),
        mod.RunSpec("0.01", "+index", "lookup", 128, 0, None),
        mod.RunSpec("0.01", "+index", "lookup", 362, 0, 0.25),
    ]


# stream CSVs
STREAM_TOTALS_HEADER = "pass,warmup,items,latency_ms,hits,misses,visits,reads,read_ms,query_misses,lookup_misses\n"
STREAM_HEADER = "pass,warmup,seq,kind,query,rows,latency_ms,hits,misses,visits,reads,read_ms,trace_start,trace_end\n"


def test_parse_stream_totals_skips_warmup(tmp_path: Path) -> None:
    path = tmp_path / "stream_totals.csv"
    path.write_text(
        STREAM_TOTALS_HEADER
        + "0,1,16,99.0,1,999,999,999,9.0,999,0\n"
        + "1,0,16,10.0,50,100,120,100,1.0,100,0\n"
        + "2,0,16,30.0,50,100,120,100,3.0,100,0\n"
        + "3,0,16,20.0,50,100,120,100,2.0,100,0\n"
    )
    totals = mod.parse_stream_totals(path)
    assert totals["median_total_ms"] == 20.0
    assert totals["misses"] == 100
    assert totals["timed_passes"] == 3
    assert totals["items"] == 16
    assert totals["visits"] == 120
    assert totals["reads"] == 100
    assert totals["read_ms"] == 2.0
    assert totals["query_misses"] == 100
    assert totals["lookup_misses"] == 0


def test_parse_stream_items_per_tag(tmp_path: Path) -> None:
    path = tmp_path / "stream.csv"
    path.write_text(
        STREAM_HEADER
        + "0,1,0,tpch,Q6,1,5.0,1,999,999,999,0.5,,\n"      # warmup: left out
        + "1,0,0,tpch,Q6,1,4.0,1,10,12,10,0.4,,\n"
        + "1,0,1,tpch,Q1,4,9.0,1,50,60,50,0.9,,\n"
        + "1,0,2,tpch,Q6,1,6.0,1,30,32,30,0.6,,\n"
        + "2,0,0,tpch,Q6,1,4.0,1,10,12,10,0.4,,\n"
        + "2,0,1,tpch,Q1,4,9.0,1,50,60,50,0.9,,\n"
        + "2,0,2,tpch,Q6,1,6.0,1,30,32,30,0.6,,\n"
    )
    items = mod.parse_stream_items(path)
    assert list(items) == ["Q1", "Q6"]
    assert items["Q6"]["items"] == 4
    assert items["Q6"]["median_misses"] == 20
    assert items["Q6"]["median_visits"] == 22
    assert items["Q6"]["median_latency_ms"] == pytest.approx(5.0)
    assert items["Q1"]["items"] == 2

    rows = [
        mod.WorkloadQueryRow("0.01", "baseline_stream_s1", "stream", tag, int(v["items"]),
                             v["median_latency_ms"], v["median_misses"], v["median_visits"])
        for tag, v in items.items()
    ]
    out = tmp_path / "workload_queries.csv"
    mod.write_workload_queries_csv(out, rows)
    lines = _read_dicts(out)
    assert list(lines[0]) == ["sf", "config", "workload", "query", "items",
                              "median_latency_ms", "median_misses", "median_visits"]
    q6 = next(r for r in lines if r["query"] == "Q6")
    assert (q6["items"], q6["median_latency_ms"], q6["median_misses"]) == ("4", "5.0000", "20")


# workloads.csv
WORKLOADS_HEADER = [
    "sf", "config", "workload", "seed", "items", "timed_passes", "pool_size", "pool_fraction",
    "miss_penalty_us", "direct_io", "median_total_ms", "misses", "visits", "page_hit_rate",
    "reads", "read_ms", "us_per_read", "query_misses", "lookup_misses", "policy_ms", "correct", "eviction",
]


def _workload_stats(config: str, pool_size: int, **overrides) -> "mod.WorkloadStats":
    fields = dict(
        sf="0.01", config=config, workload="stream", seed=1, items=16, timed_passes=3,
        pool_size=pool_size, miss_penalty_us=0, direct_io=False, median_total_ms=12.5,
        misses=40, visits=100, reads=40, read_ms=0.02, query_misses=40, lookup_misses=0,
    )
    fields.update(overrides)
    return mod.WorkloadStats(**fields)


def test_write_workloads_csv(tmp_path: Path) -> None:
    records = [
        _workload_stats("baseline_stream_s1", 724, correct="16/16"),
        _workload_stats("baseline_stream_s1_frac0.25", 362, policy_ms=1.5),
    ]
    out = tmp_path / "workloads.csv"
    mod.write_workloads_csv(out, records)
    with out.open() as f:
        assert next(csv.reader(f)) == WORKLOADS_HEADER

    plain, frac = _read_dicts(out)
    assert plain["page_hit_rate"] == "0.6000"
    assert plain["pool_fraction"] == "0.5000"           # 724 / 1448, no fraction in the label
    assert plain["us_per_read"] == "0.5000"
    assert plain["median_total_ms"] == "12.5000"
    assert plain["direct_io"] == "false"
    assert plain["policy_ms"] == ""
    assert plain["correct"] == "16/16"
    assert frac["pool_fraction"] == "0.2500"            # the label's fraction
    assert frac["policy_ms"] == "1.5000"
    assert frac["correct"] == ""


# a binary that can't run the workload
def test_workload_run_with_failing_binary_returns_none(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    binary = tmp_path / "old_load_tpch"
    binary.write_text("#!/bin/sh\necho 'usage: load_tpch [options]' >&2\nexit 1\n")
    binary.chmod(0o755)
    stats = mod.run_workload_config(
        binary=binary, sf="0.01", config_name="baseline", data_dir=tmp_path,
        output_dir=tmp_path / "out", pool_size=128, warmup_runs=1, timed_runs=2,
        timeout_seconds=30, label="baseline_stream_s1",
        extra_flags=mod.workload_flags("stream", 1, 16),
    )
    assert stats is None
    assert "usage: load_tpch" in capsys.readouterr().err


# a call that runs only workloads must not touch the single-query files or run SQLite
def test_workload_only_call_leaves_single_files_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      capsys: pytest.CaptureFixture) -> None:
    out = tmp_path / "out"
    out.mkdir()
    for name in ("latencies.csv", "hit_rates.csv", "correctness.csv"):
        (out / name).write_text("marker\n")

    sqlite_calls: list = []
    monkeypatch.setattr(mod, "run_sqlite_benchmarks", lambda *a, **k: sqlite_calls.append(a) or True)
    monkeypatch.setattr(mod, "run_workload_config",
                        lambda **k: _workload_stats(k["label"], k["pool_size"]))
    monkeypatch.setattr(mod, "workload_query_rows", lambda *a, **k: [])
    monkeypatch.setattr(sys, "argv", [
        "run_benchmarks.py", "--sf", "0.01", "--configs", "baseline,sqlite_disk",
        "--workloads", "stream", "--skip-correctness", "--output-dir", str(out)])

    assert mod.main() == 0

    for name in ("latencies.csv", "hit_rates.csv", "correctness.csv"):
        assert (out / name).read_text() == "marker\n"
    assert sqlite_calls == []
    assert len(_read_dicts(out / "workloads.csv")) == 1
    text = capsys.readouterr().out
    assert "SQLite configs only run with the single workload" in text
    assert "latencies" not in text.split("Done")[-1]


# eviction policies
def test_config_label_eviction_suffix() -> None:
    assert mod.config_label("+index", 100, False, None, "stream", 1, eviction="lirs") == "+index_stream_s1_lirs_pen100"
    assert mod.config_label("baseline", eviction="lru") == "baseline"
    assert mod.config_label("baseline", 0, False, 0.25, eviction="arc") == "baseline_arc_frac0.25"


def test_parse_label_eviction() -> None:
    o = mod.parse_label("+index_stream_s1_lirs_pen100")
    assert (o.base, o.eviction, o.workload, o.seed, o.miss_penalty_us) == ("+index", "lirs", "stream", 1, 100)
    assert mod.parse_label("baseline_frac0.25").eviction == "lru"
    assert mod.parse_label("baseline_clock").base == "baseline"


def test_planned_runs_skip_learned_for_other_policies() -> None:
    runs = mod.planned_runs(["0.01"], ["baseline", "+learned"], ["single"], [0], [], evictions=["lru", "arc"])
    assert [(r.base, r.eviction) for r in runs] == [("baseline", "lru"), ("baseline", "arc"), ("+learned", "lru")]


def test_planned_runs_keep_learned_without_lru() -> None:
    runs = mod.planned_runs(["0.01"], ["baseline", "+learned"], ["single"], [0, 100], [], evictions=["arc", "lirs"])
    assert [(r.base, r.eviction, r.penalty) for r in runs] == [
        ("baseline", "arc", 0), ("baseline", "arc", 100), ("baseline", "lirs", 0), ("baseline", "lirs", 100),
        ("+learned", "lru", 0), ("+learned", "lru", 100)]


def test_option_flags_eviction() -> None:
    assert mod.option_flags(eviction="arc") == ["--eviction-policy", "arc"]
    assert mod.option_flags(eviction="lru") == []
    assert mod.option_flags() == []


def test_eviction_policies_arg_rejects_unknown() -> None:
    import argparse
    assert mod.eviction_policy_list("lru,arc,lirs") == ["lru", "arc", "lirs"]
    for bad in ("foo", "learned-v2", "lru,learned-v1"):
        with pytest.raises(argparse.ArgumentTypeError):
            mod.eviction_policy_list(bad)


def test_improvement_over_lru_against_lru_baseline(tmp_path: Path) -> None:
    records = {
        ("0.01", "baseline", "Q1"): _stats("0.01", "baseline", "Q1", 100, 40, 100),
        ("0.01", "baseline_lirs", "Q1"): _stats("0.01", "baseline_lirs", "Q1", 100, 30, 100),
    }
    out = tmp_path / "hit_rates.csv"
    mod.write_hit_rates_csv(out, records, {("0.01", "baseline"): 128, ("0.01", "baseline_lirs"): 128})
    by_cfg = {r["config"]: r for r in _read_dicts(out)}
    assert by_cfg["baseline"]["improvement_over_lru"] == ""
    assert by_cfg["baseline_lirs"]["improvement_over_lru"] == "0.1000"


def test_eviction_column(tmp_path: Path) -> None:
    lat = tmp_path / "lat.csv"
    lat.write_text("query,run,rows,latency_ms,hits,misses\nQ1,0,4,1.0,10,5\n")
    qs = mod.parse_shilmandb_latencies(lat, "0.01", "baseline_arc")["Q1"]
    assert qs.eviction == ""
    qs.eviction = "arc"
    plain = _stats("0.01", "baseline", "Q1", 100, 40, 100)
    out = tmp_path / "latencies.csv"
    mod.write_latencies_csv(out, {("0.01", "baseline_arc", "Q1"): qs, ("0.01", "baseline", "Q1"): plain})
    rows = {r["config"]: r for r in _read_dicts(out)}
    assert list(rows["baseline_arc"])[-1] == "eviction"
    assert rows["baseline_arc"]["eviction"] == "arc"
    assert rows["baseline"]["eviction"] == ""

    hr = tmp_path / "hit_rates.csv"
    mod.write_hit_rates_csv(hr, {("0.01", "baseline_arc", "Q1"): qs}, {("0.01", "baseline_arc"): 128})
    assert _read_dicts(hr)[0]["eviction"] == "arc"

    wl = tmp_path / "workloads.csv"
    mod.write_workloads_csv(wl, [_workload_stats("baseline_stream_s1_arc", 128, eviction="arc"),
                                 _workload_stats("baseline_stream_s1", 128)])
    got = _read_dicts(wl)
    assert [r["eviction"] for r in got] == ["arc", ""]
    assert list(got[0])[-1] == "eviction"


def test_run_info_eviction_is_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / "fake_load_tpch"
    binary.write_text(
        "#!/bin/sh\nwhile [ $# -gt 0 ]; do [ \"$1\" = --results-dir ] && d=\"$2\"; shift; done\n"
        "mkdir -p \"$d\"\n"
        "printf 'query,run,rows,latency_ms,hits,misses\\nQ1,0,4,1.0,10,5\\n' > \"$d/latencies.csv\"\n"
        "printf 'key,value\\neviction,arc\\n' > \"$d/run_info.csv\"\n")
    binary.chmod(0o755)
    stats = mod.run_shilmandb_config(
        binary=binary, sf="0.01", config_name="baseline", data_dir=tmp_path,
        output_dir=tmp_path / "out", pool_size=128, warmup_runs=0, timed_runs=1,
        timeout_seconds=30, label="baseline_arc", extra_flags=mod.option_flags(eviction="arc"))
    assert stats is not None and stats["Q1"].eviction == "arc"


# DuckDB and Postgres in the same call
def _fake_shilmandb(calls: list):
    def run(**k):
        label = k.get("label") or k["config_name"]     # the pool sweep passes no label
        calls.append(label)
        return {q: mod.QueryStats(sf=k["sf"], config=label, query=q, median_latency_ms=1.0, rows=1,
                                  median_hits=10, median_misses=1, median_visits=2) for q in mod.QUERIES}
    return run


# Stands in for a runner subprocess: writes its latencies and info files, like the real one.
# configs_seen, when given, gets the configs each call asked for.
def _fake_engine(calls: list, ok: bool = True, configs_seen: Optional[list] = None):
    def run(engine, scale_factors, data_dir, output_dir, timed_runs, warmup_runs, configs=None):
        calls.append((engine, list(scale_factors), timed_runs, warmup_runs))
        if configs_seen is not None:
            configs_seen.append(list(configs) if configs is not None else None)
        if not ok:
            return False
        lines = ["sf,query,config,run,latency_ms"]
        for sf in scale_factors:
            for config in configs or mod.ENGINE_CONFIGS[engine]:
                lines += [f"{sf},{q},{config},{r},{r + 1.0}" for q in mod.QUERIES for r in (1, 2, 3)]
        (output_dir / f"{engine}_latencies.csv").write_text("\n".join(lines) + "\n")
        info = {"duckdb": "key,value\nduckdb_version,2.0.0\n",
                "postgres": "key,value\npostgres_version,PostgreSQL 18.1\nshared_buffers,16GB\njit,on\n"}[engine]
        (output_dir / f"{engine}_info.csv").write_text(info)
        return True
    return run


# Stands in for compare_results.py: answers by the --reference-name it was given.
# An answer of "CRASH" prints nothing and writes a traceback to stderr.
def _fake_compare(answers: dict, calls: list):
    def run(cmd):
        name = cmd[cmd.index("--reference-name") + 1]
        cfg, sf = cmd[cmd.index("--configs") + 1], cmd[cmd.index("--sf") + 1]
        calls.append((cfg, name))
        line = answers.get(name, "PASS")
        if line == "CRASH":
            return "", "Traceback (most recent call last):\nStopIteration\n"
        return "\n".join(f"  {line.split()[0]:<5} [{cfg}] {q} SF={sf}{line[len(line.split()[0]):]}"
                         for q in mod.QUERIES) + "\n", ""
    return run


def _stub_main(monkeypatch, shilmandb_calls, engine_calls, compare_calls, sqlite_calls, answers=None, engine_ok=True):
    monkeypatch.setattr(mod, "run_shilmandb_config", _fake_shilmandb(shilmandb_calls))
    monkeypatch.setattr(mod, "run_engine_benchmarks", _fake_engine(engine_calls, engine_ok))
    monkeypatch.setattr(mod, "run_sqlite_benchmarks", lambda *a, **k: sqlite_calls.append(a) or True)
    monkeypatch.setattr(mod, "compare_output", _fake_compare(answers or {}, compare_calls))
    monkeypatch.setattr(mod, "command_output", lambda cmd: "stub")


def test_default_call_writes_no_new_engine_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    out.mkdir()
    # an old DuckDB run in the same folder must not be picked up
    (out / "duckdb_latencies.csv").write_text("sf,query,config,run,latency_ms\n0.01,Q1,duckdb,1,1.0\n")
    shilmandb, engines, compares, sqlite = [], [], [], []
    _stub_main(monkeypatch, shilmandb, engines, compares, sqlite)
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--sf", "0.01", "--configs", "baseline",
                                      "--output-dir", str(out)])
    assert mod.main() == 0

    assert engines == []
    for name in ("latencies.csv", "correctness.csv", "hit_rates.csv"):
        configs = {r["config"] for r in _read_dicts(out / name)}
        assert configs == {"baseline"}, name
    assert not (out / "reference_checks.csv").exists()
    assert not (out / "environment.csv").exists()
    assert compares == [("baseline", "SQLite")]


def test_runner_imports_without_duckdb_or_psycopg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "duckdb", None)
    monkeypatch.setitem(sys.modules, "psycopg", None)
    fresh_spec = importlib.util.spec_from_file_location("run_benchmarks_no_engines", SPEC_PATH)
    assert fresh_spec and fresh_spec.loader
    fresh = importlib.util.module_from_spec(fresh_spec)
    monkeypatch.setitem(sys.modules, "run_benchmarks_no_engines", fresh)
    fresh_spec.loader.exec_module(fresh)
    assert fresh.resolve_configs("baseline") == ["baseline"]


def test_duckdb_flag_adds_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    shilmandb, engines, compares, sqlite = [], [], [], []
    _stub_main(monkeypatch, shilmandb, engines, compares, sqlite)
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--sf", "0.01", "--configs", "baseline", "--duckdb",
                                      "--runs", "3", "--output-dir", str(out)])
    assert mod.main() == 0

    assert engines == [("duckdb", ["0.01"], 3, 1)]
    lat = _read_dicts(out / "latencies.csv")
    assert {r["config"] for r in lat} == {"baseline", "duckdb", "duckdb_1thread"}
    q1 = next(r for r in lat if r["config"] == "duckdb" and r["query"] == "Q1")
    assert q1["median_latency_ms"] == "3.0000"
    assert q1["rows"] == "" and q1["misses"] == ""
    correctness = _read_dicts(out / "correctness.csv")
    for cfg in ("duckdb", "duckdb_1thread"):
        assert [r["status"] for r in correctness if r["config"] == cfg] == ["PASS"] * len(mod.QUERIES)
    assert list(correctness[0]) == ["sf", "query", "config", "status"]
    hit_rates = {r["config"]: r for r in _read_dicts(out / "hit_rates.csv")}
    assert hit_rates["duckdb"]["raw_hit_rate"] == ""
    assert (out / "reference_checks.csv").exists()
    assert (out / "environment.csv").exists()


def test_references_for_sf1_is_sqlite(tmp_path: Path) -> None:
    expected = [("SQLite", tmp_path / "sqlite_results", None)]
    for flags in ((False, False), (True, False), (True, True)):
        assert mod.references_for("1.0", *flags, output_dir=tmp_path) == expected
    assert mod.references_for("0.01", True, True, output_dir=tmp_path) == expected


def test_references_for_sf10_is_duckdb_and_postgres(tmp_path: Path) -> None:
    duck = ("DuckDB", tmp_path / "duckdb_results", "duckdb")
    pg = ("Postgres", tmp_path / "postgres_results", "postgres")
    assert mod.references_for("10", True, True, output_dir=tmp_path) == [duck, pg]
    assert mod.references_for("10", True, False, output_dir=tmp_path) == [duck]
    assert mod.references_for("10", False, True, output_dir=tmp_path) == [pg]
    # neither flag: SQLite, as before the engines existed
    assert mod.references_for("10", False, False, output_dir=tmp_path) == [("SQLite", tmp_path / "sqlite_results", None)]


def test_worst_status_over_references(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    answers = {"DuckDB": "PASS", "Postgres": "DRIFT  (max abs 0.0234, max rel 3.1e-12)"}
    monkeypatch.setattr(mod, "compare_output", _fake_compare(answers, calls))
    path, checks = tmp_path / "correctness.csv", tmp_path / "reference_checks.csv"
    mod.write_correctness_csv(path, ["10"], ["baseline", "duckdb", "postgres_1worker", "sqlite_disk"], tmp_path,
                              queries=["Q1"], duckdb=True, postgres=True, checks_path=checks)

    status = {r["config"]: r["status"] for r in _read_dicts(path)}
    assert status == {"baseline": "DRIFT", "duckdb": "DRIFT", "postgres_1worker": "PASS", "sqlite_disk": "SKIP"}
    rows = _read_dicts(checks)
    assert list(rows[0]) == ["sf", "query", "config", "reference", "status", "max_abs_diff", "max_rel_diff"]
    got = {(r["config"], r["reference"]): (r["status"], r["max_abs_diff"], r["max_rel_diff"]) for r in rows}
    assert got == {
        ("baseline", "DuckDB"): ("PASS", "", ""),
        ("baseline", "Postgres"): ("DRIFT", "0.0234", "3.1e-12"),
        ("duckdb", "Postgres"): ("DRIFT", "0.0234", "3.1e-12"),        # never against itself
        ("postgres_1worker", "DuckDB"): ("PASS", "", ""),
    }


def test_worst_status_order() -> None:
    assert mod.worst_status(["PASS", "DRIFT"]) == "DRIFT"
    assert mod.worst_status(["DRIFT", "FAIL", "PASS"]) == "FAIL"
    assert mod.worst_status(["SKIP", "PASS"]) == "PASS"
    assert mod.worst_status([]) == "SKIP"


def test_compare_line_parsing() -> None:
    text = ("  PASS  [baseline] Q1 SF=0.01\n"
            "  DRIFT [baseline] Q3 SF=0.01  (max abs 0.02, max rel 1e-11)\n"
            "       row 0 col 2: 1.00 vs 1.02\n"
            "  FAIL  [baseline] Q5 SF=0.01  (2 mismatch(es))\n"
            "  SKIP  [baseline] Q6 SF=0.01\n")
    assert mod.parse_compare_output(text) == {
        "Q1": ("PASS", "", ""), "Q3": ("DRIFT", "0.02", "1e-11"), "Q5": ("FAIL", "", ""), "Q6": ("SKIP", "", ""),
    }


def test_failed_engine_shows_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    shilmandb, engines, compares, sqlite = [], [], [], []
    _stub_main(monkeypatch, shilmandb, engines, compares, sqlite, engine_ok=False)
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--sf", "0.01", "--configs", "baseline", "--postgres",
                                      "--output-dir", str(out)])
    assert mod.main() == 3                      # every output is still written
    assert {r["config"] for r in _read_dicts(out / "latencies.csv")} == {"baseline"}
    status = {(r["config"], r["status"]) for r in _read_dicts(out / "correctness.csv")}
    assert status == {("baseline", "PASS"), ("postgres", "SKIP"), ("postgres_1worker", "SKIP")}
    assert ("postgres", "SQLite") not in compares


def test_environment_csv_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list = []

    def fake_run(cmd, **kwargs):
        commands.append(list(cmd))
        return mod.subprocess.CompletedProcess(cmd, 0, stdout="value\n", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    record = mod.environment_record({"duckdb_version": "2.0.0"},
                                    {"postgres_version": "PostgreSQL 18.1", "shared_buffers": "16GB"})
    path = tmp_path / "environment.csv"
    mod.write_environment_csv(path, record)
    got = {r["key"]: r["value"] for r in _read_dicts(path)}
    for key in ("date", "git_commit", "chip", "memory_bytes", "macos", "compiler", "build_type", "sqlite",
                "duckdb", "postgres", "psycopg", "libtorch", "pytorch", *mod.POSTGRES_SETTING_NAMES):
        assert key in got, key
    assert got["git_commit"] == "value"
    assert got["duckdb"] == "2.0.0"
    assert got["postgres"] == "PostgreSQL 18.1"
    assert got["shared_buffers"] == "16GB"
    assert got["jit"] == ""
    assert ["git", "rev-parse", "HEAD"] in commands
    assert all(c[0] != "git" or c[1] == "rev-parse" for c in commands)


def test_environment_without_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "command_output", lambda cmd: "")
    monkeypatch.setattr(mod, "package_version", lambda name: "")
    record = mod.environment_record({}, {})
    assert record["duckdb"] == "" and record["psycopg"] == "" and record["pytorch"] == ""


def test_configs_none_runs_only_new_engines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert mod.resolve_configs("none") == []
    with pytest.raises(SystemExit) as e:
        mod.resolve_configs("none,baseline")
    assert e.value.code != 0

    out = tmp_path / "out"
    shilmandb, engines, compares, sqlite = [], [], [], []
    _stub_main(monkeypatch, shilmandb, engines, compares, sqlite)
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--sf", "0.01", "--configs", "none", "--duckdb",
                                      "--output-dir", str(out)])
    assert mod.main() == 0
    assert shilmandb == [] and sqlite == []
    assert {r["config"] for r in _read_dicts(out / "latencies.csv")} == {"duckdb", "duckdb_1thread"}

    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--configs", "none,baseline", "--output-dir", str(out)])
    with pytest.raises(SystemExit):
        mod.main()


def test_workload_only_call_skips_engines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    shilmandb, engines, compares, sqlite = [], [], [], []
    _stub_main(monkeypatch, shilmandb, engines, compares, sqlite)
    monkeypatch.setattr(mod, "run_workload_config", lambda **k: _workload_stats(k["label"], k["pool_size"]))
    monkeypatch.setattr(mod, "workload_query_rows", lambda *a, **k: [])
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--sf", "0.01", "--configs", "baseline", "--duckdb",
                                      "--workloads", "stream", "--skip-correctness", "--output-dir", str(out)])
    assert mod.main() == 0
    assert engines == []


# a workload-only call never runs the engines, so it must not read their folders or write their files
def test_skipped_engines_are_not_references(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    shilmandb, engines, compares, sqlite = [], [], [], []
    _stub_main(monkeypatch, shilmandb, engines, compares, sqlite)
    monkeypatch.setattr(mod, "run_workload_config", lambda **k: _workload_stats(k["label"], k["pool_size"], sf=k["sf"]))
    monkeypatch.setattr(mod, "workload_query_rows", lambda *a, **k: [])
    monkeypatch.setattr(mod, "check_workload_runs", lambda *a, **k: {})
    monkeypatch.setattr(mod, "POOL_SWEEP_SIZES", [128])
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--sf", "10", "--configs", "baseline", "--duckdb", "--postgres",
                                      "--workloads", "stream", "--pool-sweep", "--output-dir", str(out)])
    assert mod.main() == 0
    assert engines == []
    assert compares and all(name == "SQLite" for _, name in compares)
    assert not (out / "reference_checks.csv").exists()
    assert not (out / "environment.csv").exists()


def test_no_engine_files_when_every_engine_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    shilmandb, engines, compares, sqlite = [], [], [], []
    _stub_main(monkeypatch, shilmandb, engines, compares, sqlite, engine_ok=False)
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--sf", "10", "--configs", "baseline", "--duckdb", "--postgres",
                                      "--output-dir", str(out)])
    assert mod.main() == 3
    assert len(engines) == 2
    assert compares == []                       # no reference left at SF=10, not a fall back to SQLite
    status = {(r["config"], r["status"]) for r in _read_dicts(out / "correctness.csv")}
    assert status == {(c, "SKIP") for c in ("baseline", "duckdb", "duckdb_1thread", "postgres", "postgres_1worker")}
    assert not (out / "reference_checks.csv").exists()
    assert not (out / "environment.csv").exists()


def _fake_cmake_build(build_dir: Path, versions: dict, cache_version: tuple) -> None:
    build_dir.mkdir(parents=True)
    major, minor, patch = cache_version
    (build_dir / "CMakeCache.txt").write_text(
        "CMAKE_BUILD_TYPE:STRING=Release\nCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++\n"
        f"CMAKE_CACHE_MAJOR_VERSION:INTERNAL={major}\nCMAKE_CACHE_MINOR_VERSION:INTERNAL={minor}\n"
        f"CMAKE_CACHE_PATCH_VERSION:INTERNAL={patch}\n")
    for folder, (compiler_version, mtime) in versions.items():
        f = build_dir / "CMakeFiles" / folder / "CMakeCXXCompiler.cmake"
        f.parent.mkdir(parents=True)
        f.write_text(f'set(CMAKE_CXX_COMPILER_ID "AppleClang")\nset(CMAKE_CXX_COMPILER_VERSION "{compiler_version}")\n')
        import os
        os.utime(f, (mtime, mtime))


def test_build_info_uses_the_cache_cmake_version(tmp_path: Path) -> None:
    # 10.1.0 sorts first as text and is newer on disk, but the cache says 3.9.0
    build = tmp_path / "build"
    _fake_cmake_build(build, {"3.9.0": ("17.0.0", 1_000), "10.1.0": ("15.0.0", 2_000)}, (3, 9, 0))
    assert mod.build_info(build) == ("/usr/bin/c++ (AppleClang 17.0.0)", "Release")


def test_build_info_falls_back_to_newest_folder(tmp_path: Path) -> None:
    build = tmp_path / "build"
    _fake_cmake_build(build, {"4.0.0": ("16.0.0", 1_000), "4.2.0": ("17.0.0", 2_000)}, (4, 3, 1))
    assert mod.build_info(build) == ("/usr/bin/c++ (AppleClang 17.0.0)", "Release")


# A crashed compare_results.py must not pass as SKIP
def test_compare_output_returns_stdout_and_stderr(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(mod.subprocess, "run",
                        lambda cmd, **k: mod.subprocess.CompletedProcess(cmd, 1, stdout="out\n", stderr="Traceback: boom\n"))
    assert mod.compare_output(["x"]) == ("out\n", "Traceback: boom\n")
    assert "Traceback: boom" in capsys.readouterr().err

    monkeypatch.setattr(mod.subprocess, "run",
                        lambda cmd, **k: mod.subprocess.CompletedProcess(cmd, 1, stdout="out\n", stderr=""))
    assert mod.compare_output(["x"]) == ("out\n", "")
    assert capsys.readouterr().err == ""


def test_crashed_compare_is_fail_in_both_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    monkeypatch.setattr(mod, "compare_output", _fake_compare({"DuckDB": "CRASH", "Postgres": "PASS"}, calls))
    path, checks = tmp_path / "correctness.csv", tmp_path / "reference_checks.csv"
    mod.write_correctness_csv(path, ["10"], ["baseline"], tmp_path, queries=["Q1"], duckdb=True, postgres=True, checks_path=checks)
    assert {r["config"]: r["status"] for r in _read_dicts(path)} == {"baseline": "FAIL"}
    got = {r["reference"]: r["status"] for r in _read_dicts(checks)}
    assert got == {"DuckDB": "FAIL", "Postgres": "PASS"}


def test_missing_line_without_stderr_stays_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "compare_output", lambda cmd: ("", ""))
    path = tmp_path / "correctness.csv"
    mod.write_correctness_csv(path, ["0.01"], ["baseline"], tmp_path, queries=["Q1"])
    assert {r["config"]: r["status"] for r in _read_dicts(path)} == {"baseline": "SKIP"}


# The real compare_results.py on an empty result file: it crashes, and that is FAIL, not SKIP.
def test_real_compare_crash_is_fail(tmp_path: Path) -> None:
    (tmp_path / "sqlite_results").mkdir()
    (tmp_path / "sqlite_results" / "Q1_sf0.01.csv").write_text("a\n1\n")
    (tmp_path / "shilmandb_results" / "baseline").mkdir(parents=True)
    (tmp_path / "shilmandb_results" / "baseline" / "Q1_sf0.01.csv").write_text("")
    path = tmp_path / "correctness.csv"
    mod.write_correctness_csv(path, ["0.01"], ["baseline"], tmp_path, queries=["Q1", "Q3"])
    assert {r["query"]: r["status"] for r in _read_dicts(path)} == {"Q1": "FAIL", "Q3": "FAIL"}


def test_main_exit_3_when_an_engine_failed_and_0_otherwise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shilmandb, engines, compares, sqlite = [], [], [], []
    _stub_main(monkeypatch, shilmandb, engines, compares, sqlite)

    # A fresh folder per run, so the second run's outputs can't be left from the first.
    def run_into(out: Path) -> int:
        monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--sf", "0.01", "--configs", "baseline", "--duckdb", "--postgres",
                                          "--output-dir", str(out)])
        return mod.main()

    assert run_into(tmp_path / "ok") == 0

    def one_fails(engine, *args):
        return engine == "duckdb" and _fake_engine(engines)(engine, *args)

    monkeypatch.setattr(mod, "run_engine_benchmarks", one_fails)
    failed = tmp_path / "failed"
    assert run_into(failed) == 3
    assert (failed / "latencies.csv").exists() and (failed / "environment.csv").exists()


def test_help_mentions_exit_3(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--help"])
    with pytest.raises(SystemExit):
        mod.parse_args()
    assert "exit 3" in " ".join(capsys.readouterr().out.split())


def test_engine_runner_gets_warmup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list = []
    monkeypatch.setattr(mod.subprocess, "run", lambda cmd, **k: commands.append(cmd) or mod.subprocess.CompletedProcess(cmd, 0))
    assert mod.run_engine_benchmarks("duckdb", ["0.01"], tmp_path, tmp_path, 3, 2) is True
    cmd = commands[0]
    assert cmd[cmd.index("--warmup") + 1] == "2"
    assert cmd[cmd.index("--runs") + 1] == "3"


def test_main_passes_warmup_to_engines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shilmandb, engines, compares, sqlite = [], [], [], []
    _stub_main(monkeypatch, shilmandb, engines, compares, sqlite)
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--sf", "0.01", "--configs", "none", "--postgres",
                                      "--warmup", "4", "--runs", "2", "--output-dir", str(tmp_path / "out")])
    assert mod.main() == 0
    assert engines == [("postgres", ["0.01"], 2, 4)]


def test_references_run_after_shilmandb_postgres_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list = []
    shilmandb = _fake_shilmandb([])
    engine = _fake_engine([])
    monkeypatch.setattr(mod, "run_shilmandb_config", lambda **k: events.append("shilmandb") or shilmandb(**k))
    monkeypatch.setattr(mod, "run_engine_benchmarks", lambda name, *a, **k: events.append(name) or engine(name, *a, **k))
    monkeypatch.setattr(mod, "run_sqlite_benchmarks", lambda *a, **k: events.append("sqlite") or True)
    monkeypatch.setattr(mod, "compare_output", _fake_compare({}, []))
    monkeypatch.setattr(mod, "command_output", lambda cmd: "stub")
    monkeypatch.setattr(mod, "POOL_SWEEP_SIZES", [128])
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--sf", "0.01", "--configs", "baseline,+index,sqlite_disk",
                                      "--duckdb", "--postgres", "--pool-sweep", "--output-dir", str(tmp_path / "out")])
    assert mod.main() == 0
    # The pool sweep is ShilmanDB too
    assert events == ["sqlite", "shilmandb", "shilmandb", "shilmandb", "duckdb", "postgres"]


def test_duckdb_20g_needs_the_duckdb_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shilmandb, engines, compares, sqlite = [], [], [], []
    _stub_main(monkeypatch, shilmandb, engines, compares, sqlite)
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--sf", "0.01", "--configs", "baseline,duckdb_20g",
                                      "--output-dir", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as err:
        mod.main()
    assert "duckdb_20g" in str(err.value) and "--duckdb" in str(err.value)
    assert shilmandb == [] and engines == []


def test_duckdb_20g_runs_through_the_duckdb_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    shilmandb, engines, compares, sqlite, configs_seen = [], [], [], [], []
    _stub_main(monkeypatch, shilmandb, engines, compares, sqlite)
    monkeypatch.setattr(mod, "run_engine_benchmarks", _fake_engine(engines, configs_seen=configs_seen))
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--sf", "10", "--configs", "baseline,duckdb_20g",
                                      "--duckdb", "--postgres", "--output-dir", str(out)])
    assert mod.main() == 0
    assert shilmandb == ["baseline"]
    assert configs_seen == [["duckdb", "duckdb_1thread", "duckdb_20g"], ["postgres", "postgres_1worker"]]
    assert {r["config"] for r in _read_dicts(out / "latencies.csv")} == {
        "baseline", "duckdb", "duckdb_1thread", "duckdb_20g", "postgres", "postgres_1worker"}
    assert mod.engine_of("duckdb_20g") == "duckdb"
    # Checked against Postgres only, never its own engine
    assert {name for cfg, name in compares if cfg == "duckdb_20g"} == {"Postgres"}
    statuses = {r["config"]: r["status"] for r in _read_dicts(out / "correctness.csv")}
    assert statuses["duckdb_20g"] == "PASS"


def test_duckdb_flag_alone_asks_for_the_two_default_configs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shilmandb, engines, compares, sqlite, configs_seen = [], [], [], [], []
    _stub_main(monkeypatch, shilmandb, engines, compares, sqlite)
    monkeypatch.setattr(mod, "run_engine_benchmarks", _fake_engine(engines, configs_seen=configs_seen))
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--sf", "0.01", "--configs", "none", "--duckdb",
                                      "--output-dir", str(tmp_path / "out")])
    assert mod.main() == 0
    assert configs_seen == [["duckdb", "duckdb_1thread"]]


def test_engine_runner_gets_configs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list = []
    monkeypatch.setattr(mod.subprocess, "run", lambda cmd, **k: commands.append(cmd) or mod.subprocess.CompletedProcess(cmd, 0))
    assert mod.run_engine_benchmarks("duckdb", ["10"], tmp_path, tmp_path, 3, 1,
                                     configs=["duckdb", "duckdb_1thread", "duckdb_20g"]) is True
    cmd = commands[0]
    assert cmd[cmd.index("--configs") + 1] == "duckdb,duckdb_1thread,duckdb_20g"


# compare_results.py gets --no-drift at SF=100, never at 10 or below
def test_runner_passes_no_drift_at_sf100_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cmds: list = []
    monkeypatch.setattr(mod, "compare_output", lambda cmd: cmds.append(cmd) or ("", ""))
    mod.write_correctness_csv(tmp_path / "correctness.csv", ["0.01", "1.0", "10", "100"],
                              ["columnar+parallel", "duckdb"], tmp_path, queries=["Q1"], duckdb=True, postgres=True)
    flags: dict = {}
    for cmd in cmds:
        flags.setdefault(cmd[cmd.index("--sf") + 1], []).append(cmd.count("--no-drift"))
    # SQLite checks both configs below 10; above, ours against two engines and DuckDB against Postgres
    assert flags == {"0.01": [0, 0], "1.0": [0, 0], "10": [0, 0, 0], "100": [1, 1, 1]}


# The real compare_results.py: the same 1e-12 difference is DRIFT at SF=10 and FAIL at SF=100.
def test_real_compare_drift_is_fail_at_sf100(tmp_path: Path) -> None:
    for sf in ("10", "100"):
        for folder in ("duckdb_results/duckdb", "postgres_results/postgres"):
            (tmp_path / folder).mkdir(parents=True, exist_ok=True)
            (tmp_path / folder / f"Q1_sf{sf}.csv").write_text("sum_charge\n1000000000000.00\n")
        (tmp_path / "shilmandb_results" / "columnar+parallel").mkdir(parents=True, exist_ok=True)
        (tmp_path / "shilmandb_results" / "columnar+parallel" / f"Q1_sf{sf}.csv").write_text("sum_charge\n999999999999.00\n")
    path, checks = tmp_path / "correctness.csv", tmp_path / "reference_checks.csv"
    mod.write_correctness_csv(path, ["10", "100"], ["columnar+parallel"], tmp_path, queries=["Q1"],
                              duckdb=True, postgres=True, checks_path=checks)
    assert {r["sf"]: r["status"] for r in _read_dicts(path)} == {"10": "DRIFT", "100": "FAIL"}
    got = {(r["sf"], r["reference"]): (r["status"], r["max_rel_diff"]) for r in _read_dicts(checks)}
    assert got == {("10", "DuckDB"): ("DRIFT", "1e-12"), ("10", "Postgres"): ("DRIFT", "1e-12"),
                   ("100", "DuckDB"): ("FAIL", ""), ("100", "Postgres"): ("FAIL", "")}
