"""Unit tests for bench/run_benchmarks.py helpers."""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

SPEC_PATH = Path(__file__).with_name("run_benchmarks.py")
spec = importlib.util.spec_from_file_location("run_benchmarks", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)

sys.modules["run_benchmarks"] = mod
spec.loader.exec_module(mod)

LATENCIES_HEADER = [
    "sf", "query", "config", "median_latency_ms", "rows", "hits", "misses",
    "visits", "page_hit_rate", "reads", "read_ms", "us_per_read", "policy_ms",
    "pool_size", "miss_penalty_us", "direct_io",
]
HIT_RATES_HEADER = [
    "sf", "config", "pool_size", "pool_fraction", "miss_penalty_us", "direct_io",
    "hit_rate", "raw_hit_rate", "improvement_over_lru",
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
    assert mod.DB_PAGES == {"0.01": 1448, "0.1": 14417, "1.0": 144032}
    assert mod.pool_for_fraction("0.01", 0.05) == 72
    assert mod.pool_for_fraction("1.0", 0.01) == 1440
    assert mod.pool_for_fraction("0.01", 0.001) == 16
    assert mod.pool_for_fraction("0.01", 1.0) == 1448


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


def test_can_run_and_fraction_config() -> None:
    assert not mod.can_run("baseline", "lookup")
    assert not mod.can_run("baseline", "mixed")
    assert mod.can_run("baseline", "stream")
    assert mod.can_run("baseline", "single")
    assert mod.can_run("+index", "lookup")
    assert mod.can_run("+vectorized", "mixed")
    assert mod.fraction_config("single") == "baseline"
    assert mod.fraction_config("stream") == "baseline"
    assert mod.fraction_config("lookup") == "+index"
    assert mod.fraction_config("mixed") == "+index"


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


def test_planned_runs_skips_and_fraction_configs() -> None:
    runs = mod.planned_runs(["0.01"], ["baseline", "+index"], ["stream", "lookup"], [0], [0.25])
    assert runs == [
        mod.RunSpec("0.01", "baseline", "stream", 128, 0, None),
        mod.RunSpec("0.01", "+index", "stream", 128, 0, None),
        mod.RunSpec("0.01", "baseline", "stream", 362, 0, 0.25),
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
    "reads", "read_ms", "us_per_read", "query_misses", "lookup_misses", "policy_ms", "correct",
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
