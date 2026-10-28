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
    assert rows[0] == ["sf", "query", "config", "median_latency_ms",
                       "rows", "hits", "misses"]
    assert len(rows) == 3
    shilmandb_row = next(r for r in rows[1:] if r[2] == "baseline")
    assert shilmandb_row[3] == "123.4567"
    assert shilmandb_row[5] == "100"
    sqlite_row = next(r for r in rows[1:] if r[2] == "sqlite_disk")
    assert sqlite_row[4] == ""
    assert sqlite_row[5] == ""
    assert sqlite_row[6] == ""


# write_hit_rates_csv
def test_write_hit_rates_csv_improvement_vs_baseline(tmp_path: Path) -> None:
    def make(sf, cfg, q, hits, misses):
        return mod.QueryStats(
            sf=sf, config=cfg, query=q, median_latency_ms=1.0, rows=1,
            median_hits=hits, median_misses=misses,
        )
    records = {
        ("0.01", "baseline", "Q1"): make("0.01", "baseline", "Q1", 90, 10),
        ("0.01", "baseline", "Q6"): make("0.01", "baseline", "Q6", 80, 20),
        ("0.01", "+learned", "Q1"): make("0.01", "+learned", "Q1", 95, 5),
        ("0.01", "+learned", "Q6"): make("0.01", "+learned", "Q6", 85, 15),
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
        rows = list(csv.reader(f))
    header = rows[0]
    assert header == ["sf", "config", "pool_size", "hit_rate", "improvement_over_lru"]
    baseline_row = next(r for r in rows[1:] if r[1] == "baseline")
    learned_row  = next(r for r in rows[1:] if r[1] == "+learned")
    sqlite_row   = next(r for r in rows[1:] if r[1] == "sqlite_disk")
    assert baseline_row[3] == "0.8500"
    assert baseline_row[4] == ""
    assert learned_row[3] == "0.9000"
    assert learned_row[4] == "0.0500"
    assert sqlite_row[3] == ""
    assert sqlite_row[4] == ""


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
