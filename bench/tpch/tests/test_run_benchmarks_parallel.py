"""Tests for run_benchmarks.py's parallel configs, the thread sweep and parallel_stats copies."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import List

import pytest

SPEC_PATH = Path(__file__).resolve().parents[1] / "run_benchmarks.py"
spec = importlib.util.spec_from_file_location("run_benchmarks_parallel_under_test", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["run_benchmarks_parallel_under_test"] = mod
spec.loader.exec_module(mod)

PARALLEL_FLAGS = ["--create-indexes", "true", "--execution-mode", "parallel", "--storage", "columnar"]
PARALLEL_STATS = "query,run,threads,morsels,latch_wait_ms\nQ1,0,18,4,0.125\n"


# The value after a flag in a command.
def _flag(cmd: List[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


# Which config a load_tpch command runs, read from its flags.
def _config_of(cmd: List[str]) -> str:
    mode = _flag(cmd, "--execution-mode")
    if mode == "parallel":
        if "--threads" not in cmd:
            return "columnar+parallel"
        return f"columnar+parallel_{_flag(cmd, '--threads')}thread"
    indexes = _flag(cmd, "--create-indexes") == "true"
    name = "+vectorized" if mode == "vectorized" else ("+index" if indexes else "baseline")
    if "--storage" in cmd:
        name = {"baseline": "columnar", "+index": "columnar+index", "+vectorized": "columnar+vectorized"}[name]
    return name


# Stands in for load_tpch: writes a run's files, and parallel_stats.csv in parallel mode.
def _fake_load_tpch(commands: List[List[str]]):
    def run(cmd, **kwargs):
        cmd = list(cmd)
        commands.append(cmd)
        results = Path(_flag(cmd, "--results-dir"))
        results.mkdir(parents=True, exist_ok=True)
        (results / "latencies.csv").write_text(
            "query,run,rows,latency_ms,hits,misses,visits,reads,read_ms\n"
            + "".join(f"{q},0,1,1.0,10,1,2,1,0.1\n" for q in mod.QUERIES))
        (results / "run_info.csv").write_text("key,value\nsf,0.01\ndb_pages,7\neviction,lru\n")
        if _flag(cmd, "--execution-mode") == "parallel":
            (results / "parallel_stats.csv").write_text(PARALLEL_STATS)
        return mod.subprocess.CompletedProcess(cmd, 0, "", "")
    return run


# Runs the runner's main with the fake load_tpch and returns every command it ran.
def _run_main(monkeypatch: pytest.MonkeyPatch, out: Path, configs: str, *extra: str) -> List[List[str]]:
    commands: List[List[str]] = []
    monkeypatch.setattr(mod.subprocess, "run", _fake_load_tpch(commands))
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--sf", "0.01", "--configs", configs,
                                      "--skip-correctness", "--output-dir", str(out), *extra])
    assert mod.main() == 0
    return commands


def test_parallel_config_flags() -> None:
    assert mod.PRIMARY_CONFIGS["columnar+parallel"] == PARALLEL_FLAGS
    assert mod.PRIMARY_CONFIGS["columnar+parallel_1thread"] == PARALLEL_FLAGS + ["--threads", "1"]


def test_parallel_configs_are_columnar_and_outside_all() -> None:
    for name in ("columnar+parallel", "columnar+parallel_1thread"):
        assert name in mod.COLUMNAR_CONFIG_NAMES
        assert name in mod.KNOWN_CONFIG_NAMES
        assert name not in mod.ALL_CONFIG_NAMES
    assert mod.resolve_configs("all") == ["baseline", "+index", "+learned", "+vectorized",
                                          "sqlite_disk", "sqlite_memory"]


def test_sweep_names_are_not_configs_on_their_own() -> None:
    with pytest.raises(SystemExit):
        mod.resolve_configs("columnar+parallel_2thread")


def test_parallel_threads_list() -> None:
    assert mod.parallel_threads_list("2,4,6,12") == [2, 4, 6, 12]
    assert mod.parallel_threads_list("4,2,4") == [4, 2]
    assert mod.parallel_threads_list("1,256") == [1, 256]
    for bad in ("0", "257", "2,x", "", "-2"):
        with pytest.raises(argparse.ArgumentTypeError):
            mod.parallel_threads_list(bad)


def test_sweep_config_names() -> None:
    assert mod.sweep_configs([2, 4, 6, 12], []) == [
        "columnar+parallel_2thread", "columnar+parallel_4thread",
        "columnar+parallel_6thread", "columnar+parallel_12thread"]
    # 1 thread is the named config, never a second copy
    assert mod.sweep_configs([1, 2], []) == ["columnar+parallel_1thread", "columnar+parallel_2thread"]
    assert mod.sweep_configs([1, 2], ["columnar+parallel_1thread"]) == ["columnar+parallel_2thread"]


def test_sweep_command_flags(tmp_path: Path) -> None:
    cmd = mod.build_command(
        binary=Path("/bin/fake"), sf="0.01", config_name="columnar+parallel_4thread",
        data_dir=tmp_path, results_dir=tmp_path / "r", db_file=tmp_path / "db",
        pool_size=128, warmup_runs=1, timed_runs=5,
    )
    assert cmd[cmd.index("--no-verify") + 1:] == PARALLEL_FLAGS + ["--threads", "4"]


def test_sweep_runs_after_named_configs_in_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _run_main(monkeypatch, tmp_path / "out", "columnar+parallel,columnar+vectorized",
                         "--parallel-threads", "6,2,1,2")
    assert [_config_of(c) for c in commands] == [
        "columnar+parallel", "columnar+vectorized", "columnar+parallel_6thread",
        "columnar+parallel_2thread", "columnar+parallel_1thread"]


def test_parallel_configs_share_the_indexed_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _run_main(monkeypatch, tmp_path / "out",
                         "columnar,columnar+index,columnar+parallel,columnar+parallel_1thread",
                         "--parallel-threads", "2")
    by = {_config_of(c): c for c in commands}
    indexed = _flag(by["columnar+index"], "--db-file")
    assert "--reuse-db" not in by["columnar+index"]
    for name in ("columnar+parallel", "columnar+parallel_1thread", "columnar+parallel_2thread"):
        assert _flag(by[name], "--db-file") == indexed
        assert "--reuse-db" in by[name]
    assert _flag(by["columnar"], "--db-file") != indexed


def test_parallel_alone_loads_the_indexed_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture) -> None:
    commands = _run_main(monkeypatch, tmp_path / "out", "columnar+parallel", "--parallel-threads", "2")
    assert ["--reuse-db" in c for c in commands] == [False, True]
    assert _flag(commands[0], "--db-file").endswith("tpch_sf0.01_indexed.db")
    # Columnar runs never check db_pages against the row store's size
    assert "db_pages" not in capsys.readouterr().err


def test_parallel_stats_copied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    _run_main(monkeypatch, out, "columnar+vectorized,columnar+parallel,columnar+parallel_1thread",
              "--parallel-threads", "2")
    assert sorted(p.name for p in out.glob("parallel_stats_*.csv")) == [
        "parallel_stats_columnar+parallel_1thread_sf0.01.csv",
        "parallel_stats_columnar+parallel_2thread_sf0.01.csv",
        "parallel_stats_columnar+parallel_sf0.01.csv"]
    assert (out / "parallel_stats_columnar+parallel_sf0.01.csv").read_text() == PARALLEL_STATS


def test_sweep_rows_in_latencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    _run_main(monkeypatch, out, "columnar+parallel", "--parallel-threads", "2")
    configs = {line.split(",")[2] for line in (out / "latencies.csv").read_text().splitlines()[1:]}
    assert configs == {"columnar+parallel", "columnar+parallel_2thread"}


def test_non_parallel_call_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    commands = _run_main(monkeypatch, out, "+vectorized,columnar+index,columnar+vectorized")
    assert all("--threads" not in c and "parallel" not in c for c in commands)
    assert not list(out.glob("parallel_stats_*.csv"))
    row = commands[0]
    assert row == mod.build_command(
        binary=mod.DEFAULT_BINARY, sf="0.01", config_name="+vectorized", data_dir=mod.DEFAULT_DATA_DIR,
        results_dir=Path(_flag(row, "--results-dir")), db_file=Path(_flag(row, "--db-file")),
        pool_size=mod.POOL_SIZES["0.01"], warmup_runs=1, timed_runs=5)
    assert row[row.index("--no-verify") + 1:] == ["--create-indexes", "true", "--execution-mode", "vectorized"]
