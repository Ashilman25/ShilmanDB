"""Tests for run_benchmarks.py's parallel configs, the thread sweep and parallel_stats copies."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Callable, List

import pytest

SPEC_PATH = Path(__file__).resolve().parents[1] / "run_benchmarks.py"
spec = importlib.util.spec_from_file_location("run_benchmarks_parallel_under_test", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["run_benchmarks_parallel_under_test"] = mod
spec.loader.exec_module(mod)

PARALLEL_FLAGS = ["--create-indexes", "true", "--execution-mode", "parallel", "--storage", "columnar"]
# load_tpch's parallel_stats.csv, with the memory and spill columns.
PARALLEL_STATS = ("query,run,threads,morsels,latch_wait_ms,memory_limit,peak_tracked_bytes,"
                  "spill_bytes_written,spill_bytes_read,spill_files,spill_partitions,spill_ms\n"
                  "Q5,0,18,27,0.000,150994944,150994944,3943119,3943119,95,215,2698.199\n")
# The keys load_tpch's run_info.csv ends with: peak memory and the memory limit.
RUN_INFO_MEMORY = "storage,columnar\nreuse_db,true\npeak_rss_bytes,192888832\nmemory_limit,150994944\n"


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
        info = "key,value\nsf,0.01\ndb_pages,7\neviction,lru\n"
        if _flag(cmd, "--execution-mode") == "parallel":
            info += RUN_INFO_MEMORY
            (results / "parallel_stats.csv").write_text(PARALLEL_STATS)
        (results / "run_info.csv").write_text(info)
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


def test_run_info_keeps_the_memory_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    _run_main(monkeypatch, out, "columnar+parallel")
    info = mod.read_run_info(out / "shilmandb_results" / "columnar+parallel" / "run_info_sf0.01.csv")
    assert info["peak_rss_bytes"] == "192888832"
    assert info["memory_limit"] == "150994944"
    assert info["eviction"] == "lru"


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


SF100_POOL, SF100_LIMIT, SPILL_LIMIT = "2097152", "4294967296", "268435456"
QUERY_TABLES = ["region", "nation", "part", "supplier", "customer", "orders", "lineitem"]


# Every value given to a flag in a command.
def _flag_values(cmd: List[str], name: str) -> List[str]:
    return [cmd[i + 1] for i, arg in enumerate(cmd) if arg == name]


# The label a command runs under, read from its temp results folder's name.
def _label_of(cmd: List[str]) -> str:
    name = Path(_flag(cmd, "--results-dir")).name
    m = re.match(rf"shilmandb_(.+)_sf{re.escape(_flag(cmd, '--sf'))}_", name)
    assert m, name
    return m.group(1)


# The fake load_tpch, plus the database file a loading run leaves,
# and the stream files of a workload.
# fails(index, cmd) picks runs that exit 1 and write nothing.
def _fake_loader(commands: List[List[str]], fails: Callable[[int, List[str]], bool] = lambda i, c: False):
    write_files = _fake_load_tpch(commands)

    def run(cmd, **kwargs):
        cmd = list(cmd)
        if fails(len(commands), cmd):
            commands.append(cmd)
            return mod.subprocess.CompletedProcess(cmd, 1, "", "load failed\n")
        done = write_files(cmd, **kwargs)
        if "--reuse-db" not in cmd:
            Path(_flag(cmd, "--db-file")).write_text("loaded")
        if "--workload" in cmd:
            results = Path(_flag(cmd, "--results-dir"))
            for name in ("stream", "stream_queries", "stream_results"):
                (results / f"{name}.csv").write_text("seq\n")
            (results / "stream_totals.csv").write_text(
                "pass,warmup,items,latency_ms,hits,misses,visits,reads,read_ms,query_misses,lookup_misses\n"
                "0,0,4,1.0,10,1,2,1,0.1,1,0\n")
        return done
    return run


# Runs main at one scale factor with the fake loader; returns every load_tpch command.
def _main_at(monkeypatch: pytest.MonkeyPatch, out: Path, sf: str, configs: str, *extra: str,
             fails: Callable[[int, List[str]], bool] = lambda i, c: False) -> List[List[str]]:
    commands: List[List[str]] = []
    monkeypatch.setattr(mod.subprocess, "run", _fake_loader(commands, fails))
    monkeypatch.setattr(mod, "workload_query_rows", lambda *a, **k: [])
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--sf", sf, "--configs", configs,
                                      "--skip-correctness", "--output-dir", str(out), *extra])
    assert mod.main() == 0
    return commands


# main's refusal message for these arguments; nothing may run.
def _refusal(monkeypatch: pytest.MonkeyPatch, out: Path, *argv: str) -> str:
    commands: List[List[str]] = []
    monkeypatch.setattr(mod.subprocess, "run", _fake_loader(commands))
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", *argv, "--skip-correctness", "--output-dir", str(out)])
    with pytest.raises(SystemExit) as err:
        mod.main()
    assert commands == []
    return str(err.value)


# One tiny .tbl per query table under data/sf<sf>/.
def _tiny_data(root: Path, sf: str = "0.01") -> Path:
    folder = root / f"sf{sf}"
    folder.mkdir(parents=True)
    for table in QUERY_TABLES:
        (folder / f"{table}.tbl").write_text("1|a|\n2|b|\n")
    return root


def _read_dicts(path: Path) -> List[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_sf100_is_known_only_when_named() -> None:
    assert mod.KNOWN_SCALE_FACTORS == ["0.01", "0.1", "1.0", "10", "100"]
    assert mod.resolve_sfs(None) == ["0.01", "0.1", "1.0"]
    assert mod.resolve_sfs(["100", "10"]) == ["10", "100"]
    assert mod.POOL_SIZES["100"] == 2097152
    assert mod.MEMORY_LIMITS == {"100": 4 << 30}


def test_new_parallel_configs_are_columnar_and_outside_all() -> None:
    assert mod.PRIMARY_CONFIGS["columnar+parallel_spill"] == PARALLEL_FLAGS + ["--memory-limit", SPILL_LIMIT]
    assert mod.PRIMARY_CONFIGS["columnar+parallel_direct"] == PARALLEL_FLAGS + ["--direct-io"]
    assert mod.PRIMARY_CONFIGS["columnar+parallel_20g"] == PARALLEL_FLAGS + ["--memory-limit", SF100_LIMIT]
    for name in ("columnar+parallel_spill", "columnar+parallel_direct", "columnar+parallel_20g"):
        assert name in mod.COLUMNAR_CONFIG_NAMES and name in mod.KNOWN_CONFIG_NAMES
        assert name not in mod.ALL_CONFIG_NAMES
        assert mod.is_parallel_config(name)
    assert mod.is_parallel_config("columnar+parallel_6thread")
    assert not any(mod.is_parallel_config(n) for n in ("baseline", "columnar+vectorized", "sqlite_disk", "duckdb"))


def test_new_parallel_configs_share_the_indexed_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _main_at(monkeypatch, tmp_path / "out", "0.01",
                        "columnar+index,columnar+parallel_spill,columnar+parallel_direct,columnar+parallel_20g")
    assert [_label_of(c) for c in commands] == ["columnar+index", "columnar+parallel_spill",
                                                "columnar+parallel_direct", "columnar+parallel_20g"]
    assert len({_flag(c, "--db-file") for c in commands}) == 1
    assert ["--reuse-db" in c for c in commands] == [False, True, True, True]


def test_sf100_parallel_configs_get_pool_and_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _main_at(monkeypatch, tmp_path / "out", "100",
                        "columnar+parallel,columnar+parallel_1thread,columnar+parallel_direct",
                        "--parallel-threads", "4")
    assert [_label_of(c) for c in commands] == ["columnar+parallel", "columnar+parallel_1thread",
                                                "columnar+parallel_direct", "columnar+parallel_4thread"]
    for cmd in commands:
        assert _flag_values(cmd, "--pool-size") == [SF100_POOL]
        assert _flag_values(cmd, "--memory-limit") == [SF100_LIMIT]

    # Below 100 a parallel config keeps the database's default limit
    for cmd in _main_at(monkeypatch, tmp_path / "out10", "10", "columnar+parallel", "--parallel-threads", "4"):
        assert "--memory-limit" not in cmd
        assert _flag_values(cmd, "--pool-size") == ["8192"]


def test_sf100_refuses_configs_without_a_memory_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    message = _refusal(monkeypatch, tmp_path / "o1", "--sf", "100", "--configs", "all")
    for name in ("baseline", "+index", "+learned", "+vectorized", "sqlite_disk", "sqlite_memory"):
        assert name in message
    assert "SF=100" in message

    message = _refusal(monkeypatch, tmp_path / "o2", "--sf", "10", "--sf", "100", "--configs",
                       "columnar,columnar+index,columnar+vectorized,columnar+parallel")
    assert "columnar, columnar+index, columnar+vectorized" in message
    assert "columnar+parallel," not in message and not message.endswith("columnar+parallel")

    for flag in (["--pool-fractions", "0.5"], ["--v1-eviction"], ["--workloads", "single,stream"]):
        message = _refusal(monkeypatch, tmp_path / "o3", "--sf", "100", "--configs", "columnar+parallel", *flag)
        assert flag[0] in message


def test_spill_config_overrides_the_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for sf in ("0.01", "100"):
        commands = _main_at(monkeypatch, tmp_path / f"out{sf}", sf, "columnar+parallel_spill")
        assert _flag_values(commands[0], "--memory-limit") == [SPILL_LIMIT]
        assert "--cold" not in commands[0]


def test_direct_parallel_config_runs_cold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    cmd = _main_at(monkeypatch, out, "0.01", "columnar+parallel_direct")[0]
    assert cmd.count("--direct-io") == 1 and cmd.count("--cold") == 1
    assert _label_of(cmd) == "columnar+parallel_direct"
    options = mod.parse_label("columnar+parallel_direct")
    assert (options.base, options.direct_io) == ("columnar+parallel", True)
    assert {r["direct_io"] for r in _read_dicts(out / "latencies.csv")} == {"true"}

    # The global --direct-io adds --cold to parallel configs only
    commands = _main_at(monkeypatch, tmp_path / "out2", "0.01", "+vectorized,columnar+vectorized,columnar+parallel",
                        "--direct-io", "--parallel-threads", "2")
    by = {_label_of(c): c for c in commands}
    assert sorted(by) == ["+vectorized_direct", "columnar+parallel_2thread_direct", "columnar+parallel_direct",
                          "columnar+vectorized_direct"]
    assert all(c.count("--direct-io") == 1 for c in commands)
    assert ["--cold" in by[k] for k in sorted(by)] == [False, True, True, False]


def test_direct_config_and_flag_together_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    message = _refusal(monkeypatch, tmp_path / "out", "--sf", "0.01", "--configs", "columnar+parallel_direct", "--direct-io")
    assert "columnar+parallel_direct" in message and "--direct-io" in message


def test_workloads_never_get_cold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    commands = _main_at(monkeypatch, out, "0.01", "columnar+parallel_direct,columnar+parallel",
                        "--workloads", "stream")
    assert all("--cold" not in c for c in commands)
    assert ["--direct-io" in c for c in commands] == [True, False]
    # The workload label puts _direct last, so it still reads as direct I/O
    rows = _read_dicts(out / "workloads.csv")
    assert [(r["config"], r["direct_io"]) for r in rows] == [
        ("columnar+parallel_stream_s1_direct", "true"), ("columnar+parallel_stream_s1", "false")]


def test_parallel_20g_overrides_the_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    commands = _main_at(monkeypatch, out, "10", "columnar+parallel,columnar+parallel_20g")
    by = {_label_of(c): c for c in commands}
    assert _flag_values(by["columnar+parallel_20g"], "--pool-size") == [SF100_POOL]
    assert _flag_values(by["columnar+parallel_20g"], "--memory-limit") == [SF100_LIMIT]
    assert _flag_values(by["columnar+parallel"], "--pool-size") == ["8192"]
    pools = {r["config"]: r["pool_size"] for r in _read_dicts(out / "hit_rates.csv")}
    assert pools == {"columnar+parallel": "8192", "columnar+parallel_20g": SF100_POOL}

    cmd = _main_at(monkeypatch, tmp_path / "out100", "100", "columnar+parallel_20g")[0]
    assert _flag_values(cmd, "--pool-size") == [SF100_POOL]
    assert _flag_values(cmd, "--memory-limit") == [SF100_LIMIT]


def test_columnar_dir_reuses_a_marked_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _tiny_data(tmp_path / "data")
    shared = tmp_path / "columnar"
    extra = ("--data-dir", str(data), "--columnar-dir", str(shared))

    first = _main_at(monkeypatch, tmp_path / "o1", "0.01", "columnar+parallel", *extra)
    assert "--reuse-db" not in first[0]
    assert Path(_flag(first[0], "--db-file")) == shared / "tpch_sf0.01_indexed.db"
    marker = shared / "tpch_sf0.01_indexed.loaded"
    assert json.loads(marker.read_text()) == {"exit_status": 0, "row_counts": {t: 2 for t in QUERY_TABLES}}

    # The folder outlives the call, so the next call reuses the file
    second = _main_at(monkeypatch, tmp_path / "o2", "0.01", "columnar+parallel,columnar+parallel_1thread", *extra)
    assert ["--reuse-db" in c for c in second] == [True, True]

    # Changed data makes the marker stale: the file is loaded again, then reused
    with (data / "sf0.01" / "region.tbl").open("a") as f:
        f.write("3|c|\n")
    third = _main_at(monkeypatch, tmp_path / "o3", "0.01", "columnar+parallel,columnar+parallel_1thread", *extra)
    assert ["--reuse-db" in c for c in third] == [False, True]
    assert json.loads(marker.read_text())["row_counts"]["region"] == 3


def test_columnar_dir_reloads_on_a_bad_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _tiny_data(tmp_path / "data")
    shared = tmp_path / "columnar"
    extra = ("--data-dir", str(data), "--columnar-dir", str(shared))
    _main_at(monkeypatch, tmp_path / "o0", "0.01", "columnar+parallel", *extra)
    marker = shared / "tpch_sf0.01_indexed.loaded"
    good = marker.read_text()

    bad_markers = [json.dumps({"exit_status": 1, "row_counts": {t: 2 for t in QUERY_TABLES}}), "not json", "[]"]
    for i, text in enumerate(bad_markers):
        marker.write_text(text)
        cmd = _main_at(monkeypatch, tmp_path / f"o{i + 1}", "0.01", "columnar+parallel", *extra)[0]
        assert "--reuse-db" not in cmd, text
        assert marker.read_text() == good

    # A marker without its file
    (shared / "tpch_sf0.01_indexed.db").unlink()
    cmd = _main_at(monkeypatch, tmp_path / "o9", "0.01", "columnar+parallel", *extra)[0]
    assert "--reuse-db" not in cmd


def test_columnar_dir_failed_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _tiny_data(tmp_path / "data")
    shared = tmp_path / "columnar"
    extra = ("--data-dir", str(data), "--columnar-dir", str(shared))
    _main_at(monkeypatch, tmp_path / "o1", "0.01", "columnar+parallel", *extra)
    marker = shared / "tpch_sf0.01_indexed.loaded"
    good = marker.read_text()

    # A failed reuse leaves the marker as it was
    cmd = _main_at(monkeypatch, tmp_path / "o2", "0.01", "columnar+parallel", *extra, fails=lambda i, c: True)[0]
    assert "--reuse-db" in cmd
    assert marker.read_text() == good

    # A load deletes the marker first; a failed one writes none,
    # so the next run loads again
    (data / "sf0.01" / "nation.tbl").write_text("1|a|\n")
    commands = _main_at(monkeypatch, tmp_path / "o3", "0.01", "columnar+parallel,columnar+parallel_1thread", *extra,
                        fails=lambda i, c: i == 0)
    assert ["--reuse-db" in c for c in commands] == [False, False]
    assert json.loads(marker.read_text())["row_counts"]["nation"] == 1


def test_without_columnar_dir_the_files_stay_temporary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _main_at(monkeypatch, tmp_path / "out", "0.01", "columnar+parallel,columnar+parallel_1thread")
    assert ["--reuse-db" in c for c in commands] == [False, True]
    assert not Path(_flag(commands[0], "--db-file")).parent.exists()
