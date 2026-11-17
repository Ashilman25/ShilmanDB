"""Tests for run_benchmarks.py's columnar configs, their shared files and load_times copies."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable, List

import pytest

SPEC_PATH = Path(__file__).resolve().parents[1] / "run_benchmarks.py"
spec = importlib.util.spec_from_file_location("run_benchmarks_columnar_under_test", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["run_benchmarks_columnar_under_test"] = mod
spec.loader.exec_module(mod)

TWINS = {"columnar": "baseline", "columnar+index": "+index", "columnar+vectorized": "+vectorized"}
ROW_AND_COLUMNAR = "baseline,+index,+vectorized,columnar,columnar+index,columnar+vectorized"
LOAD_TIMES = "table,storage,threads,rows,ms,pages\nregion,columnar,4,5,1.000,1\nTOTAL,columnar,4,5,1.000,1\n"


# The value after a flag in a command.
def _flag(cmd: List[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


# Which config a load_tpch command runs, read from its flags.
def _config_of(cmd: List[str]) -> str:
    storage = _flag(cmd, "--storage") if "--storage" in cmd else "row"
    indexes = _flag(cmd, "--create-indexes") == "true"
    vectorized = _flag(cmd, "--execution-mode") == "vectorized"
    name = "+vectorized" if vectorized else ("+index" if indexes else "baseline")
    if storage == "columnar":
        name = {"baseline": "columnar", "+index": "columnar+index", "+vectorized": "columnar+vectorized"}[name]
    return name


# Stands in for load_tpch: writes a run's files, and load_times.csv unless it reused.
def _fake_load_tpch(commands: List[List[str]], fails: Callable[[int, List[str]], bool] = lambda i, c: False):
    def run(cmd, **kwargs):
        cmd = list(cmd)
        index = len(commands)
        commands.append(cmd)
        if fails(index, cmd):
            return mod.subprocess.CompletedProcess(cmd, 1, "", "load failed\n")
        results = Path(_flag(cmd, "--results-dir"))
        results.mkdir(parents=True, exist_ok=True)
        (results / "latencies.csv").write_text(
            "query,run,rows,latency_ms,hits,misses,visits,reads,read_ms\n"
            + "".join(f"{q},0,1,1.0,10,1,2,1,0.1\n" for q in mod.QUERIES))
        (results / "run_info.csv").write_text("key,value\nsf,0.01\ndb_pages,7\neviction,lru\n")
        if "--workload" in cmd:
            (results / "stream.csv").write_text("pass\n")
            (results / "stream_queries.csv").write_text("seq\n")
            (results / "stream_results.csv").write_text("seq\n")
            (results / "stream_totals.csv").write_text(
                "pass,warmup,items,latency_ms,hits,misses,visits,reads,read_ms,query_misses,lookup_misses\n"
                "0,0,4,1.0,10,1,2,1,0.1,1,0\n")
        if "--reuse-db" not in cmd:
            (results / "load_times.csv").write_text(LOAD_TIMES)
        return mod.subprocess.CompletedProcess(cmd, 0, "", "")
    return run


# Runs the runner's main with the fake load_tpch and returns every command it ran.
def _run_main(monkeypatch: pytest.MonkeyPatch, out: Path, configs: str, *extra: str,
              fails: Callable[[int, List[str]], bool] = lambda i, c: False) -> List[List[str]]:
    commands: List[List[str]] = []
    monkeypatch.setattr(mod.subprocess, "run", _fake_load_tpch(commands, fails))
    monkeypatch.setattr(mod, "workload_query_rows", lambda *a, **k: [])
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--sf", "0.01", "--configs", configs,
                                      "--skip-correctness", "--output-dir", str(out), *extra])
    assert mod.main() == 0
    return commands


# The commands by config name (the last one wins).
def _by_config(commands: List[List[str]]) -> dict:
    return {_config_of(c): c for c in commands}


def test_columnar_configs_exist() -> None:
    for name, twin in TWINS.items():
        assert mod.PRIMARY_CONFIGS[name] == mod.PRIMARY_CONFIGS[twin] + ["--storage", "columnar"]


def test_all_configs_unchanged() -> None:
    assert mod.resolve_configs("all") == ["baseline", "+index", "+learned", "+vectorized",
                                          "sqlite_disk", "sqlite_memory"]
    assert mod.ALL_CONFIG_NAMES == ["baseline", "+index", "+learned", "+vectorized",
                                    "sqlite_disk", "sqlite_memory"]
    assert mod.resolve_configs("columnar,columnar+index,columnar+vectorized") == list(TWINS)


def test_unknown_config_error_lists_columnar_names() -> None:
    with pytest.raises(SystemExit) as err:
        mod.resolve_configs("columnar,nonsense")
    assert "columnar+vectorized" in str(err.value)


def test_row_commands_are_unchanged(tmp_path: Path) -> None:
    cmd = mod.build_command(
        binary=Path("/bin/fake"), sf="0.01", config_name="+vectorized",
        data_dir=tmp_path, results_dir=tmp_path / "r", db_file=tmp_path / "db",
        pool_size=128, warmup_runs=1, timed_runs=5,
    )
    assert cmd == ["/bin/fake", "--sf", "0.01", "--db-file", str(tmp_path / "db"),
                   "--data-dir", str(tmp_path / "sf0.01"), "--pool-size", "128", "--warmup", "1",
                   "--runs", "5", "--results-dir", str(tmp_path / "r"), "--no-verify",
                   "--create-indexes", "true", "--execution-mode", "vectorized"]


def test_reuse_flag_goes_after_config_flags(tmp_path: Path) -> None:
    cmd = mod.build_command(
        binary=Path("/bin/fake"), sf="0.01", config_name="columnar+vectorized",
        data_dir=tmp_path, results_dir=tmp_path, db_file=tmp_path / "db",
        pool_size=128, warmup_runs=1, timed_runs=5, reuse_db=True, extra_flags=["--direct-io"],
    )
    assert cmd[-5:] == ["vectorized", "--storage", "columnar", "--reuse-db", "--direct-io"]


def test_columnar_index_file_shared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _run_main(monkeypatch, tmp_path / "out", ROW_AND_COLUMNAR)
    assert [_config_of(c) for c in commands] == ROW_AND_COLUMNAR.split(",")
    by = _by_config(commands)

    assert "--reuse-db" not in by["columnar+index"]
    assert "--reuse-db" in by["columnar+vectorized"]
    assert _flag(by["columnar+vectorized"], "--db-file") == _flag(by["columnar+index"], "--db-file")
    assert "--reuse-db" not in by["columnar"]
    assert _flag(by["columnar"], "--db-file") != _flag(by["columnar+index"], "--db-file")


def test_row_commands_in_a_mixed_call_match_today(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _run_main(monkeypatch, tmp_path / "out", ROW_AND_COLUMNAR)
    row = [c for c in commands if "--storage" not in c]
    assert len(row) == 3
    db_files = {_flag(c, "--db-file") for c in row}
    assert len(db_files) == 3
    for cmd in row:
        assert "--reuse-db" not in cmd
        # Each row run has its own temp folder holding both its results and its file
        assert Path(_flag(cmd, "--db-file")).parent == Path(_flag(cmd, "--results-dir"))
        expected = mod.build_command(
            binary=mod.DEFAULT_BINARY, sf="0.01", config_name=_config_of(cmd), data_dir=mod.DEFAULT_DATA_DIR,
            results_dir=Path(_flag(cmd, "--results-dir")), db_file=Path(_flag(cmd, "--db-file")),
            pool_size=mod.POOL_SIZES["0.01"], warmup_runs=1, timed_runs=5)
        assert cmd == expected


def test_vectorized_alone_loads_its_own_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _run_main(monkeypatch, tmp_path / "out", "columnar+vectorized")
    assert len(commands) == 1
    assert "--reuse-db" not in commands[0]
    assert _flag(commands[0], "--storage") == "columnar"


def test_failed_load_is_not_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _run_main(monkeypatch, tmp_path / "out", "columnar+index,columnar+vectorized",
                         fails=lambda i, c: i == 0)
    assert len(commands) == 2
    assert "--reuse-db" not in commands[1]
    assert _flag(commands[1], "--db-file") == _flag(commands[0], "--db-file")


def test_reuse_across_workloads_and_policies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _run_main(monkeypatch, tmp_path / "out", "columnar+index",
                         "--workloads", "single,stream", "--eviction-policies", "lru,clock")
    assert len(commands) == 4
    assert ["--reuse-db" in c for c in commands] == [False, True, True, True]
    assert len({_flag(c, "--db-file") for c in commands}) == 1


def test_each_scale_factor_has_its_own_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _run_main(monkeypatch, tmp_path / "out", "columnar+index", "--sf", "0.1")
    assert [_flag(c, "--sf") for c in commands] == ["0.01", "0.1"]
    assert ["--reuse-db" in c for c in commands] == [False, False]
    assert _flag(commands[0], "--db-file") != _flag(commands[1], "--db-file")


def test_shared_files_are_removed_at_the_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _run_main(monkeypatch, tmp_path / "out", "columnar,columnar+index")
    for cmd in commands:
        assert not Path(_flag(cmd, "--db-file")).parent.exists()


def test_shared_files_are_removed_on_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands: List[List[str]] = []
    fake = _fake_load_tpch(commands)

    def run_then_crash(cmd, **kwargs):
        fake(cmd, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(mod.subprocess, "run", run_then_crash)
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--sf", "0.01", "--configs", "columnar",
                                      "--skip-correctness", "--output-dir", str(tmp_path / "out")])
    with pytest.raises(KeyboardInterrupt):
        mod.main()
    assert not Path(_flag(commands[0], "--db-file")).parent.exists()


def test_load_times_copied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    _run_main(monkeypatch, out, ROW_AND_COLUMNAR, "--miss-penalty-us", "0,100")
    copied = sorted(p.name for p in out.glob("load_times_*.csv"))
    # The reused runs (columnar+vectorized, and every columnar+index run after the first) wrote none
    assert copied == sorted([
        "load_times_baseline_sf0.01.csv", "load_times_baseline_pen100_sf0.01.csv",
        "load_times_+index_sf0.01.csv", "load_times_+index_pen100_sf0.01.csv",
        "load_times_+vectorized_sf0.01.csv", "load_times_+vectorized_pen100_sf0.01.csv",
        "load_times_columnar_sf0.01.csv", "load_times_columnar+index_sf0.01.csv",
    ])
    assert (out / "load_times_columnar+index_sf0.01.csv").read_text() == LOAD_TIMES


def test_load_times_copied_for_workloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    _run_main(monkeypatch, out, "baseline,columnar", "--workloads", "stream")
    assert sorted(p.name for p in out.glob("load_times_*.csv")) == [
        "load_times_baseline_stream_s1_sf0.01.csv", "load_times_columnar_stream_s1_sf0.01.csv"]


def test_db_pages_warning_only_for_row_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture) -> None:
    _run_main(monkeypatch, tmp_path / "out", "columnar,columnar+index,columnar+vectorized")
    assert "db_pages" not in capsys.readouterr().err
    _run_main(monkeypatch, tmp_path / "out2", "baseline")
    assert "db_pages=7" in capsys.readouterr().err


# The row-run check accepts heap pages alone or heap pages plus the two header pages.
@pytest.mark.parametrize("extra, warns", [(0, False), (2, False), (1, True), (3, True)])
def test_db_pages_check_accepts_header_pages(extra: int, warns: bool, capsys: pytest.CaptureFixture) -> None:
    assert mod.HEADER_PAGES == 2
    reported = mod.DB_PAGES["0.01"] + extra
    mod.warn_on_db_pages_mismatch({"db_pages": str(reported)}, "0.01")
    err = capsys.readouterr().err
    assert ("warning" in err) == warns
    if warns:
        assert f"db_pages={reported}" in err
