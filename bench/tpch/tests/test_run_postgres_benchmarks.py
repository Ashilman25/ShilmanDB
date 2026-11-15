"""Postgres baseline runner (bench/tpch/run_postgres_benchmarks.py) and its server script."""
from __future__ import annotations

import csv
import importlib.util
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("psycopg")

TPCH_DIR = Path(__file__).resolve().parents[1]
SERVER_SCRIPT = TPCH_DIR / "pg_baseline_server.sh"

spec = importlib.util.spec_from_file_location("run_postgres_benchmarks", TPCH_DIR / "run_postgres_benchmarks.py")
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["run_postgres_benchmarks"] = mod
spec.loader.exec_module(mod)

helper_spec = importlib.util.spec_from_file_location("tpch_tiny_tables", Path(__file__).with_name("tpch_tiny_tables.py"))
assert helper_spec and helper_spec.loader
tiny = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(tiny)


def read_csv_rows(path: Path) -> list:
    with path.open(newline="") as f:
        return list(csv.reader(f))


def read_info(path: Path) -> dict:
    return {row[0]: row[1] for row in read_csv_rows(path)[1:]}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# A copy of the server script in tmp_path/repo with a bogus PG_BIN, so nothing real ever runs.
# Returns the run function and the copy's data folder (not created).
def script_copy(tmp_path: Path):
    script_dir = tmp_path / "repo" / "bench" / "tpch"
    script_dir.mkdir(parents=True)
    script = script_dir / "pg_baseline_server.sh"
    shutil.copy(SERVER_SCRIPT, script)
    data = tmp_path / "repo" / "bench" / "results" / "tpch_postgres" / "pgdata"

    def run(*args: str, **env: str) -> subprocess.CompletedProcess:
        full_env = {k: v for k, v in os.environ.items() if k != "PG_PORT"}
        full_env.update({"PG_BIN": str(tmp_path / "no_bin"), **env})
        return subprocess.run(["bash", str(script), *args], capture_output=True, text=True, env=full_env, timeout=30)

    return run, data


# The copy's data folder holding the given files.
def fake_data_folder(data: Path, files: dict) -> Path:
    data.mkdir(parents=True)
    for name, text in files.items():
        (data / name).write_text(text)
    return data


# True when the baseline server answers on its port with the baseline settings.
def baseline_server_ready() -> bool:
    try:
        conn = mod.connect("postgres", mod.PG_PORT)
    except mod.psycopg.Error:
        return False
    try:
        return mod.settings_mismatches(mod.show_settings(conn)) == []
    finally:
        conn.close()


def test_constants() -> None:
    assert mod.PG_PORT == 5434
    assert mod.PG_HOST == "/tmp"
    assert mod.PG_DATA == "bench/results/tpch_postgres/pgdata"
    assert mod.PG_SETTINGS == {
        "shared_buffers": "16GB", "effective_cache_size": "48GB", "work_mem": "256MB",
        "maintenance_work_mem": "2GB", "max_parallel_workers_per_gather": "2",
        "max_parallel_workers": "8", "max_worker_processes": "8", "jit": "on",
    }
    assert mod.CONFIGS == {"postgres": [], "postgres_1worker": ["SET max_parallel_workers_per_gather = 0"]}


def test_settings_mismatch_lists_each_difference() -> None:
    shown = {**mod.PG_SETTINGS, "work_mem": "4MB", "jit": "off"}
    assert mod.settings_mismatches(shown) == ["work_mem: got 4MB, want 256MB", "jit: got off, want on"]


def test_settings_all_match_is_empty() -> None:
    assert mod.settings_mismatches(dict(mod.PG_SETTINGS)) == []


def test_settings_missing_counts_as_mismatch() -> None:
    shown = {k: v for k, v in mod.PG_SETTINGS.items() if k != "jit"}
    assert mod.settings_mismatches(shown) == ["jit: got None, want on"]


def test_copy_lines_strip_trailing_pipe(tmp_path: Path) -> None:
    tbl = tmp_path / "t.tbl"
    tbl.write_text("1|a|\n")
    assert list(mod.copy_lines(tbl)) == ["1|a\n"]


def test_copy_lines_double_backslashes(tmp_path: Path) -> None:
    tbl = tmp_path / "t.tbl"
    tbl.write_text("1|a\\b|\n")
    assert list(mod.copy_lines(tbl)) == ["1|a\\\\b\n"]


def test_db_name() -> None:
    assert mod.db_name("1.0") == "tpch_sf1_0"
    assert mod.db_name("10") == "tpch_sf10"
    assert mod.db_name("0.01") == "tpch_sf0_01"


def test_print_conf_has_port_and_settings(capsys) -> None:
    assert mod.main(["--print-conf"]) == 0
    out = capsys.readouterr().out
    assert "port = 5434" in out
    assert "shared_buffers = '16GB'" in out
    assert "unix_socket_directories = '/tmp'" in out
    for name in mod.PG_SETTINGS:
        assert f"\n{name} = " in out


def test_no_server_message(tmp_path: Path, capsys) -> None:
    port = free_port()
    code = mod.main(["--port", str(port), "--sf", "0.01", "--data-dir", str(tmp_path),
                     "--output-dir", str(tmp_path / "out")])
    assert code == 2
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert f"No Postgres on port {port}." in text
    assert "Start it with: bash bench/tpch/pg_baseline_server.sh start" in text
    assert not (tmp_path / "out").exists()


def test_no_server_message_default_wording() -> None:
    assert mod.no_server_message(5434) == (
        "No Postgres on port 5434. Start it with: bash bench/tpch/pg_baseline_server.sh start"
    )


def test_wrong_settings_exit_2_listing_each(monkeypatch, tmp_path: Path, capsys) -> None:
    class FakeConn:
        def close(self) -> None:
            pass

    monkeypatch.setattr(mod, "connect", lambda dbname, port: FakeConn())
    monkeypatch.setattr(mod, "show_settings", lambda conn: {**mod.PG_SETTINGS, "work_mem": "4MB", "jit": "off"})
    code = mod.main(["--sf", "0.01", "--data-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")])
    assert code == 2
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "work_mem: got 4MB, want 256MB" in text
    assert "jit: got off, want on" in text
    assert not (tmp_path / "out").exists()


def test_no_server_with_empty_error_message(monkeypatch, tmp_path: Path, capsys) -> None:
    def refuse(dbname, port):
        raise mod.psycopg.OperationalError("")

    monkeypatch.setattr(mod, "connect", refuse)
    code = mod.main(["--sf", "0.01", "--data-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")])
    assert code == 2
    assert mod.no_server_message(mod.PG_PORT) in capsys.readouterr().out


def test_runner_refuses_5433(tmp_path: Path, capsys) -> None:
    assert mod.main(["--port", "5433", "--sf", "0.01", "--data-dir", str(tmp_path)]) == 1
    assert "5433" in capsys.readouterr().out


def test_print_conf_refuses_5433(capsys) -> None:
    assert mod.main(["--print-conf", "--port", "5433"]) == 1
    out = capsys.readouterr().out
    assert "port = 5433" not in out
    assert "vector-search" in out


def test_rejects_unknown_sf_and_config(tmp_path: Path) -> None:
    common_args = ["--data-dir", str(tmp_path), "--output-dir", str(tmp_path / "o"), "--port", str(free_port())]
    assert mod.main([*common_args, "--sf", "5"]) == 1
    assert mod.main([*common_args, "--sf", "0.01", "--configs", "nope"]) == 1


def test_server_script_refuses_5433(tmp_path: Path) -> None:
    run, _ = script_copy(tmp_path)
    result = run("status", PG_PORT="5433")
    assert result.returncode != 0
    assert "5433" in result.stdout + result.stderr


def test_server_script_init_refuses_existing_folder(tmp_path: Path) -> None:
    run, data = script_copy(tmp_path)
    data.mkdir(parents=True)
    result = run("init")
    assert result.returncode != 0
    assert "already exists" in result.stdout + result.stderr
    assert list(data.iterdir()) == []


def test_server_script_has_no_data_folder_override() -> None:
    text = SERVER_SCRIPT.read_text()
    assert "${PG_DATA:-" not in text
    assert 'PG_DATA="$REPO_ROOT/bench/results/tpch_postgres/pgdata"' in text


# The shape of a folder started with pg_ctl -o "-p 5433 ...": no port in its conf.
OPTS_5433 = ('/opt/homebrew/Cellar/postgresql@17/17.11_1/bin/postgres "-D" "some/pgdata" "-p" "5433" '
             '"-k" "/tmp" "-c" "shared_buffers=8GB" "-c" "jit=off"\n')
PID_5433 = "12345\n/some/pgdata\n1763100000\n5433\n/tmp\nlocalhost\n  5433001    65536\nready   \n"


@pytest.mark.parametrize("command", ["start", "stop", "status"])
@pytest.mark.parametrize("files", [
    {"postgresql.conf": "port = 5434\n#port = 5432\nport = 5433   # the other server\n"},
    {"postgresql.conf": "port = 5434\n", "postgresql.auto.conf": "port = '5433'\n"},
    {"postgresql.conf": "#port = 5432\n", "postgresql.auto.conf": "", "postmaster.opts": OPTS_5433},
    {"postgresql.conf": "#port = 5432\n", "postmaster.pid": PID_5433},
    {"postgresql.conf": "#port = 5432\n", "postmaster.opts": '/x/postgres "-D" "d" "-c" "port=5433"\n'},
], ids=["conf", "auto_conf", "postmaster_opts", "postmaster_pid", "opts_c_port"])
def test_server_script_refuses_folder_on_5433(tmp_path: Path, command: str, files: dict) -> None:
    run, data = script_copy(tmp_path)
    fake_data_folder(data, files)
    result = run(command)
    assert result.returncode != 0
    assert "5433" in result.stderr and "vector-search" in result.stderr


def test_server_script_passes_folder_on_5434(tmp_path: Path) -> None:
    run, data = script_copy(tmp_path)
    fake_data_folder(data, {
        "postgresql.conf": "port = 5433\n#port = 5433\nport = 5434\n",
        "postmaster.opts": '/x/postgres "-D" "/repo/bench/results/tpch_postgres/pgdata"\n',
        "postmaster.pid": PID_5433.replace("\n5433\n", "\n5434\n"),
    })
    result = run("status")
    # It gets past the guard and fails only because PG_BIN has no pg_ctl.
    assert "vector-search" not in result.stdout + result.stderr
    assert "no_bin/pg_ctl" in result.stderr


def test_server_script_rejects_unknown_command(tmp_path: Path) -> None:
    run, _ = script_copy(tmp_path)
    result = run("restart-all")
    assert result.returncode != 0
    assert "init|start|stop|status" in result.stdout + result.stderr


@pytest.mark.skipif(not baseline_server_ready(), reason="no baseline Postgres on port 5434 with the baseline settings")
def test_postgres_end_to_end(tmp_path: Path, monkeypatch) -> None:
    # A database of its own, so the real tpch_sf0_01 is never touched.
    test_db = f"tpch_test_{os.getpid()}"
    monkeypatch.setattr(mod, "db_name", lambda sf: test_db)
    data_dir = tmp_path / "data"
    tiny.write_tiny_tbl_files(data_dir / "sf0.01")
    args = ["--data-dir", str(data_dir), "--sf", "0.01", "--runs", "1"]
    try:
        out = tmp_path / "out"
        assert mod.main([*args, "--output-dir", str(out)]) == 0
        for config in mod.CONFIGS:
            for tag in mod.common.QUERY_ORDER:
                path = out / "postgres_results" / config / f"{tag}_sf0.01.csv"
                assert path.exists(), path
                assert read_csv_rows(path)[0] == mod.common.QUERY_HEADERS[tag]
        latencies = read_csv_rows(out / "postgres_latencies.csv")
        assert latencies[0] == ["sf", "query", "config", "run", "latency_ms"]
        assert len(latencies) - 1 == 16

        info = read_info(out / "postgres_info.csv")
        assert info["postgres_version"].startswith("PostgreSQL 17")
        for name, value in mod.PG_SETTINGS.items():
            assert info[name] == value
        assert info["max_parallel_workers_per_gather_postgres_1worker"] == "0"
        assert "load_seconds_sf0.01" in info

        conn = mod.connect(test_db, mod.PG_PORT)
        try:
            comments = [r[0] for r in conn.execute("SELECT l_comment FROM lineitem ORDER BY l_orderkey").fetchall()]
            indexes = {r[0] for r in conn.execute("SELECT indexname FROM pg_indexes WHERE schemaname = 'public'")}
            marker = dict(conn.execute(f"SELECT table_name, row_count FROM {mod.MARKER_TABLE}").fetchall())
        finally:
            conn.close()
        assert comments == tiny.LINEITEM_COMMENTS
        assert indexes == {name for name, _, _ in mod.common.INDEX_DEFS_8}
        assert marker == {table: 2 for table in mod.common.TABLE_LOAD_ORDER}

        # The second call reuses the loaded database.
        assert mod.main([*args, "--output-dir", str(tmp_path / "again"), "--configs", "postgres"]) == 0
        assert "load_seconds_sf0.01" not in read_info(tmp_path / "again" / "postgres_info.csv")
    finally:
        admin = mod.connect("postgres", mod.PG_PORT)
        try:
            admin.execute(f"DROP DATABASE IF EXISTS {test_db}")
        finally:
            admin.close()


# A stand-in connection: SHOW reads `settings`, SET changes them, everything else is logged.
class FakeSession:
    def __init__(self, settings: dict) -> None:
        self.settings = dict(settings)
        self.queries: list = []

    def execute(self, sql: str):
        words = sql.split()
        if words[0] == "SHOW":
            value = self.settings.get(words[1])
            return FakeResult([(value,)])
        if words[0] == "SET":
            self.settings[words[1]] = words[3]
            return FakeResult([])
        if sql == "SELECT version()":
            return FakeResult([("PostgreSQL 17.0 (fake)",)])
        self.queries.append(sql)
        return FakeResult([])

    def close(self) -> None:
        pass


class FakeResult:
    def __init__(self, rows: list) -> None:
        self.rows = rows

    def fetchone(self):
        return self.rows[0]

    def fetchall(self) -> list:
        return self.rows


def run_fake_config(monkeypatch, tmp_path: Path, config: str, settings: dict, **kwargs):
    session = FakeSession(settings)
    monkeypatch.setattr(mod, "connect", lambda dbname, port: session)
    result = mod.run_config("tpch_sf0_01", 5434, "0.01", config, 1, tmp_path / "out", **kwargs)
    return session, result


def test_session_settings_match_runs_queries(monkeypatch, tmp_path: Path) -> None:
    for config in mod.CONFIGS:
        session, (rows, settings) = run_fake_config(monkeypatch, tmp_path, config, mod.PG_SETTINGS)
        assert len(session.queries) == 2 * len(mod.common.QUERY_ORDER)     # 1 warmup + 1 timed each
        assert len(rows) == len(mod.common.QUERY_ORDER)
    assert settings == {"max_parallel_workers_per_gather_postgres_1worker": "0"}


@pytest.mark.parametrize("config, session_settings, line", [
    ("postgres", {"work_mem": "4MB"}, "work_mem: got 4MB, want 256MB"),
    ("postgres", {"max_parallel_workers_per_gather": "0"}, "max_parallel_workers_per_gather: got 0, want 2"),
    ("postgres_1worker", {"jit": "off"}, "jit: got off, want on"),
])
def test_session_settings_mismatch_refuses_before_timing(monkeypatch, tmp_path: Path, config, session_settings, line) -> None:
    session = FakeSession({**mod.PG_SETTINGS, **session_settings})
    monkeypatch.setattr(mod, "connect", lambda dbname, port: session)
    with pytest.raises(mod.SettingsMismatch) as e:
        mod.run_config("tpch_sf0_01", 5434, "0.01", config, 1, tmp_path / "out")
    assert line in str(e.value)
    assert "tpch_sf0_01" in str(e.value)
    assert session.queries == []
    assert not (tmp_path / "out").exists()


def test_1worker_needs_its_own_override(monkeypatch, tmp_path: Path) -> None:
    class IgnoresSet(FakeSession):
        def execute(self, sql: str):
            return FakeResult([]) if sql.startswith("SET") else super().execute(sql)

    session = IgnoresSet(mod.PG_SETTINGS)
    monkeypatch.setattr(mod, "connect", lambda dbname, port: session)
    with pytest.raises(mod.SettingsMismatch) as e:
        mod.run_config("tpch_sf0_01", 5434, "0.01", "postgres_1worker", 1, tmp_path / "out")
    assert "max_parallel_workers_per_gather: got 2, want 0" in str(e.value)
    assert session.queries == []


def test_benchmark_database_settings_checked_by_main(monkeypatch, tmp_path: Path, capsys) -> None:
    sessions = {"postgres": FakeSession(mod.PG_SETTINGS), "tpch_sf0_01": FakeSession({**mod.PG_SETTINGS, "work_mem": "4MB"})}
    monkeypatch.setattr(mod, "connect", lambda dbname, port: sessions[dbname])
    monkeypatch.setattr(mod, "ensure_database", lambda *a, **k: None)
    (tmp_path / "data" / "sf0.01").mkdir(parents=True)
    out = tmp_path / "out"
    code = mod.main(["--sf", "0.01", "--data-dir", str(tmp_path / "data"), "--output-dir", str(out)])
    assert code == 2
    text = capsys.readouterr().out
    assert "work_mem: got 4MB, want 256MB" in text
    assert sessions["tpch_sf0_01"].queries == []
    assert not (out / "postgres_latencies.csv").exists()


def test_postgres_honors_warmup(monkeypatch, tmp_path: Path) -> None:
    session, (rows, _) = run_fake_config(monkeypatch, tmp_path, "postgres", mod.PG_SETTINGS, warmup=3)
    assert len(session.queries) == 4 * len(mod.common.QUERY_ORDER)
    assert len(rows) == len(mod.common.QUERY_ORDER)


def test_warmup_flag_default() -> None:
    assert mod.parse_args([]).warmup == 1
    assert mod.parse_args(["--warmup", "0"]).warmup == 0
