"""DuckDB baseline runner (bench/tpch/run_duckdb_benchmarks.py) on tiny .tbl fixtures."""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import duckdb

TPCH_DIR = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("run_duckdb_benchmarks", TPCH_DIR / "run_duckdb_benchmarks.py")
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["run_duckdb_benchmarks"] = mod
spec.loader.exec_module(mod)

helper_spec = importlib.util.spec_from_file_location("tpch_tiny_tables", Path(__file__).with_name("tpch_tiny_tables.py"))
assert helper_spec and helper_spec.loader
tiny = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(tiny)
write_tiny_tbl_files = tiny.write_tiny_tbl_files
LINEITEM_COMMENTS = tiny.LINEITEM_COMMENTS


def read_csv_rows(path: Path) -> list:
    with path.open(newline="") as f:
        return list(csv.reader(f))


def read_info(path: Path) -> dict:
    return {row[0]: row[1] for row in read_csv_rows(path)[1:]}


def run_main(data_dir: Path, out: Path, db_dir: Path, *extra: str) -> int:
    return mod.main([
        "--data-dir", str(data_dir), "--sf", "0.01", "--runs", "1",
        "--output-dir", str(out), "--db-dir", str(db_dir), *extra,
    ])


def test_configs() -> None:
    assert mod.CONFIGS == {"duckdb": [], "duckdb_1thread": ["SET threads = 1"]}


def test_duckdb_load_strips_trailing_pipe(tmp_path: Path) -> None:
    data = write_tiny_tbl_files(tmp_path / "sf0.01")
    db_path = tmp_path / "tpch.duckdb"
    seconds = mod.load_database(db_path, data)
    assert seconds >= 0.0

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        comments = [r[0] for r in con.execute("SELECT l_comment FROM lineitem ORDER BY l_orderkey").fetchall()]
        assert comments == LINEITEM_COMMENTS
        types = dict(con.execute(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'lineitem'"
        ).fetchall())
        assert len(types) == 16
        assert types["l_shipdate"] == "DATE"
        assert types["l_extendedprice"] == "DECIMAL(15,2)"
        marker = dict(con.execute("SELECT table_name, row_count FROM _baseline_load").fetchall())
        assert marker == {table: 2 for table in mod.common.TABLE_LOAD_ORDER}
    finally:
        con.close()


def test_duckdb_runs_and_writes_layout(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_tiny_tbl_files(data_dir / "sf0.01")
    out = tmp_path / "out"
    assert run_main(data_dir, out, tmp_path / "db") == 0

    for config in mod.CONFIGS:
        for tag in mod.common.QUERY_ORDER:
            path = out / "duckdb_results" / config / f"{tag}_sf0.01.csv"
            assert path.exists(), path
            assert read_csv_rows(path)[0] == mod.common.QUERY_HEADERS[tag]

    latencies = read_csv_rows(out / "duckdb_latencies.csv")
    assert latencies[0] == ["sf", "query", "config", "run", "latency_ms"]
    assert len(latencies) - 1 == 16
    assert (tmp_path / "db" / "tpch_sf0.01.duckdb").exists()

    info = read_info(out / "duckdb_info.csv")
    assert info["duckdb_version"] == duckdb.__version__
    assert info["threads_duckdb_1thread"] == "1"
    assert "memory_limit_duckdb" in info
    assert "load_seconds_sf0.01" in info


def test_duckdb_reuses_loaded_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    sf_dir = write_tiny_tbl_files(data_dir / "sf0.01")
    db_dir = tmp_path / "db"
    assert run_main(data_dir, tmp_path / "first", db_dir, "--configs", "duckdb") == 0
    assert "load_seconds_sf0.01" in read_info(tmp_path / "first" / "duckdb_info.csv")

    assert run_main(data_dir, tmp_path / "second", db_dir, "--configs", "duckdb") == 0
    assert "load_seconds_sf0.01" not in read_info(tmp_path / "second" / "duckdb_info.csv")

    assert run_main(data_dir, tmp_path / "third", db_dir, "--configs", "duckdb", "--reload") == 0
    assert "load_seconds_sf0.01" in read_info(tmp_path / "third" / "duckdb_info.csv")

    # A changed data file no longer matches the marker, so the file is rebuilt.
    with (sf_dir / "region.tbl").open("a") as f:
        f.write("3|r_name_3|r_comment_3|\n")
    assert run_main(data_dir, tmp_path / "fourth", db_dir, "--configs", "duckdb") == 0
    assert "load_seconds_sf0.01" in read_info(tmp_path / "fourth" / "duckdb_info.csv")


def test_rejects_unknown_sf_and_config(tmp_path: Path) -> None:
    common_args = ["--data-dir", str(tmp_path), "--output-dir", str(tmp_path / "o"), "--db-dir", str(tmp_path / "d")]
    assert mod.main([*common_args, "--sf", "5"]) == 1
    assert mod.main([*common_args, "--sf", "0.01", "--configs", "nope"]) == 1


def test_sf_and_configs_split_on_commas() -> None:
    assert mod.resolve_scale_factors(["0.01,0.1"]) == ["0.01", "0.1"]
    assert mod.resolve_scale_factors(["0.01", "0.1,1.0"]) == ["0.01", "0.1", "1.0"]
    assert mod.resolve_configs(["duckdb,duckdb_1thread"]) == ["duckdb", "duckdb_1thread"]


def test_empty_text_field_stays_empty(tmp_path: Path) -> None:
    data = write_tiny_tbl_files(tmp_path / "sf0.01")
    (data / "region.tbl").write_text("1||r_comment_1|\n2|r_name_2|r_comment_2|\n")
    db_path = tmp_path / "tpch.duckdb"
    mod.load_database(db_path, data)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        names = con.execute("SELECT r_name FROM region ORDER BY r_regionkey").fetchall()
    finally:
        con.close()
    assert names == [("",), ("r_name_2",)]


def test_warns_when_duckdb_version_is_not_pinned(monkeypatch, capsys) -> None:
    monkeypatch.setattr(mod.duckdb, "__version__", "1.9.9")
    assert mod.warn_if_unpinned() is True
    err = capsys.readouterr().err
    assert "1.9.9" in err and "2.0.0" in err

    monkeypatch.setattr(mod.duckdb, "__version__", "2.0.0")
    assert mod.warn_if_unpinned() is False
    assert capsys.readouterr().err == ""


def test_missing_marker_rebuilds(tmp_path: Path) -> None:
    data = write_tiny_tbl_files(tmp_path / "sf0.01")
    db_path = tmp_path / "tpch.duckdb"
    mod.load_database(db_path, data)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("DROP TABLE _baseline_load")
    finally:
        con.close()

    assert mod.ensure_database(db_path, data, reload=False) is not None
    assert mod.ensure_database(db_path, data, reload=False) is None


def test_duckdb_honors_warmup(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    write_tiny_tbl_files(data_dir / "sf0.01")
    seen = []
    real = mod.common.time_query

    def spy(run, runs, warmup=1):
        seen.append((runs, warmup))
        return real(run, runs, warmup)

    monkeypatch.setattr(mod.common, "time_query", spy)
    assert run_main(data_dir, tmp_path / "out", tmp_path / "db", "--configs", "duckdb", "--warmup", "3") == 0
    assert seen == [(1, 3)] * len(mod.common.QUERY_ORDER)

    seen.clear()
    assert run_main(data_dir, tmp_path / "out2", tmp_path / "db", "--configs", "duckdb") == 0
    assert seen == [(1, 1)] * len(mod.common.QUERY_ORDER)
