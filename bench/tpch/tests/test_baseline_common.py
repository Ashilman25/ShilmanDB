"""Shared helpers of the DuckDB and Postgres baseline runners (bench/tpch/baseline_common.py)."""
from __future__ import annotations

import csv
import importlib.util
import re
import sys
from pathlib import Path

TPCH_DIR = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("baseline_common", TPCH_DIR / "baseline_common.py")
assert spec and spec.loader
common = importlib.util.module_from_spec(spec)
sys.modules["baseline_common"] = common
spec.loader.exec_module(common)


def read_csv_rows(path: Path) -> list:
    with path.open(newline="") as f:
        return list(csv.reader(f))


def test_typed_schema_dates_and_decimals() -> None:
    schema = common.typed_schema("lineitem")
    assert len(schema) == 16
    assert ("l_shipdate", "DATE") in schema
    assert ("l_commitdate", "DATE") in schema
    assert ("l_receiptdate", "DATE") in schema
    assert ("l_extendedprice", "DECIMAL(15,2)") in schema
    assert ("l_comment", "VARCHAR") in schema
    assert ("l_orderkey", "INTEGER") in schema
    assert ("o_orderdate", "DATE") in common.typed_schema("orders")


def test_typed_schema_keeps_sqlite_column_order() -> None:
    for table in common.TABLE_LOAD_ORDER:
        names = [name for name, _ in common.typed_schema(table)]
        assert names == [name for name, _ in common.TABLE_SCHEMAS[table]]


def test_index_defs_match_tpch_workload() -> None:
    header = (TPCH_DIR / "tpch_workload.hpp").read_text()
    block = re.search(r"IndexDescs\(\).*?return \{(.*?)\};", header, re.S)
    assert block
    entries = re.findall(r'\{"(\w+)", "(\w+)", "(\w+)"\}', block.group(1))
    assert len(entries) == 8
    expected = [(name, table, column) for table, name, column in entries]
    assert common.INDEX_DEFS_8 == expected


def test_query_order_and_texts_come_from_sqlite_runner() -> None:
    assert common.QUERY_ORDER == ["Q1", "Q6", "Q3", "Q5", "Q10", "Q12", "Q14", "Q19"]
    assert set(common.QUERY_ORDER) == set(common.QUERIES)
    for tag in common.QUERY_ORDER:
        assert common.query_text("duckdb", tag) == common.QUERIES[tag]


def test_query_text_uses_override(monkeypatch) -> None:
    monkeypatch.setitem(common.QUERY_OVERRIDES, "duckdb", {"Q6": "SELECT 1"})
    assert common.query_text("duckdb", "Q6") == "SELECT 1"
    assert common.query_text("duckdb", "Q1") == common.QUERIES["Q1"]
    assert common.query_text("postgres", "Q6") == common.QUERIES["Q6"]


def test_stripped_lines_drop_trailing_pipe(tmp_path: Path) -> None:
    tbl = tmp_path / "t.tbl"
    tbl.write_text("1|a|\n2|b|\n")
    assert list(common.stripped_lines(tbl)) == ["1|a", "2|b"]


def test_stripped_lines_skip_blank_lines(tmp_path: Path) -> None:
    tbl = tmp_path / "t.tbl"
    tbl.write_text("1|a|\n\n2|b|")
    assert list(common.stripped_lines(tbl)) == ["1|a", "2|b"]


def test_stripped_lines_keep_empty_last_field(tmp_path: Path) -> None:
    tbl = tmp_path / "t.tbl"
    tbl.write_text("1|a||\n")
    assert list(common.stripped_lines(tbl)) == ["1|a|"]


def test_stripped_lines_handle_crlf(tmp_path: Path) -> None:
    tbl = tmp_path / "t.tbl"
    tbl.write_bytes(b"1|a|\r\n2|b||\r\n")
    assert list(common.stripped_lines(tbl)) == ["1|a", "2|b|"]


def test_tbl_row_counts(tmp_path: Path) -> None:
    (tmp_path / "region.tbl").write_text("0|AFRICA|x|\n1|AMERICA|y|\n\n")
    (tmp_path / "nation.tbl").write_text("0|ALGERIA|0|z|\n")
    (tmp_path / "partsupp.tbl").write_text("1|2|3|\n")  # not a table the queries use
    assert common.tbl_row_counts(tmp_path) == {"region": 2, "nation": 1}


def test_time_query_warmup_not_timed() -> None:
    calls = []

    def run() -> list:
        calls.append(1)
        return [(len(calls),)]

    latencies, rows = common.time_query(run, 3)
    assert len(calls) == 4
    assert len(latencies) == 3
    assert all(lat >= 0.0 for lat in latencies)
    assert rows == [(4,)]


def test_write_engine_latencies(tmp_path: Path) -> None:
    path = tmp_path / "out" / "lat.csv"
    common.write_engine_latencies(
        [{"sf": "0.01", "query": "Q1", "config": "duckdb", "run": 1, "latency_ms": 1.23456}], path
    )
    rows = read_csv_rows(path)
    assert rows == [["sf", "query", "config", "run", "latency_ms"], ["0.01", "Q1", "duckdb", "1", "1.2346"]]


def test_write_info(tmp_path: Path) -> None:
    path = tmp_path / "info.csv"
    common.write_info({"duckdb_version": "2.0.0", "load_seconds_sf0.01": "1.5"}, path)
    rows = read_csv_rows(path)
    assert rows == [["key", "value"], ["duckdb_version", "2.0.0"], ["load_seconds_sf0.01", "1.5"]]


def test_reuse_requires_complete_marker() -> None:
    assert common.marker_matches({"lineitem": 10}, {"lineitem": 10, "orders": 5}) is False
    assert common.marker_matches({"lineitem": 10, "orders": 5}, {"lineitem": 10, "orders": 5}) is True
    assert common.marker_matches({"lineitem": 9, "orders": 5}, {"lineitem": 10, "orders": 5}) is False
    assert common.marker_matches({}, {}) is False


def _one_table_dir(tmp_path: Path) -> Path:
    data = tmp_path / "sf"
    data.mkdir()
    for table in common.TABLE_LOAD_ORDER:
        (data / f"{table}.tbl").write_text("1|a|\n")
    return data


def test_ensure_loaded_prints_detail_after_load(tmp_path: Path, capsys) -> None:
    data = _one_table_dir(tmp_path)
    seconds = common.ensure_loaded("db x", data, False, lambda: {}, lambda expected: 1.5,
                                   detail=" (COPY, 8 indexes)")
    assert seconds == 1.5
    assert "  Loaded 7 rows in 1.50 s (COPY, 8 indexes)\n" in capsys.readouterr().out


def test_ensure_loaded_reuses_matching_marker(tmp_path: Path, capsys) -> None:
    data = _one_table_dir(tmp_path)
    marker = {table: 1 for table in common.TABLE_LOAD_ORDER}
    assert common.ensure_loaded("db x", data, False, lambda: marker, lambda expected: 1.0) is None
    assert "  Reusing db x (row counts match)\n" in capsys.readouterr().out
    assert common.ensure_loaded("db x", data, True, lambda: marker, lambda expected: 1.0) == 1.0
    assert "  Loaded 7 rows in 1.00 s\n" in capsys.readouterr().out
