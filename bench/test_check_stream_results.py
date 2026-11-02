"""Unit tests for bench/check_stream_results.py on small handwritten CSVs."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List

import pytest

SPEC_PATH = Path(__file__).with_name("check_stream_results.py")
spec = importlib.util.spec_from_file_location("check_stream_results", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)

sys.modules["check_stream_results"] = mod
spec.loader.exec_module(mod)

QUERIES_HEADER = "seq,kind,query,sql\n"
RESULTS_HEADER = "seq,row,col,value\n"


def write_folder(folder: Path, sqls: List[str]) -> None:
    """A folder whose items are the given SQL strings, each answered with one cell, 1."""
    folder.mkdir(parents=True, exist_ok=True)
    queries = QUERIES_HEADER + "".join(
        f'{i},tpch,Q{i},"{sql}"\n' for i, sql in enumerate(sqls)
    )
    results = RESULTS_HEADER + "".join(f"{i},0,0,1\n" for i in range(len(sqls)))
    (folder / "stream_queries.csv").write_text(queries)
    (folder / "stream_results.csv").write_text(results)


# compare_rows
def test_compare_rows_sorted_and_tolerance() -> None:
    expected = [["b", "2.0"], ["a", "1.0"]]
    assert mod.compare_rows(expected, [["a", "1.004"], ["b", "2.0"]]) is None
    message = mod.compare_rows(expected, [["a", "1.02"], ["b", "2.0"]])
    assert message is not None and "1.02" in message


def test_compare_rows_row_count_mismatch() -> None:
    message = mod.compare_rows([["1"], ["2"], ["3"]], [["1"], ["2"]])
    assert message is not None
    assert "3" in message and "2" in message


# sqlite_cells
def test_sqlite_cells_null_is_zero() -> None:
    assert mod.sqlite_cells((None, 3, 1.5, "x")) == ["0", "3", "1.5", "x"]
    assert mod.compare_rows([mod.sqlite_cells((0, None))], [["0", "0.000000"]]) is None


# load_items
def test_load_items_reads_quoted_cells(tmp_path: Path) -> None:
    (tmp_path / "stream_queries.csv").write_text(
        QUERIES_HEADER
        + '0,tpch,Q1,"SELECT a, b FROM t WHERE s = \'x\' AND c = ""q"""\n'
        + '1,lookup,LS,"SELECT COUNT(*) FROM t WHERE d = \'none\'"\n'
    )
    (tmp_path / "stream_results.csv").write_text(
        RESULTS_HEADER
        + '0,0,0,7\n'
        + '0,0,1,"12 Main St, Apt ""B"""\n'
        + '0,1,0,8\n'
        + '0,1,1,plain\n'
    )
    items = mod.load_items(tmp_path, "0.01")
    assert [item.seq for item in items] == [0, 1]
    assert [item.query for item in items] == ["Q1", "LS"]
    assert items[0].sql == 'SELECT a, b FROM t WHERE s = \'x\' AND c = "q"'
    assert items[0].rows == [["7", '12 Main St, Apt "B"'], ["8", "plain"]]
    assert items[1].rows == []


def test_load_items_prefers_sf_suffix(tmp_path: Path) -> None:
    write_folder(tmp_path, ["SELECT 1"])
    (tmp_path / "stream_queries_sf0.01.csv").write_text(
        QUERIES_HEADER + '5,lookup,LO,"SELECT 2"\n'
    )
    (tmp_path / "stream_results_sf0.01.csv").write_text(RESULTS_HEADER + "5,0,0,9\n")
    items = mod.load_items(tmp_path, "0.01")
    assert [(item.seq, item.sql, item.rows) for item in items] == [(5, "SELECT 2", [["9"]])]
    # Another scale factor has no suffixed files, so the plain names are read.
    assert [item.sql for item in mod.load_items(tmp_path, "0.1")] == ["SELECT 1"]


# check_folders
def test_check_runs_each_distinct_sql_once(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    write_folder(first, ["SELECT A", "SELECT B", "SELECT C"])
    write_folder(second, ["SELECT A", "SELECT B", "SELECT D"])
    calls: List[str] = []

    def run_sql(sql: str) -> List[tuple]:
        calls.append(sql)
        return [(1,)]

    rows = mod.check_folders("0.01", [first, second], run_sql)
    assert sorted(calls) == ["SELECT A", "SELECT B", "SELECT C", "SELECT D"]
    assert len(rows) == 6
    assert all(row.status == "PASS" for row in rows)
    assert [row.config for row in rows] == ["first"] * 3 + ["second"] * 3
    assert all(row.sf == "0.01" for row in rows)


def test_check_reports_fail(tmp_path: Path) -> None:
    write_folder(tmp_path / "run", ["SELECT A"])
    rows = mod.check_folders("0.01", [tmp_path / "run"], lambda sql: [(2,)])
    assert len(rows) == 1
    assert rows[0].status == "FAIL"
    assert rows[0].detail != ""
    assert (rows[0].seq, rows[0].query) == (0, "Q0")
