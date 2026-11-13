"""Unit tests for bench/analyst/an_common.py."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


# Load a module by path under an an_ name, clear of other bench tests.
def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, HERE.parent / file_name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mod = _load("an_common", "an_common.py")


def test_paths_layout():
    p = mod.paths(Path("R"))
    assert p.dataset_dir("tpch") == Path("R/datasets/tpch")
    assert p.run_file("test", "qwen36-35b") == Path("R/runs/test_qwen36-35b.jsonl")
    assert (p.questions, p.freeze, p.cells) == (Path("R/questions.jsonl"), Path("R/freeze.json"), Path("R/cells.csv"))
    with pytest.raises(ValueError):
        mod.check_dataset("tpcds")


def test_refuse_ai_root():
    with pytest.raises(ValueError):
        mod.refuse_ai_root(mod.AI_ROOT / "x.csv")
    assert mod.refuse_ai_root(mod.RESULTS / "x") == mod.RESULTS / "x"


def test_writers_refuse_ai_root(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "AI_ROOT", tmp_path / "ai")
    f = tmp_path / "ai" / f"x_{os.getpid()}.jsonl"
    for write in (lambda: mod.append_jsonl(f, {}), lambda: mod.write_text_atomic(f, ""), lambda: mod.repair_jsonl(f)):
        with pytest.raises(ValueError):
            write()
    assert not (tmp_path / "ai").exists()


def test_jsonl_roundtrip_and_partial_last_line(tmp_path):
    f = tmp_path / f"r_{os.getpid()}.jsonl"
    assert mod.read_jsonl(f) == [] and mod.repair_jsonl(f) == 0
    mod.append_jsonl(f, {"a": 1}); mod.append_jsonl(f, {"a": 2})
    f.write_text(f.read_text() + '{"a": 3')          # interrupted write
    assert [r["a"] for r in mod.read_jsonl(f)] == [1, 2]
    assert mod.repair_jsonl(f) == len('{"a": 3') and f.read_text().endswith("\n")


def test_read_jsonl_bad_middle_line_raises(tmp_path):
    f = tmp_path / f"bad_{os.getpid()}.jsonl"
    f.write_text('x\n{"a": 1}\n')
    with pytest.raises(ValueError, match=r"\.jsonl:1: "):
        mod.read_jsonl(f)


def test_read_jsonl_skips_blank_lines(tmp_path):
    f = tmp_path / f"blank_{os.getpid()}.jsonl"
    f.write_text('{"a": 1}\n\n{"a": 2}\n\n')
    assert mod.read_jsonl(f) == [{"a": 1}, {"a": 2}]


def test_repair_jsonl_clean_file_unchanged(tmp_path):
    f = tmp_path / f"clean_{os.getpid()}.jsonl"
    f.write_text('{"a": 1}\n')
    assert mod.repair_jsonl(f) == 0 and f.read_text() == '{"a": 1}\n'


@pytest.mark.parametrize("tail", ['{"a": 2', '{"a": 2}'])   # partial line, complete line with no \n
def test_append_after_unfinished_line_raises(tmp_path, tail):
    f = tmp_path / f"glue_{os.getpid()}.jsonl"
    f.write_text('{"a": 1}\n' + tail)
    with pytest.raises(ValueError, match="repair_jsonl"):
        mod.append_jsonl(f, {"a": 3})
    assert f.read_text() == '{"a": 1}\n' + tail


@pytest.mark.parametrize("tail, removed, kept", [('{"a": 2', 7, [1]), ('{"a": 2}', 0, [1, 2])])
def test_repair_then_append_keeps_complete_records(tmp_path, tail, removed, kept):
    f = tmp_path / f"fix_{os.getpid()}.jsonl"
    f.write_text('{"a": 1}\n' + tail)
    assert [r["a"] for r in mod.read_jsonl(f)] == kept
    assert mod.repair_jsonl(f) == removed
    mod.append_jsonl(f, {"a": 3})
    assert [r["a"] for r in mod.read_jsonl(f)] == kept + [3]


def test_repair_jsonl_without_any_newline(tmp_path):
    f = tmp_path / f"lone_{os.getpid()}.jsonl"
    f.write_text('{"a": 1')
    assert mod.repair_jsonl(f) == 7 and f.read_text() == ""
    mod.append_jsonl(f, {"a": 2})                         # an empty file takes appends
    assert f.read_text() == '{"a": 2}\n'
    f.write_text('{"a": 1}')
    assert mod.repair_jsonl(f) == 0 and f.read_text() == '{"a": 1}\n'


def test_sha1_file_matches_text(tmp_path):
    f = tmp_path / f"s_{os.getpid()}.txt"
    f.write_bytes("héllo\n".encode("utf-8"))
    assert mod.sha1_file(f) == mod.sha1_text("héllo\n") == hashlib.sha1("héllo\n".encode("utf-8")).hexdigest()


def test_write_text_atomic(tmp_path):
    f = tmp_path / f"w_{os.getpid()}" / "out.md"
    mod.write_text_atomic(f, "one")
    mod.write_text_atomic(f, "two")
    assert f.read_text() == "two" and os.listdir(f.parent) == ["out.md"]


def test_load_sibling_registers_module():
    other = mod.load_sibling("an_common_again", "an_common.py")
    try:
        assert sys.modules["an_common_again"] is other and other.SEED == 1
    finally:
        sys.modules.pop("an_common_again", None)


def test_to_sqlite_drops_the_date_keyword():
    sql = "SELECT o_orderdate FROM orders WHERE o_orderdate >= DATE '1995-01-01' AND o_orderdate < date'1996-01-01'"
    want = "SELECT o_orderdate FROM orders WHERE o_orderdate >= '1995-01-01' AND o_orderdate < '1996-01-01'"
    assert mod.to_sqlite(sql) == want
    assert mod.to_sqlite("WHERE d BETWEEN Date\n'1995-01-01' AND DATE  '1995-12-31'") == \
        "WHERE d BETWEEN '1995-01-01' AND '1995-12-31'"


@pytest.mark.parametrize("sql", [
    "SELECT l_shipdate, o_orderdate FROM lineitem, orders",
    "SELECT COUNT(*) FROM part WHERE p_comment LIKE '%date'",   # DATE inside a string stays
    "SELECT * FROM t WHERE s = 'a DATE ''x'''",
    "SELECT update_date FROM t WHERE update_date = '1995-01-01'",
    "SELECT s FROM t WHERE s = 'a -- DATE ''x'''",                     # -- inside a string isn't a comment
    "SELECT COUNT(*) FROM t -- since DATE '1995-01-01'\nWHERE k > 1",  # DATE inside a comment stays
])
def test_to_sqlite_leaves_everything_else_alone(sql):
    assert mod.to_sqlite(sql) == sql


def test_to_sqlite_skips_comments():
    sql = "SELECT COUNT(*) AS n FROM orders -- customers' orders\nWHERE o_orderdate >= DATE '1995-01-01'"
    want = "SELECT COUNT(*) AS n FROM orders -- customers' orders\nWHERE o_orderdate >= '1995-01-01'"
    assert mod.to_sqlite(sql) == want
    assert mod.to_sqlite("SELECT 1 -- it's\n, DATE '1995-01-01' -- don't") == \
        "SELECT 1 -- it's\n, '1995-01-01' -- don't"


# A small SQLite file: four names, one of them stored as bytes.
def _names_db(tmp_path: Path) -> Path:
    db = tmp_path / f"names_{os.getpid()}.sqlite"
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, name)")
        conn.execute("INSERT INTO t VALUES (1, 'Apple'), (2, 'apple'), (3, 'banana'), (4, X'00ff')")
        conn.commit()
    return db


def test_sqlite_query_shape_and_bytes_as_hex(tmp_path):
    db = _names_db(tmp_path)
    assert mod.sqlite_query(db, "SELECT id, name AS n FROM t ORDER BY id") == {
        "ok": True, "columns": ["id", "n"], "rows": [[1, "Apple"], [2, "apple"], [3, "banana"], [4, "00ff"]]}
    assert mod.sqlite_query(db, "SELEC 1") == {"ok": False, "kind": "sql", "error": 'near "SELEC": syntax error'}


def test_sqlite_query_row_cap(tmp_path):
    db = _names_db(tmp_path)
    sql = "SELECT id FROM t ORDER BY id"
    assert mod.sqlite_query(db, sql, row_cap=2) == {"ok": True, "columns": ["id"], "rows": [[1], [2]], "truncated": True}
    assert mod.sqlite_query(db, sql, row_cap=4) == {"ok": True, "columns": ["id"], "rows": [[1], [2], [3], [4]]}
    assert mod.sqlite_query(db, sql) == mod.sqlite_query(db, sql, row_cap=4)


def test_sqlite_query_is_read_only(tmp_path):
    db = _names_db(tmp_path)
    result = mod.sqlite_query(db, "INSERT INTO t VALUES (5, 'cherry')")
    assert result["ok"] is False and result["kind"] == "sql" and "readonly" in result["error"]
    assert mod.sqlite_query(db, "SELECT COUNT(*) FROM t")["rows"] == [[4]]


def test_sqlite_query_like_is_case_sensitive(tmp_path):
    db = _names_db(tmp_path)
    assert mod.sqlite_query(db, "SELECT name FROM t WHERE name LIKE 'a%'")["rows"] == [["apple"]]


def test_sqlite_query_timeout(tmp_path, monkeypatch):
    db = _names_db(tmp_path)
    loop = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT MAX(x) FROM c"
    assert mod.sqlite_query(db, loop, timeout_s=0.05) == {"ok": False, "kind": "timeout", "error": "timeout"}
    monkeypatch.setattr(mod, "TIMEOUT_S", 0.05)  # the default is read at call time
    assert mod.sqlite_query(db, loop) == {"ok": False, "kind": "timeout", "error": "timeout"}
