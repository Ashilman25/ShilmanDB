"""Tests for bench/analyst/probe_subset.py."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
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


mod = _load("an_probe_subset", "probe_subset.py")

NAMES = {
    "count_star", "sum_expr", "avg", "min_max", "group_by", "having", "order_by_column", "order_by_alias",
    "order_by_aggregate", "order_by_position", "limit", "offset", "table_alias",
    "join_on", "implicit_join", "three_way_join", "self_join", "left_join",
    "between", "in_list", "not_in_list", "like", "not_like", "case_when", "date_literal", "date_compare_string",
    "arithmetic", "integer_division",
    "distinct", "count_distinct", "subquery_where", "subquery_from", "with_cte", "union", "is_null", "coalesce",
    "extract", "substring", "upper", "cast", "select_no_from",
}

# Shapes over table t with the bridge's canned answer; None means it fails to parse.
SMALL = [
    ("same", "SELECT k FROM t ORDER BY k", (["k"], [[1], [2], [3]])),
    ("wrong_order", "SELECT k FROM t ORDER BY k DESC", (["k"], [[1], [2], [3]])),
    ("any_order", "SELECT k FROM t", (["k"], [[3], [1], [2]])),
    ("fails", "SELECT DISTINCT k FROM t", None),
    ("sqlite_raises", "SELECT EXTRACT(YEAR FROM d) AS y FROM t", (["y"], [[1995], [1996], [1995]])),
    ("date", "SELECT COUNT(*) AS n FROM t WHERE d >= DATE '1995-06-01'", (["n"], [[2]])),
    ("like", "SELECT COUNT(*) AS n FROM t WHERE s LIKE 'a%'", (["n"], [[1]])),
]
T_ROWS = [(1, "abc", "1995-03-01"), (2, "ABC", "1996-01-01"), (3, "xyz", "1995-12-31")]


class FakeBridge:
    """Answers SQL from a dict of SQL -> (columns, rows); other SQL fails to parse."""

    def __init__(self, answers: dict):
        self.answers = answers
        self.sent: list[str] = []

    def __enter__(self) -> FakeBridge:
        return self

    def __exit__(self, *exc) -> None:
        pass

    # Reply the way analyst_bridge does.
    def query(self, sql: str) -> dict:
        self.sent.append(sql)
        rid = str(len(self.sent))
        answer = self.answers.get(sql)
        if answer is None:
            return {"id": rid, "ok": False, "error": {"kind": "parse", "message": "unexpected token"}, "ms": 0}

        cols, rows = answer
        return {"id": rid, "ok": True, "columns": [{"name": c, "type": "INTEGER"} for c in cols], "rows": rows,
                "row_count": len(rows), "truncated": False, "profile": {}, "ms": 0}


# Table t with T_ROWS, in a fresh SQLite connection (memory unless a path is given).
def _sqlite_t(target: str = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(target)
    conn.execute("CREATE TABLE t (k INTEGER, s TEXT, d TEXT)")
    conn.executemany("INSERT INTO t VALUES (?, ?, ?)", T_ROWS)
    conn.commit()
    return conn


@pytest.fixture
def small(monkeypatch) -> FakeBridge:
    monkeypatch.setattr(mod, "SHAPES", [(n, s) for n, s, _ in SMALL])
    return FakeBridge({s: a for _, s, a in SMALL if a is not None})


# A probe where every real shape ran and agreed with SQLite.
def _all_ok() -> dict:
    return {"shapes": {n: {"sql": s, "ok": True, "error": None, "same_as_sqlite": True} for n, s in mod.SHAPES}}


# The rules text under each heading: (works, doesn't work).
def _sections(rules: str) -> tuple[str, str]:
    head, broken = rules.split("Doesn't work")
    return head.split("## Works")[1], broken


# ── shapes ────────────────────────────────────────────────────────────

def test_shape_names_unique_and_complete():
    assert len({n for n, _ in mod.SHAPES}) == len(mod.SHAPES) == 41
    assert {n for n, _ in mod.SHAPES} == NAMES


def test_forced_unsupported_values():
    assert mod.FORCED_UNSUPPORTED == {
        "left_join": "LEFT JOIN runs as an inner join here, so don't use it",
        "is_null": "there are no NULLs",
        "self_join": "a table can appear only once in a query",
    }


# ── render_rules ──────────────────────────────────────────────────────

def test_forced_unsupported_listed_even_if_ok():
    p = {"shapes": {n: {"sql": s, "ok": True, "error": None} for n, s in mod.SHAPES}}
    rules = mod.render_rules(p)
    works, broken = _sections(rules)
    assert "LEFT JOIN runs as an inner join" in broken
    for name, reason in mod.FORCED_UNSUPPORTED.items():
        assert f"- {name}: {reason}" in broken
        assert f"- {name}:" not in works


def test_failed_shape_goes_to_doesnt_work():
    p = _all_ok()
    p["shapes"]["distinct"].update(ok=False, error={"kind": "run", "message": "unknown column"}, same_as_sqlite=None)
    works, broken = _sections(mod.render_rules(p))
    assert "- distinct: not supported by this engine" in broken
    assert "- distinct:" not in works


def test_different_answer_goes_to_doesnt_work():
    p = _all_ok()
    p["shapes"]["order_by_position"]["same_as_sqlite"] = False
    works, broken = _sections(mod.render_rules(p))
    assert "- order_by_position: runs, but gives a different answer than SQLite" in broken
    assert "- order_by_position:" not in works


def test_forced_reason_wins_over_failure_and_difference():
    p = _all_ok()
    p["shapes"]["left_join"].update(ok=False, same_as_sqlite=None)
    p["shapes"]["self_join"]["same_as_sqlite"] = False
    _, broken = _sections(mod.render_rules(p))
    assert f"- left_join: {mod.FORCED_UNSUPPORTED['left_join']}" in broken
    assert f"- self_join: {mod.FORCED_UNSUPPORTED['self_join']}" in broken
    assert "not supported by this engine" not in broken and "different answer" not in broken


def test_sqlite_error_still_counts_as_working():
    p = _all_ok()
    p["shapes"]["extract"].update(same_as_sqlite=None, sqlite_error="no such column: YEAR")
    works, broken = _sections(mod.render_rules(p))
    assert f"- extract: `{p['shapes']['extract']['sql']}`" in works
    assert "extract" not in broken


def test_works_shows_sql_after_the_fixed_opening():
    p = _all_ok()
    rules = mod.render_rules(p)
    lines = rules.splitlines()
    assert mod.OPENING in lines[:4]
    assert rules.index(mod.OPENING) < rules.index("## Works") < rules.index("## Doesn't work")
    works, _ = _sections(rules)
    for name, sql in mod.SHAPES:
        if name not in mod.FORCED_UNSUPPORTED:
            assert f"- {name}: `{sql}`" in works


def test_works_has_an_order_by_on_an_aggregate_alias():
    sql = dict(mod.SHAPES)["order_by_alias"]
    aliases = re.findall(r"\b(?:SUM|COUNT|AVG|MIN|MAX)\([^)]*\)\s+AS\s+(\w+)", sql, re.IGNORECASE)
    order_by = re.search(r"\bORDER\s+BY\s+(\w+)", sql, re.IGNORECASE)
    assert order_by and order_by.group(1) in aliases
    works, _ = _sections(mod.render_rules(_all_ok()))
    assert f"- order_by_alias: `{sql}`" in works


def test_render_rules_is_pure():
    p = _all_ok()
    p["shapes"]["union"]["ok"] = False
    before = copy.deepcopy(p)
    assert mod.render_rules(p) == mod.render_rules(p)
    assert p == before


# ── probe ─────────────────────────────────────────────────────────────

def test_probe_compares_with_sqlite(small):
    with closing(_sqlite_t()) as conn:
        out = mod.probe(small, conn)

    s = out["shapes"]
    assert list(s) == [n for n, _, _ in SMALL]
    assert small.sent == [sql for _, sql, _ in SMALL]  # the bridge gets the SQL as written
    assert {k: s["same"][k] for k in ("sql", "ok", "error", "same_as_sqlite")} == {
        "sql": "SELECT k FROM t ORDER BY k", "ok": True, "error": None, "same_as_sqlite": True}
    assert s["wrong_order"]["same_as_sqlite"] is False   # ORDER BY means order counts
    assert s["any_order"]["same_as_sqlite"] is True      # no ORDER BY, any order
    assert s["date"]["same_as_sqlite"] is True           # DATE '...' turned into '...' for SQLite
    assert s["like"]["same_as_sqlite"] is True           # LIKE is case-sensitive, as in the engine
    json.dumps(out)


def test_probe_records_a_failed_shape(small):
    with closing(_sqlite_t()) as conn:
        entry = mod.probe(small, conn)["shapes"]["fails"]
    assert entry["ok"] is False and entry["same_as_sqlite"] is None
    assert entry["error"] == {"kind": "parse", "message": "unexpected token"}


def test_probe_records_sqlite_error_but_shape_still_works(small):
    with closing(_sqlite_t()) as conn:
        out = mod.probe(small, conn)
    entry = out["shapes"]["sqlite_raises"]
    assert entry["ok"] is True and entry["same_as_sqlite"] is None and entry["sqlite_error"]
    works, broken = _sections(mod.render_rules(out))
    assert "- sqlite_raises:" in works
    assert "- wrong_order: runs, but gives a different answer than SQLite" in broken
    assert "- fails: not supported by this engine" in broken


def test_probe_keeps_first_rows_of_both_sides(small):
    with closing(_sqlite_t()) as conn:
        entry = mod.probe(small, conn)["shapes"]["wrong_order"]
    assert entry["row_count"] == entry["sqlite_row_count"] == 3
    assert entry["rows"][0] == [1] and entry["sqlite_rows"][0] == [3]


def test_probe_without_sqlite(small):
    out = mod.probe(small)
    assert all(e["same_as_sqlite"] is None for e in out["shapes"].values())
    assert not any("sqlite_error" in e for e in out["shapes"].values())


# ── CLI ───────────────────────────────────────────────────────────────

# A results folder holding a tpch tables.tsv and a small tpch.sqlite.
def _results(root: Path) -> Path:
    ds = root / "datasets" / "tpch"
    ds.mkdir(parents=True)
    (ds / "tables.tsv").write_text("t\tt.tbl\tk:BIGINT\n", encoding="utf-8")
    _sqlite_t(str(ds / "tpch.sqlite")).close()
    return root


def test_main_writes_probe_and_rules(tmp_path, monkeypatch, small, capsys):
    res = _results(tmp_path / f"res_{os.getpid()}")
    started = []
    monkeypatch.setattr(mod.bc, "Bridge", lambda tsv: started.append(Path(tsv)) or small)

    assert mod.main(["--results", str(res)]) == 0
    p = mod.an.paths(res)
    assert started == [p.dataset_dir("tpch") / "tables.tsv"]
    probe = json.loads(p.probe.read_text(encoding="utf-8"))
    assert probe["shapes"]["like"]["same_as_sqlite"] is True
    assert p.rules.read_text(encoding="utf-8") == mod.render_rules(probe)
    assert "5 work, 2 don't" in capsys.readouterr().out


def test_main_needs_the_dataset(tmp_path, capsys):
    res = tmp_path / f"empty_{os.getpid()}"
    assert mod.main(["--results", str(res)]) == 1
    assert "prepare_datasets.py" in capsys.readouterr().err
    assert not res.exists()
