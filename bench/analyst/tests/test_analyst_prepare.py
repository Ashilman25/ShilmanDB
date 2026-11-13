"""Tests for bench/analyst/prepare_datasets.py."""
from __future__ import annotations

import csv
import importlib.util
import json
import re
import sqlite3
import sys
import time
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


mod = _load("an_prepare_datasets", "prepare_datasets.py")

REVIEW_HEADER = ["review_id", "rating", "title", "text", "parent_asin", "timestamp",
                 "helpful_vote", "verified_purchase", "user_id", "text_key"]
PRODUCT_HEADER = ["parent_asin", "title", "store", "price", "main_category", "categories",
                  "average_rating", "rating_number"]
LONG_TEXT = "a|" + "€" * 3332 + "zz"   # 10,000 bytes; € is 3 bytes
NEW_YEAR = 1577844000000              # 2020-01-01 02:00 UTC, still 2019 in New York
MAY = 1590601487963                   # 2020-05-27 17:44 UTC


# Write a CSV with a header row, every field as given.
def _csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header, *rows])


# A tiny ai_queries folder: two products and two reviews of the all-digit one.
def _ai_root(root: Path) -> Path:
    _csv(root / "data/products.csv", PRODUCT_HEADER, [
        ["B001", "Case | Clip\nRed", "", "", "", "[]", "3.7", "103"],
        ["0123456789", "Mount", "Inspeed", "12.99", "Cell Phones & Accessories",
         '["Cell Phones & Accessories", "Cases, Holsters & Sleeves", "Basic Cases"]', "4.0", "7"],
    ])
    _csv(root / "data/reviews.csv", REVIEW_HEADER, [
        ["16604130", "2", "N/A", LONG_TEXT, "0123456789", str(NEW_YEAR), "3", "False", "AEPOD", "k1"],
        ["9826524", "5", "", "", "0123456789", str(MAY), "0", "True", "AHY2", "k2"],
    ])
    return root


# Rows of one table as dicts keyed by column name.
def _dicts(table) -> list[dict]:
    name, columns, rows = table
    return [dict(zip([c for c, _ in columns], row)) for row in rows]


# One source row per TPC-H table, values made up by type.
def _tpch_dir(root: Path) -> Path:
    sample = {"BIGINT": "7", "INTEGER": "3", "DECIMAL": "0.05", "VARCHAR": "abc", "DATE": "1995-03-15"}
    dates = {"l_shipdate": "1996-02-12", "l_commitdate": "1996-03-01", "l_receiptdate": "1996-12-31"}
    root.mkdir(parents=True, exist_ok=True)
    for name, columns in mod.TPCH_COLUMNS.items():
        row = [dates.get(c, sample[t]) for c, t in columns]
        copies = 2 if name == "orders" else 1
        (root / f"{name}.tbl").write_text(("|".join(row) + "|\n") * copies, encoding="utf-8")
    return root


# Run in New York time, so a local-time date would land in 2019.
@pytest.fixture
def new_york(monkeypatch):
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_clean_text_rules():
    assert mod.clean_text("a|b\tc\r\nd") == "a b c  d"
    s = mod.clean_text("é" * 3000)                    # 6,000 bytes
    assert len(s.encode()) <= 4000 and s == "é" * 2000
    assert mod.clean_text(None) == ""
    assert mod.clean_text("ab" + "€" * 1333) == "ab" + "€" * 1332   # a cut € is dropped whole


def test_year_month_columns():
    cols, rows = mod.add_year_month([("o_orderdate", "DATE")], [["1995-03-15"]])
    assert cols == [("o_orderdate", "DATE"), ("o_orderyear", "INTEGER"), ("o_ordermonth", "INTEGER")]
    assert rows == [["1995-03-15", 1995, 199503]]


def test_year_month_for_every_date_column_in_order():
    cols, rows = mod.add_year_month(
        [("l_shipdate", "DATE"), ("l_note", "VARCHAR"), ("l_when", "DATE"), ("l_commitdate", "DATE")],
        [["1996-02-12", "x", "1990-01-01", "1996-12-31"]])
    assert [c for c, _ in cols][4:] == ["l_shipyear", "l_shipmonth", "l_commityear", "l_commitmonth"]
    assert rows == [["1996-02-12", "x", "1990-01-01", "1996-12-31", 1996, 199602, 1996, 199612]]


def test_tpch_columns_match_header():   # every TableDescs() column name in tpch_workload.hpp appears in TPCH_COLUMNS
    text = (mod.REPO_ROOT / "bench/tpch/tpch_workload.hpp").read_text()
    for t, cols in mod.TPCH_COLUMNS.items():
        for name, _ in cols: assert f'"{name}"' in text
    assert set(mod.TPCH_COLUMNS) == {"region", "nation", "part", "supplier", "customer", "orders", "lineitem"}

    # Same names, order and types as each Schema() function in the header, in TableDescs() order.
    schemas = {table.lower(): re.findall(r'\{"(\w+)", TypeId::(\w+)\}', body)
               for table, body in re.findall(r"Schema (\w+)Schema\(\) \{(.*?)\n\}", text, re.S)}
    assert schemas == mod.TPCH_COLUMNS
    assert list(mod.TPCH_COLUMNS) == ["region", "nation", "part", "supplier", "customer", "orders", "lineitem"]


def test_build_tpch_adds_date_parts(tmp_path):
    tables = mod.build_tpch(_tpch_dir(tmp_path / "tpch"))
    assert [t.name for t in tables] == list(mod.TPCH_COLUMNS)
    by_name = {t.name: t for t in tables}

    region = by_name["region"]
    assert region.columns == mod.TPCH_COLUMNS["region"] and region.rows == [["7", "abc", "abc"]]

    orders = by_name["orders"]
    assert orders.columns == mod.TPCH_COLUMNS["orders"] + [("o_orderyear", "INTEGER"), ("o_ordermonth", "INTEGER")]
    assert len(orders.rows) == 2 and orders.rows[0][-2:] == [1995, 199503]

    lineitem = _dicts(by_name["lineitem"])[0]
    assert [lineitem[c] for c in ("l_shipyear", "l_shipmonth", "l_commityear", "l_commitmonth",
                                  "l_receiptyear", "l_receiptmonth")] == [1996, 199602, 1996, 199603, 1996, 199612]


def test_build_tpch_rejects_a_short_line(tmp_path):
    root = _tpch_dir(tmp_path / "tpch")
    (root / "nation.tbl").write_text("0|ALGERIA|0|\n", encoding="utf-8")   # 3 of 4 fields
    with pytest.raises(ValueError, match="nation.tbl:1"):
        mod.build_tpch(root)


def test_reviews_mapping(tmp_path, new_york):
    reviews, products = mod.build_reviews(_ai_root(tmp_path / "ai"))

    assert products.name == "products" and products.columns == [
        ("p_asin", "VARCHAR"), ("p_title", "VARCHAR"), ("p_store", "VARCHAR"), ("p_main_category", "VARCHAR"),
        ("p_category", "VARCHAR"), ("p_price", "DECIMAL"), ("p_has_price", "INTEGER"),
        ("p_avg_rating", "DECIMAL"), ("p_rating_count", "INTEGER")]
    p, q = _dicts(products)
    assert p["p_price"] == 0.0 and p["p_has_price"] == 0 and p["p_store"] == "Unknown" and p["p_category"] == "Unknown"
    assert p["p_title"] == "Case   Clip Red" and p["p_main_category"] == "Unknown"
    assert (p["p_avg_rating"], p["p_rating_count"]) == (3.7, 103)
    assert q == {"p_asin": "0123456789", "p_title": "Mount", "p_store": "Inspeed",
                 "p_main_category": "Cell Phones & Accessories", "p_category": "Basic Cases",
                 "p_price": 12.99, "p_has_price": 1, "p_avg_rating": 4.0, "p_rating_count": 7}

    assert reviews.name == "reviews" and reviews.columns == [
        ("r_review_id", "BIGINT"), ("r_asin", "VARCHAR"), ("r_rating", "INTEGER"), ("r_title", "VARCHAR"),
        ("r_text", "VARCHAR"), ("r_date", "DATE"), ("r_helpful_votes", "INTEGER"), ("r_verified", "INTEGER"),
        ("r_user_id", "VARCHAR"), ("r_year", "INTEGER"), ("r_month", "INTEGER")]
    r, s = _dicts(reviews)
    assert r["r_verified"] == 0 and len(r["r_text"].encode()) <= 4000 and r["r_month"] == 202001
    assert r["r_text"] == "a " + "€" * 1332 and r["r_title"] == "N/A"   # only an empty field is missing
    assert (r["r_review_id"], r["r_asin"], r["r_rating"], r["r_date"], r["r_year"]) == (
        16604130, "0123456789", 2, "2020-01-01", 2020)
    assert (r["r_helpful_votes"], r["r_user_id"]) == (3, "AEPOD")
    assert s == {"r_review_id": 9826524, "r_asin": "0123456789", "r_rating": 5, "r_title": "", "r_text": "",
                 "r_date": "2020-05-27", "r_helpful_votes": 0, "r_verified": 1, "r_user_id": "AHY2",
                 "r_year": 2020, "r_month": 202005}
    assert all(type(v) in (int, float, str) for t in (reviews, products) for row in t.rows for v in row)


@pytest.mark.parametrize("file, header, row, match", [
    ("reviews.csv", REVIEW_HEADER, ["1", "5", "t", "x", "", str(MAY), "0", "True", "U", "k"], "parent_asin"),
    ("reviews.csv", REVIEW_HEADER, ["1", "4.5", "t", "x", "B001", str(MAY), "0", "True", "U", "k"], "rating"),
    ("reviews.csv", REVIEW_HEADER, ["1", "5", "t", "x", "B001", str(MAY), "0", "maybe", "U", "k"], "verified"),
    ("products.csv", PRODUCT_HEADER, ["B001", "t", "s", "1.0", "m", "[]", "", "3"], "average_rating"),
])
def test_reviews_reject_bad_source_values(tmp_path, file, header, row, match):
    root = _ai_root(tmp_path / "ai")
    _csv(root / "data" / file, header, [row])
    with pytest.raises(ValueError, match=match):
        mod.build_reviews(root)


def test_lines_end_with_delimiter_and_sqlite_counts(tmp_path):
    out = tmp_path / "ds"
    t = mod.Table("t", [("t_id", "BIGINT"), ("t_name", "VARCHAR"), ("t_price", "DECIMAL"), ("t_day", "DATE"),
                        ("t_n", "INTEGER"), ("t_note", "VARCHAR")],
                  [[1, "a b", 0.5, "1995-03-15", 3, ""], [2, "é", 12.99, "1996-01-02", -1, "x"]])
    u = mod.Table("u", [("u_id", "BIGINT"), ("u_amt", "DECIMAL")], [["5", "3"]])   # text, as read from a .tbl
    mod.write_dataset("tpch", [t, u], out)
    mod.write_dataset("tpch", [t, u], out)   # a second run replaces, never appends

    assert sorted(p.name for p in out.iterdir()) == ["manifest.json", "t.tbl", "tables.tsv", "tpch.sqlite", "u.tbl"]
    lines = (out / "t.tbl").read_bytes().split(b"\n")
    assert lines == [b"1|a b|0.5|1995-03-15|3||", "2|é|12.99|1996-01-02|-1|x|".encode(), b""]
    assert all(line.endswith(b"|") for line in lines[:-1])
    assert (out / "u.tbl").read_text() == "5|3|\n"

    assert (out / "tables.tsv").read_text() == (
        "t\tt.tbl\tt_id:BIGINT,t_name:VARCHAR,t_price:DECIMAL,t_day:DATE,t_n:INTEGER,t_note:VARCHAR\n"
        "u\tu.tbl\tu_id:BIGINT,u_amt:DECIMAL\n")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest == {"dataset": "tpch", "tables": [
        {"name": "t", "file": "t.tbl", "rows": 2, "columns": [{"name": c, "type": k} for c, k in t.columns]},
        {"name": "u", "file": "u.tbl", "rows": 1, "columns": [{"name": c, "type": k} for c, k in u.columns]}]}

    with closing(sqlite3.connect(out / "tpch.sqlite")) as conn:
        assert [conn.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in ("t", "u")] == [2, 1]
        assert [r[2] for r in conn.execute("PRAGMA table_info(t)")] == ["INTEGER", "TEXT", "REAL", "TEXT", "INTEGER", "TEXT"]
        typed = "SELECT typeof(t_id), typeof(t_name), typeof(t_price), typeof(t_day), typeof(t_n), typeof(t_note) FROM t"
        assert conn.execute(typed).fetchall() == [("integer", "text", "real", "text", "integer", "text")] * 2
        assert conn.execute("SELECT * FROM t ORDER BY t_id").fetchall() == [tuple(r) for r in t.rows]
        assert conn.execute("SELECT u_id, typeof(u_amt), u_amt FROM u").fetchall() == [(5, "real", 3.0)]


@pytest.mark.parametrize("bad", ["a|b", "a\nb", "a\rb"])
def test_write_rejects_a_value_that_breaks_the_line(tmp_path, bad):
    out = tmp_path / "ds"
    with pytest.raises(ValueError, match="row 2"):
        mod.write_dataset("tpch", [mod.Table("t", [("t_id", "BIGINT"), ("t_s", "VARCHAR")], [[1, "ok"], [2, bad]])], out)
    assert (sorted(p.name for p in out.iterdir()) if out.exists() else []) == []   # no temp file left behind


def test_write_rejects_a_short_row(tmp_path):
    with pytest.raises(ValueError, match="row 1"):
        mod.write_dataset("tpch", [mod.Table("t", [("t_a", "VARCHAR"), ("t_b", "VARCHAR")], [["a|b"]])], tmp_path / "ds")


def test_duplicate_column_names_rejected(tmp_path):
    out = tmp_path / "ds"
    a = mod.Table("a", [("id", "BIGINT"), ("title", "VARCHAR")], [[1, "x"]])
    b = mod.Table("b", [("b_id", "BIGINT"), ("title", "VARCHAR")], [[1, "y"]])
    with pytest.raises(ValueError, match="title"):
        mod.write_dataset("reviews", [a, b], out)
    assert not out.exists()


@pytest.mark.parametrize("table", [
    ("t", [("t_a", "TEXT")], [["x"]]),          # not an engine type
    ("t", [("t,a", "VARCHAR")], [["x"]]),       # would break tables.tsv
    ("t-1", [("t_a", "VARCHAR")], [["x"]]),     # not a plain name
])
def test_write_rejects_bad_names_and_types(tmp_path, table):
    with pytest.raises(ValueError):
        mod.write_dataset("tpch", [mod.Table(*table)], tmp_path / "ds")
    assert not (tmp_path / "ds").exists()


def test_refuses_ai_root_output(tmp_path, monkeypatch):
    ai = _ai_root(tmp_path / "ai")
    monkeypatch.setattr(mod.an, "AI_ROOT", ai)
    with pytest.raises(ValueError, match="won't write"):
        mod.write_dataset("reviews", mod.build_reviews(ai), ai / "out")
    with pytest.raises(ValueError, match="won't write"):   # refused before any dataset is built
        mod.main(["--results", str(ai / "res"), "--tpch-dir", str(tmp_path / "missing"), "--ai", str(ai)])
    assert sorted(p.name for p in ai.iterdir()) == ["data"]


def test_main_writes_only_the_chosen_datasets(tmp_path, capsys):
    res = tmp_path / "res"
    argv = ["--datasets", "reviews", "--ai", str(_ai_root(tmp_path / "ai")), "--results", str(res),
            "--tpch-dir", str(tmp_path / "missing")]
    assert mod.main(argv) == 0
    manifest = json.loads((res / "datasets/reviews/manifest.json").read_text())
    assert [(t["name"], t["rows"]) for t in manifest["tables"]] == [("reviews", 2), ("products", 2)]
    assert not (res / "datasets/tpch").exists()
    assert "reviews" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        mod.main(["--datasets", "tpcds", "--results", str(res)])


bridge_client = sys.modules.get("an_bridge_client") or _load("an_bridge_client", "bridge_client.py")


@pytest.mark.skipif(not bridge_client.BRIDGE.exists(), reason="analyst_bridge not built")
def test_bridge_loads_what_sqlite_holds(tmp_path):
    out = tmp_path / "ds"
    columns, rows = mod.add_year_month(
        [("t_id", "BIGINT"), ("t_price", "DECIMAL"), ("t_date", "DATE"), ("t_note", "VARCHAR")],
        [[1, 0.5, "1995-03-15", ""], [2, 3.0, "1996-12-31", mod.clean_text("x|y")]])
    mod.write_dataset("tpch", [mod.Table("t", columns, rows)], out)

    sql = "SELECT t_id, t_price, t_date, t_note, t_year, t_month FROM t ORDER BY t_id"
    with bridge_client.Bridge(out / "tables.tsv") as b:
        assert b.ready["rows"] == {t["name"]: t["rows"] for t in json.loads((out / "manifest.json").read_text())["tables"]}
        reply = b.query(sql)
    with closing(sqlite3.connect(out / "tpch.sqlite")) as conn:
        assert reply["ok"] and reply["rows"] == [list(r) for r in conn.execute(sql)]
    assert reply["rows"] == [[1, 0.5, "1995-03-15", "", 1995, 199503], [2, 3, "1996-12-31", "x y", 1996, 199612]]
