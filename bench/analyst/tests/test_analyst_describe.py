"""Tests for bench/analyst/describe_datasets.py and the dataset description files."""
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


mod = _load("an_describe_datasets", "describe_datasets.py")
an = sys.modules["an_common"]

MANIFEST = {"dataset": "tpch", "tables": [
    {"name": "orders", "file": "orders.tbl", "rows": 3, "columns": [
        {"name": "o_orderkey", "type": "BIGINT"}, {"name": "o_status", "type": "VARCHAR"},
        {"name": "o_price", "type": "DECIMAL"}]},
    {"name": "lines", "file": "lines.tbl", "rows": 4, "columns": [
        {"name": "l_orderkey", "type": "BIGINT"}, {"name": "l_no", "type": "INTEGER"},
        {"name": "l_price", "type": "DECIMAL"}, {"name": "l_disc", "type": "DECIMAL"}]},
]}

SPEC = {
    "dataset": "tpch",
    "about": "Orders and their lines.",
    "columns": {"o_orderkey": "Order number.", "o_status": "Order status.", "o_price": "Order total.",
                "l_orderkey": "Order the line belongs to.", "l_no": "Line number.", "l_price": "Line price.",
                "l_disc": "Discount fraction."},
    "primary_keys": {"orders": ["o_orderkey"], "lines": ["l_orderkey", "l_no"]},
    "foreign_keys": [{"table": "lines", "columns": ["l_orderkey"], "ref_table": "orders",
                      "ref_columns": ["o_orderkey"]}],
    "grains": {"order": [["o_orderkey"], ["l_orderkey"]], "status": [["o_status"]]},
    "entity_tables": {"order": "orders", "line": "lines"},
    "missing": ["o_status is 'Unknown' when it wasn't recorded."],
    "glossary": [{"term": "revenue", "words": [r"\brevenue\b"], "readings": [
        {"name": "gross", "default": False, "sql": "l_price", "any": ["l_price"], "all": [], "none": ["l_disc"]},
        {"name": "net", "default": True, "sql": "l_price * (1 - l_disc)", "any": ["l_price"], "all": ["l_disc"],
         "none": []},
    ]}],
}


# A fresh copy of the small spec, safe to change.
def _spec() -> dict:
    return copy.deepcopy(SPEC)


# The real manifest for a dataset, or None if prepare_datasets.py hasn't run.
def _real_manifest(ds: str) -> dict | None:
    path = an.paths().dataset_dir(ds) / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else None


# Column names a piece of SQL names, ignoring quoted strings.
def _columns_in(sql: str, names: set[str]) -> set[str]:
    bare = re.sub(r"'[^']*'", "''", sql)
    return {w.lower() for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", bare)} & names


# A dataset's columns: the manifest's if present, else the spec's (validate keeps them equal).
def _column_names(ds: str, spec: dict) -> set[str]:
    manifest = _real_manifest(ds)
    if manifest is None:
        return set(spec["columns"])
    return {c["name"] for t in manifest["tables"] for c in t["columns"]}


# A one-table manifest holding every column a spec describes.
def _manifest_from_spec(spec: dict) -> dict:
    columns = [{"name": c, "type": "INTEGER"} for c in spec["columns"]]
    tables = [{"name": t, "file": f"{t}.tbl", "rows": 1, "columns": columns if i == 0 else []}
              for i, t in enumerate(spec["primary_keys"])]
    return {"dataset": spec["dataset"], "tables": tables}


# Names of a term's readings that match this SQL.
def _matched(ds: str, term: str, sql: str) -> list[str]:
    spec = mod.load_spec(ds)
    entry = next(g for g in spec["glossary"] if g["term"] == term)
    cols = _columns_in(sql, _column_names(ds, spec))
    return [r["name"] for r in entry["readings"] if mod.reading_matches(r, cols, sql)]


def test_both_specs_validate_against_real_manifests():
    manifests = {ds: _real_manifest(ds) for ds in an.DATASETS}
    if any(m is None for m in manifests.values()):
        pytest.skip("prepare_datasets.py hasn't written the manifests")
    for ds, manifest in manifests.items():
        assert mod.validate(mod.load_spec(ds), manifest) == []


def test_small_spec_is_valid():
    assert mod.validate(_spec(), MANIFEST) == []


def test_validate_catches_gaps():
    spec = _spec()
    del spec["columns"]["o_price"]
    spec["primary_keys"]["lines"] = ["l_orderkey", "l_nope"]
    spec["glossary"][0]["readings"][0]["default"] = True
    problems = mod.validate(spec, MANIFEST)
    assert len(problems) == 3, problems
    for needle in ("o_price", "l_nope", "revenue"):
        assert any(needle in p for p in problems), (needle, problems)


# Change one thing in a spec; validate must report exactly one problem naming it.
def _break_extra_column(s): s["columns"]["o_bogus"] = "Not a column."
def _break_empty_sentence(s): s["columns"]["o_status"] = "  "
def _break_grain(s): s["grains"]["order"].append(["o_nope"])
def _break_reading_column(s): s["glossary"][0]["readings"][0]["none"] = ["l_nope"]
def _break_foreign_key(s): s["foreign_keys"][0]["ref_columns"] = ["o_nope"]
def _break_key_table(s): del s["primary_keys"]["orders"]
def _break_key_unknown_table(s): s["primary_keys"]["widgets"] = ["w_key"]
def _break_entity(s): s["entity_tables"]["widget"] = "widgets"
def _break_no_default(s): s["glossary"][0]["readings"][1]["default"] = False
def _break_pattern(s): s["glossary"][0]["readings"][1]["pattern"] = "l_price * (1"
def _break_word(s): s["glossary"][0]["words"] = ["(revenue"]
def _break_dataset(s): s["dataset"] = "reviews"
def _break_missing_section(s): del s["missing"]
def _break_meaning(s): s["glossary"][0]["meaning"] = 5


@pytest.mark.parametrize("breaker, needle", [
    (_break_extra_column, "o_bogus"), (_break_empty_sentence, "o_status"), (_break_grain, "o_nope"),
    (_break_reading_column, "l_nope"), (_break_foreign_key, "o_nope"), (_break_key_table, "orders"),
    (_break_key_unknown_table, "widgets"), (_break_entity, "widgets"), (_break_no_default, "revenue"),
    (_break_pattern, "pattern"), (_break_word, "(revenue"), (_break_dataset, "reviews"),
    (_break_missing_section, "missing"), (_break_meaning, "meaning"),
])
def test_validate_names_each_problem(breaker, needle):
    spec = _spec()
    breaker(spec)
    problems = mod.validate(spec, MANIFEST)
    assert len(problems) == 1 and needle in problems[0], problems


def test_distinct_values_cap(tmp_path):
    db = tmp_path / f"distinct_{os.getpid()}.sqlite"
    manifest = {"dataset": "tpch", "tables": [{"name": "t", "file": "t.tbl", "rows": 102, "columns": [
        {"name": "small", "type": "VARCHAR"}, {"name": "big", "type": "VARCHAR"},
        {"name": "num", "type": "INTEGER"}, {"name": "day", "type": "DATE"}]}]}
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("CREATE TABLE t (small TEXT, big TEXT, num INTEGER, day TEXT)")
        rows = [(["b", "a", "c"][i % 3], f"v{i % 51:02d}", i % 2, "1995-01-01") for i in range(102)]
        conn.executemany("INSERT INTO t VALUES (?, ?, ?, ?)", rows)
        conn.commit()
        assert mod.distinct_values(conn, manifest) == {"small": ["a", "b", "c"]}
        assert sorted(mod.distinct_values(conn, manifest, cap=51)) == ["big", "small"]


def test_render_lists_values_and_default_first():
    text = mod.render(_spec(), MANIFEST, {"o_status": ["O", "F", "Men's"]})
    sections = [line for line in text.splitlines() if line.startswith("## ")]
    assert sections == ["## About", "## Tables", "## Keys", "## Missing values", "## Glossary"]
    assert "- `o_orderkey` BIGINT — Order number." in text
    assert "  Values: 'F', 'Men''s', 'O'" in text
    assert "lines (`l_orderkey`) → orders (`o_orderkey`)" in text
    assert "o_status is 'Unknown'" in text

    glossary = text[text.index("## Glossary"):]
    assert glossary.index("net") < glossary.index("gross")
    assert "`l_price * (1 - l_disc)`" in glossary and "`l_price`" in glossary
    assert "\\b" not in text and "status" not in glossary   # words and grains stay out


def test_render_puts_meaning_under_the_heading():
    spec = _spec()
    spec["glossary"][0]["meaning"] = "Revenue means money from sales."
    lines = mod.render(spec, MANIFEST, {}).splitlines()
    at = lines.index("### revenue")
    assert lines[at + 1:at + 4] == ["", "Revenue means money from sales.", ""]
    assert lines[at + 4].startswith("- Default, net:")
    assert "meaning" not in mod.render(_spec(), MANIFEST, {})


def test_date_basis_meaning_is_rendered():
    date_basis = next(g for g in mod.load_spec("tpch")["glossary"] if g["term"] == "date basis")
    assert "order date" in date_basis["meaning"] and "in 1995" in date_basis["meaning"]
    text = mod.render(mod.load_spec("tpch"), _manifest_from_spec(mod.load_spec("tpch")), {})
    assert f"### date basis\n\n{date_basis['meaning']}\n\n- Default, order date:" in text


def test_render_says_when_nothing_is_missing():
    spec = _spec()
    spec["missing"] = []
    text = mod.render(spec, MANIFEST, {})
    assert "Values:" not in text
    missing = text[text.index("## Missing values"):text.index("## Glossary")]
    assert "- " in missing


def test_reading_matches_signature_and_pattern():
    r = {"name": "x", "default": True, "sql": "", "any": ["a", "b"], "all": ["c"], "none": ["d"]}
    assert mod.reading_matches(r, {"a", "c"}, "")
    assert not mod.reading_matches(r, {"c"}, "")            # any not met
    assert not mod.reading_matches(r, {"a"}, "")            # all not met
    assert not mod.reading_matches(r, {"a", "c", "d"}, "")  # none met
    assert mod.reading_matches({**r, "any": []}, {"c"}, "")

    p = {**r, "pattern": r"r_rating >= 4\b"}
    assert mod.reading_matches(p, {"a", "c"}, "WHERE R_RATING\n   >=\t4")
    assert not mod.reading_matches(p, {"a", "c"}, "WHERE r_rating >= 3")
    assert not mod.reading_matches(p, {"c"}, "WHERE r_rating >= 4")


def test_reading_matches_strips_qualifiers_but_not_numbers():
    r = {"name": "x", "default": True, "sql": "", "any": [], "all": [], "none": []}
    assert mod.reading_matches({**r, "pattern": r"\bl_discount = 0\.05\b"}, set(), "WHERE l.l_discount  =  0.05")
    assert mod.reading_matches({**r, "pattern": r"^select l_tax from"}, set(), "SELECT lineitem.l_tax FROM lineitem")
    assert mod.reading_matches({**r, "pattern": r"o_orderyear = 1995"}, set(), "WHERE O.o_orderyear\n= 1995")
    assert not mod.reading_matches({**r, "pattern": r"= 05\b"}, set(), "WHERE l_discount = 0.05")


def test_glossary_terms_present():
    terms = {g["term"] for g in mod.load_spec("tpch")["glossary"]}
    assert {"revenue", "date basis", "late shipment"} <= terms
    terms = {g["term"] for g in mod.load_spec("reviews")["glossary"]}
    assert {"positive review", "negative review", "helpful", "average price", "product rating"} <= terms


@pytest.mark.parametrize("ds", ["tpch", "reviews"])
def test_each_reading_matches_only_itself(ds):
    spec = mod.load_spec(ds)
    names = _column_names(ds, spec)
    for term in spec["glossary"]:
        for reading in term["readings"]:
            cols = _columns_in(reading["sql"], names)
            matched = [r["name"] for r in term["readings"] if mod.reading_matches(r, cols, reading["sql"])]
            assert matched == [reading["name"]], (term["term"], reading["name"], matched)


@pytest.mark.parametrize("ds", ["tpch", "reviews"])
def test_terms_have_words_and_one_default(ds):
    spec = mod.load_spec(ds)
    for term in spec["glossary"]:
        assert term["words"] and sum(r["default"] for r in term["readings"]) == 1, term["term"]
        assert all(re.compile(w, re.IGNORECASE) for w in term["words"])


POSITIVE, FIVE = "4 or 5 stars", "5 stars only"
NEGATIVE, ONE = "1 or 2 stars", "1 star only"


@pytest.mark.parametrize("where, expected", [
    ("r_rating >= 4", POSITIVE), ("r_rating > 3", POSITIVE), ("r_rating IN (4, 5)", POSITIVE),
    ("r_rating in (5,4)", POSITIVE), ("r_rating BETWEEN 4 AND 5", POSITIVE), ("4 <= r_rating", POSITIVE),
    ("3 < r_rating", POSITIVE), ("r_rating >= 4.0", POSITIVE), ("r_rating = 4 OR r_rating = 5", POSITIVE),
    ("r_rating = 5 OR r_rating = 4", POSITIVE),
    ("r_rating = 5", FIVE), ("r_rating >= 5", FIVE), ("r_rating > 4", FIVE), ("5 = r_rating", FIVE),
    ("r_rating = 5 AND r_helpful_votes >= 1", FIVE),
    ("r.r_rating = 4 OR r.r_rating = 5", POSITIVE), ("(r_rating = 4) OR (r_rating = 5)", POSITIVE),
    ("(r.r_rating = 5) OR (r.r_rating = 4)", POSITIVE), ("4 <= r.r_rating", POSITIVE),
    ("3 < reviews.r_rating", POSITIVE), ("r.r_rating IN (4, 5)", POSITIVE), ("r.r_rating BETWEEN 4 AND 5", POSITIVE),
    ("r.r_rating = 5", FIVE), ("(r.r_rating = 5)", FIVE), ("reviews.r_rating >= 5", FIVE),
])
def test_positive_review_forms(where, expected):
    sql = f"SELECT COUNT(*) FROM reviews WHERE {where}"
    assert _matched("reviews", "positive review", sql) == [expected]


@pytest.mark.parametrize("where, expected", [
    ("r_rating <= 2", NEGATIVE), ("r_rating < 3", NEGATIVE), ("r_rating IN (1, 2)", NEGATIVE),
    ("r_rating BETWEEN 1 AND 2", NEGATIVE), ("2 >= r_rating", NEGATIVE), ("3 > r_rating", NEGATIVE),
    ("r_rating = 1 OR r_rating = 2", NEGATIVE), ("r_rating = 2 OR r_rating = 1", NEGATIVE),
    ("r_rating = 1", ONE), ("r_rating <= 1", ONE), ("r_rating < 2", ONE), ("1 = r_rating", ONE),
    ("r.r_rating = 1 OR r.r_rating = 2", NEGATIVE), ("(r_rating = 2) OR (r_rating = 1)", NEGATIVE),
    ("2 >= r.r_rating", NEGATIVE), ("3 > reviews.r_rating", NEGATIVE), ("r.r_rating IN (1, 2)", NEGATIVE),
    ("r.r_rating BETWEEN 1 AND 2", NEGATIVE), ("r.r_rating = 1", ONE), ("(r.r_rating <= 1)", ONE),
])
def test_negative_review_forms(where, expected):
    sql = f"SELECT COUNT(*) FROM reviews WHERE {where}"
    assert _matched("reviews", "negative review", sql) == [expected]


@pytest.mark.parametrize("sql, term, expected", [
    ("SELECT p.p_store, COUNT(*) FROM reviews r JOIN products p ON r.r_asin = p.p_asin "
     "WHERE r.r_rating >= 4 GROUP BY p.p_store", "positive review", POSITIVE),
    ("SELECT COUNT(*) FROM reviews r, products p WHERE r.r_asin = p.p_asin "
     "AND (r.r_rating = 4 OR r.r_rating = 5) AND p.p_has_price = 1", "positive review", POSITIVE),
    ("SELECT SUM(CASE WHEN r.r_rating = 5 THEN 1 ELSE 0 END) FROM reviews r JOIN products p "
     "ON r.r_asin = p.p_asin", "positive review", FIVE),
    ("SELECT p.p_title FROM products p JOIN reviews r ON p.p_asin = r.r_asin WHERE 2 >= r.r_rating",
     "negative review", NEGATIVE),
    ("SELECT COUNT(*) FROM products p, reviews r WHERE p.p_asin = r.r_asin AND (r.r_rating = 1)",
     "negative review", ONE),
])
def test_rating_terms_in_joins_with_aliases(sql, term, expected):
    assert _matched("reviews", term, sql) == [expected]


def test_rating_thresholds_that_are_neither_reading():
    for where in ("r_rating >= 3", "r_rating >= 40", "p_avg_rating >= 4"):
        assert _matched("reviews", "positive review", f"SELECT COUNT(*) FROM reviews WHERE {where}") == []
    assert _matched("reviews", "negative review", "SELECT COUNT(*) FROM reviews WHERE r_rating <= 3") == []


LATE_BY_ORDER_YEAR = ("SELECT o_orderyear, COUNT(*) FROM orders JOIN lineitem ON l_orderkey = o_orderkey "
                      "WHERE l_receiptdate > l_commitdate GROUP BY o_orderyear")


@pytest.mark.parametrize("term, sql, expected", [
    ("date basis", LATE_BY_ORDER_YEAR, ["order date"]),
    ("late shipment", LATE_BY_ORDER_YEAR, ["received after commit"]),
    ("date basis", "SELECT COUNT(*) FROM lineitem WHERE l_receiptdate > l_commitdate "
                   "AND l_shipdate >= DATE '1995-01-01' AND l_shipdate < DATE '1996-01-01'", ["ship date"]),
    ("date basis", "SELECT COUNT(*) FROM lineitem WHERE DATE '1995-01-01' <= l_receiptdate", ["receipt date"]),
    ("date basis", "SELECT l_shipmonth, COUNT(*) FROM lineitem GROUP BY l_shipmonth", ["ship date"]),
    ("date basis", "SELECT COUNT(*) FROM orders WHERE o_orderdate BETWEEN '1995-01-01' AND '1995-12-31'",
     ["order date"]),
    ("late shipment", "SELECT COUNT(*) FROM lineitem WHERE l_commitdate < l_shipdate", ["shipped after commit"]),
    ("late shipment", "SELECT COUNT(*) FROM lineitem WHERE l_receiptdate <= l_commitdate "
                      "AND l_shipyear = 1995", ["received after commit"]),
    ("revenue", "SELECT SUM(l_extendedprice * (1 - l_discount)) FROM lineitem", ["net"]),
    ("revenue", "SELECT SUM(l_extendedprice) FROM lineitem", ["gross"]),
])
def test_tpch_terms_across_each_other(term, sql, expected):
    assert _matched("tpch", term, sql) == expected


LATE_BY_ORDER_YEAR_ALIASED = ("SELECT o.o_orderyear, COUNT(*) FROM orders o JOIN lineitem l "
                              "ON l.l_orderkey = o.o_orderkey WHERE l.l_receiptdate > l.l_commitdate "
                              "GROUP BY o.o_orderyear")
IMPLICIT_SHIPPED = ("SELECT COUNT(*) FROM orders o, lineitem l WHERE l.l_orderkey = o.o_orderkey "
                    "AND l.l_shipdate > l.l_commitdate AND DATE '1995-01-01' <= o.o_orderdate")


@pytest.mark.parametrize("term, sql, expected", [
    ("date basis", LATE_BY_ORDER_YEAR_ALIASED, ["order date"]),
    ("late shipment", LATE_BY_ORDER_YEAR_ALIASED, ["received after commit"]),
    ("date basis", IMPLICIT_SHIPPED, ["order date"]),
    ("late shipment", IMPLICIT_SHIPPED, ["shipped after commit"]),
    ("date basis", "SELECT COUNT(*) FROM orders o WHERE o.o_orderyear = 1995", ["order date"]),
    ("date basis", "SELECT COUNT(*) FROM lineitem WHERE lineitem.l_shipdate >= DATE '1995-01-01'", ["ship date"]),
    ("date basis", "SELECT l.l_receiptmonth, COUNT(*) FROM lineitem l GROUP BY l.l_receiptmonth", ["receipt date"]),
    ("late shipment", "SELECT COUNT(*) FROM lineitem WHERE lineitem.l_receiptdate > lineitem.l_commitdate",
     ["received after commit"]),
    ("revenue", "SELECT SUM(l.l_extendedprice * (1 - l.l_discount)) FROM lineitem l, orders o "
                "WHERE l.l_orderkey = o.o_orderkey", ["net"]),
])
def test_tpch_terms_with_aliases(term, sql, expected):
    assert _matched("tpch", term, sql) == expected


@pytest.mark.parametrize("sql, expected", [
    ("SELECT COUNT(*) FROM lineitem l WHERE l.l_receiptdate > l.l_commitdate AND l.l_shipdate >= DATE '1995-01-01'",
     "received after commit"),
    ("SELECT COUNT(*) FROM lineitem WHERE l_receiptdate > l_commitdate AND l_shipdate >= DATE '1995-01-01'",
     "received after commit"),
    ("SELECT COUNT(*) FROM lineitem WHERE l_shipdate > l_commitdate", "shipped after commit"),
    ("SELECT COUNT(*) FROM lineitem l WHERE l.l_shipdate > l.l_commitdate", "shipped after commit"),
])
def test_late_shipment_with_a_ship_date_filter(sql, expected):
    assert _matched("tpch", "late shipment", sql) == [expected]


def test_late_shipment_reads_columns_only():
    late = next(g for g in mod.load_spec("tpch")["glossary"] if g["term"] == "late shipment")
    assert all("pattern" not in r for r in late["readings"])


def test_r_verified_sentence_matches_the_data():
    sentence = mod.load_spec("reviews")["columns"]["r_verified"].lower()
    assert "every" not in sentence and "all " not in sentence
    db = an.paths().dataset_dir("reviews") / "reviews.sqlite"
    if not db.exists():
        pytest.skip("prepare_datasets.py hasn't written reviews.sqlite")
    with closing(sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)) as conn:
        assert {v for (v,) in conn.execute("SELECT DISTINCT r_verified FROM reviews")} == {0, 1}


# A small results folder with both datasets, matching the given specs.
def _results(root: Path) -> Path:
    for ds in an.DATASETS:
        out = an.paths(root).dataset_dir(ds)
        out.mkdir(parents=True)
        (out / "manifest.json").write_text(json.dumps({**MANIFEST, "dataset": ds}))
        with closing(sqlite3.connect(out / f"{ds}.sqlite")) as conn:
            conn.execute("CREATE TABLE orders (o_orderkey INTEGER, o_status TEXT, o_price REAL)")
            conn.execute("CREATE TABLE lines (l_orderkey INTEGER, l_no INTEGER, l_price REAL, l_disc REAL)")
            conn.executemany("INSERT INTO orders VALUES (?, ?, ?)", [(1, "O", 1.0), (2, "F", 2.0), (3, "O", 3.0)])
            conn.commit()
    return root


def test_main_writes_descriptions(tmp_path, monkeypatch, capsys):
    res = _results(tmp_path / f"res_{os.getpid()}")
    monkeypatch.setattr(mod, "load_spec", lambda ds: {**_spec(), "dataset": ds})
    assert mod.main(["--results", str(res)]) == 0
    for ds in an.DATASETS:
        text = (an.paths(res).dataset_dir(ds) / "description.md").read_text()
        assert text.startswith(f"# {ds}") and "Values: 'F', 'O'" in text
    assert "description.md" in capsys.readouterr().out


def test_main_refuses_a_bad_spec_and_writes_nothing(tmp_path, monkeypatch, capsys):
    res = _results(tmp_path / f"bad_{os.getpid()}")
    broken = _spec()
    del broken["columns"]["l_disc"]
    monkeypatch.setattr(mod, "load_spec", lambda ds: {**copy.deepcopy(broken), "dataset": ds})
    assert mod.main(["--results", str(res)]) == 1
    err = capsys.readouterr().err
    assert "tpch: " in err and "reviews: " in err and "l_disc" in err
    assert not list(res.rglob("description.md"))


def test_main_reports_an_unreadable_description(tmp_path, monkeypatch, capsys):
    res = _results(tmp_path / f"broken_{os.getpid()}")

    # Stands in for a description file with a JSON syntax error.
    def broken(ds):
        raise ValueError("Expecting ',' delimiter: line 3 column 5")

    monkeypatch.setattr(mod, "load_spec", broken)
    assert mod.main(["--results", str(res)]) == 1
    err = capsys.readouterr().err
    assert "tpch: can't read" in err and "line 3 column 5" in err
    assert not list(res.rglob("description.md"))


def test_validate_survives_odd_column_entries():
    spec = _spec()
    spec["grains"]["order"] = [[["o_orderkey"]]]
    problems = mod.validate(spec, MANIFEST)
    assert len(problems) == 1 and "grain 'order'" in problems[0]


def test_main_needs_the_manifests(tmp_path, capsys):
    assert mod.main(["--results", str(tmp_path / f"empty_{os.getpid()}")]) == 1
    assert "prepare_datasets.py" in capsys.readouterr().err
