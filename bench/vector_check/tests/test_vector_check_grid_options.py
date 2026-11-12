"""Tests for the grid options, SQL and plan reading in bench/vector_check/pg_vector_grid.py."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
# Captured live on reviews_70k for query r00020,
# with filters date_1, none, rating_1 and product_random.
FIXTURES = HERE / "fixtures"


# Load a module by path under a vc_ name, clear of other bench tests.
def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, HERE.parent / file_name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mod = _load("vc_grid", "pg_vector_grid.py")
filters = _load("vc_grid_filters", "vector_filters.py")

GUC_NAMES = ["enable_seqscan", "enable_bitmapscan", "enable_indexscan", "enable_sort",
             "hnsw.ef_search", "hnsw.iterative_scan", "hnsw.max_scan_tuples"]
DEFAULTS = {"enable_seqscan": "on", "enable_bitmapscan": "on", "enable_indexscan": "on", "enable_sort": "on",
            "hnsw.ef_search": "40", "hnsw.iterative_scan": "off", "hnsw.max_scan_tuples": "20000"}
FORCE_HNSW = {"enable_seqscan": "off", "enable_bitmapscan": "off", "enable_sort": "off"}
CONFIG_NAMES = ["off_ef40", "off_ef100", "off_ef200", "off_ef400", "off_ef1000",
                "strict_ef40", "strict_ef100", "strict_ef200", "strict_ef400",
                "relaxed_ef40", "relaxed_ef100", "relaxed_ef200", "relaxed_ef400",
                "relaxed100k_ef40", "relaxed100k_ef100"]
# config name -> (iterative_scan, ef_search, max_scan_tuples), written out by hand
CONFIG_GUCS = {
    **{f"off_ef{ef}": ("off", ef, 20000) for ef in (40, 100, 200, 400, 1000)},
    **{f"strict_ef{ef}": ("strict_order", ef, 20000) for ef in (40, 100, 200, 400)},
    **{f"relaxed_ef{ef}": ("relaxed_order", ef, 20000) for ef in (40, 100, 200, 400)},
    **{f"relaxed100k_ef{ef}": ("relaxed_order", ef, 100000) for ef in (40, 100)},
}


# The plan in one saved EXPLAIN fixture.
def explain(name: str) -> list:
    return json.loads((FIXTURES / name).read_text())


# The shape classify_plan reads from one saved fixture.
def classify(name: str):
    return mod.classify_plan(explain(name))


# An option by name.
def option(name: str):
    return next(o for o in mod.ALL_OPTIONS if o.name == name)


# The hnsw GUCs I expect for one config name.
def hnsw_gucs(config_name: str) -> dict[str, str]:
    iterative, ef, max_scan = CONFIG_GUCS[config_name]
    return {"hnsw.ef_search": str(ef), "hnsw.iterative_scan": iterative, "hnsw.max_scan_tuples": str(max_scan)}


# Counts and the first names match the grid I want.
def test_option_counts_and_names():
    assert len(mod.HNSW_CONFIGS) == 15 and len(mod.FORCED) == 17 and len(mod.SETTINGS) == 16
    assert [o.name for o in mod.FORCED][:3] == ["exact_seq", "exact_bitmap", "hnsw_off_ef40"]
    assert "set_noindex" in {o.name for o in mod.SETTINGS} and len({o.name for o in mod.ALL_OPTIONS}) == 33
    assert list(mod.ALL_OPTIONS) == list(mod.FORCED) + list(mod.SETTINGS)


# Every option name, in order: forced first, then settings.
def test_option_order():
    assert [o.name for o in mod.FORCED] == ["exact_seq", "exact_bitmap", *(f"hnsw_{c}" for c in CONFIG_NAMES)]
    assert [o.name for o in mod.SETTINGS] == ["set_default", *(f"set_{c}" for c in CONFIG_NAMES[1:]), "set_noindex"]


# The 15 configs come in order with their names and values.
def test_hnsw_configs():
    assert [c.name for c in mod.HNSW_CONFIGS] == CONFIG_NAMES
    assert [(c.iterative, c.ef, c.max_scan) for c in mod.HNSW_CONFIGS] == [CONFIG_GUCS[n] for n in CONFIG_NAMES]
    assert mod.HnswConfig("strict_order", 100).max_scan == 20000


# An iterative mode pgvector doesn't have is refused.
def test_hnsw_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="iterative"):
        mod.HnswConfig("loose", 40)


# GUC_DEFAULTS is the 7 GUCs at Postgres and pgvector defaults, in order.
def test_guc_defaults():
    assert list(mod.GUC_DEFAULTS) == list(DEFAULTS.items())


# The two exact options and one hnsw option, GUC by GUC.
def test_forced_gucs():
    assert dict(option("exact_seq").gucs) == {**DEFAULTS, "enable_indexscan": "off", "enable_bitmapscan": "off"}
    assert dict(option("exact_bitmap").gucs) == {**DEFAULTS, "enable_indexscan": "off", "enable_seqscan": "off"}
    assert dict(option("hnsw_relaxed100k_ef40").gucs) == {
        "enable_seqscan": "off", "enable_bitmapscan": "off", "enable_indexscan": "on", "enable_sort": "off",
        "hnsw.ef_search": "40", "hnsw.iterative_scan": "relaxed_order", "hnsw.max_scan_tuples": "100000"}


# All 33 options carry the full 7 GUCs, in order, as strings, matching their definition.
def test_every_option_gucs_match_definition():
    for o in mod.ALL_OPTIONS:
        assert [n for n, _ in o.gucs] == GUC_NAMES, o.name
        assert all(isinstance(v, str) for _, v in o.gucs), o.name

    for c in CONFIG_NAMES:
        assert dict(option(f"hnsw_{c}").gucs) == {**DEFAULTS, **FORCE_HNSW, **hnsw_gucs(c)}
    for c in CONFIG_NAMES[1:]:
        assert dict(option(f"set_{c}").gucs) == {**DEFAULTS, **hnsw_gucs(c)}
    assert dict(option("set_default").gucs) == DEFAULTS
    assert dict(option("set_noindex").gucs) == {**DEFAULTS, "enable_indexscan": "off"}


# kind, intended and config for every option.
def test_option_fields():
    assert (option("exact_seq").kind, option("exact_seq").intended, option("exact_seq").config) == ("forced", "seq", None)
    assert (option("exact_bitmap").kind, option("exact_bitmap").intended, option("exact_bitmap").config) == (
        "forced", "bitmap", None)
    for c, cfg in zip(CONFIG_NAMES, mod.HNSW_CONFIGS):
        forced, setting = option(f"hnsw_{c}"), option("set_default" if c == "off_ef40" else f"set_{c}")
        assert (forced.kind, forced.intended, forced.config) == ("forced", "hnsw", cfg)
        assert (setting.kind, setting.intended, setting.config) == ("setting", None, cfg)
    assert (option("set_noindex").kind, option("set_noindex").intended, option("set_noindex").config) == (
        "setting", None, None)


# Only relaxed and relaxed100k configs are relaxed, forced and setting alike.
def test_relaxed_flag():
    relaxed = {o.name for o in mod.ALL_OPTIONS if o.relaxed}
    want = {f"{prefix}_{c}" for c in CONFIG_NAMES if c.startswith("relaxed") for prefix in ("hnsw", "set")}
    assert relaxed == want and len(relaxed) == 12


# SET for each of the 7 GUCs, then work_mem set or reset.
def test_set_statements_cover_all_gucs_and_work_mem():
    o = option("hnsw_strict_ef200")
    gucs = [
        "SET enable_seqscan = 'off'", "SET enable_bitmapscan = 'off'", "SET enable_indexscan = 'on'",
        "SET enable_sort = 'off'", "SET hnsw.ef_search = '200'", "SET hnsw.iterative_scan = 'strict_order'",
        "SET hnsw.max_scan_tuples = '20000'",
    ]
    assert mod.set_statements(o) == [*gucs, "RESET work_mem"]
    assert mod.set_statements(o, work_mem="512MB") == [*gucs, "SET work_mem = '512MB'"]
    for o in mod.ALL_OPTIONS:
        stmts = mod.set_statements(o)
        assert len(stmts) == 8 and stmts[:7] == [f"SET {n} = '{v}'" for n, v in o.gucs]


# Plain and relaxed query text, with and without a filter.
def test_query_sql_shapes():
    assert mod.query_sql("reviews_70k", "", False) == "SELECT id, embedding <=> %(q)s AS d FROM reviews_70k ORDER BY embedding <=> %(q)s LIMIT 10"
    assert mod.query_sql("reviews_70k", "rating = 1", True).startswith("WITH c AS MATERIALIZED (SELECT id, embedding <=> %(q)s AS d FROM reviews_70k WHERE rating = 1 ")
    assert mod.query_sql("reviews_70k", "rating = 1", True) == (
        "WITH c AS MATERIALIZED (SELECT id, embedding <=> %(q)s AS d FROM reviews_70k WHERE rating = 1 "
        "ORDER BY embedding <=> %(q)s LIMIT 10) SELECT id, d FROM c ORDER BY d + 0")
    assert mod.query_sql("reviews_500k", "", True) == (
        "WITH c AS MATERIALIZED (SELECT id, embedding <=> %(q)s AS d FROM reviews_500k "
        "ORDER BY embedding <=> %(q)s LIMIT 10) SELECT id, d FROM c ORDER BY d + 0")


# A % in a filter would clash with the %(q)s placeholders, so it's refused.
def test_query_sql_refuses_percent_in_filter():
    bad = "category LIKE 'Case%'"
    for relaxed in (False, True):
        with pytest.raises(ValueError, match=re.escape(bad)):
            mod.query_sql("reviews_70k", bad, relaxed)


# The estimate is an EXPLAIN of SELECT 1 over the table, filter optional.
def test_estimate_sql():
    assert mod.estimate_sql("reviews_70k", "rating <= 2") == "EXPLAIN (FORMAT JSON) SELECT 1 FROM reviews_70k WHERE rating <= 2"
    assert mod.estimate_sql("reviews_70k", "") == "EXPLAIN (FORMAT JSON) SELECT 1 FROM reviews_70k"


# Each saved plan reads as the scan, index and node list Postgres chose.
def test_classify_fixtures():
    assert classify("explain_exact_seq.json").scan == "seq"
    assert classify("explain_exact_bitmap.json").scan == "bitmap"
    assert classify("explain_hnsw_off.json").scan == "hnsw" and classify("explain_hnsw_off.json").index == "reviews_70k_hnsw_idx"
    assert classify("explain_hnsw_relaxed.json").scan == "hnsw"

    assert classify("explain_exact_seq.json") == mod.PlanShape("seq", None, ("Limit", "Sort", "Seq Scan"))
    assert classify("explain_exact_bitmap.json") == mod.PlanShape(
        "bitmap", "reviews_70k_review_date_idx",
        ("Limit", "Sort", "Bitmap Heap Scan", "Bitmap Index Scan:reviews_70k_review_date_idx"))
    assert classify("explain_hnsw_off.json") == mod.PlanShape(
        "hnsw", "reviews_70k_hnsw_idx", ("Limit", "Index Scan:reviews_70k_hnsw_idx"))
    assert classify("explain_hnsw_relaxed.json") == mod.PlanShape(
        "hnsw", "reviews_70k_hnsw_idx", ("Sort", "Limit", "Index Scan:reviews_70k_hnsw_idx", "CTE Scan"))
    assert classify("explain_set_default.json") == mod.PlanShape(
        "bitmap", "reviews_70k_product_id_idx",
        ("Limit", "Sort", "Bitmap Heap Scan", "Bitmap Index Scan:reviews_70k_product_id_idx"))
    assert classify("explain_estimate.json") == mod.PlanShape(
        "other", "reviews_70k_review_date_idx", ("Index Only Scan:reviews_70k_review_date_idx",))


# A plain B-tree Index Scan, and a plan with no table, read as I expect.
def test_classify_btree_and_no_table():
    btree = [{"Plan": {"Node Type": "Limit", "Plans": [
        {"Node Type": "Index Scan", "Index Name": "reviews_70k_rating_idx", "Relation Name": "reviews_70k"}]}}]
    assert mod.classify_plan(btree) == mod.PlanShape(
        "btree_index", "reviews_70k_rating_idx", ("Limit", "Index Scan:reviews_70k_rating_idx"))
    assert mod.classify_plan([{"Plan": {"Node Type": "Result"}}]) == mod.PlanShape("other", None, ("Result",))


# The top node's row estimate, as a float.
def test_estimated_rows_from_fixture():
    rows = mod.estimated_rows(explain("explain_estimate.json"))
    assert rows == 712.0 and isinstance(rows, float)
    assert mod.estimated_rows(explain("explain_hnsw_off.json")) == 10.0


# Every option runs on every filter, except a bitmap scan with no filter.
def test_applies():
    assert len(filters.FILTER_NAMES) == 15
    skipped = [(o.name, f) for o in mod.ALL_OPTIONS for f in filters.FILTER_NAMES if not mod.applies(o, f)]
    assert skipped == [("exact_bitmap", "none")]
    assert sum(mod.applies(o, f) for o in mod.FORCED for f in filters.FILTER_NAMES) == 15 * 17 - 1
