"""Unit tests for bench/analyst/checks.py."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import NamedTuple

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


mod = _load("an_checks", "checks.py")
CheckResult = mod.CheckResult
SPEC = mod.dd.load_spec("tpch")

# Row counts at SF=0.1.
ROWS = {"region": 5, "nation": 25, "part": 20000, "supplier": 1000, "customer": 15000, "orders": 150000,
        "lineitem": 600572}
COLUMNS = {
    "region": "r_regionkey:BIGINT r_name:VARCHAR",
    "nation": "n_nationkey:BIGINT n_name:VARCHAR n_regionkey:BIGINT",
    "part": "p_partkey:BIGINT p_name:VARCHAR p_type:VARCHAR",
    "supplier": "s_suppkey:BIGINT s_name:VARCHAR s_nationkey:BIGINT s_acctbal:DECIMAL",
    "customer": "c_custkey:BIGINT c_name:VARCHAR c_nationkey:BIGINT c_acctbal:DECIMAL c_mktsegment:VARCHAR",
    "orders": "o_orderkey:BIGINT o_custkey:BIGINT o_orderstatus:VARCHAR o_totalprice:DECIMAL o_orderdate:DATE "
              "o_orderpriority:VARCHAR o_orderyear:INTEGER o_ordermonth:INTEGER",
    "lineitem": "l_orderkey:BIGINT l_partkey:BIGINT l_suppkey:BIGINT l_linenumber:INTEGER l_quantity:DECIMAL "
                "l_extendedprice:DECIMAL l_discount:DECIMAL l_shipdate:DATE l_commitdate:DATE l_receiptdate:DATE "
                "l_shipyear:INTEGER",
}


# A manifest in the real shape, from {table: "column:TYPE ..."}.
def _manifest(ds: str, columns: dict[str, str], rows: dict[str, int]) -> dict:
    return {"dataset": ds, "tables": [
        {"name": t, "file": f"{t}.tbl", "rows": rows[t],
         "columns": [{"name": n, "type": k} for n, k in (c.split(":") for c in cols.split())]}
        for t, cols in columns.items()
    ]}


MANIFEST = _manifest("tpch", COLUMNS, ROWS)
REVIEWS_SPEC = mod.dd.load_spec("reviews")
REVIEWS_MANIFEST = _manifest("reviews", {
    "reviews": "r_review_id:VARCHAR r_asin:VARCHAR r_rating:INTEGER r_verified:INTEGER r_year:INTEGER r_month:INTEGER",
    "products": "p_asin:VARCHAR p_title:VARCHAR p_store:VARCHAR p_category:VARCHAR",
}, {"reviews": 50000, "products": 8000})


class Case(NamedTuple):
    record: dict
    question: dict
    spec: dict = SPEC
    manifest: dict = MANIFEST


# A bridge-like reply: columns, rows and a profile in the real shape.
def _result(rows, columns=None, row_count=None, tables=(), order=None, scan_rows=None, step_rows=None,
            operators=None, types=None) -> dict:
    width = len(rows[0]) if rows else 1
    types = types or ["DECIMAL"] * width
    scan = list(scan_rows) if scan_rows is not None else [ROWS[t] for t in tables]
    step = list(step_rows) if step_rows is not None else [max(scan[:i + 1]) for i in range(len(scan))]
    if operators is None:
        operators = [{"name": f"SeqScan {t}", "depth": 1, "rows": ROWS[t], "total_ms": 1.0, "self_ms": 1.0}
                     for t in tables]
    return {
        "id": "1", "ok": True,
        "columns": [{"name": n, "type": k} for n, k in zip(columns or [f"c{i}" for i in range(width)], types)],
        "rows": rows, "row_count": len(rows) if row_count is None else row_count, "truncated": False, "ms": 5.0,
        "profile": {
            "join": {"tables": list(tables), "order": list(range(len(tables))) if order is None else list(order),
                     "source": "classical", "edges": []},
            "step_rows": step, "scan_rows": scan, "operators": operators,
        },
    }


# One saved answer, built by hand; the rephrase shows the same result unless told otherwise.
def rec(sql: str, q: str = "How much?", rows=None, claims_entity=None, grain=None, claims=None, rephrase=None,
        spec: dict = SPEC, manifest: dict = MANIFEST, **result_kw) -> Case:
    tables = result_kw.pop("tables", None) or mod.sf.facts(sql, manifest).tables
    result = _result([[1.0]] if rows is None else rows, tables=tables, **result_kw)
    if claims is None:
        width = len(result["columns"])
        claims = {"grain": grain,
                  "columns": [{"name": f"c{i}", "meaning": "", "entity": claims_entity} for i in range(width)]}
    if rephrase is None:
        rephrase = {"raw": "", "text": "Same question?", "seconds": 1.0, "prompt_sha1": "x", "attempts": [],
                    "final": "shown", "sql": sql, "result": copy.deepcopy(result)}

    attempt = {"raw": "", "kind": "sql", "sql": sql, "reply": result, "seconds": 1.0, "prompt_sha1": "x"}
    record = {
        "qid": "t900", "dataset": "tpch", "split": "dev", "model": "m", "prompt_sha1": "x",
        "primary": {"attempts": [attempt], "final": "shown", "sql": sql, "result": result, "claims": claims},
        "rephrase": rephrase, "sqlite_cause": None, "seconds_total": 2.0, "bridge_restarts": 0,
    }
    question = {"qid": "t900", "dataset": manifest["dataset"], "kind": "lookup", "question": q}
    return Case(record, question, spec, manifest)


def run(case: Case) -> list:
    return mod.run_checks(case.record, case.question, case.spec, case.manifest)


def get(case: Case, name: str):
    return next(r for r in run(case) if r.name == name)


def level_of(case: Case, name: str) -> str:
    return get(case, name).level


def level(name: str, sql: str, **kw) -> str:
    return level_of(rec(sql, **kw), name)


OL = "FROM orders JOIN lineitem ON l_orderkey = o_orderkey"


def test_left_join_refused():
    assert level("engine_traps",
                 sql="SELECT COUNT(*) FROM orders LEFT JOIN lineitem ON l_orderkey = o_orderkey") == "refuse"
def test_empty_filter_warns():
    r = rec("SELECT SUM(o_totalprice) FROM orders WHERE o_orderstatus = 'f'", row_count=1, scan_rows=[0], step_rows=[0])
    assert level_of(r, "empty_input") == "warn"
def test_fanout_sum_of_order_totals_over_lineitem():
    r = rec("SELECT SUM(o_totalprice) FROM orders JOIN lineitem ON l_orderkey = o_orderkey", scan_rows=[150000, 600572],
            step_rows=[150000, 600572])
    res = get(r, "fanout"); assert res.level == "warn" and "orders" in res.message
def test_fanout_evidence_follows_join_order():
    r = rec(f"SELECT SUM(o_totalprice) {OL}", tables=["orders", "lineitem"], order=[1, 0],
            scan_rows=[600572, 150000], step_rows=[600572, 600572])
    res = get(r, "fanout")
    assert res.level == "warn"
    assert res.evidence["repeating"] == {"orders": {"scan_rows": 150000, "joined_rows": 600572}}
    assert "150,000" in res.message and "600,572" in res.message
def test_no_fanout_sum_of_lineitem_over_orders():
    assert level_of(rec("SELECT SUM(l_quantity) FROM lineitem JOIN orders ON l_orderkey = o_orderkey"),
                    "fanout") == "pass"
def test_transitive_fanout_through_nation():
    sql = ("SELECT SUM(o_totalprice) FROM orders JOIN customer ON o_custkey = c_custkey "
           "JOIN nation ON c_nationkey = n_nationkey JOIN supplier ON s_nationkey = n_nationkey")
    res = get(rec(sql), "fanout")
    assert res.level == "warn" and "orders" in res.message
    assert set(res.evidence["repeating"]) == {"orders", "customer", "nation", "supplier"}
def test_count_star_counting_orders_over_lineitem():
    r = rec("SELECT COUNT(*) FROM orders JOIN lineitem ON l_orderkey = o_orderkey", claims_entity="orders")
    assert level_of(r, "fanout") == "warn"
def test_count_star_mapping():
    assert level_of(rec(f"SELECT COUNT(*) {OL}", claims_entity="line items"), "fanout") == "pass"
    assert level_of(rec(f"SELECT SUM(l_quantity) / COUNT(*) {OL}", claims_entity="line items"), "fanout") == "pass"
    assert level_of(rec(f"SELECT SUM(l_quantity) / COUNT(*) {OL}", claims_entity="orders"),
                    "fanout") == "warn"   # a ratio over orders
    assert level_of(rec(f"SELECT COUNT(*) {OL} ORDER BY COUNT(*) DESC", claims_entity="line items"), "fanout") == "pass"
    assert level_of(rec(f"SELECT COUNT(*) {OL} ORDER BY COUNT(*) DESC", claims_entity="orders"), "fanout") == "warn"
    assert level_of(rec(f"SELECT COUNT(*) {OL}", claims_entity="revenue"), "fanout") == "warn"   # can't be mapped
    assert level_of(rec(f"SELECT COUNT(*) {OL}", claims={"grain": None, "columns": []}),
                    "fanout") == "warn"   # past the claims
    assert level_of(rec("SELECT COUNT(*) FROM orders", claims={"grain": None, "columns": []}),
                    "fanout") == "pass"   # nothing repeats
    # A COUNT(*) only in HAVING or ORDER BY has no select item's claim.
    no_item = "SELECT n_name FROM customer JOIN nation ON c_nationkey = n_nationkey GROUP BY n_name {}"
    res = get(rec(no_item.format("ORDER BY COUNT(*) DESC"), rows=[["FRANCE"]], claims_entity="nation"), "fanout")
    assert res.level == "warn" and "outside the select list" in res.message
    assert level_of(rec(no_item.format("HAVING COUNT(*) > 5"), rows=[["FRANCE"]], claims_entity="nation"),
                    "fanout") == "warn"
def test_grain_covered_and_not():
    sql = "SELECT n_name, SUM(c_acctbal) FROM customer JOIN nation ON c_nationkey = n_nationkey GROUP BY {}"
    rows = [["FRANCE", 1.0], ["GERMANY", 2.0]]
    assert level_of(rec(sql.format("n_name"), rows=rows, grain="one row per nation"), "grain") == "pass"
    seg = ("SELECT c_mktsegment, SUM(c_acctbal) FROM customer JOIN nation ON c_nationkey = n_nationkey "
           "GROUP BY c_mktsegment")
    assert level_of(rec(seg, rows=[["AUTOMOBILE", 1.0], ["BUILDING", 2.0]], grain="one row per nation"),
                    "grain") == "warn"
    assert level_of(rec(sql.format("n_name"), rows=rows, grain="one row per widget"), "grain") == "note"
def test_grain_rows_repeat_on_shown_group_by():
    sql = ("SELECT c_name, SUM(o_totalprice) FROM customer JOIN orders ON c_custkey = o_custkey "
           "GROUP BY c_name, o_orderyear")
    res = get(rec(sql, rows=[["Customer#1", 10.0], ["Customer#1", 12.0]], grain="one row per customer"), "grain")
    assert res.level == "warn" and res.evidence["columns"] == ["c_name"] and res.evidence["repeated"] == ["Customer#1"]
    unique = [["Customer#1", 10.0], ["Customer#2", 12.0]]
    assert level_of(rec(sql, rows=unique, grain="one row per customer"), "grain") == "pass"
    assert level_of(rec(sql, rows=unique, grain="one row per nation"), "grain") == "warn"
    assert level_of(rec(sql, rows=unique, grain="one row per widget"), "grain") == "note"
def test_second_phrasing_mismatch_and_share_tolerance():
    base = rec("SELECT SUM(l_quantity) FROM lineitem", rows=[[0.2]])
    case = copy.deepcopy(base); case.record["rephrase"]["result"]["rows"] = [[20.0]]
    assert level_of(case, "second_phrasing") == "pass"
    case.record["rephrase"]["result"]["rows"] = [[0.3]]
    assert level_of(case, "second_phrasing") == "warn"
    case.record["rephrase"]["final"] = "failed"
    assert level_of(case, "second_phrasing") == "note"
def test_second_phrasing_share_on_either_side():
    case = rec("SELECT SUM(l_quantity) FROM lineitem", rows=[[20.0]])
    case.record["rephrase"]["result"]["rows"] = [[0.2]]   # the rephrase shows the fraction, the primary the percent
    assert level_of(case, "second_phrasing") == "pass"
    case.record["rephrase"]["result"]["columns"][0]["type"] = "VARCHAR"   # the swap scales only the rephrase's numbers
    assert level_of(case, "second_phrasing") == "warn"
def test_terms_default_note_other_warn():
    assert level_of(rec("SELECT SUM(l_extendedprice * (1 - l_discount)) FROM lineitem", q="Total revenue?"),
                    "terms") == "note"
    assert level_of(rec("SELECT SUM(l_extendedprice) FROM lineitem", q="Total revenue?"), "terms") == "warn"
def test_integer_division_warns():
    res = get(rec("SELECT COUNT(*) / 7 FROM orders"), "integer_division")
    assert res.level == "warn" and "COUNT(*) / 7" in res.message
    assert level_of(rec("SELECT COUNT(*) * 1.0 / 7 FROM orders"), "integer_division") == "pass"
def test_overall_and_skip():
    rs = [CheckResult("engine_traps", "refuse", "", {}), CheckResult("fanout", "warn", "", {})]
    assert mod.overall(rs) == "refused" and mod.overall(rs, skip={"engine_traps"}) == "warned"
    assert mod.overall(rs, skip=["engine_traps", "fanout"]) == "clean"
    assert mod.overall([CheckResult("exact", "pass", "", {}), CheckResult("terms", "note", "", {})]) == "clean"
def test_unreadable_sql_gives_notes():
    case = rec("SELECT FROM WHERE", rows=[[1.0]], tables=["orders"])
    levels = {r.name: r.level for r in run(case)}
    for name in ("engine_traps", "empty_input", "fanout", "grain", "terms", "integer_division"):
        assert levels[name] == "note", name
        assert get(case, name).message.startswith("couldn't read the SQL")
    assert levels["second_phrasing"] == "pass" and levels["exact"] == "pass"
    assert level_of(rec("SELECT FROM WHERE", rows=[], tables=["orders"]), "empty_input") == "warn"


def test_engine_traps_name_each():
    res = get(rec("SELECT SUM(o_totalprice) FROM orders a, orders b WHERE 1 = 0 AND o_orderdate + 1 > 2"),
              "engine_traps")
    assert res.level == "refuse"
    assert "orders" in res.message and "arithmetic" in res.message
    assert "1 = 0" not in res.message   # two table references, so the engine applies it
    assert res.evidence["repeated_tables"] == ["orders"] and res.evidence["constant_conjuncts"] == []
    assert res.evidence["text_arithmetic"] and not res.evidence["left_join"]
    left = get(rec("SELECT COUNT(*) FROM orders LEFT JOIN lineitem ON l_orderkey = o_orderkey"), "engine_traps")
    assert "LEFT JOIN" in left.message
    assert level("engine_traps", f"SELECT SUM(l_quantity) {OL}") == "pass"


def test_engine_traps_refuse_a_constant_conjunct_only_on_one_table():
    for sql in ["SELECT SUM(o_totalprice) FROM orders WHERE 1 = 0",
                "SELECT COUNT(*) FROM orders o WHERE o.o_orderyear > 1995 AND 1 = 0"]:
        res = get(rec(sql), "engine_traps")
        assert res.level == "refuse"
        assert res.message == "the engine runs this wrong: WHERE 1 = 0 names no column, so it's dropped"
        assert res.evidence["constant_conjuncts"] == ["1 = 0"]

    # With two or more table references the engine applies it.
    for sql in ["SELECT COUNT(*) FROM nation JOIN region ON n_regionkey = r_regionkey WHERE 1 = 0",
                "SELECT COUNT(*) FROM nation, region WHERE n_regionkey = r_regionkey AND 1 = 0",
                "SELECT COUNT(*) FROM nation, region WHERE 1 = 0",
                "SELECT COUNT(*) FROM supplier JOIN nation ON s_nationkey = n_nationkey "
                "JOIN region ON n_regionkey = r_regionkey WHERE 1 = 0",
                "SELECT COUNT(*) FROM supplier, nation, region WHERE s_nationkey = n_nationkey "
                "AND n_regionkey = r_regionkey AND 1 = 0"]:
        res = get(rec(sql), "engine_traps")
        assert res.level == "pass" and res.evidence["constant_conjuncts"] == [], sql


def test_engine_traps_refuse_a_case_only_term_on_one_table():
    case = "CASE WHEN n_regionkey > 1 THEN 1 ELSE 0 END = 1"
    res = get(rec(f"SELECT COUNT(*) FROM nation WHERE {case}"), "engine_traps")
    assert res.level == "refuse" and res.evidence["constant_conjuncts"] == [case]
    assert res.message == f"the engine runs this wrong: WHERE {case} names no column outside a CASE, so it's dropped"

    # Next to a column term, only the CASE term is dropped.
    res = get(rec(f"SELECT COUNT(*) FROM nation WHERE n_nationkey > 0 AND {case}"), "engine_traps")
    assert res.level == "refuse" and res.evidence["constant_conjuncts"] == [case]

    # A column outside the CASE in the same term, or a join,
    # and the engine applies it.
    for sql in ["SELECT COUNT(*) FROM nation WHERE n_nationkey + CASE WHEN n_regionkey > 1 THEN 1 ELSE 0 END > 3",
                f"SELECT COUNT(*) FROM nation WHERE {case} OR n_nationkey = 0",
                f"SELECT COUNT(*) FROM nation JOIN region ON n_regionkey = r_regionkey WHERE {case}"]:
        assert level("engine_traps", sql) == "pass", sql

    # A CASE with no column at all keeps the plain message.
    res = get(rec("SELECT COUNT(*) FROM nation WHERE CASE WHEN 1 = 0 THEN 1 ELSE 0 END = 1"), "engine_traps")
    assert res.message == ("the engine runs this wrong: WHERE CASE WHEN 1 = 0 THEN 1 ELSE 0 END = 1 "
                           "names no column, so it's dropped")


def test_engine_traps_refuse_an_on_naming_under_two_tables():
    res = get(rec("SELECT COUNT(*), SUM(n_nationkey) FROM nation JOIN region ON r_name = 'ASIA' "
                  "WHERE n_regionkey = r_regionkey", rows=[[4, 100.0]]), "engine_traps")
    assert res.level == "refuse" and "ON r_name = 'ASIA' names only region" in res.message
    assert "engine drops it" in res.message
    assert res.evidence["few_table_ons"] == [{"on": "r_name = 'ASIA'", "tables": ["region"]}]

    res = get(rec("SELECT COUNT(*) FROM nation JOIN region ON 1 = 1"), "engine_traps")
    assert res.level == "refuse" and "ON 1 = 1 names no table" in res.message and "engine drops it" in res.message


def test_engine_traps_self_join_refuses_only_for_the_repeated_table():
    res = get(rec("SELECT COUNT(*) FROM nation a JOIN nation b ON a.n_regionkey = b.n_nationkey"), "engine_traps")
    assert res.level == "refuse" and "nation appears twice" in res.message
    assert "drops it" not in res.message and res.evidence["few_table_ons"] == []


def test_engine_traps_pass_ons_naming_two_tables():
    for sql in ["SELECT COUNT(*) FROM nation JOIN region ON n_regionkey = r_regionkey AND r_name = 'ASIA'",
                "SELECT COUNT(*) FROM nation n JOIN region r ON n.n_regionkey = r.r_regionkey",
                "SELECT COUNT(*) FROM nation JOIN region ON n_regionkey < r_regionkey",
                "SELECT SUM(o_totalprice) FROM orders JOIN customer ON o_custkey = c_custkey "
                "JOIN nation ON c_nationkey = n_nationkey JOIN supplier ON s_nationkey = n_nationkey",
                f"SELECT SUM(l_quantity) {OL}"]:
        assert level("engine_traps", sql) == "pass", sql


def test_empty_result_warns_without_aggregates():
    res = get(rec("SELECT o_orderkey FROM orders WHERE o_orderstatus = 'X'", rows=[], scan_rows=[0], step_rows=[0]),
              "empty_input")
    assert res.level == "warn" and res.evidence["scan_rows"] == [0] and res.evidence["step_rows"] == [0]
    assert level("empty_input", "SELECT o_orderkey FROM orders", rows=[[1]]) == "pass"
    # Zero rows in a step matter only under an aggregate, which shows 0, not NULL.
    assert level("empty_input", "SELECT o_orderkey FROM orders", rows=[[1]], scan_rows=[0], step_rows=[0]) == "pass"
    res = get(rec(f"SELECT SUM(l_quantity) {OL}", scan_rows=[150000, 600572], step_rows=[150000, 0]), "empty_input")
    assert res.level == "warn" and "SUM, AVG, MIN and MAX" in res.message   # COUNT's 0 is the real answer


def test_empty_input_falls_back_to_operator_rows():
    sql = "SELECT SUM(o_totalprice) FROM orders WHERE o_orderstatus = 'f'"
    ops = [{"name": "Projection", "depth": 0, "rows": 1, "total_ms": 1.0, "self_ms": 0.0},
           {"name": "Aggregate", "depth": 1, "rows": 1, "total_ms": 1.0, "self_ms": 0.0},
           {"name": "Filter", "depth": 2, "rows": 0, "total_ms": 1.0, "self_ms": 0.5},
           {"name": "SeqScan orders", "depth": 3, "rows": 150000, "total_ms": 0.5, "self_ms": 0.5}]
    res = get(rec(sql, scan_rows=[], step_rows=[], operators=ops), "empty_input")
    assert res.level == "warn" and res.evidence["operator"] == {"name": "Filter", "rows": 0}

    ops[2]["rows"] = 7
    assert level_of(rec(sql, scan_rows=[], step_rows=[], operators=ops), "empty_input") == "pass"

    scan_only = [ops[0], ops[1], {**ops[3], "rows": 0}]  # no Filter: the scan's rows decide
    res = get(rec(sql, scan_rows=[], step_rows=[], operators=scan_only), "empty_input")
    assert res.level == "warn" and res.evidence["operator"] == {"name": "SeqScan orders", "rows": 0}


def test_fanout_reads_count_and_average_of_a_column_not_min_max():
    assert level("fanout", f"SELECT COUNT(o_orderkey) {OL}") == "warn"
    assert level("fanout", f"SELECT AVG(o_totalprice) {OL}") == "warn"
    assert level("fanout", f"SELECT MAX(o_totalprice), MIN(o_orderdate) {OL}") == "pass"
    res = get(rec(f"SELECT SUM(l_quantity) {OL}"), "fanout")
    assert res.level == "pass" and "orders" in res.message   # orders repeats, but nothing adds it up


def test_repeating_tables_by_hand():
    def rep(sql, spec=SPEC):
        return mod.repeating_tables(mod.sf.facts(sql, MANIFEST), spec)

    assert rep(f"SELECT 1 {OL}") == {"orders"}                                         # lineitem fans out from orders
    assert rep("SELECT 1 FROM lineitem "
               "JOIN orders ON l_orderkey = o_orderkey") == {"orders"}  # lineitem itself doesn't repeat
    assert rep("SELECT 1 FROM orders JOIN customer ON o_custkey = c_custkey JOIN nation ON c_nationkey = n_nationkey "
               "JOIN supplier ON s_nationkey = n_nationkey") == {"orders", "customer", "nation", "supplier"}
    assert rep("SELECT 1 FROM orders JOIN customer ON o_custkey = c_custkey") == {"customer"}
    assert rep("SELECT 1 FROM orders, customer WHERE o_custkey = c_custkey") == {"customer"}   # a WHERE join counts too
    assert rep("SELECT 1 FROM orders") == set()

    # Two equalities between the same tables cover a two-column key together.
    two = "SELECT 1 FROM orders JOIN lineitem ON l_orderkey = o_orderkey AND l_linenumber = o_orderyear"
    assert rep(two) == set()
    # A table with no declared key always fans out.
    assert rep(f"SELECT 1 {OL}",
               {"primary_keys": {"lineitem": ["l_orderkey", "l_linenumber"]}}) == {"orders", "lineitem"}


def test_cross_join_repeats_every_row_of_the_other_side():
    # The engine put region first: scan_rows and step_rows follow join order.
    r = rec("SELECT SUM(n_nationkey) FROM nation, region", order=[1, 0], scan_rows=[5, 25], step_rows=[5, 125])
    res = get(r, "fanout")
    assert res.level == "warn" and set(res.evidence["repeating"]) == {"nation", "region"}
    assert res.evidence["repeating"]["nation"] == {"scan_rows": 25, "joined_rows": 125}
    assert "25 rows scanned, 125 after the joins" in res.message


def test_forgotten_join_repeats_the_summed_table():
    sql = ("SELECT n_name, SUM(o_totalprice) FROM orders, customer, nation WHERE o_custkey = c_custkey "
           "GROUP BY n_name")
    r = rec(sql, rows=[["FRANCE", 1.0], ["GERMANY", 1.0]], types=["VARCHAR", "DECIMAL"],
            scan_rows=[150000, 15000, 25], step_rows=[150000, 150000, 3750000])
    res = get(r, "fanout")
    assert res.level == "warn" and "orders" in res.evidence["repeating"]
    assert res.evidence["repeating"]["orders"] == {"scan_rows": 150000, "joined_rows": 3750000}


def test_cross_join_with_a_one_row_side_passes():
    r = rec("SELECT SUM(n_nationkey) FROM nation, region WHERE r_name = 'ASIA'", order=[1, 0],
            scan_rows=[1, 25], step_rows=[1, 25])
    res = get(r, "fanout")
    assert res.level == "pass" and set(res.evidence["repeating"]) == {"region"}   # nation's rows show once


def test_forgotten_join_after_a_selective_join_repeats_the_summed_table():
    # Nation is never joined; a month of orders cuts lineitem, then nation repeats each row.
    sql = ("SELECT SUM(l_extendedprice) FROM lineitem, orders, nation WHERE l_orderkey = o_orderkey "
           "AND o_orderdate >= DATE '1995-01-01' AND o_orderdate < DATE '1995-02-01'")
    r = rec(sql, order=[2, 1, 0], scan_rows=[25, 1923, 600572], step_rows=[25, 48075, 191900])
    res = get(r, "fanout")
    assert res.level == "warn" and "SUM(l_extendedprice) adds up lineitem" in res.message
    assert res.evidence["repeating"]["lineitem"] == {"scan_rows": 600572, "joined_rows": 191900}


def test_forgotten_customer_nation_join_in_a_q5_shape_warns():
    sql = ("SELECT SUM(l_extendedprice * (1 - l_discount)) FROM lineitem, orders, customer, nation, region "
           "WHERE l_orderkey = o_orderkey AND o_custkey = c_custkey AND n_regionkey = r_regionkey "
           "AND r_name = 'EUROPE' AND o_orderdate >= DATE '1995-01-01' AND o_orderdate < DATE '1996-01-01'")
    r = rec(sql, order=[4, 3, 2, 1, 0], scan_rows=[1, 25, 15000, 22909, 600572],
            step_rows=[1, 5, 75000, 114545, 459725])
    res = get(r, "fanout")
    assert res.level == "warn" and "lineitem" in res.evidence["repeating"]


def test_non_equi_join_with_fewer_rows_out_than_in_warns():
    r = rec("SELECT SUM(n_nationkey) FROM nation JOIN region ON n_nationkey < r_regionkey", order=[1, 0],
            scan_rows=[5, 25], step_rows=[5, 10])
    res = get(r, "fanout")
    assert res.level == "warn" and "nation" in res.evidence["repeating"]   # nation 0 shows 4 times


def test_join_inside_every_or_branch_passes():
    # Every OR branch carries the equality, so it holds and covers region's key.
    sql = ("SELECT SUM(n_nationkey) FROM nation, region WHERE (n_regionkey = r_regionkey AND r_name = 'ASIA') "
           "OR (n_regionkey = r_regionkey AND r_name = 'EUROPE')")
    assert mod.sf.facts(sql, MANIFEST).equalities == [("n_regionkey", "r_regionkey")]
    r = rec(sql, order=[1, 0], scan_rows=[3, 6], step_rows=[3, 4])
    res = get(r, "fanout")
    assert res.level == "pass" and set(res.evidence["repeating"]) == {"region"}
    assert res.evidence["repeating"]["region"] == {"scan_rows": 3, "joined_rows": 4}


def test_repeating_tables_across_components():
    def rep(sql, scans=None):
        return mod.repeating_tables(mod.sf.facts(sql, MANIFEST), SPEC, scans)

    assert rep("SELECT 1 FROM nation, region") == {"nation", "region"}                      # no profile: both repeat
    assert rep("SELECT 1 FROM nation, region", {"nation": 25, "region": 1}) == {"region"}
    assert rep("SELECT 1 FROM nation, region", {"nation": 1, "region": 0}) == set()
    assert rep("SELECT 1 FROM nation JOIN region ON n_regionkey < r_regionkey",
               {"nation": 25, "region": 5}) == {"nation", "region"}                         # a non-equi ON ties nothing
    assert rep("SELECT 1 FROM orders, customer, nation WHERE o_custkey = c_custkey",
               {"orders": 150000, "customer": 15000}) == {"orders", "customer", "nation"}   # nation has no entry
    assert rep("SELECT 1 FROM orders, customer, nation WHERE o_custkey = c_custkey",
               {"orders": 150000, "customer": 15000, "nation": 1}) == {"customer", "nation"}

    # An equality in every OR branch ties the tables; in only some, it doesn't.
    every = ("SELECT 1 FROM nation, region WHERE (n_regionkey = r_regionkey AND r_name = 'ASIA') "
             "OR (r_regionkey = n_regionkey AND r_name = 'EUROPE')")
    some = "SELECT 1 FROM nation, region WHERE (n_regionkey = r_regionkey AND r_name = 'ASIA') OR r_name = 'EUROPE'"
    assert rep(every, {"nation": 6, "region": 3}) == {"region"}
    assert rep(some, {"nation": 6, "region": 3}) == {"nation", "region"}


def test_claimed_grain_and_entity_table():
    assert mod.claimed_grain("one row per customer nation", SPEC) == "customer nation"   # a tie goes to the longer key
    assert mod.claimed_grain("One row per ORDERS", SPEC) == "order"
    assert mod.claimed_grain("one row per line items", SPEC) == "line item"
    assert mod.claimed_grain("one row per border", SPEC) is None                       # whole words only
    assert mod.claimed_grain("one row", SPEC) is None
    assert mod.claimed_grain(None, SPEC) is None and mod.claimed_grain("", SPEC) is None
    assert mod.entity_table("orders", SPEC) == "orders"
    assert mod.entity_table("line items", SPEC) == "lineitem"
    assert mod.entity_table("revenue", SPEC) is None and mod.entity_table(None, SPEC) is None


def test_claimed_grain_after_per_each_by_then_earliest():
    store = "one row per store, for the three stores with the most verified-purchase reviews"
    assert mod.claimed_grain(store, REVIEWS_SPEC) == "store"
    assert mod.claimed_grain("Each row is a nation with its number of customers", SPEC) == "nation"
    assert mod.claimed_grain("number of orders per nation", SPEC) == "nation"
    assert mod.claimed_grain("one row per customer nation", SPEC) == "customer nation"
    assert mod.claimed_grain("totals by order date, one per customer", SPEC) == "order date"   # earliest after by


def test_entity_table_takes_the_earliest_match():
    assert mod.entity_table("orders placed by BUILDING customers", SPEC) == "orders"
    assert mod.entity_table("customers who placed orders", SPEC) == "customer"


def test_keys_match_hyphens_and_ies():
    assert mod.claimed_grain("one row per ship-mode", SPEC) == "ship mode"
    assert mod.entity_table("line-items", SPEC) == "lineitem"
    assert mod.claimed_grain("Each row is one of the countries", SPEC) == "country"
    assert mod.claimed_grain("one row per order priorities", SPEC) == "order priority"
    assert mod.claimed_grain("rows for the categories", REVIEWS_SPEC) == "category"
    assert mod.entity_table("countries", SPEC) == "nation"


def test_grain_on_the_prompt_reviews_example():
    sql = ("SELECT p_store, COUNT(*) AS verified_reviews FROM reviews JOIN products ON r_asin = p_asin "
           "WHERE r_verified = 1 GROUP BY p_store ORDER BY verified_reviews DESC, p_store LIMIT 3")
    claims = {"grain": "one row per store, for the three stores with the most verified-purchase reviews", "columns": [
        {"name": "p_store", "meaning": "store name on Amazon", "entity": "store"},
        {"name": "verified_reviews", "meaning": "number of verified-purchase reviews of the store's products",
         "entity": "reviews"},
    ]}
    case = rec(sql, q="Which three stores have the most verified-purchase reviews?",
               rows=[["Store A", 120], ["Store B", 90], ["Store C", 80]], types=["VARCHAR", "INTEGER"],
               columns=["p_store", "verified_reviews"], claims=claims, spec=REVIEWS_SPEC, manifest=REVIEWS_MANIFEST,
               scan_rows=[31000, 8000], step_rows=[31000, 31000], operators=[])
    res = get(case, "grain")
    assert res.level == "pass" and res.evidence["grain"] == "store"
    assert level_of(case, "fanout") == "pass"   # products repeats, but the count claims reviews


def test_prompt_ratio_example_is_clean():
    sql = ("SELECT 1.0 * SUM(CASE WHEN c_mktsegment = 'BUILDING' THEN 1 ELSE 0 END) / COUNT(*) AS building_share "
           "FROM customer JOIN nation ON c_nationkey = n_nationkey JOIN region ON n_regionkey = r_regionkey "
           "WHERE r_name = 'AMERICA'")
    claims = {"grain": "one row", "columns": [{
        "name": "building_share",
        "meaning": "customers in AMERICA in the BUILDING segment divided by all customers in "
                   "AMERICA, a fraction from 0 to 1",
        "entity": "customers",
    }]}
    case = rec(sql, q="What share of the customers in the AMERICA region belong to the BUILDING market segment?",
               rows=[[0.2]], columns=["building_share"], claims=claims,
               scan_rows=[15000, 5, 25], step_rows=[15000, 15000, 2950])
    res = get(case, "fanout")
    assert res.level == "pass" and set(res.evidence["repeating"]) == {"nation", "region"}
    assert mod.overall(run(case)) == "clean"


def test_order_by_count_star_judged_by_the_select_list_claim():
    sql = ("SELECT n_name, COUNT(*) FROM customer JOIN nation ON c_nationkey = n_nationkey "
           "GROUP BY n_name ORDER BY COUNT(*) DESC")
    def claims(entity):
        return {"grain": "one row per nation", "columns": [
            {"name": "n_name", "meaning": "nation", "entity": "nation"},
            {"name": "c1", "meaning": "number of customers", "entity": entity}]}
    rows = [["FRANCE", 600], ["GERMANY", 590]]
    assert level_of(rec(sql, rows=rows, claims=claims("customers")), "fanout") == "pass"
    assert level_of(rec(sql, rows=rows, claims=claims("nations")), "fanout") == "warn"   # nation repeats


def test_grain_without_group_by_passes():
    assert level("grain", "SELECT SUM(o_totalprice) FROM orders", grain="one row per widget") == "pass"


def test_second_phrasing_compares_rows_and_skipped_is_a_note():
    rows = [["FRANCE", 3], ["GERMANY", 4]]
    case = rec("SELECT n_name, COUNT(*) FROM nation GROUP BY n_name", rows=rows, types=["VARCHAR", "INTEGER"])
    assert level_of(case, "second_phrasing") == "pass"
    case.record["rephrase"]["result"]["rows"] = [["GERMANY", 4], ["FRANCE", 3]]  # row order doesn't matter
    assert level_of(case, "second_phrasing") == "pass"
    case.record["rephrase"]["result"].update(rows=[["FRANCE", 3]], row_count=1)
    res = get(case, "second_phrasing")
    assert res.level == "warn" and res.evidence["primary_row_count"] == 2 and res.evidence["rephrase_row_count"] == 1

    case.record["rephrase"].update(final="skipped", text=None, sql=None, result=None)
    assert level_of(case, "second_phrasing") == "note"


def test_terms_several_terms_give_one_result():
    net = "SUM(l_extendedprice * (1 - l_discount))"
    by_order_year = rec(f"SELECT o_orderyear, {net} {OL} GROUP BY o_orderyear", q="Revenue by year?")
    res = get(by_order_year, "terms")
    assert res.level == "note"
    assert "revenue read as net" in res.message and "date basis read as order date" in res.message
    assert [t["term"] for t in res.evidence["terms"]] == ["revenue", "date basis"]

    res = get(rec(f"SELECT l_shipyear, {net} FROM lineitem GROUP BY l_shipyear", q="Revenue by year?"), "terms")
    assert res.level == "warn" and "revenue read as net" in res.message and "ship date" in res.message
    assert level("terms", "SELECT COUNT(*) FROM orders", q="How many orders?") == "pass"
    assert level("terms", "SELECT COUNT(*) FROM orders", q="How many sales?") == "warn"   # no reading fits


def test_exact_always_passes():
    res = get(rec("SELECT COUNT(*) FROM orders"), "exact")
    assert res.level == "pass" and res.message == "exact: full scan, no sampling"


def test_run_checks_shape():
    cases = [rec(f"SELECT SUM(o_totalprice) {OL}"), rec("SELECT FROM WHERE", tables=["orders"]),
             rec("SELECT c_name, COUNT(*) FROM customer GROUP BY c_name", rows=[["a", 1], ["a", 2]],
                 q="Sales in 1995?")]
    for case in cases:
        results = run(case)
        assert [r.name for r in results] == list(mod.CHECKS)
        for r in results:
            assert r.level in mod.LEVELS and r.message
            json.dumps(r.evidence)   # plain JSON only


@pytest.mark.parametrize("final", ["cannot", "failed"])
def test_only_shown_answers_are_checked(final):
    case = rec("SELECT COUNT(*) FROM orders")
    case.record["primary"].update(final=final, sql=None, result=None, claims=None)
    with pytest.raises(ValueError, match=final):
        run(case)


def test_missing_profile_and_empty_claims_still_check():
    case = rec(f"SELECT SUM(o_totalprice), COUNT(*) {OL} GROUP BY o_orderyear", rows=[[1.0, 2]],
               claims={"grain": None, "columns": []})
    case.record["primary"]["result"].pop("profile")
    results = {r.name: r for r in run(case)}
    assert results["empty_input"].level == "pass"
    assert results["fanout"].level == "warn"
    assert results["fanout"].evidence["repeating"] == {"orders": {"scan_rows": None, "joined_rows": None}}
    assert results["grain"].level == "note"


def test_run_checks_leaves_its_inputs_alone():
    case = rec(f"SELECT SUM(o_totalprice) {OL}", claims_entity="orders")
    before = copy.deepcopy(case)
    run(case); run(case)
    assert case == before
