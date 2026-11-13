from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path


# Load an_common from next to this file, reusing it if it's already loaded.
def _load_common():
    path = Path(__file__).resolve().parent / "an_common.py"
    loaded = sys.modules.get("an_common")
    if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() == path:
        return loaded

    spec = importlib.util.spec_from_file_location("an_common", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"can't load {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["an_common"] = module
    spec.loader.exec_module(module)
    return module


an = _load_common()
bc = an.load_sibling("an_bridge_client", "bridge_client.py")
sc = an.load_sibling("an_scoring", "scoring.py")

DATASET = "tpch"
PREVIEW_ROWS = 3  # rows of each side kept in probe.json
OPENING = "Use only the SQL forms listed under Works below. Anything else may fail or give a wrong answer."
NOT_SUPPORTED = "not supported by this engine"
DIFFERENT = "runs, but gives a different answer than SQLite"

FORCED_UNSUPPORTED = {
    "left_join": "LEFT JOIN runs as an inner join here, so don't use it",
    "is_null": "there are no NULLs",
    "self_join": "a table can appear only once in a query",
}

# One example per SQL form, on the tpch tables.
# Every ORDER BY fixes the order completely.
SHAPES: list[tuple[str, str]] = [
    # plain SELECT parts
    ("count_star", "SELECT COUNT(*) AS order_count FROM orders"),
    ("sum_expr", "SELECT SUM(l_extendedprice * (1 - l_discount)) AS revenue FROM lineitem WHERE l_shipyear = 1995"),
    ("avg", "SELECT AVG(o_totalprice) AS avg_price FROM orders WHERE o_orderyear = 1996"),
    ("min_max", "SELECT MIN(o_orderdate) AS first_order, MAX(o_orderdate) AS last_order FROM orders"),
    ("group_by", "SELECT o_orderpriority, COUNT(*) AS order_count FROM orders GROUP BY o_orderpriority"),
    ("having",
     "SELECT c_nationkey, COUNT(*) AS customer_count FROM customer GROUP BY c_nationkey HAVING COUNT(*) > 600"),
    ("order_by_column", "SELECT n_name, n_regionkey FROM nation ORDER BY n_name"),
    ("order_by_alias",
     "SELECT l_shipmode, SUM(l_quantity) AS total_quantity FROM lineitem GROUP BY l_shipmode "
     "ORDER BY total_quantity DESC, l_shipmode"),
    ("order_by_aggregate",
     "SELECT l_shipmode, SUM(l_quantity) AS total_quantity FROM lineitem GROUP BY l_shipmode "
     "ORDER BY SUM(l_quantity) DESC, l_shipmode"),
    ("order_by_position", "SELECT n_name, n_regionkey FROM nation ORDER BY 1"),
    ("limit", "SELECT o_orderkey, o_totalprice FROM orders ORDER BY o_totalprice DESC, o_orderkey LIMIT 5"),
    ("offset", "SELECT o_orderkey, o_totalprice FROM orders ORDER BY o_totalprice DESC, o_orderkey LIMIT 5 OFFSET 5"),
    ("table_alias", "SELECT o.o_orderkey, o.o_orderdate FROM orders o WHERE o.o_orderkey <= 7"),
    # joins
    ("join_on",
     "SELECT n_name, COUNT(*) AS customer_count FROM customer "
     "JOIN nation ON c_nationkey = n_nationkey GROUP BY n_name"),
    ("implicit_join",
     "SELECT r_name, COUNT(*) AS nation_count FROM nation, region WHERE n_regionkey = r_regionkey GROUP BY r_name"),
    ("three_way_join",
     "SELECT r_name, SUM(c_acctbal) AS total_balance FROM customer JOIN nation ON c_nationkey = n_nationkey "
     "JOIN region ON n_regionkey = r_regionkey GROUP BY r_name"),
    ("self_join",
     "SELECT a.n_name, b.n_name AS neighbor FROM nation a "
     "JOIN nation b ON a.n_regionkey = b.n_regionkey WHERE a.n_nationkey = 0"),
    ("left_join",
     "SELECT c_custkey, o_orderkey FROM customer LEFT JOIN orders ON c_custkey = o_custkey WHERE c_custkey <= 3"),
    # predicates
    ("between", "SELECT COUNT(*) AS order_count FROM orders WHERE o_totalprice BETWEEN 100000 AND 200000"),
    ("in_list", "SELECT COUNT(*) AS line_count FROM lineitem WHERE l_shipmode IN ('MAIL', 'SHIP')"),
    ("not_in_list", "SELECT COUNT(*) AS line_count FROM lineitem WHERE l_shipmode NOT IN ('MAIL', 'SHIP')"),
    ("like", "SELECT COUNT(*) AS part_count FROM part WHERE p_type LIKE '%BRASS'"),
    ("not_like", "SELECT COUNT(*) AS part_count FROM part WHERE p_type NOT LIKE 'PROMO%'"),
    ("case_when",
     "SELECT o_orderpriority, SUM(CASE WHEN o_orderstatus = 'F' THEN 1 ELSE 0 END) AS finished "
     "FROM orders GROUP BY o_orderpriority"),
    ("date_literal",
     "SELECT COUNT(*) AS order_count FROM orders "
     "WHERE o_orderdate >= DATE '1995-01-01' AND o_orderdate < DATE '1996-01-01'"),
    ("date_compare_string",
     "SELECT COUNT(*) AS order_count FROM orders WHERE o_orderdate >= '1995-01-01' AND o_orderdate < '1996-01-01'"),
    # arithmetic
    ("arithmetic",
     "SELECT l_linenumber, l_extendedprice * (1 - l_discount) * (1 + l_tax) AS charge FROM lineitem "
     "WHERE l_orderkey = 1 ORDER BY l_linenumber"),
    ("integer_division",
     "SELECT p_partkey, p_size, p_size / 10 AS size_band FROM part WHERE p_partkey <= 5 ORDER BY p_partkey"),
    # what the engine lacks
    ("distinct", "SELECT DISTINCT o_orderpriority FROM orders"),
    ("count_distinct", "SELECT COUNT(DISTINCT o_custkey) AS customer_count FROM orders"),
    ("subquery_where",
     "SELECT COUNT(*) AS big_orders FROM orders WHERE o_totalprice > (SELECT AVG(o_totalprice) FROM orders)"),
    ("subquery_from",
     "SELECT AVG(line_count) AS avg_lines FROM (SELECT l_orderkey, COUNT(*) AS line_count FROM lineitem "
     "GROUP BY l_orderkey) AS per_order"),
    ("with_cte",
     "WITH big AS (SELECT o_orderkey FROM orders WHERE o_totalprice > 400000) SELECT COUNT(*) AS big_orders FROM big"),
    ("union", "SELECT n_name FROM nation WHERE n_regionkey = 0 UNION SELECT n_name FROM nation WHERE n_regionkey = 1"),
    ("is_null", "SELECT COUNT(*) AS customer_count FROM customer WHERE c_comment IS NULL"),
    ("coalesce", "SELECT c_custkey, COALESCE(c_acctbal, 0) AS balance FROM customer WHERE c_custkey <= 3"),
    ("extract",
     "SELECT EXTRACT(YEAR FROM o_orderdate) AS order_year, COUNT(*) AS order_count FROM orders "
     "GROUP BY EXTRACT(YEAR FROM o_orderdate)"),
    ("substring",
     "SELECT SUBSTRING(c_phone, 1, 2) AS country_code, COUNT(*) AS customer_count FROM customer "
     "GROUP BY SUBSTRING(c_phone, 1, 2)"),
    ("upper", "SELECT p_partkey, UPPER(p_name) AS name FROM part WHERE p_partkey <= 3"),
    ("cast", "SELECT o_orderkey, CAST(o_totalprice AS INTEGER) AS whole_price FROM orders WHERE o_orderkey <= 3"),
    ("select_no_from", "SELECT 1 + 1 AS two"),
]

_ORDER_BY = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)


# SQLite's column names and rows for the SQL; raises sqlite3.Error if it can't run.
def _sqlite_result(conn: sqlite3.Connection, sql: str) -> tuple[list[str], list[list]]:
    cur = conn.execute(an.to_sqlite(sql))
    rows = [list(row) for row in cur.fetchall()]
    return [d[0] for d in cur.description or ()], rows


# Run one shape on the bridge and, given conn, compare its answer with SQLite's.
def _check(bridge, conn: sqlite3.Connection | None, sql: str) -> dict:
    reply = bridge.query(sql)
    entry = {
        "sql": sql,
        "ok": reply.get("ok") is True,
        "error": reply.get("error"),
        "same_as_sqlite": None
    }
    if not entry["ok"]:
        return entry

    entry.update(row_count=reply["row_count"], rows=reply["rows"][:PREVIEW_ROWS])
    if conn is None:
        return entry

    try:
        cols, rows = _sqlite_result(conn, sql)
    except sqlite3.Error as e:
        entry["sqlite_error"] = str(e)
        return entry

    entry.update(sqlite_row_count=len(rows), sqlite_rows=rows[:PREVIEW_ROWS])
    shown_cols = [c["name"] for c in reply["columns"]]
    entry["same_as_sqlite"] = sc.match_result(shown_cols, reply["rows"], cols, rows, ordered=_ORDER_BY.search(sql) is not None, order_cols=[])
    return entry


# Every shape's bridge result, compared with SQLite when a connection is given.
def probe(bridge, conn: sqlite3.Connection | None = None) -> dict:
    if conn is not None:
        conn.execute("PRAGMA case_sensitive_like = ON")  # the engine's LIKE is case-sensitive
    return {"shapes": {name: _check(bridge, conn, sql) for name, sql in SHAPES}}


# Why a shape is off limits, or None if it works.
def _reason(name: str, entry: dict) -> str | None:
    if name in FORCED_UNSUPPORTED:
        return FORCED_UNSUPPORTED[name]
    if not entry["ok"]:
        return NOT_SUPPORTED
    if entry.get("same_as_sqlite") is False:
        return DIFFERENT
    return None


# The rules text: each working shape with its SQL, then each broken one with why.
def render_rules(probe: dict) -> str:
    works, broken = [], []
    for name, entry in probe["shapes"].items():
        reason = _reason(name, entry)
        if reason is None:
            works.append(f"- {name}: `{entry['sql']}`")
        else:
            broken.append(f"- {name}: {reason}")

    lines = ["# SQL you can use", "", OPENING, "", "## Works", "", *(works or ["- none"]), "", "## Doesn't work", "", *(broken or ["- none"])]
    return "\n".join(lines) + "\n"


# Probe the engine on tpch and write probe.json and subset_rules.md; 1 if it can't.
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Probe which SQL forms the engine runs and write subset_rules.md.")
    ap.add_argument("--results", type=Path, default=an.RESULTS)
    args = ap.parse_args(argv)
    p = an.paths(args.results)

    ds_dir = p.dataset_dir(DATASET)
    tsv, db = ds_dir / "tables.tsv", ds_dir / f"{DATASET}.sqlite"
    if not tsv.exists() or not db.exists():
        print(f"no tables.tsv or {db.name} in {ds_dir}; run prepare_datasets.py first", file=sys.stderr)
        return 1

    started = time.monotonic()
    try:
        with closing(sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)) as conn, bc.Bridge(tsv) as bridge:
            result = probe(bridge, conn)
    except (OSError, RuntimeError) as e:
        print(f"probe failed: {e}", file=sys.stderr)
        return 1

    rules = render_rules(result)
    an.write_text_atomic(p.probe, json.dumps(result, indent=2) + "\n")
    an.write_text_atomic(p.rules, rules)
    broken = sum(_reason(n, e) is not None for n, e in result["shapes"].items())
    print(f"{len(result['shapes']) - broken} work, {broken} don't, {time.monotonic() - started:.1f} s -> {p.rules}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
