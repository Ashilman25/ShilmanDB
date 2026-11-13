"""Unit tests for bench/analyst/sql_facts.py."""
from __future__ import annotations

import importlib.util
import sys
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


mod = _load("an_sql_facts", "sql_facts.py")
Agg = mod.Agg


# A table in the real manifest shape.
def _table(name: str, columns: list[tuple[str, str]]) -> dict:
    return {"name": name, "file": f"{name}.tbl", "rows": 3, "columns": [{"name": c, "type": t} for c, t in columns]}


M = {"dataset": "tpch", "tables": [
    _table("orders", [("o_orderkey", "BIGINT"), ("o_totalprice", "DECIMAL"), ("o_orderdate", "DATE"),
                      ("o_orderyear", "INTEGER"), ("o_custkey", "BIGINT")]),
    _table("lineitem", [("l_orderkey", "BIGINT"), ("l_quantity", "DECIMAL")]),
    _table("customer", [("c_custkey", "BIGINT"), ("c_name", "VARCHAR")]),
]}


def test_join_aggregates_and_equalities():
    f = mod.facts("SELECT c_name, SUM(o_totalprice) AS t FROM customer JOIN orders ON c_custkey = o_custkey "
                  "JOIN lineitem ON l_orderkey = o_orderkey GROUP BY c_name ORDER BY t DESC LIMIT 5", M)
    assert f.ok and f.tables == ["customer", "orders", "lineitem"]
    assert f.aggregates == [Agg("SUM", {"o_totalprice"}, False)] and f.group_by == ["c_name"]
    assert set(map(frozenset, f.equalities)) == {frozenset({"c_custkey", "o_custkey"}), frozenset({"l_orderkey", "o_orderkey"})}
    assert f.has_order_by and f.has_limit
def test_implicit_join_equalities_from_where():
    f = mod.facts("SELECT c_name FROM customer c, orders o WHERE c.c_custkey = o.o_custkey AND o_totalprice > 10 "
                  "AND (o_orderkey = 1 OR c_custkey = o_orderkey)", M)
    assert f.ok and f.tables == ["customer", "orders"]
    assert f.equalities == [("c_custkey", "o_custkey")]  # the OR term and the literal compare don't count
    assert not f.has_left_join and not f.has_order_by and not f.has_limit
def test_left_join_and_constant_conjunct():
    f = mod.facts("SELECT COUNT(*) FROM orders LEFT JOIN lineitem ON l_orderkey = o_orderkey WHERE 1 = 0 AND o_orderyear = 1995", M)
    assert f.has_left_join and f.constant_conjuncts == ["1 = 0"] and f.aggregates[0].star
def test_int_division_detected_only_for_integers():
    assert mod.facts("SELECT COUNT(*) / 7 FROM orders", M).int_divisions
    assert not mod.facts("SELECT SUM(o_orderyear) / COUNT(*) FROM orders", M).int_divisions   # SUM is DECIMAL
    assert not mod.facts("SELECT COUNT(*) * 1.0 / 7 FROM orders", M).int_divisions
def test_text_arithmetic(): assert mod.facts("SELECT c_name + 1 FROM customer", M).text_arithmetic
def test_self_join_tables_kept(): assert mod.facts("SELECT 1 FROM orders a, orders b", M).tables == ["orders", "orders"]
def test_unparseable_is_not_an_exception(): assert not mod.facts("SELECT FROM WHERE", M).ok
def test_select_columns_and_count_star_items():
    f = mod.facts("SELECT o.o_orderyear AS y, COUNT(*) AS n, COUNT(*) * 1.0 / 7, o_custkey FROM orders o GROUP BY o_orderyear, o_custkey", M)
    assert f.select_columns == ["o_orderyear", None, None, "o_custkey"] and f.count_star_items == [1, 2]


@pytest.mark.parametrize("sql", [
    "",
    "   -- only a comment",
    "SELECT c_name FROM customer WHERE c_name = 'Bob",                 # unterminated string
    "SELECT (o_orderkey FROM orders",                                  # unbalanced parentheses
    "SELECT o_orderkey) FROM orders",
    "SELECT SUM((o_totalprice) FROM orders",
    "SELECT o_orderkey FROM orders o extra",                           # trailing token
    "SELECT o_orderkey FROM orders LIMIT 5 6",
    "DELETE FROM orders",                                              # not a SELECT
    "INSERT INTO orders VALUES (1)",
    "WITH x AS (SELECT 1 FROM orders) SELECT 1 FROM x",
    "SELECT o_orderkey FROM orders WHERE GROUP = 1",                   # keyword where a column belongs
    "SELECT o_orderkey FROM orders GROUP BY ORDER",
    "SELECT o_orderkey FROM SELECT",
    "SELECT o_orderkey FROM orders JOIN lineitem",                     # JOIN without ON
    "SELECT o_orderkey FROM orders WHERE",
    "SELECT CASE ELSE 1 END FROM orders",                              # CASE needs a WHEN
    "SELECT CASE WHEN o_orderyear > 1 THEN 1 FROM orders",             # CASE without END
    "SELECT o_orderkey FROM orders WHERE o_orderdate > DATE 1995",     # DATE needs a string
    "SELECT o_orderkey FROM orders LIMIT 1.5",                         # LIMIT needs an integer
    "SELECT o_orderkey FROM orders WHERE o_orderkey = 3000000000",     # the engine's int literal is 32-bit
    "SELECT o.* FROM orders o",
    "SELECT COUNT(DISTINCT o_custkey) FROM orders",
    "SELECT upper(c_name) FROM customer",                              # no functions
    "SELECT o_orderkey FROM orders WHERE o_custkey IN (SELECT c_custkey FROM customer)",
    "SELECT o_orderkey FROM orders WHERE o_orderyear BETWEEN 1990",
    "SELECT o_orderkey FROM orders WHERE o_orderkey IN ()",
    "SELECT o_orderkey FROM orders WHERE o_orderkey # 1",              # a character the engine doesn't know
    "SELECT o_orderkey FROM orders WHERE o_orderkey NOT = 1",
    "SELECT " + "(" * 3000 + "1" + ")" * 3000 + " FROM orders",        # nested too deep
    None,
])
def test_malformed_input_gives_not_ok(sql):
    f = mod.facts(sql, M)
    assert not f.ok and f.error
    assert f.tables == [] and f.columns == set() and f.aggregates == [] and f.equalities == []


def test_semicolon_ends_the_statement():
    assert mod.facts("SELECT o_orderkey FROM orders;", M).ok
    f = mod.facts("SELECT o_orderkey FROM orders; DROP TABLE orders", M)  # the engine ignores what follows
    assert f.ok and f.tables == ["orders"]


def test_columns_drop_qualifiers_and_map_to_tables():
    f = mod.facts("SELECT o.o_orderkey, l.l_quantity FROM orders o JOIN lineitem l ON l.l_orderkey = o.o_orderkey "
                  "WHERE o.o_totalprice > 10 AND nope = 1", M)
    assert f.columns == {"o_orderkey", "l_quantity", "l_orderkey", "o_totalprice", "nope"}
    assert f.column_table == {"o_orderkey": "orders", "l_quantity": "lineitem", "l_orderkey": "lineitem",
                              "o_totalprice": "orders"}


def test_order_by_alias_is_not_a_column():
    f = mod.facts("SELECT o_custkey, SUM(o_totalprice) AS total FROM orders GROUP BY o_custkey ORDER BY total, o_orderyear", M)
    assert f.columns == {"o_custkey", "o_totalprice", "o_orderyear"}
    g = mod.facts("SELECT o_custkey AS total FROM orders WHERE total > 1", M)  # an alias only counts in ORDER BY
    assert "total" in g.columns


def test_distinct_parses_as_a_column_like_the_engine():
    f = mod.facts("SELECT DISTINCT o_custkey FROM orders", M)  # column DISTINCT, alias o_custkey
    assert f.ok and f.select_columns == ["DISTINCT"] and f.columns == {"DISTINCT"} and f.column_table == {}


def test_keywords_ignore_case_identifiers_keep_it():
    f = mod.facts("select Sum(O_TOTALPRICE) As t from ORDERS o group by o.O_CUSTKEY", M)
    assert f.ok and f.tables == ["ORDERS"] and f.columns == {"O_TOTALPRICE", "O_CUSTKEY"}
    assert f.aggregates == [Agg("SUM", {"O_TOTALPRICE"}, False)] and f.group_by == ["O_CUSTKEY"]
    assert f.column_table == {}  # the engine matches column names exactly


def test_aggregates_in_order_once_each():
    f = mod.facts("SELECT MIN(o_orderdate), COUNT(o_custkey), SUM(o_totalprice * l_quantity) FROM orders "
                  "JOIN lineitem ON l_orderkey = o_orderkey GROUP BY o_orderyear "
                  "HAVING SUM(o_totalprice) BETWEEN 1 AND 2 AND COUNT(*) IN (1, 2, 3) ORDER BY MAX(l_quantity) DESC", M)
    assert f.aggregates == [Agg("MIN", {"o_orderdate"}, False), Agg("COUNT", {"o_custkey"}, False),
                            Agg("SUM", {"o_totalprice", "l_quantity"}, False), Agg("SUM", {"o_totalprice"}, False),
                            Agg("COUNT", set(), True), Agg("MAX", {"l_quantity"}, False)]


def test_group_by_columns_from_expressions():
    f = mod.facts("SELECT COUNT(*) FROM orders GROUP BY o_orderyear + 1, o.o_custkey, o_orderyear", M)
    assert f.group_by == ["o_orderyear", "o_custkey"]


def test_equalities_split_on_and_and_follow_in():
    f = mod.facts("SELECT 1 FROM orders JOIN lineitem ON l_orderkey = o_orderkey AND l_quantity > 1 "
                  "JOIN customer ON c_custkey = o_custkey OR c_custkey = 1 WHERE o_custkey IN (c_custkey)", M)
    assert f.equalities == [("l_orderkey", "o_orderkey"), ("o_custkey", "c_custkey")]  # IN (x) is x = ..., as in the engine


def test_constant_conjuncts_as_the_engine_splits_them():
    f = mod.facts("SELECT 1 FROM orders WHERE 1 NOT IN (2, 3) AND 'a''b' = 'a''b' AND o_orderyear > 1 "
                  "AND (1 = 1 OR o_orderyear = 2) AND DATE '1995-01-01' < DATE '1996-01-01'", M)
    assert f.constant_conjuncts == ["1 <> 2", "1 <> 3", "'a''b' = 'a''b'", "DATE '1995-01-01' < DATE '1996-01-01'"]
    assert mod.facts("SELECT 1 FROM orders WHERE 2 BETWEEN o_orderyear AND 1", M).constant_conjuncts == ["2 <= 1"]


@pytest.mark.parametrize("expr, flagged", [
    ("o_orderyear / o_custkey", True),                                 # INTEGER / BIGINT
    ("o_totalprice / 2", False),
    ("MIN(o_orderyear) / 2", True),                                    # MIN keeps its argument's type
    ("MAX(o_totalprice) / 2", False),
    ("AVG(o_orderyear) / 2", False),
    ("CASE WHEN o_orderyear > 1 THEN 1 ELSE 0 END / 2", True),
    ("CASE WHEN o_orderyear > 1 THEN 1 ELSE 0.5 END / 2", False),
    ("CASE WHEN o_orderyear > 1 THEN 1 WHEN o_orderyear > 2 THEN c_name ELSE 2.5 END / 2", True),  # planner keeps the first type
    ("-o_orderyear / 2", True),
    ("(o_orderyear + 1) / 2", True),
    ("(o_orderyear + 1.5) / 2", False),
    ("(o_orderyear > 1) / 2", True),                                   # a comparison is INTEGER
    ("'7' / 2", False),
    ("DATE '1995-01-01' / 2", False),
    ("nope / 2", False),                                               # unknown column: no type
])
def test_int_division_types(expr, flagged):
    assert bool(mod.facts(f"SELECT {expr} FROM orders JOIN customer ON c_custkey = o_custkey", M).int_divisions) == flagged


def test_int_division_text_and_alias_types():
    assert mod.facts("SELECT o.o_orderyear / 2 FROM orders o", M).int_divisions == ["o.o_orderyear / 2"]
    assert mod.facts("SELECT COUNT(*) AS n FROM orders ORDER BY n / (1 + 2)", M).int_divisions == ["n / (1 + 2)"]
    assert mod.facts("SELECT SUM(o_orderyear) AS n FROM orders ORDER BY n / 2", M).int_divisions == []


@pytest.mark.parametrize("expr, flagged", [
    ("o_orderdate + 1", True),
    ("o_orderyear + 'x'", True),
    ("MIN(c_name) * 2", True),
    ("o_orderyear * 2 - o_totalprice", False),
    ("o_orderdate", False),
])
def test_text_arithmetic_sides(expr, flagged):
    f = mod.facts(f"SELECT {expr} FROM orders JOIN customer ON c_custkey = o_custkey WHERE o_orderdate > DATE '1995-01-01'", M)
    assert f.text_arithmetic == flagged


def test_select_columns_and_count_star_edge_cases():
    f = mod.facts("SELECT *, (o_custkey), (COUNT(*)), COUNT(o_orderkey), COUNT(*) + 1, COUNT(*) AS 'n' FROM orders", M)
    assert f.select_columns == [None, "o_custkey", None, None, None, None]
    assert f.count_star_items == [2, 4, 5]   # any item holding a COUNT(*), bare or in an expression
    f = mod.facts("SELECT CASE WHEN COUNT(*) > 1 THEN 1 ELSE 0 END, COUNT(o_orderkey) FROM orders "
                  "HAVING COUNT(*) > 0 ORDER BY COUNT(*)", M)
    assert f.count_star_items == [0]   # HAVING and ORDER BY aren't select items


def test_comments_strings_and_join_kinds():
    f = mod.facts("SELECT c_name -- the name\nFROM customer INNER JOIN orders ON c_custkey = o_custkey "
                  "WHERE c_name = 'x; FROM y' AND c_name LIKE 'A%' AND o_orderyear NOT BETWEEN 1 AND 2", M)
    assert f.ok and not f.has_left_join and f.tables == ["customer", "orders"]
    assert f.columns == {"c_name", "c_custkey", "o_custkey", "o_orderyear"} and f.constant_conjuncts == []
