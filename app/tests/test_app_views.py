"""Tests for app/views.py."""
from __future__ import annotations

import csv
import io

from app import sixd
from app.views import (
    CELL_CHARS,
    DISPLAY_ROWS,
    answer_view,
    chart_for,
    csv_label,
    csv_text,
    display_rows,
    error_text,
    format_cell,
    meta_line,
    plan_rows,
)

DECIMAL = {"name": "revenue", "type": "DECIMAL"}
ANSWER_KEYS = {"columns", "rows", "row_count", "shown", "truncated", "chart", "csv_label",
               "sql", "grain", "plan", "ms", "meta", "question"}

REGION_SQL = ("SELECT r_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue, COUNT(*) AS line_items "
              "FROM lineitem JOIN orders ON l_orderkey = o_orderkey JOIN customer ON o_custkey = c_custkey "
              "JOIN nation ON c_nationkey = n_nationkey JOIN region ON n_regionkey = r_regionkey "
              "GROUP BY r_name ORDER BY revenue DESC")


# The region-revenue reply as the bridge sent it on 2026-11-14.
def _region_reply() -> dict:
    return {
        "id": "1",
        "ok": True,
        "columns": [{"name": "r_name", "type": "VARCHAR"}, {"name": "revenue", "type": "DECIMAL"},
                    {"name": "line_items", "type": "INTEGER"}],
        "rows": [
            ["AFRICA", 639835184.3816025, 18713],
            ["ASIA", 639301085.3240004, 18588],
            ["EUROPE", 634777681.1926016, 18551],
            ["MIDDLE EAST", 629625131.7911983, 18439],
            ["AMERICA", 600704573.035002, 17654],
        ],
        "row_count": 5,
        "truncated": False,
        "profile": {
            "join": {"tables": ["lineitem", "orders", "customer", "nation", "region"], "order": [4, 3, 2, 1, 0]},
            "step_rows": [5, 25, 15000, 150000],
            "scan_rows": [5, 25, 15000, 150000, 600572],
            "operators": [
                {"name": "Sort", "depth": 0, "rows": 5, "total_ms": 83.0, "self_ms": 0.1},
                {"name": "Projection", "depth": 1, "rows": 5, "total_ms": 82.9, "self_ms": 0.0},
            ],
        },
        "ms": 83.130375,
    }


# A successful reply with the given columns and rows.
def _reply(columns: list[dict], rows: list[list], row_count: int | None = None, truncated: bool = False) -> dict:
    return {"id": "2", "ok": True, "columns": columns, "rows": rows,
            "row_count": len(rows) if row_count is None else row_count, "truncated": truncated,
            "profile": {"operators": []}, "ms": 1.0}


# A failed reply of one kind.
def _failure(kind: str, message: str) -> dict:
    return {"id": "3", "ok": False, "error": {"kind": kind, "message": message}, "ms": 2.0}


def test_decimal_two_places_with_separators():
    assert format_cell(639835184.3816025, DECIMAL) == "639,835,184.38"


def test_decimal_rounding_is_python_formatting():
    # Scaling by 100 and rounding gives .69 here; the correctly rounded value is .67.
    assert format_cell(123456789012345.67, DECIMAL) == "123,456,789,012,345.67"
    assert format_cell(-0.001, DECIMAL) == "0.00"
    assert format_cell(-0.0, DECIMAL) == "0.00"
    assert format_cell(-1234.567, DECIMAL) == "-1,234.57"
    assert format_cell(5, DECIMAL) == "5.00"
    assert format_cell("inf", DECIMAL) == "inf"
    assert format_cell("-inf", DECIMAL) == "-inf"


def test_integer_separators():
    assert format_cell(18713, {"name": "line_items", "type": "INTEGER"}) == "18,713"
    assert format_cell(-6001215, {"name": "total", "type": "BIGINT"}) == "-6,001,215"


def test_years_months_keys_plain():
    assert format_cell(1995, {"name": "o_orderyear", "type": "INTEGER"}) == "1995"
    assert format_cell(199503, {"name": "r_month", "type": "INTEGER"}) == "199503"
    assert format_cell(600572, {"name": "o_orderkey", "type": "BIGINT"}) == "600572"


def test_text_and_dates_pass_through():
    assert format_cell("AFRICA", {"name": "r_name", "type": "VARCHAR"}) == "AFRICA"
    assert format_cell("1995-03-15", {"name": "o_orderdate", "type": "DATE"}) == "1995-03-15"
    assert format_cell("nan", DECIMAL) == "nan"


def test_long_text_cut_for_display():
    text = "abcdefghij" * 400
    reply = _reply([{"name": "title", "type": "VARCHAR"}], [[text]])

    cell = display_rows(reply)[0][0]
    assert len(text) == 4000
    assert len(cell) == CELL_CHARS == 200
    assert cell == text[:197] + "..."

    rows = list(csv.reader(io.StringIO(csv_text(reply))))
    assert rows[1] == [text]


def test_chart_bars_for_label_and_number():
    reply = _region_reply()
    chart = chart_for(reply["columns"], reply["rows"])

    assert chart == {
        "kind": "bars",
        "label": 0,
        "value": 1,
        "points": [
            ["AFRICA", 639835184.3816025],
            ["ASIA", 639301085.3240004],
            ["EUROPE", 634777681.1926016],
            ["MIDDLE EAST", 629625131.7911983],
            ["AMERICA", 600704573.035002],
        ],
    }


def test_chart_year_label():
    columns = [{"name": "o_orderyear", "type": "INTEGER"}, {"name": "order_count", "type": "INTEGER"}]
    rows = [[1992 + i, 22000 + i] for i in range(7)]
    chart = chart_for(columns, rows)

    assert chart["kind"] == "bars"
    assert (chart["label"], chart["value"]) == (0, 1)
    assert chart["points"][0] == ["1992", 22000]
    assert len(chart["points"]) == 7


def test_chart_date_label_and_key_skipped():
    columns = [{"name": "o_orderkey", "type": "BIGINT"}, {"name": "o_orderdate", "type": "DATE"}, DECIMAL]
    rows = [[1, "1995-03-15", 10.5], [2, "1995-03-16", 20.25], [3, "1995-03-17", 30.0]]
    chart = chart_for(columns, rows)

    assert (chart["kind"], chart["label"], chart["value"]) == ("bars", 1, 2)
    assert chart["points"][1] == ["1995-03-16", 20.25]


def test_chart_number_for_one_value():
    assert chart_for([DECIMAL], [[1234.5]]) == {"kind": "number", "value": 0, "number": 1234.5, "text": "1,234.50"}


def test_chart_none():
    reply = _region_reply()
    many = [[f"NATION {i}", float(i)] for i in range(51)]
    assert chart_for(reply["columns"][:2], many) == {"kind": "none"}
    assert chart_for(reply["columns"][:2], many[:50])["kind"] == "bars"

    two_numbers = [DECIMAL, {"name": "line_items", "type": "INTEGER"}]
    assert chart_for(two_numbers, [[1234.5, 7]]) == {"kind": "none"}

    text_only = [{"name": "r_name", "type": "VARCHAR"}, {"name": "r_comment", "type": "VARCHAR"}]
    assert chart_for(text_only, [["AFRICA", "x"], ["ASIA", "y"]]) == {"kind": "none"}

    keys = [{"name": "c_name", "type": "VARCHAR"}, {"name": "o_orderkey", "type": "BIGINT"},
            {"name": "o_custkey", "type": "INTEGER"}]
    assert chart_for(keys, [["a", 1, 2], ["b", 3, 4], ["c", 5, 6]]) == {"kind": "none"}

    assert chart_for(reply["columns"], []) == {"kind": "none"}


def test_chart_none_for_non_finite():
    reply = _region_reply()
    rows = [list(row) for row in reply["rows"]]
    rows[2][1] = "nan"
    assert chart_for(reply["columns"], rows) == {"kind": "none"}
    assert chart_for([DECIMAL], [["inf"]]) == {"kind": "none"}


def test_display_rows_pages():
    columns = [{"name": "l_orderkey", "type": "BIGINT"}, {"name": "l_quantity", "type": "DECIMAL"}]
    reply = _reply(columns, [[i, i * 1000.5] for i in range(1200)])

    assert DISPLAY_ROWS == 500
    assert len(display_rows(reply)) == 500
    assert len(display_rows(reply, 1000)) == 200
    assert display_rows(reply, 1000)[0] == ["1000", "1,000,500.00"]
    assert display_rows(reply, 1200) == []


def test_csv_raw_values():
    lines = csv_text(_region_reply()).splitlines()
    assert lines[0] == "r_name,revenue,line_items"
    assert lines[1] == "AFRICA,639835184.3816025,18713"
    assert len(lines) == 6

    title = 'Smith, "Jr"'
    text = csv_text(_reply([{"name": "title", "type": "VARCHAR"}], [[title]]))
    assert text.splitlines()[1] == '"Smith, ""Jr"""'
    assert list(csv.reader(io.StringIO(text)))[1] == [title]


def test_csv_label_truncated():
    reply = _reply([DECIMAL], [[1.0]], row_count=600572, truncated=True)
    assert csv_label(reply) == "Download CSV (first 10,000 of 600,572 rows)"


def test_csv_label_up_to_cap():
    assert csv_label(_region_reply()) == "Download CSV (up to 10,000 rows)"


def test_limits_read_from_sixd(monkeypatch):
    monkeypatch.setattr(sixd.an, "ROW_CAP", 2500)
    monkeypatch.setattr(sixd.an, "TIMEOUT_S", 45.0)
    monkeypatch.setattr(sixd.an, "MAX_RSS_MB", 8192)

    assert csv_label(_reply([DECIMAL], [[1.0]], row_count=12345, truncated=True)) == "Download CSV (first 2,500 of 12,345 rows)"
    assert error_text(_failure("timeout", "over 45 s")) == "Stopped after 45 s. The engine restarted (about 2 s)."
    assert error_text(_failure("memory", "over 8192 MB")) == "The query used more than 8,192 MB, so the engine restarted."


def test_plan_rows():
    op = {"name": "Sort", "depth": 0, "rows": 5, "total_ms": 83.0, "self_ms": 0.1}
    assert plan_rows({"operators": [op]}) == [{"name": "Sort", "depth": 0, "rows": 5, "ms": 83.0}]
    assert len(plan_rows(_region_reply()["profile"])) == 2


def test_meta_line():
    assert meta_line(5, "one row per region", 83.130375) == "5 rows, one row per region. Ran in 83 ms."
    assert meta_line(1, None, 3.2) == "1 row. Ran in 3 ms."


def test_meta_line_counts_and_times():
    assert meta_line(0, None, 12.0) == "0 rows. Ran in 12 ms."
    assert meta_line(600572, None, 1234.4) == "600,572 rows. Ran in 1,234 ms."
    assert meta_line(2, None, 0.4) == "2 rows. Ran in under 1 ms."
    assert meta_line(2, None, 1.0) == "2 rows. Ran in 1 ms."
    assert meta_line(5, "one row per region.", 83.0) == "5 rows, one row per region. Ran in 83 ms."


def test_error_texts():
    assert error_text(_failure("parse", "unexpected token ';'")) == "Parse error: unexpected token ';'"
    assert error_text(_failure("plan", "unknown column x")) == "Plan error: unknown column x"
    assert error_text(_failure("run", "division by zero")) == "Run error: division by zero"
    assert error_text(_failure("timeout", "over 30 s")) == "Stopped after 30 s. The engine restarted (about 2 s)."
    assert error_text(_failure("crash", "bridge died (exit code -11)")) == "The engine stopped on this query and restarted."
    assert error_text(_failure("memory", "over 4096 MB")) == "The query used more than 4,096 MB, so the engine restarted."


def test_error_text_other_kinds_never_raise():
    assert error_text(_failure("request", "expected id<TAB>sql")) == "Engine error: expected id<TAB>sql"
    assert error_text(_failure("disk", "full")) == "Engine error: full"
    assert error_text({"ok": False, "error": {"kind": ["parse"], "message": "odd"}}).startswith("Engine error:")
    assert error_text({"ok": False}).startswith("Engine error:")
    assert error_text({"ok": False, "error": "text"}).startswith("Engine error:")


def test_answer_view_keys():
    view = answer_view(_region_reply(), REGION_SQL, grain="one row per region", question="Revenue by region?")

    assert set(view) == ANSWER_KEYS
    assert len(view) == 13
    assert view["shown"] == 5
    assert view["rows"][0] == ["AFRICA", "639,835,184.38", "18,713"]
    assert view["columns"] == _region_reply()["columns"]
    assert (view["row_count"], view["truncated"]) == (5, False)
    assert view["chart"]["kind"] == "bars"
    assert view["csv_label"] == "Download CSV (up to 10,000 rows)"
    assert (view["sql"], view["grain"], view["question"]) == (REGION_SQL, "one row per region", "Revenue by region?")
    assert view["plan"][0] == {"name": "Sort", "depth": 0, "rows": 5, "ms": 83.0}
    assert view["ms"] == 83.130375
    assert view["meta"] == "5 rows, one row per region. Ran in 83 ms."


def test_answer_view_of_a_cut_result():
    columns = [{"name": "l_orderkey", "type": "BIGINT"}, {"name": "l_comment", "type": "VARCHAR"}]
    reply = _reply(columns, [[i, "c"] for i in range(10000)], row_count=600572, truncated=True)
    view = answer_view(reply, "SELECT l_orderkey, l_comment FROM lineitem")

    assert len(view["rows"]) == 500
    assert view["shown"] == 10000
    assert (view["row_count"], view["truncated"]) == (600572, True)
    assert view["chart"] == {"kind": "none"}
    assert view["csv_label"] == "Download CSV (first 10,000 of 600,572 rows)"
    assert view["meta"] == "600,572 rows. Ran in 1 ms."
    assert (view["grain"], view["question"]) == (None, None)
