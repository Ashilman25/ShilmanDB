from __future__ import annotations

import csv
import io
import itertools
import math

from app import sixd

DISPLAY_ROWS = 500
CELL_CHARS = 200

WHOLE_TYPES = ("INTEGER", "BIGINT")
NUMBER_TYPES = (*WHOLE_TYPES, "DECIMAL")
TEXT_TYPES = ("VARCHAR", "DATE")
PLAIN_ENDINGS = ("key", "year", "month")  # whole numbers shown without separators
PERIOD_ENDINGS = ("year", "month")        # whole numbers that can label bars
MIN_BARS, MAX_BARS = 2, 50


# Text cut to CELL_CHARS, ending in ... when it was longer.
def _cut(text: str) -> str:
    return text if len(text) <= CELL_CHARS else text[:CELL_CHARS - 3] + "..."


# Whether a column's name ends in one of the given words.
def _ends_in(column: dict, endings: tuple[str, ...]) -> bool:
    return column["name"].lower().endswith(endings)


# A number a bar can be drawn from: finite, never the "nan" or "inf" strings.
def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


# One cell as the table shows it.
def format_cell(value, column: dict) -> str:
    if value is None:
        return ""
    
    if isinstance(value, str):  # text, dates, and nan / inf / -inf in a number column
        return _cut(value)

    kind = column["type"]
    if kind == "DECIMAL":
        text = f"{value:,.2f}"
        return "0.00" if text == "-0.00" else text
    
    if kind in WHOLE_TYPES:
        return str(value) if _ends_in(column, PLAIN_ENDINGS) else f"{value:,}"
    return _cut(str(value))


# The bars' label column: the first text or date column, else the first year or month.
def _label_column(columns: list[dict]) -> int | None:
    texts = (i for i, c in enumerate(columns) if c["type"] in TEXT_TYPES)
    periods = (i for i, c in enumerate(columns) if c["type"] in WHOLE_TYPES and _ends_in(c, PERIOD_ENDINGS))
    return next(itertools.chain(texts, periods), None)


# Columns that hold a measure: numbers that aren't keys, years or months.
def _value_columns(columns: list[dict]) -> list[int]:
    return [i for i, c in enumerate(columns) if c["type"] in NUMBER_TYPES and not _ends_in(c, PLAIN_ENDINGS)]


# The chart for a result: one large number, bars, or none.
def chart_for(columns: list[dict], rows: list[list]) -> dict:
    values = _value_columns(columns)

    if len(rows) == 1 and len(values) == 1:
        i = values[0]
        number = rows[0][i]
        if _finite(number):
            return {
                "kind": "number", 
                "value": i, "number": number, 
                "text": format_cell(number, columns[i])
            }
            
        return {"kind": "none"}

    label = _label_column(columns)
    if MIN_BARS <= len(rows) <= MAX_BARS and label is not None and values:
        j = values[0]
        if all(_finite(row[j]) for row in rows):
            points = [[format_cell(row[label], columns[label]), row[j]] for row in rows]
            return {"kind": "bars", "label": label, "value": j, "points": points}
        
    return {"kind": "none"}


# One page of the reply's rows, formatted; an offset or limit below 0 counts as 0.
def display_rows(reply: dict, offset: int = 0, limit: int = DISPLAY_ROWS) -> list[list[str]]:
    columns = reply["columns"]
    start = max(0, offset)
    page = reply["rows"][start:start + max(0, limit)]
    return [[format_cell(value, column) for value, column in zip(row, columns)] for row in page]


# Every row the reply holds as CSV; csv writes numbers with str(), at full precision.
def csv_text(reply: dict) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(column["name"] for column in reply["columns"])
    writer.writerows(reply["rows"])
    return out.getvalue()


# The download link's text, saying when the engine cut the result at its row cap.
def csv_label(reply: dict) -> str:
    cap = sixd.an.ROW_CAP
    if reply.get("truncated"):
        return f"Download CSV (first {cap:,} of {reply['row_count']:,} rows)"
    return f"Download CSV (up to {cap:,} rows)"


# The plan's operators, root first, with each one's total time.
def plan_rows(profile: dict | None) -> list[dict]:
    operators = (profile or {}).get("operators", [])
    return [{"name": op["name"], "depth": op["depth"], "rows": op["rows"], "ms": op["total_ms"]} for op in operators]


# The line above the result: rows, the claimed grain, and the engine's time.
def meta_line(row_count: int, grain: str | None, ms: float) -> str:
    rows = "1 row" if row_count == 1 else f"{row_count:,} rows"
    grain = (grain or "").strip().rstrip(".")
    lead = f"{rows}, {grain}." if grain else f"{rows}."
    ran = f"Ran in {ms:,.0f} ms." if ms >= 1 else "Ran in under 1 ms."
    return f"{lead} {ran}"


# A failed reply in plain words; never raises, whatever the reply holds.
def error_text(reply: dict) -> str:
    error = reply.get("error") if isinstance(reply, dict) else None
    if not isinstance(error, dict):
        error = {}
    kind, message = error.get("kind"), error.get("message", "")

    match kind:
        case "parse" | "plan" | "run":
            return f"{kind.capitalize()} error: {message}"
        case "timeout":
            return f"Stopped after {sixd.an.TIMEOUT_S:g} s. The engine restarted (about 2 s)."
        case "crash":
            return "The engine stopped on this query and restarted."
        case "memory":
            return f"The query used more than {sixd.an.MAX_RSS_MB:,} MB, so the engine restarted."
        case _:
            return f"Engine error: {message}"


# A successful reply, ready to draw: first page of rows, chart, CSV label, plan, meta line.
def answer_view(reply: dict, sql: str, grain: str | None = None, question: str | None = None) -> dict:
    columns, rows = reply["columns"], reply["rows"]
    return {
        "columns": [{"name": c["name"], "type": c["type"]} for c in columns],
        "rows": display_rows(reply),
        "row_count": reply["row_count"],
        "shown": len(rows),  # rows the reply holds, so the page knows how many more it can fetch
        "truncated": bool(reply["truncated"]),
        "chart": chart_for(columns, rows),
        "csv_label": csv_label(reply),
        "sql": sql,
        "grain": grain,
        "plan": plan_rows(reply.get("profile")),
        "ms": reply["ms"],
        "meta": meta_line(reply["row_count"], grain, reply["ms"]),
        "question": question,
    }
