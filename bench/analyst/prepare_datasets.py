from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

import pandas as pd


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
REPO_ROOT, TEXT_CAP_BYTES = an.REPO_ROOT, an.TEXT_CAP_BYTES

UNKNOWN = "Unknown"  # stand-in for a missing store or category
SQLITE_TYPES = {"INTEGER": "INTEGER", "BIGINT": "INTEGER", "DECIMAL": "REAL", "VARCHAR": "TEXT", "DATE": "TEXT"}
_TO_SQLITE = {"INTEGER": int, "BIGINT": int, "DECIMAL": float, "VARCHAR": str, "DATE": str}
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_BREAKS = str.maketrans("|\t\r\n", "    ")

# The 7 tables load_tpch loads, as bench/tpch/tpch_workload.hpp defines them.
TPCH_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "region": [("r_regionkey", "BIGINT"), ("r_name", "VARCHAR"), ("r_comment", "VARCHAR")],
    "nation": [("n_nationkey", "BIGINT"), ("n_name", "VARCHAR"), ("n_regionkey", "BIGINT"), ("n_comment", "VARCHAR")],
    "part": [
        ("p_partkey", "BIGINT"), ("p_name", "VARCHAR"), ("p_mfgr", "VARCHAR"), ("p_brand", "VARCHAR"),
        ("p_type", "VARCHAR"), ("p_size", "INTEGER"), ("p_container", "VARCHAR"), ("p_retailprice", "DECIMAL"),
        ("p_comment", "VARCHAR"),
    ],
    "supplier": [
        ("s_suppkey", "BIGINT"), ("s_name", "VARCHAR"), ("s_address", "VARCHAR"), ("s_nationkey", "BIGINT"),
        ("s_phone", "VARCHAR"), ("s_acctbal", "DECIMAL"), ("s_comment", "VARCHAR"),
    ],
    "customer": [
        ("c_custkey", "BIGINT"), ("c_name", "VARCHAR"), ("c_address", "VARCHAR"), ("c_nationkey", "BIGINT"),
        ("c_phone", "VARCHAR"), ("c_acctbal", "DECIMAL"), ("c_mktsegment", "VARCHAR"), ("c_comment", "VARCHAR"),
    ],
    "orders": [
        ("o_orderkey", "BIGINT"), ("o_custkey", "BIGINT"), ("o_orderstatus", "VARCHAR"), ("o_totalprice", "DECIMAL"),
        ("o_orderdate", "DATE"), ("o_orderpriority", "VARCHAR"), ("o_clerk", "VARCHAR"), ("o_shippriority", "INTEGER"),
        ("o_comment", "VARCHAR"),
    ],
    "lineitem": [
        ("l_orderkey", "BIGINT"), ("l_partkey", "BIGINT"), ("l_suppkey", "BIGINT"), ("l_linenumber", "INTEGER"),
        ("l_quantity", "DECIMAL"), ("l_extendedprice", "DECIMAL"), ("l_discount", "DECIMAL"), ("l_tax", "DECIMAL"),
        ("l_returnflag", "VARCHAR"), ("l_linestatus", "VARCHAR"), ("l_shipdate", "DATE"), ("l_commitdate", "DATE"),
        ("l_receiptdate", "DATE"), ("l_shipinstruct", "VARCHAR"), ("l_shipmode", "VARCHAR"), ("l_comment", "VARCHAR"),
    ],
}

REVIEW_COLUMNS = ["review_id", "rating", "title", "text", "parent_asin", "timestamp", "helpful_vote", "verified_purchase", "user_id"]
PRODUCT_COLUMNS = ["parent_asin", "title", "store", "price", "main_category", "categories", "average_rating", "rating_number"]


# One table: its name, (column, engine type) pairs and rows.
class Table(NamedTuple):
    name: str
    columns: list[tuple[str, str]]  # (name, engine type)
    rows: list[list]


# Text safe for a .tbl line: no '|', tabs or line breaks, 4,000 bytes max.
def clean_text(s: str | None) -> str:
    if s is None:
        return ""
    return s.translate(_BREAKS).encode()[:TEXT_CAP_BYTES].decode("utf-8", "ignore")


# Add <x>year and <x>month (YYYYMM) INTEGER columns at the end for every DATE column <x>date.
# Rows are extended in place and returned.
def add_year_month(columns: list[tuple[str, str]], rows: list[list]) -> tuple[list[tuple[str, str]], list[list]]:
    dates = [(i, name.removesuffix("date")) for i, (name, kind) in enumerate(columns) if kind == "DATE" and name.endswith("date")]
    added = [col for _, x in dates for col in ((f"{x}year", "INTEGER"), (f"{x}month", "INTEGER"))]

    parts: dict[str, tuple[int, int]] = {}  # date -> (year, YYYYMM); dates repeat a lot
    for row in rows:
        for i, _ in dates:
            d = row[i]
            if d not in parts:
                parts[d] = (int(d[:4]), int(d[:4] + d[5:7]))

            row.extend(parts[d])
    return list(columns) + added, rows


# Read one TPC-H .tbl; every line must have the table's fields and end with '|'.
def _read_tbl(path: Path, name: str, columns: list[tuple[str, str]]) -> Table:
    rows = []
    with open(path, encoding="utf-8", newline="\n") as f:
        for number, line in enumerate(f, 1):
            line = line.removesuffix("\n")
            if not line:
                continue

            fields = line.split("|")
            if len(fields) != len(columns) + 1 or fields[-1]:
                raise ValueError(f"{path}:{number}: expected {len(columns)} fields, each followed by '|'")
            fields.pop()
            rows.append(fields)
    return Table(name, *add_year_month(columns, rows))


# The 7 TPC-H tables, values kept as the source text, plus the date parts.
def build_tpch(tpch_dir: Path) -> list[Table]:
    return [_read_tbl(Path(tpch_dir) / f"{name}.tbl", name, cols) for name, cols in TPCH_COLUMNS.items()]


# Read a CSV where only an empty field counts as missing, and text stays text.
def _read_csv(path: Path, columns: list[str], text: Iterable[str]) -> pd.DataFrame:
    return pd.read_csv(path, usecols=columns, dtype={c: str for c in text}, keep_default_na=False, na_values=[""])


# Raise ValueError if any of these columns has a missing value.
def _require(df: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    missing = [c for c in columns if df[c].isna().any()]
    if missing:
        raise ValueError(f"{path}: missing values in {', '.join(missing)}")


# Raise ValueError unless each of these columns holds whole numbers, none missing.
def _require_ints(df: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    bad = [c for c in columns if not pd.api.types.is_integer_dtype(df[c])]
    if bad:
        raise ValueError(f"{path}: {', '.join(bad)} must be whole numbers, none missing")


# Cleaned text per value, with a stand-in for a missing one.
def _texts(values: pd.Series, missing: str = "") -> list[str]:
    return [clean_text(s) if isinstance(s, str) else missing for s in values.tolist()]


# The most specific category: the last entry of the JSON list, else 'Unknown'.
def _last_category(categories: object) -> str:
    entries = json.loads(categories) if isinstance(categories, str) else []
    return clean_text(entries[-1]) if entries else UNKNOWN


# Rows from per-column value lists.
def _rows(columns: list[list]) -> list[list]:
    return [list(row) for row in zip(*columns, strict=True)]


# One row per review; the date is the UTC day, plus its year and month.
def _reviews_table(path: Path) -> Table:
    df = _read_csv(path, REVIEW_COLUMNS, text=("title", "text", "parent_asin", "user_id"))
    _require(df, ("parent_asin", "user_id"), path)
    _require_ints(df, ("review_id", "rating", "timestamp", "helpful_vote"), path)

    verified = df["verified_purchase"].map({True: 1, False: 0})
    if verified.isna().any():
        raise ValueError(f"{path}: verified_purchase must be True or False")
    day = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")

    columns = [
        ("r_review_id", "BIGINT"), ("r_asin", "VARCHAR"), ("r_rating", "INTEGER"), ("r_title", "VARCHAR"),
        ("r_text", "VARCHAR"), ("r_date", "DATE"), ("r_helpful_votes", "INTEGER"), ("r_verified", "INTEGER"),
        ("r_user_id", "VARCHAR")
    ]
    rows = _rows([
        df["review_id"].tolist(), df["parent_asin"].tolist(), df["rating"].tolist(), _texts(df["title"]),
        _texts(df["text"]), day.tolist(), df["helpful_vote"].tolist(), [int(v) for v in verified],
        df["user_id"].tolist(),
    ])
    return Table("reviews", *add_year_month(columns, rows))


# The products table; a missing price is 0.0 with p_has_price 0.
def _products_table(path: Path) -> Table:
    df = _read_csv(path, PRODUCT_COLUMNS, text=("parent_asin", "title", "store", "main_category", "categories"))
    _require(df, ("parent_asin", "average_rating"), path)
    _require_ints(df, ("rating_number",), path)

    price = df["price"].astype(float)
    columns = [
        ("p_asin", "VARCHAR"), ("p_title", "VARCHAR"), ("p_store", "VARCHAR"), ("p_main_category", "VARCHAR"),
        ("p_category", "VARCHAR"), ("p_price", "DECIMAL"), ("p_has_price", "INTEGER"),
        ("p_avg_rating", "DECIMAL"), ("p_rating_count", "INTEGER")
    ]
    rows = _rows([
        df["parent_asin"].tolist(), _texts(df["title"]), _texts(df["store"], UNKNOWN),
        _texts(df["main_category"], UNKNOWN), [_last_category(c) for c in df["categories"].tolist()],
        price.fillna(0.0).tolist(), price.notna().astype(int).tolist(),
        df["average_rating"].astype(float).tolist(), df["rating_number"].tolist(),
    ])
    return Table("products", columns, rows)


# The reviews dataset, from <ai_root>/data/reviews.csv and products.csv.
def build_reviews(ai_root: Path) -> list[Table]:
    data = Path(ai_root) / "data"
    return [_reviews_table(data / "reviews.csv"), _products_table(data / "products.csv")]


# Raise ValueError for a repeated or odd name, or a type the engine doesn't have.
def _check_names(tables: list[Table]) -> None:
    for name, columns, _ in tables:
        bad = [n for n in [name, *(c for c, _ in columns)] if not _NAME.fullmatch(n)]
        kinds = [k for _, k in columns if k not in SQLITE_TYPES]
        if bad or kinds:
            raise ValueError(f"table {name!r}: bad names {bad} or types {kinds}")

    names = Counter(c for _, columns, _ in tables for c, _ in columns)
    repeated = sorted(c for c, n in names.items() if n > 1)
    tables_repeated = sorted(t for t, n in Counter(t.name for t in tables).items() if n > 1)
    if repeated or tables_repeated:
        raise ValueError(f"names must be unique in a dataset: columns {repeated}, tables {tables_repeated}")


# One .tbl line per row, every value followed by '|'.
# Raises ValueError if a value would break the line.
def _tbl_lines(table: Table) -> Iterator[str]:
    width = len(table.columns)
    for number, row in enumerate(table.rows, 1):
        line = "|".join(map(str, row))
        if len(row) != width or line.count("|") != width - 1 or "\n" in line or "\r" in line:
            raise ValueError(f"{table.name} row {number}: needs {width} values with no '|' or line break")
        yield line + "|\n"


# Write a file through a temp name next to it, removing the temp on failure.
def _write_via_temp(path: Path, write, *args) -> None:
    path = Path(an.refuse_ai_root(path))
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.unlink(missing_ok=True)
    try:
        write(tmp, *args)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# Write lines as UTF-8 with no newline translation.
def _write_lines(path: Path, lines: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        f.writelines(lines)


# Fill a new SQLite file with every table, typed for SQLite.
def _write_sqlite(path: Path, tables: list[Table]) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        for name, columns, rows in tables:
            col_defs = ", ".join(f'"{c}" {SQLITE_TYPES[k]}' for c, k in columns)
            conn.execute(f'CREATE TABLE "{name}" ({col_defs})')

            convert = [_TO_SQLITE[k] for _, k in columns]
            marks = ", ".join("?" * len(columns))
            conn.executemany(f'INSERT INTO "{name}" VALUES ({marks})', ([f(v) for f, v in zip(convert, row)] for row in rows))
        conn.commit()
    finally:
        conn.close()


# Write a dataset's .tbl files, tables.tsv, <ds>.sqlite, then manifest.json last.
# A missing manifest means the dataset isn't complete.
def write_dataset(ds: str, tables: list[Table], out_dir: Path) -> None:
    _check_names(tables)
    out = Path(an.refuse_ai_root(out_dir))
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").unlink(missing_ok=True)

    for table in tables:
        _write_via_temp(out / f"{table.name}.tbl", _write_lines, _tbl_lines(table))

    tsv = "".join(f"{t.name}\t{t.name}.tbl\t" + ",".join(f"{c}:{k}" for c, k in t.columns) + "\n" for t in tables)
    an.write_text_atomic(out / "tables.tsv", tsv)
    _write_via_temp(out / f"{ds}.sqlite", _write_sqlite, tables)

    manifest = {
        "dataset": ds,
        "tables": [{
            "name": t.name,
            "file": f"{t.name}.tbl",
            "rows": len(t.rows),
            "columns": [{"name": c, "type": k} for c, k in t.columns]
        } for t in tables]
    }
    an.write_text_atomic(out / "manifest.json", json.dumps(manifest, indent=2) + "\n")


# Build and write the chosen datasets, refusing any output inside ai_queries first.
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write the analyst check's datasets.")
    ap.add_argument("--datasets", default=",".join(an.DATASETS), help="comma-separated: tpch, reviews")
    ap.add_argument("--results", type=Path, default=an.RESULTS)
    ap.add_argument("--tpch-dir", type=Path, default=an.TPCH_DIR)
    ap.add_argument("--ai", type=Path, default=an.AI_ROOT, help="folder holding data/reviews.csv and products.csv")
    args = ap.parse_args(argv)

    chosen = list(dict.fromkeys(d.strip() for d in args.datasets.split(",") if d.strip()))
    if not chosen:
        ap.error("--datasets names no dataset")

    try:
        out_dirs = {ds: an.paths(args.results).dataset_dir(ds) for ds in chosen}
    except ValueError as e:
        ap.error(str(e))

    for out in out_dirs.values():
        an.refuse_ai_root(out)

    builders = {
        "tpch": lambda: build_tpch(args.tpch_dir),
        "reviews": lambda: build_reviews(args.ai)
    }

    for ds, out in out_dirs.items():
        start = time.perf_counter()
        tables = builders[ds]()
        write_dataset(ds, tables, out)
        counts = ", ".join(f"{t.name} {len(t.rows):,}" for t in tables)
        print(f"{ds}: {counts} rows in {time.perf_counter() - start:.1f} s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
