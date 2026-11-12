#Load one table size into the local Postgres: table, B-trees, HNSW index, stats and prewarm
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from pgvector.psycopg import register_vector


# Load vc_common from next to this file, reusing it if it's already loaded.
def _load_common():
    path = Path(__file__).resolve().parent / "vc_common.py"
    loaded = sys.modules.get("vc_common")
    if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() == path:
        return loaded

    spec = importlib.util.spec_from_file_location("vc_common", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"can't load {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["vc_common"] = module
    spec.loader.exec_module(module)
    return module


vc = _load_common()

EXPECTED_SETTINGS = {
    "shared_buffers": "8GB",
    "effective_cache_size": "32GB",
    "maintenance_work_mem": "8GB",
    "max_parallel_maintenance_workers": "8",
    "max_parallel_workers_per_gather": "0",
    "jit": "off",
    "statement_timeout": "10s",
    "work_mem": "64MB",
}
MIN_EXT = (0, 8, 0)
TABLE_COLUMNS = ["id", "product_id", "rating", "review_date", "verified", "category", "vec_row"]
BTREE_COLUMNS = ("review_date", "product_id", "rating", "category", "verified")
COPY_TYPES = ["int4", "int4", "int2", "date", "bool", "text", "vector"]
INT4, INT2 = np.iinfo(np.int32), np.iinfo(np.int16)


# The Postgres table for a size.
def table_name(size: str) -> str:
    return f"reviews_{vc.check_size(size)}"


# The primary key, the five filter B-trees, then the HNSW index.
def _index_names(size: str) -> list[str]:
    t = table_name(size)
    return [f"{t}_pkey", *(f"{t}_{c}_idx" for c in BTREE_COLUMNS), f"{t}_hnsw_idx"]


# Open an autocommit connection with no prepared statements.
# register adds the vector type, which must already exist.
def connect(port: int = vc.PG_PORT, host: str = vc.PG_HOST, dbname: str = vc.PG_DB, register: bool = True):
    conn = psycopg.connect(host=host, port=port, dbname=dbname, autocommit=True, prepare_threshold=None)
    if register:
        register_vector(conn)
    return conn


# What the server says for each setting I check.
def show_settings(conn) -> dict[str, str]:
    return {name: conn.execute(f"SHOW {name}").fetchone()[0] for name in EXPECTED_SETTINGS}


# One line per setting that differs from what I want.
def check_server_settings(shown: dict[str, str]) -> list[str]:
    return [f"{name}: got {shown.get(name)}, want {want}" for name, want in EXPECTED_SETTINGS.items() if shown.get(name) != want]


# Server version and pgvector's version: installed if it is, else what would install.
def versions(conn) -> dict:
    server = conn.execute("SHOW server_version").fetchone()[0]
    row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    if row is None:
        row = conn.execute("SELECT default_version FROM pg_available_extensions WHERE name = 'vector'").fetchone()
        
    return {"server_version": server, "extversion": None if row is None else row[0]}


# True for pgvector 0.8.0 or newer, compared as numbers.
def ext_ok(extversion: str | None) -> bool:
    found = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", extversion or "")
    return found is not None and tuple(int(p or 0) for p in found.groups()) >= MIN_EXT


# Session settings for the load: index builds outlast the 10 s timeout.
def load_session_sql() -> list[str]:
    return ["SET statement_timeout = 0"]


# Extensions, then a fresh empty table.
def create_sql(size: str) -> list[str]:
    t = table_name(size)
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        "CREATE EXTENSION IF NOT EXISTS pg_prewarm",
        f"DROP TABLE IF EXISTS {t}",
        f"CREATE TABLE {t} (id int PRIMARY KEY, product_id int NOT NULL, rating smallint NOT NULL, "
        f"review_date date NOT NULL, verified boolean NOT NULL, category text NOT NULL, embedding vector({vc.DIM}) NOT NULL)",
    ]


# One B-tree per filter column, then the HNSW index last.
def index_sql(size: str) -> list[str]:
    t = table_name(size)
    btrees = [f"CREATE INDEX {t}_{c}_idx ON {t} ({c})" for c in BTREE_COLUMNS]
    hnsw = f"CREATE INDEX {t}_hnsw_idx ON {t} USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    return [*btrees, hnsw]


# Binary COPY of every row in id order; a row's embedding is vectors[vec_row].
def copy_rows(conn, size: str, table: pd.DataFrame, vectors: np.ndarray) -> None:
    t = table.iloc[np.argsort(table["id"].to_numpy(), kind="stable")]
    rows = zip(*(t[c].tolist() for c in TABLE_COLUMNS))
    sql = (f"COPY {table_name(size)} (id, product_id, rating, review_date, verified, category, embedding) FROM STDIN WITH (FORMAT BINARY)")
    
    with conn.cursor() as cur, cur.copy(sql) as copy:
        copy.set_types(COPY_TYPES)
        for i, product_id, rating, day, verified, category, vec_row in rows:
            copy.write_row((i, product_id, rating, day, verified, category, vectors[vec_row]))


# Read the table and every index on it into shared buffers.
def prewarm(conn, size: str) -> None:
    for name in [table_name(size), *_index_names(size)]:
        conn.execute(f"SELECT pg_prewarm('{name}')")


# Bytes on disk: table with TOAST, the HNSW index, the primary key plus B-trees.
def relation_sizes(conn, size: str) -> dict[str, int]:
    t, names = table_name(size), _index_names(size)
    btrees = " + ".join(f"pg_relation_size('{n}')" for n in names[:-1])
    row = conn.execute(f"SELECT pg_table_size('{t}'), pg_relation_size('{names[-1]}'), {btrees}").fetchone()
    return {
        "table_bytes": int(row[0]), 
        "hnsw_bytes": int(row[1]), 
        "btree_bytes": int(row[2])
    }


# A column of whole numbers within limits, or ValueError naming the file and column.
def _ints(df: pd.DataFrame, column: str, path: Path, limits: np.iinfo | None = None) -> np.ndarray:
    try:
        values = df[column].astype(np.int64).to_numpy()
    except (ValueError, OverflowError):
        raise ValueError(f"{path}: {column} must be whole numbers") from None
    
    if limits is not None and ((values < limits.min) | (values > limits.max)).any():
        raise ValueError(f"{path}: {column} must be between {limits.min} and {limits.max}")
    return values


# Each YYYY-MM-DD string as a date; anything else raises naming the file.
def _dates(values: pd.Series, path: Path) -> list[dt.date]:
    parsed = {}
    for s in set(values):
        try:
            d = dt.date.fromisoformat(s)
        except ValueError:
            d = None
            
        if d is None or d.isoformat() != s:
            raise ValueError(f"{path}: review_date must be YYYY-MM-DD, got {s!r}")
        parsed[s] = d
    return [parsed[s] for s in values]


# Read a table CSV into typed columns; id must be 0..n-1 in file order.
def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"{path}: not found")
    
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (ValueError, OSError) as e:
        raise ValueError(f"{path}: can't read it as a CSV ({e})") from None
    
    missing = [c for c in TABLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {', '.join(missing)}")
    if df.empty:
        raise ValueError(f"{path}: no rows")

    ids = _ints(df, "id", path, INT4)
    if not np.array_equal(ids, np.arange(len(df))):
        raise ValueError(f"{path}: id must be 0..n-1 in file order")
    bad = ~df["verified"].isin(["true", "false"])
    if bad.any():
        raise ValueError(f"{path}: verified must be true or false, got {df['verified'][bad].iloc[0]!r}")

    return pd.DataFrame({
        "id": ids,
        "product_id": _ints(df, "product_id", path, INT4),
        "rating": _ints(df, "rating", path, INT2),
        "review_date": _dates(df["review_date"], path),
        "verified": (df["verified"] == "true").to_numpy(),
        "category": df["category"].to_numpy(dtype=object),
        "vec_row": _ints(df, "vec_row", path),
    })


# Base vectors as float32 rows of width 384, every value finite.
def _load_base(path: Path) -> np.ndarray:
    if not path.exists():
        raise ValueError(f"{path}: not found")
    
    try:
        vectors = np.asarray(np.load(path), dtype=np.float32)
    except (ValueError, TypeError, OSError, EOFError) as e:
        raise ValueError(f"{path}: can't read it as a numpy array ({e})") from None
    
    if vectors.ndim != 2 or vectors.shape[1] != vc.DIM:
        raise ValueError(f"{path}: shape {vectors.shape}, want rows x {vc.DIM}")
    if not np.isfinite(vectors).all():
        raise ValueError(f"{path}: every value must be finite")
    return vectors


# The table and its base vectors, checked against each other.
def load_inputs(size: str, p) -> tuple[pd.DataFrame, np.ndarray]:
    table_path = p.data / f"table_{size}.csv"
    base_path = p.ai / "embeddings" / "test.npy" if size == "70k" else p.embeddings / "slice.npy"
    
    table = read_table(table_path)
    vectors = _load_base(base_path)
    
    if not np.array_equal(np.unique(table["vec_row"].to_numpy()), np.arange(len(vectors))):
        raise ValueError(f"{table_path}: distinct vec_row values must be exactly 0..{len(vectors) - 1}, one per row of {base_path}")
    return table, vectors


# Run each statement and return the seconds they took together.
def _timed(conn, stmts: list[str]) -> float:
    start = time.perf_counter()
    for s in stmts:
        conn.execute(s)
    return time.perf_counter() - start


# Check settings and pgvector, drop the old JSON, then load, index, vacuum, analyze and prewarm.
# Returns what goes into pg_load.json, or None after printing why it stopped.
def _load(conn, size: str, table: pd.DataFrame, vectors: np.ndarray, json_path: Path) -> dict | None:
    shown = show_settings(conn)
    bad = check_server_settings(shown)
    for line in bad:
        print(f"error: server setting {line}", file=sys.stderr)
    if bad:
        return None

    ver = versions(conn)
    if not ext_ok(ver["extversion"]):
        print(f"error: pgvector must be 0.8.0 or newer, got {ver['extversion']}", file=sys.stderr)
        return None

    # From here on the table changes, so an old pg_load.json would describe the wrong one.
    json_path.unlink(missing_ok=True)
    t = table_name(size)
    for s in load_session_sql() + create_sql(size):
        conn.execute(s)
    register_vector(conn)
    ver = versions(conn)

    start = time.perf_counter()
    copy_rows(conn, size, table, vectors)
    copy_s = time.perf_counter() - start
    print(f"copy: {len(table):,} rows in {copy_s:.1f} s")

    rows = int(conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0])
    if rows != len(table):
        print(f"error: {t} has {rows:,} rows after COPY, want {len(table):,}", file=sys.stderr)
        return None

    *btrees, hnsw = index_sql(size)
    btree_s = _timed(conn, btrees)
    print(f"b-trees: {btree_s:.1f} s")
    hnsw_build_s = _timed(conn, [hnsw])
    print(f"hnsw: {hnsw_build_s:.1f} s")
    # Vacuum now, so autovacuum doesn't change the heap while the grid is timing.
    vacuum_analyze_s = _timed(conn, [f"VACUUM (ANALYZE) {t}"])
    print(f"vacuum analyze: {vacuum_analyze_s:.1f} s")
    prewarm(conn, size)

    return {
        "server_version": ver["server_version"], "extversion": ver["extversion"], "settings": shown,
        "rows": rows, "copy_s": copy_s, "btree_s": btree_s, "hnsw_build_s": hnsw_build_s, "vacuum_analyze_s": vacuum_analyze_s,
        **relation_sizes(conn, size),
    }


# Write via a temp file and rename, so a stopped run leaves no half file.
def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# Parse the command line, load the table and write pg_load.json. Returns the exit code.
def main(argv: list[str] | None = None, connect: Callable = connect) -> int:
    parser = argparse.ArgumentParser(description="Load one table size into Postgres for the vector search check.")
    parser.add_argument("--size", required=True, choices=vc.SIZES)
    parser.add_argument("--results", type=Path, default=vc.RESULTS)
    parser.add_argument("--ai", type=Path, default=vc.AI_ROOT)
    args = parser.parse_args(argv)

    p = vc.paths(args.results, args.ai)
    ai, out = p.ai.resolve(), p.size_dir(args.size).resolve()
    if out == ai or ai in out.parents:
        print(f"error: won't write inside the ai_queries folder {p.ai}", file=sys.stderr)
        return 1

    start = time.monotonic()
    try:
        table, vectors = load_inputs(args.size, p)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    try:
        conn = connect(register=False)
    except psycopg.OperationalError as e:
        print(f"error: can't connect to Postgres on port {vc.PG_PORT}: {e}", file=sys.stderr)
        return 1
    json_path = out / "pg_load.json"
    try:
        info = _load(conn, args.size, table, vectors, json_path)
    finally:
        conn.close()
    if info is None:
        return 1

    out.mkdir(parents=True, exist_ok=True)
    _write_atomic(json_path, json.dumps(info, indent=2) + "\n")
    print(f"wrote {json_path}; done in {time.monotonic() - start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
