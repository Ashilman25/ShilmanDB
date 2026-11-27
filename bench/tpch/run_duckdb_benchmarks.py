#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Sequence, Tuple

import duckdb

HERE = Path(__file__).resolve().parent
DUCKDB_VERSION = "2.0.0"
DUCKDB_PIN = f"duckdb=={DUCKDB_VERSION}"
# Stands in for NULL in the CSV reader; never occurs in TPC-H data, so "" stays "".
NULL_MARKER = "<<shilmandb-null>>"


# Loads baseline_common by path; its own load_sibling can't be used before it is loaded.
def _load_common() -> ModuleType:
    name = "baseline_common"
    if name in sys.modules:
        return sys.modules[name]
    
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[name]
        raise
    return module


common = _load_common()

CONFIGS: Dict[str, List[str]] = {
    "duckdb": [],
    "duckdb_1thread": ["SET threads = 1"],
    # ShilmanDB's SF=100 memory: a 16 GiB pool plus a 4 GiB limit.
    # Runs only when named.
    "duckdb_20g": ["SET memory_limit = '20GiB'"],
}
DEFAULT_CONFIGS: List[str] = ["duckdb", "duckdb_1thread"]
DEFAULT_DB_DIR = common.DEFAULT_OUTPUT_DIR / "tpch_duckdb"
MARKER_TABLE = "_baseline_load"


def db_path_for(db_dir: Path, sf: str) -> Path:
    return Path(db_dir) / f"tpch_sf{sf}.duckdb"


def _sql_string(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


# Row counts the last complete load wrote; empty if there is no marker.
def read_marker(db_path: Path) -> Dict[str, int]:
    if not db_path.exists():
        return {}
    try:
        con = duckdb.connect(str(db_path), read_only=True)
    except duckdb.Error:
        return {}
    try:
        rows = con.execute(f"SELECT table_name, row_count FROM {MARKER_TABLE}").fetchall()
        return {name: int(count) for name, count in rows}
    except duckdb.Error:
        return {}
    finally:
        con.close()


def _remove_db(db_path: Path) -> None:
    for path in (db_path, Path(f"{db_path}.wal")):
        if path.exists():
            path.unlink()


# Load one table with DuckDB's CSV reader. The trailing "|" reads into a dummy column
# that is never inserted.
def _load_table(con: duckdb.DuckDBPyConnection, table: str, tbl_path: Path) -> int:
    schema = common.typed_schema(table)
    col_defs = ", ".join(f"{name} {typ}" for name, typ in schema)
    con.execute(f"CREATE TABLE {table} ({col_defs})")

    reader_cols = ", ".join(f"{_sql_string(name)}: {_sql_string(typ)}" for name, typ in schema)
    reader_cols += ", '_trailing_pipe': 'VARCHAR'"
    names = ", ".join(name for name, _ in schema)
    con.execute(
        f"INSERT INTO {table} SELECT {names} FROM read_csv({_sql_string(str(tbl_path))}, "
        f"delim = '|', header = false, quote = '', escape = '', nullstr = {_sql_string(NULL_MARKER)}, "
        f"auto_detect = false, "
        f"columns = {{{reader_cols}}})"
    )
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# Build the database file from scratch; returns the load time in seconds.
# The marker table is written last, so a load that stops halfway is never reused.
def load_database(db_path: Path, sf_data_dir: Path, expected: Optional[Dict[str, int]] = None) -> float:
    db_path = Path(db_path)
    sf_data_dir = Path(sf_data_dir)
    if expected is None:
        expected = common.tbl_row_counts(sf_data_dir)
        
    missing = [t for t in common.TABLE_LOAD_ORDER if t not in expected]
    if missing:
        raise FileNotFoundError(f"missing .tbl files in {sf_data_dir}: {', '.join(missing)}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_db(db_path)
    t0 = time.perf_counter()
    con = duckdb.connect(str(db_path))
    try:
        for table in common.TABLE_LOAD_ORDER:
            loaded = _load_table(con, table, sf_data_dir / f"{table}.tbl")
            if loaded != expected[table]:
                raise RuntimeError(f"{table}: loaded {loaded} rows, the .tbl file has {expected[table]}")
        con.execute(f"CREATE TABLE {MARKER_TABLE} (table_name VARCHAR, row_count BIGINT)")
        con.executemany(f"INSERT INTO {MARKER_TABLE} VALUES (?, ?)", [[t, expected[t]] for t in common.TABLE_LOAD_ORDER])
        con.execute("CHECKPOINT")
    finally:
        con.close()
    return time.perf_counter() - t0


# Reuse the file when its marker matches the data, else load it.
# Returns the load time in seconds, or None when reused.
def ensure_database(db_path: Path, sf_data_dir: Path, reload: bool) -> Optional[float]:
    return common.ensure_loaded(str(db_path), sf_data_dir, reload, lambda: read_marker(db_path), lambda expected: load_database(db_path, sf_data_dir, expected))


# Run the 8 queries under one config.
# Returns the latency rows and the settings it ran with.
def run_config(db_path: Path, sf: str, config: str, runs: int, output_dir: Path, warmup: int = 1) -> Tuple[List[dict], Dict[str, str]]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        for statement in CONFIGS[config]:
            con.execute(statement)
            
        settings = {
            f"threads_{config}": str(con.execute("SELECT current_setting('threads')").fetchone()[0]),
            f"memory_limit_{config}": str(con.execute("SELECT current_setting('memory_limit')").fetchone()[0]),
        }
        rows = common.run_queries("duckdb", lambda sql: con.execute(sql).fetchall(), sf, config, runs, Path(output_dir) / "duckdb_results" / config, warmup)
    finally:
        con.close()
    return rows, settings


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TPC-H queries on DuckDB for the ShilmanDB baseline.",
        epilog=f"Needs {DUCKDB_PIN} in .venv (.venv/bin/pip install {DUCKDB_PIN}).",
    )
    common.add_baseline_args(parser, CONFIGS, DEFAULT_CONFIGS)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR, help="where the .duckdb files live (default: bench/results/tpch_duckdb)")
    parser.add_argument("--reload", action="store_true", help="rebuild the database files even if they match")
    return parser.parse_args(argv)


# Warn on stderr when the installed DuckDB isn't the pinned version.
def warn_if_unpinned() -> bool:
    if duckdb.__version__ == DUCKDB_VERSION:
        return False
    
    print(f"Warning: DuckDB {duckdb.__version__} is installed, the baseline pins {DUCKDB_VERSION} (.venv/bin/pip install {DUCKDB_PIN}).", file=sys.stderr)
    return True


resolve_scale_factors = common.resolve_scale_factors


def resolve_configs(names: Optional[List[str]]) -> Optional[List[str]]:
    return common.resolve_configs(names, CONFIGS, DEFAULT_CONFIGS)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    resolved = common.resolve_run_args(args, CONFIGS, DEFAULT_CONFIGS)
    if resolved is None:
        return 1
    scale_factors, configs = resolved

    warn_if_unpinned()
    print("DuckDB TPC-H Benchmark")
    print(f"  DuckDB version: {duckdb.__version__}")
    print(f"  Scale factors: {scale_factors}  Configs: {configs}  Warmup: {args.warmup}  Runs per query: {args.runs}")

    info: Dict[str, str] = {"duckdb_version": duckdb.__version__}
    return common.run_all(
        "duckdb", args.data_dir, args.output_dir, scale_factors, configs, info,
        ensure=lambda sf, sf_data_dir: ensure_database(db_path_for(args.db_dir, sf), sf_data_dir, args.reload),
        run=lambda sf, config: run_config(db_path_for(args.db_dir, sf), sf, config, args.runs, args.output_dir, args.warmup),
    )


if __name__ == "__main__":
    sys.exit(main())
