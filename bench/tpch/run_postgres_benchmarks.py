#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import psycopg

HERE = Path(__file__).resolve().parent

PG_PORT = 5434
PG_HOST = "/tmp"
PG_DATA = "bench/results/tpch_postgres/pgdata"
VECTOR_CHECK_PORT = 5433
START_COMMAND = "bash bench/tpch/pg_baseline_server.sh start"

# The server settings the baseline runs with, spelled the way SHOW prints them.
PG_SETTINGS: Dict[str, str] = {
    "shared_buffers": "16GB",
    "effective_cache_size": "48GB",
    "work_mem": "256MB",
    "maintenance_work_mem": "2GB",
    "max_parallel_workers_per_gather": "2",
    "max_parallel_workers": "8",
    "max_worker_processes": "8",
    "jit": "on",
}


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

# The settings each config changes on purpose, and the SET statements that change them.
CONFIG_SETTINGS: Dict[str, Dict[str, str]] = {"postgres": {}, "postgres_1worker": {"max_parallel_workers_per_gather": "0"}}
CONFIGS: Dict[str, List[str]] = {config: [f"SET {name} = {value}" for name, value in sets.items()] for config, sets in CONFIG_SETTINGS.items()}
MARKER_TABLE = "_baseline_load"
SESSION_FIX = ("A database or role setting (ALTER DATABASE / ALTER ROLE ... SET) overrides postgresql.conf; "
               "reset it. No latency or info file was written; per-query result files from "
               "configs that already ran may have been updated.")


# A benchmark session whose settings differ from the baseline; the message lists them.
class SettingsMismatch(Exception):
    pass


def db_name(sf: str) -> str:
    return "tpch_sf" + sf.replace(".", "_")


def no_server_message(port: int) -> str:
    return f"No Postgres on port {port}. Start it with: {START_COMMAND}"


# An autocommit connection with no prepared statements, so every run plans afresh.
def connect(dbname: str, port: int) -> psycopg.Connection:
    return psycopg.connect(host=PG_HOST, port=port, dbname=dbname, autocommit=True, prepare_threshold=None, connect_timeout=5)


# What the server says for each baseline setting.
def show_settings(conn: psycopg.Connection) -> Dict[str, str]:
    return {name: conn.execute(f"SHOW {name}").fetchone()[0] for name in PG_SETTINGS}


# One line per setting that differs from want (the baseline by default).
def settings_mismatches(shown: Dict[str, str], want: Optional[Dict[str, str]] = None) -> List[str]:
    want = PG_SETTINGS if want is None else want
    return [f"{name}: got {shown.get(name)}, want {value}" for name, value in want.items() if shown.get(name) != value]


# The refusal text: where, one line per wrong setting, then how to fix it.
def mismatch_report(where: str, bad: List[str], fix: str) -> str:
    return "\n".join([f"{where} doesn't have the baseline settings:", *(f"  {line}" for line in bad), fix])


# The settings as postgresql.conf lines, for the server script to append.
def conf_lines(port: int) -> List[str]:
    lines = ["", "# TPC-H baseline (bench/tpch/run_postgres_benchmarks.py --print-conf)"]
    for name, value in PG_SETTINGS.items():
        quoted = value if value.isdigit() else f"'{value}'"
        lines.append(f"{name} = {quoted}")
        
    lines += [f"port = {port}", f"unix_socket_directories = '{PG_HOST}'"]
    return lines


# Each .tbl line for COPY's text format, where a backslash is an escape, so I double it.
def copy_lines(tbl_path: Path) -> Iterator[str]:
    for line in common.stripped_lines(tbl_path):
        yield line.replace("\\", "\\\\") + "\n"


def _database_exists(admin: psycopg.Connection, name: str) -> bool:
    return admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone() is not None


# Row counts the last complete load wrote; empty if there is no database or marker.
def read_marker(name: str, port: int) -> Dict[str, int]:
    admin = connect("postgres", port)
    try:
        if not _database_exists(admin, name):
            return {}
    finally:
        admin.close()
        
    conn = connect(name, port)
    try:
        rows = conn.execute(f"SELECT table_name, row_count FROM {MARKER_TABLE}").fetchall()
        return {table: int(count) for table, count in rows}
    except psycopg.errors.UndefinedTable:
        return {}
    finally:
        conn.close()


def _copy_table(conn: psycopg.Connection, table: str, tbl_path: Path) -> int:
    col_defs = ", ".join(f"{name} {typ}" for name, typ in common.typed_schema(table))
    conn.execute(f"CREATE TABLE {table} ({col_defs})")
    with conn.cursor() as cur, cur.copy(f"COPY {table} FROM STDIN (DELIMITER '|')") as copy:
        for line in copy_lines(tbl_path):
            copy.write(line)
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# Load one scale factor from scratch; returns the load time in seconds.
# The marker is dropped first and written last, so a half load is never reused.
def load_database(name: str, port: int, sf_data_dir: Path, expected: Optional[Dict[str, int]] = None) -> float:
    sf_data_dir = Path(sf_data_dir)
    if expected is None:
        expected = common.tbl_row_counts(sf_data_dir)
    missing = [t for t in common.TABLE_LOAD_ORDER if t not in expected]
    if missing:
        raise FileNotFoundError(f"missing .tbl files in {sf_data_dir}: {', '.join(missing)}")

    admin = connect("postgres", port)
    try:
        if not _database_exists(admin, name):
            admin.execute(f"CREATE DATABASE {name}")
    finally:
        admin.close()

    conn = connect(name, port)
    try:
        conn.execute(f"DROP TABLE IF EXISTS {MARKER_TABLE}")
        for table in common.TABLE_LOAD_ORDER:
            conn.execute(f"DROP TABLE IF EXISTS {table}")

        t0 = time.perf_counter()
        for table in common.TABLE_LOAD_ORDER:
            loaded = _copy_table(conn, table, sf_data_dir / f"{table}.tbl")
            if loaded != expected[table]:
                raise RuntimeError(f"{table}: loaded {loaded} rows, the .tbl file has {expected[table]}")
            
        for index, table, column in common.INDEX_DEFS_8:
            conn.execute(f"CREATE INDEX {index} ON {table} ({column})")
            
        conn.execute("VACUUM (ANALYZE)")
        with conn.transaction():
            conn.execute(f"CREATE TABLE {MARKER_TABLE} (table_name VARCHAR, row_count BIGINT)")
            with conn.cursor() as cur:
                cur.executemany(f"INSERT INTO {MARKER_TABLE} VALUES (%s, %s)", [(t, expected[t]) for t in common.TABLE_LOAD_ORDER])
        return time.perf_counter() - t0
    finally:
        conn.close()


# Reuse the database when its marker matches the data, else load it.
# Returns the load time in seconds, or None when reused.
def ensure_database(name: str, port: int, sf_data_dir: Path, reload: bool) -> Optional[float]:
    return common.ensure_loaded(
        f"database {name}", sf_data_dir, reload, lambda: read_marker(name, port),
        lambda expected: load_database(name, port, sf_data_dir, expected),
        detail=" (COPY, 8 indexes, VACUUM ANALYZE)"
    )


# Run the 8 queries under one config, after checking the session's own settings.
# Returns the latency rows and the settings it ran with; raises SettingsMismatch first.
def run_config(name: str, port: int, sf: str, config: str, runs: int, output_dir: Path, warmup: int = 1) -> Tuple[List[dict], Dict[str, str]]:
    conn = connect(name, port)
    try:
        for statement in CONFIGS[config]:
            conn.execute(statement)
        shown = show_settings(conn)
        bad = settings_mismatches(shown, {**PG_SETTINGS, **CONFIG_SETTINGS[config]})
        if bad:
            raise SettingsMismatch(mismatch_report(f"Database {name} (config {config})", bad, SESSION_FIX))
        
        settings = {f"max_parallel_workers_per_gather_{config}": shown["max_parallel_workers_per_gather"]}
        rows = common.run_queries("postgres", lambda sql: conn.execute(sql).fetchall(), sf, config, runs, Path(output_dir) / "postgres_results" / config, warmup)
    finally:
        conn.close()
    return rows, settings


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TPC-H queries on Postgres for the ShilmanDB baseline.",
        epilog=f"Needs the baseline server running: {START_COMMAND}",
    )
    common.add_baseline_args(parser, CONFIGS)
    parser.add_argument("--port", type=int, default=PG_PORT, help=f"the server's port (default: {PG_PORT})")
    parser.add_argument("--reload", action="store_true", help="reload the databases even if they match")
    parser.add_argument("--print-conf", action="store_true", help="print the postgresql.conf lines for the baseline server and exit")
    return parser.parse_args(argv)


# Connect and check the settings. Returns the server's version() and settings, or None after saying why.
def check_server(port: int) -> Optional[Tuple[str, Dict[str, str]]]:
    try:
        conn = connect("postgres", port)
    except psycopg.OperationalError as e:
        print(no_server_message(port))
        detail = str(e).strip().splitlines()
        if detail:
            print(f"  ({detail[0]})")
        return None
    
    try:
        shown = show_settings(conn)
        bad = settings_mismatches(shown)
        if bad:
            print(mismatch_report(f"Postgres on port {port}", bad, f"Fix {PG_DATA}/postgresql.conf (see --print-conf) and restart the server."))
            return None
        version = conn.execute("SELECT version()").fetchone()[0]
        
    finally:
        conn.close()
    return version, shown


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.port == VECTOR_CHECK_PORT:
        print(f"Error: port {VECTOR_CHECK_PORT} is the vector-search server; the baseline uses {PG_PORT}.")
        return 1
    
    if args.print_conf:
        print("\n".join(conf_lines(args.port)))
        return 0
    
    resolved = common.resolve_run_args(args, CONFIGS)
    if resolved is None:
        return 1
    scale_factors, configs = resolved

    server = check_server(args.port)
    if server is None:
        return 2
    version, shown = server
    print("Postgres TPC-H Benchmark")
    print(f"  Server: {version} (port {args.port})")
    print(f"  Scale factors: {scale_factors}  Configs: {configs}  Warmup: {args.warmup}  Runs per query: {args.runs}")

    info: Dict[str, str] = {"postgres_version": version, **shown}
    try:
        return common.run_all(
            "postgres", args.data_dir, args.output_dir, scale_factors, configs, info,
            ensure=lambda sf, sf_data_dir: ensure_database(db_name(sf), args.port, sf_data_dir, args.reload),
            run=lambda sf, config: run_config(db_name(sf), args.port, sf, config, args.runs, args.output_dir, args.warmup),
        )
    except SettingsMismatch as e:
        print(e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
