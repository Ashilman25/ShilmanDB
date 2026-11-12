"""Tests for bench/vector_check/pg_vector_load.py. A fake connection stands in for Postgres."""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC_PATH = Path(__file__).resolve().parents[1] / "pg_vector_load.py"
spec = importlib.util.spec_from_file_location("vc_pg_load", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["vc_pg_load"] = mod
spec.loader.exec_module(mod)

DIM = 384
CASES = "Cases, Holsters & Sleeves"
REGISTER = "<register_vector>"
DDL_WORDS = ("SET ", "CREATE", "DROP", "COPY", "VACUUM", "ANALYZE", "SELECT pg_prewarm")


# One query result: fetchone gives the row I was built with.
class FakeResult:
    # Keep the row to hand back.
    def __init__(self, row):
        self.row = row

    # The single row, or None.
    def fetchone(self):
        return self.row


# A COPY in progress: keeps its SQL, types and every row written.
class FakeCopy:
    # Start empty, with the COPY SQL.
    def __init__(self, sql: str):
        self.sql, self.types, self.rows = sql, None, []

    # Enter the with block.
    def __enter__(self):
        return self

    # Leave the with block without hiding errors.
    def __exit__(self, *exc):
        return False

    # Remember the column types.
    def set_types(self, types):
        self.types = list(types)

    # Remember one row.
    def write_row(self, row):
        self.rows.append(tuple(row))


# A cursor that only knows COPY; it logs the SQL on its connection.
class FakeCursor:
    # Belongs to one fake connection.
    def __init__(self, conn):
        self.conn = conn

    # Enter the with block.
    def __enter__(self):
        return self

    # Leave the with block without hiding errors.
    def __exit__(self, *exc):
        return False

    # Start a fake COPY and log its SQL.
    def copy(self, sql: str) -> FakeCopy:
        self.conn.log.append(sql)
        c = FakeCopy(sql)
        self.conn.copies.append(c)
        return c


# Records every SQL string and answers the few queries the loader reads.
class FakeConn:
    # Settings, extension versions, a wrong row count, a file to watch, a statement that fails.
    # CREATE EXTENSION installs installs, or the available version when installs is None.
    def __init__(self, settings=None, installed=None, available="0.8.7", installs=None, count=None,
                 watch: Path | None = None, fail_on: str | None = None):
        self.settings = dict(mod.EXPECTED_SETTINGS if settings is None else settings)
        self.installed, self.available, self.installs, self.count = installed, available, installs, count
        self.watch, self.fail_on = watch, fail_on
        self.log, self.copies, self.seen, self.closed = [], [], [], False

    # Log the SQL and answer it the way Postgres would here.
    def execute(self, sql: str, params=None) -> FakeResult:
        self.log.append(sql)
        if self.watch is not None:
            self.seen.append((sql, self.watch.exists()))
        if self.fail_on is not None and sql.startswith(self.fail_on):
            raise RuntimeError(f"failed: {sql}")
        if sql == "SHOW server_version":
            return FakeResult(("17.11 (Homebrew)",))
        if sql.startswith("SHOW "):
            return FakeResult((self.settings[sql[5:]],))
        if "FROM pg_extension" in sql:
            return FakeResult(None if self.installed is None else (self.installed,))
        if "FROM pg_available_extensions" in sql:
            return FakeResult(None if self.available is None else (self.available,))
        if sql == "CREATE EXTENSION IF NOT EXISTS vector" and self.installed is None:
            self.installed = self.installs or self.available
        if sql.startswith("SELECT count(*)"):
            copied = sum(len(c.rows) for c in self.copies)
            return FakeResult((copied if self.count is None else self.count,))
        if sql.startswith("SELECT pg_table_size"):
            return FakeResult((1000, 500, 300))
        return FakeResult(None)

    # A cursor for COPY.
    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    # Mark the connection closed.
    def close(self):
        self.closed = True


# Seeded unit float32 rows.
def unit_rows(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# The table CSV as strings; rows 1, 2 and 6 share vector 1.
def table_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "id": [str(i) for i in range(8)],
        "product_id": ["7", "3", "7", "9", "1", "3", "2", "7"],
        "rating": ["5", "1", "4", "3", "2", "5", "4", "1"],
        "review_date": ["2019-12-31", "2020-02-29", "2021-01-01", "2020-06-15",
                        "2018-03-03", "2022-11-11", "2020-01-01", "2019-07-04"],
        "verified": ["true", "false", "true", "true", "false", "true", "true", "false"],
        "category": [CASES, "", "Chargers", "", CASES, "Accessories", "", "Chargers"],
        "vec_row": ["0", "1", "1", "2", "3", "4", "1", "0"],
        "text_key": [f"k{i}" for i in range(8)],
    })


# The same table with the types the loader parses to.
def typed_frame() -> pd.DataFrame:
    f = table_frame()
    return pd.DataFrame({
        "id": f["id"].astype(int), "product_id": f["product_id"].astype(int), "rating": f["rating"].astype(int),
        "review_date": [dt.date.fromisoformat(s) for s in f["review_date"]],
        "verified": f["verified"] == "true", "category": f["category"], "vec_row": f["vec_row"].astype(int),
    })


# A results and ai tree with the table and the base vectors for one size.
def write_world(tmp_path: Path, size: str = "70k", frame: pd.DataFrame | None = None,
                vectors: np.ndarray | None = None) -> dict:
    results, ai = tmp_path / "results", tmp_path / "ai"
    for d in (results / "data", results / "embeddings", ai / "embeddings"):
        d.mkdir(parents=True, exist_ok=True)
    frame = table_frame() if frame is None else frame
    vectors = unit_rows(5) if vectors is None else vectors
    table_path = results / "data" / f"table_{size}.csv"
    frame.to_csv(table_path, index=False)
    base_path = ai / "embeddings" / "test.npy" if size == "70k" else results / "embeddings" / "slice.npy"
    np.save(base_path, vectors)
    return {"results": results, "ai": ai, "table": table_path, "base": base_path, "vectors": vectors, "size": size}


# Run main against a fake connection; returns the exit code and connect calls.
# With fake None, any connect call fails the test.
def run_main(world: dict, fake: FakeConn | None) -> tuple[int, list]:
    calls = []

    # Stand-in for connect that records its arguments.
    def connect(**kw):
        calls.append(kw)
        if fake is None:
            raise AssertionError("connect must not be called")
        return fake

    argv = ["--size", world["size"], "--results", str(world["results"]), "--ai", str(world["ai"])]
    return mod.main(argv, connect=connect), calls


# Replace register_vector with one that logs itself on the fake connection.
@pytest.fixture(autouse=True)
def fake_register(monkeypatch):
    monkeypatch.setattr(mod, "register_vector", lambda conn: conn.log.append(REGISTER))


# Index of the first log line that starts with prefix.
def first(log: list[str], prefix: str) -> int:
    return next(i for i, s in enumerate(log) if s.startswith(prefix))


# Index of the last log line that starts with prefix.
def last(log: list[str], prefix: str) -> int:
    return max(i for i, s in enumerate(log) if s.startswith(prefix))


# True if any logged SQL changes the database or the session.
def any_ddl(fake: FakeConn) -> bool:
    return bool(fake.copies) or any(s.startswith(DDL_WORDS) for s in fake.log)


# One message per wrong setting, naming what I got and what I want.
def test_settings_mismatch_messages():
    bad = dict(mod.EXPECTED_SETTINGS, work_mem="4MB")
    assert mod.check_server_settings(bad) == ["work_mem: got 4MB, want 64MB"]


# All matching gives no messages; a setting not shown counts as wrong.
def test_settings_all_match_and_missing_key():
    assert mod.check_server_settings(dict(mod.EXPECTED_SETTINGS)) == []
    shown = {k: v for k, v in mod.EXPECTED_SETTINGS.items() if k != "jit"}
    assert mod.check_server_settings(shown) == ["jit: got None, want off"]


# The eight server settings, exactly.
def test_expected_settings_values():
    assert mod.EXPECTED_SETTINGS == {
        "shared_buffers": "8GB", "effective_cache_size": "32GB", "maintenance_work_mem": "8GB",
        "max_parallel_maintenance_workers": "8", "max_parallel_workers_per_gather": "0",
        "jit": "off", "statement_timeout": "10s", "work_mem": "64MB",
    }


# show_settings runs one SHOW per setting and returns what the server says.
def test_show_settings_asks_each_name():
    fake = FakeConn()
    assert mod.show_settings(fake) == mod.EXPECTED_SETTINGS
    assert fake.log == [f"SHOW {name}" for name in mod.EXPECTED_SETTINGS]


# Table names per size; other sizes are refused.
def test_table_name():
    assert mod.table_name("70k") == "reviews_70k"
    assert mod.table_name("500k") == "reviews_500k"
    with pytest.raises(ValueError):
        mod.table_name("1m")


# The HNSW index comes last, with cosine ops, m 16 and ef_construction 64.
def test_index_sql_hnsw_clause():
    assert mod.index_sql("70k")[-1] == "CREATE INDEX reviews_70k_hnsw_idx ON reviews_70k USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"


# Five B-trees come first, one per filter column, in a fixed order.
def test_index_sql_btrees_first():
    assert mod.index_sql("500k")[:-1] == [
        "CREATE INDEX reviews_500k_review_date_idx ON reviews_500k (review_date)",
        "CREATE INDEX reviews_500k_product_id_idx ON reviews_500k (product_id)",
        "CREATE INDEX reviews_500k_rating_idx ON reviews_500k (rating)",
        "CREATE INDEX reviews_500k_category_idx ON reviews_500k (category)",
        "CREATE INDEX reviews_500k_verified_idx ON reviews_500k (verified)",
    ]


# Extensions, drop and create, word for word; the session turns the timeout off.
def test_create_sql_exact():
    assert mod.create_sql("70k") == [
        "CREATE EXTENSION IF NOT EXISTS vector",
        "CREATE EXTENSION IF NOT EXISTS pg_prewarm",
        "DROP TABLE IF EXISTS reviews_70k",
        "CREATE TABLE reviews_70k (id int PRIMARY KEY, product_id int NOT NULL, rating smallint NOT NULL, "
        "review_date date NOT NULL, verified boolean NOT NULL, category text NOT NULL, embedding vector(384) NOT NULL)",
    ]
    assert mod.load_session_sql() == ["SET statement_timeout = 0"]


# 0.8.0 and up pass, compared as numbers.
def test_ext_ok():
    assert mod.ext_ok("0.8.7") and mod.ext_ok("0.10.0") and not mod.ext_ok("0.7.9")


# Two-part versions work; None, empty and junk fail.
def test_ext_ok_edge_cases():
    assert mod.ext_ok("0.8.0") and mod.ext_ok("0.8") and mod.ext_ok("1.0.0")
    assert not mod.ext_ok(None) and not mod.ext_ok("") and not mod.ext_ok("abc")


# An installed extension's version wins over the available one.
def test_versions_installed_wins_over_available():
    fake = FakeConn(installed="0.7.4", available="0.8.7")
    assert mod.versions(fake) == {"server_version": "17.11 (Homebrew)", "extversion": "0.7.4"}


# Not installed yet: the version CREATE EXTENSION would install.
def test_versions_falls_back_to_available():
    fake = FakeConn(installed=None, available="0.8.7")
    assert mod.versions(fake)["extversion"] == "0.8.7"
    assert any("pg_available_extensions" in s for s in fake.log)


# Neither installed nor available gives None.
def test_versions_none_when_not_available():
    assert mod.versions(FakeConn(installed=None, available=None))["extversion"] is None


# connect asks for autocommit and no prepared statements; register is optional.
def test_connect_options(monkeypatch):
    seen, registered = {}, []

    # Stand-in for psycopg.connect.
    def fake_connect(**kw):
        seen.update(kw)
        return "conn"

    monkeypatch.setattr(mod.psycopg, "connect", fake_connect)
    monkeypatch.setattr(mod, "register_vector", registered.append)
    assert mod.connect() == "conn"
    assert seen == {"host": "/tmp", "port": 5433, "dbname": "vector_check", "autocommit": True, "prepare_threshold": None}
    assert registered == ["conn"]
    mod.connect(register=False)
    assert registered == ["conn"]


# read_table parses each column; an empty category stays an empty string.
def test_read_table_types_and_empty_category(tmp_path):
    world = write_world(tmp_path)
    t = mod.read_table(world["table"])
    assert list(t.columns) == mod.TABLE_COLUMNS
    assert t["id"].tolist() == list(range(8))
    assert t["review_date"].tolist() == typed_frame()["review_date"].tolist()
    assert t["verified"].tolist() == [True, False, True, True, False, True, True, False]
    assert t["category"].tolist() == table_frame()["category"].tolist()
    assert t["vec_row"].tolist() == [0, 1, 1, 2, 3, 4, 1, 0]


# Binary COPY with fixed types, rows in id order, each row's own vector.
def test_copy_rows_sql_types_and_rows():
    vectors, typed = unit_rows(5), typed_frame()
    fake = FakeConn()
    mod.copy_rows(fake, "70k", typed.iloc[[5, 2, 7, 0, 3, 1, 6, 4]], vectors)

    (c,) = fake.copies
    assert c.sql == "COPY reviews_70k (id, product_id, rating, review_date, verified, category, embedding) FROM STDIN WITH (FORMAT BINARY)"
    assert c.types == ["int4", "int4", "int2", "date", "bool", "text", "vector"]
    assert [r[0] for r in c.rows] == list(range(8))
    for r, (_, want) in zip(c.rows, typed.iterrows()):
        assert r[1:6] == (want.product_id, want.rating, want.review_date, want.verified, want.category)
        assert [type(v) for v in r[:6]] == [int, int, int, dt.date, bool, str]
        np.testing.assert_array_equal(r[6], vectors[want.vec_row])
        assert r[6].dtype == np.float32
    assert [r[5] for r in c.rows].count("") == 3


# Prewarm reads the table, its primary key, five B-trees and the HNSW index.
def test_prewarm_covers_table_and_seven_indexes():
    fake = FakeConn()
    mod.prewarm(fake, "70k")
    names = ["reviews_70k", "reviews_70k_pkey", "reviews_70k_review_date_idx", "reviews_70k_product_id_idx",
             "reviews_70k_rating_idx", "reviews_70k_category_idx", "reviews_70k_verified_idx", "reviews_70k_hnsw_idx"]
    assert fake.log == [f"SELECT pg_prewarm('{n}')" for n in names]


# The 10 s timeout is off before the COPY and the index builds.
def test_load_disables_timeout_before_copy_and_index(tmp_path):
    fake = FakeConn()
    code, _ = run_main(write_world(tmp_path), fake)
    assert code == 0
    log = fake.log
    assert first(log, "SET statement_timeout = 0") < first(log, "COPY") < first(log, "CREATE INDEX") < first(log, "VACUUM")


# The whole run in order; the vector type registers after it exists.
def test_main_runs_steps_in_order(tmp_path):
    fake = FakeConn()
    code, calls = run_main(write_world(tmp_path), fake)
    assert code == 0 and calls == [{"register": False}] and fake.closed
    log = fake.log
    order = [
        last(log, "SHOW work_mem"), first(log, "SELECT extversion"), first(log, "SET statement_timeout = 0"),
        first(log, "CREATE EXTENSION IF NOT EXISTS vector"), first(log, "DROP TABLE"), first(log, "CREATE TABLE"),
        log.index(REGISTER), first(log, "COPY"), first(log, "CREATE INDEX reviews_70k_review_date_idx"),
        first(log, "CREATE INDEX reviews_70k_verified_idx"), first(log, "CREATE INDEX reviews_70k_hnsw_idx"),
        first(log, "VACUUM (ANALYZE) reviews_70k"), first(log, "SELECT pg_prewarm"),
    ]
    assert order == sorted(order)
    assert not any(s.startswith("ANALYZE") for s in log)
    assert sum(s.startswith("SELECT pg_prewarm") for s in log) == 8


# pg_load.json has exactly the fields I want, the empty categories reach COPY.
def test_main_writes_pg_load_json(tmp_path):
    world = write_world(tmp_path)
    fake = FakeConn(installed=None, available="0.8.7", installs="0.9.1")
    code, _ = run_main(world, fake)
    assert code == 0
    out = json.loads((world["results"] / "size70k" / "pg_load.json").read_text())
    assert set(out) == {"server_version", "extversion", "settings", "rows", "copy_s", "btree_s",
                        "hnsw_build_s", "vacuum_analyze_s", "table_bytes", "hnsw_bytes", "btree_bytes"}
    assert out["extversion"] == "0.9.1" and out["server_version"] == "17.11 (Homebrew)"
    assert out["settings"] == mod.EXPECTED_SETTINGS and out["rows"] == 8
    assert (out["table_bytes"], out["hnsw_bytes"], out["btree_bytes"]) == (1000, 500, 300)
    assert all(out[k] >= 0 for k in ("copy_s", "btree_s", "hnsw_build_s", "vacuum_analyze_s"))
    (c,) = fake.copies
    assert [r[5] for r in c.rows].count("") == 3


# The version is read again after CREATE EXTENSION, so the JSON has the installed one.
def test_main_writes_installed_version_after_create(tmp_path):
    world = write_world(tmp_path)
    fake = FakeConn(installed=None, available="0.8.7", installs="0.8.8")
    assert run_main(world, fake)[0] == 0
    reads = [i for i, s in enumerate(fake.log) if s.startswith("SELECT extversion")]
    assert len(reads) == 2 and reads[1] > first(fake.log, "CREATE TABLE")
    assert json.loads((world["results"] / "size70k" / "pg_load.json").read_text())["extversion"] == "0.8.8"


# Table bytes include TOAST; B-tree bytes add the primary key and five B-trees.
def test_sizes_query_sums_pkey_and_five_btrees():
    fake = FakeConn()
    assert mod.relation_sizes(fake, "70k") == {"table_bytes": 1000, "hnsw_bytes": 500, "btree_bytes": 300}
    (sql,) = fake.log
    assert "pg_table_size('reviews_70k')" in sql and "pg_relation_size('reviews_70k_hnsw_idx')" in sql
    for name in ("pkey", "review_date_idx", "product_id_idx", "rating_idx", "category_idx", "verified_idx"):
        assert f"pg_relation_size('reviews_70k_{name}')" in sql


# Wrong settings stop the run before any SQL that changes anything.
def test_main_refuses_wrong_settings_before_ddl(tmp_path, capsys):
    fake = FakeConn(settings=dict(mod.EXPECTED_SETTINGS, shared_buffers="128MB", jit="on"))
    code, _ = run_main(write_world(tmp_path), fake)
    assert code == 1 and not any_ddl(fake) and fake.closed
    err = capsys.readouterr().err
    assert "shared_buffers: got 128MB, want 8GB" in err and "jit: got on, want off" in err


# An installed pgvector under 0.8.0 stops the run before any DDL.
def test_main_refuses_old_pgvector(tmp_path, capsys):
    fake = FakeConn(installed="0.7.4")
    code, _ = run_main(write_world(tmp_path), fake)
    assert code == 1 and not any_ddl(fake)
    assert "0.7.4" in capsys.readouterr().err


# No pgvector at all stops the run too.
def test_main_refuses_missing_pgvector(tmp_path):
    fake = FakeConn(installed=None, available=None)
    code, _ = run_main(write_world(tmp_path), fake)
    assert code == 1 and not any_ddl(fake)


# An old pg_load.json from an earlier run, as the loader would find it.
def old_json(world: dict) -> Path:
    path = world["results"] / "size70k" / "pg_load.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"rows": 1}\n')
    return path


# A table that doesn't hold every row copied fails and leaves no JSON.
def test_main_refuses_row_count_mismatch(tmp_path, capsys):
    world = write_world(tmp_path)
    path = old_json(world)
    code, _ = run_main(world, FakeConn(count=7))
    assert code == 1 and not path.exists()
    assert "7" in capsys.readouterr().err


# The old JSON goes after the settings and version checks, before the session SQL.
def test_old_json_removed_just_before_session_sql(tmp_path):
    world = write_world(tmp_path)
    path = old_json(world)
    fake = FakeConn(watch=path, count=7)
    run_main(world, fake)
    seen = dict(reversed(fake.seen))
    assert seen["SHOW work_mem"] and seen["SELECT extversion FROM pg_extension WHERE extname = 'vector'"]
    assert not seen["SET statement_timeout = 0"]


# A refused run leaves the old JSON exactly as it was.
@pytest.mark.parametrize("fake", [
    FakeConn(settings=dict(mod.EXPECTED_SETTINGS, work_mem="4MB")), FakeConn(installed="0.7.4"),
], ids=["settings", "old pgvector"])
def test_refused_run_keeps_old_json(tmp_path, fake):
    world = write_world(tmp_path)
    path = old_json(world)
    assert run_main(world, fake)[0] == 1
    assert path.read_text() == '{"rows": 1}\n'


# A build that fails part way leaves no JSON describing the old table.
def test_failed_build_removes_old_json(tmp_path):
    world = write_world(tmp_path)
    path = old_json(world)
    with pytest.raises(RuntimeError):
        run_main(world, FakeConn(fail_on="CREATE INDEX reviews_70k_hnsw_idx"))
    assert not path.exists()


# The JSON is written to a temp file next to it, then renamed into place.
def test_json_written_atomically(tmp_path, monkeypatch):
    world = write_world(tmp_path)
    old_json(world)
    moves, real_replace = [], mod.os.replace

    # Record each rename, then do it.
    def replace(src, dst):
        moves.append((Path(src), Path(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(mod.os, "replace", replace)
    assert run_main(world, FakeConn())[0] == 0
    out = world["results"] / "size70k"
    assert moves == [(out / f"pg_load.json.tmp{mod.os.getpid()}", out / "pg_load.json")]
    assert [f.name for f in out.iterdir()] == ["pg_load.json"]
    assert json.loads((out / "pg_load.json").read_text())["rows"] == 8


# A failed rename leaves neither a temp file nor a JSON.
def test_failed_rename_leaves_nothing(tmp_path, monkeypatch):
    world = write_world(tmp_path)
    old_json(world)

    # A rename that always fails.
    def replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(mod.os, "replace", replace)
    with pytest.raises(OSError):
        run_main(world, FakeConn())
    assert list((world["results"] / "size70k").iterdir()) == []


# A table with one cell replaced.
def with_cell(row: int, column: str, value: str) -> pd.DataFrame:
    f = table_frame()
    f.loc[row, column] = value
    return f


# Each case breaks one input and names the file the error must report.
BROKEN = {
    "id not 0..n-1": ("table", lambda w: write_world(w, frame=with_cell(3, "id", "9"))),
    "ids out of order": ("table", lambda w: write_world(w, frame=table_frame().iloc[[1, 0, 2, 3, 4, 5, 6, 7]])),
    "verified True": ("table", lambda w: write_world(w, frame=with_cell(2, "verified", "True"))),
    "month 13": ("table", lambda w: write_world(w, frame=with_cell(4, "review_date", "2020-13-01"))),
    "compact date": ("table", lambda w: write_world(w, frame=with_cell(4, "review_date", "20200101"))),
    "rating 4.5": ("table", lambda w: write_world(w, frame=with_cell(0, "rating", "4.5"))),
    "rating 40000": ("table", lambda w: write_world(w, frame=with_cell(0, "rating", "40000"))),
    "product_id 2^31": ("table", lambda w: write_world(w, frame=with_cell(1, "product_id", str(2**31)))),
    "product_id 20 digits": ("table", lambda w: write_world(w, frame=with_cell(1, "product_id", "9" * 20))),
    "no category": ("table", lambda w: write_world(w, frame=table_frame().drop(columns=["category"]))),
    "header only": ("table", lambda w: write_world(w, frame=table_frame().iloc[:0])),
    "empty file": ("table", lambda w: write_world(w)["table"].write_bytes(b"")),
    "no table file": ("table", lambda w: write_world(w)["table"].unlink()),
    "vec_row gap": ("table", lambda w: write_world(w, frame=with_cell(5, "vec_row", "5"), vectors=unit_rows(6))),
    "extra vector": ("table", lambda w: write_world(w, vectors=unit_rows(6))),
    "width 128": ("base", lambda w: write_world(w, vectors=unit_rows(5, dim=128))),
    "nan": ("base", lambda w: write_world(w, vectors=np.where(np.arange(DIM) == 7, np.nan, unit_rows(5)).astype(np.float32))),
    "not a numpy file": ("base", lambda w: write_world(w)["base"].write_bytes(b"not a numpy file")),
    "empty npy": ("base", lambda w: write_world(w)["base"].write_bytes(b"")),
    "no base file": ("base", lambda w: write_world(w)["base"].unlink()),
}


# A bad input stops main with exit 1, naming its file, before connecting.
@pytest.mark.parametrize("case", sorted(BROKEN))
def test_input_checks_exit_before_any_sql(tmp_path, capsys, case):
    which, breaker = BROKEN[case]
    breaker(tmp_path)
    world = {"results": tmp_path / "results", "ai": tmp_path / "ai", "size": "70k"}
    code, calls = run_main(world, None)
    assert code == 1 and calls == []
    path = tmp_path / "results" / "data" / "table_70k.csv" if which == "table" else tmp_path / "ai" / "embeddings" / "test.npy"
    assert f"{path}:" in capsys.readouterr().err


# int4 and int2 limits: the edges load, one past them is refused with the range.
def test_int_ranges(tmp_path):
    edges = with_cell(0, "rating", "32767")
    edges.loc[1, "rating"], edges.loc[2, "product_id"], edges.loc[3, "product_id"] = "-32768", str(2**31 - 1), str(-2**31)
    t = mod.read_table(write_world(tmp_path, frame=edges)["table"])
    assert t["rating"].tolist()[:2] == [32767, -32768] and t["product_id"].tolist()[2:4] == [2**31 - 1, -2**31]

    for column, row, value, limits in [("rating", 0, "-32769", "-32768 and 32767"),
                                       ("product_id", 1, str(-2**31 - 1), "-2147483648 and 2147483647"),
                                       ("id", 3, str(2**31), "-2147483648 and 2147483647")]:
        path = write_world(tmp_path, frame=with_cell(row, column, value))["table"]
        with pytest.raises(ValueError, match=f"{column} must be between {limits}"):
            mod.read_table(path)


# The vec_row check names both the table and the vector file.
def test_vec_row_message_names_both_files(tmp_path, capsys):
    world = write_world(tmp_path, vectors=unit_rows(6))
    assert run_main(world, None)[0] == 1
    err = capsys.readouterr().err
    assert f"{world['table']}:" in err and str(world["base"]) in err


# 500k reads slice.npy from results; a missing slice is a clean exit 1.
def test_500k_reads_slice_and_missing_slice_is_clean(tmp_path, capsys):
    world = write_world(tmp_path, size="500k")
    fake = FakeConn()
    code, _ = run_main(world, fake)
    assert code == 0 and first(fake.log, "COPY reviews_500k") >= 0
    np.testing.assert_array_equal(fake.copies[0].rows[0][6], world["vectors"][0])

    world["base"].unlink()
    np.save(world["ai"] / "embeddings" / "test.npy", unit_rows(5))
    code, calls = run_main(world, None)
    assert code == 1 and calls == []
    assert f"{world['base']}:" in capsys.readouterr().err


# Results inside the ai_queries folder are refused before anything else.
def test_refuses_results_inside_ai_folder(tmp_path, capsys):
    world = write_world(tmp_path)
    world["results"] = world["ai"] / "inside"
    code, calls = run_main(world, None)
    assert code == 1 and calls == []
    assert "ai_queries" in capsys.readouterr().err
