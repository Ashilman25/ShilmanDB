"""Tests for the timed grid run in bench/vector_check/pg_vector_grid.py, on fakes and small synthetic files."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import pytest

HERE = Path(__file__).resolve().parent


# Load a module by path under a vc_ name, clear of other bench tests.
def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, HERE.parent / file_name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mod = _load("vc_grid_run", "pg_vector_grid.py")
vc, vf = mod.vc, mod.vf

T = "reviews_70k"
DIM = 384
OPTION = {o.name: o for o in mod.ALL_OPTIONS}
BY_STMTS = {tuple(mod.set_statements(o)[:7]): o.name for o in mod.ALL_OPTIONS}
ITERATIVE = [f"hnsw_{c}" for c in ("strict_ef40", "strict_ef100", "strict_ef200", "strict_ef400",
                                   "relaxed_ef40", "relaxed_ef100", "relaxed_ef200", "relaxed_ef400",
                                   "relaxed100k_ef40", "relaxed100k_ef100")]
COLUMNS = ["size", "qid", "group", "half", "filter", "family", "n_pass", "share", "est_rows", "est_share",
           "option", "kind", "intended", "scan", "index", "nodes", "reused_from", "recall", "time_ms",
           "t1", "t2", "t3", "n_returned", "timed_out", "deterministic", "ids", "filter_sql"]
N_ROWS, N_VECS, N_REVIEWS = 400, 120, 12
# A fake SET or EXPLAIN's time; a power of two in seconds keeps sums exact.
STEP_MS = 125.0
# Rows passing each filter in the synthetic world; four filters land in band A.
PASSING = {"date_0.1": 1, "date_0.3": 1, "date_1": 4, "date_3": 12, "date_10": 40, "date_30": 120, "none": N_ROWS,
           "product_random": 1, "product_near": 1, "rating_1": 40, "rating_2": 30, "rating_le2": 70,
           "rating_5": 200, "category_cases": 100, "unverified": 40}
FAMILY = {"date": "date", "none": "none", "product": "product", "rating": "rating", "category": "category",
          "unverified": "verified"}


# A plan tree as EXPLAIN (FORMAT JSON) returns it.
def plan(node: dict) -> list:
    return [{"Plan": node}]


# A sorted sequential scan.
def seq_plan() -> list:
    return plan({"Node Type": "Limit", "Plans": [{"Node Type": "Sort", "Plans": [
        {"Node Type": "Seq Scan", "Relation Name": T}]}]})


# A sorted bitmap scan over one B-tree.
def bitmap_plan(index: str = f"{T}_rating_idx") -> list:
    return plan({"Node Type": "Limit", "Plans": [{"Node Type": "Sort", "Plans": [
        {"Node Type": "Bitmap Heap Scan", "Relation Name": T, "Plans": [
            {"Node Type": "Bitmap Index Scan", "Index Name": index}]}]}]})


# An HNSW index scan, plain or in the relaxed CTE shape.
def hnsw_plan(relaxed: bool = False) -> list:
    scan = {"Node Type": "Index Scan", "Index Name": f"{T}_hnsw_idx", "Relation Name": T}
    if not relaxed:
        return plan({"Node Type": "Limit", "Plans": [scan]})
    return plan({"Node Type": "Sort", "Plans": [{"Node Type": "Limit", "Plans": [scan]}, {"Node Type": "CTE Scan"}]})


# A B-tree index scan with a sort on top.
def btree_plan() -> list:
    return plan({"Node Type": "Limit", "Plans": [{"Node Type": "Sort", "Plans": [
        {"Node Type": "Index Scan", "Index Name": f"{T}_rating_idx", "Relation Name": T}]}]})


# The plan an option gets unless a test scripts another.
def natural_plan(name: str) -> list:
    if name in ("exact_seq", "set_noindex"):
        return seq_plan()
    if name == "exact_bitmap":
        return bitmap_plan()
    return hnsw_plan(OPTION[name].relaxed)


# A fake clock: only the runner moves it.
class Clock:
    # Start at zero.
    def __init__(self):
        self.t = 0.0

    # The current fake time in seconds.
    def __call__(self) -> float:
        return self.t


# A runner scripted per option: plan, ids (one or per run), ms, timeout.
class FakeRunner:
    # Script, default ids callable, run time, estimate rows, and a cell to stop at.
    # SET and EXPLAIN each take step_ms on the clock, so timing them would show.
    def __init__(self, script=None, ids=None, ms: float = 1.0, est_rows: float = 50.0,
                 stop_at_cell: int | None = None, clock: Clock | None = None, step_ms: float = STEP_MS):
        self.script, self.ids_fn, self.ms, self.est_rows = script or {}, ids, ms, est_rows
        self.stop_at_cell, self.clock, self.step = stop_at_cell, clock or Clock(), step_ms / 1000
        self.current, self.work_mem, self.cells = None, None, 0
        self.log, self.counts, self.estimate_times = [], Counter(), []

    # Remember which option the SET statements belong to.
    def set(self, stmts: list[str]) -> None:
        self.clock.t += self.step
        self.current, self.work_mem = BY_STMTS[tuple(stmts[:7])], stmts[7]
        self.log.append(("set", self.current, self.work_mem))

    # The estimate plan for an estimate, else the option's scripted or natural plan.
    def explain(self, sql: str, params) -> list:
        self.log.append(("explain", self.current, sql))
        if sql.startswith("EXPLAIN"):
            self.estimate_times.append(self.clock.t)
        self.clock.t += self.step
        if sql.startswith("EXPLAIN"):
            self.cells += 1
            if self.stop_at_cell is not None and self.cells == self.stop_at_cell:
                raise KeyboardInterrupt
            return plan({"Node Type": "Seq Scan", "Relation Name": T, "Plan Rows": self.est_rows})
        return self.script.get(self.current, {}).get("plan") or natural_plan(self.current)

    # Advance the clock and return the scripted ids, or raise Timeout.
    def run(self, sql: str, params) -> list[int]:
        name, s = self.current, self.script.get(self.current, {})
        k = self.counts[name]
        self.counts[name] += 1
        self.log.append(("run", name, self.work_mem, sql, params["q"].tobytes()))
        if s.get("timeout"):
            raise mod.Timeout("canceled")
        times = s.get("ms", [self.ms])
        self.clock.t += times[k % len(times)] / 1000
        if "ids" in s:
            ids = s["ids"]
            return list(ids[k % len(ids)] if ids and isinstance(ids[0], list) else ids)
        if self.ids_fn is not None:
            return list(self.ids_fn(name, sql, params["q"], self.work_mem))
        return list(range(1, 11))

    # Option names in the order they ran, one per run.
    def ran(self) -> list[str]:
        return [e[1] for e in self.log if e[0] == "run"]


# A connection that answers the settings, version, count and prewarm queries.
class FakeConn:
    # Server settings, the table's row count, and a query to fail on.
    def __init__(self, settings=None, rows: int = N_ROWS, fail_on: str | None = None):
        self.settings = dict(mod.pgl.EXPECTED_SETTINGS if settings is None else settings)
        self.rows, self.fail_on, self.log, self.closed = rows, fail_on, [], False

    # Answer the few queries the grid sends on the bare connection.
    def execute(self, sql: str, params=None):
        self.log.append(sql)
        if self.fail_on is not None and self.fail_on in sql:
            raise RuntimeError(f"failed: {sql}")
        if sql == "SHOW server_version":
            return Result(one=("17.11 (Homebrew)",))
        if sql.startswith("SHOW "):
            return Result(one=(self.settings[sql[5:]],))
        if "FROM pg_extension" in sql:
            return Result(one=("0.8.7",))
        if sql.startswith("SELECT count(*)"):
            return Result(one=(self.rows,))
        if "pg_prewarm" in sql:
            return Result(one=(1,))
        raise AssertionError(f"unexpected SQL on the connection: {sql}")

    # Mark the connection closed.
    def close(self):
        self.closed = True


# A query result with one row or many.
class Result:
    # One row, or a list of rows.
    def __init__(self, one=None, rows=None):
        self.one, self.rows = one, rows or []

    # The single row.
    def fetchone(self):
        return self.one

    # Every row.
    def fetchall(self):
        return self.rows


# Seeded unit float32 rows.
def unit_rows(rng, n: int) -> np.ndarray:
    v = rng.standard_normal((n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# A tiny results and ai tree with truth for n_truth queries per group.
def write_world(tmp_path: Path, n_truth: int = 3, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    results, ai = tmp_path / "results", tmp_path / "ai"
    size_dir = results / "size70k"
    for d in (results / "data", results / "embeddings", ai / "embeddings", size_dir):
        d.mkdir(parents=True)

    base = unit_rows(rng, N_VECS)
    vec_row = np.concatenate([np.arange(N_VECS), rng.integers(0, N_VECS, N_ROWS - N_VECS)])
    rng.shuffle(vec_row)
    pd.DataFrame({
        "id": np.arange(N_ROWS), "product_id": rng.integers(0, 9, N_ROWS), "rating": rng.integers(1, 6, N_ROWS),
        "review_date": [(dt.date(2020, 1, 1) + dt.timedelta(days=int(d))).isoformat() for d in rng.integers(0, 300, N_ROWS)],
        "verified": np.where(rng.random(N_ROWS) < .9, "true", "false"), "category": "Chargers", "vec_row": vec_row,
    }).to_csv(results / "data" / "table_70k.csv", index=False)

    reviews = pd.DataFrame({"qid": [f"r{i:05d}" for i in range(1, N_REVIEWS + 1)],
                            "split": np.array(["test", "train", "validation"])[np.arange(N_REVIEWS) % 3]})
    reviews.to_csv(results / "data" / "query_reviews.csv", index=False)
    questions = vc.read_questions()
    review_vectors, question_vectors = unit_rows(rng, N_REVIEWS), unit_rows(rng, len(questions))
    np.save(results / "embeddings" / "query_reviews.npy", review_vectors)
    np.save(results / "embeddings" / "questions.npy", question_vectors)
    np.save(ai / "embeddings" / "test.npy", base)

    test_qids = reviews.loc[reviews["split"] == "test", "qid"].tolist()
    written = questions.loc[questions["group"] == "written", "qid"].tolist()
    ai53 = questions.loc[questions["group"] == "ai53", "qid"].tolist()
    queries = vf.part1_queries(test_qids, written, ai53, "70k", n_per_group=n_truth)
    vectors = dict(zip(reviews["qid"], review_vectors))
    vectors.update(zip(questions["qid"], question_vectors))

    names = list(vf.FILTER_NAMES)
    shape = (len(queries), len(names))
    n_pass, kth = np.zeros(shape, dtype=np.int64), np.full(shape, np.inf)
    top_ids, top_dists = np.full((*shape, 10), -1, dtype=np.int64), np.full((*shape, 10), np.inf)
    rows, cell_of, key_of = [], {}, {}
    for i, (qid, group, half) in enumerate(queries):
        d_rows = vc.cosine_dists(base, vectors[qid])[vec_row]
        for j, name in enumerate(names):
            n = PASSING[name]
            passing = np.arange(N_ROWS) if name == "none" else rng.choice(N_ROWS, n, replace=False)
            ids, dists = vc.topk(d_rows, 10, passing)
            top_ids[i, j, :len(ids)], top_dists[i, j, :len(dists)] = ids, dists
            n_pass[i, j], kth[i, j] = n, dists[-1]
            sql = "" if name == "none" else f"rating <> {100 * i + j}"
            rows.append((qid, group, half, name, FAMILY[name.split("_")[0]], sql, n, n / N_ROWS))
            for relaxed in (False, True):
                key = (mod.query_sql(T, sql, relaxed), vectors[qid].astype(np.float32).tobytes())
                cell_of[key], key_of[key] = ids.tolist(), (qid, name)

    pd.DataFrame(rows, columns=["qid", "group", "half", "filter", "family", "sql", "n_pass", "share"]).to_csv(
        size_dir / "filters.csv", index=False)
    (size_dir / "pg_load.json").write_text('{"rows": 400}\n')
    np.savez(size_dir / "truth_part1.npz", qids=np.array([q for q, _, _ in queries], dtype=str),
             groups=np.array([g for _, g, _ in queries], dtype=str), halves=np.array([h for _, _, h in queries], dtype=str),
             filters=np.array(names, dtype=str), n_pass=n_pass, top_ids=top_ids, top_dists=top_dists, kth=kth)
    return {"results": results, "ai": ai, "dir": size_dir, "cell_of": cell_of, "key_of": key_of, "vectors": vectors,
            "test_qids": test_qids, "written": written, "ai53": ai53}


# Ids callable for FakeRunner: the true top ids of whichever cell the query is.
def true_ids(world: dict):
    return lambda name, sql, q, work_mem: world["cell_of"][(sql, q.tobytes())]


# Run main on a world with fakes; --queries 2 and no warm-up unless given.
def run_main(world: dict, runner: FakeRunner, *extra: str, conn: FakeConn | None = None) -> int:
    conn = conn or FakeConn()
    argv = ["--size", "70k", "--results", str(world["results"]), "--ai", str(world["ai"]), *extra]
    if "--queries" not in extra:
        argv += ["--queries", "2"]
    if "--warmup" not in extra:
        argv += ["--warmup", "0"]
    return mod.main(argv, connect=lambda: conn, make_runner=lambda c: runner, timer=runner.clock)


# grid.csv as rows of strings.
def read_grid(world: dict) -> list[dict]:
    with open(world["dir"] / "grid.csv", newline="") as f:
        return list(csv.DictReader(f))


# The (qid, group, half) part1 queries for n per group in a world.
def part1(world: dict, n: int) -> list[tuple[str, str, str]]:
    return vf.part1_queries(world["test_qids"], world["written"], world["ai53"], "70k", n_per_group=n)


# One cell by hand, for cell_rows.
def make_cell(filter_name: str = "date_1", n_pass: int = 10, kth: float = 1.0, sql: str = "rating = 1"):
    return mod.Cell(qid="r00001", group="review", half="train", filter=filter_name, family="date", sql=sql,
                    n_pass=n_pass, share=n_pass / N_ROWS, kth=kth)


# measure_cell on one cell with a fixed rng.
def measure(runner: FakeRunner, filter_name: str = "date_1", seed: int = 0) -> dict:
    q = np.ones(DIM, dtype=np.float32)
    out = mod.measure_cell(runner, T, q, "rating = 1", filter_name, np.random.default_rng(seed), timer=runner.clock)
    return {m.option: m for m in out}


# The column list of grid.csv, in order.
def test_grid_columns():
    assert mod.GRID_COLUMNS == COLUMNS


# Each option's SET statements differ, so the fake can tell them apart.
def test_fake_tells_options_apart():
    assert len(BY_STMTS) == 33


# 33 options on a filtered cell, 32 with no filter (no bitmap scan).
def test_measure_cell_runs_every_applicable_option():
    got = measure(FakeRunner())
    assert len(got) == 33 and set(got) == set(OPTION)
    none = measure(FakeRunner(), "none")
    assert len(none) == 32 and "exact_bitmap" not in none
    assert [m.option for m in mod.measure_cell(FakeRunner(), T, np.ones(DIM, np.float32), "", "none",
                                               np.random.default_rng(0))] == [o.name for o in mod.ALL_OPTIONS
                                                                              if o.name != "exact_bitmap"]


# Forced options run first in a shuffled order, three runs each, then settings.
def test_forced_first_shuffled_three_runs():
    runner = FakeRunner(script={"set_strict_ef100": {"plan": btree_plan()}})
    measure(runner)
    ran = runner.ran()
    forced_part = ran[:17 * 3]
    assert set(forced_part) == {o.name for o in mod.FORCED} and Counter(forced_part) == {o.name: 3 for o in mod.FORCED}
    order = list(dict.fromkeys(forced_part))
    assert order != [o.name for o in mod.FORCED]
    assert ran[17 * 3:] == ["set_strict_ef100"] * 3
    sets = [e for e in runner.log if e[0] == "set"]
    assert all(e[2] == "RESET work_mem" for e in sets)


# The same seed gives the same order; another seed a different one.
def test_option_order_follows_rng():
    a, b, c = FakeRunner(), FakeRunner(), FakeRunner()
    measure(a, seed=5), measure(b, seed=5), measure(c, seed=6)
    assert a.log == b.log and a.log != c.log


# Each time is timer after minus timer before, in ms; the row takes the median.
def test_median_and_determinism_flag():
    runner = FakeRunner(script={"hnsw_off_ef100": {"ms": [9.0, 1.0, 2.0], "ids": [[1, 2], [1, 3], [1, 4]]}})
    got = measure(runner)["hnsw_off_ef100"]
    assert got.times_ms == pytest.approx([9.0, 1.0, 2.0]) and got.ids == [1, 2]
    assert got.deterministic is False and got.timed_out is False and got.reused_from is None
    assert measure(FakeRunner())["hnsw_off_ef100"].deterministic is True

    row = mod.grid_row("70k", make_cell(), 50.0, N_ROWS, got, 0.5)
    assert row[COLUMNS.index("time_ms")] == pytest.approx(2.0)
    assert row[COLUMNS.index("t1"):COLUMNS.index("t3") + 1] == pytest.approx([9.0, 1.0, 2.0])
    assert row[COLUMNS.index("ids")] == "1 2" and row[COLUMNS.index("n_returned")] == 2


# A setting whose plan matches its forced twin copies it and runs nothing.
def test_setting_reuses_forced_when_plan_matches():
    runner = FakeRunner(script={"hnsw_relaxed_ef100": {"ms": [4.0, 5.0, 6.0], "ids": [[9, 8, 7], [9, 8, 6], [9, 8, 7]]}})
    got = measure(runner)
    reused, twin = got["set_relaxed_ef100"], got["hnsw_relaxed_ef100"]
    assert reused.reused_from == "hnsw_relaxed_ef100" and "set_relaxed_ef100" not in runner.ran()
    assert (reused.ids, reused.times_ms, reused.timed_out, reused.deterministic) == (
        twin.ids, twin.times_ms, twin.timed_out, twin.deterministic)
    assert reused.plan == mod.classify_plan(hnsw_plan(True)) and reused.deterministic is False
    assert got["set_default"].reused_from == "hnsw_off_ef40"
    assert got["set_noindex"].reused_from == "exact_seq"


# A setting with a bitmap plan copies exact_bitmap when the nodes match.
def test_setting_reuses_bitmap_twin():
    got = measure(FakeRunner(script={"set_off_ef200": {"plan": bitmap_plan()}}))
    assert got["set_off_ef200"].reused_from == "exact_bitmap"


# Same scan, different nodes: measured, not copied.
def test_setting_measured_when_nodes_differ():
    other = plan({"Node Type": "Sort", "Plans": [{"Node Type": "Limit", "Plans": [
        {"Node Type": "Index Scan", "Index Name": f"{T}_hnsw_idx", "Relation Name": T}]}]})
    runner = FakeRunner(script={"set_strict_ef100": {"plan": other}, "set_relaxed_ef40": {"plan": bitmap_plan(f"{T}_date_idx")}})
    got = measure(runner)
    assert got["set_strict_ef100"].reused_from is None and runner.ran().count("set_strict_ef100") == 3
    assert got["set_relaxed_ef40"].reused_from is None and runner.ran().count("set_relaxed_ef40") == 3
    assert got["set_strict_ef100"].plan.nodes == mod.classify_plan(other).nodes


# A setting planning a B-tree index scan has no twin, so it's measured.
def test_setting_with_btree_plan_is_measured():
    runner = FakeRunner(script={"set_off_ef400": {"plan": btree_plan()}})
    got = measure(runner)["set_off_ef400"]
    assert got.reused_from is None and got.plan.scan == "btree_index" and runner.ran().count("set_off_ef400") == 3


# reuse_match: hnsw by config name, seq and bitmap by the exact options, nodes must match.
def test_reuse_match_rules():
    seq, bitmap = mod.classify_plan(seq_plan()), mod.classify_plan(bitmap_plan())
    hnsw, relaxed = mod.classify_plan(hnsw_plan()), mod.classify_plan(hnsw_plan(True))
    measured = lambda shape: mod.Measured("x", [1], [1.0] * 3, False, shape, None, True)
    forced = {"exact_seq": measured(seq), "exact_bitmap": measured(bitmap), "hnsw_off_ef40": measured(hnsw),
              "hnsw_off_ef100": measured(hnsw), "hnsw_relaxed_ef40": measured(relaxed)}
    assert mod.reuse_match(OPTION["set_default"], hnsw, forced) == "hnsw_off_ef40"
    assert mod.reuse_match(OPTION["set_off_ef100"], hnsw, forced) == "hnsw_off_ef100"
    assert mod.reuse_match(OPTION["set_relaxed_ef40"], relaxed, forced) == "hnsw_relaxed_ef40"
    assert mod.reuse_match(OPTION["set_relaxed_ef40"], hnsw, forced) is None
    assert mod.reuse_match(OPTION["set_off_ef200"], hnsw, forced) is None
    assert mod.reuse_match(OPTION["set_noindex"], seq, forced) == "exact_seq"
    assert mod.reuse_match(OPTION["set_off_ef100"], bitmap, forced) == "exact_bitmap"
    assert mod.reuse_match(OPTION["set_off_ef100"], mod.classify_plan(btree_plan()), forced) is None
    no_bitmap = {k: v for k, v in forced.items() if k != "exact_bitmap"}
    assert mod.reuse_match(OPTION["set_off_ef100"], bitmap, no_bitmap) is None


# A timeout records 10 s three times and no ids; the next option still runs.
def test_timeout_recorded():
    runner = FakeRunner(script={"exact_seq": {"timeout": True}})
    got = measure(runner)
    m = got["exact_seq"]
    assert m.timed_out is True and m.times_ms == [10000.0] * 3 and m.ids == [] and m.deterministic is False
    assert runner.ran().count("exact_seq") == 1 and len(got) == 33
    assert got["set_noindex"].reused_from == "exact_seq" and got["set_noindex"].timed_out is True

    row = mod.grid_row("70k", make_cell(), 50.0, N_ROWS, m, 0.0)
    assert row[COLUMNS.index("timed_out")] is True and row[COLUMNS.index("n_returned")] == 0
    assert row[COLUMNS.index("time_ms")] == 10000.0 and row[COLUMNS.index("ids")] == ""


# Recall uses the truth's kth and n_pass; a timed-out row scores 0.
def test_recall_with_short_filter():
    base = unit_rows(np.random.default_rng(4), 30)
    vec_row = np.arange(30)
    q = base[0]
    d = vc.cosine_dists(base, q)
    passing = np.array([3, 7, 11])
    ids, dists = vc.topk(d, 10, passing)
    cell = make_cell(n_pass=3, kth=float(dists[-1]))
    exact = mod.Measured("exact_seq", ids.tolist(), [1.0] * 3, False, mod.classify_plan(seq_plan()), None, True)
    assert mod.cell_recall(exact, cell, base, vec_row, q) == 1.0
    short = mod.Measured("hnsw_off_ef40", ids.tolist()[:2], [1.0] * 3, False, mod.classify_plan(hnsw_plan()), None, True)
    assert mod.cell_recall(short, cell, base, vec_row, q) == pytest.approx(2 / 3)
    timed = mod.Measured("exact_seq", [], [10000.0] * 3, True, mod.classify_plan(seq_plan()), None, False)
    assert mod.cell_recall(timed, cell, base, vec_row, q) == 0.0
    empty = make_cell(n_pass=0, kth=float("inf"))
    nothing = mod.Measured("exact_seq", [], [1.0] * 3, False, mod.classify_plan(seq_plan()), None, True)
    assert mod.cell_recall(nothing, empty, base, vec_row, q) == 1.0
    assert mod.cell_recall(timed, empty, base, vec_row, q) == 0.0


# cell_rows estimates under default GUCs, then writes one row per option with its recall.
def test_cell_rows_estimate_and_columns():
    base = unit_rows(np.random.default_rng(5), N_VECS)
    vec_row = np.arange(N_ROWS) % N_VECS
    q = base[3]
    runner = FakeRunner(est_rows=80.0, ids=lambda *a: [3])
    rows = mod.cell_rows(runner, "70k", T, make_cell(kth=0.0, n_pass=1), base, vec_row, q, N_ROWS,
                         np.random.default_rng(0), timer=runner.clock)
    assert runner.log[0] == ("set", "set_default", "RESET work_mem") and runner.log[1][0] == "explain"
    assert runner.log[1][2] == mod.estimate_sql(T, "rating = 1")
    assert len(rows) == 33 and all(len(r) == len(COLUMNS) for r in rows)
    by = {r[COLUMNS.index("option")]: dict(zip(COLUMNS, r)) for r in rows}
    seq, setting = by["exact_seq"], by["set_off_ef100"]
    assert seq["est_rows"] == 80.0 and seq["est_share"] == pytest.approx(80.0 / N_ROWS)
    assert (seq["kind"], seq["intended"], seq["scan"], seq["index"]) == ("forced", "seq", "seq", "")
    assert seq["nodes"] == "Limit>Sort>Seq Scan" and seq["recall"] == 1.0 and seq["reused_from"] == ""
    assert (setting["kind"], setting["intended"], setting["reused_from"]) == ("setting", "", "hnsw_off_ef100")
    assert setting["index"] == f"{T}_hnsw_idx" and seq["size"] == "70k" and seq["half"] == "train"


# The cells of a world for n per group, in filters.csv order, as (qid, filter).
def base_cells(world: dict, n: int) -> list[tuple[str, str]]:
    return [(q, f) for q, _, _ in part1(world, n) for f in vf.FILTER_NAMES]


# Cells run in the seeded shuffle; each cell's options follow its own seed.
def test_seeded_cell_and_option_order(tmp_path):
    world = write_world(tmp_path)
    runner = FakeRunner(ids=true_ids(world))
    run_main(world, runner)
    cells = base_cells(world, 2)
    order = np.random.default_rng(vc.seed_for("70k", "cells")).permutation(len(cells))
    assert list(dict.fromkeys((r["qid"], r["filter"]) for r in read_grid(world))) == [cells[i] for i in order]

    first = cells[order[0]]
    forced = [o.name for o in mod.FORCED if mod.applies(o, first[1])]
    perm = np.random.default_rng(vc.seed_for("70k", "order", *first)).permutation(len(forced))
    assert runner.ran()[:3 * len(forced):3] == [forced[i] for i in perm]


# Warm-up and memory check cells come from their own seeds.
def test_warmup_and_memcheck_cells_are_seeded(tmp_path):
    world = write_world(tmp_path)
    runner = FakeRunner(ids=true_ids(world), stop_at_cell=1)
    with pytest.raises(KeyboardInterrupt):
        run_main(world, runner, "--warmup", "3")
    cells = base_cells(world, 2)
    picks = np.random.default_rng(vc.seed_for("70k", "warmup")).choice(len(cells), size=3, replace=False)
    ran = [world["key_of"][(e[3], e[4])] for e in runner.log if e[0] == "run"]
    assert list(dict.fromkeys(ran)) == [cells[i] for i in picks]

    run_main(world, FakeRunner(ids=true_ids(world)), "--queries", "3")
    band_a = [c for c in base_cells(world, 3) if PASSING[c[1]] / N_ROWS < 0.005]
    picks = np.random.default_rng(vc.seed_for("70k", "memcheck")).choice(len(band_a), size=20, replace=False)
    memo = pd.read_csv(world["dir"] / "memcheck.csv", dtype=str)
    assert list(dict.fromkeys(zip(memo["qid"], memo["filter"]))) == [band_a[i] for i in picks]


# A full small-world run: every cell and option, recall 1 for true ids.
def test_main_writes_every_cell(tmp_path):
    world = write_world(tmp_path)
    runner = FakeRunner(ids=true_ids(world))
    assert run_main(world, runner) == 0
    rows = read_grid(world)
    assert len(rows) == 60 * 33 - 4
    assert list(rows[0]) == COLUMNS
    cells = Counter((r["qid"], r["filter"]) for r in rows)
    assert len(cells) == 60 and all(n == (32 if f == "none" else 33) for (_, f), n in cells.items())
    assert {r["recall"] for r in rows} == {"1.0"}
    halves = {(q, g, h) for q, g, h in part1(world, 2)}
    assert {(r["qid"], r["group"], r["half"]) for r in rows} == halves


# The none filter's empty SQL stays empty, never the text nan.
def test_none_filter_has_no_where(tmp_path):
    world = write_world(tmp_path)
    runner = FakeRunner(ids=true_ids(world))
    run_main(world, runner)
    sqls = {e[3] for e in runner.log if e[0] == "run"}
    assert any(s.endswith(f"FROM {T} ORDER BY embedding <=> %(q)s LIMIT 10") for s in sqls)
    assert not any("nan" in s for s in sqls)


# grid_info.json records the queries per group, cell count, warm-up, runs and versions.
def test_grid_info_records_n(tmp_path):
    world = write_world(tmp_path)
    conn = FakeConn()
    run_main(world, FakeRunner(ids=true_ids(world)), "--warmup", "1", conn=conn)
    prewarms = [i for i, sql in enumerate(conn.log) if "pg_prewarm" in sql]
    shows = [i for i, sql in enumerate(conn.log) if sql.startswith("SHOW")]
    assert len(prewarms) == 8 and conn.log[prewarms[0]] == f"SELECT pg_prewarm('{T}')" and max(shows) < prewarms[0]
    info = json.loads((world["dir"] / "grid_info.json").read_text())
    assert info == {"size": "70k", "n_per_group": 2, "cells": 60, "warmup": 1, "runs": 3,
                    "server_version": "17.11 (Homebrew)", "extversion": "0.8.7",
                    "load_sha256": hashlib.sha256(b'{"rows": 400}\n').hexdigest()}


# Settings are checked before anything else touches the database; nothing measured or written.
def test_grid_refuses_wrong_settings(tmp_path, capsys):
    world = write_world(tmp_path)
    runner = FakeRunner(ids=true_ids(world))
    conn = FakeConn(settings={**mod.pgl.EXPECTED_SETTINGS, "work_mem": "4MB", "jit": "on"})
    assert run_main(world, runner, conn=conn) == 1
    err = capsys.readouterr().err
    assert "work_mem" in err and "jit" in err
    assert runner.log == [] and all(s.startswith("SHOW") for s in conn.log) and conn.closed
    assert not any((world["dir"] / n).exists() for n in ("grid.csv", "memcheck.csv", "grid_info.json"))


# A table with a different row count than the CSV stops the run.
def test_grid_refuses_wrong_row_count(tmp_path, capsys):
    world = write_world(tmp_path)
    runner = FakeRunner(ids=true_ids(world))
    assert run_main(world, runner, conn=FakeConn(rows=N_ROWS + 1)) == 1
    assert T in capsys.readouterr().err and runner.log == []


# A second run measures nothing, keeps grid.csv identical and redoes the memory check.
def test_resume_skips_done_cells_without_duplicates(tmp_path, capsys):
    world = write_world(tmp_path)
    assert run_main(world, FakeRunner(ids=true_ids(world))) == 0
    before = (world["dir"] / "grid.csv").read_bytes()
    (world["dir"] / "memcheck.csv").unlink()
    capsys.readouterr()

    again = FakeRunner(ids=true_ids(world))
    assert run_main(world, again) == 0
    assert (world["dir"] / "grid.csv").read_bytes() == before
    assert not any(e[0] == "explain" for e in again.log)
    assert set(again.ran()) <= set(ITERATIVE) and (world["dir"] / "memcheck.csv").exists()
    out = capsys.readouterr().out
    assert "0 cells to measure" in out and "starting the grid over" not in out
    assert ("resume: kept 60 cells, dropped 0 cells (0 rows): 0 outside the current cell set, "
            "0 incomplete, 0 changed inputs") in out.splitlines()


# A run stopped part way, then resumed, ends with a straight run's rows.
def test_interrupted_run_resumes_to_same_rows(tmp_path):
    world = write_world(tmp_path)
    # Rows without the time columns, which carry the fake clock's rounding.
    def rows_without_times() -> list[tuple]:
        times = {"time_ms", "t1", "t2", "t3"}
        return sorted(tuple(v for k, v in r.items() if k not in times) for r in read_grid(world))

    run_main(world, FakeRunner(ids=true_ids(world)))
    want = rows_without_times()
    (world["dir"] / "grid.csv").unlink()

    with pytest.raises(KeyboardInterrupt):
        run_main(world, FakeRunner(ids=true_ids(world), stop_at_cell=8))
    assert len({(r["qid"], r["filter"]) for r in read_grid(world)}) == 7

    assert run_main(world, FakeRunner(ids=true_ids(world))) == 0
    got = rows_without_times()
    assert got == want and len(got) == 60 * 33 - 4


# A resumed cell is measured in the same option order as in a straight run.
def test_per_cell_order_same_on_resume(tmp_path):
    world = write_world(tmp_path)

    # Option order per cell, keyed by estimate SQL and query vector.
    # Stops at the memory check.
    def orders(runner: FakeRunner) -> dict:
        out, key = {}, None
        for e in runner.log:
            if e[0] == "set" and e[2] == "SET work_mem = '512MB'":
                break
            if e[0] == "explain" and e[2].startswith("EXPLAIN"):
                key = e[2]
            elif e[0] == "run" and key is not None:
                out.setdefault((key, e[4]), []).append(e[1])
        return out

    straight = FakeRunner(ids=true_ids(world))
    run_main(world, straight, "--redo")
    with pytest.raises(KeyboardInterrupt):
        run_main(world, FakeRunner(ids=true_ids(world), stop_at_cell=20), "--redo")
    resumed = FakeRunner(ids=true_ids(world))
    run_main(world, resumed)
    want, got = orders(straight), orders(resumed)
    assert len(got) == 41 and all(want[k] == v for k, v in got.items())


# Stale rows go: incomplete cells, cells outside the set, a cut-off last line.
# A complete cell's half is rewritten from the current query count.
def test_resume_rewrite_drops_stale_rows_and_fixes_half(tmp_path, capsys):
    world = write_world(tmp_path)
    assert run_main(world, FakeRunner(ids=true_ids(world)), "--queries", "3") == 0
    rows = read_grid(world)
    three, two = part1(world, 3), part1(world, 2)
    dropped_qids = {q for q, _, _ in three} - {q for q, _, _ in two}
    assert len(dropped_qids) == 2

    keep_qid = two[0][0]
    cut = next(i for i, r in enumerate(rows) if r["qid"] == keep_qid and r["filter"] == "date_1")
    tail = [r for r in rows if r["qid"] == keep_qid and r["filter"] == "date_3"]
    broken = [r for i, r in enumerate(rows) if i != cut and r not in tail]
    changed = {"rating_5": ("size", "500k"), "rating_1": ("n_pass", "41"), "rating_2": ("share", "0.5"),
               "rating_le2": ("filter_sql", "rating = 9")}
    for r in broken:
        if r["qid"] == keep_qid and r["filter"] in changed:
            column, value = changed[r["filter"]]
            if column == "size" or r["option"] == "set_noindex":
                r[column] = value
    path = world["dir"] / "grid.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        for r in broken + tail:
            w.writerow([r[c] if c != "half" else "stale" for c in COLUMNS])
    # The date_3 cell's last row ends cut off inside its filter SQL, field count intact.
    text, cut_sql = path.read_text(), tail[-1]["filter_sql"]
    assert text.endswith(f",{cut_sql}\n") and len(cut_sql) > 4
    path.write_text(text[:-len(cut_sql) - 1] + cut_sql[:4])

    runner = FakeRunner(ids=true_ids(world))
    capsys.readouterr()
    assert run_main(world, runner) == 0
    outside_rows = sum(r["qid"] in dropped_qids for r in rows)
    resume_lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("resume:")]
    assert resume_lines == [f"resume: kept 54 cells, dropped 36 cells ({outside_rows + 32 + 32 + 4 * 33:,} rows): "
                            "30 outside the current cell set, 2 incomplete, 4 changed inputs"]
    after = read_grid(world)
    assert len(after) == 60 * 33 - 4
    assert not {r["qid"] for r in after} & dropped_qids
    assert {(r["qid"], r["group"], r["half"]) for r in after} == set(two)
    measured_cells = {e[2] for e in runner.log if e[0] == "explain" and e[2].startswith("EXPLAIN")}
    assert len(measured_cells) == 6
    counts = Counter((r["qid"], r["filter"]) for r in after)
    assert counts[(keep_qid, "date_1")] == 33 and counts[(keep_qid, "date_3")] == 33
    assert all(counts[(keep_qid, f)] == 33 for f in changed) and {r["size"] for r in after} == {"70k"}
    frame = pd.read_csv(world["dir"] / "filters.csv", dtype=str, keep_default_na=False)
    want_sql = dict(zip(zip(frame["qid"], frame["filter"]), frame["sql"]))
    assert all(r["filter_sql"] == want_sql[(r["qid"], r["filter"])] for r in after)
    assert {r["n_pass"] for r in after if (r["qid"], r["filter"]) == (keep_qid, "rating_1")} == {"40"}
    assert {r["ids"] for r in after if (r["qid"], r["filter"]) == (keep_qid, "date_3")} == {tail[-1]["ids"]}


# grid.csv with other columns stops the run before anything is measured.
def test_resume_refuses_other_columns(tmp_path, capsys):
    world = write_world(tmp_path)
    (world["dir"] / "grid.csv").write_text("size,qid\n70k,r00001\n")
    load_sha256 = hashlib.sha256((world["dir"] / "pg_load.json").read_bytes()).hexdigest()
    (world["dir"] / "grid_info.json").write_text(json.dumps({"load_sha256": load_sha256}))
    runner = FakeRunner(ids=true_ids(world))
    assert run_main(world, runner) == 1
    assert "grid.csv" in capsys.readouterr().err and not any(e[0] == "explain" for e in runner.log)


# Warm-up runs every applicable option once on seeded cells, untimed, writing nothing.
def test_warmup_writes_nothing(tmp_path):
    world = write_world(tmp_path)
    runner = FakeRunner(ids=true_ids(world), stop_at_cell=1)
    with pytest.raises(KeyboardInterrupt):
        run_main(world, runner, "--warmup", "3")
    ran = runner.ran()
    assert 3 * 32 <= len(ran) <= 3 * 33 and len(set(e[4] for e in runner.log if e[0] == "run")) <= 3
    assert not any(e[0] == "explain" and not e[2].startswith("EXPLAIN") for e in runner.log)
    assert not (world["dir"] / "grid.csv").exists() and not (world["dir"] / "memcheck.csv").exists()

    other = FakeRunner(ids=true_ids(world), stop_at_cell=1)
    with pytest.raises(KeyboardInterrupt):
        run_main(world, other, "--warmup", "3")
    assert other.log == runner.log


# Memory check: band A cells only, the 10 iterative options at 512MB, reset afterwards.
def test_memcheck_band_a_iterative_only(tmp_path):
    world = write_world(tmp_path)
    runner = FakeRunner(ids=true_ids(world))
    assert run_main(world, runner, "--queries", "3") == 0
    memo = pd.read_csv(world["dir"] / "memcheck.csv", dtype=str, keep_default_na=False)
    assert list(memo.columns) == ["qid", "filter", "option", "ids_equal"]
    band_a = {f for f, n in PASSING.items() if n / N_ROWS < 0.005}
    assert set(memo["filter"]) <= band_a and set(memo["option"]) == set(ITERATIVE)
    cells = set(zip(memo["qid"], memo["filter"]))
    assert len(cells) == 20 and len(memo) == 200 and set(memo["ids_equal"]) == {"True"}

    big = [e for e in runner.log if e[0] == "set" and e[2] == "SET work_mem = '512MB'"]
    assert len(big) == 200 and {e[1] for e in big} == set(ITERATIVE)
    last_big = max(i for i, e in enumerate(runner.log) if e[0] == "set" and e[2] != "RESET work_mem")
    later = [e for e in runner.log[last_big + 1:] if e[0] == "set"]
    assert later and later[-1][2] == "RESET work_mem" and runner.log[-1][0] == "set"


# Different ids at 512MB, or a timeout there, show as false.
def test_memcheck_flags_different_ids(tmp_path):
    world = write_world(tmp_path)
    truth = true_ids(world)

    # True ids at 64MB, the list reversed at 512MB.
    def ids(name, sql, q, work_mem):
        found = truth(name, sql, q, work_mem)
        return found[::-1] + [999] if work_mem == "SET work_mem = '512MB'" else found

    assert run_main(world, FakeRunner(ids=ids)) == 0
    memo = pd.read_csv(world["dir"] / "memcheck.csv")
    assert len(memo) > 0 and not memo["ids_equal"].any()


# The memory check sample is seeded: the same cells on every run.
def test_memcheck_cells_are_seeded(tmp_path):
    world = write_world(tmp_path)
    run_main(world, FakeRunner(ids=true_ids(world)), "--queries", "3")
    first = (world["dir"] / "memcheck.csv").read_bytes()
    run_main(world, FakeRunner(ids=true_ids(world)), "--queries", "3")
    assert (world["dir"] / "memcheck.csv").read_bytes() == first


# After N cells: elapsed per cell times cells left to measure, in minutes.
def test_projection_message(tmp_path, capsys):
    world = write_world(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        run_main(world, FakeRunner(ids=true_ids(world), stop_at_cell=11))
    capsys.readouterr()
    runner = FakeRunner(ids=true_ids(world), ms=1000.0, script={
        name: {"plan": btree_plan()} for name in ("set_off_ef100", "set_strict_ef40")})
    assert run_main(world, runner, "--projection-cells", "2") == 0
    elapsed = runner.estimate_times[2] - runner.estimate_times[0]
    assert elapsed >= 2 * 54
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("projected grid time")]
    assert lines == [f"projected grid time: {elapsed / 2 * 50 / 60:.1f} min for 50 cells"]


# --redo deletes grid.csv, memcheck.csv and grid_info.json, then measures everything again.
def test_redo_deletes_three_files(tmp_path):
    world = write_world(tmp_path)
    run_main(world, FakeRunner(ids=true_ids(world)))
    for name in ("grid.csv", "memcheck.csv", "grid_info.json"):
        (world["dir"] / name).write_text("old\n")
    runner = FakeRunner(ids=true_ids(world))
    assert run_main(world, runner, "--redo") == 0
    assert len(read_grid(world)) == 60 * 33 - 4
    assert json.loads((world["dir"] / "grid_info.json").read_text())["cells"] == 60
    assert (world["dir"] / "memcheck.csv").read_text().startswith("qid,filter,option,ids_equal")

    for name in ("grid.csv", "memcheck.csv", "grid_info.json"):
        (world["dir"] / name).write_text("old\n")
    with pytest.raises(RuntimeError):
        run_main(world, FakeRunner(), "--redo", conn=FakeConn(fail_on="pg_extension"))
    assert not any((world["dir"] / n).exists() for n in ("grid.csv", "memcheck.csv", "grid_info.json"))


# A missing input exits 1 naming the file, before connecting.
@pytest.mark.parametrize("name", ["size70k/filters.csv", "size70k/truth_part1.npz", "data/table_70k.csv",
                                  "data/query_reviews.csv", "embeddings/questions.npy", "size70k/pg_load.json"])
def test_missing_input_exits_1(tmp_path, capsys, name):
    world = write_world(tmp_path)
    (world["results"] / name).unlink()
    connected = []
    argv = ["--size", "70k", "--results", str(world["results"]), "--ai", str(world["ai"]), "--queries", "2"]
    assert mod.main(argv, connect=lambda: connected.append(1)) == 1
    assert Path(name).name in capsys.readouterr().err and connected == []


# At 500k the base vectors are slice.npy; missing, it's named and nothing connects.
def test_missing_500k_base_exits_1(tmp_path, capsys):
    world = write_world(tmp_path)
    for name in ("table_70k.csv",):
        (world["results"] / "data" / name).rename(world["results"] / "data" / "table_500k.csv")
    (world["results"] / "size500k").mkdir()
    for name in ("filters.csv", "truth_part1.npz", "pg_load.json"):
        (world["dir"] / name).rename(world["results"] / "size500k" / name)
    connected = []
    argv = ["--size", "500k", "--results", str(world["results"]), "--ai", str(world["ai"]), "--queries", "2"]
    assert mod.main(argv, connect=lambda: connected.append(1)) == 1
    assert "slice.npy" in capsys.readouterr().err and connected == []


# Truth for fewer queries than asked for: filters.csv is named.
def test_too_few_truth_queries_exits_1(tmp_path, capsys):
    world = write_world(tmp_path, n_truth=2)
    assert run_main(world, FakeRunner(), "--queries", "3") == 1
    assert "filters.csv" in capsys.readouterr().err


# filters.csv and the truth file must agree on rows passing.
def test_truth_mismatch_exits_1(tmp_path, capsys):
    world = write_world(tmp_path)
    path = world["dir"] / "truth_part1.npz"
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["n_pass"][0, 0] += 1
    np.savez(path, **arrays)
    assert run_main(world, FakeRunner()) == 1
    assert "truth_part1.npz" in capsys.readouterr().err


# A query count part1_queries refuses exits 2 with its message.
def test_bad_queries_exits_2(tmp_path, capsys):
    world = write_world(tmp_path)
    assert run_main(world, FakeRunner(), "--queries", "0") == 2
    assert "queries per group" in capsys.readouterr().err


# Recorded times are exactly the run times: no SET or EXPLAIN inside the timed window.
def test_times_exclude_set_and_explain():
    runner = FakeRunner(ms=62.5, script={"hnsw_off_ef100": {"ms": [15.625, 31.25, 62.5]},
                                         "set_strict_ef40": {"plan": btree_plan()}})
    got = measure(runner)
    assert got["hnsw_off_ef100"].times_ms == [15.625, 31.25, 62.5]
    for name, m in got.items():
        if name != "hnsw_off_ef100" and m.reused_from is None:
            assert m.times_ms == [62.5, 62.5, 62.5], name
    assert got["set_strict_ef40"].reused_from is None
    # 33 SETs and 33 EXPLAINs, 51 runs at 62.5 ms, and hnsw_off_ef100's three.
    assert runner.clock.t == (66 * STEP_MS + 51 * 62.5 + 15.625 + 31.25 + 62.5) / 1000


# The same holds through main: every measured row's times are the run time exactly.
def test_grid_times_exact_through_main(tmp_path):
    world = write_world(tmp_path)
    assert run_main(world, FakeRunner(ids=true_ids(world), ms=62.5), "--warmup", "2") == 0
    rows = [r for r in read_grid(world) if not r["reused_from"]]
    assert rows and all((r["t1"], r["t2"], r["t3"], r["time_ms"]) == ("62.5",) * 4 for r in rows)


# Rows from another load of the table are all dropped: the HNSW graph changed.
def test_reload_starts_the_grid_over(tmp_path, capsys):
    world = write_world(tmp_path)
    assert run_main(world, FakeRunner(ids=true_ids(world))) == 0
    (world["dir"] / "pg_load.json").write_text('{"rows": 400, "hnsw_build_s": 6.1}\n')
    capsys.readouterr()

    refused = FakeRunner(ids=true_ids(world))
    bad = FakeConn(settings={**mod.pgl.EXPECTED_SETTINGS, "jit": "on"})
    assert run_main(world, refused, conn=bad) == 1
    assert len(read_grid(world)) == 60 * 33 - 4 and (world["dir"] / "memcheck.csv").exists()
    capsys.readouterr()

    (world["dir"] / "memcheck.csv").write_text("old\n")
    runner = FakeRunner(ids=true_ids(world))
    assert run_main(world, runner, "--projection-cells", "100") == 0
    out = capsys.readouterr().out.splitlines()
    assert "table reloaded since grid.csv was written (pg_load.json changed): starting the grid over" in out
    assert "grid: 60 cells to measure, 0 already done" in out and not any(ln.startswith("resume:") for ln in out)
    assert len(runner.estimate_times) == 60 and len(read_grid(world)) == 60 * 33 - 4
    assert (world["dir"] / "memcheck.csv").read_text().startswith("qid,filter,option,ids_equal")
    info = json.loads((world["dir"] / "grid_info.json").read_text())
    assert info["load_sha256"] == hashlib.sha256((world["dir"] / "pg_load.json").read_bytes()).hexdigest()


# A run stopped right after starting over leaves no old grid or memory check behind.
def test_start_over_removes_old_files_first(tmp_path):
    world = write_world(tmp_path)
    run_main(world, FakeRunner(ids=true_ids(world)))
    (world["dir"] / "pg_load.json").write_text('{"rows": 400, "reloaded": true}\n')
    with pytest.raises(KeyboardInterrupt):
        run_main(world, FakeRunner(ids=true_ids(world), stop_at_cell=1))
    assert not (world["dir"] / "grid.csv").exists() and not (world["dir"] / "memcheck.csv").exists()
    info = json.loads((world["dir"] / "grid_info.json").read_text())
    assert info["load_sha256"] == hashlib.sha256((world["dir"] / "pg_load.json").read_bytes()).hexdigest()


# No grid_info.json, or one without load_sha256, also starts over; old columns too.
@pytest.mark.parametrize("info", ["missing", "no_hash", "old_columns"])
def test_unknown_load_starts_over(tmp_path, capsys, info):
    world = write_world(tmp_path)
    run_main(world, FakeRunner(ids=true_ids(world)))
    path = world["dir"] / "grid_info.json"
    if info == "missing":
        path.unlink()
    else:
        data = json.loads(path.read_text())
        del data["load_sha256"]
        path.write_text(json.dumps(data))
    if info == "old_columns":
        grid = world["dir"] / "grid.csv"
        grid.write_text("\n".join(",".join(ln.split(",")[:-1]) for ln in grid.read_text().splitlines()) + "\n")
    capsys.readouterr()
    runner = FakeRunner(ids=true_ids(world))
    assert run_main(world, runner) == 0
    assert "starting the grid over" in capsys.readouterr().out and len(runner.estimate_times) == 60
    assert list(read_grid(world)[0]) == COLUMNS


# A grid.csv with only its header has no rows to protect: no start-over message.
def test_header_only_grid_needs_no_start_over(tmp_path, capsys):
    world = write_world(tmp_path)
    (world["dir"] / "grid.csv").write_text(",".join(COLUMNS) + "\n")
    assert run_main(world, FakeRunner(ids=true_ids(world))) == 0
    out = capsys.readouterr().out
    assert "starting the grid over" not in out and "grid: 60 cells to measure, 0 already done" in out


# PgRunner runs SETs, wraps queries in EXPLAIN, returns ids, turns a cancel into Timeout.
def test_pg_runner():
    explain_json = hnsw_plan()

    # A connection that logs SQL; SELECTs cancel when asked to.
    class Conn:
        # Cancel every SELECT when cancel is set.
        def __init__(self, cancel=False):
            self.cancel, self.log = cancel, []

        # Log the SQL and answer like Postgres would here.
        def execute(self, sql, params=None):
            self.log.append((sql, params))
            if sql.startswith("EXPLAIN"):
                return Result(one=(explain_json,))
            if self.cancel:
                raise psycopg.errors.QueryCanceled("canceling statement due to statement timeout")
            return Result(rows=[(np.int32(5), 0.1), (7, 0.2)])

    conn = Conn()
    runner = mod.PgRunner(conn)
    runner.set(["SET a = '1'", "RESET work_mem"])
    assert conn.log == [("SET a = '1'", None), ("RESET work_mem", None)]
    params = {"q": np.ones(3, dtype=np.float32)}
    assert runner.explain("SELECT id FROM t", params) == explain_json
    assert conn.log[-1] == ("EXPLAIN (FORMAT JSON) SELECT id FROM t", params)
    runner.explain(mod.estimate_sql(T, "rating = 1"), {})
    assert conn.log[-1] == (mod.estimate_sql(T, "rating = 1"), None)
    ids = runner.run("SELECT id FROM t", params)
    assert ids == [5, 7] and all(type(i) is int for i in ids)
    with pytest.raises(mod.Timeout):
        mod.PgRunner(Conn(cancel=True)).run("SELECT id FROM t", params)
