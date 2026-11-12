# The vector search grid: options, their SQL and GUCs, plan reading, and the timed run
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg


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
vf = vc.load_sibling("vc_filters", "vector_filters.py")
pgl = vc.load_sibling("vc_pg_vector_load", "pg_vector_load.py")

# Every option sets all 7, so nothing carries over from the option before.
GUC_DEFAULTS = (
    ("enable_seqscan", "on"),
    ("enable_bitmapscan", "on"),
    ("enable_indexscan", "on"),
    ("enable_sort", "on"),
    ("hnsw.ef_search", "40"),
    ("hnsw.iterative_scan", "off"),
    ("hnsw.max_scan_tuples", "20000"),
)
ITERATIVE_NAMES = {"off": "off", "strict_order": "strict", "relaxed_order": "relaxed"}
DEFAULT_MAX_SCAN = 20000
# With sorts off too, only the HNSW index gives the order without a sort.
FORCE_HNSW = {"enable_seqscan": "off", "enable_bitmapscan": "off", "enable_sort": "off"}


# One HNSW search setup: iterative scan mode, ef_search and max_scan_tuples.
@dataclass(frozen=True)
class HnswConfig:
    iterative: str  # off | strict_order | relaxed_order
    ef: int
    max_scan: int = DEFAULT_MAX_SCAN

    # Refuse a mode pgvector doesn't have.
    def __post_init__(self):
        if self.iterative not in ITERATIVE_NAMES:
            raise ValueError(f"iterative must be one of {', '.join(ITERATIVE_NAMES)}, got {self.iterative!r}")

    # Short name like off_ef40 or relaxed100k_ef40.
    @property
    def name(self) -> str:
        scan = "" if self.max_scan == DEFAULT_MAX_SCAN else f"{self.max_scan // 1000}k"
        return f"{ITERATIVE_NAMES[self.iterative]}{scan}_ef{self.ef}"

    # The three hnsw GUCs this config sets, values as strings.
    @property
    def gucs(self) -> dict[str, str]:
        return {
            "hnsw.ef_search": str(self.ef), 
            "hnsw.iterative_scan": self.iterative,
            "hnsw.max_scan_tuples": str(self.max_scan)
        }


HNSW_CONFIGS = (
    *(HnswConfig("off", ef) for ef in (40, 100, 200, 400, 1000)),
    *(HnswConfig("strict_order", ef) for ef in (40, 100, 200, 400)),
    *(HnswConfig("relaxed_order", ef) for ef in (40, 100, 200, 400)),
    *(HnswConfig("relaxed_order", ef, 100000) for ef in (40, 100)),
)


# One grid option: a forced strategy or a planner-free setting, with its full GUC set.
@dataclass(frozen=True)
class Option:
    name: str
    kind: str                           # forced | setting
    gucs: tuple[tuple[str, str], ...]   # all 7, in GUC_DEFAULTS order
    relaxed: bool                       # runs the relaxed query shape
    intended: str | None                # seq | bitmap | hnsw for forced, None for settings
    config: HnswConfig | None


# Build an option: the defaults with these overrides laid on top.
def _option(name: str, kind: str, overrides: dict[str, str], intended: str | None, config: HnswConfig | None) -> Option:
    known = {n for n, _ in GUC_DEFAULTS}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(f"{name}: unknown GUCs {', '.join(unknown)}")
    
    gucs = tuple((n, overrides.get(n, value)) for n, value in GUC_DEFAULTS)
    relaxed = config is not None and config.iterative == "relaxed_order"
    return Option(name, kind, gucs, relaxed, intended, config)


FORCED = (
    _option("exact_seq", "forced", {"enable_indexscan": "off", "enable_bitmapscan": "off"}, "seq", None),
    _option("exact_bitmap", "forced", {"enable_indexscan": "off", "enable_seqscan": "off"}, "bitmap", None),
    *(_option(f"hnsw_{c.name}", "forced", {**FORCE_HNSW, **c.gucs}, "hnsw", c) for c in HNSW_CONFIGS),
)
SETTINGS = (
    _option("set_default", "setting", HNSW_CONFIGS[0].gucs, None, HNSW_CONFIGS[0]),
    *(_option(f"set_{c.name}", "setting", c.gucs, None, c) for c in HNSW_CONFIGS[1:]),
    _option("set_noindex", "setting", {"enable_indexscan": "off"}, None, None),
)
ALL_OPTIONS = FORCED + SETTINGS


# Every option runs on every filter, except a bitmap scan with no filter to scan.
def applies(option: Option, filter_name: str) -> bool:
    return not (option.name == "exact_bitmap" and filter_name == "none")


# SET all 7 GUCs, then set work_mem or reset it to the server's value.
def set_statements(option: Option, work_mem: str | None = None) -> list[str]:
    stmts = [f"SET {name} = '{value}'" for name, value in option.gucs]
    stmts.append("RESET work_mem" if work_mem is None else f"SET work_mem = '{work_mem}'")
    return stmts


# The WHERE clause for a filter, or nothing for no filter.
def _where(filter_sql: str) -> str:
    return f" WHERE {filter_sql}" if filter_sql else ""


# The top-10 query; relaxed wraps it in a materialized CTE that sorts again.
# Postgres 17 needs the + 0, or it drops that outer sort.
def query_sql(table: str, filter_sql: str, relaxed: bool) -> str:
    if "%" in filter_sql:
        raise ValueError(f"filter can't contain % (it clashes with the query placeholders): {filter_sql}")
    plain = (f"SELECT id, embedding <=> %(q)s AS d FROM {table}{_where(filter_sql)} ORDER BY embedding <=> %(q)s LIMIT {vc.K}")
    return f"WITH c AS MATERIALIZED ({plain}) SELECT id, d FROM c ORDER BY d + 0" if relaxed else plain


# The planner's row estimate for a filter, read from its EXPLAIN.
def estimate_sql(table: str, filter_sql: str) -> str:
    return f"EXPLAIN (FORMAT JSON) SELECT 1 FROM {table}{_where(filter_sql)}"


# What a plan does: scan kind, the index it reads, and every node in order.
@dataclass(frozen=True)
class PlanShape:
    scan: str                # seq | bitmap | hnsw | btree_index | other
    index: str | None
    nodes: tuple[str, ...]   # "Node Type", or "Node Type:Index Name"


# Every node of a plan tree, parent before children.
def _preorder(node: dict):
    yield node
    for child in node.get("Plans", []):
        yield from _preorder(child)


# A node as "Node Type", with ":Index Name" when it reads an index.
def _node_label(node: dict) -> str:
    index = node.get("Index Name")
    return f"{node['Node Type']}:{index}" if index else node["Node Type"]


# Read an EXPLAIN (FORMAT JSON) result: the scan on the table decides the kind.
# The index is the scan's own, or the first under it (a bitmap's index scan).
def classify_plan(explain: list) -> PlanShape:
    nodes = list(_preorder(explain[0]["Plan"]))
    labels = tuple(_node_label(n) for n in nodes)
    scan = next((n for n in nodes if "Relation Name" in n), None)
    if scan is None:
        return PlanShape("other", None, labels)

    index = next((n["Index Name"] for n in _preorder(scan) if "Index Name" in n), None)
    node_type = scan["Node Type"]
    if node_type == "Seq Scan":
        kind = "seq"
    elif node_type == "Bitmap Heap Scan":
        kind = "bitmap"
    elif node_type == "Index Scan":
        kind = "hnsw" if (index or "").endswith("_hnsw_idx") else "btree_index"
    else:
        kind = "other"
    return PlanShape(kind, index, labels)


# The top node's estimated rows. Only means something on an estimate_sql plan;
# on a query plan it's just the LIMIT's 10.
def estimated_rows(explain: list) -> float:
    return float(explain[0]["Plan"]["Plan Rows"])


RUNS = 3
BAND_A = 0.005
MEMCHECK_CELLS = 20
MEMCHECK_WORK_MEM = "512MB"
OPTIONS = {o.name: o for o in ALL_OPTIONS}
SET_DEFAULT = OPTIONS["set_default"]
ITERATIVE = tuple(o for o in FORCED if o.config is not None and o.config.iterative != "off")
GRID_COLUMNS = [
    "size", "qid", "group", "half", "filter", "family", "n_pass", "share", "est_rows", "est_share",
    "option", "kind", "intended", "scan", "index", "nodes", "reused_from", "recall", "time_ms",
    "t1", "t2", "t3", "n_returned", "timed_out", "deterministic", "ids", "filter_sql",
]
MEMCHECK_COLUMNS = ["qid", "filter", "option", "ids_equal"]


# A query the server stopped at its statement_timeout.
class Timeout(Exception):
    pass


# Runs the grid's SQL on one autocommit connection.
class PgRunner:
    # Keep the connection the grid runs on.
    def __init__(self, conn):
        self.conn = conn

    # Run each SET or RESET statement in order.
    def set(self, stmts: list[str]) -> None:
        for s in stmts:
            self.conn.execute(s)

    # EXPLAIN (FORMAT JSON) of a query; an estimate_sql string is already an EXPLAIN.
    def explain(self, sql: str, params: dict) -> list:
        text = sql if sql.startswith("EXPLAIN") else f"EXPLAIN (FORMAT JSON) {sql}"
        return self.conn.execute(text, params or None).fetchone()[0]

    # The ids a query returns, in order; a timeout cancel raises Timeout.
    def run(self, sql: str, params: dict) -> list[int]:
        try:
            rows = self.conn.execute(sql, params).fetchall()
        except psycopg.errors.QueryCanceled as e:
            raise Timeout(str(e)) from None
        return [int(r[0]) for r in rows]


# One option's result on one cell.
@dataclass
class Measured:
    option: str
    ids: list[int]
    times_ms: list[float]
    timed_out: bool
    plan: PlanShape
    reused_from: str | None
    deterministic: bool


# The forced option a setting can copy: same scan kind, same hnsw config, same nodes.
def reuse_match(setting: Option, plan: PlanShape, forced: dict[str, Measured]) -> str | None:
    if plan.scan == "hnsw":
        name = None if setting.config is None else f"hnsw_{setting.config.name}"
    else:
        name = {"seq": "exact_seq", "bitmap": "exact_bitmap"}.get(plan.scan)
        
    twin = forced.get(name) if name else None
    return name if twin is not None and twin.plan.nodes == plan.nodes else None


# Run a query runs times, ms around execute and fetch; a timed-out run counts TIMEOUT_MS.
# Timed out only when most runs are; else ids come from the first finished run.
def _timed(runner, name: str, sql: str, params: dict, plan: PlanShape, timer: Callable, runs: int) -> Measured:
    results, times = [], []
    for _ in range(runs):
        start = timer()
        try:
            ids = runner.run(sql, params)
        except Timeout:
            times.append(vc.TIMEOUT_MS)
            continue
        times.append((timer() - start) * 1000.0)
        results.append(ids)

    if 2 * (runs - len(results)) > runs:
        return Measured(name, [], times, True, plan, None, False)
    return Measured(name, results[0], times, False, plan, None, all(r == results[0] for r in results))


# Every applicable option on one cell: forced ones shuffled, then settings shuffled.
# A setting whose plan matches a forced twin copies it instead of running.
def measure_cell(runner, table: str, q: np.ndarray, filter_sql: str, filter_name: str, rng, timer: Callable = time.perf_counter, runs: int = RUNS) -> list[Measured]:
    params = {"q": q}
    forced, done = {}, {}
    for group in (FORCED, SETTINGS):
        todo = [o for o in group if applies(o, filter_name)]
        for i in rng.permutation(len(todo)):
            o = todo[i]
            sql = query_sql(table, filter_sql, o.relaxed)
            runner.set(set_statements(o))
            plan = classify_plan(runner.explain(sql, params))
            twin = reuse_match(o, plan, forced) if o.kind == "setting" else None
            
            if twin is None:
                m = _timed(runner, o.name, sql, params, plan, timer, runs)
            else:
                t = forced[twin]
                m = Measured(o.name, list(t.ids), list(t.times_ms), t.timed_out, plan, twin, t.deterministic)
                
            if o.kind == "forced":
                forced[o.name] = m
            done[o.name] = m
    return [done[o.name] for o in ALL_OPTIONS if o.name in done]


# One query x filter cell: group, half, filter SQL, rows passing, true kth distance.
@dataclass(frozen=True)
class Cell:
    qid: str
    group: str
    half: str
    filter: str
    family: str
    sql: str
    n_pass: int
    share: float
    kth: float


# recall@10 of an option's ids against the cell's truth; a timeout scores 0.
def cell_recall(m: Measured, cell: Cell, base: np.ndarray, vec_row: np.ndarray, q: np.ndarray) -> float:
    if m.timed_out:
        return 0.0
    rows = vec_row[np.asarray(m.ids, dtype=np.int64)]
    return vc.recall_at_k(vc.cosine_dists(base[rows], q), cell.kth, cell.n_pass)


# One grid.csv row for an option's result on a cell.
def grid_row(size: str, cell: Cell, est_rows: float, table_rows: int, m: Measured, recall: float) -> list:
    o = OPTIONS[m.option]
    return [
        size, cell.qid, cell.group, cell.half, cell.filter, cell.family, cell.n_pass, cell.share,
        est_rows, est_rows / table_rows, m.option, o.kind, o.intended or "", m.plan.scan, m.plan.index or "",
        ">".join(m.plan.nodes), m.reused_from or "", recall, statistics.median(m.times_ms), *m.times_ms[:3],
        len(m.ids), m.timed_out, m.deterministic, " ".join(map(str, m.ids)), cell.sql
    ]


# Estimate a cell under default GUCs, measure every option, and build its rows.
def cell_rows(runner, size: str, table: str, cell: Cell, base: np.ndarray, vec_row: np.ndarray, q: np.ndarray, table_rows: int, rng, timer: Callable = time.perf_counter) -> list[list]:
    runner.set(set_statements(SET_DEFAULT))
    est_rows = estimated_rows(runner.explain(estimate_sql(table, cell.sql), {}))
    results = measure_cell(runner, table, q, cell.sql, cell.filter, rng, timer)
    return [grid_row(size, cell, est_rows, table_rows, m, cell_recall(m, cell, base, vec_row, q)) for m in results]


# A query count part1_queries refuses; exits 2 like a bad argument.
class QueriesError(ValueError):
    pass


# What the grid reads from disk, checked against each other.
@dataclass
class GridInputs:
    cells: list[Cell]                 # filters.csv order
    n_per_group: int
    table_rows: int
    vec_row: np.ndarray
    base: np.ndarray
    vectors: dict[str, np.ndarray]    # query vector per qid
    load_sha256: str                  # sha256 of pg_load.json, new on every table load


# A CSV as strings, empty cells kept as ""; missing columns raise naming the file.
def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (ValueError, OSError) as e:
        raise ValueError(f"{path}: can't read it as a CSV ({e})") from None
    
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {', '.join(missing)}")
    return df


# A vector file as float32 rows of width 384, every value finite.
def _vectors(path: Path) -> np.ndarray:
    try:
        v = np.asarray(np.load(path), dtype=np.float32)
    except (ValueError, TypeError, OSError, EOFError) as e:
        raise ValueError(f"{path}: can't read it as a numpy array ({e})") from None
    
    if v.ndim != 2 or v.shape[1] != vc.DIM or not np.isfinite(v).all():
        raise ValueError(f"{path}: want finite rows x {vc.DIM}, got shape {v.shape}")
    return v


# Each qid's vector: npy row i belongs to CSV row i.
def _query_vectors(npy: Path, frame: pd.DataFrame, csv_path: Path) -> dict[str, np.ndarray]:
    vectors = _vectors(npy)
    if len(vectors) != len(frame):
        raise ValueError(f"{npy}: {len(vectors):,} rows but {csv_path} has {len(frame):,}")
    return dict(zip(frame["qid"], vectors))


# The current query set's cells in filters.csv order, each checked against the truth.
def _cells(filters_path: Path, truth_path: Path, queries: list[tuple[str, str, str]]) -> list[Cell]:
    frame = _read_csv(filters_path, ["qid", "filter", "family", "sql", "n_pass", "share"])
    try:
        n_pass = frame["n_pass"].astype(np.int64).to_numpy()
        share = frame["share"].astype(float).to_numpy()
    except (ValueError, OverflowError):
        raise ValueError(f"{filters_path}: n_pass and share must be numbers") from None
    
    try:
        with np.load(truth_path) as z:
            t_qids, t_filters, t_pass, t_kth = z["qids"].tolist(), z["filters"].tolist(), z["n_pass"], z["kth"]
    except (KeyError, ValueError, OSError, EOFError) as e:
        raise ValueError(f"{truth_path}: can't read it ({e})") from None
    
    if t_pass.shape != (len(t_qids), len(t_filters)) or t_kth.shape != t_pass.shape:
        raise ValueError(f"{truth_path}: n_pass and kth must be queries x filters")

    q_index = {q: i for i, q in enumerate(t_qids)}
    f_index = {f: j for j, f in enumerate(t_filters)}
    meta = {qid: (group, half) for qid, group, half in queries}
    cells, seen = [], set()
    for qid, name, family, sql, n, s in zip(frame["qid"], frame["filter"], frame["family"], frame["sql"], n_pass, share):
        if qid not in meta:
            continue
        
        if (qid, name) in seen or name not in vf.FILTER_NAMES or "%" in sql:
            raise ValueError(f"{filters_path}: bad row for {qid} {name} (repeated, unknown filter, or a % in its SQL)")
        
        seen.add((qid, name))
        if qid not in q_index or name not in f_index:
            raise ValueError(f"{truth_path}: no truth for {qid} {name}")
        
        i, j = q_index[qid], f_index[name]
        if int(t_pass[i, j]) != int(n):
            raise ValueError(f"{truth_path}: {qid} {name} has {int(t_pass[i, j])} rows passing, {filters_path.name} says {int(n)}")
        group, half = meta[qid]
        cells.append(Cell(qid, group, half, name, family, sql, int(n), float(s), float(t_kth[i, j])))

    missing = [(q, f) for q, _, _ in queries for f in vf.FILTER_NAMES if (q, f) not in seen]
    if missing:
        raise ValueError(f"{filters_path}: no row for {missing[0][0]} {missing[0][1]} ({len(missing)} missing); rerun vector_truth.py with at least this many queries")
    return cells


# Read every input and check them against each other, before any connection.
# ValueError names the file; QueriesError is a query count part1_queries refuses.
def load_inputs(size: str, n_per_group: int | None, p) -> GridInputs:
    out = p.size_dir(size)
    filters_path, truth_path = out / "filters.csv", out / "truth_part1.npz"
    reviews_path, table_path = p.data / "query_reviews.csv", p.data / f"table_{size}.csv"
    review_npy, question_npy = p.embeddings / "query_reviews.npy", p.embeddings / "questions.npy"
    
    base_path = p.ai / "embeddings" / "test.npy" if size == "70k" else p.embeddings / "slice.npy"
    load_path = out / "pg_load.json"
    for path in (filters_path, truth_path, load_path, reviews_path, table_path, review_npy, question_npy, base_path):
        if not path.exists():
            raise ValueError(f"{path}: not found")

    reviews = _read_csv(reviews_path, ["qid", "split"])
    questions = vc.read_questions()
    test_qids = reviews.loc[reviews["split"] == "test", "qid"].tolist()
    written = questions.loc[questions["group"] == "written", "qid"].tolist()
    ai53 = questions.loc[questions["group"] == "ai53", "qid"].tolist()
    n = vf.PART1_PER_GROUP[size] if n_per_group is None else n_per_group
    try:
        queries = vf.part1_queries(test_qids, written, ai53, size, n_per_group=n)
    except ValueError as e:
        raise QueriesError(str(e)) from None

    cells = _cells(filters_path, truth_path, queries)
    vectors = {
        **_query_vectors(review_npy, reviews, reviews_path),
        **_query_vectors(question_npy, questions, vc.HERE / "questions.csv")
    }
    table, base = pgl.load_inputs(size, p)
    return GridInputs(cells, n, len(table), table["vec_row"].to_numpy(), base, {qid: vectors[qid] for qid, _, _ in queries}, hashlib.sha256(load_path.read_bytes()).hexdigest())


# Rows as CSV text with plain newlines.
def _csv_text(rows) -> str:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    return buf.getvalue()


# Write via a temp file and rename, so a stopped run leaves no half file.
def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# grid.csv's rows as strings, header first; a cut-off last line is dropped.
def _read_grid_rows(path: Path) -> list[list[str]]:
    lines = path.read_text().split("\n")
    lines.pop()  # "" after the last newline, or a line cut off mid-write
    rows = list(csv.reader(lines))
    if rows and rows[0] != GRID_COLUMNS:
        raise ValueError(f"{path}: columns differ from what this grid writes; rerun with --redo")
    return rows


# Option names that run on a filter.
def _applicable(filter_name: str) -> set[str]:
    return {o.name for o in ALL_OPTIONS if applies(o, filter_name)}


# True when a grid row matches this cell's size, rows passing, share and filter SQL.
def _same_inputs(row: list[str], cell: Cell, size: str, col: dict[str, int]) -> bool:
    try:
        return (row[col["size"]] == size and int(row[col["n_pass"]]) == cell.n_pass and float(row[col["share"]]) == cell.share and row[col["filter_sql"]] == cell.sql)
    except ValueError:
        return False


# Keep only complete, unchanged rows of current cells in grid.csv, group and half from now.
# Prints what it dropped and why, rewrites atomically, returns done cells and the dropped count.
def resume(path: Path, cells: list[Cell], size: str) -> tuple[set[tuple[str, str]], int]:
    if not path.exists():
        return set(), 0
    
    col = {name: i for i, name in enumerate(GRID_COLUMNS)}
    current = {(c.qid, c.filter): c for c in cells}
    rows = _read_grid_rows(path)[1:]
    
    found: dict[tuple[str, str], dict[str, list[str]]] = {}
    changed = set()
    for row in rows:
        if len(row) != len(GRID_COLUMNS):
            continue
        
        key = (row[col["qid"]], row[col["filter"]])
        found.setdefault(key, {}).setdefault(row[col["option"]], row)
        if key in current and not _same_inputs(row, current[key], size, col):
            changed.add(key)

    outside = {key for key in found if key not in current}
    incomplete = {key for key in found if key in current and key not in changed and set(found[key]) != _applicable(key[1])}
    done = set(found) - outside - changed - incomplete
    kept = []
    for key in (k for k in found if k in done):
        for row in found[key].values():
            row = list(row)
            row[col["group"]], row[col["half"]] = current[key].group, current[key].half
            kept.append(row)
            
    _write_atomic(path, _csv_text([GRID_COLUMNS, *kept]))
    print(f"resume: kept {len(done)} cells, dropped {len(found) - len(done)} cells ({len(rows) - len(kept):,} rows): {len(outside)} outside the current cell set, {len(incomplete)} incomplete, {len(changed)} changed inputs")
    return done, len(found) - len(done)


# The grid runs grid_info.json records; none for a missing or older file.
def _history(path: Path) -> list[dict]:
    try:
        info = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    runs = info.get("history") if isinstance(info, dict) else None
    return [r for r in runs if isinstance(r, dict)] if isinstance(runs, list) else []


# The N of a finished grid on this table load other than n, else None.
def _finished_other(history: list[dict], n: int, load_sha256: str) -> int | None:
    found = [r["n_per_group"] for r in history if r.get("complete") is True and r.get("load_sha256") == load_sha256 and isinstance(r.get("n_per_group"), int) and r["n_per_group"] != n]
    return found[-1] if found else None


# True when grid.csv has rows but grid_info.json doesn't vouch for this table load.
def _table_reloaded(out: Path, load_sha256: str) -> bool:
    grid = out / "grid.csv"
    if not grid.exists() or not any(grid.read_text().split("\n")[1:]):
        return False
    
    try:
        info = json.loads((out / "grid_info.json").read_text())
    except (OSError, ValueError):
        return True
    return not isinstance(info, dict) or info.get("load_sha256") != load_sha256


# Append one cell's rows to grid.csv, header first if the file is new.
def _append(path: Path, rows: list[list]) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        if new:
            writer.writerow(GRID_COLUMNS)
        writer.writerows(rows)


# Every grid.csv row's ids, by (qid, filter, option).
def _grid_ids(path: Path) -> dict[tuple[str, str, str], list[int]]:
    if not path.exists():
        return {}
    rows = _read_grid_rows(path)
    col = {name: i for i, name in enumerate(GRID_COLUMNS)}
    return {(r[col["qid"]], r[col["filter"]], r[col["option"]]): [int(x) for x in r[col["ids"]].split()] for r in rows[1:] if len(r) == len(GRID_COLUMNS)}


# Run every applicable option once on seeded cells, untimed, keeping nothing.
def warm_up(runner, table: str, cells: list[Cell], vectors: dict, n: int, size: str) -> None:
    rng = np.random.default_rng(vc.seed_for(size, "warmup"))
    for i in rng.choice(len(cells), size=min(n, len(cells)), replace=False):
        c = cells[i]
        for o in ALL_OPTIONS:
            if not applies(o, c.filter):
                continue
            
            runner.set(set_statements(o))
            try:
                runner.run(query_sql(table, c.sql, o.relaxed), {"q": vectors[c.qid]})
            except Timeout:
                pass


# The iterative forced options again at 512MB work_mem, on seeded band A cells.
# Each row says whether the ids equal the grid's; GUCs go back to defaults after.
def memcheck(runner, table: str, cells: list[Cell], vectors: dict, grid_ids: dict, size: str) -> list[tuple]:
    band_a = [c for c in cells if c.share < BAND_A]
    rng = np.random.default_rng(vc.seed_for(size, "memcheck"))
    picks = rng.choice(len(band_a), size=min(MEMCHECK_CELLS, len(band_a)), replace=False) if band_a else []
    rows = []
    for i in picks:
        c = band_a[i]
        for o in ITERATIVE:
            runner.set(set_statements(o, work_mem=MEMCHECK_WORK_MEM))
            try:
                ids = runner.run(query_sql(table, c.sql, o.relaxed), {"q": vectors[c.qid]})
            except Timeout:
                ids = []
                
            rows.append((c.qid, c.filter, o.name, ids == grid_ids.get((c.qid, c.filter, o.name))))
    runner.set(set_statements(SET_DEFAULT))
    return rows


# Check the server, warm up, resume, measure the cells left, then check memory.
# history is the earlier runs to keep; this run's entry goes on the end.
def _run(conn, runner, args, inputs: GridInputs, out: Path, timer: Callable, restart: bool, history: list[dict]) -> int:
    bad = pgl.check_server_settings(pgl.show_settings(conn))
    for line in bad:
        print(f"error: server setting {line}", file=sys.stderr)
    if bad:
        return 1
    
    table = pgl.table_name(args.size)
    rows = int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
    if rows != inputs.table_rows:
        print(f"error: {table} has {rows:,} rows but table_{args.size}.csv has {inputs.table_rows:,}", file=sys.stderr)
        return 1

    grid_path, memo_path, info_path = out / "grid.csv", out / "memcheck.csv", out / "grid_info.json"
    if args.redo:
        for path in (grid_path, memo_path, info_path):
            path.unlink(missing_ok=True)
            
    if restart:
        print("table reloaded since grid.csv was written (pg_load.json changed): starting the grid over", flush=True)
        grid_path.unlink(missing_ok=True)
        memo_path.unlink(missing_ok=True)
        
    entry = {
        "n_per_group": inputs.n_per_group, "warmup": args.warmup, "runs": RUNS, "load_sha256": inputs.load_sha256,
        "kept": 0, "dropped": 0, "measured": 0, "complete": False
    }
    info = {
        "size": args.size, "n_per_group": inputs.n_per_group, "cells": len(inputs.cells),
        "warmup": args.warmup, "runs": RUNS, **pgl.versions(conn), "load_sha256": inputs.load_sha256,
        "history": [*history, entry]
    }

    # Write grid_info.json with this run's entry as it stands.
    def write_info() -> None:
        _write_atomic(info_path, json.dumps(info, indent=2) + "\n")

    write_info()
    try:
        pgl.prewarm(conn, args.size)
        start = timer()
        warm_up(runner, table, inputs.cells, inputs.vectors, args.warmup, args.size)
        print(f"warm-up: {min(args.warmup, len(inputs.cells))} cells in {timer() - start:.1f} s", flush=True)

        done, dropped = resume(grid_path, inputs.cells, args.size)
        entry.update(kept=len(done), dropped=dropped)
        order = np.random.default_rng(vc.seed_for(args.size, "cells")).permutation(len(inputs.cells))
        todo = [c for c in (inputs.cells[i] for i in order) if (c.qid, c.filter) not in done]
        print(f"grid: {len(todo)} cells to measure, {len(done)} already done", flush=True)

        start = timer()
        for k, cell in enumerate(todo, 1):
            rng = np.random.default_rng(vc.seed_for(args.size, "order", cell.qid, cell.filter))
            _append(grid_path, cell_rows(runner, args.size, table, cell, inputs.base, inputs.vec_row, inputs.vectors[cell.qid], inputs.table_rows, rng, timer))
            entry["measured"] = k
            elapsed = timer() - start
            if k == args.projection_cells:
                print(f"projected grid time: {elapsed / k * len(todo) / 60:.1f} min for {len(todo)} cells", flush=True)
            elif k % 100 == 0:
                print(f"cells: {k} of {len(todo)} in {elapsed / 60:.1f} min", flush=True)
        if todo:
            elapsed = timer() - start
            print(f"grid: measured {len(todo)} cells in {elapsed:.1f} s ({elapsed / len(todo):.2f} s per cell)")

        checks = memcheck(runner, table, inputs.cells, inputs.vectors, _grid_ids(grid_path), args.size)
        _write_atomic(memo_path, _csv_text([MEMCHECK_COLUMNS, *checks]))
        entry["complete"] = True
        print(f"memcheck: {sum(c[3] for c in checks)} of {len(checks)} equal over {len(checks) // len(ITERATIVE)} cells")
    finally:
        write_info()
    return 0


# argparse type: a whole number at least lo.
def _at_least(lo: int) -> Callable[[str], int]:
    # The parsed number, or an argparse error.
    def number(text: str) -> int:
        value = int(text)
        if value < lo:
            raise argparse.ArgumentTypeError(f"must be at least {lo}, got {value}")
        return value
    return number


# Parse the command line, run the grid and the memory check. Returns the exit code.
def main(argv: list[str] | None = None, connect: Callable = pgl.connect, make_runner: Callable = PgRunner, timer: Callable = time.perf_counter) -> int:
    parser = argparse.ArgumentParser(description="Time every grid option on every query x filter cell.")
    parser.add_argument("--size", required=True, choices=vc.SIZES)
    limits = ", ".join(f"{n} at {size}" for size, n in vf.PART1_PER_GROUP.items())
    
    parser.add_argument("--queries", type=int, default=None, help=f"first N filtered-search queries per group; by default all of them ({limits})")
    parser.add_argument("--warmup", type=_at_least(0), default=50, help="cells run once, untimed, before timing")
    parser.add_argument("--projection-cells", type=_at_least(1), default=20, help="print a projected total after this many cells")
    parser.add_argument("--redo", action="store_true", help="delete grid.csv, memcheck.csv and grid_info.json first")
    parser.add_argument("--results", type=Path, default=vc.RESULTS)
    parser.add_argument("--ai", type=Path, default=vc.AI_ROOT)
    args = parser.parse_args(argv)

    p = vc.paths(args.results, args.ai)
    out = p.size_dir(args.size)
    ai = p.ai.resolve()
    if out.resolve() == ai or ai in out.resolve().parents:
        print(f"error: won't write inside the ai_queries folder {p.ai}", file=sys.stderr)
        return 1

    try:
        inputs = load_inputs(args.size, args.queries, p)
        restart = not args.redo and _table_reloaded(out, inputs.load_sha256)
        if (out / "grid.csv").exists() and not args.redo and not restart:
            _read_grid_rows(out / "grid.csv")
    except QueriesError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    # N may change only before the grid finishes: a new N picks a new split.
    old = _history(out / "grid_info.json")
    finished = None if args.redo else _finished_other(old, inputs.n_per_group, inputs.load_sha256)
    if finished is not None:
        print(f"error: the grid finished at N = {finished} queries per group; changing --queries now would pick a new split after the results exist (pass --redo to start over)", file=sys.stderr)
        return 2

    try:
        conn = connect()
    except psycopg.OperationalError as e:
        print(f"error: can't connect to Postgres on port {vc.PG_PORT}: {e}", file=sys.stderr)
        return 1
    try:
        return _run(conn, make_runner(conn), args, inputs, out, timer, restart, [] if args.redo or restart else old)
    except KeyboardInterrupt:
        print("grid: stopped by Ctrl-C; finished cells are kept in grid.csv", file=sys.stderr)
        return 130
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
