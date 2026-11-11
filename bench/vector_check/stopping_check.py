#Learned stopping study: REM over the ef grid, plain-search trajectories, the DARTH model and its replay
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


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
hs = vc.load_sibling("vc_hnsw_search", "hnsw_search.py")

EF_GRID = (10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 700, 1000, 1500, 2000)
E_CHOICES = (500, 1000, 2000)
E_RECALL = 0.995
TARGETS = (0.90, 0.95, 0.99)
MARGINS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)
FALLBACK_MARGIN = 0.05
LOG_EVERY = 8
MAX_TRAIN_ROWS = 3_000_000
TREES = 100
LEARNING_RATE = 0.1
TREE_CHARGE = 1 / 25
DETERMINISM_QUERIES = 20
DETERMINISM_EF = 100

REVIEW_SETS = ("train", "validation", "test")
SCORED_SETS = ("validation", "test", "questions")
TRAJ_SETS = (*REVIEW_SETS, "questions")
METHODS = ("rem", "learned", "plain")
TRAJ_KEYS = ("qids", "offsets", "ndis", "features", "recall", "final_ndis", "final_recall")
STOPPING_COLUMNS = ("set", "target", "method", "param", "margin_ok", "mean_recall", "mean_ndis", "mean_calls", "work_0x", "work_1x", "work_4x", "below_target_share", "n_queries")
QUERY_COLUMNS = ("set", "qid", "target", "method", "recall", "ndis", "calls")


# The smallest E choice whose validation mean recall reaches E_RECALL, else None.
def choose_E(val_means: dict[int, float]) -> int | None:
    for e in E_CHOICES:
        if e in val_means and val_means[e] >= E_RECALL:
            return e
    return None


# The smallest ef whose validation mean recall reaches the target, else None.
def rem_choice(val_means: dict[int, float], target: float) -> int | None:
    for ef in sorted(val_means):
        if val_means[ef] >= target:
            return ef
    return None


# Distances until the next predictor call: DARTH's adaptive interval, floored at mpi and 8.
def interval(ipi: int, mpi: int, target: float, predicted: float) -> int:
    return int(max(mpi, LOG_EVERY, round(mpi + (ipi - mpi) * (target - float(predicted)))))


# Replay stopping on one logged search: recall, ndis at the stop, predictor calls.
def replay_one(ndis: np.ndarray, preds: np.ndarray, recall: np.ndarray, final_ndis: int, final_recall: float, target: float, margin: float, ipi: int, mpi: int) -> tuple[float, int, int]:
    next_call = ipi
    calls = 0
    for j in range(len(ndis)):
        if ndis[j] < next_call:
            continue
        calls += 1
        
        if preds[j] >= target + margin:
            return float(recall[j]), int(ndis[j]), calls
        next_call = ndis[j] + interval(ipi, mpi, target, preds[j])
    return float(final_recall), int(final_ndis), calls


# The smallest margin whose validation mean recall reaches the target, else None.
def choose_margin(per_margin_mean_recall: dict[float, float], target: float) -> float | None:
    for m in sorted(per_margin_mean_recall):
        if per_margin_mean_recall[m] >= target:
            return m
    return None


# Work in distance computations; each predictor call costs trees / 25, times factor.
def work(ndis: float, calls: float, factor: float = 1.0, trees: int = TREES) -> float:
    return ndis + calls * trees * TREE_CHARGE * factor


# Recalls as exact multiples of 1/K; a logged float32 0.9 becomes 0.9 again.
def exact_recall(recall) -> np.ndarray:
    return np.rint(np.asarray(recall, dtype=np.float64) * vc.K) / vc.K


# Mean recall from hit counts, so it compares exactly with a target like 0.95.
def mean_recall(recalls) -> float:
    hits = np.rint(np.asarray(recalls, dtype=np.float64) * vc.K)
    return float(hits.sum() / (len(hits) * vc.K)) if len(hits) else math.nan


# Keep checkpoints 0, j, 2j of each query; j is the smallest step fitting cap.
# Returns the kept rows, the new offsets and j.
def thin_rows(offsets: np.ndarray, cap: int = MAX_TRAIN_ROWS) -> tuple[np.ndarray, np.ndarray, int]:
    offsets = np.asarray(offsets, dtype=np.int64)
    total = int(offsets[-1])
    counts = np.diff(offsets)

    # Rows kept at step j; fewer as j grows. No j below ceil(total/cap) fits.
    def kept(j: int) -> int:
        return int(((counts + j - 1) // j).sum())

    lo = max(1, -(-total // cap))
    if kept(lo) > cap:
        hi = int(counts.max())
        if kept(hi) > cap:
            raise ValueError(f"{int((counts > 0).sum()):,} queries with checkpoints can't fit a cap of {cap:,} rows")
        while lo < hi:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if kept(mid) <= cap else (mid + 1, hi)
            
    step = lo
    pos = np.arange(total) - np.repeat(offsets[:-1], counts)
    keep = np.flatnonzero(pos % step == 0)
    new_offsets = np.concatenate([[0], np.cumsum((counts + step - 1) // step)]).astype(np.int64)
    return keep, new_offsets, step


# Per query and target, the first logged ndis reaching it, else the final ndis.
def reach_ndis(offsets: np.ndarray, ndis: np.ndarray, recall: np.ndarray, final_ndis: np.ndarray, targets: tuple = TARGETS) -> np.ndarray:
    rec = exact_recall(recall)
    out = np.empty((len(final_ndis), len(targets)), dtype=np.int64)
    for i in range(len(final_ndis)):
        a, b = int(offsets[i]), int(offsets[i + 1])
        for t_i, target in enumerate(targets):
            hit = np.flatnonzero(rec[a:b] >= target)
            out[i, t_i] = ndis[a + hit[0]] if len(hit) else final_ndis[i]
            
    return out


# One set's logged plain searches: per-checkpoint rows, offsets per query, final values.
@dataclass(eq=False)
class Traj:
    qids: np.ndarray
    offsets: np.ndarray
    ndis: np.ndarray
    features: np.ndarray
    recall: np.ndarray
    final_ndis: np.ndarray
    final_recall: np.ndarray


# Replay one target and margin over a set: recall, ndis and calls per query.
def replay_set(traj: Traj, preds: np.ndarray, target: float, margin: float, ipi: int, mpi: int):
    ndis, rec = traj.ndis.tolist(), exact_recall(traj.recall).tolist()
    pred, offsets = np.asarray(preds, dtype=np.float64).tolist(), traj.offsets.tolist()
    final_ndis, final_recall = traj.final_ndis.tolist(), exact_recall(traj.final_recall).tolist()
    out = []
    for i in range(len(final_ndis)):
        a, b = offsets[i], offsets[i + 1]
        out.append(replay_one(ndis[a:b], pred[a:b], rec[a:b], final_ndis[i], final_recall[i], target, margin, ipi, mpi))
        
    recalls, stops, calls = zip(*out) if out else ((), (), ())
    return np.array(recalls, dtype=np.float64), np.array(stops, dtype=np.int64), np.array(calls, dtype=np.int64)


# The margin for one target from validation replays, whether one reached it, and the means.
# None reaching the target gives FALLBACK_MARGIN and False.
def margin_for(traj: Traj, preds: np.ndarray, target: float, ipi: int, mpi: int) -> tuple[float, bool, dict]:
    means = {m: mean_recall(replay_set(traj, preds, target, m, ipi, mpi)[0]) for m in MARGINS}
    margin = choose_margin(means, target)
    return (FALLBACK_MARGIN, False, means) if margin is None else (margin, True, means)


# One stopping.csv row; margin_ok None (rem and plain rows) writes an empty cell.
def summary_row(set_name: str, target: float, method: str, param, margin_ok: bool | None, recalls, ndis, calls) -> dict:
    recalls = exact_recall(recalls)
    mean_ndis, mean_calls = float(np.mean(ndis)), float(np.mean(calls))
    return {
        "set": set_name, "target": target, "method": method, "param": param,
        "margin_ok": "" if margin_ok is None else ("true" if margin_ok else "false"),
        "mean_recall": mean_recall(recalls), "mean_ndis": mean_ndis, "mean_calls": mean_calls,
        "work_0x": work(mean_ndis, mean_calls, 0), "work_1x": work(mean_ndis, mean_calls, 1),
        "work_4x": work(mean_ndis, mean_calls, 4),
        "below_target_share": float(np.mean(recalls < target)), "n_queries": len(recalls),
    }


# Truth rows, kth distances, the rows picked per set, and the files workers load.
@dataclass(eq=False)
class Inputs:
    qids: np.ndarray
    kth: np.ndarray
    rows: dict
    init_args: tuple


# The base vectors' file: 70k reads the ai_queries test embeddings, 500k the slice.
def base_path_for(p, size: str) -> Path:
    return p.ai / "embeddings" / "test.npy" if size == "70k" else p.embeddings / "slice.npy"


# A .npy file memory-mapped; ValueError naming it if it won't open.
def _open_npy(path: Path) -> np.ndarray:
    try:
        return np.load(path, mmap_mode="r")
    except (ValueError, TypeError, OSError, EOFError) as e:
        raise ValueError(f"{path}: can't read it as a numpy array ({e})") from None


# Check truth, csvs and vectors line up, then pick each set's rows in truth order.
# limits keeps the first n of each review set; questions are never cut.
def load_inputs(p, size: str, limits: dict) -> Inputs:
    size_dir = p.size_dir(size)
    truth_path = size_dir / "truth_part2.npz"
    reviews_path, questions_path = p.data / "query_reviews.csv", vc.HERE / "questions.csv"
    review_npy, question_npy = p.embeddings / "query_reviews.npy", p.embeddings / "questions.npy"
    
    base_path, graph_dir = base_path_for(p, size), size_dir / "graph"
    for path in (truth_path, reviews_path, review_npy, question_npy, base_path, graph_dir / "graph.json"):
        if not path.exists():
            raise ValueError(f"{path}: not found")

    with np.load(truth_path) as z:
        missing = [key for key in ("qids", "sets", "kth") if key not in z.files]
        if missing:
            raise ValueError(f"{truth_path}: missing {', '.join(missing)}")
        qids, sets, kth = z["qids"], z["sets"], z["kth"]
        
    reviews = pd.read_csv(reviews_path, dtype=str, keep_default_na=False)
    questions = vc.read_questions(questions_path)
    if qids.tolist() != reviews["qid"].tolist() + questions["qid"].tolist():
        raise ValueError(f"{truth_path}: qids differ from {reviews_path}'s then {questions_path}'s")
    
    if sets.tolist() != reviews["split"].tolist() + ["questions"] * len(questions):
        raise ValueError(f"{truth_path}: sets differ from {reviews_path}'s split column then 'questions'")
    
    if len(kth) != len(qids):
        raise ValueError(f"{truth_path}: kth has {len(kth):,} rows but qids has {len(qids):,}")

    base = _open_npy(base_path)
    if base.ndim != 2 or base.dtype != np.float32 or not base.flags.c_contiguous:
        raise ValueError(f"{base_path}: want 2-D float32 C-order rows, got {base.shape} {base.dtype}")
    
    for npy, frame, csv in ((review_npy, reviews, reviews_path), (question_npy, questions, questions_path)):
        vectors = _open_npy(npy)
        if vectors.ndim != 2 or len(vectors) != len(frame):
            raise ValueError(f"{npy}: {len(vectors):,} rows but {csv} has {len(frame):,} rows")
        
        if vectors.shape[1] != base.shape[1]:
            raise ValueError(f"{npy}: width {vectors.shape[1]} but {base_path} has width {base.shape[1]}")

    rows = {}
    for name in REVIEW_SETS:
        found = np.flatnonzero(sets == name)
        rows[name] = found[:limits[name]] if limits else found
        
    rows["questions"] = np.flatnonzero(sets == "questions")
    init_args = (str(graph_dir), str(base_path), str(review_npy), str(question_npy))
    return Inputs(qids=qids, kth=kth, rows=rows, init_args=init_args)


_STATE: dict = {}


# Load the graph and query vectors in this process, memory-mapped so workers share pages.
def _init_worker(graph_dir: str, base_path: str, review_npy: str, question_npy: str) -> None:
    base = np.load(base_path, mmap_mode="r")
    _STATE.clear()
    _STATE.update(searcher=hs.Searcher(hs.load_graph(Path(graph_dir), base)), n_base=len(base), reviews=np.load(review_npy, mmap_mode="r"), questions=np.load(question_npy, mmap_mode="r"))


# A truth row's query vector: query reviews first, then questions.
def _vector(row: int) -> np.ndarray:
    reviews = _STATE["reviews"]
    return reviews[row] if row < len(reviews) else _STATE["questions"][row - len(reviews)]


# One query at every ef on the grid: (recall, ndis) per ef.
def _rem_query(task: tuple) -> list[tuple[float, int]]:
    row, kth = task
    searcher, q, n = _STATE["searcher"], _vector(row), _STATE["n_base"]
    out = []
    for ef in EF_GRID:
        r = searcher.search(q, ef)
        out.append((vc.recall_at_k(r.dists, kth, n_pass=n), r.ndis))
    return out


# The plain search at ef with its log, plus its final ndis and recall.
def _traj_query(task: tuple) -> tuple:
    row, kth, ef = task
    r = _STATE["searcher"].search(_vector(row), ef, log_every=LOG_EVERY, kth_true=kth)
    return r.log.ndis, r.log.features, r.log.recall, r.ndis, vc.recall_at_k(r.dists, kth, n_pass=_STATE["n_base"])


# fn over tasks, results in task order: inline for one job, else a spawn pool.
# Each worker loads the graph once in its initializer.
def _run_tasks(fn: Callable, tasks: list, jobs: int, init_args: tuple) -> list:
    if jobs == 1:
        _init_worker(*init_args)
        return [fn(t) for t in tasks]
    
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(jobs, initializer=_init_worker, initargs=init_args) as pool:
        return list(pool.imap(fn, tasks, chunksize=max(1, len(tasks) // (jobs * 8))))


# Write via a temp file and rename, so a stopped run leaves no half file.
def _write_atomic(path: Path, write: Callable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            write(f)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# A frame as CSV, written atomically.
def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    _write_atomic(path, lambda f: f.write(frame.to_csv(index=False).encode("utf-8")))


# A dict as indented JSON, written atomically.
def _write_json(path: Path, data: dict) -> None:
    _write_atomic(path, lambda f: f.write((json.dumps(data, indent=2) + "\n").encode("utf-8")))


# A JSON file this script wrote earlier; its limits must equal this run's.
def _read_upstream(path: Path, limits: dict) -> dict:
    if not path.exists():
        raise ValueError(f"{path}: not found")
    
    data = json.loads(path.read_text())
    if data.get("limits") != limits:
        raise ValueError(f"{path}: limits {data.get('limits')} differ from this run's --limit {limits}")
    return data


# rem.csv as {(qid, ef): (recall, ndis)}; every given qid needs a row at every ef.
def read_rem(path: Path, qids) -> dict:
    if not path.exists():
        raise ValueError(f"{path}: not found; run rem first")

    frame = pd.read_csv(path, dtype={"set": str, "qid": str})
    missing = [c for c in ("qid", "ef", "recall", "ndis") if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {', '.join(missing)}")
    
    table = {(q, int(ef)): (float(r), int(n)) for q, ef, r, n in zip(frame["qid"], frame["ef"], frame["recall"], frame["ndis"])}
    for q in qids:
        for ef in EF_GRID:
            if (q, ef) not in table:
                raise ValueError(f"{path} has no row for {q} at ef {ef}; rerun rem with the same --limit")
            
    return table


# Validation mean recall per ef, from rem.csv's rows for the picked validation queries.
def _val_means(rem: dict, val_qids) -> dict[int, float]:
    return {ef: mean_recall([rem[(q, ef)][0] for q in val_qids]) for ef in EF_GRID}


# A traj file as Traj, plus its raw arrays; its qids must be this run's.
def read_traj(path: Path, qids, extra: tuple = ()) -> tuple[Traj, dict]:
    if not path.exists():
        raise ValueError(f"{path}: not found; run trajectories first")
    
    with np.load(path) as z:
        missing = [key for key in (*TRAJ_KEYS, *extra) if key not in z.files]
        if missing:
            raise ValueError(f"{path}: missing {', '.join(missing)}")
        arrays = {key: z[key] for key in z.files}
        
    if arrays["qids"].tolist() != list(qids):
        raise ValueError(f"{path}: its queries differ from this run's; rerun trajectories with the same --limit")
    
    offsets, m = arrays["offsets"], len(arrays["ndis"])
    if len(offsets) != len(qids) + 1 or offsets[-1] != m or len(arrays["features"]) != m or len(arrays["recall"]) != m:
        raise ValueError(f"{path}: offsets, ndis, features and recall don't line up")
    return Traj(**{key: arrays[key] for key in TRAJ_KEYS}), arrays


# Search the first validation queries twice in this process, every one once, then again.
# same_ndis needs both runs and rem.csv's ef 100 ndis to agree.
def _determinism(inp: Inputs, rem: pd.DataFrame) -> pd.DataFrame:
    _init_worker(*inp.init_args)
    searcher = _STATE["searcher"]
    picks = inp.rows["validation"][:DETERMINISM_QUERIES]
    runs = [[searcher.search(_vector(int(r)), DETERMINISM_EF) for r in picks] for _ in range(2)]
    at_ef = rem[rem["ef"] == DETERMINISM_EF].set_index("qid")["ndis"]
    out = []
    
    for i, r in enumerate(picks):
        qid, a, b = str(inp.qids[r]), runs[0][i], runs[1][i]
        out.append((qid, bool(np.array_equal(a.ids, b.ids)), bool(a.ndis == b.ndis == int(at_ef[qid]))))
    return pd.DataFrame(out, columns=["qid", "same_ids", "same_ndis"])


# Every validation, test and question query at every ef; then the determinism check.
def cmd_rem(args, p, limits: dict) -> int:
    inp = load_inputs(p, args.size, limits)
    picked = [(name, int(r)) for name in SCORED_SETS for r in inp.rows[name]]
    started = time.monotonic()
    results = _run_tasks(_rem_query, [(r, float(inp.kth[r])) for _, r in picked], args.jobs, inp.init_args)
    search_s = time.monotonic() - started

    records = [(name, str(inp.qids[r]), ef, recall, ndis) for (name, r), per_ef in zip(picked, results) for ef, (recall, ndis) in zip(EF_GRID, per_ef)]
    rem = pd.DataFrame(records, columns=["set", "qid", "ef", "recall", "ndis"])
    print(f"rem: {len(picked):,} queries x {len(EF_GRID)} ef = {len(rem):,} searches, total ndis {int(rem['ndis'].sum()):,}, search {search_s:.1f} s, jobs {args.jobs}")

    det = _determinism(inp, rem)
    size_dir = p.size_dir(args.size)
    _write_csv(size_dir / "rem.csv", rem)
    _write_csv(size_dir / "determinism.csv", det)
    print(f"determinism: {len(det)} queries at ef {DETERMINISM_EF}, same ids {int(det['same_ids'].sum())}, same ndis {int(det['same_ndis'].sum())}")
    return 0


# One set's npz arrays from its per-query search results.
def _traj_arrays(qids: list, results: list) -> dict:
    counts = [len(nd) for nd, *_ in results]
    width = len(hs.FEATURES)
    return {
        "qids": np.array(qids, dtype=str),
        "offsets": np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        "ndis": np.concatenate([nd for nd, *_ in results] or [np.zeros(0)]).astype(np.int32),
        "features": np.concatenate([f for _, f, *_ in results] or [np.zeros((0, width))]).astype(np.float32).reshape(-1, width),
        "recall": np.concatenate([r for _, _, r, *_ in results] or [np.zeros(0)]).astype(np.float32),
        "final_ndis": np.array([res[3] for res in results], dtype=np.int64),
        "final_recall": np.array([res[4] for res in results], dtype=np.float64),
    }


# Pick E, log the plain search at E for every set, check it, write traj/.
def cmd_trajectories(args, p, limits: dict) -> int:
    inp = load_inputs(p, args.size, limits)
    size_dir = p.size_dir(args.size)
    scored_qids = [str(inp.qids[r]) for name in SCORED_SETS for r in inp.rows[name]]
    rem = read_rem(size_dir / "rem.csv", scored_qids)
    val_recall = _val_means(rem, [str(q) for q in inp.qids[inp.rows["validation"]]])
    e = choose_E(val_recall)
    
    _write_json(size_dir / "plain_E.json", {
        "E": e, 
        "val_recall": {str(ef): v for ef, v in val_recall.items()},
        "limits": limits
    })
    if e is None:
        means = ", ".join(f"{c}: {val_recall[c]:.4f}" for c in E_CHOICES)
        raise ValueError(f"no E in {E_CHOICES} reaches validation recall {E_RECALL} ({means})")

    traj_dir = size_dir / "traj"
    if traj_dir.exists():
        for old in traj_dir.glob("*.npz"):
            old.unlink()
            
    picked = [(name, int(r)) for name in TRAJ_SETS for r in inp.rows[name]]
    started = time.monotonic()
    results = _run_tasks(_traj_query, [(r, float(inp.kth[r]), e) for _, r in picked], args.jobs, inp.init_args)
    search_s = time.monotonic() - started
    total = sum(res[3] for res in results)
    print(f"trajectories: E {e}, {len(picked):,} queries searched, total ndis {total:,}, search {search_s:.1f} s, jobs {args.jobs}")

    by_set = {name: ([], []) for name in TRAJ_SETS}
    for (name, r), res in zip(picked, results):
        by_set[name][0].append(str(inp.qids[r]))
        by_set[name][1].append(res)
        
    bad = [(q, res[3], rem[(q, e)][1]) for name in SCORED_SETS for q, res in zip(*by_set[name]) if res[3] != rem[(q, e)][1]]
    if bad:
        q, got, want = bad[0]
        raise ValueError(f"{len(bad)} plain searches at ef {e} differ from {size_dir / 'rem.csv'}'s ndis (first: {q}, {got} against {want})")
    print(f"check: {sum(len(by_set[n][0]) for n in SCORED_SETS):,} plain searches match rem.csv's ndis at ef {e}")

    for name in TRAJ_SETS:
        arrays = _traj_arrays(*by_set[name])
        if name == "train":
            arrays["reach"] = reach_ndis(arrays["offsets"], arrays["ndis"], arrays["recall"], arrays["final_ndis"])
            logged = len(arrays["ndis"])
            keep, arrays["offsets"], step = thin_rows(arrays["offsets"], MAX_TRAIN_ROWS)
            for key in ("ndis", "features", "recall"):
                arrays[key] = arrays[key][keep]
                
            arrays["thin_every"] = np.array(step, dtype=np.int64)
            print(f"thinning: kept {len(keep):,} of {logged:,} training checkpoints (every {step}), cap {MAX_TRAIN_ROWS:,}")
            
        _write_atomic(traj_dir / f"{name}.npz", lambda f, a=arrays: np.savez(f, **a))
        print(f"traj/{name}.npz: {len(arrays['qids']):,} queries, {len(arrays['ndis']):,} checkpoints")
    return 0


# Fit DARTH's recall regressor: 100 trees, learning rate 0.1, LightGBM's other defaults.
def train_model(x: np.ndarray, y: np.ndarray):
    import lightgbm
    params = {
        "objective": "regression", 
        "learning_rate": LEARNING_RATE, 
        "seed": vc.SEED, 
        "deterministic": True, 
        "verbose": -1
    }
    return lightgbm.train(params, lightgbm.Dataset(x, y), num_boost_round=TREES)


# Predicted recall per set from the saved model; rows are independent, so threads don't matter.
def predict_sets(model_path: Path, features: dict, jobs: int) -> dict:
    import lightgbm

    booster = lightgbm.Booster(model_file=str(model_path))
    return {name: booster.predict(x, num_threads=jobs) if len(x) else np.zeros(0) for name, x in features.items()}


# Fit the model on training checkpoints; save it with dists_t, ipi, mpi per target.
def cmd_train(args, p, limits: dict) -> int:
    inp = load_inputs(p, args.size, limits)
    size_dir = p.size_dir(args.size)
    plain = _read_upstream(size_dir / "plain_E.json", limits)
    if plain.get("E") is None:
        raise ValueError(f"{size_dir / 'plain_E.json'}: E is null, so there's nothing to train on")
    traj, arrays = read_traj(size_dir / "traj" / "train.npz", inp.qids[inp.rows["train"]].tolist(), ("reach", "thin_every"))

    started = time.monotonic()
    booster = train_model(traj.features, traj.recall)
    model_path = size_dir / "model" / "darth.txt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = model_path.with_name(f"{model_path.name}.tmp{os.getpid()}")
    
    try:
        booster.save_model(str(tmp))
        os.replace(tmp, model_path)
    finally:
        tmp.unlink(missing_ok=True)
    train_s = time.monotonic() - started

    targets = {}
    for t_i, target in enumerate(TARGETS):
        dists_t = float(np.mean(arrays["reach"][:, t_i]))
        targets[str(target)] = {"dists_t": dists_t, "ipi": max(8, round(dists_t / 2)), "mpi": max(8, round(dists_t / 10))}
        
    info = {
        "rows": len(traj.ndis), "queries": len(traj.qids), 
        "thin_every": int(arrays["thin_every"]), "E": plain["E"],
        "trees": TREES, "learning_rate": LEARNING_RATE, 
        "features": list(hs.FEATURES), "limits": limits,
        "targets": targets
    }
    
    _write_json(size_dir / "train_info.json", info)
    print(f"train: 0 queries searched (fit on {len(traj.qids):,} training queries' logs), {len(traj.ndis):,} rows, total ndis 0, {train_s:.1f} s, jobs: LightGBM's own threads")
    for target, t in targets.items():
        print(f"  target {target}: dists_t {t['dists_t']:.1f}, ipi {t['ipi']}, mpi {t['mpi']}")
    return 0


# Pick margins on validation, replay every scored set, write stopping.csv and stopping_queries.csv.
def cmd_replay(args, p, limits: dict) -> int:
    inp = load_inputs(p, args.size, limits)
    size_dir = p.size_dir(args.size)
    plain = _read_upstream(size_dir / "plain_E.json", limits)
    info = _read_upstream(size_dir / "train_info.json", limits)
    
    e = plain.get("E")
    if e is None:
        raise ValueError(f"{size_dir / 'plain_E.json'}: E is null, so there's nothing to replay")
    if info.get("E") != e:
        raise ValueError(f"{size_dir / 'train_info.json'} has E {info.get('E')} but {size_dir / 'plain_E.json'} has E {e}; rerun train after trajectories")
    
    model_path = size_dir / "model" / "darth.txt"
    if not model_path.exists():
        raise ValueError(f"{model_path}: not found; run train first")
    
    qids = {name: [str(q) for q in inp.qids[inp.rows[name]]] for name in SCORED_SETS}
    rem = read_rem(size_dir / "rem.csv", [q for name in SCORED_SETS for q in qids[name]])
    trajs = {name: read_traj(size_dir / "traj" / f"{name}.npz", qids[name])[0] for name in SCORED_SETS}

    started = time.monotonic()
    preds = predict_sets(model_path, {name: t.features for name, t in trajs.items()}, args.jobs)
    val_means = _val_means(rem, qids["validation"])
    choices = {}
    
    for target in TARGETS:
        key = str(target)
        if key not in info.get("targets", {}):
            raise ValueError(f"{size_dir / 'train_info.json'}: no entry for target {key}")
        
        ipi, mpi = int(info["targets"][key]["ipi"]), int(info["targets"][key]["mpi"])
        ef = rem_choice(val_means, target)
        if ef is None:
            raise ValueError(f"no ef on the grid reaches validation recall {target}")
        
        margin, ok, means = margin_for(trajs["validation"], preds["validation"], target, ipi, mpi)
        choices[target] = (ef, margin, ok, ipi, mpi)
        print(f"target {target}: rem ef {ef}, margin {margin} ({'ok' if ok else 'none reached the target'}); validation recall per margin " + ", ".join(f"{m}: {v:.4f}" for m, v in means.items()))

    rows, per_query, learned_ndis = [], [], 0
    for name in SCORED_SETS:
        traj = trajs[name]
        for target in TARGETS:
            ef, margin, ok, ipi, mpi = choices[target]
            rem_rows = [rem[(q, ef)] for q in qids[name]]
            results = {
                "rem": (ef, None, [r for r, _ in rem_rows], [n for _, n in rem_rows], [0] * len(rem_rows)),
                "learned": (margin, ok, *replay_set(traj, preds[name], target, margin, ipi, mpi)),
                "plain": (e, None, traj.final_recall, traj.final_ndis, [0] * len(traj.qids)),
            }
            
            for method in METHODS:
                param, margin_ok, recalls, ndis, calls = results[method]
                rows.append(summary_row(name, target, method, param, margin_ok, recalls, ndis, calls))
                per_query += [(name, q, target, method, r, int(n), int(c)) for q, r, n, c in zip(qids[name], exact_recall(recalls), ndis, calls)]
            learned_ndis += int(np.sum(results["learned"][3]))
    replay_s = time.monotonic() - started

    # Object columns keep param as written: an int ef or E, a float margin.
    _write_csv(size_dir / "stopping.csv", pd.DataFrame(rows, columns=list(STOPPING_COLUMNS), dtype=object))
    _write_csv(size_dir / "stopping_queries.csv", pd.DataFrame(per_query, columns=list(QUERY_COLUMNS)))
    n = sum(len(qids[name]) for name in SCORED_SETS)
    print(f"replay: 0 queries searched, {n:,} queries replayed x {len(TARGETS)} targets, total learned ndis {learned_ndis:,}, {replay_s:.1f} s, jobs {args.jobs} (predictor threads)")
    return 0


COMMANDS = {
    "rem": cmd_rem, 
    "trajectories": cmd_trajectories,
    "train": cmd_train, 
    "replay": cmd_replay
}


# Three whole numbers of at least 1, for --limit TRAIN,VAL,TEST.
def _limit(text: str) -> dict:
    parts = text.split(",")
    try:
        values = [int(x) for x in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(f"wants TRAIN,VAL,TEST whole numbers, got {text!r}") from None
    
    if len(values) != 3 or min(values) < 1:
        raise argparse.ArgumentTypeError(f"wants three whole numbers of at least 1, got {text!r}")
    return dict(zip(REVIEW_SETS, values))


# A whole number of at least 1, for --jobs.
def _jobs(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a whole number, got {text!r}") from None
    
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


# Parse the command line and run one step. Returns the exit code.
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Learned stopping study for the vector search check.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    helps = {
        "rem": "every scored query at every ef, and the determinism check",
        "trajectories": "pick E and log the plain search at E for every set",
        "train": "fit the recall model on training checkpoints",
        "replay": "pick margins on validation and score every set"
    }
    
    for name, text in helps.items():
        sp = sub.add_parser(name, help=text)
        sp.add_argument("--size", required=True, choices=vc.SIZES)
        if name != "train":
            sp.add_argument("--jobs", type=_jobs, default=max(1, (os.cpu_count() or 1) - 2))
            
        sp.add_argument("--limit", type=_limit, default=None, help="first TRAIN,VAL,TEST queries of each review set")
        sp.add_argument("--results", type=Path, default=vc.RESULTS)
        sp.add_argument("--ai", type=Path, default=vc.AI_ROOT)
    args = parser.parse_args(argv)

    p = vc.paths(args.results, args.ai)
    ai, out = p.ai.resolve(), p.size_dir(args.size).resolve()
    if out == ai or ai in out.parents:
        print(f"error: won't write inside the ai_queries folder {p.ai}", file=sys.stderr)
        return 1

    start = time.monotonic()
    try:
        code = COMMANDS[args.cmd](args, p, args.limit or {})
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    
    print(f"{args.cmd}: done in {time.monotonic() - start:.1f} s")
    return code


if __name__ == "__main__":
    sys.exit(main())
