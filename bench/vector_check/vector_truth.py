#Exact top-10 truth, by brute force, for every filtered query cell and every unfiltered query
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from collections.abc import Callable
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
vf = vc.load_sibling("vc_filters", "vector_filters.py")

BATCH = 256


# The k nearest rows that pass m: ids, distances, rows passing, kth distance.
# Pads to k with -1 and inf; kth is inf when nothing passes.
def cell_truth(d_rows: np.ndarray, m: np.ndarray, k: int = vc.K) -> tuple[np.ndarray, np.ndarray, int, float]:
    d_rows, m = np.asarray(d_rows), np.asarray(m)
    if m.dtype != bool or m.shape != d_rows.shape:
        raise ValueError("m must be a bool mask the same shape as d_rows")

    passing = np.flatnonzero(m)
    found_ids, found_d = vc.topk(d_rows, k, passing)
    ids = np.full(k, -1, dtype=np.int64)
    dists = np.full(k, np.inf)
    ids[:len(found_ids)] = found_ids
    dists[:len(found_d)] = found_d
    kth = float(found_d[-1]) if len(found_d) else float("inf")
    return ids, dists, len(passing), kth


# The exact k nearest base rows per query, one float32 matmul per batch.
# Ties go to the smaller id; a short base pads with -1 and inf.
def exact_topk_batch(base: np.ndarray, queries: np.ndarray, k: int = vc.K, batch: int = BATCH) -> tuple[np.ndarray, np.ndarray]:
    if batch < 1:
        raise ValueError(f"batch must be at least 1, got {batch}")
    base = np.asarray(base, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)

    ids = np.full((len(queries), k), -1, dtype=np.int64)
    dists = np.full((len(queries), k), np.inf)
    for start in range(0, len(queries), batch):
        block = queries[start:start + batch]
        block_d = np.ascontiguousarray(vc.cosine_dists(base, block.T).T)
        
        for j, d in enumerate(block_d):
            found_ids, found_d = vc.topk(d, k)
            ids[start + j, :len(found_ids)] = found_ids
            dists[start + j, :len(found_d)] = found_d
            
    return ids, dists


# Truth for every filtered query cell, plus filters.csv rows in the same order.
def part1_truth(t, base: np.ndarray, vectors: dict, queries: list, size: str, k: int = vc.K) -> tuple[dict, pd.DataFrame]:
    names = list(vf.FILTER_NAMES)
    shape = (len(queries), len(names))
    n_pass = np.zeros(shape, dtype=np.int64)
    top_ids = np.full((*shape, k), -1, dtype=np.int64)
    top_dists = np.full((*shape, k), np.inf)
    kth = np.full(shape, np.inf)

    rows = []
    for i, (qid, group, half) in enumerate(queries):
        d_rows = vc.cosine_dists(base, vectors[qid])[t.vec_row]
        near = int(t.product_id[vc.topk(d_rows, 1)[0][0]])
        filters = {f.name: f for f in vf.make_filters(t, near, vf.query_rng(size, qid))}
        for j, name in enumerate(names):
            f = filters[name]
            top_ids[i, j], top_dists[i, j], n_pass[i, j], kth[i, j] = cell_truth(d_rows, vf.mask(t, f), k)
            rows.append((qid, group, half, name, f.family, vf.to_sql(f), int(n_pass[i, j])))

    frame = pd.DataFrame(rows, columns=["qid", "group", "half", "filter", "family", "sql", "n_pass"])
    frame["share"] = frame["n_pass"] / t.n_rows
    arrays = {
        "qids": np.array([q for q, _, _ in queries], dtype=str),
        "groups": np.array([g for _, g, _ in queries], dtype=str),
        "halves": np.array([h for _, _, h in queries], dtype=str),
        "filters": np.array(names, dtype=str),
        "n_pass": n_pass, "top_ids": top_ids, "top_dists": top_dists, "kth": kth,
    }
    return arrays, frame


# Unfiltered truth over the distinct vectors: query reviews first, then questions.
def part2_truth(base: np.ndarray, reviews: pd.DataFrame, review_vectors: np.ndarray, questions: pd.DataFrame, question_vectors: np.ndarray, k: int = vc.K) -> dict:
    ids, dists = exact_topk_batch(base, np.concatenate([review_vectors, question_vectors]), k)
    last = min(k, len(base)) - 1
    return {
        "qids": np.array(reviews["qid"].tolist() + questions["qid"].tolist(), dtype=str),
        "sets": np.array(reviews["split"].tolist() + ["questions"] * len(questions), dtype=str),
        "top_ids": ids, "top_dists": dists, "kth": dists[:, last].copy(),
    }


# Write via a temp file and rename, so a stopped run leaves no half file.
def _write_atomic(path: Path, write: Callable) -> None:
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            write(f)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# Print min, median and max share, and min and max rows passing, per filter.
def _print_shares(frame: pd.DataFrame) -> None:
    print(f"{'filter':<16}{'min share':>11}{'median':>11}{'max share':>11}{'min n_pass':>12}{'max n_pass':>12}")
    for name in vf.FILTER_NAMES:
        cells = frame[frame["filter"] == name]
        s, n = cells["share"], cells["n_pass"]
        print(f"{name:<16}{s.min():>11.3%}{s.median():>11.3%}{s.max():>11.3%}{int(n.min()):>12,}{int(n.max()):>12,}")


# A vector file as float32 rows; None, after printing why, if its shape is wrong.
def _load_vectors(path: Path, width: int | None = None) -> np.ndarray | None:
    vectors = np.asarray(np.load(path), dtype=np.float32)
    if vectors.ndim != 2 or (width is not None and vectors.shape[1] != width):
        print(f"error: {path} has shape {vectors.shape}, want rows x {width or 'n'}", file=sys.stderr)
        return None
    return vectors


# Check every input, then write filters.csv and both truth files. Returns the exit code.
def _run(size: str, n_per_group: int | None, p) -> int:
    table_path = p.data / f"table_{size}.csv"
    reviews_path = p.data / "query_reviews.csv"
    review_npy, question_npy = p.embeddings / "query_reviews.npy", p.embeddings / "questions.npy"
    base_path = p.ai / "embeddings" / "test.npy" if size == "70k" else p.embeddings / "slice.npy"
    
    for path in (table_path, reviews_path, review_npy, question_npy, base_path):
        if not path.exists():
            print(f"error: {path} not found", file=sys.stderr)
            return 1

    base = _load_vectors(base_path)
    review_vectors = None if base is None else _load_vectors(review_npy, base.shape[1])
    question_vectors = None if review_vectors is None else _load_vectors(question_npy, base.shape[1])
    if question_vectors is None:
        return 1

    reviews = pd.read_csv(reviews_path, dtype=str, keep_default_na=False)
    questions = vc.read_questions()
    for npy, vectors, csv, frame in ((review_npy, review_vectors, reviews_path, reviews), (question_npy, question_vectors, vc.HERE / "questions.csv", questions)):
        if len(vectors) != len(frame):
            print(f"error: {npy} has {len(vectors):,} rows but {csv} has {len(frame):,}", file=sys.stderr)
            return 1

    t = vf.TableArrays.from_csv(table_path)
    distinct = np.unique(t.vec_row)
    if len(distinct) == 0 or distinct[0] != 0 or distinct[-1] != len(distinct) - 1:
        print(f"error: {table_path}: vec_row must be 0..n-1 with no gaps", file=sys.stderr)
        return 1
    
    if len(base) != len(distinct):
        print(f"error: {base_path} has {len(base):,} rows but {table_path} has {len(distinct):,} distinct vec_row values", file=sys.stderr)
        return 1

    test_qids = reviews.loc[reviews["split"] == "test", "qid"].tolist()
    written = questions.loc[questions["group"] == "written", "qid"].tolist()
    ai53 = questions.loc[questions["group"] == "ai53", "qid"].tolist()
    try:
        queries = vf.part1_queries(test_qids, written, ai53, size, n_per_group=n_per_group)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    vectors = dict(zip(reviews["qid"], review_vectors))
    vectors.update(zip(questions["qid"], question_vectors))
    out = p.size_dir(size)
    out.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    arrays, frame = part1_truth(t, base, vectors, queries, size)
    part1_s = time.monotonic() - started
    started = time.monotonic()
    part2 = part2_truth(base, reviews, review_vectors, questions, question_vectors)
    part2_s = time.monotonic() - started

    # I write only after both finish, so a stopped run keeps the old files.
    _write_atomic(out / "filters.csv", lambda f: f.write(frame.to_csv(index=False).encode("utf-8")))
    _write_atomic(out / "truth_part1.npz", lambda f: np.savez(f, **arrays))
    _write_atomic(out / "truth_part2.npz", lambda f: np.savez(f, **part2))
    print(f"truth_part1.npz: {len(queries)} queries x {len(vf.FILTER_NAMES)} filters over {t.n_rows:,} rows, {part1_s:.1f} s")
    print(f"truth_part2.npz: {len(part2['qids']):,} queries over {len(base):,} distinct vectors, {part2_s:.1f} s")
    _print_shares(frame)
    return 0


# Parse the command line, write the truth and return the exit code.
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the exact top-10 truth for the vector search check.")
    parser.add_argument("--size", required=True, choices=vc.SIZES)
    limits = ", ".join(f"{n} at {size}" for size, n in vf.PART1_PER_GROUP.items())
    parser.add_argument("--queries", type=int, default=None, help=f"first N filtered-search queries per group; at most {limits} (the default)")
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
        code = _run(args.size, args.queries, p)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if code == 0:
        print(f"done in {time.monotonic() - start:.1f} s")
    return code


if __name__ == "__main__":
    sys.exit(main())
