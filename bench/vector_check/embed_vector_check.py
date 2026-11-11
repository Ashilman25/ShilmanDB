#Embed the held-out query reviews, the questions and the 500K slice for the vector search check
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections.abc import Callable, Iterable
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

BATCH = 128
MAX_DRIFT = 1e-4
QUERY_VECTORS = ("query_reviews.npy", "questions.npy")


# Load the sentence model. Imported here, so the tests never need sentence-transformers.
def load_model(name: str = vc.MODEL):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


# Rows as float32, each divided by its length; a zero row stays zero.
def unit_rows(vectors) -> np.ndarray:
    out = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-12)


# Review vectors in input order, made with the same call as the table vectors.
def embed_reviews(texts: list[str], model, progress: bool = False) -> np.ndarray:
    return unit_rows(model.encode(texts, batch_size=BATCH, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=progress))


# Question vectors through encode_query when the model has it; no prefix either way.
def embed_questions(texts: list[str], model) -> np.ndarray:
    if hasattr(model, "encode_query"):
        return unit_rows(model.encode_query(texts, convert_to_numpy=True, normalize_embeddings=True))
    return embed_reviews(texts, model)


# Largest absolute difference between two arrays of the same shape.
def drift(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"can't compare arrays of shape {a.shape} and {b.shape}")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


# Every value a string, so a text like "NA" stays "NA".
def _read_strings(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


# A whole-number column as int64; a ValueError names the file if it isn't one.
def _int_column(frame: pd.DataFrame, column: str, path: Path) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"{path} has no {column} column")
    try:
        return frame[column].astype("int64").to_numpy()
    except (TypeError, ValueError) as e:
        raise ValueError(f"{path}: the {column} column isn't all whole numbers") from e


# The first path that doesn't exist, if any.
def _missing(paths: Iterable[Path]) -> Path | None:
    return next((p for p in paths if not p.exists()), None)


# Write via a temp file and a rename, so a stopped run leaves no half file.
def _write_atomic(path: Path, write: Callable) -> None:
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            write(f)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# Save an array as .npy, atomically.
def _save_npy(path: Path, array: np.ndarray) -> None:
    _write_atomic(path, lambda f: np.save(f, array))


# Save a dict as indented JSON, atomically.
def _save_json(path: Path, obj: dict) -> None:
    _write_atomic(path, lambda f: f.write((json.dumps(obj, indent=2) + "\n").encode("utf-8")))


# The saved vector row of each qid, from the ai_queries questions.csv.
def _saved_question_rows(path: Path, qids: list[str], saved_count: int) -> list[int]:
    saved = _read_strings(path)
    rows = dict(zip(saved["qid"], _int_column(saved, "row", path).tolist()))
    missing = [q for q in qids if q not in rows]
    if missing:
        raise ValueError(f"{path} has no row for {', '.join(missing)}")
    
    out = [rows[q] for q in qids]
    past = [r for r in out if not 0 <= r < saved_count]
    if past:
        raise ValueError(f"{path}: row {past[0]} is past the {saved_count} saved question vectors")
    return out


# Embed the query reviews and questions; write them only if both drifts pass.
def _run_queries(p, model) -> int:
    reviews_path, check_path = p.data / "query_reviews.csv", p.data / "check_texts_70k.csv"
    saved_dir = p.ai / "embeddings"
    if not reviews_path.exists():
        print(f"error: {reviews_path} not found; run prepare_vector_data.py queries first", file=sys.stderr)
        return 1
    
    if not check_path.exists():
        print(f"error: {check_path} not found; run prepare_vector_data.py table --size 70k first", file=sys.stderr)
        return 1
    
    missing = _missing(saved_dir / n for n in ("questions.csv", "questions.npy", "test.npy"))
    if missing is not None:
        print(f"error: {missing} not found", file=sys.stderr)
        return 1

    # Check every input before the model loads, so a bad file fails at once.
    reviews = _read_strings(reviews_path)
    check = _read_strings(check_path)
    check_rows = _int_column(check, "vec_row", check_path)
    questions = vc.read_questions()
    ai53 = np.flatnonzero((questions["group"] == "ai53").to_numpy())
    if len(ai53) == 0 or len(check_rows) == 0:
        raise ValueError("nothing to check: no ai53 questions or no check texts")
    
    saved_questions = np.load(saved_dir / "questions.npy")
    saved_rows = _saved_question_rows(saved_dir / "questions.csv", questions["qid"].iloc[ai53].tolist(), len(saved_questions))
    saved_texts = np.load(saved_dir / "test.npy", mmap_mode="r")
    if check_rows.min() < 0 or check_rows.max() >= len(saved_texts):
        raise ValueError(f"{check_path}: a vec_row is outside the {len(saved_texts):,} saved table vectors")

    started = time.monotonic()
    if model is None:
        model = load_model()
        
    loaded = time.monotonic()
    review_vectors = embed_reviews(reviews["input"].tolist(), model, progress=True)
    question_vectors = embed_questions(questions["text"].tolist(), model)
    check_vectors = embed_reviews(check["input"].tolist(), model)
    done = time.monotonic()

    question_drift = drift(question_vectors[ai53], saved_questions[saved_rows])
    review_drift = drift(check_vectors, saved_texts[check_rows])
    passed = question_drift <= MAX_DRIFT and review_drift <= MAX_DRIFT
    print(f"question drift: {question_drift:.3g} ({len(ai53)} ai53 questions against {saved_dir / 'questions.npy'})")
    print(f"review drift: {review_drift:.3g} ({len(check_rows)} table texts against {saved_dir / 'test.npy'})")

    info = {
        "model": vc.MODEL,
        "device": str(getattr(model, "device", "")),
        "counts": {
            "query_reviews": len(review_vectors), 
            "questions": len(question_vectors),
            "ai53_questions": len(ai53), 
            "check_texts": len(check_rows)
        },
        "question_drift": question_drift,
        "review_drift": review_drift,
        "max_drift": MAX_DRIFT,
        "passed": passed,
        "load_seconds": round(loaded - started, 3),
        "seconds": round(done - loaded, 3),
    }
    p.embeddings.mkdir(parents=True, exist_ok=True)
    if not passed:
        for name in QUERY_VECTORS:
            (p.embeddings / name).unlink(missing_ok=True)
            
        _save_json(p.embeddings / "embed_info.json", info)
        print(f"error: drift over {MAX_DRIFT:g}, so my embedding rule doesn't match the saved vectors; no vectors written", file=sys.stderr)
        return 1

    # query_reviews.npy goes last: a rerun treats it as the sign this command finished.
    _save_npy(p.embeddings / "questions.npy", question_vectors)
    _save_json(p.embeddings / "embed_info.json", info)
    _save_npy(p.embeddings / "query_reviews.npy", review_vectors)
    print(f"query_reviews.npy: {len(review_vectors):,} rows, questions.npy: {len(question_vectors)} rows, {done - loaded:.1f} s embedding")
    return 0


# Embed the slice texts and write slice.npy in vec_row order.
def _run_slice(p, model) -> int:
    path = p.data / "slice_texts.csv"
    if not path.exists():
        print(f"error: {path} not found; run prepare_vector_data.py table --size 500k first", file=sys.stderr)
        return 1
    
    texts = _read_strings(path)
    rows = _int_column(texts, "vec_row", path)
    order = np.argsort(rows, kind="stable")
    if len(rows) == 0 or not np.array_equal(rows[order], np.arange(len(rows))):
        print(f"error: {path}: vec_row isn't 0..{len(rows) - 1} with every row once", file=sys.stderr)
        return 1

    if model is None:
        model = load_model()
        
    started = time.monotonic()
    vectors = embed_reviews(texts["input"].to_numpy()[order].tolist(), model, progress=True)
    seconds = time.monotonic() - started
    p.embeddings.mkdir(parents=True, exist_ok=True)
    _save_npy(p.embeddings / "slice.npy", vectors)
    print(f"slice.npy: {len(vectors):,} rows in {seconds:.1f} s")
    return 0


# Parse the command line, run one command and return its exit code.
# model stands in for the real one in tests; None loads it.
def main(argv: list[str] | None = None, model=None) -> int:
    parser = argparse.ArgumentParser(description="Embed the query reviews, the questions and the slice texts.")
    sub = parser.add_subparsers(dest="command", required=True)
    
    queries = sub.add_parser("queries", help="embed the query reviews and questions, after the drift check")
    queries.add_argument("--results", type=Path, default=vc.RESULTS)
    queries.add_argument("--ai", type=Path, default=vc.AI_ROOT)
    
    slice_cmd = sub.add_parser("slice", help="embed slice_texts.csv into slice.npy")
    slice_cmd.add_argument("--results", type=Path, default=vc.RESULTS)
    args = parser.parse_args(argv)

    p = vc.paths(args.results, getattr(args, "ai", vc.AI_ROOT))
    ai, out = p.ai.resolve(), p.embeddings.resolve()
    if out == ai or ai in out.parents:
        print(f"error: won't write inside the ai_queries folder {p.ai}", file=sys.stderr)
        return 1
    
    start = time.monotonic()
    try:
        code = _run_queries(p, model) if args.command == "queries" else _run_slice(p, model)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    
    if code == 0:
        print(f"done in {time.monotonic() - start:.1f} s")
    return code


if __name__ == "__main__":
    sys.exit(main())
