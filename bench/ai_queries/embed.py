#Embed the reviews and the question wordings with a small sentence model.
import argparse
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ai_input, paths, text_key
from .data_prep import read_tables
from .questions import QUESTIONS

TIMING_COLUMNS = ["pass", "rows", "seconds", "rows_per_sec"]


@lru_cache(maxsize=2)
def _model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _unit(vectors) -> np.ndarray:
    out = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-12)


def embed_texts(texts: list[str], model_name: str, batch: int = 128) -> np.ndarray:
    vectors = _model(model_name).encode(texts, batch_size=batch, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    return _unit(vectors)


def embed_queries(texts: list[str], model_name: str, prefix: str = "") -> np.ndarray:
    model = _model(model_name)
    if hasattr(model, "encode_query"):
        return _unit(model.encode_query(texts, convert_to_numpy=True, normalize_embeddings=True))
    return embed_texts([prefix + t for t in texts], model_name)


def distinct_inputs(reviews: pd.DataFrame) -> tuple[list[str], list[str]]:
    keys: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
    for key, title, text in zip(reviews["text_key"], reviews["title"], reviews["text"]):
        if key not in seen:
            seen.add(key)
            keys.append(key)
            texts.append(ai_input(title, text))
    return keys, texts


def record_timing(path: Path, name: str, rows: int, seconds: float) -> None:
    path = Path(path)
    old = pd.read_csv(path, dtype={"pass": str}) if path.exists() else pd.DataFrame(columns=TIMING_COLUMNS)
    old = old[old["pass"] != name]
    new = pd.DataFrame([{
        "pass": name, 
        "rows": rows, 
        "seconds": seconds, 
        "rows_per_sec": rows / seconds if seconds > 0 else 0.0
    }])
    pd.concat([old, new], ignore_index=True)[TIMING_COLUMNS].to_csv(path, index=False, lineterminator="\n")


def _timed_embed(texts: list[str], model_name: str) -> tuple[np.ndarray, float]:
    _model(model_name)  # load first, so the time is encoding only
    started = time.perf_counter()
    vectors = embed_texts(texts, model_name)
    return vectors, time.perf_counter() - started


def _cmd_test(args: argparse.Namespace) -> None:
    p = paths(Path(args.root))
    p.embeddings.mkdir(parents=True, exist_ok=True)
    reviews, _ = read_tables(p.data)
    keys, texts = distinct_inputs(reviews)
    
    print(f"embedding {len(texts)} distinct reviews with {args.model}", flush=True)
    vectors, seconds = _timed_embed(texts, args.model)
    np.save(p.embeddings / "test.npy", vectors)
    pd.DataFrame({"row": range(len(keys)), "text_key": keys}).to_csv(p.embeddings / "test_keys.csv", index=False, lineterminator="\n")
    
    qids = list(QUESTIONS)
    np.save(p.embeddings / "questions.npy", embed_queries([QUESTIONS[q].wording for q in qids], args.model, args.query_prefix))
    pd.DataFrame({"row": range(len(qids)), "qid": qids}).to_csv(p.embeddings / "questions.csv", index=False, lineterminator="\n")
    
    record_timing(p.embeddings / "timing.csv", "test", len(texts), seconds)
    print(f"done: {len(texts)} rows in {seconds:.1f} s ({len(texts) / seconds:.1f} rows/s)")


def _cmd_slice(args: argparse.Namespace) -> None:
    p = paths(Path(args.root))
    p.embeddings.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(p.data / "slice" / "reviews.csv", dtype=str, keep_default_na=False)
    df["text_key"] = [text_key(t, x) for t, x in zip(df["title"], df["text"])]
    
    keys, texts = distinct_inputs(df)
    if args.limit:
        texts = texts[:args.limit]
        
    print(f"embedding {len(texts)} distinct slice reviews with {args.model}", flush=True)
    _, seconds = _timed_embed(texts, args.model)
    record_timing(p.embeddings / "timing.csv", "slice", len(texts), seconds)
    print(f"done: {len(texts)} rows in {seconds:.1f} s ({len(texts) / seconds:.1f} rows/s)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("test", help="embed the test table and the question wordings")
    a.add_argument("--root", default=str(paths().root))
    a.add_argument("--model", required=True)
    a.add_argument("--query-prefix", default="", help="added to each wording if the model has no query prompt")
    a.set_defaults(func=_cmd_test)
    b = sub.add_parser("slice-timing", help="time the embedding of the 500K slice")
    b.add_argument("--root", default=str(paths().root))
    b.add_argument("--model", required=True)
    b.add_argument("--limit", type=int, default=None, help="only the first N rows (for a quick try)")
    b.set_defaults(func=_cmd_slice)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
