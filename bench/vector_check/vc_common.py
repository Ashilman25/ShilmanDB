"""Shared constants, paths, text rules, questions and exact-search helpers for the vector search check."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = REPO_ROOT / "bench/results/ai_queries"
RESULTS = REPO_ROOT / "bench/results/vector_check"

SIZES = ("70k", "500k")
K = 10
SEED = 1
DIM = 384
MODEL = "BAAI/bge-small-en-v1.5"
PG_PORT = 5433
PG_HOST = "/tmp"
PG_DB = "vector_check"
TIE_TOL = 1e-5
TIMEOUT_MS = 10000.0


# Load a script next to this one by path, under a name in sys.modules.
def load_sibling(module_name: str, file_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, HERE / file_name)
    if spec is None or spec.loader is None:
        raise ImportError(f"can't load {HERE / file_name}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


@dataclass(frozen=True)
class Paths:
    results: Path
    ai: Path

    @property
    def data(self) -> Path:
        return self.results / "data"

    @property
    def embeddings(self) -> Path:
        return self.results / "embeddings"

    @property
    def pgdata(self) -> Path:
        return self.results / "pgdata"

    @property
    def pg_log(self) -> Path:
        return self.results / "pg.log"

    @property
    def report(self) -> Path:
        return self.results / "report.md"

    # Folder for one table size's results.
    def size_dir(self, size: str) -> Path:
        return self.results / f"size{size}"


# Paths under the given results and ai_queries folders.
def paths(results: Path = RESULTS, ai: Path = AI_ROOT) -> Paths:
    return Paths(Path(results), Path(ai))


# Return the size if it's one I run, else raise ValueError.
def check_size(size: str) -> str:
    if size not in SIZES:
        raise ValueError(f"size must be one of {', '.join(SIZES)}, got {size!r}")
    return size


# Lowercase title and text with all whitespace collapsed to single spaces.
def normalize_text(title, text) -> str:
    return " ".join(f"{title or ''} {text or ''}".lower().split())


# Same text, same key: sha1 of the normalized title and text.
def text_key(title, text) -> str:
    return hashlib.sha1(normalize_text(title, text).encode("utf-8")).hexdigest()


# The exact string a review is embedded from.
def review_input(title, text) -> str:
    return f"Title: {title or ''}\nReview: {text or ''}"


# The query questions as qid, group, text; every value a string, file order kept.
def read_questions(path: Path = HERE / "questions.csv") -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)[["qid", "group", "text"]]


# A stable 64-bit seed from SEED and the given parts.
def seed_for(*parts) -> int:
    joined = ":".join(map(str, (SEED, *parts)))
    return int(hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16], 16)


# Cosine distance from q to every unit vector, as a float32 matmul.
def cosine_dists(vectors: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 1.0 - (vectors @ np.asarray(q, dtype=np.float32)).astype(np.float64)


# The k nearest ids and distances, ascending; ties go to the smaller id.
# candidates limits the search to those row ids.
def topk(dists: np.ndarray, k: int, candidates: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    dists = np.asarray(dists)
    if candidates is None:
        ids = np.arange(len(dists), dtype=np.int64)
    else:
        if np.asarray(candidates).dtype == bool:
            raise TypeError("candidates must be row ids, not a mask")
        ids = np.asarray(candidates, dtype=np.int64)
        
    sub = dists[ids]
    if k < len(sub):
        # Keep everything up to the kth distance, ties included, before the full sort.
        kth = np.partition(sub, k - 1)[k - 1]
        keep = np.flatnonzero(sub <= kth)
        ids, sub = ids[keep], sub[keep]
    order = np.lexsort((ids, sub))[:k]
    return ids[order], sub[order]


# Share of the true top k found; within tol of the kth distance counts.
# Out of min(k, n_pass), and 1.0 when nothing passes the filter.
def recall_at_k(returned_dists: np.ndarray, kth_true: float, n_pass: int, k: int = K, tol: float = TIE_TOL) -> float:
    if n_pass == 0:
        return 1.0
    want = min(k, n_pass)
    hits = int(np.count_nonzero(np.asarray(returned_dists)[:k] <= kth_true + tol))
    return min(hits, want) / want
