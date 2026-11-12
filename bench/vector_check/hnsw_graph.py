from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np


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

M = 16
EF_CONSTRUCTION = 64
INFO_KEYS = ("dim", "m", "ef_construction", "threads", "build_s")


# faiss's HNSW graph over every row (node id = row), with build seconds and threads.
# faiss is imported here only, so loading this file never pulls it in.
def build_graph(vectors: np.ndarray, m: int = M, ef_construction: int = EF_CONSTRUCTION, threads: int | None = None) -> dict:
    import faiss

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if threads is not None:
        faiss.omp_set_num_threads(threads)
    index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction

    started = time.perf_counter()
    index.add(vectors)
    build_s = time.perf_counter() - started

    hnsw = index.hnsw
    graph = {
        "levels": faiss.vector_to_array(hnsw.levels),
        "offsets": faiss.vector_to_array(hnsw.offsets),
        "neighbors": faiss.vector_to_array(hnsw.neighbors),
        "cum": faiss.vector_to_array(hnsw.cum_nneighbor_per_level),
        "entry": int(hnsw.entry_point),
        "max_level": int(hnsw.max_level),
        "build_s": build_s,
        "threads": int(faiss.omp_get_max_threads()),
    }
    if len(graph["levels"]) != len(vectors):
        raise ValueError(f"faiss built {len(graph['levels']):,} nodes for {len(vectors):,} rows")
    return graph


# Write via a temp file and rename, so a stopped run leaves no half file.
def _write_atomic(path: Path, write: Callable) -> None:
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            write(f)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# Save the four arrays, then graph.json last, so a stopped export leaves no json.
def export(graph: dict, out_dir: Path, info: dict) -> None:
    missing = [key for key in INFO_KEYS if key not in info]
    if missing:
        raise ValueError(f"info is missing {', '.join(missing)}")
    
    arrays = {name: np.ascontiguousarray(graph[name]) for name in hs.ARRAYS}
    for name, a in arrays.items():
        if a.ndim != 1:
            raise ValueError(f"{name} must be a 1-D array, got shape {a.shape}")

    meta = {
        "n": len(arrays["levels"]),
        "dim": int(info["dim"]),
        "m": int(info["m"]),
        "ef_construction": int(info["ef_construction"]),
        "entry_point": int(graph["entry"]),
        "max_level": int(graph["max_level"]),
        "threads": int(info["threads"]),
        "build_s": float(info["build_s"]),
        "sha256": hs.graph_sha256(*arrays.values()),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graph.json").unlink(missing_ok=True)
    
    for name, a in arrays.items():
        _write_atomic(out_dir / f"{name}.npy", lambda f, a=a: np.save(f, a))
    _write_atomic(out_dir / "graph.json", lambda f: f.write((json.dumps(meta, indent=2) + "\n").encode("utf-8")))


# Base vectors: 2-D float32 rows of width DIM, every value finite.
def _load_base(path: Path) -> np.ndarray:
    if not path.exists():
        raise ValueError(f"{path}: not found")
    try:
        vectors = np.load(path)
    except (ValueError, TypeError, OSError, EOFError) as e:
        raise ValueError(f"{path}: can't read it as a numpy array ({e})") from None

    if not isinstance(vectors, np.ndarray) or vectors.ndim != 2 or vectors.shape[1] != vc.DIM or len(vectors) == 0:
        raise ValueError(f"{path}: shape {getattr(vectors, 'shape', None)}, want rows x {vc.DIM}")
    if vectors.dtype != np.float32:
        raise ValueError(f"{path}: dtype {vectors.dtype}, want float32")
    if not np.isfinite(vectors).all():
        raise ValueError(f"{path}: every value must be finite")
    return np.ascontiguousarray(vectors)


# A positive whole number for --threads.
def _threads(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a whole number, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


# Parse the command line, build and save the graph. Returns the exit code.
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the HNSW graph with faiss and save it for the numpy search.")
    parser.add_argument("--size", required=True, choices=vc.SIZES)
    parser.add_argument("--threads", type=_threads, default=None, help="faiss threads (default: faiss's own)")
    parser.add_argument("--results", type=Path, default=vc.RESULTS)
    parser.add_argument("--ai", type=Path, default=vc.AI_ROOT)
    args = parser.parse_args(argv)

    p = vc.paths(args.results, args.ai)
    out = p.size_dir(args.size) / "graph"
    ai = p.ai.resolve()
    if out.resolve() == ai or ai in out.resolve().parents:
        print(f"error: won't write inside the ai_queries folder {p.ai}", file=sys.stderr)
        return 1

    start = time.monotonic()
    base_path = p.ai / "embeddings" / "test.npy" if args.size == "70k" else p.embeddings / "slice.npy"
    try:
        base = _load_base(base_path)
        graph = build_graph(base, m=M, ef_construction=EF_CONSTRUCTION, threads=args.threads)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    info = {
        "dim": base.shape[1], 
        "m": M, 
        "ef_construction": EF_CONSTRUCTION,
        "threads": graph["threads"], 
        "build_s": round(graph["build_s"], 3)
    }
    export(graph, out, info)
    # A graph that doesn't load against its base is no use, so I drop its json.
    try:
        hs.load_graph(out, base)
    except ValueError as e:
        (out / "graph.json").unlink(missing_ok=True)
        print(f"error: the saved graph doesn't load: {e}", file=sys.stderr)
        return 1

    print(f"built in {graph['build_s']:.1f} s, threads {graph['threads']}: n {len(base):,}, max level {graph['max_level']}, entry point {graph['entry']}")
    print(f"wrote {out}; done in {time.monotonic() - start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
