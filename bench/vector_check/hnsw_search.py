from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from bisect import insort
from dataclasses import dataclass
from heapq import heappop, heappush, heapreplace
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

FEATURES = ("nstep", "ndis", "ninserts", "firstNN", "closestNN", "furthestNN", "avg", "var", "med", "perc25", "perc75")
ARRAYS = ("levels", "offsets", "neighbors", "cum")
STAMP_MAX = int(np.iinfo(np.int32).max)
CHUNK = 65536


# faiss's layout: node i is on levels 0..levels[i]-1, and its slots start at offsets[i].
# Level l's links sit in slots cum[l]..cum[l+1] of the node's slots, padded with -1.
@dataclass(frozen=True, eq=False)
class Graph:
    vectors: np.ndarray
    levels: np.ndarray
    offsets: np.ndarray
    neighbors: np.ndarray
    cum: np.ndarray
    entry: int
    max_level: int

    # A node's links at one level, in stored order, up to the first -1.
    def links(self, node: int, level: int) -> np.ndarray:
        if not 0 <= level < int(self.levels[node]):
            raise ValueError(f"node {node} has no level {level}")
        
        base = int(self.offsets[node])
        seg = self.neighbors[base + int(self.cum[level]):base + int(self.cum[level + 1])]
        pad = np.flatnonzero(seg < 0)
        return seg[:pad[0]] if len(pad) else seg


# A search's checkpoints, one row each: ndis, the 11 features, recall.
@dataclass(eq=False)
class SearchLog:
    ndis: np.ndarray
    features: np.ndarray
    recall: np.ndarray


# One search's answer, its counts, and its log (None without log_every).
@dataclass(eq=False)
class SearchResult:
    ids: np.ndarray
    dists: np.ndarray
    ndis: int
    nstep: int
    log: SearchLog | None


# sha256 over the four graph arrays' bytes, in this order.
def graph_sha256(levels, offsets, neighbors, cum) -> str:
    h = hashlib.sha256()
    for a in (levels, offsets, neighbors, cum):
        h.update(np.asarray(a).tobytes())
    return h.hexdigest()


# Raise ValueError unless the vectors are 2-D, float32 and C-contiguous.
def _check_vectors(vectors) -> None:
    if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
        raise ValueError("vectors must be a 2-D numpy array")
    if vectors.dtype != np.float32:
        raise ValueError(f"vectors must be float32, got {vectors.dtype}")
    if not vectors.flags.c_contiguous:
        raise ValueError("vectors must be C-contiguous")


# Raise ValueError if the graph's arrays don't fit faiss's layout or its vectors.
def _check_graph(g: Graph) -> None:
    _check_vectors(g.vectors)
    levels, cum = np.asarray(g.levels), np.asarray(g.cum)
    offsets = np.asarray(g.offsets).astype(np.int64)
    neighbors = np.asarray(g.neighbors)
    n = len(levels)
    
    if levels.ndim != 1 or n == 0:
        raise ValueError("levels must be a non-empty 1-D array")

    if len(g.vectors) != n:
        raise ValueError(f"vectors have {len(g.vectors):,} rows but the graph has {n:,} nodes")

    if cum.ndim != 1 or len(cum) < 2 or cum[0] != 0 or (np.diff(cum) <= 0).any():
        raise ValueError("cum must start at 0 and rise at every level")

    if levels.min() < 1 or levels.max() >= len(cum):
        raise ValueError(f"levels must be 1..{len(cum) - 1}")

    if len(offsets) != n + 1 or offsets[0] != 0 or offsets[-1] != len(neighbors):
        raise ValueError("offsets must have n + 1 entries, from 0 to the number of slots")

    if not np.array_equal(np.diff(offsets), cum[levels]):
        raise ValueError("offsets don't match each node's levels")

    if len(neighbors) and (neighbors.min() < -1 or neighbors.max() >= n):
        raise ValueError(f"neighbors must be -1 or a node id below {n:,}")

    if not 0 <= g.entry < n or g.max_level != int(levels[g.entry]) - 1 or g.max_level != int(levels.max()) - 1:
        raise ValueError(f"entry {g.entry} must be a node on the top level {g.max_level}")


# Read a saved graph and check it against its graph.json and the base vectors.
def load_graph(graph_dir: Path, vectors: np.ndarray) -> Graph:
    graph_dir = Path(graph_dir)
    _check_vectors(vectors)
    arrays = [np.load(graph_dir / f"{name}.npy", mmap_mode="r") for name in ARRAYS]
    info = json.loads((graph_dir / "graph.json").read_text())
    
    for key in ("entry_point", "max_level", "sha256"):
        if key not in info:
            raise ValueError(f"{graph_dir / 'graph.json'} has no {key}")

    sha = graph_sha256(*arrays)
    if sha != info["sha256"]:
        raise ValueError(f"{graph_dir}: the arrays' sha256 {sha} differs from graph.json's {info['sha256']}")

    if len(vectors) != len(arrays[0]):
        raise ValueError(f"vectors have {len(vectors):,} rows but the graph has {len(arrays[0]):,} nodes")

    if "n" in info and int(info["n"]) != len(arrays[0]):
        raise ValueError(f"graph.json says n {info['n']} but the arrays hold {len(arrays[0]):,} nodes")

    if "dim" in info and int(info["dim"]) != vectors.shape[1]:
        raise ValueError(f"vectors have {vectors.shape[1]} columns but graph.json says dim {info['dim']}")

    graph = Graph(vectors, *arrays, entry=int(info["entry_point"]), max_level=int(info["max_level"]))
    _check_graph(graph)
    return graph


# The linear percentile of sorted values, computed the way np.percentile does.
def _percentile(s: list[float], q: float) -> float:
    vi = (len(s) - 1) * q
    lo = math.floor(vi)
    t = vi - lo
    a, b = s[lo], s[min(lo + 1, len(s) - 1)]
    diff = b - a
    return b - diff * (1 - t) if t >= 0.5 else a + diff * t


# One checkpoint: the 11 features of the current top k, then recall.
def _checkpoint(nstep: int, ndis: int, ninserts: int, first: float, top: list, kth_true: float | None, k: int) -> tuple:
    ds = [d for d, _ in top]
    avg = sum(ds) / len(ds)
    var = sum((x - avg) * (x - avg) for x in ds) / len(ds)
    recall = math.nan if kth_true is None else sum(1 for x in ds if x <= kth_true + vc.TIE_TOL) / k
    return (nstep, ndis, ninserts, first, ds[0], ds[-1], avg, var, _percentile(ds, 0.5), _percentile(ds, 0.25), _percentile(ds, 0.75), recall)


# Checkpoint rows as a SearchLog; no rows gives empty arrays of the right shapes.
def _make_log(rows: list) -> SearchLog:
    a = np.array(rows, dtype=np.float64).reshape(len(rows), len(FEATURES) + 1)
    return SearchLog(ndis=a[:, 1].astype(np.int32), features=a[:, :-1].astype(np.float32), recall=a[:, -1].astype(np.float32))


# The finished result, from (distance, id) pairs in order.
def _result(top: list, ndis: int, nstep: int, rows: list | None) -> SearchResult:
    ids = np.array([v for _, v in top], dtype=np.int64)
    dists = np.array([d for d, _ in top], dtype=np.float64)
    return SearchResult(ids, dists, ndis, nstep, None if rows is None else _make_log(rows))


# Each node's layer-0 slots as [start, end), with end at the first -1.
def _layer0_spans(g: Graph, neighbors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.asarray(g.offsets[:-1]).astype(np.int64) + int(g.cum[0])
    width = int(g.cum[1]) - int(g.cum[0])
    cols = np.arange(width)
    lengths = np.empty(len(starts), dtype=np.int64)
    
    for a in range(0, len(starts), CHUNK):
        pad = neighbors[starts[a:a + CHUNK, None] + cols] < 0
        lengths[a:a + CHUNK] = np.where(pad.any(axis=1), pad.argmax(axis=1), width)
    return starts, starts + lengths


# Searches one graph, reusing a single visit-stamp array across searches.
class Searcher:
    def __init__(self, graph: Graph):
        _check_graph(graph)
        self.graph = graph
        self._vectors = np.asarray(graph.vectors)
        self._neighbors = np.asarray(graph.neighbors)
        self._start0, self._end0 = _layer0_spans(graph, self._neighbors)
        self._visited = np.zeros(len(graph.levels), dtype=np.int32)
        self._stamp = 0

    # A new stamp per search; at the int32 limit I clear the array first.
    def _next_stamp(self) -> int:
        if self._stamp >= STAMP_MAX:
            self._visited.fill(0)
            self._stamp = 0
        self._stamp += 1
        return self._stamp

    # Greedy descent, then the layer-0 beam search with ef; ties go to the smaller id.
    # max_ndis stops right after that many distances; log_every logs every that many.
    def search(self, q, ef: int, k: int = vc.K, log_every: int | None = None, kth_true: float | None = None, max_ndis: int | None = None) -> SearchResult:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if ef < k:
            raise ValueError(f"ef must be at least k ({k}), got {ef}")
        if log_every is not None and log_every < 1:
            raise ValueError(f"log_every must be at least 1, got {log_every}")
        if max_ndis is not None and max_ndis < 1:
            raise ValueError(f"max_ndis must be at least 1, got {max_ndis}")

        g, vectors = self.graph, self._vectors
        q = np.ascontiguousarray(q, dtype=np.float32)
        if q.shape != (vectors.shape[1],):
            raise ValueError(f"query must have shape ({vectors.shape[1]},), got {q.shape}")
        
        rows = None if log_every is None else []
        stamp = self._next_stamp()
        visited = self._visited

        # Upper levels: no visited set, every evaluation counts, move to the best seen.
        cur = int(g.entry)
        d_cur = float(vc.cosine_dists(vectors[cur:cur + 1], q)[0])
        ndis = 1
        if g.max_level > 0 and ndis == max_ndis:
            return _result([(d_cur, cur)], ndis, 0, rows)
        
        for level in range(g.max_level, 0, -1):
            while True:
                node = cur
                links = g.links(node, level)
                for v, d in zip(links.tolist(), vc.cosine_dists(vectors[links], q).tolist()):
                    ndis += 1
                    if d < d_cur or (d == d_cur and v < cur):
                        cur, d_cur = v, d
                    if ndis == max_ndis:
                        return _result([(d_cur, cur)], ndis, 0, rows)
                    
                if cur == node:
                    break

        # Layer 0 starts from the descent's node and reuses its distance.
        visited[cur] = stamp
        first = d_cur
        results = [(-d_cur, -cur)]           # max-heap on (d, id), so results[0] is the worst
        candidates = [(d_cur, cur)]
        top = [(d_cur, cur)]                 # the best k so far, sorted
        worst_d, worst_id = d_cur, cur
        nstep, ninserts = 0, 1
        
        if g.max_level == 0:
            if rows is not None and ndis % log_every == 0:
                rows.append(_checkpoint(nstep, ndis, ninserts, first, top, kth_true, k))
            if ndis == max_ndis:
                return _result(top, ndis, nstep, rows)

        start0, end0, neighbors = self._start0, self._end0, self._neighbors
        while candidates:
            d_c, c = heappop(candidates)
            if len(results) >= ef and (d_c > worst_d or (d_c == worst_d and c > worst_id)):
                break
            
            nstep += 1
            seg = neighbors[start0[c]:end0[c]]
            fresh = seg[visited[seg] != stamp]
            if not len(fresh):
                continue

            # One matmul for the node's unvisited links, then walk them in stored order.
            for v, d in zip(fresh.tolist(), vc.cosine_dists(vectors[fresh], q).tolist()):
                if visited[v] == stamp:
                    continue                 # a link listed twice counts once
                
                visited[v] = stamp
                ndis += 1
                if len(results) < ef or d < worst_d or (d == worst_d and v < worst_id):
                    heappush(candidates, (d, v))
                    if len(results) < ef:
                        heappush(results, (-d, -v))
                    else:
                        heapreplace(results, (-d, -v))
                        
                    worst_d, worst_id = -results[0][0], -results[0][1]
                    if len(top) < k or (d, v) < top[-1]:
                        insort(top, (d, v))
                        if len(top) > k:
                            top.pop()
                        ninserts += 1
                        
                if rows is not None and ndis % log_every == 0:
                    rows.append(_checkpoint(nstep, ndis, ninserts, first, top, kth_true, k))
                    
                if ndis == max_ndis:
                    return _result(top, ndis, nstep, rows)

        # results holds the best ef seen and top its best k, so top is the answer.
        return _result(top, ndis, nstep, rows)
