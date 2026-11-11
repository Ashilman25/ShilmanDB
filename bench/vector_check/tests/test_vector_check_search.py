"""Tests for bench/vector_check/hnsw_search.py, on small graphs built by hand in faiss's layout."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
TOL = 1e-5


# Load a module by path under a vc_ name, clear of other bench tests.
def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, HERE.parent / file_name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mod = _load("vc_hnsw_search", "hnsw_search.py")
vc = mod.vc


# Seeded unit float32 rows.
def unit_rows(rng, n: int, dim: int) -> np.ndarray:
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# A graph in faiss's layout from per-level link lists; unused slots stay -1.
# level_links[l][node] lists node's links at level l; levels counts each node's levels.
def make_graph(vectors, levels, level_links, widths, entry: int):
    levels = np.asarray(levels, dtype=np.int32)
    cum = np.concatenate([[0], np.cumsum(widths)]).astype(np.int32)
    offsets = np.zeros(len(levels) + 1, dtype=np.uint64)
    offsets[1:] = np.cumsum(cum[levels])
    neighbors = np.full(int(offsets[-1]), -1, dtype=np.int32)
    for level, links in enumerate(level_links):
        for node, ls in links.items():
            assert level < levels[node] and len(ls) <= widths[level]
            start = int(offsets[node]) + int(cum[level])
            neighbors[start:start + len(ls)] = ls
    return mod.Graph(vectors=np.ascontiguousarray(vectors, dtype=np.float32), levels=levels, offsets=offsets,
                     neighbors=neighbors, cum=cum, entry=entry, max_level=int(levels.max()) - 1)


# 2-D unit vectors at the given angles in degrees.
def angle_rows(degrees) -> np.ndarray:
    r = np.radians(np.asarray(degrees, dtype=np.float64))
    return np.stack([np.cos(r), np.sin(r)], axis=1).astype(np.float32)


# Six nodes at 0, 20, ..., 100 degrees and a query at 85 degrees.
# Distance order: 4 < 5 < 3 < 2 < 1 < 0.
def line_world():
    return angle_rows([0, 20, 40, 60, 80, 100]), angle_rows([85])[0]


LINE_LINKS = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]}


# A kNN-style graph: every 10th node also on level 1, every 50th on level 2.
def random_graph(n: int = 300, dim: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    vectors = unit_rows(rng, n, dim)
    sims = vectors @ vectors.T
    levels = 1 + (np.arange(n) % 10 == 0) + (np.arange(n) % 50 == 0)
    widths = (16, 8, 8)
    level_links = []
    for level, width in enumerate(widths):
        members = np.flatnonzero(levels > level)
        links = {}
        for i in members:
            order = members[np.argsort(-sims[i, members], kind="stable")]
            links[int(i)] = order[order != i][:width // 2].tolist()
        level_links.append(links)
    return make_graph(vectors, levels, level_links, widths, entry=0), rng


# Exact distances from q to every row, as the search computes them.
def exact(vectors, q) -> np.ndarray:
    return vc.cosine_dists(vectors, np.asarray(q, dtype=np.float32))


# Every field of two results is the same, log included.
def assert_same(a, b) -> None:
    assert np.array_equal(a.ids, b.ids) and np.array_equal(a.dists, b.dists)
    assert (a.ndis, a.nstep) == (b.ndis, b.nstep)
    assert (a.log is None) == (b.log is None)
    if a.log is not None:
        assert np.array_equal(a.log.ndis, b.log.ndis)
        assert np.array_equal(a.log.features, b.log.features)
        assert np.array_equal(a.log.recall, b.log.recall, equal_nan=True)


# Column index of a feature in the log.
def col(name: str) -> int:
    return mod.FEATURES.index(name)


# The feature names, in log order.
def test_features_order():
    assert mod.FEATURES == ("nstep", "ndis", "ninserts", "firstNN", "closestNN", "furthestNN",
                            "avg", "var", "med", "perc25", "perc75")


# One level, every node links to every other, ef >= n: the exact top k.
def test_complete_graph_equals_brute_force():
    rng = np.random.default_rng(3)
    n = 40
    vectors = unit_rows(rng, n, 8)
    links = {i: [j for j in range(n) if j != i] for i in range(n)}
    g = make_graph(vectors, [1] * n, [links], (n - 1,), entry=7)
    s = mod.Searcher(g)
    for q in unit_rows(rng, 12, 8):
        for ef in (n, n + 10):
            r = s.search(q, ef=ef)
            want_ids, want_d = vc.topk(exact(vectors, q), vc.K)
            assert r.ids.tolist() == want_ids.tolist()
            np.testing.assert_allclose(r.dists, want_d, rtol=0, atol=1e-6)
            # Every node is evaluated once and every node is expanded once.
            assert (r.ndis, r.nstep) == (n, n)
            assert r.ids.dtype == np.int64 and r.dists.dtype == np.float64 and r.log is None


# Six nodes on a line, entry 2, ef 3, k 2: counts traced by hand.
def test_line_graph_counts():
    # Trace (distance order 4 < 5 < 3 < 2 < 1 < 0):
    #   entry 2: ndis 1, results {2}, top-k [2], ninserts 1
    #   pop 2 (results not full): nstep 1; walk 1 (ndis 2, ninserts 2), walk 3 (ndis 3, top-k [3,2], ninserts 3)
    #   pop 3 (worst 1): nstep 2; 2 visited; walk 4 (ndis 4, drops 1, top-k [4,3], ninserts 4)
    #   pop 4 (worst 2): nstep 3; 3 visited; walk 5 (ndis 5, drops 2, top-k [4,5], ninserts 5)
    #   pop 5 (worst 3): nstep 4; 4 visited
    #   pop 1: results full and (d1, 1) > worst (d3, 3): stop. Node 0 is never evaluated.
    vectors, q = line_world()
    g = make_graph(vectors, [1] * 6, [LINE_LINKS], (2,), entry=2)
    d = exact(vectors, q)
    r = mod.Searcher(g).search(q, ef=3, k=2, log_every=1, kth_true=float(d[5]))
    assert r.ids.tolist() == [4, 5]
    assert (r.ndis, r.nstep) == (5, 4)
    np.testing.assert_allclose(r.dists, d[[4, 5]], rtol=0, atol=1e-6)

    # One level, so the entry's distance is layer 0's first and can be a checkpoint.
    f = r.log.features
    assert r.log.ndis.tolist() == [1, 2, 3, 4, 5]
    assert f[:, col("nstep")].tolist() == [0, 1, 1, 2, 3]
    assert f[:, col("ndis")].tolist() == [1, 2, 3, 4, 5]
    assert f[:, col("ninserts")].tolist() == [1, 2, 3, 4, 5]
    np.testing.assert_allclose(f[:, col("firstNN")], d[2], rtol=1e-6)
    np.testing.assert_allclose(f[:, col("closestNN")], d[[2, 2, 3, 4, 4]], rtol=1e-6)
    np.testing.assert_allclose(f[:, col("furthestNN")], d[[2, 1, 2, 3, 5]], rtol=1e-6)
    assert r.log.recall.tolist() == [0.0, 0.0, 0.0, 0.5, 1.0]
    # A top-k of one: every statistic is that distance and the variance is 0.
    stats = f[0, col("avg"):]
    np.testing.assert_allclose(stats, np.float32([d[2], 0.0, d[2], d[2], d[2]]), rtol=1e-6)

    # Recall is out of k even while the top k is short.
    r = mod.Searcher(g).search(q, ef=3, k=2, log_every=1, kth_true=float(d[2]))
    assert r.log.recall.tolist() == [0.5, 0.5, 1.0, 1.0, 1.0]


# Same line, k 1: a results insert that misses the top k isn't counted.
def test_line_graph_ninserts_with_k1():
    # Same path as with k 2; the top-1 changes only at 3 (ndis 3) and 4 (ndis 4):
    #   entry 2 (ninserts 1); 1 enters the results but not the top-1 (still 1)
    #   3 (ninserts 2); 4 (ninserts 3); 5 enters the results but not the top-1 (still 3)
    vectors, q = line_world()
    g = make_graph(vectors, [1] * 6, [LINE_LINKS], (2,), entry=2)
    r = mod.Searcher(g).search(q, ef=3, k=1, log_every=1)
    assert r.ids.tolist() == [4]
    assert (r.ndis, r.nstep) == (5, 4)
    assert r.log.features[:, col("ninserts")].tolist() == [1, 1, 2, 3, 3]


# A link listed twice in one list is walked and counted once.
def test_duplicate_link_counted_once():
    vectors, q = line_world()
    links = {**LINE_LINKS, 2: [1, 3, 3], 3: [2, 4, 2]}
    g = make_graph(vectors, [1] * 6, [links], (3,), entry=2)
    r = mod.Searcher(g).search(q, ef=3, k=2, log_every=1)
    # Same counts as the plain line: the repeats add no distance computation.
    assert r.ids.tolist() == [4, 5]
    assert (r.ndis, r.nstep) == (5, 4)
    assert r.log.ndis.tolist() == [1, 2, 3, 4, 5]
    assert r.log.features[-1, col("ninserts")] == 5


# Nodes 0, 3, 5 also on level 1, linked 0-3 and 3-5 by default.
def two_level_graph(upper: dict | None = None):
    vectors, q = line_world()
    upper = upper or {0: [3], 3: [0, 5], 5: [3]}
    g = make_graph(vectors, [2, 1, 1, 2, 1, 2], [LINE_LINKS, upper], (2, 2), entry=0)
    return g, vectors, q


# Two levels: ndis counts the entry and every greedy evaluation, repeats included.
def test_upper_layer_descent_counts():
    # Descent on level 1 (no visited set):
    #   entry 0: ndis 1
    #   links of 0 = [3]: 3 (ndis 2) beats 0, move to 3
    #   links of 3 = [0, 5]: 0 again (ndis 3), 5 (ndis 4) beats 3, move to 5
    #   links of 5 = [3]: 3 again (ndis 5), no better: descent ends at 5
    # Layer 0 from 5 reuses its distance (no ndis), ef 2, k 2:
    #   entry 5: top-k [5], ninserts 1
    #   pop 5 (results not full): nstep 1; walk 4 (ndis 6, top-k [4,5], ninserts 2)
    #   pop 4 (worst 5): nstep 2; walk 3 (ndis 7, not better than worst 5); 5 visited
    #   no candidates left.
    g, vectors, q = two_level_graph()
    d = exact(vectors, q)
    r = mod.Searcher(g).search(q, ef=2, k=2, log_every=1, kth_true=float(d[5]))
    assert r.ids.tolist() == [4, 5]
    assert (r.ndis, r.nstep) == (7, 2)
    # The descent's 5 distances aren't checkpoints; layer 0's two are.
    assert r.log.ndis.tolist() == [6, 7]
    f = r.log.features
    assert f[:, col("nstep")].tolist() == [1, 2]
    assert f[:, col("ninserts")].tolist() == [2, 2]
    np.testing.assert_allclose(f[:, col("firstNN")], d[5], rtol=1e-6)
    assert r.log.recall.tolist() == [1.0, 1.0]


# The greedy reads a node's whole list before it moves, keeping the best it saw.
def test_upper_layer_reads_whole_list():
    # Level-1 links of 0 are [3, 5]:
    #   entry 0 (ndis 1); 3 (ndis 2) beats 0; 5 (ndis 3) beats 3; move to 5
    #   links of 5 = [3]: 3 again (ndis 4), no better: descent ends at 5
    # Layer 0 is as before: walk 4 (ndis 5), walk 3 (ndis 6), 2 expansions.
    # Jumping at the first better link would cost one more (0's list, then 3's).
    g, _, q = two_level_graph({0: [3, 5], 3: [0, 5], 5: [3]})
    s = mod.Searcher(g)
    r = s.search(q, ef=2, k=2)
    assert r.ids.tolist() == [4, 5]
    assert (r.ndis, r.nstep) == (6, 2)
    # Stopped at ndis 2, the best seen is 3, though 0's list isn't done.
    assert s.search(q, ef=2, k=2, max_ndis=2).ids.tolist() == [3]


# max_ndis reached in the descent returns the node the descent is at, alone.
def test_max_ndis_in_descent_returns_current_node():
    g, vectors, q = two_level_graph()
    d = exact(vectors, q)
    s = mod.Searcher(g)
    # ndis 1 is the entry; 3 re-evaluates 0 while at 3; 4 finds 5; 5 ends the descent.
    for max_ndis, node in ((1, 0), (2, 3), (3, 3), (4, 5), (5, 5)):
        r = s.search(q, ef=2, k=2, log_every=1, max_ndis=max_ndis)
        assert r.ids.tolist() == [node]
        assert (r.ndis, r.nstep) == (max_ndis, 0)
        np.testing.assert_allclose(r.dists, d[[node]], rtol=0, atol=1e-6)
        assert r.log.ndis.shape == (0,) and r.log.features.shape == (0, 11) and r.log.recall.shape == (0,)
    assert s.search(q, ef=2, k=2, max_ndis=6).ids.tolist() == [4, 5]


# Ties break by id everywhere: a tied newcomer with a smaller id beats the worst.
def test_ties_break_by_id():
    # q = e1, so every distance is exactly 1 - x for the first coordinate x.
    # Distances by node: 0: .5, 1: 0, 2: .5, 3: .75, 4: 0, 5: .5, 6: 1, 7: .75
    # Entry 6, ef 4, k 4, links walked as 7, 5, 4, 3, 2, 1, 0:
    #   7, 5, 4 fill the results (ninserts 2, 3, 4); 3 drops 6; 2 drops 7; 1 drops 3 (ninserts 5, 6, 7)
    #   0 (.5, 0) beats the worst (.5, 5) only by id and drops it (ninserts 8)
    #   pops: 1, 4, 0, 2 expand (nstep 2..5, nothing new); (.5, 5) is worse than (.5, 2): stop
    rng = np.random.default_rng(5)
    first = np.array([0.5, 1.0, 0.5, 0.25, 1.0, 0.5, 0.0, 0.25], dtype=np.float32)
    vectors = np.column_stack([first, rng.standard_normal((8, 3)).astype(np.float32)])
    q = np.array([1, 0, 0, 0], dtype=np.float32)
    links = {i: [j for j in range(7, -1, -1) if j != i] for i in range(8)}
    g = make_graph(vectors, [1] * 8, [links], (7,), entry=6)
    r = mod.Searcher(g).search(q, ef=4, k=4, log_every=1)
    assert r.ids.tolist() == [1, 4, 0, 2]
    assert r.dists.tolist() == [0.0, 0.0, 0.5, 0.5]
    assert (r.ndis, r.nstep) == (8, 5)
    assert r.log.features[:, col("ninserts")].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]


# Checkpoints land on multiples of log_every, in order, with 11 features.
def test_log_every_multiples():
    g, rng = random_graph()
    s = mod.Searcher(g)
    for q in unit_rows(rng, 5, 16):
        kth = float(np.sort(exact(g.vectors, q))[vc.K - 1])
        r = s.search(q, ef=50, log_every=8, kth_true=kth)
        assert (r.log.ndis % 8 == 0).all() and r.log.features.shape[1] == 11 and np.all(np.diff(r.log.ndis) > 0)
        assert len(r.log.ndis) > 0 and r.ndis - r.log.ndis[-1] < 8
        assert r.log.ndis.dtype == np.int32 and r.log.features.dtype == np.float32 and r.log.recall.dtype == np.float32
        assert r.log.features[:, col("ndis")].tolist() == r.log.ndis.tolist()
        assert ((r.log.recall >= 0) & (r.log.recall <= 1)).all()


# The last checkpoint's features are the statistics of the top k returned there.
def test_last_checkpoint_features_match_topk_stats():
    g, rng = random_graph()
    s = mod.Searcher(g)
    for q in unit_rows(rng, 5, 16):
        kth = float(np.sort(exact(g.vectors, q))[vc.K - 1])
        full = s.search(q, ef=40, log_every=1, kth_true=kth)
        for j in (len(full.log.ndis) // 3, len(full.log.ndis) - 1):
            stop = int(full.log.ndis[j])
            r = s.search(q, ef=40, log_every=1, kth_true=kth, max_ndis=stop)
            assert r.ndis == stop and len(r.ids) == vc.K
            d = r.dists
            want = [r.nstep, r.ndis, full.log.features[j, col("ninserts")], full.log.features[0, col("firstNN")],
                    d[0], d[-1], np.mean(d), np.var(d), np.median(d), np.percentile(d, 25), np.percentile(d, 75)]
            np.testing.assert_allclose(r.log.features[-1], np.float32(want), rtol=1e-6, atol=1e-7)
            assert r.log.recall[-1] == np.float32(np.count_nonzero(d <= kth + TOL) / vc.K)


# A truncated search walks the same path and stops at max_ndis exactly.
def test_max_ndis_truncates():
    g, rng = random_graph()
    s = mod.Searcher(g)
    for q in unit_rows(rng, 5, 16):
        kth = float(np.sort(exact(g.vectors, q))[vc.K - 1])
        full = s.search(q, ef=40, log_every=1, kth_true=kth)
        for j in (0, 1, len(full.log.ndis) // 2, len(full.log.ndis) - 1):
            stop = int(full.log.ndis[j])
            t = s.search(q, ef=40, log_every=1, kth_true=kth, max_ndis=stop)
            assert t.ndis == stop
            assert np.array_equal(t.log.features, full.log.features[:j + 1])
            assert np.array_equal(t.log.recall, full.log.recall[:j + 1])
            # The recall of what it returns is the logged recall at that checkpoint.
            assert np.float32(np.count_nonzero(t.dists <= kth + TOL) / vc.K) == full.log.recall[j]
        # A cap at the last distance keeps the answer but skips the expansions after it.
        t = s.search(q, ef=40, log_every=1, kth_true=kth, max_ndis=full.ndis)
        assert np.array_equal(t.ids, full.ids) and t.ndis == full.ndis and t.nstep <= full.nstep
        assert np.array_equal(t.log.features, full.log.features)
        # A cap past the natural end changes nothing.
        assert_same(s.search(q, ef=40, log_every=1, kth_true=kth, max_ndis=full.ndis + 100), full)


# Without kth_true the logged recall is NaN; without log_every there's no log.
def test_recall_nan_without_truth():
    g, rng = random_graph()
    q = unit_rows(rng, 1, 16)[0]
    s = mod.Searcher(g)
    r = s.search(q, ef=30, log_every=4)
    assert len(r.log.recall) > 0 and np.isnan(r.log.recall).all()
    assert s.search(q, ef=30, kth_true=0.5).log is None


# Same query, same answer and counts, on the same or a new searcher.
def test_deterministic():
    g, rng = random_graph()
    s = mod.Searcher(g)
    for q in unit_rows(rng, 8, 16):
        a = s.search(q, ef=60, log_every=8, kth_true=0.4)
        assert_same(a, s.search(q, ef=60, log_every=8, kth_true=0.4))
        assert_same(a, mod.Searcher(g).search(q, ef=60, log_every=8, kth_true=0.4))
        # A float64 query is cast to float32 first.
        assert_same(a, s.search(q.astype(np.float64), ef=60, log_every=8, kth_true=0.4))


# One visit-stamp array reused over many searches equals a fresh searcher each time.
def test_stamp_reuse_matches_fresh_searcher():
    g, rng = random_graph()
    s = mod.Searcher(g)
    for i, q in enumerate(unit_rows(rng, 60, 16)):
        ef = (10, 25, 80, 300)[i % 4]
        assert_same(s.search(q, ef=ef, log_every=8), mod.Searcher(g).search(q, ef=ef, log_every=8))


# The stamp counter wraps past the int32 limit without stale marks.
def test_stamp_wraps():
    g, rng = random_graph()
    s = mod.Searcher(g)
    # Stamps 1..4 leave marks that a wrap back to 1..4 must not read as visited.
    for q in unit_rows(rng, 4, 16):
        s.search(q, ef=300)
    s._stamp = np.iinfo(np.int32).max - 2
    for q in unit_rows(rng, 6, 16):
        assert_same(s.search(q, ef=40), mod.Searcher(g).search(q, ef=40))
    assert s._stamp == 4


# Bad arguments raise before any search.
def test_bad_arguments_raise():
    g, rng = random_graph()
    s = mod.Searcher(g)
    q = unit_rows(rng, 1, 16)[0]
    with pytest.raises(ValueError, match="ef"):
        s.search(q, ef=5, k=10)
    with pytest.raises(ValueError):
        s.search(q, ef=10, k=0)
    with pytest.raises(ValueError, match="log_every"):
        s.search(q, ef=10, log_every=0)
    with pytest.raises(ValueError, match="max_ndis"):
        s.search(q, ef=10, max_ndis=0)
    with pytest.raises(ValueError, match="query"):
        s.search(q[:8], ef=10)


# A Searcher refuses vectors that aren't float32 and C-contiguous.
def test_searcher_refuses_bad_vectors():
    g, _ = random_graph()
    for vectors, match in ((g.vectors.astype(np.float64), "float32"), (np.asfortranarray(g.vectors), "contiguous")):
        bad = mod.Graph(vectors, g.levels, g.offsets, g.neighbors, g.cum, g.entry, g.max_level)
        with pytest.raises(ValueError, match=match):
            mod.Searcher(bad)


# links() reads one level's slots and stops at the first -1.
def test_links_stop_at_first_padding():
    vectors, _ = line_world()
    g = make_graph(vectors, [2, 1, 1, 2, 1, 2], [LINE_LINKS, {0: [3], 3: [0, 5], 5: [3]}], (3, 2), entry=0)
    assert g.links(3, 0).tolist() == [2, 4] and g.links(3, 1).tolist() == [0, 5] and g.links(5, 1).tolist() == [3]
    # A -1 in the middle hides what follows it.
    start = int(g.offsets[1])
    g.neighbors[start:start + 3] = [0, -1, 2]
    assert g.links(1, 0).tolist() == [0]
    with pytest.raises(ValueError):
        g.links(1, 1)


# The search also stops reading a layer-0 list at its first -1.
def test_search_stops_at_first_padding():
    # Node 2's slots are [1, -1, 3], so 3, 4 and 5 are out of reach. Entry 2, ef 3, k 2:
    #   entry 2 (ndis 1); pop 2: nstep 1, walk 1 (ndis 2)
    #   pop 1 (results not full): nstep 2, walk 0 (ndis 3), results full {2, 1, 0}
    #   pop 0 ties the worst (itself): nstep 3, nothing new; no candidates left
    vectors, q = line_world()
    g = make_graph(vectors, [1] * 6, [{**LINE_LINKS, 2: [1, -1, 3]}], (3,), entry=2)
    r = mod.Searcher(g).search(q, ef=3, k=2)
    assert r.ids.tolist() == [2, 1]
    assert (r.ndis, r.nstep) == (3, 3)


# Write a graph's arrays and graph.json into a folder, as the export does.
def write_graph(g, folder: Path, sha: str | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    arrays = (g.levels, g.offsets, g.neighbors, g.cum)
    for name, a in zip(("levels", "offsets", "neighbors", "cum"), arrays):
        np.save(folder / f"{name}.npy", a)
    info = {"n": len(g.levels), "dim": int(g.vectors.shape[1]), "entry_point": g.entry, "max_level": g.max_level,
            "sha256": sha if sha is not None else mod.graph_sha256(*arrays)}
    (folder / "graph.json").write_text(json.dumps(info))


# sha256 over the four arrays' bytes, in order.
def test_graph_sha256_definition():
    g, _ = random_graph()
    h = hashlib.sha256()
    for a in (g.levels, g.offsets, g.neighbors, g.cum):
        h.update(a.tobytes())
    assert mod.graph_sha256(g.levels, g.offsets, g.neighbors, g.cum) == h.hexdigest()
    assert mod.graph_sha256(g.cum, g.offsets, g.neighbors, g.levels) != h.hexdigest()


# load_graph checks the sha and the vectors, and its graph searches like the original.
def test_load_graph_checks_sha(tmp_path):
    g, rng = random_graph()
    folder = tmp_path / "graph"
    write_graph(g, folder)
    np.save(tmp_path / "base.npy", g.vectors)
    base = np.load(tmp_path / "base.npy", mmap_mode="r")

    loaded = mod.load_graph(folder, base)
    assert (loaded.entry, loaded.max_level) == (g.entry, g.max_level)
    for name in ("levels", "offsets", "neighbors", "cum"):
        assert np.array_equal(getattr(loaded, name), getattr(g, name))
    # Memory-mapped vectors and arrays search exactly like the in-memory graph.
    a, b = mod.Searcher(loaded), mod.Searcher(g)
    for q in unit_rows(rng, 5, 16):
        assert_same(a.search(q, ef=50, log_every=8), b.search(q, ef=50, log_every=8))

    with pytest.raises(ValueError, match="float32"):
        mod.load_graph(folder, np.asarray(base, dtype=np.float64))
    with pytest.raises(ValueError, match="contiguous"):
        mod.load_graph(folder, np.asfortranarray(base))
    with pytest.raises(ValueError, match="rows"):
        mod.load_graph(folder, np.ascontiguousarray(base[:-1]))

    # One changed link no longer matches graph.json's sha256.
    bad = g.neighbors.copy()
    bad[5] = (bad[5] + 1) % len(g.levels)
    np.save(folder / "neighbors.npy", bad)
    with pytest.raises(ValueError, match="sha256"):
        mod.load_graph(folder, base)

    write_graph(g, tmp_path / "wrong", sha="0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        mod.load_graph(tmp_path / "wrong", base)


# A graph with its entry off the top level, or a bad link, is refused.
def test_load_graph_checks_layout(tmp_path):
    g, _ = random_graph()
    base = g.vectors
    write_graph(mod.Graph(g.vectors, g.levels, g.offsets, g.neighbors, g.cum, entry=1, max_level=g.max_level), tmp_path / "a")
    with pytest.raises(ValueError, match="entry"):
        mod.load_graph(tmp_path / "a", base)
    bad = g.neighbors.copy()
    bad[0] = len(g.levels)
    write_graph(mod.Graph(g.vectors, g.levels, g.offsets, bad, g.cum, g.entry, g.max_level), tmp_path / "b")
    with pytest.raises(ValueError, match="neighbors"):
        mod.load_graph(tmp_path / "b", base)

    # graph.json's n and dim must match the arrays and the vectors.
    write_graph(g, tmp_path / "c")
    info = json.loads((tmp_path / "c" / "graph.json").read_text())
    for key, value in (("n", len(g.levels) + 1), ("dim", 17)):
        (tmp_path / "c" / "graph.json").write_text(json.dumps({**info, key: value}))
        with pytest.raises(ValueError, match=f"says {key} "):
            mod.load_graph(tmp_path / "c", base)


# Recall reaches 1 at a large ef on the random graph, as a sanity check.
def test_random_graph_finds_exact_at_large_ef():
    g, rng = random_graph()
    s = mod.Searcher(g)
    hits = []
    for q in unit_rows(rng, 10, 16):
        want = vc.topk(exact(g.vectors, q), vc.K)[0]
        hits.append(len(set(s.search(q, ef=300).ids.tolist()) & set(want.tolist())) / vc.K)
    assert math.isclose(min(hits), 1.0)
