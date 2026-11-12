"""Unit tests for bench/vector_check/vc_common.py."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

SPEC_PATH = Path(__file__).resolve().parents[1] / "vc_common.py"
spec = importlib.util.spec_from_file_location("vc_common", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["vc_common"] = mod
spec.loader.exec_module(mod)


def test_text_rules():
    assert mod.normalize_text("Great  Case", "Fits\nwell") == "great case fits well"
    assert mod.text_key("Great  Case", "Fits\nwell") == hashlib.sha1(b"great case fits well").hexdigest()
    assert mod.review_input("T", "R") == "Title: T\nReview: R"
    assert mod.review_input("", "R") == "Title: \nReview: R"


def test_recall_ties_count():          # many rows tied at the 10th distance all count, capped
    assert mod.recall_at_k(np.array([.1, .2, .2, .2]), .2, 100, k=3) == 1.0


def test_recall_short_filter():
    assert mod.recall_at_k(np.array([.1, .2, .3]), .3, 3) == 1.0
    assert mod.recall_at_k(np.array([.1, .3]), .3, 3) == pytest.approx(2 / 3)


def test_recall_short_result():
    assert mod.recall_at_k(np.array([.1, .1, .1, .1]), .5, 100) == pytest.approx(0.4)


def test_recall_tolerance():
    assert mod.recall_at_k(np.array([.2 + 5e-6]), .2, 1, k=1) == 1.0
    assert mod.recall_at_k(np.array([.2 + 2e-5]), .2, 1, k=1) == 0.0


def test_recall_empty_filter():
    assert mod.recall_at_k(np.array([]), 0.0, 0) == 1.0


def test_topk_breaks_ties_by_id():
    ids, d = mod.topk(np.array([.3, .1, .1, .2]), 3)
    assert ids.tolist() == [1, 2, 3]
    assert ids.dtype == np.int64
    assert d.tolist() == [.1, .1, .2]


def test_topk_candidates():            # restricts to candidate ids, still id order on ties
    dists = np.array([.1, .5, .2, .2, .9, .2])
    ids, d = mod.topk(dists, 3, candidates=np.array([5, 1, 3, 2]))
    assert ids.tolist() == [2, 3, 5]
    assert d.tolist() == [.2, .2, .2]
    ids, _ = mod.topk(dists, 10, candidates=np.array([4, 1]))
    assert ids.tolist() == [1, 4]
    ids, _ = mod.topk(np.full(4, .1), 2, candidates=np.array([3, 2, 1, 0]))
    assert ids.tolist() == [0, 1]


def test_topk_k_above_rows():
    ids, d = mod.topk(np.array([.4, .2]), 10)
    assert ids.tolist() == [1, 0] and d.tolist() == [.2, .4]


def test_topk_rejects_a_mask():        # a bool mask would quietly become ids 0 and 1
    with pytest.raises(TypeError, match="row ids"):
        mod.topk(np.array([.1, .2, .3]), 2, candidates=np.array([True, False, True]))
    with pytest.raises(TypeError, match="row ids"):
        mod.topk(np.array([.1, .2]), 1, candidates=[False, True])


def test_seed_for_stable_and_distinct():
    assert mod.seed_for("queries", 70) == mod.seed_for("queries", 70)
    assert mod.seed_for("queries", 70) != mod.seed_for("queries", 71)
    assert mod.seed_for("a", "b") != mod.seed_for("b", "a")
    want = int(hashlib.sha1(b"1:queries:70").hexdigest()[:16], 16)
    assert mod.seed_for("queries", 70) == want


def test_cosine_dists_float32_matmul():
    vecs = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
    got = mod.cosine_dists(vecs, np.array([0.6, 0.8]))
    assert got.dtype == np.float64
    q32 = np.array([0.6, 0.8], dtype=np.float32)
    assert np.array_equal(got, 1.0 - (vecs @ q32).astype(np.float64))
    assert got == pytest.approx([0.4, 0.2, 0.0], abs=1e-6)


def test_paths_layout():
    p = mod.paths(Path("R"), Path("A"))
    assert p.size_dir("500k") == Path("R/size500k") and p.pg_log == Path("R/pg.log")
    assert p.data == Path("R/data") and p.embeddings == Path("R/embeddings")
    assert p.pgdata == Path("R/pgdata") and p.report == Path("R/report.md")
    assert p.ai == Path("A")
    assert mod.check_size("70k") == "70k"
    with pytest.raises(ValueError):
        mod.check_size("1m")


def test_constants():
    assert (mod.SIZES, mod.K, mod.SEED, mod.DIM) == (("70k", "500k"), 10, 1, 384)
    assert mod.MODEL == "BAAI/bge-small-en-v1.5"
    assert (mod.PG_PORT, mod.PG_HOST, mod.PG_DB) == (5433, "/tmp", "vector_check")
    assert (mod.TIE_TOL, mod.TIMEOUT_MS) == (1e-5, 10000.0)


def test_default_roots():
    assert mod.REPO_ROOT == SPEC_PATH.parents[2]
    assert mod.paths().results == mod.REPO_ROOT / "bench/results/vector_check"
    assert mod.paths().ai == mod.REPO_ROOT / "bench/results/ai_queries"


def test_load_sibling_registers_module():
    try:
        loaded = mod.load_sibling("vc_common_again", "vc_common.py")
        assert sys.modules["vc_common_again"] is loaded
        assert loaded.K == 10
    finally:
        sys.modules.pop("vc_common_again", None)


def test_load_sibling_forgets_a_failed_module(tmp_path, monkeypatch):
    name = f"vc_broken_{os.getpid()}"
    (tmp_path / f"{name}.py").write_text("raise RuntimeError('broken on purpose')\n")
    monkeypatch.setattr(mod, "HERE", tmp_path)
    try:
        with pytest.raises(RuntimeError, match="broken on purpose"):
            mod.load_sibling(name, f"{name}.py")
        assert name not in sys.modules
    finally:
        sys.modules.pop(name, None)
