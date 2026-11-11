"""Tests for bench/vector_check/stopping_check.py; LightGBM runs only in child processes."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "stopping_check.py"


# Load a module by path under a vc_ name, clear of other bench tests.
def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, HERE.parent / file_name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mod = _load("vc_stopping_check", "stopping_check.py")
hg = _load("vc_hnsw_graph", "hnsw_graph.py")
hs = mod.hs
vc = mod.vc


# Seeded unit float32 rows.
def unit_rows(rng, n: int, dim: int) -> np.ndarray:
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# Unit rows near random base rows, so each query has a clear nearest row.
def near_rows(rng, base: np.ndarray, n: int) -> np.ndarray:
    picks = base[rng.integers(len(base), size=n)]
    v = picks + 0.7 * unit_rows(rng, n, base.shape[1])
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


# A kNN graph in faiss's layout: every 10th node on level 1, 50th on 2.
# Layer 0 also links each node to its ring neighbors, so every node is reachable.
def knn_graph(vectors: np.ndarray) -> dict:
    n = len(vectors)
    sims = vectors @ vectors.T
    levels = (1 + (np.arange(n) % 10 == 0) + (np.arange(n) % 50 == 0)).astype(np.int32)
    widths = (32, 16, 16)
    cum = np.concatenate([[0], np.cumsum(widths)]).astype(np.int32)
    offsets = np.zeros(n + 1, dtype=np.uint64)
    offsets[1:] = np.cumsum(cum[levels])
    neighbors = np.full(int(offsets[-1]), -1, dtype=np.int32)
    for level in range(len(widths)):
        members = np.flatnonzero(levels > level)
        for i in members:
            order = members[np.argsort(-sims[i, members], kind="stable")]
            links = order[order != i][:widths[level] // 2].tolist()
            if level == 0:
                links += [j for j in ((i - 1) % n, (i + 1) % n) if j not in links]
            start = int(offsets[i]) + int(cum[level])
            neighbors[start:start + len(links)] = links
    return {"levels": levels, "offsets": offsets, "neighbors": neighbors, "cum": cum, "entry": 0, "max_level": 2}


# Results and ai folders holding a tiny 70k world: base, graph, queries and truth.
def write_world(root: Path, n_train: int = 30, n_val: int = 12, n_test: int = 12, n_base: int = 300) -> dict:
    rng = np.random.default_rng(5)
    results, ai = root / f"results{os.getpid()}", root / f"ai{os.getpid()}"
    base = unit_rows(rng, n_base, vc.DIM)
    (ai / "embeddings").mkdir(parents=True)
    np.save(ai / "embeddings" / "test.npy", base)
    info = {"dim": vc.DIM, "m": 16, "ef_construction": 64, "threads": 1, "build_s": 0.0}
    hg.export(knn_graph(base), results / "size70k" / "graph", info)

    splits = np.array(["train"] * n_train + ["validation"] * n_val + ["test"] * n_test)
    rng.shuffle(splits)
    reviews = pd.DataFrame({"qid": [f"r{i:05d}" for i in range(1, len(splits) + 1)], "split": splits,
                            "text_key": "", "input": ""})
    (results / "data").mkdir(parents=True)
    reviews.to_csv(results / "data" / "query_reviews.csv", index=False)
    questions = vc.read_questions()
    review_vectors, question_vectors = near_rows(rng, base, len(reviews)), near_rows(rng, base, len(questions))
    (results / "embeddings").mkdir(parents=True)
    np.save(results / "embeddings" / "query_reviews.npy", review_vectors)
    np.save(results / "embeddings" / "questions.npy", question_vectors)

    found = [vc.topk(vc.cosine_dists(base, q), vc.K) for q in np.concatenate([review_vectors, question_vectors])]
    top_dists = np.array([d for _, d in found])
    truth = {
        "qids": np.array(reviews["qid"].tolist() + questions["qid"].tolist(), dtype=str),
        "sets": np.array(splits.tolist() + ["questions"] * len(questions), dtype=str),
        "top_ids": np.array([i for i, _ in found]), "top_dists": top_dists, "kth": top_dists[:, -1].copy(),
    }
    np.savez(results / "size70k" / "truth_part2.npz", **truth)
    return {"results": results, "ai": ai, "size_dir": results / "size70k", "reviews": reviews,
            "questions": questions, "truth": truth}


# Arguments shared by every run against a world.
def world_args(world: dict) -> list[str]:
    return ["--size", "70k", "--results", str(world["results"]), "--ai", str(world["ai"])]


# Run one subcommand as a child process.
def run_child(cmd: str, world: dict, *extra: str) -> subprocess.CompletedProcess:
    args = [sys.executable, str(SCRIPT), cmd, *world_args(world), *extra]
    return subprocess.run(args, capture_output=True, text=True, timeout=120)


# The selected qids per scored set, first n of each review set in truth order.
def selected(world: dict, n_val: int, n_test: int) -> list[str]:
    t = world["truth"]
    picks = []
    for name, n in (("validation", n_val), ("test", n_test), ("questions", None)):
        picks += t["qids"][t["sets"] == name][:n].tolist()
    return picks


# A hand-made rem.csv: every query at every ef, recall 1.0, a fixed ndis.
def write_fake_rem(world: dict, qids: list[str], ndis: int = 1) -> pd.DataFrame:
    rows = [("x", q, ef, 1.0, ndis) for q in qids for ef in mod.EF_GRID]
    frame = pd.DataFrame(rows, columns=["set", "qid", "ef", "recall", "ndis"])
    frame.to_csv(world["size_dir"] / "rem.csv", index=False)
    return frame


# A trajectory set from per-query (ndis, recall) lists and final values.
def make_traj(per_query: list, final_ndis: list, final_recall: list):
    offsets = np.concatenate([[0], np.cumsum([len(nd) for nd, _ in per_query])]).astype(np.int64)
    ndis = np.concatenate([np.asarray(nd, dtype=np.int32) for nd, _ in per_query])
    recall = np.concatenate([np.asarray(r, dtype=np.float32) for _, r in per_query])
    return mod.Traj(qids=np.array([f"q{i}" for i in range(len(per_query))]), offsets=offsets, ndis=ndis,
                    features=np.zeros((len(ndis), len(hs.FEATURES)), dtype=np.float32), recall=recall,
                    final_ndis=np.asarray(final_ndis, dtype=np.int64), final_recall=np.asarray(final_recall, dtype=np.float64))


# The constants are the fixed values, verbatim.
def test_constants():
    assert mod.EF_GRID == (10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 700,
                           1000, 1500, 2000)
    assert mod.E_CHOICES == (500, 1000, 2000) and mod.E_RECALL == 0.995 and mod.TARGETS == (0.90, 0.95, 0.99)
    assert mod.MARGINS == (0.0, 0.005, 0.01, 0.02, 0.03, 0.05) and mod.LOG_EVERY == 8
    assert mod.MAX_TRAIN_ROWS == 3_000_000 and mod.TREES == 100 and mod.LEARNING_RATE == 0.1
    assert mod.TREE_CHARGE == 1 / 25 and mod.DETERMINISM_QUERIES == 20


# E is the smallest choice reaching 0.995, or None.
def test_choose_E():
    assert mod.choose_E({500: .99, 1000: .996, 2000: .999}) == 1000
    assert mod.choose_E({500: .9, 1000: .9, 2000: .99}) is None
    assert mod.choose_E({10: 1.0, 500: 0.995, 1000: 1.0}) == 500


# REM takes the smallest ef reaching the target, or None.
def test_rem_choice():
    assert mod.rem_choice({10: .5, 40: .91, 100: .97}, .95) == 100
    assert mod.rem_choice({100: .97, 10: .5, 40: .95}, .95) == 40
    assert mod.rem_choice({10: .5, 40: .91}, .95) is None


# The adaptive interval, with its floors at mpi and LOG_EVERY.
def test_interval_formula_and_floors():
    assert mod.interval(1000, 200, .95, .5) == 560
    assert mod.interval(1000, 200, .95, .99) == 200
    assert mod.interval(40, 4, .9, .9) == 8
    assert isinstance(mod.interval(1000, 200, .95, np.float64(.5)), int)


# The first call at or past ipi whose prediction clears target + margin stops there.
def test_replay_stops_when_prediction_reaches_target():
    ndis = np.array([8, 16, 24, 32, 40])
    preds = np.array([0.1, 0.5, 0.96, 0.99, 0.99])
    recall = np.array([0.1, 0.3, 0.8, 0.9, 1.0])
    # Calls at 16 (0.5, next call at 28) and 32 (0.99, stop).
    assert mod.replay_one(ndis, preds, recall, 45, 1.0, 0.95, 0.0, 16, 8) == (0.9, 32, 2)
    # A margin of 0.05 needs 1.0: calls at 16, 32 and 40, then the end.
    assert mod.replay_one(ndis, preds, recall, 45, 1.0, 0.95, 0.05, 16, 8) == (1.0, 45, 3)
    # A prediction exactly at target + margin stops.
    assert mod.replay_one(ndis, np.full(5, 0.95), recall, 45, 1.0, 0.95, 0.0, 16, 8) == (0.3, 16, 1)


# With predictions never high enough, the plain search's end is the answer.
def test_replay_falls_back_to_plain_end():
    ndis = np.array([8, 16, 24, 32, 40])
    got = mod.replay_one(ndis, np.full(5, 0.2), np.full(5, 0.5), 45, 1.0, 0.9, 0.0, 8, 8)
    assert got == (1.0, 45, 5)
    assert mod.replay_one(np.array([], dtype=np.int32), np.array([]), np.array([]), 7, 0.4, 0.9, 0.0, 8, 8) == (0.4, 7, 0)


# Calls are spaced by the adaptive interval and each one counts.
def test_replay_counts_calls():
    ndis = np.arange(8, 88, 8)
    # ipi 24: calls at 24, then every round(8 + 16 * 0.9) = 22 ndis.
    got = mod.replay_one(ndis, np.zeros(10), np.zeros(10), 85, 0.7, 0.9, 0.0, 24, 8)
    assert got == (0.7, 85, 3)
    # Nothing at or past a large ipi, so no calls at all.
    assert mod.replay_one(ndis, np.ones(10), np.ones(10), 85, 0.7, 0.9, 0.0, 1000, 100) == (0.7, 85, 0)


# Stopping at a logged checkpoint gives what a search truncated there returns.
def test_replay_equals_truncated_search():
    rng = np.random.default_rng(11)
    base = unit_rows(rng, 400, 16)
    g = knn_graph(base)
    graph = hs.Graph(vectors=base, levels=g["levels"], offsets=g["offsets"], neighbors=g["neighbors"],
                     cum=g["cum"], entry=g["entry"], max_level=g["max_level"])
    s = hs.Searcher(graph)
    checked = 0
    for q in unit_rows(rng, 6, 16):
        kth = float(vc.topk(vc.cosine_dists(base, q), vc.K)[1][-1])
        full = s.search(q, ef=60, log_every=mod.LOG_EVERY, kth_true=kth)
        recall = mod.exact_recall(full.log.recall)
        final = vc.recall_at_k(full.dists, kth, n_pass=len(base))
        for j in (0, len(full.log.ndis) // 3, len(full.log.ndis) // 2, len(full.log.ndis) - 1):
            preds = np.zeros(len(full.log.ndis))
            preds[j] = 1.0
            # ipi = mpi = 8 calls the predictor at every checkpoint.
            got_recall, got_ndis, calls = mod.replay_one(full.log.ndis, preds, recall, full.ndis, final, 0.9, 0.0, 8, 8)
            cut = s.search(q, ef=60, max_ndis=got_ndis)
            assert got_ndis == int(full.log.ndis[j]) and calls == j + 1 and cut.ndis == got_ndis
            assert got_recall == vc.recall_at_k(cut.dists, kth, n_pass=len(base))
            checked += 1
    assert checked == 24


# Work is distances plus calls charged at trees / 25, scaled by the factor.
def test_work_accounting():
    assert mod.work(1000, 5) == 1020 and mod.work(1000, 5, 0) == 1000 and mod.work(1000, 5, 4) == 1080
    assert mod.work(1000, 5, trees=50) == 1010


# The margin is the smallest whose validation mean reaches the target, or None.
def test_choose_margin():
    means = {0.0: 0.93, 0.005: 0.94, 0.01: 0.951, 0.02: 0.96, 0.03: 0.97, 0.05: 0.98}
    assert mod.choose_margin(means, 0.95) == 0.01
    assert mod.choose_margin(means, 0.90) == 0.0
    assert mod.choose_margin(means, 0.99) is None


# Mean recall comes from hit counts, so float32 recalls compare exactly to targets.
def test_mean_recall_exact():
    assert float(np.float32(0.9)) < 0.9
    assert mod.mean_recall(np.full(7, np.float32(0.9))) == 0.9
    assert mod.mean_recall([0.9, 1.0]) == 0.95
    assert mod.exact_recall(np.array([0.9, 0.3], dtype=np.float32)).tolist() == [0.9, 0.3]


# No margin reaching the target falls back to 0.05 and marks the row not ok.
def test_margin_fallback_marks_not_ok():
    # Every query predicts 1.0 at its first call but has recall 0.5 throughout.
    traj = make_traj([([8, 16], [0.5, 0.5]), ([8, 16, 24], [0.5, 0.5, 0.5])], [20, 30], [0.5, 0.5])
    preds = np.ones(5)
    margin, ok, means = mod.margin_for(traj, preds, 0.9, 8, 8)
    assert (margin, ok) == (0.05, False) and means == {m: 0.5 for m in mod.MARGINS}
    row = mod.summary_row("test", 0.9, "learned", margin, ok, [0.9, 1.0], [100, 200], [2, 4])
    assert row["margin_ok"] == "false" and row["param"] == 0.05

    # Recall 1.0 at the first call: margin 0 is enough.
    good = make_traj([([8, 16], [1.0, 1.0]), ([8, 16, 24], [1.0, 1.0, 1.0])], [20, 30], [1.0, 1.0])
    margin, ok, _ = mod.margin_for(good, preds, 0.9, 8, 8)
    assert (margin, ok) == (0.0, True)


# A summary row: means, work at 0x, 1x and 4x, the share below target.
def test_summary_row():
    row = mod.summary_row("test", 0.95, "learned", 0.01, True, [0.9, 1.0], [100, 200], [2, 4])
    assert row == {"set": "test", "target": 0.95, "method": "learned", "param": 0.01, "margin_ok": "true",
                   "mean_recall": 0.95, "mean_ndis": 150.0, "mean_calls": 3.0, "work_0x": 150.0,
                   "work_1x": 162.0, "work_4x": 198.0, "below_target_share": 0.5, "n_queries": 2}
    rem = mod.summary_row("test", 0.9, "rem", 100, None, [0.9, 1.0], [100, 200], [0, 0])
    assert rem["margin_ok"] == "" and rem["below_target_share"] == 0.0 and rem["work_4x"] == 150.0


# Rows kept per query at step j: positions 0, j, 2j, ...
def kept_at(counts, j: int) -> int:
    return int(sum(-(-c // j) for c in counts))


# The smallest step whose kept rows fit the cap, by brute force.
def smallest_fit(counts, cap: int) -> int:
    return next(j for j in range(1, max(counts) + 1) if kept_at(counts, j) <= cap)


# Thinning keeps checkpoints 0, j, 2j of each query, j the smallest step that fits.
def test_thinning_cap():
    offsets = np.array([0, 5, 8, 12])
    keep, new_offsets, step = mod.thin_rows(offsets, cap=5)
    assert step == 3 and keep.tolist() == [0, 3, 5, 8, 11] and new_offsets.tolist() == [0, 2, 3, 5]
    keep, new_offsets, step = mod.thin_rows(offsets, cap=12)
    assert step == 1 and keep.tolist() == list(range(12)) and new_offsets.tolist() == [0, 5, 8, 12]
    # Step 2 would keep 7 rows, over the cap of 6, so step 3.
    keep, _, step = mod.thin_rows(offsets, cap=6)
    assert step == 3 and keep.tolist() == [0, 3, 5, 8, 11]


# Many short queries: ceil(total/cap) would overshoot, so j grows until it fits.
def test_thinning_never_overshoots():
    counts = [5] * 6
    offsets = np.concatenate([[0], np.cumsum(counts)])
    # ceil(30 / 10) = 3 keeps 12 rows, and so does 4; 5 keeps 6.
    assert kept_at(counts, 3) == 12 > 10 and kept_at(counts, 4) == 12 > 10
    keep, new_offsets, step = mod.thin_rows(offsets, cap=10)
    assert step == 5 and keep.tolist() == [0, 5, 10, 15, 20, 25] and new_offsets.tolist() == list(range(7))

    rng = np.random.default_rng(4)
    for _ in range(200):
        counts = rng.integers(0, 40, size=int(rng.integers(1, 30))).tolist()
        n_busy = sum(c > 0 for c in counts)
        offsets = np.concatenate([[0], np.cumsum(counts)])
        cap = int(rng.integers(max(1, n_busy), max(2, sum(counts) + 5)))
        keep, new_offsets, step = mod.thin_rows(offsets, cap=cap)
        assert len(keep) <= cap and len(keep) == new_offsets[-1] == kept_at(counts, step)
        assert step == (smallest_fit(counts, cap) if sum(counts) else 1)


# A cap below the number of queries with checkpoints can't be met, so it raises.
def test_thinning_cap_too_small_raises():
    with pytest.raises(ValueError, match="cap"):
        mod.thin_rows(np.array([0, 3, 6, 9]), cap=2)


# Reach is the first logged ndis at or above each target, else the final ndis.
def test_reach_before_thinning():
    offsets = np.array([0, 3, 5])
    ndis = np.array([8, 16, 24, 8, 16])
    recall = np.array([0.5, 0.9, 1.0, 0.8, 0.9], dtype=np.float32)
    reach = mod.reach_ndis(offsets, ndis, recall, np.array([30, 20]))
    # A float32 0.9 still reaches the 0.9 target.
    assert reach.tolist() == [[16, 24, 24], [16, 20, 20]]


# Loading the module pulls in neither faiss nor lightgbm.
def test_import_does_not_load_faiss():
    code = (
        "import importlib.util, json, sys\n"
        f"spec = importlib.util.spec_from_file_location('vc_stopping_check', {str(SCRIPT)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "sys.modules['vc_stopping_check'] = m\n"
        "spec.loader.exec_module(m)\n"
        "print(json.dumps({name: name in sys.modules for name in ('faiss', 'lightgbm')}))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {"faiss": False, "lightgbm": False}


# Each broken link between truth, csvs and vectors exits 1 naming the file.
@pytest.mark.parametrize("broken", ["qids", "sets", "npy_rows"])
def test_query_rows_must_match(tmp_path, capsys, broken):
    world = write_world(tmp_path)
    truth = dict(world["truth"])
    if broken == "qids":
        truth["qids"] = truth["qids"].copy()
        truth["qids"][[0, 1]] = truth["qids"][[1, 0]]
        np.savez(world["size_dir"] / "truth_part2.npz", **truth)
        name = "truth_part2.npz"
    elif broken == "sets":
        truth["sets"] = truth["sets"].copy()
        truth["sets"][3] = "test" if truth["sets"][3] != "test" else "train"
        np.savez(world["size_dir"] / "truth_part2.npz", **truth)
        name = "truth_part2.npz"
    else:
        q = np.load(world["results"] / "embeddings" / "questions.npy")
        np.save(world["results"] / "embeddings" / "questions.npy", q[:-1])
        name = "questions.npy"
    assert mod.main(["rem", *world_args(world), "--jobs", "1"]) == 1
    err = capsys.readouterr().err
    word = {"qids": "qids", "sets": "sets", "npy_rows": "rows"}[broken]
    assert name in err and word in err
    assert not (world["size_dir"] / "rem.csv").exists()


# A rem.csv missing one query's row stops trajectories before any search.
def test_trajectories_needs_every_rem_row(tmp_path, capsys):
    world = write_world(tmp_path)
    qids = selected(world, 3, 3)
    frame = write_fake_rem(world, qids)
    missing = qids[4]
    frame[~((frame["qid"] == missing) & (frame["ef"] == 500))].to_csv(world["size_dir"] / "rem.csv", index=False)
    assert mod.main(["trajectories", *world_args(world), "--jobs", "1", "--limit", "5,3,3"]) == 1
    err = capsys.readouterr().err
    assert "rem.csv" in err and missing in err
    assert not (world["size_dir"] / "traj").exists()


# A plain search whose ndis differs from rem.csv's at E exits 1, writing no trajectories.
def test_trajectories_final_ndis_must_match_rem(tmp_path, capsys):
    world = write_world(tmp_path)
    write_fake_rem(world, selected(world, 3, 3), ndis=1)
    assert mod.main(["trajectories", *world_args(world), "--jobs", "1", "--limit", "5,3,3"]) == 1
    err = capsys.readouterr().err
    assert "rem.csv" in err and "ndis" in err
    assert json.loads((world["size_dir"] / "plain_E.json").read_text())["E"] == 500
    assert not list((world["size_dir"] / "traj").glob("*.npz"))
    assert "lightgbm" not in sys.modules


# Reach comes from every checkpoint; only the stored training rows are thinned.
def test_trajectories_thins_after_reach(tmp_path, monkeypatch, capsys):
    world = write_world(tmp_path)
    args = [*world_args(world), "--jobs", "1", "--limit", "20,3,3"]
    assert mod.main(["rem", *args]) == 0
    assert mod.main(["trajectories", *args]) == 0
    full = dict(np.load(world["size_dir"] / "traj" / "train.npz"))
    monkeypatch.setattr(mod, "MAX_TRAIN_ROWS", 100)
    assert mod.main(["trajectories", *args]) == 0
    thin = dict(np.load(world["size_dir"] / "traj" / "train.npz"))

    step = smallest_fit(np.diff(full["offsets"]).tolist(), 100)
    assert int(full["thin_every"]) == 1 and int(thin["thin_every"]) == step > 1
    assert np.array_equal(thin["reach"], full["reach"]) and len(thin["ndis"]) <= 100 < len(full["ndis"])
    keep, offsets, _ = mod.thin_rows(full["offsets"], 100)
    assert np.array_equal(thin["offsets"], offsets)
    for key in ("ndis", "features", "recall"):
        assert np.array_equal(thin[key], full[key][keep])
    for key in ("qids", "final_ndis", "final_recall"):
        assert np.array_equal(thin[key], full[key])
    assert f"every {step}" in capsys.readouterr().out
    assert "lightgbm" not in sys.modules


# A validation mean below 0.995 at every E writes E null and exits 1.
def test_trajectories_without_E_exits_1(tmp_path, capsys):
    world = write_world(tmp_path)
    frame = write_fake_rem(world, selected(world, 3, 3))
    frame["recall"] = 0.9
    frame.to_csv(world["size_dir"] / "rem.csv", index=False)
    assert mod.main(["trajectories", *world_args(world), "--jobs", "1", "--limit", "5,3,3"]) == 1
    plain = json.loads((world["size_dir"] / "plain_E.json").read_text())
    assert plain["E"] is None and plain["limits"] == {"train": 5, "validation": 3, "test": 3}
    assert "0.995" in capsys.readouterr().err


# train and replay refuse upstream files written with other limits, before LightGBM loads.
@pytest.mark.parametrize("cmd", ["train", "replay"])
def test_limits_must_match_upstream(tmp_path, capsys, cmd):
    world = write_world(tmp_path)
    limits = {"train": 5, "validation": 3, "test": 3}
    (world["size_dir"] / "plain_E.json").write_text(json.dumps({"E": 500, "val_recall": {}, "limits": limits}))
    (world["size_dir"] / "train_info.json").write_text(json.dumps({"rows": 1, "limits": limits, "targets": {}}))
    extra = [] if cmd == "train" else ["--jobs", "1"]
    assert mod.main([cmd, *world_args(world), *extra, "--limit", "5,3,4"]) == 1
    err = capsys.readouterr().err
    assert "plain_E.json" in err and "limits" in err
    # No limit at all also differs from a limited upstream run.
    assert mod.main([cmd, *world_args(world), *extra]) == 1
    assert "limits" in capsys.readouterr().err
    assert "lightgbm" not in sys.modules


# replay also checks train_info.json's limits.
def test_replay_checks_train_info_limits(tmp_path, capsys):
    world = write_world(tmp_path)
    limits = {"train": 5, "validation": 3, "test": 3}
    (world["size_dir"] / "plain_E.json").write_text(json.dumps({"E": 500, "val_recall": {}, "limits": limits}))
    (world["size_dir"] / "train_info.json").write_text(json.dumps({"rows": 1, "limits": {}, "targets": {}}))
    assert mod.main(["replay", *world_args(world), "--jobs", "1", "--limit", "5,3,3"]) == 1
    err = capsys.readouterr().err
    assert "train_info.json" in err and "limits" in err
    assert "lightgbm" not in sys.modules


# replay refuses a model trained at another E than plain_E.json's, before LightGBM loads.
def test_replay_refuses_stale_model(tmp_path, capsys):
    world = write_world(tmp_path)
    limits = {"train": 5, "validation": 3, "test": 3}
    (world["size_dir"] / "plain_E.json").write_text(json.dumps({"E": 500, "val_recall": {}, "limits": limits}))
    (world["size_dir"] / "train_info.json").write_text(json.dumps({"rows": 1, "E": 1000, "limits": limits, "targets": {}}))
    assert mod.main(["replay", *world_args(world), "--jobs", "1", "--limit", "5,3,3"]) == 1
    err = capsys.readouterr().err
    assert "train_info.json" in err and "plain_E.json" in err and "1000" in err and "500" in err
    assert "lightgbm" not in sys.modules


# --limit takes three whole numbers >= 1, --jobs one >= 1; train takes no --jobs.
@pytest.mark.parametrize("extra", [["--limit", "1,2"], ["--limit", "0,1,1"], ["--limit", "a,1,1"], ["--jobs", "0"]])
def test_bad_arguments_exit_2(tmp_path, extra):
    world = write_world(tmp_path)
    with pytest.raises(SystemExit) as e:
        mod.main(["rem", *world_args(world), *extra])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        mod.main(["train", *world_args(world), "--jobs", "2"])
    assert e.value.code == 2


# Outputs inside the ai_queries folder are refused.
def test_output_inside_ai_refused(tmp_path, capsys):
    world = write_world(tmp_path)
    args = ["rem", "--size", "70k", "--results", str(world["ai"] / "inner"), "--ai", str(world["ai"]), "--jobs", "1"]
    assert mod.main(args) == 1
    assert "ai_queries" in capsys.readouterr().err


# rem, trajectories, train and replay end to end on a tiny world, in child processes.
def test_tiny_end_to_end(tmp_path):
    started = time.monotonic()
    world = write_world(tmp_path)
    size_dir, limit = world["size_dir"], ["--limit", "20,10,10"]
    limits = {"train": 20, "validation": 10, "test": 10}

    # rem gives the same bytes with one job or two.
    proc = run_child("rem", world, "--jobs", "1", *limit)
    assert proc.returncode == 0, proc.stderr
    one = {name: (size_dir / name).read_bytes() for name in ("rem.csv", "determinism.csv")}
    proc = run_child("rem", world, "--jobs", "2", *limit)
    assert proc.returncode == 0, proc.stderr
    assert {name: (size_dir / name).read_bytes() for name in one} == one
    assert "jobs 2" in proc.stdout

    rem = pd.read_csv(size_dir / "rem.csv", dtype={"set": str, "qid": str})
    assert list(rem.columns) == ["set", "qid", "ef", "recall", "ndis"]
    assert len(rem) == (10 + 10 + 210) * len(mod.EF_GRID) and rem["qid"].tolist()[0] == selected(world, 10, 10)[0]
    det = pd.read_csv(size_dir / "determinism.csv", dtype={"qid": str})
    assert list(det.columns) == ["qid", "same_ids", "same_ndis"] and len(det) == 10
    assert det["same_ids"].all() and det["same_ndis"].all()

    # trajectories gives the same bytes with two jobs or one.
    proc = run_child("trajectories", world, "--jobs", "2", *limit)
    assert proc.returncode == 0, proc.stderr
    sets = ("train", "validation", "test", "questions")
    traj_bytes = {s: (size_dir / "traj" / f"{s}.npz").read_bytes() for s in sets}
    proc = run_child("trajectories", world, "--jobs", "1", *limit)
    assert proc.returncode == 0, proc.stderr
    assert {s: (size_dir / "traj" / f"{s}.npz").read_bytes() for s in sets} == traj_bytes
    plain = json.loads((size_dir / "plain_E.json").read_text())
    assert plain["E"] == 500 and plain["val_recall"]["500"] >= 0.995 and plain["limits"] == limits
    train = np.load(size_dir / "traj" / "train.npz")
    assert len(train["qids"]) == 20 and train["reach"].shape == (20, 3) and len(train["offsets"]) == 21
    assert train["features"].shape == (len(train["ndis"]), 11) and train["offsets"][-1] == len(train["ndis"])
    questions = np.load(size_dir / "traj" / "questions.npz")
    assert questions["qids"].tolist() == world["questions"]["qid"].tolist()

    proc = run_child("train", world, *limit)
    assert proc.returncode == 0, proc.stderr
    info = json.loads((size_dir / "train_info.json").read_text())
    assert info["limits"] == limits and info["rows"] == len(train["ndis"])
    assert set(info["targets"]) == {"0.9", "0.95", "0.99"}
    for t in info["targets"].values():
        assert t["ipi"] == max(8, round(t["dists_t"] / 2)) and t["mpi"] == max(8, round(t["dists_t"] / 10))
    assert (size_dir / "model" / "darth.txt").exists()

    proc = run_child("replay", world, "--jobs", "2", *limit)
    assert proc.returncode == 0, proc.stderr
    stop = pd.read_csv(size_dir / "stopping.csv", dtype=str, keep_default_na=False)
    assert list(stop.columns) == ["set", "target", "method", "param", "margin_ok", "mean_recall", "mean_ndis",
                                  "mean_calls", "work_0x", "work_1x", "work_4x", "below_target_share", "n_queries"]
    got = {(r.set, float(r.target), r.method) for r in stop.itertuples()}
    assert got == {(s, t, m) for s in ("validation", "test", "questions") for t in mod.TARGETS
                   for m in ("rem", "learned", "plain")}
    assert len(stop) == 27
    for r in stop.itertuples():
        assert int(r.n_queries) == {"validation": 10, "test": 10, "questions": 210}[r.set]
        assert float(r.work_0x) == float(r.mean_ndis)
        assert float(r.work_1x) == pytest.approx(float(r.mean_ndis) + 4 * float(r.mean_calls))
        if r.method == "learned":
            assert r.margin_ok in ("true", "false") and float(r.param) in mod.MARGINS
        else:
            assert r.margin_ok == "" and float(r.mean_calls) == 0
        if r.method == "plain":
            assert int(r.param) == 500
        if r.method == "rem":
            assert int(r.param) in mod.EF_GRID
    plain_val = stop[(stop["set"] == "validation") & (stop["method"] == "plain")]
    assert (plain_val["mean_recall"].astype(float) == plain["val_recall"]["500"]).all()
    per_query = pd.read_csv(size_dir / "stopping_queries.csv", dtype={"set": str, "qid": str})
    assert list(per_query.columns) == ["set", "qid", "target", "method", "recall", "ndis", "calls"]
    assert len(per_query) == (10 + 10 + 210) * 3 * 3
    assert time.monotonic() - started < 30
