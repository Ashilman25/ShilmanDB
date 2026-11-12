"""Tests for bench/vector_check/vector_truth.py. Everything runs on small synthetic files."""
from __future__ import annotations

import datetime as dt
import importlib.util
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC_PATH = Path(__file__).resolve().parents[1] / "vector_truth.py"
spec = importlib.util.spec_from_file_location("vc_vector_truth", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["vc_vector_truth"] = mod
spec.loader.exec_module(mod)

DIM = 384
CASES = "Cases, Holsters & Sleeves"
TABLE_COLUMNS = ["id", "product_id", "rating", "review_date", "verified", "category", "vec_row", "text_key"]
NPZ1 = ["qids", "groups", "halves", "filters", "n_pass", "top_ids", "top_dists", "kth"]
NPZ2 = ["qids", "sets", "top_ids", "top_dists", "kth"]


# Seeded unit float32 rows.
def unit_rows(rng, n: int, dim: int = DIM) -> np.ndarray:
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# The k nearest by a full sort on (distance, id), with -1 / inf padding.
def slow_topk(d: np.ndarray, ids: np.ndarray, k: int = 10) -> tuple[list[int], list[float]]:
    order = sorted(ids.tolist(), key=lambda i: (d[i], i))[:k]
    return order + [-1] * (k - len(order)), [float(d[i]) for i in order] + [np.inf] * (k - len(order))


# Tie-aware match against float64 distances: same distances, every id inside the true top k.
def assert_tie_aware(ids: np.ndarray, dists: np.ndarray, ref: np.ndarray, k: int = 10) -> None:
    want = np.sort(ref)[:k]
    np.testing.assert_allclose(dists, want, rtol=0, atol=1e-6)
    assert len(set(ids.tolist())) == k
    assert (ref[ids] <= want[-1] + 1e-6).all()


# A tiny results and ai tree: table, query reviews and every vector file.
# Rows 3 and 5 share vector 0, not a product; r00001's vector is vector 0.
def write_world(tmp_path: Path, n_rows: int = 600, n_vecs: int = 150, n_reviews: int = 24, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    results, ai = tmp_path / "results", tmp_path / "ai"
    for d in (results / "data", results / "embeddings", ai / "embeddings"):
        d.mkdir(parents=True)

    base = unit_rows(rng, n_vecs)
    vec_row = np.empty(n_rows, dtype=np.int64)
    vec_row[[3, 5]] = 0
    others = np.concatenate([np.arange(1, n_vecs), rng.integers(1, n_vecs, n_rows - n_vecs - 1)])
    rng.shuffle(others)
    vec_row[np.setdiff1d(np.arange(n_rows), [3, 5])] = others

    products = np.concatenate([np.repeat(np.arange(4), 120), rng.integers(4, 10, n_rows - 480)])
    rng.shuffle(products)
    products[3], products[5] = 7, 1
    days = rng.integers(0, 400, n_rows)
    table = pd.DataFrame({
        "id": np.arange(n_rows),
        "product_id": products,
        "rating": rng.integers(1, 6, n_rows),
        "review_date": [(dt.date(2020, 1, 1) + dt.timedelta(days=int(d))).isoformat() for d in days],
        "verified": np.where(rng.random(n_rows) < .9, "true", "false"),
        "category": rng.choice([CASES, "Chargers", ""], n_rows),
        "vec_row": vec_row,
        "text_key": [f"k{v}" for v in vec_row],
    })[TABLE_COLUMNS]

    splits = np.array(["test", "train", "validation"])[np.arange(n_reviews) % 3]
    reviews = pd.DataFrame({
        "qid": [f"r{i:05d}" for i in range(1, n_reviews + 1)],
        "split": splits,
        "text_key": [f"q{i}" for i in range(n_reviews)],
        "input": [f"Title: t{i}\nReview: line one\nline two" for i in range(n_reviews)],
    })
    review_vectors = unit_rows(rng, n_reviews)
    review_vectors[0] = base[0]
    question_vectors = unit_rows(rng, len(mod.vc.read_questions()))

    files = {
        "table": results / "data" / "table_70k.csv",
        "reviews": results / "data" / "query_reviews.csv",
        "review_vectors": results / "embeddings" / "query_reviews.npy",
        "question_vectors": results / "embeddings" / "questions.npy",
        "base": ai / "embeddings" / "test.npy",
    }
    table.to_csv(files["table"], index=False)
    reviews.to_csv(files["reviews"], index=False)
    np.save(files["review_vectors"], review_vectors)
    np.save(files["question_vectors"], question_vectors)
    np.save(files["base"], base)
    return {"results": results, "ai": ai, "files": files, "table": table, "reviews": reviews, "base": base,
            "review_vectors": review_vectors, "question_vectors": question_vectors}


# Run the script on a world; returns the exit code.
def run(world: dict, *extra: str) -> int:
    return mod.main(["--size", "70k", "--results", str(world["results"]), "--ai", str(world["ai"]), *extra])


# Load an npz without pickles, as plain arrays.
def load_npz(path: Path) -> dict:
    with np.load(path) as z:
        return {name: z[name] for name in z.files}


# Two of three rows pass, so the last eight slots are padding.
def test_truth_pads_short_filters():
    ids, d, n, kth = mod.cell_truth(np.array([.5, .1, .9]), np.array([True, True, False]))
    assert n == 2 and ids.tolist()[:2] == [1, 0] and ids.tolist()[2:] == [-1] * 8 and kth == .5 and np.isinf(d[2:]).all()


# Nothing passing gives all padding and an infinite kth.
def test_truth_with_nothing_passing():
    ids, d, n, kth = mod.cell_truth(np.array([.3, .2]), np.array([False, False]))
    assert n == 0 and ids.tolist() == [-1] * 10 and np.isinf(d).all() and np.isinf(kth)
    assert ids.dtype == np.int64 and d.dtype == np.float64


# Row distances with shared vectors give exact ties; every mask must match a full sort.
def test_cell_truth_matches_slow_loop():
    rng = np.random.default_rng(3)
    base = unit_rows(rng, 40, 8)
    vec_row = rng.integers(0, 40, 300)
    for trial in range(30):
        d_rows = mod.vc.cosine_dists(base, unit_rows(rng, 1, 8)[0])[vec_row]
        share = [0.0, 0.01, 0.03, 0.2, 1.0][trial % 5]
        m = rng.random(300) < share
        if trial == 7:
            m = np.zeros(300, dtype=bool)
            m[rng.choice(300, 10, replace=False)] = True
        ids, d, n, kth = mod.cell_truth(d_rows, m)
        want_ids, want_d = slow_topk(d_rows, np.flatnonzero(m))
        assert ids.tolist() == want_ids and d.tolist() == want_d and n == int(m.sum())
        assert kth == (want_d[min(n, 10) - 1] if n else np.inf)
    # Rows on the query's own vector come first, smaller id first.
    d_rows = mod.vc.cosine_dists(base, base[vec_row[0]])[vec_row]
    twins = np.flatnonzero(vec_row == vec_row[0])
    ids, _, _, _ = mod.cell_truth(d_rows, np.ones(300, dtype=bool))
    assert ids[:len(twins)].tolist() == twins[:10].tolist()


# A mask must be a bool array the same length as the distances.
def test_cell_truth_rejects_bad_masks():
    with pytest.raises(ValueError):
        mod.cell_truth(np.array([.1, .2, .3]), np.array([0, 2]))
    with pytest.raises(ValueError):
        mod.cell_truth(np.array([.1, .2, .3]), np.array([True, False]))


# Batches smaller than the query count; exact float32 sums, so ties are exact everywhere.
def test_exact_topk_batch_equals_brute_force():
    rng = np.random.default_rng(5)
    base = (rng.integers(-3, 4, (300, 16)) / 8).astype(np.float32)
    base[150:200] = base[:50]
    queries = (rng.integers(-3, 4, (37, 16)) / 8).astype(np.float32)
    queries[:5] = base[150:155]
    ids, dists = mod.exact_topk_batch(base, queries, k=10, batch=8)
    assert ids.shape == (37, 10) and ids.dtype == np.int64 and dists.dtype == np.float64
    for i, q in enumerate(queries):
        ref = 1.0 - base.astype(np.float64) @ q.astype(np.float64)
        want_ids, want_d = slow_topk(ref, np.arange(300))
        assert ids[i].tolist() == want_ids and dists[i].tolist() == want_d

    # Real-valued vectors: tie-aware against float64, any batch size.
    base = unit_rows(rng, 500, 64)
    queries = unit_rows(rng, 23, 64)
    ids, dists = mod.exact_topk_batch(base, queries, k=10, batch=4)
    for i, q in enumerate(queries):
        assert_tie_aware(ids[i], dists[i], 1.0 - base.astype(np.float64) @ q.astype(np.float64))
    _, same_d = mod.exact_topk_batch(base, queries, k=10, batch=256)
    np.testing.assert_allclose(same_d, dists, rtol=0, atol=1e-6)


# Under k base rows pads with -1 and inf; a batch under 1 is refused.
def test_exact_topk_batch_pads_and_checks_batch():
    rng = np.random.default_rng(6)
    base, queries = unit_rows(rng, 4, 8), unit_rows(rng, 3, 8)
    ids, dists = mod.exact_topk_batch(base, queries, k=10, batch=2)
    assert (ids[:, 4:] == -1).all() and np.isinf(dists[:, 4:]).all()
    assert sorted(ids[0, :4].tolist()) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        mod.exact_topk_batch(base, queries, batch=0)


# End to end: near product is the unfiltered nearest row's product, ties to smaller rows.
# Every cell must match a slow sort over rows sqlite picks by the SQL.
def test_near_product_is_nn_product(tmp_path, capsys):
    world = write_world(tmp_path)
    assert run(world, "--queries", "4") == 0
    out = world["results"] / "size70k"
    filters = pd.read_csv(out / "filters.csv", dtype=str, keep_default_na=False)
    assert filters.columns.tolist() == ["qid", "group", "half", "filter", "family", "sql", "n_pass", "share"]
    t1 = load_npz(out / "truth_part1.npz")
    assert sorted(t1) == sorted(NPZ1)

    table, base = world["table"], world["base"]
    arrays = mod.vf.TableArrays.from_csv(world["files"]["table"])
    vec_row, product = table["vec_row"].to_numpy(), table["product_id"].to_numpy()
    review_qids, questions = world["reviews"]["qid"].tolist(), mod.vc.read_questions()
    vectors = dict(zip(review_qids, world["review_vectors"]))
    vectors.update(zip(questions["qid"], world["question_vectors"]))
    test_qids = [q for q, s in zip(review_qids, world["reviews"]["split"]) if s == "test"]
    written = questions.loc[questions["group"] == "written", "qid"].tolist()
    ai53 = questions.loc[questions["group"] == "ai53", "qid"].tolist()
    queries = mod.vf.part1_queries(test_qids, written, ai53, "70k", n_per_group=4)

    names = list(mod.vf.FILTER_NAMES)
    assert t1["qids"].tolist() == [q for q, _, _ in queries]
    assert t1["groups"].tolist() == [g for _, g, _ in queries]
    assert t1["halves"].tolist() == [h for _, _, h in queries]
    assert t1["filters"].tolist() == names
    assert t1["top_ids"].shape == (8, 15, 10) and t1["n_pass"].shape == (8, 15) and t1["kth"].shape == (8, 15)
    assert filters[["qid", "group", "half"]].values.tolist() == [list(q) for q in queries for _ in names]
    assert filters["filter"].tolist() == names * 8

    rows = [(int(r.id), int(r.product_id), int(r.rating), r.review_date, int(r.verified == "true"), r.category)
            for r in table.itertuples()]
    with closing(sqlite3.connect(":memory:")) as con:
        con.execute("CREATE TABLE t (id INTEGER, product_id INTEGER, rating INTEGER, review_date TEXT, "
                    "verified INTEGER, category TEXT)")
        con.executemany("INSERT INTO t VALUES (?, ?, ?, ?, ?, ?)", rows)
        for i, (qid, _, _) in enumerate(queries):
            q = vectors[qid]
            ref = (1.0 - base.astype(np.float64) @ q.astype(np.float64))[vec_row]
            near = product[np.lexsort((np.arange(len(ref)), ref))[0]]
            cells = filters[filters["qid"] == qid].set_index("filter")
            assert cells.loc["product_near", "sql"] == f"product_id = {near}"
            assert cells.loc["product_near", "family"] == "product"
            drawn = mod.vf.make_filters(arrays, near, mod.vf.query_rng("70k", qid))
            assert cells["sql"].tolist() == [mod.vf.to_sql(f) for f in drawn]
            assert cells["family"].tolist() == [f.family for f in drawn]

            d_rows = mod.vc.cosine_dists(base, q)[vec_row]
            for j, name in enumerate(names):
                sql = cells.loc[name, "sql"]
                picked_rows = np.array([r[0] for r in con.execute(f"SELECT id FROM t{' WHERE ' + sql if sql else ''}")])
                want_ids, want_d = slow_topk(d_rows, picked_rows)
                assert t1["top_ids"][i, j].tolist() == want_ids, (qid, name)
                assert t1["top_dists"][i, j].tolist() == want_d, (qid, name)
                assert t1["n_pass"][i, j] == len(picked_rows) == int(cells.loc[name, "n_pass"])
                assert float(cells.loc[name, "share"]) == len(picked_rows) / len(table)
                assert t1["kth"][i, j] == (want_d[min(len(picked_rows), 10) - 1] if len(picked_rows) else np.inf)

    # r00001 sits on vector 0, shared by rows 3 (product 7) and 5 (product 1).
    first = queries[0][0]
    assert first == "r00001"
    assert filters.loc[(filters["qid"] == first) & (filters["filter"] == "product_near"), "sql"].item() == "product_id = 7"
    assert t1["top_ids"][0, names.index("none"), :2].tolist() == [3, 5]
    assert (t1["n_pass"][:, names.index("none")] == len(table)).all()

    printed = capsys.readouterr().out
    for name in names:
        assert any(line.split()[:1] == [name] for line in printed.splitlines()), name


# The unfiltered truth covers every query review, then every question, over distinct vectors.
def test_part2_covers_every_query(tmp_path):
    world = write_world(tmp_path)
    assert run(world, "--queries", "2") == 0
    t2 = load_npz(world["results"] / "size70k" / "truth_part2.npz")
    assert sorted(t2) == sorted(NPZ2)

    questions = mod.vc.read_questions()
    reviews = world["reviews"]
    assert t2["qids"].tolist() == reviews["qid"].tolist() + questions["qid"].tolist()
    assert t2["sets"].tolist() == reviews["split"].tolist() + ["questions"] * len(questions)
    assert t2["qids"].dtype.kind == "U" and t2["sets"].dtype.kind == "U"
    assert t2["top_ids"].shape == (len(t2["qids"]), 10) and t2["top_ids"].dtype == np.int64
    assert (t2["top_ids"] >= 0).all() and (t2["top_ids"] < len(world["base"])).all()
    assert t2["kth"].tolist() == t2["top_dists"][:, -1].tolist()

    base = world["base"].astype(np.float64)
    for i, q in enumerate(np.concatenate([world["review_vectors"], world["question_vectors"]])):
        assert_tie_aware(t2["top_ids"][i], t2["top_dists"][i], 1.0 - base @ q.astype(np.float64))
    assert t2["top_ids"][0, 0] == 0


# Two runs write the same CSV bytes and the same arrays (zip timestamps aside).
def test_truth_is_deterministic(tmp_path):
    world = write_world(tmp_path)
    out = world["results"] / "size70k"
    assert run(world, "--queries", "3") == 0
    csv = (out / "filters.csv").read_bytes()
    first = {name: load_npz(out / name) for name in ("truth_part1.npz", "truth_part2.npz")}
    assert run(world, "--queries", "3") == 0
    assert (out / "filters.csv").read_bytes() == csv
    for name, arrays in first.items():
        again = load_npz(out / name)
        assert sorted(again) == sorted(arrays)
        for key, a in arrays.items():
            assert again[key].dtype == a.dtype and np.array_equal(again[key], a), (name, key)


# A vector file whose rows don't match its CSV stops the run, naming the file.
@pytest.mark.parametrize("broken", ["review_vectors", "question_vectors", "base", "vec_row_gap", "width"])
def test_row_count_mismatch_exits_1(tmp_path, capsys, broken):
    world = write_world(tmp_path)
    files = world["files"]
    if broken == "review_vectors":
        np.save(files["review_vectors"], world["review_vectors"][:-1])
        name = "query_reviews.npy"
    elif broken == "question_vectors":
        np.save(files["question_vectors"], world["question_vectors"][:209])
        name = "questions.npy"
    elif broken == "base":
        np.save(files["base"], np.vstack([world["base"], world["base"][:1]]))
        name = "test.npy"
    elif broken == "vec_row_gap":
        table = world["table"].copy()
        top = table["vec_row"].max()
        table.loc[table["vec_row"] == top, "vec_row"] = top + 1
        table.to_csv(files["table"], index=False)
        name = "table_70k.csv"
    else:
        np.save(files["question_vectors"], world["question_vectors"][:, :100])
        name = "questions.npy"
    assert run(world) == 1
    assert name in capsys.readouterr().err
    assert not (world["results"] / "size70k" / "truth_part1.npz").exists()


# A --queries outside 1..100 at 70K is a usage error with part1_queries' message.
@pytest.mark.parametrize("n", ["0", "101"])
def test_bad_queries_exits_2(tmp_path, capsys, n):
    world = write_world(tmp_path)
    assert run(world, "--queries", n) == 2
    assert "queries per group must be 1..100" in capsys.readouterr().err
    assert not (world["results"] / "size70k").exists()


# A missing input names the file; a results folder inside ai_queries is refused.
def test_missing_input_and_ai_folder(tmp_path, capsys):
    world = write_world(tmp_path)
    world["files"]["review_vectors"].unlink()
    assert run(world) == 1
    assert "query_reviews.npy" in capsys.readouterr().err
    code = mod.main(["--size", "70k", "--results", str(world["ai"] / "inside"), "--ai", str(world["ai"])])
    assert code == 1
    assert "ai_queries" in capsys.readouterr().err
    assert not (world["ai"] / "inside").exists()


# If the unfiltered search fails, every old output stays as it was.
def test_failed_run_keeps_old_outputs(tmp_path, monkeypatch):
    world = write_world(tmp_path)
    out = world["results"] / "size70k"
    out.mkdir()
    old = {name: f"old {name}".encode() for name in ("filters.csv", "truth_part1.npz", "truth_part2.npz")}
    for name, data in old.items():
        (out / name).write_bytes(data)

    # I stand in a search that dies partway, like a Ctrl-C.
    def boom(*args, **kwargs):
        raise RuntimeError("stopped")

    monkeypatch.setattr(mod, "exact_topk_batch", boom)
    with pytest.raises(RuntimeError):
        run(world, "--queries", "2")
    assert {p.name: p.read_bytes() for p in out.iterdir()} == old


# The --queries help gives the real limits per size.
def test_queries_help_names_the_limits(capsys):
    with pytest.raises(SystemExit):
        mod.main(["--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "at most 100 at 70k, 50 at 500k (the default)" in help_text
