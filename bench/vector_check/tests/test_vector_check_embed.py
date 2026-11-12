"""Tests for bench/vector_check/embed_vector_check.py. A fake model stands in for the real one."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC_PATH = Path(__file__).resolve().parents[1] / "embed_vector_check.py"
spec = importlib.util.spec_from_file_location("vc_embed_vector_check", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["vc_embed_vector_check"] = mod
spec.loader.exec_module(mod)

AI53 = ["F1", "C2", "F3", "F4", "F5a", "F5b", "F6", "E7", "F8", "F10"]


# A raw, non-unit vector that depends only on the text and a salt.
def raw_vector(text: str, salt: str) -> np.ndarray:
    seed = int(hashlib.sha1(f"{salt}:{text}".encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    return rng.normal(size=384) * (1.0 + rng.random() * 3.0)


# The unit float32 vector the script should end up with.
def unit(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32)
    return v / np.linalg.norm(v)


class FakeModel:
    """Deterministic encode / encode_query that record what they were given."""

    def __init__(self):
        self.encode_inputs: list[list[str]] = []
        self.encode_kwargs: list[dict] = []
        self.query_inputs: list[list[str]] = []
        self.query_kwargs: list[dict] = []

    def encode(self, texts, **kwargs):
        self.encode_inputs.append(list(texts))
        self.encode_kwargs.append(kwargs)
        return np.stack([raw_vector(t, "doc") for t in texts])

    def encode_query(self, texts, **kwargs):
        self.query_inputs.append(list(texts))
        self.query_kwargs.append(kwargs)
        return np.stack([raw_vector(t, "query") for t in texts])


class EncodeOnlyModel:
    """A model with no encode_query."""

    def __init__(self):
        self.encode_inputs: list[list[str]] = []
        self.encode_kwargs: list[dict] = []

    def encode(self, texts, **kwargs):
        self.encode_inputs.append(list(texts))
        self.encode_kwargs.append(kwargs)
        return np.stack([raw_vector(t, "doc") for t in texts])


# qid to text for the ten ai53 questions.
def ai53_texts() -> dict[str, str]:
    q = mod.vc.read_questions()
    q = q[q["group"] == "ai53"]
    return dict(zip(q["qid"], q["text"]))


QUERY_INPUTS = ["Title: a\nReview: first", "Title: NA\nReview: second", "Title: \nReview: third"]
CHECK = {0: "Title: x\nReview: row zero", 1: "Title: y\nReview: row one", 3: "Title: z\nReview: row three"}


# Results and ai folders whose saved vectors match what the fake model gives.
@pytest.fixture
def setup(tmp_path):
    results = tmp_path / f"results_{os.getpid()}"
    ai = tmp_path / f"ai_{os.getpid()}"
    data, ai_emb = results / "data", ai / "embeddings"
    data.mkdir(parents=True)
    ai_emb.mkdir(parents=True)
    pd.DataFrame({"qid": ["r00001", "r00002", "r00003"], "split": ["train", "validation", "test"],
                  "text_key": ["k1", "k2", "k3"], "input": QUERY_INPUTS}).to_csv(data / "query_reviews.csv", index=False)
    # Check texts out of vec_row order, so matching has to go by vec_row.
    pd.DataFrame({"vec_row": [3, 0, 1], "input": [CHECK[3], CHECK[0], CHECK[1]]}).to_csv(
        data / "check_texts_70k.csv", index=False)

    # Saved questions in reverse order, so matching has to go by qid.
    texts = ai53_texts()
    order = list(reversed(AI53))
    pd.DataFrame({"row": range(len(order)), "qid": order}).to_csv(ai_emb / "questions.csv", index=False)
    np.save(ai_emb / "questions.npy", np.stack([unit(raw_vector(texts[q], "query")) for q in order]))

    test = np.stack([unit(raw_vector(f"unused {i}", "doc")) for i in range(5)])
    for row, text in CHECK.items():
        test[row] = unit(raw_vector(text, "doc"))
    np.save(ai_emb / "test.npy", test)
    return results, ai


# Run the queries command on the fixture's folders.
def run_queries(setup, model) -> int:
    results, ai = setup
    return mod.main(["queries", "--results", str(results), "--ai", str(ai)], model=model)


# Names of the .npy files in the results embeddings folder.
def npy_files(results: Path) -> list[str]:
    folder = results / "embeddings"
    return sorted(p.name for p in folder.glob("*.npy")) if folder.exists() else []


def test_reviews_are_unit_and_ordered():
    fake = FakeModel()
    texts = ["b review", "a review", "c review"]
    out = mod.embed_reviews(texts, fake)
    assert out.dtype == np.float32 and out.shape == (3, 384)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
    for i, t in enumerate(texts):
        assert np.allclose(out[i], unit(raw_vector(t, "doc")), atol=1e-7)
    assert fake.encode_inputs == [texts] and fake.query_inputs == []
    kw = fake.encode_kwargs[0]
    assert kw["batch_size"] == 128 and kw["convert_to_numpy"] is True and kw["normalize_embeddings"] is True


def test_questions_use_encode_query_without_prefix():
    fake = FakeModel(); mod.embed_questions(["is it sturdy"], fake)
    assert fake.query_inputs == [["is it sturdy"]] and fake.encode_inputs == []
    assert fake.query_kwargs == [{"convert_to_numpy": True, "normalize_embeddings": True}]


def test_questions_are_unit_float32():
    out = mod.embed_questions(["is it sturdy", "does it fit"], FakeModel())
    assert out.dtype == np.float32 and out.shape == (2, 384)
    assert np.allclose(out[1], unit(raw_vector("does it fit", "query")), atol=1e-7)


def test_questions_fall_back_to_encode():
    model = EncodeOnlyModel()
    out = mod.embed_questions(["is it sturdy"], model)
    assert model.encode_inputs == [["is it sturdy"]]
    assert model.encode_kwargs[0]["batch_size"] == 128 and model.encode_kwargs[0]["normalize_embeddings"] is True
    assert np.allclose(out[0], unit(raw_vector("is it sturdy", "doc")), atol=1e-7)


def test_unit_keeps_a_zero_row_finite():
    out = mod.unit_rows(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert out.dtype == np.float32
    assert np.allclose(out, [[0.6, 0.8], [0.0, 0.0]])


def test_drift_is_max_abs_difference():
    a = np.zeros((2, 3), dtype=np.float32)
    b = a.copy()
    b[1, 2] = -2e-3
    b[0, 0] = 1e-3
    assert mod.drift(a, b) == pytest.approx(2e-3)
    assert mod.drift(a, a) == 0.0
    with pytest.raises(ValueError):
        mod.drift(a, np.zeros((3, 3)))


def test_drift_check_passes_on_match(setup, capsys):
    results, _ = setup
    fake = FakeModel()
    assert run_queries(setup, fake) == 0
    out = capsys.readouterr().out
    assert "question drift" in out and "review drift" in out

    emb = results / "embeddings"
    assert npy_files(results) == ["query_reviews.npy", "questions.npy"]
    reviews = np.load(emb / "query_reviews.npy")
    assert reviews.dtype == np.float32 and reviews.shape == (3, 384)
    for i, t in enumerate(QUERY_INPUTS):           # csv order
        assert np.allclose(reviews[i], unit(raw_vector(t, "doc")), atol=1e-7)

    q = mod.vc.read_questions()
    questions = np.load(emb / "questions.npy")
    assert questions.dtype == np.float32 and questions.shape == (210, 384)
    for i in (0, 57, 209):                         # questions.csv order
        assert np.allclose(questions[i], unit(raw_vector(q["text"].iloc[i], "query")), atol=1e-7)
    assert fake.query_inputs == [q["text"].tolist()]

    info = json.loads((emb / "embed_info.json").read_text())
    assert info["model"] == "BAAI/bge-small-en-v1.5" and info["passed"] is True
    assert info["counts"] == {"query_reviews": 3, "questions": 210, "ai53_questions": 10, "check_texts": 3}
    assert info["question_drift"] <= 1e-6 and info["review_drift"] <= 1e-6
    assert info["max_drift"] == 1e-4 and info["seconds"] >= 0
    assert not list(emb.glob("*.tmp*"))


def test_drift_under_the_limit_passes(setup):
    _, ai = setup
    saved = np.load(ai / "embeddings" / "questions.npy")
    saved[4, 7] += 5e-5
    np.save(ai / "embeddings" / "questions.npy", saved)
    assert run_queries(setup, FakeModel()) == 0


def test_unused_test_rows_are_ignored(setup):
    _, ai = setup
    test = np.load(ai / "embeddings" / "test.npy")
    test[2] += 1.0                                 # vec_row 2 isn't a check text
    np.save(ai / "embeddings" / "test.npy", test)
    assert run_queries(setup, FakeModel()) == 0


def test_drift_check_exits_1_on_mismatch(setup, capsys):
    results, ai = setup
    saved = np.load(ai / "embeddings" / "questions.npy")
    saved[0, 3] += 1e-3
    np.save(ai / "embeddings" / "questions.npy", saved)
    assert run_queries(setup, FakeModel()) == 1
    assert npy_files(results) == []
    info = json.loads((results / "embeddings" / "embed_info.json").read_text())
    assert info["passed"] is False
    assert info["question_drift"] == pytest.approx(1e-3, rel=1e-3) and info["review_drift"] <= 1e-6
    assert "over 0.0001" in capsys.readouterr().err


def test_review_drift_exits_1_on_mismatch(setup):
    results, ai = setup
    test = np.load(ai / "embeddings" / "test.npy")
    test[3, 100] -= 1e-3
    np.save(ai / "embeddings" / "test.npy", test)
    assert run_queries(setup, FakeModel()) == 1
    assert npy_files(results) == []
    info = json.loads((results / "embeddings" / "embed_info.json").read_text())
    assert info["review_drift"] == pytest.approx(1e-3, rel=1e-3) and info["passed"] is False


def test_failed_check_removes_old_vectors(setup):
    results, ai = setup
    emb = results / "embeddings"
    emb.mkdir()
    for name in ("query_reviews.npy", "questions.npy", "slice.npy"):
        np.save(emb / name, np.zeros((1, 384), dtype=np.float32))
    saved = np.load(ai / "embeddings" / "questions.npy")
    saved[9, 0] += 1e-3
    np.save(ai / "embeddings" / "questions.npy", saved)
    assert run_queries(setup, FakeModel()) == 1
    assert npy_files(results) == ["slice.npy"]     # the slice vectors aren't this command's


def test_missing_check_texts_says_what_to_run(setup, capsys, monkeypatch):
    results, _ = setup
    (results / "data" / "check_texts_70k.csv").unlink()
    monkeypatch.setattr(mod, "load_model", lambda *a, **k: pytest.fail("loaded the model"))
    assert mod.main(["queries", "--results", str(results), "--ai", str(setup[1])]) == 1
    assert "prepare_vector_data.py table --size 70k" in capsys.readouterr().err
    assert not (results / "embeddings").exists()


def test_missing_ai_qid_names_the_file(setup, capsys):
    results, ai = setup
    path = ai / "embeddings" / "questions.csv"
    saved = pd.read_csv(path, dtype=str, keep_default_na=False)
    saved[saved["qid"] != "F5b"].to_csv(path, index=False)
    fake = FakeModel()
    assert run_queries(setup, fake) == 1
    err = capsys.readouterr().err
    assert str(path) in err and "F5b" in err
    assert fake.encode_inputs == [] and fake.query_inputs == []
    assert npy_files(results) == []


def test_refuses_to_write_inside_ai_folder(setup, capsys):
    _, ai = setup
    inside = ai / "vc"
    (inside / "data").mkdir(parents=True)
    assert mod.main(["queries", "--results", str(inside), "--ai", str(ai)], model=FakeModel()) == 1
    assert "ai_queries" in capsys.readouterr().err
    assert not (inside / "embeddings").exists()


# A results folder holding only slice_texts.csv with the given rows.
def write_slice(tmp_path, vec_rows) -> Path:
    results = tmp_path / f"results_{os.getpid()}"
    (results / "data").mkdir(parents=True)
    pd.DataFrame({"vec_row": vec_rows, "text_key": [f"k{r}" for r in vec_rows],
                  "input": [f"Title: t\nReview: slice {r}" for r in vec_rows]}).to_csv(
        results / "data" / "slice_texts.csv", index=False)
    return results


def test_slice_writes_vectors_in_vec_row_order(tmp_path, capsys):
    results = write_slice(tmp_path, [2, 0, 3, 1])
    fake = FakeModel()
    assert mod.main(["slice", "--results", str(results)], model=fake) == 0
    out = np.load(results / "embeddings" / "slice.npy")
    assert out.dtype == np.float32 and out.shape == (4, 384)
    for r in range(4):
        assert np.allclose(out[r], unit(raw_vector(f"Title: t\nReview: slice {r}", "doc")), atol=1e-7)
    assert fake.encode_kwargs[0]["batch_size"] == 128 and fake.query_inputs == []
    assert "slice.npy: 4 rows" in capsys.readouterr().out


@pytest.mark.parametrize("vec_rows", [[0, 1, 3], [0, 1, 1], [1, 2, 3]])
def test_slice_needs_every_row_once(tmp_path, capsys, vec_rows):
    results = write_slice(tmp_path, vec_rows)
    fake = FakeModel()
    assert mod.main(["slice", "--results", str(results)], model=fake) == 1
    assert "slice_texts.csv" in capsys.readouterr().err
    assert fake.encode_inputs == [] and not (results / "embeddings" / "slice.npy").exists()


def test_slice_missing_texts_says_what_to_run(tmp_path, capsys):
    results = tmp_path / f"results_{os.getpid()}"
    assert mod.main(["slice", "--results", str(results)], model=FakeModel()) == 1
    assert "prepare_vector_data.py table --size 500k" in capsys.readouterr().err


def test_slice_has_no_ai_option(tmp_path):
    with pytest.raises(SystemExit):
        mod.main(["slice", "--results", str(tmp_path), "--ai", str(tmp_path)], model=FakeModel())


def test_import_does_not_load_sentence_transformers():
    assert "sentence_transformers" not in sys.modules
    # Top-level imports must stay stdlib, numpy and pandas, so the tests run without the model.
    tree = ast.parse(SPEC_PATH.read_text())
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    assert names - set(sys.stdlib_module_names) <= {"numpy", "pandas"}
