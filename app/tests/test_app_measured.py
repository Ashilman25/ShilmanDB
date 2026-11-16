"""Tests for app/measured.py and the committed app/measured.json."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest

# tested_line is reached through check_views: pytest collects any test* name in a test module.
from app import check_views, measured, sixd
from app.datasets import AppPaths, default_paths
from app.sixd import REPO_ROOT

FREEZE = REPO_ROOT / "bench/results/analyst_check/freeze.json"
HEX40 = re.compile(r"[0-9a-f]{40}")

RATES = {"test_questions": 300, "model_right": 252, "shown_right": 227, "shown_wrong": 35, "caught": 20,
         "refused_right": 0, "warned_right": 18, "second_catches": 15, "second_alone": 15, "engine_alone": 5,
         "without_terms": 17, "topn_caught": 6, "topn_wrong": 13, "median_s": 5.3, "p90_s": 7.6}
MODEL = {"path": "org/some-model-4bit", "revision": "abc123", "thinking": False, "max_tokens": 1024}
VERSIONS = {"mlx": "0.30.0", "mlx_lm": "0.32.2"}
LIBRARIES = {"mlx": "0.30.0", "mlx-lm": "0.32.2", "mlx-metal": "0.30.0", "transformers": "5.22.1",
             "tokenizers": "0.23.3", "jinja2": "3.1.6"}
CACHE_SHA1 = "c" * 40
MLX_REASON = "this machine's MLX no longer reproduces the test's answers"


# Writes bytes to path, making its folders.
def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


# Tmp trees with every locked file and a small engine, plus the record matching them.
def _setup(tmp_path: Path) -> tuple[AppPaths, dict]:
    paths = replace(default_paths(), code=tmp_path / "code", results=tmp_path / "results", src=tmp_path / "src",
                    app_dir=tmp_path / "app", app_results=tmp_path / "app_results")
    for folder, names in ((paths.code, measured.FILES_CODE + measured.FILES_WATCHED), (paths.results, measured.FILES_RESULTS)):
        for name in names:
            _write(folder / name, f"contents of {name}\n".encode())
    _write(paths.src / "main.cpp", b"int main() {}\n")
    _write(paths.src / "storage/page.hpp", b"#pragma once\n")

    record = {"files": measured.file_hashes(paths), "engine_src_sha1": measured.tree_sha1(paths.src),
              "model": {**MODEL, "id": "gemma4-26b"}, "rates": dict(RATES), "source": "docs/research.md"}
    return paths, record


# A replay proof that matches the setup, with any fields changed.
def _proof(paths: AppPaths, **changes) -> dict:
    proof = {"identical": 300, "total": 300, "differences": [], "uncached_differences": [],
             "versions": dict(VERSIONS), "library_versions": dict(LIBRARIES), "files": measured.file_hashes(paths),
             "engine_src_sha1": measured.tree_sha1(paths.src), "cache_code_sha1": CACHE_SHA1,
             "timings": {}, "written_at": "2026-11-15T00:00:00+00:00"}
    return {**proof, **changes}


# The lock over the setup, with the tested model config and no replay unless given.
def _lock(paths: AppPaths, record: dict, model_config=MODEL, proof=None, versions=VERSIONS) -> measured.Lock:
    return measured.check_lock(paths, record, model_config, proof, versions)


# The cache gate over the setup, with the tested model config unless given.
def _gate(paths: AppPaths, record: dict, proof, libraries=LIBRARIES, cache=CACHE_SHA1, model_config=MODEL):
    return measured.cache_gate(paths, record, proof, libraries, cache, model_config)


# The committed measured.json.
def _record() -> dict:
    return json.loads(default_paths().measured.read_text(encoding="utf-8"))


def test_locked_file_lists():
    assert measured.FILES_CODE == ["prompts.py", "prompt_examples.json", "sql_facts.py", "checks.py", "scoring.py",
                                   "an_common.py", "describe_datasets.py", "model_mlx.py", "datasets/tpch.json",
                                   "datasets/reviews.json"]
    assert measured.FILES_RESULTS == ["datasets/tpch/description.md", "datasets/reviews/description.md",
                                      "datasets/tpch/manifest.json", "datasets/reviews/manifest.json", "subset_rules.md"]
    assert measured.FILES_WATCHED == ["run_pipeline.py", "bridge_client.py", "analyst_bridge.cpp"]


def test_tree_sha1_order_and_bytes(tmp_path):
    files = {"b.txt": b"two\n", "a/c.cpp": b"three\n", "a.txt": b"one\n"}
    first, second = tmp_path / "first", tmp_path / "second"
    for name in files:
        _write(first / name, files[name])
    for name in reversed(files):
        _write(second / name, files[name])

    # Sorted by the path as text, so "a.txt" comes before "a/c.cpp".
    base = measured.tree_sha1(first)
    assert base == hashlib.sha1(b"a.txt\0one\n\0a/c.cpp\0three\n\0b.txt\0two\n\0").hexdigest()
    assert measured.tree_sha1(second) == base

    _write(second / "__pycache__/x.pyc", b"cache")
    _write(second / "a/__pycache__/c.pyc", b"cache")
    _write(second / ".DS_Store", b"finder")
    _write(second / "a/.DS_Store", b"finder")
    assert measured.tree_sha1(second) == base

    (second / "a.txt").write_bytes(b"onE\n")
    assert measured.tree_sha1(second) != base
    (second / "a.txt").write_bytes(b"one\n")
    assert measured.tree_sha1(second) == base

    (second / "b.txt").rename(second / "d.txt")
    assert measured.tree_sha1(second) != base


def test_file_hashes_keys_and_missing(tmp_path):
    paths, _ = _setup(tmp_path)
    (paths.results / "subset_rules.md").unlink()
    hashes = measured.file_hashes(paths)

    assert list(hashes) == ([f"code/{n}" for n in measured.FILES_CODE] + [f"results/{n}" for n in measured.FILES_RESULTS]
                            + [f"code/{n}" for n in measured.FILES_WATCHED])
    assert hashes["results/subset_rules.md"] is None
    assert hashes["code/analyst_bridge.cpp"] == sixd.an.sha1_file(paths.code / "analyst_bridge.cpp")
    assert hashes["results/datasets/tpch/description.md"] == sixd.an.sha1_file(paths.results / "datasets/tpch/description.md")


def test_lock_holds_when_all_match(tmp_path):
    paths, record = _setup(tmp_path)
    assert _lock(paths, record) == measured.Lock(holds=True, reason=None)
    assert _lock(paths, record, proof=_proof(paths)) == measured.Lock(True, None)


@pytest.mark.parametrize("key", ["code/checks.py", "results/datasets/tpch/description.md", "code/analyst_bridge.cpp"])
def test_lock_file_changed(tmp_path, key):
    paths, record = _setup(tmp_path)
    folder, name = key.split("/", 1)
    with open((paths.code if folder == "code" else paths.results) / name, "ab") as f:
        f.write(b" ")
    assert _lock(paths, record) == measured.Lock(False, f"{name} changed")


def test_lock_file_missing(tmp_path):
    paths, record = _setup(tmp_path)
    (paths.results / "subset_rules.md").unlink()
    assert _lock(paths, record) == measured.Lock(False, "subset_rules.md is missing")


def test_lock_model_changed(tmp_path):
    paths, record = _setup(tmp_path)
    assert _lock(paths, record, model_config={**MODEL, "max_tokens": 2048}) == measured.Lock(False, "the model settings changed")
    assert _lock(paths, record, model_config=None) == measured.Lock(False, "the model settings changed")


def test_lock_engine_changed(tmp_path):
    paths, record = _setup(tmp_path)
    _write(paths.src / "storage/page.cpp", b"// new\n")
    assert _lock(paths, record) == measured.Lock(False, "the engine changed since the test")


def test_lock_mlx_no_longer_reproduces(tmp_path):
    paths, record = _setup(tmp_path)
    proof = _proof(paths, differences=["q7"], uncached_differences=["q7"])
    assert _lock(paths, record, proof=proof) == measured.Lock(False, MLX_REASON)


def test_lock_ignores_old_replay_on_other_versions(tmp_path):
    paths, record = _setup(tmp_path)
    proof = _proof(paths, differences=["q7"], uncached_differences=["q7"])
    assert _lock(paths, record, proof=proof, versions={"mlx": "0.31.0", "mlx_lm": "0.32.2"}) == measured.Lock(True, None)


def test_lock_reports_only_the_first_reason(tmp_path):
    paths, record = _setup(tmp_path)
    _write(paths.src / "extra.cpp", b"// new\n")
    proof = _proof(paths, uncached_differences=["q7"])
    assert _lock(paths, record, proof=proof).reason == "the engine changed since the test"
    assert _lock(paths, record, model_config=None, proof=proof).reason == "the model settings changed"

    (paths.results / "subset_rules.md").write_bytes(b"changed")
    (paths.code / "model_mlx.py").write_bytes(b"changed")
    assert _lock(paths, record, model_config=None, proof=proof).reason == "model_mlx.py changed"


def test_lock_fits_tested_line(tmp_path):
    paths, record = _setup(tmp_path)
    assert check_views.tested_line(_lock(paths, record), "ask", RATES)["applies"] is True

    _write(paths.src / "extra.cpp", b"// new\n")
    line = check_views.tested_line(_lock(paths, record), "ask", RATES)
    assert line == {"text": "This setup differs from the tested one (the engine changed since the test), "
                            "so the test's rates don't apply.", "applies": False}


def test_cache_gate_each_reason(tmp_path):
    paths, record = _setup(tmp_path)
    off = partial(measured.CacheGate, False)

    assert _gate(paths, record, None) == off("the replay hasn't been run")
    assert _gate(paths, record, _proof(paths, identical=299, differences=["q7"])) == off("the replay found differences")
    assert _gate(paths, record, _proof(paths, uncached_differences=["q7"])) == off("the replay found differences")
    assert _gate(paths, record, _proof(paths, identical=2, total=2)) == off("the replay didn't cover all 300 test questions")
    assert _gate(paths, record, _proof(paths, identical=299)) == off("the replay didn't cover all 300 test questions")
    library = off("a model library changed since the replay")
    for name in measured.MODEL_LIBRARIES:
        assert _gate(paths, record, _proof(paths), libraries={**LIBRARIES, name: "9.9.9"}) == library
        assert _gate(paths, record, _proof(paths), libraries={**LIBRARIES, name: None}) == library
    assert _gate(paths, record, _proof(paths), libraries={}) == library
    assert _gate(paths, record, _proof(paths, library_versions=None)) == library  # a proof from before the wider set
    assert _gate(paths, record, _proof(paths), cache="d" * 40) == off("the app's cache code changed since the replay")
    assert _gate(paths, record, _proof(paths), model_config={**MODEL, "thinking": True}) == off("a tested file changed since the replay")

    # Change the engine, then a locked file, after the replay: the file is checked first.
    proof = _proof(paths)
    _write(paths.src / "extra.cpp", b"// new\n")
    assert _gate(paths, record, proof) == off("the engine changed since the replay")
    (paths.code / "checks.py").write_bytes(b"changed")
    assert _gate(paths, record, proof) == off("a tested file changed since the replay")
    (paths.code / "checks.py").unlink()
    assert _gate(paths, record, proof) == off("a tested file changed since the replay")


def test_cache_gate_on(tmp_path):
    paths, record = _setup(tmp_path)
    assert _gate(paths, record, _proof(paths)) == measured.CacheGate(on=True, reason="the replay matched 300 of 300")


def test_cache_code_sha1_tracks_both_files(tmp_path):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    nothing = measured.cache_code_sha1(app_dir)
    assert nothing == hashlib.sha1(b"").hexdigest()

    (app_dir / "model_cache.py").write_bytes(b"model\n")
    with_model = measured.cache_code_sha1(app_dir)
    assert with_model != nothing

    (app_dir / "prompt_cache.py").write_bytes(b"prompt\n")
    both = measured.cache_code_sha1(app_dir)
    assert both == hashlib.sha1(b"model\nprompt\n").hexdigest()

    (app_dir / "prompt_cache.py").write_bytes(b"prompt!\n")
    assert measured.cache_code_sha1(app_dir) not in (both, with_model)
    (app_dir / "prompt_cache.py").write_bytes(b"prompt\n")
    (app_dir / "model_cache.py").write_bytes(b"model!\n")
    assert measured.cache_code_sha1(app_dir) != both


def test_read_json(tmp_path):
    good, bad, listed, binary = (tmp_path / n for n in ("good.json", "bad.json", "list.json", "binary.json"))
    good.write_text('{"a": 1}', encoding="utf-8")
    bad.write_text("{not json", encoding="utf-8")
    listed.write_text("[1, 2]", encoding="utf-8")
    binary.write_bytes(b"\xff\xfe{}")

    assert measured.read_json(good) == {"a": 1}
    assert measured.read_json(tmp_path / "missing.json") is None
    assert measured.read_json(bad) is None
    assert measured.read_json(listed) is None
    assert measured.read_json(binary) is None


def test_mlx_versions(monkeypatch):
    installed = {"mlx": "0.30.0", "mlx-lm": "0.32.2"}

    def version(name: str) -> str:
        if name not in installed:
            raise measured.metadata.PackageNotFoundError(name)
        return installed[name]

    monkeypatch.setattr(measured.metadata, "version", version)
    assert measured.mlx_versions() == {"mlx": "0.30.0", "mlx_lm": "0.32.2"}
    del installed["mlx-lm"]
    assert measured.mlx_versions() == {}
    installed.clear()
    assert measured.mlx_versions() == {}


def test_library_versions(monkeypatch):
    installed = dict(LIBRARIES)

    def version(name: str) -> str:
        if name not in installed:
            raise measured.metadata.PackageNotFoundError(name)
        return installed[name]

    monkeypatch.setattr(measured.metadata, "version", version)
    assert measured.MODEL_LIBRARIES == ("mlx", "mlx-lm", "mlx-metal", "transformers", "tokenizers", "jinja2")
    assert measured.library_versions() == LIBRARIES
    del installed["jinja2"]
    assert measured.library_versions() == {**LIBRARIES, "jinja2": None}
    installed.clear()
    assert measured.library_versions() == dict.fromkeys(measured.MODEL_LIBRARIES)


def test_measured_imports_no_mlx():
    tree = ast.parse(Path(measured.__file__).read_text(encoding="utf-8"))
    names = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    names += [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not [n for n in names if n.split(".")[0] in ("mlx", "mlx_lm")]


def test_measured_json_shape():
    record = _record()
    keys = ([f"code/{n}" for n in measured.FILES_CODE] + [f"results/{n}" for n in measured.FILES_RESULTS]
            + [f"code/{n}" for n in measured.FILES_WATCHED])
    assert len(keys) == 18
    assert list(record["files"]) == keys
    assert all(HEX40.fullmatch(sha1) for sha1 in record["files"].values())
    assert HEX40.fullmatch(record["engine_src_sha1"])
    assert record["rates"] == RATES
    assert record["model"]["id"] == "gemma4-26b"
    assert record["source"] == "docs/research.md"


def test_measured_matches_freeze():
    if not FREEZE.is_file():
        pytest.skip("bench/results/analyst_check/freeze.json isn't here")
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    record = _record()

    for name in measured.FILES_CODE:
        assert record["files"][f"code/{name}"] == frozen["files"][name]
    for name in measured.FILES_RESULTS:
        assert record["files"][f"results/{name}"] == frozen["files"][name]
    for name in measured.FILES_WATCHED:
        assert record["files"][f"code/{name}"] == frozen["watched"][name]
    assert record["model"] == frozen["model"]


# Checks file_hashes reads each locked file from where it really lives.
def test_file_hashes_match_measured_here():
    if not FREEZE.is_file():
        pytest.skip("bench/results/analyst_check/freeze.json isn't here")
    assert measured.file_hashes(default_paths()) == _record()["files"]


def test_measured_model_matches_model_mlx():
    configs = sixd.fz.read_model_configs(REPO_ROOT / "bench/analyst/model_mlx.py")
    model = {k: v for k, v in _record()["model"].items() if k != "id"}
    assert configs["gemma4-26b"] == model
