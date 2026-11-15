"""Tests for app/sixd.py and app/datasets.py."""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from app import sixd
from app.datasets import LABELS, MissingFiles, build_hint, default_paths, load_dataset, missing_files
from app.sixd import REPO_ROOT

FIXTURE = Path(__file__).resolve().parent / "fixtures/analyst_check"
ANALYST = REPO_ROOT / "bench/analyst"


# sha1 of every file in bench/analyst, leaving out its tests and bytecode caches.
def _analyst_hashes() -> dict[str, str]:
    files = (p for p in sorted(ANALYST.rglob("*")) if p.is_file())
    kept = (p for p in files if not {"tests", "__pycache__"} & set(p.relative_to(ANALYST).parts))
    return {str(p.relative_to(ANALYST)): hashlib.sha1(p.read_bytes()).hexdigest() for p in kept}


def test_sixd_shares_one_copy_of_each_module():
    assert sixd.sf is sixd.ck.sf
    assert sixd.dd is sixd.ck.dd
    assert sixd.pr is sixd.rp.pr
    assert sixd.bc is sixd.rp.bc
    assert sixd.fz is sixd.rp.fz
    assert Path(sixd.ck.__file__).parent == REPO_ROOT / "bench/analyst"


def test_sixd_leaves_6d_files_unchanged(monkeypatch):
    before = _analyst_hashes()
    assert "run_pipeline.py" in before and "datasets/tpch.json" in before

    # A reload makes new copies of the analyst modules; put the old ones back for the other tests.
    saved = dict(vars(sixd))
    loaded = {name for name in sys.modules if name.startswith("an_")}
    for name in loaded:
        monkeypatch.setitem(sys.modules, name, sys.modules[name])
    try:
        importlib.reload(sixd)
    finally:
        vars(sixd).clear()
        vars(sixd).update(saved)
        for name in [n for n in sys.modules if n.startswith("an_") and n not in loaded]:
            sys.modules.pop(name)

    assert _analyst_hashes() == before


def test_default_paths():
    p = default_paths()
    assert p.results == REPO_ROOT / "bench/results/analyst_check"
    assert p.code == REPO_ROOT / "bench/analyst"
    assert p.src == REPO_ROOT / "src"
    assert p.bridge == REPO_ROOT / "build/bench/analyst_bridge"
    assert p.app_results == REPO_ROOT / "bench/results/app"
    assert p.app_dir == REPO_ROOT / "app"


def test_paths_layout(fixture_paths):
    p = fixture_paths
    assert p.dataset_dir("tpch") == p.results / "datasets/tpch"
    assert p.spec("reviews") == p.code / "datasets/reviews.json"
    assert p.run_file("test") == p.results / "runs/test_gemma4-26b.jsonl"
    assert (p.rules, p.examples) == (p.results / "subset_rules.md", p.code / "prompt_examples.json")
    assert (p.measured, p.cache_proof) == (p.app_dir / "measured.json", p.app_results / "cache_proof.json")
    with pytest.raises(ValueError):
        p.dataset_dir("../tpch")


def test_load_dataset_reads_fixture(fixture_paths):
    ds = load_dataset(fixture_paths, "tpch")
    assert ds.context.description == (FIXTURE / "datasets/tpch/description.md").read_text(encoding="utf-8")
    assert ds.context.rules == (FIXTURE / "subset_rules.md").read_text(encoding="utf-8")
    assert ds.context.examples == json.loads((ANALYST / "prompt_examples.json").read_text(encoding="utf-8"))["tpch"]
    assert ds.manifest["dataset"] == "tpch"
    assert ds.spec["dataset"] == "tpch"
    assert ds.label == "TPC-H, SF 0.1"
    assert (ds.name, ds.tables_tsv) == ("tpch", fixture_paths.dataset_dir("tpch") / "tables.tsv")
    assert isinstance(ds.context, sixd.rp.Context)


def test_missing_files_names_each(fixture_paths):
    d = fixture_paths.dataset_dir("tpch")
    assert missing_files(fixture_paths, "tpch") == []

    (d / "manifest.json").unlink()
    (d / "nation.tbl").unlink()
    assert missing_files(fixture_paths, "tpch") == [str(d / "manifest.json"), str(d / "nation.tbl")]


def test_load_dataset_raises_missing(fixture_paths):
    with pytest.raises(MissingFiles) as err:
        load_dataset(fixture_paths, "reviews")
    assert err.value.missing
    assert err.value.missing[0] == str(fixture_paths.dataset_dir("reviews") / "tables.tsv")
    assert "prepare_datasets.py" in build_hint(fixture_paths, "reviews")


def test_build_hint_names_what_to_run(fixture_paths, tmp_path):
    bridge = tmp_path / "analyst_bridge"
    bridge.write_text("", encoding="utf-8")
    ready = replace(fixture_paths, bridge=bridge)
    assert build_hint(ready, "tpch") == ""

    no_bridge = replace(fixture_paths, bridge=tmp_path / "no_bridge")
    assert build_hint(no_bridge, "tpch") == "Build the engine bridge: cmake --build build --target analyst_bridge"

    reviews = build_hint(ready, "reviews")
    assert "describe_datasets.py" in reviews and "byte for byte" in reviews
    assert "bench/results/ai_queries" in reviews and "cmake" not in reviews

    (fixture_paths.dataset_dir("tpch") / "description.md").unlink()
    (fixture_paths.rules).unlink()
    tpch = build_hint(no_bridge, "tpch")
    assert "cmake" in tpch and "prepare_datasets.py" in tpch and "probe_subset.py" in tpch
    assert "ai_queries" not in tpch


# Every hint at its longest is plain ASCII: no em dashes or arrows.
def test_build_hint_is_plain_text(fixture_paths, tmp_path):
    nothing = replace(fixture_paths, bridge=tmp_path / "none", results=tmp_path / "empty")
    for name in ("tpch", "reviews"):
        hint = build_hint(nothing, name)
        assert "cmake" in hint and "probe_subset.py" in hint and hint.isascii()


def test_labels_cover_every_dataset():
    assert set(LABELS) == set(sixd.an.DATASETS)
