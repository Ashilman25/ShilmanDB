from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from app import sixd
from app.datasets import AppPaths

# The analyst check's code the answers depend on, next to its scripts.
FILES_CODE = [
    "prompts.py", "prompt_examples.json", "sql_facts.py", "checks.py", "scoring.py",
    "an_common.py", "describe_datasets.py", "model_mlx.py", "datasets/tpch.json", "datasets/reviews.json",
]
# Dataset files in its results folder.
FILES_RESULTS = [
    "datasets/tpch/description.md", "datasets/reviews/description.md",
    "datasets/tpch/manifest.json", "datasets/reviews/manifest.json", "subset_rules.md",
]
# Code the check only watched, but the app runs, so I lock it too.
FILES_WATCHED = ["run_pipeline.py", "bridge_client.py", "analyst_bridge.cpp"]

SKIPPED = {"__pycache__", ".DS_Store"}
CACHE_CODE = ("model_cache.py", "prompt_cache.py")


# Whether the tested rates apply, and if not, why.
@dataclass(frozen=True)
class Lock:
    holds: bool
    reason: str | None


# Whether the prompt cache may be on, and why.
@dataclass(frozen=True)
class CacheGate:
    on: bool
    reason: str


# sha1 over every file under root, by sorted relative path: path, NUL, bytes, NUL.
# Skips bytecode caches and .DS_Store.
def tree_sha1(root: Path) -> str:
    root = Path(root)
    files = {}
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if path.is_file() and not SKIPPED & set(rel.parts):
            files[rel.as_posix()] = path

    digest = hashlib.sha1()
    for rel in sorted(files):
        digest.update(rel.encode("utf-8") + b"\0" + files[rel].read_bytes() + b"\0")
    return digest.hexdigest()


# sha1 as the check's freeze hashes it; None when the file can't be read.
def _sha1_or_none(path: Path) -> str | None:
    try:
        return sixd.an.sha1_file(path)
    except OSError:
        return None


# sha1 of every locked file, keyed "code/<name>" or "results/<name>"; None when missing.
def file_hashes(paths: AppPaths) -> dict[str, str | None]:
    named = (
        [(f"code/{n}", paths.code / n) for n in FILES_CODE]
        + [(f"results/{n}", paths.results / n) for n in FILES_RESULTS]
        + [(f"code/{n}", paths.code / n) for n in FILES_WATCHED]
    )
    return {key: _sha1_or_none(path) for key, path in named}


# A JSON object from a file; None when it's missing, unreadable or not an object.
def read_json(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# The installed mlx and mlx-lm versions; {} unless both are installed.
def mlx_versions() -> dict[str, str]:
    try:
        return {"mlx": metadata.version("mlx"), "mlx_lm": metadata.version("mlx-lm")}
    except metadata.PackageNotFoundError:
        return {}


# sha1 over model_cache.py's bytes, then prompt_cache.py's; a missing file adds nothing.
def cache_code_sha1(app_dir: Path) -> str:
    digest = hashlib.sha1()
    for name in CACHE_CODE:
        path = Path(app_dir) / name
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


# The tested model settings as model_mlx.py lists them, without the id the record adds.
def _tested_model(measured: dict) -> dict:
    return {k: v for k, v in measured["model"].items() if k != "id"}


# Whether today's setup is the tested one; only the first reason it isn't is given.
def check_lock(paths: AppPaths, measured: dict, model_config: dict | None, proof: dict | None, versions: dict) -> Lock:
    for key, sha1 in file_hashes(paths).items():
        name = key.partition("/")[2]
        if sha1 is None:
            return Lock(False, f"{name} is missing")
        if sha1 != measured["files"].get(key):
            return Lock(False, f"{name} changed")

    if model_config != _tested_model(measured):
        return Lock(False, "the model settings changed")
    if tree_sha1(paths.src) != measured["engine_src_sha1"]:
        return Lock(False, "the engine changed since the test")

    # A replay on this exact mlx found the uncached path itself no longer gives the saved answers.
    if proof is not None and proof.get("versions") == versions and proof.get("uncached_differences"):
        return Lock(False, "this machine's MLX no longer reproduces the test's answers")
    return Lock(True, None)


# Whether the cache may be on: a replay matched every test question on this setup.
def cache_gate(paths: AppPaths, measured: dict, proof: dict | None, versions: dict, cache_code_sha1: str, model_config: dict | None) -> CacheGate:
    if proof is None:
        return CacheGate(False, "the replay hasn't been run")
    if proof.get("differences") or proof.get("uncached_differences"):
        return CacheGate(False, "the replay found differences")

    questions = measured["rates"]["test_questions"]
    if proof.get("total") != questions or proof.get("identical") != proof.get("total"):
        return CacheGate(False, f"the replay didn't cover all {questions} test questions")
    if proof.get("versions") != versions:
        return CacheGate(False, "mlx changed since the replay")
    if proof.get("cache_code_sha1") != cache_code_sha1:
        return CacheGate(False, "the app's cache code changed since the replay")
    if proof.get("files") != file_hashes(paths) or model_config != _tested_model(measured):
        return CacheGate(False, "a tested file changed since the replay")
    if proof.get("engine_src_sha1") != tree_sha1(paths.src):
        return CacheGate(False, "the engine changed since the replay")
    return CacheGate(True, f"the replay matched {questions} of {questions}")
