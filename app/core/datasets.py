from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.core import analyst_check

DATASETS = analyst_check.an.DATASETS
LABELS = {"tpch": "TPC-H, SF 0.1", "reviews": "Reviews"}
MODEL_ID = "gemma4-26b"

BRIDGE_COMMAND = "cmake --build build --target analyst_bridge"
DATASET_COMMANDS = (".venv/bin/python bench/analyst/prepare_datasets.py", ".venv/bin/python bench/analyst/describe_datasets.py")
RULES_COMMAND = ".venv/bin/python bench/analyst/probe_subset.py"


# Every folder and file the app reads or writes.
@dataclass(frozen=True)
class AppPaths:
    results: Path      # the analyst check's results: datasets, rules, run files
    code: Path         # the analyst check's code: dataset specs, prompt examples
    src: Path          # the engine's source
    bridge: Path       # the analyst_bridge binary
    app_results: Path  # the app's own output
    app_dir: Path      # the app package itself

    # Folder holding one dataset's files; raises ValueError for a name I don't use.
    def dataset_dir(self, name: str) -> Path:
        return self.results / "datasets" / analyst_check.an.check_dataset(name)

    # The JSON spec a dataset's description is written from.
    def spec(self, name: str) -> Path:
        return self.code / "datasets" / f"{analyst_check.an.check_dataset(name)}.json"

    # The tested model's answers on one split.
    def run_file(self, split: str) -> Path:
        return self.results / "runs" / f"{split}_{MODEL_ID}.jsonl"

    @property
    def rules(self) -> Path:
        return self.results / "subset_rules.md"

    @property
    def examples(self) -> Path:
        return self.code / "prompt_examples.json"

    @property
    def measured(self) -> Path:
        return self.app_dir / "core/measured.json"

    # The folder holding the two cache files the cache gate hashes (measured.CACHE_CODE).
    @property
    def cache_code_dir(self) -> Path:
        return self.app_dir / "model"

    @property
    def cache_proof(self) -> Path:
        return self.app_results / "cache_proof.json"


# The paths in this repo: the analyst check's files where it left them, the app's output beside them.
def default_paths() -> AppPaths:
    root = analyst_check.REPO_ROOT
    return AppPaths(
        results=root / "bench/results/analyst_check",
        code=root / "bench/analyst",
        src=root / "src",
        bridge=root / "build/bench/analyst_bridge",
        app_results=root / "bench/results/app",
        app_dir=root / "app",
    )


# One dataset, ready to ask about.
@dataclass(frozen=True)
class Dataset:
    name: str
    label: str
    tables_tsv: Path
    manifest: dict
    spec: dict
    context: object  # run_pipeline.Context: description, rules, examples


# Raised with every file a dataset needs that isn't there.
class MissingFiles(Exception):
    def __init__(self, missing: list[str]):
        super().__init__(f"missing {', '.join(missing)}")
        self.missing = list(missing)


# The .tbl files a tables.tsv names; a relative one sits next to the tsv, as the bridge reads it.
def _table_files(tsv: Path) -> list[Path]:
    if not tsv.is_file():
        return []

    files = []
    for line in tsv.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 2 and fields[1]:
            files.append(tsv.parent / fields[1])
    return files


# Every file the dataset needs that isn't there, in a fixed order.
def missing_files(paths: AppPaths, name: str) -> list[str]:
    d = paths.dataset_dir(name)
    tsv = d / "tables.tsv"
    needed = [tsv, d / "description.md", d / "manifest.json", *_table_files(tsv), paths.rules, paths.examples, paths.spec(name)]
    return [str(p) for p in needed if not p.is_file()]


# A JSON file's contents.
def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# One dataset with the context the analyst's prompts are built from; raises MissingFiles first.
def load_dataset(paths: AppPaths, name: str) -> Dataset:
    missing = missing_files(paths, name)
    if missing:
        raise MissingFiles(missing)

    d = paths.dataset_dir(name)
    examples = analyst_check.rp._load_examples(paths.examples, [name])[name]
    context = analyst_check.rp.Context(
        (d / "description.md").read_text(encoding="utf-8"),
        paths.rules.read_text(encoding="utf-8"),
        examples,
    )
    return Dataset(name, LABELS[name], d / "tables.tsv", _read_json(d / "manifest.json"), _read_json(paths.spec(name)), context)


# A path as the hint shows it: from the repo root when it's inside the repo.
def _shown(path: Path) -> str:
    return str(path.relative_to(analyst_check.REPO_ROOT)) if path.is_relative_to(analyst_check.REPO_ROOT) else str(path)


# What to run for whatever the dataset is missing, one instruction per line; empty if nothing is.
def build_hint(paths: AppPaths, name: str) -> str:
    missing = set(missing_files(paths, name))
    rules_missing = str(paths.rules) in missing
    code_missing = [p for p in (paths.examples, paths.spec(name)) if str(p) in missing]
    data_missing = bool(missing - {str(paths.rules), str(paths.examples), str(paths.spec(name))})
    lines = []

    if not paths.bridge.is_file():
        lines.append(f"Build the engine bridge: {BRIDGE_COMMAND}")

    if data_missing:
        lines.append("Make the dataset files, from the repo root:")
        lines += [f"  {command}" for command in DATASET_COMMANDS]
        lines.append("The second one describes both datasets, so it needs both in place.")
        if name == "tpch":
            lines.append("The first one reads TPC-H SF 0.1 from bench/tpch_data/sf0.1; make it with: bash bench/tpch/generate_tpch_data.sh 0.1")
        else:
            lines.append("The reviews also need the review download: data/reviews.csv and data/products.csv under bench/results/ai_queries (the first script's --ai folder). This repo can't rebuild that download.")

    if rules_missing:
        lines.append("Make subset_rules.md once the bridge and the tpch dataset are in place:")
        lines.append(f"  {RULES_COMMAND}")

    rebuilt = [file for file, gone in (("description.md", data_missing), ("subset_rules.md", rules_missing)) if gone]
    if rebuilt:
        lines.append(f"A rebuilt {' or '.join(rebuilt)} must match the tested one byte for byte, or the app hides the tested rates.")

    if code_missing:
        lines.append(f"These come with the repo; restore them from git: {', '.join(_shown(p) for p in code_missing)}")
    return "\n".join(lines)
