from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "bench/results/analyst_check"
AI_ROOT = REPO_ROOT / "bench/results/ai_queries"
TPCH_DIR = REPO_ROOT / "bench/tpch_data/sf0.1"
BRIDGE = REPO_ROOT / "build/bench/analyst_bridge"

DATASETS = ("tpch", "reviews")
SEED = 1
TIMEOUT_S = 30.0
MAX_RETRIES = 2
ROW_CAP = 10000
TEXT_CAP_BYTES = 4000
DISTINCT_CAP = 50
REL_TOL = 1e-4
ABS_TOL = 0.01

# A 'string' or -- comment (kept), or the DATE keyword just before a string (dropped).
_DATE_LITERAL = re.compile(r"'(?:[^']|'')*'|--[^\r\n]*|(\bDATE\s*(?='))", re.IGNORECASE)


# Load a script next to this one by path, under a name in sys.modules.
def load_sibling(module_name: str, file_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, HERE / file_name)
    if spec is None or spec.loader is None:
        raise ImportError(f"can't load {HERE / file_name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


# SQL for SQLite, which has no DATE '...' literal: DATE '1995-01-01' becomes '1995-01-01'.
def to_sqlite(sql: str) -> str:
    return _DATE_LITERAL.sub(lambda m: "" if m.group(1) else m.group(), sql)


@dataclass(frozen=True)
class Paths:
    results: Path

    # Folder for one dataset's files.
    def dataset_dir(self, ds: str) -> Path:
        return self.results / "datasets" / check_dataset(ds)

    # Answers from one model config on one split.
    def run_file(self, split: str, config: str) -> Path:
        return self.results / "runs" / f"{split}_{config}.jsonl"

    @property
    def questions(self) -> Path:
        return self.results / "questions.jsonl"

    @property
    def second_answers(self) -> Path:
        return self.results / "second_answers.jsonl"

    @property
    def key_log(self) -> Path:
        return self.results / "key_log.md"

    @property
    def key_answers(self) -> Path:
        return self.results / "key_answers.json"

    @property
    def probe(self) -> Path:
        return self.results / "probe.json"

    @property
    def rules(self) -> Path:
        return self.results / "subset_rules.md"

    @property
    def freeze(self) -> Path:
        return self.results / "freeze.json"

    @property
    def report(self) -> Path:
        return self.results / "report.md"

    @property
    def dev_report(self) -> Path:
        return self.results / "dev_report.md"

    @property
    def cells(self) -> Path:
        return self.results / "cells.csv"


# Paths under the given results folder.
def paths(results: Path = RESULTS) -> Paths:
    return Paths(Path(results))


# Return the dataset if it's one I use, else raise ValueError.
def check_dataset(ds: str) -> str:
    if ds not in DATASETS:
        raise ValueError(f"dataset must be one of {', '.join(DATASETS)}, got {ds!r}")
    return ds


# Return the path unchanged, or raise ValueError if it lands inside ai_queries.
def refuse_ai_root(path: Path) -> Path:
    if Path(path).resolve().is_relative_to(AI_ROOT.resolve()):
        raise ValueError(f"won't write inside {AI_ROOT}: {path}")
    return path


# sha1 hex of a file's bytes, read in 1 MB chunks.
def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# sha1 hex of a string's UTF-8 bytes.
def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# One JSON-lines line as a dict; raises ValueError if it isn't a JSON object.
def _parse_object(line: bytes) -> dict:
    record = json.loads(line.decode("utf-8"))
    if not isinstance(record, dict):
        raise ValueError("not a JSON object")
    return record


# Every record in a JSON-lines file; a broken last line is an interrupted write.
# Blank lines are skipped; a bad line anywhere else raises ValueError.
def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []

    lines = [(n, line) for n, line in enumerate(path.read_bytes().split(b"\n"), 1) if line.strip()]
    records = []
    for i, (n, line) in enumerate(lines):
        try:
            record = _parse_object(line)
        except ValueError as e:
            if i == len(lines) - 1:
                break
            raise ValueError(f"{path}:{n}: bad JSON line ({e})") from e
        records.append(record)
    return records


# Add one record as one line, and make sure it's on disk before returning.
# Raises ValueError if the file doesn't end with a newline; repair_jsonl fixes that.
def append_jsonl(path: Path, record: dict) -> None:
    path = Path(refuse_ai_root(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if f.seek(0, os.SEEK_END) > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                raise ValueError(f"{path} doesn't end with a newline; run repair_jsonl first")
            
        f.write((json.dumps(record) + "\n").encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


# Make a file safe to append to: end a complete last line with a newline,
# or cut a partial one. Returns the bytes removed.
def repair_jsonl(path: Path) -> int:
    path = Path(refuse_ai_root(path))
    if not path.exists():
        return 0

    data = path.read_bytes()
    keep = data.rfind(b"\n") + 1
    tail = data[keep:]
    if not tail:
        return 0

    try:
        _parse_object(tail)
    except ValueError:
        with path.open("r+b") as f:
            f.truncate(keep)
            f.flush()
            os.fsync(f.fileno())
        return len(tail)

    with path.open("ab") as f:
        f.write(b"\n")
        f.flush()
        os.fsync(f.fileno())
    return 0


# Write to a temp file in the same folder, then swap it in.
def write_text_atomic(path: Path, text: str) -> None:
    path = Path(refuse_ai_root(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
