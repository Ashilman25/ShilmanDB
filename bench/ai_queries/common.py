#Shared paths and text helpers for the AI-query check

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "bench/results/ai_queries"

# Reviews are cut at this many tokens before any model sees them.
TOKEN_CAP = 512


@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    labels: Path
    embeddings: Path
    reference: Path
    runs: Path


def paths(root: Path = RESULTS) -> Paths:
    root = Path(root)
    return Paths(root, root / "data", root / "labels", root / "embeddings", root / "reference", root / "runs")


def normalize_text(title: str, text: str) -> str:
    return " ".join(f"{title or ''} {text or ''}".lower().split())


def text_key(title: str, text: str) -> str:
    return hashlib.sha1(normalize_text(title, text).encode("utf-8")).hexdigest()


def ai_input(title: str, text: str) -> str:
    return f"Title: {title or ''}\nReview: {text or ''}"


def write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
