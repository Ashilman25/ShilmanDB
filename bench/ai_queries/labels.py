#Label files: one CSV per question and tier, appended to as the model answers.
import csv
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .questions import QUESTIONS, Question


@dataclass(frozen=True)
class TierLabels:
    tier: str
    answers: Mapping[str, Mapping[str, str]]  # qid -> text key -> answer
    probs: Mapping[str, Mapping[str, np.ndarray]]  # qid -> text key -> option probabilities


def label_columns(question: Question) -> list[str]:
    ps = [f"p{i}" for i in range(len(question.options))]
    return ["text_key", "answer", *ps, "in_tokens", "out_tokens", "latency_ms"]


def _dump(rows: list[list[str]]) -> str:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    return buf.getvalue()


def _cut_partial_line(path: Path) -> None:
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        path.write_bytes(data[: data.rfind(b"\n") + 1])


def append_rows(path: Path, rows: list[dict], columns: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _cut_partial_line(path)
        
    new = not path.exists() or path.stat().st_size == 0  # decided after the cut
    lines = ([columns] if new else []) + [[row[c] for c in columns] for row in rows]
    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write(_dump(lines))
        f.flush()
        os.fsync(f.fileno())


#Read a label file. A last line that was cut off by a killed write is dropped
def read_label_file(path: Path) -> pd.DataFrame:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    rows = [r for r in csv.reader(io.StringIO(text, newline="")) if r]
    if rows and not text.endswith("\n"):
        rows.pop()
        
    if len(rows) > 1 and len(rows[-1]) != len(rows[0]):
        rows.pop()
        
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows[1:], columns=rows[0], dtype=object)
    for col in df.columns:
        if col.startswith("p") and col[1:].isdigit():
            df[col] = df[col].astype("float64")
            
        elif col in ("in_tokens", "out_tokens"):
            df[col] = df[col].astype("int64")
            
        elif col == "latency_ms":
            df[col] = df[col].astype("float64")
            
    if df["text_key"].duplicated().any():
        raise ValueError(f"{path}: duplicate text keys")
    return df


def load_tier(tier_dir: Path, tier: str) -> TierLabels:
    answers: dict[str, dict[str, str]] = {}
    probs: dict[str, dict[str, np.ndarray]] = {}
    
    for qid, question in QUESTIONS.items():
        path = Path(tier_dir) / f"{qid}.csv"
        if not path.exists():
            continue
        
        df = read_label_file(path)
        if df.empty:
            answers[qid], probs[qid] = {}, {}
            continue
        
        keys = df["text_key"].tolist()
        answers[qid] = dict(zip(keys, df["answer"].tolist()))
        p_cols = [f"p{i}" for i in range(len(question.options))]
        if p_cols:
            matrix = df[p_cols].to_numpy(dtype="float64")
            probs[qid] = {k: matrix[i] for i, k in enumerate(keys)}
        else:
            probs[qid] = {}
            
    return TierLabels(tier, answers, probs)
