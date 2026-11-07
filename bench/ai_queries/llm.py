#Label the test table with a local model.
import argparse
import hashlib
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from .common import TOKEN_CAP, ai_input, paths, write_text_atomic
from .data_prep import read_tables
from .labels import append_rows, label_columns, read_label_file
from .queries import question_scopes
from .questions import QUESTIONS, Question

MAX_NEW_TOKENS = 24
BATCH = 16
BAKEOFF_QIDS = ("F3", "F8", "C2")

YESNO_TEMPLATE = (
    "You are labeling a product review.\n\n"
    "{review}\n\n"
    "Question: {wording}\n"
    "Answer with exactly one lowercase word: yes or no.\n"
    "Answer:"
)
CLASS_TEMPLATE = (
    "You are labeling a product review.\n\n"
    "{review}\n\n"
    "Question: {wording}\n"
    "{options}\n"
    "Answer with the letter of the best choice only.\n"
    "Answer:"
)
EXTRACT_TEMPLATE = (
    "You are labeling a product review.\n\n"
    "{review}\n\n"
    "Question: {wording}\n"
    "Answer with the model name only, or none.\n"
    "Answer:"
)
_TEMPLATES = {
    "yesno": YESNO_TEMPLATE, 
    "class": CLASS_TEMPLATE, 
    "extract": EXTRACT_TEMPLATE
}
LETTERS = "ABCDEF"

# A batch whose answer options get less raw probability than this means a broken prompt or token mapping.
MIN_OPTION_MASS = 0.5
MASS_BINS = 20  # histogram of per-row option mass over [0, 1], kept in run_info.json
_OPEN_THINKING_TAGS = ("<think>", "<thinking>", "<thought>")


def template_sha1() -> str:
    joined = "\n=====\n".join(_TEMPLATES[k] for k in sorted(_TEMPLATES))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


class Backend(Protocol):
    def count_tokens(self, text: str) -> int: ...
    def truncate(self, text: str, cap: int) -> str: ...
    def option_probs(self, prompts: list[str], options: list[str]) -> np.ndarray: ...
    def generate(self, prompts: list[str], max_tokens: int) -> list[str]: ...


#Each option's first token. Every option needs one, and no two may share it
def option_token_ids(encode: Callable[[str], list[int]], options: Sequence[str]) -> list[int]:
    ids = []
    for option in options:
        tokens = encode(option)
        if not tokens:
            raise ValueError(f"option {option!r} encodes to no tokens")
        ids.append(tokens[0])
        
    if len(set(ids)) != len(ids):
        raise ValueError(f"options {list(options)} share a first token: {ids}")
    return ids


#What the model is asked to say: yes/no, or one letter per class
def option_labels(q: Question) -> list[str]:
    if q.kind == "class":
        return list(LETTERS[: len(q.options)])
    return list(q.options)


def build_prompt(q: Question, review: str) -> str:
    if q.kind == "class":
        lines = "\n".join(f"{letter}) {name}" for letter, name in zip(LETTERS, q.options))
        return CLASS_TEMPLATE.format(review=review, wording=q.wording, options=lines)
    return _TEMPLATES[q.kind].format(review=review, wording=q.wording)


#First line of a generated answer, without a trailing period or an echoed "Answer:" label
def clean_extract(s: str) -> str:
    lines = s.strip().splitlines()
    first = lines[0].strip() if lines else ""
    first = re.sub(r"^answer\s*:\s*", "", first, flags=re.IGNORECASE)
    return first.rstrip(".").strip()


def slug(model: str) -> str:
    last = model.rstrip("/").split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]+", "-", last).strip("-")


#Raise if a chat prompt ends inside an unclosed thinking block
def check_no_open_thinking(rendered: str) -> None:
    tail = rendered.rstrip()
    for tag in _OPEN_THINKING_TAGS:
        if tail.endswith(tag):
            raise ValueError(f"the chat template ends with an open {tag} block; thinking is not switched off")


#A local model run with mlx_lm: chat template, greedy, no thinking block
class MlxBackend:
    def __init__(self, model_path: str):
        from mlx_lm import load
        self.model, self.tokenizer = load(model_path)
        self.last_option_mass: float | None = None        # mean raw mass of the latest option_probs call
        self.last_option_masses: np.ndarray | None = None  # and each row's raw mass

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def count_tokens(self, text: str) -> int:
        return len(self._encode(text))

    def truncate(self, text: str, cap: int) -> str:
        ids = self._encode(text)
        return text if len(ids) <= cap else self.tokenizer.decode(ids[:cap])

    def _chat(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:  # a template that doesn't know the keyword has no thinking mode
            rendered = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            
        check_no_open_thinking(rendered)
        return rendered

    def option_probs(self, prompts: list[str], options: list[str]) -> np.ndarray:
        import mlx.core as mx
        ids = option_token_ids(self._encode, options)
        rows = []
        masses = []
        
        for prompt in prompts:
            tokens = self._encode(self._chat(prompt))
            logits = self.model(mx.array([tokens]))[0, -1].astype(mx.float32)
            probs = np.array(mx.softmax(logits)[ids], dtype=np.float64)
            mass = float(probs.sum())
            if not math.isfinite(mass) or mass <= 0:
                raise ValueError(f"option probabilities are not usable: {probs}")
            
            masses.append(mass)
            rows.append(probs / mass)
        self.last_option_masses = np.array(masses)
        self.last_option_mass = float(np.mean(masses)) if masses else None
        return np.stack(rows)

    def generate(self, prompts: list[str], max_tokens: int) -> list[str]:
        from mlx_lm import generate
        return [generate(self.model, self.tokenizer, prompt=self._chat(p), max_tokens=max_tokens) for p in prompts]


# labeling


def _check_probs(q: Question, probs: np.ndarray, n: int) -> None:
    if probs.shape != (n, len(q.options)):
        raise ValueError(f"{q.qid}: expected {(n, len(q.options))} probabilities, got {probs.shape}")
    
    if not np.isfinite(probs).all():
        raise ValueError(f"{q.qid}: probabilities contain NaN or inf")

    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError(f"{q.qid}: probabilities do not sum to 1")


#Label the keys not yet in `out_path`, a batch at a time. Returns how many it labeled
def label(q: Question, keys: list[str], inputs: Mapping[str, str], backend: Backend, out_path: Path,
          batch: int = BATCH, progress: Callable[[int, int, float], None] | None = None,
          on_masses: Callable[[np.ndarray], None] | None = None) -> int:
    
    out_path = Path(out_path)
    done: set[str] = set()
    if out_path.exists():
        existing = read_label_file(out_path)
        if not existing.empty:
            done = set(existing["text_key"])
            
    todo = [k for k in dict.fromkeys(keys) if k not in done]
    columns = label_columns(q)
    options = option_labels(q)
    started = time.perf_counter()
    
    labeled = 0
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        prompts = [build_prompt(q, backend.truncate(inputs[k], TOKEN_CAP)) for k in chunk]
        t0 = time.perf_counter()
        
        if q.kind == "extract":
            raw = backend.generate(prompts, MAX_NEW_TOKENS)
            probs = None
        else:
            probs = np.asarray(backend.option_probs(prompts, options), dtype=np.float64)
            _check_probs(q, probs, len(chunk))
            mass = getattr(backend, "last_option_mass", None)
            if mass is not None and mass < MIN_OPTION_MASS:
                raise ValueError(f"{q.qid}: the answer options get only {mass:.3f} of the probability (need {MIN_OPTION_MASS}); check the prompt and option tokens")
            
        per_row_ms = (time.perf_counter() - t0) * 1000 / len(chunk)
        rows = []
        for i, (key, prompt) in enumerate(zip(chunk, prompts)):
            row = {
                "text_key": key, 
                "in_tokens": backend.count_tokens(prompt), 
                "latency_ms": per_row_ms
            }
            
            if probs is None:
                row["answer"] = clean_extract(raw[i])
                row["out_tokens"] = backend.count_tokens(raw[i])
            else:
                row["answer"] = q.options[int(np.argmax(probs[i]))]
                row["out_tokens"] = 1
                row.update({f"p{j}": float(p) for j, p in enumerate(probs[i])})
            rows.append(row)
            
        append_rows(out_path, rows, columns)
        masses = getattr(backend, "last_option_masses", None) if probs is not None else None
        if on_masses and masses is not None:
            on_masses(np.asarray(masses, dtype=np.float64))

        labeled += len(chunk)
        if progress:
            progress(labeled, len(todo), time.perf_counter() - started)
    return labeled


def latency_summary(tier_dir: Path) -> pd.DataFrame:
    rows = []
    for qid in QUESTIONS:
        path = Path(tier_dir) / f"{qid}.csv"
        if not path.exists():
            continue
        
        df = read_label_file(path)
        if df.empty:
            continue
        
        rows.append({
            "qid": qid, "rows": len(df), 
            "median_ms": float(df["latency_ms"].median()),
            "mean_in_tokens": float(df["in_tokens"].mean()),
            "mean_out_tokens": float(df["out_tokens"].mean())
        })
    return pd.DataFrame(rows, columns=["qid", "rows", "median_ms", "mean_in_tokens", "mean_out_tokens"])


# option mass, per row, summed up across runs 

#Add rows' raw option masses to a summary: rows, mean, min, rows under 0.5 and 0.8,
def merge_mass(old: Mapping | None, masses: np.ndarray) -> dict:
    m = np.clip(np.asarray(masses, dtype=np.float64), 0.0, 1.0)
    old = old or {"rows": 0, "mean": None, "min": None, "under_0_5": 0, "under_0_8": 0, "hist": [0] * MASS_BINS}
    rows = int(old["rows"]) + len(m)
    total = (old["mean"] or 0.0) * int(old["rows"]) + float(m.sum())
    
    mins = [float(x) for x in (old["min"], m.min() if len(m) else None) if x is not None]
    hist = np.asarray(old["hist"], dtype=np.int64) + np.histogram(m, bins=MASS_BINS, range=(0.0, 1.0))[0]
    
    out = {
        "rows": rows, 
        "mean": total / rows if rows else None, 
        "min": min(mins) if mins else None,
        "under_0_5": int(old["under_0_5"]) + int((m < 0.5).sum()),
        "under_0_8": int(old["under_0_8"]) + int((m < 0.8).sum()),
        "hist": [int(c) for c in hist]
    }
    out["p05"] = hist_quantile(out["hist"], 0.05)
    return out


#The upper edge of the first bin where the share of rows at or below it reaches q
def hist_quantile(hist: Sequence[int], q: float) -> float | None:
    total = sum(hist)
    if not total:
        return None
    
    cum = np.cumsum(hist)
    return float((int(np.searchsorted(cum, q * total)) + 1) / len(hist))


# a labeling session on one tier folder


def _model_facts(model: str) -> dict:
    path = Path(model).expanduser()
    if not path.is_dir():
        return {
            "path": None, 
            "quantization": None
        }
    
    quant = None
    config = path / "config.json"
    if config.exists():
        quant = json.loads(config.read_text(encoding="utf-8")).get("quantization")
        
    return {
        "path": str(path.resolve()), 
        "quantization": quant
    }


#The tier's run_info.json, refusing a folder that was labeled by another model or prompt
def _prepare_run_info(tier_dir: Path, model: str) -> dict:
    info_path = tier_dir / "run_info.json"
    sha = template_sha1()
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("model") != model or info.get("template_sha1") != sha:
            raise SystemExit(f"{tier_dir} was labeled with model {info.get('model')!r} and prompts "
                             f"{info.get('template_sha1')}; refusing to add {model!r} with {sha}. "
                             "Use another tier name.")
        return info
    
    return {
        "model": model, 
        **_model_facts(model), 
        "template_sha1": sha,
        "settings": {
            "batch": BATCH, "token_cap": TOKEN_CAP, "max_new_tokens_extract": MAX_NEW_TOKENS,
            "greedy": True, "thinking": False
        },
        "date": datetime.now().isoformat(timespec="seconds"),
        "truncated": {}, "truncated_share": 0.0, "option_mass_mean": {}, "option_mass": {}, "questions": []
    }


def _write_info(tier_dir: Path, info: dict) -> None:
    cut = sum(c for c, _ in info["truncated"].values())
    total = sum(t for _, t in info["truncated"].values())
    info["truncated_share"] = cut / total if total else 0.0
    write_text_atomic(tier_dir / "run_info.json", json.dumps(info, indent=2))


#Label the given questions into `tier_dir`, updating run_info.json after each one
def label_tier(tier_dir: Path, model: str, backend: Backend, qids: Sequence[str], keys: Mapping[str, list[str]], inputs: Mapping[str, str], batch: int = BATCH) -> dict:
    tier_dir = Path(tier_dir)
    tier_dir.mkdir(parents=True, exist_ok=True)
    
    info = _prepare_run_info(tier_dir, model)
    info["settings"]["batch"] = batch
    info.setdefault("option_mass", {})
    
    _write_info(tier_dir, info)
    for qid in qids:
        q = QUESTIONS[qid]
        qkeys = keys[qid]
        info["truncated"][qid] = [sum(backend.count_tokens(inputs[k]) > TOKEN_CAP for k in qkeys), len(qkeys)]

        def progress(n: int, total: int, secs: float, qid=qid) -> None:
            print(f"  {qid}: {n}/{total} rows, {n / secs:.2f} rows/s", flush=True)

        # saved after every batch, so a killed run keeps what it measured
        def add_masses(masses: np.ndarray, qid=qid) -> None:
            info["option_mass"][qid] = merge_mass(info["option_mass"].get(qid), masses)
            info["option_mass_mean"][qid] = info["option_mass"][qid]["mean"]
            _write_info(tier_dir, info)

        print(f"{qid}: {len(qkeys)} keys", flush=True)
        label(q, qkeys, inputs, backend, tier_dir / f"{qid}.csv", batch, progress, add_masses)
        if qid not in info["questions"]:
            info["questions"].append(qid)
        _write_info(tier_dir, info)
    return info


# command line


def ordered_keys(reviews: pd.DataFrame, wanted: set[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for key in reviews["text_key"]:
        if key in wanted and key not in seen:
            seen.add(key)
            out.append(key)
            
    return out[:limit] if limit else out


def first_inputs(reviews: pd.DataFrame) -> dict[str, str]:
    inputs: dict[str, str] = {}
    for key, title, text in zip(reviews["text_key"], reviews["title"], reviews["text"]):
        if key not in inputs:
            inputs[key] = ai_input(title, text)
            
    return inputs


def _cmd_label(args: argparse.Namespace) -> None:
    p = paths(Path(args.root))
    reviews, products = read_tables(p.data)
    scopes = question_scopes(reviews, products)
    qids = list(QUESTIONS) if args.questions == "all" else args.questions.split(",")
    
    keys = {qid: ordered_keys(reviews, scopes[qid], args.limit) for qid in qids}
    backend = MlxBackend(args.model)
    started = time.perf_counter()
    
    label_tier(p.labels / args.tier, args.model, backend, qids, keys, first_inputs(reviews))
    print(f"done in {time.perf_counter() - started:.0f} s")
    print(latency_summary(p.labels / args.tier).to_string(index=False))


#Drop a loaded model and hand its memory back before the next one loads
def free_model(backend) -> None:
    import gc
    for name in ("model", "tokenizer"):
        if hasattr(backend, name):
            delattr(backend, name)
    gc.collect()
    
    try:
        import mlx.core as mx
    except ImportError:
        return
    
    clear = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
    if clear:
        clear()


def _cmd_bakeoff(args: argparse.Namespace) -> None:
    p = paths(Path(args.root))
    reviews, _ = read_tables(p.data)
    inputs = first_inputs(reviews)
    
    wanted = pd.read_csv(args.rows_file, dtype=str, keep_default_na=False)["text_key"].tolist()
    wanted = list(dict.fromkeys(wanted))
    
    missing = [k for k in wanted if k not in inputs]
    if missing:
        raise SystemExit(f"{len(missing)} keys in {args.rows_file} are not in the test table")
    
    keys = {qid: wanted for qid in BAKEOFF_QIDS}
    for model in args.models.split(","):
        tier_dir = p.labels / "bakeoff" / slug(model)
        backend = MlxBackend(model)
        started = time.perf_counter()
        label_tier(tier_dir, model, backend, BAKEOFF_QIDS, keys, inputs)
        calls = len(wanted) * len(BAKEOFF_QIDS)
        print(f"{model}: {calls / (time.perf_counter() - started):.2f} rows/s", flush=True)
        free_model(backend)
        del backend


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("label", help="label the test table into a tier folder")
    a.add_argument("--root", default=str(paths().root))
    a.add_argument("--tier", required=True, help="big, small, or a scratch name")
    a.add_argument("--model", required=True, help="local path or repo id")
    a.add_argument("--questions", default="all", help="all, or a list like F1,C2")
    a.add_argument("--limit", type=int, default=None, help="only the first N keys per question")
    a.set_defaults(func=_cmd_label)
    
    b = sub.add_parser("bakeoff", help="label F3, F8 and C2 on a row list with several models")
    b.add_argument("--root", default=str(paths().root))
    b.add_argument("--models", required=True, help="comma-separated paths or repo ids")
    b.add_argument("--rows-file", required=True, help="CSV with a text_key column")
    b.set_defaults(func=_cmd_bakeoff)
    
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
