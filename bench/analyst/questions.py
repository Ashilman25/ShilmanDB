from __future__ import annotations

import importlib.util
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# Load an_common from next to this file, reusing it if it's already loaded.
def _load_common():
    path = Path(__file__).resolve().parent / "an_common.py"
    loaded = sys.modules.get("an_common")
    if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() == path:
        return loaded

    spec = importlib.util.spec_from_file_location("an_common", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"can't load {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["an_common"] = module
    spec.loader.exec_module(module)
    return module


an = _load_common()

KINDS = {"single": 25, "join": 25, "time": 15, "topn": 15, "ratio": 10, "unanswerable": 10}
DEV_PER_KIND = {"single": 8, "join": 8, "time": 4, "topn": 4, "ratio": 3, "unanswerable": 3}
PREFIX = {"tpch": "t", "reviews": "r"}
FIRST_SET = 100  # questions per dataset in the first set; later ones are always test
SPLITS = ("dev", "test")

_NONE = type(None)
REQUIRED = {
    "qid": str, "dataset": str, "kind": str, "question": str,
    "gold_sqlite": (str, _NONE), "gold_subset": (str, _NONE),
    "ordered": bool, "share_cols": list, "order_cols": list,
    "unanswerable_reason": (str, _NONE),
}
OPTIONAL = {"split": str, "answer": (dict, _NONE), "empty_ok": bool}

_QID = re.compile(r"[a-z](\d{3})")


# Every question in the file; any line that isn't a JSON object raises ValueError.
def read_questions(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []

    qs = []
    for n, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as e:
            raise ValueError(f"{path}:{n}: bad JSON line ({e})") from e
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{n}: not a JSON object")
        qs.append(record)
    return qs


# Write every question as one JSON line, swapping the file in at once.
def write_questions(path: Path, qs: list[dict]) -> None:
    an.write_text_atomic(Path(path), "".join(json.dumps(q) + "\n" for q in qs))


# The number in a qid like t042, or None if it isn't that form.
def qid_number(qid: str) -> int | None:
    m = _QID.fullmatch(qid) if isinstance(qid, str) else None
    return int(m.group(1)) if m else None


# A list of ints >= 0 (bools don't count).
def _indexes(v) -> bool:
    return all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in v)


# A non-empty string once stripped.
def _text(v) -> bool:
    return isinstance(v, str) and bool(v.strip())


# What's wrong with one question's fields and their rules, as messages.
def _field_problems(q: dict) -> list[str]:
    problems = []
    missing = [f for f in REQUIRED if f not in q]
    unknown = sorted(set(q) - REQUIRED.keys() - OPTIONAL.keys())
    if missing:
        problems.append(f"missing {', '.join(missing)}")
    if unknown:
        problems.append(f"unknown field {', '.join(unknown)}")
    for field, types in {**REQUIRED, **OPTIONAL}.items():
        if field in q and not isinstance(q[field], types):
            problems.append(f"{field} has the wrong type ({type(q[field]).__name__})")
    if problems:
        return problems

    ds, kind, number = q["dataset"], q["kind"], qid_number(q["qid"])
    if ds not in PREFIX:
        problems.append(f"dataset must be one of {', '.join(PREFIX)}")
    elif not number or q["qid"][0] != PREFIX[ds]:
        problems.append(f"qid must look like {PREFIX[ds]}001 for {ds}")
    if kind not in KINDS:
        problems.append(f"kind must be one of {', '.join(KINDS)}")
    if not _text(q["question"]):
        problems.append("question is empty")

    unanswerable = kind == "unanswerable"
    if unanswerable != (q["gold_sqlite"] is None):
        problems.append("gold_sqlite must be null exactly when kind is unanswerable")
    if unanswerable != _text(q["unanswerable_reason"]):
        problems.append("unanswerable_reason must be set exactly when kind is unanswerable")
    if not unanswerable and not (_text(q["gold_sqlite"]) and _text(q["gold_subset"])):
        problems.append("an answerable question needs gold_sqlite and gold_subset")
    if unanswerable and q["gold_subset"] is not None:
        problems.append("an unanswerable question has no gold_subset")

    if not _indexes(q["share_cols"]):
        problems.append("share_cols must be ints >= 0")
    if not _indexes(q["order_cols"]):
        problems.append("order_cols must be ints >= 0")
    elif q["ordered"] and not q["order_cols"]:
        problems.append("an ordered question needs order_cols")
    elif not q["ordered"] and q["order_cols"]:
        problems.append("order_cols must be [] when not ordered")

    if "split" in q and q["split"] not in SPLITS:
        problems.append("split must be dev or test")
    answer = q.get("answer")
    if isinstance(answer, dict) and not (answer.keys() == {"columns", "rows"} and isinstance(answer["columns"], list) and isinstance(answer["rows"], list)):
        problems.append("answer must be null or {columns, rows}")
    return problems


# Every problem with the questions; [] means fine.
# initial: the first set, exactly KINDS per dataset with qids up to FIRST_SET.
def validate(qs: list[dict], initial: bool = True) -> list[str]:
    problems = []
    seen = Counter()
    counts: dict[str, Counter] = {ds: Counter() for ds in PREFIX}
    for i, q in enumerate(qs, 1):
        if not isinstance(q, dict):
            problems.append(f"question {i}: not a JSON object")
            continue

        name = q.get("qid") if isinstance(q.get("qid"), str) else f"question {i}"
        problems += [f"{name}: {p}" for p in _field_problems(q)]
        seen[name] += 1
        if q.get("dataset") in counts:
            counts[q["dataset"]][q.get("kind")] += 1
        if initial and (qid_number(q.get("qid")) or 0) > FIRST_SET:
            problems.append(f"{name}: the first set has qids up to {FIRST_SET}")

    problems += [f"{qid}: used {n} times" for qid, n in seen.items() if n > 1]
    if initial:
        for ds, counter in counts.items():
            problems += [f"{ds}: {counter[kind]} {kind} questions, expected {n}" for kind, n in KINDS.items() if counter[kind] != n]
    return problems


# Give each question without a split one; existing splits never change.
# Per (dataset, kind), a seeded shuffle of the first set's qids picks the dev ones.
def assign_split(qs: list[dict]) -> list[dict]:
    out = [dict(q) for q in qs]
    groups = defaultdict(list)
    for q in out:
        if (qid_number(q["qid"]) or 0) <= FIRST_SET:
            groups[(q["dataset"], q["kind"])].append(q["qid"])

    dev = set()
    for (ds, kind), qids in groups.items():
        qids = sorted(qids)
        random.Random(f"{an.SEED}:{ds}:{kind}").shuffle(qids)
        dev.update(qids[:DEV_PER_KIND[kind]])

    for q in out:
        q.setdefault("split", "dev" if q["qid"] in dev else "test")
    return out
