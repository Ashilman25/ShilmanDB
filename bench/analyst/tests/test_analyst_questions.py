"""Tests for bench/analyst/questions.py."""
from __future__ import annotations

import importlib.util
import os
import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


# Load a module by path under an an_ name, clear of other bench tests.
def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, HERE.parent / file_name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mod = _load("an_questions", "questions.py")
assign_split = mod.assign_split


# One valid question; unanswerable kinds get no SQL and a reason.
def new_q(qid: str, kind: str, dataset: str | None = None) -> dict:
    dataset = dataset or ("tpch" if qid.startswith("t") else "reviews")
    answerable = kind != "unanswerable"
    sql = f"SELECT {int(qid[1:])} AS n" if answerable else None
    return {"qid": qid, "dataset": dataset, "kind": kind, "question": f"Question {qid}?",
            "gold_sqlite": sql, "gold_subset": sql, "ordered": False, "share_cols": [], "order_cols": [],
            "unanswerable_reason": None if answerable else "the data doesn't have it"}


# 100 questions per dataset in the KINDS mix, qids 001-100.
def synthetic_200() -> list[dict]:
    qs = []
    for prefix in ("t", "r"):
        kinds = [kind for kind, n in mod.KINDS.items() for _ in range(n)]
        qs += [new_q(f"{prefix}{i:03d}", kind) for i, kind in enumerate(kinds, 1)]
    return qs


def test_constants():
    assert mod.KINDS == {"single": 25, "join": 25, "time": 15, "topn": 15, "ratio": 10, "unanswerable": 10}
    assert mod.DEV_PER_KIND == {"single": 8, "join": 8, "time": 4, "topn": 4, "ratio": 3, "unanswerable": 3}


def test_split_counts_and_stability():
    qs = assign_split(synthetic_200())
    dev = [q for q in qs if q["split"] == "dev"]
    assert len(dev) == 60 and sum(q["dataset"] == "tpch" for q in dev) == 30
    assert assign_split(qs) == qs                                    # stable
    extra = assign_split(qs + [new_q("t101", "single")])
    assert extra[-1]["split"] == "test"


def test_split_per_kind_and_order_free():
    qs = assign_split(synthetic_200())
    for ds in ("tpch", "reviews"):
        for kind, n in mod.DEV_PER_KIND.items():
            assert sum(q["dataset"] == ds and q["kind"] == kind and q["split"] == "dev" for q in qs) == n

    shuffled = synthetic_200()
    random.Random(7).shuffle(shuffled)
    again = {q["qid"]: q["split"] for q in assign_split(shuffled)}
    assert again == {q["qid"]: q["split"] for q in qs}             # input order doesn't matter


def test_split_keeps_existing_and_copies():
    base = synthetic_200()
    first = assign_split(base)
    assert all("split" not in q for q in base)                       # input left alone
    dev_qid = next(q["qid"] for q in first if q["split"] == "dev")
    preset = [dict(q, split="test") if q["qid"] == dev_qid else q for q in base]
    assert {q["qid"]: q["split"] for q in assign_split(preset)}[dev_qid] == "test"


def test_validate_accepts_full_set():
    qs = synthetic_200()
    assert mod.validate(qs) == []
    framed = assign_split(qs)
    framed[0]["answer"] = {"columns": ["n"], "rows": [[1]]}
    framed[1]["empty_ok"] = True
    assert mod.validate(framed) == []


def _problems_for(**changes) -> list[str]:
    q = new_q("t001", "single")
    q.update(changes)
    return mod.validate([q], initial=False)


def test_validate_rules():
    assert mod.validate([new_q("t001", "single")], initial=False) == []
    # unanswerable with gold, answerable without gold_subset, wrong prefix
    assert _problems_for(kind="unanswerable", unanswerable_reason="no data")
    assert _problems_for(gold_subset=None)
    assert mod.validate([new_q("r001", "single", dataset="tpch")], initial=False)
    # no gold on an answerable kind; a reason when answerable; a blank reason when not
    assert _problems_for(gold_sqlite=None, gold_subset=None)
    assert _problems_for(unanswerable_reason="why")
    unanswerable = new_q("t002", "unanswerable")
    assert mod.validate([unanswerable], initial=False) == []
    assert mod.validate([dict(unanswerable, unanswerable_reason="  ")], initial=False)
    assert mod.validate([dict(unanswerable, gold_subset="SELECT 1")], initial=False)


def test_validate_order_cols_and_types():
    assert _problems_for(ordered=True, order_cols=[]) != []          # ordered needs order_cols
    assert _problems_for(ordered=True, order_cols=[1, 0]) == []
    assert _problems_for(ordered=False, order_cols=[0]) != []        # not ordered: must be []
    assert _problems_for(ordered=True, order_cols=[-1]) != []
    assert _problems_for(ordered=True, order_cols=[True]) != []      # a bool isn't an index
    assert _problems_for(share_cols=["0"]) != []
    assert _problems_for(ordered="yes") != []
    assert _problems_for(question=" ") != []
    assert _problems_for(kind="other") != []
    assert _problems_for(dataset="tpcds") != []
    assert _problems_for(split="train") != []
    assert _problems_for(empty_ok="true") != []
    assert _problems_for(answer={"columns": ["n"]}) != []


def test_validate_unknown_and_missing_fields():
    q = new_q("t001", "single")
    q["order_col"] = q.pop("order_cols")                             # misspelled
    problems = mod.validate([q], initial=False)
    assert any("order_col" in p for p in problems) and any("order_cols" in p for p in problems)
    assert mod.validate(["not a dict"], initial=False)


def test_validate_unique_qids_and_counts():
    assert mod.validate([new_q("t001", "single"), new_q("t001", "join")], initial=False)
    qs = synthetic_200()
    assert mod.validate(qs[:-1]) != []                               # one short
    assert mod.validate(qs[:-1], initial=False) == []
    qs[0]["kind"] = "join"                                           # 24 single, 26 join
    assert mod.validate(qs) != []
    late = synthetic_200()
    late[0] = dict(late[0], qid="t101")                              # the first set is 001-100
    assert mod.validate(late) != [] and mod.validate(late, initial=False) == []


def test_read_write_roundtrip(tmp_path):
    f = tmp_path / f"q_{os.getpid()}.jsonl"
    qs = assign_split(synthetic_200())
    mod.write_questions(f, qs)
    assert mod.read_questions(f) == qs
    assert [p.name for p in tmp_path.iterdir()] == [f.name]          # no temp file left


def test_read_questions_is_strict(tmp_path):
    f = tmp_path / f"bad_{os.getpid()}.jsonl"
    f.write_text('{"qid": "t001"}\n{"qid": "t0')                     # a cut last line is an error here
    with pytest.raises(ValueError, match=":2"):
        mod.read_questions(f)
    f.write_text('{"qid": "t001"}\n\n[1]\n')
    with pytest.raises(ValueError, match=":3"):
        mod.read_questions(f)
