"""Tests for bench/analyst/analyze_analyst_check.py."""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ANALYST = HERE.parent


# Load a module by path under an an_ name, clear of other bench tests.
def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ANALYST / file_name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mod = _load("an_analyze_analyst_check", "analyze_analyst_check.py")
an, ck, fz = mod.an, mod.ck, mod.fz

MODEL = "qwen36-35b"
STUB_SPEC = {"grains": {}, "entity_tables": {}, "primary_keys": {}, "glossary": []}
MANIFEST = {"dataset": "tpch", "tables": [{"name": "orders", "file": "orders.tbl", "rows": 4, "columns": [
    {"name": "o_orderkey", "type": "BIGINT"}, {"name": "o_totalprice", "type": "DECIMAL"}]}]}
GOLD = {"columns": ["total"], "rows": [[10.5]]}
SUM_SQL = "SELECT SUM(o_totalprice) AS total FROM orders"
TRAP_SQL = "SELECT SUM(o_totalprice) AS total FROM orders WHERE 1 = 0"
SKIPPED = {"raw": None, "text": None, "seconds": None, "prompt_sha1": None, "attempts": [], "final": "skipped", "sql": None, "result": None}


# --- synthetic outcomes -------------------------------------------------------------

# One outcome dict as outcome() returns it.
def oc(qid: str, right: bool, label: str = "clean", shown: bool = True, crash: bool = False, seconds: float = 5.0,
       cause: str | None = None, checks: dict | None = None, messages: dict | None = None) -> dict:
    if shown and not right and cause is None:
        cause = "model"
    return {"qid": qid, "shown": shown, "right": right, "label": label if shown else "not_shown",
            "cause": cause if shown and not right else None, "checks": checks or {}, "messages": messages or {},
            "crash": crash, "seconds": seconds}


# Wrong ones first (caught ones warned), then right ones (refused, then other flagged ones warned).
def make(wrong: int = 0, caught: int = 0, right: int = 0, refused: int = 0, flagged: int = 0, crashed: int = 0) -> list[dict]:
    outs = []
    for i in range(wrong):
        outs.append(oc(f"t{len(outs) + 1:03d}", False, "warned" if i < caught else "clean"))
    for i in range(right):
        label = "refused" if i < refused else "warned" if i < flagged else "clean"
        outs.append(oc(f"t{len(outs) + 1:03d}", True, label))
    for o in outs[:crashed]:
        o["crash"] = True
    return outs


GOOD = dict(wrong=40, caught=25, right=100, refused=5, flagged=20)


# --- hand-made records -------------------------------------------------------------

# A bridge reply in the real shape.
def _result(rows) -> dict:
    rows = [list(r) for r in rows]
    return {"id": "1", "ok": True, "columns": [{"name": "total", "type": "DECIMAL"}], "rows": rows,
            "row_count": len(rows), "truncated": False, "profile": {}, "ms": 2.0}


# A failed bridge reply of one kind.
def _failed(kind: str) -> dict:
    return {"id": "1", "ok": False, "error": {"kind": kind, "message": "boom"}, "ms": 1.0}


# One run record as the pipeline writes it; crash adds a crashed try first.
def rec(qid: str, final: str = "shown", rows=((10.5,),), sql: str = SUM_SQL, sqlite_cause="same", crash: bool = False,
        model: str = MODEL, seconds: float = 5.0, levels: dict | None = None, split: str = "test") -> dict:
    shown = final == "shown"
    result = _result(rows) if shown else None
    attempts = [{"raw": "", "kind": "sql", "sql": sql, "reply": _failed("crash"), "seconds": 1.0, "prompt_sha1": "x"}] if crash else []
    if final == "shown":
        attempts.append({"raw": "", "kind": "sql", "sql": sql, "reply": result, "seconds": 1.0, "prompt_sha1": "x"})
    elif final == "cannot":
        attempts.append({"raw": "", "kind": "cannot", "sql": None, "reply": None, "seconds": 1.0, "prompt_sha1": "x"})
    else:
        attempts.append({"raw": "", "kind": "sql", "sql": sql, "reply": _failed("run"), "seconds": 1.0, "prompt_sha1": "x"})

    primary = {"attempts": attempts, "final": final, "sql": sql if shown else None, "result": result,
               "claims": {"grain": None, "columns": [{"name": "total", "meaning": "", "entity": None}]} if shown else None,
               "reason": "no such data" if final == "cannot" else None}
    rephrase = copy.deepcopy(SKIPPED)
    if shown:
        rephrase.update(raw="", text="Same question?", seconds=1.0, prompt_sha1="x", final="shown", sql=sql, result=copy.deepcopy(result),
                        attempts=[{"raw": "", "kind": "sql", "sql": sql, "reply": copy.deepcopy(result), "seconds": 1.0, "prompt_sha1": "x"}])

    record = {"qid": qid, "dataset": "tpch" if qid[0] == "t" else "reviews", "split": split, "model": model, "prompt_sha1": "x",
              "primary": primary, "rephrase": rephrase, "seconds_total": seconds, "bridge_restarts": 0}
    if shown:
        record["sqlite_cause"] = {"ok": True, "columns": ["total"], "rows": [list(r) for r in rows]} if sqlite_cause == "same" else sqlite_cause
    if levels:
        record["levels"] = levels  # read only by the fake checks below
    return record


# One question as the question file holds it.
def q(qid: str, kind: str = "single", split: str = "test", answer=GOLD) -> dict:
    unanswerable = kind == "unanswerable"
    return {"qid": qid, "dataset": "tpch" if qid[0] == "t" else "reviews", "kind": kind,
            "question": f"What is the total order value for case {qid}, counting every order ever placed?",
            "gold_sqlite": None if unanswerable else "SELECT 1", "gold_subset": None if unanswerable else "SELECT 1",
            "ordered": False, "share_cols": [], "order_cols": [],
            "unanswerable_reason": "no such data" if unanswerable else None,
            "split": split, "answer": None if unanswerable else answer}


# outcome() with the stub spec and the small manifest.
def out(record: dict, question: dict, skip=()) -> dict:
    return mod.outcome(record, question, STUB_SPEC, MANIFEST, skip)


# --- bars and verdict ------------------------------------------------------------------

def test_bars_and_go():
    outs = make(**GOOD)
    b = mod.bars(outs)
    assert (b["catch"]["count"], b["catch"]["total"]) == (25, 40)
    assert (b["refuse"]["count"], b["refuse"]["total"]) == (5, 100)
    assert (b["flag"]["count"], b["flag"]["total"]) == (20, 100)
    assert b["catch"]["rate"] == pytest.approx(0.625)
    assert b["catch"]["interval"] == pytest.approx((0.4703, 0.7578), abs=1e-3)
    assert all(b[k]["holds"] for k in ("catch", "refuse", "flag"))
    verdict, reasons = mod.verdict(outs, [], [])
    assert verdict == "GO" and len(reasons) == 3


def test_catch_fail_is_no_go():
    verdict, reasons = mod.verdict(make(**{**GOOD, "caught": 15}), [], [])
    assert verdict == "NO-GO"
    assert len(reasons) == 1 and "catch" in reasons[0] and "15 of 40" in reasons[0]


def test_flag_bar_boundary():
    assert mod.verdict(make(**{**GOOD, "flagged": 25}), [], [])[0] == "GO"
    verdict, reasons = mod.verdict(make(**{**GOOD, "flagged": 26}), [], [])
    assert verdict == "NO-GO"
    assert len(reasons) == 1 and "warnings" in reasons[0] and "26 of 100" in reasons[0]


def test_refuse_bar_boundary():
    assert mod.verdict(make(**{**GOOD, "refused": 10}), [], [])[0] == "GO"
    verdict, reasons = mod.verdict(make(**{**GOOD, "refused": 11}), [], [])
    assert verdict == "NO-GO" and "refusals" in reasons[0]


def test_catch_bar_boundary():
    assert mod.verdict(make(**{**GOOD, "caught": 20}), [], [])[0] == "GO"
    assert mod.verdict(make(**{**GOOD, "caught": 19}), [], [])[0] == "NO-GO"


def test_too_few_wrong_is_no_verdict():
    verdict, reasons = mod.verdict(make(wrong=19, caught=19, right=100), [], [])
    assert verdict == "NO VERDICT"
    assert len(reasons) == 1 and "19" in reasons[0] and "20" in reasons[0]


def test_crash_share_over_limit_is_no_verdict():
    verdict, reasons = mod.verdict(make(**GOOD, crashed=8), [], [])
    assert verdict == "NO VERDICT"
    assert len(reasons) == 1 and "8 of 140" in reasons[0]


def test_crash_share_at_limit_holds():
    assert mod.verdict(make(**GOOD, crashed=7), [], [])[0] == "GO"


def test_freeze_mismatch_is_no_verdict():
    verdict, reasons = mod.verdict(make(**GOOD), ["checks.py changed"], [])
    assert verdict == "NO VERDICT" and any("checks.py changed" in r for r in reasons)


def test_every_no_verdict_reason_listed():
    outs = make(wrong=5, right=100, crashed=10)
    verdict, reasons = mod.verdict(outs, ["checks.py changed"], ["t900: the answer key has no answer"], missing=["t901"])
    assert verdict == "NO VERDICT" and len(reasons) == 5


def test_key_problem_is_no_verdict():
    qs = [q("t001"), q("t002", answer=None), q("t003", kind="unanswerable"), q("t004", split="dev", answer=None)]
    problems = mod.check_key(qs, "test")
    assert len(problems) == 1 and "t002" in problems[0]
    verdict, reasons = mod.verdict(make(**GOOD), [], problems)
    assert verdict == "NO VERDICT" and any("t002" in r for r in reasons)
    assert mod.check_key(qs, "nosplit") == ["no nosplit questions: the key isn't frozen"]
    assert mod.check_key([q("t001", split="dev")], "test") == ["no test questions: the key isn't frozen"]


def test_missing_record_is_no_verdict():
    outs = make(wrong=40, caught=25, right=93, crashed=7)  # 7 crashed of 133 records, but of 140 test questions
    missing = [f"t{i}" for i in range(134, 141)]
    verdict, reasons = mod.verdict(outs, [], [], missing=missing)
    assert verdict == "NO VERDICT"
    assert len(reasons) == 1 and "unfinished run" in reasons[0] and "t134" in reasons[0]


def test_no_right_answers_is_no_go():
    outs = make(wrong=25, caught=25)
    b = mod.bars(outs)
    assert b["refuse"]["rate"] is None and b["refuse"]["interval"] is None and not b["refuse"]["holds"]
    assert b["flag"]["rate"] is None and b["flag"]["interval"] is None
    verdict, reasons = mod.verdict(outs, [], [])
    assert verdict == "NO-GO"
    assert len(reasons) == 2 and all("no right shown answers" in r for r in reasons)


def test_per_check_counts():
    outs = [
        oc("t001", False, "warned", cause="engine", checks={"fanout": "warn", "grain": "warn", "terms": "note"}),
        oc("t002", False, "refused", checks={"engine_traps": "refuse", "terms": "note"}),
        oc("t003", True, "warned", checks={"grain": "warn"}),
        oc("t004", False, shown=False),
    ]
    counts = mod.per_check(outs)
    assert list(counts) == list(ck.CHECKS)
    none, model = {"model": 0, "engine": 0, "unknown": 0}, {"model": 1, "engine": 0, "unknown": 0}
    assert counts["fanout"] == {"catches": 1, "causes": {**none, "engine": 1}, "sole": 0, "false_alarms": 0}
    assert counts["grain"] == {"catches": 1, "causes": {**none, "engine": 1}, "sole": 0, "false_alarms": 1}
    assert counts["engine_traps"] == {"catches": 1, "causes": model, "sole": 1, "false_alarms": 0}
    assert counts["terms"] == {"catches": 0, "causes": none, "sole": 0, "false_alarms": 0}  # a note never fires


def test_wilson_known_values():
    assert mod.wilson(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-4)
    assert mod.wilson(10, 10) == pytest.approx((0.7225, 1.0), abs=1e-4)
    assert mod.wilson(0, 0) is None


# --- outcome on hand-made records ----------------------------------------------------------

def test_outcome_right_clean():
    o = out(rec("t001"), q("t001"))
    assert set(o) == {"qid", "shown", "right", "label", "cause", "checks", "messages", "crash", "seconds"}
    assert o["shown"] and o["right"] and o["label"] == "clean" and o["cause"] is None and not o["crash"]
    assert list(o["checks"]) == list(ck.CHECKS) and o["messages"] == {} and o["seconds"] == 5.0


def test_cause_engine_vs_model():
    wrong = ((99.0,),)
    engine = out(rec("t001", rows=wrong, sqlite_cause={"ok": True, "columns": ["total"], "rows": [[10.5]]}), q("t001"))
    model = out(rec("t002", rows=wrong, sqlite_cause={"ok": True, "columns": ["total"], "rows": [[99.0]]}), q("t002"))
    failed = out(rec("t003", rows=wrong, sqlite_cause={"ok": False, "kind": "sql", "error": "no such function"}), q("t003"))
    absent = rec("t004", rows=wrong)
    del absent["sqlite_cause"]
    assert [o["right"] for o in (engine, model, failed)] == [False, False, False]
    assert [engine["cause"], model["cause"], failed["cause"], out(absent, q("t004"))["cause"]] == ["engine", "model", "unknown", "unknown"]


def test_unanswerable_right_only_when_cannot():
    question = q("t001", kind="unanswerable")
    cannot = out(rec("t001", final="cannot"), question)
    assert cannot["right"] and not cannot["shown"] and cannot["label"] == "not_shown"
    assert cannot["checks"] == {} and cannot["messages"] == {} and cannot["cause"] is None
    shown = out(rec("t001"), question)
    assert shown["shown"] and not shown["right"] and shown["cause"] == "model"
    failed = out(rec("t001", final="failed"), question)
    assert not failed["right"] and failed["cause"] is None and failed["label"] == "not_shown"
    other = out(rec("t002", final="cannot"), q("t002"))  # can't answer on an answerable question
    assert not other["right"] and other["cause"] is None


def test_truncated_result_scored_on_shown_rows():
    record = rec("t001")
    record["primary"]["result"]["truncated"] = True
    assert out(record, q("t001"))["right"]
    longer = q("t002", answer={"columns": ["total"], "rows": [[10.5], [11.0]]})  # the cut dropped a key row
    record = rec("t002")
    record["primary"]["result"]["truncated"] = True
    assert not out(record, longer)["right"]


def test_crash_from_any_attempt():
    assert out(rec("t001", crash=True), q("t001"))["crash"]
    assert not out(rec("t001", final="failed"), q("t001"))["crash"]  # a run error isn't a crash
    record = rec("t001")
    record["rephrase"]["attempts"].insert(0, {"raw": "", "kind": "sql", "sql": SUM_SQL, "reply": _failed("crash"), "seconds": 1.0, "prompt_sha1": "x"})
    assert out(record, q("t001"))["crash"]


def test_catch_without_engine_traps():
    record, question = rec("t001", sql=TRAP_SQL, rows=((99.0,),)), q("t001")
    o = out(record, question)
    assert o["label"] == "refused" and o["checks"]["engine_traps"] == "refuse"
    assert list(o["messages"]) == ["engine_traps"] and "names no column" in o["messages"]["engine_traps"]
    assert out(record, question, skip={"engine_traps"})["label"] == "clean"
    assert mod.bars([o])["catch"]["count"] == 1
    relabeled = mod.relabel([o], {"engine_traps"})
    assert relabeled[0]["label"] == "clean" and o["label"] == "refused"  # the input isn't changed
    assert mod.bars(relabeled)["catch"]["count"] == 0


# --- freeze ---------------------------------------------------------------------------------

# A results folder and a code folder, frozen for one model; questions written first.
def _folders(tmp_path: Path, questions: list[dict], frozen: str | None = MODEL) -> tuple[Path, Path]:
    res, code = tmp_path / f"res_{os.getpid()}", tmp_path / f"code_{os.getpid()}"
    for name in fz.FROZEN_CODE:
        path = code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"original {name}\n", encoding="utf-8")
    shutil.copy(ANALYST / "model_mlx.py", code / "model_mlx.py")
    for name in fz.FROZEN_RESULTS:
        path = res / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if name.endswith(".json") else f"original {name}\n", encoding="utf-8")
    (res / "questions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in questions), encoding="utf-8")
    if frozen is not None:
        fz.write(res, code, frozen, fz.read_model_configs(code / "model_mlx.py"))
    return res, code


# Write one run file of records; returns its path.
def _write_run(res: Path, records: list[dict], split: str = "test", model: str = MODEL) -> Path:
    path = an.paths(res).run_file(split, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_model_mismatch_is_freeze_problem(tmp_path):
    res, code = _folders(tmp_path, [q("t001")], frozen="gemma4-26b")
    assert mod.check_freeze(res, code, "gemma4-26b", [rec("t001", model="gemma4-26b")]) == []
    problems = mod.check_freeze(res, code, "gemma4-26b", [rec("t001", model="qwen36-35b")])
    assert problems == ["run model qwen36-35b isn't the frozen model gemma4-26b"]
    assert mod.check_freeze(res, code, MODEL, []) == [f"run model {MODEL} isn't the frozen model gemma4-26b"]


def test_check_freeze_reports_changed_files(tmp_path):
    res, code = _folders(tmp_path, [q("t001")])
    (code / "checks.py").write_text("changed\n", encoding="utf-8")
    assert mod.check_freeze(res, code, MODEL, []) == ["checks.py changed"]
    (res / "freeze.json").unlink()
    assert mod.check_freeze(res, code, MODEL, []) == ["missing freeze.json"]


# --- the test CLI ---------------------------------------------------------------------------

# Checks that read their levels from the record, so a CLI test picks each label.
def _fake_checks(record, question, spec, manifest):
    levels = record.get("levels") or {}
    return [ck.CheckResult(n, levels.get(n, "pass"), f"{n} fired on {record['qid']}", {}) for n in ck.CHECKS]


# Swap the real checks for the fake ones for one test.
@pytest.fixture
def fake_checks(monkeypatch):
    monkeypatch.setattr(mod.ck, "run_checks", _fake_checks)


# The analyzer's main on the test split of one results folder.
def _main(res: Path, code: Path, *extra: str) -> int:
    return mod.main(["--split", "test", "--results", str(res), *extra], code_dir=code)


def test_report_opens_with_verdict(tmp_path, fake_checks, capsys):
    qs = [q("t001"), q("t002"), q("r001")]
    res, code = _folders(tmp_path, qs, frozen=None)
    _write_run(res, [rec("t001"), rec("t002", rows=((1.0,),))])
    assert _main(res, code, "--model", MODEL) == 1
    report = an.paths(res).report.read_text(encoding="utf-8")
    lines = report.splitlines()
    assert lines[0] == "**NO VERDICT**"
    for reason in ("missing freeze.json", "only 1 wrong shown answer", "unfinished run", "r001"):
        assert reason in report.split("## ")[0]
    assert "NO VERDICT" in capsys.readouterr().out

    with an.paths(res).cells.open(newline="", encoding="utf-8") as f:
        rows = {r["qid"]: r for r in csv.DictReader(f)}
    assert list(rows) == ["r001", "t001", "t002"]
    empty = {k: "" for k in ("shown", "right", "cause", "crash", "seconds", *ck.CHECKS)}
    assert rows["r001"] == {"qid": "r001", "dataset": "reviews", "kind": "single", "label": "no_record", **empty}
    assert rows["t001"]["label"] == "clean" and rows["t001"]["right"] == "True"


def test_report_gives_accuracy(tmp_path, fake_checks):
    qs = [q("t001"), q("t002"), q("t003", kind="unanswerable"), q("t004", kind="unanswerable"), q("r001")]
    res, code = _folders(tmp_path, qs)
    records = [rec("t001"), rec("t002", rows=((1.0,),)), rec("t003", final="cannot"), rec("t004")]
    _write_run(res, records)
    assert _main(res, code) == 1  # too few wrong answers, and r001 has no record
    report = an.paths(res).report.read_text(encoding="utf-8")
    head = report.split("## ")[0]
    assert report.startswith("**NO VERDICT**")
    assert "Accuracy without checks: 2 of 5 test questions right (0.400)" in head


ENGINE_CAUSE = {"ok": True, "columns": ["total"], "rows": [[10.5]]}  # SQLite gets the key's answer
UNKNOWN_CAUSE = {"ok": False, "kind": "sql", "error": "no such function"}


# 40 wrong (25 caught), 100 right (5 refused, 20 flagged), on tpch and reviews.
# Wrong 10-12 are caught only by engine_traps; 20-24 have cause unknown, 30-34 engine.
def _good_run() -> tuple[list[dict], list[dict]]:
    qs, records = [], []
    for i in range(140):
        qid = f"{'t' if i % 2 else 'r'}{i + 1:03d}"
        qs.append(q(qid, kind="join" if i % 3 else "single"))
        if i < 40:
            levels = None if i >= 25 else {"engine_traps": "refuse"} if 10 <= i <= 12 else {"fanout": "warn"}
            cause = UNKNOWN_CAUSE if 20 <= i <= 24 else ENGINE_CAUSE if 30 <= i <= 34 else "same"
            records.append(rec(qid, rows=((1.0,),), levels=levels, seconds=float(i), sqlite_cause=cause))
        else:
            j = i - 40
            levels = {"engine_traps": "refuse"} if j < 5 else {"grain": "warn"} if j < 20 else None
            records.append(rec(qid, levels=levels, seconds=float(i)))
    return qs, records


def test_test_split_go(tmp_path, fake_checks, capsys):
    qs, records = _good_run()
    qs += [q("t141", kind="unanswerable"), q("r142"), q("t143")]  # three not shown, one each way
    records += [rec("t141", final="cannot"), rec("r142", final="cannot"), rec("t143", final="failed")]
    res, code = _folders(tmp_path, qs + [q("t900", split="dev")])
    _write_run(res, records + [rec("t900", split="dev"), rec("t999")])
    assert _main(res, code) == 0
    report = an.paths(res).report.read_text(encoding="utf-8")
    assert report.startswith("**GO**")
    for text in ("25 of 40", "5 of 100", "20 of 100", "Ignored 2 records for qids outside the test split",
                 "fanout fired on", "## Per check", "## Per dataset", "## Per kind", "median 68.0 s", "p90 125.0 s"):
        assert text in report
    assert "GO" in capsys.readouterr().out

    lines = report.splitlines()
    expected = [
        # bar 1 by cause, and without engine_traps
        "| cause model | 20 of 30 | 0.667 |", "| cause engine | 0 of 5 | 0.000 |", "| cause unknown | 5 of 5 | 1.000 |",
        "| all, without engine_traps | 22 of 40 | 0.550 |",
        # the without / with table's wrong rows, split by cause
        "| Wrong, shown clean (silent) | 40 (model 30, engine 5, unknown 5) | 15 (model 10, engine 5, unknown 0) |",
        "| Wrong, warned or refused (caught) | 0 (model 0, engine 0, unknown 0) | 25 (model 20, engine 0, unknown 5) |",
        # not shown
        "| Correct \"can't answer\" (unanswerable question) | 1 |", "| Other \"can't answer\" | 1 |",
        "| Failed (every try errored, crashed or timed out) | 1 |", "| No record | 0 |",
        # each check's catches, split by cause
        "| fanout | 22 | model 17, engine 0, unknown 5 | 22 | 0 |",
        "| engine_traps | 3 | model 3, engine 0, unknown 0 | 3 | 5 |",
        "| grain | 0 | model 0, engine 0, unknown 0 | 0 | 15 |",
    ]
    for text in expected:
        assert any(line.startswith(text) for line in lines), text
    # per-dataset and per-kind wrong counts, split by cause
    assert next(line for line in lines if line.startswith("| reviews |")).endswith("| model 10 of 14, engine 0 of 3, unknown 3 of 3 |")
    assert next(line for line in lines if line.startswith("| tpch |")).endswith("| model 10 of 16, engine 0 of 2, unknown 2 of 2 |")
    assert next(line for line in lines if line.startswith("| single |")).endswith("| model 7 of 10, engine 0 of 2, unknown 2 of 2 |")

    with an.paths(res).cells.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["qid", "dataset", "kind", "shown", "right", "label", "cause", "crash", "seconds", *ck.CHECKS]
    assert len(rows) == 143 and {r["qid"] for r in rows} == {x["qid"] for x in qs}
    first = next(r for r in rows if r["qid"] == "r001")
    assert (first["label"], first["fanout"], first["right"], first["cause"]) == ("warned", "warn", "False", "model")


def test_key_problem_reaches_the_cli_verdict(tmp_path, fake_checks):
    qs, records = _good_run()
    qs[50]["answer"] = None  # an answerable test question the key never settled
    res, code = _folders(tmp_path, qs)
    _write_run(res, records)
    assert _main(res, code) == 1
    head = an.paths(res).report.read_text(encoding="utf-8").split("## ")[0]
    assert head.startswith("**NO VERDICT**") and f"answer key: {qs[50]['qid']} has no frozen answer" in head


def test_test_split_no_go_exits_0(tmp_path, fake_checks):
    qs, records = _good_run()
    for r in records[:40]:
        r.pop("levels", None)  # nothing caught
    res, code = _folders(tmp_path, qs)
    _write_run(res, records)
    assert _main(res, code) == 0
    assert an.paths(res).report.read_text(encoding="utf-8").startswith("**NO-GO**")


def test_model_flag_other_than_frozen_is_no_verdict(tmp_path, fake_checks):
    qs, records = _good_run()
    res, code = _folders(tmp_path, qs, frozen="gemma4-26b")
    _write_run(res, records)
    assert _main(res, code, "--model", MODEL) == 1
    head = an.paths(res).report.read_text(encoding="utf-8").split("## ")[0]
    assert head.startswith("**NO VERDICT**") and f"run model {MODEL} isn't the frozen model gemma4-26b" in head


def test_no_frozen_model_needs_model_flag(tmp_path, capsys):
    res, code = _folders(tmp_path, [q("t001")], frozen=None)
    assert _main(res, code) == 1
    assert "no frozen model; pass --model" in capsys.readouterr().err
    assert not an.paths(res).report.exists()


def test_missing_run_file_exits_1(tmp_path, capsys):
    res, code = _folders(tmp_path, [q("t001")])
    assert _main(res, code) == 1
    assert "no run file" in capsys.readouterr().err
    assert not an.paths(res).report.exists()


def test_refuses_results_under_ai_queries(tmp_path, monkeypatch, capsys):
    res, code = _folders(tmp_path, [q("t001")])
    monkeypatch.setattr(an, "AI_ROOT", tmp_path)
    assert _main(res, code) == 1
    assert "won't write inside" in capsys.readouterr().err


# --- dev: model choice and grow ---------------------------------------------------------------

# Synthetic dev outcomes: right ones, then wrong shown ones, each taking seconds.
def dev_outs(right: int, wrong: int, seconds: float, records: int = 60) -> list[dict]:
    outs = [oc(f"t{i:03d}", True, seconds=seconds) for i in range(right)]
    outs += [oc(f"t{right + i:03d}", False, seconds=seconds) for i in range(wrong)]
    outs += [oc(f"t{len(outs) + i:03d}", False, shown=False, seconds=seconds) for i in range(records - len(outs))]
    return outs


def test_dev_pick_and_grow():
    summaries = {
        "a": mod.dev_summary(dev_outs(48, 6, 25.0), 60),
        "b": mod.dev_summary(dev_outs(45, 6, 12.0), 60),
        "c": mod.dev_summary(dev_outs(42, 6, 9.0), 60),
    }
    assert summaries["a"]["accuracy"] == pytest.approx(0.80) and summaries["b"]["accuracy"] == pytest.approx(0.75)
    pick, note = mod.pick(summaries)
    assert pick == "b" and "20.0 s" in note
    g = mod.grow(summaries["b"]["wrong"], 60, 140)
    assert g["rate"] == pytest.approx(0.10) and g["projected"] == pytest.approx(14.0) and g["needed"] == 110


def test_dev_pick_none_under_limit():
    summaries = {"a": mod.dev_summary(dev_outs(50, 5, 30.0), 60), "b": mod.dev_summary(dev_outs(40, 5, 21.0), 60)}
    pick, note = mod.pick(summaries)
    assert pick == "b" and "fastest" in note


def test_dev_pick_tie_goes_to_faster():
    summaries = {"a": mod.dev_summary(dev_outs(45, 5, 15.0), 60), "b": mod.dev_summary(dev_outs(45, 5, 11.0), 60)}
    assert mod.pick(summaries)[0] == "b"
    summaries = {"z": mod.dev_summary(dev_outs(45, 5, 11.0), 60), "y": mod.dev_summary(dev_outs(45, 5, 11.0), 60)}
    assert mod.pick(summaries)[0] == "y"
    assert mod.pick({"a": mod.dev_summary([], 60)}) == (None, "no config has a record, so there's no pick")


def test_grow_line_at_the_cap():
    line = mod._grow_line(mod.grow(1, 60, 140), 60, 140)
    assert "add 160 test questions (to 300, cap 300)" in line
    assert "even at 300 the projection is 5.0, still under 25" in line
    assert "even at 300 the projection is 0.0, still under 25" in mod._grow_line(mod.grow(0, 60, 140), 60, 140)
    assert "even at" not in mod._grow_line(mod.grow(5, 60, 140), 60, 140)   # 25.0 at 300 is enough
    assert "even at" not in mod._grow_line(mod.grow(10, 60, 140), 60, 140)  # to 150, under the cap


def test_grow_rate_zero_and_enough():
    assert mod.grow(0, 60, 140)["needed"] == 160
    assert mod.grow(11, 60, 140)["needed"] == 0   # 11 x 140 = 1540, at least 25 x 60
    assert mod.grow(10, 60, 140)["needed"] == 10  # projected 23.3; ceil(1500 / 10) = 150
    assert mod.grow(1, 60, 140)["needed"] == 160  # capped at 300
    assert mod.grow(1, 60, 320)["needed"] == 0    # already past the cap


# 60 dev questions (30 per dataset) and 140 test ones.
def _dev_questions() -> list[dict]:
    qs = [q(f"{p}{i:03d}", split="dev") for p in "tr" for i in range(1, 31)]
    return qs + [q(f"{p}{i:03d}") for p in "tr" for i in range(31, 101)]


def test_dev_missing_records_count_as_not_right(tmp_path, fake_checks, capsys):
    res, code = _folders(tmp_path, _dev_questions(), frozen=None)
    _write_run(res, [rec("t001", split="dev"), rec("t002", split="dev")], split="dev", model="gemma4-26b")
    full = [rec(f"{p}{i:03d}", split="dev", seconds=30.0, levels={"grain": "warn"}) for p in "tr" for i in range(1, 31)]
    _write_run(res, full, split="dev", model="qwen36-35b")
    assert mod.main(["--split", "dev", "--results", str(res)], code_dir=code) == 0
    lines = capsys.readouterr().out.splitlines()
    assert any("gemma4-26b" in line and "answered 2 of 60" in line and "0.033" in line for line in lines)
    assert any("qwen36-35b" in line and "answered 60 of 60" in line for line in lines)
    pick_at = next(i for i, line in enumerate(lines) if line.startswith("pick: "))
    assert lines[pick_at].startswith("pick: gemma4-26b")
    assert "incomplete" in lines[pick_at - 1] and "gemma4-26b" in lines[pick_at - 1] and "the pick" in lines[pick_at - 1]
    report = an.paths(res).dev_report.read_text(encoding="utf-8")
    assert "answered 2 of 60" in report and "## Per check" in report
    assert not an.paths(res).report.exists()


def test_dev_report_grow_and_checks(tmp_path, fake_checks, capsys):
    res, code = _folders(tmp_path, _dev_questions(), frozen=None)
    records = [rec(f"{p}{i:03d}", split="dev", seconds=10.0) for p in "tr" for i in range(1, 31)]
    for r in records[:6]:  # 6 wrong of 60, 4 of them warned by fanout; 2 right ones warned by grain
        r["primary"]["result"]["rows"] = [[1.0]]
    for r in records[:4]:
        r["levels"] = {"fanout": "warn"}
    for r in records[6:8]:
        r["levels"] = {"grain": "warn"}
    _write_run(res, records, split="dev", model="gemma4-26b")
    assert mod.main(["--split", "dev", "--results", str(res)], code_dir=code) == 0
    out_text = capsys.readouterr().out
    assert "incomplete" not in out_text and "pick: gemma4-26b" in out_text
    report = an.paths(res).dev_report.read_text(encoding="utf-8")
    assert "add 110 test questions" in report
    assert "| fanout | 4 |" in report and "| grain | 0 |" in report


def test_dev_without_runs_exits_1(tmp_path, capsys):
    res, code = _folders(tmp_path, _dev_questions(), frozen=None)
    assert mod.main(["--split", "dev", "--results", str(res)], code_dir=code) == 1
    assert "no dev runs" in capsys.readouterr().err
    assert not an.paths(res).dev_report.exists()
