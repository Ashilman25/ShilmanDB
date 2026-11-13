"""Tests for bench/analyst/run_pipeline.py."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sqlite3
import sys
from contextlib import closing
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


mod = _load("an_run_pipeline", "run_pipeline.py")
an, pr, fz = mod.an, mod.pr, mod.fz

MODEL = "qwen36-35b"
SECONDS = 1.5  # model seconds per generation
MS = 250.0     # bridge ms per query
QIDS = ["t001", "t002", "t003"]
GOOD_SQL = "SELECT COUNT(*) AS open_orders FROM orders WHERE o_status = 'O'"
BAD_SQL = "SELEC COUNT(*) AS open_orders FROM orders"
COLUMN = {"name": "open_orders", "meaning": "orders still open", "entity": "orders"}
DESCRIPTION = {"tpch": "# tpch\n\n## Tables\n\n- `orders`: one row per order\n", "reviews": "# reviews\n\n## Tables\n\n- `reviews`: one row per review\n"}
RULES = "# SQL you can use\n\n- COUNT(*) works\n"
CTX = mod.Context(description=DESCRIPTION["tpch"], rules=RULES, examples=[])
TABLES = {
    "tpch": [
        "CREATE TABLE orders (o_orderkey INTEGER, o_status TEXT, o_orderdate TEXT)",
        "INSERT INTO orders VALUES (1, 'O', '1995-03-01'), (2, 'F', '1994-07-15'), (3, 'O', '1996-01-02'), (4, 'O', '1995-11-30')",
    ],
    "reviews": [
        "CREATE TABLE orders (o_orderkey INTEGER, o_status TEXT, o_orderdate TEXT)",
        "INSERT INTO orders VALUES (1, 'O', '2022-01-01')",
    ],
}

TOP_KEYS = {"qid", "dataset", "split", "model", "prompt_sha1", "primary", "rephrase", "sqlite_cause", "seconds_total", "bridge_restarts"}
PRIMARY_KEYS = {"attempts", "final", "sql", "result", "claims", "reason"}
REPHRASE_KEYS = {"raw", "text", "seconds", "prompt_sha1", "attempts", "final", "sql", "result"}
ATTEMPT_KEYS = {"raw", "kind", "sql", "reply", "seconds", "prompt_sha1"}
SKIPPED = {"raw": None, "text": None, "seconds": None, "prompt_sha1": None, "attempts": [], "final": "skipped", "sql": None, "result": None}


# One question record as the pipeline reads it.
def question(qid: str, split: str = "dev") -> dict:
    return {"qid": qid, "dataset": "tpch" if qid[0] == "t" else "reviews", "kind": "single",
            "question": f"What is the answer to puzzle {qid}?", "split": split}


# The model's JSON reply for one SQL query.
def reply(sql: str, grain: str = "one row") -> str:
    return json.dumps({"sql": sql, "grain": grain, "columns": [COLUMN]})


# The exact prompt that asks the model to rewrite a question.
def rewrite_prompt(qid: str) -> str:
    return pr.build_rephrase_prompt(question(qid)["question"])


# A script where each question gets a rewrite and then the same SQL reply.
def script_for(qids: list[str], sql: str = GOOD_SQL) -> dict:
    script = {rewrite_prompt(q): f"Tell me the puzzle {q} answer." for q in qids}
    script.update({f"puzzle {q}": reply(sql) for q in qids})
    return script


class FakeModel:
    """Stands in for MlxModel: the first key found in a prompt picks the reply.

    A list of replies is used in turn and its last one repeats. Every prompt is kept.
    A reply that is an exception is raised instead.
    """

    def __init__(self, script: dict, config_id: str = MODEL):
        self.script = {k: list(v) if isinstance(v, list) else [v] for k, v in script.items()}
        self.config_id = config_id
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> tuple[str, float]:
        self.prompts.append(prompt)
        for key, replies in self.script.items():
            if key in prompt:
                found = replies.pop(0) if len(replies) > 1 else replies[0]
                if isinstance(found, BaseException):
                    raise found
                return found, SECONDS
        raise AssertionError(f"no scripted reply for a prompt ending {prompt[-120:]!r}")


class FakeBridge:
    """Stands in for Bridge: runs SQL on the dataset's SQLite file, or fails as scripted.

    failures maps SQL to error kinds used in turn; a timeout or crash counts a restart.
    replies maps SQL to a canned good reply, for SQL only the engine runs.
    """

    def __init__(self, tables_tsv: Path, failures: dict | None = None, replies: dict | None = None):
        self.tables_tsv = Path(tables_tsv)
        self.db = self.tables_tsv.parent / f"{self.tables_tsv.parent.name}.sqlite"
        self.failures = {sql: list(kinds) for sql, kinds in (failures or {}).items()}
        self.replies = dict(replies or {})
        self.restarts = 0
        self.sent: list[str] = []
        self.closed = False

    def __enter__(self) -> FakeBridge:
        return self

    def __exit__(self, *exc) -> None:
        self.closed = True

    def query(self, sql: str) -> dict:
        self.sent.append(sql)
        rid = str(len(self.sent))
        kinds = self.failures.get(sql)
        if kinds:
            kind = kinds.pop(0)
            self.restarts += kind in ("timeout", "crash")
            return {"id": rid, "ok": False, "error": {"kind": kind, "message": f"bad {kind}"}, "ms": MS}
        if sql in self.replies:
            return {"id": rid, **self.replies[sql], "ms": MS}

        with closing(sqlite3.connect(f"{self.db.resolve().as_uri()}?mode=ro", uri=True)) as conn:
            try:
                cur = conn.execute(an.to_sqlite(sql))
                rows = [list(r) for r in cur.fetchall()]
            except sqlite3.Error as e:
                return {"id": rid, "ok": False, "error": {"kind": "parse", "message": str(e)}, "ms": MS}
            cols = [{"name": d[0], "type": "INTEGER"} for d in cur.description]
        return {"id": rid, "ok": True, "columns": cols, "rows": rows, "row_count": len(rows),
                "truncated": False, "profile": {}, "ms": MS}


class Factory:
    """A bridge_factory that keeps every bridge it made."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.made: list[FakeBridge] = []

    def __call__(self, tables_tsv: Path) -> FakeBridge:
        self.made.append(FakeBridge(tables_tsv, **self.kwargs))
        return self.made[-1]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


# A results folder with both tiny datasets, the rules and the questions.
def make_results(tmp_path: Path, questions: list[dict] | None = None) -> Path:
    results = tmp_path / f"results_{os.getpid()}"
    for ds, statements in TABLES.items():
        d = results / "datasets" / ds
        d.mkdir(parents=True)
        with closing(sqlite3.connect(d / f"{ds}.sqlite")) as conn:
            for s in statements:
                conn.execute(s)
            conn.commit()
        (d / "tables.tsv").write_text("orders\torders.tbl\to_orderkey:INTEGER\n")
        (d / "description.md").write_text(DESCRIPTION[ds])
    (results / "subset_rules.md").write_text(RULES)
    _write_jsonl(results / "questions.jsonl", [question(q) for q in QIDS] if questions is None else questions)
    return results


# Answer t001 with run_question; returns the record, the model and the bridge.
def answer(tmp_path: Path, script: dict, **bridge_kwargs) -> tuple[dict, FakeModel, FakeBridge]:
    res = make_results(tmp_path)
    bridge = FakeBridge(res / "datasets/tpch/tables.tsv", **bridge_kwargs)
    model = FakeModel(script)
    rec = mod.run_question(question("t001"), model, bridge, res / "datasets/tpch/tpch.sqlite", CTX)
    json.dumps(rec)  # every record must be plain JSON
    return rec, model, bridge


# A model_factory that must never be called.
def no_model(config_id: str):
    raise AssertionError("the model should not load")


# The problem text a retry prompt reports.
def problem_in(prompt: str) -> str:
    return prompt.split("## Problem\n\n")[1].rsplit(f"\n\n{pr.RETRY}", 1)[0]


def test_shown_answer_record_shape(tmp_path):
    rec, model, bridge = answer(tmp_path, script_for(["t001"]))
    q = question("t001")
    assert set(rec) == TOP_KEYS
    assert (rec["qid"], rec["dataset"], rec["split"], rec["model"]) == ("t001", "tpch", "dev", MODEL)

    primary = rec["primary"]
    assert set(primary) == PRIMARY_KEYS
    assert primary["final"] == "shown" and primary["sql"] == GOOD_SQL and primary["reason"] is None
    assert primary["result"]["ok"] is True and primary["result"]["rows"] == [[3]]
    assert primary["claims"] == {"grain": "one row", "columns": [COLUMN]}
    assert len(primary["attempts"]) == 1
    attempt = primary["attempts"][0]
    assert set(attempt) == ATTEMPT_KEYS
    assert attempt == {"raw": reply(GOOD_SQL), "kind": "sql", "sql": GOOD_SQL, "reply": primary["result"],
                       "seconds": SECONDS, "prompt_sha1": an.sha1_text(model.prompts[0])}
    assert model.prompts[0] == pr.build_sql_prompt(CTX.description, CTX.rules, CTX.examples, q["question"])
    assert rec["prompt_sha1"] == an.sha1_text(model.prompts[0])

    rephrase = rec["rephrase"]
    assert set(rephrase) == REPHRASE_KEYS
    assert rephrase["text"] == "Tell me the puzzle t001 answer." and rephrase["raw"] == rephrase["text"]
    assert rephrase["final"] == "shown" and rephrase["sql"] == GOOD_SQL and rephrase["result"]["rows"] == [[3]]
    assert rephrase["seconds"] == SECONDS
    assert model.prompts[1] == rewrite_prompt("t001") and rephrase["prompt_sha1"] == an.sha1_text(model.prompts[1])
    assert model.prompts[2] == pr.build_sql_prompt(CTX.description, CTX.rules, CTX.examples, rephrase["text"])
    assert rephrase["attempts"][0]["prompt_sha1"] == an.sha1_text(model.prompts[2])
    assert len(model.prompts) == 3 and bridge.sent == [GOOD_SQL, GOOD_SQL]

    assert rec["sqlite_cause"] == {"ok": True, "columns": ["open_orders"], "rows": [[3]]}
    assert rec["bridge_restarts"] == 0
    assert rec["seconds_total"] == pytest.approx(3 * SECONDS + 2 * MS / 1000)


def test_error_then_fix_uses_retry_prompt(tmp_path):
    script = {rewrite_prompt("t001"): "Tell me the puzzle t001 answer.", "## Problem": reply(GOOD_SQL), "puzzle t001": reply(BAD_SQL)}
    rec, model, bridge = answer(tmp_path, script)
    attempts = rec["primary"]["attempts"]
    assert [a["kind"] for a in attempts] == ["sql", "sql"] and rec["primary"]["final"] == "shown"
    error = attempts[0]["reply"]["error"]
    assert error["kind"] == "parse" and "SELEC" in error["message"]
    assert model.prompts[1] == pr.build_retry_prompt(model.prompts[0], reply(BAD_SQL), f"parse error: {error['message']}")
    assert [a["prompt_sha1"] for a in attempts] == [an.sha1_text(p) for p in model.prompts[:2]]
    assert rec["prompt_sha1"] == an.sha1_text(model.prompts[0])  # the first prompt, not the retry
    assert rec["primary"]["sql"] == GOOD_SQL and rec["primary"]["result"] == attempts[1]["reply"]
    assert rec["rephrase"]["final"] == "shown" and len(rec["rephrase"]["attempts"]) == 2


def test_malformed_replies_end_as_visible_failure(tmp_path):
    rec, model, bridge = answer(tmp_path, {"": "I think the answer is 5"})
    attempts = rec["primary"]["attempts"]
    assert len(rec["primary"]["attempts"]) == 3 and rec["primary"]["final"] == "failed"
    assert rec["rephrase"]["final"] == "skipped" and "sqlite_cause" not in rec
    assert [(a["kind"], a["sql"], a["reply"]) for a in attempts] == [("malformed", None, None)] * 3
    assert {k: rec["primary"][k] for k in ("sql", "result", "claims", "reason")} == dict.fromkeys(("sql", "result", "claims", "reason"))
    assert rec["rephrase"] == SKIPPED
    assert [problem_in(p) for p in model.prompts[1:]] == [pr.NOT_A_REPLY] * 2
    assert len(model.prompts) == 3 and bridge.sent == []
    assert set(rec) == TOP_KEYS - {"sqlite_cause"}
    assert rec["seconds_total"] == pytest.approx(3 * SECONDS)


def test_cannot_answer_stops(tmp_path):
    rec, model, bridge = answer(tmp_path, {"puzzle t001": json.dumps({"cannot_answer": "no stock levels"})})
    primary = rec["primary"]
    assert primary["final"] == "cannot" and primary["reason"] == "no stock levels"
    assert len(primary["attempts"]) == 1 and primary["attempts"][0]["kind"] == "cannot"
    assert (primary["sql"], primary["result"], primary["claims"]) == (None, None, None)
    assert rec["rephrase"] == SKIPPED and "sqlite_cause" not in rec
    assert len(model.prompts) == 1 and bridge.sent == []


def test_cannot_after_a_bad_reply_stops(tmp_path):
    rec, model, _ = answer(tmp_path, {"puzzle t001": ["no idea", json.dumps({"cannot_answer": "no stock"})]})
    assert [a["kind"] for a in rec["primary"]["attempts"]] == ["malformed", "cannot"]
    assert rec["primary"]["final"] == "cannot" and len(model.prompts) == 2


def test_timeout_counts_as_attempt(tmp_path):
    rec, model, bridge = answer(tmp_path, script_for(["t001"]), failures={GOOD_SQL: ["timeout"]})
    attempts = rec["primary"]["attempts"]
    assert len(attempts) == 2 and rec["primary"]["final"] == "shown"
    assert attempts[0]["reply"]["error"]["kind"] == "timeout"
    assert problem_in(model.prompts[1]) == pr.TOO_SLOW
    assert rec["bridge_restarts"] == 1


@pytest.mark.parametrize("kind", ["parse", "plan", "run", "request", "timeout", "crash"])
def test_each_bridge_error_uses_an_attempt(tmp_path, kind):
    rec, model, bridge = answer(tmp_path, script_for(["t001"]), failures={GOOD_SQL: [kind] * 5})
    attempts = rec["primary"]["attempts"]
    assert len(attempts) == 3 and rec["primary"]["final"] == "failed"
    assert [a["reply"]["error"]["kind"] for a in attempts] == [kind] * 3
    want = {"timeout": pr.TOO_SLOW, "crash": pr.CRASHED}.get(kind, f"{kind} error: bad {kind}")
    assert [problem_in(p) for p in model.prompts[1:]] == [want] * 2
    assert len(bridge.sent) == 3 and rec["rephrase"] == SKIPPED and "sqlite_cause" not in rec
    assert rec["bridge_restarts"] == (3 if kind in ("timeout", "crash") else 0)


def test_malformed_and_errors_share_three_attempts(tmp_path):
    rec, model, bridge = answer(tmp_path, {"puzzle t001": ["no idea", reply(BAD_SQL), reply(BAD_SQL), reply(GOOD_SQL)]})
    assert [a["kind"] for a in rec["primary"]["attempts"]] == ["malformed", "sql", "sql"]
    assert rec["primary"]["final"] == "failed" and len(model.prompts) == 3


def test_retry_prompt_strips_thinking(tmp_path):
    raw = "<think>let me guess</think>I think 5"
    rec, model, _ = answer(tmp_path, {rewrite_prompt("t001"): "Again?", "puzzle t001": [raw, reply(GOOD_SQL)], "Again?": reply(GOOD_SQL)})
    assert model.prompts[1] == pr.build_retry_prompt(model.prompts[0], "I think 5", pr.NOT_A_REPLY)
    assert "let me guess" not in model.prompts[1]
    assert rec["primary"]["attempts"][0]["raw"] == raw  # the record keeps the reply whole


def test_empty_rewrite_skips_rephrase(tmp_path):
    rec, model, _ = answer(tmp_path, {rewrite_prompt("t001"): "<think>hmm</think>\n\n", "puzzle t001": reply(GOOD_SQL)})
    assert rec["rephrase"] == {**SKIPPED, "raw": "<think>hmm</think>\n\n", "seconds": SECONDS,
                               "prompt_sha1": an.sha1_text(rewrite_prompt("t001"))}
    assert rec["primary"]["final"] == "shown" and rec["sqlite_cause"]["ok"] is True
    assert len(model.prompts) == 2
    assert rec["seconds_total"] == pytest.approx(2 * SECONDS + MS / 1000)


def test_seconds_total_adds_model_and_bridge_time(tmp_path):
    script = {rewrite_prompt("t001"): "Tell me the puzzle t001 answer.", "## Problem": reply(GOOD_SQL), "puzzle t001": reply(BAD_SQL)}
    rec, model, bridge = answer(tmp_path, script)
    assert len(model.prompts) == 5 and len(bridge.sent) == 4  # two tries each, plus the rewrite
    assert rec["seconds_total"] == pytest.approx(5 * SECONDS + 4 * MS / 1000)


def test_sqlite_cause_is_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(an, "ROW_CAP", 2)
    sql = "SELECT o_orderkey FROM orders ORDER BY o_orderkey"
    rec, _, _ = answer(tmp_path, script_for(["t001"], sql))
    assert rec["sqlite_cause"] == {"ok": True, "columns": ["o_orderkey"], "rows": [[1], [2]], "truncated": True}
    assert len(rec["primary"]["result"]["rows"]) == 4  # the engine's reply is kept as it came


def test_sqlite_cause_uses_sqlite_dates(tmp_path):
    sql = "SELECT COUNT(*) AS n FROM orders WHERE o_orderdate >= DATE '1995-01-01'"
    rec, _, _ = answer(tmp_path, script_for(["t001"], sql))
    assert rec["sqlite_cause"] == {"ok": True, "columns": ["n"], "rows": [[3]]}


def test_sqlite_cause_keeps_sqlite_errors(tmp_path):
    sql = "SELECT engine_only() AS x"
    canned = {"ok": True, "columns": [{"name": "x", "type": "INTEGER"}], "rows": [[1]], "row_count": 1, "truncated": False, "profile": {}}
    rec, _, _ = answer(tmp_path, script_for(["t001"], sql), replies={sql: canned})
    assert rec["primary"]["final"] == "shown"
    assert rec["sqlite_cause"] == {"ok": False, "kind": "sql", "error": "no such function: engine_only"}


def test_resume_drops_partial_line_and_skips_done(tmp_path):
    res = make_results(tmp_path)
    model = FakeModel(script_for(QIDS))
    assert mod.run("dev", model, res, ["tpch"], limit_per_dataset=1, bridge_factory=Factory()) == 1
    f = an.paths(res).run_file("dev", MODEL)
    f.write_text(f.read_text() + '{"qid": "t0')       # interrupted
    assert mod.run("dev", model, res, ["tpch"], bridge_factory=Factory()) == 2 and [r["qid"] for r in an.read_jsonl(f)] == ["t001", "t002", "t003"]
    assert len(f.read_text().splitlines()) == 3


def test_limit_applies_before_skipping_done(tmp_path):
    res = make_results(tmp_path)
    model = FakeModel(script_for(QIDS))
    assert mod.run("dev", model, res, ["tpch"], limit_per_dataset=1, bridge_factory=Factory()) == 1
    factory = Factory()
    assert mod.run("dev", model, res, ["tpch"], limit_per_dataset=1, bridge_factory=factory) == 0
    assert factory.made == []  # nothing to do starts no bridge
    assert mod.run("dev", model, res, ["tpch"], limit_per_dataset=2, bridge_factory=Factory()) == 1
    assert [r["qid"] for r in an.read_jsonl(an.paths(res).run_file("dev", MODEL))] == ["t001", "t002"]


def test_run_order_split_and_output(tmp_path, capsys):
    qs = [question("t003"), question("t002", "test"), question("r002"), question("t001"), question("r001")]
    res = make_results(tmp_path, qs)
    model = FakeModel(script_for(["t001", "t003", "r001", "r002"]))
    factory = Factory()
    assert mod.run("dev", model, res, ["reviews", "tpch"], bridge_factory=factory) == 4

    records = an.read_jsonl(an.paths(res).run_file("dev", MODEL))
    assert [r["qid"] for r in records] == ["r001", "r002", "t001", "t003"]
    assert [(r["dataset"], r["split"], r["model"]) for r in records] == [("reviews", "dev", MODEL)] * 2 + [("tpch", "dev", MODEL)] * 2
    assert [b.tables_tsv for b in factory.made] == [res / "datasets/reviews/tables.tsv", res / "datasets/tpch/tables.tsv"]
    assert all(b.closed for b in factory.made)
    assert [len(b.sent) for b in factory.made] == [4, 4]
    assert all(set(r) == TOP_KEYS for r in records)

    out = capsys.readouterr().out.splitlines()
    assert [line for line in out if "bridge ready" in line] == ["reviews bridge ready in 0.0 s", "tpch bridge ready in 0.0 s"]
    assert [line for line in out if "bridge ready" not in line] == [f"{q} shown 5.0 s" for q in ("r001", "r002", "t001", "t003")]


def test_bridge_restarts_count_per_question(tmp_path, capsys):
    res = make_results(tmp_path, [question("t001"), question("t002")])
    factory = Factory(failures={GOOD_SQL: ["timeout"]})  # only the first query times out
    assert mod.run("dev", FakeModel(script_for(["t001", "t002"])), res, ["tpch"], bridge_factory=factory) == 2
    records = an.read_jsonl(an.paths(res).run_file("dev", MODEL))
    assert [r["bridge_restarts"] for r in records] == [1, 0] and factory.made[0].restarts == 1
    out = capsys.readouterr().out.splitlines()
    assert out[1:] == [f"t001 shown {4 * SECONDS + 3 * MS / 1000:.1f} s (+1 restarts)", "t002 shown 5.0 s"]


def test_each_record_is_saved_before_the_next_question(tmp_path):
    res = make_results(tmp_path)
    model = FakeModel({**script_for(["t001"]), "puzzle t002": RuntimeError("model died")})
    factory = Factory()
    with pytest.raises(RuntimeError, match="model died"):
        mod.run("dev", model, res, ["tpch"], bridge_factory=factory)
    f = an.paths(res).run_file("dev", MODEL)
    assert f.read_text().count("\n") == 1 and f.read_text().endswith("\n")  # t001 whole, nothing of t002
    assert [r["qid"] for r in an.read_jsonl(f)] == ["t001"]
    assert len(factory.made) == 1 and factory.made[0].closed


# A bridge command that writes a line to stderr, then runs the fake bridge.
def _noisy_bridge(tmp_path: Path) -> Path:
    script = tmp_path / f"noisy_bridge_{os.getpid()}.sh"
    script.write_text(f'#!/bin/sh\necho "noisy bridge says hello" >&2\nexec "{sys.executable}" "{HERE / "fake_bridge.py"}" "$@"\n')
    script.chmod(0o755)
    return script


def test_default_bridge_logs_stderr_next_to_the_run_file(tmp_path, monkeypatch):
    res = make_results(tmp_path)
    monkeypatch.setattr(an, "BRIDGE", _noisy_bridge(tmp_path))
    model = FakeModel(script_for(QIDS))
    assert mod.run("dev", model, res, ["tpch"], limit_per_dataset=1) == 1
    record = an.read_jsonl(an.paths(res).run_file("dev", MODEL))[0]
    assert record["primary"]["result"]["rows"] == [[GOOD_SQL]]  # the fake bridge echoes the SQL
    log = res / "runs" / f"dev_{MODEL}.bridge.log"
    assert "noisy bridge says hello" in log.read_text()


def test_rephrase_can_end_cannot(tmp_path):
    cannot = json.dumps({"cannot_answer": "no such data"})
    rec, model, _ = answer(tmp_path, {rewrite_prompt("t001"): "Other words?", "puzzle t001": reply(GOOD_SQL), "Other words?": cannot})
    rephrase = rec["rephrase"]
    assert rephrase["final"] == "cannot" and (rephrase["sql"], rephrase["result"]) == (None, None)
    assert [a["kind"] for a in rephrase["attempts"]] == ["cannot"] and rephrase["text"] == "Other words?"
    assert rec["primary"]["final"] == "shown" and rec["sqlite_cause"]["ok"] is True


def test_rephrase_can_run_out_of_attempts(tmp_path):
    rec, model, bridge = answer(tmp_path, {rewrite_prompt("t001"): "Other words?", "puzzle t001": reply(GOOD_SQL), "Other words?": "no idea"})
    rephrase = rec["rephrase"]
    assert rephrase["final"] == "failed" and (rephrase["sql"], rephrase["result"]) == (None, None)
    assert [a["kind"] for a in rephrase["attempts"]] == ["malformed"] * 3
    assert len(model.prompts) == 5 and bridge.sent == [GOOD_SQL]


def test_run_uses_each_datasets_prompt_context(tmp_path):
    res = make_results(tmp_path, [question("t001"), question("r001")])
    model = FakeModel(script_for(["t001", "r001"]))
    assert mod.run("dev", model, res, ["tpch", "reviews"], bridge_factory=Factory()) == 2
    examples = json.loads((ANALYST / "prompt_examples.json").read_text(encoding="utf-8"))
    tpch_prompt = pr.build_sql_prompt(DESCRIPTION["tpch"], RULES, examples["tpch"], question("t001")["question"])
    reviews_prompt = pr.build_sql_prompt(DESCRIPTION["reviews"], RULES, examples["reviews"], question("r001")["question"])
    assert model.prompts[0] == tpch_prompt and model.prompts[3] == reviews_prompt


def test_run_refuses_unsplit_questions(tmp_path):
    unsplit = [{k: v for k, v in question(q).items() if k != "split"} for q in QIDS]
    res = make_results(tmp_path, unsplit)
    with pytest.raises(ValueError, match="no dev questions for tpch"):
        mod.run("dev", FakeModel({}), res, ["tpch"], bridge_factory=Factory())
    assert not (res / "runs").exists()


def test_check_inputs_names_every_missing_file(tmp_path):
    res = make_results(tmp_path)
    mod.check_inputs(res, "dev", ["tpch"])
    (res / "datasets/tpch/description.md").unlink()
    (res / "datasets/tpch/tpch.sqlite").unlink()
    with pytest.raises(FileNotFoundError, match=r"description\.md.*tpch\.sqlite"):
        mod.check_inputs(res, "dev", ["tpch"])


def test_to_sqlite_dates():
    assert mod.to_sqlite("WHERE d >= date '1995-01-01'") == "WHERE d >= '1995-01-01'"


def test_test_split_refuses_without_freeze(tmp_path, capsys):
    res = make_results(tmp_path)
    assert mod.main(["--split", "test", "--model", "qwen36-35b", "--results", str(res)]) == 2
    assert capsys.readouterr().err == "test generation needs a clean freeze for --model qwen36-35b: missing freeze.json\n"
    assert not (res / "runs").exists()


# A code folder with every frozen file and the real model_mlx.py, frozen for one model.
def _frozen(tmp_path: Path, res: Path, model: str) -> Path:
    code = tmp_path / f"code_{os.getpid()}"
    for name in fz.FROZEN_CODE:
        path = code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"original {name}\n", encoding="utf-8")
    shutil.copy(ANALYST / "model_mlx.py", code / "model_mlx.py")
    (res / "probe.json").write_text("{}\n")
    for ds in ("tpch", "reviews"):
        (res / "datasets" / ds / "manifest.json").write_text("{}\n")
    fz.write(res, code, model, fz.read_model_configs(code / "model_mlx.py"))
    return code


def test_test_split_refuses_a_freeze_for_another_model(tmp_path, capsys):
    res = make_results(tmp_path, [question("t001", "test")])
    code = _frozen(tmp_path, res, "gemma4-26b")
    assert mod.main(["--split", "test", "--model", MODEL, "--results", str(res)], code_dir=code, model_factory=no_model) == 2
    assert capsys.readouterr().err == f"test generation needs a clean freeze for --model {MODEL}: freeze.json is for model gemma4-26b\n"
    assert not (res / "runs").exists()


def test_test_split_refuses_a_changed_file(tmp_path, capsys):
    res = make_results(tmp_path, [question("t001", "test")])
    code = _frozen(tmp_path, res, MODEL)
    (res / "subset_rules.md").write_text(RULES + "- one more rule\n")
    assert mod.main(["--split", "test", "--model", MODEL, "--results", str(res)], code_dir=code, model_factory=no_model) == 2
    assert capsys.readouterr().err == f"test generation needs a clean freeze for --model {MODEL}: subset_rules.md changed\n"


def test_test_split_runs_on_a_clean_freeze(tmp_path, capsys):
    res = make_results(tmp_path, [question("t001", "test"), question("t002", "test"), question("t003")])
    code = _frozen(tmp_path, res, "gemma4-26b")
    made = []

    def models(config_id: str) -> FakeModel:
        made.append(FakeModel(script_for(QIDS), config_id=config_id))
        return made[-1]

    argv = ["--split", "test", "--model", "gemma4-26b", "--results", str(res), "--datasets", "tpch"]
    assert mod.main(argv, code_dir=code, bridge_factory=Factory(), model_factory=models) == 0
    records = an.read_jsonl(an.paths(res).run_file("test", "gemma4-26b"))
    assert [(r["qid"], r["split"], r["model"]) for r in records] == [("t001", "test", "gemma4-26b"), ("t002", "test", "gemma4-26b")]
    assert [m.config_id for m in made] == ["gemma4-26b"]
    out = capsys.readouterr().out
    assert "model gemma4-26b loaded in 0.0 s\n" in out and "wrote 2 records" in out


def test_unknown_model_refused(tmp_path, capsys):
    res = make_results(tmp_path)
    assert mod.main(["--split", "dev", "--model", "llama", "--results", str(res)], model_factory=no_model) == 2
    err = capsys.readouterr().err
    assert "unknown model 'llama'" in err and MODEL in err


def test_main_dev_run(tmp_path, capsys):
    res = make_results(tmp_path)
    argv = ["--split", "dev", "--model", MODEL, "--results", str(res), "--datasets", "tpch", "--limit-per-dataset", "1"]
    assert mod.main(argv, bridge_factory=Factory(), model_factory=lambda cid: FakeModel(script_for(QIDS), cid)) == 0
    out = capsys.readouterr().out.splitlines()
    run_file = an.paths(res).run_file("dev", MODEL)
    assert out == [f"model {MODEL} loaded in 0.0 s", "tpch bridge ready in 0.0 s", "t001 shown 5.0 s", f"wrote 1 record -> {run_file}"]
    assert [r["qid"] for r in an.read_jsonl(run_file)] == ["t001"]


def test_main_checks_inputs_before_loading_the_model(tmp_path, capsys):
    res = make_results(tmp_path)
    (res / "datasets/tpch/description.md").unlink()
    argv = ["--split", "dev", "--model", MODEL, "--results", str(res), "--datasets", "tpch"]
    assert mod.main(argv, bridge_factory=Factory(), model_factory=no_model) == 1
    assert "description.md" in capsys.readouterr().err


def test_main_refuses_a_missing_bridge_binary(tmp_path, monkeypatch, capsys):
    res = make_results(tmp_path)
    monkeypatch.setattr(an, "BRIDGE", tmp_path / f"no_bridge_{os.getpid()}")
    argv = ["--split", "dev", "--model", MODEL, "--results", str(res), "--datasets", "tpch", "--limit-per-dataset", "1"]
    assert mod.main(argv, model_factory=no_model) == 1
    assert f"no_bridge_{os.getpid()}" in capsys.readouterr().err
    assert not (res / "runs").exists()
    # A bridge_factory of my own doesn't need the binary.
    assert mod.main(argv, bridge_factory=Factory(), model_factory=lambda cid: FakeModel(script_for(QIDS), cid)) == 0


@pytest.mark.parametrize("text, line", [
    ('{"qid": "t001"}\nnot json\n{"qid": "t002"}\n', 2),  # a bad middle line
    ('{"qid": "t001"}\nnot json\n', 2),                     # a whole bad line; the next append would bury it
])
def test_main_refuses_a_run_file_that_does_not_read(tmp_path, capsys, text, line):
    res = make_results(tmp_path)
    f = an.paths(res).run_file("dev", MODEL)
    f.parent.mkdir(parents=True)
    f.write_text(text)
    argv = ["--split", "dev", "--model", MODEL, "--results", str(res), "--datasets", "tpch"]
    assert mod.main(argv, bridge_factory=Factory(), model_factory=no_model) == 1
    assert f"{f}:{line}: " in capsys.readouterr().err
    assert f.read_text() == text


def test_main_resumes_after_a_cut_last_line(tmp_path):
    res = make_results(tmp_path)
    f = an.paths(res).run_file("dev", MODEL)
    f.parent.mkdir(parents=True)
    f.write_text('{"qid": "t0')
    argv = ["--split", "dev", "--model", MODEL, "--results", str(res), "--datasets", "tpch"]
    assert mod.main(argv, bridge_factory=Factory(), model_factory=lambda cid: FakeModel(script_for(QIDS), cid)) == 0
    assert [r["qid"] for r in an.read_jsonl(f)] == QIDS


def test_main_refuses_results_under_ai_queries(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(an, "AI_ROOT", tmp_path / "ai")
    res = make_results(tmp_path / "ai")
    argv = ["--split", "dev", "--model", MODEL, "--results", str(res), "--datasets", "tpch"]
    assert mod.main(argv, bridge_factory=Factory(), model_factory=no_model) == 1
    assert "won't write inside" in capsys.readouterr().err
    assert not (res / "runs").exists()


@pytest.mark.parametrize("extra", [["--datasets", "tpch,tpcds"], ["--limit-per-dataset", "0"], ["--split", "train"]])
def test_main_bad_arguments(tmp_path, extra):
    res = make_results(tmp_path)
    with pytest.raises(SystemExit) as e:
        mod.main(["--split", "dev", "--model", MODEL, "--results", str(res), *extra], model_factory=no_model)
    assert e.value.code == 2
