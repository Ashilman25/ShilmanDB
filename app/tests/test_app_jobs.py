"""Tests for app/server/jobs.py."""
from __future__ import annotations

import copy
import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from app.core import analyst_check
from app.core.datasets import LABELS, default_paths
from app.core.engine import Engine, EngineOptions
from app.core.measured import Lock, read_json
# check_views is reached as a module: pytest collects any test* name imported into a test module.
from app.server import check_views, copy_rules, jobs, views
from app.server.jobs import HISTORY_CAP, LONG_POLL_S, MODEL_WORKER, SNAPSHOT_KEYS, SQL_WORKER, JobRunner, Refusal
from app.tests.fakes import MS, FakeBridge, FakeModel, bridge_error, bridge_ok, reply
from app.tests.test_app_flows import (
    COLUMNS,
    FEWER,
    FEWER_SQL,
    GRAIN,
    LATE_COLUMNS,
    LATE_QUESTION,
    LATE_REPHRASED,
    LATE_SQL,
    LEFT_SQL,
    QUESTION,
    REGIONS,
    REPHRASED,
    REPLIES,
    SQL,
    WIDE_MANIFEST,
    script,
)

pr = analyst_check.pr

WAIT_S = 10.0
ENDED = ("done", "stopped", "error")
DASH = chr(0x2014)

OTHER = "Which regions have nations?"
BOOM = "What happens when the model breaks?"
TYPED = "SELECT n_name FROM nation WHERE n_regionkey = 1"
AMERICAS = ["ARGENTINA", "BRAZIL", "CANADA", "PERU", "UNITED STATES"]
TYPED_RESULT = bridge_ok([("n_name", "VARCHAR")], [[n] for n in AMERICAS])
BAD_SQL = "SELECT r_name FROM region WHERE"
PARSE_ERROR = bridge_error("parse", "Expected a condition after WHERE")
LATE_RESULT = bridge_ok([("late_lines", "INTEGER")], [[123]])
ALL_REPLIES = {**REPLIES, TYPED: TYPED_RESULT, BAD_SQL: PARSE_ERROR, LATE_SQL: LATE_RESULT}

HISTORY_KEYS = ["id", "kind", "dataset", "text", "verdict", "from_job"]
NO_TIMING = {"first_s": None, "verdict_s": None, "engine_ms": None, "first_text": None, "verdict_text": None}
STOPPED = {"kind": "stopped", "text": "Stopped before the second answer, so there's no verdict.", "sql": None}
RATES = read_json(default_paths().measured)["rates"]


# Fixture paths with an empty app folder, so the engine finds no measured.json and hashes nothing.
@pytest.fixture
def paths(fixture_paths, tmp_path):
    return replace(fixture_paths, app_dir=tmp_path / "app", src=tmp_path / "src")


# A runner that also keeps every snapshot it publishes, in order (a spy on its publishing method).
class Recording(JobRunner):
    def __init__(self, engine):
        super().__init__(engine)
        self.published: list[dict] = []
        self.fakes: dict[str, FakeBridge] = {}
        self._seen_lock = threading.Lock()

    def _publish(self, job) -> dict:
        snapshot = super()._publish(job)
        with self._seen_lock:
            self.published.append(snapshot)
        return snapshot

    # Every snapshot one job published, oldest first; all jobs when job_id is None.
    def seen(self, job_id: str | None = None) -> list[dict]:
        with self._seen_lock:
            return [s for s in self.published if job_id is None or s["id"] == job_id]


# A FakeModel that notes the thread of every call.
class ThreadModel(FakeModel):
    def __init__(self, script_: dict):
        super().__init__(script_)
        self.threads: set[str] = set()

    def generate(self, prompt: str) -> tuple[str, float]:
        self.threads.add(threading.current_thread().name)
        return super().generate(prompt)


# A FakeModel whose rephrase call waits for the test, so a job can be held between its two answers.
class RephraseGate(FakeModel):
    def __init__(self, script_: dict):
        super().__init__(script_)
        self.rephrasing, self.release = threading.Event(), threading.Event()

    def generate(self, prompt: str) -> tuple[str, float]:
        if pr.REPHRASE in prompt:
            self.rephrasing.set()
            if not self.release.wait(WAIT_S):
                raise TimeoutError("the test never released the rephrase")
        return super().generate(prompt)


# A FakeBridge that notes the thread of every query.
class ThreadBridge(FakeBridge):
    def __init__(self, replies: dict):
        super().__init__(replies)
        self.threads: list[str] = []

    def query(self, sql: str) -> dict:
        self.threads.append(threading.current_thread().name)
        return super().query(sql)


# A FakeBridge whose start fails.
class FailingStart(FakeBridge):
    def start(self) -> dict:
        raise RuntimeError("the bridge didn't start")


# Waits until the model worker has loaded the model, or failed to.
def wait_model(engine) -> None:
    deadline = time.monotonic() + WAIT_S
    while engine.status()["model"]["state"] == "loading":
        assert time.monotonic() < deadline, "the model never loaded"
        time.sleep(0.001)


# Builds runners over a real Engine with fake models and bridges; shuts each down and closes its engine at the end.
@pytest.fixture
def runners(paths):
    made = []

    def make(model=None, replies=None, *, model_factory=None, bridge=FakeBridge, load_bridges=True, ready=True):
        fakes = {}

        def bridge_factory(name, tables_tsv):
            fakes[name] = bridge(dict(ALL_REPLIES if replies is None else replies))
            return fakes[name]

        if model is not None:
            model_factory = lambda: model  # noqa: E731
        engine = Engine(paths, EngineOptions("mlx" if model_factory else "none"),
                        bridge_factory=bridge_factory, model_factory=model_factory)
        runner = Recording(engine)
        runner.fakes = fakes
        made.append(runner)
        if load_bridges:
            engine.start_bridges()
            assert engine.wait_bridges(WAIT_S)
        runner.start()
        if ready:
            wait_model(engine)
        return runner

    yield make
    for runner in made:
        runner.shutdown()
        runner.engine.close()


# Waits for a job to end and returns its last snapshot.
def finish(runner, job_id: str) -> dict:
    deadline = time.monotonic() + WAIT_S
    snapshot = runner.wait(job_id, 0, 0)
    while snapshot["stage"] not in ENDED:
        assert time.monotonic() < deadline, f"job {job_id} stuck at {snapshot['stage']}"
        snapshot = runner.wait(job_id, snapshot["version"], WAIT_S)
    return snapshot


# Submits a job that must be refused, and checks the refusal's status and message.
def refused(runner, kind: str, status: int, message: str, dataset: str = "tpch") -> None:
    with pytest.raises(Refusal) as e:
        runner.submit(kind, dataset, QUESTION)
    assert (e.value.status, e.value.message) == (status, message)


# Every string inside a snapshot.
def strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in strings(v)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in strings(v)]
    return []


# The check row names in a snapshot.
def row_names(snapshot: dict) -> list[str]:
    return [r["name"] for r in snapshot["checks"]]


def test_snapshot_versions_increase_and_wait_returns_newer(runners):
    gate = threading.Event()
    model = FakeModel(script(), gate=gate)
    runner = runners(model)
    queued = runner.submit("ask", "tpch", QUESTION)
    assert (queued["version"], queued["stage"], queued["ahead"]) == (1, "queued", 0)

    opener = threading.Timer(0.05, gate.set)
    try:
        assert model.called.wait(WAIT_S)
        writing = runner.wait(queued["id"], 1, WAIT_S)
        assert writing["stage"] == "writing" and writing["version"] > 1

        opener.start()
        began = time.monotonic()
        newer = runner.wait(queued["id"], writing["version"], WAIT_S)
        assert newer["version"] > writing["version"]
        assert time.monotonic() - began < 2.0  # woken by the change, long before the timeout
    finally:
        gate.set()
    finish(runner, queued["id"])
    opener.join(WAIT_S)

    seen = runner.seen(queued["id"])
    assert [s["version"] for s in seen] == list(range(1, len(seen) + 1))
    assert [s["stage"] for s in seen] == ["queued", "writing", "first", "done"]


def test_wait_times_out_with_current(runners):
    runner = runners(FakeModel(script()))
    done = finish(runner, runner.submit("ask", "tpch", QUESTION)["id"])

    began = time.monotonic()
    again = runner.wait(done["id"], done["version"], timeout_s=0.05)
    assert time.monotonic() - began >= 0.04
    assert again["version"] == done["version"] and again == done
    assert runner.wait("999", 0, 0.01) is None
    assert LONG_POLL_S == 20.0


def test_snapshots_never_change_once_handed_out(runners):
    runner = runners(FakeModel(script()))
    queued = runner.submit("ask", "tpch", QUESTION)
    kept = copy.deepcopy(queued)
    finish(runner, queued["id"])

    assert queued == kept
    seen = runner.seen(queued["id"])
    assert len({id(s) for s in seen}) == len(seen)  # a new dict per version
    assert SNAPSHOT_KEYS == ("id", "version", "kind", "dataset", "text", "asked", "stage", "ahead", "verdict",
                             "first", "second", "checks", "terms", "tested", "timing", "error")
    assert all(tuple(s) == SNAPSHOT_KEYS for s in seen)
    assert isinstance(queued["id"], str)


def test_sql_runs_while_question_mid_flight(runners):
    gate = threading.Event()
    model = FakeModel(script(), gate=gate)
    runner = runners(model)
    question = runner.submit("ask", "tpch", QUESTION)
    try:
        assert model.called.wait(WAIT_S)
        sql = finish(runner, runner.submit("sql", "tpch", TYPED)["id"])
        assert sql["stage"] == "done" and sql["first"]["row_count"] == 5
        assert runner.wait(question["id"], 0, 0)["stage"] == "writing"
    finally:
        gate.set()
    assert finish(runner, question["id"])["stage"] == "done"


# Holds a job while its rephrase runs, stops it there, then lets the rephrase finish.
def stop_during_rephrase(runner, model: RephraseGate) -> str:
    job_id = runner.submit("ask", "tpch", QUESTION)["id"]
    try:
        assert model.rephrasing.wait(WAIT_S)
        assert runner.wait(job_id, 0, 0)["stage"] == "first"
        assert runner.stop(job_id) is True
    finally:
        model.release.set()
    return job_id


def test_stop_between_first_and_second(runners):
    model = RephraseGate(script(second_sql=FEWER_SQL))
    runner = runners(model)
    job_id = stop_during_rephrase(runner, model)
    stopped = finish(runner, job_id)

    assert stopped["stage"] == "stopped"
    assert stopped["first"]["rows"] == [[r, "5"] for r in REGIONS]
    assert stopped["second"] is None
    assert stopped["verdict"] == STOPPED
    assert stopped["tested"] is None  # no tested line beside an answer without its second answer
    assert "second_phrasing" not in row_names(stopped) and "engine_traps" in row_names(stopped)
    assert runner.rows(job_id, "second", 0, 10) is None and runner.csv(job_id, "second") is None
    assert runner.rows(job_id, "first", 0, 1) == [["AFRICA", "5"]]
    assert runner.fakes["tpch"].sent == [SQL, FEWER_SQL]  # it ran (it can't be cut short), then was dropped
    assert [s["stage"] for s in runner.seen(job_id)][-2:] == ["first", "stopped"]
    assert all(s["second"] is None for s in runner.seen(job_id))


def test_failure_after_a_stop_at_first_still_stops(runners):
    model = RephraseGate({**script(), pr.REPHRASE: KeyError("no recorded reply for this prompt")})
    runner = runners(model)
    stopped = finish(runner, stop_during_rephrase(runner, model))

    assert (stopped["stage"], stopped["error"], stopped["verdict"]) == ("stopped", None, STOPPED)
    assert stopped["first"]["row_count"] == 5


def test_stop_while_writing_takes_effect_at_first(runners):
    gate = threading.Event()
    model = FakeModel(script(), gate=gate)
    runner = runners(model)
    job_id = runner.submit("ask", "tpch", QUESTION)["id"]
    try:
        assert model.called.wait(WAIT_S)  # the first answer is being written
        assert runner.stop(job_id) is True
    finally:
        gate.set()
    stopped = finish(runner, job_id)

    assert stopped["stage"] == "stopped"
    assert stopped["first"]["rows"] == [[r, "5"] for r in REGIONS]
    assert stopped["second"] is None
    assert stopped["verdict"] == STOPPED
    assert "second_phrasing" not in row_names(stopped) and "engine_traps" in row_names(stopped)
    assert stopped["tested"] is None
    assert stopped["timing"]["first_s"] is not None and stopped["timing"]["verdict_s"] is not None
    assert not any(pr.REPHRASE in p for p in model.prompts)
    assert runner.fakes["tpch"].sent == [SQL]
    assert [s["stage"] for s in runner.seen(job_id)][-2:] == ["first", "stopped"]
    assert runner.rows(job_id, "first", 0, 2) == [["AFRICA", "5"], ["AMERICA", "5"]]
    assert runner.stop(job_id) is False


def test_stop_while_writing_ends_as_a_final_answer_says(runners):
    gate = threading.Event()
    model = FakeModel({QUESTION: json.dumps({"cannot_answer": "The data has no weather records."})}, gate=gate)
    runner = runners(model)
    job_id = runner.submit("ask", "tpch", QUESTION)["id"]
    try:
        assert model.called.wait(WAIT_S)
        assert runner.stop(job_id) is True
    finally:
        gate.set()
    done = finish(runner, job_id)
    assert done["stage"] == "done" and done["verdict"]["kind"] == "cannot"


# A FakeBridge whose queries wait for the test, so a SQL job can be stopped while it runs.
class HeldBridge(FakeBridge):
    def __init__(self, replies: dict):
        super().__init__(replies)
        self.running, self.release = threading.Event(), threading.Event()

    def query(self, sql: str) -> dict:
        self.running.set()
        if not self.release.wait(WAIT_S):
            raise TimeoutError("the test never released the query")
        return super().query(sql)


def test_stop_while_sql_runs_drops_the_result(runners):
    runner = runners(bridge=HeldBridge)
    bridge = runner.fakes["tpch"]
    job_id = runner.submit("sql", "tpch", TYPED)["id"]
    try:
        assert bridge.running.wait(WAIT_S)
        assert runner.stop(job_id) is True
    finally:
        bridge.release.set()
    stopped = finish(runner, job_id)

    assert stopped["stage"] == "stopped"
    assert stopped["verdict"] == {"kind": "stopped", "text": "Stopped. The query finished, but the result was dropped.", "sql": None}
    assert (stopped["first"], stopped["second"], stopped["checks"], stopped["tested"]) == (None, None, [], None)
    assert runner.rows(job_id, "first", 0, 10) is None and runner.csv(job_id, "first") is None
    assert bridge.sent == [TYPED]  # it ran (the engine can't cancel), then the result was dropped
    assert runner.stop(job_id) is False
    assert copy_rules.problems(stopped["verdict"]["text"]) == []


def test_stop_while_queued(runners):
    gate = threading.Event()
    model = FakeModel({**script(), OTHER: reply(SQL, GRAIN, COLUMNS)}, gate=gate)
    runner = runners(model)
    running = runner.submit("ask", "tpch", QUESTION)
    try:
        assert model.called.wait(WAIT_S)
        queued = runner.submit("ask", "tpch", OTHER)
        last = runner.submit("ask", "tpch", OTHER)
        assert (queued["ahead"], last["ahead"]) == (1, 2)

        assert runner.stop(queued["id"]) is True
        stopped = runner.wait(queued["id"], 0, 0)
        assert (stopped["stage"], stopped["verdict"], stopped["first"]) == ("stopped", None, None)
        assert stopped["timing"]["verdict_s"] is not None
        assert runner.wait(last["id"], 0, 0)["ahead"] == 1  # moved up when the job before it left
        assert runner.stop(queued["id"]) is False
        assert runner.stop("999") is False
    finally:
        gate.set()
    assert finish(runner, running["id"])["stage"] == "done"
    assert finish(runner, last["id"])["stage"] == "done"
    assert [s["stage"] for s in runner.seen(queued["id"])] == ["queued", "stopped"]


def test_queued_ahead_count(runners):
    gate = threading.Event()
    model = FakeModel({**script(), OTHER: reply(SQL, GRAIN, COLUMNS)}, gate=gate)
    runner = runners(model)
    first = runner.submit("ask", "tpch", QUESTION)
    try:
        assert model.called.wait(WAIT_S)
        second = runner.submit("ask", "tpch", OTHER)
        third = runner.submit("ask", "tpch", OTHER)
        assert (second["stage"], second["ahead"]) == ("queued", 1)
        assert (third["stage"], third["ahead"]) == ("queued", 2)
    finally:
        gate.set()
    for job in (first, second, third):
        assert finish(runner, job["id"])["stage"] == "done"

    order = [(s["id"], s["stage"]) for s in runner.seen()]
    assert order.index((first["id"], "done")) < order.index((second["id"], "writing"))
    assert order.index((second["id"], "done")) < order.index((third["id"], "writing"))
    assert [(s["stage"], s["ahead"]) for s in runner.seen(third["id"])][:3] == [
        ("queued", 2), ("queued", 1), ("writing", 0)]


def test_history_newest_first_capped(runners):
    runner = runners()
    ids = [runner.submit("sql", "tpch", TYPED)["id"] for _ in range(HISTORY_CAP + 5)]
    last = finish(runner, ids[-1])
    history = runner.history()

    assert HISTORY_CAP == 100
    assert [h["id"] for h in history] == ids[::-1][:HISTORY_CAP]
    assert all(list(h) == HISTORY_KEYS for h in history)
    assert history[0] == {"id": ids[-1], "kind": "sql", "dataset": "tpch", "text": TYPED,
                          "verdict": last["verdict"], "from_job": None}

    # The oldest finished jobs are gone, with their rows.
    assert runner.wait(ids[0], 0, 0) is None
    assert runner.rows(ids[0], "first", 0, 10) is None and runner.csv(ids[0], "first") is None
    assert runner.rows(ids[5], "first", 0, 2) == [["ARGENTINA"], ["BRAZIL"]]


def test_edit_links_to_its_question_in_history(runners):
    runner = runners(FakeModel(script()))
    question = finish(runner, runner.submit("ask", "tpch", QUESTION)["id"])
    edit = finish(runner, runner.submit("edit", "tpch", SQL, from_job=question["id"])["id"])
    history = runner.history()

    assert [(h["id"], h["kind"], h["from_job"]) for h in history] == [
        (edit["id"], "edit", question["id"]), (question["id"], "ask", None)]
    assert history[1]["verdict"] == question["verdict"]


def test_model_not_ready_refused(runners):
    release = threading.Event()

    def slow():
        release.wait(WAIT_S)
        return FakeModel(script())

    def broken():
        raise RuntimeError("no weights in the cache")

    loading = runners(model_factory=slow, ready=False)
    try:
        for kind in ("ask", "reask"):
            refused(loading, kind, 409, "The model is still loading.")
        assert finish(loading, loading.submit("sql", "tpch", TYPED)["id"])["stage"] == "done"
    finally:
        release.set()

    off = runners()
    refused(off, "ask", 409, "The analyst is off (started with --no-model).")
    assert off.history() == []

    failed = runners(model_factory=broken)
    refused(failed, "ask", 409, "The model didn't load: no weights in the cache")
    assert finish(failed, failed.submit("sql", "tpch", TYPED)["id"])["stage"] == "done"


def test_dataset_not_ready_refused(runners):
    runner = runners()
    detail = runner.engine.status()["datasets"]["reviews"]["detail"]
    refused(runner, "sql", 409, f"{LABELS['reviews']} isn't available: {detail}", "reviews")
    refused(runner, "sql", 400, "Unknown dataset. Use tpch or reviews.", "nope")

    unloaded = runners(load_bridges=False)
    refused(unloaded, "sql", 409, f"{LABELS['tpch']} isn't loaded yet.")

    broken = runners(bridge=FailingStart)
    refused(broken, "sql", 409, f"{LABELS['tpch']} isn't available: the bridge didn't start")


def test_job_exception_becomes_error_state(runners, capsys):
    model = FakeModel({**script(), BOOM: RuntimeError("boom")})
    runner = runners(model)
    failed = finish(runner, runner.submit("ask", "tpch", BOOM)["id"])
    err = capsys.readouterr().err
    assert "Traceback" in err and "RuntimeError: boom" in err  # the terminal gets the trace

    assert failed["stage"] == "error" and "boom" in failed["error"]["message"]
    assert (failed["verdict"], failed["first"], failed["tested"]) == (None, None, None)
    assert finish(runner, runner.submit("ask", "tpch", QUESTION)["id"])["stage"] == "done"


def test_error_after_first_keeps_it_and_has_no_repr_quotes(runners):
    model = FakeModel({**script(), pr.REPHRASE: KeyError("no recorded reply for this prompt")})
    runner = runners(model)
    job_id = runner.submit("ask", "tpch", QUESTION)["id"]
    failed = finish(runner, job_id)

    assert failed["stage"] == "error"
    assert failed["error"] == {"message": "no recorded reply for this prompt"}
    assert failed["first"]["row_count"] == 5
    assert "second_phrasing" not in row_names(failed)
    assert (failed["verdict"], failed["tested"]) == (None, None)
    assert runner.rows(job_id, "first", 0, 1) == [["AFRICA", "5"]]


def test_refused_rows_never_leave(runners):
    runner = runners(FakeModel(script(first_sql=LEFT_SQL)))
    job_id = runner.submit("ask", "tpch", QUESTION)["id"]
    done = finish(runner, job_id)

    assert (done["stage"], done["first"], done["second"]) == ("done", None, None)
    assert done["verdict"]["kind"] == "refused"
    assert done["verdict"]["text"].startswith("Can't answer this yet.")
    assert done["verdict"]["sql"] == LEFT_SQL
    assert [(r["name"], r["word"]) for r in done["checks"]] == [
        ("second_phrasing", "Note"), ("engine_traps", "Refused")]
    assert done["checks"][0]["text"] == "Not asked a second way, because the first answer was refused."
    assert done["terms"] == []
    assert done["timing"]["first_s"] is None and done["timing"]["engine_ms"] is None
    for answer in ("first", "second"):
        assert runner.rows(job_id, answer, 0, 500) is None and runner.csv(job_id, answer) is None
    assert runner.fakes["tpch"].sent == [LEFT_SQL]  # it ran, as in the analyst check, then was withheld
    assert not any("AFRICA" in json.dumps(s) for s in runner.seen(job_id))


def test_timing_fields(runners):
    runner = runners(FakeModel(script()))
    job_id = runner.submit("ask", "tpch", QUESTION)["id"]
    done = finish(runner, job_id)
    by_stage = {s["stage"]: s["timing"] for s in runner.seen(job_id)}

    assert by_stage["queued"] == NO_TIMING and by_stage["writing"] == NO_TIMING
    first = by_stage["first"]
    assert first["first_s"] is not None and first["verdict_s"] is None and first["engine_ms"] == MS
    assert first["first_text"] == f"{first['first_s']:.1f} s" and first["verdict_text"] is None
    final = done["timing"]
    assert final["first_s"] == first["first_s"] <= final["verdict_s"]
    assert final["engine_ms"] == MS
    assert all(round(v, 1) == v for v in (final["first_s"], final["verdict_s"]))
    assert (final["first_text"], final["verdict_text"]) == (f"{final['first_s']:.1f} s", f"{final['verdict_s']:.1f} s")


# The page shows seconds as the server words them: one decimal, then " s".
def test_seconds_text():
    assert [jobs.seconds_text(s) for s in (2.5, 5.3, 0.0, 12.0, 30.1)] == ["2.5 s", "5.3 s", "0.0 s", "12.0 s", "30.1 s"]
    assert jobs.seconds_text(None) is None


def test_cannot_and_failed_end_done(runners):
    cannot = runners(FakeModel({QUESTION: json.dumps({"cannot_answer": "The data has no weather records."})}))
    done = finish(cannot, cannot.submit("ask", "tpch", QUESTION)["id"])
    assert done["stage"] == "done"
    assert done["verdict"] == {"kind": "cannot", "sql": None,
                               "text": "Can't answer this yet. Model's reason: The data has no weather records."}
    assert (done["first"], done["second"], done["checks"], done["terms"], done["error"]) == (None, None, [], [], None)
    assert done["tested"] is None
    assert done["timing"]["first_s"] is None and done["timing"]["engine_ms"] is None

    unreadable = runners(FakeModel({QUESTION: "Here is my answer, but not as JSON."}))
    done = finish(unreadable, unreadable.submit("ask", "tpch", QUESTION)["id"])
    assert done["verdict"] == {"kind": "failed", "sql": None, "text": (
        "No answer. The model's SQL failed 3 times. Last error: The last reply couldn't be read as SQL.")}

    erroring = runners(FakeModel({QUESTION: reply(BAD_SQL, GRAIN, COLUMNS)}))
    done = finish(erroring, erroring.submit("ask", "tpch", QUESTION)["id"])
    assert done["stage"] == "done" and done["first"] is None
    assert done["verdict"] == {"kind": "failed", "sql": BAD_SQL, "text": (
        "No answer. The model's SQL failed 3 times. Last error: Parse error: Expected a condition after WHERE")}


# The tested line stands only beside an answer the checks looked at: none for can't answer or failed,
# kept for a refused answer (its Refused row is a check) and a clean one.
def test_tested_line_only_where_checks_ran(runners):
    cases = {
        "cannot": FakeModel({QUESTION: json.dumps({"cannot_answer": "The data has no weather records."})}),
        "failed": FakeModel({QUESTION: "Here is my answer, but not as JSON."}),
        "refused": FakeModel(script(first_sql=LEFT_SQL)),
        "clean": FakeModel(script()),
    }
    tested = {}
    for kind, model in cases.items():
        runner = runners(model)
        runner.engine.lock, runner.engine.measured = Lock(True, None), {"rates": RATES}
        done = finish(runner, runner.submit("ask", "tpch", QUESTION)["id"])
        assert done["verdict"]["kind"] == kind
        tested[kind] = done["tested"]

    assert tested["cannot"] is None and tested["failed"] is None
    for kind in ("refused", "clean"):
        assert tested[kind] == check_views.tested_line(Lock(True, None), "ask", RATES), kind
        assert tested[kind]["text"].startswith("Checked answers."), kind


def test_sql_jobs_end_done_or_error(runners):
    runner = runners()
    ran = finish(runner, runner.submit("sql", "tpch", TYPED)["id"])
    assert ran["stage"] == "done"
    assert ran["verdict"] == {"kind": "ran", "text": "No check flagged this result.", "sql": None}
    assert ran["first"]["rows"] == [[n] for n in AMERICAS] and ran["timing"]["engine_ms"] == MS
    assert (ran["second"], ran["terms"], ran["tested"]) == (None, [], None)
    assert "second_phrasing" not in row_names(ran)
    assert runner.csv(ran["id"], "first") == views.csv_text(TYPED_RESULT)

    not_run = finish(runner, runner.submit("sql", "tpch", LEFT_SQL)["id"])
    assert not_run["stage"] == "done" and not_run["verdict"]["kind"] == "not_run"
    assert not_run["verdict"]["text"].startswith("Not run.")
    assert [(r["name"], r["word"]) for r in not_run["checks"]] == [("engine_traps", "Not run")]
    assert not_run["first"] is None and runner.rows(not_run["id"], "first", 0, 10) is None

    # A failed run shows the engine's message under the input, and no check built from the error.
    failed = finish(runner, runner.submit("edit", "tpch", BAD_SQL)["id"])
    assert failed["stage"] == "error"
    assert failed["error"] == {"message": views.error_text(PARSE_ERROR)}
    assert failed["error"]["message"] == "Parse error: Expected a condition after WHERE"
    assert (failed["verdict"], failed["first"], failed["tested"]) == (None, None, None)
    assert row_names(failed) == ["engine_traps"]
    assert failed["timing"]["engine_ms"] is None and failed["timing"]["verdict_s"] is not None
    assert runner.csv(failed["id"], "first") is None
    assert runner.history()[0]["verdict"] is None
    assert runner.fakes["tpch"].sent == [TYPED, BAD_SQL]  # the LEFT JOIN never reached the engine


def test_second_answer_rows(runners):
    runner = runners(FakeModel(script(second_sql=FEWER_SQL)))
    job_id = runner.submit("ask", "tpch", QUESTION)["id"]
    done = finish(runner, job_id)

    assert done["second"]["question"] == REPHRASED
    assert done["second"]["sql"] == FEWER_SQL and done["second"]["grain"] is None
    assert done["first"]["question"] is None and done["first"]["grain"] == GRAIN
    assert done["checks"][0]["word"] == "Differs"
    assert done["verdict"]["kind"] == "warned"
    assert runner.rows(job_id, "second", 0, 1) == [["AFRICA", "4"]]
    assert runner.csv(job_id, "second") == views.csv_text(FEWER)
    assert runner.rows(job_id, "third", 0, 1) is None


def test_second_answer_with_an_engine_trap_is_not_shown(runners):
    runner = runners(FakeModel(script(second_sql=LEFT_SQL)))
    job_id = runner.submit("ask", "tpch", QUESTION)["id"]
    done = finish(runner, job_id)

    assert done["second"] is None and done["first"] is not None
    assert runner.rows(job_id, "second", 0, 10) is None and runner.csv(job_id, "second") is None
    assert done["checks"][0]["word"] == "Same"  # the analyst check's own row, unchanged
    assert done["checks"][0]["detail"] == f'Asked as: "{REPHRASED}"'  # no count from the trapped SQL
    assert runner.fakes["tpch"].sent == [SQL, LEFT_SQL]


def test_second_answer_shown_keeps_its_row_counts(runners):
    done = finish(runner := runners(FakeModel(script())), runner.submit("ask", "tpch", QUESTION)["id"])
    assert done["second"] is not None
    assert done["checks"][0]["detail"] == f'Asked as: "{REPHRASED}" First answer 5 rows, second answer 5 rows.'


def test_model_prose_loses_em_dashes(runners):
    cannot = runners(FakeModel({QUESTION: json.dumps({"cannot_answer": f"No weather data {DASH} only orders."})}))
    done = finish(cannot, cannot.submit("ask", "tpch", QUESTION)["id"])
    assert done["verdict"]["text"] == "Can't answer this yet. Model's reason: No weather data, only orders."

    rephrased = f"For every region {DASH} count its nations."
    answered = runners(FakeModel(script(rephrased=rephrased, grain=f"one row per region {DASH} five rows")))
    done = finish(answered, answered.submit("ask", "tpch", QUESTION)["id"])
    assert done["first"]["grain"] == "one row per region, five rows"
    assert done["second"]["question"] == "For every region, count its nations."
    assert done["checks"][0]["detail"].startswith('Asked as: "For every region, count its nations."')
    assert [s for s in strings(done) if copy_rules.problems(s)] == []


def test_em_dash_in_data_stays(runners):
    cell = f"North {DASH} South"
    sql = "SELECT r_name FROM region"
    runner = runners(replies={sql: bridge_ok([("r_name", "VARCHAR")], [[cell]])})
    done = finish(runner, runner.submit("sql", "tpch", sql)["id"])

    assert done["first"]["rows"] == [[cell]]
    assert runner.rows(done["id"], "first", 0, 1) == [[cell]]
    assert cell in runner.csv(done["id"], "first")


def test_snapshot_strings_pass_copy_rules(runners, monkeypatch):
    reask = check_views.reask_question(LATE_QUESTION, "late shipment", "received after commit")
    asked = {"term": "late shipment", "reading": "received after commit", "question": LATE_QUESTION}
    late = script(LATE_SQL, LATE_SQL, LATE_QUESTION, LATE_REPHRASED, "one row", LATE_COLUMNS)
    cases = [  # kind, text, asked, script, wide manifest
        ("ask", QUESTION, None, script(), False),
        ("ask", QUESTION, None, script(second_sql=FEWER_SQL), False),
        ("ask", QUESTION, None, script(first_sql=LEFT_SQL), False),
        ("sql", TYPED, None, {}, False),
        ("ask", LATE_QUESTION, None, late, True),
        ("reask", reask, asked, late, True),
    ]
    tested = []
    for kind, text, asked_for, model_script, wide in cases:
        runner = runners(FakeModel(model_script))
        runner.engine.lock, runner.engine.measured = Lock(True, None), {"rates": RATES}
        if wide:  # the late question reads lineitem and orders
            ds = replace(runner.engine.dataset("tpch"), manifest=WIDE_MANIFEST)
            monkeypatch.setattr(runner.engine, "dataset", lambda name, ds=ds: ds)
        done = finish(runner, runner.submit(kind, "tpch", text, asked_for)["id"])

        assert done["stage"] == "done", (kind, text)
        assert bool(done["terms"]) == wide
        tested.append(done["tested"])
        for snapshot in runner.seen():
            assert [s for s in strings(snapshot) if copy_rules.problems(s)] == [], (kind, text)
    assert tested[0]["text"].startswith("Checked answers.") and tested[3] is None
    assert tested[5]["text"] == check_views.REASK_LINE


def test_model_runs_only_on_model_worker(runners):
    built = []
    model = ThreadModel(script())

    def factory():
        built.append(threading.current_thread().name)
        return model

    runner = runners(model_factory=factory, bridge=ThreadBridge)
    finish(runner, runner.submit("ask", "tpch", QUESTION)["id"])
    finish(runner, runner.submit("sql", "tpch", TYPED)["id"])

    assert built == [MODEL_WORKER]
    assert model.threads == {MODEL_WORKER}
    assert runner.fakes["tpch"].threads == [MODEL_WORKER, MODEL_WORKER, SQL_WORKER]
    assert threading.current_thread().name not in (MODEL_WORKER, SQL_WORKER)


def test_shutdown_wakes_waiters_and_joins_workers(runners):
    gate = threading.Event()
    model = FakeModel({**script(), OTHER: reply(SQL, GRAIN, COLUMNS)}, gate=gate)
    runner = runners(model)
    running = runner.submit("ask", "tpch", QUESTION)
    closer = threading.Thread(target=runner.shutdown)
    try:
        assert model.called.wait(WAIT_S)
        queued = runner.submit("ask", "tpch", OTHER)
        current = runner.wait(running["id"], 0, 0)
        got = {}
        waiter = threading.Thread(
            target=lambda: got.update(snapshot=runner.wait(running["id"], current["version"], WAIT_S)))
        waiter.start()

        closer.start()
        waiter.join(WAIT_S / 2)
        assert not waiter.is_alive() and got["snapshot"]["id"] == running["id"]
        assert runner.wait(queued["id"], 0, 0)["stage"] == "stopped"
        refused(runner, "sql", 503, "The app is quitting.")
    finally:
        gate.set()
    closer.join(WAIT_S)

    assert not closer.is_alive()
    assert runner.wait(running["id"], 0, 0)["stage"] == "stopped"  # its stop took effect at the first answer
    assert not [t for t in threading.enumerate() if t.name in (MODEL_WORKER, SQL_WORKER) and t.is_alive()]


def test_jobs_files_pass_copy_rules():
    for path in (Path(jobs.__file__), Path(__file__)):
        assert copy_rules.problems(path.read_text(encoding="utf-8")) == [], path.name
