from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

from app import check_views, sixd, views
from app.analyst import Event, analyst_events
from app.copy_rules import plain
from app.datasets import DATASETS, LABELS, Dataset
from app.sql_runner import pre_run_traps, sql_events

ck = sixd.ck

HISTORY_CAP = 100
LONG_POLL_S = 20.0
SHUTDOWN_S = 5.0  # how long shutdown waits for each worker's current step

MODEL_WORKER, SQL_WORKER = "model-worker", "sql-worker"
MODEL_KINDS = ("ask", "reask")

# What the page draws, in this order; it reads no other field.
SNAPSHOT_KEYS = ("id", "version", "kind", "dataset", "text", "asked", "stage", "ahead", "verdict", "first", "second", "checks", "terms", "tested", "timing", "error")
ENDED = ("done", "stopped", "error")

STOPPED_TEXT = "Stopped before the second answer, so there's no verdict."
NO_SQL_REPLY = "The last reply couldn't be read as SQL."
MODEL_REFUSALS = {"loading": "The model is still loading.", "off": "The analyst is off (started with --no-model)."}


# A job the runner won't take, with the status and message the server sends back.
class Refusal(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


# One question or SQL run: what was asked, where it is, and what the page draws for it.
@dataclass(eq=False)
class Job:
    id: str
    kind: str
    dataset: str
    text: str
    asked: dict | None
    from_job: str | None
    ds: Dataset
    bridge: object
    submitted: float  # monotonic seconds
    version: int = 0
    stage: str = "queued"
    ahead: int = 0
    verdict: dict | None = None
    first: dict | None = None
    second: dict | None = None
    checks: list = field(default_factory=list)
    terms: list = field(default_factory=list)
    tested: dict | None = None
    timing: dict = field(default_factory=lambda: {"first_s": None, "verdict_s": None, "engine_ms": None, "first_text": None, "verdict_text": None})
    error: dict | None = None
    replies: dict = field(default_factory=dict)  # "first" / "second": bridge replies whose rows are shown
    stop: bool = False
    snapshot: dict | None = None


# One worker thread and its queue.
@dataclass(eq=False)
class Worker:
    name: str
    queue: deque = field(default_factory=deque)
    busy: Job | None = None
    thread: threading.Thread | None = None


# Seconds as the page shows them ("2.5 s"); None stays None.
def seconds_text(seconds: float | None) -> str | None:
    return None if seconds is None else f"{seconds:.1f} s"


# Text with its em dashes made plain; None stays None.
def _plain(text: str | None) -> str | None:
    return None if text is None else plain(text)


# An exception's message without repr quotes (a KeyError's own text), or its type when it has none.
def _message(e: Exception) -> str:
    if isinstance(e, KeyError) and e.args:
        return str(e.args[0])
    return str(e) or type(e).__name__


# A result ready to draw, with the model's grain and question made plain.
def _view(reply: dict, sql: str, grain: str | None = None, question: str | None = None) -> dict:
    view = views.answer_view(reply, sql, grain=grain, question=question)
    return {**view, "grain": _plain(view["grain"]), "meta": plain(view["meta"]), "question": _plain(view["question"])}


# Check rows with their text and detail made plain.
def _rows(results: list, phase: str, kind: str) -> list[dict]:
    rows = check_views.check_rows(results, phase, kind)
    return [{**r, "text": plain(r["text"]), "detail": _plain(r["detail"])} for r in rows]


# The verdict line made plain, with the withheld or failed SQL beside it.
def _verdict(results: list, phase: str, outcome: str, detail: str | None = None, asked: dict | None = None, sql: str | None = None) -> dict:
    found = check_views.verdict(results, phase, outcome, detail, asked)
    return {**found, "text": plain(found["text"]), "sql": sql}


# Rows without the second answer's, for an answer that won't get one now.
def _without_second(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["name"] != "second_phrasing"]


# The second answer's view and reply: only when it was shown and its SQL has no engine trap.
def _second_answer(rephrase: dict, ds: Dataset) -> tuple[dict | None, dict | None]:
    reply = rephrase.get("result")
    if rephrase.get("final") != "shown" or not isinstance(reply, dict):
        return None, None
    if pre_run_traps(rephrase["sql"], ds).level == "refuse":
        return None, None
    return _view(reply, rephrase["sql"], question=rephrase["text"]), reply


# Raises a Refusal unless the model is ready to answer.
def _check_model(model: dict) -> None:
    if model["state"] == "ready":
        return
    if model["state"] == "failed":
        raise Refusal(409, plain(f"The model didn't load: {model['detail']}"))
    raise Refusal(409, MODEL_REFUSALS[model["state"]])


# Raises a Refusal unless the dataset's bridge is ready.
def _check_dataset(name: str, info: dict) -> None:
    if info["state"] == "loading":
        raise Refusal(409, f"{LABELS[name]} isn't loaded yet.")
    if info["state"] != "ready":
        raise Refusal(409, plain(f"{LABELS[name]} isn't available: {info['detail']}"))


# Runs questions on the model worker and SQL on the SQL worker, and keeps each job's snapshot.
# Every shared field is read and written under one Condition, which also wakes the waiters.
class JobRunner:
    def __init__(self, engine):
        self.engine = engine
        self._cond = threading.Condition()
        self._jobs: dict[str, Job] = {}  # by id, oldest first
        self._workers = {name: Worker(name) for name in (MODEL_WORKER, SQL_WORKER)}
        self._next_id = 0
        self._started = False
        self._closing = False

    # Starts both workers; the model worker loads the model first, on its own thread.
    def start(self) -> None:
        with self._cond:
            if self._started or self._closing:
                return
            self._started = True
            for worker in self._workers.values():
                worker.thread = threading.Thread(target=self._work, args=(worker,), name=worker.name, daemon=True)
                worker.thread.start()

    # Queues a job and returns its first snapshot; Refusal when the dataset or the model isn't ready.
    def submit(self, kind: str, dataset: str, text: str, asked: dict | None = None, from_job: str | None = None) -> dict:
        if kind not in check_views.KINDS:
            raise ValueError(f"unknown job kind {kind!r}")
        if dataset not in DATASETS:
            raise Refusal(400, f"Unknown dataset. Use {' or '.join(DATASETS)}.")

        status = self.engine.status()
        if kind in MODEL_KINDS:
            _check_model(status["model"])
        _check_dataset(dataset, status["datasets"][dataset])
        ds, bridge = self.engine.dataset(dataset), self.engine.bridge(dataset)

        with self._cond:
            if self._closing:
                raise Refusal(503, "The app is quitting.")
            self._next_id += 1
            job = Job(str(self._next_id), kind, dataset, text, dict(asked) if asked else None, from_job, ds, bridge, time.monotonic())
            worker = self._worker(kind)
            job.ahead = len(worker.queue) + (worker.busy is not None)
            worker.queue.append(job)
            self._jobs[job.id] = job
            return self._publish(job)

    # The job's snapshot once its version passes after, else the current one at the timeout; None for an unknown id.
    def wait(self, job_id, after: int, timeout_s: float = LONG_POLL_S) -> dict | None:
        with self._cond:
            job = self._jobs.get(str(job_id))
            if job is None:
                return None
            self._cond.wait_for(lambda: job.version > after or self._closing, timeout_s)
            return job.snapshot

    # Stops a job at its next step; a queued one stops at once. True when the job hadn't ended.
    def stop(self, job_id) -> bool:
        with self._cond:
            job = self._jobs.get(str(job_id))
            if job is None or job.stage in ENDED:
                return False
            if job.stage == "queued":
                self._unqueue(job)
            else:
                job.stop = True
            return True

    # One page of an answer's rows, formatted; None when the job or that answer's rows aren't there.
    def rows(self, job_id, answer: str, offset: int, limit: int) -> list | None:
        reply = self._reply(job_id, answer)
        return None if reply is None else views.display_rows(reply, offset, limit)

    # Every row of an answer as CSV; None when the job or that answer's rows aren't there.
    def csv(self, job_id, answer: str) -> str | None:
        reply = self._reply(job_id, answer)
        return None if reply is None else views.csv_text(reply)

    # This session's jobs, newest first, at most HISTORY_CAP.
    def history(self) -> list[dict]:
        with self._cond:
            jobs = list(self._jobs.values())[-HISTORY_CAP:]
            return [{
                "id": j.id, 
                "kind": j.kind, 
                "dataset": j.dataset, 
                "text": j.text, 
                "verdict": j.verdict,
                "from_job": j.from_job
            } for j in reversed(jobs)]

    # Takes no more jobs, stops queued ones, flags running ones, wakes every waiter,
    # then waits a bounded time for the workers. Safe to call twice.
    def shutdown(self, timeout_s: float = SHUTDOWN_S) -> None:
        with self._cond:
            self._closing = True
            for worker in self._workers.values():
                for job in list(worker.queue):
                    self._unqueue(job)
                if worker.busy is not None:
                    worker.busy.stop = True
            self._cond.notify_all()

        deadline = time.monotonic() + timeout_s
        for worker in self._workers.values():
            if worker.thread is not None:
                worker.thread.join(max(0.0, deadline - time.monotonic()))

    # The worker that serves a kind of job.
    def _worker(self, kind: str) -> Worker:
        return self._workers[MODEL_WORKER if kind in MODEL_KINDS else SQL_WORKER]

    # A shown answer's bridge reply, read under the lock.
    def _reply(self, job_id, answer: str) -> dict | None:
        with self._cond:
            job = self._jobs.get(str(job_id))
            return None if job is None else job.replies.get(answer)

    # One worker's loop: its queue's jobs in order until shutdown. The model worker loads the model first,
    # so the model is only ever touched on that thread.
    def _work(self, worker: Worker) -> None:
        if worker.name == MODEL_WORKER:
            self.engine.load_model()
            
        while True:
            with self._cond:
                worker.busy = None
                self._cond.wait_for(lambda: worker.queue or self._closing)
                if self._closing:
                    return
                job = worker.busy = worker.queue.popleft()
                self._apply(job, {"stage": "writing", "ahead": 0})
                self._reahead(worker)
            self._run(job)

    # Runs one job's events; any exception ends the job in error and the worker carries on.
    def _run(self, job: Job) -> None:
        try:
            if job.kind in MODEL_KINDS:
                events = analyst_events(job.text, job.ds, self.engine.model, job.bridge, job.asked)
            else:
                events = sql_events(job.text, job.ds, job.bridge)
            try:
                self._pull(job, events)
            finally:
                events.close()
                
        except Exception as e:
            with self._cond:
                if not self._stop_at_first(job):  # a stopped second answer's failure doesn't matter
                    self._apply(job, {
                        "stage": "error", 
                        "error": {"message": plain(_message(e))}, 
                        "verdict": None,
                        "tested": None, 
                        "checks": _without_second(job.checks)
                    })

    # Pulls events one at a time. A stop after the first answer ends the job there: before the
    # second answer is asked for, or, if it was already running, by dropping it when it arrives.
    def _pull(self, job: Job, events) -> None:
        while True:
            with self._cond:
                if self._stop_at_first(job):
                    return

            event = next(events, None)
            with self._cond:
                if self._stop_at_first(job):
                    return
                
            if event is None:
                return
            if event.stage != "writing":  # the worker already says writing
                updates = self._updates(job, event)
                with self._cond:
                    self._apply(job, updates)

    # What one event changes in the snapshot, built outside the lock from the event and the job's fixed fields.
    def _updates(self, job: Job, event: Event) -> dict:
        kind, stage, record, results = job.kind, event.stage, event.record, event.checks
        tested = check_views.tested_line(self.engine.lock, kind, self.engine.measured.get("rates", {}))
        ended = {
            "stage": "done", "tested": tested, 
            "first": None, "second": None, 
            "checks": [], "terms": [],
            "replies": {}
        }

        if stage in ("not_run", "ran"):
            result, rows = record["result"], _rows(results, "done", kind)
            if stage == "not_run":
                return {**ended, "checks": rows, "verdict": _verdict(results, "done", "not_run")}
            
            if result.get("ok") is not True:  # the engine's message, and no check built from an error reply
                return {
                    **ended, "stage": "error", 
                    "tested": None, "checks": rows, "verdict": None,
                    "error": {"message": plain(views.error_text(result))}
                }
                
            return {
                **ended, "checks": rows, 
                "first": _view(result, record["sql"]),
                "verdict": _verdict(results, "done", "ran"), 
                "replies": {"first": result}
            }

        # No check ran on an answer the model couldn't or didn't give, so no tested line stands beside it.
        primary = record["primary"]
        if stage == "cannot":
            return {**ended, "tested": None, "verdict": _verdict([], "done", "cannot", primary["reason"])}
        
        if stage == "failed":
            last = primary["attempts"][-1]
            detail = NO_SQL_REPLY if last["reply"] is None else views.error_text(last["reply"])
            return {**ended, "tested": None, "verdict": _verdict([], "done", "failed", detail, sql=last["sql"])}

        phase, asked = ("first" if stage == "first" else "done"), job.asked
        if ck.overall(results) == "refused":  # no numbers, only why, with the SQL one click away
            rows = [r for r in _rows(results, phase, kind) if r["word"] == check_views.WORDS["refuse"] or r["name"] == "second_phrasing"]
            verdict = _verdict(results, phase, "shown", asked=asked, sql=primary["sql"])
            return {**ended, "checks": rows, "verdict": verdict}

        reply, question = primary["result"], asked["question"] if asked else job.text
        terms = check_views.term_rows(results, job.ds.spec, question, asked)
        updates = {
            "stage": stage, "tested": tested,
            "first": _view(reply, primary["sql"], grain=(primary["claims"] or {}).get("grain")),
            "second": None, "replies": {"first": reply},
            "checks": _rows(results, phase, kind),
            "terms": [{**t, "text": plain(t["text"])} for t in terms],
            "verdict": _verdict(results, phase, "shown", asked=asked),
        }
        if stage == "done":
            second, second_reply = _second_answer(record["rephrase"], job.ds)
            if second is not None:
                updates.update(second=second, replies={"first": reply, "second": second_reply})
        return updates

    # Sets the fields, then the timing, then publishes; an ended job may push the oldest out of history.
    # Called with the lock held.
    def _apply(self, job: Job, updates: dict) -> None:
        for name, value in updates.items():
            setattr(job, name, value)

        now = round(time.monotonic() - job.submitted, 1)
        first = job.replies.get("first")
        first_s = now if job.stage == "first" else job.timing["first_s"]
        verdict_s = now if job.stage in ENDED else None
        job.timing = {
            "first_s": first_s,
            "verdict_s": verdict_s,
            "engine_ms": first.get("ms") if first else None,
            "first_text": seconds_text(first_s),
            "verdict_text": seconds_text(verdict_s),
        }
        self._publish(job)
        if job.stage in ENDED:
            self._trim()

    # Ends a job as stopped when a stop came after its first answer: that answer stays, no verdict.
    # True when it did. Called with the lock held.
    def _stop_at_first(self, job: Job) -> bool:
        if job.stage != "first" or not job.stop:
            return False
        
        verdict = {"kind": "stopped", "text": STOPPED_TEXT, "sql": None}
        self._apply(job, {"stage": "stopped", "verdict": verdict, "checks": _without_second(job.checks)})
        return True

    # Takes a queued job out of its queue as stopped. Called with the lock held.
    def _unqueue(self, job: Job) -> None:
        worker = self._worker(job.kind)
        worker.queue.remove(job)
        self._apply(job, {"stage": "stopped"})
        self._reahead(worker)

    # Each queued job's place: the jobs before it plus the one running; a change moves its version.
    # Called with the lock held.
    def _reahead(self, worker: Worker) -> None:
        running = worker.busy is not None
        for place, job in enumerate(worker.queue):
            if job.ahead != place + running:
                job.ahead = place + running
                self._publish(job)

    # Drops the oldest ended jobs beyond HISTORY_CAP, rows and all. Called with the lock held.
    def _trim(self) -> None:
        ended = [job for job in self._jobs.values() if job.stage in ENDED]
        for job in ended[:max(0, len(ended) - HISTORY_CAP)]:
            del self._jobs[job.id]

    # The job's next version as a new dict, waking every waiter. Called with the lock held.
    def _publish(self, job: Job) -> dict:
        job.version += 1
        job.snapshot = {key: getattr(job, key) for key in SNAPSHOT_KEYS}
        self._cond.notify_all()
        return job.snapshot
