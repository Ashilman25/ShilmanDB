"""Tests for app/server.py and app/__main__.py."""
from __future__ import annotations

import contextlib
import http.client
import json
import re
import socket
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

# check_views is reached as a module: pytest collects any test* name imported into a test module.
from app import check_views, copy_rules, sixd
from app.__main__ import main, pick_port
from app.datasets import DATASETS, default_paths, load_dataset, missing_files
from app.engine import Engine, EngineOptions
from app.jobs import SNAPSHOT_KEYS, JobRunner
from app.measured import Lock, read_json
from app.server import MAX_BODY, make_server
from app.tests.fakes import FakeBridge, FakeModel, bridge_ok
from app.tests.test_app_flows import LEFT_SQL, QUESTION, REPLIES, script

an, pr = sixd.an, sixd.pr

WAIT_S = 10.0
CLIENT_TIMEOUT_S = 30.0  # above the server's 20 s long poll, so only a real hang times out
TOKEN = "0123456789abcdef0123456789abcdef"
APP_DIR = Path(__file__).resolve().parents[1]
EXAMPLES = json.loads((APP_DIR / "examples.json").read_text(encoding="utf-8"))
RATES = read_json(APP_DIR / "measured.json")["rates"]
ENDED = ("done", "stopped", "error")

TYPED = "SELECT n_name FROM nation WHERE n_regionkey = 1"
AMERICAS = ["ARGENTINA", "BRAZIL", "CANADA", "PERU", "UNITED STATES"]
SERVER_REPLIES = {**REPLIES, TYPED: bridge_ok([("n_name", "VARCHAR")], [[n] for n in AMERICAS])}

DEV_QUESTIONS = Path("bench/results/analyst_check/questions.jsonl")
DEV_RUN = Path("bench/results/analyst_check/runs/dev_gemma4-26b.jsonl")
SECURITY_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"}


# One HTTP response, read in full.
@dataclass
class Reply:
    status: int
    headers: object  # http.client.HTTPMessage: case-insensitive get
    body: bytes

    def json(self):
        return json.loads(self.body)

    # The error message, which must pass the copy rules like every string the server sends.
    def error(self) -> str:
        message = self.json()["error"]
        assert copy_rules.problems(message) == []
        return message


# Talks to one server as the page would: the cookie on every call, the page's Origin on writes.
class Client:
    def __init__(self, port: int, token: str = TOKEN):
        self.port = port
        self.cookie = f"shilmandb_token_{port}={token}"
        self.origin = f"http://127.0.0.1:{port}"

    def send(self, method: str, path: str, body=None, headers=None, cookie=True, origin=True) -> Reply:
        sent = {}
        if cookie:
            sent["Cookie"] = self.cookie if cookie is True else cookie
        if method == "POST" and origin:
            sent["Origin"] = self.origin if origin is True else origin
        sent.update(headers or {})
        data = json.dumps(body).encode() if isinstance(body, (dict, list)) else body

        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=CLIENT_TIMEOUT_S)
        try:
            conn.request(method, path, body=data, headers=sent)
            resp = conn.getresponse()
            try:
                return Reply(resp.status, resp.headers, resp.read())
            finally:
                resp.close()
        finally:
            conn.close()

    def get(self, path: str, **kw) -> Reply:
        return self.send("GET", path, **kw)

    def post(self, path: str, body=None, **kw) -> Reply:
        return self.send("POST", path, body, **kw)

    # A POST sent header by header, so the test picks the Content-Length (or leaves it out) and sends no body.
    def bare_post(self, path: str, length: str | None) -> Reply:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=CLIENT_TIMEOUT_S)
        try:
            conn.putrequest("POST", path)
            conn.putheader("Cookie", self.cookie)
            conn.putheader("Origin", self.origin)
            if length is not None:
                conn.putheader("Content-Length", length)
            conn.endheaders()
            resp = conn.getresponse()
            try:
                return Reply(resp.status, resp.headers, resp.read())
            finally:
                resp.close()
        finally:
            conn.close()

    # Waits for a job to end over the long poll and returns its last snapshot.
    def finish(self, job_id: str) -> dict:
        deadline = time.monotonic() + WAIT_S
        snapshot = self.get(f"/api/jobs/{job_id}").json()
        while snapshot["stage"] not in ENDED:
            assert time.monotonic() < deadline, f"job {job_id} stuck at {snapshot['stage']}"
            snapshot = self.get(f"/api/jobs/{job_id}?after={snapshot['version']}").json()
        return snapshot


# One running app for a test: the client and what's behind it.
@dataclass
class App:
    client: Client
    runner: JobRunner
    engine: object
    port: int


# An engine stand-in for the status route, with a lock that holds or not.
class StubEngine:
    def __init__(self, holds: bool):
        self.lock = Lock(holds, None if holds else "a file changed")
        self.measured = {"rates": RATES}

    def status(self) -> dict:
        return {"datasets": {}, "model": {"state": "off", "mode": "none", "detail": ""},
                "lock": {"holds": self.lock.holds, "reason": self.lock.reason}, "cache": {"on": False, "reason": "off"}}

    def load_model(self) -> None:
        pass

    def close(self) -> None:
        pass


# Fixture paths with an empty app folder, so the engine finds no measured.json and hashes nothing.
@pytest.fixture
def paths(fixture_paths, tmp_path):
    return replace(fixture_paths, app_dir=tmp_path / "app", src=tmp_path / "src")


# Waits until the model has loaded, or failed to.
def wait_model(engine) -> None:
    deadline = time.monotonic() + WAIT_S
    while engine.status()["model"]["state"] == "loading":
        assert time.monotonic() < deadline, "the model never loaded"
        time.sleep(0.001)


# Starts servers over a runner and an engine with fake bridges (and a fake model when given), on free ports.
# Each is shut down in the app's own order at the end: server, runner, engine.
@pytest.fixture
def start(paths):
    made = []

    def make(model=None, *, engine=None, static_dir=None) -> App:
        if engine is None:
            factory = (lambda: model) if model is not None else None
            engine = Engine(paths, EngineOptions("mlx" if model is not None else "none"),
                            bridge_factory=lambda name, tsv: FakeBridge(dict(SERVER_REPLIES)), model_factory=factory)
            engine.start_bridges()
            assert engine.wait_bridges(WAIT_S)
        runner = JobRunner(engine)
        runner.start()
        server = make_server(runner, engine, TOKEN, 0, static_dir or APP_DIR / "static", EXAMPLES)
        # A short poll interval, so shutdown at the end doesn't wait the default half second.
        thread = threading.Thread(target=server.serve_forever, args=(0.01,), name="test-http", daemon=True)
        thread.start()
        made.append((server, thread, runner, engine))
        if model is not None:
            wait_model(engine)
        return App(Client(server.server_port), runner, engine, server.server_port)

    yield make
    for server, thread, runner, engine in made:
        server.shutdown()
        server.server_close()
        thread.join(WAIT_S)
        runner.shutdown()
        engine.close()


# A port nothing listens on right now.
def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# A socket listening on a port below the top of the range, so the next 20 ports exist.
@contextlib.contextmanager
def listening():
    while True:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        if s.getsockname()[1] < 65000:
            break
        s.close()
    try:
        s.listen(1)
        yield s.getsockname()[1]
    finally:
        s.close()


def test_page_needs_token(start, capsys):
    app = start()
    c = app.client

    page = c.get("/", cookie=False)
    assert page.status == 403
    assert page.headers["Content-Type"] == "text/html; charset=utf-8"
    assert "Open the link the app printed." in page.body.decode()

    login = c.get(f"/?token={TOKEN}", cookie=False)
    assert login.status == 303
    assert login.headers["Location"] == "/"
    assert login.headers["Set-Cookie"] == f"shilmandb_token_{app.port}={TOKEN}; HttpOnly; SameSite=Strict; Path=/"

    home = c.get("/", cookie=login.headers["Set-Cookie"].split(";")[0])
    assert home.status == 200
    assert home.headers["Content-Type"] == "text/html; charset=utf-8"
    assert home.body.decode().lstrip().lower().startswith("<!doctype html>")

    for reply in (page, login, home):
        for name, value in SECURITY_HEADERS.items():
            assert reply.headers[name] == value
    assert TOKEN not in capsys.readouterr().err  # no request log line, which would print the token


def test_wrong_token_gets_403_page(start):
    c = start().client
    wrong = c.get("/?token=" + "f" * 32, cookie=False)
    assert wrong.status == 403
    assert wrong.headers["Set-Cookie"] is None
    assert "Open the link the app printed." in wrong.body.decode()
    assert c.get("/?token=", cookie=False).status == 403


def test_cookie_named_for_the_port(start):
    app = start()
    c = app.client
    assert c.get("/api/status").status == 200
    assert c.get("/api/status", cookie=f"other=1; shilmandb_token_{app.port}={TOKEN}").status == 200

    for cookie in (False, f"shilmandb_token={TOKEN}", f"shilmandb_token_{app.port + 1}={TOKEN}",
                   f"shilmandb_token_{app.port}={TOKEN[:-1]}0", f"shilmandb_token_{app.port}="):
        refused = c.get("/api/status", cookie=cookie)
        assert refused.status == 403
        assert refused.error() == "Open the link the app printed."
    assert c.get("/static/index.html", cookie=False).status == 403
    assert c.get("/api/nowhere", cookie=False).status == 403  # even a 404 needs the cookie


def test_bad_host_rejected(start):
    app = start()
    c = app.client
    for host in (f"evil.example:{app.port}", f"127.0.0.1:{app.port + 1}", "127.0.0.1", f"LOCALHOST.:{app.port}"):
        refused = c.get("/api/status", headers={"Host": host})
        assert refused.status == 403, host
        assert refused.error() == "This address isn't the app's. Open the link the app printed."
    assert c.get(f"/?token={TOKEN}", cookie=False, headers={"Host": f"evil.example:{app.port}"}).status == 403
    assert c.get("/api/status", headers={"Host": f"localhost:{app.port}"}).status == 200


def test_post_needs_same_origin(start):
    app = start()
    c = app.client
    body = {"dataset": "tpch", "sql": TYPED}
    for origin in ("http://evil.example", f"http://127.0.0.1:{app.port + 1}", f"https://127.0.0.1:{app.port}", "null"):
        refused = c.post("/api/sql", body, origin=origin)
        assert refused.status == 403, origin
        assert refused.error() == "Writes must come from the app's own page."
    assert c.post("/api/sql", body, origin=False).status == 403

    ok = c.post("/api/sql", body, origin=f"http://localhost:{app.port}")
    assert ok.status == 200
    c.finish(ok.json()["id"])


def test_post_body_rules(start):
    c = start().client
    assert c.post("/api/sql", b"{nope").error() == "The request wasn't valid JSON."
    assert c.post("/api/sql", b"\xff\xfe\x00").error() == "The request wasn't valid JSON."
    assert c.post("/api/sql", [1, 2]).error() == "The request must be a JSON object."

    missing = c.post("/api/sql", {"dataset": "tpch"})
    assert (missing.status, missing.error()) == (400, 'The request needs "sql" as text.')
    wrong = c.post("/api/sql", {"dataset": 1, "sql": TYPED})
    assert (wrong.status, wrong.error()) == (400, 'The request needs "dataset" as text.')
    assert c.post("/api/sql", {"dataset": "tpch", "sql": TYPED, "from_job": 3}).error() == \
        'The request needs "from_job" as text.'

    too_big = c.bare_post("/api/sql", str(MAX_BODY + 1))
    assert (too_big.status, too_big.error()) == (413, "The request is larger than 1 MB.")
    no_length = c.bare_post("/api/sql", None)
    assert (no_length.status, no_length.error()) == (400, "The request needs a Content-Length header.")
    assert c.bare_post("/api/sql", "lots").status == 400


def test_unknown_route_and_wrong_method(start):
    c = start().client
    missing = c.get("/api/nowhere")
    assert (missing.status, missing.error()) == (404, "Not found.")
    assert c.get("/nowhere").status == 404

    for method, path in (("GET", "/api/ask"), ("POST", "/api/status"), ("PUT", "/api/history"),
                         ("DELETE", "/api/jobs/1"), ("GET", "/api/jobs/1/stop")):
        wrong = c.send(method, path)
        assert (wrong.status, wrong.error()) == (405, "Method not allowed."), (method, path)
        for name, value in SECURITY_HEADERS.items():
            assert wrong.headers[name] == value


def test_static_module_served_and_contained(start, tmp_path):
    static = tmp_path / "static"
    (static / "js").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html>\n<title>Test</title>\n", encoding="utf-8")
    (static / "js" / "x.js").write_text("export const x = 1;\n", encoding="utf-8")
    (static / "app.css").write_text("body { margin: 0; }\n", encoding="utf-8")
    (tmp_path / "server.py").write_text("SECRET = 1\n", encoding="utf-8")  # a real file just outside
    c = start(static_dir=static).client

    module = c.get("/static/js/x.js")
    assert (module.status, module.headers["Content-Type"]) == (200, "text/javascript; charset=utf-8")
    assert module.body == b"export const x = 1;\n"
    css = c.get("/static/app.css")
    assert (css.status, css.headers["Content-Type"]) == (200, "text/css; charset=utf-8")
    assert c.get("/").body == b"<!doctype html>\n<title>Test</title>\n"

    for path in ("/static/../server.py", "/static/js/../../server.py", "/static/%2e%2e/server.py",
                 "/static/js/%2E%2E/%2E%2E/server.py", "/static//etc/hosts", "/static/js", "/static/",
                 "/static/missing.js", "/static/js/x.js%00.css"):
        assert c.get(path).status == 404, path
    assert c.get("/static/js/x.js", cookie=False).status == 403


def test_ask_round_trip(start):
    app = start(FakeModel(script()))
    c = app.client
    queued = c.post("/api/ask", {"dataset": "tpch", "question": QUESTION})
    assert queued.status == 200
    assert queued.headers["Content-Type"] == "application/json; charset=utf-8"
    for name, value in SECURITY_HEADERS.items():
        assert queued.headers[name] == value
    first = queued.json()
    assert (first["kind"], first["text"], first["stage"]) == ("ask", QUESTION, "queued")

    snapshot = first
    deadline = time.monotonic() + WAIT_S
    while snapshot["stage"] != "done":
        assert time.monotonic() < deadline, f"stuck at {snapshot['stage']}"
        newer = c.get(f"/api/jobs/{first['id']}?after={snapshot['version']}")
        assert newer.status == 200
        assert newer.json()["version"] > snapshot["version"]  # the long poll skips to the newest snapshot
        snapshot = newer.json()

    assert tuple(snapshot) == SNAPSHOT_KEYS
    assert snapshot["first"]["row_count"] == 5
    assert c.get(f"/api/jobs/{first['id']}").json() == snapshot  # after defaults to -1: at once

    assert c.get("/api/jobs/999").status == 404
    assert c.get("/api/jobs/999").error() == "Job not found."
    for bad in ("1.5", "x", ""):
        refused = c.get(f"/api/jobs/{first['id']}?after={bad}")
        assert (refused.status, refused.error()) == (400, '"after" must be a whole number.')


def test_long_poll_returns_on_change(start):
    gate = threading.Event()
    model = FakeModel(script(), gate=gate)
    c = start(model).client
    job_id = c.post("/api/ask", {"dataset": "tpch", "question": QUESTION}).json()["id"]
    got = {}

    def poll(after: int) -> None:
        got["reply"] = c.get(f"/api/jobs/{job_id}?after={after}")
        got["at"] = time.monotonic()

    try:
        assert model.called.wait(WAIT_S)
        current = c.get(f"/api/jobs/{job_id}").json()
        assert current["stage"] == "writing"
        poller = threading.Thread(target=poll, args=(current["version"],), daemon=True)
        poller.start()
        time.sleep(0.2)
        assert poller.is_alive()  # it waits for the next change
        opened = time.monotonic()
        gate.set()
        poller.join(WAIT_S)
    finally:
        gate.set()

    assert not poller.is_alive()
    assert got["at"] - opened < 1.0
    assert got["reply"].json()["version"] > current["version"]
    c.finish(job_id)


def test_empty_and_too_long_rejected(start):
    c = start().client
    for text in ("", "   ", "\n\t "):
        refused = c.post("/api/ask", {"dataset": "tpch", "question": text})
        assert (refused.status, refused.error()) == (400, "Type a question first.")
        refused = c.post("/api/sql", {"dataset": "tpch", "sql": text})
        assert (refused.status, refused.error()) == (400, "Type some SQL first.")

    long_question = c.post("/api/ask", {"dataset": "tpch", "question": "x" * 2001})
    assert (long_question.status, long_question.error()) == (400, "Questions are limited to 2,000 characters.")
    long_sql = c.post("/api/sql", {"dataset": "tpch", "sql": "x" * 20001})
    assert (long_sql.status, long_sql.error()) == (400, "SQL is limited to 20,000 characters.")

    # At the limit the text passes, and the runner's own checks answer (an unknown dataset here).
    unknown = "Unknown dataset. Use tpch or reviews."
    assert c.post("/api/ask", {"dataset": "nope", "question": "x" * 2000}).error() == unknown
    assert c.post("/api/sql", {"dataset": "nope", "sql": " " + "x" * 20000 + "\n"}).error() == unknown
    off = c.post("/api/ask", {"dataset": "tpch", "question": QUESTION})
    assert (off.status, off.error()) == (409, "The analyst is off (started with --no-model).")


def test_sql_round_trip_trims_and_links(start):
    c = start().client
    typed = c.post("/api/sql", {"dataset": "tpch", "sql": f"\n  {TYPED}  \n"}).json()
    assert (typed["kind"], typed["text"]) == ("sql", TYPED)
    done = c.finish(typed["id"])
    assert done["first"]["rows"] == [[n] for n in AMERICAS]

    edit = c.post("/api/sql", {"dataset": "tpch", "sql": TYPED, "from_job": typed["id"]}).json()
    assert edit["kind"] == "edit"
    c.finish(edit["id"])
    history = c.get("/api/history").json()["history"]
    assert [(h["id"], h["kind"], h["from_job"]) for h in history] == [(edit["id"], "edit", typed["id"]),
                                                                      (typed["id"], "sql", None)]


def test_reask_builds_text(start):
    c = start(FakeModel(script())).client
    asked = c.post("/api/ask", {"dataset": "tpch", "question": QUESTION}).json()
    c.finish(asked["id"])

    reask = c.post("/api/reask", {"job": asked["id"], "term": "revenue", "reading": "gross"})
    assert reask.status == 200
    first = reask.json()
    assert first["kind"] == "reask"
    assert first["text"] == QUESTION + " Use the gross reading of revenue."
    assert first["text"].endswith(" Use the gross reading of revenue.")
    assert first["asked"] == {"term": "revenue", "reading": "gross", "question": QUESTION}
    assert c.finish(first["id"])["stage"] == "done"

    # A re-ask of a re-ask starts again from the typed question, so sentences never stack.
    again = c.post("/api/reask", {"job": first["id"], "term": "revenue", "reading": "net"}).json()
    assert again["text"] == QUESTION + " Use the net reading of revenue."
    assert again["asked"] == {"term": "revenue", "reading": "net", "question": QUESTION}
    c.finish(again["id"])
    history = c.get("/api/history").json()["history"]
    assert [h["from_job"] for h in history] == [first["id"], asked["id"], None]


def test_reask_refusals(start):
    gate = threading.Event()
    model = FakeModel(script(), gate=gate)
    c = start(model).client
    running = c.post("/api/ask", {"dataset": "tpch", "question": QUESTION}).json()
    try:
        assert model.called.wait(WAIT_S)
        busy = c.post("/api/reask", {"job": running["id"], "term": "revenue", "reading": "gross"})
        assert (busy.status, busy.error()) == (409, "That question is still running.")
    finally:
        gate.set()
    c.finish(running["id"])

    for term, reading in (("revenue", "tax"), ("profit", "gross"), ("positive review", "4 or 5 stars")):
        unknown = c.post("/api/reask", {"job": running["id"], "term": term, "reading": reading})
        assert (unknown.status, unknown.error()) == (400, "Unknown reading."), (term, reading)

    missing = c.post("/api/reask", {"job": "999", "term": "revenue", "reading": "gross"})
    assert (missing.status, missing.error()) == (404, "Job not found.")
    typed = c.post("/api/sql", {"dataset": "tpch", "sql": TYPED}).json()
    c.finish(typed["id"])
    sql = c.post("/api/reask", {"job": typed["id"], "term": "revenue", "reading": "gross"})
    assert (sql.status, sql.error()) == (400, "Only a finished question can be asked again.")
    assert c.post("/api/reask", {"job": int(running["id"]), "term": "revenue", "reading": "gross"}).error() == \
        'The request needs "job" as text.'


def test_csv_endpoint(start):
    c = start(FakeModel(script(first_sql=LEFT_SQL))).client
    typed = c.post("/api/sql", {"dataset": "tpch", "sql": TYPED}).json()
    c.finish(typed["id"])

    download = c.get(f"/api/jobs/{typed['id']}/csv?answer=first")
    assert download.status == 200
    assert download.headers["Content-Type"] == "text/csv; charset=utf-8"
    assert download.headers["Content-Disposition"] == f'attachment; filename="shilmandb-{typed["id"]}.csv"'
    assert download.body.decode() == "n_name\r\n" + "".join(f"{n}\r\n" for n in AMERICAS)

    refused = c.finish(c.post("/api/ask", {"dataset": "tpch", "question": QUESTION}).json()["id"])
    assert refused["verdict"]["kind"] == "refused"
    for path in (f"/api/jobs/{refused['id']}/csv?answer=first", f"/api/jobs/{typed['id']}/csv?answer=second",
                 f"/api/jobs/{refused['id']}/rows?answer=first"):
        gone = c.get(path)
        assert (gone.status, gone.error()) == (404, "That answer has no rows."), path
    assert c.get("/api/jobs/999/csv?answer=first").error() == "Job not found."
    bad = c.get(f"/api/jobs/{typed['id']}/csv?answer=third")
    assert (bad.status, bad.error()) == (400, '"answer" must be first or second.')


def test_rows_endpoint(start):
    c = start().client
    typed = c.post("/api/sql", {"dataset": "tpch", "sql": TYPED}).json()
    c.finish(typed["id"])

    page = c.get(f"/api/jobs/{typed['id']}/rows?answer=first&offset=3")
    assert page.status == 200
    assert page.json() == {"offset": 3, "rows": [["PERU"], ["UNITED STATES"]]}
    assert c.get(f"/api/jobs/{typed['id']}/rows?answer=first").json()["rows"] == [[n] for n in AMERICAS]
    for bad in ("-1", "two", "1e3"):
        refused = c.get(f"/api/jobs/{typed['id']}/rows?answer=first&offset={bad}")
        assert (refused.status, refused.error()) == (400, '"offset" must be a whole number, 0 or more.')


def test_stop_and_history(start):
    gate = threading.Event()
    model = FakeModel(script(), gate=gate)
    c = start(model).client
    job_id = c.post("/api/ask", {"dataset": "tpch", "question": QUESTION}).json()["id"]
    try:
        assert model.called.wait(WAIT_S)
        stopped = c.post(f"/api/jobs/{job_id}/stop")
        assert (stopped.status, stopped.json()) == (200, {"stopped": True})
    finally:
        gate.set()
    assert c.finish(job_id)["stage"] == "stopped"
    assert c.post(f"/api/jobs/{job_id}/stop").json() == {"stopped": False}
    assert c.post("/api/jobs/999/stop").status == 404
    assert c.post(f"/api/jobs/{job_id}/stop", origin=False).status == 403

    history = c.get("/api/history")
    assert history.status == 200
    assert [h["id"] for h in history.json()["history"]] == [job_id]


def test_status_has_examples_and_panel(start):
    app = start()
    status = app.client.get("/api/status").json()
    assert list(status) == ["datasets", "model", "lock", "cache", "examples", "tested_panel"]
    assert status["examples"] == EXAMPLES
    assert status["model"] == {"state": "off", "mode": "none", "detail": ""}
    assert status["datasets"]["tpch"]["state"] == "ready"
    assert status["lock"]["holds"] is False
    assert status["tested_panel"] == []  # no rate shows on a setup that differs from the tested one

    holding = start(engine=StubEngine(holds=True)).client.get("/api/status").json()
    assert holding["tested_panel"] == check_views.tested_panel(RATES)
    assert holding["tested_panel"] and holding["lock"] == {"holds": True, "reason": None}
    differs = start(engine=StubEngine(holds=False)).client.get("/api/status").json()
    assert differs["tested_panel"] == []


def test_server_strings_pass_copy_rules(start):
    c = start().client
    pages = [c.get("/").body.decode(), c.get("/", cookie=False).body.decode(),
             (APP_DIR / "static" / "index.html").read_text(encoding="utf-8")]
    for text in pages:
        assert copy_rules.problems(text) == []


# Whether a question uses a glossary term, as the terms check decides.
def _uses_term(question: str, spec: dict) -> bool:
    return any(re.search(w, question, re.IGNORECASE) for term in spec["glossary"] for w in term["words"])


def test_examples_file():
    assert list(EXAMPLES) == list(DATASETS)
    prompt_examples = json.loads((APP_DIR.parent / "bench/analyst/prompt_examples.json").read_text(encoding="utf-8"))
    for name, questions in EXAMPLES.items():
        assert len(questions) == 5 and len(set(questions)) == 5
        assert all(isinstance(q, str) and q == q.strip() and copy_rules.problems(q) == [] for q in questions)
        assert not set(questions) & {e["question"] for e in prompt_examples[name]}
        spec = json.loads((APP_DIR.parent / f"bench/analyst/datasets/{name}.json").read_text(encoding="utf-8"))
        assert any(_uses_term(q, spec) for q in questions), name

    root, paths = APP_DIR.parent, default_paths()
    if not (root / DEV_QUESTIONS).is_file() or not (root / DEV_RUN).is_file() or any(missing_files(paths, n) for n in DATASETS):
        pytest.skip("6d's dev questions, dev run or dataset files aren't here")

    dev = {(q["dataset"], q["question"]): q["qid"] for q in an.read_jsonl(root / DEV_QUESTIONS) if q["split"] == "dev"}
    run = {r["qid"]: r for r in an.read_jsonl(root / DEV_RUN)}
    for name, questions in EXAMPLES.items():
        ctx = load_dataset(paths, name).context
        for question in questions:
            qid = dev.get((name, question))
            assert qid is not None and qid in run, question
            base = an.sha1_text(pr.build_sql_prompt(ctx.description, ctx.rules, ctx.examples, question))
            assert base in {a["prompt_sha1"] for a in run[qid]["primary"]["attempts"]}, question


def test_pick_port_skips_taken():
    with listening() as taken:
        picked = pick_port(taken, False)
        assert taken < picked <= taken + 20
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", picked))  # it really is free
        with pytest.raises(OSError, match=rf"^Port {taken} is in use\.$"):
            pick_port(taken, True)
        with pytest.raises(OSError, match=rf"^No free port from {taken} to {taken}\.$"):
            pick_port(taken, False, tries=1)
    port = free_port()
    assert pick_port(port, True) == port


# Main's paths: the fixture's, with no bridge binary, so nothing starts a process.
@pytest.fixture
def main_paths(paths, tmp_path):
    return replace(paths, bridge=tmp_path / "no_bridge", app_results=tmp_path / "app_results")


# The URL line main prints, read from captured stdout as it arrives.
def wait_for_url(capsys, thread) -> str:
    seen = ""
    deadline = time.monotonic() + WAIT_S
    while "\n" not in seen:
        assert time.monotonic() < deadline and thread.is_alive(), f"no URL line yet: {seen!r}"
        seen += capsys.readouterr().out
        time.sleep(0.01)
    return seen.splitlines()[0]


def test_main_no_model_starts_and_stops(main_paths, capsys):
    port = free_port()
    stop = threading.Event()
    result = {}
    argv = ["--no-browser", "--no-model", "--port", str(port)]
    thread = threading.Thread(target=lambda: result.update(code=main(argv, stop=stop, paths=main_paths)),
                              name="test-main", daemon=True)
    thread.start()
    try:
        line = wait_for_url(capsys, thread)
        found = re.fullmatch(rf"ShilmanDB app: http://127\.0\.0\.1:{port}/\?token=([0-9a-f]{{32}})", line)
        assert found, line
        client = Client(port, found.group(1))
        login = client.get(f"/?token={found.group(1)}", cookie=False)
        assert login.status == 303
        status = client.get("/api/status")
        assert status.status == 200
        assert status.json()["model"]["state"] == "off"
        assert status.json()["examples"] == EXAMPLES
    finally:
        stop.set()
        thread.join(WAIT_S)

    assert not thread.is_alive()
    assert result == {"code": 0}
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))  # the server let go of its port


def test_main_fixed_port_in_use_exits_2(main_paths, capsys):
    with listening() as taken:
        assert main(["--no-browser", "--no-model", "--port", str(taken)], paths=main_paths) == 2
    assert f"Port {taken} is in use." in capsys.readouterr().err


def test_main_rejects_bad_model_flags(main_paths, tmp_path, capsys):
    run = tmp_path / "run.jsonl"
    run.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        main(["--no-browser", "--no-model", "--recorded-model", str(run)], paths=main_paths)
    assert e.value.code == 2
    assert "not allowed with" in capsys.readouterr().err

    with pytest.raises(SystemExit) as e:
        main(["--no-browser", "--recorded-model", str(tmp_path / "nowhere.jsonl")], paths=main_paths)
    assert e.value.code == 2
    assert f"--recorded-model: no file at {tmp_path / 'nowhere.jsonl'}" in capsys.readouterr().err

    with pytest.raises(SystemExit) as e:
        main(["--no-browser", "--recorded-model", str(tmp_path)], paths=main_paths)  # a folder isn't a file
    assert e.value.code == 2

    with pytest.raises(SystemExit) as e:
        main(["--no-browser", "--no-model", "--recorded-pace"], paths=main_paths)
    assert e.value.code == 2
    assert "--recorded-pace needs --recorded-model" in capsys.readouterr().err
