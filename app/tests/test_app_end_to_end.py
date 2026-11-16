"""End-to-end tests: the real engine bridge behind the real runner and server."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import pytest

from app.__main__ import EXAMPLES, STATIC_DIR
from app.core.datasets import default_paths
from app.core.engine import Engine, EngineOptions
from app.server.jobs import ENDED, JobRunner
from app.server.server import make_server
from app.tests.test_app_flows import LEFT_SQL, QUESTION
from app.tests.test_app_server import TOKEN, Client

BRIDGE = default_paths().bridge
needs_bridge = pytest.mark.skipif(not BRIDGE.exists(), reason="analyst_bridge not built")

WAIT_S = 30.0
NATIONS_SQL = ("SELECT r_name, COUNT(*) AS nations FROM nation JOIN region ON n_regionkey = r_regionkey "
               "GROUP BY r_name ORDER BY r_name")
BAD_SQL = "SELECT nope FROM nation"


# The bridge processes this test process started that are still running, one "pid command" line each.
def bridge_children() -> set[str]:
    found = subprocess.run(["pgrep", "-fl", "-P", str(os.getpid()), "analyst_bridge"], capture_output=True, text=True)
    return set(found.stdout.split("\n")) - {""}


# Bridge db files this test process has in the temp folder.
def leftover_db_files() -> set[Path]:
    return set(Path(tempfile.gettempdir()).glob(f"analyst_bridge_{os.getpid()}_*.db"))


# Starts the app the way python -m app does, on a free port with the given model, and returns
# its client and engine. At the end it shuts down in the app's order: server, runner, engine,
# then checks it left no bridge or db file beyond those already there when it began.
@pytest.fixture
def start(fixture_paths):
    made = []
    children_before, db_files_before = bridge_children(), leftover_db_files()

    def make(options: EngineOptions) -> tuple[Client, Engine]:
        engine = Engine(fixture_paths, options)
        runner = JobRunner(engine)
        made.append((engine, runner, None, None))
        engine.start_bridges()
        runner.start()
        server = make_server(runner, engine, TOKEN, 0, STATIC_DIR, json.loads(EXAMPLES.read_text(encoding="utf-8")))
        thread = threading.Thread(target=server.serve_forever, args=(0.01,), name="test-http", daemon=True)
        thread.start()
        made[-1] = (engine, runner, server, thread)

        assert engine.wait_bridges(WAIT_S), "the bridges never finished starting"
        assert engine.status()["datasets"]["tpch"]["state"] == "ready", engine.status()["datasets"]["tpch"]
        return Client(server.server_port), engine

    yield make
    for engine, runner, server, thread in made:
        try:
            try:
                if server is not None:
                    server.shutdown()
                    server.server_close()
                    thread.join(WAIT_S)
            finally:
                runner.shutdown()
        finally:
            engine.close()
    assert bridge_children() - children_before == set()
    assert leftover_db_files() - db_files_before == set()


# Typed SQL on TPC-H through the API, waited on over the long poll.
def run_sql(client: Client, sql: str) -> dict:
    submitted = client.post("/api/sql", {"dataset": "tpch", "sql": sql})
    assert submitted.status == 200, submitted.body
    return client.finish(submitted.json()["id"])


# Waits until the model has loaded, as the page sees it.
def wait_model(client: Client) -> None:
    deadline = time.monotonic() + WAIT_S
    while (state := client.get("/api/status").json()["model"]["state"]) == "loading":
        assert time.monotonic() < deadline, "the model never loaded"
        time.sleep(0.01)
    assert state == "ready"


@needs_bridge
def test_real_sql_round_trip(start):
    client, _ = start(EngineOptions("none"))
    done = run_sql(client, NATIONS_SQL)

    assert done["stage"] == "done" and done["error"] is None
    first = done["first"]
    assert first["row_count"] == 5 and len(first["rows"]) == 5
    assert first["rows"][0] == ["AFRICA", "5"]
    assert first["meta"].startswith("5 rows")
    assert first["plan"][0]["name"] == "Sort"


@needs_bridge
def test_real_left_join_not_run(start):
    client, engine = start(EngineOptions("none"))
    bridge = engine.bridge("tpch")
    before = bridge.queries
    done = run_sql(client, LEFT_SQL)

    assert done["stage"] == "done"
    assert done["verdict"]["kind"] == "not_run" and done["verdict"]["text"].startswith("Not run.")
    assert done["first"] is None
    assert bridge.queries == before


@needs_bridge
def test_real_engine_error(start):
    client, _ = start(EngineOptions("none"))
    failed = run_sql(client, BAD_SQL)

    assert failed["stage"] == "error" and failed["verdict"] is None
    assert failed["error"]["message"].startswith("Run error:")


# Pace makes each recorded reply take its 1.5 s, so every stage lasts long enough for the poll to see it.
# That 1.5 s per stage is the margin against a loopback poll, which takes a few ms.
@needs_bridge
def test_recorded_question_round_trip(start, fixture_paths):
    client, _ = start(EngineOptions("recorded", recorded_run=fixture_paths.run_file("dev"), pace=True))
    wait_model(client)

    submitted = client.post("/api/ask", {"dataset": "tpch", "question": QUESTION})
    assert submitted.status == 200, submitted.body
    snapshot = submitted.json()
    stages = []
    deadline = time.monotonic() + WAIT_S
    while snapshot["stage"] not in ENDED:
        assert time.monotonic() < deadline, f"stuck at {snapshot['stage']}"
        snapshot = client.get(f"/api/jobs/{snapshot['id']}?after={snapshot['version']}").json()
        if not stages or stages[-1] != snapshot["stage"]:
            stages.append(snapshot["stage"])

    assert stages == ["writing", "first", "done"]
    assert snapshot["checks"][0]["name"] == "second_phrasing"
    assert snapshot["second"] is not None
