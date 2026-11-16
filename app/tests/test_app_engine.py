"""Tests for app/core/engine.py and app/model/recorded_model.py."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from app.core import analyst_check, measured
from app.core import engine as engine_mod
from app.core.datasets import DATASETS, LABELS, MODEL_ID, Dataset, build_hint, default_paths, load_dataset, missing_files
from app.core.engine import Engine, EngineOptions, LockedBridge, default_bridge_factory
from app.model import recorded_model
from app.model.recorded_model import RecordedModel
from app.tests.fakes import FakeBridge, FakeModel, bridge_ok

an, pr = analyst_check.an, analyst_check.pr

# The questions the fixture run file answers.
QUESTION = "How many nations does each region hold?"
REPHRASED = "For every region, count its nations."
CANNOT_QUESTION = "Which nation has the most customers?"

FAKE_BRIDGE = analyst_check.REPO_ROOT / "bench/analyst/tests/fake_bridge.py"
LOADED = re.compile(r"Loaded in \d+\.\d s\.")
NO_MEASURED = "measured.json is missing or can't be read"
MODEL = {"path": "org/some-model-4bit", "revision": "abc123", "thinking": False, "max_tokens": 1024}
VERSIONS = {"mlx": "0.30.0", "mlx_lm": "0.32.2"}
LIBRARIES = {"mlx": "0.30.0", "mlx-lm": "0.32.2", "mlx-metal": "0.30.0", "transformers": "5.22.1",
             "tokenizers": "0.23.3", "jinja2": "3.1.6"}
WAIT_S = 10.0
START_ERROR = "analyst_bridge failed to load tables.tsv (exit code 1); see bridge_tpch.log"


# Writes text to path, making its folders.
def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# Fixture paths with the app folder and the engine source in small trees under tmp_path.
@pytest.fixture
def paths(fixture_paths, tmp_path):
    _write(tmp_path / "src/main.cpp", "int main() {}\n")
    return replace(fixture_paths, app_dir=tmp_path / "app", src=tmp_path / "src")


# Tmp trees holding every locked file, the cache code, a model_mlx.py with the tested config and a matching measured.json.
@pytest.fixture
def tested(tmp_path):
    p = replace(default_paths(), code=tmp_path / "code", results=tmp_path / "results", src=tmp_path / "src",
                app_dir=tmp_path / "app", app_results=tmp_path / "app_results")
    for folder, names in ((p.code, measured.FILES_CODE + measured.FILES_WATCHED), (p.results, measured.FILES_RESULTS),
                          (p.cache_code_dir, measured.CACHE_CODE)):
        for name in names:
            _write(folder / name, f"contents of {name}\n")
    _write(p.code / "model_mlx.py", f"MODEL_CONFIGS = {{{MODEL_ID!r}: {MODEL!r}}}\n")
    _write(p.src / "main.cpp", "int main() {}\n")

    record = {"files": measured.file_hashes(p), "engine_src_sha1": measured.tree_sha1(p.src),
              "model": {**MODEL, "id": MODEL_ID}, "rates": {"test_questions": 300}}
    _write(p.measured, json.dumps(record))
    return p, record


# A replay proof matching the tested setup.
def _proof(p) -> dict:
    return {"identical": 300, "total": 300, "differences": [], "uncached_differences": [], "versions": dict(VERSIONS),
            "library_versions": dict(LIBRARIES),
            "files": measured.file_hashes(p), "engine_src_sha1": measured.tree_sha1(p.src),
            "cache_code_sha1": measured.cache_code_sha1(p.cache_code_dir)}


# Builds engines and closes every one at the end of the test.
@pytest.fixture
def engines():
    made = []

    def make(p, options=EngineOptions("none"), **kwargs) -> Engine:
        made.append(Engine(p, options, **kwargs))
        return made[-1]

    yield make
    for e in made:
        e.close()


# A bridge factory that makes FakeBridges and keeps them, and the tables.tsv each got, by dataset.
class Bridges:
    def __init__(self, make=None):
        self.make = make or (lambda: FakeBridge({}, default=bridge_ok([("n", "INTEGER")], [[1]])))
        self.made: dict[str, FakeBridge] = {}
        self.tsvs: dict[str, Path] = {}

    def __call__(self, name: str, tables_tsv: Path):
        self.made[name] = self.make()
        self.tsvs[name] = tables_tsv
        return self.made[name]


# A FakeBridge whose start fails the way the analyst check's bridge does.
class FailingStart(FakeBridge):
    def start(self) -> dict:
        raise RuntimeError(START_ERROR)


# A FakeBridge whose close marks it closed, then fails.
class FailingClose(FakeBridge):
    def close(self) -> None:
        super().close()
        raise OSError("close failed")


# A FakeBridge whose start waits until it's closed, then fails, as the real bridge's does when stopped.
class StoppableStart(FakeBridge):
    def __init__(self):
        super().__init__({})
        self.entered = threading.Event()
        self.stopped = threading.Event()

    def start(self) -> dict:
        self.entered.set()
        self.stopped.wait(WAIT_S)
        if self.closed:
            raise RuntimeError("stopped while loading")
        return super().start()

    def close(self) -> None:
        super().close()
        self.stopped.set()


# A FakeBridge whose start ignores a close and succeeds once the test releases it.
class LateStart(FakeBridge):
    def __init__(self):
        super().__init__({})
        self.entered, self.release, self.close_called = threading.Event(), threading.Event(), threading.Event()
        self.close_calls = 0

    def start(self) -> dict:
        self.entered.set()
        self.release.wait(WAIT_S)
        return super().start()

    def close(self) -> None:
        super().close()
        self.close_calls += 1
        self.close_called.set()


# The bridge a factory made for a dataset, once its start thread has made it.
def _made(bridges: Bridges, name: str):
    deadline = time.monotonic() + WAIT_S
    while name not in bridges.made and time.monotonic() < deadline:
        time.sleep(0.001)
    return bridges.made[name]


# This process's child processes, by pid.
def _children() -> list[str]:
    return subprocess.run(["pgrep", "-P", str(os.getpid())], capture_output=True, text=True).stdout.split()


# The fixture run file's records, shown first.
def _records(p) -> tuple[dict, dict]:
    records = an.read_jsonl(p.run_file("dev"))
    shown = next(r for r in records if r["primary"]["final"] == "shown")
    cannot = next(r for r in records if r["primary"]["final"] == "cannot")
    return shown, cannot


# A run file of hand-made records.
def _run_file(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "run.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


# One recorded model reply, as run_attempts writes it.
def _attempt(prompt: str | None, raw: str, seconds: float) -> dict:
    return {"raw": raw, "kind": "sql", "sql": None, "reply": None, "seconds": seconds,
            "prompt_sha1": None if prompt is None else an.sha1_text(prompt)}


def test_fixture_run_file_is_6d_shape(paths):
    shown, cannot = _records(paths)
    ctx = load_dataset(paths, "tpch").context

    assert len(an.read_jsonl(paths.run_file("dev"))) == 2
    assert list(shown) == ["qid", "dataset", "split", "model", "prompt_sha1", "primary", "rephrase", "sqlite_cause",
                           "seconds_total", "bridge_restarts"]
    assert list(cannot) == [k for k in shown if k != "sqlite_cause"]
    for rec, question in ((shown, QUESTION), (cannot, CANNOT_QUESTION)):
        assert (rec["dataset"], rec["split"], rec["model"]) == ("tpch", "dev", MODEL_ID)
        assert rec["prompt_sha1"] == an.sha1_text(pr.build_sql_prompt(ctx.description, ctx.rules, ctx.examples, question))
    assert shown["rephrase"]["text"] == REPHRASED and shown["rephrase"]["final"] == "shown"
    assert shown["sqlite_cause"]["ok"] is True and len(shown["sqlite_cause"]["rows"]) == 5
    assert cannot["rephrase"]["final"] == "skipped"


def test_recorded_model_replays_by_prompt_sha1(paths):
    shown, cannot = _records(paths)
    ctx = load_dataset(paths, "tpch").context
    model = RecordedModel(paths.run_file("dev"))
    first = shown["primary"]["attempts"][0]
    second = shown["rephrase"]["attempts"][0]

    assert model.config_id == MODEL_ID  # from the records' model field
    assert model.generate(pr.build_sql_prompt(ctx.description, ctx.rules, ctx.examples, QUESTION)) == (first["raw"], first["seconds"])
    assert model.generate(pr.build_rephrase_prompt(QUESTION)) == (shown["rephrase"]["raw"], shown["rephrase"]["seconds"])
    assert model.generate(pr.build_sql_prompt(ctx.description, ctx.rules, ctx.examples, REPHRASED)) == (second["raw"], second["seconds"])

    refusal = cannot["primary"]["attempts"][0]
    assert model.generate(pr.build_sql_prompt(ctx.description, ctx.rules, ctx.examples, CANNOT_QUESTION)) == (refusal["raw"], refusal["seconds"])
    assert pr.parse_reply(refusal["raw"]).kind == "cannot"


def test_recorded_model_miss_raises(paths):
    model = RecordedModel(paths.run_file("dev"))
    with pytest.raises(KeyError) as e:
        model.generate("a prompt nobody recorded")
    assert e.value.args == ("no recorded reply for this prompt",)


def test_recorded_model_indexes_every_reply(tmp_path):
    skipped = {"raw": None, "text": None, "seconds": None, "prompt_sha1": None, "attempts": [], "final": "skipped"}
    rephrase = {**skipped, "raw": "In other words?", "seconds": 0.4, "prompt_sha1": an.sha1_text("rephrase"),
                "attempts": [_attempt("second", "second reply", 0.5)], "final": "shown"}
    records = [
        {"primary": {"attempts": [_attempt("first", "first reply", 1.0), _attempt("retry", "retry reply", 2.0),
                                  _attempt(None, "unhashed", 9.0)]}, "rephrase": rephrase},
        {"primary": {"attempts": [_attempt("first", "a later copy", 7.0)]}, "rephrase": skipped},
    ]
    model = RecordedModel(_run_file(tmp_path, records), config_id="other")

    assert model.config_id == "other"
    assert model.generate("first") == ("first reply", 1.0)  # the first record holding a sha1 wins
    assert model.generate("retry") == ("retry reply", 2.0)
    assert model.generate("rephrase") == ("In other words?", 0.4)
    assert model.generate("second") == ("second reply", 0.5)


def test_recorded_model_pace_sleeps_recorded_seconds(tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr(recorded_model, "sleep", slept.append)
    path = _run_file(tmp_path, [{"primary": {"attempts": [_attempt("first", "first reply", 1.25)]},
                                 "rephrase": {"prompt_sha1": None, "attempts": []}}])

    assert RecordedModel(path).generate("first") == ("first reply", 1.25)
    assert slept == []
    assert RecordedModel(path, pace=True).generate("first") == ("first reply", 1.25)
    assert slept == [1.25]
    with pytest.raises(KeyError):
        RecordedModel(path, pace=True).generate("unknown")
    assert slept == [1.25]


def test_recorded_model_id_from_the_records(tmp_path):
    rephrase = {"prompt_sha1": None, "attempts": []}
    other = _run_file(tmp_path, [{"primary": {"attempts": []}, "rephrase": rephrase},
                                 {"model": "qwen36-35b", "primary": {"attempts": []}, "rephrase": rephrase}])
    assert recorded_model.recorded_model_id(other) == "qwen36-35b"
    assert RecordedModel(other).config_id == "qwen36-35b"
    unnamed = _run_file(tmp_path, [{"primary": {"attempts": []}, "rephrase": rephrase}])
    assert recorded_model.recorded_model_id(unnamed) is None and RecordedModel(unnamed).config_id is None


def test_recorded_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="recorded run not found"):
        RecordedModel(tmp_path / "nope.jsonl")


# A bridge that takes 50 ms per query and records when each one ran.
class SlowBridge:
    def __init__(self):
        self.spans: list[tuple[float, float]] = []
        self.restarts = 3
        self.closed = False

    def query(self, sql: str) -> dict:
        start = time.monotonic()
        time.sleep(0.05)
        self.spans.append((start, time.monotonic()))
        return {"ok": True, "sql": sql}

    def close(self) -> None:
        self.closed = True


def test_locked_bridge_serializes():
    inner = SlowBridge()
    locked = LockedBridge(inner)
    threads = [threading.Thread(target=lambda: [locked.query("SELECT 1") for _ in range(2)]) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(WAIT_S)

    spans = sorted(inner.spans)
    assert len(spans) == 4 and locked.queries == 4
    assert all(later[0] >= earlier[1] for earlier, later in zip(spans, spans[1:]))
    assert locked.restarts == 3
    inner.restarts = 4
    assert locked.restarts == 4


def test_locked_bridge_close_skips_query_lock():
    entered, release = threading.Event(), threading.Event()
    inner = SlowBridge()

    def blocking(sql: str) -> dict:
        entered.set()
        release.wait(WAIT_S)
        return {"ok": True}

    inner.query = blocking
    locked = LockedBridge(inner)
    running = threading.Thread(target=locked.query, args=("SELECT 1",))
    running.start()
    try:
        assert entered.wait(WAIT_S)
        locked.close()  # would block until release if close took the query lock
        assert inner.closed and running.is_alive()
    finally:
        release.set()
        running.join(WAIT_S)


def test_engine_options_defaults_and_checks():
    options = EngineOptions("mlx")
    assert (options.recorded_run, options.pace, options.use_cache) == (None, False, True)
    with pytest.raises(ValueError, match="model_mode"):
        EngineOptions("gpt")
    with pytest.raises(ValueError, match="recorded_run"):
        EngineOptions("recorded")


def test_engine_missing_dataset_status(paths, engines):
    bridges = Bridges()
    engine = engines(paths, bridge_factory=bridges)
    assert {n: s["state"] for n, s in engine.status()["datasets"].items()} == {n: "loading" for n in DATASETS}

    engine.start_bridges()
    assert engine.wait_bridges(WAIT_S)
    datasets = engine.status()["datasets"]

    assert list(datasets) == list(DATASETS)
    reviews = datasets["reviews"]
    assert reviews["label"] == LABELS["reviews"] and reviews["state"] == "missing"
    assert reviews["detail"] == f"Missing: {', '.join(missing_files(paths, 'reviews'))}\n{build_hint(paths, 'reviews')}"
    assert str(paths.dataset_dir("reviews") / "tables.tsv") in reviews["detail"]

    tpch = datasets["tpch"]
    assert tpch["label"] == LABELS["tpch"] and tpch["state"] == "ready" and LOADED.fullmatch(tpch["detail"])
    assert list(bridges.made) == ["tpch"] and bridges.made["tpch"].started
    assert bridges.tsvs == {"tpch": paths.dataset_dir("tpch") / "tables.tsv"}
    assert paths.app_results.is_dir()  # the analyst check's bridge appends to a log there

    assert isinstance(engine.dataset("tpch"), Dataset) and engine.dataset("tpch").name == "tpch"
    bridge = engine.bridge("tpch")
    assert isinstance(bridge, LockedBridge) and bridge.query("SELECT 1")["ok"] is True
    for get in (engine.dataset, engine.bridge):
        with pytest.raises(KeyError):
            get("reviews")


def test_start_bridges_returns_at_once(paths, engines):
    gate = threading.Event()
    bridges = Bridges()

    def blocked(name, tables_tsv):
        gate.wait(WAIT_S)
        return bridges(name, tables_tsv)

    engine = engines(paths, bridge_factory=blocked)
    engine.start_bridges()
    try:
        assert engine.status()["datasets"]["tpch"]["state"] == "loading"
        with pytest.raises(KeyError):
            engine.bridge("tpch")
        assert not engine.wait_bridges(0.01)
    finally:
        gate.set()
    assert engine.wait_bridges(WAIT_S)
    assert engine.status()["datasets"]["tpch"]["state"] == "ready"


def test_engine_missing_bridge_binary(paths, engines):
    p = replace(paths, bridge=paths.app_dir / "no_bridge")
    engine = engines(p)
    engine.start_bridges()
    assert engine.wait_bridges(WAIT_S)
    datasets = engine.status()["datasets"]

    assert datasets["tpch"]["state"] == "missing"
    assert datasets["tpch"]["detail"] == build_hint(p, "tpch")
    assert datasets["tpch"]["detail"].startswith("Build the engine bridge:")
    assert datasets["reviews"]["detail"] == f"Missing: {', '.join(missing_files(p, 'reviews'))}\n{build_hint(p, 'reviews')}"


@pytest.mark.parametrize("where", ["start", "factory"])
def test_engine_bridge_failure_is_failed(paths, engines, where):
    def unbuildable():
        raise RuntimeError(START_ERROR)

    bridges = Bridges(lambda: FailingStart({}) if where == "start" else unbuildable())
    engine = engines(paths, bridge_factory=bridges)
    engine.start_bridges()
    assert engine.wait_bridges(WAIT_S)

    assert engine.status()["datasets"]["tpch"] == {"label": LABELS["tpch"], "state": "failed", "detail": START_ERROR}
    with pytest.raises(KeyError):
        engine.bridge("tpch")
    engine.close()
    if where == "start":
        assert bridges.made["tpch"].closed  # made but never started, still closed


def test_engine_close_closes_bridges(paths, engines):
    shutil.copytree(paths.dataset_dir("tpch"), paths.dataset_dir("reviews"))  # a second dataset the engine can load
    bridges = Bridges()
    engine = engines(paths, bridge_factory=bridges)
    engine.start_bridges()
    assert engine.wait_bridges(WAIT_S)
    assert {s["state"] for s in engine.status()["datasets"].values()} == {"ready"}

    engine.close()
    engine.close()
    assert sorted(bridges.made) == sorted(DATASETS)
    assert all(b.closed for b in bridges.made.values())


def test_engine_close_reports_a_failed_close(paths, engines, capsys):
    shutil.copytree(paths.dataset_dir("tpch"), paths.dataset_dir("reviews"))
    made = {}

    def factory(name, tables_tsv):
        made[name] = FailingClose({}) if name == "tpch" else FakeBridge({})
        return made[name]

    engine = engines(paths, bridge_factory=factory)
    engine.start_bridges()
    assert engine.wait_bridges(WAIT_S)

    engine.close()
    assert capsys.readouterr().err == f"Couldn't close the {LABELS['tpch']} bridge: close failed\n"
    assert made["reviews"].closed
    engine.close()
    assert capsys.readouterr().err == ""


def test_engine_close_during_slow_start(paths, engines):
    bridges = Bridges(StoppableStart)
    engine = engines(paths, bridge_factory=bridges)
    engine.start_bridges()
    assert _made(bridges, "tpch").entered.wait(WAIT_S)

    began = time.monotonic()
    engine.close()
    assert time.monotonic() - began < 2.0  # far under the fake's 10 s wait
    assert engine.wait_bridges(0)
    assert bridges.made["tpch"].closed
    assert engine.status()["datasets"]["tpch"]["state"] != "ready"
    with pytest.raises(KeyError):
        engine.bridge("tpch")


def test_engine_start_finishing_after_close_closes_its_bridge(paths, engines):
    bridges = Bridges(LateStart)
    engine = engines(paths, bridge_factory=bridges)
    engine.start_bridges()
    late = _made(bridges, "tpch")
    assert late.entered.wait(WAIT_S)

    closer = threading.Thread(target=engine.close)
    closer.start()
    try:
        assert late.close_called.wait(WAIT_S)  # close() reached the bridge while it was still starting
    finally:
        late.release.set()
    closer.join(WAIT_S)

    assert not closer.is_alive() and engine.wait_bridges(0)
    assert late.started and late.close_calls == 2  # started after the close, so its thread closed it again
    assert engine.status()["datasets"]["tpch"]["state"] == "loading"
    with pytest.raises(KeyError):
        engine.bridge("tpch")


def test_engine_close_stops_real_bridge_mid_load(paths, engines):
    made = []

    # A bridge process that never sends its ready line and exits once its stdin closes.
    def factory(name, tables_tsv):
        made.append(analyst_check.bc.Bridge(tables_tsv, command=[sys.executable, "-c", "import sys; sys.stdin.read()"]))
        return made[-1]

    assert _children() == []
    engine = engines(paths, bridge_factory=factory)
    engine.start_bridges()
    deadline = time.monotonic() + WAIT_S
    while not _children() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _children()

    began = time.monotonic()
    engine.close()
    assert time.monotonic() - began < 5.0
    assert engine.wait_bridges(0)
    assert _children() == []
    assert not made[0].db_file.exists()
    assert engine.status()["datasets"]["tpch"]["state"] != "ready"


def test_engine_close_removes_real_bridge_db_file(paths, engines):
    made = []

    def factory(name, tables_tsv):
        made.append(analyst_check.bc.Bridge(tables_tsv, command=[sys.executable, str(FAKE_BRIDGE)]))
        return made[-1]

    engine = engines(paths, bridge_factory=factory)
    engine.start_bridges()
    assert engine.wait_bridges(WAIT_S)
    assert engine.status()["datasets"]["tpch"]["state"] == "ready"
    assert engine.bridge("tpch").query("SELECT 1")["rows"] == [["SELECT 1"]]

    db_file = made[0].db_file
    db_file.touch()
    engine.close()
    assert not db_file.exists()


def test_default_bridge_factory(paths):
    tsv = paths.dataset_dir("tpch") / "tables.tsv"
    bridge = default_bridge_factory(paths)("tpch", tsv)
    try:
        assert isinstance(bridge, analyst_check.bc.Bridge)
        assert bridge.command == [str(paths.bridge)]
        assert bridge.log_path == paths.app_results / "bridge_tpch.log"
        assert bridge.tables_tsv == tsv
        assert bridge.ready is None  # built, never started
    finally:
        bridge.close()


def test_engine_none_mode_status(paths, engines):
    engine = engines(paths, EngineOptions("none"))
    assert engine.status()["model"] == {"state": "off", "mode": "none", "detail": ""}
    engine.load_model()
    assert engine.status()["model"]["state"] == "off" and engine.model is None


def test_engine_load_model_ready(paths, engines):
    fake = FakeModel({})
    engine = engines(paths, EngineOptions("mlx"), model_factory=lambda: fake)
    assert engine.status()["model"] == {"state": "loading", "mode": "mlx", "detail": ""}
    assert engine.model is None

    engine.load_model()
    model = engine.status()["model"]
    assert model["state"] == "ready" and LOADED.fullmatch(model["detail"])
    assert engine.model is fake


def test_engine_load_model_failed(paths, engines):
    def broken():
        raise RuntimeError("no weights in the cache")

    engine = engines(paths, EngineOptions("mlx"), model_factory=broken)
    engine.load_model()
    assert engine.status()["model"] == {"state": "failed", "mode": "mlx", "detail": "no weights in the cache"}
    assert engine.model is None


def test_engine_recorded_mode_default_model(paths, engines):
    engine = engines(paths, EngineOptions("recorded", recorded_run=paths.run_file("dev"), pace=True))
    assert engine.status()["model"]["state"] == "loading"
    engine.load_model()

    model = engine.status()["model"]
    assert (model["state"], model["mode"]) == ("ready", "recorded") and LOADED.fullmatch(model["detail"])
    assert isinstance(engine.model, RecordedModel) and engine.model.pace


def test_status_returns_fresh_dicts(paths, engines):
    engine = engines(paths)
    status = engine.status()
    status["datasets"]["tpch"]["state"] = "changed"
    status["model"]["state"] = "changed"
    status["lock"]["holds"] = True
    assert engine.status()["datasets"]["tpch"]["state"] == "loading"
    assert engine.status()["model"]["state"] == "off"
    assert engine.status()["lock"]["holds"] is False


def test_engine_lock_and_cache_in_status(tested, engines):
    p, record = tested
    record["files"]["results/subset_rules.md"] = "0" * 40
    _write(p.measured, json.dumps(record))
    engine = engines(p, EngineOptions("mlx"))

    assert engine.status()["lock"] == {"holds": False, "reason": "subset_rules.md changed"}
    assert engine.status()["cache"] == {"on": False, "reason": "the replay hasn't been run"}
    assert engine.measured == record


# The tested setup with a matching replay proof, the installed versions pinned to the proof's.
@pytest.fixture
def proven(tested, monkeypatch):
    p, record = tested
    monkeypatch.setattr(engine_mod, "mlx_versions", lambda: dict(VERSIONS))
    monkeypatch.setattr(engine_mod, "library_versions", lambda: dict(LIBRARIES))
    _write(p.cache_proof, json.dumps(_proof(p)))
    return p, record


# A FakeModel carrying the cached prefix lengths a CachedModel reports.
def _cached_fake(tokens: dict) -> FakeModel:
    fake = FakeModel({})
    fake.cached_tokens = dict(tokens)
    return fake


def test_engine_lock_holds_and_cache_on(proven, engines):
    p, _ = proven
    engine = engines(p, EngineOptions("mlx"), model_factory=lambda: _cached_fake({"tpch": 4096}))

    assert engine.status()["lock"] == {"holds": True, "reason": None}
    assert engine.status()["cache"] == {"on": True, "reason": "the replay matched 300 of 300"}  # while loading
    engine.load_model()
    assert engine.status()["cache"] == {"on": True, "reason": "the replay matched 300 of 300"}
    assert engine.lock == measured.Lock(True, None) and engine.cache.on


# Only the live model with the gate on reports the cache on; every other mode says why it's off.
@pytest.mark.parametrize("options, reason", [
    (EngineOptions("none"), "the model is off"),
    (EngineOptions("none", use_cache=False), "the model is off"),
    (EngineOptions("mlx", use_cache=False), "cache turned off with --no-cache"),
])
def test_engine_cache_off_per_mode(proven, engines, options, reason):
    p, _ = proven
    engine = engines(p, options, model_factory=lambda: _cached_fake({"tpch": 4096}))
    engine.load_model()
    assert engine.cache.on is options.use_cache  # the gate itself
    assert engine.status()["cache"] == {"on": False, "reason": reason}


def test_engine_cache_off_in_recorded_mode(proven, engines, tmp_path):
    p, _ = proven
    run = _run_file(tmp_path, [{"model": MODEL_ID, "primary": {"attempts": []}, "rephrase": {"prompt_sha1": None, "attempts": []}}])
    engine = engines(p, EngineOptions("recorded", recorded_run=run))
    engine.load_model()
    assert engine.cache.on
    assert engine.status()["cache"] == {"on": False, "reason": "answers come from a recorded run"}
    assert engine.status()["lock"] == {"holds": True, "reason": None}


def test_engine_cache_off_when_the_gate_is(tested, engines):
    p, _ = tested
    engine = engines(p, EngineOptions("mlx"), model_factory=lambda: _cached_fake({"tpch": 4096}))
    engine.load_model()
    assert engine.status()["cache"] == {"on": False, "reason": "the replay hasn't been run"}


def test_engine_cache_off_when_the_model_failed_or_cached_nothing(proven, engines):
    p, _ = proven

    def broken():
        raise RuntimeError("no weights")

    failed = engines(p, EngineOptions("mlx"), model_factory=broken)
    failed.load_model()
    assert failed.status()["cache"] == {"on": False, "reason": "the model didn't load"}

    for model in (_cached_fake({}), _cached_fake({"tpch": 0}), FakeModel({})):
        empty = engines(p, EngineOptions("mlx"), model_factory=lambda model=model: model)
        empty.load_model()
        assert empty.status()["cache"] == {"on": False, "reason": "no dataset's prompt start was cached"}


def test_engine_recorded_run_of_another_model_breaks_the_lock(proven, engines, tmp_path):
    p, _ = proven
    record = {"model": "qwen36-35b", "primary": {"attempts": []}, "rephrase": {"prompt_sha1": None, "attempts": []}}
    engine = engines(p, EngineOptions("recorded", recorded_run=_run_file(tmp_path, [record])))
    assert engine.lock == measured.Lock(False, "answers come from a recorded run of qwen36-35b")
    assert engine.status()["lock"] == {"holds": False, "reason": "answers come from a recorded run of qwen36-35b"}
    engine.load_model()
    assert engine.model.config_id == "qwen36-35b"


def test_engine_recorded_run_of_the_tested_model_keeps_the_lock(proven, engines, fixture_paths):
    p, _ = proven
    engine = engines(p, EngineOptions("recorded", recorded_run=fixture_paths.run_file("dev")))
    assert engine.lock == measured.Lock(True, None)
    engine.load_model()
    assert engine.model.config_id == MODEL_ID


# A model_mlx.py that won't parse, has no MODEL_CONFIGS, or holds other settings; its hash is re-recorded,
# so only the settings differ.
@pytest.mark.parametrize("text", ["MODEL_CONFIGS = [\n", "x = 1\n",
                                  f"MODEL_CONFIGS = {{{MODEL_ID!r}: {dict(MODEL, max_tokens=2048)!r}}}\n"])
def test_engine_model_settings_feed_the_lock(tested, engines, text):
    p, record = tested
    _write(p.code / "model_mlx.py", text)
    _write(p.measured, json.dumps({**record, "files": measured.file_hashes(p)}))
    assert engines(p).status()["lock"] == {"holds": False, "reason": "the model settings changed"}


def test_engine_lock_computed_once(tested, engines, monkeypatch):
    p, _ = tested
    calls = {"file_hashes": 0, "tree_sha1": 0}
    for name in calls:
        original = getattr(measured, name)

        def counted(*args, _name=name, _original=original):
            calls[_name] += 1
            return _original(*args)

        monkeypatch.setattr(measured, name, counted)

    engine = engines(p)
    after_init = dict(calls)
    assert after_init["file_hashes"] >= 1 and after_init["tree_sha1"] >= 1
    for _ in range(3):
        engine.status()
    assert calls == after_init


def test_engine_missing_measured_json(paths, engines):
    engine = engines(paths, EngineOptions("mlx"))
    assert not paths.measured.exists()
    assert engine.measured == {}
    assert engine.status()["lock"] == {"holds": False, "reason": NO_MEASURED}
    assert engine.status()["cache"] == {"on": False, "reason": NO_MEASURED}


# measured.json with all four keys, but one shaped so the lock or the gate can't read it.
@pytest.mark.parametrize("key, value", [("files", None), ("model", []), ("rates", [])])
@pytest.mark.parametrize("use_cache", [True, False])
def test_engine_wrong_shaped_measured_json(tested, engines, key, value, use_cache):
    p, record = tested
    _write(p.measured, json.dumps({**record, key: value}))
    _write(p.cache_proof, json.dumps(_proof(p)))  # so the gate reaches the rates
    engine = engines(p, EngineOptions("none", use_cache=use_cache))

    assert engine.measured == {}
    assert engine.lock == measured.Lock(False, NO_MEASURED)
    assert engine.cache == measured.CacheGate(False, NO_MEASURED if use_cache else "cache turned off with --no-cache")


@pytest.mark.parametrize("key", ["files", "engine_src_sha1", "model", "rates", None])
def test_engine_unusable_measured_json(tested, engines, key):
    p, record = tested
    if key is None:
        _write(p.measured, "{not json")
    else:
        _write(p.measured, json.dumps({k: v for k, v in record.items() if k != key}))
    engine = engines(p)

    assert engine.measured == {}
    assert engine.lock == measured.Lock(False, NO_MEASURED)
    assert engine.cache == measured.CacheGate(False, NO_MEASURED)


# Records how the engine built its model, standing in for CachedModel and the analyst check's MlxModel.
class MadeModel:
    def __init__(self, kind: str, config_id: str, contexts=None):
        self.kind, self.config_id, self.contexts = kind, config_id, contexts


# measured.json in place, with the lock and the gate forced to the given state.
@pytest.fixture
def swap(paths, monkeypatch):
    _write(paths.measured, json.dumps({"files": {}, "engine_src_sha1": "", "model": {}, "rates": {}}))
    made = []
    monkeypatch.setattr(engine_mod, "check_lock", lambda *a: measured.Lock(True, None))
    monkeypatch.setattr(engine_mod, "CachedModel", lambda config_id, contexts: made.append(MadeModel("cached", config_id, contexts)) or made[-1])
    stub = type("model_mlx", (), {"MlxModel": staticmethod(lambda config_id: made.append(MadeModel("plain", config_id)) or made[-1])})
    monkeypatch.setattr(analyst_check, "load_model_mlx", lambda: stub)

    def gate(on: bool):
        monkeypatch.setattr(engine_mod, "cache_gate", lambda *a: measured.CacheGate(on, "forced"))

    return paths, made, gate


def test_engine_gate_on_builds_cached_model(swap, engines):
    paths, made, gate = swap
    gate(True)
    engine = engines(paths, EngineOptions("mlx"))
    assert engine.cache.on and made == []

    engine.load_model()
    assert [m.kind for m in made] == ["cached"] and engine.model is made[0]
    assert made[0].config_id == MODEL_ID
    assert made[0].contexts == {"tpch": load_dataset(paths, "tpch").context}  # reviews is missing in the fixture
    assert engine.status()["model"]["state"] == "ready"
    assert "mlx" not in sys.modules and "mlx_lm" not in sys.modules


def test_engine_gate_off_or_no_cache_builds_plain_model(swap, engines):
    paths, made, gate = swap
    gate(False)
    engine = engines(paths, EngineOptions("mlx"))
    engine.load_model()
    gate(True)
    flagged = engines(paths, EngineOptions("mlx", use_cache=False))
    flagged.load_model()

    assert not engine.cache.on and not flagged.cache.on
    assert [(m.kind, m.config_id) for m in made] == [("plain", MODEL_ID), ("plain", MODEL_ID)]
    assert "mlx" not in sys.modules and "mlx_lm" not in sys.modules


def test_engine_cached_model_skips_a_dataset_that_fails_to_load(swap, engines, monkeypatch):
    paths, made, gate = swap
    gate(True)

    def broken(p, name):
        raise ValueError("bad manifest")

    monkeypatch.setattr(engine_mod, "load_dataset", broken)
    engine = engines(paths, EngineOptions("mlx"))
    engine.load_model()
    assert made[0].kind == "cached" and made[0].contexts == {}


def test_default_model_factory_needs_paths_for_the_cache():
    with pytest.raises(ValueError, match="needs paths"):
        engine_mod.default_model_factory(EngineOptions("mlx"), None, cache_on=True)
    assert engine_mod.default_model_factory(EngineOptions("mlx"), None, cache_on=False) is not None
    assert engine_mod.default_model_factory(EngineOptions("none"), None, cache_on=True) is None


def test_engine_recorded_mode_ignores_the_gate(swap, engines):
    paths, made, gate = swap
    gate(True)
    engine = engines(paths, EngineOptions("recorded", recorded_run=paths.run_file("dev")))
    engine.load_model()
    assert isinstance(engine.model, RecordedModel) and made == []
