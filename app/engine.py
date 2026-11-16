from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app import sixd
from app.datasets import DATASETS, LABELS, MODEL_ID, AppPaths, Dataset, build_hint, load_dataset, missing_files
from app.measured import CacheGate, Lock, cache_code_sha1, cache_gate, check_lock, library_versions, mlx_versions, read_json
from app.model_cache import CachedModel
from app.recorded_model import RecordedModel, recorded_model_id
from app.schema import dataset_schema

MODEL_MODES = ("mlx", "recorded", "none")
MEASURED_KEYS = ("files", "engine_src_sha1", "model", "rates")
NO_MEASURED = "measured.json is missing or can't be read"
NO_CACHE = "cache turned off with --no-cache"
MODEL_OFF = "the model is off"
RECORDED = "answers come from a recorded run"
MODEL_FAILED = "the model didn't load"
NOTHING_CACHED = "no dataset's prompt start was cached"


# One bridge shared by both workers: one query at a time, counted.
class LockedBridge:
    def __init__(self, bridge):
        self._bridge = bridge
        self._lock = threading.Lock()
        self.queries = 0

    def query(self, sql: str) -> dict:
        with self._lock:
            self.queries += 1
            return self._bridge.query(sql)

    # The same, unless stop() says no once the bridge is free: then nothing is sent and the answer is None.
    def query_unless(self, sql: str, stop: Callable[[], bool]) -> dict | None:
        with self._lock:
            if stop():
                return None
            self.queries += 1
            return self._bridge.query(sql)

    @property
    def restarts(self) -> int:
        return self._bridge.restarts

    # Skips the query lock, so quitting doesn't wait for the query to finish. The bridge's
    # own close can still give its process up to 10 s to exit.
    def close(self) -> None:
        self._bridge.close()


# How the app was started: which model, and whether the prompt cache may be used.
@dataclass(frozen=True)
class EngineOptions:
    model_mode: str  # "mlx", "recorded" or "none"
    recorded_run: Path | None = None
    pace: bool = False
    use_cache: bool = True

    def __post_init__(self):
        if self.model_mode not in MODEL_MODES:
            raise ValueError(f"model_mode must be one of {', '.join(MODEL_MODES)}, got {self.model_mode!r}")
        if self.model_mode == "recorded" and self.recorded_run is None:
            raise ValueError("the recorded model needs a recorded_run")


# The real bridge on paths.bridge, its stderr appended to a log per dataset.
def default_bridge_factory(paths: AppPaths) -> Callable:
    def make(name: str, tables_tsv: Path):
        return sixd.bc.Bridge(tables_tsv, command=[str(paths.bridge)], log_path=paths.app_results / f"bridge_{name}.log")
    return make


# Each dataset's prompt context, for the datasets whose files are all there.
# One that fails to load is left out, so its prompts run uncached.
def _contexts(paths: AppPaths) -> dict[str, object]:
    contexts = {}
    for name in DATASETS:
        if missing_files(paths, name):
            continue
        try:
            contexts[name] = load_dataset(paths, name).context
        except Exception:
            continue
    return contexts


# What builds the model for a mode; None when there is no model.
# In mlx mode the prompt cache is used only when the cache gate is on, which needs paths.
def default_model_factory(options: EngineOptions, paths: AppPaths | None = None, cache_on: bool = False) -> Callable | None:
    if options.model_mode == "mlx" and cache_on and paths is None:
        raise ValueError("the cached model needs paths")
    if options.model_mode == "mlx" and cache_on:
        return lambda: CachedModel(MODEL_ID, _contexts(paths))
    if options.model_mode == "mlx":
        return lambda: sixd.load_model_mlx().MlxModel(MODEL_ID)
    if options.model_mode == "recorded":
        return lambda: RecordedModel(options.recorded_run, pace=options.pace)
    return None


# The tested model's settings from model_mlx.py; None when the file can't be read.
def _model_config(paths: AppPaths) -> dict | None:
    try:
        return sixd.fz.read_model_configs(paths.code / "model_mlx.py").get(MODEL_ID)
    except (OSError, ValueError, SyntaxError):
        return None


# An exception's message, or its type when it has none.
def _message(e: Exception) -> str:
    return str(e) or type(e).__name__


# The model a recorded run was made with; None when it names none or can't be read.
def _recorded_id(options: EngineOptions) -> str | None:
    try:
        return recorded_model_id(options.recorded_run)
    except Exception:
        return None


# Wall seconds since start, as the status shows them.
def _loaded_in(start: float) -> str:
    return f"Loaded in {time.monotonic() - start:.1f} s."


# Closes one bridge; a failure is reported on stderr, never raised, so the others still close.
def _close_bridge(name: str, bridge) -> None:
    try:
        bridge.close()
    except Exception as e:
        print(f"Couldn't close the {LABELS[name]} bridge: {_message(e)}", file=sys.stderr)


# Owns the two dataset bridges, the model, and the measured lock and cache gate.
class Engine:
    def __init__(self, paths: AppPaths, options: EngineOptions, bridge_factory=None, model_factory=None):
        self.paths = paths
        self.options = options
        self.model = None
        self._check_binary = bridge_factory is None
        self._bridge_factory = bridge_factory or default_bridge_factory(paths)

        self._state_lock = threading.Lock()
        self._datasets = {name: {"state": "loading", "detail": ""} for name in DATASETS}
        self._model_state = {"state": "off" if options.model_mode == "none" else "loading", "detail": ""}
        self._loaded: dict[str, Dataset] = {}
        self._bridges: dict[str, LockedBridge] = {}
        self._made: list[tuple[str, object]] = []  # (dataset, bridge) for every bridge built, started or not
        self._threads: list[threading.Thread] = []
        self._closed = False
        self._quitting = False  # set as quitting begins, before anything is closed
        self._schema_lock = threading.Lock()
        self._schemas: dict[str, dict] = {}  # each dataset's tables, read from its files once

        # Each hashes the locked files and walks src/, so once per launch.
        self.measured, self.lock, self.cache = self._check_measured()
        self._model_factory = model_factory or default_model_factory(options, paths, self.cache.on)

    # The tested record, the lock and the cache gate; both closed when measured.json is missing,
    # unreadable, or shaped so the checks can't read it.
    def _check_measured(self) -> tuple[dict, Lock, CacheGate]:
        no_cache = None if self.options.use_cache else CacheGate(False, NO_CACHE)
        unusable = ({}, Lock(False, NO_MEASURED), no_cache or CacheGate(False, NO_MEASURED))
        record = read_json(self.paths.measured)
        if record is None or any(key not in record for key in MEASURED_KEYS):
            return unusable

        # The rates are the tested model's, so a recorded run of another model doesn't get them.
        recorded = _recorded_id(self.options) if self.options.model_mode == "recorded" else MODEL_ID
        other_model = None if recorded == MODEL_ID else Lock(False, f"{RECORDED} of {recorded or 'an unnamed model'}")

        model_config = _model_config(self.paths)
        proof = read_json(self.paths.cache_proof)
        versions = mlx_versions()
        # The gate runs even with --no-cache, so the flag never changes which records count as readable.
        try:
            lock = check_lock(self.paths, record, model_config, proof, versions)
            cache = cache_gate(self.paths, record, proof, library_versions(), cache_code_sha1(self.paths.app_dir), model_config)
        except (KeyError, TypeError, AttributeError, ValueError):
            return unusable
        return record, other_model or lock, no_cache or cache

    # Starts both datasets' bridges, one thread each, and returns at once.
    # The list is kept only once every thread runs, so a waiter never joins one that hasn't started.
    def start_bridges(self) -> None:
        with self._state_lock:
            if self._threads or self._closed:
                return
            threads = [threading.Thread(target=self._start_bridge, args=(name,), name=f"bridge-{name}", daemon=True) for name in DATASETS]
            for thread in threads:
                thread.start()
            self._threads = threads

    # Waits for the bridge threads; True once all have finished.
    def wait_bridges(self, timeout_s: float | None = None) -> bool:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        for thread in self._threads:
            thread.join(None if deadline is None else max(0.0, deadline - time.monotonic()))
        return not any(thread.is_alive() for thread in self._threads)

    # One dataset: missing files (or no bridge binary), else load it and start its bridge.
    # Once close() has begun, the bridge isn't started, or is closed here, and never marked ready.
    def _start_bridge(self, name: str) -> None:
        missing = missing_files(self.paths, name)
        if missing or (self._check_binary and not self.paths.bridge.is_file()):
            hint = build_hint(self.paths, name)
            self._set_dataset(name, "missing", f"Missing: {', '.join(missing)}\n{hint}" if missing else hint)
            return

        start = time.monotonic()
        bridge, failed = None, None
        try:
            self.paths.app_results.mkdir(parents=True, exist_ok=True)  # the bridge appends its log here
            ds = load_dataset(self.paths, name)
            bridge = self._bridge_factory(name, ds.tables_tsv)
            with self._state_lock:
                self._made.append((name, bridge))
                closing = self._closed
            if not closing:
                bridge.start()
        except Exception as e:
            failed = _message(e)

        with self._state_lock:
            closing = self._closed
            if not closing and failed is not None:
                self._datasets[name] = {"state": "failed", "detail": failed}
            elif not closing:
                self._loaded[name] = ds
                self._bridges[name] = LockedBridge(bridge)
                self._datasets[name] = {"state": "ready", "detail": _loaded_in(start)}

        # close() may have reached this bridge before its process existed, so close it again.
        if closing and bridge is not None:
            _close_bridge(name, bridge)

    def _set_dataset(self, name: str, state: str, detail: str) -> None:
        with self._state_lock:
            self._datasets[name] = {"state": state, "detail": detail}

    # Builds the model on the calling thread; nothing to do when started without one.
    def load_model(self) -> None:
        if self._model_factory is None:
            return

        start = time.monotonic()
        try:
            model = self._model_factory()
        except Exception as e:
            with self._state_lock:
                self._model_state = {"state": "failed", "detail": _message(e)}
            return

        with self._state_lock:
            self.model = model
            self._model_state = {"state": "ready", "detail": _loaded_in(start)}

    # A ready dataset; KeyError until it is.
    def dataset(self, name: str) -> Dataset:
        with self._state_lock:
            return self._loaded[name]

    # A ready dataset's bridge; KeyError until it is.
    def bridge(self, name: str) -> LockedBridge:
        with self._state_lock:
            return self._bridges[name]

    # Marks the start of quitting, so no preview reaches a bridge about to close. close() marks it too.
    def begin_quit(self) -> None:
        with self._state_lock:
            self._quitting = True

    @property
    def quitting(self) -> bool:
        with self._state_lock:
            return self._quitting

    # A table preview's query, sent once the bridge is free; None when quitting began before then.
    def preview_query(self, name: str, sql: str) -> dict | None:
        return self.bridge(name).query_unless(sql, lambda: self.quitting)

    # The prompt cache as the status shows it: on only for the live model with the gate on,
    # once it has loaded with a cached prefix (or while it loads). Called with the state lock held.
    def _cache_state(self) -> CacheGate:
        if self.options.model_mode == "none":
            return CacheGate(False, MODEL_OFF)
        if self.options.model_mode == "recorded":
            return CacheGate(False, RECORDED)
        if not self.cache.on:
            return self.cache
        if self._model_state["state"] == "failed":
            return CacheGate(False, MODEL_FAILED)
        if self._model_state["state"] == "ready" and not any(getattr(self.model, "cached_tokens", {}).values()):
            return CacheGate(False, NOTHING_CACHED)
        return self.cache

    # What's loaded, the model, the lock and the cache, as fresh dicts.
    def status(self) -> dict:
        with self._state_lock:
            cache = self._cache_state()
            return {
                "datasets": {name: {"label": LABELS[name], **self._datasets[name]} for name in DATASETS},
                "model": {
                    "state": self._model_state["state"], 
                    "mode": self.options.model_mode,
                    "detail": self._model_state["detail"]
                },
                "lock": {
                    "holds": self.lock.holds,
                    "reason": self.lock.reason
                },
                "cache": {
                    "on": cache.on,
                    "reason": cache.reason
                },
            }

    # Both datasets' tables and columns, read from their files once, each with its state as the status has it.
    def schema(self) -> dict:
        states = self.status()["datasets"]
        with self._schema_lock:
            for name in DATASETS:
                if name not in self._schemas:
                    self._schemas[name] = dataset_schema(self.paths, name)
            read = dict(self._schemas)
        return {"datasets": {name: {**states[name], **read[name]} for name in DATASETS}}

    # Closes every bridge built so far, even one still loading (its start then fails),
    # then waits for the start threads. Safe to call twice.
    def close(self) -> None:
        with self._state_lock:
            self._closed = self._quitting = True
            made, self._made = self._made, []
        for name, bridge in made:
            _close_bridge(name, bridge)
        self.wait_bridges()
