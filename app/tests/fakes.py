"""Stand-ins for the model and the engine bridge, for the app's tests."""
from __future__ import annotations

import json
import threading

from app.core.datasets import MODEL_ID

SECONDS = 1.5          # model seconds per reply
MS = 250.0             # bridge ms per query
GATE_TIMEOUT_S = 10.0  # how long a gated reply waits before the test counts as stuck
RESTART_KINDS = ("timeout", "crash", "memory")


# The next reply from a list: used in turn, the last one repeats.
def _take(replies: list):
    return replies.pop(0) if len(replies) > 1 else replies[0]


# Stands in for the model: the first script key found in a prompt picks the reply.
# A reply that is an exception is raised. With a gate, each call waits for it to be set first.
class FakeModel:
    def __init__(self, script: dict, config_id: str = MODEL_ID, gate: threading.Event | None = None):
        self.script = {k: list(v) if isinstance(v, list) else [v] for k, v in script.items()}
        self.config_id = config_id
        self.gate = gate
        self.called = threading.Event()  # set on the first call, so a test knows a reply is under way
        self.prompts: list[str] = []

    # The scripted reply and the model seconds, as the real model returns them.
    def generate(self, prompt: str) -> tuple[str, float]:
        self.prompts.append(prompt)
        self.called.set()
        if self.gate is not None and not self.gate.wait(GATE_TIMEOUT_S):
            raise TimeoutError("the test never opened the model's gate")

        for key, replies in self.script.items():
            if key in prompt:
                found = _take(replies)
                if isinstance(found, BaseException):
                    raise found
                return found, SECONDS
        raise AssertionError(f"no scripted reply for a prompt ending {prompt[-120:]!r}")


# Stands in for the engine bridge: canned replies by exact SQL, else the default.
# A failed timeout, crash or memory reply counts a restart, as the real bridge restarts then.
class FakeBridge:
    def __init__(self, replies: dict, default: dict | None = None):
        self.replies = {sql: list(r) if isinstance(r, list) else [r] for sql, r in replies.items()}
        self.default = default
        self.sent: list[str] = []
        self.restarts = 0
        self.started = False
        self.closed = False

    # Marks the bridge started and returns a ready line like the real one's.
    def start(self) -> dict:
        self.started = True
        return {"ready": True}

    # The canned reply with the bridge's id and ms; AssertionError for SQL with none.
    def query(self, sql: str) -> dict:
        self.sent.append(sql)
        if sql in self.replies:
            found = _take(self.replies[sql])
        elif self.default is not None:
            found = self.default
        else:
            raise AssertionError(f"no canned reply for {sql!r}")

        if found.get("ok") is not True and (found.get("error") or {}).get("kind") in RESTART_KINDS:
            self.restarts += 1
        return {"id": str(len(self.sent)), "ms": MS, **found}

    def close(self) -> None:
        self.closed = True


# The model's JSON reply text for one SQL query.
def reply(sql: str, grain: str = "one row", columns: list | None = None) -> str:
    return json.dumps({"sql": sql, "grain": grain, "columns": list(columns or [])})


# A good bridge reply; columns are (name, type) pairs.
def bridge_ok(columns, rows, profile: dict | None = None) -> dict:
    return {
        "ok": True, "columns": [{"name": n, "type": t} for n, t in columns], "rows": [list(r) for r in rows],
        "row_count": len(rows), "truncated": False, "profile": profile or {},
    }


# A failed bridge reply, shaped like the bridge's own.
def bridge_error(kind: str, message: str) -> dict:
    return {"ok": False, "error": {"kind": kind, "message": message}}
