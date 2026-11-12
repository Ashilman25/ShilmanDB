#Talk to build/bench/analyst_bridge
from __future__ import annotations

import importlib.util
import itertools
import json
import os
import re
import selectors
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# Load an_common from next to this file, reusing it if it's already loaded.
def _load_common():
    path = Path(__file__).resolve().parent / "an_common.py"
    loaded = sys.modules.get("an_common")
    if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() == path:
        return loaded

    spec = importlib.util.spec_from_file_location("an_common", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"can't load {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["an_common"] = module
    spec.loader.exec_module(module)
    return module


an = _load_common()
BRIDGE, TIMEOUT_S = an.BRIDGE, an.TIMEOUT_S

LOAD_TIMEOUT_S = 900.0
READ_CHUNK = 1 << 20  # bytes per read; a 10,000-row reply can be megabytes
CRASH_WAIT_S = 1.0    # time a dying bridge gets to report its exit code
CLOSE_WAIT_S = 10.0   # time a closing bridge gets to write out and remove its db file

_db_numbers = itertools.count(1)
_STRING_BREAKS = str.maketrans("\r\n\t", "   ")

# One piece of SQL: a 'string' (an escaped '' splits one in two, harmlessly),
# a -- comment to its line break, whitespace, a lone "-", or other text.
_SQL_PIECE = re.compile(r"'[^']*'?|--[^\r\n]*|\s+|-|[^'\s-]+")


class _Timeout(Exception):
    pass


class _Crash(Exception):
    pass


# One line of SQL: outside 'strings', drop -- comments and collapse whitespace to one space;
# inside, turn tabs and line breaks into spaces. Then strip and drop one trailing ";".
def one_line(sql: str) -> str:
    pieces: list[str] = []
    for piece in _SQL_PIECE.findall(sql):
        if piece.startswith("'"):
            pieces.append(piece.translate(_STRING_BREAKS))
        elif piece.startswith("--") or piece.isspace():
            if not pieces or pieces[-1] != " ":
                pieces.append(" ")
        else:
            pieces.append(piece)

    s = "".join(pieces).strip()
    if s.endswith(";"):
        s = s[:-1].rstrip()
    return s


#One analyst_bridge process over a tables.tsv, restarted when a query hangs or crashes it
class Bridge:
    def __init__(self, tables_tsv: Path, command: list[str] | None = None, timeout_s: float = TIMEOUT_S, load_timeout_s: float = LOAD_TIMEOUT_S, log_path: Path | None = None):
        self.tables_tsv = Path(tables_tsv)
        self.command = list(command) if command is not None else [str(BRIDGE)]
        self.timeout_s = timeout_s
        self.load_timeout_s = load_timeout_s
        self.log_path = Path(log_path) if log_path is not None else None
        self.db_file = Path(tempfile.gettempdir()) / f"analyst_bridge_{os.getpid()}_{next(_db_numbers)}.db"
        self.restarts = 0
        self.ready: dict | None = None  # the bridge's ready line, from the latest start

        self._proc: subprocess.Popen | None = None
        self._buf = bytearray()
        self._next_id = 1

    def __enter__(self) -> Bridge:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # Start the bridge and wait for its ready line; RuntimeError if it can't load.
    def start(self) -> dict:
        if self._proc is not None:
            raise RuntimeError("bridge already started")

        stderr = open(self.log_path, "ab") if self.log_path is not None else subprocess.DEVNULL
        try:
            self._proc = subprocess.Popen(
                [*self.command, "--tables", str(self.tables_tsv), "--db-file", str(self.db_file)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr
            )
        finally:
            if stderr is not subprocess.DEVNULL:
                stderr.close()
        self._buf.clear()

        try:
            ready = json.loads(self._read_line(time.monotonic() + self.load_timeout_s).decode("utf-8", "replace"))
        except (_Timeout, _Crash, ValueError) as e:
            code = self._stop(CRASH_WAIT_S)
            why = "timed out" if isinstance(e, _Timeout) else f"exit code {code}"
            see = f"; see {self.log_path}" if self.log_path is not None else ""
            raise RuntimeError(f"analyst_bridge failed to load {self.tables_tsv} ({why}){see}") from None

        if not isinstance(ready, dict) or ready.get("ready") is not True:
            self._stop(CRASH_WAIT_S)
            raise RuntimeError(f"analyst_bridge sent a bad ready line: {ready!r}")
        self.ready = ready
        return ready

    # Run one query. A timeout or crash restarts the bridge and returns a failed reply.
    def query(self, sql: str) -> dict:
        if self._proc is None:
            raise RuntimeError("bridge isn't running")

        rid = str(self._next_id)
        self._next_id += 1
        sent = time.monotonic()
        try:
            self._proc.stdin.write(f"{rid}\t{one_line(sql)}\n".encode("utf-8"))
            self._proc.stdin.flush()
            line = self._read_line(sent + self.timeout_s)
        except _Timeout:
            ms = (time.monotonic() - sent) * 1000
            self._restart(grace_s=0)
            return _failure(rid, "timeout", f"over {self.timeout_s:g} s", ms)
        except (_Crash, OSError):
            ms = (time.monotonic() - sent) * 1000
            code = self._restart(grace_s=CRASH_WAIT_S)
            return _failure(rid, "crash", f"bridge died (exit code {code})", ms)

        try:
            reply = json.loads(line.decode("utf-8", "replace"))
        except ValueError:
            raise RuntimeError(f"analyst_bridge sent a line that isn't JSON: {line[:200]!r}") from None
        if not isinstance(reply, dict) or reply.get("id") != rid:
            raise RuntimeError(f"expected a reply to request {rid}, got {str(reply)[:200]}")
        return reply

    # Close stdin so the bridge exits and removes its db file; kill it if slow.
    def close(self) -> None:
        self._stop(CLOSE_WAIT_S)
        self.db_file.unlink(missing_ok=True)

    # Read one line by the deadline, from my own buffer so select never misses data.
    def _read_line(self, deadline: float) -> bytes:
        fd = self._proc.stdout.fileno()
        searched = 0
        with selectors.DefaultSelector() as sel:
            sel.register(fd, selectors.EVENT_READ)
            
            while (end := self._buf.find(b"\n", searched)) < 0:
                searched = len(self._buf)
                left = deadline - time.monotonic()
                if left <= 0:
                    raise _Timeout
                if sel.select(left):
                    chunk = os.read(fd, READ_CHUNK)
                    if not chunk:
                        raise _Crash
                    self._buf += chunk

        line = bytes(self._buf[:end])
        del self._buf[:end + 1]
        return line

    # Stop the bridge and start a new one. Returns the old one's exit code.
    def _restart(self, grace_s: float) -> int | None:
        code = self._stop(grace_s)
        self.restarts += 1
        self.start()
        return code

    # Close stdin, wait up to grace_s for an exit, then kill. Removes the db file.
    def _stop(self, grace_s: float) -> int | None:
        proc, self._proc = self._proc, None
        if proc is None:
            return None

        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            code = proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = proc.wait()
            
        proc.stdout.close()
        self.db_file.unlink(missing_ok=True)
        return code


# A failed reply shaped like the bridge's own.
def _failure(rid: str, kind: str, message: str, ms: float) -> dict:
    return {"id": rid, "ok": False, "error": {"kind": kind, "message": message}, "ms": ms}
