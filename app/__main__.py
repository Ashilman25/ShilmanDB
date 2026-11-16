from __future__ import annotations

import argparse
import json
import secrets
import signal
import socket
import sys
import threading
import webbrowser
from collections.abc import Callable
from pathlib import Path

from app.datasets import AppPaths, default_paths
from app.engine import Engine, EngineOptions
from app.jobs import JobRunner
from app.server import HOST, make_server

DEFAULT_PORT = 8765
PORT_TRIES = 20
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
EXAMPLES = APP_DIR / "examples.json"
QUIT_SIGNALS = (signal.SIGINT, signal.SIGTERM)


# Whether the server could bind this port now, asked the way it binds (address reuse on).
def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((HOST, port))
        except OSError:
            return False
    return True


# The first free port from start; with fixed, start itself or OSError.
def pick_port(start: int, fixed: bool, tries: int = PORT_TRIES) -> int:
    for port in [start] if fixed else range(start, min(start + tries, 65536)):
        if _port_free(port):
            return port
    if fixed:
        raise OSError(f"Port {start} is in use.")
    raise OSError(f"No free port from {start} to {start + tries - 1}.")


# A port number from the command line.
def _port(text: str) -> int:
    port = int(text)
    if not 0 < port < 65536:
        raise argparse.ArgumentTypeError("must be from 1 to 65535")
    return port


# The flags; a bad combination or a missing recorded run exits 2 with argparse's usage line.
def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m app", description="ShilmanDB's analyst app, on this computer only.")
    parser.add_argument("--port", type=_port, help=f"the port to serve on (default {DEFAULT_PORT}, or the next free one)")
    parser.add_argument("--no-browser", action="store_true", help="print the link without opening a browser")
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--no-model", action="store_true", help="start without the model: the SQL runner only")
    model.add_argument("--recorded-model", type=Path, metavar="PATH", help="answer from a saved run file instead of the model")
    parser.add_argument("--recorded-pace", action="store_true", help="with --recorded-model, take as long as each recorded reply did")
    parser.add_argument("--no-cache", action="store_true", help="don't use the prompt cache")
    args = parser.parse_args(argv)

    if args.recorded_model is not None and not args.recorded_model.is_file():
        parser.error(f"--recorded-model: no file at {args.recorded_model}")
    if args.recorded_pace and args.recorded_model is None:
        parser.error("--recorded-pace needs --recorded-model")
    return args


# Which model the engine uses, and whether the prompt cache may.
def _options(args: argparse.Namespace) -> EngineOptions:
    use_cache = not args.no_cache
    if args.no_model:
        return EngineOptions("none", use_cache=use_cache)
    if args.recorded_model is not None:
        return EngineOptions("recorded", recorded_run=args.recorded_model, pace=args.recorded_pace, use_cache=use_cache)
    return EngineOptions("mlx", use_cache=use_cache)


# Ctrl-C and SIGTERM set quit instead of raising, and put the default handlers back, so a second
# press ends the app at once. Returns what puts the old handlers back. Main thread only.
def _on_signals(quit_event: threading.Event, install=signal.signal) -> Callable[[], None]:
    if threading.current_thread() is not threading.main_thread():
        return lambda: None

    def on_quit(*_) -> None:
        quit_event.set()
        for sig in QUIT_SIGNALS:
            install(sig, signal.SIG_DFL)

    previous = {sig: install(sig, on_quit) for sig in QUIT_SIGNALS}

    def restore() -> None:
        for sig, handler in previous.items():
            install(sig, handler)
    return restore


# The launch line about the prompt cache; None unless the live model is in use.
def cache_line(status: dict) -> str | None:
    if status["model"]["mode"] != "mlx":
        return None
    cache = status["cache"]
    return "Prompt cache: on" if cache["on"] else f"Prompt cache: off ({cache['reason']})"


# Starts the app, prints its link and serves until a signal (or stop, in tests); then closes
# the server, the runner and the engine, in that order. Returns the exit code.
def main(argv: list[str] | None = None, stop: threading.Event | None = None, paths: AppPaths | None = None) -> int:
    args = _parse_args(argv)
    try:
        port = pick_port(DEFAULT_PORT if args.port is None else args.port, fixed=args.port is not None)
    except OSError as e:
        print(e, file=sys.stderr)
        return 2

    engine = Engine(paths or default_paths(), _options(args))
    runner = JobRunner(engine)
    quit_event = stop or threading.Event()
    restore = _on_signals(quit_event)
    server = serving = None
    try:
        engine.start_bridges()
        runner.start()
        token = secrets.token_hex(16)
        examples = json.loads(EXAMPLES.read_text(encoding="utf-8"))
        try:
            server = make_server(runner, engine, token, port, STATIC_DIR, examples)
        except OSError:
            print(f"Port {port} is in use.", file=sys.stderr)
            return 2

        url = f"http://{HOST}:{server.server_port}/?token={token}"
        print(f"ShilmanDB app: {url}", flush=True)
        line = cache_line(engine.status())
        if line is not None:
            print(line, flush=True)
        print("Press Ctrl-C to quit.", flush=True)
        if not args.no_browser:
            webbrowser.open(url)

        serving = threading.Thread(target=server.serve_forever, name="http-server", daemon=True)
        serving.start()
        while not quit_event.wait(0.5):
            pass
        return 0
    finally:
        if serving is not None:
            server.shutdown()
            serving.join()
        if server is not None:
            server.server_close()
        runner.shutdown()
        engine.close()
        restore()


if __name__ == "__main__":
    sys.exit(main())
