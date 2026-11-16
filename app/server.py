from __future__ import annotations

import hmac
import html
import json
import re
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit

from app import check_views, views
from app.jobs import ENDED, Refusal

HOST = "127.0.0.1"
MAX_BODY = 1024 * 1024  # bytes in a POST body
QUESTION_CHARS, SQL_CHARS = 2_000, 20_000
REASK_KINDS = ("ask", "reask")
ANSWERS = ("first", "second")

NO_TOKEN = "Open the link the app printed."
WRONG_HOST = "This address isn't the app's. Open the link the app printed."
WRONG_ORIGIN = "Writes must come from the app's own page."
NO_LENGTH = "The request needs a Content-Length header."
TOO_LARGE = "The request is larger than 1 MB."
BAD_JSON = "The request wasn't valid JSON."
NOT_OBJECT = "The request must be a JSON object."
EMPTY_QUESTION, LONG_QUESTION = "Type a question first.", "Questions are limited to 2,000 characters."
EMPTY_SQL, LONG_SQL = "Type some SQL first.", "SQL is limited to 20,000 characters."
NO_JOB, NO_ROWS = "Job not found.", "That answer has no rows."
STILL_RUNNING = "That question is still running."
NOT_A_QUESTION = "Only a finished question can be asked again."
UNKNOWN_READING = "Unknown reading."
NOT_FOUND, WRONG_METHOD = "Not found.", "Method not allowed."
BROKEN = "The app hit an error. The terminal it runs in has the details."
BAD_REQUEST = "The request wasn't valid HTTP."
BAD_CHARACTER = "The text has a character the app can't send. Retype it and try again."

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}
JSON_TYPE = CONTENT_TYPES[".json"]

# The page loads only its own files: no inline script or style, no other site, no framing.
CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)

ERROR_PAGE = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>ShilmanDB</title></head>
<body><p>{message}</p></body>
</html>
"""

# Method, path and the handler that serves it. A path that matches with another method is a 405.
ROUTES = (
    ("GET", re.compile(r"/"), "_page"),
    ("GET", re.compile(r"/static/(?P<path>.+)"), "_static"),
    ("GET", re.compile(r"/api/status"), "_status"),
    ("GET", re.compile(r"/api/history"), "_history"),
    ("POST", re.compile(r"/api/ask"), "_ask"),
    ("POST", re.compile(r"/api/reask"), "_reask"),
    ("POST", re.compile(r"/api/sql"), "_sql"),
    ("GET", re.compile(r"/api/jobs/(?P<job>[^/]+)"), "_job"),
    ("POST", re.compile(r"/api/jobs/(?P<job>[^/]+)/stop"), "_stop"),
    ("GET", re.compile(r"/api/jobs/(?P<job>[^/]+)/rows"), "_rows"),
    ("GET", re.compile(r"/api/jobs/(?P<job>[^/]+)/csv"), "_csv"),
)


# A request I answer with an error status and a plain message.
class HttpError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


# The query string as a dict; a repeated name keeps its last value.
def _query(text: str) -> dict[str, str]:
    return dict(parse_qsl(text, keep_blank_values=True))


# A whole number from the query, or the default when it's absent; 400 when it isn't one.
def _int(query: dict, name: str, default: int, low: int | None = None) -> int:
    text = query.get(name)
    if text is None:
        return default
    if re.fullmatch(r"-?[0-9]{1,18}", text) and (low is None or int(text) >= low):
        return int(text)
    raise HttpError(400, f'"{name}" must be a whole number{"" if low is None else f", {low} or more"}.')


# Which answer of a job the query names: first (the default) or second.
def _answer(query: dict) -> str:
    answer = query.get("answer", "first")
    if answer not in ANSWERS:
        raise HttpError(400, '"answer" must be first or second.')
    return answer


# A text field of a JSON body; an optional one may be absent or null.
def _field(body: dict, name: str, required: bool = True) -> str | None:
    value = body.get(name)
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise HttpError(400, f'The request needs "{name}" as text.')
    return value


# Typed text without its outer whitespace; 400 when that leaves nothing or more than the limit,
# or when it holds a character UTF-8 can't encode (a lone surrogate).
def _typed(text: str, empty: str, limit: int, too_long: str) -> str:
    text = text.strip()
    if not text:
        raise HttpError(400, empty)
    if len(text) > limit:
        raise HttpError(400, too_long)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise HttpError(400, BAD_CHARACTER) from None
    return text


# Whether the dataset's glossary has this term with this reading.
def _has_reading(spec: dict, term: str, reading: str) -> bool:
    return any(t["term"] == term and any(r["name"] == reading for r in t["readings"]) for t in spec.get("glossary", []))


# The app's HTTP server: 127.0.0.1 only, one thread per request, with what the handlers need.
class AppServer(ThreadingHTTPServer):
    daemon_threads = True  # a 20 s long poll never holds up quitting

    def __init__(self, runner, engine, token: str, port: int, static_dir: Path, examples: dict):
        super().__init__((HOST, port), Handler)
        port = self.server_address[1]  # the real port, when 0 asked for any free one
        self.runner, self.engine, self.examples = runner, engine, examples
        self.token = token.encode()
        self.static_dir = Path(static_dir).resolve()
        self.cookie_name = f"shilmandb_token_{port}"
        self.hosts = (f"127.0.0.1:{port}", f"localhost:{port}")
        self.origins = tuple(f"http://{host}" for host in self.hosts)


# The server for the page and its API, bound to 127.0.0.1; port 0 picks a free one.
def make_server(runner, engine, token: str, port: int, static_dir: Path, examples: dict) -> ThreadingHTTPServer:
    return AppServer(runner, engine, token, port, static_dir, examples)


# One request: the Host check, the token (from the printed link once, then the cookie),
# the Origin check on writes, then the route.
class Handler(BaseHTTPRequestHandler):
    server: AppServer
    timeout = 60.0  # seconds a client may take to send its request

    # Silenced: the default log line would print the token from the first URL.
    def log_message(self, format, *args) -> None:
        pass

    # Every response carries these, the first one (which answers the tokened URL) included.
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", CSP)
        super().end_headers()

    # Every method goes through the same checks; only GET and POST have routes.
    def do_GET(self) -> None:
        self._handle()

    do_POST = do_PUT = do_PATCH = do_DELETE = do_GET

    # Runs the checks and the route, and turns any refusal into its status and message.
    def _handle(self) -> None:
        self._sent = False
        api = self.path.startswith("/api/")
        try:
            # HTTP/0.9 has no headers to check; it's answered as HTTP/1.0 so the status line is sent.
            if self.request_version == "HTTP/0.9":
                self.request_version, self.close_connection = "HTTP/1.0", True
                raise HttpError(400, BAD_REQUEST)
            try:
                url = urlsplit(self.path)
            except ValueError:
                raise HttpError(400, BAD_REQUEST) from None

            # Read first, so a refused write doesn't leave its body unread and reset the connection.
            self._raw = self._read_body() if self.command == "POST" else b""
            if self._one("Host") not in self.server.hosts:
                raise HttpError(403, WRONG_HOST)
            query = _query(url.query)
            if self.command == "GET" and url.path == "/" and "token" in query:
                self._login(query["token"])
                return
            if not self._cookie_ok():
                raise HttpError(403, NO_TOKEN)
            if self.command == "POST" and self._one("Origin") not in self.server.origins:
                raise HttpError(403, WRONG_ORIGIN)
            handler, params = self._route(url.path)
            handler(query, **params)
        except (HttpError, Refusal) as e:
            self._quiet_fail(e.status, e.message, api)
        except (ConnectionError, TimeoutError):
            pass  # the browser went away, or stalled mid-request
        except Exception:
            traceback.print_exc()
            if not self._sent:
                self._quiet_fail(500, BROKEN, api)

    # An error answer; a browser that went away meanwhile is let go without a traceback.
    def _quiet_fail(self, status: int, message: str, api: bool) -> None:
        try:
            self._fail(status, message, api)
        except (ConnectionError, TimeoutError):
            pass

    # A header that must appear exactly once; None when it's missing or repeated.
    def _one(self, name: str) -> str | None:
        values = self.headers.get_all(name) or []
        return values[0] if len(values) == 1 else None

    # The POST body, at most MAX_BODY bytes; an HttpError (not raised yet) when it can't be read.
    def _read_body(self) -> bytes | HttpError:
        length = self._one("Content-Length")
        if length is None or not re.fullmatch(r"[0-9]{1,18}", length):
            return HttpError(400, NO_LENGTH)
        if int(length) > MAX_BODY:
            return HttpError(413, TOO_LARGE)
        return self.rfile.read(int(length))

    # The body as a JSON object.
    def _body(self) -> dict:
        if isinstance(self._raw, HttpError):
            raise self._raw
        try:
            body = json.loads(self._raw)
        except (ValueError, RecursionError):
            raise HttpError(400, BAD_JSON) from None
        if not isinstance(body, dict):
            raise HttpError(400, NOT_OBJECT)
        return body

    # Whether the request carries this launch's cookie, compared in constant time.
    def _cookie_ok(self) -> bool:
        for header in self.headers.get_all("Cookie") or []:
            for part in header.split(";"):
                name, _, value = part.strip().partition("=")
                if name == self.server.cookie_name and hmac.compare_digest(value.encode(), self.server.token):
                    return True
        return False

    # The handler for a path and method, with the path's parts; 405 for a known path, else 404.
    def _route(self, path: str):
        known = False
        for method, pattern, name in ROUTES:
            found = pattern.fullmatch(path)
            if found and method == self.command:
                return getattr(self, name), found.groupdict()
            known = known or found is not None
        raise HttpError(405, WRONG_METHOD) if known else HttpError(404, NOT_FOUND)

    # One whole response: status, headers, body.
    def _send(self, status: int, body: bytes, content_type: str, headers: tuple = ()) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in headers:
            self.send_header(name, value)
            
        self.end_headers()
        self._sent = True
        self.wfile.write(body)

    # A value as a JSON response.
    def _json(self, value, status: int = 200) -> None:
        self._send(status, json.dumps(value).encode("utf-8"), JSON_TYPE)

    # An error: JSON for the API, a short page for everything else.
    def _fail(self, status: int, message: str, api: bool) -> None:
        if api:
            self._json({"error": message}, status)
        else:
            page = ERROR_PAGE.format(message=html.escape(message))
            self._send(status, page.encode("utf-8"), CONTENT_TYPES[".html"])

    # The printed link: a right token becomes the cookie and the address bar loses it.
    def _login(self, token: str) -> None:
        if not hmac.compare_digest(token.encode(), self.server.token):
            raise HttpError(403, NO_TOKEN)
        
        cookie = f"{self.server.cookie_name}={token}; HttpOnly; SameSite=Strict; Path=/"
        self._send(303, b"", CONTENT_TYPES[".html"], (("Location", "/"), ("Set-Cookie", cookie)))

    # A file under the static folder; anything outside it, not a file, or a name the file system
    # refuses (too long, say) is a 404.
    def _file(self, relative: str) -> None:
        root = self.server.static_dir
        try:
            target = None if "\0" in relative else (root / relative).resolve()
            found = target is not None and target.is_relative_to(root) and target.is_file()
        except (OSError, ValueError):
            found = False
        if not found:
            raise HttpError(404, NOT_FOUND)
        
        content_type = CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        self._send(200, target.read_bytes(), content_type)

    # The page itself, and the files it loads.
    def _page(self, query: dict) -> None:
        self._file("index.html")

    def _static(self, query: dict, path: str) -> None:
        self._file(unquote(path))

    # What's loaded, plus the examples and the tested panel, which stays empty unless the lock holds.
    def _status(self, query: dict) -> None:
        engine = self.server.engine
        panel = check_views.tested_panel(engine.measured["rates"]) if engine.lock.holds else []
        self._json({**engine.status(), "examples": self.server.examples, "tested_panel": panel})

    # This session's jobs, newest first.
    def _history(self, query: dict) -> None:
        self._json({"history": self.server.runner.history()})

    # A typed question for the analyst.
    def _ask(self, query: dict) -> None:
        body = self._body()
        dataset = _field(body, "dataset")
        question = _typed(_field(body, "question"), EMPTY_QUESTION, QUESTION_CHARS, LONG_QUESTION)
        self._json(self.server.runner.submit("ask", dataset, question))

    # Asks a finished question again with one reading named. Only the app's own sentence is added,
    # always to the question as first typed, so re-asks never stack sentences.
    def _reask(self, query: dict) -> None:
        body = self._body()
        job_id, term, reading = _field(body, "job"), _field(body, "term"), _field(body, "reading")
        source = self._snapshot(job_id)
        if source["kind"] not in REASK_KINDS:
            raise HttpError(400, NOT_A_QUESTION)
        if source["stage"] not in ENDED:
            raise HttpError(409, STILL_RUNNING)
        if source["stage"] != "done":
            raise HttpError(400, NOT_A_QUESTION)
        if not _has_reading(self.server.engine.dataset(source["dataset"]).spec, term, reading):
            raise HttpError(400, UNKNOWN_READING)

        typed = source["asked"]["question"] if source["kind"] == "reask" and source["asked"] else source["text"]
        text = check_views.reask_question(typed, term, reading)
        asked = {"term": term, "reading": reading, "question": typed}
        self._json(self.server.runner.submit("reask", source["dataset"], text, asked=asked, from_job=source["id"]))

    # Typed SQL; with from_job it's an edit of that question's SQL.
    def _sql(self, query: dict) -> None:
        body = self._body()
        dataset, from_job = _field(body, "dataset"), _field(body, "from_job", required=False) or None
        sql = _typed(_field(body, "sql"), EMPTY_SQL, SQL_CHARS, LONG_SQL)
        self._json(self.server.runner.submit("edit" if from_job else "sql", dataset, sql, from_job=from_job))

    # The job's snapshot once its version passes after (-1, the default, answers at once), or at the long poll's end.
    def _job(self, query: dict, job: str) -> None:
        snapshot = self.server.runner.wait(job, _int(query, "after", -1))
        if snapshot is None:
            raise HttpError(404, NO_JOB)
        self._json(snapshot)

    # Stops a job at its next step; says whether it was still going.
    def _stop(self, query: dict, job: str) -> None:
        self._snapshot(job)
        self._json({"stopped": self.server.runner.stop(job)})

    # One page of an answer's rows, at most views.DISPLAY_ROWS.
    def _rows(self, query: dict, job: str) -> None:
        answer, offset = _answer(query), _int(query, "offset", 0, low=0)
        self._snapshot(job)
        rows = self.server.runner.rows(job, answer, offset, views.DISPLAY_ROWS)
        if rows is None:
            raise HttpError(404, NO_ROWS)
        self._json({"offset": offset, "rows": rows})

    # Every row of an answer as a CSV download.
    def _csv(self, query: dict, job: str) -> None:
        answer = _answer(query)
        self._snapshot(job)
        text = self.server.runner.csv(job, answer)
        if text is None:
            raise HttpError(404, NO_ROWS)
        disposition = f'attachment; filename="shilmandb-{job}.csv"'
        self._send(200, text.encode("utf-8"), "text/csv; charset=utf-8", (("Content-Disposition", disposition),))

    # A job's current snapshot; 404 when there's no such job.
    def _snapshot(self, job_id: str) -> dict:
        snapshot = self.server.runner.wait(job_id, -1, 0)
        if snapshot is None:
            raise HttpError(404, NO_JOB)
        return snapshot
