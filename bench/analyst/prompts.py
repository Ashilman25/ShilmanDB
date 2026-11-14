from __future__ import annotations

import importlib.util
import json
import re
import sys
from dataclasses import dataclass
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

ROLE = "You are a data analyst. You answer business questions by writing one SQL query over the dataset below."

REPLY_FORMAT = """\
Reply with exactly one JSON object and nothing else, in one of these two forms.

If the dataset can answer the question:

{"sql": "<one SELECT statement>", "grain": "<what one row of the result means>", "columns": [{"name": "<column name in the result>", "meaning": "<what the value is>", "entity": "<what the column counts or sums>"}]}

- grain says what one row means, for example "one row per region", or "one row" for a single total.
- columns lists every column the query returns, in order. entity says what a column counts or sums, for example "orders" or "revenue".

If the dataset doesn't hold the data the question needs:

{"cannot_answer": "<what's missing>"}"""

REPHRASE = "Rewrite this question in different words with exactly the same meaning. Reply with the rewritten question only."
RETRY = "Write a corrected reply in the same format."

# Problems a retry can report, besides the engine's own error message.
NOT_A_REPLY = "the reply wasn't one JSON object in the reply format"
TOO_SLOW = f"the query took longer than {an.TIMEOUT_S:g} seconds"
CRASHED = "the engine crashed on that query"
TOO_MUCH_MEMORY = f"the query used more than {an.MAX_RSS_MB} MB of memory"

_HEADING = re.compile(r"^#+(?=[ \t]|$)", re.MULTILINE)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_QUOTES = {'"': '"', "'": "'", "“": "”", "‘": "’"}

# raw_decode matches braces the JSON way, so braces inside strings don't count.
# strict=False lets a string hold raw line breaks, as SQL written over lines does.
_DECODER = json.JSONDecoder(strict=False)


@dataclass
class Reply:
    kind: str                # "sql", "cannot" or "malformed"
    sql: str | None
    grain: str | None
    columns: list[dict]
    reason: str | None       # why the model says the data can't answer
    raw: str                 # the model's text as it came back


# Drop a "# " title line and add one # to every other heading.
def _nest(markdown: str) -> str:
    lines = markdown.strip().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return _HEADING.sub(lambda m: "#" + m.group(), "\n".join(lines)).strip()


# One example: the question, then its reply as one line of JSON.
def _example(e: dict) -> str:
    return f"Question: {e['question']}\nReply: {json.dumps(e['reply'], ensure_ascii=False)}"


# The prompt that asks for one SQL reply (or a refusal) to a question.
def build_sql_prompt(description: str, rules: str, examples: list[dict], question: str) -> str:
    sections = [
        ROLE,
        f"## Dataset\n\n{_nest(description)}",
        f"## SQL you can use\n\n{_nest(rules)}",
        "## Examples\n\n" + "\n\n".join(_example(e) for e in examples),
        f"## Reply format\n\n{REPLY_FORMAT}",
        f"## Question\n\n{question.strip()}",
    ]
    return "\n\n".join(sections)


# The first prompt again, with the reply that failed and why it failed.
def build_retry_prompt(base: str, previous_reply: str, problem: str) -> str:
    return (f"{base}\n\n## Your previous reply\n\n{previous_reply.strip()}\n\n"
            f"## Problem\n\n{problem.strip()}\n\n{RETRY}")


# The prompt that asks for the same question in other words.
def build_rephrase_prompt(question: str) -> str:
    return f"{REPHRASE}\n\n{question.strip()}"


# Text with every <think>...</think> block removed; "" if one is never closed.
def strip_thinking(text: str) -> str:
    text = _THINK_BLOCK.sub("", text)
    return "" if "<think>" in text else text


# True for an object with a non-empty sql string or a cannot_answer.
def _is_reply(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    sql = obj.get("sql")
    return (isinstance(sql, str) and bool(sql.strip())) or "cannot_answer" in obj


# The first {...} that parses as JSON and is a reply, else None.
def _first_reply_object(text: str) -> dict | None:
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        try:
            obj, _ = _DECODER.raw_decode(text, start)
        except ValueError:
            continue
        if _is_reply(obj):
            return obj
    return None


# What the model wrote back: SQL with its grain and columns, a refusal, or malformed.
def parse_reply(text: str) -> Reply:
    obj = _first_reply_object(strip_thinking(text))
    if obj is None:
        return Reply("malformed", None, None, [], None, text)

    sql = obj.get("sql")
    if isinstance(sql, str) and sql.strip():
        grain, columns = obj.get("grain"), obj.get("columns")
        return Reply(
            "sql", sql, grain if isinstance(grain, str) else None,
            [c for c in columns if isinstance(c, dict)] if isinstance(columns, list) else [],
            None, text,
        )

    reason = obj["cannot_answer"]
    return Reply("cannot", None, None, [], reason if isinstance(reason, str) else json.dumps(reason), text)


# The rewritten question: first non-empty line, outer quotes removed; None if empty.
def parse_rephrase(text: str) -> str | None:
    for line in strip_thinking(text).splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) >= 2 and _QUOTES.get(line[0]) == line[-1]:
            line = line[1:-1].strip()
        return line or None
    return None
