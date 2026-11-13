from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sqlite3
import sys
from collections import Counter
from contextlib import ExitStack
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
qm = an.load_sibling("an_questions", "questions.py")
sc = an.load_sibling("an_scoring", "scoring.py")
bc = an.load_sibling("an_bridge_client", "bridge_client.py")

# Checked in this order; the first that applies wins.
STATUSES = ("GOLD_ERROR", "SECOND_MISSING", "UNANSWERABLE_DIFFER", "WRITERS_DIFFER", "ENGINES_DIFFER", "EMPTY", "TIE", "EXAMPLE_OVERLAP", "AGREE")
EXAMPLES = an.HERE / "prompt_examples.json"
TIMEOUT_S = an.TIMEOUT_S
LOG_ROWS = 5                     # rows of each result shown in key_log.md
UNCACHED = ("timeout", "crash")  # failures that may not happen again

LIMIT_NEEDS_ORDER = "a LIMIT needs ordered: true"
ONE_TRAILING_LIMIT = "use one trailing LIMIT n; the tie check can't rerun this LIMIT"
NOT_SORTED = "rows aren't sorted on order_cols"
RERUN_DIFFERS = "the tie check's rerun didn't reproduce the gold"

_STRING = re.compile(r"'(?:[^']|'')*'")
_ORDER_BY = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)
_LIMIT = re.compile(r"\bLIMIT\b", re.IGNORECASE)
_TRAILING_LIMIT = re.compile(r"\bLIMIT\s+(\d+)\s*;?\s*$", re.IGNORECASE)
# What SQLite skips over: 'strings', "names", `names`, [names], -- comments and /* comments */.
_SQL_SKIP = re.compile(r"""'(?:[^']|'')*'?|"(?:[^"]|"")*"?|`(?:[^`]|``)*`?|\[[^\]]*\]?|--[^\n]*|/\*.*?(?:\*/|\Z)""", re.DOTALL)


# A failed result, shaped like a good one.
def _error(kind: str, message: str) -> dict:
    return {"ok": False, "kind": kind, "error": message}


# One query on a dataset's SQLite file, using this module's TIMEOUT_S at call time.
def _sqlite_query(db: Path, sql: str) -> dict:
    return an.sqlite_query(db, sql, timeout_s=TIMEOUT_S)


# A bridge reply in the same shape as a SQLite result.
def _from_bridge(reply: dict) -> dict:
    if reply.get("ok") is not True:
        err = reply.get("error") or {}
        kind = err.get("kind", "request")
        return _error(kind, f"{kind}: {err.get('message', '')}")
    
    return {
        "ok": True,
        "columns": [c["name"] for c in reply["columns"]], 
        "rows": reply["rows"],
        "truncated": bool(reply.get("truncated"))
    }


# Cached entries from key_answers.json; a missing or broken file is just no cache.
def _load_cache(path: Path) -> dict:
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    return entries if isinstance(entries, dict) else {}


#Runs SQL on SQLite or the engine, reusing cached results while the dataset files are unchanged
class _Runner:
    def __init__(self, p, bridge_factory, stack: ExitStack, datasets: list[str]):
        self.p = p
        self.factory = bridge_factory
        self.stack = stack
        self.data_sha1 = {ds: _data_sha1(p.dataset_dir(ds), ds) for ds in datasets}
        self.old = _load_cache(p.key_answers)
        self.memo: dict[str, dict] = {}  # every result this run, failures too
        self.kept: dict[str, dict] = {}  # what goes back into key_answers.json
        self.bridges: dict[str, object] = {}

    # The gold or second writer's SQL on SQLite.
    def sqlite(self, ds: str, sql: str) -> dict:
        db = self.p.dataset_dir(ds) / f"{ds}.sqlite"
        return self._run("sqlite", ds, sql, lambda: _sqlite_query(db, sql))

    # gold_subset on the engine, starting the dataset's bridge on first use.
    def engine(self, ds: str, sql: str) -> dict:
        return self._run("engine", ds, sql, lambda: _from_bridge(self._bridge(ds).query(sql)))

    # Bridge restarts so far, over every dataset.
    def restarts(self) -> int:
        return sum(b.restarts for b in self.bridges.values())

    # The dataset's bridge, opened once and closed when the stack closes.
    def _bridge(self, ds: str):
        if ds not in self.bridges:
            self.bridges[ds] = self.stack.enter_context(self.factory(self.p.dataset_dir(ds) / "tables.tsv"))
        return self.bridges[ds]

    # A cached result while still good, else a fresh run; timeouts and crashes aren't kept.
    def _run(self, engine: str, ds: str, sql: str, run) -> dict:
        key = an.sha1_text(f"{engine}\n{ds}\n{sql}")
        if key in self.memo:
            return self.memo[key]

        entry = self.old.get(key)
        fresh = isinstance(entry, dict) and entry.get("data_sha1") == self.data_sha1[ds] and entry.get("sql") == sql and isinstance(entry.get("result"), dict)
        result = entry["result"] if fresh else run()
        self.memo[key] = result
        
        if result.get("kind") not in UNCACHED:
            self.kept[key] = {
                "engine": engine, 
                "dataset": ds,
                "sql": sql,
                "data_sha1": self.data_sha1[ds], 
                "result": result
            }
        return result

    # Write this run's results to key_answers.json; entries no question used are dropped.
    def save(self) -> None:
        an.write_text_atomic(self.p.key_answers, json.dumps({"entries": self.kept}) + "\n")


# Casefolded with whitespace collapsed, for comparing question texts.
def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


# Every prompt example's question, normalized; empty when the file doesn't exist.
def _example_questions(path: Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(isinstance(v, list) for v in data.values()):
        raise ValueError(f"{path}: expected {{dataset: [{{question, reply}}, ...]}}")
    
    for ds, examples in data.items():
        for i, e in enumerate(examples):
            if not isinstance(e, dict) or not isinstance(e.get("question"), str):
                raise ValueError(f"{path}: {ds}[{i}] isn't an object with a question string: {e!r}")
            
    return {_normalize(e["question"]) for examples in data.values() for e in examples}


# The .tbl files a dataset's manifest lists, sorted by name.
def _listed_tbl_files(ds_dir: Path) -> list[Path]:
    manifest = json.loads((ds_dir / "manifest.json").read_text(encoding="utf-8"))
    tables = manifest.get("tables") if isinstance(manifest, dict) else None
    
    if not isinstance(tables, list) or not all(isinstance(t, dict) and isinstance(t.get("file"), str) for t in tables):
        raise ValueError(f"{ds_dir / 'manifest.json'}: expected tables with a file each")
    return sorted(ds_dir / t["file"] for t in tables)


# One sha1 over a dataset's manifest, SQLite file and every .tbl the manifest lists.
def _data_sha1(ds_dir: Path, ds: str) -> str:
    files = [ds_dir / "manifest.json", ds_dir / f"{ds}.sqlite", *_listed_tbl_files(ds_dir)]
    return an.sha1_text("\n".join(f"{f.name} {an.sha1_file(f)}" for f in files))


# Raise FileNotFoundError naming every dataset file the check needs and can't find.
def _require_files(p, datasets: list[str]) -> None:
    missing = []
    for ds in datasets:
        ds_dir = p.dataset_dir(ds)
        needed = [ds_dir / "manifest.json", ds_dir / f"{ds}.sqlite", ds_dir / "tables.tsv"]
        if needed[0].exists():
            needed += _listed_tbl_files(ds_dir)
            
        missing += [path for path in needed if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing {', '.join(map(str, missing))}; run prepare_datasets.py first")


# SQL on one line, without -- comments, a trailing ";" or the text inside 'strings'.
def _bare(sql: str) -> str:
    return _STRING.sub("''", bc.one_line(sql))


# SQLite SQL with strings and quoted names as '#', comments as spaces; same length.
# Also returns where each /* comment */ starts.
def _scan(sql: str) -> tuple[str, list[int]]:
    chars, blocks = list(sql), []
    for m in _SQL_SKIP.finditer(sql):
        token = m.group()
        chars[m.start():m.end()] = (" " if token.startswith(("--", "/*")) else "#") * len(token)
        if token.startswith("/*"):
            blocks.append(m.start())
            
    return "".join(chars), blocks


# The match for gold_sqlite's only LIMIT when it's a trailing LIMIT n, else None.
# Its group 1 spans the number in the raw SQL too.
def _trailing_limit(sql: str) -> re.Match | None:
    masked, blocks = _scan(sql)
    m = _TRAILING_LIMIT.search(masked)
    if m is None or len(_LIMIT.findall(masked)) != 1 or any(b > m.start(1) for b in blocks):
        return None
    return m


# Problems with LIMITs: one needs ordered, and gold_sqlite's must be one trailing LIMIT n.
def _limit_problems(q: dict) -> list[str]:
    gold_limits = len(_LIMIT.findall(_scan(q["gold_sqlite"])[0]))
    subset_limits = len(_LIMIT.findall(_bare(q["gold_subset"])))
    problems = [LIMIT_NEEDS_ORDER] if (gold_limits or subset_limits) and not q["ordered"] else []
    if gold_limits and _trailing_limit(q["gold_sqlite"]) is None:
        problems.append(ONE_TRAILING_LIMIT)
    return problems


# An ordered gold with its trailing LIMIT n made n+1, nothing else changed; and n.
def _rerun_sql(q: dict) -> tuple[str, int] | None:
    if not q["ordered"] or q["gold_sqlite"] is None or _limit_problems(q):
        return None
    
    sql, m = q["gold_sqlite"], _trailing_limit(q["gold_sqlite"])
    if m is None or int(m.group(1)) < 1:
        return None
    
    n = int(m.group(1))
    return f"{sql[:m.start(1)]}{n + 1}{sql[m.end(1):]}", n


# Why the second writer's record can't be used for this question text, or None.
def _second_problem(rec: dict | None, question: str) -> str | None:
    if rec is None:
        return "no second answer"
    
    if not isinstance(rec.get("question"), str) or "sqlite" not in rec or not isinstance(rec.get("unanswerable"), bool):
        return "the second answer needs question, sqlite and unanswerable (true or false)"
    
    if " ".join(rec["question"].split()) != " ".join(question.split()):
        return "answered an older wording"

    sql = rec["sqlite"]
    if sql is not None and not isinstance(sql, str):
        return "the second answer's sqlite must be text or null"
    
    has_sql = bool(sql and sql.strip())
    if rec["unanswerable"]:
        return "the second answer is marked unanswerable but has SQL" if has_sql else None
    return None if has_sql else "the second answer has no SQL"


# An int or float, but not a bool.
def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# No rows, or some number or NULL cells, all of them 0 or NULL.
def _is_empty(rows: list[list]) -> bool:
    cells = [v for row in rows for v in row if v is None or _is_number(v)]
    return not rows or (bool(cells) and all(v is None or v == 0 for v in cells))


# A cell's sort key: NULL before every number or text, as SQLite sorts ascending.
def _sort_key(v) -> tuple:
    return (0,) if v is None else (1, v)


# Why the rows aren't sorted on order_cols, each column one way; None if sorted.
# A column's direction comes from its first change among rows tied on earlier columns.
def _sort_problem(rows: list[list], order_cols: list[int]) -> str | None:
    for j in order_cols:
        if len({isinstance(row[j], str) for row in rows if row[j] is not None}) > 1:
            return f"{NOT_SORTED} (column {j} mixes numbers and text)"

    direction: dict[int, int] = {}
    for prev, cur in zip(rows, rows[1:]):
        for j in order_cols:
            a, b = _sort_key(prev[j]), _sort_key(cur[j])
            if a != b:
                step = 1 if a < b else -1
                if direction.setdefault(j, step) != step:
                    return f"{NOT_SORTED} (column {j} goes both ways)"
                break
    return None


# What's wrong with the gold: failed SQL, LIMIT or ORDER BY rules, bad column indexes.
def _gold_problems(q: dict, res: dict) -> list[str]:
    if q["gold_sqlite"] is None:
        return []

    problems = _limit_problems(q)
    if q["ordered"]:
        bare = {
            "gold_sqlite": _scan(q["gold_sqlite"])[0], 
            "gold_subset": _bare(q["gold_subset"])
        }
        problems += [f"ordered, but {field} has no ORDER BY" for field, sql in bare.items() if not _ORDER_BY.search(sql)]

    gold, subset, rerun = res["gold"], res["subset"], res.get("rerun")
    if not gold["ok"]:
        problems.append(f"gold_sqlite failed: {gold['error']}")
    if not subset["ok"]:
        problems.append(f"gold_subset failed: {subset['error']}")
    if rerun is not None and not rerun["ok"]:
        problems.append(f"the LIMIT n+1 rerun failed: {rerun['error']}")
        
    if gold["ok"]:
        width = len(gold["columns"])
        for field in ("order_cols", "share_cols"):
            outside = [i for i in q[field] if i >= width]
            if outside:
                problems.append(f"{field} {outside} outside the gold's {width} columns")
                
        if all(i < width for i in q["order_cols"]):
            unsorted = _sort_problem(gold["rows"], q["order_cols"])
            if unsorted:
                problems.append(unsorted)
            if rerun is not None and rerun["ok"] and not _rerun_reproduces(gold, rerun, q["order_cols"]):
                problems.append(RERUN_DIFFERS)
    return problems


# The rerun's first rows equal the gold's rows on every order column.
def _rerun_reproduces(gold: dict, rerun: dict, order_cols: list[int]) -> bool:
    rows = rerun["rows"]
    return (all(i < len(rerun["columns"]) for i in order_cols) and len(rows) >= len(gold["rows"]) and all(sc.rows_tie(r, g, order_cols) for r, g in zip(rows, gold["rows"])))


# The LIMIT n when the rerun shows the cut falls on a tie; else None.
def _tied_cut(q: dict, res: dict) -> int | None:
    cut, gold, rerun = _rerun_sql(q), res.get("gold"), res.get("rerun")
    if cut is None or rerun is None or not rerun["ok"] or not _rerun_reproduces(gold, rerun, q["order_cols"]):
        return None
    rows, n = rerun["rows"], cut[1]
    return n if len(rows) > n and sc.rows_tie(rows[n - 1], rows[n], q["order_cols"]) else None


# The answerable checks after the writers agree it's answerable: writers, engines, empty, tie.
def _compare(q: dict, res: dict) -> tuple[str, str] | None:
    gold, second, subset = res["gold"], res["second"], res["subset"]

    def matches(shown: dict) -> bool:
        return sc.match_result(shown["columns"], shown["rows"], gold["columns"], gold["rows"], ordered=q["ordered"], share_cols=q["share_cols"], order_cols=q["order_cols"])

    if not second["ok"]:
        return "WRITERS_DIFFER", f"the second writer's SQL failed: {second['error']}"
    if not matches(second):
        return "WRITERS_DIFFER", (f"the second writer's {len(second['rows'])} rows don't match the gold's {len(gold['rows'])}")
    if subset.get("truncated"):
        return "ENGINES_DIFFER", f"the engine's result was truncated at {an.ROW_CAP:,} rows"
    if not matches(subset):
        return "ENGINES_DIFFER", f"the engine's {len(subset['rows'])} rows don't match the gold's {len(gold['rows'])}"
    if not q.get("empty_ok") and _is_empty(gold["rows"]):
        return "EMPTY", "no rows" if not gold["rows"] else "every number is 0 or NULL"

    n = _tied_cut(q, res)
    if n is not None:
        return "TIE", f"rows {n} and {n + 1} tie on order_cols {q['order_cols']}, so LIMIT {n} cuts a tie"
    return None


# The first status that applies and its detail, given every result already run.
def _first_status(q: dict, rec: dict | None, res: dict, examples: set[str]) -> tuple[str, str]:
    gold_problems = _gold_problems(q, res)
    if gold_problems:
        return "GOLD_ERROR", "; ".join(gold_problems)
    
    second_problem = _second_problem(rec, q["question"])
    if second_problem:
        return "SECOND_MISSING", second_problem

    answerable = q["gold_sqlite"] is not None
    if rec["unanswerable"] == answerable:
        return "UNANSWERABLE_DIFFER", ("the second writer says it can't be answered" if answerable else "the gold says it can't be answered, but the second writer answered")
    if answerable:
        found = _compare(q, res)
        if found:
            return found
        
    if _normalize(q["question"]) in examples:
        return "EXAMPLE_OVERLAP", "the question matches a prompt example"
    return "AGREE", ""


# The first status that applies; an earlier one still notes a tied LIMIT cut.
def _decide(q: dict, rec: dict | None, res: dict, examples: set[str]) -> tuple[str, str]:
    status, detail = _first_status(q, rec, res, examples)
    if status != "TIE" and _tied_cut(q, res) is not None:
        detail += "; the cut also falls on a tie"
    return status, detail


# Run every SQL a question has, then decide its status.
def _evaluate(q: dict, rec: dict | None, runner: _Runner, examples: set[str]) -> dict:
    ds, res = q["dataset"], {}
    if q["gold_sqlite"] is not None:
        res["gold"] = runner.sqlite(ds, q["gold_sqlite"])
        res["subset"] = runner.engine(ds, q["gold_subset"])
        cut = _rerun_sql(q)
        if cut is not None and res["gold"]["ok"]:
            res["rerun"] = runner.sqlite(ds, cut[0])
            
    if _second_problem(rec, q["question"]) is None and not rec["unanswerable"]:
        res["second"] = runner.sqlite(ds, rec["sqlite"])

    status, detail = _decide(q, rec, res, examples)
    return {
        "qid": q["qid"], 
        "status": status, 
        "detail": detail, 
        "results": res, 
        "second": rec
    }


# One result in key_log.md: its SQL, then its first rows or its error.
def _result_lines(label: str, sql: str, result: dict | None) -> list[str]:
    lines = [f"**{label}**", "", "```sql", sql.strip(), "```", ""]
    if result is None:
        return lines + ["Not run.", ""]
    if not result["ok"]:
        return lines + [f"Error: {result['error']}", ""]

    rows = result["rows"]
    more = ", truncated" if result.get("truncated") else ""
    return lines + [f"{len(rows)} rows{more}; columns: {', '.join(result['columns'])}", "", "```", *(json.dumps(r) for r in rows[:LOG_ROWS]), "```", ""]


# key_log.md: counts per status, then each question that isn't AGREE with its SQL and results.
def _render_log(qs: list[dict], evaluated: list[dict], restarts: int) -> str:
    counts = Counter(e["status"] for e in evaluated)
    lines = [
        "# Answer key check", "",
        f"{len(evaluated)} questions, {len(evaluated) - counts['AGREE']} not AGREE; bridge restarts: {restarts}.",
        "", "| status | questions |", "|---|---:|", *(f"| {s} | {counts[s]} |" for s in STATUSES)
    ]

    for q, e in zip(qs, evaluated):
        if e["status"] == "AGREE":
            continue
        
        res, rec = e["results"], e["second"]
        lines += ["", f"## {q['qid']} ({q['dataset']}, {q['kind']}): {e['status']}", "", e["detail"], "", f"Question: {q['question']}", ""]
        if q["gold_sqlite"] is None:
            lines += [f"**gold_sqlite**: unanswerable ({q['unanswerable_reason']})", ""]
        else:
            lines += _result_lines("gold_sqlite (SQLite)", q["gold_sqlite"], res.get("gold"))
            if e["status"] in ("GOLD_ERROR", "ENGINES_DIFFER"):
                lines += _result_lines("gold_subset (engine)", q["gold_subset"], res.get("subset"))
            if e["status"] == "TIE":
                lines += _result_lines("gold_sqlite with LIMIT n+1", _rerun_sql(q)[0], res.get("rerun"))

        if _second_problem(rec, q["question"]) is not None:
            lines += [f"**second writer**: {_second_problem(rec, q['question'])}", ""]
        elif rec["unanswerable"]:
            lines += ["**second writer**: unanswerable", ""]
        else:
            lines += _result_lines("second writer (SQLite)", rec["sqlite"], res.get("second"))
    return "\n".join(lines) + "\n"


# The questions and each one's evaluation; writes key_log.md and key_answers.json.
def _check(results: Path, bridge_factory, examples_path: Path) -> tuple[list[dict], list[dict]]:
    p = an.paths(results)
    qs = qm.read_questions(p.questions)
    problems = qm.validate(qs, initial=False)
    if problems:
        raise ValueError(f"bad questions in {p.questions}: {'; '.join(problems)}")

    seconds = {r["qid"]: r for r in an.read_jsonl(p.second_answers) if isinstance(r.get("qid"), str)}  # last wins
    examples = _example_questions(examples_path)
    datasets = sorted({q["dataset"] for q in qs})
    _require_files(p, datasets)

    with ExitStack() as stack:
        runner = _Runner(p, bridge_factory, stack, datasets)
        try:
            evaluated = [_evaluate(q, seconds.get(q["qid"]), runner, examples) for q in qs]
            restarts = runner.restarts()
        finally:
            runner.save()

    an.write_text_atomic(p.key_log, _render_log(qs, evaluated, restarts))
    return qs, evaluated


# One status row per question, in file order; writes key_log.md and key_answers.json.
def check(results: Path, bridge_factory=None, examples_path: Path | None = None) -> list[dict]:
    _, evaluated = _check(Path(results), bc.Bridge if bridge_factory is None else bridge_factory, EXAMPLES if examples_path is None else examples_path)
    return [{
        "qid": e["qid"], 
        "status": e["status"], 
        "detail": e["detail"]
    } for e in evaluated]


# Check the key; when every row is AGREE, write answers and splits into questions.jsonl.
def freeze(results: Path, bridge_factory=None, examples_path: Path | None = None) -> list[dict]:
    qs, evaluated = _check(Path(results), bc.Bridge if bridge_factory is None else bridge_factory, EXAMPLES if examples_path is None else examples_path)
    rows = [{"qid": e["qid"], "status": e["status"], "detail": e["detail"]} for e in evaluated]
    if any(r["status"] != "AGREE" for r in rows):
        return rows

    for q, e in zip(qs, evaluated):
        gold = e["results"].get("gold")
        q["answer"] = {"columns": gold["columns"], "rows": gold["rows"]} if gold is not None else None
    qm.write_questions(an.paths(results).questions, qm.assign_split(qs))
    return rows


# Validate the questions, then check or freeze; exit 0 only when every question is AGREE.
def main(argv: list[str] | None = None, bridge_factory=None) -> int:
    ap = argparse.ArgumentParser(description="Check the answer key, or freeze it with answers and a dev/test split.")
    ap.add_argument("command", choices=("check", "freeze"))
    ap.add_argument("--results", type=Path, default=an.RESULTS)
    args = ap.parse_args(argv)
    p = an.paths(args.results)

    if not p.questions.exists():
        print(f"no {p.questions}; write the questions first", file=sys.stderr)
        return 1
    try:
        qs = qm.read_questions(p.questions)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    problems = qm.validate(qs, initial=not any("split" in q for q in qs))  # the first set until frozen
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print(f"{len(problems)} problems in {p.questions}; nothing run", file=sys.stderr)
        return 1

    run = check if args.command == "check" else freeze
    try:
        rows = run(args.results, bridge_factory=bridge_factory)
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as e:
        print(f"{args.command} failed: {e}", file=sys.stderr)
        return 1

    counts = Counter(r["status"] for r in rows)
    print(", ".join(f"{s} {counts[s]}" for s in STATUSES if counts[s]) + f" -> {p.key_log}")
    clean = counts["AGREE"] == len(rows)
    if args.command == "freeze":
        if not clean:
            print("not frozen: every question must be AGREE first", file=sys.stderr)
        else:
            splits = Counter(q["split"] for q in qm.read_questions(p.questions))
            print(f"froze {len(rows)} questions (dev {splits['dev']}, test {splits['test']}) -> {p.questions}")
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
