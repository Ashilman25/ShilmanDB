"""Tests for bench/analyst/answer_key.py."""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


# Load a module by path under an an_ name, clear of other bench tests.
def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, HERE.parent / file_name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mod = _load("an_answer_key", "answer_key.py")

SAME = object()  # gold_subset defaults to the gold SQL
TABLES = {
    "tpch": [
        "CREATE TABLE nation (n_nationkey INTEGER, n_name TEXT, n_regionkey INTEGER)",
        "INSERT INTO nation VALUES (0, 'ALGERIA', 0), (7, 'GERMANY', 3), (8, 'GHANA', 0), (9, 'georgia', 3)",
        "CREATE TABLE s (name TEXT, score INTEGER)",
        "INSERT INTO s VALUES ('a', 30), ('b', 20), ('c', 20)",
        "CREATE TABLE v (x INTEGER, label TEXT)",
        "INSERT INTO v VALUES (-5, 'b'), (NULL, NULL), (3, 'a'), (-1, 'c')",
        "CREATE TABLE w (a INTEGER, b INTEGER)",
        "INSERT INTO w VALUES (NULL, 5), (0, 1), (0, 2)",
    ],
    "reviews": [
        "CREATE TABLE products (product_id INTEGER, title TEXT, price REAL)",
        "INSERT INTO products VALUES (1, 'lamp', 19.5), (2, 'desk', 120.0)",
    ],
}


class SqliteBridge:
    """Stands in for Bridge: runs each query on the <ds>.sqlite next to tables.tsv."""

    def __init__(self, tables_tsv: Path, truncated: bool = False):
        self.tables_tsv = Path(tables_tsv)
        self.db = self.tables_tsv.parent / f"{self.tables_tsv.parent.name}.sqlite"
        self.truncated = truncated
        self.restarts = 0
        self.sent: list[str] = []
        self.closed = False

    def __enter__(self) -> SqliteBridge:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True

    # Reply the way analyst_bridge does; its LIKE is case-sensitive too.
    def query(self, sql: str) -> dict:
        self.sent.append(sql)
        rid = str(len(self.sent))
        with closing(sqlite3.connect(f"{self.db.resolve().as_uri()}?mode=ro", uri=True)) as conn:
            conn.execute("PRAGMA case_sensitive_like = ON")
            try:
                cur = conn.execute(mod.an.to_sqlite(sql))
                rows = [list(r) for r in cur.fetchall()]
            except sqlite3.Error as e:
                return {"id": rid, "ok": False, "error": {"kind": "parse", "message": str(e)}, "ms": 0}
            cols = [{"name": d[0], "type": "INTEGER"} for d in cur.description]
        return {"id": rid, "ok": True, "columns": cols, "rows": rows, "row_count": len(rows),
                "truncated": self.truncated, "profile": {}, "ms": 0}


class Factory:
    """A bridge_factory that keeps every bridge it made."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.made: list[SqliteBridge] = []

    def __call__(self, tables_tsv: Path) -> SqliteBridge:
        self.made.append(SqliteBridge(tables_tsv, **self.kwargs))
        return self.made[-1]


# One question record; gold None makes it unanswerable.
def q(qid: str, gold: str | None, subset=SAME, **extra) -> dict:
    rec = {"qid": qid, "dataset": "tpch" if qid[0] == "t" else "reviews",
           "kind": "single" if gold else "unanswerable", "question": f"Question {qid}?",
           "gold_sqlite": gold, "gold_subset": gold if subset is SAME else subset,
           "ordered": False, "share_cols": [], "order_cols": [],
           "unanswerable_reason": None if gold else "the data doesn't have it"}
    rec.update(extra)
    return rec


# The second writer's record, with the question it answered; sql None means unanswerable.
def second(qid: str, sql: str | None, question: str | None = None) -> dict:
    return {"qid": qid, "question": question or f"Question {qid}?", "sqlite": sql, "unanswerable": sql is None}


def _write_sqlite(db: Path, statements: list[str]) -> None:
    with closing(sqlite3.connect(db)) as conn:
        for s in statements:
            conn.execute(s)
        conn.commit()


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


# A results folder with both tiny datasets, the questions and the second answers.
def make_results(tmp_path: Path, questions: list[dict], seconds: list[dict]) -> Path:
    results = tmp_path / f"results_{os.getpid()}"
    for ds, statements in TABLES.items():
        d = results / "datasets" / ds
        d.mkdir(parents=True)
        _write_sqlite(d / f"{ds}.sqlite", statements)
        (d / "t.tbl").write_text("1\n2\n")
        (d / "manifest.json").write_text(json.dumps({"dataset": ds, "tables": [{"name": "t", "file": "t.tbl"}]}))
        (d / "tables.tsv").write_text("t\tt.tbl\tx:INTEGER\n")
    _write_jsonl(results / "questions.jsonl", questions)
    _write_jsonl(results / "second_answers.jsonl", seconds)
    return results


def statuses(rows: list[dict]) -> dict:
    return {r["qid"]: r["status"] for r in rows}


def details(rows: list[dict]) -> dict:
    return {r["qid"]: r["detail"] for r in rows}


@pytest.fixture(autouse=True)
def no_examples(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "EXAMPLES", tmp_path / f"no_examples_{os.getpid()}.json")


# 100 questions per dataset in the KINDS mix, all agreed by both writers.
def full_set() -> tuple[list[dict], list[dict]]:
    qs, seconds = [], []
    for prefix in ("t", "r"):
        kinds = [kind for kind, n in mod.qm.KINDS.items() for _ in range(n)]
        for i, kind in enumerate(kinds, 1):
            qid = f"{prefix}{i:03d}"
            sql = f"SELECT {i} AS n" if kind != "unanswerable" else None
            qs.append(q(qid, sql, kind=kind))
            seconds.append(second(qid, sql))
    return qs, seconds


def test_key_statuses(tmp_path):
    tie = "SELECT name, score FROM s ORDER BY score DESC LIMIT 2"
    questions = [
        q("t001", "SELECT COUNT(*) AS n FROM s"),
        q("t002", "SELECT SUM(score) AS total FROM s"),
        q("t003", "SELECT COUNT(*) AS n FROM s WHERE score > 100"),
        q("t004", None),
        q("t005", "SELECT nope FROM s"),
        q("t006", "SELECT COUNT(*) AS n FROM s WHERE score > 100", empty_ok=True),
        q("t007", tie, kind="topn", ordered=True, order_cols=[1]),
    ]
    seconds = [second("t001", "SELECT COUNT(name) FROM s"), second("t002", "SELECT AVG(score) FROM s"),
               second("t003", "SELECT COUNT(*) FROM s WHERE score > 100"), second("t004", "SELECT COUNT(*) FROM s"),
               second("t005", "SELECT COUNT(*) FROM s"), second("t006", "SELECT 0 AS n"), second("t007", tie)]
    results = make_results(tmp_path, questions, seconds)
    factory = Factory()

    rows = mod.check(results, bridge_factory=factory)
    assert statuses(rows) == {"t001": "AGREE", "t002": "WRITERS_DIFFER", "t003": "EMPTY", "t004": "UNANSWERABLE_DIFFER",
                              "t005": "GOLD_ERROR", "t006": "AGREE", "t007": "TIE"}
    assert [r["qid"] for r in rows] == [x["qid"] for x in questions]
    assert "no such column" in details(rows)["t005"]

    # one bridge, for tpch only, closed at the end
    assert [b.tables_tsv for b in factory.made] == [results / "datasets/tpch/tables.tsv"]
    assert factory.made[0].closed

    log = (results / "key_log.md").read_text()
    assert "| WRITERS_DIFFER | 1 |" in log and "| AGREE | 2 |" in log and "| ENGINES_DIFFER | 0 |" in log
    assert "## t002" in log and "SELECT AVG(score) FROM s" in log and "SELECT SUM(score) AS total FROM s" in log
    assert "## t001" not in log and "## t006" not in log
    assert "[70]" in log                                             # the gold's first row


def test_like_is_case_sensitive(tmp_path):
    rerun = "SELECT n_name, n_regionkey FROM nation WHERE n_name LIKE 'G%' ORDER BY n_regionkey DESC LIMIT 1"
    questions = [
        q("t001", "SELECT n_name FROM nation WHERE n_name LIKE 'germany'"),   # gold: no rows
        q("t002", "SELECT n_name FROM nation WHERE n_name LIKE 'GERMANY'"),
        q("t003", rerun, kind="topn", ordered=True, order_cols=[1]),          # 'georgia' would tie on rerun
    ]
    seconds = [second("t001", "SELECT n_name FROM nation WHERE n_name LIKE 'germany'"),
               second("t002", "SELECT n_name FROM nation WHERE n_name LIKE 'germany'"),
               second("t003", rerun)]
    results = make_results(tmp_path, questions, seconds)
    rows = mod.check(results, bridge_factory=Factory())
    assert statuses(rows) == {"t001": "EMPTY", "t002": "WRITERS_DIFFER", "t003": "AGREE"}


def test_engines_differ(tmp_path):
    questions = [
        q("t001", "SELECT COUNT(*) AS n FROM s", subset="SELECT COUNT(*) AS n FROM s WHERE score > 20"),
        q("t002", "SELECT name FROM s", subset="SELECT nope FROM s"),
        q("t003", "SELECT COUNT(*) AS n FROM s"),
    ]
    seconds = [second(x["qid"], x["gold_sqlite"]) for x in questions]
    results = make_results(tmp_path, questions, seconds)

    rows = mod.check(results, bridge_factory=Factory())
    assert statuses(rows) == {"t001": "ENGINES_DIFFER", "t002": "GOLD_ERROR", "t003": "AGREE"}
    assert "gold_subset" in details(rows)["t002"]
    assert "SELECT COUNT(*) AS n FROM s WHERE score > 20" in (results / "key_log.md").read_text()

    (results / "key_answers.json").unlink()                           # else the cached replies come back
    rows = mod.check(results, bridge_factory=Factory(truncated=True))
    assert statuses(rows)["t003"] == "ENGINES_DIFFER"
    assert "truncated at 10,000 rows" in details(rows)["t003"]


def test_second_missing(tmp_path):
    gold = "SELECT COUNT(*) AS n FROM s"
    questions = [q("t001", gold), q("t002", gold), q("t003", gold), q("t004", gold), q("t005", None),
                 q("t006", gold), q("t007", gold, question="How many rows are in s?"), q("t008", gold),
                 q("t009", gold, question="How many  rows\nare in s?")]
    seconds = [
        {"qid": "t002", "question": "Question t002?", "sqlite": gold},   # no unanswerable field
        {"qid": "t003", "question": "Question t003?", "sqlite": "SELECT nope FROM s", "unanswerable": False},
        {"qid": "t004", "question": "Question t004?", "sqlite": None, "unanswerable": False},  # no SQL
        {"qid": "t005", "question": "Question t005?", "sqlite": "SELECT 1", "unanswerable": True},  # says both
        second("t006", "SELECT SUM(score) FROM s"),
        second("t006", gold),                                             # a later record wins
        second("t007", gold, question="How many rows does s have?"),      # answered an older wording
        {"qid": "t008", "sqlite": gold, "unanswerable": False},           # no question
        second("t009", gold, question=" How many rows are in s? "),       # whitespace only: same wording
    ]
    results = make_results(tmp_path, questions, seconds)
    rows = mod.check(results, bridge_factory=Factory())
    assert statuses(rows) == {"t001": "SECOND_MISSING", "t002": "SECOND_MISSING", "t003": "WRITERS_DIFFER",
                              "t004": "SECOND_MISSING", "t005": "SECOND_MISSING", "t006": "AGREE",
                              "t007": "SECOND_MISSING", "t008": "SECOND_MISSING", "t009": "AGREE"}
    assert "no such column" in details(rows)["t003"]
    assert details(rows)["t007"] == "answered an older wording"


def test_example_overlap(tmp_path):
    examples = tmp_path / f"examples_{os.getpid()}.json"
    examples.write_text(json.dumps({
        "tpch": [{"question": "  how MANY rows\tare in s? ", "reply": {"sql": "SELECT 1"}}],
        "reviews": [{"question": "What does it cost to ship?", "reply": {"cannot_answer": "no shipping data"}}],
    }))
    questions = [q("t001", "SELECT COUNT(*) AS n FROM s", question="How many rows are in s?"),
                 q("t002", None, question="What does it cost to ship?"),
                 q("t003", None)]
    seconds = [second(x["qid"], x["gold_sqlite"], x["question"]) for x in questions]
    results = make_results(tmp_path, questions, seconds)

    rows = mod.check(results, bridge_factory=Factory(), examples_path=examples)
    assert statuses(rows) == {"t001": "EXAMPLE_OVERLAP", "t002": "EXAMPLE_OVERLAP", "t003": "AGREE"}
    rows = mod.check(results, bridge_factory=Factory())               # no examples file: skipped
    assert statuses(rows) == {"t001": "AGREE", "t002": "AGREE", "t003": "AGREE"}


def test_normalize_sql():
    n = mod.normalize_sql
    month = "SELECT  r_month, AVG(r_rating) AS avg_rating\nFROM reviews WHERE r_year = 2022 GROUP BY r_month;"
    assert n(month) == "select r_month , avg ( r_rating ) from reviews where r_year = ? group by r_month"
    assert n("SELECT COUNT(*) FROM t WHERE n = 'it''s'") == n("select count( * ) from t where n='X'")
    assert n("SELECT x AS 'y', z as w FROM t") == n("SELECT x, z FROM t")
    dated = "SELECT 1.5 * a FROM t WHERE d >= DATE '1995-01-01' LIMIT 3 ;"
    assert n(dated) == "select ? * a from t where d >= date ? limit ?"
    assert n("SELECT p_has_price FROM t -- a note\nWHERE c1 <> 2") == "select p_has_price from t where c1 <> ?"
    assert n("SELECT c1 FROM t") != n("SELECT c2 FROM t")       # digits inside names stay
    assert n("SELECT a FROM t ORDER BY a") != n("SELECT a FROM t ORDER BY a DESC")


def test_example_overlap_on_gold_sql(tmp_path):
    examples = tmp_path / f"examples_{os.getpid()}.json"
    examples.write_text(json.dumps({
        "tpch": [{"question": "Who scored over 25?",
                  "reply": {"sql": "SELECT name, score AS points FROM s WHERE score > 25 ORDER BY score DESC;"}},
                 {"question": "What did shipping cost?", "reply": {"cannot_answer": "no shipping costs"}}],
        "reviews": [{"question": "Odd reply?", "reply": "not an object"}],
    }))
    sqlite_form = "SELECT name, score FROM s WHERE score >= 20"
    questions = [q("t001", "select name ,score as pts\nfrom s where score > 10 order by score desc"),
                 q("t002", sqlite_form, subset="SELECT name, score AS x FROM s WHERE score > 19 ORDER BY score DESC"),
                 q("t003", "SELECT name FROM s WHERE score > 25")]
    seconds = [second(x["qid"], x["gold_sqlite"], x["question"]) for x in questions]
    results = make_results(tmp_path, questions, seconds)

    rows = mod.check(results, bridge_factory=Factory(), examples_path=examples)
    assert statuses(rows) == {"t001": "EXAMPLE_OVERLAP", "t002": "EXAMPLE_OVERLAP", "t003": "AGREE"}
    assert details(rows)["t001"] == "the gold SQL matches a prompt example's SQL"


def test_missing_examples_file_says_so(tmp_path, capsys):
    questions = [q("t001", "SELECT name FROM s WHERE score > 25")]
    results = make_results(tmp_path, questions, [second("t001", questions[0]["gold_sqlite"])])
    assert statuses(mod.check(results, bridge_factory=Factory())) == {"t001": "AGREE"}
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and f"no_examples_{os.getpid()}.json" in err and "EXAMPLE_OVERLAP" in err


def test_tie_rules(tmp_path):
    questions = [
        q("t001", "SELECT name, score FROM s ORDER BY score DESC LIMIT 1", ordered=True, order_cols=[1]),
        q("t002", "SELECT name, score FROM s ORDER BY score DESC, name limit 2 ;", ordered=True, order_cols=[1, 0]),
        q("t003", "SELECT name, score FROM s ORDER BY score DESC limit 2;\n", ordered=True, order_cols=[1]),
        q("t004", "SELECT name, score FROM s ORDER BY score DESC LIMIT 2 -- top two", ordered=True, order_cols=[1]),
        q("t005", "SELECT name, score FROM s\nORDER BY score DESC\nLIMIT 2; -- top two\n", ordered=True,
          order_cols=[1]),
        q("t006", "SELECT name, score FROM s WHERE name <> 'x LIMIT 9' ORDER BY score DESC LIMIT 2", ordered=True,
          order_cols=[1]),                                                # a LIMIT in a string doesn't count
    ]
    seconds = [second(x["qid"], x["gold_sqlite"]) for x in questions]
    results = make_results(tmp_path, questions, seconds)
    rows = mod.check(results, bridge_factory=Factory())
    # t001: 20 doesn't tie 30; t002: name breaks the tie
    assert statuses(rows) == {"t001": "AGREE", "t002": "AGREE", "t003": "TIE", "t004": "TIE", "t005": "TIE",
                              "t006": "TIE"}


def test_limit_rules(tmp_path):
    top = "SELECT name, score FROM s ORDER BY score DESC"
    questions = [
        q("t001", f"{top} LIMIT 2 OFFSET 0", ordered=True, order_cols=[1]),
        q("t002", f"{top} LIMIT 0, 2", ordered=True, order_cols=[1]),
        q("t003", f"{top} LIMIT 2 /* top two */", ordered=True, order_cols=[1]),
        q("t004", f"SELECT name, score FROM ({top} LIMIT 2) ORDER BY score DESC", ordered=True, order_cols=[1]),
        q("t005", f"SELECT name, score FROM s WHERE score <= (SELECT MAX(score) FROM s LIMIT 1) "
                  f"ORDER BY score DESC LIMIT 1", ordered=True, order_cols=[1]),
        q("t006", "SELECT name FROM s ORDER BY score DESC LIMIT 2"),     # a top-N marked unordered
        q("t007", "SELECT name FROM s", subset="SELECT name FROM s LIMIT 5"),
        q("t008", "SELECT name FROM s WHERE name <> 'LIMIT 5' -- no LIMIT here"),
    ]
    seconds = [second(x["qid"], x["gold_sqlite"]) for x in questions]
    results = make_results(tmp_path, questions, seconds)
    rows = mod.check(results, bridge_factory=Factory())
    assert statuses(rows) == {**{f"t00{i}": "GOLD_ERROR" for i in range(1, 8)}, "t008": "AGREE"}
    d = details(rows)
    rerun = "use one trailing LIMIT n; the tie check can't rerun this LIMIT"
    assert all(rerun in d[f"t00{i}"] for i in range(1, 6))
    assert "a LIMIT needs ordered: true" in d["t006"] and "a LIMIT needs ordered: true" in d["t007"]
    assert rerun not in d["t007"]                                         # only gold_sqlite must have it trailing


def test_cols_out_of_range_and_order_by(tmp_path):
    questions = [
        q("t001", "SELECT name, score FROM s ORDER BY score DESC, name", ordered=True, order_cols=[2]),
        q("t002", "SELECT COUNT(*) AS n FROM s", share_cols=[1]),
        q("t003", "SELECT name FROM s", ordered=True, order_cols=[0]),
        q("t004", "SELECT name FROM s ORDER BY name", subset="SELECT name FROM s", ordered=True, order_cols=[0]),
        q("t005", "SELECT name FROM s ORDER BY name", ordered=True, order_cols=[0]),
        q("t006", "SELECT n_name, n_regionkey FROM nation ORDER BY n_name", ordered=True, order_cols=[1, 0]),
        q("t007", "SELECT n_regionkey, n_name FROM nation ORDER BY n_regionkey DESC, n_name", ordered=True,
          order_cols=[0, 1]),
        q("t008", "SELECT CASE WHEN score > 25 THEN 'top' ELSE score END AS v FROM s ORDER BY score DESC",
          ordered=True, order_cols=[0]),
        q("t009", "SELECT name FROM s -- ORDER BY name", ordered=True, order_cols=[0]),
    ]
    seconds = [second(x["qid"], x["gold_sqlite"]) for x in questions]
    results = make_results(tmp_path, questions, seconds)
    rows = mod.check(results, bridge_factory=Factory())
    assert statuses(rows) == {"t001": "GOLD_ERROR", "t002": "GOLD_ERROR", "t003": "GOLD_ERROR",
                              "t004": "GOLD_ERROR", "t005": "AGREE", "t006": "GOLD_ERROR", "t007": "AGREE",
                              "t008": "GOLD_ERROR", "t009": "GOLD_ERROR"}
    d = details(rows)
    assert "order_cols" in d["t001"] and "share_cols" in d["t002"]
    assert "ORDER BY" in d["t003"] and "gold_subset" in d["t004"] and "ORDER BY" in d["t009"]
    assert "rows aren't sorted on order_cols" in d["t006"] and "rows aren't sorted on order_cols" in d["t008"]


def test_empty_rules(tmp_path):
    questions = [
        q("t001", "SELECT name FROM s WHERE score = 30"),                       # text only: not empty
        q("t002", "SELECT name, score - 30 AS d FROM s WHERE score = 30"),      # the one number is 0
        q("t003", "SELECT SUM(score) AS t FROM s WHERE score > 100"),           # NULL
        q("t004", "SELECT SUM(score) AS t FROM s WHERE score > 100", empty_ok=True),
        q("t005", "SELECT name FROM s WHERE score > 100"),                      # no rows
    ]
    seconds = [second(x["qid"], x["gold_sqlite"]) for x in questions]
    results = make_results(tmp_path, questions, seconds)
    rows = mod.check(results, bridge_factory=Factory())
    assert statuses(rows) == {"t001": "AGREE", "t002": "EMPTY", "t003": "EMPTY", "t004": "AGREE", "t005": "EMPTY"}


def test_bridge_opened_only_when_needed(tmp_path):
    results = make_results(tmp_path, [q("t001", None), q("r001", None)], [second("t001", None), second("r001", None)])
    factory = Factory()
    assert statuses(mod.check(results, bridge_factory=factory)) == {"t001": "AGREE", "r001": "AGREE"}
    assert factory.made == []


def test_cache_reuse_and_invalidation(tmp_path, monkeypatch):
    gold = "SELECT COUNT(*) AS n FROM s"
    results = make_results(tmp_path, [q("t001", gold)], [second("t001", gold)])
    tpch = results / "datasets/tpch"
    first = Factory()
    assert statuses(mod.check(results, bridge_factory=first)) == {"t001": "AGREE"}
    assert len(first.made) == 1
    assert len(json.loads((results / "key_answers.json").read_text())["entries"]) == 2   # one per engine

    real_query = mod._sqlite_query
    sqlite_runs = []

    def counted(db, sql):
        sqlite_runs.append(sql)
        return real_query(db, sql)

    monkeypatch.setattr(mod, "_sqlite_query", counted)

    # Run check; return its statuses, bridges opened and SQLite queries run.
    def rerun():
        sqlite_runs.clear()
        factory = Factory()
        return statuses(mod.check(results, bridge_factory=factory)), len(factory.made), len(sqlite_runs)

    assert rerun() == ({"t001": "AGREE"}, 0, 0)                                        # nothing changed: all cached

    (tpch / "t.tbl").write_text("1\n3\n")                                     # a .tbl changes, manifest doesn't
    assert rerun() == ({"t001": "AGREE"}, 1, 1)
    assert rerun() == ({"t001": "AGREE"}, 0, 0)

    _write_sqlite(tpch / "tpch.sqlite", ["DELETE FROM s"])                             # the .sqlite changes
    assert rerun() == ({"t001": "EMPTY"}, 1, 1)                                         # rerun: count 0

    manifest = json.loads((tpch / "manifest.json").read_text())
    (tpch / "manifest.json").write_text(json.dumps({**manifest, "v": 2}))              # the manifest changes
    assert rerun() == ({"t001": "EMPTY"}, 1, 1)


def test_missing_listed_tbl(tmp_path):
    results = make_results(tmp_path, [q("t001", "SELECT 1 AS n")], [second("t001", "SELECT 1 AS n")])
    (results / "datasets/tpch/t.tbl").unlink()
    with pytest.raises(FileNotFoundError, match="t.tbl"):
        mod.check(results, bridge_factory=Factory())


def test_timeouts_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "TIMEOUT_S", 0.05)
    loop = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT MAX(x) AS m FROM c"
    results = make_results(tmp_path, [q("t001", loop, subset="SELECT 1 AS m")], [second("t001", "SELECT 1 AS m")])
    rows = mod.check(results, bridge_factory=Factory())
    assert statuses(rows) == {"t001": "GOLD_ERROR"} and "timeout" in details(rows)["t001"]
    assert "RECURSIVE" not in (results / "key_answers.json").read_text()


class FailingBridge(SqliteBridge):
    """Fails every query with one error kind, the way the client reports a stopped bridge."""

    def __init__(self, tables_tsv: Path, kind: str):
        super().__init__(tables_tsv)
        self.kind = kind

    def query(self, sql: str) -> dict:
        self.sent.append(sql)
        return {"id": str(len(self.sent)), "ok": False, "error": {"kind": self.kind, "message": f"bad {self.kind}"},
                "ms": 0}


@pytest.mark.parametrize("kind", ["timeout", "crash", "memory"])
def test_engine_failures_that_may_not_repeat_are_not_cached(tmp_path, kind):
    gold = "SELECT COUNT(*) AS n FROM s"
    results = make_results(tmp_path, [q("t001", gold)], [second("t001", gold)])
    rows = mod.check(results, bridge_factory=lambda tsv: FailingBridge(tsv, kind))
    assert statuses(rows) == {"t001": "GOLD_ERROR"} and kind in details(rows)["t001"]
    entries = json.loads((results / "key_answers.json").read_text())["entries"].values()
    assert [e["engine"] for e in entries] == ["sqlite"]   # only SQLite's run is kept


def test_check_refuses_bad_records_and_missing_files(tmp_path):
    bad = q("t001", "SELECT COUNT(*) AS n FROM s")
    bad["order_col"] = bad.pop("order_cols")
    results = make_results(tmp_path, [bad], [second("t001", "SELECT COUNT(*) AS n FROM s")])
    with pytest.raises(ValueError, match="order_col"):
        mod.check(results, bridge_factory=Factory())

    _write_jsonl(results / "questions.jsonl", [q("t001", "SELECT COUNT(*) AS n FROM s")])
    (results / "datasets/tpch/tpch.sqlite").unlink()
    with pytest.raises(FileNotFoundError, match="tpch.sqlite"):
        mod.check(results, bridge_factory=Factory())


def test_cli_validate_problem_stops(tmp_path, monkeypatch, capsys):
    qs, seconds = full_set()
    qs[0]["order_col"] = qs[0].pop("order_cols")
    results = make_results(tmp_path, qs, seconds)

    def no_sql(*args, **kwargs):
        raise AssertionError("no SQL should run")

    monkeypatch.setattr(mod.sqlite3, "connect", no_sql)
    for command in ("check", "freeze"):
        assert mod.main([command, "--results", str(results)], bridge_factory=no_sql) == 1
    assert "order_col" in capsys.readouterr().err
    assert not (results / "key_log.md").exists() and not (results / "key_answers.json").exists()


def test_cli_first_set_needs_full_mix(tmp_path, capsys):
    results = make_results(tmp_path, [q("t001", "SELECT 1 AS n")], [second("t001", "SELECT 1 AS n")])
    assert mod.main(["check", "--results", str(results)], bridge_factory=Factory()) == 1
    assert "single" in capsys.readouterr().err
    assert mod.main(["check", "--results", str(tmp_path / "nowhere")], bridge_factory=Factory()) == 1


def test_freeze_refuses_when_unresolved(tmp_path):
    qs, seconds = full_set()
    seconds[1] = second("t002", "SELECT 99 AS n")
    results = make_results(tmp_path, qs, seconds)
    before = (results / "questions.jsonl").read_bytes()

    assert mod.main(["check", "--results", str(results)], bridge_factory=Factory()) == 1
    assert mod.main(["freeze", "--results", str(results)], bridge_factory=Factory()) == 1
    assert (results / "questions.jsonl").read_bytes() == before
    assert "## t002" in (results / "key_log.md").read_text()

    rows = mod.freeze(results, bridge_factory=Factory())
    assert statuses(rows)["t002"] == "WRITERS_DIFFER"
    assert (results / "questions.jsonl").read_bytes() == before


def test_freeze_writes_answers_and_split(tmp_path):
    qs, seconds = full_set()
    results = make_results(tmp_path, qs, seconds)
    assert mod.main(["freeze", "--results", str(results)], bridge_factory=Factory()) == 0

    frozen = mod.qm.read_questions(results / "questions.jsonl")
    by_qid = {x["qid"]: x for x in frozen}
    assert [x["qid"] for x in frozen] == [x["qid"] for x in qs]
    assert by_qid["t001"]["answer"] == {"columns": ["n"], "rows": [[1]]}
    assert by_qid["r090"]["answer"] == {"columns": ["n"], "rows": [[90]]}
    assert by_qid["t091"]["answer"] is None                           # unanswerable
    dev = [x for x in frozen if x["split"] == "dev"]
    assert len(dev) == 60 and sum(x["dataset"] == "tpch" for x in dev) == 30

    # a later check validates as a grown set and runs nothing new
    factory = Factory()
    assert mod.main(["check", "--results", str(results)], bridge_factory=factory) == 0
    assert factory.made == []

    # a question added later is test; the old splits stay
    grown = frozen + [q("t101", "SELECT 101 AS n")]
    _write_jsonl(results / "questions.jsonl", grown)
    _write_jsonl(results / "second_answers.jsonl", seconds + [second("t101", "SELECT 101 AS n")])
    assert mod.main(["freeze", "--results", str(results)], bridge_factory=Factory()) == 0
    regrown = {x["qid"]: x for x in mod.qm.read_questions(results / "questions.jsonl")}
    assert regrown["t101"]["split"] == "test" and regrown["t101"]["answer"] == {"columns": ["n"], "rows": [[101]]}
    assert all(regrown[k]["split"] == v["split"] for k, v in by_qid.items())


@pytest.mark.parametrize("data", [
    {"tpch": [{"q": "Question t003?"}]},
    {"tpch": ["Question t003?"]},
    {"tpch": [{"question": 3}]},
    {"tpch": {"question": "Question t003?"}},
    [{"question": "Question t003?"}],
])
def test_malformed_examples_raise(tmp_path, monkeypatch, capsys, data):
    examples = tmp_path / f"bad_examples_{os.getpid()}.json"
    examples.write_text(json.dumps(data))
    qs, seconds = full_set()
    results = make_results(tmp_path, qs, seconds)
    with pytest.raises(ValueError, match="bad_examples_"):
        mod.check(results, bridge_factory=Factory(), examples_path=examples)

    monkeypatch.setattr(mod, "EXAMPLES", examples)
    factory = Factory()
    assert mod.main(["check", "--results", str(results)], bridge_factory=factory) == 1
    assert "bad_examples_" in capsys.readouterr().err and factory.made == []


def test_tie_note_on_earlier_status(tmp_path):
    top2 = "SELECT name, score FROM s ORDER BY score DESC LIMIT 2"   # rows 2 and 3 tie on score
    top1 = "SELECT name, score FROM s ORDER BY score DESC LIMIT 1"   # 30 then 20: no tie
    plus_one = "SELECT name, score + 1 AS score FROM s ORDER BY score DESC LIMIT 2"
    questions = [q("t001", top2, ordered=True, order_cols=[1]),
                 q("t002", top1, ordered=True, order_cols=[1]),
                 q("t003", top2, subset=plus_one, ordered=True, order_cols=[1]),
                 q("t004", top2, ordered=True, order_cols=[1])]
    seconds = [second("t001", plus_one), second("t002", plus_one), second("t003", top2), second("t004", top2)]
    results = make_results(tmp_path, questions, seconds)
    rows = mod.check(results, bridge_factory=Factory())
    assert statuses(rows) == {"t001": "WRITERS_DIFFER", "t002": "WRITERS_DIFFER", "t003": "ENGINES_DIFFER",
                              "t004": "TIE"}
    d = details(rows)
    note = "; the cut also falls on a tie"
    assert d["t001"].endswith(note) and d["t003"].endswith(note)
    assert note not in d["t002"] and note not in d["t004"]
    assert note in (results / "key_log.md").read_text()


def test_rerun_keeps_raw_sql(tmp_path):
    plain = "SELECT n_name, n_regionkey FROM nation ORDER BY n_regionkey DESC LIMIT 1"   # GERMANY, georgia tie
    alias = ('SELECT n_name AS "nation\'s name", n_regionkey\nFROM nation -- every nation\n'
             'ORDER BY n_regionkey DESC\nLIMIT 1')
    header = "/* each region's first nation */\n" + plain
    tab = "SELECT name, score FROM s WHERE name <> 'a\tb' ORDER BY score DESC LIMIT 2"
    student = 'SELECT name AS "student\'s name" FROM s ORDER BY score DESC LIMIT 2 -- the class\'s top two'
    questions = [q("t001", alias, subset=plain, ordered=True, order_cols=[1]),
                 q("t002", header, subset=plain, ordered=True, order_cols=[1]),
                 q("t003", tab, ordered=True, order_cols=[1]),
                 q("t004", student)]                                       # a top-N marked unordered
    seconds = [second(x["qid"], x["gold_sqlite"]) for x in questions]
    results = make_results(tmp_path, questions, seconds)
    rows = mod.check(results, bridge_factory=Factory())
    assert statuses(rows) == {"t001": "TIE", "t002": "TIE", "t003": "TIE", "t004": "GOLD_ERROR"}
    assert "a LIMIT needs ordered: true" in details(rows)["t004"]

    # only the LIMIT's number changes; tabs, comments and quoted names stay
    assert mod._rerun_sql(questions[0]) == (alias[:-1] + "2", 1)
    assert mod._rerun_sql(questions[2]) == (tab.replace("LIMIT 2", "LIMIT 3"), 2)


def test_rerun_must_reproduce_gold(tmp_path, monkeypatch):
    gold = "SELECT name, score FROM s ORDER BY score DESC LIMIT 2"
    real_query = mod._sqlite_query

    # The LIMIT 3 rerun comes back in the wrong order.
    def broken(db, sql):
        result = real_query(db, sql)
        return dict(result, rows=result["rows"][::-1]) if "LIMIT 3" in sql else result

    monkeypatch.setattr(mod, "_sqlite_query", broken)
    results = make_results(tmp_path, [q("t001", gold, ordered=True, order_cols=[1])], [second("t001", gold)])
    rows = mod.check(results, bridge_factory=Factory())
    assert statuses(rows) == {"t001": "GOLD_ERROR"}
    assert "the tie check's rerun didn't reproduce the gold" in details(rows)["t001"]
    assert "falls on a tie" not in details(rows)["t001"]


def test_sort_check_nulls(tmp_path):
    questions = [
        q("t001", "SELECT x FROM v ORDER BY x", ordered=True, order_cols=[0]),             # NULL, -5, -1, 3
        q("t002", "SELECT x FROM v ORDER BY x DESC", ordered=True, order_cols=[0]),        # 3, -1, -5, NULL
        q("t003", "SELECT label, x FROM v ORDER BY label", ordered=True, order_cols=[0]),  # NULL, then text
        q("t004", "SELECT a, b FROM w ORDER BY a, b", ordered=True, order_cols=[0, 1]),    # NULL isn't 0
    ]
    seconds = [second(x["qid"], x["gold_sqlite"]) for x in questions]
    results = make_results(tmp_path, questions, seconds)
    rows = mod.check(results, bridge_factory=Factory())
    assert statuses(rows) == {"t001": "AGREE", "t002": "AGREE", "t003": "AGREE", "t004": "AGREE"}
