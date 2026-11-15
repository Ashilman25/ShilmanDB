"""Tests for app/analyst.py and app/sql_runner.py."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

# check_views is reached as a module: pytest collects any test* name imported into a test module.
from app import check_views, sixd
from app.analyst import PENDING, analyst_events, has_outer_join_words
from app.datasets import load_dataset
from app.sql_runner import sql_events
from app.tests.fakes import FakeBridge, FakeModel, bridge_error, bridge_ok, reply

an, ck, pr, rp = sixd.an, sixd.ck, sixd.pr, sixd.rp

QUESTION = "How many nations does each region hold?"
REPHRASED = "For every region, count its nations."
SQL = "SELECT r_name, COUNT(*) AS nations FROM nation JOIN region ON n_regionkey = r_regionkey GROUP BY r_name"
FEWER_SQL = SQL.replace("GROUP BY", "WHERE n_nationkey < 20 GROUP BY")
LEFT_SQL = SQL.replace(" JOIN ", " LEFT JOIN ")
OUTER_SQL = SQL.replace(" JOIN ", " LEFT OUTER JOIN ")
GRAIN = "one row per region"
COLUMNS = [{"name": "r_name", "meaning": "region name", "entity": "region"},
           {"name": "nations", "meaning": "nations in the region", "entity": "nation"}]

REGION_COLUMNS = [("r_name", "VARCHAR"), ("nations", "INTEGER")]
REGIONS = ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"]
BY_REGION = bridge_ok(REGION_COLUMNS, [[r, 5] for r in REGIONS])
FEWER = bridge_ok(REGION_COLUMNS, [[r, 4] for r in REGIONS])
REPLIES = {SQL: BY_REGION, FEWER_SQL: FEWER, LEFT_SQL: BY_REGION, OUTER_SQL: BY_REGION}

LATE_QUESTION = "How many line items arrived late in 1995?"
LATE_REPHRASED = "Count the line items that came in after their commit date, for 1995 orders."
LATE_SQL = ("SELECT COUNT(*) AS late_lines FROM lineitem JOIN orders ON l_orderkey = o_orderkey "
            "WHERE l_receiptdate > l_commitdate AND o_orderyear = 1995")
LATE_COLUMNS = [{"name": "late_lines", "meaning": "line items received after their commit date", "entity": "line item"}]

ENGINE_ONLY = [name for name in ck.CHECKS if name not in ("second_phrasing", "terms")]


# One table of a hand-written manifest, in the real manifest's shape.
def _table(name: str, rows: int, columns: str) -> dict:
    cols = [dict(zip(("name", "type"), c.split(":"))) for c in columns.split()]
    return {"name": name, "file": f"{name}.tbl", "rows": rows, "columns": cols}


# The fixture's two tables plus the customer, orders and lineitem columns these tests use.
WIDE_MANIFEST = {"dataset": "tpch", "tables": [
    _table("region", 5, "r_regionkey:BIGINT r_name:VARCHAR r_comment:VARCHAR"),
    _table("nation", 25, "n_nationkey:BIGINT n_name:VARCHAR n_regionkey:BIGINT n_comment:VARCHAR"),
    _table("customer", 150, "c_custkey:BIGINT c_name:VARCHAR c_nationkey:BIGINT"),
    _table("orders", 1500, "o_orderkey:BIGINT o_custkey:BIGINT o_orderdate:DATE o_orderyear:INTEGER o_ordermonth:INTEGER"),
    _table("lineitem", 6000, "l_orderkey:BIGINT l_linenumber:INTEGER l_shipdate:DATE l_commitdate:DATE "
                             "l_receiptdate:DATE l_shipyear:INTEGER l_receiptyear:INTEGER"),
]}


# The fixture dataset: region and nation only, the real spec and prompt examples.
@pytest.fixture
def ds(fixture_paths):
    return load_dataset(fixture_paths, "tpch")


# The fixture dataset with a manifest that also knows customer, orders and lineitem.
@pytest.fixture
def wide_ds(ds):
    return replace(ds, manifest=WIDE_MANIFEST)


# A script: the rephrase prompt gets the rewrite, then each question its SQL.
# The rephrase key comes first, since that prompt also holds the question.
def script(first_sql: str = SQL, second_sql: str = SQL, question: str = QUESTION, rephrased: str = REPHRASED,
           grain: str = GRAIN, columns: list | None = None) -> dict:
    columns = COLUMNS if columns is None else columns
    return {pr.REPHRASE: rephrased, question: reply(first_sql, grain, columns), rephrased: reply(second_sql, grain, columns)}


# Every event of one question, with the model and bridge it used.
def run(ds, model_script: dict, replies: dict = REPLIES, question: str = QUESTION, asked=None):
    model, bridge = FakeModel(model_script), FakeBridge(replies)
    return list(analyst_events(question, ds, model, bridge, asked)), model, bridge


# One check's result by name.
def check(checks: list, name: str):
    return next(c for c in checks if c.name == name)


# Whether any prompt asked for the question in other words.
def rephrase_asked(model: FakeModel) -> bool:
    return any(pr.REPHRASE in p for p in model.prompts)


def test_events_in_order(ds):
    events, model, bridge = run(ds, script())

    assert [e.stage for e in events] == ["writing", "first", "done"]
    assert events[0].record == {} and events[0].checks == []
    assert [c.name for c in events[-1].checks] == list(ck.CHECKS)
    assert check(events[-1].checks, "second_phrasing").level == "pass"
    assert bridge.sent == [SQL, SQL]
    for e in events:
        json.dumps(e.record)  # records stay plain JSON, as 6d's run files


def test_first_has_pending_second(ds):
    events, _, _ = run(ds, script())
    first = next(e for e in events if e.stage == "first")
    primary = first.record["primary"]

    assert check(first.checks, "second_phrasing").evidence["final"] == "pending"
    assert first.record["rephrase"] == PENDING
    assert first.record["seconds_total"] == rp._seconds_total(primary, rp._skipped())
    assert first.record["bridge_restarts"] == 0
    assert first.record.keys() == events[-1].record.keys()
    assert events[-1].record["rephrase"]["final"] == "shown"
    assert events[-1].record["seconds_total"] > first.record["seconds_total"]


def test_done_reruns_checks_with_rephrase(ds):
    events, _, bridge = run(ds, script(second_sql=FEWER_SQL))
    first, done = events[1], events[2]

    assert check(first.checks, "second_phrasing").level == "note"
    assert check(done.checks, "second_phrasing").level == "warn"
    assert check(done.checks, "second_phrasing").evidence["sql"] == FEWER_SQL
    assert bridge.sent == [SQL, FEWER_SQL]


def test_refused_first_skips_second(ds):
    events, model, bridge = run(ds, script(first_sql=LEFT_SQL))
    done = events[-1]
    traps = check(done.checks, "engine_traps")

    assert [e.stage for e in events] == ["writing", "done"]
    assert done.record["rephrase"]["final"] == "skipped"
    assert done.record["rephrase"] == rp._skipped()
    assert check(done.checks, "second_phrasing").evidence["final"] == "skipped"
    assert traps.level == "refuse" and traps.evidence["left_join"] is True
    assert ck.overall(done.checks) == "refused"
    assert not rephrase_asked(model)
    assert bridge.sent == [LEFT_SQL]  # 6d's path: the SQL runs, then the answer is withheld


def test_cannot_and_failed_end(ds):
    cannot = {QUESTION: json.dumps({"cannot_answer": "The data has no weather records."})}
    events, model, bridge = run(ds, cannot)
    assert [e.stage for e in events] == ["writing", "cannot"]
    assert events[-1].checks == []
    assert events[-1].record["primary"]["reason"] == "The data has no weather records."
    assert events[-1].record["rephrase"] == rp._skipped()
    assert bridge.sent == []

    events, model, bridge = run(ds, {QUESTION: "Here is my answer, but not as JSON."})
    assert [e.stage for e in events] == ["writing", "failed"]
    assert events[-1].checks == []
    assert len(model.prompts) == 1 + an.MAX_RETRIES
    assert events[-1].record["rephrase"] == rp._skipped()
    assert not rephrase_asked(model)


def test_record_matches_run_question(ds, monkeypatch):
    monkeypatch.setattr(rp.an, "sqlite_query", lambda *args, **kwargs: {})
    model_script = script(second_sql=FEWER_SQL)
    events, _, _ = run(ds, model_script)

    q = {"qid": "app", "dataset": "tpch", "split": "app", "kind": "single", "question": QUESTION}
    expected = rp.run_question(q, FakeModel(model_script), FakeBridge(REPLIES), Path("x.sqlite"), ds.context)
    assert expected.pop("sqlite_cause") == {}

    got = dict(events[-1].record)
    assert got.pop("asked") is None
    assert got == expected


def test_guard_hides_unreadable_outer_join(ds):
    assert not sixd.sf.facts(OUTER_SQL, ds.manifest).ok
    events, model, _ = run(ds, script(first_sql=OUTER_SQL))
    done = events[-1]
    traps = check(done.checks, "engine_traps")

    assert [e.stage for e in events] == ["writing", "done"]
    assert traps.level == "refuse" and traps.evidence == {"guard": True, "readable": False}
    assert traps.message == "the SQL couldn't be read and mentions an outer join"
    assert [c.name for c in done.checks] == list(ck.CHECKS)
    assert done.record["rephrase"]["final"] == "skipped"
    assert not rephrase_asked(model)


def test_stop_after_first_sends_no_rephrase(ds):
    model, bridge = FakeModel(script()), FakeBridge(REPLIES)
    events = analyst_events(QUESTION, ds, model, bridge)

    assert next(events).stage == "writing"
    assert model.prompts == []  # nothing asked until the next event is pulled
    assert next(events).stage == "first"
    events.close()

    assert len(model.prompts) == 1
    assert not rephrase_asked(model)
    assert bridge.sent == [SQL]


def test_asked_reading_kept(ds):
    text = check_views.reask_question(QUESTION, "date basis", "ship date")
    asked = {"term": "date basis", "reading": "ship date", "question": QUESTION}
    events, model, _ = run(ds, script(question=text), question=text, asked=asked)
    ctx = ds.context

    assert [e.stage for e in events] == ["writing", "first", "done"]
    assert all(e.record["asked"] == asked for e in events[1:])
    assert model.prompts[0] == pr.build_sql_prompt(ctx.description, ctx.rules, ctx.examples, text)
    assert model.prompts[1] == pr.build_rephrase_prompt(text)
    assert events[-1].record["prompt_sha1"] == an.sha1_text(model.prompts[0])


def test_reask_sentence_cues_no_other_term(wide_ds):
    late_script = script(LATE_SQL, LATE_SQL, LATE_QUESTION, LATE_REPHRASED, "one row", LATE_COLUMNS)
    replies = {LATE_SQL: bridge_ok([("late_lines", "INTEGER")], [[123]])}
    text = check_views.reask_question(LATE_QUESTION, "late shipment", "received after commit")
    asked = {"term": "late shipment", "reading": "received after commit", "question": LATE_QUESTION}

    plain, _, _ = run(wide_ds, late_script, replies, LATE_QUESTION)
    reasked, _, _ = run(wide_ds, late_script, replies, text, asked)

    # Date basis entry of the terms evidence in each run's last event.
    def date_basis(checks: list) -> dict:
        return next(t for t in check(checks, "terms").evidence["terms"] if t["term"] == "date basis")

    assert date_basis(reasked[-1].checks) == date_basis(plain[-1].checks)
    assert date_basis(reasked[-1].checks)["cued"] == []
    assert date_basis(reasked[-1].checks)["level"] == "note"

    # Read against the re-ask sentence, "received" would cue the receipt date.
    sentence = ck.run_checks(reasked[-1].record, {"question": text}, wide_ds.spec, wide_ds.manifest)
    assert date_basis(sentence)["cued"] == ["receipt date"]
    assert date_basis(sentence)["level"] == "warn"


def test_outer_join_words():
    for sql in ("a LEFT JOIN b", "a left outer join b", "a Right\n  Join b", "a FULL OUTER JOIN b", "a OUTER JOIN b"):
        assert has_outer_join_words(sql), sql
    for sql in ("a JOIN b", "a INNER JOIN b", "a LEFTJOIN b", "my_left JOIN b", "SELECT outer_flag FROM t"):
        assert not has_outer_join_words(sql), sql


def test_sql_left_join_never_sent(wide_ds):
    left = "SELECT c_name FROM customer LEFT JOIN orders ON c_custkey = o_custkey"
    bridge = FakeBridge({}, default=BY_REGION)
    events = list(sql_events(left, wide_ds, bridge))

    assert [e.stage for e in events] == ["not_run"]
    assert [c.name for c in events[0].checks] == ["engine_traps"]
    traps = events[0].checks[0]
    assert traps.level == "refuse" and traps.evidence["left_join"] is True
    assert traps.evidence["few_table_ons"] == []  # the trap itself, not a missing table
    assert events[0].record == {"sql": left, "result": None}
    assert bridge.sent == []

    # sql_facts can't read this one, so the outer-join guard stops it.
    right = "SELECT c_name FROM customer c RIGHT JOIN orders o ON c.c_custkey = o.o_custkey"
    assert not sixd.sf.facts(right, wide_ds.manifest).ok
    events = list(sql_events(right, wide_ds, bridge))
    assert [e.stage for e in events] == ["not_run"]
    assert [(c.name, c.level, c.evidence) for c in events[0].checks] == [
        ("engine_traps", "refuse", {"guard": True, "readable": False})]
    assert bridge.sent == []


def test_guard_withholds_readable_right_join(ds):
    right = SQL.replace(" JOIN ", " RIGHT JOIN ")
    assert sixd.sf.facts(right, ds.manifest).ok  # RIGHT reads as nation's alias
    events, model, bridge = run(ds, script(first_sql=right), {right: BY_REGION})
    done = events[-1]
    traps = check(done.checks, "engine_traps")

    assert [e.stage for e in events] == ["writing", "done"]
    assert traps.level == "refuse" and traps.evidence == {"guard": True, "readable": True}
    assert traps.message == "the SQL mentions an outer join"
    assert done.record["rephrase"]["final"] == "skipped"
    assert not rephrase_asked(model)
    assert bridge.sent == [right]  # 6d's path: the SQL runs, then the answer is withheld


def test_sql_readable_outer_joins_never_sent(wide_ds):
    bridge = FakeBridge({}, default=BY_REGION)
    for word in ("RIGHT", "FULL"):
        sql = f"SELECT c_name FROM customer {word} JOIN orders ON c_custkey = o_custkey"
        assert sixd.sf.facts(sql, wide_ds.manifest).ok, sql
        events = list(sql_events(sql, wide_ds, bridge))

        assert [e.stage for e in events] == ["not_run"], sql
        assert [(c.name, c.level, c.evidence) for c in events[0].checks] == [
            ("engine_traps", "refuse", {"guard": True, "readable": True})]
    assert bridge.sent == []


def test_readable_left_join_keeps_6d_refusal(ds, wide_ds):
    left = "SELECT c_name FROM customer LEFT JOIN orders ON c_custkey = o_custkey"
    events = list(sql_events(left, wide_ds, FakeBridge({})))
    record = {"primary": {"final": "shown", "sql": left, "result": {}, "claims": {}}, "rephrase": {}}
    own = check(ck.run_checks(record, {"question": ""}, wide_ds.spec, wide_ds.manifest), "engine_traps")
    assert events[0].checks == [own]
    assert own.evidence["left_join"] is True and "guard" not in own.evidence

    answer, _, _ = run(ds, script(first_sql=LEFT_SQL))
    traps = check(answer[-1].checks, "engine_traps")
    assert traps.evidence["left_join"] is True and "guard" not in traps.evidence
    assert traps.message == "the engine runs this wrong: LEFT JOIN runs as an inner join"


def test_sql_runner_engine_evidence_only(ds):
    sql = "SELECT n_name FROM nation WHERE n_regionkey = 1"
    result = bridge_ok([("n_name", "VARCHAR")], [["ARGENTINA"], ["BRAZIL"], ["CANADA"], ["PERU"], ["UNITED STATES"]])
    bridge = FakeBridge({sql: result})
    events = list(sql_events(sql, ds, bridge))

    assert [e.stage for e in events] == ["ran"]
    assert [c.name for c in events[0].checks] == ENGINE_ONLY
    assert check(events[0].checks, "engine_traps").level == "pass"
    assert check(events[0].checks, "empty_input").evidence["row_count"] == 5
    assert events[0].record["sql"] == sql
    assert events[0].record["result"]["rows"] == result["rows"]
    assert bridge.sent == [sql]


def test_sql_runner_engine_error_passed_through(ds):
    sql = "SELECT 1; SELECT 2"
    error = bridge_error("parse", "Expected end of input but got SELECT")
    bridge = FakeBridge({sql: error})
    events = list(sql_events(sql, ds, bridge))

    assert [e.stage for e in events] == ["ran"]
    result = events[0].record["result"]
    assert result["ok"] is False
    assert result["error"] == {"kind": "parse", "message": "Expected end of input but got SELECT"}
    assert [c.name for c in events[0].checks] == ["engine_traps"]
    assert bridge.sent == [sql]
