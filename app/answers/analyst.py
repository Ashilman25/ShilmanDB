from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from app.core import analyst_check
from app.core.datasets import Dataset

an, ck, pr, rp, sf = analyst_check.an, analyst_check.ck, analyst_check.pr, analyst_check.rp, analyst_check.sf

PENDING = {"final": "pending"}
GUARD_MESSAGES = {False: "the SQL couldn't be read and mentions an outer join", True: "the SQL mentions an outer join"}
OUTER_JOIN = re.compile(r"\b(?:LEFT|RIGHT|FULL)(?:\s+OUTER)?\s+JOIN\b|\bOUTER\s+JOIN\b", re.IGNORECASE)


# Where a job is: its stage, the record so far and the checks on it.
@dataclass
class Event:
    stage: str
    record: dict
    checks: list


# Whether the SQL says LEFT, RIGHT, FULL or OUTER right before JOIN, in any case.
def has_outer_join_words(sql: str) -> bool:
    return OUTER_JOIN.search(sql) is not None


# The checks with engine_traps refusing when the SQL mentions an outer join, read or not.
# A refusal already there (a LEFT JOIN the checks read) stays as it is.
def apply_guard(checks: list, sql: str, manifest: dict) -> list:
    if not has_outer_join_words(sql) or any(c.name == "engine_traps" and c.level == "refuse" for c in checks):
        return list(checks)

    readable = sf.facts(sql, manifest).ok
    guard = ck.CheckResult("engine_traps", "refuse", GUARD_MESSAGES[readable], {"guard": True, "readable": readable})
    return [guard if c.name == "engine_traps" else c for c in checks]


# The analyst check's checks on a shown answer, then the guard.
def _checks(record: dict, question: dict, ds: Dataset) -> list:
    checks = ck.run_checks(record, question, ds.spec, ds.manifest)
    return apply_guard(checks, record["primary"]["sql"], ds.manifest)


# The analyst check's pipeline in two halves: writing, first (second answer pending), done;
# done alone for a refused first answer; or cannot / failed. Nothing runs until its event is pulled.
def analyst_events(question: str, ds: Dataset, model, bridge, asked: dict | None = None) -> Iterator[Event]:
    yield Event("writing", {}, [])

    ctx = ds.context
    restarts = bridge.restarts
    base = pr.build_sql_prompt(ctx.description, ctx.rules, ctx.examples, question)
    primary = rp.run_attempts(model, base, bridge)
    skipped = rp._skipped()
    record = {
        "qid": "app", "dataset": ds.name, "split": "app", "model": model.config_id,
        "prompt_sha1": an.sha1_text(base), "primary": primary, "rephrase": skipped,
        "seconds_total": rp._seconds_total(primary, skipped), "bridge_restarts": bridge.restarts - restarts,
        "asked": asked,
    }
    if primary["final"] != "shown":
        yield Event(primary["final"], record, [])
        return

    # A re-ask is checked against the question as typed, so its added sentence cues no other term.
    checked = {"question": asked["question"] if asked else question}
    first = {**record, "rephrase": dict(PENDING)}
    checks = _checks(first, checked, ds)
    if ck.overall(checks) == "refused":
        yield Event("done", record, _checks(record, checked, ds))
        return

    yield Event("first", first, checks)

    rephrase = rp._rephrase(question, model, bridge, ctx)
    done = {
        **record, "rephrase": rephrase,
        "seconds_total": rp._seconds_total(primary, rephrase), "bridge_restarts": bridge.restarts - restarts,
    }
    yield Event("done", done, _checks(done, checked, ds))
