from __future__ import annotations

from collections.abc import Iterator

from app.answers.analyst import Event, apply_guard
from app.core import analyst_check
from app.core.datasets import Dataset

ck = analyst_check.ck

# Checks that need a question or a second answer, which typed SQL doesn't have.
NOT_FOR_SQL = ("second_phrasing", "terms")


# The analyst check's checks on typed SQL with this result (no claims, question or second answer), then the guard.
def _checks(sql: str, result: dict, ds: Dataset) -> list:
    record = {"primary": {"final": "shown", "sql": sql, "result": result, "claims": {}}, "rephrase": {}}
    return apply_guard(ck.run_checks(record, {"question": ""}, ds.spec, ds.manifest), sql, ds.manifest)


# The engine traps result for SQL that hasn't run yet, guard included.
def pre_run_traps(sql: str, ds: Dataset):
    return next(c for c in _checks(sql, {}, ds) if c.name == "engine_traps")


# Typed SQL: not_run when the engine would run it wrong (nothing sent), else ran with the engine's evidence.
# A failed run keeps only the engine traps result, since checks of an error reply mean nothing.
def sql_events(sql: str, ds: Dataset, bridge) -> Iterator[Event]:
    traps = pre_run_traps(sql, ds)
    if traps.level == "refuse":
        yield Event("not_run", {"sql": sql, "result": None}, [traps])
        return

    result = bridge.query(sql)
    record = {"sql": sql, "result": result}
    if result.get("ok") is not True:
        yield Event("ran", record, [traps])
        return
    yield Event("ran", record, [c for c in _checks(sql, result, ds) if c.name not in NOT_FOR_SQL])
