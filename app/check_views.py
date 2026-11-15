from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from app import sixd
from app.datasets import MODEL_ID

ck = sixd.ck

WORDS = {"pass": "Pass", "note": "Note", "warn": "Warning", "refuse": "Refused"}
SECOND_WORDS = {"pass": "Same", "warn": "Differs"}
RUNNING, NOT_RUN = "Running", "Not run"

PHASES = ("first", "done")
KINDS = ("ask", "reask", "sql", "edit")
SQL_KINDS = ("sql", "edit")
OUTCOMES = ("shown", "cannot", "failed", "not_run", "ran")

# The checks' own order with the second answer first; terms get their own rows, not a check row.
VERDICT_ORDER = ("second_phrasing", *(name for name in ck.CHECKS if name != "second_phrasing"))
ROW_ORDER = tuple(name for name in VERDICT_ORDER if name != "terms")

TRIES = sixd.an.MAX_RETRIES + 1

# The analyst check's three bars, fixed before the test: flag at least half the wrong answers,
# refuse at most 10% of the right ones, and warn on or refuse at most 25% of them.
CATCH_BAR, REFUSE_BAR, FLAG_BAR = 0.50, 0.10, 0.25

# Re-ask sentences for readings the plain sentence doesn't cue. The average of r_rating
# is only cued by words like "in this data", so its sentence says that too.
REASK_SENTENCES = {("product rating", "average of r_rating"): " Use the average of r_rating reading of product rating (the ratings in this data)."}

# Terms whose readings work as bare button labels ("Ship date").
BARE_LABEL_TERMS = ("date basis",)

PENDING_TEXT = "Verdict once the second answer is back."
REASK_LINE = "Re-asked with a reading named. Not part of the test, so its rates don't apply here."
UNREAD = "Not checked, because the SQL couldn't be read."
TRAP_FALLBACK = "The SQL uses something today's engine runs wrong."
GUARD_REASON = ("The SQL couldn't be checked, and it mentions an outer join, which today's engine runs as an inner join.")
LEFT_JOIN_REASON = ("The SQL uses LEFT JOIN, which today's engine runs as an inner join, so rows with no match would drop out without a warning.")


# Raises ValueError unless value is one of the allowed ones.
def _check(what: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"unknown {what} {value!r}, expected one of {', '.join(allowed)}")


# The text with its first letter in capitals.
def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


# Names as prose: "a", "a and b", "a, b and c".
def _join(names: Iterable) -> str:
    names = [str(n) for n in names]
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


# Whether a value is a row count.
def _is_count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# "1 row" or "1,234 rows".
def _rows(n: int) -> str:
    return "1 row" if n == 1 else f"{n:,} rows"


# A share as a whole percent, rounded the way Python formats.
def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "0%"


# The most serious of some levels; pass when there are none.
def _worst(levels: Iterable[str]) -> str:
    return max(levels, key=ck.LEVELS.index, default="pass")


# Whether a check stopped because the SQL couldn't be read.
def _unread(result) -> bool:
    return result.level == "note" and "error" in result.evidence


# The check's own message as a sentence, for a level its wording doesn't cover.
def _fallback(result) -> tuple[str, str | None]:
    return (_sentence(result.message) if result.message else WORDS.get(result.level, result.level)), None


# Plain reasons the engine would run this SQL wrong, from the engine traps evidence.
def trap_reason(result) -> str:
    evidence = result.evidence if result is not None else {}
    if evidence.get("guard"):
        return GUARD_REASON

    reasons = [LEFT_JOIN_REASON] if evidence.get("left_join") else []

    repeated = evidence.get("repeated_tables") or []
    if repeated:
        times = "twice" if len(repeated) == 1 else "twice each"
        reasons.append(f"The SQL joins {_join(repeated)} {times}, and today's engine ignores table qualifiers, so it would read the wrong columns.")

    constants = evidence.get("constant_conjuncts") or []
    if len(constants) == 1:
        reasons.append(f"The SQL's WHERE term {constants[0]} names no column, and today's engine drops a term like that, so the result would ignore it.")
    elif constants:
        reasons.append(f"The SQL's WHERE terms {_join(constants)} name no column, and today's engine drops terms like that, so the result would ignore them.")

    for on in evidence.get("few_table_ons") or []:
        tables = on.get("tables") or []
        names = f"only {tables[0]}" if tables else "no table"
        reasons.append(f"The SQL's ON term {on.get('on')} names {names}, and today's engine drops an ON term like that, so the join would ignore it.")

    if evidence.get("text_arithmetic"):
        reasons.append("The SQL does arithmetic on text or a date, which today's engine gets wrong without an error.")
    return " ".join(reasons) or TRAP_FALLBACK


# The engine traps row: the reason when it refuses.
def _engine_row(result) -> tuple[str, str | None]:
    if result.level == "refuse":
        return trap_reason(result), None
    if _unread(result):
        return "The engine checks couldn't read this SQL, so they didn't run.", f"Read error: {result.evidence['error']}"
    if result.level == "pass":
        return "Nothing in the SQL that today's engine runs wrong.", None
    return _fallback(result)


# Rows out of each scan and join step, as one line; None when there are none.
def _input_detail(evidence: dict) -> str | None:
    parts = []
    for key, label in (("scan_rows", "Rows out of each scan"), ("step_rows", "Rows after each join")):
        counts = [f"{n:,}" for n in evidence.get(key) or [] if _is_count(n)]
        if counts:
            parts.append(f"{label}: {', '.join(counts)}.")

    op = evidence.get("operator")
    if isinstance(op, dict) and _is_count(op.get("rows")):
        parts.append(f"{op.get('name')} kept {_rows(op['rows'])}.")
    return " ".join(parts) or None


# The empty input row.
def _empty_row(result) -> tuple[str, str | None]:
    if _unread(result):
        return UNREAD, None

    detail = _input_detail(result.evidence)
    if result.level == "warn" and result.evidence.get("row_count") == 0:
        return "The result is empty.", detail
    if result.level == "warn":
        return "An aggregate read no rows, so SUM, AVG, MIN and MAX show 0 where SQL would show NULL.", detail
    if result.level == "pass":
        return "Every input had rows.", detail
    return _fallback(result)


# Whether every fan-out problem is a COUNT(*), not a sum, average or count.
def _star_only(evidence: dict) -> bool:
    found = evidence.get("problems") or []
    return bool(found) and all("COUNT(*)" in p for p in found)


# Each repeating table's rows before and after the joins; None when unknown.
def _repeat_detail(evidence: dict) -> str | None:
    parts = []
    for table, rows in (evidence.get("repeating") or {}).items():
        scanned, joined = (rows or {}).get("scan_rows"), (rows or {}).get("joined_rows")
        if _is_count(scanned) and _is_count(joined):
            parts.append(f"rows of {table}: {scanned:,} scanned, {joined:,} after the joins")
    return _sentence("; ".join(parts)) + "." if parts else None


# The fan-out row.
def _fanout_row(result) -> tuple[str, str | None]:
    if _unread(result):
        return UNREAD, None

    evidence = result.evidence
    detail = _repeat_detail(evidence)
    if result.level == "warn":
        tables = _join(evidence.get("repeating") or {}) or "a table"
        if _star_only(evidence):
            return f"The joins repeat rows of {tables}, and COUNT(*) counts joined rows, so the count can be too high.", detail
        return (f"The joins repeat rows of {tables}, and a sum, average or count adds them up, so the total can count some rows more than once."), detail
    if result.level == "pass":
        return "No sum, average or count adds up rows the joins repeat.", detail
    return _fallback(result)


# The grain row.
def _grain_row(result) -> tuple[str, str | None]:
    if _unread(result):
        return UNREAD, None

    evidence = result.evidence
    group_by, grain = evidence.get("group_by") or [], evidence.get("grain")
    if result.level == "warn" and "repeated" in evidence:
        columns = evidence.get("columns") or group_by
        pairs = ", ".join(f"{c} = {v}" for c, v in zip(columns, evidence.get("repeated") or []))
        return f"Rows repeat on {_join(columns)}, so the result is split finer than it shows.", f"More than one row has {pairs}."
    
    if result.level == "warn" and grain:
        sets = " or ".join(f"({', '.join(cols)})" for cols in evidence.get("column_sets") or [])
        detail = f"One row per {grain} needs GROUP BY on {sets}." if sets else None
        return f"The answer says one row per {grain}, but GROUP BY {', '.join(group_by)} doesn't give that.", detail
    
    if result.level == "pass" and not group_by:
        return "No GROUP BY, so there's no grain to check.", None
    
    if result.level == "pass":
        return (f"GROUP BY gives one row per {grain}, as the answer says." if grain else "GROUP BY fits the stated grain."), None
    
    if result.level == "note" and evidence.get("claimed"):
        claimed = evidence["claimed"]
        return f"The stated grain, \"{claimed}\", isn't one these checks know, so GROUP BY wasn't compared with it.", None
    
    if result.level == "note":
        return "No stated grain to compare with GROUP BY.", None
    
    return _fallback(result)


# The integer division row.
def _division_row(result) -> tuple[str, str | None]:
    if _unread(result):
        return UNREAD, None

    divisions = result.evidence.get("divisions") or []
    if result.level == "warn":
        return "Dividing whole numbers drops the fraction here.", (f"Division: {'; '.join(divisions)}." if divisions else None)
    if result.level == "pass":
        return "No integer division.", None
    return _fallback(result)


# The exact row: every answer is a full scan.
def _exact_row(result) -> tuple[str, str | None]:
    return "Exact: a full scan, no sampling.", None


ROW_TEXTS = {
    "engine_traps": _engine_row,
    "empty_input": _empty_row,
    "fanout": _fanout_row,
    "grain": _grain_row,
    "integer_division": _division_row,
    "exact": _exact_row,
}


# The second answer's row counts and how it was asked; None when it wasn't answered.
def _second_detail(evidence: dict) -> str | None:
    parts = []
    asked = str(evidence.get("rephrase") or "").strip()
    if asked:
        end = "" if asked.endswith(("?", ".", "!")) else "."
        parts.append(f"Asked as: \"{asked}\"{end}")

    first, second = evidence.get("primary_row_count"), evidence.get("rephrase_row_count")
    if _is_count(first) and _is_count(second):
        parts.append(f"First answer {_rows(first)}, second answer {_rows(second)}.")
        
    if evidence.get("truncated"):
        parts.append(f"A result was cut at {sixd.an.ROW_CAP:,} rows, so only the rows kept were compared.")
    return " ".join(parts) or None


# The second answer's word, text and detail.
def _second_row(result, refused: bool) -> tuple[str, str, str | None]:
    final = result.evidence.get("final")
    if result.level == "pass":
        return SECOND_WORDS["pass"], "Asked a second way, the question gives the same result.", _second_detail(result.evidence)
    if result.level == "warn":
        return SECOND_WORDS["warn"], "Asked a second way, the question gives a different result.", _second_detail(result.evidence)
    if final == "pending":
        return RUNNING, "Asking the question a second way.", None

    if final == "skipped" and refused:
        text = "Not asked a second way, because the first answer was refused."
    elif final == "skipped":
        text = "No second answer: the question couldn't be put another way."
    elif final == "cannot":
        text = "No second answer: the question asked a second way couldn't be answered."
    elif final == "failed":
        text = f"No second answer: its SQL failed {TRIES} times."
    else:
        text = "No second answer."
    return WORDS.get(result.level, result.level), text, None


# A check row's word: Note for exact (shown for information), Not run for stopped typed SQL.
def _row_word(name: str, result, kind: str) -> str:
    if name == "exact":
        return WORDS["note"]
    if kind in SQL_KINDS and name == "engine_traps" and result.level == "refuse":
        return NOT_RUN
    return WORDS.get(result.level, result.level)


# The checks rows, the second answer first; SQL kinds show only the engine's evidence.
# The phase is only checked: the second answer's own evidence says whether it's running.
def check_rows(results, phase: str, kind: str) -> list[dict]:
    _check("phase", phase, PHASES)
    _check("kind", kind, KINDS)

    by_name = {r.name: r for r in results}
    refused = any(r.level == "refuse" for r in results)
    rows = []
    for name in ROW_ORDER:
        result = by_name.get(name)
        if result is None or (name == "second_phrasing" and kind in SQL_KINDS):
            continue

        if name == "second_phrasing":
            word, text, detail = _second_row(result, refused)
        else:
            text, detail = ROW_TEXTS[name](result)
            word = _row_word(name, result, kind)
        rows.append({"name": name, "word": word, "text": text, "detail": detail})
    return rows


# A term's level. On a re-ask the asked term must read exactly as asked;
# otherwise the terms check's own level stands.
def _term_level(entry: dict, asked: dict | None) -> str:
    if asked and entry.get("term") == asked.get("term"):
        return "note" if list(entry.get("readings") or []) == [asked.get("reading")] else "warn"
    return entry.get("level", "pass")


# The results with terms judged against the asked reading, so verdict and rows agree.
def _judged(results, asked: dict | None) -> list:
    if not asked:
        return list(results)

    judged = []
    for r in results:
        entries = r.evidence.get("terms")
        if r.name == "terms" and isinstance(entries, list) and entries:
            entries = [{**t, "level": _term_level(t, asked)} for t in entries]
            r = replace(r, level=_worst(t["level"] for t in entries), evidence={**r.evidence, "terms": entries})
        judged.append(r)
    return judged


# A flagging check in plain words, for the warning line.
def _flag(result) -> str:
    evidence = result.evidence
    match result.name:
        case "second_phrasing":
            return "the second answer differs"
        case "empty_input":
            return "the result is empty" if evidence.get("row_count") == 0 else "an aggregate read no rows"
        case "fanout":
            return "COUNT(*) counts rows the joins repeat" if _star_only(evidence) else "a total adds up rows the joins repeat"
        case "grain":
            return "rows repeat on the grouped columns" if "repeated" in evidence else "GROUP BY doesn't match the stated grain"
        case "terms":
            warned = [t.get("term") for t in evidence.get("terms") or [] if t.get("level") == "warn"]
            if not warned:
                return "a term's reading"
            return f"the reading{'s' if len(warned) > 1 else ''} of {_join(warned)}"
        case "integer_division":
            return "integer division drops the fraction"
        case _:
            return result.name.replace("_", " ")


# Where a check sits in the warning line: the second answer first, unknown checks last.
def _position(result) -> int:
    return VERDICT_ORDER.index(result.name) if result.name in VERDICT_ORDER else len(VERDICT_ORDER)


# The one overall line: pending, clean, warned, refused, can't answer, failed, not run, or ran.
# With asked, the terms check is judged against the reading asked for.
def verdict(results, phase: str, outcome: str, detail: str | None = None, asked: dict | None = None) -> dict:
    _check("phase", phase, PHASES)
    _check("outcome", outcome, OUTCOMES)

    if outcome == "cannot":
        return {"kind": "cannot", "text": f"Can't answer this yet. Model's reason: {detail or 'none given'}"}
    if outcome == "failed":
        last = f" Last error: {detail}" if detail else ""
        return {"kind": "failed", "text": f"No answer. The model's SQL failed {TRIES} times.{last}"}

    judged = _judged(results, asked)
    overall = ck.overall(judged)
    if outcome == "not_run" or overall == "refused":
        refusing = next((r for r in judged if r.level == "refuse"), None)
        reason = trap_reason(refusing or next((r for r in judged if r.name == "engine_traps"), None))
        if outcome == "not_run":
            return {"kind": "not_run", "text": f"Not run. {reason}"}
        
        return {"kind": "refused", "text": f"Can't answer this yet. {reason}"}

    # Never a clean-looking line while the second answer is still on its way.
    waiting = any(r.name == "second_phrasing" and r.evidence.get("final") == "pending" for r in judged)
    if outcome == "shown" and (phase == "first" or waiting):
        return {"kind": "pending", "text": PENDING_TEXT}

    noun = "result" if outcome == "ran" else "answer"
    flags = [_flag(r) for r in sorted(judged, key=_position) if r.level == "warn"]
    if flags:
        lead = "1 check flags" if len(flags) == 1 else f"{len(flags)} checks flag"
        return {"kind": "warned", "text": f"Warning. {lead} this {noun}: {'; '.join(flags)}."}
    return {"kind": "ran" if outcome == "ran" else "clean", "text": f"No check flagged this {noun}."}


# The question again with one sentence naming the reading to use.
def reask_question(question: str, term: str, reading: str) -> str:
    sentence = REASK_SENTENCES.get((term, reading), f" Use the {reading} reading of {term}.")
    return question.rstrip() + sentence


# How the SQL reads a term, after "the SQL": "reads it as gross" or "fits net and gross".
def _reads(fits: list[str]) -> str:
    return f"reads it as {fits[0]}" if len(fits) == 1 else f"fits {_join(fits)}"


# One term's sentence, from the readings that fit, the cued ones and the asked one.
# Without asked it follows the terms check's own cases, so it agrees with its level.
def _term_text(name: str, fits: list[str], cued: list[str], default: str | None, asked: dict | None) -> str:
    if asked and asked.get("term") == name:
        said = asked.get("reading")
        if fits == [said]:
            text = f"{name} read as {said}, as asked."
        elif not fits:
            text = f"{name}: asked for {said}, but no reading fits the SQL."
        else:
            text = f"{name}: asked for {said}, but the SQL {_reads(fits)}."
    elif len(cued) == 1 and fits == cued:
        text = f"{name} read as {cued[0]}, as the question says."
    elif len(cued) == 1 and fits and cued[0] not in fits:
        text = f"{name}: the question says {cued[0]}, but the SQL {_reads(fits)}."
    elif len(fits) == 1 and fits[0] == default:
        text = f"{name} read as {default}, the default."
    elif not fits:
        text = f"{name}: no reading fits the SQL."
    elif len(fits) == 1:
        text = f"{name} read as {fits[0]}, not the default{f' {default}' if default else ''}."
    else:
        text = f"{name} fits more than one reading: {_join(fits)}."
    return _sentence(text)


# The button for asking again with another reading.
def _alternative(term: str, reading: str) -> dict:
    label = _sentence(reading) if term in BARE_LABEL_TERMS else f"Ask again as {reading}"
    return {"reading": reading, "label": label}


# The Terms rows: each term's word, sentence and other readings.
# With asked, that term is judged against the reading asked for.
def term_rows(results, spec: dict, question: str, asked: dict | None = None) -> list[dict]:
    result = next((r for r in results if r.name == "terms"), None)
    entries = result.evidence.get("terms") if result is not None else None
    if not isinstance(entries, list):
        return []

    glossary = {t["term"]: t for t in spec.get("glossary", [])}
    rows = []
    for entry in entries:
        name, fits = entry["term"], list(entry.get("readings") or [])
        term = glossary.get(name)
        readings = term["readings"] if term else []
        cued = entry.get("cued")
        
        if cued is None and term:  # evidence without cues: read them from the question
            cued = [r["name"] for r in sixd.dd.cued_readings(term, question)]
        default = next((r["name"] for r in readings if r.get("default")), None)

        rows.append({
            "term": name,
            "word": WORDS[_term_level(entry, asked)],
            "text": _term_text(name, fits, list(cued or []), default, asked),
            "alternatives": [_alternative(name, r["name"]) for r in readings if r["name"] not in fits],
        })
    return rows


# The line under the checks: the tested rates, on an ask while the lock holds.
def tested_line(lock, kind: str, rates: dict) -> dict | None:
    _check("kind", kind, KINDS)
    if kind in SQL_KINDS:
        return None
    if kind == "reask":
        return {"text": REASK_LINE, "applies": False}
    
    if not lock.holds:
        why = f" ({lock.reason})" if lock.reason else ""
        return {"text": f"This setup differs from the tested one{why}, so the test's rates don't apply.", "applies": False}

    r = rates
    refused = "none" if r["refused_right"] == 0 else str(r["refused_right"])
    text = (
        f"Checked answers. In a {r['test_questions']}-question test with {MODEL_ID}, the checks flagged "
        f"{r['caught']} of {r['shown_wrong']} wrong answers ({_pct(r['caught'], r['shown_wrong'])}) and refused "
        f"{refused} of {r['shown_right']} right ones; {r['warned_right']} right ones "
        f"({_pct(r['warned_right'], r['shown_right'])}) got a warning. "
        f"{r['second_catches']} of the {r['caught']} catches came from the second answer."
    )
    
    return {"text": text, "applies": True}


# The "How the checks were tested" panel, one sentence per line.
def tested_panel(rates: dict) -> list[str]:
    r = rates
    questions, wrong, right = r["test_questions"], r["shown_wrong"], r["shown_right"]
    without = "fails" if r["without_terms"] / wrong < CATCH_BAR else "still clears"
    return [
        f"In a {questions}-question test with {MODEL_ID}, three bars were fixed before the model answered any test question.",
        f"Wrong answers flagged: {r['caught']} of {wrong} ({_pct(r['caught'], wrong)}), "
        f"against a bar of at least {CATCH_BAR:.0%}.",
        f"Right answers refused: {r['refused_right']} of {right} ({_pct(r['refused_right'], right)}), "
        f"against a bar of at most {REFUSE_BAR:.0%}.",
        f"Right answers that got a warning: {r['warned_right']} of {right} ({_pct(r['warned_right'], right)}), "
        f"against a bar of at most {FLAG_BAR:.0%}.",
        f"The second answer alone flagged {r['second_alone']} of {wrong} wrong answers; "
        f"the engine's own checks alone flagged {r['engine_alone']} of {wrong}.",
        f"Without the terms check, the rest flag {r['without_terms']} of {wrong} ({_pct(r['without_terms'], wrong)}), "
        f"which {without} the first bar.",
        f"The checks miss most on top-N answers: {r['topn_caught']} of {r['topn_wrong']} caught, "
        "since no check compares a ranking or its cutoff yet.",
        f"The model alone got {r['model_right']} of {questions} questions right.",
        f"Each question took {r['median_s']:g} s at the median in the test, and 9 in 10 took {r['p90_s']:g} s or less, "
        "for the first answer and the second together.",
        "Details: docs/research.md",
    ]
