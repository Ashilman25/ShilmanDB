# Label one shown answer: each check says pass, note, warn or refuse, with its evidence
from __future__ import annotations

import importlib.util
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
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
sf = an.load_sibling("an_sql_facts", "sql_facts.py")
sc = an.load_sibling("an_scoring", "scoring.py")
dd = an.load_sibling("an_describe_datasets", "describe_datasets.py")

LEVELS = ("pass", "note", "warn", "refuse")
CHECKS = ("engine_traps", "empty_input", "fanout", "grain", "second_phrasing", "terms", "integer_division", "exact")
ADDITIVE = ("SUM", "AVG", "COUNT")              # aggregates that repeated rows throw off
NUMERIC_TYPES = ("INTEGER", "BIGINT", "DECIMAL")
GRAIN_WORDS = r"\b(?:per|each|by)\s+"           # what comes right before a grain: "one row per nation"


# One check's verdict: its level, a short message and plain-JSON evidence.
@dataclass
class CheckResult:
    name: str
    level: str
    message: str
    evidence: dict


# The most serious of some levels; pass when there are none.
def _worst(levels: Iterable[str]) -> str:
    return max(levels, key=LEVELS.index, default="pass")


# The note a check gives when the SQL didn't parse.
def _unread(name: str, f: sf.SqlFacts) -> CheckResult:
    return CheckResult(name, "note", f"couldn't read the SQL ({f.error})", {"error": f.error})


# A row count with thousands separators, or ? when it's unknown.
def _count(v) -> str:
    return f"{v:,}" if isinstance(v, int) and not isinstance(v, bool) else "?"


# A key as a whole-word regex; inner spaces match hyphens, plurals take s or ies.
def _key_regex(key: str) -> str:
    *head, last = key.split()
    plural = f"{re.escape(last[:-1])}(?:ies|ys?)" if last.endswith("y") else f"{re.escape(last)}s?"
    return r"\b" + r"(?:\s+|-)".join([*map(re.escape, head), plural]) + r"\b"


# The key found earliest in the text, a tie going to the longer key.
# With after, a key only counts right after that regex.
def _earliest_key(text, keys: Iterable[str], after: str = "") -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    found = []
    for key in keys:
        if m := re.search(f"{after}({_key_regex(key)})", text, re.IGNORECASE):
            found.append((m.start(1), -len(key), key))
    return min(found)[2] if found else None


# The claimed grain: a key right after per, each or by, else any key.
def claimed_grain(text: str | None, spec: dict) -> str | None:
    grains = spec.get("grains", {})
    return _earliest_key(text, grains, GRAIN_WORDS) or _earliest_key(text, grains)


# The table a claimed entity names, through the spec's entity_tables.
def entity_table(text: str | None, spec: dict) -> str | None:
    tables = spec.get("entity_tables", {})
    key = _earliest_key(text, tables)
    return None if key is None else tables[key]


# A table repeats when key equalities from it don't reach every table joined to it,
# or a table off its equality paths scans over one row. Other equalities only filter.
def repeating_tables(facts: sf.SqlFacts, spec: dict, scan_rows: dict | None = None) -> set[str]:
    tables, keys, scan_rows = set(facts.tables), spec.get("primary_keys", {}), scan_rows or {}

    # Equalities between two tables share one edge, so together they can cover a key.
    far_side: dict[tuple[str, str], set[str]] = defaultdict(set)
    for a, b in facts.equalities:
        ta, tb = facts.column_table.get(a), facts.column_table.get(b)
        if ta in tables and tb in tables and ta != tb:
            far_side[ta, tb].add(b)
            far_side[tb, ta].add(a)

    neighbours = defaultdict(list)
    for p, q in far_side:
        neighbours[p].append(q)

    # p's equalities cover q's key, so each p row meets at most one q row.
    def covers_key(p: str, q: str) -> bool:
        key = keys.get(q)
        return bool(key) and set(key) <= far_side[p, q]

    # Tables reached from t, stepping from p to q only when step(p, q) holds.
    # With covers_key, the tables t determines; with no step rule, t's whole component.
    def reach(t: str, step=lambda p, q: True) -> set[str]:
        seen, stack = {t}, [t]
        while stack:
            p = stack.pop()
            for q in neighbours[p]:
                if q not in seen and step(p, q):
                    seen.add(q)
                    stack.append(q)
        return seen

    # A table with no scan count might have many rows.
    def many_rows(t: str) -> bool:
        n = scan_rows.get(t)
        return not isinstance(n, int) or n > 1

    # A table outside t's component crosses every row of t.
    def crossed(t: str) -> bool:
        return any(many_rows(u) for u in tables - reach(t))

    return {t for t in tables if reach(t, covers_key) != reach(t) or crossed(t)}


# Refuse SQL the engine runs wrong: LEFT JOIN, self-joins,
# dropped WHERE / ON terms, text math.
def _engine_traps(f: sf.SqlFacts) -> CheckResult:
    if not f.ok:
        return _unread("engine_traps", f)

    repeated = sorted(t for t, n in Counter(f.tables).items() if n > 1)
    # Only a one-table query drops a WHERE term with no column; a join applies it.
    constants = list(f.constant_conjuncts) if len(f.tables) == 1 else []
    traps = ["LEFT JOIN runs as an inner join"] if f.has_left_join else []
    traps += [f"{t} appears twice, and table qualifiers are ignored" for t in repeated]
    for c in constants:
        place = " outside a CASE" if c in f.case_only_conjuncts else ""
        traps.append(f"WHERE {c} names no column{place}, so it's dropped")
        
    traps += [f"ON {on} names {f'only {ts[0]}' if ts else 'no table'}, so the engine drops it" for on, ts in f.few_table_ons]
    traps += ["arithmetic on text or a date"] if f.text_arithmetic else []
    evidence = {
        "left_join": f.has_left_join,
        "repeated_tables": repeated,
        "constant_conjuncts": constants,
        "few_table_ons": [{"on": on, "tables": list(ts)} for on, ts in f.few_table_ons],
        "text_arithmetic": f.text_arithmetic,
    }

    if traps:
        return CheckResult("engine_traps", "refuse", "the engine runs this wrong: " + "; ".join(traps), evidence)
    return CheckResult("engine_traps", "pass", "no engine traps", evidence)


# The first Filter operator, else the first SeqScan; None if there's neither.
def _input_operator(operators: list) -> dict | None:
    for prefix in ("Filter", "SeqScan"):
        op = next((o for o in operators if isinstance(o, dict) and str(o.get("name", "")).startswith(prefix)), None)
        if op is not None:
            return op
    return None


# Warn on an empty result, or an aggregate over a step that kept no rows.
# With no row lists, the first Filter's (else SeqScan's) rows stand in.
def _empty_input(f: sf.SqlFacts, result: dict) -> CheckResult:
    profile = result.get("profile") or {}
    scan, step = list(profile.get("scan_rows") or []), list(profile.get("step_rows") or [])
    row_count = result.get("row_count", len(result.get("rows") or []))
    evidence = {
        "row_count": row_count,
        "scan_rows": scan,
        "step_rows": step
    }

    if row_count == 0:
        return CheckResult("empty_input", "warn", "the result is empty", evidence)
    if not f.ok:
        return _unread("empty_input", f)

    counts = scan + step
    if not counts and (op := _input_operator(profile.get("operators") or [])) is not None:
        evidence["operator"] = {"name": op.get("name"), "rows": op.get("rows")}
        counts = [op.get("rows")]

    if f.aggregates and 0 in counts:
        message = "an aggregate read 0 rows; SUM, AVG, MIN and MAX then show 0 where SQL would show NULL"
        return CheckResult("empty_input", "warn", message, evidence)
    return CheckResult("empty_input", "pass", "every input had rows", evidence)


# Rows out of each table's scan, by name; a repeated table keeps its first.
# scan_rows is in join order, so entry i belongs to tables[order[i]].
def _scan_rows_by_table(profile: dict) -> dict:
    join = profile.get("join") or {}
    tables, order, scan = join.get("tables") or [], join.get("order") or [], profile.get("scan_rows") or []
    rows = {}
    for i, j in enumerate(order):
        if isinstance(j, int) and 0 <= j < len(tables) and i < len(scan):
            rows.setdefault(tables[j], scan[i])
    return rows


# Per repeating table: rows out of its scan, and rows after the last join step.
def _fanout_rows(repeating: set[str], profile: dict) -> dict:
    scanned, step = _scan_rows_by_table(profile), profile.get("step_rows") or []
    return {t: {"scan_rows": scanned.get(t), "joined_rows": step[-1] if step else None} for t in sorted(repeating)}


# Problems with each select item holding COUNT(*): its claim must map to a non-repeating table.
# COUNT(*) outside the select list is judged by those, or can't be mapped without them.
def _count_star_problems(f: sf.SqlFacts, claims: dict, spec: dict, repeating: set[str]) -> list[str]:
    columns = claims.get("columns") or []
    problems = []
    for i in f.count_star_items:
        claim = columns[i] if i < len(columns) and isinstance(columns[i], dict) else {}
        text = claim.get("entity")
        entity = entity_table(text, spec)

        if entity is None:
            why = f"its entity {text!r} doesn't map to a table" if text else "it has no claimed entity"
            problems.append(f"COUNT(*) in column {i + 1} counts joined rows, and {why}")
        elif entity in repeating:
            problems.append(f"COUNT(*) in column {i + 1} claims to count {entity}, which repeats")

    stars = sum(1 for a in f.aggregates if a.func == "COUNT" and a.star)
    if stars and not f.count_star_items:
        problems.append(f"{stars} COUNT(*) outside the select list counts joined rows, and no select item claims what it counts")
    return problems


# Warn when a sum, average or count reads rows the joins repeat.
def _fanout(f: sf.SqlFacts, result: dict, claims: dict, spec: dict) -> CheckResult:
    if not f.ok:
        return _unread("fanout", f)

    profile = result.get("profile") or {}
    repeating = repeating_tables(f, spec, _scan_rows_by_table(profile))
    rows = _fanout_rows(repeating, profile)
    if not repeating:
        return CheckResult("fanout", "pass", "no table repeats in the joins", {"repeating": rows, "problems": []})

    # A CASE's conditions only choose rows, unless its values are all constants (a row count).
    problems = []
    for a in f.aggregates:
        read = sorted({f.column_table.get(c) for c in a.value_columns or a.columns} & repeating)
        if a.func in ADDITIVE and not a.star and read:
            problems.append(f"{a.func}({', '.join(sorted(a.columns))}) adds up {', '.join(read)}")
    problems += _count_star_problems(f, claims, spec, repeating)

    repeats = ", ".join(f"{t} ({_count(r['scan_rows'])} rows scanned, {_count(r['joined_rows'])} after the joins)" for t, r in rows.items())
    evidence = {"repeating": rows, "problems": problems}
    if problems:
        return CheckResult("fanout", "warn", f"{'; '.join(problems)}; repeating: {repeats}", evidence)
    return CheckResult("fanout", "pass", f"no sum, average or count adds up a repeating table: {repeats}", evidence)


# The first tuple that shows up twice, or None.
def _first_repeat(tuples: Iterable[tuple]) -> tuple | None:
    seen = set()
    for t in tuples:
        if t in seen:
            return t
        seen.add(t)
    return None


# Warn when shown rows repeat on GROUP BY columns,
# or when GROUP BY doesn't cover the claimed grain.
def _grain(f: sf.SqlFacts, result: dict, claims: dict, spec: dict) -> CheckResult:
    if not f.ok:
        return _unread("grain", f)
    if not f.group_by:
        return CheckResult("grain", "pass", "no GROUP BY", {"group_by": []})

    # Rows repeating on the GROUP BY columns they show are grouped finer than shown.
    group = set(f.group_by)
    shown = [i for i, c in enumerate(f.select_columns) if c in group]
    if shown and len(f.select_columns) == len(result.get("columns") or []):
        repeat = _first_repeat(tuple(row[i] for i in shown) for row in result.get("rows") or [])
        if repeat is not None:
            names = [f.select_columns[i] for i in shown]
            evidence = {"group_by": f.group_by, "columns": names, "repeated": list(repeat)}
            return CheckResult("grain", "warn", f"rows repeat on {', '.join(names)}: {list(repeat)} shows up twice", evidence)

    text = claims.get("grain")
    claimed = claimed_grain(text, spec)
    evidence = {"group_by": f.group_by, "claimed": text if isinstance(text, str) else None, "grain": claimed}
    if claimed is None:
        why = (f"the claimed grain {text!r} isn't one I can map" if evidence["claimed"] else "no claimed grain to compare")
        return CheckResult("grain", "note", why, evidence)

    sets = [list(cols) for cols in spec["grains"][claimed]]
    evidence["column_sets"] = sets
    if not any(set(cols) <= group for cols in sets):
        wanted = " or ".join("(" + ", ".join(cols) + ")" for cols in sets)
        message = f"claims one row per {claimed}, but GROUP BY {', '.join(f.group_by)} doesn't cover {wanted}"
        return CheckResult("grain", "warn", message, evidence)
    return CheckResult("grain", "pass", f"GROUP BY covers the claimed grain ({claimed})", evidence)


# Whether shown matches gold, any order; gold's numeric columns may show as percents.
def _matches(shown: dict, gold: dict) -> bool:
    shown_cols, gold_cols = shown.get("columns") or [], gold.get("columns") or []
    shares = [i for i, c in enumerate(gold_cols) if c.get("type") in NUMERIC_TYPES]

    return sc.match_result(
        [c.get("name") for c in shown_cols], shown.get("rows") or [],
        [c.get("name") for c in gold_cols], gold.get("rows") or [],
        ordered=False, share_cols=shares
    )


# Warn when the rephrased question's result differs both ways round,
# so a percent on either side matches a fraction on the other.
def _second_phrasing(primary: dict, rephrase: dict) -> CheckResult:
    final = rephrase.get("final")
    if final != "shown" or not isinstance(rephrase.get("result"), dict):
        return CheckResult("second_phrasing", "note", f"the rephrased question wasn't answered ({final})", {"final": final})

    other = rephrase["result"]
    same = _matches(other, primary) or _matches(primary, other)
    evidence = {
        "rephrase": rephrase.get("text"), "sql": rephrase.get("sql"),
        "primary_row_count": primary.get("row_count"), "rephrase_row_count": other.get("row_count"),
        "truncated": bool(primary.get("truncated") or other.get("truncated")),
    }
    if same:
        return CheckResult("second_phrasing", "pass", "the rephrased question gives the same result", evidence)
    return CheckResult("second_phrasing", "warn", "the rephrased question gives a different result", evidence)


# One glossary term's level and message, from the readings that fit the SQL.
# When the question cues just one reading, that's the one expected.
def _term_verdict(term: dict, fits: list[dict], cued: list[dict]) -> tuple[str, str]:
    name = term["term"]
    if len(cued) == 1 and fits:
        said = cued[0]["name"]
        if fits == cued:
            return "note", f"{name} read as {said}, as the question says"
        if cued[0] not in fits:
            return "warn", f"{name}: the question says {said}, the SQL reads {', '.join(r['name'] for r in fits)}"

    default = next((r["name"] for r in term["readings"] if r["default"]), "?")
    if len(fits) == 1 and fits[0]["default"]:
        others = [r["name"] for r in term["readings"] if not r["default"]]
        return "note", f"{name} read as {fits[0]['name']}; other readings: {', '.join(others) or 'none'}"

    if not fits:
        return "warn", f"{name}: no reading fits the SQL (readings: {', '.join(r['name'] for r in term['readings'])})"

    if len(fits) == 1:
        return "warn", f"{name} read as {fits[0]['name']}, not the default {default}"
    return "warn", f"{name} fits several readings: {', '.join(r['name'] for r in fits)}"


# Note a glossary term read the expected way, warn on any other.
# The expected reading is the one the question cues, else the default.
def _terms(f: sf.SqlFacts, question: dict, spec: dict, sql: str) -> CheckResult:
    if not f.ok:
        return _unread("terms", f)

    text = question.get("question") or ""
    found = []
    for term in spec.get("glossary", []):
        if any(re.search(w, text, re.IGNORECASE) for w in term["words"]):
            fits = [r for r in term["readings"] if dd.reading_matches(r, f.columns, sql)]
            cued = dd.cued_readings(term, text)
            level, message = _term_verdict(term, fits, cued)
            found.append({"term": term["term"], "readings": [r["name"] for r in fits],
                          "cued": [r["name"] for r in cued], "level": level, "message": message})

    if not found:
        return CheckResult("terms", "pass", "no glossary term in the question", {"terms": []})
    evidence = {"terms": [{k: t[k] for k in ("term", "readings", "cued", "level")} for t in found]}
    return CheckResult("terms", _worst(t["level"] for t in found), "; ".join(t["message"] for t in found), evidence)


# Warn on integer division, which drops the fraction.
def _integer_division(f: sf.SqlFacts) -> CheckResult:
    if not f.ok:
        return _unread("integer_division", f)

    evidence = {"divisions": list(f.int_divisions)}
    if f.int_divisions:
        message = "integer division drops the fraction: " + "; ".join(f.int_divisions)
        return CheckResult("integer_division", "warn", message, evidence)
    return CheckResult("integer_division", "pass", "no integer division", evidence)


# Every check on one shown answer, in CHECKS order. Reads nothing but its arguments.
def run_checks(record: dict, question: dict, spec: dict, manifest: dict) -> list[CheckResult]:
    primary = record.get("primary") or {}
    if primary.get("final") != "shown":
        raise ValueError(f"{record.get('qid')}: only a shown answer can be checked, got final={primary.get('final')!r}")

    sql, result, claims = primary.get("sql"), primary.get("result") or {}, primary.get("claims") or {}
    f = sf.facts(sql, manifest)
    return [
        _engine_traps(f),
        _empty_input(f, result),
        _fanout(f, result, claims, spec),
        _grain(f, result, claims, spec),
        _second_phrasing(result, record.get("rephrase") or {}),
        _terms(f, question, spec, sql),
        _integer_division(f),
        CheckResult("exact", "pass", "exact: full scan, no sampling", {}),
    ]


# refused if any check refuses, else warned if any warns, else clean.
# Checks named in skip don't count.
def overall(results: Iterable[CheckResult], skip: Iterable[str] = ()) -> str:
    skip = set(skip)
    levels = {r.level for r in results if r.name not in skip}
    return "refused" if "refuse" in levels else "warned" if "warn" in levels else "clean"
