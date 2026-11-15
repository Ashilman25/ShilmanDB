"""Tests for app/check_views.py and app/copy_rules.py."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

# tested_line and tested_panel are reached through cv: pytest collects any test* name in a test module.
from app import check_views as cv
from app import sixd
from app.check_views import (
    KINDS,
    OUTCOMES,
    PHASES,
    WORDS,
    check_rows,
    reask_question,
    term_rows,
    trap_reason,
    verdict,
)
from app.copy_rules import plain, problems

CheckResult = sixd.ck.CheckResult
APP_DIR = Path(__file__).resolve().parents[1]
ANALYZER = sixd.ANALYST_DIR / "analyze_analyst_check.py"
SPECS = {name: sixd.dd.load_spec(name) for name in ("tpch", "reviews")}
TPCH = SPECS["tpch"]
QUESTION = "What was revenue by region in 1995?"

RATES = {"test_questions": 300, "model_right": 252, "shown_right": 227, "shown_wrong": 35, "caught": 20,
         "refused_right": 0, "warned_right": 18, "second_catches": 15, "second_alone": 15, "engine_alone": 5,
         "without_terms": 17, "topn_caught": 6, "topn_wrong": 13, "median_s": 5.3, "p90_s": 7.6}
TESTED = ("Checked answers. In a 300-question test with gemma4-26b, the checks flagged 20 of 35 wrong answers (57%) "
          "and refused none of 227 right ones; 18 right ones (8%) got a warning. 15 of the 20 catches came from the "
          "second answer.")
REASKED = "Re-asked with a reading named. Not part of the test, so its rates don't apply here."
HOLDS = SimpleNamespace(holds=True, reason=None)
BROKEN = SimpleNamespace(holds=False, reason="the engine changed since the test")

NO_TRAPS = {"left_join": False, "repeated_tables": [], "constant_conjuncts": [], "few_table_ons": [],
            "text_arithmetic": False}
SAME = {"rephrase": "Which region had the most revenue in 1995?", "sql": "SELECT r_name FROM region",
        "primary_row_count": 5, "rephrase_row_count": 5, "truncated": False}
DIFFERENT = {**SAME, "rephrase_row_count": 4, "truncated": True}

# Evidence for each check at each level it reaches, as the analyst check writes it.
SAMPLES = {
    "engine_traps": [
        ("pass", NO_TRAPS),
        ("note", {"error": "Expected table name but got end of input"}),
        ("refuse", {**NO_TRAPS, "left_join": True}),
        ("refuse", {**NO_TRAPS, "repeated_tables": ["nation"]}),
        ("refuse", {**NO_TRAPS, "repeated_tables": ["nation", "supplier"]}),
        ("refuse", {**NO_TRAPS, "constant_conjuncts": ["1 = 0"]}),
        ("refuse", {**NO_TRAPS, "constant_conjuncts": ["1 = 0", "2 > 1"]}),
        ("refuse", {**NO_TRAPS, "few_table_ons": [{"on": "o_orderstatus = 'F'", "tables": ["orders"]}]}),
        ("refuse", {**NO_TRAPS, "few_table_ons": [{"on": "1 = 1", "tables": []}]}),
        ("refuse", {**NO_TRAPS, "text_arithmetic": True}),
        ("refuse", {**NO_TRAPS, "left_join": True, "repeated_tables": ["nation"], "text_arithmetic": True}),
        ("refuse", {"guard": True}),
        ("refuse", {"guard": True, "readable": False}),
        ("refuse", {"guard": True, "readable": True}),
    ],
    "empty_input": [
        ("pass", {"row_count": 5, "scan_rows": [5, 25, 15000], "step_rows": [25, 15000]}),
        ("pass", {"row_count": 1, "scan_rows": [], "step_rows": [], "operator": {"name": "Filter", "rows": 3}}),
        ("warn", {"row_count": 0, "scan_rows": [5, 0], "step_rows": [0]}),
        ("warn", {"row_count": 1, "scan_rows": [0, 25], "step_rows": [0]}),
        ("note", {"error": "statement is nested too deeply"}),
    ],
    "fanout": [
        ("pass", {"repeating": {}, "problems": []}),
        ("pass", {"repeating": {"orders": {"scan_rows": 150000, "joined_rows": 600572}}, "problems": []}),
        ("warn", {"repeating": {"orders": {"scan_rows": 150000, "joined_rows": 600572}},
                  "problems": ["SUM(o_totalprice) adds up orders"]}),
        ("warn", {"repeating": {"orders": {"scan_rows": None, "joined_rows": None}},
                  "problems": ["COUNT(*) in column 2 counts joined rows, and it has no claimed entity"]}),
        ("note", {"error": "x"}),
    ],
    "grain": [
        ("pass", {"group_by": []}),
        ("pass", {"group_by": ["r_name"], "claimed": "one row per region", "grain": "region",
                  "column_sets": [["r_name"]]}),
        ("warn", {"group_by": ["r_name", "o_orderyear"], "columns": ["r_name"], "repeated": ["ASIA"]}),
        ("warn", {"group_by": ["n_name"], "claimed": "one row per region", "grain": "region",
                  "column_sets": [["r_name"], ["r_regionkey"]]}),
        ("note", {"group_by": ["r_name"], "claimed": None, "grain": None}),
        ("note", {"group_by": ["r_name"], "claimed": "one row per thing", "grain": None}),
        ("note", {"error": "x"}),
    ],
    "second_phrasing": [
        ("pass", SAME),
        ("warn", DIFFERENT),
        ("note", {"final": "pending"}),
        ("note", {"final": "skipped"}),
        ("note", {"final": "cannot"}),
        ("note", {"final": "failed"}),
    ],
    "terms": [
        ("pass", {"terms": []}),
        ("note", {"terms": [{"term": "revenue", "readings": ["net"], "cued": [], "level": "note"}]}),
        ("warn", {"terms": [{"term": "revenue", "readings": ["gross"], "cued": [], "level": "warn"},
                            {"term": "date basis", "readings": [], "cued": [], "level": "warn"}]}),
        ("note", {"error": "x"}),
    ],
    "integer_division": [
        ("pass", {"divisions": []}),
        ("warn", {"divisions": ["COUNT(*) / 7"]}),
        ("note", {"error": "x"}),
    ],
    "exact": [("pass", {})],
}


# One check result built by hand, as the analyst check's own tests do.
def _r(name: str, level: str, evidence: dict | None = None) -> CheckResult:
    return CheckResult(name, level, f"{name} {level}", evidence if evidence is not None else {})


# A shown answer's results in the checks' order: all clean, the second answer as given.
def _answer(second: dict | None = None, second_level: str = "note", **over: CheckResult) -> list[CheckResult]:
    base = {
        "engine_traps": _r("engine_traps", "pass", NO_TRAPS),
        "empty_input": _r("empty_input", "pass", SAMPLES["empty_input"][0][1]),
        "fanout": _r("fanout", "pass", {"repeating": {}, "problems": []}),
        "grain": _r("grain", "pass", {"group_by": []}),
        "second_phrasing": _r("second_phrasing", second_level, second if second is not None else {"final": "pending"}),
        "terms": _r("terms", "pass", {"terms": []}),
        "integer_division": _r("integer_division", "pass", {"divisions": []}),
        "exact": _r("exact", "pass", {}),
    }
    base.update(over)
    return [base[name] for name in sixd.ck.CHECKS]


# The results a typed SQL run gets: no second answer, no terms.
def _sql(**over: CheckResult) -> list[CheckResult]:
    return [r for r in _answer(**over) if r.name not in ("second_phrasing", "terms")]


# Every string inside a value made of dicts, lists and strings.
def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _strings(v)]
    return []


def test_copy_rules_catch_each():
    assert problems("plain text") == []
    assert problems("Can't answer this yet. 5.3 s - fine, 20 of 35 (57%); a/b <= c") == []
    for code in (0x2705, 0x2014, 0x25B8, 0x2192, 0x2713, 0x2015, 0x1F600, 0x2190, 0x21FF, 0x25A0, 0x2718,
                 0x23F3, 0x27F6, 0x2934, 0x2B50, 0x2E3A):
        found = problems("ok " + chr(code) + " ok")
        assert len(found) == 1, hex(code)
        assert found[0].startswith(f"U+{code:04X} at 3"), found
    assert len(problems(chr(0x2014) * 3)) == 3
    assert problems(chr(0x2013) + chr(0x2019)) == []  # en dash and a curly apostrophe aren't on the list


def test_plain_turns_dashes_into_commas():
    dash, bar = chr(0x2014), chr(0x2015)
    assert plain(f"No weather data {dash} only orders.") == "No weather data, only orders."
    assert plain(f"a{dash}b") == "a, b"
    assert plain(f"a {bar}b") == "a, b"
    assert plain(f"a {dash} {dash} b") == "a, b"  # a run of dashes counts as one
    keep = "a - b " + chr(0x2013) + " c"
    assert plain(keep) == keep  # hyphens and en dashes stay
    assert problems(plain(f"x {dash} y {bar} z")) == []


def test_own_files_pass_copy_rules():
    for path in (APP_DIR / "check_views.py", APP_DIR / "copy_rules.py", Path(__file__)):
        assert problems(path.read_text(encoding="utf-8")) == [], path.name


def test_words_for_each_level():
    assert WORDS == {"pass": "Pass", "note": "Note", "warn": "Warning", "refuse": "Refused"}
    for level, word in WORDS.items():
        results = _answer(second=SAME, second_level="pass", integer_division=_r("integer_division", level, {"divisions": []}))
        row = next(r for r in check_rows(results, "done", "ask") if r["name"] == "integer_division")
        assert row["word"] == word


def test_exact_row_is_a_note():
    for level in WORDS:
        results = _answer(SAME, "pass", exact=_r("exact", level, {}))
        for kind in KINDS:
            (row,) = [r for r in check_rows(results, "done", kind) if r["name"] == "exact"]
            assert row["word"] == "Note"
    # The verdict still reads levels: exact never flags.
    assert verdict(_answer(SAME, "pass"), "done", "shown")["kind"] == "clean"
    assert verdict(_answer(SAME, "pass", exact=_r("exact", "note", {})), "done", "shown")["kind"] == "clean"


def test_second_answer_row_first_with_its_words():
    cases = [("pass", SAME, "done", "Same"), ("warn", DIFFERENT, "done", "Differs"),
             ("note", {"final": "pending"}, "first", "Running"), ("note", {"final": "cannot"}, "done", "Note")]
    for level, evidence, phase, word in cases:
        for kind in ("ask", "reask"):
            rows = check_rows(_answer(second=evidence, second_level=level), phase, kind)
            assert rows[0]["name"] == "second_phrasing"
            assert rows[0]["word"] == word
    assert set(check_rows(_answer(), "first", "ask")[0]) == {"name", "word", "text", "detail"}

    # Only a pending second answer is Running, even in the first phase.
    trap = _r("engine_traps", "refuse", {**NO_TRAPS, "left_join": True})
    for final in ("skipped", "cannot", "failed"):
        (row, *_) = check_rows(_answer(second={"final": final}, engine_traps=trap), "first", "ask")
        assert row["word"] == "Note", final
    (row, *_) = check_rows(_answer(second={"final": "skipped"}, engine_traps=trap), "first", "ask")
    assert row["text"] == "Not asked a second way, because the first answer was refused."


def test_rows_in_order():
    names = [r["name"] for r in check_rows(_answer(), "first", "ask")]
    assert names == ["second_phrasing", "engine_traps", "empty_input", "fanout", "grain", "integer_division", "exact"]


def test_sql_kinds_show_engine_evidence_only():
    for kind in ("sql", "edit"):
        names = [r["name"] for r in check_rows(_sql(), "done", kind)]
        assert names == ["engine_traps", "empty_input", "fanout", "grain", "integer_division", "exact"]
        # Even when handed a full answer's results, a SQL kind shows no second answer.
        assert "second_phrasing" not in [r["name"] for r in check_rows(_answer(), "done", kind)]
    for kind in KINDS:
        assert "terms" not in [r["name"] for r in check_rows(_answer(), "done", kind)]


def test_sql_trap_row_says_not_run():
    trap = _r("engine_traps", "refuse", {**NO_TRAPS, "left_join": True})
    assert check_rows(_sql(engine_traps=trap), "done", "sql")[0]["word"] == "Not run"
    ask_row = next(r for r in check_rows(_answer(engine_traps=trap), "done", "ask") if r["name"] == "engine_traps")
    assert ask_row["word"] == "Refused"
    assert ask_row["text"] == trap_reason(trap)


def test_verdict_texts():
    assert verdict(_answer(), "first", "shown") == {"kind": "pending", "text": "Verdict once the second answer is back."}
    assert verdict(_answer(), "done", "shown")["kind"] == "pending"  # never clean while the second answer runs
    assert verdict(_answer(SAME, "pass"), "done", "shown") == {"kind": "clean", "text": "No check flagged this answer."}

    two = verdict(_answer(DIFFERENT, "warn", integer_division=_r("integer_division", "warn", {"divisions": ["COUNT(*) / 7"]})),
                  "done", "shown")
    assert two["kind"] == "warned"
    assert two["text"].startswith("Warning. 2 checks flag this answer: ")
    assert "the second answer differs" in two["text"]
    one = verdict(_answer(DIFFERENT, "warn"), "done", "shown")
    assert one == {"kind": "warned", "text": "Warning. 1 check flags this answer: the second answer differs."}

    assert verdict([], "done", "cannot", detail="it needs COUNT(DISTINCT)") == {
        "kind": "cannot", "text": "Can't answer this yet. Model's reason: it needs COUNT(DISTINCT)"}
    assert verdict([], "done", "failed", detail="Parse error: x") == {
        "kind": "failed", "text": "No answer. The model's SQL failed 3 times. Last error: Parse error: x"}

    assert verdict(_sql(), "done", "ran") == {"kind": "ran", "text": "No check flagged this result."}
    warned_sql = verdict(_sql(integer_division=_r("integer_division", "warn", {"divisions": ["COUNT(*) / 7"]})), "done", "ran")
    assert warned_sql["kind"] == "warned"
    assert warned_sql["text"].startswith("Warning. 1 check flags this result: ")


def test_verdict_refusal_wins_over_pending():
    trap = _r("engine_traps", "refuse", {**NO_TRAPS, "left_join": True})
    for phase in PHASES:
        assert verdict(_answer(engine_traps=trap), phase, "shown")["kind"] == "refused"


def test_verdict_refused_left_join():
    trap = _r("engine_traps", "refuse", {**NO_TRAPS, "left_join": True})
    assert verdict(_answer(engine_traps=trap), "done", "shown") == {
        "kind": "refused",
        "text": "Can't answer this yet. The SQL uses LEFT JOIN, which today's engine runs as an inner join, "
                "so rows with no match would drop out without a warning.",
    }


def test_verdict_refused_repeated_table():
    trap = _r("engine_traps", "refuse", {**NO_TRAPS, "repeated_tables": ["nation"]})
    assert verdict(_answer(engine_traps=trap), "done", "shown")["text"] == (
        "Can't answer this yet. The SQL joins nation twice, and today's engine ignores table qualifiers, "
        "so it would read the wrong columns.")


def test_verdict_not_run_guard():
    guard = _r("engine_traps", "refuse", {"guard": True})
    assert verdict(_sql(engine_traps=guard), "done", "not_run") == {
        "kind": "not_run",
        "text": "Not run. The SQL couldn't be checked, and it mentions an outer join, "
                "which today's engine runs as an inner join.",
    }


def test_readable_guard_reason():
    sentence = ("The SQL uses an outer join that today's engine reads as a table alias and runs as an inner join, "
                "so rows with no match would drop out without a warning.")
    readable = _r("engine_traps", "refuse", {"guard": True, "readable": True})
    assert trap_reason(readable) == cv.ALIAS_JOIN_REASON == sentence
    assert problems(sentence) == []
    assert verdict(_sql(engine_traps=readable), "done", "not_run") == {"kind": "not_run", "text": f"Not run. {sentence}"}
    assert verdict(_answer(engine_traps=readable), "done", "shown") == {
        "kind": "refused", "text": f"Can't answer this yet. {sentence}"}

    # Unreadable SQL keeps the old sentence, with readable false or absent.
    for evidence in ({"guard": True, "readable": False}, {"guard": True}):
        assert trap_reason(_r("engine_traps", "refuse", evidence)) == cv.GUARD_REASON


def test_trap_reason_other_traps():
    constants = trap_reason(_r("engine_traps", "refuse", {**NO_TRAPS, "constant_conjuncts": ["1 = 0"]}))
    assert "1 = 0" in constants and "WHERE" in constants
    on = trap_reason(_r("engine_traps", "refuse", {**NO_TRAPS, "few_table_ons": [{"on": "o_orderstatus = 'F'",
                                                                                   "tables": ["orders"]}]}))
    assert "o_orderstatus = 'F'" in on and "only orders" in on
    assert "text or a date" in trap_reason(_r("engine_traps", "refuse", {**NO_TRAPS, "text_arithmetic": True}))

    both = trap_reason(_r("engine_traps", "refuse", {**NO_TRAPS, "left_join": True, "repeated_tables": ["nation"]}))
    assert both.startswith("The SQL uses LEFT JOIN") and "joins nation twice" in both
    for _, evidence in SAMPLES["engine_traps"]:
        text = trap_reason(_r("engine_traps", "refuse", evidence))
        assert text.startswith("The SQL") and text.endswith(".")


def test_reask_text():
    assert reask_question(QUESTION, "revenue", "gross") == "What was revenue by region in 1995? Use the gross reading of revenue."
    assert reask_question(QUESTION + "  \n", "revenue", "gross") == reask_question(QUESTION, "revenue", "gross")


def test_reask_cues_every_cued_reading():
    for spec in SPECS.values():
        for term in spec["glossary"]:
            for reading in term["readings"]:
                if not reading.get("cues"):
                    continue
                cued = sixd.dd.cued_readings(term, reask_question("Q?", term["term"], reading["name"]))
                assert reading in cued, (term["term"], reading["name"])


def test_term_rows_alternatives():
    evidence = {"terms": [{"term": "revenue", "readings": ["net"], "cued": [], "level": "note"},
                          {"term": "date basis", "readings": ["order date"], "cued": [], "level": "note"}]}
    revenue, dates = term_rows(_answer(terms=_r("terms", "note", evidence)), TPCH, QUESTION, None)

    assert revenue["term"] == "revenue" and revenue["word"] == "Note"
    assert revenue["text"] == "Revenue read as net, the default."
    assert revenue["alternatives"] == [{"reading": "gross", "label": "Ask again as gross"}]
    assert dates["term"] == "date basis" and dates["word"] == "Note"
    assert dates["alternatives"] == [{"reading": "ship date", "label": "Ship date"},
                                     {"reading": "receipt date", "label": "Receipt date"}]
    assert set(revenue) == {"term", "word", "text", "alternatives"}


def test_term_rows_warn_words():
    evidence = {"terms": [{"term": "revenue", "readings": ["gross"], "cued": ["net"], "level": "warn"}]}
    (row,) = term_rows(_answer(terms=_r("terms", "warn", evidence)), TPCH, "What was net revenue?", None)
    assert row["word"] == "Warning"
    assert row["text"] == "Revenue: the question says net, but the SQL reads it as gross."
    assert row["alternatives"] == [{"reading": "net", "label": "Ask again as net"}]
    assert term_rows(_answer(), TPCH, QUESTION, None) == []
    assert term_rows(_answer(terms=_r("terms", "note", {"error": "x"})), TPCH, QUESTION, None) == []


def test_term_rows_judged_against_asked():
    asked = {"term": "late shipment", "reading": "shipped after commit"}
    question = reask_question("How many late shipments were there?", asked["term"], asked["reading"])

    fits = {"terms": [{"term": "late shipment", "readings": ["shipped after commit"], "cued": [], "level": "warn"}]}
    (row,) = term_rows(_answer(terms=_r("terms", "warn", fits)), TPCH, question, asked)
    assert row["word"] == "Note"
    assert row["text"].endswith("as asked.")
    assert row["alternatives"] == [{"reading": "received after commit", "label": "Ask again as received after commit"}]

    other = {"terms": [{"term": "late shipment", "readings": ["received after commit"], "cued": [], "level": "note"}]}
    (row,) = term_rows(_answer(terms=_r("terms", "note", other)), TPCH, question, asked)
    assert row["word"] == "Warning"


def test_verdict_judges_terms_against_asked():
    asked = {"term": "late shipment", "reading": "shipped after commit"}
    fits = _r("terms", "warn", {"terms": [
        {"term": "late shipment", "readings": ["shipped after commit"], "cued": [], "level": "warn"},
        {"term": "date basis", "readings": ["order date"], "cued": [], "level": "note"}]})
    results = _answer(SAME, "pass", terms=fits)
    assert verdict(results, "done", "shown", asked=asked)["kind"] == "clean"
    assert verdict(results, "done", "shown") == {
        "kind": "warned", "text": "Warning. 1 check flags this answer: the reading of late shipment."}

    other = _r("terms", "note", {"terms": [
        {"term": "late shipment", "readings": ["received after commit"], "cued": [], "level": "note"}]})
    results = _answer(SAME, "pass", terms=other)
    assert verdict(results, "done", "shown")["kind"] == "clean"
    assert verdict(results, "done", "shown", asked=asked) == {
        "kind": "warned", "text": "Warning. 1 check flags this answer: the reading of late shipment."}
    # The rows and the verdict line agree on a re-ask.
    (row,) = term_rows(results, TPCH, QUESTION, asked)
    assert row["word"] == "Warning"


def test_tested_line_cases():
    assert cv.tested_line(HOLDS, "ask", RATES) == {"text": TESTED, "applies": True}
    for lock in (HOLDS, BROKEN):
        assert cv.tested_line(lock, "reask", RATES) == {"text": REASKED, "applies": False}
        assert cv.tested_line(lock, "sql", RATES) is None
        assert cv.tested_line(lock, "edit", RATES) is None
    assert cv.tested_line(BROKEN, "ask", RATES) == {
        "text": "This setup differs from the tested one (the engine changed since the test), so the test's rates don't apply.",
        "applies": False,
    }


def test_tested_line_built_from_rates():
    text = cv.tested_line(HOLDS, "ask", {**RATES, "caught": 21, "refused_right": 2, "warned_right": 19})["text"]
    assert "flagged 21 of 35 wrong answers (60%)" in text
    assert "refused 2 of 227 right ones; 19 right ones (8%)" in text
    assert "15 of the 21 catches" in text


# The analyzer's top-level constant assignments, read without importing or running it.
def _analyzer_constants(names: set[str]) -> dict:
    tree = ast.parse(ANALYZER.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    found[target.id] = ast.literal_eval(node.value)
    return found


def test_bars_match_the_analyst_check():
    if not ANALYZER.is_file():
        pytest.skip(f"{ANALYZER} is absent")
    bars = _analyzer_constants({"CATCH_BAR", "REFUSE_BAR", "FLAG_BAR"})
    assert bars == {"CATCH_BAR": cv.CATCH_BAR, "REFUSE_BAR": cv.REFUSE_BAR, "FLAG_BAR": cv.FLAG_BAR}


def test_tested_panel_numbers():
    panel = cv.tested_panel(RATES)
    assert all(isinstance(s, str) and s for s in panel)
    text = " ".join(panel)
    for part in ("20 of 35", "0 of 227", "18 of 227", "15 of 35", "5 of 35", "17 of 35", "6 of 13", "252 of 300",
                 "5.3 s", "7.6 s", "docs/research.md"):
        assert part in text, part
    assert "which fails the first bar" in text


def test_every_view_string_passes_copy_rules():
    strings: list[str] = []
    for name, samples in SAMPLES.items():
        for _, evidence in samples:
            for level in WORDS:
                result = _r(name, level, evidence)
                answer, sql = _answer(**{name: result}), _sql(**{name: result})
                for phase in PHASES:
                    for kind in KINDS:
                        strings += _strings(check_rows(sql if kind in ("sql", "edit") else answer, phase, kind))
                    for outcome in OUTCOMES:
                        strings += _strings(verdict(sql if outcome in ("ran", "not_run") else answer, phase, outcome,
                                                    detail="Parse error: x"))
                strings.append(trap_reason(result))
                strings += _strings(term_rows(answer, TPCH, QUESTION, None))

    # Re-asks and term rows for every glossary term: no fit, the first reading, every reading.
    for spec in SPECS.values():
        for term in spec["glossary"]:
            names = [r["name"] for r in term["readings"]]
            for name in names:
                question = reask_question(QUESTION, term["term"], name)
                strings.append(question)
                asked = {"term": term["term"], "reading": name}
                for fits in ([], names[:1], names):
                    for level in ("note", "warn"):
                        entry = {"term": term["term"], "readings": fits, "cued": names[-1:], "level": level}
                        results = _answer(terms=_r("terms", level, {"terms": [entry]}))
                        strings += _strings(term_rows(results, spec, question, asked))
                        strings += _strings(term_rows(results, spec, QUESTION, None))
                        strings += _strings(verdict(results, "done", "shown", asked=asked))

    for lock in (HOLDS, BROKEN):
        for kind in KINDS:
            strings += _strings(cv.tested_line(lock, kind, RATES))
    strings += cv.tested_panel(RATES)

    assert len(strings) > 1000
    bad = {s: problems(s) for s in strings if problems(s)}
    assert bad == {}
