from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import statistics
import sys
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple


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
ck = an.load_sibling("an_checks", "checks.py")
sc = an.load_sibling("an_scoring", "scoring.py")
fz = an.load_sibling("an_freeze", "freeze.py")
qm = an.load_sibling("an_questions", "questions.py")
dd = an.load_sibling("an_describe_datasets", "describe_datasets.py")

CATCH_BAR = 0.50       # bar 1: wrong shown answers warned or refused, at least
REFUSE_BAR = 0.10      # bar 2: right shown answers refused, at most
FLAG_BAR = 0.25        # bar 3: right shown answers warned or refused, at most
MIN_WRONG = 20         # fewer wrong shown answers can't measure bar 1
CRASH_LIMIT = 0.05     # share of test questions that may hit a bridge crash
MEDIAN_SECONDS = 20.0  # a model config qualifies at this median time or under
GROW_TARGET = 25       # wrong shown test answers I want projected
TEST_CAP = 300         # the most test questions
Z = 1.96               # 95% Wilson interval

GO, NO_GO, NO_VERDICT = "GO", "NO-GO", "NO VERDICT"
FLAGGED = ("warned", "refused")
FIRING = ("warn", "refuse")
CAUSES = ("model", "engine", "unknown")
EXAMPLES = 10        # examples of catches and of false alarms in the report
WATCHED_HEAD = "Changed since the freeze, not part of the verdict rules"
CAUSE_NOTE = "A shown answer to a question that can't be answered counts as the model's error, whatever SQLite does."
QUESTION_CUT = 90    # characters of a question an example shows
MESSAGE_CUT = 200    # characters of a check message an example shows
# Each bar: its name, what it counts, and its limit.
BARS = {
    "catch": ("catch (bar 1)", "wrong shown answers warned or refused", "at least", CATCH_BAR),
    "refuse": ("keep, refusals (bar 2)", "right shown answers refused", "at most", REFUSE_BAR),
    "flag": ("keep, warnings (bar 3)", "right shown answers warned or refused", "at most", FLAG_BAR),
}


# Column names from a result: bridge columns are dicts, SQLite ones plain names.
def _names(columns) -> list:
    return [c.get("name") if isinstance(c, dict) else c for c in columns or []]


# True if a result's rows match the question's answer key; False with no key.
# A truncated result never matches: no key is that long.
def _matches(result, question: dict) -> bool:
    answer = question.get("answer")
    if not isinstance(result, dict) or not isinstance(answer, dict) or result.get("truncated"):
        return False

    return sc.match_result(
        _names(result.get("columns")), result.get("rows") or [],
        list(answer.get("columns") or []), answer.get("rows") or [],
        ordered=bool(question.get("ordered")),
        share_cols=question.get("share_cols") or (), order_cols=question.get("order_cols") or (),
    )


# Who made a wrong answer: engine if SQLite's run of the SQL matches the key,
# model if it ran and doesn't, unknown if it didn't run or there's no key.
def _cause(record: dict, question: dict) -> str:
    if question.get("kind") == "unanswerable":
        return "model"

    sqlite = record.get("sqlite_cause")
    if not isinstance(sqlite, dict) or sqlite.get("ok") is not True or not isinstance(question.get("answer"), dict):
        return "unknown"
    return "engine" if _matches(sqlite, question) else "model"


# True if any try, first answer or rephrased, got a bridge crash.
def _crashed(record: dict) -> bool:
    for part in (record.get("primary") or {}, record.get("rephrase") or {}):
        for attempt in part.get("attempts") or []:
            error = (attempt.get("reply") or {}).get("error")
            if isinstance(error, dict) and error.get("kind") == "crash":
                return True
    return False


# One question's result: shown, right, the checks' label, the cause of a wrong one.
def outcome(record: dict, question: dict, spec: dict, manifest: dict, skip=()) -> dict:
    primary = record.get("primary") or {}
    final = primary.get("final")
    shown = final == "shown"

    checks, messages, label = {}, {}, "not_shown"
    if shown:
        results = ck.run_checks(record, question, spec, manifest)
        checks = {r.name: r.level for r in results}
        messages = {r.name: r.message for r in results if r.level in FIRING}
        label = ck.overall(results, skip)

    if question.get("kind") == "unanswerable":
        right = final == "cannot"
    else:
        right = shown and _matches(primary.get("result"), question)

    return {
        "qid": record.get("qid"), "shown": shown, "right": right, "label": label,
        "cause": _cause(record, question) if shown and not right else None,
        "checks": checks, "messages": messages, "crash": _crashed(record), "seconds": record.get("seconds_total"),
    }


# Copies of shown outcomes labeled again with some checks left out.
def relabel(outcomes: list[dict], skip) -> list[dict]:
    out = []
    for o in outcomes:
        if o["shown"]:
            results = [ck.CheckResult(name, level, "", {}) for name, level in o["checks"].items()]
            o = {**o, "label": ck.overall(results, skip)}
        out.append(o)
    return out


# 95% Wilson interval for count of total, or None when total is 0.
def wilson(count: int, total: int, z: float = Z) -> tuple[float, float] | None:
    if total == 0:
        return None

    p = count / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# One bar's counts, rate, interval and whether it holds.
# Compared as exact fractions, so 25 of 100 meets 0.25.
def _bar(count: int, total: int, limit: float, at_least: bool) -> dict:
    if total == 0:
        return {"count": 0, "total": 0, "rate": None, "interval": None, "holds": False}
    share, bar = Fraction(count, total), Fraction(str(limit))

    return {
        "count": count,
        "total": total,
        "rate": count / total,
        "interval": wilson(count, total),
        "holds": share >= bar if at_least else share <= bar
    }


# The three bars over shown answers: catch, refuse and flag.
def bars(outcomes: list[dict]) -> dict:
    wrong = [o for o in outcomes if o["shown"] and not o["right"]]
    right = [o for o in outcomes if o["shown"] and o["right"]]
    return {
        "catch": _bar(sum(o["label"] in FLAGGED for o in wrong), len(wrong), CATCH_BAR, True),
        "refuse": _bar(sum(o["label"] == "refused" for o in right), len(right), REFUSE_BAR, False),
        "flag": _bar(sum(o["label"] in FLAGGED for o in right), len(right), FLAG_BAR, False),
    }


# One line per answerable question of the split with no frozen answer.
def check_key(questions: list[dict], split: str) -> list[str]:
    in_split = [q for q in questions if q.get("split") == split]
    if not in_split:
        return [f"no {split} questions: the key isn't frozen"]

    return [f"{q.get('qid')} has no frozen answer (an unresolved disagreement)" for q in in_split if q.get("kind") != "unanswerable" and not isinstance(q.get("answer"), dict)]


# One line if any test record wasn't stamped with today's freeze.json; [] when all were.
def _stamp_problems(results: Path, records: list[dict]) -> list[str]:
    today = fz.freeze_sha1(results)
    if today is None:
        return []  # verify already names the missing or unreadable freeze.json

    missing = sum(r.get("freeze_sha1") is None for r in records)
    other = sum(r.get("freeze_sha1") not in (None, today) for r in records)
    if not missing and not other:
        return []
    noun = "record wasn't" if missing + other == 1 else "records weren't"
    return [f"{missing + other} of {len(records)} test {noun} made under today's freeze.json (freeze_sha1: {other} from another freeze, {missing} missing)"]


# Freeze problems: changed files, records from another freeze, run models that aren't frozen.
def check_freeze(results: Path, code_dir: Path, model: str, records: list[dict]) -> list[str]:
    try:
        configs = fz.read_model_configs(Path(code_dir) / "model_mlx.py")
    except (OSError, SyntaxError, ValueError) as e:
        return [f"can't read model configs: {e}"]

    problems = fz.verify(results, code_dir, configs) + _stamp_problems(results, records)
    frozen = fz.frozen_model_id(results)
    if frozen is not None:
        ids = {model} | {r.get("model") for r in records}
        problems += [f"run model {m} isn't the frozen model {frozen}" for m in sorted(ids, key=str) if m != frozen]
    return problems


# "n of total (rate)", for reasons and tables.
def _share(count: int, total: int) -> str:
    return f"{count} of {total} ({count / total:.3f})" if total else f"{count} of 0"


# How many of some wrong answers each cause made: "model 3, engine 1, unknown 0".
def _causes(wrong: list[dict]) -> str:
    return ", ".join(f"{c} {sum(o['cause'] == c for o in wrong)}" for c in CAUSES)


# A count of wrong answers with its causes: "4 (model 3, engine 1, unknown 0)".
def _count_causes(wrong: list[dict]) -> str:
    return f"{len(wrong)} ({_causes(wrong)})"


# Caught of wrong per cause, like "model 2 of 3, engine 0 of 1".
def _caught_by_cause(wrong: list[dict]) -> str:
    parts = []
    for c in CAUSES:
        mine = [o for o in wrong if o["cause"] == c]
        parts.append(f"{c} {sum(o['label'] in FLAGGED for o in mine)} of {len(mine)}")
    return ", ".join(parts)


# One line saying whether a bar holds, and why.
def _bar_line(name: str, bar: dict) -> str:
    title, counts, side, limit = BARS[name]
    if bar["rate"] is None:
        return f"{title}: no {'wrong' if name == 'catch' else 'right'} shown answers, so no rate; it doesn't hold"

    state = "holds" if bar["holds"] else "fails"
    return f"{title} {state}: {_share(bar['count'], bar['total'])} {counts}, needs {side} {limit:.2f}"


# GO, NO-GO or NO VERDICT, with every reason. Test questions are the outcomes plus missing.
def verdict(outcomes: list[dict], freeze_problems: list[str], key_problems: list[str], missing=()) -> tuple[str, list[str]]:
    b = bars(outcomes)
    reasons = []
    wrong = b["catch"]["total"]
    if wrong < MIN_WRONG:
        noun = "answer" if wrong == 1 else "answers"
        reasons.append(f"only {wrong} wrong shown {noun}; at least {MIN_WRONG} are needed to measure the checks")

    reasons += [f"freeze: {p}" for p in freeze_problems]
    reasons += [f"answer key: {p}" for p in key_problems]
    if missing:
        listed = ", ".join(sorted(missing)[:EXAMPLES]) + (", ..." if len(missing) > EXAMPLES else "")
        noun = "question has" if len(missing) == 1 else "questions have"
        reasons.append(f"unfinished run: {len(missing)} test {noun} no record ({listed})")

    tested = len(outcomes) + len(missing)
    crashed = sum(bool(o["crash"]) for o in outcomes)
    if tested and Fraction(crashed, tested) > Fraction(str(CRASH_LIMIT)):
        reasons.append(f"bridge crashes on {crashed} of {tested} test questions, over {CRASH_LIMIT:.0%}")
    if reasons:
        return NO_VERDICT, reasons

    failed = [_bar_line(name, bar) for name, bar in b.items() if not bar["holds"]]
    if failed:
        return NO_GO, failed
    return GO, [_bar_line(name, bar) for name, bar in b.items()]


# Median and p90 of the numbers in values; p90 is the nearest-rank value.
def _median_p90(values) -> tuple[float | None, float | None]:
    nums = sorted(v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool))
    if not nums:
        return None, None
    return statistics.median(nums), nums[math.ceil(0.9 * len(nums)) - 1]


# Per check: catches (wrong shown it fired on), sole catches, both by cause, false alarms.
def per_check(outcomes: list[dict]) -> dict[str, dict]:
    counts = {
        name: {
            "catches": 0, 
            "causes": dict.fromkeys(CAUSES, 0), 
            "sole": 0,
            "sole_causes": dict.fromkeys(CAUSES, 0), 
            "false_alarms": 0
        } for name in ck.CHECKS
    }
    for o in outcomes:
        if not o["shown"]:
            continue

        fired = [name for name, level in o["checks"].items() if level in FIRING and name in counts]
        for name in fired:
            if o["right"]:
                counts[name]["false_alarms"] += 1
            else:
                counts[name]["catches"] += 1
                counts[name]["causes"][o["cause"]] = counts[name]["causes"].get(o["cause"], 0) + 1

        if len(fired) == 1 and not o["right"]:
            sole = counts[fired[0]]
            sole["sole"] += 1
            sole["sole_causes"][o["cause"]] = sole["sole_causes"].get(o["cause"], 0) + 1
    return counts


class Analysis(NamedTuple):
    outcomes: list[dict]    # one per question of the split with a record, in qid order
    missing: list[str]      # qids of the split with no record
    ignored: int            # records for qids outside the split
    repeats: int            # second records for a qid already seen
    questions: dict         # qid -> question, for the split
    records: dict           # qid -> the record scored


# Each dataset's spec and manifest, each read once.
class _Context:
    def __init__(self, results: Path):
        self.paths = an.paths(results)
        self.loaded: dict[str, tuple[dict, dict]] = {}

    def get(self, ds: str) -> tuple[dict, dict]:
        if ds not in self.loaded:
            manifest = json.loads((self.paths.dataset_dir(ds) / "manifest.json").read_text(encoding="utf-8"))
            self.loaded[ds] = (dd.load_spec(ds), manifest)
        return self.loaded[ds]


# Outcomes for one split's questions; records for other qids are counted, not scored.
def analyze(records: list[dict], questions: list[dict], split: str, context: _Context) -> Analysis:
    in_split = sorted((q for q in questions if q.get("split") == split), key=lambda q: str(q.get("qid")))
    wanted = {q["qid"]: q for q in in_split}
    by_qid, ignored, repeats = {}, 0, 0
    for r in records:
        qid = r.get("qid")
        if not isinstance(qid, str) or qid not in wanted:
            ignored += 1
        elif qid in by_qid:
            repeats += 1
        else:
            by_qid[qid] = r

    outcomes = [outcome(by_qid[qid], q, *context.get(q["dataset"])) for qid, q in wanted.items() if qid in by_qid]
    missing = [qid for qid in wanted if qid not in by_qid]
    return Analysis(outcomes, missing, ignored, repeats, wanted, by_qid)


# A Markdown table.
def _table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"]
    return lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


# A rate and its interval, or n/a.
def _rate_text(bar: dict) -> tuple[str, str]:
    if bar["rate"] is None:
        return "n/a", "n/a"
    lo, hi = bar["interval"]
    return f"{bar['rate']:.3f}", f"{lo:.3f}-{hi:.3f}"


# Text cut to n characters, with ... when cut.
def _cut(text, n: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n - 3].rstrip() + "..."


# Counts for one group of questions: accuracy, then the bars' parts.
def _group_row(name: str, qids: list[str], by_qid: dict) -> list:
    outs = [by_qid[qid] for qid in qids if qid in by_qid]
    b = bars(outs)
    right = sum(o["right"] for o in outs)
    wrong = [o for o in outs if o["shown"] and not o["right"]]

    return [
        name, len(qids), _share(right, len(qids)),
        _share(b["catch"]["count"], b["catch"]["total"]),
        _share(b["refuse"]["count"], b["refuse"]["total"]),
        _share(b["flag"]["count"], b["flag"]["total"]),
        _caught_by_cause(wrong)
    ]


GROUP_HEADER = ["", "Questions", "Right without checks", "Wrong caught", "Right refused", "Right flagged", "Wrong caught, by cause"]


# Qids of the questions whose field key equals value.
def _qids_where(questions: dict, key: str, value) -> list[str]:
    return [qid for qid, q in questions.items() if q.get(key) == value]


# The without / with table and the not-shown counts.
def _without_with(a: Analysis) -> list[str]:
    shown = [o for o in a.outcomes if o["shown"]]
    right = [o for o in shown if o["right"]]
    wrong = [o for o in shown if not o["right"]]
    flagged = sum(o["label"] in FLAGGED for o in right)
    caught = [o for o in wrong if o["label"] in FLAGGED]
    silent = [o for o in wrong if o["label"] not in FLAGGED]
    lines = _table(["Shown answers", "Without checks", "With checks"], [
        ["Right, shown clean", len(right), len(right) - flagged],
        ["Right, warned or refused (false alarm)", 0, flagged],
        ["Wrong, shown clean (silent)", _count_causes(wrong), _count_causes(silent)],
        ["Wrong, warned or refused (caught)", _count_causes([]), _count_causes(caught)],
    ])

    not_shown = [o for o in a.outcomes if not o["shown"]]
    cannot = [o for o in not_shown if (a.records[o["qid"]].get("primary") or {}).get("final") == "cannot"]
    correct = sum(1 for o in cannot if o["right"])
    other = len(cannot) - correct
    lines += ["", *_table(["Not shown", "Count"], [
        ['Correct "can\'t answer" (unanswerable question)', correct],
        ['Other "can\'t answer"', other],
        ["Failed (every try errored, crashed, timed out or ran out of memory)", len(not_shown) - correct - other],
        ["No record", len(a.missing)],
    ])]
    return lines


# One example line: qid, dataset, kind, the question cut short, the firing messages.
def _example(o: dict, question: dict) -> str:
    fired = [f"{name} ({o['checks'][name]}): {_cut(o['messages'][name], MESSAGE_CUT)}" for name in ck.CHECKS if name in o["messages"]]
    text = _cut(question.get("question"), QUESTION_CUT)
    return f"- {o['qid']} ({question.get('dataset')}, {question.get('kind')}): \"{text}\" -- " + "; ".join(fired)


# The per-check table: catches and sole catches, each split by cause, then false alarms.
def _check_table(outcomes: list[dict]) -> list[str]:
    counts = per_check(outcomes)
    rows = []
    for name, c in counts.items():
        causes = ", ".join(f"{cause} {c['causes'].get(cause, 0)}" for cause in CAUSES)
        sole_causes = ", ".join(f"{cause} {c['sole_causes'].get(cause, 0)}" for cause in CAUSES)
        rows.append([name, c["catches"], causes, c["sole"], sole_causes, c["false_alarms"]])

    lines = _table(["Check", "Catches", "Catches by cause", "Only catch", "Only catch by cause", "False alarms"], rows)
    return lines + ["", "A catch counts for every check that fired on it; exact always passes today, so it can't catch anything."]


# The one line under the verdict that lists watched files changed since the freeze.
def _watched_line(watched: list[str]) -> str:
    return f"{WATCHED_HEAD}: {'; '.join(watched)}."


# The test report: verdict, reasons and watched files first, then accuracy, bars and breakdowns.
def report_text(model: str, a: Analysis, decision: str, reasons: list[str], watched=(), pick_line: str = "") -> str:
    by_qid = {o["qid"]: o for o in a.outcomes}
    tested = len(a.outcomes) + len(a.missing)
    right = sum(o["right"] for o in a.outcomes)
    crashed = sum(bool(o["crash"]) for o in a.outcomes)
    lines = [f"**{decision}**", "", *[f"- {r}" for r in reasons], ""]
    if watched:
        lines += [_watched_line(watched), ""]
        
    lines += [
        f"Accuracy without checks: {right} of {tested} test questions right ({right / tested:.3f}); "
        f"a correct \"can't answer\" counts as right."
        if tested else "Accuracy without checks: no test questions.", ""
    ]
    lines += [f"Model {model}: {len(a.outcomes)} records for {tested} test questions; bridge crashes on {crashed} of them (at most {CRASH_LIMIT:.0%} allowed).", ""]
    if pick_line:
        lines += [pick_line, ""]
    if a.ignored:
        lines += [f"Ignored {a.ignored} {'record' if a.ignored == 1 else 'records'} for qids outside the test split.", ""]
    if a.repeats:
        lines += [f"Ignored {a.repeats} repeated {'record' if a.repeats == 1 else 'records'} (the first one per qid counts).", ""]

    rows = []
    for name, bar in bars(a.outcomes).items():
        title, counts, side, limit = BARS[name]
        rows.append([
            f"{title}: {counts}", f"{bar['count']} of {bar['total']}", *_rate_text(bar),
            f"{side} {limit:.2f}", "yes" if bar["holds"] else "no"
        ])
    lines += ["## The bars", "", *_table(["Bar", "Count", "Rate", "95% Wilson", "Needs", "Holds"], rows), ""]

    wrong = [o for o in a.outcomes if o["shown"] and not o["right"]]
    cause_rows = []
    for cause in CAUSES:
        bar = bars([o for o in wrong if o["cause"] == cause])["catch"]
        cause_rows.append([f"cause {cause}", f"{bar['count']} of {bar['total']}", *_rate_text(bar)])

    without = relabel(wrong, {"engine_traps"})
    bar = bars(without)["catch"]
    caught = f"{bar['count']} of {bar['total']} ({_caught_by_cause(without)})"
    cause_rows.append(["all, without engine_traps", caught, *_rate_text(bar)])
    lines += ["### Bar 1 by cause, and without engine_traps", "", *_table(["Wrong shown answers", "Caught", "Rate", "95% Wilson"], cause_rows), ""]
    lines += [CAUSE_NOTE, ""]

    lines += ["## Without and with checks", "", *_without_with(a), ""]

    datasets = sorted({q.get("dataset") for q in a.questions.values()}, key=str)
    kinds = [k for k in qm.KINDS if any(q.get("kind") == k for q in a.questions.values())]
    by_dataset = [_group_row(ds, _qids_where(a.questions, "dataset", ds), by_qid) for ds in datasets]
    by_kind = [_group_row(k, _qids_where(a.questions, "kind", k), by_qid) for k in kinds]
    lines += ["## Per dataset", "", *_table(GROUP_HEADER, by_dataset), ""]
    lines += ["## Per kind", "", *_table(GROUP_HEADER, by_kind), ""]
    lines += ["## Per check", "", *_check_table(a.outcomes), ""]

    median, p90 = _median_p90(o["seconds"] for o in a.outcomes)
    timing = f"median {median:.1f} s, p90 {p90:.1f} s" if median is not None else "no times recorded"
    lines += ["## Time per question", "", f"First answer plus rephrase: {timing} (p90 by nearest rank).", ""]

    shown = [o for o in a.outcomes if o["shown"] and o["label"] in FLAGGED]
    catches = [o for o in shown if not o["right"]][:EXAMPLES]
    alarms = [o for o in shown if o["right"]][:EXAMPLES]
    lines += ["## Examples: catches", "", *([_example(o, a.questions[o["qid"]]) for o in catches] or ["None."]), ""]
    lines += ["## Examples: false alarms", "", *([_example(o, a.questions[o["qid"]]) for o in alarms] or ["None."]), ""]
    return "\n".join(lines)


# One CSV row per question of the split, with each check's level.
# A question with no record has label no_record and empty record fields.
def cells_csv(a: Analysis) -> str:
    by_qid = {o["qid"]: o for o in a.outcomes}
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["qid", "dataset", "kind", "shown", "right", "label", "cause", "crash", "seconds", *ck.CHECKS])
    for qid, q in a.questions.items():
        o = by_qid.get(qid)
        if o is None:
            w.writerow([qid, q.get("dataset"), q.get("kind"), "", "", "no_record", "", "", "", *[""] * len(ck.CHECKS)])
            continue
        w.writerow([
            qid, q.get("dataset"), q.get("kind"), o["shown"], o["right"], o["label"], o["cause"] or "",
            o["crash"], "" if o["seconds"] is None else o["seconds"],
            *(o["checks"].get(n, "") for n in ck.CHECKS)
        ])
    return buf.getvalue()


# One split's run files under the results folder, in name order.
def _run_files(results: Path, split: str) -> list[Path]:
    pattern = an.paths(results).run_file(split, "*")
    return sorted(pattern.parent.glob(pattern.name))


# The only test run file's model; None, with a printed reason, for zero or several.
def _only_test_run_model(results: Path) -> str | None:
    runs = _run_files(results, "test")
    if len(runs) == 1:
        return runs[0].name[len("test_"):-len(".jsonl")]

    if runs:
        found = f"{len(runs)} test run files ({', '.join(r.name for r in runs)})"
    else:
        found = f"no test run file in {an.paths(results).run_file('test', '*').parent}"
    print(f"no frozen model and {found}; pass --model", file=sys.stderr)
    return None


# The frozen model beside the dev pick, saying plainly whether they differ.
def _pick_line(frozen: str | None, chosen: str | None, why: str) -> str:
    if chosen is None:
        return f"Frozen model {frozen or 'none'}; dev pick not available ({why})."
    if frozen is None:
        return f"Frozen model none; dev pick {chosen}."
    if chosen == frozen:
        return f"Frozen model {frozen}; dev pick {chosen}, the same."
    return f"Frozen model {frozen}; dev pick {chosen}: they differ."


# Analyze the test run of one model: report.md, cells.csv; 1 on NO VERDICT.
# With no model given or frozen, the only test run file names it.
def run_test(results: Path, model: str | None, code_dir: Path) -> int:
    p = an.paths(results)
    frozen = fz.frozen_model_id(results)
    model = model or frozen or _only_test_run_model(results)
    if model is None:
        return 1

    run_file = p.run_file("test", model)
    if not run_file.is_file():
        print(f"no run file {run_file}", file=sys.stderr)
        return 1

    records = an.read_jsonl(run_file)
    questions = qm.read_questions(p.questions)
    context = _Context(results)
    a = analyze(records, questions, "test", context)
    freeze_problems = check_freeze(results, code_dir, model, records)
    decision, reasons = verdict(a.outcomes, freeze_problems, check_key(questions, "test"), a.missing)
    watched = fz.watched_changes(results, code_dir)
    pick_line = _pick_line(frozen, *dev_pick(results, questions, context))

    an.write_text_atomic(p.report, report_text(model, a, decision, reasons, watched, pick_line))
    an.write_text_atomic(p.cells, cells_csv(a))
    print(decision)
    for r in reasons:
        print(f"- {r}")
    if watched:
        print(_watched_line(watched))
    print(f"wrote {p.report} and {p.cells}")
    return 1 if decision == NO_VERDICT else 0


# One dev config: records, right, accuracy over all dev questions, median time, wrong shown.
def dev_summary(outcomes: list[dict], dev_count: int) -> dict:
    right = sum(o["right"] for o in outcomes)
    median, _ = _median_p90(o["seconds"] for o in outcomes)

    return {
        "answered": len(outcomes),
        "right": right,
        "accuracy": right / dev_count if dev_count else 0.0,
        "median": median,
        "wrong": sum(1 for o in outcomes if o["shown"] and not o["right"]),
    }


# The best accuracy at a median of MEDIAN_SECONDS or under, ties to faster then id.
# If none qualify, the fastest. Returns the pick and why.
def pick(summaries: dict[str, dict]) -> tuple[str | None, str]:
    timed = {c: s for c, s in summaries.items() if s["median"] is not None}
    if not timed:
        return None, "no config has a record, so there's no pick"

    qualify = {c: s for c, s in timed.items() if s["median"] <= MEDIAN_SECONDS}
    if qualify:
        best = min(qualify, key=lambda c: (-qualify[c]["right"], qualify[c]["median"], c))
        return best, f"best accuracy among configs with a median of at most {MEDIAN_SECONDS:.1f} s"

    fastest = min(timed, key=lambda c: (timed[c]["median"], c))
    return fastest, f"no config has a median of at most {MEDIAN_SECONDS:.1f} s, so I took the fastest"


# Projected wrong shown test answers, and the test questions to add to reach GROW_TARGET.
# Integer math: projected under target means wrong x test < target x dev.
def grow(wrong: int, dev_count: int, test_count: int) -> dict:
    enough = wrong * test_count >= GROW_TARGET * dev_count
    if enough:
        needed = 0
    elif wrong == 0:
        needed = TEST_CAP - test_count
    else:
        needed = min(TEST_CAP, -(-GROW_TARGET * dev_count // wrong)) - test_count
    needed = max(0, needed)

    return {
        "wrong": wrong, "rate": wrong / dev_count, "projected": wrong * test_count / dev_count,
        "enough": enough, "needed": needed, "at_cap": wrong * TEST_CAP / dev_count,
        "short_at_cap": test_count + needed == TEST_CAP and wrong * TEST_CAP < GROW_TARGET * dev_count
    }


# The grow line for the pick.
def _grow_line(g: dict, dev_count: int, test_count: int) -> str:
    head = (f"grow: the pick has {_share(g['wrong'], dev_count)} wrong shown dev answers, projecting {g['projected']:.1f} wrong shown answers on {test_count} test questions")
    if g["enough"]:
        return f"{head}, at least {GROW_TARGET}: no new questions needed"
    if g["needed"] == 0:
        return f"{head}, under {GROW_TARGET}, but the test set is already at the cap of {TEST_CAP}"
    line = (f"{head}, under {GROW_TARGET}: add {g['needed']} test questions (to {test_count + g['needed']}, cap {TEST_CAP})")
    if g["short_at_cap"]:
        line += f"; even at {TEST_CAP} the projection is {g['at_cap']:.1f}, still under {GROW_TARGET}"
    return line


# Each dev run file's analysis and summary, keyed by config in file-name order.
def _dev_runs(files: list[Path], questions: list[dict], context: _Context, dev_count: int) -> tuple[dict, dict]:
    analyses, summaries = {}, {}
    for f in files:
        config = f.name[len("dev_"):-len(".jsonl")]
        analyses[config] = analyze(an.read_jsonl(f), questions, "dev", context)
        summaries[config] = dev_summary(analyses[config].outcomes, dev_count)
    return analyses, summaries


# dev_report.md's pick from the dev runs, or None and why there's none.
# A dev file that doesn't read gives no pick rather than stopping the test report.
def dev_pick(results: Path, questions: list[dict], context: _Context) -> tuple[str | None, str]:
    files = _run_files(results, "dev")
    dev_count = sum(q.get("split") == "dev" for q in questions)
    if not files:
        return None, "no dev runs"
    if dev_count == 0:
        return None, "no dev questions"

    try:
        _, summaries = _dev_runs(files, questions, context, dev_count)
    except Exception as e:   # a file or record in any wrong shape
        return None, f"the dev runs don't read: {e}"
    return pick(summaries)


# Every dev run: accuracy and time per config, the pick, the grow projection; dev_report.md.
def run_dev(results: Path) -> int:
    p = an.paths(results)
    files = _run_files(results, "dev")
    if not files:
        print(f"no dev runs in {p.run_file('dev', '*').parent}", file=sys.stderr)
        return 1

    questions = qm.read_questions(p.questions)
    dev_count = sum(q.get("split") == "dev" for q in questions)
    test_count = sum(q.get("split") == "test" for q in questions)
    if dev_count == 0:
        print(f"no dev questions in {p.questions}", file=sys.stderr)
        return 1

    analyses, summaries = _dev_runs(files, questions, _Context(results), dev_count)

    config_lines = []
    for config, s in summaries.items():
        median = "n/a" if s["median"] is None else f"{s['median']:.1f} s"
        right = _share(s["right"], dev_count)
        config_lines.append(f"{config}: answered {s['answered']} of {dev_count}, right {right}, median {median}")

    chosen, why = pick(summaries)
    decision = []
    incomplete = [c for c, s in summaries.items() if s["answered"] < dev_count]
    if incomplete:
        listed = ", ".join(f"{c} ({summaries[c]['answered']} of {dev_count})" for c in incomplete)
        tail = f"; the pick {chosen} is one of them" if chosen in incomplete else ""
        decision.append(f"incomplete runs: {listed}{tail}")

    decision.append(f"pick: {chosen} ({why})" if chosen else f"pick: none ({why})")
    if chosen:
        decision.append(_grow_line(grow(summaries[chosen]["wrong"], dev_count, test_count), dev_count, test_count))
    print("\n".join(config_lines + decision))

    report = [
        "# Dev runs: model choice", "",
        f"Accuracy is right without checks over all {dev_count} dev questions: a correct \"can't answer\" counts, "
        "a visible failure or a missing record doesn't. Time is the median seconds per question over the "
        "config's records, first answer plus rephrase.", ""
    ]
    rows = []
    for c, s in summaries.items():
        fast = s["median"] is not None and s["median"] <= MEDIAN_SECONDS
        median = "n/a" if s["median"] is None else f"{s['median']:.1f}"
        rows.append([c, f"answered {s['answered']} of {dev_count}", _share(s["right"], dev_count), median, s["wrong"], "yes" if fast else "no"])
    header = ["Config", "Records", "Right", "Median s", "Wrong shown", f"Median <= {MEDIAN_SECONDS:.1f} s"]
    report += [*_table(header, rows), ""]
    report += [f"- {line}" for line in decision] + [""]
    if chosen:
        report += [f"## Per check on {chosen}", "", *_check_table(analyses[chosen].outcomes), ""]
    an.write_text_atomic(p.dev_report, "\n".join(report))
    print(f"wrote {p.dev_report}")
    return 0


# The verdict on test (1 on NO VERDICT), or the model choice on dev.
def main(argv: list[str] | None = None, code_dir: Path | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score saved answers: the verdict on test, the model choice on dev.")
    ap.add_argument("--split", required=True, choices=qm.SPLITS)
    ap.add_argument("--model", help="test only: the config id to analyze (default: the frozen one)")
    ap.add_argument("--results", type=Path, default=an.RESULTS)
    args = ap.parse_args(argv)
    if args.split == "dev" and args.model:
        ap.error("--model is only for --split test")
    code_dir = Path(code_dir or an.HERE)

    try:
        an.refuse_ai_root(args.results)
        return run_test(args.results, args.model, code_dir) if args.split == "test" else run_dev(args.results)
    except (OSError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
