"""Decide whether a shown result matches the answer key."""
from __future__ import annotations

import importlib.util
import itertools
import sys
from collections.abc import Iterable, Iterator, Sequence
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
REL_TOL, ABS_TOL = an.REL_TOL, an.ABS_TOL

ABS_SLACK = 1e-9        # float slack on the boundary, so exactly 0.01 counts
TIE_TOL = 1e-9          # tied gold rows differ only by float noise
SHARE_SCALE = 100       # a share may be shown as a percent
MAX_ASSIGNMENTS = 1000  # column assignments tried per result
MAX_GREEDY_ROWS = 1000  # biggest result the slow row match runs on

Assignment = tuple[tuple[int, int], ...]  # (shown column, scale) per gold column


# An int or float, but not a bool.
def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# Numbers equal within tolerance (None counts as 0); anything else exact as text.
def values_equal(a, b) -> bool:
    a = 0 if a is None else a
    b = 0 if b is None else b
    if _is_number(a) and _is_number(b):
        diff = abs(a - b)
        return diff <= ABS_TOL + ABS_SLACK or diff <= REL_TOL * max(abs(a), abs(b))
    return str(a) == str(b)


# Sort key that never compares a number with text: numbers (None as 0), then text.
def _sort_key(v) -> tuple:
    if v is None:
        return (0, 0.0)
    if _is_number(v):
        return (0, float(v))
    return (1, str(v))


# A whole row's sort key, column by column.
def _row_key(row: Sequence) -> tuple:
    return tuple(_sort_key(v) for v in row)


# A gold share as a percent; None counts as 0, text stays as it is.
def _times_100(v):
    if v is None:
        return 0
    return v * SHARE_SCALE if _is_number(v) else v


# Two rows equal on every column.
def _rows_equal(shown: Sequence, gold: Sequence) -> bool:
    return all(values_equal(s, g) for s, g in zip(shown, gold))


# Same values as a multiset: sort both sides, then compare pairwise.
def _same_values(shown: list, gold: list) -> bool:
    pairs = zip(sorted(shown, key=_sort_key), sorted(gold, key=_sort_key))
    return all(values_equal(s, g) for s, g in pairs)


# Shown columns (with scale) holding this gold column's values; a share also tries x100.
def _candidates(shown_columns: list[list], gold_values: list, is_share: bool) -> list[tuple[int, int]]:
    percents = [_times_100(v) for v in gold_values] if is_share else []
    found = []
    for c, values in enumerate(shown_columns):
        if _same_values(values, gold_values):
            found.append((c, 1))
        if is_share and _same_values(values, percents):
            found.append((c, SHARE_SCALE))
    return found


# Each way to give every gold column its own shown column, in itertools.product order.
# Remembers dead ends, so a wide result with no assignment returns fast.
def _distinct_assignments(candidates: list[list[tuple[int, int]]]) -> Iterator[Assignment]:
    dead = set()  # (gold column, shown columns used) with no way to finish

    def walk(j: int, used: frozenset) -> Iterator[Assignment]:
        if j == len(candidates):
            yield ()
            return
        if (j, used) in dead:
            return

        found = False
        for c, scale in candidates[j]:
            if c not in used:
                for rest in walk(j + 1, used | {c}):
                    found = True
                    yield ((c, scale),) + rest
        if not found:
            dead.add((j, used))

    return walk(0, frozenset())


# Shown rows cut to the assigned columns, and gold rows with x100 shares scaled.
def _project(shown_rows: list[list], gold_rows: list[list], assignment: Assignment) -> tuple[list, list]:
    shown = [tuple(row[c] for c, _ in assignment) for row in shown_rows]
    gold = [
        tuple(_times_100(v) if scale == SHARE_SCALE else v for v, (_, scale) in zip(row, assignment))
        for row in gold_rows
    ]
    return shown, gold


# Each gold row, in order, takes the first unused shown row equal to it.
def _greedy_match(shown: list, gold: list) -> bool:
    unused = list(range(len(shown)))
    for g in gold:
        hit = next((i for i in unused if _rows_equal(shown[i], g)), None)
        if hit is None:
            return False
        unused.remove(hit)
    return True


# Rows as a multiset: sorted pairwise, then a greedy match for small results.
def _rows_match_unordered(shown: list, gold: list) -> bool:
    pairs = zip(sorted(shown, key=_row_key), sorted(gold, key=_row_key))
    if all(_rows_equal(s, g) for s, g in pairs):
        return True
    return len(gold) <= MAX_GREEDY_ROWS and _greedy_match(shown, gold)


# Same sort value: numbers equal up to float noise (None as 0), text exact.
def _same_key(a, b) -> bool:
    a = 0 if a is None else a
    b = 0 if b is None else b
    if _is_number(a) and _is_number(b):
        return abs(a - b) <= TIE_TOL * max(1.0, abs(a), abs(b))
    return str(a) == str(b)


# Two gold rows with the same value in every order column.
def rows_tie(a: Sequence, b: Sequence, order_cols: Sequence[int]) -> bool:
    return all(_same_key(a[j], b[j]) for j in order_cols)


# Spans of gold rows that tie with the span's first row on every order column.
def _tie_runs(gold_rows: list[list], order_cols: Sequence[int]) -> list[tuple[int, int]]:
    runs, start = [], 0
    while start < len(gold_rows):
        first, end = gold_rows[start], start + 1
        while order_cols and end < len(gold_rows) and rows_tie(first, gold_rows[end], order_cols):
            end += 1
        runs.append((start, end))
        start = end
    return runs


# Rows in order, except each run of tied gold rows may come in any order.
def _rows_match_ordered(shown: list, gold: list, runs: list[tuple[int, int]]) -> bool:
    return all(_rows_match_unordered(shown[s:e], gold[s:e]) for s, e in runs)


# True if the shown result matches the gold one; names and extra columns don't matter.
# When ordered, gold rows tied on every order column may come in any order.
def match_result(shown_cols: list[str], shown_rows: list[list], gold_cols: list[str], gold_rows: list[list],
                 ordered: bool, share_cols: Iterable[int] = (), order_cols: Iterable[int] = ()) -> bool:
    if len(shown_rows) != len(gold_rows):
        return False
    if not gold_rows:
        return len(shown_cols) >= len(gold_cols)

    shares = set(share_cols)
    shown_columns = [[row[c] for row in shown_rows] for c in range(len(shown_cols))]
    gold_columns = [[row[j] for row in gold_rows] for j in range(len(gold_cols))]
    candidates = [_candidates(shown_columns, values, j in shares) for j, values in enumerate(gold_columns)]
    if not all(candidates):
        return False

    runs = _tie_runs(gold_rows, list(order_cols)) if ordered else []
    for assignment in itertools.islice(_distinct_assignments(candidates), MAX_ASSIGNMENTS):
        shown, gold = _project(shown_rows, gold_rows, assignment)
        if ordered:
            matched = _rows_match_ordered(shown, gold, runs)
        else:
            matched = _rows_match_unordered(shown, gold)
        if matched:
            return True
    return False
