"""Unit tests for bench/analyst/scoring.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


# Load a module by path under an an_ name, clear of other bench tests.
def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, HERE.parent / file_name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mod = _load("an_scoring", "scoring.py")


def test_values_equal_tolerances():
    assert mod.values_equal(1000000.0, 1000050.0)         # relative 1e-4
    assert mod.values_equal(0.123, 0.13)                   # absolute 0.01
    assert not mod.values_equal(1.0, 1.02)
    assert mod.values_equal(None, 0) and mod.values_equal(3, 3.0)
    assert not mod.values_equal("GERMANY", "Germany")
def test_extra_columns_and_names_ignored():
    assert mod.match_result(["n", "rev", "cnt"], [["A", 1.0, 5], ["B", 2.0, 6]],
                            ["region", "revenue"], [["B", 2.0], ["A", 1.0]], ordered=False)
def test_ordered_respects_order():
    rows = [["A", 2.0], ["B", 1.0]]
    assert not mod.match_result(["x", "y"], rows[::-1], ["x", "y"], rows, ordered=True)
    assert mod.match_result(["x", "y"], rows[::-1], ["x", "y"], rows, ordered=False)
def test_share_matches_times_100():
    assert mod.match_result(["s"], [[13.9]], ["share"], [[0.139]], ordered=False, share_cols=[0])
    assert not mod.match_result(["s"], [[13.9]], ["share"], [[0.139]], ordered=False)
def test_row_count_and_empty():
    assert not mod.match_result(["a"], [[1]], ["a"], [[1], [2]], ordered=False)
    assert mod.match_result(["a", "b"], [], ["a"], [], ordered=False)
def test_duplicate_valued_columns_still_pair_rows():   # two shown columns with equal multisets
    assert not mod.match_result(["a", "b"], [[1, 2], [2, 1]], ["a", "b"], [[1, 1], [2, 2]], ordered=False)


def test_values_equal_bool_is_text():
    assert not mod.values_equal(True, 1)                   # "True" vs "1"
    assert mod.values_equal(True, True)
    assert not mod.values_equal("1995-01-01", "1995-1-1")  # dates are exact text


def test_empty_needs_enough_shown_columns():
    assert not mod.match_result(["a"], [], ["a", "b"], [], ordered=False)


def test_gold_columns_need_distinct_shown_columns():
    assert not mod.match_result(["a"], [[1]], ["a", "b"], [[1, 1]], ordered=False)


def test_unordered_pairs_rows_within_tolerance():
    # Floats off in the last bits, in opposite directions: a plain sort mis-pairs the names.
    gold = [[0.30000000000000004, "A"], [0.3, "B"]]
    shown = [["A", 0.3], ["B", 0.30000000000000004]]
    assert mod.match_result(["n", "v"], shown, ["v", "n"], gold, ordered=False)


def test_greedy_match_stops_past_1000_rows():
    gold = [[0.30000000000000004, "A"], [0.3, "B"]]
    shown = [[0.3, "A"], [0.30000000000000004, "B"]]
    for fillers, expected in ((998, True), (999, False)):
        extra = [[float(i + 1), f"r{i}"] for i in range(fillers)]
        assert mod.match_result(["v", "n"], shown + extra, ["v", "n"], gold + extra, ordered=False) is expected


def test_mixed_type_column_does_not_raise():
    assert mod.match_result(["a"], [["x"], [1]], ["a"], [[1], ["x"]], ordered=False)
    assert not mod.match_result(["a"], [["x"], [2]], ["a"], [[1], ["x"]], ordered=False)


def test_share_with_none_and_plain_match():
    gold = [["A", None], ["B", 0.25]]
    assert mod.match_result(["n", "s"], [["A", 0.0], ["B", 25.0]], ["n", "s"], gold, ordered=False, share_cols=[1])
    assert not mod.match_result(["n", "s"], [["A", 0.0], ["B", 25.0]], ["n", "s"], gold, ordered=False)
    assert mod.match_result(["s"], [[0.139]], ["share"], [[0.139]], ordered=False, share_cols=[0])


def test_assignment_limit_is_1000():
    # Only the last six shown columns pair the rows; over 5,000 assignments come first.
    gold = [[0] * 6, [1] * 6]
    late = [[1, 1] + [0] * 6, [0, 0] + [1] * 6]
    early = [[0] * 6 + [1, 1], [1] * 6 + [0, 0]]
    assert not mod.match_result(list("abcdefgh"), late, list("abcdef"), gold, ordered=False)
    assert mod.match_result(list("abcdefgh"), early, list("abcdef"), gold, ordered=False)


def test_wide_result_with_no_assignment_is_fast():
    # Twelve gold columns all fit the same eleven shown columns: no assignment exists.
    gold, shown = [[0] * 12, [1] * 12], [[0] * 11, [1] * 11]
    assert not mod.match_result(list("abcdefghijk"), shown, list("abcdefghijkl"), gold, ordered=False)


def test_values_equal_includes_exact_cent():
    assert mod.values_equal(1.0, 1.01) and mod.values_equal(100.0, 100.01)
    assert not mod.values_equal(1.0, 1.0101)


def test_ordered_tied_rows_may_swap():
    gold = [["A", 10], ["B", 7], ["C", 7]]
    tied, untied = [["A", 10], ["C", 7], ["B", 7]], [["B", 7], ["A", 10], ["C", 7]]
    assert mod.match_result(["n", "c"], tied, ["n", "c"], gold, ordered=True, order_cols=[1])
    assert not mod.match_result(["n", "c"], untied, ["n", "c"], gold, ordered=True, order_cols=[1])


def test_ordered_without_order_cols_is_strict():
    gold = [["A", 10], ["B", 7], ["C", 7]]
    assert not mod.match_result(["n", "c"], [["A", 10], ["C", 7], ["B", 7]], ["n", "c"], gold, ordered=True)


def test_ordered_chained_near_values_keep_order():
    gold = [[1.000, "A"], [1.008, "B"], [1.016, "C"]]
    assert not mod.match_result(["v", "n"], gold[::-1], ["v", "n"], gold, ordered=True, order_cols=[0])
    shares = [[f"R{i}", 0.200 + 0.003 * i] for i in range(5)]
    assert not mod.match_result(["r", "s"], shares[::-1], ["r", "s"], shares, ordered=True, order_cols=[1])


def test_ordered_by_name_with_equal_counts_is_strict():
    gold = [["ALGERIA", 5], ["ARGENTINA", 5], ["BRAZIL", 3]]
    swapped = [gold[1], gold[0], gold[2]]
    assert not mod.match_result(["n", "c"], swapped, ["n", "c"], gold, ordered=True, order_cols=[0])
    assert not mod.match_result(["n", "c"], swapped[:2], ["n", "c"], gold[:2], ordered=True, order_cols=[0])


def test_ordered_by_date_with_equal_counts_is_strict():
    gold = [["1995-01-01", 12], ["1995-02-01", 12]]
    assert not mod.match_result(["d", "c"], gold[::-1], ["d", "c"], gold, ordered=True, order_cols=[0])


def test_ordered_ties_on_sort_column_ignore_other_numbers():
    for revs in ((100.0, 50.0, 80.0, 10.0), (100.0, 80.0, 50.0, 10.0)):   # rev rising or not
        gold = [[n, c, r] for n, c, r in zip("ABCD", (9, 7, 7, 3), revs)]
        shown = [gold[0], gold[2], gold[1], gold[3]]
        assert mod.match_result(["n", "c", "r"], shown, ["n", "c", "r"], gold, ordered=True, order_cols=[1])


def test_ordered_top3_tied_pair_with_names_may_swap():
    gold = [["MANGO", 10], ["APPLE", 7], ["KIWI", 7]]
    shown = [["MANGO", 10], ["KIWI", 7], ["APPLE", 7]]
    assert mod.match_result(["n", "c"], shown, ["n", "c"], gold, ordered=True, order_cols=[1])


def test_ordered_all_tied_rows_come_in_any_order():
    gold = [["MANGO", 7], ["APPLE", 7], ["KIWI", 7]]
    for shown in (gold[::-1], gold[1:] + gold[:1]):
        assert mod.match_result(["n", "c"], shown, ["n", "c"], gold, ordered=True, order_cols=[1])


def test_ordered_by_two_columns():
    gold = [["FRANCE", 1995, 10], ["FRANCE", 1996, 12], ["GERMANY", 1995, 8], ["GERMANY", 1995, 9]]
    years_swapped = [gold[1], gold[0], gold[2], gold[3]]
    both_tied_swapped = [gold[0], gold[1], gold[3], gold[2]]
    cols = ["n", "y", "c"]
    assert not mod.match_result(cols, years_swapped, cols, gold, ordered=True, order_cols=[0, 1])
    assert mod.match_result(cols, both_tied_swapped, cols, gold, ordered=True, order_cols=[0, 1])


def test_ordered_tie_is_float_noise_only():
    gold = [["A", 0.30000000000000004], ["B", 0.3], ["C", 0.1]]
    shown = [["B", 0.3], ["A", 0.30000000000000004], ["C", 0.1]]
    assert mod.match_result(["n", "v"], shown, ["n", "v"], gold, ordered=True, order_cols=[1])
    near = [["A", 10.004], ["B", 10.0]]                    # equal within 0.01, but not tied
    assert not mod.match_result(["n", "v"], near[::-1], ["n", "v"], near, ordered=True, order_cols=[1])


def test_unordered_ignores_order_cols():
    gold = [["A", 2.0], ["B", 1.0]]
    assert mod.match_result(["n", "v"], gold[::-1], ["n", "v"], gold, ordered=False, order_cols=[1])
