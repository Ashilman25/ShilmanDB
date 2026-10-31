"""Unit tests for bench/headroom/analyze_join_headroom.py on small handwritten CSVs."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

SPEC_PATH = Path(__file__).with_name("analyze_join_headroom.py")
spec = importlib.util.spec_from_file_location("analyze_join_headroom", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)

sys.modules["analyze_join_headroom"] = mod
spec.loader.exec_module(mod)

ORDERS_HEADER = (
    "sf,query,order,order_slots,connected,is_classical,is_learned,est_cost,est_step_rows,"
    "actual_step_rows,table_rows,scan_rows,median_ms,min_ms,max_ms,runs,result_ok"
)
OPERATORS_HEADER = "sf,query,order,op_index,depth,operator,rows,total_ms,self_ms"
CHOICES_HEADER = "sf,query,chooser,order,source,model_order"

# Full and filtered row counts for the tables a-b-c (a chain) and x-y.
TABLE_ROWS = {"a": 100, "b": 1000, "c": 10000, "x": 100, "y": 1000}
SCAN_ROWS = {"a": 50, "b": 1000, "c": 2000, "x": 100, "y": 1000}

# order -> (est_step_rows, actual_step_rows, median_ms)
Q3_ORDERS = {
    "a>b>c": ([100, 1000, 10000], [50, 400, 800], 10.0),
    "b>a>c": ([1000, 1000, 10000], [1000, 400, 800], 8.0),
    "b>c>a": ([1000, 10000, 10000], [1000, 2000, 800], 12.0),
    "c>b>a": ([10000, 10000, 10000], [2000, 2000, 800], 20.0),
}
# Its join estimate is a cross product: 100 x 1000 rows, against 1 real row.
Q12_ORDERS = {
    "x>y": ([100, 100000], [100, 1], 5.0),
    "y>x": ([1000, 100000], [1000, 1], 4.0),
}


# Any query can use either table set: the order picks it. qerror swaps the join estimates for
# the actual rows times qerror, so every joined set gets that q-error.
def order_row(query: str, order: str, *, ms: float | None = None, classical: str = "",
              learned: str = "", ok: int = 1, qerror: float | None = None,
              min_ms: float | None = None, max_ms: float | None = None,
              est_scan_rows: str | None = None) -> str:
    table = Q3_ORDERS if order in Q3_ORDERS else Q12_ORDERS
    est, act, default_ms = table[order]
    if qerror is not None:
        est = [est[0], *(a * qerror for a in act[1:])]
    ms = default_ms if ms is None else ms
    tables = order.split(">")

    def join(values) -> str:
        return ";".join(str(v) for v in values)

    return ",".join([
        "0.01", query, order, " ".join(str(i) for i in range(len(tables))), "1",
        str(int(order == classical)), str(int(order == learned)), str(sum(est)), join(est), join(act),
        join(TABLE_ROWS[t] for t in tables), join(SCAN_ROWS[t] for t in tables),
        str(ms), str(ms if min_ms is None else min_ms), str(ms if max_ms is None else max_ms), "5", str(ok),
        *([] if est_scan_rows is None else [est_scan_rows]),
    ])


def write_orders(path: Path, rows: list[str]) -> Path:
    # Rows with one field more than the header carry the newest column, est_scan_rows.
    header = ORDERS_HEADER + (",est_scan_rows" if any(r.count(",") == ORDERS_HEADER.count(",") + 1 for r in rows) else "")
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


def write_full_set(results_dir: Path, *, q3_orders=tuple(Q3_ORDERS), bad: str = "") -> None:
    rows = [order_row("Q3", o, classical="a>b>c", learned="a>b>c", ok=int(o != bad)) for o in q3_orders]
    rows += [order_row("Q12", o, classical="x>y", learned="x>y") for o in Q12_ORDERS]
    write_orders(results_dir / "orders_sf0.01.csv", rows)
    (results_dir / "choices_sf0.01.csv").write_text(
        CHOICES_HEADER + "\n"
        "0.01,Q3,classical,a>b>c,classical,\n"
        "0.01,Q3,learned,a>b>c,learned,a>b>c\n"
        "0.01,Q12,classical,x>y,classical,\n"
        "0.01,Q12,learned,x>y,learned_rejected,y>x\n"
    )
    (results_dir / "operators_sf0.01.csv").write_text(
        OPERATORS_HEADER + "\n"
        "0.01,Q3,a>b>c,0,0,Aggregate,1,10.0,0.5\n"
        "0.01,Q3,a>b>c,1,1,HashJoin +c,800,9.5,3.0\n"
        "0.01,Q3,a>b>c,2,2,HashJoin +b,400,2.5,1.0\n"
        "0.01,Q3,a>b>c,3,3,Filter,50,0.5,0.2\n"
        "0.01,Q3,a>b>c,4,4,SeqScan a,100,0.3,0.3\n"
        "0.01,Q3,a>b>c,5,3,SeqScan b,1000,1.0,1.0\n"
        "0.01,Q3,a>b>c,6,2,SeqScan c,10000,4.0,4.0\n"
    )


def q3_summary(results_dir: Path):
    orders, _, choices = mod.load_results(results_dir)
    summary = mod.summarize(orders, choices)
    return summary[summary["query"] == "Q3"].iloc[0]


def report_for(results_dir: Path, rows: list[str]) -> str:
    write_orders(results_dir / "orders_sf0.01.csv", rows)
    orders, operators, choices = mod.load_results(results_dir)
    return mod.build_report(mod.summarize(orders, choices), orders, operators, choices)


# The SF 0.01 row of the draft decision table: SF, within 10%, q-errors small, next move.
def decision_cells(report: str) -> list[str]:
    section = report.split("## Draft answers")[1]
    line = next(line for line in section.splitlines() if line.startswith("| 0.01 |"))
    return [cell.strip() for cell in line.strip().strip("|").split(" | ")]


# Q3 and Q5 have their classical pick fastest and their join estimates close to the truth.
# Q12 is the same, except its join estimate is a cross product (q-error 1e5).
def decision_for(results_dir: Path, *, q3_orders=tuple(Q3_ORDERS), q3_classical: str = "b>a>c",
                 q3_qerror: float = 1.5, q5_qerror: float = 1.5) -> list[str]:
    rows = [order_row("Q3", o, classical=q3_classical, learned=q3_classical, qerror=q3_qerror) for o in q3_orders]
    rows += [order_row("Q5", o, classical="y>x", learned="y>x", qerror=q5_qerror) for o in Q12_ORDERS]
    rows += [order_row("Q12", o, classical="y>x", learned="y>x") for o in Q12_ORDERS]
    return decision_cells(report_for(results_dir, rows))


@pytest.fixture
def three_query_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    # A full sweep here is Q3, Q5 and Q12 alone, so the other queries' absence decides nothing.
    monkeypatch.setattr(mod, "FULL_ORDER_COUNTS", {"Q3": 4, "Q5": 2, "Q12": 2})


def test_qerror_symmetric_and_clamped() -> None:
    assert mod.qerror(100, 10) == 10
    assert mod.qerror(10, 100) == 10
    assert mod.qerror(0, 5) == 5
    assert mod.qerror(0, 0) == 1


def test_spearman_perfect_and_short() -> None:
    assert mod.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert mod.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0
    assert mod.spearman([1, 2], [1, 2]) is None


def test_summary_ratios(tmp_path: Path) -> None:
    write_full_set(tmp_path)
    row = q3_summary(tmp_path)
    assert row["num_orders"] == 4
    assert row["best_order"] == "b>a>c"
    assert row["best_ms"] == pytest.approx(8.0, abs=1e-9)
    assert row["classical_order"] == "a>b>c"
    assert row["classical_over_best"] == pytest.approx(10.0 / 8.0, abs=1e-9)
    assert row["learned_over_best"] == pytest.approx(10.0 / 8.0, abs=1e-9)
    assert row["worst_over_best"] == pytest.approx(20.0 / 8.0, abs=1e-9)
    # est ranks 1,2,3,4 against ms ranks 2,1,3,4
    assert row["spearman_est_vs_ms"] == pytest.approx(0.8, abs=1e-9)


def test_join_qerror_dedupes_table_sets(tmp_path: Path) -> None:
    write_orders(tmp_path / "orders_sf0.01.csv", [order_row("Q3", "a>b>c"), order_row("Q3", "b>a>c")])
    orders, _, _ = mod.load_results(tmp_path)
    # {a,b}: est 1000 vs 400; {a,b,c}: est 10000 vs 800. Each set counts once.
    assert sorted(mod.join_qerrors(orders)) == pytest.approx([2.5, 12.5])
    assert sorted(mod.scan_qerrors(orders)) == pytest.approx([1.0, 2.0, 5.0])


def test_scan_qerror_uses_est_scan_rows(tmp_path: Path) -> None:
    write_orders(tmp_path / "orders_sf0.01.csv", [order_row("Q12", "x>y", est_scan_rows="100;500")])
    orders, _, _ = mod.load_results(tmp_path)
    orders = orders.assign(table_rows="1000;500", scan_rows="100;500")
    assert mod.scan_qerrors(orders) == pytest.approx([1.0, 1.0])
    # An older file has no estimate: the full row count stands in for it.
    for older in (orders.drop(columns="est_scan_rows"), orders.assign(est_scan_rows=float("nan"))):
        assert sorted(mod.scan_qerrors(older)) == pytest.approx([1.0, 10.0])


def test_cross_product_check_uses_est_scan_rows(tmp_path: Path) -> None:
    write_orders(tmp_path / "orders_sf0.01.csv", [order_row("Q12", "x>y", classical="x>y", est_scan_rows="100;500")])
    orders, _, _ = mod.load_results(tmp_path)
    # The estimate ends at 100 x 500 = 50000, the product of the filtered rows, not of the 1000 x 500 full rows.
    orders = orders.assign(est_step_rows="100;50000", table_rows="1000;500")
    row = pd.Series({"sf": "0.01", "query": "Q12", "classical_order": "x>y"})
    assert mod._cross_product_estimate(orders, row)
    assert not mod._cross_product_estimate(orders.assign(est_step_rows="100;2000"), row)
    assert mod._cross_product_estimate(orders.drop(columns="est_scan_rows").assign(est_step_rows="100;500000"), row)


def test_qerrors_reject_lists_that_dont_line_up(tmp_path: Path) -> None:
    write_orders(tmp_path / "orders_sf0.01.csv", [order_row("Q3", "a>b>c")])
    orders, _, _ = mod.load_results(tmp_path)
    with pytest.raises(ValueError):
        mod.join_qerrors(orders.assign(actual_step_rows="50;400"))
    with pytest.raises(ValueError):
        mod.scan_qerrors(orders.assign(scan_rows="50;1000"))


def test_duplicate_rows_newest_file_wins(tmp_path: Path) -> None:
    full = write_orders(tmp_path / "orders_sf0.01.csv", [order_row("Q3", "a>b>c", ms=10.0)])
    subset = write_orders(tmp_path / "orders_sf0.01_Q3.csv", [order_row("Q3", "a>b>c", ms=30.0)])

    os.utime(full, (1_000_000, 1_000_000))
    os.utime(subset, (2_000_000, 2_000_000))
    orders, _, _ = mod.load_results(tmp_path)
    assert len(orders) == 1
    assert orders["median_ms"].iloc[0] == 30.0

    # It's the time stamp that decides, not the file name.
    os.utime(full, (3_000_000, 3_000_000))
    orders, _, _ = mod.load_results(tmp_path)
    assert orders["median_ms"].iloc[0] == 10.0


def test_bad_result_rows_excluded_and_warned(tmp_path: Path) -> None:
    write_full_set(tmp_path, bad="b>a>c")
    orders, operators, choices = mod.load_results(tmp_path)
    summary = mod.summarize(orders, choices)
    row = summary[summary["query"] == "Q3"].iloc[0]
    assert row["best_order"] == "a>b>c"
    assert row["best_ms"] == 10.0

    report = mod.build_report(summary, orders, operators, choices)
    first_block = report.split("\n\n")[0]
    assert report.startswith("> **Warning")
    assert "b>a>c" in first_block


def test_bad_classical_pick_gets_no_ratio(tmp_path: Path) -> None:
    write_full_set(tmp_path)
    # Q3 alone, with its classical (and learned) pick wrong, so nothing else decides the verdict.
    write_orders(tmp_path / "orders_sf0.01.csv",
                 [order_row("Q3", o, classical="a>b>c", learned="a>b>c", ok=int(o != "a>b>c")) for o in Q3_ORDERS])
    assert mod.main(["--results-dir", str(tmp_path)]) == 0

    row = pd.read_csv(tmp_path / "summary.csv").iloc[0]
    assert row["classical_order"] == "a>b>c"
    assert row["classical_ms"] == 10.0
    assert pd.isna(row["classical_over_best"])
    assert pd.isna(row["learned_over_best"])

    report = (tmp_path / "report.md").read_text()
    q3_numbers = next(line for line in report.splitlines() if line.startswith("| Q3 | 3 |"))
    assert q3_numbers.count("wrong result") == 2
    near = decision_cells(report)[1]
    assert near.startswith("Unknown: wrong result for Q3")

    # Its operator times are still shown, with a note.
    time_section = report.split("## Where the time goes")[1].split("## Draft answers")[0]
    assert "`a>b>c` returned a wrong result" in time_section


def test_chart_labels_skip_wrong_results(tmp_path: Path) -> None:
    write_full_set(tmp_path, bad="a>b>c")  # Q3's classical and learned pick
    orders, _, choices = mod.load_results(tmp_path)
    summary = mod.summarize(orders, choices)
    labels = {}
    for query in ("Q3", "Q12"):
        fig, ax = mod.plt.subplots()
        mod._plot_query(ax, query, orders[orders["query"] == query], summary[summary["query"] == query])
        labels[query] = list(ax.texts)
        mod.plt.close(fig)

    assert labels["Q3"] == []
    assert [t.get_text() for t in labels["Q12"]] == ["1.25×"]
    # A filled box behind the label, so the dashed 1.10× line can't run through it.
    assert labels["Q12"][0].get_bbox_patch() is not None


def test_decision_ignores_cross_product_queries(tmp_path: Path, three_query_sweep: None) -> None:
    _, near, small, move = decision_for(tmp_path)
    assert near == "Yes"
    assert small == "Yes. Q12 left out: its join estimates are cross products."
    assert move == "TPC-H is too easy for join ML. Move to skewed data (JCC-H) before building join V2."

    _, near, small, move = decision_for(tmp_path, q5_qerror=5)
    assert near == "Yes"
    assert small == "No: Q5 5.0. Q12 left out: its join estimates are cross products."
    assert move == "Join V2 has room on TPC-H as-is."


def test_decision_unknown_when_every_query_is_a_cross_product(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "FULL_ORDER_COUNTS", {"Q12": 2, "Q14": 2})
    rows = [order_row(q, o, classical="y>x", learned="y>x") for q in ("Q12", "Q14") for o in Q12_ORDERS]
    _, near, small, move = decision_cells(report_for(tmp_path, rows))
    assert near == "Yes"
    assert small == "Unknown: no query left to judge. Q12, Q14 left out: their join estimates are cross products."
    assert move == "Not enough data to call yet."


def test_decision_unknown_when_a_query_is_partly_timed(tmp_path: Path, three_query_sweep: None) -> None:
    # Q3 has 2 of its 4 orders, and its classical pick is the faster of the two.
    _, near, small, move = decision_for(tmp_path, q3_orders=("b>a>c", "c>b>a"))
    assert near == "Unknown: too few orders for Q3 (2 of 4 ok)"
    assert small == "Unknown: too few orders for Q3 (2 of 4 ok). Q12 left out: its join estimates are cross products."
    assert move == "Not enough data to call yet."

    # Over the limit on the orders that were timed. That's a sure "No" for near-best: more orders can
    # only lower the best. Not for q-errors: a median over some of the table sets can move either way.
    _, near, small, move = decision_for(tmp_path, q3_orders=("b>a>c", "c>b>a"), q3_classical="c>b>a", q3_qerror=3)
    assert near == "No: Q3 2.50×"
    assert small == "Unknown: too few orders for Q3 (2 of 4 ok). Q12 left out: its join estimates are cross products."
    assert move == "Join V2 has room on TPC-H as-is."

    # A fully timed query over the limit is still a sure "No".
    _, _, small, _ = decision_for(tmp_path, q3_orders=("b>a>c", "c>b>a"), q3_qerror=3, q5_qerror=5)
    assert small == "No: Q5 5.0. Q12 left out: its join estimates are cross products."


def test_noisy_orders_warned(tmp_path: Path) -> None:
    write_orders(tmp_path / "orders_sf0.01.csv", [
        order_row("Q3", "a>b>c", min_ms=9.0, max_ms=12.0),           # 1.33x apart: noisy
        order_row("Q3", "b>a>c", min_ms=7.8, max_ms=8.58),           # 1.10x: fine
        order_row("Q3", "c>b>a", min_ms=10.0, max_ms=30.0, ok=0),    # wrong result: has its own warning
        order_row("Q12", "x>y", min_ms=4.0, max_ms=6.0),             # 1.50x: the noisiest
    ])
    assert mod.main(["--results-dir", str(tmp_path)]) == 0

    report = (tmp_path / "report.md").read_text()
    top = report[:report.index("# Join-order headroom")]
    assert "> **Warning:** 2 timed order(s) had runs more than 20% apart" in top
    assert "The machine may have been busy; rerun those queries (SF 0.01: Q3, Q12)." in top
    noisy = [line for line in top.splitlines() if line.startswith("> - SF 0.01") and "ms to" in line]
    assert noisy == [
        "> - SF 0.01 Q12 `x>y`: 1.50× (4.0 ms to 6.0 ms)",
        "> - SF 0.01 Q3 `a>b>c`: 1.33× (9.0 ms to 12.0 ms)",
    ]

    summary = pd.read_csv(tmp_path / "summary.csv").set_index("query")
    assert summary.loc["Q3", "max_run_spread"] == pytest.approx(12.0 / 9.0, rel=1e-5)
    assert summary.loc["Q12", "max_run_spread"] == pytest.approx(1.5, rel=1e-5)

    # Every order within 5%: no noise warning.
    write_orders(tmp_path / "orders_sf0.01.csv", [order_row("Q3", o, min_ms=10.0, max_ms=10.5) for o in Q3_ORDERS])
    assert mod.main(["--results-dir", str(tmp_path)]) == 0
    assert "may have been busy" not in (tmp_path / "report.md").read_text()


def test_partial_query_is_summarized(tmp_path: Path) -> None:
    write_full_set(tmp_path, q3_orders=("a>b>c", "c>b>a"))
    row = q3_summary(tmp_path)
    assert row["num_orders"] == 2
    assert row["best_order"] == "a>b>c"
    assert row["worst_over_best"] == pytest.approx(2.0, abs=1e-9)


def test_main_writes_outputs(tmp_path: Path) -> None:
    write_full_set(tmp_path)
    assert mod.main(["--results-dir", str(tmp_path)]) == 0
    for name in ("summary.csv", "report.md", "orders_sf0.01.png"):
        assert (tmp_path / name).is_file(), name
    header = (tmp_path / "summary.csv").read_text().splitlines()[0]
    assert header.split(",") == [
        "sf", "query", "num_orders", "best_order", "best_ms",
        "classical_order", "classical_ms", "classical_over_best",
        "learned_order", "learned_ms", "learned_over_best", "worst_ms", "worst_over_best",
        "spearman_est_vs_ms", "median_join_qerror", "max_join_qerror",
        "median_scan_qerror", "max_scan_qerror", "max_run_spread",
    ]
