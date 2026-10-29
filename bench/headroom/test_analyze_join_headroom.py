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
Q12_ORDERS = {
    "x>y": ([100, 100000], [100, 90], 5.0),
    "y>x": ([1000, 100000], [1000, 90], 4.0),
}


def order_row(query: str, order: str, *, ms: float | None = None, classical: str = "",
              learned: str = "", ok: int = 1) -> str:
    table = Q3_ORDERS if query == "Q3" else Q12_ORDERS
    est, act, default_ms = table[order]
    ms = default_ms if ms is None else ms
    tables = order.split(">")

    def join(values) -> str:
        return ";".join(str(v) for v in values)

    return ",".join([
        "0.01", query, order, " ".join(str(i) for i in range(len(tables))), "1",
        str(int(order == classical)), str(int(order == learned)), str(sum(est)), join(est), join(act),
        join(TABLE_ROWS[t] for t in tables), join(SCAN_ROWS[t] for t in tables),
        str(ms), str(ms), str(ms), "5", str(ok),
    ])


def write_orders(path: Path, rows: list[str]) -> Path:
    path.write_text(ORDERS_HEADER + "\n" + "\n".join(rows) + "\n")
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
    decision = next(line for line in report.splitlines() if line.startswith("| 0.01 |"))
    assert "wrong result for Q3" in decision
    assert "Yes" not in decision


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
        "median_scan_qerror", "max_scan_qerror",
    ]
