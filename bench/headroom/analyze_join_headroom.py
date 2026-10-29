#!/usr/bin/env python3
#Turns join_headroom's raw CSVs into summary.csv, report.md and one chart per scale factor.

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.use("Agg")  # writes files only, never opens a window

# How many connected orders a full sweep times for each join query.
FULL_ORDER_COUNTS = {"Q3": 4, "Q5": 16, "Q10": 8, "Q12": 2, "Q14": 2, "Q19": 2}

# Draft thresholds for the join rows of the headroom decision rule.
NEAR_BEST = 1.10
SMALL_QERROR = 2.0

KINDS = ("orders", "operators", "choices")
COLUMNS = {
    "orders": [
        "sf", "query", "order", "order_slots", "connected", "is_classical", "is_learned",
        "est_cost", "est_step_rows", "actual_step_rows", "table_rows", "scan_rows",
        "median_ms", "min_ms", "max_ms", "runs", "result_ok"
    ],
    "operators": ["sf", "query", "order", "op_index", "depth", "operator", "rows", "total_ms", "self_ms"],
    "choices": ["sf", "query", "chooser", "order", "source", "model_order"],
}
KEYS = {
    "orders": ["sf", "query", "order"],
    "operators": ["sf", "query", "order", "op_index"],
    "choices": ["sf", "query", "chooser"],
}
SUMMARY_COLUMNS = [
    "sf", "query", "num_orders", "best_order", "best_ms",
    "classical_order", "classical_ms", "classical_over_best",
    "learned_order", "learned_ms", "learned_over_best", "worst_ms", "worst_over_best",
    "spearman_est_vs_ms", "median_join_qerror", "max_join_qerror",
    "median_scan_qerror", "max_scan_qerror",
]

# Chart colors: three categorical slots for the planners' picks, a quiet gray for the rest
COLORS = {
    "classical": "#2a78d6", 
    "learned": "#eb6834", 
    "both": "#1baf7a", 
    "other": "#c3c2b7"
}

ROLE_LABELS = {
    "classical": "classical pick", 
    "learned": "learned pick",
    "both": "classical = learned", 
    "other": "other order"
}
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


#  Small helpers

#'1500;15000;60175' -> [1500.0, 15000.0, 60175.0]. A blank cell gives []
def parse_list(s: str) -> list[float]:
    if pd.isna(s) or not str(s).strip():
        return []
    return [float(v) for v in str(s).split(";")]


def qerror(est: float, act: float) -> float:
    est, act = max(est, 1.0), max(act, 1.0)
    return max(est / act, act / est)

#Rank correlation: Pearson on the two rank vectors. None below 4 points or when a side never varies
def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 4:
        return None
    rx = pd.Series(list(x), dtype=float).rank().to_numpy()
    ry = pd.Series(list(y), dtype=float).rank().to_numpy()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    spread = math.sqrt((rx * rx).sum() * (ry * ry).sum())
    return None if spread == 0 else float((rx * ry).sum() / spread)


def _sf_order(sfs) -> list[str]:
    return sorted(set(sfs), key=float)


def _query_number(tag: str) -> int:
    return int(tag[1:]) if tag[1:].isdigit() else 999


def _stat(values: list[float], fn) -> float:
    return float(fn(values)) if values else math.nan


#  Loading

def _read_csv(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, dtype={"sf": str, "order": str, "model_order": str})
    except pd.errors.EmptyDataError:  # killed before the header was written
        return pd.DataFrame()
    return frame.assign(file=path.name)


def _replaced_notes(dropped: pd.DataFrame, kept: pd.DataFrame, keys: list[str]) -> list[str]:
    if dropped.empty:
        return []
    pairs = dropped.merge(kept[[*keys, "file"]], on=keys, suffixes=("_old", "_new"))
    counts = pairs.groupby(["file_new", "file_old"]).size()
    return [f"`{new}` replaced {n} row(s) of `{old}`" for (new, old), n in counts.items()]


def _load_kind(results_dir: Path, kind: str) -> pd.DataFrame:
    # Oldest first, so keep="last" keeps the newest copy. The name breaks ties.
    files = sorted(results_dir.glob(f"{kind}_sf*.csv"), key=lambda p: (p.stat().st_mtime, p.name))
    frames = [f for f in map(_read_csv, files) if not f.empty]
    if frames:
        rows = pd.concat(frames, ignore_index=True)
        older = rows.duplicated(KEYS[kind], keep="last")
        kept = rows[~older].reset_index(drop=True)
        replaced = _replaced_notes(rows[older], kept, KEYS[kind])
    else:
        kept, replaced = pd.DataFrame(columns=[*COLUMNS[kind], "file"]), []
        
    kept.attrs.update(files=[f.name for f in files], replaced=replaced)
    return kept


def load_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    orders, operators, choices = (_load_kind(Path(results_dir), kind) for kind in KINDS)
    return orders, operators, choices


#  Summary

#Connected orders that returned the right answer: the set best, worst, Spearman and q-errors use.
def _good_rows(orders_q: pd.DataFrame) -> pd.DataFrame:
    return orders_q[(orders_q["connected"] == 1) & (orders_q["result_ok"] == 1)]


#One q-error per distinct set of joined tables. Every order that builds a set agrees on it.
def join_qerrors(orders_q: pd.DataFrame) -> list[float]:
    per_set: dict[frozenset[str], float] = {}
    for row in orders_q[orders_q["connected"] == 1].itertuples():
        tables = row.order.split(">")
        steps = zip(parse_list(row.est_step_rows), parse_list(row.actual_step_rows))
        
        for size, (est, act) in enumerate(steps, start=1):
            if size >= 2:
                per_set.setdefault(frozenset(tables[:size]), qerror(est, act))
                
    return list(per_set.values())

#One q-error per table: its full row count (what the formula uses) vs the rows its filters keep
def scan_qerrors(orders_q: pd.DataFrame) -> list[float]:
    per_table: dict[str, float] = {}
    for row in orders_q.itertuples():
        counts = zip(row.order.split(">"), parse_list(row.table_rows), parse_list(row.scan_rows))
        for table, full, scanned in counts:
            per_table.setdefault(table, qerror(full, scanned))
    return list(per_table.values())


def _choice(choices: pd.DataFrame, sf: str, query: str, chooser: str) -> pd.Series | None:
    hit = choices[(choices["sf"] == sf) & (choices["query"] == query) & (choices["chooser"] == chooser)]
    return None if hit.empty else hit.iloc[0]


#A planner's order, its median ms and whether it returned the right answer. The choices file
# wins over the is_* flags, because a later subset run without a model writes is_learned=0 on the
# orders it re-times
def _picked(orders_q: pd.DataFrame, choices: pd.DataFrame, chooser: str) -> tuple[str, float, bool]:
    choice = _choice(choices, orders_q["sf"].iloc[0], orders_q["query"].iloc[0], chooser)
    if choice is not None:
        order = choice["order"]
    else:
        flagged = orders_q[orders_q[f"is_{chooser}"] == 1]
        if flagged.empty:
            return "", math.nan, False
        order = flagged["order"].iloc[0]
        
    timed = orders_q[orders_q["order"] == order]
    if timed.empty:
        return order, math.nan, False
    return order, float(timed["median_ms"].iloc[0]), bool(timed["result_ok"].iloc[0] == 1)


def _summary_row(orders_q: pd.DataFrame, choices: pd.DataFrame) -> dict:
    good = _good_rows(orders_q)
    best = good.loc[good["median_ms"].idxmin()] if not good.empty else None
    best_ms = float(best["median_ms"]) if best is not None else math.nan
    worst_ms = float(good["median_ms"].max()) if not good.empty else math.nan
    classical_order, classical_ms, classical_ok = _picked(orders_q, choices, "classical")
    learned_order, learned_ms, learned_ok = _picked(orders_q, choices, "learned")
    rho = spearman(good["est_cost"], good["median_ms"])
    joins, scans = join_qerrors(good), scan_qerrors(good)
    
    # A pick that returned a wrong answer gets no ratio: its time isn't comparable.
    return {
        "sf": orders_q["sf"].iloc[0], 
        "query": orders_q["query"].iloc[0], 
        "num_orders": len(good),
        "best_order": best["order"] if best is not None else "", 
        "best_ms": best_ms,
        "classical_order": classical_order, 
        "classical_ms": classical_ms,
        "classical_over_best": classical_ms / best_ms if classical_ok else math.nan,
        "learned_order": learned_order, 
        "learned_ms": learned_ms,
        "learned_over_best": learned_ms / best_ms if learned_ok else math.nan,
        "worst_ms": worst_ms,
        "worst_over_best": worst_ms / best_ms,
        "spearman_est_vs_ms": math.nan if rho is None else rho,
        "median_join_qerror": _stat(joins, np.median), 
        "max_join_qerror": _stat(joins, max),
        "median_scan_qerror": _stat(scans, np.median), 
        "max_scan_qerror": _stat(scans, max),
    }


def summarize(orders: pd.DataFrame, choices: pd.DataFrame) -> pd.DataFrame:
    rows = [_summary_row(q, choices) for _, q in orders.groupby(["sf", "query"], sort=False)]
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    
    by_number = {
        "sf": float,
        "query": _query_number
    }
    summary = summary.sort_values(["sf", "query"], key=lambda col: col.map(by_number[col.name]))
    return summary.reset_index(drop=True)


#  Report

def _ms(v: float) -> str:
    return "-" if pd.isna(v) else f"{v:,.1f} ms"


def _times(v: float) -> str:
    return "-" if pd.isna(v) else f"{v:.2f}×"


def _num(v: float) -> str:
    if pd.isna(v):
        return "-"
    return f"{v:,.1f}" if abs(v) < 1e4 else f"{v:.1e}"


def _code(order: str) -> str:
    return f"`{order}`" if isinstance(order, str) and order else "-"


def _table(header: list[str], align: str, rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


#SFs named by the orders files, so a run killed before its first row still shows up
def _file_sfs(orders: pd.DataFrame) -> set[str]:
    return {name.removeprefix("orders_sf").removesuffix(".csv").split("_")[0] for name in orders.attrs.get("files", [])}


def _coverage_gaps(orders: pd.DataFrame) -> list[str]:
    gaps = []
    for sf in _sf_order(set(orders["sf"]) | _file_sfs(orders)):
        connected = orders[(orders["sf"] == sf) & (orders["connected"] == 1)]
        counts = connected.groupby("query").size()
        missing = [q for q in FULL_ORDER_COUNTS if q not in counts.index]
        if missing:
            gaps.append(f"SF {sf}: no rows for {', '.join(missing)}")
            
        gaps += [f"SF {sf} {q}: {counts[q]} of {full} connected orders timed (run cut short, or an order failed)" for q, full in FULL_ORDER_COUNTS.items() if q in counts.index and counts[q] < full]
    return gaps


def _warnings(orders: pd.DataFrame) -> str:
    blocks = []
    bad = orders[orders["result_ok"] != 1]
    if not bad.empty:
        lines = [f"> **Warning:** {len(bad)} order(s) returned a different answer from the default plan. "
                 "They are left out of best, worst, Spearman and the q-errors, and a classical or learned pick "
                 "among them gets no ratio (shown as \"wrong result\"):"]
        lines += [f"> - SF {r.sf} {r.query} `{r.order}`: {_ms(r.median_ms)} (`{r.file}`)" for r in bad.itertuples()]
        blocks.append("\n".join(lines))
        
    gaps = _coverage_gaps(orders)
    if gaps:
        lines = ["> **Warning:** not every query was fully timed, so a best order may be missing:"]
        blocks.append("\n".join(lines + [f"> - {gap}" for gap in gaps]))
    return "\n>\n".join(blocks)


def _input_note(*frames: pd.DataFrame) -> str:
    files = [name for f in frames for name in f.attrs.get("files", [])]
    if not files:
        return ""
    replaced = [note for f in frames for note in f.attrs.get("replaced", [])]
    text = "Input: " + ", ".join(f"`{name}`" for name in files) + "."
    if replaced:
        text += " Rows found in more than one file come from the newest one: " + "; ".join(replaced) + "."
    return text


HOW_TO_READ = """How to read the tables:
- Ratios are an order's median latency ÷ the fastest order's (1.00x is the fastest).
- Best, worst, **Orders**, Spearman and the q-errors use only connected orders that returned the right answer.
- A planner pick that isn't a connected order (it can't see `WHERE` joins) runs unforced. It counts only for the classical and learned ratios, so a ratio under 1.00× means it beat every connected order.
- Spearman compares the cost formula's ranking of the orders with their measured ranking (1 means the same ranking). It's blank below 4 orders.
- Join q-error is max(estimate ÷ actual, actual ÷ estimate) for the row count of each distinct set of joined tables. Scan q-error compares each table's full row count (what the cost formula uses) with the rows its filters keep.
- Learned source: `learned` means the planner used the model's order; `learned_rejected` means it didn't and fell back to the classical order."""


#True when a planner's pick was timed but returned a wrong answer
def _wrong_pick(orders: pd.DataFrame, row: pd.Series, chooser: str) -> bool:
    hit = orders[(orders["sf"] == row["sf"]) & (orders["query"] == row["query"]) & (orders["order"] == row[f"{chooser}_order"])]
    return bool((hit["result_ok"] != 1).any())


def _pick_ratio(orders: pd.DataFrame, row: pd.Series, chooser: str) -> str:
    return "wrong result" if _wrong_pick(orders, row, chooser) else _times(row[f"{chooser}_over_best"])


def _learned_label(row: pd.Series, choices: pd.DataFrame) -> str:
    if not row["learned_order"]:
        return "-"
    choice = _choice(choices, row["sf"], row["query"], "learned")
    if choice is None:
        return _code(row["learned_order"])
    same = ", same as classical" if row["learned_order"] == row["classical_order"] else ""
    return f"{_code(row['learned_order'])} ({choice['source']}{same})"


def _sf_section(rows: pd.DataFrame, orders: pd.DataFrame, choices: pd.DataFrame) -> str:
    numbers = _table(
        ["Query", "Orders", "Best", "Classical ÷ best", "Learned ÷ best", "Worst ÷ best", "Spearman", "Join q-error (median / max)", "Scan q-error (median / max)"],
        ["---"] + ["---:"] * 8,
        [[
            r["query"], str(r["num_orders"]), _ms(r["best_ms"]), _pick_ratio(orders, r, "classical"),
            _pick_ratio(orders, r, "learned"), _times(r["worst_over_best"]),
            "-" if pd.isna(r["spearman_est_vs_ms"]) else f"{r['spearman_est_vs_ms']:.2f}",
            f"{_num(r['median_join_qerror'])} / {_num(r['max_join_qerror'])}",
            f"{_num(r['median_scan_qerror'])} / {_num(r['max_scan_qerror'])}"] for _, r in rows.iterrows()
        ],
    )
    orders = _table(
        ["Query", "Best order", "Classical order", "Learned order (source)"],
        ["---"] * 4,
        [[r["query"], _code(r["best_order"]), _code(r["classical_order"]), _learned_label(r, choices)] for _, r in rows.iterrows()],
    )
    return f"## SF {rows['sf'].iloc[0]}\n\n{numbers}\n\n{orders}"


def _q5_section(summary: pd.DataFrame, orders: pd.DataFrame, choices: pd.DataFrame) -> str:
    q5 = summary[summary["query"] == "Q5"]
    if q5.empty:
        return "## Q5: classical vs learned\n\nNo Q5 rows yet."
    
    rows, notes = [], []
    for _, r in q5.iterrows():
        for chooser in ("classical", "learned"):
            choice = _choice(choices, r["sf"], "Q5", chooser)
            if not r[f"{chooser}_order"]:
                continue
            
            source = choice["source"] if choice is not None else "–"
            model_order = choice["model_order"] if choice is not None else ""
            rows.append([r["sf"], chooser, _code(r[f"{chooser}_order"]), source, _code(model_order), _ms(r[f"{chooser}_ms"]), _pick_ratio(orders, r, chooser)])
            
        wrong = [c for c in ("classical", "learned") if _wrong_pick(orders, r, c)]
        if not r["learned_order"]:
            notes.append(f"- SF {r['sf']}: no learned choice recorded (run `join_headroom` with `--join-model-path`).")
        elif wrong:
            notes.append(f"- SF {r['sf']}: no learned ÷ classical ratio, because the {' and '.join(wrong)} order returned a wrong result.")
        else:
            notes.append(f"- SF {r['sf']}: learned ÷ classical = {_times(r['learned_ms'] / r['classical_ms'])} (under 1.00x means the learned order is faster).")
    table = _table(["SF", "Planner", "Order", "Source", "Model's own order", "Median", "÷ best"], ["---"] * 5 + ["---:"] * 2, rows)
    return "## Q5: classical vs learned\n\n" + table + "\n\n" + "\n".join(notes)


#Names a Filter after the operator it sits on, since a plan has several of them
def _operator_labels(ops: pd.DataFrame) -> list[str]:
    names, depths = ops["operator"].tolist(), ops["depth"].tolist()
    labels = []
    for i, name in enumerate(names):
        has_child = i + 1 < len(names) and depths[i + 1] == depths[i] + 1
        labels.append(f"{name} over {names[i + 1]}" if name == "Filter" and has_child else name)
    return labels


def _hot_operators(query: str, summary: pd.DataFrame, operators: pd.DataFrame) -> str:
    candidates = summary[summary["query"] == query]
    for _, r in candidates.iloc[::-1].iterrows():  # largest SF first
        ops = operators[(operators["sf"] == r["sf"]) & (operators["query"] == query) & (operators["order"] == r["classical_order"])].sort_values("op_index")
        if ops.empty:
            continue
        
        root_ms = float(ops["total_ms"].iloc[0])
        top = ops.assign(label=_operator_labels(ops)).nlargest(5, "self_ms")
        table = _table(
            ["Operator", "Rows", "Self", "Share of root"], 
            ["---", "---:", "---:", "---:"],
            [[t["label"], f"{int(t['rows']):,}", _ms(t["self_ms"]), f"{t['self_ms'] / root_ms:.0%}"] for _, t in top.iterrows()]
        )
        return f"### {query} at SF {r['sf']}: {_code(r['classical_order'])} (root {_ms(root_ms)})\n\n{table}"
    return f"### {query}\n\nNo analyze rows for its classical order yet."


def _time_section(summary: pd.DataFrame, operators: pd.DataFrame) -> str:
    intro = ("## Where the time goes\n\nThe top 5 operators by self time in the classical order, at the largest SF "
             "with analyze rows for it. Self times are approximate: the profiler's own clock reads land inside "
             "every parent operator's time.")
    return "\n\n".join([intro, *(_hot_operators(q, summary, operators) for q in ("Q3", "Q5"))])


#'No' if any query is over the limit, 'Unknown' if any value or query is missing, else 'Yes'
def _verdict(rows: pd.DataFrame, column: str, limit: float, fmt, wrong: Sequence[str] = ()) -> tuple[str, bool | None]:
    over = rows[rows[column] > limit]
    if not over.empty:
        return "No: " + ", ".join(f"{r['query']} {fmt(r[column])}" for _, r in over.iterrows()), False
    
    no_value = [q for q in rows.loc[rows[column].isna(), "query"] if q not in wrong]
    no_rows = [q for q in FULL_ORDER_COUNTS if q not in set(rows["query"])]
    gaps = [f"{what} {', '.join(qs)}" for what, qs in (("wrong result for", wrong), ("no value for", no_value), ("no rows for", no_rows)) if qs]
    if gaps:
        return "Unknown: " + "; ".join(gaps), None
    return "Yes", True


def _decision_section(summary: pd.DataFrame, orders: pd.DataFrame) -> str:
    rows = []
    for sf in _sf_order(summary["sf"]):
        at_sf = summary[summary["sf"] == sf]
        wrong = [r["query"] for _, r in at_sf.iterrows() if _wrong_pick(orders, r, "classical")]
        near_text, near = _verdict(at_sf, "classical_over_best", NEAR_BEST, _times, wrong)
        small_text, small = _verdict(at_sf, "median_join_qerror", SMALL_QERROR, _num)
        
        if near is False or small is False:
            move = "Join V2 has room on TPC-H as-is."
        elif near and small:
            move = "TPC-H is too easy for join ML. Move to skewed data (JCC-H) before building join V2."
        else:
            move = "Not enough data to call yet."
        rows.append([sf, near_text, small_text, move])
        
    intro = ("## Draft answers to the join rows of the decision rule (`docs/ToDo/03`)\n\n"
             "These are drafts for a person to check. \"Within 10%\" means every query's classical ÷ best "
             f"is at most {NEAR_BEST:.2f}. \"Small q-errors\" means every query's median join q-error is at most "
             f"{SMALL_QERROR:g}.")
    table = _table(["SF", "Planner within 10% of best?", "Q-errors small?", "Draft next move"], ["---"] * 4, rows)
    return f"{intro}\n\n{table}"


def build_report(summary: pd.DataFrame, orders: pd.DataFrame, operators: pd.DataFrame, choices: pd.DataFrame) -> str:
    sections = [
        _warnings(orders),
        "# Join-order headroom",
        _input_note(orders, operators, choices),
        HOW_TO_READ,
        *(_sf_section(summary[summary["sf"] == sf], orders, choices) for sf in _sf_order(summary["sf"])),
        _q5_section(summary, orders, choices),
        _time_section(summary, operators),
        _decision_section(summary, orders),
    ]
    return "\n\n".join(s for s in sections if s) + "\n"


#  Chart

def _role(order: str, info: pd.Series) -> str:
    classical, learned = order == info["classical_order"], order == info["learned_order"]
    if classical and learned:
        return "both"
    return "classical" if classical else "learned" if learned else "other"


#One panel: a bar per timed order, fastest first. Returns the roles it drew
def _plot_query(ax, query: str, orders_q: pd.DataFrame, summary_q: pd.DataFrame) -> set[str]:
    ax.set_facecolor(SURFACE)
    if orders_q.empty or summary_q.empty or pd.isna(summary_q["best_ms"].iloc[0]):
        ax.set_title(query, color=INK)
        ax.text(0.5, 0.5, "no data", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        ax.set_axis_off()
        return set()

    info = summary_q.iloc[0]
    bars = orders_q.assign(ratio=orders_q["median_ms"] / info["best_ms"]).sort_values("ratio")
    roles = [_role(order, info) for order in bars["order"]]
    drawn = ax.bar(range(len(bars)), bars["ratio"], width=0.8, color=[COLORS[r] for r in roles], edgecolor=SURFACE, linewidth=1)
    for patch, ok in zip(drawn, bars["result_ok"]):
        if ok != 1:
            patch.set_hatch("///")
    for x, (ratio, role) in enumerate(zip(bars["ratio"], roles)):
        if role != "other":
            ax.text(x, ratio, f"{ratio:.2f}×", ha="center", va="bottom", fontsize=8, color=INK)

    ax.axhline(NEAR_BEST, color=MUTED, linestyle="--", linewidth=1)
    ax.set_title(f"{query} ({len(bars)} orders)", color=INK)
    ax.set_xticks([])
    ax.set_xlabel("orders, fastest first", color=MUTED)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED)
    return set(roles) | ({"wrong"} if (bars["result_ok"] != 1).any() else set())


def plot_orders(sf: str, orders: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), facecolor=SURFACE)
    drawn: set[str] = set()
    for ax, query in zip(axes.flat, FULL_ORDER_COUNTS):
        drawn |= _plot_query(ax, query, orders[orders["query"] == query], summary[summary["query"] == query])
    for ax in axes[:, 0]:
        ax.set_ylabel("median ÷ best", color=MUTED)

    handles = [Patch(color=COLORS[r], label=ROLE_LABELS[r]) for r in COLORS if r in drawn]
    if "wrong" in drawn:
        handles.append(Patch(facecolor=COLORS["other"], edgecolor=SURFACE, hatch="///", label="wrong result"))
    handles.append(Line2D([], [], color=MUTED, linestyle="--", label=f"{NEAR_BEST:.2f}× best"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.suptitle(f"Join orders at SF {sf}: median latency ÷ fastest order (lower is better)", color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


#  CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize join_headroom CSVs into a report and charts.")
    parser.add_argument("--results-dir", type=Path, required=True, help="directory with orders_sf*.csv, operators_sf*.csv and choices_sf*.csv")
    args = parser.parse_args(argv)

    results_dir: Path = args.results_dir
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        return 1
    orders, operators, choices = load_results(results_dir)
    if orders.empty:
        print(f"Error: no orders_sf*.csv rows in {results_dir}", file=sys.stderr)
        return 1

    summary = summarize(orders, choices)
    written = [results_dir / "summary.csv", results_dir / "report.md"]
    summary.to_csv(written[0], index=False, float_format="%.6g")
    written[1].write_text(build_report(summary, orders, operators, choices))
    for sf in _sf_order(summary["sf"]):
        path = results_dir / f"orders_sf{sf}.png"
        plot_orders(sf, orders[orders["sf"] == sf], summary[summary["sf"] == sf], path)
        written.append(path)

    print("Wrote:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
