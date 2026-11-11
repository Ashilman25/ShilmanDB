#!/usr/bin/env python3
"""Generate benchmark visualization charts from latencies.csv.

Produces three charts:
  1. query_latency.png  -- Grouped bar chart of median query latency per config
  2. join_order_accuracy.png -- Table of learned vs heuristic join ordering results
  3. hit_rate.png        -- Buffer pool eviction comparison (learned vs LRU)

Usage:
  .venv/bin/python bench/tpch/plot_results.py
  .venv/bin/python bench/tpch/plot_results.py --results-dir bench/results/
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


QUERIES = ["Q1", "Q3", "Q5", "Q6"]
QUERY_LABELS = {"Q1": "Q1 (agg)", "Q3": "Q3 (3-join)", "Q5": "Q5 (5-join)", "Q6": "Q6 (scan)"}
QUERY_JOIN_TABLES = {"Q3": 3, "Q5": 5}  # multi-join queries for accuracy table

# Display order and colors for configs.
# Blues for ShilmanDB, grays for SQLite.
CONFIG_ORDER: List[str] = [
    "lru_heuristic",
    "lru_learned_join",
    "learned_eviction_heuristic",
    "learned_all",
    "sqlite_disk",
    "sqlite_memory",
]

CONFIG_LABELS: Dict[str, str] = {
    "lru_heuristic": "LRU + Heuristic",
    "lru_learned_join": "LRU + Learned Join",
    "learned_eviction_heuristic": "Learned Evict + Heuristic",
    "learned_all": "Learned Evict + Learned Join",
    "sqlite_disk": "SQLite (disk)",
    "sqlite_memory": "SQLite (memory)",
}

CONFIG_COLORS: Dict[str, str] = {
    "lru_heuristic": "#1f4e79",
    "lru_learned_join": "#2e75b6",
    "learned_eviction_heuristic": "#5b9bd5",
    "learned_all": "#9dc3e6",
    "sqlite_disk": "#595959",
    "sqlite_memory": "#a6a6a6",
}


BELADY_HIT_RATE_IMPROVEMENT = 7.39  # % over LRU (ceiling)
LEARNED_EVICTION_IMPROVEMENT = 3.64  # % over LRU (from evaluation)


VALIDATION_EXACT_MATCH = 94.1  # %
VALIDATION_MEAN_REGRET = 0.019
VALIDATION_PREFIX2 = 96.3  # %




def load_latencies(csv_path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(csv_path):
        print(f"Error: latencies file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_hit_rates(csv_path: str) -> Optional[List[Dict[str, str]]]:
    if not os.path.isfile(csv_path):
        return None
    
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            
    return rows


def build_latency_map(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    result: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    
    for row in rows:
        try:
            sf = row["sf"]
            query = row["query"]
            config = row["config"]
            latency = float(row["median_latency_ms"])
        except (KeyError, ValueError, TypeError) as e:
            print(f"  WARNING: Skipping malformed latency row: {e}", file=sys.stderr)
            continue
        result[sf][query][config] = latency
    return dict(result)


# ---------------------------------------------------------------------------
# Chart 1: Query latency grouped bar chart
# ---------------------------------------------------------------------------

def plot_query_latency(latency_map: Dict[str, Dict[str, Dict[str, float]]], output_path: str) -> None:
    scale_factors = sorted(latency_map.keys(), key=float)
    n_sf = len(scale_factors)

    fig, axes = plt.subplots(
        1, n_sf, figsize=(6 * n_sf + 1, 5.5), squeeze=False, sharey=False
    )

    for sf_idx, sf in enumerate(scale_factors):
        ax = axes[0][sf_idx]
        sf_data = latency_map[sf]

        # Determine which configs are present for this SF
        present_configs = [c for c in CONFIG_ORDER if any(c in sf_data.get(q, {}) for q in QUERIES)]
        present_queries = [q for q in QUERIES if q in sf_data]

        if not present_configs or not present_queries:
            ax.text(0.5, 0.5, f"No data for SF={sf}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            ax.set_title(f"SF = {sf}", fontsize=12, fontweight="bold")
            continue

        n_queries = len(present_queries)
        n_configs = len(present_configs)
        bar_width = 0.8 / n_configs
        x_positions = list(range(n_queries))

        # Collect all latencies to decide on log scale
        all_latencies = [
            sf_data[q][c]
            for q in present_queries
            for c in present_configs
            if c in sf_data.get(q, {})
        ]
        use_log = (max(all_latencies) / max(min(all_latencies), 0.001)) > 50

        for cfg_idx, config in enumerate(present_configs):
            offsets = [x + cfg_idx * bar_width for x in x_positions]
            heights = [sf_data.get(q, {}).get(config, 0) for q in present_queries]
            color = CONFIG_COLORS.get(config, "#333333")
            ax.bar(
                offsets, heights, bar_width,
                label=CONFIG_LABELS.get(config, config),
                color=color, edgecolor="white", linewidth=0.5,
            )

        # Center x-ticks on the group
        center_offset = (n_configs - 1) * bar_width / 2
        ax.set_xticks([x + center_offset for x in x_positions])
        ax.set_xticklabels(
            [QUERY_LABELS.get(q, q) for q in present_queries],
            fontsize=9,
        )

        if use_log:
            ax.set_yscale("log")
            ax.set_ylabel("Median latency (ms, log scale)", fontsize=10)
        else:
            ax.set_ylabel("Median latency (ms)", fontsize=10)

        ax.set_title(f"SF = {sf}", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    # Single shared legend below all subplots
    present_all = [c for c in CONFIG_ORDER
                   if any(c in latency_map[sf].get(q, {})
                          for sf in scale_factors for q in QUERIES)]
    handles = [
        Patch(facecolor=CONFIG_COLORS.get(c, "#333"), label=CONFIG_LABELS.get(c, c))
        for c in present_all
    ]
    fig.legend(
        handles=handles, loc="lower center",
        ncol=min(len(handles), 3), fontsize=8.5,
        frameon=True, fancybox=True, shadow=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "ShilmanDB vs SQLite -- TPC-H Query Latency",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart 1 saved: {output_path}")


# ---------------------------------------------------------------------------
# Chart 2: Join ordering accuracy table
# ---------------------------------------------------------------------------

def plot_join_order_accuracy(latency_map: Dict[str, Dict[str, Dict[str, float]]], output_path: str) -> None:
    # Build table rows from available data
    table_rows: List[List[str]] = []

    for sf in sorted(latency_map.keys(), key=float):
        sf_data = latency_map[sf]
        for query in ["Q3", "Q5"]:
            if query not in sf_data:
                continue
            n_tables = QUERY_JOIN_TABLES.get(query, "?")
            heuristic = sf_data[query].get("lru_heuristic")
            learned = sf_data[query].get("lru_learned_join")
            if heuristic is None or learned is None:
                continue

            ratio = learned / heuristic if heuristic > 0 else float("inf")
            # A ratio <= 1.0 means learned was faster or equal
            faster = "Yes" if ratio <= 1.05 else "No"

            table_rows.append([
                f"SF={sf}",
                query,
                str(n_tables),
                f"{heuristic:.2f}",
                f"{learned:.2f}",
                f"{ratio:.3f}",
                faster,
            ])

    col_headers = [
        "Scale Factor", "Query", "Tables",
        "Heuristic (ms)", "Learned Join (ms)",
        "Ratio (L/H)", "Learned <= 1.05x?",
    ]

    # Layout: table on top, annotation text below
    fig, ax = plt.subplots(figsize=(10, 2.0 + 0.4 * len(table_rows)))
    ax.axis("off")

    if not table_rows:
        ax.text(
            0.5, 0.6,
            "No multi-join query data available.\n"
            "Run benchmarks with lru_heuristic and lru_learned_join configs.",
            ha="center", va="center", fontsize=11,
            transform=ax.transAxes,
        )
    else:
        table = ax.table(
            cellText=table_rows,
            colLabels=col_headers,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.4)

        # Style header row
        for col_idx in range(len(col_headers)):
            cell = table[0, col_idx]
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", fontweight="bold")

        # Color ratio cells: green if <= 1.05, light red otherwise
        for row_idx, row in enumerate(table_rows, start=1):
            ratio_val = float(row[5])
            ratio_cell = table[row_idx, 5]
            match_cell = table[row_idx, 6]
            if ratio_val <= 1.05:
                ratio_cell.set_facecolor("#d4edda")
                match_cell.set_facecolor("#d4edda")
            else:
                ratio_cell.set_facecolor("#f8d7da")
                match_cell.set_facecolor("#f8d7da")

    # Annotation block with Phase 13 validation results
    note_lines = [
        f"Validation set performance (Phase 13, 6561 training samples):",
        f"  Exact-match accuracy: {VALIDATION_EXACT_MATCH}%  |  "
        f"Mean cost regret: {VALIDATION_MEAN_REGRET}  |  "
        f"Prefix-2 accuracy: {VALIDATION_PREFIX2}%",
        f"Safety fallback: learned order used only if cost <= 1.5x exhaustive-search cost.",
    ]
    fig.text(
        0.5, 0.02, "\n".join(note_lines),
        ha="center", va="bottom", fontsize=8.5,
        fontstyle="italic", color="#444444",
    )

    fig.suptitle(
        "Learned Join Ordering -- Accuracy on TPC-H Queries",
        fontsize=13, fontweight="bold", y=0.97,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.93])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart 2 saved: {output_path}")


# ---------------------------------------------------------------------------
# Chart 3: Buffer pool hit rate / eviction comparison
# ---------------------------------------------------------------------------

def plot_hit_rate(latency_map: Dict[str, Dict[str, Dict[str, float]]], hit_rate_rows: Optional[List[Dict[str, str]]], output_path: str) -> None:

    # If a dedicated hit_rates.csv exists, plot from that directly
    if hit_rate_rows is not None:
        _plot_hit_rate_from_csv(hit_rate_rows, output_path)
        return

    # Otherwise, derive comparison from latency data
    _plot_hit_rate_from_latency(latency_map, output_path)


def _plot_hit_rate_from_csv(rows: List[Dict[str, str]], output_path: str) -> None:
    # Group by (sf, query)
    data: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    for row in rows:
        sf = row.get("sf", "")
        query = row.get("query", "")
        config = row.get("config", "")
        try:
            hit_rate = float(row.get("hit_rate", "0")) * 100  # to percent
        except ValueError:
            continue
        data[(sf, query)][config] = hit_rate

    keys = sorted(data.keys())
    if not keys:
        print("  WARNING: hit_rates.csv is empty. Skipping Chart 3.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    x_labels = [f"{sf} / {q}" for sf, q in keys]
    x = list(range(len(keys)))
    bar_w = 0.3

    lru_vals = [data[k].get("lru_heuristic", 0) for k in keys]
    learned_vals = [data[k].get("learned_eviction_heuristic", 0) for k in keys]

    ax.bar([xi - bar_w / 2 for xi in x], lru_vals, bar_w,
           label="LRU", color="#1f4e79", edgecolor="white")
    ax.bar([xi + bar_w / 2 for xi in x], learned_vals, bar_w,
           label="Learned Eviction", color="#5b9bd5", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Hit Rate (%)", fontsize=10)
    ax.set_title("Buffer Pool Hit Rate: LRU vs Learned Eviction", fontsize=13, fontweight="bold")

    # Belady ceiling: draw per-group markers (LRU value + 7.39pp for each group)
    for xi, lru_val in zip(x, lru_vals):
        ceiling = lru_val + BELADY_HIT_RATE_IMPROVEMENT
        ax.plot(
            [xi - bar_w * 0.6, xi + bar_w * 0.6], [ceiling, ceiling],
            color="#e67e22", linestyle="--", linewidth=1.5,
        )
    # Single legend entry for the ceiling markers
    ax.plot([], [], color="#e67e22", linestyle="--", linewidth=1.5,
            label=f"Belady ceiling (+{BELADY_HIT_RATE_IMPROVEMENT}pp over LRU)")

    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.text(
        0.5, -0.02,
        "Note: Learned eviction model trained on 4096-page vocabulary. "
        "At larger SFs where page IDs exceed 4096, model falls back to heuristic behavior.",
        ha="center", va="top", fontsize=8, fontstyle="italic", color="#666666",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart 3 saved: {output_path}")


def _plot_hit_rate_from_latency(latency_map: Dict[str, Dict[str, Dict[str, float]]], output_path: str) -> None:
    # Collect per-query improvement percentages across all SFs
    bar_labels: List[str] = []
    improvements: List[float] = []
    lru_latencies: List[float] = []
    learned_latencies: List[float] = []

    for sf in sorted(latency_map.keys(), key=float):
        sf_data = latency_map[sf]
        for query in QUERIES:
            if query not in sf_data:
                continue
            lru = sf_data[query].get("lru_heuristic")
            learned = sf_data[query].get("learned_eviction_heuristic")
            if lru is None or learned is None:
                continue

            pct_change = (lru - learned) / lru * 100  # positive = learned is faster
            bar_labels.append(f"{query}\nSF={sf}")
            improvements.append(pct_change)
            lru_latencies.append(lru)
            learned_latencies.append(learned)

    if not bar_labels:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(
            0.5, 0.5,
            "No paired LRU / Learned Eviction data available.\n"
            "Run benchmarks with both lru_heuristic and\n"
            "learned_eviction_heuristic configs.",
            ha="center", va="center", fontsize=11,
            transform=ax.transAxes,
        )
        ax.set_title("Buffer Pool: Learned Eviction Impact", fontsize=13, fontweight="bold")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Chart 3 saved: {output_path} (no data)")
        return

    fig, (ax_bars, ax_lat) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.3]}
    )

    # --- Left panel: latency improvement % ---
    x = list(range(len(bar_labels)))
    colors = ["#2e75b6" if v >= 0 else "#c00000" for v in improvements]
    bars = ax_bars.bar(x, improvements, color=colors, edgecolor="white", width=0.6)

    # Reference lines — these are hit-rate improvements, not latency reductions.
    # Shown as context; units differ from the Y-axis (hit rate vs latency).
    ax_bars.axhline(
        y=BELADY_HIT_RATE_IMPROVEMENT, color="#e67e22", linestyle="--",
        linewidth=1.5,
        label=f"Belady ceiling: +{BELADY_HIT_RATE_IMPROVEMENT}% hit rate (ref)",
    )
    ax_bars.axhline(
        y=LEARNED_EVICTION_IMPROVEMENT, color="#27ae60", linestyle=":",
        linewidth=1.5,
        label=f"Phase 12 eval: +{LEARNED_EVICTION_IMPROVEMENT}% hit rate (ref)",
    )
    ax_bars.axhline(y=0, color="black", linewidth=0.8)

    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(bar_labels, fontsize=8.5)
    ax_bars.set_ylabel("Latency reduction (%)\n(positive = learned is faster)", fontsize=9)
    ax_bars.set_title("Latency Improvement\n(Learned Eviction vs LRU)", fontsize=11, fontweight="bold")
    ax_bars.legend(fontsize=8, loc="upper right")
    ax_bars.grid(axis="y", alpha=0.3, linestyle="--")
    ax_bars.set_axisbelow(True)

    # Add value labels on bars
    for bar, val in zip(bars, improvements):
        y_pos = bar.get_height()
        va = "bottom" if y_pos >= 0 else "top"
        ax_bars.text(
            bar.get_x() + bar.get_width() / 2, y_pos,
            f"{val:+.1f}%", ha="center", va=va, fontsize=8, fontweight="bold",
        )

    # --- Right panel: absolute latencies side-by-side ---
    bar_w = 0.35
    ax_lat.bar(
        [xi - bar_w / 2 for xi in x], lru_latencies, bar_w,
        label="LRU + Heuristic", color="#1f4e79", edgecolor="white",
    )
    ax_lat.bar(
        [xi + bar_w / 2 for xi in x], learned_latencies, bar_w,
        label="Learned Evict + Heuristic", color="#5b9bd5", edgecolor="white",
    )

    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(bar_labels, fontsize=8.5)
    ax_lat.set_ylabel("Median latency (ms)", fontsize=9)
    ax_lat.set_title("Absolute Latency Comparison", fontsize=11, fontweight="bold")
    ax_lat.legend(fontsize=8, loc="upper left")
    ax_lat.grid(axis="y", alpha=0.3, linestyle="--")
    ax_lat.set_axisbelow(True)

    # Note about page ID limitation
    fig.text(
        0.5, -0.01,
        "Note: Learned eviction model trained on 4096-page vocabulary. "
        "At larger SFs where page IDs exceed 4096, model falls back to heuristic behavior.",
        ha="center", va="top", fontsize=8, fontstyle="italic", color="#666666",
    )

    fig.suptitle(
        "Buffer Pool Eviction: Learned vs LRU",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart 3 saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate benchmark visualization charts from latency results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  .venv/bin/python bench/tpch/plot_results.py\n"
            "  .venv/bin/python bench/tpch/plot_results.py --results-dir bench/results/\n"
        ),
    )
    parser.add_argument(
        "--results-dir",
        default="bench/results",
        help="Directory containing latencies.csv (default: bench/results/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir

    latencies_path = os.path.join(results_dir, "latencies.csv")
    hit_rates_path = os.path.join(results_dir, "hit_rates.csv")

    print("ShilmanDB Benchmark Chart Generation")
    print(f"  Results dir: {results_dir}")
    print(f"  Latencies:   {latencies_path}")

    # Load data
    rows = load_latencies(latencies_path)
    latency_map = build_latency_map(rows)

    scale_factors = sorted(latency_map.keys(), key=float)
    configs_found = sorted({row["config"] for row in rows})
    print(f"  Scale factors: {scale_factors}")
    print(f"  Configs:       {configs_found}")
    print(f"  Records:       {len(rows)}")
    print()

    # Optional hit rate data
    hit_rate_rows = load_hit_rates(hit_rates_path)
    if hit_rate_rows is not None:
        print(f"  Found hit_rates.csv ({len(hit_rate_rows)} records)")
    else:
        print("  No hit_rates.csv found; deriving eviction comparison from latencies.")
    print()

    os.makedirs(results_dir, exist_ok=True)

    # Chart 1: Query latency grouped bar chart
    plot_query_latency(
        latency_map,
        os.path.join(results_dir, "query_latency.png"),
    )

    # Chart 2: Join ordering accuracy table
    plot_join_order_accuracy(
        latency_map,
        os.path.join(results_dir, "join_order_accuracy.png"),
    )

    # Chart 3: Buffer pool hit rate comparison
    plot_hit_rate(
        latency_map,
        hit_rate_rows,
        os.path.join(results_dir, "hit_rate.png"),
    )

    print("\nDone. Charts saved to:", results_dir)


if __name__ == "__main__":
    main()
