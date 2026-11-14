#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# One palette, two selections. Slots are used in this order, never cycled.
THEMES: Dict[str, Dict[str, object]] = {
    "light": {
        "slots": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "spine": "#c3c2b7",
    },
    "dark": {
        "slots": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"],
        "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "spine": "#383835",
    },
}

FONTS = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
DPI = 200
MARKER_SIZE = 7.5

QUERIES = ["Q1", "Q3", "Q5", "Q6", "Q10", "Q12", "Q14", "Q19"]
ENGINE_SLOT = {"shilmandb": 0, "duckdb": 1, "postgres": 2, "sqlite": 3}
# Config order on the chart, its label, and its marker (engines keep one hue each).
CONFIGS = [
    ("baseline", "ShilmanDB baseline", "o"),
    ("+index", "ShilmanDB +index", "s"),
    ("+vectorized", "ShilmanDB +vectorized", "D"),
    ("+learned", "ShilmanDB +learned", "^"),
    ("sqlite_disk", "SQLite disk", "o"),
    ("sqlite_memory", "SQLite memory", "s"),
    ("duckdb", "DuckDB", "o"),
    ("duckdb_1thread", "DuckDB 1 thread", "s"),
    ("postgres", "Postgres", "o"),
    ("postgres_1worker", "Postgres 1 worker", "s"),
]

POLICY_LABEL = {"lru": "LRU", "arc_visit": "ARC", "lirs_visit": "LIRS", "v2": "V2", "belady": "Belady (best possible)"}
POLICY_SLOT = {"lru": 0, "arc_visit": 1, "lirs_visit": 2, "v2": 3}
KIND_TITLE = {"index_stream": "stream workload", "index_mixed": "mixed workload"}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def apply_rc() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONTS,
        "axes.linewidth": 1.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "svg.fonttype": "none",
    })


# A figure and axes in a theme's colors: value-axis grid only, top/right spines off.
def new_figure(theme: dict, nrows: int, ncols: int, width: float, height: float, value_axis: str = "y", **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False, **kw)
    fig.patch.set_facecolor(theme["surface"])
    for ax in axes.flat:
        style_axes(ax, theme, value_axis)
    return fig, axes


def style_axes(ax, theme: dict, value_axis: str) -> None:
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["spine"])

    ax.tick_params(colors=theme["muted"], labelsize=9, length=0, pad=4)
    ax.xaxis.label.set_color(theme["muted"])
    ax.yaxis.label.set_color(theme["muted"])
    ax.grid(True, axis=value_axis, color=theme["grid"], linewidth=0.8, linestyle="-")
    ax.grid(False, axis="x" if value_axis == "y" else "y")
    ax.set_axisbelow(True)


def add_titles(fig, theme: dict, title: str, subtitle: str) -> None:
    fig.text(0.01, 1.0, title, ha="left", va="bottom", fontsize=13, color=theme["ink"], fontweight="semibold")
    fig.text(0.01, 0.985, subtitle, ha="left", va="top", fontsize=10, color=theme["ink2"])


def style_legend(legend, theme: dict) -> None:
    for text in legend.get_texts():
        text.set_color(theme["ink2"])


def save(fig, out: Path, name: str, theme_name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}-{theme_name}.png"
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


# Calls draw(theme) for light and dark, saving each figure it returns.
def both_themes(out: Path, name: str, draw: Callable[[dict], object]) -> List[Path]:
    apply_rc()
    return [save(draw(THEMES[t]), out, name, t) for t in ("light", "dark")]


def baseline_panels(data: Path) -> List[tuple]:
    panels = []
    for sf in ("1", "10"):
        path = data / f"baseline_sf{sf}.csv"
        if path.exists():
            panels.append((sf, read_csv(path)))
    return panels


# The chip from the copied environment record, as a subtitle prefix ("" if none).
def machine_name(data: Path) -> str:
    for sf in ("1", "10"):
        path = data / f"environment_sf{sf}.csv"
        if path.exists():
            chip = {r["key"]: r["value"] for r in read_csv(path)}.get("chip")
            if chip:
                return f"{chip}, "
    return ""


# A row's time if it may be drawn: never for a missing row, a FAIL or an empty time.
def drawable_ms(row: Optional[Dict[str, str]]) -> Optional[float]:
    if not row or row.get("status") == "FAIL" or not row.get("median_ms"):
        return None
    return float(row["median_ms"])


def ms_label(v: float, _pos=None) -> str:
    if v >= 1000:
        return f"{v / 1000:g} s"
    return f"{v:g} ms"


# Chart 1: every engine config's median time per query, one log panel per scale factor.
def chart_latency(data: Path, out: Path) -> Optional[List[Path]]:
    panels = baseline_panels(data)
    if not panels:
        return None
    machine = machine_name(data)

    def draw(theme: dict):
        fig, axes = new_figure(theme, len(panels), 1, 10.5, 4.3 * len(panels) + 0.4)
        present = [c for c in CONFIGS if any(r["config"] == c[0] for _, rows in panels for r in rows)]
        width = 0.8 / max(len(present), 1)
        any_drift = False
        for ax, (sf, rows) in zip(axes[:, 0], panels):
            by_key = {(r["query"], r["config"]): r for r in rows}
            queries = [q for q in QUERIES if any(r["query"] == q for r in rows)]
            for ci, (config, _, marker) in enumerate(present):
                engine = next((r["engine"] for r in rows if r["config"] == config), "shilmandb")
                color = theme["slots"][ENGINE_SLOT.get(engine, 7)]
                xs, ys, drift_xs, drift_ys = [], [], [], []
                
                for qi, q in enumerate(queries):
                    r = by_key.get((q, config))
                    ms = drawable_ms(r)
                    if ms is None:
                        continue   # FAIL or not run: left empty
                    x = qi - 0.4 + width * (ci + 0.5)
                    xs.append(x)
                    ys.append(ms)
                    if r["status"] == "DRIFT":
                        drift_xs.append(x)
                        drift_ys.append(ms)
                        
                ax.plot(
                    xs, ys, linestyle="none", marker=marker, markersize=MARKER_SIZE,
                    markerfacecolor=color, markeredgecolor=theme["surface"],
                    markeredgewidth=2, zorder=3
                )
                
                if drift_xs:
                    any_drift = True
                    ax.plot(
                        drift_xs, drift_ys, linestyle="none", marker="o",
                        markersize=MARKER_SIZE + 6, markerfacecolor="none",
                        markeredgecolor=theme["ink"], markeredgewidth=1.2, zorder=4
                    )
                    
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(FuncFormatter(ms_label))
            ax.set_xticks(range(len(queries)))
            ax.set_xticklabels(queries, color=theme["ink2"], fontsize=10)
            ax.set_xlim(-0.5, len(queries) - 0.5)
            for qi in range(len(queries) - 1):
                ax.axvline(qi + 0.5, color=theme["grid"], linewidth=0.8, zorder=0)
                
            ax.set_ylabel("median time per query (log scale)")
            ax.text(0.0, 1.02, f"SF={sf}", transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5, color=theme["ink"], fontweight="semibold")

        handles = []
        for config, label, marker in present:
            engine = next(r["engine"] for _, rows in panels for r in rows if r["config"] == config)
            handles.append(
                Line2D([], [], linestyle="none", marker=marker, markersize=MARKER_SIZE,
                       markerfacecolor=theme["slots"][ENGINE_SLOT.get(engine, 7)],
                       markeredgecolor=theme["surface"], markeredgewidth=2,
                       label=label
                       )
                )
        if any_drift:
            handles.append(
                Line2D([], [], linestyle="none", marker="o", markersize=MARKER_SIZE + 4,
                       markerfacecolor="none", markeredgecolor=theme["ink"],
                       label="ring: answer drifts in the last digits"
                       )
                )
        legend = axes[0, 0].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, handletextpad=0.4, labelspacing=0.7)
        style_legend(legend, theme)
        fig.subplots_adjust(hspace=0.3)
        
        add_titles(
            fig, theme, "TPC-H query times: ShilmanDB against SQLite, DuckDB and Postgres",
            f"{machine}median of 5 timed runs after 1 warmup; lower is faster; a failed answer is left out"
        )
        return fig

    return both_themes(out, "latency", draw)


# (query, +vectorized ÷ +index) for every query where both times can be drawn.
def vectorized_ratios(rows: List[Dict[str, str]]) -> List[tuple]:
    by_key = {(r["query"], r["config"]): r for r in rows}
    ratios = []
    for q in QUERIES:
        vec = drawable_ms(by_key.get((q, "+vectorized")))
        idx = drawable_ms(by_key.get((q, "+index")))
        
        if vec is not None and idx is not None:
            ratios.append((q, vec / idx))
    return ratios


# Chart 2: +vectorized time ÷ +index time per query, one panel per scale factor.
def chart_vectorized(data: Path, out: Path) -> Optional[List[Path]]:
    panels = []
    for sf, rows in baseline_panels(data):
        ratios = vectorized_ratios(rows)
        if ratios:
            panels.append((sf, ratios))

    if not panels:
        return None

    def draw(theme: dict):
        fig, axes = new_figure(theme, 1, len(panels), 4.8 * len(panels) + 0.6, 4.2, sharey=True)
        top = max(1.15, max(r for _, ratios in panels for _, r in ratios) * 1.15)
        for ax, (sf, ratios) in zip(axes[0], panels):
            xs = range(len(ratios))
            values = [r for _, r in ratios]
            ax.bar(xs, values, width=0.45, color=theme["slots"][0], zorder=2)
            ax.axhline(1.0, color=theme["ink2"], linewidth=1.0, zorder=3)
            ax.text(-0.5, 1.0 + top * 0.01, "same speed as +index", ha="left", va="bottom", fontsize=8.5, color=theme["ink2"])
            lo, hi = values.index(min(values)), values.index(max(values))
            for i in {lo, hi}:
                ax.text(i, values[i] + top * 0.015, f"{values[i]:.2f}", ha="center", va="bottom", fontsize=9, color=theme["ink"])
            ax.set_xticks(list(xs))
            ax.set_xticklabels([q for q, _ in ratios], color=theme["ink2"], fontsize=10)
            ax.set_ylim(0, top)
            ax.text(0.0, 1.02, f"SF={sf}", transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5, color=theme["ink"], fontweight="semibold")
        axes[0, 0].set_ylabel("+vectorized time ÷ +index time")
        add_titles(fig, theme, "Vectorized execution against tuple-at-a-time, per query", "ShilmanDB, median times; below the line the vectorized mode is faster")
        return fig

    return both_themes(out, "vectorized", draw)


def pct(v: float, _pos=None) -> str:
    return f"{v:g}%"


def millions(v: float, _pos=None) -> str:
    return f"{v / 1e6:g}M" if v else "0"


# Chart 3: misses against pool size for LRU, ARC, LIRS, V2 and Belady, stream and mixed.
def chart_eviction(data: Path, out: Path) -> Optional[List[Path]]:
    path = data / "eviction_misses.csv"
    if not path.exists():
        return None
    
    rows = read_csv(path)
    kinds = [k for k in KIND_TITLE if any(r["kind"] == k for r in rows)]
    sf = rows[0]["sf"] if rows else ""

    def series(kind: str, policy: str):
        pts = sorted((float(r["pool_fraction"]) * 100, float(r["misses"])) for r in rows if r["kind"] == kind and r["policy"] == policy)
        return [p[0] for p in pts], [p[1] for p in pts]

    def draw(theme: dict):
        fig, axes = new_figure(theme, 1, len(kinds), 5.2 * len(kinds), 4.4, sharey=True)
        ymax = max(float(r["misses"]) for r in rows) * 1.12
        for ax, kind in zip(axes[0], kinds):
            labels = []   # (policy, xs, ys)
            bx, by = series(kind, "belady")
            if bx:
                ax.plot(bx, by, color=theme["ink2"], linewidth=2.0, zorder=2)
                labels.append(("belady", bx, by))
                
            for policy in ("lru", "arc_visit", "lirs_visit"):
                xs, ys = series(kind, policy)
                if not xs:
                    continue
                color = theme["slots"][POLICY_SLOT[policy]]
                ax.plot(xs, ys, color=color, linewidth=2.0, marker="o", markersize=MARKER_SIZE, markeredgecolor=theme["surface"], markeredgewidth=2, zorder=3)
                labels.append((policy, xs, ys))
            vx, vy = series(kind, "v2")
            if vx:
                ax.plot(
                    vx, vy, linestyle="none", marker="D", markersize=8,
                    markerfacecolor=theme["slots"][POLICY_SLOT["v2"]],
                    markeredgecolor=theme["surface"], markeredgewidth=2, zorder=5
                )
                ax.annotate("V2", (vx[0], vy[0]), xytext=(4, 9), textcoords="offset points", ha="left", va="bottom", fontsize=9, color=theme["ink"])
            direct_label_lines(ax, theme, labels)
            ax.set_xlim(-2, 102)
            ax.set_ylim(0, ymax)
            ax.xaxis.set_major_formatter(FuncFormatter(pct))
            ax.yaxis.set_major_formatter(FuncFormatter(millions))
            ax.set_xlabel("buffer pool size (% of the database's pages)")
            ax.text(0.0, 1.02, KIND_TITLE[kind], transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5, color=theme["ink"], fontweight="semibold")
        axes[0, 0].set_ylabel("misses (page reads from disk)")
        handles = [
            Line2D([], [], color=theme["slots"][POLICY_SLOT[p]], marker="o",
                   markersize=MARKER_SIZE, markeredgecolor=theme["surface"],
                   markeredgewidth=2, label=POLICY_LABEL[p]) for p in ("lru", "arc_visit", "lirs_visit")
            ]
        
        handles.append(Line2D([], [], linestyle="none", marker="D", markersize=MARKER_SIZE,
                              markerfacecolor=theme["slots"][POLICY_SLOT["v2"]],
                              markeredgecolor=theme["surface"], markeredgewidth=2,
                              label="V2 (learned)"))
        handles.append(Line2D([], [], color=theme["ink2"], linewidth=2.0, label=POLICY_LABEL["belady"]))
        legend = axes[0, -1].legend(handles=handles, loc="upper right", fontsize=9)
        style_legend(legend, theme)
        fig.subplots_adjust(wspace=0.14)
        add_titles(fig, theme, "Eviction: misses against buffer pool size",
                   f"TPC-H SF={sf}, +index, trace simulator, mean of 3 seeds; fewer is better. "
                   "V2 ran only at the default pool (2,048 pages), so it is one point")
        return fig

    return both_themes(out, "eviction", draw)


# Names each line at its 50% point; Belady below-left, LRU and ARC merged if they touch.
def direct_label_lines(ax, theme: dict, labels: list) -> None:
    def at(xs, ys, target):
        best = min(range(len(xs)), key=lambda i: abs(xs[i] - target))
        return xs[best], ys[best]

    if not labels:
        return
    target = 50.0
    points = {name: at(xs, ys, target) for name, xs, ys in labels}
    span = max(max(ys) for _, _, ys in labels) or 1.0
    text = {name: POLICY_LABEL[name] for name in points}
    if "lru" in points and "arc_visit" in points:
        if abs(points["lru"][1] - points["arc_visit"][1]) < 0.05 * span:
            text["lru"] = "LRU, ARC"
            del text["arc_visit"]
            
    for name, label in text.items():
        x, y = points[name]
        below = name == "belady"
        ax.annotate(label, (x, y), xytext=(-6, -6) if below else (4, 7),
                    textcoords="offset points", ha="right" if below else "left",
                    va="top" if below else "bottom", fontsize=9,
                    color=theme["ink"] if name != "belady" else theme["ink2"])


# Chart 4: each result over its best possible: join orders and eviction cells.
def chart_ceilings(data: Path, out: Path) -> Optional[List[Path]]:
    path = data / "ceilings.csv"
    if not path.exists():
        return None
    rows = read_csv(path)
    parts = [p for p in ("join", "eviction") if any(r["part"] == p for r in rows)]
    
    methods = {
        "join": [("classical", "classical"), ("learned", "learned")],
        "eviction": [("best_classical", "best classical"), ("v2", "V2 (learned)")]
    }
    titles = {
        "join": "Join order: chosen ÷ best order's time",
        "eviction": "Eviction: misses ÷ Belady's misses"
    }

    def draw(theme: dict):
        fig, axes = new_figure(theme, 1, len(parts), 5.0 * len(parts), 3.9, value_axis="x", sharex=True)
        xmax = max(float(r["ratio"]) for r in rows)
        for ax, part in zip(axes[0], parts):
            prows = [r for r in rows if r["part"] == part]
            items = list(dict.fromkeys(r["item"] for r in prows))
            ratio = {(r["item"], r["method"]): float(r["ratio"]) for r in prows}
            ax.axvline(1.0, color=theme["ink2"], linewidth=1.0, zorder=1)
            labelled = None   # skip the second value label when it repeats the first
            
            for mi, (method, label) in enumerate(methods[part]):
                offset = -0.17 if mi == 0 else 0.17
                ys = [i + offset for i, it in enumerate(items) if (it, method) in ratio]
                xs = [ratio[(it, method)] for it in items if (it, method) in ratio]
                ax.plot(
                    xs, ys, linestyle="none", marker="o", markersize=MARKER_SIZE,
                    markerfacecolor=theme["slots"][mi], markeredgecolor=theme["surface"],
                    markeredgewidth=2, zorder=3, label=label
                )
                
                if xs and not (labelled is not None and abs(max(xs) - labelled) < 0.005):
                    hi = xs.index(max(xs))
                    labelled = max(xs)
                    ax.annotate(f"{xs[hi]:.2f}", (xs[hi], ys[hi]), xytext=(7, 0),
                                textcoords="offset points", ha="left", va="center",
                                fontsize=9, color=theme["ink"])
                    
            ax.set_yticks(range(len(items)))
            ax.set_yticklabels(items, color=theme["ink2"], fontsize=10)
            ax.set_ylim(len(items) - 0.4, -0.6)
            ax.set_xlim(0.97, xmax + (xmax - 1) * 0.18 + 0.03)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}×"))
            sf = prows[0]["sf"]
            ax.text(0.0, 1.03, f"{titles[part]} (SF={sf})", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=10.5, color=theme["ink"],
                    fontweight="semibold")
            ax.text(1.0, -0.6, " best possible", ha="left", va="top", fontsize=8.5, color=theme["ink2"])
            legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9, handletextpad=0.3, columnspacing=1.5)
            style_legend(legend, theme)
        fig.subplots_adjust(wspace=0.35, top=0.84)
        add_titles(fig, theme, "Results next to their ceilings",
                   "1.0× is the best possible; join: every connected order timed, "
                   "eviction: the four +index decision cells")
        return fig

    return both_themes(out, "ceilings", draw)


# "at least 0.50" -> ("≥", 0.5); anything but "at least" / "at most" is an error.
def needs_text(needs: str) -> tuple:
    for prefix, sign in (("at least ", "≥"), ("at most ", "≤")):
        if needs.startswith(prefix):
            return sign, float(needs[len(prefix):])
    raise ValueError(f"needs must start with 'at least' or 'at most', got {needs!r}")


# Chart 5: the analyst check's three bars: rate, Wilson interval and threshold.
def chart_analyst(data: Path, out: Path) -> Optional[List[Path]]:
    path = data / "analyst_bars.csv"
    if not path.exists():
        return None
    rows = read_csv(path)

    def draw(theme: dict):
        fig, axes = new_figure(theme, 1, 1, 8.6, 3.6, value_axis="x")
        ax = axes[0, 0]
        color = theme["slots"][0]
        for i, r in enumerate(rows):
            rate, lo, hi = float(r["rate"]), float(r["wilson_low"]), float(r["wilson_high"])
            ax.barh(i, rate, height=0.36, color=color, zorder=2)
            ax.errorbar(rate, i, xerr=[[rate - lo], [hi - rate]], fmt="none", olor=theme["ink2"], elinewidth=1.5, capsize=4, capthick=1.5, zorder=3)
            sign, limit = needs_text(r["needs"])
            ax.plot([limit, limit], [i - 0.3, i + 0.3], color=theme["ink"], linewidth=2.0, solid_capstyle="butt", zorder=4)
            ax.text(limit, i - 0.33, f"needs {sign} {limit:.2f}", ha="center", va="bottom", fontsize=8.5, color=theme["ink"])
            ax.text(max(hi, limit) + 0.02, i, f"{r['count']} of {r['total']} = {rate:.3f}", ha="left", va="center", fontsize=9, color=theme["ink"])
            
        names = []
        for r in rows:
            head, _, tail = r["bar"].partition(": ")
            names.append(f"{head}\n{tail}" if tail else head)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(names, color=theme["ink2"], fontsize=9)
        ax.set_ylim(len(rows) - 0.45, -0.75)
        ax.set_xlim(0, 1)
        ax.set_xlabel("rate")
        fig.subplots_adjust(left=0.34, top=0.86)
        add_titles(fig, theme, "The AI analyst's answer checks against their three bars",
                   "300 test questions, gemma4-26b; bar = rate, whisker = 95% Wilson interval, "
                   "tick = the threshold it must clear")
        return fig

    return both_themes(out, "analyst", draw)


CHARTS = [chart_latency, chart_vectorized, chart_eviction, chart_ceilings, chart_analyst]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "docs" / "data")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "images")
    args = parser.parse_args(argv)
    for chart in CHARTS:
        written = chart(args.data, args.out)
        name = chart.__name__.removeprefix("chart_")
        if written:
            print(f"{name} -> " + ", ".join(str(p) for p in written))
        else:
            print(f"skip {name}: its data file is not in {args.data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
