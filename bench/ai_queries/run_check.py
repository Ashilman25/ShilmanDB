#Replay every approach over many seeds, score the runs, apply the bar and write the report.

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from . import score
from .approaches import ALL_ON, Env, Switches, run_session
from .common import paths, write_text_atomic
from .costs import Weights, weights_from
from .data_prep import read_tables
from .design_check import check as design_check
from .labels import TierLabels, load_tier
from .llm import latency_summary
from .oracle import Oracle
from .proxy import Proxy
from .queries import QUERIES, load_aliases, meets, naive_cost, question_scopes, truth
from .questions import QUESTIONS

SWITCH_NAMES = ("T1", "T2", "T3", "T4")
MAIN = (("incumbent_snowflake", Switches()), ("incumbent_bigquery", Switches()), ("shilmandb", ALL_ON))
ABLATION = tuple(("shilmandb", Switches(*(i == j for j in range(4)))) for i in range(4))
ABLATION_ORDER = ("incumbent_bigquery", "shilmandb+T1", "shilmandb+T2", "shilmandb+T3", "shilmandb+T4", "shilmandb")
MAIN_ORDER = ("naive", "incumbent_snowflake", "incumbent_bigquery", "shilmandb")
RUN_COLUMNS = ["approach", "switches", "seed", "query", "mode", "variant", "big", "small", "proxy", "embed_rows", "cost", "error", "met", "stated_held", "answer_json", "stated_json"]
FLOAT_FORMAT = "%.6f"


@dataclass
class ReportInputs:
    latency: pd.DataFrame | None = None        # llm.latency_summary per tier, with a `tier` column
    prices: pd.DataFrame | None = None
    embed_timing: pd.DataFrame | None = None
    proxy_us_per_row: float | None = None
    run_info: Mapping[str, dict] = field(default_factory=dict)  # tier -> run_info.json
    design: dict | None = None
    slice_info: dict | None = None
    agreement: Mapping[str, float] | None = None
    spotcheck: dict | None = None


def switches_label(sw: Switches) -> str:
    on = [n for n, flag in zip(SWITCH_NAMES, (sw.t1, sw.t2, sw.t3, sw.t4)) if flag]
    return "+".join(on) or "none"


def table_label(approach: str, sw: Switches) -> str:
    if approach != "shilmandb" or sw == ALL_ON:
        return approach
    return f"shilmandb+{switches_label(sw)}"  # every switch off is "shilmandb+none"


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot write {type(value).__name__} as JSON")


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, default=_plain)


#replay 


def replay(env: Env, big_labels: TierLabels, small_labels: TierLabels, truths: Mapping, seeds: int, progress: Callable[[str], None] | None = None) -> pd.DataFrame:
    big = lambda: Oracle("big", big_labels)  # noqa: E731
    small = lambda: Oracle("small", small_labels)  # noqa: E731
    rows = []
    
    for approach, sw in (*MAIN, *ABLATION):
        started = time.perf_counter()
        for seed in range(1, seeds + 1):
            for run in run_session(approach, env, big, small, seed, sw):
                t = truths[run.query]
                rows.append({
                    "approach": approach, "switches": switches_label(sw), "label": table_label(approach, sw),
                    "seed": seed, "query": run.query, "mode": run.mode, "variant": run.variant,
                    "big": run.usage.big, "small": run.usage.small, "proxy": run.usage.proxy,
                    "embed_rows": run.usage.embed_rows, "cost": run.usage.cost(env.weights),
                    "error": score.error_of(run.query, run.answer, t),
                    "met": int(meets(run.query, run.answer, t)),
                    "stated_held": int(score.stated_held(run, t, big_labels)),
                    "answer_json": _json(run.answer), "stated_json": _json(run.stated)
                })
                
            if progress:
                progress(f"{table_label(approach, sw)} seed {seed}/{seeds} ({time.perf_counter() - started:.0f} s)")
                
    df = pd.DataFrame(rows)
    return df.sort_values(["approach", "switches", "seed", "query"], kind="stable").reset_index(drop=True)


# time and dollars per tier 


#Seconds and dollars per call (per row for embeddings and the proxy)
# None if any input is missing
def _tier_rates(ri: ReportInputs) -> dict | None:
    if ri.latency is None or ri.prices is None or ri.embed_timing is None or ri.proxy_us_per_row is None:
        return None
    
    rates = {}
    for tier in ("big", "small"):
        t = ri.latency[ri.latency["tier"] == tier]
        w = t["rows"].to_numpy(dtype=float)
        mean = lambda col: float((t[col].to_numpy(dtype=float) * w).sum() / w.sum())  # noqa: E731
        price = ri.prices[ri.prices["tier"] == tier].iloc[0]
        rates[tier] = (mean("median_ms") / 1000.0, (mean("mean_in_tokens") * float(price["usd_per_m_in"]) + mean("mean_out_tokens") * float(price["usd_per_m_out"])) / 1e6)
        if tier == "big":
            big_in = mean("mean_in_tokens")
            
    test = ri.embed_timing[ri.embed_timing["pass"] == "test"].iloc[0]
    embed_price = ri.prices[ri.prices["tier"] == "embed"].iloc[0]
    rates["embed"] = (float(test["seconds"]) / float(test["rows"]), big_in * float(embed_price["usd_per_m_in"]) / 1e6)
    rates["proxy"] = (ri.proxy_us_per_row / 1e6, 0.0)
    return rates


TIME_COLUMNS = ["big_seconds", "small_seconds", "proxy_seconds", "embed_seconds", "total_seconds", "big_usd", "small_usd", "embed_usd", "total_usd"]


def _time_and_dollars(row: Mapping, rates: dict | None) -> dict:
    if rates is None:
        return {c: np.nan for c in TIME_COLUMNS}
    
    calls = {
        "big": row["median_big"], "small": row["median_small"], 
        "proxy": row.get("median_proxy", 0.0), "embed": row.get("median_embed_rows", 0.0)
    }
    sec = {t: calls[t] * rates[t][0] for t in calls}
    usd = {t: calls[t] * rates[t][1] for t in ("big", "small", "embed")}
    
    return {
        "big_seconds": sec["big"], "small_seconds": sec["small"], "proxy_seconds": sec["proxy"],
        "embed_seconds": sec["embed"], "total_seconds": sum(sec.values()),
        "big_usd": usd["big"], "small_usd": usd["small"], "embed_usd": usd["embed"],
        "total_usd": sum(usd.values())
    }


# ---- run everything ----


def _naive_rows(naive: Mapping[int, float], seeds: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "approach": "naive", "query": q, "met_runs": seeds, "met": True, "median_cost": c,
        "p95_cost": c, "median_big": c, "p95_big": c, "median_small": 0.0, "median_error": 0.0,
        "p95_error": 0.0, "charged": c, "median_proxy": 0.0, "median_embed_rows": 0.0,
        "stated_held_share": 1.0
    } for q, c in sorted(naive.items())])


def _ordered(df: pd.DataFrame, order: tuple[str, ...]) -> pd.DataFrame:
    rank = {name: i for i, name in enumerate(order)}
    return df.assign(_r=df["approach"].map(rank)).sort_values(["_r", "query"], kind="stable").drop(columns="_r").reset_index(drop=True)


#Replay, score and write runs.csv, results.csv, ablation.csv, bars.json and report.md. Returns the bars
def run_all(env: Env, big_labels: TierLabels, small_labels: TierLabels, reviews: pd.DataFrame,
            products: pd.DataFrame, seeds: int, out_dir: Path, *, report_inputs: ReportInputs | None = None,
            progress: Callable[[str], None] | None = None) -> dict:
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ri = report_inputs or ReportInputs()
    truths = {no: truth(no, reviews, products, big_labels, env.aliases) for no in QUERIES}
    naive = {no: float(naive_cost(no, reviews, products, big_labels)) for no in QUERIES}

    runs = replay(env, big_labels, small_labels, truths, seeds, progress)
    runs[RUN_COLUMNS].to_csv(out_dir / "runs.csv", index=False, lineterminator="\n", float_format=FLOAT_FORMAT)

    summary = score.summarize(runs.assign(approach=runs["label"]), naive, seeds)
    rates = _tier_rates(ri)
    main = _ordered(pd.concat([_naive_rows(naive, seeds), summary[summary["approach"].isin(MAIN_ORDER)]], ignore_index=True), MAIN_ORDER)
    main = pd.concat([main, pd.DataFrame([_time_and_dollars(r, rates) for r in main.to_dict("records")])], axis=1)
    ablation = _ordered(summary[summary["approach"].isin(ABLATION_ORDER)], ABLATION_ORDER)
    main.to_csv(out_dir / "results.csv", index=False, lineterminator="\n", float_format=FLOAT_FORMAT)
    ablation.to_csv(out_dir / "ablation.csv", index=False, lineterminator="\n", float_format=FLOAT_FORMAT)

    best = score.incumbent_best(summary, naive)
    result = score.bars(summary, best, naive, ri.agreement)
    (out_dir / "bars.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    thin = design_check(reviews, products, big_labels, env.aliases)
    sizes = {qid: len(keys) for qid, keys in sorted(question_scopes(reviews, products).items())}
    text = build_report(result, main, ablation, best, env.weights, ri, sizes, thin, seeds)
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    return result


# the report 

METHOD_CHOICES = """\
| Choice | Value |
|---|---|
| Calibration sample | m = 600 distinct keys drawn uniformly from the query's scope (all if fewer), big-labeled, the same for every approach. Proxy cascades train on the first 300 drawn and set cut-offs on the other 300. The small-LLM cascade sets cut-offs on all 600 (and pays 600 small calls). |
| Per-row targets (every approach) | Q1, Q3, Q6, Q8, Q9: precision 0.97, recall 0.97. Q2 (C2): accuracy 0.97. Q5: F5a and F5b each 0.985 / 0.985. Q10: 0.99 / 0.99. Q7: no cascade. Q4: none (see LIMIT). |
| Cut-off estimates | **Point estimates** on the cut-off rows, no confidence bound, the same for every approach (shipped-style; amended with the user before any real run). Candidate cut-offs: a grid of 20 at the 5%, 10%, ..., 100% quantiles of the cheap score on the cut-off rows. Estimates (T1) keep 99% intervals. |
| Yes/no cut-offs | `accept` = the smallest grid point whose accepted cut-off rows' observed precision is at least the target (else +inf). `reject` = the largest grid point where the observed share of calibration positives scoring at or below it is at most 1 - recall target (else -inf, also -inf when calibration has no positives). The band between goes to the next tier. |
| Class cut-off | Accept the cheap model's top class when its probability is at least the smallest grid point whose observed accuracy on the cut-off rows is at least 0.97 (else everything to the big model). |
| Proxy | One-vs-rest L2 logistic regression on standardized embeddings, lambda = 1e-2 on the mean log-loss, 25 Newton steps, numpy only. |
| Three tiers (T3 on) | Cascades tried: proxy>big, small>big, proxy>small>big; the one with the lowest estimated cost from the approach's own calibration rows wins, among those whose estimated precision and recall (accuracy for C2) on those rows meet the per-row targets (for every approach; if none does, the big model reads every key). A three-step cascade gives each stage half of each error allowance. T3 off: proxy>big. Under T3, ShilmanDB-style may pick a different cascade per filter on Q5 (a small extra edge on Q5 only). In the three-tier cascade the small stage's cut-offs are fitted on all 600 calibration rows, not only the proxy's band rows. |
| LIMIT (Q4) | Only big-confirmed rows count toward the 20. Naive streams rows in table order. Snowflake-style: batches of 64 in table order, small LLM on each, big model on rows with small p(yes) >= 0.5 in order, stop at 20. BigQuery-style: equals naive streaming. T2: rank every surviving row by cosine similarity to the question's embedding (no training), then the same small-LLM check, in rank order. |
| T1 estimates | Stage 1 is the calibration sample. Looks at n = 600 * 2^(k-1) at delta_k = delta / 2^k (delta = 0.01). Strata: 4 quartiles of a proxy trained on the first 300 calibration rows (C2: the proxy's top class; E7: a "none vs some" proxy); only rows not used to train it count. Neyman allocation per look. Agresti-Coull variance with the finite-population correction. Stopping: a count when half-width <= max(0.05 * (estimate - half-width), 10); groups when every class meets that (Bonferroni); the average (ratio, delta method) when half-width <= 0.05; a top 10 when the lower bound of the 10th > the upper bound of the 11th (Bonferroni over the top 20). Look 1 (the calibration rows) is free; from its interval ShilmanDB-style projects the sample the stopping rule needs (look 1's half-width scaled by z_k / z_1 * sqrt((1/n_k - 1/M) / (1/n_1 - 1/M)) at each later look size n_k) and keeps sampling only if those big calls, plus the keys covering several scope rows it then asks exactly, are at most 0.75 x the exact cascade's estimated cost; otherwise it goes straight to the exact cascade. Big answers already in hand (T4) count exactly, not as sampled units. The spending cap stays as a backstop: sampling stops once its big calls would pass the exact cost, and the exact cascade then finishes, reusing every answer it has. |
| T4 | One oracle per tier shared across the session's 10 queries (answers keyed by text key, question, tier). A query's calibration sample starts with earlier calibration rows for the same question that fall in its scope, then tops up. |
| T5 | Not built (the spec allows that). |
| Embeddings | One pass over the test rows. Its cost is charged once per session, split evenly over the queries in which that approach used embeddings. |
| Dedupe | Incumbent and ShilmanDB-style ask by text key (exact duplicate texts share one call). Naive pays per row. |
| Fallback | An approach that meets its target in fewer than 95 of the seeds on a query is charged naive's cost there, and such a query never counts toward bar 1 or bar 2's "incumbent / 3" count (the "never above 1.1x" check uses the charged cost on every query). The incumbent on a query is the cheaper of its two variants that met the target, but never above naive's cost; naive's cost if neither met. ShilmanDB-style is not capped. |
| Stated accuracy | A cascade's claim holds when realized precision and recall (class question: accuracy) over every decided key, including keys the big model answered itself, reach the claimed per-row targets. The error column for the rows query counts wrong rows only; a shortfall of rows shows up as "not met", not as error. |
| Time and dollars | Time = calls * the tier's median seconds per row (sequential); dollars = calls * the tier's mean tokens * its dated price. Proxy and embeddings from their timings. The proxy's per-row time comes from a fixed micro-benchmark saved in `runs/proxy_timing.json`. |
"""


def _fmt(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        return "inf" if np.isinf(value) else f"{value:,.2f}"
    return str(value)


def _md(df: pd.DataFrame) -> str:
    if df.empty:
        return "(none)"
    
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _kv(d: Mapping | None) -> str:
    if not d:
        return "not available"
    return "\n".join(f"- {k}: {json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v}" for k, v in d.items())


def _outcome_text(bars: Mapping) -> str:
    def word(b):
        return "pending (no agreement data)" if b is None else ("pass" if b else "fail")
    
    lines = [f"- Bar 1 (cost <= naive / 10 on at least 7 of 10): {word(bars['bar1'])}, "
             f"{bars['bar1_queries']} of 10",
             f"- Bar 2 (cost <= incumbent / 3 on at least 5 of 10, never above 1.1x): {word(bars['bar2'])}, "
             f"{bars['bar2_queries']} of 10; over 1.1x on queries {bars['bar2_too_slow_queries'] or 'none'}",
             f"- Bar 3 (agreement >= 90% on every question): {word(bars['bar3'])}"]

    if bars["outcome"] == "no-go":
        lines.append("\nNo-go is firm: bar 1 or bar 2 failed. Keep the Postgres roadmap; no second tries with changed bars or prompts.")
    elif bars["outcome"] == "go pending a paid run":
        lines.append("\nBars 1 and 2 passed. Bar 3 has not passed (failed or no agreement data yet), so the outcome is go pending a paid run.")
        
    return "\n".join(lines)


def _tier_section(ri: ReportInputs) -> str:
    if not ri.run_info:
        return "not available"
    
    parts = []
    for tier, info in sorted(ri.run_info.items()):
        keep = {k: info[k] for k in ("model", "path", "quantization", "template_sha1", "settings", "date", "truncated_share") if k in info}
        parts.append(f"**{tier}**\n{_kv(keep)}")
    return "\n\n".join(parts)


def _agreement_section(ri: ReportInputs, bars: Mapping) -> str:
    if not ri.agreement:
        return "Agreement with the reference: not available yet, so bar 3 is pending.\n\n" + f"Spot-check: {_kv(ri.spotcheck)}"
    table = pd.DataFrame({"question": list(QUESTIONS), "agreement": [ri.agreement.get(q, np.nan) for q in QUESTIONS]})
    
    return f"{_md(table)}\n\nBar 3: {'pass' if bars['bar3'] else 'fail' if bars['bar3'] is False else 'pending'}." \
        f"\n\nSpot-check:\n{_kv(ri.spotcheck)}"


def build_report(bars: Mapping, main: pd.DataFrame, ablation: pd.DataFrame, best: pd.DataFrame,
                 weights: Weights, ri: ReportInputs, sizes: Mapping[str, int], thin: pd.DataFrame,
                 seeds: int) -> str:
    
    cols = [
        "approach", "query", "median_cost", "p95_cost", "median_big", "p95_big", "median_small",
        "median_proxy", "met_runs", "stated_held_share", "median_error", "p95_error", "charged",
        "big_seconds", "small_seconds", "proxy_seconds", "embed_seconds", "total_seconds",
        "big_usd", "small_usd", "embed_usd", "total_usd"
    ]
    
    ab = ablation.pivot(index="query", columns="approach", values="charged")[list(ABLATION_ORDER)].reset_index()
    detail = pd.DataFrame(bars["per_query"])[[
        "query", "naive", "incumbent", "incumbent_variant", "shilmandb",
        "shilmandb_met", "within_naive_div", "within_incumbent_div",
        "too_slow"
    ]]
    
    prices = _md(ri.prices) if ri.prices is not None else "not available"
    sections = [
        f"# Outcome: {bars['outcome']}\n\n{_outcome_text(bars)}\n\nSeeds 1..{seeds}. Costs are in big-call "
        "equivalents; a query whose target was met in fewer than 95% of the seeds is charged naive's cost.",
        f"## Models, settings and truncation\n\n{_tier_section(ri)}",
        f"## Prices and weights\n\nPrices (dated):\n\n{prices}\n\nWeights (big call = 1): small "
        f"{weights.small:.6g}, proxy per row {weights.proxy:.6g}, embedding per row {weights.embed:.6g}.",
        f"## Test table\n\ndesign.json:\n{_kv(ri.design)}\n\nSlice: {_kv(ri.slice_info)}",
        f"## Label scope sizes (distinct texts per question)\n\n"
        f"{_md(pd.DataFrame({'question': list(sizes), 'texts': list(sizes.values())}))}",
        f"## Main table\n\n{_md(main[[c for c in cols if c in main]])}",
        f"## Incumbent (cheaper variant that met its target) and the bar per query\n\n{_md(detail)}",
        f"## Ablation (charged cost; incumbent_bigquery is shilmandb with every switch off)\n\n{_md(ab)}",
        f"## Bar 3: agreement and spot-check\n\n{_agreement_section(ri, bars)}",
        f"## Thin queries (design check on the big labels)\n\n{_md(thin)}",
        "## T5\n\nT5 not built.",
        f"## Fixed method choices\n\n{METHOD_CHOICES}",
    ]
    return "\n\n".join(sections) + "\n"


# the command line 

PROXY_BENCH_ROWS = 20000


#Microseconds to score one row with a fitted proxy: the median of 5 timings over 20,000 random rows
def measure_proxy_us(dim: int) -> float:
    rng = np.random.default_rng(0)
    model = Proxy().fit(rng.normal(size=(300, dim)), rng.integers(0, 2, 300))
    x = rng.normal(size=(PROXY_BENCH_ROWS, dim))
    times = []
    
    for _ in range(5):
        started = time.perf_counter()
        model.scores(x)
        times.append(time.perf_counter() - started)
    return float(np.median(times)) / PROXY_BENCH_ROWS * 1e6


#The saved timing if there is one, so reruns agree to the byte.
def proxy_us_per_row(runs_dir: Path, dim: int) -> float:
    path = Path(runs_dir) / "proxy_timing.json"
    if path.exists():
        return float(json.loads(path.read_text(encoding="utf-8"))["us_per_row"])
    
    us = measure_proxy_us(dim)
    Path(runs_dir).mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps({
        "us_per_row": us, "dim": dim, 
        "rows": PROXY_BENCH_ROWS, "how": "median of 5 timings of Proxy.scores on random rows"
    }, indent=2) + "\n")
    return us


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


#Everything a run needs from the results folder: (env, big, small, reviews, products, report inputs)
def load_inputs(root: Path):
    p = paths(root)
    
    reviews, products = read_tables(p.data)
    big, small = load_tier(p.labels / "big", "big"), load_tier(p.labels / "small", "small")
    
    emb = np.load(p.embeddings / "test.npy")
    keys = pd.read_csv(p.embeddings / "test_keys.csv", dtype=str, keep_default_na=False)["text_key"].tolist()
    q_emb = np.load(p.embeddings / "questions.npy")
    q_ids = pd.read_csv(p.embeddings / "questions.csv", dtype=str, keep_default_na=False)["qid"].tolist()
    
    latency = pd.concat([latency_summary(p.labels / t).assign(tier=t) for t in ("big", "small")], ignore_index=True)
    prices = pd.read_csv(p.root / "prices.csv", keep_default_na=False)
    timing = pd.read_csv(p.embeddings / "timing.csv", dtype={"pass": str})
    proxy_us = proxy_us_per_row(p.runs, emb.shape[1])
    weights = weights_from(latency, prices, timing, proxy_us)
    
    env = Env(reviews, products, emb, {k: i for i, k in enumerate(keys)}, {q: q_emb[i] for i, q in enumerate(q_ids)}, weights, load_aliases(p.root / "e7_aliases.json"))
    agreement_path = p.reference / "agreement.csv"
    agreement = None
    
    if agreement_path.exists():
        table = pd.read_csv(agreement_path, dtype={"qid": str}, keep_default_na=False)
        agreement = dict(zip(table["qid"], table["agreement"].astype(float)))
        
    ri = ReportInputs(
        latency, prices, timing, proxy_us, 
        {t: _read_json(p.labels / t / "run_info.json") or {} for t in ("big", "small")},
        _read_json(p.data / "design.json"), _read_json(p.data / "slice" / "slice_info.json"),
        agreement, _read_json(p.reference / "spotcheck_score.json")
    )
    return env, big, small, reviews, products, ri


#What a replay would trip over: scope keys a tier has no label for, or no embedding.
def coverage_gaps(env: Env, big: TierLabels, small: TierLabels, reviews: pd.DataFrame, products: pd.DataFrame) -> list[str]:
    scopes = question_scopes(reviews, products)
    gaps = []
    for labels in (big, small):
        for qid, keys in sorted(scopes.items()):
            if labels is small and QUESTIONS[qid].kind == "extract":
                continue
            
            have = labels.answers.get(qid, {})
            missing = sum(k not in have for k in keys)
            if missing:
                gaps.append(f"{labels.tier} {qid}: {missing} of {len(keys)} scope keys have no label")
                
    every = set().union(*scopes.values())
    missing = sum(k not in env.emb_row for k in every)
    if missing:
        gaps.append(f"embeddings: {missing} of {len(every)} scope keys have no embedding")
    return gaps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--root", default=str(paths().root))
    args = parser.parse_args(argv)
    
    env, big, small, reviews, products, ri = load_inputs(Path(args.root))
    gaps = coverage_gaps(env, big, small, reviews, products)
    
    if gaps:
        print("not every scope key is labeled and embedded; finish those first:", file=sys.stderr)
        for line in gaps:
            print(f"  {line}", file=sys.stderr)
        return 1
    
    out = paths(Path(args.root)).runs
    bars = run_all(env, big, small, reviews, products, args.seeds, out, report_inputs=ri, progress=lambda line: print(line, flush=True))
    print(f"outcome: {bars['outcome']} (bar1 {bars['bar1']}, bar2 {bars['bar2']}, bar3 {bars['bar3']})")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
