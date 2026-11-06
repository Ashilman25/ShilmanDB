#Scoring a replayed session against the truth, and the go bar.
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .queries import QUERIES, TOP_N, Truth
from .questions import QUESTIONS

MET_SHARE = 0.95        # runs that must meet the target
BAR1_DIVISOR = 10       # shilmandb must cost at most naive / 10 ...
BAR1_QUERIES = 7        # ... on this many queries
BAR2_DIVISOR = 3        # and at most the incumbent / 3 ...
BAR2_QUERIES = 5        # ... on this many
BAR2_WORST = 1.1        # and never more than this times the incumbent
BAR3_AGREEMENT = 0.90   # local model vs reference, on every question
INCUMBENTS = ("incumbent_snowflake", "incumbent_bigquery")
SLACK = 1e-9            # float noise in cost comparisons


def error_of(no: int, answer: Any, truth: Truth) -> float:
    kind = QUERIES[no].kind
    if kind == "count":
        return abs(answer - truth.answer) / max(truth.answer, 1)
    
    if kind == "groups":
        names = set(answer) | set(truth.answer)
        return max((abs(answer.get(g, 0) - truth.answer.get(g, 0)) / max(truth.answer.get(g, 0), 1) for g in names), default=0.0)

    if kind == "top10":
        listed = truth.answer
        if not listed:
            return 0.0
        
        last = truth.counts[listed[-1]]
        top_set = {g for g, c in truth.counts.items() if c >= last}  # ties at the last place count
        overlap = len(set(answer[:TOP_N]) & top_set)
        return float(max(0, min(TOP_N, len(listed)) - overlap))
    
    if kind == "rows":
        matches = truth.matches if truth.matches is not None else frozenset(truth.answer)
        return float(sum(r not in matches for r in answer))
    
    if truth.answer is None or answer is None:
        return 0.0 if answer is None and truth.answer is None else math.inf
    return abs(answer - truth.answer)


#  did the run's own claim hold? 


def _interval_held(run, truth: Truth) -> bool:
    s, kind = run.stated, QUERIES[run.query].kind
    if kind in ("count", "average"):
        if truth.answer is None:
            return False
        return s["lo"] - SLACK <= truth.answer <= s["hi"] + SLACK
    
    return all(s["lo"][g] - SLACK <= truth.counts.get(g, 0) <= s["hi"][g] + SLACK for g in s["lo"])


#Realized precision and recall (accuracy for the class question) meet the claimed per-row targets.
def _cascade_held(run, truth: Truth, big) -> bool:
    targets = run.stated.get("targets") or {}
    if not targets:
        matches = truth.matches if truth.matches is not None else frozenset(truth.answer)
        return all(r in matches for r in run.answer)
    
    if big is None or run.decisions is None:
        raise ValueError("a cascade claim needs the big labels and the run's decisions")

    for qid, want in targets.items():
        decided = run.decisions[qid]
        got = np.array(list(decided.values()), dtype=object)
        real = np.array([big.answers[qid][k] for k in decided], dtype=object)
        if "accuracy" in want:
            if np.mean(got == real) < want["accuracy"] - SLACK:
                return False
            continue
        
        pred, true = got == "yes", real == "yes"
        hit = int((pred & true).sum())
        
        precision = hit / int(pred.sum()) if pred.any() else 1.0
        recall = hit / int(true.sum()) if true.any() else 1.0
        if precision < want["precision"] - SLACK or recall < want["recall"] - SLACK:
            return False
    return True


def stated_held(run, truth: Truth, big=None) -> bool:
    method = run.stated.get("method")
    if method == "full":
        return True
    
    if method == "estimate":
        return _interval_held(run, truth)
    
    if method == "cascade":
        return _cascade_held(run, truth, big)
    
    raise ValueError(f"unknown claim method {method!r}")


# per approach and query 


def _p95(values) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), 0.95, method="higher"))


SUMMARY_COLUMNS = [
    "approach", "query", "met_runs", "met", 
    "median_cost", "p95_cost", "median_big", "p95_big", "median_small",
    "median_error", "p95_error", "charged"
]


#One row per (approach, query). Extra columns (proxy, embed_rows, stated_held) are summarized when present
def summarize(runs: pd.DataFrame, naive: Mapping[int, float], seeds: int) -> pd.DataFrame:
    rows = []
    for (approach, query), g in runs.groupby(["approach", "query"], sort=True):
        met_runs = int(g["met"].astype(bool).sum())
        met = met_runs >= MET_SHARE * seeds - SLACK
        median_cost = float(np.median(g["cost"]))
        
        row = {
            "approach": approach, "query": int(query), "met_runs": met_runs, "met": bool(met),
            "median_cost": median_cost, "p95_cost": _p95(g["cost"]),
            "median_big": float(np.median(g["big"])), "p95_big": _p95(g["big"]),
            "median_small": float(np.median(g["small"])),
            "median_error": float(np.median(g["error"])), "p95_error": _p95(g["error"]),
            "charged": median_cost if met else float(naive[int(query)])
        }
        
        if "proxy" in g:
            row["median_proxy"] = float(np.median(g["proxy"]))
        if "embed_rows" in g:
            row["median_embed_rows"] = float(np.median(g["embed_rows"]))
        if "stated_held" in g:
            row["stated_held_share"] = float(g["stated_held"].astype(float).mean())
            
        rows.append(row)
    return pd.DataFrame(rows)


#Per query, the incumbent's cost: the cheaper variant that met its target, but never above naive's cost.
def incumbent_best(summary: pd.DataFrame, naive: Mapping[int, float] | None = None) -> pd.DataFrame:
    rows = []
    inc = summary[summary["approach"].isin(INCUMBENTS)]
    
    for query, g in inc.groupby("query", sort=True):
        met = g[g["met"]]
        naive_cost = float(naive[int(query)]) if naive is not None else float(g["charged"].max())
        
        if len(met):
            best = met.sort_values(["charged", "approach"], kind="stable").iloc[0]
            if naive is None or float(best["charged"]) <= naive_cost:
                rows.append({"query": int(query), "cost": float(best["charged"]), "variant": best["approach"]})
                continue
            
        rows.append({
            "query": int(query), 
            "cost": naive_cost, 
            "variant": "naive"
        })
    return pd.DataFrame(rows, columns=["query", "cost", "variant"])

#Bar 1: shilmandb met its target and its cost <= naive / 10, on at least 7 queries.
#Bar 2: met and cost <= incumbent / 3 on at least 5 queries, and (met or not) charged cost never above 1.1 x the incumbent.
#Bar 3: agreement >= 0.90 on every question; None while agreement is missing (or has no answer for some question).
    
#Outcome: "no-go" unless bars 1 and 2 pass; then "go" if bar 3 passed, else "go pending a paid run" (so a missing bar 3 counts as not yet passed).
def bars(summary: pd.DataFrame, inc_best: pd.DataFrame, naive: Mapping[int, float], agreement: Mapping[str, float] | None) -> dict:
    mine = summary[summary["approach"] == "shilmandb"].set_index("query")
    best = inc_best.set_index("query")
    detail = []
    
    for query in sorted(naive):
        cost, inc = float(mine.loc[query, "charged"]), float(best.loc[query, "cost"])
        met = bool(mine.loc[query, "met"])  # an unmet query is charged naive and never counts toward a bar
        detail.append({
            "query": int(query), "naive": float(naive[query]), "incumbent": inc,
            "incumbent_variant": str(best.loc[query, "variant"]),
            "shilmandb": cost, "shilmandb_met": met,
            "within_naive_div": bool(met and cost * BAR1_DIVISOR <= naive[query] * (1 + SLACK)),
            "within_incumbent_div": bool(met and cost * BAR2_DIVISOR <= inc * (1 + SLACK)),
            "too_slow": bool(cost > BAR2_WORST * inc * (1 + SLACK))
        })
        
    n1 = sum(d["within_naive_div"] for d in detail)
    n2 = sum(d["within_incumbent_div"] for d in detail)
    
    worse = [d["query"] for d in detail if d["too_slow"]]
    bar1 = n1 >= BAR1_QUERIES
    bar2 = n2 >= BAR2_QUERIES and not worse
    complete = bool(agreement) and all(q in agreement for q in QUESTIONS)
    bar3 = all(agreement[q] >= BAR3_AGREEMENT - SLACK for q in QUESTIONS) if complete else None
    
    outcome = "no-go" if not (bar1 and bar2) else ("go" if bar3 else "go pending a paid run")
    return {
        "outcome": outcome, "bar1": bool(bar1), "bar2": bool(bar2), "bar3": bar3,
        "bar1_queries": n1, "bar2_queries": n2, "bar2_too_slow_queries": worse,
        "agreement": dict(agreement) if complete else None, "per_query": detail
    }
