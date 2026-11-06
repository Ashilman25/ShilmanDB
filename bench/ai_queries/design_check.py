#Does the test table have enough in it to tell groups and rows apart?
import argparse
import sys
from pathlib import Path
from typing import Mapping

import pandas as pd

from .approaches import CALIBRATION
from .common import paths
from .data_prep import read_tables
from .labels import TierLabels, load_tier
from .queries import QUERIES, ROW_LIMIT, _group_counts, load_aliases, naive_rows, scope

TOP_QUERIES = (3, 6, 7, 9)
MIN_TENTH_GROUP = 5
MIN_Q4_YES = 40
MIN_Q8_MATCHES = 30
REPORT_ONLY = "Q8"


def _yes_count(no: int, qid: str, reviews, products, small: TierLabels) -> int:
    asked = small.answers.get(qid, {})
    return sum(asked.get(k) == "yes" for k in scope(no, reviews, products)["text_key"])


#The query's scope rows whose text key the small model has answered for every question it asks
def _labeled_scope(no: int, reviews, products, small: TierLabels) -> pd.DataFrame:
    df = scope(no, reviews, products)
    have = [small.answers.get(qid, {}) for qid in QUERIES[no].qids]
    return df[[all(k in h for h in have) for k in df["text_key"]]]


#How many distinct scope keys the top-10 counts leave out because the small model has no label for them
def missing_labels(reviews, products, small: TierLabels) -> int:
    left_out: set[str] = set()
    for no in TOP_QUERIES:
        df = scope(no, reviews, products)
        left_out.update(set(df["text_key"]) - set(_labeled_scope(no, reviews, products, small)["text_key"]))
    return len(left_out)


def check(reviews: pd.DataFrame, products: pd.DataFrame, small: TierLabels, aliases: Mapping[str, str]) -> pd.DataFrame:
    rows = []
    for no in TOP_QUERIES:
        counts = sorted(_group_counts(no, _labeled_scope(no, reviews, products, small), small.answers, aliases).values(), reverse=True)
        tenth = counts[9] if len(counts) >= 10 else 0
        rows.append({
            "query": f"Q{no}", 
            "measure": "10th group count", 
            "value": tenth, 
            "ok": tenth >= MIN_TENTH_GROUP
        })
        
    q4 = _yes_count(4, "F4", reviews, products, small)
    rows.append({
        "query": "Q4",
        "measure": "verified yes count",
        "value": q4,
        "ok": q4 >= MIN_Q4_YES
    })
    
    q8 = _yes_count(8, "F8", reviews, products, small)
    rows.append({
        "query": "Q8",
        "measure": "match count",
        "value": q8,
        "ok": q8 >= MIN_Q8_MATCHES
    })
    return pd.DataFrame(rows, columns=["query", "measure", "value", "ok"])

#1 if any row fails, except the report-only one
def exit_code(table: pd.DataFrame) -> int:
    failed = table[(~table["ok"]) & (table["query"] != REPORT_ONLY)]
    return 1 if len(failed) else 0


BAR1_QUERIES = 7  # bar 1 needs this many of the 10 queries at naive / 10 or below


#Per query, from the tables alone: the range naive's cost must lie in, and the bar-1 floor.
def naive_bounds(reviews: pd.DataFrame, products: pd.DataFrame, calibration: int = CALIBRATION) -> pd.DataFrame:
    rows = []
    for no, spec in QUERIES.items():
        if no == 5:
            low, high = len(reviews), 2 * len(reviews)
        elif no == 4:
            verified = int(reviews["verified_purchase"].sum())
            low, high = min(ROW_LIMIT, verified), verified
        else:
            low = high = naive_rows(no, reviews, products)
            
        floor = 10 * (ROW_LIMIT if spec.kind == "rows" else calibration * len(spec.qids))
        reach = "yes" if low >= floor else "no" if high < floor else "unknown"
        rows.append({
            "query": f"Q{no}",
            "naive_low": low,
            "naive_high": high,
            "floor": floor,
            "reachable": reach
        })
        
    return pd.DataFrame(rows, columns=["query", "naive_low", "naive_high", "floor", "reachable"])


def _naive_main(root: Path) -> int:
    p = paths(root)
    reviews, products = read_tables(p.data)
    table = naive_bounds(reviews, products)
    
    print(table.to_string(index=False))
    can, maybe = int((table["reachable"] == "yes").sum()), int((table["reachable"] == "unknown").sum())
    print(f"{can} of {len(table)} queries can reach bar 1, {maybe} unknown without labels (bar 1 needs {BAR1_QUERIES})")
    
    return 0 if can + maybe >= BAR1_QUERIES else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(paths().root))
    parser.add_argument("--tier", default="small")
    sub = parser.add_subparsers(dest="cmd")
    
    naive = sub.add_parser("naive", help="naive cost per query and whether bar 1 is reachable (no labels needed)")
    naive.add_argument("--root", default=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.cmd == "naive":
        return _naive_main(Path(args.root))
    
    p = paths(Path(args.root))
    reviews, products = read_tables(p.data)
    small = load_tier(p.labels / args.tier, args.tier)
    table = check(reviews, products, small, load_aliases(p.root / "e7_aliases.json"))
    shown = table.assign(note=["report only" if q == REPORT_ONLY else "" for q in table["query"]])
    
    print(shown.to_string(index=False))
    if missing := missing_labels(reviews, products, small):
        print(f"{missing} scope keys had no label and were left out of the top-10 counts")
    return exit_code(table)


if __name__ == "__main__":
    sys.exit(main())
