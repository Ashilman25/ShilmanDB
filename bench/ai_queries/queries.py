#The ten queries: their plain-SQL parts, the true answers and the accuracy targets.

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .data_prep import is_case
from .labels import TierLabels

ROW_LIMIT = 20
TOP_N = 10


@dataclass(frozen=True)
class RowTargets:
    precision: float  # for the class question, this is the accuracy
    recall: float


@dataclass(frozen=True)
class QuerySpec:
    no: int
    kind: str  # "count", "groups", "top10", "rows" or "average"
    qids: tuple[str, ...]
    estimate_ok: bool
    targets: Mapping[str, RowTargets]
    group_col: str | None  # a plain column to group on; None for AI groups (Q2, Q7)


def _t(precision: float, recall: float | None = None) -> RowTargets:
    return RowTargets(precision, precision if recall is None else recall)


QUERIES: dict[int, QuerySpec] = {q.no: q for q in (
    QuerySpec(1, "count", ("F1",), True, {"F1": _t(0.97)}, None),
    QuerySpec(2, "groups", ("C2",), True, {"C2": _t(0.97)}, None),
    QuerySpec(3, "top10", ("F3",), True, {"F3": _t(0.97)}, "store"),
    QuerySpec(4, "rows", ("F4",), False, {}, None),
    QuerySpec(5, "count", ("F5a", "F5b"), True, {"F5a": _t(0.985), "F5b": _t(0.985)}, None),
    QuerySpec(6, "top10", ("F6",), True, {"F6": _t(0.97)}, "title"),
    QuerySpec(7, "top10", ("E7",), True, {}, None),
    QuerySpec(8, "count", ("F8",), True, {"F8": _t(0.97)}, None),
    QuerySpec(9, "top10", ("F3",), True, {"F3": _t(0.97)}, "store"),
    QuerySpec(10, "average", ("F10",), True, {"F10": _t(0.99)}, None),
)}

# plain SQL parts 


def newest_cutoff(reviews: pd.DataFrame) -> int:
    return int(np.quantile(reviews["timestamp"].to_numpy(), 0.75, method="lower"))


def _review_mask(no: int, reviews: pd.DataFrame) -> pd.Series:
    rating = reviews["rating"]
    if no == 1:
        return rating <= 2
    if no == 2:
        return rating <= 3
    if no == 4:
        return reviews["verified_purchase"]
    if no == 7:
        return rating == 1
    if no == 9:
        return reviews["timestamp"] >= newest_cutoff(reviews)
    
    return pd.Series(True, index=reviews.index)


#The rows that pass query `no`'s plain filters and joins, in table order.
def scope(no: int, reviews: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    rows = reviews[_review_mask(no, reviews)]
    out = rows[["review_id", "text_key", "rating", "timestamp"]].copy()
    group_col = QUERIES[no].group_col
    if group_col is None:
        return out.reset_index(drop=True)
    
    by_asin = products.drop_duplicates("parent_asin").set_index("parent_asin")
    has_product = rows["parent_asin"].isin(by_asin.index)
    if no == 6:
        small_case = by_asin["categories"].map(is_case) & (by_asin["price"] < 20)  # NaN price fails
        has_product &= rows["parent_asin"].isin(small_case.index[small_case])
        
    out["group"] = rows["parent_asin"].map(by_asin[group_col])
    return out[has_product.to_numpy()].reset_index(drop=True)


def naive_rows(no: int, reviews: pd.DataFrame, products: pd.DataFrame) -> int:
    return int(_review_mask(no, reviews).sum())


def question_scopes(reviews: pd.DataFrame, products: pd.DataFrame) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for no, spec in QUERIES.items():
        keys = set(scope(no, reviews, products)["text_key"])
        for qid in spec.qids:
            out.setdefault(qid, set()).update(keys)
            
    return out


#clean-up

_NONE_WORDS = {
    "", "none", "n/a", "na", "unknown", "not mentioned", "not specified", "not stated",
    "unspecified", "no model", "no phone model"
}
_EDGE_PUNCT = " .,;:!?\"'`()[]"


#Lowercase, collapse spaces, trim edge punctuation, map 'no answer' words to 'none'.
def normalize_model(answer: str, aliases: Mapping[str, str]) -> str:
    cleaned = " ".join((answer or "").lower().split()).strip(_EDGE_PUNCT)
    if cleaned in _NONE_WORDS:
        return "none"
    return aliases.get(cleaned, cleaned)


#A JSON object of {spelling: name}; {} if the file is missing. Both sides are cleaned.
def load_aliases(path) -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        return {}
    
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {normalize_model(k, {}): normalize_model(v, {}) for k, v in raw.items()}


#answers 


@dataclass(frozen=True)
class Truth:
    answer: Any
    counts: Mapping[str, int] | None
    matches: frozenset[int] | None = None


def _decision(decisions: Mapping[str, Mapping[str, str]], qid: str, key: str) -> str:
    try:
        return decisions[qid][key]
    except KeyError:
        raise KeyError(f"no {qid} decision for text key {key}") from None


#True when every filter says yes
# Any 'no' wins; a missing answer is an error.
def _passes(spec: QuerySpec, decisions, key: str) -> bool:
    seen = [decisions.get(q, {}).get(key) for q in spec.qids]
    if "no" in seen:
        return False
    if None in seen:
        raise KeyError(f"no {spec.qids[seen.index(None)]} decision for text key {key}")
    
    return True


def _group_counts(no: int, scope_df: pd.DataFrame, decisions, aliases) -> dict[str, int]:
    spec = QUERIES[no]
    counts: Counter[str] = Counter()
    
    for row in scope_df.itertuples(index=False):
        if spec.kind == "groups":
            counts[_decision(decisions, "C2", row.text_key)] += 1
        elif no == 7:
            model = normalize_model(_decision(decisions, "E7", row.text_key), aliases)
            if model != "none":
                counts[model] += 1
        elif _passes(spec, decisions, row.text_key):
            counts[row.group] += 1
            
    return dict(counts)


def _top(counts: Mapping[str, int]) -> list[str]:
    return [k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_N]]


#True when every filter says yes. Any 'no' wins; a missing answer is an error."""
def answer_from(no: int, scope_df: pd.DataFrame, decisions: Mapping[str, Mapping[str, str]], aliases: Mapping[str, str]) -> Any:
    spec = QUERIES[no]
    if spec.kind == "count":
        return sum(_passes(spec, decisions, k) for k in scope_df["text_key"])
    
    if spec.kind == "groups":
        return _group_counts(no, scope_df, decisions, aliases)
    
    if spec.kind == "top10":
        return _top(_group_counts(no, scope_df, decisions, aliases))
    
    if spec.kind == "rows":
        asked = decisions.get("F4", {})
        yes = [int(r.review_id) for r in scope_df.itertuples(index=False) if asked.get(r.text_key) == "yes"]
        return yes[:ROW_LIMIT]
    
    ratings = [r.rating for r in scope_df.itertuples(index=False) if _passes(spec, decisions, r.text_key)]
    return float(np.mean(ratings)) if ratings else None


def truth(no: int, reviews: pd.DataFrame, products: pd.DataFrame, big: TierLabels, aliases: Mapping[str, str]) -> Truth:
    spec = QUERIES[no]
    scope_df = scope(no, reviews, products)
    answer = answer_from(no, scope_df, big.answers, aliases)

    if spec.kind == "groups":
        return Truth(answer, dict(answer))
    
    if spec.kind == "top10":
        return Truth(answer, _group_counts(no, scope_df, big.answers, aliases))
    
    if spec.kind == "rows":
        asked = big.answers["F4"]
        return Truth(answer, None, frozenset(int(r) for r, k in zip(scope_df["review_id"], scope_df["text_key"]) if asked[k] == "yes"))
    return Truth(answer, None)


#Big-model calls naive makes. Q5 adds F5b on the rows where F5a is yes; Q4 stops at the 20th yes
def naive_cost(no: int, reviews: pd.DataFrame, products: pd.DataFrame, big: TierLabels) -> int:
    if no == 5:
        yes = sum(big.answers["F5a"][k] == "yes" for k in reviews["text_key"])
        return len(reviews) + yes
    
    if no == 4:
        asked = big.answers["F4"]
        found = 0
        keys = reviews.loc[reviews["verified_purchase"], "text_key"].tolist()
        for position, key in enumerate(keys, start=1):
            found += asked[key] == "yes"
            if found == ROW_LIMIT:
                return position
        return len(keys)
    return naive_rows(no, reviews, products)


# does an answer meet its target? 


def _margin(true_count: float) -> float:
    return max(0.05 * true_count, 10)


def meets(no: int, answer: Any, truth: Truth) -> bool:
    kind = QUERIES[no].kind
    eps = 1e-9
    if kind == "count":
        return abs(answer - truth.answer) <= _margin(truth.answer) + eps
    
    if kind == "groups":
        return all(abs(answer.get(k, 0) - truth.answer.get(k, 0)) <= _margin(truth.answer.get(k, 0)) + eps for k in set(answer) | set(truth.answer))
    
    if kind == "top10":
        listed = truth.answer
        if not listed:
            return True
        
        last = truth.counts[listed[-1]]
        top_set = {k for k, c in truth.counts.items() if c >= last}
        return len(set(answer[:TOP_N]) & top_set) >= min(9, len(listed))
    
    if kind == "rows":
        matches = truth.matches if truth.matches is not None else frozenset(truth.answer)
        return len(answer) <= ROW_LIMIT and len(set(answer) & matches) >= min(19, len(matches))
    
    if truth.answer is None or answer is None:
        return answer is None and truth.answer is None
    return abs(answer - truth.answer) <= 0.05 + eps
