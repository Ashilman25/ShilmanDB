# The 15 per-query filters as SQL and numpy masks, and the filtered-search query sets
from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# Load vc_common from next to this file, reusing it if it's already loaded.
def _load_common():
    path = Path(__file__).resolve().parent / "vc_common.py"
    loaded = sys.modules.get("vc_common")
    if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() == path:
        return loaded

    spec = importlib.util.spec_from_file_location("vc_common", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"can't load {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["vc_common"] = module
    spec.loader.exec_module(module)
    return module


vc = _load_common()

FILTER_NAMES = (
    "date_0.1", "date_0.3", "date_1", "date_3", "date_10", "date_30", "none",
    "product_random", "product_near", "rating_1", "rating_2", "rating_le2", "rating_5",
    "category_cases", "unverified"
)

DATE_SHARES = {
    "date_0.1": .001, 
    "date_0.3": .003, 
    "date_1": .01,
    "date_3": .03,
    "date_10": .10,
    "date_30": .30,
}
CASES = "Cases, Holsters & Sleeves"
PART1_PER_GROUP = {"70k": 100, "500k": 50}
MIN_PRODUCT_REVIEWS = 50
EPOCH = dt.date(1970, 1, 1)
_SQL_OPS = {"eq": "=", "le": "<="}


# One filter: its name, family, the column it reads, op and values.
@dataclass(frozen=True)
class Filter:
    name: str
    family: str          # date | none | product | rating | category | verified
    column: str | None
    op: str              # none | between | eq | le
    values: tuple


# The table's filter columns as numpy arrays, indexed by row id.
@dataclass
class TableArrays:
    product_id: np.ndarray       # int32
    rating: np.ndarray           # int16
    day: np.ndarray              # int32, days since 1970-01-01
    verified: np.ndarray         # bool
    category: np.ndarray         # object (str)
    vec_row: np.ndarray          # int64
    product_counts: np.ndarray   # reviews per product id

    # Rows in the table.
    @property
    def n_rows(self) -> int:
        return len(self.product_id)

    # Read a table CSV; ids must be 0..n-1 in order, verified true or false.
    @classmethod
    def from_csv(cls, path: Path) -> TableArrays:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if not np.array_equal(df["id"].astype(np.int64).to_numpy(), np.arange(len(df))):
            raise ValueError(f"{path}: id must be 0..n-1 in order")

        bad = ~df["verified"].isin(["true", "false"])
        if bad.any():
            raise ValueError(f"{path}: verified must be true or false, got {df['verified'][bad].iloc[0]!r}")

        product_id = df["product_id"].astype(np.int32).to_numpy()
        dates = np.array(df["review_date"].tolist(), dtype="datetime64[D]")
        if np.isnat(dates).any():
            raise ValueError(f"{path}: every review_date must be a date")
        
        return cls(
            product_id=product_id,
            rating=df["rating"].astype(np.int16).to_numpy(),
            day=dates.astype(np.int64).astype(np.int32),
            verified=(df["verified"] == "true").to_numpy(),
            category=df["category"].to_numpy(dtype=object),
            vec_row=df["vec_row"].astype(np.int64).to_numpy(),
            product_counts=np.bincount(product_id),
        )


# A value written as a SQL literal; numpy scalars become plain Python first.
def _literal(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, dt.date):
        return f"'{v.isoformat()}'"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    raise TypeError(f"can't write {v!r} as a SQL literal")


# The filter's WHERE clause, without the WHERE; empty for no filter.
def to_sql(f: Filter) -> str:
    if f.op == "none":
        return ""
    if f.op == "between":
        lo, hi = f.values
        return f"{f.column} BETWEEN {_literal(lo)} AND {_literal(hi)}"
    if f.op not in _SQL_OPS:
        raise ValueError(f"unknown filter op {f.op!r}")
    
    return f"{f.column} {_SQL_OPS[f.op]} {_literal(f.values[0])}"


# The table column a filter reads, with dates as days since 1970-01-01.
def _column(t: TableArrays, column: str) -> np.ndarray:
    columns = {
        "review_date": t.day, 
        "product_id": t.product_id, 
        "rating": t.rating,
        "verified": t.verified, 
        "category": t.category
    }
    return columns[column]


# A filter value in the column's units: dates as day numbers.
def _value(v):
    return (v - EPOCH).days if isinstance(v, dt.date) else v


# Which rows pass the filter, as a bool array by row id.
def mask(t: TableArrays, f: Filter) -> np.ndarray:
    if f.op == "none":
        return np.ones(t.n_rows, dtype=bool)

    col = _column(t, f.column)
    if f.op == "between":
        lo, hi = map(_value, f.values)
        return (col >= lo) & (col <= hi)
    if f.op == "eq":
        return np.asarray(col == _value(f.values[0]), dtype=bool)
    if f.op == "le":
        return col <= _value(f.values[0])
    raise ValueError(f"unknown filter op {f.op!r}")


# Whole-day window from a random start, its rows as close to the target as possible.
# Ties between the two candidate ends go to the longer window.
def date_window(day_values: np.ndarray, day_counts: np.ndarray, target_rows: float, rng) -> tuple[int, int]:
    counts = np.asarray(day_counts, dtype=np.int64)
    if len(counts) != len(day_values):
        raise ValueError("day_values and day_counts differ in length")

    to_end = np.cumsum(counts[::-1])[::-1]
    starts = np.flatnonzero(to_end >= target_rows)
    if len(starts) == 0:
        raise ValueError(f"no window reaches {target_rows} rows")
    s = int(starts[rng.integers(len(starts))])

    # cum[j] is the rows before day j, so a window s..e holds cum[e + 1] - cum[s].
    cum = np.concatenate([[0], np.cumsum(counts)])
    e = max(s, int(np.searchsorted(cum, cum[s] + target_rows, side="left")) - 1)
    total, shorter = cum[e + 1] - cum[s], cum[e] - cum[s]
    if e > s and target_rows - shorter < total - target_rows:
        e -= 1
    return int(day_values[s]), int(day_values[e])


# The rng one query's filters are drawn with: seeded by size, qid and "filters".
def query_rng(size: str, qid: str) -> np.random.Generator:
    return np.random.default_rng(vc.seed_for(size, qid, "filters"))


# The 15 filters for one query, in FILTER_NAMES order.
# Draws from rng in a fixed order: the six date windows, then the random product.
def make_filters(t: TableArrays, near_product: int, rng) -> list[Filter]:
    near = int(near_product)
    if not 0 <= near < len(t.product_counts) or t.product_counts[near] == 0:
        raise ValueError(f"near product {near} has no rows in the table")
    
    big = np.flatnonzero(t.product_counts >= MIN_PRODUCT_REVIEWS)
    if len(big) == 0:
        raise ValueError(f"no product has at least {MIN_PRODUCT_REVIEWS} reviews")

    day_values, day_counts = np.unique(t.day, return_counts=True)
    filters = []
    for name, share in DATE_SHARES.items():
        d1, d2 = date_window(day_values, day_counts, share * t.n_rows, rng)
        dates = (EPOCH + dt.timedelta(days=d1), EPOCH + dt.timedelta(days=d2))
        filters.append(Filter(name, "date", "review_date", "between", dates))

    product = int(big[rng.integers(len(big))])
    filters += [
        Filter("none", "none", None, "none", ()),
        Filter("product_random", "product", "product_id", "eq", (product,)),
        Filter("product_near", "product", "product_id", "eq", (near,)),
        Filter("rating_1", "rating", "rating", "eq", (1,)),
        Filter("rating_2", "rating", "rating", "eq", (2,)),
        Filter("rating_le2", "rating", "rating", "le", (2,)),
        Filter("rating_5", "rating", "rating", "eq", (5,)),
        Filter("category_cases", "category", "category", "eq", (CASES,)),
        Filter("unverified", "verified", "verified", "eq", (False,)),
    ]
    return filters


# The filtered-search queries as (qid, group, half): reviews first, then questions.
# Each group is cut to n, then split in half by a seeded permutation.
def part1_queries(test_review_qids: list[str], written_qids: list[str], ai53_qids: list[str], size: str, n_per_group: int | None = None) -> list[tuple[str, str, str]]:
    most = PART1_PER_GROUP[vc.check_size(size)]
    n = most if n_per_group is None else n_per_group
    if not 1 <= n <= most:
        raise ValueError(f"queries per group must be 1..{most} at {size}, got {n}")
    if len(ai53_qids) != 10:
        raise ValueError(f"need exactly 10 ai53 questions, got {len(ai53_qids)}")
    if len(written_qids) < 90:
        raise ValueError(f"need at least 90 written questions, got {len(written_qids)}")
    if len(test_review_qids) < n:
        raise ValueError(f"need at least {n} test reviews, got {len(test_review_qids)}")
    
    lists = {"test reviews": test_review_qids, "written": written_qids, "ai53": ai53_qids}
    for name, qids in lists.items():
        if len(set(qids)) != len(qids):
            raise ValueError(f"{name} qids have duplicates")
        
    seen = set().union(*lists.values())
    if len(seen) != sum(len(q) for q in lists.values()):
        raise ValueError("a qid is in more than one of the test review, written and ai53 lists")

    order = np.random.default_rng(vc.seed_for("part1_questions")).permutation(len(written_qids))
    questions = list(ai53_qids) + [written_qids[i] for i in order[:90]]
    reviews = sorted(test_review_qids)

    rows = []
    for group, qids in (("review", reviews[:n]), ("question", questions[:n])):
        perm = np.random.default_rng(vc.seed_for("part1_halves", size, group)).permutation(n)
        train = set(perm[: n // 2].tolist())
        rows += [(qid, group, "train" if i in train else "test") for i, qid in enumerate(qids)]
    return rows
