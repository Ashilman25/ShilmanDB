"""Tests for bench/vector_check/vector_filters.py. Everything runs on small synthetic tables."""
from __future__ import annotations

import datetime as dt
import importlib.util
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC_PATH = Path(__file__).resolve().parents[1] / "vector_filters.py"
spec = importlib.util.spec_from_file_location("vc_vector_filters", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["vc_vector_filters"] = mod
spec.loader.exec_module(mod)

COLUMNS = ["id", "product_id", "rating", "review_date", "verified", "category", "vec_row", "text_key"]
CATEGORIES = ["Cases, Holsters & Sleeves", "Chargers & Power Adapters", "", "Kid's Gear, Tote & Bag",
              "cases, holsters & sleeves", "Screen Protectors"]
EPOCH = dt.date(1970, 1, 1)


# A table frame as read_csv(dtype=str) gives it: big products 0-5, small ones 6-15.
def table_frame(n: int = 2000, n_days: int = 1100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    big = np.repeat(np.arange(6), (n * 9 // 10) // 6)
    products = np.concatenate([big, rng.integers(6, 16, n - len(big))])
    rng.shuffle(products)
    start = dt.date(2018, 1, 1)
    days = rng.integers(0, n_days, n)
    return pd.DataFrame({
        "id": [str(i) for i in range(n)],
        "product_id": [str(p) for p in products],
        "rating": [str(r) for r in rng.choice([1, 2, 3, 4, 5], n, p=[.14, .06, .1, .1, .6])],
        "review_date": [(start + dt.timedelta(days=int(d))).isoformat() for d in days],
        "verified": np.where(rng.random(n) < .9, "true", "false"),
        "category": rng.choice(CATEGORIES, n),
        "vec_row": [str(v) for v in rng.integers(0, n, n)],
        "text_key": "k",
    })[COLUMNS]


# Write the frame as a CSV and load it back as TableArrays.
def load(tmp_path: Path, df: pd.DataFrame, name: str = "table.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return mod.TableArrays.from_csv(path)


# Postgres's SQL must pick the rows the numpy mask counts, for every filter.
def test_sql_and_mask_agree_in_sqlite(tmp_path):
    df = table_frame()
    t = load(tmp_path, df)
    rows = [(int(r.id), int(r.product_id), int(r.rating), r.review_date, int(r.verified == "true"), r.category)
            for r in df.itertuples()]
    extra = [mod.Filter("c", "category", "category", "eq", (v,)) for v in ("Kid's Gear, Tote & Bag", "")]
    with closing(sqlite3.connect(":memory:")) as con:
        con.execute("CREATE TABLE t (id INTEGER, product_id INTEGER, rating INTEGER, review_date TEXT, "
                    "verified INTEGER, category TEXT)")
        con.executemany("INSERT INTO t VALUES (?, ?, ?, ?, ?, ?)", rows)
        for seed in range(5):
            filters = mod.make_filters(t, near_product=3, rng=np.random.default_rng(seed))
            assert [f.name for f in filters] == list(mod.FILTER_NAMES)
            for f in filters + extra:
                where = mod.to_sql(f)
                sql = f"SELECT id FROM t{' WHERE ' + where if where else ''} ORDER BY id"
                got = [r[0] for r in con.execute(sql)]
                want = np.flatnonzero(mod.mask(t, f)).tolist()
                assert got == want, f.name
                if f.name == "none":
                    assert len(want) == t.n_rows
                else:
                    assert 0 < len(want) < t.n_rows, f.name


# Windows land within 10% of the target, or miss by at most one boundary day.
def test_date_windows_hit_shares(tmp_path):
    t = load(tmp_path, table_frame(n=20000, n_days=3000, seed=1))
    values, counts = np.unique(t.day, return_counts=True)
    assert len(values) > 2500
    for seed in range(40):
        rng = np.random.default_rng(seed)
        for name, share in mod.DATE_SHARES.items():
            target = share * t.n_rows
            d1, d2 = mod.date_window(values, counts, target, rng)
            assert values[0] <= d1 <= d2 <= values[-1]
            rows = int(counts[(values >= d1) & (values <= d2)].sum())
            end = int(np.searchsorted(values, d2))
            boundary = int(counts[end:end + 2].max())
            assert abs(rows - target) <= max(.1 * target, boundary), (name, seed, rows, target)
    for f in mod.make_filters(t, near_product=0, rng=mod.query_rng("70k", "q001")):
        if f.family == "date":
            share = mod.mask(t, f).mean()
            assert abs(share - mod.DATE_SHARES[f.name]) <= max(.1 * mod.DATE_SHARES[f.name], counts.max() / t.n_rows)


# The end closer to the target wins, the longer window on a tie.
def test_date_window_picks_closer_end():
    values = np.array([0, 1, 2, 3])
    counts = np.array([5, 5, 5, 5])
    rng = np.random.default_rng(0)
    assert mod.date_window(values, counts, 17.5, rng) == (0, 3)   # 15 and 20 tie: longer
    assert mod.date_window(values, counts, 18.0, rng) == (0, 3)
    assert mod.date_window(values, counts, 17.0, rng) == (0, 2)
    assert mod.date_window(np.array([7, 9]), np.array([100, 1]), 50.0, rng) == (7, 7)   # never an empty window
    with pytest.raises(ValueError):
        mod.date_window(values, counts, 21.0, rng)


# The start is drawn among every day that can still reach the target.
def test_date_window_start_is_uniform_over_valid_days():
    values = np.arange(10, 20)
    counts = np.ones(10, dtype=np.int64)
    rng = np.random.default_rng(3)
    windows = [mod.date_window(values, counts, 3.0, rng) for _ in range(2000)]
    assert {d1 for d1, _ in windows} == set(range(10, 18))
    assert all(d2 - d1 == 2 for d1, d2 in windows)
    starts = np.bincount([d1 - 10 for d1, _ in windows], minlength=8)
    assert starts.min() > 2000 / 8 * .75


# One seed, one set of filters; another seed gives other windows.
def test_filters_deterministic(tmp_path):
    t = load(tmp_path, table_frame())
    a = mod.make_filters(t, 3, mod.query_rng("70k", "q001"))
    assert a == mod.make_filters(t, 3, mod.query_rng("70k", "q001"))
    assert a != mod.make_filters(t, 3, mod.query_rng("70k", "q002"))
    assert a != mod.make_filters(t, 3, mod.query_rng("500k", "q001"))


# The per-query rng is seeded from the size, the qid and "filters".
def test_query_rng_follows_the_seed_rule():
    want = np.random.default_rng(mod.vc.seed_for("500k", "r00042", "filters")).random(4)
    assert mod.query_rng("500k", "r00042").random(4).tolist() == want.tolist()
    assert mod.query_rng("70k", "r00042").random(4).tolist() != want.tolist()


# The random product always has 50+ reviews, and every such product can come up.
def test_product_random_has_50_reviews(tmp_path):
    t = load(tmp_path, table_frame())
    eligible = set(np.flatnonzero(t.product_counts >= 50).tolist())
    assert eligible == set(range(6)) and (t.product_counts[6:] < 50).all()
    at = mod.FILTER_NAMES.index("product_random")
    drawn = {mod.make_filters(t, 3, np.random.default_rng(s))[at].values[0] for s in range(200)}
    assert drawn == eligible


# Exactly 50 reviews is enough for the random product; 49 never is.
def test_product_random_bar_is_50_inclusive(tmp_path):
    df = table_frame(n=150)
    df["product_id"] = [str(p) for p in [0] * 50 + [1] * 49 + [2] * 51]
    t = load(tmp_path, df)
    assert t.product_counts.tolist() == [50, 49, 51]
    at = mod.FILTER_NAMES.index("product_random")
    drawn = {mod.make_filters(t, 1, np.random.default_rng(s))[at].values[0] for s in range(100)}
    assert drawn == {0, 2}


# Errors: no product with 50 reviews, or a near product missing from the table.
def test_make_filters_rejects_bad_products(tmp_path):
    df = table_frame(n=200)
    df["product_id"] = [str(i % 10) for i in range(200)]
    t = load(tmp_path, df)
    with pytest.raises(ValueError):
        mod.make_filters(t, 3, np.random.default_rng(0))
    t = load(tmp_path, table_frame(), name="big.csv")
    with pytest.raises(ValueError):
        mod.make_filters(t, 99, np.random.default_rng(0))


# Each filter has its family, column, op and plain Python values.
def test_make_filters_shapes(tmp_path):
    t = load(tmp_path, table_frame())
    filters = {f.name: f for f in mod.make_filters(t, np.int64(3), mod.query_rng("70k", "q007"))}
    assert list(filters) == list(mod.FILTER_NAMES)
    for name in mod.DATE_SHARES:
        f = filters[name]
        assert (f.family, f.column, f.op) == ("date", "review_date", "between")
        assert all(type(v) is dt.date for v in f.values) and f.values[0] <= f.values[1]
    assert filters["none"] == mod.Filter("none", "none", None, "none", ())
    assert filters["product_near"] == mod.Filter("product_near", "product", "product_id", "eq", (3,))
    assert type(filters["product_near"].values[0]) is int and type(filters["product_random"].values[0]) is int
    assert filters["rating_le2"] == mod.Filter("rating_le2", "rating", "rating", "le", (2,))
    assert filters["rating_5"] == mod.Filter("rating_5", "rating", "rating", "eq", (5,))
    assert filters["category_cases"] == mod.Filter("category_cases", "category", "category", "eq", (mod.CASES,))
    assert filters["unverified"] == mod.Filter("unverified", "verified", "verified", "eq", (False,))


# Quotes inside text values are doubled.
def test_text_values_escaped():
    assert "''" in mod.to_sql(mod.Filter("c", "category", "category", "eq", ("it's",)))
    assert mod.to_sql(mod.Filter("c", "category", "category", "eq", ("it's",))) == "category = 'it''s'"


# Every literal form: dates, text, booleans, bare integers (numpy ones too).
def test_to_sql_forms():
    when = mod.Filter("date_1", "date", "review_date", "between", (dt.date(2019, 1, 2), dt.date(2019, 3, 4)))
    assert mod.to_sql(when) == "review_date BETWEEN '2019-01-02' AND '2019-03-04'"
    assert mod.to_sql(mod.Filter("none", "none", None, "none", ())) == ""
    assert mod.to_sql(mod.Filter("p", "product", "product_id", "eq", (np.int64(7),))) == "product_id = 7"
    assert mod.to_sql(mod.Filter("r", "rating", "rating", "le", (np.int16(2),))) == "rating <= 2"
    assert mod.to_sql(mod.Filter("u", "verified", "verified", "eq", (False,))) == "verified = false"
    assert mod.to_sql(mod.Filter("u", "verified", "verified", "eq", (np.bool_(True),))) == "verified = true"
    assert mod.to_sql(mod.Filter("c", "category", "category", "eq", (mod.CASES,))) == \
        "category = 'Cases, Holsters & Sleeves'"
    with pytest.raises(TypeError):
        mod.to_sql(mod.Filter("x", "rating", "rating", "eq", (2.5,)))
    with pytest.raises(ValueError):
        mod.to_sql(mod.Filter("x", "rating", "rating", "ge", (2,)))


# A one-day window keeps exactly that day's rows.
def test_mask_date_window_is_inclusive(tmp_path):
    df = table_frame()
    t = load(tmp_path, df)
    day = dt.date.fromisoformat(df.review_date.iloc[0])
    m = mod.mask(t, mod.Filter("d", "date", "review_date", "between", (day, day)))
    assert m.dtype == bool and m.tolist() == (df.review_date == day.isoformat()).tolist()


# Loading parses every column, keeps quoted commas, and counts reviews per product.
def test_table_arrays_from_csv(tmp_path):
    df = table_frame(n=50)
    df.loc[0, ["review_date", "verified", "category"]] = ["1970-01-02", "false", "A, B & C"]
    t = load(tmp_path, df)
    assert t.n_rows == 50
    assert (t.product_id.dtype, t.rating.dtype, t.day.dtype, t.verified.dtype, t.vec_row.dtype) == \
        (np.int32, np.int16, np.int32, np.bool_, np.int64)
    assert t.day[0] == 1 and not t.verified[0] and t.category[0] == "A, B & C"
    want_days = [(dt.date.fromisoformat(d) - EPOCH).days for d in df.review_date]
    assert t.day.tolist() == want_days
    assert t.verified.tolist() == (df.verified == "true").tolist()
    assert t.category.tolist() == df.category.tolist()
    assert t.product_counts.tolist() == np.bincount(df.product_id.astype(int)).tolist()


# Ids must be 0..n-1 in order, verified exactly true or false, every date set.
def test_table_arrays_rejects_bad_files(tmp_path):
    df = table_frame(n=20)
    shuffled = df.iloc[::-1]
    with pytest.raises(ValueError):
        load(tmp_path, shuffled, "shuffled.csv")
    gap = df.copy()
    gap.loc[5, "id"] = "50"
    with pytest.raises(ValueError):
        load(tmp_path, gap, "gap.csv")
    shouty = df.copy()
    shouty.loc[3, "verified"] = "True"
    with pytest.raises(ValueError):
        load(tmp_path, shouty, "shouty.csv")
    undated = df.copy()
    undated.loc[4, "review_date"] = ""
    with pytest.raises(ValueError):
        load(tmp_path, undated, "undated.csv")


# Query qids as the truth script reads them: test reviews, written and ai53.
def qid_lists():
    tests = [f"r{i:05d}" for i in range(9000, 8000, -1)]   # reversed, so sorting matters
    written = [f"q{i:03d}" for i in range(1, 201)]
    ai53 = ["F1", "C2", "F3", "F4", "F5a", "F5b", "F6", "E7", "F8", "F10"]
    return tests, written, ai53


# Group sizes, ai53 first, an even review split, and 500k as 70k's prefix.
def test_part1_queries_groups_and_halves():
    tests, written, ai53 = qid_lists()
    q70 = mod.part1_queries(tests, written, ai53, "70k")
    assert len(q70) == 200 and [x[0] for x in q70 if x[1] == "question"][:10] == ai53
    assert sum(1 for x in q70 if x[1] == "review" and x[2] == "train") == 50
    q500 = mod.part1_queries(tests, written, ai53, "500k")
    assert len(q500) == 100
    assert [x[0] for x in q500 if x[1] == "review"] == [x[0] for x in q70 if x[1] == "review"][:50]


# Reviews come first in qid order; questions are ai53 then 90 distinct written ones.
def test_part1_queries_selection():
    tests, written, ai53 = qid_lists()
    q70 = mod.part1_queries(tests, written, ai53, "70k")
    assert [x[1] for x in q70] == ["review"] * 100 + ["question"] * 100
    assert [x[0] for x in q70[:100]] == sorted(tests)[:100]
    picked = [x[0] for x in q70[110:]]
    assert len(set(picked)) == 90 and set(picked) <= set(written) and picked != written[:90]
    q500 = mod.part1_queries(tests, written, ai53, "500k")
    assert [x[0] for x in q500 if x[1] == "question"] == [x[0] for x in q70 if x[1] == "question"][:50]
    assert q70 == mod.part1_queries(list(tests), list(written), list(ai53), "70k")


# Every group splits exactly in half, at both sizes and with a lowered count.
def test_part1_queries_halves_split_evenly():
    tests, written, ai53 = qid_lists()
    for size, n in (("70k", None), ("500k", None), ("500k", 30), ("70k", 31)):
        rows = mod.part1_queries(tests, written, ai53, size, n)
        n_used = n or mod.PART1_PER_GROUP[size]
        for group in ("review", "question"):
            halves = [x[2] for x in rows if x[1] == group]
            assert len(halves) == n_used
            assert halves.count("train") == n_used // 2 and halves.count("test") == n_used - n_used // 2
    rows = mod.part1_queries(tests, written, ai53, "500k", 30)
    full = mod.part1_queries(tests, written, ai53, "500k")
    assert [x[0] for x in rows] == [x[0] for x in full if x[1] == "review"][:30] + \
        [x[0] for x in full if x[1] == "question"][:30]


# The count may only go down, and short qid lists are errors.
def test_part1_queries_rejects_bad_input():
    tests, written, ai53 = qid_lists()
    for n in (0, 51, -1):
        with pytest.raises(ValueError):
            mod.part1_queries(tests, written, ai53, "500k", n)
    with pytest.raises(ValueError):
        mod.part1_queries(tests, written, ai53, "1m")
    with pytest.raises(ValueError):
        mod.part1_queries(tests[:99], written, ai53, "70k")
    with pytest.raises(ValueError):
        mod.part1_queries(tests, written[:89], ai53, "70k")
    with pytest.raises(ValueError):
        mod.part1_queries(tests, written, ai53[:9], "70k")
    assert len(mod.part1_queries(tests[:50], written, ai53, "500k")) == 100


# Duplicate qids in a list, or a qid in two lists, are errors.
def test_part1_queries_rejects_repeated_qids():
    tests, written, ai53 = qid_lists()
    bad = [
        (tests + tests[:1], written, ai53, "duplicates"),
        (tests, written + written[-1:], ai53, "duplicates"),
        (tests, written, ai53[:9] + ai53[:1], "duplicates"),
        (tests, written[:199] + ["F1"], ai53, "more than one"),
        (tests[:999] + ["q200"], written, ai53, "more than one"),
        (tests[:999] + ["C2"], written, ai53, "more than one"),
    ]
    for t, w, a, why in bad:
        with pytest.raises(ValueError, match=why):
            mod.part1_queries(t, w, a, "70k")


# The frozen seeding: questions from seed part1_questions, halves per size and group.
def test_part1_queries_follow_the_seed_rule():
    tests, written, ai53 = qid_lists()
    order = np.random.default_rng(mod.vc.seed_for("part1_questions")).permutation(200)
    questions = ai53 + [written[i] for i in order[:90]]
    for size, n in (("70k", 100), ("500k", 50), ("500k", 37)):
        rows = mod.part1_queries(tests, written, ai53, size, n)
        for group, qids in (("review", sorted(tests)[:n]), ("question", questions[:n])):
            perm = np.random.default_rng(mod.vc.seed_for("part1_halves", size, group)).permutation(n)
            train = set(perm[: n // 2].tolist())
            want = [(q, group, "train" if i in train else "test") for i, q in enumerate(qids)]
            assert [x for x in rows if x[1] == group] == want
