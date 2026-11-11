"""Tests for bench/vector_check/prepare_vector_data.py. Everything runs on small synthetic files."""
from __future__ import annotations

import gzip
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC_PATH = Path(__file__).resolve().parents[1] / "prepare_vector_data.py"
spec = importlib.util.spec_from_file_location("vc_prepare_vector_data", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["vc_prepare_vector_data"] = mod
spec.loader.exec_module(mod)

REVIEW_COLUMNS = ["review_id", "rating", "title", "text", "parent_asin", "timestamp",
                  "helpful_vote", "verified_purchase", "user_id", "text_key"]
PRODUCT_COLUMNS = ["parent_asin", "title", "store", "price", "main_category", "categories",
                   "average_rating", "rating_number"]
CASES = '["Cell Phones & Accessories", "Cases, Holsters & Sleeves", "Basic Cases"]'
CHARGERS = '["Cell Phones & Accessories", "Chargers & Power Adapters"]'


def key(title, text):
    return mod.vc.text_key(title, text)


# A reviews frame as read_csv(dtype=str) gives it; rows are (title, text, asin, ms, verified, rating).
def reviews_frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["title", "text", "parent_asin", "timestamp", "verified_purchase", "rating"])
    df["review_id"] = [str(100 + i) for i in range(len(df))]
    df["helpful_vote"] = "0"
    df["user_id"] = "U"
    df["text_key"] = [key(t, x) for t, x in zip(df.title, df.text)]
    return df[REVIEW_COLUMNS].astype(str)


def products_frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["parent_asin", "categories"])
    for col in ["title", "store", "price", "main_category", "average_rating", "rating_number"]:
        df[col] = ""
    return df[PRODUCT_COLUMNS].astype(str)


def small_reviews() -> pd.DataFrame:
    return reviews_frame([
        ("Great case", "fits well", "P1", "1577753802523", "True", "5"),
        ("Bad", "broke fast", "P0", "0", "False", "1"),
        ("great  CASE", "Fits\nwell", "P1", "86400000", "True", "4"),   # same key as row 0
        ("NA", "meh", "P1", "86399999", "False", "3"),
    ])


def small_products() -> pd.DataFrame:
    return products_frame([("P0", CHARGERS), ("P1", CASES), ("P2", "[]")])


def test_category_of():
    assert mod.category_of('["Cell Phones & Accessories", "Cases, Holsters & Sleeves", "Basic Cases"]') == "Cases, Holsters & Sleeves"
    assert mod.category_of("[]") == "" and mod.category_of("") == ""
    assert mod.category_of('["Cell Phones & Accessories"]') == ""


def test_review_date_of():
    assert mod.review_date_of(1577753802523) == "2019-12-31"
    assert mod.review_date_of(0) == "1970-01-01"
    assert mod.review_date_of(86_399_999) == "1970-01-01"     # last ms of the day, UTC
    assert mod.review_date_of(86_400_000) == "1970-01-02"
    assert type(mod.review_date_of(0)) is str


def test_build_table_shares_vectors_for_duplicate_texts():
    reviews = small_reviews()
    table, products, texts = mod.build_table(reviews, small_products(), None)

    assert table.columns.tolist() == ["id", "product_id", "rating", "review_date", "verified",
                                      "category", "vec_row", "text_key"]
    assert table.id.tolist() == [0, 1, 2, 3]
    assert table.vec_row.tolist() == [0, 1, 0, 2]           # row 2 shares row 0's vector
    assert table.product_id.tolist() == [1, 0, 1, 1]
    assert table.rating.tolist() == [5, 1, 4, 3]
    assert table.verified.tolist() == ["true", "false", "true", "false"]
    assert table.review_date.tolist() == ["2019-12-31", "1970-01-01", "1970-01-02", "1970-01-01"]
    cases, chargers = "Cases, Holsters & Sleeves", "Chargers & Power Adapters"
    assert table.category.tolist() == [cases, chargers, cases, cases]
    assert table.text_key.tolist() == reviews.text_key.tolist()

    assert products.columns.tolist() == ["product_id", "parent_asin", "n_reviews", "category"]
    assert products.product_id.tolist() == [0, 1, 2]
    assert products.parent_asin.tolist() == ["P0", "P1", "P2"]
    assert products.n_reviews.tolist() == [1, 3, 0]         # a product with no reviews is still listed
    assert products.category.tolist() == [chargers, cases, ""]
    # product ids index the products table
    assert products.parent_asin.to_numpy()[table.product_id].tolist() == reviews.parent_asin.tolist()

    assert texts.columns.tolist() == ["vec_row", "text_key", "input"]
    assert texts.vec_row.tolist() == [0, 1, 2]
    assert texts.text_key.tolist() == [reviews.text_key[0], reviews.text_key[1], reviews.text_key[3]]
    # the first row with a key gives its input; "NA" stays "NA"
    assert texts.input.tolist() == ["Title: Great case\nReview: fits well", "Title: Bad\nReview: broke fast",
                                    "Title: NA\nReview: meh"]


def test_build_table_uses_given_vec_rows():
    reviews = small_reviews()
    keys = reviews.text_key
    table, _, texts = mod.build_table(reviews, small_products(), {keys[0]: 7, keys[1]: 3, keys[3]: 5, "extra": 0})
    assert table.vec_row.tolist() == [7, 3, 7, 5]
    assert texts.vec_row.tolist() == [3, 5, 7]
    assert texts.input.tolist() == ["Title: Bad\nReview: broke fast", "Title: NA\nReview: meh",
                                    "Title: Great case\nReview: fits well"]


def test_build_table_missing_key_raises():
    reviews = small_reviews()
    keys = reviews.text_key
    with pytest.raises(ValueError, match="missing"):
        mod.build_table(reviews, small_products(), {keys[0]: 0, keys[1]: 1})
    with pytest.raises(ValueError, match="share"):          # two texts can't share one vector
        mod.build_table(reviews, small_products(), {keys[0]: 0, keys[1]: 1, keys[3]: 1})


def test_build_table_refuses_bad_rows():
    products = small_products()
    unknown = small_reviews()
    unknown.loc[3, "parent_asin"] = "P9"
    with pytest.raises(ValueError, match="P9"):
        mod.build_table(unknown, products, None)
    odd_flag = small_reviews()
    odd_flag.loc[1, "verified_purchase"] = "yes"
    with pytest.raises(ValueError, match="verified_purchase"):
        mod.build_table(odd_flag, products, None)
    half_star = small_reviews()
    half_star.loc[0, "rating"] = "4.5"
    with pytest.raises(ValueError, match="rating"):
        mod.build_table(half_star, products, None)
    with pytest.raises(ValueError, match="P1"):
        mod.build_table(small_reviews(), products_frame([("P0", "[]"), ("P1", "[]"), ("P1", "[]")]), None)


def test_reservoir_counts_lines_like_python():
    rng = np.random.default_rng(0)
    lines, total = mod.reservoir_lines([b"a\nb", b"\n\nc"], 10, rng)
    assert total == 4                                        # blank line counted, last line has no newline
    assert lines == [(0, b"a"), (1, b"b"), (2, b""), (3, b"c")]
    assert mod.reservoir_lines([b"a\n"], 10, rng) == ([(0, b"a")], 1)
    assert mod.reservoir_lines([b"", b""], 10, rng) == ([], 0)


def test_reservoir_is_uniform_and_ignores_chunking():
    data = b"".join(f"{i}\n".encode() for i in range(500))
    counts = np.zeros(500)
    for s in range(2000):
        lines, total = mod.reservoir_lines([data], 50, np.random.default_rng(s))
        assert total == 500 and len(lines) == 50
        assert all(int(raw) == n for n, raw in lines)
        assert [n for n, _ in lines] == sorted(n for n, _ in lines)
        counts[[n for n, _ in lines]] += 1
    freq = counts / 2000
    assert np.abs(freq - 0.1).max() < 0.04                   # 0.04 is about 6 standard deviations
    assert abs(freq[:100].mean() - freq[-100:].mean()) < 0.01
    pieces = [data[i:i + 7] for i in range(0, len(data), 7)]
    assert mod.reservoir_lines(pieces, 50, np.random.default_rng(3)) == \
        mod.reservoir_lines([data], 50, np.random.default_rng(3))


def words(tag, n):
    return " ".join(f"w{tag}x{j}" for j in range(n))


# 500 raw lines: 300 good, 40 duplicates, 40 short, 40 excluded,
# 40 without a product and 40 bad ones.
def write_raw(path: Path, last_newline: bool = False):
    good = [{"rating": 5.0, "title": f"T{i}", "text": words(i, 12), "parent_asin": f"G{i}"} for i in range(300)]
    dups = [{"title": f"t{i}", "text": "  " + words(i, 12).upper() + " ", "parent_asin": f"D{i}"} for i in range(40)]
    short = [{"title": "x", "text": words(1000 + i, 9), "parent_asin": "S"} for i in range(39)]
    short.append({"title": "long title words " * 5, "text": None, "parent_asin": "S"})
    excluded = [{"title": "E", "text": words(2000 + i, 12), "parent_asin": "E"} for i in range(40)]
    orphan = [{"title": "O", "text": words(3000 + i, 12)} for i in range(14)]
    orphan += [{"title": "O", "text": words(3100 + i, 12), "parent_asin": None} for i in range(13)]
    orphan += [{"title": "O", "text": words(3200 + i, 12), "parent_asin": 123} for i in range(13)]
    lines = [json.dumps(r).encode() for r in good + dups + short + excluded + orphan]
    lines += [b""] * 10 + [b'{"title": "cut off'] * 10 + [b"\xff\xfe" + lines[0]] * 10 + [b"[1, 2]"] * 10
    random.Random(0).shuffle(lines)
    assert len(lines) == 500
    with gzip.open(path, "wb") as f:
        f.write(b"\n".join(lines) + (b"\n" if last_newline else b""))
    return {
        "lines": lines,
        "good": {key(r["title"], r["text"]) for r in good},
        "excluded": {key(r["title"], r["text"]) for r in excluded},
        "bad_keys": {key(r["title"], r["text"] or "") for r in short + orphan},
    }


def test_sample_raw_reviews(tmp_path, capsys):
    path = tmp_path / f"raw_{os.getpid()}.jsonl.gz"
    info = write_raw(path)
    frame = mod.sample_raw_reviews(path, n=300, pool=1000, exclude=info["excluded"], seed=1)

    assert frame.columns.tolist() == ["title", "text", "text_key", "parent_asin"]
    assert len(frame) == 300 and frame.text_key.is_unique
    # short, excluded, product-less, bad and duplicate lines never come back
    assert set(frame.text_key) == info["good"]
    assert not set(frame.text_key) & (info["excluded"] | info["bad_keys"])
    assert frame.text_key.tolist() == [key(t, x) for t, x in zip(frame.title, frame.text)]
    # of a duplicate pair, the earlier line in the file is kept
    order = {json.loads(raw).get("parent_asin"): n for n, raw in enumerate(info["lines"])
             if raw.startswith(b"{") and raw.endswith(b"}")}
    by_key = dict(zip(frame.text_key, frame.parent_asin))
    for i in range(40):
        want = f"G{i}" if order[f"G{i}"] < order[f"D{i}"] else f"D{i}"
        assert by_key[key(f"T{i}", words(i, 12))] == want
    assert "100%" in capsys.readouterr().out

    pd.testing.assert_frame_equal(frame, mod.sample_raw_reviews(path, 300, 1000, info["excluded"], seed=1))
    other = mod.sample_raw_reviews(path, 300, 1000, info["excluded"], seed=2)
    assert set(other.text_key) == set(frame.text_key) and other.text_key.tolist() != frame.text_key.tolist()

    with pytest.raises(ValueError, match="300"):
        mod.sample_raw_reviews(path, n=301, pool=1000, exclude=info["excluded"])
    with pytest.raises(ValueError, match="pool"):
        mod.sample_raw_reviews(path, n=50, pool=40, exclude=set())


def test_sample_raw_reviews_small_pool(tmp_path):
    path = tmp_path / f"raw_{os.getpid()}.jsonl.gz"
    info = write_raw(path, last_newline=True)
    frame = mod.sample_raw_reviews(path, n=20, pool=100, exclude=info["excluded"])
    assert len(frame) == 20 and frame.text_key.is_unique
    assert set(frame.text_key) <= info["good"]


def test_assign_splits_counts():
    s = mod.assign_splits(10000)
    assert (s.count("train"), s.count("validation"), s.count("test")) == (8000, 1000, 1000)
    assert s == mod.assign_splits(10000)
    assert s[:8000] != ["train"] * 8000                      # shuffled, not in blocks
    small = mod.assign_splits(10, counts=(8, 1, 1))
    assert sorted(small) == ["test"] + ["train"] * 8 + ["validation"]
    with pytest.raises(ValueError):
        mod.assign_splits(10)
    with pytest.raises(ValueError):
        mod.assign_splits(10000, counts=(8000, 1000, 999))


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def read_back(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


# 120 reviews over 2 products; row 5 repeats row 1's text.
# Rows 1 and 2 hold tricky text.
def fake_reviews(prefix: str) -> pd.DataFrame:
    rows = []
    for i in range(120):
        title, text = f"{prefix} title {i}", f"{prefix} text {i}"
        if i == 1:
            title, text = "NA", 'He said "great", then\r\nleft'
        if i == 2:
            title, text = "", "a, b\nc"
        if i == 5:
            title, text = "na", 'He  said "GREAT", then left'
        rows.append((title, text, f"A{i % 2}", str(1577753802523 + i), "True" if i % 3 else "False", str(1 + i % 5)))
    return reviews_frame(rows)


def make_ai(root: Path, size: str) -> pd.DataFrame:
    reviews = fake_reviews(size)
    data = root / "data" if size == "70k" else root / "data" / "slice"
    write_csv(reviews, data / "reviews.csv")
    write_csv(products_frame([("A0", CASES), ("A1", CHARGERS)]), data / "products.csv")
    if size == "70k":
        distinct = list(dict.fromkeys(reviews.text_key))
        rows = list(range(len(distinct)))[::-1]              # rows on purpose not in file order
        write_csv(pd.DataFrame({"row": rows, "text_key": distinct}), root / "embeddings" / "test_keys.csv")
    return reviews


def test_main_table_70k(tmp_path, capsys):
    ai, results = tmp_path / f"ai_{os.getpid()}", tmp_path / f"res_{os.getpid()}"
    reviews = make_ai(ai, "70k")
    assert mod.main(["table", "--size", "70k", "--results", str(results), "--ai", str(ai)]) == 0
    data = results / "data"
    table = read_back(data / "table_70k.csv")
    assert table.columns.tolist() == ["id", "product_id", "rating", "review_date", "verified",
                                      "category", "vec_row", "text_key"]
    assert len(table) == 120 and table.id.tolist() == [str(i) for i in range(120)]
    assert table.vec_row[5] == table.vec_row[1] == "117"     # 119 distinct keys, rows reversed
    assert set(table.verified) == {"true", "false"}
    assert table.category[0] == "Cases, Holsters & Sleeves"
    products = read_back(data / "products_70k.csv")
    assert products.n_reviews.tolist() == ["60", "60"]
    check = read_back(data / "check_texts_70k.csv")
    assert check.columns.tolist() == ["vec_row", "input"]
    assert check.vec_row.tolist() == [str(i) for i in range(100)]
    by_row = dict(zip(check.vec_row, check.input))
    assert by_row["0"] == "Title: 70k title 119\nReview: 70k text 119"   # test_keys row 0 is the last key
    assert by_row["99"] == "Title: 70k title 20\nReview: 70k text 20"
    assert table.text_key.tolist() == reviews.text_key.tolist()
    assert not (data / "slice_texts.csv").exists()
    assert "120 rows" in capsys.readouterr().out


def test_main_table_500k(tmp_path):
    ai, results = tmp_path / f"ai_{os.getpid()}", tmp_path / f"res_{os.getpid()}"
    reviews = make_ai(ai, "500k")
    assert mod.main(["table", "--size", "500k", "--results", str(results), "--ai", str(ai)]) == 0
    data = results / "data"
    table = read_back(data / "table_500k.csv")
    assert len(table) == 120
    assert table.vec_row.tolist()[:7] == ["0", "1", "2", "3", "4", "1", "5"]
    texts = read_back(data / "slice_texts.csv")
    assert texts.columns.tolist() == ["vec_row", "text_key", "input"]
    assert texts.vec_row.tolist() == [str(i) for i in range(119)]
    assert texts.input[1] == 'Title: NA\nReview: He said "great", then\r\nleft'
    assert texts.input[2] == "Title: \nReview: a, b\nc"
    assert texts.text_key.tolist() == list(dict.fromkeys(reviews.text_key))
    assert not (data / "check_texts_70k.csv").exists()


def test_text_key_column_checked(tmp_path, capsys):
    ai, results = tmp_path / f"ai_{os.getpid()}", tmp_path / f"res_{os.getpid()}"
    reviews = make_ai(ai, "70k")
    reviews.loc[3, "text_key"] = "0" * 40
    write_csv(reviews, ai / "data" / "reviews.csv")
    assert mod.main(["table", "--size", "70k", "--results", str(results), "--ai", str(ai)]) == 1
    err = capsys.readouterr().err
    assert str(ai / "data" / "reviews.csv") in err and "text_key" in err
    assert not (results / "data" / "table_70k.csv").exists()


def test_main_table_70k_needs_vec_rows_0_to_99(tmp_path, capsys):
    ai, results = tmp_path / f"ai_{os.getpid()}", tmp_path / f"res_{os.getpid()}"
    make_ai(ai, "70k")
    keys = read_back(ai / "embeddings" / "test_keys.csv")
    keys["row"] = (keys["row"].astype(int) + 1).astype(str)   # no key gets row 0
    write_csv(keys, ai / "embeddings" / "test_keys.csv")
    assert mod.main(["table", "--size", "70k", "--results", str(results), "--ai", str(ai)]) == 1
    assert "check_texts_70k.csv" in capsys.readouterr().err
    assert not (results / "data").exists()                    # nothing written


def test_main_table_missing_input(tmp_path, capsys):
    ai, results = tmp_path / f"ai_{os.getpid()}", tmp_path / f"res_{os.getpid()}"
    make_ai(ai, "70k")
    (ai / "embeddings" / "test_keys.csv").unlink()
    assert mod.main(["table", "--size", "70k", "--results", str(results), "--ai", str(ai)]) == 1
    assert "test_keys.csv" in capsys.readouterr().err


def test_main_refuses_to_write_into_ai(tmp_path, capsys):
    ai = tmp_path / f"ai_{os.getpid()}"
    make_ai(ai, "500k")
    assert mod.main(["table", "--size", "500k", "--results", str(ai / "out"), "--ai", str(ai)]) == 1
    assert str(ai) in capsys.readouterr().err
    assert not (ai / "out").exists()


def test_main_queries(tmp_path, capsys):
    ai, results = tmp_path / f"ai_{os.getpid()}", tmp_path / f"res_{os.getpid()}"
    info = write_raw_into(ai)
    assert mod.main(["queries", "--n", "10", "--pool", "1000", "--results", str(results), "--ai", str(ai)]) == 0
    out = read_back(results / "data" / "query_reviews.csv")
    assert out.columns.tolist() == ["qid", "split", "text_key", "input"]
    assert out.qid.tolist() == [f"r{i:05d}" for i in range(1, 11)]
    assert sorted(out.split) == ["test"] + ["train"] * 8 + ["validation"]
    assert out.text_key.is_unique and set(out.text_key) <= info["good"] | info["excluded"]
    assert not set(out.text_key) & info["table_keys"]       # keys in either table are left out
    for k, text in zip(out.text_key, out.input):
        title, review = text.removeprefix("Title: ").split("\nReview: ")
        assert key(title, review) == k
    assert "%" in capsys.readouterr().out


@pytest.mark.parametrize("table", ["data/reviews.csv", "data/slice/reviews.csv"])
def test_queries_checks_text_key_columns(tmp_path, capsys, table):
    ai, results = tmp_path / f"ai_{os.getpid()}", tmp_path / f"res_{os.getpid()}"
    write_raw_into(ai)
    reviews = read_back(ai / table)
    reviews.loc[3, "text_key"] = "0" * 40
    write_csv(reviews, ai / table)
    assert mod.main(["queries", "--n", "10", "--pool", "1000", "--results", str(results), "--ai", str(ai)]) == 1
    err = capsys.readouterr().err
    assert str(ai / table) in err and "text_key" in err
    assert not (results / "data" / "query_reviews.csv").exists()


# A raw file plus 70K and slice tables that hold some of its good keys.
def write_raw_into(ai: Path):
    raw = ai / "raw" / "Cell_Phones_and_Accessories.jsonl.gz"
    raw.parent.mkdir(parents=True)
    info = write_raw(raw)
    good = sorted(info["good"])
    rows70 = reviews_frame([(f"T{i}", words(i, 12), "A0", "0", "True", "5") for i in range(0, 150)])
    rows_slice = reviews_frame([(f"T{i}", words(i, 12), "A0", "0", "True", "5") for i in range(150, 280)])
    write_csv(rows70, ai / "data" / "reviews.csv")
    write_csv(rows_slice, ai / "data" / "slice" / "reviews.csv")
    info["table_keys"] = set(rows70.text_key) | set(rows_slice.text_key)
    assert info["table_keys"] <= set(good)
    return info
