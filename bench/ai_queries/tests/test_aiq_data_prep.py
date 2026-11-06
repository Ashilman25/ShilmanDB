import gzip
import json
import math

import pandas as pd

from bench.ai_queries import common, data_prep, questions


def write_jsonl(path, rows, gz=False, extra_lines=()):
    lines = [json.dumps(r) for r in rows] + list(extra_lines)
    data = "\n".join(lines) + "\n"
    if gz:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(data)
    else:
        path.write_text(data, encoding="utf-8")


def review(i, asin, text=None, title="t"):
    return {"rating": float(i % 5 + 1), "title": title, "text": text if text is not None else f"text {i}",
            "parent_asin": asin, "user_id": f"u{i}", "timestamp": 1_600_000_000_000 + i,
            "helpful_vote": i % 3, "verified_purchase": i % 2 == 0}


def meta(asin, store="Acme", price="$9.99"):
    return {"parent_asin": asin, "title": f"Product {asin}", "store": store, "price": price,
            "main_category": "Cell Phones", "categories": ["Cell Phones", "Cases"],
            "average_rating": 4.2, "rating_number": 100}


def make_files(tmp_path, gz=False, counts=None):
    counts = counts or {"a": 100, "b": 60, "c": 49, "d": 400, "e": 80}
    rows, i = [], 0
    for asin, n in counts.items():
        for _ in range(n):
            rows.append(review(i, asin))
            i += 1
    suffix = ".jsonl.gz" if gz else ".jsonl"
    rp, mp = tmp_path / f"r{suffix}", tmp_path / f"m{suffix}"
    write_jsonl(rp, rows, gz, extra_lines=["not json"])
    write_jsonl(mp, [meta(a, store="" if a == "b" else "Acme") for a in counts if a != "e"], gz)
    return rp, mp


def test_parse_price():
    assert data_prep.parse_price("$12.99") == 12.99
    assert data_prep.parse_price("$1,299.00") == 1299.0
    assert data_prep.parse_price(8.5) == 8.5
    for bad in ("None", "", None, "from $5", "$5 - $9"):
        assert math.isnan(data_prep.parse_price(bad))


def test_is_case():
    assert data_prep.is_case(["Cell Phones & Accessories", "Cases, Holsters & Sleeves", "Basic Cases"])
    assert data_prep.is_case('["Cases"]')
    assert not data_prep.is_case(["Chargers & Power Adapters"])
    assert not data_prep.is_case([])


def test_text_key_normalizes():
    assert common.text_key("Great!", "Works  fine") == common.text_key("great!", "works fine")
    assert common.text_key("Great!", "Works  fine") != common.text_key("great", "works fine")


def test_ai_input_handles_empty_fields():
    assert common.ai_input("", "X") == "Title: \nReview: X"
    assert common.ai_input(None, None) == "Title: \nReview: "


def test_select_products_floor_cap_target():
    counts = {"a": 100, "b": 60, "c": 49, "d": 400}
    chosen = data_prep.select_products(counts, set(counts), 50, 300, 400, 1)
    assert "c" not in chosen
    total = sum(min(counts[p], 300) for p in chosen)
    assert total >= 400
    assert total - min(counts[chosen[-1]], 300) < 400
    assert chosen == data_prep.select_products(counts, set(counts), 50, 300, 400, 1)


def test_products_without_meta_are_not_eligible():
    counts = {"big": 1000, "x": 60, "y": 70}
    for seed in range(5):
        assert "big" not in data_prep.select_products(counts, {"x", "y"}, 50, 300, 100, seed)


def test_build_test_table(tmp_path):
    rp, mp = make_files(tmp_path)
    out = tmp_path / "out"
    design = data_prep.build_test_table(rp, mp, out, floor=50, cap=50, target_rows=150, seed=1)
    rev = pd.read_csv(out / "reviews.csv", dtype=str, keep_default_na=False)
    prod = pd.read_csv(out / "products.csv", dtype=str, keep_default_na=False)
    assert list(rev.columns) == ["review_id", "rating", "title", "text", "parent_asin", "timestamp",
                                 "helpful_vote", "verified_purchase", "user_id", "text_key"]
    assert list(prod.columns) == ["parent_asin", "title", "store", "price", "main_category",
                                  "categories", "average_rating", "rating_number"]
    assert (rev.groupby("parent_asin").size() <= 50).all()
    assert set(prod.parent_asin) == set(rev.parent_asin)
    assert "c" not in set(prod.parent_asin) and "e" not in set(prod.parent_asin)
    assert len(rev) == design["rows"] >= 150
    # review_id is the source line number: reviews of "a" are lines 0..99
    a = rev[rev.parent_asin == "a"]
    assert all(rev.loc[a.index, "text"] == "text " + a.review_id)
    assert json.loads(prod.categories.iloc[0]) == ["Cell Phones", "Cases"]
    for key in ("floor", "cap", "target_rows", "seed", "rows", "products", "reviews_file", "meta_file",
                "skipped_lines"):
        assert key in design
    assert design == json.loads((out / "design.json").read_text())
    assert design["skipped_lines"] == 1
    # a missing store is the empty string
    if "b" in set(prod.parent_asin):
        assert prod.loc[prod.parent_asin == "b", "store"].iloc[0] == ""

    same = tmp_path / "same"
    data_prep.build_test_table(rp, mp, same, floor=50, cap=50, target_rows=150, seed=1)
    assert (same / "reviews.csv").read_bytes() == (out / "reviews.csv").read_bytes()
    other = tmp_path / "other"
    data_prep.build_test_table(rp, mp, other, floor=50, cap=50, target_rows=150, seed=2)
    assert (other / "reviews.csv").read_bytes() != (out / "reviews.csv").read_bytes()

    gzdir = tmp_path / "gz"
    gzdir.mkdir()
    grp, gmp = make_files(gzdir, gz=True)
    gout = tmp_path / "gout"
    data_prep.build_test_table(grp, gmp, gout, floor=50, cap=50, target_rows=150, seed=1)
    assert (gout / "reviews.csv").read_bytes() == (out / "reviews.csv").read_bytes()
    assert (gout / "products.csv").read_bytes() == (out / "products.csv").read_bytes()


def test_duplicate_texts_share_key_and_na_text_survives(tmp_path):
    rows = [review(i, "a", text="Same  text" if i < 3 else "NA" if i == 3 else f"u{i}") for i in range(60)]
    rp, mp = tmp_path / "r.jsonl", tmp_path / "m.jsonl"
    write_jsonl(rp, rows)
    write_jsonl(mp, [meta("a")])
    out = tmp_path / "out"
    data_prep.build_test_table(rp, mp, out, floor=50, cap=300, target_rows=50, seed=1)
    rev = pd.read_csv(out / "reviews.csv", dtype=str, keep_default_na=False)
    same = rev[rev.text == "Same  text"]
    assert len(same) == 3 and same.text_key.nunique() == 1
    assert rev.loc[rev.review_id == "3", "text"].iloc[0] == "NA"


def test_exclude_from(tmp_path):
    rp, mp = make_files(tmp_path, counts={"a": 200, "b": 200})
    excluded = frozenset(range(0, 400, 2))
    out = tmp_path / "out"
    design = data_prep.build_test_table(rp, mp, out, floor=50, cap=300, target_rows=150, seed=1,
                                        exclude_ids=excluded)
    rev = pd.read_csv(out / "reviews.csv", dtype=str, keep_default_na=False)
    assert not (set(rev.review_id.astype(int)) & excluded)
    assert design["rows"] >= 150


def test_slice_info(tmp_path):
    rows = [review(i, "a", text=f"unique {i}") for i in range(950)]
    rows += [review(1000 + i, "a", text="copy") for i in range(50)]
    rp, mp = tmp_path / "r.jsonl", tmp_path / "m.jsonl"
    write_jsonl(rp, rows)
    write_jsonl(mp, [meta("a")])
    out = tmp_path / "out"
    info = data_prep.build_slice(rp, mp, out, n=200, seed=1)
    assert info["rows"] == 200
    rev = pd.read_csv(out / "slice" / "reviews.csv", dtype=str, keep_default_na=False)
    assert len(rev) == 200
    seen, dups = set(), 0
    for k in rev.text_key:
        dups += k in seen
        seen.add(k)
    assert info["duplicate_share"] == dups / 200
    assert json.loads((out / "slice" / "slice_info.json").read_text()) == info
    assert (out / "slice" / "products.csv").exists()
    again = tmp_path / "again"
    data_prep.build_slice(rp, mp, again, n=200, seed=1)
    assert (again / "slice" / "reviews.csv").read_bytes() == (out / "slice" / "reviews.csv").read_bytes()


def test_questions_verbatim():
    assert list(questions.QUESTIONS) == ["F1", "C2", "F3", "F4", "F5a", "F5b", "F6", "E7", "F8", "F10"]
    assert questions.QUESTIONS["F8"].wording == (
        "Does the reviewer describe a safety hazard, such as overheating, melting, smoke, sparks, "
        "fire or a swelling battery?")
    assert questions.C2_CLASSES[0] == "shipping or delivery"
    assert len(questions.C2_CLASSES) == 6
    assert questions.QUESTIONS["F1"].options == ("yes", "no")


def test_read_tables_round_trip(tmp_path):
    reviews = pd.DataFrame([
        {"review_id": 0, "rating": 5, "title": "NA", "text": "NA", "parent_asin": "a", "timestamp": 10,
         "helpful_vote": 0, "verified_purchase": True, "user_id": "u", "text_key": "k0"},
        {"review_id": 1, "rating": 1, "title": "", "text": "bad", "parent_asin": "b", "timestamp": 20,
         "helpful_vote": 3, "verified_purchase": False, "user_id": "v", "text_key": "k1"},
    ], columns=data_prep.REVIEW_COLUMNS)
    products = pd.DataFrame([
        {"parent_asin": "a", "title": "None", "store": "", "price": 12.5, "main_category": "x",
         "categories": json.dumps(["Cases"]), "average_rating": 4.5, "rating_number": 7},
        {"parent_asin": "b", "title": "t", "store": "S", "price": "", "main_category": "",
         "categories": "[]", "average_rating": 3.0, "rating_number": 2},
    ], columns=data_prep.PRODUCT_COLUMNS)
    reviews.to_csv(tmp_path / "reviews.csv", index=False, lineterminator="\n")
    products.to_csv(tmp_path / "products.csv", index=False, lineterminator="\n")
    rev, prod = data_prep.read_tables(tmp_path)
    assert rev.title[0] == "NA" and rev.text[0] == "NA" and rev.title[1] == ""
    assert rev.rating.dtype.kind == "i" and rev.timestamp.dtype.kind == "i"
    assert rev.review_id.dtype.kind == "i" and list(rev.review_id) == [0, 1]
    assert list(rev.verified_purchase) == [True, False] and rev.verified_purchase.dtype == bool
    assert prod.store[0] == "" and prod.title[0] == "None"
    assert prod.price[0] == 12.5 and math.isnan(prod.price[1])
    assert prod.rating_number.dtype.kind == "i" and prod.average_rating.dtype.kind == "f"
    assert prod.categories[0] == '["Cases"]'


def test_build_warns_below_the_row_floor(tmp_path, capsys):
    rp, mp = make_files(tmp_path)
    args = ["build", "--reviews", str(rp), "--meta", str(mp), "--root", str(tmp_path / "ai"), "--cap", "50"]
    data_prep.main([*args, "--rows", "150"])
    err = capsys.readouterr().err
    assert "warning" in err and "20,000" in err
    assert (tmp_path / "ai" / "data" / "reviews.csv").exists()  # it still builds
    data_prep.main([*args, "--rows", "20000"])
    assert "warning" not in capsys.readouterr().err
