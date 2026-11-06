import json
import math

import numpy as np
import pandas as pd
import pytest

from bench.ai_queries import labels, queries
from bench.ai_queries.common import text_key
from bench.ai_queries.questions import C2_CLASSES, QUESTIONS

CASE_CATS = json.dumps(["Cell Phones", "Cases, Holsters & Sleeves"])
OTHER_CATS = json.dumps(["Cell Phones", "Chargers"])
NAN = math.nan

# asin -> (title, store, price, categories); PX (every 12th review) has no product row
PRODUCTS = {
    "P0": ("Case A", "Acme", 19.99, CASE_CATS),
    "P1": ("Case B", "Acme", 25.0, CASE_CATS),
    "P2": ("Case C", "Bolt", NAN, CASE_CATS),
    "P3": ("Cable", "", 10.0, OTHER_CATS),
    "P4": ("Case D", "Bolt", 5.0, CASE_CATS),
    "P5": ("Charger", "Cora", 30.0, OTHER_CATS),
    "P6": ("Case E", "Dyne", 19.5, CASE_CATS),
    "P7": ("Battery", "Cora", 12.0, "[]"),
}
E7_ANSWERS = {0: "none", 5: "iPhone 13", 10: "iphone 13.", 15: "iphone13", 20: "Pixel 7", 25: "pixel  7",
              30: "N/A", 35: "Galaxy S21", 40: "Moto G", 45: "Nokia 3310", 50: "LG V30", 55: "Pixel 6"}
ALIASES = {"iphone13": "iphone 13"}


def make_tables(n=60):
    reviews = pd.DataFrame({
        "review_id": range(n),
        "rating": [[1, 2, 3, 4, 5][i % 5] for i in range(n)],
        "title": [f"title {i}" for i in range(n)],
        "text": [f"review text number {i}" for i in range(n)],
        "parent_asin": ["PX" if i % 12 == 11 else f"P{i % 8}" for i in range(n)],
        "timestamp": [1000 + (i // 2) * 10 for i in range(n)],  # pairs share a timestamp
        "helpful_vote": 0,
        "verified_purchase": [i % 3 != 0 for i in range(n)],
        "user_id": "u",
    })
    reviews["text_key"] = [text_key(t, x) for t, x in zip(reviews.title, reviews.text)]
    products = pd.DataFrame([
        {"parent_asin": a, "title": t, "store": s, "price": p, "main_category": "", "categories": c,
         "average_rating": 4.0, "rating_number": 10} for a, (t, s, p, c) in PRODUCTS.items()])
    return reviews, products


def label_rules(i):
    verified = i % 3 != 0
    return {
        "F1": "yes" if i % 4 == 0 else "no",
        "C2": C2_CLASSES[i % 6],
        "F3": "yes" if i % 3 == 1 or i % 10 == 4 else "no",
        "F4": "yes" if (verified and (i % 2 == 0 or i % 7 == 0)) or i % 12 == 0 else "no",
        "F5a": "yes" if i % 3 == 0 else "no",
        "F5b": "yes" if i % 2 == 0 else "no",
        "F6": "yes" if i % 3 != 2 else "no",
        "E7": E7_ANSWERS.get(i, "none"),
        "F8": "yes" if i % 10 == 3 else "no",
        "F10": "yes" if i % 4 == 1 else "no",
    }


def write_tier(tmp_path, reviews, rules=label_rules, tier="big"):
    tier_dir = tmp_path / tier
    for qid, question in QUESTIONS.items():
        rows = []
        for i, key in enumerate(reviews["text_key"]):
            answer = rules(i)[qid]
            row = {"text_key": key, "answer": answer, "in_tokens": 100, "out_tokens": 1, "latency_ms": 5.0}
            for j in range(len(question.options)):
                row[f"p{j}"] = 1.0 / len(question.options)
            rows.append(row)
        labels.append_rows(tier_dir / f"{qid}.csv", rows, labels.label_columns(question))
    return labels.load_tier(tier_dir, tier)


@pytest.fixture
def tables():
    return make_tables()


@pytest.fixture
def big(tmp_path, tables):
    return write_tier(tmp_path, tables[0])


def ids(df):
    return df["review_id"].tolist()


def test_scope_filters_and_joins(tables):
    reviews, products = tables
    assert ids(queries.scope(1, reviews, products)) == [i for i in range(60) if i % 5 in (0, 1)]
    # P1 costs $25 and P2 has no price; P0 at $19.99 stays. Reviews of PX have no product.
    q6 = queries.scope(6, reviews, products)
    assert set(q6["group"]) == {"Case A", "Case D", "Case E"}
    assert ids(q6) == [i for i in range(60) if i % 8 in (0, 4, 6) and i % 12 != 11]
    assert ids(queries.scope(3, reviews, products)) == [i for i in range(60) if i % 12 != 11]
    q3 = queries.scope(3, reviews, products)
    assert q3["group"].iloc[3] == ""  # review 3 sits on P3, which has no store
    # t75 is the 45th timestamp (1220); reviews 44 and 45 both carry it and both stay
    q9 = queries.scope(9, reviews, products)
    assert queries.newest_cutoff(reviews) == 1220
    assert ids(q9) == [i for i in range(44, 60) if i % 12 != 11]
    assert list(q9.columns) == ["review_id", "text_key", "rating", "timestamp", "group"]


def test_naive_rows(tables):
    reviews, products = tables
    got = {no: queries.naive_rows(no, reviews, products) for no in range(1, 11)}
    assert got == {1: 24, 2: 36, 3: 60, 4: 40, 5: 60, 6: 60, 7: 12, 8: 60, 9: 16, 10: 60}


def test_naive_cost(tmp_path, tables, big):
    reviews, products = tables
    assert queries.naive_cost(3, reviews, products, big) == 60
    assert queries.naive_cost(6, reviews, products, big) == 60
    assert queries.naive_cost(5, reviews, products, big) == 60 + 20
    # F4 yes among verified rows (i % 3 != 0): even or multiple of 7. The 20th is review 46
    verified = [i for i in range(60) if i % 3 != 0]
    yes = [i for i in verified if i % 2 == 0 or i % 7 == 0]
    assert len(yes) == 23
    assert queries.naive_cost(4, reviews, products, big) == verified.index(yes[19]) + 1

    def few(i):  # only 12 verified yes
        out = label_rules(i)
        out["F4"] = "yes" if i in yes[:12] else "no"
        return out
    thin = write_tier(tmp_path, reviews, few, tier="thin")
    assert queries.naive_cost(4, reviews, products, thin) == len(verified)


def test_truth_answers(tables, big):
    reviews, products = tables
    t = {no: queries.truth(no, reviews, products, big, ALIASES) for no in range(1, 11)}
    assert t[1].answer == 6  # rating <= 2 and i % 4 == 0
    assert t[2].answer == {"shipping or delivery": 6, "product quality or a defect": 6,
                           "a wrong, fake or missing item": 6, "customer service or returns": 6,
                           "price or value": 6, "other": 6}
    assert t[3].answer == ["Bolt", "Acme", "Cora", "Dyne", ""]
    assert t[3].counts == {"Bolt": 7, "Acme": 6, "Cora": 5, "Dyne": 4, "": 2}
    assert t[4].answer == [r for r in range(60) if r % 3 != 0 and (r % 2 == 0 or r % 7 == 0)][:20]
    assert len(t[4].answer) == 20 and len(t[4].matches) == 23
    assert t[5].answer == 10
    assert t[6].answer == ["Case A", "Case D", "Case E"]
    assert t[6].counts == {"Case A": 5, "Case D": 5, "Case E": 5}
    assert t[7].answer == ["iphone 13", "pixel 7", "galaxy s21", "lg v30", "moto g", "nokia 3310", "pixel 6"]
    assert t[7].counts["iphone 13"] == 3 and "none" not in t[7].counts
    assert t[8].answer == 6
    assert t[9].answer == ["Bolt", "Dyne", "Acme", "Cora"]  # newest 25%: Bolt 3, Dyne 2, Acme 1, Cora 1
    assert t[10].answer == pytest.approx(3.0)


def test_meets_count_margin():
    t = queries.Truth(100, None)
    assert queries.meets(1, 110, t) and not queries.meets(1, 111, t)
    t = queries.Truth(1000, None)
    assert queries.meets(1, 1050, t) and not queries.meets(1, 1051, t)
    assert queries.meets(1, 10, queries.Truth(0, None)) and not queries.meets(1, 11, queries.Truth(0, None))


def test_meets_groups():
    t = queries.Truth({"a": 1000, "b": 100, "c": 0}, None)
    assert queries.meets(2, {"a": 1050, "b": 110}, t)  # c missing counts as 0
    assert not queries.meets(2, {"a": 1051, "b": 100}, t)
    assert not queries.meets(2, {"a": 1000, "b": 100, "c": 11}, t)
    assert not queries.meets(2, {"a": 1000, "b": 100, "d": 11}, t)


def test_top10_tie_rule():
    names = [f"g{i:02d}" for i in range(15)]
    counts = {n: 50 - i for i, n in enumerate(names[:9])}  # g00..g08 clearly ahead
    counts.update({"g09": 10, "g10": 10, "g11": 10, "g12": 3})
    listed = queries._top(counts)
    assert listed == names[:10]  # the tie for 10th place is broken by key
    t = queries.Truth(listed, counts)
    assert queries.meets(3, names[:9] + ["g11"], t)  # any group tied with the 10th counts
    assert queries.meets(3, names[:8] + ["g10", "g11"], t)
    assert queries.meets(3, names[:9] + ["g12"], t)  # exactly 9 overlap
    assert not queries.meets(3, names[:8] + ["g12", "g13"], t)  # 8 overlap
    assert not queries.meets(3, names[:7] + ["g12", "g13", "g14"], t)


def test_top10_thin_truth():
    t = queries.Truth(["a", "b", "c"], {"a": 5, "b": 4, "c": 3})
    assert queries.meets(3, ["a", "b", "c"], t)
    assert not queries.meets(3, ["a", "b", "z"], t)  # needs min(9, 3) = 3


def test_meets_rows():
    matches = frozenset(range(100, 130))
    t = queries.Truth(list(range(100, 120)), None, matches)
    assert queries.meets(4, list(range(100, 119)) + [999], t)  # 19 of 20
    assert not queries.meets(4, list(range(100, 118)) + [998, 999], t)  # 18 of 20
    assert not queries.meets(4, list(range(100, 121)), t)  # 21 rows returned
    few = frozenset(range(7))
    t = queries.Truth(list(range(7)), None, few)
    assert queries.meets(4, list(range(7)), t)
    assert not queries.meets(4, list(range(6)), t)


def test_meets_average():
    t = queries.Truth(4.0, None)
    assert queries.meets(10, 4.05, t) and queries.meets(10, 3.95, t)
    assert not queries.meets(10, 4.051, t)
    assert queries.meets(10, None, queries.Truth(None, None))
    assert not queries.meets(10, 3.0, queries.Truth(None, None))
    assert not queries.meets(10, None, t)


def test_missing_store_is_a_group(tables, big):
    reviews, products = tables
    q3 = queries.scope(3, reviews, products)
    assert "" in set(q3["group"])
    assert queries.truth(3, reviews, products, big, {}).counts[""] == 2


def test_normalize_model():
    assert queries.normalize_model("iPhone 13 Pro.", {}) == "iphone 13 pro"
    assert queries.normalize_model("iPhone13", {"iphone13": "iphone 13"}) == "iphone 13"
    for none in ("None", "n/a", "", "unknown", "Not mentioned"):
        assert queries.normalize_model(none, {}) == "none"
    assert queries.normalize_model("  Pixel   7 ", {}) == "pixel 7"


def test_load_aliases(tmp_path):
    assert queries.load_aliases(tmp_path / "missing.json") == {}
    (tmp_path / "a.json").write_text(json.dumps({"iPhone13": "iPhone 13"}))
    assert queries.load_aliases(tmp_path / "a.json") == {"iphone13": "iphone 13"}


def test_question_scopes(tables):
    reviews, products = tables
    s = queries.question_scopes(reviews, products)
    everything = set(reviews["text_key"])
    assert s["F5a"] == everything and s["F5b"] == everything
    assert s["F6"] == set(queries.scope(6, reviews, products)["text_key"])
    assert s["F3"] == set(queries.scope(3, reviews, products)["text_key"])
    assert s["F1"] == set(queries.scope(1, reviews, products)["text_key"])
    assert set(s) == set(QUESTIONS)


def test_partial_last_line_dropped(tmp_path):
    cols = ["text_key", "answer", "p0", "p1", "in_tokens", "out_tokens", "latency_ms"]
    path = tmp_path / "F1.csv"
    row = lambda k: {"text_key": k, "answer": "yes", "p0": 0.9, "p1": 0.1, "in_tokens": 5, "out_tokens": 1,
                     "latency_ms": 2.5}
    labels.append_rows(path, [row("a"), row("b")], cols)
    with open(path, "a") as f:
        f.write("c,ye")  # a write cut off mid-field
    assert labels.read_label_file(path)["text_key"].tolist() == ["a", "b"]
    labels.append_rows(path, [row("c")], cols)  # resuming redoes the cut-off row
    df = labels.read_label_file(path)
    assert df["text_key"].tolist() == ["a", "b", "c"]
    # a complete-looking line with too few fields is dropped too
    with open(path, "a") as f:
        f.write("d,yes,0.5\n")
    assert labels.read_label_file(path)["text_key"].tolist() == ["a", "b", "c"]


def test_append_rows_header_cases(tmp_path):
    cols = ["text_key", "answer", "in_tokens", "out_tokens", "latency_ms"]
    row = lambda k: {"text_key": k, "answer": "yes", "in_tokens": 1, "out_tokens": 1, "latency_ms": 1.0}
    cut = tmp_path / "cut.csv"
    cut.write_text("text_key,ans")  # header cut off, no newline
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    fresh = tmp_path / "new" / "dir" / "fresh.csv"
    for path in (cut, empty, fresh):
        labels.append_rows(path, [row("a"), row("b")], cols)
        df = labels.read_label_file(path)
        assert list(df.columns) == cols and df["text_key"].tolist() == ["a", "b"]


def test_duplicate_keys_raise(tmp_path):
    cols = ["text_key", "answer", "in_tokens", "out_tokens", "latency_ms"]
    path = tmp_path / "E7.csv"
    row = {"text_key": "a", "answer": "x", "in_tokens": 1, "out_tokens": 1, "latency_ms": 1.0}
    labels.append_rows(path, [row, row], cols)
    with pytest.raises(ValueError, match="E7.csv"):
        labels.read_label_file(path)


def test_load_tier_keeps_text_answers(tmp_path):
    cols = ["text_key", "answer", "in_tokens", "out_tokens", "latency_ms"]
    labels.append_rows(tmp_path / "E7.csv", [
        {"text_key": "a", "answer": "none", "in_tokens": 1, "out_tokens": 1, "latency_ms": 1.0},
        {"text_key": "b", "answer": "NA", "in_tokens": 1, "out_tokens": 1, "latency_ms": 1.0},
        {"text_key": "c", "answer": 'iPhone, "13"', "in_tokens": 1, "out_tokens": 1, "latency_ms": 1.0}], cols)
    tier = labels.load_tier(tmp_path, "small")
    assert tier.answers["E7"] == {"a": "none", "b": "NA", "c": 'iPhone, "13"'}
    assert tier.probs["E7"] == {}


def test_load_tier_probs(tmp_path, tables):
    tier = write_tier(tmp_path, tables[0])
    key = tables[0].text_key[0]
    assert tier.answers["F1"][key] == "yes"
    assert tier.probs["F1"][key].tolist() == [0.5, 0.5]
    assert len(tier.probs["C2"][key]) == 6


def test_missing_decision_raises(tables):
    reviews, products = tables
    q1 = queries.scope(1, reviews, products)
    with pytest.raises(KeyError):
        queries.answer_from(1, q1, {"F1": {}}, {})
    # Q5: a "no" on either filter settles a row; a missing one with no "no" is an error
    q5 = queries.scope(5, reviews, products)
    keys = q5["text_key"].tolist()
    only_a_no = {"F5a": {k: "no" for k in keys}}
    assert queries.answer_from(5, q5, only_a_no, {}) == 0
    only_a_yes = {"F5a": {k: "yes" for k in keys}}
    with pytest.raises(KeyError):
        queries.answer_from(5, q5, only_a_yes, {})
    only_b_no = {"F5b": {k: "no" for k in keys}}
    assert queries.answer_from(5, q5, only_b_no, {}) == 0


def test_rows_without_a_decision_are_not_returned(tables):
    reviews, products = tables
    q4 = queries.scope(4, reviews, products)
    keys = q4["text_key"].tolist()
    answered = {k: "yes" for k in keys[:5]}
    assert queries.answer_from(4, q4, {"F4": answered}, {}) == q4["review_id"].tolist()[:5]
    allyes = {k: "yes" for k in keys}
    assert len(queries.answer_from(4, q4, {"F4": allyes}, {})) == 20


def test_top10_cut_at_ten():
    df = pd.DataFrame({"review_id": range(12 * 3), "text_key": [f"k{i}" for i in range(36)],
                       "rating": 1, "timestamp": 0, "group": [f"g{i % 12:02d}" for i in range(36)]})
    dec = {"F3": {k: "yes" for k in df.text_key}}
    out = queries.answer_from(3, df, dec, {})
    assert out == [f"g{i:02d}" for i in range(10)]


def test_empty_scope(tables):
    reviews, products = tables
    reviews = reviews.assign(rating=5)
    scope_df = queries.scope(1, reviews, products)
    assert len(scope_df) == 0
    tier = labels.TierLabels("big", {"F1": {}}, {"F1": {}})
    t = queries.truth(1, reviews, products, tier, {})
    assert t.answer == 0 and queries.meets(1, 0, t)
    t10 = queries.truth(10, reviews, products, labels.TierLabels("big", {"F10": {k: "no" for k in reviews.text_key}}, {}), {})
    assert t10.answer is None
