"""A small made-up world for tests: tables, big and small model answers, embeddings.

Everything is a pure function of the arguments. Labels come from latent truths
per distinct text, so a duplicated review has the same labels and embedding as
the review it copies.
"""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bench.ai_queries.common import text_key
from bench.ai_queries.labels import TierLabels
from bench.ai_queries.questions import C2_CLASSES, QUESTIONS

DIM = 32
SIGNAL = 1.5  # length of the push a "yes" gives the embedding, before proxy_signal
YES_RATES = {"F1": 0.12, "F3": 0.20, "F4": 0.08, "F5a": 0.30, "F5b": 0.30, "F6": 0.10, "F8": 0.01, "F10": 0.15}
C2_WEIGHTS = np.array([0.20, 0.30, 0.10, 0.15, 0.15, 0.10])
PHONES = ["iphone 13", "galaxy s21", "pixel 6", "iphone 12", "galaxy s22", "iphone 11"]
PHONE_WEIGHTS = np.array([0.30, 0.22, 0.15, 0.12, 0.10, 0.11])
NONE_SHARE = 0.7
STORES = ["Acme", "Bolt", "Cora", "Dyne", "Echo", "Fable", "Gust", "Hex", "Iris", "Jolt", "Kite", "Lumen"]
WORDS = ["fits", "phone", "great", "broke", "slow", "battery", "screen", "love", "cheap", "fast", "grip", "color",
         "arrived", "box", "price", "works", "hot", "case", "charger", "cable"]


@dataclass(frozen=True)
class World:
    reviews: pd.DataFrame
    products: pd.DataFrame
    big: TierLabels
    small: TierLabels
    emb: np.ndarray
    emb_keys: list[str]
    question_emb: dict[str, np.ndarray]


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _products(rng, n_products: int) -> pd.DataFrame:
    is_case = rng.random(n_products) < 0.45
    price = np.where(is_case, rng.uniform(5, 40, n_products), rng.uniform(5, 120, n_products)).round(2)
    price[rng.random(n_products) < 0.10] = np.nan
    store = rng.choice(STORES, n_products).astype(object)
    store[rng.random(n_products) < 0.10] = ""
    case_cats = json.dumps(["Cell Phones & Accessories", "Cases, Holsters & Sleeves"])
    other_cats = json.dumps(["Cell Phones & Accessories", "Chargers"])
    return pd.DataFrame({
        "parent_asin": [f"B{i:09d}" for i in range(n_products)],
        "title": [f"Product {i}" for i in range(n_products)],
        "store": store,
        "price": price,
        "main_category": "Cell Phones & Accessories",
        "categories": np.where(is_case, case_cats, other_cats),
        "average_rating": rng.uniform(2.5, 5.0, n_products).round(1),
        "rating_number": rng.integers(1, 5000, n_products),
    })


def _reviews(rng, n: int, products: pd.DataFrame, dup_share: float) -> pd.DataFrame:
    asin = rng.choice(products["parent_asin"].to_numpy(), n).astype(object)
    missing = rng.random(n) < 0.08
    asin[missing] = [f"X{i:09d}" for i in np.flatnonzero(missing)]
    words = rng.choice(WORDS, (n, 6))
    title = [f"{w[0]} {w[1]} {i}" for i, w in enumerate(words)]
    text = [" ".join(w) + f" review number {i}" for i, w in enumerate(words)]
    n_dup = int(dup_share * n)
    dup_at = np.sort(rng.choice(n, n_dup, replace=False))
    originals = np.setdiff1d(np.arange(n), dup_at)
    for at, src in zip(dup_at, rng.choice(originals, n_dup)):
        title[at], text[at] = title[src], text[src]
    df = pd.DataFrame({
        "review_id": np.arange(n),
        "rating": rng.choice([1, 2, 3, 4, 5], n, p=[0.15, 0.10, 0.10, 0.20, 0.45]),
        "title": title,
        "text": text,
        "parent_asin": asin,
        "timestamp": rng.integers(1_400_000_000_000, 1_700_000_000_000, n),
        "helpful_vote": rng.integers(0, 20, n),
        "verified_purchase": rng.random(n) < 0.8,
        "user_id": [f"U{u:06d}" for u in rng.integers(0, 10 * n, n)],
    })
    df["text_key"] = [text_key(t, x) for t, x in zip(df["title"], df["text"])]
    return df


def _yes_no_probs(rng, answers: np.ndarray, truth: np.ndarray, confident: np.ndarray, agree: float) -> np.ndarray:
    """p(yes) per row for a small model: agreeing rows are sure, wrong rows less so, a few sure and wrong."""
    sure = rng.uniform(0.70, 0.999, len(answers))
    unsure = rng.uniform(0.50, 0.85, len(answers))
    conf = np.where(truth == answers, sure, np.where(confident, rng.uniform(0.85, 0.99, len(answers)), unsure))
    return np.where(answers, conf, 1 - conf)


def make_world(n: int = 3000, seed: int = 0, rare: bool = False, dup_share: float = 0.05, *,
               proxy_signal: float = 1.0, small_agree: float = 0.9) -> World:
    rng = np.random.default_rng([seed, 0])
    products = _products(rng, max(20, n // 10))
    reviews = _reviews(rng, n, products, dup_share)
    keys = list(dict.fromkeys(reviews["text_key"]))
    k = len(keys)

    # latent truths, one per distinct text
    rates = {**YES_RATES, "F8": 0.002 if rare else 0.01}
    yes = {qid: rng.random(k) < rate for qid, rate in rates.items()}
    c2 = rng.choice(len(C2_CLASSES), k, p=C2_WEIGHTS)
    phone = np.where(rng.random(k) < NONE_SHARE, -1, rng.choice(len(PHONES), k, p=PHONE_WEIGHTS))

    # embeddings: noise plus a push along each question's direction
    dir_rng = np.random.default_rng([seed, 1])
    q_dir = {qid: _unit(dir_rng.normal(size=DIM)) for qid in rates}
    class_dir = np.array([_unit(v) for v in dir_rng.normal(size=(len(C2_CLASSES), DIM))])
    q_dir["E7"] = _unit(dir_rng.normal(size=DIM))
    q_dir["C2"] = _unit(class_dir.sum(axis=0))
    emb = rng.normal(size=(k, DIM))
    push = sum(yes[qid][:, None] * q_dir[qid] for qid in rates)
    push = push + class_dir[c2] + (phone >= 0)[:, None] * q_dir["E7"]
    emb = emb + proxy_signal * SIGNAL * push
    emb = (emb / np.linalg.norm(emb, axis=1, keepdims=True)).astype(np.float32)

    # the big model: right answers, confident probabilities
    big_ans, big_prob, small_ans, small_prob = {}, {}, {}, {}
    for qid, q in QUESTIONS.items():
        agree = rng.random(k) < small_agree
        if q.kind == "yesno":
            truth = yes[qid]
            p_big = np.where(truth, rng.uniform(0.85, 0.999, k), rng.uniform(0.001, 0.15, k))
            ans_s = np.where(agree, truth, ~truth)
            p_small = _yes_no_probs(rng, ans_s, truth, rng.random(k) < 0.25, small_agree)
            big_ans[qid] = dict(zip(keys, np.where(truth, "yes", "no")))
            big_prob[qid] = {key: np.array([p, 1 - p]) for key, p in zip(keys, p_big)}
            small_ans[qid] = dict(zip(keys, np.where(ans_s, "yes", "no")))
            small_prob[qid] = {key: np.array([p, 1 - p]) for key, p in zip(keys, p_small)}
        elif q.kind == "class":
            wrong = (c2 + rng.integers(1, len(C2_CLASSES), k)) % len(C2_CLASSES)
            ans_s = np.where(agree, c2, wrong)
            big_ans[qid] = dict(zip(keys, (C2_CLASSES[i] for i in c2)))
            big_prob[qid] = {key: _class_probs(rng, i, 0.9) for key, i in zip(keys, c2)}
            small_ans[qid] = dict(zip(keys, (C2_CLASSES[i] for i in ans_s)))
            small_prob[qid] = {key: _class_probs(rng, i, 0.8 if i == t else 0.6)
                               for key, i, t in zip(keys, ans_s, c2)}
        else:  # extract: no probabilities
            names = np.array([*PHONES, "none"], dtype=object)
            truth = np.where(phone < 0, len(PHONES), phone)
            other = (truth + rng.integers(1, len(names), k)) % len(names)
            ans_s = np.where(agree, truth, other)
            big_ans[qid] = dict(zip(keys, names[truth]))
            small_ans[qid] = dict(zip(keys, names[ans_s]))
            big_prob[qid], small_prob[qid] = {}, {}
    question_emb = {qid: v.astype(np.float32) for qid, v in q_dir.items()}
    return World(reviews, products, TierLabels("big", big_ans, big_prob), TierLabels("small", small_ans, small_prob),
                 emb, keys, question_emb)


def _class_probs(rng, top: int, mass: float) -> np.ndarray:
    rest = rng.dirichlet(np.ones(len(C2_CLASSES)))
    rest[top] = 0.0
    rest = rest / rest.sum() * (1 - mass)
    rest[top] = mass
    return rest
