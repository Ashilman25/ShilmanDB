import json
import math

import numpy as np
import pandas as pd
import pytest
from aiq_world import make_world

from bench.ai_queries import queries
from bench.ai_queries.cascade import Cutoffs, fit_class_cutoff, fit_cutoffs, route, wilson_bound
from bench.ai_queries.common import text_key
from bench.ai_queries.costs import Usage, Weights, weights_from
from bench.ai_queries.labels import TierLabels
from bench.ai_queries.oracle import Oracle
from bench.ai_queries.proxy import Proxy
from bench.ai_queries.queries import RowTargets
from bench.ai_queries.questions import C2_CLASSES, QUESTIONS

SEEDS = range(1, 101)
DELTA = 0.01


def _labels(tier="big"):
    answers = {"F3": {"a": "yes", "b": "no", "c": "yes"}}
    probs = {"F3": {"a": np.array([0.9, 0.1]), "b": np.array([0.2, 0.8]), "c": np.array([0.7, 0.3])}}
    return TierLabels(tier, answers, probs)


# ---- oracle ----


def test_oracle_counts_first_asks_only():
    o = Oracle("big", _labels())
    assert o.ask("F3", ["a", "b"]) == {"a": "yes", "b": "no"}
    o.ask("F3", ["a"])
    o.ask("F3", ["c", "c"])
    assert o.calls == 3
    assert o.log == [("F3", "a"), ("F3", "b"), ("F3", "c")]


def test_oracle_probs_share_the_counter():
    o = Oracle("small", _labels("small"))
    p = o.probs("F3", ["c", "a"])
    assert list(p) == ["c", "a"] and p["a"][0] == pytest.approx(0.9)
    p["a"][0] = 0.0  # a copy: the stored labels don't change
    assert o.probs("F3", ["a"])["a"][0] == pytest.approx(0.9)
    o.ask("F3", ["a", "b"])  # a was already paid for
    assert o.calls == 3
    assert o.log == [("F3", "c"), ("F3", "a"), ("F3", "b")]


def test_oracle_missing_key_raises_and_costs_nothing():
    o = Oracle("big", _labels())
    with pytest.raises(KeyError, match="F3.*zzz"):
        o.ask("F3", ["a", "zzz"])
    with pytest.raises(KeyError, match="F9"):
        o.ask("F9", ["a"])
    with pytest.raises(KeyError):
        o.probs("F3", ["zzz"])
    assert o.log == [] and o.calls == 0  # a failed ask charges nothing


def test_oracle_exposes_no_unasked_answers():
    o = Oracle("big", _labels())
    public = {n for n in dir(o) if not n.startswith("_")}
    assert public == {"tier", "calls", "log", "ask", "probs"}
    assert not any(isinstance(getattr(o, n), (dict, TierLabels)) for n in public)


# ---- bounds and cut-offs ----


def test_wilson_bounds():
    assert 0.85 < wilson_bound(95, 100, 0.01, "lower") < 0.95
    assert 0.95 < wilson_bound(95, 100, 0.01, "upper") <= 1.0
    assert wilson_bound(0, 0, 0.01, "lower") == 0.0
    assert wilson_bound(0, 0, 0.01, "upper") == 1.0
    assert wilson_bound(100, 100, 0.01, "upper") == 1.0
    with pytest.raises(ValueError):
        wilson_bound(1, 2, 0.01, "middle")


def test_fit_cutoffs_separable():
    rng = np.random.default_rng(0)
    is_yes = rng.random(4000) < 0.25
    scores = np.where(is_yes, rng.uniform(0.6, 1.0, 4000), rng.uniform(0.0, 0.4, 4000))
    cut = fit_cutoffs(scores, is_yes, RowTargets(0.97, 0.97))
    assert np.isfinite(cut.accept) and np.isfinite(cut.reject)
    assert cut.reject <= cut.accept
    assert np.mean(route(scores, cut) == -1) <= 0.10


def test_fit_cutoffs_useless_scores():
    rng = np.random.default_rng(3)
    scores = rng.random(600)
    is_yes = rng.random(600) < 0.1
    cut = fit_cutoffs(scores, is_yes, RowTargets(0.97, 0.97))
    assert cut.accept == math.inf and cut.reject == -math.inf


def test_route_values_and_precedence():
    cut = Cutoffs(accept=0.8, reject=0.2)
    r = route(np.array([0.9, 0.8, 0.5, 0.2, 0.1]), cut)
    assert r.dtype == np.int8 and r.tolist() == [1, 1, -1, 0, 0]
    both = route(np.array([0.5]), Cutoffs(accept=0.4, reject=0.6))
    assert both.tolist() == [1]  # yes wins an overlap
    assert route(np.array([0.5]), Cutoffs(math.inf, -math.inf)).tolist() == [-1]


def _ref_grid(scores, grid):
    return [np.quantile(scores, k / grid, method="lower") for k in range(1, grid + 1)]


def _ref_cutoffs(scores, is_yes, targets, grid=20):
    """Slow, plain loops over the grid."""
    accept, reject = math.inf, -math.inf
    if len(scores) == 0:
        return Cutoffs(accept, reject)
    points = _ref_grid(scores, grid)
    for t in sorted(points):
        yes = total = 0
        for sc, y in zip(scores, is_yes):
            if sc >= t:
                total += 1
                yes += bool(y)
        if yes / total >= targets.precision:
            accept = float(t)
            break
    positives = [sc for sc, y in zip(scores, is_yes) if y]
    for t in sorted(points, reverse=True):
        if positives and sum(sc <= t for sc in positives) / len(positives) <= 1 - targets.recall:
            reject = float(t)
            break
    return Cutoffs(accept, reject)


def _ref_class_cutoff(top, correct, accuracy, grid=20):
    if len(top) == 0:
        return math.inf
    for t in sorted(_ref_grid(top, grid)):
        hits = [c for p, c in zip(top, correct) if p >= t]
        if sum(hits) / len(hits) >= accuracy:
            return float(t)
    return math.inf


def test_cutoffs_match_brute_force():
    cases = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(40, 400))
        is_yes = rng.random(n) < rng.choice([0.05, 0.2, 0.5])
        cases.append((rng.normal(2.5 * is_yes, 1.0), is_yes))                 # informative
        cases.append((np.round(rng.normal(2.5 * is_yes, 1.0), 0), is_yes))    # many ties
        cases.append((rng.random(n), is_yes))                                 # useless
    n = 50
    cases += [(np.full(n, 0.5), np.arange(n) % 3 == 0),                       # all scores equal
              (np.random.default_rng(9).random(n), np.zeros(n, dtype=bool)),  # zero positives
              (np.random.default_rng(9).random(n), np.ones(n, dtype=bool)),   # all positives
              (np.array([0.3]), np.array([True])), (np.array([]), np.array([], dtype=bool))]
    for scores, is_yes in cases:
        for tgt in (RowTargets(0.97, 0.97), RowTargets(0.8, 0.9)):
            assert fit_cutoffs(scores, is_yes, tgt) == _ref_cutoffs(scores, is_yes, tgt)
        for acc in (0.97, 0.7):
            assert fit_class_cutoff(scores, is_yes, acc) == _ref_class_cutoff(scores, is_yes, acc)


def _cascade_run(seed, n, yes_rate, sep, target, m=600):
    """Cut-offs from m calibration rows; the big model (the truth) answers the band."""
    rng = np.random.default_rng(seed)
    is_yes = rng.random(n) < yes_rate
    scores = rng.normal(sep * is_yes, 1.0)
    cal = rng.choice(n, size=m, replace=False)
    cut = fit_cutoffs(scores[cal], is_yes[cal], RowTargets(target, target))
    r = route(scores, cut)
    final_yes = (r == 1) | ((r == -1) & is_yes)
    tp = np.sum(final_yes & is_yes)
    return tp / max(final_yes.sum(), 1), tp / max(is_yes.sum(), 1), np.mean(r == -1), cut


def test_cascade_hits_targets():
    results = [_cascade_run(s, 20000, 0.10, 4.5, 0.97) for s in SEEDS]
    precision, recall, band = np.mean([r[:3] for r in results], axis=0)
    assert precision >= 0.96 and recall >= 0.96
    assert band <= 0.5


def test_rare_positives_go_to_big():
    results = [_cascade_run(s, 20000, 0.01, 4.5, 0.97) for s in SEEDS]
    assert np.mean([r for _, r, _, _ in results]) >= 0.9
    # a calibration sample with no positives at all: everything not accepted goes to big
    scores = np.random.default_rng(1).random(600)
    cut = fit_cutoffs(scores, np.zeros(600, dtype=bool), RowTargets(0.97, 0.97))
    assert cut.reject == -math.inf
    assert (route(scores, cut) != 0).all()


def _class_run(seed, target, n=20000, m=600):
    rng = np.random.default_rng(seed)
    top = 1 - 0.7 * rng.beta(1, 3, n)  # lots of confident rows
    correct = rng.random(n) < top  # well calibrated
    cal = rng.choice(n, size=m, replace=False)
    cut = fit_class_cutoff(top[cal], correct[cal], target)
    accepted = top >= cut
    return (correct[accepted].mean() if accepted.any() else 1.0), accepted.mean()


def test_class_cutoff():
    results = [_class_run(s, 0.97) for s in SEEDS]
    assert np.mean([a for a, _ in results]) >= 0.96
    assert np.mean([share for _, share in results]) > 0.20
    assert fit_class_cutoff(np.array([0.9]), np.array([False]), 0.97) == math.inf


# ---- proxy ----


def _auc(scores, y):
    ranks = np.empty(len(scores))
    ranks[np.argsort(scores, kind="stable")] = np.arange(1, len(scores) + 1)
    pos = y == 1
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def test_proxy_learns_separable():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.4).astype(int)
    X = rng.normal(0, 1, (400, 16)) + 2.0 * y[:, None] * np.linspace(0.2, 1, 16)
    s1 = Proxy().fit(X[:300], y[:300]).scores(X[300:])
    s2 = Proxy().fit(X[:300], y[:300]).scores(X[300:])
    assert s1.shape == (100,)
    assert _auc(s1, y[300:]) > 0.95
    assert np.array_equal(s1, s2)


def test_proxy_multinomial_and_missing_class():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 6, 300)
    centers = rng.normal(0, 2, (6, 24))
    X = centers[y] + rng.normal(0, 1, (300, 24))
    s = Proxy().fit(X, y, n_classes=6).scores(X)
    assert s.shape == (300, 6) and np.allclose(s.sum(axis=1), 1)
    assert np.mean(s.argmax(axis=1) == y) > 0.9
    # class 5 never appears: no error, no warning, probabilities stay valid
    y2 = np.where(y == 5, 0, y)
    s2 = Proxy().fit(X, y2, n_classes=6).scores(X)
    assert np.isfinite(s2).all() and np.allclose(s2.sum(axis=1), 1)
    assert s2[:, 5].max() < 0.05
    # a binary fit on one class only
    s3 = Proxy().fit(X, np.zeros(300, dtype=int)).scores(X)
    assert np.isfinite(s3).all() and s3.max() < 0.05


def test_proxy_handles_constant_column():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (200, 5))
    X[:, 2] = 7.0
    y = (X[:, 0] > 0).astype(int)
    assert _auc(Proxy().fit(X, y).scores(X), y) > 0.95


# ---- costs ----


def _latency(small_ms, small_price_in):
    rows = []
    for tier, ms, tin in (("big", 100.0, 100.0), ("small", small_ms, 100.0)):
        for qid in ("F3", "F8"):
            rows.append({"tier": tier, "qid": qid, "rows": 50, "median_ms": ms,
                         "mean_in_tokens": tin, "mean_out_tokens": 1.0})
    prices = pd.DataFrame([
        {"tier": "big", "stand_in": "x", "usd_per_m_in": 10.0, "usd_per_m_out": 0.0, "source_url": "u", "date": "d"},
        {"tier": "small", "stand_in": "y", "usd_per_m_in": small_price_in, "usd_per_m_out": 0.0,
         "source_url": "u", "date": "d"},
        {"tier": "embed", "stand_in": "z", "usd_per_m_in": 0.0, "usd_per_m_out": 0.0, "source_url": "u", "date": "d"},
    ])
    return pd.DataFrame(rows), prices


def _embed_timing(sec_per_row):
    return pd.DataFrame([{"pass": "test", "rows": 1000, "seconds": 1000 * sec_per_row,
                          "rows_per_sec": 1 / sec_per_row}])


def test_weights_take_the_larger_ratio():
    lat, prices = _latency(small_ms=20.0, small_price_in=0.5)  # 0.2 of the time, 0.05 of the price
    w = weights_from(lat, prices, _embed_timing(0.001), proxy_us_per_row=50.0)
    assert w.small == pytest.approx(0.2)
    assert w.proxy == pytest.approx(0.05 / 100.0)  # 50 us = 0.05 ms against 100 ms
    assert w.embed == pytest.approx(1.0 / 100.0)  # 1 ms per row
    lat, prices = _latency(small_ms=5.0, small_price_in=2.0)  # 0.05 of the time, 0.2 of the price
    assert weights_from(lat, prices, _embed_timing(0.001), 50.0).small == pytest.approx(0.2)


def test_weights_missing_rows_raise():
    lat, prices = _latency(20.0, 0.5)
    with pytest.raises(ValueError, match="small"):
        weights_from(lat[lat["tier"] == "big"], prices, _embed_timing(0.001), 50.0)
    with pytest.raises(ValueError, match="embed"):
        weights_from(lat, prices[prices["tier"] != "embed"], _embed_timing(0.001), 50.0)
    with pytest.raises(ValueError, match="test"):
        weights_from(lat, prices, _embed_timing(0.001).assign(**{"pass": "slice"}), 50.0)


def test_usage_cost():
    cost = Usage(big=10, small=100, proxy=1000, embed_rows=300).cost(Weights(0.1, 1e-6, 0.01))
    assert cost == pytest.approx(10 + 10 + 0.001 + 3)


# ---- the synthetic world ----


@pytest.fixture(scope="module")
def world():
    return make_world(n=1500, seed=0)


def test_world_is_deterministic_and_seeded():
    a, b = make_world(n=300, seed=4), make_world(n=300, seed=4)
    assert a.reviews.equals(b.reviews) and a.products.equals(b.products)
    assert np.array_equal(a.emb, b.emb) and a.emb_keys == b.emb_keys
    assert dict(a.big.answers["F3"]) == dict(b.big.answers["F3"])
    c = make_world(n=300, seed=5)
    assert not a.reviews["text"].equals(c.reviews["text"]) or not np.array_equal(a.emb, c.emb)


def test_world_tables_have_the_columns(world):
    assert list(world.reviews.columns) == ["review_id", "rating", "title", "text", "parent_asin", "timestamp",
                                          "helpful_vote", "verified_purchase", "user_id", "text_key"]
    assert list(world.products.columns) == ["parent_asin", "title", "store", "price", "main_category",
                                           "categories", "average_rating", "rating_number"]
    r = world.reviews
    assert (r["text_key"] == [text_key(t, x) for t, x in zip(r["title"], r["text"])]).all()
    assert r["verified_purchase"].dtype == bool and r["review_id"].is_unique
    assert set(r["rating"]) == {1, 2, 3, 4, 5}
    assert all(isinstance(c, str) and isinstance(json.loads(c), list) for c in world.products["categories"])


def test_world_scopes_are_not_trivial(world):
    n = len(world.reviews)
    for no in queries.QUERIES:
        s = queries.scope(no, world.reviews, world.products)
        assert 30 <= len(s) <= n, no
    assert not world.reviews["parent_asin"].isin(world.products["parent_asin"]).all()
    assert world.products["price"].isna().any() and (world.products["store"] == "").any()
    assert len(queries.scope(6, world.reviews, world.products)) < len(queries.scope(2, world.reviews, world.products))
    assert queries.scope(3, world.reviews, world.products)["group"].nunique() > 3


def test_world_labels_cover_every_scope(world):
    scopes = queries.question_scopes(world.reviews, world.products)
    for tier in (world.big, world.small):
        for qid, keys in scopes.items():
            assert keys <= set(tier.answers[qid])
    assert set(world.big.answers) == set(QUESTIONS)
    assert set(world.big.answers["C2"].values()) <= set(C2_CLASSES)
    e7 = pd.Series(list(world.big.answers["E7"].values()))
    assert (e7 == "none").mean() > 0.5 and e7[e7 != "none"].nunique() >= 4
    assert world.big.probs["E7"] == {}
    p = next(iter(world.small.probs["C2"].values()))
    assert len(p) == 6 and p.sum() == pytest.approx(1)


def test_world_rates_and_agreement(world):
    keys = list(world.big.answers["F8"])
    f8 = np.mean([world.big.answers["F8"][k] == "yes" for k in keys])
    f3 = np.mean([world.big.answers["F3"][k] == "yes" for k in keys])
    assert f8 < 0.04 and 0.1 < f3 < 0.35
    rare = make_world(n=1500, seed=0, rare=True)
    assert np.mean([a == "yes" for a in rare.big.answers["F8"].values()]) < f8
    agree = np.mean([world.small.answers["F3"][k] == world.big.answers["F3"][k] for k in keys])
    assert 0.85 < agree < 0.95
    exact = make_world(n=1500, seed=0, small_agree=1.0)
    assert dict(exact.small.answers["F3"]) == dict(exact.big.answers["F3"])


def test_world_duplicates_share_everything(world):
    r = world.reviews
    dups = r[r["text_key"].duplicated(keep=False)]
    assert 0.03 * len(r) < r["text_key"].duplicated().sum() < 0.08 * len(r)
    assert len(dups) > 0
    assert list(world.emb_keys) == list(dict.fromkeys(r["text_key"]))
    assert len(world.emb_keys) == r["text_key"].nunique()
    assert make_world(n=300, seed=1, dup_share=0.0).reviews["text_key"].is_unique


def test_world_embeddings_carry_a_signal(world):
    e = world.emb
    assert e.dtype == np.float32 and e.shape == (len(world.emb_keys), 32)
    assert np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-5)
    y = np.array([world.big.answers["F3"][k] == "yes" for k in world.emb_keys]).astype(int)
    cos = e @ world.question_emb["F3"]
    assert _auc(cos, y) > 0.7
    assert _auc(cos, y) < 0.99  # imperfect on purpose
    assert np.linalg.norm(world.question_emb["F3"]) == pytest.approx(1)
    flat = make_world(n=1500, seed=0, proxy_signal=0.0)
    y0 = np.array([flat.big.answers["F3"][k] == "yes" for k in flat.emb_keys]).astype(int)
    assert abs(_auc(flat.emb @ flat.question_emb["F3"], y0) - 0.5) < 0.08
