import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import aiq_world
from aiq_world import make_world
from bench.ai_queries import queries
from bench.ai_queries.approaches import (ALL_ON, Env, QueryRun, Switches, run_session,
                                         split_embedding_rows)
from bench.ai_queries.costs import Usage, Weights
from bench.ai_queries.labels import TierLabels
from bench.ai_queries.oracle import Oracle
from bench.ai_queries.questions import C2_CLASSES, QUESTIONS

W = Weights(small=0.1, proxy=1e-6, embed=0.01)
NO_T4 = Switches(True, True, True, False)
COMBOS = [("incumbent_snowflake", Switches()), ("incumbent_bigquery", Switches()),
          ("shilmandb", Switches()), ("shilmandb", NO_T4), ("shilmandb", ALL_ON)]


class Factory:
    """An oracle factory that keeps every oracle it made, in order."""

    def __init__(self, labels: TierLabels):
        self.labels = labels
        self.made: list[Oracle] = []

    def __call__(self) -> Oracle:
        o = Oracle(self.labels.tier, self.labels)
        self.made.append(o)
        return o


def env_of(world, weights: Weights = W) -> Env:
    return Env(world.reviews, world.products, world.emb, {k: i for i, k in enumerate(world.emb_keys)},
               world.question_emb, weights, {})


def session(world, approach, switches=Switches(), seed=1, env=None, big_labels=None, small_labels=None):
    big, small = Factory(big_labels or world.big), Factory(small_labels or world.small)
    runs = run_session(approach, env or env_of(world), big, small, seed, switches)
    return runs, big, small


def distinct_keys(world, no) -> list[str]:
    return list(dict.fromkeys(queries.scope(no, world.reviews, world.products)["text_key"]))


@pytest.fixture(scope="module")
def world():
    return make_world(n=6000, seed=0)


@pytest.fixture(scope="module")
def env(world):
    return env_of(world)


@pytest.fixture(scope="module")
def big_world():
    return make_world(n=30000, seed=0)


# ---- no peeking ----


def _flip(labels: TierLabels, asked: set[tuple[str, str]]) -> TierLabels:
    """Keep every asked answer, change every other one."""
    names = [*aiq_world.PHONES, "none"]
    answers, probs = {}, {}
    for qid, table in labels.answers.items():
        kind = QUESTIONS[qid].kind
        answers[qid], probs[qid] = {}, {}
        for key, a in table.items():
            p = labels.probs[qid].get(key)
            if (qid, key) not in asked:
                if kind == "yesno":
                    a = "no" if a == "yes" else "yes"
                    p = p[::-1].copy()
                elif kind == "class":
                    a = C2_CLASSES[(C2_CLASSES.index(a) + 1) % len(C2_CLASSES)]
                    p = np.roll(p, 1)
                else:
                    a = names[(names.index(a) + 1) % len(names)]
            answers[qid][key] = a
            if p is not None:
                probs[qid][key] = p
    return TierLabels(labels.tier, answers, probs)


def _asked(factory: Factory) -> set[tuple[str, str]]:
    return {pair for o in factory.made for pair in o.log}


@pytest.mark.parametrize("approach,switches", COMBOS)
def test_answer_depends_only_on_asked_rows(world, env, approach, switches):
    runs, big, small = session(world, approach, switches, env=env)
    flipped_big, flipped_small = _flip(world.big, _asked(big)), _flip(world.small, _asked(small))
    assert flipped_big.answers["F8"] != world.big.answers["F8"]  # the flip really changes labels
    again, _, _ = session(world, approach, switches, env=env, big_labels=flipped_big, small_labels=flipped_small)
    assert again == runs


# ---- SQL first, dedupe, filter order ----


def test_pulls_ai_above_join(world, env):
    keys6 = distinct_keys(world, 6)
    assert len(keys6) > 600  # the query runs a real cascade, not the small-scope shortcut
    for approach, switches in COMBOS:
        runs, big, small = session(world, approach, switches, env=env)
        q6 = runs[5]
        assert q6.query == 6
        assert q6.usage.big <= len(keys6) and q6.usage.small <= len(keys6)
        if not switches.t4:  # a fresh oracle per query: the 6th belongs to Q6
            for o in (big.made[5], small.made[5]):
                assert {k for _, k in o.log} <= set(keys6), approach
                assert o.calls <= len(keys6)


def test_dedupe():
    w = make_world(n=3000, seed=2, dup_share=0.2)
    rows = len(queries.scope(8, w.reviews, w.products))
    keys = len(distinct_keys(w, 8))
    assert keys < 0.85 * rows
    e = env_of(w)
    for approach, switches in COMBOS:
        runs, _, _ = session(w, approach, switches, env=e)
        assert runs[7].usage.big <= keys


def test_q5_more_selective_first(monkeypatch):
    monkeypatch.setitem(aiq_world.YES_RATES, "F5a", 0.30)
    monkeypatch.setitem(aiq_world.YES_RATES, "F5b", 0.03)
    w = make_world(n=3000, seed=3)
    e = env_of(w)
    for approach, switches in COMBOS:
        runs, big, _ = session(w, approach, switches, env=e)
        if switches.t4:
            log = big.made[0].log
            start = sum(r.usage.big for r in runs[:4])
        else:
            log, start = big.made[4].log, 0
        after = log[start + 2 * 600:start + runs[4].usage.big]
        assert after, approach
        assert after[0][0] == "F5b", approach
        assert {q for q, _ in after} <= {"F5a", "F5b"}


# ---- LIMIT ----


def test_limit_streams_and_confirms(world, env):
    s4 = queries.scope(4, world.reviews, world.products)
    truth = world.big.answers["F4"]
    found, seen = 0, []
    for key in s4["text_key"]:
        seen.append(key)
        found += truth[key] == "yes"
        if found == queries.ROW_LIMIT:
            break
    expected = len(set(seen))
    naive = queries.naive_cost(4, world.reviews, world.products, world.big)

    runs, _, _ = session(world, "incumbent_bigquery", env=env)
    assert runs[3].usage.big == expected <= naive
    for approach, switches in COMBOS:
        runs, _, _ = session(world, approach, switches, env=env)
        returned = runs[3].answer
        by_id = dict(zip(s4["review_id"], s4["text_key"]))
        assert all(truth[by_id[r]] == "yes" for r in returned), approach
        assert len(returned) <= queries.ROW_LIMIT
    runs, _, _ = session(world, "shilmandb", Switches(t2=True), env=env)
    assert runs[3].usage.big < naive
    assert len(runs[3].answer) == queries.ROW_LIMIT


# ---- the four tools ----


def test_t3_respects_weights(world):
    costly = env_of(world, Weights(small=1.0, proxy=1e-6, embed=0.01))
    runs, _, _ = session(world, "shilmandb", Switches(t3=True), env=costly)
    assert not any("small" in r.variant for r in runs)
    assert sum(r.usage.small for r in runs) == 0  # it never even calibrated the small model
    cheap = env_of(world, Weights(small=0.01, proxy=1e-6, embed=0.01))
    runs, _, _ = session(world, "shilmandb", Switches(t3=True), env=cheap)
    assert any("small" in r.variant for r in runs)


def test_t4_reuses_answers(world, env):
    with_t4, _, _ = session(world, "shilmandb", Switches(t4=True), env=env)
    without, _, _ = session(world, "shilmandb", Switches(), env=env)
    assert with_t4[8].query == 9
    assert with_t4[8].usage.big < without[8].usage.big
    assert with_t4[2].usage.big == without[2].usage.big  # Q3 has nothing earlier to reuse


def test_t1_estimate_or_fallback(monkeypatch):
    monkeypatch.setitem(aiq_world.YES_RATES, "F1", 0.45)
    w = make_world(n=40000, seed=1, proxy_signal=0.3)
    runs, _, _ = session(w, "shilmandb", Switches(t1=True))
    q1 = runs[0]
    assert q1.mode == "estimate"
    assert q1.stated["lo"] <= q1.stated["est"] <= q1.stated["hi"]
    assert q1.answer == round(q1.stated["est"])
    assert q1.decisions is None
    # it paid less than the exact cascade would have
    exact, _, _ = session(w, "shilmandb", Switches(), env=None)
    assert q1.usage.big < exact[0].usage.big

    # rare matches: no confident wrong answer and no stop at look 1. Look 1 shows a count
    # this rare can't be pinned down short of a census, so T1 doesn't even try
    rare = make_world(n=3000, seed=0, rare=True)
    rare_env = env_of(rare)
    truth = queries.truth(8, rare.reviews, rare.products, rare.big, {})
    for seed in range(1, 21):
        runs, _, _ = session(rare, "shilmandb", Switches(t1=True), seed=seed, env=rare_env)
        q8 = runs[7]
        if q8.mode == "estimate":
            assert q8.stated["lo"] <= truth.answer <= q8.stated["hi"], seed
            assert queries.meets(8, q8.answer, truth), seed
            assert q8.stated["looks"] >= 2, seed
        else:
            assert q8.mode in ("exact", "full"), seed
        if seed <= 3:
            off, _, _ = session(rare, "shilmandb", Switches(), seed=seed, env=rare_env)
            assert q8.usage.big == off[7].usage.big, seed  # trying cost nothing


def test_t1_never_costs_more_than_without(big_world):
    # when look 1 shows sampling won't pay, T1 goes straight to the exact cascade instead of
    # sampling up to its cost first (that used to make Q1 / Q2 about 1.2x / 1.3x dearer here)
    w = big_world
    e = env_of(w, Weights(small=0.25, proxy=1e-6, embed=0.01))
    off = Switches(False, True, True, True)
    for seed in range(1, 6):
        with_t1, _, _ = session(w, "shilmandb", ALL_ON, seed=seed, env=e)
        without, _, _ = session(w, "shilmandb", off, seed=seed, env=e)
        for no in (1, 2):
            a, b = with_t1[no - 1].usage.cost(e.weights), without[no - 1].usage.cost(e.weights)
            assert a <= 1.01 * b, (seed, no, a, b)


def test_t1_counts_known_answers_exactly(world, env, monkeypatch):
    # with T4, Q9 comes after Q3 asked the same question on many of its keys: those are known, not sampled
    from bench.ai_queries import approaches
    seen = {}
    real = approaches._Estimator.__init__

    def spy(self, ctx, order):
        real(self, ctx, order)
        seen[ctx.no] = self

    monkeypatch.setattr(approaches._Estimator, "__init__", spy)
    session(world, "shilmandb", Switches(t1=True, t4=True), env=env)
    q9 = seen[9]
    known = q9.ctx.before["F3"]
    assert len(known & set(distinct_keys(world, 9))) > 600
    assert not set(q9.ctx.keys[q9.pop]) & known and not set(q9.ctx.keys[q9.later]) & known
    seen.clear()
    session(world, "shilmandb", Switches(t1=True), env=env)
    assert seen[9].ctx.before["F3"] == frozenset()  # a fresh oracle per query knows nothing beforehand


def test_falls_back_to_full_when_no_cascade_pays(world):
    costly = env_of(world, Weights(small=1.0, proxy=1e-6, embed=0.01))
    runs, _, small = session(world, "incumbent_snowflake", env=costly)
    for r in runs:
        if r.query == 4:
            continue
        assert r.mode == "full" and r.variant == "big", r.query
        assert r.answer == queries.truth(r.query, world.reviews, world.products, world.big, {}).answer
        assert small.made[r.query - 1].calls == 0  # it saw the small model could not pay before paying for it


def test_all_off_equals_bigquery(world, env):
    a, _, _ = session(world, "shilmandb", Switches(), env=env)
    b, _, _ = session(world, "incumbent_bigquery", env=env)
    assert a == b


def test_incumbents_take_no_switches(world, env):
    with pytest.raises(ValueError):
        run_session("incumbent_bigquery", env, Factory(world.big), Factory(world.small), 1, Switches(t1=True))
    with pytest.raises(ValueError):
        run_session("naive", env, Factory(world.big), Factory(world.small), 1)


def test_session_is_deterministic(world, env):
    for approach, switches in COMBOS:
        a, big_a, _ = session(world, approach, switches, seed=4, env=env)
        b, big_b, _ = session(world, approach, switches, seed=4, env=env)
        assert a == b
        assert [o.log for o in big_a.made] == [o.log for o in big_b.made]
        _, big_c, _ = session(world, approach, switches, seed=5, env=env)
        assert [o.log for o in big_a.made] != [o.log for o in big_c.made]


_DIGEST = """
import hashlib, sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
from aiq_world import make_world
from bench.ai_queries.approaches import ALL_ON, Env, run_session
from bench.ai_queries.costs import Weights
from bench.ai_queries.oracle import Oracle
w = make_world(n=3000, seed=0)
env = Env(w.reviews, w.products, w.emb, {k: i for i, k in enumerate(w.emb_keys)}, w.question_emb,
          Weights(0.1, 1e-6, 0.01), {})
runs = run_session("shilmandb", env, lambda: Oracle("big", w.big), lambda: Oracle("small", w.small), 7, ALL_ON)
print(hashlib.sha256(repr(runs).encode()).hexdigest())
"""


def test_session_is_the_same_in_every_process():
    repo = Path(__file__).resolve().parents[3]
    tests = Path(__file__).resolve().parent
    digests = set()
    for hash_seed in ("1", "2"):
        out = subprocess.run([sys.executable, "-c", _DIGEST, str(repo), str(tests)], check=True,
                             capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": hash_seed})
        digests.add(out.stdout.strip())
    assert len(digests) == 1


# ---- what each run claims ----


def test_stated_claims_present(world, env):
    all_runs = []
    for approach, switches in COMBOS:
        runs, _, _ = session(world, approach, switches, env=env)
        all_runs += runs
    modes = {r.mode for r in all_runs}
    assert {"exact", "full"} <= modes
    for r in all_runs:
        s = r.stated
        assert "confidence" in s
        if s["method"] == "estimate":
            assert r.mode == "estimate" and s["confidence"] == 0.99
            assert {"est", "lo", "hi", "looks"} <= set(s) and r.decisions is None
        elif s["method"] == "cascade":
            assert r.mode == "exact" and s["confidence"] is None
            assert "targets" in s
            assert {"precision_est", "recall_est"} <= set(s) or "accuracy_est" in s
            assert r.decisions is not None
        else:
            assert s["method"] == "full" and s["confidence"] == 1.0
            assert r.decisions is not None
        assert isinstance(r.variant, str) and r.variant
    # an estimate run states its interval per group for groups and top 10
    w = make_world(n=3000, seed=0)
    runs, _, _ = session(w, "shilmandb", Switches(t1=True))
    for r in runs:
        if r.mode == "estimate" and queries.QUERIES[r.query].kind in ("groups", "top10"):
            assert isinstance(r.stated["lo"], dict) and set(r.stated["lo"]) == set(r.stated["hi"])


# ---- the embedding pass, split over the queries that used it ----


def _hand_run(no: int, proxy: int) -> QueryRun:
    return QueryRun(no, 0, Usage(big=5, proxy=proxy), "exact", {"method": "full", "confidence": 1.0}, "big", {})


def test_embedding_cost_split(world, env):
    runs = [_hand_run(no, 100 if no in (1, 3, 5, 8) else 0) for no in range(1, 11)]
    split_embedding_rows(runs, 1000)
    assert [r.usage.embed_rows for r in runs] == [250.0, 0, 250.0, 0, 250.0, 0, 0, 250.0, 0, 0]

    embedded = len({k for no in queries.QUERIES for k in distinct_keys(world, no)})
    for approach, switches in COMBOS:
        runs, _, _ = session(world, approach, switches, env=env)
        users = [r for r in runs if r.usage.proxy > 0]
        for r in runs:
            want = embedded / len(users) if r in users else 0.0
            assert r.usage.embed_rows == pytest.approx(want)
        assert sum(r.usage.embed_rows for r in runs) == pytest.approx(embedded if users else 0.0)
        if approach == "incumbent_snowflake":
            assert not users  # small model only: no embeddings anywhere
        else:
            assert users


def test_answers_match_the_truth_when_full(world, env):
    for approach, switches in COMBOS:
        runs, _, _ = session(world, approach, switches, env=env)
        for r in runs:
            if r.mode == "full":
                t = queries.truth(r.query, world.reviews, world.products, world.big, {})
                assert r.answer == t.answer, (approach, r.query)


def _check_claims(runs, approach):
    for r in runs:
        if r.stated["method"] != "cascade" or not r.stated["targets"]:
            continue
        for qid, t in r.stated["targets"].items():
            if "accuracy" in t:
                assert r.stated["accuracy_est"][qid] >= t["accuracy"], (approach, r.query, r.variant)
            else:
                assert r.stated["precision_est"][qid] >= t["precision"], (approach, r.query, r.variant)
                assert r.stated["recall_est"][qid] >= t["recall"], (approach, r.query, r.variant)


def test_cascades_meet_their_own_targets(world, big_world):
    # a plan whose own calibration rows miss the per-row targets is never picked
    for small in (0.01, 0.05):
        e = env_of(world, Weights(small=small, proxy=1e-6, embed=0.01))
        for approach, switches in COMBOS:
            runs, _, _ = session(world, approach, switches, seed=1, env=e)
            _check_claims(runs, approach)
    # here the three-tier cascade is the cheapest for Q3, but its calibration recall is 0.967 < 0.97
    e = env_of(big_world, Weights(small=0.05, proxy=1e-6, embed=0.01))
    runs, _, _ = session(big_world, "shilmandb", Switches(t3=True), seed=1, env=e)
    _check_claims(runs, "shilmandb")
    assert runs[2].variant != "proxy>small>big"


def test_choose_skips_plans_that_miss(world, env, monkeypatch):
    from bench.ai_queries import approaches
    fake = {"proxy>big": (50.0, 0.99), "small>big": (40.0, 0.96), "proxy>small>big": (30.0, 0.90)}

    def plan(ctx, qid, name, r):
        cost, recall = fake[name]
        claim = {"accuracy": recall} if qid == "C2" else {"precision": 0.99, "recall": recall}
        return approaches._Plan(name, (), cost, claim)

    monkeypatch.setattr(approaches, "_plan", plan)
    monkeypatch.setattr(approaches, "_small_may_pay", lambda *a: True)
    runs, _, _ = session(world, "shilmandb", Switches(t3=True), env=env)
    assert runs[0].variant == runs[1].variant == "proxy>big"  # the cheaper two miss their 0.97 target
    fake["proxy>big"] = (50.0, 0.95)
    runs, _, _ = session(world, "shilmandb", Switches(t3=True), env=env)
    assert runs[0].mode == runs[1].mode == "full" and runs[0].variant == "big"  # none qualifies


# ---- T1 decides from free bounds before paying small-model calibration ----


@pytest.fixture(scope="module")
def t1_world():
    old = aiq_world.YES_RATES["F1"]
    aiq_world.YES_RATES["F1"] = 0.45
    try:
        world = make_world(n=40000, seed=1, proxy_signal=0.3)
    finally:
        aiq_world.YES_RATES["F1"] = old
    yield world


def _modes(runs):
    return [r.mode for r in runs]


def test_estimate_wins_without_paying_small_calibration(t1_world, monkeypatch):
    from bench.ai_queries import approaches
    e = env_of(t1_world, Weights(small=0.7, proxy=1e-6, embed=0.01))
    sw = Switches(t1=True, t3=True)
    new, _, _ = session(t1_world, "shilmandb", sw, env=e)
    # the old rule: no free bounds, so the exact cost (and its small calibration) is always worked out
    monkeypatch.setattr(approaches, "_exact_bounds", lambda ctx, order: (0.0, float("inf")))
    old, _, _ = session(t1_world, "shilmandb", sw, env=e)
    assert new[0].mode == old[0].mode == "estimate"
    assert old[0].usage.small == 600 and new[0].usage.small == 0
    assert new[0].usage.big == old[0].usage.big
    assert _modes(new) == _modes(old)


@pytest.mark.parametrize("small_weight, seed", [(0.1, 1), (0.7, 2)])
def test_free_bounds_decide_like_the_exact_cost(t1_world, monkeypatch, small_weight, seed):
    from bench.ai_queries import approaches
    e = env_of(t1_world, Weights(small=small_weight, proxy=1e-6, embed=0.01))
    sw = Switches(t1=True, t3=True)
    new, _, _ = session(t1_world, "shilmandb", sw, seed=seed, env=e)
    monkeypatch.setattr(approaches, "_exact_bounds", lambda ctx, order: (0.0, float("inf")))
    old, _, _ = session(t1_world, "shilmandb", sw, seed=seed, env=e)
    assert _modes(new) == _modes(old)
