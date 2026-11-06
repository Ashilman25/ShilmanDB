import numpy as np
import pytest

from bench.ai_queries import estimate as est

N_ROWS = 30000
TRAIN = 300  # rows the strata proxy trained on; their answers are known exactly


def _population(seed, rate, signal):
    """A latent score s; yes-chance rises with s. `signal` sets how informative s is."""
    rng = np.random.default_rng(seed)
    s = rng.normal(size=N_ROWS)
    lo, hi = -30.0, 30.0
    for _ in range(60):  # pick the intercept so the average chance is `rate`
        b = (lo + hi) / 2
        mean = (1 / (1 + np.exp(-(signal * s + b)))).mean()
        lo, hi = (b, hi) if mean < rate else (lo, b)
    y = (rng.random(N_ROWS) < 1 / (1 + np.exp(-(signal * s + b)))).astype(int)
    return s, y


def _run(seed, kind, scores, known, answer, **kw):
    return est.run_estimate(kind, scores, known, answer, rng=np.random.default_rng(seed), **kw)


def test_look_schedule():
    assert est.look_sizes(600, 5000) == [600, 1200, 2400, 4800, 5000]
    assert est.look_sizes(600, 400) == [400]
    assert sum(est.look_delta(0.01, k) for k in range(1, 40)) < 0.01


def test_neyman_allocation():
    N = np.array([1000, 1000, 1000, 1000])
    p = np.array([0.001, 0.5, 0.5, 0.2])
    have = np.array([50, 50, 50, 50])
    extra = est.neyman_allocation(N, p, 1000, have)
    assert extra.sum() == 800
    assert extra[0] < extra[1]
    assert extra[1] == extra[2]
    # never more than a stratum has left
    small = est.neyman_allocation(np.array([60, 1000]), np.array([0.5, 0.5]), 600, np.array([50, 50]))
    assert small[0] <= 10 and small.sum() == 500
    # asking for more than exists gives what is left
    assert est.neyman_allocation(N, p, 99999, have).sum() == 4 * 950


def test_exhaustive_has_zero_width():
    N = np.array([10, 20, 30, 40])
    iv = est.count_interval(N, np.array([1, 5, 7, 9]), N, 0.01)
    assert iv.lo == iv.est == iv.hi == 22
    iv = est.ratio_interval(N, np.array([4.0, 8, 12, 16]), np.array([16.0, 32, 48, 64]),
                            np.array([1, 2, 3, 4]), np.array([4.0, 8, 12, 16]), N, 0.01)
    assert iv.lo == iv.est == iv.hi


def test_sampler_draws_without_replacement():
    strata = np.repeat(np.arange(4), 25)
    s = est.Sampler(strata, np.random.default_rng(0))
    a = s.draw(np.array([5, 5, 5, 5]))
    b = s.draw(np.array([30, 0, 0, 0]))
    assert len(set(a + b)) == len(a + b) == 40
    assert list(s.n_k) == [25, 5, 5, 5]


def test_stopping_rules():
    assert est.count_done(est.Interval(1000, 960, 1040))
    assert not est.count_done(est.Interval(1000, 800, 1200))
    assert est.count_done(est.Interval(3, 0, 8))  # within 10 rows
    assert est.average_done(est.Interval(3.5, 3.46, 3.54))
    assert not est.average_done(est.Interval(3.5, 3.4, 3.6))
    assert est.groups_done({"a": est.Interval(1000, 970, 1030), "b": est.Interval(5, 0, 9)})
    assert not est.groups_done({"a": est.Interval(1000, 970, 1030), "b": est.Interval(500, 100, 900)})


def _count_cover(seed_pop, rate, signal, runs, delta=0.01):
    s, y = _population(seed_pop, rate, signal)
    scores, known, truth = s[TRAIN:], float(y[:TRAIN].sum()), int(y.sum())
    hits, stops = 0, []
    for r in range(runs):
        res = _run(r, "count", scores, known, lambda idx: y[TRAIN:][idx], delta=delta, n_train=TRAIN)
        hits += res.interval.lo <= truth <= res.interval.hi
        stops.append(res.stopped)
    return hits, stops


def test_count_coverage_with_stopping():
    hits, stops = _count_cover(1, 0.05, 2.0, 200)
    assert hits >= 194
    assert stops.count("rule") > 100  # the rule, not a census, ended most runs


def test_rare_count_does_not_stop_early():
    hits, stops = _count_cover(2, 0.002, 0.3, 100)
    assert hits >= 97


def test_groups_coverage():
    rng = np.random.default_rng(3)
    s = rng.normal(size=N_ROWS)
    w = np.exp(np.outer(s, [0.8, 0.4, 0.0, -0.4, -0.8, 0.2]) + np.array([0.0, 0.5, 0.8, 0.3, -0.5, -1.0]))
    p = w / w.sum(axis=1, keepdims=True)
    cls = (rng.random(N_ROWS)[:, None] > p.cumsum(axis=1)).sum(axis=1).clip(max=5)
    truth = np.bincount(cls, minlength=6)
    known = {str(c): float(v) for c, v in enumerate(np.bincount(cls[:TRAIN], minlength=6))}
    hits = 0
    for r in range(200):
        res = _run(r, "groups", s[TRAIN:], known, lambda idx: cls[TRAIN:][idx], groups=list(range(6)), n_train=TRAIN)
        hits += all(res.intervals[str(c)].lo <= truth[c] <= res.intervals[str(c)].hi for c in range(6))
    assert hits >= 194


def test_ratio_coverage():
    rng = np.random.default_rng(4)
    s = rng.normal(size=N_ROWS)
    match = (rng.random(N_ROWS) < 1 / (1 + np.exp(-(1.0 * s - 2.2)))).astype(int)
    rating = np.clip(np.round(3.5 + 0.5 * s + rng.normal(scale=0.7, size=N_ROWS)), 1, 5)
    truth = rating[match == 1].mean()
    known = (float((rating * match)[:TRAIN].sum()), float(match[:TRAIN].sum()))
    vals = np.column_stack([match, rating])[TRAIN:]
    hits, rule = 0, 0
    for r in range(200):
        res = _run(r, "average", s[TRAIN:], known, lambda idx: vals[idx], n_train=TRAIN)
        hits += res.interval.lo <= truth <= res.interval.hi
        rule += res.stopped == "rule"
    assert hits >= 194
    assert rule > 100


def _groups_answer(sizes, seed):
    rng = np.random.default_rng(seed)
    names = np.concatenate([np.full(n, f"g{i:02d}") for i, n in enumerate(sizes)])
    rng.shuffle(names)
    return names


def test_top10_stops_only_when_separated():
    sizes = [6000 - 450 * i for i in range(10)] + [900] * 10  # ten big, ten small
    names = _groups_answer(sizes, 5)
    res = _run(0, "top10", np.random.default_rng(1).normal(size=len(names)), {}, lambda idx: names[idx])
    assert res.stopped == "rule" and res.rows_asked < len(names)
    assert set(res.top10) == {f"g{i:02d}" for i in range(10)}
    # ten groups tied with the 11th: never separated before the population
    tied = _groups_answer([2000] * 11 + [500] * 3, 6)
    res = _run(0, "top10", np.random.default_rng(2).normal(size=len(tied)), {}, lambda idx: tied[idx])
    assert res.stopped == "census" and res.rows_asked == len(tied)
    assert not est.top10_done({f"g{i}": est.Interval(100, 90, 110) for i in range(11)})


def test_budget_stop_reports_budget():
    s, y = _population(1, 0.05, 1.2)
    res = _run(0, "count", s[TRAIN:], 0.0, lambda idx: y[TRAIN:][idx], max_rows=700)
    assert res.stopped == "budget" and res.rows_asked <= 700


def test_driver_is_deterministic():
    s, y = _population(1, 0.05, 1.2)
    a = _run(7, "count", s[TRAIN:], 0.0, lambda idx: y[TRAIN:][idx])
    b = _run(7, "count", s[TRAIN:], 0.0, lambda idx: y[TRAIN:][idx])
    assert a == b


def test_first_path_look_one_asks_nothing_new():
    s, y = _population(1, 0.05, 0.2)
    scores, ans = s[TRAIN:], y[TRAIN:]
    first = np.sort(np.random.default_rng(9).choice(len(scores), size=300, replace=False))
    calls = []

    def answer(idx):
        calls.append(list(idx))
        return ans[idx]

    # look 1 is exactly `first`
    res = _run(0, "count", scores, 0.0, answer, first=first, n_train=TRAIN, max_rows=300)
    assert calls == [list(first)] and res.looks == 1 and res.rows_asked == 300 and res.stopped == "budget"
    # look 2 tops up to 1,200 big-labeled rows in all, 300 of them training rows
    calls.clear()
    res = _run(0, "count", scores, 0.0, answer, first=first, n_train=TRAIN, max_rows=900)
    assert calls[0] == list(first)
    asked = [i for c in calls for i in c]
    assert len(asked) == len(set(asked)) == res.rows_asked == 900 and res.looks == 2


def _top10_population(seed):
    rng = np.random.default_rng(seed)
    sizes = [500 - 25 * i for i in range(10)] + [120 - 8 * i for i in range(15)]
    labels = np.concatenate([np.full(n, i) for i, n in enumerate(sizes)])
    names = np.full(20000, None, dtype=object)  # most rows match nothing
    slots = rng.permutation(20000)[:len(labels)]
    names[slots] = [f"m{i:02d}" for i in labels]
    score = (names != None).astype(float) + rng.normal(scale=0.8, size=20000)  # noqa: E711
    return names, score


def test_top10_correct_with_a_non_match_majority():
    names, score = _top10_population(11)
    counts = {}
    for v in names:
        if v is not None:
            counts[v] = counts.get(v, 0) + 1
    true_top = {g for g, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]}
    ok, by_rule = 0, 0
    for r in range(100):
        res = _run(r, "top10", score, {}, lambda idx: names[idx])
        if res.stopped == "rule":
            by_rule += 1
            ok += len(set(res.top10) & true_top) >= 9
    assert by_rule >= 50
    assert ok >= 0.97 * by_rule


def test_few_labels_seen_is_not_done_without_the_label_set():
    names = np.full(3000, None, dtype=object)
    for m in range(10):
        names[m * 300:m * 300 + 5] = f"model{m}"
    score = np.random.default_rng(5).normal(size=3000)
    for r in range(50):
        res = _run(r, "top10", score, {}, lambda idx: names[idx], n_first=300)
        assert res.stopped == "census"
    # with the full label set given, 10 groups need no separating
    res = _run(0, "top10", score, {}, lambda idx: names[idx], n_first=300, groups=[f"model{m}" for m in range(10)])
    assert res.stopped == "rule"


def test_empty_scope_uses_the_known_rows():
    none = np.array([])
    res = _run(0, "count", none, 17.0, lambda idx: [])
    assert (res.interval.lo, res.interval.est, res.interval.hi) == (17, 17, 17) and res.stopped == "census"
    res = _run(0, "groups", none, {"a": 3.0, "b": 4.0}, lambda idx: [])
    assert res.intervals["a"].est == res.intervals["a"].hi == 3 and res.stopped == "census"
    res = _run(0, "top10", none, {"a": 3.0, "b": 4.0}, lambda idx: [])
    assert res.top10 == ["b", "a"]
    res = _run(0, "average", none, (10.0, 4.0), lambda idx: [])
    assert res.interval.est == 2.5
    assert _run(0, "average", none, (0.0, 0.0), lambda idx: []).interval is None


def test_average_with_no_matches_at_census_is_none():
    vals = np.zeros((50, 2))
    res = _run(0, "average", np.arange(50.0), (0.0, 0.0), lambda idx: vals[idx])
    assert res.stopped == "census" and res.interval is None


def test_unseen_group_blocks_top10():
    # 11 well-separated seen groups, but a sample this small cannot rule out a bigger unseen one
    ivs = {f"g{i}": est.Interval(4000 - 300 * i, 3950 - 300 * i, 4050 - 300 * i) for i in range(11)}
    assert est.top10_done(ivs)
    assert not est.top10_done(ivs, unseen_hi=1500)


# ---- projecting the sample a look-1 interval would need ----


def _look1_and_full(kind, scores, known, answer, seed=0, **kw):
    first = np.sort(np.random.default_rng(seed + 100).choice(len(scores), size=300, replace=False))
    one = _run(seed, kind, scores, known, answer, first=first, n_train=TRAIN, max_rows=300, **kw)
    full = _run(seed, kind, scores, known, answer, first=first, n_train=TRAIN, **kw)
    return one, full


def test_projection_formula_by_hand():
    # look 1: 300 rows, hw 100 around 1,000 at z(delta / 2); M = 30,000
    one = est.EstimateResult("count", est.Interval(1000, 900, 1100), rows_asked=300, looks=1)
    z = lambda k: est._z(est.look_delta(0.01, k))  # noqa: E731
    sizes = [300, 900, 2100, 4500, 9300, 18900, 29700, 30000]  # 600 * 2^(k-1) - 300 training rows
    need = None
    for k, n in enumerate(sizes, start=1):
        hw = 100 * z(k) / z(1) * np.sqrt((1 / n - 1 / 30000) / (1 / 300 - 1 / 30000))
        if hw <= max(0.05 * (1000 - hw), 10):
            need = n
            break
    assert need == est.projected_sample(one, 30000, n_train=TRAIN) == 2100
    # already tight at look 1: nothing more
    tight = est.EstimateResult("count", est.Interval(1000, 980, 1020), rows_asked=300, looks=1)
    assert est.projected_sample(tight, 30000, n_train=TRAIN) == 300
    # nothing sampled, or a covered population
    assert est.projected_sample(est.EstimateResult("count", est.Interval(0, 0, 50)), 5000) == 5000
    assert est.projected_sample(est.EstimateResult("count", est.Interval(9, 9, 9), rows_asked=40), 40) == 40


def test_projection_is_near_the_real_stop():
    s, y = _population(1, 0.05, 2.0)
    for seed in range(5):
        one, full = _look1_and_full("count", s[TRAIN:], float(y[:TRAIN].sum()), lambda idx: y[TRAIN:][idx], seed)
        need = est.projected_sample(one, len(s) - TRAIN, n_train=TRAIN)
        assert full.stopped == "rule"
        assert full.rows_asked / 2.5 <= need <= full.rows_asked * 2.5, seed


def test_projection_of_a_tied_top10_is_the_population():
    tied = _groups_answer([2000] * 11 + [500] * 3, 6)
    scores = np.random.default_rng(2).normal(size=len(tied))
    one = _run(0, "top10", scores, {}, lambda idx: tied[idx], max_rows=600)
    assert est.projected_sample(one, len(tied)) == len(tied)
    # well separated groups stop early, and the projection sees it
    sizes = [6000 - 450 * i for i in range(10)] + [900] * 10
    names = _groups_answer(sizes, 5)
    scores = np.random.default_rng(1).normal(size=len(names))
    one = _run(0, "top10", scores, {}, lambda idx: names[idx], max_rows=600)
    full = _run(0, "top10", scores, {}, lambda idx: names[idx])
    assert est.projected_sample(one, len(names)) <= 2 * full.rows_asked < len(names)


def test_projection_for_groups_and_average():
    rng = np.random.default_rng(4)
    s = rng.normal(size=N_ROWS)
    match = (rng.random(N_ROWS) < 1 / (1 + np.exp(-(1.0 * s - 2.2)))).astype(int)
    rating = np.clip(np.round(3.5 + 0.5 * s + rng.normal(scale=0.7, size=N_ROWS)), 1, 5)
    known = (float((rating * match)[:TRAIN].sum()), float(match[:TRAIN].sum()))
    vals = np.column_stack([match, rating])[TRAIN:]
    one, full = _look1_and_full("average", s[TRAIN:], known, lambda idx: vals[idx])
    need = est.projected_sample(one, N_ROWS - TRAIN, n_train=TRAIN)
    assert full.stopped == "rule" and full.rows_asked / 2.5 <= need <= full.rows_asked * 2.5
    cls = rng.integers(0, 6, N_ROWS)
    gknown = {str(c): float(v) for c, v in enumerate(np.bincount(cls[:TRAIN], minlength=6))}
    one, full = _look1_and_full("groups", s[TRAIN:], gknown, lambda idx: cls[TRAIN:][idx], groups=list(range(6)))
    need = est.projected_sample(one, N_ROWS - TRAIN, n_train=TRAIN)
    assert full.rows_asked / 2.5 <= need <= full.rows_asked * 2.5


# ---- a single look bigger than `first`, and groups ranked below the top 20 ----


def test_single_look_is_a_real_census_or_a_budget_stop():
    y = (np.arange(250) % 5 == 0).astype(int)
    scores = np.arange(250.0)
    first = np.arange(0, 250, 2.5).astype(int)[:100]
    asked = []

    def answer(idx):
        asked.extend(int(i) for i in idx)
        return y[idx]

    res = _run(0, "count", scores, 0.0, answer, first=first)
    assert res.stopped == "census" and res.rows_asked == 250 and sorted(asked) == list(range(250))
    assert res.interval.est == res.interval.lo == res.interval.hi == 50
    res = _run(0, "count", scores, 0.0, lambda idx: y[idx], first=first, max_rows=100)
    assert res.stopped == "budget" and res.rows_asked == 100 and res.looks == 1


def _lightly_sampled_state():
    # stratum 0 is fully asked; stratum 3 is big and lightly sampled
    N_k, n_k = np.array([500.0, 50.0, 50.0, 1000.0]), np.array([500.0, 50.0, 50.0, 100.0])
    st = est._State("top10", 4, None)
    for i in range(10):  # the top 10: many hits in the fully asked stratum
        st.cls[f"a{i:02d}"] = np.array([114.0 + i, 0, 0, 0])
    for i in range(10):  # ranks 11-20: the same few hits each (a tie with z, which sorts last)
        st.cls[f"b{i:02d}"] = np.array([10.0, 0, 0, 0])
    st.cls["z"] = np.array([0, 0, 0, 1.0])  # ranks 21st: one hit in the thin stratum
    return st, N_k, n_k


def test_group_below_the_top_20_can_block_top10():
    st, N_k, n_k = _lightly_sampled_state()
    res = st.result(N_k, n_k, 0.005, {}, 0, 1)
    assert "z" not in res.intervals and len(res.intervals) == 20
    assert est.top10_done(res.intervals, 0.0, True)  # a rest of ranks 11-20 alone would pass
    assert res.unseen_hi > res.intervals["a00"].lo
    assert not est.top10_done(res.intervals, res.unseen_hi, True)
    assert not st.done(res)


def test_multi_look_short_first_still_stops_by_rule():
    # look 1 is exactly `first` (250 rows, short of 300); it already pins an all-yes count down
    first = np.arange(0, 5000, 20)
    res = _run(0, "count", np.arange(5000.0), 0.0, lambda idx: np.ones(len(idx), dtype=int),
               first=first, n_train=TRAIN, max_rows=250)
    assert res.stopped == "rule" and res.rows_asked == 250 and res.looks == 1


def _first_with_outside(ivs, outside):
    return est.EstimateResult("top10", intervals=ivs, outside=outside, unseen_hi=max(e + h + c for e, h, c in outside))


def _ten_over_twenty(tenth_est):
    ivs = {f"a{i}": est.Interval(100.0, 80.0, 120.0) for i in range(9)}
    ivs["a9"] = est.Interval(tenth_est, tenth_est - 10, tenth_est + 10)
    ivs.update({f"b{i}": est.Interval(5.0, 4.0, 6.0) for i in range(10)})
    return ivs


def test_projection_keeps_an_outside_groups_estimate_and_count():
    # an outside group with est 50 and half-width 10: its bound at r = 0.3 is 53, not 18
    first = _first_with_outside(_ten_over_twenty(40.0), ((50.0, 10.0, 0.0),))
    assert not est._passes(first, 0.3, True)  # the 10th group's lower bound (37) is under 53
    assert est._passes(_first_with_outside(_ten_over_twenty(60.0), ((50.0, 10.0, 0.0),)), 0.3, True)
    # the max is taken after scaling: A (hw 60) shrinks to 30 at r = 0.5, B stays at 52.5
    two = _first_with_outside(_ten_over_twenty(50.0), ((0.0, 60.0, 0.0), (0.0, 5.0, 50.0)))
    assert not est._passes(two, 0.5, True)
    assert est._passes(_first_with_outside(_ten_over_twenty(60.0), ((0.0, 60.0, 0.0), (0.0, 5.0, 50.0))), 0.5, True)


def test_result_records_each_outside_bound():
    st, N_k, n_k = _lightly_sampled_state()
    res = st.result(N_k, n_k, 0.005, {}, 0, 1)
    assert res.unseen_hi == max(e + h + c for e, h, c in res.outside)
