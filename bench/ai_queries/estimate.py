#Sampled estimates with intervals that stay valid while the sample grows

from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Callable, Sequence

import numpy as np

K_STRATA = 4
TOP_GROUPS = 20


@dataclass(frozen=True)
class Interval:
    est: float
    lo: float
    hi: float


def look_sizes(n_first: int, population: int) -> list[int]:
    sizes, n = [], n_first
    while n < population:
        sizes.append(n)
        n *= 2
    return [*sizes, population] if population > 0 else []


def look_delta(delta: float, k: int) -> float:
    return delta / 2 ** k


def _z(delta: float) -> float:
    return NormalDist().inv_cdf(1 - delta / 2)


def neyman_allocation(N_k: np.ndarray, p_tilde: np.ndarray, n_total: int, n_have: np.ndarray) -> np.ndarray:
    N_k = np.asarray(N_k, dtype=float)
    have = np.asarray(n_have, dtype=float)
    room = N_k - have
    want = min(max(n_total - have.sum(), 0.0), room.sum())
    w = N_k * np.sqrt(np.clip(p_tilde * (1 - p_tilde), 0, None))
    if w.sum() <= 0:
        w = N_k.copy()

    def take(lam: float) -> np.ndarray:
        return np.minimum(np.maximum(lam * w - have, 0.0), room)

    lo, hi = 0.0, 1.0
    while take(hi).sum() < want and hi < 1e18:
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if take(mid).sum() < want else (lo, mid)
        
    raw = take(hi)
    extra = np.floor(raw).astype(int)
    left = int(round(want)) - int(extra.sum())
    order = np.argsort(-(raw - extra), kind="stable")
    for k in order:  # hand out the leftover rows by largest fraction, where there is room
        if left <= 0:
            break
        if extra[k] < room[k]:
            extra[k] += 1
            left -= 1
            
    return extra


def count_interval(N_k, x_k, n_k, delta: float) -> Interval:
    N, x, n = (np.asarray(a, dtype=float) for a in (N_k, x_k, n_k))
    n_eff = np.maximum(n, 1.0)
    est = float(np.sum(N * np.where(n > 0, x / n_eff, 0.5)))
    pt = (x + 2) / (n + 4)
    var = float(np.sum(N ** 2 * pt * (1 - pt) / n_eff * (1 - n / np.maximum(N, 1.0))))
    hw = _z(delta) * np.sqrt(var)
    return Interval(est, max(0.0, est - hw), est + hw)


def group_intervals(N_k, x_kg: dict[str, np.ndarray], n_k, delta: float, comparisons: int | None = None) -> dict[str, Interval]:
    share = delta / max(comparisons or len(x_kg), 1)
    return {g: count_interval(N_k, x, n_k, share) for g, x in x_kg.items()}


def ratio_interval(N_k, sum_y_k, sum_yy_k, x_k, sum_xy_k, n_k, delta: float, known_y: float = 0.0, known_x: float = 0.0) -> Interval | None:
    N, sy, syy, x, sxy, n = (np.asarray(a, dtype=float) for a in (N_k, sum_y_k, sum_yy_k, x_k, sum_xy_k, n_k))
    n_eff = np.maximum(n, 1.0)
    Y = float(np.sum(N * sy / n_eff)) + known_y
    X = float(np.sum(N * x / n_eff)) + known_x
    if X <= 0:
        if np.all(n >= N):
            return None  # every row seen and none matched: there is no average
        return Interval(3.0, 1.0, 5.0)  # nothing matched yet: as wide as the rating scale
    
    R = Y / X
    se2 = syy - 2 * R * sxy + R * R * x  # sum of squared residuals (x is 0 or 1)
    se = sy - R * x
    s2 = np.maximum(se2 - se ** 2 / n_eff, 0.0) + 4.0
    s2 = s2 / (n + 1)
    var = float(np.sum(N ** 2 * (1 - n / np.maximum(N, 1.0)) * s2 / n_eff)) / (X * X)
    hw = _z(delta) * np.sqrt(var)
    return Interval(R, R - hw, R + hw)


def _half(iv: Interval) -> float:
    return iv.hi - iv.est


def count_done(iv: Interval) -> bool:
    hw = _half(iv)
    return hw <= max(0.05 * (iv.est - hw), 10)


def groups_done(ivs: dict[str, Interval]) -> bool:
    return all(count_done(iv) for iv in ivs.values())


def average_done(iv: Interval | None) -> bool:
    return iv is not None and (iv.hi - iv.lo) / 2 <= 0.05


def _ranked(ivs: dict[str, Interval]) -> list[tuple[str, Interval]]:
    return sorted(ivs.items(), key=lambda kv: (-kv[1].est, kv[0]))


#The 10th group's lower bound beats the upper bound of ranks 11-20 and of any group not yet seen.
def top10_done(ivs: dict[str, Interval], unseen_hi: float = 0.0, complete: bool = True) -> bool:
    ranked = _ranked(ivs)
    if len(ranked) <= 10:
        return complete
    
    rest = max([iv.hi for _, iv in ranked[10:]] + [unseen_hi])
    return ranked[9][1].lo > rest


class Sampler:

    def __init__(self, strata: np.ndarray, rng: np.random.Generator, taken: Sequence[int] = ()):
        self.k = int(strata.max()) + 1 if len(strata) else 0
        self.N_k = np.bincount(strata, minlength=self.k)
        gone = np.zeros(len(strata), dtype=bool)
        gone[np.asarray(taken, dtype=int)] = True
        self._queue = [rng.permutation(np.flatnonzero((strata == k) & ~gone)) for k in range(self.k)]
        self._at = [0] * self.k
        self.n_k = np.bincount(strata[gone], minlength=self.k)

    def draw(self, n_extra_per_stratum) -> list[int]:
        rows: list[int] = []
        for k, n in enumerate(np.asarray(n_extra_per_stratum, dtype=int)):
            take = self._queue[k][self._at[k]:self._at[k] + max(int(n), 0)]
            self._at[k] += len(take)
            self.n_k[k] += len(take)
            rows.extend(int(i) for i in take)
            
        return rows


@dataclass(frozen=True)
class EstimateResult:
    kind: str
    interval: Interval | None = None            # count, average
    intervals: dict[str, Interval] = field(default_factory=dict)  # groups: every class; top10: the top 20
    top10: list[str] = field(default_factory=list)
    rows_asked: int = 0                          # rows asked of the big model (training rows not counted)
    looks: int = 0
    unseen_hi: float = 0.0                       # top10: upper bound for every group outside the top 20, and for one never sampled
    stopped: str = "rule"                        # "rule", "census" or "budget"
    outside: tuple[tuple[float, float, float], ...] = ()  # top10: (estimate, half-width, exact count) per group outside the top 20, and one never sampled


def _strata_from_scores(scores: np.ndarray) -> np.ndarray:
    rank = np.empty(len(scores), dtype=int)
    rank[np.argsort(scores, kind="stable")] = np.arange(len(scores))
    return rank * K_STRATA // max(len(scores), 1)


#Run the looks until the interval is tight, the population is covered or the budget is spent
def run_estimate(kind: str, scores, known, answer: Callable, delta: float = 0.01, n_first: int = 600,
                 rng: np.random.Generator | None = None, max_rows: int | None = None,
                 first: Sequence[int] | None = None, groups: Sequence | None = None,
                 n_train: int = 0) -> EstimateResult:
    
    rng = rng if rng is not None else np.random.default_rng(0)
    scores = np.asarray(scores, dtype=float)
    M = len(scores)
    if M == 0:
        return _from_known(kind, known, groups)
    
    strata = _strata_from_scores(scores)
    sizes = [min(max(s - n_train, 0), M) for s in look_sizes(n_first, M + n_train)]
    
    if first is None:
        first = np.sort(rng.choice(M, size=max(sizes[0], 1), replace=False))
    first = [int(i) for i in first]
    
    sampler = Sampler(strata, rng, first)
    N_k = sampler.N_k.astype(float)
    state = _State(kind, sampler.k, groups)
    state.add(first, strata, answer(np.asarray(first, dtype=int)))
    asked = len(first)

    result = None
    for look, target in enumerate(sizes, start=1):
        if look > 1:
            if max_rows is not None and target > max_rows:
                return _finish(result, "budget")
            
            extra = neyman_allocation(N_k, state.p_tilde(sampler.n_k), target, sampler.n_k)
            if extra.sum() > 0:
                rows = sampler.draw(extra)
                state.add(rows, strata, answer(np.asarray(rows, dtype=int)))
                asked += len(rows)
                
        result = state.result(N_k, sampler.n_k, look_delta(delta, look), known, asked, look)
        if np.array_equal(sampler.n_k, sampler.N_k):
            return _finish(result, "census")
        
        if state.done(result):
            return _finish(result, "rule")
        
        if len(sizes) == 1 and target > len(first):
            if max_rows is not None and target > max_rows:
                return _finish(result, "budget")
            
            extra = neyman_allocation(N_k, state.p_tilde(sampler.n_k), target, sampler.n_k)
            rows = sampler.draw(extra)
            state.add(rows, strata, answer(np.asarray(rows, dtype=int)))
            asked += len(rows)
            result = state.result(N_k, sampler.n_k, look_delta(delta, look), known, asked, look)
            if np.array_equal(sampler.n_k, sampler.N_k):
                return _finish(result, "census")
            
    raise RuntimeError("the looks ended before every row was asked")


#How many non-training rows the stopping rule will likely need, judged from look 1 alone
def projected_sample(first: EstimateResult, population: int, n_train: int = 0, delta: float = 0.01, n_first: int = 600, complete: bool = True) -> int:
    M, n1 = population, first.rows_asked
    if n1 >= M:
        return M
    
    if n1 <= 0:
        return M  # nothing seen yet, nothing to project from
    
    G = {"groups": max(len(first.intervals), 1), "top10": TOP_GROUPS}.get(first.kind, 1)
    sizes = [min(max(s - n_train, 0), M) for s in look_sizes(n_first, M + n_train)]
    z1 = _z(look_delta(delta, 1) / G)
    base = 1 / n1 - 1 / M
    
    for look, n_k in enumerate(sizes, start=1):
        n_k = n1 if look == 1 else max(n_k, n1)
        r = _z(look_delta(delta, look) / G) / z1 * np.sqrt(max(1 / n_k - 1 / M, 0.0) / base)
        if _passes(first, r, complete):
            return n_k
    return M


def _scaled(iv: Interval, r: float) -> Interval:
    hw = iv.hi - iv.est  # a count's lower end may be cut at 0, its upper end never is
    return Interval(iv.est, iv.est - r * hw, iv.est + r * hw)


def _passes(first: EstimateResult, r: float, complete: bool) -> bool:
    if first.kind == "count":
        return count_done(_scaled(first.interval, r))
    
    if first.kind == "average":
        return first.interval is None or average_done(_scaled(first.interval, r))
    
    ivs = {g: _scaled(iv, r) for g, iv in first.intervals.items()}
    if first.kind == "groups":
        return groups_done(ivs)
    
    unseen = max((e + r * hw + c for e, hw, c in first.outside), default=0.0)
    return top10_done(ivs, unseen, complete)


#No rows left to sample: the training rows' exact answers are the whole answer
def _from_known(kind: str, known, groups) -> EstimateResult:
    if kind == "count":
        return EstimateResult(kind, Interval(float(known), float(known), float(known)), stopped="census")
    
    if kind == "average":
        ky, kx = known
        iv = Interval(ky / kx, ky / kx, ky / kx) if kx > 0 else None
        return EstimateResult(kind, iv, stopped="census")
    
    labels = sorted({str(g) for g in known} | {str(g) for g in (groups or [])})
    ivs = {g: Interval(float(known.get(g, 0.0)), float(known.get(g, 0.0)), float(known.get(g, 0.0))) for g in labels}
    if kind == "top10":
        ranked = _ranked(ivs)[:TOP_GROUPS]
        return EstimateResult(kind, intervals=dict(ranked), top10=[g for g, _ in ranked[:10]], stopped="census")
    
    return EstimateResult(kind, intervals=ivs, stopped="census")


def _finish(result: EstimateResult, stopped: str) -> EstimateResult:
    return EstimateResult(result.kind, result.interval, result.intervals, result.top10, result.rows_asked, result.looks, result.unseen_hi, stopped, result.outside)


class _State:

    def __init__(self, kind: str, k: int, groups: Sequence | None):
        if kind not in ("count", "groups", "average", "top10"):
            raise ValueError(f"unknown kind {kind!r}")
        
        self.kind, self.k = kind, k
        self.x = np.zeros(k)
        self.sy = np.zeros(k)
        self.syy = np.zeros(k)
        self.sxy = np.zeros(k)
        self.cls: dict[str, np.ndarray] = {str(g): np.zeros(k) for g in (groups or [])}
        self.n_groups = len(self.cls)
        self.complete = groups is not None

    def add(self, rows: Sequence[int], strata: np.ndarray, values) -> None:
        if not len(rows):
            return
        
        st = strata[np.asarray(rows, dtype=int)]
        if self.kind == "count":
            self.x += np.bincount(st, weights=np.asarray(values, dtype=float), minlength=self.k)
        elif self.kind == "average":
            v = np.asarray(values, dtype=float)
            x, y = v[:, 0], v[:, 0] * v[:, 1]
            self.x += np.bincount(st, weights=x, minlength=self.k)
            self.sy += np.bincount(st, weights=y, minlength=self.k)
            self.syy += np.bincount(st, weights=y * y, minlength=self.k)
            self.sxy += np.bincount(st, weights=x * y, minlength=self.k)
        else:
            hit = np.array([v is not None for v in values], dtype=bool)
            if not hit.any():
                return
            
            labels = np.array([str(v) for v, h in zip(values, hit) if h])
            uniq, inv = np.unique(labels, return_inverse=True)
            counts = np.bincount(inv * self.k + st[hit], minlength=len(uniq) * self.k).reshape(len(uniq), self.k)
            for lab, row in zip(uniq, counts):
                self.cls.setdefault(str(lab), np.zeros(self.k))
                self.cls[str(lab)] += row

    def p_tilde(self, n_k: np.ndarray) -> np.ndarray:
        n = np.asarray(n_k, dtype=float)
        if self.kind in ("count", "average"):
            return (self.x + 2) / (n + 4)
        
        if not self.cls:
            return np.full(self.k, 0.5)
        
        v = np.mean([((x + 2) / (n + 4)) * (1 - (x + 2) / (n + 4)) for x in self.cls.values()], axis=0)
        return (1 - np.sqrt(np.clip(1 - 4 * v, 0, None))) / 2  # p with p(1-p) = v

    def result(self, N_k, n_k, delta, known, asked, look) -> EstimateResult:
        if self.kind == "count":
            iv = count_interval(N_k, self.x, n_k, delta)
            return EstimateResult("count", _shift(iv, known), rows_asked=asked, looks=look)
        
        if self.kind == "average":
            ky, kx = known
            iv = ratio_interval(N_k, self.sy, self.syy, self.x, self.sxy, n_k, delta, ky, kx)
            return EstimateResult("average", iv, rows_asked=asked, looks=look)
        
        labels = sorted(set(self.cls) | {str(g) for g in known})
        full = {g: self.cls.get(g, np.zeros(self.k)) for g in labels}
        if self.kind == "groups":
            ivs = group_intervals(N_k, full, n_k, delta, comparisons=max(len(labels), self.n_groups))
            return EstimateResult("groups", intervals={g: _shift(iv, known.get(g, 0.0)) for g, iv in ivs.items()}, rows_asked=asked, looks=look)
        
        point = {g: count_interval(N_k, x, n_k, delta).est + known.get(g, 0.0) for g, x in full.items()}
        top = [g for g, _ in sorted(point.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_GROUPS]]
        ivs = group_intervals(N_k, {g: full[g] for g in top}, n_k, delta, comparisons=TOP_GROUPS)
        ivs = {g: _shift(iv, known.get(g, 0.0)) for g, iv in ivs.items()}
        
        outside_known = {g: known.get(g, 0.0) for g in labels if g not in ivs}
        share = delta / TOP_GROUPS
        parts = []
        for g, c in outside_known.items():
            iv = count_interval(N_k, full[g], n_k, share)
            parts.append((iv.est, iv.hi - iv.est, c))
            
        if not self.complete:
            iv = count_interval(N_k, np.zeros(self.k), n_k, share)
            parts.append((iv.est, iv.hi - iv.est, max(outside_known.values(), default=0.0)))
            
        unseen = max((e + hw + c for e, hw, c in parts), default=0.0)
        return EstimateResult("top10", intervals=ivs, top10=[g for g, _ in _ranked(ivs)[:10]], rows_asked=asked, looks=look, unseen_hi=unseen, outside=tuple(parts))

    def done(self, r: EstimateResult) -> bool:
        return {
            "count": lambda: count_done(r.interval),
            "average": lambda: average_done(r.interval),
            "groups": lambda: groups_done(r.intervals),
            "top10": lambda: top10_done(r.intervals, r.unseen_hi, self.complete)
        }[self.kind]()


def _shift(iv: Interval, c: float) -> Interval:
    return Interval(iv.est + c, iv.lo + c, iv.hi + c)
