from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from .cascade import fit_class_cutoff, fit_cutoffs, route
from .costs import Usage, Weights
from .estimate import EstimateResult, projected_sample, run_estimate
from .oracle import Oracle
from .proxy import Proxy
from .queries import QUERIES, ROW_LIMIT, RowTargets, answer_from, normalize_model, scope
from .questions import C2_CLASSES, QUESTIONS

CALIBRATION = 600  # distinct keys big-labeled per query
TRAIN = 300        # of those, the first ones drawn train the proxy; the rest set cut-offs
BATCH = 64         # LIMIT: keys per small-model batch
SMALL_YES = 0.5    # LIMIT: small p(yes) at or above this goes to the big model
DELTA = 0.01
CONFIDENCE = 0.99
APPROACHES = ("incumbent_snowflake", "incumbent_bigquery", "shilmandb")

_CLASSES = np.array(C2_CLASSES, dtype=object)
_TIERS = {
    "proxy>big": ("proxy",), 
    "small>big": ("small",), 
    "proxy>small>big": ("proxy", "small")
}


@dataclass(frozen=True)
class Switches:
    t1: bool = False
    t2: bool = False
    t3: bool = False
    t4: bool = False


ALL_ON = Switches(True, True, True, True)


@dataclass(frozen=True, eq=False)
class Env:
    reviews: pd.DataFrame
    products: pd.DataFrame
    emb: np.ndarray
    emb_row: Mapping[str, int]
    question_emb: Mapping[str, np.ndarray]
    weights: Weights
    aliases: Mapping[str, str]
    _cache: dict = field(default_factory=dict, init=False, repr=False)  # scopes, built once


@dataclass
class QueryRun:
    query: int
    answer: Any
    usage: Usage
    mode: str      # "exact", "estimate" or "full"
    stated: dict   # what the run claims about its own accuracy
    variant: str   # the cascade used, e.g. "proxy>big"
    decisions: Mapping[str, Mapping[str, str]] | None = None  # qid -> text key -> final answer


# per-query data that does not depend on any label 


@dataclass(frozen=True)
class _Scope:
    df: pd.DataFrame
    keys: np.ndarray       # distinct text keys, in order of first appearance
    row_key: np.ndarray    # each row's position in `keys`
    mult: np.ndarray       # rows per key
    first_row: np.ndarray  # each key's first row
    emb_idx: np.ndarray    # each key's row in env.emb


def _scope_of(env: Env, no: int) -> _Scope:
    if no not in env._cache:
        df = scope(no, env.reviews, env.products)
        codes, uniques = pd.factorize(df["text_key"], sort=False)
        keys = np.asarray(uniques, dtype=object)
        _, first_row = np.unique(codes, return_index=True)
        emb_idx = np.fromiter((env.emb_row[k] for k in keys), dtype=np.int64, count=len(keys))
        env._cache[no] = _Scope(df, keys, codes.astype(np.int64), np.bincount(codes, minlength=len(keys)), first_row, emb_idx)
        
    return env._cache[no]


#The session's embedding pass: every distinct text key in any query's scope
def _embedded_rows(env: Env) -> int:
    if "embedded" not in env._cache:
        env._cache["embedded"] = len({k for no in QUERIES for k in _scope_of(env, no).keys})
    return env._cache["embedded"]


# tiers: an oracle plus what this approach already read from it ----


class _Tier:
    def __init__(self, oracle: Oracle):
        self.oracle = oracle
        self.answers: dict[str, dict[str, str]] = {}
        self._probs: dict[str, dict[str, np.ndarray]] = {}

    def ask(self, qid: str, keys) -> np.ndarray:
        have = self.answers.setdefault(qid, {})
        new = [k for k in dict.fromkeys(keys) if k not in have]
        if new:
            have.update(self.oracle.ask(qid, new))
        return np.array([have[k] for k in keys], dtype=object)

    def probs(self, qid: str, keys) -> np.ndarray:
        have = self._probs.setdefault(qid, {})
        new = [k for k in dict.fromkeys(keys) if k not in have]
        if new:
            have.update(self.oracle.probs(qid, new))
        width = len(QUESTIONS[qid].options)
        return np.array([have[k] for k in keys], dtype=float).reshape(len(keys), width)

    def has_probs(self, qid: str, keys) -> int:
        have = self._probs.get(qid, {})
        return sum(k in have for k in keys)


@dataclass(frozen=True)
class _Policy:
    cascades: tuple[str, ...]  # tried in this order; ties go to the earlier one
    limit: str                 # Q4: "small" batches, "stream" or "rank"
    t1: bool
    t4: bool


def _policy(approach: str, switches: Switches) -> _Policy:
    if approach not in APPROACHES:
        raise ValueError(f"unknown approach {approach!r}")
    
    if approach != "shilmandb" and switches != Switches():
        raise ValueError(f"{approach} takes no switches")
    
    if approach == "incumbent_snowflake":
        return _Policy(("small>big",), "small", False, False)
    
    if approach == "incumbent_bigquery":
        return _Policy(("proxy>big",), "stream", False, False)
    
    cascades = tuple(_TIERS) if switches.t3 else ("proxy>big",)
    return _Policy(cascades, "rank" if switches.t2 else "stream", switches.t1, switches.t4)


@dataclass
class _Ctx:
    no: int
    env: Env
    data: _Scope
    big: _Tier
    small: _Tier
    rng: np.random.Generator
    policy: _Policy
    history: dict[str, list[str]]  # calibration keys per question, in the order drawn
    cal: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    before: dict[str, frozenset] = field(default_factory=dict)  # big answers held before calibration
    proxy_rows: int = 0
    proxies: dict[str, np.ndarray] = field(default_factory=dict)
    _x: np.ndarray | None = None

    @property
    def spec(self):
        return QUERIES[self.no]

    @property
    def keys(self) -> np.ndarray:
        return self.data.keys

    def X(self) -> np.ndarray:
        if self._x is None:
            self._x = self.env.emb[self.data.emb_idx]
        return self._x

    def remaining(self, qid: str, pop: np.ndarray) -> int:
        known = self.big.answers.get(qid, {})
        return sum(k not in known for k in self.keys[pop])


# calibration and the proxy 

# Draw up to 600 distinct keys and ask the big model every question of the query on them
def _calibrate(ctx: _Ctx, qids: tuple[str, ...]) -> None:
    n_keys = len(ctx.keys)
    ctx.before = {qid: frozenset(ctx.big.answers.get(qid, {})) for qid in qids}
    prior: list[int] = []
    
    if ctx.policy.t4:
        where = {k: i for i, k in enumerate(ctx.keys)}
        for qid in qids:
            prior += [where[k] for k in ctx.history.get(qid, []) if k in where]
        prior = list(dict.fromkeys(prior))[:CALIBRATION]
        
    m = min(CALIBRATION, n_keys)
    rest = np.setdiff1d(np.arange(n_keys), np.asarray(prior, dtype=np.int64))
    extra = rest[ctx.rng.choice(len(rest), size=m - len(prior), replace=False)]
    ctx.cal = np.concatenate([np.asarray(prior, dtype=np.int64), extra])
    
    for qid in qids:
        ctx.big.ask(qid, ctx.keys[ctx.cal])
        ctx.history.setdefault(qid, []).extend(ctx.keys[ctx.cal])


def _yes(ctx: _Ctx, qid: str, pos: np.ndarray) -> np.ndarray:
    return ctx.big.ask(qid, ctx.keys[pos]) == "yes"


#Scores of a proxy trained on the first 300 calibration keys, for every other scope key.
def _proxy(ctx: _Ctx, name: str) -> np.ndarray:
    if name not in ctx.proxies:
        train = ctx.cal[:TRAIN]
        keys = ctx.keys[train]
        n_classes = 2
        
        if name == "C2":
            y = [C2_CLASSES.index(a) for a in ctx.big.ask("C2", keys)]
            n_classes = len(C2_CLASSES)
        elif name == "E7":
            y = np.array([normalize_model(a, ctx.env.aliases) != "none" for a in ctx.big.ask("E7", keys)])
        else:
            y = np.all([_yes(ctx, q, train) for q in name.split("&")], axis=0)
            
        model = Proxy().fit(ctx.X()[train], np.asarray(y, dtype=int), n_classes)
        scored = np.ones(len(ctx.keys), dtype=bool)
        scored[train] = False
        shape = (len(ctx.keys),) if n_classes == 2 else (len(ctx.keys), n_classes)
        
        out = np.zeros(shape)
        out[scored] = model.scores(ctx.X()[scored])
        
        ctx.proxies[name] = out
        ctx.proxy_rows += int(scored.sum())
    return ctx.proxies[name]


# cascades 


@dataclass(frozen=True)
class _Stage:
    weight: float                                # cost per row in big-call equivalents (proxy: already paid)
    decide: Callable[[np.ndarray], np.ndarray]  # key positions -> answer, or None to pass the key on


@dataclass(frozen=True)
class _Plan:
    name: str
    stages: tuple[_Stage, ...]
    cost: float          # estimated cost on the remaining keys, big-call equivalents
    claim: dict | None   # estimated precision / recall (accuracy) on the calibration rows


def _halved(t: RowTargets) -> RowTargets:
    return RowTargets(1 - (1 - t.precision) / 2, 1 - (1 - t.recall) / 2)


#One cheap tier with cut-offs fit on its calibration rows
def _stage(ctx: _Ctx, qid: str, tier: str, t: RowTargets) -> _Stage:
    is_class = QUESTIONS[qid].kind == "class"
    if tier == "proxy":
        scores = _proxy(ctx, qid)
        rows, weight = ctx.cal[TRAIN:], 0.0

        def get(pos):
            return scores[pos]
    else:
        rows, weight = ctx.cal, ctx.env.weights.small

        def get(pos):
            p = ctx.small.probs(qid, ctx.keys[pos])
            return p if is_class else p[:, 0]
        
    truth = ctx.big.ask(qid, ctx.keys[rows])
    seen = get(rows)
    if is_class:
        cut = fit_class_cutoff(seen.max(axis=1), _CLASSES[seen.argmax(axis=1)] == truth, t.precision)

        def decide(pos):
            p, out = get(pos), np.full(len(pos), None, dtype=object)
            ok = p.max(axis=1) >= cut
            out[ok] = _CLASSES[p.argmax(axis=1)[ok]]
            return out
    else:
        cuts = fit_cutoffs(seen, truth == "yes", t)

        def decide(pos):
            r, out = route(get(pos), cuts), np.full(len(pos), None, dtype=object)
            out[r == 1], out[r == 0] = "yes", "no"
            return out
        
    return _Stage(weight, decide)


#Run the cheap stages on calibration rows; whatever is left gets the big model's answer
def _simulate(stages, rows: np.ndarray, truth: np.ndarray) -> tuple[list[float], float, np.ndarray]:
    out = np.full(len(rows), None, dtype=object)
    todo = np.arange(len(rows))
    reach = []
    
    for st in stages:
        reach.append(len(todo) / len(rows))
        if len(todo):
            d = st.decide(rows[todo])
            hit = np.array([a is not None for a in d], dtype=bool)
            out[todo[hit]] = d[hit]
            todo = todo[~hit]
            
    out[todo] = truth[todo]
    return reach, len(todo) / len(rows), out


def _claim(qid: str, out: np.ndarray, truth: np.ndarray) -> dict:
    if QUESTIONS[qid].kind == "class":
        return {"accuracy": float(np.mean(out == truth))}
    
    pred, real = out == "yes", truth == "yes"
    tp = int((pred & real).sum())
    return {
        "precision": tp / int(pred.sum()) if pred.any() else 1.0,
        "recall": tp / int(real.sum()) if real.any() else 1.0
    }


def _plan(ctx: _Ctx, qid: str, name: str, r: int) -> _Plan:
    tiers = _TIERS[name]
    t = ctx.spec.targets[qid]
    t = _halved(t) if len(tiers) > 1 else t  # three tiers: each stage gets half of each error allowance
    
    stages = tuple(_stage(ctx, qid, tier, t) for tier in tiers)
    rows = ctx.cal[TRAIN:] if "proxy" in tiers else ctx.cal
    
    truth = ctx.big.ask(qid, ctx.keys[rows])
    reach, to_big, out = _simulate(stages, rows, truth)
    cost = sum(share * r * st.weight for share, st in zip(reach, stages)) + to_big * r
    return _Plan(name, stages, cost, _claim(qid, out, truth))


#(calibration still to pay, the least a small-model cascade can cost on `r` rows). Makes no small calls
def _small_floors(ctx: _Ctx, qid: str, r: int, names: list[str]) -> tuple[float, float]:
    w = ctx.env.weights.small
    unpaid = (len(ctx.cal) - ctx.small.has_probs(qid, ctx.keys[ctx.cal])) * w
    floors = []
    
    if "small>big" in names:
        floors.append(r * w)
        
    if "proxy>small>big" in names:
        proxy = _stage(ctx, qid, "proxy", _halved(ctx.spec.targets[qid]))
        reach, to_big, _ = _simulate([proxy], ctx.cal[TRAIN:], ctx.big.ask(qid, ctx.keys[ctx.cal[TRAIN:]]))
        floors.append(to_big * r * w)
    return unpaid, min(floors)


#Can a cascade with a small-model step beat `bar` at all? Checked before paying its calibration
def _small_may_pay(ctx: _Ctx, qid: str, r: int, names: list[str], bar: float) -> bool:
    unpaid, floor = _small_floors(ctx, qid, r, names)
    return unpaid + floor < bar


def _meets_targets(qid: str, claim: dict, t: RowTargets) -> bool:
    if "accuracy" in claim:
        return claim["accuracy"] >= t.precision
    return claim["precision"] >= t.precision and claim["recall"] >= t.recall


#The cheapest cascade the approach may use, or the big model on everything if none beats it
def _choose(ctx: _Ctx, qid: str, r: int) -> _Plan:
    names = ctx.policy.cascades
    plans: dict[str, _Plan] = {}
    if "proxy>big" in names:
        plans["proxy>big"] = _plan(ctx, qid, "proxy>big", r)
        
    small = [n for n in names if "small" in n]
    met = [p.cost for p in plans.values() if _meets_targets(qid, p.claim, ctx.spec.targets[qid])]
    if small and _small_may_pay(ctx, qid, r, small, min(met + [r])):
        for n in small:
            plans[n] = _plan(ctx, qid, n, r)
            
    best = _Plan("big", (), float(r), None)
    for n in names:
        if n in plans and plans[n].cost < best.cost and _meets_targets(qid, plans[n].claim, ctx.spec.targets[qid]):
            best = plans[n]
    return best


#Decide every key in `pop`: known big answers first, then the cheap stages, then the big model
def _execute(ctx: _Ctx, qid: str, plan: _Plan, pop: np.ndarray) -> dict[str, str]:
    keys = ctx.keys
    known = ctx.big.answers.get(qid, {})
    todo = np.array([p for p in pop if keys[p] not in known], dtype=np.int64)
    cheap: dict[str, str] = {}
    
    for st in plan.stages:
        if not len(todo):
            break
        
        d = st.decide(todo)
        hit = np.array([a is not None for a in d], dtype=bool)
        cheap.update(zip(keys[todo[hit]], d[hit]))
        todo = todo[~hit]
        
    if len(todo):
        ctx.big.ask(qid, keys[todo])
    known = ctx.big.answers.get(qid, {})
    return {k: known[k] if k in known else cheap[k] for k in keys[pop]}


#building the run's record 


#Final answers per question for the scope keys: what was decided, else what the big model said
def _record(ctx: _Ctx, decided: Mapping[str, Mapping[str, str]]) -> dict[str, dict[str, str]]:
    out = {}
    for qid in ctx.spec.qids:
        mine, known = decided.get(qid, {}), ctx.big.answers.get(qid, {})
        out[qid] = {k: mine[k] if k in mine else known[k] for k in ctx.keys if k in mine or k in known}
    return out


_FULL = {
    "method": "full", 
    "confidence": 1.0
}


def _run(ctx: _Ctx, decisions, mode: str, stated: dict, variant: str) -> QueryRun:
    answer = answer_from(ctx.no, ctx.data.df, decisions, ctx.env.aliases)
    return QueryRun(ctx.no, answer, Usage(), mode, stated, variant, decisions)


def _full(ctx: _Ctx, qids: tuple[str, ...]) -> QueryRun:
    for qid in qids:
        ctx.big.ask(qid, ctx.keys)
    return _run(ctx, _record(ctx, {}), "full", dict(_FULL), "big")


def _cascade_stated(ctx: _Ctx, plans: list[tuple[str, _Plan]]) -> dict:
    claims = {q: p.claim or _claim(q, np.array(["yes"]), np.array(["yes"])) for q, p in plans}
    targets = ctx.spec.targets
    if QUESTIONS[plans[0][0]].kind == "class":
        return {
            "method": "cascade", 
            "accuracy_est": {q: c["accuracy"] for q, c in claims.items()},
            "targets": {
                q: {"accuracy": targets[q].precision} for q in claims
            }, 
            "confidence": None
        }
        
    return {
        "method": "cascade",
        "precision_est": {q: c["precision"] for q, c in claims.items()},
        "recall_est": {q: c["recall"] for q, c in claims.items()},
        "targets": {q: {"precision": targets[q].precision, "recall": targets[q].recall} for q in claims},
        "confidence": None
    }


# query kinds 


#Several AI filters: the one fewer calibration rows pass goes first (ties: as written)
def _filter_order(ctx: _Ctx) -> tuple[str, ...]:
    qids = ctx.spec.qids
    if len(qids) == 1:
        return qids
    share = {q: float(_yes(ctx, q, ctx.cal).mean()) for q in qids}
    return tuple(sorted(qids, key=lambda q: (share[q], qids.index(q))))


#Each filter's cascade in turn; a later filter only sees keys the earlier ones passed
def _exact(ctx: _Ctx, order: tuple[str, ...]) -> QueryRun:
    pop = np.arange(len(ctx.keys))
    decided, plans = {}, []
    for i, qid in enumerate(order):
        plan = _choose(ctx, qid, ctx.remaining(qid, pop))
        decided[qid] = _execute(ctx, qid, plan, pop)
        plans.append((qid, plan))
        if i + 1 < len(order):
            pop = pop[np.array([decided[qid][k] == "yes" for k in ctx.keys[pop]], dtype=bool)]
            
    if all(p.name == "big" for _, p in plans):
        return _run(ctx, _record(ctx, decided), "full", dict(_FULL), "big")
    
    variant = plans[0][1].name if len(plans) == 1 else ",".join(f"{q}:{p.name}" for q, p in plans)
    return _run(ctx, _record(ctx, decided), "exact", _cascade_stated(ctx, plans), variant)


#What finishing exactly would cost from here, by the approach's own estimate
def _exact_cost(ctx: _Ctx, order: tuple[str, ...]) -> float:
    pop = np.arange(len(ctx.keys))
    first = ctx.remaining(order[0], pop)
    if QUESTIONS[order[0]].kind == "extract":
        return float(first)
    
    cost = _choose(ctx, order[0], first).cost
    if len(order) > 1:  # the next filter sees about the share of keys the first one passes
        share = float(_yes(ctx, order[0], ctx.cal).mean())
        cost += _choose(ctx, order[1], int(round(share * first))).cost
    return cost


#Least and most `_choose` can cost on `r` keys, found without paying for anything
def _filter_bounds(ctx: _Ctx, qid: str, r: int) -> tuple[float, float]:
    names = ctx.policy.cascades
    hi = float(r)
    if "proxy>big" in names:
        plan = _plan(ctx, qid, "proxy>big", r)
        if _meets_targets(qid, plan.claim, ctx.spec.targets[qid]):
            hi = min(hi, plan.cost)
            
    small = [n for n in names if "small" in n]
    return (min(hi, _small_floors(ctx, qid, r, small)[1]) if small else hi), hi


#(lo, hi) around `_exact_cost`, without any small-model call: lo <= exact cost <= hi
def _exact_bounds(ctx: _Ctx, order: tuple[str, ...]) -> tuple[float, float]:
    pop = np.arange(len(ctx.keys))
    first = ctx.remaining(order[0], pop)
    if QUESTIONS[order[0]].kind == "extract":
        return float(first), float(first)
    
    lo, hi = _filter_bounds(ctx, order[0], first)
    if len(order) > 1:
        share = float(_yes(ctx, order[0], ctx.cal).mean())
        lo2, hi2 = _filter_bounds(ctx, order[1], int(round(share * first)))
        lo, hi = lo + lo2, hi + hi2
    return lo, hi


#Counts, groups, top 10s and the average: Q1-Q3, Q5, Q6, Q8-Q10
def _filter_query(ctx: _Ctx) -> QueryRun:
    _calibrate(ctx, ctx.spec.qids)
    if len(ctx.keys) <= CALIBRATION:
        return _full(ctx, ctx.spec.qids)
    
    order = _filter_order(ctx)
    if ctx.policy.t1 and ctx.spec.estimate_ok:
        run = _estimate(ctx, order)
        if run is not None:
            return run
        
    return _exact(ctx, order)


def _extract_query(ctx: _Ctx) -> QueryRun:
    if ctx.policy.t1 and len(ctx.keys) > CALIBRATION:
        _calibrate(ctx, ("E7",))
        run = _estimate(ctx, ("E7",))
        if run is not None:
            return run
        
    return _full(ctx, ("E7",))


def _limit_query(ctx: _Ctx) -> QueryRun:
    keys, mult = ctx.keys, ctx.data.mult
    if ctx.policy.limit == "stream":
        found = 0
        for pos in ctx.data.row_key:  # rows in table order, one call per new text
            if ctx.big.ask("F4", [keys[pos]])[0] == "yes":
                found += 1
                if found == ROW_LIMIT:
                    break
                
        variant = "big"
    else:
        order = np.arange(len(keys)) if ctx.policy.limit == "small" else _rank(ctx, "F4")
        found = 0
        for start in range(0, len(order), BATCH):
            batch = order[start:start + BATCH]
            likely = batch[ctx.small.probs("F4", keys[batch])[:, 0] >= SMALL_YES]
            for pos in likely:
                if ctx.big.ask("F4", [keys[pos]])[0] == "yes":
                    found += int(mult[pos])
                    if found >= ROW_LIMIT:
                        break
                    
            if found >= ROW_LIMIT:
                break
        variant = "small>big" if ctx.policy.limit == "small" else "rank>small>big"
        
    stated = {
        "method": "cascade", 
        "precision_est": {"F4": 1.0}, 
        "recall_est": {},
        "targets": {},
        "confidence": None
    }  # every returned row was confirmed by the big model
    
    return _run(ctx, _record(ctx, {}), "exact", stated, variant)


#Keys by cosine similarity to the question's embedding, most similar first (ties: table order)
def _rank(ctx: _Ctx, qid: str) -> np.ndarray:
    X = ctx.X().astype(np.float64)
    q = np.asarray(ctx.env.question_emb[qid], dtype=np.float64)
    
    sim = X @ q / (np.maximum(np.linalg.norm(X, axis=1), 1e-12) * max(float(np.linalg.norm(q)), 1e-12))
    ctx.proxy_rows += len(X)
    
    return np.argsort(-sim, kind="stable")


# T1: sampled estimates


# Feeds one query to `run_estimate`.
class _Estimator:
    
    def __init__(self, ctx: _Ctx, order: tuple[str, ...]):
        self.ctx, self.order = ctx, order
        spec, data = ctx.spec, ctx.data
        self.kind = spec.kind
        self.extract = QUESTIONS[order[0]].kind == "extract"
        
        n_keys = len(ctx.keys)
        multi = data.mult > 1
        in_cal = np.zeros(n_keys, dtype=bool)
        in_cal[ctx.cal] = True
        
        # known before this query's draw, so leaving them out keeps the rest of the draw uniform
        exact = self._decided_before() | (multi & in_cal)
        exact[ctx.cal[:TRAIN]] = True
        
        self.later = np.flatnonzero(multi & ~exact)
        self.pop = np.flatnonzero(~exact & ~multi)
        
        where = {int(p): i for i, p in enumerate(self.pop)}
        self.first = [where[int(p)] for p in ctx.cal[TRAIN:] if int(p) in where]
        self.groups = list(C2_CLASSES) if self.kind == "groups" else None
        self.scores = self._strata_scores()[self.pop]
        self.known_free = self._known(np.flatnonzero(exact))
        self.seed = int(ctx.rng.integers(2 ** 63))
        share = float(_yes(ctx, order[0], ctx.cal).mean()) if len(order) > 1 else 0.0
        self.calls_per_row = 1.0 + share

    # Keys whose value the big answers held before calibration already settle.
    def _decided_before(self) -> np.ndarray:
        ctx = self.ctx
        before = ctx.before
        out = np.zeros(len(ctx.keys), dtype=bool)
        pending = np.ones(len(ctx.keys), dtype=bool)
        
        for qid in self.order:
            had, answers = before.get(qid, frozenset()), ctx.big.answers.get(qid, {})
            asked = np.array([k in had for k in ctx.keys], dtype=bool)
            pending &= asked
            if QUESTIONS[qid].kind != "yesno":
                break

            said_no = pending & np.array([answers.get(k) == "no" for k in ctx.keys], dtype=bool)
            out |= said_no
            pending &= ~said_no
        return out | pending

    def _strata_scores(self) -> np.ndarray:
        ctx = self.ctx
        if self.kind == "groups":  # by the proxy's top class, then its probability within it
            p = _proxy(ctx, "C2")
            return p.argmax(axis=1) + p.max(axis=1)
        
        return _proxy(ctx, "E7" if self.extract else "&".join(ctx.spec.qids))

    def _values(self, pos: np.ndarray) -> np.ndarray:
        ctx = self.ctx
        if self.kind == "groups":
            return ctx.big.ask("C2", ctx.keys[pos])
        
        if self.extract:
            return np.array([normalize_model(a, ctx.env.aliases) for a in ctx.big.ask("E7", ctx.keys[pos])], dtype=object)
        
        match = np.ones(len(pos), dtype=bool)
        for qid in self.order:
            idx = np.flatnonzero(match)
            if not len(idx):
                break
            match[idx] = _yes(self.ctx, qid, pos[idx])
        return match

    def _known(self, pos: np.ndarray):
        df, data = self.ctx.data.df, self.ctx.data
        values = self._values(pos)
        if self.kind == "groups" or self.extract:
            counts: Counter[str] = Counter()
            for v, m in zip(values, data.mult[pos]):
                if v != "none" or not self.extract:
                    counts[str(v)] += int(m)
            return {g: float(c) for g, c in sorted(counts.items())}
        
        match = np.zeros(len(data.keys), dtype=bool)
        match[pos] = values
        rows = match[data.row_key]
        if self.kind == "count":
            return float(rows.sum())
        
        if self.kind == "average":
            return float(df["rating"].to_numpy(dtype=float)[rows].sum()), float(rows.sum())
        
        return {g: float(c) for g, c in sorted(Counter(str(g) for g in df["group"].to_numpy()[rows]).items())}

    def guessed_known(self):
        rows_later = int(self.ctx.data.mult[self.later].sum())
        if not len(self.first) or not rows_later:
            return self.known_free
        
        sample = self._known(self.pop[np.asarray(self.first, dtype=np.int64)])
        return _plus(self.known_free, sample, rows_later / len(self.first))

    def exact_known(self):
        if not len(self.later):
            return self.known_free
        return _plus(self.known_free, self._known(self.later), 1.0)

    def answer(self, idx) -> Any:
        pos = self.pop[np.asarray(idx, dtype=np.int64)]
        values = self._values(pos)
        df = self.ctx.data.df
        rows = self.ctx.data.first_row[pos]  # each sampled key covers just this row
        if self.kind == "count":
            return values.astype(int)
        
        if self.kind == "groups":
            return list(values)
        
        if self.extract:
            return [None if v == "none" else str(v) for v in values]
        
        if self.kind == "average":
            return np.column_stack([values.astype(float), df["rating"].to_numpy(dtype=float)[rows]])
        
        groups = df["group"].to_numpy()[rows]
        return [str(g) if m else None for g, m in zip(groups, values)]

    def run(self, known, max_rows: int) -> EstimateResult:
        return run_estimate(
            self.kind, self.scores, known, self.answer, delta=DELTA,
            rng=np.random.default_rng(self.seed), max_rows=max_rows, first=self.first,
            groups=self.groups, n_train=TRAIN
        )


#a + f * b for a known part: a number, a (sum, count) pair or {group: count}
def _plus(a, b, f: float):
    if isinstance(a, dict):
        return {g: a.get(g, 0.0) + f * b.get(g, 0.0) for g in sorted(set(a) | set(b))}
    
    if isinstance(a, tuple):
        return tuple(x + f * y for x, y in zip(a, b))
    
    return a + f * b


PAYS = 0.75  # T1 goes on sampling only if its projected big calls are at most this share of the exact cost


def t1_projection(est: _Estimator, look1: EstimateResult) -> float:
    need = projected_sample(look1, len(est.pop), n_train=TRAIN, delta=DELTA, n_first=CALIBRATION, complete=est.groups is not None)
    return (max(need - look1.rows_asked, 0) + len(est.later)) * est.calls_per_row


#Try a sampled estimate; None when it does not pay (the caller then finishes exactly).
def _estimate(ctx: _Ctx, order: tuple[str, ...]) -> QueryRun | None:
    est = _Estimator(ctx, order)
    look1 = est.run(est.guessed_known(), max_rows=len(est.first))
    projected = t1_projection(est, look1)
    
    if projected > 0:
        lo, hi = _exact_bounds(ctx, order)
        if projected > PAYS * hi:
            return None
        
        exact = lo if projected <= PAYS * lo else _exact_cost(ctx, order)
        if projected > PAYS * exact:
            return None
        
        budget = (exact - len(est.later) * est.calls_per_row) / est.calls_per_row
        res = est.run(est.exact_known(), max_rows=len(est.first) + max(int(budget), 0))
    else:
        res = look1
        
    if res.stopped == "census":  # every key was asked: the exact answer
        return _run(ctx, _record(ctx, {}), "exact", dict(_FULL), "census")
    
    if res.stopped == "budget":
        return None
    
    if res.kind in ("count", "average"):
        iv = res.interval
        answer = int(round(iv.est)) if res.kind == "count" else float(iv.est)
        stated = {
            "method": "estimate", 
            "est": float(iv.est),
            "lo": float(iv.lo), 
            "hi": float(iv.hi)
        }
    else:
        ivs = res.intervals
        answer = {g: int(round(iv.est)) for g, iv in ivs.items()} if res.kind == "groups" else list(res.top10)
        stated = {
            "method": "estimate", 
            **{part: {g: float(getattr(iv, part)) for g, iv in ivs.items()} for part in ("est", "lo", "hi")}
        }
        
    stated = {
        **stated,
        "looks": res.looks,
        "confidence": CONFIDENCE
    }
    return QueryRun(ctx.no, answer, Usage(), "estimate", stated, "estimate", None)


# the session 


def _query(ctx: _Ctx) -> QueryRun:
    if ctx.spec.kind == "rows":
        return _limit_query(ctx)
    if QUESTIONS[ctx.spec.qids[0]].kind == "extract":
        return _extract_query(ctx)
    
    return _filter_query(ctx)


#Charge the session's embedding pass evenly to the queries that used embeddings.
def split_embedding_rows(runs: list[QueryRun], embedded_rows: float) -> list[QueryRun]:
    users = sum(r.usage.proxy > 0 for r in runs)
    for r in runs:
        r.usage.embed_rows = embedded_rows / users if r.usage.proxy > 0 else 0.0
    return runs


#Run the ten queries in order. Oracles come from the factories
def run_session(approach: str, env: Env, big: Callable[[], Oracle], small: Callable[[], Oracle], seed: int, switches: Switches = Switches()) -> list[QueryRun]:
    policy = _policy(approach, switches)
    shared = (_Tier(big()), _Tier(small())) if policy.t4 else None
    history: dict[str, list[str]] = {}
    runs = []
    
    for no in sorted(QUERIES):
        b, s = shared or (_Tier(big()), _Tier(small()))
        calls = b.oracle.calls, s.oracle.calls
        ctx = _Ctx(no, env, _scope_of(env, no), b, s, np.random.default_rng([seed, no]), policy, history)
        run = _query(ctx)
        run.usage = Usage(big=b.oracle.calls - calls[0], small=s.oracle.calls - calls[1], proxy=ctx.proxy_rows)
        runs.append(run)
        
    return split_embedding_rows(runs, _embedded_rows(env))
