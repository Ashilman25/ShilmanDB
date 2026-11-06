import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from .queries import RowTargets


#One-sided Wilson score bound on a share, at level delta ("lower" or "upper")
def wilson_bound(successes: int, n: int, delta: float, side: str) -> float:
    if side not in ("lower", "upper"):
        raise ValueError(f"side must be 'lower' or 'upper', got {side!r}")
    if n == 0:
        return 0.0 if side == "lower" else 1.0
    
    z = NormalDist().inv_cdf(1 - delta)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half) if side == "lower" else min(1.0, center + half)


@dataclass(frozen=True)
class Cutoffs:
    accept: float
    reject: float


def _grid(scores: np.ndarray, grid: int) -> np.ndarray:
    return np.quantile(scores, np.arange(1, grid + 1) / grid, method="lower")


#Cut-offs from calibration rows (the big model's answers are `is_yes`)
def fit_cutoffs(scores, is_yes, targets: RowTargets, grid: int = 20) -> Cutoffs:
    scores = np.asarray(scores, dtype=float)
    is_yes = np.asarray(is_yes, dtype=bool)
    if len(scores) == 0:
        return Cutoffs(math.inf, -math.inf)
    points = _grid(scores, grid)

    accept = math.inf
    for t in points:  # ascending: the first that passes is the smallest
        picked = scores >= t
        if is_yes[picked].sum() / picked.sum() >= targets.precision:
            accept = float(t)
            break

    reject = -math.inf
    positives = scores[is_yes]
    if len(positives):
        for t in points[::-1]:  # descending: the first that passes is the largest
            if (positives <= t).sum() / len(positives) <= 1 - targets.recall:
                reject = float(t)
                break
    return Cutoffs(accept, reject)


#Smallest grid point whose rows at or above it have observed accuracy >= `accuracy` (else inf)
def fit_class_cutoff(top_prob, correct, accuracy: float, grid: int = 20) -> float:
    top_prob = np.asarray(top_prob, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if len(top_prob) == 0:
        return math.inf
    
    for t in _grid(top_prob, grid):
        picked = top_prob >= t
        if correct[picked].sum() / picked.sum() >= accuracy:
            return float(t)
    return math.inf


#1 = answer yes
# 0 = answer no
# -1 = ask the next tier (yes wins an overlap)
def route(scores, cut: Cutoffs) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    out = np.full(len(scores), -1, dtype=np.int8)
    out[scores <= cut.reject] = 0
    out[scores >= cut.accept] = 1
    return out
