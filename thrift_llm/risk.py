"""Distribution-free risk control (Learn then Test; Angelopoulos, Bates, Candes et al., 2021).

Every knob in Thrift that trades quality for tokens (router threshold, cache similarity
threshold, compression ratio) is set the same way:

  1. Order candidate settings from safest to most aggressive.
  2. For each, compute a p-value for H0: "true risk at this setting > epsilon" from a held-out
     calibration set of per-example losses in [0, 1].
  3. Walk the sequence while p <= delta (fixed-sequence testing controls the family-wise error
     at delta) and keep the last setting that passed.

Guarantee: with probability >= 1 - delta over the calibration draw, the chosen setting's
expected loss on new exchangeable traffic is <= epsilon. No assumption on the model, the
data distribution, or how good the scorer is. A bad scorer just yields a safe, timid setting.

This is the same family of guarantees Pacer uses for latency (split conformal / ACI); here it
bounds a quality loss rather than a quantile. Pure numpy + math, no scipy.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


def _log_binom_cdf(k: int, n: int, p: float) -> float:
    """log P(Binom(n, p) <= k), computed stably in log space."""
    if k < 0:
        return -math.inf
    if k >= n:
        return 0.0
    if p <= 0:
        return 0.0
    if p >= 1:
        return -math.inf
    i = np.arange(0, k + 1, dtype=np.float64)
    lg = np.array([math.lgamma(n + 1)]) - np.vectorize(math.lgamma)(i + 1) - np.vectorize(math.lgamma)(n - i + 1)
    terms = lg + i * math.log(p) + (n - i) * math.log1p(-p)
    m = terms.max()
    return float(m + math.log(np.exp(terms - m).sum()))


def binomial_pvalue(losses: np.ndarray, epsilon: float) -> float:
    """Exact p-value for H0: mean loss > epsilon, for binary (0/1) losses."""
    n = len(losses)
    if n == 0:
        return 1.0
    k = int(np.round(np.sum(losses)))
    return float(min(1.0, math.exp(_log_binom_cdf(k, n, epsilon))))


def _h1(a: float, b: float) -> float:
    a = min(max(a, 1e-12), 1 - 1e-12)
    return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))


def hb_pvalue(losses: np.ndarray, epsilon: float) -> float:
    """Hoeffding-Bentkus p-value for H0: mean loss > epsilon, for losses in [0, 1]."""
    n = len(losses)
    if n == 0:
        return 1.0
    r = float(np.mean(losses))
    hoeff = math.exp(-n * _h1(min(r, epsilon), epsilon)) if r < epsilon else 1.0
    bentkus = math.e * math.exp(_log_binom_cdf(int(math.ceil(n * r)), n, epsilon))
    return float(min(1.0, hoeff, bentkus))


def pvalue(losses: np.ndarray, epsilon: float) -> float:
    losses = np.asarray(losses, dtype=np.float64)
    binary = bool(np.all((losses == 0) | (losses == 1)))
    return binomial_pvalue(losses, epsilon) if binary else hb_pvalue(losses, epsilon)


def upper_bound(losses: np.ndarray, delta: float) -> float:
    """(1 - delta) upper confidence bound on the mean loss (inverts the p-value by bisection)."""
    losses = np.asarray(losses, dtype=np.float64)
    if len(losses) == 0:
        return 1.0
    lo, hi = float(np.mean(losses)), 1.0
    if pvalue(losses, hi - 1e-9) > delta:
        return 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if pvalue(losses, mid) <= delta:
            hi = mid
        else:
            lo = mid
    return hi


@dataclass
class Certificate:
    setting: float  # the chosen knob value
    index: int  # its position in the candidate sequence (-1: nothing passed, use the safe default)
    epsilon: float
    delta: float
    n_cal: int
    empirical_risk: float
    risk_ucb: float  # (1 - delta) upper bound on the chosen setting's risk
    tested: int

    def to_dict(self) -> dict:
        return asdict(self)


def learn_then_test(settings: list[float], loss_fn, epsilon: float, delta: float) -> Certificate:
    """Fixed-sequence Learn-then-Test.

    settings: ordered from safest to most aggressive.
    loss_fn(setting) -> array of per-example calibration losses in [0, 1].
    """
    chosen, idx, emp, ucb, tested = settings[0], -1, 0.0, 0.0, 0
    n = 0
    for i, s in enumerate(settings):
        losses = np.asarray(loss_fn(s), dtype=np.float64)
        n = len(losses)
        tested += 1
        if pvalue(losses, epsilon) > delta:
            break
        chosen, idx, emp = s, i, float(np.mean(losses))
    if idx >= 0:
        ucb = upper_bound(np.asarray(loss_fn(chosen)), delta)
    return Certificate(float(chosen), idx, epsilon, delta, n, emp, ucb, tested)
