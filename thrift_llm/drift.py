"""Drift guard: notice, while serving, when the routing guarantee stops holding.

The router's certificate says E[loss] <= epsilon on traffic like the calibration set. Traffic
changes and providers silently update models, so Thrift keeps testing that claim online.

A random fraction r of requests the router sent to a cheaper model is also sent to the
strongest model (shadow traffic) and a judge scores how much worse the cheap answer was
(loss in [0, 1]). Every routed request contributes one observation

    Z = loss / r   if it was shadowed,   0 otherwise   (strongest-model answers have loss 0)

so E[Z] is exactly the live routing risk. RiskMonitor runs a betting e-process for the null
hypothesis E[Z] <= epsilon:

    K_t = prod_i (1 + lambda_i (Z_i - epsilon)),   lambda_i chosen from Z_1..Z_{i-1} only.

Under the null K_t is a nonnegative supermartingale, so by Ville's inequality the chance it
ever reaches 1/alpha is at most alpha: a false alarm happens with probability <= alpha over
the whole (unbounded) stream, with no fixed sample size and no peeking penalty. When it does
reach 1/alpha the gateway switches to safe mode (every auto-routed request goes to the
strongest model) until someone re-calibrates and resets it.
"""
from __future__ import annotations

import math
import re
import threading
import time

JUDGE_PROMPT = (
    "You compare two answers to the same request. Answer A is the reference. Reply with only a number from 0 to 1: "
    "how much worse answer B is than answer A (0 = B is as good or better, 1 = B is wrong or useless).\n\n"
    "Request:\n{q}\n\nAnswer A:\n{a}\n\nAnswer B:\n{b}"
)
_NUM = re.compile(r"[01](?:\.\d+)?|\.\d+")


def parse_judgement(text: str) -> float | None:
    m = _NUM.search(text or "")
    if not m:
        return None
    return min(1.0, max(0.0, float(m.group())))


def agreement_loss(a: str, b: str) -> float:
    """Label-free fallback judge: 1 - token F1 between the answers (conservative for long answers)."""
    ta, tb = re.findall(r"\w+", a.lower()), re.findall(r"\w+", b.lower())
    if not ta and not tb:
        return 0.0
    common = {}
    for w in ta:
        common[w] = common.get(w, 0) + 1
    overlap = 0
    for w in tb:
        if common.get(w, 0) > 0:
            overlap += 1
            common[w] -= 1
    if overlap == 0:
        return 1.0
    p, r = overlap / len(tb), overlap / len(ta)
    return 1.0 - 2 * p * r / (p + r)


class RiskMonitor:
    """Anytime-valid test of E[Z] <= epsilon for nonnegative observations Z."""

    def __init__(self, epsilon: float, alpha: float = 0.05, max_bet: float = 0.5, warmup: int = 20):
        if not 0 < epsilon < 1:
            raise ValueError("epsilon must be in (0, 1)")
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_bet = max_bet / epsilon  # keeps 1 + lambda (Z - eps) >= 1 - max_bet > 0
        self.warmup = warmup
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.n = 0
            self.n_judged = 0
            self.sum = 0.0
            self.sumsq = 0.0
            self.log_k = 0.0
            self.alarm = False
            self.alarm_at: int | None = None
            self.alarm_ts: float | None = None

    def _bet(self) -> float:
        if self.n < self.warmup:
            return 0.0
        # aGRAPA: the growth-optimal bet for a mean of (mu - eps) and variance sigma^2
        mu = (self.sum + self.epsilon) / (self.n + 1)  # shrunk toward the null
        var = max(1e-12, (self.sumsq + self.epsilon**2) / (self.n + 1) - mu * mu)
        d = mu - self.epsilon
        return min(self.max_bet, max(0.0, d / (var + d * d)))

    def observe(self, z: float, judged: bool = False) -> bool:
        """Add one observation; returns True when this one raised the alarm."""
        with self._lock:
            if self.alarm:  # latched until reset(); later data cannot un-ring the bell
                return False
            lam = self._bet()
            self.log_k += math.log1p(lam * (z - self.epsilon))
            self.n += 1
            self.n_judged += int(judged)
            self.sum += z
            self.sumsq += z * z
            if not self.alarm and self.log_k >= math.log(1 / self.alpha):
                self.alarm, self.alarm_at, self.alarm_ts = True, self.n, time.time()
                return True
            return False

    @property
    def e_value(self) -> float:
        return math.exp(min(self.log_k, 700.0))

    def state(self) -> dict:
        with self._lock:
            return {"epsilon": self.epsilon, "alpha": self.alpha, "observations": self.n, "judged": self.n_judged,
                    "risk_estimate": self.sum / self.n if self.n else 0.0, "e_value": self.e_value,
                    "alarm_threshold": 1 / self.alpha, "safe_mode": self.alarm, "alarm_at": self.alarm_at, "alarm_ts": self.alarm_ts}
