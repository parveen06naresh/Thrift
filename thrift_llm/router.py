"""Difficulty-aware model routing with a certified quality guarantee.

A request goes to the cheap model when the scorer believes the cheap model will answer as
well as the strong one. The scorer is a small logistic model over hashed n-grams plus a few
structural features (length, digits, maths symbols, code, multiple-choice options). It is
trained on any data where both models' quality is known per prompt (offline evals, an LLM
judge, user feedback, or Thrift's own shadow traffic).

The threshold is never hand-tuned. `Router.calibrate` runs Learn-then-Test on held-out
prompts so that

    E[ quality(strong) - quality(routed) ]  <=  epsilon     with probability >= 1 - delta

where quality is per-prompt accuracy (or any score in [0, 1]). The loss counts only the
prompts sent to the cheap model, and only the part where the cheap model did worse
(clipped at 0), so the bound is conservative: prompts where the cheap model wins don't buy
slack elsewhere.
"""
from __future__ import annotations

import base64
import json
import math
import re
import zlib
from dataclasses import dataclass, field

import numpy as np

from . import risk
from .text import STOP, words

DIM = 1 << 13
_MATH = re.compile(r"[=+\-*/^√∑∫≤≥<>]|\\frac|\\sqrt|\$")
_CODE = re.compile(r"```|def |class |function |;\s*$|\{|\}|import ", re.M)
_CHOICE = re.compile(r"(?m)^\s*(?:\(?[A-Ea-e][).:])\s")
_HARD = re.compile(r"\b(prove|derive|why|explain|step by step|calculate|compute|how many|probability|integral|optimi[sz]e|algorithm)\b", re.I)

DENSE = ["bias", "log_chars", "log_words", "digit_frac", "math_syms", "code", "choices", "questions", "hard_words", "newlines"]


def featurize(text: str) -> dict[int, float]:
    """Sparse feature dict: dense structural features in [0, len(DENSE)), hashed n-grams after."""
    t = text[-6000:]  # the end of a prompt (the actual question) carries most of the signal
    ws = words(t)
    n = max(1, len(ws))
    f = {
        0: 1.0,
        1: math.log1p(len(text)) / 10,
        2: math.log1p(len(ws)) / 8,
        3: sum(c.isdigit() for c in t) / max(1, len(t)) * 10,
        4: min(1.0, len(_MATH.findall(t)) / 20),
        5: min(1.0, len(_CODE.findall(t)) / 5),
        6: min(1.0, len(_CHOICE.findall(t)) / 4),
        7: min(1.0, t.count("?") / 3),
        8: min(1.0, len(_HARD.findall(t)) / 3),
        9: min(1.0, t.count("\n") / 50),
    }
    off = len(DENSE)
    span = DIM - off
    scale = 1.0 / math.sqrt(n)
    for i, w in enumerate(ws):
        if w in STOP:
            continue
        k = off + zlib.crc32(b"u" + w.encode()) % span
        f[k] = f.get(k, 0.0) + scale
        if i + 1 < len(ws):
            k = off + zlib.crc32(b"b" + (w + " " + ws[i + 1]).encode()) % span
            f[k] = f.get(k, 0.0) + 0.5 * scale
    return f


def _dense(rows: list[dict[int, float]]) -> np.ndarray:
    X = np.zeros((len(rows), DIM), dtype=np.float32)
    for r, f in enumerate(rows):
        for k, v in f.items():
            X[r, k] = v
    return X


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _fit_w(feats: list[dict[int, float]], y, l2: float = 1e-4, epochs: int = 40, lr: float = 0.05, seed: int = 0) -> np.ndarray:
    """Logistic regression with soft labels, mini-batch Adam."""
    y = np.asarray(y, dtype=np.float32)
    rng = np.random.default_rng(seed)
    w = np.zeros(DIM, dtype=np.float32)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    step = 0
    bs = 256
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for b in range(0, len(y), bs):
            idx = order[b : b + bs]
            X = _dense([feats[i] for i in idx])
            g = X.T @ (_sigmoid(X @ w) - y[idx]) / len(idx) + l2 * w
            step += 1
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            w -= lr * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
    return w


def _pack(w: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(w.astype(np.float16).tobytes(), 9)).decode()


def _unpack(s: str) -> np.ndarray:
    return np.frombuffer(zlib.decompress(base64.b64decode(s)), dtype=np.float16).astype(np.float32)


@dataclass
class Router:
    """Two-tier router. `w` scores P(cheap model is good enough); route cheap if score >= threshold."""

    cheap: str = "cheap"
    strong: str = "strong"
    w: np.ndarray = field(default_factory=lambda: np.zeros(DIM, dtype=np.float32))
    threshold: float = 1.01  # > 1 means "never route cheap" until calibrated
    certificate: dict | None = None
    meta: dict = field(default_factory=dict)

    # ---- training -----------------------------------------------------------------------------
    @staticmethod
    def targets(q_cheap: np.ndarray, q_strong: np.ndarray) -> np.ndarray:
        """Soft label: 1 - (how much quality the cheap model gives up), clipped to [0, 1]."""
        return 1.0 - np.clip(np.asarray(q_strong) - np.asarray(q_cheap), 0.0, 1.0)

    def fit(self, texts: list[str], q_cheap, q_strong, l2: float = 1e-4, epochs: int = 40, lr: float = 0.05, seed: int = 0) -> Router:
        self.w = _fit_w([featurize(t) for t in texts], self.targets(q_cheap, q_strong), l2, epochs, lr, seed)
        return self

    def score(self, texts: list[str]) -> np.ndarray:
        out = np.empty(len(texts))
        for b in range(0, len(texts), 512):
            out[b : b + 512] = _sigmoid(_dense([featurize(t) for t in texts[b : b + 512]]) @ self.w)
        return out

    def score_one(self, text: str) -> float:
        z = sum(self.w[k] * v for k, v in featurize(text).items())
        return float(_sigmoid(z))

    # ---- calibration --------------------------------------------------------------------------
    def calibrate(self, texts: list[str], q_cheap, q_strong, epsilon: float = 0.02, delta: float = 0.1) -> risk.Certificate:
        s = self.score(texts)
        cert = calibrate_scores(s, q_cheap, q_strong, epsilon, delta)
        self.threshold = cert.setting
        self.certificate = cert.to_dict()
        return cert

    # ---- decisions ----------------------------------------------------------------------------
    def route(self, text: str) -> tuple[str, float]:
        s = self.score_one(text)
        return (self.cheap if s >= self.threshold else self.strong), s

    # ---- persistence --------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format": "thrift-router-v1",
            "cheap": self.cheap,
            "strong": self.strong,
            "threshold": self.threshold,
            "certificate": self.certificate,
            "meta": self.meta,
            "dim": DIM,
            "w": _pack(self.w),
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> Router:
        with open(path) as f:
            d = json.load(f)
        if d.get("format") != "thrift-router-v1" or d.get("dim") != DIM:
            raise ValueError(f"{path}: not a thrift-router-v1 file with dim {DIM}")
        return cls(d["cheap"], d["strong"], _unpack(d["w"]), d["threshold"], d.get("certificate"), d.get("meta", {}))

    @property
    def tiers(self) -> list[str]:
        return [self.cheap, self.strong]


def calibrate_scores(scores, q_cheap, q_strong, epsilon: float, delta: float) -> risk.Certificate:
    """Learn-then-Test over thresholds, strictest (route nothing cheap) first."""
    scores = np.asarray(scores, dtype=np.float64)
    gap = np.clip(np.asarray(q_strong, dtype=np.float64) - np.asarray(q_cheap, dtype=np.float64), 0, 1)
    qs = np.unique(np.quantile(scores, np.linspace(1, 0, 201)))[::-1]
    grid = [1.01] + [float(x) for x in qs]
    return risk.learn_then_test(grid, lambda t: (scores >= t) * gap, epsilon, delta)


@dataclass
class Ladder:
    """N-model router across providers, ordered cheapest -> strongest (the reference model).

    One scorer per cheaper model k estimates P(model k answers as well as the strongest). A
    request goes to the cheapest model whose score clears the threshold t, else to the
    strongest. The single knob t is certified exactly like the two-tier router:

        E[ quality(strongest) - quality(chosen) ]  <=  epsilon     with probability >= 1 - delta
    """

    models: list = field(default_factory=list)  # tier names, cheapest first, strongest last
    W: np.ndarray = field(default_factory=lambda: np.zeros((DIM, 0), dtype=np.float32))
    threshold: float = 1.01
    certificate: dict | None = None
    meta: dict = field(default_factory=dict)

    @property
    def tiers(self) -> list[str]:
        return list(self.models)

    def fit(self, texts: list[str], Q, **kw) -> Ladder:
        """Q: (n, K) per-prompt quality of each model, columns in `models` order."""
        Q = np.asarray(Q, dtype=np.float64)
        feats = [featurize(t) for t in texts]
        cols = [_fit_w(feats, Router.targets(Q[:, k], Q[:, -1]), **kw) for k in range(Q.shape[1] - 1)]
        self.W = np.stack(cols, axis=1)
        return self

    def scores(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self.W.shape[1]))
        for b in range(0, len(texts), 512):
            out[b : b + 512] = _sigmoid(_dense([featurize(t) for t in texts[b : b + 512]]) @ self.W)
        return out

    @staticmethod
    def choose(S: np.ndarray, t: float) -> np.ndarray:
        """Index of the cheapest model whose score >= t; the strongest (K-1) if none."""
        ok = np.asarray(S) >= t
        first = np.argmax(ok, axis=1)
        return np.where(ok.any(axis=1), first, S.shape[1])

    def calibrate(self, texts: list[str], Q, epsilon: float = 0.02, delta: float = 0.1) -> risk.Certificate:
        cert = calibrate_ladder(self.scores(texts), Q, epsilon, delta)
        self.threshold = cert.setting
        self.certificate = cert.to_dict()
        return cert

    def route(self, text: str) -> tuple[str, float]:
        f = featurize(text)
        idx = np.fromiter(f.keys(), dtype=np.int64)
        val = np.fromiter(f.values(), dtype=np.float32)
        s = _sigmoid(val @ self.W[idx])
        k = int(self.choose(s[None, :], self.threshold)[0])
        return self.models[k], float(s.max()) if len(s) else 0.0

    def to_dict(self) -> dict:
        return {"format": "thrift-ladder-v1", "models": self.models, "threshold": self.threshold, "certificate": self.certificate,
                "meta": self.meta, "dim": DIM, "W": [_pack(self.W[:, k]) for k in range(self.W.shape[1])]}

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> Ladder:
        with open(path) as f:
            d = json.load(f)
        if d.get("format") != "thrift-ladder-v1" or d.get("dim") != DIM:
            raise ValueError(f"{path}: not a thrift-ladder-v1 file with dim {DIM}")
        W = np.stack([_unpack(c) for c in d["W"]], axis=1)
        return cls(d["models"], W, d["threshold"], d.get("certificate"), d.get("meta", {}))


def calibrate_ladder(S, Q, epsilon: float, delta: float) -> risk.Certificate:
    """Learn-then-Test over the ladder threshold, strictest (everything to the strongest) first."""
    S = np.asarray(S, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    ref = Q[:, -1]
    qs = np.unique(np.quantile(S, np.linspace(1, 0, 201)))[::-1]
    grid = [1.01] + [float(x) for x in qs]
    rows = np.arange(len(S))

    def loss(t):
        return np.clip(ref - Q[rows, Ladder.choose(S, t)], 0, 1)

    return risk.learn_then_test(grid, loss, epsilon, delta)


def load_policy(path: str):
    """A router file of either kind."""
    with open(path) as f:
        fmt = json.load(f).get("format")
    return Ladder.load(path) if fmt == "thrift-ladder-v1" else Router.load(path)
