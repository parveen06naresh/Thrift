"""Two-tier response cache: exact (hash of the whole request) and semantic (nearest neighbour).

Correctness rules, because a wrong cached answer is worse than a paid one:
  * Scope. A semantic hit is only allowed between requests that share everything except the
    last user message: same model tier, same system prompt and history, same sampling params,
    same tools. That scope is hashed into a namespace.
  * Key-term guard. Numbers and negations in the question must match exactly.
  * Calibrated threshold. The similarity threshold comes from Learn-then-Test on labelled
    pairs (`calibrate_threshold`), so P(a query receives a wrong cached answer) <= alpha with
    probability >= 1 - delta. The shipped default (0.92) is slightly stricter than every
    alpha=2% threshold certified on Quora Question Pairs (0.900-0.915 over 10 draws); see
    results/cache.json. Recalibrate on your own traffic with `calibrate_threshold`.
  * Never cached: errors, truncated answers (finish_reason == "length"), n > 1, requests with
    `"thrift": {"cache": false}` or header `x-thrift-cache: off`.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from . import risk
from .text import HashingEmbedder, key_terms, normalize_query, stable_hash

DEFAULT_THRESHOLD = 0.92


@dataclass
class CacheHit:
    kind: str  # "exact" | "semantic"
    response: dict
    similarity: float
    matched_query: str
    age_s: float


@dataclass
class _Entry:
    key: str
    namespace: str
    query: str
    terms: frozenset
    response: dict
    created: float
    hits: int = 0


class _Index:
    """Growable embedding matrix for one namespace (amortised O(1) append, O(n) delete)."""

    def __init__(self, dim: int):
        self.buf = np.zeros((16, dim), dtype=np.float32)
        self.n = 0
        self.keys: list[str] = []

    def add(self, key: str, vec: np.ndarray) -> None:
        if self.n == len(self.buf):
            self.buf = np.vstack([self.buf, np.zeros_like(self.buf)])
        self.buf[self.n] = vec
        self.keys.append(key)
        self.n += 1

    def remove(self, key: str) -> None:
        j = self.keys.index(key)
        self.buf[j : self.n - 1] = self.buf[j + 1 : self.n]
        self.keys.pop(j)
        self.n -= 1

    def search(self, vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self.n == 0:
            return []
        sims = self.buf[: self.n] @ vec
        top = np.argpartition(-sims, min(k, self.n - 1))[:k] if self.n > k else np.arange(self.n)
        top = top[np.argsort(-sims[top])]
        return [(self.keys[j], float(sims[j])) for j in top]


class SemanticCache:
    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        max_entries: int = 50_000,
        ttl_s: float = 7 * 24 * 3600,
        embedder=None,
        path: str | None = None,
        semantic: bool = True,
    ):
        self.threshold = threshold
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self.embedder = embedder or HashingEmbedder()
        self.semantic = semantic
        self._lock = threading.RLock()
        self._entries: OrderedDict[str, _Entry] = OrderedDict()  # LRU order
        # Brute-force cosine per namespace: ~1 ms per 50k entries at d=2048. Swap in an ANN
        # index behind _Index.search past a few million entries.
        self._index: dict[str, _Index] = {}
        self._db = None
        if path:
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, namespace TEXT, query TEXT, response TEXT, created REAL)"
            )
            for key, ns, q, resp, created in self._db.execute("SELECT key, namespace, query, response, created FROM cache ORDER BY created"):
                self._insert(key, ns, q, json.loads(resp), created, persist=False)

    # ---- keys ---------------------------------------------------------------------------------
    @staticmethod
    def namespace(scope: dict) -> str:
        return stable_hash(json.dumps(scope, sort_keys=True, default=str))

    @staticmethod
    def exact_key(namespace: str, query: str) -> str:
        return stable_hash(namespace, normalize_query(query))

    # ---- lookup -------------------------------------------------------------------------------
    def get(self, scope: dict, query: str) -> CacheHit | None:
        ns = self.namespace(scope)
        now = time.time()
        with self._lock:
            key = self.exact_key(ns, query)
            e = self._entries.get(key)
            if e is not None and now - e.created <= self.ttl_s:
                return self._hit(e, "exact", 1.0, now)
            if not self.semantic or ns not in self._index or not query.strip():
                return None
            vec = self.embedder.embed([query])[0]
            terms = key_terms(query)
            for k, s in self._index[ns].search(vec, 5):
                if s < self.threshold:
                    break
                e = self._entries.get(k)
                if e is None or now - e.created > self.ttl_s or e.terms != terms:
                    continue
                return self._hit(e, "semantic", s, now)
        return None

    def _hit(self, e: _Entry, kind: str, sim: float, now: float) -> CacheHit:
        e.hits += 1
        self._entries.move_to_end(e.key)
        return CacheHit(kind, e.response, sim, e.query, now - e.created)

    # ---- insert -------------------------------------------------------------------------------
    def put(self, scope: dict, query: str, response: dict) -> None:
        ns = self.namespace(scope)
        self._insert(self.exact_key(ns, query), ns, query, response, time.time(), persist=True)

    def _insert(self, key, ns, query, response, created, persist) -> None:
        with self._lock:
            if key in self._entries:
                self._entries[key].response = response
                self._entries[key].created = created
                self._entries.move_to_end(key)
            else:
                self._entries[key] = _Entry(key, ns, query, key_terms(query), response, created)
                if self.semantic:
                    v = self.embedder.embed([query])[0]
                    self._index.setdefault(ns, _Index(len(v))).add(key, v)
                while len(self._entries) > self.max_entries:
                    self._evict(next(iter(self._entries)))
            if persist and self._db is not None:
                self._db.execute(
                    "INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?, ?)", (key, ns, query, json.dumps(response), created)
                )
                self._db.commit()

    def _evict(self, key: str) -> None:
        e = self._entries.pop(key)
        if self.semantic and e.namespace in self._index:
            self._index[e.namespace].remove(key)
        if self._db is not None:
            self._db.execute("DELETE FROM cache WHERE key = ?", (key,))

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._index.clear()
            if self._db is not None:
                self._db.execute("DELETE FROM cache")
                self._db.commit()

    def __len__(self) -> int:
        return len(self._entries)


def calibrate_threshold(
    similarity: np.ndarray, correct: np.ndarray, alpha: float = 0.02, delta: float = 0.1, grid: np.ndarray | None = None
) -> risk.Certificate:
    """Pick the lowest similarity threshold whose wrong-answer rate is certified <= alpha.

    similarity[i]: similarity of query i to its best eligible cached entry (-inf if none).
    correct[i]: whether serving that entry would have been a correct answer.
    Loss at threshold t: 1{similarity >= t and not correct}. Settings run from strict to loose.
    """
    similarity = np.asarray(similarity, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    grid = np.round(np.arange(1.0, 0.5, -0.005), 3) if grid is None else grid
    return risk.learn_then_test(list(grid), lambda t: ((similarity >= t) & ~correct).astype(float), alpha, delta)
