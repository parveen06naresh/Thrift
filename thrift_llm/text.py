"""Shared text utilities: normalisation, tokenising, hashed embeddings, key-term extraction."""
from __future__ import annotations

import hashlib
import re
import zlib

import numpy as np

WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
STOP = frozenset(
    """a an the of to in on for and or but is are was were be been being am do does did done have has had
    i me my we our you your he she it its they them their this that these those there here what which who whom
    whose when where why how can could would should will shall may might must at by from with about as into
    than then so if not no yes up out over under again very just also any all some most more such only own
    same too s t don doesn didn isn aren wasn weren won wouldn shouldn couldn get got best way much many""".split()
)


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in STOP]


def _h(s: str) -> int:
    return zlib.crc32(s.encode())


class HashingEmbedder:
    """Dependency-free sentence embedding: signed feature hashing of word unigrams, word bigrams
    and character 3-5-grams, sublinear TF, L2-normalised. Deterministic, ~20 us per query.

    It is a lexical embedding: it catches rephrasings and typos, not synonyms. Thrift's cache
    can take any embedder with the same `embed(list[str]) -> np.ndarray` interface (e.g. a
    sentence-transformers model) via `thrift_llm.cache.SemanticCache(embedder=...)`.
    """

    name = "hashing-v1"

    def __init__(self, dim: int = 2048, char: bool = True):
        self.dim = dim
        self.char = char

    def _features(self, text: str) -> dict[int, float]:
        f: dict[int, float] = {}
        ws = words(text)
        cw = [w for w in ws if w not in STOP]

        def add(key: str, w: float) -> None:
            h = _h(key)
            idx = h % self.dim
            sign = 1.0 if (h >> 31) & 1 else -1.0
            f[idx] = f.get(idx, 0.0) + sign * w

        for w in cw:
            add("w:" + w, 1.0)
        for w in ws:
            if w in STOP:
                add("w:" + w, 0.15)
        for a, b in zip(cw, cw[1:]):
            add("b:" + a + "_" + b, 0.7)
        if self.char:
            s = " " + " ".join(cw) + " "
            for n in (3, 4, 5):
                for i in range(len(s) - n + 1):
                    add("c:" + s[i : i + n], 0.25)
        return f

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            for i, v in self._features(t).items():
                out[r, i] = np.sign(v) * np.log1p(abs(v))
            n = np.linalg.norm(out[r])
            if n > 0:
                out[r] /= n
        return out


def key_terms(text: str) -> frozenset[str]:
    """Terms that change the answer if they change: numbers and negations.

    Two questions can be 0.95-similar and still need different answers ("convert 5 km" vs
    "convert 50 km", "is it safe" vs "is it not safe"). The cache refuses a semantic hit when
    these sets differ.
    """
    t = text.lower()
    terms = {"#" + n.replace(",", "") for n in NUMBER.findall(t)}
    for neg in ("not", "never", "no", "without", "n't"):
        if re.search(r"\b" + re.escape(neg) + r"\b" if neg != "n't" else "n't", t):
            terms.add("!neg")
    return frozenset(terms)


def stable_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\x00")
    return h.hexdigest()


def normalize_query(text: str) -> str:
    return " ".join(text.lower().split()).strip(" ?!.")
