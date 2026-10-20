"""Prompt compression.

Three stages, applied in order. Each is optional and reported separately.

1. normalize (lossless): collapse whitespace runs, blank-line runs, decorative rules and
   invisible characters outside fenced code blocks. Tokenizers spend tokens on all of these.
   Code blocks are never touched.
2. dedup (lossless for meaning): drop paragraphs that repeat an earlier paragraph verbatim
   (after normalisation), e.g. the same RAG chunk retrieved twice or a re-pasted document.
3. select (lossy, query-aware): for long context (RAG passages, pasted documents, old chat
   history), keep the sentences most relevant to the user's actual question (BM25 against the
   final paragraph of the last user message) up to `keep_ratio` of the tokens, in original
   order. The last user paragraph, the system prompt and the most recent turns are never cut.

`keep_ratio` is the only quality knob. It is calibrated with Learn-then-Test against a reader
model so the expected quality drop is certified (see scripts/eval_compress.py).
"""
from __future__ import annotations

import copy
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from . import tokens as tok
from .text import STOP, words

_FENCE = re.compile(r"(```.*?```|~~~.*?~~~)", re.S)
_INVIS = re.compile(r"[​‌‍⁠﻿­]")
_SPACES = re.compile(r"[ \t  - 　]+")
_TRAIL = re.compile(r"[ \t]+\n")
_BLANKS = re.compile(r"\n\s*\n(\s*\n)+")
_RULES = re.compile(r"([-=_*~#.])\1{4,}")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def normalize_text(text: str) -> str:
    parts = _FENCE.split(text)
    out = []
    for i, p in enumerate(parts):
        if i % 2 == 1:  # fenced code: keep exactly
            out.append(p)
            continue
        p = _INVIS.sub("", p)
        p = _SPACES.sub(" ", p)
        p = _TRAIL.sub("\n", p)
        p = _BLANKS.sub("\n\n", p)
        p = _RULES.sub(lambda m: m.group(1) * 3, p)
        out.append(p)
    return "".join(out).strip()


def _paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


def _units(par: str, max_tokens: int = 80) -> list[str]:
    """Selection units: a short paragraph stays whole (keeps Q/A pairs and list items together);
    a long one is split into lines if it is tabular, else into sentences."""
    if tok.count(par) <= max_tokens:
        return [par]
    if par.count("\n") >= 4:
        return [ln for ln in par.split("\n") if ln.strip()]
    return [s for s in _SENT.split(par) if s and s.strip()]


@dataclass
class CompressionResult:
    messages: list[dict]
    tokens_before: int
    tokens_after: int
    stages: dict = field(default_factory=dict)  # stage -> tokens removed

    @property
    def saved(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def ratio(self) -> float:
        return self.tokens_after / max(1, self.tokens_before)


class Compressor:
    def __init__(
        self,
        normalize: bool = True,
        dedup: bool = True,
        select: bool = True,
        keep_ratio: float = 0.5,
        min_tokens: int = 500,
        keep_recent_turns: int = 2,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
    ):
        self.normalize = normalize
        self.dedup = dedup
        self.select = select
        self.keep_ratio = keep_ratio
        self.min_tokens = min_tokens
        self.keep_recent_turns = keep_recent_turns
        self.k1 = bm25_k1
        self.b = bm25_b

    # ---- public -------------------------------------------------------------------------------
    def compress(self, messages: list[dict], keep_ratio: float | None = None) -> CompressionResult:
        ratio = self.keep_ratio if keep_ratio is None else keep_ratio
        msgs = copy.deepcopy(messages)
        before = tok.count_messages(msgs)
        stages = {}

        if self.normalize:
            for m in msgs:
                _map_text(m, normalize_text)
            stages["normalize"] = before - tok.count_messages(msgs)

        if self.dedup:
            t0 = tok.count_messages(msgs)
            seen: set[str] = set()
            for m in msgs:
                _map_text(m, lambda t: self._dedup(t, seen))
            stages["dedup"] = t0 - tok.count_messages(msgs)

        if self.select and ratio < 1.0:
            t0 = tok.count_messages(msgs)
            query = self._query(msgs)
            for i, m in enumerate(msgs):
                if self._eligible(msgs, i):
                    _map_text(m, lambda t: self.select_text(t, query, ratio, protect_last=(i == self._last_user(msgs))))
            stages["select"] = t0 - tok.count_messages(msgs)

        return CompressionResult(msgs, before, tok.count_messages(msgs), stages)

    def select_text(self, text: str, query: str, ratio: float, protect_last: bool = False) -> str:
        """Keep the sentences of `text` most relevant to `query`, up to `ratio` of its tokens."""
        if _FENCE.search(text):
            return text  # documents with code are left alone; cutting code is never safe
        pars = _paragraphs(text)
        cut = _question_index(pars) if (protect_last and pars) else len(pars)
        tail = "\n\n".join(pars[cut:]) if cut < len(pars) else None
        body = pars[:cut]
        units: list[tuple[int, str]] = []  # (paragraph index, sentence)
        for pi, p in enumerate(body):
            for s in _units(p):
                units.append((pi, s.strip()))
        if len(units) < 3:
            return text
        sizes = [tok.count(u) for _, u in units]
        budget = ratio * sum(sizes)
        scores = self._bm25([u for _, u in units], query)
        # Small lead bias: the first sentence of a document often names its subject.
        scores[0] += 0.1 * (max(scores) if max(scores) > 0 else 1.0)
        keep, used = set(), 0
        for j in sorted(range(len(units)), key=lambda j: -scores[j]):
            if used + sizes[j] > budget and keep:
                continue
            keep.add(j)
            used += sizes[j]
        out_pars: dict[int, list[str]] = {}
        for j in sorted(keep):
            out_pars.setdefault(units[j][0], []).append(units[j][1])
        kept = [("\n" if "\n" in body[pi] else " ").join(v) for pi, v in sorted(out_pars.items())]
        if tail is not None:
            kept.append(tail)
        return "\n\n".join(kept)

    # ---- internals ----------------------------------------------------------------------------
    @staticmethod
    def _dedup(text: str, seen: set[str]) -> str:
        out = []
        for p in _paragraphs(text):
            key = " ".join(p.lower().split())
            if len(key) >= 40 and key in seen:
                continue
            seen.add(key)
            out.append(p)
        return "\n\n".join(out) if out else text

    @staticmethod
    def _last_user(msgs: list[dict]) -> int:
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                return i
        return -1

    def _query(self, msgs: list[dict]) -> str:
        i = self._last_user(msgs)
        if i < 0:
            return ""
        pars = _paragraphs(tok.content_text(msgs[i].get("content")))
        if not pars:
            return ""
        return "\n".join(pars[_question_index(pars) :])[-2000:]

    def _eligible(self, msgs: list[dict], i: int) -> bool:
        m = msgs[i]
        role = m.get("role")
        if role not in ("user", "tool", "assistant") or not isinstance(m.get("content"), str):
            return False
        if tok.count(m["content"]) < self.min_tokens:
            return False
        last_user = self._last_user(msgs)
        if i == last_user:
            return True  # long context pasted with the question; the question paragraph is protected
        # Older history beyond the most recent turns.
        recent_start = len(msgs) - 2 * self.keep_recent_turns
        return i < recent_start or role == "tool"

    def _bm25(self, units: list[str], query: str) -> list[float]:
        q = [w for w in words(query) if w not in STOP]
        if not q:
            return [0.0] * len(units)
        docs = [Counter(words(u)) for u in units]
        n = len(docs)
        avgdl = sum(sum(d.values()) for d in docs) / n or 1.0
        df = Counter(w for d in docs for w in set(d))
        out = []
        for d in docs:
            dl = sum(d.values()) or 1
            s = 0.0
            for w in set(q):
                if w not in d:
                    continue
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                f = d[w]
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avgdl))
            out.append(s)
        return out


def _question_index(pars: list[str]) -> int:
    """Index of the paragraph holding the user's question: the last one with a '?' among the
    final three paragraphs (instructions often follow the question), else the last one."""
    for j in range(len(pars) - 1, max(-1, len(pars) - 4), -1):
        if "?" in pars[j]:
            return j
    return len(pars) - 1


def _map_text(m: dict, fn) -> None:
    c = m.get("content")
    if isinstance(c, str):
        m["content"] = fn(c)
    elif isinstance(c, list):
        for part in c:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                part["text"] = fn(part["text"])
