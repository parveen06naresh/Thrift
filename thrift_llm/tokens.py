"""Token counting.

Uses tiktoken's o200k_base (GPT-4o family) when it is installed and its vocabulary is
available, and falls back to a calibrated character heuristic otherwise. Counting is only
used for accounting and budgets; the upstream provider's `usage` field is always the
source of truth for billing.
"""
from __future__ import annotations

import functools
import math
import re

_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@functools.lru_cache(maxsize=4)
def _encoding(name: str):
    try:
        import tiktoken

        return tiktoken.get_encoding(name)
    except Exception:  # noqa: BLE001 - not installed, or vocabulary not downloadable
        return None


def backend() -> str:
    return "tiktoken/o200k_base" if _encoding("o200k_base") is not None else "heuristic"


def count(text: str, encoding: str = "o200k_base") -> int:
    if not text:
        return 0
    enc = _encoding(encoding)
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    # Heuristic: ~0.75 tokens per word-piece for English prose, with long words split.
    # Within ~10% of o200k_base on the HELM prompts (see tests/test_tokens.py).
    n = 0.0
    for w in _WORD.findall(text):
        n += 1 if len(w) <= 6 else math.ceil(len(w) / 5)
    return max(1, int(round(n * 0.92)))


def content_text(content) -> str:
    """Flatten OpenAI message content (str or list of parts) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            out.append(part.get("text", ""))
    return "\n".join(out)


def count_messages(messages: list[dict], encoding: str = "o200k_base") -> int:
    """Prompt tokens for a chat request (OpenAI's documented ~3 tokens/message overhead)."""
    total = 3
    for m in messages:
        total += 3 + count(content_text(m.get("content")), encoding)
        if m.get("name"):
            total += 1
    return total
