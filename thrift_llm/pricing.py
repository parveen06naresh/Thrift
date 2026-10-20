"""USD list prices per 1M tokens (input, output). Override or extend in config `[prices]`.

These are public list prices as published by each provider at the time the HELM runs in
results/ were priced (checked 2026-10). Prices change; the gateway reports savings with
whatever table it is given and never guesses: an unknown model is priced at 0 and flagged.
"""
from __future__ import annotations

PRICES: dict[str, tuple[float, float]] = {
    # Anthropic Claude 5 (current; platform.claude.com pricing page, read 2026-10-20). Haiku 5.5: prompts <= 100k tokens.
    "claude-fable-5-1": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    # Models in the HELM Capabilities benchmark (v1.15.0, Nov 2025), list prices at that time
    "gpt-5.1-2025-11-13": (1.25, 10.00),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5-2025-08-07": (1.25, 10.00),
    "gpt-5-mini-2025-08-07": (0.25, 2.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano-2025-08-07": (0.05, 0.40),
    "gpt-5-nano": (0.05, 0.40),
    "claude-sonnet-4-5-20250929": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gemini-3-pro-preview": (2.00, 12.00),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "grok-4-0709": (3.00, 15.00),
    "grok-4": (3.00, 15.00),
    "grok-3-mini-beta": (0.30, 0.50),
    "grok-3-mini": (0.30, 0.50),
    "kimi-k2-instruct": (0.60, 2.50),
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    # Anthropic
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    # Google (<=128k-token prompts)
    "gemini-1.5-flash-002": (0.075, 0.30),
    "gemini-1.5-pro-002": (1.25, 5.00),
    # DeepSeek (deepseek-chat = V3, standard hours)
    "deepseek-v3": (0.27, 1.10),
    "deepseek-chat": (0.27, 1.10),
    # Mistral
    "mistral-large-2407": (2.00, 6.00),
    # Meta Llama 3.1 (Together AI serverless list prices)
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": (3.50, 3.50),
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": (0.18, 0.18),
    "llama-3.1-405b-instruct-turbo": (3.50, 3.50),
    "llama-3.1-8b-instruct-turbo": (0.18, 0.18),
}


# Provider prompt caching: (price multiplier for cached input tokens, multiplier for tokens
# written to the cache), by model-name prefix. Unlisted models get no discount, so savings are
# never overstated. Usage fields are normalised to OpenAI's prompt_tokens_details.cached_tokens.
CACHE_MULT: list[tuple[str, float, float]] = [
    ("claude-opus-5", 0.05, 1.25),  # Claude 5: cache reads 5% (Fable 2.5%), 5-minute writes 125%
    ("claude-sonnet-5", 0.05, 1.25),
    ("claude-haiku-5", 0.10, 1.25),
    ("claude-fable-5", 0.025, 1.25),
    ("claude", 0.10, 1.25),  # earlier Claude: reads 10%, 5-minute writes 125%
    ("gpt-4o", 0.50, 1.0),
    ("gpt-4.1", 0.25, 1.0),
    ("deepseek", 0.10, 1.0),
    ("gemini", 0.25, 1.0),
]


def cached_tokens(usage: dict | None) -> tuple[int, int]:
    """(cached read tokens, cache write tokens) from any provider's usage block."""
    u = usage or {}
    read = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or u.get("prompt_cache_hit_tokens") or 0
    return int(read), int(u.get("thrift_cache_write_tokens") or 0)


class PriceBook:
    def __init__(self, overrides: dict | None = None):
        self.table = dict(PRICES)
        for k, v in (overrides or {}).items():
            self.table[k] = (float(v[0]), float(v[1]))
        self.unknown: set[str] = set()

    def lookup(self, model: str) -> tuple[float, float] | None:
        if model in self.table:
            return self.table[model]
        base = model.split("/")[-1]
        return self.table.get(base)

    def cost(self, model: str, tokens_in: int, tokens_out: int, usage: dict | None = None) -> float:
        p = self.lookup(model)
        if p is None:
            self.unknown.add(model)
            return 0.0
        read, write = cached_tokens(usage)
        rm, wm = next(((r, w) for pre, r, w in CACHE_MULT if model.split("/")[-1].startswith(pre)), (1.0, 1.0))
        read, write = min(read, tokens_in), min(write, max(0, tokens_in - read))
        plain = tokens_in - read - write
        return ((plain + read * rm + write * wm) * p[0] + tokens_out * p[1]) / 1e6
