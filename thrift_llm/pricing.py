"""USD list prices per 1M tokens (input, output). Override or extend in config `[prices]`.

These are public list prices as published by each provider at the time the HELM runs in
results/ were priced (checked 2026-10). Prices change; the gateway reports savings with
whatever table it is given and never guesses: an unknown model is priced at 0 and flagged.
"""
from __future__ import annotations

PRICES: dict[str, tuple[float, float]] = {
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
    ("claude", 0.10, 1.25),  # Anthropic: reads 10%, 5-minute writes 125%
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
