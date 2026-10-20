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
    # Meta Llama 3.1 (Together AI serverless list prices)
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": (3.50, 3.50),
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": (0.18, 0.18),
    "llama-3.1-405b-instruct-turbo": (3.50, 3.50),
    "llama-3.1-8b-instruct-turbo": (0.18, 0.18),
}


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

    def cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        p = self.lookup(model)
        if p is None:
            self.unknown.add(model)
            return 0.0
        return (tokens_in * p[0] + tokens_out * p[1]) / 1e6
