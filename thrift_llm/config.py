"""Configuration: a TOML file (stdlib tomllib on 3.11+, `tomli` on 3.10) plus env overrides."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


# provider -> (API style, base URL, env var holding the key). Every provider except Anthropic
# speaks the OpenAI chat-completions API; Anthropic gets a native adapter (backends.py) so that
# prompt caching and exact token usage work.
PROVIDERS: dict[str, tuple[str, str, str | None]] = {
    "openai": ("openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "anthropic": ("anthropic", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
    "xai": ("openai", "https://api.x.ai/v1", "XAI_API_KEY"),
    "perplexity": ("openai", "https://api.perplexity.ai", "PERPLEXITY_API_KEY"),
    "gemini": ("openai", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "deepseek": ("openai", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "mistral": ("openai", "https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "moonshot": ("openai", "https://api.moonshot.ai/v1", "MOONSHOT_API_KEY"),
    "groq": ("openai", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together": ("openai", "https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "fireworks": ("openai", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "openrouter": ("openai", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("openai", "http://localhost:11434/v1", None),
    "vllm": ("openai", "http://localhost:8000/v1", None),
}
ALIASES = {"chatgpt": "openai", "claude": "anthropic", "grok": "xai", "google": "gemini", "sonar": "perplexity", "kimi": "moonshot"}
# Providers whose answers come from a live web search: caching them would serve stale news.
LIVE_PROVIDERS = {"perplexity"}


@dataclass
class Tier:
    name: str
    model: str
    provider: str = "openai"
    base_url: str | None = None  # default: the provider's public endpoint
    api_key_env: str | None = None  # default: the provider's usual variable, e.g. ANTHROPIC_API_KEY
    api_key: str | None = None
    timeout_s: float = 120.0
    headers: dict = field(default_factory=dict)
    cache: bool | None = None  # None: on, except for live-search providers (Perplexity)
    prompt_cache: bool = True  # Anthropic: mark long stable prefixes cacheable (cached reads cost 10%)
    fallback: list = field(default_factory=list)  # tiers to try, in order, if this one fails; default: the strongest

    def __post_init__(self):
        self.provider = ALIASES.get(self.provider.lower(), self.provider.lower())
        if self.provider not in PROVIDERS:
            raise ValueError(f"tier {self.name!r}: unknown provider {self.provider!r}; known: {sorted(PROVIDERS)}")
        _, url, env = PROVIDERS[self.provider]
        if self.base_url is None:
            self.base_url = url
        if self.api_key_env is None:
            self.api_key_env = env
        if self.cache is None:
            self.cache = self.provider not in LIVE_PROVIDERS

    @property
    def api_style(self) -> str:
        return "mock" if self.base_url.startswith("mock://") else PROVIDERS[self.provider][0]

    def key(self) -> str | None:
        return self.api_key or (os.environ.get(self.api_key_env) if self.api_key_env else None)


@dataclass
class RouterConfig:
    enabled: bool = True
    path: str | None = None  # None: the router bundled with Thrift (trained on HELM, see README)
    threshold: float | None = None  # override the calibrated threshold
    auto_models: list = field(default_factory=lambda: ["thrift-auto", "auto"])


@dataclass
class CacheConfig:
    enabled: bool = True
    semantic: bool = True
    threshold: float = 0.92
    ttl_s: float = 7 * 24 * 3600
    max_entries: int = 50_000
    max_temperature: float = 2.0  # don't cache requests sampled hotter than this
    path: str | None = None


@dataclass
class CompressConfig:
    enabled: bool = True
    normalize: bool = True
    dedup: bool = True
    select: bool = True
    keep_ratio: float = 0.5
    min_tokens: int = 500
    keep_recent_turns: int = 2


@dataclass
class DriftConfig:
    enabled: bool = True  # active whenever shadow traffic is on ([shadow] rate > 0)
    epsilon: float | None = None  # default: the epsilon the router was certified at
    alpha: float = 0.05  # false-alarm probability over the whole stream
    judge: str = "llm"  # "llm": the judge tier grades the cheap answer against the strong one; "agreement": token overlap
    judge_tier: str | None = None  # default: the strongest tier
    action: str = "safe_mode"  # "safe_mode": route everything to the strongest tier on alarm; "alert": only report


@dataclass
class Config:
    tiers: list[Tier] = field(default_factory=list)  # ordered cheapest -> strongest
    router: RouterConfig = field(default_factory=RouterConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    compress: CompressConfig = field(default_factory=CompressConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    prices: dict = field(default_factory=dict)
    ledger_path: str | None = None
    shadow_rate: float = 0.0  # fraction of cheaper-routed requests also sent to the strongest tier (feeds the drift guard)
    shadow_path: str = "thrift-shadow.jsonl"
    api_keys: list = field(default_factory=list)  # if set, clients must send one as Bearer token
    host: str = "127.0.0.1"
    port: int = 8787

    @property
    def strong(self) -> Tier:
        return self.tiers[-1]

    @property
    def cheap(self) -> Tier:
        return self.tiers[0]

    def tier(self, name_or_model: str) -> Tier | None:
        for t in self.tiers:
            if name_or_model in (t.name, t.model):
                return t
        return None


def _section(cls, d: dict | None):
    d = d or {}
    known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
    unknown = set(d) - set(known)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**known)


def from_dict(d: dict) -> Config:
    server = d.get("server", {})
    cfg = Config(
        tiers=[_section(Tier, t) for t in d.get("tier", [])],
        router=_section(RouterConfig, d.get("router")),
        cache=_section(CacheConfig, d.get("cache")),
        compress=_section(CompressConfig, d.get("compress")),
        drift=_section(DriftConfig, d.get("drift")),
        prices=d.get("prices", {}),
        ledger_path=d.get("ledger", {}).get("path"),
        shadow_rate=float(d.get("shadow", {}).get("rate", 0.0)),
        shadow_path=d.get("shadow", {}).get("path", "thrift-shadow.jsonl"),
        api_keys=server.get("api_keys", []),
        host=server.get("host", "127.0.0.1"),
        port=int(server.get("port", 8787)),
    )
    if not cfg.tiers:
        raise ValueError("config needs at least one [[tier]] (cheapest first, strongest last)")
    if cfg.drift.judge not in ("llm", "agreement") or cfg.drift.action not in ("safe_mode", "alert"):
        raise ValueError('[drift] judge must be "llm" or "agreement", action "safe_mode" or "alert"')
    names = [t.name for t in cfg.tiers]
    if len(set(names)) != len(names):
        raise ValueError(f"tier names must be unique: {names}")
    for t in cfg.tiers:
        bad = [f for f in t.fallback if f not in names]
        if bad:
            raise ValueError(f"tier {t.name!r}: fallback names unknown tiers {bad}")
    if cfg.drift.judge_tier and cfg.drift.judge_tier not in names:
        raise ValueError(f"[drift] judge_tier {cfg.drift.judge_tier!r} is not a tier")
    return cfg


def parse_spec(spec: str) -> dict:
    """"provider:model" or "provider:model=name" or "name=provider:model" -> a [[tier]] table."""
    name = None
    if "=" in spec:
        left, right = spec.split("=", 1)
        if ":" in left:
            spec, name = left, right
        else:
            name, spec = left, right
    if ":" not in spec:
        raise ValueError(f"tier spec {spec!r}: expected provider:model, e.g. anthropic:claude-sonnet-5-5")
    provider, model = spec.split(":", 1)
    if provider.lower() == "mock":
        return {"name": name or model, "model": model, "base_url": f"mock://{name or model}"}
    return {"name": name or model, "model": model, "provider": provider}


def from_specs(specs: list[str], **sections) -> Config:
    """Config from short tier specs, cheapest first: ["openai:gpt-5-mini", "anthropic:claude-sonnet-5-5"]."""
    return from_dict({"tier": [parse_spec(s) for s in specs], **sections})


def load(path: str) -> Config:
    with open(path, "rb") as f:
        return from_dict(tomllib.load(f))


def demo_config() -> Config:
    """Two mock tiers priced like gpt-4o-mini / gpt-4o: runs with no network and no API key."""
    return from_dict(
        {
            "tier": [
                {"name": "cheap", "model": "gpt-4o-mini", "base_url": "mock://cheap?latency_ms=40"},
                {"name": "strong", "model": "gpt-4o", "base_url": "mock://strong?latency_ms=120"},
            ]
        }
    )
