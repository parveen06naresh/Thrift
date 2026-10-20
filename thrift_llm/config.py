"""Configuration: a TOML file (stdlib tomllib on 3.11+, `tomli` on 3.10) plus env overrides."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


@dataclass
class Tier:
    name: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    timeout_s: float = 120.0
    headers: dict = field(default_factory=dict)

    def key(self) -> str | None:
        return self.api_key or os.environ.get(self.api_key_env)


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
class Config:
    tiers: list[Tier] = field(default_factory=list)  # ordered cheapest -> strongest
    router: RouterConfig = field(default_factory=RouterConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    compress: CompressConfig = field(default_factory=CompressConfig)
    prices: dict = field(default_factory=dict)
    ledger_path: str | None = None
    shadow_rate: float = 0.0  # fraction of cheap-routed requests also sent to the strong tier
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
    return cfg


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
