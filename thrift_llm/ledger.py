"""Per-request accounting: what each request cost, what it would have cost without Thrift,
and where the savings came from. Feeds /thrift/stats, /metrics and the dashboard."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Event:
    id: str
    ts: float
    tier: str
    model: str
    cache: str  # miss | exact | semantic | off
    routed: bool  # chosen by the router (vs pinned by the client)
    route_score: float | None
    tokens_in_original: int  # prompt tokens before compression
    tokens_in_sent: int  # prompt tokens actually billed (0 on a cache hit)
    tokens_out: int
    cost_usd: float
    baseline_usd: float  # same request, uncompressed, on the strongest tier, no cache
    overhead_ms: float  # time Thrift added (cache lookup, compression, routing)
    upstream_ms: float
    status: int = 200
    fallback: bool = False
    strong_sent_usd: float = 0.0  # what the *compressed* prompt would have cost on the strongest tier

    @property
    def saved_usd(self) -> float:
        return self.baseline_usd - self.cost_usd


class Ledger:
    def __init__(self, path: str | None = None, recent: int = 200):
        self._lock = threading.Lock()
        self.recent: deque[Event] = deque(maxlen=recent)
        self.overheads: deque[float] = deque(maxlen=5000)
        self.t0 = time.time()
        self.totals = {
            "requests": 0, "errors": 0, "cache_exact": 0, "cache_semantic": 0, "routed_cheap": 0, "fallbacks": 0,
            "tokens_in_original": 0, "tokens_in_sent": 0, "tokens_out": 0, "tokens_saved_compression": 0,
            "tokens_saved_cache": 0, "cost_usd": 0.0, "baseline_usd": 0.0,
            "saved_cache_usd": 0.0, "saved_compression_usd": 0.0, "saved_routing_usd": 0.0,
        }
        self.by_tier: dict[str, int] = {}
        self.minutes: dict[int, dict] = {}
        self._db = None
        if path:
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.execute("CREATE TABLE IF NOT EXISTS events (ts REAL, json TEXT)")

    def record(self, e: Event, cheap_tier: str) -> None:
        with self._lock:
            t = self.totals
            t["requests"] += 1
            if e.status != 200:
                t["errors"] += 1
            if e.cache == "exact":
                t["cache_exact"] += 1
            elif e.cache == "semantic":
                t["cache_semantic"] += 1
            if e.cache in ("exact", "semantic"):
                t["tokens_saved_cache"] += e.tokens_in_original + e.tokens_out
                t["saved_cache_usd"] += e.baseline_usd
            elif e.status == 200:
                t["tokens_saved_compression"] += max(0, e.tokens_in_original - e.tokens_in_sent)
                t["saved_compression_usd"] += e.baseline_usd - e.strong_sent_usd
                t["saved_routing_usd"] += e.strong_sent_usd - e.cost_usd
            if e.tier == cheap_tier and e.cache not in ("exact", "semantic"):
                t["routed_cheap"] += 1
            t["fallbacks"] += int(e.fallback)
            t["tokens_in_original"] += e.tokens_in_original
            t["tokens_in_sent"] += e.tokens_in_sent
            t["tokens_out"] += e.tokens_out
            t["cost_usd"] += e.cost_usd
            t["baseline_usd"] += e.baseline_usd
            self.by_tier[e.tier] = self.by_tier.get(e.tier, 0) + 1
            self.recent.append(e)
            self.overheads.append(e.overhead_ms)
            m = int(e.ts // 60)
            b = self.minutes.setdefault(m, {"requests": 0, "cost_usd": 0.0, "baseline_usd": 0.0, "tokens_sent": 0, "tokens_original": 0})
            b["requests"] += 1
            b["cost_usd"] += e.cost_usd
            b["baseline_usd"] += e.baseline_usd
            b["tokens_sent"] += e.tokens_in_sent + (0 if e.cache in ("exact", "semantic") else e.tokens_out)
            b["tokens_original"] += e.tokens_in_original + e.tokens_out
            for old in [k for k in self.minutes if k < m - 180]:
                del self.minutes[old]
            if self._db is not None:
                self._db.execute("INSERT INTO events VALUES (?, ?)", (e.ts, json.dumps(asdict(e))))
                self._db.commit()

    def stats(self) -> dict:
        with self._lock:
            t = dict(self.totals)
            ov = np.array(self.overheads) if self.overheads else np.zeros(1)
            n = max(1, t["requests"])
            original = t["tokens_in_original"] + t["tokens_out"]
            billed = original - t["tokens_saved_cache"] - t["tokens_saved_compression"]
            return {
                "uptime_s": time.time() - self.t0,
                "totals": t,
                "by_tier": dict(self.by_tier),
                "derived": {
                    "savings_usd": t["baseline_usd"] - t["cost_usd"],
                    "savings_pct": 100 * (1 - t["cost_usd"] / t["baseline_usd"]) if t["baseline_usd"] > 0 else 0.0,
                    "cache_hit_rate": (t["cache_exact"] + t["cache_semantic"]) / n,
                    "cheap_route_rate": t["routed_cheap"] / n,
                    "prompt_tokens_saved_pct": 100 * (1 - t["tokens_in_sent"] / t["tokens_in_original"]) if t["tokens_in_original"] else 0.0,
                    "tokens_billed": billed,
                    "tokens_original": original,
                    "overhead_ms_p50": float(np.percentile(ov, 50)),
                    "overhead_ms_p99": float(np.percentile(ov, 99)),
                },
                "minutes": [{"t": k * 60, **v} for k, v in sorted(self.minutes.items())],
                "recent": [asdict(e) for e in list(self.recent)[-50:]][::-1],
            }

    def prometheus(self) -> str:
        s = self.stats()
        t, d = s["totals"], s["derived"]
        lines = [
            "# HELP thrift_requests_total Requests handled.", "# TYPE thrift_requests_total counter",
            f"thrift_requests_total {t['requests']}",
            "# TYPE thrift_errors_total counter", f"thrift_errors_total {t['errors']}",
            "# TYPE thrift_cache_hits_total counter",
            f'thrift_cache_hits_total{{kind="exact"}} {t["cache_exact"]}',
            f'thrift_cache_hits_total{{kind="semantic"}} {t["cache_semantic"]}',
            "# TYPE thrift_tier_requests_total counter",
            *[f'thrift_tier_requests_total{{tier="{k}"}} {v}' for k, v in s["by_tier"].items()],
            "# TYPE thrift_tokens_total counter",
            f'thrift_tokens_total{{kind="prompt_original"}} {t["tokens_in_original"]}',
            f'thrift_tokens_total{{kind="prompt_sent"}} {t["tokens_in_sent"]}',
            f'thrift_tokens_total{{kind="completion"}} {t["tokens_out"]}',
            "# TYPE thrift_cost_usd_total counter",
            f'thrift_cost_usd_total{{kind="actual"}} {t["cost_usd"]:.6f}',
            f'thrift_cost_usd_total{{kind="baseline"}} {t["baseline_usd"]:.6f}',
            "# TYPE thrift_overhead_ms gauge",
            f'thrift_overhead_ms{{quantile="0.5"}} {d["overhead_ms_p50"]:.3f}',
            f'thrift_overhead_ms{{quantile="0.99"}} {d["overhead_ms_p99"]:.3f}',
        ]
        return "\n".join(lines) + "\n"
