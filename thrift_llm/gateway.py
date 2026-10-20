"""The request pipeline, independent of any web framework.

    request -> [cache lookup] -> [compress] -> [route] -> upstream -> [cache store] -> ledger

Use it in-process (`await Gateway(cfg).chat(body)`) or behind the HTTP server (server.py).
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import random
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from importlib import resources

from . import tokens as tok
from .backends import UpstreamError, make_backend
from .cache import SemanticCache
from .compress import Compressor
from .config import Config
from .drift import JUDGE_PROMPT, RiskMonitor, agreement_loss, parse_judgement
from .ledger import Event, Ledger
from .pricing import PriceBook
from .router import Ladder, Router, load_policy

log = logging.getLogger("thrift")
THRIFT_KEYS = ("thrift",)
SCOPE_PARAMS = (
    "temperature", "top_p", "max_tokens", "max_completion_tokens", "stop", "seed", "tools", "tool_choice",
    "response_format", "functions", "function_call", "logit_bias", "presence_penalty", "frequency_penalty",
)


@dataclass
class Options:
    cache: bool = True
    compress: bool = True
    route: bool = True
    keep_ratio: float | None = None

    @classmethod
    def parse(cls, body: dict, headers: dict) -> Options:
        o = cls()
        h = {k.lower(): str(v).lower() for k, v in (headers or {}).items()}
        if h.get("x-thrift-bypass") in ("1", "true", "yes"):
            return cls(False, False, False)
        if h.get("x-thrift-cache") == "off":
            o.cache = False
        if h.get("x-thrift-compress") == "off":
            o.compress = False
        t = body.get("thrift") or {}
        if isinstance(t, dict):
            o.cache = bool(t.get("cache", o.cache))
            c = t.get("compress", o.compress)
            if isinstance(c, (int, float)) and not isinstance(c, bool):
                o.keep_ratio = float(c)
                o.compress = True
            else:
                o.compress = bool(c)
            o.route = bool(t.get("route", o.route))
        return o


@dataclass
class Result:
    response: dict | None  # non-streaming body
    stream: AsyncIterator[dict] | None  # streaming chunks
    headers: dict = field(default_factory=dict)
    status: int = 200


BUNDLED = {"helm": "data/router_helm.json", "ladder": "data/ladder_helm.json"}


def bundled_router(name: str = "helm") -> Router | Ladder | None:
    try:
        p = resources.files("thrift_llm").joinpath(BUNDLED[name])
        with resources.as_file(p) as f:
            return load_policy(str(f))
    except (FileNotFoundError, ValueError, ModuleNotFoundError, KeyError):
        return None


class Gateway:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backends = {t.name: make_backend(t) for t in cfg.tiers}
        self.prices = PriceBook(cfg.prices)
        self.ledger = Ledger(cfg.ledger_path)
        self.cache = (
            SemanticCache(cfg.cache.threshold, cfg.cache.max_entries, cfg.cache.ttl_s, path=cfg.cache.path, semantic=cfg.cache.semantic)
            if cfg.cache.enabled
            else None
        )
        c = cfg.compress
        self.compressor = Compressor(c.normalize, c.dedup, c.select, c.keep_ratio, c.min_tokens, c.keep_recent_turns) if c.enabled else None
        tok.count("warm up")  # load the tokenizer now, not on the first request
        self.router = None
        if cfg.router.enabled and len(cfg.tiers) >= 2:
            p = cfg.router.path
            if p and p.startswith("bundled:"):
                self.router = bundled_router(p.split(":", 1)[1])
                if self.router is None:
                    raise ValueError(f"no bundled router {p!r}; choose from {['bundled:' + k for k in BUNDLED]}")
            else:
                self.router = load_policy(p) if p else bundled_router()
            if isinstance(self.router, Ladder):
                missing = [m for m in self.router.models if cfg.tier(m) is None]
                if missing:
                    raise ValueError(f"router ladder names tiers {missing} that are not in the config")
            if self.router is not None and cfg.router.threshold is not None:
                self.router.threshold = cfg.router.threshold
        self.monitor = None
        d = cfg.drift
        if d.enabled and cfg.shadow_rate > 0 and self.router is not None:
            eps = d.epsilon or (self.router.certificate or {}).get("epsilon") or 0.02
            self.monitor = RiskMonitor(eps, d.alpha)
        self._tasks: set = set()

    # ---- helpers ------------------------------------------------------------------------------
    def _pick_tier(self, body: dict, opts: Options, text: str) -> tuple[str, bool, float | None]:
        model = body.get("model") or "thrift-auto"
        if model not in self.cfg.router.auto_models:
            t = self.cfg.tier(model)
            return (t.name if t else self.cfg.strong.name), False, None
        if not opts.route or self.router is None:
            return self.cfg.strong.name, False, None
        if self.safe_mode:
            return self.cfg.strong.name, True, None
        name, score = self.router.route(text)
        if isinstance(self.router, Router):  # two-tier routers name abstract "cheap" / "strong"
            name = self.cfg.cheap.name if name == self.router.cheap else self.cfg.strong.name
        return self.cfg.tier(name).name, True, score

    @property
    def safe_mode(self) -> bool:
        return self.monitor is not None and self.monitor.alarm and self.cfg.drift.action == "safe_mode"

    def _chain(self, tier: str) -> list[str]:
        """Tiers to try in order: the chosen one, then its fallbacks (default: the strongest)."""
        t = self.cfg.tier(tier)
        rest = t.fallback or [self.cfg.strong.name]
        out = [tier]
        for n in rest:
            if n not in out:
                out.append(n)
        return out

    @staticmethod
    def _scope(body: dict, tier: str) -> tuple[dict, str]:
        msgs = body.get("messages", [])
        last = next((i for i in range(len(msgs) - 1, -1, -1) if msgs[i].get("role") == "user"), None)
        query = tok.content_text(msgs[last].get("content")) if last is not None else ""
        prefix = msgs[:last] if last is not None else msgs
        scope = {"tier": tier, "prefix": prefix, "after": msgs[last + 1 :] if last is not None else [], **{k: body.get(k) for k in SCOPE_PARAMS if k in body}}
        return scope, query

    def _cacheable(self, body: dict, opts: Options, tier: str) -> bool:
        if self.cache is None or not opts.cache:
            return False
        if not all(self.cfg.tier(n).cache for n in self._chain(tier)):
            return False
        if (body.get("n") or 1) > 1:
            return False
        return float(body.get("temperature", 1.0) or 0.0) <= self.cfg.cache.max_temperature

    def _cost(self, tier: str, tin: int, tout: int, usage: dict | None = None) -> float:
        return self.prices.cost(self.cfg.tier(tier).model, tin, tout, usage)

    # ---- main entry ---------------------------------------------------------------------------
    async def chat(self, body: dict, headers: dict | None = None) -> Result:
        t0 = time.perf_counter()
        opts = Options.parse(body, headers or {})
        body = {k: v for k, v in body.items() if k not in THRIFT_KEYS}
        rid = "thr-" + uuid.uuid4().hex[:16]
        stream = bool(body.get("stream"))
        msgs = body.get("messages") or []
        original_tokens = tok.count_messages(msgs)
        last_text = next((tok.content_text(m.get("content")) for m in reversed(msgs) if m.get("role") == "user"), "")

        tier, routed, score = self._pick_tier(body, opts, last_text)
        strong = self.cfg.strong.name
        cacheable = self._cacheable(body, opts, tier)

        # 1. cache (scoped to the tier the request would be served by)
        scope, query = self._scope(body, tier)
        if cacheable:
            hit = self.cache.get(scope, query)
            if hit is not None:
                resp = copy.deepcopy(hit.response)
                resp["id"] = "chatcmpl-" + rid
                out_tokens = int((resp.get("usage") or {}).get("completion_tokens", 0))
                baseline = self._cost(strong, original_tokens, out_tokens)
                overhead = (time.perf_counter() - t0) * 1000
                self.ledger.record(
                    Event(rid, time.time(), tier, self.cfg.tier(tier).model, hit.kind, routed, score, original_tokens, 0, out_tokens,
                          0.0, baseline, overhead, 0.0), self.cfg.strong.name)
                hdr = self._headers(rid, tier, hit.kind, original_tokens, 0, 0.0, baseline, score)
                hdr["x-thrift-cache-similarity"] = f"{hit.similarity:.3f}"
                resp["usage"] = {**(resp.get("usage") or {}), "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                                 "thrift_cached": True}
                if stream:
                    return Result(None, _replay(resp), hdr)
                return Result(resp, None, hdr)

        # 2. compress
        send = body
        if self.compressor is not None and opts.compress and msgs:
            cr = self.compressor.compress(msgs, opts.keep_ratio)
            if cr.saved > 0:
                send = {**body, "messages": cr.messages}
        overhead = (time.perf_counter() - t0) * 1000

        # 3. upstream (with fallback to the strong tier if the cheap one fails)
        if stream:
            return Result(None, self._stream(send, tier, routed, score, rid, original_tokens, overhead, scope, query, cacheable),
                          self._headers(rid, tier, "miss" if cacheable else "off", original_tokens, None, None, None, score))
        t1 = time.perf_counter()
        chosen = tier
        resp = None
        for tier in self._chain(chosen):
            try:
                resp = await self.backends[tier].chat(send)
                break
            except UpstreamError as e:
                err = e
        if resp is None:
            self._record_error(rid, tier, routed, score, original_tokens, overhead, err.status)
            return Result(_error_body(err), None, {"x-thrift-request-id": rid}, err.status)
        fallback = tier != chosen
        upstream_ms = (time.perf_counter() - t1) * 1000
        usage = resp.get("usage") or {}
        tin = int(usage.get("prompt_tokens", tok.count_messages(send.get("messages", []))))
        tout = int(usage.get("completion_tokens", 0))
        cost = self._cost(tier, tin, tout, usage)
        baseline = self._cost(strong, original_tokens, tout)
        self.ledger.record(Event(rid, time.time(), tier, self.cfg.tier(tier).model, "miss" if cacheable else "off", routed, score,
                                 original_tokens, tin, tout, cost, baseline, overhead, upstream_ms, 200, fallback,
                                 self._cost(strong, tin, tout), self._cost(tier, tin, tout)), strong)
        if cacheable and not fallback and _complete(resp):
            self.cache.put(scope, query, resp)
        if routed:
            self._after_routed(send, tier, _text(resp))
        hdr = self._headers(rid, tier, "miss" if cacheable else "off", original_tokens, tin, cost, baseline, score)
        if fallback:
            hdr["x-thrift-fallback"] = "1"
        if self.safe_mode:
            hdr["x-thrift-safe-mode"] = "1"
        return Result(resp, None, hdr)

    async def _stream(self, send, tier, routed, score, rid, original_tokens, overhead, scope, query, cacheable):
        t1 = time.perf_counter()
        parts, usage, finish, first = [], None, None, None
        chain = self._chain(tier)
        for i, tier in enumerate(chain):
            try:
                async for chunk in self.backends[tier].stream(send):
                    first = first or chunk
                    for ch in chunk.get("choices") or []:
                        parts.append((ch.get("delta") or {}).get("content") or "")
                        finish = ch.get("finish_reason") or finish
                    usage = chunk.get("usage") or usage
                    yield chunk
                break
            except UpstreamError as e:
                if first is None and i + 1 < len(chain):
                    continue  # nothing sent to the client yet: fall back silently
                self._record_error(rid, tier, routed, score, original_tokens, overhead, e.status)
                body = _error_body(e)
                yield body if "error" in body else {"error": body}
                return
        text = "".join(parts)
        tin = int((usage or {}).get("prompt_tokens", tok.count_messages(send.get("messages", []))))
        tout = int((usage or {}).get("completion_tokens", tok.count(text)))
        cost = self._cost(tier, tin, tout, usage)
        baseline = self._cost(self.cfg.strong.name, original_tokens, tout)
        self.ledger.record(Event(rid, time.time(), tier, self.cfg.tier(tier).model, "miss" if cacheable else "off", routed, score,
                                 original_tokens, tin, tout, cost, baseline, overhead, (time.perf_counter() - t1) * 1000, 200, False,
                                 self._cost(self.cfg.strong.name, tin, tout), self._cost(tier, tin, tout)), self.cfg.strong.name)
        if routed and finish in ("stop", "length"):
            self._after_routed(send, tier, text)
        if cacheable and finish == "stop" and first is not None and tier == chain[0]:
            self.cache.put(scope, query, {
                "id": first.get("id"), "object": "chat.completion", "created": first.get("created"), "model": first.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
                "usage": {"prompt_tokens": tin, "completion_tokens": tout, "total_tokens": tin + tout},
            })

    def _record_error(self, rid, tier, routed, score, original_tokens, overhead, status):
        self.ledger.record(Event(rid, time.time(), tier, self.cfg.tier(tier).model, "off", routed, score, original_tokens, 0, 0,
                                 0.0, 0.0, overhead, 0.0, status), self.cfg.strong.name)

    def _after_routed(self, send: dict, tier: str, answer: str) -> None:
        """Shadow sampling + drift guard. Each routed request is one observation for the risk
        monitor: 0 when the strongest model answered (no loss), and for a random fraction
        `shadow_rate` of the rest, the judged loss divided by that rate (see drift.py)."""
        r = self.cfg.shadow_rate
        if self.safe_mode:
            return
        if tier == self.cfg.strong.name:
            if self.monitor is not None:
                self.monitor.observe(0.0)
            return
        if r <= 0 or random.random() >= r:
            if self.monitor is not None:
                self.monitor.observe(0.0)
            return
        task = asyncio.get_running_loop().create_task(self._shadow(send, tier, answer, r))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _shadow(self, send: dict, tier: str, answer: str, rate: float) -> None:
        try:
            strong = _text(await self.backends[self.cfg.strong.name].chat(send))
        except UpstreamError:
            if self.monitor is not None:
                self.monitor.observe(0.0)  # keep the sampling unbiased: count it, as unjudged
            return
        loss = await self._judge(send, strong, answer) if self.monitor is not None else None
        if self.monitor is not None:
            if self.monitor.observe((loss or 0.0) / rate, judged=loss is not None):
                log.warning("thrift drift guard: routing risk above epsilon=%.3f (e-value %.1f); %s", self.monitor.epsilon,
                            self.monitor.e_value, "safe mode on" if self.cfg.drift.action == "safe_mode" else "alert only")
        row = {"messages": send.get("messages"), "tier": tier, "cheap": answer, "strong": strong, "loss": loss, "ts": time.time()}
        with open(self.cfg.shadow_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    async def _judge(self, send: dict, reference: str, candidate: str) -> float | None:
        if self.cfg.drift.judge == "agreement":
            return agreement_loss(reference, candidate)
        q = next((tok.content_text(m.get("content")) for m in reversed(send.get("messages") or []) if m.get("role") == "user"), "")
        prompt = JUDGE_PROMPT.format(q=q[-4000:], a=reference[:4000], b=candidate[:4000])
        judge = self.cfg.drift.judge_tier or self.cfg.strong.name
        try:
            r = await self.backends[judge].chat({"messages": [{"role": "user", "content": prompt}], "max_tokens": 5, "temperature": 0})
        except UpstreamError:
            return None
        return parse_judgement(_text(r))

    def reset_drift(self) -> None:
        if self.monitor is not None:
            self.monitor.reset()

    def stats(self) -> dict:
        s = self.ledger.stats()
        s["drift"] = self.monitor.state() if self.monitor is not None else None
        s["router"] = None if self.router is None else {
            "kind": "ladder" if isinstance(self.router, Ladder) else "two-tier", "tiers": self.router.tiers,
            "threshold": self.router.threshold, "certificate": self.router.certificate}
        s["unpriced_models"] = sorted(self.prices.unknown)
        return s

    def _headers(self, rid, tier, cache, original, sent, cost, baseline, score) -> dict:
        h = {"x-thrift-request-id": rid, "x-thrift-tier": tier, "x-thrift-model": self.cfg.tier(tier).model, "x-thrift-cache": cache,
             "x-thrift-prompt-tokens-original": str(original)}
        if sent is not None:
            h["x-thrift-prompt-tokens-sent"] = str(sent)
        if cost is not None:
            h["x-thrift-cost-usd"] = f"{cost:.8f}"
            h["x-thrift-baseline-usd"] = f"{baseline:.8f}"
        if score is not None:
            h["x-thrift-route-score"] = f"{score:.4f}"
        return h

    async def aclose(self) -> None:
        for t in list(self._tasks):
            t.cancel()
        for b in self.backends.values():
            await b.aclose()


def _text(resp: dict) -> str:
    try:
        return resp["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def _complete(resp: dict) -> bool:
    ch = resp.get("choices") or []
    return len(ch) == 1 and ch[0].get("finish_reason") in ("stop", "tool_calls") and ch[0].get("message") is not None


def _error_body(e: UpstreamError) -> dict:
    try:
        return json.loads(e.body)
    except (ValueError, TypeError):
        return {"error": {"message": e.body.decode(errors="replace"), "type": "upstream_error"}}


async def _replay(resp: dict) -> AsyncIterator[dict]:
    """Serve a cached response as an OpenAI-style stream."""
    base = {"id": resp.get("id"), "object": "chat.completion.chunk", "created": int(time.time()), "model": resp.get("model")}
    msg = resp["choices"][0]["message"]
    yield {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
    text = msg.get("content") or ""
    for i in range(0, len(text), 64):
        yield {**base, "choices": [{"index": 0, "delta": {"content": text[i : i + 64]}, "finish_reason": None}]}
    if msg.get("tool_calls"):
        yield {**base, "choices": [{"index": 0, "delta": {"tool_calls": msg["tool_calls"]}, "finish_reason": None}]}
    yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": resp["choices"][0].get("finish_reason", "stop")}]}
    yield {**base, "choices": [], "usage": resp.get("usage")}

