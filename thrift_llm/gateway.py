"""The request pipeline, independent of any web framework.

    request -> [cache lookup] -> [compress] -> [route] -> upstream -> [cache store] -> ledger

Use it in-process (`await Gateway(cfg).chat(body)`) or behind the HTTP server (server.py).
"""
from __future__ import annotations

import asyncio
import copy
import json
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
from .ledger import Event, Ledger
from .pricing import PriceBook
from .router import Router

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


def bundled_router() -> Router | None:
    try:
        p = resources.files("thrift_llm").joinpath("data/router_helm.json")
        with resources.as_file(p) as f:
            return Router.load(str(f))
    except (FileNotFoundError, ValueError, ModuleNotFoundError):
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
            self.router = Router.load(cfg.router.path) if cfg.router.path else bundled_router()
            if self.router is not None and cfg.router.threshold is not None:
                self.router.threshold = cfg.router.threshold

    # ---- helpers ------------------------------------------------------------------------------
    def _pick_tier(self, body: dict, opts: Options, text: str) -> tuple[str, bool, float | None]:
        model = body.get("model") or "thrift-auto"
        if model not in self.cfg.router.auto_models:
            t = self.cfg.tier(model)
            return (t.name if t else self.cfg.strong.name), False, None
        if not opts.route or self.router is None:
            return self.cfg.strong.name, False, None
        score = self.router.score_one(text)
        name = self.cfg.cheap.name if score >= self.router.threshold else self.cfg.strong.name
        return name, True, score

    @staticmethod
    def _scope(body: dict, tier: str) -> tuple[dict, str]:
        msgs = body.get("messages", [])
        last = next((i for i in range(len(msgs) - 1, -1, -1) if msgs[i].get("role") == "user"), None)
        query = tok.content_text(msgs[last].get("content")) if last is not None else ""
        prefix = msgs[:last] if last is not None else msgs
        scope = {"tier": tier, "prefix": prefix, "after": msgs[last + 1 :] if last is not None else [], **{k: body.get(k) for k in SCOPE_PARAMS if k in body}}
        return scope, query

    def _cacheable(self, body: dict, opts: Options) -> bool:
        if self.cache is None or not opts.cache:
            return False
        if (body.get("n") or 1) > 1:
            return False
        return float(body.get("temperature", 1.0) or 0.0) <= self.cfg.cache.max_temperature

    def _cost(self, tier: str, tin: int, tout: int) -> float:
        return self.prices.cost(self.cfg.tier(tier).model, tin, tout)

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
        cacheable = self._cacheable(body, opts)

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
                          0.0, baseline, overhead, 0.0), self.cfg.cheap.name)
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
        fallback = False
        try:
            resp = await self.backends[tier].chat(send)
        except UpstreamError as e:
            if tier == strong:
                self._record_error(rid, tier, routed, score, original_tokens, overhead, e.status)
                return Result(_error_body(e), None, {"x-thrift-request-id": rid}, e.status)
            fallback, tier = True, strong
            try:
                resp = await self.backends[tier].chat(send)
            except UpstreamError as e2:
                self._record_error(rid, tier, routed, score, original_tokens, overhead, e2.status)
                return Result(_error_body(e2), None, {"x-thrift-request-id": rid}, e2.status)
        upstream_ms = (time.perf_counter() - t1) * 1000
        usage = resp.get("usage") or {}
        tin = int(usage.get("prompt_tokens", tok.count_messages(send.get("messages", []))))
        tout = int(usage.get("completion_tokens", 0))
        cost = self._cost(tier, tin, tout)
        baseline = self._cost(strong, original_tokens, tout)
        self.ledger.record(Event(rid, time.time(), tier, self.cfg.tier(tier).model, "miss" if cacheable else "off", routed, score,
                                 original_tokens, tin, tout, cost, baseline, overhead, upstream_ms, 200, fallback,
                                 self._cost(strong, tin, tout)), self.cfg.cheap.name)
        if cacheable and not fallback and _complete(resp):
            self.cache.put(scope, query, resp)
        self._maybe_shadow(send, tier, resp)
        hdr = self._headers(rid, tier, "miss" if cacheable else "off", original_tokens, tin, cost, baseline, score)
        if fallback:
            hdr["x-thrift-fallback"] = "1"
        return Result(resp, None, hdr)

    async def _stream(self, send, tier, routed, score, rid, original_tokens, overhead, scope, query, cacheable):
        t1 = time.perf_counter()
        parts, usage, finish, first = [], None, None, None
        try:
            async for chunk in self.backends[tier].stream(send):
                first = first or chunk
                for ch in chunk.get("choices") or []:
                    parts.append((ch.get("delta") or {}).get("content") or "")
                    finish = ch.get("finish_reason") or finish
                usage = chunk.get("usage") or usage
                yield chunk
        except UpstreamError as e:
            self._record_error(rid, tier, routed, score, original_tokens, overhead, e.status)
            body = _error_body(e)
            yield body if "error" in body else {"error": body}
            return
        text = "".join(parts)
        tin = int((usage or {}).get("prompt_tokens", tok.count_messages(send.get("messages", []))))
        tout = int((usage or {}).get("completion_tokens", tok.count(text)))
        cost = self._cost(tier, tin, tout)
        baseline = self._cost(self.cfg.strong.name, original_tokens, tout)
        self.ledger.record(Event(rid, time.time(), tier, self.cfg.tier(tier).model, "miss" if cacheable else "off", routed, score,
                                 original_tokens, tin, tout, cost, baseline, overhead, (time.perf_counter() - t1) * 1000, 200, False,
                                 self._cost(self.cfg.strong.name, tin, tout)), self.cfg.cheap.name)
        if cacheable and finish == "stop" and first is not None:
            self.cache.put(scope, query, {
                "id": first.get("id"), "object": "chat.completion", "created": first.get("created"), "model": first.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
                "usage": {"prompt_tokens": tin, "completion_tokens": tout, "total_tokens": tin + tout},
            })

    def _record_error(self, rid, tier, routed, score, original_tokens, overhead, status):
        self.ledger.record(Event(rid, time.time(), tier, self.cfg.tier(tier).model, "off", routed, score, original_tokens, 0, 0,
                                 0.0, 0.0, overhead, 0.0, status), self.cfg.cheap.name)

    def _maybe_shadow(self, send: dict, tier: str, resp: dict) -> None:
        """Send a sample of cheap-routed requests to the strong tier too and log both answers,
        so the router can be re-calibrated on your own traffic (`thrift-llm calibrate`)."""
        if tier == self.cfg.strong.name or self.cfg.shadow_rate <= 0 or random.random() >= self.cfg.shadow_rate:
            return

        async def run():
            try:
                strong = await self.backends[self.cfg.strong.name].chat(send)
            except UpstreamError:
                return
            row = {"messages": send.get("messages"), "cheap": _text(resp), "strong": _text(strong), "ts": time.time()}
            with open(self.cfg.shadow_path, "a") as f:
                f.write(json.dumps(row) + "\n")

        asyncio.get_running_loop().create_task(run())

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

