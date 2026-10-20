"""Upstream backends.

OpenAIBackend serves every provider that speaks the OpenAI chat-completions API: OpenAI, xAI
(Grok), Perplexity, Google Gemini, DeepSeek, Mistral, Groq, Together, Fireworks, OpenRouter, and
self-hosted vLLM / SGLang / TGI / llama.cpp / Ollama. AnthropicBackend talks to Claude's native
Messages API (see anthropic.py) so prompt caching works.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
import urllib.parse
from collections.abc import AsyncIterator

import httpx

from . import anthropic as ant
from . import tokens as tok
from .config import Tier

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}  # 529: Anthropic overloaded


class UpstreamError(Exception):
    def __init__(self, status: int, body: bytes | str):
        self.status = status
        self.body = body if isinstance(body, bytes) else body.encode()
        super().__init__(f"upstream {status}: {self.body[:200]!r}")


class OpenAIBackend:
    def __init__(self, tier: Tier, client: httpx.AsyncClient | None = None, max_retries: int = 2):
        self.tier = tier
        self.max_retries = max_retries
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(tier.timeout_s, connect=10.0))

    def _headers(self) -> dict:
        h = {"content-type": "application/json", **self.tier.headers}
        key = self.tier.key()
        if key:
            h["authorization"] = f"Bearer {key}"
        return h

    def _url(self, path: str) -> str:
        return self.tier.base_url.rstrip("/") + path

    async def _post(self, path: str, body: dict) -> dict:
        for attempt in range(self.max_retries + 1):
            try:
                r = await self.client.post(self._url(path), headers=self._headers(), content=json.dumps(body))
            except httpx.TransportError as e:
                if attempt == self.max_retries:
                    raise UpstreamError(502, f"transport error: {e}") from e
            else:
                if r.status_code == 200:
                    return r.json()
                if r.status_code not in RETRY_STATUS or attempt == self.max_retries:
                    raise UpstreamError(r.status_code, r.content)
            await asyncio.sleep(min(8.0, 0.5 * 2**attempt) * (0.5 + random.random()))
        raise AssertionError("unreachable")

    async def _events(self, path: str, body: dict) -> AsyncIterator[dict]:
        try:
            async with self.client.stream("POST", self._url(path), headers=self._headers(), content=json.dumps(body)) as r:
                if r.status_code != 200:
                    raise UpstreamError(r.status_code, await r.aread())
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    yield json.loads(data)
        except httpx.TransportError as e:
            raise UpstreamError(502, f"transport error: {e}") from e

    async def chat(self, body: dict) -> dict:
        return await self._post("/chat/completions", {**body, "model": self.tier.model, "stream": False})

    async def stream(self, body: dict) -> AsyncIterator[dict]:
        body = {**body, "model": self.tier.model, "stream": True}
        body.setdefault("stream_options", {"include_usage": True})
        async for ev in self._events("/chat/completions", body):
            yield ev

    async def passthrough(self, method: str, path: str, body: bytes, headers: dict) -> httpx.Response:
        h = {k: v for k, v in headers.items() if k.lower() in ("content-type", "accept")}
        h.update({k: v for k, v in self._headers().items() if k != "content-type"})
        return await self.client.request(method, self._url(path), content=body, headers=h)

    async def aclose(self) -> None:
        await self.client.aclose()


class AnthropicBackend(OpenAIBackend):
    """Claude through the native Messages API; requests and answers stay in OpenAI format."""

    def _headers(self) -> dict:
        h = {"content-type": "application/json", "anthropic-version": "2023-06-01", **self.tier.headers}
        key = self.tier.key()
        if key:
            h["x-api-key"] = key
        return h

    async def chat(self, body: dict) -> dict:
        req = ant.to_anthropic(body, self.tier.model, self.tier.prompt_cache)
        return ant.from_anthropic(await self._post("/messages", req))

    async def stream(self, body: dict) -> AsyncIterator[dict]:
        req = {**ant.to_anthropic(body, self.tier.model, self.tier.prompt_cache), "stream": True}
        tr = ant.StreamTranslator()
        async for ev in self._events("/messages", req):
            if ev.get("type") == "error":
                raise UpstreamError(529 if (ev.get("error") or {}).get("type") == "overloaded_error" else 502, json.dumps(ev))
            for chunk in tr.feed(ev):
                yield chunk


class MockBackend:
    """Deterministic offline model for tests, demos and overhead benchmarks.

    base_url: mock://<name>?latency_ms=50&tokens_per_s=0 . Answers echo the question so that
    cache hits are visibly correct, and report usage counted with the same tokenizer.
    """

    def __init__(self, tier: Tier):
        self.tier = tier
        q = urllib.parse.parse_qs(urllib.parse.urlparse(tier.base_url).query)
        self.latency_s = float(q.get("latency_ms", ["0"])[0]) / 1000
        self.fail = float(q.get("fail", ["0"])[0])
        self.calls = 0

    def _answer(self, body: dict) -> str:
        last = next((m for m in reversed(body.get("messages", [])) if m.get("role") == "user"), {})
        q = tok.content_text(last.get("content"))
        q = q.strip().splitlines()[-1] if q.strip() else ""
        return f"[{self.tier.model}] Answer to: {q[:200]}"

    def _usage(self, body: dict, text: str) -> dict:
        p = tok.count_messages(body.get("messages", []))
        c = tok.count(text)
        return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}

    async def chat(self, body: dict) -> dict:
        self.calls += 1
        if self.fail and random.random() < self.fail:
            raise UpstreamError(503, b'{"error":{"message":"mock overloaded"}}')
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        text = self._answer(body)
        return {
            "id": f"chatcmpl-mock-{self.calls}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.tier.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": self._usage(body, text),
        }

    async def stream(self, body: dict) -> AsyncIterator[dict]:
        r = await self.chat(body)
        text = r["choices"][0]["message"]["content"]
        base = {"id": r["id"], "object": "chat.completion.chunk", "created": r["created"], "model": r["model"]}
        yield {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
        for i in range(0, len(text), 16):
            yield {**base, "choices": [{"index": 0, "delta": {"content": text[i : i + 16]}, "finish_reason": None}]}
        yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield {**base, "choices": [], "usage": r["usage"]}

    async def passthrough(self, method, path, body, headers):
        return httpx.Response(404, json={"error": {"message": f"mock backend has no {path}"}})

    async def aclose(self) -> None:
        pass


def make_backend(tier: Tier):
    style = tier.api_style
    if style == "mock":
        return MockBackend(tier)
    return AnthropicBackend(tier) if style == "anthropic" else OpenAIBackend(tier)
