"""Thrift as a library: an OpenAI-SDK-shaped client that runs the whole pipeline in-process.

    from thrift_llm import Thrift

    client = Thrift("openai:gpt-5-mini", "anthropic:claude-sonnet-5-5")   # cheapest first
    r = client.chat.completions.create(model="thrift-auto", messages=[{"role": "user", "content": "hi"}])
    print(r.choices[0].message.content, r.thrift["x-thrift-tier"])

No server, no extra hop. Use `AsyncThrift` inside an event loop. Pass a TOML path or a Config
instead of tier specs for full control (cache, compression, drift guard, router).
"""
from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator

from . import config as cfgmod
from .gateway import Gateway


class Obj(dict):
    """dict with attribute access, so responses read like the OpenAI SDK's (r.choices[0].message.content)."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            raise AttributeError(k) from None
        return _wrap(v)

    def __getitem__(self, k):
        return _wrap(dict.__getitem__(self, k))


def _wrap(v):
    if isinstance(v, dict) and not isinstance(v, Obj):
        return Obj(v)
    if isinstance(v, list):
        return [_wrap(x) for x in v]
    return v


class ThriftError(Exception):
    def __init__(self, status: int, body: dict):
        self.status = status
        self.body = body
        msg = (body.get("error") or {}).get("message") if isinstance(body, dict) else None
        super().__init__(f"{status}: {msg or body}")


def _config(specs, config) -> cfgmod.Config:
    if config is not None:
        return cfgmod.load(config) if isinstance(config, str) else config
    if not specs:
        raise ValueError('give tier specs like Thrift("openai:gpt-5-mini", "anthropic:claude-sonnet-5-5") or config=...')
    return cfgmod.from_specs(list(specs))


class _AsyncCompletions:
    def __init__(self, gw: Gateway):
        self._gw = gw

    async def create(self, *, extra_headers: dict | None = None, **body):
        r = await self._gw.chat(body, extra_headers or {})
        if r.status != 200:
            raise ThriftError(r.status, r.response)
        if r.stream is not None:
            return _astream(r.stream, r.headers)
        out = Obj(r.response)
        out["thrift"] = r.headers
        return out


async def _astream(it: AsyncIterator[dict], headers: dict):
    async for c in it:
        if "error" in c and "choices" not in c:
            raise ThriftError(502, c)
        o = Obj(c)
        o["thrift"] = headers
        yield o


class _Ns:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class AsyncThrift:
    def __init__(self, *specs: str, config=None):
        self.config = _config(specs, config)
        self.gateway = Gateway(self.config)
        self.chat = _Ns(completions=_AsyncCompletions(self.gateway))

    def stats(self) -> dict:
        return self.gateway.stats()

    async def aclose(self) -> None:
        await self.gateway.aclose()


class Thrift:
    """Synchronous facade: the gateway runs on a private event loop in a daemon thread."""

    def __init__(self, *specs: str, config=None):
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True, name="thrift-loop").start()
        self._inner = self._run(_make_async(specs, config))
        self.config = self._inner.config
        self.gateway = self._inner.gateway
        self.chat = _Ns(completions=_Ns(create=self._create))

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _create(self, **kw):
        r = self._run(self._inner.chat.completions.create(**kw))
        if isinstance(r, Obj):
            return r
        return _sync_iter(r, self._loop)

    def stats(self) -> dict:
        return self.gateway.stats()

    def close(self) -> None:
        self._run(self._inner.aclose())
        self._loop.call_soon_threadsafe(self._loop.stop)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


async def _make_async(specs, config) -> AsyncThrift:
    return AsyncThrift(*specs, config=config)


def _sync_iter(agen, loop) -> Iterator[Obj]:
    while True:
        try:
            yield asyncio.run_coroutine_threadsafe(agen.__anext__(), loop).result()
        except StopAsyncIteration:
            return
