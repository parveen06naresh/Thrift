"""The in-process OpenAI-shaped client and tier shorthand."""
import asyncio

import pytest

from thrift_llm import AsyncThrift, Thrift, config


def test_specs():
    assert config.parse_spec("grok:grok-4") == {"name": "grok-4", "model": "grok-4", "provider": "grok"}
    assert config.parse_spec("anthropic:claude-sonnet-5-5=smart")["name"] == "smart"
    assert config.parse_spec("fast=gemini:gemini-2.5-flash-lite")["name"] == "fast"
    with pytest.raises(ValueError):
        config.parse_spec("gpt-5")
    cfg = config.from_specs(["openai:gpt-5-mini", "claude:claude-opus-5-5"])
    assert cfg.cheap.provider == "openai" and cfg.strong.provider == "anthropic"


def test_sync_client_answers_streams_and_counts():
    with Thrift("mock:gpt-5-mini=cheap", "mock:claude-sonnet-5-5=strong") as c:
        r = c.chat.completions.create(model="strong", messages=[{"role": "user", "content": "hello there"}])
        assert r.choices[0].message.content.endswith("hello there") and r.thrift["x-thrift-tier"] == "strong"
        again = c.chat.completions.create(model="strong", messages=[{"role": "user", "content": "hello there"}])
        assert again.thrift["x-thrift-cache"] == "exact"
        text = "".join(ch.choices[0].delta.get("content") or "" for ch in
                       c.chat.completions.create(model="strong", stream=True, messages=[{"role": "user", "content": "stream me"}]) if ch.choices)
        assert text.endswith("stream me")
        assert c.stats()["totals"]["requests"] == 3


def test_async_client():
    async def run():
        c = AsyncThrift("mock:gpt-5-mini=cheap", "mock:gpt-5.1=strong")
        r = await c.chat.completions.create(model="thrift-auto", messages=[{"role": "user", "content": "2+2?"}])
        await c.aclose()
        return r

    assert asyncio.run(run()).thrift["x-thrift-tier"] in ("cheap", "strong")
